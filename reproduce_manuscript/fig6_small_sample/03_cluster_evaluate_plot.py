from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_ROOT = "/hpc/group/xielab/xj58/xVerse_results/fig6/sampled_data"

GROUPS = [
    "ILC_NK_group",
    "Plasmablast_boundary_group",
    "HSPC_boundary_group",
    "Eryth_platelet_group",
]

SAMPLE_FILES = {
    g: f"{DATA_ROOT}/3P_{g}.h5ad" for g in GROUPS
}

PRETRAIN_INFER_ROOT = Path("/hpc/group/xielab/xj58/xVerse_results/fig6/pretrain_inference")
PLOT_ROOT = Path("/hpc/group/xielab/xj58/xVerse_results/fig6/plots")

# Resolution for Poisson clustering
RESOLUTION = 0.5
DOT = 1000

LABEL_KEY = "celltype.l2"
MAJORITY_KEY = "poisson_majority_leiden"
POISSON_CLUSTER_KEY = "poisson_leiden"

# Poisson replicates control
MAX_REPLICATES: Optional[int] = 5

def preprocess_for_neighbors(adata: sc.AnnData) -> None:
    """
    Unified Preprocessing Pipeline.
    Method: Normalize -> Log1p -> HVG(Seurat/Dispersion) -> Scale -> PCA -> Neighbors -> UMAP
    
    Args:
        n_neighbors: Number of neighbors for graph construction (default: 15)
    """
    # 1. Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # 2. Log Transformation (Required for 'seurat' flavor HVG)
    sc.pp.log1p(adata)
    
    # 3. Feature Selection (HVG)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        batch_key=None
    )
    
    # 4. Scaling
    # Automatically uses highly_variable genes if present
    sc.pp.scale(adata)
    
    # 5. PCA
    sc.tl.pca(adata)
    
    # 6. Neighbors
    sc.pp.neighbors(adata)
    
    # 8. UMAP
    sc.tl.umap(adata)


def parse_cell_indices(obs_names: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (cell_idx, replicate_idx) from 'idx_copy' strings."""
    cell_idx = []
    rep_idx = []
    for name in obs_names:
        parts = str(name).split("_")
        if len(parts) < 2:
            raise ValueError(f"Obs name '{name}' malformed.")
        cell_idx.append(int(parts[0]) - 1)
        rep_idx.append(int(parts[1]))
    return np.asarray(cell_idx, dtype=int), np.asarray(rep_idx, dtype=int)


def cluster_poisson_replicates(poisson_path: Path, resolution: float) -> Tuple[sc.AnnData, np.ndarray]:
    """Load xVerse Poisson replicates -> Preprocess -> Leiden Cluster."""
    if not poisson_path.exists():
        raise FileNotFoundError(f"Not found: {poisson_path}")
    
    adata = sc.read_h5ad(poisson_path)
    print(f"  [xVerse] Loading replicates: {adata.shape}")
    
    # Filter replicates if MAX_REPLICATES is set
    if MAX_REPLICATES is not None:
        # Parse cell and replicate indices
        cell_idx_all, rep_idx_all = parse_cell_indices(adata.obs_names)
        
        # Keep only replicates 1 to MAX_REPLICATES
        keep_mask = rep_idx_all <= MAX_REPLICATES
        n_before = adata.n_obs
        adata = adata[keep_mask].copy()
        n_after = adata.n_obs
        
        print(f"  [Filter] Using first {MAX_REPLICATES} replicates: {n_before} -> {n_after} cells")
    
    # Use the UNIFIED preprocessing function with higher n_neighbors for Poisson replicates
    preprocess_for_neighbors(adata)

    print(f"  [xVerse] Clustering replicates (res={resolution})...")
    sc.tl.leiden(adata, resolution=resolution, key_added=POISSON_CLUSTER_KEY)

    cell_idx, _ = parse_cell_indices(adata.obs_names)
    return adata, cell_idx


def majority_vote(poisson_labels: pd.Series, cell_idx: np.ndarray, n_cells: int) -> pd.Series:
    """Consolidate replicate labels to single-cell labels."""
    df = pd.DataFrame({"cell_idx": cell_idx, "label": poisson_labels.to_numpy()})
    voted = (
        df.groupby("cell_idx")["label"]
        .agg(lambda x: x.value_counts().idxmax())
    )
    voted = voted.reindex(range(n_cells))
    return voted


# ==============================================================================
# PLOTTING & BENCHMARKING
# ==============================================================================


def run_benchmark_clustering(adata: sc.AnnData, resolutions: List[float] = [0.1, 0.5, 1.0]) -> List[str]:
    """
    Run Standard Clustering Benchmark on the provided AnnData.
    Adds 'leiden_res_{r}' columns to adata.obs.
    Returns the list of added keys.
    """
    print(f"  [Benchmark] Running Standard Clustering (Res: {resolutions})...")

    keys = []
    for res in resolutions:
        key = f"leiden_res_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key)
        keys.append(key)
        
    return keys


def calculate_metrics(adata: sc.AnnData, true_key: str, pred_keys: List[str]) -> dict:
    """Calculate and print ARI and NMI for predicted keys against true key."""
    print("\n  [Metrics] Clustering Performance:")
    print(f"  {'Method':<25} | {'ARI':<8} | {'NMI':<8}")
    print("  " + "-"*47)
    
    true_labels = adata.obs[true_key].values
    scores = {}
    
    for key in pred_keys:
        if key not in adata.obs:
            continue
            
        pred_labels = adata.obs[key].values
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        print(f"  {key:<25} | {ari:.4f}   | {nmi:.4f}")
        
        scores[key] = {"ARI": ari, "NMI": nmi}
        
    print("  " + "-"*47 + "\n")
    return scores


def align_cluster_colors(adata: sc.AnnData, true_key: str, pred_key: str):
    """
    Assign colors to pred_key categories.
    1. Identify the best matching predicted cluster for each true category.
    2. Assign the true category's color to that predicted cluster.
    3. Assign distinct colors to remaining predicted clusters (avoiding duplicates).
    """
    # Ensure categorical
    if not isinstance(adata.obs[true_key].dtype, pd.CategoricalDtype):
        adata.obs[true_key] = adata.obs[true_key].astype("category")
    if not isinstance(adata.obs[pred_key].dtype, pd.CategoricalDtype):
        adata.obs[pred_key] = adata.obs[pred_key].astype("category")

    # Ensure true_key has colors
    if f"{true_key}_colors" not in adata.uns:
        sc.pl.umap(adata, color=true_key, show=False)
        plt.close()

    true_cats = adata.obs[true_key].cat.categories
    true_colors = adata.uns[f"{true_key}_colors"]
    true_color_map = dict(zip(true_cats, true_colors))

    pred_cats = adata.obs[pred_key].cat.categories
    
    # Compute confusion matrix (rows=Pred, cols=True)
    confusion = pd.crosstab(adata.obs[pred_key], adata.obs[true_key])
    
    # Find best match for each True category
    matches = []
    for t_cat in true_cats:
        if t_cat in confusion.columns:
            # Find pred_cat with max overlap for this t_cat
            p_cat = confusion[t_cat].idxmax()
            count = confusion.loc[p_cat, t_cat]
            matches.append((count, t_cat, p_cat))
            
    # Sort by count descending
    matches.sort(key=lambda x: x[0], reverse=True)
    
    pred_color_map = {}
    used_colors = set()
    
    # Pre-populate used_colors with ALL true colors to avoid collisions
    # Normalize to lowercase to ensure robust matching
    for c in true_color_map.values():
        c_hex = mcolors.to_hex(c, keep_alpha=False).lower()
        used_colors.add(c_hex)
    
    # Assign colors to best matches
    for _, t_cat, p_cat in matches:
        if p_cat not in pred_color_map:
            c = true_color_map[t_cat]
            c_hex = mcolors.to_hex(c, keep_alpha=False).lower()
            pred_color_map[p_cat] = c_hex
            # used_colors.add(c_hex) # Already added above
            
    # Assign distinct colors to remaining clusters
    remaining_preds = [p for p in pred_cats if p not in pred_color_map]
    
    if remaining_preds:
        # Generate a large pool of candidate colors
        candidates = []
        # Combine multiple palettes to get enough distinct colors
        # Use Dark2/Set2 first for high contrast, then tab20/Paired
        for name in ['Dark2', 'Set2', 'tab20', 'Paired']:
            cmap = plt.get_cmap(name)
            for i in range(cmap.N):
                candidates.append(mcolors.to_hex(cmap(i), keep_alpha=False).lower())
        
        idx = 0
        for p_cat in remaining_preds:
            # Find next available color that hasn't been used
            while idx < len(candidates) and candidates[idx] in used_colors:
                idx += 1
            
            if idx < len(candidates):
                c = candidates[idx]
                pred_color_map[p_cat] = c
                used_colors.add(c)
                idx += 1
            else:
                # Fallback if we somehow run out of colors
                pred_color_map[p_cat] = "#808080" # Grey

    # Construct final list in order of categories
    final_colors = [pred_color_map[cat] for cat in pred_cats]
    adata.uns[f"{pred_key}_colors"] = final_colors


def plot_combined_grid(collected_data: List[Tuple[str, sc.AnnData]], all_results: dict, label_key: str, umap_keys: List[str], output_path: Path):
    """
    Generate a single 4x6 grid figure.
    Rows: Samples (Groups)
    Cols: Pie Chart, Ground Truth, xVerse, Leiden 0.5, Leiden 1.0, Leiden 2.0
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_rows = len(collected_data)
    # Pie + Truth + xVerse + 3 Benchmarks = 6 columns
    # We assume umap_keys has 4 items: [MAJORITY_KEY, leiden_0.5, leiden_1.0, leiden_2.0]
    # Total cols = 1 (Pie) + 1 (Truth) + 4 (Predictions) = 6
    all_umap_keys = [label_key] + umap_keys
    n_cols = 1 + len(all_umap_keys) 
    
    # Figure size: 6 cols * 5 inch, 4 rows * 5 inch
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    
    # Global font size settings are handled in main(), but we can enforce here too if needed.
    # User requested font size 20.
    
    for row_idx, (sample_name, adata) in enumerate(collected_data):
        # --- Column 0: Pie Chart ---
        ax_pie = axes[row_idx, 0]
        counts = adata.obs[label_key].value_counts()
        
        # Custom labels: "Type\nN (X%)" for inside the pie
        total = sum(counts)
        labels = [f"{idx}\n{val} ({val/total*100:.0f}%)" for idx, val in zip(counts.index, counts.values)]
        
        # Use Set3 colors
        cmap = plt.get_cmap("Set3")
        colors = cmap(np.linspace(0, 1, len(counts)))
        
        ax_pie.pie(
            counts, 
            labels=labels, 
            colors=colors, 
            autopct=None, 
            startangle=140,
            labeldistance=0.5, # Put labels inside
            textprops={'fontsize': 20, 'weight': 'bold', 'ha': 'center'}
        )
        
        # Ensure colors exist for label_key before alignment
        if f"{label_key}_colors" not in adata.uns:
             sc.pl.umap(adata, color=label_key, show=False)
             plt.close()

        # --- Columns 1-5: UMAPs ---
        for col_idx, key in enumerate(all_umap_keys):
            ax = axes[row_idx, col_idx + 1]

            if key == MAJORITY_KEY and key != label_key:
                align_cluster_colors(adata, label_key, key)

            # Only show legend for Ground Truth (first UMAP)
            loc = 'upper right' if col_idx == 0 else None
            
            sc.pl.umap(
                adata,
                color=key,
                show=False,
                ax=ax,
                title="", 
                frameon=False,
                size=DOT,
                legend_loc=loc,
                legend_fontsize=25,
            )
            
            if col_idx > 0:
                # Retrieve metrics for this method
                # key is the method name (e.g., MAJORITY_KEY or leiden_res_...)
                scores = all_results.get(sample_name, {}).get(key, {})
                ari = scores.get("ARI", 0.0)
                nmi = scores.get("NMI", 0.0)
                
                # ax.set_xlabel(f"ARI={ari:.2f} NMI={nmi:.2f}", fontsize=25, fontweight='bold')
                ax.text(0.5, -0.05, f"ARI={ari:.2f}\nNMI={nmi:.2f}", 
                        transform=ax.transAxes, 
                        fontsize=25, fontweight='bold', 
                        ha='center', va='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved Combined Grid -> {output_path.name}")


def plot_metrics_summary(all_results: dict, output_path: Path):
    """
    Plot grouped bar charts for ARI and NMI across all samples.
    all_results structure: { 'SampleName': { 'Method1': {'ARI': x, 'NMI': y}, ... }, ... }
    """
    if not all_results:
        print("  [Warning] No results to plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    samples = list(all_results.keys())
    methods = list(next(iter(all_results.values())).keys())
    metrics = ["ARI", "NMI"]
    
    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar configuration
    x = np.arange(len(samples))
    width = 0.8 / len(methods)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, method in enumerate(methods):
            # Extract scores for this method across all samples
            scores = [all_results[s].get(method, {}).get(metric, 0) for s in samples]
            
            # Plot bar
            offset = (j - len(methods)/2) * width + width/2
            
            cmap = plt.get_cmap("Set2")
            color = cmap(j % 8)
            
            ax.bar(x + offset, scores, width, label=method, color=color)
            
        ax.set_title(f"{metric} Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(samples, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        if i == 0: # Legend only on first plot to save space
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved Metrics Summary -> {output_path.name}")


def plot_poisson_replicates(adata: sc.AnnData, cluster_key: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sc.pl.umap(
        adata,
        color=[cluster_key],
        title="Poisson Replicates UMAP",
        show=False,
        frameon=False,
        size=DOT,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved Replicates UMAP -> {output_path.name}")


# ==============================================================================
# MAIN LOOP
# ==============================================================================

def main():
    # Increase font size globally to 20
    plt.rcParams.update({'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 20})

    all_results = {}
    collected_data = [] # List to store (sample_name, adata) for final grid plot

    for label, annotated_path in SAMPLE_FILES.items():
        annotated_path = Path(annotated_path)
        sample_name = annotated_path.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"File: {sample_name}")
        print(f"{'='*60}")

        if not annotated_path.exists():
            print(f"Error: Annotated file not found at {annotated_path}")
            continue

        annotated = sc.read_h5ad(annotated_path)
        if LABEL_KEY not in annotated.obs:
            raise KeyError(f"{LABEL_KEY} missing in annotated AnnData.")
            
        # 1. Unified Preprocessing (PCA, Neighbors, UMAP) ONCE for everything
        print("  [Preprocess] Running unified preprocessing...")
        preprocess_for_neighbors(annotated)

        # 2. Run Benchmarks (Leiden 0.5, 1.0, 2.0)
        # This adds columns to annotated.obs
        bench_keys = run_benchmark_clustering(annotated, resolutions=[0.5, 1.0, 2.0])

        # ------------------------------------------------------------------
        # 3. Poisson-only clustering workflow
        # ------------------------------------------------------------------
        infer_dir = PRETRAIN_INFER_ROOT / sample_name
        poisson_path = infer_dir / f"{sample_name}_estimated_expr_poisson.h5ad"
        
        if not poisson_path.exists():
            print(f"Warning: Inference output not found at {poisson_path}")
            continue

        # A. Cluster Replicates
        poisson, cell_idx = cluster_poisson_replicates(poisson_path, RESOLUTION)
        
        if cell_idx.max() >= annotated.n_obs:
            print(f"  [Error] Poisson contains cell_idx {cell_idx.max()} >= annotated n_obs {annotated.n_obs}!")
        
        # B. Majority Vote back to annotated cells
        voted = majority_vote(poisson.obs[POISSON_CLUSTER_KEY], cell_idx, annotated.n_obs)
        voted = voted.fillna("Unassigned").astype(str)
        annotated.obs[MAJORITY_KEY] = voted.values

        # C. Save labels CSV
        labels_csv = infer_dir / f"{sample_name}_poisson_majority_labels.csv"
        labels_csv.parent.mkdir(parents=True, exist_ok=True)
        annotated.obs[[LABEL_KEY, MAJORITY_KEY] + bench_keys].to_csv(labels_csv)
        print(f"  [IO] Saved labels -> {labels_csv.name}")

        # F. Calculate Metrics (Moved before plotting)
        sample_scores = calculate_metrics(annotated, LABEL_KEY, [MAJORITY_KEY] + bench_keys)
        all_results[sample_name] = sample_scores

        # D. Collect for Grid Plot (instead of plotting row immediately)
        # Keys to plot: Majority + Benchmarks (Truth is handled inside function)
        # We need to ensure adata has all these keys.
        collected_data.append((sample_name, annotated))

        # G. Save Updated Object with all labels
        updated_path = annotated_path.parent / f"{sample_name}_with_results.h5ad"
        annotated.write_h5ad(updated_path)

    # H. Plot Summary Metrics
    summary_path = PLOT_ROOT / "summary_metrics_comparison.png"
    plot_metrics_summary(all_results, summary_path)

    # I. Plot Combined Grid
    if collected_data:
        # Safer: re-define
        bench_keys = [f"leiden_res_{r}" for r in [0.5, 1.0, 2.0]]
        umap_plot_keys = [MAJORITY_KEY] + bench_keys
        
        grid_path = PLOT_ROOT / "combined_grid_plot.png"
        plot_combined_grid(collected_data, all_results, LABEL_KEY, umap_plot_keys, grid_path)


if __name__ == "__main__":
    main()
