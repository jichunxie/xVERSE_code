#!/usr/bin/env python3
"""
Evaluate embedding quality for all models (xVerse, Harmony, scGPT, Nicheformer, Geneformer).
Computes ASW, NMI, ARI metrics for cell type clustering.
Creates 1x3 visualization for each tissue showing the three metrics.
#SBATCH --gres=gpu:h200:3
#SBATCH -c 20                                  
#SBATCH --mem=200G      
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import scanpy.external as sce
import scib

# ================= CONFIGURATION =================
# Use local paths for testing if /hpc is not available
if os.path.exists("/hpc/group/xielab/xj58"):
    liver_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/liver"
    brain_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/brain"
    output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/evaluation"
else:
    # Local fallback for testing
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    liver_dir = os.path.join(base_dir, "data", "liver")
    brain_dir = os.path.join(base_dir, "data", "brain")
    output_dir = os.path.join(base_dir, "results", "evaluation")
    print(f"Using local paths: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

# Models to evaluate (Order matters for visualization)
# Display names (Capitalized# Models to evaluate
MODELS = [
    "xVerse",
    "Harmony",
    "scGPT", 
    "Nicheformer",
    "Geneformer"
]

# Map display names to adata.obsm keys
MODEL_KEYS = {
    "xVerse": "xVerse",
    "Harmony": "harmony",
    "scGPT": "scgpt",
    "Nicheformer": "nicheformer",
    "Geneformer": "geneformer"
}
# ===============================================


def load_and_merge_tissue(tissue_dir, tissue_name, output_dir, gene_set="all"):
    """
    Load all donor files for a tissue and merge them.
    Computes Harmony if not present.
    
    Parameters:
    -----------
    tissue_dir : str
        Directory containing donor h5ad files
    tissue_name : str
        Name of tissue ('liver' or 'brain')
    output_dir : str
        Directory for output files
    gene_set : str
        Gene set identifier ('all', '5k', or 'xenium')
    """
    print(f"\n{'='*60}")
    print(f"Loading {tissue_name.upper()} tissue - {gene_set.upper()} gene set")
    print(f"{'='*60}")
    
    merged_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_merged.h5ad")
    
    if os.path.exists(merged_path):
        print(f"Found cached merged file: {merged_path}")
        adata = sc.read_h5ad(merged_path)
        print(f"Loaded: {adata.n_obs} cells")
    else:
        print("No cached file found, merging donor files...")
        # Find files matching the gene set pattern
        h5ad_files = list(Path(tissue_dir).glob(f"{tissue_name}_*_{gene_set}.h5ad"))
        
        if len(h5ad_files) == 0:
            raise ValueError(f"No {gene_set} h5ad files found in {tissue_dir}")
        
        print(f"Found {len(h5ad_files)} donor files for {gene_set} gene set")
        
        adata_list = []
        for h5ad_path in h5ad_files:
            donor_id = h5ad_path.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
            print(f"  Loading {donor_id}: ", end="")
            
            adata = sc.read_h5ad(h5ad_path)
            
            # Ensure donor_id is in obs
            if 'donor_id' not in adata.obs.columns:
                adata.obs['donor_id'] = donor_id
            
            print(f"{adata.n_obs} cells")
            adata_list.append(adata)
        
        # Merge all donors
        print(f"\n  Merging {len(adata_list)} donors...")
        adata = sc.concat(adata_list, join="outer", index_unique=None)
        print(f"  Total: {adata.n_obs} cells × {adata.n_vars} genes")
    
    updated = False
    
    # Check/Compute Harmony
    if "harmony" not in adata.obsm:
        print(f"\nComputing Harmony integration for {gene_set} gene set...")
        print("  Preprocessing for Harmony...")
        adata_harmony = adata.copy()
        
        # Basic normalization for PCA
        sc.pp.normalize_total(adata_harmony, target_sum=1e4)
        sc.pp.log1p(adata_harmony)
        sc.pp.highly_variable_genes(adata_harmony, n_top_genes=min(2000, adata_harmony.n_vars), subset=True)
        sc.pp.scale(adata_harmony, max_value=10)
        sc.tl.pca(adata_harmony, svd_solver='arpack')
        
        print("  Running Harmony...")
        sce.pp.harmony_integrate(adata_harmony, 'donor_id')
        
        adata.obsm['harmony'] = adata_harmony.obsm['X_pca_harmony']
        print("  Harmony computed.")
        updated = True
            
    if updated or not os.path.exists(merged_path):
        print(f"Saving merged file: {merged_path}")
        adata.write(merged_path)
            
    return adata


def compute_metrics(adata, embedding_key, cell_type_col, batch_col):
    """
    Compute only ASW (label) metric.
    """
    print(f"    Computing metrics for {embedding_key}...")
    
    # --- Bio Conservation ---
    print("      -> ASW (label)")
    asw_label = scib.metrics.silhouette(adata, group_key=cell_type_col, embed=embedding_key)
    
    return {
        "ASW_label": float(asw_label)
    }


def evaluate_tissue(adata, tissue_name, models):
    """
    Evaluate all available embeddings for a tissue.
    
    Args:
        adata: Merged AnnData object
        tissue_name: Name of tissue
        models: List of model names to evaluate
    
    Returns:
        DataFrame with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating embeddings for {tissue_name.upper()}")
    print(f"{'='*60}")
    
    # Find cell type column
    cell_type_col = None
    for col in ['cell_type', 'celltype', 'cell_type_ontology_term_id', 'celltype.l2']:
        if col in adata.obs.columns:
            cell_type_col = col
            break
    
    if cell_type_col is None:
        raise ValueError(f"No cell type column found in {tissue_name} data")
    
    # Get labels
    cell_type_labels = adata.obs[cell_type_col].astype(str).to_numpy()
    batch_labels = adata.obs['donor_id'].astype(str).to_numpy()
    
    n_cell_types = len(np.unique(cell_type_labels))
    n_batches = len(np.unique(batch_labels))
    
    print(f"  Cell type column: {cell_type_col} ({n_cell_types} types)")
    print(f"  Batch column: donor_id ({n_batches} donors)")
    
    results = []
    
    for model in models:
        key = MODEL_KEYS.get(model, model)
        if key not in adata.obsm:
            print(f"  ⚠ {model} (key={key}) embeddings not found, skipping...")
            continue
        
        print(f"\n  Evaluating {model}...")
        embeddings = adata.obsm[key]
        print(f"    Embedding shape: {embeddings.shape}")
        
        metrics = compute_metrics(adata, key, cell_type_col, 'donor_id')
        
        print(f"    Scores: {metrics}")
        
        result_row = {"Model": model}
        result_row.update(metrics)
        results.append(result_row)
    
    return pd.DataFrame(results)


def plot_metrics(df, tissue_name, output_dir, gene_set="all"):
    """
    Create bar plot showing Cell Type Purity for each model.
    """
    print(f"\n  Creating metrics visualization for {tissue_name} - {gene_set}...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metric = "ASW_label"
    colors = sns.color_palette("husl", len(df))
    
    # Create bar plot
    bars = ax.bar(df["Model"], df[metric], color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=14)
    
    ax.set_ylabel(metric, fontsize=16, fontweight='bold')
    ax.set_title(f'{metric} - {gene_set.upper()}', fontsize=18, fontweight='bold')
    ax.set_ylim([0, max(1.0, df[metric].max() * 1.1)])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels
    ax.set_xticklabels(df["Model"], rotation=45, ha='right', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_metrics.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()



def compute_and_cache_umaps(adata, tissue_name, models, output_dir, gene_set="all"):
    """
    Compute UMAPs for all models and cache them in the merged h5ad file.
    """
    print(f"\n{'='*60}")
    print(f"Computing/Caching UMAPs for {tissue_name.upper()} - {gene_set.upper()}")
    print(f"{'='*60}")
    
    save_needed = False
    
    for model in models:
        key = MODEL_KEYS.get(model, model)
        if key not in adata.obsm:
            continue
            
        umap_key = f"X_umap_{key}"
        
        if umap_key in adata.obsm:
            print(f"  {model}: UMAP already exists.")
        else:
            print(f"  {model}: Computing UMAP...")
            # Compute neighbors and UMAP
            # We use a temporary key for neighbors to avoid overwriting
            neighbors_key = f"neighbors_{key}"
            sc.pp.neighbors(adata, use_rep=key, key_added=neighbors_key, n_neighbors=15)
            sc.tl.umap(adata, neighbors_key=neighbors_key)
            
            # Store UMAP in specific key
            adata.obsm[umap_key] = adata.obsm['X_umap']
            save_needed = True
            print(f"    Computed and cached.")
            
    if save_needed:
        merged_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_merged.h5ad")
        print(f"\n  Saving updated cache to {merged_path}...")
        adata.write(merged_path)
    
    return adata



def plot_umap_grid(adata, tissue_name, models, output_dir, gene_set="all", metrics_df=None):
    """
    Create 5x2 UMAP grid: rows=models, columns=[donor, cell_type].
    Save legends separately.
    """
    print(f"\n  Creating UMAP grid visualization for {tissue_name} - {gene_set}...")
    
    # Find cell type column
    cell_type_col = None
    for col in ['cell_type', 'celltype', 'cell_type_ontology_term_id', 'celltype.l2']:
        if col in adata.obs.columns:
            cell_type_col = col
            break
    
    if cell_type_col is None:
        print(f"    Warning: No cell type column found, skipping UMAP grid")
        return
    
    # Filter models that have embeddings
    available_models = [m for m in models if MODEL_KEYS.get(m, m) in adata.obsm]
    
    if len(available_models) == 0:
        print(f"    Warning: No embeddings found, skipping UMAP grid")
        return
    
    # Create 5x2 grid (or however many models available)
    n_rows = len(available_models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
    
    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Color palettes (different for donor vs cell_type)
    donor_palette = "tab20"  # For donor
    celltype_palette = "tab20b"  # For cell_type (different scheme)
    
    for idx, model in enumerate(available_models):
        print(f"    Plotting {model}...")
        
        key = MODEL_KEYS.get(model, model)
        umap_key = f"X_umap_{key}"
        if umap_key not in adata.obsm:
             print(f"    Warning: {umap_key} not found, skipping plot for {model}")
             continue
             
        # Get metrics if available
        asw_label_str = ""
        # asw_batch_str = "" # Removed as we are not computing ASW_batch anymore
        if metrics_df is not None:
            row = metrics_df[metrics_df["Model"] == model]
            if not row.empty:
                val_l = row.iloc[0]["ASW_label"]
                asw_label_str = f"ASW: {val_l:.3f}"

        # Create a temporary adata for plotting to set X_umap correctly
        adata_plot = sc.AnnData(obs=adata.obs)
        adata_plot.obsm['X_umap'] = adata.obsm[umap_key]
        
        # Left column: donor_id
        sc.pl.umap(
            adata_plot,
            color='donor_id',
            ax=axes[idx, 0],
            show=False,
            title="", # Removed title
            frameon=False,
            legend_loc=None,  # No legend on plot
            palette=donor_palette,
        )
        
        # Right column: cell_type
        sc.pl.umap(
            adata_plot,
            color=cell_type_col,
            ax=axes[idx, 1],
            show=False,
            title="", # Removed title
            frameon=False,
            legend_loc=None,  # No legend on plot
            palette=celltype_palette,
        )
        # Add Label ASW score
        if asw_label_str:
            axes[idx, 1].text(0.5, -0.02, asw_label_str, transform=axes[idx, 1].transAxes,
                            fontsize=48, fontweight='bold', va='top', ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
    
    plt.tight_layout()
    
    # Save main figure
    fig_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_umap_grid.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()
    
    # Create separate legend figures
    print(f"    Creating separate legends...")
    
    # Donor legend
    fig_legend_donor = plt.figure(figsize=(3, 6))
    unique_donors = adata.obs['donor_id'].unique()
    donor_colors = sns.color_palette(donor_palette, len(unique_donors))
    
    for i, donor in enumerate(unique_donors):
        plt.plot([], [], 'o', color=donor_colors[i % len(donor_colors)], 
                label=donor, markersize=8)
    
    plt.legend(loc='center', frameon=False, fontsize=10)
    plt.axis('off')
    plt.title('Donor ID', fontsize=12, fontweight='bold')
    
    legend_donor_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_legend_donor.png")
    plt.savefig(legend_donor_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {legend_donor_path}")
    plt.close()
    
    # Cell type legend
    fig_legend_celltype = plt.figure(figsize=(3, 8))
    unique_celltypes = adata.obs[cell_type_col].unique()
    celltype_colors = sns.color_palette(celltype_palette, len(unique_celltypes))
    
    for i, celltype in enumerate(unique_celltypes):
        plt.plot([], [], 'o', color=celltype_colors[i % len(celltype_colors)], 
                label=celltype, markersize=8)
    
    plt.legend(loc='center', frameon=False, fontsize=9)
    plt.axis('off')
    plt.title('Cell Type', fontsize=12, fontweight='bold')
    
    legend_celltype_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_legend_celltype.png")
    plt.savefig(legend_celltype_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {legend_celltype_path}")
    plt.close()



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def save_timing_data(output_dir):
    """
    Collect timing logs from all models and save combined CSV.
    """
    print(f"\n{'='*60}")
    print("Saving Model Inference Time Data")
    print(f"{'='*60}")
    
    # Define paths to timing CSVs
    base_dir = os.path.dirname(output_dir)
    
    timing_files = {
        "xVerse": os.path.join(base_dir, "xverse_embeddings", "xverse_inference_timing.csv"),
        "Geneformer": os.path.join(base_dir, "geneformer_embeddings", "geneformer_inference_timing.csv"),
        "Nicheformer": os.path.join(base_dir, "nicheformer_embeddings", "nicheformer_inference_timing.csv"),
        "scGPT": os.path.join(base_dir, "scgpt_embeddings", "scgpt_inference_timing.csv")
    }
    
    dfs = []
    for model, filepath in timing_files.items():
        if os.path.exists(filepath):
            print(f"  Loading {model} timing from: {filepath}")
            try:
                df = pd.read_csv(filepath)
                # Ensure model name is consistent
                df['model'] = model
                dfs.append(df)
            except Exception as e:
                print(f"    Error loading {filepath}: {e}")
        else:
            print(f"  ⚠️ Warning: Timing file not found for {model}: {filepath}")
    
    if not dfs:
        print("  No timing data found.")
        return
    
    # Combine all data
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Save combined CSV
    combined_csv_path = os.path.join(output_dir, "all_models_inference_timing.csv")
    df_all.to_csv(combined_csv_path, index=False)
    print(f"  Saved combined timing data: {combined_csv_path}")


if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("Embedding Quality Evaluation - Multi-Panel Gene Sets")
    print("="*60)
    
    # Gene sets to evaluate
    gene_sets = ["all", "5k", "xenium"]
    
    all_results = {}
    
    for gene_set in gene_sets:
        print(f"\n{'='*80}")
        print(f"EVALUATING GENE SET: {gene_set.upper()}")
        print(f"{'='*80}")
        
        # Evaluate liver
        adata_liver = load_and_merge_tissue(liver_dir, "liver", output_dir, gene_set)
        
        adata_liver = compute_and_cache_umaps(adata_liver, "liver", MODELS, output_dir, gene_set)
        
        df_liver = evaluate_tissue(adata_liver, "liver", MODELS)
        
        plot_metrics(df_liver, "liver", output_dir, gene_set)
        plot_umap_grid(adata_liver, "liver", MODELS, output_dir, gene_set, metrics_df=df_liver)
        
        all_results[f"liver_{gene_set}"] = df_liver
        
        # Evaluate brain
        adata_brain = load_and_merge_tissue(brain_dir, "brain", output_dir, gene_set)
        
        adata_brain = compute_and_cache_umaps(adata_brain, "brain", MODELS, output_dir, gene_set)
        
        df_brain = evaluate_tissue(adata_brain, "brain", MODELS)
        
        plot_metrics(df_brain, "brain", output_dir, gene_set)
        plot_umap_grid(adata_brain, "brain", MODELS, output_dir, gene_set, metrics_df=df_brain)
        
        all_results[f"brain_{gene_set}"] = df_brain
    
    # Save results to CSV
    print(f"\n{'='*60}")
    print("Saving results")
    print(f"{'='*60}")
    
    for key, df in all_results.items():
        csv_path = os.path.join(output_dir, f"{key}_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    
    # Save Timing Data (Visualization moved to 07_visualize_results.py)
    save_timing_data(output_dir)
    
    # --- Calculate and Print Manuscript Metrics ---
    print(f"\n{'='*60}")
    print("Manuscript Metrics Calculation")
    print(f"{'='*60}")

    # 1. Inference Speed Improvement (xVERSE vs scGPT)
    timing_df = pd.read_csv(os.path.join(output_dir, "all_models_inference_timing.csv"))
    if not timing_df.empty:
        # Average time across all gene sets for each model
        avg_times = timing_df.groupby('model')['time_seconds'].mean()
        if 'xVerse' in avg_times and 'scGPT' in avg_times:
            time_xverse = avg_times['xVerse']
            time_scgpt = avg_times['scGPT']
            speed_improvement = (time_scgpt - time_xverse) / time_scgpt * 100
            print(f"Average Inference Time - xVerse: {time_xverse:.2f}s, scGPT: {time_scgpt:.2f}s")
            print(f"Inference speed improvement (xVERSE vs scGPT): {speed_improvement:.1f}%")
        else:
            print("Could not calculate speed improvement: xVerse or scGPT timing missing.")

    # 2. ASW Improvement (xVERSE vs Harmony) and (xVERSE vs scGPT)
    # Collect all ASW scores for xVerse, Harmony, and scGPT across all tasks
    xverse_asw = []
    harmony_asw = []
    scgpt_asw = []
    
    for key, df in all_results.items():
        # df has columns: Model, ASW_label
        row_xverse = df[df['Model'] == 'xVerse']
        row_harmony = df[df['Model'] == 'Harmony']
        row_scgpt = df[df['Model'] == 'scGPT']
        
        if not row_xverse.empty:
            xverse_asw.append(row_xverse.iloc[0]['ASW_label'])
        if not row_harmony.empty:
            harmony_asw.append(row_harmony.iloc[0]['ASW_label'])
        if not row_scgpt.empty:
            scgpt_asw.append(row_scgpt.iloc[0]['ASW_label'])
            
    if xverse_asw and harmony_asw:
        avg_asw_xverse = np.mean(xverse_asw)
        avg_asw_harmony = np.mean(harmony_asw)
        asw_improvement_harmony = (avg_asw_xverse - avg_asw_harmony) / avg_asw_harmony * 100
        
        print(f"Average ASW - xVerse: {avg_asw_xverse:.4f}, Harmony: {avg_asw_harmony:.4f}")
        print(f"Average ASW increase (xVERSE vs Harmony): +{asw_improvement_harmony:.1f}%")
    else:
        print("Could not calculate ASW improvement vs Harmony: scores missing.")

    if xverse_asw and scgpt_asw:
        avg_asw_scgpt = np.mean(scgpt_asw)
        asw_improvement_scgpt = (avg_asw_xverse - avg_asw_scgpt) / avg_asw_scgpt * 100
        
        print(f"Average ASW - xVerse: {avg_asw_xverse:.4f}, scGPT: {avg_asw_scgpt:.4f}")
        print(f"Average ASW increase (xVERSE vs scGPT): +{asw_improvement_scgpt:.1f}%")

        # Calculate specific improvements for 5k and xenium panels
        # Keys in all_results are like "liver_5k", "brain_xenium"
        xverse_5k = []
        scgpt_5k = []
        xverse_xenium = []
        scgpt_xenium = []

        for key, df in all_results.items():
            if "5k" in key:
                row_x = df[df['Model'] == 'xVerse']
                row_s = df[df['Model'] == 'scGPT']
                if not row_x.empty: xverse_5k.append(row_x.iloc[0]['ASW_label'])
                if not row_s.empty: scgpt_5k.append(row_s.iloc[0]['ASW_label'])
            elif "xenium" in key:
                row_x = df[df['Model'] == 'xVerse']
                row_s = df[df['Model'] == 'scGPT']
                if not row_x.empty: xverse_xenium.append(row_x.iloc[0]['ASW_label'])
                if not row_s.empty: scgpt_xenium.append(row_s.iloc[0]['ASW_label'])
        
        if xverse_5k and scgpt_5k:
            imp_5k = (np.mean(xverse_5k) - np.mean(scgpt_5k)) / np.mean(scgpt_5k) * 100
            print(f"ASW increase (xVERSE vs scGPT) on 5K panel: +{imp_5k:.1f}%")
        
        if xverse_xenium and scgpt_xenium:
            imp_xenium = (np.mean(xverse_xenium) - np.mean(scgpt_xenium)) / np.mean(scgpt_xenium) * 100
            print(f"ASW increase (xVERSE vs scGPT) on Xenium panel: +{imp_xenium:.1f}%")

    else:
        print("Could not calculate ASW improvement vs scGPT: scores missing.")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    for key, df in all_results.items():
        print(f"\n{key.upper()}:")
        print(df.to_string(index=False))
    
    # Save timing log
    log_path = os.path.join(output_dir, "timing_log.json")
    
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
