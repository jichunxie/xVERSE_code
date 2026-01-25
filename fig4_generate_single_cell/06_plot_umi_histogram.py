# -*- coding: utf-8 -*-
"""Compare UMI samples from XVerse and scVI by plotting histograms."""

import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


# ================================================================
# Configuration
# ================================================================

SAMPLES = ["PM-A", "PM-B", "PM-C", "PM-D", "PM-E"]

RAW_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad"
XV_INFER_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/ft_model/inference_full"
SCVI_INFER_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/scvi_full/umi_samples_full"
GENE_IDS_PATH = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

N_SAMPLES = 3  # number of per-cell draws saved per model
TOP_HVG = 20   # 10 rows * 2 columns = 20 genes

OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/plots/umi_sample_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# Helper functions
# ================================================================

def load_allowed_gene_ids(path):
    with open(path, "r") as f:
        genes = [line.strip() for line in f if line.strip()]
    return set(genes)


def read_filtered_h5ad(path, allowed_gene_ids=None):
    ad = sc.read_h5ad(path)
    if allowed_gene_ids is not None and "gene_ids" in ad.var:
        mask = ad.var["gene_ids"].astype(str).isin(allowed_gene_ids)
        ad = ad[:, mask].copy()
    return ad


def load_counts(path, allowed_gene_ids=None):
    """Load counts as dense numpy array along with gene_ids."""
    ad = read_filtered_h5ad(path, allowed_gene_ids)
    X = ad.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    gene_ids = (
        ad.var["gene_ids"].astype(str).tolist()
        if "gene_ids" in ad.var
        else ad.var_names.astype(str).tolist()
    )
    var_ids = ad.var_names.astype(str).tolist()
    return X, gene_ids, var_ids


def subset_by_gene_ids(matrix, gene_ids, target_genes):
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    idx = [gene_to_idx[g] for g in target_genes]
    return matrix[:, idx]


def load_xverse_samples(sample_name, allowed_gene_ids):
    """Load XVerse Poisson samples derived from concatenated sample file."""
    sample_path = os.path.join(
        XV_INFER_DIR, sample_name, f"{sample_name}_estimated_expr_poisson.h5ad"
    )
    if not os.path.exists(sample_path):
        print(f"[WARN] Missing XVerse sample file: {sample_path}")
        return [], None, None

    full_counts, gene_ids, var_ids = load_counts(sample_path, allowed_gene_ids)
    n_cells = full_counts.shape[0]
    if n_cells % N_SAMPLES != 0:
        print(
            f"[WARN] Cell count {n_cells} not divisible by {N_SAMPLES} for {sample_name}; "
            "splitting by floor."
        )
    split = np.array_split(full_counts, N_SAMPLES, axis=0)
    return split, gene_ids, var_ids


def load_scvi_samples(sample_name, allowed_gene_ids):
    """Load scVI samples from separate files."""
    draws = []
    gene_ids = None
    var_ids = None

    for i in range(1, N_SAMPLES + 1):
        path = os.path.join(
            SCVI_INFER_DIR, sample_name, f"{sample_name}_umi_sample{i}_full.h5ad"
        )
        if not os.path.exists(path):
            print(f"[WARN] Missing scVI sample file: {path}")
            continue
        
        counts, g_ids, v_ids = load_counts(path, allowed_gene_ids)
        draws.append(counts)
        
        # Assume all draws have same genes
        if gene_ids is None:
            gene_ids = g_ids
            var_ids = v_ids
            
    return draws, gene_ids, var_ids


def select_top_hvg(ad_raw, top_n=TOP_HVG):
    """Select top HVG genes from raw data (simple variance-based approach)."""
    data = ad_raw.X
    if hasattr(data, "toarray"):
        data = data.toarray()
    var = np.var(data, axis=0)
    idx = np.argsort(var)[::-1][:top_n]
    gene_ids = (
        ad_raw.var["gene_ids"].astype(str).tolist()
        if "gene_ids" in ad_raw.var
        else ad_raw.var_names.astype(str).tolist()
    )
    var_names = ad_raw.var_names.astype(str).tolist()
    return [(i, gene_ids[i], var_names[i]) for i in idx]


def plot_individual_histograms(sample_name, genes_info, raw_counts, model_draws, model_name, gene_map):
    """Plot separate histograms for the top 20 genes."""
    bins = np.arange(0, 151)
    
    for i, (gene_idx, gene_id, gene_label) in enumerate(genes_info):
        if gene_id not in gene_map:
            print(f"[WARN] Gene {gene_id} missing in common set; skip.")
            continue
            
        idx = gene_map[gene_id]
        raw_series = raw_counts[:, idx].flatten()
        model_series_list = [draw[:, idx].flatten() for draw in model_draws]
        
        # Determine Y limit
        hist_raw = np.histogram(raw_series, bins=bins)[0]
        max_count = hist_raw.max()
        for ms in model_series_list:
            h = np.histogram(ms, bins=bins)[0]
            max_count = max(max_count, h.max())
            
        y_lim = max(1, max_count) * 1.05
        
        # Create 1x4 figure (Height increased 1.5x: 4 -> 6)
        fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
        
        # Plot Raw (Biological Cells)
        ax = axes[0]
        ax.hist(raw_series, bins=bins, color="black", alpha=0.5, label="Biological Cells", density=False)
        ax.set_title("Biological Cells", fontsize=28) # Font size 2x
        ax.set_xlim(0, 150)
        ax.set_ylim(0, y_lim)
        ax.tick_params(axis='both', which='major', labelsize=24) # Increase tick label size
        
        # Plot Virtual Samples
        for j, ms in enumerate(model_series_list):
            if j >= 3: break # Ensure we don't exceed 3 virtual samples
            ax = axes[j+1]
            color = "#377eb8" if model_name == "xVERSE" else "#e41a1c"
            ax.hist(ms, bins=bins, color=color, alpha=0.5, label=f"Virtual C{j+1}", density=False)
            ax.set_title(f"Virtual C{j+1}", fontsize=28) # Font size 2x
            ax.set_xlim(0, 150)
            ax.tick_params(axis='both', which='major', labelsize=24) # Increase tick label size
            # ShareY handles ylim
            
        fig.suptitle(f"{sample_name} - {gene_label} ({model_name})", fontsize=32) # Font size 2x
        fig.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust rect to accommodate larger title

        out_path = os.path.join(
            OUTPUT_DIR, f"{sample_name}_{model_name}_{gene_label}_1x4_hist.jpg"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved 1x4 histogram: {out_path}")


def plot_combined_grid(sample_name, genes_info, raw_counts, model_draws, model_name, gene_map):
    """Plot 20x4 combined grid histograms."""
    # 20 rows (genes), 4 columns (Biological + 3 Virtual)
    fig, axes = plt.subplots(len(genes_info), 4, figsize=(20, len(genes_info) * 3), sharey='row')
    
    bins = np.arange(0, 151)
    
    for i, (gene_idx, gene_id, gene_label) in enumerate(genes_info):
        if gene_id not in gene_map:
            continue
            
        idx = gene_map[gene_id]
        raw_series = raw_counts[:, idx].flatten()
        model_series_list = [draw[:, idx].flatten() for draw in model_draws]
        
        # Determine Y limit
        hist_raw = np.histogram(raw_series, bins=bins)[0]
        max_count = hist_raw.max()
        for ms in model_series_list:
            h = np.histogram(ms, bins=bins)[0]
            max_count = max(max_count, h.max())
        y_lim = max(1, max_count) * 1.05
        
        # Plot Raw (Biological Cells) - Column 0
        ax = axes[i, 0]
        ax.hist(raw_series, bins=bins, color="black", alpha=0.5, label="Biological Cells", density=False)
        ax.set_title(f"{gene_label} - Biological Cells", fontsize=12)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, y_lim)
        
        # Plot Virtual Samples - Columns 1-3
        for j, ms in enumerate(model_series_list):
            if j >= 3: break
            ax = axes[i, j+1]
            color = "#377eb8" if model_name == "xVERSE" else "#e41a1c"
            ax.hist(ms, bins=bins, color=color, alpha=0.5, label=f"Virtual C{j+1}", density=False)
            ax.set_title(f"{gene_label} - Virtual C{j+1}", fontsize=12)
            ax.set_xlim(0, 150)
            
    fig.suptitle(f"{sample_name} – {model_name} Combined Grid", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_path = os.path.join(
        OUTPUT_DIR, f"{sample_name}_{model_name}_combined_20x4_grid.jpg"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved combined grid: {out_path}")

# ================================================================
# Main
# ================================================================

def main():
    allowed_gene_ids = load_allowed_gene_ids(GENE_IDS_PATH)

    for sample in SAMPLES:
        print(f"\n=== Processing {sample} ===")
        raw_path = os.path.join(RAW_DIR, f"{sample}.h5ad")
        if not os.path.exists(raw_path):
            print(f"[WARN] Missing raw file: {raw_path}")
            continue

        # Load Raw
        ad_raw = read_filtered_h5ad(raw_path, allowed_gene_ids)
        raw_counts = ad_raw.X
        if hasattr(raw_counts, "toarray"):
            raw_counts = raw_counts.toarray()
        ad_raw_dense = ad_raw.copy()
        ad_raw_dense.X = raw_counts
        raw_gene_ids = (
            ad_raw.var["gene_ids"].astype(str).tolist()
            if "gene_ids" in ad_raw.var
            else ad_raw.var_names.astype(str).tolist()
        )

        # Load xVerse
        xv_draws, xv_gene_ids, _ = load_xverse_samples(sample, allowed_gene_ids)
        
        # Load scVI
        scvi_draws, scvi_gene_ids, _ = load_scvi_samples(sample, allowed_gene_ids)

        # Find common genes (intersection of all available)
        common_genes = set(raw_gene_ids)
        if xv_gene_ids:
            common_genes &= set(xv_gene_ids)
        if scvi_gene_ids:
            common_genes &= set(scvi_gene_ids)
            
        common_genes = sorted(common_genes)
        if len(common_genes) == 0:
            print(f"[WARN] No common gene_ids after filtering for {sample}; skip.")
            continue
            
        # Subset Raw
        raw_counts_sub = subset_by_gene_ids(raw_counts, raw_gene_ids, common_genes)
        
        # Select Top HVGs from Raw
        genes_info = select_top_hvg(ad_raw_dense, TOP_HVG)
        
        gene_map = {g: i for i, g in enumerate(common_genes)}
        
        # Plot xVerse
        if xv_draws:
            xv_draws_sub = [subset_by_gene_ids(draw, xv_gene_ids, common_genes) for draw in xv_draws]
            plot_individual_histograms(sample, genes_info, raw_counts_sub, xv_draws_sub, "xVERSE", gene_map)
            plot_combined_grid(sample, genes_info, raw_counts_sub, xv_draws_sub, "xVERSE", gene_map)
        else:
            print(f"[WARN] Skipping XVerse plot for {sample} due to missing data.")

        # Plot scVI
        if scvi_draws:
            scvi_draws_sub = [subset_by_gene_ids(draw, scvi_gene_ids, common_genes) for draw in scvi_draws]
            plot_individual_histograms(sample, genes_info, raw_counts_sub, scvi_draws_sub, "scVI", gene_map)
            plot_combined_grid(sample, genes_info, raw_counts_sub, scvi_draws_sub, "scVI", gene_map)
        else:
            print(f"[WARN] Skipping scVI plot for {sample} due to missing data.")


if __name__ == "__main__":
    main()
