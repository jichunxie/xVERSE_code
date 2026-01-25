import os
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# ================= CONFIGURATION =================
BASE_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results"
GT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split"
FT_DIR = os.path.join(BASE_DIR, "ft")

OUTPUT_DIR = os.path.join(BASE_DIR, "figures", "spatial_heatmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = ["Lung5_Rep1", "Lung6", "Lung9_Rep1", "Lung13"]
# ================= HELPER FUNCTIONS =================

def load_ground_truth(sample):
    """Load Ground Truth (Test Set)"""
    path = os.path.join(GT_DIR, f"test_{sample}.h5ad")
    if not os.path.exists(path):
        print(f"Warning: GT file not found for {sample}")
        return None
    adata = sc.read_h5ad(path)
    # Ensure gene_ids are index or column
    if "gene_ids" in adata.var.columns:
        # Save original index as gene_name (assuming it might be symbol)
        adata.var["gene_name"] = adata.var.index
        adata.var.index = adata.var["gene_ids"]
    return adata

def load_imputed_xverse(sample, mode="ft"):
    """Load xVerse Imputed Data (mu_bio)"""
    subdir = FT_DIR # Only FT needed
    path = os.path.join(subdir, f"{sample}_mu_bio.h5ad")
    if not os.path.exists(path):
        print(f"Warning: xVerse {mode} file not found for {sample}")
        return None
    adata = sc.read_h5ad(path)
    if "gene_ids" in adata.var.columns:
        adata.var.index = adata.var["gene_ids"]
    return adata

def get_spatial_coords(adata_gt):
    """Extract spatial coordinates from GT adata"""
    # Check for user-specified location first
    if "spatial_global" in adata_gt.obsm:
        return adata_gt.obsm["spatial_global"]
    
    if "CenterX_global_px" in adata_gt.obs and "CenterY_global_px" in adata_gt.obs:
        return adata_gt.obs[["CenterX_global_px", "CenterY_global_px"]].values

    # Fallbacks
    if "spatial" in adata_gt.obsm:
        return adata_gt.obsm["spatial"]
    elif "global_x" in adata_gt.obs and "global_y" in adata_gt.obs:
        return adata_gt.obs[["global_x", "global_y"]].values
    elif "fov_x" in adata_gt.obs and "fov_y" in adata_gt.obs:
        return adata_gt.obs[["fov_x", "fov_y"]].values
    return None

def plot_gene_spatial_comparison(sample, gene_id, gene_name, adata_gt, adata_ft):
    """Plot spatial heatmap for a single gene: GT vs xVerse FT"""
    
    coords = get_spatial_coords(adata_gt)
    if coords is None:
        print(f"Error: No spatial coords for {sample}")
        return

    # Print coordinate ranges
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    print(f"Sample: {sample} | Spatial Coords Range:")
    print(f"  X: min={x_min:.2f}, max={x_max:.2f}, range={x_max-x_min:.2f}")
    print(f"  Y: min={y_min:.2f}, max={y_max:.2f}, range={y_max-y_min:.2f}")

    # Extract expression values
    if gene_id not in adata_gt.var_names or gene_id not in adata_ft.var_names:
        print(f"Gene {gene_id} ({gene_name}) not found in GT or FT for {sample}")
        return

    # GT Expression
    if hasattr(adata_gt.X, "toarray"):
        gt_expr = adata_gt[:, gene_id].X.toarray().flatten()
    else:
        gt_expr = adata_gt[:, gene_id].X.flatten()

    # FT Expression
    if hasattr(adata_ft.X, "toarray"):
        ft_expr = adata_ft[:, gene_id].X.toarray().flatten()
    else:
        ft_expr = adata_ft[:, gene_id].X.flatten()

    # Calculate Pearson Correlation
    correlation = np.corrcoef(gt_expr, ft_expr)[0, 1]
    print(f"Sample: {sample}, Gene: {gene_name} ({gene_id}), Pearson Corr: {correlation:.4f}")

    # Plotting
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    plt.suptitle(gene_name, fontsize=30)
    

    vmin_gt, vmax_gt = np.percentile(gt_expr, 0), np.percentile(gt_expr, 90)
    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=gt_expr, s=0.5, linewidth=0, cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)
    axes[0].axis('off')
    axes[0].set_aspect('equal')
    
    # xVerse Fine-Tuned
    vmin_ft, vmax_ft = np.percentile(ft_expr, 0), np.percentile(ft_expr, 90)
    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=ft_expr, s=0.5, linewidth=0, cmap="viridis", vmin=vmin_ft, vmax=vmax_ft)
    axes[1].axis('off')
    axes[1].set_aspect('equal')
          

    save_path = os.path.join(OUTPUT_DIR, f"{sample}_{gene_name}_spatial_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

# ================= MAIN =================

def calculate_all_correlations(adata_gt, adata_ft):
    """Calculate Pearson correlation for all common genes efficiently"""
    # Verify alignment (Indices should be gene_ids)
    print(f"  GT Index Sample: {adata_gt.var_names[:3].tolist()}")
    print(f"  FT Index Sample: {adata_ft.var_names[:3].tolist()}")

    # Find common genes
    common_genes = list(set(adata_gt.var_names) & set(adata_ft.var_names))
    
    # Check spot alignment by order (User specified: obs names differ, align by order)
    if adata_gt.n_obs != adata_ft.n_obs:
        print(f"  Warning: Spot count mismatch! GT: {adata_gt.n_obs}, FT: {adata_ft.n_obs}")
        # Truncate to minimum to avoid errors
        n_spots = min(adata_gt.n_obs, adata_ft.n_obs)
        adata_gt = adata_gt[:n_spots, :]
        adata_ft = adata_ft[:n_spots, :]
    else:
        n_spots = adata_gt.n_obs

    print(f"  Common Genes: {len(common_genes)}")
    print(f"  Spots used: {n_spots}")
    
    if not common_genes or n_spots == 0:
        return {}, []

    # Subset data (genes only, spots aligned by order)
    gt_sub = adata_gt[:, common_genes]
    ft_sub = adata_ft[:, common_genes]
    
    # Convert to dense if sparse
    X_gt = gt_sub.X.toarray() if hasattr(gt_sub.X, "toarray") else gt_sub.X
    X_ft = ft_sub.X.toarray() if hasattr(ft_sub.X, "toarray") else ft_sub.X
    
    # Vectorized correlation
    # Center columns
    X_gt_c = X_gt - X_gt.mean(axis=0)
    X_ft_c = X_ft - X_ft.mean(axis=0)
    
    # Compute correlation
    num = np.sum(X_gt_c * X_ft_c, axis=0)
    den = np.sqrt(np.sum(X_gt_c**2, axis=0) * np.sum(X_ft_c**2, axis=0))
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = num / den
    
    # Map back to gene names/ids
    gene_corrs = dict(zip(common_genes, corrs))
    
    # Remove NaNs
    gene_corrs = {k: v for k, v in gene_corrs.items() if not np.isnan(v)}
    
    return gene_corrs, common_genes

def main():
    print("Generating 4x2 spatial heatmap figure...")
    
    # Create 4x2 figure
    fig, axes = plt.subplots(4, 2, figsize=(8, 18)) 
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(SAMPLES):
        print(f"\nProcessing {sample} ({i+1}/{len(SAMPLES)})...")
        adata_gt = load_ground_truth(sample)
        if adata_gt is None: continue
        
        adata_ft = load_imputed_xverse(sample, "ft")
        if adata_ft is None: continue
        
        # Calculate correlations for all genes
        print("Calculating correlations...")
        gene_corrs, common_genes = calculate_all_correlations(adata_gt, adata_ft)
        
        if not gene_corrs:
            print(f"No valid correlations found for {sample}")
            continue
            
        # Find best gene
        best_gene_id = max(gene_corrs, key=gene_corrs.get)
        best_corr = gene_corrs[best_gene_id]
        
        # Get gene name (symbol) for title
        best_gene_name = best_gene_id
        
        # We saved the original index (gene name) into 'gene_name' column in load_ground_truth
        if "gene_name" in adata_gt.var.columns:
             best_gene_name = adata_gt.var.loc[best_gene_id, "gene_name"]
        else:
            # Fallback checks
            for col in ['gene_symbols', 'Symbol', 'feature_name']:
                if col in adata_gt.var.columns:
                    best_gene_name = adata_gt.var.loc[best_gene_id, col]
                    break

        print(f"Best Gene: {best_gene_name} ({best_gene_id}) | Pearson: {best_corr:.4f}")
        
        # Plot on the specific row
        ax_gt = axes[i, 0]
        ax_ft = axes[i, 1]
        
        # Get coords
        coords = get_spatial_coords(adata_gt)
        if coords is None:
            print(f"Error: No spatial coords for {sample}")
            continue

        # Extract expression
        if hasattr(adata_gt.X, "toarray"):
            gt_expr = adata_gt[:, best_gene_id].X.toarray().flatten()
        else:
            gt_expr = adata_gt[:, best_gene_id].X.flatten()

        if hasattr(adata_ft.X, "toarray"):
            ft_expr = adata_ft[:, best_gene_id].X.toarray().flatten()
        else:
            ft_expr = adata_ft[:, best_gene_id].X.flatten()
            
        # Ground Truth
        vmin_gt, vmax_gt = np.percentile(gt_expr, 0), np.percentile(gt_expr, 95)
        ax_gt.scatter(coords[:, 0], coords[:, 1], c=gt_expr, s=0.5, linewidth=0, cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)
        ax_gt.axis('off')
        ax_gt.set_aspect('equal')
        
        # Set title centered over the row (spanning both columns)
        ax_gt.set_title(" ", fontsize=30, pad=0)
        
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(fig.transFigure, ax_gt.transAxes)
        ax_gt.text(0.5, 1.0, best_gene_name, transform=trans, ha='center', va='bottom', fontsize=30)

        # xVerse Fine-Tuned
        vmin_ft, vmax_ft = np.percentile(ft_expr, 0), np.percentile(ft_expr, 95)
        ax_ft.scatter(coords[:, 0], coords[:, 1], c=ft_expr, s=0.5, linewidth=0, cmap="viridis", vmin=vmin_ft, vmax=vmax_ft)
        ax_ft.axis('off')
        ax_ft.set_aspect('equal')

    save_path = os.path.join(OUTPUT_DIR, "spatial_heatmaps_4x2_best_genes.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
