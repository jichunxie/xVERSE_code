#!/usr/bin/env python3
# https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/nsclc-ffpe-dataset/
"""
Preprocess Nanostring CosMx SMI data
Convert CSV files (expression matrix + metadata) to h5ad format
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import anndata as ad
import scanpy as sc
from pathlib import Path

# Configuration
DATA_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/data"
OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data"

# Sample names based on the folder structure
SAMPLES = [
    "Lung13",
    "Lung9_Rep1", 
    "Lung6",
    "Lung5_Rep1"
]

# Gene mapping configuration
GENE_INFO_PATH = "/hpc/group/xielab/xj58/general/gene_info_table.csv"


def load_gene_mapping():
    """Load gene_name -> ensembl_id mapping, keeping only unique gene names."""
    gene_info = pd.read_csv(GENE_INFO_PATH)
    gene_info_unique = gene_info.drop_duplicates(subset=["gene_name"], keep=False)
    return dict(zip(gene_info_unique["gene_name"], gene_info_unique["ensembl_id"]))


def keep_unique_gene_ids(adata, mapping):
    """Attach gene IDs and drop genes with ambiguous or missing IDs."""
    adata.var_names = adata.var_names.astype(str)
    adata.var["gene_symbol"] = adata.var_names
    adata.var["gene_ids"] = adata.var["gene_symbol"].map(mapping)
    mask = adata.var["gene_ids"].notnull() & ~adata.var["gene_ids"].duplicated(keep=False)
    return adata[:, mask].copy()


def apply_qc_filters(adata, min_counts=10, min_genes=5, max_counts_percentile=99):
    """
    Apply lenient QC filters for targeted spatial transcriptomics data.
    
    Args:
        adata: AnnData object
        min_counts: Minimum total counts per cell (default: 10, very lenient)
        min_genes: Minimum number of detected genes per cell (default: 5, very lenient)
        max_counts_percentile: Remove cells above this percentile of total counts (default: 99)
    
    Returns:
        Filtered AnnData object
    """
    n_cells_before = adata.n_obs
    
    # Calculate max_counts threshold from percentile
    max_counts = np.percentile(adata.obs['n_counts'], max_counts_percentile)
    
    # Filter cells
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    
    n_cells_after = adata.n_obs
    n_removed = n_cells_before - n_cells_after
    
    print(f"  QC filtering: {n_cells_before} -> {n_cells_after} cells ({n_removed} removed)")
    print(f"    Thresholds: min_counts={min_counts}, min_genes={min_genes}, max_counts={max_counts:.0f} (p{max_counts_percentile})")
    
    return adata


def print_adata_summary(adata, sample_name):
    """Print summary information about the AnnData object."""
    print(f"\n{'='*60}")
    print(f"Summary for {sample_name}")
    print(f"{'='*60}")
    
    # Basic dimensions
    print(f"Cells: {adata.n_obs:,}")
    print(f"Genes: {adata.n_vars:,}")
    
    # Sparsity
    if hasattr(adata.X, 'nnz'):  # sparse matrix
        sparsity = 100 * (1 - adata.X.nnz / (adata.n_obs * adata.n_vars))
        print(f"Sparsity: {sparsity:.2f}%")
    
    # Count statistics
    print(f"\nCounts per cell:")
    print(f"  Mean: {adata.obs['n_counts'].mean():.1f}")
    print(f"  Median: {adata.obs['n_counts'].median():.1f}")
    print(f"  Min: {adata.obs['n_counts'].min():.0f}")
    print(f"  Max: {adata.obs['n_counts'].max():.0f}")
    
    print(f"\nGenes per cell:")
    print(f"  Mean: {adata.obs['n_genes'].mean():.1f}")
    print(f"  Median: {adata.obs['n_genes'].median():.1f}")
    print(f"  Min: {adata.obs['n_genes'].min():.0f}")
    print(f"  Max: {adata.obs['n_genes'].max():.0f}")
    
    # FOV information
    n_fovs = adata.obs['fov'].nunique()
    print(f"\nField of Views (FOVs): {n_fovs}")
    print(f"Cells per FOV: {adata.n_obs / n_fovs:.1f} (average)")
    
    # Spatial extent
    print(f"\nSpatial extent (pixels):")
    print(f"  X: [{adata.obs['CenterX_global_px'].min():.0f}, {adata.obs['CenterX_global_px'].max():.0f}]")
    print(f"  Y: [{adata.obs['CenterY_global_px'].min():.0f}, {adata.obs['CenterY_global_px'].max():.0f}]")
    
    # Gene info
    if 'is_neg_probe' in adata.var.columns:
        n_neg_probes = adata.var['is_neg_probe'].sum()
        print(f"\nNegative control probes: {n_neg_probes}")
        print(f"Protein-coding genes: {adata.n_vars - n_neg_probes}")
    
    print(f"{'='*60}\n")


def find_csv_files(sample_dir):
    """
    Find the three CSV files in nested directory structure
    
    Args:
        sample_dir: Path to sample directory (e.g., /path/to/Lung13)
    
    Returns:
        dict with keys: 'expr', 'metadata', 'transcript' (paths to CSV files)
    """
    csv_files = {}
    
    # Search recursively for CSV files
    for csv_path in glob.glob(os.path.join(sample_dir, "**", "*.csv"), recursive=True):
        basename = os.path.basename(csv_path)
        
        if "exprMat_file.csv" in basename:
            csv_files['expr'] = csv_path
        elif "metadata_file.csv" in basename:
            csv_files['metadata'] = csv_path
        elif "tx_file.csv" in basename:
            csv_files['transcript'] = csv_path  # Not used for now
    
    return csv_files


def create_anndata(expr_path, metadata_path, sample_name):
    """
    Create AnnData object from expression matrix and metadata CSV files
    
    Args:
        expr_path: Path to expression matrix CSV
        metadata_path: Path to metadata CSV
        sample_name: Name of the sample (e.g., "Lung13")
    
    Returns:
        AnnData object
    """
    print(f"  Reading expression matrix: {expr_path}")
    expr_df = pd.read_csv(expr_path)
    
    print(f"  Reading metadata: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    
    # Filter out cell_ID = 0 (unassigned/extracellular transcripts)
    print(f"  Filtering out cell_ID=0 (extracellular transcripts)...")
    n_total = len(expr_df)
    expr_df = expr_df[expr_df['cell_ID'] != 0].copy()
    metadata_df = metadata_df[metadata_df['cell_ID'] != 0].copy()
    n_filtered = len(expr_df)
    print(f"    Kept {n_filtered}/{n_total} cells ({n_total - n_filtered} extracellular removed)")
    
    # Extract gene names (all columns except 'fov' and 'cell_ID')
    gene_cols = [col for col in expr_df.columns if col not in ['fov', 'cell_ID']]

    
    # Create cell IDs: combine fov and cell_ID for unique identification
    cell_ids = expr_df['fov'].astype(str) + "_" + expr_df['cell_ID'].astype(str)
    
    # Extract expression matrix (cells x genes) as sparse matrix
    X = csr_matrix(expr_df[gene_cols].values)
    
    # Create obs (cell metadata)
    obs = pd.DataFrame(index=cell_ids)
    obs['fov'] = expr_df['fov'].values
    obs['cell_ID'] = expr_df['cell_ID'].values
    obs['sample'] = sample_name
    obs['tissue'] = 'lung'
    
    # Add all metadata columns
    metadata_cell_ids = metadata_df['fov'].astype(str) + "_" + metadata_df['cell_ID'].astype(str)
    metadata_df.index = metadata_cell_ids
    
    # Merge metadata into obs
    for col in metadata_df.columns:
        if col not in ['fov', 'cell_ID']:  # Skip duplicate columns
            obs[col] = metadata_df.loc[cell_ids, col].values
    
    # Create var (gene metadata)
    var = pd.DataFrame(index=gene_cols)
    var['gene_name'] = gene_cols
    
    # Identify negative control probes
    var['is_neg_probe'] = var.index.str.startswith('NegPrb')
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Add spatial coordinates to obsm (in pixels)
    adata.obsm['spatial_local'] = obs[['CenterX_local_px', 'CenterY_local_px']].values
    adata.obsm['spatial_global'] = obs[['CenterX_global_px', 'CenterY_global_px']].values
    
    # Add basic statistics
    adata.obs['n_counts'] = np.array(X.sum(axis=1)).flatten()
    adata.obs['n_genes'] = np.array((X > 0).sum(axis=1)).flatten()
    
    # Apply QC filters (lenient thresholds for targeted panel)
    adata = apply_qc_filters(adata, min_counts=10, min_genes=5)
    
    print(f"  Final AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    
    return adata


def main():
    """Main preprocessing pipeline"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Processing {len(SAMPLES)} samples from {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Load gene mapping once
    print("Loading gene mapping...")
    gene_mapping = load_gene_mapping()
    print(f"  Loaded {len(gene_mapping)} unique gene mappings\n")
    
    for sample_name in SAMPLES:
        print(f"Processing sample: {sample_name}")
        
        sample_dir = os.path.join(DATA_DIR, sample_name)
        
        if not os.path.exists(sample_dir):
            print(f"  WARNING: Directory not found: {sample_dir}")
            continue
        
        # Find CSV files
        csv_files = find_csv_files(sample_dir)
        
        if 'expr' not in csv_files or 'metadata' not in csv_files:
            print(f"  ERROR: Missing required CSV files")
            print(f"    Found: {csv_files.keys()}")
            continue
        
        # Create AnnData object
        adata = create_anndata(
            expr_path=csv_files['expr'],
            metadata_path=csv_files['metadata'],
            sample_name=sample_name
        )
        
        # Apply gene mapping and filter
        print(f"  Applying gene mapping...")
        n_genes_before = adata.n_vars
        adata = keep_unique_gene_ids(adata, gene_mapping)
        n_genes_after = adata.n_vars
        print(f"  Filtered genes: {n_genes_before} -> {n_genes_after} ({n_genes_before - n_genes_after} removed)")
        
        # Print summary statistics
        print_adata_summary(adata, sample_name)
        
        # Save to h5ad
        output_path = os.path.join(OUTPUT_DIR, f"{sample_name}.h5ad")
        print(f"  Saving to: {output_path}")
        adata.write_h5ad(output_path)
        
        print(f"  ✓ Completed {sample_name}\n")
    
    print("All samples processed successfully!")



if __name__ == "__main__":
    main()