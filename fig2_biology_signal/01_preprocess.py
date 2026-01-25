#!/usr/bin/env python3
# https://cellxgene.cziscience.com/collections/ff69f0ee-fef6-4895-9f48-6c64a68c8289
# - https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad
# - Single-Cell Atlas Of Human Pediatric Liver Reveals Age-related Hepatic Gene Signatures
# /hpc/group/xielab/xj58/xVerse_results/fig2/data/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad

# https://cellxgene.cziscience.com/collections/0986e4cd-7a58-405d-9b91-4b199bb4124e
# - https://datasets.cellxgene.cziscience.com/0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad
# - ALS motor cortex and spinal cord single-nucleus multiome dataset
# /hpc/group/xielab/xj58/xVerse_results/fig2/data/0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad

import os
import numpy as np
import pandas as pd
import scanpy as sc

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

data_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/data"
output_base_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2"

# Input files
liver_file = os.path.join(data_dir, "10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad")
brain_file = os.path.join(data_dir, "0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad")

# Panel metadata files
panel_dir = os.path.join(os.path.dirname(__file__), "panels")
brain_panel_file = os.path.join(panel_dir, "Xenium_hBrain_v1_metadata.csv")
multi_panel_file = os.path.join(panel_dir, "Xenium_hMulti_v1_metadata_annotations.csv")
panel_5k_file = os.path.join(panel_dir, "XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv")

# Output directories
liver_output_dir = os.path.join(output_base_dir, "liver")
brain_output_dir = os.path.join(output_base_dir, "brain")

# Create output directories
os.makedirs(liver_output_dir, exist_ok=True)
os.makedirs(brain_output_dir, exist_ok=True)

# ---------------------------------------------------------------
# Panel Loading Functions
# ---------------------------------------------------------------

def load_xenium_panel(panel_file):
    """
    Load Xenium panel CSV and extract Ensembl IDs.
    
    Parameters:
    -----------
    panel_file : str
        Path to panel CSV file
        
    Returns:
    --------
    set
        Set of Ensembl gene IDs
    """
    print(f"  Loading panel from {os.path.basename(panel_file)}...")
    df = pd.read_csv(panel_file)
    
    # Column 2 contains Ensembl IDs (either 'Ensembl_ID' or 'Ensembl ID' or 'gene_id')
    ensembl_col = df.columns[1]
    gene_ids = set(df[ensembl_col].dropna().unique())
    
    print(f"    Loaded {len(gene_ids)} unique genes from panel")
    return gene_ids


def load_5k_panel(panel_file):
    """
    Load XeniumPrimeHuman5K panel CSV and extract Ensembl IDs.
    
    Parameters:
    -----------
    panel_file : str
        Path to XeniumPrimeHuman5Kpan panel CSV file
        
    Returns:
    --------
    set
        Set of Ensembl gene IDs from 5K panel
    """
    print(f"  Loading 5K panel from {os.path.basename(panel_file)}...")
    df = pd.read_csv(panel_file)
    
    # Column 2 is 'gene_id' containing Ensembl IDs
    gene_ids = set(df['gene_id'].dropna().unique())
    
    print(f"    Loaded {len(gene_ids)} unique genes from 5K panel")
    return gene_ids


# ---------------------------------------------------------------
# Standard Scanpy Preprocessing Function
# ---------------------------------------------------------------

def standard_preprocessing(adata, min_genes=200, min_cells=3, max_pct_mt=20):
    """
    Perform basic QC filtering:
    1. Calculate QC metrics
    2. Filter cells and genes
    Note: No normalization or log-transform - keeps raw UMI counts
    """
    print(f"Initial shape: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells
    print(f"Filtering cells: min_genes={min_genes}, max_pct_mt={max_pct_mt}")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :].copy()
    
    # Filter genes
    print(f"Filtering genes: min_cells={min_cells}")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"After filtering: {adata.n_obs} cells × {adata.n_vars} genes")
    print("Keeping raw UMI counts (no normalization or log-transform)")
    
    return adata


# ---------------------------------------------------------------
# Process and Split by Donor
# ---------------------------------------------------------------

def process_and_split_by_donor(input_file, output_dir, tissue_name, donor_id_key='donor_id'):
    """
    Load h5ad file, perform standard preprocessing, add tissue annotation,
    and split by donor_id into separate files.
    
    Parameters:
    -----------
    input_file : str
        Path to input h5ad file
    output_dir : str
        Directory to save split files
    tissue_name : str
        Tissue type ('liver' or 'brain')
    donor_id_key : str
        Column name in obs containing donor IDs
    """
    print(f"\n{'='*60}")
    print(f"Processing {tissue_name.upper()} dataset")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading data from {input_file}...")
    adata = sc.read_h5ad(input_file)
    
    # Extract raw counts from adata.raw.X if available
    if adata.raw is not None:
        print("✓ Found raw counts in adata.raw, using raw.X as main matrix")
        # Create new AnnData with raw counts
        adata_raw = sc.AnnData(
            X=adata.raw.X,
            obs=adata.obs,
            var=adata.raw.var
        )
        # Copy over any useful info
        adata = adata_raw
    else:
        print("⚠ No adata.raw found, using adata.X (may be normalized)")
    
    # Add gene_ids to var
    adata.var["gene_ids"] = adata.var.index
    
    # Print obs and var columns
    print(f"\n--- obs columns ({len(adata.obs.columns)} total) ---")
    print(adata.obs.columns.tolist())
    print(f"\n--- var columns ({len(adata.var.columns)} total) ---")
    print(adata.var.columns.tolist())
    print(f"\n--- var index (first 5 genes) ---")
    print(adata.var_names[:5].tolist())
    
    # Check available layers
    print(f"\n--- Available layers ---")
    if adata.layers:
        print(f"Layers found: {list(adata.layers.keys())}")
        for layer_name in adata.layers.keys():
            print(f"  - {layer_name}: shape {adata.layers[layer_name].shape}")
    else:
        print("No layers found")
    
    print(f"\n--- Main matrix (adata.X) info ---")
    print(f"Shape: {adata.X.shape}")
    print(f"Type: {type(adata.X)}")
    if hasattr(adata.X, 'dtype'):
        print(f"Dtype: {adata.X.dtype}")
    
    # Print UMI distribution
    print(f"\n--- UMI Distribution (individual gene-cell pairs) ---")
    print(f"Data source: adata.X (raw counts)")
    
    # Get all UMI values from the matrix
    if hasattr(adata.X, 'toarray'):
        # Sparse matrix - only get non-zero values for efficiency
        umi_values = adata.X.data  # Non-zero values only
        print(f"Total matrix elements: {adata.X.shape[0] * adata.X.shape[1]:,}")
        print(f"Non-zero elements: {len(umi_values):,}")
        print(f"Zero elements: {adata.X.shape[0] * adata.X.shape[1] - len(umi_values):,}")
        print(f"Sparsity: {(1 - len(umi_values)/(adata.X.shape[0] * adata.X.shape[1]))*100:.2f}%")
    else:
        # Dense matrix
        umi_values = adata.X.flatten()
        umi_values = umi_values[umi_values > 0]  # Only non-zero
        print(f"Total matrix elements: {adata.X.size:,}")
    
    print(f"\nUMI values per gene-cell pair (non-zero only):")
    print(f"  Mean: {umi_values.mean():.4f}")
    print(f"  Median: {np.median(umi_values):.4f}")
    print(f"  Min: {umi_values.min():.0f}")
    print(f"  Max: {umi_values.max():.0f}")
    print(f"  Std: {umi_values.std():.4f}")
    print(f"  Percentiles [25%, 50%, 75%, 90%, 95%, 99%]: {np.percentile(umi_values, [25, 50, 75, 90, 95, 99])}")
    
    # Count distribution
    unique, counts = np.unique(umi_values, return_counts=True)
    print(f"\nMost common UMI values:")
    top_indices = np.argsort(counts)[-10:][::-1]  # Top 10
    for idx in top_indices:
        val = unique[idx]
        cnt = counts[idx]
        pct = cnt / len(umi_values) * 100
        print(f"  UMI={val:.0f}: {cnt:,} occurrences ({pct:.2f}%)")
    
    print()
    
    # Add tissue annotation
    print(f"Adding tissue annotation: {tissue_name}")
    adata.obs['tissue'] = tissue_name
    
    # Perform standard preprocessing
    adata = standard_preprocessing(adata)
    
    # Check if donor_id exists
    if donor_id_key not in adata.obs.columns:
        print(f"Warning: '{donor_id_key}' not found in obs. Available columns:")
        print(adata.obs.columns.tolist())
        # Try to find alternative donor ID column
        possible_keys = [col for col in adata.obs.columns if 'donor' in col.lower()]
        if possible_keys:
            donor_id_key = possible_keys[0]
            print(f"Using '{donor_id_key}' instead")
        else:
            raise ValueError(f"No donor ID column found in obs")
    
    # Get unique donor IDs
    donor_ids = adata.obs[donor_id_key].unique()
    print(f"\nFound {len(donor_ids)} unique donors: {sorted(donor_ids)}")
    
    # Determine which panel to use for this tissue
    if tissue_name == 'brain':
        panel_file = brain_panel_file
    elif tissue_name == 'liver':
        panel_file = multi_panel_file
    else:
        raise ValueError(f"Unknown tissue: {tissue_name}")
    
    # Load Xenium panel genes
    print(f"\nLoading Xenium panel for {tissue_name}...")
    xenium_genes = load_xenium_panel(panel_file)
    
    # Load 5K panel genes
    print(f"\nLoading 5K panel...")
    panel_5k_genes = load_5k_panel(panel_5k_file)
    
    # Split by donor and save 3 versions each
    print(f"\nSplitting by donor and saving 3 versions (all, 5k, xenium)...")
    for donor_id in donor_ids:
        print(f"\n{'='*60}")
        print(f"Processing donor: {donor_id}")
        print(f"{'='*60}")
        
        # Subset data for this donor
        adata_donor = adata[adata.obs[donor_id_key] == donor_id].copy()
        print(f"  Donor has {adata_donor.n_obs} cells × {adata_donor.n_vars} genes")
        
        # Create safe filename (replace special characters)
        safe_donor_id = str(donor_id).replace('/', '_').replace(' ', '_')
        
        # 1. Save ALL genes version
        output_file_all = os.path.join(output_dir, f"{tissue_name}_{safe_donor_id}_all.h5ad")
        print(f"  [1/3] Saving ALL genes: {adata_donor.n_vars} genes → {output_file_all}")
        adata_donor.write(output_file_all)
        
        # 2. Save 5K panel version
        # Filter to genes that exist in the data
        panel_5k_genes_available = [g for g in adata_donor.var_names if g in panel_5k_genes]
        adata_donor_5k = adata_donor[:, panel_5k_genes_available].copy()
        
        output_file_5k = os.path.join(output_dir, f"{tissue_name}_{safe_donor_id}_5k.h5ad")
        print(f"  [2/3] Saving 5K panel: {adata_donor_5k.n_vars} genes → {output_file_5k}")
        adata_donor_5k.write(output_file_5k)
        
        # 3. Save Xenium panel version
        # Filter to genes that exist in both panel and data
        xenium_genes_available = [g for g in adata_donor.var_names if g in xenium_genes]
        adata_donor_xenium = adata_donor[:, xenium_genes_available].copy()
        
        output_file_xenium = os.path.join(output_dir, f"{tissue_name}_{safe_donor_id}_xenium.h5ad")
        print(f"  [3/3] Saving Xenium panel: {adata_donor_xenium.n_vars} genes → {output_file_xenium}")
        adata_donor_xenium.write(output_file_xenium)
        
        print(f"  ✓ Donor {donor_id} complete: 3 versions saved")

    
    print(f"\n{tissue_name.upper()} processing complete!")
    return adata


# ---------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Process liver dataset
    adata_liver = process_and_split_by_donor(
        input_file=liver_file,
        output_dir=liver_output_dir,
        tissue_name='liver'
    )
    
    # Process brain dataset
    adata_brain = process_and_split_by_donor(
        input_file=brain_file,
        output_dir=brain_output_dir,
        tissue_name='brain'
    )
    
    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")
    print(f"Liver files saved to: {liver_output_dir}")
    print(f"Brain files saved to: {brain_output_dir}")
