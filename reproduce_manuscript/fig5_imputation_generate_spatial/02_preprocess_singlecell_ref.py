#!/usr/bin/env python3
"""
Preprocess Lung Single-Cell Reference Data for Fig 4
1. Load dataset
2. Filter for NSCLC (non-small cell lung carcinoma)
3. Apply standard QC
4. Split by donor and save
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Input file path (User provided)
INPUT_FILE = "/hpc/group/xielab/xj58/xVerse_results/fig5/data/c1870f1f-ca36-4d96-b03b-7dc0e96d83ee.h5ad"

# Output directory
OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc"

# QC Parameters (from fig2/01_preprocess.py)
MIN_GENES = 200
MIN_CELLS = 3
MAX_PCT_MT = 20

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def standard_preprocessing(adata, min_genes=200, min_cells=3, max_pct_mt=20):
    """
    Perform basic QC filtering:
    1. Calculate QC metrics
    2. Filter cells and genes
    Note: No normalization or log-transform - keeps raw UMI counts
    """
    print(f"  Initial shape: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Calculate QC metrics
    # Identify mitochondrial genes (start with MT-)
    adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells
    print(f"  Filtering cells: min_genes={min_genes}, max_pct_mt={max_pct_mt}")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :].copy()
    
    # Filter genes
    print(f"  Filtering genes: min_cells={min_cells}")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"  After filtering: {adata.n_obs} cells × {adata.n_vars} genes")
    
    return adata

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # 1. Load Data
    print(f"\n{'='*60}")
    print(f"Loading data from: {INPUT_FILE}")
    print(f"{'='*60}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    adata = sc.read_h5ad(INPUT_FILE)
    print(f"Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check for raw counts
    if adata.raw is not None:
        print("✓ Found raw counts in adata.raw, using raw.X as main matrix")
        adata_raw = sc.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)
        adata = adata_raw
    else:
        print("⚠ No adata.raw found, using adata.X (may be normalized)")

    # 2. Filter for NSCLC
    print(f"\n{'='*60}")
    print("Filtering for NSCLC")
    print(f"{'='*60}")
    
    if 'disease' not in adata.obs.columns:
        print("ERROR: 'disease' column not found in obs")
        print(f"Available columns: {adata.obs.columns.tolist()}")
        return

    target_disease = "non-small cell lung carcinoma"
    print(f"Target disease: '{target_disease}'")
    
    # Case-insensitive matching
    mask = adata.obs['disease'].astype(str).str.lower() == target_disease.lower()
    n_match = mask.sum()
    
    if n_match == 0:
        print(f"WARNING: No cells found matching '{target_disease}'")
        print("Available diseases:")
        print(adata.obs['disease'].value_counts())
        return
        
    adata = adata[mask].copy()
    print(f"Kept {adata.n_obs} cells matching disease")

    # 3. Apply QC
    print(f"\n{'='*60}")
    print("Applying QC")
    print(f"{'='*60}")
    adata = standard_preprocessing(adata, min_genes=MIN_GENES, min_cells=MIN_CELLS, max_pct_mt=MAX_PCT_MT)

    # 4. Set Gene IDs
    print(f"\n{'='*60}")
    print("Setting Gene IDs")
    print(f"{'='*60}")
    
    # User specified that var.index are the IDs
    adata.var["gene_ids"] = adata.var.index
    print("Set adata.var['gene_ids'] = adata.var.index")

    # 5. Split by Donor
    print(f"\n{'='*60}")
    print("Splitting by Donor")
    print(f"{'='*60}")
    
    if 'donor_id' not in adata.obs.columns:
        print("ERROR: 'donor_id' column not found in obs")
        return

    donor_ids = adata.obs['donor_id'].unique()
    print(f"Found {len(donor_ids)} unique donors: {sorted(donor_ids)}")
    
    for donor in donor_ids:
        # Subset
        adata_donor = adata[adata.obs['donor_id'] == donor].copy()
        
        # Sanitize filename
        safe_donor = str(donor).replace('/', '_').replace(' ', '_')
        filename = f"nsclc_{safe_donor}.h5ad"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"  Saving donor {donor}: {adata_donor.n_obs} cells -> {filename}")
        adata_donor.write_h5ad(output_path)

    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
