# Copyright (C) 2024 Xiaohui Jiang and Jichun Xie
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
"""
Split Spatial Data into Train/Test based on HVGs
1. Load processed spatial data
2. Identify top 100 Highly Variable Genes (HVGs) per sample
3. Split:
   - Test: Top 100 HVGs
   - Train: All other genes
4. Save to new directory
"""

import os
import scanpy as sc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

INPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data"
OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split"

SAMPLES = [
    "Lung13",
    "Lung9_Rep1", 
    "Lung6",
    "Lung5_Rep1"
]

N_TOP_GENES = 50

# ---------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    for sample in SAMPLES:
        print(f"\n{'='*60}")
        print(f"Processing sample: {sample}")
        print(f"{'='*60}")
        
        input_path = os.path.join(INPUT_DIR, f"{sample}.h5ad")
        
        if not os.path.exists(input_path):
            print(f"ERROR: Input file not found: {input_path}")
            continue
            
        # Load data
        print(f"Loading {input_path}...")
        adata = sc.read_h5ad(input_path)
        print(f"  Shape: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Calculate HVGs
        # We use seurat_v3 flavor which works on count data (expects integer counts)
        # If data is not integers, we might need to check. Assuming raw counts from 01_preprocess.
        print(f"Calculating top {N_TOP_GENES} HVGs...")
        
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=N_TOP_GENES,
                flavor="seurat_v3",
                subset=False
            )
        except Exception as e:
            print(f"  Error calculating HVGs with seurat_v3: {e}")
            print("  Falling back to seurat flavor (log-normalized)...")
            # Fallback: normalize, log1p, then HVG
            adata_norm = adata.copy()
            sc.pp.normalize_total(adata_norm, target_sum=1e4)
            sc.pp.log1p(adata_norm)
            sc.pp.highly_variable_genes(
                adata_norm,
                n_top_genes=N_TOP_GENES,
                flavor="seurat",
                subset=False
            )
            # Transfer results back to original adata
            adata.var['highly_variable'] = adata_norm.var['highly_variable']
            adata.var['highly_variable_rank'] = adata_norm.var['highly_variable_rank']
            del adata_norm

        # Get HVG names
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        print(f"  Identified {len(hvg_genes)} HVGs")
        
        if len(hvg_genes) != N_TOP_GENES:
            print(f"  WARNING: Expected {N_TOP_GENES} HVGs, got {len(hvg_genes)}")
            
        # Split data
        # Test: Top 100 HVGs
        adata_test = adata[:, hvg_genes].copy()
        
        # Train: All other genes
        non_hvg_genes = adata.var_names[~adata.var['highly_variable']]
        adata_train = adata[:, non_hvg_genes].copy()
        
        print(f"  Split results:")
        print(f"    Test (HVG): {adata_test.n_vars} genes")
        print(f"    Train (Rest): {adata_train.n_vars} genes")
        print(f"    Total: {adata_test.n_vars + adata_train.n_vars} (Original: {adata.n_vars})")
        
        # Save files
        test_path = os.path.join(OUTPUT_DIR, f"test_{sample}.h5ad")
        train_path = os.path.join(OUTPUT_DIR, f"train_{sample}.h5ad")
        
        print(f"  Saving test set to: {test_path}")
        adata_test.write_h5ad(test_path)
        
        print(f"  Saving train set to: {train_path}")
        adata_train.write_h5ad(train_path)
        
        print(f"  ✓ Completed {sample}")

    print(f"\n{'='*60}")
    print("All samples processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
