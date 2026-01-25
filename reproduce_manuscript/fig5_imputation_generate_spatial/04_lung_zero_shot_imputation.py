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

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import scanpy as sc
import pandas as pd
from scipy.stats import pearsonr

# ==============================================================
# 1. Reproducibility
# ==============================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==============================================================
# 2. Import custom modules
# ==============================================================
from main.utils_ft import (
    XVerseFineTuneModel,
    XVerseFineTuneDataset,
    save_poisson_parameter_ft,
)
from main.utils_model import load_gene_ids, XVerseModel

# ==============================================================
# 3. Paths and settings
# ==============================================================
tissue_map = {'lung': 32} 

# Updated paths based on fig4 reference
base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

gene_ids = load_gene_ids(gene_ids_path)

output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results/zero_shot"
os.makedirs(output_dir, exist_ok=True)

# Define samples and their paths
samples = [
    "Lung13",
    "Lung9_Rep1", 
    "Lung6",
    "Lung5_Rep1"
]

data_base_dir = "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split"
sample_id_map = {}

for s in samples:
    train_path = os.path.join(data_base_dir, f"train_{s}.h5ad")
    sample_id_map[train_path] = None

# ==============================================================
# 4. Load model
# ==============================================================
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize Base Model
base_model = XVerseModel(num_samples=None, hidden_dim=384).to(device)

# 2. Load Pretrained Weights
if os.path.exists(base_model_ckpt):
    checkpoint = torch.load(base_model_ckpt, map_location=device)
    # Handle potential "model_state_dict" wrapping and "module." prefix
    state = checkpoint.get("model_state_dict", checkpoint)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state.items()}
    base_model.load_state_dict(clean_state_dict, strict=False)
    print("Base model weights loaded.")
else:
    print(f"WARNING: Model checkpoint not found at {base_model_ckpt}")

# 3. Wrap with FineTune Model (Zero-Shot Mode)
# num_samples_finetune=None disables sample-specific parameters
model = XVerseFineTuneModel(base_model=base_model, num_samples_finetune=None).to(device)
model.eval()
print("Model ready for zero-shot inference.")

# ==============================================================
# 5. Evaluation Function
# ==============================================================
def align_and_evaluate(test_path, mu_path):
    """
    Aligns test data and imputed data by gene IDs, calculates Pearson correlation
    per gene, and returns the correlations.
    """
    print(f"Loading test data: {test_path}")
    adata_test = sc.read_h5ad(test_path)
    
    print(f"Loading imputed data: {mu_path}")
    adata_mu = sc.read_h5ad(mu_path)
    
    # Ensure gene_ids are available in var
    if "gene_ids" not in adata_test.var.columns:
        adata_test.var["gene_ids"] = adata_test.var.index
    if "gene_ids" not in adata_mu.var.columns:
        adata_mu.var["gene_ids"] = adata_mu.var.index
        
    # Find common genes
    genes_test = set(adata_test.var["gene_ids"])
    genes_mu = set(adata_mu.var["gene_ids"])
    common_genes = sorted(list(genes_test.intersection(genes_mu)))
    
    print(f"Found {len(common_genes)} common genes.")
    
    if len(common_genes) == 0:
        return None

    # Subset and align
    # Create index maps for fast lookup
    test_gene_to_idx = {g: i for i, g in enumerate(adata_test.var["gene_ids"])}
    mu_gene_to_idx = {g: i for i, g in enumerate(adata_mu.var["gene_ids"])}
    
    test_indices = [test_gene_to_idx[g] for g in common_genes]
    mu_indices = [mu_gene_to_idx[g] for g in common_genes]
    
    X_test = adata_test.X[:, test_indices]
    X_mu = adata_mu.X[:, mu_indices]
    
    # Convert to dense if sparse
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    if hasattr(X_mu, "toarray"):
        X_mu = X_mu.toarray()
        
    # Calculate Pearson correlation per gene
    correlations = {}
    for i, gene in enumerate(common_genes):
        vec_test = X_test[:, i]
        vec_mu = X_mu[:, i]
        
        # Avoid correlation of constant vectors
        if np.std(vec_test) == 0 or np.std(vec_mu) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(vec_test, vec_mu)
            
        correlations[gene] = corr
        
    return correlations

# ==============================================================
# 6. Inference and Evaluation Loop
# ==============================================================
batch_size = 64
num_workers = 4

print("\n=== Starting Inference and Evaluation ===")

for file_path, sample_id in sample_id_map.items():
    file_name = os.path.basename(file_path)
    sample_name = os.path.splitext(file_name.replace("train_", ""))[0]
    
    if not os.path.exists(file_path):
        print(f"Skipping missing file: {file_path}")
        continue
        
    print(f"\n=== Processing sample {sample_name} ===")
    
    # Create Dataset
    dataset = XVerseFineTuneDataset(
        {file_path: sample_id},
        gene_ids,
        tissue_map,
        use_qc=False
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Load test data to get gene subset
    # Fix: Only replace in filename to avoid modifying directory path 'train_test_split'
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    test_file_name = file_name.replace("train_", "test_")
    test_path = os.path.join(dir_name, test_file_name)
    
    gene_id_subset = None
    if os.path.exists(test_path):
        print(f"Loading test data to filter genes: {test_path}")
        adata_test = sc.read_h5ad(test_path)
        if "gene_ids" in adata_test.var.columns:
            gene_id_subset = adata_test.var["gene_ids"].tolist()
        else:
            gene_id_subset = adata_test.var.index.tolist()
        print(f"Filtering output to {len(gene_id_subset)} genes from test set.")
    else:
        print(f"WARNING: Test file not found at {test_path}. Saving all genes.")

    # Run Inference
    # This will save {sample_name}_mu_bio.h5ad
    save_poisson_parameter_ft(
        model=model,
        loader=loader,
        sample_name=sample_name,
        gene_ids=gene_ids,
        output_dir=output_dir,
        device=device,
        use_amp=torch.cuda.is_available(),
        estimate_genes=False, 
        save_panel_genes_only=False,
        gene_id_subset=gene_id_subset
    )
    
    # Immediate Evaluation
    mu_path = os.path.join(output_dir, f"{sample_name}_mu_bio.h5ad")
    
    if os.path.exists(test_path) and os.path.exists(mu_path):
        print(f"Evaluating {sample_name}...")
        try:
            corrs = align_and_evaluate(test_path, mu_path)
            
            if corrs:
                # Calculate and print quartiles
                all_corrs = list(corrs.values())
                quartiles = np.percentile(all_corrs, [0, 25, 50, 75, 100])
                print(f"Pearson Correlation Quartiles for {sample_name}:")
                print(f"  Min (0%):     {quartiles[0]:.4f}")
                print(f"  25%:          {quartiles[1]:.4f}")
                print(f"  Median (50%): {quartiles[2]:.4f}")
                print(f"  75%:          {quartiles[3]:.4f}")
                print(f"  Max (100%):   {quartiles[4]:.4f}")

                # Sort by correlation descending
                sorted_corrs = sorted(corrs.items(), key=lambda item: item[1], reverse=True)
                
                print(f"Top 10 genes with highest Pearson correlation for {sample_name}:")
                for gene, corr in sorted_corrs[:10]:
                    print(f"  {gene}: {corr:.4f}")
                    
                # Optional: Save all correlations
                corr_df = pd.DataFrame(sorted_corrs, columns=["gene_id", "pearson_corr"])
                save_path = os.path.join(output_dir, f"{sample_name}_gene_correlations.csv")
                corr_df.to_csv(save_path, index=False)
                print(f"Saved all correlations to {save_path}")
                
        except Exception as e:
            print(f"Error evaluating {sample_name}: {e}")
    else:
        print(f"Skipping evaluation for {sample_name} due to missing files.")

print("\n=== All Done ===")
