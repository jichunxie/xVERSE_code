# -*- coding: utf-8 -*-
"""
Evaluate gene importance by comparing imputation accuracy (Pearson Correlation) 
of High vs Low scoring genes (Figure 4 - Lung Spatial).
"""

import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr

from main.utils_ft import (
    XVerseFineTuneModel,
    XVerseFineTuneDataset,
)
from main.utils_model import load_gene_ids, XVerseModel


# ==============================================================  
# Configurations (Matched to fig5/05_lung_ft_imputation.py)
# ==============================================================  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tissue_map = {'lung': 32}

# Paths
base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

# Single Cell Reference Paths (Needed for sample ID consistency)
SC_PATHS = {
    "nsclc_Wu_Zhou_2021_P11": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Wu_Zhou_2021_P11.h5ad",
    "nsclc_Lambrechts_Thienpont_2018_6653_8": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Lambrechts_Thienpont_2018_6653_8.h5ad",
    "nsclc_Chen_Zhang_2020_NSCLC-1": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Chen_Zhang_2020_NSCLC-1.h5ad",
    "nsclc_Chen_Zhang_2020_NSCLC-2": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Chen_Zhang_2020_NSCLC-2.h5ad",
    "nsclc_Chen_Zhang_2020_NSCLC-3": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Chen_Zhang_2020_NSCLC-3.h5ad",
}

# Spatial Samples
ST_SAMPLES = [
    "Lung13",
    "Lung9_Rep1", 
    "Lung6",
    "Lung5_Rep1"
]
ST_BASE_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split"

# Construct File-Sample Dictionary
file_sample_dict = {}
sample_idx = 0

# Add SC files
for name, path in SC_PATHS.items():
    file_sample_dict[path] = sample_idx
    sample_idx += 1

# Add Spatial files
for s in ST_SAMPLES:
    path = os.path.join(ST_BASE_DIR, f"train_{s}.h5ad")
    file_sample_dict[path] = sample_idx
    sample_idx += 1

# Model Checkpoint to Load
ft_model_dir = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results/ft_model"
model_label = "lung_full_ft"
ckpt_path = os.path.join(ft_model_dir, f"best_model_{model_label}.pth")

# Output Directory (Same as where weights are saved)
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig3/gene_weights"
os.makedirs(output_dir, exist_ok=True)

batch_size = 256
num_workers = 16


# ==============================================================  
# Utility  
# ==============================================================  

def build_model(base_ckpt_path, num_samples_finetune, hidden_dim=384):
    """Load pretrained XVerse and wrap with fine-tune head."""
    print("Initializing base model...")
    base_model = XVerseModel(num_samples=None, hidden_dim=hidden_dim).to(device)

    if os.path.exists(base_ckpt_path):
        ckpt = torch.load(base_ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        base_model.load_state_dict(state, strict=False)
    
    model = XVerseFineTuneModel(
        base_model=base_model,
        num_samples_finetune=num_samples_finetune,
    ).to(device)
    return model


def get_imputed_expression(model, loader):
    """
    Run inference and return mu_bio (imputed expression).
    """
    model.eval()
    mu_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Imputing expression", leave=False):
            # Unpack batch (sample_id, value, tissue_id)
            sample_id, value, tissue_id = batch
            
            sample_id = sample_id.to(device)
            tissue_id = tissue_id.to(device)
            value = value.to(device)
            
            # Forward pass
            outputs = model(
                value=value, 
                sample_id=sample_id, 
                tissue_id=tissue_id
            )
            mu_bio = outputs["mu_bio"]
            mu_list.append(mu_bio.cpu().numpy())
            
    return np.concatenate(mu_list, axis=0)


def evaluate_imputation(test_path, imputed_matrix, gene_ids):
    """
    Compute Pearson correlation between imputed data and test data.
    """
    print(f"Loading test data: {test_path}")
    adata_test = sc.read_h5ad(test_path)
    
    # Ensure gene_ids are available in var
    if "gene_ids" not in adata_test.var.columns:
        adata_test.var["gene_ids"] = adata_test.var.index
        
    # Find common genes
    genes_test = set(adata_test.var["gene_ids"])
    genes_model = set(gene_ids)
    common_genes = sorted(list(genes_test.intersection(genes_model)))
    
    if len(common_genes) == 0:
        return []

    # Map genes to indices
    test_gene_to_idx = {g: i for i, g in enumerate(adata_test.var["gene_ids"])}
    model_gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    
    test_indices = [test_gene_to_idx[g] for g in common_genes]
    model_indices = [model_gene_to_idx[g] for g in common_genes]
    
    X_test = adata_test.X[:, test_indices]
    X_imputed = imputed_matrix[:, model_indices]
    
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
        
    correlations = []
    for i in range(len(common_genes)):
        vec_test = X_test[:, i]
        vec_imp = X_imputed[:, i]
        
        if np.std(vec_test) == 0 or np.std(vec_imp) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(vec_test, vec_imp)
        correlations.append(corr)
        
    return correlations


# ==============================================================  
# Main  
# ==============================================================  

def main():
    print(f"Using device: {device}")

    # 1. Load Gene IDs
    print("Loading gene IDs...")
    gene_ids = load_gene_ids(gene_ids_path)
    
    # 2. Build Model
    model = build_model(
        base_ckpt_path=base_model_ckpt,
        num_samples_finetune=sample_idx,
        hidden_dim=384,
    )

    # 3. Load Fine-Tuned Weights
    if os.path.exists(ckpt_path):
        print(f"Loading fine-tuned weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    # 4. Process Each Spatial Sample
    results = []
    
    # Identify spatial paths
    spatial_paths = [p for p in file_sample_dict.keys() if ST_BASE_DIR in p]
    valid_spatial_dict = {p: file_sample_dict[p] for p in spatial_paths if os.path.exists(p)}
    
    for path, sample_id in valid_spatial_dict.items():
        sample_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'='*40}")
        print(f"Processing {sample_name} (ID: {sample_id})")
        print(f"{'='*40}")
        
        # Load weights
        weight_file = os.path.join(output_dir, f"spatial_{sample_name}_mean_gene_weights.csv")
        if not os.path.exists(weight_file):
            print(f"Weight file not found: {weight_file}. Skipping.")
            continue
            
        df_weights = pd.read_csv(weight_file)
        df_weights = df_weights.sort_values(by="mean_weight", ascending=False)
        df_weights = df_weights[df_weights["mean_weight"] > 0]
        
        total_genes = len(df_weights)
        if total_genes < 5:
            print(f"Not enough genes ({total_genes}) to split into 5 bins. Skipping.")
            continue
            
        # Determine Test Path
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path)
        test_file_name = file_name.replace("train_", "test_")
        test_path = os.path.join(dir_name, test_file_name)
        
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}. Skipping.")
            continue

        # Iterate over different bin counts
        # 1 bin = Full Panel (Benchmark)
        n_bins_list = [1, 2, 3, 4]
        
        for n_bins in n_bins_list:
            print(f"  Evaluating with {n_bins} bins...")
            chunk_size = total_genes // n_bins
            
            for i in range(n_bins):
                start_idx = i * chunk_size
                # For the last bin, take all remaining genes
                end_idx = (i + 1) * chunk_size if i < n_bins - 1 else total_genes
                
                bin_genes = df_weights.iloc[start_idx:end_idx]["gene_id"].tolist()
                # print(f"    Bin {i+1}/{n_bins}: {len(bin_genes)} genes")
                
                # Create dataset with ONLY these genes visible
                dataset = XVerseFineTuneDataset(
                    {path: sample_id},
                    gene_ids,
                    tissue_map,
                    visible_gene_ids=bin_genes,
                    use_qc=False,
                )
                
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                
                # Get imputed expression
                imputed_matrix = get_imputed_expression(model, loader)
                
                # Evaluate
                corrs = evaluate_imputation(test_path, imputed_matrix, gene_ids)
                
                if corrs:
                    mean_corr = np.mean(corrs)
                    print(f"    [Bins={n_bins} | Bin {i+1}] Mean Pearson Corr: {mean_corr:.4f}")
                    
                    results.append({
                        "Sample": sample_name,
                        "NumBins": n_bins,
                        "Bin": i + 1,
                        "NumGenes": len(bin_genes),
                        "MeanCorr": mean_corr
                    })
                else:
                    print(f"    [Bins={n_bins} | Bin {i+1}] No common genes found.")

    # Save Results
    if results:
        res_df = pd.DataFrame(results)
        save_path = os.path.join(output_dir, "spatial_gene_importance_corr.csv")
        res_df.to_csv(save_path, index=False)
        print(f"\nSaved results to {save_path}")
        print(res_df)
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
