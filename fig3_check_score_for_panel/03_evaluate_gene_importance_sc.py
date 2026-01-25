# -*- coding: utf-8 -*-
"""
Evaluate gene importance by calculating ASW for gene score bins
"""

import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scib
from torch.utils.data import DataLoader
from tqdm import tqdm

from main.utils_ft import (
    XVerseFineTuneModel,
    XVerseFineTuneDataset,
)
from main.utils_model import load_gene_ids, XVerseModel


# ==============================================================  
# Configurations (Matched to fig4/02_train_xverse_full.py)
# ==============================================================  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tissue_map = {"breast": 8}

# Paths
base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

# Data Dictionary
file_sample_dict = {
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-A.h5ad": 0,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-B.h5ad": 1,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-C.h5ad": 2,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-D.h5ad": 3,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-E.h5ad": 4,
}

# Model Checkpoint to Load
ft_model_dir = "/hpc/group/xielab/xj58/xVerse_results/fig4/ft_model"
model_label = "full"
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


def get_embeddings(model, loader):
    """
    Run inference and return z_bio embeddings.
    """
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
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
            z_bio = outputs["z_bio"]
            embeddings_list.append(z_bio.cpu().numpy())
            
    return np.concatenate(embeddings_list, axis=0)


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
        num_samples_finetune=len(file_sample_dict),
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

    # 4. Process Each Sample
    results = []
    
    # Filter out missing files
    valid_file_sample_dict = {p: i for p, i in file_sample_dict.items() if os.path.exists(p)}
    
    for path, sample_id in valid_file_sample_dict.items():
        sample_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'='*40}")
        print(f"Processing {sample_name} (ID: {sample_id})")
        print(f"{'='*40}")
        
        # Load weights
        weight_file = os.path.join(output_dir, f"sc_{sample_name}_mean_gene_weights.csv")
        if not os.path.exists(weight_file):
            print(f"Weight file not found: {weight_file}. Skipping.")
            continue
            
        df_weights = pd.read_csv(weight_file)
        # Ensure sorted descending
        df_weights = df_weights.sort_values(by="mean_weight", ascending=False)
        
        # Filter out zero weights (should already be done, but safe check)
        df_weights = df_weights[df_weights["mean_weight"] > 0]
        
        total_genes = len(df_weights)
        if total_genes < 5:
            print(f"Not enough genes ({total_genes}) to split into 5 bins. Skipping.")
            continue
            
        # Load Labels (Major.subtype)
        print(f"Loading labels from {path}...")
        adata = sc.read_h5ad(path)
        if "Major.subtype" not in adata.obs.columns:
            print("Error: 'Major.subtype' not found in obs. Skipping.")
            continue
        labels = adata.obs["Major.subtype"].values
        
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
                    visible_gene_ids=bin_genes, # This masks other genes as -1
                    use_qc=False,
                )
                
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                
                # Get embeddings
                embeddings = get_embeddings(model, loader)
                
                # Compute ASW using scib
                if len(np.unique(labels)) > 1:
                    # Create temp AnnData for scib
                    adata_temp = sc.AnnData(X=embeddings)
                    adata_temp.obs['label'] = labels
                    adata_temp.obsm['X_emb'] = embeddings
                    
                    # scib.metrics.silhouette expects: adata, group_key, embed
                    asw = scib.metrics.silhouette(
                        adata_temp, 
                        group_key='label', 
                        embed='X_emb'
                    )
                    print(f"    [Bins={n_bins} | Bin {i+1}] ASW: {asw:.4f}")
                else:
                    asw = np.nan
                    print(f"    [Bins={n_bins} | Bin {i+1}] ASW: NaN (only 1 label)")
                    
                results.append({
                    "Sample": sample_name,
                    "NumBins": n_bins,
                    "Bin": i + 1,
                    "NumGenes": len(bin_genes),
                    "ASW": asw
                })
            
    # Save Results
    if results:
        res_df = pd.DataFrame(results)
        save_path = os.path.join(output_dir, "sc_gene_importance_asw.csv")
        res_df.to_csv(save_path, index=False)
        print(f"\nSaved results to {save_path}")
        print(res_df)
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
