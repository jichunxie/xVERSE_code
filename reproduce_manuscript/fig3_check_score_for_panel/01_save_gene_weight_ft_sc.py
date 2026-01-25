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

# -*- coding: utf-8 -*-
"""
Extract and save gene weights (attention scores) from the fine-tuned XVerse model (Figure 3 - Breast).
"""

import os
import torch
import numpy as np
import pandas as pd
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

# Output Directory
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

    # Note: We are loading the fine-tuned state dict later, so we just need the architecture here.
    # However, to be safe and consistent with training script:
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


def save_gene_weights(model, loader, gene_ids, save_name):
    """
    Run inference and compute mean gene weights (attention scores).
    """
    print(f"Computing gene weights for {save_name}...")
    model.eval()
    
    total_weights = None
    total_cells = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            # Unpack batch (sample_id, value, tissue_id)
            # Note: XVerseFineTuneDataset returns (sample_id, value, tissue_id)
            sample_id, value, tissue_id = batch
            
            sample_id = sample_id.to(device)
            tissue_id = tissue_id.to(device)
            value = value.to(device)
            
            # Forward pass
            # model() returns a dictionary: {"mu_bio": ..., "z_bio": ..., "gene_weight": ..., "mu": ...}
            outputs = model(
                value=value, 
                sample_id=sample_id, 
                tissue_id=tissue_id
            )
            gene_weight = outputs["gene_weight"]
            
            # gene_weight shape: [Batch, n_genes]
            
            # Accumulate sum
            if total_weights is None:
                total_weights = torch.zeros(gene_weight.shape[1], device=device)
            
            total_weights += gene_weight.sum(dim=0)
            total_cells += value.size(0)
            
    # Compute mean
    mean_weights = total_weights / total_cells
    mean_weights_np = mean_weights.cpu().numpy()
    
    # Save results
    df = pd.DataFrame({
        "gene_id": gene_ids,
        "mean_weight": mean_weights_np
    })
    
    # Filter out genes with 0 mean weight (not in panel)
    df = df[df["mean_weight"] > 0]

    # Sort by weight descending
    df = df.sort_values(by="mean_weight", ascending=False)
    
    csv_path = os.path.join(output_dir, f"{save_name}_mean_gene_weights.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved mean gene weights to {csv_path}")
    
    return df


# ==============================================================  
# Main  
# ==============================================================  

def main():
    print(f"Using device: {device}")

    # 1. Load Gene IDs
    print("Loading gene IDs...")
    gene_ids = load_gene_ids(gene_ids_path)
    
    # 2. Build Dataset
    print("Building full dataset...")
    # Filter out missing files to avoid errors
    valid_file_sample_dict = {p: i for p, i in file_sample_dict.items() if os.path.exists(p)}
    if len(valid_file_sample_dict) < len(file_sample_dict):
        print(f"Warning: Only {len(valid_file_sample_dict)}/{len(file_sample_dict)} files exist.")
        
    # 3. Build Model
    model = build_model(
        base_ckpt_path=base_model_ckpt,
        num_samples_finetune=len(file_sample_dict),
        hidden_dim=384,
    )

    # 4. Load Fine-Tuned Weights
    if os.path.exists(ckpt_path):
        print(f"Loading fine-tuned weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    # 5. Compute and Save Weights
    print("Computing weights per sample...")
    
    for path, sample_id in valid_file_sample_dict.items():
        sample_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\nProcessing {sample_name} (ID: {sample_id})...")
        
        # Create dataset for single sample
        dataset = XVerseFineTuneDataset(
            {path: sample_id},
            gene_ids,
            tissue_map,
            use_qc=False,
        )
        
        if len(dataset) == 0:
            print(f"Skipping empty dataset: {sample_name}")
            continue
            
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        save_gene_weights(model, loader, gene_ids, save_name=f"sc_{sample_name}")
    
    print("Done.")


if __name__ == "__main__":
    main()
