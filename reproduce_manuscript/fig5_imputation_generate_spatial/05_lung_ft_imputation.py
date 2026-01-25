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
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    finetune_one_epoch,
    evaluate_one_epoch,
    save_poisson_parameter_ft,
    BalancedSampler,
)
from main.utils_model import load_gene_ids, XVerseModel

# ==============================================================
# 3. Paths and settings
# ==============================================================
tissue_map = {'lung': 32} 

base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results/ft_model"
os.makedirs(output_dir, exist_ok=True)

# Single Cell Reference Paths
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

file_sample_dict = {}
spatial_file_map = {} # Keep track for inference later
sample_idx = 0

# Add SC files
for name, path in SC_PATHS.items():
    # We add it even if it doesn't exist locally, assuming it exists on HPC
    file_sample_dict[path] = sample_idx
    sample_idx += 1

# Add Spatial files
for s in ST_SAMPLES:
    path = os.path.join(ST_BASE_DIR, f"train_{s}.h5ad")
    file_sample_dict[path] = sample_idx
    spatial_file_map[s] = (path, sample_idx)
    sample_idx += 1

# Training Hyperparameters
max_epochs = 500
batch_size = 128
num_workers = 16
learning_rate = 1e-4
lambda_recon_bio = 1
train_frac = 0.9
model_label = "lung_full_ft"

# ==============================================================
# 4. Helper Functions
# ==============================================================

def build_model(base_ckpt_path, num_samples_finetune, hidden_dim=384, device='cpu'):
    """Load pretrained XVerse and wrap with fine-tune head."""
    print("Initializing base model...")
    base_model = XVerseModel(num_samples=None, hidden_dim=hidden_dim).to(device)

    if os.path.exists(base_ckpt_path):
        print(f"Loading weights from {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        base_model.load_state_dict(state, strict=False)
    else:
        print(f"WARNING: Checkpoint not found at {base_ckpt_path}")

    print(f"Wrapping with FineTuneModel")
    model = XVerseFineTuneModel(
        base_model=base_model,
        num_samples_finetune=num_samples_finetune,
    ).to(device)
    return model

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
    test_gene_to_idx = {g: i for i, g in enumerate(adata_test.var["gene_ids"])}
    mu_gene_to_idx = {g: i for i, g in enumerate(adata_mu.var["gene_ids"])}
    
    test_indices = [test_gene_to_idx[g] for g in common_genes]
    mu_indices = [mu_gene_to_idx[g] for g in common_genes]
    
    X_test = adata_test.X[:, test_indices]
    X_mu = adata_mu.X[:, mu_indices]
    
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    if hasattr(X_mu, "toarray"):
        X_mu = X_mu.toarray()
        
    correlations = {}
    for i, gene in enumerate(common_genes):
        vec_test = X_test[:, i]
        vec_mu = X_mu[:, i]
        
        if np.std(vec_test) == 0 or np.std(vec_mu) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(vec_test, vec_mu)
            
        correlations[gene] = corr
        
    return correlations

# ==============================================================
# 5. Main Execution
# ==============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Gene IDs
    print("Loading gene IDs...")
    gene_ids = load_gene_ids(gene_ids_path)

    # 2. Print Sample Mapping (already constructed)
    print(f"Total samples for fine-tuning: {len(file_sample_dict)}")
    for path, idx in file_sample_dict.items():
        print(f"  ID {idx}: {os.path.basename(path)}")
        # Check existence here to warn user
        if not os.path.exists(path):
             print(f"  [WARNING] File not found: {path}")

    # 3. Build Full Dataset
    print("Building full dataset...")
    # Filter out missing files to avoid errors during dataset construction
    valid_file_sample_dict = {p: i for p, i in file_sample_dict.items() if os.path.exists(p)}
    
    if len(valid_file_sample_dict) < len(file_sample_dict):
        print(f"Warning: Only {len(valid_file_sample_dict)}/{len(file_sample_dict)} files exist.")
    
    full_dataset = XVerseFineTuneDataset(
        valid_file_sample_dict,
        gene_ids,
        tissue_map,
        use_qc=False,
    )
    n_total = len(full_dataset)
    print(f"Total cells = {n_total}")

    # 4. Split Train/Valid
    n_train = int(n_total * train_frac)
    n_valid = n_total - n_train
    train_dataset, valid_dataset = random_split(
        full_dataset,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train size: {n_train} | Valid size: {n_valid}")

    # 5. Build Model
    model = build_model(
        base_ckpt_path=base_model_ckpt,
        num_samples_finetune=sample_idx, 
        hidden_dim=384,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # 6. Training Loop
    best_val = float("inf")
    no_improve = 0
    early_stop_patience = 5
    ckpt_path = os.path.join(output_dir, f"best_model_{model_label}.pth")

    if os.path.exists(ckpt_path):
        print(f"Found existing best model at {ckpt_path}. Skipping training.")
    else:
        print("\n=== Starting Fine-Tuning ===")
        for epoch in range(1, max_epochs + 1):
            # Use BalancedSampler for training
            train_sampler = BalancedSampler(train_dataset, samples_per_dataset=1000)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                sampler=train_sampler, # Use sampler instead of shuffle
                num_workers=num_workers, 
                pin_memory=True
            )
            
            valid_loader = DataLoader(
                valid_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=True
            )

            train_metrics = finetune_one_epoch(
                model, optimizer, scaler, train_loader, device, 
                lambda_recon_bio=lambda_recon_bio
            )

            # Evaluate every 10 epochs or on the last epoch
            if epoch % 10 == 0 or epoch == max_epochs:
                val_metrics = evaluate_one_epoch(model, valid_loader, device)
                val_loss = val_metrics.get("loss_poisson", 0.0) + val_metrics["loss_poisson_bio"]
                
                scheduler.step(val_loss)

                print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss_total']:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  -> New best model saved (val={best_val:.4f})")
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= early_stop_patience:
                    print(f"Early stopping after {early_stop_patience} epochs.")
                    break
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss_total']:.4f}")

        print(f"Finished training. Best val = {best_val:.4f}")

    # 7. Inference & Evaluation on Spatial Samples
    print("\n=== Starting Inference and Evaluation on Spatial Samples ===")
    
    # Load best model
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()

    inference_output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results/ft" 
    os.makedirs(inference_output_dir, exist_ok=True)
    
    for sample_name, (file_path, sample_id) in spatial_file_map.items():
        print(f"\nProcessing {sample_name} (ID: {sample_id})")
        
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        # Create Dataset for this specific sample
        dataset = XVerseFineTuneDataset(
            {file_path: sample_id},
            gene_ids,
            tissue_map,
            use_qc=False
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )

        # Determine test genes
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
            print(f"Filtering output to {len(gene_id_subset)} genes.")
        
        # Run Inference
        save_poisson_parameter_ft(
            model=model,
            loader=loader,
            sample_name=sample_name,
            gene_ids=gene_ids,
            output_dir=inference_output_dir,
            device=device,
            use_amp=torch.cuda.is_available(),
            estimate_genes=False, 
            save_panel_genes_only=False,
            gene_id_subset=gene_id_subset
        )

        # Evaluate
        mu_path = os.path.join(inference_output_dir, f"{sample_name}_mu_bio.h5ad")
        if os.path.exists(test_path) and os.path.exists(mu_path):
            print(f"Evaluating {sample_name}...")
            try:
                corrs = align_and_evaluate(test_path, mu_path)
                if corrs:
                    # Print Quartiles
                    all_corrs = list(corrs.values())
                    quartiles = np.percentile(all_corrs, [0, 25, 50, 75, 100])
                    print(f"Pearson Correlation Quartiles for {sample_name}:")
                    print(f"  Min:    {quartiles[0]:.4f}")
                    print(f"  25%:    {quartiles[1]:.4f}")
                    print(f"  Median: {quartiles[2]:.4f}")
                    print(f"  75%:    {quartiles[3]:.4f}")
                    print(f"  Max:    {quartiles[4]:.4f}")

                    # Save Correlations
                    sorted_corrs = sorted(corrs.items(), key=lambda item: item[1], reverse=True)
                    corr_df = pd.DataFrame(sorted_corrs, columns=["gene_id", "pearson_corr"])
                    save_path = os.path.join(inference_output_dir, f"{sample_name}_gene_correlations.csv")
                    corr_df.to_csv(save_path, index=False)
                    print(f"Saved correlations to {save_path}")
            except Exception as e:
                print(f"Error evaluating {sample_name}: {e}")
    print("\n=== All Done ===")

if __name__ == "__main__":
    main()
