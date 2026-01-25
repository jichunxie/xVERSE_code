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
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from main.utils_ft import (
    XVerseFineTuneDataset,
    save_poisson_parameter_ft,
    XVerseFineTuneModel,
    finetune_one_epoch,
)
from main.utils_model import XVerseModel, load_gene_ids

# ================= CONFIGURATION =================
# Use 5P data
adata_5p_path = "/hpc/group/xielab/xj58/xVerse_results/fig6/GSE164378_RAW/rna_5P.h5ad"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"
base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/last_model.pth"
infer_out_root = "/hpc/group/xielab/xj58/xVerse_results/fig6/pretrain_inference"

# Output directory for sampled data
outdir = "/hpc/group/xielab/xj58/xVerse_results/fig6/sampled_1pct_per_ct"
os.makedirs(outdir, exist_ok=True)

CELLTYPE_KEY = "celltype.l1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tissue_map = {"blood": 5}
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
FINETUNE_EPOCHS = 1500
NUM_ESTIMATION = 100

gene_ids = load_gene_ids(gene_ids_path)

def split_and_sample_per_celltype(adata, celltype_key, sample_frac=0.01):
    """
    Split adata by cell type, sample `sample_frac` (1%) of cells for each type (excluding "other"),
    and save each as a separate file.
    
    Returns: List of tuples (cell_type, file_path)
    """
    if celltype_key not in adata.obs.columns:
        print(f"  [Sample] Warning: '{celltype_key}' column not found. Skipping.")
        return []

    # Merge T cells
    print("  [Sample] Merging T cell subtypes (CD4 T, CD8 T, other T) into 'T'...")
    labels = adata.obs[celltype_key].astype(str).values
    t_subtypes = ["CD4 T", "CD8 T", "other T"]
    for t in t_subtypes:
        labels[labels == t] = "T"
    adata.obs[celltype_key] = labels
    adata.obs[celltype_key] = adata.obs[celltype_key].astype("category")

    cell_types = adata.obs[celltype_key].unique()
    print(f"  [Sample] Splitting and sampling {sample_frac*100}% cells per cell type (excluding 'other')...")
    
    generated_files = []
    
    for ct in cell_types:
        if ct == "other":
            continue
        
        # Subset to cell type
        adata_ct = adata[adata.obs[celltype_key] == ct].copy()
        n_total_ct = adata_ct.n_obs
        
        if n_total_ct == 0:
            continue
            
        # Sample 1%
        n_sample_ct = max(1, int(n_total_ct * sample_frac))
        
        # Randomly select indices
        all_indices_ct = np.arange(n_total_ct)
        selected_indices_ct = np.random.choice(all_indices_ct, n_sample_ct, replace=False)
        selected_indices_ct.sort()
        
        sampled_adata_ct = adata_ct[selected_indices_ct].copy()
        
        # Clean cell type name for filename
        ct_clean = str(ct).replace(" ", "_").replace("/", "_")
        save_name = f"5P_{ct_clean}_sampled.h5ad"
        save_path = os.path.join(outdir, save_name)
        
        sampled_adata_ct.write(save_path)
        print(f"    - '{ct}': {n_total_ct} -> {n_sample_ct} cells. Saved to {save_path}")
        
        generated_files.append((ct_clean, save_path))

    return generated_files


def load_pretrained_model():
    """Load the pretrained base model."""
    print("Loading pretrained model...")
    base = XVerseModel(num_samples=None, hidden_dim=384).to(device)
    
    if not os.path.exists(base_model_ckpt):
        raise FileNotFoundError(f"Pretrained checkpoint not found at {base_model_ckpt}")
    
    ckpt = torch.load(base_model_ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    base.load_state_dict(state, strict=False)
    
    model = XVerseFineTuneModel(base_model=base, num_samples_finetune=None).to(device)
    return model


def finetune_model(sample_path, sample_name):
    """Fine-tune model for a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning model for: {sample_name}")
    print(f"{'='*60}")
    
    finetune_dir = os.path.join(infer_out_root, sample_name, "finetuned_models")
    os.makedirs(finetune_dir, exist_ok=True)
    final_model_path = os.path.join(finetune_dir, "finetuned_model.pth")
    dataset = XVerseFineTuneDataset(
        {sample_path: None},
        gene_ids,
        tissue_map,
        use_qc=False,
    )
    
    model = load_pretrained_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Scheduler: Reduce LR by factor of 0.5 if loss doesn't improve for 50 epochs
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_metrics = finetune_one_epoch(
            model,
            optimizer,
            scaler,
            train_loader,
            device,
            lambda_recon_bio=1.0,
        )
        loss = train_metrics['loss_total']
        print(f"[Epoch {epoch}/{FINETUNE_EPOCHS}] Loss={loss:.4f}")
        
        # Step scheduler
        scheduler.step(loss)
            
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Fine-tuning complete for {sample_name}")
    return final_model_path


def run_inference(sample_path, sample_name, finetuned_model_path):
    """Run inference and replicate obs."""
    print(f"\n{'='*60}")
    print(f"Running inference for: {sample_name}")
    print(f"{'='*60}")
    
    sample_dir = os.path.join(infer_out_root, sample_name)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Load Model
    base = XVerseModel(num_samples=None, hidden_dim=384).to(device)
    model = XVerseFineTuneModel(base_model=base, num_samples_finetune=None).to(device)
    state = torch.load(finetuned_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    dataset_inf = XVerseFineTuneDataset(
        {sample_path: None},
        gene_ids,
        tissue_map,
        use_qc=False,
    )
    
    loader_inf = DataLoader(
        dataset_inf,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # 1. Run Inference
    save_poisson_parameter_ft(
        model=model,
        loader=loader_inf,
        sample_name=sample_name,
        gene_ids=gene_ids,
        output_dir=sample_dir,
        device=device,
        use_amp=torch.cuda.is_available(),
        estimate_genes=True,
        save_panel_genes_only=True,
        num_estimation=NUM_ESTIMATION,
    )
    
    # 2. Post-processing: Replicate obs
    inf_path = os.path.join(sample_dir, f"{sample_name}_estimated_expr_poisson.h5ad")
    if os.path.exists(inf_path):
        adata_orig = sc.read_h5ad(sample_path)
        adata_inf = sc.read_h5ad(inf_path)
        
        obs_list = [adata_orig.obs.copy() for _ in range(NUM_ESTIMATION)]
        obs_rep = pd.concat(obs_list, axis=0, ignore_index=True)
        
        if len(obs_rep) == len(adata_inf):
            for col in adata_orig.obs.columns:
                adata_inf.obs[col] = obs_rep[col].values
            adata_inf.write(inf_path)
            print(f"  ✅ Obs replicated and saved to {inf_path}")
        else:
            print(f"  [Error] Length mismatch during obs replication.")
            
    return inf_path


def aggregate_results(file_list):
    """
    Aggregate all subsampled files and all inference output files into single combined files.
    """
    print(f"\n{'='*60}")
    print(f"Aggregating Results")
    print(f"{'='*60}")
    
    subsampled_adatas = []
    inference_adatas = []
    
    for ct_clean, sample_path in file_list:
        sample_name = os.path.splitext(os.path.basename(sample_path))[0]
        inf_path = os.path.join(infer_out_root, sample_name, f"{sample_name}_estimated_expr_poisson.h5ad")
        
        if os.path.exists(sample_path):
            subsampled_adatas.append(sc.read_h5ad(sample_path))
        
        if os.path.exists(inf_path):
            inference_adatas.append(sc.read_h5ad(inf_path))
            
    # Combine Subsampled
    if subsampled_adatas:
        combined_sub = sc.concat(subsampled_adatas, axis=0, join="outer", merge="unique")
        save_sub = os.path.join(outdir, "5P_sampled_1pct_combined.h5ad")
        combined_sub.write(save_sub)
        print(f"Saved Combined Subsampled: {save_sub} (n_obs={combined_sub.n_obs})")
    else:
        save_sub = None
        
    # Combine Inference
    if inference_adatas:
        combined_inf = sc.concat(inference_adatas, axis=0, join="outer", merge="unique")
        # Save in the parent inference dir
        save_inf = os.path.join(infer_out_root, "5P_sampled_1pct_combined_estimated.h5ad")
        combined_inf.write(save_inf)
        print(f"Saved Combined Inference: {save_inf} (n_obs={combined_inf.n_obs})")
    else:
        save_inf = None
        
    return save_sub, save_inf


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    
    print("\n" + "="*60)
    print("STEP 1: Split and Sample 1% per Cell Type")
    print("="*60)
    
    if not os.path.exists(adata_5p_path):
        print(f"File not found: {adata_5p_path}")
        exit(1)
        
    adata_full = sc.read_h5ad(adata_5p_path)
    # Returns list of (ct_name, file_path)
    generated_files = split_and_sample_per_celltype(adata_full, CELLTYPE_KEY, sample_frac=0.01)
    
    if not generated_files:
        print("No files generated. Exiting.")
        exit(1)
    
    print("\n" + "="*60)
    print("STEP 2 & 3: Fine-tune and Inference per Cell Type")
    print("="*60)

    for ct_clean, sample_path in generated_files:
        sample_name = os.path.splitext(os.path.basename(sample_path))[0]
        
        # Fine-tune
        ft_model_path = finetune_model(sample_path, sample_name)
        
        # Inference
        run_inference(sample_path, sample_name, ft_model_path)
        
    # Aggregate
    aggregate_results(generated_files)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
