import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from main.utils_ft import (
    XVerseFineTuneDataset,
    save_poisson_parameter_ft,
    XVerseFineTuneModel,
    finetune_one_epoch,
    evaluate_one_epoch,
)
from main.utils_model import XVerseModel, load_gene_ids

# ================= CONFIGURATION =================
adata_3p_path = "/hpc/group/xielab/xj58/xVerse_results/fig6/GSE164378_RAW/rna_3P.h5ad"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"
base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/last_model.pth"
infer_out_root = "/hpc/group/xielab/xj58/xVerse_results/fig6/pretrain_inference"

outdir = "/hpc/group/xielab/xj58/xVerse_results/fig6/sampled_data"
os.makedirs(outdir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tissue_map = {"blood": 5}
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
FINETUNE_EPOCHS = 25
NUM_ESTIMATION = 5

# Sampling configuration
SAMPLE_SIZE = 60
MIN_PER_TYPE = 3

# Groups definition
GROUPS = {
  "ILC_NK_group": [
    "ILC",
    "NK",
  ],
  "HSPC_boundary_group": [
    "HSPC",
    "CD4 Proliferating",
  ],
  "Eryth_platelet_group": [
    "Eryth",
    "Platelet"
  ],
  "Plasmablast_boundary_group": [
    "Plasmablast",
    "B memory",
  ],
}

# ===============================================

print("Loading gene ids...")
gene_ids = load_gene_ids(gene_ids_path)


def print_l2_summary(label, adata):
    """Helper to print cell type composition."""
    if adata.n_obs == 0:
        print(f"\n>>> {label} is Empty!")
        return
    counts = adata.obs["celltype.l2"].value_counts()
    print(f"\n>>> {label} (total={adata.n_obs})")
    for subtype, n in counts.items():
        if n > 0:
            print(f"  {subtype}: {n}")


def sample_group_cells(adata, group_types, n_total=150, min_per_type=5):
    """
    Sample n_total cells from the group, ensuring at least min_per_type for each type if available.
    """
    # 1. Filter for group types
    mask = adata.obs["celltype.l2"].isin(group_types)
    adata_group = adata[mask].copy()
    
    n_available = adata_group.n_obs
    if n_available <= n_total:
        print(f"  [Sample] Taking all {n_available} cells (<= {n_total})")
        return adata_group
    
    # 2. Identify indices for each type
    type_indices = {}
    for ct in group_types:
        indices = np.where(adata_group.obs["celltype.l2"] == ct)[0]
        if len(indices) > 0:
            type_indices[ct] = indices
            
    # 3. Select mandatory cells
    selected_indices = set()
    pool_indices = []
    
    for ct, indices in type_indices.items():
        n_ct = len(indices)
        n_keep = min(n_ct, min_per_type)
        
        # Randomly select n_keep mandatory cells
        keep_idx = np.random.choice(indices, n_keep, replace=False)
        selected_indices.update(keep_idx)
        
        # Add the rest to the pool
        rest_idx = list(set(indices) - set(keep_idx))
        pool_indices.extend(rest_idx)
        
    # 4. Fill the rest
    n_current = len(selected_indices)
    n_needed = n_total - n_current
    
    if n_needed > 0:
        if len(pool_indices) >= n_needed:
            fill_idx = np.random.choice(pool_indices, n_needed, replace=False)
            selected_indices.update(fill_idx)
        else:
            selected_indices.update(pool_indices)
            
    # 5. Return subset
    final_indices = sorted(list(selected_indices))
    return adata_group[final_indices].copy()


def process_dataset(path, tag):
    """Process dataset by extracting DC cells and creating downsampled versions."""
    print(f"\n========== PROCESS {tag} ==========")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []

    adata = sc.read_h5ad(path)
    print(f"{tag} raw: obs={adata.n_obs}, var={adata.n_vars}")

    if "celltype.l2" not in adata.obs.columns:
        print("Error: 'celltype.l2' missing.")
        return []

    saved_paths = []
    
    for group_name, group_types in GROUPS.items():
        print(f"\nProcessing Group: {group_name}")
        
        adata_sampled = sample_group_cells(
            adata, 
            group_types, 
            n_total=SAMPLE_SIZE, 
            min_per_type=MIN_PER_TYPE
        )
        
        if adata_sampled is not None and adata_sampled.n_obs > 0:
            print_l2_summary(f"{group_name} (Sampled)", adata_sampled)
            
            save_name = f"{tag}_{group_name}.h5ad"
            save_path = os.path.join(outdir, save_name)
            adata_sampled.write(save_path)
            print(f"Saved: {save_path}")
            saved_paths.append(save_path)
        else:
            print(f"  [Warning] No cells found for group {group_name}")

    return saved_paths


def load_pretrained_model():
    """Load the pretrained base model."""
    print("Loading pretrained model...")
    base = XVerseModel(num_samples=None, hidden_dim=384).to(device)
    
    if not os.path.exists(base_model_ckpt):
        raise FileNotFoundError(f"Pretrained checkpoint not found at {base_model_ckpt}")
    
    ckpt = torch.load(base_model_ckpt, map_location=device)
    
    # Handle both formats: direct state_dict or wrapped in "model_state_dict" key
    state = ckpt.get("model_state_dict", ckpt)
    
    # Remove 'module.' prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    
    base.load_state_dict(state, strict=False)
    
    model = XVerseFineTuneModel(base_model=base, num_samples_finetune=None).to(device)
    
    print("Pretrained model loaded successfully.")
    return model


def finetune_model(sample_path, sample_name):
    """Fine-tune model on ALL data for fixed epochs (no split)."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning model for: {sample_name}")
    print(f"Training on ALL data for {FINETUNE_EPOCHS} epochs")
    print(f"{'='*60}")
    
    finetune_dir = os.path.join(infer_out_root, sample_name, "finetuned_models")
    os.makedirs(finetune_dir, exist_ok=True)
    
    dataset = XVerseFineTuneDataset(
        {sample_path: None},
        gene_ids,
        tissue_map,
        use_qc=False,
    )
    print(f"Dataset size: {len(dataset)} cells")
    
    print("\nInitializing model from pretrained checkpoint...")
    model = load_pretrained_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    final_model_path = os.path.join(finetune_dir, "finetuned_model.pth")
    
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"\nStarting fine-tuning for {FINETUNE_EPOCHS} epochs...")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_metrics = finetune_one_epoch(
            model,
            optimizer,
            scaler,
            train_loader,
            device,
            lambda_recon_bio=1.0,
        )
        
        if epoch % 5 == 0 or epoch == 1 or epoch == FINETUNE_EPOCHS:
            print(
                f"[Epoch {epoch}/{FINETUNE_EPOCHS}] "
                f"loss={train_metrics['loss_total']:.4f}"
            )
    
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model saved to: {final_model_path}")
    
    return final_model_path


def load_finetuned_model(model_path):
    """Load a fine-tuned model from disk."""
    print(f"Loading fine-tuned model from: {model_path}")
    
    base = XVerseModel(num_samples=None, hidden_dim=384).to(device)
    model = XVerseFineTuneModel(base_model=base, num_samples_finetune=None).to(device)
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    print("Fine-tuned model loaded successfully.")
    return model


def run_inference(sample_path, sample_name, finetuned_model_path):
    """Run inference using the fine-tuned model."""
    print(f"\n{'='*60}")
    print(f"Running inference for: {sample_name}")
    print(f"{'='*60}")
    
    sample_dir = os.path.join(infer_out_root, sample_name)
    os.makedirs(sample_dir, exist_ok=True)
    
    model = load_finetuned_model(finetuned_model_path)
    
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
        num_workers=16,
        pin_memory=True,
    )
    
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
    print(f"Inference results saved to: {sample_dir}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    
    print("\n" + "="*60)
    print("STEP 1: Generate scenario datasets")
    print("="*60)
    generated_paths = process_dataset(adata_3p_path, "3P")
    
    if not generated_paths:
        print("No datasets generated. Exiting.")
        exit(1)
    
    print("\n" + "="*60)
    print("STEP 2: Fine-tune models on all data")
    print("="*60)
    finetuned_models = {}
    for sample_path in generated_paths:
        sample_name = os.path.splitext(os.path.basename(sample_path))[0]
        model_path = finetune_model(sample_path, sample_name)
        finetuned_models[sample_name] = model_path
    
    print("\n" + "="*60)
    print("STEP 3: Run inference with fine-tuned models")
    print("="*60)
    for sample_path in generated_paths:
        sample_name = os.path.splitext(os.path.basename(sample_path))[0]
        finetuned_model_path = finetuned_models[sample_name]
        run_inference(sample_path, sample_name, finetuned_model_path)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
