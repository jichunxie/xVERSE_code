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
xVERSE for Extraction, Fine-tuning, and Generation.
"""

import os
import argparse
import glob
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse

# ==============================================================
# Import custom modules
# ==============================================================
from main.utils_ft import (
    XVerseFineTuneDataset,
    XVerseFineTuneModel,
    finetune_one_epoch,
    evaluate_one_epoch,
)
from main.utils_model import XVerseModel, load_gene_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Unified xVerse Runner")

    # --- Mode and Task ---
    parser.add_argument("--mode", type=str, choices=["0shot", "ft"], default="0shot",
                        help="Execution mode: '0shot' (load pretrained) or 'ft' (fine-tune first).")
    parser.add_argument("--task", type=str, choices=["embedding", "generation"], required=True,
                        help="Task to perform: 'embedding' (save latent reps) or 'generation' (save Poisson params).")

    # --- Paths ---
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .h5ad files (or path to single .h5ad).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results.")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to pretrained base model checkpoint (.pth).")
    
    # Default gene_ids path relative to this script
    default_gene_ids = os.path.join(os.path.dirname(__file__), "ensg_keys_high_quality.txt")
    parser.add_argument("--gene_ids", type=str, default=default_gene_ids,
                        help="Path to gene IDs text file.")

    # --- Configuration ---
    parser.add_argument("--tissue_name", type=str, required=True, help="Tissue name (e.g., 'liver').")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers.")

    # --- Fine-tuning Hyperparameters (only used if --mode ft) ---
    parser.add_argument("--epochs", type=int, default=20, help="Max fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=384, help="Hidden dimension.")
    parser.add_argument("--train_frac", type=float, default=0.9, help="Train/Val split fraction.")
    parser.add_argument("--lambda_recon_bio", type=float, default=1.0, help="Loss weight.")
    
    # --- Generation Hyperparameters ---
    parser.add_argument("--num_samples_gen", type=int, default=5, help="Number of Poisson samples to generate (default: 5).")

    return parser.parse_args()


# ==============================================================
# Helper Functions
# ==============================================================

def get_files_from_path(path_str):
    """Return a list of .h5ad files from a directory or a single file."""
    if os.path.isdir(path_str):
        files = sorted(glob.glob(os.path.join(path_str, "*.h5ad")))
    elif os.path.isfile(path_str) and path_str.endswith(".h5ad"):
        files = [path_str]
    else:
        files = []
    return files


def load_tissue_map(tissue_name):
    """Load tissue map and return ID for given tissue name."""
    map_path = os.path.join(os.path.dirname(__file__), "tissue_name_to_id_map.csv")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Tissue map not found at {map_path}")
    
    df = pd.read_csv(map_path)
    # Normalize map validation
    df['tissue_name_clean'] = df['tissue_name'].astype(str).str.strip().str.lower()
    mapping = dict(zip(df['tissue_name_clean'], df['tissue_id']))
    
    clean_name = tissue_name.strip().lower()
    if clean_name not in mapping:
        available = sorted(df['tissue_name'].unique())
        raise ValueError(f"Tissue '{tissue_name}' not found in map. Available tissues: {available}")
    
    return mapping[clean_name]


def build_model(base_ckpt_path, num_samples, hidden_dim=384, device='cpu', mode="0shot"):
    """
    Load model.
    If mode="ft", wraps in XVerseFineTuneModel.
    If mode="0shot", loads into base model (or wrapped with num_samples=0/1 depending on need).
    """
    print(f"Loading base model from {base_ckpt_path}...")
    base_model = XVerseModel(num_samples=None, hidden_dim=hidden_dim).to(device)
    
    if os.path.exists(base_ckpt_path):
        ckpt = torch.load(base_ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        base_model.load_state_dict(state, strict=False)
    else:
         raise FileNotFoundError(f"Checkpoint not found at {base_ckpt_path}")

    if mode == "ft":
        print(f"Wrapping with FineTuneModel (num_samples={num_samples})")
        model = XVerseFineTuneModel(
            base_model=base_model,
            num_samples_finetune=num_samples,
        ).to(device)
    else:
        print("Wrapping with FineTuneModel (Zero-Shot mode)")
        model = XVerseFineTuneModel(
             base_model=base_model,
             num_samples_finetune=None,
        ).to(device)
        
    return model


# ==============================================================
# Execution Blocks
# ==============================================================

def run_finetuning_loop(model, full_dataset, args, device, output_dir):
    """Run fine-tuning."""
    print(f"\n=== Starting Fine-Tuning ({args.epochs} epochs) ===")
    
    n_total = len(full_dataset)
    n_train = int(n_total * args.train_frac)
    n_valid = n_total - n_train
    
    # Deterministic split
    train_dataset, valid_dataset = random_split(
        full_dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train size: {n_train} | Valid size: {n_valid}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_val = float("inf")
    no_improve = 0
    early_stop_patience = 10
    ckpt_path = os.path.join(output_dir, f"best_model_{args.tissue_name}_ft.pth")

    for epoch in range(1, args.epochs + 1):
        # Training
        # Standard DataLoader with shuffle
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        
        train_metrics = finetune_one_epoch(
            model, optimizer, scaler, train_loader, device,
            lambda_recon_bio=args.lambda_recon_bio
        )
        
        # Validation
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        val_metrics = evaluate_one_epoch(model, valid_loader, device)
        val_loss = val_metrics.get("loss_poisson", 0.0) + val_metrics["loss_poisson_bio"]
        
        scheduler.step(val_loss)
        print(f"[Epoch {epoch}] Train: {train_metrics['loss_total']:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved.")
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= early_stop_patience:
            print("Early stopping.")
            break
            
    print(f"Fine-tuning complete. Best Val: {best_val:.4f}")
    
    # Reload best
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def run_embedding_extraction(model, file_sample_dict, gene_ids, tissue_map, args, device):
    """Extract embeddings and save to .h5ad."""
    print("\n=== Running Embedding Extraction ===")
    
    processed_files = []
    
    for fpath, _ in file_sample_dict.items():
        fname = os.path.basename(fpath)
        print(f"Processing: {fname}")
        
        try:
            adata = sc.read_h5ad(fpath)
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")
            continue

        # Determine sample_id based on mode
        # If 0-shot, pass None. If ft, pass the assigned ID (although embedding usually ignores it, this keeps consistent)
        sid_for_dataset = file_sample_dict[fpath] if args.mode == "ft" else None

        # Create temporary dataset wrapper for inference
        dataset = XVerseFineTuneDataset(
            {fpath: sid_for_dataset}, 
            gene_ids,
            tissue_map,
            use_qc=False,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        embed_list = []
        model.eval()
        
        with torch.no_grad():
            for _, values, tissue_ids in tqdm(loader, leave=False):
                values = values.to(device)
                tissue_ids = tissue_ids.to(device)
                
                with torch.amp.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=torch.cuda.is_available()):
                    if hasattr(model, 'bio_encoder'):
                        z_bio, _ = model.bio_encoder(values, tissue_ids)
                    else:
                        z_bio, _ = model.base_model.bio_encoder(values, tissue_ids)
                
                embed_list.append(z_bio.cpu().numpy())
                
        embeddings = np.concatenate(embed_list, axis=0)
        adata.obsm["xVerse"] = embeddings
        
        save_path = os.path.join(args.output_dir, fname)
        adata.write(save_path)
        print(f"  Saved to {save_path}")
        processed_files.append(save_path)
        
    # Optional Combined UMAP
    if len(processed_files) > 1:
        print("Generating combined UMAP...")
        try:
            adatas = [sc.read_h5ad(p) for p in processed_files]
            combined = sc.concat(adatas, index_unique=None, keys=[os.path.basename(p) for p in processed_files], label="batch")
            sc.pp.neighbors(combined, use_rep="xVerse", n_neighbors=15)
            sc.tl.umap(combined)
            fig, ax = plt.subplots(figsize=(8,8))
            sc.pl.umap(combined, color='batch', ax=ax, show=False)
            plt.savefig(os.path.join(args.output_dir, "combined_umap.png"), dpi=300)
            print("  Saved UMAP.")
        except Exception as e:
            print(f"  UMAP failed: {e}")


def run_generation_task(model, file_sample_dict, gene_ids, tissue_map, args, device):
    """Generate Poisson parameters."""
    print("\n=== Running Generation (Poisson Parameters) ===")
    
    inference_dir = os.path.join(args.output_dir, "inference_params")
    os.makedirs(inference_dir, exist_ok=True)
    
    for fpath, sample_id in file_sample_dict.items():
        fname = os.path.basename(fpath)
        print(f"Processing: {fname} (ID: {sample_id})")
        
        try:
            adata = sc.read_h5ad(fpath)
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")
            continue

        # Determine sample_id based on mode
        # If 0-shot, we don't use sample-specific params, so pass None
        # If ft, we use the specific sample_id
        dataset_sid = sample_id if args.mode == "ft" else None
        
        dataset = XVerseFineTuneDataset(
            {fpath: dataset_sid},
            gene_ids,
            tissue_map,
            use_qc=False
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        mu_bio_list = []
        model.eval()
        
        # Inference Loop
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Generating parameters", leave=False):
                if len(batch) == 3:
                     sample_id_batch, values, tissue_ids = batch
                     sample_id_batch = sample_id_batch.to(device)
                else:
                     values, tissue_ids = batch
                     sample_id_batch = None
                
                values = values.to(device)
                tissue_ids = tissue_ids.to(device)
                
                with torch.amp.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=torch.cuda.is_available()):
                    # Match utils_ft logic
                    has_sample = getattr(model, "use_sample_specific", False)
                    if has_sample and sample_id_batch is not None:
                        outputs = model(value=values, tissue_id=tissue_ids, sample_id=sample_id_batch)
                    else:
                        outputs = model(value=values, tissue_id=tissue_ids)
                
                mu_bio_list.append(outputs["mu_bio"].cpu().numpy())
        
        mu_bio = np.concatenate(mu_bio_list, axis=0)
        
        # Update AnnData
        # Always save as independent file to ensure gene order matches model (ensg_keys_high_quality.txt)
        print(f"  Saving output to separate file with model genes ({len(gene_ids)} genes)...")
        
        # Create new AnnData with model genes
        adata_out = sc.AnnData(X=sparse.csr_matrix(mu_bio)) # mu_bio can be dense or sparse, usually dense is fine for rate, but let's keep it numpy array if dense
        # Actually mu_bio is typically dense float. 
        # User asked for "sample matrix need to be sparse". mu_bio is usually kept as is (often dense).
        # Let's check previous code. Previous code: adata_out = sc.AnnData(X=mu_bio)
        # We will keep X=mu_bio (dense) for precision, but samples as sparse.
        
        adata_out = sc.AnnData(X=mu_bio)
        adata_out.var_names = gene_ids
        adata_out.obs = adata.obs.copy()
        
        # Save sparse samples
        if args.num_samples_gen > 0:
            print(f"  Generating {args.num_samples_gen} Poisson samples...")
            for i in range(args.num_samples_gen):
                # Generate counts
                sample_counts = np.random.poisson(mu_bio)
                # Save as sparse CSR
                adata_out.layers[f"sample_{i}"] = sparse.csr_matrix(sample_counts)
        
        # Save mu_bio mainly for reference, though it is in X
        # adata_out.layers['mu_bio'] = mu_bio # Optional, X is already mu_bio
        # The previous code had layers['mu_bio']. Let's keep consistency if users expect it.
        adata_out.layers['mu_bio'] = mu_bio

        save_path = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_mu_bio.h5ad")
        adata_out.write(save_path)
        print(f"  Saved to {save_path}")



def main():
    args = parse_args()
    
    # Setup Device
    if args.gpu is not None:
         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Resources
    tissue_id = load_tissue_map(args.tissue_name)
    tissue_map = {args.tissue_name: tissue_id}
    print(f"Tissue: {args.tissue_name} (ID: {tissue_id})")
    
    print(f"Loading gene IDs from {args.gene_ids}")
    gene_ids = load_gene_ids(args.gene_ids)
    
    # 2. Get Input Files
    files = get_files_from_path(args.input_dir)
    if not files:
        print(f"No .h5ad files found in {args.input_dir}")
        return
    print(f"Found {len(files)} input files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Map files to sample IDs (0, 1, 2...)
    file_sample_dict = {f: i for i, f in enumerate(files)}
    
    # 3. Mode Handling (Fine-tuning vs Zero-shot)
    model = build_model(
        args.base_model, 
        num_samples=len(files), 
        hidden_dim=args.hidden_dim, 
        device=device,
        mode=args.mode
    )
    
    if args.mode == "ft":
        # Create full dataset for training
        full_dataset = XVerseFineTuneDataset(
            file_sample_dict,
            gene_ids,
            tissue_map,
            use_qc=False,
        )
        print(f"Fine-tuning on {len(full_dataset)} total cells.")
        model = run_finetuning_loop(model, full_dataset, args, device, args.output_dir)
    else:
        print("Zero-shot mode: Using pretrained model without modification.")
        model.eval()

    # 4. Task Handling
    if args.task == "embedding":
        run_embedding_extraction(model, file_sample_dict, gene_ids, tissue_map, args, device)
    elif args.task == "generation":
        run_generation_task(model, file_sample_dict, gene_ids, tissue_map, args, device)
    else:
        print(f"Unknown task: {args.task}")

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
