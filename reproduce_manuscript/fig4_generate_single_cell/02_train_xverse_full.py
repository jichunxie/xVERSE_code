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
Fine-tune XVerse on all available breast cells (no subsampling).
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from main.utils_ft import (
    XVerseFineTuneModel,
    XVerseFineTuneDataset,
    finetune_one_epoch,
    evaluate_one_epoch,
    save_poisson_parameter_ft,
)
from main.utils_model import load_gene_ids, XVerseModel


# ==============================================================  
# Configurations  
# ==============================================================  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tissue_map = {"breast": 8}

base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

file_sample_dict = {
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-A.h5ad": 0,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-B.h5ad": 1,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-C.h5ad": 2,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-D.h5ad": 3,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-E.h5ad": 4,
}

output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig4/ft_model"
os.makedirs(output_dir, exist_ok=True)

max_epochs = 300
batch_size = 64
num_workers = 16
learning_rate = 1e-3
lambda_recon_bio = 1
train_frac = 0.9
model_label = "full"


# ==============================================================  
# Utility  
# ==============================================================  

def build_model(base_ckpt_path, num_samples_finetune, hidden_dim=384):
    """Load pretrained XVerse and wrap with fine-tune head."""
    base_model = XVerseModel(num_samples=None, hidden_dim=hidden_dim).to(device)

    ckpt = torch.load(base_ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    base_model.load_state_dict(state, strict=False)

    model = XVerseFineTuneModel(
        base_model=base_model,
        num_samples_finetune=num_samples_finetune,
    ).to(device)
    return model


def run_inference(model, label, gene_ids):
    """Run inference on the full dataset and save Poisson parameters."""
    print(f"Running inference for label '{label}'...")
    model.eval()

    inference_root = os.path.join(output_dir, f"inference_{label}")
    os.makedirs(inference_root, exist_ok=True)

    for sample_path, sample_id in file_sample_dict.items():
        sample_name = os.path.splitext(os.path.basename(sample_path))[0]
        sample_dir = os.path.join(inference_root, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        dataset_inf = XVerseFineTuneDataset(
            {sample_path: sample_id},
            gene_ids,
            tissue_map,
            use_qc=False,
        )
        loader_inf = DataLoader(
            dataset_inf,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
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
            num_estimation=3,
        )

    print(f"Inference for '{label}' completed.\n")


# ==============================================================  
# Main  
# ==============================================================  

def main():

    # ---------------------------------------------
    # Load gene IDs
    # ---------------------------------------------
    print("Loading gene IDs...")
    gene_ids = load_gene_ids(gene_ids_path)

    # ---------------------------------------------
    # Build full dataset
    # ---------------------------------------------
    print("Building full dataset...")
    full_dataset = XVerseFineTuneDataset(
        file_sample_dict,
        gene_ids,
        tissue_map,
        use_qc=False,
    )
    n_total = len(full_dataset)
    print(f"Total cells = {n_total}\n")

    # ---------------------------------------------
    # Split into train/valid on full dataset
    # ---------------------------------------------
    n_train = int(n_total * train_frac)
    n_valid = n_total - n_train
    train_dataset, valid_dataset = random_split(
        full_dataset,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train = {n_train} | Valid = {n_valid}")

    # -----------------------------------------
    # Build model
    # -----------------------------------------
    model = build_model(
        base_ckpt_path=base_model_ckpt,
        num_samples_finetune=len(file_sample_dict),
        hidden_dim=384,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    no_improve = 0
    early_stop_patience = 20
    ckpt_path = os.path.join(output_dir, f"best_model_{model_label}.pth")

    # -----------------------------------------
    # Training loop
    # -----------------------------------------
    for epoch in range(1, max_epochs + 1):

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        train_metrics = finetune_one_epoch(
            model,
            optimizer,
            scaler,
            train_loader,
            device,
            lambda_recon_bio=lambda_recon_bio,
        )

        val_metrics = evaluate_one_epoch(model, valid_loader, device)
        val_loss = val_metrics.get("loss_poisson", 0.0) + val_metrics["loss_poisson_bio"]

        scheduler.step(val_loss)

        print(
            f"[{model_label} | Epoch {epoch}] "
            f"train_total={train_metrics['loss_total']:.4f} "
            f"val_total={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved (val={best_val:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print(f"Early stopping after {early_stop_patience} epochs without improvement.")
            break

    print(f"Finished training. Best val = {best_val:.4f}\n")

    # -----------------------------------------
    # Inference using best model
    # -----------------------------------------
    print(f"Loading best model and running inference on full data...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    run_inference(model, label=model_label, gene_ids=gene_ids)


if __name__ == "__main__":
    main()
