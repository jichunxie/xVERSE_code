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
import os
import torch
import scanpy as sc
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
# Configuration
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tissue mapping used in your model
tissue_map = {"blood": 5}

base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig7/ft_model_GSE291290"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# Batch files (you only need to edit these paths)
# ==============================================================

file_sample_dict = {
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL1_1_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL1_2_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL2_1_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL2_2_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL3_1_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL3_2_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL4_1_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL4_2_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL5_1_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL5_2_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL6_1_rna.h5ad": None,
    "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample/CONTROL6_2_rna.h5ad": None,
}

# ==============================================================
# ==============================================================
# Inference function
# ==============================================================

def run_inference(model, label, estimate_genes=True, num_estimation=20):
    """
    Run inference on all batch files and save Poisson parameters.
    """
    print(f"Running inference after fine-tuning: {label}")
    root = os.path.join(output_dir, f"inference_{label}")
    os.makedirs(root, exist_ok=True)

    for sample_path, sample_id in file_sample_dict.items():
        sample_name = os.path.splitext(os.path.basename(sample_path))[0]
        sample_dir = os.path.join(root, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        dataset_inf = XVerseFineTuneDataset(
            {sample_path: sample_id},
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
            estimate_genes=estimate_genes,
            save_panel_genes_only=True,
            num_estimation=num_estimation,
        )

    print(f"Inference saved to: {root}")


# ==============================================================
# Training preparation
# ==============================================================

print("Loading gene ids...")
gene_ids = load_gene_ids(gene_ids_path)

print("Building dataset...")
dataset = XVerseFineTuneDataset(file_sample_dict, gene_ids, tissue_map, use_qc=False)

# Split dataset
n_total = len(dataset)
n_train = int(n_total * 0.9)
train_dataset, valid_dataset = random_split(dataset, [n_train, n_total - n_train])
print(f"Total={n_total}, train={n_train}, valid={n_total - n_train}")

# ==============================================================
# Load base model
# ==============================================================

base_model = XVerseModel(num_samples=None, hidden_dim=384).to(device)
ckpt = torch.load(base_model_ckpt, map_location=device)
state = ckpt.get("model_state_dict", ckpt)
state = {k.replace("module.", ""): v for k, v in state.items()}
base_model.load_state_dict(state, strict=False)

model = XVerseFineTuneModel(
    base_model=base_model,
    num_samples_finetune=None
).to(device)


# ==============================================================
# Optimizer / scheduler
# ==============================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_val = float("inf")
best_ckpt = os.path.join(output_dir, "best_model.pth")


# ==============================================================
# Training loop
# ==============================================================

for epoch in range(1, 51):
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    # One epoch train
    train_metrics = finetune_one_epoch(
        model,
        optimizer,
        scaler,
        train_loader,
        device,
        lambda_recon_bio=0.1,
    )

    # Evaluation
    val_metrics = evaluate_one_epoch(model, valid_loader, device)
    val_loss = val_metrics.get("loss_poisson", 0.0) + val_metrics["loss_poisson_bio"]
    scheduler.step(val_loss)

    print(
        f"[Epoch {epoch}] "
        f"train={train_metrics['loss_total']:.4f} "
        f"val={val_loss:.4f}"
    )

    # Save best model
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), best_ckpt)
        print(f"  Updated best checkpoint. val={best_val:.4f}")

print(f"Training completed. Best model → {best_ckpt}")


# ==============================================================
# Final inference
# ==============================================================

model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)
run_inference(model, label="finetuned")
