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

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import gc
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from main_energy.utils_model import (
    load_gene_ids,
    MaskFiLMGMMVAE,
    train_gmm_vae_one_epoch,
    evaluate_gmm_vae_one_epoch,
    FastXVerseBatchDataset,
    build_pair_to_sample_id_and_paths,
    build_cell_type_to_index,
    BalancedSampleSampler,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the mask-FiLM GMM-VAE with original xVERSE data pipeline.")
    parser.add_argument("--data-root", default="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor",
                        help="Root directory containing gene ids and summary csv files.")
    parser.add_argument("--gene-ids-path", default=None,
                        help="Path to gene id list. Defaults to <data-root>/ensg_keys_high_quality.txt.")
    parser.add_argument("--summary-csv", default=None,
                        help="Path to dataset summary csv.")
    parser.add_argument("--use-tissue", default=None,
                        help="Train/validate using only a specific tissue name (e.g., kidney).")
    parser.add_argument("--cell-type-csv", default="/hpc/group/xielab/xj58/sparest_code/standard_type/cellxgene_cell_type_mapped.csv",
                        help="CSV containing cell-type mapping info.")
    parser.add_argument("--result-dir", default="/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118",
                        help="Directory for checkpoints.")
    parser.add_argument("--total-gene", type=int, default=17999, help="Total number of genes.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--val-batch-size", type=int, default=512, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=20, help="DataLoader workers for both train/val.")
    parser.add_argument("--samples-per-id", type=int, default=1000, help="Samples drawn per id in sampler.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="LR scheduler factor.")
    parser.add_argument("--scheduler-patience", type=int, default=3, help="LR scheduler patience.")
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4, help="LR scheduler threshold.")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="LR scheduler minimum LR.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--last-ckpt-name", default="last_model.pth", help="Filename for last checkpoint.")
    parser.add_argument("--best-ckpt-name", default="best_model.pth", help="Filename for best checkpoint.")
    parser.add_argument("--beta-kl", type=float, default=1.0, help="KL weight.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dim.")
    parser.add_argument("--num-components", type=int, default=16, help="GMM component count K.")
    parser.add_argument("--expr-hidden-dim", type=int, default=1024, help="Expression encoder hidden dim.")
    parser.add_argument("--mask-hidden-dim", type=int, default=512, help="Mask encoder hidden dim.")
    parser.add_argument("--dec-hidden-dim", type=int, default=1024, help="Decoder hidden dim.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for gmm_vae MLP blocks.")
    parser.add_argument("--recon-observed-only", action="store_true",
                        help="For gmm_vae, compute Poisson NLL only on observed genes.")
    parser.add_argument("--mask-aug-prob", type=float, default=1.0,
                        help="For gmm_vae training, probability of applying random observed->unobserved masking per cell.")
    parser.add_argument("--mask-aug-policy", choices=["xverse", "simple"], default="xverse",
                        help="Mask augmentation policy. 'xverse' mimics main_energy/utils_model.py random masking style.")
    parser.add_argument("--mask-aug-min-frac", type=float, default=0.1,
                        help="Minimum fraction of observed genes to hide when mask augmentation is applied.")
    parser.add_argument("--mask-aug-max-frac", type=float, default=0.5,
                        help="Maximum fraction of observed genes to hide when mask augmentation is applied.")
    parser.add_argument("--lambda-score", type=float, default=0.0,
                        help="Weight of auxiliary score loss. Keep small when enabled.")
    parser.add_argument("--score-noise-std", type=float, default=0.1,
                        help="Gaussian noise std for score head training.")
    parser.add_argument("--score-detach-z", action="store_true", default=True,
                        help="Detach z before score head to prevent score loss from updating encoder.")
    parser.add_argument("--no-score-detach-z", dest="score_detach_z", action="store_false",
                        help="Allow score loss to update encoder (not recommended for your current objective).")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    args = parse_args()
    set_seed(args.seed)

    gene_ids_path = args.gene_ids_path or os.path.join(args.data_root, "ensg_keys_high_quality.txt")
    summary_csv_path = args.summary_csv or os.path.join(args.data_root, "pantissue_full_updated.csv")
    os.makedirs(args.result_dir, exist_ok=True)
    ckpt_path = os.path.join(args.result_dir, args.last_ckpt_name)
    best_ckpt_path = os.path.join(args.result_dir, args.best_ckpt_name)

    pair_to_idx, train_pairs, val_pairs, pair_to_tissue_id, _ = build_pair_to_sample_id_and_paths(
        summary_csv_path,
        use_tissue=args.use_tissue
    )
    cell_type_to_index = build_cell_type_to_index(args.cell_type_csv)
    gene_ids = load_gene_ids(gene_ids_path)
    if args.total_gene != len(gene_ids):
        print(f"[Info] Override total_gene from {args.total_gene} to {len(gene_ids)} based on gene id file.")
        args.total_gene = len(gene_ids)

    print("Creating Dataset...")
    ds = FastXVerseBatchDataset(train_pairs, gene_ids, pair_to_idx, cell_type_to_index, pair_to_tissue_id=pair_to_tissue_id)
    val_ds = FastXVerseBatchDataset(val_pairs, gene_ids, pair_to_idx, cell_type_to_index, pair_to_tissue_id=pair_to_tissue_id)
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, num_workers=args.num_workers,
        pin_memory=True, shuffle=False, drop_last=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskFiLMGMMVAE(
        num_genes=args.total_gene,
        latent_dim=args.latent_dim,
        num_components=args.num_components,
        expr_hidden_dim=args.expr_hidden_dim,
        mask_hidden_dim=args.mask_hidden_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        cooldown=0,
        min_lr=args.scheduler_min_lr
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    start_round = 1
    best_val_metric = float("inf")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_round = checkpoint["epoch"] + 1
        best_val_metric = checkpoint.get("best_val_metric", best_val_metric)

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"[Resume] Loaded checkpoint from epoch {checkpoint['epoch']}, best_val_metric={best_val_metric:.4f}")

    epoch_id = start_round

    while epoch_id <= args.num_epochs:
        start_time = time.time()
        print(f"\n[Epoch {epoch_id}] Starting...")

        sampler = BalancedSampleSampler(ds, samples_per_id=args.samples_per_id)
        train_loader = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, sampler=sampler, drop_last=True
        )

        loss_full, loss_recon, loss_kl, loss_score = train_gmm_vae_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            device=device,
            beta_kl=args.beta_kl,
            recon_observed_only=args.recon_observed_only,
            mask_aug_prob=args.mask_aug_prob,
            mask_aug_policy=args.mask_aug_policy,
            mask_aug_min_frac=args.mask_aug_min_frac,
            mask_aug_max_frac=args.mask_aug_max_frac,
            lambda_score=args.lambda_score,
            score_noise_std=args.score_noise_std,
            score_detach_z=args.score_detach_z,
        )
        print(
            f"[Epoch {epoch_id}] "
            f"Loss={loss_full:.4f}, Recon={loss_recon:.4f}, KL={loss_kl:.4f}, Score={loss_score:.4f}"
        )

        val_loss_full, val_loss_recon, val_loss_kl, val_loss_score = evaluate_gmm_vae_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            beta_kl=args.beta_kl,
            recon_observed_only=args.recon_observed_only,
            lambda_score=args.lambda_score,
            score_noise_std=args.score_noise_std,
            score_detach_z=args.score_detach_z,
        )
        print(
            f"[Epoch {epoch_id}] Validation Loss: "
            f"Loss={val_loss_full:.4f}, Recon={val_loss_recon:.4f}, KL={val_loss_kl:.4f}, Score={val_loss_score:.4f}"
        )
        val_metric = val_loss_full

        scheduler.step(val_metric)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch_id}] Current Learning Rate: {current_lr:.6f}")

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save({
                "epoch": epoch_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_metric": best_val_metric,
                "args": vars(args),
            }, best_ckpt_path)
            print(f"[Best Model] Updated at epoch {epoch_id} with metric={val_metric:.4f}")

        torch.save({
            "epoch": epoch_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_metric": best_val_metric,
            "args": vars(args),
        }, ckpt_path)
        print(f"[Checkpoint] Saved as {args.last_ckpt_name} at epoch {epoch_id}")

        del train_loader
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[Epoch {epoch_id}] Time elapsed: {time.time() - start_time:.2f}s")
        epoch_id += 1


if __name__ == "__main__":
    main()
