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

from main.utils_model import (
    load_gene_ids,
    pretrain_one_epoch,
    evaluate_one_epoch,
    XVerseModel,
    FastXVerseBatchDataset,
    build_pair_to_sample_id_and_paths,
    build_cell_type_to_index,
    BalancedSampleSampler,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the XVerse pan-tissue model.")
    parser.add_argument("--data-root", default="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor",
                        help="Root directory containing gene ids and summary csv files.")
    parser.add_argument("--gene-ids-path", default=None,
                        help="Path to gene id list. Defaults to <data-root>/ensg_keys_high_quality.txt.")
    parser.add_argument("--summary-csv", default=None,
                        help="Path to dataset summary csv.")
    parser.add_argument("--cell-type-csv", default="/hpc/group/xielab/xj58/sparest_code/standard_type/cellxgene_cell_type_mapped.csv",
                        help="CSV containing cell-type mapping info.")
    parser.add_argument("--result-dir", default="/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118",
                        help="Directory for checkpoints.")
    parser.add_argument("--num-samples", type=int, default=10509,
                        help="Number of unique sample embeddings.")
    parser.add_argument("--hidden-dim", type=int, default=384, help="Hidden dimension size.")
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

    pair_to_idx, train_pairs, val_pairs, pair_to_tissue_id, _ = build_pair_to_sample_id_and_paths(summary_csv_path)
    cell_type_to_index = build_cell_type_to_index(args.cell_type_csv)
    gene_ids = load_gene_ids(gene_ids_path)

    print("Creating Dataset...")
    ds = FastXVerseBatchDataset(train_pairs, gene_ids, pair_to_idx, cell_type_to_index, pair_to_tissue_id=pair_to_tissue_id)
    val_ds = FastXVerseBatchDataset(val_pairs, gene_ids, pair_to_idx, cell_type_to_index, pair_to_tissue_id=pair_to_tissue_id)
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, num_workers=args.num_workers,
        pin_memory=True, shuffle=False, drop_last=False
    )
    masks = ds.get_spatial_panel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XVerseModel(num_samples=args.num_samples, hidden_dim=args.hidden_dim,
                        masks=masks, total_gene=args.total_gene).to(device)

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
    best_val_loss_bio = float("inf")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_round = checkpoint["epoch"] + 1
        best_val_loss_bio = checkpoint.get("best_val_loss_bio", best_val_loss_bio)

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"[Resume] Loaded checkpoint from epoch {checkpoint['epoch']}, best_val_loss_bio={best_val_loss_bio:.4f}")

    epoch_id = start_round

    while epoch_id <= args.num_epochs:
        start_time = time.time()
        print(f"\n[Epoch {epoch_id}] Starting...")

        sampler = BalancedSampleSampler(ds, samples_per_id=args.samples_per_id)
        train_loader = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, sampler=sampler, drop_last=True
        )

        (
            loss_full, loss_bio, loss_adv, loss_ct
        ) = pretrain_one_epoch(
            model, optimizer, scaler, train_loader, device
        )
        print(
            f"[Epoch {epoch_id}] "
            f"Loss_full={loss_full:.4f}, Loss_bio={loss_bio:.4f}, "
            f"Adv={loss_adv:.4f}, CellType={loss_ct:.4f}"
        )

        (
            val_loss_full, val_loss_bio, val_loss_adv, val_loss_ct
        ) = evaluate_one_epoch(model, val_loader, device)

        print(
            f"[Epoch {epoch_id}] Validation Loss: "
            f"Loss_full={val_loss_full:.4f}, Loss_bio={val_loss_bio:.4f}, "
            f"Adv={val_loss_adv:.4f}, CellType={val_loss_ct:.4f}"
        )

        scheduler.step(val_loss_bio)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch_id}] Current Learning Rate: {current_lr:.6f}")

        if val_loss_bio < best_val_loss_bio:
            best_val_loss_bio = val_loss_bio
            torch.save({
                "epoch": epoch_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss_bio": best_val_loss_bio
            }, best_ckpt_path)
            print(f"[Best Model] Updated at epoch {epoch_id} with Loss_bio={val_loss_bio:.4f}")

        torch.save({
            "epoch": epoch_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss_bio": best_val_loss_bio
        }, ckpt_path)
        print(f"[Checkpoint] Saved as {args.last_ckpt_name} at epoch {epoch_id}")

        del train_loader
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[Epoch {epoch_id}] Time elapsed: {time.time() - start_time:.2f}s")
        epoch_id += 1


if __name__ == "__main__":
    main()
