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
import json
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import GradScaler
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from main_energy.utils_model import (
    load_gene_ids,
    MaskFiLMGMMVAE,
    train_gmm_vae_one_epoch,
    evaluate_gmm_vae_one_epoch,
    FastXVerseBatchDataset,
    SparseBatchCollator,
    CompiledShardDataset,
    CompiledSparseBatchCollator,
    build_pair_to_sample_id_and_paths,
    build_cell_type_to_index,
    BalancedSampleSampler,
    DistributedBalancedSampler,
    CompiledBalancedSampler,
    DistributedCompiledBalancedSampler,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the mask-FiLM GMM-VAE with original xVERSE data pipeline.")
    parser.add_argument("--compiled-dataset-root", default=None,
                        help="Path to compiled dataset root (format=xverse_train_v1). If set, training reads compiled shards directly.")
    parser.add_argument("--compiled-max-cached-shards", type=int, default=8,
                        help="Max opened shard arrays cached per process for compiled dataset reading.")
    parser.add_argument("--sampler-shard-reorder-window", type=int, default=4096,
                        help="For compiled dataset samplers: window size for local shard-aware reordering after random sid sampling. <=1 disables.")
    parser.add_argument("--sampler-active-shards", type=int, default=0,
                        help="For compiled dataset samplers: number of active shards per phase. >0 enables staged shard-pool ordering.")
    parser.add_argument("--data-root", default="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor",
                        help="Root directory containing gene ids and summary csv files.")
    parser.add_argument("--gene-ids-path", default=None,
                        help="Path to gene id list. Defaults to <data-root>/ensg_keys_high_quality.txt.")
    parser.add_argument("--summary-csv", default=None,
                        help="Path to dataset summary csv.")
    parser.add_argument("--train-index-cache", default=None,
                        help="Optional prebuilt train index cache (.npz).")
    parser.add_argument("--val-index-cache", default=None,
                        help="Optional prebuilt val index cache (.npz).")
    parser.add_argument("--allow-stale-index-cache", action="store_true",
                        help="Allow using index cache even when signature mismatches (use with caution).")
    parser.add_argument("--use-tissue", default=None,
                        help="Train/validate using only a specific tissue name (e.g., kidney).")
    parser.add_argument("--filter-bad-cells", action="store_true",
                        help="Filter cells with max count > 1000 or total count > 200000 when building dataset/index.")
    parser.add_argument("--cell-type-csv", default=None,
                        help="CSV containing cell-type mapping info. Must be provided.")
    parser.add_argument("--result-dir", default=None,
                        help="Directory for checkpoints. Must be provided.")
    parser.add_argument("--total-gene", type=int, default=17999, help="Total number of genes.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--val-batch-size", type=int, default=512, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=20, help="DataLoader workers for both train/val.")
    parser.add_argument("--val-num-workers", type=int, default=None,
                        help="Validation DataLoader workers. Defaults to min(num_workers, 4).")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (when num_workers>0).")
    parser.add_argument("--persistent-workers", action="store_true", default=True,
                        help="Keep DataLoader workers alive across epochs.")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false",
                        help="Disable persistent DataLoader workers.")
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
    parser.add_argument("--beta-kl", type=float, default=0.01, help="KL weight.")
    parser.add_argument("--prior-type", choices=["gmm", "gaussian"], default="gmm",
                        help="Latent prior type. 'gaussian' uses N(0,I) with closed-form KL.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dim.")
    parser.add_argument("--num-components", type=int, default=16, help="GMM component count K.")
    parser.add_argument("--prior-cov-rank", type=int, default=8,
                        help="Low-rank size R for each GMM component covariance: diag + U U^T.")
    parser.add_argument("--prior-shared-cov", action="store_true",
                        help="Share GMM prior log-variance and low-rank covariance factor across components.")
    parser.add_argument("--posterior-cov-rank", type=int, default=0,
                        help="Low-rank size R for posterior q(z|x,c) covariance: diag + U U^T. 0 keeps diagonal posterior.")
    parser.add_argument("--expr-hidden-dim", type=int, default=1024, help="Expression encoder hidden dim.")
    parser.add_argument("--mask-hidden-dim", type=int, default=512, help="Mask encoder hidden dim.")
    parser.add_argument("--dec-hidden-dim", type=int, default=1024, help="Decoder hidden dim.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for gmm_vae MLP blocks.")
    parser.add_argument("--recon-observed-only", action="store_true",
                        help="For gmm_vae, compute Poisson NLL only on observed genes.")
    parser.add_argument("--recon-loss", choices=["poisson", "nb"], default="poisson",
                        help="Reconstruction loss type for count model.")
    parser.add_argument("--recon-gene-weight-mode", choices=["none", "inv_log1p_mean_ema"], default="none",
                        help="Optional gene-wise reconstruction reweighting mode.")
    parser.add_argument("--recon-gene-weight-alpha", type=float, default=0.0,
                        help="Mixing strength for gene-wise recon weighting. 0 disables weighting.")
    parser.add_argument("--recon-gene-weight-ema-momentum", type=float, default=0.99,
                        help="EMA momentum for gene mean used by recon weighting.")
    parser.add_argument("--recon-gene-weight-min", type=float, default=0.3,
                        help="Lower clip bound for gene recon weights.")
    parser.add_argument("--recon-gene-weight-max", type=float, default=3.0,
                        help="Upper clip bound for gene recon weights.")
    parser.add_argument("--recon-gene-weight-eps", type=float, default=1e-6,
                        help="Numerical epsilon for gene recon weighting.")
    parser.add_argument("--recon-cell-weight-mode", choices=["none", "component_usage", "batch_kmeans"], default="none",
                        help="Optional cell-wise reconstruction reweighting mode.")
    parser.add_argument("--recon-cell-weight-alpha", type=float, default=0.0,
                        help="Mixing strength for cell-wise recon weighting. 0 disables weighting.")
    parser.add_argument("--recon-cell-weight-min", type=float, default=0.5,
                        help="Lower clip bound for cell recon weights.")
    parser.add_argument("--recon-cell-weight-max", type=float, default=2.0,
                        help="Upper clip bound for cell recon weights.")
    parser.add_argument("--recon-cell-weight-eps", type=float, default=1e-6,
                        help="Numerical epsilon for cell recon weighting.")
    parser.add_argument("--recon-cell-weight-clusters", type=int, default=32,
                        help="Number of batch k-means clusters for cell recon weighting.")
    parser.add_argument("--recon-cell-weight-kmeans-iters", type=int, default=2,
                        help="Fast k-means iterations for batch_kmeans cell recon weighting.")
    parser.add_argument("--mask-aug-prob", type=float, default=1.0,
                        help="For gmm_vae training, probability of applying random observed->unobserved masking per cell.")
    parser.add_argument("--mask-aug-policy", choices=["xverse", "simple"], default="xverse",
                        help="Mask augmentation policy. 'xverse' mimics main_energy/utils_model.py random masking style.")
    parser.add_argument("--mask-aug-min-frac", type=float, default=0.1,
                        help="Minimum fraction of observed genes to hide when mask augmentation is applied.")
    parser.add_argument("--mask-aug-max-frac", type=float, default=0.5,
                        help="Maximum fraction of observed genes to hide when mask augmentation is applied.")
    parser.add_argument("--num-cell-types", type=int, default=0,
                        help="Number of supervised cell types for auxiliary classification head. <=0 means auto infer from dataset.")
    parser.add_argument("--num-tissues", type=int, default=0,
                        help="Number of tissue ids for conditional prior. <=0 means auto infer from dataset.")
    parser.add_argument("--num-batches", type=int, default=0,
                        help="Number of sample/batch ids for decoder conditioning. <=0 means auto infer from dataset.")
    parser.add_argument("--batch-emb-dim", type=int, default=0,
                        help="Sample/batch embedding dim for decoder FiLM conditioning. 0 disables batch conditioning.")
    parser.add_argument("--batch-cond-drop-prob", type=float, default=0.0,
                        help="Probability of dropping decoder batch condition during training.")
    parser.add_argument("--lambda-batchless-recon", type=float, default=0.0,
                        help="Weight of reconstruction loss with decoder batch condition disabled.")
    parser.add_argument("--conditional-prior-on-tissue", action="store_true",
                        help="Use tissue-conditional GMM prior p(z|tissue).")
    parser.add_argument("--lambda-celltype-cls", type=float, default=0.0,
                        help="Weight for auxiliary celltype cross-entropy loss (ignore label -1).")
    parser.add_argument("--prior-logvar-max", type=float, default=4.0,
                        help="Upper clamp bound for GMM prior log-variance.")
    parser.add_argument("--lambda-prior-pi-balance", type=float, default=0.0,
                        help="Weight for balancing global mixture weights toward uniform.")
    parser.add_argument("--lambda-prior-mu-spread", type=float, default=0.0,
                        help="Weight for repulsive regularization between prior component means.")
    parser.add_argument("--prior-mu-spread-tau", type=float, default=1.0,
                        help="Length scale for prior mean spread regularization (smaller => stronger local repulsion).")
    parser.add_argument("--lambda-contrast", type=float, default=1.0,
                        help="Weight of contrastive loss between real-mask and fake-mask views.")
    parser.add_argument("--lambda-real-recon", type=float, default=0.1,
                        help="Weight of real-mask reconstruction loss term.")
    parser.add_argument("--contrast-temp", type=float, default=0.1,
                        help="Temperature for bidirectional InfoNCE contrastive loss.")
    parser.add_argument("--ddp", action="store_true", default=True,
                        help="Use torch DistributedDataParallel when launched with torchrun.")
    parser.add_argument("--no-ddp", dest="ddp", action="store_false",
                        help="Disable DistributedDataParallel.")
    parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl",
                        help="Distributed backend for DDP.")
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


def setup_distributed(args):
    if not args.ddp:
        return False, 0, 1, 0
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Fallback to non-DDP when not launched by torchrun.
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = args.dist_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("CUDA is available but no visible devices were found.")
        if local_rank >= device_count:
            # Guard against mismatched torchrun nproc_per_node vs visible GPU count.
            mapped_rank = local_rank % device_count
            print(
                f"[WARN] LOCAL_RANK={local_rank} but only {device_count} CUDA devices are visible. "
                f"Falling back to cuda:{mapped_rank}. Please align --nproc_per_node with visible GPU count."
            )
            local_rank = mapped_rank
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _filter_state_dict_by_shape(model, state_dict):
    model_state = model.state_dict()
    kept = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in model_state:
            skipped.append((k, "missing_in_model"))
            continue
        if model_state[k].shape != v.shape:
            skipped.append((k, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}"))
            continue
        kept[k] = v
    return kept, skipped


def _set_requires_grad(module, enabled: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = bool(enabled)


def _get_last_linear(module):
    if module is None:
        return None
    if isinstance(module, torch.nn.Linear):
        return module
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    return last


def _write_args_csv(result_dir: str, args_namespace, rank: int):
    if not is_main_process(rank):
        return
    args_dict = vars(args_namespace).copy()
    items = sorted(args_dict.items(), key=lambda kv: kv[0])

    cur_path = os.path.join(result_dir, "train_args.csv")
    with open(cur_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in items:
            w.writerow([k, v])

    hist_path = os.path.join(result_dir, "train_args_history.csv")
    run_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_exists = os.path.exists(hist_path)
    with open(hist_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["run_timestamp", "key", "value"])
        for k, v in items:
            w.writerow([run_ts, k, v])


def _apply_training_stage(base_model, stage: str):
    # Single-stage training (former stage3): train all modules jointly.
    _set_requires_grad(getattr(base_model, "encoder", None), True)
    _set_requires_grad(getattr(base_model, "decoder", None), True)
    _set_requires_grad(getattr(base_model, "library_head", None), True)
    _set_requires_grad(getattr(base_model, "prior", None), True)
    _set_requires_grad(getattr(base_model, "post_c_logits", None), True)
    _set_requires_grad(getattr(base_model, "post_mu", None), True)
    _set_requires_grad(getattr(base_model, "post_logvar", None), True)
    _set_requires_grad(getattr(base_model, "post_factor", None), True)
    _set_requires_grad(getattr(base_model, "score_head", None), True)


def _kmeans_torch(x: torch.Tensor, k: int, iters: int, seed: int):
    n = x.size(0)
    if n < k:
        raise ValueError(f"Not enough points for KMeans: n={n}, k={k}")
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))
    centers = x[torch.randperm(n, generator=g, device=x.device)[:k]].clone()
    assign = torch.zeros((n,), dtype=torch.long, device=x.device)
    for _ in range(max(1, int(iters))):
        dist2 = torch.cdist(x, centers, p=2) ** 2
        assign = torch.argmin(dist2, dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(assign, minlength=k).to(x.dtype).unsqueeze(1)
        new_centers.index_add_(0, assign, x)
        non_empty = counts.squeeze(1) > 0
        new_centers[non_empty] = new_centers[non_empty] / counts[non_empty]
        if (~non_empty).any():
            refill = x[torch.randperm(n, generator=g, device=x.device)[: int((~non_empty).sum().item())]]
            new_centers[~non_empty] = refill
        centers = new_centers
    return centers, assign


def main():
    args = parse_args()
    if args.result_dir is None:
        raise ValueError("`--result-dir` is required.")
    if (not args.compiled_dataset_root) and (args.cell_type_csv is None):
        raise ValueError("`--cell-type-csv` is required when not using --compiled-dataset-root.")
    ddp_enabled, rank, world_size, local_rank = setup_distributed(args)
    if args.ddp and not ddp_enabled:
        print("[Info] --ddp is enabled by default, but torchrun env was not found. Falling back to single-process mode.")
    set_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def log(msg: str):
        if is_main_process(rank):
            print(msg)

    os.makedirs(args.result_dir, exist_ok=True)
    _write_args_csv(args.result_dir, args, rank)
    ckpt_path = os.path.join(args.result_dir, args.last_ckpt_name)
    best_ckpt_path = os.path.join(args.result_dir, args.best_ckpt_name)

    if args.compiled_dataset_root:
        ignored = ["--data-root"]
        if args.summary_csv is not None:
            ignored.append("--summary-csv")
        if args.gene_ids_path is not None:
            ignored.append("--gene-ids-path")
        if args.cell_type_csv != "/hpc/group/xielab/xj58/sparest_code/standard_type/cellxgene_cell_type_mapped.csv":
            ignored.append("--cell-type-csv")
        if args.train_index_cache is not None:
            ignored.append("--train-index-cache")
        if args.val_index_cache is not None:
            ignored.append("--val-index-cache")
        if args.allow_stale_index_cache:
            ignored.append("--allow-stale-index-cache")
        if args.use_tissue is not None:
            ignored.append("--use-tissue")
        if args.filter_bad_cells:
            ignored.append("--filter-bad-cells")
        if ignored:
            log(f"[Info] Ignored in compiled mode: {', '.join(ignored)}")

        manifest_path = os.path.join(args.compiled_dataset_root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Compiled dataset manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        fmt = manifest.get("format")
        if fmt != "xverse_train_v1":
            raise ValueError(f"Unsupported compiled dataset format: {fmt}")
        compiled_num_genes = int(manifest.get("global_num_genes", -1))
        if compiled_num_genes <= 0:
            raise ValueError("Invalid global_num_genes in compiled manifest.")
        if args.total_gene != compiled_num_genes:
            log(f"[Info] Override total_gene from {args.total_gene} to {compiled_num_genes} based on compiled manifest.")
            args.total_gene = compiled_num_genes

        log("Creating Dataset from compiled shards...")
        log(
            "[CompiledIO] "
            f"max_cached_shards={args.compiled_max_cached_shards}, "
            f"sampler_shard_reorder_window={args.sampler_shard_reorder_window}, "
            f"sampler_active_shards={args.sampler_active_shards}"
        )
        ds = CompiledShardDataset(
            compiled_root=args.compiled_dataset_root,
            split="train",
            max_cached_shards=args.compiled_max_cached_shards,
        )
        val_ds = CompiledShardDataset(
            compiled_root=args.compiled_dataset_root,
            split="val",
            max_cached_shards=args.compiled_max_cached_shards,
        )
        train_collator = CompiledSparseBatchCollator(
            num_genes=args.total_gene,
            apply_mask_aug=True,
            mask_aug_prob=args.mask_aug_prob,
            mask_aug_policy=args.mask_aug_policy,
            mask_aug_min_frac=args.mask_aug_min_frac,
            mask_aug_max_frac=args.mask_aug_max_frac,
        )
        val_collator = CompiledSparseBatchCollator(
            num_genes=args.total_gene,
            apply_mask_aug=False,
        )
        inferred_num_cell_types = max(ds.infer_num_celltypes(), val_ds.infer_num_celltypes())
    else:
        gene_ids_path = args.gene_ids_path or os.path.join(args.data_root, "ensg_keys_high_quality.txt")
        summary_csv_path = args.summary_csv or os.path.join(args.data_root, "pantissue_full_updated.csv")
        pair_to_idx, train_pairs, val_pairs, pair_to_tissue_id, _ = build_pair_to_sample_id_and_paths(
            summary_csv_path,
            use_tissue=args.use_tissue
        )
        cell_type_to_index = build_cell_type_to_index(args.cell_type_csv)
        gene_ids = load_gene_ids(gene_ids_path)
        if args.total_gene != len(gene_ids):
            log(f"[Info] Override total_gene from {args.total_gene} to {len(gene_ids)} based on gene id file.")
            args.total_gene = len(gene_ids)

        log("Creating Dataset...")
        tissue_tag = "all" if args.use_tissue is None else str(args.use_tissue).strip().replace(" ", "_")
        default_train_cache = os.path.join(args.data_root, f"xverse_index_cache_train_{tissue_tag}.npz")
        default_val_cache = os.path.join(args.data_root, f"xverse_index_cache_val_{tissue_tag}.npz")
        train_index_cache = args.train_index_cache or (default_train_cache if os.path.exists(default_train_cache) else None)
        val_index_cache = args.val_index_cache or (default_val_cache if os.path.exists(default_val_cache) else None)
        log(f"[IndexCache] train={train_index_cache if train_index_cache else 'None'}, val={val_index_cache if val_index_cache else 'None'}")
        ds = FastXVerseBatchDataset(
            train_pairs,
            gene_ids,
            pair_to_idx,
            cell_type_to_index,
            pair_to_tissue_id=pair_to_tissue_id,
            filter_bad_cells=args.filter_bad_cells,
            index_cache_path=train_index_cache,
            allow_stale_index_cache=args.allow_stale_index_cache,
        )
        val_ds = FastXVerseBatchDataset(
            val_pairs,
            gene_ids,
            pair_to_idx,
            cell_type_to_index,
            pair_to_tissue_id=pair_to_tissue_id,
            filter_bad_cells=args.filter_bad_cells,
            index_cache_path=val_index_cache,
            allow_stale_index_cache=args.allow_stale_index_cache,
        )
        train_collator = SparseBatchCollator(
            ds,
            num_genes=args.total_gene,
            apply_mask_aug=True,
            mask_aug_prob=args.mask_aug_prob,
            mask_aug_policy=args.mask_aug_policy,
            mask_aug_min_frac=args.mask_aug_min_frac,
            mask_aug_max_frac=args.mask_aug_max_frac,
        )
        val_collator = SparseBatchCollator(
            val_ds,
            num_genes=args.total_gene,
            apply_mask_aug=False,
        )
        inferred_num_cell_types = max(ds.infer_num_celltypes(), val_ds.infer_num_celltypes())

    if int(args.num_cell_types) > 0:
        num_cell_types = int(args.num_cell_types)
    else:
        num_cell_types = int(inferred_num_cell_types)
    log(f"[CellType] num_cell_types={num_cell_types}, lambda_celltype_cls={args.lambda_celltype_cls}")

    inferred_num_tissues = max(ds.infer_num_tissues(), val_ds.infer_num_tissues())
    if int(args.num_tissues) > 0:
        num_tissues = int(args.num_tissues)
    else:
        num_tissues = int(inferred_num_tissues)
    log(f"[Tissue] num_tissues={num_tissues}, conditional_prior_on_tissue={args.conditional_prior_on_tissue}")

    inferred_num_batches = max(ds.infer_num_samples(), val_ds.infer_num_samples())
    if int(args.num_batches) > 0:
        num_batches = int(args.num_batches)
    else:
        num_batches = int(inferred_num_batches)
    log(
        f"[BatchCond] num_batches={num_batches}, batch_emb_dim={args.batch_emb_dim}, "
        f"drop_prob={args.batch_cond_drop_prob}, lambda_batchless_recon={args.lambda_batchless_recon}"
    )

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    val_num_workers = min(args.num_workers, 4) if args.val_num_workers is None else max(0, int(args.val_num_workers))
    val_loader_kwargs = dict(num_workers=val_num_workers, pin_memory=True)
    if val_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        # Keep val workers non-persistent to avoid train/val worker resource contention.
        val_loader_kwargs["persistent_workers"] = False
    log(
        f"[Loader] train_workers={args.num_workers}, train_persistent={args.persistent_workers}, "
        f"val_workers={val_num_workers}, val_persistent={val_loader_kwargs.get('persistent_workers', False)}"
    )

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp_enabled else None
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False if val_sampler is None else False,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=val_collator,
        **val_loader_kwargs,
    )

    if args.compiled_dataset_root:
        if ddp_enabled:
            train_sampler = DistributedCompiledBalancedSampler(
                ds,
                samples_per_id=args.samples_per_id,
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
                shard_reorder_window=args.sampler_shard_reorder_window,
                active_shards=args.sampler_active_shards,
            )
        else:
            train_sampler = CompiledBalancedSampler(
                ds,
                samples_per_id=args.samples_per_id,
                seed=args.seed,
                shard_reorder_window=args.sampler_shard_reorder_window,
                active_shards=args.sampler_active_shards,
            )
    else:
        if ddp_enabled:
            train_sampler = DistributedBalancedSampler(
                ds,
                samples_per_id=args.samples_per_id,
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
            )
        else:
            train_sampler = BalancedSampleSampler(ds, samples_per_id=args.samples_per_id, seed=args.seed)
    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=train_collator,
        **loader_kwargs,
    )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if ddp_enabled else "cuda")
    else:
        device = torch.device("cpu")
    model = MaskFiLMGMMVAE(
        num_genes=args.total_gene,
        latent_dim=args.latent_dim,
        num_components=args.num_components,
        prior_cov_rank=args.prior_cov_rank,
        prior_shared_covariance=args.prior_shared_cov,
        posterior_cov_rank=args.posterior_cov_rank,
        expr_hidden_dim=args.expr_hidden_dim,
        mask_hidden_dim=args.mask_hidden_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        dropout=args.dropout,
        prior_type=args.prior_type,
        num_cell_types=num_cell_types,
        conditional_prior_on_tissue=args.conditional_prior_on_tissue,
        num_tissues=num_tissues,
        num_batches=num_batches,
        batch_emb_dim=args.batch_emb_dim,
        batch_cond_drop_prob=args.batch_cond_drop_prob,
        recon_loss_type=args.recon_loss,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    log(f"Total parameters: {total_params:,}")
    log(f"Trainable parameters: {trainable_params:,}")

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    elif torch.cuda.device_count() > 1:
        log(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    # Keep optimizer on stable AMP-compatible path for this environment.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if torch.cuda.is_available():
        log("[Optimizer] Using standard Adam (AMP-compatible).")
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
        map_location = device
        ckpt = torch.load(ckpt_path, map_location=map_location)
        ckpt_state = ckpt["model_state_dict"]
        filtered_state, skipped = _filter_state_dict_by_shape(model, ckpt_state)
        load_ret = model.load_state_dict(filtered_state, strict=False)
        if skipped:
            log(f"[Resume] Skipped {len(skipped)} incompatible/missing keys from checkpoint.")
            for name, reason in skipped[:20]:
                log(f"[Resume][Skip] {name}: {reason}")
            if len(skipped) > 20:
                log(f"[Resume] ... and {len(skipped) - 20} more skipped keys.")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                log(
                    f"[Resume][WARN] Optimizer state is incompatible with current model/param groups ({e}). "
                    "Skip optimizer resume and continue with freshly initialized optimizer."
                )
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except ValueError as e:
                log(
                    f"[Resume][WARN] Scheduler state is incompatible ({e}). "
                    "Skip scheduler resume and continue with freshly initialized scheduler."
                )
        best_val_metric = float(ckpt.get("best_val_metric", best_val_metric))
        last_epoch = int(ckpt.get("epoch", 0))
        start_round = last_epoch + 1
        log(
            f"[Resume] Loaded {ckpt_path} (epoch={last_epoch}, best_val_metric={best_val_metric:.6f}). "
            f"Continue from epoch {start_round}."
        )
        if getattr(load_ret, "missing_keys", None):
            log(f"[Resume] Missing keys (expected with new heads): {load_ret.missing_keys}")
        if getattr(load_ret, "unexpected_keys", None):
            log(f"[Resume] Unexpected keys: {load_ret.unexpected_keys}")

    epoch_id = start_round

    while epoch_id <= args.num_epochs:
        start_time = time.time()
        log(f"\n[Epoch {epoch_id}] Starting...")
        beta_t = float(args.beta_kl)
        stage_name = "stage3"
        base_model = _unwrap_model(model)
        _apply_training_stage(base_model, stage_name)

        # Default: global schedules.
        lambda_resp_anchor_t = 0.0

        log(
            f"[Epoch {epoch_id}] stage={stage_name}, beta_kl={beta_t:.6f}, "
            f"lambda_resp_anchor={lambda_resp_anchor_t:.6f}"
        )

        force_base_posterior = False

        train_sampler.set_epoch(epoch_id)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch_id)
        loss_full, loss_recon, loss_kl, loss_score, loss_contrast, loss_cov, loss_prior_pi_balance, loss_celltype_cls, loss_batchless_recon = train_gmm_vae_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            device=device,
            beta_kl=beta_t,
            recon_observed_only=args.recon_observed_only,
            mask_aug_prob=args.mask_aug_prob,
            mask_aug_policy=args.mask_aug_policy,
            mask_aug_min_frac=args.mask_aug_min_frac,
            mask_aug_max_frac=args.mask_aug_max_frac,
            lambda_contrast=args.lambda_contrast,
            contrast_temp=args.contrast_temp,
            lambda_real_recon=args.lambda_real_recon,
            lambda_resp_anchor=lambda_resp_anchor_t,
            lambda_score=0.0,
            score_noise_std=0.0,
            score_detach_z=True,
            lambda_cov=0.0,
            cov_use_mu=True,
            lambda_resp_balance=0.0,
            lambda_resp_confidence=0.0,
            resp_temperature=1.0,
            resp_topk=0,
            prior_logvar_min=None,
            prior_logvar_max=args.prior_logvar_max,
            lambda_prior_mu_l2=0.0,
            lambda_prior_factor_l2=0.0,
            lambda_prior_pi_balance=args.lambda_prior_pi_balance,
            lambda_prior_mu_spread=args.lambda_prior_mu_spread,
            prior_mu_spread_tau=args.prior_mu_spread_tau,
            lambda_celltype_cls=args.lambda_celltype_cls,
            lambda_prior_logvar_l2=0.0,
            prior_logvar_target=0.0,
            recon_gene_weight_mode=args.recon_gene_weight_mode,
            recon_gene_weight_alpha=args.recon_gene_weight_alpha,
            recon_gene_weight_ema_momentum=args.recon_gene_weight_ema_momentum,
            recon_gene_weight_min=args.recon_gene_weight_min,
            recon_gene_weight_max=args.recon_gene_weight_max,
            recon_gene_weight_eps=args.recon_gene_weight_eps,
            recon_cell_weight_mode=args.recon_cell_weight_mode,
            recon_cell_weight_alpha=args.recon_cell_weight_alpha,
            recon_cell_weight_min=args.recon_cell_weight_min,
            recon_cell_weight_max=args.recon_cell_weight_max,
            recon_cell_weight_eps=args.recon_cell_weight_eps,
            recon_cell_weight_clusters=args.recon_cell_weight_clusters,
            recon_cell_weight_kmeans_iters=args.recon_cell_weight_kmeans_iters,
            lambda_batchless_recon=args.lambda_batchless_recon,
            force_base_posterior=force_base_posterior,
        )
        train_msg = (
            f"[Epoch {epoch_id}] "
            f"Loss={loss_full:.4f}, Recon={loss_recon:.4f}, KL={loss_kl:.4f}, "
            f"Score={loss_score:.4f}, priorPiBal={loss_prior_pi_balance:.4f}, cls={loss_celltype_cls:.4f}"
        )
        if args.lambda_contrast > 0:
            train_msg += f", Contrast={loss_contrast:.4f}"
        if args.lambda_batchless_recon > 0:
            train_msg += f", BatchlessRecon={loss_batchless_recon:.4f}"
        log(train_msg)

        do_val = (
            (int(args.val_every) <= 1)
            or (epoch_id % int(args.val_every) == 0)
            or (epoch_id == args.num_epochs)
        )
        if do_val:
            val_loss_full, val_loss_recon, val_loss_kl, val_loss_score, val_loss_contrast, val_loss_cov, val_loss_prior_pi_balance, val_loss_celltype_cls, val_loss_batchless_recon = evaluate_gmm_vae_one_epoch(
                model=model,
                val_loader=val_loader,
                device=device,
                beta_kl=beta_t,
                recon_observed_only=args.recon_observed_only,
                lambda_score=0.0,
                score_noise_std=0.0,
                score_detach_z=True,
                lambda_contrast=args.lambda_contrast,
                contrast_temp=args.contrast_temp,
                lambda_real_recon=args.lambda_real_recon,
                lambda_cov=0.0,
                cov_use_mu=True,
                lambda_resp_balance=0.0,
                lambda_resp_confidence=0.0,
                lambda_resp_anchor=lambda_resp_anchor_t,
                resp_temperature=1.0,
                resp_topk=0,
                prior_logvar_min=None,
                prior_logvar_max=args.prior_logvar_max,
                lambda_prior_mu_l2=0.0,
                lambda_prior_factor_l2=0.0,
                lambda_prior_pi_balance=args.lambda_prior_pi_balance,
                lambda_prior_mu_spread=args.lambda_prior_mu_spread,
                prior_mu_spread_tau=args.prior_mu_spread_tau,
                lambda_celltype_cls=args.lambda_celltype_cls,
                lambda_prior_logvar_l2=0.0,
                prior_logvar_target=0.0,
                recon_gene_weight_mode=args.recon_gene_weight_mode,
                recon_gene_weight_alpha=args.recon_gene_weight_alpha,
                recon_gene_weight_ema_momentum=args.recon_gene_weight_ema_momentum,
                recon_gene_weight_min=args.recon_gene_weight_min,
                recon_gene_weight_max=args.recon_gene_weight_max,
                recon_gene_weight_eps=args.recon_gene_weight_eps,
                recon_cell_weight_mode=args.recon_cell_weight_mode,
                recon_cell_weight_alpha=args.recon_cell_weight_alpha,
                recon_cell_weight_min=args.recon_cell_weight_min,
                recon_cell_weight_max=args.recon_cell_weight_max,
                recon_cell_weight_eps=args.recon_cell_weight_eps,
                recon_cell_weight_clusters=args.recon_cell_weight_clusters,
                recon_cell_weight_kmeans_iters=args.recon_cell_weight_kmeans_iters,
                lambda_batchless_recon=args.lambda_batchless_recon,
                mask_aug_prob=args.mask_aug_prob,
                mask_aug_policy=args.mask_aug_policy,
                mask_aug_min_frac=args.mask_aug_min_frac,
                mask_aug_max_frac=args.mask_aug_max_frac,
                force_base_posterior=force_base_posterior,
            )
            val_msg = (
                f"[Epoch {epoch_id}] Validation Loss: "
                f"Loss={val_loss_full:.4f}, Recon={val_loss_recon:.4f}, KL={val_loss_kl:.4f}, "
                f"Score={val_loss_score:.4f}, priorPiBal={val_loss_prior_pi_balance:.4f}, cls={val_loss_celltype_cls:.4f}"
            )
            if args.lambda_contrast > 0:
                val_msg += f", Contrast={val_loss_contrast:.4f}"
            if args.lambda_batchless_recon > 0:
                val_msg += f", BatchlessRecon={val_loss_batchless_recon:.4f}"
            log(val_msg)
            val_metric = val_loss_full

            scheduler.step(val_metric)
            current_lr = optimizer.param_groups[0]['lr']
            log(f"[Epoch {epoch_id}] Current Learning Rate: {current_lr:.6f}")

            if val_metric < best_val_metric and is_main_process(rank):
                best_val_metric = val_metric
                torch.save({
                    "epoch": epoch_id,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_metric": best_val_metric,
                    "args": vars(args),
                }, best_ckpt_path)
                log(f"[Best Model] Updated at epoch {epoch_id} with metric={val_metric:.4f}")
        else:
            log(f"[Epoch {epoch_id}] Skip validation (val_every={args.val_every}).")
            current_lr = optimizer.param_groups[0]['lr']
            log(f"[Epoch {epoch_id}] Current Learning Rate: {current_lr:.6f}")

        if is_main_process(rank):
            torch.save({
                "epoch": epoch_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_metric": best_val_metric,
                "args": vars(args),
            }, ckpt_path)
            log(f"[Checkpoint] Saved as {args.last_ckpt_name} at epoch {epoch_id}")

        log(f"[Epoch {epoch_id}] Time elapsed: {time.time() - start_time:.2f}s")
        epoch_id += 1

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
