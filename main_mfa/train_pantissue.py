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
import math
import os
import random
import json
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import GradScaler
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from main_mfa.utils_model import (
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
    prior_parameter_snapshot,
    prior_parameter_delta,
    format_prior_delta,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the mask-FiLM MFA-VAE with original xVERSE data pipeline.")
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
    parser.add_argument("--init-ckpt", default=None,
                        help="Optional checkpoint used to initialize model weights when result-dir last checkpoint does not exist.")
    parser.add_argument("--train-mode", choices=["full", "prior_only"], default="full",
                        help="full trains all modules; prior_only freezes everything except model.prior.")
    parser.add_argument("--total-gene", type=int, default=17999, help="Total number of genes.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--val-fraction", type=float, default=1.0,
                        help="Random fraction of validation dataset evaluated at each validation call. 1.0 uses full validation.")
    parser.add_argument("--log-every", type=int, default=1000,
                        help="Print train/validation batch diagnostics every N batches. <=0 disables batch diagnostics.")
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
    parser.add_argument("--val-persistent-workers", action="store_true", default=False,
                        help="Keep validation DataLoader workers alive across validation calls.")
    parser.add_argument("--samples-per-id", type=int, default=1000, help="Samples drawn per id in sampler.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--encoder-lr-multiplier", type=float, default=1.0,
                        help="Multiplier for encoder and posterior-head learning rate in full training mode.")
    parser.add_argument("--prior-lr-multiplier", type=float, default=1.0,
                        help="Multiplier for model.prior parameter learning rate in full training mode.")
    parser.add_argument("--decoder-lr-multiplier", type=float, default=1.0,
                        help="Multiplier for expression decoder learning rate in full training mode.")
    parser.add_argument("--recon-head-lr-multiplier", type=float, default=1.0,
                        help="Multiplier for reconstruction auxiliary heads such as library/NB theta.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="LR scheduler factor.")
    parser.add_argument("--scheduler-patience", type=int, default=3, help="LR scheduler patience.")
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4, help="LR scheduler threshold.")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="LR scheduler minimum LR.")
    parser.add_argument("--epoch-lr-gamma", type=float, default=1.0,
                        help="Multiply every optimizer param-group LR by this value after each epoch. 1.0 disables fixed epoch decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--last-ckpt-name", default="last_model.pth", help="Filename for last checkpoint.")
    parser.add_argument("--best-ckpt-name", default="best_model.pth", help="Filename for best checkpoint.")
    parser.add_argument(
        "--resume-from",
        choices=["last", "best", "none"],
        default="last",
        help="Checkpoint to resume from inside result-dir. 'none' disables automatic result-dir resume.",
    )
    parser.add_argument("--beta-kl", type=float, default=0.01, help="KL weight.")
    parser.add_argument("--beta-u-kl-multiplier", type=float, default=1.0,
                        help="Extra multiplier on the MFA factor u KL inside the KL term. 1.0 keeps standard KL.")
    parser.add_argument("--beta-eps-kl-multiplier", type=float, default=1.0,
                        help="Extra multiplier on the MFA residual eps KL inside the KL term. 1.0 keeps standard KL.")
    parser.add_argument("--beta-kl-warmup-epochs", type=int, default=0,
                        help="Linearly warm KL weight from --beta-kl-warmup-start to --beta-kl over this many epochs. 0 disables.")
    parser.add_argument("--beta-kl-warmup-start", type=float, default=0.0,
                        help="Starting KL weight for linear KL warmup.")
    parser.add_argument("--vae-pretrain-epochs", type=int, default=0,
                        help="Train the backbone as a standard VAE with N(0,I) prior for this many epochs before switching to MFA.")
    parser.add_argument("--vae-pretrain-beta-kl", type=float, default=None,
                        help="KL weight used during --vae-pretrain-epochs. Defaults to --beta-kl when unset.")
    parser.add_argument("--prior-type", choices=["gmm", "gaussian"], default="gmm",
                        help="Latent prior type. 'gaussian' uses N(0,I) with closed-form KL.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dim.")
    parser.add_argument("--num-components", type=int, default=16, help="MFA component count K.")
    parser.add_argument("--prior-cov-rank", type=int, default=8,
                        help="Factor rank R for each MFA prior component covariance: diag + A_k A_k^T.")
    parser.add_argument("--prior-shared-cov", action="store_true",
                        help="Ignored in main_mfa; MFA prior always uses component-specific covariance/factors.")
    parser.add_argument("--prior-mu-init", choices=["normal", "sphere", "grouped_sphere", "zero"], default="normal",
                        help="Initial placement of prior component means before any delayed prior init.")
    parser.add_argument("--prior-mu-init-radius", type=float, default=1.0,
                        help="Coarse radius used by sphere/grouped_sphere prior mean initialization.")
    parser.add_argument("--prior-mu-init-groups", type=int, default=8,
                        help="Number of coarse groups for --prior-mu-init grouped_sphere.")
    parser.add_argument("--prior-mu-init-local-radius", type=float, default=0.5,
                        help="Local radius around each coarse group for --prior-mu-init grouped_sphere.")
    parser.add_argument("--posterior-cov-rank", type=int, default=0,
                        help="Ignored in main_mfa; q(u|x,c) uses --prior-cov-rank as the explicit MFA factor dimension.")
    parser.add_argument("--expr-hidden-dim", type=int, default=1024, help="Expression encoder hidden dim.")
    parser.add_argument("--mask-hidden-dim", type=int, default=512, help="Mask encoder hidden dim.")
    parser.add_argument("--dec-hidden-dim", type=int, default=1024, help="Decoder hidden dim.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for gmm_vae MLP blocks.")
    parser.add_argument("--recon-observed-only", action="store_true",
                        help="For gmm_vae, compute Poisson NLL only on observed genes.")
    parser.add_argument("--recon-loss", choices=["poisson", "nb"], default="poisson",
                        help="Reconstruction loss type for count model.")
    parser.add_argument("--nb-theta-mode", choices=["gene", "cell_gene"], default="gene",
                        help="NB dispersion parameterization. 'gene' uses one global theta per gene; 'cell_gene' decodes theta per cell per gene.")
    parser.add_argument("--recon-gene-weight-mode", choices=["none", "inv_log1p_mean_ema", "cv_ema"], default="none",
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
                        help="Mask augmentation policy. 'xverse' mimics main_mfa/utils_model.py random masking style.")
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
                        help="Use tissue-conditional MFA prior p(z|tissue).")
    parser.add_argument("--lambda-celltype-cls", type=float, default=0.0,
                        help="Weight for auxiliary celltype cross-entropy loss (ignore label -1).")
    parser.add_argument("--celltype-text-embedding-path", default=None,
                        help="Optional .npz/.npy file with cell-type language embeddings. If set, --lambda-celltype-cls uses text-embedding contrastive loss instead of hard CE.")
    parser.add_argument("--celltype-text-temp", type=float, default=0.1,
                        help="Temperature for cell-type text embedding contrastive loss.")
    parser.add_argument("--prior-logvar-min", type=float, default=-4.0,
                        help="Lower clamp bound for MFA prior log-variance used in KL. Prevents tiny prior variance from exploding KL.")
    parser.add_argument("--prior-logvar-max", type=float, default=4.0,
                        help="Upper clamp bound for MFA prior log-variance used in KL.")
    parser.add_argument("--lambda-prior-pi-balance", type=float, default=0.0,
                        help="Weight for balancing global mixture weights toward uniform.")
    parser.add_argument("--lambda-prior-mu-spread", type=float, default=0.0,
                        help="Weight for repulsive regularization between prior component means.")
    parser.add_argument("--prior-mu-spread-tau", type=float, default=1.0,
                        help="Length scale for prior mean spread regularization (smaller => stronger local repulsion).")
    parser.add_argument("--lambda-prior-factor-l2", type=float, default=0.0,
                        help="Weak L2 penalty on MFA prior factor loadings to prevent direction variance blow-up.")
    parser.add_argument("--lambda-prior-logvar-l2", type=float, default=0.0,
                        help="Weak L2 penalty that shrinks MFA diagonal prior log-variance toward --prior-logvar-target.")
    parser.add_argument("--prior-logvar-target", type=float, default=-0.5,
                        help="Target log-variance for --lambda-prior-logvar-l2. -0.5 means variance about 0.61.")
    parser.add_argument("--prior-init-epoch", type=int, default=3,
                        help="If >0, run one-shot MFA prior initialization after this training epoch.")
    parser.add_argument("--prior-init-before-train", action="store_true",
                        help="Run one-shot MFA prior initialization from current encoder embeddings before epoch 1.")
    parser.add_argument("--prior-init-samples", type=int, default=100000,
                        help="Maximum cells used for delayed prior initialization.")
    parser.add_argument("--prior-init-kmeans-iters", type=int, default=20,
                        help="K-means iterations for delayed prior initialization.")
    parser.add_argument("--prior-init-logvar-mode", choices=["cluster", "constant", "shrink"], default="constant",
                        help="How to initialize MFA diagonal prior log-variance after delayed k-means init.")
    parser.add_argument("--prior-init-logvar-value", type=float, default=0.0,
                        help="Baseline prior log-variance used by constant/shrink init modes.")
    parser.add_argument("--prior-init-logvar-shrink-alpha", type=float, default=0.1,
                        help="For --prior-init-logvar-mode shrink: mix cluster variance into baseline variance by this weight.")
    parser.add_argument("--prior-init-logvar-min", type=float, default=-4.0,
                        help="Lower clamp for initialized prior log-variance.")
    parser.add_argument("--prior-init-logvar-max", type=float, default=2.0,
                        help="Upper clamp for initialized prior log-variance.")
    parser.add_argument("--prior-init-factor-pca", action="store_true", default=False,
                        help="Initialize MFA prior factors from per-cluster local PCA.")
    parser.add_argument("--no-prior-init-factor-pca", dest="prior_init_factor_pca", action="store_false",
                        help="Do not initialize MFA factors from local PCA; reset factors to small noise instead.")
    parser.add_argument("--prior-init-factor-scale", type=float, default=0.05,
                        help="Scale applied to PCA-initialized MFA prior factors.")
    parser.add_argument("--prior-init-factor-std", type=float, default=0.01,
                        help="Std for small-random MFA prior factors when PCA init is disabled.")
    parser.add_argument("--prior-freeze-after-init-epochs", type=int, default=0,
                        help="Freeze prior parameters for this many training epochs after delayed prior init.")
    parser.add_argument("--lambda-post-c-balance", type=float, default=0.0,
                        help="Weight for batch-level posterior component usage balance KL(q_mean(c)||uniform).")
    parser.add_argument("--lambda-contrast", type=float, default=1.0,
                        help="Weight of contrastive loss between real-mask and fake-mask views.")
    parser.add_argument("--contrast-view-mode", choices=["real_fake", "random_random"], default="real_fake",
                        help="Contrastive view pairing: real_fake uses original mask vs random mask; random_random uses two independent random masks and averages their reconstruction losses.")
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


def _load_celltype_text_embeddings(path: str, expected_n: int, log):
    if path is None or str(path).strip() == "":
        return None
    fp = os.path.expanduser(str(path))
    if not os.path.exists(fp):
        raise FileNotFoundError(f"celltype text embedding file not found: {fp}")
    if fp.endswith(".npz"):
        data = np.load(fp, allow_pickle=True)
        if "embeddings" not in data:
            raise KeyError(f"{fp} must contain an 'embeddings' array")
        emb = data["embeddings"]
        ids = data["ids"].astype(str).tolist() if "ids" in data else None
        names = data["names"].astype(str).tolist() if "names" in data else None
    else:
        emb = np.load(fp, allow_pickle=True)
        ids = None
        names = None
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"celltype text embeddings must be 2D, got shape={emb.shape}")
    if expected_n > 0 and emb.shape[0] != int(expected_n):
        raise ValueError(
            f"celltype text embedding rows ({emb.shape[0]}) do not match num_cell_types ({expected_n}). "
            "This usually means the compiled dataset was built with a different cell-type map."
        )
    log(f"[CellTypeText] loaded {emb.shape[0]} embeddings, dim={emb.shape[1]} from {fp}")
    if ids is not None and names is not None and len(ids) > 0:
        log(f"[CellTypeText] first={ids[0]}:{names[0]}, last={ids[-1]}:{names[-1]}")
    return torch.from_numpy(emb)


def _apply_training_stage(base_model, stage: str):
    # Single-stage training (former stage3): train all modules jointly.
    _set_requires_grad(getattr(base_model, "encoder", None), True)
    _set_requires_grad(getattr(base_model, "decoder", None), True)
    _set_requires_grad(getattr(base_model, "library_head", None), True)
    _set_requires_grad(getattr(base_model, "prior", None), True)
    _set_requires_grad(getattr(base_model, "post_c_logits", None), True)
    _set_requires_grad(getattr(base_model, "post_u_mu", None), True)
    _set_requires_grad(getattr(base_model, "post_u_logvar", None), True)
    _set_requires_grad(getattr(base_model, "post_eps_mu", None), True)
    _set_requires_grad(getattr(base_model, "post_eps_logvar", None), True)
    _set_requires_grad(getattr(base_model, "score_head", None), True)


def _apply_train_mode(base_model, train_mode: str):
    if str(train_mode) == "prior_only":
        for p in base_model.parameters():
            p.requires_grad = False
        _set_requires_grad(getattr(base_model, "prior", None), True)
        return
    _apply_training_stage(base_model, "stage3")


def _linear_kl_warmup(epoch: int, target_beta: float, warmup_epochs: int, start_beta: float = 0.0) -> float:
    if int(warmup_epochs) <= 0:
        return float(target_beta)
    if int(warmup_epochs) == 1:
        return float(target_beta)
    progress = min(max(float(epoch - 1) / float(max(int(warmup_epochs) - 1, 1)), 0.0), 1.0)
    return float(start_beta) + progress * (float(target_beta) - float(start_beta))


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


def _clear_optimizer_state_for_params(optimizer, params):
    for p in params:
        if p is not None and p in optimizer.state:
            optimizer.state.pop(p, None)


def _is_prior_param_name(name: str) -> bool:
    parts = str(name).split(".")
    while parts and parts[0] == "module":
        parts = parts[1:]
    return bool(parts) and parts[0] == "prior"


def _root_param_name(name: str) -> str:
    parts = str(name).split(".")
    while parts and parts[0] == "module":
        parts = parts[1:]
    return parts[0] if parts else ""


def _optimizer_group_name(name: str) -> str:
    root = _root_param_name(name)
    if root == "prior":
        return "prior"
    if root in {"encoder", "post_c_logits", "post_u_mu", "post_u_logvar", "post_eps_mu", "post_eps_logvar"}:
        return "encoder"
    if root == "decoder":
        return "decoder"
    if root in {"library_head", "nb_log_theta", "nb_theta_decoder"}:
        return "recon_head"
    return "other"


def _build_full_optimizer(
    model,
    lr: float,
    weight_decay: float,
    encoder_lr_multiplier: float,
    prior_lr_multiplier: float,
    decoder_lr_multiplier: float,
    recon_head_lr_multiplier: float,
):
    multipliers = {
        "encoder": float(encoder_lr_multiplier),
        "prior": float(prior_lr_multiplier),
        "decoder": float(decoder_lr_multiplier),
        "recon_head": float(recon_head_lr_multiplier),
        "other": 1.0,
    }
    grouped_params = {key: [] for key in ("encoder", "prior", "decoder", "recon_head", "other")}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grouped_params[_optimizer_group_name(name)].append(p)

    param_groups = []
    for name in ("encoder", "prior", "decoder", "recon_head", "other"):
        params = grouped_params[name]
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "lr": lr * multipliers[name],
                "weight_decay": weight_decay,
                "name": name,
                "lr_multiplier": multipliers[name],
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)


def _restore_optimizer_group_lrs(
    optimizer,
    lr: float,
    encoder_lr_multiplier: float,
    prior_lr_multiplier: float,
    decoder_lr_multiplier: float,
    recon_head_lr_multiplier: float,
):
    multipliers = {
        "encoder": float(encoder_lr_multiplier),
        "prior": float(prior_lr_multiplier),
        "decoder": float(decoder_lr_multiplier),
        "recon_head": float(recon_head_lr_multiplier),
        "other": 1.0,
        "main": 1.0,
    }
    for group in optimizer.param_groups:
        name = group.get("name", "other")
        group["lr"] = lr * multipliers.get(name, 1.0)
        group["lr_multiplier"] = multipliers.get(name, 1.0)


def _format_optimizer_lrs(optimizer) -> str:
    parts = []
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{i}")
        parts.append(f"{name}={float(group['lr']):.6g}")
    return ", ".join(parts)


def _apply_epoch_lr_decay(optimizer, gamma: float, min_lr: float) -> bool:
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError(f"--epoch-lr-gamma must be > 0, got {gamma}")
    if abs(gamma - 1.0) < 1e-12:
        return False
    min_lr = max(0.0, float(min_lr))
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * gamma, min_lr)
    return True


def _build_val_loader_for_epoch(
    val_ds,
    val_collator,
    args,
    val_loader_kwargs,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
    epoch: int,
    log,
):
    val_fraction = float(args.val_fraction)
    if val_fraction <= 0.0 or val_fraction > 1.0:
        raise ValueError(f"--val-fraction must be in (0, 1], got {val_fraction}")
    dataset_for_val = val_ds
    if val_fraction < 1.0:
        n_total = len(val_ds)
        n_keep = max(1, int(math.ceil(float(n_total) * val_fraction)))
        rng = np.random.default_rng(int(args.seed) + int(epoch) * 1009)
        indices = rng.choice(n_total, size=n_keep, replace=False)
        indices.sort()
        dataset_for_val = Subset(val_ds, indices.tolist())
        log(f"[ValSubset] epoch={epoch}, fraction={val_fraction:.4g}, n={n_keep}/{n_total}")
    val_sampler = (
        DistributedSampler(dataset_for_val, num_replicas=world_size, rank=rank, shuffle=False)
        if ddp_enabled
        else None
    )
    if val_sampler is not None:
        val_sampler.set_epoch(epoch)
    val_loader = DataLoader(
        dataset_for_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=val_collator,
        **val_loader_kwargs,
    )
    return val_loader, val_sampler


def _collect_prior_init_embeddings(model, train_loader, device, max_samples: int):
    base_model = _unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    chunks = []
    n_seen = 0
    with torch.no_grad():
        for sample_id, tissue_id, celltype_id, x_count, x_mask, x_mask_encoder in train_loader:
            x_count = x_count.to(device, non_blocking=True)
            x_mask = x_mask.to(device, non_blocking=True)
            # Use the real observed panel for initialization; avoid sampling noise and
            # avoid random-mask views so centers represent the current posterior manifold.
            out = base_model(
                x_count=x_count,
                x_mask=x_mask,
                tissue_id=tissue_id.to(device, non_blocking=True),
                sample_id=None,
                use_batch_condition=False,
            )
            z = out.get("mu_base", out.get("mu")).detach().float()
            take = min(z.size(0), max(0, int(max_samples) - n_seen))
            if take > 0:
                chunks.append(z[:take].cpu())
                n_seen += take
            if n_seen >= int(max_samples):
                break
    if was_training:
        base_model.train()
    if not chunks:
        return torch.empty((0, base_model.latent_dim), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def _gather_init_embeddings(local_z: torch.Tensor, device, rank: int, world_size: int):
    if (not dist.is_available()) or (not dist.is_initialized()) or world_size <= 1:
        return local_z.to(device) if rank == 0 else None
    local_z = local_z.to(device)
    local_n = torch.tensor([local_z.size(0)], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    max_n = int(max(s.item() for s in sizes))
    d = local_z.size(1)
    padded = torch.zeros((max_n, d), device=device, dtype=local_z.dtype)
    if local_z.size(0) > 0:
        padded[: local_z.size(0)] = local_z
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    if rank != 0:
        return None
    parts = [g[: int(sizes[i].item())] for i, g in enumerate(gathered) if int(sizes[i].item()) > 0]
    return torch.cat(parts, dim=0) if parts else torch.empty((0, d), device=device, dtype=local_z.dtype)


def _init_factor_from_cluster(xc: torch.Tensor, rank: int):
    d = xc.size(1)
    if rank <= 0:
        return None
    if xc.size(0) < 2:
        return torch.randn((d, rank), device=xc.device, dtype=xc.dtype) * 0.01
    xc = xc - xc.mean(dim=0, keepdim=True)
    try:
        # xc = U S Vh; covariance eigenvalues are S^2 / (n - 1).
        _, s, vh = torch.linalg.svd(xc, full_matrices=False)
        r = min(rank, vh.size(0), s.numel())
        fac = torch.zeros((d, rank), device=xc.device, dtype=xc.dtype)
        scale = s[:r] / float(max(xc.size(0) - 1, 1)) ** 0.5
        fac[:, :r] = vh[:r].T * scale.view(1, r)
        if r < rank:
            fac[:, r:] = torch.randn((d, rank - r), device=xc.device, dtype=xc.dtype) * 0.01
        return fac
    except RuntimeError:
        return torch.randn((d, rank), device=xc.device, dtype=xc.dtype) * 0.01


def delayed_init_mfa_prior_from_loader(
    model,
    optimizer,
    train_loader,
    device,
    samples: int,
    kmeans_iters: int,
    logvar_mode: str,
    logvar_value: float,
    logvar_shrink_alpha: float,
    logvar_min: float,
    logvar_max: float,
    factor_pca: bool,
    factor_scale: float,
    factor_std: float,
    seed: int,
    rank: int,
    world_size: int,
    log,
):
    base_model = _unwrap_model(model)
    if getattr(base_model, "prior_type", None) != "gmm":
        log("[PriorInit][Skip] prior_type is not gmm.")
        return False
    prior = base_model.prior
    k = int(prior.K)
    d = int(prior.D)
    r = int(getattr(prior, "R", 0))
    local_z = _collect_prior_init_embeddings(model, train_loader, device, max_samples=max(1, int(samples)))
    z = _gather_init_embeddings(local_z, device=device, rank=rank, world_size=world_size)

    if rank == 0:
        if z is None or z.size(0) < k:
            log(f"[PriorInit][Skip] not enough embeddings: n={0 if z is None else z.size(0)}, K={k}")
            ok = torch.tensor([0], device=device, dtype=torch.long)
            centers = torch.zeros((k, d), device=device)
            logvar = torch.zeros((k, d), device=device)
            logits = torch.zeros((k,), device=device)
            factor_new = torch.zeros((k, d, r), device=device) if r > 0 else torch.empty((0,), device=device)
        else:
            z = z.float()
            centers, assign = _kmeans_torch(z, k=k, iters=int(kmeans_iters), seed=int(seed))
            counts = torch.bincount(assign, minlength=k).float()
            pi = torch.clamp(counts / torch.clamp(counts.sum(), min=1.0), min=1e-6)
            pi = pi / pi.sum()
            logits = torch.log(pi)
            logvar_rows = []
            factor_rows = []
            global_var = torch.var(z, dim=0, unbiased=False).clamp_min(1e-6)
            base_var = torch.full((d,), float(math.exp(float(logvar_value))), device=device, dtype=z.dtype)
            shrink_alpha = min(max(float(logvar_shrink_alpha), 0.0), 1.0)
            for kk in range(k):
                members = z[assign == kk]
                if members.size(0) >= 2:
                    raw_var = torch.var(members - centers[kk].view(1, -1), dim=0, unbiased=False).clamp_min(1e-6)
                else:
                    raw_var = global_var
                factor_kk = None
                factor_is_pca = False
                if r > 0:
                    if bool(factor_pca) and members.size(0) >= 2:
                        factor_kk = _init_factor_from_cluster(members, r) * float(factor_scale)
                        factor_is_pca = True
                    else:
                        factor_kk = torch.randn((d, r), device=device, dtype=z.dtype) * float(factor_std)
                if str(logvar_mode) == "constant":
                    diag_var = base_var
                else:
                    var = raw_var
                    if str(logvar_mode) == "shrink":
                        var = (1.0 - shrink_alpha) * base_var + shrink_alpha * var
                    diag_var = var
                if factor_is_pca and factor_kk is not None:
                    factor_diag_var = factor_kk.pow(2).sum(dim=1)
                    diag_var = (diag_var - factor_diag_var).clamp_min(float(math.exp(float(logvar_min))))
                logvar_rows.append(torch.log(diag_var).clamp(float(logvar_min), float(logvar_max)))
                if r > 0:
                    factor_rows.append(factor_kk)
            logvar = torch.stack(logvar_rows, dim=0)
            factor_new = torch.stack(factor_rows, dim=0) if r > 0 else torch.empty((0,), device=device)
            factor_var_mean = float(factor_new.pow(2).sum(dim=1).mean().item()) if r > 0 and factor_new.numel() > 0 else 0.0
            factor_var_max = float(factor_new.pow(2).sum(dim=1).max().item()) if r > 0 and factor_new.numel() > 0 else 0.0
            ok = torch.tensor([1], device=device, dtype=torch.long)
            log(
                f"[PriorInit] done: samples={z.size(0)}, K={k}, "
                f"pi_min={pi.min().item():.4f}, pi_max={pi.max().item():.4f}, "
                f"logvar_min={logvar.min().item():.3f}, logvar_max={logvar.max().item():.3f}, "
                f"logvar_mode={logvar_mode}, logvar_value={float(logvar_value):.3f}, "
                f"logvar_shrink_alpha={shrink_alpha:.3f}, "
                f"factor_pca={bool(factor_pca)}, factor_scale={float(factor_scale):.4f}, "
                f"factor_std={float(factor_std):.4f}, "
                f"factor_var_mean={factor_var_mean:.4g}, factor_var_max={factor_var_max:.4g}"
            )
    else:
        ok = torch.tensor([0], device=device, dtype=torch.long)
        centers = torch.zeros((k, d), device=device)
        logvar = torch.zeros((k, d), device=device)
        logits = torch.zeros((k,), device=device)
        factor_new = torch.zeros((k, d, r), device=device) if r > 0 else torch.empty((0,), device=device)

    if dist.is_available() and dist.is_initialized():
        dist.broadcast(ok, src=0)
        dist.broadcast(centers, src=0)
        dist.broadcast(logvar, src=0)
        dist.broadcast(logits, src=0)
        if r > 0:
            dist.broadcast(factor_new, src=0)
    if int(ok.item()) != 1:
        return False

    with torch.no_grad():
        prior.prior_mu.copy_(centers.to(device=prior.prior_mu.device, dtype=prior.prior_mu.dtype))
        prior.prior_logvar.copy_(logvar.to(device=prior.prior_logvar.device, dtype=prior.prior_logvar.dtype))
        prior.pi_logits.copy_(logits.to(device=prior.pi_logits.device, dtype=prior.pi_logits.dtype))
        if r > 0 and getattr(prior, "prior_factor", None) is not None:
            prior.prior_factor.copy_(factor_new.to(device=prior.prior_factor.device, dtype=prior.prior_factor.dtype))
    _clear_optimizer_state_for_params(
        optimizer,
        [prior.prior_mu, prior.prior_logvar, prior.pi_logits, getattr(prior, "prior_factor", None)],
    )
    log("[PriorInit] broadcast done and optimizer state cleared.")
    return True


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
    resume_ckpt_path = None
    if args.resume_from == "last":
        resume_ckpt_path = ckpt_path
    elif args.resume_from == "best":
        resume_ckpt_path = best_ckpt_path

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
    celltype_text_embeddings = _load_celltype_text_embeddings(
        args.celltype_text_embedding_path,
        expected_n=num_cell_types,
        log=log,
    )

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
    if args.prior_shared_cov:
        log("[MFA] Ignoring --prior-shared-cov: main_mfa uses component-specific factor covariances.")
    log(f"[MFA] explicit latent structure: c ~ Cat(pi), u_dim={args.prior_cov_rank}, z = mu_c + A_c u + eps_c.")

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    val_num_workers = min(args.num_workers, 4) if args.val_num_workers is None else max(0, int(args.val_num_workers))
    val_loader_kwargs = dict(num_workers=val_num_workers, pin_memory=True)
    if val_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader_kwargs["persistent_workers"] = bool(args.val_persistent_workers)
    log(
        f"[Loader] train_workers={args.num_workers}, train_persistent={args.persistent_workers}, "
        f"val_workers={val_num_workers}, val_persistent={val_loader_kwargs.get('persistent_workers', False)}"
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
        prior_shared_covariance=False,
        posterior_cov_rank=args.posterior_cov_rank,
        prior_mu_init=args.prior_mu_init,
        prior_mu_init_radius=args.prior_mu_init_radius,
        prior_mu_init_groups=args.prior_mu_init_groups,
        prior_mu_init_local_radius=args.prior_mu_init_local_radius,
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
        nb_theta_mode=args.nb_theta_mode,
        celltype_text_embeddings=celltype_text_embeddings,
        celltype_text_temperature=args.celltype_text_temp,
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
    optimizer = _build_full_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder_lr_multiplier=args.encoder_lr_multiplier,
        prior_lr_multiplier=args.prior_lr_multiplier,
        decoder_lr_multiplier=args.decoder_lr_multiplier,
        recon_head_lr_multiplier=args.recon_head_lr_multiplier,
    )
    if torch.cuda.is_available():
        log("[Optimizer] Using standard Adam (AMP-compatible).")
    log(f"[Optimizer] lr groups: {_format_optimizer_lrs(optimizer)}")
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
    prior_initialized = False
    prior_freeze_until_epoch = 0

    if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        map_location = device
        ckpt = torch.load(resume_ckpt_path, map_location=map_location)
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
                _restore_optimizer_group_lrs(
                    optimizer,
                    args.lr,
                    args.encoder_lr_multiplier,
                    args.prior_lr_multiplier,
                    args.decoder_lr_multiplier,
                    args.recon_head_lr_multiplier,
                )
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
        prior_initialized = bool(ckpt.get("prior_initialized", False))
        prior_freeze_until_epoch = int(ckpt.get("prior_freeze_until_epoch", 0))
        if int(args.prior_freeze_after_init_epochs) <= 0 and prior_freeze_until_epoch > 0:
            log(
                f"[Resume] Clear checkpoint prior_freeze_until_epoch={prior_freeze_until_epoch} "
                "because --prior-freeze-after-init-epochs <= 0."
            )
            prior_freeze_until_epoch = 0
        last_epoch = int(ckpt.get("epoch", 0))
        start_round = last_epoch + 1
        log(
            f"[Resume] Loaded {resume_ckpt_path} via --resume-from {args.resume_from} "
            f"(epoch={last_epoch}, best_val_metric={best_val_metric:.6f}). "
            f"Continue from epoch {start_round}."
        )
        if getattr(load_ret, "missing_keys", None):
            log(f"[Resume] Missing keys (expected with new heads): {load_ret.missing_keys}")
        if getattr(load_ret, "unexpected_keys", None):
            log(f"[Resume] Unexpected keys: {load_ret.unexpected_keys}")
    elif resume_ckpt_path is not None and args.resume_from == "best":
        raise FileNotFoundError(f"--resume-from best requested but checkpoint not found: {resume_ckpt_path}")
    elif args.init_ckpt:
        if not os.path.exists(args.init_ckpt):
            raise FileNotFoundError(f"--init-ckpt not found: {args.init_ckpt}")
        map_location = device
        ckpt = torch.load(args.init_ckpt, map_location=map_location)
        ckpt_state = ckpt.get("model_state_dict", ckpt)
        filtered_state, skipped = _filter_state_dict_by_shape(model, ckpt_state)
        load_ret = model.load_state_dict(filtered_state, strict=False)
        if skipped:
            log(f"[InitCkpt] Skipped {len(skipped)} incompatible/missing keys from checkpoint.")
            for name, reason in skipped[:20]:
                log(f"[InitCkpt][Skip] {name}: {reason}")
            if len(skipped) > 20:
                log(f"[InitCkpt] ... and {len(skipped) - 20} more skipped keys.")
        log(f"[InitCkpt] Loaded model weights from {args.init_ckpt}. Training starts from epoch 1 in {args.result_dir}.")
        if getattr(load_ret, "missing_keys", None):
            log(f"[InitCkpt] Missing keys: {load_ret.missing_keys}")
        if getattr(load_ret, "unexpected_keys", None):
            log(f"[InitCkpt] Unexpected keys: {load_ret.unexpected_keys}")

    base_model = _unwrap_model(model)
    _apply_train_mode(base_model, args.train_mode)
    if args.train_mode == "prior_only":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("--train-mode prior_only selected but no trainable prior parameters were found.")
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
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
        total_params, trainable_count = count_parameters(base_model)
        log(f"[TrainMode] prior_only: trainable parameters reset to {trainable_count:,}/{total_params:,}; optimizer rebuilt.")
        log(f"[Optimizer] lr groups: {_format_optimizer_lrs(optimizer)}")
    else:
        log("[TrainMode] full")

    if bool(args.prior_init_before_train) and (not prior_initialized):
        log("[PriorInit] Triggered before training")
        if train_sampler is not None:
            train_sampler.set_epoch(0)
        initialized = delayed_init_mfa_prior_from_loader(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            samples=args.prior_init_samples,
            kmeans_iters=args.prior_init_kmeans_iters,
            logvar_mode=args.prior_init_logvar_mode,
            logvar_value=args.prior_init_logvar_value,
            logvar_shrink_alpha=args.prior_init_logvar_shrink_alpha,
            logvar_min=args.prior_init_logvar_min,
            logvar_max=args.prior_init_logvar_max,
            factor_pca=args.prior_init_factor_pca,
            factor_scale=args.prior_init_factor_scale,
            factor_std=args.prior_init_factor_std,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            log=log,
        )
        prior_initialized = bool(prior_initialized or initialized)

    epoch_id = start_round
    vae_pretrain_done = int(args.vae_pretrain_epochs) <= 0 or start_round > int(args.vae_pretrain_epochs)
    if int(args.vae_pretrain_epochs) > 0:
        log(f"[VAEPretrain] enabled for first {int(args.vae_pretrain_epochs)} epoch(s).")

    while epoch_id <= args.num_epochs:
        start_time = time.time()
        log(f"\n[Epoch {epoch_id}] Starting...")
        beta_t = _linear_kl_warmup(
            epoch=epoch_id,
            target_beta=args.beta_kl,
            warmup_epochs=args.beta_kl_warmup_epochs,
            start_beta=args.beta_kl_warmup_start,
        )
        base_model = _unwrap_model(model)
        in_vae_pretrain = (
            getattr(base_model, "prior_type", None) == "gmm"
            and int(args.vae_pretrain_epochs) > 0
            and epoch_id <= int(args.vae_pretrain_epochs)
        )
        if in_vae_pretrain and args.vae_pretrain_beta_kl is not None:
            beta_t = float(args.vae_pretrain_beta_kl)
        stage_name = "vae_pretrain" if in_vae_pretrain else "stage3"
        _apply_train_mode(base_model, args.train_mode)
        if prior_initialized and int(prior_freeze_until_epoch) >= epoch_id:
            _set_requires_grad(getattr(base_model, "prior", None), False)
            log(f"[PriorFreeze] prior frozen for adaptation epoch {epoch_id}/{prior_freeze_until_epoch}")

        # Default: global schedules.
        lambda_resp_anchor_t = 0.0

        log(
            f"[Epoch {epoch_id}] stage={stage_name}, beta_kl={beta_t:.6f}, "
            f"lambda_resp_anchor={lambda_resp_anchor_t:.6f}"
        )
        prior_snapshot_start = prior_parameter_snapshot(model)
        if prior_snapshot_start and is_main_process(rank):
            log(f"[Epoch {epoch_id}] PriorSnapshot captured keys={','.join(sorted(prior_snapshot_start.keys()))}")

        force_base_posterior = bool(in_vae_pretrain)

        train_sampler.set_epoch(epoch_id)
        loss_full, loss_recon, loss_kl, loss_score, loss_contrast, loss_cov, loss_prior_pi_balance, loss_celltype_cls, loss_batchless_recon = train_gmm_vae_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            device=device,
            beta_kl=beta_t,
            beta_u_kl_multiplier=args.beta_u_kl_multiplier,
            beta_eps_kl_multiplier=args.beta_eps_kl_multiplier,
            recon_observed_only=args.recon_observed_only,
            mask_aug_prob=args.mask_aug_prob,
            mask_aug_policy=args.mask_aug_policy,
            mask_aug_min_frac=args.mask_aug_min_frac,
            mask_aug_max_frac=args.mask_aug_max_frac,
            lambda_contrast=args.lambda_contrast,
            contrast_temp=args.contrast_temp,
            contrast_view_mode=args.contrast_view_mode,
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
            prior_logvar_min=args.prior_logvar_min,
            prior_logvar_max=args.prior_logvar_max,
            lambda_prior_mu_l2=0.0,
            lambda_prior_factor_l2=args.lambda_prior_factor_l2,
            lambda_prior_pi_balance=args.lambda_prior_pi_balance,
            lambda_prior_mu_spread=args.lambda_prior_mu_spread,
            prior_mu_spread_tau=args.prior_mu_spread_tau,
            lambda_post_c_balance=args.lambda_post_c_balance,
            lambda_celltype_cls=args.lambda_celltype_cls,
            lambda_prior_logvar_l2=args.lambda_prior_logvar_l2,
            prior_logvar_target=args.prior_logvar_target,
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
            prior_snapshot_start=prior_snapshot_start,
            log_every=args.log_every,
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
        if prior_snapshot_start:
            train_msg += ", " + format_prior_delta(prior_parameter_delta(model, prior_snapshot_start))
        log(train_msg)

        if (
            int(args.prior_init_epoch) > 0
            and (not prior_initialized)
            and (not in_vae_pretrain)
            and epoch_id >= int(args.prior_init_epoch)
        ):
            log(f"[PriorInit] Triggered after epoch {epoch_id}")
            initialized = delayed_init_mfa_prior_from_loader(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                samples=args.prior_init_samples,
                kmeans_iters=args.prior_init_kmeans_iters,
                logvar_mode=args.prior_init_logvar_mode,
                logvar_value=args.prior_init_logvar_value,
                logvar_shrink_alpha=args.prior_init_logvar_shrink_alpha,
                logvar_min=args.prior_init_logvar_min,
                logvar_max=args.prior_init_logvar_max,
                factor_pca=args.prior_init_factor_pca,
                factor_scale=args.prior_init_factor_scale,
                factor_std=args.prior_init_factor_std,
                seed=args.seed + epoch_id,
                rank=rank,
                world_size=world_size,
                log=log,
            )
            prior_initialized = bool(prior_initialized or initialized)
            if initialized and int(args.prior_freeze_after_init_epochs) > 0:
                prior_freeze_until_epoch = epoch_id + int(args.prior_freeze_after_init_epochs)
                log(f"[PriorFreeze] prior will be frozen through epoch {prior_freeze_until_epoch}")

        if (
            int(args.vae_pretrain_epochs) > 0
            and (not vae_pretrain_done)
            and epoch_id >= int(args.vae_pretrain_epochs)
        ):
            log(f"[VAEPretrain] completed at epoch {epoch_id}; initializing MFA prior from base posterior.")
            initialized = delayed_init_mfa_prior_from_loader(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                samples=args.prior_init_samples,
                kmeans_iters=args.prior_init_kmeans_iters,
                logvar_mode=args.prior_init_logvar_mode,
                logvar_value=args.prior_init_logvar_value,
                logvar_shrink_alpha=args.prior_init_logvar_shrink_alpha,
                logvar_min=args.prior_init_logvar_min,
                logvar_max=args.prior_init_logvar_max,
                factor_pca=args.prior_init_factor_pca,
                factor_scale=args.prior_init_factor_scale,
                factor_std=args.prior_init_factor_std,
                seed=args.seed + epoch_id,
                rank=rank,
                world_size=world_size,
                log=log,
            )
            prior_initialized = bool(prior_initialized or initialized)
            vae_pretrain_done = True
            if initialized and int(args.prior_freeze_after_init_epochs) > 0:
                prior_freeze_until_epoch = epoch_id + int(args.prior_freeze_after_init_epochs)
                log(f"[PriorFreeze] prior will be frozen through epoch {prior_freeze_until_epoch}")

        do_val = (
            (int(args.val_every) <= 1)
            or (epoch_id % int(args.val_every) == 0)
            or (epoch_id == args.num_epochs)
        )
        if do_val:
            val_loader, _ = _build_val_loader_for_epoch(
                val_ds=val_ds,
                val_collator=val_collator,
                args=args,
                val_loader_kwargs=val_loader_kwargs,
                ddp_enabled=ddp_enabled,
                world_size=world_size,
                rank=rank,
                epoch=epoch_id,
                log=log,
            )
            val_loss_full, val_loss_recon, val_loss_kl, val_loss_score, val_loss_contrast, val_loss_cov, val_loss_prior_pi_balance, val_loss_celltype_cls, val_loss_batchless_recon = evaluate_gmm_vae_one_epoch(
                model=model,
                val_loader=val_loader,
                device=device,
                beta_kl=beta_t,
                beta_u_kl_multiplier=args.beta_u_kl_multiplier,
                beta_eps_kl_multiplier=args.beta_eps_kl_multiplier,
                recon_observed_only=args.recon_observed_only,
                lambda_score=0.0,
                score_noise_std=0.0,
                score_detach_z=True,
                lambda_contrast=args.lambda_contrast,
                contrast_temp=args.contrast_temp,
                contrast_view_mode=args.contrast_view_mode,
                lambda_real_recon=args.lambda_real_recon,
                lambda_cov=0.0,
                cov_use_mu=True,
                lambda_resp_balance=0.0,
                lambda_resp_confidence=0.0,
                lambda_resp_anchor=lambda_resp_anchor_t,
                resp_temperature=1.0,
                resp_topk=0,
                prior_logvar_min=args.prior_logvar_min,
                prior_logvar_max=args.prior_logvar_max,
                lambda_prior_mu_l2=0.0,
                lambda_prior_factor_l2=args.lambda_prior_factor_l2,
                lambda_prior_pi_balance=args.lambda_prior_pi_balance,
                lambda_prior_mu_spread=args.lambda_prior_mu_spread,
                prior_mu_spread_tau=args.prior_mu_spread_tau,
                lambda_post_c_balance=args.lambda_post_c_balance,
                lambda_celltype_cls=args.lambda_celltype_cls,
                lambda_prior_logvar_l2=args.lambda_prior_logvar_l2,
                prior_logvar_target=args.prior_logvar_target,
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
                log_every=args.log_every,
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

            if val_metric < best_val_metric and is_main_process(rank):
                best_val_metric = val_metric
                torch.save({
                    "epoch": epoch_id,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_metric": best_val_metric,
                    "prior_initialized": prior_initialized,
                    "prior_freeze_until_epoch": prior_freeze_until_epoch,
                    "args": vars(args),
                }, best_ckpt_path)
                log(f"[Best Model] Updated at epoch {epoch_id} with metric={val_metric:.4f}")
        else:
            log(f"[Epoch {epoch_id}] Skip validation (val_every={args.val_every}).")

        if _apply_epoch_lr_decay(optimizer, args.epoch_lr_gamma, args.scheduler_min_lr):
            log(
                f"[Epoch {epoch_id}] Applied epoch LR decay gamma={args.epoch_lr_gamma:g}; "
                f"next Learning Rate: {_format_optimizer_lrs(optimizer)}"
            )
        else:
            log(f"[Epoch {epoch_id}] Current Learning Rate: {_format_optimizer_lrs(optimizer)}")

        if is_main_process(rank):
            torch.save({
                "epoch": epoch_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_metric": best_val_metric,
                "prior_initialized": prior_initialized,
                "prior_freeze_until_epoch": prior_freeze_until_epoch,
                "args": vars(args),
            }, ckpt_path)
            log(f"[Checkpoint] Saved as {args.last_ckpt_name} at epoch {epoch_id}")

        log(f"[Epoch {epoch_id}] Time elapsed: {time.time() - start_time:.2f}s")
        epoch_id += 1

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
