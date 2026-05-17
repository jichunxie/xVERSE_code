#!/usr/bin/env python3
"""Diagnostics for standard-VAE embeddings from main_mfa checkpoints.

This script is intentionally posterior/embedding focused. For --prior-type gaussian,
GMM/MFA prior parameters are unused, so the useful diagnostics are mu_base geometry,
posterior uncertainty, KL per cell, and real-mask vs random-mask view stability.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_mfa.utils_model import (  # noqa: E402
    CompiledShardDataset,
    CompiledSparseBatchCollator,
    MaskFiLMGMMVAE,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Diagnose VAE mu_base embedding geometry and mask invariance.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--compiled-dataset-root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-cells", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--compiled-max-cached-shards", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-aug-min-frac", type=float, default=0.1)
    ap.add_argument("--mask-aug-max-frac", type=float, default=0.5)
    ap.add_argument("--knn-k", type=int, default=30)
    ap.add_argument("--plot-max-cells", type=int, default=20000)
    return ap.parse_args()


def _filter_state_dict_by_shape(model, state):
    model_state = model.state_dict()
    kept = {}
    skipped = []
    for k, v in state.items():
        k2 = k.replace("module.", "")
        if k2 not in model_state:
            skipped.append((k2, "missing_in_model"))
            continue
        if tuple(model_state[k2].shape) != tuple(v.shape):
            skipped.append((k2, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_state[k2].shape)}"))
            continue
        kept[k2] = v
    return kept, skipped


def build_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    num_batches = int(args.get("num_batches", 0))
    if num_batches <= 0 and "batch_embedding.weight" in state:
        num_batches = int(state["batch_embedding.weight"].shape[0])
    batch_emb_dim = int(args.get("batch_emb_dim", 0))
    if batch_emb_dim <= 0 and "batch_embedding.weight" in state:
        batch_emb_dim = int(state["batch_embedding.weight"].shape[1])

    model = MaskFiLMGMMVAE(
        num_genes=int(args.get("total_gene", 17999)),
        latent_dim=int(args.get("latent_dim", 128)),
        num_components=int(args.get("num_components", 16)),
        prior_cov_rank=int(args.get("prior_cov_rank", 8)),
        prior_shared_covariance=bool(args.get("prior_shared_cov", False)),
        posterior_cov_rank=int(args.get("posterior_cov_rank", 0)),
        prior_mu_init=str(args.get("prior_mu_init", "normal")),
        prior_mu_init_radius=float(args.get("prior_mu_init_radius", 1.0)),
        prior_mu_init_groups=int(args.get("prior_mu_init_groups", 8)),
        prior_mu_init_local_radius=float(args.get("prior_mu_init_local_radius", 0.5)),
        expr_hidden_dim=int(args.get("expr_hidden_dim", 1024)),
        mask_hidden_dim=int(args.get("mask_hidden_dim", 512)),
        dec_hidden_dim=int(args.get("dec_hidden_dim", 1024)),
        dropout=float(args.get("dropout", 0.1)),
        prior_type=str(args.get("prior_type", "gaussian")),
        num_cell_types=int(args.get("num_cell_types", 0)),
        conditional_prior_on_tissue=bool(args.get("conditional_prior_on_tissue", False)),
        num_tissues=int(args.get("num_tissues", 0)),
        num_batches=num_batches,
        batch_emb_dim=batch_emb_dim,
        batch_cond_drop_prob=0.0,
        recon_loss_type=str(args.get("recon_loss", args.get("recon_loss_type", "nb"))),
    ).to(device)
    kept, skipped = _filter_state_dict_by_shape(model, state)
    ret = model.load_state_dict(kept, strict=False)
    model.eval()
    print(f"[Load] ckpt={ckpt_path}")
    print(f"[Load] prior_type={getattr(model, 'prior_type', None)}, kept={len(kept)}, skipped={len(skipped)}, missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")
    if skipped:
        print("[Load] first skipped:", skipped[:5])
    return model, args


def random_hide_observed(x_mask, min_frac=0.1, max_frac=0.5, seed=0):
    g = torch.Generator(device=x_mask.device)
    g.manual_seed(int(seed))
    enc = x_mask.clone().bool()
    bsz = enc.size(0)
    for i in range(bsz):
        obs = torch.nonzero(enc[i], as_tuple=False).view(-1)
        n_obs = int(obs.numel())
        if n_obs <= 1:
            continue
        frac = float(min_frac) + (float(max_frac) - float(min_frac)) * float(torch.rand((), generator=g, device=x_mask.device).item())
        n_hide = max(1, min(int(n_obs * frac), n_obs - 1))
        perm = torch.randperm(n_obs, generator=g, device=x_mask.device)[:n_hide]
        enc[i, obs[perm]] = False
    return enc


def pca2(x):
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T


def embed2d(x, seed=0):
    try:
        import umap
        n_neighbors = min(30, max(5, x.shape[0] // 100))
        emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.2, metric="euclidean", random_state=int(seed)).fit_transform(x)
        return np.asarray(emb), "umap"
    except Exception as exc:
        print(f"[WARN] UMAP failed, using PCA: {exc}")
        return pca2(x), "pca"


def knn_density(x, k=30):
    try:
        from sklearn.neighbors import NearestNeighbors
        kk = min(max(2, int(k) + 1), x.shape[0])
        nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
        nn.fit(x)
        dist, _ = nn.kneighbors(x)
        kth = dist[:, -1]
        return 1.0 / np.clip(kth, 1e-8, None), kth
    except Exception as exc:
        print(f"[WARN] kNN density failed: {exc}")
        return np.full((x.shape[0],), np.nan), np.full((x.shape[0],), np.nan)


def save_scatter(path, xy, color, title, colorbar_label, cmap="viridis", categorical=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    if categorical:
        vals = pd.Series(color).astype("category").cat.codes.to_numpy()
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=3, cmap="tab20", alpha=0.65, linewidth=0)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=color, s=3, cmap=cmap, alpha=0.7, linewidth=0)
        fig.colorbar(sc, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model, saved_args = build_model_from_ckpt(args.ckpt, device)
    with open(outdir / "ckpt_args.json", "w") as f:
        json.dump(saved_args, f, indent=2, sort_keys=True, default=str)

    manifest_path = Path(args.compiled_dataset_root) / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    num_genes = int(manifest.get("global_num_genes", saved_args.get("total_gene", 17999)))
    ds = CompiledShardDataset(args.compiled_dataset_root, split=args.split, max_cached_shards=args.compiled_max_cached_shards)
    n = len(ds)
    rng = np.random.default_rng(args.seed)
    if args.max_cells > 0 and n > args.max_cells:
        indices = rng.choice(n, size=int(args.max_cells), replace=False)
        indices.sort()
    else:
        indices = np.arange(n)
    print(f"[Data] split={args.split}, total={n}, sampled={len(indices)}")

    collator = CompiledSparseBatchCollator(num_genes=num_genes, apply_mask_aug=False)
    loader = DataLoader(
        Subset(ds, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    rows = []
    mu_chunks = []
    logvar_chunks = []
    mu_masked_chunks = []
    with torch.no_grad():
        cursor = 0
        for batch_idx, (sample_id, tissue_id, celltype_id, x_count, x_mask, x_mask_encoder) in enumerate(loader):
            x_count = x_count.to(device, non_blocking=True)
            x_mask = x_mask.to(device, non_blocking=True)
            sample_id_d = sample_id.to(device, non_blocking=True)
            tissue_id_d = tissue_id.to(device, non_blocking=True)
            out_real = model(x_count=x_count, x_mask=x_mask, tissue_id=tissue_id_d, sample_id=sample_id_d, use_batch_condition=False)
            x_mask_aug = random_hide_observed(x_mask, min_frac=args.mask_aug_min_frac, max_frac=args.mask_aug_max_frac, seed=args.seed + batch_idx)
            out_mask = model(x_count=x_count, x_mask=x_mask_aug, tissue_id=tissue_id_d, sample_id=sample_id_d, use_batch_condition=False)

            mu = out_real["mu_base"].detach().float()
            logvar = out_real.get("logvar", out_real["mu_base"] * 0).detach().float()
            mu_masked = out_mask["mu_base"].detach().float()
            kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
            cos = torch.nn.functional.cosine_similarity(mu, mu_masked, dim=-1)
            l2 = torch.norm(mu - mu_masked, dim=-1)
            mean_logvar = logvar.mean(dim=-1)
            mean_std = torch.exp(0.5 * logvar).mean(dim=-1)
            mu_norm = torch.norm(mu, dim=-1)
            mask_frac = x_mask.float().mean(dim=-1)
            aug_frac = x_mask_aug.float().mean(dim=-1)

            bsz = mu.size(0)
            global_idx = indices[cursor: cursor + bsz]
            cursor += bsz
            batch_df = pd.DataFrame({
                "global_index": global_idx,
                "sample_id": sample_id.cpu().numpy(),
                "tissue_id": tissue_id.cpu().numpy(),
                "celltype_id": celltype_id.cpu().numpy(),
                "kl": kl.cpu().numpy(),
                "mu_norm": mu_norm.cpu().numpy(),
                "mean_logvar": mean_logvar.cpu().numpy(),
                "mean_posterior_std": mean_std.cpu().numpy(),
                "real_mask_frac": mask_frac.cpu().numpy(),
                "masked_mask_frac": aug_frac.cpu().numpy(),
                "view_cosine": cos.cpu().numpy(),
                "view_l2": l2.cpu().numpy(),
            })
            rows.append(batch_df)
            mu_chunks.append(mu.cpu().numpy())
            logvar_chunks.append(logvar.cpu().numpy())
            mu_masked_chunks.append(mu_masked.cpu().numpy())
            if (batch_idx + 1) % 20 == 0:
                print(f"[Batch {batch_idx + 1}] collected={cursor}")

    cell_df = pd.concat(rows, ignore_index=True)
    mu_arr = np.concatenate(mu_chunks, axis=0)
    logvar_arr = np.concatenate(logvar_chunks, axis=0)
    density, kth_dist = knn_density(mu_arr, k=args.knn_k)
    cell_df["knn_density"] = density
    cell_df["knn_kth_dist"] = kth_dist
    cell_df.to_csv(outdir / "vae_cell_diagnostics.csv", index=False)

    summary = []
    for col in ["kl", "mu_norm", "mean_logvar", "mean_posterior_std", "view_cosine", "view_l2", "knn_density", "knn_kth_dist"]:
        vals = pd.to_numeric(cell_df[col], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        summary.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p01": float(np.quantile(vals, 0.01)),
            "p05": float(np.quantile(vals, 0.05)),
            "p50": float(np.quantile(vals, 0.50)),
            "p95": float(np.quantile(vals, 0.95)),
            "p99": float(np.quantile(vals, 0.99)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    pd.DataFrame(summary).to_csv(outdir / "vae_embedding_summary.csv", index=False)

    dim_df = pd.DataFrame({
        "dim": np.arange(mu_arr.shape[1]),
        "mu_mean": mu_arr.mean(axis=0),
        "mu_std": mu_arr.std(axis=0),
        "logvar_mean": logvar_arr.mean(axis=0),
        "logvar_std": logvar_arr.std(axis=0),
    })
    dim_df.to_csv(outdir / "vae_latent_dim_summary.csv", index=False)

    plot_n = min(int(args.plot_max_cells), mu_arr.shape[0])
    plot_idx = rng.choice(mu_arr.shape[0], size=plot_n, replace=False) if mu_arr.shape[0] > plot_n else np.arange(mu_arr.shape[0])
    xy, method = embed2d(mu_arr[plot_idx], seed=args.seed)
    plot_df = cell_df.iloc[plot_idx].reset_index(drop=True).copy()
    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]
    plot_df.to_csv(outdir / f"vae_mu_base_{method}_coords.csv", index=False)

    save_scatter(outdir / f"vae_mu_base_{method}_density.png", xy, plot_df["knn_density"].to_numpy(), f"VAE mu_base {method.upper()} colored by kNN density", "1 / kth NN distance")
    save_scatter(outdir / f"vae_mu_base_{method}_kl.png", xy, plot_df["kl"].to_numpy(), f"VAE mu_base {method.upper()} colored by KL", "KL per cell", cmap="magma")
    save_scatter(outdir / f"vae_mu_base_{method}_view_cosine.png", xy, plot_df["view_cosine"].to_numpy(), f"VAE mu_base {method.upper()} colored by real-vs-mask cosine", "cosine", cmap="viridis")
    save_scatter(outdir / f"vae_mu_base_{method}_celltype.png", xy, plot_df["celltype_id"].to_numpy(), f"VAE mu_base {method.upper()} colored by celltype id", "celltype_id", categorical=True)
    save_scatter(outdir / f"vae_mu_base_{method}_sample.png", xy, plot_df["sample_id"].to_numpy(), f"VAE mu_base {method.upper()} colored by sample id", "sample_id", categorical=True)
    save_scatter(outdir / f"vae_mu_base_{method}_tissue.png", xy, plot_df["tissue_id"].to_numpy(), f"VAE mu_base {method.upper()} colored by tissue id", "tissue_id", categorical=True)

    print(f"[Done] wrote diagnostics to {outdir}")


if __name__ == "__main__":
    main()
