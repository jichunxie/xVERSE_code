#!/usr/bin/env python3
"""
Extract current xVERSE embeddings for fig2 biology signal datasets.

This mirrors the original fig2 xVerse embedding extraction flow but uses:
- main_energy or main_mfa MaskFiLMGMMVAE checkpoint
- same donor h5ad files and gene-set splits
- output embedding key: obsm["xVERSE"] by default
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repo root is importable when running as a script path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.utils_ft import XVerseFineTuneDataset
from main.utils_model import load_gene_ids
from main_energy.utils_model import MaskFiLMGMMVAE as EnergyMaskFiLMGMMVAE
from main_mfa.utils_model import MaskFiLMGMMVAE as MFAMaskFiLMGMMVAE


def parse_args():
    ap = argparse.ArgumentParser(description="Extract current xVERSE embeddings for fig2 donor datasets.")
    ap.add_argument(
        "--gene-ids-path",
        default="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt",
    )
    ap.add_argument(
        "--ckpt",
        default="/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue3/last_model.pth",
        help="Path to trained xVERSE checkpoint.",
    )
    ap.add_argument("--liver-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/liver")
    ap.add_argument("--brain-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/brain")
    ap.add_argument(
        "--dataset-manifest",
        default=None,
        help=(
            "Optional CSV for gold-standard h5ad extraction. Required columns: dataset,path. "
            "Optional columns: tissue,gene_id_col,count_layer. When set, liver/brain dirs are ignored."
        ),
    )
    ap.add_argument("--default-gene-id-col", default="gene_ids", help="Default adata.var column containing Ensembl IDs.")
    ap.add_argument("--output-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2_gmmvae_current")
    ap.add_argument("--embedding-key", default="xVERSE")
    ap.add_argument(
        "--embedding-mode",
        default="mixmu",
        choices=["mixmu", "encoder_hidden", "z", "mu_base"],
        help=(
            "Which representation to write. mixmu uses sum_k q(c|x) posterior_mu_k; "
            "encoder_hidden uses the encoder hidden state before posterior heads."
        ),
    )
    ap.add_argument(
        "--model-family",
        default="auto",
        choices=["auto", "main_energy", "main_mfa"],
        help="Which model implementation to instantiate. auto detects explicit-MFA checkpoints from state_dict keys.",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-files-per-set", type=int, default=0, help="0 means all files.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-prior-viz", action="store_true", help="Disable prior visualization before embedding extraction.")
    ap.add_argument("--prior-viz-dir", default=None, help="Directory for prior visualization outputs. Defaults to output-dir/prior_viz.")
    ap.add_argument("--prior-viz-max-components", type=int, default=32, help="Max top-pi components for factor-arrow overlay.")
    ap.add_argument(
        "--prior-viz-active-min-fold",
        type=float,
        default=1.5,
        help="Draw active components with pi >= min(pi) * this fold in prior figures.",
    )
    ap.add_argument("--prior-viz-factor-scale", type=float, default=1.0, help="Scale for projected MFA factor arrows.")
    ap.add_argument("--prior-viz-grid", type=int, default=120, help="Grid size for projected prior density surface.")
    ap.add_argument("--prior-viz-sample-n", type=int, default=20000, help="Number of true high-dimensional prior samples for UMAP/KDE visualization.")
    ap.add_argument(
        "--prior-viz-sample-embed",
        default="auto",
        choices=["auto", "umap", "mds", "pca"],
        help="2D embedding for high-dimensional prior samples. auto uses UMAP if available, else PCA.",
    )
    return ap.parse_args()


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_model_family(state, requested: str) -> str:
    if requested != "auto":
        return requested
    keys = set(state.keys())
    if any(k.startswith(("post_u_mu.", "post_u_logvar.", "post_eps_mu.", "post_eps_logvar.")) for k in keys):
        return "main_mfa"
    return "main_energy"


def _load_celltype_text_embeddings_from_args(saved_args: dict, state: dict):
    path = saved_args.get("celltype_text_embedding_path", None)
    if path is None or str(path).strip() == "":
        if "celltype_text_embeddings" not in state:
            return None
        # Older or moved checkpoints may include the buffer but not the path.
        return state["celltype_text_embeddings"].detach().cpu().float()
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        if "celltype_text_embeddings" in state:
            print(f"[Load][WARN] celltype text embedding path missing, using checkpoint buffer: {path}")
            return state["celltype_text_embeddings"].detach().cpu().float()
        raise FileNotFoundError(f"celltype text embedding file not found: {path}")
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "embeddings" not in data:
            raise KeyError(f"{path} must contain an 'embeddings' array")
        emb = data["embeddings"]
    else:
        emb = np.load(path, allow_pickle=True)
    emb = np.asarray(emb, dtype=np.float32)
    print(f"[Load] celltype text embeddings shape={emb.shape} from {path}")
    return torch.from_numpy(emb)


def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    prior_shared_cov = bool(saved_args.get("prior_shared_cov", saved_args.get("prior_shared_covariance", False)))
    num_batches = int(saved_args.get("num_batches", 0))
    if num_batches <= 0 and "batch_embedding.weight" in state:
        num_batches = int(state["batch_embedding.weight"].shape[0])
    num_tissues = int(saved_args.get("num_tissues", 0))
    if num_tissues <= 0 and "tissue_embedding.weight" in state:
        num_tissues = int(state["tissue_embedding.weight"].shape[0])
    batch_emb_dim = int(saved_args.get("batch_emb_dim", 0))
    if batch_emb_dim <= 0 and "batch_embedding.weight" in state:
        batch_emb_dim = int(state["batch_embedding.weight"].shape[1])
    tissue_emb_dim = int(saved_args.get("tissue_emb_dim", 0))
    if tissue_emb_dim <= 0 and "tissue_embedding.weight" in state:
        tissue_emb_dim = int(state["tissue_embedding.weight"].shape[1])
    # Backward/transition compatibility: some checkpoints were saved before
    # tissue_emb_dim was recorded in args, but decoder FiLM already includes it.
    cond_dim_from_decoder = None
    for key in ("decoder.film1.weight", "decoder.film2.weight"):
        if key in state:
            cond_dim_from_decoder = int(state[key].shape[1])
            break
    if cond_dim_from_decoder is not None:
        inferred_tissue_dim = max(0, cond_dim_from_decoder - batch_emb_dim)
        if tissue_emb_dim <= 0 and inferred_tissue_dim > 0:
            tissue_emb_dim = inferred_tissue_dim
            print(f"[Load] inferred tissue_emb_dim={tissue_emb_dim} from decoder cond_dim={cond_dim_from_decoder}")
    model_family = detect_model_family(state, str(saved_args.get("model_family", "auto")))
    if getattr(build_model_from_ckpt, "_requested_family", "auto") != "auto":
        model_family = getattr(build_model_from_ckpt, "_requested_family")
    ModelCls = MFAMaskFiLMGMMVAE if model_family == "main_mfa" else EnergyMaskFiLMGMMVAE
    print(f"[Load] model_family={model_family}")
    celltype_text_embeddings = None
    if model_family == "main_mfa":
        celltype_text_embeddings = _load_celltype_text_embeddings_from_args(saved_args, state)

    common_kwargs = dict(
        num_genes=int(saved_args.get("total_gene", 17999)),
        latent_dim=int(saved_args.get("latent_dim", 128)),
        num_components=int(saved_args.get("num_components", 16)),
        prior_cov_rank=int(saved_args.get("prior_cov_rank", 8)),
        prior_shared_covariance=prior_shared_cov,
        posterior_cov_rank=int(saved_args.get("posterior_cov_rank", 0)),
        expr_hidden_dim=int(saved_args.get("expr_hidden_dim", 1024)),
        mask_hidden_dim=int(saved_args.get("mask_hidden_dim", 512)),
        dec_hidden_dim=int(saved_args.get("dec_hidden_dim", 1024)),
        dropout=float(saved_args.get("dropout", 0.1)),
        prior_type=str(saved_args.get("prior_type", "gmm")),
        num_cell_types=int(saved_args.get("num_cell_types", 0)),
        conditional_prior_on_tissue=bool(saved_args.get("conditional_prior_on_tissue", False)),
        num_tissues=num_tissues,
        num_batches=num_batches,
        batch_emb_dim=batch_emb_dim,
        batch_cond_drop_prob=0.0,
        recon_loss_type=str(saved_args.get("recon_loss", saved_args.get("recon_loss_type", "poisson"))),
    )
    if model_family == "main_mfa":
        model = ModelCls(
            **common_kwargs,
            celltype_text_embeddings=celltype_text_embeddings,
            celltype_text_temperature=float(saved_args.get("celltype_text_temp", 0.1)),
        ).to(device)
    else:
        model = ModelCls(**common_kwargs, tissue_emb_dim=tissue_emb_dim).to(device)

    ret = model.load_state_dict(state, strict=False)
    model.eval()
    return model, ret


def _pca2(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    center = x.mean(axis=0, keepdims=True)
    xc = x - center
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T
    coords = xc @ basis
    return coords, basis, center.reshape(-1)


def _gaussian_pdf_2d(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    cov = cov + np.eye(2, dtype=np.float64) * 1e-6
    inv = np.linalg.inv(cov)
    det = max(float(np.linalg.det(cov)), 1e-12)
    delta = points - mean.reshape(1, 2)
    maha = np.einsum("ni,ij,nj->n", delta, inv, delta)
    return np.exp(-0.5 * maha) / (2.0 * np.pi * np.sqrt(det))


def _ellipse_xy(mean: np.ndarray, cov: np.ndarray, nsig: float = 2.0, n_points: int = 96):
    cov = np.asarray(cov, dtype=np.float64) + np.eye(2, dtype=np.float64) * 1e-8
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points))
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    transform = vecs @ np.diag(nsig * np.sqrt(vals))
    xy = transform @ circle + mean.reshape(2, 1)
    return xy[0], xy[1]


def _metric_mds_from_dist(dmat: np.ndarray, seed: int = 0):
    dmat = np.asarray(dmat, dtype=np.float64)
    try:
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=2,
            metric=True,
            dissimilarity="precomputed",
            random_state=int(seed),
            normalized_stress="auto",
            n_init=4,
            max_iter=1000,
        )
        return mds.fit_transform(dmat), "metric_mds"
    except Exception:
        n = dmat.shape[0]
        h = np.eye(n) - np.ones((n, n), dtype=np.float64) / float(n)
        b = -0.5 * h @ (dmat ** 2) @ h
        vals, vecs = np.linalg.eigh(b)
        order = np.argsort(vals)[::-1][:2]
        vals = np.clip(vals[order], 0.0, None)
        return vecs[:, order] * np.sqrt(vals).reshape(1, -1), "classical_mds"


def _embed_prior_samples(samples: np.ndarray, method: str = "auto", seed: int = 0):
    samples = np.asarray(samples, dtype=np.float32)
    method = str(method).lower()
    if method in ("auto", "umap"):
        try:
            import umap

            emb = umap.UMAP(
                n_components=2,
                n_neighbors=min(30, max(5, samples.shape[0] // 50)),
                min_dist=0.15,
                metric="euclidean",
                random_state=int(seed),
            ).fit_transform(samples)
            return np.asarray(emb, dtype=np.float64), "umap"
        except Exception as exc:
            if method == "umap":
                print(f"[PriorViz][WARN] UMAP failed, falling back to PCA: {exc}")
    if method == "mds":
        try:
            from sklearn.manifold import MDS
            from sklearn.metrics import pairwise_distances

            max_n = min(samples.shape[0], 1500)
            rng = np.random.default_rng(seed)
            idx = np.arange(samples.shape[0])
            if samples.shape[0] > max_n:
                idx = rng.choice(samples.shape[0], size=max_n, replace=False)
                idx.sort()
            dmat = pairwise_distances(samples[idx], metric="euclidean")
            emb_small, _ = _metric_mds_from_dist(dmat, seed=seed)
            if samples.shape[0] == max_n:
                return emb_small, "metric_mds"
            # Use PCA for all points if exact MDS had to be subsampled; this keeps output complete.
            print("[PriorViz][WARN] MDS requested but sample count is large; using PCA for full sample KDE.")
        except Exception as exc:
            print(f"[PriorViz][WARN] MDS failed, falling back to PCA: {exc}")
    coords, _, _ = _pca2(samples)
    return coords, "pca"


def _sample_mfa_prior(mu: np.ndarray, logvar: np.ndarray, factor: np.ndarray, pi: np.ndarray, active_idx: np.ndarray, n_sample: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    active_pi = pi[active_idx]
    active_pi = active_pi / np.clip(active_pi.sum(), 1e-12, None)
    comp = rng.choice(active_idx, size=int(n_sample), replace=True, p=active_pi)
    samples = np.zeros((int(n_sample), mu.shape[1]), dtype=np.float32)
    for idx in active_idx:
        mask = comp == idx
        n_idx = int(mask.sum())
        if n_idx == 0:
            continue
        eps = rng.normal(size=(n_idx, mu.shape[1])).astype(np.float32) * np.exp(0.5 * logvar[idx]).astype(np.float32)
        if factor is not None:
            u = rng.normal(size=(n_idx, factor.shape[2])).astype(np.float32)
            shift = u @ factor[idx].T.astype(np.float32)
        else:
            shift = 0.0
        samples[mask] = mu[idx].astype(np.float32) + shift + eps
    return samples, comp


def visualize_prior(
    model,
    output_dir: Path,
    max_components: int = 32,
    active_min_fold: float = 1.5,
    factor_scale: float = 1.0,
    grid_size: int = 120,
    sample_n: int = 5000,
    sample_embed: str = "auto",
):
    """Save PCA/density/factor visualizations for GMM/MFA prior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    prior = getattr(model, "prior", None)
    if prior is None or not hasattr(prior, "prior_mu"):
        print("[PriorViz] skipped: model has no mixture prior.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        mu = prior.prior_mu.detach().float().cpu().numpy()
        pi = torch.softmax(prior.pi_logits.detach().float(), dim=0).cpu().numpy()
        logvar = prior._expanded_logvar().detach().float().cpu().numpy()
        factor_t = prior._expanded_factor()
        factor = None if factor_t is None else factor_t.detach().float().cpu().numpy()

    if mu.ndim != 2 or mu.shape[0] < 2:
        print("[PriorViz] skipped: not enough prior components.")
        return

    coords, basis, center = _pca2(mu)
    diag_var = np.exp(logvar)
    k, d = mu.shape
    cov2 = np.zeros((k, 2, 2), dtype=np.float64)
    factor2 = None
    for idx in range(k):
        cov2[idx] = basis.T @ (diag_var[idx][:, None] * basis)
        if factor is not None:
            f2 = factor[idx].T @ basis  # (R, 2), one u-std direction in PCA plane.
            cov2[idx] += f2.T @ f2
            if factor2 is None:
                factor2 = np.zeros((k, f2.shape[0], 2), dtype=np.float64)
            factor2[idx] = f2

    pad_x = max(float(np.ptp(coords[:, 0])) * 0.2, 1e-3)
    pad_y = max(float(np.ptp(coords[:, 1])) * 0.2, 1e-3)
    x_grid = np.linspace(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x, int(grid_size))
    y_grid = np.linspace(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y, int(grid_size))
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    density_grid = np.zeros(grid_points.shape[0], dtype=np.float64)
    density_center = np.zeros(k, dtype=np.float64)
    for idx in range(k):
        density_grid += pi[idx] * _gaussian_pdf_2d(grid_points, coords[idx], cov2[idx])
        density_center += pi[idx] * _gaussian_pdf_2d(coords, coords[idx], cov2[idx])
    zz = density_grid.reshape(xx.shape)

    table = pd.DataFrame(
        {
            "component": np.arange(k),
            "pi": pi,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "density_2d": density_center,
            "mu_norm": np.linalg.norm(mu, axis=1),
            "diag_var_mean": diag_var.mean(axis=1),
            "total_var_mean": np.trace(cov2, axis1=1, axis2=2) / 2.0,
        }
    )
    table.to_csv(output_dir / "prior_pca_components.csv", index=False)

    pi_min_all = float(np.min(pi)) if pi.size else 0.0
    active_threshold = pi_min_all * float(active_min_fold)
    active_idx = np.where(pi >= active_threshold)[0]
    if active_idx.size == 0:
        active_idx = np.argsort(-pi)[: max(1, min(int(max_components), k))]
        print(f"[PriorViz] no pi >= min(pi)*{active_min_fold:g}; fallback to top {active_idx.size} components.")
    else:
        if active_idx.size > int(max_components):
            active_idx = active_idx[np.argsort(-pi[active_idx])[: int(max_components)]]
        print(
            f"[PriorViz] drawing {active_idx.size}/{k} active components with "
            f"pi >= min(pi)*{active_min_fold:g} = {active_threshold:.6g}."
        )
    table.loc[active_idx].to_csv(output_dir / "prior_pca_active_components.csv", index=False)
    sizes = 30.0 + 600.0 * pi / max(float(pi.max()), 1e-12)

    mu_active = mu[active_idx]
    pairwise = np.linalg.norm(mu_active[:, None, :] - mu_active[None, :, :], axis=-1)
    pd.DataFrame(pairwise, index=active_idx, columns=active_idx).to_csv(output_dir / "prior_mu_active_pairwise_dist.csv")
    nn_rows = []
    for local_i, comp_i in enumerate(active_idx):
        row = pairwise[local_i].copy()
        row[local_i] = np.inf
        local_j = int(np.argmin(row))
        comp_j = int(active_idx[local_j])
        nn_rows.append(
            {
                "component": int(comp_i),
                "nearest_component": comp_j,
                "nearest_mu_l2": float(pairwise[local_i, local_j]),
                "pi": float(pi[comp_i]),
                "nearest_pi": float(pi[comp_j]),
            }
        )
    pd.DataFrame(nn_rows).to_csv(output_dir / "prior_mu_active_nearest_neighbors.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    im = ax.imshow(pairwise, cmap="mako" if "mako" in plt.colormaps() else "viridis")
    ax.set_xticks(np.arange(active_idx.size))
    ax.set_yticks(np.arange(active_idx.size))
    ax.set_xticklabels(active_idx, rotation=90, fontsize=7)
    ax.set_yticklabels(active_idx, fontsize=7)
    ax.set_xlabel("component")
    ax.set_ylabel("component")
    ax.set_title("Active prior center distances in latent space")
    fig.colorbar(im, ax=ax, label="L2 distance in original latent space")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_mu_active_distance_heatmap.png")
    plt.close(fig)

    mds_coords, mds_name = _metric_mds_from_dist(pairwise, seed=0)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    sca = ax.scatter(
        mds_coords[:, 0],
        mds_coords[:, 1],
        c=pi[active_idx],
        s=sizes[active_idx],
        cmap="magma",
        edgecolor="k",
        linewidth=0.3,
        zorder=3,
    )
    for local_i, comp_i in enumerate(active_idx):
        ax.text(mds_coords[local_i, 0], mds_coords[local_i, 1], str(comp_i), fontsize=7, ha="center", va="center", zorder=4)
        nn_local = int(np.argmin(np.where(np.arange(active_idx.size) == local_i, np.inf, pairwise[local_i])))
        ax.plot(
            [mds_coords[local_i, 0], mds_coords[nn_local, 0]],
            [mds_coords[local_i, 1], mds_coords[nn_local, 1]],
            color="grey",
            alpha=0.35,
            linewidth=0.8,
            zorder=1,
        )
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.set_title(f"Active prior centers preserving latent L2 distances ({mds_name})")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sca, ax=ax, label="mixture weight pi")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_mu_active_mds_distances.png")
    plt.close(fig)

    def _plot_prior_sample_kde(component_idx: np.ndarray, stem: str, title_suffix: str, seed: int):
        prior_samples, prior_sample_comp = _sample_mfa_prior(
            mu=mu,
            logvar=logvar,
            factor=factor,
            pi=pi,
            active_idx=component_idx,
            n_sample=int(sample_n),
            seed=int(seed),
        )
        sample_emb, sample_method = _embed_prior_samples(prior_samples, method=sample_embed, seed=int(seed))
        sample_df = pd.DataFrame(
            {
                "x": sample_emb[:, 0],
                "y": sample_emb[:, 1],
                "component": prior_sample_comp,
            }
        )
        sample_df.to_csv(output_dir / f"{stem}_embedding.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
        sca = ax.scatter(
            sample_emb[:, 0],
            sample_emb[:, 1],
            c=prior_sample_comp,
            cmap="tab20",
            s=2,
            alpha=0.35,
            linewidth=0,
        )
        for idx in component_idx:
            mask = prior_sample_comp == idx
            if not np.any(mask):
                continue
            cx = float(np.median(sample_emb[mask, 0]))
            cy = float(np.median(sample_emb[mask, 1]))
            ax.scatter([cx], [cy], c="white", s=70, edgecolor="black", linewidth=0.6, zorder=4)
            ax.text(cx, cy, str(idx), fontsize=7, ha="center", va="center", zorder=5)
        ax.set_xlabel(f"{sample_method.upper()}1")
        ax.set_ylabel(f"{sample_method.upper()}2")
        ax.set_title(f"Raw high-dimensional prior samples ({title_suffix}, {sample_method}, n={int(sample_n)})")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}_raw_{sample_method}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
        kde_payload = None
        try:
            import scipy.stats as st

            xy = np.vstack([sample_emb[:, 0], sample_emb[:, 1]])
            kde = st.gaussian_kde(xy)
            pad_x_s = max(float(np.ptp(sample_emb[:, 0])) * 0.08, 1e-3)
            pad_y_s = max(float(np.ptp(sample_emb[:, 1])) * 0.08, 1e-3)
            gx = np.linspace(sample_emb[:, 0].min() - pad_x_s, sample_emb[:, 0].max() + pad_x_s, int(grid_size))
            gy = np.linspace(sample_emb[:, 1].min() - pad_y_s, sample_emb[:, 1].max() + pad_y_s, int(grid_size))
            gxx, gyy = np.meshgrid(gx, gy)
            kde_z = kde(np.vstack([gxx.ravel(), gyy.ravel()])).reshape(gxx.shape)
            kde_payload = (gxx, gyy, kde_z)
            ax.contourf(gxx, gyy, kde_z, levels=18, cmap="viridis", alpha=0.65)
            ax.contour(gxx, gyy, kde_z, levels=10, colors="black", alpha=0.18, linewidths=0.5)
        except Exception as exc:
            print(f"[PriorViz][WARN] sample KDE failed, using hexbin only: {exc}")
            ax.hexbin(sample_emb[:, 0], sample_emb[:, 1], gridsize=55, cmap="viridis", mincnt=1, alpha=0.75)
        ax.scatter(
            sample_emb[:, 0],
            sample_emb[:, 1],
            c=prior_sample_comp,
            cmap="tab20",
            s=2,
            alpha=0.22,
            linewidth=0,
        )
        # Place component labels at the median embedded sample position for each sampled component.
        for idx in component_idx:
            mask = prior_sample_comp == idx
            if not np.any(mask):
                continue
            cx = float(np.median(sample_emb[mask, 0]))
            cy = float(np.median(sample_emb[mask, 1]))
            ax.scatter([cx], [cy], c="white", s=70, edgecolor="black", linewidth=0.6, zorder=4)
            ax.text(cx, cy, str(idx), fontsize=7, ha="center", va="center", zorder=5)
        ax.set_xlabel(f"{sample_method.upper()}1")
        ax.set_ylabel(f"{sample_method.upper()}2")
        ax.set_title(f"True high-dimensional MFA prior samples with KDE ({title_suffix}, {sample_method})")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}_kde.png")
        plt.close(fig)

        if kde_payload is not None:
            gxx, gyy, kde_z = kde_payload
            finite_z = kde_z[np.isfinite(kde_z)]
            if finite_z.size > 0:
                top_threshold = float(np.quantile(finite_z, 0.90))
                kde_top = np.where(kde_z >= top_threshold, kde_z, np.nan)
                fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
                ax.contourf(gxx, gyy, kde_top, levels=12, cmap="inferno", alpha=0.8)
                ax.contour(gxx, gyy, kde_z, levels=[top_threshold], colors="black", linewidths=1.0)
                high_sample = kde(sample_emb.T) >= top_threshold
                ax.scatter(
                    sample_emb[~high_sample, 0],
                    sample_emb[~high_sample, 1],
                    c="lightgrey",
                    s=1,
                    alpha=0.08,
                    linewidth=0,
                )
                ax.scatter(
                    sample_emb[high_sample, 0],
                    sample_emb[high_sample, 1],
                    c=prior_sample_comp[high_sample],
                    cmap="tab20",
                    s=3,
                    alpha=0.45,
                    linewidth=0,
                )
                top_rows = []
                for idx in component_idx:
                    mask = prior_sample_comp == idx
                    if not np.any(mask):
                        continue
                    frac_top = float(np.mean(high_sample[mask]))
                    top_rows.append({"component": int(idx), "frac_samples_in_top10_kde": frac_top})
                    if frac_top <= 0:
                        continue
                    cx = float(np.median(sample_emb[mask & high_sample, 0]))
                    cy = float(np.median(sample_emb[mask & high_sample, 1]))
                    ax.scatter([cx], [cy], c="white", s=70, edgecolor="black", linewidth=0.6, zorder=4)
                    ax.text(cx, cy, str(idx), fontsize=7, ha="center", va="center", zorder=5)
                pd.DataFrame(top_rows).to_csv(output_dir / f"{stem}_kde_top10_component_fraction.csv", index=False)
                ax.set_xlabel(f"{sample_method.upper()}1")
                ax.set_ylabel(f"{sample_method.upper()}2")
                ax.set_title(f"Top 10% KDE density region ({title_suffix}, {sample_method})")
                ax.set_aspect("equal", adjustable="datalim")
                fig.tight_layout()
                fig.savefig(output_dir / f"{stem}_kde_top10.png")
                plt.close(fig)

    if int(sample_n) > 0:
        _plot_prior_sample_kde(active_idx, "prior_sample_active", "active components", seed=0)
        _plot_prior_sample_kde(np.arange(k, dtype=int), "prior_sample_all", "all components", seed=1)
        # Backward-compatible copies for scripts expecting the old filenames.
        shutil.copyfile(output_dir / "prior_sample_active_embedding.csv", output_dir / "prior_sample_embedding.csv")
        shutil.copyfile(output_dir / "prior_sample_active_kde.png", output_dir / "prior_sample_umap_kde.png")

    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.35, linewidth=0, antialiased=True)
    sca = ax.scatter(
        coords[active_idx, 0],
        coords[active_idx, 1],
        density_center[active_idx],
        c=pi[active_idx],
        s=sizes[active_idx],
        cmap="magma",
        edgecolor="k",
        linewidth=0.3,
    )
    ax.set_xlabel("prior mu PC1")
    ax.set_ylabel("prior mu PC2")
    ax.set_zlabel("projected mixture density")
    ax.set_title("MFA prior centers on PCA density surface")
    fig.colorbar(sca, ax=ax, shrink=0.65, pad=0.1, label="mixture weight pi")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_mu_pca_density_3d.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    sca = ax.scatter(
        coords[active_idx, 0],
        coords[active_idx, 1],
        c=pi[active_idx],
        s=sizes[active_idx],
        cmap="magma",
        edgecolor="k",
        linewidth=0.3,
        zorder=3,
    )
    ax.contour(xx, yy, zz, levels=12, cmap="viridis", alpha=0.55, linewidths=0.8)
    if factor2 is not None:
        colors = plt.get_cmap("tab10")
        for idx in active_idx:
            for r in range(factor2.shape[1]):
                dx, dy = factor2[idx, r] * float(factor_scale)
                ax.arrow(
                    coords[idx, 0],
                    coords[idx, 1],
                    dx,
                    dy,
                    color=colors(r % 10),
                    alpha=0.28,
                    width=0.0,
                    head_width=0.025 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1.0),
                    length_includes_head=True,
                    zorder=2,
                )
    for idx in active_idx:
        ax.text(coords[idx, 0], coords[idx, 1], str(idx), fontsize=6, ha="center", va="center", zorder=4)
    ax.set_xlabel("prior mu PC1")
    ax.set_ylabel("prior mu PC2")
    ax.set_title("MFA prior centers and projected factor directions")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sca, ax=ax, label="mixture weight pi")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_mu_pca_factor_arrows.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    ax.contour(xx, yy, zz, levels=12, cmap="Greys", alpha=0.45, linewidths=0.8)
    order = active_idx[np.argsort(pi[active_idx])]
    cmap = plt.get_cmap("magma")
    pi_min = float(pi[active_idx].min())
    pi_ptp = max(float(np.ptp(pi[active_idx])), 1e-12)
    for idx in order:
        color = cmap((float(pi[idx]) - pi_min) / pi_ptp)
        ex, ey = _ellipse_xy(coords[idx], cov2[idx], nsig=2.0)
        ax.plot(ex, ey, color=color, alpha=0.55, linewidth=1.0)
    sca = ax.scatter(
        coords[active_idx, 0],
        coords[active_idx, 1],
        c=pi[active_idx],
        s=sizes[active_idx],
        cmap="magma",
        edgecolor="k",
        linewidth=0.3,
        zorder=3,
    )
    for idx in active_idx:
        ax.text(coords[idx, 0], coords[idx, 1], str(idx), fontsize=6, ha="center", va="center", zorder=4)
    ax.set_xlabel("prior mu PC1")
    ax.set_ylabel("prior mu PC2")
    ax.set_title("Projected MFA Gaussian ellipses (2 sigma)")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sca, ax=ax, label="mixture weight pi")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_mu_pca_gaussian_ellipses.png")
    plt.close(fig)

    rng = np.random.default_rng(0)
    n_sample = 8000
    active_pi = pi[active_idx]
    comp = rng.choice(active_idx, size=n_sample, replace=True, p=active_pi / np.clip(active_pi.sum(), 1e-12, None))
    sample_pc = np.zeros((n_sample, 2), dtype=np.float64)
    for idx in active_idx:
        mask = comp == idx
        n_idx = int(mask.sum())
        if n_idx == 0:
            continue
        sample_pc[mask] = rng.multivariate_normal(mean=coords[idx], cov=cov2[idx] + np.eye(2) * 1e-6, size=n_idx)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    ax.scatter(sample_pc[:, 0], sample_pc[:, 1], c=comp, cmap="tab20", s=2, alpha=0.35, linewidth=0)
    ax.scatter(coords[active_idx, 0], coords[active_idx, 1], c="black", s=18, marker="x", linewidth=0.8)
    ax.set_xlabel("prior mu PC1")
    ax.set_ylabel("prior mu PC2")
    ax.set_title("Samples from projected MFA prior")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_sample_pca.png")
    plt.close(fig)

    print(f"[PriorViz] saved to {output_dir}")


def _extract_model_embedding(model, x_count, x_mask, embedding_mode: str, tissue_id=None):
    mode = str(embedding_mode).lower()
    if mode == "encoder_hidden":
        x_expr = torch.log1p(x_count.float())
        _, _, h = model.encoder(x_expr=x_expr, x_mask=x_mask.float(), return_hidden=True)
        return h

    forward_kwargs = dict(
        x_count=x_count,
        x_mask=x_mask,
        tissue_id=tissue_id,
        sample_id=None,
        use_batch_condition=False,
    )
    if model.__class__.__module__.startswith("main_energy"):
        forward_kwargs["use_tissue_condition"] = True
    out = model(**forward_kwargs)
    if mode == "mixmu":
        if ("q_c" in out) and ("mu_comp" in out):
            return torch.sum(out["q_c"].unsqueeze(-1) * out["mu_comp"], dim=1)
        return out["mu"]
    if mode == "mu_base":
        return out.get("mu_base", out.get("mu", out["z"]))
    if mode == "z":
        return out["z"]
    raise ValueError(f"Unsupported embedding_mode: {embedding_mode}")


def _require_finite_embedding(emb: np.ndarray, context: str) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    finite = np.isfinite(emb)
    if finite.all():
        return emb
    bad = ~finite
    bad_rows = np.where(bad.any(axis=1))[0]
    bad_cols = np.where(bad.any(axis=0))[0]
    first = np.argwhere(bad)[0]
    raise ValueError(
        f"Non-finite embedding in {context}: shape={emb.shape}, "
        f"bad_values={int(bad.sum())}, bad_rows={len(bad_rows)}, bad_cols={len(bad_cols)}, "
        f"first_bad=(row={int(first[0])}, col={int(first[1])}, value={emb[tuple(first)]})"
    )


def extract_embedding_for_file(
    model,
    h5ad_path: Path,
    tissue_name: str,
    gene_ids,
    tissue_map,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    embedding_mode: str,
):
    dataset = XVerseFineTuneDataset(
        {str(h5ad_path): None},
        gene_ids,
        tissue_map,
        use_qc=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    z_list = []
    with torch.no_grad():
        for _, values, tissue_ids in tqdm(loader, desc=f"extract {h5ad_path.name}", leave=False):
            values = values.to(device, non_blocking=True)
            tissue_ids = tissue_ids.to(device, non_blocking=True)
            x_mask = (values != -1).float()
            x_count = torch.where(x_mask > 0, values, torch.zeros_like(values))
            x_count = torch.clamp(x_count, min=0.0)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                emb = _extract_model_embedding(
                    model,
                    x_count=x_count,
                    x_mask=x_mask,
                    embedding_mode=embedding_mode,
                    tissue_id=tissue_ids,
                )
            z_list.append(_require_finite_embedding(emb.detach().cpu().numpy(), f"{h5ad_path.name} batch"))

    return _require_finite_embedding(np.concatenate(z_list, axis=0), str(h5ad_path))


def _gene_ids_from_adata(adata, gene_id_col: str):
    if gene_id_col and gene_id_col in adata.var.columns:
        return adata.var[gene_id_col].astype(str).values
    for col in ["gene_ids", "gene_id", "ensembl_id", "ensembl_ids", "feature_id"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).values
    return adata.var_names.astype(str).values


def extract_embedding_from_adata(
    model,
    adata,
    gene_ids,
    gene_id_col: str,
    count_layer: str,
    tissue_id: int,
    device: torch.device,
    batch_size: int,
    embedding_mode: str,
):
    source_gene_ids = _gene_ids_from_adata(adata, gene_id_col)
    gene_to_idx = {g: i for i, g in enumerate(source_gene_ids)}
    valid_pos = []
    valid_gene_idx = []
    for pos, g in enumerate(gene_ids):
        idx = gene_to_idx.get(str(g))
        if idx is not None:
            valid_pos.append(pos)
            valid_gene_idx.append(idx)
    if not valid_pos:
        raise ValueError("No overlap between model gene ids and adata genes.")

    if count_layer and count_layer != "nan" and count_layer in adata.layers:
        raw = adata.layers[count_layer]
    elif count_layer == "raw" and adata.raw is not None:
        raw = adata.raw.X
    else:
        raw = adata.raw.X if adata.raw is not None else adata.X
    n = adata.n_obs
    z_list = []
    with torch.no_grad():
        for st in tqdm(range(0, n, batch_size), desc="extract manifest h5ad", leave=False):
            ed = min(st + batch_size, n)
            block = raw[st:ed, :]
            block = block[:, valid_gene_idx]
            arr = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            x_np = np.full((ed - st, len(gene_ids)), -1.0, dtype=np.float32)
            x_np[:, valid_pos] = np.asarray(arr, dtype=np.float32)
            values = torch.tensor(x_np, dtype=torch.float32, device=device)
            x_mask = (values != -1).float()
            x_count = torch.where(x_mask > 0, values, torch.zeros_like(values))
            x_count = torch.clamp(x_count, min=0.0)
            tissue_ids = torch.full((ed - st,), int(tissue_id), dtype=torch.long, device=device)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                emb = _extract_model_embedding(
                    model,
                    x_count=x_count,
                    x_mask=x_mask,
                    embedding_mode=embedding_mode,
                    tissue_id=tissue_ids,
                )
            z_list.append(_require_finite_embedding(emb.detach().cpu().numpy(), f"manifest batch {st}:{ed}"))
    return _require_finite_embedding(np.concatenate(z_list, axis=0), "manifest h5ad")


def _resolve_manifest_tissue_id(row, tissue_map):
    if "tissue_id" in row and not pd.isna(row["tissue_id"]):
        return int(row["tissue_id"])
    tissue = str(row.get("tissue", row.get("dataset", ""))).strip().lower()
    if tissue in tissue_map:
        return int(tissue_map[tissue])
    raise KeyError(
        f"Cannot resolve tissue id for manifest row. Provide tissue_id or tissue in {sorted(tissue_map)}; got tissue={tissue!r}."
    )


def process_dataset_manifest(args, model, gene_ids, device: torch.device, timing_records):
    df = pd.read_csv(args.dataset_manifest)
    required = {"dataset", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset manifest missing columns: {sorted(missing)}")
    for _, row in df.iterrows():
        fp = Path(str(row["path"]))
        if not fp.exists():
            print(f"[WARN] h5ad path not found: {fp}")
            continue
        dataset = str(row["dataset"])
        tissue = str(row.get("tissue", dataset))
        tissue_id = _resolve_manifest_tissue_id(row, {"liver": 31, "brain": 7})
        gene_id_col = str(row.get("gene_id_col", args.default_gene_id_col))
        count_layer = str(row.get("count_layer", ""))
        print(f"[INFO] manifest dataset={dataset} file={fp}")
        adata = sc.read_h5ad(fp)
        start = time.time()
        emb = extract_embedding_from_adata(
            model=model,
            adata=adata,
            gene_ids=gene_ids,
            gene_id_col=gene_id_col,
            count_layer=count_layer,
            tissue_id=tissue_id,
            device=device,
            batch_size=args.batch_size,
            embedding_mode=args.embedding_mode,
        )
        cost = time.time() - start
        emb = _require_finite_embedding(emb, f"{fp} before write")
        adata.obsm[args.embedding_key] = emb
        adata.write(fp)
        print(f"[OK] wrote {args.embedding_key} to {fp} shape={emb.shape} time={cost:.2f}s")
        timing_records.append(
            {
                "model": "xVERSE",
                "tissue": tissue,
                "gene_set": "all",
                "file": fp.name,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "time_seconds": cost,
            }
        )


def process_tissue_dir(args, model, gene_ids, tissue_name: str, tissue_dir: Path, tissue_map, timing_records):
    if not tissue_dir.exists():
        print(f"[WARN] tissue dir not found: {tissue_dir}")
        return
    gene_set = "all"
    files = sorted(tissue_dir.glob(f"{tissue_name}_*_{gene_set}.h5ad"))
    if args.max_files_per_set > 0:
        files = files[: args.max_files_per_set]
    if not files:
        print(f"[INFO] no files for {tissue_name} / {gene_set} in {tissue_dir}")
        return
    print(f"[INFO] {tissue_name}/{gene_set}: {len(files)} files")
    for fp in files:
        adata = sc.read_h5ad(fp)
        start = time.time()
        emb = extract_embedding_for_file(
            model=model,
            h5ad_path=fp,
            tissue_name=tissue_name,
            gene_ids=gene_ids,
            tissue_map=tissue_map,
            device=choose_device(args.device),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            embedding_mode=args.embedding_mode,
        )
        cost = time.time() - start
        emb = _require_finite_embedding(emb, f"{fp.name} before write")
        adata.obsm[args.embedding_key] = emb
        adata.write(fp)
        print(f"[OK] wrote {args.embedding_key} to {fp.name} shape={emb.shape} time={cost:.2f}s")
        timing_records.append(
            {
                "model": "xVERSE",
                "tissue": tissue_name,
                "gene_set": gene_set,
                "file": fp.name,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "time_seconds": cost,
            }
        )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = choose_device(args.device)
    print(f"[Device] {device}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    print("[Load] gene ids ...")
    gene_ids = load_gene_ids(args.gene_ids_path)

    print("[Load] model ...")
    build_model_from_ckpt._requested_family = args.model_family
    model, load_ret = build_model_from_ckpt(args.ckpt, device)
    print(f"[Load] missing={len(load_ret.missing_keys)} unexpected={len(load_ret.unexpected_keys)}")
    if load_ret.missing_keys:
        print(f"[Load] missing_keys={load_ret.missing_keys[:20]}")
    if load_ret.unexpected_keys:
        print(f"[Load] unexpected_keys={load_ret.unexpected_keys[:20]}")

    if not args.no_prior_viz:
        prior_viz_dir = Path(args.prior_viz_dir) if args.prior_viz_dir else Path(args.output_dir) / "prior_viz"
        try:
            visualize_prior(
                model=model,
                output_dir=prior_viz_dir,
                max_components=args.prior_viz_max_components,
                active_min_fold=args.prior_viz_active_min_fold,
                factor_scale=args.prior_viz_factor_scale,
                grid_size=args.prior_viz_grid,
                sample_n=args.prior_viz_sample_n,
                sample_embed=args.prior_viz_sample_embed,
            )
        except Exception as exc:
            print(f"[PriorViz][WARN] failed: {exc}")

    tissue_map = {"liver": 31, "brain": 7}
    timing_records = []

    if args.dataset_manifest:
        process_dataset_manifest(args, model, gene_ids, device, timing_records)
    else:
        process_tissue_dir(
            args=args,
            model=model,
            gene_ids=gene_ids,
            tissue_name="liver",
            tissue_dir=Path(args.liver_dir),
            tissue_map=tissue_map,
            timing_records=timing_records,
        )
        process_tissue_dir(
            args=args,
            model=model,
            gene_ids=gene_ids,
            tissue_name="brain",
            tissue_dir=Path(args.brain_dir),
            tissue_map=tissue_map,
            timing_records=timing_records,
        )

    if timing_records:
        out_csv = os.path.join(args.output_dir, "xverse_current_inference_timing.csv")
        pd.DataFrame(timing_records).to_csv(out_csv, index=False)
        print(f"[Done] timing saved to {out_csv}")
    else:
        print("[Done] no files processed.")


if __name__ == "__main__":
    main()
