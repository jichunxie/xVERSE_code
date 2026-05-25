#!/usr/bin/env python3
"""
Overlay real cells and one-shot model-generated NB cells in expression UMAP space.

Workflow:
1. Load a trained xVERSE GM/MFA checkpoint.
2. Load real fig2 h5ad files and align counts to the model gene order.
3. Sample z from the learned prior once, decode NB rate/theta, and sample generated counts once.
4. Concatenate real + generated counts and run the standard Scanpy expression pipeline:
   normalize_total -> log1p -> HVG -> PCA -> neighbors -> UMAP.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.utils_model import load_gene_ids

_EXTRACT_PATH = Path(__file__).resolve().with_name("02_extract_gmmvae_embedding.py")
_spec = importlib.util.spec_from_file_location("fig2_extract_gmmvae_embedding", _EXTRACT_PATH)
_extract_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extract_mod)
build_model_from_ckpt = _extract_mod.build_model_from_ckpt
choose_device = _extract_mod.choose_device


def parse_args():
    ap = argparse.ArgumentParser(description="Plot real + generated NB expression UMAP for current xVERSE model.")
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint.")
    ap.add_argument("--model-family", default="auto", choices=["auto", "main_energy", "main_mfa"])
    ap.add_argument(
        "--gene-ids-path",
        default="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt",
    )
    ap.add_argument("--liver-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/liver")
    ap.add_argument("--brain-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/brain")
    ap.add_argument("--tissues", default="liver,brain", help="Comma-separated subset from: liver,brain.")
    ap.add_argument("--gene-set", default="all", choices=["all", "5k", "xenium"])
    ap.add_argument("--output-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2_gmmvae_current/generated_nb_umap")
    ap.add_argument("--max-real-cells", type=int, default=20000, help="Max real cells per tissue. 0 means all.")
    ap.add_argument("--generated-cells", type=int, default=0, help="Generated cells per tissue. 0 means match real n.")
    ap.add_argument("--batch-size", type=int, default=256, help="Generation batch size.")
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument("--n-top-genes", type=int, default=3000)
    ap.add_argument("--neighbors-k", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--save-h5ad", action="store_true", help="Save combined real/generated AnnData used for UMAP.")
    return ap.parse_args()


def _gene_ids_from_adata(adata):
    for col in ["gene_ids", "gene_id", "ensembl_id", "ensembl_ids", "feature_id"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).values
    return adata.var_names.astype(str).values


def _raw_matrix(adata):
    if adata.raw is not None:
        return adata.raw.X, _gene_ids_from_adata(adata.raw.to_adata())
    return adata.X, _gene_ids_from_adata(adata)


def _pick_celltype_col(obs):
    for col in ["cell_type", "celltype", "cell_type_ontology_term_id", "celltype.l2"]:
        if col in obs.columns:
            return col
    return None


def _load_real_counts(tissue_name: str, tissue_dir: Path, gene_set: str, gene_ids, max_cells: int, seed: int):
    files = sorted(tissue_dir.glob(f"{tissue_name}_*_{gene_set}.h5ad"))
    if not files:
        raise FileNotFoundError(f"No h5ad files found for {tissue_name}/{gene_set} in {tissue_dir}")

    rng = np.random.default_rng(seed)
    rows = []
    obs_rows = []
    n_total = 0
    for fp in files:
        adata = sc.read_h5ad(fp)
        n_total += int(adata.n_obs)
    if max_cells > 0 and n_total > max_cells:
        keep_global = set(rng.choice(n_total, size=max_cells, replace=False).tolist())
    else:
        keep_global = None

    model_gene_to_pos = {str(g): i for i, g in enumerate(gene_ids)}
    offset = 0
    for fp in tqdm(files, desc=f"load real {tissue_name}/{gene_set}"):
        adata = sc.read_h5ad(fp)
        local_idx = np.arange(adata.n_obs)
        if keep_global is not None:
            mask = np.array([(offset + int(i)) in keep_global for i in local_idx], dtype=bool)
            local_idx = local_idx[mask]
        offset += int(adata.n_obs)
        if local_idx.size == 0:
            continue

        raw, source_gene_ids = _raw_matrix(adata)
        src_gene_to_idx = {str(g): i for i, g in enumerate(source_gene_ids)}
        src_idx = []
        dst_idx = []
        for g, dst in model_gene_to_pos.items():
            src = src_gene_to_idx.get(g)
            if src is not None:
                src_idx.append(src)
                dst_idx.append(dst)
        if not src_idx:
            raise ValueError(f"No gene overlap between model genes and {fp}")

        block = raw[local_idx, :]
        block = block[:, src_idx]
        block = sparse.csr_matrix(block) if not sparse.issparse(block) else block.tocsr()
        aligned = sparse.csr_matrix((local_idx.size, len(gene_ids)), dtype=np.float32)
        aligned[:, np.asarray(dst_idx, dtype=np.int64)] = block.astype(np.float32)
        rows.append(aligned)

        donor_id = fp.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
        obs = pd.DataFrame(index=[f"real_{tissue_name}_{donor_id}_{i}" for i in local_idx])
        obs["source"] = "real"
        obs["tissue"] = tissue_name
        obs["donor_id"] = donor_id
        ct_col = _pick_celltype_col(adata.obs)
        if ct_col is not None:
            obs["cell_type"] = adata.obs.iloc[local_idx][ct_col].astype(str).values
        obs_rows.append(obs)

    if not rows:
        raise ValueError(f"No real cells loaded for {tissue_name}/{gene_set}")
    x = sparse.vstack(rows, format="csr")
    obs = pd.concat(obs_rows, axis=0)
    print(f"[Real] {tissue_name}/{gene_set}: n={x.shape[0]}, genes={x.shape[1]}, nnz={x.nnz}")
    return x, obs


def _sample_prior_z(model, n: int, device: torch.device, seed: int):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    prior = getattr(model, "prior", None)
    latent_dim = int(getattr(model, "latent_dim"))
    if prior is None or getattr(model, "prior_type", "gaussian") != "gmm":
        z = torch.randn((n, latent_dim), device=device, generator=gen)
        comp = torch.full((n,), -1, dtype=torch.long, device=device)
        return z, comp

    with torch.no_grad():
        pi = torch.softmax(prior.pi_logits.detach().float().to(device), dim=0)
        comp = torch.multinomial(pi, num_samples=n, replacement=True, generator=gen)
        mu = prior.prior_mu.detach().float().to(device)[comp]
        logvar = prior._expanded_logvar().detach().float().to(device)[comp]
        eps = torch.randn(mu.shape, device=device, generator=gen) * torch.exp(0.5 * logvar)
        factor = prior._expanded_factor()
        if factor is not None:
            factor = factor.detach().float().to(device)[comp]
            rank = int(factor.shape[-1])
            u = torch.randn((n, rank), device=device, generator=gen)
            shift = torch.einsum("bdr,br->bd", factor, u)
        else:
            shift = torch.zeros_like(mu)
        z = mu + shift + eps
        return z, comp


def _decode_nb_params(model, z: torch.Tensor):
    cond = None
    gene_logits = model.decoder(z, cond=cond)
    gene_logits = torch.nan_to_num(gene_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    library_size = torch.nn.functional.softplus(model.library_head(z).float()) + 1e-8
    library_size = torch.nan_to_num(library_size, nan=1.0, posinf=1e5, neginf=1.0).clamp(1e-8, 1e5)
    rate = torch.softmax(gene_logits, dim=-1) * library_size
    rate = torch.nan_to_num(rate, nan=1e-8, posinf=1e5, neginf=1e-8).clamp(1e-8, 1e5)
    if getattr(model, "nb_theta_mode", "gene") == "cell_gene":
        theta_logits = model.nb_theta_decoder(z, cond=cond)
        theta = torch.nn.functional.softplus(theta_logits.float()) + 1e-8
    else:
        theta = torch.nn.functional.softplus(model.nb_log_theta.float()).view(1, -1) + 1e-8
        theta = theta.expand_as(rate)
    theta = torch.nan_to_num(theta, nan=1.0, posinf=1e4, neginf=1e-8).clamp(1e-8, 1e4)
    return rate, theta


def _generate_nb_counts(model, n: int, gene_ids, device: torch.device, batch_size: int, seed: int):
    rows = []
    comp_rows = []
    model.eval()
    with torch.no_grad():
        for st in tqdm(range(0, n, batch_size), desc="generate NB cells"):
            ed = min(st + batch_size, n)
            z, comp = _sample_prior_z(model, ed - st, device=device, seed=seed + st)
            rate, theta = _decode_nb_params(model, z)
            logits = torch.log(rate) - torch.log(theta)
            dist = torch.distributions.NegativeBinomial(total_count=theta, logits=logits)
            counts = dist.sample().detach().cpu().numpy().astype(np.float32)
            counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(sparse.csr_matrix(counts))
            comp_rows.append(comp.detach().cpu().numpy())
    x = sparse.vstack(rows, format="csr")
    comp = np.concatenate(comp_rows, axis=0)
    obs = pd.DataFrame(index=[f"generated_{i}" for i in range(n)])
    obs["source"] = "generated"
    obs["tissue"] = "generated_prior"
    obs["donor_id"] = "generated"
    obs["component"] = comp.astype(int)
    print(f"[Generated] n={x.shape[0]}, genes={x.shape[1]}, nnz={x.nnz}")
    return x, obs


def _standard_scanpy_umap(adata, args):
    sc.pp.normalize_total(adata, target_sum=float(args.target_sum))
    sc.pp.log1p(adata)
    n_top = min(int(args.n_top_genes), adata.n_vars)
    if n_top > 0 and n_top < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat", subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", random_state=int(args.seed))
    sc.pp.neighbors(adata, n_neighbors=int(args.neighbors_k), random_state=int(args.seed))
    sc.tl.umap(adata, min_dist=float(args.min_dist), random_state=int(args.seed))
    return adata


def _plot_overlay(adata, tissue_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=180)
    sc.pl.umap(
        adata,
        color="source",
        ax=axes[0],
        show=False,
        frameon=False,
        title=f"{tissue_name}: real vs generated NB",
        palette={"real": "#2b6cb0", "generated": "#e53e3e"},
    )
    if "component" in adata.obs.columns:
        sc.pl.umap(
            adata[adata.obs["source"] == "generated"].copy(),
            color="component",
            ax=axes[1],
            show=False,
            frameon=False,
            title="generated cells by prior component",
            legend_loc="right margin",
        )
    else:
        axes[1].axis("off")
    fig.tight_layout()
    out_png = out_dir / f"{tissue_name}_real_generated_nb_umap.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_png}")


def main():
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    build_model_from_ckpt._requested_family = args.model_family
    print(f"[Load] model {args.ckpt} on {device}")
    model, load_ret = build_model_from_ckpt(args.ckpt, device)
    print(f"[Load] missing={len(load_ret.missing_keys)} unexpected={len(load_ret.unexpected_keys)}")

    gene_ids = load_gene_ids(args.gene_ids_path)
    var = pd.DataFrame(index=np.asarray(gene_ids, dtype=str))
    var["gene_id"] = np.asarray(gene_ids, dtype=str)

    tissue_dirs = {"liver": Path(args.liver_dir), "brain": Path(args.brain_dir)}
    for tissue_name in [x.strip() for x in str(args.tissues).split(",") if x.strip()]:
        if tissue_name not in tissue_dirs:
            raise ValueError(f"Unsupported tissue {tissue_name!r}; choose from {sorted(tissue_dirs)}")
        real_x, real_obs = _load_real_counts(
            tissue_name=tissue_name,
            tissue_dir=tissue_dirs[tissue_name],
            gene_set=args.gene_set,
            gene_ids=gene_ids,
            max_cells=int(args.max_real_cells),
            seed=int(args.seed),
        )
        n_gen = int(args.generated_cells) if int(args.generated_cells) > 0 else int(real_x.shape[0])
        gen_x, gen_obs = _generate_nb_counts(
            model=model,
            n=n_gen,
            gene_ids=gene_ids,
            device=device,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + 100000,
        )
        gen_obs["tissue"] = tissue_name
        x = sparse.vstack([real_x, gen_x], format="csr")
        obs = pd.concat([real_obs, gen_obs], axis=0)
        combined = ad.AnnData(X=x, obs=obs, var=var.copy())
        combined.obs_names_make_unique()
        print(f"[UMAP] {tissue_name}: combined n={combined.n_obs}, genes={combined.n_vars}")
        combined = _standard_scanpy_umap(combined, args)
        if args.save_h5ad:
            out_h5ad = out_dir / f"{tissue_name}_real_generated_nb_umap.h5ad"
            combined.write(out_h5ad)
            print(f"[OK] saved {out_h5ad}")
        _plot_overlay(combined, tissue_name=tissue_name, out_dir=out_dir)


if __name__ == "__main__":
    main()
