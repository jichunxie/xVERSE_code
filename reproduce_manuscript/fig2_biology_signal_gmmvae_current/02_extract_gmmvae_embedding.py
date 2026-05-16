#!/usr/bin/env python3
"""
Extract GMVAE/MFA embeddings for fig2 biology signal datasets.

This mirrors the original fig2 xVerse embedding extraction flow but uses:
- main_energy or main_mfa MaskFiLMGMMVAE checkpoint
- same donor h5ad files and gene-set splits
- output embedding key: obsm["xVerse_gmmvae_mixmu"] by default
"""

import argparse
import os
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
    ap = argparse.ArgumentParser(description="Extract GMVAE embeddings for fig2 donor datasets.")
    ap.add_argument(
        "--gene-ids-path",
        default="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt",
    )
    ap.add_argument(
        "--ckpt",
        default="/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue3/last_model.pth",
        help="Path to trained GMVAE checkpoint.",
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
    ap.add_argument("--embedding-key", default="xVerse_gmmvae_mixmu")
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
        model = ModelCls(**common_kwargs).to(device)
    else:
        model = ModelCls(**common_kwargs, tissue_emb_dim=tissue_emb_dim).to(device)

    ret = model.load_state_dict(state, strict=False)
    model.eval()
    return model, ret


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
            z_list.append(emb.detach().cpu().numpy())

    return np.concatenate(z_list, axis=0)


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
            z_list.append(emb.detach().cpu().numpy())
    return np.concatenate(z_list, axis=0)


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
        adata.obsm[args.embedding_key] = emb
        adata.write(fp)
        print(f"[OK] wrote {args.embedding_key} to {fp} shape={emb.shape} time={cost:.2f}s")
        timing_records.append(
            {
                "model": "gmmvae_current",
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
        adata.obsm[args.embedding_key] = emb
        adata.write(fp)
        print(f"[OK] wrote {args.embedding_key} to {fp.name} shape={emb.shape} time={cost:.2f}s")
        timing_records.append(
            {
                "model": "gmmvae_current",
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
        out_csv = os.path.join(args.output_dir, "gmmvae_current_inference_timing.csv")
        pd.DataFrame(timing_records).to_csv(out_csv, index=False)
        print(f"[Done] timing saved to {out_csv}")
    else:
        print("[Done] no files processed.")


if __name__ == "__main__":
    main()
