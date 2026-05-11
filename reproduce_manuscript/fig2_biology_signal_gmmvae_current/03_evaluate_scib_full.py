#!/usr/bin/env python3
"""
Evaluate GMVAE and existing foundation model embeddings with a broader scIB metric set.

Design goals:
1) Reuse already computed embeddings in donor h5ad files (no re-extraction for other FMs).
2) Add GMVAE embedding key into comparison.
3) Compute a broader set of scIB metrics with robust try/except fallback.
4) Save per-tissue/per-gene-set csv files and one global summary csv.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scib


def parse_args():
    ap = argparse.ArgumentParser(description="Comprehensive scIB evaluation for fig2 embeddings.")
    ap.add_argument("--liver-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/liver")
    ap.add_argument("--brain-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/brain")
    ap.add_argument("--output-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2_gmmvae_current/evaluation_scib_full")
    ap.add_argument("--old-eval-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/evaluation")
    ap.add_argument("--gmm-key", default="xVerse_gmmvae")
    ap.add_argument("--neighbors-k", type=int, default=15)
    ap.add_argument("--max-cells", type=int, default=0, help="0 means all cells; otherwise random subsample for speed.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


MODELS = [
    ("xVerse", "xVerse"),
    ("GMVAE", "xVerse_gmmvae"),
    ("Harmony", "harmony"),
    ("scGPT", "scgpt"),
    ("Nicheformer", "nicheformer"),
    ("Geneformer", "geneformer"),
]


def pick_celltype_col(obs: pd.DataFrame):
    for col in ["cell_type", "celltype", "cell_type_ontology_term_id", "celltype.l2"]:
        if col in obs.columns:
            return col
    return None


def load_merged_tissue(tissue_dir: str, tissue_name: str, gene_set: str):
    files = sorted(Path(tissue_dir).glob(f"{tissue_name}_*_{gene_set}.h5ad"))
    if not files:
        raise FileNotFoundError(f"No files found for {tissue_name}/{gene_set} in {tissue_dir}")

    adata_list = []
    for fp in files:
        ad = sc.read_h5ad(fp)
        donor_id = fp.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
        if "donor_id" not in ad.obs.columns:
            ad.obs["donor_id"] = donor_id
        adata_list.append(ad)

    adata = sc.concat(adata_list, join="outer", index_unique=None)
    return adata


def maybe_subsample(adata, max_cells: int, seed: int):
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
    idx.sort()
    return adata[idx].copy()


def run_one_metric(name, fn):
    try:
        return float(fn()), ""
    except Exception as e:
        return np.nan, str(e)


def eval_one_embedding(adata, embed_key: str, celltype_col: str, batch_col: str, neighbors_k: int):
    """
    Broader scIB-like set:
    - Bio conservation: ASW_label, isolated label ASW, cLISI
    - Batch mixing: iLISI, graph connectivity, PCR batch
    """
    if embed_key not in adata.obsm:
        return None

    work = sc.AnnData(obs=adata.obs.copy())
    work.obsm[embed_key] = adata.obsm[embed_key]
    sc.pp.neighbors(work, use_rep=embed_key, n_neighbors=neighbors_k)

    metrics = {}
    errors = {}

    metrics["ASW_label"], errors["ASW_label"] = run_one_metric(
        "ASW_label",
        lambda: scib.metrics.silhouette(work, label_key=celltype_col, embed=embed_key),
    )

    metrics["ASW_batch"], errors["ASW_batch"] = run_one_metric(
        "ASW_batch",
        lambda: scib.metrics.silhouette_batch(work, batch_key=batch_col, label_key=celltype_col, embed=embed_key),
    )

    metrics["isolated_ASW"], errors["isolated_ASW"] = run_one_metric(
        "isolated_ASW",
        lambda: scib.metrics.isolated_labels_asw(work, batch_key=batch_col, label_key=celltype_col, embed=embed_key),
    )

    metrics["graph_conn"], errors["graph_conn"] = run_one_metric(
        "graph_conn",
        lambda: scib.metrics.graph_connectivity(work, label_key=celltype_col),
    )

    metrics["iLISI"], errors["iLISI"] = run_one_metric(
        "iLISI",
        lambda: scib.metrics.ilisi_graph(work, batch_key=batch_col, type_="knn"),
    )

    metrics["cLISI"], errors["cLISI"] = run_one_metric(
        "cLISI",
        lambda: scib.metrics.clisi_graph(work, label_key=celltype_col, type_="knn"),
    )

    metrics["PCR_batch"], errors["PCR_batch"] = run_one_metric(
        "PCR_batch",
        lambda: scib.metrics.pcr_comparison(adata, work, covariate=batch_col, embed=embed_key),
    )

    return metrics, errors


def copy_existing_timing_csv(old_eval_dir: str, out_dir: str):
    src = os.path.join(old_eval_dir, "all_models_inference_timing.csv")
    if os.path.exists(src):
        dst = os.path.join(out_dir, "all_models_inference_timing.csv")
        shutil.copy2(src, dst)
        print(f"[copy] {src} -> {dst}")
    else:
        print(f"[skip] timing csv not found: {src}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    copy_existing_timing_csv(args.old_eval_dir, args.output_dir)

    model_pairs = [(n, (args.gmm_key if k == "xVerse_gmmvae" else k)) for n, k in MODELS]

    all_rows = []
    all_err_rows = []
    for tissue_name, tissue_dir in [("liver", args.liver_dir), ("brain", args.brain_dir)]:
        for gene_set in ["all", "5k", "xenium"]:
            print(f"\n=== {tissue_name}/{gene_set} ===")
            adata = load_merged_tissue(tissue_dir, tissue_name, gene_set)
            adata = maybe_subsample(adata, args.max_cells, args.seed)

            celltype_col = pick_celltype_col(adata.obs)
            if celltype_col is None:
                print(f"[skip] no cell type column for {tissue_name}/{gene_set}")
                continue
            if "donor_id" not in adata.obs.columns:
                print(f"[skip] no donor_id for {tissue_name}/{gene_set}")
                continue

            for model_name, key in model_pairs:
                if key not in adata.obsm:
                    print(f"[skip] {model_name} key={key} missing")
                    continue
                print(f"[eval] {model_name} key={key} n={adata.n_obs}")
                out = eval_one_embedding(
                    adata=adata,
                    embed_key=key,
                    celltype_col=celltype_col,
                    batch_col="donor_id",
                    neighbors_k=args.neighbors_k,
                )
                if out is None:
                    continue
                metrics, errors = out

                row = {
                    "tissue": tissue_name,
                    "gene_set": gene_set,
                    "model": model_name,
                    "key": key,
                    "n_cells": int(adata.n_obs),
                }
                row.update(metrics)
                all_rows.append(row)

                erow = {
                    "tissue": tissue_name,
                    "gene_set": gene_set,
                    "model": model_name,
                    "key": key,
                }
                erow.update(errors)
                all_err_rows.append(erow)

            df_tg = pd.DataFrame([r for r in all_rows if r["tissue"] == tissue_name and r["gene_set"] == gene_set])
            if not df_tg.empty:
                out_csv = os.path.join(args.output_dir, f"{tissue_name}_{gene_set}_scib_full_metrics.csv")
                df_tg.to_csv(out_csv, index=False)
                print(f"[save] {out_csv}")

    df_all = pd.DataFrame(all_rows)
    df_err = pd.DataFrame(all_err_rows)
    all_csv = os.path.join(args.output_dir, "all_scib_full_metrics.csv")
    err_csv = os.path.join(args.output_dir, "all_scib_metric_errors.csv")
    df_all.to_csv(all_csv, index=False)
    df_err.to_csv(err_csv, index=False)
    print(f"\n[done] metrics: {all_csv}")
    print(f"[done] errors:  {err_csv}")


if __name__ == "__main__":
    main()
