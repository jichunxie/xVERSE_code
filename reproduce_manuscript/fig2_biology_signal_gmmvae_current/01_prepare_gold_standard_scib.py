#!/usr/bin/env python3
"""
Download a small set of public scIB-style benchmark h5ad files and write a
manifest for GMVAE embedding extraction/evaluation.

The script is intentionally conservative: it only downloads missing files and
infers the batch/label/gene columns from common AnnData field names.
"""

import argparse
import os
import urllib.request
from pathlib import Path

import pandas as pd
import scanpy as sc


DATASETS = [
    {
        "dataset": "pancreas_scib",
        "url": "https://figshare.com/ndownloader/files/24539828",
        "filename": "pancreas_scib.h5ad",
        "tissue": "pancreas",
        "notes": "scIB pancreas integration benchmark from Luecken et al.",
    },
    {
        "dataset": "lung_scib",
        "url": "https://figshare.com/ndownloader/files/24539942",
        "filename": "lung_scib.h5ad",
        "tissue": "lung",
        "notes": "scIB lung atlas integration benchmark.",
    },
]


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare gold-standard scIB benchmark manifest.")
    ap.add_argument("--output-root", default="/hpc/group/xielab/xj58/xVerse_results/gold_standard_scib")
    ap.add_argument("--manifest-name", default="gold_standard_manifest.csv")
    ap.add_argument("--skip-download", action="store_true")
    return ap.parse_args()


def download_if_needed(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        print(f"[exists] {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"[download] {url} -> {path}")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def pick_col(obs, candidates):
    for col in candidates:
        if col in obs.columns:
            return col
    return ""


def pick_gene_col(var):
    for col in ["gene_ids", "gene_id", "ensembl_id", "ensembl_ids", "feature_id"]:
        if col in var.columns:
            return col
    return ""


def pick_count_layer(adata):
    for layer in ["counts", "count", "raw_counts", "raw"]:
        if layer in adata.layers:
            return layer
    return "raw" if adata.raw is not None else ""


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in DATASETS:
        fp = out_root / spec["filename"]
        if not args.skip_download:
            download_if_needed(spec["url"], fp)
        if not fp.exists():
            print(f"[skip] missing {fp}")
            continue

        print(f"[inspect] {fp}")
        adata = sc.read_h5ad(fp, backed="r")
        batch_key = pick_col(adata.obs, ["batch", "study", "donor_id", "sample", "sample_id", "tech", "protocol"])
        label_key = pick_col(adata.obs, ["cell_type", "celltype", "cell_type_ontology_term_id", "celltype.l2", "labels"])
        gene_id_col = pick_gene_col(adata.var)
        count_layer = pick_count_layer(adata)
        if not batch_key or not label_key:
            print(f"[warn] {spec['dataset']} inferred batch_key={batch_key!r}, label_key={label_key!r}")

        rows.append(
            {
                "dataset": spec["dataset"],
                "path": str(fp),
                "batch_key": batch_key,
                "label_key": label_key,
                "tissue": spec["tissue"],
                "gene_id_col": gene_id_col,
                "count_layer": count_layer,
                "notes": spec["notes"],
            }
        )
        print(
            f"[ok] {spec['dataset']}: n={adata.n_obs}, genes={adata.n_vars}, "
            f"batch_key={batch_key}, label_key={label_key}, gene_id_col={gene_id_col}, count_layer={count_layer}"
        )
        adata.file.close()

    manifest = out_root / args.manifest_name
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"[done] manifest={manifest}")


if __name__ == "__main__":
    main()
