#!/usr/bin/env python3
"""
Plot UMAPs for a given embedding key from fig2 donor h5ad files.
Default uses full panel ("all") and GMVAE mixmu embedding.
"""

import argparse
import os
from pathlib import Path

import scanpy as sc
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Plot UMAP for fig2 embeddings.")
    ap.add_argument("--liver-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/liver")
    ap.add_argument("--brain-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/brain")
    ap.add_argument("--output-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2_gmmvae_current/umap")
    ap.add_argument("--embedding-key", default="xVerse_gmmvae_mixmu")
    ap.add_argument("--gene-set", default="all", choices=["all", "5k", "xenium"])
    ap.add_argument("--neighbors-k", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-cells", type=int, default=0, help="0 means all cells.")
    return ap.parse_args()


def pick_celltype_col(obs):
    for col in ["cell_type", "celltype", "cell_type_ontology_term_id", "celltype.l2"]:
        if col in obs.columns:
            return col
    return None


def load_merge_tissue(tissue_dir: str, tissue_name: str, gene_set: str):
    files = sorted(Path(tissue_dir).glob(f"{tissue_name}_*_{gene_set}.h5ad"))
    if not files:
        raise FileNotFoundError(f"No files found: {tissue_name}/{gene_set} in {tissue_dir}")
    lst = []
    for fp in files:
        ad = sc.read_h5ad(fp)
        donor_id = fp.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
        if "donor_id" not in ad.obs.columns:
            ad.obs["donor_id"] = donor_id
        lst.append(ad)
    return sc.concat(lst, join="outer", index_unique=None)


def maybe_subsample(adata, max_cells: int, seed: int):
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    idx = sc.pp.subsample(adata, n_obs=max_cells, random_state=seed, copy=True).obs_names
    return adata[idx].copy()


def plot_one(adata, tissue_name: str, embedding_key: str, out_dir: str, gene_set: str, neighbors_k: int, min_dist: float, seed: int):
    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key} not found in adata.obsm")

    celltype_col = pick_celltype_col(adata.obs)
    if celltype_col is None:
        raise ValueError("No cell type column found.")

    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=neighbors_k)
    sc.tl.umap(adata, min_dist=min_dist, random_state=seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(
        adata,
        color="donor_id",
        ax=axes[0],
        show=False,
        frameon=False,
        title=f"{tissue_name} ({gene_set}) - donor_id",
        legend_loc="right margin",
    )
    sc.pl.umap(
        adata,
        color=celltype_col,
        ax=axes[1],
        show=False,
        frameon=False,
        title=f"{tissue_name} ({gene_set}) - {celltype_col}",
        legend_loc="right margin",
    )
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{tissue_name}_{gene_set}_{embedding_key}_umap.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_png}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for tissue_name, tissue_dir in [("liver", args.liver_dir), ("brain", args.brain_dir)]:
        print(f"[Load] {tissue_name}/{args.gene_set}")
        adata = load_merge_tissue(tissue_dir, tissue_name, args.gene_set)
        adata = maybe_subsample(adata, args.max_cells, args.seed)
        print(f"[Info] {tissue_name}: n_cells={adata.n_obs}, n_genes={adata.n_vars}")
        plot_one(
            adata=adata,
            tissue_name=tissue_name,
            embedding_key=args.embedding_key,
            out_dir=args.output_dir,
            gene_set=args.gene_set,
            neighbors_k=args.neighbors_k,
            min_dist=args.min_dist,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

