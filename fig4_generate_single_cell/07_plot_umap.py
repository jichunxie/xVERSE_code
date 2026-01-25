# -*- coding: utf-8 -*-
"""Compare UMI samples via Scanpy workflow and UMAP visualization."""

import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20})


# ================================================================
# Configuration
# ================================================================

SAMPLES = ["PM-A", "PM-B", "PM-C", "PM-D", "PM-E"]
RAW_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad"
XV_INFER_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/ft_model/inference_full"
GENE_IDS_PATH = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

N_SAMPLES = 3
OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/plots/umi_umap_compare"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LEIDEN_KEY = "leiden_label"
SUBTYPE_KEY = "Major.subtype"


# ================================================================
# Helper functions
# ================================================================


def load_allowed_gene_ids(path):
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def read_filtered_h5ad(path, allowed_gene_ids=None):
    ad = sc.read_h5ad(path)
    ad.obs_names_make_unique()
    if "gene_ids" in ad.var:
        ad.var["gene_symbol"] = ad.var_names.astype(str)
        ad.var_names = ad.var["gene_ids"].astype(str)
    else:
        print(f"[WARN] gene_ids missing in {path}; using var_names instead.")
    if allowed_gene_ids is not None:
        keep = ad.var_names.isin(allowed_gene_ids)
        if keep.sum() == 0:
            print(f"[WARN] No genes retained after filtering for {path}; keeping original genes.")
            return ad
        ad = ad[:, keep].copy()
    return ad


def load_raw(sample_name, allowed_gene_ids):
    path = os.path.join(RAW_DIR, f"{sample_name}.h5ad")
    ad = read_filtered_h5ad(path, allowed_gene_ids)
    if ad is None:
        return None
    print(f"    Raw {sample_name}: cells={ad.n_obs}, genes={ad.n_vars}")
    ad.obs["source"] = "raw"
    return ad


def load_xverse(sample_name, allowed_gene_ids):
    path = os.path.join(
        XV_INFER_DIR, sample_name, f"{sample_name}_estimated_expr_poisson.h5ad"
    )
    if not os.path.exists(path):
        print(f"[WARN] Missing XVerse sample: {path}")
        return None
    ad = read_filtered_h5ad(path, allowed_gene_ids)
    if ad is None:
        return None
    print(f"    XVerse {sample_name}: cells={ad.n_obs}, genes={ad.n_vars}")
    splits = np.array_split(np.arange(ad.n_obs), N_SAMPLES)
    subs = []
    for i, idx in enumerate(splits, 1):
        sub = ad[idx, :].copy()
        sub.obs["source"] = f"sample{i}"
        subs.append(sub)
    ad_concat = sc.concat(subs, axis=0, label="concat", index_unique=None)
    return ad_concat


def intersect_genes(ad_list):
    gene_sets = []
    for ad in ad_list:
        if ad is None:
            return None
        if "gene_ids" not in ad.var:
            print("[WARN] gene_ids missing for intersection; adding from var_names")
            ad.var["gene_ids"] = ad.var_names.astype(str)
        gene_sets.append(set(ad.var["gene_ids"].astype(str)))
    common = set.intersection(*gene_sets)
    print(f"    Gene intersections: {[len(s) for s in gene_sets]} -> {len(common)} common")
    if len(common) == 0:
        raise ValueError("No common gene_ids across datasets")
    common = sorted(common)
    subsetted = []
    for ad in ad_list:
        gene_to_idx = {g: i for i, g in enumerate(ad.var["gene_ids"].astype(str))}
        idx = [gene_to_idx[g] for g in common]
        subsetted.append(ad[:, idx].copy())
    return subsetted


def run_scanpy_workflow(adat):
    ad = adat.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True, flavor="seurat_v3")
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=50)
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=30)
    sc.tl.umap(ad)
    return ad


def compute_leiden_labels(adat, resolution=1.0, key=LEIDEN_KEY):
    ad_proc = run_scanpy_workflow(adat)
    sc.tl.leiden(ad_proc, resolution=resolution, key_added=key)
    return ad_proc.obs[key].astype(str)


def assign_leiden_by_order(adat, labels, sample_name, dataset_name):
    label_array = np.asarray(labels, dtype=str)
    n_labels = label_array.size
    if n_labels == 0:
        raise ValueError("Leiden label array is empty.")
    n_cells = adat.n_obs
    repeats = (n_cells + n_labels - 1) // n_labels
    assigned = np.tile(label_array, repeats)[:n_cells]
    if n_cells != n_labels:
        print(
            f"[INFO] {sample_name} – {dataset_name}: assigning {n_cells} cells by repeating {n_labels} raw Leiden labels sequentially."
        )
    adat.obs[LEIDEN_KEY] = assigned


def assign_obs_by_order(adat, values, key, sample_name):
    label_array = np.asarray(values, dtype=str)
    n_labels = label_array.size
    if n_labels == 0:
        adat.obs[key] = "Unknown"
        return
    n_cells = adat.n_obs
    repeats = (n_cells + n_labels - 1) // n_labels
    assigned = np.tile(label_array, repeats)[:n_cells]
    if n_cells != n_labels:
        print(
            f"[INFO] {sample_name}: assigning {n_cells} cells for {key} by repeating {n_labels} reference values sequentially."
        )
    adat.obs[key] = assigned


def plot_umap(ax_source, ax_subtype, ad_source, ad_subtype, color_key):
    sc.pl.umap(ad_source, color="source", ax=ax_source, show=False)
    sc.pl.umap(ad_subtype, color=color_key, ax=ax_subtype, show=False)


# ================================================================
# Main
# ================================================================


def main():
    allowed_gene_ids = load_allowed_gene_ids(GENE_IDS_PATH)
    fig, axes = plt.subplots(len(SAMPLES), 2, figsize=(14, len(SAMPLES) * 4))
    if len(SAMPLES) == 1:
        axes = np.array([axes])

    for row_idx, sample in enumerate(SAMPLES):
        print(f"\n=== Processing {sample} ===")
        ad_raw = load_raw(sample, allowed_gene_ids)
        ad_xv = load_xverse(sample, allowed_gene_ids)

        if ad_raw is None or ad_xv is None:
            print(f"[WARN] Missing data for {sample}; skip.")
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")
            continue

        subset = intersect_genes([ad_raw, ad_xv])
        if subset is None:
            print(f"[WARN] Could not intersect gene sets for {sample}; skip.")
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")
            continue
        ad_raw_i, ad_xv_i = subset

        if SUBTYPE_KEY not in ad_raw_i.obs:
            ad_raw_i.obs[SUBTYPE_KEY] = "Unknown"
        raw_subtypes = ad_raw_i.obs[SUBTYPE_KEY].astype(str).values
        if SUBTYPE_KEY not in ad_xv_i.obs:
            assign_obs_by_order(ad_xv_i, raw_subtypes, SUBTYPE_KEY, sample)
        if "source" not in ad_raw_i.obs:
            ad_raw_i.obs["source"] = "Biological Cells"
        else:
            ad_raw_i.obs["source"] = "Biological Cells"
        ad_xv_i.obs["source"] = ad_xv_i.obs["source"].astype(str).str.replace("sample", "Virtual C")

        sc.pp.normalize_total(ad_raw_i, target_sum=1e4)
        sc.pp.log1p(ad_raw_i)
        sc.pp.normalize_total(ad_xv_i, target_sum=1e4)
        sc.pp.log1p(ad_xv_i)

        combined = sc.concat([ad_raw_i, ad_xv_i], axis=0, label="dataset", keys=["Biological Cells", "xVERSE"], index_unique=None)
        combined.obs["source"] = combined.obs["source"].astype(str)

        ad_proc = run_scanpy_workflow(combined)

        ax_source = axes[row_idx, 0]
        sc.pl.umap(ad_proc, color="source", ax=ax_source, show=False, title="Source", size=1)

        ax_subtype = axes[row_idx, 1]
        if SUBTYPE_KEY not in ad_proc.obs:
            ad_proc.obs[SUBTYPE_KEY] = ad_proc.obs["source"]
        sc.pl.umap(ad_proc, color=SUBTYPE_KEY, ax=ax_subtype, show=False, title=SUBTYPE_KEY, size=1)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "xverse_umap_grid.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved UMAP grid to {out_path}")


if __name__ == "__main__":
    main()
