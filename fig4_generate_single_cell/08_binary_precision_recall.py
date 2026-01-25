#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train per-sample binary classifiers (Raw vs generated samples) and plot ROC curves."""

import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

plt.rcParams.update({"font.size": 24})

TITLE_FONTSIZE = 30
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 20
ANNOTATION_FONTSIZE = 22
PANEL_SIZE = 5.5  # inches per panel to preserve square axes despite larger fonts


SAMPLES = ["PM-A", "PM-B", "PM-C", "PM-D", "PM-E"]
RAW_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad"
XV_INFER_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/ft_model/inference_full"
SCVI_SAMPLE_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/scvi_full/umi_samples_full"
GENE_IDS_PATH = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

N_SAMPLES = 3
OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/plots"
CACHE_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/cache/umi_binary_classifiers"
PLOT_PATH = os.path.join(OUTPUT_DIR, "umi_binary_roc.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def load_allowed_gene_ids(path: str) -> set:
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def read_filtered_h5ad(path: str, allowed_gene_ids: set):
    ad = sc.read_h5ad(path)
    ad.obs_names_make_unique()
    if "gene_ids" in ad.var:
        ad.var["gene_symbol"] = ad.var_names.astype(str)
        ad.var_names = ad.var["gene_ids"].astype(str)
    keep = ad.var_names.isin(list(allowed_gene_ids))
    if keep.sum() == 0:
        return None
    return ad[:, keep].copy()


def load_raw(sample: str, allowed_gene_ids: set):
    path = os.path.join(RAW_DIR, f"{sample}.h5ad")
    return read_filtered_h5ad(path, allowed_gene_ids)


def load_xverse_samples(sample: str, allowed_gene_ids: set) -> List[sc.AnnData]:
    path = os.path.join(
        XV_INFER_DIR, sample, f"{sample}_estimated_expr_poisson.h5ad"
    )
    if not os.path.exists(path):
        return []
    ad = read_filtered_h5ad(path, allowed_gene_ids)
    if ad is None:
        return []
    splits = np.array_split(np.arange(ad.n_obs), N_SAMPLES)
    return [ad[idx, :].copy() for idx in splits]


def load_scvi_samples(sample: str, allowed_gene_ids: set) -> List[sc.AnnData]:
    samples = []
    for i in range(1, N_SAMPLES + 1):
        path = os.path.join(
            SCVI_SAMPLE_DIR, sample, f"{sample}_umi_sample{i}_full.h5ad"
        )
        if not os.path.exists(path):
            continue
        ad = read_filtered_h5ad(path, allowed_gene_ids)
        if ad is not None:
            samples.append(ad)
    return samples


def flatten_counts(ad):
    X = ad.X
    return X.toarray() if hasattr(X, "toarray") else X


def common_gene_list(*ann_datas: sc.AnnData) -> List[str]:
    genes = None
    for ad in ann_datas:
        if ad is None:
            return []
        gene_set = set(ad.var_names)
        genes = gene_set if genes is None else genes & gene_set
    return sorted(genes) if genes else []


def prepare_binary_dataset(ad_raw, ad_sample, genes: List[str]):
    raw = flatten_counts(ad_raw[:, genes])
    sample = flatten_counts(ad_sample[:, genes])
    X = np.vstack([raw, sample])
    y = np.array(["raw"] * raw.shape[0] + ["sample"] * sample.shape[0])
    return X, y


def cache_path(sample: str, model: str, idx: int) -> str:
    return os.path.join(CACHE_DIR, f"{sample}_{model}_sample{idx}.pkl")


def load_cached_result(sample: str, model: str, idx: int, genes: List[str]):
    path = cache_path(sample, model, idx)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load cache {path}: {exc}")
        return None
    if data.get("genes") != genes:
        print(f"[INFO] Cache gene mismatch for {sample}-{model}-sample{idx}; retraining.")
        return None
    return data.get("result")


def save_cached_result(sample: str, model: str, idx: int, result: dict, genes: List[str]):
    path = cache_path(sample, model, idx)
    with open(path, "wb") as f:
        pickle.dump({"result": result, "genes": genes}, f)


def train_binary_classifier(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    raw_idx = classes.index("raw") if "raw" in classes else 0
    scores = probs[:, raw_idx]
    y_true = (y_test == "raw").astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}


def collect_metrics():
    allowed_gene_ids = load_allowed_gene_ids(GENE_IDS_PATH)
    all_metrics = {sample: {i: {} for i in range(1, N_SAMPLES + 1)} for sample in SAMPLES}

    for sample in SAMPLES:
        print(f"\n=== {sample} ===")
        ad_raw = load_raw(sample, allowed_gene_ids)
        xv_samples = load_xverse_samples(sample, allowed_gene_ids)
        sv_samples = load_scvi_samples(sample, allowed_gene_ids)

        if ad_raw is None or len(xv_samples) < N_SAMPLES or len(sv_samples) < N_SAMPLES:
            print(f"[WARN] Missing data for {sample}; skip.")
            continue

        def print_stats(name, datasets):
            counts = np.array([ds.n_obs for ds in datasets])
            genes = np.array([ds.n_vars for ds in datasets])
            count_q = np.quantile(counts, [0.25, 0.5, 0.75])
            gene_q = np.quantile(genes, [0.25, 0.5, 0.75])
            print(
                f"    {name}: cells min/med/max = {counts.min()}/{np.median(counts)}/{counts.max()} | "
                f"quantiles (25/50/75%) = {count_q[0]:.0f}/{count_q[1]:.0f}/{count_q[2]:.0f}; "
                f"genes min/med/max = {genes.min()}/{np.median(genes)}/{genes.max()} | "
                f"quantiles = {gene_q[0]:.0f}/{gene_q[1]:.0f}/{gene_q[2]:.0f}"
            )

        print(f"Sample {sample}: raw cells={ad_raw.n_obs}, genes={ad_raw.n_vars}")
        print_stats("xVERSE", xv_samples)
        print_stats("SCVI", sv_samples)

        def print_matrix_quantiles(name, datasets):
            print(f"    {name} total-count quantiles (25/50/75%)")
            for idx, ds in enumerate(datasets, 1):
                totals = np.asarray(ds.X.sum(axis=1)).ravel()
                q = np.quantile(totals, [0.25, 0.5, 0.75])
                print(f"        sample{idx}: {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}")

        print_matrix_quantiles("Raw", [ad_raw])
        print_matrix_quantiles("xVERSE", xv_samples)
        print_matrix_quantiles("SCVI", sv_samples)

        for model_name, sample_list in [("xVERSE", xv_samples), ("SCVI", sv_samples)]:
            for idx, ad_sample in enumerate(sample_list, 1):
                genes = common_gene_list(ad_raw, ad_sample)
                if not genes:
                    print(f"[WARN] {sample}-{model_name}-sample{idx}: no common genes.")
                    continue
                result = load_cached_result(sample, model_name, idx, genes)
                if result is None:
                    X, y = prepare_binary_dataset(ad_raw, ad_sample, genes)
                    if X.size == 0:
                        continue
                    result = train_binary_classifier(X, y)
                    save_cached_result(sample, model_name, idx, result, genes)
                all_metrics[sample][idx][model_name] = result
    return all_metrics


def plot_metrics(metrics: Dict[str, Dict[int, Dict[str, dict]]]):
    fig, axes = plt.subplots(
        len(SAMPLES),
        N_SAMPLES,
        figsize=(N_SAMPLES * PANEL_SIZE, len(SAMPLES) * PANEL_SIZE),
        sharex=True,
        sharey=True,
    )
    if len(SAMPLES) == 1:
        axes = np.array([axes])

    colors = {"xVERSE": "#1f77b4", "SCVI": "#ff7f0e"}

    for row, sample in enumerate(SAMPLES):
        sample_data = metrics.get(sample, {})
        for col in range(N_SAMPLES):
            ax = axes[row, col]
            generation = col + 1
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            has_curve = False
            for model_name in ["xVERSE", "SCVI"]:
                data = sample_data.get(generation, {}).get(model_name)
                if not data:
                    continue
                ax.plot(data["fpr"], data["tpr"], color=colors[model_name], linewidth=2.5)
                ax.text(
                    0.55,
                    0.15 + 0.15 * (["xVERSE", "SCVI"].index(model_name)),
                    f"{model_name} AUC={data['auc']:.2f}",
                    transform=ax.transAxes,
                    color=colors[model_name],
                    fontsize=ANNOTATION_FONTSIZE,
                )
                has_curve = True
            if not has_curve:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=ANNOTATION_FONTSIZE)
            ax.set_title(f"Virtual C{generation}", fontsize=TITLE_FONTSIZE)
            if row == len(SAMPLES) - 1:
                ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
            if col == 0:
                ax.set_ylabel("True Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved ROC comparison to {PLOT_PATH}")


def main():
    metrics = collect_metrics()
    plot_metrics(metrics)


if __name__ == "__main__":
    main()
