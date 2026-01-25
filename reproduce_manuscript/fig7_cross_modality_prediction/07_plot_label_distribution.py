#!/usr/bin/env python
"""Plot label distributions for VDJ C-gene dominance tasks."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


DATA_ROOT = Path("/hpc/group/xielab/xj58/xVerse_results/fig7/crossmodal_datasets_GSE291290")
TRAIN_BASE = DATA_ROOT / "original_train.h5ad"
VAL_FILE = DATA_ROOT / "original_val.h5ad"
TEST_NGD_FILE = DATA_ROOT / "original_test_ngd.h5ad"
TEST_CAV_FILE = DATA_ROOT / "original_test_cav.h5ad"
PLOT_DIR = DATA_ROOT / "metrics" / "plots"

LABEL_CANDIDATES = {
    "BCR_Heavy_C_gene_Dominant": ["BCR_Heavy_C_gene_Dominant"],
    "BCR_Light_C_gene_Dominant": ["BCR_Light_C_gene_Dominant"],
    "TCR_Alpha_Gamma_C_gene_Dominant": ["TCR_Alpha_Gamma_C_gene_Dominant"],
}

PLOT_TASKS = [
    "BCR_Heavy_C_gene_Dominant",
    "TCR_Alpha_Gamma_C_gene_Dominant",
]

MIN_TRAIN_COUNT = 10

plt.rcParams.update({"font.size": 18})


def normalize_label(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def find_label_columns(adata: sc.AnnData) -> Dict[str, str]:
    resolved = {}
    for logical, candidates in LABEL_CANDIDATES.items():
        for cand in candidates:
            if cand in adata.obs.columns:
                resolved[logical] = cand
                break
    missing = [k for k in LABEL_CANDIDATES if k not in resolved]
    if missing:
        raise KeyError(f"Missing label columns in obs: {missing}")
    return resolved


def load_labels(path: Path, label_cols: Dict[str, str]) -> Dict[str, np.ndarray]:
    adata = sc.read_h5ad(path)
    labels: Dict[str, np.ndarray] = {}
    for logical, col in label_cols.items():
        arr = adata.obs[col].map(normalize_label).to_numpy(object)
        labels[logical] = arr
    return labels


def filter_known(y: np.ndarray, classes: List[str]):
    mask = np.isin(y, classes)
    return y[mask]


def compute_classes(y_train: np.ndarray) -> List[str]:
    uniq, counts = np.unique(y_train, return_counts=True)
    return [c for c, n in zip(uniq, counts) if n >= MIN_TRAIN_COUNT]


def plot_task(task: str, task_data: Dict[str, np.ndarray], ax):
    cmap = {
        "BCR_Heavy_C_gene_Dominant": plt.cm.Blues,
        "TCR_Alpha_Gamma_C_gene_Dominant": plt.cm.Oranges,
    }.get(task, plt.cm.tab20)

    classes = task_data["classes"]
    colors = cmap(np.linspace(0.4, 0.9, max(len(classes), 2)))
    split_order = ["train", "val", "test_ngd", "test_cav"]
    split_labels = [
        "Normal HTx (Train)",
        "Normal HTx (Val)",
        "NGD",
        "CAV",
    ]

    bar_x = np.arange(len(split_order))
    bar_width = 0.6
    handles = []
    labels = []

    for i, split_name in enumerate(split_order):
        arr = task_data[split_name]
        bottom = 0.0
        for idx, cls in enumerate(classes):
            total = len(arr)
            frac = (arr == cls).sum() / total if total > 0 else 0.0
            if frac == 0:
                continue
            h = ax.bar(bar_x[i], frac, bar_width, bottom=bottom, color=colors[idx])
            bottom += frac
            if cls not in labels:
                handles.append(h[0])
                labels.append(cls)

    ax.set_xticks(bar_x)
    ax.set_xticklabels(split_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    title_map = {
        "BCR_Heavy_C_gene_Dominant": "B-cell Heavy-Chain Isotype Distribution",
        "TCR_Alpha_Gamma_C_gene_Dominant": "αβ vs γδ T-cell Lineage Distribution",
    }
    ax.set_title(title_map.get(task, task))
    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", title="Class")



def load_adt_data(path: Path) -> np.ndarray:
    adata = sc.read_h5ad(path)
    if "protein" not in adata.obsm:
        print(f"[Warn] No 'protein' obsm in {path}")
        return np.array([])
    return adata.obsm["protein"]


def plot_adt_violin(adt_splits: Dict[str, np.ndarray], ax):
    split_order = ["train", "val", "test_ngd", "test_cav"]
    split_labels = [
        "Normal HTx (Train)",
        "Normal HTx (Val)",
        "NGD",
        "CAV",
    ]
    
    plot_data = []
    labels = []
    
    for name in split_order:
        data = adt_splits.get(name, np.array([]))
        if data.size > 0:
            # Flatten to show global distribution of protein values
            flat = data.flatten()
            plot_data.append(flat)
        else:
            plot_data.append([])
        labels.append(name)

    # Violin Plot
    ax.violinplot(plot_data, showmeans=False, showmedians=True)
    
    # Styling
    ax.set_xticks(np.arange(1, len(split_labels) + 1))
    ax.set_xticklabels(split_labels, rotation=20, ha="right")
    ax.set_ylabel("ADT Protein (Z-scored)")
    ax.set_title("ADT Protein Distribution")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    base_adata = sc.read_h5ad(TRAIN_BASE)
    label_cols = find_label_columns(base_adata)
    print(f"[Info] Using label columns: {label_cols}")

    labels_train = load_labels(TRAIN_BASE, label_cols)
    labels_val = load_labels(VAL_FILE, label_cols)
    labels_test_ngd = load_labels(TEST_NGD_FILE, label_cols)
    labels_test_cav = load_labels(TEST_CAV_FILE, label_cols)

    # Load ADT Data
    adt_splits = {
        "train": load_adt_data(TRAIN_BASE),
        "val": load_adt_data(VAL_FILE),
        "test_ngd": load_adt_data(TEST_NGD_FILE),
        "test_cav": load_adt_data(TEST_CAV_FILE),
    }

    task_splits: Dict[str, Dict[str, np.ndarray]] = {}
    for task in PLOT_TASKS:
        y_train = labels_train[task][pd.notna(labels_train[task])]
        keep_classes = compute_classes(y_train)
        if len(keep_classes) < 2:
            continue
        task_splits[task] = {
            "classes": sorted(keep_classes),
            "train": filter_known(y_train, keep_classes),
            "val": filter_known(labels_val[task][pd.notna(labels_val[task])], keep_classes),
            "test_ngd": filter_known(labels_test_ngd[task][pd.notna(labels_test_ngd[task])], keep_classes),
            "test_cav": filter_known(labels_test_cav[task][pd.notna(labels_test_cav[task])], keep_classes),
        }

    if not task_splits:
        print("[Warn] No tasks with >=2 classes after filtering.")
        return

    # Plot: 1 row for ADT + rows for each task
    n_rows = 1 + len(task_splits)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), constrained_layout=True)
    
    if n_rows == 1:
        axes = [axes]
    
    # 1. Plot ADT Violin (First Row)
    print("\n[Plot] ADT Distribution")
    plot_adt_violin(adt_splits, axes[0])

    # 2. Plot Tasks
    for i, (task, data) in enumerate(task_splits.items()):
        ax = axes[i + 1]
        print(f"\n[Task] {task}")
        for split_name in ["train", "val", "test_ngd", "test_cav"]:
            arr = data[split_name]
            uniq, counts = np.unique(arr, return_counts=True)
            print(f"  {split_name}: {len(arr)} samples | {dict(zip(uniq, counts))}")
        plot_task(task, data, ax)

    out_path = PLOT_DIR / "distribution_combined.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[IO] Saved {out_path}")


if __name__ == "__main__":
    main()
