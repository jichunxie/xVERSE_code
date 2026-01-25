# Copyright (C) 2024 Xiaohui Jiang and Jichun Xie
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python
"""Plot Poisson copy sweep metrics: Pearson/MSE boxes and F1/Recall curves."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_ROOT = Path("/hpc/group/xielab/xj58/xVerse_results/fig7/crossmodal_datasets_GSE291290")
METRIC_DIR = DATA_ROOT / "metrics"
PLOT_DIR = METRIC_DIR / "plots"

POISSON_COUNTS = [0] + list(range(1, 21))  # 0 = baseline
SPLITS = ["ngd", "cav"]
TASKS = ["BCR_Paired_Chains", "TCR_Paired_Chains"]
TASKS_05 = [
    "BCR_Heavy_C_gene_Dominant",
    "TCR_Alpha_Gamma_C_gene_Dominant",
]
TICKS = [0, 5, 10, 15, 20]

plt.rcParams.update({"font.size": 20})


def find_metric_files(pattern: str):
    return sorted(METRIC_DIR.glob(pattern))


def parse_copy_from_model(model_name: str) -> int:
    match = re.search(r"augmented_copy(\d+)", model_name)
    if match:
        return int(match.group(1))
    return 0  # baseline


def load_pearson(split: str) -> pd.DataFrame:
    records = []
    for path in find_metric_files(f"*_test_{split}_pearson.csv"):
        model = path.stem.replace(f"_test_{split}_pearson", "")
        copy = parse_copy_from_model(model)
        df = pd.read_csv(path)
        df["copy_count"] = copy
        df["model_name"] = model
        records.append(df[["pearson_corr", "copy_count", "model_name"]])
    if records:
        return pd.concat(records, axis=0, ignore_index=True)
    return pd.DataFrame(columns=["pearson_corr", "copy_count", "model_name"])


def load_mse(split: str) -> pd.DataFrame:
    records = []
    for path in find_metric_files(f"*_test_{split}_mse.csv"):
        model = path.stem.replace(f"_test_{split}_mse", "")
        copy = parse_copy_from_model(model)
        mse_val = pd.read_csv(path)["mse"].iloc[0]
        records.append({"copy_count": copy, "model_name": model, "mse": mse_val})
    return pd.DataFrame(records)





def load_loss_auc(task: str, split: str) -> pd.DataFrame:
    path = METRIC_DIR / "mlp_vdj_cgene_dominance.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[(df["task"] == task) & (df["split"] == f"test_{split}")]
    return df


def scatter_with_trend(x, y, ax, label, color):
    ax.scatter(x, y, color=color, label=label)
    if len(x) >= 3:
        order = np.argsort(x)
        x_sorted = np.asarray(x)[order]
        y_sorted = np.asarray(y)[order]
        smooth = sm.nonparametric.lowess(y_sorted, x_sorted, frac=0.6, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color=color, linestyle="--")
    elif len(x) == 2:
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(min(x), max(x), 50)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color=color, linestyle="--")


def mark_best(ax, x_vals, y_vals, mode: str):
    if len(x_vals) == 0:
        return None
    if mode == "max":
        idx = int(np.nanargmax(y_vals))
    else:
        idx = int(np.nanargmin(y_vals))
    ax.scatter([x_vals[idx]], [y_vals[idx]], color="red", s=60, marker="*", label="Best")
    ax.annotate(f"best {int(x_vals[idx])}", (x_vals[idx], y_vals[idx]), textcoords="offset points", xytext=(-45, 5), color="red")
    return idx


def mark_baseline(ax, x_vals, y_vals, mode: str):
    if len(x_vals) == 0:
        return None
    mask = x_vals == 0
    if not np.any(mask):
        return None
    ref = float(np.nanmean(y_vals[mask]))
    ax.axhline(ref, color="red", linestyle=":", linewidth=1.5)
    ax.text(0, ref, " no xVERSE augmentation", va="bottom", ha="left", color="red")
    return ref


def apply_xtick_format(ax):
    ax.set_xticks(TICKS)
    labels = ["0" if t == 0 else str(t) for t in TICKS]
    ax.set_xticklabels(labels, rotation=0)


def plot_pearson(split: str, ax):
    pearson_df = load_pearson(split)
    if pearson_df.empty:
        ax.set_title(f"{split.upper()} Pearson (no data)")
        return
    pearson_df = pearson_df[pearson_df["copy_count"].isin(POISSON_COUNTS)]
    stats = pearson_df.groupby("copy_count")["pearson_corr"].median().reset_index()
    y = stats["pearson_corr"].values
    x = stats["copy_count"].values
    ax.scatter(x, y, color="red")
    scatter_with_trend(x, y, ax, "", "red")
    best_idx = mark_best(ax, x, y, mode="max")
    ref = mark_baseline(ax, x, y, mode="max")
    if best_idx is not None and ref is not None:
        ax.vlines(x[best_idx], min(y[best_idx], ref), max(y[best_idx], ref), colors="red", linestyles="-")
        delta_pct = ((y[best_idx] - ref) / ref * 100) if ref != 0 else float("inf")
        ax.annotate(
            f"{delta_pct:+.1f}%",
            (x[best_idx], (y[best_idx] + ref) / 2),
            textcoords="offset points",
            xytext=(-45, -10),
            color="red",
        )
    ax.set_xlabel("")
    ax.set_ylabel("Pearson")
    ax.set_xlim(-0.5, 20.5)
    apply_xtick_format(ax)
    ax.legend().set_visible(False)


def plot_mse(split: str, ax):
    mse_df = load_mse(split)
    if mse_df.empty:
        ax.set_title(f"{split.upper()} MSE (no data)")
        return
    mse_df = mse_df[mse_df["copy_count"].isin(POISSON_COUNTS)]
    mse_df = mse_df.sort_values("copy_count")
    scatter_with_trend(mse_df["copy_count"].values, mse_df["mse"].values, ax, "", "green")
    best_idx = mark_best(ax, mse_df["copy_count"].values, mse_df["mse"].values, mode="min")
    ref = mark_baseline(ax, mse_df["copy_count"].values, mse_df["mse"].values, mode="min")
    if best_idx is not None and ref is not None:
        x_arr = mse_df["copy_count"].values
        y_arr = mse_df["mse"].values
        ax.vlines(x_arr[best_idx], min(y_arr[best_idx], ref), max(y_arr[best_idx], ref), colors="red", linestyles="-")
        delta_pct = ((y_arr[best_idx] - ref) / ref * 100) if ref != 0 else float("inf")
        ax.annotate(
            f"{delta_pct:+.1f}%",
            (x_arr[best_idx], (y_arr[best_idx] + ref) / 2),
            textcoords="offset points",
            xytext=(-45, -10),
            color="red",
        )
    ax.set_xlabel("")
    ax.set_ylabel("MSE")
    ax.set_xlim(-0.5, 20.5)
    apply_xtick_format(ax)





def plot_loss_auc(task: str, split: str, metric: str, ax):
    df = load_loss_auc(task, split)
    if df.empty:
        ax.set_title(f"{task} {split.upper()} {metric} (no data)")
        return
    df = df[(df["copy_count"].isin(POISSON_COUNTS)) & (df["metric"] == metric)]
    df = df.sort_values("copy_count")
    y = df["value"].values
    x = df["copy_count"].values
    color = "green" if metric == "loss" else "red"
    scatter_with_trend(x, y, ax, "", color)
    mode = "min" if metric == "loss" else "max"
    best_idx = mark_best(ax, x, y, mode=mode)
    ref = mark_baseline(ax, x, y, mode=mode)
    if best_idx is not None and ref is not None:
        ax.vlines(x[best_idx], min(y[best_idx], ref), max(y[best_idx], ref), colors="red", linestyles="-")
        delta_pct = ((y[best_idx] - ref) / ref * 100) if ref != 0 else float("inf")
        ax.annotate(
            f"{delta_pct:+.1f}%",
            (x[best_idx], (y[best_idx] + ref) / 2),
            textcoords="offset points",
            xytext=(-45, -10),
            color="red",
        )
    ax.set_xlabel("")
    if metric == "loss":
        ax.set_ylabel("CrossEntropy (weighted)")
    else:
        ax.set_ylabel(metric.upper())
    ax.set_xlim(-0.5, 20.5)
    apply_xtick_format(ax)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = 1 + len(TASKS_05)  # row0: mse/pearson, rest: task loss/auc
    fig_size = (12, rows * 4)

    for split in ["ngd", "cav"]:
        fig, axes = plt.subplots(rows, 2, figsize=fig_size, constrained_layout=True)
        # Row1
        plot_mse(split, axes[0, 0])
        plot_pearson(split, axes[0, 1])
        # Rows for 05 tasks
        for idx, task in enumerate(TASKS_05, start=1):
            plot_loss_auc(task, split, "loss", axes[idx, 0])
            plot_loss_auc(task, split, "auc", axes[idx, 1])
        out_path = PLOT_DIR / f"{split}_combined.png"
        fig.savefig(out_path, dpi=300)
        print(f"[IO] Saved {out_path}")


if __name__ == "__main__":
    main()
