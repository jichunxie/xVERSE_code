# -*- coding: utf-8 -*-
"""
Evaluate XVerse and SCVI by computing median gene-wise Pearson correlations
against raw expression.

Rules:
    - raw / XVerse / SCVI all aligned ONLY by var["gene_ids"]
    - var_names are ignored completely
    - HVG is computed AFTER alignment
    - Pearson is computed gene-wise: corr(true[:, g], model[:, g])
    - Report median Pearson for each HVG size

All comments are in English.
"""

import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

plt.rcParams.update({"font.size": 20})


# -------------------------------------------------------
# Config
# -------------------------------------------------------

SAMPLES = ["PM-A", "PM-B", "PM-C", "PM-D", "PM-E"]

RAW_DIR   = "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad"
XV_DIR    = "/hpc/group/xielab/xj58/xVerse_results/fig4/ft_model"
SCVI_DIR  = "/hpc/group/xielab/xj58/xVerse_results/fig4/scvi_full"

MODEL_LABEL = "full"
HVG_LIST  = [50, 100, 500, 1000]
PLOT_SUBSET = MODEL_LABEL

PLOT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig4/plots"
os.makedirs(PLOT_DIR, exist_ok=True)
PLOT_FILE = os.path.join(PLOT_DIR, f"pearson_model_compare_{MODEL_LABEL}.png")


# -------------------------------------------------------
# Utils
# -------------------------------------------------------

def compute_gene_wise_pearson(raw_mat, model_mat, gene_idx):
    """Compute Pearson per gene; return list of values."""
    vals = []
    for g in gene_idx:
        x = raw_mat[:, g]
        y = model_mat[:, g]

        if np.all(x == 0) or np.all(y == 0):
            continue

        try:
            r, _ = pearsonr(x, y)
            if not np.isnan(r):
                vals.append(r)
        except Exception:
            continue

    return vals


def select_hvg_after_alignment(adata_raw, hvg_n):
    """Select HVGs on aligned raw matrix."""
    ad = adata_raw.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=hvg_n, flavor="seurat_v3")
    mask = ad.var["highly_variable"].values
    idx = np.where(mask)[0]
    return idx[:hvg_n]


def plot_sample_comparisons(records, subset_filter, output_path):
    """Plot grouped boxplots (XVerse vs SCVI) per sample stacked in one figure."""
    subset_records = [
        rec
        for rec in records
        if rec["subset"] == subset_filter and not np.isnan(rec["median_r"])
    ]
    if len(subset_records) == 0:
        print(f"[WARN] No records for subset {subset_filter}; skip plotting.")
        return

    hvgs = sorted({rec["hvg"] for rec in subset_records})
    if not hvgs:
        print(f"[WARN] No HVG data collected for subset {subset_filter}.")
        return
    models = ["xVERSE", "SCVI"]

    fig, axes = plt.subplots(
        len(SAMPLES), 1, figsize=(10, len(SAMPLES) * 3.5), sharex=True
    )
    if len(SAMPLES) == 1:
        axes = [axes]

    color_map = {"xVERSE": "#1f77b4", "SCVI": "#ff7f0e"}
    group_width = 0.6
    offsets = np.linspace(
        -group_width / 2 + group_width / (2 * len(models)),
        group_width / 2 - group_width / (2 * len(models)),
        len(models),
    )

    last_valid_axis = None
    legend_axis = None
    for ax, sample in zip(axes, SAMPLES):
        sample_records = [
            rec for rec in subset_records if rec["sample"] == sample
        ]
        if len(sample_records) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{sample}: no data", ha="center", va="center")
            continue

        box_positions = []
        box_data = []
        box_colors = []
        for i, h in enumerate(hvgs):
            center = i + 1
            for j, model in enumerate(models):
                rec = next(
                    (r for r in sample_records if r["model"] == model and r["hvg"] == h),
                    None,
                )
                if rec is None or len(rec.get("values", [])) == 0:
                    continue
                box_positions.append(center + offsets[j])
                box_data.append(rec["values"])
                box_colors.append(color_map[model])

        if len(box_data) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{sample}: insuf. data", ha="center", va="center")
            continue

        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=group_width / len(models) * 0.9,
            patch_artist=True,
            manage_ticks=False,
        )

        for box, color in zip(bp["boxes"], box_colors):
            box.set(facecolor=color, edgecolor="black", alpha=0.5)
        for whisker in bp["whiskers"]:
            whisker.set(color="black")
        for cap in bp["caps"]:
            cap.set(color="black")
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.5)

        ax.set_xticks(np.arange(1, len(hvgs) + 1))
        ax.set_xticklabels([str(h) for h in hvgs])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Pearson")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if sample != SAMPLES[-1]:
            ax.tick_params(labelbottom=False)
        last_valid_axis = ax
        if legend_axis is None:
            legend_axis = ax

    if last_valid_axis is None:
        print(f"[WARN] No valid axes to plot for subset {subset_filter}.")
        plt.close(fig)
        return

    last_valid_axis.set_xlabel("HVG count")

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color_map[m],
            markeredgecolor="black",
            markersize=10,
            label=m,
        )
        for m in models
    ]
    if legend_axis is not None:
        legend_axis.legend(handles=handles, loc="upper right")

    # fig.suptitle(f"Gene-wise Pearson comparison ({subset_filter})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved stacked comparison to {output_path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    records = []

    for sample in SAMPLES:
        print(f"\n===================== Sample {sample} =====================")

        # -------- Load RAW --------
        raw_path = f"{RAW_DIR}/{sample}.h5ad"
        ad_raw = sc.read_h5ad(raw_path)

        raw_expr = ad_raw.X.toarray() if hasattr(ad_raw.X, "toarray") else ad_raw.X
        raw_gene_ids = ad_raw.var["gene_ids"].astype(str).tolist()
        raw_map = {g: i for i, g in enumerate(raw_gene_ids)}

        subset = MODEL_LABEL
        print(f"\n----- subset {subset} -----")

        # -------- Load XVerse --------
        xv_path = f"{XV_DIR}/inference_{subset}/{sample}/{sample}_mu_bio.h5ad"
        if not os.path.exists(xv_path):
            print(f"[skip] no XVerse file {xv_path}")
            continue

        ad_xv = sc.read_h5ad(xv_path)
        xv_expr = ad_xv.X.toarray() if hasattr(ad_xv.X, "toarray") else ad_xv.X
        xv_gene_ids = ad_xv.var["gene_ids"].astype(str).tolist()
        xv_map = {g: i for i, g in enumerate(xv_gene_ids)}

        # -------- Load SCVI --------
        sv_path = f"{SCVI_DIR}/{sample}_scale_{subset}.h5ad"
        if not os.path.exists(sv_path):
            print(f"[skip] no SCVI file {sv_path}")
            continue

        ad_sv = sc.read_h5ad(sv_path)
        sv_expr = ad_sv.X.toarray() if hasattr(ad_sv.X, "toarray") else ad_sv.X
        sv_gene_ids = ad_sv.var["gene_ids"].astype(str).tolist()
        sv_map = {g: i for i, g in enumerate(sv_gene_ids)}

        # --------------------------------------------------
        # Align using STRICT gene_ids intersection only
        # --------------------------------------------------
        common = sorted(set(raw_gene_ids) & set(xv_gene_ids) & set(sv_gene_ids))

        if len(common) == 0:
            print("[WARN] no overlapping gene_ids")
            continue

        idx_raw = [raw_map[g] for g in common]
        idx_xv  = [xv_map[g]  for g in common]
        idx_sv  = [sv_map[g]  for g in common]

        raw_mat = raw_expr[:, idx_raw]
        xv_mat  = xv_expr[:, idx_xv]
        sv_mat  = sv_expr[:, idx_sv]

        # Build aligned AnnData for HVG selection
        ad_aln = sc.AnnData(
            X=raw_mat,
            var={"gene_ids": common},
        )
        ad_aln.var_names = common

        # --------------------------------------------------
        # Evaluate under multiple HVG sizes
        # --------------------------------------------------
        for hvg_n in HVG_LIST:
            if hvg_n > len(common):
                print(f"HVG={hvg_n}: skip (only {len(common)} genes)")
                continue

            hvg_idx = select_hvg_after_alignment(ad_aln, hvg_n)

            # ------ XVerse Eval ------
            xv_vals = compute_gene_wise_pearson(raw_mat, xv_mat, hvg_idx)
            xv_r = float(np.median(xv_vals)) if len(xv_vals) > 0 else np.nan

            # ------ SCVI Eval ------
            sv_vals = compute_gene_wise_pearson(raw_mat, sv_mat, hvg_idx)
            sv_r = float(np.median(sv_vals)) if len(sv_vals) > 0 else np.nan

            print(
                f"subset={subset} | HVG={hvg_n} | "
                f"XVerse={xv_r:.4f} | SCVI={sv_r:.4f}"
            )

            records.append(
                {
                    "sample": sample,
                    "subset": subset,
                    "hvg": hvg_n,
                    "model": "xVERSE",
                    "median_r": xv_r,
                    "values": xv_vals,
                }
            )
            records.append(
                {
                    "sample": sample,
                    "subset": subset,
                    "hvg": hvg_n,
                    "model": "SCVI",
                    "median_r": sv_r,
                    "values": sv_vals,
                }
            )

    plot_sample_comparisons(records, PLOT_SUBSET, PLOT_FILE)

    # Calculate average improvement
    improvements = []
    for rec_xv in records:
        if rec_xv["model"] == "xVERSE":
            # Find corresponding SCVI record
            rec_sv = next(
                (r for r in records if r["sample"] == rec_xv["sample"] 
                 and r["subset"] == rec_xv["subset"] 
                 and r["hvg"] == rec_xv["hvg"] 
                 and r["model"] == "SCVI"),
                None
            )
            if rec_sv and not np.isnan(rec_xv["median_r"]) and not np.isnan(rec_sv["median_r"]):
                imp = (rec_xv["median_r"] - rec_sv["median_r"]) / rec_sv["median_r"] * 100
                improvements.append(imp)
    
    if improvements:
        avg_imp = np.mean(improvements)
        print(f"\n[RESULT] Average improvement of xVERSE over scVI: +{avg_imp:.1f}%")
    else:
        print("\n[RESULT] No valid comparisons found for improvement calculation.")


if __name__ == "__main__":
    main()
