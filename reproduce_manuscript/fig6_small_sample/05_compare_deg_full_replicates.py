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

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================
DATA_ROOT = "/hpc/group/xielab/xj58/xVerse_results/fig6"
ORIGINAL_PATH = f"{DATA_ROOT}/GSE164378_RAW/rna_5P.h5ad"
SUBSAMPLED_PATH = f"{DATA_ROOT}/sampled_1pct_per_ct/5P_sampled_1pct_combined.h5ad"
GENERATED_PATH = f"{DATA_ROOT}/pretrain_inference/5P_sampled_1pct_combined_estimated.h5ad"
PLOT_DIR = f"{DATA_ROOT}/plots"

CELLTYPE_KEY = "celltype.l1"
NUM_REPLICATES = 100

# ===============================================

def intersect_genes(adata_list):
    """
    Find common genes across all datasets using 'gene_ids' and subset them.
    """
    print(f"\n[Intersect] Finding common genes across {len(adata_list)} datasets...")
    
    # 1. Identify common gene_ids
    common_ids = None
    
    for i, adata in enumerate(adata_list):
        if "gene_ids" not in adata.var.columns:
            raise KeyError(f"Dataset {i} missing 'gene_ids' in var columns.")
            
        current_ids = set(adata.var["gene_ids"].astype(str))
        if common_ids is None:
            common_ids = current_ids
        else:
            common_ids = common_ids.intersection(current_ids)
            
    common_ids = sorted(list(common_ids))
    print(f"  Common genes: {len(common_ids)}")
    
    # 2. Subset all datasets
    aligned_list = []
    for adata in adata_list:
        # Set var_names to gene_ids to allow subsetting by string ID
        adata_sub = adata.copy()
        adata_sub.var_names = adata_sub.var["gene_ids"].astype(str)
        adata_sub.var_names_make_unique()
        
        # Subset
        aligned_list.append(adata_sub[:, common_ids].copy())
        
    return aligned_list


def filter_and_merge_cells(adata):
    """
    Filter 'other' cell types and merge T cell subtypes.
    """
    if CELLTYPE_KEY not in adata.obs:
        return adata

    # 1. Filter 'other'
    adata = adata[adata.obs[CELLTYPE_KEY] != "other"].copy()
    
    # 2. Merge T cells
    labels = adata.obs[CELLTYPE_KEY].astype(str).values
    t_subtypes = ["CD4 T", "CD8 T", "other T"]
    
    for t in t_subtypes:
        labels[labels == t] = "T"
        
    adata.obs[CELLTYPE_KEY] = labels
    adata.obs[CELLTYPE_KEY] = adata.obs[CELLTYPE_KEY].astype("category")
    
    return adata


def normalize_and_log(adata):
    """
    Normalize to 10k and log1p.
    """
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def run_deg_analysis(adata, groupby):
    """
    Run rank_genes_groups on all genes.
    """
    print(f"  Running DEG analysis (groupby='{groupby}', n_cells={adata.n_obs})...")
    if len(adata.obs[groupby].unique()) < 2:
        print("    Skipping: < 2 groups found.")
        return False

    try:
        # Get all genes
        sc.tl.rank_genes_groups(
            adata, 
            groupby=groupby, 
            method="wilcoxon", 
            use_raw=False,
            n_genes=adata.n_vars  # Important: Get all genes for thresholding
        )
        return True
    except Exception as e:
        print(f"    Error in rank_genes_groups: {e}")
        return False


def precompute_rank_genes_groups_dfs(adata, groupby):
    """
    Precompute rank_genes_groups DFs for all groups to speed up thresholding.
    """
    dfs = {}
    groups = adata.obs[groupby].unique()
    for group in groups:
        try:
            dfs[group] = sc.get.rank_genes_groups_df(adata, group=group)
        except Exception:
            dfs[group] = pd.DataFrame()
    return dfs

def extract_degs_from_dfs(dfs_dict, pval_thresh):
    """
    Extract DEGs from precomputed DFs at a specific p-value threshold.
    """
    deg_dict = {}
    for group, df in dfs_dict.items():
        if df.empty:
            deg_dict[group] = set()
            continue
        # Filter by p-value AND LogFC (LogFC kept constant)
        mask = (df["pvals_adj"] <= pval_thresh)
        degs = set(df[mask]["names"].values)
        deg_dict[group] = degs
    return deg_dict

def sample_uniform_precision(df, n_points=10):
    """
    Select points from df such that they are uniformly distributed along the Precision axis.
    """
    if df.empty: return df
    
    # Define range based on data
    p_min, p_max = df["Precision"].min(), df["Precision"].max()
    if p_min == p_max: return df
    
    # Create target grid
    targets = np.linspace(p_min, p_max, n_points)
    
    selected_indices = []

    # Sort by Precision
    df_sorted = df.sort_values("Precision")
    prec_values = df_sorted["Precision"].values
    
    for t in targets:
        idx = (np.abs(prec_values - t)).argmin()
        selected_indices.append(df_sorted.index[idx])

    selected_indices = list(set(selected_indices))
    return df.loc[selected_indices].sort_values("Precision")

def calculate_metrics(true_degs_dict, pred_degs_dict):
    """
    Calculate Precision and Recall.
    """
    metrics = {}
    for group, true_genes in true_degs_dict.items():
        if len(true_genes) == 0:
            continue
            
        pred_genes = pred_degs_dict.get(group, set())
        intersection = len(true_genes.intersection(pred_genes))
        
        recall = intersection / len(true_genes) if len(true_genes) > 0 else 0.0
        precision = intersection / len(pred_genes) if len(pred_genes) > 0 else 0.0
        
        metrics[group] = {"Recall": recall, "Precision": precision}
        
    return metrics

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Load Datasets
    print("Loading datasets...")
    if not os.path.exists(ORIGINAL_PATH):
        raise FileNotFoundError(f"Original not found: {ORIGINAL_PATH}")
    if not os.path.exists(SUBSAMPLED_PATH):
        raise FileNotFoundError(f"Subsampled not found: {SUBSAMPLED_PATH}")
    if not os.path.exists(GENERATED_PATH):
        raise FileNotFoundError(f"Generated not found: {GENERATED_PATH}")
        
    adata_orig = sc.read_h5ad(ORIGINAL_PATH)
    adata_sub = sc.read_h5ad(SUBSAMPLED_PATH)
    adata_gen_all = sc.read_h5ad(GENERATED_PATH)
    
    # 2. Intersect Genes
    adata_orig, adata_sub, adata_gen_all = intersect_genes([adata_orig, adata_sub, adata_gen_all])
    
    # 3. Filter & Preprocess
    print("\nPreprocessing (Filter -> Normalize -> Log)...")
    adata_orig = normalize_and_log(filter_and_merge_cells(adata_orig))
    adata_sub = normalize_and_log(filter_and_merge_cells(adata_sub))
    adata_gen_all = normalize_and_log(filter_and_merge_cells(adata_gen_all))
    
    # 3.5 Plot Cell Type Pie Chart (Subsampled)
    print("Plotting Cell Type Distribution Pie Chart...")
    ct_counts = adata_sub.obs[CELLTYPE_KEY].value_counts()
    
    plt.figure(figsize=(10, 8))
    total = ct_counts.sum()
    labels = [f"{ct}: {count} ({count/total:.1%})" for ct, count in zip(ct_counts.index, ct_counts.values)]
    
    # Use a nice color palette
    colors = sns.color_palette('Set3')
    
    plt.pie(
        ct_counts.values, 
        labels=labels,
        # autopct='%1.1f%%', # Removed text on pie
        startangle=140,
        colors=colors,
        textprops={'fontsize': 18} # Larger outside font
    )
    plt.title(f"Cell Type Distribution (Subsampled 1%)\nTotal Cells: {total}", fontsize=20)
    
    pie_path = os.path.join(PLOT_DIR, "celltype_distribution_pie.png")
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"  Saved pie chart: {pie_path}")
    plt.close()
    
    # 4. SKIP HVG Selection
    print("\nSkipping HVG selection. Using all common genes.")
    
    # 5. Prepare Full Replicates
    adata_gen_100x = adata_gen_all.copy()
    
    print(f"\n[Baseline] Creating {NUM_REPLICATES}x replicate dataset...")
    adatas_to_concat = [adata_sub.copy() for _ in range(NUM_REPLICATES)]
    adata_base_100x = sc.concat(adatas_to_concat, join="outer", label="batch", keys=range(NUM_REPLICATES))
    if CELLTYPE_KEY not in adata_base_100x.obs:
         adata_base_100x.obs[CELLTYPE_KEY] = np.tile(adata_sub.obs[CELLTYPE_KEY].values, NUM_REPLICATES)
         adata_base_100x.obs[CELLTYPE_KEY] = adata_base_100x.obs[CELLTYPE_KEY].astype("category")
    
    # 6. Run DEG Analysis (Once per dataset)
    print("\n[Ground Truth] Running DEG on Original Data...")
    if not run_deg_analysis(adata_orig, CELLTYPE_KEY): return
    
    print(f"\n[Generated] Running DEG on xVerse Data ({NUM_REPLICATES}x)...")
    if not run_deg_analysis(adata_gen_100x, CELLTYPE_KEY): return
    
    print(f"\n[Baseline] Running DEG on Baseline Data ({NUM_REPLICATES}x)...")
    if not run_deg_analysis(adata_base_100x, CELLTYPE_KEY): return
    
    # Precompute DFs for fast extraction
    print("Precomputing DEG DataFrames...")
    dfs_orig = precompute_rank_genes_groups_dfs(adata_orig, CELLTYPE_KEY)
    dfs_gen = precompute_rank_genes_groups_dfs(adata_gen_100x, CELLTYPE_KEY)
    dfs_base = precompute_rank_genes_groups_dfs(adata_base_100x, CELLTYPE_KEY)
    
    # 7. Calculate Metrics across Thresholds
    # Define thresholds: Dense logspace
    thresholds = np.logspace(-8, np.log10(0.05), 5000)
    
    # Ground Truth is FIXED at standard threshold (0.05)
    gt_degs_fixed = extract_degs_from_dfs(dfs_orig, pval_thresh=0.05)
    
    print("\n[Ground Truth] Number of DEGs per cell type (p < 0.05):")
    for ct, degs in gt_degs_fixed.items():
        print(f"  {ct}: {len(degs)}")
    
    results = []
    print(f"\nCalculating metrics across {len(thresholds)} thresholds...")
    
    for p_thresh in thresholds:
        # Extract for xVerse
        gen_degs = extract_degs_from_dfs(dfs_gen, pval_thresh=p_thresh)
        gen_metrics = calculate_metrics(gt_degs_fixed, gen_degs)
        
        for group, m in gen_metrics.items():
            results.append({
                "Cell Type": group,
                "Method": "xVerse Augmented",
                "Threshold": p_thresh,
                "Recall": m["Recall"],
                "Precision": m["Precision"],
                "FDR": 1.0 - m["Precision"]
            })
            
        # Extract for Baseline
        base_degs = extract_degs_from_dfs(dfs_base, pval_thresh=p_thresh)
        base_metrics = calculate_metrics(gt_degs_fixed, base_degs)
        
        for group, m in base_metrics.items():
            results.append({
                "Cell Type": group,
                "Method": "Baseline (Simple Copy)",
                "Threshold": p_thresh,
                "Recall": m["Recall"],
                "Precision": m["Precision"],
                "FDR": 1.0 - m["Precision"]
            })
            
    df_results = pd.DataFrame(results)
    
    # 8. Plotting (Separate Plot per Cell Type)
    print("\nPlotting Precision vs Recall Curves per Cell Type...")
    
    cell_types = df_results["Cell Type"].unique()
    
    for ct in cell_types:
        df_ct = df_results[df_results["Cell Type"] == ct].copy()
        # Sort by Precision for plotting
        df_ct = df_ct.sort_values("Precision")
        
        # Calculate dynamic xlim (intersection of Precision ranges)
        # We use the full dense data to find the intersection range
        prec_gen = df_ct[df_ct["Method"] == "xVerse Augmented"]["Precision"].values
        prec_base = df_ct[df_ct["Method"] == "Baseline (Simple Copy)"]["Precision"].values
        
        if len(prec_gen) > 0 and len(prec_base) > 0:
            x_min = max(prec_gen.min(), prec_base.min())
            x_max = min(prec_gen.max(), prec_base.max())
            # Ensure min < max
            if x_min >= x_max:
                x_min, x_max = 0, 1.0 # Fallback
        else:
            x_min, x_max = 0, 1.0
            
        plt.figure(figsize=(8, 8))
        
        # Plot Regression Lines (Scatter + Fit + CI)
        methods = df_ct["Method"].unique()
        colors = {"xVerse Augmented": "tab:blue", "Baseline (Simple Copy)": "tab:orange"}
        
        for method in methods:
            data = df_ct[df_ct["Method"] == method]
            # Filter data to only include points within visible x-range
            data_visible = data[(data["Precision"] >= x_min) & (data["Precision"] <= x_max)]
            
            # Uniformly sample points in Precision for regression fit
            data_sampled = sample_uniform_precision(data_visible, n_points=50)
            
            if not data_sampled.empty:
                sns.regplot(
                    data=data_sampled, 
                    x="Precision", 
                    y="Recall", 
                    label=method,
                    color=colors.get(method, "black"),
                    scatter_kws={'s': 30, 'alpha': 0.6},
                    line_kws={'linewidth': 2},
                    ci=95
                )
        
        plt.ylabel("Recall", fontsize=24)
        plt.xlabel("Precision", fontsize=24)
        plt.xlim(x_min, x_max)
        
        # Calculate dynamic ylim based on visible x-range
        df_visible = df_ct[(df_ct["Precision"] >= x_min) & (df_ct["Precision"] <= x_max)]
        if not df_visible.empty:
            y_min = df_visible["Recall"].min()
            y_max = df_visible["Recall"].max()
            # Add 5% padding
            y_range = y_max - y_min
            if y_range == 0: y_range = 0.1
            plt.ylim(max(0, y_min - 0.05 * y_range), min(1.05, y_max + 0.05 * y_range))
        else:
            plt.ylim(0, 1.05)
            
        plt.grid(True, alpha=0.3)
        # plt.legend(loc='lower right') # Legend removed
        plt.tight_layout()
        
        # Clean filename
        ct_clean = str(ct).replace(" ", "_").replace("/", "_")
        save_path = os.path.join(PLOT_DIR, f"deg_precision_recall_{ct_clean}_{NUM_REPLICATES}x.png")
        plt.savefig(save_path, dpi=300)
        print(f"  Saved plot for {ct}: {save_path}")
        plt.close()
        
    # 8.5 Plot Combined 2x3 Figure (Custom Layout)
    print("\nPlotting Combined 2x3 Precision vs Recall Figure...")
    
    # Identify Cell Types
    nk_type = [ct for ct in cell_types if "NK" in str(ct)]
    dc_type = [ct for ct in cell_types if "DC" in str(ct) or "Dendritic" in str(ct)]
    other_types = [ct for ct in cell_types if ct not in nk_type and ct not in dc_type]
    
    # Flatten lists
    nk_type = nk_type[0] if nk_type else None
    dc_type = dc_type[0] if dc_type else None
    
    layout_cts = other_types + ([nk_type] if nk_type else []) + ([dc_type] if dc_type else [])
    
    # Create 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    # Map cell types to axes
    ax_map = {}
    for i, ct in enumerate(other_types):
        if i < 3: ax_map[i] = ct
    if nk_type: ax_map[3] = nk_type
    if dc_type: ax_map[4] = dc_type
    
    for i, ax in enumerate(axes_flat):
        if i in ax_map:
            ct = ax_map[i]
            df_ct = df_results[df_results["Cell Type"] == ct].copy()
            df_ct = df_ct.sort_values("Precision")
            
            # Dynamic xlim (Intersection)
            prec_gen = df_ct[df_ct["Method"] == "xVerse Augmented"]["Precision"].values
            prec_base = df_ct[df_ct["Method"] == "Baseline (Simple Copy)"]["Precision"].values
            
            if len(prec_gen) > 0 and len(prec_base) > 0:
                x_min = max(prec_gen.min(), prec_base.min())
                x_max = min(prec_gen.max(), prec_base.max())
                if x_min >= x_max: x_min, x_max = 0, 1.0
            else:
                x_min, x_max = 0, 1.0
                
            # Plot
            methods = df_ct["Method"].unique()
            colors = {"xVerse Augmented": "tab:blue", "Baseline (Simple Copy)": "tab:orange"}
            
            for method in methods:
                data = df_ct[df_ct["Method"] == method]
                data_visible = data[(data["Precision"] >= x_min) & (data["Precision"] <= x_max)]
                
                # Uniformly sample points in Precision for regression fit
                data_sampled = sample_uniform_precision(data_visible, n_points=50)
                
                if not data_sampled.empty:
                    sns.regplot(
                        data=data_sampled, 
                        x="Precision", 
                        y="Recall", 
                        label=method,
                        color=colors.get(method, "black"),
                        scatter_kws={'s': 30, 'alpha': 0.6},
                        line_kws={'linewidth': 2},
                        ci=95,
                        ax=ax
                    )
            
            ax.set_title(ct, fontsize=24, fontweight='bold')
            ax.set_xlabel("Precision", fontsize=18)
            if i % 3 == 0:
                ax.set_ylabel("Recall", fontsize=18)
            else:
                ax.set_ylabel("")
                
            ax.set_xlim(x_min, x_max)
            
            # Dynamic ylim
            df_visible = df_ct[(df_ct["Precision"] >= x_min) & (df_ct["Precision"] <= x_max)]
            if not df_visible.empty:
                y_min = df_visible["Recall"].min()
                y_max = df_visible["Recall"].max()
                y_range = y_max - y_min
                if y_range == 0: y_range = 0.1
                ax.set_ylim(max(0, y_min - 0.05 * y_range), min(1.05, y_max + 0.05 * y_range))
            else:
                ax.set_ylim(0, 1.05)
                
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
            
    plt.tight_layout()
    save_combined = os.path.join(PLOT_DIR, f"deg_precision_recall_combined_2x3_{NUM_REPLICATES}x.png")
    plt.savefig(save_combined, dpi=300)
    print(f"  Saved Combined 2x3 plot: {save_combined}")
    plt.close()
        
    # 9. Plot Average Curve
    print("\nPlotting Average Precision vs Recall Curve...")
    df_avg = df_results.groupby(["Method", "Threshold"])[["Recall", "Precision"]].mean().reset_index()
    df_avg = df_avg.sort_values("Precision")
    
    # Calculate dynamic xlim for average plot
    prec_gen_avg = df_avg[df_avg["Method"] == "xVerse Augmented"]["Precision"].values
    prec_base_avg = df_avg[df_avg["Method"] == "Baseline (Simple Copy)"]["Precision"].values
    
    if len(prec_gen_avg) > 0 and len(prec_base_avg) > 0:
        x_min_avg = min(prec_gen_avg.min(), prec_base_avg.min())
        x_max_avg = max(prec_gen_avg.max(), prec_base_avg.max())
        if x_min_avg >= x_max_avg:
             x_min_avg, x_max_avg = 0, 1.0
    else:
        x_min_avg, x_max_avg = 0, 1.0
    
    plt.figure(figsize=(8, 8))
    
    # Plot Regression Lines for Average
    methods = df_avg["Method"].unique()
    colors = {"xVerse Augmented": "tab:blue", "Baseline (Simple Copy)": "tab:orange"}
    
    for method in methods:
        data = df_avg[df_avg["Method"] == method]
        # Filter data to only include points within visible x-range
        data_visible = data[(data["Precision"] >= x_min_avg) & (data["Precision"] <= x_max_avg)]
        
        # Uniformly sample points in Precision for regression fit
        data_sampled = sample_uniform_precision(data_visible, n_points=50)
        
        if not data_sampled.empty:
            sns.regplot(
                data=data_sampled, 
                x="Precision", 
                y="Recall", 
                label=method,
                color=colors.get(method, "black"),
                scatter_kws={'s': 30, 'alpha': 0.6},
                line_kws={'linewidth': 2},
                ci=95
            )
    
    plt.title(f"Precision vs Recall: Average across Cell Types ({NUM_REPLICATES}x Replicates)")
    plt.ylabel("Average Recall (Sensitivity)", fontsize=24)
    plt.xlabel("Average Precision", fontsize=24)
    plt.xlim(x_min_avg, x_max_avg)
    
    # Calculate dynamic ylim for average plot
    df_visible_avg = df_avg[(df_avg["Precision"] >= x_min_avg) & (df_avg["Precision"] <= x_max_avg)]
    if not df_visible_avg.empty:
        y_min_avg = df_visible_avg["Recall"].min()
        y_max_avg = df_visible_avg["Recall"].max()
        # Add 5% padding
        y_range_avg = y_max_avg - y_min_avg
        if y_range_avg == 0: y_range_avg = 0.1
        plt.ylim(max(0, y_min_avg - 0.05 * y_range_avg), min(1.05, y_max_avg + 0.05 * y_range_avg))
    else:
        plt.ylim(0, 1.05)
        
    plt.grid(True, alpha=0.3)
    # plt.legend(loc='lower right') # Legend removed
    plt.tight_layout()
    
    save_avg = os.path.join(PLOT_DIR, f"deg_precision_recall_AVERAGE_{NUM_REPLICATES}x.png")
    plt.savefig(save_avg, dpi=300)
    print(f"  Saved Average plot: {save_avg}")
    plt.close()
    
    # 10. Save Separate Legend
    print("\nSaving Separate Legend...")
    plt.figure(figsize=(4, 2))
    # Create dummy plot to get handles
    # Use lineplot here as it's easier to generate a clean legend handle
    # But to match colors, we use the same palette
    for method in methods:
        plt.plot([], [], label=method, color=colors.get(method, "black"), linewidth=2)
        
    plt.axis('off')
    legend = plt.legend(loc='center', frameon=False, ncol=1, fontsize=14)
    
    # Save just the legend
    def export_legend(legend, filename="legend.png"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=300, bbox_inches=bbox)

    legend_path = os.path.join(PLOT_DIR, f"deg_legend_{NUM_REPLICATES}x.png")
    export_legend(legend, legend_path)
    print(f"  Saved Legend: {legend_path}")
    plt.close()
    
    # Save CSV
    csv_path = os.path.join(PLOT_DIR, f"deg_pr_curve_data_{NUM_REPLICATES}x.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved data to: {csv_path}")


if __name__ == "__main__":
    main()
