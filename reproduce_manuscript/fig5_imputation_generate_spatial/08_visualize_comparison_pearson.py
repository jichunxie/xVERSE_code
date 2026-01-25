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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# ================= CONFIGURATION =================
BASE_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results"
SPAGE_FILE = os.path.join(BASE_DIR, "spaGE", "spaGE_lung_samples_gene_corr.csv")
GIMVI_FILE = os.path.join(BASE_DIR, "gimvi", "gimvi_lung_samples_gene_corr.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = ["Lung5_Rep1", "Lung6", "Lung9_Rep1", "Lung13"]

FT_DIR = os.path.join(BASE_DIR, "ft")
ZS_DIR = os.path.join(BASE_DIR, "zero_shot")

# SC References for Coloring
SC_REFS = [
    "nsclc_Wu_Zhou_2021_P11",
    "nsclc_Lambrechts_Thienpont_2018_6653_8",
    "nsclc_Chen_Zhang_2020_NSCLC-1",
    "nsclc_Chen_Zhang_2020_NSCLC-2",
    "nsclc_Chen_Zhang_2020_NSCLC-3"
]

# Short names for legend if needed, or just use full names
SC_LABELS = {
    "nsclc_Wu_Zhou_2021_P11": "Wu_Zhou_2021_P11",
    "nsclc_Lambrechts_Thienpont_2018_6653_8": "Lambrechts_2018",
    "nsclc_Chen_Zhang_2020_NSCLC-1": "Chen_Zhang_2020_1",
    "nsclc_Chen_Zhang_2020_NSCLC-2": "Chen_Zhang_2020_2",
    "nsclc_Chen_Zhang_2020_NSCLC-3": "Chen_Zhang_2020_3"
}

# Colors for SC References
PALETTE = sns.color_palette("Set2", n_colors=len(SC_REFS))
SC_COLORS = {ref: color for ref, color in zip(SC_REFS, PALETTE)}

# Markers for Methods
METHOD_MARKERS = {
    "SpaGE": "^",    # Triangle
    "gimVI": "D"     # Diamond
}

# Colors for Methods (for average lines)
METHOD_COLORS = {
    "SpaGE": "#377EB8",    # Blue
    "gimVI": "#4DAF4A"     # Green
}

# Colors for xVerse Lines
XVERSE_COLORS = {
    "Zero-Shot": "#E41A1C",   # Red
    "Fine-Tuned": "#984EA3"   # Purple
}

# ... (Helper functions unchanged)

def load_xverse_metric(sample, mode="ft", metric="mean"):
    """Load xVerse results and calculate metric (mean or min) from CSV."""
    subdir = FT_DIR if mode == "ft" else ZS_DIR
    path = os.path.join(subdir, f"{sample}_gene_correlations.csv")
    
    if not os.path.exists(path):
        print(f"Warning: xVerse {mode} file not found for {sample} at {path}")
        return None
        
    df = pd.read_csv(path)
    if "pearson_corr" not in df.columns:
        print(f"Warning: 'pearson_corr' column missing in {path}")
        return None
        
    if metric == "mean":
        return df["pearson_corr"].mean()
    elif metric == "min":
        return df["pearson_corr"].min()
    else:
        raise ValueError(f"Unknown metric: {metric}")

def load_comparison_metric(file_path, method_name, metric="mean"):
    """Load SpaGE or gimVI results and calculate metric per (sample, sc_ref)."""
    if not os.path.exists(file_path):
        print(f"Warning: {method_name} file not found at {file_path}")
        return None
    df = pd.read_csv(file_path)
    
    # Group by sample and sc (reference), then calculate metric
    if metric == "mean":
        metric_df = df.groupby(['sample', 'sc'])['pearson_r'].mean().reset_index()
    elif metric == "min":
        metric_df = df.groupby(['sample', 'sc'])['pearson_r'].min().reset_index()
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    return metric_df

# ================= MAIN PLOTTING =================

def plot_comparison(metric="mean", output_suffix=""):
    print(f"Generating comparison plot for metric: {metric}...")
    
    # Load comparison datasets
    spage_df = load_comparison_metric(SPAGE_FILE, "SpaGE", metric=metric)
    gimvi_df = load_comparison_metric(GIMVI_FILE, "gimVI", metric=metric)
    
    # Setup figure: 1 row (samples) x 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    if len(SAMPLES) == 1: axes = [axes]
    plt.subplots_adjust(wspace=0.3)
    
    for i, sample in enumerate(SAMPLES):
        ax = axes[i]
        print(f"Processing {sample}...")
        
        # 1. Plot xVerse Lines
        val_zs = load_xverse_metric(sample, mode="zero_shot", metric=metric)
        val_ft = load_xverse_metric(sample, mode="ft", metric=metric)
        
        if val_zs is not None:
            ax.axhline(y=val_zs, color=XVERSE_COLORS["Zero-Shot"], linestyle='--', linewidth=4, label="xVERSE (Zero-Shot)")
        
        if val_ft is not None:
            ax.axhline(y=val_ft, color=XVERSE_COLORS["Fine-Tuned"], linestyle='--', linewidth=4, label="xVERSE (Fine-Tuned)")
            
        # 2. Plot SpaGE Points and Average Line
        current_x = 0
        if spage_df is not None:
            sample_spage = spage_df[spage_df['sample'] == sample]
            if not sample_spage.empty:
                spage_vals = []
                for ref in SC_REFS:
                    row = sample_spage[sample_spage['sc'] == ref]
                    if not row.empty:
                        val = row['pearson_r'].values[0]
                        spage_vals.append(val)
                        ax.scatter(current_x, val, color=SC_COLORS[ref], marker=METHOD_MARKERS["SpaGE"], s=250, zorder=5)
                        current_x += 1
                    else:
                        pass
                
                # Plot SpaGE Average Line
                if spage_vals:
                    spage_avg = np.mean(spage_vals)
                    ax.axhline(y=spage_avg, color=METHOD_COLORS["SpaGE"], linestyle='-.', linewidth=3, alpha=0.8, label="SpaGE (Avg)")

        # Add some spacing between methods
        current_x += 1 
        
        # 3. Plot gimVI Points and Average Line
        if gimvi_df is not None:
            sample_gimvi = gimvi_df[gimvi_df['sample'] == sample]
            if not sample_gimvi.empty:
                gimvi_vals = []
                for ref in SC_REFS:
                    row = sample_gimvi[sample_gimvi['sc'] == ref]
                    if not row.empty:
                        val = row['pearson_r'].values[0]
                        gimvi_vals.append(val)
                        ax.scatter(current_x, val, color=SC_COLORS[ref], marker=METHOD_MARKERS["gimVI"], s=250, zorder=5)
                        current_x += 1
                    else:
                        pass
                
                # Plot gimVI Average Line
                if gimvi_vals:
                    gimvi_avg = np.mean(gimvi_vals)
                    ax.axhline(y=gimvi_avg, color=METHOD_COLORS["gimVI"], linestyle='-.', linewidth=3, alpha=0.8, label="gimVI (Avg)")
        
        y_label = "Mean Pearson Correlation" if metric == "mean" else "Min Pearson Corr"
        
        if i == 0:
            ax.set_ylabel(y_label, fontsize=20)
            ax.tick_params(axis='y', labelsize=16)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)
            
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Color '0' label red if metric is min
        if metric == "min":
            # Get current ticks and labels
            ticks = ax.get_yticks()
            tick_labels = ax.get_yticklabels()
            
            for tick, label in zip(ticks, tick_labels):
                # Check if tick value is close to 0
                if abs(tick) < 1e-6:
                    label.set_color('red')
                    label.set_fontweight('bold')

        # Remove x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_title(f"{sample}", fontsize=18)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"imputation_comparison_{output_suffix}.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def plot_legend():
    print("Generating legend...")
    fig_leg = plt.figure(figsize=(14, 4)) # Increased width more
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    
    # 1. xVerse Lines
    lines_handles = [
        Line2D([0], [0], color=XVERSE_COLORS["Zero-Shot"], lw=2, linestyle='--', label='xVERSE (Zero-Shot)'),
        Line2D([0], [0], color=XVERSE_COLORS["Fine-Tuned"], lw=2, linestyle='--', label='xVERSE (Fine-Tuned)')
    ]
    
    # 2. Methods (Shapes & Lines)
    method_handles = [
        Line2D([0], [0], marker=METHOD_MARKERS["SpaGE"], color='w', label='SpaGE', markerfacecolor='k', markersize=10),
        Line2D([0], [0], color=METHOD_COLORS["SpaGE"], lw=2, linestyle='-.', label='SpaGE (Avg)'),
        Line2D([0], [0], marker=METHOD_MARKERS["gimVI"], color='w', label='gimVI', markerfacecolor='k', markersize=10),
        Line2D([0], [0], color=METHOD_COLORS["gimVI"], lw=2, linestyle='-.', label='gimVI (Avg)')
    ]
    
    # 3. SC References (Colors)
    sc_handles = [
        Patch(facecolor=SC_COLORS[ref], label=SC_LABELS[ref]) for ref in SC_REFS
    ]
    
    # Column 1: xVerse
    leg1 = ax_leg.legend(handles=lines_handles, title="xVERSE Models", loc='center left', bbox_to_anchor=(0.05, 0.5), frameon=False, fontsize=12, title_fontsize=16)
    ax_leg.add_artist(leg1)
    
    # Column 2: Methods
    leg2 = ax_leg.legend(handles=method_handles, title="Comparison Methods", loc='center', bbox_to_anchor=(0.5, 0.5), frameon=False, ncol=2, fontsize=12, title_fontsize=16) 
    ax_leg.add_artist(leg2)
    
    # Column 3: SC References
    leg3 = ax_leg.legend(handles=sc_handles, title="SC References", loc='center right', bbox_to_anchor=(0.95, 0.5), frameon=False, fontsize=12, title_fontsize=16)
    
    leg_path = os.path.join(OUTPUT_DIR, "imputation_comparison_legend_v3.jpg")
    fig_leg.savefig(leg_path, dpi=300, bbox_inches='tight')
    print(f"Saved legend to {leg_path}")

if __name__ == "__main__":
    # 1. Plot Mean (keep plotting functionality as it might be needed for the figure, but focus output on print)
    plot_comparison(metric="mean", output_suffix="points_lines_mean")
    
    # 2. Plot Min
    plot_comparison(metric="min", output_suffix="points_lines_min")
    
    # 3. Plot Legend (Shared)
    plot_legend()

    # 4. Print Specific Metrics requested
    print("\n" + "="*40)
    print("      METRIC SUMMARY      ")
    print("="*40)
    
    spage_df = load_comparison_metric(SPAGE_FILE, "SpaGE", metric="mean")
    gimvi_df = load_comparison_metric(GIMVI_FILE, "gimVI", metric="mean")

    all_improvements = []

    for sample in SAMPLES:
        print(f"\n>>> Sample: {sample}")
        
        # xVerse
        xv_zs = load_xverse_metric(sample, mode="zero_shot", metric="mean")
        xv_ft = load_xverse_metric(sample, mode="ft", metric="mean")
        
        # Handle cases where metric might be None
        xv_zs_str = f"{xv_zs:.4f}" if xv_zs is not None else "N/A"
        xv_ft_str = f"{xv_ft:.4f}" if xv_ft is not None else "N/A"
        
        print(f"  xVerse Zero-Shot: {xv_zs_str}")
        print(f"  xVerse Fine-Tuned: {xv_ft_str}")
        
        # SpaGE
        print("  SpaGE:")
        spage_scores = []
        if spage_df is not None:
             sample_data = spage_df[spage_df['sample'] == sample]
             if not sample_data.empty:
                 for _, row in sample_data.iterrows():
                     print(f"    - Ref {row['sc']}: {row['pearson_r']:.4f}")
                 spage_scores = sample_data['pearson_r'].tolist()
             else:
                 print("    (No data)")
        else:
             print("    (No data)")

        # gimVI
        print("  gimVI:")
        gimvi_scores = []
        if gimvi_df is not None:
             sample_data = gimvi_df[gimvi_df['sample'] == sample]
             if not sample_data.empty:
                 for _, row in sample_data.iterrows():
                     print(f"    - Ref {row['sc']}: {row['pearson_r']:.4f}")
                     gimvi_scores.append(row['pearson_r'])
             else:
                 print("    (No data)")
        else:
             print("    (No data)")
             
        # Calculate Percentage Improvement (Comparison vs xVerse FT)
        # Second best is the best of SpaGE/gimVI across all references
        all_comparison_scores = spage_scores + gimvi_scores
        
        if all_comparison_scores and xv_ft is not None:
            best_comparison = max(all_comparison_scores)
            worst_comparison = min(all_comparison_scores)
            avg_comparison = np.mean(all_comparison_scores)
            
            improvement = (xv_ft - best_comparison) / best_comparison * 100
            all_improvements.append(improvement)
            
            print("-" * 20)
            print(f"  Best Comparison (Second Best Model): {best_comparison:.4f}")
            print(f"  xVerse FT Improvement over Second Best: {improvement:.2f}%")
            print("-" * 20)

    if all_improvements:
        avg_imp = np.mean(all_improvements)
        print("\n" + "="*40)
        print(f"OVERALL AVERAGE IMPROVEMENT: {avg_imp:.2f}%")
        print("="*40)
