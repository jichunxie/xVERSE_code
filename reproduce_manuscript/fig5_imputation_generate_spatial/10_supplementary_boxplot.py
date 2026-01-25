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

# ================= CONFIGURATION =================
BASE_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results"
SPAGE_FILE = os.path.join(BASE_DIR, "spaGE", "spaGE_lung_samples_gene_corr.csv")
GIMVI_FILE = os.path.join(BASE_DIR, "gimvi", "gimvi_lung_samples_gene_corr.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = ["Lung5_Rep1", "Lung6", "Lung9_Rep1", "Lung13"]

FT_DIR = os.path.join(BASE_DIR, "ft")
ZS_DIR = os.path.join(BASE_DIR, "zero_shot")

# SC References
SC_REFS = [
    "nsclc_Wu_Zhou_2021_P11",
    "nsclc_Lambrechts_Thienpont_2018_6653_8",
    "nsclc_Chen_Zhang_2020_NSCLC-1",
    "nsclc_Chen_Zhang_2020_NSCLC-2",
    "nsclc_Chen_Zhang_2020_NSCLC-3"
]

# Short Labels for X-axis
SC_SHORT = {
    "nsclc_Wu_Zhou_2021_P11": "Ref1",
    "nsclc_Lambrechts_Thienpont_2018_6653_8": "Ref2",
    "nsclc_Chen_Zhang_2020_NSCLC-1": "Ref3",
    "nsclc_Chen_Zhang_2020_NSCLC-2": "Ref4",
    "nsclc_Chen_Zhang_2020_NSCLC-3": "Ref5"
}

# Descriptive names for legend (Matching 08 script)
SC_LEGEND_LABELS = {
    "nsclc_Wu_Zhou_2021_P11": "Wu_Zhou_2021_P11",
    "nsclc_Lambrechts_Thienpont_2018_6653_8": "Lambrechts_2018",
    "nsclc_Chen_Zhang_2020_NSCLC-1": "Chen_Zhang_2020_1",
    "nsclc_Chen_Zhang_2020_NSCLC-2": "Chen_Zhang_2020_2",
    "nsclc_Chen_Zhang_2020_NSCLC-3": "Chen_Zhang_2020_3"
}

# Colors
COLOR_XV_ZS = "#E41A1C"
COLOR_XV_FT = "#984EA3"
COLOR_GIMVI = "#4DAF4A"
COLOR_SPAGE = "#377EB8"

def load_xverse_data(sample, mode="ft"):
    subdir = FT_DIR if mode == "ft" else ZS_DIR
    path = os.path.join(subdir, f"{sample}_gene_correlations.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "pearson_corr" not in df.columns:
        return None
    return df["pearson_corr"].values

def load_comparison_data(file_path):
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

# Colors for SC References (Set2)
PALETTE = sns.color_palette("Set2", n_colors=len(SC_REFS))
SC_COLORS = {ref: color for ref, color in zip(SC_REFS, PALETTE)}

def plot_supplementary_boxplots():
    print("Loading data...")
    spage_full_df = load_comparison_data(SPAGE_FILE)
    gimvi_full_df = load_comparison_data(GIMVI_FILE)
    
    # 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()
    
    for i, sample in enumerate(SAMPLES):
        print(f"\nProcessing {sample}...")
        ax = axes[i]
        
        plot_data = []
        labels = []
        colors = []
        hatches = [] # Added hatches list
        
        # 1. xVERSE Zero-Shot
        xv_zs = load_xverse_data(sample, "zero_shot")
        if xv_zs is not None:
            plot_data.append(xv_zs)
            labels.append("XV-ZS")
            colors.append(COLOR_XV_ZS)
            hatches.append(None) # Solid
            print(f"  [Mean] XV-ZS: {np.mean(xv_zs):.4f}")
            
        # 2. xVERSE Fine-Tuned
        xv_ft = load_xverse_data(sample, "ft")
        if xv_ft is not None:
            plot_data.append(xv_ft)
            labels.append("XV-FT")
            colors.append(COLOR_XV_FT)
            hatches.append(None) # Solid
            print(f"  [Mean] XV-FT: {np.mean(xv_ft):.4f}")
            
        # 3. gimVI (5 refs)
        if gimvi_full_df is not None:
            for ref in SC_REFS:
                sub = gimvi_full_df[(gimvi_full_df["sample"] == sample) & (gimvi_full_df["sc"] == ref)]
                labels.append(f"G-{SC_SHORT[ref]}")
                colors.append(SC_COLORS[ref]) # Color by Ref
                hatches.append(None) # Solid for gimVI
                
                if not sub.empty:
                    data = sub["pearson_r"].dropna().values
                    if len(data) > 0:
                        plot_data.append(data)
                        print(f"  [Mean] G-{SC_SHORT[ref]}: {np.mean(data):.4f}")
                    else:
                        print(f"  [WARN] GimVI {ref} has no valid data. Placeholder.")
                        plot_data.append([])
                else:
                    print(f"  [WARN] GimVI {ref} has no rows. Placeholder.")
                    plot_data.append([])
        
        # 4. SpaGE (5 refs)
        if spage_full_df is not None:
            for ref in SC_REFS:
                sub = spage_full_df[(spage_full_df["sample"] == sample) & (spage_full_df["sc"] == ref)]
                labels.append(f"S-{SC_SHORT[ref]}")
                colors.append(SC_COLORS[ref]) # Color by Ref
                hatches.append('///') # Hatching for SpaGE
                
                if not sub.empty:
                    data = sub["pearson_r"].dropna().values
                    if len(data) > 0:
                        plot_data.append(data)
                        print(f"  [Mean] S-{SC_SHORT[ref]}: {np.mean(data):.4f}")
                    else:
                        print(f"  [WARN] SpaGE {ref} has no valid data. Placeholder.")
                        plot_data.append([])
                else:
                    print(f"  [WARN] SpaGE {ref} has no rows. Placeholder.")
                    plot_data.append([])
        
        # Plot Boxplot
        bplot = ax.boxplot(plot_data, patch_artist=True, labels=labels, showfliers=False)
        
        # Color and Hatch boxes
        for patch, color, hatch in zip(bplot['boxes'], colors, hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            if hatch:
                patch.set_hatch(hatch)
            
        for median in bplot['medians']:
            median.set_color('black')
            
        ax.set_title(sample, fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Rotate x labels
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
        
        if i % 2 == 0:
            ax.set_ylabel("Gene Correlation (Pearson)", fontsize=14)
            
    # Add Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_XV_ZS, label='xVERSE (Zero-Shot)'),
        Patch(facecolor=COLOR_XV_FT, label='xVERSE (Fine-Tuned)')
    ]
    # Add Ref colors
    for ref in SC_REFS:
        label_text = SC_LEGEND_LABELS.get(ref, SC_SHORT[ref])
        legend_elements.append(Patch(facecolor=SC_COLORS[ref], label=label_text))
    
    # Add Method distinction (Hatch vs Solid)
    legend_elements.append(Patch(facecolor='white', edgecolor='black', label='gimVI (Solid)'))
    legend_elements.append(Patch(facecolor='white', edgecolor='black', hatch='///', label='SpaGE (Hatched)'))
        
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "supplementary_boxplot_sensitivity.jpg")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_supplementary_boxplots()
