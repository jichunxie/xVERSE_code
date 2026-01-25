# -*- coding: utf-8 -*-
"""
Plot single-cell gene importance evaluation results (Figure 7).
For each single-cell dataset, plot ASW vs Gene Score Bins (2, 3, 4 bins).
Include a benchmark line for Full Panel (All Genes).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================
# Configuration
# ==============================================================

# Input Files
DATA_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig3/gene_weights"
SC_FILE = os.path.join(DATA_DIR, "sc_gene_importance_asw.csv")

# Output
OUTPUT_DIR = DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot Settings
sns.set_context("paper", font_scale=2.0) # Large font as requested previously
sns.set_style("ticks")
FIG_SIZE = (8, 6) # Per subplot size

# ==============================================================
# Main
# ==============================================================

def get_bin_labels(n_bins):
    if n_bins == 2:
        return ["Top 50%", "Bottom 50%"]
    elif n_bins == 3:
        return ["Top 33%", "Mid 33%", "Bottom 33%"]
    elif n_bins == 4:
        return ["Top 25%", "25-50%", "50-75%", "Bottom 25%"]
    else:
        return [f"Bin {i+1}" for i in range(n_bins)]

def plot_for_bins(df, n_bins):
    print(f"\nGenerating plots for {n_bins} bins...")
    
    # Filter for relevant bin counts
    df_trend = df[df["NumBins"] == n_bins].copy()
    df_bench = df[df["NumBins"] == 1].copy()
    
    if df_trend.empty:
        print(f"No data found for {n_bins} bins. Skipping.")
        return

    # Get unique samples
    all_samples = df["Sample"].unique()
    
    # Filter for valid samples (must have trend data)
    valid_samples = []
    for sample in all_samples:
        sample_trend = df_trend[df_trend["Sample"] == sample]
        if not sample_trend.empty:
            valid_samples.append(sample)
            
    if not valid_samples:
        print("No valid samples found.")
        return

    n_samples = len(valid_samples)
    print(f"Found {n_samples} valid samples.")
    
    # Create Vertical Subplots
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 6 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    # Colors
    color_trend = "#2E86C1" # Professional Blue
    color_bench = "#E74C3C" # Professional Red
    
    bin_labels = get_bin_labels(n_bins)
    subset_pct = int(100 / n_bins)
    
    for i, sample in enumerate(valid_samples):
        print(f"Processing {sample}...")
        ax = axes[i]
        
        # Get Sample Data
        sample_trend = df_trend[df_trend["Sample"] == sample].sort_values("Bin")
        sample_bench = df_bench[df_bench["Sample"] == sample]
        
        # 1. Plot Benchmark Line
        if not sample_bench.empty:
            bench_val = sample_bench["ASW"].values[0]
            ax.axhline(y=bench_val, color=color_bench, linestyle='--', linewidth=6, label="Full Panel (All Genes)", alpha=0.8)
            
            # 3. Add Vertical Drop Lines & Annotations
            for idx, row in sample_trend.iterrows():
                bin_x = row["Bin"]
                val_y = row["ASW"]
                
                # Draw vertical line
                ax.vlines(x=bin_x, ymin=min(val_y, bench_val), ymax=max(val_y, bench_val), colors='gray', linestyles=':', linewidth=2)
                
                # Calculate drop
                if bench_val != 0:
                    drop_pct = (bench_val - val_y) / bench_val * 100
                else:
                    drop_pct = 0.0
                
                # Add text
                # If val_y is higher than bench_val (positive improvement), show +X%
                # If val_y is lower (drop), show -X%
                
                sign = "-" if drop_pct > 0 else "+"
                annot_text = f"{sign}{abs(drop_pct):.1f}%"
                
                ax.text(
                    bin_x + 0.05, (val_y + bench_val) / 2, 
                    annot_text, 
                    color='black', 
                    fontsize=24, 
                    ha='left', 
                    va='center',
                    fontweight='bold'
                )
        
        # 2. Plot Trend Line
        x = sample_trend["Bin"]
        y = sample_trend["ASW"]
        
        ax.plot(x, y, marker='o', color=color_trend, linewidth=4, markersize=12, label=f"Gene Subsets ({subset_pct}%)", markeredgecolor='white', markeredgewidth=2)
        
        # Formatting
        # Removed Title and X-Label as requested
        # ax.set_ylabel("ASW Score", fontsize=40, labelpad=20) # Removed as requested
        
        # Set X-ticks
        ax.set_xticks(range(1, n_bins + 1))
        ax.set_xticklabels(bin_labels, rotation=0, fontsize=32)
        ax.tick_params(axis='y', labelsize=32)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')
        
        # Restore Frame (Do not despine)
        # sns.despine(trim=True)

    plt.tight_layout()
    
    # Save Combined Figure
    save_name = f"sc_gene_importance_asw_combined_{n_bins}bins.jpg"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot to {save_path}")
    
    # Save Separate Legend (Only need to do this once really, but doing it per bin count is fine to capture correct %)
    print("Saving separate legend...")
    fig_leg, ax_leg = plt.subplots(figsize=(8, 2))
    
    # Create handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_bench, lw=6, linestyle='--', label='Full Panel (All Genes)'),
        Line2D([0], [0], color=color_trend, lw=4, marker='o', markersize=12, markeredgecolor='white', markeredgewidth=2, label=f'Gene Subsets ({subset_pct}%)')
    ]
    
    ax_leg.legend(handles=legend_elements, loc='center', ncol=2, frameon=False, fontsize=24)
    ax_leg.axis('off')
    
    leg_path = os.path.join(OUTPUT_DIR, f"sc_gene_importance_asw_legend_{n_bins}bins.jpg")
    plt.savefig(leg_path, dpi=300, bbox_inches="tight")
    print(f"Saved legend to {leg_path}")
    
    plt.close('all')


def main():
    print("Generating Single-Cell Gene Importance ASW Plots...")
    
    if not os.path.exists(SC_FILE):
        print(f"Error: File not found {SC_FILE}")
        return
        
    df = pd.read_csv(SC_FILE)
    
    # Loop over requested bin counts
    for n_bins in [2, 3, 4]:
        try:
            plot_for_bins(df, n_bins)
        except Exception as e:
            print(f"Failed to plot for {n_bins} bins: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
