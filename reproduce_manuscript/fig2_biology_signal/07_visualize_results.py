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

#!/usr/bin/env python3
"""
Generate publication-quality figures for xVerse evaluation.
Visualizes:
1. Performance Superiority (Grouped Bar Charts of ASW)
2. Robustness (Line Plots of ASW across gene sets)
3. Efficiency (Scatter plot of Accuracy vs Time)
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
# Path to evaluation results (same as in 06_evaluate_embeddings.py)
if os.path.exists("/hpc/group/xielab/xj58"):
    base_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2"
    eval_dir = os.path.join(base_dir, "evaluation")
else:
    # Local fallback
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(base_dir, "results", "evaluation")

output_dir = os.path.join(eval_dir, "paper_figures")
os.makedirs(output_dir, exist_ok=True)

# Models to include and their display colors
MODELS = ["xVERSE", "Harmony", "scGPT", "Nicheformer", "Geneformer"]
PALETTE = sns.color_palette("husl", len(MODELS))
COLOR_MAP = dict(zip(MODELS, PALETTE))

# ================= HELPER FUNCTIONS =================

def load_metrics_data():
    """Load all metrics CSV files into a single DataFrame."""
    dfs = []
    tissues = ["liver", "brain"]
    gene_sets = ["all", "5k", "xenium"]
    
    for tissue in tissues:
        for gene_set in gene_sets:
            filename = f"{tissue}_{gene_set}_metrics.csv"
            filepath = os.path.join(eval_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df["Tissue"] = tissue.capitalize()
                    
                    # Map gene sets to display names
                    if gene_set == "all":
                        display_name = "Whole Transcriptome"
                        order = 3
                    elif gene_set == "5k":
                        display_name = "Xenium Prime (5K)"
                        order = 2
                    else: # xenium
                        display_name = "Xenium V1 (~300)"
                        order = 1
                        
                    df["Gene Set"] = display_name
                    df["Gene Set Order"] = order
                    
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
            else:
                print(f"Warning: Missing file {filepath}")
    
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)

def load_timing_data():
    """Load the combined timing data."""
    filepath = os.path.join(eval_dir, "all_models_inference_timing.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

# ================= PLOTTING FUNCTIONS =================

def set_global_style():
    """Set global style and font size."""
    sns.set_style("white") 
    sns.set_context("poster", font_scale=1.5) # Increased scale
    plt.rcParams.update({
        'font.size': 30,
        'axes.labelsize': 20,
        'axes.titlesize': 30,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 30
    })

def plot_efficiency_tradeoff_final(metrics_df, timing_df, tissue):
    """
    Generate only the Efficiency Tradeoff (Scatter) plot.
    """
    print(f"Generating Efficiency Tradeoff Plot for {tissue}...")
    
    # --- PREPARE DATA ---
    
    # 1. Metrics Data
    df_perf = metrics_df[metrics_df["Tissue"] == tissue].copy()
    if df_perf.empty:
        print(f"  No metrics data for {tissue}, skipping.")
        return
        
    order = ["Xenium V1 (~300)", "Xenium Prime (5K)", "Whole Transcriptome"]
    # df_perf['Gene Set'] = pd.Categorical(df_perf['Gene Set'], categories=order, ordered=True)

    # 2. Timing Data
    if timing_df is not None:
        df_time = timing_df[timing_df["tissue"].str.lower() == tissue.lower()].copy()
    else:
        df_time = pd.DataFrame()
        
    if not df_time.empty:
        # Map gene sets
        def map_gene_set(gs):
            if gs.lower() == "all": return "Whole Transcriptome"
            if gs.lower() == "5k": return "Xenium Prime (5K)"
            if gs.lower() == "xenium": return "Xenium V1 (~300)"
            return gs.capitalize()
        
        df_time['Gene Set Display'] = df_time['gene_set'].apply(map_gene_set)
        df_time = df_time[df_time['model'] != 'Harmony']
        # df_time['Gene Set Display'] = pd.Categorical(df_time['Gene Set Display'], categories=order, ordered=True)
        
        # Prepare merged data for Tradeoff
        timing_agg = df_time.groupby(['model', 'gene_set'])['time_seconds'].mean().reset_index()
        timing_agg['Gene Set'] = timing_agg['gene_set'].apply(map_gene_set)
        timing_agg = timing_agg.rename(columns={'model': 'Model'})
        
        metrics_agg = df_perf.groupby(['Model', 'Gene Set'])['ASW_label'].mean().reset_index()
        metrics_agg = metrics_agg[metrics_agg['Model'] != 'Harmony']
        
        merged_tradeoff = pd.merge(metrics_agg, timing_agg, on=['Model', 'Gene Set'])
    else:
        merged_tradeoff = pd.DataFrame()

    # --- PLOTTING ---
    
    plt.figure(figsize=(14, 14)) # Square size, height=14
    
    if not merged_tradeoff.empty:
        # Plot with legend='full' to ensure we can extract handles
        ax = sns.scatterplot(
            data=merged_tradeoff,
            x="time_seconds",
            y="ASW_label",
            hue="Model",
            style="Gene Set",
            s=800, 
            palette=COLOR_MAP,
            hue_order=[m for m in MODELS if m != 'Harmony'],
            alpha=0.8,
            legend='full'
        )
        plt.xscale('log')
        plt.xlabel("Log Inference Time", fontsize=36, fontweight='bold') 
        plt.ylabel("ASW Label Score", fontsize=36, fontweight='bold') 
        
        # Extract handles and labels for separate legend
        handles, labels = ax.get_legend_handles_labels()
        
        # Remove legend from the main plot
        if ax.legend_:
            ax.legend_.remove()
        
        # Save Main Plot
        save_path = os.path.join(output_dir, f"Supplementary_Fig2_efficiency_tradeoff_{tissue}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved Plot: {save_path}")
        plt.close()
        
        # --- Save Legend Separately ---
        fig_leg = plt.figure(figsize=(10, 4))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis('off')
        
        # Create legend with 2 columns
        ax_leg.legend(handles, labels, loc='center', ncol=2, fontsize=27, markerscale=1.5, framealpha=1.0)
        
        leg_save_path = os.path.join(output_dir, f"Supplementary_Fig2_efficiency_tradeoff_legend_{tissue}.png")
        fig_leg.savefig(leg_save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved Legend: {leg_save_path}")
        plt.close(fig_leg)
        
    else:
        plt.text(0.5, 0.5, "No Data", ha='center', va='center')
        save_path = os.path.join(output_dir, f"Supplementary_Fig2_efficiency_tradeoff_{tissue}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ================= MAIN =================

def plot_inference_time(timing_df):
    """
    Generate Figure 2c: Zero-shot inference time comparison (2x1 grid for Liver and Brain).
    """
    print("Generating Figure 2c (Inference Time)...")
    
    if timing_df is None or timing_df.empty:
        print("  No timing data found, skipping.")
        return

    # Filter for Liver and Brain
    tissues = ["liver", "brain"]
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("poster", font_scale=1.2)
    
    # Create figure with 2 subplots (1 row, 2 cols)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Define order: Xenium (smallest) -> 5K -> All (largest)
    gene_set_order = ["xenium", "5k", "all"]
    gene_set_labels = ["Xenium V1 (~300)", "Xenium Prime (5K)", "Whole Transcriptome"]
    
    # Map gene sets to display names for plotting
    gene_set_map = dict(zip(gene_set_order, gene_set_labels))

    for idx, tissue in enumerate(tissues):
        ax = axes[idx]
        
        # Filter data for current tissue
        df_tissue = timing_df[timing_df['tissue'].str.lower() == tissue].copy()
        
        if df_tissue.empty:
            ax.text(0.5, 0.5, f"No Data for {tissue}", ha='center', va='center')
            continue

        # Calculate average time per donor for each model and gene set
        df_avg = df_tissue.groupby(['model', 'gene_set'])['time_seconds'].mean().reset_index()
        
        # Ensure correct order
        df_avg['gene_set_cat'] = pd.Categorical(df_avg['gene_set'], categories=gene_set_order, ordered=True)
        df_avg = df_avg.sort_values('gene_set_cat')
        
        # Map gene sets to display labels
        df_avg['Gene Set'] = df_avg['gene_set'].map(gene_set_map)
        
        # Plot line chart
        sns.lineplot(
            data=df_avg,
            x='Gene Set',
            y='time_seconds',
            hue='model',
            style='model',
            markers=True,
            dashes=False,
            markersize=14,
            linewidth=3.5,
            palette=COLOR_MAP, # Use consistent palette
            ax=ax,
            hue_order=[m for m in MODELS if m != 'Harmony'], # Exclude Harmony if not in timing
            style_order=[m for m in MODELS if m != 'Harmony']
        )
        
        # Customize Axis
        ax.set_title(f"{tissue.capitalize()}", fontsize=28, fontweight='bold', pad=20)
        ax.set_xlabel("", fontsize=24)
        if idx == 0:
            ax.set_ylabel("Inference Time (seconds)", fontsize=24, fontweight='bold')
        else:
            ax.set_ylabel("")
            
        ax.tick_params(axis='x', labelsize=20, rotation=0)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Log scale for y-axis if range is large (optional, but often good for time)
        ax.set_yscale('log') 
        
        # Add value labels
        for i in range(df_avg.shape[0]):
            row = df_avg.iloc[i]
            # Find x-coordinate (0, 1, 2)
            x_idx = gene_set_order.index(row['gene_set'])
            
            ax.text(
                x=x_idx, 
                y=row['time_seconds'] + (df_avg['time_seconds'].max() * 0.03),
                s=f"{row['time_seconds']:.1f}s", 
                ha='center', 
                fontsize=16,
                fontweight='bold',
                color='black'
            )

        # Handle Legend
        if idx == 0:
            ax.get_legend().remove() # Remove individual legend
        else:
            # Move legend to outside
            ax.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, frameon=False)
            
    plt.tight_layout()
    
    # Save
    fig_path = os.path.join(output_dir, "fig2c_inference_time_2x1.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved Figure 2c: {fig_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    metrics_df = load_metrics_data()
    timing_df = load_timing_data()
    
    if metrics_df is not None:
        # Set global style
        set_global_style()
        
        # Get unique tissues
        tissues = metrics_df["Tissue"].unique()
        print(f"Found tissues: {tissues}")
        
        for tissue in tissues:
            print(f"\n--- Processing {tissue} ---")
            # Generate Supplementary Figure 2 (Efficiency Tradeoff)
            # Rename output file inside the function by modifying the function call or the function itself
            # Since I can't easily change the function signature in this replace block without replacing the whole function,
            # I will modify the save path inside the function in a separate edit if needed, 
            # OR I can just rename the file after generation if I don't want to replace the whole function.
            # But wait, I am replacing the main block, so I can't change the function definition here.
            # I will assume I need to replace the function definition in a separate call or replace the whole file content if it's small enough.
            # The file is ~220 lines. I can replace the function `plot_efficiency_tradeoff_final` and `main` in one go if I select the range correctly.
            pass

    # Actually, I should split this into two edits: one to add the new function and update main, and one to update the existing function to rename the file.
    # Let's do the new function and main first.
    
    if timing_df is not None:
        plot_inference_time(timing_df)

    if metrics_df is not None:
        # Set global style
        set_global_style()
        
        # Get unique tissues
        tissues = metrics_df["Tissue"].unique()
        print(f"Found tissues: {tissues}")
        
        for tissue in tissues:
            print(f"\n--- Processing {tissue} ---")
            plot_efficiency_tradeoff_final(metrics_df, timing_df, tissue)
            
        print("\nAll visualizations generated successfully!")
    else:
        print("No metrics data found. Please run 06_evaluate_embeddings.py first.")
