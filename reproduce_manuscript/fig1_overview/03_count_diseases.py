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
from collections import defaultdict
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor"
OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig1"


def generate_latex_table(df, output_path):
    """Generate a high-quality LaTeX table from the disease stats DataFrame."""
    # Sort by Tissue then Count (descending)
    df_sorted = df.sort_values(['Tissue', 'Count'], ascending=[True, False])
    
    # Start LaTeX content
    tex = [
        r"\begin{longtable}{|l|l|r|}",
        r"\hline",
        r"\textbf{Tissue} & \textbf{Disease} & \textbf{Count} \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot"
    ]
    
    # Add rows
    for _, row in df_sorted.iterrows():
        tissue = str(row['Tissue']).replace("_", r"\_")
        disease = str(row['Disease']).replace("_", r"\_")
        count = int(row['Count'])
        
        tex.append(f"{tissue} & {disease} & {count} \\\\")
        tex.append(r"\hline")
    
    tex.append(r"\end{longtable}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(tex))
    print(f"Saved LaTeX table to {output_path}")

def plot_disease_distribution(stats_dict, output_dir):
    """
    Generate a stacked bar chart of disease distribution for the top 10 tissues and top 30 diseases.
    X-axis: Tissue (Top 10 by total count)
    Y-axis: Proportion (0-100%)
    Segments: Diseases (Top 30)
    """
    # Convert nested dict to list of dicts for DataFrame
    rows = []
    for tissue, d_counts in stats_dict.items():
        for disease, count in d_counts.items():
            rows.append({'Tissue': tissue, 'Disease': disease, 'Count': count})
    
    if not rows:
        print("No data to plot.")
        return

    df = pd.DataFrame(rows)
    
    # Calculate total counts per tissue to find Top 15
    tissue_totals = df.groupby('Tissue')['Count'].sum().sort_values(ascending=False)
    top_tissues = tissue_totals.head(15).index.tolist()
    
    # Filter for Top 15 tissues
    df_top = df[df['Tissue'].isin(top_tissues)].copy()
    
    # Calculate total counts per disease to find Top 30
    disease_totals = df_top.groupby('Disease')['Count'].sum().sort_values(ascending=False)
    top_diseases = disease_totals.head(30).index.tolist()
    
    # Filter for Top 30 diseases
    df_top = df_top[df_top['Disease'].isin(top_diseases)].copy()
    
    # Pivot for plotting: Index=Tissue, Columns=Disease, Values=Count
    pivot_df = df_top.groupby(['Tissue', 'Disease'])['Count'].sum().unstack(fill_value=0)
    
    # Sort Pivot by Total Count - REVERSE order so highest is at top in barh
    pivot_df = pivot_df.reindex(top_tissues[::-1])  # Reverse the list
    
    # Reorder columns (diseases) by total count descending for color mapping
    disease_order = pivot_df.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df[disease_order]
    
    # Normalize to obtain proportions (row-wise sum = 1)
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Create a custom color palette for diseases (30 colors needed)
    cmap1 = plt.get_cmap('tab20')
    cmap2 = plt.get_cmap('tab20b')
    cmap3 = plt.get_cmap('Set3')
    disease_colors = list(cmap1.colors) + list(cmap2.colors) + list(cmap3.colors[:10])
    
    # ===== MAIN PLOT (horizontal bar, without legend) =====
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bar chart - tissues on Y-axis, diseases stacked horizontally
    pivot_pct.plot(kind='barh', stacked=True, ax=ax, color=disease_colors[:len(pivot_pct.columns)], width=0.8, legend=False)
    
    # Styling - NO TITLE, font size 24
    ax.set_ylabel('Tissue', fontsize=24)
    ax.set_xlabel('Proportion (%)', fontsize=24)
    ax.set_xlim(0, 100)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    
    # Adjust subplot parameters to prevent text clipping
    plt.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.1)
    
    out_path = os.path.join(output_dir, "disease_distribution_stacked_bar.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Stacked Bar Chart to {out_path}")
    
    # ===== SEPARATE LEGEND (2 columns) =====
    fig_legend, ax_legend = plt.subplots(figsize=(10, 8))
    ax_legend.axis('off')
    
    # Create legend handles manually for diseases
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=disease_colors[i], label=disease) 
                     for i, disease in enumerate(pivot_pct.columns)]
    
    # Create legend with 2 columns
    ax_legend.legend(handles=legend_handles, loc='center', ncol=2, 
                    fontsize=18, title='Disease', title_fontsize=20, frameon=False)
    
    legend_path = os.path.join(output_dir, "disease_distribution_legend.png")
    plt.savefig(legend_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Legend to {legend_path}")

def main():
    # --- Configuration ---
    PANTISSUE_CSV = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/pantissue_full_updated.csv"
    
    # --- Output Path ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, "disease_stats_by_tissue.csv")
    
    df_rows = None

    # Check if CSV exists to skip scanning
    if os.path.exists(output_csv):
        print(f"Found existing stats file: {output_csv}")
        print("Loading directly from CSV...")
        try:
            df_rows = pd.read_csv(output_csv)
            print(f"Loaded {len(df_rows)} rows from existing CSV.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            df_rows = None

    if df_rows is None:
        # --- Load Pantissue CSV to get whitelist ---
        print(f"Loading pantissue CSV from: {PANTISSUE_CSV}")
        if not os.path.exists(PANTISSUE_CSV):
            print(f"Error: Pantissue CSV not found at {PANTISSUE_CSV}")
            return
            
        pantissue_df = pd.read_csv(PANTISSUE_CSV)
        print(f"Loaded {len(pantissue_df)} donor-dataset pairs from pantissue CSV")
        
        # Extract the obs_paths (these are the files we should process)
        if 'obs_paths' not in pantissue_df.columns:
            print("Error: 'obs_paths' column not found in pantissue CSV")
            return
        
        # Get all obs paths (handle semicolon-separated multiple paths per row)
        files_to_process = []
        for obs_paths_str in pantissue_df['obs_paths'].dropna():
            # Split by semicolon to get individual paths
            paths = str(obs_paths_str).split(';')
            for path in paths:
                path = path.strip()  # Remove any whitespace
                if path and os.path.exists(path):
                    files_to_process.append(path)
                elif path:  # Path is not empty but doesn't exist
                    print(f"Warning: File not found: {path}")
        
        print(f"Found {len(files_to_process)} valid obs files from pantissue CSV")
        
        # Dictionary to store counts: tissue -> disease -> count
        stats = defaultdict(lambda: defaultdict(int))
        
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                if os.path.getsize(file_path) == 0:
                    continue

                with np.load(file_path, allow_pickle=True) as data:
                    if 'disease' not in data:
                        continue
                    
                    disease_arr = data['disease']
                    if disease_arr.size == 0:
                        continue
                        
                    # Use only tissue_general
                    if 'tissue_general' not in data:
                        continue
                        
                    tissue_arr = data['tissue_general']
                    if tissue_arr.size != disease_arr.size:
                        continue
                    
                    # Group by tissue_general and disease
                    df = pd.DataFrame({
                        'tissue': tissue_arr.astype(str),
                        'disease': disease_arr.astype(str)
                    })
                    counts = df.groupby(['tissue', 'disease']).size()
                    for (t, d), count in counts.items():
                        stats[t][d] += count

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # --- Output Report ---
        print("\n" + "="*60)
        print("DISEASE OCCURRENCE STATISTICS BY TISSUE")
        print("="*60)
        
        sorted_tissues = sorted(stats.keys())
        total_cells_all = 0
        
        for tissue in sorted_tissues:
            disease_counts = stats[tissue]
            sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
            total_tissue_cells = sum(c for d, c in sorted_diseases)
            total_cells_all += total_tissue_cells

        print("\n" + "="*60)
        print(f"Grand Total Cells with Disease Info: {total_cells_all:,}")
        print("="*60)

        # Prepare DataFrame
        rows = []
        for tissue, d_counts in stats.items():
            for disease, count in d_counts.items():
                rows.append({'Tissue': tissue, 'Disease': disease, 'Count': count})
        
        df_rows = pd.DataFrame(rows)
        print(f"\nSaving summary to {output_csv}...")
        df_rows.to_csv(output_csv, index=False)
        print("Done.")
    
    # --- Generate LaTeX Table ---
    tex_path = os.path.join(OUTPUT_DIR, "disease_stats.tex")
    generate_latex_table(df_rows, tex_path)
    
    # --- Visualization ---
    # Reconstruct stats dict from df_rows for plotting
    stats_dict = defaultdict(dict)
    if df_rows is not None:
        for _, row in df_rows.iterrows():
            stats_dict[row['Tissue']][row['Disease']] = row['Count']
            
    plot_disease_distribution(stats_dict, OUTPUT_DIR)

if __name__ == "__main__":
    main()
