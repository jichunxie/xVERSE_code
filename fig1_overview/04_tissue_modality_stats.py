
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
CSV_PATH = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/pantissue_full_updated.csv"
OUTPUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig1"

def generate_latex_table(stats_df, output_path):
    """Generate a high-quality LaTeX table from the stats DataFrame (Cell Counts)."""
    # Reset index to make Tissue a column
    df = stats_df.reset_index()
    
    # Start LaTeX content
    tex = [
        r"\begin{longtable}{|l|r|r|r|}",
        r"\hline",
        r"\textbf{Tissue Name} & \textbf{Single-cell} & \textbf{Spatial} & \textbf{Total} \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot"
    ]
    
    # Add rows
    for _, row in df.iterrows():
        tissue = str(row['tissue_name']).replace("_", r"\_")
        # Handle potential missing columns if 0
        sc = int(row.get('single-cell', 0)) if 'single-cell' in row else 0
        sp = int(row.get('spatial', 0)) if 'spatial' in row else 0
        total = int(row['Total'])
        
        # Format numbers with commas
        sc_str = f"{sc:,}"
        sp_str = f"{sp:,}"
        total_str = f"{total:,}"
        
        tex.append(f"{tissue} & {sc_str} & {sp_str} & {total_str} \\\\")
        tex.append(r"\hline")
    
    tex.append(r"\end{longtable}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(tex))
    print(f"Saved LaTeX table to {output_path}")

def plot_single_pie_chart(df, modality, output_path):
    """Plot a single Pie Chart for a specific modality using total_num_cells."""
    
    # Filter by modality
    sub_df = df[df['modality_type'] == modality]
    
    if sub_df.empty:
        print(f"No data for {modality}")
        return
        
    # Aggregate by tissue, summing total_num_cells
    counts = sub_df.groupby('tissue_name')['total_num_cells'].sum().sort_values(ascending=False)
    
    # Top 10 + Others
    top_n = 10
    if len(counts) > top_n:
        top_counts = counts.iloc[:top_n]
        others_count = counts.iloc[top_n:].sum()
        top_counts['Others'] = others_count
    else:
        top_counts = counts
    
    # Prepare colors (merge tab20, tab20b, tab20c for more variety)
    cmap1 = plt.get_cmap('tab20')
    cmap2 = plt.get_cmap('tab20b')
    cmap3 = plt.get_cmap('tab20c')
    colors = cmap1.colors + cmap2.colors + cmap3.colors
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    wedges, texts, autotexts = ax.pie(top_counts, labels=top_counts.index, autopct='%1.1f%%', 
                                      startangle=140, colors=colors[:len(top_counts)], 
                                      textprops={'fontsize': 24}, pctdistance=0.85)
    
    # Style
    ax.set_title(f"{modality.capitalize()} Tissue Distribution (Cell Count)", fontsize=24)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Pie Chart to {output_path}")

def main():
    print(f"Loading data from: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows.")
        
        required_cols = ['tissue_name', 'modality_type', 'total_num_cells']
        
        # Check if columns exist
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Error: Missing required columns {missing}. Found: {df.columns.tolist()}")
            return
            
        # Clean data
        df['tissue_name'] = df['tissue_name'].astype(str).str.strip()
        df['modality_type'] = df['modality_type'].astype(str).str.strip()
        # Ensure numeric
        df['total_num_cells'] = pd.to_numeric(df['total_num_cells'], errors='coerce').fillna(0)
        
        # --- Statistics ---
        print("\n--- Generating Statistics (Weighted by Cell Count) ---")
        # Pivot table: Sum total_num_cells
        stats = df.groupby(['tissue_name', 'modality_type'])['total_num_cells'].sum().unstack(fill_value=0)
        
        # Ensure columns exist even if 0
        for m in ['single-cell', 'spatial']:
            if m not in stats.columns:
                stats[m] = 0
                
        # Sort by total count descending
        stats['Total'] = stats.sum(axis=1)
        stats = stats.sort_values('Total', ascending=False)
        
        # --- Print Totals ---
        total_sc = stats['single-cell'].sum()
        total_sp = stats['spatial'].sum()
        print(f"\n==========================================")
        print(f"Total Single-cell cells: {total_sc:,}")
        print(f"Total Spatial cells:     {total_sp:,}")
        print(f"Total All cells:         {total_sc + total_sp:,}")
        print(f"==========================================\n")
        
        print(stats.head())
        
        # Save Stats CSV
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stats_path = os.path.join(OUTPUT_DIR, "tissue_modality_distribution.csv")
        stats.to_csv(stats_path)
        print(f"\nSaved statistics to {stats_path}")
        
        # --- Generate LaTeX Table ---
        tex_path = os.path.join(OUTPUT_DIR, "tissue_modality_stats.tex")
        generate_latex_table(stats, tex_path)
        
        # --- Generate Pie Charts ---
        plot_single_pie_chart(df, 'single-cell', os.path.join(OUTPUT_DIR, "tissue_distribution_single_cell.png"))
        plot_single_pie_chart(df, 'spatial', os.path.join(OUTPUT_DIR, "tissue_distribution_spatial.png"))

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
