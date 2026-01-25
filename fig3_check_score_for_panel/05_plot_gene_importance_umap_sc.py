# -*- coding: utf-8 -*-
"""
Generate a 5x6 UMAP grid for Single Cell Gene Importance Evaluation (Figure 7).
Rows: 5 Samples (PM-A to PM-E)
"""

import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from main.utils_ft import (
    XVerseFineTuneModel,
    XVerseFineTuneDataset,
)
from main.utils_model import load_gene_ids, XVerseModel

# ==============================================================  
# Configurations (Matched to fig4/02_train_xverse_full.py)
# ==============================================================  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tissue_map = {"breast": 8}

# Paths
base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

# Data Dictionary (Ordered for Plotting)
file_sample_dict = {
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-A.h5ad": 0,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-B.h5ad": 1,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-C.h5ad": 2,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-D.h5ad": 3,
    "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-E.h5ad": 4,
}

# Model Checkpoint
ft_model_dir = "/hpc/group/xielab/xj58/xVerse_results/fig4/ft_model"
model_label = "full"
ckpt_path = os.path.join(ft_model_dir, f"best_model_{model_label}.pth")

# Output Directory
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig3/gene_weights"
os.makedirs(output_dir, exist_ok=True)

batch_size = 256
num_workers = 16

# Plot Settings
FIG_SIZE = (24, 20) # 6 cols * 4 inch, 5 rows * 4 inch
LABEL_KEY = "Major.subtype"


# ==============================================================  
# Utility  
# ==============================================================  

def build_model(base_ckpt_path, num_samples_finetune, hidden_dim=384):
    """Load pretrained XVerse and wrap with fine-tune head."""
    print("Initializing base model...")
    base_model = XVerseModel(num_samples=None, hidden_dim=hidden_dim).to(device)

    if os.path.exists(base_ckpt_path):
        ckpt = torch.load(base_ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        base_model.load_state_dict(state, strict=False)
    
    model = XVerseFineTuneModel(
        base_model=base_model,
        num_samples_finetune=num_samples_finetune,
    ).to(device)
    return model


def get_embeddings(model, loader):
    """Run inference and return z_bio embeddings."""
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
            sample_id, value, tissue_id = batch
            sample_id = sample_id.to(device)
            tissue_id = tissue_id.to(device)
            value = value.to(device)
            
            outputs = model(value=value, sample_id=sample_id, tissue_id=tissue_id)
            z_bio = outputs["z_bio"]
            embeddings_list.append(z_bio.cpu().numpy())
            
    return np.concatenate(embeddings_list, axis=0)


def save_legend(adata, key, output_path):
    """Save the legend as a separate image."""
    print(f"Saving legend to {output_path}...")
    
    # Create a dummy plot to extract legend
    fig_leg, ax_leg = plt.subplots(figsize=(4, 4))
    sc.pl.umap(adata, color=key, ax=ax_leg, show=False, legend_loc="on data", frameon=False)
    
    categories = adata.obs[key].cat.categories
    colors = adata.uns[f"{key}_colors"]
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat, 
                          markerfacecolor=col, markersize=10) 
               for cat, col in zip(categories, colors)]
    
    fig_clean, ax_clean = plt.subplots(figsize=(2, len(categories)*0.3))
    ax_clean.legend(handles=handles, loc='center', frameon=False, ncol=1)
    ax_clean.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_leg)
    plt.close(fig_clean)


# ==============================================================  
# Main  
# ==============================================================  

def main():
    print(f"Using device: {device}")

    # 1. Load Gene IDs & Model
    gene_ids = load_gene_ids(gene_ids_path)
    
    model = build_model(
        base_ckpt_path=base_model_ckpt,
        num_samples_finetune=len(file_sample_dict),
        hidden_dim=384,
    )

    if os.path.exists(ckpt_path):
        print(f"Loading fine-tuned weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    # 2. Setup Plot Grid
    # 5 Rows (Samples), 5 Cols (Full + 4 Bins)
    fig, axes = plt.subplots(5, 5, figsize=(25, 25)) 
    
    # Column Titles
    col_titles = ["Full Panel", "Top 25%", "25-50%", "50-75%", "Bottom 25%"]
    
    # Legend saved flag
    legend_saved = False
    
    # 3. Iterate Samples (Rows)
    valid_file_sample_dict = {p: i for p, i in file_sample_dict.items() if os.path.exists(p)}
    
    for row_idx, (path, sample_id) in enumerate(valid_file_sample_dict.items()):
        sample_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'='*40}")
        print(f"Processing Row {row_idx+1}: {sample_name}")
        print(f"{'='*40}")
        
        # Load Weights
        weight_file = os.path.join(output_dir, f"sc_{sample_name}_mean_gene_weights.csv")
        if not os.path.exists(weight_file):
            print(f"Weight file not found: {weight_file}. Skipping row.")
            continue
            
        df_weights = pd.read_csv(weight_file)
        df_weights = df_weights.sort_values(by="mean_weight", ascending=False)
        df_weights = df_weights[df_weights["mean_weight"] > 0]
        
        total_genes = len(df_weights)
        if total_genes < 4:
            print("Not enough genes. Skipping row.")
            continue
            
        # Load Data & Labels
        adata_orig = sc.read_h5ad(path)
        if LABEL_KEY not in adata_orig.obs.columns:
            print(f"Label {LABEL_KEY} not found. Skipping row.")
            continue
        labels = adata_orig.obs[LABEL_KEY].values
        
        # 4. Iterate Columns (Gene Sets)
        # Col 0: Full Panel
        # Col 1-4: Bins (4 bins)
        
        chunk_size = total_genes // 4
        
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            
            # Determine Gene Set
            if col_idx == 0:
                # Full Panel
                visible_genes = None # None means all genes in gene_ids
                desc = "Full"
            else:
                # Bins (0-indexed for calculation)
                bin_idx = col_idx - 1
                start_idx = bin_idx * chunk_size
                end_idx = (bin_idx + 1) * chunk_size if bin_idx < 3 else total_genes
                visible_genes = df_weights.iloc[start_idx:end_idx]["gene_id"].tolist()
                desc = f"Bin {bin_idx+1}"
            
            print(f"  Col {col_idx+1}: {desc}")
            
            # Create Dataset
            dataset = XVerseFineTuneDataset(
                {path: sample_id},
                gene_ids,
                tissue_map,
                visible_gene_ids=visible_genes,
                use_qc=False,
            )
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            
            # Get Embeddings
            z_bio = get_embeddings(model, loader)
            
            # Compute ASW
            # Compute ASW using scib
            if len(np.unique(labels)) > 1:
                # Create temp AnnData for scib
                adata_temp = sc.AnnData(X=z_bio)
                adata_temp.obs['label'] = labels
                adata_temp.obsm['X_emb'] = z_bio
                
                asw = scib.metrics.silhouette(
                    adata_temp, 
                    group_key='label', 
                    embed='X_emb'
                )
            else:
                asw = 0.0
            
            # Compute UMAP
            # Create temporary adata for plotting
            adata_plot = sc.AnnData(X=z_bio)
            adata_plot.obs[LABEL_KEY] = labels
            # Ensure categorical
            adata_plot.obs[LABEL_KEY] = adata_plot.obs[LABEL_KEY].astype("category")
            
            sc.pp.neighbors(adata_plot, use_rep='X')
            sc.tl.umap(adata_plot)
            
            # Save Legend (Once)
            if not legend_saved:
                legend_path = os.path.join(output_dir, "sc_gene_importance_umap_legend.png")
                save_legend(adata_plot, LABEL_KEY, legend_path)
                legend_saved = True
            
            # Plot
            sc.pl.umap(
                adata_plot, 
                color=LABEL_KEY, 
                ax=ax, 
                show=False, 
                legend_loc=None, # No legend on plot
                title="", 
                frameon=False,
                s=20 # Marker size increased
            )
            
            # Add ASW Text
            ax.text(
                0.5, -0.1, 
                f"ASW: {asw:.3f}", 
                transform=ax.transAxes, 
                ha='center', 
                fontsize=24, # Increased font size
                fontweight='bold'
            )
            
            # Clean up axis
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.axis('off') # Remove box frame
            
            # Add Row/Col Labels if needed
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=32, fontweight='bold', pad=20) # Increased font size
            
            if col_idx == 0:
                # Add Sample Name to the left
                ax.text(
                    -0.2, 0.5, 
                    sample_name, 
                    transform=ax.transAxes, 
                    va='center', 
                    ha='right', 
                    fontsize=32, # Increased font size
                    fontweight='bold', 
                    rotation=90
                )

    # 5. Save Grid
    plt.tight_layout()
    save_path = os.path.join(output_dir, "sc_gene_importance_umap_grid.png")
    print(f"Saving grid to {save_path}...")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Also PDF
    pdf_path = save_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print("Done.")


if __name__ == "__main__":
    main()
