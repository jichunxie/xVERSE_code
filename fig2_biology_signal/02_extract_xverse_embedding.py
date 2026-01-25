#!/usr/bin/env python3
"""
Extract xVerse embeddings for liver and brain donor datasets.
For each h5ad file:
1. Load pretrained model
2. Extract bio embeddings
3. Save to obsm["xVerse"]
4. Create UMAP visualization: left (xVerse embedding), right (cell_type)
"""

import os
import time
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from main.utils_ft import XVerseFineTuneDataset
from main.utils_model import XVerseModel, load_gene_ids

# ================= CONFIGURATION =================
gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"
base_model_ckpt = "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_1118/best_model.pth"

liver_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/liver"
brain_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/brain"
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/xverse_embeddings"

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tissue_map = {"liver": 31, "brain": 7}  # Tissue IDs for liver and brain

# Timing data collection
timing_records = []

# ===============================================

print("Loading gene ids...")
gene_ids = load_gene_ids(gene_ids_path)


def load_pretrained_model():
    """Load the pretrained base model."""
    print("Loading pretrained model...")
    model = XVerseModel(num_samples=None, hidden_dim=384).to(device)
    
    # Check if checkpoint exists
    if not os.path.exists(base_model_ckpt):
        raise FileNotFoundError(f"Pretrained checkpoint not found at {base_model_ckpt}")
    
    # Load checkpoint
    ckpt = torch.load(base_model_ckpt, map_location=device)
    
    # Handle both formats: direct state_dict or wrapped in "model_state_dict" key
    state = ckpt.get("model_state_dict", ckpt)
    
    # Remove 'module.' prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    
    # Load state dict
    model.load_state_dict(state, strict=False)
    model.eval()
    
    print("Pretrained model loaded successfully.")
    return model


def extract_bio_embedding(model, h5ad_path, tissue_name):
    """
    Extract bio embeddings from the pretrained model for a given h5ad file.
    
    Args:
        model: Pretrained XVerseModel
        h5ad_path: Path to h5ad file
        tissue_name: Name of tissue ('liver' or 'brain')
    
    Returns:
        numpy array of embeddings (n_cells, embedding_dim)
    """
    print(f"  Extracting embeddings from {h5ad_path.name}...")
    
    # Create dataset
    dataset = XVerseFineTuneDataset(
        {str(h5ad_path): None},
        gene_ids,
        tissue_map,
        use_qc=False,
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    # Extract embeddings
    embed_list = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Processing batches", leave=False):
            sample_ids, values, tissue_ids = batch
            values = values.to(device)
            tissue_ids = tissue_ids.to(device)
            
            with torch.amp.autocast(device.type if device.type == 'cuda' else 'cpu', 
                                   enabled=torch.cuda.is_available()):
                # Model returns: mu_bio, attn_weight, bio_embedding, celltype_logits
                _, _, bio_embedding, _ = model(value=values, tissue_id=tissue_ids)
            
            embed_list.append(bio_embedding.cpu().numpy())
    
    embedding_arr = np.concatenate(embed_list, axis=0)
    print(f"  Extracted embeddings shape: {embedding_arr.shape}")
    
    return embedding_arr


def extract_and_save_embedding(h5ad_path, tissue_name, gene_set, model):
    """
    Extract embeddings for one h5ad file and save to obsm["xVerse"] in-place.
    
    Args:
        h5ad_path: Path to h5ad file
        tissue_name: Name of tissue ('liver' or 'brain')
        gene_set: Gene set identifier ('all', '5k', or 'xenium')
        model: Pretrained model
    """
    print(f"  Processing: {h5ad_path.name}")
    
    # Extract donor ID from filename
    donor_id = h5ad_path.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
    
    # Load h5ad
    adata = sc.read_h5ad(h5ad_path)
    print(f"    Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Time the embedding extraction
    start_time = time.time()
    embeddings = extract_bio_embedding(model, h5ad_path, tissue_name)
    inference_time = time.time() - start_time
    
    # Record timing
    timing_records.append({
        'model': 'xVerse',
        'tissue': tissue_name,
        'gene_set': gene_set,
        'donor': donor_id,
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'time_seconds': inference_time
    })
    
    print(f"    ⏱️  Inference time: {inference_time:.2f}s")
    
    # Save to obsm
    adata.obsm["xVerse"] = embeddings
    
    # Save updated h5ad with embeddings (in-place modification)
    adata.write(h5ad_path)
    print(f"    Updated with xVerse embeddings")
    
    return h5ad_path


def merge_and_visualize_tissue(h5ad_files, tissue_name, gene_set="all"):
    """
    Merge all donor files for a tissue and create combined UMAP visualization.
    
    Args:
        h5ad_files: List of paths to h5ad files (with embeddings already saved)
        tissue_name: Name of tissue ('liver' or 'brain')
        gene_set: Gene set identifier ('all', '5k', or 'xenium')
    """
    print(f"\n{'='*60}")
    print(f"Merging and visualizing {tissue_name.upper()} tissue - {gene_set.upper()} gene set")
    print(f"{'='*60}")
    
    # Load all h5ad files
    adata_list = []
    for h5ad_path in h5ad_files:
        donor_id = h5ad_path.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
        print(f"  Loading {donor_id}: ", end="")
        
        adata = sc.read_h5ad(h5ad_path)
        
        # Ensure donor_id is in obs
        if 'donor_id' not in adata.obs.columns:
            # Try to find donor column
            donor_col = None
            for col in ['donor', 'donor_id_ontology']:
                if col in adata.obs.columns:
                    donor_col = col
                    break
            
            if donor_col:
                adata.obs['donor_id'] = adata.obs[donor_col]
            else:
                # Use filename as donor_id
                adata.obs['donor_id'] = donor_id
        
        print(f"{adata.n_obs} cells")
        adata_list.append(adata)
    
    # Merge all donors
    print(f"\n  Merging {len(adata_list)} donors...")
    adata_merged = sc.concat(adata_list, join="outer", index_unique=None)
    print(f"  Total: {adata_merged.n_obs} cells")
    
    # Compute UMAP on merged xVerse embeddings
    print("  Computing UMAP on xVerse embeddings...")
    sc.pp.neighbors(adata_merged, use_rep="xVerse", n_neighbors=15)
    sc.tl.umap(adata_merged)
    
    # Find cell_type column
    cell_type_col = None
    for col in ['cell_type', 'celltype', 'cell_type_ontology_term_id', 'celltype.l2']:
        if col in adata_merged.obs.columns:
            cell_type_col = col
            break
    
    if cell_type_col is None:
        print(f"  Warning: No cell_type column found. Available: {adata_merged.obs.columns.tolist()}")
        cell_type_col = adata_merged.obs.columns[0]
    
    # Create visualization
    print("  Creating UMAP visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Color by donor
    sc.pl.umap(
        adata_merged,
        color='donor_id',
        ax=axes[0],
        show=False,
        title=f"{tissue_name.capitalize()} - {gene_set.upper()} - Colored by Donor",
        frameon=False,
    )
    
    # Right: Color by cell_type
    sc.pl.umap(
        adata_merged,
        color=cell_type_col,
        ax=axes[1],
        show=False,
        title=f"{tissue_name.capitalize()} - {gene_set.upper()} - Colored by Cell Type",
        frameon=False,
        legend_loc='right margin',
    )
    
    plt.tight_layout()
    
    # Save figure with gene set in filename
    fig_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_merged_umap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization: {fig_path}")
    plt.close()


def process_tissue_directory(tissue_dir, tissue_name, model):
    """Process all h5ad files in a tissue directory for all gene sets."""
    print(f"\n{'='*60}")
    print(f"Processing {tissue_name.upper()} tissue")
    print(f"{'='*60}")
    
    # Process each gene set separately
    for gene_set in ["all", "5k", "xenium"]:
        print(f"\n{'='*60}")
        print(f"Gene Set: {gene_set.upper()}")
        print(f"{'='*60}")
        
        # Find files matching this gene set
        h5ad_files = list(Path(tissue_dir).glob(f"{tissue_name}_*_{gene_set}.h5ad"))
        
        if len(h5ad_files) == 0:
            print(f"No {gene_set} files found in {tissue_dir}")
            continue
        
        print(f"Found {len(h5ad_files)} donor files for {gene_set} gene set")
        
        # Extract embeddings for each donor and save in-place
        print(f"\nExtracting embeddings for each donor...")
        for h5ad_path in h5ad_files:
            extract_and_save_embedding(h5ad_path, tissue_name, gene_set, model)
        
        # Merge all donors and create combined visualization
        print(f"\nCreating combined visualization for {gene_set}...")
        merge_and_visualize_tissue(h5ad_files, tissue_name, gene_set)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("xVerse Embedding Extraction and Visualization")
    print("="*60)
    
    # Load pretrained model once
    model = load_pretrained_model()
    
    # Process liver
    process_tissue_directory(liver_dir, "liver", model)
    
    # Process brain
    process_tissue_directory(brain_dir, "brain", model)
    
    # Save timing records to CSV
    if timing_records:
        df_timing = pd.DataFrame(timing_records)
        timing_csv_path = os.path.join(output_dir, "xverse_inference_timing.csv")
        df_timing.to_csv(timing_csv_path, index=False)
        print(f"\n⏱️  Timing log saved to: {timing_csv_path}")
        print(f"\nTiming Summary:")
        print(df_timing.to_string(index=False))
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"Output directory: {output_dir}")
