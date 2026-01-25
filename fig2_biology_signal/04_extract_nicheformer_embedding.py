#!/usr/bin/env python3
"""
Extract Nicheformer embeddings for liver and brain donor datasets.
Zero-shot inference using pre-trained Nicheformer model.
"""

import os
import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from pathlib import Path
from tqdm import tqdm

from nicheformer.models import Nicheformer
from nicheformer.data import NicheformerDataset

# ================= CONFIGURATION =================
liver_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/liver"
brain_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/brain"
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/nicheformer_embeddings"

os.makedirs(output_dir, exist_ok=True)

# Nicheformer model paths
model_h5ad = "/hpc/group/xielab/xj58/nicheformer/data/model_means/model.h5ad"
technology_mean_path = "/hpc/group/xielab/xj58/nicheformer/data/model_means/dissociated_mean_script.npy"
checkpoint_path = "/hpc/group/xielab/xj58/nicheformer/data/nicheformer.ckpt"

batch_size = 32
max_seq_len = 1500
aux_tokens = 30
chunk_size = 1000
num_workers = 4
embedding_layer = -1  # Last layer

# Metadata for Nicheformer
# modality: dissociated=3, spatial=4
# specie: human=5, mouse=6
# assay: 10x transcription profiling=15
MODALITY = 3  # dissociated
SPECIE = 5    # human
ASSAY = 15    # 10x transcription profiling
ASSAY = 15    # 10x transcription profiling

# Timing data collection
timing_records = []
# ===============================================

# Set random seed
pl.seed_everything(42)


def add_nicheformer_embedding(h5ad_path, model_h5ad, technology_mean_path, checkpoint_path, tissue_name, gene_set):
    """
    Extract Nicheformer embeddings for a single h5ad file.
    
    Args:
        h5ad_path: Path to h5ad file
        model_h5ad: Path to model h5ad for gene alignment
        technology_mean_path: Path to technology mean file
        checkpoint_path: Path to Nicheformer checkpoint
    
    Returns:
        Updated AnnData with embeddings in obsm["nicheformer"]
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(h5ad_path).name}")
    print(f"{'='*60}")
    
    # Load data
    adata = ad.read_h5ad(h5ad_path)
    
    # Extract donor ID
    donor_id = h5ad_path.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
    
    # Keep reference to original adata for saving later
    adata_original = adata
    
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Load model h5ad and technology mean
    print("  Loading model and technology mean...")
    model = ad.read_h5ad(model_h5ad)
    technology_mean = np.load(technology_mean_path)
    
    print(f"  Model: {model.n_obs} obs × {model.n_vars} genes")
    print(f"  Technology mean: {len(technology_mean)} values")
    
    # Verify model and technology_mean match
    if model.n_vars != len(technology_mean):
        raise ValueError(
            f"Model has {model.n_vars} genes but technology_mean has {len(technology_mean)} values. "
            f"These should match. Please check your model.h5ad and technology_mean files."
        )
    
    # CRITICAL: Subset data to only include genes that are in the model
    # This ensures we keep all model genes and drop any extra genes in data
    model_genes = set(model.var.index)
    data_genes = set(adata.var.index)
    
    common_genes = model_genes & data_genes
    extra_genes = data_genes - model_genes
    missing_genes = model_genes - data_genes
    
    print(f"  Data genes: {len(data_genes)}")
    print(f"  Common genes: {len(common_genes)}")
    print(f"  Extra genes in data (will be dropped): {len(extra_genes)}")
    print(f"  Missing genes in data (will be filled with 0): {len(missing_genes)}")
    
    # Subset data to only common genes (drops extra genes)
    common_genes_list = sorted(list(common_genes))
    adata = adata[:, common_genes_list].copy()
    print(f"  Data after subsetting: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Format data properly with the model
    # Use OUTER join - this will keep all model genes and fill missing ones with 0
    
    # Start timing (including alignment and dataset creation as part of model prep)
    start_time = time.time()
    
    print("  Aligning with model (outer join to keep all model genes)...")
    adata = ad.concat([model, adata], join='outer', axis=0)
    # Drop the first observation (model placeholder)
    adata = adata[1:].copy()
    
    print(f"  After alignment: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Verify dimensions match
    if adata.n_vars != len(technology_mean):
        raise ValueError(
            f"Dimension mismatch: adata has {adata.n_vars} genes but expected {len(technology_mean)} genes"
        )
    
    # Add required metadata
    adata.obs['modality'] = MODALITY
    adata.obs['specie'] = SPECIE
    adata.obs['assay'] = ASSAY
    adata.obs['nicheformer_split'] = 'train'
    
    # Create dataset (following official notebook)
    print("  Creating Nicheformer dataset...")
    dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=max_seq_len,
        aux_tokens=aux_tokens,
        chunk_size=chunk_size,
        metadata_fields={'obs': ['modality', 'specie', 'assay']}  # Include assay!
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Load pre-trained model
    print("  Loading Nicheformer model...")
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False
    )
    model.eval()
    
    # Extract embeddings
    print("  Extracting embeddings...")
    embeddings = []
    device = model.embeddings.weight.device
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Processing batches"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Get embeddings from the model
            emb = model.get_embeddings(
                batch=batch,
                layer=embedding_layer
            )
            embeddings.append(emb.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    
    inference_time = time.time() - start_time
    
    # Record timing
    timing_records.append({
        'model': 'Nicheformer',
        'tissue': tissue_name,
        'gene_set': gene_set,
        'donor': donor_id,
        'n_cells': adata_original.n_obs,
        'n_genes': adata_original.n_vars,
        'time_seconds': inference_time
    })
    print(f"    ⏱️  Inference time: {inference_time:.2f}s")
    
    print(f"  Extracted embeddings shape: {embeddings.shape}")
    
    # Load original data and add embeddings
    adata_original = ad.read_h5ad(h5ad_path)
    adata_original.obsm["nicheformer"] = embeddings
    
    # Save updated h5ad (in-place)
    adata_original.write(h5ad_path)
    print(f"  Updated h5ad with nicheformer embeddings")
    
    return adata_original


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
        print(f"{adata.n_obs} cells")
        adata_list.append(adata)
    
    # Merge all donors
    print(f"\n  Merging {len(adata_list)} donors...")
    adata_merged = sc.concat(adata_list, join="outer", index_unique=None)
    print(f"  Total: {adata_merged.n_obs} cells")
    
    # Compute UMAP on merged nicheformer embeddings
    print("  Computing UMAP on nicheformer embeddings...")
    sc.pp.neighbors(adata_merged, use_rep="nicheformer", n_neighbors=15)
    sc.tl.umap(adata_merged)
    
    # Find cell_type column
    cell_type_col = None
    for col in ['cell_type', 'celltype', 'cell_type_ontology_term_id', 'celltype.l2']:
        if col in adata_merged.obs.columns:
            cell_type_col = col
            break
    
    if cell_type_col is None:
        print(f"  Warning: No cell_type column found")
        cell_type_col = adata_merged.obs.columns[0]
    
    # Plot UMAP
    print(f"  Plotting UMAP colored by donor_id and {cell_type_col}...")
    sc.pl.umap(
        adata_merged,
        color=["donor_id", cell_type_col],
        show=False,
        title=[f"Donor ID ({gene_set})", f"Cell Type ({gene_set})"],
        wspace=0.3
    )
    
    # Save figure with gene set in filename
    fig_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_nicheformer_umap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization: {fig_path}")
    plt.close()




def process_tissue_directory(tissue_dir, tissue_name):
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
        
        # Extract embeddings for each donor
        print(f"\nExtracting embeddings for each donor...")
        for h5ad_path in h5ad_files:
            add_nicheformer_embedding(
                h5ad_path,
                model_h5ad,
                technology_mean_path,
                checkpoint_path,
                tissue_name,
                gene_set
            )
        
        # Merge and visualize
        print(f"\nCreating combined visualization for {gene_set}...")
        merge_and_visualize_tissue(h5ad_files, tissue_name, gene_set)




# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Nicheformer Embedding Extraction (Zero-shot)")
    print("="*60)
    
    # Process liver
    process_tissue_directory(liver_dir, "liver")
    
    # Process brain
    process_tissue_directory(brain_dir, "brain")
    
    # Save timing records to CSV
    if timing_records:
        df_timing = pd.DataFrame(timing_records)
        timing_csv_path = os.path.join(output_dir, "nicheformer_inference_timing.csv")
        df_timing.to_csv(timing_csv_path, index=False)
        print(f"\n⏱️  Timing log saved to: {timing_csv_path}")
        print(f"\nTiming Summary:")
        print(df_timing.to_string(index=False))
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
