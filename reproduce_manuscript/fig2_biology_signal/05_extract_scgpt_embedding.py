#!/usr/bin/env python3
"""
Extract scGPT embeddings for liver and brain donor datasets.
Zero-shot inference using pre-trained scGPT model.
"""

import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scgpt as scg
from pathlib import Path

# ================= CONFIGURATION =================
liver_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/liver"
brain_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/brain"
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/scgpt_embeddings"

os.makedirs(output_dir, exist_ok=True)

# scGPT model and mapping paths
model_dir = "/hpc/group/xielab/xj58/general/scGPT_human"
mapping_csv_path = "/hpc/group/xielab/xj58/general/gene_info_table.csv"

batch_size = 64

# Timing data collection
timing_records = []
# ===============================================


def add_scgpt_embedding(h5ad_path, model_dir, ensembl2name_dict, batch_size, tissue_name, gene_set):
    """
    Extract scGPT embeddings for a single h5ad file.
    
    Args:
        h5ad_path: Path to h5ad file
        model_dir: Path to pre-trained scGPT model
        ensembl2name_dict: Dictionary mapping ensembl_id to gene_name
        batch_size: Batch size for embedding extraction
    
    Returns:
        Updated AnnData with embeddings in obsm["scgpt"]
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(h5ad_path).name}")
    print(f"{'='*60}")
    
    # Load data
    adata = sc.read_h5ad(h5ad_path)
    
    # Extract donor ID
    donor_id = h5ad_path.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
    
    # Keep reference to original adata for saving later
    adata_original = adata
    
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Map ensembl_id to gene_name
    print("  Mapping ensembl IDs to gene names...")
    gene_ids = adata.var["gene_ids"].tolist()
    gene_names = [ensembl2name_dict.get(eid, "") for eid in gene_ids]
    adata.var["gene_name"] = gene_names
    
    # Convert any NaN values to empty strings (important for scGPT)
    adata.var["gene_name"] = adata.var["gene_name"].fillna("")
    
    # Filter out genes without valid names
    n_before = adata.n_vars
    adata = adata[:, adata.var["gene_name"] != ""].copy()
    n_after_empty = adata.n_vars
    print(f"  Removed genes without names: {n_before} → {n_after_empty} ({n_before - n_after_empty} removed)")
    
    # Keep only genes with unique gene names (remove duplicates)
    gene_name_counts = adata.var["gene_name"].value_counts()
    unique_gene_names = gene_name_counts[gene_name_counts == 1].index.tolist()
    adata = adata[:, adata.var["gene_name"].isin(unique_gene_names)].copy()
    n_after_unique = adata.n_vars
    print(f"  Kept only unique gene names: {n_after_empty} → {n_after_unique} ({n_after_empty - n_after_unique} duplicates removed)")
    
    # Run scGPT embedding
    print("  Extracting scGPT embeddings...")
    
    # Start timing
    start_time = time.time()
    
    embed_adata = scg.tasks.embed_data(
        adata,
        Path(model_dir),
        gene_col="gene_name",
        batch_size=batch_size,
    )
    
    inference_time = time.time() - start_time
    
    # Record timing
    timing_records.append({
        'model': 'scGPT',
        'tissue': tissue_name,
        'gene_set': gene_set,
        'donor': donor_id,
        'n_cells': adata_original.n_obs,
        'n_genes': adata_original.n_vars,
        'time_seconds': inference_time
    })
    print(f"    ⏱️  Inference time: {inference_time:.2f}s")
    
    embeddings = embed_adata.obsm["X_scGPT"]
    print(f"  Extracted embeddings shape: {embeddings.shape}")
    
    # Load original data and add embeddings
    adata_original = sc.read_h5ad(h5ad_path)
    adata_original.obsm["scgpt"] = embeddings
    
    # Save updated h5ad (in-place)
    adata_original.write(h5ad_path)
    print(f"  Updated h5ad with scgpt embeddings")
    
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
    
    # Compute UMAP on merged scgpt embeddings
    print("  Computing UMAP on scgpt embeddings...")
    sc.pp.neighbors(adata_merged, use_rep="scgpt", n_neighbors=15)
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
    fig_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_scgpt_umap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization: {fig_path}")
    plt.close()




def process_tissue_directory(tissue_dir, tissue_name, model_dir, ensembl2name_dict, batch_size):
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
            add_scgpt_embedding(h5ad_path, model_dir, ensembl2name_dict, batch_size, tissue_name, gene_set)
        
        # Merge and visualize
        print(f"\nCreating combined visualization for {gene_set}...")
        merge_and_visualize_tissue(h5ad_files, tissue_name, gene_set)




# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("scGPT Embedding Extraction (Zero-shot)")
    print("="*60)
    
    # Load gene mapping once
    print("\nLoading gene mapping table...")
    df_map = pd.read_csv(mapping_csv_path)
    ensembl2name_dict = pd.Series(df_map.gene_name.values, index=df_map.ensembl_id).to_dict()
    print(f"Loaded {len(ensembl2name_dict)} gene mappings")
    
    # Process liver
    process_tissue_directory(liver_dir, "liver", model_dir, ensembl2name_dict, batch_size)
    
    # Process brain
    process_tissue_directory(brain_dir, "brain", model_dir, ensembl2name_dict, batch_size)
    
    # Save timing records to CSV
    if timing_records:
        df_timing = pd.DataFrame(timing_records)
        timing_csv_path = os.path.join(output_dir, "scgpt_inference_timing.csv")
        df_timing.to_csv(timing_csv_path, index=False)
        print(f"\n⏱️  Timing log saved to: {timing_csv_path}")
        print(f"\nTiming Summary:")
        print(df_timing.to_string(index=False))
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"Output directory: {output_dir}")
