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
import time
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from geneformer import TranscriptomeTokenizer, EmbExtractor

# ================= CONFIGURATION =================
liver_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/liver"
brain_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/brain"
model_dir = "/hpc/group/xielab/xj58/Geneformer/Geneformer-V2-104M"
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig2/geneformer_embeddings"

os.makedirs(output_dir, exist_ok=True)

batch_size = 64

# Timing data collection
timing_records = []
# ===============================================


def add_geneformer_embedding(h5ad_path, model_dir, batch_size, tissue_name, gene_set):
    """
    Extract Geneformer embeddings for a single h5ad file.
    
    Args:
        h5ad_path: Path to h5ad file
        model_dir: Path to pre-trained Geneformer model
        batch_size: Batch size for embedding extraction
    
    Returns:
        Updated AnnData with embeddings in obsm["geneformer"]
    """
    print(f"  Processing: {Path(h5ad_path).name}")
    
    # Load data
    adata = sc.read_h5ad(h5ad_path)
    print(f"    Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Ensure required fields
    if "ensembl_id" not in adata.var.columns:
        if "gene_ids" in adata.var.columns:
            adata.var["ensembl_id"] = adata.var["gene_ids"]
        else:
            adata.var["ensembl_id"] = adata.var.index
    
    if "n_counts" not in adata.obs.columns:
        adata.obs["n_counts"] = adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=1)
    
    # Create temporary directories
    file_stem = Path(h5ad_path).stem
    base_dir = os.path.join(os.path.dirname(h5ad_path), f"{file_stem}_geneformer_temp")
    
    processed_dir = os.path.join(base_dir, "processed")
    tokenized_dir = os.path.join(base_dir, "tokenized")
    embed_dir = os.path.join(base_dir, "embed")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(tokenized_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    
    dataset_name = file_stem
    processed_file = os.path.join(processed_dir, f"{file_stem}.h5ad")
    
    # Extract donor ID
    donor_id = Path(h5ad_path).stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")

    # Save processed file
    adata.write_h5ad(processed_file)
    
    # Start timing (including tokenization as it's part of model prep)
    start_time = time.time()
    
    # Tokenize
    tk = TranscriptomeTokenizer({}, nproc=16, model_version="V2")
    tk.tokenize_data(processed_dir, tokenized_dir, dataset_name, file_format="h5ad")
    
    # Extract embeddings
    embex = EmbExtractor(
        model_version="V2",
        emb_mode="cell",
        max_ncells=None,
        forward_batch_size=batch_size
    )
    embex.extract_embs(
        model_dir,
        os.path.join(tokenized_dir, f"{dataset_name}.dataset"),
        embed_dir,
        dataset_name
    )
    
    inference_time = time.time() - start_time
    
    # Record timing
    timing_records.append({
        'model': 'Geneformer',
        'tissue': tissue_name,
        'gene_set': gene_set,
        'donor': donor_id,
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'time_seconds': inference_time
    })
    print(f"    ⏱️  Inference time: {inference_time:.2f}s")
    
    # Load embeddings
    emb_file = os.path.join(embed_dir, f"{dataset_name}.csv")
    df = pd.read_csv(emb_file, index_col=0)
    embeddings = df.values
    
    print(f"    Extracted embeddings shape: {embeddings.shape}")
    
    # Save to obsm
    adata.obsm["geneformer"] = embeddings
    
    # Save updated h5ad (in-place)
    adata.write(h5ad_path)
    print(f"    Updated with geneformer embeddings")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(base_dir)
    
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
    
    # Compute UMAP on merged Geneformer embeddings
    print("  Computing UMAP on Geneformer embeddings...")
    sc.pp.neighbors(adata_merged, use_rep="geneformer", n_neighbors=15)
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
    fig_path = os.path.join(output_dir, f"{tissue_name}_{gene_set}_geneformer_umap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization: {fig_path}")
    plt.close()



def process_tissue_directory(tissue_dir, tissue_name, model_dir, batch_size):
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
            add_geneformer_embedding(h5ad_path, model_dir, batch_size, tissue_name, gene_set)
        
        # Merge all donors and create combined visualization
        print(f"\nCreating combined visualization for {gene_set}...")
        merge_and_visualize_tissue(h5ad_files, tissue_name, gene_set)



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Geneformer Embedding Extraction (Zero-shot)")
    print("="*60)
    
    # Process liver
    process_tissue_directory(liver_dir, "liver", model_dir, batch_size)
    
    # Process brain
    process_tissue_directory(brain_dir, "brain", model_dir, batch_size)
    
    # Save timing records to CSV
    if timing_records:
        df_timing = pd.DataFrame(timing_records)
        timing_csv_path = os.path.join(output_dir, "geneformer_inference_timing.csv")
        df_timing.to_csv(timing_csv_path, index=False)
        print(f"\n⏱️  Timing log saved to: {timing_csv_path}")
        print(f"\nTiming Summary:")
        print(df_timing.to_string(index=False))
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
