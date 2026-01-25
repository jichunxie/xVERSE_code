import os
import numpy as np
import pandas as pd
import scanpy as sc
import tiledbsoma
import scipy.sparse
import cellxgene_census

VALID_TISSUE_GENERAL_IDS = [
    "UBERON:0000004", "UBERON:0000010", "UBERON:0000029", "UBERON:0000030", "UBERON:0000033",
    "UBERON:0000056", "UBERON:0000059", "UBERON:0000160", "UBERON:0000178", "UBERON:0000310",
    "UBERON:0000344", "UBERON:0000403", "UBERON:0000473", "UBERON:0000916", "UBERON:0000922",
    "UBERON:0000945", "UBERON:0000948", "UBERON:0000955", "UBERON:0000970", "UBERON:0000974",
    "UBERON:0000977", "UBERON:0000990", "UBERON:0000992", "UBERON:0000995", "UBERON:0001004",
    "UBERON:0001007", "UBERON:0001013", "UBERON:0001015", "UBERON:0001017", "UBERON:0001032",
    "UBERON:0001040", "UBERON:0001043", "UBERON:0001087", "UBERON:0001155", "UBERON:0001255",
    "UBERON:0001264", "UBERON:0001434", "UBERON:0001723", "UBERON:0001836", "UBERON:0001851",
    "UBERON:0001913", "UBERON:0001987", "UBERON:0002048", "UBERON:0002049", "UBERON:0002097",
    "UBERON:0002102", "UBERON:0002103", "UBERON:0002106", "UBERON:0002107", "UBERON:0002108",
    "UBERON:0002110", "UBERON:0002113", "UBERON:0002240", "UBERON:0002365", "UBERON:0002367",
    "UBERON:0002368", "UBERON:0002369", "UBERON:0002371", "UBERON:0002405", "UBERON:0003688",
    "UBERON:0003889", "UBERON:0007650", "UBERON:0009472", "UBERON:0016435", "UBERON:0018707",
    "UBERON:0035210", "UBERON:8480009"
]

def extract_metadata_fields(census_version="2025-01-30"):
    fields = [
        'assay_ontology_term_id', 'cell_type_ontology_term_id', 'development_stage_ontology_term_id',
        'disease_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'sex_ontology_term_id',
        'tissue_general_ontology_term_id', 'dataset_id', 'donor_id'
    ]
    with cellxgene_census.open_soma(census_version=census_version) as census:
        soma = census["census_data"]["homo_sapiens"]
        obs_df = soma.obs.read().concat().to_pandas()
    obs_df = obs_df[fields]
    obs_df = obs_df[obs_df['cell_type_ontology_term_id'] != 'unknown']
    return obs_df

def extract_raw_expression_from_tissue(tissue_id, donor_id, dataset_id, census):
    # Use provided open census object
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        measurement_name="RNA",
        obs_value_filter=(
            f'dataset_id == "{dataset_id}" and '
            f'donor_id == "{donor_id}" and '
            f'tissue_general_ontology_term_id == "{tissue_id}"'
        ),
        X_name="raw"
    )

    if adata.n_obs == 0:
        return None
    return adata

def qc_cellxgene(adata, gene_ids_filter):
    if adata.n_obs < 500:
        print(f"Skipped QC (only {adata.n_obs} cells)")
        return None

    adata.var["gene_ids"] = adata.var["feature_id"]
    adata.var.drop(columns=["feature_id"], inplace=True)

    mask = adata.var["gene_ids"].isin(gene_ids_filter)
    adata = adata[:, mask].copy()  # avoid ImplicitModificationWarning
    print(f"Filtered genes to {mask.sum()}")

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    num_genes = adata.shape[1]
    gene_threshold = 0.01 * num_genes

    gene_median = np.median(adata.obs['n_genes_by_counts'])
    gene_std = np.std(adata.obs['n_genes_by_counts'])
    umi_median = np.median(adata.obs['total_counts'])
    umi_std = np.std(adata.obs['total_counts'])

    max_gene_threshold = gene_median + 3 * gene_std
    max_umi_threshold = min(20000, umi_median + 3 * umi_std)

    valid_indices = np.where(
        (adata.obs['n_genes_by_counts'] >= gene_threshold) &
        (adata.obs['n_genes_by_counts'] <= max_gene_threshold) &
        (adata.obs['total_counts'] <= max_umi_threshold)
    )[0]

    adata = adata[valid_indices].copy()
    print(f'Number of cells after QC is {adata.n_obs}')
    return adata

def save_adata_in_chunks(adata, donor_id, dataset_id, output_dir, chunk_size=10000):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize donor_id and dataset_id to prevent path issues
    donor_id_safe = donor_id.replace('/', '-')
    dataset_id_safe = dataset_id.replace('/', '-')
    
    total_cells = adata.n_obs
    num_chunks = (total_cells + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_cells)
        chunk = adata[start:end]
        
        matrix_file = os.path.join(output_dir, f"{donor_id_safe}__{dataset_id_safe}__chunk{i}_X.npz")
        meta_file = os.path.join(output_dir, f"{donor_id_safe}__{dataset_id_safe}__chunk{i}_obs.npz")
        
        scipy.sparse.save_npz(matrix_file, chunk.X.tocsr())
        metadata = chunk.obs.reset_index()
        metadata_dict = {col: metadata[col].values for col in metadata.columns}
        metadata_dict["gene_ids"] = chunk.var["gene_ids"].values
        np.savez_compressed(meta_file, **metadata_dict)
        
        print(f"Saved chunk {i+1}/{num_chunks} to {matrix_file} and {meta_file}")