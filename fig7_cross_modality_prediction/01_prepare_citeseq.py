import pandas as pd
import scanpy as sc
import os
import scipy.sparse as sp

input_dir = "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290"
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample"
os.makedirs(output_dir, exist_ok=True)

# ============================
# 1. Load all input tables
# ============================
rna = pd.read_csv(f"{input_dir}/GSE291290_Genes_raw.csv")
adt = pd.read_csv(f"{input_dir}/GSE291290_Ab_raw.csv")
vdj = pd.read_csv(f"{input_dir}/GSE291290_Vdj_combined.csv")
meta = pd.read_csv(f"{input_dir}/GSE291290_Metadata.csv")

cell_col = "Cell_Index"
rna = rna.set_index(cell_col)
adt = adt.set_index(cell_col)
vdj = vdj.set_index(cell_col)
meta = meta.set_index(cell_col)

# ============================
# 2. Load gene name → Ensembl ID mapping
# ============================
gene_info_path = "/hpc/group/xielab/xj58/general/gene_info_table.csv"
gene_info = pd.read_csv(gene_info_path)

# Keep only genes with unique gene_name
gene_info_unique = gene_info.drop_duplicates(subset=["gene_name"], keep=False)

# Build mapping dict
mapping = dict(zip(gene_info_unique["gene_name"], gene_info_unique["ensembl_id"]))


def get_ensembl(gene):
    """Return Ensembl ID if exists and is unique; otherwise return None."""
    return mapping.get(gene, None)


# ============================
# 3. Split by sample and save h5ad
# ============================
samples = meta["SampleName"].unique()

for sample in samples:
    print(f"\nProcessing sample: {sample}")
    sample_cells = meta[meta["SampleName"] == sample].index
    print(f"  Number of cells: {len(sample_cells)}")

    # ---------------------------------------
    # RNA modality (sparse)
    # ---------------------------------------
    if len(set(sample_cells) & set(rna.index)) > 0:
        rna_sub = rna.loc[sample_cells].copy()

        # Map gene names to Ensembl IDs
        gene_names = list(rna_sub.columns)
        ensembl_ids = [get_ensembl(g) for g in gene_names]

        # Filter genes: keep only those with valid Ensembl ID
        valid_mask = [eid is not None for eid in ensembl_ids]
        rna_sub = rna_sub.loc[:, valid_mask]

        # Build var table
        valid_gene_names = rna_sub.columns
        print('  RNA modality: keeping {}/{} genes with valid Ensembl IDs.'.format(
            len(valid_gene_names), len(gene_names)
        ))

        var = pd.DataFrame(index=valid_gene_names)
        var["gene_ids"] = var.index.map(get_ensembl)

        # Convert to sparse matrix
        X_sparse = sp.csr_matrix(rna_sub.values.astype("float32"))

        adata_rna = sc.AnnData(
            X=X_sparse,
            obs=pd.DataFrame(index=rna_sub.index),
            var=var
        )
        adata_rna.obs['tissue'] = 'blood' 
        adata_rna.write_h5ad(f"{output_dir}/{sample}_rna.h5ad")

    # ---------------------------------------
    # ADT modality (sparse)
    # ---------------------------------------
    if len(set(sample_cells) & set(adt.index)) > 0:
        adt_sub = adt.loc[sample_cells].copy()

        var = pd.DataFrame(index=adt_sub.columns)

        X_sparse = sp.csr_matrix(adt_sub.values.astype("float32"))

        adata_adt = sc.AnnData(
            X=X_sparse,
            obs=pd.DataFrame(index=adt_sub.index),
            var=var
        )
        adata_adt.write_h5ad(f"{output_dir}/{sample}_adt.h5ad")

    # ---------------------------------------
    # VDJ modality (store as metadata in obs)
    # ---------------------------------------
    if len(set(sample_cells) & set(vdj.index)) > 0:
        vdj_sub = vdj.loc[sample_cells].copy()

        # Create empty numeric matrix (required by AnnData)
        X_dummy = sp.csr_matrix((vdj_sub.shape[0], 0))

        # obs = the VDJ table
        obs = vdj_sub.copy()

        # var = empty
        var = pd.DataFrame(index=[])

        adata_vdj = sc.AnnData(
            X=X_dummy,
            obs=obs,
            var=var
        )

        adata_vdj.write_h5ad(f"{output_dir}/{sample}_vdj.h5ad")

print("\nAll samples processed. Finished.")
