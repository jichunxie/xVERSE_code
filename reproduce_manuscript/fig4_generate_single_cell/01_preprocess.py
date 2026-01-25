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

"""https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE180878"""
"""	A total of 16 breast samples were assayed (4 samples from reductive mammoplasties and 12 from prophylactic mastectomies)."""
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad

# ----------------------------------------
# 0. Paths
# ----------------------------------------
matrix_path = "/hpc/group/xielab/xj58/xVerse_results/fig4/data/GSE180878_Li_Brugge_10XscRNAseq_GeneCellMatrix_RNAcounts_human.csv"
meta_path = "/hpc/group/xielab/xj58/xVerse_results/fig4/data/GSE180878_Li_Brugge_10XscRNAseq_Metadata_human.csv"
gene_info_path = "/hpc/group/xielab/xj58/general/gene_info_table.csv"
output_dir = "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------
# 1. Load gene × cell matrix
# ----------------------------------------
print("Loading CSV matrix...")
df = pd.read_csv(matrix_path, index_col=0)

genes = df.index.astype(str)
cells = df.columns.astype(str)

# Convert to sparse and transpose to cell × gene
print("Converting to sparse matrix...")
X = sp.csr_matrix(df.values.T.astype(np.float32))
adata = ad.AnnData(X=X)
adata.obs_names = cells
adata.var_names = genes

del df

# ----------------------------------------
# 2. Build mapping and filter genes (unique mapping)
# ----------------------------------------
print("Loading gene_info and building unique mapping...")
gene_info = pd.read_csv(gene_info_path)
gene_info["gene_name"] = gene_info["gene_name"].astype(str)
adata.var_names = adata.var_names.astype(str)

gene_info_unique = gene_info.drop_duplicates(subset=["gene_name"], keep=False)
mapping = dict(zip(gene_info_unique["gene_name"], gene_info_unique["ensembl_id"]))

genes_to_keep = [g for g in adata.var_names if g in mapping]
print(f"Total genes in matrix: {adata.n_vars}")
print(f"Genes with unique mapping: {len(genes_to_keep)}")

adata = adata[:, genes_to_keep].copy()
adata.var["gene_symbol"] = adata.var_names
adata.var["gene_ids"] = adata.var["gene_symbol"].map(mapping)

# ----------------------------------------
# 3. Minimal QC (simple, safe, common)
# ----------------------------------------

# 3.1 Compute QC metrics
print("Computing QC metrics...")
adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

# Per-cell maximum gene count (to catch extreme expression spikes)
row_max_mat = adata.X.max(axis=1)
if sp.issparse(row_max_mat):
    row_max = row_max_mat.toarray().ravel()
else:
    row_max = np.asarray(row_max_mat).ravel()
adata.obs["max_gene_count"] = row_max

# Print quantiles for inspection
quantiles = [0.5, 0.9, 0.95, 0.99, 0.999]
q_total = np.quantile(adata.obs["total_counts"], quantiles)
q_rowmax = np.quantile(row_max, quantiles)
print(f"Quantiles for total_counts: {dict(zip(quantiles, q_total))}")
print(f"Quantiles for max_gene_count: {dict(zip(quantiles, q_rowmax))}")

# mitochondrial genes: gene_symbol starts with MT-
adata.var["is_mito"] = adata.var["gene_symbol"].str.upper().str.startswith("MT-")
mt_idx = np.where(adata.var["is_mito"])[0]

if len(mt_idx) > 0:
    mt_counts = np.asarray(adata.X[:, mt_idx].sum(axis=1)).ravel()
    adata.obs["pct_mt"] = mt_counts / adata.obs["total_counts"]
else:
    adata.obs["pct_mt"] = 0.0

# 3.2 Filters (standard thresholds)
print("Running QC filters...")
cell_filter = (
    (adata.obs["total_counts"] >= 200) &
    (adata.obs["n_genes"] >= 200) &
    (adata.obs["pct_mt"] <= 0.20) &
    (adata.obs["max_gene_count"] <= 500) &
    (adata.obs["total_counts"] <= 200000)
)

print(f"Cells before QC: {adata.n_obs}")
print(f"Cells after QC:  {cell_filter.sum()}")

adata = adata[cell_filter, :].copy()

# 3.3 Set tissue label to breast for all cells
adata.obs["tissue"] = "breast"

# 3.4 Remove genes with zero counts
gene_filter = np.asarray((adata.X.sum(axis=0) > 0)).ravel()
adata = adata[:, gene_filter].copy()

print(f"Genes after removing zero-count genes: {adata.n_vars}")

# ----------------------------------------
# 3.5 Load metadata and store in obs
# ----------------------------------------
print("Loading metadata and merging into obs...")
metadata = pd.read_csv(meta_path)
if "cellID" not in metadata.columns:
    raise ValueError("Metadata file missing 'cellID' column")
metadata["cellID"] = metadata["cellID"].astype(str)
metadata = metadata.set_index("cellID")
missing_meta = np.setdiff1d(adata.obs_names.astype(str), metadata.index.astype(str))
if len(missing_meta) > 0:
    print(f"[WARN] {len(missing_meta)} cells missing metadata; filling NA")
metadata = metadata.reindex(adata.obs_names)
adata.obs = adata.obs.join(metadata)

# ----------------------------------------
# 4. Parse sample name and split
# ----------------------------------------
print("Parsing sample names from cell IDs...")
adata.obs["sample"] = adata.obs_names.str.split("_").str[0]
samples = adata.obs["sample"].unique()
print(f"Found {len(samples)} samples:", samples)

# ----------------------------------------
# 5. Save one h5ad per sample
# ----------------------------------------
for s in samples:
    print(f"Processing sample {s}...")

    mask = adata.obs["sample"] == s
    adata_sub = adata[mask, :].copy()

    print(f"  {s}: {adata_sub.n_obs} cells, {adata_sub.n_vars} genes")

    out_path = os.path.join(output_dir, f"{s}.h5ad")
    adata_sub.write_h5ad(out_path)

    print(f"  Saved to {out_path}")
