import os
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


data_dir = "/hpc/group/xielab/xj58/xVerse_results/fig6/GSE164378_RAW"
gene_info_path = "/hpc/group/xielab/xj58/general/gene_info_table.csv"


def read_10x_rna(prefix):
    """
    Load RNA modality from 10x-style files:
    <prefix>-matrix.mtx.gz
    <prefix>-features.tsv.gz
    <prefix>-barcodes.tsv.gz
    """
    mtx_file = prefix + "-matrix.mtx.gz"
    feat_file = prefix + "-features.tsv.gz"
    bc_file = prefix + "-barcodes.tsv.gz"

    X = sc.read_mtx(mtx_file).X.transpose().tocsr()

    with gzip.open(feat_file, "rt") as f:
        feat = pd.read_csv(f, sep="\t", header=None)
    feat.columns = ["gene_id", "gene_symbol", "feature_type"]

    with gzip.open(bc_file, "rt") as f:
        barcodes = pd.read_csv(f, sep="\t", header=None)[0].tolist()

    adata = sc.AnnData(X=X)
    adata.var["gene_ids"] = feat["gene_id"].values
    adata.var["gene_symbols"] = feat["gene_symbol"].values
    adata.var_names = adata.var["gene_symbols"]
    adata.obs_names = barcodes

    return adata


def collect_prefixes(files, tag):
    """Extract unique prefixes for RNA_<tag> (3P or 5P)."""
    prefixes = [
        f.replace("-matrix.mtx.gz", "")
         .replace("-barcodes.tsv.gz", "")
         .replace("-features.tsv.gz", "")
        for f in files
        if f"RNA_{tag}" in f
    ]
    return sorted(list(set(prefixes)))


def load_gene_mapping():
    """Load gene_name -> ensembl_id mapping, keeping only unique gene names."""
    gene_info = pd.read_csv(gene_info_path)
    gene_info_unique = gene_info.drop_duplicates(subset=["gene_name"], keep=False)
    return dict(zip(gene_info_unique["gene_name"], gene_info_unique["ensembl_id"]))


def keep_unique_gene_ids(adata, mapping):
    """Attach gene IDs and drop genes with ambiguous or missing IDs."""
    adata.var_names = adata.var_names.astype(str)
    adata.var["gene_symbol"] = adata.var_names
    adata.var["gene_ids"] = adata.var["gene_symbol"].map(mapping)
    mask = adata.var["gene_ids"].notnull() & ~adata.var["gene_ids"].duplicated(keep=False)
    return adata[:, mask].copy()


def standard_qc(adata, min_genes=200, max_genes=8000, max_pct_mt=0.20, min_cells=3):
    """Perform QC filtering."""
    adata.var["mt"] = adata.var["gene_symbol"].str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    n_genes = adata.obs["n_genes_by_counts"]
    pct_mt = adata.obs["pct_counts_mt"]
    total_counts = adata.obs["total_counts"]

    tc_q = np.quantile(total_counts, [0.95, 0.99, 1.0])
    print(f"QC total_counts quantiles: 95%={tc_q[0]:.1f}, 99%={tc_q[1]:.1f}")

    mask_min = n_genes >= min_genes
    mask_max = n_genes <= max_genes
    mask_mt = pct_mt <= max_pct_mt * 100

    tc_limit = np.quantile(total_counts, 0.95)
    mask_tc = total_counts <= tc_limit

    adata = adata[mask_min & mask_max & mask_mt & mask_tc].copy()

    gene_mask = adata.var["n_cells_by_counts"] >= min_cells
    adata = adata[:, gene_mask].copy()

    return adata


def global_expression_quantiles(adata):
    quantiles = [0.5, 0.9, 0.99]
    X = adata.X
    if sp.issparse(X):
        data = np.asarray(X.data, dtype=float)
        if data.size == 0:
            return {q: 0.0 for q in quantiles}
        data.sort()
        return {q: float(data[int(q * (len(data) - 1))]) for q in quantiles}
    else:
        return {q: 0.0 for q in quantiles}


def print_global_quantiles(adata, tag):
    q = global_expression_quantiles(adata)
    q_str = ", ".join(f"{k}: {v:.4f}" for k, v in q.items())
    print(f"{tag} global expression quantiles -> {q_str}")


def print_celltype_hierarchy(adata, tag):
    """Print celltype.l1 and their corresponding celltype.l2."""
    print(f"\n[{tag}] Cell Type Hierarchy (L1 -> L2):")
    if "celltype.l1" not in adata.obs or "celltype.l2" not in adata.obs:
        print("  Missing celltype.l1 or celltype.l2 columns.")
        return

    l1_counts = adata.obs["celltype.l1"].value_counts()
    
    for l1 in l1_counts.index:
        subset = adata.obs[adata.obs["celltype.l1"] == l1]
        l2_counts = subset["celltype.l2"].value_counts()
        
        print(f"  L1: {l1} (n={len(subset)})")
        for l2, count in l2_counts.items():
            print(f"    - {l2} (n={count})")


def prepare_adata(adata, tag, meta):
    """Map genes, QC filter, join metadata, and add tissue annotation."""
    adata = keep_unique_gene_ids(adata, mapping)
    adata = standard_qc(adata)
    adata.obs = adata.obs.join(meta, how="left")
    adata.obs["tissue"] = "blood"
    print(f"{tag} processed: obs={adata.n_obs}, var={adata.n_vars}")
    print_global_quantiles(adata, tag)
    print_celltype_hierarchy(adata, tag)
    return adata


def load_rna_group(prefixes):
    """
    Sum RNA matrices from multiple GSMs if multiple exist.
    All share the same genes and barcodes structure per GSM.
    """
    adata_list = []
    for p in prefixes:
        full_prefix = os.path.join(data_dir, p)
        adata_list.append(read_10x_rna(full_prefix))

    if len(adata_list) == 1:
        return adata_list[0]

    return sc.concat(adata_list, join="inner", axis=0)


# ---------------------------------------------------------------
# Identify 3P and 5P RNA prefixes
# ---------------------------------------------------------------

files = os.listdir(data_dir)

prefix_3p = collect_prefixes(files, "3P")
prefix_5p = collect_prefixes(files, "5P")

# ---------------------------------------------------------------
# Load meta
# meta CSV first column = cell id, header = obs columns
# ---------------------------------------------------------------

meta_3p = pd.read_csv(os.path.join(data_dir, "GSE164378_sc.meta.data_3P.csv"), index_col=0)
meta_5p = pd.read_csv(os.path.join(data_dir, "GSE164378_sc.meta.data_5P.csv"), index_col=0)

mapping = load_gene_mapping()

# ---------------------------------------------------------------
# Build RNA AnnData
# ---------------------------------------------------------------

adata_3p = load_rna_group(prefix_3p)
adata_5p = load_rna_group(prefix_5p)

# ---------------------------------------------------------------
# Gene mapping, QC, meta integration, and tissue annotation
# ---------------------------------------------------------------

adata_3p = prepare_adata(adata_3p, "3P", meta_3p)
adata_5p = prepare_adata(adata_5p, "5P", meta_5p)

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------

adata_3p.write(os.path.join(data_dir, "rna_3P.h5ad"))
adata_5p.write(os.path.join(data_dir, "rna_5P.h5ad"))

print("Finished.")
