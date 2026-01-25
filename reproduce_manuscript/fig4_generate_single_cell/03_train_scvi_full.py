import os
import scanpy as sc
import scvi


# ================================================================
# Configuration
# ================================================================

FILE_SAMPLE_LIST = [
    ("PM-A", "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-A.h5ad"),
    ("PM-B", "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-B.h5ad"),
    ("PM-C", "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-C.h5ad"),
    ("PM-D", "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-D.h5ad"),
    ("PM-E", "/hpc/group/xielab/xj58/xVerse_results/fig4/data/per_sample_h5ad/PM-E.h5ad"),
]

gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

OUTPUT_ROOT = "/hpc/group/xielab/xj58/xVerse_results/fig4/scvi_full"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
MODEL_LABEL = "full"


# ================================================================
# Helper functions
# ================================================================

def load_allowed_gene_ids(path):
    """Read allowed gene_ids (newline separated)."""
    with open(path, "r") as f:
        gene_ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(gene_ids)} allowed gene_ids from {path}")
    return set(gene_ids)


def load_filtered_samples(allowed_gene_ids=None):
    """Load raw h5ad samples and optionally filter genes."""
    raw_dict = {}
    total_cells = 0

    for sample_name, path in FILE_SAMPLE_LIST:
        print(f"Loading {sample_name} from {path} ...")
        ad = sc.read_h5ad(path)

        if allowed_gene_ids is not None:
            if "gene_ids" not in ad.var:
                raise KeyError(f"'gene_ids' not found in var for {sample_name}")
            mask = ad.var["gene_ids"].astype(str).isin(allowed_gene_ids)
            keep = int(mask.sum())
            print(f"  Filtering genes by allowed gene_ids: keeping {keep}/{ad.n_vars}")
            ad = ad[:, mask].copy()

        raw_dict[sample_name] = ad
        total_cells += ad.n_obs
        print(f"  {sample_name}: {ad.n_obs} cells")

    print(f"\nTotal cells across samples: {total_cells}\n")
    return raw_dict


def build_full_training_adata(raw_dict):
    """Concatenate all cells from each sample to form training AnnData."""
    ad_list = []

    for sample_name, ad in raw_dict.items():
        ad_with_batch = ad.copy()
        ad_with_batch.obs["sample_batch"] = sample_name
        ad_list.append(ad_with_batch)

    if not ad_list:
        raise ValueError("No samples loaded for training.")

    if len(ad_list) == 1:
        return ad_list[0]

    return ad_list[0].concatenate(
        ad_list[1:], index_unique=None, batch_key="concat_batch"
    )


# ================================================================
# Main pipeline
# ================================================================

def main():
    allowed_gene_ids = load_allowed_gene_ids(gene_ids_path)

    # Load raw samples
    raw_dict = load_filtered_samples(allowed_gene_ids)

    print("\n" + "=" * 70)
    print(f"Training SCVI on all cells ({MODEL_LABEL})")
    print("=" * 70)

    # ---- build training data ----
    ad_train = build_full_training_adata(raw_dict)
    print(f"Training cells = {ad_train.n_obs}, genes = {ad_train.n_vars}")

    # ---- setup + train SCVI ----
    print("Setting up SCVI ...")
    scvi.model.SCVI.setup_anndata(ad_train, batch_key="sample_batch")

    print("Initializing SCVI model ...")
    model = scvi.model.SCVI(ad_train)

    print("Training ...")
    model.train(early_stopping=True)

    # Save model
    model_path = os.path.join(OUTPUT_ROOT, f"scvi_{MODEL_LABEL}.pt")
    model.save(model_path, overwrite=True)
    print(f"Saved model: {model_path}")

    # ---- full-data inference on all samples ----
    print("Running full-data inference for all samples ...")

    for sample_name, _ in FILE_SAMPLE_LIST:
        print(f"  Inference on {sample_name} ...")
        ad_raw = raw_dict[sample_name].copy()

        # Add the same obs column used during training
        ad_raw.obs["sample_batch"] = sample_name

        # Setup anndata with the same key used in training
        scvi.model.SCVI.setup_anndata(ad_raw, batch_key="sample_batch")

        expr_mu = model.get_likelihood_parameters(adata=ad_raw)["mean"]

        ad_out = sc.AnnData(
            X=expr_mu,
            obs=ad_raw.obs.copy(),
            var=ad_raw.var.copy(),
        )

        out_path = os.path.join(
            OUTPUT_ROOT, f"{sample_name}_scale_{MODEL_LABEL}.h5ad"
        )
        ad_out.write_h5ad(out_path)
        print(f"    Saved: {out_path}")

    print("Finished SCVI training/inference.\n")


if __name__ == "__main__":
    main()
