from create_cellxgene_utils import (
    extract_metadata_fields,
    extract_raw_expression_from_tissue,
    qc_cellxgene,
    save_adata_in_chunks,
    VALID_TISSUE_GENERAL_IDS
)
import os

# Base output directory
output_root = "/hpc/group/xielab/xj58/SpaRestAtlas/npz_tissue_dataset_donor"
os.makedirs(output_root, exist_ok=True)

# Load filtered metadata
obs_df = extract_metadata_fields()

# Load valid Ensembl gene IDs
with open('/hpc/group/xielab/xj58/sparest_code/ensg_keys.txt', "r") as file:
    ensembl_ids = [line.strip() for line in file if line.strip()]

# Open census once outside the loop
import cellxgene_census
with cellxgene_census.open_soma(census_version="2025-01-30") as census:

    # Process each tissue separately
    for tissue_id in VALID_TISSUE_GENERAL_IDS:
        print(f"\n===== Processing tissue: {tissue_id} =====")
        tissue_obs = obs_df[obs_df["tissue_general_ontology_term_id"] == tissue_id]

        if tissue_obs.shape[0] < 5000:
            print(f"  -> Skipped: only {tissue_obs.shape[0]} cells for this tissue")
            continue

        tissue_output_dir = os.path.join(output_root, tissue_id)
        os.makedirs(tissue_output_dir, exist_ok=True)

        # Process all (donor, dataset) pairs for this tissue
        for (donor_id, dataset_id), sub_obs in tissue_obs.groupby(["donor_id", "dataset_id"]):
            print(f"  - Subset: donor={donor_id}, dataset={dataset_id}")

            # Directly get AnnData from utils
            adata = extract_raw_expression_from_tissue(tissue_id, donor_id, dataset_id, census)
            if adata is None or adata.n_obs == 0:
                print("    -> Skipped: no expression data")
                continue

            if "feature_id" not in adata.var.columns:
                print("    -> Skipped: missing 'feature_id'")
                continue

            # QC filtering
            adata = qc_cellxgene(adata, gene_ids_filter=ensembl_ids)
            if adata is None or adata.n_obs == 0:
                print("    -> Skipped after QC: no valid cells")
                continue

            # Save chunks
            save_adata_in_chunks(adata, donor_id=donor_id, dataset_id=dataset_id, output_dir=tissue_output_dir)