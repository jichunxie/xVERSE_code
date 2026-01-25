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

# -*- coding: utf-8 -*-
"""
Sample UMI counts from the trained SCVI model (multiple draws per cell).
"""

import os
import numpy as np
import torch
import scanpy as sc
import scvi
from scipy import sparse as sp
import sparse  # pydata sparse for GCXS handling


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
SAMPLE_PATH_MAP = dict(FILE_SAMPLE_LIST)

gene_ids_path = "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"

SCVI_MODEL_ROOT = "/hpc/group/xielab/xj58/xVerse_results/fig4/scvi_full"
MODEL_LABEL = "full"
MODEL_PATH = os.path.join(SCVI_MODEL_ROOT, f"scvi_{MODEL_LABEL}.pt")

N_UMI_SAMPLES = 3
SAMPLE_OUTPUT_ROOT = os.path.join(SCVI_MODEL_ROOT, f"umi_samples_{MODEL_LABEL}")
os.makedirs(SAMPLE_OUTPUT_ROOT, exist_ok=True)


# ================================================================
# Helper functions
# ================================================================

def load_allowed_gene_ids(path):
    """Read allowed gene_ids (newline separated)."""
    with open(path, "r") as f:
        gene_ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(gene_ids)} allowed gene_ids from {path}")
    return set(gene_ids)


def load_filtered_sample(sample_name, allowed_gene_ids):
    """Load and filter a single sample."""
    path = SAMPLE_PATH_MAP[sample_name]
    ad = sc.read_h5ad(path)

    if allowed_gene_ids is not None:
        if "gene_ids" not in ad.var:
            raise KeyError(f"'gene_ids' not found in var for {sample_name}")
        mask = ad.var["gene_ids"].astype(str).isin(allowed_gene_ids)
        ad = ad[:, mask].copy()
    return ad


def load_filtered_samples(allowed_gene_ids):
    """Load and filter every sample."""
    return {name: load_filtered_sample(name, allowed_gene_ids) for name in SAMPLE_PATH_MAP}


def build_full_training_adata(raw_dict):
    """Concatenate all cells from each sample to form training AnnData."""
    ad_list = []
    for sample_name, ad in raw_dict.items():
        ad_tmp = ad.copy()
        ad_tmp.obs["sample_batch"] = sample_name
        ad_list.append(ad_tmp)

    if not ad_list:
        raise ValueError("No samples loaded.")

    if len(ad_list) == 1:
        return ad_list[0]

    return ad_list[0].concatenate(
        ad_list[1:], index_unique=None, batch_key="concat_batch"
    )


def load_scvi_model(ad_train):
    """Load SCVI model weights using reference AnnData."""
    scvi.model.SCVI.setup_anndata(ad_train, batch_key="sample_batch")

    if os.path.isdir(MODEL_PATH):
        print(f"Loading SCVI model from directory {MODEL_PATH} ...")
        return scvi.model.SCVI.load(MODEL_PATH, adata=ad_train)

    print(f"Loading SCVI model state dict from {MODEL_PATH} ...")
    model = scvi.model.SCVI(ad_train)
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    return model


# ================================================================
# Main pipeline
# ================================================================

def main():
    allowed_gene_ids = load_allowed_gene_ids(gene_ids_path)
    raw_dict = load_filtered_samples(allowed_gene_ids)
    ad_train = build_full_training_adata(raw_dict)
    model = load_scvi_model(ad_train)

    for sample_name, _ in FILE_SAMPLE_LIST:
        print("\n" + "-" * 60)
        print(f"Sampling UMI counts for {sample_name} ...")
        ad_raw = raw_dict[sample_name].copy()
        ad_raw.obs["sample_batch"] = sample_name

        scvi.model.SCVI.setup_anndata(ad_raw, batch_key="sample_batch")

        gcxs_samples = model.posterior_predictive_sample(
            adata=ad_raw, n_samples=N_UMI_SAMPLES
        )
        sample_dir = os.path.join(SAMPLE_OUTPUT_ROOT, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        if gcxs_samples.ndim == 2:
            gcxs_samples = gcxs_samples[:, :, None]

        for idx in range(gcxs_samples.shape[2]):
            slice_gcxs = gcxs_samples[:, :, idx]
            coo = slice_gcxs.tocoo()
            rows, cols = coo.coords
            X = sp.csr_matrix((coo.data, (rows, cols)), shape=coo.shape)
            ad_out = sc.AnnData(
                X=X,
                obs=ad_raw.obs.copy(),
                var=ad_raw.var.copy(),
            )
            out_path = os.path.join(
                sample_dir,
                f"{sample_name}_umi_sample{idx + 1}_{MODEL_LABEL}.h5ad",
            )
            ad_out.write_h5ad(out_path)
            print(f"    Saved sample {idx + 1}: {out_path}")

    print("\nAll UMI samples generated.")


if __name__ == "__main__":
    main()
