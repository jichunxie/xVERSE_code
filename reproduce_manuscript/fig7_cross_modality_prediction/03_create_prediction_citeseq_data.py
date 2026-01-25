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

#!/usr/bin/env python
"""Build train/val/test datasets (with Poisson augmentation) for RNA→ADT prediction."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DATA_ROOT = Path("/hpc/group/xielab/xj58/xVerse_results/fig7/citeseq_data/GSE291290/split_by_sample")
POISSON_ROOT = Path("/hpc/group/xielab/xj58/xVerse_results/fig7/ft_model_GSE291290/inference_finetuned")
OUTPUT_DIR = Path("/hpc/group/xielab/xj58/xVerse_results/fig7/crossmodal_datasets_GSE291290")

# Train/val split within CONTROL cohort; NGD and CAV each get their own held-out test set
TRAIN_SAMPLES = [
    "CONTROL1_1",
    "CONTROL1_2",
    "CONTROL2_1",
    "CONTROL2_2",
    "CONTROL3_1",
    "CONTROL3_2",
    "CONTROL4_1",
    "CONTROL4_2",
]
VAL_SAMPLES = [
    "CONTROL5_1",
    "CONTROL5_2",
    "CONTROL6_1",
    "CONTROL6_2",
]
TEST_NGD_SAMPLES = ["NGD1", "NGD2", "NGD3", "NGD4", "NGD5", "NGD6"]
TEST_CAV_SAMPLES = ["CAV1", "CAV2", "CAV3", "CAV4", "CAV5", "CAV6"]


def ensure_gene_ids(adata: sc.AnnData) -> sc.AnnData:
    if "gene_ids" not in adata.var.columns:
        if "gene_id" in adata.var.columns:
            adata.var = adata.var.rename(columns={"gene_id": "gene_ids"})
        else:
            raise KeyError("gene_ids column missing.")
    adata.var["gene_ids"] = adata.var["gene_ids"].astype(str)
    return adata


def read_rna(sample: str) -> sc.AnnData:
    path = DATA_ROOT / f"{sample}_rna.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    return ensure_gene_ids(sc.read_h5ad(path))


def read_adt(sample: str) -> sc.AnnData:
    path = DATA_ROOT / f"{sample}_adt.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    return sc.read_h5ad(path)


def read_vdj(sample: str) -> sc.AnnData:
    path = DATA_ROOT / f"{sample}_vdj.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    return sc.read_h5ad(path)


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def get_adt_matrix(adata: sc.AnnData) -> np.ndarray:
    """Return raw ADT counts when available, otherwise X."""
    mat = adata.raw.X if adata.raw is not None else adata.X
    return dense(mat)


def get_gene_intersection(samples: List[str]) -> List[str]:
    shared = None
    for sample in samples:
        genes = set(read_rna(sample).var["gene_ids"].astype(str))
        shared = genes if shared is None else shared & genes
    if not shared:
        raise ValueError("No overlapping genes across batches.")
    first = read_rna(samples[0]).var["gene_ids"].astype(str).tolist()
    return [gid for gid in first if gid in shared]


def align_rna(adata: sc.AnnData, gene_order: List[str]) -> np.ndarray:
    gene_to_idx = {gid: idx for idx, gid in enumerate(adata.var["gene_ids"].astype(str))}
    cols = [gene_to_idx[gid] for gid in gene_order]
    X = dense(adata.X[:, cols])
    return X.astype(np.float32)


def create_obs(batch: str, obs_names: List[str], split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch": batch,
            "original_cell": obs_names,
            "split": split,
        },
        index=pd.Index(obs_names, name="cell_id"),
    )


def create_var(gene_order: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"gene_ids": gene_order}, index=pd.Index(gene_order, name="gene_id"))


def normalize_adt(adt_matrix: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    """
    Apply per-protein clipping at the given quantile, then log1p and library-size normalization.
    """
    q = np.quantile(adt_matrix, quantile, axis=0)
    adt_clipped = np.minimum(adt_matrix, q)
    adt_log = np.log1p(adt_clipped)
    lib = adt_log.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return adt_log / lib


def merge_vdj_obs(base_obs: pd.DataFrame, vdj_obs: pd.DataFrame) -> pd.DataFrame:
    """Attach VDJ obs columns (if available) to the existing obs table."""
    vdj_obs = vdj_obs.reindex(base_obs.index)
    overlap = [col for col in vdj_obs.columns if col in base_obs.columns]
    if overlap:
        vdj_obs = vdj_obs.drop(columns=overlap)
    return base_obs.join(vdj_obs)


def parse_poisson_indices(obs_names: List[str], n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    base_idx = []
    copy_ids = []
    for name in obs_names:
        parts = str(name).split("_", 1)
        base = parts[0]
        copy = parts[1] if len(parts) > 1 else "1"
        try:
            idx = int(base) - 1
            copy_val = int(copy)
        except ValueError:
            raise ValueError(f"Unexpected Poisson obs format: {name}")
        if idx < 0 or idx >= n_cells:
            raise IndexError(f"Poisson cell index {idx} out of range (n={n_cells})")
        base_idx.append(idx)
        copy_ids.append(copy_val)
    return np.asarray(base_idx, dtype=int), np.asarray(copy_ids, dtype=int)


def summarize_random_proteins(adt_matrix: np.ndarray, protein_names: List[str], n: int = 10):
    protein_names = np.asarray(protein_names)
    if adt_matrix.size == 0 or protein_names.size == 0:
        print("[Summary] No proteins to summarize.")
        return
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(protein_names.size), size=min(n, protein_names.size), replace=False)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"[Summary] Random {len(idx)} proteins (raw ADT) quantiles:")
    for i in idx:
        vals = adt_matrix[:, i]
        qs = np.quantile(vals, quantiles)
        q_str = ", ".join(f"q{int(q*100):02d}={v:.4f}" for q, v in zip(quantiles, qs))
        print(f"  - {protein_names[i]}: {q_str}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = TRAIN_SAMPLES + VAL_SAMPLES + TEST_NGD_SAMPLES + TEST_CAV_SAMPLES
    gene_order = get_gene_intersection(all_samples)
    print(f"[Info] Shared genes (RNA batches): {len(gene_order)}")

    poisson_shared = set(gene_order)
    for sample in TRAIN_SAMPLES:
        sample_name = f"{sample}_rna"
        poisson_path = POISSON_ROOT / sample_name / f"{sample_name}_estimated_expr_poisson.h5ad"
        if not poisson_path.exists():
            print(f"[Warn] Missing Poisson file for {sample}; skip.")
            continue
        poisson_genes = set(ensure_gene_ids(sc.read_h5ad(poisson_path)).var["gene_ids"].astype(str))
        poisson_shared &= poisson_genes
    if poisson_shared:
        gene_order = [gid for gid in gene_order if gid in poisson_shared]
    var_df = create_var(gene_order)
    print(f"[Info] Shared genes (including Poisson): {len(gene_order)}")

    X_train_list, Y_train_list, obs_frames = [], [], []
    adt_cache: Dict[str, Dict[str, np.ndarray]] = {}
    protein_names = None

    for sample in TRAIN_SAMPLES:
        print(f"[Load] Training sample {sample}")
        rna = read_rna(sample)
        adt = read_adt(sample)
        vdj = read_vdj(sample)
        if not np.array_equal(rna.obs_names, adt.obs_names):
            raise ValueError(f"RNA/ADT obs mismatch for {sample}")
        if not np.array_equal(rna.obs_names, vdj.obs_names):
            raise ValueError(f"RNA/VDJ obs mismatch for {sample}")

        adt_full = get_adt_matrix(adt).astype(np.float32)
        keep_mask = (adt_full.sum(axis=1) <= 10000)
        if not keep_mask.all():
            print(f"  - Filtered out {(~keep_mask).sum()} cells with ADT sum > 10000.")
        rna = rna[keep_mask]
        adt_full = adt_full[keep_mask]
        vdj = vdj[keep_mask]
        adt_norm = normalize_adt(adt_full, quantile=0.90)
        X = align_rna(rna, gene_order)
        Y = adt_norm
        X_train_list.append(X)
        Y_train_list.append(Y)
        base_obs = create_obs(sample, rna.obs_names.tolist(), "train")
        obs_frames.append(merge_vdj_obs(base_obs, vdj.obs))
        if protein_names is None:
            protein_names = adt.var_names.astype(str).tolist()
        elif not np.array_equal(adt.var_names.astype(str), np.asarray(protein_names)):
            raise ValueError("Protein feature sets differ across batches.")
        orig_to_keep = np.full(keep_mask.size, -1, dtype=int)
        orig_to_keep[np.where(keep_mask)[0]] = np.arange(keep_mask.sum())
        adt_cache[sample] = {
            "protein": Y,
            "obs_names": rna.obs_names.to_numpy(copy=True),
            "vdj_obs": vdj.obs.copy(),
            "orig_to_keep": orig_to_keep,
        }

    # Filter proteins with low total counts in training set
    train_protein_sum = np.sum(np.concatenate(Y_train_list, axis=0), axis=0)
    protein_keep_mask = train_protein_sum >= 100
    if not protein_keep_mask.any():
        raise ValueError("No proteins left after filtering sum<100 in training set.")
    if not protein_keep_mask.all():
        print(f"[Filter] Dropped {(~protein_keep_mask).sum()} proteins with train sum < 100.")
    protein_names = [p for p, keep in zip(protein_names, protein_keep_mask) if keep]
    Y_train_list = [y[:, protein_keep_mask] for y in Y_train_list]
    for sample in adt_cache:
        adt_cache[sample]["protein"] = adt_cache[sample]["protein"][:, protein_keep_mask]

    # Compute z-score stats on training proteins and apply to all splits
    train_concat = np.concatenate(Y_train_list, axis=0)
    protein_mean = train_concat.mean(axis=0)
    protein_std = train_concat.std(axis=0)
    protein_std[protein_std < 1e-6] = 1e-6

    def zscore_protein(arr: np.ndarray) -> np.ndarray:
        return (arr - protein_mean) / protein_std

    Y_train_list = [zscore_protein(y) for y in Y_train_list]
    for sample in adt_cache:
        adt_cache[sample]["protein"] = zscore_protein(adt_cache[sample]["protein"])

    adata_train = sc.AnnData(
        X=np.concatenate(X_train_list, axis=0),
        obs=pd.concat(obs_frames, axis=0),
        var=var_df.copy(),
    )
    adata_train.obsm["protein"] = np.concatenate(Y_train_list, axis=0)
    adata_train.uns["protein_feature_names"] = protein_names
    adata_train.uns["protein_zscore_mean"] = protein_mean
    adata_train.uns["protein_zscore_std"] = protein_std
    adata_train.write_h5ad(OUTPUT_DIR / "original_train.h5ad")
    print(f"[IO] Saved original_train with {adata_train.n_obs} cells.")
    summarize_random_proteins(adata_train.obsm["protein"], protein_names)

    # Validation batches
    X_val, Y_val, obs_val = [], [], []
    for sample in VAL_SAMPLES:
        print(f"[Load] Validation sample {sample}")
        rna = read_rna(sample)
        adt = read_adt(sample)
        vdj = read_vdj(sample)
        if not np.array_equal(rna.obs_names, adt.obs_names):
            raise ValueError(f"RNA/ADT obs mismatch for {sample}")
        if not np.array_equal(rna.obs_names, vdj.obs_names):
            raise ValueError(f"RNA/VDJ obs mismatch for {sample}")
        adt_full = get_adt_matrix(adt).astype(np.float32)
        keep_mask = (adt_full.sum(axis=1) <= 10000)
        if not keep_mask.all():
            print(f"  - Filtered out {(~keep_mask).sum()} cells with ADT sum > 10000.")
        rna = rna[keep_mask]
        adt_full = adt_full[keep_mask]
        vdj = vdj[keep_mask]
        adt_norm = normalize_adt(adt_full, quantile=0.90)
        X_val.append(align_rna(rna, gene_order))
        adt_z = (adt_norm[:, protein_keep_mask] - protein_mean) / protein_std
        Y_val.append(adt_z.astype(np.float32))
        base_obs = create_obs(sample, rna.obs_names.tolist(), "val")
        obs_val.append(merge_vdj_obs(base_obs, vdj.obs))
    adata_val = sc.AnnData(
        X=np.concatenate(X_val, axis=0),
        obs=pd.concat(obs_val, axis=0),
        var=var_df.copy(),
    )
    adata_val.obsm["protein"] = np.concatenate(Y_val, axis=0)
    adata_val.uns["protein_feature_names"] = protein_names
    adata_val.write_h5ad(OUTPUT_DIR / "original_val.h5ad")
    print(f"[IO] Saved original_val with {adata_val.n_obs} cells.")

    # Test batches: NGD
    X_test_ngd, Y_test_ngd, obs_test_ngd = [], [], []
    for sample in TEST_NGD_SAMPLES:
        print(f"[Load] Test NGD sample {sample}")
        rna = read_rna(sample)
        adt = read_adt(sample)
        vdj = read_vdj(sample)
        if not np.array_equal(rna.obs_names, adt.obs_names):
            raise ValueError(f"RNA/ADT obs mismatch for {sample}")
        if not np.array_equal(rna.obs_names, vdj.obs_names):
            raise ValueError(f"RNA/VDJ obs mismatch for {sample}")
        adt_full = get_adt_matrix(adt).astype(np.float32)
        keep_mask = (adt_full.sum(axis=1) <= 10000)
        if not keep_mask.all():
            print(f"  - Filtered out {(~keep_mask).sum()} cells with ADT sum > 10000.")
        rna = rna[keep_mask]
        adt_full = adt_full[keep_mask]
        vdj = vdj[keep_mask]
        adt_norm = normalize_adt(adt_full, quantile=0.90)
        X_test_ngd.append(align_rna(rna, gene_order))
        adt_z = (adt_norm[:, protein_keep_mask] - protein_mean) / protein_std
        Y_test_ngd.append(adt_z.astype(np.float32))
        base_obs = create_obs(sample, rna.obs_names.tolist(), "test_ngd")
        obs_test_ngd.append(merge_vdj_obs(base_obs, vdj.obs))
    adata_test_ngd = sc.AnnData(
        X=np.concatenate(X_test_ngd, axis=0),
        obs=pd.concat(obs_test_ngd, axis=0),
        var=var_df.copy(),
    )
    adata_test_ngd.obsm["protein"] = np.concatenate(Y_test_ngd, axis=0)
    adata_test_ngd.uns["protein_feature_names"] = protein_names
    adata_test_ngd.write_h5ad(OUTPUT_DIR / "original_test_ngd.h5ad")
    print(f"[IO] Saved original_test_ngd with {adata_test_ngd.n_obs} cells.")

    # Test batches: CAV
    X_test_cav, Y_test_cav, obs_test_cav = [], [], []
    for sample in TEST_CAV_SAMPLES:
        print(f"[Load] Test CAV sample {sample}")
        rna = read_rna(sample)
        adt = read_adt(sample)
        vdj = read_vdj(sample)
        if not np.array_equal(rna.obs_names, adt.obs_names):
            raise ValueError(f"RNA/ADT obs mismatch for {sample}")
        if not np.array_equal(rna.obs_names, vdj.obs_names):
            raise ValueError(f"RNA/VDJ obs mismatch for {sample}")
        adt_full = get_adt_matrix(adt).astype(np.float32)
        keep_mask = (adt_full.sum(axis=1) <= 10000)
        if not keep_mask.all():
            print(f"  - Filtered out {(~keep_mask).sum()} cells with ADT sum > 10000.")
        rna = rna[keep_mask]
        adt_full = adt_full[keep_mask]
        vdj = vdj[keep_mask]
        adt_norm = normalize_adt(adt_full, quantile=0.90)
        X_test_cav.append(align_rna(rna, gene_order))
        adt_z = (adt_norm[:, protein_keep_mask] - protein_mean) / protein_std
        Y_test_cav.append(adt_z.astype(np.float32))
        base_obs = create_obs(sample, rna.obs_names.tolist(), "test_cav")
        obs_test_cav.append(merge_vdj_obs(base_obs, vdj.obs))
    adata_test_cav = sc.AnnData(
        X=np.concatenate(X_test_cav, axis=0),
        obs=pd.concat(obs_test_cav, axis=0),
        var=var_df.copy(),
    )
    adata_test_cav.obsm["protein"] = np.concatenate(Y_test_cav, axis=0)
    adata_test_cav.uns["protein_feature_names"] = protein_names
    adata_test_cav.write_h5ad(OUTPUT_DIR / "original_test_cav.h5ad")
    print(f"[IO] Saved original_test_cav with {adata_test_cav.n_obs} cells.")

    # Poisson augmentation
    poisson_X, poisson_Y, poisson_obs = [], [], []
    for sample in TRAIN_SAMPLES:
        sample_name = f"{sample}_rna"
        path = POISSON_ROOT / sample_name / f"{sample_name}_estimated_expr_poisson.h5ad"
        if not path.exists():
            print(f"[Warn] Missing Poisson file for {sample}; skip.")
            continue
        print(f"[Load] Poisson replicates {path}")
        adata = ensure_gene_ids(sc.read_h5ad(path))
        Xp = align_rna(adata, gene_order)
        base_idx_orig, copy_ids = parse_poisson_indices(
            adata.obs_names.tolist(),
            adt_cache[sample]["orig_to_keep"].shape[0],
        )
        mapped_idx = adt_cache[sample]["orig_to_keep"][base_idx_orig]
        valid_mask = mapped_idx >= 0
        if not valid_mask.any():
            print(f"[Warn] All Poisson replicates for {sample} correspond to filtered cells; skip.")
            continue
        Xp = Xp[valid_mask]
        copy_ids = copy_ids[valid_mask]
        mapped_idx = mapped_idx[valid_mask]
        proteins = adt_cache[sample]["protein"][mapped_idx]
        original_cells = adt_cache[sample]["obs_names"][mapped_idx]
        base_obs = pd.DataFrame(
            {
                "batch": sample,
                "original_cell": original_cells,
                "copy_id": copy_ids,
                "split": "train",
            },
            index=pd.Index([f"{sample}:{name}" for name in np.asarray(adata.obs_names)[valid_mask]], name="cell_id"),
        )
        vdj_obs_sub = adt_cache[sample]["vdj_obs"].iloc[mapped_idx].copy()
        vdj_obs_sub.index = base_obs.index
        obs = merge_vdj_obs(base_obs, vdj_obs_sub)
        poisson_X.append(Xp)
        poisson_Y.append(proteins.astype(np.float32))
        poisson_obs.append(obs)

    if poisson_X:
        adata_poisson = sc.AnnData(
            X=np.concatenate(poisson_X, axis=0),
            obs=pd.concat(poisson_obs, axis=0),
            var=var_df.copy(),
        )
        adata_poisson.obsm["protein"] = np.concatenate(poisson_Y, axis=0)
        adata_poisson.uns["protein_feature_names"] = protein_names
        adata_poisson.write_h5ad(OUTPUT_DIR / "poisson_train.h5ad")
        print(f"[IO] Saved poisson_train with {adata_poisson.n_obs} replicates.")
    else:
        print("[Warn] No Poisson replicates were found; poisson_train not created.")


if __name__ == "__main__":
    main()
