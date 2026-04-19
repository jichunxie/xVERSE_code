import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from main_energy.utils_model import (
    build_cell_type_to_index,
    build_pair_to_sample_id_and_paths,
    load_gene_ids,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _read_sparse_npz_shape(matrix_path: str) -> Tuple[int, int]:
    with np.load(matrix_path, allow_pickle=False) as npz:
        shape = npz["shape"]
        return int(shape[0]), int(shape[1])


def _choose_values_dtype(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    if arr.size == 0:
        return arr.astype(np.uint16), "uint16"
    if np.issubdtype(arr.dtype, np.integer):
        mx = int(arr.max())
        mn = int(arr.min())
        if mn >= 0 and mx <= np.iinfo(np.uint16).max:
            return arr.astype(np.uint16, copy=False), "uint16"
        if mn >= 0 and mx <= np.iinfo(np.uint32).max:
            return arr.astype(np.uint32, copy=False), "uint32"
        return arr.astype(np.float32, copy=False), "float32"

    rounded = np.rint(arr)
    if np.allclose(arr, rounded):
        iarr = rounded.astype(np.int64, copy=False)
        mx = int(iarr.max()) if iarr.size > 0 else 0
        mn = int(iarr.min()) if iarr.size > 0 else 0
        if mn >= 0 and mx <= np.iinfo(np.uint16).max:
            return iarr.astype(np.uint16, copy=False), "uint16"
        if mn >= 0 and mx <= np.iinfo(np.uint32).max:
            return iarr.astype(np.uint32, copy=False), "uint32"
    return arr.astype(np.float32, copy=False), "float32"


def _row_sum_max_from_csr(indptr: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_rows = indptr.size - 1
    row_sum = np.zeros(n_rows, dtype=np.float64)
    row_max = np.zeros(n_rows, dtype=np.float64)
    if data.size == 0:
        return row_sum, row_max

    non_empty = indptr[1:] > indptr[:-1]
    starts = indptr[:-1][non_empty]
    row_sum[non_empty] = np.add.reduceat(data.astype(np.float64, copy=False), starts)
    row_max[non_empty] = np.maximum.reduceat(data.astype(np.float64, copy=False), starts)
    return row_sum, row_max


def _build_block_mapping(global_gene_ids: List[str], block_gene_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gene_to_idx = {g: i for i, g in enumerate(block_gene_ids)}
    gene_idx_array = np.full(len(global_gene_ids), -1, dtype=np.int32)
    for global_idx, g in enumerate(global_gene_ids):
        if g in gene_to_idx:
            gene_idx_array[global_idx] = gene_to_idx[g]

    observed_global_idx = np.where(gene_idx_array >= 0)[0].astype(np.int32, copy=False)
    observed_local_idx = gene_idx_array[observed_global_idx].astype(np.int32, copy=False)
    local_to_rel = np.full(len(block_gene_ids), -1, dtype=np.int32)
    if observed_local_idx.size > 0:
        local_to_rel[observed_local_idx] = np.arange(observed_local_idx.size, dtype=np.int32)
    return observed_global_idx, local_to_rel


def _compile_one_pair(
    shard_id: int,
    split: str,
    pair: Tuple[str, str],
    global_gene_ids: List[str],
    pair_to_sample_id: Dict[Tuple[str, str], int],
    pair_to_tissue_id: Dict[Tuple[str, str], int],
    cell_type_to_index: Optional[Dict[str, int]],
    out_dir: Path,
    filter_bad_cells: bool,
    max_count_threshold: float,
    sum_count_threshold: float,
) -> Optional[Dict]:
    matrix_path, meta_path = pair
    sample_id = int(pair_to_sample_id.get(pair, -1))
    tissue_id = int(pair_to_tissue_id.get(pair, -1))
    if sample_id < 0:
        raise ValueError(f"sample_id not found for pair={pair}")

    X = sp.load_npz(matrix_path).tocsr()
    meta = np.load(meta_path, allow_pickle=True)
    block_gene_ids = np.asarray(meta["gene_ids"])
    cell_type_arr = np.asarray(meta["cell_type_ontology_term_id"]) if "cell_type_ontology_term_id" in meta else None
    if cell_type_arr is None:
        cell_type_arr = np.array(["unknown"] * X.shape[0], dtype=object)

    observed_global_idx, local_to_rel = _build_block_mapping(global_gene_ids, block_gene_ids)

    n_rows = X.shape[0]
    if filter_bad_cells:
        row_sum, row_max = _row_sum_max_from_csr(X.indptr, X.data)
        valid_rows = np.where((row_max <= max_count_threshold) & (row_sum <= sum_count_threshold))[0]
        if valid_rows.size != n_rows:
            X = X[valid_rows].tocsr()
            cell_type_arr = cell_type_arr[valid_rows]
        dropped_rows = int(n_rows - valid_rows.size)
    else:
        dropped_rows = 0

    rel_all = local_to_rel[X.indices]
    valid_nz = rel_all >= 0
    indices = rel_all[valid_nz].astype(np.uint32, copy=False)
    values_raw = X.data[valid_nz]
    values, values_dtype = _choose_values_dtype(values_raw)

    # Rebuild indptr after nz filtering.
    csum = np.concatenate([[0], np.cumsum(valid_nz.astype(np.uint8), dtype=np.uint64)])
    indptr = csum[X.indptr].astype(np.uint64, copy=False)

    n_kept = X.shape[0]
    if n_kept == 0:
        return None

    if cell_type_to_index is not None:
        celltype_idx = np.asarray([cell_type_to_index.get(str(x), -1) for x in cell_type_arr], dtype=np.int32)
    else:
        celltype_idx = np.full((n_kept,), -1, dtype=np.int32)

    # mask dictionary for this shard
    # In most blocks this is a single panel mask; still stored as dictionary format.
    mask_offsets = np.array([0, observed_global_idx.size], dtype=np.uint64)
    mask_genes = observed_global_idx.astype(np.uint32, copy=False)
    cell_mask_id = np.zeros((n_kept,), dtype=np.uint32)

    shard_dir = out_dir / f"shard_{shard_id:06d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "cell_mask_id.npy", cell_mask_id, allow_pickle=False)
    np.save(shard_dir / "cell_celltype_id.npy", celltype_idx, allow_pickle=False)
    np.save(shard_dir / "nnz_indptr.npy", indptr, allow_pickle=False)
    np.save(shard_dir / "nnz_indices.npy", indices, allow_pickle=False)
    np.save(shard_dir / "nnz_values.npy", values, allow_pickle=False)
    np.save(shard_dir / "mask_offsets.npy", mask_offsets, allow_pickle=False)
    np.save(shard_dir / "mask_genes.npy", mask_genes, allow_pickle=False)

    rec = {
        "shard_id": shard_id,
        "split": split,
        "shard_dir": str(shard_dir),
        "matrix_path": matrix_path,
        "meta_path": meta_path,
        "sample_id": sample_id,
        "tissue_id": tissue_id,
        "rows_kept": int(n_kept),
        "rows_dropped": int(dropped_rows),
        "nnz": int(indices.size),
        "values_dtype": values_dtype,
        "mask_count": 1,
        "mask_gene_count": int(mask_genes.size),
    }
    return rec


def parse_args():
    p = argparse.ArgumentParser(description="Compile NPZ dataset to mask-dictionary + CSR shard format.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--gene-ids-path", default=None)
    p.add_argument("--cell-type-csv", default=None)
    p.add_argument("--use-tissue", default=None)
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--filter-bad-cells", action="store_true")
    p.add_argument("--max-count-threshold", type=float, default=1000.0)
    p.add_argument("--sum-count-threshold", type=float, default=200000.0)
    return p.parse_args()


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(max(0, n))
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    summary_csv = args.summary_csv or str(data_root / "pantissue_full_updated.csv")
    gene_ids_path = args.gene_ids_path or str(data_root / "ensg_keys_high_quality.txt")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_to_sample_id, train_pairs, val_pairs, pair_to_tissue_id, _ = build_pair_to_sample_id_and_paths(
        summary_csv,
        use_tissue=args.use_tissue,
    )
    global_gene_ids = load_gene_ids(gene_ids_path)
    cell_type_to_index = build_cell_type_to_index(args.cell_type_csv) if args.cell_type_csv else None

    pairs_with_split = []
    if args.split in ("train", "both"):
        pairs_with_split.extend([("train", p) for p in train_pairs])
    if args.split in ("val", "both"):
        pairs_with_split.extend([("val", p) for p in val_pairs])

    manifest_rows = []
    cursor = 0
    print(f"[CompileMaskDict] total_pairs={len(pairs_with_split)}, genes={len(global_gene_ids)}")
    uniq_files = set()
    for _, (m, o) in pairs_with_split:
        uniq_files.add(m)
        uniq_files.add(o)
    source_bytes = 0
    for p in uniq_files:
        try:
            source_bytes += Path(p).stat().st_size
        except FileNotFoundError:
            pass

    iter_pairs = enumerate(pairs_with_split)
    if tqdm is not None:
        iter_pairs = enumerate(tqdm(pairs_with_split, total=len(pairs_with_split), desc="Compiling shards", unit="shard"))

    for shard_id, (split, pair) in iter_pairs:
        rec = _compile_one_pair(
            shard_id=shard_id,
            split=split,
            pair=pair,
            global_gene_ids=global_gene_ids,
            pair_to_sample_id=pair_to_sample_id,
            pair_to_tissue_id=pair_to_tissue_id,
            cell_type_to_index=cell_type_to_index,
            out_dir=out_dir,
            filter_bad_cells=args.filter_bad_cells,
            max_count_threshold=args.max_count_threshold,
            sum_count_threshold=args.sum_count_threshold,
        )
        if rec is None:
            continue
        rec["global_start"] = cursor
        rec["global_end"] = cursor + rec["rows_kept"]
        cursor = rec["global_end"]
        manifest_rows.append(rec)
        if tqdm is None and (shard_id + 1) % 100 == 0:
            print(f"[CompileMaskDict] processed {shard_id + 1}/{len(pairs_with_split)} shards, rows={cursor}")

    manifest = {
        "format": "maskdict_csr_v1",
        "total_rows": cursor,
        "num_shards": len(manifest_rows),
        "filter_bad_cells": bool(args.filter_bad_cells),
        "max_count_threshold": float(args.max_count_threshold),
        "sum_count_threshold": float(args.sum_count_threshold),
        "global_gene_ids_path": str(gene_ids_path),
        "rows": manifest_rows,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out_dir / "global_gene_ids.txt", "w") as f:
        for g in global_gene_ids:
            f.write(f"{g}\n")

    compiled_bytes = _dir_size_bytes(out_dir)
    ratio = (compiled_bytes / source_bytes) if source_bytes > 0 else float("nan")
    print(f"[CompileMaskDict] Done. total_rows={cursor}, shards={len(manifest_rows)}, out={out_dir}")
    print(
        f"[CompileMaskDict] Size: compiled={_human_bytes(compiled_bytes)}, "
        f"source_inputs={_human_bytes(source_bytes)}, ratio={ratio:.3f}x"
    )


if __name__ == "__main__":
    main()
