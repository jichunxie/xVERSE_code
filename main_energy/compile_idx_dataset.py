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


def _safe_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _choose_value_dtype(x: np.ndarray) -> Tuple[np.ndarray, str]:
    if x.size == 0:
        return x.astype(np.uint16), "uint16"
    if np.issubdtype(x.dtype, np.integer):
        xmax = int(x.max())
        if xmax <= np.iinfo(np.uint16).max:
            return x.astype(np.uint16, copy=False), "uint16"
        if xmax <= np.iinfo(np.uint32).max:
            return x.astype(np.uint32, copy=False), "uint32"
        return x.astype(np.uint64, copy=False), "uint64"

    rounded = np.rint(x)
    if np.allclose(x, rounded):
        xi = rounded.astype(np.int64, copy=False)
        xmax = int(xi.max()) if xi.size > 0 else 0
        if xmax <= np.iinfo(np.uint16).max and xi.min() >= 0:
            return xi.astype(np.uint16, copy=False), "uint16"
        if xmax <= np.iinfo(np.uint32).max and xi.min() >= 0:
            return xi.astype(np.uint32, copy=False), "uint32"
    return x.astype(np.float32, copy=False), "float32"


def _build_local_to_rel(global_gene_ids: List[str], gene_ids_available: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gene_to_idx = {g: i for i, g in enumerate(gene_ids_available)}
    gene_idx_array = np.full(len(global_gene_ids), -1, dtype=np.int32)
    for global_idx, g in enumerate(global_gene_ids):
        if g in gene_to_idx:
            gene_idx_array[global_idx] = gene_to_idx[g]
    observed_global_idx = np.where(gene_idx_array >= 0)[0].astype(np.int32, copy=False)
    observed_local_idx = gene_idx_array[observed_global_idx].astype(np.int32, copy=False)
    local_to_rel = np.full(len(gene_ids_available), -1, dtype=np.int32)
    if observed_local_idx.size > 0:
        local_to_rel[observed_local_idx] = np.arange(observed_local_idx.size, dtype=np.int32)
    return observed_global_idx, local_to_rel


def _compile_one_pair(
    shard_id: int,
    pair: Tuple[str, str],
    global_gene_ids: List[str],
    pair_to_sample_id: Dict[Tuple[str, str], int],
    pair_to_tissue_id: Dict[Tuple[str, str], int],
    celltype_map: Optional[Dict[str, int]],
    out_dir: Path,
    filter_bad_cells: bool,
    max_count_threshold: float,
    sum_count_threshold: float,
) -> Optional[Dict]:
    matrix_path, meta_path = pair
    sample_id = int(pair_to_sample_id.get(pair, -1))
    tissue_id = int(pair_to_tissue_id.get(pair, -1))
    if sample_id < 0:
        raise ValueError(f"sample_id missing for pair={pair}")

    X = sp.load_npz(matrix_path).tocsr()
    meta = np.load(meta_path, allow_pickle=True)
    gene_ids_available = np.asarray(meta["gene_ids"])
    celltype_arr = np.asarray(meta["cell_type_ontology_term_id"]) if "cell_type_ontology_term_id" in meta else None
    if celltype_arr is None:
        celltype_arr = np.array(["unknown"] * X.shape[0], dtype=object)

    observed_global_idx, local_to_rel = _build_local_to_rel(global_gene_ids, gene_ids_available)

    indptr = [0]
    indices_parts = []
    values_parts = []
    row_celltype = []
    row_local_idx = []
    kept_rows = 0
    dropped_rows = 0

    for i in range(X.shape[0]):
        st = int(X.indptr[i])
        ed = int(X.indptr[i + 1])
        row_vals = X.data[st:ed]

        if filter_bad_cells:
            if row_vals.size > 0:
                if float(row_vals.max()) > max_count_threshold or float(row_vals.sum()) > sum_count_threshold:
                    dropped_rows += 1
                    continue

        row_cols = X.indices[st:ed]
        rel = local_to_rel[row_cols]
        valid = rel >= 0
        if np.any(valid):
            rel = rel[valid].astype(np.int32, copy=False)
            vals = row_vals[valid]
        else:
            rel = np.empty((0,), dtype=np.int32)
            vals = np.empty((0,), dtype=X.data.dtype)

        indices_parts.append(rel)
        values_parts.append(vals)
        indptr.append(indptr[-1] + rel.size)

        ct_raw = str(celltype_arr[i])
        if celltype_map is None:
            ct_idx = -1
        else:
            ct_idx = int(celltype_map.get(ct_raw, -1))
        row_celltype.append(ct_idx)
        row_local_idx.append(i)
        kept_rows += 1

    if kept_rows == 0:
        return None

    shard_dir = out_dir / f"shard_{shard_id:06d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    indptr_arr = np.asarray(indptr, dtype=np.uint64)
    indices_arr = np.concatenate(indices_parts).astype(np.uint32, copy=False) if indices_parts else np.empty((0,), dtype=np.uint32)
    values_raw = np.concatenate(values_parts) if values_parts else np.empty((0,), dtype=np.float32)
    values_arr, values_dtype = _choose_value_dtype(values_raw)
    celltype_idx_arr = np.asarray(row_celltype, dtype=np.int32)
    row_local_idx_arr = np.asarray(row_local_idx, dtype=np.int32)

    np.save(shard_dir / "indptr.npy", indptr_arr, allow_pickle=False)
    np.save(shard_dir / "indices.npy", indices_arr, allow_pickle=False)
    np.save(shard_dir / "values.npy", values_arr, allow_pickle=False)
    np.save(shard_dir / "observed_global_idx.npy", observed_global_idx, allow_pickle=False)
    np.save(shard_dir / "celltype_idx.npy", celltype_idx_arr, allow_pickle=False)
    np.save(shard_dir / "row_local_idx.npy", row_local_idx_arr, allow_pickle=False)

    return {
        "shard_id": shard_id,
        "shard_dir": str(shard_dir),
        "matrix_path": matrix_path,
        "meta_path": meta_path,
        "sample_id": sample_id,
        "tissue_id": tissue_id,
        "rows_kept": kept_rows,
        "rows_dropped": dropped_rows,
        "nnz": int(indices_arr.size),
        "values_dtype": values_dtype,
        "observed_genes": int(observed_global_idx.size),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compile current xVERSE NPZ dataset into idx-friendly shard format.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--gene-ids-path", default=None)
    p.add_argument("--cell-type-csv", default=None, help="Optional mapping CSV for cell type to index.")
    p.add_argument("--use-tissue", default=None)
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--filter-bad-cells", action="store_true")
    p.add_argument("--max-count-threshold", type=float, default=1000.0)
    p.add_argument("--sum-count-threshold", type=float, default=200000.0)
    return p.parse_args()


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
    celltype_map = build_cell_type_to_index(args.cell_type_csv) if args.cell_type_csv else None

    selected_pairs = []
    split_name = []
    if args.split in ("train", "both"):
        selected_pairs.extend(train_pairs)
        split_name.extend(["train"] * len(train_pairs))
    if args.split in ("val", "both"):
        selected_pairs.extend(val_pairs)
        split_name.extend(["val"] * len(val_pairs))

    manifest_rows = []
    cursor = 0
    print(f"[Compile] pairs={len(selected_pairs)}, genes={len(global_gene_ids)}")

    for shard_id, pair in enumerate(selected_pairs):
        rec = _compile_one_pair(
            shard_id=shard_id,
            pair=pair,
            global_gene_ids=global_gene_ids,
            pair_to_sample_id=pair_to_sample_id,
            pair_to_tissue_id=pair_to_tissue_id,
            celltype_map=celltype_map,
            out_dir=out_dir,
            filter_bad_cells=args.filter_bad_cells,
            max_count_threshold=args.max_count_threshold,
            sum_count_threshold=args.sum_count_threshold,
        )
        if rec is None:
            continue
        rec["split"] = split_name[shard_id]
        rec["global_start"] = cursor
        rec["global_end"] = cursor + rec["rows_kept"]
        cursor = rec["global_end"]
        manifest_rows.append(rec)
        if (shard_id + 1) % 100 == 0:
            print(f"[Compile] processed {shard_id + 1}/{len(selected_pairs)} shards, rows={cursor}")

    manifest = {
        "format_version": 1,
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

    # Save a local copy of global gene ids for portability.
    with open(out_dir / "global_gene_ids.txt", "w") as f:
        for g in global_gene_ids:
            f.write(f"{g}\n")

    print(f"[Compile] Done. rows={cursor}, shards={len(manifest_rows)}, out={out_dir}")


if __name__ == "__main__":
    main()

