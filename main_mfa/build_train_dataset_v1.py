import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a training-optimized sharded sparse dataset with cell-level metadata."
    )
    p.add_argument("--summary-csv", required=True, help="CSV with matrix/obs paths and split/sample/tissue columns.")
    p.add_argument("--gene-ids-path", required=True, help="Global gene id file (one gene id per line).")
    p.add_argument("--out-dir", required=True, help="Output dataset directory.")
    p.add_argument("--cell-type-csv", default=None, help="Optional cell type mapping CSV.")
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--use-tissue", default=None, help="Optional tissue name filter.")
    p.add_argument("--allowed-sample-ids", default=None, help="Optional comma-separated sample_id list.")
    p.add_argument("--target-cells-per-shard", type=int, default=200000, help="Target rows per output shard.")
    p.add_argument("--filter-bad-cells", action="store_true")
    p.add_argument("--max-count-threshold", type=float, default=1000.0)
    p.add_argument("--sum-count-threshold", type=float, default=200000.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_gene_ids(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_allowed_sample_ids(s: Optional[str]) -> Optional[set]:
    if s is None or str(s).strip() == "":
        return None
    return {int(x.strip()) for x in str(s).split(",") if x.strip()}


def build_cell_type_to_index(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    if "id" not in df.columns or "classification_result" not in df.columns:
        raise ValueError("cell-type CSV must contain columns: id, classification_result")
    df = df.dropna(subset=["classification_result"])
    uncategorized = {"Other/Unknown"}
    valid = sorted(c for c in df["classification_result"].unique() if c not in uncategorized)
    cls_to_idx = {c: i for i, c in enumerate(valid)}
    for c in uncategorized:
        cls_to_idx[c] = -1
    return {str(row["id"]): int(cls_to_idx.get(row["classification_result"], -1)) for _, row in df.iterrows()}


def parse_summary_rows(
    summary_csv: str,
    split_mode: str,
    use_tissue: Optional[str],
    allowed_sample_ids: Optional[set],
) -> List[dict]:
    df = pd.read_csv(summary_csv)
    required = {"sample_id", "matrix_paths", "obs_paths", "split", "tissue_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary csv missing columns: {missing}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    if allowed_sample_ids is not None:
        df = df[df["sample_id"].isin(allowed_sample_ids)]

    if use_tissue is not None:
        if "tissue_name" not in df.columns:
            raise ValueError("use_tissue was provided but summary csv has no tissue_name column")
        q = str(use_tissue).strip().casefold()
        df = df[df["tissue_name"].astype(str).str.strip().str.casefold() == q]

    accepted = {"train", "val", "valid", "validation"}
    if split_mode == "train":
        accepted = {"train"}
    elif split_mode == "val":
        accepted = {"val", "valid", "validation"}

    rows = []
    for _, row in df.iterrows():
        split = str(row["split"]).strip().lower()
        if split not in accepted:
            continue
        split_norm = "train" if split == "train" else "val"
        sample_id = int(row["sample_id"])
        tissue_id = int(row["tissue_id"])
        matrix_paths = [p.strip() for p in str(row["matrix_paths"]).split(";") if p.strip()]
        obs_paths = [p.strip() for p in str(row["obs_paths"]).split(";") if p.strip()]
        for matrix_path, obs_path in zip(matrix_paths, obs_paths):
            rows.append(
                {
                    "split": split_norm,
                    "sample_id": sample_id,
                    "tissue_id": tissue_id,
                    "matrix_path": matrix_path,
                    "meta_path": obs_path,
                }
            )
    return rows


def _row_sum_max_from_csr(indptr: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_rows = indptr.size - 1
    row_sum = np.zeros((n_rows,), dtype=np.float64)
    row_max = np.zeros((n_rows,), dtype=np.float64)
    if data.size == 0:
        return row_sum, row_max

    non_empty = indptr[1:] > indptr[:-1]
    starts = indptr[:-1][non_empty]
    vals = data.astype(np.float64, copy=False)
    row_sum[non_empty] = np.add.reduceat(vals, starts)
    row_max[non_empty] = np.maximum.reduceat(vals, starts)
    return row_sum, row_max


def _choose_values_dtype(x: np.ndarray) -> Tuple[np.ndarray, str]:
    if x.size == 0:
        return x.astype(np.uint16), "uint16"
    if np.issubdtype(x.dtype, np.integer):
        mn = int(x.min())
        mx = int(x.max())
        if mn >= 0 and mx <= np.iinfo(np.uint16).max:
            return x.astype(np.uint16, copy=False), "uint16"
        if mn >= 0 and mx <= np.iinfo(np.uint32).max:
            return x.astype(np.uint32, copy=False), "uint32"
    rounded = np.rint(x)
    if np.allclose(x, rounded):
        xi = rounded.astype(np.int64, copy=False)
        mn = int(xi.min())
        mx = int(xi.max())
        if mn >= 0 and mx <= np.iinfo(np.uint16).max:
            return xi.astype(np.uint16, copy=False), "uint16"
        if mn >= 0 and mx <= np.iinfo(np.uint32).max:
            return xi.astype(np.uint32, copy=False), "uint32"
    return x.astype(np.float32, copy=False), "float32"


def _build_local_to_global(block_gene_ids: Sequence[str], global_gene_to_idx: Dict[str, int]) -> np.ndarray:
    local_to_global = np.full((len(block_gene_ids),), -1, dtype=np.int32)
    for local_idx, gene_id in enumerate(block_gene_ids):
        gidx = global_gene_to_idx.get(str(gene_id))
        if gidx is not None:
            local_to_global[local_idx] = int(gidx)
    return local_to_global


def _slice_csr_rows(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    row_start: int,
    row_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_nz = int(indptr[row_start])
    end_nz = int(indptr[row_end])
    sub_indices = indices[start_nz:end_nz]
    sub_values = values[start_nz:end_nz]
    sub_ptr = indptr[row_start : row_end + 1].astype(np.uint64, copy=False) - np.uint64(start_nz)
    return sub_ptr, sub_indices, sub_values


class SplitShardWriter:
    def __init__(self, split: str, out_dir: Path, target_cells_per_shard: int):
        self.split = split
        self.out_dir = out_dir / split
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target = max(1, int(target_cells_per_shard))

        self.shard_id = 0
        self.global_cell_cursor = 0
        self.records = []

        self._reset_buffers()

    def _reset_buffers(self):
        self.cell_ptr_parts: List[np.ndarray] = [np.array([0], dtype=np.uint64)]
        self.gene_idx_parts: List[np.ndarray] = []
        self.gene_val_parts: List[np.ndarray] = []
        self.celltype_parts: List[np.ndarray] = []
        self.sample_parts: List[np.ndarray] = []
        self.tissue_parts: List[np.ndarray] = []
        self.source_pair_parts: List[np.ndarray] = []
        self.source_row_parts: List[np.ndarray] = []
        self.rows = 0
        self.nnz = 0

    def _append_chunk(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        celltype_id: np.ndarray,
        sample_id: np.ndarray,
        tissue_id: np.ndarray,
        source_pair_id: np.ndarray,
        source_row_id: np.ndarray,
    ):
        n_rows = int(indptr.size - 1)
        if n_rows <= 0:
            return
        cur_nnz = int(self.cell_ptr_parts[-1][-1])
        self.cell_ptr_parts.append(indptr[1:] + np.uint64(cur_nnz))
        self.gene_idx_parts.append(indices)
        self.gene_val_parts.append(values)
        self.celltype_parts.append(celltype_id)
        self.sample_parts.append(sample_id)
        self.tissue_parts.append(tissue_id)
        self.source_pair_parts.append(source_pair_id)
        self.source_row_parts.append(source_row_id)
        self.rows += n_rows
        self.nnz += int(indices.size)

    def add_block(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        celltype_id: np.ndarray,
        sample_id: np.ndarray,
        tissue_id: np.ndarray,
        source_pair_id: np.ndarray,
        source_row_id: np.ndarray,
    ):
        n_rows = int(indptr.size - 1)
        if n_rows == 0:
            return

        row_cursor = 0
        while row_cursor < n_rows:
            room = self.target - self.rows
            if room <= 0:
                self.flush()
                room = self.target
            take = min(room, n_rows - row_cursor)
            rs = row_cursor
            re = row_cursor + take
            sub_ptr, sub_idx, sub_val = _slice_csr_rows(indptr, indices, values, rs, re)
            self._append_chunk(
                indptr=sub_ptr,
                indices=sub_idx,
                values=sub_val,
                celltype_id=celltype_id[rs:re],
                sample_id=sample_id[rs:re],
                tissue_id=tissue_id[rs:re],
                source_pair_id=source_pair_id[rs:re],
                source_row_id=source_row_id[rs:re],
            )
            row_cursor = re

    def flush(self):
        if self.rows == 0:
            return

        shard_name = f"shard_{self.shard_id:06d}"
        shard_dir = self.out_dir / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)

        cell_ptr = np.concatenate(self.cell_ptr_parts).astype(np.uint64, copy=False)
        gene_idx = (
            np.concatenate(self.gene_idx_parts).astype(np.uint32, copy=False)
            if self.gene_idx_parts
            else np.empty((0,), dtype=np.uint32)
        )
        gene_val_raw = np.concatenate(self.gene_val_parts) if self.gene_val_parts else np.empty((0,), dtype=np.float32)
        gene_val, gene_val_dtype = _choose_values_dtype(gene_val_raw)
        celltype_id = np.concatenate(self.celltype_parts).astype(np.int32, copy=False)
        sample_id = np.concatenate(self.sample_parts).astype(np.int32, copy=False)
        tissue_id = np.concatenate(self.tissue_parts).astype(np.int32, copy=False)
        source_pair_id = np.concatenate(self.source_pair_parts).astype(np.int32, copy=False)
        source_row_id = np.concatenate(self.source_row_parts).astype(np.int32, copy=False)

        np.save(shard_dir / "cell_ptr.npy", cell_ptr, allow_pickle=False)
        np.save(shard_dir / "gene_idx.npy", gene_idx, allow_pickle=False)
        np.save(shard_dir / "gene_val.npy", gene_val, allow_pickle=False)
        np.save(shard_dir / "celltype_id.npy", celltype_id, allow_pickle=False)
        np.save(shard_dir / "sample_id.npy", sample_id, allow_pickle=False)
        np.save(shard_dir / "tissue_id.npy", tissue_id, allow_pickle=False)
        np.save(shard_dir / "source_pair_id.npy", source_pair_id, allow_pickle=False)
        np.save(shard_dir / "source_row_id.npy", source_row_id, allow_pickle=False)

        rec = {
            "split": self.split,
            "shard_id": self.shard_id,
            "path": str(shard_dir),
            "num_cells": int(self.rows),
            "num_nnz": int(gene_idx.size),
            "gene_val_dtype": gene_val_dtype,
            "global_cell_start": int(self.global_cell_cursor),
            "global_cell_end": int(self.global_cell_cursor + self.rows),
        }
        self.records.append(rec)
        self.global_cell_cursor += self.rows
        self.shard_id += 1
        self._reset_buffers()


def _process_one_pair(
    pair_id: int,
    matrix_path: str,
    meta_path: str,
    sample_id: int,
    tissue_id: int,
    global_gene_to_idx: Dict[str, int],
    celltype_map: Optional[Dict[str, int]],
    filter_bad_cells: bool,
    max_count_threshold: float,
    sum_count_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = sp.load_npz(matrix_path).tocsr()
    meta = np.load(meta_path, allow_pickle=True)
    if "gene_ids" not in meta:
        raise ValueError(f"{meta_path} does not contain 'gene_ids'")
    block_gene_ids = np.asarray(meta["gene_ids"])

    if "cell_type_ontology_term_id" in meta:
        ct_raw = np.asarray(meta["cell_type_ontology_term_id"])
    else:
        ct_raw = np.asarray(["unknown"] * X.shape[0], dtype=object)

    if filter_bad_cells:
        row_sum, row_max = _row_sum_max_from_csr(X.indptr, X.data)
        valid_rows = np.where((row_max <= max_count_threshold) & (row_sum <= sum_count_threshold))[0]
        if valid_rows.size != X.shape[0]:
            X = X[valid_rows].tocsr()
            ct_raw = ct_raw[valid_rows]
            source_rows = valid_rows.astype(np.int32, copy=False)
        else:
            source_rows = np.arange(X.shape[0], dtype=np.int32)
    else:
        source_rows = np.arange(X.shape[0], dtype=np.int32)

    local_to_global = _build_local_to_global(block_gene_ids, global_gene_to_idx)
    mapped = local_to_global[X.indices]
    valid_nz = mapped >= 0
    new_indices = mapped[valid_nz].astype(np.uint32, copy=False)
    new_values = X.data[valid_nz]

    csum = np.concatenate([[0], np.cumsum(valid_nz.astype(np.uint8), dtype=np.uint64)])
    new_indptr = csum[X.indptr].astype(np.uint64, copy=False)

    if celltype_map is None:
        celltype_id = np.full((X.shape[0],), -1, dtype=np.int32)
    else:
        celltype_id = np.asarray([celltype_map.get(str(x), -1) for x in ct_raw], dtype=np.int32)

    sample_arr = np.full((X.shape[0],), int(sample_id), dtype=np.int32)
    tissue_arr = np.full((X.shape[0],), int(tissue_id), dtype=np.int32)
    pair_arr = np.full((X.shape[0],), int(pair_id), dtype=np.int32)

    return new_indptr, new_indices, new_values, celltype_id, sample_arr, tissue_arr, pair_arr, source_rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise ValueError(f"output directory is not empty: {out_dir}. Use --overwrite to allow writing into it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    gene_ids = load_gene_ids(args.gene_ids_path)
    global_gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    allowed_sample_ids = parse_allowed_sample_ids(args.allowed_sample_ids)
    celltype_map = build_cell_type_to_index(args.cell_type_csv) if args.cell_type_csv else None

    rows = parse_summary_rows(
        summary_csv=args.summary_csv,
        split_mode=args.split,
        use_tissue=args.use_tissue,
        allowed_sample_ids=allowed_sample_ids,
    )
    if len(rows) == 0:
        raise ValueError("No matrix/meta pairs selected after filtering.")

    # Stable processing order for reproducibility.
    rows = sorted(rows, key=lambda x: (x["split"], x["sample_id"], x["matrix_path"], x["meta_path"]))

    writers = {
        "train": SplitShardWriter(split="train", out_dir=out_dir, target_cells_per_shard=args.target_cells_per_shard),
        "val": SplitShardWriter(split="val", out_dir=out_dir, target_cells_per_shard=args.target_cells_per_shard),
    }

    iterator = enumerate(rows)
    if tqdm is not None:
        iterator = enumerate(tqdm(rows, total=len(rows), desc="Building dataset", unit="pair"))

    source_pairs = []
    dropped_or_empty_pairs = 0
    for pair_id, rec in iterator:
        split = rec["split"]
        matrix_path = rec["matrix_path"]
        meta_path = rec["meta_path"]
        sample_id = rec["sample_id"]
        tissue_id = rec["tissue_id"]
        source_pairs.append(
            {
                "pair_id": int(pair_id),
                "split": split,
                "matrix_path": matrix_path,
                "meta_path": meta_path,
                "sample_id": int(sample_id),
                "tissue_id": int(tissue_id),
            }
        )

        indptr, indices, values, celltype_id, sample_arr, tissue_arr, pair_arr, source_rows = _process_one_pair(
            pair_id=pair_id,
            matrix_path=matrix_path,
            meta_path=meta_path,
            sample_id=sample_id,
            tissue_id=tissue_id,
            global_gene_to_idx=global_gene_to_idx,
            celltype_map=celltype_map,
            filter_bad_cells=args.filter_bad_cells,
            max_count_threshold=args.max_count_threshold,
            sum_count_threshold=args.sum_count_threshold,
        )
        if indptr.size <= 1:
            dropped_or_empty_pairs += 1
            continue

        writers[split].add_block(
            indptr=indptr,
            indices=indices,
            values=values,
            celltype_id=celltype_id,
            sample_id=sample_arr,
            tissue_id=tissue_arr,
            source_pair_id=pair_arr,
            source_row_id=source_rows,
        )

    for w in writers.values():
        w.flush()

    with open(out_dir / "global_gene_ids.txt", "w") as f:
        for g in gene_ids:
            f.write(f"{g}\n")

    manifest = {
        "format": "xverse_train_v1",
        "created_at_unix": int(time.time()),
        "elapsed_sec": float(time.time() - started_at),
        "source": {
            "summary_csv": str(args.summary_csv),
            "gene_ids_path": str(args.gene_ids_path),
            "cell_type_csv": str(args.cell_type_csv) if args.cell_type_csv else None,
        },
        "build_args": {
            "split": args.split,
            "use_tissue": args.use_tissue,
            "allowed_sample_ids": sorted(list(allowed_sample_ids)) if allowed_sample_ids is not None else None,
            "target_cells_per_shard": int(args.target_cells_per_shard),
            "filter_bad_cells": bool(args.filter_bad_cells),
            "max_count_threshold": float(args.max_count_threshold),
            "sum_count_threshold": float(args.sum_count_threshold),
        },
        "global_num_genes": int(len(gene_ids)),
        "splits": {
            "train": {
                "num_shards": int(len(writers["train"].records)),
                "num_cells": int(sum(r["num_cells"] for r in writers["train"].records)),
                "num_nnz": int(sum(r["num_nnz"] for r in writers["train"].records)),
                "shards": writers["train"].records,
            },
            "val": {
                "num_shards": int(len(writers["val"].records)),
                "num_cells": int(sum(r["num_cells"] for r in writers["val"].records)),
                "num_nnz": int(sum(r["num_nnz"] for r in writers["val"].records)),
                "shards": writers["val"].records,
            },
        },
        "source_pairs": source_pairs,
        "dropped_or_empty_pairs": int(dropped_or_empty_pairs),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("[BuildDone]")
    print(f"out_dir={out_dir}")
    print(f"train_shards={manifest['splits']['train']['num_shards']}, train_cells={manifest['splits']['train']['num_cells']}")
    print(f"val_shards={manifest['splits']['val']['num_shards']}, val_cells={manifest['splits']['val']['num_cells']}")
    print(f"genes={manifest['global_num_genes']}, elapsed_sec={manifest['elapsed_sec']:.2f}")


if __name__ == "__main__":
    main()
