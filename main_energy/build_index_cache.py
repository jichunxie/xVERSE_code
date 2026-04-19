import argparse
import hashlib
import os
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from main_energy.utils_model import load_gene_ids, build_pair_to_sample_id_and_paths


def _read_sparse_npz_shape(matrix_path: str) -> Tuple[int, int]:
    with np.load(matrix_path, allow_pickle=False) as npz:
        shape = npz["shape"]
        return int(shape[0]), int(shape[1])


def _to_object_array(seq: List[np.ndarray]) -> np.ndarray:
    out = np.empty((len(seq),), dtype=object)
    for i, v in enumerate(seq):
        out[i] = v
    return out


def _build_signature(
    pairs: List[Tuple[str, str]],
    gene_ids: List[str],
    pair_to_sample_id,
    pair_to_tissue_id,
    filter_bad_cells: bool,
) -> str:
    h = hashlib.sha1()
    h.update(str(len(gene_ids)).encode("utf-8"))
    h.update(str(filter_bad_cells).encode("utf-8"))
    for matrix_path, meta_path in pairs:
        h.update(str(matrix_path).encode("utf-8"))
        h.update(b"|")
        h.update(str(meta_path).encode("utf-8"))
        h.update(b"|")
        pair = (matrix_path, meta_path)
        h.update(str(pair_to_sample_id.get(pair, -1)).encode("utf-8"))
        h.update(b"|")
        h.update(str(pair_to_tissue_id.get(pair, -1) if pair_to_tissue_id is not None else -1).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_cache_for_pairs(
    pairs: List[Tuple[str, str]],
    gene_ids: List[str],
    pair_to_sample_id,
    pair_to_tissue_id,
    cache_path: str,
    filter_bad_cells: bool = False,
):
    index_map = []
    block_matrix_paths = []
    block_meta_paths = []
    block_nrows = []
    block_observed_global_idx = []
    block_local_to_rel = []

    for block_idx, (matrix_path, meta_path) in enumerate(pairs):
        meta = np.load(meta_path, allow_pickle=True)
        gene_ids_available = meta["gene_ids"]
        gene_to_idx = {g: i for i, g in enumerate(gene_ids_available)}
        gene_idx_array = np.full(len(gene_ids), -1, dtype=np.int32)
        for global_idx, gene in enumerate(gene_ids):
            if gene in gene_to_idx:
                gene_idx_array[global_idx] = gene_to_idx[gene]

        observed_global_idx = np.where(gene_idx_array >= 0)[0].astype(np.int32, copy=False)
        observed_local_idx = gene_idx_array[observed_global_idx].astype(np.int32, copy=False)

        n_rows, n_cols = _read_sparse_npz_shape(matrix_path)
        local_to_rel = np.full(n_cols, -1, dtype=np.int32)
        if observed_local_idx.size > 0:
            local_to_rel[observed_local_idx] = np.arange(observed_local_idx.size, dtype=np.int32)

        block_observed_global_idx.append(observed_global_idx)
        block_local_to_rel.append(local_to_rel)
        block_matrix_paths.append(matrix_path)
        block_meta_paths.append(meta_path)
        block_nrows.append(n_rows)

        pair = (matrix_path, meta_path)
        sample_id = int(pair_to_sample_id.get(pair, -1))
        if sample_id < 0:
            raise ValueError(f"Sample ID not found for {pair}")
        tissue_id = int(pair_to_tissue_id.get(pair, -1))

        if filter_bad_cells:
            X = sp.load_npz(matrix_path).tocsr()
            row_sum = np.asarray(X.sum(axis=1)).ravel()
            row_max = np.asarray(X.max(axis=1).toarray()).ravel()
            valid_rows = np.where((row_max <= 1000.0) & (row_sum <= 200000.0))[0]
        else:
            valid_rows = np.arange(n_rows, dtype=np.int64)

        index_map.extend((block_idx, int(i), sample_id, tissue_id) for i in valid_rows)
        if (block_idx + 1) % 100 == 0:
            print(f"[Cache] Processed {block_idx + 1}/{len(pairs)} blocks")

    signature = _build_signature(
        pairs=pairs,
        gene_ids=gene_ids,
        pair_to_sample_id=pair_to_sample_id,
        pair_to_tissue_id=pair_to_tissue_id,
        filter_bad_cells=filter_bad_cells,
    )
    index_map = np.asarray(index_map, dtype=np.int64)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=np.array(signature, dtype=object),
        index_map=index_map,
        block_matrix_paths=np.asarray(block_matrix_paths, dtype=object),
        block_meta_paths=np.asarray(block_meta_paths, dtype=object),
        block_nrows=np.asarray(block_nrows, dtype=np.int64),
        block_observed_global_idx=_to_object_array(block_observed_global_idx),
        block_local_to_rel=_to_object_array(block_local_to_rel),
    )
    print(f"[Cache] Saved: {cache_path}")
    print(f"[Cache] Rows: {index_map.shape[0]}, Blocks: {len(block_matrix_paths)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build offline index caches for xVERSE dataset.")
    parser.add_argument("--data-root", required=True, help="Dataset root.")
    parser.add_argument("--summary-csv", default=None, help="Summary CSV path.")
    parser.add_argument("--gene-ids-path", default=None, help="Gene IDs path.")
    parser.add_argument("--use-tissue", default=None, help="Optional tissue filter.")
    parser.add_argument("--train-cache-path", required=True, help="Output cache path for train split.")
    parser.add_argument("--val-cache-path", required=True, help="Output cache path for val split.")
    parser.add_argument("--filter-bad-cells", action="store_true", help="Apply row filter (max>1000 or sum>200000).")
    return parser.parse_args()


def main():
    args = parse_args()
    summary_csv = args.summary_csv or os.path.join(args.data_root, "pantissue_full_updated.csv")
    gene_ids_path = args.gene_ids_path or os.path.join(args.data_root, "ensg_keys_high_quality.txt")

    pair_to_sample_id, train_pairs, val_pairs, pair_to_tissue_id, _ = build_pair_to_sample_id_and_paths(
        summary_csv,
        use_tissue=args.use_tissue,
    )
    gene_ids = load_gene_ids(gene_ids_path)
    print(f"[Cache] Genes: {len(gene_ids)}, Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    build_cache_for_pairs(
        pairs=train_pairs,
        gene_ids=gene_ids,
        pair_to_sample_id=pair_to_sample_id,
        pair_to_tissue_id=pair_to_tissue_id,
        cache_path=args.train_cache_path,
        filter_bad_cells=args.filter_bad_cells,
    )
    build_cache_for_pairs(
        pairs=val_pairs,
        gene_ids=gene_ids,
        pair_to_sample_id=pair_to_sample_id,
        pair_to_tissue_id=pair_to_tissue_id,
        cache_path=args.val_cache_path,
        filter_bad_cells=args.filter_bad_cells,
    )


if __name__ == "__main__":
    main()

