#!/usr/bin/env python3
"""
Update only celltype_id.npy in an existing xverse_train_v1 compiled dataset.

This avoids rebuilding expression shards. It uses manifest["source_pairs"],
each shard's source_pair_id.npy/source_row_id.npy, and the original meta npz
files to remap cell_type_ontology_term_id to new integer labels.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from main_mfa.build_train_dataset_v1 import build_cell_type_to_index


def parse_args():
    ap = argparse.ArgumentParser(description="Update compiled dataset celltype_id.npy only.")
    ap.add_argument("--compiled-root", required=True, help="Existing compiled dataset root.")
    ap.add_argument("--cell-type-csv", required=True, help="New cell type map CSV.")
    ap.add_argument("--splits", default="train,val", help="Comma-separated splits to update.")
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Rename the original celltype_id.npy to celltype_id_raw.npy before overwriting.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Compute and report, but do not write files.")
    return ap.parse_args()


def _load_manifest(compiled_root: Path):
    manifest_path = compiled_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if manifest.get("format") != "xverse_train_v1":
        raise ValueError(f"Unsupported compiled dataset format: {manifest.get('format')}")
    return manifest_path, manifest


def _load_pair_celltypes(manifest, celltype_map):
    pair_to_celltypes = {}
    for rec in manifest.get("source_pairs", []):
        pair_id = int(rec["pair_id"])
        meta_path = str(rec["meta_path"])
        meta = np.load(meta_path, allow_pickle=True)
        if "cell_type_ontology_term_id" in meta:
            raw = np.asarray(meta["cell_type_ontology_term_id"])
        else:
            raw = np.asarray(["unknown"] * int(meta.get("n_obs", 0)), dtype=object)
        mapped = np.asarray([celltype_map.get(str(x), -1) for x in raw], dtype=np.int32)
        pair_to_celltypes[pair_id] = mapped
    return pair_to_celltypes


def _update_one_shard(shard_dir: Path, pair_to_celltypes, backup: bool, dry_run: bool):
    source_pair = np.load(shard_dir / "source_pair_id.npy", mmap_mode="r")
    source_row = np.load(shard_dir / "source_row_id.npy", mmap_mode="r")
    out = shard_dir / "celltype_id.npy"
    raw_backup = shard_dir / "celltype_id_raw.npy"
    old = np.load(out, mmap_mode="r")
    new = np.empty((old.shape[0],), dtype=np.int32)

    for pair_id in np.unique(np.asarray(source_pair)):
        pair_id = int(pair_id)
        if pair_id not in pair_to_celltypes:
            raise KeyError(f"pair_id={pair_id} not found in manifest source_pairs")
        mask = np.asarray(source_pair == pair_id)
        rows = np.asarray(source_row[mask], dtype=np.int64)
        ct = pair_to_celltypes[pair_id]
        if rows.size and (rows.min() < 0 or rows.max() >= ct.shape[0]):
            raise IndexError(
                f"{shard_dir}: source_row_id out of range for pair_id={pair_id}: "
                f"min={rows.min()}, max={rows.max()}, n={ct.shape[0]}"
            )
        new[mask] = ct[rows]

    changed = int(np.sum(np.asarray(old) != new))
    valid = int(np.sum(new >= 0))
    if not dry_run:
        # Close the mmap before renaming on filesystems that lock open files.
        del old
        if backup and not raw_backup.exists():
            os.replace(out, raw_backup)
        np.save(out, new, allow_pickle=False)
    return {"shard": str(shard_dir), "cells": int(new.size), "valid": valid, "changed": changed}


def main():
    args = parse_args()
    compiled_root = Path(args.compiled_root)
    manifest_path, manifest = _load_manifest(compiled_root)
    celltype_map = build_cell_type_to_index(args.cell_type_csv)
    pair_to_celltypes = _load_pair_celltypes(manifest, celltype_map)

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    rows = []
    for split in splits:
        split_info = manifest.get("splits", {}).get(split)
        if split_info is None:
            print(f"[skip] split not found in manifest: {split}")
            continue
        for rec in split_info.get("shards", []):
            shard_dir = Path(rec["path"])
            rows.append(_update_one_shard(shard_dir, pair_to_celltypes, backup=args.backup, dry_run=args.dry_run))

    total_cells = int(sum(r["cells"] for r in rows))
    total_valid = int(sum(r["valid"] for r in rows))
    total_changed = int(sum(r["changed"] for r in rows))
    summary = {
        "compiled_root": str(compiled_root),
        "cell_type_csv": str(args.cell_type_csv),
        "splits": splits,
        "dry_run": bool(args.dry_run),
        "updated_at_unix": int(time.time()),
        "num_shards": int(len(rows)),
        "num_cells": total_cells,
        "num_valid_celltype": total_valid,
        "valid_fraction": float(total_valid / max(total_cells, 1)),
        "num_changed": total_changed,
    }
    print("[CellTypeUpdateDone]")
    print(json.dumps(summary, indent=2))

    if not args.dry_run:
        out_summary = compiled_root / "celltype_update_summary.json"
        with open(out_summary, "w") as f:
            json.dump(summary, f, indent=2)
        manifest.setdefault("source", {})["cell_type_csv"] = str(args.cell_type_csv)
        manifest["celltype_updated_at_unix"] = int(time.time())
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
