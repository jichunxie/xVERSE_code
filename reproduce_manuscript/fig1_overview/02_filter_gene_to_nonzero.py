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

import os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

tids  = [
    "UBERON:0000004", "UBERON:0000010", "UBERON:0000029", "UBERON:0000030", "UBERON:0000033",
    "UBERON:0000056", "UBERON:0000059", "UBERON:0000160", "UBERON:0000178", "UBERON:0000310",
    "UBERON:0000344", "UBERON:0000403", "UBERON:0000473", "UBERON:0000916", "UBERON:0000922",
    "UBERON:0000945", "UBERON:0000948", "UBERON:0000955", "UBERON:0000970", "UBERON:0000974",
    "UBERON:0000977", "UBERON:0000990", "UBERON:0000992", "UBERON:0000995", "UBERON:0001004",
    "UBERON:0001007", "UBERON:0001013", "UBERON:0001015", "UBERON:0001017", "UBERON:0001032",
    "UBERON:0001040", "UBERON:0001043", "UBERON:0001087", "UBERON:0001155", "UBERON:0001255",
    "UBERON:0001264", "UBERON:0001434", "UBERON:0001723", "UBERON:0001836", "UBERON:0001851",
    "UBERON:0001913", "UBERON:0001987", "UBERON:0002048", "UBERON:0002049", "UBERON:0002097",
    "UBERON:0002102", "UBERON:0002103", "UBERON:0002106", "UBERON:0002107", "UBERON:0002108",
    "UBERON:0002110", "UBERON:0002113", "UBERON:0002240", "UBERON:0002365", "UBERON:0002367",
    "UBERON:0002368", "UBERON:0002369", "UBERON:0002371", "UBERON:0002405", "UBERON:0003688",
    "UBERON:0003889", "UBERON:0007650", "UBERON:0009472", "UBERON:0016435", "UBERON:0018707",
    "UBERON:0035210", "UBERON:8480009"
]


for tissue_id in tids:
    # Paths
    input_folder = f"/hpc/group/xielab/xj58/SpaRestAtlas/npz_tissue_dataset_donor/{tissue_id}"
    
    # Skip if input folder does not exist
    if not os.path.exists(input_folder):
        print(f"Skipping {tissue_id}: input folder does not exist.")
        continue  # Skip to the next tissue_id
    output_folder = input_folder.rstrip("/") + "_filtered"

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Gather all matrix files grouped by dataset_id
    dataset_to_files = defaultdict(list)

    for fname in os.listdir(input_folder):
        if fname.endswith("_X.npz"):
            parts = fname.split("__")
            if len(parts) >= 3:
                donor_id = parts[0]
                dataset_id = parts[1]
                dataset_to_files[dataset_id].append(fname)

    # Step 2: Process each dataset
    for dataset_id, file_list in dataset_to_files.items():
        zero_col_sets = []
        example_gene_ids = None

        print(f"\nProcessing dataset_id: {dataset_id}")
        
        # First pass: find columns to remove
        for fname in sorted(file_list):
            matrix_path = os.path.join(input_folder, fname)
            meta_path = matrix_path.replace("_X.npz", "_obs.npz")

            # Load matrix
            X = sp.load_npz(matrix_path)
            
            # Find columns where all entries are zero
            col_nonzero_counts = X.getnnz(axis=0)
            zero_cols = set(np.where(col_nonzero_counts == 0)[0])
            zero_col_sets.append(zero_cols)

            # Load gene_ids (only once)
            if example_gene_ids is None:
                meta = np.load(meta_path, allow_pickle=True)
                example_gene_ids = meta["gene_ids"]

        # Find intersection of zero columns across all chunks
        if zero_col_sets:
            final_zero_cols = set.intersection(*zero_col_sets)
        else:
            final_zero_cols = set()

        final_zero_cols_sorted = sorted(final_zero_cols)

        if example_gene_ids is None:
            print(f"  -> Warning: No gene_ids found for {dataset_id}, skipping.")
            continue

        genes_to_remove = [example_gene_ids[idx] for idx in final_zero_cols_sorted]
        genes_remaining = len(example_gene_ids) - len(genes_to_remove)

        print(f"  Genes remaining after removal: {genes_remaining}")

        # Second pass: save modified files to output folder
        for fname in sorted(file_list):
            matrix_path = os.path.join(input_folder, fname)
            meta_path = matrix_path.replace("_X.npz", "_obs.npz")

            # Load original matrix and metadata
            X = sp.load_npz(matrix_path)
            meta = np.load(meta_path, allow_pickle=True)

            # Remove columns
            if final_zero_cols_sorted:
                cols_to_keep = np.setdiff1d(np.arange(X.shape[1]), final_zero_cols_sorted)
                X_filtered = X[:, cols_to_keep]
            else:
                X_filtered = X  # No filtering needed

            # Remove corresponding gene_ids
            gene_ids = meta["gene_ids"]
            if final_zero_cols_sorted:
                gene_ids_filtered = np.delete(gene_ids, final_zero_cols_sorted)
            else:
                gene_ids_filtered = gene_ids

            # Save new matrix
            output_matrix_path = os.path.join(output_folder, fname)
            sp.save_npz(output_matrix_path, X_filtered)

            # Save new meta
            output_meta_path = output_matrix_path.replace("_X.npz", "_obs.npz")
            new_meta = {k: v for k, v in meta.items()}
            new_meta["gene_ids"] = gene_ids_filtered
            np.savez_compressed(output_meta_path, **new_meta)