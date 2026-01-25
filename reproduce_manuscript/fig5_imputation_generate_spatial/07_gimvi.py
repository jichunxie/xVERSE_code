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
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scvi.external import GIMVI

# ==============================================================
# 0) User configuration
# ==============================================================
SC_PATHS = {
    "nsclc_Wu_Zhou_2021_P11": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Wu_Zhou_2021_P11.h5ad",
    "nsclc_Lambrechts_Thienpont_2018_6653_8": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Lambrechts_Thienpont_2018_6653_8.h5ad",
    "nsclc_Chen_Zhang_2020_NSCLC-1": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Chen_Zhang_2020_NSCLC-1.h5ad",
    "nsclc_Chen_Zhang_2020_NSCLC-2": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Chen_Zhang_2020_NSCLC-2.h5ad",
    "nsclc_Chen_Zhang_2020_NSCLC-3": "/hpc/group/xielab/xj58/xVerse_results/fig5/refsc/nsclc_Chen_Zhang_2020_NSCLC-3.h5ad",
}

ST_TRAIN_PATHS = {
    "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split/train_Lung13.h5ad",
    "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split/train_Lung9_Rep1.h5ad",
    "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split/train_Lung6.h5ad",
    "/hpc/group/xielab/xj58/xVerse_results/fig5/cosmx_data/train_test_split/train_Lung5_Rep1.h5ad",
}

OUT_DIR = "/hpc/group/xielab/xj58/xVerse_results/fig5/imputation_results/gimvi"
os.makedirs(OUT_DIR, exist_ok=True)


def run_gimvi_pair(sc_name, sc_path, st_train_path):
    sample_name = os.path.splitext(os.path.basename(st_train_path).replace("train_", ""))[0]
    dir_name = os.path.dirname(st_train_path)
    file_name = os.path.basename(st_train_path)
    test_file_name = file_name.replace("train_", "test_")
    st_test_path = os.path.join(dir_name, test_file_name)

    if not os.path.exists(st_test_path):
        print(f"[Skip] test missing for {sample_name}")
        return None

    print(f"\n=== {sc_name} × {sample_name} (GIMVI Stable) ===")

    # ---------- Load + preprocess ----------
    SC = ad.read_h5ad(sc_path)
    ST_train = ad.read_h5ad(st_train_path)
    ST_test = ad.read_h5ad(st_test_path)

    # Use gene_ids for alignment if available
    print("Aligning genes using var['gene_ids']...")
    for adata in [SC, ST_train, ST_test]:
        if "gene_ids" in adata.var.columns:
            adata.var_names = adata.var["gene_ids"].astype(str)
        else:
            print("  'gene_ids' column not found, using index.")
            adata.var["gene_ids"] = adata.var.index
            adata.var_names = adata.var["gene_ids"].astype(str)
        adata.var_names_make_unique()

    # ---------- Gene selection ----------
    missing_genes = [g for g in SC.var_names if g not in ST_train.var_names]
    eval_genes = [g for g in missing_genes if g in ST_test.var_names]
    if len(eval_genes) == 0:
        print(f"[Info] No evaluable missing genes for {sample_name}")
        return None

    common_genes = SC.var_names.intersection(ST_train.var_names)
    all_genes = list(common_genes.union(eval_genes))
    SC = SC[:, all_genes].copy()
    ST_train = ST_train[:, common_genes].copy()

    # ---------- Model setup ----------
    GIMVI.setup_anndata(SC)
    GIMVI.setup_anndata(ST_train)
    model = GIMVI(SC, ST_train)

    # ---------- Train ----------
    try:
        model.train(
            early_stopping=True,
            max_epochs=20,
        )
    except Exception as e:
        print(f"[Error] Training failed for {sc_name} × {sample_name}: {e}")
        return None

    # ---------- Impute ----------
    _, imputed_st = model.get_imputed_values(normalized=False)
    imputed_st = np.nan_to_num(imputed_st, nan=0.0, posinf=0.0, neginf=0.0)
    imputed_adata = ad.AnnData(X=imputed_st, var=pd.DataFrame(index=SC.var_names))
    imputed_adata = imputed_adata[:, eval_genes].copy()

    # ---------- Compute correlation ----------
    truth = ST_test[:, eval_genes].X
    truth = truth.toarray() if hasattr(truth, "toarray") else truth
    truth = np.nan_to_num(truth, nan=0.0, posinf=0.0, neginf=0.0)
    pred = imputed_adata.X

    corr_list = []
    valid_corrs = []
    for i, g in enumerate(eval_genes):
        x, y = pred[:, i], truth[:, i]
        if np.std(x) == 0 or np.std(y) == 0:
            r = np.nan
        else:
            r, _ = pearsonr(x, y)
            valid_corrs.append(r)
        corr_list.append({"sc": sc_name, "sample": sample_name, "gene": g, "pearson_r": r})

    # Save imputed data
    save_path = os.path.join(OUT_DIR, f"gimvi_{sample_name}_{sc_name}_imputed.h5ad")
    imputed_adata.write_h5ad(save_path)
    print(f"  Saved imputed data to {save_path}")

    # Print Quartiles
    if valid_corrs:
        quartiles = np.percentile(valid_corrs, [0, 25, 50, 75, 100])
        print(f"Pearson Correlation Quartiles for {sc_name} x {sample_name}:")
        print(f"  Min (0%):     {quartiles[0]:.4f}")
        print(f"  25%:          {quartiles[1]:.4f}")
        print(f"  Median (50%): {quartiles[2]:.4f}")
        print(f"  75%:          {quartiles[3]:.4f}")
        print(f"  Max (100%):   {quartiles[4]:.4f}")

    df = pd.DataFrame(corr_list)
    print(f"[Done] {sc_name} × {sample_name} (n_genes={len(eval_genes)}).")
    return df

# ==============================================================
# 2) Batch run
# ==============================================================
all_results = []
for sc_name, sc_path in SC_PATHS.items():
    for st_train_path in sorted(ST_TRAIN_PATHS):
        res_df = run_gimvi_pair(sc_name, sc_path, st_train_path)
        if res_df is not None:
            all_results.append(res_df)

# ==============================================================
# 3) Save final merged results
# ==============================================================
if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(OUT_DIR, "gimvi_lung_samples_gene_corr.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Summary Saved] {summary_path}")
    print(f"Total rows: {summary_df.shape[0]}")
else:
    print("\nNo valid results.")