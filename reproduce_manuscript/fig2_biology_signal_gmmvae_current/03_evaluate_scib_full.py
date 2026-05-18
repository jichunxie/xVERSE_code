#!/usr/bin/env python3
"""
Evaluate GMVAE and existing foundation model embeddings with a broader scIB metric set.

Design goals:
1) Reuse already computed embeddings in donor h5ad files (no re-extraction for other FMs).
2) Add GMVAE embedding key into comparison.
3) Compute a broader set of scIB metrics with robust try/except fallback.
4) Save per-tissue/per-gene-set csv files and one global summary csv.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import chisquare


def parse_args():
    ap = argparse.ArgumentParser(description="Comprehensive scIB evaluation for fig2 embeddings.")
    ap.add_argument("--liver-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/liver")
    ap.add_argument("--brain-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/brain")
    ap.add_argument(
        "--dataset-manifest",
        default=None,
        help=(
            "Optional CSV for gold-standard h5ad evaluation. Required columns: dataset,path,batch_key,label_key. "
            "When set, liver/brain dirs are ignored."
        ),
    )
    ap.add_argument("--output-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2_gmmvae_current/evaluation_scib_full")
    ap.add_argument("--old-eval-dir", default="/hpc/group/xielab/xj58/xVerse_results/fig2/evaluation")
    ap.add_argument("--gmm-key", default="xVerse_gmmvae_mixmu")
    ap.add_argument("--neighbors-k", type=int, default=15)
    ap.add_argument("--max-cells", type=int, default=20000, help="0 means all cells; otherwise random subsample for speed.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scib-n-cores", type=int, default=1, help="Cores passed to scIB LISI/kBET style metrics.")
    ap.add_argument("--scib-subsample", type=float, default=0.5, help="Subsample fraction passed to scIB metrics().")
    ap.add_argument("--scib-organism", default="human", help="Organism passed to scIB cell-cycle metric.")
    ap.add_argument(
        "--skip-official-metrics-all",
        action="store_true",
        help="Skip the official scib.metrics.metrics(... all flags ...) call and run only robust individual metrics.",
    )
    return ap.parse_args()


MODELS = [
    ("Unintegrated", "X_pca"),
    ("xVerse", "xVerse"),
    ("GMVAE", "xVerse_gmmvae"),
    ("Harmony", "harmony"),
    ("scGPT", "scgpt"),
    ("Nicheformer", "nicheformer"),
    ("Geneformer", "geneformer"),
]

SCIB_SCORE_TABLE = [
    ("Isolated labels", "Bio conservation"),
    ("KMeans NMI", "Bio conservation"),
    ("KMeans ARI", "Bio conservation"),
    ("Silhouette label", "Bio conservation"),
    ("cLISI", "Bio conservation"),
    ("Silhouette batch", "Batch correction"),
    ("iLISI", "Batch correction"),
    ("KBET", "Batch correction"),
    ("Graph connectivity", "Batch correction"),
    ("Bio conservation", "Aggregate score"),
    ("Batch correction", "Aggregate score"),
]

SCIB_METRIC_COLUMNS = [name for name, _ in SCIB_SCORE_TABLE]
METRIC_OUTPUT_COLUMNS = ["tissue", "gene_set", "model", "key", "n_cells"] + SCIB_METRIC_COLUMNS

PCR_COLUMNS = {
    "PCR_batch",
    "PCR_before",
    "PCR_after",
    "PCR_delta_after_minus_before",
    "PCR comparison",
}


def drop_pcr_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.drop(columns=[c for c in PCR_COLUMNS if c in df.columns], errors="ignore")


def format_metric_output_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = drop_pcr_columns(df.copy())
    for col in METRIC_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    bio_cols = ["Isolated labels", "KMeans NMI", "KMeans ARI", "Silhouette label", "cLISI"]
    batch_cols = ["Silhouette batch", "iLISI", "KBET", "Graph connectivity"]
    bio = df[bio_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    batch = df[batch_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    df["Bio conservation"] = pd.to_numeric(df["Bio conservation"], errors="coerce").fillna(bio)
    df["Batch correction"] = pd.to_numeric(df["Batch correction"], errors="coerce").fillna(batch)
    return df[METRIC_OUTPUT_COLUMNS]


def pick_celltype_col(obs: pd.DataFrame):
    for col in ["cell_type", "celltype", "cell_type_ontology_term_id", "celltype.l2"]:
        if col in obs.columns:
            return col
    return None


def load_merged_tissue(tissue_dir: str, tissue_name: str, gene_set: str):
    files = sorted(Path(tissue_dir).glob(f"{tissue_name}_*_{gene_set}.h5ad"))
    if not files:
        raise FileNotFoundError(f"No files found for {tissue_name}/{gene_set} in {tissue_dir}")

    adata_list = []
    for fp in files:
        ad = sc.read_h5ad(fp)
        donor_id = fp.stem.replace(f"{tissue_name}_", "").replace(f"_{gene_set}", "")
        if "donor_id" not in ad.obs.columns:
            ad.obs["donor_id"] = donor_id
        adata_list.append(ad)

    adata = sc.concat(adata_list, join="outer", index_unique=None)
    return adata


def load_merged_manifest_dataset(df: pd.DataFrame, dataset_name: str):
    sub = df[df["dataset"].astype(str) == str(dataset_name)]
    if sub.empty:
        raise ValueError(f"No rows for dataset={dataset_name}")
    adata_list = []
    for _, row in sub.iterrows():
        fp = Path(str(row["path"]))
        ad = sc.read_h5ad(fp)
        if "source_file" not in ad.obs.columns:
            ad.obs["source_file"] = fp.name
        adata_list.append(ad)
    if len(adata_list) == 1:
        return adata_list[0].copy()
    return sc.concat(adata_list, join="outer", index_unique=None)


def maybe_subsample(adata, max_cells: int, seed: int):
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
    idx.sort()
    return adata[idx].copy()


def ensure_unintegrated_pca(adata, key: str = "X_pca", n_comps: int = 50):
    if key in adata.obsm:
        return
    n_comps = max(2, min(int(n_comps), adata.n_obs - 1, adata.n_vars - 1))
    if n_comps < 2:
        return
    sc.pp.pca(adata, n_comps=n_comps)


def run_one_metric(name, fn):
    try:
        return float(fn()), ""
    except Exception as e:
        return np.nan, str(e)


def add_kmeans_metrics(metrics, errors, adata_int, embed_key: str, label_key: str, seed: int):
    try:
        labels = adata_int.obs[label_key].astype(str).values
        n_clusters = int(pd.Series(labels).nunique())
        if n_clusters < 2:
            raise ValueError(f"Need >=2 labels for KMeans metrics, got {n_clusters}")
        emb = np.asarray(adata_int.obsm[embed_key], dtype=np.float32)
        pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(seed)).fit_predict(emb)
        metrics["KMeans_NMI"] = float(normalized_mutual_info_score(labels, pred))
        metrics["KMeans_ARI"] = float(adjusted_rand_score(labels, pred))
        errors["KMeans_NMI"] = ""
        errors["KMeans_ARI"] = ""
    except Exception as e:
        metrics["KMeans_NMI"] = np.nan
        metrics["KMeans_ARI"] = np.nan
        errors["KMeans_NMI"] = str(e)
        errors["KMeans_ARI"] = str(e)


def _kbet_python_fallback(adata_int, embed_key: str, batch_key: str, label_key: str, alpha: float = 0.05):
    emb_all = np.asarray(adata_int.obsm[embed_key], dtype=np.float32)
    labels = adata_int.obs[label_key].astype(str).values
    batches = adata_int.obs[batch_key].astype(str).values
    label_scores = []

    for label in pd.unique(labels):
        idx = np.flatnonzero(labels == label)
        if idx.size < 10:
            continue
        batch_sub = batches[idx]
        batch_counts = pd.Series(batch_sub).value_counts()
        if batch_counts.size <= 1:
            continue
        k0 = int(min(70, max(10, np.floor(batch_counts.mean() / 4.0))))
        if idx.size <= k0:
            continue

        emb = emb_all[idx]
        nn = NearestNeighbors(n_neighbors=k0 + 1, metric="euclidean")
        nn.fit(emb)
        neigh = nn.kneighbors(emb, return_distance=False)[:, 1:]

        batch_levels = batch_counts.index.to_numpy()
        probs = (batch_counts / batch_counts.sum()).reindex(batch_levels).to_numpy(dtype=float)
        rejections = []
        for row in neigh:
            local = pd.Series(batch_sub[row]).value_counts().reindex(batch_levels, fill_value=0).to_numpy(dtype=float)
            expected = probs * local.sum()
            keep = expected > 0
            if keep.sum() <= 1:
                continue
            _, pval = chisquare(f_obs=local[keep], f_exp=expected[keep])
            rejections.append(float(pval < alpha))

        if rejections:
            label_scores.append(float(np.mean(rejections)))

    if not label_scores:
        raise ValueError("No label group had enough cells and >=2 batches for Python kBET fallback.")
    # scIB reports scaled kBET as 1 - rejection rate; higher is better.
    return 1.0 - float(np.mean(label_scores))


def add_kbet_metrics(metrics, errors, adata_int, embed_key: str, batch_key: str, label_key: str):
    try:
        metrics["kBET"] = float(scib.metrics.kBET(adata_int, batch_key=batch_key, label_key=label_key, type_="knn"))
        metrics["kBET_backend"] = "scib_r"
        errors["kBET"] = ""
    except Exception as e:
        try:
            metrics["kBET"] = float(_kbet_python_fallback(adata_int, embed_key=embed_key, batch_key=batch_key, label_key=label_key))
            metrics["kBET_backend"] = "python_fallback"
            errors["kBET"] = f"scib/r kBET failed; used python fallback: {e}"
        except Exception as e2:
            metrics["kBET"] = np.nan
            metrics["kBET_backend"] = "failed"
            errors["kBET"] = f"scib/r kBET failed: {e}; python fallback failed: {e2}"


def add_scib_score_aliases(metrics):
    alias_map = {
        "Isolated labels": "isolated_ASW",
        "KMeans NMI": "KMeans_NMI",
        "KMeans ARI": "KMeans_ARI",
        "Silhouette label": "ASW_label",
        "cLISI": "cLISI",
        "Silhouette batch": "ASW_batch",
        "iLISI": "iLISI",
        "KBET": "kBET",
        "Graph connectivity": "graph_conn",
    }
    for display_name, source_name in alias_map.items():
        metrics[display_name] = metrics.get(source_name, np.nan)

    bio_vals = [metrics.get(k, np.nan) for k, t in SCIB_SCORE_TABLE[:5]]
    batch_vals = [metrics.get(k, np.nan) for k, t in SCIB_SCORE_TABLE[5:9]]
    metrics["Bio conservation"] = float(np.nanmean(bio_vals)) if np.isfinite(bio_vals).any() else np.nan
    metrics["Batch correction"] = float(np.nanmean(batch_vals)) if np.isfinite(batch_vals).any() else np.nan
    if np.isfinite(metrics["Bio conservation"]) and np.isfinite(metrics["Batch correction"]):
        metrics["Total"] = 0.6 * metrics["Bio conservation"] + 0.4 * metrics["Batch correction"]
    else:
        metrics["Total"] = np.nan


def write_score_table(df: pd.DataFrame, out_csv: str):
    if df.empty:
        return
    rows = []
    for metric_name, metric_type in SCIB_SCORE_TABLE:
        row = {"Embedding": metric_name, "Metric Type": metric_type}
        for _, rec in df.iterrows():
            if df["tissue"].nunique() > 1 or df["gene_set"].nunique() > 1:
                col = f"{rec['tissue']}/{rec['gene_set']}/{rec['model']}"
            else:
                col = str(rec["model"])
            row[col] = rec.get(metric_name, np.nan)
        rows.append(row)
    score_df = pd.DataFrame(rows)
    if df["tissue"].nunique() > 1 or df["gene_set"].nunique() > 1:
        metric_cols = [f"{r['tissue']}/{r['gene_set']}/{r['model']}" for _, r in df.iterrows()]
    else:
        metric_cols = [str(x) for x in df["model"].tolist()]
    cols = ["Embedding"] + metric_cols + ["Metric Type"]
    score_df = score_df[[c for c in cols if c in score_df.columns]]
    score_df.to_csv(out_csv, index=False)
    write_score_table_png(score_df, out_csv.replace(".csv", ".png"))


def _score_col_group(col: str) -> str:
    parts = str(col).split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2])
    return "single"


def write_score_table_png(score_df: pd.DataFrame, out_png: str):
    if score_df.empty:
        return
    value_cols = [c for c in score_df.columns if c not in {"Embedding", "Metric Type"}]
    if not value_cols:
        return
    cell_text = []
    bold_cells = set()
    for ridx, row in score_df.iterrows():
        row_vals = []
        for col in score_df.columns:
            val = row.get(col, "")
            if col in value_cols:
                try:
                    fval = float(val)
                    row_vals.append("" if not np.isfinite(fval) else f"{fval:.4f}")
                except Exception:
                    row_vals.append("")
            else:
                row_vals.append(str(val))
        for group in sorted({_score_col_group(c) for c in value_cols}):
            cols_g = [c for c in value_cols if _score_col_group(c) == group]
            vals = pd.to_numeric(row[cols_g], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if vals.empty:
                continue
            best = float(vals.max())
            for col in cols_g:
                try:
                    if np.isfinite(float(row[col])) and np.isclose(float(row[col]), best, rtol=1e-8, atol=1e-12):
                        bold_cells.add((ridx + 1, score_df.columns.get_loc(col)))
                except Exception:
                    pass
        cell_text.append(row_vals)

    n_rows = len(score_df)
    n_cols = len(score_df.columns)
    fig_w = max(10.0, 1.15 * n_cols)
    fig_h = max(3.5, 0.42 * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=list(score_df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_weight("bold")
        elif col in (0, n_cols - 1):
            cell.set_facecolor("#f8f8f8")
        if (row, col) in bold_cells:
            cell.get_text().set_weight("bold")
            cell.set_facecolor("#fff2b3")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _flatten_metric_output(obj):
    flat = {}
    if obj is None:
        return flat
    if isinstance(obj, pd.DataFrame):
        if obj.shape[0] == 1:
            for k, v in obj.iloc[0].items():
                if pd.api.types.is_number(v):
                    flat[str(k)] = float(v)
        elif obj.shape[1] == 1:
            col = obj.columns[0]
            for k, v in obj[col].items():
                if pd.api.types.is_number(v):
                    flat[str(k)] = float(v)
        else:
            for ridx, row in obj.iterrows():
                for col, v in row.items():
                    if pd.api.types.is_number(v):
                        flat[f"{ridx}_{col}"] = float(v)
    elif isinstance(obj, pd.Series):
        for k, v in obj.items():
            if pd.api.types.is_number(v):
                flat[str(k)] = float(v)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                flat[str(k)] = float(v)
    return flat


def _make_scib_pair(adata, embed_key: str):
    adata_pre = adata.copy()
    adata_int = adata.copy()
    emb = np.asarray(adata.obsm[embed_key], dtype=np.float32)
    if not np.isfinite(emb).all():
        raise ValueError(f"{embed_key} contains non-finite values after filtering.")
    adata_int.obsm[embed_key] = emb
    return adata_pre, adata_int


def _ensure_neighbors(adata_int, embed_key: str, neighbors_k: int):
    sc.pp.neighbors(adata_int, use_rep=embed_key, n_neighbors=neighbors_k)


def _drop_nonfinite_embedding_rows(adata, embed_key: str, context: str):
    emb = np.asarray(adata.obsm[embed_key], dtype=np.float32)
    finite_rows = np.isfinite(emb).all(axis=1)
    if finite_rows.all():
        return adata
    n_bad = int((~finite_rows).sum())
    bad_values = int((~np.isfinite(emb)).sum())
    print(
        f"[WARN] {context}: dropping {n_bad}/{adata.n_obs} cells with non-finite "
        f"{embed_key} values ({bad_values} bad values)."
    )
    if int(finite_rows.sum()) < 2:
        raise ValueError(f"{context}: fewer than 2 cells remain after dropping non-finite {embed_key}.")
    out = adata[finite_rows].copy()
    out.obsm[embed_key] = emb[finite_rows]
    return out


def _ensure_cluster(adata_int, cluster_key: str = "scib_leiden"):
    if cluster_key in adata_int.obs.columns:
        return cluster_key
    sc.tl.leiden(adata_int, resolution=1.0, key_added=cluster_key)
    return cluster_key


def eval_one_embedding(
    adata,
    embed_key: str,
    celltype_col: str,
    batch_col: str,
    neighbors_k: int,
    run_official_metrics_all: bool = True,
    scib_n_cores: int = 1,
    scib_subsample: float = 0.5,
    scib_organism: str = "human",
    seed: int = 42,
):
    """
    Comprehensive scIB diagnosis.

    It first attempts the official scIB metrics() call with all metric flags enabled.
    Because some official metrics require optional annotations (cell-cycle genes,
    pseudotime, HVG metadata) or slow dependencies, it then runs robust individual
    metrics and records failures in the error table instead of stopping the run.
    """
    if embed_key not in adata.obsm:
        return None

    adata = _drop_nonfinite_embedding_rows(adata, embed_key, context=f"eval {embed_key}")
    adata_pre, adata_int = _make_scib_pair(adata, embed_key)
    _ensure_neighbors(adata_int, embed_key=embed_key, neighbors_k=neighbors_k)

    metrics = {}
    errors = {}

    if run_official_metrics_all:
        try:
            official = scib.metrics.metrics(
                adata_pre,
                adata_int,
                batch_key=batch_col,
                label_key=celltype_col,
                embed=embed_key,
                cluster_key="scib_cluster",
                ari_=True,
                nmi_=True,
                silhouette_=True,
                pcr_=False,
                cell_cycle_=True,
                organism=scib_organism,
                hvg_score_=True,
                isolated_labels_=True,
                isolated_labels_f1_=True,
                isolated_labels_asw_=True,
                graph_conn_=True,
                trajectory_=True,
                morans_i_=True,
                kBET_=True,
                lisi_graph_=True,
                ilisi_=True,
                clisi_=True,
                subsample=scib_subsample,
                n_cores=scib_n_cores,
                type_="knn",
                verbose=False,
            )
            for k, v in _flatten_metric_output(official).items():
                metrics[f"official_{k}"] = v
            errors["official_metrics_all"] = ""
        except Exception as e:
            errors["official_metrics_all"] = str(e)

    def add_metric(name, fn):
        metrics[name], errors[name] = run_one_metric(name, fn)

    add_metric("ASW_label", lambda: scib.metrics.silhouette(adata_int, label_key=celltype_col, embed=embed_key))
    add_metric(
        "ASW_batch",
        lambda: scib.metrics.silhouette_batch(
            adata_int,
            batch_key=batch_col,
            label_key=celltype_col,
            embed=embed_key,
        ),
    )
    add_metric(
        "isolated_ASW",
        lambda: scib.metrics.isolated_labels_asw(
            adata_int,
            batch_key=batch_col,
            label_key=celltype_col,
            embed=embed_key,
            verbose=False,
        ),
    )
    if hasattr(scib.metrics, "isolated_labels_f1"):
        add_metric(
            "isolated_F1",
            lambda: scib.metrics.isolated_labels_f1(
                adata_int,
                batch_key=batch_col,
                label_key=celltype_col,
                embed=embed_key,
                verbose=False,
            ),
        )
    add_metric("graph_conn", lambda: scib.metrics.graph_connectivity(adata_int, label_key=celltype_col))
    add_metric(
        "iLISI",
        lambda: scib.metrics.ilisi_graph(
            adata_int,
            batch_key=batch_col,
            type_="knn",
            n_cores=scib_n_cores,
        ),
    )
    add_metric(
        "cLISI",
        lambda: scib.metrics.clisi_graph(
            adata_int,
            label_key=celltype_col,
            type_="knn",
            n_cores=scib_n_cores,
        ),
    )
    add_kbet_metrics(metrics, errors, adata_int, embed_key=embed_key, batch_key=batch_col, label_key=celltype_col)
    add_kmeans_metrics(metrics, errors, adata_int, embed_key=embed_key, label_key=celltype_col, seed=seed)
    add_metric("NMI", lambda: scib.metrics.nmi(adata_int, _ensure_cluster(adata_int), celltype_col))
    add_metric("ARI", lambda: scib.metrics.ari(adata_int, _ensure_cluster(adata_int), celltype_col))

    add_scib_score_aliases(metrics)
    return metrics, errors


def copy_existing_timing_csv(old_eval_dir: str, out_dir: str):
    src = os.path.join(old_eval_dir, "all_models_inference_timing.csv")
    if os.path.exists(src):
        dst = os.path.join(out_dir, "all_models_inference_timing.csv")
        shutil.copy2(src, dst)
        print(f"[copy] {src} -> {dst}")
    else:
        print(f"[skip] timing csv not found: {src}")


def load_cached_metric_rows(cache_path: str, tissue_name: str, gene_set: str) -> pd.DataFrame:
    candidates = [cache_path]
    dfs = []
    for fp in candidates:
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[cache skip] cannot read {fp}: {e}")
            continue
        required = {"tissue", "gene_set", "model", "key"}
        if not required.issubset(df.columns):
            continue
        sub = df[
            (df["tissue"].astype(str) == str(tissue_name))
            & (df["gene_set"].astype(str) == str(gene_set))
        ].copy()
        if not sub.empty:
            dfs.append(sub)
    if not dfs:
        return pd.DataFrame()
    cached = pd.concat(dfs, ignore_index=True)
    cached = format_metric_output_df(cached)
    cached = cached.drop_duplicates(subset=["tissue", "gene_set", "model", "key"], keep="first")
    return cached


def cached_row_for(cached: pd.DataFrame, tissue_name: str, gene_set: str, model_name: str, key: str):
    if cached.empty:
        return None
    sub = cached[
        (cached["tissue"].astype(str) == str(tissue_name))
        & (cached["gene_set"].astype(str) == str(gene_set))
        & (cached["model"].astype(str) == str(model_name))
        & (cached["key"].astype(str) == str(key))
    ]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def update_metric_cache(cache_path: str, rows: list):
    if not rows:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    new_df = format_metric_output_df(pd.DataFrame(rows))
    if os.path.exists(cache_path):
        try:
            old_df = format_metric_output_df(pd.read_csv(cache_path))
            df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"[cache warn] cannot read existing cache {cache_path}: {e}")
            df = new_df
    else:
        df = new_df
    df = format_metric_output_df(df)
    df = df.drop_duplicates(subset=["tissue", "gene_set", "model", "key"], keep="last")
    df.to_csv(cache_path, index=False)
    print(f"[cache save] {cache_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    copy_existing_timing_csv(args.old_eval_dir, args.output_dir)

    model_pairs = [(n, (args.gmm_key if k == "xVerse_gmmvae" else k)) for n, k in MODELS]

    all_rows = []
    all_err_rows = []
    if args.dataset_manifest:
        manifest = pd.read_csv(args.dataset_manifest)
        required = {"dataset", "path", "batch_key", "label_key"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"dataset manifest missing columns: {sorted(missing)}")
        dataset_jobs = []
        for dataset_name in manifest["dataset"].astype(str).unique():
            sub = manifest[manifest["dataset"].astype(str) == dataset_name]
            dataset_jobs.append(
                {
                    "tissue_name": dataset_name,
                    "gene_set": "gold",
                    "adata": load_merged_manifest_dataset(manifest, dataset_name),
                    "batch_col": str(sub.iloc[0]["batch_key"]),
                    "celltype_col": str(sub.iloc[0]["label_key"]),
                    "cache_path": str(Path(str(sub.iloc[0]["path"])).parent / "scib_metric_cache_gold.csv"),
                }
            )
    else:
        dataset_jobs = []
        for tissue_name, tissue_dir in [("liver", args.liver_dir), ("brain", args.brain_dir)]:
            dataset_jobs.append(
                {
                    "tissue_name": tissue_name,
                    "gene_set": "all",
                    "adata": load_merged_tissue(tissue_dir, tissue_name, "all"),
                    "batch_col": "donor_id",
                    "celltype_col": None,
                    "cache_path": str(Path(tissue_dir) / "scib_metric_cache_all.csv"),
                }
            )

    for job in dataset_jobs:
            tissue_name = job["tissue_name"]
            gene_set = job["gene_set"]
            print(f"\n=== {tissue_name}/{gene_set} ===")
            adata = job["adata"]
            adata = maybe_subsample(adata, args.max_cells, args.seed)
            ensure_unintegrated_pca(adata, key="X_pca")
            cache_path = job["cache_path"]
            cached_metrics = load_cached_metric_rows(cache_path, tissue_name, gene_set)
            new_cache_rows = []

            celltype_col = job["celltype_col"] or pick_celltype_col(adata.obs)
            if celltype_col is None:
                print(f"[skip] no cell type column for {tissue_name}/{gene_set}")
                continue
            batch_col = job["batch_col"]
            if batch_col not in adata.obs.columns:
                print(f"[skip] no batch column {batch_col} for {tissue_name}/{gene_set}")
                continue

            for model_name, key in model_pairs:
                cached = None if model_name == "GMVAE" else cached_row_for(cached_metrics, tissue_name, gene_set, model_name, key)
                if cached is not None:
                    print(f"[cache] {model_name} key={key}")
                    all_rows.append(cached)
                    continue
                if key not in adata.obsm:
                    print(f"[skip] {model_name} key={key} missing")
                    continue
                print(f"[eval] {model_name} key={key} n={adata.n_obs}")
                out = eval_one_embedding(
                    adata=adata,
                    embed_key=key,
                    celltype_col=celltype_col,
                    batch_col=batch_col,
                    neighbors_k=args.neighbors_k,
                    run_official_metrics_all=(not args.skip_official_metrics_all),
                    scib_n_cores=args.scib_n_cores,
                    scib_subsample=args.scib_subsample,
                    scib_organism=args.scib_organism,
                    seed=args.seed,
                )
                if out is None:
                    continue
                metrics, errors = out

                row = {
                    "tissue": tissue_name,
                    "gene_set": gene_set,
                    "model": model_name,
                    "key": key,
                    "n_cells": int(adata.n_obs),
                }
                row.update(metrics)
                all_rows.append(row)
                new_cache_rows.append(row)

                erow = {
                    "tissue": tissue_name,
                    "gene_set": gene_set,
                    "model": model_name,
                    "key": key,
                }
                erow.update(errors)
                all_err_rows.append(erow)

            df_tg = pd.DataFrame([r for r in all_rows if r["tissue"] == tissue_name and r["gene_set"] == gene_set])
            df_tg = format_metric_output_df(df_tg)
            if not df_tg.empty:
                out_csv = os.path.join(args.output_dir, f"{tissue_name}_{gene_set}_scib_full_metrics.csv")
                df_tg.to_csv(out_csv, index=False)
                print(f"[save] {out_csv}")
                score_csv = os.path.join(args.output_dir, f"{tissue_name}_{gene_set}_scib_score_table.csv")
                write_score_table(df_tg, score_csv)
                print(f"[save] {score_csv}")
                update_metric_cache(cache_path, new_cache_rows)

    df_all = format_metric_output_df(pd.DataFrame(all_rows))
    df_err = drop_pcr_columns(pd.DataFrame(all_err_rows))
    all_csv = os.path.join(args.output_dir, "all_scib_full_metrics.csv")
    err_csv = os.path.join(args.output_dir, "all_scib_metric_errors.csv")
    score_csv = os.path.join(args.output_dir, "all_scib_score_table.csv")
    df_all.to_csv(all_csv, index=False)
    df_err.to_csv(err_csv, index=False)
    write_score_table(df_all, score_csv)
    print(f"\n[done] metrics: {all_csv}")
    print(f"[done] errors:  {err_csv}")
    print(f"[done] scores:  {score_csv}")


if __name__ == "__main__":
    main()
