#!/usr/bin/env python3
"""
Decode active MFA prior components and factor directions.

This script diagnoses the prior itself only; it does not load cell h5ad files.

Outputs:
- active_components.csv
- decoded_center_top_genes.csv
- decoded_center_vs_rest_top_genes.csv
- decoded_center_log1p_rate_corr.csv
- factor_direction_summary.csv
- factor_direction_delta_top_genes.csv
- factor_direction_delta_corr.csv
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.utils_model import load_gene_ids


def _load_extract_module():
    fp = Path(__file__).resolve().parent / "02_extract_gmmvae_embedding.py"
    spec = importlib.util.spec_from_file_location("extract_gmmvae_embedding", fp)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    ap = argparse.ArgumentParser(description="Decode active MFA prior components and compute assigned-cell DEGs.")
    ap.add_argument("--ckpt", required=True, help="Path to MFA checkpoint.")
    ap.add_argument(
        "--gene-ids-path",
        default="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt",
    )
    ap.add_argument(
        "--gene-symbol-csv",
        default="/hpc/group/xielab/xj58/general/gene_info_table.csv",
        help="CSV mapping Ensembl IDs to gene names. Default uses the shared gene_info_table.csv.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-family", default="main_mfa", choices=["auto", "main_energy", "main_mfa"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--active-min-pi", type=float, default=0.02)
    ap.add_argument("--active-top-k", type=int, default=0, help="Fallback/use top-K components by pi. 0 disables.")
    ap.add_argument("--top-genes", type=int, default=100)
    ap.add_argument("--factor-steps", default="-2,-1,1,2", help="Comma-separated factor traversal strengths.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gene_symbols_from_csv(path: str):
    if not path:
        return {}
    fp = Path(path)
    if not fp.exists():
        print(f"[WARN] gene symbol csv not found: {fp}")
        return {}
    df = pd.read_csv(fp)
    id_col = None
    name_col = None
    for col in ["ensembl_id", "gene_id", "gene_ids", "ensg"]:
        if col in df.columns:
            id_col = col
            break
    for col in ["gene_name", "gene_symbol", "symbol", "name"]:
        if col in df.columns:
            name_col = col
            break
    if id_col is None or name_col is None:
        print(f"[WARN] gene symbol csv missing id/name columns: {fp}")
        return {}
    mapping = (
        df[[id_col, name_col]]
        .dropna()
        .astype(str)
        .drop_duplicates(subset=[id_col], keep="first")
        .set_index(id_col)[name_col]
        .to_dict()
    )
    print(f"[GeneMap] loaded {len(mapping)} gene symbols from {fp}")
    return mapping


def select_active_components(pi: np.ndarray, min_pi: float, top_k: int):
    active = np.where(pi >= float(min_pi))[0]
    if top_k > 0:
        top = np.argsort(-pi)[: min(int(top_k), pi.size)]
        active = np.array(sorted(set(active.tolist()) | set(top.tolist())), dtype=int)
    if active.size == 0:
        active = np.argsort(-pi)[: min(10, pi.size)]
    return np.asarray(active, dtype=int)


@torch.no_grad()
def decode_z(model, z: torch.Tensor):
    gene_logits = model.decoder(z, cond=None)
    library_size = torch.nn.functional.softplus(model.library_head(z)) + 1e-8
    gene_probs = torch.softmax(gene_logits.float(), dim=-1)
    rate = gene_probs * library_size.float()
    return rate.float(), gene_logits.float(), library_size.float()


def top_rows(values, gene_ids, gene_symbols, component, name, top_n):
    order = np.argsort(-values)[:top_n]
    rows = []
    for rank, idx in enumerate(order, start=1):
        gid = str(gene_ids[idx])
        rows.append(
            {
                "component": int(component),
                "name": name,
                "rank": rank,
                "gene_id": gid,
                "gene_symbol": gene_symbols.get(gid, gid),
                "score": float(values[idx]),
            }
        )
    return rows


def _safe_corrcoef(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.eye(x.shape[0], dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(denom, 1e-12, None)
    return np.clip(x @ x.T, -1.0, 1.0)


def _write_gene_matrix(path: Path, matrix: np.ndarray, row_names, gene_ids, gene_symbols):
    df = pd.DataFrame(np.asarray(matrix, dtype=np.float32), columns=[str(g) for g in gene_ids])
    df.insert(0, "name", list(row_names))
    df.to_csv(path, index=False)
    symbol_df = pd.DataFrame(
        {
            "gene_id": [str(g) for g in gene_ids],
            "gene_symbol": [gene_symbols.get(str(g), str(g)) for g in gene_ids],
        }
    )
    symbol_df.to_csv(path.with_name(path.stem + "_genes.csv"), index=False)


def decode_prior(model, gene_ids, gene_symbols, active, output_dir: Path, top_n: int, factor_steps):
    prior = model.prior
    device = next(model.parameters()).device
    mu = prior.prior_mu.detach().float().to(device)
    pi = torch.softmax(prior.pi_logits.detach().float(), dim=0).cpu().numpy()
    factor = prior._expanded_factor()
    factor = None if factor is None else factor.detach().float().to(device)

    active_rows = []
    center_rows = []
    factor_rows = []
    center_rate_all, center_logits_all, lib_all = decode_z(model, mu[active])
    center_rate_np = center_rate_all.cpu().numpy()
    center_lograte_np = np.log1p(center_rate_np)
    center_logits_np = center_logits_all.cpu().numpy()
    center_names = [f"component_{int(k)}" for k in active]
    _write_gene_matrix(output_dir / "decoded_center_log1p_rate_matrix.csv", center_lograte_np, center_names, gene_ids, gene_symbols)
    _write_gene_matrix(output_dir / "decoded_center_gene_logits_matrix.csv", center_logits_np, center_names, gene_ids, gene_symbols)

    center_corr = _safe_corrcoef(center_lograte_np)
    pd.DataFrame(center_corr, index=active, columns=active).to_csv(output_dir / "decoded_center_log1p_rate_corr.csv")

    center_vs_rest_rows = []
    center_vs_rest_delta = []
    for j, k in enumerate(active):
        if len(active) > 1:
            rest = np.delete(center_lograte_np, j, axis=0).mean(axis=0)
        else:
            rest = np.zeros_like(center_lograte_np[j])
        delta = center_lograte_np[j] - rest
        center_vs_rest_delta.append(delta)
        for rank, gidx in enumerate(np.argsort(-np.abs(delta))[:top_n], start=1):
            gid = str(gene_ids[gidx])
            center_vs_rest_rows.append(
                {
                    "component": int(k),
                    "rank_abs_delta": rank,
                    "gene_id": gid,
                    "gene_symbol": gene_symbols.get(gid, gid),
                    "delta_log1p_rate_vs_other_centers": float(delta[gidx]),
                    "center_log1p_rate": float(center_lograte_np[j, gidx]),
                    "other_centers_mean_log1p_rate": float(rest[gidx]),
                }
            )
    pd.DataFrame(center_vs_rest_rows).to_csv(output_dir / "decoded_center_vs_rest_top_genes.csv", index=False)
    if center_vs_rest_delta:
        _write_gene_matrix(
            output_dir / "decoded_center_vs_rest_delta_matrix.csv",
            np.stack(center_vs_rest_delta, axis=0),
            center_names,
            gene_ids,
            gene_symbols,
        )

    for j, k in enumerate(active):
        center_rows.extend(top_rows(center_lograte_np[j], gene_ids, gene_symbols, int(k), "center_log1p_rate", top_n))
        center_rows.extend(top_rows(center_logits_np[j], gene_ids, gene_symbols, int(k), "center_gene_logits", top_n))
        active_rows.append(
            {
                "component": int(k),
                "pi": float(pi[k]),
                "library_size_decoded": float(lib_all[j].item()),
                "mu_norm": float(mu[k].norm().item()),
            }
        )

    factor_summary = []
    factor_delta_rows = []
    factor_delta_vectors = []
    factor_delta_names = []
    if factor is not None:
        for k in active:
            for r in range(factor.shape[2]):
                direction = factor[k, :, r]
                strength = float(direction.norm().item())
                z_plus = (mu[k] + direction).unsqueeze(0)
                z_minus = (mu[k] - direction).unsqueeze(0)
                z_center = mu[k].unsqueeze(0)
                rate_plus, logits_plus, lib_plus = decode_z(model, z_plus)
                rate_minus, logits_minus, lib_minus = decode_z(model, z_minus)
                rate_center, logits_center, lib_center = decode_z(model, z_center)
                lograte_plus = torch.log1p(rate_plus).squeeze(0).cpu().numpy()
                lograte_minus = torch.log1p(rate_minus).squeeze(0).cpu().numpy()
                lograte_center = torch.log1p(rate_center).squeeze(0).cpu().numpy()
                delta_lograte = lograte_plus - lograte_minus
                delta_logits = (logits_plus - logits_minus).squeeze(0).cpu().numpy()
                factor_delta_vectors.append(delta_lograte)
                factor_delta_names.append(f"component_{int(k)}_factor_{int(r)}")
                factor_summary.append(
                    {
                        "component": int(k),
                        "factor": int(r),
                        "factor_strength": strength,
                        "library_center": float(lib_center.item()),
                        "library_plus": float(lib_plus.item()),
                        "library_minus": float(lib_minus.item()),
                        "delta_log1p_l2": float(np.linalg.norm(delta_lograte)),
                        "delta_log1p_abs_mean": float(np.mean(np.abs(delta_lograte))),
                        "delta_logits_l2": float(np.linalg.norm(delta_logits)),
                        "top_up_genes": ";".join(
                            gene_symbols.get(str(gene_ids[i]), str(gene_ids[i]))
                            for i in np.argsort(-delta_lograte)[: min(10, len(gene_ids))]
                        ),
                        "top_down_genes": ";".join(
                            gene_symbols.get(str(gene_ids[i]), str(gene_ids[i]))
                            for i in np.argsort(delta_lograte)[: min(10, len(gene_ids))]
                        ),
                    }
                )
                for step in factor_steps:
                    z = (mu[k] + float(step) * direction).unsqueeze(0)
                    rate, _, lib = decode_z(model, z)
                    vals = torch.log1p(rate).squeeze(0).cpu().numpy()
                    rows = top_rows(vals, gene_ids, gene_symbols, int(k), f"factor{r}_step{step:g}_log1p_rate", top_n)
                    for row in rows:
                        row["factor"] = int(r)
                        row["step"] = float(step)
                        row["factor_strength"] = strength
                        row["library_size_decoded"] = float(lib.item())
                    factor_rows.extend(rows)

                up = top_rows(delta_lograte, gene_ids, gene_symbols, int(k), f"factor{r}_plus_vs_minus_log1p_rate", top_n)
                down = top_rows(-delta_lograte, gene_ids, gene_symbols, int(k), f"factor{r}_minus_vs_plus_log1p_rate", top_n)
                for row in up + down:
                    row["factor"] = int(r)
                    row["step"] = np.nan
                    row["factor_strength"] = strength
                factor_rows.extend(up + down)
                for rank, gidx in enumerate(np.argsort(-np.abs(delta_lograte))[:top_n], start=1):
                    gid = str(gene_ids[gidx])
                    factor_delta_rows.append(
                        {
                            "component": int(k),
                            "factor": int(r),
                            "rank_abs_delta": rank,
                            "gene_id": gid,
                            "gene_symbol": gene_symbols.get(gid, gid),
                            "delta_log1p_rate_plus_minus": float(delta_lograte[gidx]),
                            "delta_gene_logits_plus_minus": float(delta_logits[gidx]),
                            "center_log1p_rate": float(lograte_center[gidx]),
                            "plus_log1p_rate": float(lograte_plus[gidx]),
                            "minus_log1p_rate": float(lograte_minus[gidx]),
                            "factor_strength": strength,
                        }
                    )

    pd.DataFrame(active_rows).to_csv(output_dir / "active_components.csv", index=False)
    pd.DataFrame(center_rows).to_csv(output_dir / "decoded_center_top_genes.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(output_dir / "factor_direction_top_genes.csv", index=False)
    pd.DataFrame(factor_summary).to_csv(output_dir / "factor_direction_summary.csv", index=False)
    pd.DataFrame(factor_delta_rows).to_csv(output_dir / "factor_direction_delta_top_genes.csv", index=False)
    if factor_delta_vectors:
        delta_mat = np.stack(factor_delta_vectors, axis=0)
        _write_gene_matrix(output_dir / "factor_direction_delta_log1p_rate_matrix.csv", delta_mat, factor_delta_names, gene_ids, gene_symbols)
        delta_corr = _safe_corrcoef(delta_mat)
        pd.DataFrame(delta_corr, index=factor_delta_names, columns=factor_delta_names).to_csv(
            output_dir / "factor_direction_delta_corr.csv"
        )
    return pd.DataFrame(active_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    gene_ids = load_gene_ids(args.gene_ids_path)

    extract_mod = _load_extract_module()
    extract_mod.build_model_from_ckpt._requested_family = args.model_family
    model, load_ret = extract_mod.build_model_from_ckpt(args.ckpt, device)
    print(f"[Load] missing={len(load_ret.missing_keys)} unexpected={len(load_ret.unexpected_keys)}")
    model.eval()

    prior = model.prior
    pi = torch.softmax(prior.pi_logits.detach().float(), dim=0).cpu().numpy()
    active = select_active_components(pi, args.active_min_pi, args.active_top_k)
    print(f"[Active] {len(active)}/{len(pi)} components: {active.tolist()}")

    gene_symbols = {str(g): str(g) for g in gene_ids}
    gene_symbols.update(gene_symbols_from_csv(args.gene_symbol_csv))

    factor_steps = [float(x) for x in str(args.factor_steps).split(",") if str(x).strip()]
    decode_prior(model, gene_ids, gene_symbols, active, output_dir, args.top_genes, factor_steps)

    print(f"[Done] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
