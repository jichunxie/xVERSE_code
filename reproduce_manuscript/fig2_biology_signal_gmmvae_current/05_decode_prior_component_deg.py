#!/usr/bin/env python3
"""
Decode active MFA prior components and optionally compute component-vs-rest DEGs.

Outputs:
- active_components.csv
- decoded_center_top_genes.csv
- factor_direction_top_genes.csv
- cell_component_assignment_summary.csv, if h5ad input is provided
- deg_component_*.csv, if h5ad input is provided
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm


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
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-family", default="main_mfa", choices=["auto", "main_energy", "main_mfa"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--active-min-pi", type=float, default=0.02)
    ap.add_argument("--active-top-k", type=int, default=0, help="Fallback/use top-K components by pi. 0 disables.")
    ap.add_argument("--top-genes", type=int, default=100)
    ap.add_argument("--factor-steps", default="-2,-1,1,2", help="Comma-separated factor traversal strengths.")
    ap.add_argument("--h5ad", action="append", default=[], help="h5ad path. Can be repeated.")
    ap.add_argument("--h5ad-dir", action="append", default=[], help="Directory containing *_all.h5ad files. Can be repeated.")
    ap.add_argument("--h5ad-glob", default="*_all.h5ad")
    ap.add_argument("--gene-id-col", default="gene_ids")
    ap.add_argument("--count-layer", default="", help="Layer to use for DEG counts. Empty uses raw.X if present else X.")
    ap.add_argument("--max-cells", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--assign-min-prob", type=float, default=0.0)
    ap.add_argument("--min-cells-per-component", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gene_symbols_from_adata(adata, gene_id_col: str):
    ids = None
    if gene_id_col and gene_id_col in adata.var.columns:
        ids = adata.var[gene_id_col].astype(str).values
    else:
        for col in ["gene_ids", "gene_id", "ensembl_id", "ensembl_ids", "feature_id"]:
            if col in adata.var.columns:
                ids = adata.var[col].astype(str).values
                break
    if ids is None:
        ids = adata.var_names.astype(str).values
    symbols = adata.var_names.astype(str).values
    return dict(zip(ids, symbols))


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
    rate_all, _, lib_all = decode_z(model, mu[active])
    rate_np = rate_all.cpu().numpy()
    for j, k in enumerate(active):
        center_rows.extend(top_rows(rate_np[j], gene_ids, gene_symbols, int(k), "center_rate", top_n))
        active_rows.append(
            {
                "component": int(k),
                "pi": float(pi[k]),
                "library_size_decoded": float(lib_all[j].item()),
                "mu_norm": float(mu[k].norm().item()),
            }
        )

    if factor is not None:
        for k in active:
            for r in range(factor.shape[2]):
                direction = factor[k, :, r]
                strength = float(direction.norm().item())
                for step in factor_steps:
                    z = (mu[k] + float(step) * direction).unsqueeze(0)
                    rate, _, lib = decode_z(model, z)
                    vals = rate.squeeze(0).cpu().numpy()
                    rows = top_rows(vals, gene_ids, gene_symbols, int(k), f"factor{r}_step{step:g}_rate", top_n)
                    for row in rows:
                        row["factor"] = int(r)
                        row["step"] = float(step)
                        row["factor_strength"] = strength
                        row["library_size_decoded"] = float(lib.item())
                    factor_rows.extend(rows)

                z_plus = (mu[k] + direction).unsqueeze(0)
                z_minus = (mu[k] - direction).unsqueeze(0)
                rate_plus, _, _ = decode_z(model, z_plus)
                rate_minus, _, _ = decode_z(model, z_minus)
                diff = (torch.log1p(rate_plus) - torch.log1p(rate_minus)).squeeze(0).cpu().numpy()
                up = top_rows(diff, gene_ids, gene_symbols, int(k), f"factor{r}_plus_vs_minus", top_n)
                down = top_rows(-diff, gene_ids, gene_symbols, int(k), f"factor{r}_minus_vs_plus", top_n)
                for row in up + down:
                    row["factor"] = int(r)
                    row["step"] = np.nan
                    row["factor_strength"] = strength
                factor_rows.extend(up + down)

    pd.DataFrame(active_rows).to_csv(output_dir / "active_components.csv", index=False)
    pd.DataFrame(center_rows).to_csv(output_dir / "decoded_center_top_genes.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(output_dir / "factor_direction_top_genes.csv", index=False)
    return pd.DataFrame(active_rows)


def _gene_ids_from_adata(adata, gene_id_col: str):
    if gene_id_col and gene_id_col in adata.var.columns:
        return adata.var[gene_id_col].astype(str).values
    for col in ["gene_ids", "gene_id", "ensembl_id", "ensembl_ids", "feature_id"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).values
    return adata.var_names.astype(str).values


def _matrix_from_adata(adata, count_layer: str):
    if count_layer and count_layer != "nan" and count_layer in adata.layers:
        return adata.layers[count_layer]
    if count_layer == "raw" and adata.raw is not None:
        return adata.raw.X
    return adata.raw.X if adata.raw is not None else adata.X


@torch.no_grad()
def collect_assignments_and_counts(model, h5ads, gene_ids, gene_id_col, count_layer, max_cells, batch_size, device, seed):
    rng = np.random.default_rng(seed)
    all_counts = []
    all_masks = []
    all_q = []
    gene_symbols = {}

    for fp in h5ads:
        adata = sc.read_h5ad(fp)
        gene_symbols.update(gene_symbols_from_adata(adata, gene_id_col))
        source_gene_ids = _gene_ids_from_adata(adata, gene_id_col)
        gene_to_idx = {str(g): i for i, g in enumerate(source_gene_ids)}
        valid_pos = []
        valid_gene_idx = []
        for pos, g in enumerate(gene_ids):
            idx = gene_to_idx.get(str(g))
            if idx is not None:
                valid_pos.append(pos)
                valid_gene_idx.append(idx)
        if not valid_pos:
            print(f"[WARN] no gene overlap: {fp}")
            continue

        n = adata.n_obs
        take = np.arange(n)
        if max_cells > 0 and n > max_cells:
            take = rng.choice(n, size=max_cells, replace=False)
            take.sort()
        raw = _matrix_from_adata(adata, count_layer)
        for st in tqdm(range(0, len(take), batch_size), desc=f"assign {Path(fp).name}", leave=False):
            idx = take[st : st + batch_size]
            block = raw[idx, :][:, valid_gene_idx]
            arr = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            x_np = np.full((len(idx), len(gene_ids)), -1.0, dtype=np.float32)
            x_np[:, valid_pos] = np.asarray(arr, dtype=np.float32)
            x_count = torch.tensor(np.where(x_np >= 0, x_np, 0.0), dtype=torch.float32, device=device)
            x_mask = torch.tensor((x_np >= 0).astype(np.float32), dtype=torch.float32, device=device)
            out = model(x_count=x_count, x_mask=x_mask, sample_id=None, tissue_id=None, use_batch_condition=False)
            if "q_c" not in out:
                raise RuntimeError("Model output has no q_c; this script expects a mixture posterior.")
            all_q.append(out["q_c"].detach().cpu().numpy())
            all_counts.append(x_count.detach().cpu().numpy())
            all_masks.append(x_mask.detach().cpu().numpy())

    if not all_q:
        return None, None, None, gene_symbols
    return np.concatenate(all_q, axis=0), np.concatenate(all_counts, axis=0), np.concatenate(all_masks, axis=0), gene_symbols


def compute_component_deg(q, counts, masks, gene_ids, gene_symbols, active, output_dir, top_n, min_prob, min_cells):
    top1 = q.argmax(axis=1)
    top_prob = q.max(axis=1)
    summary_rows = []
    logx = np.log1p(counts)
    for k in active:
        in_mask = (top1 == int(k)) & (top_prob >= float(min_prob))
        out_mask = ~in_mask
        n_in = int(in_mask.sum())
        n_out = int(out_mask.sum())
        summary_rows.append(
            {
                "component": int(k),
                "n_top1": int((top1 == int(k)).sum()),
                "n_top1_prob_filtered": n_in,
                "mean_q": float(q[:, int(k)].mean()),
                "mean_top_prob_in": float(top_prob[in_mask].mean()) if n_in else np.nan,
            }
        )
        if n_in < int(min_cells) or n_out < int(min_cells):
            continue

        obs_in = masks[in_mask].sum(axis=0)
        obs_out = masks[out_mask].sum(axis=0)
        mean_in = (logx[in_mask] * masks[in_mask]).sum(axis=0) / np.maximum(obs_in, 1.0)
        mean_out = (logx[out_mask] * masks[out_mask]).sum(axis=0) / np.maximum(obs_out, 1.0)
        det_in = ((counts[in_mask] > 0) * masks[in_mask]).sum(axis=0) / np.maximum(obs_in, 1.0)
        det_out = ((counts[out_mask] > 0) * masks[out_mask]).sum(axis=0) / np.maximum(obs_out, 1.0)
        logfc = mean_in - mean_out
        score = logfc * np.sqrt(np.minimum(obs_in, obs_out) / np.maximum(obs_in + obs_out, 1.0))
        order = np.argsort(-score)
        rows = []
        for rank, gidx in enumerate(order[:top_n], start=1):
            gid = str(gene_ids[gidx])
            rows.append(
                {
                    "component": int(k),
                    "rank": rank,
                    "gene_id": gid,
                    "gene_symbol": gene_symbols.get(gid, gid),
                    "score": float(score[gidx]),
                    "log1p_mean_in": float(mean_in[gidx]),
                    "log1p_mean_out": float(mean_out[gidx]),
                    "log1p_logfc": float(logfc[gidx]),
                    "det_frac_in": float(det_in[gidx]),
                    "det_frac_out": float(det_out[gidx]),
                    "obs_in": int(obs_in[gidx]),
                    "obs_out": int(obs_out[gidx]),
                }
            )
        pd.DataFrame(rows).to_csv(output_dir / f"deg_component_{int(k)}_vs_rest.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(output_dir / "cell_component_assignment_summary.csv", index=False)


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

    h5ads = [Path(p) for p in args.h5ad]
    for d in args.h5ad_dir:
        h5ads.extend(sorted(Path(d).glob(args.h5ad_glob)))
    h5ads = [p for p in h5ads if p.exists()]

    gene_symbols = {str(g): str(g) for g in gene_ids}
    q = counts = masks = None
    if h5ads:
        q, counts, masks, h5ad_gene_symbols = collect_assignments_and_counts(
            model=model,
            h5ads=h5ads,
            gene_ids=gene_ids,
            gene_id_col=args.gene_id_col,
            count_layer=args.count_layer,
            max_cells=args.max_cells,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        gene_symbols.update(h5ad_gene_symbols)

    factor_steps = [float(x) for x in str(args.factor_steps).split(",") if str(x).strip()]
    decode_prior(model, gene_ids, gene_symbols, active, output_dir, args.top_genes, factor_steps)

    if q is not None:
        compute_component_deg(
            q=q,
            counts=counts,
            masks=masks,
            gene_ids=gene_ids,
            gene_symbols=gene_symbols,
            active=active,
            output_dir=output_dir,
            top_n=args.top_genes,
            min_prob=args.assign_min_prob,
            min_cells=args.min_cells_per_component,
        )
    print(f"[Done] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
