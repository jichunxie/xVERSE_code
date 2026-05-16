#!/usr/bin/env python3
import argparse
import math
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
KEYVAL_RE = re.compile(rf"([A-Za-z][A-Za-z0-9_]*)=({FLOAT_RE})")
BATCH_RE = re.compile(r"^\[Batch\s+(\d+)\]")
VAL_BATCH_RE = re.compile(r"^\[Val Batch\s+(\d+)\]")
EPOCH_STAGE_RE = re.compile(r"^\[Epoch\s+(\d+)\]\s+stage=([A-Za-z0-9_]+)")
EPOCH_VAL_RE = re.compile(r"^\[Epoch\s+(\d+)\]\s+Validation Loss:")


@dataclass
class Point:
    step: int
    values: Dict[str, float]


def _parse_keyvals(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in KEYVAL_RE.findall(line):
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _slope(y: List[float]) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    sx = (n - 1) * n / 2.0
    sy = float(sum(y))
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sxy = float(sum(i * v for i, v in enumerate(y)))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _tail(points: List[Point], key: str, n: int) -> List[float]:
    vals = [p.values[key] for p in points if key in p.values]
    if n > 0:
        vals = vals[-n:]
    return vals


def _latest(points: List[Point], key: str) -> Optional[float]:
    for p in reversed(points):
        if key in p.values:
            return p.values[key]
    return None


def diagnose(batch_points: List[Point], val_points: List[Point], tail_n: int) -> List[str]:
    msgs: List[str] = []
    k_eff_vals = _tail(batch_points, "K_eff", tail_n)
    active_vals = _tail(batch_points, "activeK", tail_n)
    resp_top1_batch_vals = _tail(batch_points, "respTop1Batch", tail_n)
    resp_top1_vals = _tail(batch_points, "respTop1", tail_n)
    min_mu_vals = _tail(batch_points, "minMuDist", tail_n)
    pih_vals = _tail(batch_points, "piH", tail_n)
    val_loss_vals = _tail(val_points, "Loss", tail_n)

    k_eff_latest = k_eff_vals[-1] if k_eff_vals else None
    active_latest = active_vals[-1] if active_vals else None
    top1b_latest = resp_top1_batch_vals[-1] if resp_top1_batch_vals else None
    top1_latest = resp_top1_vals[-1] if resp_top1_vals else None
    min_mu_latest = min_mu_vals[-1] if min_mu_vals else None
    pih_latest = pih_vals[-1] if pih_vals else None
    val_loss_latest = val_loss_vals[-1] if val_loss_vals else None

    # Infer K from entropy if possible.
    inferred_k = None
    if pih_latest is not None:
        inferred_k = max(1, int(round(math.exp(max(0.0, pih_latest)))))

    msgs.append("Latest Snapshot")
    msgs.append(
        f"- K_eff={k_eff_latest}, activeK={active_latest}, respTop1Batch={top1b_latest}, "
        f"respTop1={top1_latest}, minMuDist={min_mu_latest}, piH={pih_latest}, valLoss={val_loss_latest}"
    )
    if inferred_k is not None:
        msgs.append(f"- inferred_num_components~{inferred_k}")

    # Trend diagnostics.
    k_eff_s = _slope(k_eff_vals)
    min_mu_s = _slope(min_mu_vals)
    top1b_s = _slope(resp_top1_batch_vals)

    msgs.append("Trend (tail)")
    msgs.append(f"- slope(K_eff)={k_eff_s:.6f}, slope(minMuDist)={min_mu_s:.6f}, slope(respTop1Batch)={top1b_s:.6f}")

    alerts: List[str] = []
    if k_eff_latest is not None and active_latest is not None:
        if k_eff_latest < 6 or active_latest <= 4:
            alerts.append("collapse_risk_high: effective components are low.")
        elif k_eff_latest < 10:
            alerts.append("collapse_risk_moderate: component usage is narrowing.")

    if top1b_latest is not None and inferred_k is not None:
        uniform_top1 = 1.0 / float(inferred_k)
        if top1b_latest < 1.5 * uniform_top1:
            alerts.append("posterior_too_soft: per-cell assignment is near-uniform.")
        elif top1b_latest > 0.9:
            alerts.append("posterior_too_sharp: per-cell assignment is near one-hot.")

    if min_mu_latest is not None and min_mu_s < -1e-3:
        alerts.append("center_gap_shrinking: minMuDist is decreasing; watch for component merging.")

    if not alerts:
        alerts.append("no_critical_alert: diagnostics look stable.")

    msgs.append("Alerts")
    for a in alerts:
        msgs.append(f"- {a}")
    return msgs


def parse_log(path: Path):
    batch_points: List[Point] = []
    val_points: List[Point] = []
    epochs: List[int] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_stage = EPOCH_STAGE_RE.match(line)
            if m_stage:
                epochs.append(int(m_stage.group(1)))
                continue

            m_batch = BATCH_RE.match(line)
            if m_batch:
                step = int(m_batch.group(1))
                vals = _parse_keyvals(line)
                batch_points.append(Point(step=step, values=vals))
                continue

            m_vb = VAL_BATCH_RE.match(line)
            if m_vb:
                step = int(m_vb.group(1))
                vals = _parse_keyvals(line)
                val_points.append(Point(step=step, values=vals))
                continue

            m_ev = EPOCH_VAL_RE.match(line)
            if m_ev:
                step = int(m_ev.group(1))
                vals = _parse_keyvals(line)
                val_points.append(Point(step=step, values=vals))

    return batch_points, val_points, epochs


def _find_state_tensor(state: Dict[str, torch.Tensor], key_suffix: str) -> Optional[torch.Tensor]:
    # Match exact key first, then DDP-prefixed key, then any suffix match.
    if key_suffix in state:
        return state[key_suffix]
    ddp_key = f"module.{key_suffix}"
    if ddp_key in state:
        return state[ddp_key]
    for k, v in state.items():
        if k.endswith(key_suffix):
            return v
    return None


def diagnose_prior_from_ckpt(ckpt_path: Path, topk: int = 3, csv_out: Optional[Path] = None) -> List[str]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    pi_logits = _find_state_tensor(state, "prior.pi_logits")
    mu = _find_state_tensor(state, "prior.prior_mu")
    logvar = _find_state_tensor(state, "prior.prior_logvar")
    factor = _find_state_tensor(state, "prior.prior_factor")
    if pi_logits is None or mu is None or logvar is None:
        raise KeyError(
            f"Cannot find prior tensors in checkpoint: {ckpt_path}. "
            "Expected keys like prior.pi_logits/prior.prior_mu/prior.prior_logvar."
        )

    pi_logits = pi_logits.float().view(-1)
    mu = mu.float()
    logvar = logvar.float()
    if mu.dim() != 2 or logvar.dim() != 2:
        raise ValueError(f"Invalid prior shapes: mu={tuple(mu.shape)}, logvar={tuple(logvar.shape)}")

    k, d = int(mu.size(0)), int(mu.size(1))
    pi = torch.softmax(pi_logits, dim=0)
    pi_entropy = float((-(pi * torch.log(pi + 1e-12))).sum().item())
    k_eff = float(math.exp(max(0.0, pi_entropy)))

    dmat = torch.cdist(mu, mu, p=2)
    eye = torch.eye(k, dtype=torch.bool)
    dmat_masked = dmat.masked_fill(eye, float("inf"))
    nn_dist, nn_idx = torch.min(dmat_masked, dim=1)

    factor_norm = None
    if factor is not None:
        factor_norm = factor.float().pow(2).mean(dim=(1, 2)).sqrt()

    rows = []
    for i in range(k):
        row = {
            "k": i,
            "pi": float(pi[i].item()),
            "pi_logit": float(pi_logits[i].item()),
            "mu_norm": float(mu[i].norm().item()),
            "mu_mean": float(mu[i].mean().item()),
            "mu_std": float(mu[i].std(unbiased=False).item()),
            "logvar_mean": float(logvar[i].mean().item()),
            "logvar_min": float(logvar[i].min().item()),
            "logvar_max": float(logvar[i].max().item()),
            "var_mean": float(torch.exp(logvar[i]).mean().item()),
            "nn_comp": int(nn_idx[i].item()),
            "nn_dist": float(nn_dist[i].item()),
        }
        if factor_norm is not None:
            row["factor_norm"] = float(factor_norm[i].item())
        rows.append(row)

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    lines: List[str] = []
    lines.append("Prior Snapshot")
    lines.append(f"- K={k}, D={d}, piH={pi_entropy:.6f}, K_eff={k_eff:.4f}")
    lines.append(
        f"- pi range: min={float(pi.min().item()):.6f}, max={float(pi.max().item()):.6f}, "
        f"std={float(pi.std(unbiased=False).item()):.6f}"
    )
    lines.append(
        f"- minMuDist={float(nn_dist.min().item()):.6f}, "
        f"medianNN={float(nn_dist.median().item()):.6f}, maxNN={float(nn_dist.max().item()):.6f}"
    )

    # Show top/bottom components by pi.
    pi_sorted = torch.argsort(pi, descending=True)
    top = pi_sorted[: max(1, min(topk, k))].tolist()
    bot = torch.argsort(pi)[: max(1, min(topk, k))].tolist()
    lines.append("Top components by pi")
    for i in top:
        r = rows[i]
        lines.append(
            f"- k={i}: pi={r['pi']:.6f}, nn_dist={r['nn_dist']:.6f}, "
            f"logvar_mean={r['logvar_mean']:.4f}, mu_norm={r['mu_norm']:.4f}"
        )
    lines.append("Bottom components by pi")
    for i in bot:
        r = rows[i]
        lines.append(
            f"- k={i}: pi={r['pi']:.6f}, nn_dist={r['nn_dist']:.6f}, "
            f"logvar_mean={r['logvar_mean']:.4f}, mu_norm={r['mu_norm']:.4f}"
        )

    if csv_out is not None:
        lines.append(f"- per-component csv: {csv_out}")
    return lines


def main():
    ap = argparse.ArgumentParser(description="Diagnose xVERSE GMVAE training logs.")
    ap.add_argument("log_path", type=str, help="Path to training log text file.")
    ap.add_argument("--tail", type=int, default=50, help="Use last N points for trend diagnosis.")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to diagnose per-prior-component stats.")
    ap.add_argument("--topk", type=int, default=3, help="Top/bottom K components to print by pi.")
    ap.add_argument("--prior-csv", type=str, default=None, help="Optional output csv path for per-component prior stats.")
    args = ap.parse_args()

    path = Path(args.log_path)
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")

    batch_points, val_points, epochs = parse_log(path)

    print(f"Parsed: epochs={len(epochs)}, train_batch_points={len(batch_points)}, val_points={len(val_points)}")
    if not batch_points:
        print("No [Batch ...] lines found; cannot compute diagnostics.")
        return

    for line in diagnose(batch_points, val_points, tail_n=max(1, int(args.tail))):
        print(line)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt file not found: {ckpt_path}")
        csv_out = Path(args.prior_csv) if args.prior_csv else None
        for line in diagnose_prior_from_ckpt(ckpt_path, topk=max(1, int(args.topk)), csv_out=csv_out):
            print(line)


if __name__ == "__main__":
    main()
