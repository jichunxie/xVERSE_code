#!/usr/bin/env python
"""Train baseline vs. augmented RNA→ADT predictors and evaluate copy sweeps."""

from pathlib import Path
import copy

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_ROOT = Path("/hpc/group/xielab/xj58/xVerse_results/fig7/crossmodal_datasets_GSE291290")
TRAIN_BASE = DATA_ROOT / "original_train.h5ad"
TRAIN_AUG = DATA_ROOT / "poisson_train.h5ad"
VAL_FILE = DATA_ROOT / "original_val.h5ad"
TEST_NGD_FILE = DATA_ROOT / "original_test_ngd.h5ad"
TEST_CAV_FILE = DATA_ROOT / "original_test_cav.h5ad"
METRIC_DIR = DATA_ROOT / "metrics"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-4
SEED = 0
GRAD_CLIP = 500.0
WEIGHT_DECAY = 1e-4
MLP_DROPOUT = 0.2
EARLY_STOP_PATIENCE = 10
POISSON_COPY_COUNTS = list(range(1, 21))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path, return_gene_ids=False, return_obs=False, return_proteins=False):
    if not path.exists():
        raise FileNotFoundError(path)
    adata = sc.read_h5ad(path)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.log1p(X.astype(np.float32, copy=False))
    if "protein" not in adata.obsm:
        raise KeyError(f"'protein' not found in {path}")
    Y = adata.obsm["protein"]
    Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
    outputs = [X.astype(np.float32), Y.astype(np.float32)]
    if return_gene_ids:
        outputs.append(adata.var["gene_ids"].astype(str).values)
    if return_obs:
        outputs.append(adata.obs_names.to_numpy(copy=True))
    if return_proteins:
        names = adata.uns.get("protein_feature_names")
        if names is None:
            names = [f"protein_{i}" for i in range(Y.shape[1])]
        outputs.append(np.asarray(names, dtype=str))
    return tuple(outputs)


def assert_same_features(ref_ids, other_ids, name: str):
    if len(ref_ids) != len(other_ids) or not np.array_equal(ref_ids, other_ids):
        raise ValueError(f"{name} gene mismatch")


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_model(train_loader, val_loader, in_dim, out_dim, label="baseline"):
    model = MLP(in_dim, out_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()
            steps += 1
        avg_train = running_loss / max(steps, 1)

        model.eval()
        val_running = 0.0
        val_steps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                val_loss = criterion(preds, yb)
                val_running += val_loss.item()
                val_steps += 1
        avg_val = val_running / max(val_steps, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{label}] Epoch {epoch+1}/{EPOCHS} train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if EARLY_STOP_PATIENCE and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"[{label}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def prepare_loader(X, Y, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def predict(model, X):
    model.eval()
    loader = DataLoader(torch.from_numpy(X), batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def per_protein_pearson(pred, target):
    corrs = np.full(pred.shape[1], np.nan, dtype=np.float32)
    for j in range(pred.shape[1]):
        p = pred[:, j]
        t = target[:, j]
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() < 2:
            continue
        p = p[mask]
        t = t[mask]
        if np.std(p) == 0 or np.std(t) == 0:
            corrs[j] = 0.0
        else:
            corrs[j] = np.corrcoef(p, t)[0, 1]
    return corrs


def median_pearson(pred, target):
    corrs = per_protein_pearson(pred, target)
    finite = corrs[np.isfinite(corrs)]
    if finite.size == 0:
        return float("nan"), corrs
    return float(np.median(finite)), corrs


def mse_score(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def save_corr_csv(corrs, protein_ids, path: Path, label: str):
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"protein": protein_ids, "pearson_corr": corrs, "model": label})
    df.to_csv(path, index=False)
    print(f"[IO] Saved {path}")


def save_mse_csv(mse_value: float, path: Path, label: str):
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"model": [label], "mse": [mse_value]})
    df.to_csv(path, index=False)
    print(f"[IO] Saved {path}")


def summarize_random_proteins(matrix: np.ndarray, protein_names: np.ndarray, label: str, n: int = 5):
    if matrix.size == 0 or protein_names.size == 0:
        print(f"[{label}] No proteins to summarize.")
        return
    rng = np.random.default_rng(SEED)
    idx = rng.choice(np.arange(protein_names.size), size=min(n, protein_names.size), replace=False)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"[{label}] Random {len(idx)} proteins (raw ADT) quantiles:")
    for i in idx:
        vals = matrix[:, i]
        qs = np.quantile(vals, quantiles)
        q_str = ", ".join(f"q{int(q*100):02d}={v:.4f}" for q, v in zip(quantiles, qs))
        print(f"  - {protein_names[i]}: {q_str}")


def main():
    set_seed(SEED)

    X_train, Y_train, genes_train = load_dataset(TRAIN_BASE, return_gene_ids=True)
    X_val, Y_val, genes_val = load_dataset(VAL_FILE, return_gene_ids=True)
    assert_same_features(genes_train, genes_val, "Validation")
    X_test_ngd, Y_test_ngd, genes_test_ngd, protein_names = load_dataset(
        TEST_NGD_FILE, return_gene_ids=True, return_proteins=True
    )
    assert_same_features(genes_train, genes_test_ngd, "Test NGD")
    X_test_cav, Y_test_cav, genes_test_cav = load_dataset(
        TEST_CAV_FILE, return_gene_ids=True, return_proteins=False
    )
    assert_same_features(genes_train, genes_test_cav, "Test CAV")
    summarize_random_proteins(Y_train, np.asarray(protein_names), label="Train ADT", n=5)

    train_loader_base = prepare_loader(X_train, Y_train)
    val_loader = prepare_loader(X_val, Y_val, shuffle=False)

    print("[Training] Baseline (original only)")
    model_base = train_model(train_loader_base, val_loader, X_train.shape[1], Y_train.shape[1], label="baseline")
    base_pred_ngd = predict(model_base, X_test_ngd)
    base_median_ngd, base_corrs_ngd = median_pearson(base_pred_ngd, Y_test_ngd)
    base_mse_ngd = mse_score(base_pred_ngd, Y_test_ngd)
    print(f"[Eval] Baseline NGD median Pearson: {base_median_ngd:.4f} | MSE: {base_mse_ngd:.4f}")
    save_corr_csv(base_corrs_ngd, protein_names, METRIC_DIR / "baseline_test_ngd_pearson.csv", "baseline_ngd")
    save_mse_csv(base_mse_ngd, METRIC_DIR / "baseline_test_ngd_mse.csv", "baseline_ngd")

    base_pred_cav = predict(model_base, X_test_cav)
    base_median_cav, base_corrs_cav = median_pearson(base_pred_cav, Y_test_cav)
    base_mse_cav = mse_score(base_pred_cav, Y_test_cav)
    print(f"[Eval] Baseline CAV median Pearson: {base_median_cav:.4f} | MSE: {base_mse_cav:.4f}")
    save_corr_csv(base_corrs_cav, protein_names, METRIC_DIR / "baseline_test_cav_pearson.csv", "baseline_cav")
    save_mse_csv(base_mse_cav, METRIC_DIR / "baseline_test_cav_mse.csv", "baseline_cav")

    if not TRAIN_AUG.exists():
        print("[Warn] poisson_train.h5ad missing; skipping augmented training.")
        return
    print(f"[Load] Reading augmented dataset from {TRAIN_AUG} (this can take a while)...")
    X_aug_full, Y_aug_full, genes_aug, poisson_obs = load_dataset(
        TRAIN_AUG, return_gene_ids=True, return_obs=True
    )
    print(f"[Load] Done. Augmented cells: {X_aug_full.shape[0]}")
    assert_same_features(genes_train, genes_aug, "Augmented train")

    def subset_mask(max_copy: int):
        mask = np.zeros(len(poisson_obs), dtype=bool)
        for idx, name in enumerate(poisson_obs):
            name_str = str(name).split(":", 1)[-1]
            parts = name_str.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) <= max_copy:
                mask[idx] = True
        return mask

    results = []
    for copy_count in POISSON_COPY_COUNTS:
        mask = subset_mask(copy_count)
        if not mask.any():
            print(f"[Warn] No replicates for copy count {copy_count}; skipping.")
            continue
        print(f"[Prep] copy<={copy_count}: using {mask.sum()} Poisson replicates.")
        X_aug = np.concatenate([X_train, X_aug_full[mask]], axis=0)
        Y_aug = np.concatenate([Y_train, Y_aug_full[mask]], axis=0)
        train_loader_aug = prepare_loader(X_aug, Y_aug)

        label = f"augmented_copy{copy_count}"
        print(f"[Training] {label} (original + first {copy_count} copies)")
        model_aug = train_model(train_loader_aug, val_loader, X_train.shape[1], Y_train.shape[1], label=label)
        aug_pred_ngd = predict(model_aug, X_test_ngd)
        aug_median_ngd, aug_corrs_ngd = median_pearson(aug_pred_ngd, Y_test_ngd)
        aug_mse_ngd = mse_score(aug_pred_ngd, Y_test_ngd)
        print(f"[Eval] {label} NGD median Pearson: {aug_median_ngd:.4f} | MSE: {aug_mse_ngd:.4f}")
        save_corr_csv(
            aug_corrs_ngd,
            protein_names,
            METRIC_DIR / f"{label}_test_ngd_pearson.csv",
            f"{label}_ngd",
        )
        save_mse_csv(
            aug_mse_ngd,
            METRIC_DIR / f"{label}_test_ngd_mse.csv",
            f"{label}_ngd",
        )
        diff_ngd = aug_median_ngd - base_median_ngd
        results.append((copy_count, "ngd", aug_median_ngd, diff_ngd))
        print(f"[Result] {label} NGD diff vs baseline: {diff_ngd:.4f}")

        aug_pred_cav = predict(model_aug, X_test_cav)
        aug_median_cav, aug_corrs_cav = median_pearson(aug_pred_cav, Y_test_cav)
        aug_mse_cav = mse_score(aug_pred_cav, Y_test_cav)
        print(f"[Eval] {label} CAV median Pearson: {aug_median_cav:.4f} | MSE: {aug_mse_cav:.4f}")
        save_corr_csv(
            aug_corrs_cav,
            protein_names,
            METRIC_DIR / f"{label}_test_cav_pearson.csv",
            f"{label}_cav",
        )
        save_mse_csv(
            aug_mse_cav,
            METRIC_DIR / f"{label}_test_cav_mse.csv",
            f"{label}_cav",
        )
        diff_cav = aug_median_cav - base_median_cav
        results.append((copy_count, "cav", aug_median_cav, diff_cav))
        print(f"[Result] {label} CAV diff vs baseline: {diff_cav:.4f}")

    if results:
        print("[Summary] Poisson copy sweep:")
        for copy_count, domain, med, diff in results:
            print(f"  copies <= {copy_count} ({domain}): median={med:.4f}, diff={diff:.4f}")


if __name__ == "__main__":
    main()
