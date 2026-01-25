#!/usr/bin/env python
"""MLP classifiers for VDJ C-gene dominance tasks with Poisson copy sweeps."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
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
WEIGHT_DECAY = 1e-5
MLP_DROPOUT = 0.2
EARLY_STOP_PATIENCE = 10
GRAD_CLIP = 500.0
SEED = 0
POISSON_COPY_COUNTS = list(range(1, 21))

LABEL_CANDIDATES = {
    "BCR_Heavy_C_gene_Dominant": ["BCR_Heavy_C_gene_Dominant"],
    "TCR_Alpha_Gamma_C_gene_Dominant": ["TCR_Alpha_Gamma_C_gene_Dominant"],
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log1p_dense(x) -> np.ndarray:
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.log1p(x.astype(np.float32, copy=False))


def normalize_label(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def find_label_columns(adata: sc.AnnData) -> Dict[str, str]:
    resolved = {}
    for logical, candidates in LABEL_CANDIDATES.items():
        for cand in candidates:
            if cand in adata.obs.columns:
                resolved[logical] = cand
                break
    missing = [k for k in LABEL_CANDIDATES if k not in resolved]
    if missing:
        raise KeyError(f"Missing label columns in obs: {missing}")
    return resolved


def load_split(path: Path, label_cols: Dict[str, str], return_obs: bool = False):
    if not path.exists():
        raise FileNotFoundError(path)
    adata = sc.read_h5ad(path)
    X = log1p_dense(adata.X)
    labels: Dict[str, np.ndarray] = {}
    for logical, col in label_cols.items():
        arr = adata.obs[col].map(normalize_label).to_numpy(object)
        labels[logical] = arr
    # Do not drop rows here; filtering is done per-task later
    outputs = [X, labels]
    if return_obs:
        outputs.append(adata.obs_names.to_numpy(copy=True))
    return tuple(outputs)


def encode_labels(y_str: np.ndarray, classes: List[str]) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return np.array([class_to_idx[s] for s in y_str], dtype=int)


def filter_known(X: np.ndarray, y_str: np.ndarray, classes: List[str]):
    mask = np.isin(y_str, classes)
    return X[mask], y_str[mask], mask


def compute_metrics(y_true: np.ndarray, proba, n_classes: int, average: str, class_weight: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    if n_classes == 2:
        preds = (proba >= 0.5).astype(int)
        probs_full = np.stack([1.0 - proba, proba], axis=1)
        try:
            auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = float("nan")
    else:
        preds = np.argmax(proba, axis=1)
        probs_full = proba
        try:
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

    weights = class_weight[y_true]
    logp = np.log(probs_full[np.arange(len(y_true)), y_true] + eps)
    loss = -np.mean(weights * logp)

    return {
        "auc": float(auc),
        "loss": float(loss),
    }


def build_class_weight(y: np.ndarray, n_classes: int) -> Dict[int, float]:
    classes = np.arange(n_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {cls: w for cls, w in zip(classes, weights)}


def print_nonempty_counts(labels: Dict[str, np.ndarray], mask: np.ndarray, split: str, task: str):
    vals = [v for v in labels[task][mask] if v is not None and str(v).strip()]
    uniq, counts = np.unique(vals, return_counts=True)
    print(f"  {task} ({split}): {len(vals)} non-empty | unique={dict(zip(uniq, counts))}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
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


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_mlp(X_train, y_train, X_val, y_val, n_classes: int):
    class_weight = build_class_weight(y_train, n_classes)
    cw_tensor = torch.tensor([class_weight[i] for i in range(n_classes)], dtype=torch.float32, device=DEVICE)
    model = MLP(X_train.shape[1], n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

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
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()
            steps += 1
        avg_train = running_loss / max(steps, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                val_steps += 1
        avg_val = val_loss / max(val_steps, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[MLP] Epoch {epoch+1}/{EPOCHS} train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if EARLY_STOP_PATIENCE and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"[MLP] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def predict_proba(model, X, n_classes: int):
    loader = make_loader(X, np.zeros(len(X), dtype=int), shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            if n_classes == 2:
                prob = torch.softmax(logits, dim=1)[:, 1]
            else:
                prob = torch.softmax(logits, dim=1)
            probs.append(prob.cpu().numpy())
    proba = np.concatenate(probs, axis=0)
    if n_classes == 2:
        return proba
    return proba


def main():
    set_seed(SEED)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    base_adata = sc.read_h5ad(TRAIN_BASE)
    label_cols = find_label_columns(base_adata)
    print(f"[Info] Using label columns: {label_cols}")

    X_train_base, labels_train_base = load_split(TRAIN_BASE, label_cols)
    X_val, labels_val = load_split(VAL_FILE, label_cols)
    X_test_ngd, labels_test_ngd = load_split(TEST_NGD_FILE, label_cols)
    X_test_cav, labels_test_cav = load_split(TEST_CAV_FILE, label_cols)

    has_poisson = TRAIN_AUG.exists()
    if has_poisson:
        X_aug_full, labels_aug_full, poisson_obs = load_split(TRAIN_AUG, label_cols, return_obs=True)
        print(f"[Load] Augmented cells: {X_aug_full.shape[0]}")
    else:
        print("[Warn] poisson_train.h5ad missing; augmented sweeps will be skipped.")

    rows = []
    for task in LABEL_CANDIDATES:
        print(f"\n[Task] {task}")
        mask_train = pd.notna(labels_train_base[task])
        mask_val = pd.notna(labels_val[task])
        mask_ngd = pd.notna(labels_test_ngd[task])
        mask_cav = pd.notna(labels_test_cav[task])

        X_train_task = X_train_base[mask_train]
        y_train_str = labels_train_base[task][mask_train]
        X_val_task = X_val[mask_val]
        y_val_str = labels_val[task][mask_val]
        X_test_ngd_task = X_test_ngd[mask_ngd]
        y_test_ngd_str = labels_test_ngd[task][mask_ngd]
        X_test_cav_task = X_test_cav[mask_cav]
        y_test_cav_str = labels_test_cav[task][mask_cav]

        # Remove rare classes (<10 samples in train)
        uniq, counts = np.unique(y_train_str, return_counts=True)
        keep_classes = [c for c, n in zip(uniq, counts) if n >= 10]
        class_mask = np.isin(y_train_str, keep_classes)
        X_train_task = X_train_task[class_mask]
        y_train_str = y_train_str[class_mask]

        if len(keep_classes) < 2:
            print("  [Skip] fewer than 2 classes with >=10 samples in train")
            continue

        print_nonempty_counts({task: y_train_str}, np.ones_like(y_train_str, dtype=bool), split="train", task=task)

        classes = sorted(keep_classes)
        X_val_f, y_val_str_f, _ = filter_known(X_val_task, y_val_str, classes)
        X_test_ngd_f, y_test_ngd_str_f, _ = filter_known(X_test_ngd_task, y_test_ngd_str, classes)
        X_test_cav_f, y_test_cav_str_f, _ = filter_known(X_test_cav_task, y_test_cav_str, classes)

        if len(y_val_str_f) == 0 or len(y_test_ngd_str_f) == 0 or len(y_test_cav_str_f) == 0:
            print("  [Skip] validation or test missing known classes after filtering")
            continue

        print_nonempty_counts({task: y_val_str_f}, np.ones_like(y_val_str_f, dtype=bool), split="val", task=task)
        print_nonempty_counts({task: y_test_ngd_str_f}, np.ones_like(y_test_ngd_str_f, dtype=bool), split="test_ngd", task=task)
        print_nonempty_counts({task: y_test_cav_str_f}, np.ones_like(y_test_cav_str_f, dtype=bool), split="test_cav", task=task)

        y_train = encode_labels(y_train_str, classes)
        y_val = encode_labels(y_val_str_f, classes)
        y_test_ngd = encode_labels(y_test_ngd_str_f, classes)
        y_test_cav = encode_labels(y_test_cav_str_f, classes)
        n_classes = len(classes)
        average = "binary" if n_classes == 2 else "macro"
        cw_vec = np.array([build_class_weight(y_train, n_classes)[i] for i in range(n_classes)], dtype=float)

        model_base = train_mlp(X_train_task, y_train, X_val_f, y_val, n_classes)
        proba_ngd = predict_proba(model_base, X_test_ngd_f, n_classes)
        proba_cav = predict_proba(model_base, X_test_cav_f, n_classes)
        base_ngd = compute_metrics(y_test_ngd, proba_ngd, n_classes, average, cw_vec)
        base_cav = compute_metrics(y_test_cav, proba_cav, n_classes, average, cw_vec)
        print(
            "  Baseline "
            f"NGD AUC={base_ngd['auc']:.4f}, NGD LOSS={base_ngd['loss']:.4f}; "
            f"CAV AUC={base_cav['auc']:.4f}, CAV LOSS={base_cav['loss']:.4f}"
        )
        for split_name, metrics in [("test_ngd", base_ngd), ("test_cav", base_cav)]:
            for metric_name in metrics:
                rows.append(
                    {
                        "model": "baseline",
                        "task": task,
                        "copy_count": 0,
                        "split": split_name,
                        "metric": metric_name,
                        "value": metrics[metric_name],
                    }
                )

        if not has_poisson:
            continue

        for copy_count in POISSON_COPY_COUNTS:
            mask = np.array([str(name).split(":", 1)[-1].rsplit("_", 1)[-1].isdigit() and int(str(name).split(":", 1)[-1].rsplit("_", 1)[-1]) <= copy_count for name in poisson_obs])
            if not mask.any():
                print(f"[Warn] No Poisson replicates for copy<={copy_count}; skip.")
                continue
            # Filter aug to this task's known labels
            aug_labels_task = labels_aug_full[task][mask]
            known_aug_mask = np.isin(aug_labels_task, classes)
            if not known_aug_mask.any():
                print(f"  [Skip copy {copy_count}] no known classes in augmentation")
                continue
            X_aug_task = X_aug_full[mask][known_aug_mask]
            y_aug_task = aug_labels_task[known_aug_mask]

            X_train_aug = np.concatenate([X_train_task, X_aug_task], axis=0)
            y_train_aug_str = np.concatenate([y_train_str, y_aug_task], axis=0)
            if not np.all(np.isin(y_train_aug_str, classes)):
                print(f"  [Skip copy {copy_count}] unseen classes in augmentation")
                continue
            y_train_aug = encode_labels(y_train_aug_str, classes)
            cw_aug = np.array([build_class_weight(y_train_aug, n_classes)[i] for i in range(n_classes)], dtype=float)

            model_aug = train_mlp(X_train_aug, y_train_aug, X_val_f, y_val, n_classes)
            aug_proba_ngd = predict_proba(model_aug, X_test_ngd_f, n_classes)
            aug_proba_cav = predict_proba(model_aug, X_test_cav_f, n_classes)
            aug_ngd = compute_metrics(y_test_ngd, aug_proba_ngd, n_classes, average, cw_aug)
            aug_cav = compute_metrics(y_test_cav, aug_proba_cav, n_classes, average, cw_aug)
            print(
                f"  copy<={copy_count} "
                f"NGD AUC={aug_ngd['auc']:.4f}, NGD LOSS={aug_ngd['loss']:.4f}; "
                f"CAV AUC={aug_cav['auc']:.4f}, CAV LOSS={aug_cav['loss']:.4f}"
            )
            for split_name, metrics in [("test_ngd", aug_ngd), ("test_cav", aug_cav)]:
                for metric_name in metrics:
                    rows.append(
                        {
                            "model": f"augmented_copy{copy_count}",
                            "task": task,
                            "copy_count": copy_count,
                            "split": split_name,
                            "metric": metric_name,
                            "value": metrics[metric_name],
                        }
                    )

    if rows:
        df = pd.DataFrame(rows)
        out_path = METRIC_DIR / "mlp_vdj_cgene_dominance.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[IO] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
