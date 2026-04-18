import gc
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


# =========================
# Model blocks (GMM-VAE)
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMMaskEncoder(nn.Module):
    """
    Encoder with expression path + mask path + FiLM modulation.
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        expr_hidden_dim: int = 1024,
        mask_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.expr_encoder = MLP(
            input_dim=num_genes,
            hidden_dims=[expr_hidden_dim, expr_hidden_dim],
            output_dim=expr_hidden_dim,
            dropout=dropout,
        )
        self.mask_encoder = MLP(
            input_dim=num_genes,
            hidden_dims=[mask_hidden_dim, mask_hidden_dim],
            output_dim=mask_hidden_dim,
            dropout=dropout,
        )
        self.film_gamma = nn.Linear(mask_hidden_dim, expr_hidden_dim)
        self.film_beta = nn.Linear(mask_hidden_dim, expr_hidden_dim)
        self.encoder_mu = nn.Linear(expr_hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(expr_hidden_dim, latent_dim)

    def forward(self, x_expr: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unobserved entries are removed before expression path encoding.
        x_expr_masked = x_expr * x_mask
        h_expr = self.expr_encoder(x_expr_masked)
        h_mask = self.mask_encoder(x_mask)

        gamma = torch.tanh(self.film_gamma(h_mask)) + 1.0
        beta = self.film_beta(h_mask)
        h = gamma * h_expr + beta

        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar


class GaussianMixturePrior(nn.Module):
    """
    Learnable diagonal-covariance GMM prior.
    """

    def __init__(self, num_components: int, latent_dim: int):
        super().__init__()
        self.K = num_components
        self.D = latent_dim
        self.pi_logits = nn.Parameter(torch.zeros(num_components))
        self.prior_mu = nn.Parameter(torch.randn(num_components, latent_dim) * 0.01)
        self.prior_logvar = nn.Parameter(torch.zeros(num_components, latent_dim))

    def component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z_exp = z.unsqueeze(1)  # (B, 1, D)
        mu = self.prior_mu.unsqueeze(0)  # (1, K, D)
        logvar = self.prior_logvar.unsqueeze(0)  # (1, K, D)
        inv_var = torch.exp(-logvar)
        quad = ((z_exp - mu) ** 2) * inv_var
        log_det = logvar.sum(dim=-1)
        log_norm = self.D * math.log(2.0 * math.pi)
        return -0.5 * (quad.sum(dim=-1) + log_det + log_norm)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_weights = F.log_softmax(self.pi_logits, dim=0).unsqueeze(0)  # (1, K)
        log_comp = self.component_log_prob(z)  # (B, K)
        return torch.logsumexp(log_weights + log_comp, dim=1)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Analytic score of GMM prior: grad_z log p(z), shape (B, D).
        """
        z_exp = z.unsqueeze(1)  # (B, 1, D)
        mu = self.prior_mu.unsqueeze(0)  # (1, K, D)
        logvar = self.prior_logvar.unsqueeze(0)  # (1, K, D)
        var = torch.exp(logvar)

        log_weights = F.log_softmax(self.pi_logits, dim=0).unsqueeze(0)  # (1, K)
        log_comp = self.component_log_prob(z)  # (B, K)
        log_post = log_weights + log_comp
        resp = torch.softmax(log_post, dim=1).unsqueeze(-1)  # (B, K, 1)

        per_comp_score = (mu - z_exp) / (var + 1e-8)  # (B, K, D)
        score = (resp * per_comp_score).sum(dim=1)  # (B, D)
        return score


class PoissonDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_genes: int, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=num_genes,
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(z)
        return F.softplus(raw) + 1e-8


class MaskFiLMGMMVAE(nn.Module):
    def __init__(
        self,
        num_genes: int,
        latent_dim: int = 64,
        num_components: int = 16,
        expr_hidden_dim: int = 1024,
        mask_hidden_dim: int = 512,
        dec_hidden_dim: int = 1024,
        dropout: float = 0.1,
        prior_type: str = "gmm",
    ):
        super().__init__()
        if prior_type not in ("gmm", "gaussian"):
            raise ValueError(f"Unsupported prior_type: {prior_type}")
        self.prior_type = prior_type
        self.encoder = FiLMMaskEncoder(
            num_genes=num_genes,
            latent_dim=latent_dim,
            expr_hidden_dim=expr_hidden_dim,
            mask_hidden_dim=mask_hidden_dim,
            dropout=dropout,
        )
        self.prior = GaussianMixturePrior(num_components=num_components, latent_dim=latent_dim)
        self.decoder = PoissonDecoder(
            latent_dim=latent_dim,
            num_genes=num_genes,
            hidden_dim=dec_hidden_dim,
            dropout=dropout,
        )
        self.score_head = MLP(
            input_dim=latent_dim,
            hidden_dims=[dec_hidden_dim, dec_hidden_dim],
            output_dim=latent_dim,
            dropout=dropout,
        )

    def forward(self, x_count: torch.Tensor, x_mask: torch.Tensor, x_expr: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if x_expr is None:
            x_expr = torch.log1p(x_count.float())
        mu, logvar = self.encoder(x_expr=x_expr, x_mask=x_mask.float())
        z = reparameterize(mu, logvar)
        rate = self.decoder(z)
        return {"mu": mu, "logvar": logvar, "z": z, "rate": rate}

    def loss(
        self,
        x_count: torch.Tensor,
        x_mask: torch.Tensor,
        beta: float = 1.0,
        encoder_mask: torch.Tensor = None,
        recon_mask: torch.Tensor = None,
        lambda_score: float = 0.0,
        score_noise_std: float = 0.1,
        score_detach_z: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if encoder_mask is None:
            encoder_mask = x_mask
        out = self.forward(x_count=x_count, x_mask=encoder_mask)

        mu = out["mu"]
        logvar = out["logvar"]
        z = out["z"]
        rate = out["rate"]

        if recon_mask is None:
            recon_loss = poisson_nll(x_count=x_count, rate=rate)
        else:
            recon_loss = poisson_nll_masked(x_count=x_count, rate=rate, mask=recon_mask)

        if self.prior_type == "gaussian":
            # Closed-form KL(q(z|x)||N(0,I)) for diagonal Gaussian posterior.
            kl_per_cell = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
            kl_loss = kl_per_cell.mean()
            log_q = gaussian_log_prob_diag(z=z, mu=mu, logvar=logvar)
            zero_mu = torch.zeros_like(z)
            zero_logvar = torch.zeros_like(z)
            log_p = gaussian_log_prob_diag(z=z, mu=zero_mu, logvar=zero_logvar)
        else:
            log_q = gaussian_log_prob_diag(z=z, mu=mu, logvar=logvar)
            log_p = self.prior.log_prob(z)
            kl_loss = (log_q - log_p).mean()
        score_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        score_norm_pred = torch.zeros((), device=z.device, dtype=z.dtype)
        score_norm_tgt = torch.zeros((), device=z.device, dtype=z.dtype)

        if lambda_score > 0:
            z_for_score = z.detach() if score_detach_z else z
            z_noisy = z_for_score + score_noise_std * torch.randn_like(z_for_score)
            score_pred = self.score_head(z_noisy)
            if self.prior_type == "gmm":
                score_tgt = self.prior.score(z_noisy).detach()
            else:
                # score of standard Gaussian N(0, I): grad_z log p(z) = -z
                score_tgt = (-z_noisy).detach()
            score_loss = F.mse_loss(score_pred, score_tgt)
            score_norm_pred = score_pred.norm(dim=-1).mean()
            score_norm_tgt = score_tgt.norm(dim=-1).mean()

        total_loss = recon_loss + beta * kl_loss + lambda_score * score_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "score_loss": score_loss,
            "score_norm_pred": score_norm_pred,
            "score_norm_tgt": score_norm_tgt,
            "log_q_mean": log_q.mean(),
            "log_p_mean": log_p.mean(),
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "rate": rate,
        }


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_log_prob_diag(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    quad = ((z - mu) ** 2) * inv_var
    log_det = logvar.sum(dim=-1)
    d = z.size(-1)
    return -0.5 * (quad.sum(dim=-1) + log_det + d * math.log(2.0 * math.pi))


def poisson_nll(x_count: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    x = x_count.float()
    r = torch.clamp(rate, min=1e-8)
    nll = r - x * torch.log(r) + torch.lgamma(x + 1.0)
    return nll.sum(dim=-1).mean()


def poisson_nll_masked(x_count: torch.Tensor, rate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x_count.float()
    r = torch.clamp(rate, min=1e-8)
    m = mask.float()
    nll = r - x * torch.log(r) + torch.lgamma(x + 1.0)
    nll = nll * m
    denom = torch.clamp(m.sum(), min=1.0)
    return nll.sum() / denom


# =========================
# Training / evaluation (for GMM-VAE)
# =========================
def _build_vae_inputs(value: torch.Tensor):
    x_mask = (value != -1).float()
    x_count = torch.where(x_mask > 0, value, torch.zeros_like(value))
    x_count = torch.clamp(x_count, min=0.0)
    return x_count, x_mask


def _random_hide_observed(
    x_mask: torch.Tensor,
    apply_prob: float = 1.0,
    policy: str = "xverse",
    min_frac: float = 0.1,
    max_frac: float = 0.5,
):
    """
    Randomly set a subset of observed entries from 1 -> 0 for encoder input.
    """
    if apply_prob <= 0:
        return x_mask

    enc_mask = x_mask.clone()
    B = enc_mask.size(0)

    for i in range(B):
        if torch.rand(1, device=enc_mask.device).item() > apply_prob:
            continue

        obs_idx = (enc_mask[i] > 0).nonzero(as_tuple=False).flatten()
        n_obs = int(obs_idx.numel())
        if n_obs <= 1:
            continue

        if policy == "xverse":
            if n_obs < 1000:
                max_to_mask = max(5, int(n_obs * (1.0 / 5.0)))
                if n_obs > 10:
                    low = 10
                    high = max(low, max_to_mask)
                    n_hide = int(torch.randint(low=low, high=high + 1, size=(1,), device=enc_mask.device).item())
                else:
                    n_hide = n_obs
            else:
                p = torch.rand(1, device=enc_mask.device).item()
                if p < 0.2:
                    frac = 0.1 + 0.2 * torch.rand(1, device=enc_mask.device).item()  # 0.1 - 0.3
                elif p < 0.9:
                    frac = 0.3 + 0.2 * torch.rand(1, device=enc_mask.device).item()  # 0.3 - 0.5
                else:
                    frac = 0.5 + 0.25 * torch.rand(1, device=enc_mask.device).item()  # 0.5 - 0.75
                n_hide = int(n_obs * frac)
        else:
            min_frac = max(0.0, min(1.0, min_frac))
            max_frac = max(min_frac, min(1.0, max_frac))
            frac = min_frac + (max_frac - min_frac) * torch.rand(1, device=enc_mask.device).item()
            n_hide = int(n_obs * frac)

        n_hide = max(1, min(n_hide, n_obs - 1))
        perm = torch.randperm(n_obs, device=enc_mask.device)[:n_hide]
        hide_idx = obs_idx[perm]
        enc_mask[i, hide_idx] = 0.0

    return enc_mask


def train_gmm_vae_one_epoch(
    model,
    optimizer,
    scaler,
    train_loader,
    device,
    beta_kl=1.0,
    recon_observed_only=False,
    mask_aug_prob=1.0,
    mask_aug_policy="xverse",
    mask_aug_min_frac=0.1,
    mask_aug_max_frac=0.5,
    lambda_score=0.0,
    score_noise_std=0.1,
    score_detach_z=True,
):
    model.train()
    loss_fn = model.module.loss if hasattr(model, "module") else model.loss
    total_loss = total_recon = total_kl = total_score = 0.0
    n_cells = 0

    for batch_idx, (_, _, _, value) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        value = value.to(device, non_blocking=True)
        x_count, x_mask = _build_vae_inputs(value)
        x_mask_encoder = _random_hide_observed(
            x_mask=x_mask,
            apply_prob=mask_aug_prob,
            policy=mask_aug_policy,
            min_frac=mask_aug_min_frac,
            max_frac=mask_aug_max_frac,
        )

        bsz = x_count.size(0)
        n_cells += bsz

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            out = loss_fn(
                x_count=x_count,
                x_mask=x_mask,
                beta=beta_kl,
                encoder_mask=x_mask_encoder,
                recon_mask=x_mask if recon_observed_only else None,
                lambda_score=lambda_score,
                score_noise_std=score_noise_std,
                score_detach_z=score_detach_z,
            )
            loss = out["loss"]
            recon = out["recon_loss"]
            kl = out["kl_loss"]
            score = out["score_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * bsz
        total_recon += recon.item() * bsz
        total_kl += kl.item() * bsz
        total_score += score.item() * bsz

        if (batch_idx + 1) % 100 == 0:
            print(
                f"[Batch {batch_idx + 1}] "
                f"Loss={loss.item():.4f}, Recon={recon.item():.4f}, KL={kl.item():.4f}, "
                f"Score={score.item():.4f}, "
                f"||s_pred||={out['score_norm_pred'].item():.4f}, ||s_tgt||={out['score_norm_tgt'].item():.4f}"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / n_cells, total_recon / n_cells, total_kl / n_cells, total_score / n_cells


def evaluate_gmm_vae_one_epoch(
    model,
    val_loader,
    device,
    beta_kl=1.0,
    recon_observed_only=False,
    lambda_score=0.0,
    score_noise_std=0.1,
    score_detach_z=True,
):
    model.eval()
    loss_fn = model.module.loss if hasattr(model, "module") else model.loss
    total_loss = total_recon = total_kl = total_score = 0.0
    n_cells = 0

    with torch.no_grad():
        for _, _, _, value in val_loader:
            value = value.to(device, non_blocking=True)
            x_count, x_mask = _build_vae_inputs(value)
            bsz = x_count.size(0)
            n_cells += bsz

            out = loss_fn(
                x_count=x_count,
                x_mask=x_mask,
                beta=beta_kl,
                encoder_mask=x_mask,
                recon_mask=x_mask if recon_observed_only else None,
                lambda_score=lambda_score,
                score_noise_std=score_noise_std,
                score_detach_z=score_detach_z,
            )
            total_loss += out["loss"].item() * bsz
            total_recon += out["recon_loss"].item() * bsz
            total_kl += out["kl_loss"].item() * bsz
            total_score += out["score_loss"].item() * bsz

    return total_loss / n_cells, total_recon / n_cells, total_kl / n_cells, total_score / n_cells


# =========================
# Data pipeline (reused from original)
# =========================
def load_gene_ids(filepath):
    with open(filepath, 'r') as f:
        gene_list = [line.strip() for line in f if line.strip()]
    return gene_list


class FastXVerseBatchDataset(Dataset):
    def __init__(
        self,
        matrix_meta_pairs,
        gene_ids,
        pair_to_sample_id,
        cell_type_to_index,
        use_cache=False,
        available_genes=None,
        io_workers=32,
        pair_to_tissue_id=None,
    ):
        self.gene_ids = gene_ids
        self.pair_to_sample_id = pair_to_sample_id
        self.pair_to_tissue_id = pair_to_tissue_id
        self.cell_type_to_index = cell_type_to_index
        self.use_cache = use_cache
        self.available_genes = set(available_genes) if available_genes is not None else None

        self.X_blocks = []
        self.index_map = []
        self.gene_idx_maps = []
        self.cell_types = []
        self.cache = {} if use_cache else None

        def load_one_block(args):
            matrix_path, meta_path = args
            X = sp.load_npz(matrix_path)
            meta = np.load(meta_path, allow_pickle=True)
            gene_ids_available = meta["gene_ids"]
            cell_type_array = meta["cell_type_ontology_term_id"]
            return (X, gene_ids_available, cell_type_array, matrix_path, meta_path)

        blocks = []
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            futures = [executor.submit(load_one_block, pair) for pair in matrix_meta_pairs]
            for future in as_completed(futures):
                blocks.append(future.result())

        block_order = {(mat, meta): i for i, (mat, meta) in enumerate(matrix_meta_pairs)}
        blocks.sort(key=lambda x: block_order[(x[3], x[4])])

        for block_idx, (X, gene_ids_available, cell_type_array, matrix_path, meta_path) in enumerate(blocks):
            gene_to_idx = {g: i for i, g in enumerate(gene_ids_available)}
            gene_idx_array = np.full(len(self.gene_ids), -1, dtype=np.int32)
            for global_idx, gene in enumerate(self.gene_ids):
                if gene in gene_to_idx:
                    if self.available_genes is None or gene in self.available_genes:
                        gene_idx_array[global_idx] = gene_to_idx[gene]

            self.X_blocks.append(X)
            self.gene_idx_maps.append(gene_idx_array)
            self.cell_types.append(cell_type_array)

            pair = (matrix_path, meta_path)
            sample_id = int(self.pair_to_sample_id.get(pair, -1))
            assert sample_id >= 0, f"Sample ID not found for {pair}!"
            tissue_id = int(self.pair_to_tissue_id.get(pair, -1))

            for i in range(X.shape[0]):
                row = X.getrow(i)
                max_count = row.max()
                sum_count = row.sum()
                if (max_count > 1000) or (sum_count > 200000):
                    continue
                self.index_map.append((block_idx, i, sample_id, tissue_id))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        block_idx, local_idx, sample_id_int, tissue_id_int = self.index_map[idx]
        X = self.X_blocks[block_idx]
        gene_idx_array = self.gene_idx_maps[block_idx]
        cell_type_str = self.cell_types[block_idx][local_idx]
        celltype_index = self.cell_type_to_index.get(cell_type_str, -1)

        row_data = X.getrow(local_idx).toarray().flatten()
        result = np.full(len(self.gene_ids), -1, dtype=np.float32)
        valid = gene_idx_array >= 0
        if np.any(valid):
            result[valid] = row_data[gene_idx_array[valid]].astype(np.float32)

        output = (
            torch.tensor(sample_id_int, dtype=torch.long),
            torch.tensor(tissue_id_int, dtype=torch.long),
            torch.tensor(celltype_index, dtype=torch.long),
            torch.tensor(result, dtype=torch.float32),
        )

        if self.use_cache:
            self.cache[idx] = output

        return output


class BalancedSampleSampler(Sampler):
    def __init__(self, dataset, samples_per_id=None):
        self.dataset = dataset
        self.samples_by_id = defaultdict(list)
        for idx, (_, _, sample_id, _) in enumerate(dataset.index_map):
            self.samples_by_id[sample_id].append(idx)

        self.sample_ids = list(self.samples_by_id.keys())
        self.samples_per_id = samples_per_id or min(len(v) for v in self.samples_by_id.values())

    def __iter__(self):
        indices = []
        for sample_id in self.sample_ids:
            candidates = self.samples_by_id[sample_id]
            if len(candidates) >= self.samples_per_id:
                selected = random.sample(candidates, self.samples_per_id)
            else:
                selected = random.choices(candidates, k=self.samples_per_id)
            indices.extend(selected)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_id * len(self.sample_ids)


def build_cell_type_to_index(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["classification_result"])

    uncategorized_set = {"Other/Unknown"}
    filtered_classes = sorted([
        c for c in df["classification_result"].unique()
        if c not in uncategorized_set
    ])
    print(f"Number of valid cell types (excluding 'Other/Unknown'): {len(filtered_classes)}")

    classification_to_index = {cls: idx for idx, cls in enumerate(filtered_classes)}
    for cls in uncategorized_set:
        classification_to_index[cls] = -1

    id_to_index = {
        row["id"]: classification_to_index.get(row["classification_result"], -1)
        for _, row in df.iterrows()
    }

    return id_to_index


def build_pair_to_sample_id_and_paths(csv_path, allowed_sample_ids=None, use_tissue=None):
    df = pd.read_csv(csv_path)

    required_cols = {'sample_id', 'matrix_paths', 'obs_paths', 'split', 'tissue_id', 'tech_id'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if allowed_sample_ids is not None:
        df = df[df['sample_id'].isin(allowed_sample_ids)]

    df['split'] = df['split'].astype(str).str.strip().str.lower()

    if use_tissue is not None:
        col = df['tissue_name'].astype(str).str.strip().str.casefold()
        if isinstance(use_tissue, (list, tuple, set)):
            queries = [str(x).strip().casefold() for x in use_tissue]
            mask = col.isin(queries)
        else:
            q = str(use_tissue).strip().casefold()
            mask = col.eq(q)
        df = df[mask]

    pair_to_sample_id = {}
    pair_to_tissue_id = {}
    pair_to_tech_id = {}
    train_pairs, valid_pairs = [], []

    for _, row in df.iterrows():
        matrix_paths = [p.strip() for p in str(row['matrix_paths']).split(';') if p.strip()]
        obs_paths = [p.strip() for p in str(row['obs_paths']).split(';') if p.strip()]

        sample_id = int(row['sample_id'])
        tissue_id = int(row['tissue_id'])
        tech_id = int(row['tech_id'])
        split = row['split']

        for m_path, o_path in zip(matrix_paths, obs_paths):
            pair = (m_path, o_path)
            pair_to_sample_id[pair] = sample_id
            pair_to_tissue_id[pair] = tissue_id
            pair_to_tech_id[pair] = tech_id

            if split == 'train':
                train_pairs.append(pair)
            elif split in ('val', 'valid', 'validation'):
                valid_pairs.append(pair)

    print(f"Total filtered pairs: {len(pair_to_sample_id)}")
    print(f"Train: {len(train_pairs)}, Val: {len(valid_pairs)}")

    return (
        pair_to_sample_id,
        train_pairs, valid_pairs,
        pair_to_tissue_id, pair_to_tech_id
    )
