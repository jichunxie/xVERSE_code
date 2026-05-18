import math
import os
import random
import time
import hashlib
import json
import bisect
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


# =========================
# Model blocks (MFA-VAE)
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

    def forward(self, x_expr: torch.Tensor, x_mask: torch.Tensor, return_hidden: bool = False):
        # Unobserved entries are removed before expression path encoding.
        x_expr_masked = x_expr * x_mask
        h_expr = self.expr_encoder(x_expr_masked)
        h_mask = self.mask_encoder(x_mask)

        gamma = torch.tanh(self.film_gamma(h_mask)) + 1.0
        beta = self.film_beta(h_mask)
        h = gamma * h_expr + beta

        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        if return_hidden:
            return mu, logvar, h
        return mu, logvar


class GaussianMixturePrior(nn.Module):
    """
    Mixture-of-factor-analyzers prior:
      c ~ Cat(pi)
      u ~ N(0, I)
      eps ~ N(0, diag(exp(logvar_c)))
      z = mu_c + A_c u + eps
    Marginally, p(z|c=k) = N(mu_k, diag(exp(logvar_k)) + A_k A_k^T).
    Optional tissue-conditional prior: one MFA parameter set per tissue id.
    """

    def __init__(
        self,
        num_components: int,
        latent_dim: int,
        cov_rank: int = 8,
        conditional_on_tissue: bool = False,
        num_tissues: int = 0,
        shared_covariance: bool = False,
        mu_init: str = "normal",
        mu_init_radius: float = 1.0,
        mu_init_groups: int = 8,
        mu_init_local_radius: float = 0.5,
        hierarchy_groups: int = 8,
    ):
        super().__init__()
        self.K = int(num_components)
        self.D = int(latent_dim)
        self.R = max(0, int(cov_rank))
        # Hierarchical prior: g -> s -> z, with K flattened components.
        self.shared_covariance = False
        self.conditional_on_tissue = bool(conditional_on_tissue and int(num_tissues) > 0)
        self.num_tissues = max(0, int(num_tissues))

        self.G = max(1, min(int(hierarchy_groups), self.K))
        self.S = int(math.ceil(float(self.K) / float(self.G)))
        comp_group = torch.arange(self.K, dtype=torch.long) % self.G
        comp_sub = torch.arange(self.K, dtype=torch.long) // self.G
        self.register_buffer("component_group", comp_group, persistent=False)
        self.register_buffer("component_sub", comp_sub, persistent=False)

        self.group_logits = nn.Parameter(torch.zeros(self.G))
        self.sub_logits = nn.Parameter(torch.zeros(self.G, self.S))
        if str(mu_init) in ("sphere", "grouped_sphere"):
            group_mu = F.normalize(torch.randn(self.G, self.D), dim=-1) * float(mu_init_radius)
        elif str(mu_init) == "zero":
            group_mu = torch.zeros(self.G, self.D)
        else:
            group_mu = torch.randn(self.G, self.D)
        sub_delta = F.normalize(torch.randn(self.G, self.S, self.D), dim=-1) * float(mu_init_local_radius)
        self.group_mu = nn.Parameter(group_mu)
        self.sub_delta = nn.Parameter(sub_delta)
        self.prior_logvar_h = nn.Parameter(torch.zeros(self.G, self.S, self.D))
        if self.R > 0:
            # Shared local factor directions per coarse group.
            self.group_factor = nn.Parameter(torch.randn(self.G, self.D, self.R) * 0.01)
        else:
            self.register_parameter("group_factor", None)

        # Tissue-specific parameters: pi only (component usage prior).
        if self.conditional_on_tissue:
            self.pi_logits_t = nn.Parameter(torch.zeros(self.num_tissues, self.K))
            self.register_parameter("prior_mu_t", None)
            self.register_parameter("prior_logvar_t", None)
            self.register_parameter("prior_factor_t", None)
        else:
            self.register_parameter("pi_logits_t", None)
            self.register_parameter("prior_mu_t", None)
            self.register_parameter("prior_logvar_t", None)
            self.register_parameter("prior_factor_t", None)

    @property
    def prior_mu(self) -> torch.Tensor:
        g = self.component_group.to(self.group_mu.device)
        s = self.component_sub.to(self.group_mu.device)
        return self.group_mu[g] + self.sub_delta[g, s]

    @property
    def prior_logvar(self) -> torch.Tensor:
        g = self.component_group.to(self.prior_logvar_h.device)
        s = self.component_sub.to(self.prior_logvar_h.device)
        return self.prior_logvar_h[g, s]

    @property
    def prior_factor(self) -> Optional[torch.Tensor]:
        if self.group_factor is None:
            return None
        g = self.component_group.to(self.group_factor.device)
        return self.group_factor[g]

    @property
    def pi_logits(self) -> torch.Tensor:
        # Return flat normalized log-weights; softmax(pi_logits) equals p(g)p(s|g).
        log_w = self._hierarchical_log_weights_flat()
        return log_w

    def _hierarchical_log_weights_flat(self) -> torch.Tensor:
        log_g = F.log_softmax(self.group_logits.float(), dim=0)
        log_s = F.log_softmax(self.sub_logits.float(), dim=-1)
        g = self.component_group.to(self.group_logits.device)
        s = self.component_sub.to(self.group_logits.device)
        flat = log_g[g] + log_s[g, s]
        # Renormalize if K is not exactly G*S and some sub slots are unused.
        return flat - torch.logsumexp(flat, dim=0)

    def _sanitize_tissue_id(self, tissue_id: torch.Tensor) -> torch.Tensor:
        if (not self.conditional_on_tissue) or tissue_id is None:
            return None
        tid = tissue_id.long()
        tid = torch.where(tid < 0, torch.zeros_like(tid), tid)
        tid = torch.where(tid >= self.num_tissues, torch.zeros_like(tid), tid)
        return tid

    def _clamp_logvar(self, logvar: torch.Tensor, logvar_min: float = None, logvar_max: float = None) -> torch.Tensor:
        if logvar_min is None and logvar_max is None:
            return logvar
        min_v = float("-inf") if logvar_min is None else float(logvar_min)
        max_v = float("inf") if logvar_max is None else float(logvar_max)
        return torch.clamp(logvar, min=min_v, max=max_v)

    def _expanded_logvar(self, logvar_min: float = None, logvar_max: float = None) -> torch.Tensor:
        logvar = self._clamp_logvar(self.prior_logvar.float(), logvar_min, logvar_max)
        if self.shared_covariance:
            return logvar.view(1, self.D).expand(self.K, self.D)
        return logvar

    def _expanded_factor(self) -> Optional[torch.Tensor]:
        if self.prior_factor is None:
            return None
        factor = self.prior_factor.float()
        if self.shared_covariance:
            return factor.view(1, self.D, self.R).expand(self.K, self.D, self.R)
        return factor

    def component_log_prob(
        self,
        z: torch.Tensor,
        logvar_min: float = None,
        logvar_max: float = None,
        tissue_id: torch.Tensor = None,
    ) -> torch.Tensor:
        device_type = z.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            zf = z.float()
            mu = self.prior_mu.float().unsqueeze(0)  # (1,K,D)
            logvar = self._expanded_logvar(logvar_min, logvar_max).unsqueeze(0)  # (1,K,D)
            factor_exp = self._expanded_factor()
            factor = factor_exp.unsqueeze(0) if factor_exp is not None else None

            d_inv = torch.exp(-logvar)
            delta = zf.unsqueeze(1) - mu  # (B,K,D)
            delta_d = delta * d_inv
            quad = (delta * delta_d).sum(dim=-1)  # (B,K)
            log_det = logvar.sum(dim=-1)  # (B,K)

            if factor is not None:
                dinv_u = d_inv.unsqueeze(-1) * factor  # (B,K,D,R)
                ut_dinv_u = torch.einsum("bkdr,bkds->bkrs", factor, dinv_u)
                eye = torch.eye(self.R, device=factor.device, dtype=factor.dtype).view(1, 1, self.R, self.R)
                s = ut_dinv_u + eye + 1e-6 * eye
                chol_s = torch.linalg.cholesky(s)
                inv_s = torch.cholesky_inverse(chol_s)
                t = torch.einsum("bkd,bkdr->bkr", delta_d, factor)
                quad_corr = torch.einsum("bkr,bkrs,bks->bk", t, inv_s, t)
                quad = quad - quad_corr
                log_det = log_det + 2.0 * torch.log(torch.diagonal(chol_s, dim1=-2, dim2=-1)).sum(dim=-1)

            out = -0.5 * (quad + log_det + self.D * math.log(2.0 * math.pi))
        return out

    def component_log_prob_aligned(
        self,
        z_comp: torch.Tensor,
        logvar_min: float = None,
        logvar_max: float = None,
        tissue_id: torch.Tensor = None,
    ) -> torch.Tensor:
        zc = z_comp.float()
        if zc.dim() != 3 or zc.size(1) != self.K or zc.size(2) != self.D:
            raise ValueError(f"Expected z_comp shape (B,{self.K},{self.D}), got {tuple(zc.shape)}")
        return self.component_log_prob_from_zcomp(zc, logvar_min, logvar_max, tissue_id)

    def component_log_prob_from_zcomp(
        self,
        zc: torch.Tensor,
        logvar_min: float = None,
        logvar_max: float = None,
        tissue_id: torch.Tensor = None,
    ) -> torch.Tensor:
        device_type = zc.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            mu = self.prior_mu.float().unsqueeze(0)
            logvar = self._expanded_logvar(logvar_min, logvar_max).unsqueeze(0)
            factor_exp = self._expanded_factor()
            factor = factor_exp.unsqueeze(0) if factor_exp is not None else None
            delta = zc - mu
            d_inv = torch.exp(-logvar)
            delta_d = delta * d_inv
            quad = (delta * delta_d).sum(dim=-1)
            log_det = logvar.sum(dim=-1)
            if factor is not None:
                dinv_u = d_inv.unsqueeze(-1) * factor
                ut_dinv_u = torch.einsum("bkdr,bkds->bkrs", factor, dinv_u)
                eye = torch.eye(self.R, device=factor.device, dtype=factor.dtype).view(1, 1, self.R, self.R)
                s = ut_dinv_u + eye + 1e-6 * eye
                chol_s = torch.linalg.cholesky(s)
                inv_s = torch.cholesky_inverse(chol_s)
                t = torch.einsum("bkd,bkdr->bkr", delta_d, factor)
                quad_corr = torch.einsum("bkr,bkrs,bks->bk", t, inv_s, t)
                quad = quad - quad_corr
                log_det = log_det + 2.0 * torch.log(torch.diagonal(chol_s, dim1=-2, dim2=-1)).sum(dim=-1)
            return -0.5 * (quad + log_det + self.D * math.log(2.0 * math.pi))

    def _log_weights(self, tissue_id: torch.Tensor = None) -> torch.Tensor:
        tid = self._sanitize_tissue_id(tissue_id)
        if tid is None:
            return F.log_softmax(self.pi_logits.float(), dim=0).unsqueeze(0)
        return F.log_softmax(self.pi_logits_t[tid].float(), dim=-1)

    def log_prob(
        self,
        z: torch.Tensor,
        topk: int = 0,
        logvar_min: float = None,
        logvar_max: float = None,
        tissue_id: torch.Tensor = None,
    ) -> torch.Tensor:
        log_weights = self._log_weights(tissue_id=tissue_id)
        log_comp = self.component_log_prob(z, logvar_min=logvar_min, logvar_max=logvar_max, tissue_id=tissue_id)
        logits = log_weights + log_comp
        k = int(topk)
        if k > 0 and k < logits.size(1):
            top_vals, _ = torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
            return torch.logsumexp(top_vals, dim=1)
        return torch.logsumexp(logits, dim=1)

    def score(
        self,
        z: torch.Tensor,
        logvar_min: float = None,
        logvar_max: float = None,
        tissue_id: torch.Tensor = None,
    ) -> torch.Tensor:
        device_type = z.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            mu = self.prior_mu.float().unsqueeze(0)
            logvar = self._expanded_logvar(logvar_min, logvar_max).unsqueeze(0)
            factor_exp = self._expanded_factor()
            factor = factor_exp.unsqueeze(0) if factor_exp is not None else None
            zf = z.float()
            delta = zf.unsqueeze(1) - mu
            d_inv = torch.exp(-logvar)
            delta_d = delta * d_inv
            log_post = self._log_weights(tissue_id=tissue_id) + self.component_log_prob(z, logvar_min=logvar_min, logvar_max=logvar_max, tissue_id=tissue_id)
            resp = torch.softmax(log_post, dim=1).unsqueeze(-1)
            inv_delta = delta_d
            if factor is not None:
                dinv_u = d_inv.unsqueeze(-1) * factor
                ut_dinv_u = torch.einsum("bkdr,bkds->bkrs", factor, dinv_u)
                eye = torch.eye(self.R, device=factor.device, dtype=factor.dtype).view(1, 1, self.R, self.R)
                s = ut_dinv_u + eye + 1e-6 * eye
                chol_s = torch.linalg.cholesky(s)
                inv_s = torch.cholesky_inverse(chol_s)
                t = torch.einsum("bkd,bkdr->bkr", delta_d, factor)
                sproj = torch.einsum("bkrs,bks->bkr", inv_s, t)
                corr = torch.einsum("bkdr,bkr->bkd", dinv_u, sproj)
                inv_delta = delta_d - corr
            per_comp_score = -inv_delta
            return (resp * per_comp_score).sum(dim=1).to(z.dtype)

    def component_covariances(self) -> torch.Tensor:
        logvar = self._expanded_logvar()
        diag_cov = torch.diag_embed(torch.exp(logvar.float()))
        if self.R <= 0:
            return diag_cov
        f = self._expanded_factor()
        low_rank_cov = torch.einsum("kdr,ksr->kds", f, f)
        return diag_cov + low_rank_cov

    def global_covariance(self) -> torch.Tensor:
        w = torch.softmax(self.pi_logits.float(), dim=0)
        cov_k = self.component_covariances()
        mu = self.prior_mu.float()
        mean = (w.unsqueeze(1) * mu).sum(dim=0)
        second = (w.unsqueeze(1).unsqueeze(2) * (cov_k + mu.unsqueeze(2) * mu.unsqueeze(1))).sum(dim=0)
        return second - mean.unsqueeze(1) * mean.unsqueeze(0)

    def posterior_responsibilities(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 0,
        logvar_min: float = None,
        logvar_max: float = None,
        tissue_id: torch.Tensor = None,
    ) -> torch.Tensor:
        t = max(float(temperature), 1e-6)
        logits = (self._log_weights(tissue_id=tissue_id) + self.component_log_prob(z, logvar_min=logvar_min, logvar_max=logvar_max, tissue_id=tissue_id)) / t
        k = int(topk)
        if k <= 0 or k >= logits.size(1):
            return torch.softmax(logits, dim=1)
        top_vals, top_idx = torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
        top_resp = torch.softmax(top_vals, dim=1)
        resp = torch.zeros_like(logits)
        resp.scatter_(1, top_idx, top_resp)
        return resp

    def clamp_logvar_(self, min_val: float = -6.0, max_val: float = 4.0):
        with torch.no_grad():
            self.prior_logvar_h.clamp_(min=min_val, max=max_val)
            if self.prior_logvar_t is not None:
                self.prior_logvar_t.clamp_(min=min_val, max=max_val)


class PoissonDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_genes: int, hidden_dim: int = 1024, dropout: float = 0.1, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = max(0, int(cond_dim))
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_genes)
        self.drop = nn.Dropout(dropout)
        if self.cond_dim > 0:
            self.film1 = nn.Linear(self.cond_dim, hidden_dim * 2)
            self.film2 = nn.Linear(self.cond_dim, hidden_dim * 2)
        else:
            self.film1 = None
            self.film2 = None

    @staticmethod
    def _apply_film(h: torch.Tensor, film: nn.Linear, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = film(cond).chunk(2, dim=-1)
        return h * (1.0 + torch.tanh(gamma)) + beta

    def forward(self, z: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(self.fc1(z))
        if cond is not None and self.film1 is not None:
            h = self._apply_film(h, self.film1, cond)
        h = self.drop(F.gelu(h))
        h = self.norm2(self.fc2(h))
        if cond is not None and self.film2 is not None:
            h = self._apply_film(h, self.film2, cond)
        h = self.drop(F.gelu(h))
        return self.out(h)


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
        prior_cov_rank: int = 8,
        prior_shared_covariance: bool = False,
        posterior_cov_rank: int = 0,
        prior_mu_init: str = "normal",
        prior_mu_init_radius: float = 1.0,
        prior_mu_init_groups: int = 8,
        prior_mu_init_local_radius: float = 0.5,
        hierarchy_groups: int = 8,
        num_cell_types: int = 0,
        conditional_prior_on_tissue: bool = False,
        num_tissues: int = 0,
        num_batches: int = 0,
        batch_emb_dim: int = 0,
        batch_cond_drop_prob: float = 0.0,
        recon_loss_type: str = "poisson",
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
        self.prior = GaussianMixturePrior(
            num_components=num_components,
            latent_dim=latent_dim,
            cov_rank=prior_cov_rank,
            conditional_on_tissue=conditional_prior_on_tissue,
            num_tissues=num_tissues,
            shared_covariance=prior_shared_covariance,
            mu_init=prior_mu_init,
            mu_init_radius=prior_mu_init_radius,
            mu_init_groups=prior_mu_init_groups,
            mu_init_local_radius=prior_mu_init_local_radius,
            hierarchy_groups=hierarchy_groups,
        )
        self.num_batches = max(0, int(num_batches))
        self.batch_emb_dim = max(0, int(batch_emb_dim))
        self.batch_cond_drop_prob = max(0.0, min(1.0, float(batch_cond_drop_prob)))
        if self.num_batches > 0 and self.batch_emb_dim > 0:
            self.batch_embedding = nn.Embedding(self.num_batches, self.batch_emb_dim)
        else:
            self.batch_embedding = None
        self.decoder = PoissonDecoder(
            latent_dim=latent_dim,
            num_genes=num_genes,
            hidden_dim=dec_hidden_dim,
            dropout=dropout,
            cond_dim=self.batch_emb_dim if self.batch_embedding is not None else 0,
        )
        self.nb_theta_decoder = PoissonDecoder(
            latent_dim=latent_dim,
            num_genes=num_genes,
            hidden_dim=dec_hidden_dim,
            dropout=dropout,
            cond_dim=self.batch_emb_dim if self.batch_embedding is not None else 0,
        )
        self.num_components = int(num_components)
        self.latent_dim = int(latent_dim)
        self.posterior_cov_rank = max(0, int(posterior_cov_rank))
        self.recon_loss_type = str(recon_loss_type).lower()
        if self.recon_loss_type not in ("poisson", "nb"):
            raise ValueError(f"Unsupported recon_loss_type: {self.recon_loss_type}")
        # EMA mean for optional gene-wise reconstruction reweighting.
        self.register_buffer("recon_gene_mean_ema", torch.zeros(num_genes), persistent=True)
        self.register_buffer("recon_gene_sq_mean_ema", torch.zeros(num_genes), persistent=True)
        if self.prior_type == "gmm":
            post_hidden = max(64, int(expr_hidden_dim) // 2)
            self.mfa_factor_dim = int(self.prior.R)
            self.post_c_logits = MLP(
                input_dim=expr_hidden_dim,
                hidden_dims=[post_hidden],
                output_dim=num_components,
                dropout=dropout,
            )
            if self.mfa_factor_dim > 0:
                self.post_u_mu = MLP(
                    input_dim=expr_hidden_dim,
                    hidden_dims=[post_hidden],
                    output_dim=num_components * self.mfa_factor_dim,
                    dropout=dropout,
                )
                self.post_u_logvar = MLP(
                    input_dim=expr_hidden_dim,
                    hidden_dims=[post_hidden],
                    output_dim=num_components * self.mfa_factor_dim,
                    dropout=dropout,
                )
            else:
                self.post_u_mu = None
                self.post_u_logvar = None
            self.post_eps_mu = MLP(
                input_dim=expr_hidden_dim,
                hidden_dims=[post_hidden],
                output_dim=num_components * latent_dim,
                dropout=dropout,
            )
            self.post_eps_logvar = MLP(
                input_dim=expr_hidden_dim,
                hidden_dims=[post_hidden],
                output_dim=num_components * latent_dim,
                dropout=dropout,
            )
        # Library-size head from latent z.
        lib_hidden = max(32, int(latent_dim) // 2)
        self.library_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, lib_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lib_hidden, 1),
        )
        self.num_cell_types = max(0, int(num_cell_types))
        if self.num_cell_types > 0:
            self.celltype_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.num_cell_types),
            )
        else:
            self.celltype_head = None

    def forward(
        self,
        x_count: torch.Tensor,
        x_mask: torch.Tensor,
        tissue_id: torch.Tensor = None,
        sample_id: torch.Tensor = None,
        x_expr: torch.Tensor = None,
        force_base_posterior: bool = False,
        use_batch_condition: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if x_expr is None:
            x_expr = torch.log1p(x_count.float())
        mu_enc, logvar_enc, h = self.encoder(x_expr=x_expr, x_mask=x_mask.float(), return_hidden=True)
        use_mixture_post = (self.prior_type == "gmm") and (not force_base_posterior)
        if use_mixture_post:
            bsz = h.size(0)
            k = self.num_components
            d = self.latent_dim
            q_c_logits = self.post_c_logits(h)  # (B, K)
            q_c = torch.softmax(q_c_logits, dim=-1)
            prior_mu = self.prior.prior_mu.to(device=h.device, dtype=h.dtype)  # (K, D)
            prior_factor = self.prior._expanded_factor()
            if prior_factor is not None:
                prior_factor = prior_factor.to(device=h.device, dtype=h.dtype)  # (K, D, R)
            eps_mu = self.post_eps_mu(h).view(bsz, k, d)
            eps_logvar = torch.clamp(self.post_eps_logvar(h).view(bsz, k, d), min=-8.0, max=8.0)
            eps_comp = eps_mu + torch.randn_like(eps_mu) * torch.exp(0.5 * eps_logvar)
            if self.mfa_factor_dim > 0 and self.post_u_mu is not None and prior_factor is not None:
                r = self.mfa_factor_dim
                u_mu = self.post_u_mu(h).view(bsz, k, r)
                u_logvar = torch.clamp(self.post_u_logvar(h).view(bsz, k, r), min=-8.0, max=8.0)
                u_comp = u_mu + torch.randn_like(u_mu) * torch.exp(0.5 * u_logvar)
                factor_shift = torch.einsum("kdr,bkr->bkd", prior_factor, u_comp)
                factor_mean_shift = torch.einsum("kdr,bkr->bkd", prior_factor, u_mu)
                factor_var_diag = torch.einsum("kdr,bkr,kdr->bkd", prior_factor, torch.exp(u_logvar), prior_factor)
            else:
                u_mu = None
                u_logvar = None
                u_comp = None
                factor_shift = torch.zeros((bsz, k, d), device=h.device, dtype=h.dtype)
                factor_mean_shift = factor_shift
                factor_var_diag = torch.zeros_like(factor_shift)
            mu_comp = prior_mu.unsqueeze(0) + factor_mean_shift + eps_mu
            z_comp = prior_mu.unsqueeze(0) + factor_shift + eps_comp  # (B, K, D)
            var_comp = torch.clamp(factor_var_diag + torch.exp(eps_logvar), min=1e-8)
            logvar_comp = torch.log(var_comp)
            if self.training:
                c_sel = F.gumbel_softmax(q_c_logits, tau=1.0, hard=False, dim=-1)  # (B, K)
            else:
                hard_idx = torch.argmax(q_c, dim=-1)
                c_sel = F.one_hot(hard_idx, num_classes=k).to(z_comp.dtype)
            z = torch.sum(c_sel.unsqueeze(-1) * z_comp, dim=1)  # (B, D)

            # Moment-matched aggregate posterior stats for compatibility logging.
            mu = torch.sum(q_c.unsqueeze(-1) * mu_comp, dim=1)
            second = torch.sum(q_c.unsqueeze(-1) * (torch.exp(logvar_comp) + mu_comp * mu_comp), dim=1)
            var = torch.clamp(second - mu * mu, min=1e-8)
            logvar = torch.log(var)
        else:
            mu = mu_enc
            logvar = logvar_enc
            z = reparameterize(mu, logvar)
            q_c_logits = None
            q_c = None
            mu_comp = None
            logvar_comp = None
            z_comp = None
            eps_mu = None
            eps_logvar = None
            eps_comp = None
            u_mu = None
            u_logvar = None
            u_comp = None
        batch_cond = self._batch_condition(sample_id=sample_id, use_batch_condition=use_batch_condition)
        gene_logits = self.decoder(z, cond=batch_cond)
        nb_theta_logits = self.nb_theta_decoder(z, cond=batch_cond)
        library_size = F.softplus(self.library_head(z)) + 1e-8
        gene_probs = F.softmax(gene_logits, dim=-1)
        rate = gene_probs * library_size
        nb_theta = F.softplus(nb_theta_logits) + 1e-8
        out = {
            "mu": mu,
            "mu_base": mu_enc,
            "logvar": logvar,
            "z": z,
            "library_size": library_size,
            "gene_logits": gene_logits,
            "rate": rate,
            "nb_theta": nb_theta,
        }
        if self.celltype_head is not None:
            out["celltype_logits"] = self.celltype_head(z)
        if use_mixture_post:
            out.update(
                {
                    "q_c_logits": q_c_logits,
                    "q_c": q_c,
                    "mu_comp": mu_comp,
                    "logvar_comp": logvar_comp,
                    "z_comp": z_comp,
                    "eps_mu": eps_mu,
                    "eps_logvar": eps_logvar,
                    "eps_comp": eps_comp,
                    "u_mu": u_mu,
                    "u_logvar": u_logvar,
                    "u_comp": u_comp,
                }
            )
        return out

    def _batch_condition(self, sample_id: torch.Tensor = None, use_batch_condition: bool = True) -> Optional[torch.Tensor]:
        if self.batch_embedding is None or sample_id is None or not bool(use_batch_condition):
            return None
        sid = sample_id.long().to(self.batch_embedding.weight.device)
        sid = torch.where(sid < 0, torch.zeros_like(sid), sid)
        sid = torch.where(sid >= self.num_batches, torch.zeros_like(sid), sid)
        emb = self.batch_embedding(sid)
        if self.training and self.batch_cond_drop_prob > 0:
            keep = (torch.rand((emb.size(0), 1), device=emb.device) >= self.batch_cond_drop_prob).to(emb.dtype)
            emb = emb * keep
        return emb

    def loss(
        self,
        x_count: torch.Tensor,
        x_mask: torch.Tensor,
        tissue_id: torch.Tensor = None,
        sample_id: torch.Tensor = None,
        celltype_id: torch.Tensor = None,
        force_base_posterior: bool = False,
        beta: float = 1.0,
        beta_u_kl_multiplier: float = 1.0,
        beta_eps_kl_multiplier: float = 1.0,
        encoder_mask: torch.Tensor = None,
        use_batch_condition: bool = True,
        recon_mask: torch.Tensor = None,
        lambda_score: float = 0.0,
        score_noise_std: float = 0.1,
        score_detach_z: bool = True,
        lambda_cov: float = 0.0,
        cov_use_mu: bool = True,
        lambda_resp_balance: float = 0.0,
        lambda_resp_confidence: float = 0.0,
        lambda_resp_anchor: float = 0.0,
        resp_temperature: float = 1.0,
        resp_topk: int = 0,
        prior_logvar_min: float = -6.0,
        prior_logvar_max: float = 4.0,
        lambda_prior_mu_l2: float = 0.0,
        lambda_prior_factor_l2: float = 0.0,
        lambda_prior_pi_balance: float = 0.0,
        lambda_prior_mu_spread: float = 0.0,
        prior_mu_spread_tau: float = 1.0,
        lambda_post_c_balance: float = 0.0,
        lambda_celltype_cls: float = 0.0,
        lambda_prior_logvar_l2: float = 0.0,
        prior_logvar_target: float = -2.0,
        recon_gene_weight_mode: str = "none",
        recon_gene_weight_alpha: float = 0.0,
        recon_gene_weight_ema_momentum: float = 0.99,
        recon_gene_weight_min: float = 0.3,
        recon_gene_weight_max: float = 3.0,
        recon_gene_weight_eps: float = 1e-6,
        recon_cell_weight_mode: str = "none",
        recon_cell_weight_alpha: float = 0.0,
        recon_cell_weight_min: float = 0.5,
        recon_cell_weight_max: float = 2.0,
        recon_cell_weight_eps: float = 1e-6,
        recon_cell_weight_clusters: int = 32,
        recon_cell_weight_kmeans_iters: int = 2,
    ) -> Dict[str, torch.Tensor]:
        if encoder_mask is None:
            encoder_mask = x_mask
        out = self.forward(
            x_count=x_count,
            x_mask=encoder_mask,
            tissue_id=tissue_id,
            sample_id=sample_id,
            force_base_posterior=force_base_posterior,
            use_batch_condition=use_batch_condition,
        )

        mu = out["mu"]
        logvar = out["logvar"]
        z = out["z"]
        rate = out["rate"]
        nb_theta = out["nb_theta"]

        gene_weight = self._build_recon_gene_weight(
            x_count=x_count,
            x_mask=recon_mask if recon_mask is not None else x_mask,
            mode=recon_gene_weight_mode,
            alpha=recon_gene_weight_alpha,
            ema_momentum=recon_gene_weight_ema_momentum,
            w_min=recon_gene_weight_min,
            w_max=recon_gene_weight_max,
            eps=recon_gene_weight_eps,
        )
        q_c_for_weight = None
        if self.prior_type == "gmm" and (not force_base_posterior):
            q_c_for_weight = out.get("q_c", None)
        cell_weight = self._build_recon_cell_weight(
            q_c=q_c_for_weight,
            z=z,
            mode=recon_cell_weight_mode,
            alpha=recon_cell_weight_alpha,
            w_min=recon_cell_weight_min,
            w_max=recon_cell_weight_max,
            eps=recon_cell_weight_eps,
            num_clusters=recon_cell_weight_clusters,
            kmeans_iters=recon_cell_weight_kmeans_iters,
        )
        if self.recon_loss_type == "nb":
            if recon_mask is None:
                recon_loss = nb_nll(x_count=x_count, mu=rate, theta=nb_theta, gene_weight=gene_weight, cell_weight=cell_weight)
            else:
                recon_loss = nb_nll_masked(x_count=x_count, mu=rate, mask=recon_mask, theta=nb_theta, gene_weight=gene_weight, cell_weight=cell_weight)
        else:
            if recon_mask is None:
                recon_loss = poisson_nll(x_count=x_count, rate=rate, gene_weight=gene_weight, cell_weight=cell_weight)
            else:
                recon_loss = poisson_nll_masked(x_count=x_count, rate=rate, mask=recon_mask, gene_weight=gene_weight, cell_weight=cell_weight)

        if self.prior_type == "gaussian" or (self.prior_type == "gmm" and force_base_posterior):
            # Closed-form KL(q(z|x)||N(0,I)) for diagonal Gaussian posterior.
            kl_per_cell = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
            kl_loss = kl_per_cell.mean()
            kl_c = torch.zeros((), device=z.device, dtype=z.dtype)
            kl_u = torch.zeros((), device=z.device, dtype=z.dtype)
            kl_eps = kl_loss
            log_q = gaussian_log_prob_diag(z=z, mu=mu, logvar=logvar)
            zero_mu = torch.zeros_like(z)
            zero_logvar = torch.zeros_like(z)
            log_p = gaussian_log_prob_diag(z=z, mu=zero_mu, logvar=zero_logvar)
        else:
            q_c_logits = out["q_c_logits"]  # (B, K)
            q_c = out["q_c"]  # (B, K)
            eps_mu = out["eps_mu"]  # (B, K, D)
            eps_logvar = out["eps_logvar"]  # (B, K, D)
            eps_comp = out["eps_comp"]  # (B, K, D)
            u_mu = out.get("u_mu", None)  # (B, K, R) or None
            u_logvar = out.get("u_logvar", None)  # (B, K, R) or None
            u_comp = out.get("u_comp", None)  # (B, K, R) or None

            log_q_c = F.log_softmax(q_c_logits, dim=-1)  # (B, K)
            if getattr(self.prior, "conditional_on_tissue", False) and tissue_id is not None:
                tid = self.prior._sanitize_tissue_id(tissue_id.to(mu.device))
                log_p_c = F.log_softmax(self.prior.pi_logits_t[tid], dim=-1)  # (B, K)
            else:
                log_p_c = F.log_softmax(self.prior.pi_logits, dim=0).unsqueeze(0)  # (1, K)
            kl_c = (q_c * (log_q_c - log_p_c)).sum(dim=1).mean()

            if u_mu is not None and u_logvar is not None and u_comp is not None:
                kl_u_per_comp = 0.5 * torch.sum(
                    torch.exp(u_logvar.float()) + u_mu.float().pow(2) - 1.0 - u_logvar.float(),
                    dim=-1,
                )  # (B, K)
                zero_u = torch.zeros_like(u_comp)
                zero_u_logvar = torch.zeros_like(u_comp)
                log_q_u = gaussian_log_prob_diag(z=u_comp, mu=u_mu, logvar=u_logvar)
                log_p_u = gaussian_log_prob_diag(z=u_comp, mu=zero_u, logvar=zero_u_logvar)
            else:
                kl_u_per_comp = torch.zeros_like(q_c.float())
                log_q_u = torch.zeros_like(q_c)
                log_p_u = torch.zeros_like(q_c)

            prior_eps_logvar = self.prior._expanded_logvar(
                logvar_min=prior_logvar_min,
                logvar_max=prior_logvar_max,
            ).to(device=eps_mu.device, dtype=eps_mu.dtype)  # (K, D)
            prior_eps_logvar_b = prior_eps_logvar.unsqueeze(0)
            kl_eps_per_comp = 0.5 * torch.sum(
                (
                    torch.exp(eps_logvar.float())
                    + eps_mu.float().pow(2)
                )
                * torch.exp(-prior_eps_logvar_b.float())
                - 1.0
                + prior_eps_logvar_b.float()
                - eps_logvar.float(),
                dim=-1,
            )  # (B, K)
            zero_eps = torch.zeros_like(eps_comp)
            log_q_eps = gaussian_log_prob_diag(z=eps_comp, mu=eps_mu, logvar=eps_logvar)
            log_p_eps = gaussian_log_prob_diag(
                z=eps_comp,
                mu=zero_eps,
                logvar=prior_eps_logvar_b.expand_as(eps_comp),
            )

            kl_u = (q_c.float() * kl_u_per_comp).sum(dim=1).mean().to(z.dtype)
            kl_eps = (q_c.float() * kl_eps_per_comp).sum(dim=1).mean().to(z.dtype)
            u_kl_mult = max(float(beta_u_kl_multiplier), 0.0)
            eps_kl_mult = max(float(beta_eps_kl_multiplier), 0.0)
            kl_z = u_kl_mult * kl_u + eps_kl_mult * kl_eps

            kl_loss = kl_c + kl_z
            log_q = (q_c * (log_q_c + log_q_u + log_q_eps)).sum(dim=1)
            log_p = (q_c * (log_p_c + log_p_u + log_p_eps)).sum(dim=1)
        score_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        score_norm_pred = torch.zeros((), device=z.device, dtype=z.dtype)
        score_norm_tgt = torch.zeros((), device=z.device, dtype=z.dtype)
        cov_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        cov_offdiag_post = torch.zeros((), device=z.device, dtype=z.dtype)
        cov_offdiag_prior = torch.zeros((), device=z.device, dtype=z.dtype)
        resp_entropy = torch.zeros((), device=z.device, dtype=z.dtype)
        resp_top1 = torch.zeros((), device=z.device, dtype=z.dtype)
        resp_balance_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        resp_confidence_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        resp_anchor_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        prior_mu_l2_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        prior_factor_l2_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        prior_pi_balance_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        prior_mu_spread_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        post_c_balance_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        celltype_cls_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        prior_logvar_l2_loss = torch.zeros((), device=z.device, dtype=z.dtype)

        # score / covariance alignment / posterior-balance constraints are removed.

        if self.prior_type == "gmm" and (lambda_prior_mu_l2 > 0 or lambda_prior_factor_l2 > 0):
            if lambda_prior_mu_l2 > 0:
                if getattr(self.prior, "conditional_on_tissue", False) and getattr(self.prior, "prior_mu_t", None) is not None:
                    prior_mu_l2_loss = self.prior.prior_mu_t.float().pow(2).mean().to(z.dtype)
                else:
                    prior_mu_l2_loss = self.prior.prior_mu.float().pow(2).mean().to(z.dtype)
            if lambda_prior_factor_l2 > 0 and getattr(self.prior, "prior_factor", None) is not None:
                if getattr(self.prior, "conditional_on_tissue", False) and getattr(self.prior, "prior_factor_t", None) is not None:
                    prior_factor_l2_loss = self.prior.prior_factor_t.float().pow(2).mean().to(z.dtype)
                else:
                    prior_factor_l2_loss = self.prior.prior_factor.float().pow(2).mean().to(z.dtype)
        if self.prior_type == "gmm" and lambda_prior_pi_balance > 0:
            if getattr(self.prior, "conditional_on_tissue", False) and getattr(self.prior, "pi_logits_t", None) is not None:
                pi = torch.softmax(self.prior.pi_logits_t.float(), dim=-1).mean(dim=0)
            else:
                pi = torch.softmax(self.prior.pi_logits.float(), dim=0)
            target = torch.full_like(pi, 1.0 / float(pi.numel()))
            prior_pi_balance_loss = F.kl_div(
                torch.log(torch.clamp(pi, min=1e-12)),
                target,
                reduction="sum",
            ).to(z.dtype)
        if self.prior_type == "gmm" and lambda_prior_mu_spread > 0:
            if getattr(self.prior, "conditional_on_tissue", False) and getattr(self.prior, "prior_mu_t", None) is not None:
                prior_mu_for_spread = self.prior.prior_mu_t.float().reshape(-1, self.prior.prior_mu_t.size(-1))
            else:
                prior_mu_for_spread = self.prior.prior_mu.float()
            if prior_mu_for_spread.size(0) > 1:
                d2 = torch.cdist(prior_mu_for_spread, prior_mu_for_spread, p=2).pow(2)
                eye = torch.eye(d2.size(0), device=d2.device, dtype=torch.bool)
                off = d2.masked_fill(eye, float("inf"))
                tau = max(float(prior_mu_spread_tau), 1e-6)
                prior_mu_spread_loss = torch.exp(-off / tau).mean().to(z.dtype)
        if self.prior_type == "gmm" and lambda_post_c_balance > 0 and (not force_base_posterior):
            q_c_cur = out.get("q_c", None)
            if q_c_cur is not None:
                q_mean = q_c_cur.float().mean(dim=0)
                target = torch.full_like(q_mean, 1.0 / float(q_mean.numel()))
                post_c_balance_loss = F.kl_div(
                    torch.log(torch.clamp(q_mean, min=1e-12)),
                    target,
                    reduction="sum",
                ).to(z.dtype)
        if self.prior_type == "gmm" and lambda_prior_logvar_l2 > 0:
            if getattr(self.prior, "conditional_on_tissue", False) and getattr(self.prior, "prior_logvar_t", None) is not None:
                tgt = torch.full_like(self.prior.prior_logvar_t, float(prior_logvar_target)).float()
                prior_logvar_l2_loss = F.mse_loss(self.prior.prior_logvar_t.float(), tgt).to(z.dtype)
            else:
                tgt = torch.full_like(self.prior.prior_logvar, float(prior_logvar_target)).float()
                prior_logvar_l2_loss = F.mse_loss(self.prior.prior_logvar.float(), tgt).to(z.dtype)
        if lambda_celltype_cls > 0 and celltype_id is not None and self.celltype_head is not None:
            ct = celltype_id.view(-1).to(z.device).long()
            valid = ct >= 0
            if valid.any():
                logits = out["celltype_logits"][valid]
                tgt = ct[valid]
                celltype_cls_loss = F.cross_entropy(logits, tgt, label_smoothing=0.05).to(z.dtype)

        # Build total loss with explicit gating to avoid 0 * inf -> nan when a term is disabled.
        total_loss = recon_loss
        if float(beta) != 0.0:
            total_loss = total_loss + float(beta) * kl_loss
        if float(lambda_score) != 0.0:
            total_loss = total_loss + float(lambda_score) * score_loss
        # lambda_cov is intentionally ignored.
        if float(lambda_resp_balance) != 0.0:
            total_loss = total_loss + float(lambda_resp_balance) * resp_balance_loss
        if float(lambda_resp_confidence) != 0.0:
            total_loss = total_loss + float(lambda_resp_confidence) * resp_confidence_loss
        # lambda_resp_anchor is intentionally ignored.
        if float(lambda_prior_mu_l2) != 0.0:
            total_loss = total_loss + float(lambda_prior_mu_l2) * prior_mu_l2_loss
        if float(lambda_prior_factor_l2) != 0.0:
            total_loss = total_loss + float(lambda_prior_factor_l2) * prior_factor_l2_loss
        if float(lambda_prior_pi_balance) != 0.0:
            total_loss = total_loss + float(lambda_prior_pi_balance) * prior_pi_balance_loss
        if float(lambda_prior_mu_spread) != 0.0:
            total_loss = total_loss + float(lambda_prior_mu_spread) * prior_mu_spread_loss
        if float(lambda_post_c_balance) != 0.0:
            total_loss = total_loss + float(lambda_post_c_balance) * post_c_balance_loss
        if float(lambda_celltype_cls) != 0.0:
            total_loss = total_loss + float(lambda_celltype_cls) * celltype_cls_loss
        if float(lambda_prior_logvar_l2) != 0.0:
            total_loss = total_loss + float(lambda_prior_logvar_l2) * prior_logvar_l2_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_c_loss": kl_c.to(z.dtype),
            "kl_u_loss": kl_u.to(z.dtype),
            "kl_eps_loss": kl_eps.to(z.dtype),
            "score_loss": score_loss,
            "cov_loss": cov_loss,
            "cov_offdiag_post": cov_offdiag_post,
            "cov_offdiag_prior": cov_offdiag_prior,
            "resp_entropy": resp_entropy,
            "resp_top1": resp_top1,
            "resp_balance_loss": resp_balance_loss,
            "resp_confidence_loss": resp_confidence_loss,
            "resp_anchor_loss": resp_anchor_loss,
            "prior_mu_l2_loss": prior_mu_l2_loss,
            "prior_factor_l2_loss": prior_factor_l2_loss,
            "prior_pi_balance_loss": prior_pi_balance_loss,
            "prior_mu_spread_loss": prior_mu_spread_loss,
            "post_c_balance_loss": post_c_balance_loss,
            "celltype_cls_loss": celltype_cls_loss,
            "prior_logvar_l2_loss": prior_logvar_l2_loss,
            "score_norm_pred": score_norm_pred,
            "score_norm_tgt": score_norm_tgt,
            "log_q_mean": log_q.mean(),
            "log_p_mean": log_p.mean(),
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "rate": rate,
        }

    def _build_recon_gene_weight(
        self,
        x_count: torch.Tensor,
        x_mask: torch.Tensor,
        mode: str = "none",
        alpha: float = 0.0,
        ema_momentum: float = 0.99,
        w_min: float = 0.3,
        w_max: float = 3.0,
        eps: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        mode = str(mode).lower()
        alpha = float(alpha)
        if mode == "none" or alpha <= 0.0:
            return None
        if mode not in ("inv_log1p_mean_ema", "cv_ema"):
            return None

        with torch.no_grad():
            obs = torch.clamp(x_mask.float(), min=0.0, max=1.0)
            cnt = torch.clamp(x_count.float(), min=0.0)
            obs_sum = obs.sum(dim=0)
            mean_g = (cnt * obs).sum(dim=0) / torch.clamp(obs_sum, min=1.0)
            sq_mean_g = (cnt.square() * obs).sum(dim=0) / torch.clamp(obs_sum, min=1.0)
            if self.training:
                m = max(0.0, min(0.9999, float(ema_momentum)))
                self.recon_gene_mean_ema.mul_(m).add_((1.0 - m) * mean_g.detach())
                self.recon_gene_sq_mean_ema.mul_(m).add_((1.0 - m) * sq_mean_g.detach())
            base = self.recon_gene_mean_ema if torch.any(self.recon_gene_mean_ema > 0) else mean_g
            if mode == "cv_ema":
                sq_base = self.recon_gene_sq_mean_ema if torch.any(self.recon_gene_sq_mean_ema > 0) else sq_mean_g
                var = torch.clamp(sq_base - base.square(), min=0.0)
                cv = torch.sqrt(var + float(eps)) / (base + float(eps))
                w = 1.0 + torch.log1p(cv)
            else:
                w = 1.0 / torch.log1p(torch.clamp(base, min=float(eps)) + float(eps))
            w = w / torch.clamp(w.mean(), min=float(eps))
            w = torch.clamp(w, min=float(w_min), max=float(w_max))
            w = (1.0 - alpha) + alpha * w
        return w.to(dtype=x_count.dtype, device=x_count.device)

    def _build_recon_cell_weight(
        self,
        q_c: Optional[torch.Tensor],
        z: Optional[torch.Tensor] = None,
        mode: str = "none",
        alpha: float = 0.0,
        w_min: float = 0.5,
        w_max: float = 2.0,
        eps: float = 1e-6,
        num_clusters: int = 32,
        kmeans_iters: int = 2,
    ) -> Optional[torch.Tensor]:
        mode = str(mode).lower()
        alpha = float(alpha)
        if mode == "none" or alpha <= 0.0:
            return None
        if mode not in ("component_usage", "batch_kmeans"):
            return None

        with torch.no_grad():
            if mode == "component_usage":
                if q_c is None:
                    return None
                qc = q_c.detach().float()
                usage = qc.mean(dim=0)
                cell_usage = (qc * usage.view(1, -1)).sum(dim=1)
                w = torch.rsqrt(torch.clamp(cell_usage, min=float(eps)))
                dtype = q_c.dtype
                device = q_c.device
            else:
                if z is None or z.size(0) <= 1:
                    return None
                zf = F.normalize(z.detach().float(), dim=-1)
                bsz = int(zf.size(0))
                k = max(1, min(int(num_clusters), bsz))
                stride = max(1, bsz // k)
                centers = zf[torch.arange(0, stride * k, stride, device=zf.device)[:k]].clone()
                assign = torch.zeros((bsz,), dtype=torch.long, device=zf.device)
                for _ in range(max(1, int(kmeans_iters))):
                    assign = torch.argmin(torch.cdist(zf, centers, p=2), dim=1)
                    new_centers = torch.zeros_like(centers)
                    counts = torch.bincount(assign, minlength=k).to(zf.dtype)
                    new_centers.index_add_(0, assign, zf)
                    non_empty = counts > 0
                    new_centers[non_empty] = new_centers[non_empty] / counts[non_empty].unsqueeze(1)
                    if (~non_empty).any():
                        new_centers[~non_empty] = centers[~non_empty]
                    centers = F.normalize(new_centers, dim=-1)
                counts = torch.bincount(assign, minlength=k).to(zf.dtype)
                cell_usage = counts[assign] / float(bsz)
                w = torch.rsqrt(torch.clamp(cell_usage, min=float(eps)))
                dtype = z.dtype
                device = z.device
            w = w / torch.clamp(w.mean(), min=float(eps))
            w = torch.clamp(w, min=float(w_min), max=float(w_max))
            w = (1.0 - alpha) + alpha * w
        return w.to(dtype=dtype, device=device)


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


def gaussian_log_prob_lowrank(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """
    Log-prob for per-sample/per-component low-rank Gaussian:
      Sigma = diag(exp(logvar)) + U U^T
    Shapes:
      z, mu, logvar: (B, K, D)
      factor: (B, K, D, R)
    Returns:
      log_prob: (B, K)
    """
    device_type = z.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        zf = z.float()
        muf = mu.float()
        logvarf = logvar.float()
        factorf = factor.float()

        d_inv = torch.exp(-logvarf)  # (B, K, D)
        delta = zf - muf  # (B, K, D)
        delta_d = delta * d_inv  # (B, K, D)
        quad = (delta * delta_d).sum(dim=-1)  # (B, K)

        # U^T D^{-1} U  -> (B, K, R, R)
        dinv_u = d_inv.unsqueeze(-1) * factorf
        ut_dinv_u = torch.einsum("bkdr,bkds->bkrs", factorf, dinv_u)
        r = int(factorf.size(-1))
        eye = torch.eye(r, device=factorf.device, dtype=factorf.dtype).view(1, 1, r, r)
        s = ut_dinv_u + eye + 1e-6 * eye
        chol_s = torch.linalg.cholesky(s)
        inv_s = torch.cholesky_inverse(chol_s)
        logdet_extra = 2.0 * torch.log(torch.diagonal(chol_s, dim1=-2, dim2=-1)).sum(dim=-1)  # (B, K)

        t = torch.einsum("bkd,bkdr->bkr", delta_d, factorf)  # (B, K, R)
        quad_corr = torch.einsum("bkr,bkrs,bks->bk", t, inv_s, t)
        quad = quad - quad_corr

        log_det = logvarf.sum(dim=-1) + logdet_extra  # (B, K)
        d = zf.size(-1)
        out = -0.5 * (quad + log_det + d * math.log(2.0 * math.pi))
    return out.to(z.dtype)


def batch_covariance(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor for covariance, got shape={tuple(x.shape)}")
    bsz = x.size(0)
    xc = x - x.mean(dim=0, keepdim=True)
    denom = max(bsz - 1, 1)
    return (xc.transpose(0, 1) @ xc) / float(denom)


def offdiag_part(m: torch.Tensor) -> torch.Tensor:
    return m - torch.diag_embed(torch.diagonal(m, dim1=-2, dim2=-1))


RECON_RATE_MIN = 1e-8
RECON_RATE_MAX = 1e6
RECON_THETA_MIN = 1e-6
RECON_THETA_MAX = 1e6


def _safe_count_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=RECON_RATE_MAX, neginf=0.0).clamp_min(0.0)


def _safe_positive_tensor(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=lower, posinf=upper, neginf=lower).clamp(min=lower, max=upper)


def _safe_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)


def _safe_weight_vector(weight: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = weight.view(-1).to(device=device, dtype=dtype)
    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _safe_nll_tensor(nll: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(nll, nan=RECON_RATE_MAX, posinf=RECON_RATE_MAX, neginf=0.0)


def poisson_nll(
    x_count: torch.Tensor,
    rate: torch.Tensor,
    gene_weight: torch.Tensor = None,
    cell_weight: torch.Tensor = None,
) -> torch.Tensor:
    device_type = x_count.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        x = _safe_count_tensor(x_count)
        r = _safe_positive_tensor(rate, RECON_RATE_MIN, RECON_RATE_MAX)
        nll = _safe_nll_tensor(r - x * torch.log(r))
        if gene_weight is not None:
            gw = _safe_weight_vector(gene_weight, nll.device, nll.dtype)
            nll = nll * gw.view(1, -1)
        loss_cell = nll.mean(dim=1)
        if cell_weight is not None:
            cw = _safe_weight_vector(cell_weight, loss_cell.device, loss_cell.dtype)
            loss_cell = loss_cell * cw
            return loss_cell.sum() / torch.clamp(cw.sum(), min=1e-8)
        return loss_cell.mean()


def poisson_nll_masked(
    x_count: torch.Tensor,
    rate: torch.Tensor,
    mask: torch.Tensor,
    gene_weight: torch.Tensor = None,
    cell_weight: torch.Tensor = None,
) -> torch.Tensor:
    device_type = x_count.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        x = _safe_count_tensor(x_count)
        r = _safe_positive_tensor(rate, RECON_RATE_MIN, RECON_RATE_MAX)
        m = _safe_mask_tensor(mask)
        nll = _safe_nll_tensor(r - x * torch.log(r))
        if gene_weight is not None:
            gw = _safe_weight_vector(gene_weight, nll.device, nll.dtype)
            nll = nll * gw.view(1, -1)
        nll = nll * m
        denom_cell = torch.clamp(m.sum(dim=1), min=1.0)
        loss_cell = nll.sum(dim=1) / denom_cell
        if cell_weight is not None:
            cw = _safe_weight_vector(cell_weight, loss_cell.device, loss_cell.dtype)
            loss_cell = loss_cell * cw
            return loss_cell.sum() / torch.clamp(cw.sum(), min=1e-8)
        return loss_cell.mean()


def nb_nll(
    x_count: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    gene_weight: torch.Tensor = None,
    cell_weight: torch.Tensor = None,
) -> torch.Tensor:
    """
    Negative binomial NLL with mean mu and inverse-dispersion theta.
    log_theta is gene-wise parameter of shape (G,).
    """
    device_type = x_count.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        x = _safe_count_tensor(x_count)
        m = _safe_positive_tensor(mu, RECON_RATE_MIN, RECON_RATE_MAX)
        theta = _safe_positive_tensor(theta, RECON_THETA_MIN, RECON_THETA_MAX)
        log_theta_mu = torch.log(theta + m)
        log_prob = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
            + theta * (torch.log(theta) - log_theta_mu)
            + x * (torch.log(m) - log_theta_mu)
        )
        nll = _safe_nll_tensor(-log_prob)
        if gene_weight is not None:
            gw = _safe_weight_vector(gene_weight, nll.device, nll.dtype)
            nll = nll * gw.view(1, -1)
        loss_cell = nll.mean(dim=1)
        if cell_weight is not None:
            cw = _safe_weight_vector(cell_weight, loss_cell.device, loss_cell.dtype)
            loss_cell = loss_cell * cw
            return loss_cell.sum() / torch.clamp(cw.sum(), min=1e-8)
        return loss_cell.mean()


def nb_nll_masked(
    x_count: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    theta: torch.Tensor,
    gene_weight: torch.Tensor = None,
    cell_weight: torch.Tensor = None,
) -> torch.Tensor:
    device_type = x_count.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        x = _safe_count_tensor(x_count)
        m = _safe_positive_tensor(mu, RECON_RATE_MIN, RECON_RATE_MAX)
        ms = _safe_mask_tensor(mask)
        theta = _safe_positive_tensor(theta, RECON_THETA_MIN, RECON_THETA_MAX)
        log_theta_mu = torch.log(theta + m)
        log_prob = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
            + theta * (torch.log(theta) - log_theta_mu)
            + x * (torch.log(m) - log_theta_mu)
        )
        nll = _safe_nll_tensor(-log_prob)
        if gene_weight is not None:
            gw = _safe_weight_vector(gene_weight, nll.device, nll.dtype)
            nll = nll * gw.view(1, -1)
        nll = nll * ms
        denom_cell = torch.clamp(ms.sum(dim=1), min=1.0)
        loss_cell = nll.sum(dim=1) / denom_cell
        if cell_weight is not None:
            cw = _safe_weight_vector(cell_weight, loss_cell.device, loss_cell.dtype)
            loss_cell = loss_cell * cw
            return loss_cell.sum() / torch.clamp(cw.sum(), min=1e-8)
        return loss_cell.mean()


def bidirectional_contrastive_loss(z_real: torch.Tensor, z_fake: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Symmetric InfoNCE:
    - z_real[i] <-> z_fake[i] is a positive pair
    - other samples in batch are negatives
    """
    if z_real.size(0) <= 1:
        return torch.zeros((), device=z_real.device, dtype=z_real.dtype)
    z1 = F.normalize(z_real, dim=-1)
    z2 = F.normalize(z_fake, dim=-1)
    logits = torch.matmul(z1, z2.transpose(0, 1)) / max(temperature, 1e-6)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_12 + loss_21)


def deterministic_contrast_embedding(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    if ("q_c" in out) and ("mu_comp" in out):
        return torch.sum(out["q_c"].unsqueeze(-1) * out["mu_comp"], dim=1)
    return out["mu"]


def gmm_collapse_diagnostics(
    prior: GaussianMixturePrior,
    z: torch.Tensor,
    tissue_id: torch.Tensor = None,
    active_thresh: float = 1e-3,
) -> Dict[str, float]:
    with torch.no_grad():
        if getattr(prior, "conditional_on_tissue", False) and tissue_id is not None:
            tid = prior._sanitize_tissue_id(tissue_id.to(z.device))
            pi = torch.softmax(prior.pi_logits_t.float()[tid], dim=-1).mean(dim=0)  # (K,)
        else:
            pi = torch.softmax(prior.pi_logits.float(), dim=0)  # (K,)
        pi_entropy = float((-(pi * torch.log(pi + 1e-12))).sum().item())
        k_eff = float(torch.exp(torch.tensor(pi_entropy, device=pi.device)).item())

        log_w = torch.log(pi + 1e-12).unsqueeze(0)  # (1, K)
        log_comp = prior.component_log_prob(z.detach(), tissue_id=tissue_id)  # (B, K)
        resp = torch.softmax(log_w + log_comp, dim=1)  # (B, K)
        usage = resp.mean(dim=0)  # (K,)
        active_comp = int((usage > float(active_thresh)).sum().item())
        resp_top1 = float(resp.max(dim=1).values.mean().item())

        mu = prior.prior_mu.float()  # (K, D)
        mean_prior_mu_norm = float(mu.norm(dim=1).mean().item())
        if mu.size(0) > 1:
            dmat = torch.cdist(mu, mu, p=2)
            diag_mask = torch.eye(dmat.size(0), device=dmat.device, dtype=torch.bool)
            off_dmat = dmat.masked_fill(diag_mask, float("nan"))
            finite_dists = off_dmat[~torch.isnan(off_dmat)]
            min_mu_dist = float(finite_dists.min().item()) if finite_dists.numel() > 0 else 0.0
            mean_mu_dist = float(finite_dists.mean().item()) if finite_dists.numel() > 0 else 0.0
        else:
            min_mu_dist = 0.0
            mean_mu_dist = 0.0
        prior_logvar = prior._expanded_logvar().float()
        prior_var = torch.exp(prior_logvar)
        mean_prior_var = float(prior_var.mean().item())
        max_prior_var = float(prior_var.max().item())
        mean_prior_std = float(torch.sqrt(prior_var).mean().item())
        factor = prior._expanded_factor()
        if factor is not None:
            factor = factor.float()
            factor_var_diag = torch.sum(factor * factor, dim=-1)  # diag(A A^T), (K, D)
            mean_factor_var = float(factor_var_diag.mean().item())
            max_factor_var = float(factor_var_diag.max().item())
            mean_factor_norm = float(factor.norm(dim=(1, 2)).mean().item())
            total_var = prior_var + factor_var_diag
            mean_total_var = float(total_var.mean().item())
            max_total_var = float(total_var.max().item())
            mean_total_std = float(torch.sqrt(total_var).mean().item())
        else:
            mean_factor_var = 0.0
            max_factor_var = 0.0
            mean_factor_norm = 0.0
            mean_total_var = mean_prior_var
            max_total_var = max_prior_var
            mean_total_std = mean_prior_std

        group_k_eff = 0.0
        group_active = 0
        group_resp_top1 = 0.0
        sub_k_eff_mean = 0.0
        sub_active_mean = 0.0
        sub_active_min = 0.0
        sub_active_max = 0.0
        group_mu_min_dist = 0.0
        group_mu_mean_dist = 0.0
        sub_delta_norm_mean = 0.0
        sub_delta_norm_max = 0.0
        if hasattr(prior, "component_group") and hasattr(prior, "component_sub"):
            comp_group = prior.component_group.to(usage.device)
            comp_sub = prior.component_sub.to(usage.device)
            g_count = int(getattr(prior, "G", int(comp_group.max().item()) + 1))
            s_count = int(getattr(prior, "S", int(comp_sub.max().item()) + 1))
            group_usage = torch.zeros((g_count,), device=usage.device, dtype=usage.dtype)
            group_usage.scatter_add_(0, comp_group, usage)
            group_entropy = -(group_usage * torch.log(group_usage + 1e-12)).sum()
            group_k_eff = float(torch.exp(group_entropy).item())
            group_active = int((group_usage > float(active_thresh)).sum().item())
            resp_group = torch.zeros((resp.size(0), g_count), device=resp.device, dtype=resp.dtype)
            resp_group.scatter_add_(1, comp_group.view(1, -1).expand(resp.size(0), -1), resp)
            group_resp_top1 = float(resp_group.max(dim=1).values.mean().item())

            sub_eff_rows = []
            sub_active_rows = []
            for gg in range(g_count):
                mask_g = comp_group == gg
                if not bool(mask_g.any()):
                    continue
                sub_usage = torch.zeros((s_count,), device=usage.device, dtype=usage.dtype)
                sub_usage.scatter_add_(0, comp_sub[mask_g], usage[mask_g])
                mass = torch.clamp(sub_usage.sum(), min=1e-12)
                sub_usage = sub_usage / mass
                sub_entropy = -(sub_usage * torch.log(sub_usage + 1e-12)).sum()
                sub_eff_rows.append(torch.exp(sub_entropy))
                sub_active_rows.append((sub_usage > float(active_thresh)).sum().float())
            if sub_eff_rows:
                sub_eff = torch.stack(sub_eff_rows)
                sub_act = torch.stack(sub_active_rows)
                sub_k_eff_mean = float(sub_eff.mean().item())
                sub_active_mean = float(sub_act.mean().item())
                sub_active_min = float(sub_act.min().item())
                sub_active_max = float(sub_act.max().item())

            if hasattr(prior, "group_mu"):
                group_mu = prior.group_mu.float()
                if group_mu.size(0) > 1:
                    gd = torch.cdist(group_mu, group_mu, p=2)
                    eye_g = torch.eye(gd.size(0), device=gd.device, dtype=torch.bool)
                    goff = gd.masked_fill(eye_g, float("nan"))
                    gfinite = goff[~torch.isnan(goff)]
                    if gfinite.numel() > 0:
                        group_mu_min_dist = float(gfinite.min().item())
                        group_mu_mean_dist = float(gfinite.mean().item())
            if hasattr(prior, "sub_delta"):
                sd_norm = prior.sub_delta.float().norm(dim=-1)
                sub_delta_norm_mean = float(sd_norm.mean().item())
                sub_delta_norm_max = float(sd_norm.max().item())

    return {
        "pi_entropy": pi_entropy,
        "k_eff": k_eff,
        "active_comp": active_comp,
        "resp_top1": resp_top1,
        "min_mu_dist": min_mu_dist,
        "mean_mu_dist": mean_mu_dist,
        "mean_prior_mu_norm": mean_prior_mu_norm,
        "mean_prior_var": mean_prior_var,
        "max_prior_var": max_prior_var,
        "mean_prior_std": mean_prior_std,
        "mean_factor_var": mean_factor_var,
        "max_factor_var": max_factor_var,
        "mean_factor_norm": mean_factor_norm,
        "mean_total_var": mean_total_var,
        "max_total_var": max_total_var,
        "mean_total_std": mean_total_std,
        "group_k_eff": group_k_eff,
        "group_active": group_active,
        "group_resp_top1": group_resp_top1,
        "sub_k_eff_mean": sub_k_eff_mean,
        "sub_active_mean": sub_active_mean,
        "sub_active_min": sub_active_min,
        "sub_active_max": sub_active_max,
        "group_mu_min_dist": group_mu_min_dist,
        "group_mu_mean_dist": group_mu_mean_dist,
        "sub_delta_norm_mean": sub_delta_norm_mean,
        "sub_delta_norm_max": sub_delta_norm_max,
    }


def batch_kmeans_usage_diagnostics(
    z: torch.Tensor,
    num_clusters: int = 32,
    kmeans_iters: int = 2,
) -> Dict[str, float]:
    with torch.no_grad():
        if z is None or z.size(0) <= 1:
            return {"km_active": 0, "km_eff": 0.0, "km_max": 0.0, "km_min": 0.0}
        zf = F.normalize(z.detach().float(), dim=-1)
        bsz = int(zf.size(0))
        k = max(1, min(int(num_clusters), bsz))
        stride = max(1, bsz // k)
        centers = zf[torch.arange(0, stride * k, stride, device=zf.device)[:k]].clone()
        assign = torch.zeros((bsz,), dtype=torch.long, device=zf.device)
        for _ in range(max(1, int(kmeans_iters))):
            assign = torch.argmin(torch.cdist(zf, centers, p=2), dim=1)
            new_centers = torch.zeros_like(centers)
            counts = torch.bincount(assign, minlength=k).to(zf.dtype)
            new_centers.index_add_(0, assign, zf)
            non_empty = counts > 0
            new_centers[non_empty] = new_centers[non_empty] / counts[non_empty].unsqueeze(1)
            if (~non_empty).any():
                new_centers[~non_empty] = centers[~non_empty]
            centers = F.normalize(new_centers, dim=-1)
        counts = torch.bincount(assign, minlength=k).float()
        usage = counts / max(float(bsz), 1.0)
        nonzero = usage[usage > 0]
        entropy = -(nonzero * torch.log(nonzero + 1e-12)).sum()
        return {
            "km_active": int((counts > 0).sum().item()),
            "km_eff": float(torch.exp(entropy).item()),
            "km_max": float(usage.max().item()),
            "km_min": float(nonzero.min().item()) if nonzero.numel() > 0 else 0.0,
        }


# =========================
# Training / evaluation (for MFA-VAE)
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
    # Fast path for CPU tensors: avoid many tiny torch RNG calls and .item() sync points.
    if enc_mask.device.type == "cpu":
        arr = enc_mask.numpy()
        B = arr.shape[0]
        apply_flags = np.random.rand(B) <= float(apply_prob)
        simple_min = max(0.0, min(1.0, float(min_frac)))
        simple_max = max(simple_min, min(1.0, float(max_frac)))

        for i in range(B):
            if not apply_flags[i]:
                continue
            obs_idx = np.flatnonzero(arr[i] > 0)
            n_obs = int(obs_idx.size)
            if n_obs <= 1:
                continue

            if policy == "xverse":
                if n_obs < 1000:
                    max_to_mask = max(5, int(n_obs * (1.0 / 5.0)))
                    if n_obs > 10:
                        low = 10
                        high = max(low, max_to_mask)
                        n_hide = int(np.random.randint(low, high + 1))
                    else:
                        n_hide = n_obs
                else:
                    p = float(np.random.rand())
                    if p < 0.2:
                        frac = 0.1 + 0.2 * float(np.random.rand())  # 0.1 - 0.3
                    elif p < 0.9:
                        frac = 0.3 + 0.2 * float(np.random.rand())  # 0.3 - 0.5
                    else:
                        frac = 0.5 + 0.25 * float(np.random.rand())  # 0.5 - 0.75
                    n_hide = int(n_obs * frac)
            else:
                frac = simple_min + (simple_max - simple_min) * float(np.random.rand())
                n_hide = int(n_obs * frac)

            n_hide = max(1, min(n_hide, n_obs - 1))
            hide_idx = np.random.choice(obs_idx, size=n_hide, replace=False)
            arr[i, hide_idx] = 0
        return enc_mask

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
                    frac = 0.1 + 0.2 * torch.rand(1, device=enc_mask.device).item()
                elif p < 0.9:
                    frac = 0.3 + 0.2 * torch.rand(1, device=enc_mask.device).item()
                else:
                    frac = 0.5 + 0.25 * torch.rand(1, device=enc_mask.device).item()
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
    beta_u_kl_multiplier=1.0,
    beta_eps_kl_multiplier=1.0,
    recon_observed_only=False,
    mask_aug_prob=1.0,
    mask_aug_policy="xverse",
    mask_aug_min_frac=0.1,
    mask_aug_max_frac=0.5,
    lambda_score=0.0,
    score_noise_std=0.1,
    score_detach_z=True,
    lambda_contrast=0.0,
    contrast_temp=0.1,
    lambda_real_recon=0.0,
    lambda_cov=0.0,
    cov_use_mu=True,
    lambda_resp_balance=0.0,
    lambda_resp_confidence=0.0,
    lambda_resp_anchor=0.0,
    resp_temperature=1.0,
    resp_topk=0,
    prior_logvar_min=-6.0,
    prior_logvar_max=4.0,
    lambda_prior_mu_l2=0.0,
    lambda_prior_factor_l2=0.0,
    lambda_prior_pi_balance=0.0,
    lambda_prior_mu_spread=0.0,
    prior_mu_spread_tau=1.0,
    lambda_post_c_balance=0.0,
    lambda_celltype_cls=0.0,
    lambda_prior_logvar_l2=0.0,
    prior_logvar_target=-2.0,
    recon_gene_weight_mode="none",
    recon_gene_weight_alpha=0.0,
    recon_gene_weight_ema_momentum=0.99,
    recon_gene_weight_min=0.3,
    recon_gene_weight_max=3.0,
    recon_gene_weight_eps=1e-6,
    recon_cell_weight_mode="none",
    recon_cell_weight_alpha=0.0,
    recon_cell_weight_min=0.5,
    recon_cell_weight_max=2.0,
    recon_cell_weight_eps=1e-6,
    recon_cell_weight_clusters=32,
    recon_cell_weight_kmeans_iters=2,
    lambda_batchless_recon=0.0,
    force_base_posterior=False,
):
    model.train()
    loss_fn = model.module.loss if hasattr(model, "module") else model.loss
    prior_ref = model.module.prior if hasattr(model, "module") else model.prior
    total_loss = total_recon = total_kl = total_score = total_contrast = total_cov = total_prior_pi_balance = total_celltype_cls = 0.0
    total_batchless_recon = 0.0
    n_cells = 0
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    for batch_idx, (sample_id, tissue_id, celltype_id, x_count, x_mask, x_mask_encoder) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        sample_id = sample_id.to(device, non_blocking=True)
        tissue_id = tissue_id.to(device, non_blocking=True)
        celltype_id = celltype_id.to(device, non_blocking=True)
        x_count = x_count.to(device, non_blocking=True)
        x_mask = x_mask.to(device, non_blocking=True)
        x_mask_encoder = x_mask_encoder.to(device, non_blocking=True)

        bsz = x_count.size(0)

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            out_fake = loss_fn(
                x_count=x_count,
                x_mask=x_mask,
                tissue_id=tissue_id,
                sample_id=sample_id,
                celltype_id=celltype_id,
                force_base_posterior=force_base_posterior,
                beta=beta_kl,
                beta_u_kl_multiplier=beta_u_kl_multiplier,
                beta_eps_kl_multiplier=beta_eps_kl_multiplier,
                encoder_mask=x_mask_encoder,
                recon_mask=x_mask if recon_observed_only else None,
                lambda_score=lambda_score,
                score_noise_std=score_noise_std,
                score_detach_z=score_detach_z,
                lambda_cov=lambda_cov,
                cov_use_mu=cov_use_mu,
                lambda_resp_balance=lambda_resp_balance,
                lambda_resp_confidence=lambda_resp_confidence,
                lambda_resp_anchor=lambda_resp_anchor,
                resp_temperature=resp_temperature,
                resp_topk=resp_topk,
                prior_logvar_min=prior_logvar_min,
                prior_logvar_max=prior_logvar_max,
                lambda_prior_mu_l2=lambda_prior_mu_l2,
                lambda_prior_factor_l2=lambda_prior_factor_l2,
                lambda_prior_pi_balance=lambda_prior_pi_balance,
                lambda_prior_mu_spread=lambda_prior_mu_spread,
                prior_mu_spread_tau=prior_mu_spread_tau,
                lambda_post_c_balance=lambda_post_c_balance,
                lambda_celltype_cls=lambda_celltype_cls,
                lambda_prior_logvar_l2=lambda_prior_logvar_l2,
                prior_logvar_target=prior_logvar_target,
                recon_gene_weight_mode=recon_gene_weight_mode,
                recon_gene_weight_alpha=recon_gene_weight_alpha,
                recon_gene_weight_ema_momentum=recon_gene_weight_ema_momentum,
                recon_gene_weight_min=recon_gene_weight_min,
                recon_gene_weight_max=recon_gene_weight_max,
                recon_gene_weight_eps=recon_gene_weight_eps,
                recon_cell_weight_mode=recon_cell_weight_mode,
                recon_cell_weight_alpha=recon_cell_weight_alpha,
                recon_cell_weight_min=recon_cell_weight_min,
                recon_cell_weight_max=recon_cell_weight_max,
                recon_cell_weight_eps=recon_cell_weight_eps,
                recon_cell_weight_clusters=recon_cell_weight_clusters,
                recon_cell_weight_kmeans_iters=recon_cell_weight_kmeans_iters,
            )
            need_real_view = (lambda_contrast > 0) or (lambda_real_recon > 0)
            real_recon = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)
            if need_real_view:
                out_real = loss_fn(
                    x_count=x_count,
                    x_mask=x_mask,
                    tissue_id=tissue_id,
                    sample_id=sample_id,
                    celltype_id=celltype_id,
                    force_base_posterior=force_base_posterior,
                    beta=0.0,
                    beta_u_kl_multiplier=beta_u_kl_multiplier,
                    beta_eps_kl_multiplier=beta_eps_kl_multiplier,
                    encoder_mask=x_mask,
                    recon_mask=x_mask if recon_observed_only else None,
                    lambda_score=0.0,
                    score_noise_std=score_noise_std,
                    score_detach_z=score_detach_z,
                    lambda_cov=0.0,
                    cov_use_mu=cov_use_mu,
                    lambda_resp_balance=0.0,
                    lambda_resp_confidence=0.0,
                    lambda_resp_anchor=0.0,
                    resp_temperature=resp_temperature,
                    resp_topk=0,
                    prior_logvar_min=prior_logvar_min,
                    prior_logvar_max=prior_logvar_max,
                    lambda_prior_mu_l2=0.0,
                    lambda_prior_factor_l2=0.0,
                    lambda_prior_pi_balance=0.0,
                    lambda_prior_mu_spread=0.0,
                    prior_mu_spread_tau=prior_mu_spread_tau,
                    lambda_post_c_balance=0.0,
                    lambda_celltype_cls=0.0,
                    lambda_prior_logvar_l2=0.0,
                    prior_logvar_target=prior_logvar_target,
                    recon_gene_weight_mode=recon_gene_weight_mode,
                    recon_gene_weight_alpha=recon_gene_weight_alpha,
                    recon_gene_weight_ema_momentum=recon_gene_weight_ema_momentum,
                    recon_gene_weight_min=recon_gene_weight_min,
                    recon_gene_weight_max=recon_gene_weight_max,
                    recon_gene_weight_eps=recon_gene_weight_eps,
                    recon_cell_weight_mode=recon_cell_weight_mode,
                    recon_cell_weight_alpha=recon_cell_weight_alpha,
                    recon_cell_weight_min=recon_cell_weight_min,
                    recon_cell_weight_max=recon_cell_weight_max,
                    recon_cell_weight_eps=recon_cell_weight_eps,
                    recon_cell_weight_clusters=recon_cell_weight_clusters,
                    recon_cell_weight_kmeans_iters=recon_cell_weight_kmeans_iters,
                )
                real_recon = out_real["recon_loss"]
            batchless_recon = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)
            if float(lambda_batchless_recon) != 0.0:
                out_batchless = loss_fn(
                    x_count=x_count,
                    x_mask=x_mask,
                    tissue_id=tissue_id,
                    sample_id=sample_id,
                    celltype_id=celltype_id,
                    force_base_posterior=force_base_posterior,
                    beta=0.0,
                    beta_u_kl_multiplier=beta_u_kl_multiplier,
                    beta_eps_kl_multiplier=beta_eps_kl_multiplier,
                    encoder_mask=x_mask_encoder,
                    use_batch_condition=False,
                    recon_mask=x_mask if recon_observed_only else None,
                    lambda_score=0.0,
                    score_noise_std=score_noise_std,
                    score_detach_z=score_detach_z,
                    lambda_cov=0.0,
                    cov_use_mu=cov_use_mu,
                    lambda_resp_balance=0.0,
                    lambda_resp_confidence=0.0,
                    lambda_resp_anchor=0.0,
                    resp_temperature=resp_temperature,
                    resp_topk=0,
                    prior_logvar_min=prior_logvar_min,
                    prior_logvar_max=prior_logvar_max,
                    lambda_prior_mu_l2=0.0,
                    lambda_prior_factor_l2=0.0,
                    lambda_prior_pi_balance=0.0,
                    lambda_prior_mu_spread=0.0,
                    prior_mu_spread_tau=prior_mu_spread_tau,
                    lambda_post_c_balance=0.0,
                    lambda_celltype_cls=0.0,
                    lambda_prior_logvar_l2=0.0,
                    prior_logvar_target=prior_logvar_target,
                    recon_gene_weight_mode=recon_gene_weight_mode,
                    recon_gene_weight_alpha=recon_gene_weight_alpha,
                    recon_gene_weight_ema_momentum=recon_gene_weight_ema_momentum,
                    recon_gene_weight_min=recon_gene_weight_min,
                    recon_gene_weight_max=recon_gene_weight_max,
                    recon_gene_weight_eps=recon_gene_weight_eps,
                    recon_cell_weight_mode=recon_cell_weight_mode,
                    recon_cell_weight_alpha=recon_cell_weight_alpha,
                    recon_cell_weight_min=recon_cell_weight_min,
                    recon_cell_weight_max=recon_cell_weight_max,
                    recon_cell_weight_eps=recon_cell_weight_eps,
                    recon_cell_weight_clusters=recon_cell_weight_clusters,
                    recon_cell_weight_kmeans_iters=recon_cell_weight_kmeans_iters,
                )
                batchless_recon = out_batchless["recon_loss"]
            if lambda_contrast > 0:
                contrast = bidirectional_contrastive_loss(
                    z_real=deterministic_contrast_embedding(out_real),
                    z_fake=deterministic_contrast_embedding(out_fake),
                    temperature=contrast_temp,
                )
            else:
                contrast = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)

            loss = out_fake["loss"]
            if float(lambda_contrast) != 0.0:
                loss = loss + float(lambda_contrast) * contrast
            if float(lambda_real_recon) != 0.0:
                loss = loss + float(lambda_real_recon) * real_recon
            if float(lambda_batchless_recon) != 0.0:
                loss = loss + float(lambda_batchless_recon) * batchless_recon
            recon = out_fake["recon_loss"]
            kl = out_fake["kl_loss"]
            score = out_fake["score_loss"]
            cov = out_fake["cov_loss"]
            prior_pi_balance = out_fake.get("prior_pi_balance_loss", torch.zeros_like(cov))
            post_c_balance = out_fake.get("post_c_balance_loss", torch.zeros_like(cov))
            celltype_cls = out_fake.get("celltype_cls_loss", torch.zeros_like(cov))

        if not torch.isfinite(loss):
            if is_rank0:
                kl_c_dbg = out_fake.get("kl_c_loss", torch.zeros_like(kl))
                kl_u_dbg = out_fake.get("kl_u_loss", torch.zeros_like(kl))
                kl_eps_dbg = out_fake.get("kl_eps_loss", torch.zeros_like(kl))
                prior_factor_l2_dbg = out_fake.get("prior_factor_l2_loss", torch.zeros_like(kl))
                prior_logvar_l2_dbg = out_fake.get("prior_logvar_l2_loss", torch.zeros_like(kl))
                prior_mu_spread_dbg = out_fake.get("prior_mu_spread_loss", torch.zeros_like(kl))
                post_c_balance_dbg = out_fake.get("post_c_balance_loss", torch.zeros_like(kl))
                rate = out_fake.get("rate", torch.empty(0, device=x_count.device))
                theta = out_fake.get("nb_theta", torch.empty(0, device=x_count.device))
                z_cur = out_fake.get("z", torch.empty(0, device=x_count.device))
                term_checks = {
                    "loss": loss,
                    "fake_loss": out_fake.get("loss", torch.zeros_like(kl)),
                    "recon": recon,
                    "kl": kl,
                    "real_recon": real_recon,
                    "batchless_recon": batchless_recon,
                    "contrast": contrast,
                    "cls": celltype_cls,
                    "prior_factor_l2": prior_factor_l2_dbg,
                    "prior_logvar_l2": prior_logvar_l2_dbg,
                    "prior_mu_spread": prior_mu_spread_dbg,
                    "post_c_balance": post_c_balance_dbg,
                }
                bad_terms = [
                    name for name, val in term_checks.items()
                    if torch.is_tensor(val) and val.numel() > 0 and (not torch.isfinite(val.detach()).all())
                ]
                diag_msg = ""
                if rate.numel() > 0:
                    rate_f = rate.detach().float()
                    diag_msg += f", rate[min/max]={rate_f.nan_to_num().min().item():.3g}/{rate_f.nan_to_num().max().item():.3g}"
                if theta.numel() > 0:
                    theta_f = theta.detach().float()
                    diag_msg += f", theta[min/max]={theta_f.nan_to_num().min().item():.3g}/{theta_f.nan_to_num().max().item():.3g}"
                if z_cur.numel() > 0:
                    z_f = z_cur.detach().float()
                    diag_msg += f", zAbsMax={z_f.nan_to_num().abs().max().item():.3g}"
                if getattr(model.module if hasattr(model, "module") else model, "prior_type", None) == "gmm":
                    diag = gmm_collapse_diagnostics(prior=prior_ref, z=z_cur.detach() if z_cur.numel() > 0 else None, tissue_id=tissue_id)
                    diag_msg += (
                        f", K_eff={diag['k_eff']:.2f}, maxVar={diag['max_prior_var']:.3g}, "
                        f"maxFactorVar={diag['max_factor_var']:.3g}, maxTotalVar={diag['max_total_var']:.3g}"
                    )
                print(
                    f"[WARN] Non-finite train loss at batch {batch_idx + 1}, skip step. "
                    f"Recon={recon.item():.4f}, KL={kl.item():.4f}, "
                    f"KLc={kl_c_dbg.item():.4f}, KLu={kl_u_dbg.item():.4f}, KLeps={kl_eps_dbg.item():.4f}, "
                    f"Contrast={contrast.item():.4f}, "
                    f"cls={celltype_cls.item():.4f}, RealRecon={real_recon.item():.4f}, "
                    f"BatchlessRecon={batchless_recon.item():.4f}, "
                    f"priorFactorL2={prior_factor_l2_dbg.item():.4g}, "
                    f"priorLogvarL2={prior_logvar_l2_dbg.item():.4g}, "
                    f"priorMuSpread={prior_mu_spread_dbg.item():.4g}, "
                    f"postCBal={post_c_balance_dbg.item():.4g}, "
                    f"badTerms={bad_terms}{diag_msg}"
                )
            optimizer.zero_grad(set_to_none=True)
            continue

        n_cells += bsz

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * bsz
        total_recon += recon.item() * bsz
        total_kl += kl.item() * bsz
        total_score += score.item() * bsz
        total_contrast += contrast.item() * bsz
        total_cov += cov.item() * bsz
        total_prior_pi_balance += prior_pi_balance.item() * bsz
        total_celltype_cls += celltype_cls.item() * bsz
        total_batchless_recon += batchless_recon.item() * bsz

        if (batch_idx + 1) % 1000 == 0 and is_rank0:
            msg = (
                f"[Batch {batch_idx + 1}] "
                f"Loss={loss.item():.4f}, Recon={recon.item():.4f}, KL={kl.item():.4f}, "
                f"KLc={out_fake.get('kl_c_loss', torch.zeros_like(kl)).item():.4f}, "
                f"KLu={out_fake.get('kl_u_loss', torch.zeros_like(kl)).item():.4f}, "
                f"KLeps={out_fake.get('kl_eps_loss', torch.zeros_like(kl)).item():.4f}, "
                f"cls={celltype_cls.item():.4f}"
            )
            if lambda_contrast > 0:
                msg += f", Contrast={contrast.item():.4f}"
            if lambda_real_recon > 0:
                msg += f", RealRecon={real_recon.item():.4f}"
            if lambda_batchless_recon > 0:
                msg += f", BatchlessRecon={batchless_recon.item():.4f}"
            if lambda_post_c_balance > 0:
                msg += f", postCBal={post_c_balance.item():.4f}"
            if str(recon_cell_weight_mode).lower() == "batch_kmeans" and float(recon_cell_weight_alpha) > 0:
                km = batch_kmeans_usage_diagnostics(
                    z=out_fake["z"],
                    num_clusters=recon_cell_weight_clusters,
                    kmeans_iters=recon_cell_weight_kmeans_iters,
                )
                msg += (
                    f", kmKeff={km['km_eff']:.2f}, kmActive={km['km_active']}, "
                    f"kmMax={km['km_max']:.3f}, kmMin={km['km_min']:.3f}"
                )
            msg += (
                f", respH={out_fake['resp_entropy'].item():.4f}"
            )
            if getattr(model.module if hasattr(model, "module") else model, "prior_type", None) == "gmm":
                diag = gmm_collapse_diagnostics(prior=prior_ref, z=out_fake["z"], tissue_id=tissue_id)
                msg += (
                    f", K_eff={diag['k_eff']:.2f}, activeK={diag['active_comp']}, "
                    f"respTop1={diag['resp_top1']:.3f}, "
                    f"groupKeff={diag['group_k_eff']:.2f}, groupActive={diag['group_active']}, "
                    f"groupTop1={diag['group_resp_top1']:.3f}, subKeffMean={diag['sub_k_eff_mean']:.2f}, "
                    f"subActiveMean={diag['sub_active_mean']:.2f}, subActiveMinMax={diag['sub_active_min']:.0f}/{diag['sub_active_max']:.0f}, "
                    f"minMuDist={diag['min_mu_dist']:.3f}, meanMuDist={diag['mean_mu_dist']:.3f}, "
                    f"groupMuMinDist={diag['group_mu_min_dist']:.3f}, groupMuMeanDist={diag['group_mu_mean_dist']:.3f}, "
                    f"subDeltaNorm={diag['sub_delta_norm_mean']:.3f}/{diag['sub_delta_norm_max']:.3f}, "
                    f"meanPriorMuNorm={diag['mean_prior_mu_norm']:.3f}, meanVar={diag['mean_prior_var']:.3f}, "
                    f"maxVar={diag['max_prior_var']:.3f}, meanStd={diag['mean_prior_std']:.3f}, "
                    f"factorVar={diag['mean_factor_var']:.3f}, maxFactorVar={diag['max_factor_var']:.3f}, "
                    f"factorNorm={diag['mean_factor_norm']:.3f}, totalVar={diag['mean_total_var']:.3f}, "
                    f"maxTotalVar={diag['max_total_var']:.3f}, totalStd={diag['mean_total_std']:.3f}"
                )
            print(msg)

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, total_prior_pi_balance, total_celltype_cls, total_batchless_recon, float(n_cells)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, total_prior_pi_balance, total_celltype_cls, total_batchless_recon, n_cells = stats.tolist()
        n_cells = max(float(n_cells), 1.0)

    return (
        total_loss / n_cells,
        total_recon / n_cells,
        total_kl / n_cells,
        total_score / n_cells,
        total_contrast / n_cells,
        total_cov / n_cells,
        total_prior_pi_balance / n_cells,
        total_celltype_cls / n_cells,
        total_batchless_recon / n_cells,
    )


def evaluate_gmm_vae_one_epoch(
    model,
    val_loader,
    device,
    beta_kl=1.0,
    beta_u_kl_multiplier=1.0,
    beta_eps_kl_multiplier=1.0,
    recon_observed_only=False,
    lambda_score=0.0,
    score_noise_std=0.1,
    score_detach_z=True,
    lambda_contrast=0.0,
    contrast_temp=0.1,
    lambda_real_recon=0.0,
    lambda_cov=0.0,
    cov_use_mu=True,
    lambda_resp_balance=0.0,
    lambda_resp_confidence=0.0,
    lambda_resp_anchor=0.0,
    resp_temperature=1.0,
    resp_topk=0,
    prior_logvar_min=-6.0,
    prior_logvar_max=4.0,
    lambda_prior_mu_l2=0.0,
    lambda_prior_factor_l2=0.0,
    lambda_prior_pi_balance=0.0,
    lambda_prior_mu_spread=0.0,
    prior_mu_spread_tau=1.0,
    lambda_post_c_balance=0.0,
    lambda_celltype_cls=0.0,
    lambda_prior_logvar_l2=0.0,
    prior_logvar_target=-2.0,
    mask_aug_prob=1.0,
    mask_aug_policy="xverse",
    mask_aug_min_frac=0.1,
    mask_aug_max_frac=0.5,
    recon_gene_weight_mode="none",
    recon_gene_weight_alpha=0.0,
    recon_gene_weight_ema_momentum=0.99,
    recon_gene_weight_min=0.3,
    recon_gene_weight_max=3.0,
    recon_gene_weight_eps=1e-6,
    recon_cell_weight_mode="none",
    recon_cell_weight_alpha=0.0,
    recon_cell_weight_min=0.5,
    recon_cell_weight_max=2.0,
    recon_cell_weight_eps=1e-6,
    recon_cell_weight_clusters=32,
    recon_cell_weight_kmeans_iters=2,
    lambda_batchless_recon=0.0,
    force_base_posterior=False,
):
    model.eval()
    loss_fn = model.module.loss if hasattr(model, "module") else model.loss
    prior_ref = model.module.prior if hasattr(model, "module") else model.prior
    total_loss = total_recon = total_kl = total_score = total_contrast = total_cov = total_prior_pi_balance = total_celltype_cls = 0.0
    total_batchless_recon = 0.0
    n_cells = 0
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    with torch.no_grad():
        for batch_idx, (sample_id, tissue_id, celltype_id, x_count, x_mask, x_mask_encoder) in enumerate(val_loader):
            sample_id = sample_id.to(device, non_blocking=True)
            tissue_id = tissue_id.to(device, non_blocking=True)
            celltype_id = celltype_id.to(device, non_blocking=True)
            x_count = x_count.to(device, non_blocking=True)
            x_mask = x_mask.to(device, non_blocking=True)
            x_mask_encoder = x_mask_encoder.to(device, non_blocking=True)
            bsz = x_count.size(0)
            n_cells += bsz

            out_real = loss_fn(
                # Validation main metric uses the original observed mask, not random-mask augmentation.
                x_count=x_count,
                x_mask=x_mask,
                tissue_id=tissue_id,
                sample_id=sample_id,
                celltype_id=celltype_id,
                force_base_posterior=force_base_posterior,
                beta=beta_kl,
                beta_u_kl_multiplier=beta_u_kl_multiplier,
                beta_eps_kl_multiplier=beta_eps_kl_multiplier,
                use_batch_condition=False,
                encoder_mask=x_mask,
                recon_mask=x_mask if recon_observed_only else None,
                lambda_score=lambda_score,
                score_noise_std=score_noise_std,
                score_detach_z=score_detach_z,
                lambda_cov=lambda_cov,
                cov_use_mu=cov_use_mu,
                lambda_resp_balance=lambda_resp_balance,
                lambda_resp_confidence=lambda_resp_confidence,
                lambda_resp_anchor=lambda_resp_anchor,
                resp_temperature=resp_temperature,
                resp_topk=resp_topk,
                prior_logvar_min=prior_logvar_min,
                prior_logvar_max=prior_logvar_max,
                lambda_prior_mu_l2=lambda_prior_mu_l2,
                lambda_prior_factor_l2=lambda_prior_factor_l2,
                lambda_prior_pi_balance=lambda_prior_pi_balance,
                lambda_prior_mu_spread=lambda_prior_mu_spread,
                prior_mu_spread_tau=prior_mu_spread_tau,
                lambda_post_c_balance=lambda_post_c_balance,
                lambda_celltype_cls=lambda_celltype_cls,
                lambda_prior_logvar_l2=lambda_prior_logvar_l2,
                prior_logvar_target=prior_logvar_target,
                recon_gene_weight_mode=recon_gene_weight_mode,
                recon_gene_weight_alpha=recon_gene_weight_alpha,
                recon_gene_weight_ema_momentum=recon_gene_weight_ema_momentum,
                recon_gene_weight_min=recon_gene_weight_min,
                recon_gene_weight_max=recon_gene_weight_max,
                recon_gene_weight_eps=recon_gene_weight_eps,
                recon_cell_weight_mode=recon_cell_weight_mode,
                recon_cell_weight_alpha=recon_cell_weight_alpha,
                recon_cell_weight_min=recon_cell_weight_min,
                recon_cell_weight_max=recon_cell_weight_max,
                recon_cell_weight_eps=recon_cell_weight_eps,
                recon_cell_weight_clusters=recon_cell_weight_clusters,
                recon_cell_weight_kmeans_iters=recon_cell_weight_kmeans_iters,
            )
            out_fake = out_real
            real_recon = out_real["recon_loss"]
            if lambda_contrast > 0:
                fake_encoder_mask = _random_hide_observed(
                    x_mask=x_mask,
                    apply_prob=mask_aug_prob,
                    policy=mask_aug_policy,
                    min_frac=mask_aug_min_frac,
                    max_frac=mask_aug_max_frac,
                )
                out_fake = loss_fn(
                    x_count=x_count,
                    x_mask=x_mask,
                    tissue_id=tissue_id,
                    sample_id=sample_id,
                    celltype_id=celltype_id,
                    force_base_posterior=force_base_posterior,
                    beta=0.0,
                    use_batch_condition=False,
                    encoder_mask=fake_encoder_mask,
                    recon_mask=x_mask if recon_observed_only else None,
                    lambda_score=0.0,
                    score_noise_std=score_noise_std,
                    score_detach_z=score_detach_z,
                    lambda_cov=0.0,
                    cov_use_mu=cov_use_mu,
                    lambda_resp_balance=0.0,
                    lambda_resp_confidence=0.0,
                    lambda_resp_anchor=0.0,
                    resp_temperature=resp_temperature,
                    resp_topk=0,
                    prior_logvar_min=prior_logvar_min,
                    prior_logvar_max=prior_logvar_max,
                    lambda_prior_mu_l2=0.0,
                    lambda_prior_factor_l2=0.0,
                    lambda_prior_pi_balance=0.0,
                    lambda_prior_mu_spread=0.0,
                    prior_mu_spread_tau=prior_mu_spread_tau,
                    lambda_post_c_balance=0.0,
                    lambda_celltype_cls=0.0,
                    lambda_prior_logvar_l2=0.0,
                    prior_logvar_target=prior_logvar_target,
                    recon_gene_weight_mode=recon_gene_weight_mode,
                    recon_gene_weight_alpha=recon_gene_weight_alpha,
                    recon_gene_weight_ema_momentum=recon_gene_weight_ema_momentum,
                    recon_gene_weight_min=recon_gene_weight_min,
                    recon_gene_weight_max=recon_gene_weight_max,
                    recon_gene_weight_eps=recon_gene_weight_eps,
                    recon_cell_weight_mode=recon_cell_weight_mode,
                    recon_cell_weight_alpha=recon_cell_weight_alpha,
                    recon_cell_weight_min=recon_cell_weight_min,
                    recon_cell_weight_max=recon_cell_weight_max,
                    recon_cell_weight_eps=recon_cell_weight_eps,
                    recon_cell_weight_clusters=recon_cell_weight_clusters,
                    recon_cell_weight_kmeans_iters=recon_cell_weight_kmeans_iters,
                )
            batchless_recon = out_real["recon_loss"]
            if float(lambda_batchless_recon) != 0.0:
                # Validation main path is already batchless. Reuse it to avoid an extra
                # stochastic forward pass that would make Recon and BatchlessRecon differ.
                batchless_recon = out_real["recon_loss"]
            if lambda_contrast > 0:
                contrast = bidirectional_contrastive_loss(
                    z_real=deterministic_contrast_embedding(out_real),
                    z_fake=deterministic_contrast_embedding(out_fake),
                    temperature=contrast_temp,
                )
            else:
                contrast = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)
            loss = out_real["loss"]
            if float(lambda_contrast) != 0.0:
                loss = loss + float(lambda_contrast) * contrast
            if float(lambda_real_recon) != 0.0:
                loss = loss + float(lambda_real_recon) * real_recon
            if float(lambda_batchless_recon) != 0.0:
                loss = loss + float(lambda_batchless_recon) * batchless_recon

            total_loss += loss.item() * bsz
            total_recon += out_real["recon_loss"].item() * bsz
            total_kl += out_real["kl_loss"].item() * bsz
            total_score += out_real["score_loss"].item() * bsz
            total_contrast += contrast.item() * bsz
            total_cov += out_real["cov_loss"].item() * bsz
            total_prior_pi_balance += out_real.get("prior_pi_balance_loss", torch.zeros_like(out_real["cov_loss"])).item() * bsz
            total_celltype_cls += out_real.get("celltype_cls_loss", torch.zeros_like(out_real["cov_loss"])).item() * bsz
            total_batchless_recon += batchless_recon.item() * bsz

            if (batch_idx + 1) % 1000 == 0 and is_rank0:
                msg = (
                    f"[Val Batch {batch_idx + 1}] "
                    f"Loss={loss.item():.4f}, Recon={out_real['recon_loss'].item():.4f}, "
                    f"KL={out_real['kl_loss'].item():.4f}, "
                    f"KLc={out_real.get('kl_c_loss', torch.zeros_like(out_real['kl_loss'])).item():.4f}, "
                    f"KLu={out_real.get('kl_u_loss', torch.zeros_like(out_real['kl_loss'])).item():.4f}, "
                    f"KLeps={out_real.get('kl_eps_loss', torch.zeros_like(out_real['kl_loss'])).item():.4f}, "
                    f"cls={out_real.get('celltype_cls_loss', torch.zeros_like(out_real['cov_loss'])).item():.4f}"
                )
                if lambda_batchless_recon > 0:
                    msg += f", BatchlessRecon={batchless_recon.item():.4f}"
                if getattr(model.module if hasattr(model, "module") else model, "prior_type", None) == "gmm":
                    diag = gmm_collapse_diagnostics(prior=prior_ref, z=out_real["z"], tissue_id=tissue_id)
                    msg += (
                        f", K_eff={diag['k_eff']:.2f}, activeK={diag['active_comp']}, "
                        f"respTop1={diag['resp_top1']:.3f}, "
                        f"groupKeff={diag['group_k_eff']:.2f}, groupActive={diag['group_active']}, "
                        f"groupTop1={diag['group_resp_top1']:.3f}, subKeffMean={diag['sub_k_eff_mean']:.2f}, "
                        f"subActiveMean={diag['sub_active_mean']:.2f}, subActiveMinMax={diag['sub_active_min']:.0f}/{diag['sub_active_max']:.0f}, "
                        f"minMuDist={diag['min_mu_dist']:.3f}, meanMuDist={diag['mean_mu_dist']:.3f}, "
                        f"groupMuMinDist={diag['group_mu_min_dist']:.3f}, groupMuMeanDist={diag['group_mu_mean_dist']:.3f}, "
                        f"subDeltaNorm={diag['sub_delta_norm_mean']:.3f}/{diag['sub_delta_norm_max']:.3f}, "
                        f"meanPriorMuNorm={diag['mean_prior_mu_norm']:.3f}, meanVar={diag['mean_prior_var']:.3f}, "
                        f"maxVar={diag['max_prior_var']:.3f}, meanStd={diag['mean_prior_std']:.3f}, "
                        f"factorVar={diag['mean_factor_var']:.3f}, maxFactorVar={diag['max_factor_var']:.3f}, "
                        f"factorNorm={diag['mean_factor_norm']:.3f}, totalVar={diag['mean_total_var']:.3f}, "
                        f"maxTotalVar={diag['max_total_var']:.3f}, totalStd={diag['mean_total_std']:.3f}"
                    )
                print(msg)

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, total_prior_pi_balance, total_celltype_cls, total_batchless_recon, float(n_cells)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, total_prior_pi_balance, total_celltype_cls, total_batchless_recon, n_cells = stats.tolist()
        n_cells = max(float(n_cells), 1.0)

    return (
        total_loss / n_cells,
        total_recon / n_cells,
        total_kl / n_cells,
        total_score / n_cells,
        total_contrast / n_cells,
        total_cov / n_cells,
        total_prior_pi_balance / n_cells,
        total_celltype_cls / n_cells,
        total_batchless_recon / n_cells,
    )


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
        max_cached_blocks=8,
        filter_bad_cells=False,
        index_cache_path=None,
        allow_stale_index_cache=False,
    ):
        self.gene_ids = gene_ids
        self.pair_to_sample_id = pair_to_sample_id
        self.pair_to_tissue_id = pair_to_tissue_id
        self.cell_type_to_index = cell_type_to_index
        self.use_cache = use_cache
        self.available_genes = set(available_genes) if available_genes is not None else None
        self.max_cached_blocks = max(1, int(max_cached_blocks))
        self.filter_bad_cells = bool(filter_bad_cells)
        self.index_cache_path = index_cache_path
        self.allow_stale_index_cache = bool(allow_stale_index_cache)

        self.index_map = []
        self.block_observed_global_idx = []
        self.block_local_to_rel = []
        self.block_matrix_paths = []
        self.block_meta_paths = []
        self.block_nrows = []
        self.cache = {} if use_cache else None
        self._block_cache = OrderedDict()

        self._cache_signature = self._build_cache_signature(matrix_meta_pairs)
        if self._try_load_index_cache():
            return

        for block_idx, (matrix_path, meta_path) in enumerate(matrix_meta_pairs):
            meta = np.load(meta_path, allow_pickle=True)
            gene_ids_available = meta["gene_ids"]
            gene_to_idx = {g: i for i, g in enumerate(gene_ids_available)}
            gene_idx_array = np.full(len(self.gene_ids), -1, dtype=np.int32)
            for global_idx, gene in enumerate(self.gene_ids):
                if gene in gene_to_idx:
                    if self.available_genes is None or gene in self.available_genes:
                        gene_idx_array[global_idx] = gene_to_idx[gene]

            observed_global_idx = np.where(gene_idx_array >= 0)[0].astype(np.int32, copy=False)
            observed_local_idx = gene_idx_array[observed_global_idx].astype(np.int32, copy=False)
            n_rows, n_cols = self._read_sparse_npz_shape(matrix_path)
            local_to_rel = np.full(n_cols, -1, dtype=np.int32)
            if observed_local_idx.size > 0:
                local_to_rel[observed_local_idx] = np.arange(observed_local_idx.size, dtype=np.int32)

            self.block_observed_global_idx.append(observed_global_idx)
            self.block_local_to_rel.append(local_to_rel)
            self.block_matrix_paths.append(matrix_path)
            self.block_meta_paths.append(meta_path)
            self.block_nrows.append(n_rows)

            pair = (matrix_path, meta_path)
            sample_id = int(self.pair_to_sample_id.get(pair, -1))
            assert sample_id >= 0, f"Sample ID not found for {pair}!"
            tissue_id = int(self.pair_to_tissue_id.get(pair, -1))

            if self.filter_bad_cells:
                X = sp.load_npz(matrix_path).tocsr()
                row_sum = np.asarray(X.sum(axis=1)).ravel()
                row_max = np.asarray(X.max(axis=1).toarray()).ravel()
                valid_rows = np.where((row_max <= 1000.0) & (row_sum <= 200000.0))[0]
            else:
                valid_rows = np.arange(n_rows, dtype=np.int64)

            self.index_map.extend((block_idx, int(i), sample_id, tissue_id) for i in valid_rows)

    def _build_cache_signature(self, matrix_meta_pairs):
        h = hashlib.sha1()
        h.update(str(len(self.gene_ids)).encode("utf-8"))
        h.update(str(self.filter_bad_cells).encode("utf-8"))
        if self.available_genes is None:
            h.update(b"available_genes:none")
        else:
            for g in sorted(self.available_genes):
                h.update(str(g).encode("utf-8"))
                h.update(b"\n")
        for matrix_path, meta_path in matrix_meta_pairs:
            h.update(str(matrix_path).encode("utf-8"))
            h.update(b"|")
            h.update(str(meta_path).encode("utf-8"))
            h.update(b"|")
            pair = (matrix_path, meta_path)
            h.update(str(self.pair_to_sample_id.get(pair, -1)).encode("utf-8"))
            h.update(b"|")
            h.update(str(self.pair_to_tissue_id.get(pair, -1) if self.pair_to_tissue_id is not None else -1).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _try_load_index_cache(self):
        if not self.index_cache_path:
            return False
        if not os.path.exists(self.index_cache_path):
            return False
        try:
            with np.load(self.index_cache_path, allow_pickle=True) as npz:
                sig = str(npz["signature"].item())
                if sig != self._cache_signature:
                    if not self.allow_stale_index_cache:
                        print(f"[Dataset] Cache signature mismatch: {self.index_cache_path}. Rebuilding from source.")
                        return False
                    print(
                        f"[Dataset][WARN] Cache signature mismatch but --allow-stale-index-cache is enabled. "
                        f"Using existing cache: {self.index_cache_path}"
                    )
                self.index_map = np.asarray(npz["index_map"], dtype=np.int64)
                self.block_matrix_paths = [str(x) for x in npz["block_matrix_paths"].tolist()]
                self.block_meta_paths = [str(x) for x in npz["block_meta_paths"].tolist()]
                self.block_nrows = [int(x) for x in npz["block_nrows"].tolist()]
                self.block_observed_global_idx = [arr.astype(np.int32, copy=False) for arr in npz["block_observed_global_idx"].tolist()]
                self.block_local_to_rel = [arr.astype(np.int32, copy=False) for arr in npz["block_local_to_rel"].tolist()]
            print(f"[Dataset] Loaded index cache: {self.index_cache_path}")
            return True
        except Exception as e:
            print(f"[Dataset] Failed to load cache ({self.index_cache_path}): {e}. Rebuilding from source.")
            return False

    @staticmethod
    def _read_sparse_npz_shape(matrix_path: str):
        with np.load(matrix_path, allow_pickle=False) as npz:
            shape = npz["shape"]
            return int(shape[0]), int(shape[1])


    @staticmethod
    def _compress_row_data_dtype(arr: np.ndarray):
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.uint32, copy=False)
        rounded = np.rint(arr)
        if np.allclose(arr, rounded):
            return rounded.astype(np.uint32, copy=False)
        return arr.astype(np.float32, copy=False)

    def _load_block(self, block_idx: int):
        matrix_path = self.block_matrix_paths[block_idx]
        meta_path = self.block_meta_paths[block_idx]
        X = sp.load_npz(matrix_path).tocsr()
        meta = np.load(meta_path, allow_pickle=True)
        block = {
            "indptr": X.indptr.astype(np.int64, copy=False),
            "indices": X.indices.astype(np.int32, copy=False),
            "data": self._compress_row_data_dtype(X.data),
            "cell_types": np.asarray(meta["cell_type_ontology_term_id"]),
        }
        return block

    def _get_block(self, block_idx: int):
        block = self._block_cache.get(block_idx)
        if block is not None:
            self._block_cache.move_to_end(block_idx)
            return block

        block = self._load_block(block_idx)
        self._block_cache[block_idx] = block
        if len(self._block_cache) > self.max_cached_blocks:
            self._block_cache.popitem(last=False)
        return block

    def __len__(self):
        return len(self.index_map)

    def infer_num_celltypes(self) -> int:
        vals = [int(v) for v in self.cell_type_to_index.values() if int(v) >= 0]
        if not vals:
            return 0
        return int(max(vals)) + 1

    def infer_num_tissues(self) -> int:
        if len(self.index_map) == 0:
            return 0
        tids = self.index_map[:, 3] if isinstance(self.index_map, np.ndarray) else [r[3] for r in self.index_map]
        tids = np.asarray(tids, dtype=np.int64)
        tids = tids[tids >= 0]
        if tids.size == 0:
            return 0
        return int(tids.max()) + 1

    def infer_num_samples(self) -> int:
        if len(self.index_map) == 0:
            return 0
        sids = self.index_map[:, 2] if isinstance(self.index_map, np.ndarray) else [r[2] for r in self.index_map]
        sids = np.asarray(sids, dtype=np.int64)
        sids = sids[sids >= 0]
        if sids.size == 0:
            return 0
        return int(sids.max()) + 1

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        block_idx, local_idx, sample_id_int, tissue_id_int = self.index_map[idx]
        block = self._get_block(block_idx)
        cell_type_str = block["cell_types"][local_idx]
        celltype_index = self.cell_type_to_index.get(cell_type_str, -1)
        indptr = block["indptr"]
        indices = block["indices"]
        data = block["data"]
        local_to_rel = self.block_local_to_rel[block_idx]
        st = indptr[local_idx]
        ed = indptr[local_idx + 1]
        if st < ed:
            row_cols = indices[st:ed]
            row_vals = data[st:ed]
            rel_idx = local_to_rel[row_cols]
            keep = rel_idx >= 0
            nz_gene_rel_idx = rel_idx[keep].astype(np.int32, copy=False)
            nz_value = row_vals[keep]
        else:
            nz_gene_rel_idx = np.empty((0,), dtype=np.int32)
            nz_value = np.empty((0,), dtype=np.uint32)

        output = (
            sample_id_int,
            tissue_id_int,
            celltype_index,
            block_idx,
            nz_gene_rel_idx,
            nz_value,
        )

        if self.use_cache:
            self.cache[idx] = output

        return output


class SparseBatchCollator:
    """
    Batch-level densification to model-ready tensors:
    - x_mask: observed genes (bool), shape (B, G)
    - x_count: observed counts (float32), zeros for unobserved, shape (B, G)
    """

    def __init__(
        self,
        dataset: FastXVerseBatchDataset,
        num_genes: int,
        apply_mask_aug: bool = False,
        mask_aug_prob: float = 1.0,
        mask_aug_policy: str = "xverse",
        mask_aug_min_frac: float = 0.1,
        mask_aug_max_frac: float = 0.5,
    ):
        self.dataset = dataset
        self.num_genes = int(num_genes)
        self.apply_mask_aug = bool(apply_mask_aug)
        self.mask_aug_prob = float(mask_aug_prob)
        self.mask_aug_policy = str(mask_aug_policy)
        self.mask_aug_min_frac = float(mask_aug_min_frac)
        self.mask_aug_max_frac = float(mask_aug_max_frac)

    @staticmethod
    def _random_hide_observed_numpy(
        x_mask: np.ndarray,
        apply_prob: float,
        policy: str,
        min_frac: float,
        max_frac: float,
    ) -> np.ndarray:
        """
        Numpy implementation used in DataLoader workers.
        x_mask: bool array, shape (B, G)
        """
        if apply_prob <= 0:
            return x_mask.copy()

        enc = x_mask.copy()
        B = enc.shape[0]
        apply_flags = np.random.rand(B) <= float(apply_prob)
        simple_min = max(0.0, min(1.0, float(min_frac)))
        simple_max = max(simple_min, min(1.0, float(max_frac)))

        for i in range(B):
            if not apply_flags[i]:
                continue
            obs_idx = np.flatnonzero(enc[i])
            n_obs = int(obs_idx.size)
            if n_obs <= 1:
                continue

            if policy == "xverse":
                if n_obs < 1000:
                    max_to_mask = max(5, int(n_obs * (1.0 / 5.0)))
                    if n_obs > 10:
                        low = 10
                        high = max(low, max_to_mask)
                        n_hide = int(np.random.randint(low, high + 1))
                    else:
                        n_hide = n_obs
                else:
                    p = float(np.random.rand())
                    if p < 0.2:
                        frac = 0.1 + 0.2 * float(np.random.rand())
                    elif p < 0.9:
                        frac = 0.3 + 0.2 * float(np.random.rand())
                    else:
                        frac = 0.5 + 0.25 * float(np.random.rand())
                    n_hide = int(n_obs * frac)
            else:
                frac = simple_min + (simple_max - simple_min) * float(np.random.rand())
                n_hide = int(n_obs * frac)

            n_hide = max(1, min(n_hide, n_obs - 1))
            hide_idx = np.random.choice(obs_idx, size=n_hide, replace=False)
            enc[i, hide_idx] = False
        return enc

    def __call__(self, batch):
        bsz = len(batch)
        x_count = np.zeros((bsz, self.num_genes), dtype=np.float32)
        x_mask = np.zeros((bsz, self.num_genes), dtype=np.bool_)
        sample_ids = np.empty((bsz,), dtype=np.int64)
        tissue_ids = np.empty((bsz,), dtype=np.int64)
        celltype_ids = np.empty((bsz,), dtype=np.int64)

        for i, (sample_id_int, tissue_id_int, celltype_index, block_idx, nz_gene_rel_idx, nz_value) in enumerate(batch):
            sample_ids[i] = sample_id_int
            tissue_ids[i] = tissue_id_int
            celltype_ids[i] = celltype_index

            observed_global_idx = self.dataset.block_observed_global_idx[block_idx]
            if observed_global_idx.size > 0:
                x_mask[i, observed_global_idx] = True
            if nz_gene_rel_idx.size > 0:
                x_count[i, observed_global_idx[nz_gene_rel_idx]] = nz_value

        if self.apply_mask_aug:
            x_mask_encoder = self._random_hide_observed_numpy(
                x_mask=x_mask,
                apply_prob=self.mask_aug_prob,
                policy=self.mask_aug_policy,
                min_frac=self.mask_aug_min_frac,
                max_frac=self.mask_aug_max_frac,
            )
        else:
            x_mask_encoder = x_mask

        return (
            torch.from_numpy(sample_ids),
            torch.from_numpy(tissue_ids),
            torch.from_numpy(celltype_ids),
            torch.from_numpy(x_count),
            torch.from_numpy(x_mask),
            torch.from_numpy(x_mask_encoder),
        )


class CompiledShardDataset(Dataset):
    """
    Read-only dataset for compiled xverse_train_v1 shards.
    Output tuple matches training loop expectations:
    (sample_id, tissue_id, celltype_id, nz_gene_global_idx, nz_value)
    """

    def __init__(
        self,
        compiled_root: str,
        split: str = "train",
        max_cached_shards: int = 8,
    ):
        self.compiled_root = str(compiled_root)
        self.split = str(split)
        self.max_cached_shards = max(1, int(max_cached_shards))

        manifest_path = os.path.join(self.compiled_root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest.json not found in {self.compiled_root}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        fmt = manifest.get("format")
        if fmt != "xverse_train_v1":
            raise ValueError(f"Unsupported compiled dataset format: {fmt}")

        split_obj = manifest.get("splits", {}).get(self.split)
        if split_obj is None:
            raise ValueError(f"Split '{self.split}' not found in manifest.")

        self.global_num_genes = int(manifest.get("global_num_genes", -1))
        self.total_cells = int(split_obj.get("num_cells", 0))
        if self.global_num_genes <= 0:
            raise ValueError("Invalid global_num_genes in manifest.")

        shard_rows = split_obj.get("shards", [])
        self.shards = sorted(shard_rows, key=lambda x: int(x["global_cell_start"]))
        self._shard_starts = [int(x["global_cell_start"]) for x in self.shards]
        self._shard_ends = [int(x["global_cell_end"]) for x in self.shards]
        if self._shard_ends and self._shard_ends[-1] != self.total_cells:
            raise ValueError(
                f"Manifest inconsistency: last shard end={self._shard_ends[-1]} vs total_cells={self.total_cells}"
            )

        self._shard_cache = OrderedDict()

    def __len__(self):
        return self.total_cells

    def _load_shard_arrays(self, shard_idx: int):
        rec = self.shards[shard_idx]
        shard_dir = rec["path"]
        arrays = {
            "cell_ptr": np.load(os.path.join(shard_dir, "cell_ptr.npy"), mmap_mode="r"),
            "gene_idx": np.load(os.path.join(shard_dir, "gene_idx.npy"), mmap_mode="r"),
            "gene_val": np.load(os.path.join(shard_dir, "gene_val.npy"), mmap_mode="r"),
            "celltype_id": np.load(os.path.join(shard_dir, "celltype_id.npy"), mmap_mode="r"),
            "sample_id": np.load(os.path.join(shard_dir, "sample_id.npy"), mmap_mode="r"),
            "tissue_id": np.load(os.path.join(shard_dir, "tissue_id.npy"), mmap_mode="r"),
        }
        return arrays

    def _get_shard_arrays(self, shard_idx: int):
        arrays = self._shard_cache.get(shard_idx)
        if arrays is not None:
            self._shard_cache.move_to_end(shard_idx)
            return arrays

        arrays = self._load_shard_arrays(shard_idx)
        self._shard_cache[shard_idx] = arrays
        if len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return arrays

    def _locate_shard(self, idx: int) -> int:
        pos = bisect.bisect_right(self._shard_starts, int(idx)) - 1
        if pos < 0 or pos >= len(self.shards):
            raise IndexError(f"Index out of range: {idx}")
        if int(idx) >= self._shard_ends[pos]:
            raise IndexError(f"Index out of range: {idx}")
        return pos

    def shard_ids_for_indices(self, indices: List[int]) -> np.ndarray:
        if len(indices) == 0:
            return np.empty((0,), dtype=np.int32)
        arr = np.asarray(indices, dtype=np.int64)
        ends = np.asarray(self._shard_ends, dtype=np.int64)
        shard_ids = np.searchsorted(ends, arr, side="right").astype(np.int32, copy=False)
        return shard_ids

    def get_all_sample_ids(self) -> np.ndarray:
        out = np.empty((self.total_cells,), dtype=np.int32)
        cursor = 0
        for shard_idx, rec in enumerate(self.shards):
            arrays = self._get_shard_arrays(shard_idx)
            sids = np.asarray(arrays["sample_id"], dtype=np.int32)
            n = int(sids.size)
            out[cursor:cursor + n] = sids
            cursor += n
        return out

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self.total_cells:
            raise IndexError(f"Index out of range: {idx}")

        shard_idx = self._locate_shard(idx)
        rec = self.shards[shard_idx]
        local_idx = idx - int(rec["global_cell_start"])
        arrays = self._get_shard_arrays(shard_idx)

        cell_ptr = arrays["cell_ptr"]
        st = int(cell_ptr[local_idx])
        ed = int(cell_ptr[local_idx + 1])
        if st < ed:
            nz_gene_global_idx = np.asarray(arrays["gene_idx"][st:ed], dtype=np.int32)
            nz_value = np.asarray(arrays["gene_val"][st:ed])
        else:
            nz_gene_global_idx = np.empty((0,), dtype=np.int32)
            nz_value = np.empty((0,), dtype=np.uint16)

        sample_id_int = int(arrays["sample_id"][local_idx])
        tissue_id_int = int(arrays["tissue_id"][local_idx])
        celltype_index = int(arrays["celltype_id"][local_idx])

        return (
            sample_id_int,
            tissue_id_int,
            celltype_index,
            nz_gene_global_idx,
            nz_value,
        )

    def infer_num_celltypes(self) -> int:
        max_id = -1
        for shard_idx in range(len(self.shards)):
            arrays = self._get_shard_arrays(shard_idx)
            arr = np.asarray(arrays["celltype_id"])
            if arr.size <= 0:
                continue
            cur = int(arr.max())
            if cur > max_id:
                max_id = cur
        return int(max_id + 1) if max_id >= 0 else 0

    def infer_num_tissues(self) -> int:
        max_id = -1
        for shard_idx in range(len(self.shards)):
            arrays = self._get_shard_arrays(shard_idx)
            arr = np.asarray(arrays["tissue_id"])
            if arr.size <= 0:
                continue
            cur = int(arr.max())
            if cur > max_id:
                max_id = cur
        return int(max_id + 1) if max_id >= 0 else 0

    def infer_num_samples(self) -> int:
        max_id = -1
        for shard_idx in range(len(self.shards)):
            arrays = self._get_shard_arrays(shard_idx)
            arr = np.asarray(arrays["sample_id"])
            if arr.size <= 0:
                continue
            cur = int(arr.max())
            if cur > max_id:
                max_id = cur
        return int(max_id + 1) if max_id >= 0 else 0


class CompiledSparseBatchCollator:
    """
    Collator for CompiledShardDataset. Produces model-ready tensors:
    - x_count: (B, G) float32
    - x_mask: (B, G) bool
    - x_mask_encoder: optional augmented mask for encoder input
    """

    def __init__(
        self,
        num_genes: int,
        apply_mask_aug: bool = False,
        mask_aug_prob: float = 1.0,
        mask_aug_policy: str = "xverse",
        mask_aug_min_frac: float = 0.1,
        mask_aug_max_frac: float = 0.5,
    ):
        self.num_genes = int(num_genes)
        self.apply_mask_aug = bool(apply_mask_aug)
        self.mask_aug_prob = float(mask_aug_prob)
        self.mask_aug_policy = str(mask_aug_policy)
        self.mask_aug_min_frac = float(mask_aug_min_frac)
        self.mask_aug_max_frac = float(mask_aug_max_frac)

    def __call__(self, batch):
        bsz = len(batch)
        x_count = np.zeros((bsz, self.num_genes), dtype=np.float32)
        x_mask = np.zeros((bsz, self.num_genes), dtype=np.bool_)
        sample_ids = np.empty((bsz,), dtype=np.int64)
        tissue_ids = np.empty((bsz,), dtype=np.int64)
        celltype_ids = np.empty((bsz,), dtype=np.int64)

        for i, (sample_id_int, tissue_id_int, celltype_index, nz_gene_global_idx, nz_value) in enumerate(batch):
            sample_ids[i] = sample_id_int
            tissue_ids[i] = tissue_id_int
            celltype_ids[i] = celltype_index
            if nz_gene_global_idx.size > 0:
                x_mask[i, nz_gene_global_idx] = True
                x_count[i, nz_gene_global_idx] = nz_value

        if self.apply_mask_aug:
            x_mask_encoder = SparseBatchCollator._random_hide_observed_numpy(
                x_mask=x_mask,
                apply_prob=self.mask_aug_prob,
                policy=self.mask_aug_policy,
                min_frac=self.mask_aug_min_frac,
                max_frac=self.mask_aug_max_frac,
            )
        else:
            x_mask_encoder = x_mask

        return (
            torch.from_numpy(sample_ids),
            torch.from_numpy(tissue_ids),
            torch.from_numpy(celltype_ids),
            torch.from_numpy(x_count),
            torch.from_numpy(x_mask),
            torch.from_numpy(x_mask_encoder),
        )


class CompiledBalancedSampler(Sampler):
    def __init__(
        self,
        dataset: CompiledShardDataset,
        samples_per_id=None,
        seed: int = 0,
        shard_reorder_window: int = 4096,
        active_shards: int = 0,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.shard_reorder_window = max(0, int(shard_reorder_window))
        self.active_shards = max(0, int(active_shards))
        self.num_shards = len(self.dataset.shards)

        sample_ids = dataset.get_all_sample_ids()
        self.samples_by_id = defaultdict(list)
        for idx, sid in enumerate(sample_ids.tolist()):
            self.samples_by_id[int(sid)].append(idx)

        self.sample_ids = list(self.samples_by_id.keys())
        self.samples_per_id = samples_per_id or min(len(v) for v in self.samples_by_id.values())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for sample_id in self.sample_ids:
            candidates = self.samples_by_id[sample_id]
            if len(candidates) >= self.samples_per_id:
                selected = rng.sample(candidates, self.samples_per_id)
            else:
                selected = [candidates[rng.randrange(len(candidates))] for _ in range(self.samples_per_id)]
            indices.extend(selected)
        rng.shuffle(indices)
        if self.active_shards > 0 and self.num_shards > self.active_shards and len(indices) > 0:
            shard_order = list(range(self.num_shards))
            rng.shuffle(shard_order)
            pool_rank_of_shard = np.empty((self.num_shards,), dtype=np.int32)
            for rank, sid in enumerate(shard_order):
                pool_rank_of_shard[sid] = rank // self.active_shards
            shard_ids = self.dataset.shard_ids_for_indices(indices)
            pool_rank = pool_rank_of_shard[shard_ids]
            order = np.argsort(pool_rank, kind="stable")
            indices = [indices[int(i)] for i in order]
        if self.shard_reorder_window > 1 and len(indices) > self.shard_reorder_window:
            w = self.shard_reorder_window
            reordered = []
            for st in range(0, len(indices), w):
                chunk = indices[st:st + w]
                shard_ids = self.dataset.shard_ids_for_indices(chunk)
                order = np.argsort(shard_ids, kind="stable")
                reordered.extend(chunk[int(i)] for i in order)
            indices = reordered
        return iter(indices)

    def __len__(self):
        return self.samples_per_id * len(self.sample_ids)


class DistributedCompiledBalancedSampler(Sampler):
    def __init__(
        self,
        dataset: CompiledShardDataset,
        samples_per_id=None,
        num_replicas=None,
        rank=None,
        seed: int = 0,
        shard_reorder_window: int = 4096,
        active_shards: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed package is required for DistributedCompiledBalancedSampler")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed package is required for DistributedCompiledBalancedSampler")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.shard_reorder_window = max(0, int(shard_reorder_window))
        self.active_shards = max(0, int(active_shards))
        self.num_shards = len(self.dataset.shards)

        sample_ids = dataset.get_all_sample_ids()
        self.samples_by_id = defaultdict(list)
        for idx, sid in enumerate(sample_ids.tolist()):
            self.samples_by_id[int(sid)].append(idx)
        self.sample_ids = list(self.samples_by_id.keys())
        self.samples_per_id = samples_per_id or min(len(v) for v in self.samples_by_id.values())

        self.global_num_samples = self.samples_per_id * len(self.sample_ids)
        self.num_samples = int(math.ceil(self.global_num_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for sample_id in self.sample_ids:
            candidates = self.samples_by_id[sample_id]
            if len(candidates) >= self.samples_per_id:
                selected = rng.sample(candidates, self.samples_per_id)
            else:
                selected = [candidates[rng.randrange(len(candidates))] for _ in range(self.samples_per_id)]
            indices.extend(selected)
        rng.shuffle(indices)
        if self.active_shards > 0 and self.num_shards > self.active_shards and len(indices) > 0:
            shard_order = list(range(self.num_shards))
            rng.shuffle(shard_order)
            pool_rank_of_shard = np.empty((self.num_shards,), dtype=np.int32)
            for rank, sid in enumerate(shard_order):
                pool_rank_of_shard[sid] = rank // self.active_shards
            shard_ids = self.dataset.shard_ids_for_indices(indices)
            pool_rank = pool_rank_of_shard[shard_ids]
            order = np.argsort(pool_rank, kind="stable")
            indices = [indices[int(i)] for i in order]
        if self.shard_reorder_window > 1 and len(indices) > self.shard_reorder_window:
            w = self.shard_reorder_window
            reordered = []
            for st in range(0, len(indices), w):
                chunk = indices[st:st + w]
                shard_ids = self.dataset.shard_ids_for_indices(chunk)
                order = np.argsort(shard_ids, kind="stable")
                reordered.extend(chunk[int(i)] for i in order)
            indices = reordered

        if len(indices) < self.total_size:
            indices.extend(indices[: self.total_size - len(indices)])
        else:
            indices = indices[: self.total_size]

        rank_indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples


class BalancedSampleSampler(Sampler):
    def __init__(self, dataset, samples_per_id=None, seed: int = 0):
        self.dataset = dataset
        self.samples_by_id = defaultdict(list)
        for idx, (_, _, sample_id, _) in enumerate(dataset.index_map):
            self.samples_by_id[sample_id].append(idx)

        self.sample_ids = list(self.samples_by_id.keys())
        self.samples_per_id = samples_per_id or min(len(v) for v in self.samples_by_id.values())
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for sample_id in self.sample_ids:
            candidates = self.samples_by_id[sample_id]
            if len(candidates) >= self.samples_per_id:
                selected = rng.sample(candidates, self.samples_per_id)
            else:
                selected = [candidates[rng.randrange(len(candidates))] for _ in range(self.samples_per_id)]
            indices.extend(selected)
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_id * len(self.sample_ids)


class DistributedBalancedSampler(Sampler):
    def __init__(self, dataset, samples_per_id=None, num_replicas=None, rank=None, seed: int = 0):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed package is required for DistributedBalancedSampler")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed package is required for DistributedBalancedSampler")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        self.samples_by_id = defaultdict(list)
        for idx, (_, _, sample_id, _) in enumerate(dataset.index_map):
            self.samples_by_id[sample_id].append(idx)
        self.sample_ids = list(self.samples_by_id.keys())
        self.samples_per_id = samples_per_id or min(len(v) for v in self.samples_by_id.values())

        self.global_num_samples = self.samples_per_id * len(self.sample_ids)
        self.num_samples = int(math.ceil(self.global_num_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for sample_id in self.sample_ids:
            candidates = self.samples_by_id[sample_id]
            if len(candidates) >= self.samples_per_id:
                selected = rng.sample(candidates, self.samples_per_id)
            else:
                selected = [candidates[rng.randrange(len(candidates))] for _ in range(self.samples_per_id)]
            indices.extend(selected)
        rng.shuffle(indices)

        if len(indices) < self.total_size:
            indices.extend(indices[: self.total_size - len(indices)])
        else:
            indices = indices[: self.total_size]

        rank_indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples


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
