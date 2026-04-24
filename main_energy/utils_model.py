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
    Learnable low-rank-covariance GMM prior.
    Component covariance:
        Sigma_k = diag(exp(prior_logvar_k)) + U_k U_k^T
    """

    def __init__(self, num_components: int, latent_dim: int, cov_rank: int = 8):
        super().__init__()
        self.K = num_components
        self.D = latent_dim
        self.R = max(0, int(cov_rank))
        self.pi_logits = nn.Parameter(torch.zeros(num_components))
        self.prior_mu = nn.Parameter(torch.randn(num_components, latent_dim) * 0.01)
        self.prior_logvar = nn.Parameter(torch.zeros(num_components, latent_dim))
        if self.R > 0:
            self.prior_factor = nn.Parameter(torch.randn(num_components, latent_dim, self.R) * 0.01)
        else:
            self.register_parameter("prior_factor", None)

    def _cache_matrices(self):
        device_type = self.prior_mu.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # (K, D)
            logvar = self.prior_logvar.float()
            d_inv = torch.exp(-logvar)
            logdet_base = logvar.sum(dim=-1)  # (K,)

            if self.R <= 0:
                return {
                    "d_inv": d_inv,
                    "dinv_u": None,
                    "inv_s": None,
                    "logdet_extra": torch.zeros_like(logdet_base),
                }

            # U: (K, D, R)
            u = self.prior_factor.float()
            dinv_u = d_inv.unsqueeze(-1) * u  # (K, D, R)
            # S = I + U^T D^{-1} U  -> (K, R, R)
            ut_dinv_u = torch.einsum("kdr,kds->krs", u, dinv_u)
            eye = torch.eye(self.R, device=ut_dinv_u.device, dtype=ut_dinv_u.dtype).unsqueeze(0)
            s = ut_dinv_u + eye
            # Numerical jitter for safety.
            s = s + 1e-6 * eye
            chol_s = torch.linalg.cholesky(s)
            inv_s = torch.cholesky_inverse(chol_s)
            logdet_extra = 2.0 * torch.log(torch.diagonal(chol_s, dim1=-2, dim2=-1)).sum(dim=-1)  # (K,)
            return {
                "d_inv": d_inv,
                "dinv_u": dinv_u,
                "inv_s": inv_s,
                "logdet_extra": logdet_extra,
            }

    def component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        device_type = z.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            z_exp = z.float().unsqueeze(1)  # (B, 1, D)
            mu = self.prior_mu.float().unsqueeze(0)  # (1, K, D)
            delta = z_exp - mu  # (B, K, D)

            cache = self._cache_matrices()
            d_inv = cache["d_inv"].unsqueeze(0)  # (1, K, D)
            delta_d = delta * d_inv  # (B, K, D)
            quad = (delta * delta_d).sum(dim=-1)  # (B, K)

            if self.R > 0:
                # t = U^T D^{-1} delta  -> (B, K, R)
                u = self.prior_factor.float()  # (K, D, R)
                t = torch.einsum("bkd,kdr->bkr", delta_d, u)
                inv_s = cache["inv_s"]  # (K, R, R)
                quad_corr = torch.einsum("bkr,krs,bks->bk", t, inv_s, t)
                quad = quad - quad_corr

            log_det = self.prior_logvar.float().sum(dim=-1) + cache["logdet_extra"]  # (K,)
            log_norm = self.D * math.log(2.0 * math.pi)
            out = -0.5 * (quad + log_det.unsqueeze(0) + log_norm)
        return out

    def component_log_prob_aligned(self, z_comp: torch.Tensor) -> torch.Tensor:
        """
        Aligned component log-probability for z sampled per component.
        Input:
            z_comp: (B, K, D), where z_comp[:, k, :] is evaluated under component k.
        Output:
            log_prob: (B, K)
        """
        device_type = z_comp.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            zc = z_comp.float()
            if zc.dim() != 3 or zc.size(1) != self.K or zc.size(2) != self.D:
                raise ValueError(
                    f"Expected z_comp shape (B,{self.K},{self.D}), got {tuple(zc.shape)}"
                )
            mu = self.prior_mu.float().unsqueeze(0)  # (1, K, D)
            delta = zc - mu  # (B, K, D)

            cache = self._cache_matrices()
            d_inv = cache["d_inv"].unsqueeze(0)  # (1, K, D)
            delta_d = delta * d_inv  # (B, K, D)
            quad = (delta * delta_d).sum(dim=-1)  # (B, K)

            if self.R > 0:
                u = self.prior_factor.float()  # (K, D, R)
                t = torch.einsum("bkd,kdr->bkr", delta_d, u)  # (B, K, R)
                inv_s = cache["inv_s"]  # (K, R, R)
                quad_corr = torch.einsum("bkr,krs,bks->bk", t, inv_s, t)
                quad = quad - quad_corr

            log_det = self.prior_logvar.float().sum(dim=-1) + cache["logdet_extra"]  # (K,)
            log_norm = self.D * math.log(2.0 * math.pi)
            out = -0.5 * (quad + log_det.unsqueeze(0) + log_norm)
        return out

    def log_prob(self, z: torch.Tensor, topk: int = 0) -> torch.Tensor:
        log_weights = F.log_softmax(self.pi_logits, dim=0).unsqueeze(0)  # (1, K)
        log_comp = self.component_log_prob(z)  # (B, K)
        logits = log_weights + log_comp
        k = int(topk)
        if k > 0 and k < logits.size(1):
            top_vals, _ = torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
            return torch.logsumexp(top_vals, dim=1)
        return torch.logsumexp(logits, dim=1)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Analytic score of GMM prior: grad_z log p(z), shape (B, D).
        """
        device_type = z.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            z_exp = z.float().unsqueeze(1)  # (B, 1, D)
            mu = self.prior_mu.float().unsqueeze(0)  # (1, K, D)
            delta = z_exp - mu  # (B, K, D)
            cache = self._cache_matrices()
            d_inv = cache["d_inv"].unsqueeze(0)  # (1, K, D)
            delta_d = delta * d_inv  # (B, K, D)

            log_weights = F.log_softmax(self.pi_logits.float(), dim=0).unsqueeze(0)  # (1, K)
            log_comp = self.component_log_prob(z)  # (B, K)
            log_post = log_weights + log_comp
            resp = torch.softmax(log_post, dim=1).unsqueeze(-1)  # (B, K, 1)

            inv_delta = delta_d
            if self.R > 0:
                inv_s = cache["inv_s"]  # (K, R, R)
                # t = U^T D^{-1} delta  -> (B, K, R)
                t = torch.einsum("bkd,kdr->bkr", delta_d, self.prior_factor.float())
                # s = inv(S) t
                s = torch.einsum("krs,bks->bkr", inv_s, t)
                # correction = D^{-1} U s
                corr = torch.einsum("kdr,bkr->bkd", cache["dinv_u"], s)
                inv_delta = delta_d - corr

            per_comp_score = -inv_delta  # grad_z log N = -Sigma^{-1}(z-mu)
            score = (resp * per_comp_score).sum(dim=1)  # (B, D)
        return score.to(z.dtype)

    def component_covariances(self) -> torch.Tensor:
        # (K, D, D)
        diag_cov = torch.diag_embed(torch.exp(self.prior_logvar.float()))
        if self.R <= 0:
            return diag_cov
        f = self.prior_factor.float()
        low_rank_cov = torch.einsum("kdr,ksr->kds", f, f)
        return diag_cov + low_rank_cov

    def global_covariance(self) -> torch.Tensor:
        w = torch.softmax(self.pi_logits.float(), dim=0)  # (K,)
        cov_k = self.component_covariances()  # (K, D, D)
        mu = self.prior_mu.float()
        mean = (w.unsqueeze(1) * mu).sum(dim=0)  # (D,)
        second = (
            w.unsqueeze(1).unsqueeze(2)
            * (cov_k + mu.unsqueeze(2) * mu.unsqueeze(1))
        ).sum(dim=0)
        return second - mean.unsqueeze(1) * mean.unsqueeze(0)

    def posterior_responsibilities(self, z: torch.Tensor, temperature: float = 1.0, topk: int = 0) -> torch.Tensor:
        t = max(float(temperature), 1e-6)
        log_weights = F.log_softmax(self.pi_logits.float(), dim=0).unsqueeze(0)  # (1, K)
        log_comp = self.component_log_prob(z)  # (B, K)
        logits = (log_weights + log_comp) / t
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
            self.prior_logvar.clamp_(min=min_val, max=max_val)


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
        return self.decoder(z)


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
        posterior_cov_rank: int = 0,
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
        )
        self.decoder = PoissonDecoder(
            latent_dim=latent_dim,
            num_genes=num_genes,
            hidden_dim=dec_hidden_dim,
            dropout=dropout,
        )
        self.num_components = int(num_components)
        self.latent_dim = int(latent_dim)
        self.posterior_cov_rank = max(0, int(posterior_cov_rank))
        if self.prior_type == "gmm":
            self.post_c_logits = nn.Linear(expr_hidden_dim, num_components)
            self.post_mu = nn.Linear(expr_hidden_dim, num_components * latent_dim)
            self.post_logvar = nn.Linear(expr_hidden_dim, num_components * latent_dim)
            if self.posterior_cov_rank > 0:
                self.post_factor = nn.Linear(expr_hidden_dim, num_components * latent_dim * self.posterior_cov_rank)
            else:
                self.post_factor = None
        # Library-size head from latent z.
        self.library_head = nn.Linear(latent_dim, 1)
        self.score_head = MLP(
            input_dim=latent_dim,
            hidden_dims=[dec_hidden_dim, dec_hidden_dim],
            output_dim=latent_dim,
            dropout=dropout,
        )

    def forward(self, x_count: torch.Tensor, x_mask: torch.Tensor, x_expr: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if x_expr is None:
            x_expr = torch.log1p(x_count.float())
        mu_enc, logvar_enc, h = self.encoder(x_expr=x_expr, x_mask=x_mask.float(), return_hidden=True)
        if self.prior_type == "gmm":
            bsz = h.size(0)
            k = self.num_components
            d = self.latent_dim
            q_c_logits = self.post_c_logits(h)  # (B, K)
            q_c = torch.softmax(q_c_logits, dim=-1)
            mu_comp = self.post_mu(h).view(bsz, k, d)
            logvar_comp = torch.clamp(self.post_logvar(h).view(bsz, k, d), min=-8.0, max=8.0)
            std_comp = torch.exp(0.5 * logvar_comp)
            eps_d = torch.randn_like(std_comp)
            z_comp = mu_comp + eps_d * std_comp  # (B, K, D)
            factor_comp = None
            if self.posterior_cov_rank > 0 and self.post_factor is not None:
                r = self.posterior_cov_rank
                # Keep factor bounded for stability.
                factor_comp = 0.1 * torch.tanh(self.post_factor(h).view(bsz, k, d, r))
                eps_r = torch.randn((bsz, k, r), device=h.device, dtype=h.dtype)
                z_comp = z_comp + torch.einsum("bkdr,bkr->bkd", factor_comp, eps_r)
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
            factor_comp = None
        gene_logits = self.decoder(z)
        library_size = F.softplus(self.library_head(z)) + 1e-8
        gene_probs = F.softmax(gene_logits, dim=-1)
        rate = gene_probs * library_size
        out = {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "library_size": library_size,
            "gene_logits": gene_logits,
            "rate": rate,
        }
        if self.prior_type == "gmm":
            out.update(
                {
                    "q_c_logits": q_c_logits,
                    "q_c": q_c,
                    "mu_comp": mu_comp,
                    "logvar_comp": logvar_comp,
                    "z_comp": z_comp,
                    "factor_comp": factor_comp,
                }
            )
        return out

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
        lambda_cov: float = 0.0,
        cov_use_mu: bool = True,
        lambda_resp_balance: float = 0.0,
        lambda_resp_confidence: float = 0.0,
        resp_temperature: float = 1.0,
        resp_topk: int = 0,
        prior_logvar_min: float = -6.0,
        prior_logvar_max: float = 4.0,
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
            self.prior.clamp_logvar_(min_val=prior_logvar_min, max_val=prior_logvar_max)
            q_c_logits = out["q_c_logits"]  # (B, K)
            q_c = out["q_c"]  # (B, K)
            mu_comp = out["mu_comp"]  # (B, K, D)
            logvar_comp = out["logvar_comp"]  # (B, K, D)
            z_comp = out["z_comp"]  # (B, K, D)
            factor_comp = out.get("factor_comp", None)

            log_q_c = F.log_softmax(q_c_logits, dim=-1)  # (B, K)
            log_p_c = F.log_softmax(self.prior.pi_logits, dim=0).unsqueeze(0)  # (1, K)
            kl_c = (q_c * (log_q_c - log_p_c)).sum(dim=1).mean()

            if factor_comp is not None:
                log_q_z_given_c = gaussian_log_prob_lowrank(
                    z=z_comp,
                    mu=mu_comp,
                    logvar=logvar_comp,
                    factor=factor_comp,
                )  # (B, K)
            else:
                log_q_z_given_c = gaussian_log_prob_diag(z=z_comp, mu=mu_comp, logvar=logvar_comp)  # (B, K)
            log_p_z_given_c = self.prior.component_log_prob_aligned(z_comp)  # (B, K)
            kl_z = (q_c * (log_q_z_given_c - log_p_z_given_c)).sum(dim=1).mean()

            kl_loss = kl_c + kl_z
            log_q = (q_c * (log_q_c + log_q_z_given_c)).sum(dim=1)
            log_p = (q_c * (log_p_c + log_p_z_given_c)).sum(dim=1)
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

        if lambda_cov > 0:
            cov_src = mu if cov_use_mu else z
            post_cov = batch_covariance(cov_src.float())
            if self.prior_type == "gmm":
                prior_cov = self.prior.global_covariance().float()
            else:
                d = int(cov_src.size(-1))
                prior_cov = torch.eye(d, device=cov_src.device, dtype=torch.float32)

            post_off = offdiag_part(post_cov)
            prior_off = offdiag_part(prior_cov)
            cov_loss = F.mse_loss(post_off, prior_off)
            cov_offdiag_post = post_off.abs().mean().to(z.dtype)
            cov_offdiag_prior = prior_off.abs().mean().to(z.dtype)

        if self.prior_type == "gmm" and (lambda_resp_balance > 0 or lambda_resp_confidence > 0):
            if "q_c" in out:
                q_logits = out["q_c_logits"]
                t = max(float(resp_temperature), 1e-6)
                logits = q_logits / t
                k_keep = int(resp_topk)
                if k_keep > 0 and k_keep < logits.size(1):
                    top_vals, top_idx = torch.topk(logits, k=k_keep, dim=1, largest=True, sorted=False)
                    top_resp = torch.softmax(top_vals, dim=1)
                    resp = torch.zeros_like(logits)
                    resp.scatter_(1, top_idx, top_resp)
                else:
                    resp = torch.softmax(logits, dim=1)
            else:
                resp = self.prior.posterior_responsibilities(z=z, temperature=resp_temperature, topk=resp_topk)
            resp_entropy = (-(resp * torch.log(resp + 1e-12)).sum(dim=1)).mean().to(z.dtype)
            resp_top1 = resp.max(dim=1).values.mean().to(z.dtype)
            # minimize entropy => encourage each cell to use fewer components
            resp_confidence_loss = resp_entropy
            if lambda_resp_balance > 0:
                usage = resp.mean(dim=0)  # (K,)
                target = torch.full_like(usage, 1.0 / float(usage.numel()))
                resp_balance_loss = F.kl_div(
                    torch.log(usage + 1e-12),
                    target,
                    reduction="sum",
                ).to(z.dtype)

        total_loss = (
            recon_loss
            + beta * kl_loss
            + lambda_score * score_loss
            + lambda_cov * cov_loss
            + lambda_resp_balance * resp_balance_loss
            + lambda_resp_confidence * resp_confidence_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "score_loss": score_loss,
            "cov_loss": cov_loss,
            "cov_offdiag_post": cov_offdiag_post,
            "cov_offdiag_prior": cov_offdiag_prior,
            "resp_entropy": resp_entropy,
            "resp_top1": resp_top1,
            "resp_balance_loss": resp_balance_loss,
            "resp_confidence_loss": resp_confidence_loss,
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
    d_inv = torch.exp(-logvar)  # (B, K, D)
    delta = z - mu  # (B, K, D)
    delta_d = delta * d_inv  # (B, K, D)
    quad = (delta * delta_d).sum(dim=-1)  # (B, K)

    # U^T D^{-1} U  -> (B, K, R, R)
    dinv_u = d_inv.unsqueeze(-1) * factor
    ut_dinv_u = torch.einsum("bkdr,bkds->bkrs", factor, dinv_u)
    r = int(factor.size(-1))
    eye = torch.eye(r, device=factor.device, dtype=factor.dtype).view(1, 1, r, r)
    s = ut_dinv_u + eye + 1e-6 * eye
    chol_s = torch.linalg.cholesky(s)
    inv_s = torch.cholesky_inverse(chol_s)
    logdet_extra = 2.0 * torch.log(torch.diagonal(chol_s, dim1=-2, dim2=-1)).sum(dim=-1)  # (B, K)

    t = torch.einsum("bkd,bkdr->bkr", delta_d, factor)  # (B, K, R)
    quad_corr = torch.einsum("bkr,bkrs,bks->bk", t, inv_s, t)
    quad = quad - quad_corr

    log_det = logvar.sum(dim=-1) + logdet_extra  # (B, K)
    d = z.size(-1)
    return -0.5 * (quad + log_det + d * math.log(2.0 * math.pi))


def batch_covariance(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor for covariance, got shape={tuple(x.shape)}")
    bsz = x.size(0)
    xc = x - x.mean(dim=0, keepdim=True)
    denom = max(bsz - 1, 1)
    return (xc.transpose(0, 1) @ xc) / float(denom)


def offdiag_part(m: torch.Tensor) -> torch.Tensor:
    return m - torch.diag_embed(torch.diagonal(m, dim1=-2, dim2=-1))


def poisson_nll(x_count: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    x = x_count.float()
    r = torch.clamp(rate, min=1e-8)
    nll = r - x * torch.log(r)
    return nll.mean()


def poisson_nll_masked(x_count: torch.Tensor, rate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x_count.float()
    r = torch.clamp(rate, min=1e-8)
    m = mask.float()
    nll = r - x * torch.log(r)
    nll = nll * m
    denom = torch.clamp(m.sum(), min=1.0)
    return nll.sum() / denom


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


def gmm_collapse_diagnostics(prior: GaussianMixturePrior, z: torch.Tensor, active_thresh: float = 1e-3) -> Dict[str, float]:
    with torch.no_grad():
        pi = torch.softmax(prior.pi_logits.float(), dim=0)  # (K,)
        pi_entropy = float((-(pi * torch.log(pi + 1e-12))).sum().item())
        k_eff = float(torch.exp(torch.tensor(pi_entropy, device=pi.device)).item())

        log_w = torch.log(pi + 1e-12).unsqueeze(0)  # (1, K)
        log_comp = prior.component_log_prob(z.detach())  # (B, K)
        resp = torch.softmax(log_w + log_comp, dim=1)  # (B, K)
        usage = resp.mean(dim=0)  # (K,)
        active_comp = int((usage > float(active_thresh)).sum().item())
        resp_top1 = float(resp.max(dim=1).values.mean().item())

        mu = prior.prior_mu.float()  # (K, D)
        if mu.size(0) > 1:
            dmat = torch.cdist(mu, mu, p=2)
            diag_mask = torch.eye(dmat.size(0), device=dmat.device, dtype=torch.bool)
            dmat = dmat.masked_fill(diag_mask, float("inf"))
            min_mu_dist = float(torch.min(dmat).item())
        else:
            min_mu_dist = 0.0

    return {
        "pi_entropy": pi_entropy,
        "k_eff": k_eff,
        "active_comp": active_comp,
        "resp_top1": resp_top1,
        "min_mu_dist": min_mu_dist,
    }


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
    resp_temperature=1.0,
    resp_topk=0,
    prior_logvar_min=-6.0,
    prior_logvar_max=4.0,
):
    model.train()
    loss_fn = model.module.loss if hasattr(model, "module") else model.loss
    prior_ref = model.module.prior if hasattr(model, "module") else model.prior
    total_loss = total_recon = total_kl = total_score = total_contrast = total_cov = 0.0
    n_cells = 0
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    for batch_idx, (_, _, _, x_count, x_mask, x_mask_encoder) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        x_count = x_count.to(device, non_blocking=True)
        x_mask = x_mask.to(device, non_blocking=True)
        x_mask_encoder = x_mask_encoder.to(device, non_blocking=True)

        bsz = x_count.size(0)

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            out_fake = loss_fn(
                x_count=x_count,
                x_mask=x_mask,
                beta=beta_kl,
                encoder_mask=x_mask_encoder,
                recon_mask=x_mask if recon_observed_only else None,
                lambda_score=lambda_score,
                score_noise_std=score_noise_std,
                score_detach_z=score_detach_z,
                lambda_cov=lambda_cov,
                cov_use_mu=cov_use_mu,
                lambda_resp_balance=lambda_resp_balance,
                lambda_resp_confidence=lambda_resp_confidence,
                resp_temperature=resp_temperature,
                resp_topk=resp_topk,
                prior_logvar_min=prior_logvar_min,
                prior_logvar_max=prior_logvar_max,
            )
            need_real_view = (lambda_contrast > 0) or (lambda_real_recon > 0)
            real_recon = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)
            if need_real_view:
                out_real = loss_fn(
                    x_count=x_count,
                    x_mask=x_mask,
                    beta=0.0,
                    encoder_mask=x_mask,
                    recon_mask=x_mask if recon_observed_only else None,
                    lambda_score=0.0,
                    score_noise_std=score_noise_std,
                    score_detach_z=score_detach_z,
                    lambda_cov=0.0,
                    cov_use_mu=cov_use_mu,
                    lambda_resp_balance=0.0,
                    lambda_resp_confidence=0.0,
                    resp_temperature=resp_temperature,
                    resp_topk=0,
                    prior_logvar_min=prior_logvar_min,
                    prior_logvar_max=prior_logvar_max,
                )
                real_recon = out_real["recon_loss"]
            if lambda_contrast > 0:
                contrast = bidirectional_contrastive_loss(
                    z_real=out_real["z"],
                    z_fake=out_fake["z"],
                    temperature=contrast_temp,
                )
            else:
                contrast = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)

            loss = out_fake["loss"] + lambda_contrast * contrast + lambda_real_recon * real_recon
            recon = out_fake["recon_loss"]
            kl = out_fake["kl_loss"]
            score = out_fake["score_loss"]
            cov = out_fake["cov_loss"]

        if not torch.isfinite(loss):
            if is_rank0:
                print(
                    f"[WARN] Non-finite train loss at batch {batch_idx + 1}, skip step. "
                    f"Recon={recon.item():.4f}, KL={kl.item():.4f}, Score={score.item():.4f}, Cov={cov.item():.4f}"
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

        if (batch_idx + 1) % 100 == 0 and is_rank0:
            msg = (
                f"[Batch {batch_idx + 1}] "
                f"Loss={loss.item():.4f}, Recon={recon.item():.4f}, KL={kl.item():.4f}, "
                f"Score={score.item():.4f}, Cov={cov.item():.4f}"
            )
            if lambda_contrast > 0:
                msg += f", Contrast={contrast.item():.4f}"
            if lambda_real_recon > 0:
                msg += f", RealRecon={real_recon.item():.4f}"
            msg += (
                f", ||s_pred||={out_fake['score_norm_pred'].item():.4f}, ||s_tgt||={out_fake['score_norm_tgt'].item():.4f}, "
                f"||cov_post_off||={out_fake['cov_offdiag_post'].item():.6f}, ||cov_prior_off||={out_fake['cov_offdiag_prior'].item():.6f}, "
                f"respH={out_fake['resp_entropy'].item():.4f}, respBal={out_fake['resp_balance_loss'].item():.4f}, "
                f"respConf={out_fake['resp_confidence_loss'].item():.4f}, "
                f"respTop1Batch={out_fake['resp_top1'].item():.4f}"
            )
            if getattr(model.module if hasattr(model, "module") else model, "prior_type", None) == "gmm":
                diag = gmm_collapse_diagnostics(prior=prior_ref, z=out_fake["z"])
                msg += (
                    f", piH={diag['pi_entropy']:.3f}, K_eff={diag['k_eff']:.2f}, "
                    f"activeK={diag['active_comp']}, respTop1={diag['resp_top1']:.3f}, "
                    f"minMuDist={diag['min_mu_dist']:.3f}"
                )
            print(msg)

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, float(n_cells)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, n_cells = stats.tolist()
        n_cells = max(float(n_cells), 1.0)

    return (
        total_loss / n_cells,
        total_recon / n_cells,
        total_kl / n_cells,
        total_score / n_cells,
        total_contrast / n_cells,
        total_cov / n_cells,
    )


def evaluate_gmm_vae_one_epoch(
    model,
    val_loader,
    device,
    beta_kl=1.0,
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
    resp_temperature=1.0,
    resp_topk=0,
    prior_logvar_min=-6.0,
    prior_logvar_max=4.0,
):
    model.eval()
    loss_fn = model.module.loss if hasattr(model, "module") else model.loss
    prior_ref = model.module.prior if hasattr(model, "module") else model.prior
    total_loss = total_recon = total_kl = total_score = total_contrast = total_cov = 0.0
    n_cells = 0
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    with torch.no_grad():
        for batch_idx, (_, _, _, x_count, x_mask, x_mask_encoder) in enumerate(val_loader):
            x_count = x_count.to(device, non_blocking=True)
            x_mask = x_mask.to(device, non_blocking=True)
            x_mask_encoder = x_mask_encoder.to(device, non_blocking=True)
            bsz = x_count.size(0)
            n_cells += bsz

            out_fake = loss_fn(
                x_count=x_count,
                x_mask=x_mask,
                beta=beta_kl,
                encoder_mask=x_mask_encoder,
                recon_mask=x_mask if recon_observed_only else None,
                lambda_score=lambda_score,
                score_noise_std=score_noise_std,
                score_detach_z=score_detach_z,
                lambda_cov=lambda_cov,
                cov_use_mu=cov_use_mu,
                lambda_resp_balance=lambda_resp_balance,
                lambda_resp_confidence=lambda_resp_confidence,
                resp_temperature=resp_temperature,
                resp_topk=resp_topk,
                prior_logvar_min=prior_logvar_min,
                prior_logvar_max=prior_logvar_max,
            )
            need_real_view = (lambda_contrast > 0) or (lambda_real_recon > 0)
            real_recon = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)
            if need_real_view:
                out_real = loss_fn(
                    x_count=x_count,
                    x_mask=x_mask,
                    beta=0.0,
                    encoder_mask=x_mask,
                    recon_mask=x_mask if recon_observed_only else None,
                    lambda_score=0.0,
                    score_noise_std=score_noise_std,
                    score_detach_z=score_detach_z,
                    lambda_cov=0.0,
                    cov_use_mu=cov_use_mu,
                    lambda_resp_balance=0.0,
                    lambda_resp_confidence=0.0,
                    resp_temperature=resp_temperature,
                    resp_topk=0,
                    prior_logvar_min=prior_logvar_min,
                    prior_logvar_max=prior_logvar_max,
                )
                real_recon = out_real["recon_loss"]
            if lambda_contrast > 0:
                contrast = bidirectional_contrastive_loss(
                    z_real=out_real["z"],
                    z_fake=out_fake["z"],
                    temperature=contrast_temp,
                )
            else:
                contrast = torch.zeros((), device=x_count.device, dtype=out_fake["z"].dtype)
            loss = out_fake["loss"] + lambda_contrast * contrast + lambda_real_recon * real_recon

            total_loss += loss.item() * bsz
            total_recon += out_fake["recon_loss"].item() * bsz
            total_kl += out_fake["kl_loss"].item() * bsz
            total_score += out_fake["score_loss"].item() * bsz
            total_contrast += contrast.item() * bsz
            total_cov += out_fake["cov_loss"].item() * bsz

            if (batch_idx + 1) % 100 == 0 and is_rank0:
                msg = (
                    f"[Val Batch {batch_idx + 1}] "
                    f"Loss={loss.item():.4f}, Recon={out_fake['recon_loss'].item():.4f}, "
                    f"KL={out_fake['kl_loss'].item():.4f}, Score={out_fake['score_loss'].item():.4f}, "
                    f"Cov={out_fake['cov_loss'].item():.4f}, "
                    f"respConf={out_fake['resp_confidence_loss'].item():.4f}"
                )
                if getattr(model.module if hasattr(model, "module") else model, "prior_type", None) == "gmm":
                    diag = gmm_collapse_diagnostics(prior=prior_ref, z=out_fake["z"])
                    msg += (
                        f", piH={diag['pi_entropy']:.3f}, K_eff={diag['k_eff']:.2f}, "
                        f"activeK={diag['active_comp']}, respTop1={diag['resp_top1']:.3f}, "
                        f"minMuDist={diag['min_mu_dist']:.3f}"
                    )
                print(msg)

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, float(n_cells)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_recon, total_kl, total_score, total_contrast, total_cov, n_cells = stats.tolist()
        n_cells = max(float(n_cells), 1.0)

    return (
        total_loss / n_cells,
        total_recon / n_cells,
        total_kl / n_cells,
        total_score / n_cells,
        total_contrast / n_cells,
        total_cov / n_cells,
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
