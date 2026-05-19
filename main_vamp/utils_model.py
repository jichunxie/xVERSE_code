import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

import main_mfa.utils_model as base
from main_mfa.utils_model import *  # re-export datasets, samplers, train/eval loops, and helpers


class PseudoCellVampPrior(nn.Module):
    """
    VampPrior / pseudo-cell prior:
      p(z) = sum_k pi_k q_encoder(z | u_k)
    where u_k is a learnable pseudo expression profile and q_encoder is the same
    mask-FiLM encoder used by real cells. This keeps prior prototypes on the
    encoder manifold instead of learning free Gaussian centers.
    """

    def __init__(self, num_components: int, num_genes: int, pseudo_init: str = "log_normal"):
        super().__init__()
        self.K = int(num_components)
        self.num_genes = int(num_genes)
        self.pi_logits = nn.Parameter(torch.zeros(self.K))
        init = str(pseudo_init).lower()
        if init == "zero":
            raw = torch.zeros(self.K, self.num_genes)
        else:
            # raw pseudo expression is mapped through softplus before encoding.
            raw = torch.randn(self.K, self.num_genes) * 0.02
        self.raw_pseudo_expr = nn.Parameter(raw)

    def pseudo_expr(self) -> torch.Tensor:
        return F.softplus(self.raw_pseudo_expr)

    def component_params(
        self,
        encoder: nn.Module,
        dtype: torch.dtype = None,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
    ):
        pseudo = self.pseudo_expr()
        mask = torch.ones_like(pseudo)
        mu, logvar = encoder(pseudo, mask)
        logvar = torch.clamp(logvar, min=float(logvar_min), max=float(logvar_max))
        if dtype is not None:
            mu = mu.to(dtype=dtype)
            logvar = logvar.to(dtype=dtype)
        return mu, logvar


class MaskFiLMGMMVAE(nn.Module):
    def __init__(
        self,
        num_genes: int,
        latent_dim: int = 64,
        num_components: int = 64,
        expr_hidden_dim: int = 1024,
        mask_hidden_dim: int = 512,
        dec_hidden_dim: int = 1024,
        dropout: float = 0.1,
        prior_type: str = "vamp",
        prior_cov_rank: int = 0,
        prior_shared_covariance: bool = False,
        posterior_cov_rank: int = 0,
        prior_mu_init: str = "normal",
        prior_mu_init_radius: float = 1.0,
        prior_mu_init_groups: int = 8,
        prior_mu_init_local_radius: float = 0.5,
        num_cell_types: int = 0,
        conditional_prior_on_tissue: bool = False,
        num_tissues: int = 0,
        num_batches: int = 0,
        batch_emb_dim: int = 0,
        batch_cond_drop_prob: float = 0.0,
        recon_loss_type: str = "poisson",
    ):
        super().__init__()
        if prior_type not in ("vamp", "gaussian"):
            raise ValueError(f"Unsupported prior_type for main_vamp: {prior_type}")
        self.prior_type = str(prior_type)
        self.encoder = base.FiLMMaskEncoder(
            num_genes=num_genes,
            latent_dim=latent_dim,
            expr_hidden_dim=expr_hidden_dim,
            mask_hidden_dim=mask_hidden_dim,
            dropout=dropout,
        )
        self.prior = PseudoCellVampPrior(num_components=num_components, num_genes=num_genes)
        self.num_components = int(num_components)
        self.latent_dim = int(latent_dim)
        self.num_batches = max(0, int(num_batches))
        self.batch_emb_dim = max(0, int(batch_emb_dim))
        self.batch_cond_drop_prob = max(0.0, min(1.0, float(batch_cond_drop_prob)))
        if self.num_batches > 0 and self.batch_emb_dim > 0:
            self.batch_embedding = nn.Embedding(self.num_batches, self.batch_emb_dim)
        else:
            self.batch_embedding = None
        self.decoder = base.PoissonDecoder(
            latent_dim=latent_dim,
            num_genes=num_genes,
            hidden_dim=dec_hidden_dim,
            dropout=dropout,
            cond_dim=self.batch_emb_dim if self.batch_embedding is not None else 0,
        )
        self.nb_log_theta = nn.Parameter(torch.zeros(num_genes))
        self.recon_loss_type = str(recon_loss_type).lower()
        if self.recon_loss_type not in ("poisson", "nb"):
            raise ValueError(f"Unsupported recon_loss_type: {self.recon_loss_type}")
        self.prior_logvar_min = -6.0
        self.prior_logvar_max = 2.0
        self.register_buffer("recon_gene_mean_ema", torch.zeros(num_genes), persistent=True)
        self.register_buffer("recon_gene_sq_mean_ema", torch.zeros(num_genes), persistent=True)
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

    def _build_recon_gene_weight(self, x_count, x_mask, mode, alpha, ema_momentum, w_min, w_max, eps):
        mode = str(mode).lower()
        if mode == "none" or float(alpha) <= 0:
            return None
        with torch.no_grad():
            x = base._safe_count_tensor(x_count)
            m = base._safe_mask_tensor(x_mask)
            denom = torch.clamp(m.sum(dim=0), min=1.0)
            mean = (x * m).sum(dim=0) / denom
            sq_mean = ((x * x) * m).sum(dim=0) / denom
            mom = float(ema_momentum)
            self.recon_gene_mean_ema.mul_(mom).add_(mean.detach(), alpha=1.0 - mom)
            self.recon_gene_sq_mean_ema.mul_(mom).add_(sq_mean.detach(), alpha=1.0 - mom)
        mean = torch.clamp(self.recon_gene_mean_ema, min=float(eps))
        if mode == "cv_ema":
            var = torch.clamp(self.recon_gene_sq_mean_ema - self.recon_gene_mean_ema.pow(2), min=0.0)
            raw = torch.sqrt(var + float(eps)) / mean
        elif mode == "inv_log1p_mean_ema":
            raw = 1.0 / torch.log1p(mean)
        else:
            return None
        raw = raw / torch.clamp(raw.mean(), min=float(eps))
        raw = torch.clamp(raw, min=float(w_min), max=float(w_max))
        return (1.0 - float(alpha)) + float(alpha) * raw

    def _build_recon_cell_weight(self, q_c, z, mode, alpha, w_min, w_max, eps, num_clusters, kmeans_iters):
        mode = str(mode).lower()
        if mode == "none" or float(alpha) <= 0:
            return None
        if mode != "batch_kmeans" or z.size(0) < 2:
            return None
        with torch.no_grad():
            x = z.detach().float()
            k = max(1, min(int(num_clusters), x.size(0)))
            centers = x[torch.randperm(x.size(0), device=x.device)[:k]].clone()
            assign = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
            for _ in range(max(1, int(kmeans_iters))):
                assign = torch.argmin(torch.cdist(x, centers), dim=1)
                new = torch.zeros_like(centers)
                counts = torch.bincount(assign, minlength=k).float().to(x.device)
                new.index_add_(0, assign, x)
                ok = counts > 0
                new[ok] = new[ok] / counts[ok].unsqueeze(1)
                new[~ok] = centers[~ok]
                centers = new
            counts = torch.bincount(assign, minlength=k).float().to(x.device)
            freq = counts[assign] / torch.clamp(counts.sum(), min=1.0)
            w = 1.0 / torch.clamp(freq, min=float(eps))
            w = w / torch.clamp(w.mean(), min=float(eps))
            w = torch.clamp(w, min=float(w_min), max=float(w_max))
            return ((1.0 - float(alpha)) + float(alpha) * w).to(dtype=z.dtype)

    def forward(self, x_count, x_mask, tissue_id=None, sample_id=None, x_expr=None, force_base_posterior=False, use_batch_condition=True):
        if x_expr is None:
            x_expr = torch.log1p(x_count.float())
        mu, logvar, h = self.encoder(x_expr=x_expr, x_mask=x_mask.float(), return_hidden=True)
        logvar = torch.clamp(logvar, min=-8.0, max=8.0)
        z = base.reparameterize(mu, logvar)
        batch_cond = self._batch_condition(sample_id=sample_id, use_batch_condition=use_batch_condition)
        gene_logits = self.decoder(z, cond=batch_cond)
        library_size = F.softplus(self.library_head(z)) + 1e-8
        gene_probs = F.softmax(gene_logits, dim=-1)
        rate = gene_probs * library_size
        nb_theta = F.softplus(self.nb_log_theta).view(1, -1) + 1e-8
        out = {
            "mu": mu,
            "mu_base": mu,
            "logvar": logvar,
            "z": z,
            "encoder_hidden": h,
            "library_size": library_size,
            "gene_logits": gene_logits,
            "rate": rate,
            "nb_theta": nb_theta,
        }
        if self.celltype_head is not None:
            out["celltype_logits"] = self.celltype_head(z)
        if self.prior_type == "vamp":
            p_mu, p_logvar = self.prior.component_params(
                self.encoder,
                dtype=mu.dtype,
                logvar_min=self.prior_logvar_min,
                logvar_max=self.prior_logvar_max,
            )
            log_r = F.log_softmax(self.prior.pi_logits.float(), dim=0).to(dtype=mu.dtype) + base.gaussian_log_prob_diag(
                z=z.unsqueeze(1),
                mu=p_mu.unsqueeze(0),
                logvar=p_logvar.unsqueeze(0),
            )
            q_c = torch.softmax(log_r, dim=-1)
            out.update({"q_c": q_c, "prior_mu": p_mu, "prior_logvar": p_logvar})
        return out

    def prototype_nb_params(
        self,
        component_ids: torch.Tensor = None,
        sample_z: bool = False,
        n_samples_per_component: int = 1,
        use_batch_condition: bool = False,
        sample_id: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode pseudo-cell prior components into NB parameters.

        By default this uses z = E[q(z|u_k)] for each pseudo-cell prototype.
        If sample_z=True, it samples n_samples_per_component latent z values
        from q(z|u_k), so callers can generate prototype-neighborhood cells.
        """
        if self.prior_type != "vamp":
            raise RuntimeError("prototype_nb_params is only defined for --prior-type vamp.")
        p_mu, p_logvar = self.prior.component_params(
            self.encoder,
            logvar_min=self.prior_logvar_min,
            logvar_max=self.prior_logvar_max,
        )
        pseudo_expr = self.prior.pseudo_expr()
        pi = torch.softmax(self.prior.pi_logits.float(), dim=0).to(device=p_mu.device, dtype=p_mu.dtype)
        if component_ids is None:
            idx = torch.arange(p_mu.size(0), device=p_mu.device)
        else:
            idx = component_ids.to(device=p_mu.device, dtype=torch.long).view(-1)
        p_mu = p_mu.index_select(0, idx)
        p_logvar = p_logvar.index_select(0, idx)
        pseudo_expr = pseudo_expr.index_select(0, idx)
        pi = pi.index_select(0, idx)

        n = max(1, int(n_samples_per_component))
        if sample_z:
            std = torch.exp(0.5 * p_logvar)
            z = p_mu.unsqueeze(1) + torch.randn(
                p_mu.size(0), n, p_mu.size(1), device=p_mu.device, dtype=p_mu.dtype
            ) * std.unsqueeze(1)
        else:
            z = p_mu.unsqueeze(1).expand(-1, n, -1)
        z_flat = z.reshape(-1, z.size(-1))

        if sample_id is not None:
            sid = sample_id.to(device=p_mu.device, dtype=torch.long).view(-1)
            if sid.numel() == 1:
                sid = sid.expand(z_flat.size(0))
            elif sid.numel() == p_mu.size(0):
                sid = sid.repeat_interleave(n)
            elif sid.numel() != z_flat.size(0):
                raise ValueError(
                    "sample_id must have length 1, num_components, or num_components*n_samples_per_component; "
                    f"got {sid.numel()} for {p_mu.size(0)} components and n={n}."
                )
        else:
            sid = None
        cond = self._batch_condition(sample_id=sid, use_batch_condition=use_batch_condition)

        gene_logits = self.decoder(z_flat, cond=cond)
        library_size = F.softplus(self.library_head(z_flat)) + 1e-8
        gene_probs = F.softmax(gene_logits, dim=-1)
        rate = gene_probs * library_size
        theta = F.softplus(self.nb_log_theta).view(1, -1) + 1e-8
        theta = theta.expand_as(rate)

        k = p_mu.size(0)
        g = rate.size(-1)
        return {
            "component_ids": idx,
            "pi": pi,
            "pseudo_expr": pseudo_expr,
            "mu": p_mu,
            "logvar": p_logvar,
            "z": z,
            "rate": rate.view(k, n, g),
            "theta": theta.view(k, n, g),
            "library_size": library_size.view(k, n, 1),
            "gene_logits": gene_logits.view(k, n, g),
        }

    def sample_from_prototypes(
        self,
        component_ids: torch.Tensor = None,
        n_samples_per_component: int = 1,
        sample_z: bool = True,
        use_batch_condition: bool = False,
        sample_id: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample count profiles from prototype NB distributions.

        Returns counts with shape [K_selected, n_samples_per_component, num_genes].
        """
        params = self.prototype_nb_params(
            component_ids=component_ids,
            sample_z=sample_z,
            n_samples_per_component=n_samples_per_component,
            use_batch_condition=use_batch_condition,
            sample_id=sample_id,
        )
        mu = torch.clamp(params["rate"].float(), min=base.RECON_RATE_MIN, max=base.RECON_RATE_MAX)
        theta = torch.clamp(params["theta"].float(), min=base.RECON_THETA_MIN, max=base.RECON_THETA_MAX)
        probs = torch.clamp(theta / (theta + mu), min=1e-8, max=1.0 - 1e-8)
        counts = torch.distributions.NegativeBinomial(total_count=theta, probs=probs).sample()
        params["counts"] = counts.to(device=mu.device)
        return params

    def loss(self, x_count, x_mask, tissue_id=None, sample_id=None, celltype_id=None, force_base_posterior=False,
             beta=1.0, beta_u_kl_multiplier=1.0, beta_eps_kl_multiplier=1.0, encoder_mask=None,
             use_batch_condition=True, recon_mask=None, lambda_score=0.0, score_noise_std=0.1,
             score_detach_z=True, lambda_cov=0.0, cov_use_mu=True, lambda_resp_balance=0.0,
             lambda_resp_confidence=0.0, lambda_resp_anchor=0.0, resp_temperature=1.0, resp_topk=0,
            prior_logvar_min=-6.0, prior_logvar_max=4.0, lambda_prior_mu_l2=0.0,
             lambda_prior_factor_l2=0.0, lambda_prior_pi_balance=0.0, lambda_prior_mu_spread=0.0,
             prior_mu_spread_tau=1.0, lambda_post_c_balance=0.0, lambda_celltype_cls=0.0,
             lambda_prior_logvar_l2=0.0, prior_logvar_target=-2.0, recon_gene_weight_mode="none",
             kl_robust_mode="none", kl_robust_cap=0.0,
             recon_gene_weight_alpha=0.0, recon_gene_weight_ema_momentum=0.99, recon_gene_weight_min=0.3,
             recon_gene_weight_max=3.0, recon_gene_weight_eps=1e-6, recon_cell_weight_mode="none",
             recon_cell_weight_alpha=0.0, recon_cell_weight_min=0.5, recon_cell_weight_max=2.0,
             recon_cell_weight_eps=1e-6, recon_cell_weight_clusters=32, recon_cell_weight_kmeans_iters=2):
        if encoder_mask is None:
            encoder_mask = x_mask
        out = self.forward(x_count=x_count, x_mask=encoder_mask, tissue_id=tissue_id, sample_id=sample_id,
                           force_base_posterior=force_base_posterior, use_batch_condition=use_batch_condition)
        mu, logvar, z = out["mu"], out["logvar"], out["z"]
        rate, nb_theta = out["rate"], out["nb_theta"]
        x_count_safe = base._safe_count_tensor(x_count)
        valid_cell_mask = base._valid_cell_mask(x_count_safe, z.device)

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
        cell_weight = self._build_recon_cell_weight(
            q_c=out.get("q_c"), z=z, mode=recon_cell_weight_mode, alpha=recon_cell_weight_alpha,
            w_min=recon_cell_weight_min, w_max=recon_cell_weight_max, eps=recon_cell_weight_eps,
            num_clusters=recon_cell_weight_clusters, kmeans_iters=recon_cell_weight_kmeans_iters,
        )
        if self.recon_loss_type == "nb":
            recon_loss = base.nb_nll_masked(x_count, rate, recon_mask, nb_theta, gene_weight, cell_weight) if recon_mask is not None else base.nb_nll(x_count, rate, nb_theta, gene_weight, cell_weight)
        else:
            recon_loss = base.poisson_nll_masked(x_count, rate, recon_mask, gene_weight, cell_weight) if recon_mask is not None else base.poisson_nll(x_count, rate, gene_weight, cell_weight)

        log_q = base.gaussian_log_prob_diag(z=z, mu=mu, logvar=logvar)
        if self.prior_type == "gaussian" or force_base_posterior:
            zero = torch.zeros_like(z)
            log_p = base.gaussian_log_prob_diag(z=z, mu=zero, logvar=zero)
        else:
            self.prior_logvar_min = float(prior_logvar_min)
            self.prior_logvar_max = float(prior_logvar_max)
            p_mu, p_logvar = self.prior.component_params(
                self.encoder,
                dtype=z.dtype,
                logvar_min=self.prior_logvar_min,
                logvar_max=self.prior_logvar_max,
            )
            comp_logp = base.gaussian_log_prob_diag(z=z.unsqueeze(1), mu=p_mu.unsqueeze(0), logvar=p_logvar.unsqueeze(0))
            log_pi = F.log_softmax(self.prior.pi_logits.float(), dim=0).to(dtype=z.dtype)
            log_p = torch.logsumexp(comp_logp + log_pi.view(1, -1), dim=1)
        kl_per_cell_raw = (log_q - log_p).to(z.dtype)
        kl_per_cell = kl_per_cell_raw
        mode = str(kl_robust_mode).lower()
        cap = float(kl_robust_cap)
        if mode != "none" and cap > 0:
            # Negative Monte-Carlo KL estimates can happen for individual samples.
            # Robust mode uses only the positive KL tail so KL cannot become a
            # reward term, while OOD cells still cannot dominate linearly.
            pos = torch.clamp(kl_per_cell_raw, min=0.0)
            if mode == "clip":
                pos = torch.clamp(pos, max=cap)
            elif mode == "log1p":
                pos = cap * torch.log1p(pos / cap)
            else:
                raise ValueError(f"Unsupported kl_robust_mode: {kl_robust_mode}")
            kl_per_cell = pos
        kl_loss = base._valid_cell_mean(kl_per_cell, x_count_safe)
        kl_raw_loss = base._valid_cell_mean(kl_per_cell_raw, x_count_safe)

        zero_scalar = torch.zeros((), device=z.device, dtype=z.dtype)
        resp_entropy = zero_scalar
        resp_top1 = zero_scalar
        if "q_c" in out:
            q_c = out["q_c"]
            resp_entropy = (-(q_c * torch.log(torch.clamp(q_c, min=1e-12))).sum(dim=1)).mean().to(z.dtype)
            resp_top1 = q_c.max(dim=1).values.mean().to(z.dtype)
        celltype_cls_loss = zero_scalar
        if lambda_celltype_cls > 0 and celltype_id is not None and self.celltype_head is not None:
            ct = celltype_id.view(-1).to(z.device).long()
            valid = (ct >= 0) & (ct < self.num_cell_types) & valid_cell_mask
            if valid.any():
                celltype_cls_loss = F.cross_entropy(out["celltype_logits"][valid], ct[valid], label_smoothing=0.05).to(z.dtype)

        prior_pi_balance_loss = zero_scalar
        if self.prior_type == "vamp" and lambda_prior_pi_balance > 0:
            pi = torch.softmax(self.prior.pi_logits.float(), dim=0)
            target = torch.full_like(pi, 1.0 / float(pi.numel()))
            prior_pi_balance_loss = F.kl_div(torch.log(torch.clamp(pi, min=1e-12)), target, reduction="sum").to(z.dtype)

        total_loss = recon_loss
        if float(beta) != 0.0:
            total_loss = total_loss + float(beta) * kl_loss
        if float(lambda_celltype_cls) != 0.0:
            total_loss = total_loss + float(lambda_celltype_cls) * celltype_cls_loss
        if float(lambda_prior_pi_balance) != 0.0:
            total_loss = total_loss + float(lambda_prior_pi_balance) * prior_pi_balance_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_raw_loss": kl_raw_loss,
            "kl_c_loss": zero_scalar,
            "kl_u_loss": zero_scalar,
            "kl_eps_loss": kl_loss,
            "score_loss": zero_scalar,
            "cov_loss": zero_scalar,
            "cov_offdiag_post": zero_scalar,
            "cov_offdiag_prior": zero_scalar,
            "resp_entropy": resp_entropy,
            "resp_top1": resp_top1,
            "resp_balance_loss": zero_scalar,
            "resp_confidence_loss": zero_scalar,
            "resp_anchor_loss": zero_scalar,
            "prior_mu_l2_loss": zero_scalar,
            "prior_factor_l2_loss": zero_scalar,
            "prior_pi_balance_loss": prior_pi_balance_loss,
            "prior_mu_spread_loss": zero_scalar,
            "post_c_balance_loss": zero_scalar,
            "celltype_cls_loss": celltype_cls_loss,
            "prior_logvar_l2_loss": zero_scalar,
            "score_norm_pred": zero_scalar,
            "score_norm_tgt": zero_scalar,
            "log_q_mean": base._valid_cell_mean(log_q.to(z.dtype), x_count_safe),
            "log_p_mean": base._valid_cell_mean(log_p.to(z.dtype), x_count_safe),
            "valid_cell_mask": valid_cell_mask,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "rate": rate,
            "nb_theta": nb_theta,
        }


def prior_parameter_snapshot(model) -> Dict[str, torch.Tensor]:
    base_model = model.module if hasattr(model, "module") else model
    prior = getattr(base_model, "prior", None)
    if prior is None:
        return {}
    snap = {}
    for key in ("raw_pseudo_expr", "pi_logits"):
        val = getattr(prior, key, None)
        if torch.is_tensor(val):
            snap[key] = val.detach().float().cpu().clone()
    return snap


def prior_parameter_delta(model, snapshot: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
    if not snapshot:
        return {}
    base_model = model.module if hasattr(model, "module") else model
    prior = getattr(base_model, "prior", None)
    if prior is None:
        return {}
    out = {}
    for key, old in snapshot.items():
        val = getattr(prior, key, None)
        if not torch.is_tensor(val):
            continue
        cur = val.detach().float().cpu()
        if tuple(cur.shape) != tuple(old.shape):
            continue
        diff = (cur - old).reshape(-1)
        if diff.numel() == 0:
            continue
        out[f"{key}_mean"] = float(diff.abs().mean().item())
        out[f"{key}_max"] = float(diff.abs().max().item())
    return out


def format_prior_delta(delta: Dict[str, float]) -> str:
    if not delta:
        return "priorDelta=NA"
    parts = []
    for prefix, label in (("raw_pseudo_expr", "pseudo"), ("pi_logits", "piLogit")):
        mean_key = f"{prefix}_mean"
        max_key = f"{prefix}_max"
        if mean_key in delta:
            parts.append(f"{label}={delta[mean_key]:.3g}/{delta[max_key]:.3g}")
    return "priorDelta[" + ", ".join(parts) + "]" if parts else "priorDelta=NA"
