# Copyright (C) 2024 Xiaohui Jiang and Jichun Xie
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Sampler
import gc  
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
    

class MLPBlock(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU, dropout_rate=0.1):
        """
        Multi-layer Perceptron (MLP) block with Layernorm and Dropout.
        
        Args:
            input_dim (int): The input feature dimension.
            hidden_dims (list of int, optional): List of hidden layer dimensions. .
            output_dim (int): The output feature dimension.
            activation (nn.Module): Activation function to use (default: nn.ReLU).
            dropout_rate (float): Dropout probability (default: 0.1).
        """
        super(MLPBlock, self).__init__()
        layers = []
        in_dim = input_dim

        # Create hidden layers based on provided dimensions
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP block.
        """
        return self.block(x)

           
class CellEmbeddingbyGene(nn.Module):
    """
    Cell encoder that pools shared gene embeddings with learned per-gene attention.
    Adds FiLM conditioning (mask pattern -> gamma/beta) instead of additive mask embedding.
    - values: [B, N], -1 = unmeasured, >=0 observed
    - gene_emb is inherited from the outer model via set_gene_emb()
    - masks: list of one-hot numpy arrays (0/1) that define spatial panels
    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_genes: int = 23045,
                 temp: float = 1.0, masks: Optional[List] = None,
                 num_tissues: int = 1):
        super().__init__()
        self.temp = float(temp)
        self.num_genes = int(num_genes)
        self.gene_emb: Optional[torch.Tensor] = None

        # Tissue embedding table (required)
        self.tissue_emb = nn.Embedding(num_tissues, num_genes)

        # Per-gene weight network: [value, mask_bit, tissue_bias] -> scalar
        self.value_net = MLPBlock(input_dim=3, output_dim=1,
                                  hidden_dims=[hidden_dim, hidden_dim])

        # Mixer along gene dimension
        self.mlp_block = MLPBlock(input_dim=num_genes, output_dim=num_genes,
                                  hidden_dims=[2*hidden_dim, 2*hidden_dim])

        # FiLM layers for mask embedding (panel effect)
        self.film_gamma = MLPBlock(input_dim=num_genes, output_dim=embed_dim,
                                   hidden_dims=[hidden_dim])
        self.film_beta  = MLPBlock(input_dim=num_genes, output_dim=embed_dim,
                                   hidden_dims=[hidden_dim])

        # Final projection
        self.output_proj = MLPBlock(input_dim=embed_dim, output_dim=hidden_dim,
                                    hidden_dims=[hidden_dim, hidden_dim])

        # Pre-convert panel masks
        self.masks: List[torch.Tensor] = []
        if masks is not None and len(masks) > 0:
            for m in masks:
                t = torch.as_tensor(m, dtype=torch.bool)
                if t.numel() != self.num_genes:
                    raise ValueError(f"Panel mask length {t.numel()} != num_genes={self.num_genes}")
                self.masks.append(t)

    def set_gene_emb(self, gene_emb: torch.Tensor):
        """Register a reference to shared gene embeddings [N, E]."""
        if gene_emb.dim() != 2:
            raise ValueError("gene_emb must be 2D [N, E].")
        if gene_emb.size(0) != self.num_genes:
            raise ValueError(f"gene_emb.size(0)={gene_emb.size(0)} != num_genes={self.num_genes}")
        self.gene_emb = gene_emb

    @torch.no_grad()
    def _apply_random_mask(self, values: torch.Tensor) -> torch.Tensor:
        """Apply stochastic masking or panel masking."""
        value_modified = values.clone()
        if not self.training:
            return value_modified

        B, N = value_modified.shape
        if N != self.num_genes:
            raise ValueError(f"values N={N} must equal num_genes={self.num_genes}")

        for i in range(B):
            detected_idx = (value_modified[i] != -1).nonzero(as_tuple=False).squeeze()
            num_detected = int(detected_idx.numel())
            if num_detected == 0:
                continue

            if num_detected < 1000:
                # Mask between 10 and ~1/5 of detected genes
                max_to_mask = max(5, int(num_detected * (1.0 / 5.0)))
                num_to_mask = random.randint(10, max_to_mask) if num_detected > 10 else num_detected
                if num_to_mask > 0:
                    selected = detected_idx[torch.randperm(num_detected,
                                                           device=value_modified.device)[:num_to_mask]]
                    value_modified[i, selected] = -1
            else:
                # 10% chance to apply a predefined panel
                use_panel = (random.random() < 0.1) and (len(self.masks) > 0)
                if use_panel:
                    panel_mask_cpu = random.choice(self.masks)             # bool tensor on CPU
                    panel_mask = panel_mask_cpu.to(value_modified.device)  # move to device
                    value_modified[i, ~panel_mask] = -1

                    if random.random() < 0.8:
                        # --- Case A: panel mask + drop 10–100 ---
                        kept_idx = (panel_mask & (value_modified[i] != -1)).nonzero(as_tuple=False).squeeze()
                        num_kept = int(kept_idx.numel())
                        if num_kept > 50:
                            num_extra_mask = random.randint(10, 100)
                            num_extra_mask = min(num_extra_mask, num_kept)
                            selected = kept_idx[torch.randperm(num_kept, device=value_modified.device)[:num_extra_mask]]
                            value_modified[i, selected] = -1
                    else:
                        # --- Case B: panel mask + add 50 extra detected genes ---
                        detected_idx_full = (values[i] != -1).nonzero(as_tuple=False).squeeze()
                        num_to_add = min(random.randint(50, 3000), int(detected_idx_full.numel()))
                        if num_to_add > 0:
                            add_back = detected_idx_full[torch.randperm(detected_idx_full.numel(), device=value_modified.device)[:num_to_add]]
                            value_modified[i, add_back] = values[i, add_back]
                else:
                    p = random.random()
                    if p < 0.2:
                        frac = random.uniform(0.1, 0.3)  # Mask 10–30%
                    elif p < 0.9:
                        frac = random.uniform(0.3, 0.5)  # Mask 30–50%
                    else:
                        frac = random.uniform(0.5, 0.75) # Mask 50–75%

                    num_to_mask = int(num_detected * frac)
                    if num_to_mask > 0:
                        selected = detected_idx[torch.randperm(num_detected,
                                                               device=value_modified.device)[:num_to_mask]]
                        value_modified[i, selected] = -1
        return value_modified

    @staticmethod
    def _build_mask(value_tensor: torch.Tensor) -> torch.Tensor:
        """Return float mask in {0,1}: 1 if measured (>=0), 0 if -1."""
        return (value_tensor != -1).float()

    def forward(self, values: torch.Tensor, tissue_id: torch.Tensor, weight_only: bool = False):
        if self.gene_emb is None:
            raise RuntimeError("gene_emb is not set. Call set_gene_emb().")

        # 1) stochastic masking
        value_modified = self._apply_random_mask(values)

        # 2) build mask and cleaned values
        mask = self._build_mask(value_modified)
        values_cleaned = torch.where(value_modified == -1,
                                     torch.zeros_like(value_modified),
                                     value_modified)
        values_cleaned = torch.log1p(values_cleaned)

        # 3) tissue bias per gene
        tissue_bias = self.tissue_emb(tissue_id)              # [B, N]
        value_input = torch.stack([values_cleaned, mask], dim=-1)  # [B, N, 2]
        value_input = torch.cat([value_input, tissue_bias.unsqueeze(-1)], dim=-1)  # [B, N, 3]

        # 4) compute raw weights
        raw_weight = self.value_net(value_input).squeeze(-1)           # [B, N]

        # 5) mixer + softmax
        processed_weight = self.mlp_block(raw_weight).unsqueeze(-1)
        masked_weight = processed_weight.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = torch.softmax(masked_weight / self.temp, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        if weight_only:
            return attn_weights, mask

        # 6) weighted pooling
        B, N = values.shape
        gene_emb = self.gene_emb
        gene_emb_expanded = gene_emb.unsqueeze(0).expand(B, -1, -1)
        weighted_emb = gene_emb_expanded * attn_weights
        cell_emb = weighted_emb.sum(dim=1)   # [B, E]

        # 7) FiLM conditioning with mask
        gamma = torch.tanh(self.film_gamma(mask)) + 1.0   # [B, E]
        beta  = self.film_beta(mask)                      # [B, E]
        cell_emb = gamma * cell_emb + beta

        # 8) projection
        cell_emb = self.output_proj(cell_emb)
        return cell_emb, attn_weights


class XVerseModel(nn.Module):
    def __init__(self, num_samples=None, hidden_dim=256,
                 total_gene=17999, eps=1e-8, masks=None,
                 num_tissues=64, total_cell_type=29):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.total_gene = total_gene
        self.has_sample = num_samples is not None

        # Embeddings
        self.gene_embedding = nn.Embedding(total_gene, hidden_dim)
        if self.has_sample:
            self.sample_emb = nn.Embedding(num_samples, hidden_dim)

        # Tissue-specific bias
        self.tissue_gene_bias = nn.Embedding(num_tissues, total_gene)

        # Encoder
        self.bio_encoder = CellEmbeddingbyGene(
            embed_dim=hidden_dim, hidden_dim=hidden_dim,
            num_genes=total_gene, masks=masks, num_tissues=num_tissues
        )
        self.bio_encoder.set_gene_emb(self.gene_embedding.weight)

        # FiLM layers (only if sample info exists)
        if self.has_sample:
            self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
            self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        # Decoder and heads
        self.gene_decoder = MLPBlock(input_dim=2 * hidden_dim, output_dim=1,
                                     hidden_dims=[hidden_dim, hidden_dim])
        self.size_factor = MLPBlock(input_dim=hidden_dim, output_dim=1,
                                    hidden_dims=[hidden_dim, hidden_dim])
        if self.has_sample:
            self.sample_classifier_bio = MLPBlock(input_dim=hidden_dim, output_dim=num_samples,
                                                  hidden_dims=[hidden_dim, hidden_dim])
        self.cell_type_head = MLPBlock(input_dim=hidden_dim, output_dim=total_cell_type,
                                       hidden_dims=[hidden_dim, hidden_dim])

    def forward(self, value, tissue_id, sample_id=None, lambda_adv=1.0):
        # Encode biological signal
        z_bio, gene_weight = self.bio_encoder(value, tissue_id)

        # Always compute bio outputs
        mu_bio_logits = self.compute_mu(z_bio, tissue_id)
        size_factor_bio = F.softplus(self.size_factor(z_bio))
        mu_bio = torch.clamp(F.softmax(mu_bio_logits, dim=-1), min=self.eps) * size_factor_bio
        celltype_logits = self.cell_type_head(z_bio)

        # If no sample branch, return bio-only
        if not self.has_sample:
            return mu_bio, gene_weight.squeeze(-1), z_bio, celltype_logits

        # Otherwise use sample conditioning
        s = self.sample_emb(sample_id)
        gamma = torch.tanh(self.film_gamma(s)) + 1.0
        beta = self.film_beta(s)
        z_dec = gamma * z_bio + beta

        mu_logits = self.compute_mu(z_dec, tissue_id)
        size_factor = F.softplus(self.size_factor(z_dec))
        mu = torch.clamp(F.softmax(mu_logits, dim=-1), min=self.eps) * size_factor

        z_bio_grl = GradientReversal.apply(z_bio, lambda_adv)
        sample_logits_bio = self.sample_classifier_bio(z_bio_grl)

        return mu, mu_bio, gene_weight.squeeze(-1), z_bio, z_dec, sample_logits_bio, celltype_logits

    def compute_mu(self, z, tissue_id):
        """Compute gene-level logits = decoder(z, gene_emb) + tissue_bias"""
        gene_emb = self.gene_embedding.weight  # [G, d]
        B, G, d = z.size(0), gene_emb.size(0), gene_emb.size(1)

        z_expand = z.unsqueeze(1).expand(-1, G, -1)
        gene_expand = gene_emb.unsqueeze(0).expand(B, -1, -1)
        mu_input = torch.cat([z_expand, gene_expand], dim=-1)

        mu_logits = self.gene_decoder(mu_input).squeeze(-1)
        bias = self.tissue_gene_bias(tissue_id)
        mu_logits = mu_logits + bias
        return mu_logits  


class PoissonLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, mu, y_true):
        """
        Computes the Poisson loss (negative log-likelihood).

        Args:
            mu (Tensor): Predicted Poisson mean (λ).
            y_true (Tensor): Observed count values.

        Returns:
            Tensor: Computed Poisson loss.
        """
        # Clamp mu to avoid log(0)
        mu = torch.clamp(mu, min=self.epsilon)
        loss = mu - y_true * torch.log(mu)
        return loss.mean()
    
      
class ConditionalBatchLoss(nn.Module):
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, sample_id: torch.Tensor, cell_type_id: torch.Tensor) -> torch.Tensor:
        """
        Balanced cross-entropy loss for batch (sample) classification, stratified by cell type.
        Args:
            logits: [B, n_samples], output of classifier
            sample_id: [B], int batch/sample id in [0, n_samples-1]
            cell_type_id: [B], int cell type
        Returns:
            Scalar loss (averaged by class)
        """
        valid_mask = cell_type_id != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits = logits[valid_mask]
        sample_id = sample_id[valid_mask]
        cell_type_id = cell_type_id[valid_mask]

        # Ensure sample_id are in valid range!
        assert (sample_id >= 0).all(), "sample_id contains negative"
        assert (sample_id < logits.shape[1]).all(), f"sample_id exceeds classifier classes {logits.shape[1]}"

        unique_cell_types = torch.unique(cell_type_id)
        total_loss = 0.0
        count = 0

        for ct in unique_cell_types:
            ct_mask = (cell_type_id == ct)
            ct_indices = ct_mask.nonzero(as_tuple=True)[0]
            if ct_indices.numel() <= 1:
                continue
            ct_sample_id = sample_id[ct_indices]
            ct_logits = logits[ct_indices]

            sid_list, sid_counts = torch.unique(ct_sample_id, return_counts=True)
            if sid_list.numel() < 2:
                continue
            median_n = int(sid_counts.float().median().floor().item())
            if median_n < 1:
                continue

            sampled_indices = []
            for sid in sid_list:
                sid_idx = (ct_sample_id == sid).nonzero(as_tuple=True)[0]
                if sid_idx.numel() == 0:
                    continue
                if sid_idx.numel() >= median_n:
                    rand_idx = torch.randperm(sid_idx.numel(), device=logits.device)[:median_n]
                else:
                    # Upsample with replacement
                    rand_idx = torch.randint(sid_idx.numel(), (median_n,), device=logits.device)
                selected = ct_indices[sid_idx[rand_idx]]
                sampled_indices.append(selected)
            if len(sampled_indices) < 2:
                continue

            balanced_idx = torch.cat(sampled_indices, dim=0)
            loss_ct = self.ce_loss(logits[balanced_idx], sample_id[balanced_idx])
            total_loss += loss_ct
            count += 1

        if count == 0:
            return torch.tensor(1e-8, device=logits.device, requires_grad=True)
        return total_loss / count
   
   
def pretrain_one_epoch(
    model,
    optimizer,
    scaler,
    train_loader,
    device,
    lambda_adv=1.0,
    lambda_ct=1.0,
    lambda_recon_bio=1
):
    """
    Single epoch of model pretraining with all loss components.

    Returns: 
        Tuple of averaged losses:
        (poisson_loss, poisson_loss_bio, sample_loss, celltype_loss)
    """
    # Define loss functions
    poisson_loss_fn = PoissonLoss()
    sample_loss_fn = ConditionalBatchLoss(ignore_index=-1)
    celltype_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)

    model.train()

    # Initialize loss accumulators
    total_poisson_loss, total_poisson_loss_bio = 0.0, 0.0
    total_sample_loss, total_cell_type_loss = 0.0, 0.0
    n_samples = 0

    for batch_idx, (sample_id, tissue_id, cell_type_id, value) in enumerate(train_loader):
        optimizer.zero_grad()
        sample_id = sample_id.to(device, non_blocking=True)
        tissue_id = tissue_id.to(device, non_blocking=True)
        cell_type_id = cell_type_id.to(device, non_blocking=True)
        value = value.to(device, non_blocking=True)
        batch_n = value.size(0)
        n_samples += batch_n

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            mu, mu_bio, _, _, _, sample_logits_bio, celltype_logits = model(
                value=value, sample_id=sample_id, tissue_id=tissue_id, lambda_adv=lambda_adv
            )
            mask = (value != -1)

            # Losses
            loss_recon_poisson = poisson_loss_fn(mu[mask], value[mask])
            loss_recon_poisson_bio_only = poisson_loss_fn(mu_bio[mask], value[mask])
            sample_loss = sample_loss_fn(sample_logits_bio, sample_id, cell_type_id)     
            cell_type_loss = celltype_loss_fn(celltype_logits, cell_type_id)

            loss = (
                loss_recon_poisson +
                lambda_recon_bio * loss_recon_poisson_bio_only +
                lambda_adv * sample_loss +
                lambda_ct * cell_type_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate
        total_poisson_loss += loss_recon_poisson.item() * batch_n
        total_poisson_loss_bio += loss_recon_poisson_bio_only.item() * batch_n
        total_sample_loss += sample_loss.item() * batch_n
        total_cell_type_loss += cell_type_loss.item() * batch_n

        if (batch_idx + 1) % 100 == 0:
            print(
                f"[Batch {batch_idx + 1}] "
                f"Poisson: {loss_recon_poisson.item():.4f}, "
                f"PoissonBio: {loss_recon_poisson_bio_only.item():.4f}, "
                f"Sample: {sample_loss.item():.4f}, "
                f"CellType: {cell_type_loss.item():.4f}"
            )
            gc.collect()
            torch.cuda.empty_cache()

        del mu, mu_bio, sample_logits_bio, celltype_logits
        del loss_recon_poisson, loss_recon_poisson_bio_only, sample_loss, cell_type_loss, loss
    return (
        total_poisson_loss / n_samples,
        total_poisson_loss_bio / n_samples,
        total_sample_loss / n_samples,
        total_cell_type_loss / n_samples
    )


def evaluate_one_epoch(
    model,
    val_loader,
    device,
    lambda_adv=1.0
):
    poisson_loss_fn = PoissonLoss()
    sample_loss_fn = ConditionalBatchLoss(ignore_index=-1)
    celltype_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)

    model.eval()
    total_poisson_loss, total_poisson_loss_bio = 0.0, 0.0
    total_sample_loss, total_cell_type_loss = 0.0, 0.0
    n_samples, n_valid_ct = 0, 0

    with torch.no_grad():
        for batch_idx, (sample_id, tissue_id, cell_type_id, value) in enumerate(val_loader):
            sample_id = sample_id.to(device, non_blocking=True)
            tissue_id = tissue_id.to(device, non_blocking=True)
            cell_type_id = cell_type_id.to(device, non_blocking=True)
            value = value.to(device, non_blocking=True)
            batch_size = value.size(0)
            n_samples += batch_size

            with torch.amp.autocast(device_type='cuda', enabled=False):
                mu, mu_bio, _, _, _, sample_logits_bio, celltype_logits = model(
                    value=value, sample_id=sample_id, tissue_id=tissue_id, lambda_adv=lambda_adv
                )

                mask = (value != -1)
                loss_recon_poisson = poisson_loss_fn(mu[mask], value[mask])
                loss_recon_poisson_bio_only = poisson_loss_fn(mu_bio[mask], value[mask])
                sample_loss = sample_loss_fn(sample_logits_bio, sample_id, cell_type_id)

                valid_mask = (cell_type_id != -1)
                if valid_mask.any():
                    cell_type_loss = celltype_loss_fn(celltype_logits[valid_mask], cell_type_id[valid_mask])
                    n_valid_ct += int(valid_mask.sum().item())
                else:
                    cell_type_loss = torch.tensor(0.0, device=device)

            total_poisson_loss += loss_recon_poisson.item() * batch_size
            total_poisson_loss_bio += loss_recon_poisson_bio_only.item() * batch_size
            total_sample_loss += sample_loss.item() * batch_size
            total_cell_type_loss += cell_type_loss.item() * (int(valid_mask.sum().item()) if valid_mask.any() else 0)

    avg_cell_type_loss = total_cell_type_loss / n_valid_ct if n_valid_ct > 0 else 0.0

    return (
        total_poisson_loss / n_samples,
        total_poisson_loss_bio / n_samples,
        total_sample_loss / n_samples,
        avg_cell_type_loss
    )
    
    
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
        """
        Args:
            matrix_meta_pairs: list of (matrix_path, meta_path) pairs.
            gene_ids: list of gene names, union across all blocks.
            pair_to_sample_id: dict[(matrix_path, meta_path)] -> int sample_id.
            cell_type_to_index: dict cell type string -> int index.
            use_cache: whether to cache loaded rows.
            available_genes: optional, restrict to this gene set.
            io_workers: threads for parallel I/O.
            pair_to_tissue_id: dict[(matrix_path, meta_path)] -> int tissue_id.
        Returns in __getitem__:
            sample_id (LongTensor)
            tissue_id (LongTensor)
            celltype_index (LongTensor)
            result (FloatTensor)  # -1 for unmeasured genes
        """
        self.gene_ids = gene_ids
        self.pair_to_sample_id = pair_to_sample_id
        self.pair_to_tissue_id = pair_to_tissue_id
        self.cell_type_to_index = cell_type_to_index
        self.use_cache = use_cache
        self.available_genes = set(available_genes) if available_genes is not None else None

        self.X_blocks = []
        # (block_idx, local_idx, sample_id, tissue_id)
        self.index_map = []
        self.gene_idx_maps = []
        self.cell_types = []
        self.cache = {} if use_cache else None

        # Parallel loading of all matrix/meta blocks
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
            # Build mapping from global gene IDs to local indices for this block
            gene_to_idx = {g: i for i, g in enumerate(gene_ids_available)}
            gene_idx_array = np.full(len(self.gene_ids), -1, dtype=np.int32)
            for global_idx, gene in enumerate(self.gene_ids):
                if gene in gene_to_idx:
                    if self.available_genes is None or gene in self.available_genes:
                        gene_idx_array[global_idx] = gene_to_idx[gene]

            self.X_blocks.append(X)
            self.gene_idx_maps.append(gene_idx_array)
            self.cell_types.append(cell_type_array)

            num_genes_in_block = X.shape[1]
            pair = (matrix_path, meta_path)

            sample_id = int(self.pair_to_sample_id.get(pair, -1))
            assert sample_id >= 0, f"Sample ID not found for {pair}!"

            tissue_id = int(self.pair_to_tissue_id.get(pair, -1))
            # Allow -1 if not provided, but keep as integers

            for i in range(X.shape[0]):
                row = X.getrow(i)
                max_count = row.max()
                sum_count = row.sum()
                # Basic QC: skip outlier cells
                if (max_count > 1000) or (sum_count > 200000):
                    continue
                self.index_map.append((block_idx, i, sample_id, tissue_id))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Returns:
            sample_id (LongTensor)
            tissue_id (LongTensor)
            celltype_index (LongTensor)
            result (FloatTensor)  # -1 for unmeasured genes
        """
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
    
    def get_spatial_panel(self):
        """
        Return a deduplicated list of one-hot gene masks for blocks with fewer than 1000 valid genes.
        Each mask is a binary vector (1 if the gene is present in that panel).
        """
        unique_panels = set()
        panel_list = []
        for gene_idx_array in self.gene_idx_maps:
            valid_mask = gene_idx_array >= 0
            num_valid_genes = valid_mask.sum()
            if num_valid_genes < 1000:
                one_hot_mask = np.zeros_like(gene_idx_array, dtype=np.uint8)
                one_hot_mask[valid_mask] = 1
                # Convert to tuple for deduplication
                mask_tuple = tuple(one_hot_mask.tolist())
                if mask_tuple not in unique_panels:
                    unique_panels.add(mask_tuple)
                    panel_list.append(one_hot_mask)
        print(f"Number of Spatial Panels: {int(len(panel_list))}")
        return panel_list

   
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
                # With replacement if not enough samples
                selected = random.choices(candidates, k=self.samples_per_id)
            indices.extend(selected)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_id * len(self.sample_ids)


def build_cell_type_to_index(csv_path):
    """
    Build a mapping from cell type ontology ID ('id') to an integer index
    based on the 'classification_result' column.
    Assign -1 to 'Uncategorized' and 'Non-specific or artificial cell'.
    """
    df = pd.read_csv(csv_path)

    # Drop rows where classification_result is NaN
    df = df.dropna(subset=["classification_result"])

    # Define the set of classes considered as uncategorized
    uncategorized_set = {"Other/Unknown"}

    # Get all unique classification results excluding the uncategorized ones
    filtered_classes = sorted([
        c for c in df["classification_result"].unique()
        if c not in uncategorized_set
    ])
    print(f"Number of valid cell types (excluding 'Other/Unknown'): {len(filtered_classes)}")

    # Assign a unique index to each valid classification
    classification_to_index = {cls: idx for idx, cls in enumerate(filtered_classes)}

    # Assign -1 to each uncategorized class
    for cls in uncategorized_set:
        classification_to_index[cls] = -1

    # Build the mapping from 'id' to classification index
    id_to_index = {
        row["id"]: classification_to_index.get(row["classification_result"], -1)
        for _, row in df.iterrows()
    }

    return id_to_index


def build_pair_to_sample_id_and_paths(csv_path, allowed_sample_ids=None, use_tissue=None):
    """
    Return mappings for (matrix_path, obs_path) to sample_id, tissue_id, tech_id.
    Use the integer ids already present in the CSV.

    Returns:
        pair_to_sample_id: dict[(matrix_path, obs_path)] = int sample_id
        train_pairs, valid_pairs, test_pairs: lists of pairs
        pair_to_tissue_id: dict[(matrix_path, obs_path)] = int tissue_id
        pair_to_tech_id: dict[(matrix_path, obs_path)] = int tech_id
    """
    df = pd.read_csv(csv_path)

    required_cols = {'sample_id','matrix_paths','obs_paths','split','tissue_id','tech_id'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if allowed_sample_ids is not None:
        df = df[df['sample_id'].isin(allowed_sample_ids)]

    df['split'] = df['split'].astype(str).str.strip().str.lower()

    # --- Tissue filter (always exact match, supports str or list) ---
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
            elif split in ('val','valid','validation'):
                valid_pairs.append(pair)

    print(f"Total filtered pairs: {len(pair_to_sample_id)}")
    print(f"Train: {len(train_pairs)}, Val: {len(valid_pairs)}")

    return (pair_to_sample_id,
            train_pairs, valid_pairs,
            pair_to_tissue_id, pair_to_tech_id)
