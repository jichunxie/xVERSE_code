# -*- coding: utf-8 -*-
"""
Simplified XVerseModel fine-tuning pipeline (no cell typing heads)
Focus: reconstruction with optional sample-wise modulation
"""

import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
import scanpy as sc
from scipy.sparse import issparse
from collections import defaultdict
import anndata
import pandas as pd
from scipy import sparse

# =========================
# 0. Reproducibility
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Imports from environment
# =========================
from main.utils_model import (
    load_gene_ids,
    XVerseModel,
    PoissonLoss,
)

# =========================
# 3. Fine-tune model (no cell typing)
# =========================
class XVerseFineTuneModel(nn.Module):
    def __init__(self, base_model: XVerseModel, num_samples_finetune: int = None):
        super().__init__()
        self.hidden_dim = base_model.hidden_dim
        self.eps = base_model.eps
        self.total_gene = base_model.total_gene
        self.gene_embedding = base_model.gene_embedding
        self.tissue_gene_bias = base_model.tissue_gene_bias
        self.bio_encoder = base_model.bio_encoder
        self.gene_decoder = base_model.gene_decoder
        self.size_factor = base_model.size_factor
        self.use_sample_specific = num_samples_finetune is not None

        if self.use_sample_specific:
            # Fine-tune–specific components
            self.sample_emb_ft = nn.Embedding(num_samples_finetune, self.hidden_dim)
            self.film_gamma_ft = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.film_beta_ft = nn.Linear(self.hidden_dim, self.hidden_dim)

    def freeze_base_model(self):
        """
        Freeze all base model parameters (from pretrained model).
        Only keep sample-specific fine-tuning parameters trainable.
        """
        # Freeze all base model components
        for param in self.gene_embedding.parameters():
            param.requires_grad = False
        for param in self.tissue_gene_bias.parameters():
            param.requires_grad = False
        for param in self.bio_encoder.parameters():
            param.requires_grad = False
        for param in self.gene_decoder.parameters():
            param.requires_grad = False
        for param in self.size_factor.parameters():
            param.requires_grad = False
        
        # Keep sample-specific parameters trainable (if they exist)
        if self.use_sample_specific:
            for param in self.sample_emb_ft.parameters():
                param.requires_grad = True
            for param in self.film_gamma_ft.parameters():
                param.requires_grad = True
            for param in self.film_beta_ft.parameters():
                param.requires_grad = True
        
        print("Base model frozen. Only sample-specific parameters are trainable.")
        
        # Print trainable parameter count
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Trainable: {trainable_params:,} / {total_params:,} parameters ({100*trainable_params/total_params:.2f}%)")

    def compute_mu_logits(self, z, tissue_id):
        gene_emb = self.gene_embedding.weight
        B, G, d = z.size(0), gene_emb.size(0), gene_emb.size(1)
        z_expand = z.unsqueeze(1).expand(-1, G, -1)
        gene_expand = gene_emb.unsqueeze(0).expand(B, -1, -1)
        mu_input = torch.cat([z_expand, gene_expand], dim=-1)
        mu_logits = self.gene_decoder(mu_input).squeeze(-1)
        bias = self.tissue_gene_bias(tissue_id)
        return mu_logits + bias

    def forward(self, value, tissue_id, sample_id=None):

        # Base biological encoding
        z_bio, gene_weight = self.bio_encoder(value, tissue_id)
        mu_bio_logits = self.compute_mu_logits(z_bio, tissue_id)
        size_factor_bio = F.softplus(self.size_factor(z_bio))
        mu_bio = torch.clamp(F.softmax(mu_bio_logits, dim=-1), min=self.eps) * size_factor_bio

        # ----- Determine whether "mu" should be returned -----
        if not self.use_sample_specific:
            suppress_mu_output = True
            sample_ft_ready = False

        else:
            # sample-specific is enabled
            if sample_id is None:
                suppress_mu_output = True
                sample_ft_ready = False
            else:
                if isinstance(sample_id, torch.Tensor):
                    sample_ft_ready = bool(torch.all(sample_id >= 0).item())
                else:
                    sample_ft_ready = (sample_id >= 0)

                suppress_mu_output = not sample_ft_ready

        # ----- Compute z_dec (FT modulation) -----
        if sample_ft_ready:
            s_ft = self.sample_emb_ft(sample_id)
            gamma_ft = torch.tanh(self.film_gamma_ft(s_ft)) + 1.0
            beta_ft = self.film_beta_ft(s_ft)
            z_dec = gamma_ft * z_bio + beta_ft
        else:
            z_dec = z_bio

        # ----- Compute final mu -----
        mu_logits = self.compute_mu_logits(z_dec, tissue_id)
        size_factor = F.softplus(self.size_factor(z_dec))
        mu = torch.clamp(F.softmax(mu_logits, dim=-1), min=self.eps) * size_factor

        # ----- Final outputs -----
        outputs = {
            "mu_bio": mu_bio,
            "z_bio": z_bio,
            "gene_weight": gene_weight.squeeze(-1),
        }

        if not suppress_mu_output:
            outputs["mu"] = mu

        return outputs 

# =========================
# 5. Dataset (no cluster labels)
# =========================
class XVerseFineTuneDataset(Dataset):
    """Return (sample_id, value, tissue_id)."""
    def __init__(self, file_sample_dict, gene_ids, tissue_map,
                 visible_gene_ids=None, use_qc=False):
        self.file_sample_dict = file_sample_dict
        self.ensg_list = gene_ids
        self.visible_gene_set = set(visible_gene_ids) if visible_gene_ids else set(gene_ids)
        self.tissue_map = tissue_map
        self.use_qc = use_qc
        self.index = []
        self._build_index()
        self.loaded_files = {}

    def _read_adata(self, fp):
        return sc.read_h5ad(fp) if fp.endswith(".h5ad") else sc.read_10x_h5(fp)

    def _build_index(self):
        for fp, sid in self.file_sample_dict.items():
            adata = self._read_adata(fp)
            X = adata.raw.X if adata.raw is not None else adata.X
            X = X.tocsr() if issparse(X) else X

            n = X.shape[0]
            print(f"[Index] {os.path.basename(fp)} cells: {n}")

            if self.use_qc:
                num_genes_in_block = X.shape[1]
                kept = 0
                for i in range(n):
                    row = X.getrow(i) if issparse(X) else X[i, :]
                    max_count = row.max() if issparse(X) else np.max(row)
                    sum_count = row.sum() if issparse(X) else np.sum(row)
                    nnz = row.getnnz() if issparse(X) else np.count_nonzero(row)

                    # Basic QC filtering
                    if max_count > 2000:  # very high expression spike
                        continue
                    if sum_count > num_genes_in_block * 5:  # total count too high
                        continue
                    
                    self.index.append((fp, sid, i))
                    kept += 1

                print(f"[QC] kept {kept}/{n} cells after basic count control")
            else:
                for i in range(n):
                    self.index.append((fp, sid, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, sample_id, cell_idx = self.index[idx]
        if file_path not in self.loaded_files:
            self.loaded_files[file_path] = self._read_adata(file_path)
        adata = self.loaded_files[file_path]

        gene_to_idx = {g: i for i, g in enumerate(adata.var["gene_ids"])}
        n_genes = len(self.ensg_list)
        dtype = adata.X.dtype if not issparse(adata.X) else np.float32
        row_full = np.full((n_genes,), -1, dtype=dtype)

        valid_pos, valid_gene_idx = [], []
        for pos, g in enumerate(self.ensg_list):
            if g in gene_to_idx and g in self.visible_gene_set:
                valid_pos.append(pos)
                valid_gene_idx.append(gene_to_idx[g])

        raw = adata.raw.X if adata.raw is not None else adata.X
        row = raw.getrow(cell_idx).toarray()[0] if issparse(raw) else np.array(raw[cell_idx, :]).flatten()
        row_full[valid_pos] = np.asarray(row[valid_gene_idx], dtype=np.float32)
        value = torch.tensor(row_full, dtype=torch.float32)

        tissue_value = adata.obs["tissue"].iloc[cell_idx]
        if tissue_value not in self.tissue_map:
            raise KeyError(f"Tissue '{tissue_value}' not in tissue_map.")
        tissue_tensor = torch.tensor(self.tissue_map[tissue_value], dtype=torch.long)
        sample_tensor = torch.tensor(sample_id if sample_id is not None else -1, dtype=torch.long)
        return sample_tensor, value, tissue_tensor


def finetune_one_epoch(model, optimizer, scaler, train_loader, device,
                       lambda_recon_bio=1.0):
    poisson_loss_fn = PoissonLoss()
    model.train()
    total_loss = total_poisson = total_poisson_bio = 0.0
    n = 0
    n_mu = 0

    for batch_idx, (sample_id, value, tissue_id) in enumerate(tqdm(train_loader, desc="Fine-tune")):
        optimizer.zero_grad(set_to_none=True)
        value = value.to(device)
        tissue_id = tissue_id.to(device)
        sample_id = sample_id.to(device)
        mask = (value != -1)
        bsz = value.size(0)
        n += bsz

        with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            outputs = model(value=value, tissue_id=tissue_id, sample_id=sample_id)
            mu = outputs.get("mu")
            mu_bio = outputs["mu_bio"]
            loss_poisson_bio = poisson_loss_fn(mu_bio[mask], value[mask])
            loss = lambda_recon_bio * loss_poisson_bio

            if mu is not None:
                loss_poisson = poisson_loss_fn(mu[mask], value[mask])
                loss = loss + loss_poisson
            else:
                loss_poisson = None

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * bsz
        total_poisson_bio += loss_poisson_bio.item() * bsz
        if loss_poisson is not None:
            total_poisson += loss_poisson.item() * bsz
            n_mu += bsz

        if (batch_idx + 1) % 50 == 0:
            msg = f"[Batch {batch_idx+1}] PoissonBio={loss_poisson_bio.item():.4f}"
            if loss_poisson is not None:
                msg = (f"[Batch {batch_idx+1}] Poisson={loss_poisson.item():.4f} | "
                       f"PoissonBio={loss_poisson_bio.item():.4f}")
            print(msg)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del outputs, mu, mu_bio, loss, loss_poisson, loss_poisson_bio

    results = {
        "loss_total": total_loss / n,
        "loss_poisson_bio": total_poisson_bio / n,
    }
    if n_mu > 0:
        results["loss_poisson"] = total_poisson / n_mu

    return results
    
def evaluate_one_epoch(model, val_loader, device, use_amp=True):
    """
    Evaluate XVerseFineTuneModel on validation data.
    Returns average losses and concatenated mu/mu_bio arrays.
    """
    poisson_loss_fn = PoissonLoss()
    model.eval()

    total_loss = total_poisson = total_poisson_bio = 0.0
    n = 0
    n_mu = 0

    with torch.no_grad():
        for batch_idx, (sample_id, value, tissue_id) in enumerate(tqdm(val_loader, desc="Evaluate")):
            value = value.to(device)
            tissue_id = tissue_id.to(device)
            sample_id = sample_id.to(device)
            mask = (value != -1)
            bsz = value.size(0)
            n += bsz

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(value=value, tissue_id=tissue_id, sample_id=sample_id)
                mu = outputs.get("mu")
                mu_bio = outputs["mu_bio"]
                loss_poisson_bio = poisson_loss_fn(mu_bio[mask], value[mask])
                loss = loss_poisson_bio

                if mu is not None:
                    loss_poisson = poisson_loss_fn(mu[mask], value[mask])
                    loss = loss + loss_poisson
                else:
                    loss_poisson = None

            total_loss += loss.item() * bsz
            total_poisson_bio += loss_poisson_bio.item() * bsz
            if loss_poisson is not None:
                total_poisson += loss_poisson.item() * bsz
                n_mu += bsz

            del outputs, mu, mu_bio, loss, loss_poisson, loss_poisson_bio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results = {
        "loss_total": total_loss / n,
        "loss_poisson_bio": total_poisson_bio / n,
    }
    if n_mu > 0:
        results["loss_poisson"] = total_poisson / n_mu

    return results
    
def save_poisson_parameter_ft(
    model,
    loader,
    sample_name,
    gene_ids,
    output_dir,
    device,
    use_amp=None,
    estimate_genes=False,
    gene_id_subset=None,
    save_panel_genes_only=True,
    num_estimation=3,
    ensembl_to_name=None  
):
    """
    Save Poisson parameter (mu, when available) outputs from XVerseFineTuneModel inference.

    Args:
        model: `XVerseFineTuneModel` (on `device`) used for inference.
        loader: DataLoader that yields `(sample_id, value, tissue_id)` or `(value, tissue_id)` batches.
        sample_name: Prefix used for the saved `.h5ad`/`.csv` files.
        gene_ids: Ordered list of Ensembl IDs matching the model output dimension.
        output_dir: Destination directory where all artifacts are written.
        device: Torch device on which inference happens.
        use_amp: Optional bool to enable autocast; defaults to CUDA availability.
        estimate_genes: Whether to generate Poisson-sampled expression replicates.
        gene_id_subset: Optional iterable of gene IDs to keep before saving.
        save_panel_genes_only: If True, keep only genes observed (>=0) in the first batch.
        num_estimation: Number of Poisson replicate matrices to sample when `estimate_genes` is True.
        ensembl_to_name: Optional dict mapping Ensembl IDs to gene symbols for annotation.

    Notes:
        Adds a `gene_name` column (when mapping provided) and uses that as the AnnData var index.
    """
    import gc
    import anndata
    from scipy import sparse

    if use_amp is None:
        use_amp = torch.cuda.is_available()

    model.eval()
    mu_list, mu_bio_list = [], []
    mu_available = None
    has_sample = getattr(model, "use_sample_specific", False)
    panel_idx = None  # indices of genes to keep

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Processing batches", leave=False)):
            if len(batch) == 3:
                sample_id, value, tissue_id = batch
                sample_id = sample_id.to(device)
            elif len(batch) == 2:
                value, tissue_id = batch
                sample_id = None
            else:
                raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")

            if not has_sample:
                sample_id = None
            elif sample_id is None:
                raise ValueError("Model expects sample-specific inputs but batch lacks sample_id.")

            value = value.to(device)
            tissue_id = tissue_id.to(device)

            # --- record first-batch gene mask if required ---
            if save_panel_genes_only and i == 0:
                value_np = value.detach().cpu().numpy()
                panel_mask = (value_np >= 0).any(axis=0)
                panel_idx = np.where(panel_mask)[0]
                print(f"Panel genes retained: {len(panel_idx)} / {value_np.shape[1]}")

            with torch.amp.autocast(
                device_type=device.type if hasattr(device, "type") else "cuda",
                enabled=use_amp
            ):
                if has_sample:
                    outputs = model(value=value, tissue_id=tissue_id, sample_id=sample_id)
                else:
                    outputs = model(value=value, tissue_id=tissue_id)

            mu_tensor = outputs.get("mu")
            batch_has_mu = mu_tensor is not None
            if mu_available is None:
                mu_available = batch_has_mu
            elif mu_available != batch_has_mu:
                raise RuntimeError("Inconsistent presence of 'mu' across batches.")
            if batch_has_mu:
                mu = mu_tensor.detach().cpu().numpy()
                mu_list.append(mu)

            mu_bio = outputs["mu_bio"].detach().cpu().numpy()
            mu_bio_list.append(mu_bio)


    # --- concatenate results ---
    if mu_available:
        mu_arr = np.concatenate(mu_list, axis=0)
    else:
        mu_arr = None
    mu_bio_arr = np.concatenate(mu_bio_list, axis=0)

    # --- prepare var DataFrame ---
    var_df = pd.DataFrame(index=gene_ids)
    var_df["gene_ids"] = gene_ids

    # --- optional subset from argument ---
    if gene_id_subset is not None:
        subset = set(gene_id_subset)
        selected = [i for i, g in enumerate(gene_ids) if g in subset]
        gene_ids = [gene_ids[i] for i in selected]
        if mu_arr is not None:
            mu_arr = mu_arr[:, selected]
        mu_bio_arr = mu_bio_arr[:, selected]
        var_df = var_df.loc[gene_ids]

    # --- optional subset from panel genes ---
    if save_panel_genes_only and panel_idx is not None:
        if mu_arr is not None:
            mu_arr = mu_arr[:, panel_idx]
        mu_bio_arr = mu_bio_arr[:, panel_idx]
        gene_ids = [gene_ids[i] for i in panel_idx]
        var_df = pd.DataFrame(index=gene_ids)
        var_df["gene_ids"] = gene_ids

    # --- map Ensembl ID to gene name ---
    if ensembl_to_name is not None:
        var_df["gene_name"] = [ensembl_to_name.get(gid, gid) for gid in var_df["gene_ids"]]
        var_df.index = pd.Index(
            var_df["gene_name"].astype(str) + "_" + var_df["gene_ids"].astype(str),
            name="gene_index"
        )
        
    var_df["gene_ids"] = var_df["gene_ids"].astype(str)
    os.makedirs(output_dir, exist_ok=True)

    # --- save mu and mu_bio ---
    if mu_arr is not None:
        adata_mu = anndata.AnnData(X=mu_arr, var=var_df)
        adata_mu.write_h5ad(os.path.join(output_dir, f"{sample_name}_mu.h5ad"))

    adata_mu_bio = anndata.AnnData(X=mu_bio_arr, var=var_df)
    adata_mu_bio.write_h5ad(os.path.join(output_dir, f"{sample_name}_mu_bio.h5ad"))

    # --- optional Poisson sampling ---
    if estimate_genes and num_estimation > 0:
        lam_source = mu_arr if (mu_arr is not None) else mu_bio_arr
        sampled_blocks = []
        obs_names = []

        for copy_idx in range(1, num_estimation + 1):
            sampled_expr = np.random.poisson(lam=lam_source)
            sampled_blocks.append(sampled_expr)
            obs_names.extend([f"{cell_idx + 1}_{copy_idx}" for cell_idx in range(sampled_expr.shape[0])])

        sampled_expr_concat = np.concatenate(sampled_blocks, axis=0)
        sampled_expr_sparse = sparse.csr_matrix(sampled_expr_concat, dtype=np.float32)
        obs_df = pd.DataFrame(index=pd.Index(obs_names, name="cell_id"))
        adata_sampled = anndata.AnnData(X=sampled_expr_sparse, var=var_df, obs=obs_df)

        out_path = os.path.join(output_dir, f"{sample_name}_estimated_expr_poisson.h5ad")
        adata_sampled.write_h5ad(out_path)

        del sampled_blocks, sampled_expr_concat, sampled_expr_sparse, adata_sampled
        gc.collect()
            

def save_gene_weight_ft(
    model,
    loader,
    sample_name,
    gene_ids,
    output_dir,
    device,
    use_amp=None,
    gene_id_subset=None,
    save_panel_genes_only=False,
    ensembl_to_name=None,
    save_mean=True,
):

    if use_amp is None:
        use_amp = torch.cuda.is_available()

    model.eval()
    has_sample = getattr(model, "use_sample_specific", False)

    gw_list = []          # list of per-batch gene_weight arrays
    panel_idx = None      # indices of genes to keep if save_panel_genes_only

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Processing batches", leave=False)):
            if has_sample:
                sample_id, value, tissue_id = batch
                sample_id = sample_id.to(device)
            else:
                value, tissue_id = batch
                sample_id = None

            value = value.to(device)
            tissue_id = tissue_id.to(device)

            # Determine panel genes from first batch if requested
            if save_panel_genes_only and i == 0:
                value_np = value.detach().cpu().numpy()
                panel_mask = (value_np >= 0).any(axis=0)
                panel_idx = np.where(panel_mask)[0]
                print(f"Panel genes retained: {len(panel_idx)} / {value_np.shape[1]}")

            with torch.amp.autocast(
                device_type=device.type if hasattr(device, "type") else "cuda",
                enabled=use_amp
            ):
                if has_sample and sample_id is not None:
                    outputs = model(value=value, tissue_id=tissue_id, sample_id=sample_id)
                else:
                    outputs = model(value=value, tissue_id=tissue_id)

            gene_weight = outputs["gene_weight"].detach().cpu().numpy()  # (B, G)
            gw_list.append(gene_weight)

    # Concatenate across all batches -> (N_cells_total, G)
    gw_arr = np.concatenate(gw_list, axis=0)

    # Build var dataframe
    var_df = pd.DataFrame(index=gene_ids)
    var_df["gene_ids"] = list(gene_ids)

    # Subset by explicit gene_id_subset first
    if gene_id_subset is not None:
        subset = set(gene_id_subset)
        selected = [i for i, g in enumerate(gene_ids) if g in subset]
        gene_ids = [gene_ids[i] for i in selected]
        gw_arr = gw_arr[:, selected]
        var_df = var_df.loc[gene_ids]

    # Subset by detected panel_idx second
    if save_panel_genes_only and panel_idx is not None:
        gw_arr = gw_arr[:, panel_idx]
        gene_ids = [gene_ids[i] for i in panel_idx]
        var_df = pd.DataFrame(index=gene_ids)
        var_df["gene_ids"] = gene_ids

    # Map Ensembl -> gene_name if provided
    if ensembl_to_name is not None:
        var_df["gene_name"] = [
            ensembl_to_name.get(gid, gid) for gid in var_df["gene_ids"]
        ]
        # make combined index "SYMBOL_ENSEMBL" to avoid duplicates
        var_df.index = pd.Index(
            var_df["gene_name"].astype(str) + "_" + var_df["gene_ids"].astype(str),
            name="gene_index"
        )

    # Ensure gene_ids is string
    var_df["gene_ids"] = var_df["gene_ids"].astype(str)

    os.makedirs(output_dir, exist_ok=True)

    # 1) Save per-cell gene_weight as AnnData
    adata_gw = anndata.AnnData(X=gw_arr, var=var_df)
    adata_gw.write_h5ad(os.path.join(output_dir, f"{sample_name}_gene_weight.h5ad"))

    # 2) Save per-gene mean as CSV (one row)
    if save_mean:
        mean_vec = gw_arr.mean(axis=0, dtype=np.float64)  # shape (G,)
        # Build a DataFrame with one row. Columns = var_df.index, which is gene_name_gene_id if available
        col_index = var_df.index.to_list()
        df_mean = pd.DataFrame([mean_vec], columns=col_index)
        csv_path = os.path.join(output_dir, f"{sample_name}_gene_weight_mean.csv")
        df_mean.to_csv(csv_path, index=False)

    # cleanup
    del gw_list, gw_arr
    gc.collect()


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, samples_per_dataset=1000):
        self.dataset = dataset
        self.samples_per_dataset = samples_per_dataset
        
        # Handle Subset
        if isinstance(dataset, torch.utils.data.Subset):
            parent_dataset = dataset.dataset
            subset_indices = dataset.indices
        else:
            parent_dataset = dataset
            subset_indices = range(len(dataset))
            
        # Group indices by file path
        self.indices_by_file = defaultdict(list)
        
        # We need to iterate over the *subset's* indices (0 to len(dataset)-1)
        # and find which file they belong to in the parent dataset.
        for i in range(len(dataset)):
            if isinstance(dataset, torch.utils.data.Subset):
                # The actual index in the parent dataset
                parent_idx = dataset.indices[i]
            else:
                parent_idx = i
                
            # Look up file path using the parent index
            # parent_dataset.index is a list of (fp, sid, cell_idx)
            fp = parent_dataset.index[parent_idx][0]
            
            # Store the SUBSET index 'i', not the parent index
            self.indices_by_file[fp].append(i)
            
    def __iter__(self):
        final_indices = []
        for fp, indices in self.indices_by_file.items():
            if len(indices) == 0:
                continue
                
            if len(indices) >= self.samples_per_dataset:
                sampled = np.random.choice(indices, self.samples_per_dataset, replace=False)
            else:
                # If fewer than target, sample with replacement to match target size
                # This ensures every dataset contributes exactly samples_per_dataset cells
                sampled = np.random.choice(indices, self.samples_per_dataset, replace=True)
            
            final_indices.extend(sampled)
            
        np.random.shuffle(final_indices)
        return iter(final_indices)

    def __len__(self):
        return len(self.indices_by_file) * self.samples_per_dataset

