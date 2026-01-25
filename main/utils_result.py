import numpy as np
import torch
import os
import anndata
import pandas as pd
from tqdm import tqdm
import gc  


def save_poisson_parameter_for_dataset(model, loader, sample_name, gene_ids, output_dir, device,
                                       use_amp=None, estimate_genes=False, gene_id_subset=None, lambda_adv=1.0):
    """
    Save Poisson parameter (mu) and cell type logits from model inference for a given sample.
    Works for both bio-only and sample-conditioned XVerseModel.
    """
    if use_amp is None:
        use_amp = torch.cuda.is_available()

    model.eval()
    mu_list, logit_list = [], []

    # --- Determine model mode once ---
    has_sample = getattr(model, "has_sample", False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches", leave=False):
            sample_id, value, tissue_id = batch
            sample_id = sample_id.to(device)
            value = value.to(device)
            tissue_id = tissue_id.to(device)

            with torch.amp.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=use_amp):
                if has_sample:
                    mu, _, _, _, _, _, celltype_logits = model(
                        value=value,
                        sample_id=sample_id,
                        tissue_id=tissue_id,
                        lambda_adv=lambda_adv
                    )
                    mu_out = mu
                else:
                    mu_bio, _, _, celltype_logits = model(
                        value=value,
                        tissue_id=tissue_id
                    )
                    mu_out = mu_bio

            mu_list.append(mu_out.cpu().numpy())
            logit_list.append(celltype_logits.cpu().numpy())

    # --- concatenate results ---
    mu_arr = np.concatenate(mu_list, axis=0)
    logit_arr = np.concatenate(logit_list, axis=0)

    var_df = pd.DataFrame(index=gene_ids)
    var_df["gene_ids"] = gene_ids

    # --- optional gene subset ---
    if gene_id_subset is not None:
        gene_id_subset_set = set(gene_id_subset)
        selected_genes = [gid for gid in gene_ids if gid in gene_id_subset_set]
        selected_indices = [i for i, gid in enumerate(gene_ids) if gid in gene_id_subset_set]

        mu_arr = mu_arr[:, selected_indices]
        var_df = var_df.loc[selected_genes]

    os.makedirs(output_dir, exist_ok=True)

    # --- save mu ---
    adata_mu = anndata.AnnData(X=mu_arr, var=var_df)
    adata_mu.write_h5ad(os.path.join(output_dir, f"{sample_name}_mu.h5ad"))

    # --- save logits ---
    adata_logit = anndata.AnnData(X=logit_arr)
    adata_logit.write_h5ad(os.path.join(output_dir, f"{sample_name}_logits.h5ad"))

    # --- optional Poisson sampling ---
    if estimate_genes:
        for i in range(3):
            sampled_expr = np.random.poisson(lam=mu_arr)
            adata_sampled = anndata.AnnData(X=sampled_expr, var=var_df)
            output_path = os.path.join(output_dir, f"{sample_name}_estimated_expr_{i}.h5ad")
            adata_sampled.write_h5ad(output_path)
            del sampled_expr, adata_sampled
            gc.collect()
            
            
def save_bio_embedding_for_dataset(model, loader, sample_name, gene_ids, output_dir, device, use_amp=None):
    """
    Process a single dataset (for one sample) to compute and save bio embedding.

    Args:
        model: Trained model that outputs bio_embedding
        loader: DataLoader for the subset dataset corresponding to a single sample.
                Expected batch format: (sample_id, cell_id, value)
        sample_name: The sample name (derived from the file path) used for naming output files.
        gene_ids: List of gene identifiers (not used here but retained for compatibility)
        output_dir: Output folder where the npy files will be stored.
        device: Device for inference (e.g., torch.device('cuda') or 'cpu')
        use_amp: Boolean flag for using automatic mixed precision; if None, defaults to torch.cuda.is_available()
    """
    if use_amp is None:
        use_amp = torch.cuda.is_available()
    
    model.eval()
    has_sample = getattr(model, "has_sample", False)
    embed_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches", leave=False):
            sample_ids, _, values = batch
            sample_ids = sample_ids.to(device)
            values = values.to(device)
            
            with torch.amp.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=use_amp):
                if has_sample:
                    _, _, _, bio_embedding, _, _, _ = model(
                        value=values, sample_id=sample_ids, lambda_adv=1
                    )
                    embed = bio_embedding
                else:
                    _, _, bio_embedding, _ = model(
                        value=values
                    )
                    embed = bio_embedding

            embed_list.append(embed.cpu().numpy())
    
    embedding_arr = np.concatenate(embed_list, axis=0)  # shape: (n_cells, embedding_dim)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sample_name}_bio_embedding.npy")
    np.save(output_path, embedding_arr)

    print(f"Saved bio embedding for sample {sample_name} to {output_path}")
    print(f"Embedding shape: {embedding_arr.shape}")
    
     
def save_average_attn_weight_for_dataset(model, loader, gene_ids, output_dir, device, sample_name, use_amp=None):

    if use_amp is None:
        use_amp = torch.cuda.is_available()

    model.eval()
    has_sample = getattr(model, "has_sample", False)
    attn_sum = None  # To accumulate total weight
    mask_sum = None  # To accumulate valid counts

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches", leave=False):
            sample_ids, values = batch
            sample_ids = sample_ids.to(device)
            values = values.to(device)

            with torch.amp.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=use_amp):
                if has_sample:
                    _, _, attn_weight, _, _, _, _ = model(
                        value=values, sample_id=sample_ids, lambda_adv=1
                    )
                else:
                    _, attn_weight, _, _ = model(
                        value=values
                    )

            mask = (values != -1)

            if attn_sum is None:
                N = attn_weight.shape[1]
                attn_sum = torch.zeros(N, dtype=torch.float32, device='cpu')
                mask_sum = torch.zeros(N, dtype=torch.float32, device='cpu')

            attn_sum += attn_weight.sum(dim=0).cpu()
            mask_sum += mask.sum(dim=0).cpu()

    # Final average calculation with masking
    avg_attn = attn_sum / (mask_sum + 1e-8)

    # Save to CSV
    avg_df = pd.DataFrame({'gene_id': gene_ids, 'avg_attn_weight': avg_attn.numpy()})
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sample_name}_avg_attn.csv")
    avg_df.to_csv(output_path, index=False)
    print(f"Saved average attention weights to {output_path}")
   

def build_index_to_cell_type(csv_path):
    """
    Build a reverse mapping from integer index to classification name,
    consistent with build_cell_type_to_index().
    
    Args:
        csv_path (str): Path to CSV file containing 'classification_result' column.
        
    Returns:
        dict: Mapping from integer index to classification name.
              -1 is mapped to "Uncategorized".
    """
    df = pd.read_csv(csv_path)

    # Drop missing classifications
    df = df.dropna(subset=["classification_result"])

    # Get sorted unique classification results, excluding 'Uncategorized'
    unique_classes = sorted(df['classification_result'].unique())
    unique_classes = [c for c in unique_classes if c!= "Other/Unknown"]

    index_to_classification = {idx: cls for idx, cls in enumerate(unique_classes)}
    index_to_classification[-1] = "Other/Unknown"

    return index_to_classification
