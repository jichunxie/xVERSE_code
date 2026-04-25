#!/usr/bin/env python3
import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make "main_energy" importable when running this file directly by path.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_energy.train_pantissue import _linear_warmup_scale
from main_energy.utils_model import (
    MaskFiLMGMMVAE,
    evaluate_gmm_vae_one_epoch,
    load_gene_ids,
    FastXVerseBatchDataset,
    SparseBatchCollator,
    CompiledShardDataset,
    CompiledSparseBatchCollator,
    build_pair_to_sample_id_and_paths,
    build_cell_type_to_index,
)


def _get(a, k, d):
    return getattr(a, k, d)


def _normalize_state_keys_for_model(state_dict, model_state_keys):
    """Try to align checkpoint keys with current model keys by toggling 'module.' prefix."""
    model_has_module = any(k.startswith("module.") for k in model_state_keys)
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())

    if model_has_module == ckpt_has_module:
        return state_dict, "none"

    if ckpt_has_module and not model_has_module:
        remap = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                remap[k[len("module."):]] = v
            else:
                remap[k] = v
        return remap, "strip_module"

    # not ckpt_has_module and model_has_module
    remap = {f"module.{k}": v for k, v in state_dict.items()}
    return remap, "add_module"


def _stage_and_weights(a, epoch_id: int):
    beta_end = _get(a, "beta_kl", 1.0) if _get(a, "beta_kl_end", None) is None else _get(a, "beta_kl_end", 1.0)
    beta_t = beta_end
    if _get(a, "beta_kl_warmup_epochs", 0) > 0:
        alpha = min(1.0, max(0.0, (epoch_id - 1) / float(_get(a, "beta_kl_warmup_epochs", 0))))
        beta_t = _get(a, "beta_kl_start", 0.0) + (beta_end - _get(a, "beta_kl_start", 0.0)) * alpha

    phase1_epochs = max(0, int(_get(a, "gmm_init_after_epochs", 0)))
    stage2_epochs = max(0, int(_get(a, "gmm_stage2_epochs", 0)))
    post_init_epoch = epoch_id - phase1_epochs
    in_stage1 = bool(phase1_epochs > 0 and epoch_id <= phase1_epochs)
    in_stage2 = bool(phase1_epochs > 0 and stage2_epochs > 0 and epoch_id > phase1_epochs and epoch_id <= phase1_epochs + stage2_epochs)
    stage_name = "stage1" if in_stage1 else ("stage2" if in_stage2 else "stage3")

    bal_scale = _linear_warmup_scale(epoch_id, int(_get(a, "lambda_resp_balance_warmup_epochs", 0)))
    conf_scale = _linear_warmup_scale(epoch_id, int(_get(a, "lambda_resp_confidence_warmup_epochs", 0)))
    lambda_resp_balance_t = float(_get(a, "lambda_resp_balance", 0.0)) * bal_scale
    lambda_resp_confidence_t = float(_get(a, "lambda_resp_confidence", 0.0)) * conf_scale
    lambda_resp_anchor_t = float(_get(a, "lambda_resp_anchor", 0.0))
    lambda_cov_t = float(_get(a, "lambda_cov", 0.0)) * _linear_warmup_scale(epoch_id, int(_get(a, "lambda_cov_warmup_epochs", 0)))

    resp_temp_start = float(_get(a, "resp_temperature", 1.0)) if _get(a, "resp_temperature_start", None) is None else float(_get(a, "resp_temperature_start", 1.0))
    resp_temp_end = float(_get(a, "resp_temperature", 1.0))
    temp_scale = _linear_warmup_scale(epoch_id, int(_get(a, "resp_temperature_warmup_epochs", 0)))
    resp_temperature_t = resp_temp_start + (resp_temp_end - resp_temp_start) * temp_scale

    if phase1_epochs > 0:
        if in_stage1:
            if phase1_epochs <= 1:
                beta_t = beta_end
            else:
                stage1_prog = min(1.0, max(0.0, (epoch_id - 1) / float(phase1_epochs - 1)))
                beta_t = _get(a, "beta_kl_start", 0.0) + (beta_end - _get(a, "beta_kl_start", 0.0)) * stage1_prog
            lambda_resp_balance_t = 0.0
            lambda_resp_confidence_t = 0.0
            lambda_resp_anchor_t = 0.0
            resp_temperature_t = resp_temp_start
        else:
            post_warm_kl = max(0, int(_get(a, "gmm_post_init_kl_warmup_epochs", 0)))
            if post_warm_kl > 0:
                kl_prog = min(1.0, max(0.0, (post_init_epoch - 1) / float(post_warm_kl)))
                beta_t = beta_end * kl_prog
            else:
                beta_t = beta_end
            bal_scale = _linear_warmup_scale(post_init_epoch, int(_get(a, "lambda_resp_balance_warmup_epochs", 0)))
            conf_scale = _linear_warmup_scale(post_init_epoch, int(_get(a, "lambda_resp_confidence_warmup_epochs", 0)))
            temp_scale = _linear_warmup_scale(post_init_epoch, int(_get(a, "resp_temperature_warmup_epochs", 0)))
            lambda_resp_balance_t = float(_get(a, "lambda_resp_balance", 0.0)) * bal_scale
            lambda_resp_confidence_t = float(_get(a, "lambda_resp_confidence", 0.0)) * conf_scale
            lambda_resp_anchor_t = float(_get(a, "lambda_resp_anchor", 0.0))
            resp_temperature_t = resp_temp_start + (resp_temp_end - resp_temp_start) * temp_scale

    return {
        "stage_name": stage_name,
        "force_base_posterior": in_stage1,
        "beta_t": float(beta_t),
        "lambda_cov_t": float(lambda_cov_t),
        "lambda_resp_balance_t": float(lambda_resp_balance_t),
        "lambda_resp_confidence_t": float(lambda_resp_confidence_t),
        "lambda_resp_anchor_t": float(lambda_resp_anchor_t),
        "resp_temperature_t": float(resp_temperature_t),
    }


def _print_prior_per_component(model):
    base = model.module if hasattr(model, "module") else model
    if getattr(base, "prior", None) is None:
        print("[Prior] not available")
        return
    pi = torch.softmax(base.prior.pi_logits.detach().float().cpu(), dim=0)
    mu = base.prior.prior_mu.detach().float().cpu()
    logvar = base.prior.prior_logvar.detach().float().cpu()
    dmat = torch.cdist(mu, mu, p=2)
    eye = torch.eye(dmat.size(0), dtype=torch.bool)
    dmat = dmat.masked_fill(eye, float("inf"))
    nn_dist, nn_idx = torch.min(dmat, dim=1)
    print("[Prior Per Component]")
    print("k\tpi\tmu_norm\tlogvar_mean\tlogvar_min\tlogvar_max\tnn_k\tnn_dist")
    for k in range(mu.size(0)):
        print(
            f"{k}\t{pi[k].item():.6f}\t{mu[k].norm().item():.4f}\t{logvar[k].mean().item():.4f}\t"
            f"{logvar[k].min().item():.4f}\t{logvar[k].max().item():.4f}\t{int(nn_idx[k].item())}\t{nn_dist[k].item():.4f}"
        )


def main():
    ap = argparse.ArgumentParser(description="Load checkpoint, run one validation, and print diagnostics.")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (e.g. last_model.pth).")
    ap.add_argument("--compiled-dataset-root", default=None, help="Override compiled dataset root.")
    ap.add_argument("--data-root", default=None, help="Override raw data root.")
    ap.add_argument("--summary-csv", default=None, help="Override summary csv path.")
    ap.add_argument("--gene-ids-path", default=None, help="Override gene ids path.")
    ap.add_argument("--val-batch-size", type=int, default=None, help="Override validation batch size.")
    ap.add_argument("--val-num-workers", type=int, default=4, help="Validation dataloader workers.")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved = SimpleNamespace(**ckpt.get("args", {}))
    ckpt_epoch = int(ckpt.get("epoch", 0))

    compiled_root = args.compiled_dataset_root or _get(saved, "compiled_dataset_root", None)
    total_gene = int(_get(saved, "total_gene", 17999))

    if compiled_root:
        manifest_path = f"{compiled_root}/manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        total_gene = int(manifest.get("global_num_genes", total_gene))
        val_ds = CompiledShardDataset(
            compiled_root=compiled_root,
            split="val",
            max_cached_shards=int(_get(saved, "compiled_max_cached_shards", 8)),
        )
        val_collator = CompiledSparseBatchCollator(num_genes=total_gene, apply_mask_aug=False)
        inferred_num_cell_types = val_ds.infer_num_celltypes()
    else:
        data_root = args.data_root or _get(saved, "data_root", None)
        if data_root is None:
            raise ValueError("data_root not found in checkpoint args; please pass --data-root.")
        summary_csv = args.summary_csv or _get(saved, "summary_csv", None) or f"{data_root}/pantissue_full_updated.csv"
        gene_ids_path = args.gene_ids_path or _get(saved, "gene_ids_path", None) or f"{data_root}/ensg_keys_high_quality.txt"
        pair_to_idx, _, val_pairs, pair_to_tissue_id, _ = build_pair_to_sample_id_and_paths(
            summary_csv, use_tissue=_get(saved, "use_tissue", None)
        )
        cell_type_to_index = build_cell_type_to_index(_get(saved, "cell_type_csv", "/hpc/group/xielab/xj58/sparest_code/standard_type/cellxgene_cell_type_mapped.csv"))
        gene_ids = load_gene_ids(gene_ids_path)
        total_gene = len(gene_ids)
        val_ds = FastXVerseBatchDataset(
            val_pairs,
            gene_ids,
            pair_to_idx,
            cell_type_to_index,
            pair_to_tissue_id=pair_to_tissue_id,
            filter_bad_cells=bool(_get(saved, "filter_bad_cells", False)),
            index_cache_path=_get(saved, "val_index_cache", None),
            allow_stale_index_cache=bool(_get(saved, "allow_stale_index_cache", False)),
        )
        val_collator = SparseBatchCollator(val_ds, num_genes=total_gene, apply_mask_aug=False)
        inferred_num_cell_types = val_ds.infer_num_celltypes()

    num_cell_types = int(_get(saved, "num_cell_types", 0))
    if num_cell_types <= 0:
        num_cell_types = int(inferred_num_cell_types)

    vbs = int(args.val_batch_size or _get(saved, "val_batch_size", 512))
    val_loader = DataLoader(
        val_ds,
        batch_size=vbs,
        shuffle=False,
        drop_last=False,
        collate_fn=val_collator,
        num_workers=max(0, int(args.val_num_workers)),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskFiLMGMMVAE(
        num_genes=total_gene,
        latent_dim=int(_get(saved, "latent_dim", 128)),
        num_components=int(_get(saved, "num_components", 16)),
        prior_cov_rank=int(_get(saved, "prior_cov_rank", 8)),
        posterior_cov_rank=int(_get(saved, "posterior_cov_rank", 0)),
        expr_hidden_dim=int(_get(saved, "expr_hidden_dim", 1024)),
        mask_hidden_dim=int(_get(saved, "mask_hidden_dim", 512)),
        dec_hidden_dim=int(_get(saved, "dec_hidden_dim", 1024)),
        dropout=float(_get(saved, "dropout", 0.1)),
        prior_type=str(_get(saved, "prior_type", "gmm")),
        num_cell_types=num_cell_types,
    ).to(device)
    state = ckpt["model_state_dict"]
    normalized_state, strategy = _normalize_state_keys_for_model(state, model.state_dict().keys())
    ret = model.load_state_dict(normalized_state, strict=False)
    print(
        f"[Load] ckpt={args.ckpt}, epoch={ckpt_epoch}, key_strategy={strategy}, "
        f"missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}"
    )
    if ret.missing_keys:
        print(f"[Load][Missing] sample={ret.missing_keys[:8]}")
    if ret.unexpected_keys:
        print(f"[Load][Unexpected] sample={ret.unexpected_keys[:8]}")

    sched = _stage_and_weights(saved, ckpt_epoch)
    print(
        f"[EvalCfg] stage={sched['stage_name']}, beta_kl={sched['beta_t']:.6f}, "
        f"lambda_resp_balance={sched['lambda_resp_balance_t']:.6f}, "
        f"lambda_resp_confidence={sched['lambda_resp_confidence_t']:.6f}, "
        f"resp_temperature={sched['resp_temperature_t']:.6f}"
    )

    val = evaluate_gmm_vae_one_epoch(
        model=model,
        val_loader=val_loader,
        device=device,
        beta_kl=sched["beta_t"],
        recon_observed_only=bool(_get(saved, "recon_observed_only", False)),
        lambda_score=float(_get(saved, "lambda_score", 0.0)),
        score_noise_std=float(_get(saved, "score_noise_std", 0.1)),
        score_detach_z=bool(_get(saved, "score_detach_z", True)),
        lambda_contrast=float(_get(saved, "lambda_contrast", 0.0)),
        contrast_temp=float(_get(saved, "contrast_temp", 0.1)),
        lambda_real_recon=float(_get(saved, "lambda_real_recon", 0.0)),
        lambda_cov=sched["lambda_cov_t"],
        cov_use_mu=bool(_get(saved, "cov_use_mu", True)),
        lambda_resp_balance=sched["lambda_resp_balance_t"],
        lambda_resp_confidence=sched["lambda_resp_confidence_t"],
        lambda_resp_anchor=sched["lambda_resp_anchor_t"],
        resp_temperature=sched["resp_temperature_t"],
        resp_topk=int(_get(saved, "resp_topk", 0)),
        prior_logvar_min=float(_get(saved, "prior_logvar_min", -6.0)),
        prior_logvar_max=float(_get(saved, "prior_logvar_max", 4.0)),
        lambda_prior_mu_l2=float(_get(saved, "lambda_prior_mu_l2", 0.0)),
        lambda_prior_factor_l2=float(_get(saved, "lambda_prior_factor_l2", 0.0)),
        lambda_prior_pi_balance=float(_get(saved, "lambda_prior_pi_balance", 0.0)),
        lambda_celltype_cls=float(_get(saved, "lambda_celltype_cls", 0.0)),
        force_base_posterior=bool(sched["force_base_posterior"]),
    )
    loss_full, loss_recon, loss_kl, loss_score, loss_contrast, loss_cov, loss_prior_pi_balance, loss_celltype_cls = val
    print(
        f"[Val] Loss={loss_full:.4f}, Recon={loss_recon:.4f}, KL={loss_kl:.4f}, "
        f"Score={loss_score:.4f}, Contrast={loss_contrast:.4f}, "
        f"priorPiBal={loss_prior_pi_balance:.4f}, cls={loss_celltype_cls:.4f}"
    )
    _print_prior_per_component(model)


if __name__ == "__main__":
    main()
