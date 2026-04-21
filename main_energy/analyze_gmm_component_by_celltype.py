import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from main_energy.utils_model import (
    CompiledShardDataset,
    CompiledSparseBatchCollator,
    MaskFiLMGMMVAE,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Load a trained checkpoint and print dominant GMM component for each cell type."
    )
    p.add_argument("--ckpt-path", required=True, help="Path to checkpoint file, e.g. best_model.pth")
    p.add_argument("--compiled-dataset-root", required=True, help="Compiled dataset root (xverse_train_v1)")
    p.add_argument("--split", choices=["train", "val", "both"], default="val")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None, help="Optional limit for quick analysis.")
    p.add_argument(
        "--cell-type-csv",
        default=None,
        help="Optional CSV used to map celltype index to readable class name (classification_result).",
    )
    p.add_argument("--device", default=None, help="cuda / cpu. Defaults to cuda if available.")
    return p.parse_args()


def build_index_to_celltype_name(csv_path: str):
    df = pd.read_csv(csv_path)
    if "classification_result" not in df.columns:
        return {}
    df = df.dropna(subset=["classification_result"])
    uncategorized_set = {"Other/Unknown"}
    filtered_classes = sorted([c for c in df["classification_result"].unique() if c not in uncategorized_set])
    idx_to_name = {idx: cls for idx, cls in enumerate(filtered_classes)}
    return idx_to_name


def build_model_from_ckpt_args(args_dict: dict, device: torch.device):
    model = MaskFiLMGMMVAE(
        num_genes=int(args_dict.get("total_gene", 17999)),
        latent_dim=int(args_dict.get("latent_dim", 128)),
        num_components=int(args_dict.get("num_components", 16)),
        expr_hidden_dim=int(args_dict.get("expr_hidden_dim", 1024)),
        mask_hidden_dim=int(args_dict.get("mask_hidden_dim", 512)),
        dec_hidden_dim=int(args_dict.get("dec_hidden_dim", 1024)),
        dropout=float(args_dict.get("dropout", 0.1)),
        prior_type=str(args_dict.get("prior_type", "gmm")),
        prior_cov_rank=int(args_dict.get("prior_cov_rank", 8)),
    ).to(device)
    return model


def strip_module_prefix(state_dict: dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def make_loader(compiled_root: str, split: str, num_genes: int, batch_size: int, num_workers: int):
    ds = CompiledShardDataset(
        compiled_root=compiled_root,
        split=split,
        max_cached_shards=64,
    )
    collator = CompiledSparseBatchCollator(
        num_genes=num_genes,
        apply_mask_aug=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    return loader


def analyze_split(model, loader, device, max_batches=None):
    prior = model.prior
    k = int(prior.K)
    sum_resp_by_ct = defaultdict(lambda: torch.zeros((k,), dtype=torch.float64))
    count_by_ct = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for bidx, (_, _, celltype_ids, x_count, x_mask, x_mask_encoder) in enumerate(loader):
            if max_batches is not None and bidx >= int(max_batches):
                break
            x_count = x_count.to(device, non_blocking=True)
            x_mask_encoder = x_mask_encoder.to(device, non_blocking=True)
            out = model.forward(x_count=x_count, x_mask=x_mask_encoder)
            z = out["z"]
            resp = prior.posterior_responsibilities(z=z, temperature=1.0).detach().cpu().double()
            cts = celltype_ids.cpu().numpy().astype(np.int64)

            for i, ct in enumerate(cts):
                sum_resp_by_ct[int(ct)] += resp[i]
                count_by_ct[int(ct)] += 1

    result = {}
    for ct, s in sum_resp_by_ct.items():
        n = max(1, count_by_ct[ct])
        mean_resp = s / float(n)
        top_comp = int(torch.argmax(mean_resp).item())
        result[ct] = {
            "count": int(count_by_ct[ct]),
            "top_component": top_comp,
            "top_component_resp": float(mean_resp[top_comp].item()),
            "mean_resp": mean_resp.numpy(),
        }
    return result


def print_result(result, idx_to_name, split_name):
    print(f"\n=== Split: {split_name} ===")
    if len(result) == 0:
        print("No results.")
        return
    keys = sorted(result.keys())
    for ct in keys:
        info = result[ct]
        name = idx_to_name.get(ct, f"celltype_{ct}")
        print(
            f"celltype={ct} ({name}) | n={info['count']} | "
            f"top_component={info['top_component']} | top_resp={info['top_component_resp']:.4f}"
        )


def main():
    args = parse_args()
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = build_model_from_ckpt_args(ckpt_args, device=device)
    state = strip_module_prefix(ckpt["model_state_dict"])
    model.load_state_dict(state, strict=True)

    idx_to_name = build_index_to_celltype_name(args.cell_type_csv) if args.cell_type_csv else {}
    num_genes = int(ckpt_args.get("total_gene", 17999))

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        loader = make_loader(
            compiled_root=args.compiled_dataset_root,
            split=split,
            num_genes=num_genes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        result = analyze_split(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )
        print_result(result=result, idx_to_name=idx_to_name, split_name=split)


if __name__ == "__main__":
    main()
