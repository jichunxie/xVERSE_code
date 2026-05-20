#!/usr/bin/env python3
"""
Build language embeddings for CL cell types.

Input CSV columns:
  id,name,count

Rows named "cell", "unknown", or "Other/Unknown" are skipped. The output row
order is the same order used by main_mfa/build_train_dataset_v1.py when this
CSV is passed as --cell-type-csv.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Generate cell-type descriptions and OpenAI text embeddings.")
    ap.add_argument(
        "--cell-type-csv",
        default="/hpc/group/xielab/xj58/general/cellxgene_cell_type_id2name.csv",
        help="CSV with id,name,count columns.",
    )
    ap.add_argument(
        "--api-key-path",
        default="/hpc/group/xielab/xj58/general/openai_xielab.txt",
        help="Text file containing the OpenAI API key.",
    )
    ap.add_argument(
        "--output-prefix",
        default="/hpc/group/xielab/xj58/general/cellxgene_cell_type_text",
        help="Output prefix. Writes <prefix>_descriptions.csv and <prefix>_embeddings.npz.",
    )
    ap.add_argument("--chat-model", default="gpt-4o-mini")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between API calls.")
    ap.add_argument("--reuse-descriptions", action="store_true", help="Reuse existing descriptions CSV if present.")
    return ap.parse_args()


def load_api_key(path: str) -> str:
    fp = Path(path).expanduser()
    if not fp.exists():
        env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key
        raise FileNotFoundError(f"OpenAI API key not found: {fp}")
    return fp.read_text().strip()


def load_celltypes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cell type CSV missing columns: {sorted(missing)}")
    rows = []
    for _, row in df.iterrows():
        cid = str(row["id"])
        name = str(row["name"])
        lname = name.strip().lower()
        if lname in {"cell", "unknown", "other/unknown"}:
            continue
        rows.append(
            {
                "index": len(rows),
                "id": cid,
                "name": name,
                "count": int(row["count"]) if "count" in df.columns and pd.notna(row["count"]) else -1,
            }
        )
    return pd.DataFrame(rows)


def generate_description(client, model: str, cid: str, name: str) -> str:
    prompt = (
        "Write 3-5 concise scientific sentences describing this cell type for a single-cell biology model. "
        "Mention core biological function, typical tissue/context if broadly known, and characteristic molecular or morphological features. "
        "Do not overclaim. Return only the description.\n\n"
        f"Cell ontology id: {cid}\n"
        f"Cell type name: {name}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise cell biology ontology assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def embed_texts(client, model: str, texts, batch_size: int, sleep_s: float):
    embs = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend([item.embedding for item in resp.data])
        if sleep_s > 0:
            time.sleep(sleep_s)
        print(f"[Embed] {min(start + batch_size, len(texts))}/{len(texts)}")
    return np.asarray(embs, dtype=np.float32)


def main():
    args = parse_args()
    from openai import OpenAI

    api_key = load_api_key(args.api_key_path)
    client = OpenAI(api_key=api_key)

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    desc_csv = out_prefix.with_name(out_prefix.name + "_descriptions.csv")
    emb_npz = out_prefix.with_name(out_prefix.name + "_embeddings.npz")

    if args.reuse_descriptions and desc_csv.exists():
        desc_df = pd.read_csv(desc_csv)
        print(f"[Reuse] descriptions from {desc_csv}")
    else:
        df = load_celltypes(args.cell_type_csv)
        desc_rows = []
        for i, row in df.iterrows():
            desc = generate_description(client, args.chat_model, row["id"], row["name"])
            desc_rows.append({**row.to_dict(), "description": desc})
            print(f"[Describe] {i + 1}/{len(df)} {row['id']} {row['name']}")
            if args.sleep > 0:
                time.sleep(args.sleep)
        desc_df = pd.DataFrame(desc_rows)
        desc_df.to_csv(desc_csv, index=False)
        print(f"[Write] {desc_csv}")

    texts = [
        f"Cell ontology id: {row.id}. Cell type name: {row.name}. {row.description}"
        for row in desc_df.itertuples(index=False)
    ]
    embeddings = embed_texts(client, args.embedding_model, texts, args.batch_size, args.sleep)
    np.savez_compressed(
        emb_npz,
        embeddings=embeddings,
        ids=desc_df["id"].astype(str).to_numpy(),
        names=desc_df["name"].astype(str).to_numpy(),
        descriptions=desc_df["description"].astype(str).to_numpy(),
        counts=desc_df["count"].to_numpy() if "count" in desc_df.columns else np.full((len(desc_df),), -1),
        embedding_model=np.asarray(args.embedding_model),
        chat_model=np.asarray(args.chat_model),
    )
    print(f"[Write] {emb_npz} shape={embeddings.shape}")


if __name__ == "__main__":
    main()
