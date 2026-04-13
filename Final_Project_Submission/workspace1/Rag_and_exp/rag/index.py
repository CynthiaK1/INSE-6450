"""
index.py — embed chunks with sentence-transformers and build a FAISS index.

Default model: BAAI/bge-small-en-v1.5 (fast, strong on retrieval, runs on a
single GPU or CPU). Switch to bge-large-en-v1.5 if you have headroom.

Output:
  - chunks.faiss   : FAISS index (cosine via normalized inner product)
  - chunks.meta.jsonl : aligned metadata records (one per index row)

Usage:
    python index.py --jsonl chunks.jsonl --out_dir index/
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def build(jsonl_path, out_dir, model_name, batch_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in open(jsonl_path)]
    texts = [r["text"] for r in records]
    print(f"Loaded {len(records)} chunks")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")
    print(f"Embedding shape: {embs.shape}")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(out_dir / "chunks.faiss"))

    with open(out_dir / "chunks.meta.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(out_dir / "index_config.json", "w") as f:
        json.dump({"model": model_name, "dim": dim, "n_chunks": len(records)}, f)

    print(f"Index written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out_dir", default="index")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    build(args.jsonl, args.out_dir, args.model, args.batch_size)
