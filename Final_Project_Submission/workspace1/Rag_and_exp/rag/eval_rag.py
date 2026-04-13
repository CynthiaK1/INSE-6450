"""
eval_rag.py — measure retrieval quality and citation faithfulness.

Three metrics:
  1. Recall@k on a hand-built question set (you provide ground-truth pages).
  2. Citation faithfulness: fraction of [source:page] tags in generated text
     that point to chunks actually retrieved (catches hallucinated citations).
  3. Citation groundedness: for each cited chunk, does it actually contain
     evidence for the surrounding sentence? Approximated via cross-encoder
     entailment score (sentence ↔ chunk).

Build the eval set as a small JSON file:
[
  {"q": "What does NIST AI RMF Measure require?",
   "relevant": [{"source": "nist", "page": 14}, {"source": "nist", "page": 15}]},
  ...
]
"""

import argparse
import json
import re

import torch
from sentence_transformers import CrossEncoder

from generator import CITE_RE
from retriever import HybridRetriever


def recall_at_k(retriever, eval_set, k=5):
    hits = 0
    total = 0
    for ex in eval_set:
        results = retriever.retrieve(ex["q"], k=k)
        retrieved = {(r["source"], r["page"]) for r in results}
        relevant = {(r["source"], r["page"]) for r in ex["relevant"]}
        if retrieved & relevant:
            hits += 1
        total += 1
    return hits / max(total, 1)


def citation_faithfulness(generated_text, allowed_pairs):
    tags = list(CITE_RE.finditer(generated_text))
    if not tags:
        return 0.0
    good = sum(1 for m in tags if (m.group(1), int(m.group(2))) in allowed_pairs)
    return good / len(tags)


def groundedness(generated_text, retrieval_index, threshold=0.3):
    """
    For each (sentence, citation) pair, score the cross-encoder entailment.
    retrieval_index: dict (source, page) -> chunk text.
    Returns mean score and pass rate at threshold.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    sentences = re.split(r"(?<=[.!?])\s+", generated_text)
    pairs = []
    keys = []
    for s in sentences:
        for m in CITE_RE.finditer(s):
            key = (m.group(1), int(m.group(2)))
            if key in retrieval_index:
                clean_s = CITE_RE.sub("", s).strip()
                pairs.append((clean_s, retrieval_index[key]))
                keys.append(key)
    if not pairs:
        return {"mean_score": 0.0, "pass_rate": 0.0, "n_pairs": 0}

    scores = ce.predict(pairs)
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return {
        "mean_score": float(norm.mean()),
        "pass_rate": float((norm >= threshold).mean()),
        "n_pairs": len(pairs),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_set", required=True)
    p.add_argument("--index_dir", default="index")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    retriever = HybridRetriever(args.index_dir, rerank=True)
    eval_set = json.load(open(args.eval_set))
    r_at_k = recall_at_k(retriever, eval_set, k=args.k)
    print(f"Recall@{args.k}: {r_at_k:.4f}")
