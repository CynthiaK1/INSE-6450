"""
retriever.py — hybrid dense + BM25 retriever with optional cross-encoder rerank.

Why hybrid: NIST/AMLAS contain a lot of standardized terminology
("MEASURE 2.6", "GOVERN", "AML.05") that BM25 nails and dense embeddings
sometimes miss. Conversely, paraphrased queries from RobustOps need dense
similarity. We score candidates with both, normalize, and combine.

Optional rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 over top-50, returning top-k.

Usage:
    from retriever import HybridRetriever
    r = HybridRetriever("index/")
    hits = r.retrieve("How should risk be measured under NIST AI RMF?", k=5)
"""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def _tokenize(text):
    return [t.lower() for t in text.split() if t.strip()]


class HybridRetriever:
    def __init__(self, index_dir, rerank=True,
                 reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        index_dir = Path(index_dir)
        cfg = json.load(open(index_dir / "index_config.json"))
        self.faiss_index = faiss.read_index(str(index_dir / "chunks.faiss"))
        self.meta = [json.loads(line) for line in open(index_dir / "chunks.meta.jsonl")]
        self.texts = [m["text"] for m in self.meta]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(cfg["model"], device=device)
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        self.rerank = rerank
        self.reranker = CrossEncoder(reranker_name, device=device) if rerank else None

    def _dense(self, query, k):
        q = self.encoder.encode([query], normalize_embeddings=True,
                                convert_to_numpy=True).astype("float32")
        scores, idx = self.faiss_index.search(q, k)
        return idx[0], scores[0]

    def _sparse(self, query, k):
        scores = self.bm25.get_scores(_tokenize(query))
        top = np.argsort(-scores)[:k]
        return top, scores[top]

    @staticmethod
    def _norm(x):
        x = np.asarray(x, dtype="float32")
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    def retrieve(self, query, k=5, fetch_k=50, alpha=0.6,
                 source_filter=None):
        d_idx, d_sc = self._dense(query, fetch_k)
        s_idx, s_sc = self._sparse(query, fetch_k)

        cand = {}
        for i, s in zip(d_idx, self._norm(d_sc)):
            cand[int(i)] = cand.get(int(i), 0) + alpha * float(s)
        for i, s in zip(s_idx, self._norm(s_sc)):
            cand[int(i)] = cand.get(int(i), 0) + (1 - alpha) * float(s)

        ranked = sorted(cand.items(), key=lambda x: -x[1])

        if source_filter:
            ranked = [(i, s) for i, s in ranked
                      if self.meta[i]["source"] in source_filter]

        if self.rerank and ranked:
            top_pool = ranked[:fetch_k]
            pairs = [(query, self.texts[i]) for i, _ in top_pool]
            rerank_scores = self.reranker.predict(pairs)
            ranked = sorted(
                [(top_pool[j][0], float(rerank_scores[j])) for j in range(len(top_pool))],
                key=lambda x: -x[1],
            )

        out = []
        for i, score in ranked[:k]:
            m = self.meta[i]
            out.append({
                "id": m["id"], "source": m["source"], "page": m["page"],
                "heading": m["heading"], "text": m["text"], "score": score,
            })
        return out
