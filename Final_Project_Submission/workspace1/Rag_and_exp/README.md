# RobustOps Add-ons (Turn 3 deliverables)

Two packages: `experiments/` and `rag/`. Drop into your existing INSE-6450 repo.

## Setup

```bash
pip install numpy pandas scipy scikit-learn torch faiss-cpu \
    sentence-transformers rank_bm25 pypdf anthropic
# faiss-gpu if you have it
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run sequence

### 1. Stronger MIL evidence (H3)
```bash
cd experiments/
python mil_baselines.py --n_trials 50 --n_window 500 --n_drift 50 --noise 2.5
```
Outputs: `mil_results.csv`, paired Wilcoxon p-values vs PSI / IsoForest / C2ST / KNN, Cohen's d for each. **Replace `attention_mil_proxy` in `mil_baselines.py` with your trained attention-MIL forward pass before reporting in the paper.** The proxy gives a sane Mahalanobis-style baseline so the harness runs end-to-end now.

### 2. Learnable risk score + smarter gate (H4)
```bash
python learnable_risk_score.py --csv risk_features.csv --target failed
```
Expects a CSV with columns `delta_acc, delta_conf, asr, wcasr, d_px, d_pyx, d_py, failed`. Build it from your existing per-window evaluation outputs across Adult / California Housing / SST-2 / CIFAR-10 — one row per (model, drift_severity) cell. The script reports learned softmax weights, train AUC, and a calibrated logistic gate with τ chosen for FPR ≤ 5%.

### 3. RAG for NIST + AMLAS grounding
```bash
cd rag/
# ingest both PDFs into a single chunks file
python ingest.py --pdf nist_ai_rmf.pdf --source nist --out chunks.jsonl
python ingest.py --pdf amlas.pdf --source amlas --out chunks.jsonl --append

# build the FAISS index
python index.py --jsonl chunks.jsonl --out_dir index/

# generate a grounded compliance report from a RobustOps audit log
python generator.py --audit_log audit.json --index_dir index/ --out report.md

# evaluate retrieval quality on a hand-built question set
python eval_rag.py --eval_set eval.json --index_dir index/
```

## What each module does

| File | Pillar | What it adds |
|---|---|---|
| `experiments/mil_baselines.py` | Observe / H3 | 4 baselines (PSI, IsoForest, C2ST, KNN) + paired Wilcoxon vs attention-MIL |
| `experiments/learnable_risk_score.py` | Score / H4 | Softmax-constrained learnable weights + logistic meta-gate calibrated to target FPR |
| `rag/ingest.py` | Observe / NIST | Sentence-aware chunking with heading detection |
| `rag/index.py` | — | bge-small embeddings + FAISS |
| `rag/retriever.py` | — | Hybrid dense + BM25 + cross-encoder rerank |
| `rag/generator.py` | NIST / AMLAS | Constrained generation with citation verification + retry |
| `rag/eval_rag.py` | — | Recall@k, citation faithfulness, groundedness |

## Paper additions

### New §6.1 (after the Dashboard section)

> **Grounded compliance reporting via retrieval-augmented generation.** To convert RobustOps audit logs into NIST AI RMF and AMLAS compliance text without hallucinated citations, we built a RAG layer over the NIST AI RMF 1.0 and AMLAS PDFs. The pipeline ingests both documents, chunks them with sentence-boundary respecting overlap and best-effort heading detection, and embeds the chunks with a BGE-small encoder into a FAISS index. Retrieval is hybrid: BM25 captures NIST-specific tokens such as `MEASURE 2.6` that dense models tend to miss, and dense similarity captures paraphrased queries from the audit log. The top fifty candidates are re-ranked by an MS-MARCO cross-encoder. The generator (Claude Sonnet) is constrained to use only retrieved evidence, must cite every factual claim with a `[source:page]` tag drawn from the retrieved set, and is forced to write "evidence insufficient" rather than fabricate when retrieval is empty. A post-generation verifier scans every emitted citation tag against the retrieved set and triggers a single retry if any tag is unsupported. We measure citation faithfulness (fraction of emitted tags that point to actually-retrieved chunks) and groundedness (cross-encoder entailment between the cited sentence and the cited chunk) on a hand-built question set, in addition to recall@k for retrieval. This addresses a recurring problem with LLM-generated compliance documents: ungrounded citations are worse than no citations at all because they create the appearance of evidence where none exists.

### Add to §7 Limitations and Future Work

> **Adversarial threats to the model artifact itself.** Our threat model is restricted to inference-time input perturbation, but the binary serialization format of the audited model is itself an attack surface. PyTorch's `torch.load` uses Python's `pickle` module, which permits arbitrary code execution during deserialization and has multiple disclosed CVEs (e.g., CVE-2025-32434, CVE-2022-45907); the `safetensors` format was introduced specifically to provide a deserialization path with no code execution. A complete risk-assessment pipeline should verify the integrity and provenance of the model artifact before any of the four pillars run, including refusing to load pickle-format checkpoints from untrusted sources, validating cryptographic signatures, and scanning for embedded payloads. We flag this as the highest-priority extension to the framework's threat model and explicitly scope it as future work rather than claim coverage we do not have.

### Update H4's status in §5.4

After running `learnable_risk_score.py` on your assembled CSV, replace the current Risk Score Weight Ablation paragraph with: "Learned weights converge to (w₁=…, w₂=…, w₃=…, w₄=…), placing the largest mass on the [highest-weight component], consistent with the H4 prediction that the unified score acts as a stable decision signal. The smarter-gate logistic meta-classifier achieves test AUC = … and Brier = …, with the calibrated threshold producing an actual FPR of … against a 5% target." Drop in the actual numbers from the run.

## Constraints I want to be honest about

- The MIL script uses a Mahalanobis proxy in place of your trained attention-MIL forward pass. I cannot import your trained module from chat. Replace `attention_mil_proxy` with `your_module.forward(buffer, window)` before publishing numbers.
- The RAG eval depends on a hand-built `eval.json` of 20–40 question / ground-truth-page pairs. Build this yourself from sections of NIST AI RMF you already understand — it should take about an hour and is the only manual step in the whole pipeline.
- Citation faithfulness and groundedness are *necessary but not sufficient* for trustworthiness. They catch hallucinated tags and obvious topic drift; they do not catch subtle misreadings of the cited passage.
- I have not tested any of these scripts end-to-end — they are written to be runnable but you should expect 30 minutes of debugging when you actually wire them up. The Wilcoxon, hybrid retriever, and citation verifier are the parts I'm most confident in; the heading-detection regex and the smarter-gate FPR calibration are the parts most likely to need tuning to your data.
