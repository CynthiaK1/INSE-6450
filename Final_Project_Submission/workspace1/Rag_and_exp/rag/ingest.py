"""
ingest.py — chunk NIST AI RMF and AMLAS PDFs into retrievable passages.

Strategy:
  - Use pypdf to extract text per page.
  - Chunk by ~400 tokens with 80-token overlap, snapping to sentence boundaries.
  - Store rich metadata: source, page, chunk_idx, section_heading (best-effort
    by detecting ALL-CAPS or numbered headings near the chunk start).

Output: chunks.jsonl with {id, source, page, chunk_idx, heading, text}.

Usage:
    python ingest.py --pdf nist_ai_rmf.pdf --source nist --out chunks.jsonl
    python ingest.py --pdf amlas.pdf --source amlas --out chunks.jsonl --append
"""

import argparse
import json
import re
import uuid
from pathlib import Path

from pypdf import PdfReader

HEADING_RE = re.compile(r"^(\d+(\.\d+)*\s+[A-Z][^\n]{3,80}|[A-Z][A-Z\s]{4,60})$")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def words(text):
    return text.split()


def detect_heading(page_text, chunk_start_char):
    """Find the most recent heading-like line before chunk_start_char."""
    head_text = page_text[:chunk_start_char]
    lines = [ln.strip() for ln in head_text.splitlines() if ln.strip()]
    for ln in reversed(lines[-15:]):
        if HEADING_RE.match(ln):
            return ln[:120]
    return ""


def chunk_text(text, target_tokens=400, overlap_tokens=80):
    sents = SENT_SPLIT_RE.split(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        s_len = len(words(s))
        if cur_len + s_len > target_tokens and cur:
            chunks.append(" ".join(cur))
            # rebuild overlap from the tail of cur
            tail, tail_len = [], 0
            for prev in reversed(cur):
                if tail_len + len(words(prev)) > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_len += len(words(prev))
            cur, cur_len = tail[:], tail_len
        cur.append(s)
        cur_len += s_len
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if c.strip()]


def ingest(pdf_path, source, out_path, append):
    reader = PdfReader(str(pdf_path))
    mode = "a" if append and Path(out_path).exists() else "w"
    n_written = 0
    with open(out_path, mode) as f:
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            page_text = re.sub(r"\s+\n", "\n", page_text)
            page_text = re.sub(r"[ \t]+", " ", page_text)
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                start = page_text.find(chunk[:60])
                heading = detect_heading(page_text, max(0, start)) if start >= 0 else ""
                rec = {
                    "id": str(uuid.uuid4()),
                    "source": source,
                    "page": page_num,
                    "chunk_idx": chunk_idx,
                    "heading": heading,
                    "text": chunk,
                }
                f.write(json.dumps(rec) + "\n")
                n_written += 1
    print(f"Wrote {n_written} chunks from {pdf_path} to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--source", required=True, choices=["nist", "amlas", "iso42001"])
    p.add_argument("--out", default="chunks.jsonl")
    p.add_argument("--append", action="store_true")
    args = p.parse_args()
    ingest(args.pdf, args.source, args.out, args.append)
