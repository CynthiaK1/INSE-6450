"""
generator.py — grounded NIST/AMLAS report generation for RobustOps audit logs.

Uses the Anthropic API. Pipeline:
  1. Take a RobustOps audit_log.json (diagnose + score + mitigate + observe).
  2. For each section of the report we want to generate (e.g., "MEASURE 2.6
     compliance", "AMLAS assurance argument"), formulate a query.
  3. Retrieve top-k passages via HybridRetriever.
  4. Build a constrained prompt: model is allowed to use ONLY retrieved
     passages, must cite each claim by [source:page], and must abstain if
     evidence is insufficient.
  5. Verify that every [source:page] tag in the output corresponds to an
     actually-retrieved chunk. Drop the response and retry once if not.

This is the H-anchor for your point 19 ("RAG for NIST AIRMF & AMLAS quoting").

Usage:
    python generator.py --audit_log audit.json --index_dir index/ --out report.md
"""

import argparse
import json
import os
import re
from pathlib import Path

import anthropic

from retriever import HybridRetriever

CITE_RE = re.compile(r"\[(nist|amlas|iso42001):(\d+)\]")

SECTION_QUERIES = {
    "measure_function": (
        "How does the NIST AI RMF Measure function require risks to be "
        "identified, characterized, and quantified for AI systems?"
    ),
    "manage_function": (
        "How does the NIST AI RMF Manage function require risks to be "
        "prioritized, responded to, and recovered from?"
    ),
    "amlas_assurance": (
        "What evidence does AMLAS require to construct an assurance case "
        "for a machine learning component, and what is a Goal Structuring "
        "Notation argument?"
    ),
    "human_oversight": (
        "What are the requirements for human oversight and human-in-the-loop "
        "review of automated AI decisions?"
    ),
    "monitoring_obligations": (
        "What ongoing monitoring and post-deployment obligations exist for "
        "high-impact AI systems?"
    ),
}

SYSTEM_PROMPT = """You are an AI risk assessment writer producing a section of a compliance report.

STRICT RULES:
1. You may use ONLY the information in the provided EVIDENCE block. Do not use any prior knowledge.
2. Every factual claim must end with a citation tag in the form [source:page], for example [nist:14] or [amlas:7]. Use only the source and page numbers that appear in the EVIDENCE block.
3. If the evidence does not support a claim the audit log makes, write: "Evidence insufficient to assess this claim." Do not invent citations.
4. Be concise, factual, and avoid hedging language. No marketing tone.
5. Connect each evidence point back to the specific RobustOps audit log field it justifies."""

USER_TEMPLATE = """AUDIT LOG (RobustOps output):
```json
{audit}
```

SECTION TO WRITE: {section_name}

EVIDENCE (retrieved passages — these are your only allowed sources):
{evidence}

Write the section now. Every claim cited as [source:page]."""


def format_evidence(hits):
    blocks = []
    for h in hits:
        head = f" — {h['heading']}" if h["heading"] else ""
        blocks.append(
            f"[{h['source']}:{h['page']}]{head}\n{h['text']}\n"
        )
    return "\n---\n".join(blocks)


def verify_citations(text, allowed):
    """Return (ok, bad_tags). allowed is a set of (source, page) tuples."""
    bad = []
    for m in CITE_RE.finditer(text):
        key = (m.group(1), int(m.group(2)))
        if key not in allowed:
            bad.append(m.group(0))
    return len(bad) == 0, bad


def generate_section(client, retriever, audit_log, section_name, section_query, k=6):
    hits = retriever.retrieve(section_query, k=k)
    if not hits:
        return "Evidence insufficient to assess this claim.", []
    allowed = {(h["source"], h["page"]) for h in hits}
    evidence = format_evidence(hits)
    user_msg = USER_TEMPLATE.format(
        audit=json.dumps(audit_log, indent=2)[:4000],
        section_name=section_name, evidence=evidence,
    )

    for attempt in range(2):
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        ok, bad = verify_citations(text, allowed)
        if ok:
            return text, hits
        user_msg += (
            f"\n\nYour previous response contained invalid citations: {bad}. "
            "Retry using ONLY the source:page tags present in EVIDENCE."
        )
    return text + f"\n\n[WARN: unverified citations: {bad}]", hits


def build_report(audit_log, index_dir, out_path):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    retriever = HybridRetriever(index_dir, rerank=True)

    sections = []
    for name, query in SECTION_QUERIES.items():
        print(f"Generating: {name}")
        body, hits = generate_section(client, retriever, audit_log, name, query)
        sections.append((name, body, hits))

    with open(out_path, "w") as f:
        f.write("# RobustOps Compliance Report (RAG-grounded)\n\n")
        f.write(f"Generated from audit log: `{audit_log.get('id', 'unknown')}`\n\n")
        for name, body, hits in sections:
            f.write(f"## {name.replace('_', ' ').title()}\n\n{body}\n\n")
            f.write("**Retrieved evidence:**\n")
            for h in hits:
                f.write(f"- [{h['source']}:{h['page']}] {h['heading'] or '(no heading)'}\n")
            f.write("\n")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit_log", required=True)
    p.add_argument("--index_dir", default="index")
    p.add_argument("--out", default="report.md")
    args = p.parse_args()
    audit = json.load(open(args.audit_log))
    build_report(audit, args.index_dir, args.out)
