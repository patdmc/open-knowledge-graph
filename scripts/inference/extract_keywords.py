#!/usr/bin/env python3
"""
extract_keywords.py — Extract keywords and structural claims from chunks.

For each chunk:
  - Regex extraction: theorems, definitions, corollaries, lemmas (with numbers)
  - Regex extraction: citations referenced
  - KeyBERT: top keyphrases from prose
  - LaTeX math: flag presence of equations

Reads:  inference/00-chunks/chunks.jsonl
Writes: inference/03-keywords/chunk_NNN.json
        inference/03-keywords/_status.json

Usage:
    python scripts/inference/extract_keywords.py [--chunks path] [--resume]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from keybert import KeyBERT

REPO_ROOT      = Path(__file__).parent.parent.parent
DEFAULT_CHUNKS = REPO_ROOT / "inference" / "00-chunks" / "chunks.jsonl"
DEFAULT_OUT    = REPO_ROOT / "inference" / "03-keywords"

# ---------------------------------------------------------------------------
# LaTeX structural patterns
# ---------------------------------------------------------------------------

THEOREM_RE    = re.compile(r'\\textbf\{(Theorem|Lemma|Corollary|Proposition)\s*([\d]+[a-z]?)\}', re.I)
DEF_RE        = re.compile(r'\\textbf\{(Definition|Def\.?)\s*([\d]+[a-z]?)\}', re.I)
CLAIM_RE      = re.compile(r'\\textbf\{(Claim|Conjecture|Remark|Open Question)\s*([\d]+[a-z]?)\}', re.I)
CITE_RE       = re.compile(r'\{?\[([A-Za-z][A-Za-z0-9]+\d{4}[a-z]?(?:,\s*[A-Za-z][A-Za-z0-9]+\d{4}[a-z]?)*)\]\}?')
MATH_RE       = re.compile(r'\\\(|\\\[|\$')
BOLD_CLAIM_RE = re.compile(r'\\textbf\{([^}]{10,80})\}')  # bold text that might be a claim

# Strip LaTeX for KeyBERT (we don't want \textbf in keyphrases)
_LATEX_CMD    = re.compile(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^}]*)\})?')
_MATH_BLOCK   = re.compile(r'\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$[^$]+\$')


def strip_for_keybert(text: str) -> str:
    text = _MATH_BLOCK.sub(' [MATH] ', text)
    text = _LATEX_CMD.sub(lambda m: m.group(1) or '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_structure(text: str) -> dict:
    theorems   = [{"type": m.group(1), "number": m.group(2)} for m in THEOREM_RE.finditer(text)]
    definitions= [{"type": m.group(1), "number": m.group(2)} for m in DEF_RE.finditer(text)]
    claims     = [{"type": m.group(1), "number": m.group(2)} for m in CLAIM_RE.finditer(text)]
    citations  = []
    for m in CITE_RE.finditer(text):
        for cid in re.split(r',\s*', m.group(1)):
            citations.append(cid.strip())
    bold_claims = [m.group(1).strip() for m in BOLD_CLAIM_RE.finditer(text)]
    has_math    = bool(MATH_RE.search(text))

    return {
        "theorems":    theorems,
        "definitions": definitions,
        "claims":      claims,
        "citations":   list(dict.fromkeys(citations)),  # deduplicate, preserve order
        "bold_claims": bold_claims[:10],
        "has_math":    has_math,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract keywords and structure from chunks.")
    parser.add_argument("--chunks",   default=str(DEFAULT_CHUNKS))
    parser.add_argument("--out-dir",  default=str(DEFAULT_OUT))
    parser.add_argument("--resume",   action="store_true",
                        help="Skip chunks already in _status.json")
    parser.add_argument("--top-n",    type=int, default=8,
                        help="KeyBERT keyphrases per chunk (default: 8)")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    out_dir     = Path(args.out_dir)

    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found.", file=sys.stderr)
        sys.exit(1)

    chunks = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "_status.json"

    done = set()
    if args.resume and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        done   = set(status.get("done", []))
        print(f"Resuming: {len(done)} already done, {len(chunks) - len(done)} remaining")

    print("Loading KeyBERT model ...")
    kw_model = KeyBERT()

    failed = []
    for chunk in chunks:
        idx = chunk["chunk_index"]
        if idx in done:
            continue

        out_path = out_dir / f"chunk_{idx:03d}.json"
        print(f"  [{idx:03d}] {chunk['section_title'][:60]}", end=" ", flush=True)

        try:
            text         = chunk["text"]
            clean        = strip_for_keybert(text)
            structure    = extract_structure(text)

            keyphrases = []
            if len(clean) > 50:
                kws = kw_model.extract_keywords(
                    clean,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=args.top_n,
                    use_mmr=True,
                    diversity=0.4,
                )
                keyphrases = [{"phrase": kw, "score": round(score, 4)} for kw, score in kws]

            result = {
                "chunk_index":   idx,
                "source":        chunk["source"],
                "section_title": chunk["section_title"],
                "section_path":  chunk["section_path"],
                "char_count":    chunk["char_count"],
                "keyphrases":    keyphrases,
                "structure":     structure,
            }

            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            done.add(idx)
            print(f"→ {len(keyphrases)} phrases, {len(structure['theorems'])} theorems", flush=True)

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            failed.append({"chunk_index": idx, "error": str(e)})

    status = {"done": sorted(done), "failed": failed, "total": len(chunks)}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nDone: {len(done)}/{len(chunks)} chunks")
    if failed:
        print(f"Failed: {len(failed)}")
    print(f"Written to {out_dir}/")


if __name__ == "__main__":
    main()
