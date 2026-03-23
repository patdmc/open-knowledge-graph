#!/usr/bin/env python3
"""
score_cross_pairs.py — Score cross-class pairs using NLI and cosine similarity.

For each pair in the cross_manifest, computes:
  - Cosine similarity between class centroid embeddings
  - NLI relationship: entailment / contradiction / neutral
    (using cross-encoder/nli-deberta-v3-small locally)

No LLM tokens required. Writes scored pairs as JSONL checkpoint.

Reads:  inference/05-representatives/cross_manifest.json
        inference/05-representatives/class_*.json
        inference/01-embeddings/embeddings.npy + metadata.json
        inference/00-chunks/chunks.jsonl         (for representative text)
Writes: inference/06-cross-scores/pairs.jsonl
        inference/06-cross-scores/_status.json

Usage:
    python scripts/inference/score_cross_pairs.py [--resume] [--top N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

REPO_ROOT     = Path(__file__).parent.parent.parent
DEFAULT_REPS  = REPO_ROOT / "inference" / "05-representatives"
DEFAULT_EMB   = REPO_ROOT / "inference" / "01-embeddings"
DEFAULT_CHUNKS= REPO_ROOT / "inference" / "00-chunks" / "chunks.jsonl"
DEFAULT_OUT   = REPO_ROOT / "inference" / "06-cross-scores"

NLI_LABELS = ["contradiction", "neutral", "entailment"]


def load_chunks(chunks_path: Path) -> dict:
    """Return {chunk_index: chunk_dict}."""
    chunks = {}
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                c = json.loads(line)
                chunks[c["chunk_index"]] = c
    return chunks


def get_representative_text(class_rep: dict, chunks: dict, max_chars: int = 1000) -> str:
    idx = class_rep["representative"]["chunk_index"]
    chunk = chunks.get(idx)
    if not chunk:
        return class_rep.get("label", "")
    text = chunk.get("text", "")
    # Strip heavy LaTeX preamble lines
    lines = [l for l in text.split("\n") if not l.strip().startswith("\\")]
    return " ".join(lines)[:max_chars].strip()


def main():
    parser = argparse.ArgumentParser(description="Score cross-class pairs with NLI.")
    parser.add_argument("--reps-dir",    default=str(DEFAULT_REPS))
    parser.add_argument("--emb-dir",     default=str(DEFAULT_EMB))
    parser.add_argument("--chunks",      default=str(DEFAULT_CHUNKS))
    parser.add_argument("--out-dir",     default=str(DEFAULT_OUT))
    parser.add_argument("--top",         type=int, default=None,
                        help="Only score top N pairs by priority")
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--no-nli",      action="store_true",
                        help="Skip NLI model, cosine similarity only (faster)")
    args = parser.parse_args()

    reps_dir   = Path(args.reps_dir)
    emb_dir    = Path(args.emb_dir)
    out_dir    = Path(args.out_dir)

    manifest_path = reps_dir / "cross_manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found. Run find_representatives.py first.", file=sys.stderr)
        sys.exit(1)

    pairs    = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks   = load_chunks(Path(args.chunks))
    metadata = json.loads((emb_dir / "metadata.json").read_text(encoding="utf-8"))

    embeddings = np.load(str(emb_dir / "embeddings.npy"))
    normed     = normalize(embeddings, norm="l2")

    if args.top:
        pairs = pairs[:args.top]
        print(f"Scoring top {len(pairs)} pairs")
    else:
        print(f"Scoring {len(pairs)} pairs")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path    = out_dir / "pairs.jsonl"
    status_path = out_dir / "_status.json"

    # Resume support
    done_pairs = set()
    if args.resume and status_path.exists():
        status     = json.loads(status_path.read_text(encoding="utf-8"))
        done_pairs = {(s["class_a"], s["class_b"]) for s in status.get("done", [])}
        print(f"Resuming: {len(done_pairs)} pairs already done")

    # Load NLI model (optional)
    nli_pipe = None
    if not args.no_nli:
        try:
            from transformers import pipeline
            print("Loading NLI model (cross-encoder/nli-deberta-v3-small) ...")
            nli_pipe = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,  # CPU
            )
            print("NLI model loaded.")
        except Exception as e:
            print(f"Warning: NLI model unavailable ({e}). Using cosine only.", file=sys.stderr)

    done   = []
    failed = []

    mode = "a" if args.resume else "w"
    with open(out_path, mode, encoding="utf-8") as out_fh:
        for pair in pairs:
            ca, cb = pair["class_a"], pair["class_b"]
            key    = (ca, cb)
            if key in done_pairs:
                continue

            rep_a_path = reps_dir / f"class_{ca}.json"
            rep_b_path = reps_dir / f"class_{cb}.json"

            if not rep_a_path.exists() or not rep_b_path.exists():
                print(f"  {ca} × {cb}: missing rep file, skipping")
                failed.append({"class_a": ca, "class_b": cb, "error": "missing rep file"})
                continue

            rep_a = json.loads(rep_a_path.read_text(encoding="utf-8"))
            rep_b = json.loads(rep_b_path.read_text(encoding="utf-8"))

            # Centroid cosine similarity
            idx_a = rep_a["representative"]["chunk_index"]
            idx_b = rep_b["representative"]["chunk_index"]
            cosine_sim = float(normed[idx_a] @ normed[idx_b])

            # NLI
            nli_result = None
            if nli_pipe:
                try:
                    text_a = get_representative_text(rep_a, chunks)
                    text_b = get_representative_text(rep_b, chunks)
                    hypothesis = f"This text is related to: {rep_b['label']}"
                    result     = nli_pipe(text_a[:512], candidate_labels=["entailment", "neutral", "contradiction"])
                    nli_result = {
                        "labels": result["labels"],
                        "scores": [round(s, 4) for s in result["scores"]],
                        "top":    result["labels"][0],
                    }
                except Exception as e:
                    nli_result = {"error": str(e)}

            scored = {
                **pair,
                "cosine_similarity": round(cosine_sim, 4),
                "nli":               nli_result,
                "rep_a_title":       rep_a["representative"]["section_title"],
                "rep_b_title":       rep_b["representative"]["section_title"],
                "top_phrases_a":     [k["phrase"] for k in rep_a.get("top_keyphrases", [])[:5]],
                "top_phrases_b":     [k["phrase"] for k in rep_b.get("top_keyphrases", [])[:5]],
            }

            out_fh.write(json.dumps(scored, ensure_ascii=False) + "\n")
            out_fh.flush()

            nli_label = nli_result["top"] if nli_result and "top" in nli_result else "n/a"
            print(f"  {ca} × {cb}  cos={cosine_sim:.3f}  nli={nli_label}  [{pair['priority']}]")
            done.append({"class_a": ca, "class_b": cb})

    status = {"done": done, "failed": failed, "total": len(pairs)}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nScored {len(done)} pairs → {out_path}")


if __name__ == "__main__":
    main()
