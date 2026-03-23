#!/usr/bin/env python3
"""
embed_chunks.py — Embed chunks from chunks.jsonl using sentence-transformers.

Reads:  inference/00-chunks/chunks.jsonl
Writes: inference/01-embeddings/embeddings.npy
        inference/01-embeddings/metadata.json

Usage:
    python scripts/inference/embed_chunks.py [--chunks path] [--out-dir path]
    python scripts/inference/embed_chunks.py --model all-MiniLM-L6-v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CHUNKS = REPO_ROOT / "inference" / "00-chunks" / "chunks.jsonl"
DEFAULT_OUT    = REPO_ROOT / "inference" / "01-embeddings"
DEFAULT_MODEL  = "all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(description="Embed chunks for clustering.")
    parser.add_argument("--chunks",  default=str(DEFAULT_CHUNKS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help=f"sentence-transformers model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    out_dir     = Path(args.out_dir)

    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run chunk_tex.py first.", file=sys.stderr)
        sys.exit(1)

    # Load chunks
    chunks = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {chunks_path.name}")

    # Build texts to embed — section title + text for richer signal
    texts = []
    for c in chunks:
        header = c.get("section_title", "")
        body   = c.get("text", "")
        texts.append(f"{header}\n\n{body}"[:4096])  # cap to avoid OOM

    # Embed
    print(f"Loading model: {args.model} ...")
    model = SentenceTransformer(args.model)

    print(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Write
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path  = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.json"

    np.save(str(emb_path), embeddings)

    metadata = {
        "model":       args.model,
        "n_chunks":    len(chunks),
        "shape":       list(embeddings.shape),
        "chunks_file": str(chunks_path),
        "chunks": [
            {
                "index":         c["chunk_index"],
                "source":        c["source"],
                "section_title": c["section_title"],
                "section_path":  c["section_path"],
                "char_count":    c["char_count"],
                "token_estimate": c["token_estimate"],
            }
            for c in chunks
        ],
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWritten:")
    print(f"  {emb_path}  ({embeddings.nbytes // 1024} KB)")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
