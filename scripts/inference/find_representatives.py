#!/usr/bin/env python3
"""
find_representatives.py — Extractive synthesis: find representative chunks per class.

For each equivalence class, identifies:
  - The centroid-nearest chunk (most representative)
  - Top N chunks by cosine similarity to centroid
  - Top keyphrases aggregated across all class chunks
  - All theorems/definitions/claims referenced in the class
  - All citations referenced in the class

This is extractive synthesis — no LLM needed. The centroid-nearest
chunk IS the representative claim for the class.

Reads:  inference/01-embeddings/embeddings.npy + metadata.json
        inference/02-clusters/classes.json
        inference/04-collated/class_*.json
Writes: inference/05-representatives/class_{ID}.json
        inference/05-representatives/cross_manifest.json  (ordered pair list)

Usage:
    python scripts/inference/find_representatives.py [options]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

REPO_ROOT        = Path(__file__).parent.parent.parent
DEFAULT_EMB      = REPO_ROOT / "inference" / "01-embeddings"
DEFAULT_CLUSTERS = REPO_ROOT / "inference" / "02-clusters"
DEFAULT_COLLATED = REPO_ROOT / "inference" / "04-collated"
DEFAULT_OUT      = REPO_ROOT / "inference" / "05-representatives"


def centroid_of(indices: list[int], normed: np.ndarray) -> np.ndarray:
    vecs = normed[indices]
    c    = vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c


def main():
    parser = argparse.ArgumentParser(description="Extractive synthesis per equivalence class.")
    parser.add_argument("--emb-dir",      default=str(DEFAULT_EMB))
    parser.add_argument("--clusters-dir", default=str(DEFAULT_CLUSTERS))
    parser.add_argument("--collated-dir", default=str(DEFAULT_COLLATED))
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT))
    parser.add_argument("--top-chunks",   type=int, default=3,
                        help="Top N chunks to include per class (default: 3)")
    parser.add_argument("--top-phrases",  type=int, default=10,
                        help="Top N keyphrases to aggregate per class (default: 10)")
    args = parser.parse_args()

    emb_dir      = Path(args.emb_dir)
    clusters_dir = Path(args.clusters_dir)
    collated_dir = Path(args.collated_dir)
    out_dir      = Path(args.out_dir)

    embeddings = np.load(str(emb_dir / "embeddings.npy"))
    metadata   = json.loads((emb_dir / "metadata.json").read_text(encoding="utf-8"))
    classes    = json.loads((clusters_dir / "classes.json").read_text(encoding="utf-8"))
    bridge_pairs = json.loads((clusters_dir / "bridge_pairs.json").read_text(encoding="utf-8"))

    normed = normalize(embeddings, norm="l2")
    chunks_meta = metadata["chunks"]

    out_dir.mkdir(parents=True, exist_ok=True)

    class_centroids = {}  # class_id → centroid vector

    for class_id, class_data in sorted(classes.items()):
        collated_path = collated_dir / f"class_{class_id}.json"
        if not collated_path.exists():
            print(f"  {class_id}: missing collated file, skipping")
            continue

        collated     = json.loads(collated_path.read_text(encoding="utf-8"))
        chunk_idxs   = class_data["chunk_indices"]

        # Compute centroid
        centroid = centroid_of(chunk_idxs, normed)
        class_centroids[class_id] = centroid

        # Rank by similarity to centroid
        sims = normed[chunk_idxs] @ centroid
        ranked = sorted(zip(chunk_idxs, sims.tolist()), key=lambda x: -x[1])

        top_chunks = []
        for idx, sim in ranked[:args.top_chunks]:
            top_chunks.append({
                "chunk_index":   idx,
                "section_title": chunks_meta[idx]["section_title"],
                "similarity":    round(sim, 4),
                "source":        chunks_meta[idx]["source"],
            })

        # Aggregate keyphrases across all chunks in class
        phrase_scores = defaultdict(float)
        for chunk_data in collated["chunks"]:
            for kw in chunk_data.get("keyphrases", []):
                phrase_scores[kw["phrase"]] += kw["score"]

        top_phrases = sorted(phrase_scores, key=lambda p: -phrase_scores[p])[:args.top_phrases]

        # Aggregate structural elements
        all_theorems   = []
        all_definitions = []
        all_claims     = []
        all_citations  = []
        all_bold       = []

        for chunk_data in collated["chunks"]:
            struct = chunk_data.get("structure", {})
            all_theorems.extend(struct.get("theorems", []))
            all_definitions.extend(struct.get("definitions", []))
            all_claims.extend(struct.get("claims", []))
            all_citations.extend(struct.get("citations", []))
            all_bold.extend(struct.get("bold_claims", []))

        # Deduplicate citations
        seen_cites = {}
        for c in all_citations:
            seen_cites[c] = True

        result = {
            "class_id":      class_id,
            "label":         class_data["label"],
            "n_chunks":      len(chunk_idxs),
            "representative": {
                "chunk_index":   ranked[0][0],
                "section_title": chunks_meta[ranked[0][0]]["section_title"],
                "similarity":    round(ranked[0][1], 4),
            },
            "top_chunks":    top_chunks,
            "top_keyphrases": [
                {"phrase": p, "score": round(phrase_scores[p], 4)}
                for p in top_phrases
            ],
            "theorems":      all_theorems,
            "definitions":   all_definitions,
            "claims":        all_claims,
            "citations":     list(seen_cites.keys()),
            "bold_claims":   list(dict.fromkeys(all_bold))[:15],
            "bridge_chunks": [
                c["chunk_index"] for c in collated["chunks"] if c.get("is_bridge")
            ],
        }

        out_path = out_dir / f"class_{class_id}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  {class_id} [{class_data['label'][:35]:35s}] "
              f"rep={chunks_meta[ranked[0][0]]['section_title'][:40]!r}")

    # Build cross-manifest: ordered pairs for cross-class synthesis
    # Priority: bridge pairs first (by weight), then centroid similarity
    cross_pairs = []

    # 1. Bridge pairs
    for bp in bridge_pairs:
        cross_pairs.append({
            "class_a":    bp["class_a"],
            "class_b":    bp["class_b"],
            "priority":   "bridge",
            "weight":     bp["weight"],
            "chunk_indices": bp["chunk_indices"],
            "label_a":    classes[bp["class_a"]]["label"],
            "label_b":    classes[bp["class_b"]]["label"],
        })

    # 2. Non-bridge pairs by centroid similarity
    class_ids = sorted(class_centroids.keys())
    bridge_set = {(bp["class_a"], bp["class_b"]) for bp in bridge_pairs}

    for i, ca in enumerate(class_ids):
        for cb in class_ids[i+1:]:
            pair = tuple(sorted([ca, cb]))
            if pair in bridge_set:
                continue
            sim = float(class_centroids[ca] @ class_centroids[cb])
            cross_pairs.append({
                "class_a":   pair[0],
                "class_b":   pair[1],
                "priority":  "similarity",
                "weight":    round(sim, 4),
                "label_a":   classes[pair[0]]["label"],
                "label_b":   classes[pair[1]]["label"],
            })

    # Sort: bridge first (by weight desc), then similarity (by weight desc)
    cross_pairs.sort(key=lambda x: (0 if x["priority"] == "bridge" else 1, -x["weight"]))

    manifest_path = out_dir / "cross_manifest.json"
    manifest_path.write_text(json.dumps(cross_pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(cross_pairs)} cross-class pairs in manifest")
    print(f"  {sum(1 for p in cross_pairs if p['priority'] == 'bridge')} bridge pairs")
    print(f"  {sum(1 for p in cross_pairs if p['priority'] == 'similarity')} similarity pairs")
    print(f"\nWritten to {out_dir}/")


if __name__ == "__main__":
    main()
