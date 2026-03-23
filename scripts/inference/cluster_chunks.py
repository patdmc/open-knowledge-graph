#!/usr/bin/env python3
"""
cluster_chunks.py — Cluster chunk embeddings into equivalence classes.

A chunk can belong to multiple clusters (multiplexing) if its cosine
similarity to multiple centroids exceeds --bridge-threshold.

Reads:  inference/01-embeddings/embeddings.npy
        inference/01-embeddings/metadata.json
Writes: inference/02-clusters/classes.json
        inference/02-clusters/bridge_chunks.json
        inference/02-clusters/bridge_pairs.json
        inference/02-clusters/summary.json

Usage:
    python scripts/inference/cluster_chunks.py [options]

Options:
    --n-clusters        Number of equivalence classes (default: auto via elbow)
    --max-clusters      Max k to try in elbow search (default: 15)
    --bridge-threshold  Cosine similarity to be in a secondary cluster (default: 0.55)
    --min-cluster-size  Merge clusters smaller than N chunks (default: 2)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

REPO_ROOT   = Path(__file__).parent.parent.parent
DEFAULT_EMB = REPO_ROOT / "inference" / "01-embeddings"
DEFAULT_OUT = REPO_ROOT / "inference" / "02-clusters"


# ---------------------------------------------------------------------------
# Elbow / silhouette auto-selection
# ---------------------------------------------------------------------------

def best_k(embeddings: np.ndarray, max_k: int) -> int:
    """Pick k via silhouette score."""
    n = len(embeddings)
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 2

    best_score = -1.0
    best_k_val = 2
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        print(f"  k={k:2d}  silhouette={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_k_val = k

    print(f"  → Best k={best_k_val} (silhouette={best_score:.4f})")
    return best_k_val


# ---------------------------------------------------------------------------
# Cluster label generation from top terms in section titles
# ---------------------------------------------------------------------------

def cluster_label(chunk_indices: list[int], metadata: dict) -> str:
    chunks = metadata["chunks"]
    words = defaultdict(int)
    for i in chunk_indices:
        title = chunks[i]["section_title"]
        for w in title.lower().replace(">", " ").split():
            w = w.strip("[].,;:()")
            if len(w) > 3 and w not in {"part", "with", "and", "the", "for", "from"}:
                words[w] += 1
    if not words:
        return f"cluster_{chunk_indices[0]}"
    top = sorted(words, key=lambda w: -words[w])[:3]
    return "-".join(top)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cluster chunks into equivalence classes.")
    parser.add_argument("--emb-dir",         default=str(DEFAULT_EMB))
    parser.add_argument("--out-dir",         default=str(DEFAULT_OUT))
    parser.add_argument("--n-clusters",      type=int,   default=None,
                        help="Fixed number of clusters (default: auto)")
    parser.add_argument("--max-clusters",    type=int,   default=15,
                        help="Max k for auto-selection (default: 15)")
    parser.add_argument("--bridge-threshold", type=float, default=0.55,
                        help="Cosine sim threshold for secondary cluster membership (default: 0.55)")
    parser.add_argument("--min-cluster-size", type=int,  default=2,
                        help="Merge clusters smaller than N into nearest neighbour (default: 2)")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)

    emb_path  = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "metadata.json"

    if not emb_path.exists():
        print(f"Error: {emb_path} not found. Run embed_chunks.py first.", file=sys.stderr)
        sys.exit(1)

    embeddings = np.load(str(emb_path))
    metadata   = json.loads(meta_path.read_text(encoding="utf-8"))
    n_chunks   = len(metadata["chunks"])

    print(f"Loaded {n_chunks} embeddings, dim={embeddings.shape[1]}")

    # L2-normalise for cosine similarity
    normed = normalize(embeddings, norm="l2")

    # Pick k
    if args.n_clusters:
        k = args.n_clusters
        print(f"Using fixed k={k}")
    else:
        print(f"Auto-selecting k (max={args.max_clusters}) via silhouette ...")
        k = best_k(normed, args.max_clusters)

    # Fit
    print(f"\nFitting KMeans k={k} ...")
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    primary_labels = km.fit_predict(normed)
    centroids      = km.cluster_centers_  # already in normed space

    # Primary membership
    chunk_to_classes = defaultdict(list)   # chunk_idx → [class_ids]
    class_to_chunks  = defaultdict(list)   # class_id  → [chunk_idxs]

    for idx, label in enumerate(primary_labels):
        class_id = f"EC{label:02d}"
        chunk_to_classes[idx].append(class_id)
        class_to_chunks[class_id].append(idx)

    # Secondary membership (bridge detection)
    # For each chunk, compute cosine sim to all centroids
    # Assign to any centroid above bridge_threshold (excluding primary)
    sims = normed @ centroids.T  # (n_chunks, k)

    bridge_chunks = {}   # chunk_idx → [class_ids] (only chunks in 2+ classes)
    bridge_pairs  = defaultdict(set)  # (class_a, class_b) → set of chunk_idxs

    for idx in range(n_chunks):
        primary_class = f"EC{primary_labels[idx]:02d}"
        for j in range(k):
            secondary_class = f"EC{j:02d}"
            if secondary_class == primary_class:
                continue
            if sims[idx, j] >= args.bridge_threshold:
                chunk_to_classes[idx].append(secondary_class)
                class_to_chunks[secondary_class].append(idx)

        if len(chunk_to_classes[idx]) > 1:
            bridge_chunks[idx] = chunk_to_classes[idx]
            classes = chunk_to_classes[idx]
            for i_c in range(len(classes)):
                for j_c in range(i_c + 1, len(classes)):
                    pair = tuple(sorted([classes[i_c], classes[j_c]]))
                    bridge_pairs[pair].add(idx)

    # Build class records with labels
    classes_out = {}
    for class_id, chunk_idxs in sorted(class_to_chunks.items()):
        # Deduplicate
        chunk_idxs = sorted(set(chunk_idxs))
        label = cluster_label(chunk_idxs, metadata)
        centroid_idx = int(np.argmax(normed[chunk_idxs] @ centroids[int(class_id[2:])]))
        representative_chunk = chunk_idxs[centroid_idx]
        classes_out[class_id] = {
            "class_id":           class_id,
            "label":              label,
            "n_chunks":           len(chunk_idxs),
            "chunk_indices":      chunk_idxs,
            "representative":     representative_chunk,
            "representative_title": metadata["chunks"][representative_chunk]["section_title"],
            "member_titles":      [metadata["chunks"][i]["section_title"] for i in chunk_idxs],
        }

    # Bridge pairs as sorted list with weight
    bridge_pairs_out = []
    for (class_a, class_b), chunk_set in sorted(bridge_pairs.items()):
        bridge_pairs_out.append({
            "class_a":     class_a,
            "class_b":     class_b,
            "weight":      len(chunk_set),
            "chunk_indices": sorted(chunk_set),
        })
    bridge_pairs_out.sort(key=lambda x: -x["weight"])

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    classes_path = out_dir / "classes.json"
    classes_path.write_text(json.dumps(classes_out, indent=2, ensure_ascii=False), encoding="utf-8")

    bridge_chunks_path = out_dir / "bridge_chunks.json"
    bridge_chunks_out  = {str(k): v for k, v in bridge_chunks.items()}
    bridge_chunks_path.write_text(json.dumps(bridge_chunks_out, indent=2), encoding="utf-8")

    bridge_pairs_path = out_dir / "bridge_pairs.json"
    bridge_pairs_path.write_text(json.dumps(bridge_pairs_out, indent=2), encoding="utf-8")

    chunk_membership_path = out_dir / "chunk_class_membership.json"
    chunk_membership_out  = {str(k): v for k, v in sorted(chunk_to_classes.items())}
    chunk_membership_path.write_text(json.dumps(chunk_membership_out, indent=2), encoding="utf-8")

    summary = {
        "n_chunks":          n_chunks,
        "n_classes":         len(classes_out),
        "bridge_threshold":  args.bridge_threshold,
        "n_bridge_chunks":   len(bridge_chunks),
        "n_bridge_pairs":    len(bridge_pairs_out),
        "classes_summary": [
            {
                "class_id": cid,
                "label":    cdata["label"],
                "n_chunks": cdata["n_chunks"],
                "representative_title": cdata["representative_title"],
            }
            for cid, cdata in sorted(classes_out.items())
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults:")
    print(f"  {len(classes_out)} equivalence classes")
    print(f"  {len(bridge_chunks)} bridge chunks (in 2+ classes)")
    print(f"  {len(bridge_pairs_out)} bridge pairs")
    print()
    for cid, cdata in sorted(classes_out.items()):
        print(f"  {cid} [{cdata['label'][:40]:40s}] — {cdata['n_chunks']} chunks")
    print()
    print(f"Written to {out_dir}/")


if __name__ == "__main__":
    main()
