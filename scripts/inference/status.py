#!/usr/bin/env python3
"""
status.py — Pipeline status reporter.

Shows progress across all inference pipeline phases.

Usage:
    python scripts/inference/status.py [--dir inference/]
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DIR = REPO_ROOT / "inference"

PHASES = [
    ("00-chunks",          "chunks.jsonl",           "Chunked tex files"),
    ("01-embeddings",      "embeddings.npy",         "Embeddings"),
    ("02-clusters",        "summary.json",           "Equivalence classes"),
    ("03-keywords",        "_status.json",           "Keyword extraction"),
    ("04-collated",        None,                     "Collated classes"),
    ("05-representatives", "cross_manifest.json",    "Representatives + cross manifest"),
    ("06-cross-scores",    "_status.json",           "Cross-pair scoring"),
]


def fmt_status(done, total):
    if total == 0:
        return "empty"
    pct = 100 * done // total
    bar = ("█" * (pct // 10)).ljust(10)
    return f"{bar} {done}/{total} ({pct}%)"


def main():
    parser = argparse.ArgumentParser(description="Show inference pipeline status.")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    inf_dir = Path(args.dir)
    print(f"\nPipeline status: {inf_dir}\n")
    print(f"  {'Phase':<30} {'Status':<40} Notes")
    print(f"  {'─'*30} {'─'*40} {'─'*30}")

    for phase_dir, sentinel, label in PHASES:
        phase_path = inf_dir / phase_dir

        if not phase_path.exists():
            print(f"  {label:<30} {'not started':<40}")
            continue

        if sentinel is None:
            # Count files
            files = list(phase_path.glob("*.json"))
            files = [f for f in files if not f.name.startswith("_")]
            print(f"  {label:<30} {fmt_status(len(files), len(files)):<40} {len(files)} files")
            continue

        sentinel_path = phase_path / sentinel
        if not sentinel_path.exists():
            print(f"  {label:<30} {'in progress':<40}")
            continue

        # Parse sentinel for detail
        notes = ""
        if sentinel.endswith(".json"):
            try:
                data = json.loads(sentinel_path.read_text(encoding="utf-8"))

                if sentinel == "summary.json":
                    n_cls = data.get("n_classes", "?")
                    n_br  = data.get("n_bridge_chunks", "?")
                    notes = f"{n_cls} classes, {n_br} bridge chunks"
                    print(f"  {label:<30} {'complete':<40} {notes}")

                elif sentinel == "_status.json":
                    done  = len(data.get("done", []))
                    total = data.get("total", done)
                    fail  = len(data.get("failed", []))
                    notes = f"{fail} failed" if fail else ""
                    print(f"  {label:<30} {fmt_status(done, total):<40} {notes}")

                elif sentinel == "cross_manifest.json":
                    n_pairs  = len(data) if isinstance(data, list) else "?"
                    n_bridge = sum(1 for p in data if p.get("priority") == "bridge") if isinstance(data, list) else "?"
                    notes    = f"{n_pairs} pairs ({n_bridge} bridge)"
                    print(f"  {label:<30} {'complete':<40} {notes}")

                elif sentinel == "chunks.jsonl":
                    lines = sentinel_path.read_text(encoding="utf-8").strip().splitlines()
                    notes = f"{len(lines)} chunks"
                    print(f"  {label:<30} {'complete':<40} {notes}")

                else:
                    print(f"  {label:<30} {'complete':<40}")
            except Exception:
                print(f"  {label:<30} {'complete (unreadable)':<40}")
        elif sentinel.endswith(".npy"):
            import numpy as np
            emb = np.load(str(sentinel_path))
            notes = f"shape={emb.shape}"
            print(f"  {label:<30} {'complete':<40} {notes}")
        else:
            print(f"  {label:<30} {'complete':<40}")

    # Show cluster detail if available
    clusters_path = inf_dir / "02-clusters" / "summary.json"
    if clusters_path.exists():
        summary = json.loads(clusters_path.read_text(encoding="utf-8"))
        print(f"\n  Equivalence classes:")
        for cls in summary.get("classes_summary", []):
            print(f"    {cls['class_id']}  [{cls['label'][:35]:35s}]  {cls['n_chunks']} chunks")

    # Show cross-scores if available
    scores_path = inf_dir / "06-cross-scores" / "pairs.jsonl"
    if scores_path.exists():
        pairs = [json.loads(l) for l in scores_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if pairs:
            print(f"\n  Top cross-pair scores:")
            sorted_pairs = sorted(pairs, key=lambda p: -p.get("cosine_similarity", 0))
            for p in sorted_pairs[:5]:
                ca, cb = p["class_a"], p["class_b"]
                cos    = p.get("cosine_similarity", "?")
                nli    = p.get("nli", {})
                top    = nli.get("top", "n/a") if isinstance(nli, dict) else "n/a"
                print(f"    {ca} × {cb}  cos={cos:.3f}  nli={top}")

    print()


if __name__ == "__main__":
    main()
