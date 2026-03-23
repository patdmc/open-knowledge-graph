#!/usr/bin/env python3
"""
collate_classes.py — Collate keyword extractions into per-class files.

Multiplexing: a chunk belonging to multiple classes appears in each class's
collated file, annotated with its full class membership.

Reads:  inference/02-clusters/classes.json
        inference/02-clusters/chunk_class_membership.json
        inference/03-keywords/chunk_NNN.json
Writes: inference/04-collated/class_{ID}.json

Usage:
    python scripts/inference/collate_classes.py [options]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT        = Path(__file__).parent.parent.parent
DEFAULT_CLUSTERS = REPO_ROOT / "inference" / "02-clusters"
DEFAULT_KEYWORDS = REPO_ROOT / "inference" / "03-keywords"
DEFAULT_OUT      = REPO_ROOT / "inference" / "04-collated"


def main():
    parser = argparse.ArgumentParser(description="Collate extractions into per-class files.")
    parser.add_argument("--clusters-dir", default=str(DEFAULT_CLUSTERS))
    parser.add_argument("--keywords-dir", default=str(DEFAULT_KEYWORDS))
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT))
    args = parser.parse_args()

    clusters_dir = Path(args.clusters_dir)
    keywords_dir = Path(args.keywords_dir)
    out_dir      = Path(args.out_dir)

    classes_path     = clusters_dir / "classes.json"
    membership_path  = clusters_dir / "chunk_class_membership.json"

    if not classes_path.exists():
        print(f"Error: {classes_path} not found. Run cluster_chunks.py first.", file=sys.stderr)
        sys.exit(1)

    classes    = json.loads(classes_path.read_text(encoding="utf-8"))
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    # membership keys are strings (from JSON)

    out_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_data in sorted(classes.items()):
        chunk_indices = class_data["chunk_indices"]
        collated      = []
        missing       = []

        for idx in chunk_indices:
            kw_path = keywords_dir / f"chunk_{idx:03d}.json"
            if not kw_path.exists():
                missing.append(idx)
                continue

            kw = json.loads(kw_path.read_text(encoding="utf-8"))

            # Annotate with bridge info
            all_classes    = membership.get(str(idx), [class_id])
            is_bridge      = len(all_classes) > 1
            other_classes  = [c for c in all_classes if c != class_id]

            kw["is_bridge"]     = is_bridge
            kw["all_classes"]   = all_classes
            kw["other_classes"] = other_classes
            kw["is_representative"] = (idx == class_data["representative"])

            collated.append(kw)

        # Sort: representative first, then by chunk index
        collated.sort(key=lambda c: (0 if c.get("is_representative") else 1, c["chunk_index"]))

        out = {
            "class_id":    class_id,
            "label":       class_data["label"],
            "n_chunks":    len(collated),
            "n_missing":   len(missing),
            "representative_title": class_data["representative_title"],
            "chunks":      collated,
        }

        out_path = out_dir / f"class_{class_id}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        bridge_count = sum(1 for c in collated if c["is_bridge"])
        print(f"  {class_id} [{class_data['label'][:35]:35s}] "
              f"— {len(collated)} chunks, {bridge_count} bridges"
              + (f", {len(missing)} missing" if missing else ""))

    print(f"\nWritten to {out_dir}/")


if __name__ == "__main__":
    main()
