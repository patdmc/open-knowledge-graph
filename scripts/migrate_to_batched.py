#!/usr/bin/env python3
"""
migrate_to_batched.py — Consolidate per-file author/reference nodes into
range-named batch files.

Individual files:
  nodes/authors/AUTHOR-AbeS.yaml            (128,273 files)
  nodes/references/REF-Abe2004-3.yaml       (81,957 files)

Become:
  nodes/authors/authors_Abe-Cur.yaml        (~9 files, each ~10K nodes)
  nodes/references/references_Abe-Fra.yaml  (~8 files, each ~10K nodes)

Range names are auto-derived from the sorted first/last ID in each batch,
truncated to the first meaningful characters after the type prefix.

Usage:
    python scripts/migrate_to_batched.py [--dry-run] [--batch-size 10000]
    python scripts/migrate_to_batched.py --types authors             # authors only
    python scripts/migrate_to_batched.py --types references          # references only
    python scripts/migrate_to_batched.py --types authors,references  # both (default)
"""

import argparse
import multiprocessing
import re
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
NODES = REPO / "knowledge-graph" / "nodes"

DEFAULT_BATCH = 10_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(raw: str) -> str:
    allowed = {0x09, 0x0A, 0x0D}
    return "".join(ch for ch in raw if ord(ch) >= 0x20 or ord(ch) in allowed)


def _load_yaml(path: Path) -> dict | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = _sanitize(raw)
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            # Minimal fallback for malformed files
            data = {}
            m = re.search(r'^id:\s*(\S+)', raw, re.MULTILINE)
            if m:
                data["id"] = m.group(1)
            m = re.search(r'^type:\s*(\S+)', raw, re.MULTILINE)
            if m:
                data["type"] = m.group(1)
            m = re.search(r'^name:\s*(.+)', raw, re.MULTILINE)
            if m:
                data["name"] = m.group(1).strip().strip('"').strip("'")
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"  WARN load error {path.name}: {e}", file=sys.stderr)
        return None


def _load_yaml_worker(path_str: str) -> dict | None:
    return _load_yaml(Path(path_str))


def _id_stem(node_id: str, prefix: str, chars: int = 5) -> str:
    """Strip type prefix and return first `chars` chars of remaining ID."""
    stem = node_id
    if stem.upper().startswith(prefix.upper() + "-"):
        stem = stem[len(prefix) + 1:]
    # Take up to `chars` alphanumeric chars
    clean = re.sub(r"[^A-Za-z0-9]", "", stem)
    return clean[:chars] if len(clean) >= chars else clean


def _range_name(batch: list[dict], prefix: str, chars: int = 5) -> str:
    """Build a range name from first and last node ID in the batch."""
    first = _id_stem(batch[0].get("id", ""), prefix, chars)
    last  = _id_stem(batch[-1].get("id", ""), prefix, chars)
    # Fall back to full ID stem if short stem is empty
    if not first:
        first = re.sub(r"[^A-Za-z0-9]", "", str(batch[0].get("id", "")))[:chars] or "000"
    if not last:
        last  = re.sub(r"[^A-Za-z0-9]", "", str(batch[-1].get("id", "")))[:chars] or "zzz"
    if first == last:
        return first
    return f"{first}-{last}"


def _write_batch(path: Path, nodes: list[dict], dry_run: bool) -> None:
    """Write a list of node dicts as a YAML list file."""
    if dry_run:
        print(f"    WOULD WRITE  {path.name}  ({len(nodes):,} nodes)")
        return
    # Dump as a YAML list; allow_unicode keeps accented chars intact
    content = yaml.dump(
        nodes,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    path.write_text(content, encoding="utf-8")
    print(f"    WROTE  {path.name}  ({len(nodes):,} nodes,  {path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Core migration for one node type
# ---------------------------------------------------------------------------

def migrate_type(subdir: str, prefix: str, batch_size: int, dry_run: bool) -> None:
    src_dir = NODES / subdir
    if not src_dir.exists():
        print(f"  {subdir}: directory not found, skipping")
        return

    # Only process individual-node YAML files (not already-batched list files)
    individual = sorted(
        p for p in src_dir.glob("*.yaml")
        if not p.stem.startswith(f"{subdir}_")
    )
    if not individual:
        print(f"  {subdir}: no individual node files found")
        return

    print(f"\n{subdir}: loading {len(individual):,} files with {multiprocessing.cpu_count()} workers...")
    t0 = time.time()

    path_strs = [str(p) for p in individual]
    nodes: list[dict] = []
    errors = 0
    chunk = max(1, len(individual) // 20)

    with multiprocessing.Pool() as pool:
        for i, data in enumerate(pool.imap_unordered(_load_yaml_worker, path_strs, chunksize=128)):
            if data:
                nodes.append(data)
            else:
                errors += 1
            if (i + 1) % chunk == 0:
                print(f"  loaded {i+1:,}/{len(individual):,}...")

    print(f"  loaded {len(nodes):,} nodes in {time.time()-t0:.1f}s  ({errors} errors)")

    # Sort by ID for stable, predictable ranges
    nodes.sort(key=lambda n: str(n.get("id", "")).lower())

    # Split into batches
    batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]
    print(f"  splitting into {len(batches)} batch(es) of up to {batch_size:,} nodes each")

    # Write batch files
    for batch in batches:
        rng = _range_name(batch, prefix)
        out_path = src_dir / f"{subdir}_{rng}.yaml"
        _write_batch(out_path, batch, dry_run)

    # Remove individual files
    if not dry_run:
        print(f"  removing {len(individual):,} individual files...")
        removed = 0
        for path in individual:
            try:
                path.unlink()
                removed += 1
            except Exception as e:
                print(f"  WARN could not remove {path.name}: {e}", file=sys.stderr)
        print(f"  removed {removed:,} files")
    else:
        print(f"  WOULD REMOVE {len(individual):,} individual files")

    total_time = time.time() - t0
    print(f"  {subdir} done in {total_time:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Consolidate per-node YAML files into range-named batches.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing or deleting")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help=f"Nodes per batch file (default: {DEFAULT_BATCH})")
    parser.add_argument("--types", default="authors,references,definitions,theorems,emergent,equivalency,novel,overlap,open-questions",
                        help="Comma-separated list of node types to migrate (default: all)")
    args = parser.parse_args()

    TYPE_CONFIGS = {
        "authors":        ("authors",        "AUTHOR"),
        "references":     ("references",     "REF"),
        "definitions":    ("definitions",    "D"),
        "theorems":       ("theorems",       "T"),
        "emergent":       ("emergent",       "EM"),
        "equivalency":    ("equivalency",    "EC"),
        "novel":          ("novel",          "NV"),
        "overlap":        ("overlap",        "OV"),
        "open-questions": ("open-questions", "OQ"),
    }
    ALL_TYPES = ",".join(TYPE_CONFIGS.keys())

    requested = [t.strip() for t in args.types.split(",")]
    unknown = [t for t in requested if t not in TYPE_CONFIGS]
    if unknown:
        print(f"Unknown types: {unknown}. Valid: {list(TYPE_CONFIGS)}", file=sys.stderr)
        sys.exit(1)

    print(f"Batch size: {args.batch_size:,}  |  Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")

    for t in requested:
        subdir, prefix = TYPE_CONFIGS[t]
        migrate_type(subdir, prefix, args.batch_size, args.dry_run)

    print("\nMigration complete.")
    print("Next steps:")
    print("  1. Run: python scripts/yaml_to_tex.py          (tex archive handles list-format YAML)")
    print("  2. Run: python scripts/generate_keywords_index.py")


if __name__ == "__main__":
    main()
