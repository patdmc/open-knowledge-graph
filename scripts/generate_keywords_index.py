#!/usr/bin/env python3
"""
generate_keywords_index.py — Derive keywords for every knowledge-graph node.

Keywords are derived from:
  1. Node type
  2. Evidence type (provenance.evidence.type)
  3. Edge relation names and target-node prefixes
  4. Framework/concept mentions in statement text
  5. arXiv categories (for reference nodes)

Outputs:
  tex-archive/keywords_index.jsonl   — one JSON line per node: {id, type, keywords}
  tex-archive/keywords_by_tag.json   — inverted index: tag → [node_ids]  (for O(1) group lookup)

Usage:
    python scripts/generate_keywords_index.py
    python scripts/generate_keywords_index.py --nodes-dir knowledge-graph/nodes --out-dir tex-archive
"""

import argparse
import json
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
DEFAULT_NODES = REPO / "knowledge-graph" / "nodes"
DEFAULT_OUT = REPO / "tex-archive"

# ---------------------------------------------------------------------------
# Control-char sanitiser (same as yaml_to_tex.py)
# ---------------------------------------------------------------------------

def _sanitize(raw: str) -> str:
    allowed = {0x09, 0x0A, 0x0D}
    return "".join(ch for ch in raw if ord(ch) >= 0x20 or ord(ch) in allowed)


# ---------------------------------------------------------------------------
# Framework / concept vocabulary (text → tag)
# Order matters: longer phrases first to avoid partial matches
# ---------------------------------------------------------------------------

VOCAB: list[tuple[str, str]] = [
    # Frameworks
    ("free energy principle", "framework:FEP"),
    ("variational free energy", "framework:FEP"),
    ("active inference", "framework:FEP"),
    ("friston",             "framework:FEP"),
    ("fep",                 "framework:FEP"),
    ("global workspace",    "framework:GWT"),
    ("baars",               "framework:GWT"),
    ("dehaene",             "framework:GWT"),
    ("kahneman",            "framework:Kahneman"),
    ("system 2",            "framework:Kahneman"),
    ("tishby",              "framework:IB"),
    ("information bottleneck", "framework:IB"),
    ("schmidhuber",         "framework:Schmidhuber"),
    ("tarski",              "framework:Tarski"),
    ("peirce",              "framework:Peirce"),
    ("pragmatism",          "framework:Peirce"),
    ("miller",              "framework:Miller"),
    ("chunking",            "framework:Miller"),
    ("bekenstein",          "framework:HolographicBound"),
    ("holographic bound",   "framework:HolographicBound"),
    ("polchinski",          "framework:HolographicBound"),
    ("cover and thomas",    "framework:InformationTheory"),
    ("data processing inequality", "framework:InformationTheory"),
    ("kolmogorov",          "framework:Kolmogorov"),
    ("mdl",                 "framework:MDL"),
    ("minimum description length", "framework:MDL"),
    # Concepts
    ("uncertainty",         "concept:uncertainty"),
    ("entropy",             "concept:entropy"),
    ("knowledge graph",     "concept:knowledge-graph"),
    ("graph structure",     "concept:graph"),
    ("graph construction",  "concept:graph"),
    (" graph ",             "concept:graph"),
    ("compression",         "concept:compression"),
    ("encoding",            "concept:encoding"),
    ("gradient descent",    "concept:gradient-descent"),
    ("bounded context",     "concept:bounded-context"),
    ("context window",      "concept:bounded-context"),
    ("c_n",                 "concept:bounded-context"),
    ("agency",              "concept:agency"),
    ("survival",            "concept:survival"),
    ("selection pressure",  "concept:selection"),
    ("information loss",    "concept:information-loss"),
    ("confidence",          "concept:confidence"),
    ("novelty",             "concept:novelty"),
    ("equivalen",           "concept:equivalence"),     # matches equivalency/equivalent
    # Domains
    ("pangenome",           "domain:biology"),
    ("genome",              "domain:biology"),
    ("de bruijn",           "domain:biology"),
    ("neuroscience",        "domain:neuroscience"),
    ("neural",              "domain:neuroscience"),
    ("quantum",             "domain:physics"),
    ("thermodynamic",       "domain:physics"),
    ("boltzmann",           "domain:physics"),
    ("consciousness",       "domain:consciousness"),
    ("language",            "domain:language"),
    ("semantic",            "domain:language"),
    ("mathematics",         "domain:math"),
    ("formal proof",        "domain:math"),
    # Sources
    ("arxiv",               "source:arxiv"),
    ("doi",                 "source:doi"),
]


# ---------------------------------------------------------------------------
# Keyword extractor
# ---------------------------------------------------------------------------

def extract_keywords(data: dict) -> list[str]:
    kws: set[str] = set()

    # 1. Node type
    node_type = str(data.get("type") or "")
    if node_type:
        kws.add(f"type:{node_type}")

    # 2. Evidence type
    prov = data.get("provenance") or {}
    if isinstance(prov, dict):
        ev = prov.get("evidence") or {}
        if isinstance(ev, dict):
            ev_type = ev.get("type") or ""
            if ev_type:
                kws.add(f"evidence:{ev_type}")

    # 3. Fidelity (for reference nodes)
    fidelity = data.get("fidelity") or ""
    if fidelity:
        kws.add(f"fidelity:{fidelity}")

    # 4. arXiv categories (reference nodes)
    stmt = str(data.get("statement") or "")
    cat_match = re.search(r"Categories:\s*([^\n]+)", stmt)
    if cat_match:
        for cat in cat_match.group(1).split(","):
            cat = cat.strip()
            if cat:
                # top-level prefix only (e.g. cs.AI → cs)
                top = cat.split(".")[0]
                kws.add(f"arxiv-cat:{top}")
                kws.add(f"arxiv-cat:{cat}")

    # 5. Edges: relation names and target-ID prefixes
    for edge in (data.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        relation = edge.get("relation") or ""
        if relation:
            kws.add(f"rel:{relation}")
        target = str(edge.get("to") or "")
        m = re.match(r"^([A-Z]+)", target)
        if m:
            kws.add(f"links:{m.group(1)}")

    # 6. Framework / concept vocabulary scan
    # Collect all text to scan
    text_parts = [
        stmt,
        str(data.get("underlying_proposition") or ""),
        str(data.get("equivalence_analysis") or ""),
        str(data.get("name") or ""),
    ]
    # For equivalency: member formulations
    for member in (data.get("members") or []):
        if isinstance(member, dict):
            text_parts.append(str(member.get("formulation") or ""))
            text_parts.append(str(member.get("framework") or ""))
    text = " ".join(text_parts).lower()

    for phrase, tag in VOCAB:
        if phrase in text:
            kws.add(tag)

    return sorted(kws)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _parse_yaml_file(yaml_path: Path):
    """Parse a YAML file; return a list of node dicts (handles both single and batch format)."""
    try:
        raw = yaml_path.read_text(encoding="utf-8", errors="replace")
        raw = _sanitize(raw)
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            # Minimal fallback for malformed single-node files
            data = {}
            m = re.search(r'^id:\s*(\S+)', raw, re.MULTILINE)
            if m:
                data["id"] = m.group(1)
            m = re.search(r'^type:\s*(\S+)', raw, re.MULTILINE)
            if m:
                data["type"] = m.group(1)
            m = re.search(r'^statement:\s*\|?\s*\n(.*)', raw, re.MULTILINE)
            if m:
                data["statement"] = m.group(1)

        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
        return []
    except Exception:
        return []


def process_file(yaml_path: Path) -> list[dict]:
    results = []
    for data in _parse_yaml_file(yaml_path):
        node_id = data.get("id") or yaml_path.stem
        node_type = data.get("type") or ""
        keywords = extract_keywords(data)
        results.append({"id": node_id, "type": node_type, "keywords": keywords})
    return results


def _worker(path_str: str) -> list[dict]:
    return process_file(Path(path_str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate keywords index for all graph nodes.")
    parser.add_argument("--nodes-dir", default=str(DEFAULT_NODES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    nodes_dir = Path(args.nodes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = list(nodes_dir.rglob("*.yaml"))
    total = len(yaml_files)
    print(f"Found {total:,} YAML files. Extracting keywords with {multiprocessing.cpu_count()} workers...")

    t0 = time.time()
    records: list[dict] = []
    inverted: dict[str, list[str]] = {}

    path_strs = [str(p) for p in yaml_files]
    chunk = max(1, total // 100)

    with multiprocessing.Pool() as pool:
        for i, batch in enumerate(pool.imap_unordered(_worker, path_strs, chunksize=128)):
            for result in (batch or []):
                records.append(result)
                for kw in result["keywords"]:
                    inverted.setdefault(kw, []).append(result["id"])
            if (i + 1) % chunk == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                rem = (total - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1:>9,}/{total:,}  {elapsed:.0f}s elapsed  ~{rem:.0f}s remaining", flush=True)

    # Write keywords_index.jsonl — one record per node
    index_path = out_dir / "keywords_index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in sorted(records, key=lambda r: r["id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write keywords_by_tag.json — inverted index, sorted by tag
    # Sort node lists for determinism
    tag_path = out_dir / "keywords_by_tag.json"
    with open(tag_path, "w", encoding="utf-8") as f:
        sorted_inv = {k: sorted(v) for k, v in sorted(inverted.items())}
        json.dump(sorted_inv, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {len(records):,} nodes indexed")
    print(f"  {len(inverted):,} unique tags")
    print(f"  keywords_index.jsonl  ({index_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  keywords_by_tag.json  ({tag_path.stat().st_size / 1e6:.1f} MB)")

    # Print tag summary (top 30 by node count)
    print("\nTop 30 tags by node count:")
    top = sorted(inverted.items(), key=lambda x: len(x[1]), reverse=True)[:30]
    for tag, ids in top:
        print(f"  {len(ids):>8,}  {tag}")


if __name__ == "__main__":
    main()
