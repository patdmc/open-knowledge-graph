#!/usr/bin/env python3
"""
arxiv_references_to_graph.py — Build citation edges from arXiv internal references.

Reads a JSONL file where each line is:
    {"paper_id": ["cited_id1", "cited_id2", ...]}

For each paper_id that has a REF node in the graph:
  - If the cited paper also has a REF node: adds a `cites` edge (paper → cited)
    and a `cited_by` edge (cited → paper)
  - If the cited paper is NOT in the graph: optionally creates a stub REF node
    (use --create-stubs)

Run AFTER arxiv_to_graph.py so REF nodes exist to link against.

Usage:
    python scripts/arxiv_references_to_graph.py <references.jsonl> [options]

Options:
    --ref-dir        Directory containing REF YAML nodes
                     Default: knowledge-graph/nodes/references/
    --create-stubs   Create minimal stub REF nodes for cited papers not in graph
    --dry-run        Print what would happen without writing files
    --limit          Stop after N lines

Examples:
    python scripts/arxiv_references_to_graph.py arxiv-refs.jsonl
    python scripts/arxiv_references_to_graph.py arxiv-refs.jsonl --create-stubs
    python scripts/arxiv_references_to_graph.py arxiv-refs.jsonl --dry-run --limit 50000
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path


def yaml_dq_escape(s: str) -> str:
    """Escape a string for safe use inside a YAML double-quoted scalar."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\x7f", "")


def _clean_str(s) -> str:
    """Strip control characters (except TAB/LF/CR) from a string value."""
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    allowed = {0x09, 0x0A, 0x0D}
    return "".join(ch for ch in s if ord(ch) >= 0x20 or ord(ch) in allowed)

TODAY = date.today().isoformat()
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_REF_DIR = REPO_ROOT / "knowledge-graph" / "nodes" / "references"


# ---------------------------------------------------------------------------
# arXiv ID normalisation
# ---------------------------------------------------------------------------

def normalise_arxiv_id(raw: str) -> str:
    """
    Return a canonical arXiv ID string.
    Old format:  hep-th/9212071  -> hep-th/9212071  (keep as-is)
    New format:  2301.04567      -> 2301.04567       (keep as-is)
    Handles minor variants like leading/trailing whitespace.
    """
    return raw.strip()


def node_stem_from_arxiv_id(arxiv_id: str) -> str:
    """
    Derive a filesystem-safe stem for use in filenames when creating stubs.
    hep-th/9212071 -> hep-th_9212071
    2301.04567     -> 2301.04567
    """
    return arxiv_id.replace("/", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph_arxiv_ids(ref_dir: Path) -> dict[str, str]:
    """
    Scan REF YAML files for arxiv_id in provenance.
    Returns {normalised_arxiv_id: node_id}.
    """
    arxiv_to_node = {}
    for yaml_path in ref_dir.glob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        m = re.search(r"arxiv_id:\s*[\"']?([^\s\"'\n]+)[\"']?", text)
        if m:
            aid = normalise_arxiv_id(m.group(1))
            if aid:
                arxiv_to_node[aid] = yaml_path.stem
    return arxiv_to_node


# ---------------------------------------------------------------------------
# Edge writing
# ---------------------------------------------------------------------------

def edge_already_exists(yaml_text: str, target_node_id: str, relation: str) -> bool:
    # Simple check: both strings appear in the file
    return target_node_id in yaml_text and relation in yaml_text


def append_edge(yaml_path: Path, target_node_id: str, relation: str,
                description: str, dry_run: bool) -> bool:
    """Append an edge to a REF or stub YAML. Returns True if modified."""
    text = yaml_path.read_text(encoding="utf-8")
    if edge_already_exists(text, target_node_id, relation):
        return False

    edge_block = f"""  - to: {target_node_id}
    relation: {relation}
    provenance:
      attribution:
        author: "arxiv_references_to_graph.py"
        source: "arXiv internal references"
        date: "{TODAY}"
      evidence:
        type: cited
        description: "{description}"
      derivation:
        method: "Extracted from arXiv references JSONL"
"""
    if dry_run:
        return True

    if re.search(r"^edges:\s*\[\]", text, re.MULTILINE):
        text = re.sub(r"^edges:\s*\[\]", f"edges:\n{edge_block}", text, flags=re.MULTILINE)
    elif re.search(r"^edges:", text, re.MULTILINE):
        text = text.rstrip() + f"\n{edge_block}"
    else:
        text = text.rstrip() + f"\nedges:\n{edge_block}"

    yaml_path.write_text(text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Stub creation
# ---------------------------------------------------------------------------

def make_stub_node_id(arxiv_id: str, existing_ids: set) -> str:
    stem = node_stem_from_arxiv_id(arxiv_id)
    base = f"REF-Arxiv-{stem}"
    node_id = base
    suffix = 2
    while node_id in existing_ids:
        node_id = f"{base}-{suffix}"
        suffix += 1
    return node_id


def build_stub_yaml(node_id: str, arxiv_id: str) -> str:
    url = (
        f"https://doi.org/{arxiv_id}"
        if not ("/" in arxiv_id or "." in arxiv_id)
        else f"https://arxiv.org/abs/{arxiv_id}"
    )
    return f"""id: {node_id}
type: reference
fidelity: stub
name: "arXiv:{arxiv_id}"
statement: |
  Stub node — cited by a paper in the graph.
  arXiv ID: {arxiv_id}
  Full metadata not yet imported. Run arxiv_to_graph.py to populate.

provenance:
  attribution:
    author: "arxiv_references_to_graph.py"
    source: "arXiv internal references"
    date: "{TODAY}"
    arxiv_id: "{arxiv_id}"
    url: "{url}"
  evidence:
    type: cited
    description: "Stub created because this paper is cited by a graph member."
    references: []
  derivation:
    from: []
    method: "Stub created by arxiv_references_to_graph.py --create-stubs"

edges: []
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build citation edges from arXiv references JSONL."
    )
    parser.add_argument("input", help="Path to arXiv references JSONL")
    parser.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR))
    parser.add_argument("--create-stubs", action="store_true",
                        help="Create stub REF nodes for cited papers not yet in graph")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load existing graph
    print(f"Scanning REF nodes in {ref_dir} ...")
    arxiv_to_node = load_graph_arxiv_ids(ref_dir)
    existing_node_ids = {v for v in arxiv_to_node.values()} | \
                        {f.stem for f in ref_dir.glob("*.yaml")}
    print(f"  {len(arxiv_to_node)} REF nodes with arXiv IDs")
    print()

    if not arxiv_to_node:
        print("No REF nodes with arXiv IDs. Run arxiv_to_graph.py first.")
        sys.exit(0)

    lines_read = 0
    lines_matched = 0       # citing paper is in graph
    edges_added = 0
    edges_skipped = 0
    stubs_created = 0
    errors = 0

    print(f"Streaming {input_path} ...")

    with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if args.limit and lines_read >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            lines_read += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            # Each line: {paper_id: [cited_id, ...]}
            for paper_id, cited_list in record.items():
                paper_id = normalise_arxiv_id(paper_id)

                if paper_id not in arxiv_to_node:
                    continue  # citing paper not in graph — skip

                lines_matched += 1
                paper_node_id = arxiv_to_node[paper_id]
                paper_yaml = ref_dir / f"{paper_node_id}.yaml"

                for cited_raw in cited_list:
                    cited_id = normalise_arxiv_id(cited_raw)
                    cited_node_id = arxiv_to_node.get(cited_id)

                    # If cited paper not in graph, optionally create stub
                    if cited_node_id is None:
                        if not args.create_stubs:
                            continue
                        cited_node_id = make_stub_node_id(cited_id, existing_node_ids)
                        stub_path = ref_dir / f"{cited_node_id}.yaml"
                        if not stub_path.exists():
                            if args.dry_run:
                                print(f"  STUB           {cited_node_id}  ({cited_id})")
                            else:
                                stub_path.write_text(build_stub_yaml(cited_node_id, cited_id))
                                print(f"  STUB CREATED   {cited_node_id}  ({cited_id})")
                            existing_node_ids.add(cited_node_id)
                            arxiv_to_node[cited_id] = cited_node_id
                            stubs_created += 1

                    # Add cites edge: paper → cited
                    cited_yaml = ref_dir / f"{cited_node_id}.yaml"
                    modified = append_edge(
                        paper_yaml,
                        cited_node_id,
                        "cites",
                        f"{paper_node_id} cites {cited_node_id} per arXiv reference data",
                        args.dry_run,
                    )
                    if modified:
                        if args.dry_run:
                            print(f"  EDGE           {paper_node_id} --cites--> {cited_node_id}")
                        edges_added += 1
                    else:
                        edges_skipped += 1

                    # Add cited_by edge: cited → paper (only if cited node exists on disk)
                    if cited_yaml.exists() or (args.create_stubs and not args.dry_run):
                        modified = append_edge(
                            cited_yaml,
                            paper_node_id,
                            "cited_by",
                            f"{cited_node_id} is cited by {paper_node_id}",
                            args.dry_run,
                        )
                        if modified:
                            if args.dry_run:
                                print(f"  EDGE           {cited_node_id} --cited_by--> {paper_node_id}")
                            edges_added += 1
                        else:
                            edges_skipped += 1

    print()
    print("=" * 60)
    print(f"Lines read:          {lines_read}")
    print(f"Citing papers matched: {lines_matched}")
    print(f"Stubs {'would create' if args.dry_run else 'created'}:  {stubs_created}")
    print(f"Edges {'would add' if args.dry_run else 'added'}:    {edges_added}")
    print(f"Edges skipped:       {edges_skipped}")
    if errors:
        print(f"Parse errors:        {errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()
