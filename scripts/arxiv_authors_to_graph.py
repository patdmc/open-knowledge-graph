#!/usr/bin/env python3
"""
arxiv_authors_to_graph.py — Extract author nodes from arXiv authors JSONL.

Reads a JSONL file where each line is:
    {"arxiv_id": [["LastName", "FirstInitial", "Suffix"], ...]}

For each arXiv ID that already has a REF node in the graph, creates:
  - AUTHOR-{LastName}{FirstInitial} nodes in knowledge-graph/nodes/authors/
  - Edges: AUTHOR → REF paper (authored)
  - Edges added to existing REF nodes: REF → AUTHOR (authored_by)

Run AFTER arxiv_to_graph.py so REF nodes exist to link against.

Usage:
    python scripts/arxiv_authors_to_graph.py <authors.jsonl> [options]

Options:
    --ref-dir       Directory containing REF YAML nodes
                    Default: knowledge-graph/nodes/references/
    --out-dir       Output directory for AUTHOR YAML nodes
                    Default: knowledge-graph/nodes/authors/
    --dry-run       Print what would be created without writing files
    --limit         Stop after N lines (useful for testing)

Examples:
    python scripts/arxiv_authors_to_graph.py arxiv-authors.jsonl
    python scripts/arxiv_authors_to_graph.py arxiv-authors.jsonl --dry-run --limit 10000
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path


def yaml_dq_escape(s: str) -> str:
    """Escape a string for safe use inside a YAML double-quoted scalar."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\x7f", "")


_TEX_ACCENTS: dict[str, dict[str, str]] = {
    '"': {'a':'ä','e':'ë','i':'ï','o':'ö','u':'ü','y':'ÿ','A':'Ä','E':'Ë','I':'Ï','O':'Ö','U':'Ü','Y':'Ÿ'},
    "'": {'a':'á','e':'é','i':'í','o':'ó','u':'ú','y':'ý','c':'ć','n':'ń','s':'ś','z':'ź',
          'A':'Á','E':'É','I':'Í','O':'Ó','U':'Ú','Y':'Ý','C':'Ć','N':'Ń','S':'Ś','Z':'Ź'},
    '`': {'a':'à','e':'è','i':'ì','o':'ò','u':'ù','A':'À','E':'È','I':'Ì','O':'Ò','U':'Ù'},
    '^': {'a':'â','e':'ê','i':'î','o':'ô','u':'û','A':'Â','E':'Ê','I':'Î','O':'Ô','U':'Û'},
    '~': {'a':'ã','n':'ñ','o':'õ','A':'Ã','N':'Ñ','O':'Õ'},
    '=': {'a':'ā','e':'ē','i':'ī','o':'ō','u':'ū','ı':'ī',  # ı = dotless-i (U+0131)
          'A':'Ā','E':'Ē','I':'Ī','O':'Ō','U':'Ū'},
    '.': {'a':'ȧ','e':'ė','o':'ȯ','z':'ż','A':'Ȧ','E':'Ė','O':'Ȯ','Z':'Ż'},
    'u': {'a':'ă','e':'ĕ','i':'ĭ','o':'ŏ','u':'ŭ','A':'Ă','E':'Ĕ','I':'Ĭ','O':'Ŏ','U':'Ŭ'},
    'v': {'c':'č','n':'ň','s':'š','z':'ž','e':'ě','r':'ř','C':'Č','N':'Ň','S':'Š','Z':'Ž','E':'Ě','R':'Ř'},
    'H': {'o':'ő','u':'ű','O':'Ő','U':'Ű'},
    'k': {'a':'ą','e':'ę','A':'Ą','E':'Ę'},
    'c': {'c':'ç','s':'ş','C':'Ç','S':'Ş'},
}
# Longest first to avoid prefix collisions (e.g. \oe before \o)
_TEX_STANDALONE: list[tuple[str, str]] = sorted([
    (r'\ss', 'ß'), (r'\SS', 'ẞ'),
    (r'\ae', 'æ'), (r'\AE', 'Æ'),
    (r'\oe', 'œ'), (r'\OE', 'Œ'),
    (r'\aa', 'å'), (r'\AA', 'Å'),
    (r'\dj', 'đ'), (r'\DJ', 'Đ'),
    (r'\ng', 'ŋ'), (r'\NG', 'Ŋ'),
    (r'\th', 'þ'), (r'\TH', 'Þ'),
    (r'\o',  'ø'), (r'\O',  'Ø'),
    (r'\l',  'ł'), (r'\L',  'Ł'),
    (r'\i',  'ı'), (r'\j',  'ȷ'),
], key=lambda x: -len(x[0]))


def _decode_tex(s: str) -> str:
    """Decode TeX accent commands to Unicode. E.g. \\\"u → ü, \\=i → ī."""
    if '\\' not in s and '"' not in s:
        return s
    # Full TeX escape sequences (backslash present)
    if '\\' in s:
        for acc, mapping in _TEX_ACCENTS.items():
            for letter, uni in mapping.items():
                s = s.replace(f'\\{acc}{{{letter}}}', uni)  # \"{u}
                s = s.replace(f'\\{acc}{letter}', uni)       # \"u
        for tex, uni in _TEX_STANDALONE:
            s = s.replace(tex, uni)
        # Strip remaining \cmd{...} wrappers, keep inner content
        s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\{([^}]*)\}', r'\1', s)
    # Second pass: bare "X umlaut patterns (backslash dropped in source JSON).
    # Only replace when the " is immediately followed by a known umlaut letter,
    # and the " is not at word boundary (i.e. preceded by a letter) — avoids
    # munging legitimate quoted nicknames like "Honza".
    if '"' in s:
        _BARE_UMLAUT = str.maketrans({
            # These are only safe mid-word; we use a regex to be precise
        })
        s = re.sub(
            r'(?<=[A-Za-zÀ-ÿ])"([aeiouAEIOUaäeëiïoöuü])',
            lambda m: _TEX_ACCENTS['"'].get(m.group(1), m.group(1)),
            s,
        )
    return s


def _clean_str(s) -> str:
    """Decode TeX escapes, strip control characters, normalize whitespace."""
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    s = _decode_tex(s)
    allowed = {0x09, 0x0A, 0x0D}
    return "".join(ch for ch in s if ord(ch) >= 0x20 or ord(ch) in allowed)

TODAY = date.today().isoformat()
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_REF_DIR = REPO_ROOT / "knowledge-graph" / "nodes" / "references"
DEFAULT_OUT_DIR = REPO_ROOT / "knowledge-graph" / "nodes" / "authors"


# ---------------------------------------------------------------------------
# Batch file helpers
# ---------------------------------------------------------------------------

def _id_stem(node_id: str, prefix: str, chars: int = 5) -> str:
    import re as _re
    stem = node_id
    if stem.upper().startswith(prefix.upper() + "-"):
        stem = stem[len(prefix) + 1:]
    clean = _re.sub(r"[^A-Za-z0-9]", "", stem)
    return clean[:chars] if len(clean) >= chars else clean


def load_existing_author_ids(out_dir: Path) -> set[str]:
    """Collect all AUTHOR node IDs already present (batch or individual files)."""
    existing: set[str] = set()
    for path in out_dir.glob("*.yaml"):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            data = yaml.safe_load(raw)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("id"):
                        existing.add(item["id"])
            elif isinstance(data, dict) and data.get("id"):
                existing.add(data["id"])
        except Exception:
            pass
    return existing


def write_author_batch(nodes: list[dict], out_dir: Path, batch_size: int = 10_000) -> None:
    """Sort nodes by ID, split into range-named batch files, write YAML lists."""
    nodes_sorted = sorted(nodes, key=lambda n: str(n.get("id", "")).lower())
    for i in range(0, len(nodes_sorted), batch_size):
        batch = nodes_sorted[i:i + batch_size]
        first = _id_stem(batch[0].get("id", ""), "AUTHOR") or re.sub(r"[^A-Za-z0-9]", "", str(batch[0].get("id", "")))[:5] or "000"
        last  = _id_stem(batch[-1].get("id", ""), "AUTHOR") or re.sub(r"[^A-Za-z0-9]", "", str(batch[-1].get("id", "")))[:5] or "zzz"
        rng = first if first == last else f"{first}-{last}"
        out_path = out_dir / f"authors_{rng}.yaml"
        content = yaml.dump(batch, allow_unicode=True, default_flow_style=False,
                            sort_keys=False, width=120)
        out_path.write_text(content, encoding="utf-8")
        print(f"  WROTE  {out_path.name}  ({len(batch):,} nodes,  {out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_graph_arxiv_ids(ref_dir: Path) -> dict[str, str]:
    """
    Scan REF YAML files for arxiv_id in provenance.
    Returns {arxiv_id: node_id} for all REF nodes that have an arxiv_id.
    """
    arxiv_to_node = {}
    for yaml_path in ref_dir.glob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        m = re.search(r"arxiv_id:\s*[\"']?([^\s\"']+)[\"']?", text)
        if m:
            arxiv_id = m.group(1).strip()
            if arxiv_id and arxiv_id != '""' and arxiv_id != "''":
                arxiv_to_node[arxiv_id] = yaml_path.stem
    return arxiv_to_node


def make_author_id(lastname: str, first: str, existing: set) -> str:
    """
    Generate AUTHOR-{LastName}{FirstInitial} with disambiguation suffix.
    e.g. AUTHOR-FristonK, AUTHOR-SmithJ, AUTHOR-SmithJ-2
    """
    ln = re.sub(r"[^A-Za-z0-9]", "", lastname)
    fi = re.sub(r"[^A-Za-z0-9]", "", first[:1]) if first else ""
    base = f"AUTHOR-{ln}{fi}"
    node_id = base
    suffix = 2
    while node_id in existing:
        node_id = f"{base}-{suffix}"
        suffix += 1
    return node_id


def format_full_name(lastname: str, first: str, suffix: str) -> str:
    parts = []
    if first:
        parts.append(first)
    if lastname:
        parts.append(lastname)
    if suffix:
        parts.append(suffix)
    return " ".join(parts) if parts else "Unknown"


def build_author_yaml(node_id: str, lastname: str, first: str, suffix: str,
                       paper_node_ids: list[str]) -> str:
    full_name = format_full_name(lastname, first, suffix)
    papers_list = "\n".join(f"    - {pid}" for pid in sorted(paper_node_ids))

    edges_block = ""
    for pid in sorted(paper_node_ids):
        edges_block += f"""  - to: {pid}
    relation: authored
    provenance:
      attribution:
        author: "arxiv_authors_to_graph.py"
        source: "arXiv authors metadata"
        date: "{TODAY}"
      evidence:
        type: cited
        description: "{yaml_dq_escape(full_name)} is listed as author of {pid} in arXiv metadata"
      derivation:
        method: "Extracted from arXiv authors JSONL"
"""

    return f"""id: {node_id}
type: author
name: "{yaml_dq_escape(full_name)}"
statement: |
  Researcher: {full_name}
  Papers in graph:
{papers_list}

  NOTE: Bio and affiliations not available from arXiv metadata alone.
  Statement to be updated after reviewing the author's work.

provenance:
  attribution:
    author: "arxiv_authors_to_graph.py"
    source: "arXiv authors metadata"
    date: "{TODAY}"
  evidence:
    type: cited
    description: |
      Author node extracted from arXiv structured author data.
      Linked to {len(paper_node_ids)} paper(s) in the knowledge graph.
    references: {json.dumps(sorted(paper_node_ids))}
  derivation:
    from: {json.dumps(sorted(paper_node_ids))}
    method: "Parsed from arXiv authors JSONL via arxiv_authors_to_graph.py"

edges:
{edges_block}"""


def build_author_dict(node_id: str, lastname: str, first: str, suffix: str,
                       paper_node_ids: list[str]) -> dict:
    """Build an author node as a plain dict (for batch YAML writing)."""
    full_name = format_full_name(lastname, first, suffix)
    edges = [
        {
            "to": pid,
            "relation": "authored",
            "provenance": {
                "attribution": {
                    "author": "arxiv_authors_to_graph.py",
                    "source": "arXiv authors metadata",
                    "date": TODAY,
                },
                "evidence": {
                    "type": "cited",
                    "description": f"{full_name} is listed as author of {pid} in arXiv metadata",
                },
                "derivation": {"method": "Extracted from arXiv authors JSONL"},
            },
        }
        for pid in sorted(paper_node_ids)
    ]
    return {
        "id": node_id,
        "type": "author",
        "name": full_name,
        "statement": (
            f"Researcher: {full_name}\n"
            f"Papers in graph:\n"
            + "\n".join(f"  - {pid}" for pid in sorted(paper_node_ids))
            + "\n\nNOTE: Bio and affiliations not available from arXiv metadata alone."
        ),
        "provenance": {
            "attribution": {
                "author": "arxiv_authors_to_graph.py",
                "source": "arXiv authors metadata",
                "date": TODAY,
            },
            "evidence": {
                "type": "cited",
                "description": f"Author node extracted from arXiv structured author data. Linked to {len(paper_node_ids)} paper(s).",
                "references": sorted(paper_node_ids),
            },
            "derivation": {
                "from": sorted(paper_node_ids),
                "method": "Parsed from arXiv authors JSONL via arxiv_authors_to_graph.py",
            },
        },
        "edges": edges,
    }


def add_authored_by_edge(ref_yaml_path: Path, author_node_id: str,
                          full_name: str, dry_run: bool) -> bool:
    """
    Append an authored_by edge to an existing REF node YAML if not already present.
    Returns True if the file was (or would be) modified.
    """
    text = ref_yaml_path.read_text(encoding="utf-8")

    # Skip if edge already exists
    if author_node_id in text:
        return False

    edge_block = f"""  - to: {author_node_id}
    relation: authored_by
    provenance:
      attribution:
        author: "arxiv_authors_to_graph.py"
        source: "arXiv authors metadata"
        date: "{TODAY}"
      evidence:
        type: cited
        description: "{full_name} is listed as author of this paper in arXiv metadata"
      derivation:
        method: "Extracted from arXiv authors JSONL"
"""

    if dry_run:
        return True

    # If the file has an "edges: []" placeholder, replace it; otherwise append
    if re.search(r"^edges:\s*\[\]", text, re.MULTILINE):
        text = re.sub(r"^edges:\s*\[\]", f"edges:\n{edge_block}", text, flags=re.MULTILINE)
    elif re.search(r"^edges:", text, re.MULTILINE):
        # Append after the last edge block — find end of edges section
        text = text.rstrip() + f"\n{edge_block}"
    else:
        text = text.rstrip() + f"\nedges:\n{edge_block}"

    ref_yaml_path.write_text(text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build AUTHOR nodes from arXiv authors JSONL."
    )
    parser.add_argument("input", help="Path to arXiv authors JSONL file")
    parser.add_argument(
        "--ref-dir",
        default=str(DEFAULT_REF_DIR),
        help=f"REF nodes directory (default: {DEFAULT_REF_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory for AUTHOR nodes (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N lines")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    out_dir = Path(args.out_dir)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load arXiv IDs already in the graph
    print(f"Scanning REF nodes in {ref_dir} ...")
    arxiv_to_node = load_graph_arxiv_ids(ref_dir)
    print(f"  Found {len(arxiv_to_node)} REF nodes with arXiv IDs")

    if not arxiv_to_node:
        print("No REF nodes with arXiv IDs found. Run arxiv_to_graph.py first.")
        sys.exit(0)

    # Step 2: collect existing AUTHOR node IDs for disambiguation
    existing_author_ids = set()
    if out_dir.exists():
        existing_author_ids = {f.stem for f in out_dir.glob("*.yaml")}

    # Step 3: stream the authors file, collecting data for matched papers
    # Structure: {(lastname, first, suffix): [paper_node_id, ...]}
    author_papers: dict[tuple, list[str]] = defaultdict(list)
    # And: {paper_node_id: [(lastname, first, suffix), ...]} for back-edges
    paper_authors: dict[str, list[tuple]] = defaultdict(list)

    print(f"Streaming {input_path} ...")
    lines_read = 0
    lines_matched = 0
    lines_errored = 0

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
            except json.JSONDecodeError as e:
                lines_errored += 1
                continue

            # Each line: {arxiv_id: [[last, first, suffix], ...]}
            for arxiv_id, author_list in record.items():
                if arxiv_id not in arxiv_to_node:
                    continue

                lines_matched += 1
                paper_node_id = arxiv_to_node[arxiv_id]

                for entry in author_list:
                    # entry is [LastName, FirstInitial, Suffix]
                    if not isinstance(entry, list) or len(entry) < 2:
                        continue
                    lastname = _clean_str((entry[0] or "").strip())
                    first = _clean_str((entry[1] or "").strip())
                    suffix = _clean_str((entry[2] or "").strip()) if len(entry) > 2 else ""

                    if not lastname:
                        continue

                    key = (lastname, first, suffix)
                    if paper_node_id not in author_papers[key]:
                        author_papers[key].append(paper_node_id)
                    if key not in paper_authors[paper_node_id]:
                        paper_authors[paper_node_id].append(key)

    print(f"  Lines read:    {lines_read}")
    print(f"  Papers matched: {lines_matched}")
    print(f"  Unique authors: {len(author_papers)}")
    if lines_errored:
        print(f"  Parse errors:  {lines_errored}")
    print()

    if not author_papers:
        print("No matching authors found. Check that the arXiv IDs in the authors file")
        print("match the arxiv_id fields in your REF nodes.")
        sys.exit(0)

    # Step 4: build a name→node_id mapping for deduplication
    # Authors with identical (lastname, first) across different papers are the same node
    # Key for dedup: (lastname, first) — ignore suffix for matching
    name_to_node_id: dict[tuple, str] = {}
    author_node_map: dict[tuple, str] = {}  # (last, first, suffix) → node_id

    for key in author_papers:
        lastname, first, suffix = key
        dedup_key = (lastname, first)
        if dedup_key not in name_to_node_id:
            node_id = make_author_id(lastname, first, existing_author_ids)
            existing_author_ids.add(node_id)
            name_to_node_id[dedup_key] = node_id
        author_node_map[key] = name_to_node_id[dedup_key]

    # Merge paper lists for authors that share the same node_id
    merged: dict[str, dict] = {}  # node_id → {key, papers}
    for key, paper_list in author_papers.items():
        node_id = author_node_map[key]
        if node_id not in merged:
            merged[node_id] = {"key": key, "papers": set()}
        merged[node_id]["papers"].update(paper_list)

    # Step 5: write AUTHOR nodes (batched)
    print(f"Scanning {out_dir} for existing author IDs ...")
    existing_in_batches = load_existing_author_ids(out_dir)
    print(f"  {len(existing_in_batches):,} existing author IDs found")

    print(f"Writing {len(merged)} AUTHOR node(s) to {out_dir} ...")
    authors_created = 0
    authors_skipped = 0
    new_nodes: list[dict] = []

    for node_id, data in sorted(merged.items()):
        lastname, first, suffix = data["key"]
        paper_list = sorted(data["papers"])
        full_name = format_full_name(lastname, first, suffix)

        if node_id in existing_in_batches:
            authors_skipped += 1
        else:
            if args.dry_run:
                print(f"  WOULD CREATE   {node_id}  —  {full_name}  ({len(paper_list)} paper(s))")
            else:
                new_nodes.append(build_author_dict(node_id, lastname, first, suffix, paper_list))
            authors_created += 1

    if new_nodes and not args.dry_run:
        write_author_batch(new_nodes, out_dir)

    # Step 6: add authored_by edges to REF nodes
    print()
    print("Adding authored_by edges to REF nodes ...")
    edges_added = 0
    edges_skipped = 0

    for paper_node_id, author_keys in paper_authors.items():
        ref_path = ref_dir / f"{paper_node_id}.yaml"
        if not ref_path.exists():
            continue
        for key in author_keys:
            lastname, first, suffix = key
            author_node_id = author_node_map[key]
            full_name = format_full_name(lastname, first, suffix)
            modified = add_authored_by_edge(ref_path, author_node_id, full_name, args.dry_run)
            if modified:
                if args.dry_run:
                    print(f"  WOULD ADD EDGE {paper_node_id} → {author_node_id}")
                else:
                    print(f"  EDGE ADDED     {paper_node_id} → {author_node_id}")
                edges_added += 1
            else:
                edges_skipped += 1

    print()
    print("=" * 60)
    print(f"Authors {'would create' if args.dry_run else 'created'}: {authors_created}")
    print(f"Authors skipped (exists): {authors_skipped}")
    print(f"Edges {'would add' if args.dry_run else 'added'}:   {edges_added}")
    print(f"Edges skipped (exists):   {edges_skipped}")
    print("=" * 60)

    if args.dry_run and authors_created > 0:
        print()
        print("Run without --dry-run to write files.")


if __name__ == "__main__":
    main()
