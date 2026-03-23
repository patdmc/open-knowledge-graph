#!/usr/bin/env python3
"""
arxiv_to_graph.py — Convert arXiv JSON/JSONL records to knowledge graph REF nodes.

Usage:
    python scripts/arxiv_to_graph.py <input.jsonl> [options]

Options:
    --categories    Comma-separated arXiv category prefixes to include
                    Default: cs.AI,cs.LG,cs.NE,cs.IT,cs.CL,stat.ML,math.IT,
                             q-bio,cond-mat.stat-mech,physics.bio-ph,nlin.AO
    --keywords      Comma-separated keywords to match in title (case-insensitive)
                    Papers matching ANY keyword are included (in addition to category filter)
    --out-dir       Output directory for YAML files
                    Default: knowledge-graph/nodes/references/
    --dry-run       Print what would be created without writing files
    --all           Skip relevance filtering — process every record
    --limit         Stop after N records (useful for testing)

Examples:
    # Filter by default categories
    python scripts/arxiv_to_graph.py arxiv-metadata.jsonl

    # Add keyword filter on top of category filter
    python scripts/arxiv_to_graph.py arxiv-metadata.jsonl --keywords "free energy,knowledge graph,uncertainty"

    # Dry run to preview
    python scripts/arxiv_to_graph.py arxiv-metadata.jsonl --dry-run --limit 100

    # Custom categories
    python scripts/arxiv_to_graph.py arxiv-metadata.jsonl --categories "cs.AI,q-bio.NC,math.IT"
"""

import argparse
import json
import os
import re
import sys
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
    '=': {'a':'ā','e':'ē','i':'ī','o':'ō','u':'ū','ı':'ī',
          'A':'Ā','E':'Ē','I':'Ī','O':'Ō','U':'Ū'},
    '.': {'a':'ȧ','e':'ė','o':'ȯ','z':'ż','A':'Ȧ','E':'Ė','O':'Ȯ','Z':'Ż'},
    'u': {'a':'ă','e':'ĕ','i':'ĭ','o':'ŏ','u':'ŭ','A':'Ă','E':'Ĕ','I':'Ĭ','O':'Ŏ','U':'Ŭ'},
    'v': {'c':'č','n':'ň','s':'š','z':'ž','e':'ě','r':'ř','C':'Č','N':'Ň','S':'Š','Z':'Ž','E':'Ě','R':'Ř'},
    'H': {'o':'ő','u':'ű','O':'Ő','U':'Ű'},
    'k': {'a':'ą','e':'ę','A':'Ą','E':'Ę'},
    'c': {'c':'ç','s':'ş','C':'Ç','S':'Ş'},
}
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
    if '\\' in s:
        for acc, mapping in _TEX_ACCENTS.items():
            for letter, uni in mapping.items():
                s = s.replace(f'\\{acc}{{{letter}}}', uni)
                s = s.replace(f'\\{acc}{letter}', uni)
        for tex, uni in _TEX_STANDALONE:
            s = s.replace(tex, uni)
        s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\{([^}]*)\}', r'\1', s)
    if '"' in s:
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

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES = {
    "cs.AI",       # Artificial Intelligence
    "cs.LG",       # Machine Learning
    "cs.NE",       # Neural and Evolutionary Computing
    "cs.IT",       # Information Theory (cs)
    "cs.CL",       # Computation and Language
    "cs.MA",       # Multiagent Systems
    "stat.ML",     # Statistics - Machine Learning
    "math.IT",     # Mathematics - Information Theory
    "q-bio.NC",    # Quantitative Biology - Neurons and Cognition
    "q-bio.PE",    # Quantitative Biology - Populations and Evolution
    "q-bio.GN",    # Quantitative Biology - Genomics
    "cond-mat.stat-mech",  # Statistical Mechanics
    "physics.bio-ph",      # Biological Physics
    "nlin.AO",     # Nonlinear Sciences - Adaptation and Self-Organizing Systems
}

# Default keywords — match ANY of these in the title
DEFAULT_KEYWORDS = [
    "free energy",
    "active inference",
    "predictive coding",
    "uncertainty",
    "knowledge graph",
    "information theory",
    "bayesian",
    "reinforcement learning",
    "meta-learning",
    "collective intelligence",
    "self-organization",
    "entropy",
    "agency",
    "intelligence",
    "cognition",
    "evolution",
    "natural selection",
    "gradient descent",
    "generalization",
    "representation learning",
]

TODAY = date.today().isoformat()
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "knowledge-graph" / "nodes" / "references"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_year(record: dict) -> str:
    """Extract year from arXiv record. Tries journal-ref first, then arXiv ID."""
    # Try journal-ref: look for 4-digit year
    jref = record.get("journal-ref") or ""
    m = re.search(r"\b(19|20)\d{2}\b", jref)
    if m:
        return m.group(0)

    # Try arXiv ID: old format quant-ph/YYMMNNN or new format YYMM.NNNN
    arxiv_id = record.get("id", "")
    # New format: YYMM.NNNN
    m = re.match(r"^(\d{2})(\d{2})\.\d+", arxiv_id)
    if m:
        yy = int(m.group(1))
        year = (2000 + yy) if yy < 90 else (1900 + yy)
        return str(year)
    # Old format: category/YYMMNNN
    m = re.search(r"/(\d{2})\d{2}\d+", arxiv_id)
    if m:
        yy = int(m.group(1))
        year = (2000 + yy) if yy < 90 else (1900 + yy)
        return str(year)

    return "unknown"


def parse_first_author_lastname(authors_str: str) -> str:
    """Extract the last name of the first author from the authors string."""
    if not authors_str:
        return "Unknown"

    # Authors are typically separated by commas or 'and'
    # First author is the first entry
    first = re.split(r",\s*|\s+and\s+", authors_str.strip())[0].strip()

    # Handle 'Last, First' format
    if "," in first:
        return first.split(",")[0].strip()

    # Handle 'First [Middle] Last' format — take the last word
    parts = first.split()
    if parts:
        # Filter out suffixes like Jr., II, III
        suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "md"}
        last = parts[-1]
        if last.lower() in suffixes and len(parts) > 1:
            last = parts[-2]
        return last

    return "Unknown"


def make_node_id(record: dict, existing_ids: set) -> str:
    """Generate a REF node ID following the REF-AuthorYear convention."""
    lastname = parse_first_author_lastname(record.get("authors", ""))
    year = parse_year(record)

    # Sanitize: remove non-alphanumeric characters
    lastname = re.sub(r"[^A-Za-z0-9]", "", lastname)

    base = f"REF-{lastname}{year}"
    node_id = base

    # Disambiguate with suffix if needed
    suffix = 2
    while node_id in existing_ids:
        node_id = f"{base}-{suffix}"
        suffix += 1

    return node_id


def matches_categories(record: dict, category_prefixes: set) -> bool:
    """Return True if any of the record's categories match the allowed prefixes."""
    cats = record.get("categories", [])
    if isinstance(cats, str):
        cats = cats.split()
    for cat in cats:
        # Check exact match and prefix match (e.g. "q-bio" matches "q-bio.NC")
        for prefix in category_prefixes:
            if cat == prefix or cat.startswith(prefix + "."):
                return True
    return False


def matches_keywords(record: dict, keywords: list) -> bool:
    """Return True if any keyword appears in the title."""
    title = (record.get("title") or "").lower()
    return any(kw.lower() in title for kw in keywords)


def is_relevant(record: dict, category_prefixes: set, keywords: list) -> bool:
    """Return True if the record should be included."""
    return matches_categories(record, category_prefixes) or matches_keywords(record, keywords)


def format_authors(authors_str: str) -> str:
    """Clean up author string for display."""
    if not authors_str:
        return "Unknown"
    # Collapse internal newlines/whitespace
    return re.sub(r"\s+", " ", authors_str.strip())


def format_categories(record: dict) -> str:
    """Format categories as a readable string."""
    cats = record.get("categories", [])
    if isinstance(cats, str):
        cats = cats.split()
    return ", ".join(cats)


def build_yaml(node_id: str, record: dict) -> str:
    """Build a REF node YAML string from an arXiv record."""
    arxiv_id = _clean_str(record.get("id", ""))
    title = _clean_str((record.get("title") or "").replace("\n", " ").strip())
    title = re.sub(r"\s+", " ", title)
    authors = format_authors(_clean_str(record.get("authors", "")))
    year = parse_year(record)
    journal_ref = _clean_str((record.get("journal-ref") or "").strip())
    doi = _clean_str((record.get("doi") or "").strip())
    categories = format_categories(record)

    # Build citation string
    citation_parts = [f"{authors}."]
    if year != "unknown":
        citation_parts.append(f"({year}).")
    citation_parts.append(f"{title}.")
    if journal_ref:
        citation_parts.append(journal_ref + ".")
    if doi:
        citation_parts.append(f"https://doi.org/{doi}")
    else:
        citation_parts.append(f"https://arxiv.org/abs/{arxiv_id}")
    citation = " ".join(citation_parts)

    # Build statement — what we know without the abstract
    statement_lines = [
        f"arXiv paper: {title}",
        f"",
        f"Authors: {authors}",
        f"Categories: {categories}",
        f"arXiv ID: {arxiv_id}",
    ]
    if journal_ref:
        statement_lines.append(f"Published: {journal_ref}")
    statement_lines.extend([
        "",
        "NOTE: Abstract not available in metadata-only record.",
        "Statement to be updated after reading the full paper.",
    ])
    statement = "\n".join(statement_lines)

    # Determine source string
    source = journal_ref if journal_ref else f"arXiv:{arxiv_id}"
    date_str = year if year != "unknown" else TODAY

    # DOI or arxiv URL for the provenance
    url = f"https://doi.org/{doi}" if doi else f"https://arxiv.org/abs/{arxiv_id}"

    first_author_last = yaml_dq_escape(authors.split(',')[0].strip().split()[-1])
    name_field = yaml_dq_escape(f"{authors.split(',')[0].strip().split()[-1]} ({year}) \u2014 {title[:60]}{'...' if len(title) > 60 else ''}")
    yaml = f"""id: {node_id}
type: reference
fidelity: low
name: "{name_field}"
statement: |
{chr(10).join('  ' + line for line in statement_lines)}

provenance:
  attribution:
    author: "{yaml_dq_escape(authors)}"
    source: "{yaml_dq_escape(source)}"
    date: "{yaml_dq_escape(date_str)}"
    doi: "{yaml_dq_escape(doi)}"
    arxiv_id: "{yaml_dq_escape(arxiv_id)}"
    url: "{yaml_dq_escape(url)}"
  evidence:
    type: cited
    description: |
      Imported from arXiv metadata. Categories: {categories}.
      Statement and edges to be completed after reading the paper.
    references: []
  derivation:
    from: []
    method: "External citation — imported from arXiv metadata via arxiv_to_graph.py"

edges: []
# TODO: Add edges to relevant theory nodes after reading the paper.
# Common relations: cited_by, overlaps_with, supports_by, grounds, contrasts_with
# Common targets: D02-knowledge-graph, D05-uncertainty, T01-minimization-imperative,
#                 NV01-graph-necessity, REF-Friston2010, etc.
"""
    return yaml


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert arXiv JSONL records to knowledge graph REF nodes."
    )
    parser.add_argument("input", help="Path to arXiv JSONL file (one record per line)")
    parser.add_argument(
        "--categories",
        help="Comma-separated category prefixes (default: see script header)",
        default=None,
    )
    parser.add_argument(
        "--keywords",
        help="Comma-separated title keywords (case-insensitive, OR logic)",
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
        default=str(DEFAULT_OUT_DIR),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Skip relevance filtering — process every record",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N records",
    )
    args = parser.parse_args()

    # Resolve filters
    category_prefixes = (
        set(args.categories.split(",")) if args.categories else DEFAULT_CATEGORIES
    )
    keywords = (
        [k.strip() for k in args.keywords.split(",")] if args.keywords else DEFAULT_KEYWORDS
    )

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Collect existing node IDs to avoid collisions
    existing_ids = set()
    if out_dir.exists():
        for f in out_dir.glob("*.yaml"):
            existing_ids.add(f.stem)

    # Process
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    total = 0
    matched = 0
    skipped_existing = 0
    written = 0
    errors = 0

    print(f"Reading: {input_path}")
    print(f"Output:  {out_dir}")
    print(f"Mode:    {'dry-run' if args.dry_run else 'write'}")
    print(f"Filter:  {'none (--all)' if args.all else f'{len(category_prefixes)} categories + {len(keywords)} keywords'}")
    print()

    with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, 1):
            if args.limit and total >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: JSON parse error — {e}", file=sys.stderr)
                errors += 1
                continue

            # Relevance filter
            if not args.all and not is_relevant(record, category_prefixes, keywords):
                continue

            matched += 1
            node_id = make_node_id(record, existing_ids)
            out_path = out_dir / f"{node_id}.yaml"

            title = (record.get("title") or "").replace("\n", " ").strip()
            title = re.sub(r"\s+", " ", title)
            arxiv_id = record.get("id", "")

            # Skip if already exists
            if out_path.exists():
                print(f"  SKIP (exists)  {node_id}  —  {title[:60]}")
                skipped_existing += 1
                continue

            yaml_content = build_yaml(node_id, record)

            if args.dry_run:
                print(f"  WOULD CREATE   {node_id}  —  {title[:60]}")
            else:
                out_path.write_text(yaml_content, encoding="utf-8")
                existing_ids.add(node_id)
                written += 1
                print(f"  CREATED        {node_id}  —  {title[:60]}")

    print()
    print("=" * 60)
    print(f"Records read:     {total}")
    print(f"Matched filter:   {matched}")
    print(f"Skipped (exists): {skipped_existing}")
    print(f"{'Would create' if args.dry_run else 'Created'}:      {written if not args.dry_run else matched - skipped_existing}")
    if errors:
        print(f"Errors:           {errors}")
    print("=" * 60)

    if args.dry_run and (matched - skipped_existing) > 0:
        print()
        print("Run without --dry-run to write files.")


if __name__ == "__main__":
    main()
