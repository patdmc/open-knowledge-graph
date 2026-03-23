#!/usr/bin/env python3
"""
arxiv_fetch_references.py — Download arXiv source tarballs and extract citation edges.

For each REF node in the graph that has an arXiv ID:
  1. Downloads https://arxiv.org/src/{arxiv_id}  (source tarball)
  2. Extracts .bbl files and \cite{} / \bibitem{} from .tex files
  3. Matches cited papers against other REF nodes in the graph
  4. Writes cites/cited_by edges between matched nodes

Source tarballs are cached in --cache-dir so repeated runs don't re-download.

Usage:
    python scripts/arxiv_fetch_references.py [options]

Options:
    --ref-dir     REF nodes directory
                  Default: knowledge-graph/nodes/references/
    --cache-dir   Local cache for downloaded tarballs
                  Default: .arxiv-cache/
    --dry-run     Parse and match but don't write edges
    --limit       Process at most N papers
    --delay       Seconds to wait between requests (default: 1.0)
    --ids         Comma-separated arXiv IDs to process (overrides scanning ref-dir)

Examples:
    # Process all papers in the graph
    python scripts/arxiv_fetch_references.py

    # Process specific papers
    python scripts/arxiv_fetch_references.py --ids 2301.04567,2205.09991

    # Preview without writing
    python scripts/arxiv_fetch_references.py --dry-run --limit 5
"""

import argparse
import io
import re
import sys
import tarfile
import time
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

TODAY = date.today().isoformat()
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_REF_DIR = REPO_ROOT / "knowledge-graph" / "nodes" / "references"
DEFAULT_CACHE_DIR = REPO_ROOT / ".arxiv-cache"
SOURCE_URL = "https://arxiv.org/src/{arxiv_id}"


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(ref_dir: Path) -> dict[str, str]:
    """Returns {arxiv_id: node_id} for all REF nodes with an arxiv_id."""
    result = {}
    for yaml_path in ref_dir.glob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        m = re.search(r"arxiv_id:\s*[\"']?([^\s\"'\n]+)[\"']?", text)
        if m:
            aid = m.group(1).strip()
            if aid:
                result[aid] = yaml_path.stem
    return result


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------

def fetch_tarball(arxiv_id: str, cache_dir: Path, delay: float) -> bytes | None:
    """
    Return the raw bytes of the source tarball for arxiv_id.
    Uses cache_dir to avoid re-downloading. Returns None on failure.
    """
    # Normalise cache filename: replace / with _ for old-format IDs
    cache_name = arxiv_id.replace("/", "_") + ".tar.gz"
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        return cache_path.read_bytes()

    url = SOURCE_URL.format(arxiv_id=arxiv_id)
    print(f"    GET {url}", flush=True)
    time.sleep(delay)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "open-knowledge-graph/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code} — skipping", flush=True)
        return None
    except Exception as e:
        print(f"    Error: {e} — skipping", flush=True)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# Reference extraction from tarball
# ---------------------------------------------------------------------------

def extract_cited_ids(tarball_bytes: bytes) -> set[str]:
    """
    Parse a source tarball and return all arXiv IDs cited within it.
    Looks in:
      - .bbl files: \bibitem tags and arXiv IDs in URLs/eprint fields
      - .tex files: \cite{}, arXiv IDs in hrefs and eprint fields
    """
    cited = set()

    try:
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:*") as tar:
            for member in tar.getmembers():
                name = member.name.lower()
                if not (name.endswith(".bbl") or name.endswith(".tex")):
                    continue
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    text = f.read().decode("utf-8", errors="replace")
                except Exception:
                    continue

                cited.update(_parse_arxiv_ids(text))
    except tarfile.TarError as e:
        print(f"    Tar error: {e}", flush=True)

    return cited


# arXiv ID patterns to search for in source text
_ARXIV_PATTERNS = [
    # New format in URLs or eprint fields: 2301.04567
    re.compile(r"\b(\d{4}\.\d{4,5})\b"),
    # Old format: hep-th/9212071  alg-geom/9307001  etc.
    re.compile(r"\b([a-z\-]+/\d{7})\b", re.IGNORECASE),
    # arXiv:2301.04567 or arXiv:hep-th/9212071
    re.compile(r"arxiv:([^\s,}\]]+)", re.IGNORECASE),
    # eprint = {2301.04567}
    re.compile(r"eprint\s*=\s*\{([^}]+)\}", re.IGNORECASE),
]

def _parse_arxiv_ids(text: str) -> set[str]:
    ids = set()
    for pattern in _ARXIV_PATTERNS:
        for m in pattern.finditer(text):
            candidate = m.group(1).strip().rstrip(".,;")
            # Basic validity check
            if re.match(r"^\d{4}\.\d{4,5}$", candidate):
                ids.add(candidate)
            elif re.match(r"^[a-z\-]+/\d{7}$", candidate, re.IGNORECASE):
                ids.add(candidate.lower())
    return ids


# ---------------------------------------------------------------------------
# Edge writing
# ---------------------------------------------------------------------------

def edge_exists(yaml_text: str, target: str, relation: str) -> bool:
    return target in yaml_text and relation in yaml_text


def append_edge(yaml_path: Path, target_node_id: str, relation: str,
                description: str, dry_run: bool) -> bool:
    text = yaml_path.read_text(encoding="utf-8")
    if edge_exists(text, target_node_id, relation):
        return False

    edge_block = f"""  - to: {target_node_id}
    relation: {relation}
    provenance:
      attribution:
        author: "arxiv_fetch_references.py"
        source: "arXiv LaTeX source (parsed .bbl/.tex)"
        date: "{TODAY}"
      evidence:
        type: cited
        description: "{description}"
      derivation:
        method: "Parsed from downloaded arXiv source tarball"
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv source tarballs and extract citation edges."
    )
    parser.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between requests (default: 1.0)")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated arXiv IDs to process")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    cache_dir = Path(args.cache_dir)

    # Load graph
    print(f"Scanning {ref_dir} ...")
    arxiv_to_node = load_graph(ref_dir)
    print(f"  {len(arxiv_to_node)} REF nodes with arXiv IDs")

    # Determine which IDs to process
    if args.ids:
        to_process = [i.strip() for i in args.ids.split(",")]
    else:
        to_process = list(arxiv_to_node.keys())

    if args.limit:
        to_process = to_process[:args.limit]

    print(f"  Processing {len(to_process)} paper(s)")
    print(f"  Cache: {cache_dir}")
    print(f"  Mode:  {'dry-run' if args.dry_run else 'write'}")
    print()

    # Stats
    fetched = 0
    failed = 0
    edges_added = 0
    edges_skipped = 0

    for arxiv_id in to_process:
        node_id = arxiv_to_node.get(arxiv_id)
        if not node_id:
            print(f"  [{arxiv_id}] No REF node — skipping")
            continue

        print(f"  [{arxiv_id}] → {node_id}")

        tarball = fetch_tarball(arxiv_id, cache_dir, args.delay)
        if tarball is None:
            failed += 1
            continue

        fetched += 1
        cited_ids = extract_cited_ids(tarball)
        print(f"    Found {len(cited_ids)} arXiv ID(s) in source")

        paper_yaml = ref_dir / f"{node_id}.yaml"

        for cited_id in cited_ids:
            cited_node_id = arxiv_to_node.get(cited_id)
            if not cited_node_id:
                continue  # not in graph — skip

            if cited_node_id == node_id:
                continue  # self-reference — skip

            cited_yaml = ref_dir / f"{cited_node_id}.yaml"

            # cites edge: paper → cited
            added = append_edge(
                paper_yaml, cited_node_id, "cites",
                f"{node_id} cites {cited_node_id} (parsed from LaTeX source)",
                args.dry_run,
            )
            if added:
                print(f"    + {node_id} --cites--> {cited_node_id}")
                edges_added += 1
            else:
                edges_skipped += 1

            # cited_by edge: cited → paper
            added = append_edge(
                cited_yaml, node_id, "cited_by",
                f"{cited_node_id} cited by {node_id} (parsed from LaTeX source)",
                args.dry_run,
            )
            if added:
                print(f"    + {cited_node_id} --cited_by--> {node_id}")
                edges_added += 1
            else:
                edges_skipped += 1

    print()
    print("=" * 60)
    print(f"Papers fetched:  {fetched}")
    print(f"Papers failed:   {failed}")
    print(f"Edges {'would add' if args.dry_run else 'added'}:   {edges_added}")
    print(f"Edges skipped:   {edges_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
