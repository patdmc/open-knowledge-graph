#!/usr/bin/env python3
"""
follow_references.py — Follow citation chains from an author pipeline output,
download cited papers, and run the full inference pipeline on the expanded corpus.

For each paper in the source directory:
  1. Parses .bbl and .tex files to extract cited arXiv IDs
  2. Downloads source tarballs for cited papers from arxiv.org/src/
  3. Chunks, embeds, clusters, and scores the full expanded corpus
  4. Optionally recurses to depth N (citations of citations)

Writes everything into the same author output directory structure,
with cited papers added to 01-sources/ and the pipeline re-run from
Phase 2 onward on the combined corpus.

Usage:
    python scripts/follow_references.py --author-dir leek_j_t/ [options]

Options:
    --author-dir    Author pipeline output directory (required)
    --depth         Citation chain depth to follow (default: 1)
    --delay         Seconds between HTTP requests (default: 2.0)
    --limit         Max cited papers to download per depth level
    --n-clusters    Clusters for re-run (default: auto)
    --no-nli        Skip NLI scoring
    --from-phase    Re-run pipeline from this phase after downloading (default: 2)

Examples:
    # Follow one level of citations
    python scripts/follow_references.py --author-dir leek_j_t/

    # Follow two levels deep, max 50 papers per level
    python scripts/follow_references.py --author-dir leek_j_t/ --depth 2 --limit 50

    # Just extract and download, don't re-run inference
    python scripts/follow_references.py --author-dir leek_j_t/ --from-phase 999
"""

import argparse
import io
import json
import re
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

_SSL_CTX = ssl.create_default_context()
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode    = ssl.CERT_NONE


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_get(url: str, delay: float, retries: int = 4) -> bytes | None:
    for attempt in range(retries):
        time.sleep(delay * (2 ** attempt))
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "open-knowledge-graph/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 15 * (2 ** attempt)
                print(f"    Rate limited — waiting {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            print(f"    HTTP {e.code}: {url}", flush=True)
            return None
        except Exception as e:
            print(f"    Error: {e}", flush=True)
            return None
    return None


# ---------------------------------------------------------------------------
# arXiv ID extraction from .bbl and .tex files
# ---------------------------------------------------------------------------

ARXIV_PATTERNS = [
    # New format: 2301.04567 or 2301.04567v2
    re.compile(r"\barXiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE),
    re.compile(r"\barxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE),
    re.compile(r"\barxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE),
    re.compile(r"\beprint\s*=\s*\{(\d{4}\.\d{4,5}(?:v\d+)?)\}", re.IGNORECASE),
    # Bare new format in URLs/eprint fields
    re.compile(r"arXiv\s+preprint\s+arXiv:(\d{4}\.\d{4,5})", re.IGNORECASE),
    # Old format: hep-th/9212071
    re.compile(r"\b([a-z][a-z\-]+/\d{7})\b", re.IGNORECASE),
]

def extract_arxiv_ids_from_text(text: str) -> set[str]:
    ids = set()
    for pat in ARXIV_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1).strip().rstrip(".,;v").split("v")[0]
            # Validate
            if re.match(r"^\d{4}\.\d{4,5}$", raw):
                ids.add(raw)
            elif re.match(r"^[a-z][a-z\-]+/\d{7}$", raw, re.IGNORECASE):
                ids.add(raw.lower())
    return ids


def extract_arxiv_ids_from_dir(src_dir: Path) -> set[str]:
    """Extract all cited arXiv IDs from .bbl and .tex files in a source dir."""
    ids = set()
    for pattern in ("*.bbl", "*.tex"):
        for f in src_dir.glob(pattern):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                ids |= extract_arxiv_ids_from_text(text)
            except Exception:
                pass
    return ids


# ---------------------------------------------------------------------------
# Download source tarball
# ---------------------------------------------------------------------------

def download_source(arxiv_id: str, dest_dir: Path, delay: float) -> str:
    """
    Download and extract arXiv source tarball into dest_dir/{arxiv_id}/.
    Returns source type: 'arxiv' | 'none'
    """
    paper_dir = dest_dir / arxiv_id.replace("/", "_")
    if paper_dir.exists() and (paper_dir / "source_type").exists():
        return (paper_dir / "source_type").read_text().strip()

    paper_dir.mkdir(parents=True, exist_ok=True)
    url  = f"https://arxiv.org/src/{arxiv_id}"
    data = http_get(url, delay)

    if not data:
        (paper_dir / "source_type").write_text("none")
        return "none"

    # Try tarball
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            members = [m for m in tar.getmembers()
                       if m.name.lower().endswith((".tex", ".bbl"))]
            if not members:
                raise tarfile.TarError("no tex/bbl files")
            for m in members:
                f = tar.extractfile(m)
                if f:
                    out = paper_dir / Path(m.name).name
                    out.write_bytes(f.read())
        (paper_dir / "source_type").write_text("arxiv")
        return "arxiv"
    except tarfile.TarError:
        # Might be a bare .tex file
        if data[:1] in (b"\\", b"%"):
            (paper_dir / "paper.tex").write_bytes(data)
            (paper_dir / "source_type").write_text("arxiv")
            return "arxiv"
        (paper_dir / "source_type").write_text("none")
        return "none"


# ---------------------------------------------------------------------------
# Reference chain crawler
# ---------------------------------------------------------------------------

def crawl_references(sources_dir: Path, depth: int, delay: float,
                     limit: int | None) -> dict[str, int]:
    """
    BFS over citation chains starting from papers already in sources_dir.
    Returns {arxiv_id: depth_found} for all newly discovered papers.
    """
    # Papers already present
    existing = set()
    for d in sources_dir.iterdir():
        if d.is_dir():
            st = (d / "source_type")
            if st.exists() and st.read_text().strip() != "none":
                existing.add(d.name.replace("_", "/"))

    status_path = sources_dir / "_refs_status.json"
    discovered  = {}  # arxiv_id → depth
    if status_path.exists():
        discovered = json.loads(status_path.read_text())
        print(f"  Resuming: {len(discovered)} previously discovered")

    frontier = list(existing)  # start from already-downloaded papers
    current_depth = 0

    while current_depth < depth and frontier:
        print(f"\n  Depth {current_depth + 1}: scanning {len(frontier)} papers for citations ...")
        next_frontier = []

        for paper_id_raw in frontier:
            src_dir = sources_dir / paper_id_raw.replace("/", "_")
            if not src_dir.exists():
                continue
            cited = extract_arxiv_ids_from_dir(src_dir)
            new   = cited - existing - set(discovered.keys()) - set(next_frontier)
            for cid in new:
                if cid not in discovered:
                    next_frontier.append(cid)

        print(f"  Found {len(next_frontier)} new cited arXiv IDs")
        if limit:
            next_frontier = next_frontier[:limit]
            print(f"  Limited to {len(next_frontier)}")

        # Download
        fetched = 0
        for i, arxiv_id in enumerate(next_frontier):
            print(f"    [{i+1}/{len(next_frontier)}] {arxiv_id}", end=" ", flush=True)
            src_type = download_source(arxiv_id, sources_dir, delay)
            print(f"→ {src_type}")
            discovered[arxiv_id] = current_depth + 1
            if src_type == "arxiv":
                fetched += 1
                existing.add(arxiv_id)

            # Save checkpoint after each download
            status_path.write_text(json.dumps(discovered, indent=2))

        print(f"  Depth {current_depth + 1}: {fetched} new sources downloaded")
        frontier = [aid for aid in next_frontier
                    if (sources_dir / aid.replace("/", "_") / "source_type").exists()
                    and (sources_dir / aid.replace("/", "_") / "source_type").read_text().strip() == "arxiv"]
        current_depth += 1

    return discovered


# ---------------------------------------------------------------------------
# Re-run pipeline phases 2+ on expanded corpus
# ---------------------------------------------------------------------------

def rerun_pipeline(author_dir: Path, from_phase: int,
                   n_clusters: int | None, no_nli: bool) -> None:
    """Import and re-run author_pipeline phases on the expanded source set."""
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    import author_pipeline as ap
    import numpy as np

    dirs = {
        "papers":   author_dir / "00-papers",
        "sources":  author_dir / "01-sources",
        "chunks":   author_dir / "02-chunks",
        "emb":      author_dir / "03-embeddings",
        "clusters": author_dir / "04-clusters",
        "keywords": author_dir / "05-keywords",
        "collated": author_dir / "06-collated",
        "reps":     author_dir / "07-representatives",
        "scores":   author_dir / "08-scores",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    papers_path = dirs["papers"] / "papers.jsonl"
    chunks_path = dirs["chunks"] / "chunks.jsonl"
    scores_path = dirs["scores"] / "pairs.jsonl"

    p = from_phase

    if p <= 2:
        print("\n=== Phase 2: Chunk text (expanded corpus) ===")
        # Clear existing chunks to rebuild from full source set
        chunks_path.unlink(missing_ok=True)
        (dirs["chunks"] / "_status_phase2.json").unlink(missing_ok=True)
        ap.phase2_chunk(papers_path, dirs["sources"], dirs["chunks"])

    embeddings = None
    if p <= 3:
        print("\n=== Phase 3: Embed ===")
        embeddings, _ = ap.phase3_embed(chunks_path, dirs["emb"], "all-MiniLM-L6-v2")

    if embeddings is None and p <= 9:
        emb_path = dirs["emb"] / "embeddings.npy"
        if emb_path.exists():
            embeddings = np.load(str(emb_path))

    if p <= 4 and embeddings is not None:
        print("\n=== Phase 4: Cluster ===")
        metadata = json.loads((dirs["emb"] / "metadata.json").read_text())
        ap.phase4_cluster(embeddings, metadata, dirs["clusters"], n_clusters, 0.50)

    if p <= 5:
        print("\n=== Phase 5: Extract keywords ===")
        # Clear status so new chunks get processed
        (dirs["keywords"] / "_status_phase5.json").unlink(missing_ok=True)
        ap.phase5_keywords(chunks_path, dirs["keywords"])

    if p <= 6:
        print("\n=== Phase 6: Collate ===")
        ap.phase6_collate(dirs["clusters"], dirs["keywords"], dirs["collated"])

    if p <= 7:
        print("\n=== Phase 7: Representatives ===")
        ap.phase7_representatives(dirs["clusters"], dirs["collated"],
                                   dirs["emb"], dirs["reps"])

    if p <= 8:
        print("\n=== Phase 8: Score cross-pairs ===")
        scores_path.unlink(missing_ok=True)
        (dirs["scores"] / "_status_phase8.json").unlink(missing_ok=True)
        ap.phase8_score(dirs["reps"], dirs["emb"], chunks_path,
                         dirs["scores"], no_nli, top=None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Follow citation chains and expand author pipeline corpus."
    )
    parser.add_argument("--author-dir", required=True,
                        help="Author pipeline output directory (e.g. leek_j_t/)")
    parser.add_argument("--depth",      type=int,   default=1,
                        help="Citation depth to follow (default: 1)")
    parser.add_argument("--delay",      type=float, default=2.0,
                        help="Seconds between requests (default: 2.0)")
    parser.add_argument("--limit",      type=int,   default=None,
                        help="Max cited papers to download per depth level")
    parser.add_argument("--n-clusters", type=int,   default=None)
    parser.add_argument("--no-nli",     action="store_true")
    parser.add_argument("--from-phase", type=int,   default=2,
                        help="Re-run pipeline from this phase after download (default: 2)")
    args = parser.parse_args()

    author_dir  = Path(args.author_dir)
    sources_dir = author_dir / "01-sources"

    if not sources_dir.exists():
        print(f"Error: {sources_dir} not found.", file=sys.stderr)
        sys.exit(1)

    # Count existing sources
    existing = [d for d in sources_dir.iterdir()
                if d.is_dir() and (d / "source_type").exists()
                and (d / "source_type").read_text().strip() == "arxiv"]
    print(f"Starting corpus: {len(existing)} papers with sources")

    # Crawl
    print(f"\n=== Crawling citation chains (depth={args.depth}) ===")
    discovered = crawl_references(sources_dir, args.depth, args.delay, args.limit)

    new_arxiv = {k: v for k, v in discovered.items()
                 if (sources_dir / k.replace("/","_") / "source_type").exists()
                 and (sources_dir / k.replace("/","_") / "source_type").read_text().strip() == "arxiv"}

    print(f"\nDiscovered: {len(discovered)} cited papers")
    print(f"Downloaded with source: {len(new_arxiv)}")

    # Summary by depth
    by_depth = {}
    for aid, d in discovered.items():
        by_depth.setdefault(d, []).append(aid)
    for d, aids in sorted(by_depth.items()):
        got = sum(1 for a in aids
                  if (sources_dir / a.replace("/","_") / "source_type").exists()
                  and (sources_dir / a.replace("/","_") / "source_type").read_text().strip() == "arxiv")
        print(f"  Depth {d}: {len(aids)} cited, {got} downloaded")

    # Add discovered papers to papers.jsonl so chunker picks them up
    papers_path = author_dir / "00-papers" / "papers.jsonl"
    existing_ids = set()
    if papers_path.exists():
        existing_ids = {json.loads(l).get("arxiv_id", json.loads(l).get("paperId",""))
                        for l in papers_path.read_text().splitlines() if l.strip()}

    added = 0
    with open(papers_path, "a", encoding="utf-8") as fh:
        for arxiv_id, depth in discovered.items():
            if arxiv_id in existing_ids:
                continue
            src_dir   = sources_dir / arxiv_id.replace("/", "_")
            src_type  = (src_dir / "source_type").read_text().strip() if (src_dir / "source_type").exists() else "none"
            if src_type != "arxiv":
                continue
            record = {
                "arxiv_id":    arxiv_id,
                "paperId":     arxiv_id,
                "externalIds": {"ArXiv": arxiv_id},
                "title":       f"arXiv:{arxiv_id}",
                "authors":     "",
                "abstract":    "",
                "year":        0,
                "categories":  [],
                "url":         f"https://arxiv.org/abs/{arxiv_id}",
                "source_url":  f"https://arxiv.org/src/{arxiv_id}",
                "ref_depth":   depth,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_ids.add(arxiv_id)
            added += 1

    print(f"\nAdded {added} new papers to papers.jsonl")

    # Re-run pipeline
    if args.from_phase <= 9:
        print(f"\n=== Re-running pipeline from phase {args.from_phase} ===")
        rerun_pipeline(author_dir, args.from_phase, args.n_clusters, args.no_nli)

    print(f"\nDone. Expanded corpus in {author_dir}/")


if __name__ == "__main__":
    main()
