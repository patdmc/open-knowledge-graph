#!/usr/bin/env python3
"""
scrape_arxiv_author.py — Scrape arXiv author search page for all papers.

Uses the arXiv search URL format:
  https://arxiv.org/search/?searchtype=author&query=Leek%2C+J+T

Writes:
    <out>          JSONL file, one paper per line with arXiv ID, title, year, categories

Usage:
    python scripts/scrape_arxiv_author.py --query "Leek, J T" [options]
    python scripts/scrape_arxiv_author.py --url "https://arxiv.org/search/?searchtype=author&query=Leek%2C+J+T"

Options:
    --query     Author name in arXiv format: "Last, F I"  (e.g. "Leek, J T")
    --url       Full arXiv search URL (overrides --query)
    --out       Output JSONL file (default: stdout)
    --delay     Seconds between page requests (default: 2.0)
    --limit     Stop after N papers

Examples:
    python scripts/scrape_arxiv_author.py --query "Leek, J T" --out leek_arxiv.jsonl
    python scripts/scrape_arxiv_author.py --query "Friston, K" --out friston_arxiv.jsonl
"""

import argparse
import html
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_SSL_CTX = ssl.create_default_context()
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode    = ssl.CERT_NONE


def fetch_page(url: str, delay: float) -> str | None:
    time.sleep(delay)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "open-knowledge-graph/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Error fetching {url}: {e}", file=sys.stderr)
        return None


def clean(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_results(page: str) -> list[dict]:
    """Parse one page of arXiv search results into paper dicts."""
    papers = []
    blocks = re.split(r'<li class="arxiv-result">', page)[1:]

    for block in blocks:
        # arXiv ID — full URL form: https://arxiv.org/abs/2301.04567
        m_id = re.search(r'href="https://arxiv\.org/abs/([\d.v]+)"', block)
        if not m_id:
            # try relative form as fallback
            m_id = re.search(r'href="/abs/([\d.v]+)"', block)
        if not m_id:
            continue
        arxiv_id = m_id.group(1)

        # Title
        m_title = re.search(r'<p class="title[^"]*">([\s\S]+?)</p>', block)
        title   = clean(m_title.group(1)) if m_title else ""

        # Authors
        m_authors = re.search(r'<p class="authors">([\s\S]+?)</p>', block)
        authors   = clean(m_authors.group(1)) if m_authors else ""
        authors   = re.sub(r"^Authors:\s*", "", authors)

        # Abstract
        m_abs = re.search(r'<span class="abstract-short[^"]*">([\s\S]+?)</span>', block)
        if not m_abs:
            m_abs = re.search(r'<p class="abstract[^"]*">([\s\S]+?)</p>', block)
        abstract = clean(m_abs.group(1)) if m_abs else ""
        abstract = re.sub(r"^\s*Abstract:\s*", "", abstract)

        # Categories (tags)
        cats = re.findall(r'data-tooltip="([^"]+)"', block)

        # Submission date / year from arXiv ID (YYMM.NNNNN)
        m_yr = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
        if m_yr:
            yy   = int(m_yr.group(1))
            year = (2000 + yy) if yy < 90 else (1900 + yy)
        else:
            # Try to parse from "Submitted DD Month, YYYY"
            m_date = re.search(r"Submitted\s+\d+\s+\w+,\s+(\d{4})", block)
            year   = int(m_date.group(1)) if m_date else 0

        papers.append({
            "arxiv_id":   arxiv_id,
            "title":      title,
            "authors":    authors,
            "abstract":   abstract[:1000],
            "year":       year,
            "categories": cats,
            "url":        f"https://arxiv.org/abs/{arxiv_id}",
            "source_url": f"https://arxiv.org/src/{arxiv_id}",
        })

    return papers


def build_url(base_url: str, start: int) -> str:
    """Add or update the start= parameter."""
    parts = urllib.parse.urlparse(base_url)
    params = dict(urllib.parse.parse_qsl(parts.query))
    params["start"] = str(start)
    new_query = urllib.parse.urlencode(params)
    return urllib.parse.urlunparse(parts._replace(query=new_query))


def query_to_url(query: str) -> str:
    """Convert "Last, F I" to arXiv search URL."""
    # arXiv wants: query=Last%2C+F+I  (comma encoded, spaces as +)
    encoded = query.replace(" ", "+").replace(",", "%2C")
    return f"https://arxiv.org/search/?searchtype=author&query={encoded}"


def main():
    parser = argparse.ArgumentParser(
        description="Scrape arXiv author search page for all papers."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help='Author in arXiv format: "Last, F I"')
    group.add_argument("--url",   help="Full arXiv author search URL")
    parser.add_argument("--out",   default=None, help="Output JSONL (default: stdout)")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int,   default=None)
    args = parser.parse_args()

    base_url = args.url if args.url else query_to_url(args.query)
    print(f"Scraping: {base_url}", file=sys.stderr)

    out_fh   = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    all_papers = []
    start      = 0
    page_size  = 25  # arXiv default

    try:
        while True:
            url  = build_url(base_url, start)
            print(f"  Page start={start} ...", file=sys.stderr, flush=True)
            page = fetch_page(url, args.delay)
            if not page:
                break

            papers = parse_results(page)
            if not papers:
                print(f"  No results on this page — done.", file=sys.stderr)
                break

            for p in papers:
                out_fh.write(json.dumps(p, ensure_ascii=False) + "\n")
                out_fh.flush()
                all_papers.append(p)
                print(f"    {p['arxiv_id']}  ({p['year']})  {p['title'][:60]}", file=sys.stderr)

                if args.limit and len(all_papers) >= args.limit:
                    break

            if args.limit and len(all_papers) >= args.limit:
                break

            # Check if there's a next page
            if "next" not in page.lower() or len(papers) < page_size:
                break
            start += page_size

    finally:
        if args.out:
            out_fh.close()

    print(f"\nTotal: {len(all_papers)} papers", file=sys.stderr)
    if args.out:
        print(f"Written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
