#!/usr/bin/env python3
"""
author_pipeline.py — Full confidence chain pipeline for any author's papers.

Given an author name (in arXiv format: "Last, F I"), runs the complete pipeline:
  Phase 0: Discover papers by scraping arXiv author search page
  Phase 1: Download LaTeX source tarballs from arxiv.org/src/{id}
  Phase 2: Extract and chunk text (LaTeX or PDF fallback)
  Phase 3: Embed chunks with sentence-transformers
  Phase 4: Cluster into equivalence classes
  Phase 5: Extract keywords and structural claims
  Phase 6: Collate per equivalence class
  Phase 7: Find representatives + build cross-class manifest
  Phase 8: Score cross-class pairs (cosine + NLI)
  Phase 9: Generate knowledge graph YAML nodes

Each phase writes checkpoints to disk. Re-running resumes from last checkpoint.

Usage:
    python scripts/author_pipeline.py --author "Leek, J T" [options]

Options:
    --author        Author in arXiv format: "Last, F I" (required)
    --arxiv-url     Full arXiv author search URL (overrides --author query)
    --out-dir       Root output directory (default: {author_slug}/)
    --email         Email for Unpaywall API (PDF fallback for non-arXiv papers)
    --delay         Seconds between HTTP requests (default: 1.5)
    --limit         Stop after N papers (for testing)
    --n-clusters    Fixed number of equivalence classes (default: auto)
    --bridge-threshold  Cosine sim for secondary cluster membership (default: 0.50)
    --from-phase    Start from phase N (default: 0)
    --to-phase      Stop after phase N (default: 9)
    --no-nli        Skip NLI model in scoring (faster)
    --kg-dir        Knowledge graph nodes directory (default: knowledge-graph/nodes/)

Examples:
    # Full run for Jeffrey Leek
    python scripts/author_pipeline.py --author "Leek, J T"

    # Test with 5 papers, phases 0-2 only
    python scripts/author_pipeline.py --author "Leek, J T" --limit 5 --to-phase 2

    # Resume from clustering
    python scripts/author_pipeline.py --author "Leek, J T" --from-phase 4

    # Different author
    python scripts/author_pipeline.py --author "Friston, K J"

    # Use exact URL from browser
    python scripts/author_pipeline.py \\
        --arxiv-url "https://arxiv.org/search/?searchtype=author&query=Leek%2C+J+T"
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
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

# macOS Python 3.10 ships without bundled CA certs — create an unverified context
# as fallback so HTTP calls work without running the Apple certificate installer.
_SSL_CTX = ssl.create_default_context()
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode    = ssl.CERT_NONE

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

REPO_ROOT = Path(__file__).parent.parent.parent
TODAY     = date.today().isoformat()

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_bytes(url: str, delay: float, label: str = "",
                   retries: int = 4) -> bytes | None:
    for attempt in range(retries):
        wait = delay * (2 ** attempt)
        time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "open-knowledge-graph/1.0"})
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                backoff = 10 * (2 ** attempt)
                print(f"      Rate limited — waiting {backoff}s ...", flush=True)
                time.sleep(backoff)
                continue
            if label:
                print(f"      HTTP {e.code} ({label})", flush=True)
            return None
        except Exception as e:
            if label:
                print(f"      Error ({label}): {e}", flush=True)
            return None
    return None


def http_get_json(url: str, delay: float) -> dict | list | None:
    data = http_get_bytes(url, delay)
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 0: Discover papers by scraping arXiv author search
# ---------------------------------------------------------------------------

import html as _html


def _scrape_arxiv_page(page: str) -> list[dict]:
    """Parse one page of arXiv search results."""
    papers = []
    blocks = re.split(r'<li class="arxiv-result">', page)[1:]
    for block in blocks:
        # Full URL form: https://arxiv.org/abs/2301.04567
        m_id = re.search(r'href="https://arxiv\.org/abs/([\d.v]+)"', block)
        if not m_id:
            m_id = re.search(r'href="/abs/([\d.v]+)"', block)
        if not m_id:
            continue
        arxiv_id = m_id.group(1)

        m_title  = re.search(r'<p class="title[^"]*">([\s\S]+?)</p>', block)
        title    = _html.unescape(re.sub(r"<[^>]+>", " ", m_title.group(1))) if m_title else ""
        title    = re.sub(r"\s+", " ", title).strip()

        m_auth   = re.search(r'<p class="authors">([\s\S]+?)</p>', block)
        authors  = _html.unescape(re.sub(r"<[^>]+>", " ", m_auth.group(1))) if m_auth else ""
        authors  = re.sub(r"^Authors:\s*", "", re.sub(r"\s+", " ", authors).strip())

        m_abs    = re.search(r'<span class="abstract-short[^"]*">([\s\S]+?)</span>', block)
        abstract = _html.unescape(re.sub(r"<[^>]+>", " ", m_abs.group(1))) if m_abs else ""
        abstract = re.sub(r"\s+", " ", abstract).strip()[:800]

        cats     = re.findall(r'data-tooltip="([^"]+)"', block)

        m_yr     = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
        if m_yr:
            yy   = int(m_yr.group(1))
            year = (2000 + yy) if yy < 90 else (1900 + yy)
        else:
            year = 0

        papers.append({
            "arxiv_id":    arxiv_id,
            "paperId":     arxiv_id,
            "externalIds": {"ArXiv": arxiv_id},
            "title":       title,
            "authors":     authors,
            "abstract":    abstract,
            "year":        year,
            "categories":  cats,
            "url":         f"https://arxiv.org/abs/{arxiv_id}",
            "source_url":  f"https://arxiv.org/src/{arxiv_id}",
        })
    return papers


def _author_query_to_url(author: str) -> str:
    """Convert "Last, F I" to arXiv search URL."""
    encoded = author.replace(" ", "+").replace(",", "%2C")
    return f"https://arxiv.org/search/?searchtype=author&query={encoded}"


def _add_start(base_url: str, start: int) -> str:
    parts  = urllib.parse.urlparse(base_url)
    params = dict(urllib.parse.parse_qsl(parts.query))
    params["start"] = str(start)
    return urllib.parse.urlunparse(parts._replace(query=urllib.parse.urlencode(params)))


def phase0_discover(author: str, arxiv_url: str | None,
                    out_dir: Path, delay: float, limit: int | None) -> Path:
    papers_path = out_dir / "papers.jsonl"
    status_path = out_dir / "_status_phase0.json"

    done_ids = set()
    if status_path.exists():
        done_ids = set(json.loads(status_path.read_text()).get("done", []))
        print(f"  Resuming: {len(done_ids)} already fetched")

    base_url = arxiv_url if arxiv_url else _author_query_to_url(author)
    print(f"  Scraping: {base_url}")

    all_papers = []
    start      = 0
    page_size  = 25

    while True:
        url  = _add_start(base_url, start)
        data = http_get_bytes(url, delay)
        if not data:
            break
        page   = data.decode("utf-8", errors="replace")
        papers = _scrape_arxiv_page(page)
        if not papers:
            break
        all_papers.extend(papers)
        print(f"  Page start={start}: {len(papers)} papers (total {len(all_papers)})")
        if limit and len(all_papers) >= limit:
            all_papers = all_papers[:limit]
            break
        if len(papers) < page_size:
            break
        start += page_size

    # Write new papers
    fetched = 0
    with open(papers_path, "a", encoding="utf-8") as fh:
        for paper in all_papers:
            pid = paper["arxiv_id"]
            if pid in done_ids:
                continue
            fh.write(json.dumps(paper, ensure_ascii=False) + "\n")
            fh.flush()
            done_ids.add(pid)
            fetched += 1
            print(f"    + [{paper['year']}] {paper['title'][:65]}")

    status_path.write_text(json.dumps({"done": sorted(done_ids)}))
    print(f"  Phase 0 complete: {len(done_ids)} papers ({fetched} new)")
    return papers_path


# ---------------------------------------------------------------------------
# Phase 1: Download sources
# ---------------------------------------------------------------------------

def _try_arxiv(paper: dict, dest: Path, delay: float) -> bool:
    # Prefer source_url from scraper; fall back to constructing from arxiv_id
    src_url  = paper.get("source_url", "")
    arxiv_id = paper.get("arxiv_id") or paper.get("externalIds", {}).get("ArXiv", "")
    url      = src_url if src_url else (f"https://arxiv.org/src/{arxiv_id}" if arxiv_id else "")
    if not url:
        return False
    data = http_get_bytes(url, delay, f"arXiv:{arxiv_id}")
    if not data:
        return False
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            members = [m for m in tar.getmembers() if m.name.lower().endswith((".tex", ".bbl"))]
            if not members:
                return False
            for m in members:
                f = tar.extractfile(m)
                if f:
                    (dest / Path(m.name).name).write_bytes(f.read())
        (dest / "source_type").write_text("arxiv")
        return True
    except tarfile.TarError:
        if data[:1] in (b"\\", b"%"):
            (dest / "paper.tex").write_bytes(data)
            (dest / "source_type").write_text("arxiv")
            return True
    return False


def _try_biorxiv(paper: dict, dest: Path, delay: float) -> bool:
    doi = paper.get("externalIds", {}).get("DOI", "")
    if not doi or "10.1101" not in doi:
        return False
    data = http_get_bytes(f"https://www.biorxiv.org/content/{doi}.full.pdf", delay, "bioRxiv")
    if not data or not data.startswith(b"%PDF"):
        return False
    (dest / "paper.pdf").write_bytes(data)
    (dest / "source_type").write_text("biorxiv")
    return True


def _try_unpaywall(paper: dict, dest: Path, delay: float, email: str) -> bool:
    doi = paper.get("externalIds", {}).get("DOI", "")
    if not doi or not email:
        return False
    meta = http_get_json(f"https://api.unpaywall.org/v2/{doi}?email={email}", delay)
    if not meta:
        return False
    loc     = meta.get("best_oa_location") or {}
    pdf_url = loc.get("url_for_pdf") or loc.get("url", "")
    if not pdf_url:
        return False
    data = http_get_bytes(pdf_url, delay, "Unpaywall")
    if not data or not data.startswith(b"%PDF"):
        return False
    (dest / "paper.pdf").write_bytes(data)
    (dest / "source_type").write_text("unpaywall")
    return True


def phase1_sources(papers_path: Path, out_dir: Path, delay: float, email: str) -> None:
    status_path = out_dir / "_status_phase1.json"
    done_ids    = set()
    if status_path.exists():
        s        = json.loads(status_path.read_text())
        done_ids = set(s.get("done", []))
        print(f"  Phase 1 resuming: {len(done_ids)} already done")

    papers = [json.loads(l) for l in open(papers_path, encoding="utf-8") if l.strip()]
    fetched, not_found = 0, 0

    for paper in papers:
        pid   = paper.get("paperId", "?")
        title = paper.get("title", "")[:55]
        if pid in done_ids:
            continue

        dest = out_dir / pid
        dest.mkdir(exist_ok=True)
        print(f"    {title}", end=" ", flush=True)

        ok = _try_arxiv(paper, dest, delay)
        if ok:
            print("→ arXiv", flush=True)
        if not ok:
            ok = _try_biorxiv(paper, dest, delay)
            if ok:
                print("→ bioRxiv", flush=True)
        if not ok and email:
            ok = _try_unpaywall(paper, dest, delay, email)
            if ok:
                print("→ Unpaywall", flush=True)
        if not ok:
            (dest / "source_type").write_text("none")
            print("→ not found", flush=True)
            not_found += 1
        else:
            fetched += 1

        done_ids.add(pid)
        status_path.write_text(json.dumps({"done": sorted(done_ids),
                                           "fetched": fetched, "not_found": not_found}))

    print(f"  Phase 1 complete: {fetched} fetched, {not_found} not found")


# ---------------------------------------------------------------------------
# Phase 2: Extract and chunk text
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(
    r"^\\(?:sub)*section\*?\{(.+?)\}|^\\paragraph\*?\{(.+?)\}", re.MULTILINE
)

def _chunk_tex(tex_text: str, source: str, paper_id: str, base_index: int) -> list[dict]:
    # Isolate body
    m = re.search(r"\\begin\{document\}([\s\S]+?)\\end\{document\}", tex_text)
    body = m.group(1) if m else tex_text

    # Split on section boundaries
    splits = [(m.start(), (m.group(1) or m.group(2)).strip())
              for m in SECTION_RE.finditer(body)]

    sections = []
    if splits:
        if splits[0][0] > 0 and body[:splits[0][0]].strip():
            sections.append(("[body]", body[:splits[0][0]]))
        for i, (pos, title) in enumerate(splits):
            end = splits[i+1][0] if i+1 < len(splits) else len(body)
            sections.append((title, body[pos:end]))
    else:
        sections = [("[body]", body)]

    chunks = []
    for title, text in sections:
        text = text.strip()
        if len(text) < 100:
            continue
        chunks.append({
            "chunk_index":   base_index + len(chunks),
            "paper_id":      paper_id,
            "source":        source,
            "source_type":   "tex",
            "section_title": title,
            "section_path":  [title],
            "char_count":    len(text),
            "token_estimate": len(text) // 4,
            "text":          text,
        })
    return chunks


def _chunk_pdf(pdf_bytes: bytes, source: str, paper_id: str, base_index: int) -> list[dict]:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(io.BytesIO(pdf_bytes))
    except Exception as e:
        print(f"      PDF extract failed: {e}", flush=True)
        return []

    # Split into ~3000-char paragraph chunks
    paras  = re.split(r"\n{2,}", text.strip())
    chunks = []
    buf    = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) < 3000:
            buf = (buf + "\n\n" + para).strip()
        else:
            if len(buf) >= 100:
                chunks.append({
                    "chunk_index":   base_index + len(chunks),
                    "paper_id":      paper_id,
                    "source":        source,
                    "source_type":   "pdf",
                    "section_title": f"[part {len(chunks)+1}]",
                    "section_path":  [],
                    "char_count":    len(buf),
                    "token_estimate": len(buf) // 4,
                    "text":          buf,
                })
            buf = para
    if len(buf) >= 100:
        chunks.append({
            "chunk_index":   base_index + len(chunks),
            "paper_id":      paper_id,
            "source":        source,
            "source_type":   "pdf",
            "section_title": f"[part {len(chunks)+1}]",
            "section_path":  [],
            "char_count":    len(buf),
            "token_estimate": len(buf) // 4,
            "text":          buf,
        })
    return chunks


def phase2_chunk(papers_path: Path, sources_dir: Path, out_dir: Path) -> Path:
    chunks_path = out_dir / "chunks.jsonl"
    status_path = out_dir / "_status_phase2.json"

    done_ids = set()
    if status_path.exists():
        s        = json.loads(status_path.read_text())
        done_ids = set(s.get("done", []))

    papers     = [json.loads(l) for l in open(papers_path, encoding="utf-8") if l.strip()]
    base_index = 0

    # Count existing chunks to get base_index
    if chunks_path.exists():
        with open(chunks_path) as fh:
            for line in fh:
                if line.strip():
                    base_index += 1

    written = 0
    with open(chunks_path, "a", encoding="utf-8") as out_fh:
        for paper in papers:
            pid   = paper.get("paperId", "?")
            title = paper.get("title", "")[:55]
            if pid in done_ids:
                continue

            src_dir     = sources_dir / pid
            source_type = (src_dir / "source_type").read_text().strip() if (src_dir / "source_type").exists() else "none"

            if source_type == "none":
                done_ids.add(pid)
                continue

            print(f"    Chunking [{source_type}] {title}", flush=True)
            chunks = []

            if source_type == "arxiv":
                for tex_file in sorted(src_dir.glob("*.tex")):
                    text = tex_file.read_text(encoding="utf-8", errors="replace")
                    chunks.extend(_chunk_tex(text, tex_file.name, pid, base_index + len(chunks)))
            else:
                pdf_files = list(src_dir.glob("*.pdf"))
                if pdf_files:
                    chunks.extend(_chunk_pdf(pdf_files[0].read_bytes(), pdf_files[0].name, pid, base_index))

            for c in chunks:
                out_fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            out_fh.flush()
            base_index += len(chunks)
            written    += len(chunks)
            done_ids.add(pid)

            status_path.write_text(json.dumps({"done": sorted(done_ids), "chunks_written": base_index}))
            print(f"      → {len(chunks)} chunks")

    print(f"  Phase 2 complete: {base_index} total chunks in {chunks_path.name}")
    return chunks_path


# ---------------------------------------------------------------------------
# Phase 3: Embed
# ---------------------------------------------------------------------------

def phase3_embed(chunks_path: Path, out_dir: Path, model_name: str) -> tuple[np.ndarray, list]:
    from sentence_transformers import SentenceTransformer

    emb_path  = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.json"

    chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8") if l.strip()]
    print(f"  Embedding {len(chunks)} chunks with {model_name} ...")

    model      = SentenceTransformer(model_name)
    texts      = [f"{c.get('section_title','')}\n\n{c.get('text','')}"[:4096] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(str(emb_path), embeddings)
    meta = {"model": model_name, "n_chunks": len(chunks), "shape": list(embeddings.shape),
            "chunks": [{k: c[k] for k in ("chunk_index","source","section_title","section_path",
                                           "char_count","token_estimate","paper_id")} for c in chunks]}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"  Phase 3 complete: embeddings shape {embeddings.shape}")
    return embeddings, chunks


# ---------------------------------------------------------------------------
# Phase 4: Cluster
# ---------------------------------------------------------------------------

def phase4_cluster(embeddings: np.ndarray, metadata: dict, out_dir: Path,
                   n_clusters: int | None, bridge_threshold: float) -> dict:
    normed = normalize(embeddings, norm="l2")
    n      = len(normed)

    if n_clusters:
        k = n_clusters
        print(f"  Using fixed k={k}")
    else:
        print(f"  Auto-selecting k via silhouette (max 15) ...")
        best_score, k = -1.0, 2
        for ki in range(2, min(16, n)):
            labels = KMeans(n_clusters=ki, n_init=10, random_state=42).fit_predict(normed)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(normed, labels, metric="cosine")
            print(f"    k={ki:2d}  sil={score:.4f}", flush=True)
            if score > best_score:
                best_score, k = score, ki
        print(f"  → Best k={k}")

    km             = KMeans(n_clusters=k, n_init=20, random_state=42)
    primary_labels = km.fit_predict(normed)
    centroids      = km.cluster_centers_

    chunk_to_classes = defaultdict(list)
    class_to_chunks  = defaultdict(list)
    for idx, label in enumerate(primary_labels):
        cid = f"EC{label:02d}"
        chunk_to_classes[idx].append(cid)
        class_to_chunks[cid].append(idx)

    sims = normed @ centroids.T
    bridge_chunks, bridge_pairs = {}, defaultdict(set)
    for idx in range(n):
        pc = f"EC{primary_labels[idx]:02d}"
        for j in range(k):
            sc = f"EC{j:02d}"
            if sc != pc and sims[idx, j] >= bridge_threshold:
                chunk_to_classes[idx].append(sc)
                class_to_chunks[sc].append(idx)
        if len(chunk_to_classes[idx]) > 1:
            bridge_chunks[idx] = chunk_to_classes[idx]
            for i_c, ca in enumerate(chunk_to_classes[idx]):
                for cb in chunk_to_classes[idx][i_c+1:]:
                    bridge_pairs[tuple(sorted([ca, cb]))].add(idx)

    chunks_meta = metadata["chunks"]

    def label_for(idxs):
        words = defaultdict(int)
        for i in idxs:
            for w in chunks_meta[i]["section_title"].lower().replace(">","").split():
                w = w.strip("[].,;:()")
                if len(w) > 3 and w not in {"part","with","and","the","for","from","that"}:
                    words[w] += 1
        top = sorted(words, key=lambda x: -words[x])[:3]
        return "-".join(top) if top else "cluster"

    classes_out = {}
    for cid, idxs in sorted(class_to_chunks.items()):
        idxs = sorted(set(idxs))
        ki   = int(cid[2:])
        sims_to_centroid = normed[idxs] @ centroids[ki]
        rep  = idxs[int(np.argmax(sims_to_centroid))]
        classes_out[cid] = {
            "class_id": cid, "label": label_for(idxs),
            "n_chunks": len(idxs), "chunk_indices": idxs,
            "representative": rep,
            "representative_title": chunks_meta[rep]["section_title"],
        }

    bridge_pairs_out = sorted(
        [{"class_a": a, "class_b": b, "weight": len(s), "chunk_indices": sorted(s)}
         for (a,b), s in bridge_pairs.items()],
        key=lambda x: -x["weight"]
    )

    (out_dir / "classes.json").write_text(json.dumps(classes_out, indent=2, ensure_ascii=False))
    (out_dir / "bridge_chunks.json").write_text(json.dumps({str(k):v for k,v in bridge_chunks.items()}, indent=2))
    (out_dir / "bridge_pairs.json").write_text(json.dumps(bridge_pairs_out, indent=2))
    (out_dir / "chunk_class_membership.json").write_text(
        json.dumps({str(k):v for k,v in sorted(chunk_to_classes.items())}, indent=2))
    summary = {"n_chunks": n, "n_classes": len(classes_out),
               "bridge_threshold": bridge_threshold,
               "n_bridge_chunks": len(bridge_chunks), "n_bridge_pairs": len(bridge_pairs_out),
               "classes_summary": [{"class_id": cid, "label": d["label"], "n_chunks": d["n_chunks"],
                                    "representative_title": d["representative_title"]}
                                   for cid, d in sorted(classes_out.items())]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"  Phase 4 complete: {len(classes_out)} classes, {len(bridge_chunks)} bridge chunks")
    for cid, d in sorted(classes_out.items()):
        print(f"    {cid} [{d['label'][:40]:40s}] {d['n_chunks']} chunks")
    return classes_out


# ---------------------------------------------------------------------------
# Phase 5: Extract keywords + claims
# ---------------------------------------------------------------------------

THEOREM_RE    = re.compile(r'\\textbf\{(Theorem|Lemma|Corollary)\s*([\d]+[a-z]?)\}', re.I)
DEF_RE        = re.compile(r'\\textbf\{(Definition)\s*([\d]+[a-z]?)\}', re.I)
CITE_RE       = re.compile(r'\{?\[([A-Za-z][A-Za-z0-9]+\d{4}[a-z]?)\]\}?')
CLAIM_SENT_RE = re.compile(
    r'(?:we show|we find|we demonstrate|our results|we propose|results indicate|'
    r'we observe|we conclude|this shows|this demonstrates)[^.!?]{10,200}[.!?]',
    re.IGNORECASE
)
LATEX_STRIP   = re.compile(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^}]*)\})?')
MATH_STRIP    = re.compile(r'\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$[^$]+\$')


def _strip_for_kw(text):
    text = MATH_STRIP.sub(" [MATH] ", text)
    text = LATEX_STRIP.sub(lambda m: m.group(1) or "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tfidf_keyphrases(texts: list[str], top_n: int = 8) -> list[list[tuple]]:
    """Extract keyphrases from a corpus using TF-IDF. No model loading required."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    if not texts:
        return []

    # Build 1-3 gram TF-IDF over the entire corpus for IDF context
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words="english",
        max_features=20000,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return [[] for _ in texts]

    feature_names = vectorizer.get_feature_names_out()
    results = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = row.argsort()[::-1][:top_n]
        kws = [(feature_names[j], round(float(row[j]), 4)) for j in top_indices if row[j] > 0]
        results.append(kws)
    return results


def phase5_keywords(chunks_path: Path, out_dir: Path, top_n: int = 8) -> None:
    status_path = out_dir / "_status_phase5.json"
    done = set()
    if status_path.exists():
        done = set(json.loads(status_path.read_text()).get("done", []))

    chunks   = [json.loads(l) for l in open(chunks_path, encoding="utf-8") if l.strip()]
    failed   = []

    # Compute TF-IDF keyphrases over the full corpus in one pass (memory-efficient)
    all_texts  = [_strip_for_kw(c["text"]) for c in chunks]
    all_kws    = _tfidf_keyphrases(all_texts, top_n=top_n)

    for chunk, kws in zip(chunks, all_kws):
        idx = chunk["chunk_index"]
        if idx in done:
            continue
        out_path = out_dir / f"chunk_{idx:04d}.json"
        text     = chunk["text"]

        try:
            keyphrases = [{"phrase": kw, "score": s} for kw, s in kws if kw]

            claim_sentences = CLAIM_SENT_RE.findall(text)

            result = {
                "chunk_index":   idx,
                "paper_id":      chunk.get("paper_id", ""),
                "source":        chunk.get("source", ""),
                "source_type":   chunk.get("source_type", ""),
                "section_title": chunk.get("section_title", ""),
                "section_path":  chunk.get("section_path", []),
                "char_count":    chunk.get("char_count", 0),
                "keyphrases":    keyphrases,
                "structure": {
                    "theorems":       [{"type": m.group(1), "number": m.group(2)} for m in THEOREM_RE.finditer(text)],
                    "definitions":    [{"type": m.group(1), "number": m.group(2)} for m in DEF_RE.finditer(text)],
                    "citations":      list(dict.fromkeys(m.group(1) for m in CITE_RE.finditer(text))),
                    "claim_sentences": [s.strip() for s in claim_sentences[:5]],
                    "has_math":       bool(re.search(r'\\\(|\\\[|\$', text)),
                },
            }
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            done.add(idx)
        except Exception as e:
            failed.append({"chunk_index": idx, "error": str(e)})

    status_path.write_text(json.dumps({"done": sorted(done), "failed": failed, "total": len(chunks)}))
    print(f"  Phase 5 complete: {len(done)}/{len(chunks)} chunks")


# ---------------------------------------------------------------------------
# Phase 6: Collate per class
# ---------------------------------------------------------------------------

def phase6_collate(clusters_dir: Path, keywords_dir: Path, out_dir: Path) -> None:
    classes    = json.loads((clusters_dir / "classes.json").read_text())
    membership = json.loads((clusters_dir / "chunk_class_membership.json").read_text())

    for cid, cdata in sorted(classes.items()):
        collated = []
        for idx in cdata["chunk_indices"]:
            kw_path = keywords_dir / f"chunk_{idx:04d}.json"
            if not kw_path.exists():
                continue
            kw = json.loads(kw_path.read_text())
            all_cls = membership.get(str(idx), [cid])
            kw["is_bridge"]          = len(all_cls) > 1
            kw["all_classes"]        = all_cls
            kw["other_classes"]      = [c for c in all_cls if c != cid]
            kw["is_representative"]  = (idx == cdata["representative"])
            collated.append(kw)

        collated.sort(key=lambda c: (0 if c.get("is_representative") else 1, c["chunk_index"]))
        out = {"class_id": cid, "label": cdata["label"],
               "n_chunks": len(collated),
               "representative_title": cdata["representative_title"],
               "chunks": collated}
        (out_dir / f"class_{cid}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"  Phase 6 complete: {len(classes)} class files")


# ---------------------------------------------------------------------------
# Phase 7: Representatives + cross-manifest
# ---------------------------------------------------------------------------

def phase7_representatives(clusters_dir: Path, collated_dir: Path,
                            emb_dir: Path, out_dir: Path, top_n: int = 3) -> None:
    embeddings = np.load(str(emb_dir / "embeddings.npy"))
    metadata   = json.loads((emb_dir / "metadata.json").read_text())
    classes    = json.loads((clusters_dir / "classes.json").read_text())
    bridge_pairs = json.loads((clusters_dir / "bridge_pairs.json").read_text())
    normed       = normalize(embeddings, norm="l2")
    chunks_meta  = metadata["chunks"]

    class_centroids = {}
    for cid, cdata in sorted(classes.items()):
        col_path = collated_dir / f"class_{cid}.json"
        if not col_path.exists():
            continue
        idxs     = cdata["chunk_indices"]
        centroid = normalize(normed[idxs].mean(axis=0, keepdims=True), norm="l2")[0]
        class_centroids[cid] = centroid
        sims     = normed[idxs] @ centroid
        ranked   = sorted(zip(idxs, sims.tolist()), key=lambda x: -x[1])

        collated = json.loads(col_path.read_text())
        phrase_scores = defaultdict(float)
        all_theorems, all_defs, all_cites, all_claims = [], [], [], []

        for cd in collated["chunks"]:
            for kw in cd.get("keyphrases", []):
                phrase_scores[kw["phrase"]] += kw["score"]
            s = cd.get("structure", {})
            all_theorems.extend(s.get("theorems", []))
            all_defs.extend(s.get("definitions", []))
            all_cites.extend(s.get("citations", []))
            all_claims.extend(s.get("claim_sentences", []))

        top_phrases = sorted(phrase_scores, key=lambda p: -phrase_scores[p])[:10]
        result = {
            "class_id": cid, "label": cdata["label"],
            "n_chunks": len(idxs),
            "representative": {"chunk_index": ranked[0][0],
                               "section_title": chunks_meta[ranked[0][0]]["section_title"],
                               "similarity": round(ranked[0][1], 4)},
            "top_chunks": [{"chunk_index": i, "section_title": chunks_meta[i]["section_title"],
                            "similarity": round(s, 4)} for i, s in ranked[:top_n]],
            "top_keyphrases": [{"phrase": p, "score": round(phrase_scores[p], 4)} for p in top_phrases],
            "theorems": all_theorems, "definitions": all_defs,
            "citations": list(dict.fromkeys(all_cites)),
            "claim_sentences": list(dict.fromkeys(all_claims))[:10],
            "bridge_chunks": [c["chunk_index"] for c in collated["chunks"] if c.get("is_bridge")],
        }
        (out_dir / f"class_{cid}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Cross-manifest
    bridge_set  = {(bp["class_a"], bp["class_b"]) for bp in bridge_pairs}
    cross_pairs = []
    for bp in bridge_pairs:
        cross_pairs.append({**bp, "priority": "bridge",
                            "label_a": classes[bp["class_a"]]["label"],
                            "label_b": classes[bp["class_b"]]["label"]})

    class_ids = sorted(class_centroids.keys())
    for i, ca in enumerate(class_ids):
        for cb in class_ids[i+1:]:
            pair = tuple(sorted([ca, cb]))
            if pair in bridge_set:
                continue
            sim = float(class_centroids[ca] @ class_centroids[cb])
            cross_pairs.append({"class_a": pair[0], "class_b": pair[1],
                                "priority": "similarity", "weight": round(sim, 4),
                                "label_a": classes[pair[0]]["label"],
                                "label_b": classes[pair[1]]["label"]})

    cross_pairs.sort(key=lambda x: (0 if x["priority"] == "bridge" else 1, -x["weight"]))
    (out_dir / "cross_manifest.json").write_text(json.dumps(cross_pairs, indent=2, ensure_ascii=False))
    print(f"  Phase 7 complete: {len(classes)} reps, {len(cross_pairs)} cross-pairs")


# ---------------------------------------------------------------------------
# Phase 8: Score cross-class pairs
# ---------------------------------------------------------------------------

def phase8_score(reps_dir: Path, emb_dir: Path, chunks_path: Path,
                 out_dir: Path, no_nli: bool, top: int | None) -> None:
    manifest = json.loads((reps_dir / "cross_manifest.json").read_text())
    if top:
        manifest = manifest[:top]

    embeddings = np.load(str(emb_dir / "embeddings.npy"))
    normed     = normalize(embeddings, norm="l2")
    chunks     = {c["chunk_index"]: c
                  for c in (json.loads(l) for l in open(chunks_path, encoding="utf-8") if l.strip())}

    nli_pipe = None
    if not no_nli:
        try:
            from transformers import pipeline as hf_pipeline
            print("  Loading NLI model ...")
            nli_pipe = hf_pipeline("zero-shot-classification",
                                   model="cross-encoder/nli-deberta-v3-small", device=-1)
        except Exception as e:
            print(f"  NLI unavailable ({e}), cosine only", file=sys.stderr)

    out_path    = out_dir / "pairs.jsonl"
    status_path = out_dir / "_status_phase8.json"

    done_keys = set()
    if status_path.exists():
        s         = json.loads(status_path.read_text())
        done_keys = {(r["class_a"], r["class_b"]) for r in s.get("done", [])}

    done = []
    with open(out_path, "a", encoding="utf-8") as fh:
        for pair in manifest:
            ca, cb = pair["class_a"], pair["class_b"]
            if (ca, cb) in done_keys:
                continue
            ra_path = reps_dir / f"class_{ca}.json"
            rb_path = reps_dir / f"class_{cb}.json"
            if not ra_path.exists() or not rb_path.exists():
                continue

            ra  = json.loads(ra_path.read_text())
            rb  = json.loads(rb_path.read_text())
            ia  = ra["representative"]["chunk_index"]
            ib  = rb["representative"]["chunk_index"]
            cos = float(normed[ia] @ normed[ib])

            nli_result = None
            if nli_pipe:
                try:
                    text_a = chunks.get(ia, {}).get("text", "")[:512]
                    result = nli_pipe(text_a, candidate_labels=["entailment","neutral","contradiction"])
                    nli_result = {"top": result["labels"][0],
                                  "labels": result["labels"],
                                  "scores": [round(s, 4) for s in result["scores"]]}
                except Exception as e:
                    nli_result = {"error": str(e)}

            scored = {**pair, "cosine_similarity": round(cos, 4), "nli": nli_result,
                      "rep_a_title": ra["representative"]["section_title"],
                      "rep_b_title": rb["representative"]["section_title"],
                      "top_phrases_a": [k["phrase"] for k in ra.get("top_keyphrases",[])[:5]],
                      "top_phrases_b": [k["phrase"] for k in rb.get("top_keyphrases",[])[:5]]}
            fh.write(json.dumps(scored, ensure_ascii=False) + "\n")
            fh.flush()
            done.append({"class_a": ca, "class_b": cb})
            nli_top = nli_result.get("top","n/a") if isinstance(nli_result, dict) else "n/a"
            print(f"    {ca} × {cb}  cos={cos:.3f}  nli={nli_top}  [{pair['priority']}]")

    status_path.write_text(json.dumps({"done": done, "total": len(manifest)}))
    print(f"  Phase 8 complete: {len(done)} pairs scored")


# ---------------------------------------------------------------------------
# Phase 9: Generate knowledge graph YAML nodes
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text.title().replace(" ", ""))[:20]


def phase9_graph(papers_path: Path, reps_dir: Path, clusters_dir: Path,
                 scores_path: Path, kg_dir: Path, author_name: str) -> None:
    papers  = {p["paperId"]: p
               for p in (json.loads(l) for l in open(papers_path, encoding="utf-8") if l.strip())}
    classes = json.loads((clusters_dir / "classes.json").read_text())

    ref_dir  = kg_dir / "references"
    ovl_dir  = kg_dir / "overlap"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ovl_dir.mkdir(parents=True, exist_ok=True)

    # Handle arXiv format "Last, F I" — take the part before the comma
    last_name = author_name.split(",")[0].strip()

    # REF nodes — one per paper
    existing = {f.stem for f in ref_dir.glob("*.yaml")}
    for pid, paper in papers.items():
        year  = str(paper.get("year", "unknown"))
        base  = f"REF-{last_name}{year}"
        node_id = base
        suffix  = 2
        while node_id in existing:
            node_id = f"{base}-{suffix}"
            suffix += 1
        existing.add(node_id)

        title  = paper.get("title", "Unknown")
        ext    = paper.get("externalIds", {})
        arxiv  = ext.get("ArXiv", "")
        doi    = ext.get("DOI", "")
        url    = f"https://arxiv.org/abs/{arxiv}" if arxiv else (f"https://doi.org/{doi}" if doi else "")
        raw_authors = paper.get("authors", [])
        if isinstance(raw_authors, str):
            auths = raw_authors[:200]
        else:
            auths = ", ".join(a.get("name","") for a in raw_authors[:5])

        yaml = f"""id: {node_id}
type: reference
fidelity: low
name: "{last_name} ({year}) — {title[:60]}{'...' if len(title)>60 else ''}"
statement: |
  {title}

  Authors: {auths}
  Year: {year}
  Citations: {paper.get('citationCount', 0)}

provenance:
  attribution:
    author: "{auths}"
    source: "Semantic Scholar"
    date: "{year}"
    arxiv_id: "{arxiv}"
    doi: "{doi}"
    url: "{url}"
  evidence:
    type: cited
    description: |
      Imported via author_pipeline.py for {author_name}.
    references: []
  derivation:
    from: []
    method: "Fetched from Semantic Scholar via author_pipeline.py"

edges: []
"""
        out_path = ref_dir / f"{node_id}.yaml"
        if not out_path.exists():
            out_path.write_text(yaml, encoding="utf-8")
            print(f"    REF: {node_id}")

    # Overlap nodes — one per equivalence class with ≥2 chunks from ≥2 papers
    for cid, cdata in sorted(classes.items()):
        rep_path = reps_dir / f"class_{cid}.json"
        if not rep_path.exists():
            continue
        rep = json.loads(rep_path.read_text())

        label    = cdata["label"]
        node_id  = f"OV-{last_name}-{_slugify(label)}"
        phrases  = [k["phrase"] for k in rep.get("top_keyphrases", [])[:5]]
        claims   = rep.get("claim_sentences", [])[:3]

        yaml = f"""id: {node_id}
type: overlap
fidelity: medium
name: "{author_name} — {label}"
statement: |
  Equivalency class extracted from {author_name}'s papers.
  Top keyphrases: {', '.join(phrases)}
  Representative section: {rep['representative']['section_title']}

  Key empirical claims found in this cluster:
{chr(10).join('  - ' + c[:120] for c in claims) if claims else '  (none extracted)'}

provenance:
  attribution:
    author: "author_pipeline.py"
    source: "Semantic Scholar + source tarballs"
    date: "{TODAY}"
  evidence:
    type: empirical
    description: |
      Cluster of {cdata['n_chunks']} chunks from {author_name}'s papers.
      Identified via sentence-transformer embeddings + KMeans clustering.
    references: []
  derivation:
    from: []
    method: "Embedding-based clustering via author_pipeline.py"

edges: []
"""
        out_path = ovl_dir / f"{node_id}.yaml"
        if not out_path.exists():
            out_path.write_text(yaml, encoding="utf-8")
            print(f"    OVL: {node_id}")

    print(f"  Phase 9 complete: YAML nodes written to {kg_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Full confidence chain pipeline for any author."
    )
    parser.add_argument("--author",            required=True, help='Author in arXiv format: "Last, F I"')
    parser.add_argument("--arxiv-url",         default=None, help="Full arXiv author search URL")
    parser.add_argument("--out-dir",          default=None)
    parser.add_argument("--email",            default="", help="Email for Unpaywall")
    parser.add_argument("--delay",            type=float, default=1.5)
    parser.add_argument("--limit",            type=int,   default=None)
    parser.add_argument("--n-clusters",       type=int,   default=None)
    parser.add_argument("--bridge-threshold", type=float, default=0.50)
    parser.add_argument("--from-phase",       type=int,   default=0)
    parser.add_argument("--to-phase",         type=int,   default=9)
    parser.add_argument("--no-nli",           action="store_true")
    parser.add_argument("--model",            default="all-MiniLM-L6-v2")
    parser.add_argument("--kg-dir",           default=None)
    args = parser.parse_args()

    base = Path(args.out_dir) if args.out_dir else REPO_ROOT / slug(args.author)
    base.mkdir(parents=True, exist_ok=True)

    dirs = {
        "papers":   base / "00-papers",
        "sources":  base / "01-sources",
        "chunks":   base / "02-chunks",
        "emb":      base / "03-embeddings",
        "clusters": base / "04-clusters",
        "keywords": base / "05-keywords",
        "collated": base / "06-collated",
        "reps":     base / "07-representatives",
        "scores":   base / "08-scores",
        "kg":       Path(args.kg_dir) if args.kg_dir else REPO_ROOT / "knowledge-graph" / "nodes",
    }
    for d in list(dirs.values())[:-1]:
        d.mkdir(parents=True, exist_ok=True)

    papers_path = dirs["papers"] / "papers.jsonl"
    chunks_path = dirs["chunks"] / "chunks.jsonl"
    scores_path = dirs["scores"] / "pairs.jsonl"

    p = args.from_phase

    if p <= 0 <= args.to_phase:
        print("\n=== Phase 0: Discover papers ===")
        phase0_discover(args.author, args.arxiv_url, dirs["papers"], args.delay, args.limit)

    if p <= 1 <= args.to_phase:
        print("\n=== Phase 1: Download sources ===")
        phase1_sources(papers_path, dirs["sources"], args.delay, args.email)

    if p <= 2 <= args.to_phase:
        print("\n=== Phase 2: Chunk text ===")
        phase2_chunk(papers_path, dirs["sources"], dirs["chunks"])

    embeddings, chunks = None, None
    if p <= 3 <= args.to_phase:
        print("\n=== Phase 3: Embed ===")
        embeddings, chunks = phase3_embed(chunks_path, dirs["emb"], args.model)
    elif any(p <= ph <= args.to_phase for ph in range(4, 10)):
        # Load cached embeddings for later phases
        if (dirs["emb"] / "embeddings.npy").exists():
            embeddings = np.load(str(dirs["emb"] / "embeddings.npy"))
            metadata   = json.loads((dirs["emb"] / "metadata.json").read_text())

    if p <= 4 <= args.to_phase and embeddings is not None:
        print("\n=== Phase 4: Cluster ===")
        metadata = json.loads((dirs["emb"] / "metadata.json").read_text())
        phase4_cluster(embeddings, metadata, dirs["clusters"], args.n_clusters, args.bridge_threshold)

    if p <= 5 <= args.to_phase:
        print("\n=== Phase 5: Extract keywords ===")
        phase5_keywords(chunks_path, dirs["keywords"])

    if p <= 6 <= args.to_phase:
        print("\n=== Phase 6: Collate classes ===")
        phase6_collate(dirs["clusters"], dirs["keywords"], dirs["collated"])

    if p <= 7 <= args.to_phase:
        print("\n=== Phase 7: Representatives ===")
        phase7_representatives(dirs["clusters"], dirs["collated"], dirs["emb"], dirs["reps"])

    if p <= 8 <= args.to_phase:
        print("\n=== Phase 8: Score cross-pairs ===")
        phase8_score(dirs["reps"], dirs["emb"], chunks_path, dirs["scores"], args.no_nli, top=None)

    if p <= 9 <= args.to_phase:
        print("\n=== Phase 9: Generate KG nodes ===")
        phase9_graph(papers_path, dirs["reps"], dirs["clusters"], scores_path, dirs["kg"], args.author)

    print(f"\nDone. Output in {base}/")


if __name__ == "__main__":
    main()
