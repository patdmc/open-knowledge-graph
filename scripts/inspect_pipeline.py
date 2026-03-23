#!/usr/bin/env python3
"""
inspect_pipeline.py — Inspect pipeline state for any author directory.

Shows per-phase status, equivalence class summaries, top cross-pair scores,
and representative keyphrases.

Usage:
    python scripts/inspect_pipeline.py --author-dir jeffrey_leek/
    python scripts/inspect_pipeline.py --author-dir friston_k_j/ --phase 8
    python scripts/inspect_pipeline.py --author-dir jeffrey_leek/ --show-classes
    python scripts/inspect_pipeline.py --author-dir jeffrey_leek/ --show-pairs
    python scripts/inspect_pipeline.py --author-dir jeffrey_leek/ --show-class EC05
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                lines.append(json.loads(line))
            except Exception:
                pass
    return lines


def fmt_bar(done, total, width=20):
    if total == 0:
        return "empty"
    pct = done / total
    filled = int(pct * width)
    bar = ("█" * filled).ljust(width)
    return f"[{bar}] {done}/{total} ({int(pct*100)}%)"


# ---------------------------------------------------------------------------
# Phase-level status checks
# ---------------------------------------------------------------------------

def check_phase0(d: Path) -> str:
    papers = load_jsonl(d / "00-papers" / "papers.jsonl")
    if not papers:
        status = load_json(d / "00-papers" / "_status_phase0.json")
        if status is None:
            return "not started"
        return f"0 papers (status: {status})"
    return f"{len(papers)} papers"


def check_phase1(d: Path) -> str:
    sources_dir = d / "01-sources"
    if not sources_dir.exists():
        return "not started"
    subdirs = [p for p in sources_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return "no source dirs"
    arxiv = sum(1 for p in subdirs if (p / "source_type").exists()
                and (p / "source_type").read_text().strip() == "arxiv")
    pdf   = sum(1 for p in subdirs if (p / "source_type").exists()
                and (p / "source_type").read_text().strip() == "pdf")
    return f"{len(subdirs)} papers ({arxiv} arxiv, {pdf} pdf)"


def check_phase2(d: Path) -> str:
    chunks = load_jsonl(d / "02-chunks" / "chunks.jsonl")
    return f"{len(chunks)} chunks" if chunks else "not started"


def check_phase3(d: Path) -> str:
    emb = d / "03-embeddings" / "embeddings.npy"
    meta = load_json(d / "03-embeddings" / "metadata.json")
    if not emb.exists():
        return "not started"
    try:
        import numpy as np
        arr = np.load(str(emb))
        return f"shape={arr.shape}"
    except Exception:
        return f"exists (numpy unavailable)"


def check_phase4(d: Path) -> str:
    summary = load_json(d / "04-clusters" / "summary.json")
    if summary is None:
        return "not started"
    n_cls = summary.get("n_classes", "?")
    n_br  = summary.get("n_bridge_chunks", 0)
    return f"{n_cls} classes, {n_br} bridge chunks"


def check_phase5(d: Path) -> str:
    kw_dir = d / "05-keywords"
    if not kw_dir.exists():
        return "not started"
    status = load_json(kw_dir / "_status_phase5.json")
    if status:
        done  = len(status.get("done", []))
        total = status.get("total", done)
        fail  = len(status.get("failed", []))
        s = fmt_bar(done, total)
        if fail:
            s += f" ({fail} failed)"
        return s
    files = list(kw_dir.glob("chunk_*.json"))
    return f"{len(files)} chunk files (no status)"


def check_phase6(d: Path) -> str:
    col_dir = d / "06-collated"
    if not col_dir.exists():
        return "not started"
    files = list(col_dir.glob("class_*.json"))
    return f"{len(files)} class files"


def check_phase7(d: Path) -> str:
    reps_dir = d / "07-representatives"
    if not reps_dir.exists():
        return "not started"
    manifest = load_json(reps_dir / "cross_manifest.json")
    n_reps = len(list(reps_dir.glob("class_*.json")))
    n_pairs = len(manifest) if manifest else 0
    return f"{n_reps} reps, {n_pairs} cross-pairs"


def check_phase8(d: Path) -> str:
    scores_dir = d / "08-scores"
    if not scores_dir.exists():
        return "not started"
    status = load_json(scores_dir / "_status_phase8.json")
    pairs  = load_jsonl(scores_dir / "pairs.jsonl")
    if status:
        done  = len(status.get("done", []))
        total = status.get("total", done)
        return f"{fmt_bar(done, total)}"
    return f"{len(pairs)} pairs (no status)"


PHASE_CHECKS = [
    ("Phase 0: Papers",         check_phase0),
    ("Phase 1: Sources",        check_phase1),
    ("Phase 2: Chunks",         check_phase2),
    ("Phase 3: Embeddings",     check_phase3),
    ("Phase 4: Clusters",       check_phase4),
    ("Phase 5: Keywords",       check_phase5),
    ("Phase 6: Collated",       check_phase6),
    ("Phase 7: Representatives",check_phase7),
    ("Phase 8: Scores",         check_phase8),
]


# ---------------------------------------------------------------------------
# Detail views
# ---------------------------------------------------------------------------

def show_classes(d: Path) -> None:
    summary = load_json(d / "04-clusters" / "summary.json")
    if not summary:
        print("  No cluster summary found.")
        return

    print(f"\n  Equivalence classes ({summary.get('n_classes', '?')} total):\n")
    print(f"  {'Class':<8} {'Label':<40} {'Chunks':>6}  Top keyphrases")
    print(f"  {'─'*8} {'─'*40} {'─'*6}  {'─'*50}")

    reps_dir = d / "07-representatives"
    for cls in summary.get("classes_summary", []):
        cid     = cls["class_id"]
        label   = cls["label"][:40]
        n_ch    = cls["n_chunks"]

        phrases = ""
        rep_path = reps_dir / f"class_{cid}.json"
        if rep_path.exists():
            rep = load_json(rep_path)
            if rep:
                kws = [k["phrase"] for k in rep.get("top_keyphrases", [])[:4]]
                phrases = ", ".join(kws)

        print(f"  {cid:<8} {label:<40} {n_ch:>6}  {phrases}")


def show_pairs(d: Path, top: int = 20) -> None:
    pairs = load_jsonl(d / "08-scores" / "pairs.jsonl")
    if not pairs:
        # Fall back to manifest with cosine only
        manifest = load_json(d / "07-representatives" / "cross_manifest.json")
        if manifest:
            print(f"\n  No scored pairs yet. Top {top} from cross-manifest (cosine only):\n")
            print(f"  {'Class A':<8} {'Class B':<8} {'Priority':<10} {'Weight':>6}  Label A / Label B")
            print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*6}  {'─'*60}")
            for p in manifest[:top]:
                la = p.get("label_a","")[:25]
                lb = p.get("label_b","")[:25]
                print(f"  {p['class_a']:<8} {p['class_b']:<8} {p.get('priority','?'):<10} {p.get('weight',0):>6.3f}  {la} / {lb}")
        else:
            print("  No pairs or manifest found.")
        return

    sorted_pairs = sorted(pairs, key=lambda p: -p.get("cosine_similarity", 0))
    print(f"\n  Top {min(top, len(sorted_pairs))} scored pairs (of {len(pairs)} total):\n")
    print(f"  {'Class A':<8} {'Class B':<8} {'cos':>6}  {'NLI':<15} Label A / Label B")
    print(f"  {'─'*8} {'─'*8} {'─'*6}  {'─'*15} {'─'*60}")
    for p in sorted_pairs[:top]:
        nli    = p.get("nli") or {}
        top_nli = nli.get("top", "n/a") if isinstance(nli, dict) else "n/a"
        cos    = p.get("cosine_similarity", 0)
        la     = p.get("label_a", "")[:25]
        lb     = p.get("label_b", "")[:25]
        print(f"  {p['class_a']:<8} {p['class_b']:<8} {cos:>6.3f}  {top_nli:<15} {la} / {lb}")

    # Contradiction summary
    contradictions = [p for p in pairs
                      if isinstance(p.get("nli"), dict) and p["nli"].get("top") == "contradiction"]
    if contradictions:
        print(f"\n  Contradictions detected ({len(contradictions)}):")
        for p in contradictions:
            print(f"    {p['class_a']} × {p['class_b']}  cos={p.get('cosine_similarity',0):.3f}")
            print(f"      {p.get('label_a','')} vs {p.get('label_b','')}")
            ra = p.get("rep_a_title","")
            rb = p.get("rep_b_title","")
            if ra:
                print(f"      Rep A: {ra}")
            if rb:
                print(f"      Rep B: {rb}")


def show_class_detail(d: Path, class_id: str) -> None:
    rep_path = d / "07-representatives" / f"class_{class_id}.json"
    col_path = d / "06-collated" / f"class_{class_id}.json"

    rep = load_json(rep_path)
    col = load_json(col_path)

    if not rep:
        print(f"  No representative file for {class_id}")
        return

    print(f"\n  Class {class_id}: {rep.get('label','')}")
    print(f"  Chunks: {rep.get('n_chunks','?')}")
    print(f"  Representative: chunk {rep['representative']['chunk_index']} — {rep['representative']['section_title']}")

    kws = rep.get("top_keyphrases", [])
    if kws:
        print(f"\n  Top keyphrases:")
        for k in kws[:10]:
            print(f"    {k['phrase']:<40}  {k['score']:.4f}")

    claims = rep.get("claim_sentences", [])
    if claims:
        print(f"\n  Claim sentences:")
        for s in claims[:5]:
            print(f"    · {s[:120]}")

    cites = rep.get("citations", [])
    if cites:
        print(f"\n  Citations: {', '.join(cites[:10])}")

    bridge = rep.get("bridge_chunks", [])
    if bridge:
        print(f"\n  Bridge chunks: {bridge}")

    if col:
        top_chunks = col.get("chunks", [])[:5]
        print(f"\n  Top chunks in class (showing first 5 of {col.get('n_chunks','?')}):")
        for c in top_chunks:
            is_rep = " [REP]" if c.get("is_representative") else ""
            is_br  = " [BRIDGE]" if c.get("is_bridge") else ""
            print(f"    chunk {c['chunk_index']:>4}  {c.get('section_title','')[:60]}{is_rep}{is_br}")


def show_papers(d: Path) -> None:
    papers = load_jsonl(d / "00-papers" / "papers.jsonl")
    if not papers:
        print("  No papers found.")
        return
    print(f"\n  Papers ({len(papers)}):\n")
    for i, p in enumerate(papers, 1):
        title = (p.get("title") or "")[:65]
        year  = p.get("year", "?")
        arxiv = p.get("arxiv_id", p.get("paperId","?"))[:15]
        print(f"  {i:>3}. [{year}] {arxiv:<15} {title}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect inference pipeline state for any author directory."
    )
    parser.add_argument("--author-dir", required=True,
                        help="Author pipeline output directory (e.g. jeffrey_leek/)")
    parser.add_argument("--show-classes", action="store_true",
                        help="Show equivalence class summaries")
    parser.add_argument("--show-pairs", action="store_true",
                        help="Show top cross-pair scores")
    parser.add_argument("--show-class", metavar="CLASS_ID",
                        help="Show detail for a specific class (e.g. EC05)")
    parser.add_argument("--show-papers", action="store_true",
                        help="Show discovered papers list")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of pairs to show (default: 20)")
    args = parser.parse_args()

    d = Path(args.author_dir)
    if not d.exists():
        print(f"Error: directory not found: {d}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPipeline status: {d.resolve()}\n")
    print(f"  {'Phase':<28} Status")
    print(f"  {'─'*28} {'─'*50}")
    for label, check_fn in PHASE_CHECKS:
        status = check_fn(d)
        print(f"  {label:<28} {status}")

    if args.show_papers:
        show_papers(d)

    if args.show_classes:
        show_classes(d)

    if args.show_pairs:
        show_pairs(d, top=args.top)

    if args.show_class:
        show_class_detail(d, args.show_class)

    print()


if __name__ == "__main__":
    main()
