#!/usr/bin/env python3
"""
yaml_to_tex.py — Convert all knowledge-graph YAML nodes to LaTeX files.

Output structure mirrors knowledge-graph/nodes/ under tex-archive/:
  tex-archive/
    preamble.tex           shared preamble (not standalone)
    catalog.tex            master compilable document
    definitions/D01-*.tex
    theorems/T01-*.tex
    references/REF-*.tex
    authors/AUTHOR-*.tex
    emergent/EM*.tex
    novel/NV*.tex
    equivalency/EC*.tex
    overlap/OV*.tex
    open-questions/OQ*.tex
"""

import os
import sys
import yaml
import re
import time
import multiprocessing
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
NODES_DIR = REPO / "knowledge-graph" / "nodes"
OUT_DIR = REPO / "tex-archive"

# Node type → subfolder name (in case the YAML type string differs from dir)
TYPE_TO_DIR = {
    "definition": "definitions",
    "theorem": "theorems",
    "lemma": "theorems",
    "reference": "references",
    "author": "authors",
    "emergent": "emergent",
    "novel": "novel",
    "equivalency-class": "equivalency",
    "equivalency_class": "equivalency",
    "overlap": "overlap",
    "open-question": "open-questions",
    "open_question": "open-questions",
}

# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------
_SPECIAL = str.maketrans({
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
})

def esc(s: str) -> str:
    """Escape LaTeX special characters in a plain string."""
    if not s:
        return ""
    return str(s).translate(_SPECIAL)


def esc_block(s: str) -> str:
    """Escape a multi-line block, preserving newlines."""
    if not s:
        return ""
    return esc(str(s).rstrip())


# ---------------------------------------------------------------------------
# Edge rendering
# ---------------------------------------------------------------------------
def render_edges(edges) -> str:
    if not edges:
        return ""
    lines = [r"\subsection*{Edges}", r"\begin{description}"]
    for e in edges:
        if not isinstance(e, dict):
            continue
        target = esc(e.get("to", "?"))
        relation = esc(e.get("relation", ""))
        desc = ""
        ev = e.get("provenance", {}).get("evidence", {})
        if isinstance(ev, dict):
            desc = esc(ev.get("description", ""))
        lines.append(
            rf"  \item[\texttt{{{target}}}] \textit{{{relation}}}"
            + (rf" — {desc}" if desc else "")
        )
    lines.append(r"\end{description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provenance rendering
# ---------------------------------------------------------------------------
def render_provenance(prov: dict) -> str:
    if not prov:
        return ""
    lines = [r"\subsection*{Provenance}"]
    attr = prov.get("attribution", {})
    if isinstance(attr, dict):
        for k, v in attr.items():
            if v:
                lines.append(rf"\textbf{{{esc(k)}:}} {esc(str(v))} \\")
    ev = prov.get("evidence", {})
    if isinstance(ev, dict):
        ev_type = ev.get("type", "")
        ev_desc = ev.get("description", "")
        if ev_type:
            lines.append(rf"\textbf{{Evidence type:}} \textit{{{esc(ev_type)}}} \\")
        if ev_desc:
            lines.append(rf"\textbf{{Evidence:}} {esc_block(ev_desc)} \\")
    deriv = prov.get("derivation", {})
    if isinstance(deriv, dict):
        from_nodes = deriv.get("from", [])
        method = deriv.get("method", "")
        if from_nodes:
            refs = ", ".join(rf"\texttt{{{esc(n)}}}" for n in from_nodes)
            lines.append(rf"\textbf{{Derived from:}} {refs} \\")
        if method:
            lines.append(rf"\textbf{{Method:}} {esc_block(method)} \\")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-type renderers
# ---------------------------------------------------------------------------

def render_definition(data: dict) -> str:
    num = data.get("number", "")
    name = esc(data.get("name", data.get("id", "")))
    stmt = esc_block(data.get("statement", ""))
    label = esc(data.get("id", ""))
    parts = [
        rf"\section*{{Definition {num} — {name}}}",
        rf"\label{{node:{label}}}",
        r"\begin{quote}",
        stmt,
        r"\end{quote}",
        render_provenance(data.get("provenance", {})),
        render_edges(data.get("edges", [])),
    ]
    return "\n\n".join(p for p in parts if p.strip())


def render_theorem(data: dict) -> str:
    num = data.get("number", "")
    name = esc(data.get("name", data.get("id", "")))
    stmt = esc_block(data.get("statement", ""))
    label = esc(data.get("id", ""))
    parts = [
        rf"\section*{{Theorem {num} — {name}}}",
        rf"\label{{node:{label}}}",
        r"\begin{quote}",
        stmt,
        r"\end{quote}",
        render_provenance(data.get("provenance", {})),
        render_edges(data.get("edges", [])),
    ]
    return "\n\n".join(p for p in parts if p.strip())


def render_reference(data: dict) -> str:
    name = esc(data.get("name", data.get("id", "")))
    fidelity = esc(data.get("fidelity", ""))
    stmt = esc_block(data.get("statement", ""))
    label = esc(data.get("id", ""))
    parts = [
        rf"\section*{{{name}}}",
        rf"\label{{node:{label}}}",
    ]
    if fidelity:
        parts.append(rf"\textit{{Fidelity: {fidelity}}} \\[0.5em]")
    if stmt:
        parts += [r"\begin{quote}", stmt, r"\end{quote}"]
    parts += [
        render_provenance(data.get("provenance", {})),
        render_edges(data.get("edges", [])),
    ]
    return "\n\n".join(p for p in parts if p.strip())


def render_author(data: dict) -> str:
    name = esc(data.get("name", data.get("id", "")))
    stmt = esc_block(data.get("statement", ""))
    label = esc(data.get("id", ""))
    parts = [
        rf"\section*{{{name}}}",
        rf"\label{{node:{label}}}",
    ]
    if stmt:
        parts += [r"\begin{quote}", stmt, r"\end{quote}"]
    parts += [
        render_provenance(data.get("provenance", {})),
        render_edges(data.get("edges", [])),
    ]
    return "\n\n".join(p for p in parts if p.strip())


def render_emergent(data: dict) -> str:
    name = esc(data.get("name", data.get("id", "")))
    stmt = esc_block(data.get("statement", ""))
    label = esc(data.get("id", ""))
    parts = [
        rf"\section*{{Emergent: {name}}}",
        rf"\label{{node:{label}}}",
        r"\begin{quote}",
        stmt,
        r"\end{quote}",
        render_provenance(data.get("provenance", {})),
        render_edges(data.get("edges", [])),
    ]
    return "\n\n".join(p for p in parts if p.strip())


def render_novel(data: dict) -> str:
    name = esc(data.get("name", data.get("id", "")))
    stmt = esc_block(data.get("statement", ""))
    label = esc(data.get("id", ""))
    parts = [
        rf"\section*{{Novel Result: {name}}}",
        rf"\label{{node:{label}}}",
        r"\begin{quote}",
        stmt,
        r"\end{quote}",
        render_provenance(data.get("provenance", {})),
    ]
    # Independent derivations
    ind = data.get("independent_derivations", [])
    if ind:
        parts.append(r"\subsection*{Independent Derivations}")
        for d in ind:
            if not isinstance(d, dict):
                continue
            src = esc(d.get("source", ""))
            desc = esc_block(d.get("description", ""))
            method = esc_block(d.get("method", ""))
            parts.append(
                rf"\paragraph{{{src}}} {desc}"
                + (rf"\\ \textit{{Method: {method}}}" if method else "")
            )
    parts.append(render_edges(data.get("edges", [])))
    return "\n\n".join(p for p in parts if p.strip())


def render_equivalency(data: dict) -> str:
    name = esc(data.get("name", data.get("id", "")))
    label = esc(data.get("id", ""))
    underlying = esc_block(data.get("underlying_proposition", ""))
    analysis = esc_block(data.get("equivalence_analysis", ""))
    conf = data.get("confidence_C1p", {})
    parts = [
        rf"\section*{{Equivalency Class: {name}}}",
        rf"\label{{node:{label}}}",
    ]
    if underlying:
        parts += [r"\subsection*{Underlying Proposition}", r"\begin{quote}", underlying, r"\end{quote}"]
    members = data.get("members", [])
    if members:
        parts.append(r"\subsection*{Members}")
        parts.append(r"\begin{description}")
        for m in members:
            if not isinstance(m, dict):
                continue
            mid = esc(m.get("id", ""))
            fw = esc(m.get("framework", ""))
            form = esc_block(m.get("formulation", ""))
            status = esc(m.get("epistemic_status", ""))
            parts.append(
                rf"  \item[\texttt{{{mid}}}] \textbf{{{fw}}}"
                + (rf" \textit{{({status})}}" if status else "")
                + (rf"\\ {form}" if form else "")
            )
        parts.append(r"\end{description}")
    if analysis:
        parts += [r"\subsection*{Equivalence Analysis}", r"\begin{quote}", analysis, r"\end{quote}"]
    if isinstance(conf, dict):
        est = conf.get("estimate", "")
        basis = esc_block(conf.get("basis", ""))
        if est or basis:
            parts.append(rf"\subsection*{{Confidence $C_1(p)$}}")
            if est:
                parts.append(rf"\textbf{{Estimate:}} {esc(str(est))} \\")
            if basis:
                parts.append(rf"\textbf{{Basis:}} {basis} \\")
    parts += [
        render_provenance(data.get("provenance", {})),
        render_edges(data.get("edges", [])),
    ]
    return "\n\n".join(p for p in parts if p.strip())


def render_generic(data: dict) -> str:
    node_type = esc(data.get("type", "node"))
    name = esc(data.get("name", data.get("id", "")))
    stmt = esc_block(data.get("statement", ""))
    label = esc(data.get("id", ""))
    parts = [
        rf"\section*{{{name}}}",
        rf"\label{{node:{label}}}",
        rf"\textit{{Type: {node_type}}} \\[0.5em]",
    ]
    if stmt:
        parts += [r"\begin{quote}", stmt, r"\end{quote}"]
    parts += [
        render_provenance(data.get("provenance", {})),
        render_edges(data.get("edges", [])),
    ]
    return "\n\n".join(p for p in parts if p.strip())


RENDERERS = {
    "definition": render_definition,
    "theorem": render_theorem,
    "lemma": render_theorem,
    "reference": render_reference,
    "author": render_author,
    "emergent": render_emergent,
    "novel": render_novel,
    "equivalency-class": render_equivalency,
    "equivalency_class": render_equivalency,
    "overlap": render_equivalency,
    "open-question": render_generic,
    "open_question": render_generic,
}


# ---------------------------------------------------------------------------
# Single-file conversion (called in worker processes)
# ---------------------------------------------------------------------------

def _sanitize_yaml(raw: str) -> str:
    """Remove control characters that YAML disallows (DEL=0x7F, C0/C1 except TAB/LF/CR)."""
    allowed = {0x09, 0x0A, 0x0D}  # TAB, LF, CR
    return "".join(
        ch for ch in raw
        if ord(ch) >= 0x20 or ord(ch) in allowed
    )


def _fallback_parse(yaml_path: Path, raw: str) -> dict:
    """Minimal regex-based extraction when YAML is malformed."""
    data: dict = {}
    # id
    m = re.search(r'^id:\s*(\S+)', raw, re.MULTILINE)
    if m:
        data["id"] = m.group(1)
    # type
    m = re.search(r'^type:\s*(\S+)', raw, re.MULTILINE)
    if m:
        data["type"] = m.group(1)
    # name — grab everything after 'name: ' up to newline, strip quotes
    m = re.search(r'^name:\s*(.+)', raw, re.MULTILINE)
    if m:
        data["name"] = m.group(1).strip().strip('"').strip("'")
    # statement — grab first line of block
    m = re.search(r'^statement:\s*\|?\s*\n(.*)', raw, re.MULTILINE)
    if m:
        data["statement"] = m.group(1).strip()
    if not data.get("id"):
        data["id"] = yaml_path.stem
    return data


def _render_node(data: dict) -> str:
    node_type = data.get("type", "")
    renderer = RENDERERS.get(node_type, render_generic)
    return renderer(data)


def convert_file(args):
    yaml_path, out_path = args
    try:
        with open(yaml_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        raw = _sanitize_yaml(raw)
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            data = _fallback_parse(yaml_path, raw)

        # Batch file: YAML list → one tex file per node
        if isinstance(data, list):
            out_dir = out_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for item in data:
                if not isinstance(item, dict):
                    continue
                node_id = item.get("id") or ""
                if not node_id:
                    continue
                body = _render_node(item)
                node_out = out_dir / f"{node_id}.tex"
                with open(node_out, "w", encoding="utf-8") as f:
                    f.write(body)
                    f.write("\n")
                written += 1
            return (str(yaml_path), "ok", f"batch:{written}")

        # Single-node file (original format)
        if not isinstance(data, dict):
            return (str(yaml_path), "skip", "not a dict or list")

        body = _render_node(data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(body)
            f.write("\n")
        return (str(yaml_path), "ok", "")
    except Exception as exc:
        return (str(yaml_path), "error", str(exc))


# ---------------------------------------------------------------------------
# Preamble and master catalog
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{xcolor}

% Node cross-reference shorthand
\newcommand{\noderef}[1]{\hyperref[node:#1]{\texttt{#1}}}

\title{Open Knowledge Graph --- Full Node Archive}
\author{Patrick McCarthy}
\date{Generated \today}
"""

CHAPTER_ORDER = [
    ("definitions",    "Definitions"),
    ("theorems",       "Theorems \& Lemmas"),
    ("novel",          "Novel Results"),
    ("emergent",       "Emergent Confirmations"),
    ("equivalency",    "Equivalency Classes"),
    ("overlap",        "Overlaps"),
    ("open-questions", "Open Questions"),
    ("references",     "References"),
    ("authors",        "Authors"),
]


def write_preamble():
    p = OUT_DIR / "preamble.tex"
    p.write_text(PREAMBLE, encoding="utf-8")


def write_catalog(category_files: dict[str, list[Path]]):
    lines = [
        PREAMBLE,
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
    ]
    for subdir, title in CHAPTER_ORDER:
        files = sorted(category_files.get(subdir, []))
        if not files:
            continue
        lines.append(rf"\chapter{{{esc(title)}}}")
        for fp in files:
            # Path relative to catalog.tex (same dir)
            rel = fp.relative_to(OUT_DIR)
            lines.append(rf"\input{{{rel.as_posix()}}}")
            lines.append(r"\clearpage")
    lines += [r"\end{document}", ""]
    (OUT_DIR / "catalog.tex").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all YAML input/output pairs
    jobs = []
    for yaml_path in NODES_DIR.rglob("*.yaml"):
        rel = yaml_path.relative_to(NODES_DIR)
        out_path = OUT_DIR / rel.with_suffix(".tex")
        jobs.append((yaml_path, out_path))

    total = len(jobs)
    print(f"Found {total:,} YAML files. Converting with {multiprocessing.cpu_count()} workers...")

    # Track category → output files (for catalog)
    category_files: dict[str, list[Path]] = {s: [] for s, _ in CHAPTER_ORDER}
    for yaml_path, out_path in jobs:
        subdir = yaml_path.parent.name
        if subdir in category_files:
            category_files[subdir].append(out_path)

    # Run conversion in parallel
    ok = errors = skipped = 0
    chunk = max(1, total // 200)  # progress every ~0.5%
    with multiprocessing.Pool() as pool:
        for i, (path, status, msg) in enumerate(pool.imap_unordered(convert_file, jobs, chunksize=64)):
            if status == "ok":
                ok += 1
            elif status == "error":
                errors += 1
                print(f"  ERROR {path}: {msg}", file=sys.stderr)
            else:
                skipped += 1
            if (i + 1) % chunk == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1:>9,}/{total:,}  ok={ok:,}  err={errors}  "
                      f"{elapsed:.0f}s elapsed  ~{remaining:.0f}s remaining", flush=True)

    write_preamble()
    write_catalog(category_files)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {ok:,} converted, {errors} errors, {skipped} skipped.")
    print(f"Output: {OUT_DIR}")

    # Report output size
    total_bytes = sum(f.stat().st_size for f in OUT_DIR.rglob("*.tex"))
    print(f"Total tex archive size: {total_bytes / 1e6:.1f} MB ({total_bytes:,} bytes)")


if __name__ == "__main__":
    main()
