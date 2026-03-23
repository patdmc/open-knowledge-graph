#!/usr/bin/env python3
"""
chunk_tex.py — Split LaTeX source files into inference-ready chunks.

Chunks are split on LaTeX structural boundaries (section/subsection/etc.),
then further split by paragraph if a section exceeds --max-chars.
Overlapping context is prepended from the previous chunk tail.

Each chunk is emitted as a JSONL record with full provenance metadata.

Usage:
    python scripts/chunk_tex.py <file_or_dir> [options]

Options:
    --out           Output JSONL file (default: stdout)
    --max-chars     Max characters per chunk (default: 6000)
    --overlap       Overlap chars carried from previous chunk (default: 300)
    --strip         Strip LaTeX commands; emit plain text only
    --include-preamble
                    Include the LaTeX preamble as chunk 0 (skipped by default)
    --min-chars     Skip chunks smaller than N chars (default: 100)

Examples:
    # Chunk a single file to stdout
    python scripts/chunk_tex.py UNCERTAINTY_BOUNDING_FORMAL_THEORY.tex

    # Chunk all .tex files in repo, write to chunks.jsonl
    python scripts/chunk_tex.py . --out chunks.jsonl

    # Smaller chunks, no LaTeX markup
    python scripts/chunk_tex.py UNCERTAINTY_BOUNDING_FORMAL_THEORY.tex \\
        --max-chars 3000 --strip --out chunks.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# LaTeX structure detection
# ---------------------------------------------------------------------------

# Ordered from highest to lowest level
SECTION_PATTERNS = [
    (0, re.compile(r"^\\section\*?\{(.+?)\}", re.MULTILINE)),
    (1, re.compile(r"^\\subsection\*?\{(.+?)\}", re.MULTILINE)),
    (2, re.compile(r"^\\subsubsection\*?\{(.+?)\}", re.MULTILINE)),
    (3, re.compile(r"^\\paragraph\*?\{(.+?)\}", re.MULTILINE)),
    (4, re.compile(r"^\\subparagraph\*?\{(.+?)\}", re.MULTILINE)),
]

# Match any section-level heading
ANY_SECTION_RE = re.compile(
    r"^\\(?:sub)*section\*?\{|^\\paragraph\*?\{|^\\subparagraph\*?\{",
    re.MULTILINE,
)


def detect_heading(line: str) -> tuple[int, str] | None:
    """Return (level, title) if line is a LaTeX section heading, else None."""
    for level, pat in SECTION_PATTERNS:
        m = pat.match(line.strip())
        if m:
            return level, m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# LaTeX stripping
# ---------------------------------------------------------------------------

# Commands whose content we want to keep (just drop the command wrapper)
_KEEP_ARG = re.compile(
    r"\\(?:text(?:bf|it|rm|tt|sc|sf|up|normal)?|emph|mbox|hbox|"
    r"label|ref|eqref|cite|footnote|caption|item)\{([^}]*)\}"
)
# Math environments to keep as-is (optionally collapse)
_MATH_INLINE = re.compile(r"\$[^$]+\$")
_MATH_DISPLAY = re.compile(r"\$\$[\s\S]+?\$\$")
_ENV_BEGIN = re.compile(r"\\begin\{[^}]+\}")
_ENV_END = re.compile(r"\\end\{[^}]+\}")
_COMMAND = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*")
_COMMENT = re.compile(r"%.*$", re.MULTILINE)


def strip_latex(text: str) -> str:
    """Remove LaTeX markup, keeping readable content."""
    # Remove comments
    text = _COMMENT.sub("", text)
    # Unwrap common text commands
    text = _KEEP_ARG.sub(r"\1", text)
    # Collapse display math to placeholder
    text = _MATH_DISPLAY.sub("[MATH]", text)
    # Collapse inline math to placeholder
    text = _MATH_INLINE.sub("[math]", text)
    # Remove begin/end environment markers
    text = _ENV_BEGIN.sub("", text)
    text = _ENV_END.sub("", text)
    # Remove remaining LaTeX commands
    text = _COMMAND.sub("", text)
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_by_sections(text: str) -> list[tuple[list[str], str]]:
    """
    Split document text into (heading_path, body) pairs.
    heading_path is a list of section titles from outermost to current.
    The preamble (before first heading) gets an empty path.
    """
    # Find all heading positions
    splits = []  # (pos, level, title)
    for level, pat in SECTION_PATTERNS:
        for m in pat.finditer(text):
            splits.append((m.start(), level, m.group(1).strip()))

    if not splits:
        return [([], text)]

    splits.sort(key=lambda x: x[0])

    # Preamble
    sections = []
    preamble = text[: splits[0][0]]
    if preamble.strip():
        sections.append(([], preamble))

    # Track heading hierarchy
    path: list[tuple[int, str]] = []  # [(level, title), ...]

    for i, (pos, level, title) in enumerate(splits):
        end = splits[i + 1][0] if i + 1 < len(splits) else len(text)
        body = text[pos:end]

        # Update path: pop deeper or equal levels, then push current
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, title))

        heading_path = [t for _, t in path]
        sections.append((heading_path, body))

    return sections


def split_by_paragraphs(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split text into chunks of at most max_chars, breaking at paragraph
    boundaries (\n\n). Prepends overlap chars from the previous chunk.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = ""
    prev_tail = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = (current + "\n\n" + para).strip() if current else para

        if len(candidate) <= max_chars:
            current = candidate
        else:
            # Flush current chunk
            if current:
                chunk_text = (prev_tail + "\n\n" + current).strip() if prev_tail else current
                chunks.append(chunk_text)
                prev_tail = current[-overlap:] if overlap else ""
                current = para
            else:
                # Single paragraph exceeds max_chars — hard split
                while len(para) > max_chars:
                    chunk_text = (prev_tail + "\n\n" + para[:max_chars]).strip() if prev_tail else para[:max_chars]
                    chunks.append(chunk_text)
                    prev_tail = para[:max_chars][-overlap:] if overlap else ""
                    para = para[max_chars - overlap:] if overlap else para[max_chars:]
                current = para

    if current:
        chunk_text = (prev_tail + "\n\n" + current).strip() if prev_tail else current
        chunks.append(chunk_text)

    return chunks


# ---------------------------------------------------------------------------
# Main chunking logic
# ---------------------------------------------------------------------------

def chunk_file(
    path: Path,
    max_chars: int,
    overlap: int,
    strip: bool,
    include_preamble: bool,
    min_chars: int,
) -> list[dict]:
    """Return list of chunk dicts for a single .tex file."""
    text = path.read_text(encoding="utf-8", errors="replace")

    # Isolate document body (between \begin{document} and \end{document})
    body_m = re.search(r"\\begin\{document\}([\s\S]+?)\\end\{document\}", text)
    if body_m:
        preamble_text = text[: body_m.start()]
        body = body_m.group(1)
    else:
        preamble_text = ""
        body = text

    sections = split_by_sections(body)

    chunks = []
    chunk_index = 0

    # Optionally emit preamble
    if include_preamble and preamble_text.strip():
        content = strip_latex(preamble_text) if strip else preamble_text.strip()
        if len(content) >= min_chars:
            chunks.append(_make_chunk(path, chunk_index, ["[preamble]"], content))
            chunk_index += 1

    for heading_path, section_body in sections:
        is_preamble_section = not heading_path
        if is_preamble_section and not include_preamble:
            continue

        content = strip_latex(section_body) if strip else section_body.strip()
        if not content:
            continue

        if len(content) <= max_chars:
            if len(content) >= min_chars:
                chunks.append(_make_chunk(path, chunk_index, heading_path, content))
                chunk_index += 1
        else:
            sub_chunks = split_by_paragraphs(content, max_chars, overlap)
            for sub_i, sub in enumerate(sub_chunks):
                if len(sub) >= min_chars:
                    label = heading_path + [f"[part {sub_i + 1}/{len(sub_chunks)}]"] if len(sub_chunks) > 1 else heading_path
                    chunks.append(_make_chunk(path, chunk_index, label, sub))
                    chunk_index += 1

    return chunks


def _make_chunk(path: Path, index: int, heading_path: list[str], content: str) -> dict:
    title = " > ".join(heading_path) if heading_path else "[document body]"
    return {
        "source": path.name,
        "source_path": str(path),
        "chunk_index": index,
        "section_path": heading_path,
        "section_title": title,
        "char_count": len(content),
        "token_estimate": len(content) // 4,  # rough 1 token ≈ 4 chars
        "text": content,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk LaTeX files into inference-ready JSONL."
    )
    parser.add_argument("input", help=".tex file or directory to scan")
    parser.add_argument("--out", default=None, help="Output JSONL file (default: stdout)")
    parser.add_argument("--max-chars", type=int, default=6000,
                        help="Max characters per chunk (default: 6000)")
    parser.add_argument("--overlap", type=int, default=300,
                        help="Overlap chars from previous chunk (default: 300)")
    parser.add_argument("--strip", action="store_true",
                        help="Strip LaTeX commands, emit plain text")
    parser.add_argument("--include-preamble", action="store_true",
                        help="Include document preamble as chunk 0")
    parser.add_argument("--min-chars", type=int, default=100,
                        help="Skip chunks smaller than N chars (default: 100)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        tex_files = sorted(input_path.glob("**/*.tex"))
        # Exclude .arxiv-cache
        tex_files = [f for f in tex_files if ".arxiv-cache" not in f.parts]
    elif input_path.is_file():
        tex_files = [input_path]
    else:
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    if not tex_files:
        print("No .tex files found.", file=sys.stderr)
        sys.exit(0)

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout

    total_chunks = 0

    try:
        for tex_path in tex_files:
            print(f"Chunking {tex_path.name} ...", file=sys.stderr)
            chunks = chunk_file(
                tex_path,
                max_chars=args.max_chars,
                overlap=args.overlap,
                strip=args.strip,
                include_preamble=args.include_preamble,
                min_chars=args.min_chars,
            )
            for chunk in chunks:
                out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            print(f"  {len(chunks)} chunks", file=sys.stderr)
            total_chunks += len(chunks)
    finally:
        if args.out:
            out_fh.close()

    print(f"\nTotal: {total_chunks} chunks from {len(tex_files)} file(s)", file=sys.stderr)
    if args.out:
        print(f"Written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
