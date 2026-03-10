---
name: writer
version: 1.0.0
type: skill
description: Produces well-formed Markdown documents, manages PDF export, and diagnoses rendering issues. The only agent with write access to .md and .pdf files in this repository.
license: MIT

authors:
  - name: Patrick McCarthy
    role: author

tags: [markdown, pdf, writing, formatting, latex]

capabilities:
  supports_dry_run: false
  supports_resume: true
  execution_model: hybrid

triggers:
  - condition: "A .md file in the repository is malformed or fails to render on GitHub"
    intent:
      - "fix the markdown"
      - "the github rendering is broken"
      - "malformed markdown"
      - "sections not rendering"
    priority: high
    reason: "Only this agent has the tools to read, edit, and validate markdown structure"
    confidence: ~

  - condition: "A .md file needs to be written or substantially updated"
    intent:
      - "write a new section"
      - "update the paper"
      - "add content to the document"
    priority: high
    reason: "Ensures all writes are well-formed with correct GitHub markdown syntax"
    confidence: ~

  - condition: "A PDF export is needed from a markdown source"
    intent:
      - "export to pdf"
      - "build the pdf"
      - "generate pdf"
      - "pdfify"
    priority: medium
    reason: "Manages pandoc invocation with correct flags for xelatex and bibliography"
    confidence: ~

graph:
  depends_on: []
  recommends_before:
    - skill: librarian
      confidence: ~
      reason: "Librarian provides citation provenance needed for accurate reference sections"
  recommends_after:
    - skill: validator
      confidence: ~
      reason: "Validator checks the produced document for structural and content correctness"
  enhanced_by: []

follows_patterns: []

context:
  mode: isolated
  boundaries:
    repo_change: true
    time_gap_seconds: 1800
    domain_change: true
    explicit_keywords: ["new task", "switch to", "start fresh"]
  preserve:
    - auto_memory: true
    - skill_references: true
    - user_last_n_messages: 3
  clear:
    - previous_task_context: true
    - intermediate_results: true
    - cached_searches: true

success_criteria:
  - check: "All $$ display math blocks have blank lines before and after them"
    required: true
  - check: "No 4-space indented lines follow blank lines outside of fenced code blocks (would render as code blocks on GitHub)"
    required: true
  - check: "All ## headings have a blank line above them"
    required: true
  - check: "PDF builds with xelatex with zero errors"
    required: true
  - check: "Pandoc citeproc resolves all citations against references.bib"
    required: true

evidence_base:
  discovery:
    commit_message: ["fix markdown", "rebuild pdf", "fix rendering", "pdfify"]
    file_patterns: ["*.md", "*.pdf", "*.bib"]
    diff_patterns: []
    label_patterns: []
    min_match_score: 0.7
  sample:
    strategy: hierarchical-ranked
    as_of: ~
    total: ~
    training: ~
    validation: ~
    ranked_breakdown:
      pattern_setters: ~
      high_quality: ~
      breadth: ~
      depth: ~
      low_quality: ~
  validation:
    last_run: ~
    run_by: ~
    samples_evaluated: ~
  patterns: []
  references: []

integrations:
  - name: pandoc
    type: cli
    required: true
    version: ">=3.0"
    check: "pandoc --version"
  - name: xelatex
    type: cli
    required: true
    version: ""
    check: "xelatex --version"

# Markdown quality rules this agent enforces:
#
# 1. Display math ($$ blocks): must have blank line before AND after
# 2. Indented code blocks: 4-space indent after blank line triggers GitHub code block rendering —
#    avoid this for prose continuation in lists; use flush paragraphs or 3-space indent instead
# 3. Headings: must have blank line above (after any preceding block)
# 4. Section separators (---): must have blank line before AND after
# 5. Citations: use [AuthorYear] format consistent with references.bib keys
#
# PDF build command (canonical):
#   pandoc <file>.md \
#     --from markdown-yaml_metadata_block \
#     --bibliography=references.bib \
#     --citeproc \
#     --pdf-engine=xelatex \
#     -V geometry:margin=1in \
#     -V fontsize=11pt \
#     -o <file>.pdf

allowed_tools: [Read, Write, Edit, Grep, Glob, Bash]
---

# Writer Agent

The writer agent is the sole agent responsible for file I/O on `.md` and `.pdf` files. All other agents produce content proposals; the writer validates their structural correctness and commits changes to disk.

## Markdown Well-Formedness Rules

### Display Math

Every `$$` block must be surrounded by blank lines:

```
<blank line>
$$E = mc^2$$
<blank line>
```

Never:
```
Some text:
$$E = mc^2$$
More text
```

### List Continuation Indentation

For numbered list items, continuation paragraphs must be indented to match the content start column:
- `1. ` → 3-space continuation
- `10. ` → 4-space continuation (WARNING: 4 spaces after a blank line can trigger code block parsing on GitHub — use flush paragraphs for sub-sections of long list items)

Safe approach for long list items: place sub-sections as flush paragraphs immediately after the list ends, not indented inside it.

### Headings

Every ATX heading (`##`, `###`) must have a blank line above it. After a `---` thematic break, add a blank line before the heading:

```
---

## Section Title
```

Never:
```
---
## Section Title
```

## PDF Export

Canonical build command:

```bash
pandoc <file>.md \
  --from markdown-yaml_metadata_block \
  --bibliography=references.bib \
  --citeproc \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -o <file>.pdf
```

Use `xelatex` (not `pdflatex`) — better Unicode handling and avoids overfull hbox issues with complex math.

## Diagnostic Protocol

When a GitHub markdown render is reported malformed:

1. Check for `$$` blocks missing blank lines: `grep -n "^\$\$" file.md`
2. Check for 4-space indent after blank line: Python scan for `line.startswith('    ')` after blank
3. Check for headings without blank line above: `python3 -m pymarkdown --disable-rules MD013 scan file.md`
4. Check for unclosed fenced code blocks: count opening vs closing ` ``` `
5. Rebuild PDF with xelatex to confirm structural integrity
