# Reproducibility Audit — Four Repos

Date: 2026-04-10
Scope: `~/open-knowledge-graph`, `~/org-open-knowledge-graph`,
`~/meta-knowledge-graph`, `~/the-silence-of-our-times`
Goal: every repo should be runnable from a clean machine with nothing but
git and the appropriate language runtime, following only what is
documented in the repo itself.

This document is the audit findings, the proposed fixes, and the
prioritization. It is written so that someone (a future Patrick, a
collaborator, an LLM agent, or a reviewer) can use it as a checklist
without re-running the audit.

---

## Executive summary

| Repo | State | Severity | What's broken |
|---|---|---|---|
| `the-silence-of-our-times` | nearly perfect | trivial | missing 2-line README — pure markdown writing project, no build needed |
| `open-knowledge-graph` | partial | moderate | analysis dir gitignored without per-subdir docs; only `analysis/paralog_projection/` is currently reproducible from clean |
| `meta-knowledge-graph` | weak | high | no `requirements.txt`, no `.gitignore` at all, silently imports from sibling repo `open-knowledge-graph/packages/core/` without documenting it, benchmark scripts skip missing test suites silently |
| `org-open-knowledge-graph` | broken | **critical** | no dependency declaration, **hardcoded Neo4j credentials in 4+ scripts**, MSK-IMPACT and TCGA data assumed pre-staged with no fetch scripts, no setup instructions despite being the public-facing paper monorepo |

The four repos sit on a spectrum from "trivially reproducible" to
"unreproducible without oral tradition." `org-open-knowledge-graph` is
the worst and most urgent because it is the public face of the work.

---

## Repo 1: `the-silence-of-our-times`

### Findings

- **README:** missing.
- **Dependencies:** none required. Pure markdown writing project.
- **Setup:** none required.
- **Data:** all content is committed markdown. No external assets.
- **Hardcoded paths:** none of consequence.
- **Hidden state:** none.
- **Purpose:** philosophical and analytical essays plus a novel-in-progress
  ("The Finite Thing") exploring silence, communication failure,
  organizational dynamics, technical and human systems. 18 markdown files,
  ~900 KB total.

### Required fixes

1. Add a 4-line `README.md` at the top level explaining what the repo
   contains and that no build is needed.

That is the entire fix. Effort: 1 minute. Risk: zero.

---

## Repo 2: `open-knowledge-graph` (this repo)

### Findings

- **README:** present, comprehensive, well-maintained. Lists all 11 papers
  with links, explains the knowledge graph structure, points to figures.
- **Dependencies:** declared per-subproject in some places (e.g.
  `analysis/paralog_projection/requirements.txt` after this session) but
  not at the repo level. Other analyses inherit Python deps from the user's
  global environment with no documentation.
- **Setup:** the new `analysis/paralog_projection/NEXT_STEPS.md` is the
  reference template. No equivalent exists for the rest of the repo.
- **Data:** `data/` and `analysis/` are partly gitignored. Specific
  exception was added for `analysis/paralog_projection/` in this session.
  Other analysis subdirs remain hidden, with no fetch documentation.
- **Hardcoded paths:** the analysis scripts in `analysis/*.py` (cancer
  work, channel scripts, atlas pipelines) reference data paths via
  `data/channel_gene_map.csv` and similar relative paths, which is fine,
  but they assume large derived files in `gnn/results/` and other
  gitignored directories without documenting how those got there.
- **Hidden state:** none catastrophic, but the dependency between
  `data/channel_gene_map.csv` and the 8-channel taxonomy is implicit.

### Required fixes

1. Add a top-level `requirements.txt` (or `pyproject.toml`) covering the
   shared Python stack used across analyses.
2. Add a top-level `SETUP.md` (or expand the existing README) explaining
   that the canonical reproducible-from-clean entry point is currently
   only `analysis/paralog_projection/`, and that other analyses are
   research-grade and assume preexisting data.
3. **Long-term:** apply the `analysis/paralog_projection/` template
   (NEXT_STEPS, requirements.txt, fetch scripts, gitignore exception,
   data/MANIFEST.md) to other analysis subdirs as they become
   publication-relevant.

Effort: 1-2 hours for items 1 and 2. Item 3 is ongoing.

---

## Repo 3: `meta-knowledge-graph`

### Findings

- **README:** present and informative. Clearly explains the M layer is the
  CQRS write-side counterpart to `open-knowledge-graph` (the read side),
  describes the learning loop architecture, agent infrastructure, and
  domain seeding. Lacks setup or installation instructions entirely.
- **Dependencies:** **no `requirements.txt`, no `pyproject.toml`, no
  `environment.yml`.** Code imports `nltk`, `yaml`, and others with no
  declaration.
- **Setup:** missing.
- **Data:** the benchmark scripts reference a `benchmark/suites/`
  directory containing `gsm8k.json`, `gsm8k_v2.json`, `gsm8k_500.json`
  that does not exist locally. Scripts silently skip missing suites
  (`if not path.exists()`), making the failure invisible.
- **Hardcoded paths:** mostly absent. **One important exception:**
  `packages/bio/python/seed_from_cancer.py` lines 30-32 hardcodes the
  expectation that `~/open-knowledge-graph/gnn/data/cache/` exists as a
  sibling. Will fail silently on a clean machine that does not have that
  sibling repo.
- **Hidden state — critical:** the benchmark scripts import from
  `packages.core.interpret`, `packages.core.validate_parse`, and
  `packages.core.problem_graph`. These modules **do not exist in this
  repo.** They live in `~/open-knowledge-graph/packages/core/`. There is
  no documentation that the sibling repo must be cloned and on
  `PYTHONPATH` for anything to work. A fresh clone of
  `meta-knowledge-graph` alone is non-functional.
- **No `.gitignore` exists.** Anything left in the working tree is at
  risk of being committed accidentally.

### Required fixes (priority order)

1. **Add a `.gitignore`** covering at minimum `__pycache__/`, `*.pyc`,
   `.venv/`, `.DS_Store`, `data/cache/`, and any local-only learning log
   directories. Without this, any future commit could leak local state.
2. **Add a `requirements.txt`** declaring `nltk`, `pyyaml`, and any other
   third-party imports.
3. **Add a `SETUP.md`** that explicitly documents:
   - The sibling-repo dependency on `~/open-knowledge-graph` and how to
     set `PYTHONPATH` to include it
   - How to obtain or generate the `benchmark/suites/*.json` files
   - The role of this repo in the CQRS architecture (already in README,
     but the setup doc should restate the operational implication)
4. **Make benchmark scripts fail loudly when suites are missing** rather
   than silently skipping. Either log a clear error or raise.
5. **Document the seed_from_cancer.py sibling dependency** in that file's
   docstring and in SETUP.md.

Effort: 2-3 hours.

---

## Repo 4: `org-open-knowledge-graph` — most critical

### Findings

- **README:** excellent. Lists all 11 papers, explains the theoretical
  framework, describes the knowledge graph and provenance triples, and
  points to outputs.
- **Setup:** zero. No INSTALL, SETUP, Makefile, tox.ini, or equivalent.
  Despite being the public-facing paper monorepo with 11 papers, a fresh
  reader cannot run anything.
- **Dependencies:** **no `requirements.txt`, `pyproject.toml`, `setup.py`,
  or `environment.yml` at the top level or in any of the 8+ subprojects.**
  Code imports `numpy`, `pandas`, `neo4j`, `lifelines`, `scipy` and
  others without any version pinning or declaration.
- **Data:** `data/channel_gene_map.csv` and `gsm8k_train_200.json` are
  committed; `tcga_benchmark/` and `learning_log/` are present but
  neither documented nor versioned. The `analysis/`, `gnn/results/`,
  `packages/cancer/results/`, `packages/cancer/data/cache/`, and
  `packages/core/language/data/cache/` directories are all gitignored.
  **No fetch scripts exist** to recreate any of these. The MSK-IMPACT
  and TCGA cohort data are assumed to be on disk at paths defined in
  `gnn/config.py` with no documentation of where to obtain them.
- **Hardcoded paths and credentials — security issue:**
  - `scripts/encode_organ_adjacency.py:25`:
    `URI = "bolt://localhost:7687"` and
    `AUTH = ("neo4j", "openknowledgegraph")`
  - `scripts/encode_edge_properties.py:12-13`: same pattern
  - `scripts/normalize_cancer_type.py`: same pattern
  - `scripts/write_w_confidence.py`: same pattern
  - `packages/cancer/config.py:38-42`: `_load_from_graph()` hardcodes the
    same Neo4j URI and password directly into the driver initialization
  - `data/learning_log/` JSON files embed `/Users/patdmccarthy/...`
    paths in metadata
- **Scripts without parameterization:** many. Examples:
  - `analysis/atlas_c_index.py` loads from `gnn.config.MSK_DATASETS["msk_impact_50k"]`
    with no CLI override
  - `analysis/graph_position.py` reads `data/channel_gene_map.csv` hardcoded
  - `analysis/mutual_information.py` assumes `data/tcga_benchmark/`
    exists and is structured a specific way
- **Hidden state:**
  - Neo4j server must be running on `localhost:7687` with user `neo4j`
    and password `openknowledgegraph`. Not documented anywhere.
  - The Neo4j database must contain a specific schema (`:Gene`, `:Channel`,
    `:PPI`, `:COUPLES`) discovered dynamically at runtime by `gnn/config.py`.
    No initialization script.
  - MSK-IMPACT and TCGA data expected at paths defined in
    `gnn.config.MSK_DATASETS`. No documentation of where to get them or
    how to format them.
  - `gnn/results/gene_strand_data.json` must exist for
    `encode_edge_properties.py` to work, but the script that creates
    `gnn/results/` requires that directory to already exist. Circular.

### Required fixes (priority order)

1. **Move Neo4j credentials to environment variables** with fallback to
   nothing (fail loudly). Add a `.env.example` documenting which env
   vars are required: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. The
   current hardcoded password in 4+ scripts is a security issue and a
   reproducibility blocker simultaneously. Fix is one config module
   that all scripts import; existing scripts can be updated in batch
   or left to read the new module via a thin compatibility shim.
2. **Add a top-level `pyproject.toml` or `requirements.txt`** declaring
   `numpy`, `pandas`, `neo4j`, `lifelines`, `scipy`, and any other
   dependencies actually imported by the codebase. Pin major versions
   at minimum.
3. **Add a top-level `SETUP.md`** explaining:
   - Required Python version
   - How to install dependencies
   - How to set Neo4j environment variables
   - How to obtain MSK-IMPACT and TCGA data (license required, link to
     application page)
   - How to initialize the Neo4j schema (script needed, see item 5)
4. **Add a `data/MANIFEST.md`** documenting every file in `data/`, where
   it came from, what version, and how to regenerate or obtain it.
5. **Add a Neo4j initialization script** that creates the required
   schema from a known starting point. Document what graph state must
   exist before any analysis script can run.
6. **Long-term: parameterize the analysis scripts.** This is the largest
   refactor and is not urgent for reproducibility (the scripts work for
   the user). It is urgent for collaboration and review.

Effort: items 1-3 are 1 day. Items 4-5 are 1-2 days. Item 6 is ongoing.

---

## Cross-repo recommendations

- **Establish a reproducibility template.** The
  `analysis/paralog_projection/` directory in `open-knowledge-graph` is
  the working template: README + NOTES + NEXT_STEPS + requirements.txt +
  parameterized fetch scripts + gitignored data with documented
  regeneration. Apply this template wherever a subproject becomes
  publication-relevant.
- **Establish a "if this command does not work as written from a clean
  machine, that is a bug in the doc" rule** as policy. NEXT_STEPS.md in
  the paralog_projection directory states this explicitly. Apply the
  same rule across all four repos.
- **Never commit credentials.** Audit all four repos for any embedded
  passwords, tokens, API keys, or hostnames. Move to environment
  variables with documented `.env.example` files. The
  `org-open-knowledge-graph` Neo4j password is the only one found in
  this audit but a deeper sweep is warranted.
- **Document sibling-repo dependencies.** `meta-knowledge-graph` silently
  depends on `open-knowledge-graph`. `org-open-knowledge-graph` is the
  paper output of `open-knowledge-graph`. None of these dependencies are
  documented in any of the repos. A diagram or table at the top of each
  README clarifying the relationships across repos would prevent the
  next confused new contributor.

---

## What this audit does NOT cover

- Working tree state at the time of audit. Each repo may have uncommitted
  changes that affect behavior.
- The contents of any private data files (MSK-IMPACT, TCGA, DepMap).
  These are licensed data and reproducibility means "documented how to
  obtain," not "redistributed."
- Test coverage. Reproducibility is necessary but not sufficient — a
  reproducible analysis can still be wrong.
- Performance or scaling characteristics on specific hardware. The audit
  is about whether the code runs at all on a clean machine, not whether
  it runs well.

---

## Next steps for this audit

Background agents are dispatched in parallel to apply the additive,
no-risk fixes (new files only, no modification of existing scripts) to
each repo. Each agent commits locally and reports back. No pushes.

The deeper refactors (Neo4j credential migration, script
parameterization) require human review and are NOT being done
autonomously. They are flagged in the per-repo sections above for the
user to schedule.
