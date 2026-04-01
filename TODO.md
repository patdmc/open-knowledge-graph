# TODO

Items to work on — not now. Candidates for overnight/background runs.

## Monorepo Structure

```
open-knowledge-graph/
├── packages/
│   ├── core/                 # schema, validation, hooks, nouns/verbs, formal vocabulary, language layer
│   ├── math/                 # math graph
│   ├── bio/                  # biology nodes, healthy graph
│   ├── cancer/               # gnn/, cancer graph, scoring, clinical
│   ├── infotheory/           # channel capacity, encoding
│   ├── neuro/
│   ├── endo/
│   ├── atlas/                # meta-graph: cartographer's output
│   └── ...
├── agents/
│   ├── observer/             # text → claims (inbound)
│   ├── researcher/           # validates claims against graph
│   ├── librarian/            # index integrity, source lookup
│   ├── editor/               # consistency across assertions
│   ├── curator/              # gap review → edge commitment (ratchet)
│   ├── cartographer/          # cross-graph edge discovery, structural analogues
│   ├── external_researcher/  # connects to remote sources (PubMed, arxiv, STRING-DB)
│   ├── translator/           # converts external data to okg.core format
│   ├── mentor/               # sharpens queries before LLM — strategy layer
│   └── author/               # graph → text (outbound), consults all others
├── bench/                    # test harness — read-only, never feeds back
│   ├── gpqa/                 # primary benchmark
│   ├── runs/                 # timestamped results per graph snapshot
│   ├── harness.py            # bare LLM vs RAG vs graph runner
│   └── provenance.py         # edge-to-answer attribution tracker
├── publications/
│   ├── biorxiv/
│   ├── arxiv/
│   ├── medium/
│   └── shared/
├── scripts/
└── pyproject.toml
```

## Monorepo Refactoring

- [ ] Define package boundaries: core, math, bio, cancer, infotheory, neuro, endo, atlas
- [ ] Dependency tree: core is the single root. All domain packages depend only on core. New domain = one dependency.
- [ ] Create pyproject.toml with workspace/monorepo config
- [ ] Move nodes/ subdirectories into their respective packages
- [ ] Winnow scripts/ — migrate one-off analysis into graph packages, graduate proto-agents into agents/
- [ ] Move .tex files into publications/ with venue subdirs (biorxiv/, arxiv/, medium/)
- [ ] Move gnn/ into packages/cancer/
- [ ] Shared publication assets: references.bib, figures/, macros into publications/shared/

## Agent Framework

- [ ] Agent definition schema (YAML): name, role, tools, depends_on, prompt_file
- [ ] Thin runtime adapters: Claude Code agents/, generic CLI, API
- [ ] Observer agent: text → structured claims (inbound, currently observe_claims.py)
- [ ] Researcher agent: two modes — defensive (validate claims against graph) and offensive (investigate gaps, find sources in PubMed/arxiv/web). Same agent, different trigger
- [ ] Librarian agent: index validation, edge/node definition, source lookup
- [ ] Editor agent: consistency checking across assertions
- [ ] Curator agent: gap review → edge commitment (the ratchet, currently precipitate.py)
- [ ] Cartographer agent: cross-graph edge discovery, structural analogues between packages. Currently edge_discovery.py map/reduce. Depends on language+math substrate for shared vocabulary
- [ ] Atlas package: edge-only meta-graph (no nodes of its own). Cartographer writes, everyone else reads. Cross-package edges with foreign keys into source packages
- [ ] Tombstone/redirect pattern: when a node moves or splits, old location becomes a 301 pointing to successor(s). Any edge that hits a tombstone follows the redirect and rewrites itself on read. Splits point to multiple successors — researcher resolves which the edge meant. Gives eventual consistency without requiring consistent writes
- [ ] Author agent: graph → accessible text (outbound), orchestrates researcher/editor/librarian
- [ ] External researcher agent: connects to PubMed, arxiv, STRING-DB, GO, UniProt. Fetches raw data. Never writes to graph.
- [ ] Translator agent: converts external data to okg.core format. Normalizes IDs, types, nouns, verbs, provenance. The boundary between outside and inside.
- [ ] Ingest pipeline: external_researcher → translator → curator → human → graph. Nothing from outside enters without translation. Nothing translated enters without human review.
- [ ] Mentor agent: confidence-aware router and query sharpener. Sits between agents and LLM. Three roles: (1) decompose hard questions into testable sub-claims, (2) route based on graph confidence — answer locally, sharpen query, or go external, (3) ask "why" — force LLM to articulate reasoning chain before accepting claims. Each link in the chain becomes an independent validation target. Cheapest possible filter before spending resources.
- [ ] Agent dependency protocol: ANY agent consults mentor when stuck or before querying LLM. Mentor is universal — not just for researcher. Researcher consults cartographer's maps before investigating. Author consults researcher for truth, editor for consistency, librarian for sources. Author never touches graph directly

## Experiments

- [ ] Foundation dependency test: run edge discovery WITH vs WITHOUT language+math packages. Measure cross-domain edges found. Hypothesis: without language/math substrate, domains reinvent vocabulary and cross-domain connections crater
- [ ] Ad-hoc vocabulary drift: build a biology subgraph in isolation (no language graph). Count synonym collisions, undefined predicates, inconsistent naming. Quantify the cost of missing the naming layer
- [ ] Graph efficiency without math: store biology assertions without formal predicate structure. Measure storage redundancy, query ambiguity, validation failures vs the structured version
- [ ] Public benchmark: bare LLM vs LLM+RAG vs LLM+knowledge graph on cross-domain reasoning. Best candidate: GPQA (graduate-level Q&A, requires multi-hop cross-domain synthesis). Measure accuracy AND confidence calibration — graph system can report path confidence per answer, bare LLM cannot. Secondary candidates: ARC Challenge, BioASQ, MMLU cross-domain subset
  - CRITICAL: never train on benchmark, never feed results back. Test is read-only — the ratchet only turns via new sources, not test signal
  - Re-run after each major source addition. Plot accuracy vs graph size (edges). Each gain traceable to specific edge/package addition
  - Track which specific edges caused which answers to flip. Full provenance: "added PMID:X → edge Y → question Z now correct"

## Core Design Decisions (AGREED)

### Package namespace: okg
okg.core (includes language layer + formal vocabulary), okg.math, okg.bio, okg.cancer, okg.infotheory, okg.neuro, okg.endo, okg.atlas

### Node ID scheme: {package}:{type}:{id}
bio:gene:BRCA1, math:theorem:channel_capacity, atlas:bridge:bio-infotheory-0001
Tombstones use same ID, node body is a redirect to successor(s).

### Core nouns (node types every package inherits)
axiom, definition, assertion, conjecture, proof, example, source, context

### Core verbs (edge types every package inherits)
PROVES, SUPPORTS, REFUTES
INSTANTIATES, DEFINES
REQUIRES, ASSUMES
IMPLIES, CONTRADICTS
CITES, DERIVES_FROM, SUPERSEDES, RETRACTED
SPECIALIZES, GENERALIZES
PART_OF, HAS_PART

Domain packages add their own nouns and verbs (bio: gene, channel, pathway + PHOSPHORYLATES, RECRUITS; math: theorem, lemma + BOUNDS, CONVERGES) but compose with core vocabulary.

### Formal vocabulary (okg.core)
Quantifiers: FOR_ALL, THERE_EXISTS, THERE_EXISTS_UNIQUE
Connectives: AND, OR, NOT, IMPLIES, IFF
Set operations: UNION, INTERSECT, SUBSET, ELEMENT_OF, COMPLEMENT, EMPTY_SET
Relations: EQUALS, LESS_THAN, GREATER_THAN, MAPS_TO

Every statement in every package decomposes into these plus domain-specific terms. Core owns the formal grammar and the natural language layer.

### Node fields (minimum schema)
Every node in every package must have:
- **id** — {package}:{type}:{name} format
- **type** — core noun or registered domain noun
- **name** — human-readable
- **statement** — what it says in natural language (always required)
- **formal** — machine-readable version (optional, domain-specific: latex for math, triples for bio, logic for proofs)
- **provenance** — who, when, source, method. No node without attribution.
- **edges** — list of {to, relation, provenance}. Every edge has its own provenance.

### Human-in-the-loop: no automated graph writes
Every proposed change to the knowledge graph requires human review. Agents observe, research, validate, find sources, check consistency — but the final commit is always a person. No automated precipitation. The curator flags, a human confirms. This is a core invariant, not a configuration option.

### Append-only: changes are always adds
Never delete an edge, never modify an edge. Corrections add a new assertion with SUPERSEDES pointing at the old one. Disagreements add CONTRADICTS. Retractions add RETRACTED with provenance of what went wrong. The old edge stays visible, marked, not deleted. Same as journal retractions.

Only true delete: operational fix for data that should never have existed (test data, accidental entry). That's a git revert with commit message, not a graph operation.

### Merge strategy: content-addressed, not file-addressed
Contributions merge as assertion deltas, not file diffs. Observer extracts assertions from any input format, researcher validates against graph, curator commits. Three outcomes: already known (no-op), new edge (precipitate), conflict (store as CONTRADICTS edge between assertions). Contributors don't need to match schema — input format is irrelevant.

Contradictions are first-class edges, not errors. Two sources disagree = CONTRADICTS edge linking the two assertions. Both stay. Resolution may add CONTEXT nodes (both true in different contexts). Unresolved contradictions = knowledge frontier.

Reverse sync: pulling our graph gives canonical assertions. Diff against their graph is a list of missing/conflicting assertions. Same pipeline, reversed.

### Package internal structure: hierarchical with bridges and indexes
Each package is a graph with subgraphs. Same pattern at every level:
- Subdirectories are subgraphs (bio/ddr/, bio/cellcycle/, bio/immune/)
- Each subgraph can nest further (bio/ddr/hr_repair/, bio/ddr/nhej/)
- bridges/ at each level holds cross-subgraph edges within that package
- atlas/ holds cross-package edges (the top-level bridges/)
- INDEX.yaml at each level — lists nodes, edges, and child subgraphs in this directory. You never scan the tree. You read the index, follow the pointer.

Librarian agent maintains the indexes. Node added = index updated. Node moved = tombstone in old index, entry in new index.

Directory structure matches the encoding hierarchy: sub-pathway → channel → cross-channel.

### Migrate hardcoded language interpretation to graph
- [ ] Verb → operation mappings (operations.py) → graph edges: `verb_node --RESOLVES_TO--> operation_node`
- [ ] Unit multipliers (dozen=12, pair=context-dependent) → graph edges with context predicates
- [ ] Disambiguation rules (pair of shoes=1 unit, pair of dice=2) → context-conditioned edges, not Python sets
- [ ] Rate operator equivalences: "per" = "each" = "for every" = "for each" → single equivalence class node
- [ ] Greedy matching: graph traversal naturally picks longest/most specific edge first ("gets rid of" > "gets")
- [ ] The compiler becomes a graph walker: verb resolution, unit conversion, and disambiguation all via edge lookup
- [ ] New world knowledge = new edges, not code changes

### Graph consolidation pass ("sleep")
- [ ] Walk graph, collapse synonyms into equivalence clusters (words → cluster → concept, O(n) not O(n²))
- [ ] Merge redundant paths (if A→B→C and A→C exist, check if they say the same thing)
- [ ] Promote validated working memory edges to language graph (edges that "travel" — solve many problems)
- [ ] Prune working memory edges that didn't travel (single-problem heuristics, not axioms)
- [ ] Detect conflicts between working memory and knowledge graph — flag for teacher/expert review
- [ ] Conflicts mean an edge or node needs to split into finer-grained subclasses (disambiguation)
- [ ] Analogous to human sleep: consolidate free context → working memory → long-term knowledge

### Still needs discussion
- [ ] Schema version strategy
- [ ] How agents reference the core vocabulary (import? config?)

## Publications

- [ ] Papers 5, 6, 7, 8, 9 — target biorxiv + bio journal (eLife? Cancer Research?)
- [ ] Paper 6 (Channel Structure) — share with Dr. Yap, verify all numbers from fresh runs
- [ ] Paper 5 (Cancer) — TMB paradox numbers need re-verification with correct data paths
- [ ] Three-ring test result (non-channel mutations p=0.95) — add to Paper 6
