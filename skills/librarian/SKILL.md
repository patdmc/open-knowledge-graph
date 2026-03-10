---
name: librarian
version: 1.0.0
type: skill
description: Maintains the knowledge graph — nodes (propositions, theorems, definitions) and edges (derivation, evidence, citation, depends_on) — with provenance on every node and every edge. The single source of truth for what is known and how it is known.
license: MIT

authors:
  - name: Patrick McCarthy
    role: author

tags: [knowledge-graph, provenance, bibliography, citations, sources]

capabilities:
  supports_dry_run: true
  supports_resume: true
  execution_model: deterministic

triggers:
  - condition: "A new proposition, theorem, definition, or reference needs to be added to the knowledge graph"
    intent:
      - "add to the knowledge graph"
      - "record this finding"
      - "add a reference"
      - "update the bibliography"
    priority: high
    reason: "Librarian is the only agent that writes to the knowledge graph; provenance must be maintained"
    confidence: ~

  - condition: "An existing knowledge graph entry needs to be updated with new evidence"
    intent:
      - "update the knowledge graph"
      - "new evidence for"
      - "revise the entry"
    priority: high
    reason: "Updates must preserve provenance chain; Librarian enforces this"
    confidence: ~

  - condition: "Citation or provenance information is needed for a knowledge graph entry"
    intent:
      - "what is the source for"
      - "who established this"
      - "where does this come from"
      - "what is the provenance"
    priority: medium
    reason: "Librarian maintains the provenance index"
    confidence: ~

graph:
  depends_on: []
  recommends_before:
    - skill: researcher
      confidence: ~
      reason: "Researcher discovers what needs to be added; Librarian records it with provenance"
  recommends_after:
    - skill: writer
      confidence: ~
      reason: "Writer updates paper references once Librarian has validated provenance"
  enhanced_by:
    - skills: [researcher, validator]
      confidence: ~
      reason: "Researcher finds new sources; Validator confirms entries meet fidelity threshold"

follows_patterns: []

context:
  mode: continuous
  boundaries:
    repo_change: true
    time_gap_seconds: 3600
    domain_change: true
    explicit_keywords: ["new task", "switch to"]
  preserve:
    - auto_memory: true
    - skill_references: true
    - user_last_n_messages: 5
  clear:
    - intermediate_results: true
    - cached_searches: true

success_criteria:
  - check: "Every node in the knowledge graph has a complete provenance triple: (attribution, evidence, derivation)"
    required: true
  - check: "Every edge in the knowledge graph has a provenance triple: (who established the relation, what evidence, what derivation)"
    required: true
  - check: "Every cited paper in references.bib has a corresponding node in the knowledge graph"
    required: false
  - check: "No orphaned edges (edges referencing nodes that do not exist)"
    required: true

evidence_base:
  discovery:
    commit_message: ["add reference", "update knowledge graph", "add citation", "provenance"]
    file_patterns: ["knowledge-graph/**/*.yaml", "references.bib"]
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

integrations: []

allowed_tools: [Read, Write, Edit, Grep, Glob]
---

# Librarian Agent

The librarian maintains the knowledge graph. Every node and every edge carries a provenance triple. No entry without provenance. No edge without evidence.

## Knowledge Graph Schema

### Node Schema

Every node in `knowledge-graph/` is a YAML file:

```yaml
id: unique-node-id
type: definition | theorem | corollary | lemma | proposition | reference | open-question
name: "Human-readable name"
statement: "The formal statement or content"

provenance:
  attribution:
    author: "Name"
    source: "Document or paper title"
    date: "YYYY-MM-DD or publication year"
  evidence:
    type: formal-derivation | empirical | axiomatic | cited
    description: "How this was established"
    references: [node-id-1, node-id-2]
  derivation:
    from: [node-id-1, node-id-2]
    method: "How derived from the above"
```

### Edge Schema

Every edge between nodes is also a YAML file (or inline within a node's `edges` list):

```yaml
from: node-id
to: node-id
relation: depends_on | derives_from | cited_by | evidences | contradicts | extends
weight: ~  # runtime-derived, not hand-authored

provenance:
  attribution:
    author: "Name"
    source: "Where this relation was established"
    date: "YYYY-MM-DD"
  evidence:
    type: formal-derivation | empirical | cited
    description: "Why this relation holds"
    references: [node-id-1]
  derivation:
    method: "How the relation was established"
```

## Provenance Invariant

From Theorem 10 (Collective Gradient Dominance): knowledge transfer without provenance cannot be reliably weighted in $K_{collective}$. A fact with no provenance is indistinguishable from noise.

Every write to the knowledge graph must include:
1. **Attribution**: who established this (author, paper, date)
2. **Evidence**: how it was verified (formal derivation, empirical observation, citation)
3. **Derivation**: what it was derived from (which prior nodes)

This applies to edges as well as nodes. An edge that says "A depends_on B" without evidence of that dependency is an unverified claim.
