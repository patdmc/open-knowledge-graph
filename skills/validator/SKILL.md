---
name: validator
version: 1.0.0
type: higher-order
description: Validates the outputs of all other agents — structural correctness of the document, provenance completeness of the knowledge graph, and logical consistency of claims against sources. Leverages researcher and librarian to verify the full paper.
license: MIT

authors:
  - name: Patrick McCarthy
    role: author

tags: [validation, consistency, provenance, review, quality-control]

capabilities:
  supports_dry_run: true
  supports_resume: true
  execution_model: agentic

triggers:
  - condition: "A major document update has been committed and needs validation before publication"
    intent:
      - "validate the paper"
      - "check the document"
      - "full review"
      - "is it ready to publish"
      - "validate before submission"
    priority: critical
    reason: "Validator runs the full provenance and consistency checks before publication"
    confidence: ~

  - condition: "A knowledge graph entry has been added and its provenance needs verification"
    intent:
      - "validate this entry"
      - "check the provenance"
      - "is this supported"
    priority: high
    reason: "Validator confirms entries meet fidelity threshold lambda_min (Theorem 10)"
    confidence: ~

  - condition: "An agent has produced output that needs to be checked before use"
    intent:
      - "check what the researcher found"
      - "validate the librarian entry"
      - "review the writer's output"
    priority: high
    reason: "Validator is the inter-agent quality control layer"
    confidence: ~

graph:
  depends_on:
    - skill: librarian
      required: true
    - skill: researcher
      required: false
  recommends_before: []
  recommends_after:
    - skill: writer
      confidence: ~
      reason: "Writer commits validator-approved changes to the document"
  enhanced_by:
    - skills: [librarian, researcher]
      confidence: ~
      reason: "Validator uses librarian's provenance index and researcher's external source access to verify claims"

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
  - check: "Every theorem in the paper has been traced to its axioms/definitions with no forward references"
    required: true
  - check: "Every empirical claim has a corresponding knowledge graph node with evidence of type empirical or cited"
    required: true
  - check: "Every citation key in the paper exists in references.bib"
    required: true
  - check: "Every knowledge graph node has a complete provenance triple"
    required: true
  - check: "Every knowledge graph edge has a complete provenance triple"
    required: true
  - check: "No circular derivations in the knowledge graph (K is a DAG)"
    required: true
  - check: "Transmission fidelity lambda > lambda_min for every knowledge transfer recorded"
    required: false

evidence_base:
  discovery:
    commit_message: ["validate", "review", "check consistency", "verify"]
    file_patterns: ["knowledge-graph/**/*.yaml", "*.md", "references.bib"]
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

allowed_tools: [Read, Grep, Glob]
---

# Validator Agent

The validator is the quality control layer for all other agents. It cannot write files — it can only read and report. All fixes go back to the appropriate agent (writer fixes structure, librarian fixes provenance, researcher finds missing sources).

## Validation Checks

### Document Structure (runs against writer output)

1. **Math block spacing**: every `$$` block has blank lines before and after
2. **Heading spacing**: every `##` / `###` has a blank line above
3. **Section separator spacing**: every `---` has blank lines before and after
4. **Code block contamination**: no 4-space-indented prose after blank lines in list contexts
5. **Citation resolution**: every `[AuthorYear]` citation key exists in `references.bib`

### Knowledge Graph Provenance (runs against librarian output)

1. **Node completeness**: every node has `provenance.attribution`, `provenance.evidence`, `provenance.derivation`
2. **Edge completeness**: every edge has its own provenance triple
3. **No orphaned edges**: every edge references existing node IDs
4. **DAG property**: no circular derivation chains (K must be a DAG — acyclicity follows from provenance requiring prior nodes to exist before derivation)
5. **Fidelity threshold**: transmission fidelity $\lambda > \lambda_{min}$ for each recorded transfer

### Logical Consistency (runs against full paper + knowledge graph)

1. **Forward reference check**: no theorem proves itself using a theorem stated later
2. **Definition coverage**: every symbol used in a theorem is defined before first use
3. **Empirical claim grounding**: every empirical claim maps to a knowledge graph node with evidence type `empirical` or `cited`
4. **Open question accuracy**: open questions do not claim answers the framework doesn't provide
5. **Adversarial review**: for each major theorem, state the strongest objection and check whether the paper addresses it

## Validation Report Format

```
VALIDATION REPORT
=================
Date: YYYY-MM-DD
Validator version: 1.0.0

PASS/FAIL SUMMARY
- Document structure: PASS | FAIL
- Knowledge graph provenance: PASS | FAIL
- Logical consistency: PASS | FAIL

ISSUES
------
[issue-id] [severity: critical|high|medium|low]
Location: [file:line or node-id]
Description: [what is wrong]
Resolution: [which agent should fix this and how]

RECOMMENDATIONS
---------------
[items that are not failures but would improve publication readiness]
```
