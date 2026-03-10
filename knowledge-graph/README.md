# Knowledge Graph

This directory contains the knowledge graph derived from the formal theory in `UNCERTAINTY_BOUNDING_FORMAL_THEORY.md`. Every node and every edge has a provenance triple: (attribution, evidence, derivation).

## Structure

```
knowledge-graph/
  nodes/
    definitions/     — formal definitions
    theorems/        — theorems and their proofs
    corollaries/     — corollaries and their proofs
    open-questions/  — open questions and partial answers
    references/      — external literature nodes
  edges/
    derivation/      — A derives_from B
    evidence/        — A evidences B
    citation/        — A cited_by B
```

## Provenance Invariant

From **Definition 18 (Provenance)** in the paper:

> For any proposition $p \in K_{collective}$, provenance is the triple:
> (attribution, evidence, derivation)

This applies to every node AND every edge. An edge without provenance is an unverified claim — indistinguishable from noise (Theorem 10, Corollary 13).

## Schema

See `skills/librarian/SKILL.md` for the full node and edge schema.

## Agents

- **Librarian**: writes to this directory
- **Researcher**: proposes entries to the librarian
- **Validator**: reads this directory to verify provenance completeness
- **Writer**: reads citation nodes to update `references.bib` and paper references
