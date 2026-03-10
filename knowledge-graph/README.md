# Knowledge Graph

This directory contains the knowledge graph derived from the formal theory in `UNCERTAINTY_BOUNDING_FORMAL_THEORY.md`. Every node and every edge has a provenance triple: (attribution, evidence, derivation).

## Structure

```
knowledge-graph/
  nodes/
    definitions/     — formal definitions (D01-D14+)
    theorems/        — theorems and their proofs
    corollaries/     — corollaries and their proofs
    open-questions/  — open questions and partial answers
    references/      — external literature nodes (15 frameworks + sources)
    overlap/         — confirmed precipitation sites: propositions grounded by
                       multiple independent frameworks (OV01-OV04+)
    novel/           — propositions novel to this framework: not grounded in
                       any prior work (NV01-NV05+)
    emergent/        — propositions produced by intersection: not in any
                       individual framework, emerge from comparing projections (EM01-EM03+)
  edges/
    derivation/      — A derives_from B
    evidence/        — A evidences B
    citation/        — A cited_by B
  NOVELTY-ANALYSIS.md — full cross-framework analysis, related-work text, and
                        implications for the confidence chain problem
```

## Node Types

| Type | Meaning | C_1(p) basis |
|------|---------|--------------|
| `definition` | Formal definition in UB framework | Framework-internal |
| `theorem` | Proved theorem in UB framework | Formal proof |
| `reference` | External framework or source | Attribution + citation |
| `overlap` | Proposition grounded in UB + at least one other framework independently | Multiple independent groundings — higher C_1(p) |
| `novel` | Proposition grounded only in UB; gap in prior work documented | Single grounding — C_1(p) from proof strength alone |
| `emergent` | Proposition produced by cross-framework intersection; not in any individual framework | Intersection of at least two projections — new knowledge by Corollary 8a |

## Edge Relations

In addition to `derives_from`, `evidences`, `cited_by`:

| Relation | Meaning |
|----------|---------|
| `overlaps_with` | Same proposition grounded in both nodes independently |
| `generalizes` | This node generalizes the target (target is a special case) |
| `generalized_by` | Target generalizes this node |
| `grounds` | This node provides formal grounding for the target |
| `grounded_in` | This node's content is grounded by the target |
| `qualifies` | This node adds a qualification or scope restriction to the target |
| `philosophical_precursor` | Target is a qualitative predecessor of this formal result |
| `instantiated_in` | This abstract claim is instantiated by the target node |
| `scopes` | This node defines the scope or limits of the target |
| `characterizes` | This node characterizes the relationship between other nodes |
| `supported_by` | Target provides empirical or theoretical support for this node |
| `enables` | This node is a precondition for the target |
| `precedes` | This node is a necessary predecessor of the target |
| `contrasts_with` | This node highlights a difference with the target |

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
