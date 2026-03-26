# The Architecture of Intelligence from Bounded Active Context

Patrick D. McCarthy

---

## Paper Series

A paper series deriving the architecture of intelligence from a single generative constraint: bounded active context — a hard limit on the number of propositions simultaneously available for inference.

### Formal chain (Papers 1–4)

| Paper | Title | Source | Pages |
|---|---|---|---|
| 1 | Graph Structure Is Necessary for Information Preservation Under Bounded Context | [McCarthy2026_1_GraphNecessity.tex](McCarthy2026_1_GraphNecessity.tex) | 18 |
| 2 | Convergence of Control Architectures to Escalation Under Bounded Context | [McCarthy2026_2_Escalation.tex](McCarthy2026_2_Escalation.tex) | 18 |
| 3 | Knowledge as Normalization: Bounded Context Separates Shared Propositions from Action | [McCarthy2026_3_Normalization.tex](McCarthy2026_3_Normalization.tex) | 12 |
| 4 | Evaluation-Driven Descent, Encoding Permanence, and the Structural Invariants of Intelligence | [McCarthy2026_4_Intelligence.tex](McCarthy2026_4_Intelligence.tex) | 18 |

### Application papers

| Paper | Title | Source | Pages | Status |
|---|---|---|---|---|
| 5 | Cancer as Escalation Chain Severance Under Bounded Context | [McCarthy2026_5_Cancer.tex](McCarthy2026_5_Cancer.tex) | 25 | Draft — empirical grounding |
| 8 | Context Window Diseases and Cancer Risk: Neural Computation Cost as a Unifying Mechanism | [McCarthy2026_8_ContextDiseases.tex](McCarthy2026_8_ContextDiseases.tex) | 19 | Draft |

Paper 5 applies the framework to cancer biology — mutations as escalation chain breaks, coupling channels as the control hierarchy cancer must sever. Paper 8 extends to psychiatric disorders as context window failures, deriving an inverted-U relationship between context dysfunction severity and cancer risk across 16 conditions, with a three-tier clinical protocol and 12 falsifiable predictions.

### Upcoming

| Paper | Title | Source | Status |
|---|---|---|---|
| 6 | Substrate Transition | [McCarthy2026_6_SubstrateTransition.tex](McCarthy2026_6_SubstrateTransition.tex) | Draft — empirical grounding from agentic software systems |

---

## Submission

Upload-ready PDFs are in [`ssrn_upload/`](ssrn_upload/). Submission metadata and checklist are in [`SSRN_SUBMISSION.md`](SSRN_SUBMISSION.md).

---

## Knowledge Graph

The [`knowledge-graph/`](knowledge-graph/) directory contains ~150 YAML nodes with full provenance — definitions, theorems, equivalency claims, novel results, emergent predictions, and references. Every node carries `(attribution, evidence, derivation)` triples. Edge relations include `derives_from`, `evidences`, `grounds`, `overlaps_with`, `generalizes`, and others.

See [`CONFIDENCE_CHAIN_PAPER.md`](CONFIDENCE_CHAIN_PAPER.md) for the formal epistemic provenance framework and [`knowledge-graph/NOVELTY-ANALYSIS.md`](knowledge-graph/NOVELTY-ANALYSIS.md) for cross-framework comparison.

---

## GNN Pipeline

The [`gnn/`](gnn/) directory contains the graph neural network models for cancer genomics prediction, built on the knowledge graph and trained on MSK-IMPACT clinical data (~44K patients, 66 cancer types, 509 genes).

### Models

| Model | Description |
|---|---|
| **AtlasTransformer V5** | Hierarchical block-sparse attention matching encoding hierarchy theory — mutation → pathway block → channel → organism. Edge types from Neo4j bias attention directly. |
| **ChannelNet V8** | Channel-aware mutation interaction networks |
| **CoxSAGE** | Graph attention + Cox proportional hazards survival loss |
| **DirectionalWalk** | Interpretable random walk inference on patient mutation subgraphs |
| **HierarchicalScorer** | Multi-level graph scoring with learned edge weights |

### Learning loop

The system discovers its own graph structure: transformer attention → candidate edges → gate by C-index delta → commit to Neo4j if beneficial → invalidate caches → retrain. See [`gnn/learning_loop.py`](gnn/learning_loop.py).

### Key results

- C-index 0.628 on 5-fold cross-validation (best: AtlasTransformer V5)
- Per-mutation coefficients, not per-cancer fitting — predictions compose across mutation profiles
- 10 edge types discovered dynamically from Neo4j schema, 24 edge feature dimensions

---

## Analysis

The [`analysis/`](analysis/) directory contains empirical studies on the MSK-IMPACT cohort:

- **Mutation survival atlas** — per-gene per-cancer-type hazard ratios
- **Channel survival** — Kaplan-Meier by coupling channel
- **Mutation interactions** — gene-pair interaction effects on survival
- **Metastatic tropism** — organ-specific mutation patterns
- **Escalation entropy** — information-theoretic measures of escalation chain disruption
- **Treatment-channel matching** — drug sensitivity aligned to escape channel
- **Protective variants** — hypothesis: coupling-channel-strengthening GOF variants enrich in high-risk individuals who never develop cancer

---

## Infrastructure

- **Neo4j** — graph database backing the knowledge graph and GNN data pipeline. Schema discovered dynamically at runtime.
- **Inference layer** ([`inference/`](inference/)) — document embedding, clustering, and keyword extraction pipeline for semantic search across papers.
- **Annotated formulas** ([`docs/annotated/`](docs/annotated/)) — every formula in the paper series explained.

---

## References

[`references.bib`](references.bib) — BibTeX file shared across all papers.

---

## License

© Patrick D. McCarthy. All rights reserved.
