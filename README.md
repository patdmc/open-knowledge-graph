# The Architecture of Intelligence from Bounded Active Context

Patrick D. McCarthy

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18923066-blue)](https://doi.org/10.5281/zenodo.18923066)

---

## Paper Series

A paper series deriving the architecture of intelligence from a single generative constraint: bounded active context — a hard limit on the number of propositions simultaneously available for inference.

### Theory (Papers 1–4)

| Paper | Title | PDF |
|---|---|---|
| 1 | Graph Structure Is Necessary for Managing Consistency and Availability Under Bounded Context | [PDF](publications/research/McCarthy2026_1_GraphNecessity.pdf) |
| 2 | Convergence of Control Architectures to Escalation Under Bounded Context | [PDF](publications/research/McCarthy2026_2_Escalation.pdf) |
| 3 | Knowledge as Normalization: Consistency Under Partition Separates Shared Propositions from Action | [PDF](publications/research/McCarthy2026_3_Normalization.pdf) |
| 4 | Evaluation-Driven Descent, Encoding Permanence, and the Structural Invariants of Intelligence | [PDF](publications/research/McCarthy2026_4_Intelligence.pdf) |

### Empirical (Papers 5–8)

| Paper | Title | PDF |
|---|---|---|
| 5 | Genome as Projection: Coupling Channels Predict Cancer Survival and Suggest Drug Response Principles | [PDF](publications/research/McCarthy2026_5_Cancer.pdf) |
| 6 | Coupling-Channel Structure Predicts Cancer Survival | [PDF](publications/research/McCarthy2026_6_ChannelStructure.pdf) |
| 7 | Graph Consequences of DNA Serialization | [PDF](publications/research/McCarthy2026_7_GraphPosition.pdf) |
| 8 | The Cost of Computation: Shared Molecular Machinery Between Learning and Cancer | [PDF](publications/research/McCarthy2026_8_ContextDiseases.pdf) |

### Bridge (Papers 9–11)

| Paper | Title | PDF |
|---|---|---|
| 9 | Determinism Precipitates from Uncertainty: When Agents Should Become Scripts | [PDF](publications/research/McCarthy2026_9_SubstrateTransition.pdf) |
| 10 | The Genome as Knowledge Graph: Governance, Scaling, and the Architecture of Multicellular Life | [PDF](publications/research/McCarthy2026_10_GenomeAsGraph.pdf) |
| 11 | The Ratchet: Evolution and Cancer as Opposing Failures of the Same Mechanism | [PDF](publications/research/McCarthy2026_11_TheRatchet.pdf) |

Paper 5 applies the framework to cancer biology — mutations as escalation chain breaks, coupling channels as the control hierarchy cancer must sever. Paper 8 extends to psychiatric disorders as context window failures, deriving an inverted-U relationship between context dysfunction severity and cancer risk across 16 conditions.

---

## Submission

Submission metadata and checklist are in [`publications/research/SSRN_SUBMISSION.md`](publications/research/SSRN_SUBMISSION.md).

---

## Reproducibility Notebooks

Interactive Colab/Jupyter notebooks that reproduce the empirical results from the paper series using public cBioPortal data.

| Notebook | Paper | Description |
|---|---|---|
| [paper5_reproduce.ipynb](notebooks/paper5_reproduce.ipynb) | 5 | Channel count vs mutation count survival analysis across MSK-IMPACT cohorts |

---

## Essays

Companion essays for a general audience, each grounded in one or more papers from the series.

| Essay | Paper | Core argument |
|---|---|---|
| [Separating and Bounding Concerns](publications/medium/separating_bounding_concerns.md) | 1–3 | Factoring is mandatory under bounded context |
| [Transformers Are Dinosaurs](publications/medium/transformers-are-dinosaurs.md) | 3 | CAP theorem forces the graph; LLMs are the availability layer |
| [Stop Building Ralph Loops](publications/medium/stop_building_ralph_loops.md) | 4 | Learning that doesn't build durable structure isn't learning |
| [Cancer Is an Org Chart Problem](publications/medium/cancer_org_chart.md) | 5 | Channel count = levels of org chart compromised; resolves TMB and Peto's paradoxes |
| [Shadows on the Wall](publications/medium/shadows_on_the_wall.md) | 6 | The right simplicity explains more than the wrong complexity |
| [Every Story Has a Spine](publications/medium/every_story_has_a_spine.md) | 7 | Language and DNA are both serial recordings of graphs; hubs fall first |
| [What if Cancer Is Rational?](publications/medium/cancer_post.md) | 8 | The machinery that enables learning is the machinery cancer exploits |

---

## Knowledge Graph

The [`knowledge-graph/`](knowledge-graph/) directory contains ~150 YAML nodes with full provenance — definitions, theorems, equivalency claims, novel results, emergent predictions, and references. Every node carries `(attribution, evidence, derivation)` triples. Edge relations include `derives_from`, `evidences`, `grounds`, `overlaps_with`, `generalizes`, and others.

See [`publications/research/CONFIDENCE_CHAIN_PAPER.md`](publications/research/CONFIDENCE_CHAIN_PAPER.md) for the formal epistemic provenance framework and [`knowledge-graph/NOVELTY-ANALYSIS.md`](knowledge-graph/NOVELTY-ANALYSIS.md) for cross-framework comparison.

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

- **Neo4j** — graph database backing the knowledge graph and data pipeline. Schema discovered dynamically at runtime.
- **Inference layer** ([`inference/`](inference/)) — document embedding, clustering, and keyword extraction pipeline for semantic search across papers.
- **Annotated formulas** ([`docs/annotated/`](docs/annotated/)) — every formula in the paper series explained.

---

## References

[`references.bib`](publications/shared/references.bib) — BibTeX file shared across all papers.

---

## License

© Patrick D. McCarthy. All rights reserved.
