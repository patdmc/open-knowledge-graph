# The Architecture of Intelligence from Bounded Active Context

Patrick D. McCarthy

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18923066-blue)](https://doi.org/10.5281/zenodo.18923066)

---

## Paper Series

A paper series deriving the architecture of intelligence from a single generative constraint: bounded active context — a hard limit on the number of propositions simultaneously available for inference.

### Foundation (Paper 0)

| Paper | Title | PDF |
|---|---|---|
| 0 | All Software Is a Graph | [PDF](publications/research/McCarthy2026_0_AllSoftwareIsAGraph.pdf) |

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
| 5b | Not Incidental: Developmental Phenotype Predicts Adult Cancer in HR-Repair Carriers | [PDF](publications/research/McCarthy2026_5b_NotIncidental.pdf) |
| 6 | Coupling-Channel Structure Predicts Cancer Survival | [PDF](publications/research/McCarthy2026_6_ChannelStructure.pdf) |
| 7 | Graph Consequences of DNA Serialization | [PDF](publications/research/McCarthy2026_7_GraphPosition.pdf) |
| 8 | The Cost of Computation: Shared Molecular Machinery Between Learning and Cancer | [PDF](publications/research/McCarthy2026_8_ContextDiseases.pdf) |
| 8b | Presence Without Performance | [PDF](publications/research/McCarthy2026_8b_PresenceWithoutPerformance.pdf) |

### Bridge (Papers 9–11)

| Paper | Title | PDF |
|---|---|---|
| 9 | Determinism Precipitates from Uncertainty: When Agents Should Become Scripts | *forthcoming* |
| 10 | The Genome as Knowledge Graph: Governance, Scaling, and the Architecture of Multicellular Life | [PDF](publications/research/McCarthy2026_10_GenomeAsGraph.pdf) |
| 11 | The Ratchet: Evolution and Cancer as Opposing Failures of the Same Mechanism | [PDF](publications/research/McCarthy2026_11_TheRatchet.pdf) |

### Synthesis (Paper 12)

| Paper | Title | PDF |
|---|---|---|
| 12 | Why Different Folds: Book Thickness, the Anti-Index, and the Functional Contact Graph of the Human Genome | [PDF](publications/research/McCarthy2026_12_WhyDifferentFolds.pdf) |

Paper 12 applies the book thickness theorem from graph theory to chromatin folding, showing that the genome's functional contact graph requires multiple tissue-specific folds. Co-essentiality clustering (DepMap, 14,129 genes) independently recovers the cancer channel architecture from Papers 5-6 at the same rate as Reactome's curated pathways (9.0x vs 8.5x enrichment, statistically indistinguishable). Three cancer failure modes — broken node, broken edge, broken fold — map to LOF, missense, and GOF mutations respectively.

---

## Reproducibility Notebooks

Interactive Jupyter notebooks that reproduce the empirical results using public data. Designed to run in Google Colab or locally.

| Notebook | Paper | Description | Data needed |
|---|---|---|---|
| [paper5_reproduce.ipynb](notebooks/paper5_reproduce.ipynb) | 5 | Channel count vs mutation count survival analysis across MSK-IMPACT cohorts | cBioPortal (auto-download) |
| [paper6_reproduce.ipynb](notebooks/paper6_reproduce.ipynb) | 6 | Channel structure survival prediction | cBioPortal (auto-download) |
| [paper7_reproduce.ipynb](notebooks/paper7_reproduce.ipynb) | 7 | Graph position consequences of DNA serialization | cBioPortal (auto-download) |
| [paper10_reproduce.ipynb](notebooks/paper10_reproduce.ipynb) | 10 | Genome as knowledge graph | cBioPortal (auto-download) |
| [paper12_why_different_folds.ipynb](notebooks/paper12_why_different_folds.ipynb) | 12 | Book thickness, co-essentiality clustering, tissue-specific folds, three failure modes | DepMap (~430 MB, auto-download) + committed derived data |

The Paper 12 notebook is the most comprehensive — it walks the deductive chain from graph theory through polymer physics to cancer biology, generating all figures computationally. Pre-computed Hi-C results from 21 tissues (Schmitt 2016) are included as derived data; the raw Hi-C (~467 GB) is not required.

---

## Essays

Companion essays for a general audience, each grounded in one or more papers from the series.

| Essay | Paper | Core argument |
|---|---|---|
| [Separating and Bounding Concerns](publications/essays/separating_bounding_concerns.md) | 1–3 | Factoring is mandatory under bounded context |
| [Transformers Are Dinosaurs](publications/essays/transformers-are-dinosaurs.md) | 3 | CAP theorem forces the graph; LLMs are the availability layer |
| [Stop Building Ralph Loops](publications/essays/stop_building_ralph_loops.md) | 4 | Learning that doesn't build durable structure isn't learning |
| [Cancer Is an Org Chart Problem](publications/essays/cancer_org_chart.md) | 5 | Channel count = levels of org chart compromised |
| [Shadows on the Wall](publications/essays/shadows_on_the_wall.md) | 6 | The right simplicity explains more than the wrong complexity |
| [Every Story Has a Spine](publications/essays/every_story_has_a_spine.md) | 7 | Language and DNA are both serial recordings of graphs |
| [What if Cancer Is Rational?](publications/essays/cancer_post.md) | 8 | The machinery that enables learning is the machinery cancer exploits |
| [Sam Altman Can Eat My Ass](publications/essays/sam_altman_can_eat_my_ass.md) | 0 | The software engineers are coming for you |

---

## Knowledge Graph

The [`knowledge-graph/`](knowledge-graph/) directory contains ~150 YAML nodes with full provenance — definitions, theorems, equivalency claims, novel results, emergent predictions, and references. Every node carries `(attribution, evidence, derivation)` triples. Edge relations include `derives_from`, `evidences`, `grounds`, `overlaps_with`, `generalizes`, and others.

See [`publications/research/CONFIDENCE_CHAIN_PAPER.md`](publications/research/CONFIDENCE_CHAIN_PAPER.md) for the formal epistemic provenance framework and [`knowledge-graph/NOVELTY-ANALYSIS.md`](knowledge-graph/NOVELTY-ANALYSIS.md) for cross-framework comparison.

---

## Analysis

The [`analysis/`](analysis/) directory contains empirical studies:

### MSK-IMPACT cohort
- **Mutation survival atlas** — per-gene per-cancer-type hazard ratios
- **Channel survival** — Kaplan-Meier by coupling channel
- **Mutation interactions** — gene-pair interaction effects on survival
- **Metastatic tropism** — organ-specific mutation patterns
- **Escalation entropy** — information-theoretic measures of escalation chain disruption
- **Treatment-channel matching** — drug sensitivity aligned to escape channel
- **Protective variants** — hypothesis: coupling-channel-strengthening GOF variants enrich in high-risk individuals who never develop cancer

### Paralog projection (Paper 12)
- **Co-essentiality clustering** — DepMap CRISPR gene effect → functional contact graph → channel recovery
- **Hi-C tissue-specific folds** — GM12878, K562, IMR90 (Rao 2014) + 21 tissues (Schmitt 2016)
- **Book thickness computation** — edge counts at multiple granularities → predicted fold count
- **Mutation spectra by channel** — IntOGen driver mechanisms (LOF/GOF) × channel × age
- **SEER age distributions** — cancer type incidence by age → three failure mode classification

---

## Infrastructure

- **Inference layer** ([`inference/`](inference/)) — document embedding, clustering, and keyword extraction pipeline
- **Annotated formulas** ([`docs/annotated/`](docs/annotated/)) — every formula in the paper series explained

---

## References

[`references.bib`](publications/shared/references.bib) — BibTeX file shared across all papers.

---

## License

© Patrick D. McCarthy. All rights reserved.
