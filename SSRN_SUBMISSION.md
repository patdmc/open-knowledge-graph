# SSRN Submission Package — Bounded Context Series

**Author:** Patrick D. McCarthy
**Original submission date:** 2026-03-21
**Revision date:** 2026-03-22
**Series title:** The Architecture of Intelligence from Bounded Active Context

---

## Revision Plan

Papers 1–3 submitted in presubmission on 2026-03-21. Old Papers 4 and 5 have been merged into a single paper. Old Paper 6 (Cancer) is unchanged.

| SSRN slot | Action | New source |
|---|---|---|
| Paper 1 | Update | GRAPH_NECESSITY.pdf |
| Paper 2 | Update | PAPER3_ESCALATION.pdf |
| Paper 3 | Update | PAPER2_KF_INSEPARABILITY.pdf |
| Paper 4 (old Gradient/Encoding) | Delete | merged into new Paper 4 |
| Paper 5 (old Intelligence) | Update → becomes Paper 4 | PAPER4_INTELLIGENCE.pdf |
| Paper 6 (old Cancer) | Update → becomes Paper 5 | PAPER6_CANCER.pdf |

---

## Submission Order and Metadata

5 papers total. Papers 1–4 form the formal chain; Paper 5 is the first application paper.

---

### Paper 1 of 5

**Title:** Graph Structure Is Necessary for Information Preservation Under Bounded Context

**SSRN Category:** Computer Science > Artificial Intelligence
**Secondary:** Cognitive Science

**Abstract:**
Any system that (i) accumulates propositions about a world it cannot fully observe, (ii) operates under a bounded active context — a hard limit on the number of propositions simultaneously available for inference, (iii) must preserve information as the proposition count grows beyond that bound, and (iv) receives evidence that can be corrupted or invalidated, necessarily maintains a directed graph over its propositions, with typed edges emerging under competitive pressure (selective necessity). We prove this claim from two independent arguments. The information-preservation argument shows that the only space-creating operation that does not lose information is factoring — extracting shared structure into named nodes with typed edges — and that factoring is graph construction. The retention-dynamics argument shows that any representation supporting sub-linear dependency queries, contextual retrieval, and selective removal with cascade necessarily encodes directed adjacency, which is a semantic graph regardless of substrate. The falsifiable content is the cost prediction: answering "what depends on p?" costs Ω(n) without stored directed adjacency and O(d) with it. We show that continuous-weight mechanisms (e.g., transformer self-attention) implement graph structure rather than constitute an alternative to it. The result applies to any system that stores propositions under bounded context, including artificial neural networks, institutional knowledge systems, and — to the extent that biological representations admit propositional decomposition — biological neural networks. We derive seven falsifiable predictions including: context-optimizing architectures will outcompete context-expanding architectures on cost, and sparse attention patterns will converge toward topology-aware routing.

**Keywords:** knowledge representation, bounded rationality, graph theory, information preservation, bounded context, knowledge graphs, transformer architectures

**File:** GRAPH_NECESSITY.pdf

---

### Paper 2 of 5

**Title:** Convergence of Control Architectures to Escalation Under Bounded Context

**SSRN Category:** Computer Science > Artificial Intelligence
**Secondary:** Cognitive Science

**Abstract:**
Top-down allocation — where a supervisory process directs subproblems to lower levels — is viable when the concurrent delegation count k fits within bounded active context C_n. We prove that as the action repertoire |A| grows under bounded C_n, concurrent supervisory demands k grow because total domain knowledge |K| grows with |A|, exposing more dependency chains to evidence revision. A formal model of evidence cascades shows E[k] = ν · d̄ · τ, where ν is novelty rate, d̄ is average out-degree, and τ is resolution time; since d̄ grows with |K|, k diverges as |A| → ∞. Batching delegations changes the constant but not the scaling behavior. Any control architecture that remains viable must progressively shed supervisory tracking of routine actions, offloading them to autonomous lower levels that signal upward only on failure. This shedding is escalation. The crossover k* = C_n/(2α) is a convention-dependent estimate (the qualitative result — a finite crossover exists — holds for any positive fraction of C_n reserved for reasoning); as |A| → ∞, φ → 1. The pressure is one-directional: escalation never converges toward top-down, because it never introduces anticipatory delegation context. This result refines precision allocation models including the Free Energy Principle and the System M routing architecture of Dupoux, LeCun, and Malik (2026).

**Keywords:** bounded rationality, cognitive architecture, escalation, top-down allocation, free energy principle, hierarchical control, scaling limits, automatic processing

**File:** PAPER3_ESCALATION.pdf

---

### Paper 3 of 5

**Title:** Knowledge as Normalization: Bounded Context Separates Shared Propositions from Action

**SSRN Category:** Computer Science > Artificial Intelligence
**Secondary:** Cognitive Science, Philosophy of Mind

**Abstract:**
When multiple action policies share propositions and active context is bounded, storing each policy's propositions independently produces context load that grows with the number of policies — progressively restricting the entity's ability to compare and select actions. We prove that bounded context forces the extraction of shared propositions into a separate structure K, reducing per-evaluation context load so that each shared proposition is loaded once regardless of how many policies reference it. The overlap fraction β is defined over the raw proposition pool P and grounding sets P^[a], well-defined before K exists. This is knowledge as normalization: K precipitates from the action space A the same way normal forms precipitate from redundant relations in Codd's relational model. The normalization is not a design choice but a scaling constraint imposed by bounded C_n. From the normalization theorem we derive the retention criterion (retain when benefit is uncertain; prune only when benefit is certainly zero) and K/A inseparability (the content of K is constitutively determined by A). These results engage directly with recent proposals to give observation and action separate training objectives (Dupoux, LeCun, and Malik 2026), showing that such separation violates a structural constraint imposed by bounded context.

**Keywords:** knowledge-action coupling, bounded rationality, survival pressure, knowledge representation, normalization, retention criteria, AI architecture

**File:** PAPER2_KF_INSEPARABILITY.pdf

---

### Paper 4 of 5

**Title:** Evaluation-Driven Descent, Encoding Permanence, and the Structural Invariants of Intelligence

**SSRN Category:** Computer Science > Artificial Intelligence
**Secondary:** Cognitive Science, Philosophy of Mind, Psychometrics

**Abstract:**
We establish four linked results for any entity with structure (K, A, σ, γ) persisting under bounded active context, survival pressure, and positive novelty rate ν(E) > 0. First, agency (γ > 0) necessarily produces evaluation-driven state revision — monotone local descent on expected uncertainty H(W|K) — where each informative update reduces H(W|K) by at most the information gain G(a, K), with equality under exact Bayesian conditioning. Under non-stationary task distribution, descent is local and instantaneous rather than globally convergent. Descent is selectively necessary, not merely optimal: under competition, entities that fail to descend are eliminated. Second, the certainty threshold for encoding an action at level L_i is θ_i = 1 − ε/C_i, which derives the learning rate hierarchy from cost-of-failure differences rather than assumption. Third, separating the action space A from the higher-order function space M is selectively favored: collapsing them is possible but increasingly costly under positive novelty rate, so competition eliminates collapsed architectures. Fourth, γ > 0 and learning efficiency η_M > 0 are the unique structural invariants of persistence: the free-context exhaustion argument shows that γ = 0 entities are eliminated under positive novelty rate; without η_M > 0, active context fills without relief and inference degrades to failure. Every other property can equal zero for some persistent entity. Therefore I(E) = γ · η_M — the rate of adaptation to novelty — is the unique measure of intelligence, derived not stipulated.

**Keywords:** intelligence, agency, learning efficiency, structural invariants, bounded rationality, encoding hierarchy, psychometrics, fluid intelligence, g-factor, catastrophic forgetting

**File:** PAPER4_INTELLIGENCE.pdf

---

### Paper 5 of 5

**Title:** Cancer as Escalation Chain Severance Under Bounded Context

**SSRN Category:** Computer Science > Artificial Intelligence
**Secondary:** Theoretical Biology, Oncology, Systems Biology

**Abstract:**
We apply the bounded-context framework of Papers 1–4 to multicellular organization, proposing that cancer is a predictable failure mode of any multi-level escalation hierarchy. In a multicellular organism, individual cells participate in an escalation chain — cell → tissue → organ → organism — where each level extends the effective bounded context C_n^eff of the level below via intercellular signaling. We argue that severing the chain at level k forces the cell back to C_k-level optimization, where the action space available under cell-level bounded context converges on the recognized hallmarks of cancer: aerobic glycolysis, unconstrained proliferation, immune evasion, and motility. This convergence is not atavistic reversion to ancient programs but structural necessity — the same constraint ceiling that governed unicellular life now governs the severed cell. The framework offers structural explanations for three phenomena that existing theories address incompletely: (1) the velocity of malignant transformation (structural failure, not stochastic accumulation), (2) the convergent phenotype across tissue types (same constraint, same necessary structure), and (3) the therapeutic resistance of established tumors (locally rational optimization under cell-level C_n). Clinical oncology is already trending toward the therapeutic strategies this framework predicts — differentiation therapy, immunotherapy, and microenvironment normalization — but largely without a unified theoretical justification. We propose that the bounded-context framework provides one, and offer testable predictions that distinguish this account from the somatic mutation theory, the atavistic theory, and the tissue organization field theory.

**Keywords:** cancer, escalation chain, bounded context, hallmarks of cancer, atavistic theory, tissue organization, differentiation therapy, immunotherapy, Warburg effect, therapeutic resistance

**File:** PAPER6_CANCER.pdf

---

## Series-Level Metadata

**Series Abstract:**
This five-paper series derives the architecture of intelligence from a single generative constraint: bounded active context — a hard limit on the number of propositions simultaneously available for inference. Paper 1 establishes that graph structure is necessary for information preservation. Paper 2 proves that control architectures converge to escalation at scale. Paper 3 proves that bounded context forces normalization of shared propositions into separated knowledge, with asymmetric retention. Paper 4 derives monotone descent on uncertainty, the encoding hierarchy from cost-of-failure, convergence to A/M stratification, and identifies agency and learning efficiency as the unique structural invariants of intelligence, with their product as the derived measure. Paper 5 applies the framework to cancer biology, proposing that cancer is escalation chain severance — the loss of multi-level organizational coupling — and deriving the convergent hallmarks as structurally necessary under cell-level bounded context. Each result is a consequence of bounded context interacting with properties of the physical world: positive entropy, survival pressure, and feedback. The series includes 15 falsifiable predictions spanning machine learning, cognitive psychology, oncology, knowledge engineering, and AI architecture.

**Series Keywords:** bounded rationality, bounded context, intelligence, knowledge representation, graph theory, cognitive architecture, agency, learning, cancer, escalation, information theory

---

## File Mapping (source → PDF)

Note: the LaTeX filenames do not match the logical paper numbering. Here is the mapping:

| Logical Order | Title | LaTeX Source | PDF for Upload |
|---|---|---|---|
| Paper 1 | Graph Structure Is Necessary... | GRAPH_NECESSITY.tex | GRAPH_NECESSITY.pdf |
| Paper 2 | Convergence to Escalation... | PAPER3_ESCALATION.tex | PAPER3_ESCALATION.pdf |
| Paper 3 | Knowledge as Normalization... | PAPER2_KF_INSEPARABILITY.tex | PAPER2_KF_INSEPARABILITY.pdf |
| Paper 4 | Evaluation-Driven Descent... | PAPER4_INTELLIGENCE.tex | PAPER4_INTELLIGENCE.pdf |
| Paper 5 | Cancer as Escalation Chain Severance... | PAPER6_CANCER.tex | PAPER6_CANCER.pdf |

---

## SSRN-Specific Notes

- **Format:** PDF only (no .tex).
- **Category:** CompSciRN (Computer Science Research Network) for all 5. Paper 5 can also be cross-listed to BioRN.
- **No series mechanism:** Each paper is submitted individually. Link them via cross-references in abstracts and your Author Page.
- **Revisions:** You can upload revised versions at any time; old versions are retained.
- **Cost:** Free.

## Pre-Upload Checklist

- [x] All 5 PDFs compile cleanly (verified 2026-03-22)
- [x] Cross-references between papers are consistent (forward chain only)
- [x] Author name consistent: "Patrick D. McCarthy"
- [x] Selective necessity framing consistent across all 5 papers
- [x] Joint likelihood L_K replaces conditional independence (Paper 1)
- [x] A typed over P not K (Paper 3); K emerges from theorem
- [x] Intelligence = rate of adaptation to novelty (Paper 4)
- [x] Benign tumors / non-progression addressed (Paper 5)
- [ ] Delete old Paper 4 (Gradient/Encoding) from SSRN
- [ ] Update old Paper 5 slot with merged Paper 4 (PAPER4_INTELLIGENCE.pdf)
- [ ] Copy PDFs from .temp/ with clean filenames
- [ ] Upload to SSRN
- [ ] Optional: upload to Zenodo for DOI assignment

## Suggested PDF Filenames for Upload

1. `McCarthy2026_BoundedContext_1_GraphNecessity.pdf`
2. `McCarthy2026_BoundedContext_2_Escalation.pdf`
3. `McCarthy2026_BoundedContext_3_Normalization.pdf`
4. `McCarthy2026_BoundedContext_4_Intelligence.pdf`
5. `McCarthy2026_BoundedContext_5_Cancer.pdf`
