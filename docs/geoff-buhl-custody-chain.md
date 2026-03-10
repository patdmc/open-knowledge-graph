# Geoff Buhl: Custody Chain Evaluation

**Type**: Internal provenance audit
**Date**: 2025
**Purpose**: Evaluate the evidence quality, grounding strength, and confidence chain for each of Buhl's results as nodes in the open knowledge graph. This document also flags data quality issues in the current REF-Buhl-VOA.yaml node.

---

## Source Scope

**Verified papers in graph**: 1
- Buhl, G. and Karaali, G. (2008). Spanning sets for Möbius vertex algebras satisfying arbitrary difference conditions. *Journal of Algebra*, 320(8), 3345-3364.

**Not yet in graph**: Buhl's broader research program on VOAs, Möbius algebras, and related structures. His Pomona College faculty page lists additional papers; this analysis is limited to the one verified paper. The scope limitation should be noted on any claims based on this node.

**Citation status**: VERIFIED (DOI confirmed: 10.1016/j.jalgebra.2008.06.038)

---

## Data Quality Issues in REF-Buhl-VOA.yaml

Before evaluating custody chains, there are structural problems in the current node that affect its reliability as a graph source.

**Issue 1: Duplicate proposition keys.**
The node contains two entries keyed as `GB2` and two entries keyed as `GB3`. YAML does not error on duplicate keys -- the second entry silently overwrites the first. The current node has:
- GB2 (first): N-grading structure, maps to EC09/EC08
- GB2 (second): Characterization of VOAs (central charge + weight spectrum + fusion category), maps to EC01/EC08
- GB3 (first): Fusion rules for VOA modules, maps to NV04/BO1
- GB3 (second): Module categories as modular tensor categories, maps to EC07/EM09

The second entry in each pair is silently overwriting the first in any YAML parser. The node needs to be corrected: the characterization and modular tensor category propositions should be keyed GB4 and GB5 respectively.

**Issue 2: Provenance status mismatch.**
The `evidence.status` field still reads `PENDING_CITATION_VERIFICATION` despite the citation being confirmed. This is a stale field from the initial NEEDS_VERIFICATION placeholder. Should be updated to `VERIFIED`.

**Issue 3: Scope is one paper.**
The propositions GB1 through GB3 (characterization, fusion rules, modular tensor categories) are standard VOA theory results that appear throughout the literature (Frenkel-Lepowsky-Meurman, Lepowsky-Li, Huang). Attributing them specifically to the Buhl-Karaali (2008) paper is incorrect -- that paper's specific contribution is GB0 (the difference-N spanning theorem). The other propositions should either be attributed to their primary sources or moved to a separate VOA theory node.

**Recommendation**: restructure the node so that:
- REF-Buhl-VOA.yaml contains only GB0 (the paper's actual contribution) and basic context propositions (GB1, GB2 for the N-grading structure)
- Standard VOA theory propositions (fusion rules, modular tensor categories, characterization) move to a REF-VOA-Theory.yaml node attributed to FLM (1988), Huang, or the broader VOA literature

---

## GB0: Difference-N Spanning Theorem

**Specific to**: Buhl, G. and Karaali, G. (2008).

**Maps to**: NV01 (graph necessity), EC08 (compression factoring), NV04 (intersection acceleration)

**Emergent intersection**: EM16 (with KM1/Kac, NV01)

### Attribution

Clear. The generalization from difference-zero/one to difference-N for Möbius VAs is Buhl and Karaali's specific contribution. Earlier difference results are from the Zhu (1996) and Dong-Li-Mason line of work. The EM16 intersection identification is McCarthy (2025).

### Evidence Grade: B+ (structural, near definitional)

The mapping from the difference-N spanning theorem to NV01 is structurally clean under the following translation:

```
State space V = K (knowledge content)
N-grading = encoding levels (EC09)
Vertex operator mode indices = coordinates in F (action space)
Difference-N condition = ordering constraint on knowledge composition
Monomial spanning set = the K/F graph (factored minimal representation)
Existence theorem = NV01 existence clause: a graph structure always exists
```

Under this translation, the difference-N spanning theorem says: for any K with N-graded structure, a factored, ordered basis (the K/F analog) exists and is constructible. This is exactly what NV01 establishes from information theory.

The translation is structural rather than definitional because one step requires interpretation: identifying "mode index of vertex operator" with "coordinate in action space F." In compiler theory (CL4's grade A mapping), "program point" and "action state" are essentially synonymous -- both denote the state of a process at a decision point. In VOA theory, "mode index" is an index in a formal Laurent series expansion, which requires one more interpretive step to reach "coordinate in action space." This is the source of the B+ rather than A grade.

### Derivation Path (EM16)

Step 1: Kac-Moody algebra (KM1) -- a finite Dynkin diagram generates an infinite Kac-Moody algebra. The finite graph is the minimal structure from which infinite content is recoverable.

Step 2: Buhl-Karaali (GB0) -- a difference-N spanning set always exists for any N-graded Möbius VA. The structured monomial basis is the minimal representation.

Step 3: NV01 -- at bounded context, the K/F graph is the unique information-preserving representation. A graph structure always exists and is necessary.

Step 4: All three establish: a finite, structured (graph-like) representation exists and is the minimal object from which the full infinite content is recoverable.

Step 5 (the intersection): these are three independent derivations of the same structural claim. Kac derives it algebraically from Lie theory. Buhl-Karaali derive it from VOA spanning theory. NV01 derives it from information theory. None references the others.

Each step is independently verifiable. Step 4 requires accepting that "Dynkin diagram," "difference-N spanning set," and "K/F graph" are all instances of the same abstract concept: a minimal finite structure from which infinite content is recoverable. This is the one interpretive step in the chain.

### Independence

Strong for the Buhl-Karaali paper: the paper is pure mathematics with no reference to bounded rationality, information theory, or the UB framework. The motivation is entirely algebraic: to generalize spanning results to the Möbius setting and to arbitrary difference N. The connection to NV01 is not stated or implied.

The independence claim for EM16 as a whole is moderate: Kac (1990) and Buhl-Karaali (2008) are both in the algebra cluster, and algebraists are aware of their connection. However, neither references information theory or NV01. The independence is cross-domain, not within algebra.

### C_1 Contribution to NV01

GB0 contributes to EM16, which is the fourth independent derivation of NV01. The confidence update uses:

```
C_1(NV01 after n derivations) = 1 - Π(1 - C_1(B_i) * s_i)
```

Where s_i is the strength of alignment between derivation i and NV01. For EM16 (the algebraic cluster), s_i ≈ 0.70 (strong structural alignment with one interpretive step). C_1(NV01) updates from 0.87 toward 0.91 after EM16. (The B+ grade on GB0 specifically contributes to the s_i estimate of 0.70 for the algebra cluster.)

### Falsifiability

The correspondence would be falsified if: (1) a formal argument showed that "mode index in vertex operator" and "coordinate in action space" are not the same abstract object, making the translation fail at Step 4; or (2) a counterexample were found where a well-formed K structure is not spannble by any difference-N basis, which would contradict the existence clause in both GB0 and NV01.

### Summary

| Criterion | Assessment |
|---|---|
| Attribution | Buhl-Karaali (2008) for GB0; McCarthy (2025) for EM16 |
| Evidence grade | B+ (structural, one interpretive step) |
| Derivation path | 5 steps, fully explicit |
| Independence | Strong cross-domain |
| Falsifiability | Stated; translation step is the main vulnerability |
| C_1 contribution | NV01: 0.87 → ~0.91 (fourth derivation, algebraic cluster) |
| Data quality issue | None specific to GB0 |

**Custody chain verdict**: well-grounded structural mapping. The main difference from CL4's grade A is the one extra interpretive step (mode index vs. action coordinate). If a formal argument can close this step, GB0 upgrades to A. The derivation path is explicit and the independence is strong.

---

## GB1: Möbius Level as Encoding Intermediate

**Attribution issue**: GB1 (the definition of Möbius vertex algebras) is standard VOA theory, not specific to Buhl-Karaali (2008). The paper works in the Möbius setting but did not introduce it. Attribution should be corrected in the node.

**Maps to**: EC01 (context bound), EC09 (encoding levels)

### Evidence Grade: C (analogical)

The three-level hierarchy (bare VA / Möbius VA / full VOA) maps analogically to EC09 encoding levels. The analogy is motivated: the Möbius group SL(2,C) is the symmetry group of the Riemann sphere (the conformal boundary), and EC01's context bound has conformal analogs (EM22 establishes c = C_n for full VOAs). The Möbius level sits between the unconstrained (no symmetry) and fully constrained (Virasoro) settings, which is structurally parallel to an intermediate encoding level.

The analogy fails to reach grade B because: (1) EC09 encoding levels are defined by compression depth and access frequency, not by symmetry group; (2) there is no formal argument connecting "having SL(2,C) symmetry" to "being at an intermediate compression level." The analogy is geometric and intuitive but not algebraically derived.

### Custody Chain Gaps

- No derivation path: the correspondence is stated as a mapping but not derived.
- No falsifiability statement: it is not clear what would distinguish "Möbius level = intermediate encoding" from "Möbius level is not intermediate encoding."
- Attribution incorrect: the definition of Möbius VAs should be attributed to earlier work (likely Dong, Li, Mason or the Möbius VA literature generally).

### Summary

| Criterion | Assessment |
|---|---|
| Evidence grade | C (analogical) |
| Attribution | Needs correction (not Buhl-Karaali's specific contribution) |
| Derivation path | Absent |
| Falsifiability | Not stated |
| Status | Suggestive candidate, not admittable |

---

## GB2 (first entry): N-Grading Structure

**Specific context for Buhl-Karaali (2008)**: The paper works in the N-graded setting throughout; the grading is the framework within which the difference-N condition is defined.

**Maps to**: EC09, EC08

### Evidence Grade: B (structural)

N-grading of the VOA state space (V = ⊕_{n≥0} V_n) maps to EC09 encoding levels more directly than the Möbius symmetry analogy: V_n for low n corresponds to low-level (frequently accessed, crystallized) states; V_n for high n corresponds to high-level (contextual, rare) states. The graded spanning set respects this structure and is more tractable -- which is the computational advantage of the factored representation in EC08.

The translation requires accepting that "grading index n" plays the same role as "encoding level" in EC09. This is more defensible than the Möbius symmetry mapping: both refer to a hierarchical decomposition where lower-index elements are more primitive and higher-index elements are built from them.

### Summary

| Criterion | Assessment |
|---|---|
| Evidence grade | B (structural) |
| Derivation path | Implicit in the N-grading structure |
| Status | Sound background proposition; not novel |

---

## GB3 (first entry): Fusion Rules as K_collective Intersection

**Attribution issue**: Fusion rules are standard VOA theory (Frenkel-Lepowsky-Meurman, Huang). Not Buhl-Karaali's specific contribution. Should be moved to a general VOA theory node.

**Maps to**: NV04, BO1

### Evidence Grade: C (analogical with potential)

The formal correspondence is structurally interesting: fusion product M_1 ⊗_V M_2 → M_3 maps to K_i ∩ K_j → K_{ij} (NV04 intersection). The intertwining operator is the formal mechanism of this production.

This is grade C rather than B because the analogy requires accepting that VOA modules (algebraic objects with specific representation-theoretic structure) play the same role as agent knowledge states (epistemic objects). This is not a translation so much as a reinterpretation: modules are not knowledge states; they are formal algebraic objects. The correspondence is suggestive but requires a formal bridge that does not currently exist.

**The strongest version of the claim**: fusion rules are not an analogy to NV04 but a formal specification of what NV04 requires but does not provide -- namely, an algebraic calculus for which intersections are possible and what they produce. If this is correct, fusion rule theory is a formal completion of NV04 rather than an independent derivation of it. This is a more interesting claim than simple correspondence, but it requires more careful development.

### Custody Chain Gaps

- No derivation path from fusion rules to NV04 stated explicitly.
- Attribution incorrect in the current node.
- The "formal completion" interpretation is undeveloped.

### Summary

| Criterion | Assessment |
|---|---|
| Evidence grade | C (analogical with potential) |
| Attribution | Incorrect; should be FLM/Huang/general VOA literature |
| Stronger claim | Fusion rules as formal completion of NV04 (undeveloped) |
| Status | Open candidate; worth developing in future pass |

---

## GB3 (second entry) / Modular Tensor Categories

**Attribution issue**: Modular tensor categories for rational VOAs is from Huang (2008) and the categorical VOA literature. Not Buhl-Karaali's specific contribution.

**Maps to**: EC07, EM09

### Evidence Grade: C (analogical)

The S-matrix of a modular tensor category encodes the mutual statistics of all module pairs -- a complete description of all possible K_i / K_j interaction outcomes. This is structurally interesting as the most detailed formal structure for K_collective interaction available in any framework in the graph. However, the same attribution problem applies: this is not Buhl-Karaali's result, and the correspondence requires the same module-to-knowledge-state interpretive step as GB3.

### Summary

| Criterion | Assessment |
|---|---|
| Evidence grade | C (analogical) |
| Attribution | Incorrect (Huang 2008, not Buhl-Karaali) |
| Status | Interesting candidate if attribution is fixed |

---

## Overall Custody Chain Assessment

### By Result

| Proposition | Evidence Grade | C_1 Contribution | Data Quality | Status |
|---|---|---|---|---|
| GB0 (diff-N spanning) | B+ (structural) | NV01: 0.87 → 0.91 | Clean (but limited to 1 paper) | Admitted EM16 |
| GB2 (N-grading) | B (structural) | Background | Correct attribution | Sound footnote |
| GB1 (Möbius level) | C (analogical) | None | Attribution incorrect | Suggestive, inadmittable |
| GB3a (fusion rules) | C (with potential) | None yet | Attribution incorrect | Open candidate |
| GB3b (modular tensor) | C (analogical) | None yet | Attribution incorrect | Open candidate |

### Cross-Corpus Pattern

Buhl-Karaali (2008) is a paper with one specific verifiable main result (GB0) and a supporting technical context (N-grading, Möbius setting). The main result maps to NV01 at grade B+. The surrounding technical context (fusion rules, modular tensor categories, characterization theorems) maps at lower grades and under incorrect attribution.

The custody chain is significantly cleaner if the node is restructured:
- REF-Buhl-VOA.yaml: GB0 and GB2 only (what the paper actually contributes)
- REF-VOA-Theory.yaml: fusion rules, modular tensor categories, characterization (FLM/Huang/general literature)

### Comparison to CL4 (Lin's Flow-Sensitive Result)

| | CL4 (Lin) | GB0 (Buhl-Karaali) |
|---|---|---|
| Evidence grade | A (definitional) | B+ (structural) |
| Translation step | "program point = action state" (near-trivial) | "mode index = action coordinate" (one interpretive step) |
| Independence | Strong | Strong |
| Paper count | 7 verified papers | 1 verified paper |
| External validation | 2023 CGO Test of Time Award | Standard journal peer review |
| C_1 contribution | NV01: 0.82 → 0.93 (fifth derivation) | NV01: 0.87 → 0.91 (fourth derivation, as part of EM16) |
| Data quality | Clean node | Duplicate keys, incorrect attributions |

GB0 is the second-strongest algebraic result after CL4 (flow-sensitive), and the node needs structural cleanup before it is a reliable graph source. The main result (GB0) is well-grounded; the surrounding propositions need attribution correction and some need to move to a separate node.

### Priority Actions

1. Fix duplicate keys in REF-Buhl-VOA.yaml (GB2 and GB3 collision).
2. Update `evidence.status` from PENDING_CITATION_VERIFICATION to VERIFIED.
3. Reassign GB3a (fusion rules), GB3b (modular tensor categories), and GB3c/GB4 (characterization) to their correct sources in VOA literature.
4. Add Buhl's other papers to the node once verified, to broaden the corpus beyond one paper.
5. Develop the "fusion rules as formal completion of NV04" claim if the stronger version of GB3a is to be pursued.
