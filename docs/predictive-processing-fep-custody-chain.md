# Predictive Processing and Free Energy: Custody Chain Evaluation

**Cluster**: Rao-Ballard (1999), Friston (2010), Friston et al. (2017), Friston et al. (2021)

---

## Framework Role Classification

| Category | Nodes | Custody concern |
|---|---|---|
| **EM03 (graph necessity grounds FEP)** | FEP7 + NV01 | Central novelty finding; must be precisely stated |
| **EM02 (FEP6/NV02 tension)** | FEP6 vs NV02 | Tension document; resolution by scope (stationary vs. non-stationary environments) |
| **Theorem 5 foil** | FEP4 precision weighting | Used as the architecture that fails; must not misrepresent FEP4 |
| **Neural confirmation** | RB1 (escalation) | Empirical evidence for escalation over precision allocation |
| **OV04 (sentience overlap)** | SI4 + D14 | Derivation vs. definition distinction must be preserved |
| **OV01 (bounded model overlap)** | FEP1 + EC01 | Markov blanket vs. C_n; same structural claim, different formalisms |

---

## REF-RaoBallard1999

**Citation status**: Verified (DOI: 10.1038/4580)

### RB1 (predictive coding = escalation) → Theorem 2a

**Evidence grade**: B+ (structural, empirical confirmation)

RB1 is empirical evidence that the neural architecture implements bottom-up error propagation (escalation), not top-down precision allocation. This confirms Theorem 5's efficiency argument: the cortex, shaped by selection pressure, uses the architecture UB validates.

One step: "bottom-up prediction error propagation in cortex" must be identified with "escalation architecture in Theorem 2a." Well-motivated — both describe a system where routine processing is handled locally and only errors/failures propagate to higher levels — and the Rao-Ballard model is explicitly the computational model of cortical hierarchy used in FEP. Grade B+ (not A) because the identification requires accepting that the cortical hierarchy is a physical instantiation of the formal escalation architecture.

**Unique contribution**: the cortex implements the architecture that UB proves is efficient under selection pressure. This is not just structural confirmation — it is empirical evidence from the most selection-pressure-tested cognitive system available (the primate cortex over millions of years of evolution).

**Custody concern**: the claim "the cortex implements escalation, confirming Theorem 5" should be stated as "consistent with" rather than "proves." Theorem 5 is a formal result; RB1 is empirical evidence for the architecture that Theorem 5 validates. These are logically distinct.

### RB2 (hierarchical generative model) → EC03, NV01

**Evidence grade**: B (empirical confirmation)

RB2 confirms that the cortex uses the hierarchical generative model structure that NV01 proves necessary. One step: "cortical hierarchical generative model" must be identified with "K/F graph as hierarchical generative model (EC03)." Well-motivated — the neural architecture and the formal graph structure serve the same function — but the neural implementation and the formal definition are not identical. Grade B.

**Combined with EM06**: RB2 provides empirical evidence that gradient descent toward prediction error minimization converges to graph structure, confirming EM06 (gradient drives toward graph construction). The cortex did not have graph structure built in — it emerged through learning.

---

## REF-Friston2010

**Citation status**: Verified (DOI: 10.1038/nrn2787)

### FEP (free energy principle) as a neighboring framework

The FEP (2010) citation establishes the prior art context. The custody concern here is one of positioning: FEP is not a derivation that grounds UB, nor is UB a derivation that grounds FEP. They are independent formal frameworks addressing the same domain. The relationship is overlap and comparison.

The specific edges in the graph (FEP grounded by UB via EM03, FEP in tension with UB via EM02) are the correct relationship type. The custody concern: do not cite FEP as grounding UB, and do not cite UB as derived from FEP. The relationship is formal comparison.

**Independence**: Very strong. Friston (2010) is neuroscience, not information theory. FEP is derived from Helmholtz's free energy and Bayesian brain hypothesis, not from Shannon.

---

## REF-Friston2017

**Citation status**: Verified (DOI: 10.1162/NECO_a_00912)

### FEP7 (assumed graph structure) → EM03

**Evidence grade**: A (definitional for EM03)

EM03 (graph necessity grounds FEP) documents a clean logical relationship: FEP7 assumes what NV01 proves. The evidence grade is A because EM03 is defined as exactly this gap. The identification requires no interpretive step: FEP7 explicitly states that the generative model is hierarchically structured by assumption, and NV01 provides the information-theoretic proof that this structure is necessary.

**What makes EM03 significant**: this is one of the rare cases where UB provides a formal grounding that a prior, well-established formal framework explicitly lacks. FEP is a rigorously developed theory; the absence of a derivation for FEP7 is a known gap in the FEP literature. UB closes that gap.

**Custody concern**: the claim should be stated precisely. "NV01 proves graph structure is necessary; FEP7 assumes it" — not "FEP requires UB" (stronger than warranted) or "FEP is consistent with UB" (weaker than warranted). EM03 is the precise middle ground: UB provides the proof that FEP's assumption needs.

### FEP6 (long-run equilibrium) vs NV02 → EM02

**Evidence grade**: A (definitional for EM02)

EM02 documents a formal tension between two theoretically derived claims. Grade A because the tension is definitional: in a stationary environment, FEP6 (approaching equilibrium) and NV02 (M < N always) are formally inconsistent. The resolution — environments are non-stationary for A > 0 entities — is stated in EM02.

**Precision required**: the tension is real and significant, but it should not be stated as FEP being wrong. FEP6 holds in stationary environments; NV02 holds under A > 0 dynamics in non-stationary environments. Both are correct within their respective scopes. EM02 documents where the scopes diverge.

### FEP4 (precision weighting) as Theorem 5 foil

**Evidence grade**: A (definitional for the Theorem 5 argument)

Theorem 5 (Escalation Principle) uses FEP4's precision allocation as the architecture that fails under selection pressure. The argument is: top-down precision allocation requires updating the entire generative hierarchy when context changes; bottom-up escalation only activates when prediction fails. Under selection pressure, bottom-up escalation is less costly. Grade A for the foil role: FEP4 is explicitly cited as the architecture the theorem is evaluated against.

**Custody concern**: the Theorem 5 argument must not misrepresent FEP4. FEP4 (precision weighting) is a valid model of attentional weighting in FEP. The claim is not that FEP4 is incorrect — it is that it is not the dominant architecture under selection pressure. The distinction: FEP4 describes one valid mechanism; Theorem 5 argues that a different mechanism (bottom-up escalation) dominates when selection pressure acts.

### FEP8 (no A parameterization) → NV03

**Evidence grade**: A (definitional for NV03 gap)

FEP8 documents an explicit gap: FEP has no A parameter. NV03 (agency as gate) is the UB node that fills this gap. Grade A because the gap is documented in the node's provenance and FEP8 is the explicit statement of the absence.

**Custody concern**: same concern as Schmidhuber S4. The FEP8 gap should be cited when arguing for NV03's novelty: both Schmidhuber and Friston lack the A parameter; this convergence strengthens the claim that A is a genuine formal addition rather than a minor variation.

---

## REF-Friston2021

**Citation status**: Verified (DOI: 10.1371/journal.pcbi.1008762)

### SI4 (meta-cognitive awareness) → OV04, D14

**Evidence grade**: B (structural, overlap)

SI4 (sophisticated inference = meta-cognitive awareness) overlaps with Definition 14 (higher-order sentience). One step: "modeling own future inference states" must be identified with "representing that other projections of W exist (functional sentience)." Well-motivated — both describe the extension of the agent's model to include its own modeling process — but the FEP architectural formulation (the agent models future belief states under policies) and the UB inductive formulation (the entity represents that W is larger than its current K) are different characterizations of the same structural property.

Grade B because this is overlap, not equivalence: the two frameworks describe the same structural extension from different theoretical positions. The distinction (derivation vs. definition) must be preserved.

**Key distinction**: UB derives functional sentience from the information-theoretic structure; FEP sophisticated inference defines it as a richer inference architecture. Both reach the same structural claim. The derivation is more powerful: it establishes that functional sentience is not a design choice but a consequence of the information-theoretic constraints.

**Independence**: Strong. Friston et al. (2021) develop sophisticated inference from FEP's own theoretical trajectory, not from UB.

---

## Overall Cluster Assessment

| Source | Best grade | Key claim | Independence |
|---|---|---|---|
| Rao-Ballard 1999 | B+ (RB1/escalation) | Cortex implements escalation architecture; confirms Theorem 5 | Strong |
| Friston 2010 | A (FEP4 foil) | FEP as neighboring framework; precision allocation = Theorem 5 foil | Very strong |
| Friston et al. 2017 | A (EM03, EM02) | FEP7 assumes what NV01 proves; FEP6/NV02 tension documented | Very strong |
| Friston et al. 2021 | B (SI4/OV04) | Independent FEP grounding of functional sentience; derivation vs. definition | Strong |

**Priority actions**:
1. State EM03 precisely: "NV01 proves graph structure necessary; FEP7 assumes it" — neither "FEP requires UB" nor "FEP is consistent with UB." The proof fills a documented gap.
2. State EM02 scope condition explicitly: FEP6 holds in stationary environments; NV02 holds for A > 0 in non-stationary environments. The tension resolves by scope, not by one framework being wrong.
3. Preserve the Theorem 5/FEP4 distinction: FEP4 (precision weighting) is not wrong; it is not the dominant architecture under selection pressure. State this precisely to avoid misrepresenting FEP.
4. Preserve the OV04/D14 derivation-vs-definition distinction: UB derives functional sentience; FEP sophisticated inference defines it architecturally. Both reach the same structural claim; the derivation is the stronger statement.
5. Add the FEP8 gap (no A parameter) to the citation support for NV03 novelty, alongside Schmidhuber S4. Two major independent frameworks lack A; this strengthens NV03's claim as a genuine formal addition.
6. Document RB1 as empirical evidence that selection pressure shaped the cortex toward the escalation architecture — consistent with Theorem 5 but not a proof of it.

---

## Completion Note

With this cluster, all 47 REF nodes in the corpus have received cross-framework profile and custody chain evaluation. The seven completed cluster pairs are:

1. Foundational Information Theory (Shannon, Cover, Kolmogorov, Chaitin, Solomonoff, Jaynes)
2. Logic, Language, and Pragmatics (Gödel, Tarski, Peirce ×2, Gibson)
3. Statistical Learning and AI (Vapnik, Hinton-Rumelhart, Thrun, Vaswani, Schmidhuber, Tishby)
4. Cognitive Psychology (Miller, Cowan, Kahneman, Baars, Dehaene)
5. Consciousness and Self-Reference (Chalmers, Tononi, Clark-Chalmers, Hofstadter, Bateson, Deacon)
6. Biology, Thermodynamics, and Complex Systems (Schrödinger, England, Maynard Smith, Waddington, Dunbar, Prigogine, Watts-Strogatz, Axelrod)
7. Predictive Processing and FEP (Rao-Ballard, Friston 2010, Friston 2017, Friston 2021)

Previously documented clusters (from session 1):
- Calvin Lin (compiler theory, flow analysis)
- Geoff Buhl (vertex operator algebras)
- String Theory (Polchinski, holography)
- Algebraic cluster (Kac, Borcherds, FLM, Buhl-Karaali)
