# Cognitive Psychology and Consciousness Architecture: Custody Chain Evaluation

**Cluster**: Miller (1956), Cowan (2001), Kahneman (2011), Baars (1988), Dehaene-Changeux (2011)

---

## Framework Role Classification

| Category | Nodes | Custody concern |
|---|---|---|
| **Empirical measurement of C_n** | MI1, CO1 | Empirical vs. formal derivation; values differ (7±2 vs. 4) |
| **Independent confirmation of EM04** | MI3 | Predates UB; explicit Shannon grounding |
| **Encoding hierarchy grounding** | KH1, GWT3, CO2 | Three independent descriptions of same hierarchy |
| **A (Definition 6) neural grounding** | NGW3 | Probabilistic gate; binary at unit level, probabilistic at system level |
| **Provenance requirement grounding** | KH4 | False positives in K = System 1 heuristics without provenance |

---

## REF-Miller1956

**Citation status**: Verified (Psychological Review, 1956; canonical citation)

### MI1 (7±2 chunks) → EC01

**Evidence grade**: B (empirical, not formal derivation)

Miller's 7±2 is a behavioral measurement of C_n. The mapping to EC01 (bounded context) requires accepting that behavioral working memory span is the correct empirical instantiation of C_n. One step: C_n in UB is a formal bound on the information channel of an entity's inference; working memory span is a behavioral measurement of how many items a human can actively maintain. The two are related but not definitionally identical — working memory span can be affected by rehearsal strategies and chunking that don't apply to all entities with A > 0.

Grade B (not A) because the formal C_n and the behavioral 7±2 are parallel but distinct measures. The behavioral measure is constrained by human cognitive architecture; C_n is a general information-theoretic bound.

**Empirical value**: MI1 is the most widely cited empirical anchor for C_n in the corpus. Despite the one-step translation, the convergence of the formal derivation (Shannon) and the behavioral measurement (Miller) on the same structural constraint is significant evidence for C_n's generality.

### MI3 (C_n as Shannon channel capacity) → EM04

**Evidence grade**: A (independent empirical confirmation)

Miller (1956) explicitly uses Shannon's (1948) framework: working memory is a channel with capacity approximately 7±2 bits per chunk. This is an independent empirical confirmation of EM04 (C_n = Shannon channel capacity) from 1956. Miller did not derive EM04 formally — he used Shannon's framework as the interpretive lens — but the explicit Shannon citation means MI3 is not just an analogy. Miller treats the channel capacity framework as the correct formal description of working memory limits.

Grade A for the EM04 confirmation specifically: Miller's paper explicitly states that C_n = Shannon channel capacity for the human cognitive system. No interpretive gap.

**Independence**: Strong. Miller (1956) is cognitive psychology. The Shannon grounding is explicit in the paper, predating the UB framework's EM04 derivation by decades.

**Custody concern**: EM04 should cite Miller (1956) as an independent empirical confirmation predating the UB formalization. Currently EM04's grounding is primarily SH3 (Shannon channel capacity). Adding MI3 strengthens C_1(EM04) with empirical evidence from a different discipline.

### MI2 (chunking) → EC08 and EM06 (potential)

**Evidence grade**: B (structural with potential EM status)

Chunking is the psychological observation of EC08 (compression factoring). Recoding into larger chunks is extracting shared structure. One step: "chunk" must be identified with "factored node in K." Well-motivated (both denote compressed representations of multiple items), but chunking is a behavioral process while factoring is a formal property of K.

**Potential EM**: MI2 + NV01 + EM06 may warrant a named emergent node: Miller's chunking is empirical confirmation that cognitive systems under C_n constraint spontaneously construct the graph-structured compressed representations that NV01 proves necessary. This is not currently a named node in the graph. The custody recommendation: evaluate whether to add this as an emergent intersection or leave it as structural grounding.

---

## REF-Cowan2001

**Citation status**: Verified (Behavioral and Brain Sciences, 2001; empirical review)

### CO1 (focus of attention: ~4 chunks) → EC01 refined

**Evidence grade**: B (empirical refinement of Miller)

Cowan's refinement of C_n to approximately 4 chunks (focus of attention) rather than 7±2 is an important clarification for Definition 7. The custody question: which value is the binding C_n for UB? The answer depends on whether C_n is the focus-of-attention capacity or the activated-LTM capacity.

For UB's definition of inference at any instant: the focus of attention (Cowan's 4) is the operative C_n. Items in activated LTM are available for retrieval but not currently integrated into the forward inference step. Cowan's distinction between activated LTM and the focus of attention maps cleanly onto UB's distinction between K (stored) and the active inference step operating on K at any moment.

Grade B because this is a refinement of an empirical measurement, not a formal derivation. The "correct" value of C_n for human cognition remains contested; the UB framework's C_n is derived formally and is not bound to either value.

### CO2 (embedded process model) → EC09

**Evidence grade**: B (structural)

CO2 maps to the encoding hierarchy: activated LTM is L1 (available); focus of attention is L2-L3 (active). One step: "activated long-term memory" and "focus of attention" must be identified with specific encoding levels. The identification is plausible but architectural — Cowan's model is a cognitive architecture; the encoding hierarchy is a formal structure derived from C_n constraints. Grade B.

### CO3 (slot structure) → NV01 (structural)

**Evidence grade**: B (structural)

Slot structure (4 slots, each usable regardless of content size) maps to graph node structure. One step: "memory slot" must be identified with "K-node." The per-slot coverage improvement through compression is the same dynamic in both frameworks, but the mechanism differs: slots are memory addresses; K-nodes are formal propositions.

**Independence**: Strong. Cowan (2001) is cognitive neuroscience with no information-theoretic derivation.

---

## REF-Kahneman2011

**Citation status**: Verified (Farrar, Straus and Giroux, 2011; widely cited popular and academic)

### KH1 (dual process) → EC09

**Evidence grade**: B (structural)

Fast (System 1) / slow (System 2) maps to the L0 vs. higher-level encoding hierarchy. One step: "System 1/System 2" must be identified with "L0 vs. L2-L3 encoding levels." Well-motivated — both frameworks describe qualitatively distinct processing modes that differ in speed, cost, and type of content — but the two-system model is a psychological taxonomy while the encoding hierarchy is derived from C_n constraints. The taxonomy and the derivation converge on the same structure from different starting points.

### KH3 (finite attention) → C_n

**Evidence grade**: B (structural)

Attention depletion provides psychological grounding for C_n as a real constraint. One step: "attentional resource depletion" must be identified with "bounded context." Well-motivated (both describe the same constraint operating at different levels of description), but Kahneman's attentional resource is a psychological construct while C_n is a formal information-theoretic bound.

### KH4 (heuristics and biases) → D13 provenance requirement

**Evidence grade**: B (structural, significant)

KH4 (fast processing treats easily-processed patterns as true) corresponds to propositions retained in K with insufficient provenance. The provenance requirement (D13) is, in part, the formal mechanism that prevents the KH4 failure mode.

One step: "System 1 heuristic" must be identified with "K-proposition with low C_1(p)." Well-motivated — both describe the situation where a representation is retained/applied without sufficient evidence — but D13's provenance triple is a formal criterion while System 1 heuristics operate through associative fluency. The identification is useful and well-grounded; it is not definitional.

**Significance**: KH4 provides a rich behavioral literature on the failure modes of insufficient provenance enforcement. The bias literature is indirect empirical evidence for why D13 is necessary: cognitive systems that don't enforce provenance exhibit systematic errors. The UB framework's provenance requirement has psychological motivation from Kahneman's research program.

**Independence**: Strong. Kahneman (2011) is behavioral economics/cognitive psychology with no formal derivation.

---

## REF-Baars1988

**Citation status**: Verified (Cambridge University Press, 1988; foundational GWT citation)

### GWT2 (workspace capacity) → C_n

**Evidence grade**: B (structural)

The limited capacity global workspace maps to C_n: the bounded context window is the workspace capacity. One step: "global workspace" must be identified with "active K." Well-motivated (both are the shared integration resource that all processes draw on), but the workspace is a psychological construct while C_n is a formal information bound. Grade B.

### GWT3 (modularity + escalation) → L0-L3 hierarchy

**Evidence grade**: B (structural)

Specialized processors operating below the workspace threshold map to the encoding hierarchy. One step: "below workspace threshold" must be identified with "L0-L1 encoding levels." Well-motivated (both describe specialized processing below a bounded integration capacity), but GWT's architecture is motivated by consciousness research while the encoding hierarchy is derived from C_n.

### GWT4 (K as global broadcast content) → K/F coupling

**Evidence grade**: B (structural)

K as the globally-broadcast content that informs all of F maps to GWT4. One step: "global broadcast" must be identified with "K-propositions available to all F." Well-motivated and the correspondence is widely noted in the consciousness literature, but the broadcast mechanism is a cognitive architectural claim while K/F coupling is a formal relation.

**Overall Baars assessment**: GWT provides architectural grounding for C_n (GWT2), the encoding hierarchy (GWT3), and K/F coupling (GWT4) simultaneously. No grade-A results, but three B-grade structural correspondences from a single framework is the highest density of well-motivated mappings in the cognitive psychology cluster.

---

## REF-Dehaene2011

**Citation status**: Verified (DOI: 10.1016/j.neuron.2011.03.018)

### NGW3 (attention as gating) → A in Definition 6

**Evidence grade**: B+ (structural, near independence)

NGW3 provides the strongest neural evidence for A as a probability in the corpus. The key finding: ignition (workspace entry) is binary at the neural level (threshold crossing) but probabilistic at the system level (probability of threshold crossing is continuous in [0,1] as a function of attentional load, prior context, stimulus strength). This is exactly Definition 6's A: the probability that a gradient signal triggers M engagement.

One step: the "probability of ignition given stimulus" must be identified with "A" (the probability that the entity engages M on a gradient signal). Well-motivated — both describe the system-level probabilistic gate — and the neural mechanism is more specific than the UB abstraction. Grade B+ because the mapping requires one identification step (neural ignition probability → A) but the neural evidence is direct and specific.

**Why this is the most important finding in the cluster**: the probabilistic formulation of A in Definition 6 might appear to be a modeling choice. NGW3 establishes that it is the empirically correct formulation: the neural mechanism underlying attentional gating is stochastic at the system level. A is not an arbitrary parameter; it reflects the actual stochastic nature of neural workspace entry.

### NGW2 (threshold ignition) → C_1(p)

**Evidence grade**: B (structural)

The non-linear ignition threshold (below threshold: local; above threshold: global broadcast) maps to the C_1(p) confidence threshold for retention in K. One step: "ignition threshold" must be identified with "C_1(p)." Well-motivated (both describe a threshold that determines whether information is retained/globally available), but C_1(p) is a confidence threshold on proposition evidence while ignition is a physical threshold on neural activation. Grade B.

### NGW4 (predictive coding integration) → FEP bridge

**Evidence grade**: B (structural, cross-cluster significance)

NGW4 (neural workspace compatible with predictive coding; conscious access when prediction error exceeds threshold) bridges GWT with the FEP cluster. One step: "prediction error" (FEP) must be identified with "U(w,K)" (UB). This is the same mapping made in the FEP cluster documents. NGW4 is the architectural confirmation that the two frameworks (GWT and FEP) are compatible, which indirectly supports the UB interpretation of both.

**Independence**: Strong. Dehaene and Changeux (2011) are neuroscientists. No information-theoretic derivation of A.

---

## Overall Cluster Assessment

| Source | Best grade | Key claim | Independence |
|---|---|---|---|
| Miller 1956 | A (MI3/EM04) | Empirical confirmation C_n = Shannon channel capacity | Strong |
| Cowan 2001 | B (CO1/EC01 refined) | Focus of attention (~4) = operative C_n | Strong |
| Kahneman 2011 | B (KH1-KH4) | Dual process = encoding hierarchy; KH4 = D13 motivation | Strong |
| Baars 1988 | B (GWT2-GWT4) | Workspace = C_n; modular hierarchy; K as broadcast | Strong |
| Dehaene-Changeux 2011 | B+ (NGW3/A) | Neural evidence A is probabilistic gate | Strong |

**Priority actions**:
1. Add MI3 (Miller 1956) as an independent empirical confirmation of EM04 (C_n = Shannon channel capacity). Update C_1(EM04).
2. Evaluate whether MI2 + NV01 + EM06 warrants a named emergent node (spontaneous chunking as empirical confirmation of graph-structured compression under C_n).
3. Update C_n value documentation: note both MI1 (7±2, activated LTM) and CO1 (~4, focus of attention), specify which is operative for Definition 7.
4. Document NGW3 as the neural evidence for A being probabilistic rather than a rate or binary flag. This upgrades the empirical grounding for Definition 6.
5. Document KH4 as psychological motivation for D13's provenance requirement: the bias literature provides a behavioral evidence base for why K requires provenance-verified propositions.
6. Document the cross-cluster bridge: NGW4 (Dehaene) connects this cluster to the FEP/predictive processing cluster via the prediction-error/U(w,K) identification.
