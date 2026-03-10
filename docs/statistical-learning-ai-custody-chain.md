# Statistical Learning and AI: Custody Chain Evaluation

**Cluster**: Vapnik (1995), Hinton-Rumelhart (1986), Thrun-Pratt (1998), Vaswani et al. (2017), Schmidhuber (2010), Tishby-Polani (2011)

---

## Framework Role Classification

| Category | Nodes | Custody concern |
|---|---|---|
| **Cross-disciplinary equivalence** | VC2 + EM07 | Independent derivation of same bound |
| **Engineering instantiation** | TR4 + C_n | Most direct evidence C_n is real |
| **Emergent intersection** | EM01 (Schmidhuber = UB at A=1) | Scope boundary; requires precision |
| **Empirical confirmation** | BP2-BP3, TH1-TH2 | Confirm predictions; do not derive them |
| **Overlap with gaps documented** | S6-S8 (Schmidhuber), TB3 (Tishby) | Gap nodes; define UB's novel content |

---

## REF-Vapnik1995

**Citation status**: Verified (Springer, 1995; widely cited)

### VC2 (Generalization bound) as EM07 derivation

**Evidence grade**: B+ (structural, near equivalence)

The VC generalization bound and EM07 (action accuracy bounded by M/N) express the same structural constraint from different theoretical directions. In VC terms: generalization error is bounded by sqrt(h/l) where h is capacity and l is coverage. In UB terms: U_residual(f) ≥ U(w,K) * (N-M)/N where M is modeled dimensions and N is total dimensions. The h/l ratio and the M/N ratio play identical structural roles.

The translation step: "VC dimension" (hypothesis class shattering capacity) must be identified with "M" (modeled dimensional coverage). This requires one interpretive step: VC dimension measures the maximum set a hypothesis class can correctly classify, not the number of world dimensions modeled. The identification is well-motivated — both measure the capacity of the learning system relative to the space it operates in — but is not definitional.

Grade B+ (not A) because the two formalisms use different languages for capacity: statistical (hypothesis-class shattering) versus dimensional (world-coverage). The structural equivalence is clear; the formal identification requires one step.

**Independence**: Strong. Vapnik (1995) is statistical learning theory with no information-theoretic derivation of EM07. The overlap is not motivated by UB.

**Custody concern**: The VC2/EM07 equivalence should be explicitly documented in the framework's treatment of EM07. Currently EM07 has an information-theoretic derivation. The VC derivation is an independent confirmation from a different discipline, which strengthens C_1(EM07).

### VC3 (SRM) → EC08

**Evidence grade**: B (structural)

SRM minimizes empirical risk plus a complexity penalty. This is the statistical learning theory statement of EC08 (compression factoring): at C_n, retain the most compressed representation that preserves predictive accuracy. One step: SRM selects among hypothesis classes by complexity penalty; EC08 selects among knowledge structures by information content. The selection criterion is the same (minimize complexity subject to accuracy); the objects selected differ (hypothesis classes vs. knowledge graph structures).

### VC4 (PAC learning) → eta_M upper bound

**Evidence grade**: B (structural)

PAC learning gives a polynomial sample complexity bound on how fast learning can proceed. This bounds eta_M growth as a function of VC dimension and sample access. One step: "sample complexity" must be identified with "rate of M growth." Well-motivated but not definitional.

**Custody gap**: UB has the Solomonoff upper bound on eta_M (K(mu)/ln(2)) but not the PAC bound as a function of VC dimension. These are complementary bounds. The Solomonoff bound is tighter for the optimal computable learner; the PAC bound applies for specific hypothesis class sizes. Both should be cited.

**Independence**: Strong. Vapnik (1995) does not discuss information theory.

---

## REF-HintonRumelhart1986

**Citation status**: Verified (DOI: 10.1038/323533a0)

### BP1 (backpropagation) → EC02, EC10

**Evidence grade**: B (structural)

Backprop is the computational implementation of eta_M under gradient descent. The loss gradient is U(w,K). One step: the loss function (empirical prediction error) must be identified with U(w,K) (epistemic uncertainty about W). Well-motivated — both measure how far K is from perfectly modeling W — but the identification is not definitional: a neural network's loss function is computed on labeled samples, while U(w,K) is defined over the full distribution of W. The entities that backprop trains are bounded to their training distribution.

### BP2 (distributed representations) → EC08, NV01

**Evidence grade**: B (structural)

Distributed representations are the factored K that NV01 proves necessary. One step: "distributed pattern of activation" must be identified with "factored proposition in K." Well-motivated (both denote compressed shared-feature representations), but the neural implementation is not co-extensive with the formal K/F graph structure. The network's weights encode relational structure implicitly; K/F edges are explicit.

**Key contribution**: BP2 is empirical confirmation that gradient descent naturally converges on distributed (factored) representations when operating under capacity constraints. This confirms the EM06 prediction (gradient drives toward graph structure) from a different direction.

### BP3 (hidden layer hierarchy) → EC09

**Evidence grade**: B (structural)

Hidden layers spontaneously develop hierarchical representations. One step: "layer depth" must be identified with "encoding level." Well-motivated — the correspondence between layer depth and abstraction level is well-established in the deep learning literature — but layer depth and encoding level are not definitionally identical.

**Independence**: Strong. Rumelhart, Hinton, and Williams (1986) do not discuss bounded context or action-relevance.

---

## REF-Thrun1998

**Citation status**: Verified (Kluwer Academic Publishers, 1998)

### TH1 (meta-learning improvability) → eta_M

**Evidence grade**: B (structural)

TH1 establishes empirically that the learning rate is improvable, not just learning outcomes. This is empirical grounding for eta_M as a meaningful, measurable quantity. One step: "improvement of the learning algorithm" must be identified with "improvement of eta_M." Well-motivated — both denote second-order learning — but TH1 is empirical (demonstrated in experiments) while eta_M is a formal parameter.

**Citation role**: Thrun and Pratt are explicitly cited in Definition 7. This citation is well-grounded.

### TH2 (inductive bias transfer) → K/F structure

**Evidence grade**: B (structural)

The transferred inductive bias is the K/F structure. One step: "inductive bias" (prior constraints on the hypothesis space) must be identified with "K" (retained propositions that ground actions). The identification requires accepting that an inductive bias and a knowledge graph are the same kind of entity (both constrain what hypotheses are considered). Well-motivated but not definitional.

**Independence**: Strong. Thrun and Pratt do not derive K/F graph necessity.

---

## REF-VaswaniEtAl2017

**Citation status**: Verified (NeurIPS 2017; widely cited)

### TR4 (context window) → C_n

**Evidence grade**: A (definitional for the engineering instantiation of EC01)

The context window is C_n measured in tokens. This is not a one-step interpretation; it is a direct definitional equivalence. The Transformer does not approximate bounded context — it implements it exactly. The context window is the architectural parameter that realizes C_n: every token outside it is not in K for that forward pass.

This is the strongest engineering evidence for EC01 in the corpus. The Transformer context window is a design choice that operationalizes Definition 7's bounded context.

**Note**: the question of whether C_n is the right way to model the context window (rather than, say, a soft attention decay) is a modeling question. The Transformer uses hard cutoff; some alternatives use soft decay. UB's C_n is a hard bound. The Transformer TR4 is an exact instantiation of the hard-bound model.

### TR1 (self-attention as K/F graph) → EC01, EC08

**Evidence grade**: B+ (structural)

Self-attention scores at inference time are G(f,K) values — they measure how much each proposition (token) in K is relevant to each action (output position). One step: "attention weights" (learned dot-product similarity in embedding space) must be identified with "G(f,K)" (UB's relevance measure defined by action-relevance). Well-motivated and widely accepted in the interpretability literature, but attention weights do not directly equal G(f,K) without establishing that the embedding space encodes action-relevance rather than semantic similarity.

### TR2 (parallel processing) → NV04

**Evidence grade**: B (structural)

Parallel processing at all positions is the architectural realization of NV04 (intersection acceleration). One step: "processing all positions simultaneously" must be identified with "K_i intersecting simultaneously." Well-motivated — the speedup argument in NV04 and the O(1) parallel time in Transformers are structurally parallel — but NV04 is a formal result about knowledge intersection while TR2 is an architectural design choice.

**Unique contribution**: The O(n^2) attention cost as the quantitative overhead of full K/F coupling is not stated in UB. TR1 provides this specific scaling law.

**Independence**: Strong. Vaswani et al. (2017) do not discuss bounded rationality or knowledge graphs.

---

## REF-Schmidhuber2010

**Citation status**: Verified (DOI: 10.1109/TAMD.2010.2051167)

### S1 (compression-progress drive) → EM01

**Evidence grade**: A (definitional for EM01)

EM01 (Schmidhuber's framework is UB at A = 1) is a precise formal claim: S4 (implicit A = 1) is the special case of Definition 6's A in [0,1]. The mapping requires no interpretive step — the UB framework generalizes by adding the A parameter that Schmidhuber's framework leaves fixed at 1. EM01's grade is A because EM01 is defined as exactly this comparison.

**Independence of S1 from UB**: Very strong. Schmidhuber (2010) is motivated by compression theory and LSTM-based agents. No information-theoretic or selection-pressure derivation.

### S6-S8 (documented gaps) as novelty evidence

**Evidence grade**: A (definitional for NV02, NV03)

S6 (no M < N bound), S7 (no graph necessity), S8 (no A = 0 case) are documented absence of results. The gaps are not failures of Schmidhuber's theory — they are outside its scope. Their significance for UB: each gap marks a place where UB provides a formal result that the prior literature does not. The novelty nodes (NV02, NV03) are grounded in part by these documented absences.

**Custody concern**: the S6-S8 gap analysis should cite Schmidhuber explicitly. Claiming novelty without documenting the prior art boundary is a provenance gap.

---

## REF-Tishby2011

**Citation status**: Verified (chapter in Perception-Action Cycle, 2011)

### TB1, TB3 (information bottleneck) → Theorem 2b Part 3

**Evidence grade**: B (structural overlap, not derivation)

The information bottleneck approach and Theorem 2b Part 3 overlap in their conclusion (structured compression is necessary) but differ in their derivation. Tishby grounds the claim as a rate-distortion optimization: minimize I(past; representation) subject to maintaining I(representation; future reward). UB grounds it as a two-option argument: at C_n, either discard without structure (losing information, proved irreversible by DPI) or factor (preserving information). Both arrive at "factored compression is required."

The distinction matters: Tishby's result is "factored compression is optimal" (rate-distortion optimum). UB's result is "factored compression is necessary" (no alternative preserves information). These are formally different claims: optimality does not imply necessity, and necessity does not follow from optimality. Grade B.

**Independence**: Strong. Tishby and Polani do not prove graph necessity from the DPI.

---

## Overall Cluster Assessment

| Source | Best grade | Key claim | Independence |
|---|---|---|---|
| Vapnik 1995 | B+ (VC2/EM07) | Statistical derivation of EM07; sample complexity bound | Strong |
| Hinton-Rumelhart 1986 | B (BP1-BP3) | Empirical confirmation EC08, EC10; backprop = eta_M | Strong |
| Thrun-Pratt 1998 | B (TH1-TH2) | Empirical grounding eta_M improvability; cited in D7 | Strong |
| Vaswani et al. 2017 | A (TR4/C_n) | Engineering instantiation of C_n; context window = C_n | Strong |
| Schmidhuber 2010 | A (EM01) | Schmidhuber = UB at A=1; S6-S8 define UB novelty | Strong |
| Tishby-Polani 2011 | B (TB1/Thm 2b) | Overlap at compression necessity; UB adds structural necessity | Strong |

**Priority actions**:
1. Document VC2 + EM07 as independent cross-disciplinary confirmation; update C_1(EM07).
2. Cite the PAC bound (VC4) alongside the Solomonoff bound as complementary upper bounds on eta_M growth.
3. Document TR1's O(n^2) scaling as the quantitative overhead of full K/F coupling (not currently in UB).
4. State explicitly in the novelty analysis that S6-S8 gaps are the prior art boundary that NV02-NV03 cross.
5. Clarify the Tishby/Theorem 2b Part 3 distinction: optimality vs. necessity.
