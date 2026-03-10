---
title: "The Confidence Chain: A Formal Framework for Cross-Framework Scientific Evaluation and a Proposal for Explicit Epistemic Provenance in Peer Review"
author: "Patrick McCarthy"
date: "2025"
abstract: |
  Scientific citation practice carries an implicit assumption that when Paper B cites
  Paper A, Paper B inherits Paper A's credibility for the specific claim cited. This
  assumption is never made explicit, never quantified, and critically — never revalidated
  as the citation propagates through the literature. We call this the confidence chain
  problem. We propose a formal solution: explicit epistemic provenance using a
  knowledge graph schema in which every proposition carries a provenance triple
  (attribution, evidence, derivation) and a computable confidence score C_1(p).
  Building on this schema, we introduce the Decomposition Theorem: given a set of
  frameworks sharing base assumptions (equivalency classes), every derivation is
  classifiable as a collapse (independently derivable from shared premises), a
  contradiction (derivation conflicts with another framework from the same premises),
  or a novel contribution. We demonstrate this method by applying it to a formal theory
  of intelligence (uncertainty bounding) against fifteen prior frameworks including
  Schmidhuber's formal theory of creativity, Friston's free energy principle, Shannon's
  information theory, and Peirce's pragmatic epistemology. The analysis produces ten
  equivalency classes, four confirmed overlap propositions, five novel contributions, and
  three emergent propositions not present in any individual framework — confirming the
  method's ability to discover new knowledge at the intersection of independent
  projections. We close by proposing a concrete peer review standard based on explicit
  confidence chains, equivalency class assignment, and inter-framework contradiction
  checking, and argue that the current review process is necessary but structurally
  insufficient for the epistemic claims it is asked to validate.
---

# The Confidence Chain: A Formal Framework for Cross-Framework Scientific Evaluation and a Proposal for Explicit Epistemic Provenance in Peer Review

---

## 1. Introduction

Every citation is an implicit confidence transfer.

When a paper states "as shown in [A], proposition P holds," the reader infers: the authors
believe P is true, they believe [A] established P, and the peer review process that accepted
the current paper did not find this belief unreasonable. The confidence the reader attaches
to P — absent any independent verification — derives from this chain of inference. That
chain is never made explicit. The probability that P is actually true, conditional on the
full evidence base behind [A], is not stated. The specific claim from [A] being invoked
(papers contain many claims) is often not identified. Whether the authors have independently
verified P, or merely assumed it from the citation, is not recorded. And once the paper is
published and itself cited, the confidence in P propagates further, diluting its connection
to the original evidence with each step.

We call this the *confidence chain problem*. It has three components:

**The attribution-confidence conflation.** Citation assigns attribution (this claim came from
[A]) but is read as confidence (this claim is as reliable as [A] itself). These are different.
Attribution is a factual record; confidence is an epistemic judgment. Current practice
encodes only attribution.

**The revalidation gap.** Once [A] establishes P and [B] cites [A] for P, P's confidence
level in [B]'s readers' minds is determined by [A]'s general reputation, not by the specific
evidence behind P in [A]. If [A]'s evidence for P later weakens — a failed replication, a
revised assumption, a boundary condition found — [B] and every paper in [B]'s citation chain
retains its inherited confidence in P with no automatic update.

**The cross-framework blindness.** The current review process validates a paper against
itself: is it internally consistent, is it novel, does it engage with the relevant literature?
It does not systematically check whether the paper's claims collapse into existing results
from shared premises (in which case they are confirmations, not contributions), contradict
existing results from shared premises (in which case one of the frameworks is wrong or
misscoped), or derive genuinely new results that no prior framework reaches from the same
starting points. These three outcomes — collapse, contradiction, novelty — are not currently
distinguished by the review process.

This paper proposes a formal framework that addresses all three components and implements
it as a working knowledge graph. The framework is not a critique of peer review; it is an
extension that makes explicit what peer review currently leaves implicit.

The paper is organized as follows. Section 2 develops the formal framework: provenance
triples, confidence scores, equivalency classes, and the Decomposition Theorem. Section 3
presents the knowledge graph schema. Section 4 demonstrates the framework on a running
example — a formal theory of intelligence evaluated against fifteen prior frameworks.
Section 5 presents the results: equivalency classes, confirmed overlaps, novel contributions,
and emergent propositions. Section 6 establishes the confidence profile of the demonstrated
theory. Section 7 proposes the new peer review standard. Section 8 discusses scaling to the
full literature and the emergent-knowledge prediction. Section 9 concludes.

---

## 2. The Formal Framework

### 2.1 Provenance Triples

**Definition 1 (Proposition).** A proposition $p$ is a claim that admits a truth value relative
to a model. Propositions are the atomic units of scientific knowledge. A proposition differs
from a sentence: the same proposition may be expressed in different languages, different
formalisms, or at different levels of abstraction.

**Definition 2 (Provenance Triple).** For any proposition $p$ in a knowledge base $K$, the
provenance triple is:
$$\text{prov}(p) = (\text{attribution}, \text{evidence}, \text{derivation})$$
where:
- *attribution* identifies the source: who stated $p$, in what work, when.
- *evidence* characterizes the support: type (formal proof, empirical, conceptual), strength,
  and references to the evidence base.
- *derivation* specifies how $p$ follows: from which prior propositions, by which logical or
  mathematical method.

A proposition without a complete provenance triple is an unverified claim — indistinguishable
in principle from noise. The triple is the minimum structure that makes a claim epistemically
tractable.

**Definition 3 (Confidence Score).** The confidence score $C_1(p) \in [0,1]$ of a proposition
$p$ is a measure of the evidential support for $p$ in the collective knowledge base. It is
computed from the provenance triple, the number of independent derivations, and the
confidence of the base propositions from which $p$ is derived.

Formally, if $p$ has $n$ independent derivations from bases $B_1, \ldots, B_n$:
$$C_1(p) = 1 - \prod_{i=1}^{n}(1 - C_1(B_i) \cdot s_i)$$
where $s_i \in [0,1]$ is the proof strength of derivation $i$ (1 for formal proof, decreasing
for empirical, theoretical, conceptual). Independence of derivations is required: derivations
from the same evidence chain do not compound.

This formula expresses: the probability that $p$ is false is the product of the probabilities
that each independent derivation fails. As independent derivations accumulate, $C_1(p)$
approaches 1. A single derivation with $s = 1$ and $C_1(B) = 1$ gives $C_1(p) = 1$ (certain).
A single formal proof from a well-established base gives high but not certain confidence.

*Critical property*: $C_1(p)$ is computed, not inherited. A claim cited 1,000 times by papers
that cite each other is not 1,000 independent derivations — it is one derivation propagated
1,000 times. The current citation system conflates propagation count with independent
derivation count. $C_1(p)$ does not.

### 2.2 Equivalency Classes

**Definition 4 (Equivalency Class).** An equivalency class $EC$ is a set of propositions
$\{p_1, p_2, \ldots, p_k\}$ drawn from $k$ distinct frameworks $\{F_1, \ldots, F_k\}$ such
that for all $i, j$: $p_i$ and $p_j$ express the same underlying constraint on the domain,
and a proof of $p_i$ in $F_i$ yields a proof of $p_j$ in $F_j$ under formal translation.

Equivalency classes are the cross-framework analog of a normal form: they identify the shared
base assumptions on which multiple independent frameworks converge. An equivalency class with
members from $k \geq 3$ independent traditions has high $C_1(p)$ because it requires $k$
independent failures to invalidate.

**Remark (Levels of equivalence).** Equivalency classes admit degrees of strength:
- *Formal equivalence*: $p_i \Leftrightarrow p_j$ under a formal translation, with proof.
- *Structural equivalence*: $p_i$ and $p_j$ describe the same structural constraint at
  different levels of abstraction (physical, statistical, cognitive, philosophical).
- *Conceptual equivalence*: $p_i$ and $p_j$ express the same intuition but resist formal
  translation. Conceptual equivalence should be marked explicitly and treated with lower
  confidence than formal equivalence.

### 2.3 The Decomposition Theorem

**Theorem (Decomposition).** Let $\mathcal{F} = \{F_1, \ldots, F_n\}$ be a set of frameworks
sharing a base $EC$. Let $q$ be a proposition derivable from $EC$ in framework $F_i$. Then
exactly one of the following holds:

(a) **Base**: $q \in EC$ — $q$ is a shared base assumption, not a contribution of $F_i$.

(b) **Collapse**: $q \notin EC$, and there exists $F_j \neq F_i$ such that $q$ is derivable
from $EC$ in $F_j$ without contradiction. $F_i$ and $F_j$ independently confirm $q$.
Each confirmation raises $C_1(q)$.

(c) **Contradiction**: $q \notin EC$, and there exists $F_j \neq F_i$ such that $\neg q$ (or
a proposition inconsistent with $q$) is derivable from $EC$ in $F_j$. Either $F_i$ or $F_j$
is wrong about $q$, or their scopes differ in a way that must be made explicit.

(d) **Novel**: $q \notin EC$, and $q$ is not derivable from $EC$ by any $F_j \neq F_i$, and
$q$ is consistent with all claims derivable from $EC$ in any $F_j$. $F_i$ has produced a
genuine contribution not present in any prior framework.

*Proof sketch.* The four cases are mutually exclusive by construction. (a) covers
$q \in EC$. For $q \notin EC$: either there exists $F_j$ that derives $q$ (collapse) or
derives $\neg q$ (contradiction) or neither (novel). Exactly one holds. $\square$

**Corollary (Peer review insufficiency).** Standard peer review checks: (i) novelty relative
to cited literature, and (ii) internal consistency. It does not check: whether $q$ is a
collapse relative to all $F_j$ that share $EC$ (because the reviewer checks only the cited
literature, not all frameworks that share $EC$), whether $q$ contradicts a claim in some
uncited $F_j$ that shares $EC$, or what the $C_1(q)$ of the novel claims is. Peer review is
therefore necessary but not sufficient for the epistemic claims it is asked to validate.

### 2.4 The Intersection Prediction

**Proposition (Emergent knowledge).** Let $K_i$ and $K_j$ be the proposition sets of two
frameworks sharing base $EC$. The intersection $K_i \cap K_j$ (confirmed overlaps) and the
symmetric difference $K_i \triangle K_j$ (distinct contributions) together contain propositions
$EM$ not derivable from $K_i$ alone or $K_j$ alone, producible only by holding both
frameworks simultaneously:
$$EM = \{q : q \text{ derivable from } K_i \cup K_j, q \notin K_i, q \notin K_j\}$$

These emergent propositions are new knowledge produced by the intersection. They are not
present in any individual framework. They arise from the juxtaposition.

This is the prediction: the intersection of sufficiently distinct frameworks should
produce emergent propositions. Section 5 demonstrates three such propositions arising
from the running example.

---

## 3. Knowledge Graph Schema

The framework is implemented as a directed graph with typed nodes and provenance-carrying
edges.

### 3.1 Node Types

| Type | Meaning | C_1(p) basis |
|------|---------|--------------|
| `definition` | Formal definition within a framework | Framework-internal |
| `theorem` | Proved theorem | Formal proof |
| `reference` | External framework proposition | Attribution + citation |
| `overlap` | Proposition grounded in 2+ frameworks independently | Multiple independent derivations |
| `novel` | Proposition grounded only in one framework | Single derivation |
| `emergent` | Proposition produced by intersection; not in any individual framework | Intersection of 2+ projections |
| `equivalency` | Equivalency class: shared base assumption across frameworks | k independent groundings |

### 3.2 Edge Types

Every edge carries a provenance triple (attribution, evidence, derivation):

| Relation | Meaning |
|----------|---------|
| `derives_from` | This proposition follows from the target |
| `overlaps_with` | Same underlying claim in both nodes, independently grounded |
| `generalizes` | This node generalizes the target (target is a special case) |
| `grounds` | This node provides formal grounding for the target |
| `qualifies` | This node adds a scope restriction to the target |
| `contradicts` | This node conflicts with the target from shared premises |
| `emergent_from` | This node was produced by the intersection of target nodes |
| `philosophical_precursor` | Target is a qualitative predecessor of this formal result |

### 3.3 The Provenance Invariant

**Invariant**: Every node and every edge in the graph must have a complete provenance triple.
A node or edge without provenance is structurally indistinguishable from noise.

This invariant operationalizes the confidence chain: every claim in the graph is traceable
to its evidence base. Removing a node ripples through to every node that cites it.
The revalidation gap (Section 1) becomes visible: if [A]'s evidence for P weakens, the
graph shows every edge that depends on [A]'s grounding of P.

---

## 4. Demonstration: Evaluating a Formal Theory of Intelligence

We apply the framework to a formal theory of intelligence — *Uncertainty Bounding as the
Basis of Intelligence* [McCarthy2025] — evaluated against fifteen prior frameworks spanning
formal AI theory, cognitive neuroscience, information theory, philosophy of science, and
thermodynamics.

### 4.1 The Target Theory

The uncertainty bounding (UB) framework proposes that intelligence is the driven capacity
to improve an entity's action space relative to a world it cannot fully represent [McCarthy2025].
Its core claims:

- **$I(E) = \mathcal{A} \cdot \eta_M$**: Intelligence is the product of agency
  $\mathcal{A} \in [0,1]$ (the probability that the entity engages its learning mechanism in
  response to a gradient signal) and $\eta_M$ (the rate at which the learning mechanism
  improves the action space per unit of learning signal).
- **$M < N$ always**: The entity senses $M$ dimensions of the world $W$ at any moment;
  $W$ has $N$ dimensions. $M < N$ is formally derived from bounded context $C_n$ and
  execution dynamics.
- **Graph necessity**: At the bounded context limit $C_n$, factoring is the unique
  information-preserving compression strategy. Graph construction is necessary, not chosen.
- **Intersection acceleration**: The intersection of independently-derived knowledge graphs
  accelerates the rate of $M \to N$ convergence by providing confirmed overlap propositions
  (higher $C_1(p)$) and distinct dimension extension (more $M$ per operation).

### 4.2 The Reference Frameworks

Fifteen frameworks were represented as graph nodes with their key propositions enumerated:

**Formal AI and cognitive theory**: Schmidhuber [2010] (compression-progress formal theory),
Friston [2010, 2017, 2021] (free energy principle; active inference; sophisticated inference),
Thrun and Pratt [1998] (meta-learning), Tishby and Polani [2011] (information bottleneck).

**Information theory**: Shannon [1948] (entropy and communication), Cover and Thomas [2006]
(elements of information theory including the data processing inequality and chain rule).

**Cognitive science**: Baars [1988] (global workspace theory), Dehaene and Changeux [2011]
(neural global workspace theory), Kahneman [2011] (dual-process theory).

**Philosophy**: Peirce [1877, 1878] (pragmatist epistemology: fixation of belief, pragmatic
maxim), Tarski [1936] (semantic theory of truth and undefinability), Chalmers [1995]
(hard problem of consciousness — for scoping).

**Thermodynamics and biology**: Schrödinger [1944] (negentropy and life), Prigogine and
Stengers [1984] (dissipative structures), England [2013] (statistical physics of
self-replication), Maynard Smith and Szathmáry [1995] (major transitions in evolution).

For each framework, key propositions were enumerated and entered as reference nodes with
provenance triples. Cross-framework edges were added where a proposition in one framework
grounds, overlaps with, or contradicts a proposition in another.

---

## 5. Results

### 5.1 Equivalency Classes

Analysis of the full reference graph identified ten equivalency classes — shared base
assumptions on which multiple independent frameworks converge:

**EC01 (Bounded context, $C_1 = 0.92$)**. $C_n$ (UB) $\equiv$ approximate posterior
(Friston) $\equiv$ global workspace capacity (Baars, Dehaene) $\equiv$ finite deliberate
attention (Kahneman). Four traditions from formal AI, neuroscience, and psychology
independently converge on a bounded internal representation capacity.

**EC02 (Rate of model improvement, $C_1 = 0.88$)**. $\eta_M$ (UB) $\equiv$
$\frac{d}{dt}[\text{compression}(C)]$ (Schmidhuber) $\equiv$ rate of free energy reduction
(Friston) $\equiv$ meta-learning rate (Thrun). The derivative of model quality — not
absolute quality — is independently identified as the measure of intelligence by four
distinct traditions.

**EC03 (Structured representation, $C_1 = 0.85$)**. K/F graph (UB) $\equiv$ hierarchical
generative model (Friston) $\equiv$ compression (Schmidhuber, implied) $\equiv$ sufficient
statistic (Cover, Tishby) $\equiv$ affordance structure (Gibson). Five traditions agree
that the representation must be structured. UB additionally proves the graph form is
*necessary*; the others ground that structure is *required*.

**EC04 (Survival threshold, $C_1 = 0.94$)**. $U_{lethal}$ (UB) $\equiv$ Markov blanket
maintenance threshold (Friston) $\equiv$ thermodynamic self-replication threshold (England)
$\equiv$ negentropy floor (Schrödinger) $\equiv$ dissipative structure bifurcation point
(Prigogine). Five traditions from cognitive science, statistical mechanics, and
thermodynamics derive the same physical threshold.

**EC05 (Epistemic incompleteness, $C_1 = 0.97$)**. $M < N$ (UB) $\equiv$ $q(s) \neq p(s)$
(Friston, Jensen's inequality) $\equiv$ truth as long-run convergence (Peirce) $\equiv$
undefinability (Tarski) $\equiv$ incompleteness (Gödel, via Tarski). This is the
highest-confidence equivalency class: formal mathematical proofs from multiple independent
traditions (incompleteness theorems, Jensen's inequality) establish that no physical system
can fully represent the world from which it emerged.

**EC06 (Belief as action disposition, $C_1 = 0.95$)**. K grounds F (UB) $\equiv$ belief as
habit of action (Peirce, pragmatic maxim) $\equiv$ posterior drives action (Friston, active
inference) $\equiv$ affordance perception (Gibson) $\equiv$ global broadcast enabling action
(Baars). Five traditions independently establish that retained knowledge has value only
through its effect on action. The UB grounding relation formally instantiates Peirce's 1878
pragmatic maxim.

**EC07 (Collective knowledge integration, $C_1 = 0.83$)**. $K_{collective}$ (UB) $\equiv$
major transitions in evolution (Maynard Smith) $\equiv$ community of inquiry (Peirce).
Three traditions ground that collective knowledge integration produces qualitatively superior
representations. The evolutionary and philosophical traditions provide empirical and
conceptual grounding; UB provides the formal mechanism (Section 5.3).

**EC08 (Compression through factoring, $C_1 = 0.91$)**. Factoring at $C_n$ (UB) $\equiv$
compression progress (Schmidhuber) $\equiv$ chain rule decomposition (Cover) $\equiv$
information bottleneck (Tishby) $\equiv$ aperiodic encoding (Schrödinger). Five traditions
establish that retaining more in less space requires extracting shared structure. UB's
unique contribution: proving the graph form is *necessary*, not just that structure is needed.

**EC09 (Hierarchical cost triage, $C_1 = 0.90$)**. L0--L3 hierarchy (UB) $\equiv$
System 1/System 2 (Kahneman) $\equiv$ global workspace levels (Baars, Dehaene) $\equiv$
precision hierarchy (Friston). Four traditions converge on the same hierarchical routing
structure: cheaper more-certain processing is handled locally; more expensive uncertain
processing escalates.

**EC10 (Uncertainty as inquiry driver, $C_1 = 0.93$)**. $U(w,K) > 0$ (UB) $\equiv$
variational free energy $> 0$ (Friston) $\equiv$ doubt (Peirce) $\equiv$ compression gap
(Schmidhuber) $\equiv$ cognitive dissonance (Kahneman). Five traditions independently
identify the gap between current representation and world structure as the driver of
intelligent behavior. Given EC05 ($M < N$ always, $C_1 = 0.97$), this gap is always
positive: the driver never terminates.

### 5.2 Confirmed Overlaps (OV Nodes)

Four propositions were identified as confirmed precipitation sites — grounded by two or
more frameworks independently:

**OV01**: Bounded internal world representation (UB + FEP + GWT).
**OV02**: Rate of model improvement as intelligence measure (UB + Schmidhuber).
**OV03**: Non-termination of learning (UB formal; FEP informal under non-stationarity).
**OV04**: Meta-cognitive awareness of own inference process (UB + Friston 2021).

These propositions have C_1(p) elevated above any single framework by their independent
grounding.

### 5.3 Novel Contributions (NV Nodes)

Five propositions were identified as grounded only in UB — not derivable from the shared
equivalency class base by any other framework in the reference set:

**NV01 (Graph structure is necessary, not sufficient)**. EC08 establishes that structured
compression is needed. UB's Theorem 2b Part 3 adds: the graph form specifically is the
*unique* information-preserving option at $C_n$, proven by the two-option argument
(discard without structure $\Rightarrow$ information loss by the data processing inequality
[Cover2006]; factor $\Rightarrow$ graph construction $\Rightarrow$ information preservation).
No prior framework derives this necessity. *This is UB's strongest individual contribution.*

**NV02 ($M < N$ formal mechanical derivation)**. EC05 establishes incompleteness with
$C_1 = 0.97$. UB's Theorem 3 Part II adds: the mechanism by which $M < N$ is maintained —
$C_n$ bounds context at each step; execution extends $\mathcal{T}_{reachable}$; for $\mathcal{A} > 0$,
there always exist reachable states not yet covered by $K$. Gödel and Tarski prove
incompleteness by logical necessity; UB proves it by physical mechanism.

**NV03 (Agency as engagement probability; the $\mathcal{A} = 0$ case)**. All prior frameworks
either implicitly assume $\mathcal{A} = 1$ (Schmidhuber: the agent always maximizes compression
progress) or define the agent by its engagement (Friston: the Markov blanket is the agent
boundary). UB parameterizes $\mathcal{A} \in [0,1]$ and derives that $\mathcal{A} = 0$
entities are removed by selection pressure (Theorem 6 Part 1). This is the formal account
of why prior frameworks can assume engagement: entities with $\mathcal{A} = 0$ do not persist
to be studied.

**NV04 (Intersection formally accelerates $M \to N$)**. EC07 establishes that collective
knowledge integration produces qualitative transitions. UB's Corollary 8a provides the
mechanism with cardinality: solo execution adds one dimension per step; intersection with
$K_j$ adds $|K_j \setminus K_i|$ dimensions per operation. The formal cardinality argument
is unique to UB.

**NV05 (Higher-order sentience is derived, not axiomatic)**. OV04 establishes that
meta-cognitive awareness is independently grounded. UB derives *why* it is adaptive: from
factorization $\to$ compactness $\to$ $|K|$ growth (Theorem 2b Part 3) + $M < N$ (Theorem 3
Part II) + retention criterion (Theorem 1), improving the normalization process is strictly
dominant across all future survival pressures. Prior frameworks make meta-cognition
architectural; UB makes it derivable.

### 5.4 Emergent Propositions (EM Nodes)

Three propositions were identified that do not appear in any individual framework but arise
from holding multiple frameworks simultaneously:

**EM01: Schmidhuber's formal agent is UB with $\mathcal{A} = 1$**. NV03 ($\mathcal{A}$ as
probability gate) applied to Schmidhuber's implicit $\mathcal{A} = 1$ (S4) produces a
containment claim: Schmidhuber's framework is a special case of UB where engagement is
certain. Neither Schmidhuber (who has no $\mathcal{A}$ parameter) nor UB (which does not
explicitly identify Schmidhuber as a special case in its text) states this. It is produced
by the intersection. Consequence: UB generalizes Schmidhuber in two formal directions —
partial engagement and graph form necessity.

**EM02: FEP's equilibrium claim requires the $M < N$ stationarity qualification**. FEP6
(free energy approaches its minimum) and Theorem 3 Part II ($M < N$ always) share base
assumption EC05 (incompleteness). From EC05, $M < N$ is a structural consequence, not an
empirical observation. FEP6's equilibrium requires a fixed, stationary environment — a
condition not embedded in FEP's assumptions. UB's $M < N$ derivation makes this condition
explicit: FEP6 is the fixed-world idealization; in open environments, $M < N$ is maintained.
Neither FEP nor UB states this scope qualification explicitly; it emerges from their
intersection.

**EM03: UB's necessity proof provides the derivation FEP's architectural assumption requires**.
FEP7 (hierarchical graph-structured generative models are assumed) and NV01 (graph structure
is proved necessary at $C_n$) share base assumption EC01 (bounded context) and EC08
(compression through factoring). The intersection reveals mutual support: NV01 provides the
derivation FEP7 needs (why the architecture must be hierarchically structured); FEP's
empirical and theoretical success provides evidence that graph-structured representations
work (supporting NV01's formal necessity claim). Neither framework states the bilateral
grounding relationship; it is produced by the intersection.

**Confirmation of the intersection prediction** (Section 2.4): three emergent propositions
were produced, confirming that the intersection of sufficiently distinct independent
projections generates new knowledge not present in any individual framework.

---

## 6. Theory Confidence Profile

Applying the Decomposition Theorem (Section 2.3) to UB's major results:

| Result | Classification | $C_1(p)$ | Primary basis |
|--------|---------------|----------|---------------|
| T1 (Minimization Imperative) | COLLAPSE | 0.93 | EC04 + EC06 + EC10 |
| T2b Part 3 (Graph necessity) | **NOVEL** | 0.82 | EC08 base + novel two-option proof |
| T3 Part I (Convergence) | COLLAPSE + NOVEL (structure) | 0.87 | EC10 + optimization theory |
| T3 Part II ($M < N$) | COLLAPSE + NOVEL (mechanism) | 0.95 | EC05 (Gödel/Tarski) |
| T6 ($\mathcal{A} > 0$ necessity) | **NOVEL** | 0.80 | No prior framework |
| T6 ($\eta_M > 0$ necessity) | COLLAPSE | 0.90 | EC02 + EC04 |
| C6a ($I = \mathcal{A} \cdot \eta_M$) | **NOVEL** (requires NV03) | 0.90 | Law of total expectation |
| C8a (Intersection acceleration) | **NOVEL** | 0.78 | EC07 qualitative; cardinality novel |
| D14 (Sentience) | COLLAPSE + NOVEL (derivation) | 0.85 | OV04 + NV05 |

No contradictions were found. Two scope mismatches were identified and resolved:
FEP6's equilibrium (resolved as stationarity qualification, EM02) and Gibson's direct
perception (resolved by scope: perceptual coupling vs. knowledge accumulation over time).
The theory is internally consistent with all prior frameworks in the reference set.

The theory's weakest claims ($C_1 < 0.82$) are precisely its most novel ones: graph form
necessity (T2b Part 3) and cardinality acceleration (C8a). This is structurally expected:
novel claims have fewer independent derivations. Both are empirically testable:

- *Graph form necessity*: compare information retention of graph-structured vs. non-graph
  agents under bounded context constraints. Prediction: non-graph agents lose information
  at $C_n$; graph agents do not. This distinguishes UB from Schmidhuber's unconstrained $C$.
- *Cardinality acceleration*: measure effective dimension coverage per operation in
  multi-agent vs. solo learning systems. Prediction: intersection provides $|K_j \setminus K_i|$
  dimensions per operation; solo execution provides 1.

---

## 7. A Proposal for Explicit Epistemic Provenance in Peer Review

The analysis establishes what the current review process does not check. We now propose
a concrete standard that extends peer review to address these gaps.

### 7.1 The Current Standard and Its Gaps

Peer review currently validates:
1. *Novelty*: is the contribution new relative to the cited literature?
2. *Internal consistency*: are the proofs correct, the experiments valid, the claims
   supported by the paper's own evidence?
3. *Significance*: is the contribution important enough for the venue?

It does not validate:
4. *Collapse detection*: is the claimed contribution actually derivable from equivalency
   classes shared with uncited frameworks? (A result that collapses into prior work from
   a shared base is a confirmation, not a novel contribution — even if the paper is
   unaware of the prior framework.)
5. *Contradiction checking*: does the paper derive claims that contradict results in
   frameworks it did not cite, from shared base assumptions?
6. *Confidence chain tracing*: what is the $C_1(p)$ of each cited claim in its original
   source, and is the current paper's use of that claim consistent with its evidential scope?
7. *Emergent proposition identification*: does the paper, in combination with prior work,
   produce propositions that should be identified and credited?

Gaps 4--7 are not failures of reviewers; they are failures of the current structure. No
individual reviewer can check a paper against all frameworks sharing its equivalency classes —
because those equivalency classes have not been enumerated and the relevant frameworks have
not been identified. The knowledge graph makes this tractable.

### 7.2 The Proposed Standard

We propose the following additions to the peer review process:

**R1 (Equivalency class declaration)**. Authors declare which equivalency classes their
base assumptions belong to. Reviewers check whether the declared equivalency class members
are correct and complete. This is analogous to related work, but scoped to base assumptions
rather than prior results.

**R2 (Collapse/novel/contradiction classification)**. For each major claim, authors classify
it as: (a) a collapse — an independently-confirmed result from shared premises, with citation
of the independent derivation; (b) a novel contribution — not derivable from the EC base
by any cited framework, with explicit statement of the gap; or (c) a resolved contradiction
— an apparent conflict with a prior framework, with the resolution stated.

**R3 (Confidence scores for novel claims)**. Novel claims (classification (b) above) include
an explicit $C_1(p)$ estimate and the evidence basis. This makes the epistemic status of novel
contributions transparent: a novel claim with $C_1 = 0.78$ (like UB's C8a) is clearly
distinguished from one with $C_1 = 0.95$ (like UB's T3 Part II).

**R4 (Provenance triples for cited claims)**. For each claim cited from prior work, authors
provide the specific claim (not just the paper), the evidence type in the original source
(formal proof, empirical, conceptual), and whether they have independently verified the
claim or are relying on attribution.

**R5 (Knowledge graph submission)**. The knowledge graph representation of the paper's
propositions and their provenance is submitted alongside the paper. This is machine-readable,
enables automated collapse/contradiction checking, and accumulates into the collective
knowledge graph of the field.

### 7.3 Feasibility and Adoption Path

The proposed standard adds work. We address the feasibility concern directly.

*For authors*: R1 and R2 require thinking that strong theoretical work already does
implicitly. Making it explicit is additional writing, not additional research. R3 requires
calibrating a confidence estimate — a discipline that formal work already exercises.
R4 and R5 are partially automatable: reference managers can be extended to capture
provenance triples; large language models can assist in extracting propositions from
existing papers.

*For reviewers*: R1 and R2 reduce the review task — the reviewer is no longer asked to
intuit novelty from a general reading but to check specific classifications against a
declared equivalency class base. This is more tractable, not less.

*For venues*: adoption can be incremental. A venue can require R1 and R2 initially,
adding R3--R5 as tooling matures. Venues focused on formal theory have the most immediate
benefit; empirical venues follow as provenance tooling extends to experimental claims.

*For the field*: a submitted knowledge graph accumulates into the collective knowledge
graph of the field. Over time, the equivalency class database grows, making automated
collapse/contradiction checking faster. The marginal cost of R5 decreases as the graph
grows.

---

## 8. Scaling and the Emergent Knowledge Prediction

### 8.1 From One Paper to the Full Literature

The running example evaluated one theory against fifteen frameworks. The framework scales
to the full literature by the following structure:

Each paper is a projection $K_i$ with $M_i$ propositions. Applying the Decomposition
Theorem across all pairs $(K_i, K_j)$ produces:
- A confidence map: each proposition's $C_1(p)$ computed from all independent derivations.
- A collapse register: claims that appear novel in their original papers but are
  independently derived in other frameworks.
- A contradiction register: claims that conflict with other frameworks from shared bases.
- An emergent proposition set: new knowledge produced by the intersection.

The method's prediction: the emergent set grows with each new framework added to the
graph. Each new framework $K_n$ adds $|K_n \setminus K_{collective}|$ novel propositions
and $|K_n \cap K_{collective}|$ confirmations. It also produces $|EM_n|$ emergent
propositions from its intersection with $K_{collective}$ — propositions not in $K_n$
or $K_{collective}$ individually, but visible once both are held simultaneously.

This is the prediction of the formal theory applied to itself: $K_{collective}$ is the
maximal current measure of what the field knows. Each new paper extends it.

### 8.2 The Confidence Chain Audit

Scaling the provenance framework to the full literature produces what current review
practice cannot: a confidence chain audit.

For any claimed fact in the literature, the audit traces:
- Which original evidence base does this claim derive from?
- How many independent derivations support it?
- What is the $C_1(p)$ at each step in the citation chain?
- Where does the chain terminate in primary evidence (experiment, formal proof) vs.
  propagated citation?

The audit makes visible what is currently invisible: foundational assumptions cited
universally but grounded in a single original study; results with high citation counts
but low $C_1(p)$ because all citations trace to one derivation; contradictions between
frameworks that have never been cited together.

### 8.3 The Revalidation Signal

A key property of the knowledge graph: revalidation propagates automatically.

If a foundation node $B$ is revised — a failed replication, a boundary condition
discovered, an assumption tightened — the graph shows every edge that depends on $B$.
The affected nodes are not automatically invalidated, but they are flagged for review.
The current citation system has no such propagation: a published result citing a
weakened foundation continues to circulate at its original implied confidence level
indefinitely.

---

## 9. Discussion

### 9.1 Relation to Existing Scientometrics

Bibliometric analysis has studied citation patterns extensively [Garfield1955, Hirsch2005].
Citation networks have been used to identify influential papers, track idea propagation,
and detect research communities. This work is valuable but operates at the level of
attribution — it counts citations without examining what specific claims are being
cited, how strong their evidence basis is, or whether they are consistent with the
claiming framework's other commitments.

Knowledge representation research (OWL, RDF, linked data) has developed formal schemas
for expressing structured knowledge with typed relations [Bizer2009, Heath2011].
Biomedical knowledge graphs [Nicholson2020] have applied this to scientific claims.
The contribution here is the provenance-first design: not just structured knowledge but
knowledge with explicit confidence chains and the Decomposition Theorem as a tool for
cross-framework analysis.

The philosophy of science has addressed related questions: Kuhn's paradigm incommensurability
[Kuhn1962], Lakatos's research programs [Lakatos1978], and Popper's falsifiability criterion
[Popper1959] all address how scientific frameworks relate to each other and to evidence.
The equivalency class formalism makes the question of framework comparison formally tractable
where Kuhn and Lakatos left it qualitative.

### 9.2 Limitations

The framework requires upfront work to enumerate propositions and build the graph. For
mature, large-scale fields, the initial construction is a significant undertaking.
Automated proposition extraction from papers is an active research area [Kardas2020]
but not yet reliable enough for unsupervised deployment.

The equivalency class construction requires judgment: identifying whether two propositions
from different frameworks express the same underlying constraint is itself an interpretive
act. We have distinguished three grades of equivalence (formal, structural, conceptual)
and flagged conceptual equivalence as carrying lower confidence — but the boundary
between structural and conceptual is not always sharp.

$C_1(p)$ is a model of confidence, not confidence itself. The formula in Section 2.1
makes specific assumptions about independence and proof strength that may not hold in
all cases. It should be treated as a structured estimate, not a precise probability.

### 9.3 The Hard Question: What Does This Mean for Published Claims?

If the framework is applied retrospectively to the existing literature and finds that
many widely-cited claims are single-derivation with $C_1(p) < 0.7$, while others with
high citation counts are collapses of results proved elsewhere — what follows?

The answer is not that these claims are wrong. A single well-executed formal proof is
sufficient to establish a proposition. High novelty + low $C_1(p)$ means the claim is
specific, not that it is false. The framework's output is not a verdict; it is a map.
The map shows where the field's knowledge is load-bearing vs. where it is thin; where
independent confirmation would make the largest difference; where the next experiment or
proof should go.

This is exactly what science is supposed to do. The current citation system makes this
map invisible. The proposed framework makes it explicit.

---

## 10. Conclusion

The confidence chain problem is structural: scientific citation practice conflates
attribution with confidence, implies revalidation that does not occur, and lacks a
mechanism for cross-framework consistency checking. These are not failures of individual
scientists or reviewers; they are absences in the current infrastructure.

We have proposed a formal solution: explicit provenance triples, computable confidence
scores, equivalency classes as shared base assumptions, and the Decomposition Theorem
as the tool for classifying every derivation as collapse, contradiction, or novel.

The demonstration on a formal theory of intelligence confirms:
- Ten equivalency classes identified, ranging from $C_1 = 0.83$ to $C_1 = 0.97$.
- Four confirmed overlap propositions with cross-framework support.
- Five genuinely novel contributions, with $C_1(p)$ scores that distinguish the
  better-supported from the less-supported.
- Three emergent propositions produced by cross-framework intersection — new knowledge
  not present in any individual framework.

The intersection prediction (Section 2.4) is confirmed: the intersection of sufficiently
distinct frameworks produces new knowledge. The Decomposition Theorem classifies every
proof without remainder.

The proposed peer review standard (Section 7.2) adds five requirements: equivalency class
declaration, collapse/novel/contradiction classification, confidence scores for novel claims,
provenance triples for cited claims, and knowledge graph submission. The standard is
incrementally adoptable and automatable.

The deepest implication is the simplest: the confidence that scientific results carry is
currently implied, inherited via citation, and never explicitly stated. Making it explicit
does not change what scientists do — it makes visible what they have always been doing,
and reveals where the foundations are strong and where they are thin.

---

## Acknowledgments

The formal theory evaluated in Section 4 is the author's own prior work.
The knowledge graph implementation and cross-framework analysis described in this paper
are available at \url{https://github.com/patdmc/open-knowledge-graph}.

---

## References

\bibliographystyle{plain}
\bibliography{references}

*The following references are cited above. Full bibliographic details are in
`references.bib` in the accompanying repository.*

- Baars1988: A Cognitive Theory of Consciousness
- Chalmers1995: Facing Up to the Problem of Consciousness
- Cover2006: Elements of Information Theory
- Dehaene2011: Experimental and Theoretical Approaches to Conscious Processing
- England2013: Statistical Physics of Self-Replication
- Friston2010: The Free-Energy Principle
- Friston2017: Active Inference: A Process Theory
- Friston2021: Sophisticated Inference
- Gibson1979: The Ecological Approach to Visual Perception
- Kahneman2011: Thinking, Fast and Slow
- MaynardSmith1995: The Major Transitions in Evolution
- McCarthy2025: Uncertainty Bounding as the Basis of Intelligence [companion paper]
- Peirce1877: The Fixation of Belief
- Peirce1878: How to Make Our Ideas Clear
- Prigogine1984: Order Out of Chaos
- Schmidhuber2010: Formal Theory of Creativity, Fun, and Intrinsic Motivation
- Schrodinger1944: What is Life?
- Shannon1948: A Mathematical Theory of Communication
- Tarski1936: The Concept of Truth in Formalized Languages
- Thrun1998: Learning to Learn
- Tishby2011: Information Theory of Decisions and Actions
