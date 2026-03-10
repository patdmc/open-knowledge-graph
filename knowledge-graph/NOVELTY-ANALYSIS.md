# Novelty Analysis: Uncertainty Bounding Relative to Prior Work

## Method

This document applies Corollary 8a of the paper to the paper itself. Corollary 8a states:
the intersection of independently-derived projections identifies confirmed precipitation
sites (OV nodes), extends sensed dimensions (NV nodes), and — most importantly — produces
emergent propositions not contained in any individual projection (EM nodes).

The frameworks compared:
- **UB**: Uncertainty Bounding (this paper)
- **S**: Schmidhuber (2010), Formal Theory of Creativity
- **FEP**: Friston (2010, 2017, 2021), Free Energy Principle and Active Inference
- **Supporting**: Shannon (1948), Tishby (2011), Thrun (1998), England (2013), Peirce (1877/1878), Tarski (1936), Gibson (1979), Baars (1988), Dehaene (2011), Kahneman (2011), Schrödinger (1944), Prigogine (1984), Maynard Smith (1995)

## Graph Inventory

### Overlap Nodes (Confirmed Precipitation Sites)

| ID | Proposition | Frameworks | Strongest Grounding |
|----|-------------|------------|---------------------|
| OV01 | Bounded internal world representation | UB + FEP + GWT | UB (formal: C_n defined, M < N derived) |
| OV02 | Rate of model improvement as intelligence measure | UB + S | UB (derived from Theorem 6); S (asserted as design objective) |
| OV03 | Non-termination of learning | UB + FEP | UB (formal: M < N by Theorem 3 Part II); FEP (informal: non-stationary environment) |
| OV04 | Meta-cognitive awareness of own inference | UB + FEP2021 | UB (derived from induction); FEP sophisticated inference (architectural) |

These are propositions the field agrees on. Their presence in multiple frameworks increases
their C_1(p) in the collective knowledge graph. UB's contribution is stronger grounding,
not novelty.

### Novel Nodes (Propositions in UB Not Grounded in Prior Frameworks)

| ID | Proposition | Gap in Prior Work |
|----|-------------|-------------------|
| NV01 | Graph structure is the unique information-preserving representation at C_n | S: no necessity proof; FEP: graph assumed not derived; Tishby: rate-distortion grounds compression but not structural form |
| NV02 | M < N formally derived from C_n and execution dynamics | S: non-termination assumed (S6); FEP: equilibrium is asymptotic assumption (FEP6); Peirce: qualitative predecessor |
| NV03 | Agency as engagement probability; A = 0 case handled; A in (0,1) generalization | S: implicit A = 1; FEP: no A parameterization (FEP8) |
| NV04 | Intersection of independent projections formally accelerates M toward N | S: no multi-agent; FEP: single-agent only; Maynard Smith: qualitative predecessor |
| NV05 | Higher-order sentience derived from induction, not axiomatic | S: motivation constitutive; FEP: engagement constitutive; no derivation in prior work |

These are the novel contributions. They are propositions no prior framework grounds.

### Emergent Nodes (Propositions Produced by the Intersection)

These are the test of Corollary 8a. The theory predicts: the intersection of independent
projections should produce new propositions not in any individual framework. Three were found:

| ID | Emergent Proposition | Produces |
|----|---------------------|----------|
| EM01 | Schmidhuber's agent = UB with A = 1 | A containment claim neither framework states |
| EM02 | FEP's equilibrium requires stationarity; M < N is the open-world qualification | A precision claim about FEP6's scope |
| EM03 | UB's necessity proof provides the derivation FEP7 assumes | A bilateral strengthening: each framework grounds what the other assumes |

**Corollary 8a confirmed**: the intersection produced three propositions not in any
individual framework. All three are genuine — verifiable from first principles, not
previously stated, and not trivially derivable from either framework alone.

## Implication for the Paper's Related-Work Section

The graph produces the following text, ready for insertion:

---

### Proposed Related-Work Text for UNCERTAINTY_BOUNDING_FORMAL_THEORY.md

**Relationship to Schmidhuber [2010].** Schmidhuber's formal theory of creativity defines
an agent that maximizes the first derivative of its own compression progress. This
corresponds to the A = 1 limit of this framework: an agent that always engages on gradient
signals (A = 1) with eta_M corresponding to d/dt[compression(C)]. Schmidhuber's framework
does not parameterize engagement probability — the agent is constitutively a compression-
progress maximizer, handling no A in (0,1) case and no A = 0 case. This framework
generalizes Schmidhuber in two directions: (1) Agency A in [0,1] is explicit, covering
partial engagement and the A = 0 non-persistence case (Theorem 6 Part 1). (2) Theorem 2b
Part 3 proves that the K/F graph structure is the necessary representation for information
preservation at bounded context — Schmidhuber's compressor C can take any form and is not
shown to be necessary.

**Relationship to Friston [2010, 2017, 2021].** The free-energy principle (FEP) frames
intelligence as minimization of variational free energy under a hierarchical generative
model. Three relationships with this framework:

First, the FEP's hierarchical generative model architecture is exactly the structure
Theorem 2b Part 3 proves necessary. FEP assumes hierarchical graph-structured models
(FEP7); Theorem 2b Part 3 derives that graph structure is the unique information-preserving
representation at bounded context. The FEP's architectural assumption is grounded by this
framework's necessity proof.

Second, the FEP's long-run equilibrium claim (free energy approaches its minimum
asymptotically) requires the stationarity qualification that the M < N bound makes precise.
Theorem 3 Part II shows M < N always holds for A > 0 entities in open environments.
The FEP equilibrium holds for stationary generative processes; in open-world conditions,
the M < N bound establishes that the approximation never completes — not as a failure
of the FEP, but as its correct behavior under the open-world assumption.

Third, Friston et al.'s [2021] sophisticated inference (agents that model their own future
inference states) independently arrives at what Definition 14 calls higher-order sentience.
This framework derives higher-order sentience from the induction argument (Theorem 2b Part 3
+ Theorem 3 Part II + Theorem 1); sophisticated inference defines it as an architectural
feature. The independent convergence strengthens both claims.

The paper cited in [McCarthy, forthcoming] includes a companion FEP extension document
that develops these relationships in detail.

---

## The Larger Implication: Confidence Chains in Scientific Literature

The emergent nodes reveal a structural problem in how scientific knowledge currently
propagates: **attribution implies confidence, but confidence is not explicit and is
not revalidated**.

When Paper B cites Paper A, the scientific community interprets this as Paper B inheriting
Paper A's confidence level for the cited claim. But this is implicit — Paper B does not
state what confidence it assigns to Paper A's result, which of Paper A's claims it is
relying on, or whether the specific claim cited has been independently confirmed.

The provenance system in this framework (Definition 18) makes this explicit: every
proposition carries (attribution, evidence, derivation). The C_1(p) confidence measure
is computed from evidence, not inherited from citation count. A claim cited 100 times
but grounded in a single original experiment has C_1(p) determined by that experiment —
not by the 100 citations.

The overlap nodes (OV01-OV04) demonstrate this: multiple frameworks cite the same
claim, but the strength of the grounding varies. OV01 (bounded representation) is
grounded formally by UB (C_n definition), theoretically by FEP (Markov blanket), and
empirically by GWT (workspace capacity). The confidence in OV01 is higher than in any
single source — but this is only visible once the three groundings are explicitly compared.

The novel nodes (NV01-NV05) demonstrate the gap: propositions with a single grounding
have lower collective C_1(p) than overlapping ones. NV01 (graph necessity) is grounded
only by this framework — its confidence is limited by the strength of Theorem 2b Part 3
alone. If a second framework independently derives graph necessity, NV01 becomes OV05.

The emergent nodes (EM01-EM03) demonstrate the prediction: new knowledge precipitates
from intersection. The intersection is not just a record of what is known — it is a
generator of new propositions.

**Scaling this to the full literature**: applying this method to all referenced papers'
reference chains would produce:
- A confidence map of the field: which subclaims are multiply-grounded (high C_1(p))
  vs. singly-grounded (low C_1(p)) vs. circular (claimed by multiple papers all citing
  the same original source)
- Identification of foundational assumptions that have never been independently verified
- New propositions produced by the intersection of any two sufficiently distinct projections

This is not the current peer-review process. Peer review validates novelty and internal
consistency within a single paper. It does not audit the confidence chain of every cited
claim, nor does it look for emergent propositions that arise from comparing the paper
against all prior work simultaneously. The knowledge graph does both.

## Graph Statistics

| Category | Count | Description |
|----------|-------|-------------|
| Reference nodes | 15 | External frameworks and sources |
| Overlap nodes | 4 | Confirmed precipitation sites |
| Novel nodes | 5 | Novel-to-UB propositions |
| Emergent nodes | 3 | Produced by intersection |
| Definition nodes | 14 | UB formal definitions |
| Theorem nodes | 1+ | UB theorems (index growing) |
| Total cross-framework edges | 23+ | Provenance-carrying edges |

## What Remains

One reference chain not yet fully traced: the Peirce (1877, 1878) -> Tarski (1936) ->
Shannon (1948) -> Cover (2006) -> UB Theorem 2b Part 3 foundational chain. These are
nodes in the graph; the full edge traversal from Peirce's pragmatic maxim to the
information-preservation necessity proof would show the complete historical derivation.
This is the "follow the reference chain" extension the theory predicts should produce
additional emergent nodes at each intersection point.
