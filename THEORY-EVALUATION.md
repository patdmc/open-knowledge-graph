# Theory Evaluation: Uncertainty Bounding Against the Reference Graph

## The Classification Principle

Once equivalency classes (EC nodes) are established, every derivation in every framework
must fall into one of three categories:

**1. COLLAPSE** — The proof derives something already derivable from a shared base
assumption. The proposition is in an EC node; the proof re-establishes a known claim.
This is not a failure — it raises C_1(p) by providing an independent derivation.
But it is not a novel contribution.

**2. CONTRADICTION** — The proof derives a claim that conflicts with a claim in another
framework that shares the same base assumptions. If both frameworks accept EC01 (bounded
context), and one derives X and the other derives not-X, one of them is wrong, or the
scope is different. Contradictions must be resolved.

**3. NOVEL** — The proof derives something not derivable from the same base assumptions
by any prior framework. This is a genuine contribution. It expands the collective K.

---

## Base Assumption Map

The equivalency classes establish the shared base assumptions across frameworks:

| EC | Assumption | C_1(p) | Frameworks |
|----|-----------|--------|------------|
| EC01 | Bounded context | 0.92 | UB, FEP, GWT, Kahneman |
| EC04 | Survival threshold | 0.94 | UB, FEP, England, Schrödinger, Prigogine |
| EC05 | Epistemic incompleteness | 0.97 | UB, FEP, Tarski, Gödel, Peirce |
| EC06 | Belief as action disposition | 0.95 | UB, Peirce, FEP, Gibson, Baars |
| EC08 | Compression = factoring | 0.91 | UB, Schmidhuber, Cover, Tishby, Schrödinger |
| EC10 | Uncertainty as driver | 0.93 | UB, FEP, Peirce, Schmidhuber, Kahneman |

These are UB's foundation. Every theorem in UB derives from a subset of these.

---

## Theorem-by-Theorem Classification

### Theorem 1: Minimization Imperative
**Claim**: E retains p iff its survival contribution is not certainly zero.
**Base assumptions used**: EC04 (survival threshold), EC06 (belief as action), EC10 (uncertainty as driver)
**Classification**: COLLAPSE
**Evidence**: This is Peirce's pragmatic maxim (EC06) + EC04's survival selection. Any agent
that retains only action-relevant beliefs under selection pressure derives Theorem 1 from
these shared premises. FEP derives the same result (retain what reduces free energy; discard
when marginal free energy reduction = 0).
**C_1(p)**: 0.93 — from cross-framework confirmation
**Novel contribution**: The formal retention criterion (U contribution = 0 exactly, not
approximately) and its application to the K/F structure. The formalization is UB's contribution.

---

### Theorem 2b Part 3: Graph Structure Necessity
**Claim**: At C_n, factoring is the unique information-preserving option. Graph construction
is necessary, not chosen.
**Base assumptions used**: EC01 (bounded context), EC08 (compression = factoring), EC05 (incompleteness)
**Classification**: NOVEL
**Evidence**: All frameworks in EC08 ground "factoring is needed." None of them derive that
graph structure (specifically) is necessary. Tishby grounds compression necessity;
Cover/Thomas grounds the chain rule; Schmidhuber grounds compression progress. But none
prove the two-option argument: (A) discard without structure = information loss (by Cover
CT1 data processing inequality); (B) factor = graph construction = information preservation;
no third option. The necessity proof is unique to UB.
**C_1(p)**: 0.82 — strong base support (EC08 at 0.91) but the specific two-option necessity
argument has no independent grounding outside UB. C_1(p) = f(base, novel derivation) ~
0.91 * 0.90 = 0.82 (rough: base confidence * internal proof confidence)
**Status**: UB's strongest individual contribution. Highest novelty score. If this is wrong,
the graph structure requirement falls.

---

### Theorem 3 Part I: Convergence by Induction
**Claim**: Gradient descent converges for any M-dimensional parameterization of F.
**Base assumptions used**: EC10 (uncertainty as driver), EC02 (improvement rate), EC01 (bounded context)
**Classification**: COLLAPSE (convergence claim) + NOVEL (inductive structure)
**Evidence**: Convergence of gradient descent on bounded-below functions is EC02 (rate of
improvement) + standard optimization theory. The claim that U >= 0 and bounded below is
EC05/EC10. The *inductive structure* (by M, not functional analysis on infinite-dimensional
spaces) is novel. All prior frameworks assume gradient descent converges; UB derives it
inductively for M sensed dimensions at any moment, without assuming anything about N.
**C_1(p)**: 0.87 — convergence part is well-grounded (EC10 + optimization theory);
inductive structure adds novel precision
**Status**: The inductive proof is novel in structure (not in conclusion). Prior frameworks
get the same answer differently. UB's proof is cleaner because it matches the actual M < N
structure.

---

### Theorem 3 Part II: M < N Always
**Claim**: For A > 0 entities, M < N is maintained at every t.
**Base assumptions used**: EC01 (bounded context), EC05 (incompleteness), EC10 (driver)
**Classification**: COLLAPSE (conclusion) + NOVEL (formal derivation)
**Evidence**: EC05 (incompleteness) already establishes that the representation is always
partial. Tarski's undefinability theorem, Gödel's incompleteness, and FEP's variational gap
all ground the conclusion. The *formal derivation from C_n and execution dynamics* (not just
logical necessity) is novel: UB shows HOW M < N is maintained mechanically, not just that
it must be. Tarski proves it by diagonal argument; UB proves it by C_n + T_reachable
extension. Different proofs of the same conclusion.
**C_1(p)**: 0.95 — conclusion is EC05 level (0.97); derivation-specific confidence 0.90;
combined ~ 0.94
**Status**: High confidence. The conclusion is robustly established. UB's mechanical
derivation adds precision.

---

### Theorem 6: Necessity of A > 0 and eta_M > 0
**Claim**: Both A > 0 and eta_M > 0 are necessary for persistence.
**Base assumptions used**: EC04 (survival threshold), EC02 (improvement rate), EC10 (driver)
**Classification**: COLLAPSE for eta_M > 0 part; NOVEL for A > 0 necessity proof
**Evidence**:
- eta_M > 0 necessity: EC02 establishes rate of improvement as the measure. EC04 says
  entities below threshold are removed. Schmidhuber and Friston both ground "must improve
  or be selected out." The eta_M > 0 necessity is a COLLAPSE — derivable from shared premises.
- A > 0 necessity: no prior framework derives that engagement probability must be positive.
  Schmidhuber and Friston both assume engagement. UB's derivation that A = 0 entities are
  selected out (even if eta_M > 0, if A = 0, no gradient signals are engaged, no improvement
  occurs, survival is compromised) is NOVEL. This is NV03.
**C_1(p)**: eta_M part: 0.90; A > 0 part: 0.80 (novel, no independent grounding)
**Status**: The A > 0 necessity is UB's second most novel individual theorem. Without it,
the paper would not have a formal account of failure-to-engage.

---

### Corollary 6a: I(E) = A * eta_M via Law of Total Expectation
**Claim**: The product form follows from A as probability (response gate) and eta_M as
conditional rate. E[improvement] = P(engage) * E[improvement | engage] = A * eta_M.
**Base assumptions used**: EC02 (improvement rate), NV03 (A as probability gate)
**Classification**: NOVEL (product form derivation) — requires A as probability, which is NV03
**Evidence**: The law of total expectation is standard probability theory. The derivation
follows immediately once A is defined as a probability (NV03). Prior frameworks don't have
the product form because they don't have A as an explicit probability.
**C_1(p)**: 0.90 — proof is tight; rests on A's probabilistic definition (NV03 at ~0.80)
**Status**: Now fully resolved by the Definition 6 correction (A leads with probabilistic
definition). The product form follows directly.

---

### Corollary 8a: Intersection Efficiency and M → N Acceleration
**Claim**: Intersection of K_i and K_j with overlapping + distinct dimensions accelerates M → N.
**Base assumptions used**: NV02 (M < N), NV04 (intersection mechanism), EC07 (collective knowledge)
**Classification**: NOVEL
**Evidence**: EC07 (collective knowledge) grounds the qualitative claim (collective > individual).
Maynard Smith grounds the historical evidence. But the formal cardinality argument (|K_j \ K_i|
dimensions per operation vs. 1 per solo execution step) is unique to UB. No prior framework
derives the specific acceleration mechanism with this precision.
**C_1(p)**: 0.78 — qualitative claim supported (EC07 at 0.83); quantitative mechanism
novel (no independent verification of the cardinality argument)
**Status**: The weakest well-grounded claim in UB. The qualitative result is solid; the
quantitative mechanism needs empirical verification.

---

### Definition 14: Sentience
**Claim**: Sentience = representation that other projections of W exist beyond K_i.
**Base assumptions used**: EC05 (incompleteness), EC06 (belief as action), OV04 (meta-cognitive awareness)
**Classification**: NOVEL (derivation) + COLLAPSE (conclusion)
**Evidence**: OV04 shows FEP sophisticated inference independently grounds meta-cognitive
awareness. The conclusion (there is a higher-order cognitive capacity of modeling own inference)
is a COLLAPSE. The *derivation from the induction chain* (factorization + M < N + retention
criterion -> normalization improvement is dominant) is NOVEL. Prior frameworks make meta-
cognition architectural; UB makes it derivable.
**C_1(p)**: 0.85 — conclusion supported by OV04 (Friston 2021 independent grounding);
specific derivation chain novel
**Status**: Well-grounded. The independent grounding from Friston 2021 sophisticated
inference strengthens this claim significantly.

---

## Full Classification Summary

| Theorem/Definition | Classification | C_1(p) | Key Evidence |
|-------------------|----------------|--------|--------------|
| T1: Minimization Imperative | COLLAPSE | 0.93 | EC04 + EC06 + EC10 |
| T2b Part 3: Graph Necessity | **NOVEL** | 0.82 | EC08 base + novel two-option proof |
| T3 Part I: Convergence | COLLAPSE + NOVEL (proof structure) | 0.87 | EC10 + EC02; novel induction |
| T3 Part II: M < N | COLLAPSE + NOVEL (derivation) | 0.95 | EC05 (Gödel/Tarski); novel mechanical proof |
| T6: A > 0 and eta_M > 0 necessary | eta_M: COLLAPSE; A > 0: **NOVEL** | 0.85 avg | EC04/EC02; A > 0 unique to UB |
| C6a: I(E) = A * eta_M | **NOVEL** (requires NV03) | 0.90 | Law of total expectation + NV03 |
| C8a: Intersection acceleration | **NOVEL** | 0.78 | EC07 qualitative; cardinality novel |
| D6: Agency (probabilistic) | COLLAPSE + **NOVEL** (A=0 case) | 0.88 | EC10; A in (0,1) novel |
| D14: Sentience | COLLAPSE + **NOVEL** (derivation) | 0.85 | OV04; derivation chain novel |

---

## Contradiction Analysis

Three potential contradictions identified:

### CONTRADICTION 1: FEP Equilibrium vs. M < N (EM02)
**Conflict**: FEP6 (free energy approaches minimum) vs. Theorem 3 Part II (M < N always).
**Shared base**: EC01 (bounded context), EC05 (incompleteness).
**Resolution**: FEP6 requires stationarity. In an open, non-stationary environment,
FEP6's equilibrium is never reached — consistent with Theorem 3 Part II. The contradiction
resolves to a scope qualification: FEP6 is the fixed-world idealization; Theorem 3 Part II
is the open-world reality. NOT a true contradiction — a scope mismatch.
**Status**: Resolved by EM02.

### CONTRADICTION 2: Gibson Direct Perception vs. Internal Structured Representation (EC03)
**Conflict**: GI2 (no internal reconstruction needed) vs. Theorem 2b Part 3 (structured
representation necessary).
**Shared base**: Both accept that agents act on W.
**Resolution**: Scope mismatch. Gibson addresses perceptual coupling (moment-to-moment
sensorimotor loops). UB addresses knowledge accumulation over time under bounded context.
Direct perception is consistent with UB for L0 actions (certain, no inference cost);
UB's Theorem 2b Part 3 applies to the knowledge that makes L0 possible, not to the
L0 action itself. NOT a true contradiction — different scopes.
**Status**: Resolved by scoping.

### NOT A CONTRADICTION BUT A TENSION: Schmidhuber compression vs. graph necessity
**Tension**: Schmidhuber's compressor C can take any form; UB's graph structure is necessary.
**Shared base**: EC08 (compression = factoring).
**Analysis**: This is not a contradiction. Schmidhuber says compression progress is
the measure; he does not say non-graph compression is sufficient. His C can include graph
structures. UB adds: among all compression strategies, the graph form is the only one
that preserves information at C_n. Schmidhuber's framework is consistent with NV01;
it just doesn't prove it.
**Status**: Not a contradiction. UB extends Schmidhuber; doesn't contradict.

---

## Confidence Profile: Strongest to Weakest

```
T3 Part II (M < N)         ████████████████████ 0.95  ← Gödel/Tarski/FEP all ground this
T1 (Minimization)          ████████████████████ 0.93  ← EC04+EC06+EC10 converge
EC10 (Driver = U > 0)      ████████████████████ 0.93  ← 5 frameworks
C6a (I = A * eta_M)        ██████████████████   0.90  ← Tight; rests on NV03
T3 Part I (Convergence)    █████████████████    0.87  ← EC10 + optimization theory
D6 (Agency probabilistic)  █████████████████    0.88  ← Neural evidence (Dehaene)
D14 (Sentience)            █████████████████    0.85  ← OV04 (Friston 2021) supports
T6 (A>0 and eta_M>0)       █████████████████    0.85  ← Mixed: eta_M strong, A novel
T2b Part 3 (Graph needed)  ████████████████     0.82  ← Strongest NOVEL; only in UB
C8a (Intersection accel)   ███████████████      0.78  ← Weakest: cardinality novel only
```

---

## Key Finding: The Derivation Chain

The strongest path through the theory:

```
EC05 (Incompleteness, 0.97)
  → EC10 (Driver always positive, 0.93)
    → T3 Part I (Convergence per dimension, 0.87)
    → T3 Part II (M < N always, 0.95)
      → NV02 (M < N formal bound)
        → NV04 (Intersection accelerates M→N)
          → C8a (0.78)

EC01 (Bounded context, 0.92)
  → T2b Part 3 (Graph necessity, 0.82)  ← Most novel; least cross-grounded
    → NV01 → EM03 (grounds FEP architecture)

EC04 (Survival threshold, 0.94)
  → T1 (Minimization, 0.93)
  → T6 (A>0 and eta_M>0, 0.85)
    → NV03 (A as gate) → EM01 (Schmidhuber special case)
```

The theory is strongest where it derives from EC05 (incompleteness, 0.97) and
EC04 (survival, 0.94). It is weakest at Corollary 8a's cardinality claim and
Theorem 2b Part 3's graph-form specificity. Both are novel; both lack independent
external grounding. These are the claims most in need of empirical verification
or independent theoretical derivation.

---

## What the Theory Adds to the Field

Ranked by novelty (proportion of content not derivable from prior frameworks):

1. **Theorem 2b Part 3** (graph form is necessary, not just structured form): fully novel
2. **A > 0 necessity in Theorem 6** (formal treatment of failure-to-engage): fully novel
3. **Corollary 8a cardinality** (specific acceleration rate): fully novel
4. **I(E) = A * eta_M product form** (requires probabilistic A): novel conditioned on NV03
5. **NV05 sentience derivation** (motivation derived, not axiomatic): novel conditioned on base theorems
6. **M < N mechanical derivation** (same conclusion as EC05, but the mechanism is new): partially novel
7. **Inductive convergence proof** (same conclusion as prior work, clean inductive structure): partially novel
8. **Theorem 1, Theorem 3 conclusions**: well-grounded collapses — confirm known claims from new base

---

## Implication: What Needs Empirical Verification

The two fully novel claims with lowest C_1(p):
1. **Graph form necessity** (T2b Part 3, 0.82): needs an independent proof that non-graph
   structured representations fail at C_n — from outside the two-option argument.
2. **Cardinality acceleration** (C8a, 0.78): needs empirical measurement of M → N rate
   in multi-agent vs. solo learning systems, comparing |K_j \ K_i| per operation vs.
   1 per solo step.

Both are empirically testable. The graph form necessity could be tested computationally:
compare information retention of graph-structured vs. non-graph-structured agents under
bounded context constraints. The cardinality acceleration could be tested in multi-agent
learning systems by measuring effective dimension coverage per operation.

This is the experimental program that would raise the C_1(p) of UB's two weakest claims.
