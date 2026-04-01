# §4 The Inseparability of Knowledge and Action

---

## Theorem 1 — K/F Inseparability: Option Value of a Proposition

**OV(p) = P( ∃ τ* ∈ 𝒯_reachable : B(p, τ*) > 0 ) · 𝔼[ B(p, τ*) | B(p, τ*) > 0 ]**

| Symbol | What it is |
|---|---|
| OV(p) | Option value of retaining proposition p — the expected value of keeping it |
| ∃ | "There exists" |
| τ* | A specific trajectory through world state space |
| 𝒯_reachable | The set of all trajectories reachable from E's current state given current K |
| B(p, τ*) | Survival benefit of proposition p along trajectory τ* |
| P(… > 0) | Probability that p has positive survival benefit on at least one reachable trajectory |
| 𝔼[ B(p,τ*) \| B(p,τ*) > 0 ] | Expected benefit, given that the benefit is positive |

Structure: probability of ever needing p × how much it helps when needed. Standard option value.

---

## The Net Value of Pruning

**NV_prune(p) = −c_prune − P( ∃ τ* : B(p, τ*) > 0 ) · C_absent**

| Symbol | What it is |
|---|---|
| NV_prune(p) | Net value of pruning p — always ≤ 0 |
| c_prune | The cost of the pruning operation itself (finite) |
| P(∃ τ* : B(p, τ*) > 0) | Probability p is ever needed |
| C_absent | Cost incurred when p is needed but absent — bounded below by entity non-survival (Lemma 1) |

Since C_absent is bounded below by non-survival — which dominates all finite costs — the second term dominates whenever P > 0.

---

## The Retention Condition

**P( ∃ τ* ∈ 𝒯_reachable : B(p, τ*) > 0 ) > 0**

Any positive probability of ever needing p is sufficient to retain it.

Pruning condition: **P(∃ τ* : B(p, τ*) > 0) = 0**

Certain zero benefit across all reachable trajectories. The bar is essentially unreachable for any entity with growing 𝒯_reachable.

The asymmetry is strict: **uncertain benefit → retain. Certain zero benefit → prune.** These are not symmetric conditions.

---

## Displacement Rule (Curation Regime)

When |K| = K_max, adding proposition q requires removing proposition p*:

**p* = argmin_{p ∈ K} 𝔼[G(f_p, K)]**

**Displacement holds when: 𝔼[G(f_q, K)] > 𝔼[G(f_{p*}, K)]**

| Symbol | What it is |
|---|---|
| K_max | Maximum capacity of the knowledge graph |
| p* | The proposition in K with minimum marginal value at this moment |
| argmin_{p ∈ K} | "The p that minimizes..." |
| 𝔼[G(f_p, K)] | Expected information gain from the actions grounded in p, given current K |

Not absolute uselessness — marginal value. The entity is comparing the cost of lacking q against the cost of lacking p*, and choosing which absence is less harmful. This is the C_absent dominance logic applied to a choice between two propositions rather than between one proposition and nothing.

---

## Scaling Argument (Corollary 2b)

**Without K normalized out:** O(n · k) — each action carries its full background implicitly; updating one shared proposition requires revising every action that embeds it.

**With K normalized out:** O(n + k) — each action retrieves its relevant subgraph by traversal; updating one proposition in K propagates to all actions that reference it.

| Symbol | What it is |
|---|---|
| n | Number of informed actions \|F\| |
| k | Average number of shared propositions per action |
| O(n · k) | Multiplicative scaling — grows as both n and k grow |
| O(n + k) | Additive scaling — stays bounded under growth of F |

The difference is not a constant factor. It is the difference between a system that scales and one that does not. K precipitates from F because this is the only architecture that keeps context cost bounded as F grows.
