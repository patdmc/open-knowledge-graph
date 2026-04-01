# §10 The Social Imperative

---

## Definition 17 — Collective Knowledge

**K_collective = ⋃_i K_i ∪ K_interaction**

| Symbol | What it is |
|---|---|
| K_collective | The collective knowledge graph — everything the group knows |
| ⋃_i K_i | Union of all individual knowledge graphs — everything any individual knows |
| ∪ | Set union |
| K_interaction | Knowledge no individual holds but that emerges from combining perspectives |

**K_interaction formally:**

**K_interaction = { p ∉ ⋃_i K_i | ∃ S ⊆ {E_i}, |S| ≥ 2 : P(E_p | ⋃_{j ∈ S} K_j) > max_{j ∈ S} P(E_p | K_j) }**

| Symbol | What it is |
|---|---|
| p ∉ ⋃_i K_i | p is not in any individual's knowledge graph |
| ∃ S ⊆ {E_i}, \|S\| ≥ 2 | There exists some subset S of at least 2 entities |
| P(E_p \| ⋃_{j ∈ S} K_j) | Posterior probability of the event corresponding to p, given the combined knowledge of S |
| max_{j ∈ S} P(E_p \| K_j) | The best any single member of S can achieve alone |
| > | The combined posterior strictly exceeds the best individual posterior |

K_interaction is knowledge that requires a specific combination of distinct K_i to become accessible. The number of contributing subsets grows as 2^n − n − 1 (all subsets of size ≥ 2), so K_interaction grows super-additively with collective size.

---

## Definition 18 — Provenance

**prov(p) = (attribution(p), evidence(p), derivation(p))**

| Symbol | What it is |
|---|---|
| prov(p) | Provenance triple for proposition p |
| attribution(p) | Who established p — the source |
| evidence(p) | How it was verified — what observations ground it |
| derivation(p) | How it was inferred — what prior knowledge it was derived from |

**P(p is reliable) ∝ prov(p)**

| Symbol | What it is |
|---|---|
| P(p is reliable) | The probability that p accurately represents W |
| ∝ prov(p) | Proportional to: reliability can be assessed only through provenance |

A fact with no provenance is indistinguishable from noise. This is an epistemic requirement, not a social norm. Provenance is what separates a knowledge graph from a rumor.

---

## Theorem 10 — Collective Gradient Dominance

**Transmission fidelity:**

**λ = σ(transfer) / (σ(transfer) + ε(transfer))**

| Symbol | What it is |
|---|---|
| λ | Transmission fidelity — ratio of signal to total content transferred |
| σ(transfer) | Signal gain in E_j's K from the transfer — accurate propositions received |
| ε(transfer) | Noise introduced — distortion, error, decontextualization |
| λ_min | Break-even fidelity: the point where signal gain from union equals noise cost of transfer |

**Main inequality:**

**dU_collective/dt < dU_i/dt ≤ 0**

| Symbol | What it is |
|---|---|
| dU_collective/dt | Rate of change of collective uncertainty over time — negative means decreasing |
| dU_i/dt | Rate of change of any individual's uncertainty |
| < | The collective reduces uncertainty strictly faster than any individual |
| ≤ 0 | Both are non-increasing — both are descending |

Holds when λ > λ_min. Below λ_min, the inequality inverts: the collective degrades rather than grows.

**Rate bound for individual:**

**dU_i/dt ≤ max_{f ∈ F_i} G(f, K_i) = max_{f ∈ F_i} I(f ; w | K_i)**

| Symbol | What it is |
|---|---|
| max_{f ∈ F_i} G(f, K_i) | Best information gain achievable by the best individual action |
| I(f ; w \| K_i) | Mutual information between action f and world w, given K_i — information theory name for same quantity |

**Rate bound for collective:**

**dU_collective/dt ≤ max_{f ∈ F_collective} I(f ; w | K_collective)**

| Symbol | What it is |
|---|---|
| F_collective = ⋃_i F_i | Union of all individual action spaces |
| K_collective ⊇ K_i | Collective knowledge dominates any individual's |

Since F_collective ⊇ F_i and K_interaction grounds actions unavailable to any individual, the collective's ceiling strictly exceeds any individual's ceiling.

---

## The Precipitation Advantage

**|Prec(K_collective)| > |Prec(K_j)|**

| Symbol | What it is |
|---|---|
| Prec(K) | Set of factors extractable from M given K — dimensions that can be named and factored out |
| \|Prec(K_collective)\| | Number of precipitable factors available from collective knowledge |
| \|Prec(K_j)\| | Number of precipitable factors available from any individual's knowledge |
| > | Strictly more — the collective can precipitate dimensions no individual can reach |

**The collective advantage by type:**

| Shared item | Advantage |
|---|---|
| p ∈ K | Reduces U at a specific region of W — linear, local |
| f ∈ F | Bounds a region of W directly — linear with coverage leverage |
| m ∈ M | Raises η_M — accelerates all future reductions — multiplicative, global |

**Super-additive precipitation:**

**|Prec(K_collective)| ≫ ∑_i |Prec(K_i)|**

| Symbol | What it is |
|---|---|
| ≫ | Much greater than |
| ∑_i \|Prec(K_i)\| | Sum of what all individuals could precipitate working independently |

The collective enables strictly more precipitations than the sum of individual precipitations — because cross-dimensional factors are visible only from multiple combined projections. Entities working independently can only extract factors within their own observable subspace of M. The collective pools projections and can identify co-variance patterns across dimensions that no individual can observe simultaneously.

**The intelligence amplification consequence (from Corollary 6a):**

When E_j receives an M-factor from the collective, η_{M,j} increases. Since I(E_j) = 𝒜_j · η_{M,j}, every subsequent K acquisition and F precipitation runs faster. A shared M-factor does not reduce uncertainty at one point — it accelerates the rate of reduction across all of W reachable by E_j.
