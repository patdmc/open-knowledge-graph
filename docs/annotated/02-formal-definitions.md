# §3 Formal Definitions

---

## Definition 3 — Informed Action Space

**F ⊆ (K × W_obs → ΔA)**

| Symbol | What it is |
|---|---|
| F | The full set of informed actions available to E |
| ⊆ | "is a subset of" — F is drawn from this function space, not every possible function |
| K | The knowledge graph — what E knows |
| × | Cartesian product — both K and W_obs are inputs simultaneously |
| W_obs | The observable projection of W — what E can perceive through its sensing function σ |
| → | "maps to" |
| ΔA | A probability distribution over possible actions — a simplex, not a single action |

Each f ∈ F has type: **given (what I know, what I can see) → output a probability distribution over actions.**

Not a lookup table. Not a reflex. A function that reasons from knowledge about what to do.

---

## Definition 3a — Action Feedback

**K_f = K ∪ {p_f}**

| Symbol | What it is |
|---|---|
| K_f | The updated knowledge graph after executing action f |
| K | The knowledge graph before execution |
| ∪ | Set union — adding to the existing graph |
| {p_f} | A set containing the single new proposition learned by observing what happened when f was executed |

**δ = φ(f, w) − 𝔼[φ(f, ·) | K]**

| Symbol | What it is |
|---|---|
| δ | The gradient signal — the surprise, the learning signal |
| φ(f, w) | The actual feedback received from world state w when f was executed |
| 𝔼[φ(f, ·) \| K] | The expected feedback — what K predicted would happen |
| − | The difference: actual minus expected. Positive = better than predicted; negative = worse |

This is the burn. The world tells you whether your action worked. The gap between what happened and what you expected is the signal M acts on.

---

## Definition 3b — Information Gain

**G(f, K) = 𝔼_w[ U(w, K) − U(w, K_f) ] = H(W | K) − H(W | K, φ(f,w)) = I(f ; W | K)**

| Symbol | What it is |
|---|---|
| G(f, K) | Information gain of executing f given current knowledge K — average uncertainty removed |
| 𝔼_w[·] | Expected value over all possible world states w |
| U(w, K) | Uncertainty about specific state w before executing f |
| U(w, K_f) | Uncertainty about specific state w after executing f and updating K |
| H(W \| K) | Conditional entropy — average uncertainty over all world states given K |
| H(W \| K, φ(f,w)) | Average uncertainty after also conditioning on the feedback from f |
| I(f ; W \| K) | Mutual information between action f and the world W, given K |
| G(f, K) ≥ 0 | Conditioning on true information cannot increase entropy — lower bound is zero |

All three expressions are equal. The first is intuitive (average drop in uncertainty). The second is the entropy form. The third is the standard information-theory name.

---

## Definition 5 — Uncertainty

**U(w, K) = −log P(w | K)**

| Symbol | What it is |
|---|---|
| U(w, K) | Uncertainty about specific world state w given knowledge K |
| −log | Negative logarithm — surprisal. High when P is low; zero when P = 1 |
| P(w \| K) | Probability that w is the true world state given everything in K |

**P(w | K) ∝ P₀(w) · ∏_{p ∈ K} ℓ_p(w)**

| Symbol | What it is |
|---|---|
| ∝ | Proportional to — the right side is then normalized so all probabilities sum to 1 |
| P₀(w) | Prior probability of world state w before any knowledge is applied |
| ∏_{p ∈ K} | Product over every proposition p in the knowledge graph |
| ℓ_p(w) | Likelihood factor for proposition p: probability of having observed the evidence that grounds p, if the true state were w |

This is Bayes' theorem. Start with a prior over world states. Multiply by the likelihood contribution of every proposition in K. Normalize. The result is how probable each world state is given everything E knows.

**H(W | K) = 𝔼_w[ U(w, K) ] = −∑_w P(w | K) log P(w | K)**

| Symbol | What it is |
|---|---|
| H(W \| K) | Conditional entropy — the average of U over all possible world states |
| 𝔼_w[·] | Expectation over w, weighted by P(w \| K) |
| ∑_w | Sum over all world states w |
| P(w \| K) log P(w \| K) | Shannon entropy term for state w |

The minimization imperative (Theorem 1) operates on H(W | K) — the expected quantity. U(w, K) is the pointwise uncertainty the entity faces when it encounters a specific w.

---

## Definition 6 — Agency

Primary definition: 𝒜 ∈ [0, 1] is the probability that entity E engages M in response to a gradient signal — the response gate.

**Empirical operationalization:**

**𝒜(E) ∝ lim_{U^d_lethal → ∞} dU/dt (E)**

| Symbol | What it is |
|---|---|
| 𝒜(E) | Agency of entity E |
| ∝ | Proportional to |
| lim_{U^d_lethal → ∞} | Take the limit as the lethality threshold goes to infinity — remove survival pressure entirely |
| U^d_lethal | The uncertainty ceiling in domain d above which E does not survive |
| dU/dt | Rate of change of uncertainty over time — how fast E is reducing uncertainty |

**The test:** remove survival pressure. Does the entity keep learning? If yes, 𝒜 > 0. If it stops as soon as it doesn't have to learn, it was compelled updating — not genuine agency.

**Precipitation of 𝒜 from M:**

𝒜 is the first factor to precipitate from the cross product of M. An entity that re-computes its directional drive at every M-application incurs O(n·k) cost. An entity that maintains 𝒜 as the shared orientation across all M-applications achieves O(1) access. Same normalization argument that precipitates K from F.

---

## Definition 7 — Intelligence

**I(E) = 𝒜 · η_M**

| Symbol | What it is |
|---|---|
| I(E) | Intelligence of entity E |
| 𝒜 | Agency — the drive (Definition 6) |
| η_M | Efficiency of M — the mechanism (Definition 11a) |

**Performance measure (not the definition of intelligence):**

**𝔼_W[ U(w_t, K_t) − U(w_{t+1}, K_{t+1}) ]**

| Symbol | What it is |
|---|---|
| 𝔼_W[·] | Expected value over world states |
| U(w_t, K_t) | Uncertainty at time t |
| U(w_{t+1}, K_{t+1}) | Uncertainty at time t+1, after one learning step |
| Difference | Uncertainty reduction achieved in one step — a consequence of past M activity, not the definition of intelligence |

Intelligence is not the size of K or F. K and F are what intelligence has built. Intelligence is the efficiency with which M continues building.
