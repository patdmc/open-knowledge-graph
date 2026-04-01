# §8 The Unified Model

---

## Definition 12 — Trajectory

**τ = (f₁, f₂, …, f_n),   f_i ∈ F**

| Symbol | What it is |
|---|---|
| τ | A trajectory — a sequence of informed actions executed over time |
| f_i | The i-th informed action in the sequence |
| f_i ∈ F | Each action is drawn from E's informed action space |
| n | The length of the trajectory |

**Performance realized by trajectory τ:**

**P(E, τ) = 𝔼_W[ U(w₀, K₀) − U(w_n, K_n) ]**

| Symbol | What it is |
|---|---|
| P(E, τ) | Total uncertainty reduction achieved by following trajectory τ |
| 𝔼_W[·] | Expected value over world states |
| U(w₀, K₀) | Uncertainty at the start of the trajectory |
| U(w_n, K_n) | Uncertainty at the end of the trajectory |
| Difference | Total uncertainty reduced — a measure of what current (F, K) accomplished |

P(E, τ) is a performance measure — the consequence of past M activity. It is not the definition of intelligence. I(E) = 𝒜 · η_M determines how quickly P improves across successive trajectories.

---

## The Universal Invariants

**∀ E that is intelligent: 𝒜(E) > 0 and η_M(E) > 0**

| Symbol | What it is |
|---|---|
| ∀ | "For all" |
| 𝒜(E) > 0 | Agency is positive — E has internal reason to engage the gradient |
| η_M(E) > 0 | M efficiency is positive — E can improve (F, K) from evidence |

Everything else can vary arbitrarily:
- The specific actions in F
- The content of K
- The substrate (carbon, silicon, collective)
- The sensing function σ
- The depth of the encoding hierarchy

These two cannot.

---

## The Operate Loop

1. **Orient (𝒜):** 𝒜 > 0 directs E toward regions of W where U(w, K) > 0
2. **Sense:** σ(w_t) → W_obs — project the world into the observable space
3. **Act:** F(K, W_obs) → f_i ∈ τ — select the next informed action
4. **Learn:** M((F, K), ℒ) → (F', K') — incorporate feedback; promote sufficiently certain actions toward lower encoding levels

| Symbol | What it is |
|---|---|
| σ(w_t) | Sensing function applied to true world state w at time t |
| W_obs | The entity's observable space — what σ makes visible |
| F(K, W_obs) | The action selection process: given knowledge and observations, choose an action |
| ℒ | The learning signal from executing f_i and observing its feedback |
| (F', K') | The updated joint state after M acts |

---

## Graph Relations on F

F carries graph structure for the same reason K does (Theorem 2b Part 3). The edge types on F:

| Relation | Formal meaning |
|---|---|
| depends_on | f_j cannot appear in τ before f_i — hard ordering constraint on the trajectory |
| recommends_before | Evidence shows (…, f_i, f_j, …) reduces more U than (…, f_j, f_i, …) — soft, confidence-scored |
| enhanced_by | A τ containing both f_i and f_j reduces more U than either trajectory alone |

M discovers better trajectories from evidence: learning which sequences reduce U most reliably, encoding that knowledge as graph relations on F.
