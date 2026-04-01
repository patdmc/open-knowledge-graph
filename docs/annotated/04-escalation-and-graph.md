# §5 The Escalation Principle and Graph Necessity

---

## Theorem 2 — Escalation

**U(w, K^[f]) ≈ 0**

| Symbol | What it is |
|---|---|
| U(w, K^[f]) | Uncertainty about world state w using only the knowledge relevant to action f |
| K^[f] | The subgraph of K that applies to f — not all of K, just the relevant slice retrieved by graph traversal |
| ≈ 0 | Approximately zero — the action can handle the problem without escalating |

The rule: handle f at the cheapest encoding level where this holds. Escalate to the next level only when this fails.

**Encoding levels (interfaces in the continuous space 𝓔(c, r)):**

| Level | Domain | Runtime Cost | Reversibility |
|---|---|---|---|
| L_n | Active reasoning / novel uncertainty | O(compute) | Fully reversible |
| ⋮ | Learned patterns, habits | Decreasing | Decreasing |
| L₁ | Reflex / autonomic | ≈ 0 | Semi-permanent |
| L₀ | Genetic encoding | 0 | Permanent within lineage |

| Symbol | What it is |
|---|---|
| c | Runtime cost — the cost of engaging this encoding level |
| r | Reversibility — how easily the encoding can be revised |
| 𝓔(c, r) | The continuous two-dimensional encoding space parameterized by cost and reversibility |
| L_i | The interface at the i-th significant boundary in 𝓔(c, r) |
| C_i | Cost of failure when an encoding error propagates at level L_i's permanence |

---

## Definition 8a — Inference Quality

**ρ(L_n, f) = |K^[f]| / |ctx(L_n, t)|**

| Symbol | What it is |
|---|---|
| ρ(L_n, f) | Inference quality — signal-to-noise ratio of active context for problem f |
| \|K^[f]\| | Number of propositions in context that are actually relevant to f |
| ctx(L_n, t) | The set of all propositions loaded into active context at time t |
| \|ctx(L_n, t)\| | Total number of propositions in context |

ρ = 1: everything in context is relevant — maximum signal, zero noise.
ρ → 0: context dominated by irrelevant content — inference degrades.

Every proposition in ctx(L_n) that is not in K^[f] is noise. It contributes variance to inference without contributing signal.

---

## Theorem 2a — Top-Down Allocation Degrades ρ

**ρ(L_n, f_novel) = |K^[f_novel]| / ( |K^[f_novel]| + k(t) · α )**

| Symbol | What it is |
|---|---|
| f_novel | The genuinely new problem L_n is trying to reason about |
| k(t) | Number of subproblems currently delegated and being monitored at time t |
| α | Minimum context cost per active delegation — overhead per managed subprocess, α > 0 |
| k(t) · α | Total delegation overhead consuming context slots |

As |F| grows → k(t) grows → denominator grows → ρ → 0.

The context fills with management overhead. When k(t) · α ≥ C_n − |K^[f_novel]|, L_n lacks capacity for genuine inference entirely. The only recovery is growing C_n — but growing C_n adds more noise, compounding the failure.

---

## Theorem 2b — Graph-Structured K Bounds Active Context

**Part 1:** |ctx(L_n, t)| = |K^[f]| — no delegation overhead in escalation architecture.

**Part 2:** ρ(L_n, f) = 1 — inference quality is maximum.

| Symbol | What it is |
|---|---|
| \|ctx(L_n, t)\| = \|K^[f]\| | Context holds exactly what f needs — nothing more |
| ρ = 1 | Pure signal — the ratio is 1 because numerator and denominator are equal |

Conditional on edge fidelity: ρ = 1 holds when graph edges correctly encode all and only the inferential relevance of each proposition. Missing edges → under-retrieval. Spurious edges → ρ < 1.

**Part 3 — Graph is the necessary structure:** Three simultaneous constraints force graph structure:

- **Retrieval:** cost of accessing knowledge relevant to f must be O(|K^[f]|), not O(|K|)
- **Context:** |ctx(L_n, t)| = |K^[f]| exactly — no extraneous propositions
- **Memory:** shared propositions stored once, not duplicated at each point of use

**Option A (no graph):** Discard an existing proposition without extracting relational structure → information loss. No mechanism to know which propositions are redundant without explicit edges.

**Option B (graph):** Factor — extract shared structure, reducing |K| without discarding. Naming the shared factor f*, creating a node, storing typed edges: this is graph construction. Update cost: O(1) instead of O(|F|).

No third option exists. Any representation that makes space without factoring loses information. Any representation that makes space by factoring is constructing a graph. The graph is not where alternatives end up — it is what factoring is.

**Part 4:** As K grows, more problems are handled below L_n. L_n stays occupied by genuinely novel uncertainty. C_n need not grow — K grows without filling L_n.

**∀ E that persists under bounded C_n and selection pressure: K has graph structure.**
