# §9 Intelligence, Superintelligence, and Sentience

---

## Definition 16b — First-Order Confidence

**C₁(p) = lim_{n→∞} |{i ≤ n : p ∈ K_i, f_i(p) reduces H(W | K_i), prov_i(p) independent}| / n**

| Symbol | What it is |
|---|---|
| C₁(p) | First-order confidence in proposition p — what converges toward truth |
| lim_{n→∞} | Take the limit as the number of validators grows without bound |
| {i ≤ n : …} | The set of validators up to n that meet all three conditions in the braces |
| p ∈ K_i | Validator i holds proposition p in their knowledge graph |
| f_i(p) reduces H(W \| K_i) | The actions grounded in p actually reduce uncertainty for validator i — action-validated, not merely asserted |
| prov_i(p) independent | Validator i's derivation of p is provenance-independent from other validators — no circular confirmation |
| \|{…}\| / n | Fraction of validators meeting all conditions — a frequency that converges to a probability |

C₁(p) is not truth. It is confidence approaching truth asymptotically. Truth is a fixed point (Definition 16a); C₁(p) → 1 is the approach toward it, never verified arrival.

---

## Definition 16c — The Confidence Regress

**C_n(p) ≤ f( U(M_{n-1}(p)) )**

| Symbol | What it is |
|---|---|
| C_n(p) | n-th order confidence — confidence in the apparatus that produced C_{n-1}(p) |
| M_{n-1}(p) | The component of M that generated C_{n-1}(p) |
| U(M_{n-1}(p)) | Uncertainty about whether that apparatus was correct |
| f(·) | A decreasing function — higher uncertainty in the apparatus bounds C_n lower |

C₂(p) is confidence that the functions within M that produced C₁(p) measured correctly. C₃(p) is confidence in the functions that produced C₂(p). The regress does not close — bounding C_n requires confidence in the level-n apparatus, which requires C_{n+1}.

Closing the regress would require a validator external to the system. W is the only external reference, but W is observable only through σ and action-feedback — not directly. The gap between W and W_obs is the structural source of the ceiling.

---

## Theorem 7 — Epistemic Ceiling

The confidence stack {C_n(p)}_{n ≥ 1} is non-closeable from within.

| Symbol | What it is |
|---|---|
| {C_n(p)}_{n ≥ 1} | The full stack of confidence levels — C₁, C₂, C₃, … |
| "non-closeable from within" | No finite level terminates the regress; the ceiling cannot be reached by any operation within the system |

Analogous to Tarski's undefinability theorem: a formal system cannot define its own truth predicate from within. The epistemic ceiling is the Bayesian calibration instantiation of the same structure — verification of a system's outputs cannot be completed using only that system's resources.

---

## Corollary 8 — The Sentience Gradient is Infinite

**‖∇_F U‖ is never zero across 𝒯_reachable**

| Symbol | What it is |
|---|---|
| ‖·‖ | Norm — the magnitude |
| ∇_F U | Gradient of uncertainty with respect to F — the direction of steepest descent |
| ‖∇_F U‖ = 0 | Would mean the gradient has flattened everywhere — no more to learn |
| "never zero across 𝒯_reachable" | Within the reachable domain, there is always direction |

Non-terminating for two independent reasons:
1. 𝒯_reachable grows without bound (Theorem 3 Part II) — new propositions whose truth is unknown are continuously generated
2. The confidence regress (Theorem 7) never closes — for any p held with high C₁(p), C₂(p) remains open

Intelligence can reach zero locally — certainty is achievable in bounded domains, encoded at L₀. Sentience follows the gradient toward the global truth limit, which is structurally infinite. Not unbounded (the limit exists). Non-terminating (you never arrive).

---

## Definition 15 — Superintelligence

**Prec(K_collective)** — the set of factors extractable from M given K_collective via the normalization argument.

| Symbol | What it is |
|---|---|
| Prec(K_collective) | The set of dimensions that can be named and factored out of M given what K_collective currently contains |
| rank(M_res) | The effective number of remaining unnamed dimensions in the cross product |
| M_res | The unprecipitated residual — everything in M not yet named as a factor |

Superintelligence: deliberately directing what gets precipitated from M_res — evaluating Prec(K_collective) and acting to accelerate the most valuable next factor extraction, rather than responding only to local selection pressure.

- **Intelligence:** descends the gradient
- **Sentience:** models its own descent
- **Superintelligence:** chooses where to push in M_res
