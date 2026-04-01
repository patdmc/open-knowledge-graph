# Annotated Formulas — Uncertainty Bounding as the Basis of Intelligence

> **Note:** These annotations reference the original monolithic paper
> (`UNCERTAINTY_BOUNDING_FORMAL_THEORY.tex`), which has since been
> superseded by the 5-paper series. Section numbers and theorem
> numbering may not match the current papers. See root `README.md`
> for the current paper series.

Every formula in the original paper, with each symbol explained inline.

## Files

| File | Covers |
|---|---|
| [01-abstract-and-central-claims.md](01-abstract-and-central-claims.md) | Abstract, Section 2 — the core equation I(E) = 𝒜 · η_M |
| [02-formal-definitions.md](02-formal-definitions.md) | Section 3 — Definitions 1–8: world, knowledge, action, entity, uncertainty, agency, intelligence |
| [03-inseparability-and-retention.md](03-inseparability-and-retention.md) | Section 4 — Theorem 1: why K and F cannot be separated; option value; retention criterion |
| [04-escalation-and-graph.md](04-escalation-and-graph.md) | Section 5 — Theorems 2, 2a, 2b: escalation, inference quality, why the graph is necessary |
| [05-gradient-descent.md](05-gradient-descent.md) | Section 6 — Theorems 3, 4: descent on uncertainty, multi-timescale learning, rigor threshold |
| [06-higher-order-function-space.md](06-higher-order-function-space.md) | Section 7 — Definitions 11, 11a; Theorems 5, 6; Corollary 6a: M, η_M, type irreducibility, I(E) derived |
| [07-unified-model.md](07-unified-model.md) | Section 8 — Definition 12: trajectory, performance, universal invariants |
| [08-intelligence-superintelligence-sentience.md](08-intelligence-superintelligence-sentience.md) | Section 9 — Definitions 13–16; Theorems 7, 8; certainty horizon, sentience, confidence |
| [09-social-imperative.md](09-social-imperative.md) | Section 10 — Definitions 17–18; Theorem 10: collective knowledge, provenance, collective gradient |

## Symbol Reference

| Symbol | Meaning |
|---|---|
| 𝒜 | Agency — probability E engages M on a gradient signal |
| η_M | M efficiency — rate of improvement per unit learning signal |
| I(E) | Intelligence of entity E |
| E | An entity: (K, F, σ, 𝒜) |
| K | Knowledge graph |
| F | Informed action space |
| M | Higher-order function space — the learning mechanism |
| W | World state space |
| W_obs | Observable projection of W |
| U(w,K) | Uncertainty about state w given knowledge K |
| 𝒯_reachable | Set of trajectories reachable from current state |
| C_n | Capacity of the active context window |
| θ_i | Rigor threshold for promotion to encoding level L_i |
| λ | Transmission fidelity |
| ρ | Inference quality — signal-to-noise in active context |
