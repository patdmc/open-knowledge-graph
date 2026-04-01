# §6 Gradient Descent on Uncertainty

---

## Definition 9 — Survival Threshold

**U(w, K) > U^d_lethal ⟹ E does not survive in d**

| Symbol | What it is |
|---|---|
| U^d_lethal | The uncertainty ceiling in domain d — above this, E cannot act well enough to persist |
| d | A domain — a subset of world state space W |
| ⟹ | "Implies" — if uncertainty exceeds the threshold, the consequence follows necessarily |

Not a preference. Not adjustable by the entity. Imposed by W.

---

## Theorem 3, Part I — The Update Step

Every executed action f_i in world state w generates a gradient signal:

**δ = φ(f_i, w) − 𝔼[φ(f_i, ·) | K]**

| Symbol | What it is |
|---|---|
| δ | The gradient signal — signed deviation of observed outcome from expectation |
| φ(f_i, w) | Actual outcome when action f_i was executed in state w |
| 𝔼[φ(f_i, ·) \| K] | Expected outcome predicted by current knowledge K |
| − | Actual minus expected — the surprise |

The signal arrives from W but is applied in F: it is a subgradient of U as a functional on F. It arrives whether or not the entity has 𝒜. W imposes it.

**The update when M engages:**

**Δf_i = −η · δ**

| Symbol | What it is |
|---|---|
| Δf_i | The change applied to informed action f_i |
| − | Negative — descend the gradient, move toward lower uncertainty |
| η | Learning rate — how large each update step is |
| δ | The gradient signal from above |

𝒜 is the gate: the probability that M engages on δ when δ arrives. At 𝒜 = 0, M never engages; (F, K) stays fixed regardless of the signals W sends.

**Convergence (by induction on sensed dimensions):**

Base case M = 1: δ₁ is a scalar; U ≥ 0 and bounded below; each update Δf₁ = −η · δ₁ reduces U in expectation; by monotone convergence, f₁ converges to a local minimum.

Inductive step M → M+1: new dimension p_{M+1} is independent at the moment of addition (no edges yet to existing propositions); the base case applies to f_{M+1} independently; combined with the M-dimensional inductive hypothesis, convergence follows for M+1.

Convergence holds for any M the entity currently senses, without assuming anything about the total dimensionality of W.

---

## Theorem 3, Part II — Non-Termination

**𝒯_reachable(K_f) ⊇ 𝒯_reachable(K)**

| Symbol | What it is |
|---|---|
| 𝒯_reachable(K) | Set of all trajectories reachable given current knowledge K |
| 𝒯_reachable(K_f) | Set of all trajectories reachable after executing f and learning p_f |
| ⊇ | "Is a superset of" — the new set contains at least everything the old one did |
| Strict ⊃ | Holds whenever p_f grounds at least one new action not available from K alone |

The frontier always expands. At any finite time, K(t) has not covered 𝒯_reachable(K(t)). Therefore:

**∇_F U is never globally zero over 𝒯_reachable**

| Symbol | What it is |
|---|---|
| ∇_F | Gradient with respect to F — the direction of steepest descent in the action space |
| ∇_F U = 0 | Would mean no more uncertainty to reduce anywhere in the reachable domain |
| "never globally zero" | There is always more to learn — the gradient never flattens everywhere |

Descent never terminates within the reachable epistemic domain. This is not a practical limit; it is a structural property.

---

## Remark — Multi-Timescale Gradient Descent

Both individual learning and evolutionary selection share the same structure: generate a candidate state, submit to W's evaluation, retain if valid.

**Individual learning (directed):**

**ΔF_individual = −η_ind · ∇_F U(w, K)**

| Symbol | What it is |
|---|---|
| ΔF_individual | Change to the action space for one entity after one learning step |
| η_ind | Individual learning rate — relatively large (cost of a single wrong action is finite) |
| ∇_F U(w, K) | Gradient of uncertainty with respect to F at specific encountered state w |

**Evolutionary selection (undirected):**

**ΔF_species = select( mutate(F_population) )**

| Symbol | What it is |
|---|---|
| ΔF_species | Shift in the population-level distribution over F-variants across a generation |
| F_population | The distribution of F across all individuals in the population |
| mutate(·) | Random perturbation of L₀ encoding — generates candidate configurations |
| select(·) | W evaluates each through the survival criterion — retains what persists |

Same structure. Different mechanism. Individual: directed step via continuous δ. Evolutionary: random perturbation + binary survival filter.

---

## Theorem 4 — Rigor Threshold

**θ_i = 1 − ε_acceptable / C_i**

| Symbol | What it is |
|---|---|
| θ_i | Certainty threshold required before promoting an action to encoding level L_i |
| 1 | Maximum possible certainty |
| ε_acceptable | Maximum acceptable expected cost of an encoding error |
| C_i | Cost of failure at level L_i — what goes wrong if the encoding is wrong |
| ε_acceptable / C_i | The fraction of failure cost the entity is willing to risk — shrinks as C_i grows |

**Derivation:**

**𝔼[encoding error cost] = (1 − θ) · C_i ≤ ε_acceptable**

| Symbol | What it is |
|---|---|
| (1 − θ) | Probability the action is wrong — complement of current confidence |
| C_i | Cost incurred when it is wrong |
| ≤ ε_acceptable | Must not exceed the acceptable error budget |

Rearranges to: θ ≥ 1 − ε_acceptable / C_i. The minimum safe threshold is θ_i.

**Derived learning rate relationship:**

**η_i ∝ 1/θ_i = C_i / (C_i − ε_acceptable)**

| Symbol | What it is |
|---|---|
| η_i ∝ 1/θ_i | Learning rate is inversely proportional to the rigor threshold |
| C_i / (C_i − ε) | As C_i → ∞ (genetic encoding, L₀), this → 1, meaning η_evo is very small |
| For L_n | C_n is small, ε/C_n is non-negligible, θ_n is meaningfully < 1, so η_ind is large |

**η_evo ≪ η_ind**

Evolution is slow not by assumption — it is derived from the cost of failure at L₀ being lineage survival. The learning rate difference is a consequence of the cost-of-failure difference, not an independent assumption.

**The consequence:** You must be nearly certain before permanently changing the source code. A bad encoding at L₀ does not kill one individual — it kills the lineage. The rigor of the test is proportional to the permanence of the consequence.
