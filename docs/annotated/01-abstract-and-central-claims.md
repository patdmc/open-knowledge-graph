# §1–2 Abstract and Central Claims

## The Core Equation

**I(E) = 𝒜 · η_M**

| Symbol | What it is |
|---|---|
| I(E) | Intelligence of entity E — the thing being defined and derived |
| E | An entity: any system that perceives, knows things, and acts |
| 𝒜 | Agency — the probability that E engages its learning mechanism when it receives a gradient signal. The drive to know. Range: [0, 1] |
| · | Multiplication — specifically, probability × conditional rate (see Corollary 6a) |
| η_M | Efficiency of the higher-order function space M — how effectively the learning mechanism converts gradient signals into improved (F, K). How well E learns when it tries |

**Why multiplication and not addition or minimum:**

From Corollary 6a, the derivation is:

𝔼[improvement rate] = 𝒜 · η_M + (1 − 𝒜) · 0 = 𝒜 · η_M

| Symbol | What it is |
|---|---|
| 𝔼[improvement rate] | Expected rate of improvement in (F, K) |
| 𝒜 · η_M | Contribution when M engages: probability of engagement × rate when engaged |
| (1 − 𝒜) · 0 | Contribution when M does not engage: probability of non-engagement × zero improvement |

𝒜 is a probability; η_M is the rate conditional on engagement. These combine as a product by the law of total expectation. The sum 𝒜 + η_M would assign positive intelligence to a zero-drive entity, which Theorem 6 proves cannot persist.

---

## What the Symbols Mean Informally

**𝒜 = 0:** The entity receives gradient signals but M never engages. It does not update. Eventually it encounters uncertainty above the lethality threshold and does not survive. Selection removes it.

**η_M = 0:** The entity has drive but M cannot improve (F, K) from evidence. It executes whatever prior gradient descent built into it, but does not itself run gradient descent. Rich knowledge, sophisticated actions — but an automaton, not an intelligence.

**I(E) = 0 if either is zero.** Both are necessary. Neither is sufficient.

---

## Supporting variables introduced in the abstract

| Symbol | What it is |
|---|---|
| K | Knowledge graph — a directed graph of propositions about W |
| F | Informed action space — the set of actions E knows how to execute |
| L_n | The active context window — the highest encoding level, where novel uncertainty is resolved |
| C_n | Capacity of L_n — physically bounded, cannot grow without limit |
| \|F\| | The size (cardinality) of the action space |
| ρ | Inference quality — ratio of relevant-to-total content in active context |
| K^[f] | The subgraph of K relevant to informed action f |
