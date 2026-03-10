# Predictive Processing and Free Energy: Cross-Framework Profile

**Cluster**: Rao-Ballard (1999), Friston (2010), Friston et al. (2017), Friston et al. (2021)
**Prepared by**: Patrick McCarthy
**Date**: 2025

---

## Overview

This is the cluster most technically similar to the UB framework. Where other clusters provide philosophical antecedents, empirical measurements, or biological metaphors, FEP is a formal, information-theoretic theory of self-organizing agents that overlaps with UB in content, language, and formal apparatus. The relationship between FEP and UB is not analogy but structural comparison between two formal frameworks addressing the same domain.

The cluster generates the most named emergent/overlap nodes in the corpus:
- **EM02**: FEP6 (long-run equilibrium) is in tension with NV02 (M < N always)
- **EM03**: FEP7 (assumed graph structure) is grounded by NV01 (derived graph structure)
- **OV01**: FEP1 (bounded approximate posterior) overlaps with bounded context (EC01)
- **OV04**: SI4 (meta-cognitive awareness) overlaps with Definition 14 (higher-order sentience)

And one structural comparison: FEP4 (precision weighting) is the top-down architecture that Theorem 5 (Escalation Principle) uses as its foil — the architecture that fails under selection pressure.

---

## Rao-Ballard (1999): Predictive Coding as the Escalation Architecture

**RB1 (predictive coding)**: higher cortical areas send top-down predictions to lower areas; lower areas send bottom-up prediction errors. Only the error (surprise) propagates upward.

RB1 is the neural implementation of UB's escalation architecture (Theorem 2a). The top-down prediction is the L0/L1 crystallized expectation. The bottom-up error is the U(w,K) signal that something does not match. Only the error (the gradient signal) propagates up — exactly the escalation criterion: routine processing stays at L0; only prediction failures escalate to higher levels.

**Significance for Theorem 5**: Theorem 5 (Escalation Principle) argues that top-down precision allocation fails under selection pressure while bottom-up escalation succeeds. Rao and Ballard establish empirically that the cortex implements bottom-up error propagation (escalation), not top-down allocation. The cortex, shaped by evolution under selection pressure, uses the architecture that UB's Theorem 5 validates. This is empirical confirmation of the escalation principle's efficiency argument from neural implementation.

**RB2 (hierarchical generative model)**: each cortical level maintains a generative model of the level below. The model predicts the lower level's activity; the error is the unpredicted residual.

RB2 confirms EC03 (K/F graph = hierarchical generative model) empirically: the cortex uses the graph structure that NV01 proves necessary, not because someone designed it, but because gradient descent toward prediction error minimization converges to this structure (EM06). The hierarchical generative model is an empirically observed consequence of the same optimization that NV01 proves is necessary.

**RB3 (receptive field effects)**: context-dependent receptive field modulation is explained as top-down predictions subtracting the expected component. This is EC08 (compression factoring) at the neural level: the predictable part is factored out, leaving only the informative residual. The C_n constraint at each level is reduced by subtracting the predicted component.

---

## Friston (2010, 2017): FEP as a Neighboring Formal Framework

The free-energy principle (FEP) states that any self-organizing system that maintains its existence necessarily minimizes variational free energy — formally equivalent to bounding uncertainty about hidden world states.

**FEP1 (free energy minimization)**: agents minimize F = E_q[log q(s) - log p(o,s)], where q is the approximate posterior over hidden states and p is the generative model. F is an upper bound on surprise.

FEP1 overlaps with OV01 (bounded world model): both frameworks independently ground that agents operate under a bounded internal representation of W. FEP grounds this via the Markov blanket and approximate posterior; UB grounds it via C_n (bounded context window). The formal approaches are different but the structural claim is the same.

**FEP7 (assumed graph structure) → EM03 (graph necessity grounds FEP)**: the generative model in FEP is hierarchically (graph) structured by assumption. The necessity of this structure is not derived within FEP.

EM03 is the most important finding in the FEP cluster. NV01 (Theorem 2b Part 3) proves that graph structure is necessary for information preservation under bounded context. FEP7 assumes this structure without derivation. This means UB provides a grounding that FEP explicitly lacks: the justification for why the generative model must be hierarchically structured. EM03 is the node where UB's derivation fills a gap in FEP.

**FEP6 (long-run equilibrium) vs NV02 → EM02**: FEP states that free-energy minimization approaches an equilibrium where the generative model closely approximates the generative process. NV02 states M < N always holds (entities never fully sense W). These are in tension.

EM02 is the formal documentation of this tension. FEP implies the agent can in principle reach full knowledge in a stationary environment; UB proves it cannot. The resolution: FEP6's equilibrium requires a stationary environment; NV02 holds for any A > 0 entity with a non-trivially large world. In practice, environments are non-stationary for any entity with A > 0. The tension is real and only visible when both frameworks are held simultaneously — the kind of emergent clarity that cross-framework analysis produces.

**FEP8 (no A parameterization)**: engagement with the environment is defined by the Markov blanket — there is no separate parameter for engagement probability. The agent always minimizes free energy by definition.

FEP8 is the same gap as Schmidhuber S4. Both frameworks assume A = 1. NV03 (agency as gate) addresses this: A in [0,1] parameterizes the probability that a gradient signal triggers M engagement. The generalization is necessary for entities that sometimes fail to engage — the non-engagement cases that both Schmidhuber and FEP do not analyze.

**FEP4 (precision weighting)**: attention is formally precision — the confidence in prediction errors at each level. High precision = more influence on updating.

Theorem 5 (Escalation Principle) uses FEP4's precision allocation as the foil for the escalation argument. Top-down precision allocation is the architecture where higher levels control which prediction errors receive attention. UB proves this architecture fails under selection pressure: entities using top-down precision allocation are outcompeted by entities using bottom-up escalation (Rao-Ballard's cortical architecture). FEP4 is not wrong — it is a valid model of one attentional mechanism — but the selection pressure argument shows it is not the dominant architecture.

---

## Friston et al. (2021): Sophisticated Inference and Higher-Order Sentience

**SI4 (meta-cognitive awareness)**: sophisticated inference is structurally analogous to higher-order sentience — awareness of the inference process itself, not just its outputs. A sophisticated agent models not just the world but its own future inference states.

SI4 overlaps with Definition 14 (higher-order sentience): both describe an agent that models its own inference/normalization process. This is OV04. The two frameworks independently arrive at the same structural extension: agents that model their own modeling process have higher-order cognitive capabilities.

The key distinction: UB derives functional sentience from the information-theoretic structure (induction argument: if the entity can represent that other projections of W exist, it models W as larger than its K). FEP's sophisticated inference defines it as a richer inference architecture (the agent models its own future belief states). These are derivation versus definition — the same pattern as UB vs. IIT (Tononi) and UB vs. Schmidhuber, but with less structural distance.

**SI1 (self-modeling)** and **SI3 (counterfactual reasoning)** extend this: the sophisticated agent evaluates counterfactual policies by simulating what it would believe under each. This is the formal structure of what UB's higher-order sentience enables: an entity that can represent other projections of W can evaluate counterfactual futures.

---

## The UB-FEP Comparison: What Each Provides the Other

**What FEP provides to UB**:
- A formal, widely-adopted prior framework against which UB's claims can be positioned
- Neural evidence (via Rao-Ballard) that the escalation architecture is what evolution converged on
- The sophisticated inference extension (SI4) as independent grounding of functional sentience
- Precision weighting (FEP4) as an alternative attentional model that UB's Theorem 5 evaluates

**What UB provides to FEP**:
- Derivation of graph structure necessity (NV01 → EM03): FEP7 assumes what UB proves
- Formal resolution of FEP6/NV02 tension: M < N holds even when FEP would predict equilibrium
- A parameterization (NV03): the engagement probability that FEP's Markov blanket definition elides
- Selection pressure framing: why escalation (the empirically observed architecture) outcompetes precision allocation

---

## Summary

| Source | Key propositions | UB nodes | Notable contribution |
|---|---|---|---|
| Rao-Ballard 1999 | RB1-RB3 | EC09, EC03, EC08 | Predictive coding = escalation architecture; cortex implements bottom-up error (confirms Theorem 5) |
| Friston 2010 | FEP basics | OV01, T06 | FEP is the closest neighboring formal framework; precision allocation = Theorem 5 foil |
| Friston et al. 2017 | FEP1-FEP8 | EM02, EM03, OV01, NV03 | FEP7 assumes what NV01 proves; FEP6/NV02 tension; FEP8 same gap as Schmidhuber S4 |
| Friston et al. 2021 | SI1-SI4 | D14, OV04 | SI4 = independent FEP grounding of functional sentience (Definition 14) |
