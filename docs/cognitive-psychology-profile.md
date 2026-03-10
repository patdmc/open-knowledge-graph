# Cognitive Psychology and Consciousness Architecture: Cross-Framework Profile

**Cluster**: Miller (1956), Cowan (2001), Kahneman (2011), Baars (1988), Dehaene-Changeux (2011)
**Prepared by**: Patrick McCarthy
**Date**: 2025

---

## Overview

This cluster spans cognitive psychology (working memory limits), behavioral economics (dual-process cognition), and consciousness science (global workspace theory at cognitive and neural levels). What unites them: all five independently establish that information integration in cognitive systems is subject to a bounded capacity constraint, and all five describe the hierarchical and selective mechanisms the system uses to operate under that constraint.

This is the empirical cluster for the UB framework's C_n bound. The cluster does not independently derive why C_n exists (that requires information theory and formal logic). It establishes empirically that C_n is real, what its value is, how the encoding hierarchy that operates under it is structured, and how the parameter A is implemented neurally.

---

## Miller (1956): The First Empirical Measurement of C_n

**MI1 (working memory capacity)**: humans can hold 7 ± 2 chunks in immediate memory. This is the empirical bound on simultaneous active processing.

MI1 is the empirical measurement of C_n for the human cognitive system: C_n ≈ 7±2 chunks. Definition 7 (bounded context) is grounded here. The UB framework derives C_n formally as a constraint imposed by information theory; Miller measured it behaviorally in 1956.

**MI3 (capacity in Shannon terms)**: the limit is channel capacity in Shannon's sense — 7±2 bits per chunk, not per stimulus. Working memory is an information channel with bounded capacity.

MI3 is an independent empirical confirmation of EM04 (C_n = Shannon channel capacity). Miller (1956) explicitly uses Shannon's (1948) framework to interpret the 7±2 finding: the channel capacity framework is the right one for understanding working memory. This predates the UB framework by decades and establishes EM04 in the psychological literature.

**MI2 (chunking)**: the limit is on chunks, not bits. Information can be recoded into larger chunks to expand effective capacity. This is EC08 (compression factoring) observed in human behavior: recoding into larger chunks extracts shared structure and is the mechanism for operating under C_n. Spontaneous chunking under bounded context is empirical confirmation of the gradient-toward-graph prediction (EM06): humans, under selection pressure (cognitive load), construct graph-structured compressed representations.

**MI2 as potential EM**: MI2 + NV01 + EC08 could be documented as an emergent result (EM-Miller): humans spontaneously construct graph-structured chunks under C_n constraint, confirming the mechanism NV01 identifies. This is not currently a named emergent node.

---

## Cowan (2001): Refinement of C_n and the Slot Model

**CO1 (focus of attention)**: the focus of attention holds approximately 4 chunks, not 7±2. Miller's 7±2 conflated chunks with sub-chunks. The true capacity of the focus of attention is 4 items.

CO1 refines the empirical value of C_n. The key question for Definition 7: which capacity is the binding constraint? Cowan's embedded process model distinguishes two capacities: activated long-term memory (approximately 7±2 items available for retrieval) and the focus of attention (approximately 4 items currently integrated). UB's C_n is the focus-of-attention bound: the items currently operative in K for an inference step, not the broader set available for retrieval.

**CO2 (embedded process model)**: working memory is the activated portion of long-term memory plus the focus of attention. This maps to UB's encoding levels: activated long-term memory is L1 (available, chunked knowledge); the focus of attention is L2-L3 (active, contextual). The escalation architecture (Theorem 2a) is structurally consistent with CO2: routine items remain in activated LTM at L1; novel items requiring integration escalate to the focus of attention.

**CO3 (slot structure)**: the 4-slot limit is independent of item complexity — a single letter and a 10-letter word each occupy one slot. Slot structure maps to graph node structure. Each node in K occupies one slot regardless of how much it encodes. This means compression (factoring) directly determines W-coverage per slot: high-compression nodes (L0) provide more W-coverage per context slot than low-compression nodes (L3). The value of the factored representation is per-slot coverage, not per-item complexity.

---

## Kahneman (2011): Dual Process as the Encoding Hierarchy

**KH1 (dual process)**: cognition operates at two speeds. System 1 (fast, automatic) handles routine pattern matching; System 2 (slow, deliberate) handles novel reasoning.

KH1 maps to the encoding hierarchy: L0 handles certain, low-cost actions (System 1 analogous); higher levels handle uncertain, expensive inference (System 2 analogous). The two-speed architecture is the psychological observation of what the C_n constraint forces: the system must delegate routine actions below the context bound (automatic processing) to preserve context slots for novel integration.

**KH3 (finite attention)**: deliberate attention is limited and depletes. This is C_n at the psychological level: the formal bounded context window reflects the actual depletion of attentional resources. The formal and psychological treatments converge on the same constraint.

**KH4 (heuristics and biases)**: fast processing introduces systematic biases — patterns easy to process are treated as true. This corresponds to false positives in K: propositions retained with insufficient evidence (low C_1(p)). The provenance requirement in D13 is, in part, the formal mechanism for preventing the KH4 failure mode: a proposition is not retained in K unless its attribution, evidence, and derivation meet the threshold. System 1 heuristics correspond to propositions in K with unverified provenance.

**What Kahneman provides**: the two-system model gives the encoding hierarchy psychological reality. The C_n bound is not just a formal artifact or an engineering limit — it is the architectural reason a cognitive system develops two qualitatively different processing modes. UB derives why the hierarchy exists; Kahneman observes how it appears in behavior and documents the failure modes when it misaligns with evidence.

---

## Baars (1988): Global Workspace as K

**GWT2 (limited workspace)**: the global workspace has strictly limited capacity — only a small amount of information is globally available at once. This is C_n at the cognitive architecture level: the bounded context window is the workspace capacity. GWT2 provides architectural grounding for why C_n is a real constraint rather than a formal abstraction.

**GWT3 (modularity)**: specialized processors operate below the workspace threshold. This maps to the L0-L3 hierarchy: lower levels handle specialized processing (L0-L1 routine actions, specialized sub-systems); higher levels escalate to the workspace (L2-L3 novel integration). The modular architecture is the cognitive implementation of what the encoding hierarchy requires.

**GWT4 (broadcast as integration)**: the workspace integrates information from disparate sources. K is the shared knowledge structure that informs all of F: the globally broadcast workspace content is K. This is the architectural statement of the K/F coupling: everything in K is globally available to all processes in F.

**Baars and Dehaene together**: GWT is a cognitive-level theory (Baars) extended to a neural implementation (Dehaene). The two form a complete account of the C_n mechanism at cognitive and neural levels. Baars specifies what the workspace does; Dehaene specifies how it is implemented. UB's C_n is the formal bound on the workspace capacity both describe.

---

## Dehaene-Changeux (2011): Neural Evidence for A as a Probabilistic Gate

**NGW2 (threshold ignition)**: the workspace threshold is a non-linear ignition — below threshold, processing is local and unconscious; above threshold, global broadcast. This corresponds to the C_1(p) confidence threshold for retention in K. A proposition below the confidence threshold remains locally processed (not retained); above threshold, it enters global broadcast (is retained and available to F).

**NGW3 (attention as gating)**: attention gates entry into the workspace. High-attention items enter the global broadcast. This is the strongest neural evidence for A as a probability.

The neural mechanism is binary at the level of individual neurons (ignition or no ignition). But at the system level, the probability that a stimulus triggers ignition is continuous in [0,1] — it depends on attentional load, prior context, and stimulus strength. This is exactly Definition 6's A: A is not the force applied to a stimulus but the probability that the gradient signal triggers M engagement. The neural workspace confirms that this probabilistic formulation is the correct one: the attentional gate is stochastic at the system level even though deterministic at the unit level.

**NGW4 (predictive coding integration)**: the neural workspace is compatible with predictive coding. Conscious access occurs when prediction error is large enough to trigger global broadcast. This bridges GWT with the FEP cluster (Friston): when prediction error (U(w,K) in FEP terms) exceeds a threshold, the signal enters the workspace (triggers M engagement). NGW4 is the architectural connection between the cognitive psychology cluster and the predictive processing cluster.

---

## Cross-Cluster Convergence on C_n

Five independent traditions establish the C_n bound as a real constraint on cognitive systems:

| Source | Evidence type | C_n value | Mechanism |
|---|---|---|---|
| Miller MI1 (1956) | Behavioral experiment | 7±2 chunks | Working memory span |
| Miller MI3 (1956) | Information-theoretic analysis | Shannon channel capacity | Explicit use of Shannon framework |
| Cowan CO1 (2001) | Review and re-analysis | ~4 chunks (focus) | Slot capacity of focal attention |
| Kahneman KH3 (2011) | Behavioral evidence | Finite/depletable | Attention as resource |
| Baars GWT2 (1988) | Cognitive architecture | Small and bounded | Global workspace capacity |
| Dehaene NGW2 (2011) | Neural imaging | Threshold ignition | Non-linear workspace gating |

Six independent empirical observations from behavioral experiments, information theory, cognitive architecture, and neuroscience all converge on the same structural claim: cognitive systems have a bounded integration capacity that is approximately 4-7 units, approximately equal to Shannon channel capacity, and implemented as a limited-capacity global workspace with threshold gating.

This convergence is the strongest empirical evidence base for EC01 (bounded context) and EM04 (C_n = Shannon channel capacity) in the corpus.

---

## Summary

| Source | Key propositions | UB framework nodes | Notable contribution |
|---|---|---|---|
| Miller 1956 | MI1-MI3 | EC01, EM04 (empirical confirmation) | First empirical measurement of C_n; explicit Shannon grounding |
| Cowan 2001 | CO1-CO3 | EC01 (refined), EC09 | Slot model; focus of attention = operative C_n |
| Kahneman 2011 | KH1-KH4 | EC09, C_n, D13 provenance | Dual process = encoding hierarchy; KH4 = false K positives |
| Baars 1988 | GWT1-GWT4 | C_n, L0-L3 hierarchy, K/F | Global workspace = K broadcast to all F |
| Dehaene-Changeux 2011 | NGW1-NGW4 | A (Definition 6), C_1(p), NGW4 bridges FEP | Neural evidence A is probabilistic; ignition threshold = C_1(p) |
