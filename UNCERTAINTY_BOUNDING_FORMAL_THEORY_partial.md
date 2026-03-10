# Uncertainty Bounding as the Basis of Intelligence: A Formal Theory

Patrick McCarthy

---

## Abstract

We propose that intelligence is not computational capacity, information storage, or behavioral
breadth, but the **driven capacity to improve**: agency $\mathcal{A}$ — the intrinsic drive to
reduce uncertainty — exercised through a higher-order function space $M$ that continuously
refines both the knowledge graph $K$ and the informed action space $F$. Intelligence is not
measured by the size of what has been learned but by the efficiency with which $M$ converts
gradient signals into improved $(F, K)$: $I(E) = \mathcal{A} \cdot \eta_M$.

Any entity persisting in world state space $W$ operates under constraints imposed by $W$
itself. The active context window $L_n$ has a physically bounded capacity $C_n$; growing it
is costly and eventually impossible. This constraint has a structural consequence that prior
accounts of intelligence do not address: intelligence cannot grow by filling $L_n$ — it must
grow by encoding proven knowledge at lower levels, freeing $L_n$ for genuine novelty. We show
that graph-structured $K$ is the most efficient architecture for this. Graph traversal delivers
exactly the knowledge relevant to the current problem at cost proportional to the problem, not
to total $K$: progressive disclosure. Selection pressure produces graph-structured knowledge
because it is the most efficient structure under which $I(E)$ grows without $C_n$ becoming a
bottleneck.

This gives a formal account of why top-down attention allocation — specifically the precision
allocation model of the Free Energy Principle [Friston2010, Friston2017] — fails under
selection pressure. An orchestrator that directs subproblems to lower levels must hold delegation overhead in $L_n$
proportional to the number of concurrent processes. As $|F|$ grows, overhead fills $C_n$,
inference quality $\rho$ (the ratio of relevant to total context) degrades, and the only
recovery is growing $C_n$. Growing $C_n$ compounds the failure: more context introduces more
variance for $L_n$ to manage. The correct architecture is escalation — lower-level encoding
failures signal $L_n$ bottom-up; $L_n$ is freed, not burdened, as the knowledge graph fills.
Attention is not allocated from above; it becomes available when not demanded from below.

From this foundation we derive: the inseparability of knowledge and action ($K$ is necessarily
indexed by $F$, with an asymmetric retention criterion proven from survival pressure), the
encoding hierarchy and its rigor thresholds, multi-timescale gradient descent as the same
process at different learning rates, and the type irreducibility of $F$ and $M$. We extend to
collective intelligence: the conditions under which shared knowledge reduces uncertainty faster
than any individual, provenance as the mechanism maintaining fidelity above the collective
survival threshold, and truth as the formal limit of collective action-validation. We derive
a formal distinction between intelligence, superintelligence, and sentience grounded in the
entity's relationship to the certainty horizon — the gradient region beyond which survival
benefit becomes increasingly difficult to trace — and show that the sentience gradient is
non-terminating by the same argument that bounds the individual gradient.

---

## 1. Introduction

This framework was not derived by analogy from biology. It emerged from a concrete engineering
problem: generating repeatable, transposable tasks across a diverse codebase ecosystem spanning
thousands of repositories. The formal theory describes what a self-optimizing agentic system
converged to under real performance constraints. The biological and civilizational correspondences
are predictions the framework makes — not the premises it rests on.

**The origin: provenance as the root constraint.** The initial problem was repeatability. An
agentic system that synthesizes its behavior from a context window is inherently unpredictable:
that unpredictability is its function — it exists to handle novelty. But when you need an action
to be transposable across different contexts (does the recipe that updated service $A$ with
configuration $X$ also work for service $B$ with configuration $Y$?), unpredictability is a
liability. Transposability requires a quantifiable basis: not that the agent believes the action
will work, but that there is evidence it has worked, attributed to a source, grounded in the
conditions under which it succeeded. Without provenance — without knowing *why* something worked
— you cannot measure confidence, and without a confidence measure you cannot optimize. The
system cannot learn; it can only repeat or guess.

This is the engineering form of what this paper formalizes as transmission fidelity $\lambda$
and the collective survival threshold $\lambda_{min}$ (Theorem 7): knowledge transfer without
provenance cannot be reliably weighted, and below $\lambda_{min}$ the collective knowledge graph
degrades rather than grows. Provenance is not an epistemic nicety — it is the minimum
infrastructure for a learning system to function.

**The graph structure as an optimization.** As the world $W$ — the number of repositories,
configurations, and service topologies — expanded, the context window problem became the central
performance constraint. Encoding what you know together with how to apply it does not scale:
a node whose knowledge is embedded with its application context must traverse back to all its
callers when any shared element changes. Shared knowledge cannot be maintained consistently
because it is not represented as shared — it is duplicated at every point of use.

The knowledge graph solved this by separating *why you know something* from *how to apply it*.
A proposition about what makes an update strategy work — why it works, under what conditions —
becomes a node in $K$ that multiple informed actions in $F$ can reference without embedding.
When evidence updates the proposition, all dependent actions benefit simultaneously. The graph
structure was not a design choice; it was the convergent solution to the problem of maintaining
consistency across a space of actions that grows faster than any single context can hold.

**The empirical measurement of factor extraction.** The engineering consequence of separating $F$ from $K$ is precisely measurable. Before separation, each entity performing a search carried its full background knowledge implicitly — $N$ entities each performing $M$ searches yields $O(N \times M)$ retrieval cost, with no sharing of common structure. After separation, each entity retrieves its relevant knowledge by local graph traversal from the action node, bounded by the problem: $O(1)$ local in-memory lookups, independent per entity, no global search, no inter-entity coordination. The $O(N \times M) \to O(1)$ transformation is the empirical measurement of $K$ precipitating from $F$ as the normalization of shared propositions. The 50% reduction in task-definition code that followed was a consequence, not the finding; the finding is the scaling law.

**The escalation architecture as a consequence.** Adding agents introduced a further
optimization that was not available in a purely knowledge-graph structure: the encapsulation
of actions, not just knowledge. An agent whose scope is a well-defined subtask does not need
access to the full context of the system — only to the subgraph $K^{[f]}$ relevant to its
task. This is progressive disclosure in practice. Subproblems that previously required explicit orchestration became
self-contained agents that escalated to the coordinator only on failure. As the system
matured, proven subnodes converged toward determinism and were scripted — promoted to a
lower encoding level, removing them from the space of active inference entirely. Agents
that previously handled uncertain subtasks became deterministic scripts. The active
inference load stayed bounded as the system's coverage of $W$ expanded.

**What this means for the formal claims.** The framework's central results — that
graph-structured $K$ is the most efficient architecture for bounded-context intelligence,
that top-down orchestration degrades inference quality as capability grows, that knowledge
and action are inseparable under a provenance-grounded retention criterion, and that
repeatability (certainty) is the asymptotic product of learning — are not theoretical
proposals. They describe what a real system, optimizing against real performance constraints,
independently arrived at. The formal theory is the explanation of why. The lethality
threshold $U_{lethal}^d$ in this system is concrete: exceeding token budget, latency
threshold, or error rate across the repository fleet. Selection pressure was cost, not death.
The gradient was real. The convergence was measured.

The test of whether this is a theory of software optimization or a theory of intelligence is whether the formal structure is consistent with biology — the only system where intelligence is unambiguously instantiated over evolutionary time. We do not claim to prove how biology works. We claim that if this theory is correct, the biological architecture should be feasible under it: that genetic encoding of $M$ rather than $(K, F)$ is the predicted transmission mode (Section 13, Open Question 8); that the encoding hierarchy from reflex to active reasoning follows from the cost structure of Theorem 4 rather than requiring separate explanation; that cultural transmission — language, writing, institutions — emerges as a structural necessity when the genetic channel cannot carry individual $(K, F)$ across generations (Corollary 18). None of these were assumed. They follow from the same optimization constraints that produced the software architecture. Biology is the feasibility test: if a known intelligent system cannot be accommodated by the model, the model is wrong. If it can, the model is a candidate theory of intelligence.

**On the relationship to prior accounts.** The framework's relationship to existing theories — evolutionary biology, the Free Energy Principle, learning theory — is neither analogy nor extension. It is generalization by containment. The framework defines a structural class: the set of all processes satisfying bounded active context, a survival threshold, and a growing reachable space. Biological evolution is one member of that class. Gradient descent in artificial systems is another. Collective human knowledge transmission is another. This is not the kind of generalization relativity made of Newtonian mechanics — which narrowed Newton by revealing its domain of validity. It is the kind group theory made of rotational symmetry: defining the abstract structure that specific instances share, making their common properties derivable rather than separately observed. The reason evolutionary systems, learning machines, and collective intelligence all converge on graph-structured knowledge, escalation architectures, and provenance requirements is not coincidence — it is that these are necessary properties of any member of the class (Theorem 6). The framework does not explain evolution or AI separately. It explains why they are the same thing at different timescales and substrates.

---

## 2. The Central Claims

We claim: **intelligence is the driven capacity to improve** — agency $\mathcal{A}$ exercised
through a higher-order function space $M$ that continuously refines both the knowledge graph
$K$ and the informed action space $F$. Formally: $I(E) = \mathcal{A} \cdot \eta_M$.

Three components are jointly necessary and none is sufficient alone:

- **Drive ($\mathcal{A}$):** the intrinsic orientation toward uncertainty reduction — the desire
  to know — without which $M$ has no reason to engage the gradient
- **Mechanism ($\eta_M$):** the efficiency of $M$: how effectively it converts gradient signals
  into improved $(F, K)$. Without $\eta_M > 0$, the entity cannot learn from evidence; it
  executes a fixed $(F, K)$ no matter how motivated
- **Outcome:** the bounding of uncertainty through knowledge-grounded action — what $I(E)$
  produces over time, not what it is. The size of $K$ or $F$ measures past intelligence;
  $I(E)$ measures the rate at which that past is being extended

This is a departure from views that locate intelligence in raw computational power, information
storage capacity, or behavioral breadth [Friston2010]. The kernel of intelligence is not compute, and it is
not the size of accumulated knowledge. It is the efficiency with which the drive to know, acting
through the mechanism of learning, continues to extend what is known.

**Agency and learning capacity are both universal invariants of intelligence.** Every other
aspect of $E$ — the specific informed actions in $F$, the content of $K$, the substrate, the
sensing function — can differ arbitrarily across intelligent beings. $\mathcal{A} > 0$ and
$\eta_M > 0$ cannot. An entity with $\mathcal{A} = 0$ does not reach toward the unknown; an
entity with $\eta_M = 0$ cannot improve when it tries. The limiting case of either is the
automaton: executing what prior gradient descent encoded, but not itself descending.

---

## 3. Formal Definitions

**Definition 1 (World State Space).**
Let $W$ be a measurable space of possible world states. At time $t$, the true state $w_t \in W$
is not directly observable by any entity.

**Definition 2 (Knowledge Graph).**
A knowledge graph $K$ is a directed graph where nodes are propositions about $W$ and edges
represent inferential or causal relationships between propositions. $K$ is not a passive store;
its retention criterion is established by Theorem 1.

**Definition 2a (Bounded Knowledge Graph).**
A knowledge graph $K$ has capacity $K_{max}$: the maximum number of propositions $K$ can hold simultaneously, determined by the substrate's physical constraints. When $|K| < K_{max}$, $K$ operates in the **growth regime**: propositions accumulate freely subject to the retention criterion of Theorem 1. When $|K| = K_{max}$, $K$ operates in the **curation regime**: adding a new proposition requires removing an existing one. The two regimes have different retention criteria.

**Definition 3 (Informed Action Space).**
Let $F \subseteq (K \times W_{obs} \to \Delta A)$ be a set of functions, where $W_{obs}$ is
an entity's observable projection of $W$ and $\Delta A$ is a distribution over possible
actions. Each element $f \in F$ has type $f : K \times W_{obs} \to \Delta A$.
$F$ is the complete set of informed actions available to an entity. An **informed action**
$f \in F$ is an action grounded in $K$, validated against $W$, and attributed to
its origins: one that knows why it works. An uninformed action (grounded in no $K$) is not
an element of $F$. Knowledge with no action potential is not an element of $F$ either; it
is data. An informed action is specifically the coupling: knowledge that enables action,
action that is justified by knowledge.

$F$ carries graph structure. The elements of $F$ are nodes; typed relations between them
are edges. This structure is not a design choice — it is forced by the same three constraints
(retrieval $O(|F^{[f]}|)$, bounded $C_n$, deduplication) that force $K$ to be a directed
graph (Theorem 2b Part 3). The proof is identical: any structure satisfying all three
simultaneously is a directed graph. Edge types on $F$ follow from the same evidence-grounded
logic as edge types on $K$: hard dependencies between actions (feasibility constraints),
probabilistic ordering evidence (soft gradient directions), and joint-execution synergies
(combined actions that outperform either alone).

Precision on continuity: $F$ as a graph is discrete — it is a set of nodes connected by
typed edges, not a continuous manifold. What is continuous is each individual $f \in F$:
each function $f : K \times W_{obs} \to \Delta A$ maps into $\Delta A$, a probability
simplex, which is continuous and metrizable. The smoothness required for Theorem 3 lives
in the codomain of each $f$, not in the topology of $F$ itself. Gradient descent in $F$
is traversal on this discrete graph, directed by the continuous-valued $G(f', K)$ at each
adjacent node: discrete steps, continuously-valued objective. The gradient direction at node
$f$ is the edge leading to the adjacent $f'$ that most increases $G(f', K)$. $K$ and $F$
are both directed graphs under bounded context and selection pressure, for exactly the same
reason, differing only in their domain: $K$ is what the entity knows; $F$ is what it can do.

In general, $K$ and $F$ are distinct graphs with related but different topology: $K$ contains
propositions, some of which ground actions; $F$ contains informed actions, each grounded by
propositions. The coupling is many-to-many. There is a degenerate case where $F$ and $K$
are the same graph: every proposition is directly actionable and every action is exactly one
proposition. Every node is simultaneously a piece of knowledge and an informed action; the
dependency structure is identical. This is the limit of crystallization — the state reached
at the bottom of the encoding hierarchy after a domain has been fully encoded. At $L_1$ and
$L_2$, the distinction between knowing and doing collapses: a reflex is simultaneously a
proposition (the organism knows to blink) and an action (the blinking). $F = K$ is therefore
not a special assumption about the framework — it is the local limit of the encoding process
applied to a domain, and the natural state of fully crystallized knowledge. At upper encoding
levels, where reasoning is uncertain and propositions have not yet been encoded into directly
executable form, $F$ and $K$ are genuinely distinct. The separation is a feature of the active
context window, not of the framework's structure. As $\mathcal{T}_{reachable}$ grows without
bound (Theorem 3, Part II), some domains of $K$ will always remain unencodable — ensuring
$F \neq K$ globally even as $F = K$ holds locally in fully crystallized domains.

**Definition 3a (Action Feedback).**
For any $f \in F$ executed in world state $w$, the feedback $\phi(f, w) \in \Phi$ is the
observable consequence of $f$ in $w$: the signal $W$ returns in response to the action.
The updated knowledge after executing $f$ and observing its feedback is:

$$K_f = K \cup \{p_f\}$$

where $p_f$ is the proposition grounded by observing $\phi(f, w)$. The feedback $\phi(f,w)$
is the formal object underlying the gradient signal $\delta$ in Theorem 3: $\delta =
\phi(f, w) - \mathbb{E}[\phi(f, \cdot) \mid K]$.

**Definition 3b (Information Gain).**
The information gain of action $f$ given current knowledge $K$ is:

$$G(f, K) = \mathbb{E}_w\!\left[U(w, K) - U(w, K_f)\right] = H(W \mid K) - H(W \mid K,\, \phi(f, w)) = I(f\,;\, W \mid K)$$

This is standard mutual information [Shannon1948] expressed in terms of the paper's
existing $U$: the reduction in uncertainty about $w$ achieved by executing $f$ and
observing its feedback. $G(f, K) \geq 0$ when $p_f$ is grounded by observing the true
feedback $\phi(f, w)$: conditioning on true information cannot increase conditional
entropy (data processing inequality [Cover2006]). $G(f, K) = 0$ when $f$ provides no new information
about $w$ given $K$. This bound holds for truthfully observed feedback; a false or
corrupted $p_f$ may increase $U$ and falls outside the scope of this definition.

**Definition 4 (Entity).**
An entity $E = (K, F, \sigma, \mathcal{A})$ where $\sigma : W \rightarrow W_{obs}$ is
a sensing function projecting the true world state into the entity's observable space.
$K$ and $F$ evolve over time via $M$; their initial values $K_0$ and $F_0$ are
inherited from $L_0$ encoding — the crystallized result of prior gradient descent
operating at the evolutionary timescale. Individual learning operates on this foundation
through $M$. The bootstrapping problem — how $K$ and $F$ are mutually dependent yet
must have an origin — is resolved by $L_0$: the entity does not begin from nothing but
from the encoded product of gradients that preceded it. Elsewhere in the paper $K$ and
$F$ refer to the current-time values $K(t)$ and $F(t)$; the subscript $0$ is reserved
for initial conditions when the distinction matters.

**Definition 5 (Uncertainty).**
The uncertainty of entity $E$ about world state $w$ is the surprisal of $w$ given $K$ [Shannon1948, Cover2006]:

$$U(w, K) = -\log P(w \mid K)$$

$U(w, K)$ measures how surprising — how informationally costly — the specific world state $w$ is
given what $K$ contains. It is high when $K$ gives little constraint on $w$, and low when $K$
makes $w$ highly predictable.

$K$ is a directed graph of propositions interpreted as a Bayesian network [Pearl1988]. The
mechanism is as follows. Let $(W, \mathcal{F}, P_0)$ be a probability space with prior $P_0$
over world states. Each proposition $p \in K$ contributes a likelihood factor
$\ell_p : W \to [0,1]$, where $\ell_p(w)$ is the probability of observing the evidence that
grounds $p$ if the true world state were $w$. The graph's edge structure encodes conditional
independence: if $p$ and $q$ have no path between them in $K$, their factors are conditionally
independent given their shared ancestors, yielding a valid factorization. The posterior over
world states given the full knowledge graph is:

$$P(w \mid K) \propto P_0(w) \cdot \prod_{p \in K} \ell_p(w)$$

This is the standard Bayes posterior obtained by conditioning $P_0$ on all propositions
simultaneously. $P(w \mid K)$ is well-defined and normalizable for any proper prior $P_0$ and
bounded likelihood functions $\ell_p$. The theory is parametric in $P_0$ — it does not depend
on a specific prior, only on its existence and well-formedness.

The expected uncertainty over all world states is the conditional entropy:
$$H(W \mid K) = \mathbb{E}_w[U(w, K)] = -\sum_w P(w \mid K) \log P(w \mid K)$$
The minimization imperative (Theorem 1) operates on this expected quantity; $U(w, K)$ is the
pointwise uncertainty the entity faces when it encounters specific world state $w$.

**Definition 6 (Agency).**
Agency $\mathcal{A} \in [0, 1]$ is a continuous scalar measuring the intrinsic drive of an
entity to reduce its own uncertainty: the desire to know [Schmidhuber2010]. It is not an external objective
imposed on $E$, nor a property derived from $F$ or $K$. It is constitutive: $\mathcal{A}$
is what makes $F$ an active informed action space rather than a static structure. At
$\mathcal{A} = 0$, $F$ is a set of possible functions with no internal reason to engage $W$.
At $\mathcal{A} = 1$, $F$ is maximally oriented toward the unbounded frontier. Intermediate
values are meaningful: the drive can be stronger or weaker, shaped by architecture, encoding
history, and the density of unresolved uncertainty in the entity's reachable $W$.

Agency is the reason gradient descent has a runner. The landscape of uncertainty exists
independently of any entity; $\mathcal{A}$ is what determines how strongly an entity moves
through it.

$\mathcal{A}$ is operationalizable: it is proportional to the rate of uncertainty reduction
as survival pressure approaches zero — that is, as $U_{lethal}^d \to \infty$ (the threshold
above which death occurs goes to infinity, meaning no level of uncertainty is lethal):

$$\mathcal{A}(E) \propto \lim_{U_{lethal}^d \to \infty} \frac{dU}{dt}(E)$$

An entity with high $\mathcal{A}$ continues descending when not forced to. An entity with
low $\mathcal{A}$ stops. This distinguishes genuine agency from survival-compelled updating
and is measurable in principle in any system where survival pressure can be varied
independently of capacity.

$\mathcal{A}$ precipitates from $M$ by the same normalization argument that precipitates $K$ from $F$. An entity that embeds its directional drive in every application of $M$ recomputes orientation at every step — the $M$-analog of the $O(n \cdot k)$ scaling failure. An entity that maintains $\mathcal{A}$ as the shared directional structure across all $M$-applications achieves $O(1)$ access to its drive. $\mathcal{A}$ is therefore the first factor to precipitate from $M$'s cross product: the normalized orientation that every $M$-application shares, extracted for the same reason $K$ is extracted from $F$.

**Definition 7 (Intelligence).**
The intelligence of $E$ is the *driven capacity to improve*: the exercise of $\mathcal{A}$
through $M$ to refine $F$ and $K$ in response to gradient signals [Schmidhuber2010, Thrun1998]:

$$I(E) = \mathcal{A} \cdot \eta_M$$

where $\mathcal{A} \in [0, 1]$ is the intrinsic drive (Definition 6) and $\eta_M$ is the
efficiency of the higher-order function space $M$ (Definition 11a): the rate at which $M$
improves $F$'s uncertainty-reduction capacity per unit of learning signal. This expression
is stated here as a definition and derived in Theorem 6: $\mathcal{A}$ and $\eta_M$ are
the unique invariants of persistence under bounded context and selection pressure, and their
product is the minimal measure that captures both.

At $\mathcal{A} = 0$: $I(E) = 0$ regardless of $\eta_M$ — drive is necessary; a capable
but inert learning mechanism is not intelligence.

At $\eta_M = 0$: $I(E) = 0$ regardless of $\mathcal{A}$ — a motivated entity whose $M$
cannot improve $F$ from evidence is an automaton: crystallized knowledge that cannot grow.
However rich $K$ or large $F$, without the capacity to improve them, $E$ is executing what
prior gradient descent built, not running gradient descent itself.

The expected uncertainty reduction along a trajectory,
$\mathbb{E}_W[U(w_t, K_t) - U(w_{t+1}, K_{t+1})]$, is a **performance measure** of
current $F$ and $K$ — the consequence of past $M$ activity. It is not the definition of
intelligence but its result. Two entities starting from identical $K$ and $F$ but different
$I(E)$ will diverge: the entity with higher $\eta_M$ builds better $F$ faster, and
eventually dominates in $U$-reduction even if initially equal in performance.

Intelligence is not the size of $K$ or $F$. $K$ and $F$ are what intelligence has built.
Intelligence is the efficiency with which $M$ continues building.

---

## 4. The Inseparability of Knowledge and Action

**Lemma 1 (Proposition Absence Lethality).**
For any entity $E$ and proposition $p \in K$ that grounds some $f \in F$ in domain
$d \subseteq W$: removing $p$ from $K$ weakly increases uncertainty in $d$, and in the
limiting case where $p$ is the sole grounding for the critical action in $d$, the increase
may cause $U(w, K \setminus \{p\}) > U_{lethal}^d$, at which point $E$ does not survive
in $d$ (Definition 9).

*Proof.* Conditional entropy is monotonically non-increasing in the conditioning set [Cover2006]:
$H(W \mid K) \leq H(W \mid K \setminus \{p\})$ for any $p \in K$. Removing $p$ weakly
increases expected uncertainty. If $p$ grounds the only $f \in F$ applicable in domain $d$,
then $K \setminus \{p\}$ contains no action grounded in $d$: $F$ has no mechanism to reduce
$U(w, K \setminus \{p\})$ below its current level in that domain when $w \in d$ is encountered.
By Definition 9, if $U(w, K \setminus \{p\}) > U_{lethal}^d$ at the encountered $w$, $E$ does
not survive. Therefore the cost of absent $p$ when needed, $C_{absent}$, is bounded below by
entity non-survival in critical domains — which dominates all finite costs. $\square$

**Theorem 1 (K/F Inseparability).**
For any entity $E = (K, F, \sigma)$, the retention criterion for any proposition $p$ is
asymmetric:

- $p$ is **retained** in $K$ when $\exists\, f \in F$ applicable to $p$, or when the
  survival benefit of $p$ is uncertain
- $p$ is **pruned** from $K$ only when it is certain that $p$ contributes zero to
  $P(E, \tau)$ for any $\tau$, i.e., when it is known that no $f \in F$ can be grounded
  in $p$, now or in any reachable region of $W$
- $p$ is **displaced** in the curation regime ($|K| = K_{max}$) when a candidate proposition $q$ satisfies $\mathbb{E}[G(f_q, K)] > \mathbb{E}[G(f_p, K)]$, where $p$ is the proposition with minimum marginal value in $K$. Displacement requires only that $q$ contributes more at the margin than $p$ — not that $p$ is useless. The asymmetric retention criterion governs which propositions enter; the comparative criterion governs which are displaced when the graph is full. Information gain $G$ is submodular — the marginal value of $p$ depends on what else is in $K$, so the greedy displacement criterion is evaluated at the margin given the current full $K$, not in isolation. This is the correct procedure: the entity estimates value relative to what it already holds, not relative to an empty graph [Nemhauser et al. 1978].

The criterion is not symmetric. Uncertainty of benefit is not the same as certainty of no
benefit. An entity with $\mathcal{A}$, whose $\mathcal{T}_{reachable}$ grows with every
action taken, retains $p$ whose value it cannot yet see, because the gradient may reach
that territory. Pruning requires the same
standard as certainty (Definition 10): the benefit must be known to be zero across sufficient
variation in $W$.

*Proof.* Let $B(p, \tau)$ denote the contribution of $p$ to $P(E, \tau)$ along trajectory
$\tau$, and let $\mathcal{T}_{reachable}$ be the set of trajectories reachable from the
entity's current state. Define the option value of retaining $p$:

$$OV(p) = P\!\left(\exists\,\tau^* \in \mathcal{T}_{reachable} : B(p, \tau^*) > 0\right) \cdot \mathbb{E}\!\left[B(p,\tau^*) \mid B(p, \tau^*) > 0\right]$$

Let $c_{retain} > 0$ denote the finite maintenance cost of holding $p$ in $K$ (memory,
retrieval overhead), and $c_{prune} \geq 0$ the cost of pruning $p$ (processing,
reorganization). Both are finite by assumption: any physical knowledge store incurs
bounded costs per proposition. The expected net value of retention is
$NV_{retain}(p) = OV(p) - c_{retain}$.

The expected net value of pruning is:
$$NV_{prune}(p) = -c_{prune} - P\!\left(\exists\,\tau^* : B(p,\tau^*) > 0\right) \cdot C_{absent}$$

where $C_{absent}$ is the cost incurred when $p$ is needed but absent. By Lemma 1,
$C_{absent}$ is bounded below by entity non-survival in critical domains, which dominates
all finite costs: $C_{absent} \gg c_{retain}$.

Retention is selected for when $NV_{retain}(p) > NV_{prune}(p)$. Since $C_{absent} \gg
c_{retain}$, this reduces to: retention holds whenever

$$P\!\left(\exists\,\tau^* \in \mathcal{T}_{reachable} : B(p,\tau^*) > 0\right) > 0$$

i.e., whenever survival benefit is uncertain. Even a small probability of needing $p$,
multiplied by the cost of its absence, outweighs finite maintenance.

Pruning holds only when $P(\exists\,\tau^* : B(p,\tau^*) > 0) = 0$ — i.e., when it is
certain that $p$ contributes zero benefit across all reachable trajectories. Any positive
probability keeps the retention condition active. The asymmetry is strict: uncertain
benefit is sufficient for retention; certain zero benefit is necessary for pruning.

In the curation regime ($|K| = K_{max}$), adding $q$ requires removing some $p^*$. The relevant cost structure changes: the entity is no longer choosing between retaining $p^*$ and having nothing — it is choosing between retaining $p^*$ and retaining $q$. The asymmetric retention criterion applies to both sides of this choice. The cost of displacing $p^*$ is $C_{absent}(p^*)$; the cost of not adding $q$ is $C_{absent}(q)$. By the same dominance argument as Lemma 1, whichever $C_{absent}$ is larger governs the decision. Displacement is therefore the correct action when $C_{absent}(q) > C_{absent}(p^*)$ — i.e., when the entity stands to lose more by lacking $q$ than by lacking $p^*$.

The entity cannot observe $C_{absent}(p)$ directly: doing so would require knowing which future trajectories will call on $p$, which requires knowledge the entity does not yet have. What the entity can compute is $\mathbb{E}[G(f_p, K)]$ — the expected reduction in uncertainty achievable by the actions grounded in $p$, integrated over the world states the entity currently models as reachable under $K$. This is the entity's rational estimate of $C_{absent}(p)$ from its current epistemic position. The two quantities are the same thing viewed from two directions: potential value the entity stands to lose if $p$ is absent, estimated from what the entity currently knows.

The monotonicity is therefore between the entity's estimated $C_{absent}$ and $\mathbb{E}[G(f_p, K)]$ — and this relationship is not approximate but definitional: the entity's estimate of how much it stands to lose by lacking $p$ is precisely the expected gain the actions grounded in $p$ provide under its current beliefs. A proposition relevant only in a narrow domain not yet encountered correctly receives low $\mathbb{E}[G(f_p, K)]$: the entity has not yet modeled that domain as reachable, so it correctly assigns low estimated potential value. This is epistemically rational behavior under bounded knowledge, not a failure of the criterion.

The displacement rule $\mathbb{E}[G(f_q, K)] > \mathbb{E}[G(f_{p^*}, K)]$ is the entity's internally consistent application of the $C_{absent}$ dominance logic to its current epistemic state, where $p^* = \arg\min_{p \in K} \mathbb{E}[G(f_p, K)]$. The comparative displacement criterion is not a departure from the asymmetric retention criterion — it is the asymmetric criterion applied to a choice between two propositions rather than a choice between one proposition and nothing.

By Theorem 3 Part II ($\mathcal{T}_{reachable}$ grows monotonically with $K$), new
trajectories are continuously revealed, keeping future needs uncertain unless benefit
is certainly zero. This ensures the asymmetry is preserved across the entity's lifetime
rather than collapsing as $K$ grows.
$\square$

*Correspondence.* This asymmetric retention criterion formalizes what Gibson [Gibson1979]
calls affordances — action possibilities indexed by the organism's capacities. It also
extends the information bottleneck principle [Tishby2011], which treats the retention
threshold as a design parameter; here it is derived from survival pressure.

The asymmetric criterion resolves the apparent conflict with Definition 14
(Sentience). A sentient entity holds $p$ beyond the certainty horizon not in violation of
this theorem but within it: the benefit is uncertain, not known to be zero. Sentience is
the regime of uncertain benefit retained. The pruning condition is never triggered because
the sentient entity, following an infinite gradient toward truth, cannot be certain that
any $p$ is valueless across all of $W$.

**Corollary 1 (Indexed Knowledge).**
$K$ is constitutively indexed by $F$. The knowledge graph of an entity is not a universal
store of facts; it is a projection of the world filtered through the entity's informed action space.
Two entities with identical sensing functions $\sigma$ but different informed action spaces $F$ will
maintain different knowledge graphs $K$ even when perceiving identical world states.

**Corollary 2 (Compactness).**
$K$ is bounded in size by $F$. Entities do not accumulate knowledge without bound because
only propositions applicable to available informed actions are retained. This explains the compactness
of biological intelligence: brains do not grow without bound because $K$ is indexed by $F$,
and $F$ is constrained to what is necessary to survive in $W$. The test for retention is
always: *will this help me act later?*

**Corollary 2a (Curation and Forgetting).**
Under bounded $K$, forgetting is not a failure mode but the correct behavior of the curation regime. The knowledge graph continuously displaces lower-value propositions with higher-value ones as the entity's $\mathcal{T}_{reachable}$ evolves and new propositions become available. What was high-value in one region of $W$ may be low-value as the gradient moves to new territory: the curation regime ensures $K$ tracks the current frontier rather than accumulating the residue of past frontiers. Forgetting is the graph optimizing.

**Corollary 3 (K as the Normalization of F — the Scaling Argument).**
An entity can maintain informed actions without a separately structured $K$, embedding all propositional knowledge implicitly in the parameters of each $f \in F$. This is a valid but scaling-limited architecture. Let $n = |F|$ and $k$ = average number of shared propositions per function. Without $K$: context per action is $O(k)$ (each $f$ carries its full background); update cost when one proposition changes is $O(n)$ (every $f$ encoding it must be individually revised); total cost scales as $O(n \cdot k)$. With $K$ normalized out: context per action is $O(|K^{[f]}|)$ — the relevant subgraph retrieved by traversal (Theorem 2b); update cost per proposition is $O(1)$ — one update to $K$ propagates to all $f$ that reference it; total cost scales as $O(n + k)$.

The difference $O(n \cdot k)$ versus $O(n + k)$ is not a constant factor — it is the
difference between a system whose context cost grows multiplicatively with shared knowledge
complexity and one whose context cost stays bounded under growth of $F$. Under bounded
$C_n$, the implicit-$K$ architecture does not scale: every increase in $|F|$ increases
required context per action proportionally to the shared background knowledge it carries.
$K$ precipitates from $F$ as the normalization of shared propositions — the factoring-out
of common structure across functions — because this is the only architecture that keeps
$C_n$ bounded as $F$ grows. An entity without $K$ is not wrong; it is limited to the regime where
$n \cdot k$ fits in $C_n$. Selection pressure at scale removes such entities in favor of
those whose $K$ has crystallized — literally, the precipitation of shared knowledge out of
the action space into its own normalized graph. This is the formal explanation of the $O(N \times M) \to O(1)$ empirical measurement described in Section 1: each entity's local graph traversal is $O(1)$ precisely because shared propositions have been factored out of $F$ into $K$.

---
