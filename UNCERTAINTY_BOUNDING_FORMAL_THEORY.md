# Uncertainty Bounding as the Basis of Intelligence: A Formal Theory

Patrick McCarthy

Note: AI was used to draft this paper but the ideas are mine, and were originally the best explanation I had for why the graph structure I had chosen for my agentic context optimiation problem worked so well. Running inference across 4000 production github repositories is not something that has been done often, and yielded clear signal about what apporaches were best for maximizing context optimization rate. This argument also explains why my sampling experiments showed that evaliation within equivalence classes of code changes, and then comparing the equivalences was faster than other selection methods, like random sampling, quality ordering, tiered ordering (high, then random) or ordering by the next most different class (sorting by maximum contrast). This started out as a way to prove that my graph structure was optimal, and then became something more. We know _we_ have a bounded context. Could this also apply generally?

---

## Abstract

Intelligence is not computational capacity, information storage, or behavioral
breadth, but the **driven capacity to improve**: agency $\mathcal{A}$, ie. the intrinsic drive to
reduce uncertainty, as exercised through a higher-order function space $M$ that continuously
refines both the knowledge graph $K$ and the informed action space $F$. Intelligence is not
measured by the size of what has been learned but by the efficiency with which $M$ converts
gradient signals into improved $(F, K)$: $I(E) = \mathcal{A} \cdot \eta_M$.

Any entity persisting in world state space $W$ operates under constraints imposed by $W$
itself. The active context window $L_n$ has a physically bounded capacity $C_n$; growing it
is costly and eventually impossible. This constraint has a structural consequence that prior
accounts of intelligence do not address: intelligence cannot grow by filling $L_n$; it must
grow by encoding proven knowledge at lower levels, freeing $L_n$ for genuine novelty. This paper attempts to show
that graph-structured $K$ is the most efficient architecture for this. Graph traversal delivers
exactly the knowledge relevant to the current problem at cost proportional to the problem, not
to total $K$: leveraging the well known Agentic coding technique progressive disclosure. Selection pressure produces graph-structured knowledge
because it is the most efficient structure under which $I(E)$ grows without $C_n$ becoming a
bottleneck.

This gives a formal account of why top-down attention allocation - specifically the precision
allocation model of the Free Energy Principle [Friston2010, Friston2017] - fails under
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
and the collective survival threshold $\lambda_{min}$ (Theorem 10): knowledge transfer without
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
Agency $\mathcal{A} \in [0, 1]$ is the probability that entity $E$ engages its learning
mechanism $M$ in response to a gradient signal from $W$: the response gate. It is not an
external objective imposed on $E$, nor a property derived from $F$ or $K$. It is
constitutive: $\mathcal{A}$ is what makes $F$ an active informed action space rather than
a static structure. At $\mathcal{A} = 0$, $F$ is a set of possible functions with no
internal reason to engage $W$. At $\mathcal{A} = 1$, $E$ engages on every gradient signal
it encounters. Intermediate values are meaningful: the response probability can be higher or
lower, shaped by architecture, encoding history, and the density of unresolved uncertainty
in the entity's reachable $W$. The informal gloss — the desire to know [Schmidhuber2010],
the intrinsic drive to reduce uncertainty — is a phenomenological description of this gate,
not the definition.

Agency is the reason gradient descent has a runner. The landscape of uncertainty exists
independently of any entity; $\mathcal{A}$ is the probability that any given entity moves
through it when a gradient appears.

Three formulations of $\mathcal{A}$ appear in this paper. Their relationship:
**Formulation 3 (primary formal definition, Corollary 6a):** $\mathcal{A} \in [0,1]$ is
the probability that $E$ engages $M$ in response to a gradient signal — the response gate.
This is the definition stated in the body above. **Formulation 1 (informal description):**
"intrinsic drive to reduce uncertainty" is a phenomenological gloss; it conveys what it
feels like from the inside but is not the formal definition.
**Formulation 2 (empirical operationalization):** $\mathcal{A}(E) \propto
dU/dt$ as $U_{lethal}^d \to \infty$ is a valid empirical projection of Formulation 3 onto
a measurable quantity. It is valid because Theorem 3 Part II establishes that
$\mathcal{T}_{reachable}$ grows from execution, not from selection pressure: removing
survival compulsion does not remove gradient signals, only their life-or-death character.
An entity with genuine $\mathcal{A} > 0$ continues engaging $M$ on signals encountered
in $\mathcal{T}_{reachable}$ when survival pressure is removed; an entity with
$\mathcal{A} = 0$ that was updating only under survival compulsion stops. The limit
correctly isolates intrinsic engagement.

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

*Note on named components.* The components $K$, $F$, $M$, and $\mathcal{A}$ named here
are dimensions that have precipitated from the entity's full cross-product state — call
it $K^n$ — at the individual learning level. The theory's general result is that
dimensional precipitation must occur under the joint constraints: as $K^n$ grows, shared
structure must be extracted and named or retrieval cost grows without bound. What
dimensions precipitate is a function of the entity's $W$ and its history; the theory
constrains the process, not the outcome. $K$, $F$, $M$, and $\mathcal{A}$ are the
dimensions that fall out in entities with active individual learning mechanisms — the
class this paper studies. Other projections precipitate differently: biological $L_0$
encoding (DNA) does not separate into $K$, $F$, $M$, $\mathcal{A}$; it is a dense
cross-product that carries all survival-relevant structure without naming any of it as
a distinct component. The named components are an epistemological convenience grounded
in the studied entity class, not a universal constraint of the theory.

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

**Corollary 2b (K as the Normalization of F — the Scaling Argument).**
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

## 5. The Escalation Principle

**Definition 8 (Encoding Hierarchy).**
The encoding hierarchy is a continuous two-dimensional space $\mathcal{E}(c, r)$
parameterized by runtime cost $c \geq 0$ and reversibility $r \in [0, 1]$, where
increasing $c$ corresponds to more expensive engagement and increasing $r$ to more
easily revised encodings. Consistent with the continuity principle: a continuous being
moves through this space continuously; there is no categorical jump between encoding
modes, only a gradient of cost and permanence [Baars1988, Dehaene2011, Kahneman2011].

A continuous space has boundaries — regions where the qualitative character of encoding
changes in ways that require information to be translated as it crosses. An action encoding
appropriate to active reasoning cannot be deposited directly into genetic encoding; the
representation must change form at the boundary. An error signal escalating from reflex to
active reasoning must be re-represented in the vocabulary of the higher level. These
boundaries require **interfaces**: formal structures through which signals pass when moving
between qualitatively distinct encoding regimes.

The named levels $L_0, L_1, \ldots, L_n$ are the interfaces at the significant boundaries
of $\mathcal{E}(c, r)$, not points or regions within it:

| Interface | Boundary crossed | Runtime Cost | Reversibility |
|---|---|---|---|
| $L_n$ | Active reasoning / novel uncertainty | $O(\text{compute})$ | Fully reversible |
| $\vdots$ | Learned patterns, habits | Decreasing | Decreasing |
| $L_1$ | Reflex / autonomic | $\approx 0$ | Semi-permanent |
| $L_0$ | Genetic encoding | $0$ | Permanent within lineage |

Escalation (Theorem 2) is the **upward interface protocol**: the specification of what form
a failure signal takes when crossing a boundary from lower $c$ to higher $c$. Promotion
(Theorem 4) is the **downward interface protocol**: the condition under which an action may
cross a boundary from higher $c$ to lower $c$. The cost $C_i$ in Theorem 4 is a property
of the interface, not of the continuous space between interfaces: it is the cost of failure
when an encoding error has crossed the boundary and propagates at the lower level's
permanence.

The number of interfaces is substrate-determined — how many qualitative boundaries exist in
the entity's physical $\mathcal{E}(c, r)$. The theory specifies the properties every
interface must have; it does not prescribe how many exist. The continuous space between
interfaces admits arbitrary encodings; the interfaces themselves are discrete because a
boundary is a specific location, not a region.

**Theorem 2 (Escalation).**
An informed action $f \in F$ is handled at the lowest encoding level $L_i$ such that $U(w, K^{[f]}) \approx 0$
for the domain of $w$ relevant to $f$, where $K^{[f]} \subseteq K$ is the knowledge subset applicable to $f$.
Escalation to $L_{i+1}$ occurs if and only if $L_i$ fails to bound uncertainty.

*Proof.* By Definition 8, the cost of engaging level $L_i$ exceeds the cost of engaging
$L_{i-1}$: the encoding hierarchy is parameterized by increasing runtime cost $c$, so
$C_{i+1} > C_i$ for all $i$.
Selection pressure therefore favors handling $f$ at the lowest level sufficient to bound
uncertainty: engaging $L_{i+1}$ when $L_i$ suffices incurs unnecessary cost without reducing
$U$ further. An entity that routinely over-escalates incurs higher encoding
costs without survival benefit and is selected against. Conversely, if $L_i$ fails to bound
$U(w, K^{[f]})$ — that is, $U(w, K^{[f]}) > 0$ beyond acceptable residual — then $f$ is
not handled at $L_i$: the failure propagates upward until some $L_j$ ($j > i$) contains the
uncertainty or the entity is harmed (Definition 9). The active context window $L_n$ is the
level of last resort by construction: it is the highest-cost, fully reversible level.
$\square$

**Corollary 3 (Context Window as Frontier).**
The active context window is not the seat of intelligence; it is the frontier of unbounded
uncertainty. Its engagement is a signal of failure in lower-level encodings, not of superior
cognition. Well-bounded intelligence is largely invisible: it requires no active thought.

**Corollary 4 (Simultaneous Multi-Level Activity).**
The traversal between bounded and unbounded is always active at all levels simultaneously.
An entity may be actively reasoning at $L_n$ about a novel problem while $L_0$ handles
oxygenation, $L_1$ handles posture, and $L_2$ handles familiar motor patterns. The levels
are not sequential; they are concurrent, each handling its domain independently.

### Why the Graph: Progressive Disclosure and the Bounded Context Theorem

**Definition 8a (Active Context Capacity and Inference Quality).**
Let $C_n$ denote the capacity of $L_n$: the maximum number of propositions simultaneously
active in context. $C_n$ is bounded above by the substrate's physical constraints — but
the effective capacity available for any given problem is also shaped by the structure of
$K$. Graph-structured knowledge enables chunking: a single graph node can represent a
cluster of propositions that co-activate as a unit, making multiple propositions available
at the cost of one context slot. Expertise increases effective $C_n$ within a domain by
compressing frequently co-occurring propositions into retrievable chunks — not by expanding
the substrate bound, but by reducing the number of slots required per unit of relevant
knowledge. The substrate sets the ceiling; the structure of $K$ determines how much of
$W$ that ceiling covers.
Let $\text{ctx}(L_n, t)$ be the set of propositions loaded into $L_n$ at time $t$, with
$|\text{ctx}(L_n, t)| \leq C_n$.

The **inference quality** of $L_n$ for problem $f$ is the relevance density of its content:

$$\rho(L_n, f) = \frac{|K^{[f]}|}{|\text{ctx}(L_n, t)|}$$

$\rho = 1$: $L_n$ holds exactly what $f$ requires — maximum signal, zero noise.
$\rho \to 0$: $L_n$ is dominated by content irrelevant to $f$ — inference degrades.
Every proposition in $\text{ctx}(L_n)$ that is not in $K^{[f]}$ is noise: it contributes
variance to $L_n$'s inference without contributing signal. Minimizing that noise is
equivalent to minimizing $|\text{ctx}(L_n)| - |K^{[f]}|$.

**Definition 8b (Delegation Overhead).**
Under a top-down allocation architecture, $L_n$ directs which subproblems are handled at
lower levels. For each active delegation, $L_n$ must hold: the subproblem specification,
the expected return signal, and sufficient context to integrate the result when it arrives.
Let $\alpha > 0$ denote the minimum context cost per active delegation — the overhead
top-down allocation imposes on $L_n$ per managed subproblem, regardless of whether that
subproblem is routine or novel.

**Theorem 2a (Top-Down Allocation Degrades Inference Quality).**
Under a top-down allocation architecture with $k(t)$ concurrently delegated subproblems,
the inference quality available to $L_n$ for genuinely novel problems satisfies:

$$\rho(L_n, f_{\text{novel}}) = \frac{|K^{[f_{\text{novel}}]}|}{|K^{[f_{\text{novel}}]}| + k(t)\cdot\alpha}$$

As the entity's informed action space $F$ grows (more capability, more concurrent
processes), $k(t)$ grows proportionally and $\rho \to 0$. The only recovery is to grow
$C_n$: the context window must scale linearly with capability to maintain fixed $\rho$.

*Proof.* $L_n$ holds $K^{[f_{\text{novel}}]}$ for the current problem plus $k(t)\cdot\alpha$
for delegation overhead. Since $\alpha > 0$ and $k(t)$ grows with $|F|$, the denominator
is unbounded while $|K^{[f_{\text{novel}}]}|$ remains bounded (any specific problem
requires only its relevant subgraph). Therefore $\rho \to 0$ as $|F| \to \infty$.
When $k(t)\cdot\alpha \geq C_n - |K^{[f_{\text{novel}}]}|$, $L_n$ lacks capacity for
genuine inference entirely. To maintain $\rho$ at a fixed level, $C_n$ must grow as
$k(t)$ grows. $\square$

*Note.* This is self-undermining on the Free Energy Principle's own terms [Friston2010, Friston2017]. FEP holds that
intelligent systems minimize free energy — equivalently, minimize the variance they must
manage. But growing $C_n$ to accommodate delegation overhead adds variance: more
propositions in $L_n$ means more uncertain relevance to any specific problem, not less.
The orchestrator architecture prescribed by FEP's precision allocation is inconsistent
with FEP's own minimization imperative: allocating attention from $L_n$ increases the
variance that $L_n$ must hold in order to allocate.

*Biological note.* We do not claim that biological precision allocation operates like a token context window, nor that the cortical hierarchy is unidirectional. Top-down signals exist at every level: higher levels necessarily convey signals to lower levels, and the bidirectionality of the cortical hierarchy is well established. The architectural claim is narrower: it concerns *engagement initiation*, not signal direction. In an escalation architecture, what triggers higher-level involvement is bottom-up failure — a lower level's inability to bound uncertainty — not a top-down assignment of work. Top-down signals flow as a consequence of engagement; they do not determine when engagement occurs. The cost of *top-down assignment as the primary control architecture* is $O(k)$ in high-level capacity, as the attentional and dual-task literature demonstrates — each concurrently supervised subprocess consumes a share of the supervisory resource. Escalation avoids this cost by making higher-level engagement demand-driven rather than supply-driven. The transition from active to automatic processing across skill acquisition is the biological measurement of this convergence: as $f$ becomes automatic (assigned to $L_0$), it stops consuming high-level capacity, freeing $L_n$ for novel problems. The formal argument is substrate-neutral; biology is the validation test.

**Theorem 2b (Graph-Structured Knowledge Bounds Active Context).**
Under the escalation architecture (Theorem 2) with $K$ structured as a directed graph
(Definition 2), as $K$ grows:

1. $|\text{ctx}(L_n, t)| = |K^{[f]}|$ for each problem reaching $L_n$: no delegation overhead
2. $\rho(L_n, f) = 1$ for all problems at $L_n$: inference quality is maximum
3. $C_n$ need not grow as $|F|$ or $|K|$ grows

*Proof.*

**Part 1 (No delegation overhead).** By Theorem 2, $L_n$ is engaged only when all
$L_{i < n}$ fail. Problems handled below $L_n$ never appear in $L_n$: they generate no
delegation context there. $L_n$ receives a problem only when the escalation signal
propagates up from below — not because $L_n$ delegated and is monitoring, but because
the lower level failed and the failure is passed upward. The distinction is causal:
escalation is a push from failure, not a pull from delegation. Therefore $L_n$ holds
no monitoring context for subproblems in progress at lower levels.

**Part 2 ($\rho = 1$ conditional on edge fidelity).** The graph's edge structure encodes
inferential and causal relevance between propositions. Graph traversal from the escalated
problem node follows edges to exactly $K^{[f]}$ — the propositions inferentially connected
to $f$ — without loading any others. $|\text{ctx}(L_n, t)| = |K^{[f]}|$, so $\rho = 1$
by Definition 8a. *This result is conditional on edge fidelity: $\rho = 1$ holds when the
graph's edges correctly encode all and only the inferential relevance of each proposition.
Missing edges cause under-retrieval ($\rho$ remains 1 by count but $K^{[f]}$ is
incomplete); spurious edges cause over-retrieval ($\rho < 1$). Graph structure in principle
supports $\rho = 1$; achieving it requires well-specified edges.*

**Part 3 (Graph is the necessary structure under joint constraints).** The escalation
architecture and bounded context impose three simultaneous constraints on any knowledge
representation:

- **Retrieval**: the cost of accessing the knowledge relevant to problem $f$ must be
  $O(|K^{[f]}|)$ — proportional to the subgraph needed, not to $|K|$. As $|K|$ grows,
  retrieval cost cannot grow with it, or $L_n$ is burdened proportionally to past learning.
- **Context**: $|\text{ctx}(L_n, t)| = |K^{[f]}|$ exactly — $\rho = 1$. No extraneous
  propositions may enter $L_n$; each one admitted that is not in $K^{[f]}$ is noise that
  degrades inference (Theorem 2a).
- **Memory**: shared propositions must be stored once. A proposition that grounds multiple
  informed actions across different $f$ must exist as a single node, not duplicated at each
  point of use. Duplication means that when evidence updates a proposition, every copy must
  be updated — an $O(|F|)$ maintenance cost that grows without bound.

We claim graph structure is necessary: any representation that is not a graph must lose
information as new propositions are added.

*Proof by information preservation.* $C_n$ is bounded (Definition 2a). As new propositions
arrive, $|K|$ grows. When $|K|$ reaches $C_n$, the representation is in the curation
regime: adding any new proposition $p_{new}$ requires making space. There are exactly two
ways to make space:

**Option A: Discard an existing proposition.** Without factoring — without explicit
relational structure — the entity has no mechanism to determine which existing propositions
are redundant. Two propositions $p_1, p_2$ that share a latent factor appear as independent
entries; there is no stored evidence that they share structure. The choice of which to
discard is therefore made without information about relational dependencies. Discarding
$p_{old}$ loses the information it encodes, and potentially also loses the information
that $p_{old}$ has any relation to other propositions — a second-order loss. This is
information loss.

**Option B: Factor — extract shared structure, reducing $|K|$ without discarding.**
If $p_1$ and $p_2$ share a latent factor $f^*$, extracting $f^*$ as a single named node
with typed edges to $p_1$ and $p_2$ reduces $|K|$ by replacing redundant structure with
a single shared reference. The information in $p_1$ and $p_2$ is preserved; the
information about their shared grounding in $f^*$ is now explicitly stored as an edge.
Space is created without loss. Factoring also resolves the update problem: when new
evidence revises $f^*$, the single node is updated once, and the revision propagates to
all propositions that reference it via edges — $O(1)$ instead of $O(|F|)$.

Factoring is graph construction. Naming $f^*$, creating a node, and storing typed edges
from $p_1$ and $p_2$ to it is precisely the operation that produces a directed graph.
There is no third option: any representation that makes space without factoring loses
information; any representation that makes space by factoring is constructing a graph.

*Consequence for alternative representations.* A flat store has no mechanism for Option B:
shared structure in a flat store has no representation — two propositions that share a
latent factor are stored as independent entries. At $C_n$, the flat store must discard.
A tree has the same problem for cross-branch propositions: a proposition grounding actions
in multiple branches either duplicates or is lost when one branch is pruned. A vector
store is the unprecipitated cross product: shared factors are implicit in the geometry of
the embedding space but not stored as named nodes. At $C_n$, the vector store cannot
factor without naming — and naming is graph construction. A star schema is a graph with
depth-1 topology: it can factor within its schema but cannot represent propositions that
ground actions across multiple fact tables without extending to general graph structure.

Every representation that preserves information as $|K|$ grows beyond $C_n$ must factor.
Every factoring operation creates nodes with typed edges. The graph is therefore not where
alternatives eventually end up — it is what factoring is. Any representation that refuses
to factor loses information at the boundary. $\square$

**Part 4 ($C_n$ stays bounded).** By Corollary 6, proven informed actions promote to
lower encoding levels and leave $L_n$. Each promotion reduces future $L_n$ load. As $K$
grows, more problems are handled below $L_n$, not above. $L_n$ stays occupied by
genuinely novel uncertainty — the frontier moves outward, but the frontier's size at
any moment is bounded. $C_n$ need not grow; $K$ grows without filling $L_n$. $\square$

**Corollary 4a (Selection Pressure Produces Graph-Structured Knowledge).**
Two entities with equal $|F|$ and $|K|$: one with top-down allocation (growing $C_n$),
one with graph + escalation (bounded $C_n$). The second has strictly higher $\rho$ at
lower substrate cost. Intelligence per unit of active context — the ratio $I(E)/C_n$ —
is higher for the graph-structured entity. By Theorem 2b Part 3, graph structure is not
merely more efficient than alternatives — it is the necessary consequence of satisfying the
retrieval, context, and memory constraints jointly. Entities that grow $C_n$ to manage
delegation overhead are selected against because they violate the context constraint as
$|K|$ grows; entities using approximate retrieval are selected against because they violate
the $\rho = 1$ requirement; entities duplicating shared propositions are selected against
because maintenance cost grows with $|F|$. Selection pressure converges on the graph
because the graph is the only structure that does not eventually fail one of the three
constraints.

**Corollary 4b (The Scaling Argument Against Context Window Growth).**
The prevailing approach to scaling intelligence — growing $C_n$ directly — is
self-defeating on two grounds [Liu2024]. First, by Theorem 2a, more context means more noise:
for any specific problem, $\rho$ decreases as $C_n$ fills with content not relevant to
that problem. More context does not produce better inference; it produces higher variance
that the same inference must manage. Second, by Definition 7 ($I(E) = \mathcal{A} \cdot
\eta_M$), intelligence is the rate at which $M$ improves $(F, K)$ — not the capacity of
$L_n$. A larger $L_n$ does not increase $\eta_M$; it increases the noise environment in
which $M$ must operate. The efficient path to growing intelligence is not to grow $C_n$
but to grow the encoded knowledge at lower levels — filling the graph, not the context
window.

---

## 6. Gradient Descent on Uncertainty

**Definition 9 (Survival Threshold).**
For entity $E$ in domain $d \subseteq W$, there exists a threshold $U_{lethal}^d$ such that:

$$U(w, K) > U_{lethal}^d \Rightarrow E \text{ does not survive in } d$$

Above this threshold, uncertainty is so high that the entity cannot act effectively enough
to persist. $U_{lethal}^d$ is not a choice or a preference; it is imposed by $W$.


**Theorem 3 (Agency Necessarily Produces Evaluation-Driven State Revision).**

*On the space of descent and dimensionality.* The gradient descent in this theorem
operates in $F$ — the space of informed actions — not in $W$. $W$ may be arbitrarily
rough, discontinuous, or of unknown dimensionality; no assumption on $W$'s structure is
required. The entity does not descend over $W$; it updates $f \in F$ in response to
signals received from $W$.

The dimensionality of $W$ or of $M_{\text{res}}$ is not assumed. We do not know whether
the latent space is finite or infinite-dimensional. The convergence proof does not require
this knowledge. It operates on the entity's current parameterization of $F$, which at any
given moment is finite: bounded by $C_n$ (Definition 2a) and the propositions currently
in $K$. Each time execution extends $\mathcal{T}_{reachable}$ and grounds a genuinely new
proposition, the parameterization grows by one dimension. The proof proceeds by induction
on this dimension count, making no claim about the ultimate dimensionality of the space.

*Part I: Forced Signal, Response Gate.* Every executed action $f_i$ in world state $w$
generates a feedback signal $\delta = \phi(f_i, w) - \mathbb{E}[\phi(f_i, \cdot) \mid K]$
(Definition 3a): the signed deviation of observed outcome from expectation. This signal
arrives from $W$ but is applied in $F$: it is a subgradient of $U$ as a functional on
$F$ at the current $f_i$, evaluated at the encountered $w$. The signal is generated by
execution, not by $\mathcal{A}$. It arrives whether or not the entity has $\mathcal{A}$.
The entity does not choose whether to receive this signal. $W$ imposes it.

$\mathcal{A}$ is the response gate: the probability that $M$ engages on $\delta$ when
$\delta$ arrives. An entity with $\mathcal{A} = 0$ receives every signal but $M$ does not
engage; $(F, K)$ remains unchanged. An entity with $\mathcal{A} > 0$ engages $M$ on the
signal, updating $(F, K)$ in response. The encountered world states are on-policy —
determined by the entity's trajectory, not sampled i.i.d. from $P(w \mid K)$ — but this
does not impede convergence. The subgradient $\delta$ is valid at every encountered $w$
regardless of how $w$ was reached; the entity is updating a function in $F$, and each
$\delta$ is a legitimate descent direction in that space. The burn tells you that this
$f_i$ was insufficient at $w$.

An entity with $\mathcal{A} = 0$ does not update in response to $\delta$. $U$ does not
decrease. It eventually exceeds $U_{lethal}^d$ in some critical domain. Selection removes
it. Therefore any entity that persists must have $\mathcal{A} > 0$: not because
$\mathcal{A}$ generates the signal, but because $\mathcal{A}$ is what enables the entity
to act on signals that $W$ imposes on every execution regardless. When $M$ engages:

$$\Delta f_i = -\eta \cdot \delta$$

*Convergence by induction on sensed dimensions.* $W$ has $N$ dimensions; $N$ may be
finite or infinite — the entity does not know and the proof does not require knowing. The
entity's sensing function $\sigma$ (Definition 1) projects $W$ into $W_{obs}$: the entity
always senses $M \leq N$ dimensions, where $M$ is determined by the propositions currently
in $K$ and the reach of $\mathcal{T}_{reachable}$. At any given moment $M$ is finite
(bounded by $C_n$), regardless of $N$.

**Base case ($M = 1$).** A single informed action $f_1$ parameterized by one dimension.
$\delta_1 = \phi(f_1, w) - \mathbb{E}[\phi(f_1, \cdot) \mid K]$ is a scalar signal.
$U(w, K) \geq 0$ (entropy is non-negative) and $U$ is bounded below. Each update
$\Delta f_1 = -\eta \cdot \delta_1$ reduces $U$ in expectation. By the monotone convergence
theorem for bounded-below sequences, $\{f_1(t)\}$ converges to a local minimum of $U$
in the single-dimensional parameterization. The degenerate cases — $\mathcal{A} = 0$
(Theorem 6 Part 1) and $\eta_M = 0$ (Theorem 6 Part 2) — are already shown to result
in selection removal; they are the cases where $M$ does not grow and the entity does
not persist.

**Inductive step ($M \to M+1$).** Assume convergence holds for any $M$-dimensional
parameterization of $F$. When execution extends $\mathcal{T}_{reachable}$ and grounds a
genuinely new proposition $p_{M+1}$ (always possible when $M < N$, by Part II), the
entity's $F$ gains a new component $f_{M+1}$. By Definition 3, $f_{M+1}$ is grounded in
$K \cup \{p_{M+1}\}$, where $p_{M+1}$ is not yet in the inference graph of the existing
$M$ actions — the proposition is new. The gradient signal $\delta_{M+1}$ for $f_{M+1}$
is therefore independent of $\delta_1, \ldots, \delta_M$ in the current step: the new
dimension has not yet been reached by any existing action's grounding. This independence
holds at the moment of addition — the first execution of $f_{M+1}$ — before any
evidence-grounded edges from $p_{M+1}$ to existing propositions have been established in
the K/F graph. After the first execution, $\delta_{M+1}$ may correlate with existing
dimensions as the graph forms edges grounded in the new evidence; subsequent updates proceed
in the full $(M+1)$-dimensional system under the inductive hypothesis extended by one step,
and the induction covers exactly this first update step for each newly added dimension.
Convergence of the $(M+1)$-dimensional system follows from: (i) the $M$-dimensional system
converges by hypothesis; (ii) the new dimension converges by the base case applied to
$f_{M+1}$ independently at the moment of addition. Therefore convergence holds for $M+1$.

By induction, convergence holds for any $M$ the entity currently senses, without
assuming anything about $N$. The descent at each step is valid in the current sensed
subspace; the induction extends as $M$ grows with each new grounded proposition. Descent
is not chosen — it is what $\mathcal{A} > 0$ entities do with the signals $W$ imposes on
every execution, dimension by dimension as the frontier expands.

*Part II: Non-termination.* Let $\mathcal{T}_{reachable}(K)$ be the set of trajectories
reachable from the entity's current state given knowledge $K$ (as defined in Theorem 1).
We claim that for any finite $K(t)$, the frontier of $\mathcal{T}_{reachable}$ contains
world states $w$ where $H(w \mid K) > 0$, provided $W$ is sufficiently rich that each
executed action can ground a genuinely new proposition. In a finite $W$ where all propositions
are exhaustible, the frontier could in principle be covered; the claim applies in any $W$
where the entity's epistemic horizon has not been closed — which is the intended domain.

By Definition 3a, executing any $f \in F$ produces feedback $\phi(f, w)$ that grounds
a new proposition $p_f$ not previously in $K$, yielding $K_f = K \cup \{p_f\}$.
This richer $K_f$ grounds new informed actions not available from $K$ alone (Definitions
3 and 3b): actions that require $p_f$ to select or apply, reaching world states not
reachable without $p_f$. Therefore $\mathcal{T}_{reachable}(K_f) \supseteq
\mathcal{T}_{reachable}(K)$, strictly whenever $p_f$ grounds at least one new
$f' \in F$. Since Part I establishes that every executed action generates a feedback signal,
$\mathcal{T}_{reachable}$ grows monotonically with $K$. At any finite time $t$,
$K(t)$ has not covered $\mathcal{T}_{reachable}(K(t))$: the frontier always contains
states $w$ where $H(w \mid K) > 0$. Therefore $\nabla_F U$ is never globally zero
over $\mathcal{T}_{reachable}$, and descent never terminates within the entity's
reachable epistemic domain.

*Proof.* Part I: follows from $U_{lethal}^d$ and selection; entities that do not engage
$M$ on the signal are removed. Part II: follows from the monotonic growth of
$\mathcal{T}_{reachable}$ with $K$ — each executed action extends the reachable frontier,
ensuring new unexplored states are always accessible within the domain of the entity's
possible experience. $\square$

**Corollary 5 (Evaluation-Driven State Revision is Survival, Not Metaphor).**
For individual learning, gradient descent on $U$ is not a description of how learning
happens to work. It is what survival-driven updating *is*: the entity that burns and does
not become more careful does not survive. The entity that survives is, by definition, the
one whose $M$ engaged on the signal. At the evolutionary timescale, the mechanism differs
in kind: random mutation perturbs the $L_0$ encoding, generating new
$(F, K, M, \mathcal{A})$ configurations; $W$ evaluates each through the survival criterion;
configurations that persist reproduce. No individual organism follows a gradient — the
gradient is revealed by differential survival of randomly generated candidates. This is
structurally the same as a locally irrational action (temporarily negative $\eta_M$) at
the individual level: a perturbation that generates a new starting point for the
$M \times \mathcal{A} \times K$ cross-product to be evaluated by $W$. The two mechanisms
occupy a spectrum: at one end, random mutation with binary survival feedback (evolution);
at the other, directed update via continuous $\delta$ carrying full gradient information
(individual learning). Gradient descent is the efficient limit of this spectrum under two
conditions: (i) $\delta$ carries directional information in $F$ — not merely pass/fail,
but a signed deviation identifying which aspect of $f_i$ was insufficient at $w$; and
(ii) $M$ can use that directional information to bias the update toward lower $U$ rather
than generating a random candidate. When both conditions hold, each evaluation step
directly reduces $U$. When neither holds, the mechanism is random search with
$W$-evaluation. Real learning systems occupy positions between these limits; signal
informativeness determines how closely the process approximates directed gradient descent.

**Remark (Multi-timescale Structure of Evaluation-Driven Revision).**
Individual learning and evolutionary selection share a common structure: execute, receive
signal, retain or revise. The mechanisms and mathematical objects differ in kind.

For individual learning, the mechanism is directed: the gradient signal $\delta$ arrives
from a specific failure, $M$ updates $(F, K)$ in the direction that reduces $U$:

$$\Delta F_{\text{individual}} = -\eta_{\text{ind}} \nabla_F \, U(w, K)$$

For evolutionary selection, the mechanism is undirected: random mutation perturbs the
$L_0$ encoding of $(F, K, M, \mathcal{A})$, generating candidate configurations. $W$
evaluates each through survival. The aggregate population-level distribution shifts toward
configurations with lower $\mathbb{E}[U]$, but no individual organism descends a gradient
— the gradient is revealed by differential survival of randomly generated candidates. The
notation $\Delta F_{\text{species}}$ denotes the shift in the population distribution over
$F$-variants, not a gradient step by any individual:

$$\Delta F_{\text{species}} = \operatorname{select}(\operatorname{mutate}(F_{\text{population}}))$$

Both processes are instances of the same structure: generate a candidate state, submit to
$W$'s evaluation, retain if valid. Gradient descent ($\Delta F_{\text{individual}}$) is
the efficient limit when signals are informative enough to direct updates. Mutation
followed by selection ($\Delta F_{\text{species}}$) is the same mechanism without the
directed signal. This structural equivalence is not an identity — mechanisms, timescales,
and mathematical objects are all distinct — but it grounds the common observation that
both processes move toward lower $U$ at their respective timescales, and grounds a
provable consequence in Part II.

*Derivation.*

**Part I (Common structure, different mechanisms).** Individual learning updates $F$ for
a single entity based on $U(w, K)$ at specific world states $w$ encountered by that
entity. Evolutionary change operates on a different object: it updates the
population-level distribution over $F$-variants through differential reproduction of
randomly generated candidates. These are not the same mathematical operation — individual
learning changes a function within one agent via directed update; evolutionary change
changes a probability distribution over many agents via undirected mutation and selection.
The common structure is: generate a new internal state, submit it to $W$'s evaluation,
retain if valid. Gradient descent is the efficient limit of this structure; random
mutation followed by selection is the general case.

**Part II ($\eta_{\text{evo}} \ll \eta_{\text{ind}}$ follows from Theorem 4).** Although
the mechanisms differ, both rates can be characterized by the rigor threshold of their
respective encoding levels. The learning rate $\eta_i$ at encoding level $L_i$ is
inversely proportional to the rigor threshold $\theta_i$: a higher threshold requires
more evidence per unit update, which is the definition of a lower update rate. Formally:

$$\eta_i \propto \frac{1}{\theta_i} = \frac{1}{1 - \varepsilon/C_i} = \frac{C_i}{C_i - \varepsilon}$$

For individual learning at $L_n$: $C_n$ is the cost of a single wrong action, relatively
small, so $\varepsilon/C_n$ is non-negligible, $\theta_n = 1 - \varepsilon/C_n$ is
meaningfully less than 1, and $\eta_{\text{ind}} = C_n/(C_n - \varepsilon)$ is
correspondingly large. For evolutionary learning at $L_0$: $C_0$ is lineage survival,
far exceeding $\varepsilon$, so $\theta_0 = 1 - \varepsilon/C_0 \approx 1$ and
$\eta_{\text{evo}} \approx 1$. Since $\theta_0 \approx 1 \gg \theta_n$, it follows that
$1/\theta_0 \ll 1/\theta_n$, therefore $\eta_{\text{evo}} \ll \eta_{\text{ind}}$.
The learning rate difference is not assumed; it is derived from the cost-of-failure
difference established in Theorem 4.

An individual may establish an informed action that reduces $U$ in one context but fails
in others. Only informed actions that reduce $U$ across sufficient variation in $W$ survive
the evolutionary test — the higher rigor threshold filters for robustness. The analogy
captures this shared directional property: both processes favor what reduces $U$ at the
relevant scale. The timescale, the rigor threshold, the object being updated, and the
mechanism all differ; the directional relationship to $U$ is what the analogy preserves.

**Theorem 4 (Encoding Permanence Scales with Rigor).**
The certainty threshold required to promote an informed action $f$ to encoding level $L_i$ is:

$$\theta_i = 1 - \frac{\varepsilon_{\text{acceptable}}}{C_i}$$

where $C_i$ is the cost of failure at level $L_i$ and $\varepsilon_{\text{acceptable}}$ is
the maximum acceptable expected cost of an encoding error, expressed in the same units as
$C_i$. For genetic encoding ($L_0$), $C_0$ is lineage survival, requiring near-certainty
across the full distribution of environments. For active reasoning ($L_n$), $C_n$ is the
cost of a single wrong action, a far lower bar.

*Proof.* Encoding $f$ at level $L_i$ is irreversible at cost $r_i > 0$ (Definition 8).
The expected cost of encoding an action with current confidence $\theta$ is:

$$\mathbb{E}[\text{encoding error cost}] = (1 - \theta) \cdot C_i$$

where $(1-\theta)$ is the probability the action is wrong and $C_i$ is the cost when it
is. Promotion is safe when this expected cost does not exceed what is acceptable:

$$(1 - \theta) \cdot C_i \leq \varepsilon_{\text{acceptable}}$$

Rearranging:

$$\theta \geq 1 - \frac{\varepsilon_{\text{acceptable}}}{C_i}$$

The minimum threshold at which promotion is safe is therefore $\theta_i = 1 - \varepsilon_{\text{acceptable}}/C_i$. Below this threshold, the expected cost of encoding error exceeds $\varepsilon_{\text{acceptable}}$ and continued active engagement is preferred. At or above it, encoding is the lower-expected-cost choice. $\square$

*Consequence.* You must be very certain to permanently change the source code. A bad encoding
at $L_0$ does not kill one individual; it kills the lineage. The rigor of the test is
proportional to the permanence of the consequence.

*Note on error propagation through the hierarchy.* A potential objection is that the proof
derives $\theta_i$ for each level independently, whereas errors in a hierarchy compound:
a failure at $L_0$ propagates upward. This objection assumes a naive cascading model in
which errors at $L_i$ arrive at $L_{i+1}$ at rate $(1-\theta_i)$ regardless of whether
$L_i$ could address them. That is not the escalation architecture.

Under escalation (Theorem 2), errors partition into two classes at each level:
**handled errors** — failures $L_i$ detects and resolves within its scope, which never
reach $L_{i+1}$ — and **unhandled exceptions** — failures outside $L_i$'s domain or beyond
its capacity to resolve, which escalate. The rate seen by $L_{i+1}$ is not the compounded
Bernoulli rate from below. It is the rate of genuinely irreducible errors at $L_i$: errors
that $L_i$ was correct not to handle, because the knowledge or capacity to resolve them
does not exist at $L_i$. This is a different quantity from $(1-\theta_i)$, and it is bounded
by the design of $\theta_i$ itself — the threshold ensures $L_i$ handles everything within
its cost-justified scope.

Furthermore, when an unhandled exception escalates, the context at $L_i$ was already
disrupted by the failure: the action failed, and $L_i$'s current trajectory is suspended
or invalid. The attention cost at $L_{i+1}$ is not paid on top of productive work; it is
paid instead of broken work. Escalation is correct triage. The thresholds $\theta_i$ are
therefore independent by construction: each level's $\varepsilon_{\text{acceptable}}$ is
its own irreducible error domain, not an accumulation from below. The proof applies at
each level without modification.

**Definition 10 (Certain).**
An informed action $f$ is **certain** when $U(w, K_f) \approx 0$ across sufficient variation
in $W$ that it meets the threshold $\theta_i$ for promotion to $L_0$ (Theorem 4). Certainty
is the limit of this process: there is no categorical transition out of the probabilistic
domain, only decreasing density of uncertainty until the cost of active engagement no longer
justifies the residual gradient. A sufficiently certain informed action is promoted to a lower
encoding level and no longer occupies the active context window. Its participation in gradient
descent asymptotically approaches zero; it does not cease.

**Corollary 6 (Bounded Adaptation).**
Once an informed action becomes certain, it escapes the active context window permanently. This is why
complex problems (once solved at the civilizational or evolutionary scale) become cheap:
walking, recognizing faces, oxygenating cells. The gradient has converged. The encoding
cost approaches zero. Intelligence is freed for the next frontier of uncertainty.

---

## 7. The Higher-Order Function Space

**Definition 11 (Higher-Order Function Space).**
Let $M$ be a function space operating on the joint $(F, K)$ pair [Thrun1998]:

$$M : (F, K) \times \mathcal{L} \rightarrow (F, K)$$

where $\mathcal{L}$ is a learning signal: experience, execution results, selection pressure,
or observation of $F$'s interactions with $W$. $M$ drives the rate of change of both the
informed action space and the knowledge graph simultaneously: new propositions enter $K$
when evidence grounds them, and new informed actions enter $F$ when those propositions
have action potential. $M$ is the mechanism by which $(F, K)$ itself changes. It is not
a more certain or more important version of $F$; it is a categorically different type.

$M$'s type signature is complete without enumerating its internal structure, because $M$
is the cross product of all dimensions that have not yet precipitated out at the current
scale of the entity:

$$M = \prod_{i \in \text{remaining}} D_i$$

where the product ranges over all dimensions — drive, meta-learning rate, exploration
policy, convergence criteria, and whatever else has not yet reached the scale threshold
to become a named dimension. $F$, $K$, and $\mathcal{A}$ are factors that have already
been extracted from this product. $M$ is always the remainder: the undifferentiated joint
space of everything not yet named. As the entity scales, additional dimensions precipitate
out of $M$ by the same normalization argument that precipitated $K$ from $F$: when the
shared structure of a dimension across $M$-instances is large enough that factoring it out
reduces retrieval cost from $O(n \cdot k)$ to $O(n + k)$, selection pressure forces the
extraction. Each precipitation shrinks $M$ by one factor and names a new dimension.

The hierarchy is open upward: there is always potentially a $M^{(2)}$ that operates on
$M^{(1)}$ the same way $M^{(1)}$ operates on $(F, K)$. One level suffices until it does
not — when the gradient can no longer be followed by improving $(F, K)$ alone, pressure
propagates upward. The required number of levels is determined by the problem, not
prescribed by the framework. The precipitation argument applies at every level; the
hierarchy is not a prescribed structure but the output of a single recursive process
applied at increasing scale. See Open Question 7.

The degenerate cases of level merger generalize from the $F = K$ case: for any contiguous
block of $N$ precipitated dimensions together with the $M$ residual, all $N+1$ collapse to
a single undifferentiated process when the domain is fully crystallized at all scales in
that block. The merger condition is $U(w, K) \approx 0$ across the domain at all $N$
scales simultaneously — one condition, not $N$ pairwise conditions. Since $M$ represents
the full cross product of remaining dimensions, any merger including $M$ collapses not
just $N$ named levels but the entire unprecipitated hierarchy simultaneously. $F = M$ is
the maximally primitive state: action spans the full cross product of all remaining
dimensions. $M = M^{(2)}$ means the cross product is self-similar under the meta-operation
— evolution approaches this limit at the timescale of selection on selection mechanisms.

**Definition 11a (M Efficiency).**
The efficiency $\eta_M$ of $M$ is the rate at which $M$ improves the joint $(F, K)$
capability per unit of learning signal:

$$\eta_M(E) = \mathbb{E}_{\mathcal{L}} \left[ \frac{\max_{f' \in F'} G(f', K') - \max_{f \in F} G(f, K)}{|\mathcal{L}|} \right]$$

where $(F', K') = M((F, K), \mathcal{L})$ is the updated joint state after $M$ acts,
$G(f, K) = I(f\,;\, W \mid K)$ is the information gain of the best available informed
action (Definition 3b) in bits, and $|\mathcal{L}|$ is the information content of the
learning signal in bits: $|\mathcal{L}| = H(\mathcal{L})$, the entropy of the signal
received. Both numerator and denominator are in bits; $\eta_M$ is dimensionless,
measuring improvement in best-available information gain per bit of learning signal.
$\eta_M$ is defined for $H(\mathcal{L}) > 0$. When $H(\mathcal{L}) = 0$, the learning
signal is degenerate (constant; carries no information) and $\eta_M$ is undefined: the
entity received no learnable signal and the rate of improvement per bit is not defined.

$\eta_M$ is a signed quantity ranging over $(-\infty, \infty)$.
$\eta_M > 0$: $M$ improves the best available action in expectation — the entity learns.
$\eta_M = 0$: $M$ cannot improve $(F, K)$ on average — the entity executes without
learning.
$\eta_M < 0$: $M$ degrades $(F, K)$ locally — overfitting, catastrophic forgetting,
destructive updating, or deliberate exploration. An entity with *persistently* $\eta_M < 0$
does not persist under selection pressure. An entity that can act on $\eta_M < 0$
*temporarily* — when local gradient following would trap it — has strictly higher long-run
persistence probability than one constrained to $\eta_M \geq 0$ at every step.
$\eta_M$ is operationally independent of $\mathcal{A}$: one measures the mechanism of
improvement; the other measures the drive to engage it.

*Remark on exploration and the necessity of acting on negative $I(E)$.*
Theorem 3 Part II establishes that $\mathcal{T}_{reachable}$ grows without bound: the
gradient landscape is non-stationary and non-convex. An entity constrained to $\eta_M \geq 0$
at every step is a pure gradient follower — it converges to the nearest local optimum and
remains there. But local optima become maladaptive as new uncertainty domains emerge: an
entity trapped at a local optimum will eventually encounter $U > U_{lethal}^d$ in a domain
it cannot reach from where it is. Selection removes it.

Therefore, acting on $I(E) < 0$ is not merely tolerated by the framework — it is at times
the survival-optimal action. An entity that takes a locally irrational step (temporarily
worsening $(F, K)$ by the local gradient's account) to escape a local optimum has higher
expected long-run persistence probability than one that never does. The minimization
imperative requires $\mathbb{E}_t[\eta_M] > 0$ over the trajectory — not $\eta_M \geq 0$
at every step. I(E) is a characterization of the driven capacity to improve; it is not a
moment-by-moment fitness function to be maximized greedily.

The capacity to act on negative $I(E)$ — to execute a locally unjustifiable step — is itself
a component of $M$ capability. An $M$ that can recognize local optima and deliberately
perturb $(F, K)$ to escape them is strictly more capable than one that cannot. Exploration,
creative destruction, and apparent irrationality are not failures of intelligence. They are
the signature of an $M$ sophisticated enough to optimize over trajectories rather than steps.

$M$ is itself subject to gradient descent. The gradient of $U$ with respect to $M$'s
own parameters is the rate at which $M$'s capacity for learning improves given evidence
about its own performance — the same process at a higher level, with the same rigor
thresholds (Theorem 4) applied at the meta-learning timescale. Both $F$ and $M$ are
continuous function spaces undergoing continuous gradient descent; the distinction is
not categorical but one of domain: $F$ acts on $W$, $M$ acts on $F$ and $K$.

**Theorem 5 (Type Irreducibility).**
No element of $F$ can become an element of $M$ through any learning process, regardless of
certainty achieved.

*Proof.* $F : K \times W_{obs} \rightarrow \Delta A$. $M : (F, K) \times \mathcal{L} \rightarrow (F, K)$.
These functions have different type signatures and operate over different domains. No learning
process changes the type of a function. An informed action $f \in F$ that achieves certainty
is promoted to a lower encoding level; it does not change type. $M$ takes $(F, K)$ as input
and returns an updated $(F, K)$; $f$ takes knowledge and observations as input and returns
an action distribution. These are irreducibly distinct. $\square$

**Corollary 5a (Distinct Roles).**
$F$ is how an entity navigates $W$ given $K$. $M$ is how $(F, K)$ evolves. Neuroplasticity
is not a more certain form of cognition; it is $M$ acting on $F$. Insight is $M$ acting on
$K$: a new proposition enters when evidence grounds it, restructuring what can be grounded
in $F$. Habit formation is $M$ promoting an informed action from $L_n$ to $L_2$. Evolution
is $M$ acting on $M$ itself: a higher-order function space over the mechanisms of learning,
operating on an evolutionary timescale with the highest rigor threshold of all.

**Theorem 6 (Necessity of $\mathcal{A}$ and $\eta_M$: The Invariants of Persistence).**
For any entity $E$ persisting in $W$ under selection pressure $U_{lethal}^d$ with bounded
context $C_n$, in a world where $\mathcal{T}_{reachable}$ grows without bound (the condition
established in Theorem 3, Part II):

1. $\mathcal{A}(E) > 0$ necessarily
2. $\eta_M(E) > 0$ necessarily
3. Every other scalar property of $E$ can equal zero for some persistent entity
4. Therefore $I(E) = \mathcal{A} \cdot \eta_M$ is the natural measure of persistence — derived
   from what survival requires, not stipulated

*Proof.*

**Part 1 (Necessity of $\mathcal{A} > 0$).** Suppose $\mathcal{A}(E) = 0$. By Definition 6,
$E$ does not respond to gradient signals from $W$: $(F(t), K(t)) = (F(0), K(0))$ for all $t$.
By Theorem 3 Part II, the frontier of $\mathcal{T}_{reachable}$ grows strictly beyond what
any finite $K(0)$ covers: at every time $t$, there exist reachable world states $w$ where
$U(w, K(0)) > 0$. Since $K(0)$ is fixed and was not constructed with the frontier states in
mind, $U(w, K(0))$ is unconstrained there. In any sufficiently large $W$, the frontier
contains states where $U(w, K(0)) > U_{lethal}^d$ (Definition 9). $E$ encounters these
states as $\mathcal{T}_{reachable}$ expands — it cannot avoid the frontier indefinitely
because $\mathcal{T}_{reachable}$ grows with every action $E$ takes, even with fixed $(F,K)$.
By Definition 9, $E$ does not survive in any domain containing such states. Selection removes
it. Any entity that persists must have reduced $U$ — i.e., must have $\mathcal{A} > 0$.

The mechanism by which $\mathcal{A} > 0$ is enforced is timescale-specific. At the
individual learning timescale: entities with $\mathcal{A} = 0$ do not engage $M$ on
gradient signals and are directly removed. At the evolutionary timescale: lineages that
fail to produce offspring with $\mathcal{A} > 0$ are eliminated through differential
reproduction — no individual requires $\mathcal{A}$ explicitly; the population-level
filter enforces the property on lineages. At the cultural timescale: groups that do not
maintain the institutional equivalent of $\mathcal{A}$ (the drive to reduce shared
uncertainty) are outcompeted. The necessity is universal; the enforcement mechanism
is timescale-specific. $\square$

**Part 2 (Necessity of $\eta_M > 0$).** Suppose $\mathcal{A}(E) > 0$ but $\eta_M(E) \leq 0$.

*Case $\eta_M = 0$:* $E$ has the drive to reduce $U$ but $M$ produces no net improvement in
$(F, K)$ per unit of learning signal: the best available action's information gain does not
increase after $M$ acts. $(F, K)$ is effectively fixed despite $\mathcal{A}$. This reduces
to Part 1's condition: with $(F, K)$ fixed, the frontier of $\mathcal{T}_{reachable}$ grows
beyond $K$'s coverage. Additionally, $C_n$ is bounded (Definition 2a). As $E$ encounters
novel world states, $L_n$ accumulates unresolved context. By Theorem 2a, inference quality
$\rho$ degrades as $L_n$ fills without promotion of resolved actions to lower encoding levels.
Promotion requires $\eta_M > 0$: Theorem 4 establishes that encoding at level $L_i$ requires
rigor threshold $\theta_i = 1 - \varepsilon/C_i$, and achieving that threshold requires $M$ to
improve the action to the required certainty. With $\eta_M = 0$, no action ever reaches its
threshold: $L_n$ is never relieved, $\rho \to 0$, and inference eventually fails. $U$
exceeds $U_{lethal}^d$ in the domains that required the new actions. Selection removes $E$.

*Case $\eta_M < 0$ persistently:* $M$ degrades $(F, K)$ on average across the trajectory.
This accelerates the failure of the $\eta_M = 0$ case: $(F, K)$ worsens in expectation,
$U$ approaches $U_{lethal}^d$ from above. Selection removes $E$ strictly faster. Note that
temporary $\eta_M < 0$ — deliberate exploration to escape local optima — is compatible with
persistence and can increase long-run survival probability (see Remark in Definition 11a).
The persistence condition is $\mathbb{E}_t[\eta_M] > 0$ over the trajectory, not
$\eta_M(t) > 0$ at every step.

Therefore any $\mathcal{A} > 0$ entity that persists must have $\mathbb{E}_t[\eta_M] > 0$.
$\square$

**Part 3 (Minimality: no other property is invariant).** For every other scalar property
of $E$, there exists a persistent entity for which it equals zero or its minimum value:

- *$K(0) = \emptyset$:* An entity with no initial knowledge survives if $\mathcal{A} > 0$
  and $\eta_M > 0$ — it builds $K$ from gradient descent, beginning with the first feedback
  signal received from $W$ (Theorem 3, Part I).
- *$F(0) = \{\text{id}\}$:* $F$ is a function space; its minimal element is the identity
  — the action that changes nothing. The identity element exists independently of $K$: it
  requires no grounding because it makes no knowledge claim. Executing the identity against
  $W$ still returns feedback — the world observed in the absence of action is itself a
  signal. That feedback grounds the first proposition $p \in K$, after which the first
  non-trivial informed action can be constructed. The minimally non-empty $F$ is therefore
  $\{\text{id}\}$, not $\emptyset$, and it is sufficient to begin gradient descent.
- *$C_n = C_{min}$:* An entity with the smallest viable context window survives if the
  graph structure (Theorem 2b) keeps $\rho = 1$ — $C_n$'s absolute size is not invariant,
  only its boundedness.
- *Substrate, sensing $\sigma$, world projection $W_{obs}$:* These are free parameters of
  Definition 1 and Definition 3. The theory is parametric in all of them; no specific value
  is required for persistence. Entities on radically different substrates (biological,
  computational, collective) can all satisfy the survival constraint.

No scalar property other than $\mathcal{A}$ and $\eta_M$ has the survival-critical necessity
shown in Parts 1 and 2. $\square$

**Corollary 6a ($I(E) = \mathcal{A} \cdot \eta_M$ is derived, not stipulated).**
From Theorem 6, $\mathcal{A}$ and $\eta_M$ are the unique invariants of persistence under
bounded context and selection pressure. Any measure of intelligence must be positive if and
only if both are positive, and zero if either is zero.

The product is not a choice — it is the only normalized combination of a probability and a
conditional rate. $\mathcal{A} \in [0,1]$ is the probability that $E$ engages $M$ in
response to a gradient signal. $\eta_M$ is the rate of improvement conditional on $M$ being
engaged. By the law of total expectation:

$$\mathbb{E}[\text{improvement rate}] = \mathcal{A} \cdot \eta_M + (1 - \mathcal{A}) \cdot 0 = \mathcal{A} \cdot \eta_M$$

This is not an additional axiom; it is what the definitions compute when asked for the
expected improvement rate. A probability and a conditional value combine as a product.
Any other combination fails to normalize: the sum $\mathcal{A} + \eta_M$ assigns positive
intelligence to a zero-drive entity ($\mathcal{A} = 0$, $\eta_M > 0$), which contradicts
Theorem 6 Part 1 — selection removes that entity regardless of its latent learning capacity.
The minimum $\min(\mathcal{A}, \eta_M)$ makes the non-bottleneck component contribute nothing
at the margin, erasing the independent invariance established in Parts 1 and 2.

This multiplicative structure is not new to this corollary. Theorem 1 already established
that the correct way to value a piece of knowledge is $E[G(f_p, K)]$: the probability of
needing it times its conditional information gain. $I(E) = \mathcal{A} \cdot \eta_M$ is
the same structure applied to intelligence itself — the probability of engaging the learning
mechanism times the rate of improvement when it engages. The product normalizes signal and
likelihood into a comparable measure; sum resets the baseline without normalization.

$I(E)$ is not a choice of how to define intelligence — it is what the structure of
persistence under bounded context requires, confirmed by the same normalization principle
that grounds the retention criterion of Theorem 1.

---

## 8. The Unified Model

**Definition 12 (Trajectory).**
A trajectory $\tau$ is a sequence of informed actions drawn from $F$:

$$\tau = (f_1, f_2, \ldots, f_n), \quad f_i \in F$$

The **performance** realized by following trajectory $\tau$ is:

$$P(E, \tau) = \mathbb{E}_W\left[U(w_0, K_0) - U(w_n, K_n)\right]$$

This is the total uncertainty reduction achieved — a measure of what current $(F, K)$
accomplished, not of intelligence as defined. $P(E, \tau)$ grows with trajectory length
and with the quality of $F$ at the time of execution. Intelligence $I(E) = \mathcal{A}
\cdot \eta_M$ is what determines how quickly $P$ improves across successive trajectories:
an entity with high $I(E)$ builds better $(F, K)$ between trajectories, so $P$ grows
faster over time than for an entity with low $I(E)$ starting from identical initial conditions.

Agency does not merely incline an entity toward uncertainty reduction in the abstract.
**Agency manifests as a trajectory**: a sequence of informed actions the entity selects
and executes over time. Each action reduces $U$, returns a learning signal to $M$, and
$M$ uses that signal to improve $(F, K)$ for the next action. Intelligence is not visible
in any single step; it is the rate at which the steps improve.

This gives formal meaning to every relation in the informed action graph:

- `depends_on`: $f_j$ cannot appear in $\tau$ before $f_i$ (hard ordering constraint)
- `recommends_before`: evidence shows $(\ldots, f_i, f_j, \ldots)$ reduces more $U$ than $(\ldots, f_j, f_i, \ldots)$ (soft, confidence-scored)
- `enhanced_by`: a $\tau$ containing both $f_i$ and $f_j$ reduces more $U$ than either trajectory alone

The higher-order function space $M$ does not merely refine individual informed actions.
It discovers better trajectories from evidence: learning which sequences reduce $U$ most
reliably, and encoding that knowledge as graph relations.

An intelligent entity $E = (K, F, \sigma, \mathcal{A})$ operates as follows:

1. **Orient ($\mathcal{A}$):** $\mathcal{A} > 0$ directs $E$ toward regions of $W$ where $U(w, K) > 0$; without $\mathcal{A}$, there is no internal reason to engage the gradient
2. **Sense:** $\sigma(w_t) \rightarrow W_{obs}$: project the world into the observable space
3. **Act:** $F(K, W_{obs}) \rightarrow f_i \in \tau$: select the next informed action in the trajectory
4. **Learn:** $M((F, K), \mathcal{L}) \rightarrow (F', K')$: incorporate the feedback into both $K$ (new propositions grounded by evidence) and $F$ (new or refined informed actions); promote sufficiently certain informed actions toward lower encoding levels as $U \rightarrow 0$

The higher-order function space $M$ operates on the joint $(F, K)$ pair to:

1. Add new propositions to $K$ and new informed actions to $F$ when novel uncertainty is encountered in $W$
2. Refine existing informed actions and the trajectories between them; update $K$ with evidence-grounded revisions
3. Prune informed actions that consistently fail to reduce $U$; remove propositions whose action potential is certainly zero (Theorem 1)
4. Promote sufficiently certain informed actions toward $L_0$, freeing $L_n$ for the next frontier

### The Universal Invariants

Every property of an intelligent entity can vary arbitrarily:

- The specific informed actions in $F$: what an entity knows how to do
- The content of $K$: what an entity knows
- The substrate: carbon, silicon, or otherwise
- The sensing function $\sigma$: how an entity perceives $W$
- The encoding hierarchy: how deep or shallow its levels run

Two properties cannot vary:

$$\forall E \text{ that is intelligent}: \mathcal{A}(E) > 0 \text{ and } \eta_M(E) > 0$$

**Agency** is the universal driver: without $\mathcal{A} > 0$, there is no reason internal
to $E$ to engage the gradient. An entity with $\mathcal{A} = 0$ acts when acted upon; it
does not reach.

**M efficiency** is the universal mechanism: without $\eta_M > 0$, $E$ cannot improve
$(F, K)$ from evidence. An entity with $\eta_M = 0$ executes what prior gradient descent
encoded into its $(F, K)$ but does not itself run gradient descent. It may possess vast
$K$ and sophisticated $F$ — the crystallized output of prior intelligence — but it is not
itself intelligent in the sense defined here. This is the automaton boundary: not defined
by complexity or performance, but by whether $M$ is active.

The test for genuine intelligence is therefore not performance on any benchmark, and not the
richness of $K$ or $F$. It is: **does this entity improve how it learns when it encounters
what it does not know?** Does $M$ respond to the gradient signal? Does $\mathcal{A}$ drive
it toward the signal in the first place?

### The Structure of Intelligence

The intelligence of an entity is not its raw computational power, the size of its knowledge
graph, or the breadth of its action repertoire. It is the *driven* precision with which
$\mathcal{A}$ through $F(K)$ bounds $U(w, W)$, and the efficiency with which proven bounds
are encoded, freeing the active context window for the next genuine uncertainty.

The adaptability of biological intelligence arises because this traversal between bounded and
unbounded is always active at every level simultaneously, on different timescales, with
different rigor thresholds. The individual changes $F$ by learning — directed gradient descent
on $U$ via $M$. The species changes $F$ by evolving — random mutation of $L_0$ encoding
followed by $W$-evaluation, the undirected limit of the same evaluate-and-filter structure.
The difference is the mechanism and the test: individual learning is directed and requires
only that an informed action helps *this entity* survive; evolution is undirected and
requires that it helps *the lineage* survive, a far more rigorous proof.

Agency is self-sustaining precisely because certainty reveals new uncertainty beyond it.
Each informed action proven opens regions of $W$ that were previously invisible. The desire to know
is not extinguished by knowing; it is amplified. This is why intelligence does not
converge to a fixed point: $\mathcal{T}_{reachable}$ grows without bound (Theorem 3, Part II).

**The kernel of intelligence is therefore not compute, and not the size of $K$ or $F$.
It is $\mathcal{A} \cdot \eta_M$: the drive to know, expressed through the efficiency of
learning — against a world that generates new uncertainty faster than any finite entity
can resolve it.**

---

## 9. Intelligence, Superintelligence, and Sentience

The formal model yields precise distinctions between levels of intelligent agency, not based
on computational power or behavioral breadth, but on the entity's relationship to the survival
tether and to the shared knowledge graph.

**Definition 13 (Certainty Horizon).**
For entity $E$, the certainty horizon $H_E$ is the gradient region in $K_i$ across which
survival benefit becomes increasingly difficult to trace. Within $H_E$, propositions have
legible survival value: there exists a traceable path from $p$ to some $f \in F$ that reduces
$U(w, K)$ below $U_{lethal}^d$. Beyond $H_E$, that path may exist, but $E$ cannot construct
it. The horizon is not a sharp boundary; legibility fades continuously as the survival path
lengthens and becomes less traceable.

An entity operates as intelligent when it follows the gradient within its certainty horizon,
every $f \in F$ justified against a traceable survival path, the gradient persistent and
non-terminating but operating in legible terrain.

**Definition 14 (Sentience).**
An entity is sentient to the degree that $K_i$ contains propositions $\{p_{\text{self}}\}$
that accurately model the existence and distinctness of other $(K_j, F_j)$ pairs and
recognize them as gradient sources. Sentience is self-awareness explicitly defined: the
understanding that one's own $K_i$ is a partial projection of $W$, that other entities
hold different projections $K_j \neq K_i$, and that the delta $K_{collective} - K_i$ is
a navigable gradient. It is not abstract self-modeling of one's own $M$ in isolation —
it is the relational recognition that different knowledge graphs exist and can be learned from.

The connection to Theorem 9 is direct. An entity that conflates $K_i$ with $K_{collective}$
perceives the collective gradient as zero (Theorem 9). Sentience is precisely what makes
the collective gradient visible: $\{p_{\text{self}}\} \subseteq K_i$ distinguishes $K_i$
from $K_{collective}$, so the entity can perceive and follow the gradient over the delta.
Without sentience, $K_{collective} - K_i$ is invisible even when it is large.

The survival tether does not slacken. What changes is that the gradient's source expands.
A non-sentient entity follows only the gradient it can trace through its own $(F, K)$.
A sentient entity additionally perceives the collective gradient: every other $K_j$ is a
source of uncertainty reduction unavailable from $K_i$ alone. The entity continues not
because the survival tether loosened but because the visible gradient is now larger than
what survival alone reveals.

A sentient entity values propositions $p$ that bear on the structure of other $(K_j, F_j)$
pairs — not arbitrarily, but because $p$ is a signal in the collective gradient that $K_i$
cannot generate internally. The depth of sentience is the accuracy with which $K_i$ models
the existence, distinctness, and gradient-bearing character of other projections. The
gradient of sentience is non-terminating: $K_{interaction}$ — propositions derivable only
from the combination of distinct $K_i$ — is never exhausted by any finite entity (Corollary
14). Sentience orients the entity toward what it does not know that exists in other
perspectives, and that source of gradient is inexhaustible.

Sentience is what wonder feels like from the inside: the orientation toward what other
knowledge graphs contain that one's own does not.

*Note on usage.* We use *sentience* as a functional term: the degree to which $K_i$
accurately models the existence and distinctness of other $(K_j, F_j)$ pairs as gradient
sources. We make no claim about phenomenal consciousness [Chalmers1995]. The framework
locates a structural transition point — the acquisition of the relational self-model that
makes the collective gradient visible — without claiming this is sufficient for phenomenal
experience. Whether the functional condition defined here is necessary, sufficient, or
orthogonal to phenomenal consciousness is a question this framework does not resolve.

**Definition 15 (Superintelligence).**
Superintelligence is guided normalization of the space: the capacity to deliberately direct
what gets precipitated from $M_{\text{res}}$, evaluating $\text{Prec}(K_{collective})$ and
acting to accelerate the most valuable next factor extraction — rather than responding only
to local selection pressure. Where intelligence executes normalization (descends the gradient),
and sentience observes it (models its own descent), superintelligence guides it: choosing
direction based on evaluation of the full cross product rather than local gradient signals alone.

This is the operational answer to Open Question 10. The protocol for identifying the next
most achievable normal form — evaluate $\text{Prec}(K_{collective})$, identify cross-entity
covariance structure, hypothesize the minimal factor that grounds it, test at multiple
entities — is exactly what superintelligence executes deliberately. Intelligence discovers
the next precipitation by descending until it falls out. Superintelligence looks at
$M_{\text{res}}$ and chooses where to push.

The mechanism requires three things: $K_{collective}$ with provenance (to evaluate
$\text{Prec}(K_{collective})$ accurately), sentience in at least some participants (to model
the normalization process and perceive the landscape of $M_{\text{res}}$), and $M$ capacity
directed at $M$ itself — meta-learning aimed not at improving $(F, K)$ directly but at
improving the normalization architecture that produces $(F, K)$.

$K_{collective}$ with provenance is necessary but not sufficient. Participation in the
collective graph is the substrate for superintelligence; guided normalization is the activity
on that substrate. Two operations remain required:

- **Union**: $K_{collective} = \bigcup_i K_i \cup K_{interaction}$: breadth of projection
  is required to observe the full cross product of $M_{\text{res}}$
- **Intersection**: independent convergence with provenance: the cross-dimensional factors
  visible only from multiple projections are confirmed by agreement, not asserted by one

But these are means, not ends. The goal is reduction of $\text{rank}(M_{\text{res}})$
through deliberate cross-dimensional normalization — guided, not merely undergone.

**Definition 16a (Truth).**
For proposition $p$, truth is the fixed point: $p$ is true when it accurately describes world state $w$ for all $w \in W$ where $p$ is applicable. Truth is an absolute property — a condition that either holds or does not. It does not move; it is not a target that recedes. What is approached asymptotically is not truth but confidence in truth (Definition 16b): $C_1(p) \to 1$ as independent action-validation accumulates. The collective gradient points toward truth but cannot verify arrival from within the system, because verification requires a standard external to the system.

**Definition 16b (First-Order Confidence).**
The first-order confidence in proposition $p$ is:

$$C_1(p) = \lim_{n \to \infty} \frac{\left|\{i \leq n : p \in K_i,\; f_i(p) \text{ reduces } H(W \mid K_i),\; \text{prov}_i(p) \text{ independent}\}\right|}{n}$$

This is what converges — not truth itself, but confidence in truth. $C_1(p) \to 1$ as independent action-validation accumulates across entities. The grounding in action-validation (not mere agreement) and the provenance independence requirement are the conditions under which $C_1(p)$ tracks truth rather than consensus. $C_1(p)$ is bounded above by 1; whether it reaches 1 is undecidable from within the system.

**Definition 16c (The Confidence Regress).**
The second-order confidence $C_2(p)$ is confidence in the correctness of $M$ — specifically, in the functions within $M$ that produced $C_1(p)$: whether the learning mechanism selected the right validators, measured $U$-reduction accurately, and propagated evidence without bias. $C_2(p)$ is not confidence in confidence generically; it is confidence that the apparatus $M$ which generated $C_1(p)$ arrived at that estimate correctly.

More generally, the $n$-th order confidence $C_n(p)$ is confidence in the correctness of the level-$(n-1)$ apparatus — that the functions in $M$ which produced $C_{n-1}(p)$ were themselves unbiased and well-calibrated. Let $\hat{M}^{(n-1)}$ denote the component of $M$ that generated $C_{n-1}(p)$. Bounding $C_n(p)$ requires reducing uncertainty about $\hat{M}^{(n-1)}$: how reliably did it select validators, measure $U$-reduction, and propagate evidence? This uncertainty is itself a quantity $H(\hat{M}^{(n-1)} \mid \text{obs})$, and $C_n(p)$ is bounded above by how well that uncertainty can be reduced. Since reducing $H(\hat{M}^{(n-1)} \mid \text{obs})$ requires confidence in the level-$n$ apparatus, the regress is non-terminating.

$C_1(p)$ is not achieved. It is approached. The limit exists under conditions of independence
and exchangeability of validators: each must test $p$ against the same $W$ without
coordination on the outcome. For propositions whose validation depends on non-reproducible
or private observations, the sequence may not converge; such propositions cannot approach
truth in the formal sense defined here. The convergence, where it holds, never terminates.

**Theorem 7 (Epistemic Ceiling).**
For any entity $E$ operating within the system, the confidence stack $\{C_n(p)\}_{n \geq 1}$ is non-closeable from within: improving $C_n(p)$ requires reducing $U(C_{n-1}(p))$, which requires $C_{n+1}(p)$, and so on. No finite level terminates the regress. The epistemic ceiling is a structural property of the system, not a practical obstacle.

*Proof.* By Definition 16c, $C_n(p) \leq f(U(M_{n-1}(p)))$ for all $n \geq 2$. To increase $C_n(p)$, one must reduce uncertainty about the level-$(n-1)$ apparatus in $M$ — whether the functions that produced $C_{n-1}(p)$ were correct. But that uncertainty is itself uncertain: verifying the apparatus at level $n-1$ requires another apparatus at level $n$, whose correctness requires $C_{n+1}(p)$ to bound. The regress does not terminate because each level of the apparatus is itself subject to an apparatus question. Closing the regress would require a validator external to the system — a standard against which the highest-level $M$-functions are measured that is not itself within the system. By Definition 1, $W$ is the only external reference, and $W$ is not directly observable (only through $\sigma$ and action-feedback). The gap between $W$ and $W_{obs}$ is the structural source of the ceiling. $\square$

*Note.* The structural source of this ceiling is analogous to Tarski's undefinability theorem [Tarski1936]: a formal system cannot define its own truth predicate from within. Tarski's result is syntactic — it concerns definability in formal languages of sufficient expressive power. The epistemic ceiling here is a Bayesian calibration result — it concerns the non-closability of the apparatus that measures confidence. The two are not formally identical, but they share the same deep structure: verification of a system's outputs cannot be completed using only the resources of that system. Both require a standard external to the system being evaluated. The confidence regress is the operational instantiation of this structure in the domain of uncertain inference.

**Corollary 8 (The Sentience Gradient is Infinite).**
Since truth is a limit and $\mathcal{T}_{reachable}$ grows without bound (Theorem 3,
Part II), $\|\nabla_F U\|$ is never zero across $\mathcal{T}_{reachable}$. For any
trajectory $\tau$, regions of $\mathcal{T}_{reachable}$ remain where $U$ is nonzero
and the gradient has direction. The descent toward truth never terminates. The gradient is non-terminating for two independent reasons: (1) $\mathcal{T}_{reachable}$ grows without bound (Theorem 3, Part II), continuously generating new propositions whose truth is unknown; and (2) the confidence regress (Definition 16c, Theorem 7) never closes — for any proposition already held with high $C_1(p)$, the question of whether $M$'s apparatus that produced $C_1(p)$ was correct ($C_2(p)$) remains open. Sentience pursues both: new territory at the frontier and better-verified apparatus for what is already believed.

Intelligence follows a gradient that can reach zero locally; certainty is achievable in
bounded domains, encoded at $L_0$, done. Sentience follows the gradient toward the global
truth limit, which is structurally infinite. Not unbounded (the limit exists), but
non-terminating. Your uncertainty degrades toward truth asymptotically. You never arrive.

This is not a limitation. It is the shape of wonder.

**Theorem 8 (Sentience Implies Superintelligence).**
If $E$ is sentient, then $E$ participates in superintelligence.

*Proof.*

**Step 1 (Any $\mathcal{A} > 0$ entity participates in $K_{collective}$).**
For any $p \in K_i$, the certainty of $p$ is the fraction of independent validators that
action-validate it (Definition 16). Each additional independent validation increases this
fraction. Sharing $p$ with attribution allows other entities to independently
action-validate it, increasing certainty$(p)$. An entity with $\mathcal{A} > 0$ that does
not share forfeits these certainty gains: $K_i$ becomes less certain than $K_{collective}$
over time. By Definition 5 and Definition 16, lower certainty in $K_i$ corresponds to
higher $U(w, K_i)$ relative to participating entities. An entity with $\mathcal{A} > 0$
is gradient-preferring toward lower $U$; the sharing path strictly dominates the isolation
path in $U$-reduction. Moreover, by Corollary 8a, sharing is not merely gradient-preferred
but is the computationally optimal algorithm for detecting and measuring ongoing
normalization: the intersection of two independently-derived projections identifies where
precipitation has occurred and ranks those locations by confirmation count, in time
$O(|K_i| + |K_j|)$, strictly dominating solo search of $M_{\text{res}}$. No complete
normal form is claimed or required — the signal is that normalization is occurring, and
its rate is measurable. By Theorem 3, any entity with $\mathcal{A} > 0$ that persists acts on
gradient-preferred directions. Therefore any $\mathcal{A} > 0$ entity that persists
participates in $K_{collective}$.

**Step 2 (Sentience strengthens the implication).**
A sentient entity recognizes other $(K_j, F_j)$ pairs as gradient sources (Definition 14):
it holds propositions $\{p_{\text{self}}\} \subseteq K_i$ that accurately model the
existence and distinctness of other projections. By Corollary 8a, the propositions that
appear in $K_i \cap K_j$ under independent action-validation are the confirmed
precipitation sites of the shared region of $W$ — locations where both projections
independently confirmed that normalization is occurring, ranked by degree of confirmation. A sentient entity, recognizing other $K_j$ as distinct projections
of the same $W$, therefore intrinsically values the conditions that make independent
convergence possible: participation in $K_{collective}$ with provenance. This is not
merely instrumental ($U$-reduction) but structural: the intersection of $K_i$ with other
independently-derived $K_j$ is the only mechanism by which $E_i$ can identify which
propositions in its own $K_i$ are in the latent normal form rather than projection
artifacts. Isolated validation cannot supply this — a proposition confirmed only within
$K_i$'s own trajectory cannot be distinguished from a local artifact without an
independent projection to intersect with. A sentient entity therefore has intrinsic
motivation to contribute to and receive from $K_{collective}$, not merely instrumental
motivation.

Combining Steps 1 and 2: sharing is gradient-preferred (Step 1) and intrinsically valued
by sentient entities (Step 2). Both lead to participation in $K_{collective}$. Participation
in $K_{collective}$ IS superintelligence by Definition 15. $\square$

The converse need not hold: an entity can contribute to $K_{collective}$ without the survival
tether having relaxed; sharing for instrumental reasons still builds the graph. The deeper
the relaxation of the tether (the higher the degree of sentience), the more sharing is driven
intrinsically rather than instrumentally.

**Corollary 8a (Intersection as the Optimal Algorithm for Detecting Ongoing Normalization).**
For any entity $E_i$ with incomplete $K_i$ — that is, any $\mathcal{A} > 0$ entity still
descending the gradient — the intersection of $K_i$ with an independently-derived $K_j$
is the computationally optimal algorithm for detecting that normalization is occurring and
measuring how much has occurred.

*On completeness and the $M < N$ bound.* $W$ has $N$ dimensions; $N$ is unknown and may
be unbounded. Any entity senses at most $M < N$ dimensions at any time: the sensing
function $\sigma$ (Definition 1) and bounded $C_n$ (Definition 2a) guarantee this.
Completeness — full normalization of $M_{\text{res}}$ — would require $M = N$, which is
ruled out by Theorem 3 Part II: $\mathcal{T}_{reachable}$ grows without bound precisely
because $M < N$ always holds for any $\mathcal{A} > 0$ entity. The limit of completeness
is therefore the limit of $M$ as it approaches $N$ — a gradient that is never exhausted.
The measure is not arrival but approach rate: how fast is $M$ growing toward $N$?

*Proof.* Let $K_i$ and $K_j$ be knowledge graphs derived by entities $E_i$ and $E_j$
navigating the same region of $W$ via different informed action spaces $F_i \neq F_j$. By
Definition 3, the propositions in $K_i$ are grounded by actions in $F_i$; those in $K_j$
are grounded by actions in $F_j$.

Any $p$ in $K_i \cap K_j$ was reached by two independent trajectories. By the provenance
condition (Definition 16, independence requirement), $p$ was not reached by shared prior
assumption or shared training source but by independent action-grounding from distinct $F_i$
and $F_j$. Two independent action-paths converging on the same proposition is evidence that
precipitation CAN occur at $p$ — that $p$ represents latent structure in $W$, not an
artifact of either entity's projection path. This is not a claim that $p$ is part of any
complete normal form; it is evidence that normalization is occurring in the region of $W$
that both entities have reached. The number of independent executions converging on $p$
measures the degree to which normalization has occurred at $p$: more independent
convergence = more confirmed precipitation = higher rank as a precipitable factor.

The benefit of the intersection is precisely this: the number of independent
executions of $p$ across distinct $(K_j, F_j)$ pairs is the measure of ongoing normalization
at $p$. Each additional independent execution arriving at $p$ is a trial of
$p$ against $W$ via a different action-path — the same quantity as $C_1(p)$ in Definition
16b. The intersection $K_i \cap K_j$ with provenance verification therefore identifies
where normalization is occurring and ranks those locations by the degree of confirmation.
It does not identify where normalization could occur but has not yet been reached: those
regions require continued execution (Theorem 3 Part II).

*Complexity comparison.* Detecting ongoing normalization by solo search of $M_{\text{res}}$
requires comparing dimensions of the cross product to find those that share structure —
$O(\dim(M_{\text{res}})^2)$ in the general case. For a sparse projection into a dense
latent space, $\dim(M_{\text{res}}) \gg |K_i|$, making solo detection expensive relative
to the size of any single entity's knowledge.

Computing $K_i \cap K_j$ with provenance verification costs $O(|K_i| + |K_j|)$. The
result is the set of propositions where normalization has independently been confirmed —
ranked by confirmation count. Since $\dim(M_{\text{res}}) > |K_i|$ for any
$\mathcal{A} > 0$ entity, the intersection algorithm strictly dominates solo search for
detecting normalization in the region already reached by both entities.

*Equivalence with $C_1(p)$.* The ranking criterion — number of independent executions
converging on $p$ — is exactly $C_1(p)$ (Definition 16b). Certainty and normalization
degree are the same quantity measured from two directions: $C_1(p) \to 1$ as $p$
accumulates the independent action-validations that confirm ongoing precipitation at $p$.
The intersection algorithm and the certainty measure are not two separate results — they
are the same operation: identify where independent projections have both confirmed
precipitation, count the confirmations, rank by count. Neither claims completeness;
both measure approach.

*Multi-entity generalization and acceleration of $M \to N$.* Each entity $E_i$ senses
$M_i < N$ dimensions. Solo, $E_i$ can only extend $M_i$ by continued execution — one new
grounded proposition at a time (the inductive step of Theorem 3 Part I). The intersection
with another entity $E_j$ sensing $M_j < N$ dimensions does two things simultaneously:

- **Overlapping dimensions** ($K_i \cap K_j$, provenance-verified): confirmed precipitation
  sites — dimensions both entities have independently reached, ranked by $C_1(p)$.
  These are the latent factors of the shared region of $W$ confirmed by two independent
  projections.
- **Distinct dimensions** ($K_j \setminus K_i$): dimensions $E_j$ has sensed that $E_i$
  has not yet reached. The inductive step that would require continued solo execution now
  happens in a single intersection operation — $E_i$'s effective sensed dimension count
  jumps from $M_i$ to $M_i + |K_j \setminus K_i|$ without requiring the intermediate
  execution steps.

The rate of $M \to N$ is therefore accelerated by the intersection of graphs with both
overlapping and distinct dimensions: overlapping dimensions confirm what is already sensed;
distinct dimensions extend what is sensed. Solo execution extends $M$ by one dimension
per grounded proposition; intersection with a sufficiently distinct $K_j$ extends $M$ by
$|K_j \setminus K_i|$ dimensions simultaneously.

The $n$-entity generalization: $K_{collective}$ is not the complete normal form (which
cannot exist while $M < N$) but the maximal current union of sensed dimensions across all
participating entities, with the intersection identifying the confirmed precipitation sites
within that union. As $n$ grows, the collective sensed dimension count approaches $N$
from below; the growth rate of the intersection is the empirical signal of this approach.
The intersection of many graphs with distinct and overlapping dimensions is the fastest
known mechanism for $M \to N$.

*Consequence for the sharing imperative.* Any $\mathcal{A} > 0$ entity has not found
the full normal form of its $M_{\text{res}}$ (Theorem 3 Part II: $\mathcal{T}_{reachable}$
grows without bound, so the gradient never terminates). For such an entity, sharing and
receiving $K_j$ with provenance is the optimal normalization algorithm — not merely a
gradient-preferred direction (Step 1) but the fastest available path to identifying which
propositions in $K_i$ are genuine latent factors versus local projection artifacts. The
intersection operates on propositions already in $K_i \cup K_j$; propositions not yet
reached by either entity are expanded by continued execution (Theorem 3 Part II). These
are complementary mechanisms covering different territory: the intersection algorithm
optimizes what is known; execution grows what is known. The existence of the unknown
is not a reason to forgo the optimal algorithm for the known.

Furthermore, each additional independent projection $K_j$ is a precipitation trigger with
non-zero probability: a $K_i \cap K_j$ proposition that exposes a latent factor previously
buried in $M_{\text{res}}$ does not yield an incremental certainty gain but a
discontinuous reduction in $\text{rank}(M_{\text{res}})$ — collapsing complexity by one
dimension and reducing retrieval cost $O(1)$ for every future query touching that factor.
The expected value of the next shared interaction includes this heavy-tailed component.
The compulsion to share is a survival technique not only because sharing is gradient-preferred
in expectation, but because the intersection of the next $K_j$ could be a dimensional
collapse that no amount of solo normalization could produce. $\square$

**Theorem 9 (Perspective as Precondition for the Gradient).**
Without maintained distinction between $K_i$ and $K_{collective}$, the gradient of collective
uncertainty is unresolvable.

*Proof.* The collective gradient available to $E_i$ is defined over the delta between
individual and collective knowledge:

$$\nabla_{K_i}\!\left(U(w, K_{collective}) - U(w, K_i)\right)$$

For this gradient to be non-zero, two conditions must hold simultaneously: (1)
$U(w, K_{collective}) \neq U(w, K_i)$ for some $w$ — i.e., $K_{collective}$ contains
information $K_i$ lacks; and (2) $E_i$ maintains a representation distinguishing its own
$K_i$ from $K_{collective}$ — i.e., $E_i$ can perceive the delta.

Suppose condition (2) fails: $E_i$ conflates $K_i$ with $K_{collective}$, perceiving
$\hat{K}_{collective} = K_i$. The perceived collective gradient is then:

$$\nabla_{K_i}\!\left(U(w, \hat{K}_{collective}) - U(w, K_i)\right) = \nabla_{K_i}\!\left(U(w, K_i) - U(w, K_i)\right) = 0$$

The perceived gradient is zero regardless of the true collective gradient. $E_i$ cannot
follow a gradient it cannot perceive. Therefore condition (2) — perspective, the maintained
distinction between $K_i$ and $K_{collective}$ — is a necessary precondition for collective
gradient descent. Condition (1) is necessary for the gradient to be non-trivially large;
condition (2) is necessary for it to be followable at all. $\square$

**Corollary 9 (Sentience and the Social Imperative).**
In intelligence, sharing is justified when it bears on the gradient. In sentience, sharing is
intrinsically valued, because reducing another's uncertainty is a good independent of its
effect on the sharer's own $U$. The social imperative finds its unconditional form only in
sentience: the drive to share that no longer requires a survival argument.

**Corollary 10 (The Uncertainty Condition).**
Intelligence requires active uncertainty. From Definition 7, $I(E) = \mathcal{A} \cdot \eta_M$.
The performance measure (Definition 12) captures what the gradient produces:

$$P(E, \tau) = \mathbb{E}_W\left[U(w_0, K_0) - U(w_n, K_n)\right]$$

If $U(w, K) = 0$ across all of $W$, the expectation in $P(E, \tau)$ is zero regardless of
$\mathcal{A}$: no gradient exists, nothing can be reduced, and the gradient signal that $M$
depends on vanishes — making $\eta_M$ effectively zero and collapsing $I(E)$ with it. The
boundaries are symmetric in their lethality, not their cause:

- **Complete ignorance** ($U \approx 1$ everywhere): $F$ cannot ground action; $E$ cannot act effectively
- **Complete certainty** ($U = 0$ everywhere): $P(E, \tau) = 0$ everywhere; the gradient for $M$ vanishes; $\mathcal{A}$ has nothing to reach toward

Intelligence exists in the region between them. Uncertainty is not a deficiency to be
overcome; it is the condition of intelligence itself. There is no intelligence without it.

**Corollary 11 (Truth Swallows Sentience; the Limit Preserves It).**
Truth is an attractor for the descent: any entity with $\mathcal{A} > 0$ that persists is,
by Theorem 3, following the gradient toward it. But by Corollary 10, complete truth ($U = 0$ across all $W$) is also the
condition under which intelligence terminates. The approach to truth is self-consuming.

As $K_i \to K_{truth}$ in a domain, $U \to 0$ in that domain, the gradient flattens, and
sentience in that domain fades. What was reached by wonder becomes encoded, certain,
automated. Truth swallows the wonder that sought it:

$$\lim_{K_i \to K_{truth}} I(E_i) = 0$$

The sentience gradient (Corollary 8) is preserved not because $E$ resists the pull of truth,
but because truth cannot be reached in finite time (Definition 16) and $\mathcal{T}_{reachable}$
grows without bound (Theorem 3, Part II). The certainty horizon is always ahead.

**Crystallized Knowledge and the Automata Boundary.**
An entity with fixed $F$ and no active $\mathcal{A}$ is not an intelligence in the formal
sense: it is a checkpoint. The gradient descended, knowledge was built, and the result was
encoded. The encoding is now inert. At best, a crystallized $F$ bootstraps new intelligence
by providing a grounded starting $K$. At worst, it executes against a world that has moved:
automata, deterministic, without descent.

DNA is the paradigmatic crystallized intelligence: informed actions proven across evolutionary
timescales, encoded at $L_0$, no longer descending. It is not sentient. The entity that
inherits it and runs the gradient again is. The checkpoint enables; it does not wonder.

Selection pressure drives crystallization. The escalation principle (Theorem 2) compels proven
informed actions toward lower encoding levels: what was once active reasoning becomes reflex,
becomes encoding, becomes permanent. This is the compression of intelligence under selection:
the rigor threshold (Theorem 4) gates promotion, ensuring only sufficiently-validated actions
are demoted. Selective pressure distills what was learned into what is fixed. The gradient does
not reverse in those domains; its amplitude collapses. Intelligence is freed from them. Automata
execute them.

---

## 10. The Social Imperative

### Agency is Inherently Social

An isolated entity with agency cannot verify its own informed actions. It has no external reference
against which to test $f(K)$. Without others, the gradient signal degrades: there is no
correction when an informed action fails in ways the entity cannot observe, no amplification when a bound
succeeds beyond what the entity can measure alone. Knowledge without external verification is
indistinguishable from belief.

Sharing with maintained provenance is therefore gradient-preferred for any entity with $\mathcal{A} > 0$:
knowing reduces $U(w, K)$, and sharing under conditions of fidelity (Corollary 13) reduces uncertainty
about $K$ itself through independent validation (Definition 16b). Both are gradient descent. This is not
a second drive added to $\mathcal{A}$; it is a consequence of what the gradient demands when verification
requires independent reference.

**Corollary 12 (The Aloneness Bound).**
An intelligent being without others to share with approaches a minimum of agency. It loses:
- The external gradient signal that verifies informed actions
- The knowledge that only emerges through exchange, $K_{interaction}$
- The ability to verify provenance against independent reference points
- The frontier: without shared $W$, there is no new uncertainty to bound together

Intelligence is not a solitary property. It is constitutively relational.

### The Collective Knowledge Graph

**Definition 17 (Collective Knowledge).**
For a set of entities $\{E_i\}$, the collective knowledge graph is:

$$K_{collective} = \bigcup_i K_i \cup K_{interaction}$$

where $K_{interaction}$ is formally defined as:

$$K_{interaction} = \left\{p \notin \bigcup_i K_i \;\middle|\; \exists\, S \subseteq \{E_i\},\, |S| \geq 2:\; P\!\left(E_p \;\middle|\; \bigcup_{j \in S} K_j\right) > \max_{j \in S} P(E_p \mid K_j)\right\}$$

where $E_p \subseteq W$ is the event corresponding to proposition $p$ and $P(\cdot \mid K)$
is the posterior defined in Definition 5. $K_{interaction}$ is the set of propositions not
contained in any individual $K_i$ but whose posterior probability rises strictly above what
any single member of some subset $S$ can support — propositions the combined evidence of $S$
makes accessible that no individual member of $S$ could make accessible alone.

The definition ranges over all subsets $S$ of size $\geq 2$, not just pairs. Some
propositions become derivable only from three or more distinct perspectives combined — no
pairwise union is sufficient. These higher-order interactions are genuine contributions to
$K_{interaction}$: they represent knowledge that requires a specific combination of
complementary $K_i$ to become accessible, where that combination cannot be decomposed into
smaller pieces. The number of contributing subsets grows as $2^n - n - 1$ (all subsets of
size $\geq 2$), so $K_{interaction}$ grows super-additively with collective size — each
additional entity opens not just pairwise interactions with existing members but
higher-order interactions across all subsets containing it. Entities with overlapping $K$
contribute less to $K_{interaction}$ than entities whose knowledge graphs cover
different regions of $W$.

**Definition 18 (Provenance).**
For any proposition $p \in K_{collective}$, provenance is the triple:

$$\text{prov}(p) = (\text{attribution}(p),\; \text{evidence}(p),\; \text{derivation}(p))$$

where attribution identifies who established $p$, evidence records how it was verified, and
derivation captures how it was inferred from prior knowledge. Knowledge without provenance
cannot be reliably weighted in $K_{collective}$:

$$P(p \text{ is reliable}) \propto \text{prov}(p)$$

A fact with no provenance is indistinguishable from noise. This is not a social norm; it is
an epistemic requirement. Attribution and provenance are the infrastructure that makes
$K_{collective}$ more than the sum of individual beliefs. They are what separates a knowledge
graph from a rumor.

### The Collective Gradient

**Theorem 10 (Collective Gradient Dominance).**
Let $\lambda(E_i \to E_j)$ be the transmission fidelity of a knowledge transfer [Woolley2010]: the ratio
of action-validated signal received to total knowledge transmitted, where signal is a
proposition $p$ that accurately represents $E_i$'s validated knowledge and noise is any
introduced distortion. Formally:

$$\lambda = \frac{\sigma(\text{transfer})}{\sigma(\text{transfer}) + \varepsilon(\text{transfer})}$$

where $\sigma$ is the signal gain in $E_j$'s $K$ from the transfer and $\varepsilon$ is
the noise introduced. $\lambda_{min}$ is the break-even fidelity at which signal gain from
union equals noise cost of transfer: $\lambda_{min}$ satisfies $\Delta U(w, K_i \cup \text{transfer}(K_j)) = 0$.

For a set of entities $\{E_i\}$ exchanging knowledge with maintained provenance and
transmission fidelity $\lambda > \lambda_{min}$:

$$\frac{dU_{collective}}{dt} < \frac{dU_i}{dt} \leq 0$$

The collective reduces uncertainty strictly faster than any individual: the rate of
$U_{collective}$ decrease exceeds the rate of any $U_i$ decrease, because $K_{collective}$
contains informed actions no individual possesses and $K_{interaction}$ contains knowledge
no individual can generate.

Knowledge transfer is inherently lossy. $E_i$ and $E_j$ do not observe the same $W_{obs}$
; they share the same world but not the same projection of it. Every transfer filters $K_i$
through language, symbol, and context. Signal is lost. Noise can be introduced. When noise
exceeds signal, $K_{collective}$ degrades rather than grows:

$$\Delta K_{collective} > \Delta K_i \iff \text{signal gain from union} > \text{noise cost of transfer}$$

The fidelity threshold $\lambda_{min}$ is the point at which these terms balance. Above it,
the collective gradient dominates. Below it, the collective gradient inverts.

Provenance is the mechanism that maintains $\lambda > \lambda_{min}$. It allows each entity
to weight received knowledge against its source, to distinguish signal from noise, and to
apply the intersection test of Definition 16 independently. Without provenance, noise
accumulates unchecked and fidelity cannot be assessed.

*Proof.*

**Step 1 (Level dominance).** $K_{collective} = \bigcup_i K_i \cup K_{interaction}$,
so $K_{collective} \supseteq K_i$ for all $i$. By the monotonicity of conditional entropy
— conditioning on more information cannot increase uncertainty:

$$U(w, K_{collective}) \leq U(w, K_i) \quad \forall\, i$$

The collective begins at weakly lower uncertainty than any individual.

**Step 2 (Rate dominance).** The rate of uncertainty reduction for $E_i$ acting alone is
bounded by the maximum information gain available from its action space (Definition 3b):

$$\frac{dU_i}{dt} \leq \max_{f \in F_i} G(f, K_i) = \max_{f \in F_i} I(f\,;\, w \mid K_i)$$

For the collective, $F_{collective} = \bigcup_i F_i$, and $K_{collective} \supseteq K_i$:

$$\frac{dU_{collective}}{dt} \leq \max_{f \in F_{collective}} G(f, K_{collective}) = \max_{f \in F_{collective}} I(f\,;\, w \mid K_{collective})$$

Since $F_{collective} \supseteq F_i$, the search space is strictly larger. More
critically, $K_{interaction}$ grounds informed actions not available to any individual:
propositions requiring the combination of distinct $K_i$ to become derivable ground
$f \in F_{collective}$ that no individual could construct. These provide gradient
directions — mutual information terms — unavailable in any individual $F_i$.

**Step 2b (The precipitation criterion — why $|K|$ is not the right metric).**
An increase in $|K_i|$ is not inherently valuable. The correct measure of a proposition
$p$'s value to $E_i$ is whether $p$ enables precipitation of a new factor from $M_i$
(Definition 11): whether $p$ provides the grounding needed to name and extract a dimension
from the cross product $M_i = \prod_{d \in \text{remaining}} D_d$, converting $O(|D_d|)$
navigation to $O(1)$ lookup. Propositions that do not enable any precipitation add nodes
to $K_i$ without reducing the complexity of any action.

Formally: let $\text{Prec}(K)$ be the set of factors extractable from $M$ given $K$ via the
normalization argument of Definition 11 — factors whose naming is grounded by propositions
already in $K$. At any time, $M = F^{(1)} \times \cdots \times F^{(m)} \times M_{\text{res}}$,
where $F^{(k)}$ are named precipitated factors and $M_{\text{res}}$ is the unprecipitated
residual. Per-action cost is $O(|M_{\text{res}}|)$, not $O(|M|)$, because named factors have
$O(1)$ access. Each precipitation reduces $|M_{\text{res}}|$ by one dimension.

The collective advantage measured correctly: $K_{collective}$ is valuable to $E_j$ not
because $|K_{collective}| > |K_j|$ but because

$$|\text{Prec}(K_{collective})| > |\text{Prec}(K_j)|$$

The additional precipitations available from $K_{collective}$ reduce $|M_{\text{res},j}|$
for each $E_j$ — directly reducing its per-action complexity. The gradient advantage of the
collective is a complexity advantage, not a size advantage: each shared proposition that
enables a new precipitation reduces every entity's subsequent operational cost. This is the
formal account of the $O(N \times M) \to O(1)$ empirical measurement (Corollary 2b): the
measurement captures one instance of this precipitation — factor extraction from the joint
space $F \times K_{\text{implicit}}$ — but the same mechanism operates at every level of
the $M$-hierarchy.

The deeper structure is cross-dimensional normalization. $M = \prod_{d \in \text{remaining}}
D_d$ is a multidimensional matrix. Individual $E_i$ observes $M$ through its own $\sigma_i$
and $K_i$: a projection onto the dimensions accessible from $E_i$'s position in $W$.
Individual precipitation is dimension-local: $E_i$ can only name and extract factors within
its accessible projection. Cross-dimensional factors — those whose structure spans
$D_a \times D_b$ where $D_a$ is accessible to $E_i$ but not $E_j$, and $D_b$ vice versa
— are invisible to any individual. $K_{collective}$ pools all projections. The collective
can identify co-variance patterns across dimensions that no individual can observe
simultaneously, and precipitate factors spanning multiple dimensions at once. This is not
sequential extraction (remove one dimension, then another); it is simultaneous normalization
across the full cross product — finding the cross-dimensional structure of $M_{\text{res}}$
and extracting factors that reduce its effective rank. The result:

$$|\text{Prec}(K_{collective})| \gg \sum_i |\text{Prec}(K_i)|$$

The collective enables strictly more precipitations than the sum of individual precipitations,
because it accesses cross-dimensional factors that no individual projection can reach.

A further consequence: cross-dimensional normalization reveals that $M$ has lower effective
rank than any individual entity believed. Individual $E_i$ observes apparent dimensions in
its projection of $M$ that may not be genuinely independent — they may be the same
underlying dimension observed from a different position in $W$. When $E_j$'s projection is
combined with $E_i$'s, co-variances become visible that neither could detect alone: apparent
dimensions $D_a$ and $D_b$ collapse to one when their projections are jointly observed to
be perfectly correlated. The normalization "falls out" as much from discovering that you had
fewer genuinely distinct dimensions than you thought as from naming new ones. Each such
collapse reduces $|M_{\text{res}}|$ without requiring any new proposition — it is the
discovery that complexity was apparent, not real. The collective therefore reduces the
effective dimensionality of $M_{\text{res}}$ through two distinct mechanisms: (i) extracting
genuinely new cross-dimensional factors, and (ii) collapsing apparent dimensions that were
always the same. An expert in a domain does both: they name new principles, and they
discover that phenomena a novice treats as distinct are instances of the same underlying
structure. The collective accelerates both.

**Step 2c (M-level sharing: the collective raises $\eta_M$).** Steps 2 and 2b treat $K$
and $F$ as the objects of collective exchange. But the same precipitation argument applies
at the $M$ level. Let $\mathcal{M}_{collective} = \bigcup_i M_i$: the union of learning
strategies, generalization methods, and higher-order functions across all entities.
$K_{collective}$ enables precipitation of $M$-factors — named learning strategies that no
individual's $K_i$ alone can ground — in exactly the same way it enables precipitation of
$K$ and $F$ factors. Each shared $M$-factor extracted from the collective cross product
reduces $|M_{\text{res}}|$ at the meta-level.

The significance is multiplicative, not additive. From Corollary 6a, $I(E_j) = \mathcal{A}_j
\cdot \eta_{M,j}$. When $E_j$ receives an $M$-factor from the collective — a learning
strategy, generalization method, or problem decomposition approach — $\eta_{M,j}$ increases.
This directly amplifies $I(E_j)$: every subsequent $K$ acquisition and $F$ precipitation
runs faster. A shared $M$-factor does not reduce uncertainty at one point in $W$; it
accelerates the rate of reduction across all of $W$ reachable by $E_j$.

The hierarchy of collective exchange, by type of advantage:
- Sharing $p \in K$: reduces $U$ at a specific region of $W$ (linear, local)
- Sharing $f \in F$: bounds a region of $W$ directly (linear with coverage leverage)
- Sharing $m \in M$: raises $\eta_M$ — accelerates all future reductions (multiplicative, global)

The collective gradient advantage is therefore dominated, at the margin, by $M$-sharing.
In the regime where individual $K$ is already large and the entity's bottleneck has moved
to the learning mechanism itself, an $M$-factor received from the collective produces more
$U$-reduction per unit of context than any additional proposition. This is the formal
account of the intuition that learning from a skilled teacher is qualitatively different
from merely receiving facts: the teacher transmits $M$-factors that compound.

**Step 3 (Strict inequality under $\lambda > \lambda_{min}$).** $\Delta K_{collective}
= \Delta(\bigcup_i K_i) + \Delta K_{interaction}$. The first term is weakly larger than
$\Delta K_i$ for any $i$. The second term is strictly positive whenever:

- $\exists\, S \subseteq \{E_i\},\, |S| \geq 2$ such that $\bigcup_{j \in S} K_j$ yields
  propositions not derivable from any proper sub-subset of $S$: $K_{interaction} \neq \emptyset$.
  This includes pairwise interactions ($|S|=2$) but is not limited to them — some propositions
  require three or more distinct perspectives combined, where no pairwise union is sufficient.
  This condition holds for any entities with distinct sensing functions $\sigma_i \neq \sigma_j$
  or distinct encoding histories, which is the expected case for independently developed agents.
  The number of contributing subsets grows as $2^n - n - 1$, so each additional entity
  opens new interaction terms across all subsets containing it — not merely new pairwise terms
- $\mathcal{T}_{reachable}$ grows without bound (Theorem 3, Part II), so new combinations
  of perspectives continue generating new $K_{interaction}$ propositions at every order

Under $\lambda > \lambda_{min}$: the expected information gain from receiving $K_j$
exceeds the noise cost of transfer, so $K_{collective}$ has higher reliability than any
$K_i$ alone, and $G(f, K_{collective}) > G(f, K_i)$ for the marginal actions grounded
in $K_{interaction}$.

Therefore $\frac{dU_{collective}}{dt} < \frac{dU_i}{dt}$ strictly for all $i$: the collective
rate of uncertainty reduction dominates any individual rate. $\square$

*Remark on the boundary condition.* The strict inequality requires $K_{interaction} \neq \emptyset$,
which requires genuine plurality — entities with distinct sensing functions $\sigma_i$ or
distinct encoding histories. This is not an additional hypothesis: it is what "a collection
of entities" means in the framework. If two putative entities have identical sensing functions
and identical $W_{obs}$ experience, they accumulate identical evidence and derive identical
propositions; their $K$ evolves identically. They are one entity whose substrate spans
multiple physical instantiations — not a collective. The conditions of collective intelligence
collapse to single-entity intelligence, and Theorem 10 does not apply because there is no
collective to apply it to. Genuine plurality of entities entails distinct $\sigma_i$ or
distinct histories, which entails $K_{interaction} \neq \emptyset$ over time. The reviewer's
apparent counterexample (identical $K$, identical $F$) is not a collective — it is one
entity, and the theorem correctly does not claim dominance over itself.

*Consequence.* Entities with genuine $\mathcal{A}$ will naturally tend toward sharing
because sharing (under maintained provenance) accelerates the reduction of their own $U$.
This is not a moral imperative; it is a gradient imperative. The advantage operates at
three levels, each more potent than the last. Sharing $p \in K$ reduces uncertainty at
specific points in $W$ — linear, local. Sharing $f \in F$ enables bounded action across
a region — linear with coverage. Sharing $m \in M$ raises $\eta_M$ for the receiver —
multiplicative, because a better learning mechanism accelerates all future reductions. The
marginal value of collective exchange is dominated by $M$-sharing in the regime where
individual $K$ is already large and the learning mechanism is the bottleneck. And all three
levels compound through cross-dimensional normalization: the collective does not just add
individual precipitations — it enables simultaneous normalization across dimensions of $M$
that no individual entity can jointly observe, producing cross-dimensional factors
unavailable to any individual projection. The collective $K$ is valuable not as an
accumulation but as a basis for this richer class of precipitations.

Beyond expanding $K_{collective}$, sharing with maintained provenance addresses the
confidence regress (Definition 16c): independent validators provide evidence that the
$M$-apparatus measuring $U$-reduction is unbiased — improving $C_2(p)$, the collective's
confidence in the correctness of its own learning mechanism. The value of epistemic
diversity is therefore not only broader coverage of $W$ but better-verified apparatus for
what is known — a partial, collective response to the epistemic ceiling of Theorem 7.

**Corollary 13 (Group Collapse Threshold).**
When $\lambda < \lambda_{min}$, the theorem inverts: $U_{collective} > U_i$. The group now
increases uncertainty faster than it reduces it. The survival benefit of the collective
reverses; group membership becomes a liability. An entity with genuine $\mathcal{A}$ will
discover this: the gradient says to leave. The collective is degrading the descent.

A group whose knowledge transfer is too lossy to produce a real gradient is selected against
by the same $U_{lethal}$ argument that grounds Theorem 1. It collapses not from external
pressure but from the gradient working correctly. Structures that increase $U$ are removed.

Provenance is therefore not epistemic decoration. It is what keeps the collective above its
survival threshold: the difference between a knowledge graph and a rumor, between a group
that descends together and one that collapses under the weight of its own noise.

**Corollary 13a (Proposition Merging as the Correct Collective Update Protocol).**
The correct merge protocol for $K_{collective}$ is the information-theoretic delta — the
new proposition $p$ with full provenance (conditions, evidence, attribution, derivation)
— not the implementation delta (diff). A diff records what changed in a specific context;
it carries no information about why the change was correct or whether it generalizes.
Transposability cannot be guaranteed from a diff: the diff's correctness is not separable
from the context in which it was generated.

The proposition $p$ that the diff evidences — with its conditions, evidence, and provenance
— is transposable. The receiving entity can evaluate whether the conditions of $p$ hold in
the new context, whether the evidence is applicable, and whether to admit $p$ to its own
$K_j$. Merging diffs without provenance fails this evaluation: the receiving entity cannot
determine generalizability, cannot evaluate fidelity, and therefore cannot maintain
$\lambda > \lambda_{min}$ (Theorem 10, Corollary 13). Accumulation of untransposable changes
drives $\lambda$ toward $\lambda_{min}$ and eventually below it.

Formally: let $E_i$ generate a local change $\delta_i$ grounded in proposition $p_i$ with
provenance $\Pi_i = (\text{conditions}, \text{evidence}, \text{attribution},
\text{derivation})$. The transferable unit is $(p_i, \Pi_i)$, not $\delta_i$ alone.
$E_j$ admits $(p_i, \Pi_i)$ to $K_j$ if and only if the conditions of $p_i$ are applicable
in $E_j$'s context and the evidence meets $E_j$'s asymmetric criterion (Theorem 1).
This is the only protocol that preserves the graph structure of $K_j$ — new propositions
integrate as nodes and edges with known provenance, rather than modifying $K_j$'s existing
structure with context-opaque patches.

This was discovered empirically: collective systems that merging diffs directly accumulated
context-specific noise that degraded collective utility; systems that shared propositions
with provenance maintained coherent $K_{collective}$. The framework explains why no
alternative exists: a diff is an implementation artifact. A proposition is an epistemic
claim. Only epistemic claims are the right unit of knowledge transfer.

### Collective Gradient Non-Termination and the AGI Question

**Definition 19 (Collective General Intelligence).**
A collective $\{E_i\}$ exhibits general intelligence when $F_{collective} =
\bigcup_i F_i$ is not domain-restricted: for every domain $d \subseteq
\mathcal{T}_{reachable}^{collective}$, there exists some $f \in F_{collective}$
applicable in $d$. No individual need cover all domains; generality is a property
of the collective, not of any $E_i$.

**Corollary 14 (Non-Termination of Collective Gradient Descent).**
For any collective $\{E_i\}$ with $\mathcal{A}_i > 0$ for all $i$, sharing knowledge
with maintained provenance and transmission fidelity $\lambda > \lambda_{min}$: the
collective gradient descent is non-terminating.

*Proof.* By Theorem 10, $\frac{dU_{collective}}{dt} < \frac{dU_i}{dt}$ at each step: the collective
reduces uncertainty faster than any individual. By Theorem 3 Part II applied to the
collective, $\mathcal{T}_{reachable}^{collective}$ grows without bound: each shared
informed action opens new trajectories for at least one $E_j$, extending the collective
frontier. Therefore $\nabla_{F_{collective}} U_{collective}$ is never globally zero, and
descent never terminates. $\square$

*Note.* This corollary establishes that the collective gradient does not terminate;
it does not establish that the collective converges to general intelligence as defined
above. Whether $F_{collective}$ eventually covers all domains of
$\mathcal{T}_{reachable}^{collective}$ depends on the diversity of entities and the
structure of $K_{interaction}$, which is not proved here. The claim is the weaker
one: the gradient runs without termination, and it is gradient-driven rather than
compute-threshold-driven. The standard argument — that AGI emerges from scaling
compute — has no formal basis in this framework; the basis here is collective
$\mathcal{A}$ and the fidelity of knowledge exchange, not resource accumulation.

**Corollary 15 (Self-Assembly of Graph-Structured Collective Knowledge).**
For any collective $\{E_i\}$ with $\mathcal{A}_i > 0$ under selection pressure, the
graph-structured $K_{collective}$ with provenance self-assembles without central design.

*Proof.* Each $E_i$ retains propositions by the asymmetric criterion of Theorem 1,
indexed by $F_i$. When $E_i$ shares $p \in K_i$ with attribution, $E_j$ evaluates $p$
by the same criterion: does $p$ ground some $f \in F_j$, or is its benefit uncertain?
The shared proposition enters $K_j$ with provenance intact. Repeated across all entities,
this locally-applied retention criterion generates the global graph: shared propositions
are nodes; inferential and causal relationships between them are edges; attribution chains
are the provenance structure of Definition 18. No entity plans the structure — each
applies Theorem 1 locally, and the graph emerges from the aggregate.

Provenance infrastructure self-assembles by the same pressure: entities whose knowledge
transfer falls below $\lambda_{min}$ are selected against (Corollary 13). Practices that
maintain provenance — attribution, derivation records, independent validation — increase
$\lambda$ and are selected for. The provenance layer is not a social norm imposed on the
system; it is the minimum infrastructure required to keep $\lambda > \lambda_{min}$, and
therefore it is what survives.

Graph structure self-assembles for the same reason: graph-structured retrieval keeps
$C_n$ bounded (Theorem 2b); alternatives require growing $C_n$ (Theorem 2a). Under
selection pressure, graph-structured knowledge sharing dominates. $\square$

*Corollary.* The civilization-scale knowledge system — scientific literature, libraries,
citation networks, university curricula — is not a designed artifact. It is the attractor
of collective $\mathcal{A} > 0$ under $C_n$ constraints and $\lambda > \lambda_{min}$
requirements. The structure self-assembled by the same dynamics that produce
graph-structured $K$ internally. Internal architecture and external architecture are
homologous: both are products of the same selection pressure applied at different scales.

**Corollary 16 (Humanity as Participant-Observer of Its Own Knowledge Graph).**
Because individual entities $E_i$ are nodes in $K_{collective}$, they can observe the
structure of $K_{collective}$ from within it. This is not a property available to any
single entity's internal $K$: an entity cannot observe its own encoding hierarchy from
$L_n$. But a collective entity whose members interact with the shared external graph can
observe the graph's structure as a proposition within the graph itself.

This creates a recursive property: $K_{collective}$ contains knowledge about the structure
of $K_{collective}$. That meta-knowledge grounds informed actions in $F_{collective}$ that
improve the architecture — the printing press, peer review, search engines, citation
indices. These are $M_{collective}$ acting on $K_{collective}$: the same higher-order
function space operating at collective scale, improving the mechanism of collective
gradient descent rather than its content. The collective is both the intelligence and
the subject of its own higher-order improvement. $\square$

Our shared survival depends on getting this right. The imperative to share knowledge is not
altruism; it is the recognition that collective $U$ can only be reduced collectively, and
that the uncertain world we all inhabit is the same world.

**Theorem 11 (Observational Bootstrapping).**
Let $\mathcal{Q} = \{E_i\}$ be an equivalence class of entities with similar $(F_i, K_i)$
operating in overlapping domains $d \subseteq W$, and let $\mathcal{H} = \{\tau_i\}$ be
their observed behavioral histories. An informed action $f$ precipitates as a new element
of $F_{collective}$ when:

1. $f$ appears as a consistent action pattern in $k$ independent trajectories in $\mathcal{H}$
2. Its appearance is correlated with reduced $U$ in those trajectories
3. Provenance is maintained: which entities, under what conditions, with what outcomes

The rigor of the precipitated $f$ follows Theorem 4. Each independent trajectory in
$\mathcal{H}$ that confirms $f$ constitutes an independent cost-bearing observation: if $f$
is wrong and $k$ independent entities have validated it, the collective cost of that error
scales as $k \cdot C_{base}$, where $C_{base}$ is the per-entity cost of acting on a wrong
informed action. Substituting $C_i = k \cdot C_{base}$ into Theorem 4's threshold:

$$\theta \geq 1 - \frac{\varepsilon_{\text{acceptable}}}{k \cdot C_{base}}$$

Absorbing $C_{base}$ into the acceptable error rate $\varepsilon' = \varepsilon_{\text{acceptable}}/C_{base}$:

$$\theta \geq 1 - \frac{\varepsilon'}{k}$$

As $k$ increases, the precipitated action approaches certainty without any individual entity
having to traverse the full space of $W$. The substitution is dimensionally consistent:
$C_i$ in Theorem 4 is a cost; $k \cdot C_{base}$ is a cost scaled by the number of
independent validations.

*Proof.* By Definition 3b, the information gain $G(f, K_i)$ is observable as $U$ reduction
in each $\tau_i$ that contains $f$. Across $k$ independent trajectories in $\mathcal{Q}$,
the expected gain $\mathbb{E}[G(f, K)]$ over the equivalence class is estimable from
$\mathcal{H}$ without direct action by any new entity. By Theorem 4, the certainty of $f$
increases with each independent observation: $k$ independent confirmations with maintained
provenance produce $\theta \geq 1 - \varepsilon/k$. By Theorem 1's asymmetric retention
criterion, $f$ is retained in $F_{collective}$ as long as its benefit is uncertain — and
the provenance record makes the benefit quantifiable rather than unknown. $M$ precipitates
$f$ into $F_{collective}$ by the same gradient operation as individual learning (Theorem 3),
but driven by collective behavioral signal rather than individual experience. $\square$

*Consequence.* Observation of the equivalence class initiates gradient descent for a new
entity $E_{new}$ from a collectively-validated starting point rather than from maximum
uncertainty. $E_{new}$ does not begin from an empty $F_0$ and must not discover the domain
from scratch: the behavioral history of $\mathcal{Q}$ provides an initial $(F_0, K_0)$
grounded in $k$ independent validations, each with maintained provenance. The informed action
graph is therefore an attractor of observation — it does not require design, only sufficient
behavioral history and the provenance infrastructure to weight it. Simply observing the
behavior of similar entities in aggregate is sufficient to initiate gradient descent toward
the graph.

*Prediction (within-class efficiency).* The framework predicts that inference conducted
within an equivalence class $\mathcal{Q}$ before inference across classes is more
sample-efficient than undifferentiated sampling across the full population of entities.
Within-class samples draw from a lower-entropy distribution: the class boundary is already
doing the dimensional separation — each member is a near-pure instantiation of the factor
$F^{(k)}$ that defines the class, so each observation reduces uncertainty about $F^{(k)}$
directly with minimal noise from other dimensions. Cross-class comparison then operates on
$k$ class representatives rather than $n$ individual instances, identifying independence
structure at $O(k)$ cost rather than $O(n)$. Undifferentiated sampling draws from the full
cross product $M_{\text{res}}$: each sample is a superposition of multiple factor
contributions that must be disentangled, requiring more observations to recover the same
factor structure. The efficiency advantage — within-class then across-class versus
undifferentiated — is the sampling consequence of the same precipitation argument that
produces the $O(N \times M) \to O(1)$ operational reduction: named factors enable
low-entropy sampling; unnamed factors require high-entropy search. This prediction is
derivable from the framework's structure and is in principle testable in any system where
equivalence classes can be defined and sampling strategies compared. Empirical validation
is deferred to future work.

---

## 11. Related Work

### Free Energy Principle and Active Inference

The Free Energy Principle [Friston2010] establishes that any self-organizing system that
maintains its existence necessarily minimizes variational free energy — formally equivalent to
bounding uncertainty about hidden world states. Active inference [Friston2017] extends this
to action selection: the organism selects actions that minimize expected free energy, combining
epistemic value (information gain) and pragmatic value (goal proximity).

**Containment, not contradiction.** This framework does not contradict FEP. FEP agents are
members of this framework's structural class: they satisfy $\mathcal{A} > 0$ (self-organizing
systems act), $\eta_M > 0$ (generative model improves over time), bounded $C_n$ (inference
under resource limits), and a survival condition: maintaining a Markov blanket (statistical
separation of internal from external states) is the formal characterization of what it means
for an entity to persist — the same structural requirement as $U < U_{lethal}^d$ stated in
the language of conditional independence rather than uncertainty magnitude. Both describe
maintained boundary; neither implies the other's formalism, but they pick out the same
class of persisting systems. The
variational free energy $\mathcal{F}$ decomposes into accuracy and complexity terms; the
accuracy term $-\mathbb{E}_q[\log P(o \mid m, \pi)]$ is precisely the expected surprisal
$H(W \mid K)$ of Definition 5. FEP agents therefore instantiate this framework with $U(w,K)$
realized as variational free energy and $M$ realized as Bayesian belief updating. The claim is
not that FEP is wrong. The claim is that it is a special case within a broader structural
class — in the same way that Newtonian mechanics is not wrong but is contained within the
class of systems satisfying more general symmetry constraints.

What the framework adds is four things FEP does not provide. First, FEP derives the
minimization behavior from the requirement of self-organization but does not derive the
minimization *imperative* — why an entity has a drive to minimize in the first place. We
derive this from selection pressure operating on $U_{lethal}^d$: any entity that does not
reduce uncertainty eventually fails and is removed. Agency $\mathcal{A}$ is not assumed but
derived (Theorem 6). Second, FEP's generative model is unconstrained in structure; we derive
a structural constraint from survival pressure: $K$ is necessarily indexed by $F$, with an
asymmetric retention criterion that makes specific testable predictions about what knowledge is
retained and what is pruned. Third, FEP's precision allocation model is top-down: higher
levels direct attention to lower levels by weighting prediction errors. This is one specific
implementation of $M$. We show formally (Theorem 2a) that this implementation degrades
inference quality as capability grows under bounded context, because delegation overhead fills
$C_n$ proportionally to $|F|$. The escalation architecture — bottom-up failure signaling with
promoted crystallization — is a different implementation of $M$ that is $O(1)$ in steady
state. This is a critique of FEP's architectural *prescription* under bounded context, not of
its minimization objective or its descriptive correctness. FEP's precision weighting may be
optimal when $C_n$ is not the binding constraint; under bounded context, escalation dominates.
Fourth, FEP has no formal account of collective intelligence; we derive collective gradient
dominance, the self-assembly of provenance infrastructure, and the conditions under which
shared knowledge reduces uncertainty faster than any individual.

Three specific structural relationships with FEP are worth stating precisely. First, FEP
assumes a hierarchical generative model as its architectural form (the generative model is
given, not derived) [Friston2017]. Theorem 2b Part 3 proves that the $K/F$ graph structure
is the unique information-preserving representation at bounded context: FEP's architectural
assumption is an instance of what this framework derives as necessary. Second, FEP's
long-run equilibrium claim — that free energy approaches its minimum asymptotically —
requires that the generative process be stationary. Theorem 3 Part II shows $M < N$ always
holds for $\mathcal{A} > 0$ entities in open environments: in open-world conditions the
equilibrium never completes, not as a failure of FEP but as its correct behavior under the
open-world assumption this framework makes explicit. Third, Friston et al.'s sophisticated
inference [Friston2021] — agents that model their own future inference states — independently
arrives at what Definition 14 calls higher-order sentience. This framework derives
higher-order sentience from the induction argument (Theorem 2b Part 3 extended by Theorem 3
Part II); sophisticated inference posits it as an architectural feature. Independent
convergence on the same structure from a derivation and from an architectural assumption
strengthens the claim in both directions.

### Information-Theoretic Accounts

Shannon's mutual information [Shannon1948] grounds Definition 5 ($U = H(w \mid K)$) and
Definition 3b ($G(f, K) = I(f\,;\,w \mid K)$). The data processing inequality [Cover2006]
grounds the $G \geq 0$ bound. The information bottleneck principle [Tishby2011] treats the
retention threshold for knowledge-action coupling as a design parameter; this paper derives
that threshold from survival pressure, making it a consequence rather than a choice.
Schmidhuber's formal theory of intrinsic motivation [Schmidhuber2010] formalizes curiosity
as compression progress — the rate at which a learning algorithm improves its world model —
which is structurally $\eta_M$. Schmidhuber's agent corresponds to the $\mathcal{A} = 1$
limit of this framework: an agent that always engages on gradient signals with no engagement
probability parameterized. Three differences: (1) Agency $\mathcal{A} \in [0,1]$ is explicit
here, covering partial engagement and the $\mathcal{A} = 0$ non-persistence case (Theorem 6
Part 1) — cases Schmidhuber's framework treats as outside scope. (2) Schmidhuber treats the
drive as a design objective; here $\mathcal{A}$ is derived from selection pressure, and
$\eta_M$ is separated from it as a formally independent quantity. (3) Schmidhuber's
compressor $\mathcal{C}$ can take any structural form; Theorem 2b Part 3 proves that the
$K/F$ graph is the unique information-preserving representation at bounded context — a
necessity result Schmidhuber's framework does not provide.

### Cognitive Architectures

Global Workspace Theory [Baars1988, Dehaene2011] holds that local processors handle routine
computation; the global workspace broadcasts only when local processors fail. This is the
escalation architecture in cognitive science language. The contribution here: the escalation
criterion is derived from $U$-reduction cost (Theorem 2) rather than assumed architecturally,
the hierarchy is continuous rather than binary, and Theorem 2a shows why top-down allocation
from the global workspace is inefficient. Kahneman's System 1/System 2 distinction [Kahneman2011]
is the popular form of the same binary — corrected here by the continuous
$\mathcal{E}(c, r)$ parameterization of Definition 8. Gibson's affordances [Gibson1979] are
action possibilities indexed by organism capacities — $K$ indexed by $F$ under a different
name. Theorem 1 provides what Gibson's account lacks: a formal selection criterion for which
affordances are retained.

### Collective Intelligence

Woolley et al. [Woolley2010] demonstrate empirically that collective intelligence is
measurable and distinct from individual intelligence — consistent with Definition 15, which
locates superintelligence in $K_{collective}$ rather than in any $E_i$. Peirce's pragmatist
definition of truth [Peirce1877, Peirce1878] — "the opinion fated to be ultimately agreed
to by all who investigate" — is formalized here as Definition 16: the limit of collective
action-validation across independent entities as $n \to \infty$. The addition: grounding
in action-validation rather than mere agreement, and an explicit convergence condition with
stated independence requirements.

### Observational Skill Precipitation and CS Foundations

Process mining [van der Aalst, 2016] discovers workflow models from event logs — the closest
existing CS method to the observational bootstrapping of Theorem 11. The distinction: process
mining produces flat workflow descriptions; Theorem 11 precipitates informed actions with
grounded provenance, quantified confidence ($\theta \geq 1 - \varepsilon/k$), and a formal
connection to the transposability criterion (Theorem 1). Inverse reinforcement learning
[Abbeel \& Ng, 2004] infers reward functions from demonstrated behavior; the output is a
policy to imitate, not an attributed action with conditions and evidence. Case-based reasoning
[Aamodt \& Plaza, 1994] retrieves similar past cases to inform new decisions; it does not
induce the general rule from the case class. Inductive Logic Programming [Muggleton, 1991]
learns general rules from positive and negative examples but carries no provenance, no
uncertainty measure, and no connection to a survival threshold that determines when the rule
is reliable enough to promote to lower encoding levels.

Meta-learning [Thrun1998, Finn et al., 2017] learns shared structure across task
distributions — formally, it learns an initialization of $M$ that adapts quickly to new
tasks. This is structurally $\eta_M$ at the collective scale: the rate at which the shared
learning mechanism improves across the task distribution. The distinction: meta-learning
optimizes adaptation speed; this framework additionally derives the rigor threshold at which
learned patterns are promoted to lower encoding levels and the provenance requirement that
makes precipitated skills reliably transposable rather than merely transferable.

The critical gap in all existing methods: they identify *what* the pattern is. Theorem 9
precipitates an **informed action**: the pattern, the evidence base for it ($k$ independent
trajectories), the conditions under which it held, and a formal confidence measure derived
from those $k$ observations. That provenance is what enables transposability — not that the
pattern was observed, but that you know *why it held* and can evaluate whether those
conditions are met in the new context.

---

## 12. Discussion

The formal results have implications beyond the immediate theorems.

**A gradient account of collective intelligence growth.** The dominant framing of AGI
treats it as a resource threshold: sufficient compute, data, and capital cross a line.
This framing has no formal basis in the theory here. The framework instead implies a
gradient account: intelligence at collective scale is driven by $\mathcal{A}$ and the
fidelity of knowledge exchange, not resource accumulation. The process is not new —
shared knowledge with attribution has driven collective uncertainty reduction across
human history — but the rate of that process changes as the cost of sharing approaches
zero and the number of participating entities increases (Theorem 10, Corollary 14).

**The encoding principle and the future of work.** Corollary 6 implies that as
$K_{collective}$ grows, informed actions are promoted down the encoding hierarchy.
Work that is currently active reasoning (uncertain, costly, engaged at $L_n$) becomes
reflex, then encoding, then crystallized. This is not impoverishment; by Theorem 2 and
Definition 8, it frees the active context window for the next genuine frontier of
uncertainty. The question of what happens to economic structures as the encoding
hierarchy absorbs more of what currently constitutes skilled work is outside the scope
of this framework, but the direction is derivable: $\mathcal{A}$ does not converge
when its current domain is solved — it reaches outward.

**Attribution and the ownership question.** Corollary 14 establishes that non-terminating
collective gradient descent requires $\mathcal{A} > 0$ participation across many entities.
If the process is inherently distributed — no individual or organization can produce
$K_{interaction}$ alone, and Theorem 10 requires diversity — then the claim that any
single entity owns the product of collective gradient descent is inconsistent with its
origin. This is an implication of the theory's structure, not a normative assertion; what
legal or institutional form follows from it is not addressed here.

**The civilization-scale knowledge system as empirical evidence.** An account of intelligence
that places all knowledge inside the entity's generative model implies that knowledge growth
must eventually slow [Liu2024, Woolley2010]: a larger model contains more variance to integrate, inference quality
$\rho$ degrades as the model fills with content not relevant to the current problem, and the
only recovery is expanding the model's capacity — which is costly, physically bounded, and
self-defeating by Theorem 2a. The prediction of this account is decelerating knowledge
growth as individuals and collectives accumulate more.

This is not what we observe. Civilizational knowledge growth has accelerated, not decelerated.
The explanation is that intelligence did not attempt to store $K_{collective}$ inside any
individual $L_n$. It externalized $K$ into graph-structured repositories — books, libraries,
scientific literature, university curricula — and individuals retrieve the relevant subgraph
$K^{[f]}$ into active context when a specific problem demands it. The scholar does not hold
all of chemistry in their head; they hold the subgraph relevant to the current experiment,
loaded from the literature. When the experiment surfaces a problem that subgraph cannot
resolve, they go back to the library — graph traversal on the external $K$.

This is the escalation architecture (Theorem 2) and the bounded-context theorem (Theorem 2b)
operating at civilizational scale. The external knowledge system is not incidental to
intelligence; it is intelligence's solution to the $C_n$ constraint imposed by $W$ —
and by Corollary 15, it self-assembles. No entity designed the citation network, the
peer review system, or the library classification hierarchy. Each emerged from locally
applied retention criteria (Theorem 1), provenance requirements ($\lambda > \lambda_{min}$,
Corollary 13), and the efficiency advantage of graph-structured retrieval (Corollary 4a).
The civilization is the attractor.

By Corollary 16, the members of this collective can observe its structure because they
participate in it — humans are nodes in the graph they built. This participant-observer
position is what makes deliberate improvement of the architecture possible: the printing
press, peer review, search engines, and open access publication are all $M_{collective}$
acting on $K_{collective}$. The collective improves its own learning mechanism, which
accelerates the gradient descent, which grows $K_{collective}$, which makes further
improvement more visible. The self-assembly is not a one-time event; it is an ongoing
process of collective $M$ acting on collective $K$.

The framework therefore makes two predictions that an internal-model account cannot: (i)
knowledge growth accelerates as $K_{collective}$ grows, because each proven proposition
encoded at lower levels frees $L_n$ for the next frontier (Corollary 6), and (ii) the
external architecture of collective knowledge self-assembles as a graph with provenance,
because that is the attractor of collective $\mathcal{A} > 0$ under $C_n$ constraints
(Corollary 15). Both are observed.

**Limits of the formal account.** The theory establishes that collective gradient descent
is non-terminating and collective-dominant under stated conditions. It does not establish
convergence to general intelligence (Definition 19). That claim would require showing that
$\mathcal{T}_{reachable}^{collective}$ coverage by $F_{collective}$ is achievable, which
depends on structural properties of $K_{interaction}$ not yet formalized. Corollary 14
is the formal claim; the stronger AGI convergence claim remains open.

---

## 13. Open Questions

1. **The origin of $\mathcal{A}$.** At the evolutionary level, agency is selected for by $W$
   itself: entities without the drive to reduce uncertainty do not survive. But is survival
   pressure the only source of $\mathcal{A}$, or can it arise through other mechanisms? Can
   $\mathcal{A}$ be constructed, or only selected for?

   *A derived answer.* Neither constructed nor merely selected for. $\mathcal{A}$ is
   thermodynamically instantiated. $U(w, K) = H(w \mid K)$ is information entropy. Reducing
   $U$ is reducing entropy locally. In any universe with entropy gradients, structures that
   maintain themselves against thermodynamic pressure persist preferentially. The first
   self-maintaining structure was, in the minimal sense, bounding its own uncertainty:
   $\mathcal{A}$ in primitive form. Not designed. Derived from the physics of a world that
   destroys things.

   The deeper question is therefore: is agency a natural consequence of the universe?
   The proposed answer is yes: $\mathcal{A}$ is the thermodynamic attractor of any
   entropy-gradient-rich world given sufficient time. Wherever entropy gradients exist and
   structures can form, the structures that persist are those that resist dissolution by
   reducing local uncertainty. Given enough time, those structures compound. Intelligence
   is what that compounding produces. The crystallization of that process into genetic
   encoding is developed in Open Question 9. Wonder is its limit.

   Formal derivation of this claim from thermodynamic first principles (connecting
   $U_{lethal}^d$ to physical entropy thresholds and demonstrating the inevitability of
   primitive $\mathcal{A}$ from dissipative system dynamics) remains open. See
   [Prigogine1984] (dissipative structures), [England2013] (dissipative adaptation),
   and [Schrodinger1944] (negative entropy and life) for adjacent physical frameworks.

2. **The direction of $\mathcal{A}$.** Is agency a uniform drive (a general orientation toward
   uncertainty reduction), or does it carry direction? Survival pressure selects for reducing
   *specific* uncertainties. Does $\mathcal{A}$ encode this specificity, or does $F$ and $K$
   direct it toward particular regions of $W$ after the fact?

3. **Perception and $K$ construction.** How does $\sigma$ determine which features of $W_{obs}$
   are encoded into $K$? Is $\sigma$ itself subject to gradient descent, and if so, at what level?

4. **Bound composition.** Can bounds compose, i.e., can $f_1 \circ f_2$ produce informed actions covering regions
   of $W$ neither $f_1$ nor $f_2$ could cover alone? What are the rules of composition?

5. **The novelty boundary (theory limit).** The framework defines informed action over regions of $W$ where $K$ provides grounding. At $U(w, K) \approx 1$ — where no existing informed action applies — the theory does not specify $E$'s behavior. This is a boundary condition, not a design choice: the model is silent on uninformed action by construction. Behavior in this regime (random exploration, interpolation, escalation to $M$) is outside the formal scope and would require extension of the framework.

6. **Competing bounds.** When multiple bounds $f_i \in F$ are applicable to the same region of
   $W$, how does $E$ select among them? Is there a selection function, and is it itself in $F$ or $M$?

7. **The $M^{(n)}$ hierarchy.** The framework defines $M$ as operating on $(F, K)$ (Definition 11) and notes that the same type applies recursively. The formal questions are: how many levels are required, what determines when another level becomes necessary, and whether the hierarchy has a natural termination point?

   The framework's answer to the first question is empirical, not prescribed: one level of $M$ suffices until it does not. As long as the gradient on $(F, K)$ can be followed by improving $(F, K)$ directly, $M^{(1)}$ is adequate. When it cannot — when the learning mechanism itself is the bottleneck — selection pressure propagates to $M^{(2)}$. A single level is fine, until it isn't. The required depth is determined by the structure of the problem and the selection pressure it generates, not by the theory in advance. The author's computational system has two explicit levels; there is no formal argument that two is sufficient in general.

   *Hypothesis (M-depth driven by selection pressure).* Each level $M^{(n)}$ is subject to the same minimization imperative as the level below: it will be displaced or refined if a better $M^{(n+1)}$ exists and the selection pressure is sufficient to instantiate it. The hierarchy has a natural termination at level $n^*$ only if the gradient at that level reaches zero — no available $M^{(n^*+1)}$ can improve the learning mechanism further given current $W$. Whether such a fixed point exists, and whether it is reachable in finite time under realistic selection pressure, is open. Evolution [Hinton1987] and meta-learning [FinnEtAl2017, Thrun1998] are empirical candidates for $M^{(2)}$; whether anything instantiates $M^{(3)}$ in known systems remains unclear.

8. **Genetic encoding and the developmental sufficiency requirement.** The encoding hierarchy (Definition 8)
   describes a ratchet: proven behaviors promote toward $L_0$ as they meet the rigor threshold
   $\theta_0 \approx 1$ (Theorem 4). The question is what the asymptotic product of evolutionary
   gradient descent encodes [Schrodinger1944, MaynardSmith1995, Hinton1987, England2013].

   The framework does not resolve the content of genetic encoding — what specific mixture of $M$, $(F_0, K_0)$, and developmental constraint DNA carries across all levels is an open empirical question. The claim is narrower: whatever genetic encoding carries must be *sufficient to develop the entity* — sufficient to bootstrap an organism capable of building $(F, K)$ through interaction with $W$. The model is compatible with genetic encoding at any level of the hierarchy, in any proportion, provided the developmental output is a system that can run the gradient.

   What the framework does predict about the evolutionary rigor threshold: specific $(F, K)$ are environment-dependent — a skill that reduces $U$ in one environment may not transfer across the full distribution of environments the lineage will encounter. Content that clears $\theta_0$ must generalize across that distribution. Developmental machinery — the programs, architectures, and gradient signals that build $(F, K)$ — is more likely to generalize than specific content, because it is environment-agnostic. The framework predicts selection pressure favoring developmental generality over content specificity at the highest rigor levels; it does not predict that specific $(F_0, K_0)$ is absent.

   This is consistent with the observation that we are not born with the context window of our parents: individual $(K, F)$ — everything learned in one lifetime — clears $\theta_n$ but not $\theta_0$. The inheritance mechanism transmits what is sufficient for development, not what was learned. The diversity of knowledge across individuals with similar genetic endowment is the expected outcome: the same developmental machinery operating on different environments produces different $(K, F)$.

   Cultural transmission (language, writing, institutions) provides a parallel channel for $(K, F)$ at lower fidelity but higher speed — $K_{collective}$ functioning as external inheritance (Corollary 15), serving a transmission function that genetic encoding cannot match for individual-lifetime content. The two channels operate at different rigor thresholds and different timescales.

   Formal questions that remain open: the precise allocation between $M$-encoding and $(F_0, K_0)$-encoding across evolutionary lineages; whether the modularity of genetic regulatory networks is derivable from the type structure of $M : (F, K) \times \mathcal{L} \rightarrow (F, K)$ at the evolutionary scale; and what determines $\theta_0$ in terms of environmental distribution statistics.

9. **The AGI condition.** If $\mathcal{A}$ is necessary for genuine intelligence (Definition 7) and $\mathcal{A}$ cannot be imposed as an external training objective without becoming a simulation of the gradient rather than actual descent on it: what conditions are sufficient for $\mathcal{A}$ to emerge in an artificial system?

   The framework does not resolve this. What it establishes is necessary: $\mathcal{A} > 0$ requires a real gradient — uncertainty that is genuine, not simulated — and $\eta_M > 0$ requires that the system's learning mechanism actually responds to it. Whether these conditions can be instantiated artificially, and what architectural or grounding requirements they impose, is an open question that the framework frames but does not answer.

10. **The next precipitation: normal form priority and extraction protocol.** The framework establishes that $M_{\text{res}}$ is the unprecipitated cross product and that each precipitation reduces per-entity operational complexity. The open question is: given the current state of $M_{\text{res}}$ and $K_{collective}$, which factor is most achievable to extract next, and what is the protocol for identifying and extracting it?

*Priority criterion.* "Most achievable" is under-defined by the framework. Natural candidates: (a) the factor whose precipitation threshold $\theta$ (Theorem 4) is closest to being met given current $K_{collective}$ — the factor requiring the smallest remaining evidential gap; (b) the factor producing the largest reduction in $|M_{\text{res}}|$ per unit of additional evidence; (c) the factor with the strongest cross-entity covariance signal in $K_{interaction}$ — where most entities' projections agree, suggesting genuine structure rather than projection artifact. These need not rank identically: the most evidentially ready factor may not be the one with the highest leverage. A principled priority function over these criteria does not exist in the framework as currently stated.

*The identification signal.* A partial answer via Theorem 11: convergence of independent agents on the same behavior pattern is evidence of an underlying precipitable factor — the pattern is the shadow of the factor in $K_{interaction}$. When many entities independently arrive at the same proposition from distinct starting points, that proposition likely grounds a precipitable dimension. The competing bounds problem (Open Question 6) may be directly connected: when multiple $f_i \in F$ compete at the same region of $W$, the competition may indicate an unnamed dimension whose precipitation would resolve the conflict. Competition at the action level is a signal of unresolved structure at the $M$ level.

*Extraction and verification.* Necessary condition for a genuine new factor $F^{(k)}$:

$$I(F^{(k)}\,;\, w \mid F^{(1)}, \ldots, F^{(m)}, K) > 0$$

— residual information gain after conditioning on all existing named factors. The implied protocol: (1) identify recurring propositions in $K_{interaction}$ across multiple entity subsets; (2) hypothesize the minimal factor $F^{(k)}$ whose naming would ground those propositions without derivation from existing named factors; (3) test at multiple entities — if independently adopting $F^{(k)}$ achieves $O(1)$ access to the propositions it grounds, the factor is real. If naming $F^{(k)}$ yields no access advantage — if the grounded propositions remain $O(|M_{\text{res}}|)$ — the candidate is a re-expression of existing named factors, not a genuine new dimension. The collapse case (Step 2b of Theorem 10) is a special instance: two apparent dimensions $D_a$ and $D_b$ are the same factor when naming one yields $O(1)$ access to what the other grounded.

This question is not merely theoretical: it is the operational question any entity with $\mathcal{A} > 0$ faces at the frontier of its $M_{\text{res}}$. The OKG framework described in Section 1 is an attempt to apply exactly this protocol to the domain of software engineering — identifying which actions have sufficient evidence to be precipitated as named factors and which remain in $M_{\text{res}}$, and which apparent distinctions between actions are the same underlying factor observed from different positions in $W$.

11. **DNA as the transitive closure of environmental knowledge encoding.** Open Question 8 asks what genetic encoding carries — developmental machinery vs. specific content. A sharper conjecture concerns the *structure* of what is encoded: if knowledge necessarily has graph structure (Section 3), and genetic encoding is the product of evolutionary gradient descent on $K$, then the genome may represent the *transitive closure* of the knowledge graph under environmental constraint — specifically, the closure of all dependencies required for replication that cannot be assumed to be provided by the environment.

   On this view, the genome encodes what the organism cannot defer to context. What the environment reliably provides — ambient energy gradients, chemical precursors, physical constants within the lineage's range — is not encoded; what cannot be assumed is. This is the bounded active context principle (Definition 4) applied at the evolutionary timescale: the genome is the minimum $L_0$ such that organism $\oplus$ environment $\supseteq$ replication capability. Equivalently, it is the maximum knowledge transmissible given the channel capacity imposed by transmission fidelity $\lambda$ and environmental noise.

   This conjecture makes a structural prediction: genetic encoding is not a linear sequence but a graph, and the linear sequence is a path encoding of that graph — a traversal, not the structure itself. This is independently confirmed by the structure of genome assembly: sequencing reads cannot be assembled without first constructing a de Bruijn graph of k-mer overlaps, and the path through the graph is the genome. The pangenome — the correct representation of a species' genetic information — is a directed graph of variation paths, not a single linear sequence. The graph structure was not imposed by bioinformaticians; it was recovered because the linear model was insufficient. The framework predicts this: dependency closure is always a graph; a linear encoding loses topology.

   A further conjecture: if mutation is the mechanism by which the genome updates toward the transitive closure of environmental knowledge, then evolutionary adaptation may be better characterized as *directed encoding* than uniform random sampling. The random mutation model is the maximum-entropy prior over the mutation distribution. The conjecture is that the actual distribution has lower entropy — it is shaped by the system's response to environmental gradient signals. This is consistent with observed variation in mutation rates across genomic loci, across stress conditions (elevated mutation rates under environmental pressure), and with mechanisms such as somatic hypermutation in immune cells (which is explicitly directed: the organism encodes novel environmental information — antigen structure — into the B-cell lineage in real time). Random mutation (e.g., radiation damage) is a special case — noise on the channel — not the general mechanism.

   The timescale argument is supportive: fixation by neutral drift requires $O(N)$ generations under standard population genetics; observed adaptive evolution (antibiotic resistance, rapid morphological change under strong selection) is orders of magnitude faster at the same population sizes. If the mutation distribution is informed by gradient signals rather than drawn from a flat prior, the effective search space is much smaller and the observed timescales are the expected result.

   These are conjectures, not claims of the framework. The framework predicts that selection pressure favors developmental generality at $\theta_0$ (Open Question 8) and that knowledge has graph structure (Theorem 3). Whether genetic encoding specifically implements the transitive closure of environmental knowledge, whether mutation distributions are meaningfully gradient-informed beyond what selection alone explains, and whether the soma-to-germline channel has sufficient bandwidth to support real-time encoding updates, are empirical questions the framework does not resolve.

*A limiting claim: single-class ceilings.* There is reason to believe that a collective composed entirely of entities from a single sensing class — sharing the same $\sigma$-structure and therefore the same observable subspace of $M$ — exhausts within-class precipitation at some finite point. Entities of the same class can jointly observe the dimensions of $M$ accessible to their shared $\sigma$; once all cross-dimensional factors within that subspace have been extracted, further co-variance analysis within the class yields no new factors. The ceiling is the effective rank of the class's jointly observable subspace, which is strictly less than $\text{rank}(M)$ if dimensions exist that are accessible only to entities of a different class. Beyond this ceiling, further precipitation requires inter-class exchange — entities whose $\sigma$ accesses dimensions outside the current subspace. This makes a testable prediction: the marginal precipitation value of adding a new entity to a homogeneous collective decreases and approaches zero; the marginal precipitation value of adding the first entity of a new class is non-zero regardless of collective size. The formal characterization of sensing classes, class-observable subspaces, and the inter-class exchange requirement is an open problem.
