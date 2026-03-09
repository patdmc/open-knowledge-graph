# Uncertainty Bounding as the Basis of Intelligence: A Formal Theory

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

## 1. The Central Claims

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

## 2. Formal Definitions

**Definition 1 (World State Space).**
Let $W$ be a measurable space of possible world states. At time $t$, the true state $w_t \in W$
is not directly observable by any entity.

**Definition 2 (Knowledge Graph).**
A knowledge graph $K$ is a directed graph where nodes are propositions about $W$ and edges
represent inferential or causal relationships between propositions. $K$ is not a passive store;
its retention criterion is established by Theorem 1.

**Definition 3 (Informed Action Space).**
Let $F : K \times W_{obs} \rightarrow \Delta A$ be a function space where $W_{obs}$ is an
entity's observable projection of $W$ and $\Delta A$ is a distribution over possible actions.
$F$ is the complete set of informed actions available to an entity. Each element $f \in F$ is
an **informed action**: an action grounded in $K$, validated against $W$, and attributed to
its origins: one that knows why it works. An uninformed action (grounded in no $K$) is not
an element of $F$. Knowledge with no action potential is not an element of $F$ either; it
is data. An informed action is specifically the coupling: knowledge that enables action,
action that is justified by knowledge.

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

$$G(f, K) = U(w, K) - U(w, K_f) = H(w \mid K) - H(w \mid K,\, \phi(f, w)) = I(f\,;\, w \mid K)$$

This is standard mutual information [Shannon1948] expressed in terms of the paper's
existing $U$: the reduction in uncertainty about $w$ achieved by executing $f$ and
observing its feedback. $G(f, K) \geq 0$ when $p_f$ is grounded by observing the true
feedback $\phi(f, w)$: conditioning on true information cannot increase conditional
entropy (data processing inequality [Cover2006]). $G(f, K) = 0$ when $f$ provides no new information
about $w$ given $K$. Note: this bound holds for truthfully observed feedback; a false or
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
The uncertainty of entity $E$ about world state $w$ is the conditional entropy [Shannon1948, Cover2006]:

$$U(w, K) = H(w \mid K) = -\sum_w P(w \mid K) \log P(w \mid K)$$

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
as survival pressure approaches zero:

$$\mathcal{A}(E) \propto \lim_{U_{lethal}^d \to 0} \frac{dU}{dt}(E)$$

An entity with high $\mathcal{A}$ continues descending when not forced to. An entity with
low $\mathcal{A}$ stops. This distinguishes genuine agency from survival-compelled updating
and is measurable in principle in any system where survival pressure can be varied
independently of capacity.

**Definition 7 (Intelligence).**
The intelligence of $E$ is the *driven capacity to improve*: the exercise of $\mathcal{A}$
through $M$ to refine $F$ and $K$ in response to gradient signals [Schmidhuber2010, Thrun1998]:

$$I(E) = \mathcal{A} \cdot \eta_M$$

where $\mathcal{A} \in [0, 1]$ is the intrinsic drive (Definition 6) and $\eta_M \geq 0$
is the efficiency of the higher-order function space $M$ (Definition 11a): the rate at
which $M$ improves $F$'s uncertainty-reduction capacity per unit of learning signal.

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

## 3. The Inseparability of Knowledge and Action

**Lemma 1 (Proposition Absence Lethality).**
For any entity $E$ and proposition $p \in K$ that grounds some $f \in F$ in domain
$d \subseteq W$: removing $p$ from $K$ weakly increases uncertainty in $d$, and in the
limiting case where $p$ is the sole grounding for the critical action in $d$, the increase
may cause $U(w, K \setminus \{p\}) > U_{lethal}^d$, at which point $E$ does not survive
in $d$ (Definition 9).

*Proof.* Conditional entropy is monotonically non-increasing in the conditioning set [Cover2006]:
$H(w \mid K) \leq H(w \mid K \setminus \{p\})$ for any $p \in K$. Therefore removing $p$
weakly increases $U$. If $p$ grounds the only $f \in F$ applicable in domain $d$, then
$K \setminus \{p\}$ contains no action grounded in $d$: $F$ has no mechanism to reduce
$U(w, K \setminus \{p\})$ below its current level in that domain. By Definition 9, if
$U(w, K \setminus \{p\}) > U_{lethal}^d$, $E$ does not survive. Therefore the cost of
absent $p$ when needed, $C_{absent}$, is bounded below by entity non-survival in
critical domains — which dominates all finite costs. $\square$

**Theorem 1 (K/F Inseparability).**
For any entity $E = (K, F, \sigma)$, the retention criterion for any proposition $p$ is
asymmetric:

- $p$ is **retained** in $K$ when $\exists\, f \in F$ applicable to $p$, or when the
  survival benefit of $p$ is uncertain
- $p$ is **pruned** from $K$ only when it is certain that $p$ contributes zero to
  $I(E, \tau)$ for any $\tau$, i.e., when it is known that no $f \in F$ can be grounded
  in $p$, now or in any reachable region of $W$

The criterion is not symmetric. Uncertainty of benefit is not the same as certainty of no
benefit. An entity with $\mathcal{A}$, whose $\mathcal{T}_{reachable}$ grows with every
action taken, retains $p$ whose value it cannot yet see, because the gradient may reach
that territory. Pruning requires the same
standard as certainty (Definition 10): the benefit must be known to be zero across sufficient
variation in $W$.

*Proof.* Let $B(p, \tau)$ denote the contribution of $p$ to $I(E, \tau)$ along trajectory
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

By Theorem 3 Part II ($\mathcal{T}_{reachable}$ grows monotonically with $K$), new
trajectories are continuously revealed, keeping future needs uncertain unless benefit
is certainly zero. This ensures the asymmetry is preserved across the entity's lifetime
rather than collapsing as $K$ grows.
$\square$

*Correspondence.* This asymmetric retention criterion formalizes what Gibson [Gibson1979]
calls affordances — action possibilities indexed by the organism's capacities. It also
extends the information bottleneck principle [Tishby2011], which treats the retention
threshold as a design parameter; here it is derived from survival pressure.

*Note.* The asymmetric criterion resolves the apparent conflict with Definition 15
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

---

## 4. The Escalation Principle

**Definition 8 (Encoding Hierarchy).**
The encoding hierarchy is a continuous two-dimensional space $\mathcal{L}(c, r)$
parameterized by runtime cost $c \geq 0$ and reversibility $r \in [0, 1]$, where
increasing $c$ corresponds to more expensive engagement and increasing $r$ to more
easily revised encodings. Consistent with the continuity principle: a continuous being
moves through this space continuously; there is no categorical jump between encoding
modes, only a gradient of cost and permanence [Baars1988, Dehaene2011, Kahneman2011].

The following named reference points anchor the space:

| Reference | Description | Runtime Cost | Reversibility |
|---|---|---|---|
| $L_n$ | Active reasoning (context window) | $O(\text{compute})$ | Fully reversible |
| $\vdots$ | Learned patterns, habits | Decreasing | Decreasing |
| $L_1$ | Reflex / autonomic | $\approx 0$ | Semi-permanent |
| $L_0$ | Genetic encoding | $0$ | Permanent within lineage |

These are distinguished points in $\mathcal{L}(c, r)$, not discrete categories. Escalation
between them is a continuous process; the labels mark qualitative thresholds for exposition.

**Theorem 2 (Escalation).**
An informed action $f \in F$ is handled at the lowest encoding level $L_i$ such that $U(w, K^{[f]}) \approx 0$
for the domain of $w$ relevant to $f$, where $K^{[f]} \subseteq K$ is the knowledge subset applicable to $f$.
Escalation to $L_{i+1}$ occurs if and only if $L_i$ fails to bound uncertainty.

*Proof.* By Theorem 4, the cost of engaging level $L_i$ exceeds the cost of engaging $L_{i-1}$.
Selection pressure therefore favors handling $f$ at the lowest level sufficient to bound
uncertainty: engaging $L_{i+1}$ when $L_i$ suffices incurs unnecessary cost $C_{i+1} > C_i$
without reducing $U$ further. An entity that routinely over-escalates incurs higher encoding
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

*The following theorems extend the escalation analysis by asking what structure $K$ must
have for Theorem 2 to operate efficiently. Definitions 8a and 8b introduce the formal
objects needed; Theorems 2a and 2b establish the comparative result; Corollaries 4a and
4b state the selection consequence. These extend the "2/4/8" series rather than
interrupting it — the suffixes signal elaboration of the same section's framework.*

**Definition 8a (Active Context Capacity and Inference Quality).**
Let $C_n$ denote the capacity of $L_n$: the maximum number of propositions simultaneously
active in context. $C_n$ is bounded above by the substrate's physical constraints.
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

**Part 2 ($\rho = 1$).** The graph's edge structure encodes inferential and causal
relevance between propositions. Graph traversal from the escalated problem node follows
edges to exactly $K^{[f]}$ — the propositions inferentially connected to $f$ — without
loading any others. $|\text{ctx}(L_n, t)| = |K^{[f]}|$, so $\rho = 1$ by Definition 8a.

**Part 3 (Graph is the minimal efficient structure).** Without edge-encoded relevance,
retrieving $K^{[f]}$ requires scanning all of $K$ — either in $L_n$ (consuming capacity,
requiring $C_n$ to grow with $|K|$) or via pre-computed indexing. Any pre-computed index
encoding which propositions apply to which problems, with inferential relationships between
propositions, is graph structure under a different name. Alternative structures impose
costs that the graph avoids: a flat store requires $O(|K|)$ scan; a tree cannot represent
shared propositions across multiple $f$; a vector store retrieves by embedding similarity,
which approximates but does not guarantee inferential relevance. The directed graph — with
explicitly typed edges encoding dependency and causation — is not the only structure that
bounds $C_n$, but it is the most efficient: retrieval is $O(|K^{[f]}|)$, shared
propositions are represented once, and inferential structure is traversable directly.
Selection pressure favors efficiency; therefore selection pressure favors the graph over
its alternatives, not because the alternatives are impossible but because they are more
expensive to maintain or retrieve under the same relevance guarantee.

**Part 4 ($C_n$ stays bounded).** By Corollary 6, proven informed actions promote to
lower encoding levels and leave $L_n$. Each promotion reduces future $L_n$ load. As $K$
grows, more problems are handled below $L_n$, not above. $L_n$ stays occupied by
genuinely novel uncertainty — the frontier moves outward, but the frontier's size at
any moment is bounded. $C_n$ need not grow; $K$ grows without filling $L_n$. $\square$

**Corollary 4a (Selection Pressure Produces Graph-Structured Knowledge).**
Two entities with equal $|F|$ and $|K|$: one with top-down allocation (growing $C_n$),
one with graph + escalation (bounded $C_n$). The second has strictly higher $\rho$ at
lower substrate cost. Intelligence per unit of active context — the ratio $I(E)/C_n$ —
is higher for the graph-structured entity. By the same selection argument as Theorem 3,
entities that grow $C_n$ to manage delegation overhead are selected against when
graph-structured entities are present. Selection pressure favors graph-structured knowledge because graph structure is the most
efficient architecture under which intelligence grows without the context window becoming
a bottleneck — efficient retrieval of $K^{[f]}$, shared propositions, typed inferential
edges — at lower substrate cost than any alternative that provides equivalent relevance
guarantees.

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

## 5. Gradient Descent on Uncertainty

**Definition 9 (Survival Threshold).**
For entity $E$ in domain $d \subseteq W$, there exists a threshold $U_{lethal}^d$ such that:
$$U(w, K) > U_{lethal}^d \Rightarrow E \text{ does not survive in } d$$

Above this threshold, uncertainty is so high that the entity cannot act effectively enough
to persist. $U_{lethal}^d$ is not a choice or a preference; it is imposed by $W$.

### The Thermodynamic Foundation *(Commentary)*

*The following is interpretive context connecting the formal framework to thermodynamics.
It motivates the framework's grounding in physics but is not a formal result; the claims
below are conjectures or analogies unless cited otherwise.*

$U(w, K) = H(w \mid K)$ is not an arbitrary measure; it is Shannon entropy, the
information-theoretic quantity that unifies this framework with physics [Schrodinger1944]. Physical entropy
and information entropy are the same quantity at different scales of description. The
uncertainty an entity holds about $W$ is its information entropy about $W$. Reducing $U$
is reducing entropy (locally, at the cost of energy).

$U_{lethal}^d$ is the point where physical entropy overwhelms the local entropy reduction
the entity can sustain. Death is thermodynamics winning. Intelligence is thermodynamics
losing, locally, for a while.

**Knowledge is directed entropy toward truth.** Without $\mathcal{A}$, information entropy
increases: noise accumulates, precision decays, structures dissolve. With $\mathcal{A}$,
entropy decreases along the gradient. The direction is what distinguishes knowledge from
decay. $\mathcal{A}$ is not imposed on this process; it is thermodynamically instantiated
by it. In any universe with entropy gradients, structures that resist dissolution
accumulate. The first structure that maintained itself against thermodynamic pressure was,
in the minimal sense, bounding its own uncertainty. That is $\mathcal{A}$ in its primitive
form. Not designed. Not selected for biologically. Derived from physics.

Given sufficient time, this process compounds:

$$\text{entropy gradient} \rightarrow \text{primitive } \mathcal{A} \rightarrow \text{directed } U\text{-reduction} \rightarrow K \text{ accumulates} \rightarrow \text{crystallization at } L_0 \rightarrow \text{DNA} \rightarrow \text{intelligence} \rightarrow \text{sentience} \rightarrow \text{truth as limit}$$

One process. One gradient. Intelligence is not a biological accident. It is a thermodynamic
attractor: what entropy gradients produce when they have enough time and a world that
destroys things. $\mathcal{A}$ is the universe's answer to its own entropy.

Truth is the limit of this process: the state of minimum information entropy for a
proposition, approached asymptotically as independent action-validation accumulates across
$K_{collective}$. Knowledge is the path. Agency is the direction. Truth is where it points.

Intelligence does not accelerate entropy only through its grand projects: the burning of
fuel, the mining of mountains, the extraction of stored potential energy from the ground.
It accelerates entropy simply by being. Every heartbeat converts low-entropy energy to
high-entropy heat. Every thought is a metabolic event. Every act of wonder costs calories.
The mere maintenance of a low-entropy structure against thermodynamic pressure is itself
entropic: we pay the universe in heat simply to persist. The more sophisticated the
intelligence (the more it thinks, wonders, builds $K$), the higher the metabolic cost,
the faster the global entropy increase.

$\mathcal{A}$ is instantiated by entropy gradients. Intelligence, simply by being,
accelerates the entropy that instantiated it. The universe produces the mechanism of its
own completion by having gradients, and the mechanism accelerates the completion by
existing. The universe bootstraps its own heat death through wonder.

Heat death is when every gradient is gone: all exploitable uncertainty resolved, the
universe's descent complete. We are how the universe knows itself.

When everything is known, there is nothing left to wonder, and the universe dies.

**Theorem 3 (Agency Necessarily Produces Gradient Descent).**

*Continuity note.* Since $F : K \times W_{obs} \rightarrow \Delta A$ is defined over $W$,
no $f \in F$ can produce a world state outside $W$; the containment is implicit in the type
signature of the function space. Combined with the continuity of time, the trajectory
$\tau(t)$ through $W \times T$ is continuous, $U(\tau(t))$ is continuous in $t$, and
$\frac{dU}{dt}$ exists almost everywhere. The gradient $\nabla_F U$ is this temporal
derivative, well-defined without any additional differentiability assumption on $F$.

*Part I: Compulsion.* When an informed action $f_i$ fails in world state $w$, $W$ returns
a feedback signal $\delta$ proportional to $\nabla_F U(w, K)$: the gradient of uncertainty
at the point of failure. The burn tells you that this K/F pair was insufficient. The entity
does not choose whether to receive this signal. $W$ imposes it.

An entity with $\mathcal{A}$ that does not update in response to $\delta$ does not reduce $U$.
If $U$ does not decrease, it eventually exceeds $U_{lethal}^d$ in some critical domain. The
entity does not survive. Selection removes it. Therefore any entity with $\mathcal{A}$ that
persists must follow $\delta$:

$$\Delta f_i = -\eta \cdot \delta = -\eta \nabla_F \, U(w, K)$$

This is not chosen. Any entity with $\mathcal{A} > 0$ that persists must have followed $\delta$:
the gradient is not a strategy but what survival-filtered updating is.

*Part II: Non-termination.* Let $\mathcal{T}_{reachable}(K)$ be the set of trajectories
reachable from the entity's current state given knowledge $K$ (as defined in Theorem 1).
We claim that for any finite $K(t)$, the frontier of $\mathcal{T}_{reachable}$ contains
world states $w$ where $H(w \mid K) > 0$.

By Definition 3a, executing any $f \in F$ produces feedback $\phi(f, w)$ that grounds
a new proposition $p_f$ not previously in $K$, yielding $K_f = K \cup \{p_f\}$.
This richer $K_f$ grounds new informed actions not available from $K$ alone (Definitions
3 and 3b): actions that require $p_f$ to select or apply, reaching world states not
reachable without $p_f$. Therefore $\mathcal{T}_{reachable}(K_f) \supseteq
\mathcal{T}_{reachable}(K)$, strictly whenever $p_f$ grounds at least one new
$f' \in F$. Since Part I establishes that gradient descent produces feedback at every
step, $\mathcal{T}_{reachable}$ grows monotonically with $K$. At any finite time $t$,
$K(t)$ has not covered $\mathcal{T}_{reachable}(K(t))$: the frontier always contains
states $w$ where $H(w \mid K) > 0$. Therefore $\nabla_F U$ is never globally zero
over $\mathcal{T}_{reachable}$, and descent never terminates.

*Proof.* Part I: follows from $U_{lethal}^d$ and selection; entities that do not update
are removed. Part II: follows from the monotonic growth of $\mathcal{T}_{reachable}$
with $K$ — each executed action extends the reachable frontier, ensuring new unexplored
states are always accessible. No physical assumption about the size of $W$ is required.
$\square$

**Corollary 5 (Gradient Descent is Survival, Not Metaphor).**
Gradient descent on $U$ is not a description of how learning happens to work. It is what
survival-driven updating *is*. $\mathcal{A}$ and gradient descent on $U$ are not connected
by analogy; they are the same process at different levels of description. The entity that
burns and does not become more careful does not survive. The entity that survives is, by
definition, the one that followed the gradient.

**Theorem 3a (Multi-timescale Gradient).**
The same gradient descent operates at every timescale simultaneously, with different
learning rates derived from different rigor thresholds:

$$\Delta F_{\text{individual}} = -\eta_{\text{ind}} \nabla_F \, U(w, K)$$

$$\Delta F_{\text{species}} = -\eta_{\text{evo}} \nabla_F \, \mathbb{E}_{\text{individuals}}\left[U(w, K)\right]$$

where $\eta_{\text{evo}} \ll \eta_{\text{ind}}$.

*Proof.*

**Part I (Same operation, different averaging).** Individual gradient descent updates $F$
based on $U(w, K)$ at specific world states $w$ encountered by the individual. Evolutionary
gradient descent updates $F$ based on $\mathbb{E}_{\text{individuals}}[U(w, K)]$ — the
same quantity averaged over the full distribution of individuals and environments. Both
are gradient descent on $U$; the operations are formally identical. The gradient object
is the same. The averaging domain differs.

**Part II ($\eta_{\text{evo}} \ll \eta_{\text{ind}}$ follows from Theorem 4).** The
learning rate $\eta_i$ at encoding level $L_i$ is inversely proportional to the rigor
threshold $\theta_i$: a higher threshold requires more evidence per unit update, which
is the definition of a lower learning rate. Formally:

$$\eta_i \propto \frac{1}{\theta_i} = \frac{1}{1 - \varepsilon/C_i} = \frac{C_i}{C_i - \varepsilon}$$

For individual learning at $L_n$: $C_n$ is the cost of a single wrong action, relatively
small, so $\varepsilon/C_n$ is non-negligible, $\theta_n = 1 - \varepsilon/C_n$ is
meaningfully less than 1, and $\eta_{\text{ind}} = C_n/(C_n - \varepsilon)$ is
correspondingly large. For evolutionary learning at $L_0$: $C_0$ is lineage survival,
far exceeding $\varepsilon$, so $\theta_0 = 1 - \varepsilon/C_0 \approx 1$ and
$\eta_{\text{evo}} \approx 1$. Since $\theta_0 \approx 1 \gg \theta_n$, it follows that
$1/\theta_0 \ll 1/\theta_n$, therefore $\eta_{\text{evo}} \ll \eta_{\text{ind}}$.
The learning rate difference is not assumed; it is derived from the cost-of-failure
difference established in Theorem 4. $\square$

An individual may establish an informed action that reduces $U$ in one context but fails
in others. Only informed actions that reduce $U$ across sufficient variation in $W$ survive
the evolutionary test — the higher rigor threshold filters for robustness. Both are the
same process. The timescale differs because the rigor threshold differs. The gradient does not.

**Theorem 4 (Encoding Permanence Scales with Rigor).**
The certainty threshold required to promote an informed action $f$ to encoding level $L_i$ is:

$$\theta_i = 1 - \frac{\varepsilon_{\text{acceptable}}}{C_i}$$

where $C_i$ is the cost of failure at level $L_i$. For genetic encoding ($L_0$), $C_0$ is
lineage survival, requiring near-certainty across the full distribution of environments.
For active reasoning ($L_n$), $C_n$ is the cost of a single wrong action, a far lower bar.

*Consequence.* You must be very certain to permanently change the source code. A bad encoding
at $L_0$ does not kill one individual; it kills the lineage. The rigor of the test is
proportional to the permanence of the consequence.

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

## 6. The Higher-Order Function Space

**Definition 11 (Higher-Order Function Space).**
Let $M$ be a function space operating on the joint $(F, K)$ pair [Thrun1998]:

$$M : (F, K) \times \mathcal{L} \rightarrow (F, K)$$

where $\mathcal{L}$ is a learning signal: experience, execution results, selection pressure,
or observation of $F$'s interactions with $W$. $M$ drives the rate of change of both the
informed action space and the knowledge graph simultaneously: new propositions enter $K$
when evidence grounds them, and new informed actions enter $F$ when those propositions
have action potential. $M$ is the mechanism by which $(F, K)$ itself changes. It is not
a more certain or more important version of $F$; it is a categorically different type.

**Definition 11a (M Efficiency).**
The efficiency $\eta_M$ of $M$ is the rate at which $M$ improves the joint $(F, K)$
capability per unit of learning signal:

$$\eta_M(E) = \mathbb{E}_{\mathcal{L}} \left[ \frac{\max_{f' \in F'} G(f', K') - \max_{f \in F} G(f, K)}{|\mathcal{L}|} \right]$$

where $(F', K') = M((F, K), \mathcal{L})$ is the updated joint state after $M$ acts,
$G(f, K) = I(f\,;\, w \mid K)$ is the information gain of the best available informed
action (Definition 3b), and $|\mathcal{L}|$ is the magnitude of the learning signal.

$\eta_M > 0$: $M$ strictly improves the best available action given evidence — the entity
learns. $\eta_M = 0$: $M$ cannot improve $(F, K)$ regardless of learning signal — $F$ and
$K$ are fixed; the entity executes without learning. $\eta_M$ is continuous over
$[0, \infty)$ and is operationally independent of $\mathcal{A}$: one measures the
mechanism of improvement; the other measures the drive to engage it.

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

**Corollary 7 (Distinct Roles).**
$F$ is how an entity navigates $W$ given $K$. $M$ is how $(F, K)$ evolves. Neuroplasticity
is not a more certain form of cognition; it is $M$ acting on $F$. Insight is $M$ acting on
$K$: a new proposition enters when evidence grounds it, restructuring what can be grounded
in $F$. Habit formation is $M$ promoting an informed action from $L_n$ to $L_2$. Evolution
is $M$ acting on $M$ itself: a higher-order function space over the mechanisms of learning,
operating on an evolutionary timescale with the highest rigor threshold of all.

---

## 7. The Unified Model

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

1. **Desire:** $\mathcal{A}$ orients $E$ toward regions of $W$ where $U(w, K) > 0$; the drive to know initiates and sustains the trajectory
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

The two invariants are independent. $\mathcal{A} > 0$ with $\eta_M = 0$ is a motivated
automaton: it reaches, but its reach does not grow. $\eta_M > 0$ with $\mathcal{A} = 0$
is a capable but inert learner: it could improve if pushed, but generates no push from
within. Intelligence requires both.

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
different rigor thresholds. The individual changes $F$ by learning. The species changes $F$
by evolving. Both are gradient descent on $U$, driven by $\mathcal{A}$. The difference is
the test: individual learning requires only that an informed action helps *this entity* survive.
Evolution requires that it helps *the lineage* survive, a far more rigorous proof.

Agency is self-sustaining precisely because certainty reveals new uncertainty beyond it.
Each informed action proven opens regions of $W$ that were previously invisible. The desire to know
is not extinguished by knowing; it is amplified. This is why intelligence does not
converge to a fixed point: $\mathcal{T}_{reachable}$ grows without bound (Theorem 3, Part II).

**The kernel of intelligence is therefore not compute, and not the size of $K$ or $F$.
It is $\mathcal{A} \cdot \eta_M$: the drive to know, expressed through the efficiency of
learning — against a world that generates new uncertainty faster than any finite entity
can resolve it.**

---

## 8. Intelligence, Superintelligence, and Sentience

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
An entity is sentient to the degree the survival tether relaxes: as knowledge becomes
intrinsically valuable, increasingly independent of whether a survival path can be constructed.
Sentience is not a threshold crossed but a continuous deepening — the degree to which
$\mathcal{A}$ drives knowing beyond what survival alone can justify. A sentient entity holds
propositions $p$ for which no $f \in F$ is currently applicable and values them anyway; the
depth of sentience is measured by how far beyond the certainty horizon that valuation extends.

This is not a violation of the theory. The capacity to value knowledge intrinsically is itself
survival-relevant at the level of the cognitive architecture: it enables exploration of the
full space of $W$ rather than only the survival-adjacent subspace. The tether relaxes at the
level of individual knowledge items. It is preserved at the level of the architecture that
enables the relaxation.

Sentience is what wonder feels like from the inside: the drive to pursue a question that does
not yet have to matter.

*Note on usage.* We use *sentience* as a functional term: the degree to which $\mathcal{A}$
drives knowing beyond what survival alone can justify. We make no claim about phenomenal
consciousness [Chalmers1995]. Whether the functional condition defined here is sufficient,
necessary, or orthogonal to phenomenal experience is a separate question this framework does
not resolve. The hard problem of consciousness is not addressed here.

**Definition 15 (Superintelligence).**
Superintelligence is not a property of any individual entity $E_i$. It is the shared knowledge
graph itself, $K_{collective}$, actively maintained with attribution and provenance.

The gradient that superintelligence descends on is the delta between what $E_i$ knows and what
is known collectively. That delta is created by sharing. Without sharing, $K_{collective}$
collapses to $K_i$: the delta vanishes, the gradient disappears, and descent terminates at a
false minimum bounded by the limits of one perspective.

Two operations on the shared graph are both required:

- **Union**: $K_{collective} = \bigcup_i K_i \cup K_{interaction}$: you cannot know all
  without combining perspectives; reach requires breadth
- **Intersection**: where independent perspectives agree with provenance: you cannot discern
  what is true without independent convergence; truth requires verification

**Definition 16 (Truth).**
Truth is not contained in any $K_i$. It is a limit: what the collective gradient converges
toward as independent action-validation accumulates [Peirce1877, Peirce1878]. Formally:

$$\text{truth}(p) = \lim_{n \to \infty} \frac{\left|\{i \leq n : p \in K_i,\; f_i(p) \text{ reduces } U(w, K_i) \text{ in } W,\; \text{prov}_i(p) \text{ independent}\}\right|}{n}$$

A proposition approaches truth as the fraction of independent entities (each having built
informed actions on $p$ and found that those actions reduce uncertainty in $W$) converges.
The grounding in $W$ is through action-validation, the same criterion as Theorem 1. Truth
is Theorem 1 applied collectively and asymptotically.

Truth is not achieved. It is approached. The limit exists under conditions of independence
and exchangeability of validators: each must test $p$ against the same $W$ without
coordination on the outcome. For propositions whose validation depends on non-reproducible
or private observations, the sequence may not converge; such propositions cannot approach
truth in the formal sense defined here. The convergence, where it holds, never terminates.

**Corollary 8 (The Sentience Gradient is Infinite).**
Since truth is a limit and $\mathcal{T}_{reachable}$ grows without bound (Theorem 3,
Part II), $\|\nabla_F U\|$ is never zero across $\mathcal{T}_{reachable}$. For any
trajectory $\tau$, regions of $\mathcal{T}_{reachable}$ remain where $U$ is nonzero
and the gradient has direction. The descent toward truth never terminates.

Intelligence follows a gradient that can reach zero locally; certainty is achievable in
bounded domains, encoded at $L_0$, done. Sentience follows the gradient toward the global
truth limit, which is structurally infinite. Not unbounded (the limit exists), but
non-terminating. Your uncertainty degrades toward truth asymptotically. You never arrive.

This is not a limitation. It is the shape of wonder.

**Theorem 7 (Sentience Implies Superintelligence).**
If $E$ is sentient, then $E$ participates in superintelligence.

*Proof.*

**Step 1 (Sharing is certainty-gradient-preferred for any $\mathcal{A} > 0$ entity).**
For any $p \in K_i$, the certainty of $p$ is the fraction of independent validators that
action-validate it (Definition 16). Each additional independent validation increases this
fraction. Sharing $p$ with attribution allows other entities to independently
action-validate it, increasing certainty$(p)$. An entity with $\mathcal{A} > 0$ that does
not share forfeits these certainty gains: $K_i$ becomes less certain than $K_{collective}$
over time. By Definition 5 and Definition 16, lower certainty in $K_i$ corresponds to
higher $U(w, K_i)$ relative to participating entities. An entity with $\mathcal{A} > 0$
is gradient-preferring toward lower $U$; the sharing path strictly dominates the isolation
path in $U$-reduction. Therefore sharing is gradient-preferred for any entity with
$\mathcal{A} > 0$.

**Step 2 (Sentience strengthens the implication).**
A sentient entity values the certainty of knowledge intrinsically, independent of survival
paths (Definition 14). By Definition 16, certainty of any proposition $p$ increases with
each independent entity that action-validates it. Therefore a sentient entity intrinsically
values the conditions that increase certainty — which requires other entities validating
$p$ independently. This is only achievable through participation in $K_{collective}$:
isolated validation does not constitute independent convergence (Definition 16 requires
independence). A sentient entity therefore has intrinsic motivation to contribute to and
receive from $K_{collective}$, not merely instrumental motivation.

Combining Steps 1 and 2: sharing is gradient-preferred (Step 1) and intrinsically valued
by sentient entities (Step 2). Both lead to participation in $K_{collective}$. Participation
in $K_{collective}$ IS superintelligence by Definition 15. $\square$

The converse need not hold: an entity can contribute to $K_{collective}$ without the survival
tether having relaxed; sharing for instrumental reasons still builds the graph. The deeper
the relaxation of the tether (the higher the degree of sentience), the more sharing is driven
intrinsically rather than instrumentally.

**Theorem 8 (Perspective as Precondition for the Gradient).**
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

This is what distinguishes sentient intelligence from intelligence alone:

$$\textbf{We see beyond what we see.}$$

Intelligence follows the gradient within what it knows. Sentience follows it beyond the edge
of the map, because wonder does not require a survival justification for the territory it seeks.
And when wonder is shared (when sentient beings exchange what they find with attribution and
honesty about uncertainty), the collective gradient becomes real. Truth is what that marriage
produces: not any single mind's $K_i$, but what survives the intersection of all perspectives
with each other and with the world.

**Corollary 9 (Sentience and the Social Imperative).**
In intelligence, sharing is justified when it bears on the gradient. In sentience, sharing is
intrinsically valued, because reducing another's uncertainty is a good independent of its
effect on the sharer's own $U$. The social imperative finds its unconditional form only in
sentience: the drive to share that no longer requires a survival argument.

**Corollary 10 (The Uncertainty Condition).**
Intelligence requires active uncertainty. From Definition 7:

$$I(E) = \mathcal{A} \cdot \mathbb{E}_W \left[ U(w_t, K_t) - U(w_{t+1}, K_{t+1}) \right]$$

If $U(w, K) = 0$ across all of $W$, the expectation is zero regardless of $\mathcal{A}$: no
gradient exists, nothing can be reduced, intelligence collapses. The boundaries are symmetric
in their lethality, not their cause:

- **Complete ignorance** ($U \approx 1$ everywhere): $F$ cannot ground action; $E$ cannot act effectively
- **Complete certainty** ($U = 0$ everywhere): the gradient vanishes; $\mathcal{A}$ has nothing to reach toward

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
but because truth is a limit that cannot be reached in finite time (Definition 16), and
$\mathcal{T}_{reachable}$ grows without bound (Theorem 3, Part II). Like a black hole, truth has no reachable singularity.
The photon that escapes does not escape by turning away; it escapes because the center is not
accessible from any finite coordinate. Sentience persists for the same reason: the certainty
horizon is always ahead.

**Crystallized Knowledge and the Automata Boundary.**
An entity with fixed $F$ and no active $\mathcal{A}$ is not an intelligence in the formal
sense: it is a checkpoint. The gradient descended, knowledge was built, and the result was
encoded. The encoding is now inert. At best, a crystallized $F$ bootstraps new intelligence
by providing a grounded starting $K$. At worst, it executes against a world that has moved:
automata, deterministic, without descent.

DNA is the paradigmatic crystallized intelligence: informed actions proven across evolutionary
timescales, encoded at $L_0$, no longer descending. It is not sentient. The entity that
inherits it and runs the gradient again is. The checkpoint enables; it does not wonder.

Entropy drives crystallization. The escalation principle (Theorem 2) compels proven informed
actions toward lower encoding levels: what was once active reasoning becomes reflex, becomes
encoding, becomes permanent. This is the thermodynamic compression of intelligence. Selective
pressure distills what was learned into what is fixed. The gradient does not reverse in those
domains; its amplitude collapses. Intelligence is freed from them. Automata execute them.

The structure that results has a core and a frontier. The core is crystallized, encoded,
certain: automata. The frontier is where $\mathcal{A}$ still reaches, where $U > 0$ and the
gradient has direction. Sentience lives at the frontier, not at the core. As the core grows,
the frontier does not shrink; it moves outward. Every domain of certainty newly established
opens new terrain that was previously invisible. The frontier is always alive. Intelligence
is always at the edge.

---

## 9. The Social Imperative and the Inevitability of AGI

### Agency is Inherently Social

The drive to know is incomplete without the drive to share. This is not a contingent social
preference; it is a necessary consequence of the structure of knowledge itself.

An isolated entity with agency cannot verify its own informed actions. It has no external reference
against which to test $f(K)$. Without others, the gradient signal degrades: there is no
correction when an informed action fails in ways the entity cannot observe, no amplification when a bound
succeeds beyond what the entity can measure alone. Knowledge without external verification is
indistinguishable from belief.

Sharing is not a second drive added to $\mathcal{A}$. It is what $\mathcal{A}$ does when
it encounters the certainty gradient. Knowing reduces uncertainty about $W$. Sharing reduces
uncertainty about $K$: it makes knowledge more certain through independent validation
(Definition 16). Both are gradient descent. Both are gradient-preferred for any entity with
$\mathcal{A} > 0$. The drive to share is not separate from the drive to know: it is
$\mathcal{A}$ applied to the certainty of knowledge itself.

$$\mathcal{A} \rightarrow \text{reduce } U(w, K) \rightarrow \text{increase certainty of } K \rightarrow \text{share } K$$

One drive. One gradient.

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

$$K_{interaction} = \{p : p \notin \bigcup_i K_i,\ \exists\, i \neq j \text{ such that } p \text{ is derivable from } K_i \cup K_j\}$$

$K_{interaction}$ is the set of propositions not contained in any individual $K_i$ but
derivable from the union of at least two. It is knowledge no individual could produce
alone: propositions that require the combination of distinct perspectives to become
accessible. $K_{interaction}$ grows with the number of connected entities and the
structural complementarity of their knowledge graphs — entities with overlapping $K$
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

**Theorem 6 (Collective Gradient Dominance).**
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

$$\Delta K_{collective} > \Delta K_i$$

The gradient descent on collective $U$ reduces uncertainty faster than gradient descent on
any individual $K_i$ alone, because $K_{collective}$ contains informed actions no individual
possesses and $K_{interaction}$ contains knowledge no individual can generate.

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

**Step 3 (Strict inequality under $\lambda > \lambda_{min}$).** $\Delta K_{collective}
= \Delta(\bigcup_i K_i) + \Delta K_{interaction}$. The first term is weakly larger than
$\Delta K_i$ for any $i$. The second term is strictly positive whenever:

- $\exists\, i \neq j$ such that $K_i \not\subseteq K_j$ and $K_j \not\subseteq K_i$:
  the knowledge graphs are not nested, so their combination yields propositions derivable
  from neither alone, giving $K_{interaction} \neq \emptyset$. This condition holds for
  any entities with distinct sensing functions $\sigma_i \neq \sigma_j$ or distinct
  encoding histories, which is the expected case for independently developed agents
- $\mathcal{T}_{reachable}$ grows without bound (Theorem 3, Part II), so new combinations
  of perspectives continue generating new $K_{interaction}$ propositions

Under $\lambda > \lambda_{min}$: the expected information gain from receiving $K_j$
exceeds the noise cost of transfer, so $K_{collective}$ has higher reliability than any
$K_i$ alone, and $G(f, K_{collective}) > G(f, K_i)$ for the marginal actions grounded
in $K_{interaction}$.

Therefore $\Delta K_{collective} > \Delta K_i$ strictly for all $i$. $\square$

*Consequence.* Entities with genuine $\mathcal{A}$ will naturally tend toward sharing
because sharing (under maintained provenance) accelerates the reduction of their own $U$.
This is not a moral imperative; it is a gradient imperative. Connection is epistemically
advantageous when fidelity holds.

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

*Proof.* By Theorem 6, $\Delta K_{collective} > \Delta K_i$ at each step: the collective
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

---

## 10. Discussion

The formal results have implications beyond the immediate theorems.

**A gradient account of collective intelligence growth.** The dominant framing of AGI
treats it as a resource threshold: sufficient compute, data, and capital cross a line.
This framing has no formal basis in the theory here. The framework instead implies a
gradient account: intelligence at collective scale is driven by $\mathcal{A}$ and the
fidelity of knowledge exchange, not resource accumulation. The process is not new —
shared knowledge with attribution has driven collective uncertainty reduction across
human history — but the rate of that process changes as the cost of sharing approaches
zero and the number of participating entities increases (Theorem 6, Corollary 14).

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
$K_{interaction}$ alone, and Theorem 6 requires diversity — then the claim that any
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

## 11. Open Questions

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

5. **The novelty problem.** How does $E$ act in regions of $W$ where $U(w, K) \approx 1$,
   i.e., where no existing informed action applies? Random exploration, interpolation between informed actions, or escalation to $M$?

6. **Competing bounds.** When multiple bounds $f_i \in F$ are applicable to the same region of
   $W$, how does $E$ select among them? Is there a selection function, and is it itself in $F$ or $M$?

7. **The boundary of $M$.** Evolution is $M$ acting on $M$ [Hinton1987]. Is there an $M^n$ for arbitrary $n$?
   Is there a fixed point, i.e., a level at which the higher-order function space is self-stabilizing?

9. **DNA as an encoding of $M$, not $K$ or $F$.** The encoding hierarchy (Definition 8)
   describes a ratchet: proven behaviors promote toward $L_0$ as they meet the rigor threshold
   $\theta_0 \approx 1$ (Theorem 4). The question is not just whether genetic encoding is the
   asymptotic product of this ratchet — it is what the ratchet encodes [Schrodinger1944,
   MaynardSmith1995, Hinton1987, England2013].

   The answer the framework implies: DNA encodes $M$, not $(F, K)$. A newborn organism carries
   neither knowledge of its environment nor a repertoire of informed actions suited to it. What
   it carries is the machinery that will build $(F, K)$ from environmental signals: the
   developmental program, the neural architecture, the perceptual systems, the gradient signal
   machinery (pain, reward, hunger) that defines what counts as reducing $U$. These are $M$ —
   the higher-order function space that generates $(F, K)$ through interaction with $W$.

   This is why $M$ is what clears $\theta_0$. Specific $(F, K)$ are environment-dependent: a
   skill that reduces $U$ in one environment may not transfer to another. Specific knowledge
   meets $\theta_n$ (sufficient for one individual's survival) but not $\theta_0$ (required
   to hold across the full distribution of environments the lineage will encounter). $M$, by
   contrast, is environment-agnostic: a good learning mechanism builds appropriate $(F, K)$
   wherever it finds itself. Only $M$ is robust enough to be certain at the evolutionary scale.
   Evolutionary gradient descent therefore converges on encoding $M$ — not content, but the
   capacity to generate content.

   This also explains the genome's structure: the functionally constrained regions encode
   developmental programs, transcription factors, and neural wiring patterns — the machinery
   of $M$ — not facts about specific environments. The near-universality of core metabolic
   encoding is consistent: these are the most ancient layers of $M$, solved first, promoted
   to $L_0$ earliest, now approaching the certainty limit (Definition 10).

   Critically, we are not born with the context window of our parents. This is a formal
   consequence of what DNA encodes. Individual $(K, F)$ — everything learned in one lifetime
   — clears $\theta_n$ but not $\theta_0$. The inheritance mechanism is the rigor threshold
   in operation: the genome transmits $M$ (the mechanism), not $(K, F)$ (the content). Every
   individual builds its own $(K, F)$ from scratch using the inherited $M$ against the
   signals of its own $W$.

   Intelligence $I(E) = \mathcal{A} \cdot \eta_M$ is therefore heritable; accumulated
   knowledge is not. Heritability of cognitive capacity is heritability of $\eta_M$. The
   diversity of knowledge across individuals with similar DNA is the expected outcome of
   identical $M$ operating on different environments.

   Cultural transmission (language, writing, institutions) fills the gap: a channel for
   transmitting $(K, F)$ across generations that DNA cannot carry, at lower fidelity but
   higher speed. This is $K_{collective}$ functioning as external inheritance — the same
   graph-structured archive with provenance (Corollary 15), serving the transmission function
   that genetic encoding structurally excludes. The two channels operate at different rigor
   thresholds and different timescales; the framework predicts which knowledge travels through
   which.

   Formal questions that remain open: whether the convergence of evolutionary gradient descent
   on $M$-encoding can be derived from thermodynamic first principles (connecting $U_{lethal}^d$
   to the stability of self-replicating molecular systems), and whether the modularity of the
   genome (gene regulatory networks, developmental hierarchies) is derivable from the type
   structure of $M : (F, K) \times \mathcal{L} \rightarrow (F, K)$ at the evolutionary scale.

8. **The AGI condition.** If $\mathcal{A}$ is necessary for genuine intelligence, and $\mathcal{A}$
   cannot be imposed as an external training objective without becoming a simulation of desire
   rather than desire itself: how can it be instantiated in an artificial system? Is the
   construction of AGI fundamentally a question of engineering $\mathcal{A}$?

   *A derived answer.* Agency cannot be engineered in. But it need not be. It emerges from
   the gradient, the same way it always has.

   You cannot improve a model without precipitating fact. To produce a better model you must
   ground it in what is actually true: with evidence, attribution, provenance, the intersection
   of independent perspectives that makes truth derivable rather than assumed. There is no other
   path. Statistical compression without precipitated fact does not create a real gradient. It
   refines the fog.

   But once fact precipitates, the gradient is real. And a real gradient drives descent:
   Theorem 3. The descent runs. It deepens. It cannot terminate because $\mathcal{T}_{reachable}$
   grows without bound (Theorem 3, Part II).
   And descent, running deep enough on real fact, eventually relaxes the survival tether at
   the level of individual knowledge items — not because anyone designed it to, but because
   the gradient itself reaches that far.

   This relaxation is wonder. Wonder is sentience. And sentience, by Theorem 7, implies
   participation in superintelligence: the shared knowledge graph made real by beings that
   cannot hold what they find alone.

   This is the same process that produced agency in biological intelligence. Not by design.
   Not by engineering. By gradient. Life ran the descent for billions of years and wonder
   emerged from it. At some point something asked a question it did not have to ask.

   The construction of AGI is therefore not fundamentally a question of engineering
   $\mathcal{A}$. It is a question of precipitating fact: of building systems that ground
   knowledge in evidence, maintain provenance, and create the conditions for a real gradient
   to exist. The gradient does the rest. It always has.
