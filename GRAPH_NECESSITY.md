# Graph Structure Is Necessary for Information Preservation Under Bounded Context

**Patrick D. McCarthy**

---

## Abstract

Any system that (i) accumulates propositions about a world it cannot fully observe, (ii) operates under a bounded active context — a hard limit on the number of propositions simultaneously available for inference — and (iii) must preserve information as the proposition count grows beyond that bound, necessarily maintains a directed graph with typed edges over its propositions. We prove this claim from two independent arguments. The *information-preservation argument* shows that the only space-creating operation that does not lose information is factoring — extracting shared structure into named nodes with typed edges — and that factoring *is* graph construction. The *retention-dynamics argument* shows that any representation supporting sub-linear dependency queries, contextual retrieval, and selective removal with cascade necessarily encodes directed adjacency, which is a semantic graph regardless of substrate. We show that continuous-weight mechanisms (e.g., transformer self-attention) implement graph structure rather than constitute an alternative to it. The result is substrate-independent: it applies to biological neural networks, artificial neural networks, institutional knowledge systems, or any other medium that stores propositions under bounded context.

---

## 1. Introduction

Systems that accumulate knowledge face a fundamental tension. The knowledge base grows, but the capacity to reason over it at any single moment does not grow proportionally. Biological working memory holds roughly 7 ± 2 items [Miller 1956]. Transformer context windows, though large, degrade in inference quality as they fill with content irrelevant to the current problem [Liu et al. 2024]. Institutional decision-makers cannot hold an entire organization's knowledge in a single meeting. In each case, there is a hard or effective bound on *active context*: the number of propositions simultaneously available for inference.

This paper asks a narrow question: **what structure must a knowledge representation have if information is not to be lost as the number of stored propositions exceeds the active context bound?**

The answer is a directed graph with typed edges. We prove this from two independent arguments, each sufficient on its own:

1. **Information preservation** (Theorem 1). When a bounded system must make space for new propositions, the only operation that creates space without losing information is *factoring*: extracting shared structure into a named node with typed edges. Factoring is graph construction. Discarding without adjacency information is blind; lossy compression without adjacency is blind discarding with extra steps.

2. **Retention dynamics** (Theorem 2). Any representation that supports sub-linear dependency queries — answering "what depends on this proposition?" in less than O(n) time — necessarily stores directed adjacency. Selective removal with cascade (invalidating a proposition and propagating to its dependents) requires typed edges to scope correctly. These are the operations any system performs when maintaining knowledge under continuous evidence arrival.

The result is substrate-independent. It does not prescribe an implementation (adjacency lists, edge tables, attention masks, synaptic connections) — it proves that the *information content* of any adequate representation is a directed graph, regardless of how that graph is physically realized.

### Related work

Graph-structured knowledge representations are widely used in practice: knowledge graphs [Hogan et al. 2021], Bayesian networks [Pearl 1988], and semantic networks have a long history. The question of whether graph structure is *necessary* — as opposed to convenient — has received less formal attention.

Dupoux, LeCun, and Malik [2026] propose a dual-system architecture (observation + action) with meta-control routing, motivated by cognitive science. Their framework is architectural and prescriptive: it argues that autonomous learning *should* integrate observation, action, and meta-control. The present work is complementary: rather than proposing what intelligent systems should contain, we derive what they *must* contain under bounded context.

The information bottleneck principle [Tishby et al. 2000] establishes that optimal representations compress input while preserving information relevant to output. Our information-preservation argument (Theorem 1) addresses a different question: not what to compress, but what structure the representation must have to compress without information loss.

Simon's bounded rationality [Simon 1955] establishes that cognitive agents satisfice rather than optimize. Our bounded context constraint (C_n) formalizes a specific consequence: when the bound is hard, structure is not optional but necessary.

---

## 2. Formal Framework

We introduce the minimal definitions required for the two main results.

**Definition 1 (World State Space).** Let W be a measurable space of possible world states. At time t, the true state w_t ∈ W is not directly observable by any entity.

**Definition 2 (Proposition and Knowledge Base).** A *proposition* p is a claim about W with an associated likelihood function ℓ_p : W → [0,1]. A *knowledge base* K is a collection of propositions. The posterior over world states given K is:

> P(w | K) ∝ P₀(w) · ∏_{p ∈ K} ℓ_p(w)

where P₀ is a prior. The *uncertainty* of K about w is the surprisal U(w, K) = −log P(w | K) [Shannon 1948, Cover & Thomas 2006].

**Definition 3 (Bounded Active Context).** Let C_n denote the *active context capacity*: the maximum number of propositions simultaneously available for inference. Let ctx(t) ⊆ K be the set of propositions loaded into active context at time t, with |ctx(t)| ≤ C_n.

The *inference quality* for problem f is the relevance density:

> ρ(f) = |K^[f]| / |ctx(t)|

where K^[f] ⊆ K is the set of propositions relevant to f. ρ = 1 when active context holds exactly the relevant propositions; ρ → 0 when context is dominated by irrelevant content.

**Definition 4 (Growth and Curation Regimes).** K has a maximum capacity K_max determined by substrate constraints. When |K| < K_max, K is in the *growth regime*: propositions accumulate freely. When |K| = K_max, K is in the *curation regime*: adding a new proposition requires making space.

**Definition 5 (Contextual Retrieval).** Given problem f, *contextual retrieval* is the operation that loads K^[f] into active context. The retrieval cost is the number of propositions examined to identify K^[f].

---

## 3. Graph Necessity from Information Preservation

**Theorem 1 (Graph Necessity from Information Preservation).** Let K be a knowledge base with bounded active context C_n (Definition 3) in the curation regime (Definition 4). Any representation of K that preserves information as |K| grows beyond C_n necessarily maintains a directed graph with typed edges over propositions.

*Proof.* In the curation regime, adding any new proposition p_new requires making space. There are exactly three ways to make space:

**Option A: Discard an existing proposition.** Without explicit relational structure, the entity has no mechanism to determine which existing propositions are redundant. Two propositions p₁, p₂ that share a latent factor appear as independent entries; there is no stored evidence that they share structure. The choice of which to discard is therefore made without information about relational dependencies. Discarding p_old loses the information it encodes, and potentially loses the information that p_old has any relation to other propositions — a second-order loss. This is information loss.

**Option B: Factor — extract shared structure, reducing |K| without discarding.** If p₁ and p₂ share a latent factor f*, extracting f* as a single named node with typed edges to p₁ and p₂ reduces |K| by replacing redundant structure with a single shared reference. The information in p₁ and p₂ is preserved; the information about their shared grounding in f* is now explicitly stored as an edge. Space is created without loss. Factoring also resolves the update problem: when new evidence revises f*, the single node is updated once, and the revision propagates to all propositions that reference it via edges — O(1) instead of O(|K|).

Factoring is graph construction. Naming f*, creating a node, and storing typed edges from p₁ and p₂ to it is precisely the operation that produces a directed graph.

**Option C: Lossy compression — replace propositions with a summary that occupies less space.** This is a genuine third option, but it reduces to Option A or Option B. Compressing p₁ and p₂ into a summary p_s that discards detail is a form of discarding: the lost detail may ground other propositions or actions. If p₁ has dependents — propositions that reference it — then replacing p₁ with p_s silently invalidates those dependents unless the entity knows what depends on the lost detail. But knowing what depends on p₁ *is* knowing p₁'s adjacency — its edges. Without stored adjacency, compression is blind discarding: the entity cannot evaluate what breaks. With stored adjacency, the entity can scope the damage, notify dependents, and compress safely — but storing adjacency is maintaining a graph. Compression that preserves correctness requires the same relational structure as factoring. Compression that ignores relational structure is Option A with extra steps.

Any representation that makes space without factoring or correctly scoped compression loses information; any representation that factors or correctly scopes compression is constructing a graph.

**Consequence for alternative representations.** A flat store cannot factor: shared structure has no representation, so at C_n it must discard. A tree loses cross-branch propositions when one branch is pruned. A vector store holds shared factors implicitly in embedding geometry but cannot factor without naming — and naming is graph construction. Every representation that preserves information as |K| grows beyond C_n must factor. Every factoring operation creates nodes with typed edges. The graph is what factoring is. ∎

---

## 4. Graph Necessity from Retention Dynamics

**Theorem 2 (Graph Necessity from Retention Dynamics).** Let K be a knowledge base supporting continuous knowledge maintenance under evidence arrival. The maintenance operations are:

1. **Addition with provenance**: insert a proposition p together with its evidential grounds — which existing propositions p depends on, and by what relation.
2. **Contextual retrieval**: given problem f, retrieve exactly K^[f].
3. **Dependency query**: given proposition p, determine which propositions are grounded through p.
4. **Selective removal with cascade**: when p is invalidated, remove p and re-evaluate every proposition whose grounding depends on p, transitively, until K stabilizes.

Any representation achieving sub-linear cost on all four operations simultaneously necessarily encodes directed adjacency between propositions — and any structure encoding directed adjacency over propositions with typed relations *is* a semantic graph, regardless of implementation.

*Proof.*

**Part 1: Dependency queries are necessary and continuous.** Knowledge maintenance operates continuously: every new piece of evidence potentially shifts the status of existing propositions. When evidence revises or invalidates a proposition p, the system must evaluate every proposition whose grounding passes through p. This is not a batch operation that can be deferred — acting on a proposition whose ground has been invalidated risks incorrect inference. The dependency query "what is grounded through p?" is therefore invoked at the frequency of evidence arrival.

**Part 2: Lower bound without adjacency.** Consider a representation R that stores n propositions without explicit directed relationships between them (a flat store, hash table, embedding space, or any structure that does not record which propositions depend on which). To answer the dependency query for p, R must examine each of the other n − 1 propositions to determine whether its grounding involves p. Without stored adjacency, there is no mechanism to restrict this examination: any proposition might depend on p, and the only way to determine this is to check.

The cost of a single dependency query is therefore Ω(n) for any representation without stored adjacency. Since dependency queries are invoked at evidence-arrival frequency (Part 1), the amortized cost of maintenance is Ω(n) per evidence event — growing linearly with |K|.

**Part 3: Cascade amplifies the cost.** Selective removal is not a single deletion. When p is removed, each proposition q grounded through p must be re-evaluated: does q have alternative grounds, or should it also be removed? If q also fails, its dependents must be re-evaluated in turn. This is a transitive closure on the dependency relation.

With explicit adjacency, cascade follows edges: cost is O(|D_p|) where |D_p| is the transitive dependency set. Without adjacency, each level of cascade requires an Ω(n) scan. Total cost: Ω(n · L) where L is the cascade depth, versus O(|D_p|) in the graph where |D_p| ≪ n for typical propositions with bounded connectivity.

**Part 4: Any sub-linear solution encodes adjacency.** Suppose R achieves o(n) cost for dependency queries. Then R must store, for each proposition, information about which other propositions depend on it — otherwise the Ω(n) lower bound of Part 2 applies. This stored information is a set of directed relationships: for each proposition p, a record of the propositions that reference p as part of their grounding.

This is precisely a directed adjacency structure. The nodes are propositions. The edges are directed dependency relations. Any representation that stores this information — regardless of whether it is implemented as adjacency lists, edge tables, pointers, or any other mechanism — *is* a directed graph over propositions.

**Part 5: Typed edges are necessary for correct cascade.** A bare adjacency structure (untyped edges) is insufficient. When p is invalidated, cascade must distinguish between propositions *inferentially grounded* through p (which require re-evaluation) and propositions merely *associated* with p (which do not). An untyped edge conflates these: every neighbor must be treated as potentially dependent, degrading cascade to the full neighborhood rather than the true dependency set.

Typed edges — carrying the relation type ("grounds," "evidences," "causes," "supports") — enable cascade to follow only dependency-bearing edges, keeping cost proportional to |D_p| rather than |adj(p)|. Since |adj(p)| can be much larger than |D_p|, typed edges are necessary for efficient cascade.

A directed graph with typed edges over propositions is a semantic graph. ∎

---

## 5. Continuous-Weight Mechanisms Are Graph Implementations

A natural objection: modern attention mechanisms achieve contextual retrieval without explicit graph construction. Does this constitute a non-graph alternative?

**Theorem 3 (Attention Mechanisms Implement Graph Structure).** Any continuous-weight retrieval mechanism that achieves contextual retrieval over n propositions either (a) computes directed adjacency dynamically at O(n²) per query, or (b) stores adjacency explicitly at O(d) per query where d is the local degree. In either case, the mechanism encodes a directed graph.

*Proof.* The attention matrix A_ij = softmax(Q_i K_j⊤ / √d_k) is a dense directed adjacency matrix: entry A_ij is the weight of the edge from proposition i to proposition j. Full self-attention computes the complete n × n adjacency at every query: O(n²) per retrieval. This is the cost of *not* pre-computing the graph — the system rediscovers dependency structure from scratch at each step.

A pre-built graph with explicit adjacency pays O(d) per retrieval, amortizing adjacency computation over all future queries.

Sparse attention mechanisms — learned sparsity patterns, local windows, routing functions — reduce the O(n²) cost by learning which edges to keep: they are learned graph pruning. The KV cache materializes a subgraph for reuse across queries. Flash attention optimizes memory access patterns without changing the structure being computed. Each optimization is a step toward the pre-built explicit graph: reducing the cost of dynamic adjacency by progressively storing more structure.

A fixed sparsity pattern over propositions with typed relationships is a graph. The structural requirement is substrate-independent: whether adjacency is stored as edge lists, learned sparse masks, pointer structures, or synaptic weights, the information content is the same — which propositions are connected to which, by what relation. ∎

**Corollary (Scaling Context Windows Is Self-Defeating).** Growing C_n directly — the prevailing approach to scaling inference capacity — is self-defeating. For any specific problem f, ρ decreases as C_n fills with content not relevant to f (Definition 3). More context does not produce better inference; it produces higher variance that inference must manage. The efficient path is not to grow the context window but to grow the encoded knowledge accessible via graph traversal — filling the graph, not the context window [Liu et al. 2024].

---

## 6. Discussion

The two theorems establish graph necessity from independent premises. Theorem 1 requires only the curation regime (bounded capacity, new propositions arriving). Theorem 2 requires only continuous knowledge maintenance (evidence arrives, dependencies must be traced). Any system subject to both constraints — which includes every physical system that accumulates knowledge — is doubly committed to graph structure.

**What the result does not claim.** The theorems prove that the *information content* of the representation must be a directed graph. They do not prescribe a specific graph formalism (property graph, RDF, Bayesian network) or implementation (adjacency lists, sparse matrices, attention masks). The graph is the abstract structure; the substrate is a free parameter.

The result also does not claim that all knowledge must be stored in an *explicit*, human-readable graph database. A neural network whose learned weights encode directed dependencies between features is maintaining a graph in the sense of Theorem 2, even if no symbolic node labels exist. The claim is structural, not implementational.

**Implications for AI architecture.** Theorem 3 reframes the relationship between graph-based and neural approaches: they are not competing paradigms but different implementations of the same structural necessity. Systems that compute adjacency dynamically (full attention) pay O(n²); systems that store it explicitly (graphs, sparse attention, KV caches) pay O(d). The trend in architecture design — from dense attention toward sparse, structured, and cached retrieval — is convergence toward explicit graph structure, driven by the same cost pressures the theorems identify.

The result suggests that hybrid architectures combining explicit knowledge graphs with neural retrieval are not an engineering convenience but a response to a structural constraint: as |K| grows, systems must encode adjacency to maintain inference quality under bounded context.

**Implications for scaling.** The corollary challenges the prevailing approach of growing context windows as the primary path to more capable inference. Under bounded context, larger windows with undifferentiated content *reduce* inference quality for any specific problem. The alternative — encoding knowledge into a graph and retrieving the relevant subgraph per problem — maintains ρ = 1 regardless of total knowledge size. This is the difference between scaling the window and scaling the knowledge.

**Biological evidence.** Biological neural networks maintain sparse, structured connectivity rather than dense all-to-all connections. Working memory is bounded (C_n ≈ 7 ± 2) while long-term knowledge grows without apparent limit [Miller 1956]. Retrieval is associative and context-driven — consistent with graph traversal from a query node rather than exhaustive scan. The theorems suggest this architecture is not an accident of evolutionary history but a necessary consequence of bounded context under information-preservation pressure.

---

## 7. Conclusion

We have shown that graph structure over propositions is not a design choice but a necessary consequence of two independently sufficient constraints: information preservation under bounded capacity, and sub-linear knowledge maintenance under continuous evidence arrival. Continuous-weight mechanisms such as transformer self-attention implement graph structure dynamically rather than constituting alternatives to it. The result is substrate-independent: any system that accumulates propositions under bounded active context and must preserve information necessarily maintains a directed graph with typed edges, whether that graph is realized as explicit edges, learned sparse attention, synaptic connectivity, or any other medium.

---

## References

- Cover, T. M. and Thomas, J. A. *Elements of Information Theory*. Wiley-Interscience, 2nd edition, 2006.
- Dupoux, E., LeCun, Y., and Malik, J. Why AI systems don't learn and what to do about it: Lessons on autonomous learning from cognitive science. *arXiv preprint arXiv:2603.15381*, 2026.
- Hogan, A. et al. Knowledge graphs. *ACM Computing Surveys*, 54(4):1–37, 2021.
- Liu, N. F. et al. Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics*, 12:157–173, 2024.
- Miller, G. A. The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review*, 63(2):81–97, 1956.
- Pearl, J. *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann, 1988.
- Shannon, C. E. A mathematical theory of communication. *Bell System Technical Journal*, 27(3):379–423, 1948.
- Simon, H. A. A behavioral model of rational choice. *The Quarterly Journal of Economics*, 69(1):99–118, 1955.
- Tishby, N., Pereira, F. C., and Bialek, W. The information bottleneck method. In *Proceedings of the 37th Annual Allerton Conference on Communication, Control, and Computing*, pages 368–377, 2000.
