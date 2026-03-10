# Calvin Lin: Custody Chain Evaluation

**Type**: Internal provenance audit
**Date**: 2025
**Purpose**: Evaluate the evidence quality, grounding strength, and confidence chain for each of Lin's results as nodes in the open knowledge graph. This is not a judgment of Lin's papers -- it is an evaluation of how securely each mapping is grounded within this framework.

---

## What "Custody Chain" Means Here

Each knowledge graph node carries a provenance triple: (attribution, evidence, derivation). The custody chain evaluation asks:

1. **Attribution**: Is it clear where the claim originates?
2. **Evidence grade**: How strong is the evidence for the mapping between Lin's result and the UB node it maps to? (definitional correspondence / structural / analogical / speculative)
3. **Derivation path**: Can the mapping be reconstructed step by step? Is each step independently verifiable?
4. **Independence**: Is this result genuinely independent of the framework it maps to, or could it be argued that the mapping is post-hoc?
5. **Falsifiability**: What would falsify the mapping?

Grades used: **A** (definitional -- the mapping holds under translation with no interpretive gap), **B** (structural -- the mapping holds under clear formal translation with one step of interpretation), **C** (analogical -- the mapping is structurally suggestive but requires non-trivial interpretation), **D** (speculative -- the mapping is interesting but cannot be made formal without additional work).

---

## CL4: Flow-Sensitive Pointer Analysis

**Paper**: Lin, C. and Hardekopf, B. (2011). Flow-Sensitive Pointer Analysis for Millions of Lines of Code. *CGO 2011*. [Best Paper + 2023 CGO Test of Time Award]
**Also**: Lin, C. and Hardekopf, B. (2009). Semi-Sparse Flow-Sensitive Pointer Analysis. *POPL 2009*.

**Maps to**: NV01 (graph necessity), EC08 (compression factoring), EC06 (belief-as-action)

### Attribution

Clear and unambiguous. The result is Lin and Hardekopf's, the correspondence identification is McCarthy (2025). No prior work states this mapping.

### Evidence Grade: A (definitional)

The mapping from flow-sensitive pointer analysis to the K/F graph is definitional under the following translation:

```
K := points-to information (what each variable may point to)
F := control flow graph (the set of program points = reachable program states)
K/F graph := flow-sensitive representation (K indexed by F)
K without F coupling := flow-insensitive representation
```

Under this translation, every theorem Lin and Hardekopf prove about flow-sensitive analysis is a theorem about the K/F graph. The precision loss theorem (flow-insensitive analysis produces false positives that are not recoverable) is, under translation, a proof that K not indexed by F loses information. NV01's two-option proof reaches the same conclusion from information theory.

There is no interpretive gap. The translation is not metaphorical. "Program point" means the same thing as "action state" under the obvious interpretation: a program point is the state the program occupies while taking some action (executing an instruction).

### Derivation Path

Step 1: Lin proves flow-sensitive analysis is strictly more precise than flow-insensitive.
Step 2: The precision loss is formally characterized as false positives arising from collapsing per-point information.
Step 3: Under translation, "collapsing per-point information" = "not indexing K by F."
Step 4: NV01 proves that not indexing K by F causes information loss at bounded context.
Step 5: The precision loss measured in Step 2 is the information loss proved in Step 4.

Each step is independently verifiable against the published papers and the formal statement of NV01.

### Independence

Strong. Lin's papers do not reference information theory, bounded rationality, or the UB framework. The motivation is entirely within compiler theory: false positives in static analysis are a practical problem, and flow-sensitivity is the solution. The result that K/F indexing is necessary is arrived at empirically (measured precision improvement) and formally (proof that flow-insensitive analysis cannot recover the lost information), with no knowledge of the information-theoretic derivation.

### The Test of Time Award

The 2023 CGO Test of Time Award for the 2011 paper adds a second custody layer beyond the original peer review. It means that a committee of compiler researchers, 12 years later, judged that the result had lasting impact and had been validated by the broader community. For the purposes of this framework, this functions as a replication and community validation: the K/F graph representation has been tested against alternatives in industrial compilers for over a decade and has not been superseded.

### Falsifiability

The mapping would be falsified if: (1) a counterexample were found where K not indexed by F preserves information at bounded context, or (2) flow-insensitive analysis were shown to be equally precise as flow-sensitive under some reformulation that does not amount to K/F indexing by another name.

### Summary

| Criterion | Assessment |
|---|---|
| Attribution | Clear, no dispute |
| Evidence grade | A (definitional) |
| Derivation path | Fully explicit, 5 steps |
| Independence | Strong -- no cross-domain reference in either direction |
| Test of Time validation | Yes (2023 CGO award) |
| Falsifiability | Clearly stated |
| C_1 contribution to NV01 | Raises from 0.82 to ~0.93 (fifth independent derivation) |

**Custody chain verdict**: Highest confidence of any Lin result. The mapping is definitional, the independence is strong, the validation is multi-stage, and the C_1 update is material.

---

## CL1: Belady's OPT and Cache Replacement

**Papers**: Lin, C. and Jain, A. (2016). Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement. *ISCA 2016*.
Lin, C. and Jain, A. (2022). Effective Mimicry of Belady's MIN Policy. *HPCA 2022*.
Lin, C. and Jain, A. (2019). *Cache Replacement Policies*. Morgan and Claypool.

**Maps to**: EC01 (context bound), KC4 (uncomputability of optimal compression), EC08 (compression factoring)

**Emergent intersection**: EM26, with REF-VaswaniEtAl2017 (context window = C_n)

### Attribution

The Belady optimality result is Belady (1966). The claim that OPT is uncomputable in practice is standard in the cache theory literature. Lin's contribution is the computable approximation (Hawkeye/PC-OPT). The EM26 intersection (cache eviction = LLM context management) is McCarthy (2025).

### Evidence Grade for CL1 → KC4: B (structural)

Belady's OPT requires knowing future access patterns. The UB framework's uncomputability result (KC4, from Kolmogorov) states that the optimal K target cannot be computed for any computable entity. These are structurally the same claim: the optimal eviction/compression decision requires an oracle that computable processes cannot instantiate.

The translation requires one step of interpretation: "future access pattern" = "the temporal dimension of W beyond the current context." This is clear but not definitional in the same sense as CL4.

### Evidence Grade for EM26 (CL1 + Vaswani → LLM context = cache): B (structural)

The formal correspondence (cache = context window, eviction = token removal, cache miss = reloading) is structurally clean. The translation requires accepting that attention context and hardware cache serve the same formal role (bounded-capacity active state), which is not a trivial step but is well-grounded in the definitions.

The evidence for this correspondence is presently theoretical: neither the LLM literature nor the cache theory literature has stated it, and there is no empirical test yet showing that Hawkeye-style heuristics applied to LLM context management improve performance. The mapping is well-grounded but untested in the target domain.

### Derivation Path for EM26

Step 1: Vaswani et al. define the context window as a hard token-count limit (C_n in tokens).
Step 2: Long-context LLM operation requires deciding which tokens to retain when context exceeds the window.
Step 3: This is formally a cache replacement problem: bounded-capacity store, items arriving over time, eviction policy required.
Step 4: Belady's OPT is the optimal eviction policy for any cache replacement problem.
Step 5: Belady's OPT for the LLM context window = always remove the token least likely to be needed in future output.
Step 6: This is uncomputable for the same reason OPT is uncomputable: requires knowing future outputs.
Step 7: Lin's PC-history heuristic (use past access history to approximate future need) maps to: use past query/generation history to approximate future token need.

Steps 1-6 follow from definitions. Step 7 requires accepting the analogical structure of "program counter" and "query/generation context" as both being finite histories that predict future access. This is a strong analogy but not definitional.

### Independence

Strong for the original Belady result (1966, well before UB). Lin's contribution (computable approximations) is independent of UB. The EM26 intersection node is identified by McCarthy but the two source papers are fully independent of each other and of UB.

### Falsifiability

The structural mapping (EM26) would be falsified if a formal distinction were shown between LLM context management and cache replacement that makes the problems structurally different -- for example, if the dependency structure of tokens (tokens attend to each other) creates constraints that have no analog in cache theory. This is a legitimate open question. Attention is not a simple read operation; it is a function over all retained tokens. Whether this changes the optimal eviction problem is not yet resolved.

### Summary

| Criterion | Assessment |
|---|---|
| Attribution | Belady (1966) for OPT; Lin for approximations; McCarthy for EM26 intersection |
| Evidence grade | B (structural) for both CL1→KC4 and EM26 |
| Derivation path | Explicit for both; Step 7 in EM26 is analogical not definitional |
| Independence | Strong |
| Empirical validation | None in target domain (LLM context management) yet |
| Falsifiability | Stated; attention dependency structure is open question |
| C_1 contribution | Confirms KC4 (no update to already-high prior); EM26 is new node, not confidence update |

**Custody chain verdict**: Well-grounded structural mapping. The main custody gap is the absence of empirical validation of the engineering transfer (Hawkeye → LLM context). The mapping is sound and the derivation is explicit, but it remains theoretical until tested.

---

## CL5: Neural Prefetching

**Paper**: Lin, C. et al. (2020). A Hierarchical Neural Model of Data Prefetching. *ASPLOS 2020*.

**Maps to**: EC02 (improvement rate), EC01 (context bound), NV03 (agency as gate)

**Candidate emergent**: CL5 + EM26 -- neural prefetching and RAG as architecturally identical

### Evidence Grade: C (analogical)

The structural claim is: neural prefetching (predict future cache accesses using LSTM over PC-address history) and retrieval-augmented generation (predict future knowledge needs using retrieval model over past queries) are architecturally identical. Both are learned approximations to Belady's OPT at different scales.

The mapping requires accepting three analogies simultaneously: (1) hardware cache ≈ LLM context, (2) PC-address history ≈ query/generation history, (3) LSTM sequence model ≈ retrieval model. Each analogy individually is grade B. The composition of three B-grade analogies gives C for the composite.

### Custody Gap

The specific architecture of Lin's hierarchical prefetcher (separate LSTM models per cache level) maps to the RAG hierarchy (in-context, retrieved, parametric knowledge) with a clean 3-level structure. However, RAG retrieves from an external corpus while Lin's prefetcher predicts from access history. These are different operations. Prefetching is predictive; retrieval is reactive. The analogy captures the optimization target (Belady's OPT) but may mischaracterize the mechanism.

### Falsifiability

The architectural identity claim would be falsified if a formal distinction between prediction from history (prefetching) and retrieval from a corpus (RAG) were shown to require different optimization objectives. This is a live question.

### Summary

| Criterion | Assessment |
|---|---|
| Evidence grade | C (analogical) |
| Main custody gap | Predictive vs. reactive mechanism distinction |
| Status | Open PE -- not admitted to graph in EC evaluation pass |

**Custody chain verdict**: Interesting candidate, not yet admittable. Requires more careful analysis of whether the mechanism distinction matters for the optimization equivalence claim.

---

## CL2: Cache Coherence and K_collective Maintenance

**Source**: Lin, C. and Snyder, L. (2008). *Principles of Parallel Programming*. Addison-Wesley. (Also standard distributed systems literature.)

**Maps to**: NV04 (intersection acceleration), EM10 (K_collective edge maintenance)

**Candidate emergent**: CL2 + EM10 -- MESI protocol as K_collective provenance state machine

### Evidence Grade: B-minus (structural with gaps)

The MESI state mapping (Modified = authoritative local write, Exclusive = unshared, Shared = replicated, Invalid = stale/unverified) to K_collective provenance states is structurally clean. The 4-state machine captures exactly the provenance distinctions the UB framework needs for K_collective maintenance.

However, the UB framework's K_collective maintenance problem (EM10) is stated at a higher level of abstraction than MESI. The gap: MESI operates on cache lines (fixed-size memory blocks) with symmetric update rules. K_collective operates on propositions with asymmetric provenance structure (attribution matters, derivation chains matter). It is not clear that MESI's symmetric coherence protocol is the right model for asymmetric provenance chains.

### Custody Gap

The main gap: coherence failure in MESI (stale read) is symmetric -- any processor can write, any processor can have a stale cache. K_collective provenance failure is asymmetric -- attribution matters (who claimed what). A stale cache with unknown writer is different from a claim with unknown attribution. MESI does not track attribution; it tracks currency.

This is a meaningful distinction. The mapping is more precisely: MESI captures the currency dimension of K_collective provenance (is this knowledge current?) but not the attribution dimension (who is responsible for this knowledge?). The full K_collective provenance problem requires both dimensions.

### Falsifiability

The correspondence would be falsified if a formal argument showed that coherence protocols are fundamentally about currency and K_collective provenance is fundamentally about attribution, and that these cannot be reduced to a common abstract problem.

### Summary

| Criterion | Assessment |
|---|---|
| Evidence grade | B-minus (structural with gap) |
| Main custody gap | MESI tracks currency; K_collective needs attribution+currency |
| Status | Open PE -- flagged for future EC pass |

**Custody chain verdict**: The mapping captures a real subset of the K_collective maintenance problem (currency dimension), but the attribution dimension is not covered by MESI. If the scope is narrowed to "MESI captures the currency dimension of K_collective provenance," the grade improves to B. Admittable as a partial correspondence, not a full equivalence.

---

## CL3: Amdahl's Law and NV04 Acceleration

**Source**: Standard parallel programming literature; stated in Lin, C. and Snyder, L. (2008).

**Maps to**: NV04 (intersection acceleration), EC02 (improvement rate)

**Assessment**: Collapse result, not emergent.

Amdahl's law states that maximum speedup from parallelism is bounded by the serial fraction. Within the UB framework, this is a corollary of NV04: the acceleration from K_collective intersection is bounded by the serial cost of provenance verification (which cannot be parallelized without losing attribution). Amdahl provides the formal bound; NV04 identifies the mechanism that creates the serial fraction (provenance verification).

This is a useful footnote, not an emergent intersection. Both frameworks state a version of the same constraint. The correspondence does not add information beyond what either framework already says.

**Evidence grade**: B (structural) as a confirmation of NV04's bound, not as an independent derivation.

---

## Overall Custody Chain Assessment for Lin's Corpus

| Result | Evidence Grade | C_1 Contribution | Status |
|---|---|---|---|
| CL4 (flow-sensitive) | A (definitional) | NV01: 0.82 → 0.93 | Admitted EM27, highest confidence |
| CL1 (Belady's OPT) | B (structural) | None (confirms KC4) | Admitted via EM26 |
| EM26 (CL1 + Vaswani) | B (structural) | New node (LLM/cache) | Admitted, untested in target domain |
| CL5 (neural prefetch) | C (analogical) | None yet | Open PE, mechanism gap |
| CL2 (cache coherence) | B-minus | None yet | Open PE, attribution gap |
| CL3 (Amdahl) | B (structural) | None (collapse) | Useful footnote |

**Cross-corpus pattern**: Lin's work is strongest where the formal structure of his problems most directly matches the formal structure of the UB framework (CL4 is definitional for this reason: pointer analysis is formally about what is known at what state, which is exactly the K/F structure). It is weaker where hardware constraints (cache line granularity, MESI symmetry) diverge from the framework's propositional and attributive structure.

**The most secure result in Lin's corpus for this framework**: CL4. The custody chain is clean, the independence is strong, the validation is multi-stage (POPL + CGO + Test of Time Award), and the C_1 update to NV01 is material (0.11 increase, larger than any other single derivation in the graph except the original information-theoretic proof).

**The most interesting open question**: Whether the attention dependency structure in Transformers (tokens attending to each other) creates a version of the cache replacement problem that is formally distinct from standard cache, and whether Lin's algorithms need modification before the engineering transfer is valid. This question determines whether EM26 should carry a caveat or whether it is clean.
