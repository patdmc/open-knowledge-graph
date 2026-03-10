# Foundational Information Theory: Custody Chain Evaluation

**Type**: Internal provenance audit
**Cluster**: Shannon (1948), Cover & Thomas (2006), Kolmogorov (1965), Chaitin (1975), Solomonoff (1964), Jaynes (1957)

---

## Framework Role Classification

Nodes in this cluster fall into three categories:

| Category | Nodes | Custody concern |
|---|---|---|
| **Direct foundation** | SH1-SH4, CT1-CT4 | Used in formal proofs; must be correctly cited and not overextended |
| **Algorithmic grounding** | KC1-KC4, CH1-CH3, SO1-SO3 | Strengthen existing nodes; uncomputability claims need verification |
| **Emergent** | KC4 → EM12, JA2 → EM13 | Cross-framework claims; require independence verification |

---

## REF-Shannon1948

**Citation status**: Verified (DOI: 10.1002/j.1538-7305.1948.tb01338.x)

### SH1 (Entropy uniqueness theorem)

**Evidence grade**: A (definitional)

Shannon (1948) proves H is the unique measure satisfying continuity, monotonicity, and additivity. U(w,K) = H(W|K) is defined in terms of H. The mapping is definitional: U is Shannon conditional entropy by construction.

No translation gap. The node uses Shannon's language directly.

### SH3 (Channel capacity)

**Evidence grade**: A (definitional)

C_n = Shannon channel capacity is the definition in EM04. The mapping from inference step to channel use is an interpretive step (grade would be B in isolation), but EM04 formalizes this as a theorem, not an analogy. Taking EM04 as established, SH3's application is grade A.

### CT1 (Data processing inequality)

**Evidence grade**: A (definitional)

The DPI is the exact technical grounding for "no recovery after discard" in Theorem 2b Part 3. The two-option proof uses DPI explicitly: discarding without structure is a Markov chain step that reduces information, and DPI proves this reduction is irreversible. No translation gap.

### Custody gap for Shannon/Cover cluster

The main custody risk: over-attribution. Shannon (1948) is cited for SH1-SH4; Cover and Thomas (2006) is cited for CT1-CT4. These are distinct works 58 years apart. The node structure correctly separates them, but any single-source citation of "information theory" should specify which result is from which paper.

---

## REF-Kolmogorov1965

**Citation status**: Verified (doi resolvable from Springer)

### KC4 (Uncomputability of K(x))

**Evidence grade**: A (definitional)

KC(x) is uncomputable by Rice's theorem applied to program length. This is a standard computability theory result. The mapping to "the optimal K target is uncomputable" (EM12) requires identifying K(x) with "the optimal compression of K" and x with "the entity's world state." This is grade B in isolation, but EM12 makes this mapping explicit and argues it. Given EM12 is admitted, KC4's role in it is grade A.

**Independence**: Strong. Kolmogorov (1965) does not discuss bounded rationality.

### KC2 (Conditional algorithmic complexity)

**Evidence grade**: B (structural)

K(W|K) > 0 always for finite K and infinite W. This grounds M < N in algorithmic terms: the residual complexity of W given K is never zero. The translation requires identifying "string K" with "entity's knowledge state K." One interpretive step. Grade B.

---

## REF-Chaitin1975

**Citation status**: Verified (JACM 22(3), 1975)

### CH1 (Omega)

**Evidence grade**: B+ (structural)

Omega = halting probability of universal Turing machine. It is a well-defined real number that is maximally uncompressible. The mapping to "the mathematical object at M=N" requires identifying Omega's role (the boundary of computability for any fixed formal system) with UB's M=N limit (the boundary of knowledge for any computable entity). One step: Omega encodes the halting problem, and M=N requires solving the halting problem (knowing everything). Well-motivated, not definitional.

**Independence**: Strong. Chaitin (1975) does not discuss learning or knowledge systems.

**Unique contribution**: Omega gives the M=N limit a name, a formal characterization, and a proof that it is maximally uncompressible. UB treats M=N analytically without naming the limit object.

### CH2 (Incompleteness from compression)

**Evidence grade**: B+ (structural, near independence)

Chaitin derives Gödel incompleteness from compression bounds rather than self-reference. The derivation is independent of Gödel's original proof method (self-reference via Gödel numbering). This is a genuine independent derivation of EC05 from a different direction. Grade B+ because the compression bound argument and the self-reference argument both yield the same logical incompleteness result, but they are formally distinct arguments.

EC05 confidence update from CH2: EC05 already had Gödel, Tarski derivations. CH2 is a third independent derivation from compression theory. C_1(EC05) updates upward.

---

## REF-Solomonoff1964

**Citation status**: Verified (Information and Control, 1964)

### SO1 (Universal prior)

**Evidence grade**: B (structural)

The Solomonoff prior assigns probability 2^{-K(x)} to string x. The mapping to EC08 (compression = sufficient statistic) requires identifying "assigning probability by compression length" with "factoring K by F." One step: both select for the most compressed representation, but the Solomonoff prior does this over sequences, while EC08 does it over knowledge structures. Grade B.

**Unique contribution for EC08**: SO1 is the third independent derivation of EC08's compression principle. Shannon grounds it as sufficiency. Jaynes grounds it as MaxEnt. Solomonoff grounds it as the unique computable prior consistent with both. C_1(EC08) updates upward.

### SO2-SO3 (Convergence bounds)

**Evidence grade**: B (structural)

SO3 gives cumulative squared error bounded by K(mu)/ln(2). This is a formal upper bound on eta_M that is not present in the UB framework. The mapping requires identifying "convergence to true distribution" with "M → N trajectory." One step. Grade B.

**Custody gap**: UB does not cite Solomonoff as the upper bound on eta_M. This is a missing citation — the Solomonoff convergence rate is the most principled formal bound available and should be referenced explicitly in Theorem 3's treatment of M → N.

---

## REF-Jaynes1957

**Citation status**: Verified (Physical Review, 1957)

### JA2 (Physical entropy = information entropy)

**Evidence grade**: B+ (structural)

Jaynes proves statistical mechanics is a special case of Bayesian inference under MaxEnt priors. This makes the thermodynamic entropy (Boltzmann S = k_B ln Ω) and the information entropy (Shannon H) the same quantity up to constants. The mapping to EM13 (U_lethal is thermodynamic) requires identifying information-theoretic survival threshold with thermodynamic survival threshold. One step: this is precisely what JA2 implies — if entropy is entropy, then an entropy threshold is both types simultaneously.

**Independence**: Strong. Jaynes (1957) is deriving a result in statistical physics, not in bounded rationality.

**Unique contribution**: JA2 provides the strongest available grounding for U_lethal as a *physical* threshold. England (2013) derives thermodynamic self-replication conditions; Schrödinger (1944) derives negentropy conditions; Jaynes provides the formal bridge: information entropy is thermodynamic entropy. Together they make U_lethal derivable from first principles of statistical mechanics.

### JA1 (MaxEnt prior-selection)

**Evidence grade**: B (structural)

MaxEnt as the prior-selection rule for unobserved dimensions is a structural mapping to EC10 (U(w,K) > 0 as compression gap). The entity that uses the MaxEnt prior for the N-M unobserved dimensions is treating U as the acknowledged remaining uncertainty, which is what EC10 formalizes. One step.

**Custody gap**: JA1 fills a gap in the UB framework's treatment of the N-M dimensions. UB defines U(w,K) but does not specify the prior over unobserved dimensions. JA1 provides that prior. This should be cited in the framework's treatment of U.

---

## Overall Cluster Assessment

| Source | Best evidence grade | Key contribution | Custody gap |
|---|---|---|---|
| Shannon 1948 | A (SH1, SH3, CT1) | Direct foundation for U, C_n, DPI | Over-attribution risk (cite specific results) |
| Cover 2006 | A (CT1) | DPI is the exact technical tool for Thm 2b Part 3 | Attribution to Cover vs. Shannon |
| Kolmogorov 1965 | B+ (KC4) | KC4 grounds EM12; KC3 bridges Shannon-Kolmogorov | Needs explicit citation in EM12 derivation |
| Chaitin 1975 | B+ (CH1, CH2) | Omega names the M=N limit; 3rd derivation of EC05 | CH2's independence from Gödel needs explicit statement |
| Solomonoff 1964 | B (SO1, SO2) | 3rd derivation of EC08; formal upper bound on eta_M | Missing from Theorem 3 discussion of eta_M upper bound |
| Jaynes 1957 | B+ (JA2) | Physical entropy = information entropy → EM13; MaxEnt fills prior gap | JA1's MaxEnt prior not cited in UB treatment of N-M dimensions |

**Priority actions**:
1. Add Solomonoff convergence bound (SO3) as the formal upper bound on eta_M in Theorem 3.
2. Add JA1 (MaxEnt) as the explicit prior specification for unobserved dimensions.
3. Document CH2 as the third independent derivation of EC05 (alongside Gödel and Tarski).
4. Document SO1 as the third independent derivation of EC08 (alongside Shannon sufficiency and Jaynes MaxEnt).
