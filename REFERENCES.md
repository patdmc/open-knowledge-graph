# References

Working bibliography for THEORY.md and THEORY_FEP.md.
Citation keys are in `[AuthorYear]` format for inline placement.

---

## Tier 1 — Must cite before submission

### [Friston2010]
Friston, K. (2010). The free-energy principle: a unified brain theory?
*Nature Reviews Neuroscience*, 11(2), 127–138.
https://doi.org/10.1038/nrn2787

**Supports:** Abstract, Section 4 (Theorem 2a note), THEORY_FEP.md throughout.
**Stance:** Extension and partial correction. FEP derives that intelligent systems
minimize free energy (equivalent to U-reduction) but does not derive the minimization
imperative from selection pressure, does not constrain the generative model via K/F
inseparability, assumes top-down precision allocation (which Theorem 2a shows is
self-undermining under C_n constraints), and has no account of collective intelligence.

### [Friston2021]
Friston, K., Da Costa, L., Hafner, D., Hesp, C., & Parr, T. (2021).
Sophisticated inference. *Neural Computation*, 33(3), 713–763.
https://doi.org/10.1162/neco_a_01351

**Supports:** Section 11 Related Work (FEP extension 4 — collective intelligence);
THEORY_FEP.md Section 6 (Extension 4, multi-agent FEP).
**Stance:** Engaged directly. Sophisticated inference extends active inference to
recursive, multi-step planning and multi-agent coordination. Extension 4 shows what
this framework lacks: a formal fidelity threshold ($\lambda_{min}$), provenance as
the mechanism maintaining it, and truth as the limit of collective action-validation.

### [Friston2017]
Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017).
Active inference: A process theory. *Neural Computation*, 29(1), 1–49.
https://doi.org/10.1162/NECO_a_00912

**Supports:** THEORY_FEP.md Section on precision hierarchy; Theorem 2a note on
top-down allocation.
**Stance:** Correction. Active inference formalizes the precision-weighted prediction
error as the attention allocation mechanism. This is the specific target of the
allocation tax argument and the escalation critique.

### [Shannon1948]
Shannon, C. E. (1948). A mathematical theory of communication.
*Bell System Technical Journal*, 27(3), 379–423.
https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

**Supports:** Definition 5 (U = H(w|K)), Definition 3b (G = I(f;w|K)),
all information-theoretic grounding throughout.
**Stance:** Foundation. H(w|K) is Shannon conditional entropy; I(f;w|K) is
Shannon mutual information. The paper's uncertainty measure is this quantity
exactly.

### [Tarski1936]
Tarski, A. (1936). The concept of truth in formalized languages.
In J. H. Woodger (Trans.), *Logic, Semantics, Metamathematics* (pp. 152–278).
Oxford University Press (1956).

**Supports:** Theorem 10 (Epistemic Ceiling); Definition 16a (Truth as fixed point).
**Stance:** Formal grounding. Tarski's undefinability theorem establishes that truth
in a formal system cannot be defined within that system. Theorem 10 is the
information-theoretic operational form of this result: the confidence regress is
non-closeable from within, which is the epistemic consequence of Tarski's theorem
applied to the framework's truth definition.

### [Peirce1877]
Peirce, C. S. (1877). The fixation of belief. *Popular Science Monthly*, 12, 1–15.

### [Peirce1878]
Peirce, C. S. (1878). How to make our ideas clear. *Popular Science Monthly*, 12,
286–302.

**Supports:** Definition 16 (Truth as limit of collective action-validation).
**Stance:** Correspondence. Peirce defines truth as "the opinion which is fated to
be ultimately agreed to by all who investigate." Definition 16 formalizes this as an
information-theoretic limit: the fraction of independent entities whose action-validation
of p converges as n → ∞. The grounding in action-validation (not mere agreement) and
the explicit convergence condition are the formal additions this paper makes.

---

## Tier 2 — Should cite

### [Schmidhuber2010]
Schmidhuber, J. (2010). Formal theory of creativity, fun, and intrinsic motivation
(1990–2010). *IEEE Transactions on Autonomous Mental Development*, 2(3), 230–247.
https://doi.org/10.1109/TAMD.2010.2056368

**Supports:** Definition 6 (Agency A as intrinsic drive), Definition 11a (η_M as
learning efficiency), Section 1 (intelligence as driven capacity to improve).
**Stance:** Correspondence and extension. Schmidhuber formalizes curiosity as
compression progress — the rate at which a learning algorithm improves its world
model. This is structurally η_M. The difference: (1) Schmidhuber's drive is
about compression rate, not uncertainty reduction; (2) this paper separates drive
(A) from mechanism (η_M) as formally independent factors; (3) A is grounded in
survival pressure (Theorem 3) rather than defined as a design objective.

### [Gibson1979]
Gibson, J. J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin.

**Supports:** Section 3 (K/F Inseparability), Corollary 1 (Indexed Knowledge),
Definition 3 (Informed Action Space).
**Stance:** Correspondence and formalization. Gibson's affordances are the action
possibilities a world offers relative to an organism's capacities — K indexed by F
under a different name. Theorem 1 provides what Gibson's account lacks: a formal
selection criterion (the asymmetric retention criterion from survival pressure) for
which affordances are retained and which are pruned.

### [Baars1988]
Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

**Supports:** Definition 8 (Encoding Hierarchy), Theorem 2 (Escalation), Corollary 3
(Context Window as Frontier), Theorem 2a (Top-Down Allocation Degrades Inference Quality).
**Stance:** Correspondence and formalization. Global Workspace Theory (GWT) holds that
local processors handle routine computation; the global workspace broadcasts only when
local processors fail. This is the escalation architecture in cognitive science language.
The paper's contribution: (1) derives the escalation criterion from U-reduction and
survival pressure rather than assuming it architecturally; (2) makes the hierarchy
continuous (Definition 8) rather than binary (local vs. global); (3) shows via Theorem 2a
why top-down allocation from the global workspace is inefficient.

### [Dehaene2011]
Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to
conscious processing. *Neuron*, 70(2), 200–227.
https://doi.org/10.1016/j.neuron.2011.03.018

**Supports:** Same as [Baars1988]; use alongside or as alternative.
**Stance:** Same as [Baars1988]. Dehaene & Changeux provide the neural implementation
evidence for GWT, making this the more empirically grounded citation for the escalation
architecture claim.

### [Kahneman2011]
Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

**Supports:** Definition 8 (Encoding Hierarchy), Section 1 (continuity claim).
**Stance:** Discrete precursor that the continuity principle corrects. System 1
(automatic) and System 2 (deliberate) are the popular formulation of the encoding
hierarchy — but as a binary. The paper's contribution is making this continuous
(Definition 8, L(c,r) parameterization) and deriving the boundary criterion from
U-reduction cost rather than asserting it.

### [Chalmers1995]
Chalmers, D. J. (1995). Facing up to the problem of consciousness.
*Journal of Consciousness Studies*, 2(3), 200–219.

**Supports:** Definition 14 (Sentience), note on usage.
**Stance:** Explicit disclaimer. The paper uses "sentience" as a functional term
(degree to which A drives knowing beyond survival justification) and explicitly makes
no claim about phenomenal consciousness. Chalmers' hard problem is the reason for
that disclaimer. Cite in the Definition 14 note: the functional condition defined
here is not offered as a solution to the hard problem; whether it is sufficient,
necessary, or orthogonal to phenomenal experience is left open.

### [Tishby2011]
Tishby, N., & Polani, D. (2011). Information theory of decisions and actions.
In A. Cutsuridis, A. Hussain, & J. G. Taylor (Eds.), *Perception-Action Cycle:
Models, Architectures, and Hardware* (pp. 601–636). Springer.
https://doi.org/10.1007/978-1-4419-1452-1_19

**Supports:** Section 3 (K/F Inseparability), Definition 3 (Informed Action Space).
**Stance:** Correspondence. The information bottleneck principle applied to action
selection: an agent retains only information useful for action. Theorem 1's asymmetric
retention criterion is a formal extension grounded in survival pressure, which
Tishby & Polani's framework treats as a design parameter rather than a derived result.

### [Thrun1998]
Thrun, S., & Pratt, L. (1998). Learning to learn: Introduction and overview.
In S. Thrun & L. Pratt (Eds.), *Learning to Learn* (pp. 3–17). Springer.
https://doi.org/10.1007/978-1-4615-5529-2_1

**Supports:** Definition 11 (Higher-Order Function Space M), Definition 11a (η_M),
Definition 7 (I(E) = A·η_M).
**Stance:** Correspondence. Meta-learning is the CS field that formalizes the rate
at which a system improves its own learning given experience — η_M. The paper connects
meta-learning rate to survival pressure (Theorem 4 rigor thresholds) rather than
treating it as a design parameter, and separates drive (A) from mechanism (η_M)
as formally distinct quantities.

### [Schrodinger1944]
Schrödinger, E. (1944). *What is Life? The Physical Aspect of the Living Cell*.
Cambridge University Press.

**Supports:** Section 5 thermodynamic commentary, Open Question 1 (origin of A),
Open Question 9 (DNA as encoding of M).
**Stance:** Foundation. Schrödinger's "negative entropy" argument — that living
systems maintain local order against thermodynamic pressure — is the physical grounding
for the claim that U_lethal connects to thermodynamic thresholds and that A is
thermodynamically instantiated.

---

## Tier 3 — Cite for completeness

### [Pearl1988]
Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible
Inference*. Morgan Kaufmann.

**Supports:** Definition 5 (H(w|K) — how K as a directed graph induces P(w|K)).
**Stance:** Foundation. Bayesian networks are the standard formalism for conditioning on
graph-structured knowledge; Definition 5's interpretation of K as a Bayesian network
where nodes are propositions and edges are conditional dependencies uses Pearl's framework
directly.

### [Cover2006]
Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.).
Wiley-Interscience.

**Supports:** Definition 5 (H(w|K)), Definition 3b (I(f;w|K)), Lemma 1 proof
(monotonicity of conditional entropy).
**Stance:** Standard reference. Cite for the monotonicity of conditional entropy
(Lemma 1 proof), the data processing inequality (Definition 3b G≥0 claim), and
as the textbook reference for all information-theoretic identities used.

### [Prigogine1984]
Prigogine, I., & Stengers, I. (1984). *Order Out of Chaos: Man's New Dialogue
with Nature*. Bantam Books.

**Supports:** Open Question 1 (thermodynamic instantiation of A), Open Question 9
(DNA and dissipative structures).
**Stance:** Adjacent framework. Dissipative structures — self-organizing systems
that maintain local order by dissipating entropy — are the physical precursor of A
in primitive form. Already cited in text; add full reference.

### [England2013]
England, J. L. (2013). Statistical physics of self-replication.
*Journal of Chemical Physics*, 139(12), 121923.
https://doi.org/10.1063/1.4818538

**Supports:** Open Question 1 (thermodynamic instantiation of A), Open Question 9
(DNA as encoding of M, convergence from thermodynamics).
**Stance:** Adjacent framework. England's dissipative adaptation shows that
self-replicating structures are thermodynamically favored in driven systems.
Already cited in text; add full reference.

### [MaynardSmith1995]
Maynard Smith, J., & Szathmáry, E. (1995). *The Major Transitions in Evolution*.
W. H. Freeman.

**Supports:** Open Question 9 (DNA as encoding of M, evolutionary encoding
hierarchy), Theorem 3a (multi-timescale gradient).
**Stance:** Correspondence. Major transitions in evolution (e.g., the origin of
the genetic code, eukaryotes, multicellularity) correspond to shifts in what level
of the encoding hierarchy holds the primary gradient signal. Each transition is a
change in where M operates and at what rigor threshold.

### [Hinton1987]
Hinton, G. E., & Nowlan, S. J. (1987). How learning can guide evolution.
*Complex Systems*, 1(3), 495–502.

**Supports:** Theorem 3a (multi-timescale gradient, η_evo vs η_ind), Open Question 9.
**Stance:** Correspondence. The Baldwin effect — learned adaptations can guide
genetic assimilation — is the empirical phenomenon corresponding to the multi-timescale
gradient. Knowledge that first appears at L_n (individual learning) can eventually
meet θ_0 and crystallize at L_0 (genetic encoding) over evolutionary timescales.

### [Sutton2018]
Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
(2nd ed.). MIT Press.

**Supports:** Theorem 3 (gradient descent on U), Theorem 3a (multi-timescale gradient).
**Stance:** Correspondence. Reinforcement learning is gradient descent on a
reward/uncertainty signal — the computational formalization of the same process
Theorem 3 derives from survival pressure. RL's learning rate parameter η is the
computational analog of η_ind in Theorem 3a.

### [Woolley2010]
Woolley, A. W., Chabris, C. F., Pentland, A., Hashmi, N., & Malone, T. W. (2010).
Evidence for a collective intelligence factor in the performance of human groups.
*Science*, 330(6004), 686–688.
https://doi.org/10.1126/science.1193147

**Supports:** Section 9 (Collective Knowledge), Theorem 6 (Collective Gradient
Dominance), Corollary 15 (Self-Assembly).
**Stance:** Empirical support. Collective intelligence is measurable and distinct
from individual intelligence — consistent with Definition 15 (Superintelligence as
a property of K_collective, not any individual E_i).

### [Liu2024]
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., &
Liang, P. (2024). Lost in the middle: How language models use long contexts.
*Transactions of the Association for Computational Linguistics*, 12, 157–173.
https://doi.org/10.1162/tacl_a_00638

**Supports:** Corollary 4b (Scaling Argument Against Context Window Growth),
Discussion section (civilization-scale knowledge system).
**Stance:** Empirical support. Language models with longer contexts perform worse
at retrieving relevant information from non-salient positions. This is the empirical
signature of ρ degradation under Theorem 2a: more context does not improve inference;
it introduces noise that degrades retrieval of the relevant subgraph K^{[f]}.

### [vanderAalst2016]
van der Aalst, W. M. P. (2016). *Process Mining: Data Science in Action* (2nd ed.).
Springer.
https://doi.org/10.1007/978-3-662-49851-4

**Supports:** Section 11 Related Work (Theorem 9, Observational Bootstrapping).
**Stance:** Adjacent method. Process mining discovers workflow models from event logs —
the closest existing CS method to equivalence class skill precipitation. Distinction:
produces flat workflow descriptions without provenance, confidence measures, or connection
to a transposability threshold.

### [AbbeelNg2004]
Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning.
*Proceedings of the 21st International Conference on Machine Learning (ICML)*, 1.
https://doi.org/10.1145/1015330.1015430

**Supports:** Section 11 Related Work (Theorem 9, Observational Bootstrapping).
**Stance:** Adjacent method. IRL infers reward functions from demonstrated behavior;
the output is a policy to imitate. Distinction: infers what to optimize for, not an
informed action with grounded evidence, conditions, and provenance.

### [AamodtPlaza1994]
Aamodt, A., & Plaza, E. (1994). Case-based reasoning: Foundational issues, methodological
variations, and system approaches. *AI Communications*, 7(1), 39–59.

**Supports:** Section 11 Related Work (Theorem 9, Observational Bootstrapping).
**Stance:** Adjacent method. CBR retrieves similar past cases to inform new decisions.
Distinction: retrieves the similar case; Theorem 9 induces the general informed action
from the equivalence class, with formal confidence derived from $k$ independent instances.

### [Muggleton1991]
Muggleton, S. (1991). Inductive logic programming.
*New Generation Computing*, 8(4), 295–318.
https://doi.org/10.1007/BF03037089

**Supports:** Section 11 Related Work (Theorem 9, Observational Bootstrapping).
**Stance:** Adjacent method. ILP learns general rules from positive and negative examples.
Distinction: no provenance, no uncertainty measure, no rigor threshold determining when
the induced rule is reliable enough to promote to lower encoding levels.

### [FinnEtAl2017]
Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation
of deep networks. *Proceedings of the 34th International Conference on Machine Learning
(ICML)*, 70, 1126–1135.

**Supports:** Section 11 Related Work (Theorem 9, Observational Bootstrapping); Definition
11a (η_M at collective scale).
**Stance:** Correspondence and extension. MAML learns an initialization of M that adapts
quickly across a task distribution — structurally η_M at collective scale. Distinction:
optimizes adaptation speed; this framework additionally derives the rigor threshold for
promotion and the provenance requirement for transposability.

---

## Citation placement map

The following lists where each key needs to be inserted in THEORY.md.
Work through this list when adding inline markers.

| Location | Citation(s) | Note |
|---|---|---|
| Abstract para 3 (FEP critique) | [Friston2010], [Friston2017] | Name FEP explicitly |
| Section 1, opening | [Friston2010] | "departure from" FEP |
| Definition 5 (H(w\|K)) | [Shannon1948], [Cover2006], [Pearl1988] | Grounding U formally; Bayesian network interpretation |
| Definition 3b (I(f;w\|K)) | [Shannon1948] | Mutual information |
| Lemma 1 proof (monotonicity) | [Cover2006] | "conditional entropy is monotonically non-increasing" |
| Definition 3b G≥0 (data processing) | [Cover2006] | Data processing inequality |
| Section 3 (K/F Inseparability) | [Gibson1979], [Tishby2011] | Affordances / bottleneck |
| Definition 6 (Agency A) | [Schmidhuber2010] | Formal curiosity drive |
| Definition 7 (I(E) = A·η_M) | [Schmidhuber2010], [Thrun1998] | Compression progress / meta-learning |
| Definition 8 / Theorem 2 (Escalation) | [Baars1988], [Dehaene2011], [Kahneman2011] | GWT / System 1-2 |
| Theorem 2a note (FEP self-undermining) | [Friston2010], [Friston2017] | Precision allocation |
| Corollary 4b (context window scaling) | [Liu2024] | Empirical support |
| Definition 11 / 11a (M, η_M) | [Thrun1998] | Meta-learning |
| Definition 14 note (sentience disclaimer) | [Chalmers1995] | Hard problem |
| Definition 16 (Truth as limit) | [Peirce1877], [Peirce1878] | Pragmatist truth |
| Theorem 6 (Collective Gradient) | [Woolley2010] | Empirical collective intelligence |
| Section 5 thermodynamic commentary | [Schrodinger1944] | Negative entropy |
| Open Question 1 | [Prigogine1984], [England2013], [Schrodinger1944] | Dissipative structures |
| Open Question 7 (boundary of M) | [Hinton1987] | Baldwin effect / M acting on M |
| Open Question 9 (DNA as encoding of M) | [Schrodinger1944], [MaynardSmith1995], [Hinton1987], [England2013] | Evolution / encoding |
| Discussion (civilization-scale) | [Liu2024], [Woolley2010] | Empirical grounding |
| Section 11 Related Work (FEP) | All Tier 1 and 2 | Full engagement |
| Theorem 9 / Section 11 (CS-side) | [vanderAalst2016], [AbbeelNg2004], [AamodtPlaza1994], [Muggleton1991], [FinnEtAl2017] | Observational bootstrapping |
