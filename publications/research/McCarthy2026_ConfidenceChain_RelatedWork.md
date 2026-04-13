---
title: "Related Work Survey for 'The Confidence Chain'"
author: "Patrick McCarthy"
date: "2026"
companion: "CONFIDENCE_CHAIN_PAPER.md"
purpose: |
  Structured survey of existing systems, frameworks, and research related to
  scientific knowledge provenance, citation graph analysis, and cross-framework
  evaluation. Organized for direct incorporation into the Related Work section.
---

# Related Work: The Confidence Chain

---

## Overview

The Confidence Chain paper sits at the intersection of six research threads: citation analysis, knowledge provenance, philosophy of science (cross-framework comparison), information theory applied to knowledge, automated knowledge extraction, and peer review reform. Each thread has produced significant infrastructure. None has produced the specific combination this paper proposes: explicit epistemic provenance with computable confidence scores, a decomposition theorem for cross-framework evaluation, and a machine-readable peer review standard.

The tables below document what exists, what it provides, and what it lacks.

---

## 1. Citation Analysis and Scientometrics

| System / Framework | Key Citation | What It Does | Relevance to Confidence Chain | What It Lacks |
|---|---|---|---|---|
| **Science Citation Index (SCI)** | Garfield, E. (1955). "Citation Indexes for Science." *Science*, 122(3159), 108--111. | First systematic index of who-cites-whom in the scientific literature. Commercialized as Web of Science (ISI, now Clarivate). | Established that citation is a tractable, computable relation. The entire field of scientometrics descends from this. The Confidence Chain's provenance triples extend citation from a binary relation (A cites B) to a typed, evidence-bearing relation. | Treats citation as a homogeneous binary edge. No distinction between "cites for method," "cites for claim," "cites to refute." No confidence propagation. No evidence typing. |
| **h-index** | Hirsch, J. E. (2005). "An index to quantify an individual's scientific research output." *Proceedings of the National Academy of Sciences*, 102(46), 16569--16572. | Single scalar summarizing a researcher's impact: h papers with at least h citations each. | Demonstrates demand for quantified epistemic reputation. The h-index is a proxy for what the Confidence Chain measures directly: how much confidence should propagate from an author's work. | Measures volume/impact, not correctness. A retracted paper with 500 citations has the same h-index contribution as a replicated one. Conflates attribution with confidence -- the exact problem Section 1 of the paper identifies. |
| **PageRank for citation networks** | Chen, P., Xie, H., Maslov, S., & Redner, S. (2007). "Finding scientific gems with Google's PageRank algorithm." *Journal of Informetrics*, 1(1), 8--15. Also: Walker, D., Xie, H., Yan, K.-K., & Maslov, S. (2007). "Ranking scientific publications using a model of network traffic." *JSTAT*, P06010. | Applies the PageRank eigenvector centrality measure to citation graphs. Ranks papers by the "importance" of their citers, not just citation count. | Structural ancestor of confidence propagation. PageRank propagates a scalar (importance) through the citation graph, exactly as the Confidence Chain propagates C_1 scores. The difference is what is propagated: PageRank propagates attention; the Confidence Chain propagates epistemic confidence with evidence typing. | No evidence typing. No distinction between claim-level and paper-level citation. The propagated quantity (eigenvector centrality) is a popularity measure, not an epistemic one. A highly-cited wrong paper accumulates PageRank. |
| **CiteScore / SNIP / SJR** | Moed, H. F. (2010). "Measuring contextual citation impact of scientific journals." *Journal of Informetrics*, 4(3), 265--277. Scopus metrics (Elsevier). | Journal-level citation metrics that attempt to normalize for field differences (SNIP) or weight by citing journal prestige (SJR). | Recognize that raw citation count is field-dependent and attempt correction. The Confidence Chain's equivalency classes (ECs) solve the same problem structurally: two claims from different fields are comparable when they share an EC, not when their citation counts are normalized. | Still journal-level, not claim-level. Normalization is statistical, not semantic. Cannot distinguish whether a specific claim within a paper is well-supported or riding on the journal's reputation. |
| **Epistemic citation typing** | Teufel, S., Siddharthan, A., & Tidhar, D. (2006). "Automatic classification of citation function." *Proceedings of EMNLP*, 103--110. Also: Jurgens, D., Kumar, S., Hoover, R., McFarland, D., & Jurafsky, D. (2018). "Measuring the evolution of a scientific field through citation frames." *TACL*, 6, 391--406. | Classifies citations by function: background, method, comparison, refutation, etc. Jurgens et al. identify "citation frames" that capture how a citing paper positions itself relative to the cited work. | Direct precursor to the Confidence Chain's evidence typing. These systems recognize that not all citations are equal and attempt to classify the relationship. The Confidence Chain's provenance triple (attribution, evidence, derivation) is the formalization of what citation typing approximates empirically. | Classification is post-hoc and approximate (NLP on text). Not part of the authoring process. Not machine-readable at submission time. No confidence score attached. No propagation model. |

---

## 2. Knowledge Representation and Provenance Systems

| System / Framework | Key Citation | What It Does | Relevance to Confidence Chain | What It Lacks |
|---|---|---|---|---|
| **W3C PROV-O / PROV-DM** | Moreau, L. & Missier, P. (Eds.) (2013). "PROV-DM: The PROV Data Model." W3C Recommendation. Also: Lebo, T. et al. (2013). "PROV-O: The PROV Ontology." W3C Recommendation. | W3C standard for representing provenance as a directed acyclic graph of Entities, Activities, and Agents. PROV-O is the OWL ontology; PROV-DM is the data model. Became a W3C Recommendation in April 2013. | The closest existing standard to the Confidence Chain's provenance triples. PROV-DM's Entity/Activity/Agent triple maps structurally to the Confidence Chain's attribution/evidence/derivation triple. PROV-O provides the RDF vocabulary for encoding provenance in machine-readable form. | General-purpose provenance, not scientific-knowledge-specific. No confidence scores. No equivalency classes. No decomposition theorem. No model for how confidence propagates through derivation chains. PROV records *what happened*; the Confidence Chain evaluates *how much to believe it*. |
| **Nanopublications** | Groth, P., Gibson, A., & Velterop, J. (2010). "The Anatomy of a Nanopublication." *Information Services & Use*, 30(1-2), 51--56. Mons, B. et al. (2011). "The value of data." *Nature Genetics*, 43, 281--283. | Smallest unit of publishable information: an RDF triple (assertion) plus provenance and publication info. Designed to make individual scientific claims citable and machine-readable. Nanopub servers exist (nanopub.net). | The most direct architectural predecessor. Nanopublications decompose papers into atomic claims with provenance -- exactly what the Confidence Chain requires as input. The Confidence Chain can be viewed as the evaluation layer that nanopublications lack: given a nanopublication, compute its C_1. | No confidence scoring. No cross-framework evaluation. No equivalency classes. No decomposition theorem. Nanopublications record provenance but do not evaluate it. Adoption has been limited to the Linked Data / semantic web community; mainstream scientific publishing has not adopted the format. |
| **OpenCitations** | Peroni, S. & Shotton, D. (2020). "OpenCitations, an infrastructure organization for open scholarship." *Quantitative Science Studies*, 1(1), 428--443. Uses the CiTO (Citation Typing Ontology): Shotton, D. (2010). "CiTO, the Citation Typing Ontology." *Journal of Biomedical Semantics*, 1(Suppl 1), S6. | Open, machine-readable corpus of citation data (>1.8 billion citation links as of 2024). Uses RDF and the CiTO ontology to type citations (cites, citesAsAuthority, citesAsDataSource, etc.). | Provides the open citation graph infrastructure that confidence propagation would operate on. CiTO's citation typing is a partial implementation of the Confidence Chain's evidence typing. OpenCitations + CiTO + confidence scores = a significant fraction of the Confidence Chain infrastructure. | CiTO types are descriptive, not epistemic. "citesAsAuthority" records a relationship but does not evaluate whether the authority is warranted. No confidence propagation. No equivalency classes. No decomposition theorem. The graph is structural, not evaluative. |
| **Semantic Scholar** | Ammar, W. et al. (2018). "Construction of the Literature Graph in Semantic Scholar." *Proceedings of NAACL-HLT*, 84--91. Fricke, S. (2018). "Semantic Scholar." *Journal of the Medical Library Association*, 106(1), 145--147. | AI-powered academic search engine (Allen Institute for AI). Indexes >200 million papers. Extracts structured metadata, citation graphs, and (via TLDR) paper summaries. Provides API access to the citation graph. | Large-scale citation graph with extracted metadata. The Confidence Chain's graph could be seeded from Semantic Scholar's citation data and enriched with provenance triples and confidence scores. SPECTER embeddings (see Section 5) provide vector representations that could initialize equivalency class detection. | No confidence scores. No provenance triples beyond standard citation. No cross-framework evaluation. The graph is a citation graph, not a knowledge graph -- it connects papers, not propositions. |
| **Microsoft Academic Graph (MAG)** | Sinha, A. et al. (2015). "An Overview of Microsoft Academic Service (MAS) and Applications." *Proceedings of WWW Companion*, 243--246. Discontinued 2021; succeeded by OpenAlex. | Large-scale academic knowledge graph with >260 million papers, fields of study, authors, institutions. Provided entity-level linking (paper-author-institution-concept). | Demonstrated that academic knowledge can be represented as a typed graph at scale. The fields-of-study hierarchy is a coarse precursor to equivalency classes. OpenAlex (its successor) continues this work. | Discontinued. Fields of study are taxonomic, not structural (no shared-premise detection). No claim-level granularity. No confidence propagation. |
| **OpenAlex** | Priem, J., Piwowar, H., & Orber, R. (2022). "OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts." arXiv:2205.01833. | Open replacement for MAG. Indexes >250 million works with structured metadata, citation links, and concept tagging. Fully open API and data dumps. | Current best open infrastructure for academic metadata at scale. A Confidence Chain implementation would likely build on OpenAlex's citation graph as the base layer, adding provenance triples and confidence scores on top. | Same structural limitations as MAG: paper-level, not claim-level. No evidence typing. No confidence propagation. Concept tagging is taxonomic, not based on shared premises. |
| **Connected Papers** | Eitan, I., Smolyansky, E., Harpaz, I., & Perets, S. (2021). Connected Papers (tool). connectedpapers.com. | Visual exploration tool that builds similarity graphs (not just citation graphs) of papers using co-citation and bibliographic coupling. | Demonstrates that the citation graph alone is insufficient -- structural similarity between papers matters. Co-citation proximity is a rough proxy for shared equivalency class membership. | Visualization tool, not an evaluation framework. No claim-level analysis. No confidence scores. No formal cross-framework comparison. |
| **ResearchGraph** | Aryani, A. & Wang, J. (2017). "Research Graph: Building a Distributed Graph of Scholarly Works, Datasets, and Researchers." *eResearch Australasia*. | Connects publications, datasets, researchers, and grants into a single graph. Emphasizes linking across identifier systems (DOI, ORCID, ROR). | Demonstrates the infrastructure need for connecting heterogeneous scholarly objects. The Confidence Chain's provenance triples are a specific type of scholarly object that ResearchGraph's linking infrastructure could support. | Infrastructure for linking, not for evaluation. No confidence propagation. No claim-level granularity. No cross-framework evaluation. |
| **RDF / OWL for scientific knowledge** | Berners-Lee, T., Hendler, J., & Lassila, O. (2001). "The Semantic Web." *Scientific American*, 284(5), 34--43. OWL: W3C Recommendation (2004, revised 2012). | Standard languages for encoding knowledge as typed graphs (RDF triples) with formal ontological reasoning (OWL). The Semantic Web vision: machine-readable, interlinked knowledge. | The Confidence Chain's knowledge graph schema (Section 3 of the paper) is implementable in RDF/OWL. Provenance triples are RDF triples. Equivalency classes are OWL classes. The Decomposition Theorem's classification (collapse / contradiction / novel) could be encoded as OWL axioms with inference rules. | General-purpose infrastructure, not scientific-evaluation-specific. The Semantic Web vision has been partially realized for data (Wikidata, DBpedia) but not for scientific claims. No confidence propagation built into RDF/OWL. No decomposition theorem. The gap between the vision and scientific practice remains large. |

---

## 3. Cross-Framework Comparison in Science (Philosophy of Science)

| Framework | Key Citation | What It Does | Relevance to Confidence Chain | What It Lacks |
|---|---|---|---|---|
| **Kuhn's paradigm incommensurability** | Kuhn, T. S. (1962). *The Structure of Scientific Revolutions*. University of Chicago Press. | Argues that scientific paradigms are incommensurable: practitioners of different paradigms literally cannot compare their theories because the terms mean different things. Theory choice is not fully rational. | The Confidence Chain's equivalency classes are the formal response to incommensurability. Two frameworks are commensurable to the extent that they share equivalency classes. The Decomposition Theorem operationalizes comparison: given shared ECs, derivations are classifiable. Kuhn says comparison is impossible; the Confidence Chain says comparison is possible exactly where premises are shared, and specifies the method. | Descriptive, not constructive. Kuhn identifies the problem (incommensurability) but provides no formal method for determining *where* two frameworks are commensurable and *where* they are not. No computable procedure. No decomposition theorem. |
| **Lakatos's research programmes** | Lakatos, I. (1978). *The Methodology of Scientific Research Programmes*. Cambridge University Press. | Distinguishes a "hard core" of unfalsifiable assumptions from a "protective belt" of auxiliary hypotheses. Research programmes are progressive (predicting novel facts) or degenerating (only accommodating known facts). | The hard core / protective belt distinction maps structurally to the Confidence Chain's equivalency classes (hard core = shared ECs) and framework-specific derivations (protective belt = NV and OV nodes). Progressive vs. degenerating maps to the intersection prediction: a programme that produces emergent propositions at framework intersections is progressive. | No formal specification of what constitutes the hard core. No computable method for identifying shared premises across programmes. No confidence scores. The progressive/degenerating distinction is qualitative, not quantifiable. |
| **Laudan's problem-solving model** | Laudan, L. (1977). *Progress and Its Problems*. University of California Press. | Evaluates theories by problem-solving effectiveness: how many empirical problems solved minus how many conceptual problems generated. Rejects Kuhn's incommensurability -- theories from different traditions can be compared by problem-solving power. | Agrees with the Confidence Chain that cross-framework comparison is possible and necessary. Laudan's "conceptual problems" (inconsistencies between a theory and other accepted theories) are a qualitative version of the Decomposition Theorem's contradiction detection. | No formal method. "Problem-solving effectiveness" is not computable. No provenance model. No confidence propagation. The comparison criterion is pragmatic, not structural. |
| **Bayesian confirmation theory** | Howson, C. & Urbach, P. (2006). *Scientific Reasoning: The Bayesian Approach*. 3rd ed. Open Court. Also: Earman, J. (1992). *Bayes or Bust?* MIT Press. | Formalizes confirmation as Bayesian updating: evidence E confirms hypothesis H when P(H\|E) > P(H). Provides a probability calculus for scientific inference. | The Confidence Chain's C_1 scores are computable confidence values that could, in principle, be interpreted as posterior probabilities. The confidence update formula in Section 6 of the paper is structurally Bayesian: multiple independent derivations update C_1 via a product formula analogous to independent likelihood combination. | Requires prior probabilities, which are notoriously difficult to assign for scientific hypotheses. Does not address cross-framework comparison (no equivalency classes). Does not address provenance -- Bayesian updating conditions on evidence, but the provenance of the evidence is not modeled. No decomposition theorem. |
| **Structural realism** | Worrall, J. (1989). "Structural Realism: The Best of Both Worlds?" *Dialectica*, 43(1-2), 99--124. Also: Ladyman, J. (1998). "What is Structural Realism?" *Studies in History and Philosophy of Science*, 29(3), 409--424. | Claims that what is preserved across theory change is structure (mathematical relations), not ontology (what the world is made of). Theories can be compared by their structural content. | The Confidence Chain's equivalency classes ARE structural content. Two frameworks share an EC when their formal structures entail the same relation, regardless of their ontological commitments. The Decomposition Theorem is a structural realist's tool: it compares derivations, not interpretations. | Philosophical position, not a method. No computable procedure for extracting "structure" from a theory. No confidence propagation. No automation. |
| **Theory choice criteria (Kuhn's five values)** | Kuhn, T. S. (1977). "Objectivity, Value Judgment, and Theory Choice." In *The Essential Tension*, 320--339. University of Chicago Press. | Lists five criteria for theory choice: accuracy, consistency, scope, simplicity, fruitfulness. Acknowledges these can conflict and require judgment. | The Confidence Chain operationalizes several of these: accuracy maps to C_1 scores; consistency maps to contradiction detection in the Decomposition Theorem; scope maps to equivalency class coverage; fruitfulness maps to emergent proposition production (Corollary 8a). | The criteria are qualitative and can conflict. No formal method for resolving conflicts. No provenance model. No machine-readable evaluation. |

---

## 4. Information Theory Applied to Knowledge

| Framework | Key Citation | What It Does | Relevance to Confidence Chain | What It Lacks |
|---|---|---|---|---|
| **Shannon's channel capacity** | Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379--423. | Establishes the mathematical foundations of information theory: entropy, mutual information, channel capacity, the noisy channel coding theorem. | Foundation for the Confidence Chain's information-theoretic framing. The paper's C_n (bounded context) is formally related to Shannon's channel capacity. The uncertainty U(w,K) = H(W\|K) is conditional Shannon entropy. Citation is an information channel; the Confidence Chain asks what its capacity and noise characteristics are. | Shannon's theory is about signal transmission, not epistemic evaluation. No model of confidence, provenance, or cross-framework comparison. The theory provides the mathematical vocabulary but not the scientific-knowledge-specific application. |
| **Kolmogorov complexity** | Kolmogorov, A. N. (1965). "Three Approaches to the Quantitative Definition of Information." *Problemy Peredachi Informatsii*, 1(1), 3--11. | Defines the information content of an individual object as the length of its shortest program. Algorithmic information theory. | Already integrated into the Confidence Chain (see Addendum, EM12). K(x) provides the formal bound on compression -- the target the M-to-N gradient approaches. The uncomputability of K(x) is the formal basis for EM12's "the target is uncomputable." | Not about scientific knowledge or provenance. No confidence model. No cross-framework comparison. Pure mathematics of information content. |
| **Information-theoretic scientometrics** | Leydesdorff, L. (2006). "Can Scientific Journals Be Classified in Terms of Aggregated Journal-Journal Citation Relations Using the Journal Citation Reports?" *JASIST*, 57(5), 601--613. Also: Leydesdorff, L. & Rafols, I. (2009). "A Global Map of Science Based on the ISI Subject Categories." *JASIST*, 60(2), 348--362. | Applies Shannon entropy and mutual information to citation networks to measure the information content of citation patterns, detect disciplinary structure, and map interdisciplinarity. | Direct application of information theory to the citation graph. Leydesdorff's mutual information between journal citation distributions is a coarse-grained version of what the Confidence Chain's equivalency classes measure: shared information content between frameworks. | Journal-level, not claim-level. No confidence propagation. No provenance triples. No decomposition theorem. The information-theoretic analysis is descriptive (measuring existing patterns), not prescriptive (proposing new standards). |
| **Knowledge flow in citation networks** | Shi, X., Leskovec, J., & McFarland, D. A. (2010). "Citing for High Impact." *Proceedings of JCDL*, 49--58. Also: Hummon, N. P. & Dereian, P. (1989). "Connectivity in a Citation Network: The Development of DNA Theory." *Social Networks*, 11(1), 39--63. | Models how ideas propagate through citation chains. Hummon & Dereian's "main path analysis" identifies the most significant chains. Shi et al. model strategic citation for impact. | Citation chain analysis is the empirical counterpart to confidence chain analysis. Main path analysis identifies the structural backbone of a research field -- the chain along which confidence propagates. The Confidence Chain asks: what is the epistemic quality of that chain, not just its structure? | No epistemic evaluation. Citation flow is measured in volume (how many citations), not quality (how much confidence should propagate). Strategic citation is modeled but not corrected for. No provenance triples. |

---

## 5. Automated Knowledge Extraction from Papers

| System | Key Citation | What It Does | Relevance to Confidence Chain | What It Lacks |
|---|---|---|---|---|
| **GROBID** | Lopez, P. (2009). "GROBID: Combining Automatic Bibliographic Data Recognition and Term Extraction for Scholarship Publications." *ECDL*, LNCS 5714, 473--474. | Machine learning system for extracting structured bibliographic data from PDF scholarly documents. Extracts headers, authors, affiliations, references, citations. | Infrastructure for automated provenance extraction. To build confidence chains at scale, structured data must be extracted from unstructured papers. GROBID provides the first layer: identifying what is cited and where. | Extracts bibliographic metadata, not claims. Does not identify which specific claim from a cited paper is being invoked. No proposition extraction. No confidence scoring. |
| **Science Parse / ScienceParse2** | Allen Institute for AI (2017). Science Parse. github.com/allenai/science-parse. | Parses PDFs of scientific papers to extract title, authors, abstract, body text, references, and citation contexts. Successor to earlier parsers; now largely superseded by GROBID and S2ORC. | Similar to GROBID: infrastructure for structured extraction. S2ORC (Semantic Scholar Open Research Corpus) builds on these parsers to provide full-text and citation contexts at scale. | Same limitations as GROBID: metadata and text extraction, not claim extraction. No proposition-level granularity. |
| **SPECTER / SPECTER2** | Cohan, A. et al. (2020). "SPECTER: Document-level Representation Learning using Citation-informed Transformers." *Proceedings of ACL*, 2270--2282. Singh, A. et al. (2023). "SciRepEval: A Multi-Format Benchmark for Scientific Document Representations." *TACL*. | Produces vector embeddings of scientific papers using a SciBERT backbone trained on citation links. Papers that cite each other are embedded nearby. SPECTER2 adds multi-task training for different downstream tasks. | SPECTER embeddings could be used to initialize equivalency class detection: papers with similar embeddings may share premises. The embedding space is a continuous approximation to the discrete equivalency class structure. | Embeddings capture topical similarity, not structural similarity of premises. Two papers can be embedded nearby because they cite the same literature, not because they share formal assumptions. No claim-level granularity. No confidence scores. No decomposition theorem. |
| **Scientific claim extraction** | Wadden, D. et al. (2020). "Fact or Fiction: Verifying Scientific Claims." *Proceedings of EMNLP*, 7534--7550. (SciFact dataset and VERISCI system.) Also: Wright, D. et al. (2022). "Generating Scientific Claims for Zero-Shot Scientific Fact Checking." *Proceedings of ACL*, 2726--2737. | Extracts and verifies individual scientific claims against evidence from paper abstracts. SciFact provides a dataset of claims with supporting/refuting evidence. | Closest existing work to automated proposition extraction for the Confidence Chain. SciFact's claim + evidence pairs are a simplified version of provenance triples. Claim verification is a simplified version of confidence scoring (binary: supported/refuted vs. continuous C_1). | Binary verification (supported/refuted), not continuous confidence. Claims are extracted from abstracts only, not full text. No cross-framework evaluation. No equivalency classes. No decomposition theorem. Scale is small (dataset of ~1,400 claims). |
| **SciClaim / scientific IE** | Jain, S. et al. (2017). "Extracting Scientific Figures, Tables, and Captions." *Proceedings of JCDL*. Also: Luan, Y. et al. (2018). "Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction." *Proceedings of EMNLP*, 3219--3232. (SciERC dataset.) | Extracts entities (methods, tasks, metrics, materials) and relations from scientific text. Builds structured knowledge graphs from unstructured papers. | SciERC's entity-relation extraction is a building block for automated provenance triple construction. The extracted relations (USED-FOR, COMPARE, PART-OF, etc.) are coarse-grained versions of the Confidence Chain's derivation types. | Relations are generic, not epistemic. No confidence scores. No equivalency classes. Extraction accuracy is imperfect (~70-80% F1). Scale is limited. |

---

## 6. Proposals for Reforming Peer Review

| Proposal / System | Key Citation | What It Does | Relevance to Confidence Chain | What It Lacks |
|---|---|---|---|---|
| **Open peer review** | Ross-Hellauer, T. (2017). "What is Open Peer Review? A Systematic Review." *F1000Research*, 6:588. | Umbrella term for review practices that open identities, reports, or participation. Includes signed reviews, published reviews, open participation. Platforms: F1000Research, PeerJ, eLife. | Addresses the transparency component of the confidence chain problem: if reviews are public, the evidence base for acceptance is visible. Open peer review makes the review's reasoning auditable. | Transparency without structure. Open reviews are natural language text, not machine-readable provenance. No confidence scores. No cross-framework evaluation requirement. No decomposition theorem. The review is public but not formally evaluable. |
| **Post-publication peer review** | Knoepfler, P. (2015). "Reviewing Post-Publication Peer Review." *Trends in Genetics*, 31(5), 221--223. Platforms: PubPeer, Publons (now part of Web of Science). | Reviews occur after publication. Anyone can comment. Errors, concerns, and failures to replicate are documented publicly. | Addresses the revalidation gap: confidence in a claim should update when new evidence arrives. Post-publication review provides a mechanism for this. PubPeer has been instrumental in identifying fraud, errors, and irreproducibility. | No formal confidence update mechanism. Comments are unstructured. No propagation model (if Paper A's claim is challenged, papers citing A for that claim are not automatically flagged). No requirement for authors to respond. No integration with the citation graph. |
| **Registered Reports** | Chambers, C. D. (2013). "Registered Reports: A New Publishing Initiative at Cortex." *Cortex*, 49(3), 609--610. Nosek, B. A. & Lakens, D. (2014). "Registered Reports: A Method to Increase the Credibility and Transparency of Published Results." *Social Psychology*, 45(3), 137--141. | Two-stage review: Stage 1 reviews the hypothesis and methodology *before* data collection. If accepted, the paper is published regardless of results. Adopted by >300 journals. | Addresses a specific failure mode in the confidence chain: publication bias. By decoupling acceptance from results, Registered Reports ensure that the evidence base is not systematically biased toward positive findings. The Confidence Chain's C_1 scores would be more accurate in a Registered Reports regime because the evidence base would be less distorted. | Addresses one failure mode (publication bias) but not the broader confidence chain problem. No provenance triples. No cross-framework evaluation. No confidence propagation model. No decomposition theorem. Applies only to hypothesis-testing research, not to theoretical or formal work. |
| **Structured peer review (review forms)** | Superchi, C. et al. (2019). "Tools Used to Assess the Quality of Peer Review Reports: A Methodological Systematic Review." *BMC Medical Research Methodology*, 19(1), 48. Also: EQUATOR Network guidelines (CONSORT, PRISMA, STROBE, etc.). | Standardized reporting guidelines and structured review forms that require specific information (e.g., CONSORT checklist for clinical trials). Journals increasingly require structured review reports. | The closest existing practice to the Confidence Chain's proposed peer review standard. CONSORT checklists are machine-parseable structured requirements. The Confidence Chain's proposed five-part review standard (Section 7 of the paper) extends this idea from methodological reporting to epistemic provenance. | Domain-specific (mostly biomedical). Not claim-level: checklists verify methodological quality, not the provenance of each individual claim. No confidence scores. No cross-framework evaluation. No equivalency classes. No decomposition theorem. |
| **Overlay journals and decoupled review** | Priem, J. & Hemminger, B. (2012). "Decoupling the scholarly journal." *Frontiers in Computational Neuroscience*, 6:19. Also: Episciences platform (episciences.org); Discrete Analysis (overlay journal on arXiv). | Separates review from publication. Papers are posted on preprint servers; overlay journals add a review layer. Decouples the four functions of journals: registration, certification, dissemination, archiving. | Architectural precursor to the Confidence Chain's proposal. If review is decoupled from publication, it becomes possible to add structured epistemic evaluation as a separate layer -- exactly what the Confidence Chain proposes. An overlay confidence layer on arXiv is the natural implementation. | Decoupling is architectural, not epistemic. Overlay journals still use traditional review criteria. No confidence propagation. No provenance triples. No cross-framework evaluation requirement. |
| **REVIEWS ontology / structured review output** | Shotton, D. & Peroni, S. (2018). Various extensions to SPAR (Semantic Publishing and Referencing) ontologies for modeling review activities. Also: Tennant, J. P. et al. (2017). "A multi-disciplinary perspective on emergent and future innovations in peer review." *F1000Research*, 6:1151. | Proposals to make peer review output machine-readable using semantic web ontologies. SPAR ontologies model the publishing workflow including review. | Direct infrastructure for machine-readable review. If reviews are encoded in RDF using SPAR ontologies, confidence chains could be computed automatically. Tennant et al.'s survey of peer review innovations catalogs the landscape the Confidence Chain extends. | Proposals, not widely adopted infrastructure. No confidence propagation model. No decomposition theorem. No equivalency classes. The ontologies model the *process* of review, not the *epistemic content* of review judgments. |

---

## 7. Summary: The Gap

The table below maps each component of the Confidence Chain proposal to the closest existing work and identifies what remains unaddressed.

| Confidence Chain Component | Closest Existing Work | What Exists | What Is Missing |
|---|---|---|---|
| **Provenance triples** (attribution, evidence, derivation) | W3C PROV-O; Nanopublications | General provenance model (PROV-O). Atomic claim publishing (Nanopubs). | Scientific-knowledge-specific provenance. Evidence typing (formal/empirical/conceptual). Derivation chain recording. Integration with peer review. |
| **Computable confidence scores** (C_1) | Bayesian confirmation theory; h-index; PageRank | Theoretical framework (Bayes). Author-level proxy (h-index). Graph-propagation model (PageRank). | Claim-level confidence. Evidence-weighted scoring. Propagation through derivation chains (not citation chains). Automatic updating on new evidence. |
| **Equivalency classes** (shared premises across frameworks) | Structural realism; Lakatos's hard core | Philosophical arguments for structural comparison (Worrall). Informal notion of shared core assumptions (Lakatos). | Formal specification. Computable identification. Machine-readable encoding. No existing system detects shared premises across frameworks automatically. |
| **Decomposition Theorem** (collapse / contradiction / novel) | Laudan's problem-solving model; Bayesian model comparison | Qualitative comparison criteria (Laudan). Bayesian model selection (Bayes factors). | Formal proof that these three categories are exhaustive. Computable classification. Application to arbitrary framework pairs. No existing work provides a decomposition theorem for cross-framework derivations. |
| **Emergent propositions from intersection** | Small-world networks; interdisciplinarity literature | Empirical observation that interdisciplinary work produces novel results. Network models of knowledge flow. | Formal prediction (Corollary 8a). Systematic method for producing emergent propositions. Confirmation across 43 frameworks and 27 emergent nodes. No existing work provides a method that reliably generates new propositions from framework intersection. |
| **Machine-readable peer review standard** | Registered Reports; CONSORT checklists; SPAR ontologies | Structured reporting guidelines. Machine-readable publishing ontologies. Two-stage review. | Requirement for provenance triples. Requirement for equivalency class assignment. Requirement for cross-framework contradiction checking. Integration of all components into a single review standard. |

---

## 8. Key References (BibTeX-ready)

```
@article{garfield1955citation,
  author = {Garfield, Eugene},
  title = {Citation Indexes for Science: A New Dimension in Documentation through Association of Ideas},
  journal = {Science},
  volume = {122},
  number = {3159},
  pages = {108--111},
  year = {1955}
}

@article{hirsch2005index,
  author = {Hirsch, Jorge E.},
  title = {An Index to Quantify an Individual's Scientific Research Output},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {102},
  number = {46},
  pages = {16569--16572},
  year = {2005}
}

@article{chen2007pagerank,
  author = {Chen, Peng and Xie, Huafeng and Maslov, Sergei and Redner, Sidney},
  title = {Finding Scientific Gems with {Google}'s {PageRank} Algorithm},
  journal = {Journal of Informetrics},
  volume = {1},
  number = {1},
  pages = {8--15},
  year = {2007}
}

@inproceedings{teufel2006citation,
  author = {Teufel, Simone and Siddharthan, Advaith and Tidhar, Dan},
  title = {Automatic Classification of Citation Function},
  booktitle = {Proceedings of EMNLP},
  pages = {103--110},
  year = {2006}
}

@article{jurgens2018citation,
  author = {Jurgens, David and Kumar, Srijan and Hoover, Raine and McFarland, Daniel and Jurafsky, Dan},
  title = {Measuring the Evolution of a Scientific Field through Citation Frames},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {6},
  pages = {391--406},
  year = {2018}
}

@techreport{moreau2013provdm,
  author = {Moreau, Luc and Missier, Paolo},
  title = {{PROV-DM}: The {PROV} Data Model},
  institution = {W3C},
  type = {Recommendation},
  year = {2013},
  url = {https://www.w3.org/TR/prov-dm/}
}

@techreport{lebo2013provo,
  author = {Lebo, Timothy and Sahoo, Satya and McGuinness, Deborah and others},
  title = {{PROV-O}: The {PROV} Ontology},
  institution = {W3C},
  type = {Recommendation},
  year = {2013},
  url = {https://www.w3.org/TR/prov-o/}
}

@article{groth2010nanopublication,
  author = {Groth, Paul and Gibson, Andrew and Velterop, Jan},
  title = {The Anatomy of a Nanopublication},
  journal = {Information Services \& Use},
  volume = {30},
  number = {1-2},
  pages = {51--56},
  year = {2010}
}

@article{mons2011value,
  author = {Mons, Barend and van Haagen, Herman and others},
  title = {The Value of Data},
  journal = {Nature Genetics},
  volume = {43},
  pages = {281--283},
  year = {2011}
}

@article{peroni2020opencitations,
  author = {Peroni, Silvio and Shotton, David},
  title = {{OpenCitations}, an Infrastructure Organization for Open Scholarship},
  journal = {Quantitative Science Studies},
  volume = {1},
  number = {1},
  pages = {428--443},
  year = {2020}
}

@article{shotton2010cito,
  author = {Shotton, David},
  title = {{CiTO}, the Citation Typing Ontology},
  journal = {Journal of Biomedical Semantics},
  volume = {1},
  number = {Suppl 1},
  pages = {S6},
  year = {2010}
}

@inproceedings{ammar2018semantic,
  author = {Ammar, Waleed and Groeneveld, Dirk and Bhagavatula, Chandra and others},
  title = {Construction of the Literature Graph in {Semantic Scholar}},
  booktitle = {Proceedings of NAACL-HLT},
  pages = {84--91},
  year = {2018}
}

@inproceedings{sinha2015mag,
  author = {Sinha, Arnab and Shen, Zhihong and Song, Yang and Ma, Hao and Eide, Darrin and Hsu, Bo-June and Wang, Kuansan},
  title = {An Overview of {Microsoft Academic Service} ({MAS}) and Applications},
  booktitle = {Proceedings of WWW Companion},
  pages = {243--246},
  year = {2015}
}

@article{priem2022openalex,
  author = {Priem, Jason and Piwowar, Heather and Orber, Richard},
  title = {{OpenAlex}: A Fully-Open Index of Scholarly Works, Authors, Venues, Institutions, and Concepts},
  journal = {arXiv preprint arXiv:2205.01833},
  year = {2022}
}

@book{kuhn1962structure,
  author = {Kuhn, Thomas S.},
  title = {The Structure of Scientific Revolutions},
  publisher = {University of Chicago Press},
  year = {1962}
}

@book{lakatos1978methodology,
  author = {Lakatos, Imre},
  title = {The Methodology of Scientific Research Programmes},
  publisher = {Cambridge University Press},
  year = {1978}
}

@book{laudan1977progress,
  author = {Laudan, Larry},
  title = {Progress and Its Problems},
  publisher = {University of California Press},
  year = {1977}
}

@book{howson2006scientific,
  author = {Howson, Colin and Urbach, Peter},
  title = {Scientific Reasoning: The Bayesian Approach},
  edition = {3rd},
  publisher = {Open Court},
  year = {2006}
}

@article{worrall1989structural,
  author = {Worrall, John},
  title = {Structural Realism: The Best of Both Worlds?},
  journal = {Dialectica},
  volume = {43},
  number = {1-2},
  pages = {99--124},
  year = {1989}
}

@article{shannon1948mathematical,
  author = {Shannon, Claude E.},
  title = {A Mathematical Theory of Communication},
  journal = {Bell System Technical Journal},
  volume = {27},
  number = {3},
  pages = {379--423},
  year = {1948}
}

@article{kolmogorov1965three,
  author = {Kolmogorov, Andrey N.},
  title = {Three Approaches to the Quantitative Definition of Information},
  journal = {Problemy Peredachi Informatsii},
  volume = {1},
  number = {1},
  pages = {3--11},
  year = {1965}
}

@article{leydesdorff2006journals,
  author = {Leydesdorff, Loet},
  title = {Can Scientific Journals Be Classified in Terms of Aggregated Journal-Journal Citation Relations Using the {Journal Citation Reports}?},
  journal = {Journal of the American Society for Information Science and Technology},
  volume = {57},
  number = {5},
  pages = {601--613},
  year = {2006}
}

@inproceedings{shi2010citing,
  author = {Shi, Xiaolin and Leskovec, Jure and McFarland, Daniel A.},
  title = {Citing for High Impact},
  booktitle = {Proceedings of JCDL},
  pages = {49--58},
  year = {2010}
}

@article{hummon1989connectivity,
  author = {Hummon, Norman P. and Dereian, Patrick},
  title = {Connectivity in a Citation Network: The Development of {DNA} Theory},
  journal = {Social Networks},
  volume = {11},
  number = {1},
  pages = {39--63},
  year = {1989}
}

@inproceedings{lopez2009grobid,
  author = {Lopez, Patrice},
  title = {{GROBID}: Combining Automatic Bibliographic Data Recognition and Term Extraction for Scholarship Publications},
  booktitle = {ECDL},
  series = {LNCS},
  volume = {5714},
  pages = {473--474},
  year = {2009}
}

@inproceedings{cohan2020specter,
  author = {Cohan, Arman and Feldman, Sergey and Beltagy, Iz and Downey, Doug and Weld, Daniel S.},
  title = {{SPECTER}: Document-level Representation Learning using Citation-informed Transformers},
  booktitle = {Proceedings of ACL},
  pages = {2270--2282},
  year = {2020}
}

@inproceedings{wadden2020scifact,
  author = {Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and Cohan, Arman and others},
  title = {Fact or Fiction: Verifying Scientific Claims},
  booktitle = {Proceedings of EMNLP},
  pages = {7534--7550},
  year = {2020}
}

@inproceedings{luan2018scierc,
  author = {Luan, Yi and He, Luheng and Ostendorf, Mari and Hajishirzi, Hannaneh},
  title = {Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction},
  booktitle = {Proceedings of EMNLP},
  pages = {3219--3232},
  year = {2018}
}

@article{ross-hellauer2017open,
  author = {Ross-Hellauer, Tony},
  title = {What is Open Peer Review? A Systematic Review},
  journal = {F1000Research},
  volume = {6},
  pages = {588},
  year = {2017}
}

@article{chambers2013registered,
  author = {Chambers, Christopher D.},
  title = {Registered Reports: A New Publishing Initiative at {Cortex}},
  journal = {Cortex},
  volume = {49},
  number = {3},
  pages = {609--610},
  year = {2013}
}

@article{priem2012decoupling,
  author = {Priem, Jason and Hemminger, Bradley},
  title = {Decoupling the Scholarly Journal},
  journal = {Frontiers in Computational Neuroscience},
  volume = {6},
  pages = {19},
  year = {2012}
}

@article{tennant2017innovations,
  author = {Tennant, Jonathan P. and Dugan, Jonathan M. and Graziotin, Daniel and others},
  title = {A Multi-Disciplinary Perspective on Emergent and Future Innovations in Peer Review},
  journal = {F1000Research},
  volume = {6},
  pages = {1151},
  year = {2017}
}

@article{berners-lee2001semantic,
  author = {Berners-Lee, Tim and Hendler, James and Lassila, Ora},
  title = {The Semantic Web},
  journal = {Scientific American},
  volume = {284},
  number = {5},
  pages = {34--43},
  year = {2001}
}

@incollection{kuhn1977objectivity,
  author = {Kuhn, Thomas S.},
  title = {Objectivity, Value Judgment, and Theory Choice},
  booktitle = {The Essential Tension},
  pages = {320--339},
  publisher = {University of Chicago Press},
  year = {1977}
}

@article{ladyman1998structural,
  author = {Ladyman, James},
  title = {What is Structural Realism?},
  journal = {Studies in History and Philosophy of Science},
  volume = {29},
  number = {3},
  pages = {409--424},
  year = {1998}
}
```
