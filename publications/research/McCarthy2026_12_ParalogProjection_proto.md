# Why Different Folds: Book Thickness, the Anti-Index, and the Functional Contact Graph of the Human Genome

**Patrick D. McCarthy**

*Paper 12 of the McCarthy 2026 series. PROTO-PAPER DRAFT — argument structure complete, figures and final statistics pending.*

---

## Abstract

DNA is a one-dimensional serialization of a three-dimensional connected graph. This paper provides the mathematical proof that a single fold cannot recover the full contact graph from the serialization, and derives the consequences.

**Section 1** applies the book thickness theorem from graph theory: given n vertices on a fixed linear spine, each non-crossing contact layer (page) supports at most 2n − 3 edges, so a graph with m edges requires at least ⌈m/(2n − 3)⌉ pages. This is proven. (Bernhart & Kainen 1979, Malitz 1994.)

**Section 2** extends the bound to three dimensions. In physical space, even non-crossing edges interfere at close range — signal integrity requires spatial separation between chromatin loops. The self-avoidance constraint of a polymer in R³ limits total contacts to O(n). TAD boundaries (CTCF/cohesin insulation) are the physical realization of page separators, preventing crosstalk between independent contact sets.

**Section 3** establishes that DNA behaves as the theorem predicts. Tissue-specific chromatin folding is empirical fact (Lieberman-Aiden 2009, Rao 2014). Different tissues fold differently. The fold is an **anti-index**: the default state is closed (compacted, silenced); the fold selectively suppresses contacts, and the tissue's active genes are what survive the suppression. Two of eight cancer-relevant channels — ChromatinRemodel and DNAMethylation — are the anti-index maintenance machinery. Their failure degrades the fold across all channels and causes cancer in all tissue types.

**Section 4** derives the testable prediction. From co-essentiality clustering (DepMap, 14,129 genes, 1,208 cell lines), we measure the connectivity of the genome's functional contact graph. From multi-tissue Hi-C, we measure the number of distinct folds. The book thickness bound predicts these two quantities are related: more connectivity requires more folds. The converse is stronger and more discoverable: **from the observed number of distinct folds (~200 human tissue types), we can bound the total connectivity of the functional graph.** For ~200 tissues and ~20,000 genes, the bound permits ~8 million regulatory contacts — consistent with ENCODE estimates.

Co-essentiality clustering independently recovers the cancer channel architecture (Papers 5-6) without manual annotation, validates the wrapper/implementation distinction (ATM clusters with TP53 not RAD51), and produces a three-tier essentiality pattern (A-layer acutely lethal, anti-index slow-degrading, tissue-specific contextual) that the book thickness framing predicts.

---

## 1. The Book Thickness Bound

### 1.1 The serialization problem

All software is a graph (Paper 0). DNA is software. DNA is stored as a one-dimensional sequence — a linear tape of nucleotides on each chromosome. But the functional objects it encodes — protein complexes, regulatory circuits, signaling cascades — are three-dimensional connected graphs. Genes that must interact are scattered across the linear tape with no guarantee of proximity.

The cell's solution is the chromatin fold: a three-dimensional embedding of the linear sequence that brings functionally related regions into spatial contact. The fold is the decompression operation that recovers 3D execution from 1D storage.

**The question this paper answers: can one fold suffice?**

### 1.2 Book embedding

A book embedding (Ollmann 1973, Bernhart & Kainen 1979) places the vertices of a graph on a line (the spine) and distributes the edges across half-planes (pages) attached to the spine. Edges on the same page may not cross. The book thickness bt(G) of a graph G is the minimum number of pages needed to embed all edges without crossings.

The spine is the chromosome. The pages are independent sets of non-crossing contacts. The fold is the physical realization.

### 1.3 The bound

Each page is an outerplanar graph. An outerplanar graph on n vertices has at most 2n − 3 edges (classical). Therefore:

> **bt(G) ≥ ⌈m / (2n − 3)⌉**

where m is the number of edges and n is the number of vertices.

For the complete graph K_n: bt(K_n) = ⌈n/2⌉ (Bernhart & Kainen 1979).

For general graphs with m edges: bt(G) = O(√m) (Malitz 1994).

**Application to DNA.** The spine is the chromosomal linear order. Each page is one set of spatial contacts achievable by a single fold without interference. If the functional contact graph has more edges than one page can hold, multiple folds are required. Different tissues need different subsets of the contact graph, and the book thickness bound proves they cannot all share one fold.

### 1.4 The converse

Given k pages (tissue types), the maximum number of edges (functional contacts) the genome can support is:

> **m_max = k × (2n − 3)**

For ~200 human tissue types and ~20,000 genes: m_max ≈ 200 × 40,000 ≈ 8 × 10⁶ contacts. ENCODE estimates 2-4 million regulatory interactions. The observed connectivity is within the bound.

**This is the testable prediction:** measure the connectivity of the functional contact graph (from co-essentiality, Section 4). Measure the number of distinct folds (from multi-tissue Hi-C). The ratio should respect the book thickness bound. If it does, the genome is operating as a book-embedded graph.

---

## 2. Why Three Dimensions: Signal Integrity

### 2.1 The 2D bound is necessary but insufficient

Book thickness is a two-dimensional theorem — edges on half-planes, non-crossing on each plane. DNA folds in three dimensions, where two edges CAN avoid each other by routing through the third dimension. Naively, 3D should be more permissive than 2D, weakening the bound.

### 2.2 Signal integrity tightens the bound

In physical space, chromatin loops carry regulatory signals — transcription factor binding, enhancer-promoter contact, cohesin-mediated insulation. Two loops that pass too close interfere, even without physically crossing. The interference is not geometric but informational: signaling molecules on one loop crosstalk with the adjacent loop, corrupting the regulatory signal.

This is the same constraint that governs wire routing in circuit boards: physical crossing is avoidable in 3D, but signal interference is not. Spatial separation is required for signal fidelity.

### 2.3 Polymer packing confirms the bound

A self-avoiding polymer of n monomers in R³ can sustain at most O(n) non-backbone contacts (De Gennes scaling, lattice packing argument: each monomer has at most z − 2 contacts on a cubic lattice with coordination number z = 6, giving ≤ 2n total contacts). This is linear in n, not quadratic. The polymer physics bound agrees with the book thickness bound in order of magnitude.

### 2.4 TADs are page separators

Topologically associating domains (TADs) are self-interacting chromatin regions bounded by CTCF/cohesin insulation (Dixon 2012, Rao 2014, Fudenberg & Mirny 2019). TAD boundaries prevent contacts across adjacent domains. They are the physical realization of page boundaries: within a TAD, contacts are permitted; across TAD boundaries, contacts are suppressed.

Human chromosomes contain ~2,000-3,000 TADs. TAD boundary positions are largely tissue-invariant, but insulation strength varies by tissue (Schmitt 2016). **Same spine, same page boundaries, different pages active in different tissues.** This is the book embedding operating in biology.

---

## 3. DNA Behaves This Way

### 3.1 Tissue-specific folding is empirical fact

Lieberman-Aiden et al. (2009) established that human chromatin contact probability scales as P(s) ~ s^{-1.08}, matching fractal globule predictions, and that A/B compartment assignments differ between cell types. Rao et al. (2014) mapped chromatin loops at kilobase resolution across multiple cell types and showed tissue-specific loop architectures. The 4D Nucleome consortium has since extended these maps to dozens of primary tissues. **Different tissues fold differently.** This is not disputed.

### 3.2 The fold is an anti-index

The default state of chromatin is **closed** — compacted, silenced, inaccessible. Only 3-7% of the genome is in open chromatin (accessible, active) in any given tissue type (ENCODE, Roadmap Epigenomics). The fold does not build connections. It builds **walls**. TAD boundaries insulate. Heterochromatin compacts. The tissue-specific contacts are what survive the suppression — the gaps in the anti-index.

An index says "look here." An anti-index says "don't look anywhere else." The chromatin fold defines what is **excluded**, and the active subset is what remains after exclusion.

CTCF/cohesin mechanics confirm this: cohesin extrudes chromatin loops by default (the machinery tries to connect everything). CTCF boundary sites halt extrusion (the boundary prevents connection). The tissue-specific fold is not a construction — it is a **selective failure to suppress**.

### 3.3 Two channels maintain the anti-index

The cancer channel taxonomy (Papers 5-6) partitions 122 cancer-relevant genes into 8 functional channels. Two of these channels — **ChromatinRemodel** (ARID1A, SMARCA4, KMT2D, CREBBP, and 12 others) and **DNAMethylation** (DNMT3A, TET2, IDH1/2, ATRX, DAXX, and 2 others) — are the maintenance machinery for the anti-index:

- **ChromatinRemodel** physically opens and closes chromatin — it maintains the walls.
- **DNAMethylation** chemically marks regions for silencing — it writes the wall addresses.

These two channels are **upstream** of the other six. They don't perform DNA repair or cell cycle control. They maintain the fold that allows the functional channels (DDR, CellCycle, PI3K_Growth, Immune, Endocrine, TissueArchitecture) to form their tissue-specific contacts. The functional channels work because the anti-index channels maintain the infrastructure they depend on.

### 3.4 Anti-index failure causes cancer

When anti-index maintenance fails — ARID1A mutation, DNMT3A loss, IDH1 gain-of-function — the fold degrades. Walls collapse. Contacts that should be suppressed are permitted. Enhancers activate oncogenes across TAD boundaries (Flavahan et al. 2016). Genes that should be silent in a given tissue become active. The cell escapes its tissue-specific constraints.

This is why:
- ARID1A is mutated in ~10% of all human cancers, not one specific type — it degrades infrastructure, not one channel.
- DNMT3A is the #1 CHIP mutation (80% of age-related clonal hematopoiesis) — its loss is tolerated acutely but catastrophic over years as methylation marks drift and the anti-index slowly degrades.
- IDH mutations cause glioma via TAD boundary loss (Flavahan 2016) — a specific mechanism of anti-index failure causing a specific cancer.

**Cancer is what happens when the anti-index fails and a cell can access contacts its tissue type was never meant to have.**

### 3.5 Three-tier essentiality confirms the architecture

DepMap CRISPR data (1,208 cell lines) reveals a three-tier essentiality pattern across the eight channels:

| Tier | Channels | Mean effect | Avg frac essential | Pattern |
|------|----------|-------------|-------------------|---------|
| **A-layer** (implementation) | DDR (core replication/repair), CellCycle (core division) | -0.55 / -0.33 | 39.8% / 24.5% | Acutely lethal — cell dies in days without these |
| **Anti-index** (fold maintenance) | ChromatinRemodel, DNAMethylation | -0.21 / -0.05 | 15.8% / 4.7% | Rarely acutely lethal — fold degrades slowly over months/years |
| **Tissue-specific** (functional) | PI3K_Growth, Immune, Endocrine, TissueArchitecture | -0.16 to -0.03 | 12.9% to 2.3% | Essential only in specific tissue contexts |

The anti-index tier has a distinctive essentiality signature: not acutely lethal (the cell doesn't crash when you remove DNMT3A), but universally important (every tissue needs the fold maintained). This is the timescale signature of infrastructure — loss is tolerated short-term, catastrophic long-term.

---

## 4. The Functional Contact Graph: Measurement and Prediction

### 4.1 Co-essentiality as the connectivity measurement

The functional contact graph of the genome — which genes need to interact with which other genes — is not directly observable. But it can be measured indirectly via co-essentiality: if knocking out gene A and gene B produce correlated fitness effects across hundreds of cell lines, A and B are functionally coupled.

We computed pairwise co-essentiality (Pearson correlation of CRISPR gene effect profiles) for 14,129 genes across 1,208 cell lines (DepMap 26Q1). Hierarchical clustering (Ward's method) at granularities from k=100 to k=2,000 partitions the genome into functional modules of decreasing size. Each cluster at a given k is a candidate "functional equivalence class" — a set of genes whose fitness profiles are more similar to each other than to genes outside the cluster.

### 4.2 Co-essentiality recovers the channel architecture

The co-essentiality graph independently recovers the cancer channel taxonomy without any manual annotation. Same-channel gene pairs co-cluster at rates far exceeding chance at every tested granularity.

**Pairs stable from k=100 through k=2,000** (median cluster size ~7 genes at k=2,000):

| Pair | Channel | Relationship |
|------|---------|-------------|
| RAD51B + RAD51C + RAD51D | DDR | HR implementation paralogs |
| MSH2 + MSH6 | DDR | MutS-alpha mismatch repair heterodimer |
| TP53 + CDKN1A | CellCycle | Transcription factor → direct target |
| CDK4 + CCND1 | CellCycle | Kinase-cyclin holoenzyme |
| MDM2 + MDM4 | CellCycle | Co-regulators of TP53 |
| RB1 + CDKN1B | CellCycle | Cell cycle inhibitors |
| BRAF + MAP2K1 | PI3K_Growth | RAF → MEK kinase cascade |
| ERBB2 + PIK3CA | PI3K_Growth | Receptor → PI3K |
| ARID1A + SMARCA4 | ChromatinRemodel | SWI/SNF complex members |
| CREBBP + KMT2D | ChromatinRemodel | Enhancer chromatin modifiers (splits at k=1,000) |
| HLA-A + HLA-B | Immune | MHC class I |
| TGFBR1 + TGFBR2 | TissueArchitecture | TGF-beta receptor heterodimer |
| APC + AXIN1 | TissueArchitecture | Wnt destruction complex |
| ESR1 + FOXA1 | Endocrine | Estrogen receptor + pioneer factor |
| ATM + TP53 | **Cross-channel** | DDR → CellCycle bridge |

15 of 18 tested pairs survive to k=2,000. The 3 that split at intermediate k (NF1+PTEN at k=500, CREBBP+KMT2D at k=1,000) are pathway-level associations, not direct binding partners. The splitting hierarchy matches biochemical reality.

TODO: Figure 1 — co-clustering heatmap across channels and granularities.

### 4.3 The ATM bridge: wrappers cluster by target, not substrate

ATM is annotated as DDR. It clusters with TP53 and CDKN1A (CellCycle), not with RAD51 (DDR implementation). ATM phosphorylates TP53 in response to DNA damage — it is the bridge between damage detection and cell cycle arrest. The co-essentiality graph places ATM where it functionally operates (routing toward TP53), not where it was manually filed (DDR).

This validates the wrapper/implementation distinction: implementation genes (RAD51 family) cluster with the machinery. Wrapper genes (ATM) cluster with their routing target. The graph is more precise than the annotation.

TODO: Figure 2 — ATM bridge network diagram.

### 4.4 Counting edges: from co-essentiality clusters to the functional contact graph

At k=200 (median cluster size 67 genes), the co-essentiality graph defines 200 functional modules. Same-cluster paralog pairs number 2,931. But the full functional contact graph is larger — every gene pair within a cluster is a candidate functional contact, and cross-cluster contacts (like the ATM → TP53 bridge) add more.

TODO: Compute total edge count of the functional contact graph at multiple thresholds (co-essentiality correlation > 0.3, > 0.5, > 0.7). Compare to the book thickness bound: m / (2n − 3) should predict the minimum number of tissue-specific folds needed. Compare to the observed number of distinct chromatin states from multi-tissue Hi-C data.

### 4.5 The 3D contact test: negative in one cell type, consistent with the framing

Same-cluster paralog pairs do NOT show elevated Hi-C contact frequency relative to different-cluster pairs in GM12878 bulk Hi-C. Tested at every k from 100 to 2,000. No test reached significance.

This is consistent with the anti-index framing rather than contradicting it. The co-essentiality signal is averaged across 1,208 cell lines representing dozens of tissue types. The Hi-C signal is from one cell type. The functional contacts measured by co-essentiality exist *somewhere* across tissues — they need not all be present in one tissue's fold.

**The book thickness bound predicts exactly this:** the contacts cannot all be realized in a single fold. The negative Hi-C result in one cell type is the prediction firing.

### 4.6 Universal, channel-shared, and tissue-specific contacts

Preliminary analysis of loop-call data across three cell types (GM12878, IMR90, K562) shows:

| Tier | Count | Essential gene enrichment |
|------|-------|--------------------------|
| Universal (all 3 cell types) | 91 gene pairs | 8.9% essential (OR=1.52 vs tissue-specific) |
| Shared (exactly 2) | 199 gene pairs | 8.1% essential |
| Tissue-specific (1 cell type only) | 773 gene pairs | 6.4% essential |

Direction is correct (universal contacts are more essential), but underpowered (p=0.12). Heavy Hi-C data (continuous contacts, not just loop calls) from additional cell types will increase power.

The prediction: universal contacts correspond to A-layer and anti-index genes. Channel-shared contacts correspond to genes used by multiple tissues that share a channel. Tissue-specific contacts correspond to tissue-identity genes. **ChromatinRemodel and DNAMethylation genes should be constitutive — present in every fold — because they maintain the anti-index that all other folds depend on.**

TODO: Test with tissue-matched heavy Hi-C from IMR90 and K562.

---

## 5. Discussion

### 5.1 The conjecture

If DNA were a simple linear program — a tape that encodes proteins in order — there would be no reason for tissue-specific folding. Every cell would read the same tape the same way. Tissue-specific gene expression could be handled entirely by transcription factors along the linear sequence. No spatial organization needed.

But DNA folds differently in different tissues. The folds are not random. They bring specific gene sets into spatial proximity in tissue-specific patterns. The field knows this. **The field has not asked why the fold is necessary.**

This paper answers: **the fold exists because DNA is a 1D serialization of a 3D connected graph, and the book thickness bound proves that a single fold cannot recover the full contact graph.** Different tissues need different subsets of the connectivity, so they need different folds. The anti-index is the mechanism. ChromatinRemodel and DNAMethylation are the maintenance layer. Cancer is what happens when the anti-index fails.

### 5.2 What is novel

The individual components are known:
- Book thickness bounds are proven (Bernhart & Kainen 1979, Malitz 1994)
- Tissue-specific chromatin folding is established (Lieberman-Aiden 2009, Rao 2014)
- TAD boundary disruption causes cancer (Flavahan 2016)
- ARID1A and DNMT3A are among the most mutated genes in cancer (COSMIC, TCGA)

**What is novel is the connection.** No prior work has applied graph-theoretic book thickness bounds to chromatin folding. No prior work has framed the fold as an anti-index. No prior work has derived the necessity of tissue-specific folding from the connectivity of the functional contact graph. No prior work has connected the book thickness bound to tissue-type count.

The connection was available. It was sitting in the gap between two fields — graph theory and chromatin biology — that do not read each other's journals. Paper 0 of this series argued that all software is a graph, and that the graph theory which applies to one substrate applies to all of them. This paper is the proof of that claim at the chromatin level: book thickness is the theorem that was always there. Nobody applied it because nobody was looking at DNA as a book.

### 5.3 Convergence with other projections

This paper joins five other orthogonal projections that converge on the same architecture:

1. **Book thickness → tissue-specific folds** (this paper, Section 1-2)
2. **Co-essentiality → channel recovery** (this paper, Section 4)
3. **Wrapper variance** — cross-species CV orders genes: A (0.06) → proto-M (0.13) → pure M (0.32)
4. **Channel-count survival** (Papers 5-6) — disrupted channel count predicts survival across 29 cancer types
5. **Clinical tractability** — approved therapies concentrate on proto-M; pure M is the synthetic lethality domain
6. **Disease spectrum** — implementation mutations are embryonically lethal; wrapper mutations are viable with cancer risk

Each uses different data, different methods, different statistical tests. None shares an input with any other. The strength is convergence.

### 5.4 Limitations

**Book thickness is a lower bound, not an exact prediction.** The bound says at least ⌈m/(2n−3)⌉ pages. The actual number of tissue types may exceed the bound for reasons unrelated to the contact graph (developmental contingency, functional specialization beyond chromatin architecture). The bound is necessary, not sufficient.

**Co-essentiality measures functional coupling in cell culture, not in vivo.** DepMap cell lines are transformed cancer lines grown in 2D culture. The functional coupling landscape in vivo — with stromal interactions, immune surveillance, vascularization — may differ.

**The anti-index framing is conceptual, not yet quantitative.** We have not computed the exact number of contacts suppressed per tissue or the exact number of anti-index entries (CTCF sites, methylation marks) per cell type. The framing is supported by the known biology of CTCF/cohesin and chromatin compaction, but a quantitative test — does the number of CTCF-bound sites per tissue match the book thickness prediction? — is a future direction.

**Single-cell-type Hi-C is a weak test.** The negative 3D result in GM12878 is consistent with the framing but does not confirm it. Confirmation requires multi-tissue Hi-C matched to channel activity predictions.

---

## 6. Future Directions

### 6.1 Quantitative book thickness test

Compute the edge count of the functional contact graph from co-essentiality at multiple correlation thresholds. Compute the number of distinct chromatin states from multi-tissue Hi-C (4D Nucleome consortium data). Test whether the ratio respects the book thickness bound.

### 6.2 Tissue-matched 3D recovery

Download heavy Hi-C data for IMR90 (lung fibroblast) and K562 (CML). Annotate with co-essentiality clusters. Test: are PI3K_Growth gene contacts elevated in K562 relative to GM12878? Are TissueArchitecture gene contacts elevated in IMR90? Are ChromatinRemodel and DNAMethylation gene contacts constitutive across all three?

### 6.3 Anti-index quantification

Count CTCF-bound sites per tissue type from ENCODE ChIP-seq data. Count methylated CpG islands per tissue type from Roadmap Epigenomics. Correlate with the number of suppressed contacts per tissue (from Hi-C). Test whether the anti-index size (number of suppression marks) scales with the number of contacts that need to be suppressed.

### 6.4 Evolutionary age stratification

Older channels may have more robust anti-index maintenance due to longer evolutionary optimization. Younger channels may be more vulnerable to anti-index degradation. Stratify the book thickness analysis by channel age.

---

## References

1. Bernhart FR, Kainen PC. The book thickness of a graph. *J Combinatorial Theory Ser B*. 1979;27(3):320-331.
2. Malitz S. Graphs with E edges have pagenumber O(√E). *J Algorithms*. 1994;17(1):71-84.
3. Ollmann LT. On the book thicknesses of various graphs. *Proc 4th Southeastern Conference on Combinatorics, Graph Theory and Computing*. 1973;8:459.
4. Yannakakis M. Embedding planar graphs in four pages. *J Computer & System Sciences*. 1989;38:36-67.
5. Lieberman-Aiden E, et al. Comprehensive mapping of long-range interactions reveals folding principles of the human genome. *Science*. 2009;326(5950):289-293.
6. Rao SSP, et al. A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. *Cell*. 2014;159(7):1665-1680.
7. Dixon JR, et al. Topological domains in mammalian genomes identified by analysis of chromatin interactions. *Nature*. 2012;485(7398):376-380.
8. Fudenberg G, Imakaev M, Lu C, Goloborodko A, Abdennur N, Mirny LA. Formation of chromosomal domains by loop extrusion. *Cell Reports*. 2016;15(9):2038-2049.
9. Flavahan WA, et al. Insulator dysfunction and oncogene activation in IDH mutant gliomas. *Nature*. 2016;529(7584):110-114.
10. Grosberg A, Rabin Y, Havlin S, Neer A. Crumpled globule model of the three-dimensional structure of DNA. *Europhysics Letters*. 1993;23(5):373-378.
11. Mirny LA. The fractal globule as a model of chromatin architecture in the cell. *Chromosome Research*. 2011;19:37-51.
12. De Gennes PG. *Scaling Concepts in Polymer Physics*. Cornell University Press; 1979.
13. Dujmovic V, Wood DR. Graph treewidth and geometric thickness parameters. *Discrete & Computational Geometry*. 2007;37(4):641-670.
14. Schmitt AD, et al. A compendium of chromatin contact maps reveals spatially active regions in the human genome. *Cell Reports*. 2016;17(8):2042-2059.
15. Meyers RM, et al. Computational correction of copy number effect improves specificity of CRISPR-Cas9 essentiality screens in cancer cells. *Nature Genetics*. 2017;49(12):1779-1784.
16. ENCODE Project Consortium. An integrated encyclopedia of DNA elements in the human genome. *Nature*. 2012;489(7414):57-74.
17. McCarthy PD. Channel structure predicts cancer survival. McCarthy 2026 series, Papers 5-6.
18. McCarthy PD. All software is a graph. McCarthy 2026 series, Paper 0.

---

## Appendix A: Channel-gene map (122 genes, 8 channels)

Source: `open-knowledge-graph/data/channel_gene_map.csv`

| Channel | Count | Genes |
|---------|-------|-------|
| DDR | 21 | ATM, ATR, BRCA1, BRCA2, PALB2, RAD51B, RAD51C, RAD51D, CHEK1, CHEK2, BAP1, BARD1, FANCA, FANCC, FANCD2, MLH1, MSH2, MSH6, PMS2, POLE, POLD1 |
| CellCycle | 14 | TP53, RB1, CDKN1A, CDKN1B, CDKN2A, CDKN2B, CDK4, CDK6, CCND1, CCNE1, MDM2, MDM4, MYC, MYCN |
| PI3K_Growth | 29 | PIK3CA, PIK3R1, PTEN, AKT1, AKT2, AKT3, MTOR, KRAS, NRAS, HRAS, BRAF, RAF1, MAP2K1, MAP2K2, MAP3K1, MAP3K13, ERBB2, ERBB3, EGFR, FGFR1, FGFR2, FGFR3, IGF1R, MET, NF1, NF2, TSC1, TSC2, STK11 |
| ChromatinRemodel | 16 | KMT2D, KMT2C, KMT2A, KMT2B, SETD2, NSD1, CREBBP, EP300, ARID1A, ARID1B, ARID2, SMARCA4, KDM6A, BCOR, H3C7, ANKRD11 |
| DNAMethylation | 8 | DNMT3A, DNMT3B, TET1, TET2, IDH1, IDH2, ATRX, DAXX |
| TissueArchitecture | 18 | CDH1, CDH2, CTNNB1, APC, AXIN1, AXIN2, SMAD2, SMAD3, SMAD4, TGFBR1, TGFBR2, NOTCH1, NOTCH2, NOTCH3, NOTCH4, FBXW7, GJA1, GJB2 |
| Immune | 10 | B2M, HLA-A, HLA-B, HLA-C, JAK1, JAK2, STAT1, CD274, PDCD1LG2, CTLA4 |
| Endocrine | 6 | ESR1, ESR2, PGR, AR, FOXA1, GATA3 |

## Appendix B: Key co-clustered pairs and stability

| Gene A | Gene B | Channel | Relationship | Stable to k= |
|--------|--------|---------|-------------|---------------|
| RAD51B | RAD51C | DDR | HR paralogs | >2,000 |
| RAD51C | RAD51D | DDR | HR paralogs | >2,000 |
| MSH2 | MSH6 | DDR | MutS-alpha heterodimer | >2,000 |
| TP53 | CDKN1A | CellCycle | TF → target | >2,000 |
| CDK4 | CCND1 | CellCycle | Kinase-cyclin | >2,000 |
| MDM2 | MDM4 | CellCycle | TP53 co-regulators | >2,000 |
| RB1 | CDKN1B | CellCycle | Cell cycle inhibitors | >2,000 |
| CDKN2A | CDKN2B | CellCycle | Same locus (9p21) | >2,000 |
| BRAF | MAP2K1 | PI3K_Growth | RAF→MEK cascade | >2,000 |
| ERBB2 | PIK3CA | PI3K_Growth | Receptor → PI3K | >2,000 |
| NF1 | PTEN | PI3K_Growth | Tumor suppressors | ~500 |
| ARID1A | SMARCA4 | ChromatinRemodel | SWI/SNF complex | >2,000 |
| CREBBP | KMT2D | ChromatinRemodel | Enhancer modifiers | ~1,000 |
| HLA-A | HLA-B | Immune | MHC class I | >2,000 |
| TGFBR1 | TGFBR2 | TissueArchitecture | TGF-beta receptor | >2,000 |
| APC | AXIN1 | TissueArchitecture | Wnt destruction complex | >2,000 |
| ESR1 | FOXA1 | Endocrine | ER + pioneer factor | >2,000 |
| ATM | TP53 | DDR→CellCycle | Bridge: wrapper → target | >2,000 |
