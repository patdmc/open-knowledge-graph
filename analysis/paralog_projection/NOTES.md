# Paralog as Projection — Working Notes

Title: **Paralog as Projection**
Subtitle (placeholder): *A Two-Dimensional Serialization of Three-Dimensional
Functional Objects, Recovered by the Fold*

Everything in this file is preserved verbatim from the conversation that
developed the framing. Do not edit for brevity — edit for publication
in separate drafts. This is the source-of-truth theoretical notebook.

---

## Core claim: 3 → 2 → 3, round trip

Biological objects are 3D. Folded proteins. Chromatin complexes. Assembled
machines. They have literal shapes in nuclear space.

DNA is a 2D serialization of those 3D objects. Linear sequence position is
one axis. Helical phase / groove selection is the second axis — which reader
gets access to which base pair depends on rotational position, so the helix
is not decorative; it is a channel selector.

The fold is a decompression operation that recovers 3D execution from 2D
storage. It adds exactly one dimension, the minimum possible.

**Round trip: 3 → 2 → 3. Store in 2D, execute in 3D.**

The projection is lossy. The recovery is context-conditioned. Both of these
properties are forced, not incidental.

---

## Why 2D is forced by information theory (not N-dimensional)

Earlier drafts of the framing invoked higher-dimensional functional graphs.
That was overcomplication. The real thing is 3D because biology happens in
3D space. Proteins, complexes, chromatin, reactions — all 3D.

DNA is 2D:
- Axis 1: linear position along the tape (5'→3')
- Axis 2: helical phase — rotation angle determining which machines can
  physically reach which base pairs. Major-groove readers and minor-groove
  readers see different slices of the same sequence.

The fold is 3D:
- Adds exactly one dimension to the 2D storage
- Minimum possible dimensional expansion to hold a 3D object

No exotic math. No high-dimensional topology. Just: storage format,
execution format, and the operation between them.

---

## Why the projection must be lossy

Information theory forces it. A projection from 3D to 2D collapses one
dimension. Any finite collection of 2D projections is **strictly less
dense** than the 3D object that cast them. The Radon transform recovers
3D from 2D only in the limit of infinitely many projections. Every
real-world reconstruction is an approximation constrained by a prior.

So DNA cannot fully encode the 3D functional object. This is not a
hand-wave. It is forced.

Where does the missing information come from? **Context.** The cellular
environment, the inherited protein machinery, the chromatin state, the
physical laws of folding, the chemistry of the nucleus. The 2D encoding is
under-determined. The context that reads the encoding disambiguates it.

Different contexts recover different 3D objects from the same 2D tape.
That is what differential gene expression, cell type specialization, and
tissue-specific chromatin folding *are*, mechanically. Same tape, different
interpreter, different 3D output.

**This is CAP-under-finite-context applied to the genome.** A finite
reader operating on a finite encoding cannot recover an unbounded object.
The reader needs context. The context is finite. So the set of 3D objects
any given genome can recover is bounded by the contexts the cell can hold.
The same DNA produces radically different cell types because different
cellular contexts project the same 2D tape onto different 3D attractors.

The fold is not deterministic decompression. It is context-conditioned
recovery.

---

## Why this format survived evolution (the efficiency argument)

A 2D encoding that requires context to unpack is on its face "inefficient"
— the encoding doesn't fully specify the output. Yet it has survived 3.5
billion years of selection pressure. Either the format is optimal for the
job, or it would have been eliminated.

The job imposes constraints the 2D format uniquely satisfies:

- **Replication.** You can copy a linear tape base-by-base with high
  fidelity. You cannot directly copy a folded 3D protein. Life requires
  that the storage format be replicable; the execution format need not be.
- **Inheritance.** A 2D encoding is compact and portable across cell
  divisions. A 3D-stored organism would have to pass folded machinery
  directly, which doesn't scale.
- **Editability.** Point mutations, insertions, deletions, and
  rearrangements all operate cleanly on a linear tape. 3D editing would
  require disassembly and reassembly.
- **Cell-type multiplicity.** A 2D genome read by context-specific readers
  produces many 3D phenotypes from one genome. A 3D-stored organism would
  need one complete copy of each cell-type's machinery separately.
- **Density per unit mass.** The helix is extraordinarily dense storage.

The 2D format's apparent "inefficiency" (needs context to unpack) is
exactly what enables the efficiency at the organism level. A cell that
tried to inherit 3D objects directly would be unable to replicate, unable
to differentiate, and unable to evolve. The projection structure is
therefore under strong selection, and the specific 2D shape of DNA —
double helix, antiparallel strands, base pairing, major/minor grooves —
is the minimum structure that satisfies all four constraints.

Watson and Crick saw the helix as a beautiful solution for replication.
The projection frame says it is also the minimum structure for
bidirectional edge encoding, cyclic graph serialization, and
context-conditioned readback. Same structure, multiple forced requirements.

---

## Paralogs as the clearest projection signature

If DNA is a 2D serialization of 3D objects, and two genes encode the same
3D functional node, they are two shadows of one object on the tape. The
question is: when and how do we see it?

Two proximity dimensions are available to us:
- **2D proximity:** linear chromosomal distance (the marginal of the 2D
  encoding that we can measure with current public data)
- **3D proximity:** Hi-C contact frequency after the fold

Projection theory predicts a **handoff** from 2D proximity to 3D proximity
as paralogs age.

| Evolutionary age | 2D proximity | 3D contact | Interpretation |
|---|---|---|---|
| Very recent (NIPs, ≥98% identity) | Adjacent on chromosome | Incidental | Two shadows of one node, still stored where the copy operation left them. No drift, no fold work needed. Degenerate case. |
| Middle age, still in same equivalence class | Decaying | Rising | The cell still needs them as one node, but the linear representation has broken down. The fold takes over. 3D contact rises to compensate for lost 2D adjacency. |
| Ancient, still in same equivalence class | Gone | Dominant | BRCA/RAD51/PALB2 family. Scattered across chromosomes. Held together only by the fold. The paper's anchor case. |
| Ancient, drifted out of equivalence class | Gone | Gone | Olfactory receptors. Two real nodes now. No coupling needed. Negative control / floor. |

**The two curves crossing is the fingerprint of projection recovery.**

Pure descent predicts 2D proximity decays with divergence time (true,
known). Descent has no reason to predict 3D contact rising to compensate.
Projection theory predicts both because it says the cell is defending one
invariant: the shadows of one node must be close in *some* dimension
accessible to the reader.

---

## NIPs are the smoking gun (and why the field missed it)

Nearly Identical Paralogs (≥98% sequence identity, adjacent on the
chromosome) are the recent end of the curve. Two copies of a 2D fragment
encoding the same 3D piece. They sit next to each other because the
duplication put them there. They're identical because nothing has edited
them yet. They encode the same 3D node because there has been no time for
either to mean anything different.

The field sees this and says: "recent tandem duplicates cluster because
duplication is local." True at the mechanism level. Silent at the meaning
level. The deeper statement: **the tape is holding two serialized copies of
one 3D node, stored adjacent because they haven't been moved or edited.**
NIPs are the degenerate case of the projection. Two copies of one thing,
stored where the copy operation left them.

Why did the field not name it this way? Because they inherited their
epistemology from taxonomy and natural history, which assume the thing you
are looking at is the thing itself. That assumption works for beetles and
finches and breaks silently in genomics. Everyone from a wet-lab background
sees a string of letters and builds phylogenies. Everyone from a
mathematical background sees a projection and asks what it is a projection
of. Two communities, same data, different epistemology.

"Science is dancing around the simplest explanation because it is
non-linear." — Patrick, this conversation.

---

## Prior art: Dennler & Ryan 2025

Dennler & Ryan 2025 (NAR Genomics and Bioinformatics, "Evaluating sequence
and structural similarity metrics for predicting shared paralog functions")
already showed that sequence identity alone is insufficient to predict
paralog functional sharing. Structural similarity (AlphaFold TM, LDDT) and
PLM embeddings (ProtT5, ESM2) add 0.05–0.15 AUROC over sequence identity
across shared PPIs, synthetic lethality, and GO semantic similarity.
Dataset: 107,103 human paralog pairs from Ensembl 111.

**What they established:** functional equivalence between paralogs is
richer than sequence descent can capture.

**What they stopped short of:** the spatial / 3D axis. They never checked
whether the richer equivalence signal also predicts 3D contact in the fold.

**Our contribution:** extend the same equivalence features to a new axis
(Hi-C contact), and show that the fold is reading off the same equivalence
structure that sequence + structure + PLM embeddings capture at the
protein layer. Their result at the protein layer; our result at the
nuclear layer.

Their Figure 6 (AUROC for predicting shared function from feature
combinations) becomes our Figure 2 (AUROC for predicting 3D contact from
the same features).

---

## The four panels of the paper

**Panel 1: Dual-proximity handoff.**
X-axis: divergence (Ensembl subtype ladder — Homo sapiens / Primates /
Eutheria / Mammalia / Amniota / Vertebrata / Bilateria / Opisthokonta, plus
perc_id for continuous sorting). Y-axis: two curves, linear chromosomal
distance and 3D Hi-C contact frequency. The prediction: the curves cross
around the decoherence point. Anchor cases: BRCA/RAD51/PALB2 (stayed
coupled, scattered on 13/14/16/17) and olfactory receptors (drifted, same
class family).

**Panel 2: Equivalence-class prediction of 3D contact.**
AUROC for predicting 3D contact from (a) sequence identity alone, (b)
Dennler-style combined features, (c) co-essentiality, (d) shared channel
membership. Direct extension of Dennler Figure 6 with Hi-C contact as the
dependent variable. Sequence-only is the baseline; equivalence features
are expected to beat it by more than in Dennler's protein-layer test,
because the fold is reading directly off the equivalence structure.

**Panel 3: Channel-stratified elbow position.**
Elbow of the handoff curve per channel, plotted against channel
essentiality (DepMap mean dependency score). Projection theory predicts
channels with stronger essentiality have longer projection memory — the
cell holds them coupled longer. Tests whether the fingerprint is
modulated by selection pressure. Connects to the cancer work without
requiring it to be accepted.

**Panel 4: Cell-type-specific context-conditioned recovery.**
For paralog pairs that are tissue-specifically co-essential (from DepMap),
are they tissue-specifically in 3D contact in the matching tissues? Hi-C
from GM12878, IMR90, K562, HMEC, HUVEC, etc. intersected with DepMap
per-cell-line essentiality. If yes, the fold is context-conditioned
recovery — same 2D encoding, different 3D readout per cellular context.
If no, the fold is static and the "finite context" framing has to retreat.
This is the sharpest possible test of the information-theoretic claim that
context supplies the information the 2D encoding cannot.

---

## Discriminating tests (what descent cannot predict)

1. **Rising 3D contact compensating for falling 2D proximity.** Descent
   predicts 2D decay with divergence. Descent has no mechanism for a rise
   in 3D contact as compensation. Projection theory predicts the rise
   because it says the cell defends a dimension-agnostic proximity
   invariant.

2. **Drift-out cohort loses 3D contact.** Take ancient paralogs and split
   by current equivalence class status. Stayed-in-class pairs should keep
   3D contact. Drifted-out pairs should lose it. Pure descent predicts
   both groups touch equally because they are both old duplicates.

3. **Cell-type-specific contact tracks cell-type-specific co-essentiality.**
   Descent predicts contact is a fixed property of the genome. Projection
   theory predicts contact varies with context because the fold is
   context-conditioned recovery.

4. **The elbow position is under selection.** Channels with higher
   essentiality should have later elbows (longer projection memory).
   Descent has no framework in which "elbow position" is a meaningful
   quantity at all.

---

## The mathematical framing: projections of higher-dimensional objects

The method is standard. Studying an N-dimensional object through
(N−1)-dimensional projections is 150 years of differential topology and
algebraic geometry. You take a 4-sphere, you project to 3-space, you get a
3-sphere. Slice with a plane, you get a circle. Slice with a line, you get
two points. No mathematician pretends the 4-sphere is not real because
they can only ever see 3-spheres. They take projections from many angles
and reconstruct the object from the agreement between projections.

Biology never adopted this epistemology because it inherited from taxonomy.
Projection theory restores the geometric epistemology to the genome layer.

---

## The cave allegory — and why it is literal, not metaphorical

Plato's cave is dimensional reduction with observers trapped on the
lower-dimensional surface. That is the exact mathematical situation. The
prisoners see shadows and argue about relationships between shadows
without ever asking whether two shadows might be the same object viewed
from different angles. That is what the field has been doing with DNA for
seventy years.

DNA is the cave wall. Genes are the shadows. The functional node is the
3D object casting them. The upgrade is not "leaving the cave." It is
**refusing to look at only one wall.** For seventy years the field has
stared at the sequence wall and built elaborate genealogies of which
shadow descended from which. The projection frame says: turn your head.
There is a second wall — the Hi-C contact map — and the shadows on wall
two are the shadows wall one predicts if and only if a single object is
casting both.

One paper. Two walls. Shadow agreement.

---

## Humility: triangulation, not reconstruction

We will never directly see the 3D functional object. Any collection of 2D
projections is strictly less dense than the 3D thing that cast them. Our
paper cannot and should not claim to have reconstructed the object.

It claims something smaller and cleaner: **two independent projections
agree in a way that descent alone cannot explain.** That is triangulation,
not direct observation. The object remains inferred. The limit of
inference is exactly the limit the information theory sets: finite
projections on a finite context can only ever bound the set of candidate
objects, not identify a unique one.

That is fine. That is what science does. The paper's contribution is
showing that the bound is tighter than descent-only models allow.

---

## One-paragraph abstract

> Biological objects — folded proteins, chromatin complexes, assembled
> machines — exist in 3D space. DNA is a 2D encoding of these objects
> (linear sequence × helical phase), and the nuclear fold is the
> decompression operation that recovers 3D execution from 2D storage,
> adding exactly one dimension, the minimum possible. Under this frame,
> paralogs encoding the same 3D node should be close in the 2D encoding
> when they have not yet drifted (nearly identical paralogs, tandem
> duplicates) and close in the 3D fold when they have drifted but still
> encode the same object (paralogs scattered across chromosomes that
> remain in functional and spatial contact). We test this using public
> Ensembl paralog data and published Hi-C contact maps, and show that the
> handoff from 2D-proximity to 3D-proximity with evolutionary distance is
> the signature of a projection being recovered. The fit of this signature
> to the data tightens as equivalence-class features (Dennler & Ryan 2025)
> replace sequence identity alone, and strengthens further when Hi-C is
> stratified by cell type, consistent with context-conditioned recovery of
> a 3D object from a 2D encoding with finite cellular context supplying
> the missing dimension.

---

## Data anchor (already in the main graph)

`data/channel_gene_map.csv` already places BRCA1, BRCA2, PALB2, RAD51B,
RAD51C, RAD51D all on the DDR channel. Six "different" genes, one node by
projection theory.

`knowledge-graph/nodes/empirical/EMP04-sl-same-channel-protective.yaml`
already shows SL + same-channel co-mutation gives HR 0.626 versus 0.85
for SL alone — the survival signature of collapsing to one node. The
projection theory's *prediction* for that HR pattern has been sitting in
the repo waiting for this framing.

RAD51C + AZD5305 response in Tricia's case is the in-vivo validation.
This paper is the spatial validation of the same invariant.

---

## Pipeline scaffolded (2026-04-10)

- `README.md` — the test and the dimensional framing
- `fetch_paralogs.py` — Ensembl BioMart human paralog pull. Columns:
  gene coordinates, paralog coordinates, perc_id, perc_id_reciprocal,
  orthology_type, subtype. Ensembl removed dN/dS from BioMart; we use
  `subtype` (named divergence ladder) and `perc_id` (continuous) instead.
- `fetch_coessentiality.py` — DepMap scaffold, URL-parameterized, not run.
- `fetch_hic.py` — Rao 2014 GSE63525 loop calls (GM12878, IMR90, K562).
- `build_pair_table.py` — join scaffold, logic in comments, not executed.

First fetch run on 2026-04-10 produced `data/paralogs.tsv` (~380 MB).
Verify row count and column shape before building the pair table.
