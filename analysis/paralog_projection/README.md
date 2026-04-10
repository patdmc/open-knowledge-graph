# Paralog Projection Test

Does functional equivalence between genes predict their 3D proximity in the
nucleus above a sequence-distance and chromosome-matched null?

## The claim

Biological objects — folded proteins, chromatin complexes, assembled
machines — are 3D. DNA is a 2D serialization of those 3D objects: linear
sequence position along the tape, helical phase selecting which reader
gets access. The fold is a decompression operation that recovers 3D
execution from 2D storage, adding exactly one dimension (the minimum
possible).

Round trip: 3 → 2 → 3. Store in 2D, execute in 3D.

Under this framing, paralogs that encode the same 3D object should be
close in the 2D encoding when they have not yet drifted (nearly identical
paralogs, tandem duplicates) and close in the 3D fold when they have
drifted but still encode the same object (paralogs scattered across
chromosomes that remain in functional and spatial contact). The handoff
from 2D-proximity to 3D-proximity with evolutionary distance is the
signature of a projection being recovered.

Note: this first-pass analysis collapses the helical phase axis and uses
only linear distance as the 2D-side measurement. Phase as a covariate
(nucleosome positioning, groove preferences) is a v2 extension.

The test inverts the usual paralog question. Instead of asking "do paralogs
cluster in 3D" (predictor = sequence, outcome = spatial) we ask "does the
equivalence class a gene belongs to predict its 3D neighbors" (predictor =
functional equivalence, outcome = spatial). The flip catches:

1. Non-paralog pairs that collapsed to one functional node (convergent)
2. Paralog pairs that drifted apart and became real independent nodes (the
   discriminating negative cases — pure descent predicts they still touch,
   projection theory predicts they do not)

## Prior art we stand on

Dennler & Ryan 2025 (NAR Genomics and Bioinformatics) already showed that
sequence identity alone is insufficient to predict shared paralog function.
Structural similarity (AlphaFold TM, LDDT) and PLM embeddings (ProtT5, ESM2)
add 0.05–0.15 AUROC over sequence identity across shared PPIs, synthetic
lethality, and GO semantic similarity. They stopped at the protein layer.
We extend to the nuclear layer.

## The single-figure paper

Three panels:

1. Dual-proximity handoff. Linear chromosomal distance and 3D Hi-C contact
   on the same Ks axis. The prediction: proximity migrates from the linear
   dimension to the fold as paralogs age. Recent paralogs (NIPs, ≥98% identity)
   are adjacent on chromosome and contact in 3D is incidental. Middle Ks:
   linear proximity decays, 3D contact rises to compensate — the fold taking
   over as the projection breaks down. High Ks stayed-in-class: 3D dominates,
   linear is gone (BRCA/RAD51 family anchor). High Ks drifted-out: both
   decay (olfactory receptor anchor). The two curves crossing around the
   decoherence point is the fingerprint of projection recovery — a
   prediction descent cannot make, because descent has no reason to predict
   a rise in 3D contact to compensate for lost linear proximity.

2. AUROC for predicting 3D contact from (a) sequence identity alone, (b)
   Dennler-style combined features, (c) co-essentiality, (d) shared channel
   membership. Direct extension of Dennler Figure 6 with Hi-C contact as
   the dependent variable.

3. Elbow position per channel vs channel essentiality (DepMap mean dependency).
   Tests whether channels with stronger essentiality have longer projection
   memory — connects to the cancer work without requiring it to be accepted.

## Discriminating test

Take ancient paralogs (high Ks), split by current equivalence class status:

- Stayed in same class → projection theory predicts still in 3D contact
- Drifted to different classes → projection theory predicts no contact above null
- Pure descent predicts both groups touch equally

If the drifted-out group loses 3D proximity and the stayed-in group keeps it,
the genome is reorganizing the fold to track functional equivalence in real
time. This is the prediction descent literally cannot make.

## Data sources (step 1)

- Ensembl Compara human paralogs with Ks (BioMart)
- DepMap CRISPR gene effect matrix → co-essentiality
- 4DN / Rao 2014 GM12878 Hi-C contact matrix
- data/channel_gene_map.csv (local, for channel panel)
- Singh/Makino ohnolog lists (2R whole-genome duplicates)

## Anchor case (already in graph)

BRCA1, BRCA2, PALB2, RAD51B, RAD51C, RAD51D all on DDR channel per
`data/channel_gene_map.csv`. EMP04-sl-same-channel-protective shows SL +
same-channel co-mutation gives HR 0.626 vs 0.85 for SL alone — the survival
signature of collapsing to one node. RAD51C + AZD5305 response is the in-vivo
confirmation. This paper provides the spatial confirmation.
