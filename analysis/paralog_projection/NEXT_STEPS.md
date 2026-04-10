# Next Steps — Reproducible Setup

This document is written for someone starting from a clean machine with
nothing but a git client and Python 3.11+. Nothing on disk is assumed.
No state is copied from another machine. Every command is meant to be
copy-paste runnable. URLs are pinned to specific releases where the
underlying data evolves over time.

If a command in this document does not work as written, that is a bug
in this document, not in the machine. Update this document when you fix
the bug so the next clean run benefits.

---

## 0. Prerequisites

You need:

- **git** (any modern version)
- **Python 3.11 or newer**
- About **150 GB of free disk** for full Hi-C plus AlphaFold downloads
  (the minimum for Panel 1 alone is about 5 GB)
- **A modern NVIDIA GPU with at least 16 GB VRAM** (only required for
  the optional Dennler-style PLM feature recomputation in Panel 2; not
  needed for Panels 1, 3, or 4)
- An internet connection that can sustain large downloads (Hi-C contact
  matrices and AlphaFold structures are tens of GB)

Verify Python:

```bash
python3 --version
# Expect: Python 3.11.x or newer
```

---

## 1. Clone the repo

```bash
git clone git@github.com:<owner>/open-knowledge-graph.git
cd open-knowledge-graph
```

(If you do not have ssh keys configured, use the https clone URL instead.
Replace `<owner>` with the actual GitHub owner — the canonical location is
in the repo metadata, not hardcoded here, because it has moved before and
may move again.)

---

## 2. Create a virtual environment and install Python dependencies

```bash
cd analysis/paralog_projection
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the CPU-only stack needed for fetches, joins, and Panels 1,
3, and 4. The GPU stack for Panel 2's PLM feature recomputation is a
separate file (`requirements-gpu.txt`) installed only if you plan to run
that path.

To verify the install:

```bash
python3 -c "import pandas, numpy, scipy, cooler, sklearn, matplotlib; print('ok')"
# Expect: ok
```

---

## 3. Pre-stage all input data (one-time downloads)

All data lives under `analysis/paralog_projection/data/` which is
gitignored. Nothing in this directory is checked in. Every file here is
regenerable from a script in this directory.

### 3a. Ensembl human paralog table

Pinned to Ensembl release 113 (Nov 2024). The fetch script defaults to
the production endpoint, which always points at the current release; pass
`--mart-host` to pin a specific archive.

```bash
python3 fetch_paralogs.py \
  --mart-host http://nov2024.archive.ensembl.org \
  --out data/paralogs.tsv
```

Expected output:
- `data/paralogs.tsv`
- About 380 MB
- About 3.25 million rows including header
- Header columns (in order):
  ```
  Gene stable ID
  Gene name
  Chromosome/scaffold name
  Gene start (bp)
  Gene end (bp)
  Human paralogue gene stable ID
  Human paralogue associated gene name
  Human paralogue chromosome/scaffold name
  Human paralogue chromosome/scaffold start (bp)
  Human paralogue chromosome/scaffold end (bp)
  Paralogue %id. target Human gene identical to query gene
  Paralogue %id. query gene identical to target Human gene
  Human paralogue homology type
  Paralogue last common ancestor with Human
  ```

Verify the anchor case (RAD51C should have 6 paralogs across chromosomes
7, 14, 14, 15, 17, 22 with 15-26% identity, all Opisthokonta):

```bash
awk -F'\t' '$2=="RAD51C" && $7!=""' data/paralogs.tsv
# Expect 6 lines: paralogs to DMC1, RAD51, RAD51B, RAD51D, XRCC2, XRCC3
```

If the row count or anchor case does not match, do not proceed. The
schema or release has drifted and the rest of the pipeline will silently
break. Update `fetch_paralogs.py` and this document together.

### 3b. Hi-C loop calls (light path, ~1 MB)

Rao 2014 GSE63525 HiCCUPS loop calls for GM12878 primary+replicate.
These are high-confidence contact pairs already thresholded by the
authors. The light path is enough to make a binarized Panel 1 figure on
a laptop in minutes.

```bash
python3 fetch_hic.py --cell-line gm12878_primary --out data/gm12878_loops.txt.gz
python3 fetch_hic.py --cell-line imr90 --out data/imr90_loops.txt.gz
python3 fetch_hic.py --cell-line k562 --out data/k562_loops.txt.gz
```

Expected output: three gzipped text files, each well under 1 MB. The
file format is HiCCUPS looplist (tab-separated, first row is header,
columns include `chr1 x1 x2 chr2 y1 y2 ...`). See the GSE63525
supplementary README on GEO for full column definitions.

### 3c. Hi-C contact matrices (heavy path, tens of GB per cell line)

Required for the continuous-contact-frequency Panel 1 and for Panel 4.
Skip on a laptop. Run on a workstation.

The Rao 2014 .hic files are at:

```
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_combined.hic
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_IMR90_combined.hic
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined.hic
```

Pull with curl (each is roughly 4-12 GB):

```bash
mkdir -p data/hic
curl -L -o data/hic/GSE63525_GM12878_combined.hic \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_combined.hic
curl -L -o data/hic/GSE63525_IMR90_combined.hic \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_IMR90_combined.hic
curl -L -o data/hic/GSE63525_K562_combined.hic \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined.hic
```

Verify file sizes against the GEO directory listing before proceeding —
partial downloads will silently corrupt the analysis. Compare the
expected sizes to:

```
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/
```

Read with `hicstraw` (Python wrapper for the Juicer straw library, in
requirements.txt). Example query:

```python
import hicstraw
hic = hicstraw.HiCFile("data/hic/GSE63525_GM12878_combined.hic")
matrix = hic.getMatrixZoomData("17", "17", "observed", "KR", "BP", 5000)
records = matrix.getRecords(43044292, 43170245, 43044292, 43170245)
# BRCA1 self-contact at 5kb resolution, KR-normalized
```

### 3d. DepMap CRISPR gene effect matrix

Required for Panels 2 and 4. The DepMap release URL changes per release
and is not stable across releases. To pin reproducibly:

1. Go to https://depmap.org/portal/download/all/
2. Pick a release (e.g. 24Q4) and find the file named `CRISPRGeneEffect.csv`
3. Copy the download URL for that file
4. Pass it to the fetch script:

```bash
python3 fetch_coessentiality.py \
  --release 24Q4 \
  --url <PASTE-DEPMAP-URL-HERE> \
  --out data/depmap_gene_effect_24Q4.csv
```

Record the release version and the URL in this section of the document
(or in a separate `data/MANIFEST.md`) the first time you successfully
pull a release, so that the next reproduction can use the exact same URL.

Expected file: about 150 MB CSV. Rows are cell lines, columns are genes.
Cell values are Chronos-scaled gene effect scores.

### 3e. AlphaFold human proteome (optional, only for GPU/Panel 2 enrichment)

Required only if you plan to recompute Dennler & Ryan structural features
(TM scores, LDDT) ourselves. Skip otherwise.

```bash
mkdir -p data/alphafold
curl -L -o data/alphafold/UP000005640_9606_HUMAN_v4.tar \
  https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar
cd data/alphafold && tar -xf UP000005640_9606_HUMAN_v4.tar && cd ../..
```

Roughly 30 GB compressed, more uncompressed. Per-protein PDB files for
the entire reviewed human proteome. Pinned to AlphaFold v4 deliberately —
v3 had different confidence calibration and structures are not directly
comparable.

### 3f. Channel-gene map (already in repo)

This is the only input that is not regenerated from a fetch script
because it lives in the main repo data directory. Reference it directly:

```
../../data/channel_gene_map.csv
```

The current schema is:
```
gene,channel
BRCA1,DDR
BRCA2,DDR
PALB2,DDR
RAD51B,DDR
RAD51C,DDR
RAD51D,DDR
...
```

If this file moves or its schema changes, update `build_pair_table.py`
to match and update this section.

---

## 4. Build the joined pair table

Run after all required inputs in step 3 are present.

```bash
python3 build_pair_table.py \
  --paralogs data/paralogs.tsv \
  --channels ../../data/channel_gene_map.csv \
  --hic-loops data/gm12878_loops.txt.gz \
  --coess data/depmap_gene_effect_24Q4.csv \
  --out data/pair_table.parquet
```

The `--coess` flag is optional. Without it, you get the pair table
without co-essentiality features, which is enough for Panel 1.

Expected output: `data/pair_table.parquet`, columns documented in the
header of `build_pair_table.py`. Roughly 3 million rows (one per paralog
pair), with optional rows-with-loops being a much smaller subset.

---

## 5. Generate the panels

Panel scripts are not yet written. They will live as:

```
panel1_dual_proximity.py
panel2_equivalence_auroc.py
panel3_channel_elbows.py
panel4_tissue_specific_recovery.py
```

Each script takes `--pair-table data/pair_table.parquet` as input and
writes a PDF and a PNG to `data/figures/`. Each script is meant to be
runnable in isolation and should print summary statistics to stdout
sufficient to verify the figure matches expectations without opening it.

**Important:** run Panel 1 first and look at the result before running
the other three. Panel 1 is the gate. If Panel 1 does not show the
2D-to-3D proximity handoff that projection theory predicts, the rest of
the panels are not informative and the framing has to retreat.

---

## 6. Decision points after Panel 1

The first Panel 1 run will yield one of three outcomes.

**Outcome A: clean handoff.**
2D proximity decays with divergence. 3D contact rises to compensate,
peaks for the middle-Ks-but-stayed-in-class cohort, then both fall for
drifted-out pairs. The fingerprint is real. Proceed to Panels 2-4 and
write the paper as drafted in `NOTES.md`.

**Outcome B: parallel decay.**
3D contact tracks 2D proximity and never compensates. The fold is not
doing the recovery work projection theory predicts. The framing has to
retreat — possibly to "the recovery happens at the action-graph layer
(enhancer loops), not the contact-map layer." This becomes a v2 paper
with a different test. Stop and rethink before running Panels 2-4.

**Outcome C: channel-dependent crossing.**
The curves cross for some channels and not others. This is actually the
most interesting outcome because it predicts Panel 3 directly. Channels
under stronger selection should show the crossing; channels under weaker
selection should not. Run Panel 3 immediately to test the channel-
stratified prediction. If Panel 3 confirms it, the paper is *stronger*
than the original framing because the variation across channels is the
prediction.

---

## 7. Reproducing a published figure from scratch

Once panels exist and a figure is published, anyone should be able to
reproduce a figure by following these steps:

```bash
git clone <repo>
cd open-knowledge-graph/analysis/paralog_projection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Steps 3a, 3b, 3d (omit 3c, 3e for the light path)
python3 fetch_paralogs.py --mart-host http://nov2024.archive.ensembl.org --out data/paralogs.tsv
python3 fetch_hic.py --cell-line gm12878_primary --out data/gm12878_loops.txt.gz
python3 fetch_coessentiality.py --release 24Q4 --url <pinned-url> --out data/depmap_gene_effect_24Q4.csv
python3 build_pair_table.py --out data/pair_table.parquet
python3 panel1_dual_proximity.py --pair-table data/pair_table.parquet
```

If a published figure cannot be reproduced from these steps starting from
a clean clone, that is a bug in the repo, not in the reader's machine.
Open an issue.

---

## 8. What to commit and what not to commit

**Never commit:**
- Anything in `data/`. It is gitignored. Use the fetch scripts to
  recreate.
- The `.venv/` directory.
- Per-machine configuration (hostnames, paths, environment variables).

**Always commit:**
- Updates to fetch scripts when source schemas drift.
- Updates to this document when commands stop working.
- New panel scripts.
- New requirement pins (in `requirements.txt`).
- Anchor-case verifications when the data has changed enough that the
  numbers in this document need to be updated.

**Sometimes commit (with thought):**
- Small derived feature tables that are very expensive to recompute and
  unlikely to change. Put them in a `derived/` subdirectory and add an
  exception to the gitignore. Document the version of the input data
  they were derived from.

---

## 9. Why this document exists

Reproducibility is not optional. A paper that cannot be regenerated from
its source is a paper whose conclusions cannot be independently checked.
Every step in this document exists so that the analysis can be re-run by
a clean machine, by a future collaborator, by a reviewer, by an LLM
agent, or by a future version of you who has forgotten the details.

If you are tempted to skip a step because "it works on my machine," do
not. Update this document so that the missing step is captured and the
next clean reproduction works.
