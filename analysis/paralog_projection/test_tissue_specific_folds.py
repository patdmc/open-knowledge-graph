"""Test tissue-specific fold prediction across 21 tissues (Schmitt 2016).

For each channel's gene pairs, extract contact values from each tissue's
40kb HiCNorm matrix and compute tissue-specific enrichment. Tests:

1. Do channel gene contacts vary by tissue in the predicted direction?
2. Are ChromatinRemodel/DNAMethylation contacts constitutive (anti-index)?
3. Do tissues cluster by shared channel usage rather than developmental lineage?
4. How many distinct chromatin states exist (for the book thickness test)?

Usage:
    python test_tissue_specific_folds.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


SCHMITT_DIR = Path(__file__).parent / "data/schmitt2016/data/schmitt2016/contacts/contact_maps/HiCNorm/primary_cohort"
RESOLUTION = 40_000  # 40kb bins

# Map tissue codes to readable names and expected dominant channels
TISSUE_INFO = {
    "AD":      ("Adrenal gland",     "Endocrine"),
    "AO":      ("Aorta",            "TissueArchitecture"),
    "BL":      ("Bladder",          "TissueArchitecture"),
    "CO":      ("Cortex",           "ChromatinRemodel"),
    "GM12878": ("Lymphoblastoid",   "Immune"),
    "h1":      ("H1 ESC",          "CellCycle"),
    "HC":      ("Hippocampus",      "ChromatinRemodel"),
    "IMR90":   ("Lung fibroblast",  "TissueArchitecture"),
    "LG1":     ("Lung (rep1)",      "TissueArchitecture"),
    "LI":      ("Liver",           "PI3K_Growth"),
    "LV":      ("Left ventricle",   "TissueArchitecture"),
    "mes":     ("Mesendoderm",      "CellCycle"),
    "msc":     ("Mesenchymal SC",   "TissueArchitecture"),
    "npc":     ("Neural progenitor","ChromatinRemodel"),
    "OV":      ("Ovary",           "Endocrine"),
    "PA2":     ("Pancreas (rep1)",  "PI3K_Growth"),
    "PO1":     ("Psoas muscle",     "TissueArchitecture"),
    "RV":      ("Right ventricle",  "TissueArchitecture"),
    "SB2":     ("Small bowel",      "TissueArchitecture"),
    "SX1":     ("Spleen (rep1)",    "Immune"),
    "tro":     ("Trophoblast",      "Endocrine"),
}

# Use one replicate per tissue to avoid double-counting
TISSUES = list(TISSUE_INFO.keys())


def load_gene_coords():
    """Load gene coordinates from the pair table."""
    df = pd.read_parquet(Path(__file__).parent / "data/pair_table_gm12878_primary_heavy.parquet",
                         columns=["gene_a", "chrom_a", "start_a", "end_a",
                                  "gene_b", "chrom_b", "start_b", "end_b"])
    ga = df[["gene_a", "chrom_a", "start_a", "end_a"]].rename(
        columns={"gene_a": "gene", "chrom_a": "chrom", "start_a": "start", "end_a": "end"})
    gb = df[["gene_b", "chrom_b", "start_b", "end_b"]].rename(
        columns={"gene_b": "gene", "chrom_b": "chrom", "start_b": "start", "end_b": "end"})
    genes = pd.concat([ga, gb]).drop_duplicates("gene")
    genes["mid"] = ((genes.start + genes.end) / 2).astype(int)
    genes["bin"] = genes["mid"] // RESOLUTION
    genes["chrom"] = genes["chrom"].astype(str)
    return genes.set_index("gene")


def load_contact_matrix(tissue, chrom):
    """Load a single tissue x chromosome HiCNorm matrix."""
    # Try with and without 'chr' prefix
    for chr_name in [f"chr{chrom}", chrom]:
        path = SCHMITT_DIR / f"{tissue}.nor.{chr_name}.mat"
        if path.exists():
            mat = np.loadtxt(path, delimiter="\t")
            return mat
    # Try replicate naming
    for suffix in [".rep1", ".rep2", ""]:
        for chr_name in [f"chr{chrom}", chrom]:
            path = SCHMITT_DIR / f"{tissue}{suffix}.nor.{chr_name}.mat"
            if path.exists():
                mat = np.loadtxt(path, delimiter="\t")
                return mat
    return None


def get_contact(mat, bin_i, bin_j):
    """Get contact value from matrix, handling bounds."""
    if mat is None:
        return np.nan
    n = mat.shape[0]
    if bin_i >= n or bin_j >= n or bin_i < 0 or bin_j < 0:
        return np.nan
    val = mat[bin_i, bin_j]
    return val if val > 0 else np.nan


def main():
    print("[fold] Loading gene coordinates...", file=sys.stderr)
    genes = load_gene_coords()

    # Load channel gene map
    cgm = pd.read_csv(Path(__file__).parent / "../../data/channel_gene_map.csv")
    channel_map = dict(zip(cgm.gene, cgm.channel))

    # Load co-essentiality clusters
    clusters = pd.read_csv(Path(__file__).parent / "data/coess_clusters_k200.csv", index_col=0)
    clusters = clusters.iloc[:, 0].to_dict()

    # Build gene pairs to test:
    # 1. Same-channel pairs (same chrom only — we need matrix contact values)
    # 2. Same-cluster pairs (same chrom)
    # 3. Random baseline pairs (same chrom, not same channel or cluster)
    print("[fold] Building gene pair lists...", file=sys.stderr)

    channel_pairs = []
    for channel in cgm.channel.unique():
        ch_genes = cgm[cgm.channel == channel].gene.tolist()
        ch_genes = [g for g in ch_genes if g in genes.index]
        for i, g1 in enumerate(ch_genes):
            for g2 in ch_genes[i+1:]:
                if genes.loc[g1, "chrom"] == genes.loc[g2, "chrom"]:
                    channel_pairs.append((g1, g2, channel))

    print(f"[fold] Same-channel, same-chrom pairs: {len(channel_pairs)}", file=sys.stderr)

    # Also build same-cluster pairs for the broader test
    cluster_pairs = []
    all_clustered = [g for g in genes.index if g in clusters]
    by_cluster_chrom = {}
    for g in all_clustered:
        key = (clusters[g], genes.loc[g, "chrom"])
        by_cluster_chrom.setdefault(key, []).append(g)

    for (cid, chrom), gs in by_cluster_chrom.items():
        if len(gs) >= 2:
            for i, g1 in enumerate(gs):
                for g2 in gs[i+1:]:
                    cluster_pairs.append((g1, g2, cid))

    print(f"[fold] Same-cluster, same-chrom pairs: {len(cluster_pairs)}", file=sys.stderr)

    # For each tissue, extract contact values for channel pairs
    print("[fold] Extracting contacts across tissues...", file=sys.stderr)

    # Cache loaded matrices
    mat_cache = {}

    def get_tissue_contact(tissue, g1, g2):
        chrom = genes.loc[g1, "chrom"]
        key = (tissue, chrom)
        if key not in mat_cache:
            mat_cache[key] = load_contact_matrix(tissue, chrom)
        mat = mat_cache[key]
        b1 = int(genes.loc[g1, "bin"])
        b2 = int(genes.loc[g2, "bin"])
        return get_contact(mat, b1, b2)

    # Build channel pair x tissue contact matrix
    pair_labels = [(g1, g2, ch) for g1, g2, ch in channel_pairs]
    tissue_contacts = np.full((len(pair_labels), len(TISSUES)), np.nan)

    for j, tissue in enumerate(TISSUES):
        print(f"  {tissue}...", end=" ", file=sys.stderr, flush=True)
        for i, (g1, g2, ch) in enumerate(pair_labels):
            if g1 in genes.index and g2 in genes.index:
                tissue_contacts[i, j] = get_tissue_contact(tissue, g1, g2)
        mat_cache.clear()  # Free memory between tissues
    print("", file=sys.stderr)

    # Build DataFrame
    pair_df = pd.DataFrame(tissue_contacts, columns=TISSUES)
    pair_df["gene_a"] = [p[0] for p in pair_labels]
    pair_df["gene_b"] = [p[1] for p in pair_labels]
    pair_df["channel"] = [p[2] for p in pair_labels]

    # === ANALYSIS 1: Per-channel contact enrichment across tissues ===
    print("\n=== PER-CHANNEL CONTACT ACROSS TISSUES ===\n")

    # Compute per-tissue baseline (median contact across ALL channel pairs)
    baselines = pair_df[TISSUES].median()

    for channel in sorted(cgm.channel.unique()):
        ch_rows = pair_df[pair_df.channel == channel]
        if len(ch_rows) == 0:
            continue
        medians = ch_rows[TISSUES].median()
        enrichments = medians / baselines
        enrichments = enrichments.replace([np.inf, -np.inf], np.nan)

        # Find top 3 tissues for this channel
        top3 = enrichments.dropna().nlargest(3)
        top_str = ", ".join([f"{t}({TISSUE_INFO[t][0]})={enrichments[t]:.2f}" for t in top3.index])
        predicted = None
        for t in TISSUES:
            if TISSUE_INFO[t][1] == channel:
                predicted = t
                break
        pred_str = f"[predicted: {TISSUE_INFO[predicted][0]}]" if predicted else ""

        print(f"{channel:>22s} (n={len(ch_rows)} pairs): top={top_str} {pred_str}")

    # === ANALYSIS 2: Tissue clustering by channel contact profiles ===
    print("\n=== TISSUE CLUSTERING BY CHANNEL CONTACT PROFILE ===\n")

    # Build channel x tissue enrichment matrix
    channels = sorted(cgm.channel.unique())
    enrich_matrix = np.full((len(channels), len(TISSUES)), np.nan)
    for ci, channel in enumerate(channels):
        ch_rows = pair_df[pair_df.channel == channel]
        if len(ch_rows) == 0:
            continue
        medians = ch_rows[TISSUES].median()
        enrich_matrix[ci, :] = (medians / baselines).values

    enrich_df = pd.DataFrame(enrich_matrix, index=channels, columns=TISSUES)
    print("Channel x Tissue enrichment matrix:")
    print(enrich_df.round(2).to_string())

    # === ANALYSIS 3: Constitutive vs tissue-specific contacts ===
    print("\n\n=== CONSTITUTIVE vs TISSUE-SPECIFIC ===\n")

    for channel in channels:
        ch_rows = pair_df[pair_df.channel == channel]
        if len(ch_rows) == 0:
            continue
        # Coefficient of variation across tissues
        medians = ch_rows[TISSUES].median()
        valid = medians.dropna()
        if len(valid) > 2:
            cv = valid.std() / valid.mean() if valid.mean() > 0 else np.nan
            n_nonzero = (valid > 0).sum()
            print(f"{channel:>22s}: CV={cv:.3f}, nonzero in {n_nonzero}/{len(valid)} tissues, "
                  f"{'CONSTITUTIVE' if cv < 0.3 else 'TISSUE-SPECIFIC'}")

    # === ANALYSIS 4: Book thickness — distinct chromatin states ===
    print("\n=== BOOK THICKNESS: DISTINCT CHROMATIN STATES ===\n")

    # For the broader test, compute per-cluster contact profiles across tissues
    # Use same-cluster pairs
    print(f"Computing cluster contact profiles across {len(TISSUES)} tissues...", file=sys.stderr)

    # Sample up to 500 clusters for tractability
    cluster_ids = sorted(set(clusters.values()))
    cluster_contacts = {}
    for cid in cluster_ids:
        cid_pairs = [(g1, g2) for g1, g2, c in cluster_pairs if c == cid]
        if len(cid_pairs) < 2:
            continue
        tissue_vals = []
        for tissue in TISSUES:
            vals = []
            for g1, g2 in cid_pairs[:20]:  # Cap at 20 pairs per cluster for speed
                if g1 in genes.index and g2 in genes.index:
                    v = get_tissue_contact(tissue, g1, g2)
                    if not np.isnan(v):
                        vals.append(v)
            tissue_vals.append(np.median(vals) if vals else 0)
            mat_cache.clear()
        cluster_contacts[cid] = tissue_vals

    # Build cluster x tissue matrix
    cc_df = pd.DataFrame(cluster_contacts, index=TISSUES).T
    cc_valid = cc_df.dropna(how="all").fillna(0)
    print(f"Clusters with contact data: {len(cc_valid)}")

    # Cluster the TISSUES by their contact profiles
    if len(cc_valid) > 5:
        tissue_corr = cc_valid.corr(method="spearman")
        tissue_dist = 1 - tissue_corr.values
        np.fill_diagonal(tissue_dist, 0)
        tissue_dist = np.clip((tissue_dist + tissue_dist.T) / 2, 0, 2)

        # How many distinct states?
        condensed = pdist(cc_valid.T.values, metric="correlation")
        Z = linkage(condensed, method="ward")

        print("\nDistinct chromatin states by cluster threshold:")
        for t in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            labels = fcluster(Z, t=t, criterion="distance")
            n_states = len(set(labels))
            print(f"  threshold={t:.1f}: {n_states} distinct states")

        # Save dendrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        tissue_names = [f"{t} ({TISSUE_INFO[t][0]})" for t in TISSUES]
        dendrogram(Z, labels=tissue_names, ax=ax, leaf_rotation=45, leaf_font_size=8)
        ax.set_title("Tissue clustering by co-essentiality cluster contact profiles")
        ax.set_ylabel("Ward distance")
        fig.tight_layout()
        out = Path(__file__).parent / "data/figures/tissue_dendrogram_schmitt.png"
        fig.savefig(out, dpi=150)
        fig.savefig(out.with_suffix(".pdf"))
        print(f"\nWrote {out}", file=sys.stderr)

    # Save the enrichment matrix
    enrich_out = Path(__file__).parent / "data/figures/channel_tissue_enrichment.tsv"
    enrich_df.to_csv(enrich_out, sep="\t")
    print(f"Wrote {enrich_out}", file=sys.stderr)

    # Save the pair-level data
    pair_out = Path(__file__).parent / "data/channel_pairs_21tissues.tsv"
    pair_df.to_csv(pair_out, sep="\t", index=False)
    print(f"Wrote {pair_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
