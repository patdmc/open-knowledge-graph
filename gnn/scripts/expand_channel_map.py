"""
Expand the 99-gene CHANNEL_MAP to cover all ~410 unmapped MSK-IMPACT genes.

Primary signal:  MSigDB Hallmark gene-set membership -> channel mapping.
Secondary signal: Jaccard co-mutation with mapped genes (tiebreaker / fallback).

Usage:
    python3 -u -m gnn.scripts.expand_channel_map
"""

import json
import math
import os
from collections import Counter, defaultdict

import pandas as pd

from gnn.config import (
    CHANNEL_MAP,
    CHANNEL_NAMES,
    GNN_RESULTS,
    MSK_DATASETS,
)

# ── Hallmark set -> channel mapping ─────────────────────────────────────────

HALLMARK_TO_CHANNEL = {
    # PI3K_Growth
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING": "PI3K_Growth",
    "HALLMARK_MTORC1_SIGNALING": "PI3K_Growth",
    "HALLMARK_KRAS_SIGNALING_UP": "PI3K_Growth",
    "HALLMARK_KRAS_SIGNALING_DN": "PI3K_Growth",
    "HALLMARK_HEDGEHOG_SIGNALING": "PI3K_Growth",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING": "PI3K_Growth",
    "HALLMARK_NOTCH_SIGNALING": "PI3K_Growth",
    "HALLMARK_TGF_BETA_SIGNALING": "PI3K_Growth",
    # CellCycle
    "HALLMARK_E2F_TARGETS": "CellCycle",
    "HALLMARK_G2M_CHECKPOINT": "CellCycle",
    "HALLMARK_MITOTIC_SPINDLE": "CellCycle",
    "HALLMARK_MYC_TARGETS_V1": "CellCycle",
    "HALLMARK_MYC_TARGETS_V2": "CellCycle",
    "HALLMARK_P53_PATHWAY": "CellCycle",
    # DDR
    "HALLMARK_DNA_REPAIR": "DDR",
    "HALLMARK_UV_RESPONSE_UP": "DDR",
    "HALLMARK_UV_RESPONSE_DN": "DDR",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": "DDR",
    # Endocrine
    "HALLMARK_ESTROGEN_RESPONSE_EARLY": "Endocrine",
    "HALLMARK_ESTROGEN_RESPONSE_LATE": "Endocrine",
    "HALLMARK_ANDROGEN_RESPONSE": "Endocrine",
    "HALLMARK_BILE_ACID_METABOLISM": "Endocrine",
    "HALLMARK_CHOLESTEROL_HOMEOSTASIS": "Endocrine",
    # Immune
    "HALLMARK_INFLAMMATORY_RESPONSE": "Immune",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "Immune",
    "HALLMARK_IL2_STAT5_SIGNALING": "Immune",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": "Immune",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE": "Immune",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": "Immune",
    "HALLMARK_COMPLEMENT": "Immune",
    "HALLMARK_ALLOGRAFT_REJECTION": "Immune",
    # TissueArch
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "TissueArch",
    "HALLMARK_APICAL_JUNCTION": "TissueArch",
    "HALLMARK_APICAL_SURFACE": "TissueArch",
    "HALLMARK_ANGIOGENESIS": "TissueArch",
    "HALLMARK_HYPOXIA": "TissueArch",
    "HALLMARK_ADIPOGENESIS": "TissueArch",
}


# ── 1. Hallmark GMT ─────────────────────────────────────────────────────────

def load_hallmark_gmt(gmt_path: str) -> dict[str, set[str]]:
    """Parse GMT file -> {set_name: {gene1, gene2, ...}}."""
    gene_sets: dict[str, set[str]] = {}
    with open(gmt_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            name = parts[0]
            genes = set(parts[2:])  # skip description column
            gene_sets[name] = genes
    return gene_sets


def download_hallmark_gmt() -> str:
    """Download Hallmark GMT or use cached copy."""
    import shutil
    import urllib.request

    url = (
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
        "2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"
    )
    cache_path = os.path.join(GNN_RESULTS, "h.all.v2024.1.Hs.symbols.gmt")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached GMT: {cache_path}")
        return cache_path

    # Try downloading
    print("  Downloading Hallmark GMT from Broad ...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        print(f"  Saved to {cache_path}")
        return cache_path
    except Exception as e:
        print(f"  Download failed ({e})")

    # Fall back to /tmp copy (may have been downloaded by curl)
    tmp = "/tmp/hallmark.gmt"
    if os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
        shutil.copy2(tmp, cache_path)
        print(f"  Copied from {tmp} to {cache_path}")
        return cache_path

    raise RuntimeError("Cannot obtain Hallmark GMT file")


# ── 2. Hallmark-based channel scoring ──────────────────────────────────────

def score_gene_by_hallmark(
    gene: str,
    gene_sets: dict[str, set[str]],
) -> dict:
    """Return channel scores and matched Hallmark sets for a gene."""
    channel_scores: Counter = Counter()
    matched_sets: list[str] = []

    for set_name, channel in HALLMARK_TO_CHANNEL.items():
        if set_name in gene_sets and gene in gene_sets[set_name]:
            channel_scores[channel] += 1
            matched_sets.append(set_name.replace("HALLMARK_", ""))

    return {"scores": channel_scores, "hallmark_sets": matched_sets}


def classify_confidence(scores: Counter) -> tuple[str, str]:
    """Return (best_channel, confidence) from channel score counts."""
    if not scores:
        return ("", "none")
    ranked = scores.most_common()
    best_ch, best_count = ranked[0]
    if len(ranked) == 1:
        return (best_ch, "high")
    second_count = ranked[1][1]
    if best_count > second_count:
        return (best_ch, "medium")
    # Tie at the top
    return (best_ch, "low")


# ── 3. Co-mutation analysis ────────────────────────────────────────────────

def build_comutation_data(
    mut_df: pd.DataFrame,
    mapped_genes: set[str],
    unmapped_genes: set[str],
) -> tuple[dict[str, Counter], dict[str, int]]:
    """
    For each unmapped gene, count patient-level co-occurrence with
    each mapped gene.

    Returns:
        comut: {unmapped_gene: Counter({mapped_gene: n_patients})}
        unmapped_patient_counts: {unmapped_gene: n_patients}
    """
    print("  Building co-mutation matrix ...")
    relevant = mapped_genes | unmapped_genes
    df = mut_df[mut_df["gene.hugoGeneSymbol"].isin(relevant)].copy()
    patient_genes = df.groupby("patientId")["gene.hugoGeneSymbol"].apply(set)

    comut: dict[str, Counter] = defaultdict(Counter)
    unmapped_patient_counts: Counter = Counter()

    for genes_in_patient in patient_genes:
        unmapped_in_patient = genes_in_patient & unmapped_genes
        mapped_in_patient = genes_in_patient & mapped_genes
        for ug in unmapped_in_patient:
            unmapped_patient_counts[ug] += 1
            for mg in mapped_in_patient:
                comut[ug][mg] += 1

    print(f"  Co-mutation profiles for {len(comut)} unmapped genes")
    return dict(comut), dict(unmapped_patient_counts)


def comutation_channel(
    gene: str,
    comut_profiles: dict[str, Counter],
    channel_map: dict[str, str],
    gene_patient_counts: dict[str, int],
    unmapped_patient_counts: dict[str, int],
) -> str | None:
    """
    Assign channel based on Jaccard co-occurrence with mapped genes.

    For each mapped gene, compute:
        jaccard = n_co_patients / (n_unmapped + n_mapped - n_co_patients)

    Assign to the channel of the single strongest Jaccard partner.
    Jaccard is naturally normalized for gene frequency, making it
    robust to the hypermutator confound that biases channel-level
    metrics toward rare or large channels.
    """
    if gene not in comut_profiles:
        return None

    profile = comut_profiles[gene]
    n_unmapped = unmapped_patient_counts.get(gene, 1)

    best_jaccard = 0.0
    best_channel = None

    for mapped_gene, n_comut in profile.items():
        ch = channel_map.get(mapped_gene)
        if not ch:
            continue
        n_mapped = gene_patient_counts.get(mapped_gene, 1)
        if n_mapped < 30:
            continue
        union = n_unmapped + n_mapped - n_comut
        jaccard = n_comut / union if union > 0 else 0.0

        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_channel = ch

    return best_channel


# ── 4. Main pipeline ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPAND CHANNEL MAP")
    print("MSigDB Hallmark gene sets + Jaccard co-mutation")
    print("=" * 70)

    # Load mutation data
    mut_path = MSK_DATASETS["msk_impact_50k"]["mutations"]
    print(f"\n1. Loading mutations: {mut_path}")
    mut_df = pd.read_csv(mut_path, usecols=["patientId", "gene.hugoGeneSymbol"])
    mut_df = mut_df.dropna(subset=["gene.hugoGeneSymbol"])

    all_data_genes = set(mut_df["gene.hugoGeneSymbol"].unique())
    mapped_genes = set(CHANNEL_MAP.keys())
    unmapped_genes = sorted(all_data_genes - mapped_genes)
    print(f"   Total genes in data: {len(all_data_genes)}")
    print(f"   Already mapped:      {len(mapped_genes & all_data_genes)}")
    print(f"   Unmapped:            {len(unmapped_genes)}")

    # Download / load Hallmark GMT
    print("\n2. Loading MSigDB Hallmark gene sets")
    gmt_path = download_hallmark_gmt()
    gene_sets = load_hallmark_gmt(gmt_path)
    mapped_hallmarks = {k for k in gene_sets if k in HALLMARK_TO_CHANNEL}
    print(f"   Loaded {len(gene_sets)} Hallmark sets, {len(mapped_hallmarks)} mapped to channels")

    # Score each unmapped gene via Hallmark membership
    print("\n3. Scoring unmapped genes via Hallmark sets")
    hallmark_results: dict[str, dict] = {}
    for gene in unmapped_genes:
        hallmark_results[gene] = score_gene_by_hallmark(gene, gene_sets)

    hallmark_hit = sum(1 for r in hallmark_results.values() if r["scores"])
    print(f"   Hallmark hit: {hallmark_hit}/{len(unmapped_genes)} genes")

    # Co-mutation analysis
    print("\n4. Co-mutation analysis")
    comut_profiles, unmapped_patient_counts = build_comutation_data(
        mut_df, mapped_genes, set(unmapped_genes),
    )
    gene_patient_counts = (
        mut_df[mut_df["gene.hugoGeneSymbol"].isin(mapped_genes)]
        .groupby("gene.hugoGeneSymbol")["patientId"]
        .nunique()
        .to_dict()
    )
    print(f"   Patient counts for {len(gene_patient_counts)} mapped genes")

    # Combine signals and assign channels
    print("\n5. Assigning channels")
    expanded: dict[str, dict] = {}
    source_counts: Counter = Counter()
    confidence_counts: Counter = Counter()
    channel_counts: Counter = Counter()

    for gene in unmapped_genes:
        hr = hallmark_results[gene]
        scores = hr["scores"]
        hallmark_sets = hr["hallmark_sets"]

        if scores:
            best_ch, confidence = classify_confidence(scores)
            # Use co-mutation as tiebreaker when confidence is low
            if confidence == "low":
                comut_ch = comutation_channel(
                    gene, comut_profiles, CHANNEL_MAP,
                    gene_patient_counts, unmapped_patient_counts,
                )
                if comut_ch and comut_ch in scores:
                    best_ch = comut_ch
                    confidence = "medium"
                    source = "both"
                else:
                    source = "hallmark"
            else:
                source = "hallmark"
        else:
            # No Hallmark signal -- use Jaccard co-mutation
            comut_ch = comutation_channel(
                gene, comut_profiles, CHANNEL_MAP,
                gene_patient_counts, unmapped_patient_counts,
            )
            if comut_ch:
                best_ch = comut_ch
                confidence = "low"
                source = "comutation"
                hallmark_sets = []
            else:
                best_ch = "PI3K_Growth"  # default fallback
                confidence = "low"
                source = "default"
                hallmark_sets = []

        expanded[gene] = {
            "channel": best_ch,
            "confidence": confidence,
            "source": source,
            "hallmark_sets": hallmark_sets,
        }
        source_counts[source] += 1
        confidence_counts[confidence] += 1
        channel_counts[best_ch] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal unmapped genes assigned: {len(expanded)}")

    print("\nBy channel:")
    for ch in CHANNEL_NAMES:
        orig = sum(1 for g, c in CHANNEL_MAP.items() if c == ch)
        new = channel_counts.get(ch, 0)
        print(f"  {ch:15s}  {orig:3d} original + {new:3d} new = {orig + new:3d} total")

    print("\nBy confidence:")
    for conf in ["high", "medium", "low"]:
        print(f"  {conf:8s}  {confidence_counts.get(conf, 0):3d}")

    print("\nBy source:")
    for src in ["hallmark", "comutation", "both", "default"]:
        print(f"  {src:12s}  {source_counts.get(src, 0):3d}")

    # Save output
    os.makedirs(GNN_RESULTS, exist_ok=True)
    out_path = os.path.join(GNN_RESULTS, "expanded_channel_map.json")

    # Include original mapped genes (curated, high confidence)
    full_map: dict[str, dict] = {}
    for gene, ch in CHANNEL_MAP.items():
        full_map[gene] = {
            "channel": ch,
            "confidence": "high",
            "source": "curated",
            "hallmark_sets": [],
        }
    full_map.update(expanded)

    with open(out_path, "w") as fh:
        json.dump(full_map, fh, indent=2, sort_keys=True)
    print(f"\nSaved expanded channel map ({len(full_map)} genes) to:\n  {out_path}")


if __name__ == "__main__":
    main()
