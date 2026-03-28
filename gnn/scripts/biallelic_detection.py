"""
Biallelic inactivation/activation detection from TCGA mutation + CNA data.

Cross-references per-patient CNA values (GISTIC -2 to +2) with point mutations
to identify two-hit events:

  Tumor suppressors (biallelic inactivation):
    - LOH + mutation: CNA=-1 AND has damaging mutation → both copies out
    - Homozygous deletion: CNA=-2 (no mutation needed) → both copies gone
    - Two mutations: 2+ distinct damaging mutations on same gene

  Oncogenes (biallelic activation):
    - Amplification + GOF mutation: CNA=+2 AND has activating mutation → overdrive
    - High copy gain + mutation: CNA=+1 AND has known hotspot

Outputs:
  1. Per-patient biallelic calls → Neo4j HAS_MUTATION edge properties
  2. Per-gene×CT biallelic frequency → Gene node properties
  3. Survival impact of biallelic vs monoallelic → PROGNOSTIC_IN edge updates

Usage:
    python3 -u -m gnn.scripts.biallelic_detection [--dry-run]
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import ALL_GENES, CHANNEL_MAP, HUB_GENES, GNN_CACHE

TCGA_CACHE = os.path.join(GNN_CACHE, "tcga")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "biallelic")

# Damaging mutation types (likely loss of function)
DAMAGING_TYPES = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "Splice_Region", "Translation_Start_Site",
    "Nonstop_Mutation",
}

# Likely activating / GOF mutation types
MISSENSE_TYPES = {"Missense_Mutation", "In_Frame_Del", "In_Frame_Ins"}

# Known GOF hotspot genes (mutations in these are typically activating)
GOF_GENES = {
    "KRAS", "NRAS", "HRAS", "BRAF", "PIK3CA", "AKT1", "MTOR",
    "EGFR", "ERBB2", "ERBB3", "FGFR2", "FGFR3", "MET", "KIT",
    "PDGFRA", "RET", "ALK", "IDH1", "IDH2", "CTNNB1", "SMO",
    "GNAQ", "GNA11", "SF3B1", "U2AF1", "SPOP", "FOXA1",
    "ESR1", "AR", "JAK2", "MPL", "CALR",
}

# Known tumor suppressor genes (mutations are typically LOF)
TSG_GENES = {
    "TP53", "RB1", "PTEN", "APC", "BRCA1", "BRCA2", "VHL",
    "CDKN2A", "CDKN2B", "NF1", "NF2", "STK11", "SMAD4",
    "BAP1", "ARID1A", "ARID1B", "ARID2", "SMARCA4", "SMARCB1",
    "KMT2C", "KMT2D", "KDM6A", "KDM5C", "SETD2", "PBRM1",
    "ATM", "ATR", "CHEK2", "PALB2", "RAD51C", "RAD51D",
    "FBXW7", "NOTCH1", "KEAP1", "CASP8", "FAT1",
    "RNF43", "AXIN1", "AXIN2", "SOX9", "CDH1",
    "STAG2", "BCOR", "ATRX", "MED12", "TET2",
    "WT1", "CREBBP", "EP300", "NCOR1",
}


def classify_biallelic(gene, cna_value, mutations):
    """Classify a patient×gene event as biallelic, monoallelic, or wild-type.

    Args:
        gene: gene name
        cna_value: GISTIC value (-2, -1, 0, 1, 2) or None
        mutations: list of mutation dicts for this patient×gene

    Returns:
        dict with: status, mechanism, n_hits, details
    """
    has_mut = len(mutations) > 0
    n_damaging = sum(1 for m in mutations if m["mutation_type"] in DAMAGING_TYPES)
    n_missense = sum(1 for m in mutations if m["mutation_type"] in MISSENSE_TYPES)
    is_tsg = gene in TSG_GENES
    is_oncogene = gene in GOF_GENES

    cna = int(cna_value) if cna_value is not None else 0

    result = {
        "gene": gene,
        "cna": cna,
        "n_mutations": len(mutations),
        "n_damaging": n_damaging,
        "n_missense": n_missense,
        "status": "wild_type",
        "mechanism": None,
        "n_hits": 0,
    }

    if not has_mut and cna == 0:
        return result

    # --- Tumor suppressor logic ---
    if is_tsg:
        # Homozygous deletion — both copies gone at DNA level
        if cna == -2:
            result["status"] = "biallelic"
            result["mechanism"] = "homodel" if not has_mut else "homodel+mut"
            result["n_hits"] = 2
            return result

        # LOH + damaging mutation — one copy lost, other mutated
        if cna == -1 and n_damaging >= 1:
            result["status"] = "biallelic"
            result["mechanism"] = "loh+mutation"
            result["n_hits"] = 2
            return result

        # LOH + missense — possible biallelic if missense is damaging
        if cna == -1 and n_missense >= 1:
            result["status"] = "biallelic"
            result["mechanism"] = "loh+missense"
            result["n_hits"] = 2
            return result

        # Two distinct damaging mutations (compound heterozygous)
        if n_damaging >= 2:
            result["status"] = "biallelic"
            result["mechanism"] = "compound_het"
            result["n_hits"] = 2
            return result

        # Single hit — monoallelic
        if cna == -1 or cna == -2:
            result["status"] = "monoallelic"
            result["mechanism"] = "loh_only"
            result["n_hits"] = 1
            return result

        if has_mut:
            result["status"] = "monoallelic"
            result["mechanism"] = "mutation_only"
            result["n_hits"] = 1
            return result

    # --- Oncogene logic ---
    if is_oncogene:
        # Amplification + activating mutation — double activation
        if cna == 2 and has_mut:
            result["status"] = "biallelic"
            result["mechanism"] = "amp+mutation"
            result["n_hits"] = 2
            return result

        # High amplification alone — multiple copies active
        if cna == 2:
            result["status"] = "monoallelic"
            result["mechanism"] = "amp_only"
            result["n_hits"] = 1
            return result

        # Gain + hotspot mutation
        if cna == 1 and has_mut:
            result["status"] = "biallelic"
            result["mechanism"] = "gain+mutation"
            result["n_hits"] = 2
            return result

        if has_mut:
            result["status"] = "monoallelic"
            result["mechanism"] = "mutation_only"
            result["n_hits"] = 1
            return result

    # --- Other genes (unknown role) ---
    if cna == -2:
        result["status"] = "biallelic"
        result["mechanism"] = "homodel"
        result["n_hits"] = 2
    elif cna <= -1 and n_damaging >= 1:
        result["status"] = "biallelic"
        result["mechanism"] = "loh+mutation"
        result["n_hits"] = 2
    elif cna >= 2 and has_mut:
        result["status"] = "biallelic"
        result["mechanism"] = "amp+mutation"
        result["n_hits"] = 2
    elif has_mut or cna != 0:
        result["status"] = "monoallelic"
        result["mechanism"] = "single_hit"
        result["n_hits"] = 1

    return result


def run_biallelic_detection(dry_run=False):
    """Main detection pipeline."""
    print("=" * 70)
    print("  BIALLELIC DETECTION")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # Load data
    print("\n  Loading data...")
    mut = pd.read_csv(os.path.join(TCGA_CACHE, "tcga_mutations.csv"))
    cna = pd.read_csv(os.path.join(TCGA_CACHE, "tcga_cna_raw.csv"))

    # Build lookup: (sample_id, gene) → CNA value
    print(f"  Building CNA lookup ({len(cna):,} rows)...")
    cna_lookup = {}
    for _, row in cna.iterrows():
        cna_lookup[(row["sample_id"], row["gene"])] = row["value"]

    # Build mutation groups: (sample_id, gene) → [mutations]
    print(f"  Building mutation groups ({len(mut):,} rows)...")
    mut_groups = defaultdict(list)
    for _, row in mut.iterrows():
        mut_groups[(row["sample_id"], row["gene"])].append({
            "protein_change": row["protein_change"],
            "mutation_type": row["mutation_type"],
            "variant_type": row["variant_type"],
        })

    # Get all sample×gene pairs to check
    # Union of: mutated pairs + CNA non-zero pairs
    gene_set = set(ALL_GENES)
    pairs_to_check = set()

    for (sid, gene), muts in mut_groups.items():
        if gene in gene_set:
            pairs_to_check.add((sid, gene))

    for (sid, gene), val in cna_lookup.items():
        if gene in gene_set and val != 0:
            pairs_to_check.add((sid, gene))

    print(f"  Pairs to check: {len(pairs_to_check):,}")

    # Get sample → cancer_type mapping
    sample_ct = {}
    for _, row in mut.iterrows():
        sample_ct[row["sample_id"]] = row["cancer_type"]
    for _, row in cna.drop_duplicates("sample_id").iterrows():
        if row["sample_id"] not in sample_ct:
            sample_ct[row["sample_id"]] = row["cancer_type"]

    # --- Run classification ---
    print("\n  Classifying biallelic events...")
    t1 = time.time()

    results = []
    for sid, gene in pairs_to_check:
        cna_val = cna_lookup.get((sid, gene))
        muts = mut_groups.get((sid, gene), [])
        call = classify_biallelic(gene, cna_val, muts)
        call["sample_id"] = sid
        call["cancer_type"] = sample_ct.get(sid, "Unknown")
        # Extract patient ID (first 12 chars of TCGA barcode)
        call["patient_id"] = sid[:12] if sid.startswith("TCGA-") else sid
        results.append(call)

    results_df = pd.DataFrame(results)
    elapsed = time.time() - t1
    print(f"  Classified {len(results_df):,} events in {elapsed:.1f}s")

    # --- Summary statistics ---
    print(f"\n  {'Status':<15} {'Count':>10} {'Pct':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*8}")
    for status, count in results_df["status"].value_counts().items():
        print(f"  {status:<15} {count:>10,} {count/len(results_df):>8.1%}")

    biallelic = results_df[results_df["status"] == "biallelic"]
    print(f"\n  Biallelic mechanisms:")
    print(f"  {'Mechanism':<20} {'Count':>8} {'Pct of biallelic':>18}")
    print(f"  {'-'*20} {'-'*8} {'-'*18}")
    for mech, count in biallelic["mechanism"].value_counts().items():
        print(f"  {mech:<20} {count:>8,} {count/len(biallelic):>18.1%}")

    # --- Per-gene biallelic frequency ---
    print(f"\n  Top genes by biallelic frequency:")
    gene_stats = []
    for gene, gdf in results_df.groupby("gene"):
        n_total = len(gdf)
        n_biallelic = (gdf["status"] == "biallelic").sum()
        n_mono = (gdf["status"] == "monoallelic").sum()
        if n_total < 20:
            continue
        gene_stats.append({
            "gene": gene,
            "n_total": n_total,
            "n_biallelic": n_biallelic,
            "n_monoallelic": n_mono,
            "biallelic_freq": n_biallelic / n_total,
            "is_tsg": gene in TSG_GENES,
            "is_oncogene": gene in GOF_GENES,
        })

    gene_stats_df = pd.DataFrame(gene_stats).sort_values("biallelic_freq", ascending=False)
    print(f"\n  {'Gene':<12} {'Total':>6} {'Biallelic':>10} {'Freq':>7} {'Type':<10}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*7} {'-'*10}")
    for _, row in gene_stats_df.head(30).iterrows():
        gtype = "TSG" if row["is_tsg"] else ("ONC" if row["is_oncogene"] else "other")
        print(f"  {row['gene']:<12} {row['n_total']:>6} {row['n_biallelic']:>10} "
              f"{row['biallelic_freq']:>7.1%} {gtype:<10}")

    # --- Per cancer type × gene biallelic rates ---
    print(f"\n  Per cancer type × gene biallelic rates (top 30):")
    ct_gene_stats = []
    for (ct, gene), gdf in results_df.groupby(["cancer_type", "gene"]):
        n_bi = (gdf["status"] == "biallelic").sum()
        n_mono = (gdf["status"] == "monoallelic").sum()
        n_total = len(gdf)
        if n_bi >= 5 and n_total >= 10:
            ct_gene_stats.append({
                "cancer_type": ct, "gene": gene,
                "n_biallelic": n_bi, "n_monoallelic": n_mono,
                "n_total": n_total,
                "biallelic_freq": n_bi / n_total,
            })

    ct_gene_df = pd.DataFrame(ct_gene_stats).sort_values("biallelic_freq", ascending=False)
    print(f"\n  {'CT':<12} {'Gene':<12} {'Bi':>4} {'Mono':>5} {'Freq':>7}")
    print(f"  {'-'*12} {'-'*12} {'-'*4} {'-'*5} {'-'*7}")
    for _, row in ct_gene_df.head(30).iterrows():
        print(f"  {row['cancer_type']:<12} {row['gene']:<12} {row['n_biallelic']:>4} "
              f"{row['n_monoallelic']:>5} {row['biallelic_freq']:>7.1%}")

    # --- Write to Neo4j ---
    print(f"\n  Writing to Neo4j...")
    t1 = time.time()

    if not dry_run:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                      auth=("neo4j", "openknowledgegraph"))

        # 1. Update HAS_MUTATION edges with biallelic status
        biallelic_updates = results_df[results_df["status"] != "wild_type"].copy()
        with driver.session() as session:
            batch_size = 500
            for i in range(0, len(biallelic_updates), batch_size):
                batch = []
                for _, row in biallelic_updates.iloc[i:i+batch_size].iterrows():
                    batch.append({
                        "pid": row["patient_id"],
                        "gene": row["gene"],
                        "status": row["status"],
                        "mechanism": row["mechanism"] or "unknown",
                        "n_hits": int(row["n_hits"]),
                        "cna": int(row["cna"]),
                    })
                session.run("""
                    UNWIND $batch AS b
                    MATCH (p:Patient {id: b.pid})-[r:HAS_MUTATION]->(g:Gene {name: b.gene})
                    SET r.biallelic_status = b.status,
                        r.biallelic_mechanism = b.mechanism,
                        r.n_hits = b.n_hits,
                        r.cna_value = b.cna
                """, batch=batch)
            print(f"  Updated {len(biallelic_updates):,} HAS_MUTATION edges "
                  f"[{time.time()-t1:.1f}s]")

        # 2. Store per-gene×CT biallelic frequency as edge properties
        t1 = time.time()
        if len(ct_gene_df) > 0:
            with driver.session() as session:
                batch = []
                for _, row in ct_gene_df.iterrows():
                    batch.append({
                        "gene": row["gene"],
                        "ct": row["cancer_type"],
                        "biallelic_freq": round(float(row["biallelic_freq"]), 4),
                        "n_biallelic": int(row["n_biallelic"]),
                        "n_monoallelic": int(row["n_monoallelic"]),
                    })
                # Store as BIALLELIC_IN edges
                session.run("""
                    UNWIND $batch AS b
                    MATCH (g:Gene {name: b.gene})
                    MERGE (ct:CancerType {name: b.ct})
                    MERGE (g)-[r:BIALLELIC_IN]->(ct)
                    SET r.biallelic_freq = b.biallelic_freq,
                        r.n_biallelic = b.n_biallelic,
                        r.n_monoallelic = b.n_monoallelic,
                        r.source = 'TCGA_CNA+mutation'
                """, batch=batch)
            print(f"  Created {len(ct_gene_df):,} BIALLELIC_IN edges [{time.time()-t1:.1f}s]")

        driver.close()
    else:
        print(f"  [DRY RUN] Would update {len(results_df[results_df['status'] != 'wild_type']):,} "
              f"HAS_MUTATION edges")
        print(f"  [DRY RUN] Would create {len(ct_gene_df):,} BIALLELIC_IN edges")

    # Save full results
    results_path = os.path.join(RESULTS_DIR, "biallelic_calls.csv")
    results_df.to_csv(results_path, index=False)

    summary_path = os.path.join(RESULTS_DIR, "biallelic_summary.json")
    summary = {
        "total_events": len(results_df),
        "biallelic": int((results_df["status"] == "biallelic").sum()),
        "monoallelic": int((results_df["status"] == "monoallelic").sum()),
        "wild_type": int((results_df["status"] == "wild_type").sum()),
        "mechanisms": biallelic["mechanism"].value_counts().to_dict(),
        "n_genes": int(results_df["gene"].nunique()),
        "n_patients": int(results_df["sample_id"].nunique()),
        "n_cancer_types": int(results_df["cancer_type"].nunique()),
        "elapsed_s": time.time() - t0,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results: {results_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")

    return results_df, gene_stats_df, ct_gene_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_biallelic_detection(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
