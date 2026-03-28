#!/usr/bin/env python3
"""
Discover ANALOGOUS edges — genes that share the same edit mechanism profile.

"Same edit basically" — if two genes are predominantly hit by the same mutation
types (e.g., both are missense-GOF hotspot genes, or both are biallelic-LOF
truncation targets), they are ANALOGOUS. This is the edit-level equivalent of
TRANSPOSES (same copy) and CONVERGES (same delete).

Also builds GOF/LOF prediction features: for each gene, what fraction of its
mutations are truncating vs missense, and does this predict the CIViC GOF/LOF
label? This validates the ANALOGOUS edge type and provides a pretraining target.

Usage:
    python3 -u -m gnn.scripts.discover_analogous_edges              # discover + write to Neo4j
    python3 -u -m gnn.scripts.discover_analogous_edges --dry-run    # analysis only
    python3 -u -m gnn.scripts.discover_analogous_edges --stats      # print stats only
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import TRUNCATING, GENE_FUNCTION, ANALYSIS_CACHE, GNN_CACHE

# ---------------------------------------------------------------------------
# Mutation type categories for edit mechanism profiling
# ---------------------------------------------------------------------------

# Group mutation types into functional categories
EDIT_CATEGORIES = {
    "truncating": {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
                   "Nonstop_Mutation"},
    "splice": {"Splice_Site", "Splice_Region"},
    "missense": {"Missense_Mutation"},
    "inframe_indel": {"In_Frame_Del", "In_Frame_Ins"},
    "regulatory": {"5'Flank", "Translation_Start_Site"},
}

EDIT_NAMES = list(EDIT_CATEGORIES.keys())  # fixed order for profile vector


def load_mutation_profiles(min_mutations=20):
    """Build per-gene edit mechanism profiles from MSK mutation data.

    Returns:
        profiles: dict gene -> np.array of shape (5,) — fraction in each edit category
        raw_counts: dict gene -> np.array of shape (5,) — raw counts
        gene_meta: dict gene -> {n_mutations, dominant_edit, gof_frac, lof_frac}
    """
    csv_path = os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv")
    print(f"  Loading mutations from {csv_path}...", flush=True)

    df = pd.read_csv(csv_path, usecols=["patientId", "gene.hugoGeneSymbol",
                                         "mutationType", "proteinChange"])
    df = df.rename(columns={"gene.hugoGeneSymbol": "gene"})
    df = df.dropna(subset=["mutationType", "gene"])

    # Count per gene × edit category
    gene_counts = defaultdict(lambda: np.zeros(len(EDIT_NAMES)))
    gene_total = defaultdict(int)
    gene_mut_types = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        gene = row["gene"]
        mt = row["mutationType"]
        gene_total[gene] += 1
        gene_mut_types[gene][mt] += 1

        for i, (cat, types) in enumerate(EDIT_CATEGORIES.items()):
            if mt in types:
                gene_counts[gene][i] += 1
                break

    # Filter to genes with enough mutations
    profiles = {}
    raw_counts = {}
    gene_meta = {}

    for gene, counts in gene_counts.items():
        total = gene_total[gene]
        if total < min_mutations:
            continue

        profile = counts / total  # fraction in each category
        profiles[gene] = profile
        raw_counts[gene] = counts

        # Determine dominant edit and GOF/LOF fraction
        dominant_idx = np.argmax(profile)
        trunc_frac = profile[0] + profile[1]  # truncating + splice
        missense_frac = profile[2]

        gene_meta[gene] = {
            "n_mutations": int(total),
            "dominant_edit": EDIT_NAMES[dominant_idx],
            "dominant_frac": float(profile[dominant_idx]),
            "trunc_frac": float(trunc_frac),
            "missense_frac": float(missense_frac),
            "inframe_frac": float(profile[3]),
            "regulatory_frac": float(profile[4]),
        }

    print(f"  Profiled {len(profiles)} genes (>= {min_mutations} mutations)")
    return profiles, raw_counts, gene_meta


def discover_analogous_pairs(profiles, gene_meta, min_cosine_sim=0.92,
                             min_mutations_both=50):
    """Find gene pairs with highly similar edit mechanism profiles.

    Uses z-scored profiles (deviation from population mean) so that the
    dominant missense baseline doesn't swamp the signal. Two genes are
    ANALOGOUS if their deviation vectors point in the same direction —
    both are anomalously truncating, both are anomalously missense-hotspot, etc.

    Returns list of dicts with edge properties.
    """
    genes = sorted(profiles.keys())
    prof_matrix = np.array([profiles[g] for g in genes])

    # Z-score against population mean — this is the key insight.
    # Raw profiles are 70% missense for most genes, so cosine on raw
    # fractions just says "both genes exist." The *deviation* from the
    # population mean tells us what's special about each gene's edit profile.
    pop_mean = prof_matrix.mean(axis=0)
    pop_std = prof_matrix.std(axis=0)
    pop_std[pop_std == 0] = 1
    z_matrix = (prof_matrix - pop_mean) / pop_std

    print(f"  Population mean profile: {', '.join(f'{n}={v:.3f}' for n, v in zip(EDIT_NAMES, pop_mean))}")
    print(f"  Population std profile:  {', '.join(f'{n}={v:.3f}' for n, v in zip(EDIT_NAMES, pop_std))}")

    # Cosine similarity on z-scored profiles
    norms = np.linalg.norm(z_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = z_matrix / norms
    sim_matrix = normed @ normed.T

    # Also require that the z-scored profile has non-trivial magnitude
    # (genes near the population mean have no distinctive edit signature)
    magnitudes = np.linalg.norm(z_matrix, axis=1)
    min_magnitude = 1.5  # at least 1.5 std away from mean in some direction

    pairs = []
    for i in range(len(genes)):
        if gene_meta[genes[i]]["n_mutations"] < min_mutations_both:
            continue
        if magnitudes[i] < min_magnitude:
            continue
        for j in range(i + 1, len(genes)):
            if gene_meta[genes[j]]["n_mutations"] < min_mutations_both:
                continue
            if magnitudes[j] < min_magnitude:
                continue
            sim = float(sim_matrix[i, j])
            if sim >= min_cosine_sim:
                g1, g2 = genes[i], genes[j]
                pairs.append({
                    "from": g1,
                    "to": g2,
                    "cosine_similarity": round(sim, 4),
                    "z_magnitude_a": round(float(magnitudes[i]), 3),
                    "z_magnitude_b": round(float(magnitudes[j]), 3),
                    "dominant_edit_a": gene_meta[g1]["dominant_edit"],
                    "dominant_edit_b": gene_meta[g2]["dominant_edit"],
                    "shared_dominant": gene_meta[g1]["dominant_edit"] == gene_meta[g2]["dominant_edit"],
                    "n_mutations_a": gene_meta[g1]["n_mutations"],
                    "n_mutations_b": gene_meta[g2]["n_mutations"],
                })

    # Count genes that passed magnitude filter
    n_distinctive = sum(1 for m in magnitudes if m >= min_magnitude)
    print(f"  Distinctive genes (z-magnitude >= {min_magnitude}): {n_distinctive}/{len(genes)}")
    print(f"  Found {len(pairs)} ANALOGOUS pairs (z-cosine >= {min_cosine_sim})")
    return pairs


def validate_gof_lof_prediction(profiles, gene_meta):
    """Can we predict CIViC GOF/LOF labels from mutation type profiles?

    This validates the ANALOGOUS edge type: if edit mechanism profiles predict
    functional direction, then genes with similar profiles truly share the same
    biological editing process.

    Returns dict with accuracy, AUC, and feature importances.
    """
    from neo4j import GraphDatabase

    # Get CIViC labels from Neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                      auth=("neo4j", "openknowledgegraph"))
    with driver.session() as session:
        result = session.run("""
            MATCH (g:Gene)
            WHERE g.civic_dominant_function IS NOT NULL
              AND g.civic_dominant_function <> 'unknown'
            RETURN g.name AS gene, g.civic_dominant_function AS func,
                   g.civic_gof_count AS gof, g.civic_lof_count AS lof
        """)
        civic = {r["gene"]: r["func"] for r in result}
    driver.close()

    print(f"  CIViC labels: {len(civic)} genes ({sum(1 for v in civic.values() if v == 'GOF')} GOF, "
          f"{sum(1 for v in civic.values() if v == 'LOF')} LOF, "
          f"{sum(1 for v in civic.values() if v == 'mixed')} mixed)")

    # Build feature matrix for genes with both profile and CIViC label
    X, y, gene_names = [], [], []
    for gene, profile in profiles.items():
        if gene not in civic:
            continue
        label = civic[gene]
        if label == "mixed":
            continue  # skip ambiguous
        X.append(profile)
        y.append(1 if label == "GOF" else 0)
        gene_names.append(gene)

    if len(X) < 10:
        print("  Too few labeled genes for GOF/LOF prediction validation")
        return None

    X = np.array(X)
    y = np.array(y)
    print(f"  GOF/LOF prediction: {len(X)} genes ({sum(y)} GOF, {len(y) - sum(y)} LOF)")

    # Simple logistic regression — no train/test split needed, this is validation
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    model = LogisticRegression(random_state=42, max_iter=1000)
    y_pred_proba = cross_val_predict(model, X, y, cv=min(5, len(X)), method="predict_proba")[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_pred_proba)
    except ValueError:
        auc = float("nan")

    # Fit on all data for feature importances
    model.fit(X, y)
    coefs = dict(zip(EDIT_NAMES, model.coef_[0]))

    results = {
        "n_genes": len(X),
        "n_gof": int(sum(y)),
        "n_lof": int(len(y) - sum(y)),
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "feature_coefficients": {k: round(v, 4) for k, v in coefs.items()},
        "interpretation": {
            "truncating_coef": round(coefs["truncating"], 4),
            "missense_coef": round(coefs["missense"], 4),
        },
        "top_misclassified": [],
    }

    # Find misclassified genes (interesting — what breaks the heuristic?)
    for i, gene in enumerate(gene_names):
        if y_pred[i] != y[i]:
            results["top_misclassified"].append({
                "gene": gene,
                "true_label": "GOF" if y[i] == 1 else "LOF",
                "predicted": "GOF" if y_pred[i] == 1 else "LOF",
                "confidence": round(float(y_pred_proba[i]), 3),
                "profile": {k: round(v, 4) for k, v in zip(EDIT_NAMES, X[i])},
            })

    print(f"  GOF/LOF prediction: accuracy={acc:.3f}, AUC={auc:.3f}")
    print(f"  Coefficients: {', '.join(f'{k}={v:.3f}' for k, v in coefs.items())}")
    return results


def cluster_by_edit_profile(profiles, gene_meta, n_clusters=8):
    """Cluster genes by edit mechanism profile to find edit-type groups.

    Returns cluster assignments and cluster centers.
    """
    from sklearn.cluster import KMeans

    genes = sorted(profiles.keys())
    X = np.array([profiles[g] for g in genes])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    clusters = defaultdict(list)
    for gene, label in zip(genes, labels):
        clusters[int(label)].append(gene)

    print(f"\n  Edit mechanism clusters ({n_clusters}):")
    for c in sorted(clusters.keys()):
        center = km.cluster_centers_[c]
        dominant = EDIT_NAMES[np.argmax(center)]
        members = clusters[c]
        print(f"    Cluster {c} ({dominant:>15s}): {len(members):>3d} genes  "
              f"[{', '.join(f'{n}={v:.2f}' for n, v in zip(EDIT_NAMES, center))}]")
        # Show top 5 by mutation count
        top = sorted(members, key=lambda g: gene_meta[g]["n_mutations"], reverse=True)[:5]
        print(f"      Top: {', '.join(top)}")

    return {int(k): v for k, v in clusters.items()}, km.cluster_centers_.tolist()


def write_to_neo4j(pairs, gene_meta, dry_run=False):
    """Write ANALOGOUS edges and edit profile properties to Neo4j."""
    from gnn.data.graph_changelog import GraphGateway

    gw = GraphGateway(dry_run=dry_run)

    # Write ANALOGOUS edges
    n = gw.merge_edges("ANALOGOUS", pairs,
                       source="discover_analogous_edges",
                       source_detail=f"{len(pairs)} pairs, cosine >= 0.95")
    print(f"  Wrote {n} ANALOGOUS edges to Neo4j" + (" (dry run)" if dry_run else ""))

    # Update gene nodes with edit profile properties
    if not dry_run:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                      auth=("neo4j", "openknowledgegraph"))
        with driver.session() as session:
            batch = []
            for gene, meta in gene_meta.items():
                batch.append({
                    "gene": gene,
                    "dominant_edit": meta["dominant_edit"],
                    "dominant_edit_frac": meta["dominant_frac"],
                    "trunc_frac": meta["trunc_frac"],
                    "missense_frac": meta["missense_frac"],
                    "inframe_frac": meta["inframe_frac"],
                    "n_mutations": meta["n_mutations"],
                })
                if len(batch) >= 500:
                    session.run("""
                        UNWIND $batch AS row
                        MATCH (g:Gene {name: row.gene})
                        SET g.dominant_edit = row.dominant_edit,
                            g.dominant_edit_frac = row.dominant_edit_frac,
                            g.trunc_frac = row.trunc_frac,
                            g.missense_frac = row.missense_frac,
                            g.inframe_frac = row.inframe_frac,
                            g.n_msk_mutations = row.n_mutations
                    """, batch=batch)
                    batch = []
            if batch:
                session.run("""
                    UNWIND $batch AS row
                    MATCH (g:Gene {name: row.gene})
                    SET g.dominant_edit = row.dominant_edit,
                        g.dominant_edit_frac = row.dominant_edit_frac,
                        g.trunc_frac = row.trunc_frac,
                        g.missense_frac = row.missense_frac,
                        g.inframe_frac = row.inframe_frac,
                        g.n_msk_mutations = row.n_mutations
                """, batch=batch)
        driver.close()
        print(f"  Updated {len(gene_meta)} Gene nodes with edit profiles")

    gw.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true", help="Print stats only, no Neo4j writes")
    parser.add_argument("--min-sim", type=float, default=0.97)
    parser.add_argument("--min-mutations", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("  ANALOGOUS EDGE DISCOVERY")
    print("  Same edit mechanism across different genes")
    print("=" * 70)

    t0 = time.time()

    # 1. Build mutation type profiles
    profiles, raw_counts, gene_meta = load_mutation_profiles(min_mutations=20)

    # 2. Discover analogous pairs
    pairs = discover_analogous_pairs(profiles, gene_meta,
                                      min_cosine_sim=args.min_sim,
                                      min_mutations_both=args.min_mutations)

    # Show distribution of dominant edit types in pairs
    dom_dist = defaultdict(int)
    for p in pairs:
        if p["shared_dominant"]:
            dom_dist[p["dominant_edit_a"]] += 1
        else:
            key = f"{p['dominant_edit_a']}×{p['dominant_edit_b']}"
            dom_dist[key] += 1
    print("\n  ANALOGOUS pairs by dominant edit:")
    for k, v in sorted(dom_dist.items(), key=lambda x: -x[1]):
        print(f"    {k:>30s}: {v:>4d}")

    # 3. Validate: can edit profiles predict GOF/LOF?
    print("\n  --- GOF/LOF Prediction Validation ---")
    gof_lof_results = validate_gof_lof_prediction(profiles, gene_meta)

    # 4. Cluster genes by edit profile
    clusters, centers = cluster_by_edit_profile(profiles, gene_meta)

    # 5. Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", "analogous_edges")
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "n_genes_profiled": len(profiles),
        "n_analogous_pairs": len(pairs),
        "min_cosine_sim": args.min_sim,
        "min_mutations": args.min_mutations,
        "dominant_edit_distribution": dict(dom_dist),
        "gof_lof_validation": gof_lof_results,
        "clusters": {str(k): {"n_genes": len(v), "top_genes": v[:10]}
                     for k, v in clusters.items()},
        "cluster_centers": {str(i): dict(zip(EDIT_NAMES, c))
                            for i, c in enumerate(centers)},
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # Save top pairs
    results["top_pairs"] = sorted(pairs, key=lambda p: -p["cosine_similarity"])[:50]

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_dir}/results.json")

    # Save full profiles for downstream use
    profile_data = {gene: {
        "profile": profiles[gene].tolist(),
        "meta": gene_meta[gene],
    } for gene in profiles}
    with open(os.path.join(results_dir, "gene_edit_profiles.json"), "w") as f:
        json.dump(profile_data, f, indent=2)

    # 6. Write to Neo4j
    if not args.stats:
        print("\n  --- Writing to Neo4j ---")
        write_to_neo4j(pairs, gene_meta, dry_run=args.dry_run)

    print(f"\n{'=' * 70}")
    print(f"  DONE [{time.time() - t0:.1f}s]")
    print(f"  {len(profiles)} genes profiled, {len(pairs)} ANALOGOUS pairs")
    if gof_lof_results:
        print(f"  GOF/LOF prediction: AUC={gof_lof_results['auc']:.3f}, "
              f"accuracy={gof_lof_results['accuracy']:.3f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
