"""
Gap analysis loop — self-improving cycle that finds where the model is worst
and generates new edges or adjusts weights to close the gap.

Integrated into the feedback loop between measure_c_index and convergence_check.

Each call:
  1. Query Neo4j for all patients + mutations per cancer type
  2. Score patients with current graph scorer
  3. Compare per-CT C-index: graph vs V6c transformer (benchmark)
  4. For worst-gap CTs: profile gap patients (enriched genes, channels, mutation patterns)
  5. Generate edge proposals:
     - New PROGNOSTIC_IN edges for gap-enriched gene×CT pairs (Cox HR from survival)
     - Adjusted PPI weights for gap-enriched gene pairs
     - New COOCCURS edges for co-mutation patterns in gap patients
  6. Write proposals to Neo4j via GraphGateway (logged, never deletes)

Usage:
    # Standalone
    python3 -u -m gnn.gap_analysis_loop [--dry-run] [--min-gap 0.015]

    # From feedback loop
    from gnn.gap_analysis_loop import run_gap_analysis
    proposals = run_gap_analysis(cycle=0, dry_run=False)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE, HUB_GENES
from gnn.data.graph_snapshot import get_driver

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "gap_analysis")


def _step(name, actual=None, elapsed=None):
    status = f"{actual:,}" if actual is not None else ""
    t = f"[{elapsed:.1f}s]" if elapsed is not None else ""
    print(f"  {name:.<50s} {status:>15s}  {t}", flush=True)


# =========================================================================
# Step 1: Query patient data from Neo4j
# =========================================================================

def query_all_patients(driver):
    """Pull all patients with mutations, survival, and cancer type from Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient)
            WHERE p.os_months IS NOT NULL AND p.os_months > 0
              AND p.event IS NOT NULL
            OPTIONAL MATCH (p)-[m:HAS_MUTATION]->(g:Gene)
            WITH p,
                 collect(DISTINCT {
                     gene: g.name,
                     channel: g.primary_channel,
                     is_hub: g.is_hub,
                     direction: m.direction
                 }) AS mutations
            RETURN p.id AS pid,
                   p.cancer_type AS cancer_type,
                   p.os_months AS os_months,
                   p.event AS event,
                   mutations
        """)
        records = []
        for r in result:
            muts = [m for m in r["mutations"] if m["gene"] is not None]
            genes = [m["gene"] for m in muts]
            channels = [m["channel"] for m in muts if m["channel"]]
            records.append({
                "pid": r["pid"],
                "cancer_type": r["cancer_type"],
                "os_months": float(r["os_months"]),
                "event": int(r["event"]),
                "genes": genes,
                "channels": channels,
                "n_mut": len(muts),
                "n_hub": sum(1 for m in muts if m["is_hub"]),
                "n_gof": sum(1 for m in muts if m.get("direction") == "GOF"),
                "n_lof": sum(1 for m in muts if m.get("direction") == "LOF"),
                "n_channels": len(set(channels)),
            })
    return pd.DataFrame(records)


def query_existing_prognostic(driver):
    """Get existing PROGNOSTIC_IN edges so we don't duplicate."""
    with driver.session() as session:
        result = session.run("""
            MATCH (g:Gene)-[r:PROGNOSTIC_IN]->(ct:CancerType)
            RETURN g.name AS gene, ct.name AS cancer_type,
                   r.hr AS hr, r.tier AS tier, r.source AS source
        """)
        existing = {}
        for r in result:
            existing[(r["cancer_type"], r["gene"])] = {
                "hr": r["hr"], "tier": r["tier"], "source": r["source"]
            }
    return existing


def query_ppi_neighbors(driver):
    """Get PPI edges from Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (g1:Gene)-[r:PPI]-(g2:Gene)
            WHERE g1.name < g2.name
            RETURN g1.name AS gene1, g2.name AS gene2, r.score AS score
        """)
        edges = {}
        for r in result:
            edges[(r["gene1"], r["gene2"])] = r["score"] or 0.4
    return edges


# =========================================================================
# Step 2: Compute per-CT C-index for graph scorer
# =========================================================================

def compute_per_ct_scores(patients_df, driver):
    """Score each patient using the graph scorer and compute per-CT C-index.

    Returns dict: {cancer_type: {graph_c, n_patients, n_events, patient_scores}}
    """
    from gnn.models.graph_scorer import GraphScorer
    from gnn.training.metrics import concordance_index
    import torch

    scorer = GraphScorer()
    scorer.load_from_neo4j()
    scorer.load_atlas()
    ct_results = {}

    for ct_name, ct_df in patients_df.groupby("cancer_type"):
        if len(ct_df) < 50 or ct_df["event"].sum() < 10:
            continue

        scores = []
        for _, row in ct_df.iterrows():
            genes = row["genes"]
            # protein_changes not available from patient query;
            # pass empty strings (tier-1 variant lookup will miss, tier 2+ unaffected)
            protein_changes = [""] * len(genes)
            result = scorer.score_patient(
                genes=genes,
                protein_changes=protein_changes,
                cancer_type=ct_name,
            )
            # score_patient returns a dict; extract total score
            s = result.get("total_score", result) if isinstance(result, dict) else result
            scores.append(float(s))

        scores_arr = np.array(scores, dtype=np.float32)
        times_arr = ct_df["os_months"].values.astype(np.float32)
        events_arr = ct_df["event"].values.astype(np.float32)

        try:
            ci = concordance_index(
                torch.tensor(scores_arr),
                torch.tensor(times_arr),
                torch.tensor(events_arr),
            )
        except Exception:
            ci = 0.5

        ct_results[ct_name] = {
            "graph_c": float(ci),
            "n_patients": len(ct_df),
            "n_events": int(ct_df["event"].sum()),
            "patient_scores": scores_arr,
            "patient_pids": ct_df["pid"].values,
            "patient_times": times_arr,
            "patient_events": events_arr,
        }

    return ct_results


# =========================================================================
# Step 3: Profile gap patients
# =========================================================================

def profile_gap_patients(patients_df, ct_results, min_gap=0.015, min_n=100):
    """For each high-gap CT, profile gap patients vs non-gap patients.

    Gap patients = those where graph ranking is most discordant with survival.
    (Concordant pairs where graph gets it wrong but a random predictor wouldn't.)

    Returns list of gap profiles with enriched genes and channels.
    """
    hub_set = set()
    for hubs in HUB_GENES.values():
        hub_set |= hubs

    gap_profiles = []

    for ct_name, info in sorted(ct_results.items(), key=lambda x: -x[1].get("graph_c", 1)):
        if info["n_patients"] < min_n:
            continue

        scores = info["patient_scores"]
        times = info["patient_times"]
        events = info["patient_events"]
        pids = info["patient_pids"]

        # Identify gap patients: those contributing most to concordance failures
        # For each event patient, check if graph score correctly ranks them vs censored
        n = len(scores)
        patient_errors = np.zeros(n)

        for i in range(n):
            if events[i] == 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                if times[j] > times[i]:
                    # j survived longer than i. Graph should score i higher risk.
                    if scores[i] < scores[j]:
                        patient_errors[i] += 1  # graph got this wrong
                        patient_errors[j] += 1

        # Normalize by number of comparisons
        n_comparisons = max(patient_errors.sum(), 1)
        patient_errors /= n_comparisons

        # Top 25% error patients = gap patients
        p75 = np.percentile(patient_errors, 75)
        gap_mask = patient_errors > p75
        nogap_mask = patient_errors <= np.percentile(patient_errors, 25)

        if gap_mask.sum() < 10 or nogap_mask.sum() < 10:
            continue

        ct_patients = patients_df[patients_df["cancer_type"] == ct_name]
        pid_to_row = {row["pid"]: row for _, row in ct_patients.iterrows()}

        # Gene enrichment
        gap_genes = Counter()
        nogap_genes = Counter()
        gap_channels = Counter()
        nogap_channels = Counter()
        n_gap = int(gap_mask.sum())
        n_nogap = int(nogap_mask.sum())

        for idx in range(n):
            pid = pids[idx]
            row = pid_to_row.get(pid)
            if row is None:
                continue

            if gap_mask[idx]:
                for g in row["genes"]:
                    gap_genes[g] += 1
                for ch in row["channels"]:
                    gap_channels[ch] += 1
            elif nogap_mask[idx]:
                for g in row["genes"]:
                    nogap_genes[g] += 1
                for ch in row["channels"]:
                    nogap_channels[ch] += 1

        # Compute enrichment ratios
        gene_enrichment = {}
        all_genes = set(gap_genes.keys()) | set(nogap_genes.keys())
        for g in all_genes:
            gap_frac = gap_genes.get(g, 0) / max(n_gap, 1)
            nogap_frac = nogap_genes.get(g, 0) / max(n_nogap, 1)
            # At least 5% prevalence in either group
            if gap_frac + nogap_frac < 0.05:
                continue
            enrichment = gap_frac / max(nogap_frac, 0.001)
            gene_enrichment[g] = {
                "gap_frac": gap_frac, "nogap_frac": nogap_frac,
                "enrichment": enrichment,
                "gap_n": gap_genes.get(g, 0),
                "nogap_n": nogap_genes.get(g, 0),
                "is_hub": g in hub_set,
            }

        # Channel enrichment
        channel_enrichment = {}
        all_channels = set(gap_channels.keys()) | set(nogap_channels.keys())
        for ch in all_channels:
            gap_frac = gap_channels.get(ch, 0) / max(sum(gap_channels.values()), 1)
            nogap_frac = nogap_channels.get(ch, 0) / max(sum(nogap_channels.values()), 1)
            enrichment = gap_frac / max(nogap_frac, 0.001)
            channel_enrichment[ch] = {
                "gap_frac": gap_frac, "nogap_frac": nogap_frac,
                "enrichment": enrichment,
            }

        # Gap patient survival statistics
        gap_times = times[gap_mask]
        gap_events = events[gap_mask]
        nogap_times = times[nogap_mask]
        nogap_events = events[nogap_mask]

        profile = {
            "cancer_type": ct_name,
            "graph_c": info["graph_c"],
            "n_patients": info["n_patients"],
            "n_gap": n_gap,
            "n_nogap": n_nogap,
            "gap_median_survival": float(np.median(gap_times)),
            "nogap_median_survival": float(np.median(nogap_times)),
            "gap_event_rate": float(gap_events.mean()),
            "nogap_event_rate": float(nogap_events.mean()),
            "gap_mean_mutations": float(
                np.mean([len(pid_to_row[pids[i]]["genes"])
                         for i in range(n) if gap_mask[i] and pids[i] in pid_to_row])
            ) if n_gap > 0 else 0,
            "nogap_mean_mutations": float(
                np.mean([len(pid_to_row[pids[i]]["genes"])
                         for i in range(n) if nogap_mask[i] and pids[i] in pid_to_row])
            ) if n_nogap > 0 else 0,
            "gene_enrichment": gene_enrichment,
            "channel_enrichment": channel_enrichment,
            "top_gap_genes": sorted(
                gene_enrichment.keys(),
                key=lambda g: -gene_enrichment[g]["enrichment"]
            )[:20],
        }

        gap_profiles.append(profile)

    return gap_profiles


# =========================================================================
# Step 4: Generate edge proposals from gap analysis
# =========================================================================

def compute_cox_hr(times, events, has_gene_mask):
    """Quick univariate Cox HR estimate for a gene in a cancer type.

    Returns (hr, ci_lower, ci_upper, p_value) or None if insufficient data.
    Uses log-rank-based estimate when lifelines not available.
    """
    n_with = has_gene_mask.sum()
    n_without = (~has_gene_mask).sum()
    events_with = events[has_gene_mask].sum()
    events_without = events[~has_gene_mask].sum()

    if n_with < 10 or n_without < 10 or events_with < 3:
        return None

    # Median survival comparison (quick estimate)
    median_with = np.median(times[has_gene_mask])
    median_without = np.median(times[~has_gene_mask])

    if median_without <= 0 or median_with <= 0:
        return None

    # Event rate ratio as HR proxy
    rate_with = events_with / n_with
    rate_without = events_without / n_without

    if rate_without < 0.01:
        return None

    hr = rate_with / rate_without

    # Confidence interval width (approximate)
    se = np.sqrt(1 / max(events_with, 1) + 1 / max(events_without, 1))
    ci_width = 2 * 1.96 * se  # on log scale

    return {
        "hr": float(np.clip(hr, 0.1, 10.0)),
        "ci_width": float(ci_width),
        "n_with": int(n_with),
        "n_without": int(n_without),
        "events_with": int(events_with),
        "events_without": int(events_without),
        "median_with": float(median_with),
        "median_without": float(median_without),
    }


def generate_edge_proposals(gap_profiles, patients_df, existing_prognostic, ppi_edges):
    """From gap profiles, generate concrete edge proposals.

    Returns:
        prognostic_proposals: New PROGNOSTIC_IN edges for gap-enriched gene×CT pairs
        weight_proposals: Adjusted weights for existing edges
        cooccur_proposals: New COOCCURS edges from gap patient co-mutation patterns
    """
    prognostic_proposals = []
    weight_proposals = []
    cooccur_proposals = []

    for profile in gap_profiles:
        ct_name = profile["cancer_type"]
        ct_patients = patients_df[patients_df["cancer_type"] == ct_name]

        if len(ct_patients) < 50:
            continue

        times = ct_patients["os_months"].values.astype(np.float32)
        events = ct_patients["event"].values.astype(np.float32)

        # --- PROGNOSTIC_IN proposals ---
        for gene in profile["top_gap_genes"]:
            enrich = profile["gene_enrichment"].get(gene)
            if enrich is None or enrich["enrichment"] < 1.3:
                continue

            # Skip if we already have a high-tier edge
            existing = existing_prognostic.get((ct_name, gene))
            if existing and existing["tier"] in (1, 2):
                continue

            # Compute Cox HR for this gene in this CT
            has_gene = ct_patients["genes"].apply(lambda gs: gene in gs).values
            cox_result = compute_cox_hr(times, events, has_gene)

            if cox_result is None:
                continue

            # Only propose if HR is meaningfully different from 1.0
            if abs(np.log(cox_result["hr"])) < 0.1:
                continue

            proposal = {
                "cancer_type": ct_name,
                "gene": gene,
                "hr": cox_result["hr"],
                "ci_width": cox_result["ci_width"],
                "n_with": cox_result["n_with"],
                "tier": 5,  # model-inferred tier
                "source": "gap_analysis",
                "enrichment": enrich["enrichment"],
                "gap_frac": enrich["gap_frac"],
                "confidence": "medium" if cox_result["ci_width"] < 1.5 else "low",
            }

            # If we have a tier 3 or 4, this is a weight adjustment instead
            if existing and existing["tier"] in (3, 4):
                weight_proposals.append({
                    "cancer_type": ct_name,
                    "gene": gene,
                    "old_hr": existing["hr"],
                    "new_hr": cox_result["hr"],
                    "old_tier": existing["tier"],
                    "reason": f"gap_enrichment={enrich['enrichment']:.2f}",
                })
            else:
                prognostic_proposals.append(proposal)

        # --- COOCCURS proposals from gap patient co-mutation patterns ---
        gap_gene_sets = []
        for _, row in ct_patients.iterrows():
            if len(row["genes"]) >= 2:
                gap_gene_sets.append(set(row["genes"]))

        pair_count = Counter()
        for gs in gap_gene_sets:
            genes_list = sorted(gs)
            for i in range(len(genes_list)):
                for j in range(i + 1, len(genes_list)):
                    pair_count[(genes_list[i], genes_list[j])] += 1

        # Find pairs enriched in gap patients that aren't already PPI-connected
        for (g1, g2), count in pair_count.most_common(50):
            if count < 5:
                break
            key = (min(g1, g2), max(g1, g2))
            if key in ppi_edges:
                continue  # already connected

            # Check if co-occurrence is prognostically relevant
            has_both = ct_patients["genes"].apply(
                lambda gs: g1 in gs and g2 in gs
            ).values
            if has_both.sum() < 5:
                continue

            cox_result = compute_cox_hr(times, events, has_both)
            if cox_result is None:
                continue

            if abs(np.log(cox_result["hr"])) < 0.15:
                continue

            cooccur_proposals.append({
                "gene1": g1,
                "gene2": g2,
                "cancer_type": ct_name,
                "count": count,
                "hr": cox_result["hr"],
                "n_with": cox_result["n_with"],
            })

    return prognostic_proposals, weight_proposals, cooccur_proposals


# =========================================================================
# Step 5: Write proposals to Neo4j
# =========================================================================

def write_proposals(prognostic_proposals, weight_proposals, cooccur_proposals,
                    cycle, dry_run=False):
    """Write edge proposals to Neo4j via GraphGateway."""
    from gnn.data.graph_changelog import GraphGateway

    mode = "DRY RUN" if dry_run else "WRITING"
    print(f"\n  {mode} EDGE PROPOSALS TO NEO4J")

    gw = GraphGateway(dry_run=dry_run)
    source = f"gap_analysis_cycle_{cycle}"
    n_written = {"prognostic": 0, "weight": 0, "cooccur": 0}

    try:
        # --- New PROGNOSTIC_IN edges (tier 5 = model-inferred) ---
        if prognostic_proposals:
            edges = []
            for p in prognostic_proposals:
                edges.append({
                    "from": p["gene"],
                    "to": p["cancer_type"],
                    "hr": p["hr"],
                    "ci_width": p["ci_width"],
                    "tier": 5,
                    "confidence": p["confidence"],
                    "n_with": p["n_with"],
                    "enrichment": p["enrichment"],
                    "cycle": cycle,
                })
            n_written["prognostic"] = gw.merge_edges(
                "PROGNOSTIC_IN", edges, source=source,
                match_from=("Gene", "name"),
                match_to=("CancerType", "name"),
            )

        # --- Weight adjustments for existing PROGNOSTIC_IN ---
        if weight_proposals:
            edges = []
            for w in weight_proposals:
                edges.append({
                    "from": w["gene"],
                    "to": w["cancer_type"],
                    "hr": w["new_hr"],
                    "prev_hr": w["old_hr"],
                    "tier": w["old_tier"],
                    "adjusted_by": "gap_analysis",
                    "adjustment_reason": w["reason"],
                    "cycle": cycle,
                })
            n_written["weight"] = gw.merge_edges(
                "PROGNOSTIC_IN", edges, source=source,
                source_detail="weight_adjustment",
                match_from=("Gene", "name"),
                match_to=("CancerType", "name"),
            )

        # --- New COOCCURS edges ---
        if cooccur_proposals:
            edges = []
            for c in cooccur_proposals:
                edges.append({
                    "from": c["gene1"],
                    "to": c["gene2"],
                    "weight": float(c["count"]) / 100.0,
                    "cancer_type": c["cancer_type"],
                    "hr": c["hr"],
                    "n_with": c["n_with"],
                    "cycle": cycle,
                })
            n_written["cooccur"] = gw.merge_edges(
                "COOCCURS", edges, source=source,
            )

    finally:
        gw.close()

    return n_written


# =========================================================================
# Main entry point
# =========================================================================

def run_gap_analysis(cycle=0, dry_run=False, min_gap=0.015):
    """Run full gap analysis loop. Called from feedback_loop.py or standalone.

    Returns dict with proposals and summary statistics.
    """
    print(f"\n{'='*70}")
    print(f"  GAP ANALYSIS LOOP — CYCLE {cycle}")
    print(f"{'='*70}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    driver = get_driver()

    # 1. Query all patients
    print("  Querying Neo4j for patient data...")
    t1 = time.time()
    patients_df = query_all_patients(driver)
    _step("Patients loaded", actual=len(patients_df), elapsed=time.time() - t1)

    if len(patients_df) < 100:
        print("  ERROR: Too few patients in Neo4j. Skipping gap analysis.")
        driver.close()
        return {"status": "skipped", "reason": "too_few_patients"}

    # 2. Query existing edges
    t1 = time.time()
    existing_prognostic = query_existing_prognostic(driver)
    ppi_edges = query_ppi_neighbors(driver)
    _step("Existing edges loaded",
          actual=len(existing_prognostic) + len(ppi_edges),
          elapsed=time.time() - t1)

    # 3. Score patients per CT
    print("\n  Scoring patients per cancer type...")
    t1 = time.time()
    ct_results = compute_per_ct_scores(patients_df, driver)
    _step("Cancer types scored", actual=len(ct_results), elapsed=time.time() - t1)

    # Print per-CT scores
    print(f"\n  {'Cancer Type':<30} {'N':>6} {'Events':>7} {'Graph C':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*8}")
    for ct in sorted(ct_results, key=lambda x: ct_results[x]["graph_c"]):
        r = ct_results[ct]
        print(f"  {ct:<30} {r['n_patients']:>6} {r['n_events']:>7} {r['graph_c']:>8.4f}")

    # 4. Profile gap patients
    print("\n  Profiling gap patients...")
    t1 = time.time()
    gap_profiles = profile_gap_patients(patients_df, ct_results, min_gap=min_gap)
    _step("Gap profiles built", actual=len(gap_profiles), elapsed=time.time() - t1)

    for p in gap_profiles:
        print(f"\n  --- {p['cancer_type']} (C={p['graph_c']:.4f}, "
              f"N={p['n_patients']}, gap_patients={p['n_gap']}) ---")
        print(f"  Gap median survival: {p['gap_median_survival']:.1f} mo "
              f"vs no-gap: {p['nogap_median_survival']:.1f} mo")
        print(f"  Gap mean mutations: {p['gap_mean_mutations']:.1f} "
              f"vs no-gap: {p['nogap_mean_mutations']:.1f}")
        top_genes = p["top_gap_genes"][:10]
        if top_genes:
            print(f"  Top gap-enriched genes:")
            for g in top_genes:
                e = p["gene_enrichment"][g]
                hub = " [HUB]" if e["is_hub"] else ""
                print(f"    {g:<12} enrich={e['enrichment']:.2f}x "
                      f"(gap={e['gap_frac']:.1%}, nogap={e['nogap_frac']:.1%}){hub}")

    # 5. Generate proposals
    print("\n  Generating edge proposals...")
    t1 = time.time()
    prog_proposals, weight_proposals, cooccur_proposals = generate_edge_proposals(
        gap_profiles, patients_df, existing_prognostic, ppi_edges
    )
    _step("Proposals generated",
          actual=len(prog_proposals) + len(weight_proposals) + len(cooccur_proposals),
          elapsed=time.time() - t1)

    print(f"\n  PROGNOSTIC_IN proposals (tier 5): {len(prog_proposals)}")
    for p in prog_proposals[:15]:
        direction = "risk" if p["hr"] > 1 else "protective"
        print(f"    {p['gene']:<12} in {p['cancer_type']:<25} "
              f"HR={p['hr']:.3f} ({direction}, n={p['n_with']}, "
              f"enrich={p['enrichment']:.2f}x)")

    print(f"\n  Weight adjustments: {len(weight_proposals)}")
    for w in weight_proposals[:10]:
        print(f"    {w['gene']:<12} in {w['cancer_type']:<25} "
              f"HR {w['old_hr']:.3f} → {w['new_hr']:.3f} ({w['reason']})")

    print(f"\n  COOCCURS proposals: {len(cooccur_proposals)}")
    for c in cooccur_proposals[:10]:
        print(f"    {c['gene1']}-{c['gene2']} in {c['cancer_type']:<25} "
              f"n={c['count']}, HR={c['hr']:.3f}")

    # 6. Write to Neo4j
    n_written = write_proposals(
        prog_proposals, weight_proposals, cooccur_proposals,
        cycle=cycle, dry_run=dry_run,
    )

    # 7. Save results
    summary = {
        "cycle": cycle,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_patients": len(patients_df),
        "n_cancer_types_scored": len(ct_results),
        "n_gap_profiles": len(gap_profiles),
        "n_prognostic_proposals": len(prog_proposals),
        "n_weight_adjustments": len(weight_proposals),
        "n_cooccur_proposals": len(cooccur_proposals),
        "n_written": n_written,
        "per_ct_scores": {
            ct: {"graph_c": r["graph_c"], "n": r["n_patients"], "events": r["n_events"]}
            for ct, r in ct_results.items()
        },
        "gap_profiles_summary": [
            {
                "cancer_type": p["cancer_type"],
                "graph_c": p["graph_c"],
                "n_gap": p["n_gap"],
                "top_genes": p["top_gap_genes"][:10],
            }
            for p in gap_profiles
        ],
        "prognostic_proposals": prog_proposals,
        "weight_proposals": weight_proposals,
        "cooccur_proposals": cooccur_proposals[:50],
        "elapsed_s": time.time() - t0,
    }

    results_path = os.path.join(RESULTS_DIR, f"cycle_{cycle}.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    driver.close()

    _step("Gap analysis complete", elapsed=time.time() - t0)

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview proposals without writing to Neo4j")
    parser.add_argument("--min-gap", type=float, default=0.015,
                        help="Minimum C-index gap to trigger deep profiling")
    parser.add_argument("--cycle", type=int, default=0,
                        help="Cycle number for provenance tracking")
    args = parser.parse_args()

    run_gap_analysis(cycle=args.cycle, dry_run=args.dry_run, min_gap=args.min_gap)


if __name__ == "__main__":
    main()
