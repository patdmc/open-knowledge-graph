#!/usr/bin/env python3
"""
Walk the knowledge graph from a mutation and summarize everything connected.

Given a gene (and optionally protein change + cancer type), traverses all
edges to build a complete profile: where it shows up, what it's sensitive to,
who has it, what happens to them.

Usage:
    python3 -m gnn.scripts.mutation_profile TP53
    python3 -m gnn.scripts.mutation_profile TP53 --variant R175H
    python3 -m gnn.scripts.mutation_profile BRCA2 --cancer BRCA
    python3 -m gnn.scripts.mutation_profile KRAS --variant G12C --cancer LUAD
"""

import sys, os, argparse
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("gene", type=str, help="Gene symbol (e.g., TP53, KRAS, BRCA2)")
    parser.add_argument("--variant", type=str, default=None, help="Protein change (e.g., R175H, G12C)")
    parser.add_argument("--cancer", type=str, default=None, help="Cancer type (e.g., BRCA, LUAD)")
    return parser.parse_args()


def walk_gene(driver, gene, variant=None, cancer=None):
    """Walk all edges from a gene node and collect everything."""
    data = {}

    with driver.session() as session:

        # --- Gene properties ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})
            RETURN g.function AS function, g.position AS position,
                   g.is_hub AS is_hub, g.primary_channel AS channel,
                   g.profile_entropy AS entropy,
                   g.channel_profile AS profile,
                   g.civic_function AS civic_function
        """, gene=gene)
        rec = result.single()
        if rec is None:
            return None
        data["gene"] = {
            "name": gene,
            "function": rec["function"],
            "position": rec["position"],
            "is_hub": rec["is_hub"],
            "channel": rec["channel"],
            "entropy": rec["entropy"],
            "profile": rec["profile"],
            "civic_function": rec.get("civic_function"),
        }

        # --- Channel membership ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[b:BELONGS_TO]->(c:Channel)
            RETURN c.name AS channel, b.weight AS weight, c.tier_name AS tier
            ORDER BY b.weight DESC
        """, gene=gene)
        data["channels"] = [dict(r) for r in result]

        # --- PPI neighbors ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[p:PPI]-(neighbor:Gene)
            RETURN neighbor.name AS gene, p.score AS score,
                   neighbor.primary_channel AS channel,
                   neighbor.function AS function
            ORDER BY p.score DESC
            LIMIT 20
        """, gene=gene)
        data["ppi"] = [dict(r) for r in result]

        # --- SL partners ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[:SL_PARTNER]->(partner:Gene)
            RETURN partner.name AS gene, partner.primary_channel AS channel,
                   partner.function AS function
        """, gene=gene)
        data["sl_partners"] = [dict(r) for r in result]

        # --- Co-occurring mutations ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[c:COOCCURS]-(other:Gene)
            RETURN other.name AS gene, c.count AS count, c.cancer_type AS cancer_type
            ORDER BY c.count DESC
            LIMIT 30
        """, gene=gene)
        data["cooccurs"] = [dict(r) for r in result]

        # --- Drug sensitivity (GDSC) ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[r:SENSITIVE_TO]->(d:Drug)
            RETURN d.name AS drug, r.effect_size AS effect_size,
                   r.n_mut AS n_mut, r.source AS source
            ORDER BY r.effect_size ASC
        """, gene=gene)
        data["sensitive_to"] = [dict(r) for r in result]

        # --- Drug resistance (GDSC) ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[r:RESISTANT_TO]->(d:Drug)
            RETURN d.name AS drug, r.effect_size AS effect_size,
                   r.n_mut AS n_mut, r.source AS source
            ORDER BY r.effect_size DESC
        """, gene=gene)
        data["resistant_to"] = [dict(r) for r in result]

        # --- CIViC evidence ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[r:HAS_SENSITIVITY_EVIDENCE]->(g)
            RETURN r.protein_change AS variant, r.source AS source
        """, gene=gene)
        data["civic_sensitivity"] = [dict(r) for r in result]

        result = session.run("""
            MATCH (g:Gene {name: $gene})-[r:HAS_RESISTANCE_EVIDENCE]->(g)
            RETURN r.protein_change AS variant, r.source AS source
        """, gene=gene)
        data["civic_resistance"] = [dict(r) for r in result]

        # --- Essential in lineages (DepMap) ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[r:ESSENTIAL_IN]->(l:Lineage)
            RETURN l.name AS lineage, r.dependency_score AS score
            ORDER BY r.dependency_score ASC
        """, gene=gene)
        data["essential_in"] = [dict(r) for r in result]

        # --- Expression across cancer types ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[r:EXPRESSION_IN]->(ct:CancerType)
            RETURN ct.name AS cancer_type, r.z_score AS z_score,
                   r.direction AS direction, r.n_samples AS n_samples
            ORDER BY r.z_score DESC
        """, gene=gene)
        data["expression"] = [dict(r) for r in result]

        # --- CNA across cancer types ---
        result = session.run("""
            MATCH (g:Gene {name: $gene})-[r:CNA_IN]->(ct:CancerType)
            RETURN ct.name AS cancer_type, r.amp_freq AS amp_freq,
                   r.del_freq AS del_freq, r.direction AS direction
            ORDER BY r.amp_freq + r.del_freq DESC
        """, gene=gene)
        data["cna"] = [dict(r) for r in result]

        # --- Patient outcomes (aggregated by cancer type) ---
        ct_filter = ""
        params = {"gene": gene}
        if cancer:
            ct_filter = " AND p.cancer_type = $cancer"
            params["cancer"] = cancer

        result = session.run(f"""
            MATCH (p:Patient)-[m:HAS_MUTATION]->(g:Gene {{name: $gene}})
            WHERE true {ct_filter}
            WITH p.cancer_type AS ct, p.os_months AS os, p.event AS event,
                 m.direction AS direction, m.protein_change AS pc
            RETURN ct, count(*) AS n_patients,
                   sum(event) AS n_events,
                   avg(os) AS mean_os,
                   percentileDisc(os, 0.5) AS median_os,
                   collect(DISTINCT direction) AS directions,
                   collect(DISTINCT pc)[0..5] AS sample_variants
            ORDER BY n_patients DESC
        """, **params)
        data["patient_outcomes"] = [dict(r) for r in result]

        # --- Treatment received by patients with this mutation ---
        result = session.run(f"""
            MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene {{name: $gene}})
            WHERE true {ct_filter}
            OPTIONAL MATCH (p)-[:RECEIVED]->(t:Treatment)
            WITH t.name AS treatment, p.event AS event, p.os_months AS os
            WHERE treatment IS NOT NULL
            RETURN treatment, count(*) AS n_patients,
                   sum(event) AS n_events,
                   avg(os) AS mean_os,
                   percentileDisc(os, 0.5) AS median_os
            ORDER BY n_patients DESC
        """, **params)
        data["treatment_outcomes"] = [dict(r) for r in result]

        # --- Variant-specific (if provided) ---
        if variant:
            result = session.run(f"""
                MATCH (p:Patient)-[m:HAS_MUTATION {{protein_change: $variant}}]->(g:Gene {{name: $gene}})
                WHERE true {ct_filter}
                WITH p.cancer_type AS ct, p.os_months AS os, p.event AS event
                RETURN ct, count(*) AS n_patients,
                       sum(event) AS n_events,
                       avg(os) AS mean_os,
                       percentileDisc(os, 0.5) AS median_os
                ORDER BY n_patients DESC
            """, gene=gene, variant=variant, **({k: v for k, v in params.items() if k != "gene"}))
            data["variant_outcomes"] = [dict(r) for r in result]

    return data


def format_profile(data, gene, variant=None, cancer=None):
    """Format the walked data into a readable summary."""
    lines = []
    g = data["gene"]

    # Header
    title = gene
    if variant:
        title += f":{variant}"
    if cancer:
        title += f" in {cancer}"

    lines.append(f"\n{'='*70}")
    lines.append(f"  MUTATION PROFILE: {title}")
    lines.append(f"{'='*70}")

    # Identity
    lines.append(f"\n--- Identity ---")
    lines.append(f"  Gene: {g['name']}")
    lines.append(f"  Function: {g['function'] or 'unknown'}" +
                 (f" (CIViC: {g['civic_function']})" if g.get('civic_function') else ""))
    lines.append(f"  Position: {g['position']} ({'HUB' if g['is_hub'] else 'leaf'})")
    lines.append(f"  Primary channel: {g['channel']}")
    lines.append(f"  Entropy: {g['entropy']:.2f} bits" if g['entropy'] else "  Entropy: N/A")

    # Channels
    if data["channels"]:
        lines.append(f"\n--- Channel Membership ---")
        for ch in data["channels"]:
            bar = "#" * int(ch["weight"] * 20)
            lines.append(f"  {ch['channel']:20s} {ch['weight']:.3f} {bar}  ({ch['tier']})")

    # Where it shows up (patient outcomes)
    if data["patient_outcomes"]:
        lines.append(f"\n--- Where It Shows Up (patient data) ---")
        total_patients = sum(r["n_patients"] for r in data["patient_outcomes"])
        total_events = sum(r["n_events"] for r in data["patient_outcomes"])
        lines.append(f"  Total: {total_patients} patients, {total_events} events")
        lines.append(f"  {'Cancer Type':<25s} {'N':>5s} {'Events':>7s} {'Med OS':>8s} {'Variants'}")
        lines.append(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*8} {'-'*20}")
        for r in data["patient_outcomes"]:
            med_os = f"{r['median_os']:.0f}mo" if r["median_os"] else "N/A"
            variants = ", ".join(str(v) for v in (r.get("sample_variants") or []) if v)
            lines.append(f"  {r['ct']:<25s} {r['n_patients']:>5d} {r['n_events']:>7d} {med_os:>8s} {variants[:30]}")

    # Variant-specific
    if variant and data.get("variant_outcomes"):
        lines.append(f"\n--- Variant {variant} Specifically ---")
        for r in data["variant_outcomes"]:
            med_os = f"{r['median_os']:.0f}mo" if r["median_os"] else "N/A"
            lines.append(f"  {r['ct']:<25s} N={r['n_patients']}, events={r['n_events']}, median OS={med_os}")

    # When it's bad vs protective
    if data["patient_outcomes"]:
        lines.append(f"\n--- Prognostic Signal ---")
        for r in data["patient_outcomes"]:
            if r["n_patients"] >= 20 and r["n_events"] >= 5:
                event_rate = r["n_events"] / r["n_patients"]
                med_os = r["median_os"]
                signal = "HARMFUL" if event_rate > 0.5 else "MIXED" if event_rate > 0.3 else "FAVORABLE"
                lines.append(f"  {r['ct']:<25s} event rate={event_rate:.0%}, median OS={med_os:.0f}mo -> {signal}")

    # Treatment outcomes
    if data["treatment_outcomes"]:
        lines.append(f"\n--- Treatment Outcomes (patients with {gene}) ---")
        lines.append(f"  {'Treatment':<20s} {'N':>5s} {'Events':>7s} {'Event%':>7s} {'Med OS':>8s}")
        lines.append(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*8}")
        for r in data["treatment_outcomes"]:
            med_os = f"{r['median_os']:.0f}mo" if r["median_os"] else "N/A"
            event_pct = f"{100*r['n_events']/max(r['n_patients'],1):.0f}%"
            lines.append(f"  {r['treatment']:<20s} {r['n_patients']:>5d} {r['n_events']:>7d} {event_pct:>7s} {med_os:>8s}")

    # Drug sensitivity
    if data["sensitive_to"]:
        lines.append(f"\n--- Drug Sensitivity (mutation = MORE sensitive) ---")
        for d in data["sensitive_to"][:10]:
            effect = d["effect_size"]
            source = d.get("source", "")
            n = d.get("n_mut", "?")
            lines.append(f"  {d['drug']:<25s} effect={effect:+.2f}  n_mut={n}  [{source}]")

    # Drug resistance
    if data["resistant_to"]:
        lines.append(f"\n--- Drug Resistance (mutation = LESS sensitive) ---")
        for d in data["resistant_to"][:10]:
            effect = d["effect_size"]
            source = d.get("source", "")
            n = d.get("n_mut", "?")
            lines.append(f"  {d['drug']:<25s} effect={effect:+.2f}  n_mut={n}  [{source}]")

    # CIViC variant evidence
    if data["civic_sensitivity"] or data["civic_resistance"]:
        lines.append(f"\n--- CIViC Clinical Evidence ---")
        for e in data["civic_sensitivity"]:
            lines.append(f"  SENSITIVITY: {e['variant']} [{e['source']}]")
        for e in data["civic_resistance"]:
            lines.append(f"  RESISTANCE:  {e['variant']} [{e['source']}]")

    # SL partners
    if data["sl_partners"]:
        lines.append(f"\n--- Synthetic Lethality Partners ---")
        lines.append(f"  If {gene} is mutated AND one of these is lost, the cell dies:")
        for p in data["sl_partners"]:
            lines.append(f"  -> {p['gene']:<15s} ({p['channel']}, {p['function']})")

    # Essential in (DepMap)
    if data["essential_in"]:
        lines.append(f"\n--- Essential In (DepMap CRISPR) ---")
        lines.append(f"  Knocking out {gene} kills cells in:")
        for e in data["essential_in"]:
            lines.append(f"  -> {e['lineage']:<30s} dependency={e['score']:.3f}")

    # Expression
    if data["expression"]:
        lines.append(f"\n--- Expression Pattern ---")
        over = [e for e in data["expression"] if e["z_score"] > 0]
        under = [e for e in data["expression"] if e["z_score"] < 0]
        if over:
            over_parts = [f"{e['cancer_type']}(z={e['z_score']:.1f})" for e in over[:5]]
            lines.append(f"  Over-expressed in: {', '.join(over_parts)}")
        if under:
            under_parts = [f"{e['cancer_type']}(z={e['z_score']:.1f})" for e in under[:5]]
            lines.append(f"  Under-expressed in: {', '.join(under_parts)}")

    # CNA
    if data["cna"]:
        lines.append(f"\n--- Copy Number Pattern ---")
        for c in data["cna"][:5]:
            lines.append(f"  {c['cancer_type']:<15s} {c['direction']:>10s} "
                         f"(amp={c['amp_freq']:.0%}, del={c['del_freq']:.0%})")

    # Co-occurring mutations
    if data["cooccurs"]:
        lines.append(f"\n--- Top Co-occurring Mutations ---")
        # Aggregate across cancer types
        gene_counts = defaultdict(int)
        for c in data["cooccurs"]:
            gene_counts[c["gene"]] += c["count"]
        sorted_genes = sorted(gene_counts.items(), key=lambda x: -x[1])
        for g_name, count in sorted_genes[:10]:
            lines.append(f"  {g_name:<15s} co-occurs {count:>5d} times")

    # PPI
    if data["ppi"]:
        lines.append(f"\n--- Protein Interactions (STRING) ---")
        for p in data["ppi"][:10]:
            lines.append(f"  {p['gene']:<15s} score={p['score']:.3f}  "
                         f"({p['channel']}, {p['function']})")

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


def main():
    args = parse_args()
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    data = walk_gene(driver, args.gene, args.variant, args.cancer)
    if data is None:
        print(f"Gene '{args.gene}' not found in graph.")
        driver.close()
        return

    print(format_profile(data, args.gene, args.variant, args.cancer))
    driver.close()


if __name__ == "__main__":
    main()
