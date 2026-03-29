#!/usr/bin/env python3
"""
Detailed ridge traversal walk for glioma patients.
Shows step-by-step how the graph scorer computes its prediction
and where it diverges from the V6c transformer.

Usage:
    python3 -u -m gnn.scripts.glioma_ridge_walk
"""

import sys, os, json, numpy as np, networkx as nx, torch, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import (
    GNN_RESULTS, GNN_CACHE, ANALYSIS_CACHE,
    HUB_GENES, GENE_POSITION, GENE_FUNCTION, TRUNCATING,
)
from gnn.data.channel_dataset_v6c import build_channel_features_v6c
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_TIER_MAP, CHANNEL_FEAT_DIM_V6,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from sklearn.model_selection import StratifiedKFold


def main():
    # Build graph
    with open(os.path.join(GNN_CACHE, "string_ppi_edges.json")) as f:
        ppi = json.load(f)

    G = nx.Graph()
    for gene in V6_CHANNEL_MAP:
        ch = V6_CHANNEL_MAP[gene]
        G.add_node(gene, channel=ch, tier=V6_TIER_MAP.get(ch, -1))
    for ch_name in ppi:
        for src, tgt, score in ppi[ch_name]:
            if src in V6_CHANNEL_MAP and tgt in V6_CHANNEL_MAP:
                G.add_edge(src, tgt, weight=score / 1000.0)

    # Load data
    data = build_channel_features_v6c("msk_impact_50k")
    N = len(data["times"])
    events = data["events"].numpy().astype(int)
    times = data["times"].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ct_vocab = data["cancer_type_vocab"]
    ct_idx = data["cancer_type_idx"].numpy()

    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)
    config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6, "hidden_dim": 128,
        "cross_channel_heads": 4, "cross_channel_layers": 2,
        "dropout": 0.3, "n_cancer_types": len(ct_vocab),
    }
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue
        model = ChannelNetV6c(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        for start in range(0, len(val_idx), 2048):
            end = min(start + 2048, len(val_idx))
            idx = val_idx[start:end]
            batch = {k: data[k][idx] for k in [
                "channel_features", "tier_features", "cancer_type_idx",
                "age", "sex", "msi_score", "msi_high", "tmb",
            ]}
            with torch.no_grad():
                h = model(batch)
            all_hazards[idx] = h
            all_in_val[idx] = True

    hazards = all_hazards.numpy()
    valid_mask = all_in_val.numpy()
    baseline = hazards[valid_mask].mean()

    # Load mutations
    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol", "proteinChange", "mutationType"],
    )
    clin = pd.read_csv(os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"), low_memory=False)
    clin = clin[clin["OS_MONTHS"].notna() & clin["OS_STATUS"].notna()].copy()
    clin["time"] = clin["OS_MONTHS"].astype(float)
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)
    clin = clin[clin["time"] > 0]
    patient_ids = clin["patientId"].unique()
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    channel_genes = set(V6_CHANNEL_MAP.keys())
    mut = mut[mut["gene.hugoGeneSymbol"].isin(channel_genes)].copy()
    mut["patient_idx"] = mut["patientId"].map(pid_to_idx)
    mut = mut[mut["patient_idx"].notna()]
    mut["patient_idx"] = mut["patient_idx"].astype(int)

    # Gene shifts
    gene_patients_map = defaultdict(set)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            gene_patients_map[row["gene.hugoGeneSymbol"]].add(idx)
    gene_shift = {}
    for gene, pts in gene_patients_map.items():
        pts_arr = np.array(sorted(pts))
        pts_valid = pts_arr[valid_mask[pts_arr]]
        if len(pts_valid) >= 20:
            gene_shift[gene] = float(hazards[pts_valid].mean() - baseline)

    # Per-patient mutations
    patient_muts = defaultdict(list)
    for _, row in mut.iterrows():
        idx = int(row["patient_idx"])
        if 0 <= idx < N:
            patient_muts[idx].append({
                "gene": row["gene.hugoGeneSymbol"],
                "protein": row["proteinChange"] if pd.notna(row["proteinChange"]) else "",
                "type": row["mutationType"] if pd.notna(row["mutationType"]) else "",
            })

    # Find glioma
    glioma_ct_idx = None
    for i, name in enumerate(ct_vocab):
        if "lioma" in name.lower() or "Glioma" in name:
            glioma_ct_idx = i
            print(f"Glioma: idx={i}, name='{name}'")
            break

    glioma_patients = [
        idx for idx in range(N)
        if int(ct_idx[idx]) == glioma_ct_idx
        and valid_mask[idx]
        and len(patient_muts.get(idx, [])) >= 2
    ]
    print(f"Glioma patients with 2+ mutations: {len(glioma_patients)}")

    # Build examples sorted by divergence
    examples = []
    for idx in glioma_patients:
        muts_list = patient_muts[idx]
        genes = set(m["gene"] for m in muts_list)
        gs = sum(gene_shift.get(g, 0) for g in genes)
        v6c = hazards[idx]
        div = gs - v6c
        has_idh = any("IDH" in m["gene"] for m in muts_list)
        has_tp53 = any("TP53" in m["gene"] for m in muts_list)
        examples.append((idx, muts_list, genes, gs, v6c, div, has_idh, has_tp53,
                         times[idx], events[idx]))

    examples.sort(key=lambda x: -abs(x[5]))

    # Pick cases
    idh_pos = [e for e in examples if e[6] and abs(e[5]) > 0.3]
    idh_neg = [e for e in examples if not e[6] and abs(e[5]) > 0.3]
    agree = [e for e in examples if abs(e[5]) < 0.15 and len(e[2]) >= 3]

    picks = []
    if idh_pos:
        picks.append(("IDH+ DIVERGENT", idh_pos[0]))
    if len(idh_pos) > 3:
        picks.append(("IDH+ DIVERGENT #2", idh_pos[3]))
    if idh_neg:
        picks.append(("IDH- DIVERGENT", idh_neg[0]))
    if agree:
        picks.append(("AGREE (graph ~ V6c)", agree[0]))

    # Ridge coefficients from the traversal run
    coefs = {
        "gene_shift": 0.2367,
        "total_isolation": 0.0686,
        "total_hub_damage": 0.0164,
        "channels_severed": -0.0447,
        "hub_shift_sum": 0.2367,
        "tier_connectivity": -2.2289,
        "lcc_fraction": 1.2658,
        "n_mutated": 0.1518,
    }

    tier_names = {
        0: "Cell Intrinsic",
        1: "Tissue Level",
        2: "Organism Level",
        3: "Meta Regulatory",
    }

    for label, (idx, muts_list, genes, gs, v6c_pred, divergence, has_idh, has_tp53, t, e) in picks:
        outcome = "DECEASED" if e else "ALIVE"
        direction = "graph thinks WORSE" if divergence > 0 else "graph thinks BETTER"

        print(f"\n{'=' * 90}")
        print(f"  PATIENT {idx}: {label}")
        print(f"  Survival: {t:.1f} months, {outcome}")
        print(f"  V6c prediction: {v6c_pred:+.3f} (hazard)")
        print(f"  Gene shift sum:  {gs:+.3f}")
        print(f"  Divergence:      {divergence:+.3f} ({direction})")
        print(f"{'=' * 90}")

        # --- Mutations ---
        print(f"\n  MUTATIONS:")
        for m in sorted(muts_list, key=lambda x: x["gene"]):
            g = m["gene"]
            ch = V6_CHANNEL_MAP.get(g, "?")
            tier = V6_TIER_MAP.get(ch, -1)
            shift = gene_shift.get(g, None)
            pos = GENE_POSITION.get(g, "?")
            func = GENE_FUNCTION.get(g, "?")
            is_trunc = m["type"] in TRUNCATING if m["type"] else False
            shift_str = f"{shift:+.3f}" if shift is not None else "  n/a "
            trunc_str = "TRUNC" if is_trunc else ""
            print(f"    {g:<12} {m['protein']:<20} {m['type']:<20} "
                  f"ch={ch:<14} tier={tier} {pos:<5} {func:<4} "
                  f"shift={shift_str} {trunc_str}")

        # --- Graph Traversal Walk ---
        print(f"\n  GRAPH TRAVERSAL WALK:")
        mutated = genes & set(V6_CHANNEL_MAP.keys())

        # Step 1: per-channel damage
        print(f"\n  Step 1: PER-CHANNEL DAMAGE")
        total_isolation = 0
        total_hub_damage = 0
        channels_severed = 0
        hub_shift_sum_val = 0

        for ch_name in V6_CHANNEL_NAMES:
            ch_genes_all = [g for g, c in V6_CHANNEL_MAP.items() if c == ch_name]
            ch_hubs = HUB_GENES.get(ch_name, set())
            ch_mutated = mutated & set(ch_genes_all)

            if not ch_mutated:
                continue

            tier = V6_TIER_MAP.get(ch_name, -1)
            G_ch = G.subgraph(ch_genes_all).copy()

            print(f"\n    {ch_name} (Tier {tier}):")
            print(f"      Mutated: {', '.join(sorted(ch_mutated))}")
            print(f"      Hubs: {', '.join(sorted(ch_hubs))}")

            # Show each mutated gene's connections
            for mg in sorted(ch_mutated):
                neighbors = sorted(G_ch.neighbors(mg)) if mg in G_ch else []
                is_hub = mg in ch_hubs
                hub_nbrs = [n for n in neighbors if n in ch_hubs]

                tag = "[HUB]" if is_hub else "[leaf]"
                print(f"      REMOVING {mg} {tag} (degree={len(neighbors)})")
                if neighbors:
                    print(f"        Severs edges to: {', '.join(neighbors)}")
                    if hub_nbrs:
                        print(f"        !! Disconnects from hub(s): {', '.join(hub_nbrs)}")
                else:
                    print(f"        (isolated node — no edges to sever)")

            # After removal
            G_damaged = G_ch.copy()
            G_damaged.remove_nodes_from(ch_mutated)

            surviving_hubs = ch_hubs - ch_mutated
            reachable = set()
            for hub in surviving_hubs:
                if hub in G_damaged:
                    reachable |= set(nx.descendants(G_damaged, hub)) | {hub}

            surviving = set(ch_genes_all) - ch_mutated
            isolated_after = surviving - reachable

            if surviving:
                reach_pct = 100 * len(reachable & surviving) / len(surviving)
                iso_frac = len(isolated_after) / len(surviving)
            else:
                reach_pct = 0
                iso_frac = 1.0

            comp_after = nx.number_connected_components(G_damaged) if G_damaged.number_of_nodes() > 0 else 0

            print(f"      After removal:")
            print(f"        Reachable from surviving hubs: {reach_pct:.0f}%")
            print(f"        Components: {comp_after}")
            if isolated_after and len(isolated_after) <= 8:
                print(f"        Isolated genes: {', '.join(sorted(isolated_after))}")
            elif isolated_after:
                print(f"        Isolated genes: {len(isolated_after)} cut off")

            total_isolation += iso_frac
            hub_hit = len(ch_mutated & ch_hubs)
            total_hub_damage += hub_hit / max(len(ch_hubs), 1)
            if len(ch_mutated) >= 2 or hub_hit > 0:
                channels_severed += 1
            for g in ch_mutated:
                if g in gene_shift:
                    hub_shift_sum_val += gene_shift[g]

        # Step 2: tier escalation paths
        print(f"\n  Step 2: TIER ESCALATION PATHS")
        G_damaged_full = G.copy()
        G_damaged_full.remove_nodes_from(mutated)

        tier_genes = defaultdict(set)
        for g, ch in V6_CHANNEL_MAP.items():
            if g not in mutated:
                tier_genes[V6_TIER_MAP.get(ch, -1)].add(g)

        tier_connected = 0
        tier_tested = 0
        for t1 in range(4):
            for t2 in range(t1 + 1, 4):
                found = False
                path_example = None
                for g1 in sorted(tier_genes.get(t1, set()))[:10]:
                    for g2 in sorted(tier_genes.get(t2, set()))[:10]:
                        if g1 in G_damaged_full and g2 in G_damaged_full:
                            if nx.has_path(G_damaged_full, g1, g2):
                                found = True
                                path = nx.shortest_path(G_damaged_full, g1, g2)
                                if path_example is None or len(path) < len(path_example):
                                    path_example = path
                    if found:
                        break
                tier_tested += 1
                if found:
                    tier_connected += 1

                status = "INTACT" if found else "SEVERED"
                t1n = tier_names[t1]
                t2n = tier_names[t2]
                print(f"    Tier {t1} ({t1n}) -> Tier {t2} ({t2n}): {status}")
                if path_example and len(path_example) <= 6:
                    path_str = " -> ".join(
                        f"{g}[{V6_CHANNEL_MAP.get(g, '?')}]" for g in path_example
                    )
                    print(f"      Shortest path: {path_str}")

        tier_conn = tier_connected / max(tier_tested, 1)

        # LCC
        if G_damaged_full.number_of_nodes() > 0:
            lcc = max(nx.connected_components(G_damaged_full), key=len)
            lcc_frac = len(lcc) / G.number_of_nodes()
        else:
            lcc_frac = 0
        n_mut = len(mutated)

        # Step 3: Ridge score
        print(f"\n  Step 3: RIDGE SCORE COMPUTATION")

        features = {
            "gene_shift": gs,
            "total_isolation": total_isolation,
            "total_hub_damage": total_hub_damage,
            "channels_severed": channels_severed,
            "hub_shift_sum": hub_shift_sum_val,
            "tier_connectivity": tier_conn,
            "lcc_fraction": lcc_frac,
            "n_mutated": n_mut,
        }

        ridge_score = sum(features[k] * coefs[k] for k in coefs)

        print(f"\n    {'Feature':<25} {'Value':>8} {'Coef':>8} {'Contrib':>8}")
        print(f"    {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8}")
        for k in coefs:
            contrib = features[k] * coefs[k]
            print(f"    {k:<25} {features[k]:>8.3f} {coefs[k]:>+8.4f} {contrib:>+8.4f}")

        print(f"    {'':25} {'':8} {'':8} {'------':>8}")
        print(f"    {'RIDGE SCORE':<25} {'':8} {'':8} {ridge_score:>+8.4f}")
        print(f"    {'V6c PREDICTION':<25} {'':8} {'':8} {v6c_pred:>+8.4f}")
        print(f"    {'DELTA':<25} {'':8} {'':8} {ridge_score - v6c_pred:>+8.4f}")

        # Who is right?
        actual_outcome = "bad" if e == 1 and t < 36 else "good" if t > 60 or e == 0 else "moderate"
        print(f"\n    Actual outcome: {actual_outcome} ({t:.0f}mo, {outcome})")
        if ridge_score > v6c_pred + 0.2:
            print(f"    Ridge says WORSE than V6c.")
            if actual_outcome == "bad":
                print(f"    --> Ridge was RIGHT: patient did poorly.")
            else:
                print(f"    --> Ridge was WRONG: patient survived. V6c was closer.")
        elif ridge_score < v6c_pred - 0.2:
            print(f"    Ridge says BETTER than V6c.")
            if actual_outcome == "good":
                print(f"    --> Ridge was RIGHT: patient survived.")
            else:
                print(f"    --> Ridge was WRONG: patient did poorly. V6c was closer.")

        # Step 4: What V6c knows
        print(f"\n  Step 4: WHAT THE GRAPH MISSES")
        if has_idh:
            idh_shift = gene_shift.get("IDH1", gene_shift.get("IDH2", None))
            idh_str = f"{idh_shift:+.3f}" if idh_shift else "n/a"
            print(f"    ** IDH mutation present.")
            print(f"       Global IDH gene shift: {idh_str}")
            print(f"       But in Glioma specifically:")
            print(f"         IDH1 R132H = -1.01 hazard (profoundly protective)")
            print(f"         Triggers 2-HG production -> epigenetic reprogramming")
            print(f"         Glioma + DNAMethylation channel: defense_shift = -0.710")
            print(f"       V6c captures this via cancer_type_embedding x channel_attention.")
            print(f"       Our graph uses the GLOBAL gene shift ({idh_str}),")
            print(f"       which averages across all cancer types where IDH mutates.")
            print(f"       The missing edge: CancerType(Glioma) -> Channel(DNAMethylation)")
            print(f"       with weight -0.710. This is the cancer_type_modulates_channel")
            print(f"       edge from EV03 that the graph scorer doesn't encode yet.")
        if has_tp53:
            tp53_shift = gene_shift.get("TP53", None)
            tp53_str = f"{tp53_shift:+.3f}" if tp53_shift else "n/a"
            print(f"    ** TP53 mutation present.")
            print(f"       Global TP53 shift: {tp53_str}")
            print(f"       TP53 is the most connected node (degree=13, 55 cross-channel)")
            print(f"       In glioma, TP53 often co-occurs with IDH and indicates")
            print(f"       astrocytoma subtype with better prognosis.")
            print(f"       V6c learns this TP53-in-glioma context; our graph cannot.")
        if not has_idh and not has_tp53:
            print(f"    ** No IDH or TP53.")
            print(f"       The gap is from cancer-type-specific channel weights")
            print(f"       the transformer learns via its embedding layer.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
