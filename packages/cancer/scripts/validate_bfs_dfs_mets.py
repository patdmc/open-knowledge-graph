"""
Validate BFS vs DFS escalation theory using held-out metastasis site data.

Theory: mutations that escalate within a single channel (DFS = depth-first)
represent the body fighting back along a known pathway. Mutations that spread
across channels (BFS = breadth-first) represent systemic failure where the
body hasn't contained the problem.

Prediction: patients with higher cross-channel attention weight should have
more distant metastasis sites (BFS spread to more organ systems).

Validation data: msk_met_2021 DMETS_DX_* fields (3,893 patients overlapping
with training data, never used as model features).

Usage:
    python3 -u -m gnn.scripts.validate_bfs_dfs_mets
    python3 -u -m gnn.scripts.validate_bfs_dfs_mets --edges path/to/attention_edges.json
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, GNN_RESULTS, MSK_DATASETS

RESULTS_DIR = os.path.join(GNN_RESULTS, "depmap_pretrain")
CLINICAL_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cache", "msk", "msk_clinical_enriched.csv",
)

# Distant metastasis site columns (excluding UNSPECIFIED)
DMETS_SITES = [
    "DMETS_DX_LIVER", "DMETS_DX_LUNG", "DMETS_DX_BONE", "DMETS_DX_DIST_LN",
    "DMETS_DX_INTRA_ABDOMINAL", "DMETS_DX_CNS_BRAIN", "DMETS_DX_PLEURA",
    "DMETS_DX_BOWEL", "DMETS_DX_BILIARY_TRACT", "DMETS_DX_ADRENAL_GLAND",
    "DMETS_DX_SKIN", "DMETS_DX_BLADDER_UT", "DMETS_DX_FEMALE_GENITAL",
    "DMETS_DX_PNS", "DMETS_DX_MEDIASTINUM", "DMETS_DX_OVARY",
    "DMETS_DX_MALE_GENITAL", "DMETS_DX_KIDNEY", "DMETS_DX_HEAD_NECK",
    "DMETS_DX_BREAST",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=str,
                        default=os.path.join(RESULTS_DIR, "attention_edges.json"))
    return parser.parse_args()


def load_met_site_counts():
    """Load per-patient met site counts from cBioPortal clinical data.

    Returns:
        dict: {patient_id: n_met_sites} for patients with DMETS data
    """
    import ssl, urllib.request, urllib.parse
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    from gnn.config import CBIOPORTAL_BASE

    print("  Fetching DMETS data from cBioPortal...")
    url = f"{CBIOPORTAL_BASE}/studies/msk_met_2021/clinical-data?clinicalDataType=SAMPLE&pageSize=100000"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    # Count met sites per patient
    patient_mets = defaultdict(set)
    for item in data:
        attr = item.get("clinicalAttributeId", "")
        if attr.startswith("DMETS_DX_") and attr != "DMETS_DX_UNSPECIFIED":
            if item.get("value") == "Yes":
                patient_mets[item["patientId"]].add(attr)

    met_counts = {pid: len(sites) for pid, sites in patient_mets.items()}
    print(f"  Patients with DMETS data: {len(met_counts)}")
    return met_counts, patient_mets


def compute_patient_mutation_profiles(mutations_df):
    """For each patient, compute channel distribution of mutations.

    Returns:
        dict: {patient_id: {
            'n_mutations': int,
            'n_channels': int (distinct channels hit),
            'channel_counts': {channel: count},
            'channel_entropy': float (Shannon entropy of channel distribution),
            'bfs_ratio': float (n_channels / n_mutations, higher = more BFS),
        }}
    """
    profiles = {}
    grouped = mutations_df.groupby("patientId")

    for pid, group in grouped:
        genes = group["gene.hugoGeneSymbol"].values
        channels = [CHANNEL_MAP.get(g) for g in genes]
        channels = [c for c in channels if c is not None]

        if not channels:
            continue

        ch_counts = defaultdict(int)
        for c in channels:
            ch_counts[c] += 1

        n_muts = len(channels)
        n_channels = len(ch_counts)

        # Shannon entropy of channel distribution
        probs = np.array(list(ch_counts.values())) / n_muts
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # BFS ratio: how spread out across channels (normalized)
        max_entropy = np.log2(min(n_muts, len(CHANNEL_NAMES)))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

        profiles[pid] = {
            "n_mutations": n_muts,
            "n_channels": n_channels,
            "channel_counts": dict(ch_counts),
            "channel_entropy": float(entropy),
            "bfs_ratio": float(norm_entropy),
        }

    return profiles


def compute_attention_cross_channel_ratio(edges_path):
    """From attention edges, compute per-gene cross-channel attention weight.

    Returns:
        cross_ratio_by_gene: {gene: fraction of attention weight that is cross-channel}
    """
    with open(edges_path) as f:
        edges = json.load(f)

    gene_intra = defaultdict(float)
    gene_cross = defaultdict(float)

    for e in edges:
        ga, gb = e["from"], e["to"]
        if ga == gb:
            continue
        w = e.get("weight", 0)
        ch_a = CHANNEL_MAP.get(ga)
        ch_b = CHANNEL_MAP.get(gb)

        if ch_a and ch_b and ch_a != ch_b:
            gene_cross[ga] += w
            gene_cross[gb] += w
        else:
            gene_intra[ga] += w
            gene_intra[gb] += w

    cross_ratio = {}
    all_genes = set(gene_intra) | set(gene_cross)
    for g in all_genes:
        total = gene_intra.get(g, 0) + gene_cross.get(g, 0)
        if total > 0:
            cross_ratio[g] = gene_cross.get(g, 0) / total

    return cross_ratio


def main():
    args = parse_args()

    print("=" * 60)
    print("  BFS/DFS VALIDATION: MUTATION SPREAD vs MET SITES")
    print("=" * 60)

    # Load met site data
    met_counts, patient_mets = load_met_site_counts()

    # Load training mutations
    paths = MSK_DATASETS["msk_impact_50k"]
    mutations = pd.read_csv(paths["mutations"])
    clinical = pd.read_csv(paths["clinical"])

    # Find overlapping patients
    train_pids = set(clinical["patientId"])
    overlap_pids = train_pids & set(met_counts.keys())
    print(f"  Training patients: {len(train_pids)}")
    print(f"  Overlapping with DMETS data: {len(overlap_pids)}")

    if len(overlap_pids) < 50:
        print("  Not enough overlap for meaningful analysis.")
        return

    # Compute mutation channel profiles
    print("\n  Computing mutation channel profiles...")
    profiles = compute_patient_mutation_profiles(mutations)
    overlap_with_profile = overlap_pids & set(profiles.keys())
    print(f"  Patients with both DMETS and mutation profiles: {len(overlap_with_profile)}")

    # === TEST 1: Channel entropy vs met site count ===
    print("\n  TEST 1: Channel entropy (BFS measure) vs met site count")
    print("  " + "-" * 55)

    entropies = []
    n_mets = []
    for pid in overlap_with_profile:
        entropies.append(profiles[pid]["channel_entropy"])
        n_mets.append(met_counts[pid])

    entropies = np.array(entropies)
    n_mets = np.array(n_mets)

    rho, p_val = spearmanr(entropies, n_mets)
    print(f"  Spearman correlation: rho={rho:.3f}, p={p_val:.4f}")

    # === TEST 2: n_channels hit vs met site count ===
    print("\n  TEST 2: Distinct channels hit vs met site count")
    print("  " + "-" * 55)

    n_channels = np.array([profiles[pid]["n_channels"] for pid in overlap_with_profile])
    rho2, p2 = spearmanr(n_channels, n_mets)
    print(f"  Spearman correlation: rho={rho2:.3f}, p={p2:.4f}")

    # === TEST 3: BFS ratio vs met site count ===
    print("\n  TEST 3: BFS ratio (normalized entropy) vs met site count")
    print("  " + "-" * 55)

    bfs_ratios = np.array([profiles[pid]["bfs_ratio"] for pid in overlap_with_profile])
    rho3, p3 = spearmanr(bfs_ratios, n_mets)
    print(f"  Spearman correlation: rho={rho3:.3f}, p={p3:.4f}")

    # === TEST 4: High vs low met sites — mutation profile comparison ===
    print("\n  TEST 4: Mutation profiles by met site count")
    print("  " + "-" * 55)

    pids_list = sorted(overlap_with_profile)
    low_met = [pid for pid in pids_list if met_counts[pid] <= 1]
    high_met = [pid for pid in pids_list if met_counts[pid] >= 4]

    if len(low_met) >= 20 and len(high_met) >= 20:
        low_entropy = [profiles[pid]["channel_entropy"] for pid in low_met]
        high_entropy = [profiles[pid]["channel_entropy"] for pid in high_met]
        low_nch = [profiles[pid]["n_channels"] for pid in low_met]
        high_nch = [profiles[pid]["n_channels"] for pid in high_met]
        low_nmut = [profiles[pid]["n_mutations"] for pid in low_met]
        high_nmut = [profiles[pid]["n_mutations"] for pid in high_met]

        u_ent, p_ent = mannwhitneyu(low_entropy, high_entropy, alternative="less")
        u_nch, p_nch = mannwhitneyu(low_nch, high_nch, alternative="less")
        u_nmut, p_nmut = mannwhitneyu(low_nmut, high_nmut, alternative="less")

        print(f"  Low met (≤1 site): n={len(low_met)}")
        print(f"    Mean entropy: {np.mean(low_entropy):.3f}, mean channels: {np.mean(low_nch):.2f}, "
              f"mean mutations: {np.mean(low_nmut):.1f}")
        print(f"  High met (≥4 sites): n={len(high_met)}")
        print(f"    Mean entropy: {np.mean(high_entropy):.3f}, mean channels: {np.mean(high_nch):.2f}, "
              f"mean mutations: {np.mean(high_nmut):.1f}")
        print(f"\n  Mann-Whitney U (low < high):")
        print(f"    Channel entropy: p={p_ent:.4f}")
        print(f"    Distinct channels: p={p_nch:.4f}")
        print(f"    Total mutations: p={p_nmut:.4f}")

    # === TEST 4b: BFS vs DFS survival ===
    print("\n  TEST 4b: Survival by mutation spread pattern")
    print("  " + "-" * 55)

    # Load OS data for overlapping patients
    clinical_os = clinical.set_index("patientId")[["OS_MONTHS", "OS_STATUS"]].to_dict("index")

    # Group by channel spread: DFS (1-2 channels) vs BFS (3+ channels)
    dfs_pids = [pid for pid in overlap_with_profile if profiles[pid]["n_channels"] <= 2]
    bfs_pids = [pid for pid in overlap_with_profile if profiles[pid]["n_channels"] >= 3]

    def os_stats(pids):
        os_vals = []
        deaths = 0
        for pid in pids:
            info = clinical_os.get(pid)
            if info and pd.notna(info["OS_MONTHS"]):
                os_vals.append(info["OS_MONTHS"])
                if "DECEASED" in str(info["OS_STATUS"]).upper() or "1" in str(info["OS_STATUS"]):
                    deaths += 1
        return os_vals, deaths

    dfs_os, dfs_deaths = os_stats(dfs_pids)
    bfs_os, bfs_deaths = os_stats(bfs_pids)

    if dfs_os and bfs_os:
        u_os, p_os = mannwhitneyu(bfs_os, dfs_os, alternative="less")
        print(f"  DFS (≤2 channels): n={len(dfs_pids)}, median OS={np.median(dfs_os):.1f}mo, "
              f"death rate={dfs_deaths/len(dfs_os):.0%}")
        print(f"  BFS (≥3 channels): n={len(bfs_pids)}, median OS={np.median(bfs_os):.1f}mo, "
              f"death rate={bfs_deaths/len(bfs_os):.0%}")
        print(f"  Mann-Whitney (BFS OS < DFS OS): p={p_os:.4f}")

    # Also test by met site count directly
    low_met_os, low_deaths = os_stats(low_met)
    high_met_os, high_deaths = os_stats(high_met)

    if low_met_os and high_met_os:
        u_os2, p_os2 = mannwhitneyu(high_met_os, low_met_os, alternative="less")
        print(f"\n  Low met (≤1 site): median OS={np.median(low_met_os):.1f}mo, "
              f"death rate={low_deaths/len(low_met_os):.0%}")
        print(f"  High met (≥4 sites): median OS={np.median(high_met_os):.1f}mo, "
              f"death rate={high_deaths/len(high_met_os):.0%}")
        print(f"  Mann-Whitney (high met OS < low met OS): p={p_os2:.4f}")

    # Cross-test: does BFS predict met spread AND worse OS independently?
    # BFS patients with LOW met sites vs BFS patients with HIGH met sites
    bfs_low_met = [pid for pid in bfs_pids if met_counts.get(pid, 0) <= 1]
    bfs_high_met = [pid for pid in bfs_pids if met_counts.get(pid, 0) >= 4]
    bfs_low_os, bfs_low_d = os_stats(bfs_low_met)
    bfs_high_os, bfs_high_d = os_stats(bfs_high_met)

    if len(bfs_low_os) >= 10 and len(bfs_high_os) >= 10:
        print(f"\n  Within BFS patients (≥3 channels):")
        print(f"    Low met: n={len(bfs_low_met)}, median OS={np.median(bfs_low_os):.1f}mo")
        print(f"    High met: n={len(bfs_high_met)}, median OS={np.median(bfs_high_os):.1f}mo")

    # === TEST 5: Per-channel met site affinity ===
    print("\n  TEST 5: Which channels associate with which met sites?")
    print("  " + "-" * 55)

    # For each channel, find patients whose mutations are concentrated there
    channel_met_sites = {}
    for ch in CHANNEL_NAMES:
        ch_patients = []
        for pid in overlap_with_profile:
            counts = profiles[pid]["channel_counts"]
            total = profiles[pid]["n_mutations"]
            if counts.get(ch, 0) / total > 0.5 and total >= 2:
                ch_patients.append(pid)

        if len(ch_patients) >= 10:
            site_rates = {}
            for site in DMETS_SITES:
                site_short = site.replace("DMETS_DX_", "")
                n_with = sum(1 for pid in ch_patients if site in patient_mets.get(pid, set()))
                site_rates[site_short] = n_with / len(ch_patients)
            channel_met_sites[ch] = {"n_patients": len(ch_patients), "site_rates": site_rates}

    for ch, info in channel_met_sites.items():
        top_sites = sorted(info["site_rates"].items(), key=lambda x: -x[1])[:5]
        print(f"  {ch} (n={info['n_patients']}):")
        for site, rate in top_sites:
            if rate > 0:
                print(f"    {site}: {rate:.0%}")

    # === TEST 6: Attention edge cross-channel ratio (if available) ===
    if os.path.exists(args.edges):
        print("\n  TEST 6: Attention cross-channel ratio vs met sites")
        print("  " + "-" * 55)

        cross_ratio = compute_attention_cross_channel_ratio(args.edges)
        print(f"  Genes with attention cross-channel ratio: {len(cross_ratio)}")

        # For each patient, compute weighted average cross-channel ratio
        # based on their mutated genes
        patient_cross = {}
        for pid in overlap_with_profile:
            genes = [g for g in mutations[mutations["patientId"] == pid]["gene.hugoGeneSymbol"]
                     if g in cross_ratio]
            if genes:
                patient_cross[pid] = np.mean([cross_ratio[g] for g in genes])

        if len(patient_cross) >= 50:
            cross_vals = []
            met_vals = []
            for pid in patient_cross:
                cross_vals.append(patient_cross[pid])
                met_vals.append(met_counts[pid])

            rho6, p6 = spearmanr(cross_vals, met_vals)
            print(f"  Patients with attention + DMETS data: {len(patient_cross)}")
            print(f"  Spearman (attention cross-channel ratio vs met sites): rho={rho6:.3f}, p={p6:.4f}")
    else:
        print("\n  TEST 6: Skipped (no attention edges file)")

    # Save results
    results = {
        "n_overlap": len(overlap_with_profile),
        "test1_entropy_vs_mets": {"rho": float(rho), "p": float(p_val)},
        "test2_nchannels_vs_mets": {"rho": float(rho2), "p": float(p2)},
        "test3_bfs_ratio_vs_mets": {"rho": float(rho3), "p": float(p3)},
        "channel_met_affinities": channel_met_sites,
    }

    out_path = os.path.join(RESULTS_DIR, "bfs_dfs_validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
