#!/usr/bin/env python3
"""
Error–unmapped mutation analysis.

1. Run V2 model, get per-patient hazard predictions
2. Find patients with worst prediction error (high hazard but survived, or low hazard but died early)
3. Check if unmapped mutations are enriched in high-error patients
4. Cluster unmapped genes in high-error patients to find potential 7th channel
5. Rank unmapped genes by their association with prediction error

Usage:
    python3 -u -m gnn.scripts.error_unmapped_analysis
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.channel_dataset_v2 import build_channel_features_v2, ChannelDatasetV2, CHANNEL_FEAT_DIM
from gnn.models.channel_net_v2 import ChannelNetV2
from gnn.config import GNN_RESULTS, CHANNEL_MAP, CHANNEL_NAMES, NON_SILENT, MSK_DATASETS


def collate_fn(batch):
    return {
        "channel_features": torch.stack([b["channel_features"] for b in batch]),
        "tier_features": torch.stack([b["tier_features"] for b in batch]),
        "cancer_type_idx": torch.stack([b["cancer_type_idx"] for b in batch]),
        "age": torch.stack([b["age"] for b in batch]),
        "sex": torch.stack([b["sex"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
    }


def get_predictions(model, loader):
    """Get hazard predictions for all patients."""
    model.eval()
    all_h, all_t, all_e = [], [], []
    with torch.no_grad():
        for batch in loader:
            hazard = model(batch)
            all_h.append(hazard)
            all_t.append(batch["time"])
            all_e.append(batch["event"])
    return torch.cat(all_h), torch.cat(all_t), torch.cat(all_e)


def main():
    CONFIG = {
        "channel_feat_dim": CHANNEL_FEAT_DIM,
        "hidden_dim": 128,
        "cross_channel_heads": 4,
        "cross_channel_layers": 2,
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 15,
        "batch_size": 512,
        "n_folds": 5,
        "random_seed": 42,
        "n_cancer_types": 80,
    }

    model_base = os.path.join(GNN_RESULTS, "channelnet_v2")

    print("=" * 75)
    print("  ERROR–UNMAPPED MUTATION ANALYSIS")
    print("  Do unmapped mutations explain where the model fails?")
    print("=" * 75)

    # ── Load data ──
    data_dict = build_channel_features_v2("msk_impact_50k")
    n_ct = len(data_dict["cancer_type_vocab"])
    CONFIG["n_cancer_types"] = n_ct
    N = len(data_dict["times"])

    # ── Load raw mutation data to get unmapped genes ──
    paths = MSK_DATASETS["msk_impact_50k"]
    mut = pd.read_csv(paths["mutations"], low_memory=False)
    clin = pd.read_csv(paths["clinical"], low_memory=False)

    # Same patient filtering as dataset builder
    clin = clin[clin["OS_STATUS"].notna() & clin["OS_MONTHS"].notna()].copy()
    clin["event"] = clin["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    clin["time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin = clin.dropna(subset=["time"])
    clin = clin[clin["time"] > 0]
    patients = clin["patientId"].unique()
    patient_to_idx = {p: i for i, p in enumerate(patients)}

    # Filter to non-silent mutations
    channel_genes = set(CHANNEL_MAP.keys())
    mut_ns = mut[mut["mutationType"].isin(NON_SILENT)].copy()

    # Separate mapped vs unmapped
    mut_mapped = mut_ns[mut_ns["gene.hugoGeneSymbol"].isin(channel_genes)]
    mut_unmapped = mut_ns[~mut_ns["gene.hugoGeneSymbol"].isin(channel_genes)]

    # Per-patient unmapped mutation count
    unmapped_per_patient = mut_unmapped.groupby("patientId")["gene.hugoGeneSymbol"].nunique()
    mapped_per_patient = mut_mapped.groupby("patientId")["gene.hugoGeneSymbol"].nunique()

    # Per-patient unmapped gene list
    patient_unmapped_genes = mut_unmapped.groupby("patientId")["gene.hugoGeneSymbol"].apply(set).to_dict()

    print(f"  {N} patients, {len(channel_genes)} mapped genes, "
          f"{mut_unmapped['gene.hugoGeneSymbol'].nunique()} unmapped genes")

    # ── Get model predictions across all folds ──
    # Use validation-set predictions to avoid train leakage
    event_arr = data_dict["events"].numpy().astype(int)
    ch_count = (data_dict["channel_features"][:, :, 0] > 0).sum(dim=1).numpy().clip(max=3)
    strat = event_arr * 4 + ch_count

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_hazards = torch.zeros(N)
    all_in_val = torch.zeros(N, dtype=torch.bool)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), strat)):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue

        model = ChannelNetV2(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        val_ds = ChannelDatasetV2(data_dict, indices=val_idx.tolist())
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                                shuffle=False, collate_fn=collate_fn)

        hazards, times, events = get_predictions(model, val_loader)

        for i, vi in enumerate(val_idx):
            all_hazards[vi] = hazards[i]
            all_in_val[vi] = True

    # ── Compute prediction error ──
    # For deceased patients: error = predicted low hazard (should have been high)
    # For survivors: error = predicted high hazard (should have been low)
    # We use rank-based error: concordance violations

    times_np = data_dict["times"].numpy()
    events_np = data_dict["events"].numpy()
    hazards_np = all_hazards.numpy()
    val_mask = all_in_val.numpy()

    # Simple error metric: for events, residual = time * exp(-hazard)
    # High residual = patient died faster than model predicted
    # For censored: residual = -time * exp(hazard) — penalize high hazard for survivors

    # Normalize hazard to [0,1] range for interpretability
    h_valid = hazards_np[val_mask]
    h_rank = np.zeros_like(hazards_np)
    h_rank[val_mask] = (np.argsort(np.argsort(h_valid)) / len(h_valid))

    # Error: for deceased, high hazard rank is correct → error = 1 - h_rank if died early
    # Simpler: use |predicted_rank - actual_rank| on time
    t_valid = times_np[val_mask]
    t_rank = np.zeros_like(times_np)
    t_rank[val_mask] = (np.argsort(np.argsort(-t_valid)) / len(t_valid))  # lower time = higher risk rank

    # Only look at events (deceased) for clean error signal
    error = np.abs(h_rank - t_rank)

    # Split into quintiles by error
    event_val_mask = val_mask & (events_np == 1)
    error_among_events = error[event_val_mask]
    patient_indices_events = np.where(event_val_mask)[0]

    # Top 20% worst predictions vs bottom 20% best
    n_events = len(error_among_events)
    sorted_by_error = np.argsort(-error_among_events)
    top20_idx = patient_indices_events[sorted_by_error[:n_events // 5]]
    bot20_idx = patient_indices_events[sorted_by_error[-n_events // 5:]]

    print(f"\n  Deceased patients with val predictions: {n_events}")
    print(f"  Top 20% worst predicted: {len(top20_idx)}")
    print(f"  Bottom 20% best predicted: {len(bot20_idx)}")

    # ── Compare unmapped mutation burden ──
    def get_unmapped_count(indices):
        counts = []
        for idx in indices:
            pid = patients[idx]
            counts.append(unmapped_per_patient.get(pid, 0))
        return np.array(counts)

    def get_mapped_count(indices):
        counts = []
        for idx in indices:
            pid = patients[idx]
            counts.append(mapped_per_patient.get(pid, 0))
        return np.array(counts)

    unmapped_worst = get_unmapped_count(top20_idx)
    unmapped_best = get_unmapped_count(bot20_idx)
    mapped_worst = get_mapped_count(top20_idx)
    mapped_best = get_mapped_count(bot20_idx)

    print(f"\n  {'Metric':<35} {'Worst 20%':>12} {'Best 20%':>12} {'Delta':>8}")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*8}")
    print(f"  {'Unmapped genes/patient (mean)':<35} {unmapped_worst.mean():>12.2f} "
          f"{unmapped_best.mean():>12.2f} {unmapped_worst.mean()-unmapped_best.mean():>+8.2f}")
    print(f"  {'Mapped genes/patient (mean)':<35} {mapped_worst.mean():>12.2f} "
          f"{mapped_best.mean():>12.2f} {mapped_worst.mean()-mapped_best.mean():>+8.2f}")
    ratio_worst = unmapped_worst.mean() / max(mapped_worst.mean(), 0.01)
    ratio_best = unmapped_best.mean() / max(mapped_best.mean(), 0.01)
    print(f"  {'Unmapped/mapped ratio':<35} {ratio_worst:>12.2f} "
          f"{ratio_best:>12.2f} {ratio_worst-ratio_best:>+8.2f}")

    # ── Which unmapped genes are enriched in high-error patients? ──
    print(f"\n  {'─'*72}")
    print(f"  UNMAPPED GENES ENRICHED IN HIGH-ERROR PATIENTS")
    print(f"  {'─'*72}")

    # Count each unmapped gene in worst vs best groups
    gene_in_worst = Counter()
    gene_in_best = Counter()

    for idx in top20_idx:
        pid = patients[idx]
        for gene in patient_unmapped_genes.get(pid, set()):
            gene_in_worst[gene] += 1

    for idx in bot20_idx:
        pid = patients[idx]
        for gene in patient_unmapped_genes.get(pid, set()):
            gene_in_best[gene] += 1

    # Enrichment ratio: freq in worst / freq in best
    all_unmapped_genes = set(gene_in_worst.keys()) | set(gene_in_best.keys())
    enrichment = []
    for gene in all_unmapped_genes:
        w = gene_in_worst.get(gene, 0)
        b = gene_in_best.get(gene, 0)
        # Need minimum count to be meaningful
        if w + b < 20:
            continue
        # Normalize by group size
        rate_worst = w / len(top20_idx)
        rate_best = b / len(bot20_idx)
        if rate_best > 0:
            ratio = rate_worst / rate_best
        else:
            ratio = float('inf') if rate_worst > 0 else 1.0
        enrichment.append({
            "gene": gene,
            "count_worst": w,
            "count_best": b,
            "rate_worst": rate_worst,
            "rate_best": rate_best,
            "enrichment": ratio,
        })

    enrichment.sort(key=lambda x: -x["enrichment"])

    print(f"\n  Top 25 genes enriched in worst-predicted patients:")
    print(f"  {'Gene':<15} {'Worst':>6} {'Best':>6} {'Rate_W':>8} {'Rate_B':>8} {'Enrich':>8}")
    print(f"  {'─'*15} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
    for e in enrichment[:25]:
        print(f"  {e['gene']:<15} {e['count_worst']:>6} {e['count_best']:>6} "
              f"{e['rate_worst']:>8.3f} {e['rate_best']:>8.3f} {e['enrichment']:>8.2f}x")

    # ── Do these genes cluster into a coherent function? ──
    print(f"\n  {'─'*72}")
    print(f"  CO-OCCURRENCE CLUSTERING OF TOP ERROR-ENRICHED GENES")
    print(f"  {'─'*72}")

    # Take top 30 enriched genes and check co-occurrence
    top_error_genes = [e["gene"] for e in enrichment[:30]]

    # Build co-occurrence matrix among these genes across ALL patients
    gene_patients = {}
    for gene in top_error_genes:
        gene_mask = mut_unmapped[mut_unmapped["gene.hugoGeneSymbol"] == gene]["patientId"].unique()
        gene_patients[gene] = set(gene_mask)

    # Jaccard similarity
    print(f"\n  Top gene pairs by co-occurrence (Jaccard > 0.05):")
    pairs = []
    for i, g1 in enumerate(top_error_genes):
        for j, g2 in enumerate(top_error_genes):
            if j <= i:
                continue
            s1, s2 = gene_patients.get(g1, set()), gene_patients.get(g2, set())
            if len(s1 | s2) == 0:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard > 0.05:
                pairs.append((g1, g2, jaccard, len(s1 & s2)))

    pairs.sort(key=lambda x: -x[2])
    for g1, g2, j, n in pairs[:15]:
        print(f"    {g1:<15} — {g2:<15}  Jaccard={j:.3f}  ({n} shared patients)")

    # ── Check if top error genes map to known pathways ──
    # Use simple keyword-based pathway hints
    GENE_HINTS = {
        "KMT2D": "chromatin/epigenetic", "KMT2C": "chromatin/epigenetic",
        "KMT2A": "chromatin/epigenetic", "KMT2B": "chromatin/epigenetic",
        "CREBBP": "chromatin/epigenetic", "EP300": "chromatin/epigenetic",
        "ARID1A": "chromatin/SWI-SNF", "ARID1B": "chromatin/SWI-SNF",
        "ARID2": "chromatin/SWI-SNF", "SMARCA4": "chromatin/SWI-SNF",
        "SMARCB1": "chromatin/SWI-SNF",
        "KDM6A": "chromatin/epigenetic", "KDM5C": "chromatin/epigenetic",
        "SETD2": "chromatin/epigenetic", "H3C7": "chromatin/histone",
        "BCOR": "chromatin/polycomb", "BAP1": "chromatin/polycomb",
        "FAT1": "Wnt/tissue", "FAT3": "Wnt/tissue", "FAT4": "Wnt/tissue",
        "RNF43": "Wnt/tissue",
        "ZFHX3": "transcription", "SPEN": "transcription/Notch",
        "MGA": "transcription/MYC", "MED12": "transcription/mediator",
        "PTPRT": "signaling/phosphatase", "PTPRD": "signaling/phosphatase",
        "PTPRS": "signaling/phosphatase",
        "ATRX": "chromatin/telomere", "DAXX": "chromatin/telomere",
        "GRIN2A": "glutamate_receptor", "EPHA3": "ephrin/receptor",
        "EPHA5": "ephrin/receptor", "ERBB4": "growth_factor/HER",
        "ROS1": "kinase/growth", "PREX2": "PI3K/Rac",
        "ANKRD11": "chromatin/coactivator",
        "NSD1": "chromatin/epigenetic", "NSD2": "chromatin/epigenetic",
        "NSD3": "chromatin/epigenetic",
        "TERT": "telomere", "CIC": "transcription/RTK",
        "DNMT3A": "epigenetic/methylation", "DNMT3B": "epigenetic/methylation",
        "TET1": "epigenetic/methylation", "TET2": "epigenetic/methylation",
        "IDH1": "epigenetic/metabolism", "IDH2": "epigenetic/metabolism",
    }

    print(f"\n  {'─'*72}")
    print(f"  FUNCTIONAL ANNOTATION OF TOP ERROR-ENRICHED GENES")
    print(f"  {'─'*72}")

    pathway_counts = Counter()
    for e in enrichment[:30]:
        gene = e["gene"]
        hint = GENE_HINTS.get(gene, "unknown")
        pathway_counts[hint.split("/")[0]] += 1
        print(f"  {gene:<15} {hint:<30} enrich={e['enrichment']:.2f}x")

    print(f"\n  Functional category counts (top 30 error-enriched genes):")
    for cat, count in pathway_counts.most_common():
        bar = "█" * (count * 3)
        print(f"    {cat:<25} {count:>3}  {bar}")

    # ── Final: rank ALL unmapped genes by potential signal contribution ──
    print(f"\n{'='*75}")
    print(f"  RANKED UNMAPPED GENES BY SIGNAL POTENTIAL")
    print(f"  (enrichment in errors × mutation frequency × functional coherence)")
    print(f"{'='*75}")

    # Score = enrichment * sqrt(total_count) * (1.5 if known pathway, 1.0 if not)
    scored = []
    for e in enrichment:
        gene = e["gene"]
        total = e["count_worst"] + e["count_best"]
        hint = GENE_HINTS.get(gene, "unknown")
        pathway_bonus = 1.5 if hint != "unknown" else 1.0
        score = e["enrichment"] * np.sqrt(total) * pathway_bonus
        scored.append({
            **e,
            "score": score,
            "pathway": hint,
            "total_count": total,
        })

    scored.sort(key=lambda x: -x["score"])

    print(f"\n  {'Gene':<15} {'Score':>8} {'Enrich':>8} {'N_total':>8} {'Pathway':<30}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*30}")
    for s in scored[:40]:
        print(f"  {s['gene']:<15} {s['score']:>8.1f} {s['enrichment']:>7.2f}x "
              f"{s['total_count']:>7}  {s['pathway']}")

    # ── Save results ──
    out_path = os.path.join(GNN_RESULTS, "channelnet_v2", "error_unmapped_analysis.json")
    results = {
        "n_events_analyzed": int(n_events),
        "n_worst_20pct": int(len(top20_idx)),
        "unmapped_burden": {
            "worst_20pct": float(unmapped_worst.mean()),
            "best_20pct": float(unmapped_best.mean()),
        },
        "top_enriched_genes": enrichment[:50],
        "top_signal_candidates": [
            {"gene": s["gene"], "score": s["score"], "enrichment": s["enrichment"],
             "pathway": s["pathway"]}
            for s in scored[:50]
        ],
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
