#!/usr/bin/env python3
"""
Inference for AtlasTransformer V4 — counterfactual treatment predictions.

Given a patient's mutations (gene + protein change) and cancer type,
predicts relative hazard under each treatment modality. The delta
between treatments IS the treatment recommendation.

Usage:
    # Interactive single patient:
    python3 -m gnn.scripts.inference_v4 \
        --cancer BRCA \
        --mutations "TP53:R175H,PIK3CA:H1047R,BRCA1:E1836fs"

    # Compare all treatments for a patient:
    python3 -m gnn.scripts.inference_v4 \
        --cancer LUAD \
        --mutations "KRAS:G12C,TP53:R248W,STK11:Q37*" \
        --counterfactual

    # Batch inference from CSV:
    python3 -m gnn.scripts.inference_v4 --batch patients.csv
"""

import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.treatment_dataset import (
    TreatmentDataset, NODE_FEAT_DIM, MAX_NODES, get_channel_pos_id,
)
from gnn.models.atlas_transformer_v4 import AtlasTransformerV4
from gnn.config import GNN_RESULTS, CHANNEL_MAP

TREATMENT_NAMES = [
    "surgery", "radiation", "chemotherapy", "endocrine",
    "targeted", "immunotherapy",
]

# Standard treatment scenarios for counterfactual comparison
TREATMENT_SCENARIOS = {
    "no_treatment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "surgery_only": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "surgery+radiation": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "surgery+chemo": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "surgery+chemo+radiation": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "surgery+endocrine": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "surgery+targeted": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "surgery+immuno": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "chemo_platinum": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "chemo_taxane": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "chemo+immuno": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "targeted+immuno": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', type=str, help='TCGA cancer type abbreviation (e.g., BRCA, LUAD)')
    parser.add_argument('--mutations', type=str,
                        help='Comma-separated gene:protein_change pairs (e.g., "TP53:R175H,PIK3CA:H1047R")')
    parser.add_argument('--treatment', type=str, default=None,
                        help='Comma-separated treatment flags: surgery,radiation,chemo,endocrine,targeted,immuno')
    parser.add_argument('--counterfactual', action='store_true',
                        help='Compare all standard treatment scenarios')
    parser.add_argument('--batch', type=str, default=None,
                        help='Path to CSV with columns: cancer, mutations, treatment (optional)')
    parser.add_argument('--age', type=float, default=60.0, help='Patient age')
    parser.add_argument('--sex', type=str, default='unknown', choices=['male', 'female', 'unknown'])
    return parser.parse_args()


def load_model():
    """Load trained V4 model."""
    results_dir = os.path.join(GNN_RESULTS, "atlas_transformer_v4")
    results_path = os.path.join(results_dir, "results.json")
    model_path = os.path.join(results_dir, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run train_atlas_transformer_v4.py first.")

    with open(results_path) as f:
        results = json.load(f)

    config = results["config"]
    model = AtlasTransformerV4(config)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, config


def load_dataset():
    """Load dataset for feature lookups (atlas, enrichment data)."""
    ds = TreatmentDataset()
    return ds


def parse_mutations(mutation_str):
    """Parse 'TP53:R175H,PIK3CA:H1047R' into [(gene, protein_change), ...]."""
    muts = []
    for pair in mutation_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            gene, pc = pair.split(":", 1)
            muts.append((gene.strip(), pc.strip()))
        else:
            muts.append((pair.strip(), ""))
    return muts


def parse_treatment(treatment_str):
    """Parse 'surgery,chemo' into 11-dim treatment vector."""
    vec = np.zeros(11, dtype=np.float32)
    if not treatment_str:
        return vec

    name_to_idx = {
        "surgery": 0, "radiation": 1, "chemo": 2, "chemotherapy": 2,
        "endocrine": 3, "hormone": 3, "targeted": 4, "immuno": 5,
        "immunotherapy": 5, "platinum": 6, "taxane": 7,
        "anthracycline": 8, "antimetabolite": 9, "alkylating": 10,
    }

    for name in treatment_str.split(","):
        name = name.strip().lower()
        if name in name_to_idx:
            vec[name_to_idx[name]] = 1.0

    return vec


def build_patient_tensors(ds, cancer, mutations, treatment_vec=None, age=60.0, sex="unknown"):
    """Build model input tensors for a single patient.

    Args:
        ds: TreatmentDataset (for atlas + enrichment lookups)
        cancer: TCGA cancer abbreviation (e.g., 'BRCA')
        mutations: list of (gene, protein_change) tuples
        treatment_vec: 11-dim treatment vector (needed for per-node treatment features)
        age: patient age
        sex: 'male', 'female', or 'unknown'

    Returns:
        dict of tensors, each with batch dim = 1
    """
    if treatment_vec is None:
        treatment_vec = np.zeros(11, dtype=np.float32)

    atlas_ct = ds._get_atlas_cancer_type(cancer)
    patient_genes_set = set(g for g, _ in mutations)

    nodes = []
    cp_ids = []
    log_hrs = []

    for gene, pc in mutations:
        if gene not in CHANNEL_MAP:
            continue

        ch = CHANNEL_MAP.get(gene)
        entry = None
        if atlas_ct:
            if (atlas_ct, gene, pc) in ds.t1:
                entry = ds.t1[(atlas_ct, gene, pc)]
            elif (atlas_ct, gene) in ds.t2:
                entry = ds.t2[(atlas_ct, gene)]
            elif ch and (atlas_ct, ch) in ds.t3:
                entry = ds.t3[(atlas_ct, ch)]
            elif hasattr(ds, 't4') and (atlas_ct, gene) in ds.t4:
                entry = ds.t4[(atlas_ct, gene)]

        if entry is not None:
            hr = entry["hr"]
            ci_w = entry.get("ci_width", 1.0)
            tier = entry["tier"]
            n_w = entry.get("n_with", 50)
        else:
            hr = 1.0
            ci_w = 0.0
            tier = 0
            n_w = 0

        mt = "Missense_Mutation"
        feat = ds._make_enriched_node(
            gene, pc, mt, hr, ci_w, tier, n_w,
            cancer, patient_genes_set, treatment_vec,
        )
        nodes.append(feat)
        cp_ids.append(get_channel_pos_id(gene))
        log_hrs.append(np.log(max(hr, 0.01)))

    if len(nodes) == 0:
        nodes = [np.zeros(NODE_FEAT_DIM, dtype=np.float32)]
        cp_ids = [0]
        atlas_sum = 0.0
    else:
        atlas_sum = float(sum(log_hrs))

    n_nodes = len(nodes)
    if n_nodes > MAX_NODES:
        abs_scores = np.abs([n[0] for n in nodes])
        top_idx = np.argsort(abs_scores)[-MAX_NODES:]
        nodes = [nodes[i] for i in top_idx]
        cp_ids = [cp_ids[i] for i in top_idx]
        n_nodes = MAX_NODES

    mask = [1] * n_nodes
    while len(nodes) < MAX_NODES:
        nodes.append(np.zeros(NODE_FEAT_DIM, dtype=np.float32))
        cp_ids.append(0)
        mask.append(0)

    ct_idx = 0
    age_z = (age - 60.0) / 15.0
    sex_val = 1.0 if sex == "female" else 0.0

    return {
        "node_features": torch.tensor(np.stack(nodes), dtype=torch.float32).unsqueeze(0),
        "node_mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        "channel_pos_ids": torch.tensor(cp_ids, dtype=torch.long).unsqueeze(0),
        "cancer_type": torch.tensor([ct_idx], dtype=torch.long),
        "clinical": torch.tensor([[age_z, sex_val]], dtype=torch.float32),
        "atlas_sum": torch.tensor([[atlas_sum]], dtype=torch.float32),
        "graph_features": torch.zeros(1, 47, dtype=torch.float32),
        "n_matched": sum(1 for n in nodes[:n_nodes] if n[2] > 0),
        "n_zero_conf": n_nodes - sum(1 for n in nodes[:n_nodes] if n[2] > 0),
    }


def predict_single(model, ds, cancer, mutations, treatment_vec, age=60.0, sex="unknown"):
    """Run a single prediction. Rebuilds node features with treatment for per-node treatment×channel."""
    pt = build_patient_tensors(ds, cancer, mutations, treatment_vec, age, sex)
    tv = torch.tensor(treatment_vec, dtype=torch.float32).unsqueeze(0)
    tm = torch.ones(1, 11, dtype=torch.float32)

    with torch.no_grad():
        hazard = model(
            pt["node_features"], pt["node_mask"], pt["channel_pos_ids"],
            pt["cancer_type"], pt["clinical"], pt["atlas_sum"],
            pt["graph_features"], tv, tm,
        ).squeeze().item()

    return hazard


def predict_counterfactual(model, ds, cancer, mutations, age=60.0, sex="unknown"):
    """Run patient through all treatment scenarios."""
    results = {}
    for name, vec in TREATMENT_SCENARIOS.items():
        tvec = np.array(vec, dtype=np.float32)
        hazard = predict_single(model, ds, cancer, mutations, tvec, age, sex)
        results[name] = hazard
    return results


def format_counterfactual(results, cancer, mutations):
    """Format counterfactual results for display."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"COUNTERFACTUAL TREATMENT ANALYSIS")
    lines.append(f"{'='*60}")
    lines.append(f"Cancer type: {cancer}")
    lines.append(f"Mutations: {', '.join(f'{g}:{pc}' for g, pc in mutations)}")
    lines.append(f"")
    lines.append(f"{'Treatment Scenario':<30s} {'Log Hazard':>12s} {'Rel Risk':>10s}")
    lines.append(f"{'-'*30} {'-'*12} {'-'*10}")

    # Sort by hazard (lower = better survival)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    baseline = sorted_results[0][1]  # best scenario as reference

    for name, hazard in sorted_results:
        rel_risk = np.exp(hazard - baseline)
        marker = " <-- best" if hazard == baseline else ""
        lines.append(f"  {name:<28s} {hazard:>10.4f}   {rel_risk:>8.2f}x{marker}")

    lines.append(f"")
    lines.append(f"Interpretation:")
    lines.append(f"  Lower log hazard = better predicted survival")
    lines.append(f"  Rel Risk: fold-increase in hazard vs best option")

    best_name = sorted_results[0][0]
    worst_name = sorted_results[-1][0]
    delta = sorted_results[-1][1] - sorted_results[0][1]
    lines.append(f"  Best: {best_name} | Worst: {worst_name} | Delta: {delta:.4f}")
    lines.append(f"")
    lines.append(f"  NOTE: This is a research model (holdback C=0.61).")
    lines.append(f"  NOT for clinical decision-making.")

    return "\n".join(lines)


def main():
    args = parse_args()

    print("Loading model...", flush=True)
    model, config = load_model()

    print("Loading dataset (for feature lookups)...", flush=True)
    ds = load_dataset()

    if args.batch:
        import csv
        with open(args.batch) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cancer = row["cancer"]
                mutations = parse_mutations(row["mutations"])
                treatment = parse_treatment(row.get("treatment", ""))
                hazard = predict_single(model, ds, cancer, mutations, treatment, args.age, args.sex)
                print(f"{cancer}\t{row['mutations']}\t{hazard:.4f}")
        return

    if not args.cancer or not args.mutations:
        print("Error: --cancer and --mutations required for single patient inference")
        return

    mutations = parse_mutations(args.mutations)
    # Quick check for node stats
    pt = build_patient_tensors(ds, args.cancer, mutations, np.zeros(11, dtype=np.float32), args.age, args.sex)
    print(f"\nPatient: {args.cancer}, {len(mutations)} mutations")
    print(f"  Atlas-matched nodes: {pt['n_matched']}")
    print(f"  Zero-confidence nodes: {pt['n_zero_conf']}")

    if args.counterfactual:
        results = predict_counterfactual(model, ds, args.cancer, mutations, args.age, args.sex)
        print(format_counterfactual(results, args.cancer, mutations))
    else:
        treatment = parse_treatment(args.treatment) if args.treatment else np.zeros(11, dtype=np.float32)
        hazard = predict_single(model, ds, args.cancer, mutations, treatment, args.age, args.sex)
        print(f"\nPredicted log hazard: {hazard:.4f}")
        print(f"Treatment: {args.treatment or 'none specified'}")


if __name__ == "__main__":
    main()
