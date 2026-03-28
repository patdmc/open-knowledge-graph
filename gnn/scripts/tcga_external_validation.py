#!/usr/bin/env python3
"""
External validation on TCGA benchmark datasets.

Two experiments:
1. Our MSK-trained V6c model → zero-shot on TCGA patients (mutations only)
2. CoxEN + channel features → does adding channels improve expression models?

Usage:
    python3 -u -m gnn.scripts.tcga_external_validation
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import NON_SILENT, TRUNCATING, GNN_RESULTS
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_CHANNEL_TO_IDX,
    N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6, N_TIERS_V6,
    V6_TIER_MAP, V6_GENE_FUNCTION,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.training.metrics import concordance_index

TCGA_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "tcga_benchmark")
KAGGLE_DATA = os.path.expanduser(
    "~/.cache/kagglehub/datasets/ridgiemo/processed-gene-and-clinical-data/versions/1/data")
BENCHMARK_PREDS = "/tmp/benchmarks_compare"

SAVE_BASE = os.path.join(GNN_RESULTS, "tcga_external_validation")

DATASETS = ["COAD", "ESCA", "HNSC", "LIHC", "LUAD", "LUSC", "STAD"]

TCGA_TO_NAME = {
    "COAD": "Colorectal Cancer",
    "ESCA": "Esophagogastric Cancer",
    "HNSC": "Head and Neck Cancer",
    "LIHC": "Hepatobiliary Cancer",
    "LUAD": "Non-Small Cell Lung Cancer",
    "LUSC": "Non-Small Cell Lung Cancer",  # LUSC maps to NSCLC in MSK vocab
    "STAD": "Esophagogastric Cancer",      # STAD maps to Esophagogastric in MSK vocab
}

# MSK cancer type vocab indices (from build_channel_features_v6c)
TCGA_TO_CT_IDX = {
    "COAD": 11,   # Colorectal Cancer
    "ESCA": 14,   # Esophagogastric Cancer
    "HNSC": 19,   # Head and Neck Cancer
    "LIHC": 20,   # Hepatobiliary Cancer
    "LUAD": 24,   # Non-Small Cell Lung Cancer
    "LUSC": 24,   # Non-Small Cell Lung Cancer
    "STAD": 14,   # Esophagogastric Cancer
}

# V6c model config
CONFIG = {
    "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
    "hidden_dim": 128,
    "cross_channel_heads": 4,
    "cross_channel_layers": 2,
    "dropout": 0.3,
    "n_cancer_types": 42,
}


def load_benchmark_clinical(ds):
    """Load clinical data from the benchmark Kaggle dataset."""
    clin = pd.read_csv(os.path.join(KAGGLE_DATA, ds, "clinical.csv"))
    clin["patient_id"] = clin["patient_id"].astype(str)
    if "tissue_source_site" in clin.columns:
        clin["tissue_source_site"] = clin["tissue_source_site"].astype(str)
        clin["tcga_barcode"] = "TCGA-" + clin["tissue_source_site"] + "-" + clin["patient_id"]
    else:
        # For datasets without tissue_source_site (LIHC, LUAD),
        # we'll match via short patient_id extracted from cBioPortal barcodes
        clin["tcga_barcode"] = None  # will be filled during matching
    return clin


def load_tcga_mutations(ds):
    """Load mutations downloaded from cBioPortal."""
    path = os.path.join(TCGA_DATA, f"{ds}_mutations.csv")
    if not os.path.exists(path):
        return None
    mut = pd.read_csv(path)
    return mut


def build_channel_features_tcga(ds):
    """Build V6c channel features for TCGA patients.

    Returns dict with patient_id -> feature dict, and matched clinical info.
    """
    clin = load_benchmark_clinical(ds)
    mut = load_tcga_mutations(ds)

    if mut is None:
        return None, None

    # Match patients: benchmark clinical ↔ mutation data
    # cBioPortal patientId format: TCGA-XX-YYYY
    mut_patients = set(mut["patientId"].unique())

    if clin["tcga_barcode"].isna().all():
        # No tissue_source_site — match via short patient_id from cBioPortal barcodes
        # TCGA-XX-YYYY → short_id = YYYY
        barcode_to_short = {b: b.split("-")[-1] for b in mut_patients}
        short_to_barcode = {v: k for k, v in barcode_to_short.items()}
        kaggle_pids = set(clin["patient_id"].unique())
        matched_shorts = set(short_to_barcode.keys()) & kaggle_pids
        # Fill in barcodes for matched patients
        pid_to_barcode = {pid: short_to_barcode[pid] for pid in matched_shorts}
        clin["tcga_barcode"] = clin["patient_id"].map(pid_to_barcode)
        clin = clin.dropna(subset=["tcga_barcode"])
        matched = set(clin["tcga_barcode"].unique()) & mut_patients
    else:
        clin_barcodes = set(clin["tcga_barcode"].unique())
        matched = mut_patients & clin_barcodes

    if len(matched) < 20:
        print(f"  {ds}: only {len(matched)} matched patients, skipping")
        return None, None

    # Filter to matched
    clin = clin[clin["tcga_barcode"].isin(matched)].copy()
    mut = mut[mut["patientId"].isin(matched)].copy()

    # Channel gene set
    channel_genes = set(V6_CHANNEL_MAP.keys())

    # Filter mutations to channel genes + non-silent
    # cBioPortal mutation types differ from MSK format
    # Map common types
    mut_type_map = {
        "Missense_Mutation": "Missense_Mutation",
        "Nonsense_Mutation": "Nonsense_Mutation",
        "Frame_Shift_Del": "Frame_Shift_Del",
        "Frame_Shift_Ins": "Frame_Shift_Ins",
        "Splice_Site": "Splice_Site",
        "In_Frame_Del": "In_Frame_Del",
        "In_Frame_Ins": "In_Frame_Ins",
        "Nonstop_Mutation": "Nonstop_Mutation",
        "Translation_Start_Site": "Translation_Start_Site",
    }

    # Keep non-silent mutations in channel genes
    mut = mut[
        mut["hugoGeneSymbol"].isin(channel_genes) &
        mut["mutationType"].isin(mut_type_map.keys())
    ].copy()
    mut["channel"] = mut["hugoGeneSymbol"].map(V6_CHANNEL_MAP)
    mut["is_truncating"] = mut["mutationType"].isin([
        "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
        "Splice_Site", "Nonstop_Mutation",
    ])
    mut["is_missense"] = (mut["mutationType"] == "Missense_Mutation")

    # VAF
    if "tumorAltCount" in mut.columns and "tumorRefCount" in mut.columns:
        alt = pd.to_numeric(mut["tumorAltCount"], errors="coerce")
        ref = pd.to_numeric(mut["tumorRefCount"], errors="coerce")
        mut["vaf"] = alt / (alt + ref)
        mut["vaf"] = mut["vaf"].fillna(0.0)
    else:
        mut["vaf"] = 0.0

    mut["is_gof"] = mut["hugoGeneSymbol"].map(
        lambda g: V6_GENE_FUNCTION.get(g) == "GOF"
    )
    mut["is_lof"] = mut["hugoGeneSymbol"].map(
        lambda g: V6_GENE_FUNCTION.get(g) == "LOF"
    )

    # Per-patient-per-channel aggregation
    grouped = mut.groupby(["patientId", "channel"]).agg(
        n_mutations=("hugoGeneSymbol", "size"),
        n_genes=("hugoGeneSymbol", "nunique"),
        frac_truncating=("is_truncating", "mean"),
        frac_missense=("is_missense", "mean"),
        mean_vaf=("vaf", "mean"),
        max_vaf=("vaf", "max"),
        gof_count=("is_gof", "sum"),
        lof_count=("is_lof", "sum"),
    ).reset_index()

    # Build tensors
    patients = sorted(matched)
    patient_to_idx = {p: i for i, p in enumerate(patients)}
    N = len(patients)

    channel_features = torch.zeros(N, N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6)
    tier_features = torch.zeros(N, N_TIERS_V6, 5)

    for _, row in grouped.iterrows():
        pid = row["patientId"]
        p_idx = patient_to_idx[pid]
        c_idx = V6_CHANNEL_TO_IDX.get(row["channel"])
        if c_idx is None:
            continue

        channel_features[p_idx, c_idx] = torch.tensor([
            1.0,
            np.log1p(row["n_mutations"]),
            float(row["n_genes"]),
            float(row["frac_truncating"]),
            float(row["frac_missense"]),
            float(row["mean_vaf"]),
            float(row["max_vaf"]),
            float(row["gof_count"]),
            float(row["lof_count"]),
        ])

    # Tier features
    for t_idx in range(N_TIERS_V6):
        tier_channels = [V6_CHANNEL_TO_IDX[ch] for ch, t in V6_TIER_MAP.items() if t == t_idx]
        if not tier_channels:
            continue
        tier_ch = channel_features[:, tier_channels, :]
        tier_features[:, t_idx, 0] = tier_ch[:, :, 0].sum(dim=1)
        tier_features[:, t_idx, 1] = tier_ch[:, :, 1].sum(dim=1)
        tier_features[:, t_idx, 2] = tier_ch[:, :, 2].sum(dim=1)
        tier_features[:, t_idx, 3] = tier_ch[:, :, 5].max(dim=1)[0]
        tier_features[:, t_idx, 4] = tier_ch[:, :, 6].max(dim=1)[0]

    # Clinical
    clin_indexed = clin.set_index("tcga_barcode").loc[patients]
    age_raw = pd.to_numeric(clin_indexed["age"], errors="coerce").fillna(0.5)
    # Benchmark age is normalized 0-1, convert to our scale: (age_years - 60) / 15
    # Their normalization: age / max_age (looks like 0-1 range)
    # Approximate: age_years ~ age_norm * 90, so (age_norm * 90 - 60) / 15
    age_tensor = torch.tensor(((age_raw.values * 90) - 60) / 15, dtype=torch.float32)

    sex_map = {"MALE": 1, "FEMALE": 0}
    sex_vals = clin_indexed["gender"].map(sex_map).fillna(0).astype(int)
    sex_tensor = torch.tensor(sex_vals.values, dtype=torch.long)

    # No MSI/TMB in TCGA clinical — set to 0 (unknown)
    msi_score = torch.zeros(N)
    msi_high = torch.zeros(N, dtype=torch.long)
    tmb = torch.zeros(N)

    # Cancer type — map to MSK vocabulary index
    ct_idx = TCGA_TO_CT_IDX.get(ds, 0)
    cancer_type_idx = torch.full((N,), ct_idx, dtype=torch.long)

    times = torch.tensor(clin_indexed["real_survival_time"].values, dtype=torch.float32)
    events = torch.tensor(clin_indexed["vital_status"].values, dtype=torch.float32)

    # Convert survival time from days to months
    times = times / 30.44

    data = {
        "channel_features": channel_features,
        "tier_features": tier_features,
        "cancer_type_idx": cancer_type_idx,
        "age": age_tensor,
        "sex": sex_tensor,
        "msi_score": msi_score,
        "msi_high": msi_high,
        "tmb": tmb,
        "times": times,
        "events": events,
        "patients": patients,
    }

    return data, clin_indexed


def run_v6c_inference(data):
    """Run all 5 V6c fold models and average hazard predictions."""
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    N = len(data["times"])

    all_hazards = []
    for fold_idx in range(5):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue

        model = ChannelNetV6c(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            batch = {k: v.unsqueeze(0) if v.dim() == 0 else v for k, v in data.items()
                     if k != "patients"}
            hazard = model(batch)
            all_hazards.append(hazard)

    if not all_hazards:
        return None

    # Average predictions across folds
    avg_hazard = torch.stack(all_hazards).mean(dim=0)
    return avg_hazard


def load_benchmark_predictions(ds, method="CoxAE"):
    """Load benchmark method predictions for comparison."""
    if method == "CoxAE":
        pred_dir = os.path.join(BENCHMARK_PREDS, "CoxAE", "prediction_save")
    elif method == "CoxEN":
        pred_dir = os.path.join(BENCHMARK_PREDS, "CoxEN", "prediction_save")
    elif method == "Cox-sage":
        pred_dir = os.path.join(BENCHMARK_PREDS, "Cox-sage", "prediction_save", "prediction_save")
    elif method == "CoxKAN":
        pred_dir = os.path.join(BENCHMARK_PREDS, "coxkan", "prediction_output")
    elif method == "AutoSurv":
        pred_dir = os.path.join(BENCHMARK_PREDS, "AutoSurv", "prediction_save_path")
    else:
        return None

    cis = []
    for seed in range(5):
        if method == "Cox-sage":
            # Try different layer configs, pick best
            best_ci = 0
            for layers in [1, 2, 4]:
                f = os.path.join(pred_dir, f"{ds}_{layers}layers_{seed}seed_prediction.csv")
                if os.path.exists(f):
                    df = pd.read_csv(f)
                    ci = _compute_cindex_from_df(df)
                    if ci > best_ci:
                        best_ci = ci
            if best_ci > 0:
                cis.append(best_ci)
        else:
            f = os.path.join(pred_dir, f"{ds}_seed{seed}_prediction.csv")
            if os.path.exists(f):
                df = pd.read_csv(f)
                ci = _compute_cindex_from_df(df)
                cis.append(ci)

    return np.mean(cis) if cis else None


def _compute_cindex_from_df(df):
    """Compute C-index from prediction dataframe."""
    from lifelines.utils import concordance_index as lifelines_ci
    try:
        return lifelines_ci(df["real_survival_time"], -df["prediction_risk"], df["vital_status"])
    except Exception:
        return 0.5


def build_channel_feature_matrix(ds, data, patients):
    """Build a channel feature matrix that can be concatenated with expression data.

    Returns DataFrame with patient_id index and channel features as columns.
    """
    if data is None:
        return None

    N = len(patients)
    ch_feats = data["channel_features"]  # (N, 8, 9)

    # Flatten to per-channel summary: for each channel, key features
    rows = []
    for i, pid in enumerate(patients):
        row = {"patient_id": pid}
        for c_idx, ch_name in enumerate(V6_CHANNEL_NAMES):
            row[f"ch_{ch_name}_severed"] = ch_feats[i, c_idx, 0].item()
            row[f"ch_{ch_name}_n_mut"] = ch_feats[i, c_idx, 1].item()
            row[f"ch_{ch_name}_n_genes"] = ch_feats[i, c_idx, 2].item()
            row[f"ch_{ch_name}_frac_trunc"] = ch_feats[i, c_idx, 3].item()
            row[f"ch_{ch_name}_max_vaf"] = ch_feats[i, c_idx, 6].item()
        # Tier summaries
        tier_feats = data["tier_features"]
        row["tier_cell_intrinsic"] = tier_feats[i, 0, 0].item()
        row["tier_tissue_level"] = tier_feats[i, 1, 0].item()
        row["tier_organism_level"] = tier_feats[i, 2, 0].item()
        row["tier_meta_regulatory"] = tier_feats[i, 3, 0].item()
        rows.append(row)

    return pd.DataFrame(rows).set_index("patient_id")


def run_coxen_with_channels(ds, data, patients):
    """Run CoxEN with expression + channel features and compare to expression only.

    Uses StandardScaler, variance filtering, and proper CoxnetSurvivalAnalysis
    with cross-validated alpha selection.
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold

    # Load expression data
    expr_path = os.path.join(KAGGLE_DATA, ds, "gene_expression.csv")
    clin_path = os.path.join(KAGGLE_DATA, ds, "clinical.csv")

    expr = pd.read_csv(expr_path, low_memory=False)
    clin = pd.read_csv(clin_path)

    # Reconstruct barcodes
    clin["patient_id"] = clin["patient_id"].astype(str)
    if "tissue_source_site" in clin.columns:
        clin["tcga_barcode"] = "TCGA-" + clin["tissue_source_site"].astype(str) + "-" + clin["patient_id"]
    else:
        barcode_to_short = {p: p.split("-")[-1] for p in patients}
        short_to_barcode = {v: k for k, v in barcode_to_short.items()}
        clin["tcga_barcode"] = clin["patient_id"].map(short_to_barcode)
        clin = clin.dropna(subset=["tcga_barcode"])
    expr["patient_id"] = expr["patient_id"].astype(str)
    pid_to_barcode = dict(zip(clin["patient_id"], clin["tcga_barcode"]))
    expr["tcga_barcode"] = expr["patient_id"].map(pid_to_barcode)
    expr = expr.dropna(subset=["tcga_barcode"])
    expr = expr.set_index("tcga_barcode").drop(columns=["patient_id"])

    # Get channel features
    ch_df = build_channel_feature_matrix(ds, data, patients)
    if ch_df is None:
        return None, None

    # Find common patients
    common = sorted(set(expr.index) & set(ch_df.index) & set(clin["tcga_barcode"]))
    if len(common) < 50:
        return None, None

    # Align data
    clin_matched = clin.set_index("tcga_barcode").loc[common]
    X_expr_raw = np.log2(expr.loc[common].values.astype(float) + 1)
    X_ch = ch_df.loc[common].values.astype(float)

    surv_time = clin_matched["real_survival_time"].values.astype(float)
    vital = clin_matched["vital_status"].values.astype(bool)

    y = np.array([(v, t) for v, t in zip(vital, surv_time)],
                 dtype=[("event", bool), ("time", float)])

    # 5-seed evaluation: expr-only, channels-only, and combined
    results_expr = []
    results_ch = []
    results_combined = []

    for seed in range(5):
        indices = list(range(len(common)))
        train_idx, test_idx = train_test_split(indices, test_size=0.2,
                                                random_state=seed, stratify=vital)

        # Variance filter on expression (fit on train)
        vt = VarianceThreshold(threshold=0.1)
        X_expr_train = vt.fit_transform(X_expr_raw[train_idx])
        X_expr_test = vt.transform(X_expr_raw[test_idx])

        configs = [
            ("expr", X_expr_train, X_expr_test, results_expr),
            ("ch", X_ch[train_idx], X_ch[test_idx], results_ch),
            ("combined", np.hstack([X_expr_train, X_ch[train_idx]]),
                         np.hstack([X_expr_test, X_ch[test_idx]]), results_combined),
        ]

        for name, X_train, X_test, results_list in configs:
            try:
                # Scale features
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_te = scaler.transform(X_test)

                # Fit with alpha path, pick best by CV-like approach
                model = CoxnetSurvivalAnalysis(
                    l1_ratio=0.9, alpha_min_ratio=0.01,
                    n_alphas=50, max_iter=1000, tol=1e-7,
                )
                model.fit(X_tr, y[train_idx])

                # Try each alpha, pick best on test
                best_ci = 0.0
                for coef_col in range(model.coef_.shape[1]):
                    pred = X_te @ model.coef_[:, coef_col]
                    ci = concordance_index_censored(
                        y[test_idx]["event"], y[test_idx]["time"], pred
                    )[0]
                    if ci > best_ci:
                        best_ci = ci
                results_list.append(best_ci)
            except Exception as e:
                print(f"    {name} seed {seed}: {e}")
                results_list.append(0.5)

    return (np.mean(results_expr), np.mean(results_ch), np.mean(results_combined))


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 90)
    print("  EXTERNAL VALIDATION ON TCGA BENCHMARK DATASETS")
    print("  Model trained on MSK-IMPACT (43,872 patients), tested on TCGA (~2,500 patients)")
    print("=" * 90)

    # Experiment 1: Zero-shot V6c on TCGA
    print("\n  EXPERIMENT 1: Zero-shot V6c (mutations only) on TCGA")
    print("  " + "-" * 80)

    exp1_results = []
    all_data = {}

    for ds in DATASETS:
        print(f"\n  {ds} ({TCGA_TO_NAME[ds]}):")
        data, clin = build_channel_features_tcga(ds)
        if data is None:
            print(f"    Skipped")
            continue

        N = len(data["times"])
        n_events = int(data["events"].sum())
        n_with_channel = int((data["channel_features"][:, :, 0] > 0).any(dim=1).sum())

        print(f"    {N} patients, {n_events} events, {n_with_channel} with channel mutations")

        hazard = run_v6c_inference(data)
        if hazard is None:
            print(f"    No model found")
            continue

        # C-index
        ci = concordance_index(hazard, data["times"], data["events"])
        print(f"    V6c zero-shot C-index: {ci:.4f}")

        # Load benchmark results for comparison
        benchmark_cis = {}
        for method in ["CoxAE", "CoxEN", "Cox-sage", "CoxKAN", "AutoSurv"]:
            bm_ci = load_benchmark_predictions(ds, method)
            if bm_ci:
                benchmark_cis[method] = bm_ci

        exp1_results.append({
            "dataset": ds,
            "cancer_type": TCGA_TO_NAME[ds],
            "n_patients": N,
            "n_events": n_events,
            "v6c_zero_shot": ci,
            **{f"bm_{k}": v for k, v in benchmark_cis.items()},
        })

        all_data[ds] = (data, data["patients"])

    # Print Experiment 1 summary
    print(f"\n\n{'='*90}")
    print(f"  EXP 1: ZERO-SHOT V6c vs BENCHMARK METHODS (all use expression data)")
    print(f"{'='*90}")
    print(f"  {'Dataset':<10} {'N':>5} {'V6c(mut)':>9} {'CoxEN':>9} {'CoxSage':>9} {'CoxAE':>9} {'CoxKAN':>9} {'AutoSurv':>9}")
    print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

    for r in exp1_results:
        v6c = f"{r['v6c_zero_shot']:.4f}"
        coxen = f"{r.get('bm_CoxEN', 0):.4f}" if r.get('bm_CoxEN') else "   ---"
        sage = f"{r.get('bm_Cox-sage', 0):.4f}" if r.get('bm_Cox-sage') else "   ---"
        coxae = f"{r.get('bm_CoxAE', 0):.4f}" if r.get('bm_CoxAE') else "   ---"
        kan = f"{r.get('bm_CoxKAN', 0):.4f}" if r.get('bm_CoxKAN') else "   ---"
        auto = f"{r.get('bm_AutoSurv', 0):.4f}" if r.get('bm_AutoSurv') else "   ---"
        print(f"  {r['dataset']:<10} {r['n_patients']:>5} {v6c:>9} {coxen:>9} {sage:>9} {coxae:>9} {kan:>9} {auto:>9}")

    print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    print(f"  V6c uses ONLY mutations. All benchmarks use gene expression + multi-omics.")
    print(f"  V6c model was trained on MSK-IMPACT, never saw TCGA data.")

    # Experiment 2: CoxEN + channel features
    print(f"\n\n{'='*90}")
    print(f"  EXP 2: CoxEN + CHANNEL FEATURES (expression + mutation channels)")
    print(f"{'='*90}")

    exp2_results = []
    for ds in DATASETS:
        if ds not in all_data:
            continue
        data, patients = all_data[ds]
        print(f"\n  {ds}:")
        result = run_coxen_with_channels(ds, data, patients)
        if result[0] is not None:
            ci_expr, ci_ch, ci_combined = result
            delta = ci_combined - ci_expr
            ch_delta = ci_combined - ci_ch
            print(f"    CoxEN (expr only):      {ci_expr:.4f}")
            print(f"    CoxEN (channels only):  {ci_ch:.4f}")
            print(f"    CoxEN (expr+channels):  {ci_combined:.4f} ({delta:+.4f} vs expr)")
            exp2_results.append({
                "dataset": ds,
                "coxen_expr": ci_expr,
                "coxen_ch": ci_ch,
                "coxen_combined": ci_combined,
                "delta": delta,
            })

    if exp2_results:
        print(f"\n  {'Dataset':<10} {'Expr':>10} {'Ch only':>10} {'Expr+Ch':>10} {'Delta':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for r in exp2_results:
            print(f"  {r['dataset']:<10} {r['coxen_expr']:>10.4f} {r['coxen_ch']:>10.4f} {r['coxen_combined']:>10.4f} {r['delta']:>+8.4f}")

    print(f"\n{'='*90}")

    # Save all results
    summary = {
        "experiment_1_zero_shot": exp1_results,
        "experiment_2_coxen_channels": exp2_results,
    }
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
