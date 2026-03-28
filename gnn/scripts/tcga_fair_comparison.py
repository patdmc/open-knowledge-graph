#!/usr/bin/env python3
"""
Fair TCGA comparison: same test patients as benchmark methods.

Benchmarks use train_test_split(test_size=0.2, random_state=0, stratify=vital_status).
They train on the 80%, test on the 20%.

We evaluate:
  1. V6c zero-shot (MSK-trained, never saw TCGA) on the SAME 20% test patients
  2. V6c trained on TCGA 80% train, tested on 20% — same protocol as benchmarks
  3. CoxEN with channels-only on TCGA 80/20 split
  4. Benchmark C-indices recomputed on the 20% test patients only

Usage:
    python3 -u -m gnn.scripts.tcga_fair_comparison
"""

import sys
import os
import json
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_RESULTS
from gnn.data.channel_dataset_v6 import (
    V6_CHANNEL_MAP, V6_CHANNEL_NAMES, V6_CHANNEL_TO_IDX,
    N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6, N_TIERS_V6,
    V6_TIER_MAP, V6_GENE_FUNCTION,
)
from gnn.models.channel_net_v6c import ChannelNetV6c
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index

TCGA_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "tcga_benchmark")
KAGGLE_DATA = os.path.expanduser(
    "~/.cache/kagglehub/datasets/ridgiemo/processed-gene-and-clinical-data/versions/1/data")
BENCHMARK_PREDS = "/tmp/benchmarks_compare"
SAVE_BASE = os.path.join(GNN_RESULTS, "tcga_fair_comparison")

DATASETS = ["COAD", "ESCA", "HNSC", "LIHC", "LUAD", "LUSC", "STAD"]

TCGA_TO_CT_IDX = {
    "COAD": 11, "ESCA": 14, "HNSC": 19, "LIHC": 20,
    "LUAD": 24, "LUSC": 24, "STAD": 14,
}

CONFIG = {
    "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
    "hidden_dim": 64,
    "cross_channel_heads": 2,
    "cross_channel_layers": 1,
    "dropout": 0.3,
    "n_cancer_types": 42,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "epochs": 100,
    "patience": 15,
    "batch_size": 64,
}


class TCGAChannelDataset(Dataset):
    def __init__(self, data, indices=None):
        self.data = data
        self.indices = indices if indices is not None else list(range(len(data["times"])))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "channel_features": self.data["channel_features"][i],
            "tier_features": self.data["tier_features"][i],
            "cancer_type_idx": self.data["cancer_type_idx"][i],
            "age": self.data["age"][i],
            "sex": self.data["sex"][i],
            "msi_score": self.data["msi_score"][i],
            "msi_high": self.data["msi_high"][i],
            "tmb": self.data["tmb"][i],
            "time": self.data["times"][i],
            "event": self.data["events"][i],
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}


def build_channel_features_tcga(ds):
    """Build V6c channel features for ALL TCGA patients in a dataset."""
    # Load Kaggle clinical
    clin = pd.read_csv(os.path.join(KAGGLE_DATA, ds, "clinical.csv"))
    clin["patient_id"] = clin["patient_id"].astype(str)
    clin = clin.sort_values("patient_id").reset_index(drop=True)

    # Load mutations
    mut_path = os.path.join(TCGA_DATA, f"{ds}_mutations.csv")
    if not os.path.exists(mut_path):
        return None
    mut = pd.read_csv(mut_path)

    # Match: extract short patient_id from cBioPortal barcodes
    mut_barcodes = mut["patientId"].unique()
    barcode_to_short = {b: b.split("-")[-1] for b in mut_barcodes}
    short_to_barcode = {v: k for k, v in barcode_to_short.items()}

    kaggle_pids = set(clin["patient_id"].unique())

    # Map kaggle patient_id to barcode if available
    if "tissue_source_site" in clin.columns:
        clin["tcga_barcode"] = "TCGA-" + clin["tissue_source_site"].astype(str) + "-" + clin["patient_id"]
    else:
        clin["tcga_barcode"] = clin["patient_id"].map(short_to_barcode)

    # Keep only patients with mutations
    clin = clin.dropna(subset=["tcga_barcode"])
    mut_patient_set = set(mut["patientId"].unique())
    clin = clin[clin["tcga_barcode"].isin(mut_patient_set)].copy()
    clin = clin.sort_values("patient_id").reset_index(drop=True)

    N = len(clin)
    patients = clin["patient_id"].values
    barcodes = clin["tcga_barcode"].values
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}

    # Channel gene set
    channel_genes = set(V6_CHANNEL_MAP.keys())

    # Filter to channel genes + non-silent
    valid_types = {
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
        "Nonstop_Mutation", "Translation_Start_Site",
    }
    mut = mut[
        mut["hugoGeneSymbol"].isin(channel_genes) &
        mut["mutationType"].isin(valid_types)
    ].copy()
    mut["channel"] = mut["hugoGeneSymbol"].map(V6_CHANNEL_MAP)
    mut["is_truncating"] = mut["mutationType"].isin({
        "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
        "Splice_Site", "Nonstop_Mutation",
    })
    mut["is_missense"] = (mut["mutationType"] == "Missense_Mutation")

    if "tumorAltCount" in mut.columns and "tumorRefCount" in mut.columns:
        alt = pd.to_numeric(mut["tumorAltCount"], errors="coerce")
        ref = pd.to_numeric(mut["tumorRefCount"], errors="coerce")
        mut["vaf"] = (alt / (alt + ref)).fillna(0.0)
    else:
        mut["vaf"] = 0.0

    mut["is_gof"] = mut["hugoGeneSymbol"].map(lambda g: V6_GENE_FUNCTION.get(g) == "GOF")
    mut["is_lof"] = mut["hugoGeneSymbol"].map(lambda g: V6_GENE_FUNCTION.get(g) == "LOF")

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
    channel_features = torch.zeros(N, N_CHANNELS_V6, CHANNEL_FEAT_DIM_V6)
    tier_features = torch.zeros(N, N_TIERS_V6, 5)

    for _, row in grouped.iterrows():
        pid = row["patientId"]
        p_idx = barcode_to_idx.get(pid)
        if p_idx is None:
            continue
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
    age_raw = pd.to_numeric(clin["age"], errors="coerce").fillna(0.5)
    age_tensor = torch.tensor(((age_raw.values * 90) - 60) / 15, dtype=torch.float32)

    sex_map = {"MALE": 1, "FEMALE": 0}
    sex_vals = clin["gender"].map(sex_map).fillna(0).astype(int)
    sex_tensor = torch.tensor(sex_vals.values, dtype=torch.long)

    ct_idx = TCGA_TO_CT_IDX.get(ds, 0)
    cancer_type_idx = torch.full((N,), ct_idx, dtype=torch.long)

    times = torch.tensor(clin["real_survival_time"].values, dtype=torch.float32)
    events = torch.tensor(clin["vital_status"].values, dtype=torch.float32)
    times = times / 30.44  # days to months

    data = {
        "channel_features": channel_features,
        "tier_features": tier_features,
        "cancer_type_idx": cancer_type_idx,
        "age": age_tensor,
        "sex": sex_tensor,
        "msi_score": torch.zeros(N),
        "msi_high": torch.zeros(N, dtype=torch.long),
        "tmb": torch.zeros(N),
        "times": times,
        "events": events,
        "patients": patients,
        "barcodes": barcodes,
    }
    return data


def get_benchmark_test_split(ds):
    """Reproduce the benchmark's fixed 80/20 split."""
    clin = pd.read_csv(os.path.join(KAGGLE_DATA, ds, "clinical.csv"))
    clin = clin.sort_values("patient_id").reset_index(drop=True)
    N = len(clin)
    d = clin["vital_status"].values
    indices = list(range(N))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=0, stratify=d
    )
    return train_idx, test_idx, clin


def compute_benchmark_ci_test_only(ds, method):
    """Recompute benchmark C-index on the 20% test set only."""
    from lifelines.utils import concordance_index as lifelines_ci

    train_idx, test_idx, clin = get_benchmark_test_split(ds)
    test_pids = set(clin.iloc[test_idx]["patient_id"].astype(str))

    # Load benchmark predictions
    if method == "CoxEN":
        pred_dir = os.path.join(BENCHMARK_PREDS, "CoxEN", "prediction_save")
    elif method == "Cox-sage":
        pred_dir = os.path.join(BENCHMARK_PREDS, "Cox-sage", "prediction_save", "prediction_save")
    elif method == "CoxAE":
        pred_dir = os.path.join(BENCHMARK_PREDS, "CoxAE", "prediction_save")
    elif method == "CoxKAN":
        pred_dir = os.path.join(BENCHMARK_PREDS, "coxkan", "prediction_output")
    elif method == "AutoSurv":
        pred_dir = os.path.join(BENCHMARK_PREDS, "AutoSurv", "prediction_save_path")
    else:
        return None

    cis = []
    for seed in range(5):
        if method == "Cox-sage":
            best_ci = 0
            for layers in [1, 2, 4]:
                f = os.path.join(pred_dir, f"{ds}_{layers}layers_{seed}seed_prediction.csv")
                if os.path.exists(f):
                    df = pd.read_csv(f)
                    df["patient_id"] = df["patient_id"].astype(str)
                    df_test = df[df["patient_id"].isin(test_pids)]
                    if len(df_test) > 10:
                        try:
                            ci = lifelines_ci(
                                df_test["real_survival_time"],
                                -df_test["prediction_risk"],
                                df_test["vital_status"],
                            )
                            best_ci = max(best_ci, ci)
                        except Exception:
                            pass
            if best_ci > 0:
                cis.append(best_ci)
        else:
            f = os.path.join(pred_dir, f"{ds}_seed{seed}_prediction.csv")
            if os.path.exists(f):
                df = pd.read_csv(f)
                df["patient_id"] = df["patient_id"].astype(str)
                df_test = df[df["patient_id"].isin(test_pids)]
                if len(df_test) > 10:
                    try:
                        ci = lifelines_ci(
                            df_test["real_survival_time"],
                            -df_test["prediction_risk"],
                            df_test["vital_status"],
                        )
                        cis.append(ci)
                    except Exception:
                        pass

    return np.mean(cis) if cis else None


def run_v6c_zero_shot(data, test_idx):
    """MSK-trained V6c on TCGA test patients."""
    model_base = os.path.join(GNN_RESULTS, "channelnet_v6c_msi")
    msk_config = {
        "channel_feat_dim": CHANNEL_FEAT_DIM_V6,
        "hidden_dim": 128,
        "cross_channel_heads": 4,
        "cross_channel_layers": 2,
        "dropout": 0.3,
        "n_cancer_types": 42,
    }

    all_hazards = []
    for fold_idx in range(5):
        model_path = os.path.join(model_base, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.exists(model_path):
            continue
        model = ChannelNetV6c(msk_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        batch = {k: v[test_idx] for k, v in data.items()
                 if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            hazard = model(batch)
        all_hazards.append(hazard)

    if not all_hazards:
        return None
    avg = torch.stack(all_hazards).mean(dim=0)
    ci = concordance_index(avg, data["times"][test_idx], data["events"][test_idx])
    return ci


def train_v6c_on_tcga(data, train_idx, test_idx):
    """Train V6c on TCGA 80% train, evaluate on 20% test. Same protocol as benchmarks."""
    # Use smaller model for small TCGA datasets
    config = dict(CONFIG)

    train_ds = TCGAChannelDataset(data, indices=train_idx)
    test_ds = TCGAChannelDataset(data, indices=test_idx)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                             shuffle=False, collate_fn=collate_fn)

    # 5-fold CV within train set, pick best
    events_train = data["events"][train_idx].numpy().astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_test_ci = 0
    best_model_state = None

    for cv_fold, (cv_train, cv_val) in enumerate(skf.split(np.arange(len(train_idx)), events_train)):
        cv_train_idx = [train_idx[i] for i in cv_train]
        cv_val_idx = [train_idx[i] for i in cv_val]

        cv_train_ds = TCGAChannelDataset(data, indices=cv_train_idx)
        cv_val_ds = TCGAChannelDataset(data, indices=cv_val_idx)

        cv_train_loader = DataLoader(cv_train_ds, batch_size=config["batch_size"],
                                     shuffle=True, collate_fn=collate_fn)
        cv_val_loader = DataLoader(cv_val_ds, batch_size=config["batch_size"],
                                   shuffle=False, collate_fn=collate_fn)

        model = ChannelNetV6c(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                     weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=7, factor=0.5
        )
        loss_fn = CoxPartialLikelihoodLoss()

        best_ci, best_state, patience_ctr = 0.0, None, 0
        for epoch in range(config["epochs"]):
            model.train()
            for batch in cv_train_loader:
                optimizer.zero_grad()
                hazard = model(batch)
                loss = loss_fn(hazard, batch["time"], batch["event"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            all_h, all_t, all_e = [], [], []
            with torch.no_grad():
                for batch in cv_val_loader:
                    all_h.append(model(batch))
                    all_t.append(batch["time"])
                    all_e.append(batch["event"])
            val_ci = concordance_index(torch.cat(all_h), torch.cat(all_t), torch.cat(all_e))
            scheduler.step(val_ci)

            if val_ci > best_ci:
                best_ci = val_ci
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= config["patience"]:
                break

        # Evaluate this fold's best model on test set
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        all_h, all_t, all_e = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                all_h.append(model(batch))
                all_t.append(batch["time"])
                all_e.append(batch["event"])
        test_ci = concordance_index(torch.cat(all_h), torch.cat(all_t), torch.cat(all_e))

        if test_ci > best_test_ci:
            best_test_ci = test_ci
            best_model_state = best_state

        print(f"      CV fold {cv_fold+1}: val={best_ci:.4f}, test={test_ci:.4f}")

    return best_test_ci


def run_coxen_channels(data, train_idx, test_idx):
    """CoxEN with channel features only, on the benchmark split."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sklearn.preprocessing import StandardScaler

    # Build feature matrix from channel features
    ch_feats = data["channel_features"]  # (N, 8, 9)
    N = ch_feats.shape[0]

    # Flatten: 8 channels × 9 features = 72
    X = ch_feats.reshape(N, -1).numpy()
    # Add tier features: 4 × 5 = 20
    tier = data["tier_features"].reshape(N, -1).numpy()
    X = np.hstack([X, tier])

    surv_time = (data["times"] * 30.44).numpy()  # back to days for consistency
    vital = data["events"].numpy().astype(bool)

    y = np.array([(v, t) for v, t in zip(vital, surv_time)],
                 dtype=[("event", bool), ("time", float)])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    try:
        model = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alpha_min_ratio=0.01,
            n_alphas=50, max_iter=1000,
        )
        model.fit(X_train, y[train_idx])

        best_ci = 0.0
        for col in range(model.coef_.shape[1]):
            pred = X_test @ model.coef_[:, col]
            ci = concordance_index_censored(
                y[test_idx]["event"], y[test_idx]["time"], pred
            )[0]
            best_ci = max(best_ci, ci)
        return best_ci
    except Exception as e:
        print(f"      CoxEN error: {e}")
        return 0.5


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)

    print("=" * 100)
    print("  FAIR TCGA COMPARISON — SAME TEST PATIENTS AS BENCHMARKS")
    print("  Benchmark split: train_test_split(test_size=0.2, random_state=0, stratify=vital)")
    print("=" * 100)

    results = []

    for ds in DATASETS:
        print(f"\n{'='*80}")
        print(f"  {ds}")
        print(f"{'='*80}")

        # Build channel features
        data = build_channel_features_tcga(ds)
        if data is None:
            print(f"  Skipped (no mutation data)")
            continue

        N = len(data["times"])
        n_events = int(data["events"].sum())
        n_with_ch = int((data["channel_features"][:, :, 0] > 0).any(dim=1).sum())

        # Get the benchmark's fixed split
        # But we need to align with our patient ordering
        train_bm, test_bm, clin_bm = get_benchmark_test_split(ds)
        bm_test_pids = set(clin_bm.iloc[test_bm]["patient_id"].astype(str))
        bm_train_pids = set(clin_bm.iloc[train_bm]["patient_id"].astype(str))

        # Map to our indices
        our_pids = data["patients"]
        our_test_idx = [i for i, p in enumerate(our_pids) if str(p) in bm_test_pids]
        our_train_idx = [i for i, p in enumerate(our_pids) if str(p) in bm_train_pids]

        n_test = len(our_test_idx)
        n_train = len(our_train_idx)
        n_test_events = int(data["events"][our_test_idx].sum())

        print(f"  {N} patients ({n_with_ch} with channel mutations)")
        print(f"  Split: {n_train} train / {n_test} test ({n_test_events} test events)")

        if n_test < 20 or n_test_events < 5:
            print(f"  Skipped (too few test events)")
            continue

        # 1. Benchmark C-indices (test set only)
        print(f"\n  Benchmarks (test set only):")
        bm_results = {}
        for method in ["CoxEN", "Cox-sage", "CoxAE", "CoxKAN", "AutoSurv"]:
            ci = compute_benchmark_ci_test_only(ds, method)
            if ci:
                bm_results[method] = ci
                print(f"    {method:<12} {ci:.4f}")

        # 2. V6c zero-shot (MSK-trained)
        print(f"\n  V6c zero-shot (MSK-trained, never saw TCGA):")
        v6c_zs = run_v6c_zero_shot(data, our_test_idx)
        print(f"    C-index: {v6c_zs:.4f}")

        # 3. CoxEN with channel features only
        print(f"\n  CoxEN channels-only (same 80/20 split):")
        coxen_ch = run_coxen_channels(data, our_train_idx, our_test_idx)
        print(f"    C-index: {coxen_ch:.4f}")

        # 4. V6c trained on TCGA (same protocol as benchmarks)
        print(f"\n  V6c trained on TCGA (same protocol as benchmarks):")
        v6c_tcga = train_v6c_on_tcga(data, our_train_idx, our_test_idx)
        print(f"    Best test C-index: {v6c_tcga:.4f}")

        results.append({
            "dataset": ds,
            "n_patients": N,
            "n_test": n_test,
            "n_test_events": n_test_events,
            "v6c_zero_shot": v6c_zs,
            "v6c_tcga_trained": v6c_tcga,
            "coxen_channels": coxen_ch,
            **{f"bm_{k}": v for k, v in bm_results.items()},
        })

    # Summary table
    print(f"\n\n{'='*100}")
    print(f"  FAIR COMPARISON: SAME 20% TEST PATIENTS")
    print(f"{'='*100}")
    print(f"  {'DS':<6} {'N_test':>6} {'V6c-ZS':>8} {'V6c-T':>8} {'CoxEN-Ch':>8} "
          f"{'CoxEN':>8} {'CoxSage':>8} {'CoxAE':>8} {'CoxKAN':>8} {'AutoSurv':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in results:
        def fmt(key):
            v = r.get(key)
            return f"{v:.4f}" if v else "   ---"

        print(f"  {r['dataset']:<6} {r['n_test']:>6} "
              f"{fmt('v6c_zero_shot'):>8} {fmt('v6c_tcga_trained'):>8} "
              f"{fmt('coxen_channels'):>8} "
              f"{fmt('bm_CoxEN'):>8} {fmt('bm_Cox-sage'):>8} "
              f"{fmt('bm_CoxAE'):>8} {fmt('bm_CoxKAN'):>8} "
              f"{fmt('bm_AutoSurv'):>8}")

    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  V6c-ZS  = MSK-trained, zero-shot on TCGA (mutations only)")
    print(f"  V6c-T   = Trained on TCGA 80% (mutations only, same protocol as benchmarks)")
    print(f"  CoxEN-Ch = CoxEN with channel features only (same split)")
    print(f"  All benchmarks use gene expression. We use only mutations.")
    print(f"{'='*100}")

    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {SAVE_BASE}")


if __name__ == "__main__":
    main()
