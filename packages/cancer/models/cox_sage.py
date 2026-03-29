"""
Cox-SAGE: GraphSAGE on heterogeneous mutation graph with Cox loss.

Message passing over the full graph topology:
  Patient → Gene (via HAS_MUTATION)
  Gene ↔ Gene (via COOCCURS, PPI, ATTENDS_TO, COUPLES)
  Patient → MutationGroup (via MEMBER_OF)
  MutationGroup → Gene (via HAS_GENE)
  MutationGroup ↔ MutationGroup (via GROUP_ATTENDS_TO)

Patient embeddings from 2-layer heterogeneous SAGE → Cox partial likelihood.

Per-patient concordance for histogram comparison.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.data import HeteroData

from gnn.config import GNN_CACHE

GRAPH_PATH = os.path.join(GNN_CACHE, "hetero_graph.pt")


class CoxSAGE(nn.Module):
    """Heterogeneous GraphSAGE with Cox survival head."""

    def __init__(self, metadata, gene_dim=11, patient_dim=3, group_dim=4,
                 hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden = hidden

        # Input projections per node type
        self.gene_enc = nn.Sequential(
            nn.Linear(gene_dim, hidden), nn.ELU(), nn.Dropout(dropout))
        self.patient_enc = nn.Sequential(
            nn.Linear(patient_dim, hidden), nn.ELU(), nn.Dropout(dropout))
        self.group_enc = nn.Sequential(
            nn.Linear(group_dim, hidden), nn.ELU(), nn.Dropout(dropout))

        # Heterogeneous SAGE layers
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src, rel, dst = edge_type
                conv_dict[edge_type] = SAGEConv(
                    (hidden, hidden), hidden, aggr='mean')
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        self.norms = nn.ModuleList([
            nn.ModuleDict({
                'gene': nn.LayerNorm(hidden),
                'patient': nn.LayerNorm(hidden),
                'mutation_group': nn.LayerNorm(hidden),
            })
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Cox head: patient embedding → scalar hazard
        self.cox_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        # Encode node features
        h = {
            'gene': self.gene_enc(x_dict['gene']),
            'patient': self.patient_enc(x_dict['patient']),
            'mutation_group': self.group_enc(x_dict['mutation_group']),
        }

        # Message passing
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            # Residual + norm
            for ntype in h:
                if ntype in h_new:
                    h[ntype] = self.norms[i][ntype](h[ntype] + h_new[ntype])
                    h[ntype] = F.elu(h[ntype])
                    h[ntype] = self.dropout(h[ntype])

        # Patient hazard scores
        hazard = self.cox_head(h['patient']).squeeze(-1)
        return hazard


def cox_ph_loss(hazard, time, event):
    """Cox partial likelihood loss (Breslow approximation).

    Args:
        hazard: (N,) predicted log-hazard
        time: (N,) survival time
        event: (N,) event indicator (1=event, 0=censored)
    """
    # Sort by descending time
    order = torch.argsort(time, descending=True)
    hazard = hazard[order]
    event = event[order].float()

    # Log-sum-exp trick for numerical stability
    log_cumsum_h = torch.logcumsumexp(hazard, dim=0)

    # Loss = -sum_{events}(h_i - log(sum_{j in risk set} exp(h_j)))
    loss = -torch.mean((hazard - log_cumsum_h) * event)
    return loss


def concordance_pairs(hazard, time, event):
    """Compute per-patient concordance contribution.

    Returns array of per-patient scores: +1 (concordant), -1 (discordant),
    0 (tied or non-comparable). Used for histogram comparison.
    """
    n = len(hazard)
    scores = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] > time[i]:
                # j survived longer than i (who had event)
                # concordant if hazard[i] > hazard[j]
                if hazard[i] > hazard[j]:
                    scores[i] += 1
                    scores[j] += 1
                elif hazard[i] < hazard[j]:
                    scores[i] -= 1
                    scores[j] -= 1
                counts[i] += 1
                counts[j] += 1

    # Normalize to [-1, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        per_patient = np.where(counts > 0, scores / counts, 0.0)
    return per_patient


def calibration_comparison(models_dict, time, event, n_bins=10):
    """Compare pessimism across models via decile calibration.

    For each model's predicted hazards, bin patients into deciles and compare
    predicted risk rank to observed event rate within each bin.

    Args:
        models_dict: {'model_name': hazard_array} — predicted log-hazards
        time: (N,) survival time
        event: (N,) event indicator
        n_bins: number of risk bins (default 10 = deciles)

    Returns:
        dict of {model_name: {
            'mean_hazard': float,        # overall mean predicted hazard
            'hazard_std': float,         # spread of predictions
            'pessimism_ratio': float,    # mean(predicted rank) / mean(observed rate)
            'top_decile_event_rate': float,  # event rate in highest-risk bin
            'bottom_decile_event_rate': float,
            'calibration_slope': float,  # >1 = pessimistic, <1 = optimistic
            'bins': list of dicts,       # per-bin detail
        }}
    """
    from scipy import stats as sp_stats

    results = {}
    for name, hazard in models_dict.items():
        h = np.asarray(hazard, dtype=np.float64)
        t = np.asarray(time, dtype=np.float64)
        e = np.asarray(event, dtype=np.float64)

        # Rank patients by predicted hazard (higher = more risk)
        order = np.argsort(h)
        bin_size = len(h) // n_bins
        bins = []

        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else len(h)
            idx = order[start:end]

            obs_event_rate = e[idx].mean()
            obs_median_time = np.median(t[idx])
            mean_h = h[idx].mean()

            bins.append({
                'bin': b,
                'n': len(idx),
                'mean_hazard': mean_h,
                'event_rate': obs_event_rate,
                'median_time': obs_median_time,
            })

        # Calibration slope: regress observed event rate on predicted rank
        predicted_ranks = np.array([b['mean_hazard'] for b in bins])
        observed_rates = np.array([b['event_rate'] for b in bins])
        slope, intercept, r, p, se = sp_stats.linregress(predicted_ranks, observed_rates)

        # Pessimism: does the model spread predictions wider than outcomes justify?
        # slope > 0 means higher predicted hazard → higher observed events (good)
        # Compare top vs bottom decile separation
        top_rate = bins[-1]['event_rate']
        bottom_rate = bins[0]['event_rate']
        separation = top_rate - bottom_rate  # how well it stratifies

        results[name] = {
            'mean_hazard': h.mean(),
            'hazard_std': h.std(),
            'calibration_slope': slope,
            'calibration_r2': r ** 2,
            'top_decile_event_rate': top_rate,
            'bottom_decile_event_rate': bottom_rate,
            'stratification': separation,
            'bins': bins,
        }

    return results


def print_calibration(cal_results):
    """Print calibration comparison table."""
    print(f"\n{'='*70}")
    print("  CALIBRATION / PESSIMISM COMPARISON")
    print(f"{'='*70}\n")

    # Header
    print(f"  {'Model':30s} {'Mean h':>8s} {'Std h':>8s} {'Cal slope':>10s} "
          f"{'R²':>6s} {'Top 10%':>8s} {'Bot 10%':>8s} {'Strat':>8s}")
    print(f"  {'-'*88}")

    for name, r in cal_results.items():
        print(f"  {name:30s} {r['mean_hazard']:8.3f} {r['hazard_std']:8.3f} "
              f"{r['calibration_slope']:10.4f} {r['calibration_r2']:6.3f} "
              f"{r['top_decile_event_rate']:8.3f} {r['bottom_decile_event_rate']:8.3f} "
              f"{r['stratification']:8.3f}")

    # Interpretation
    print(f"\n  Interpretation:")
    names = list(cal_results.keys())
    if len(names) >= 2:
        a, b = names[0], names[1]
        ra, rb = cal_results[a], cal_results[b]
        if ra['hazard_std'] > rb['hazard_std']:
            print(f"    {a} spreads predictions wider (std {ra['hazard_std']:.3f} vs {rb['hazard_std']:.3f})")
            more_spread = a
        else:
            print(f"    {b} spreads predictions wider (std {rb['hazard_std']:.3f} vs {ra['hazard_std']:.3f})")
            more_spread = b

        if ra['stratification'] > rb['stratification']:
            print(f"    {a} stratifies better (top-bottom gap {ra['stratification']:.3f} vs {rb['stratification']:.3f})")
        else:
            print(f"    {b} stratifies better (top-bottom gap {rb['stratification']:.3f} vs {ra['stratification']:.3f})")

        # Pessimism = wider spread without proportionally better stratification
        for name, r in cal_results.items():
            efficiency = r['stratification'] / max(r['hazard_std'], 1e-8)
            label = "pessimistic" if efficiency < 0.5 else "well-calibrated" if efficiency < 1.5 else "optimistic"
            print(f"    {name}: stratification efficiency = {efficiency:.3f} ({label})")

    # Per-decile detail for each model
    for name, r in cal_results.items():
        print(f"\n  {name} — decile detail:")
        print(f"    {'Bin':>4s} {'N':>6s} {'Mean h':>8s} {'Event %':>8s} {'Med OS':>8s}")
        for b in r['bins']:
            print(f"    {b['bin']:4d} {b['n']:6d} {b['mean_hazard']:8.3f} "
                  f"{b['event_rate']:8.3f} {b['median_time']:8.1f}")


def train_and_evaluate():
    """Train Cox-SAGE on the graph, compare to baselines."""
    from sklearn.model_selection import StratifiedKFold
    from sksurv.metrics import concordance_index_censored

    print(f"\n{'='*70}")
    print("  COX-SAGE: GraphSAGE + Cox Loss on Heterogeneous Graph")
    print(f"{'='*70}\n")

    # Load graph
    t0 = time.time()
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"  Loaded graph: {sum(data[nt].x.shape[0] for nt in data.node_types):,} nodes, "
          f"{sum(data[et].edge_index.shape[1] for et in data.edge_types):,} edges",
          flush=True)

    os_months = data['patient'].os_months.numpy()
    events = data['patient'].event.numpy()
    n_patients = len(os_months)

    # Filter to valid patients (os > 0)
    valid_mask = os_months > 0
    valid_idx = np.where(valid_mask)[0]
    print(f"  Valid patients (OS > 0): {len(valid_idx):,}", flush=True)

    # Prepare edge_index_dict (no edge_attr for now, SAGE doesn't use it natively)
    edge_index_dict = {}
    for etype in data.edge_types:
        edge_index_dict[etype] = data[etype].edge_index

    x_dict = {
        'gene': data['gene'].x,
        'patient': data['patient'].x,
        'mutation_group': data['mutation_group'].x.clone(),  # will be overwritten per fold
    }

    # Build group → patient membership for per-fold stat recomputation
    member_of_ei = data['patient', 'member_of', 'mutation_group'].edge_index
    n_groups = data['mutation_group'].x.shape[0]
    group_patients = [[] for _ in range(n_groups)]  # group_id -> [patient_ids]
    for i in range(member_of_ei.shape[1]):
        pid = member_of_ei[0, i].item()
        gid = member_of_ei[1, i].item()
        group_patients[gid].append(pid)

    # Base group features (n_patients, n_genes) — these don't leak
    group_base = data['mutation_group'].x.clone()

    # Smoothing strength: a group needs ~20 train patients before we fully
    # trust its own stats. Below that, blend toward the global train prior.
    SMOOTH_LAMBDA = 20.0

    def recompute_group_features(train_mask_np):
        """Recompute event_rate and median_os from train patients only.

        Uses empirical Bayes shrinkage: for group with k train patients,
        stat = (k / (k + λ)) * group_stat + (λ / (k + λ)) * global_stat

        This stabilizes small groups without leaking validation outcomes.
        """
        # Global priors from all train patients
        train_idx_all = np.where(train_mask_np)[0]
        global_event_rate = events[train_idx_all].mean()
        global_median_os = float(np.median(os_months[train_idx_all])) / 60.0

        gx = group_base.clone()
        for gid in range(n_groups):
            pids = group_patients[gid]
            if not pids:
                gx[gid, 0] = 0.0
                gx[gid, 1] = global_event_rate
                gx[gid, 2] = global_median_os
                continue

            train_pids = [p for p in pids if train_mask_np[p]]
            k = len(train_pids)

            if k == 0:
                gx[gid, 0] = 0.0
                gx[gid, 1] = global_event_rate
                gx[gid, 2] = global_median_os
                continue

            # Group-level stats from train patients
            group_event_rate = events[train_pids].sum() / k
            group_median_os = float(np.median(os_months[train_pids])) / 60.0

            # Shrinkage weight: trust group more as k grows
            w = k / (k + SMOOTH_LAMBDA)

            gx[gid, 0] = min(k / 100.0, 5.0)
            gx[gid, 1] = w * group_event_rate + (1 - w) * global_event_rate
            gx[gid, 2] = w * group_median_os + (1 - w) * global_median_os
        return gx

    # Cancer type labels for per-cancer analysis
    cancer_type_idx = data['patient'].cancer_type_idx.numpy()
    ct_map_inv = {v: k for k, v in data.cancer_type_map.items()}

    # Cross-validation — collect out-of-fold predictions for every patient
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_c_indices = []
    oof_hazard = np.full(n_patients, np.nan)  # out-of-fold Cox-SAGE predictions

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(valid_idx, events[valid_idx])):

        train_patients = valid_idx[train_idx]
        val_patients = valid_idx[val_idx]

        print(f"\n  Fold {fold}: train={len(train_patients):,}, "
              f"val={len(val_patients):,}", flush=True)

        # Create train mask
        train_mask = torch.zeros(n_patients, dtype=torch.bool)
        train_mask[train_patients] = True

        # Recompute group features from train patients only (no target leakage)
        train_mask_np = train_mask.numpy()
        x_dict['mutation_group'] = recompute_group_features(train_mask_np)

        # Model
        model = CoxSAGE(
            metadata=data.metadata(),
            gene_dim=data['gene'].x.shape[1],
            patient_dim=data['patient'].x.shape[1],
            group_dim=data['mutation_group'].x.shape[1],
            hidden=64, n_layers=2, dropout=0.1,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        time_t = torch.tensor(os_months, dtype=torch.float32)
        event_t = torch.tensor(events, dtype=torch.long)

        best_val_c = 0.0
        best_state = None
        patience = 20
        no_improve = 0

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()

            hazard = model(x_dict, edge_index_dict)

            # Train loss: only on train patients
            train_h = hazard[train_mask]
            train_t = time_t[train_mask]
            train_e = event_t[train_mask]

            loss = cox_ph_loss(train_h, train_t, train_e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Validate every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    h_all = model(x_dict, edge_index_dict).numpy()

                h_val = h_all[val_patients]
                t_val = os_months[val_patients]
                e_val = events[val_patients].astype(bool)

                try:
                    c = concordance_index_censored(e_val, t_val, h_val)[0]
                except Exception:
                    c = 0.5

                if c > best_val_c:
                    best_val_c = c
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, "
                          f"val_C={c:.4f} (best={best_val_c:.4f})", flush=True)

                if no_improve >= patience // 10:
                    break

        # Evaluate best model
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            h_all = model(x_dict, edge_index_dict).numpy()

        # Store out-of-fold predictions
        oof_hazard[val_patients] = h_all[val_patients]

        h_val = h_all[val_patients]
        t_val = os_months[val_patients]
        e_val = events[val_patients].astype(bool)

        c = concordance_index_censored(e_val, t_val, h_val)[0]
        all_c_indices.append(c)
        print(f"  Fold {fold}: best C = {c:.4f}", flush=True)

    # Summary
    mean_c = np.mean(all_c_indices)
    std_c = np.std(all_c_indices)

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Cox-SAGE C-index:             {mean_c:.4f} +/- {std_c:.4f}")
    print(f"  Per-fold: {[f'{c:.4f}' for c in all_c_indices]}")
    print(f"")
    print(f"  BASELINES:")
    print(f"  Atlas lookup (zero-param):    C = 0.577")
    print(f"  Graph walk + Cox (linear):    C = 0.640")
    print(f"  AtlasTransformer V1 (neural): C = 0.673")
    print(f"  Total time: {time.time() - t0:.1f}s")

    # ===== Per-cancer-type C-index =====
    h_atlas = -x_dict['patient'][:, 0].numpy()

    print(f"\n{'='*70}")
    print("  PER-CANCER-TYPE CONCORDANCE")
    print(f"{'='*70}\n")

    print(f"  {'Cancer Type':40s} {'N':>6s} {'Events':>7s} {'Cox-SAGE':>9s} {'Atlas':>9s} {'Lift':>7s}")
    print(f"  {'-'*79}")

    cancer_rows = []
    for ct_i in sorted(ct_map_inv.keys()):
        ct_name = ct_map_inv[ct_i]
        mask = (cancer_type_idx == ct_i) & valid_mask & ~np.isnan(oof_hazard)
        idx = np.where(mask)[0]

        n_ct = len(idx)
        n_events = int(events[idx].sum())

        # Need >= 5 events and >= 5 censored for meaningful C-index
        if n_events < 5 or (n_ct - n_events) < 5:
            continue

        e_ct = events[idx].astype(bool)
        t_ct = os_months[idx]

        try:
            c_sage = concordance_index_censored(e_ct, t_ct, oof_hazard[idx])[0]
        except Exception:
            c_sage = np.nan

        try:
            c_atlas = concordance_index_censored(e_ct, t_ct, h_atlas[idx])[0]
        except Exception:
            c_atlas = np.nan

        lift = c_sage - c_atlas if not (np.isnan(c_sage) or np.isnan(c_atlas)) else np.nan

        cancer_rows.append({
            'cancer_type': ct_name, 'n': n_ct, 'n_events': n_events,
            'c_sage': c_sage, 'c_atlas': c_atlas, 'lift': lift,
        })

    # Sort by Cox-SAGE C descending
    cancer_rows.sort(key=lambda r: r['c_sage'] if not np.isnan(r['c_sage']) else 0, reverse=True)

    for r in cancer_rows:
        lift_str = f"{r['lift']:+7.3f}" if not np.isnan(r['lift']) else "    N/A"
        sage_str = f"{r['c_sage']:9.4f}" if not np.isnan(r['c_sage']) else "      N/A"
        atlas_str = f"{r['c_atlas']:9.4f}" if not np.isnan(r['c_atlas']) else "      N/A"
        print(f"  {r['cancer_type']:40s} {r['n']:6d} {r['n_events']:7d} "
              f"{sage_str} {atlas_str} {lift_str}")

    # Aggregate stats
    valid_lifts = [r['lift'] for r in cancer_rows if not np.isnan(r['lift'])]
    n_better = sum(1 for l in valid_lifts if l > 0)
    n_worse = sum(1 for l in valid_lifts if l < 0)
    print(f"\n  {len(cancer_rows)} cancer types with sufficient data")
    print(f"  Cox-SAGE beats atlas in {n_better}/{len(valid_lifts)} types, "
          f"loses in {n_worse}/{len(valid_lifts)}")
    if valid_lifts:
        print(f"  Mean lift: {np.mean(valid_lifts):+.4f}, "
              f"median lift: {np.median(valid_lifts):+.4f}")

    # ===== Calibration / pessimism comparison =====
    # Use only patients with out-of-fold predictions (all valid patients should have them)
    cal_mask = valid_mask & ~np.isnan(oof_hazard)
    cal_idx = np.where(cal_mask)[0]

    cal = calibration_comparison(
        {'Cox-SAGE': oof_hazard[cal_idx],
         'Atlas (zero-param)': h_atlas[cal_idx]},
        os_months[cal_idx],
        events[cal_idx],
    )
    print_calibration(cal)


if __name__ == "__main__":
    train_and_evaluate()
