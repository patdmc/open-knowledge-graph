#!/usr/bin/env python3
"""
Cox-SAGE graph topology ablation.

Question: does PPI/co-occurrence/mutation-group topology actually help,
or is Cox-SAGE getting 0.78 from (a) the bipartite patient↔gene structure
(which is just a mutation vector) and (b) mutation group features that
contain survival-derived statistics (event rate, median OS)?

Ablations:
  A. Full graph (all edges)                    — baseline
  B. Bipartite only (patient↔gene + reverse)   — is topology just a mutation vector?
  C. No mutation groups (patient↔gene + gene↔gene) — are groups doing the work?
  D. MLP baseline (patient features + mean-pooled gene features, no message passing)
  E. Full graph, zeroed mutation group survival features — leak check

Usage:
    python3 -u -m gnn.scripts.cox_sage_ablation
"""

import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored

from gnn.config import GNN_CACHE, GNN_RESULTS
from gnn.models.cox_sage import CoxSAGE, cox_ph_loss

GRAPH_PATH = os.path.join(GNN_CACHE, "hetero_graph.pt")
SAVE_BASE = os.path.join(GNN_RESULTS, "cox_sage_ablation")


class CoxMLP(nn.Module):
    """MLP baseline: no message passing. Patient gets mean-pooled gene features."""

    def __init__(self, input_dim, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def filter_edges(data, keep_relations):
    """Create a new HeteroData with only specified edge types."""
    new_data = HeteroData()

    # Copy node data
    for nt in data.node_types:
        new_data[nt].x = data[nt].x.clone()
        if hasattr(data[nt], 'os_months'):
            new_data[nt].os_months = data[nt].os_months.clone()
        if hasattr(data[nt], 'event'):
            new_data[nt].event = data[nt].event.clone()

    # Copy only specified edge types
    for et in data.edge_types:
        src, rel, dst = et
        if rel in keep_relations:
            new_data[et].edge_index = data[et].edge_index.clone()

    return new_data


def build_mlp_features(data):
    """Build flat feature vector for MLP: patient features + mean-pooled gene features."""
    patient_x = data['patient'].x  # (N_patients, 3)
    gene_x = data['gene'].x        # (N_genes, 11)
    has_mut = data[('patient', 'has_mutation', 'gene')].edge_index  # (2, E)

    n_patients = patient_x.shape[0]
    n_genes = gene_x.shape[0]

    # Mean-pool gene features per patient
    gene_sum = torch.zeros(n_patients, gene_x.shape[1])
    gene_count = torch.zeros(n_patients, 1)

    patient_idx = has_mut[0]
    gene_idx = has_mut[1]

    gene_sum.index_add_(0, patient_idx, gene_x[gene_idx])
    gene_count.index_add_(0, patient_idx, torch.ones(len(gene_idx), 1))

    gene_mean = gene_sum / gene_count.clamp(min=1)

    # Also add mutation count as feature
    mut_count = gene_count / 20.0  # normalize

    # Concatenate: patient_features + mean_gene_features + mut_count
    features = torch.cat([patient_x, gene_mean, mut_count], dim=1)
    return features


def train_sage(model, x_dict, edge_index_dict, train_mask, time_t, event_t,
               val_patients, os_months, events, n_epochs=200, lr=1e-3):
    """Train a SAGE model, return best val C-index."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_c = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        hazard = model(x_dict, edge_index_dict)
        train_h = hazard[train_mask]
        train_t = time_t[train_mask]
        train_e = event_t[train_mask]

        loss = cox_ph_loss(train_h, train_t, train_e)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

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

            if no_improve >= 2:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val_c


def train_mlp(model, features, train_mask, time_t, event_t,
              val_patients, os_months, events, n_epochs=200, lr=1e-3):
    """Train an MLP model, return best val C-index."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_c = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        hazard = model(features)
        train_h = hazard[train_mask]
        train_t = time_t[train_mask]
        train_e = event_t[train_mask]

        loss = cox_ph_loss(train_h, train_t, train_e)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                h_all = model(features).numpy()
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

            if no_improve >= 2:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val_c


def main():
    os.makedirs(SAVE_BASE, exist_ok=True)
    t0 = time.time()

    print("=" * 90)
    print("  COX-SAGE TOPOLOGY ABLATION")
    print("  Does the graph structure actually help?")
    print("=" * 90)

    # Load full graph
    data = torch.load(GRAPH_PATH, weights_only=False)
    os_months = data['patient'].os_months.numpy()
    events = data['patient'].event.numpy()
    n_patients = len(os_months)

    valid_mask = os_months > 0
    valid_idx = np.where(valid_mask)[0]

    print(f"  {n_patients} patients, {len(valid_idx)} valid")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types ({len(data.edge_types)}):")
    for et in data.edge_types:
        src, rel, dst = et
        n_edges = data[et].edge_index.shape[1]
        print(f"    {src} --{rel}--> {dst}: {n_edges:,} edges")

    # Check mutation group features for survival leak
    mg_x = data['mutation_group'].x
    print(f"\n  MutationGroup features: shape={mg_x.shape}")
    print(f"    dim0 (patient count):  mean={mg_x[:,0].mean():.3f}")
    print(f"    dim1 (event rate):     mean={mg_x[:,1].mean():.3f}  *** SURVIVAL LEAK ***")
    print(f"    dim2 (median OS):      mean={mg_x[:,2].mean():.3f}  *** SURVIVAL LEAK ***")
    print(f"    dim3 (gene count):     mean={mg_x[:,3].mean():.3f}")

    # =========================================================================
    # Build ablation variants
    # =========================================================================

    # A: Full graph
    full_edge_dict = {}
    for et in data.edge_types:
        full_edge_dict[et] = data[et].edge_index
    full_x_dict = {
        'gene': data['gene'].x,
        'patient': data['patient'].x,
        'mutation_group': data['mutation_group'].x,
    }

    # B: Bipartite only (patient↔gene)
    bipartite_data = filter_edges(data, {'has_mutation', 'rev_has_mutation'})

    # C: No mutation groups (patient↔gene + gene↔gene)
    no_groups_data = filter_edges(data, {
        'has_mutation', 'rev_has_mutation',
        'cooccurs', 'ppi', 'attends_to', 'couples',
    })

    # E: Full graph but zero out survival features in mutation groups
    noleak_x_dict = {
        'gene': data['gene'].x.clone(),
        'patient': data['patient'].x.clone(),
        'mutation_group': data['mutation_group'].x.clone(),
    }
    noleak_x_dict['mutation_group'][:, 1] = 0.0  # zero event rate
    noleak_x_dict['mutation_group'][:, 2] = 0.0  # zero median OS

    # D: MLP features
    mlp_features = build_mlp_features(data)
    print(f"\n  MLP feature dim: {mlp_features.shape[1]}")

    # =========================================================================
    # Cross-validated training
    # =========================================================================

    time_t = torch.tensor(os_months, dtype=torch.float32)
    event_t = torch.tensor(events, dtype=torch.long)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ablations = {}

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(valid_idx, events[valid_idx])):

        train_patients = valid_idx[train_idx]
        val_patients = valid_idx[val_idx]

        train_mask = torch.zeros(n_patients, dtype=torch.bool)
        train_mask[train_patients] = True

        print(f"\n  {'='*80}")
        print(f"  Fold {fold}: train={len(train_patients):,}, val={len(val_patients):,}")
        print(f"  {'='*80}")

        # A: Full graph
        print(f"  [A] Full graph...", end="", flush=True)
        model_a = CoxSAGE(
            metadata=data.metadata(),
            gene_dim=data['gene'].x.shape[1],
            patient_dim=data['patient'].x.shape[1],
            group_dim=data['mutation_group'].x.shape[1],
            hidden=64, n_layers=2, dropout=0.1,
        )
        c_a = train_sage(model_a, full_x_dict, full_edge_dict,
                         train_mask, time_t, event_t,
                         val_patients, os_months, events)
        ablations.setdefault("A_full_graph", []).append(c_a)
        print(f" C={c_a:.4f}")

        # B: Bipartite only
        print(f"  [B] Bipartite only (patient↔gene)...", end="", flush=True)
        model_b = CoxSAGE(
            metadata=bipartite_data.metadata(),
            gene_dim=data['gene'].x.shape[1],
            patient_dim=data['patient'].x.shape[1],
            group_dim=data['mutation_group'].x.shape[1],
            hidden=64, n_layers=2, dropout=0.1,
        )
        bi_edge_dict = {}
        for et in bipartite_data.edge_types:
            bi_edge_dict[et] = bipartite_data[et].edge_index
        bi_x_dict = {
            'gene': bipartite_data['gene'].x,
            'patient': bipartite_data['patient'].x,
            'mutation_group': bipartite_data['mutation_group'].x,
        }
        c_b = train_sage(model_b, bi_x_dict, bi_edge_dict,
                         train_mask, time_t, event_t,
                         val_patients, os_months, events)
        ablations.setdefault("B_bipartite_only", []).append(c_b)
        print(f" C={c_b:.4f}")

        # C: No mutation groups
        print(f"  [C] No mutation groups (patient↔gene + gene↔gene)...", end="", flush=True)
        model_c = CoxSAGE(
            metadata=no_groups_data.metadata(),
            gene_dim=data['gene'].x.shape[1],
            patient_dim=data['patient'].x.shape[1],
            group_dim=data['mutation_group'].x.shape[1],
            hidden=64, n_layers=2, dropout=0.1,
        )
        ng_edge_dict = {}
        for et in no_groups_data.edge_types:
            ng_edge_dict[et] = no_groups_data[et].edge_index
        ng_x_dict = {
            'gene': no_groups_data['gene'].x,
            'patient': no_groups_data['patient'].x,
            'mutation_group': no_groups_data['mutation_group'].x,
        }
        c_c = train_sage(model_c, ng_x_dict, ng_edge_dict,
                         train_mask, time_t, event_t,
                         val_patients, os_months, events)
        ablations.setdefault("C_no_groups", []).append(c_c)
        print(f" C={c_c:.4f}")

        # D: MLP (no graph)
        print(f"  [D] MLP (no message passing)...", end="", flush=True)
        model_d = CoxMLP(input_dim=mlp_features.shape[1], hidden=64, dropout=0.1)
        c_d = train_mlp(model_d, mlp_features, train_mask, time_t, event_t,
                        val_patients, os_months, events)
        ablations.setdefault("D_mlp_no_graph", []).append(c_d)
        print(f" C={c_d:.4f}")

        # E: Full graph, no survival leak
        print(f"  [E] Full graph, zeroed survival features...", end="", flush=True)
        model_e = CoxSAGE(
            metadata=data.metadata(),
            gene_dim=data['gene'].x.shape[1],
            patient_dim=data['patient'].x.shape[1],
            group_dim=data['mutation_group'].x.shape[1],
            hidden=64, n_layers=2, dropout=0.1,
        )
        c_e = train_sage(model_e, noleak_x_dict, full_edge_dict,
                         train_mask, time_t, event_t,
                         val_patients, os_months, events)
        ablations.setdefault("E_full_no_leak", []).append(c_e)
        print(f" C={c_e:.4f}")

    # =========================================================================
    # Results
    # =========================================================================
    print(f"\n\n{'='*90}")
    print(f"  ABLATION RESULTS")
    print(f"{'='*90}")

    print(f"\n  {'Variant':<40} {'Mean C':>8} {'Std':>7} {'Folds':>40}")
    print(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*40}")

    for name in ["A_full_graph", "B_bipartite_only", "C_no_groups",
                  "D_mlp_no_graph", "E_full_no_leak"]:
        cis = ablations[name]
        fold_str = " ".join(f"{c:.4f}" for c in cis)
        print(f"  {name:<40} {np.mean(cis):>8.4f} {np.std(cis):>7.4f} {fold_str:>40}")

    # Interpretation
    mean_a = np.mean(ablations["A_full_graph"])
    mean_b = np.mean(ablations["B_bipartite_only"])
    mean_c = np.mean(ablations["C_no_groups"])
    mean_d = np.mean(ablations["D_mlp_no_graph"])
    mean_e = np.mean(ablations["E_full_no_leak"])

    print(f"\n  INTERPRETATION:")
    print(f"  Graph topology value (A vs B):          {mean_a - mean_b:+.4f}  "
          f"{'topology helps' if mean_a > mean_b + 0.005 else 'topology is decorative'}")
    print(f"  Gene-gene edges value (C vs B):         {mean_c - mean_b:+.4f}  "
          f"{'PPI/cooccur helps' if mean_c > mean_b + 0.005 else 'PPI adds nothing'}")
    print(f"  Mutation groups value (A vs C):          {mean_a - mean_c:+.4f}  "
          f"{'groups help' if mean_a > mean_c + 0.005 else 'groups add nothing'}")
    print(f"  Message passing vs MLP (B vs D):        {mean_b - mean_d:+.4f}  "
          f"{'SAGE > MLP' if mean_b > mean_d + 0.005 else 'MLP matches'}")
    print(f"  Survival leak check (A vs E):            {mean_a - mean_e:+.4f}  "
          f"{'LEAK detected!' if mean_a > mean_e + 0.01 else 'no major leak'}")

    total_time = time.time() - t0
    print(f"\n  Total runtime: {total_time:.0f}s")

    import json
    results = {name: {"mean": float(np.mean(cis)), "std": float(np.std(cis)),
                       "folds": [float(c) for c in cis]}
               for name, cis in ablations.items()}
    with open(os.path.join(SAVE_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  Done.")


if __name__ == "__main__":
    main()
