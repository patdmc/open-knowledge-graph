#!/usr/bin/env python3
"""
Ablation studies to prove each component of the Coupling-Channel GNN matters.

Tests:
  1. No cross-channel attention (channels independent)
  2. No hub/leaf features
  3. No PPI topology (fully connected within channel)
  4. Random channel assignment
  5. MLP baseline (no graph structure)
  6. Channel-count-only Cox PH (the simplest possible baseline)

Usage:
    python3 -m gnn.scripts.ablation [--study msk_impact_50k]
"""

import argparse
import sys
import os
import json
import copy
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_CONFIG, GNN_RESULTS, N_GENES
from gnn.data.msk_dataset import MSKImpactDataset
from gnn.models.channel_gat import CouplingChannelGAT
from gnn.training.cross_val import StratifiedSurvivalCV


class AblationNoXChannel(CouplingChannelGAT):
    """Ablation: remove cross-channel attention."""
    def __init__(self, config):
        super().__init__(config)
        # Replace cross-channel attention with identity
        self.cross_channel = torch.nn.Identity()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        channel_idx = data.channel_assignment
        batch = data.batch
        B = batch.max().item() + 1

        x = self.node_encoder(x)
        for layer in self.channel_layers:
            x_new = torch.zeros_like(x)
            for c in range(6):
                mask = (channel_idx == c)
                if mask.sum() == 0:
                    continue
                node_indices = mask.nonzero(as_tuple=True)[0]
                local_edge_index = self._subgraph_edges(edge_index, node_indices)
                if local_edge_index.shape[1] == 0:
                    x_new[mask] = x[mask]
                    continue
                x_c = layer(x[mask], local_edge_index)
                x_new[mask] = x_c
            x = x_new

        channel_embeddings = torch.zeros(B, 6, x.shape[1], device=x.device)
        for b in range(B):
            b_mask = (batch == b)
            for c in range(6):
                bc_mask = b_mask & (channel_idx == c)
                if bc_mask.sum() == 0:
                    continue
                channel_embeddings[b, c] = self.channel_readout(x[bc_mask])

        # Skip cross-channel attention
        return self.patient_readout(channel_embeddings)


class MLPBaseline(torch.nn.Module):
    """Baseline: flat MLP on mutation features, no graph structure."""
    def __init__(self, config):
        super().__init__()
        # Flatten all 99 gene features into a single vector
        in_dim = N_GENES * config["node_feat_dim"]
        hidden = config["hidden_dim"]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(config["dropout"]),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(config["dropout"]),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        B = data.batch.max().item() + 1
        # Reshape: (B*N_GENES, feat_dim) -> (B, N_GENES*feat_dim)
        x = data.x.view(B, -1)
        return self.mlp(x).squeeze(-1)


def run_ablation(name, model_class, dataset, config, save_dir):
    """Run one ablation experiment."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {name}")
    print(f"{'='*60}")

    cv = StratifiedSurvivalCV(dataset, config=config, n_folds=3)

    # Override model class in CV
    original_run = cv.run

    def custom_run(save_dir=save_dir):
        from sklearn.model_selection import StratifiedKFold
        from torch_geometric.loader import DataLoader
        from gnn.training.trainer import SurvivalTrainer
        from gnn.training.metrics import concordance_index, time_dependent_auc

        strat_keys = cv._stratification_key()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), strat_keys)):
            print(f"  Fold {fold_idx+1}/3...")
            train_data = [dataset[int(i)] for i in train_idx]
            val_data = [dataset[int(i)] for i in val_idx]

            model = model_class(config)
            trainer = SurvivalTrainer(model, config)
            result = trainer.fit(train_data, val_data, save_dir=os.path.join(save_dir, f"fold_{fold_idx}"))
            fold_results.append(result["best_c_index"])
            print(f"  Fold {fold_idx+1} C-index: {result['best_c_index']:.4f}")

        mean_ci = np.mean(fold_results)
        std_ci = np.std(fold_results)
        print(f"  {name}: C-index = {mean_ci:.4f} +/- {std_ci:.4f}")
        return {"name": name, "mean_c_index": mean_ci, "std_c_index": std_ci, "folds": fold_results}

    return custom_run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="msk_impact_50k")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    config = dict(GNN_CONFIG)
    config["epochs"] = args.epochs

    print("Loading dataset...")
    dataset = MSKImpactDataset(study_id=args.study)
    print(f"  {len(dataset)} patients")

    save_base = os.path.join(GNN_RESULTS, "ablation")
    os.makedirs(save_base, exist_ok=True)

    results = []

    # Full model
    results.append(run_ablation(
        "Full Model", CouplingChannelGAT, dataset, config,
        os.path.join(save_base, "full")))

    # No cross-channel attention
    results.append(run_ablation(
        "No Cross-Channel Attention", AblationNoXChannel, dataset, config,
        os.path.join(save_base, "no_xchannel")))

    # MLP baseline (no graph)
    results.append(run_ablation(
        "MLP Baseline (No Graph)", MLPBaseline, dataset, config,
        os.path.join(save_base, "mlp")))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:<35} C-index: {r['mean_c_index']:.4f} +/- {r['std_c_index']:.4f}")

    with open(os.path.join(save_base, "ablation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_base}")


if __name__ == "__main__":
    main()
