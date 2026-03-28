#!/usr/bin/env python3
"""
Fast training config: smaller model, larger batch, fewer epochs.
~1 hour for 3-fold CV on CPU.

Usage:
    python3 -u -m gnn.scripts.train_fast
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.msk_dataset import MSKImpactDataset
from gnn.training.cross_val import StratifiedSurvivalCV

FAST_CONFIG = {
    "node_feat_dim": 18,
    "hidden_dim": 32,
    "num_gat_heads": 4,
    "num_channel_layers": 1,
    "cross_channel_heads": 4,
    "cross_channel_layers": 1,
    "readout_hidden": 16,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "patience": 10,
    "batch_size": 256,
    "n_folds": 3,
    "random_seed": 42,
}


def main():
    print("=" * 60)
    print("  COUPLING-CHANNEL GNN — FAST TRAINING")
    print(f"  hidden=32, layers=1, batch=256, epochs=30, folds=3")
    print(f"  Estimated time: ~1 hour on CPU")
    print("=" * 60)

    print("\nLoading dataset...")
    t0 = time.time()
    dataset = MSKImpactDataset(study_id="msk_impact_50k")
    print(f"  {len(dataset)} patients in {time.time()-t0:.1f}s")

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "cv_fast"
    )
    cv = StratifiedSurvivalCV(dataset, config=FAST_CONFIG, n_folds=3)
    results = cv.run(save_dir=save_dir)

    print(f"\nFinal: C-index = {results['mean_c_index']:.4f} +/- {results['std_c_index']:.4f}")
    print(f"Results: {save_dir}")


if __name__ == "__main__":
    main()
