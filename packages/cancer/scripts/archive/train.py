#!/usr/bin/env python3
"""
Train the Coupling-Channel GNN on MSK-IMPACT data.

Usage:
    python3 -m gnn.scripts.train [--study msk_impact_50k] [--folds 5]
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import argparse
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_CONFIG, GNN_RESULTS
from gnn.data.msk_dataset import MSKImpactDataset
from gnn.training.cross_val import StratifiedSurvivalCV


def main():
    parser = argparse.ArgumentParser(description="Train Coupling-Channel GNN")
    parser.add_argument("--study", default="msk_impact_50k",
                        help="MSK dataset ID (default: msk_impact_50k)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    config = dict(GNN_CONFIG)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr

    print("=" * 60)
    print("  COUPLING-CHANNEL GNN TRAINING")
    print(f"  Study: {args.study}")
    print(f"  Folds: {args.folds}")
    print(f"  Config: hidden={config['hidden_dim']}, "
          f"heads={config['num_gat_heads']}, "
          f"lr={config['lr']}, "
          f"batch={config['batch_size']}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    t0 = time.time()
    dataset = MSKImpactDataset(study_id=args.study)
    print(f"  Loaded {len(dataset)} patient graphs in {time.time()-t0:.1f}s")
    print(f"  Node features: {dataset[0].x.shape}")
    print(f"  Edges: {dataset[0].edge_index.shape[1]}")

    # Run CV
    save_dir = os.path.join(GNN_RESULTS, f"cv_{args.study}")
    cv = StratifiedSurvivalCV(dataset, config=config, n_folds=args.folds)
    results = cv.run(save_dir=save_dir)

    print(f"\nResults saved to: {save_dir}")
    print(f"Final C-index: {results['mean_c_index']:.4f} +/- {results['std_c_index']:.4f}")


if __name__ == "__main__":
    main()
