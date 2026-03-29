"""
Stratified 5-fold cross-validation for survival prediction.
"""

import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import InMemoryDataset

from ..config import GNN_CONFIG, GNN_RESULTS
from ..models.channel_gat import CouplingChannelGAT
from .trainer import SurvivalTrainer
from .metrics import concordance_index, time_dependent_auc


class StratifiedSurvivalCV:
    """5-fold CV stratified by (event, channel_count_bin)."""

    def __init__(self, dataset, config=None, n_folds=5):
        self.dataset = dataset
        self.config = config or GNN_CONFIG
        self.n_folds = n_folds

    def _stratification_key(self):
        """Create composite stratification key for balanced folds."""
        events = []
        ch_bins = []
        for data in self.dataset:
            events.append(int(data.y_event.item()))
            # Bin channel count: 0, 1, 2, 3+
            n_ch = min(int(data.n_channels_severed.item()), 3)
            ch_bins.append(n_ch)

        # Composite key: event * 4 + channel_bin
        keys = [e * 4 + c for e, c in zip(events, ch_bins)]
        return np.array(keys)

    def run(self, save_dir=None):
        """Run full cross-validation.

        Returns:
            dict with per-fold and aggregate metrics
        """
        if save_dir is None:
            save_dir = os.path.join(GNN_RESULTS, "cv_results")
        os.makedirs(save_dir, exist_ok=True)

        strat_keys = self._stratification_key()
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.config["random_seed"],
        )

        fold_results = []
        all_indices = np.arange(len(self.dataset))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, strat_keys)):
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx + 1}/{self.n_folds}")
            print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
            print(f"{'='*60}")

            train_data = [self.dataset[int(i)] for i in train_idx]
            val_data = [self.dataset[int(i)] for i in val_idx]

            # Fresh model for each fold
            model = CouplingChannelGAT(self.config)
            trainer = SurvivalTrainer(model, self.config)

            fold_save = os.path.join(save_dir, f"fold_{fold_idx}")
            result = trainer.fit(train_data, val_data, save_dir=fold_save)

            # Final evaluation with full metrics
            from torch_geometric.loader import DataLoader
            val_loader = DataLoader(val_data, batch_size=self.config["batch_size"])
            final_metrics = trainer.evaluate(val_loader)

            # Time-dependent AUC
            all_h, all_t, all_e = [], [], []
            with torch.no_grad():
                model.eval()
                for batch in val_loader:
                    batch = batch.to(trainer.device)
                    h = model(batch)
                    all_h.append(h.cpu())
                    all_t.append(batch.y_time.squeeze().cpu())
                    all_e.append(batch.y_event.squeeze().cpu())

            all_h = torch.cat(all_h)
            all_t = torch.cat(all_t)
            all_e = torch.cat(all_e)
            td_auc = time_dependent_auc(all_h, all_t, all_e)

            fold_result = {
                "fold": fold_idx,
                "c_index": final_metrics["c_index"],
                "loss": final_metrics["loss"],
                "td_auc": td_auc,
                "epochs": result["epochs_trained"],
                "best_c_index": result["best_c_index"],
            }
            fold_results.append(fold_result)

            print(f"  Fold {fold_idx+1} C-index: {final_metrics['c_index']:.4f}")
            for t, auc in td_auc.items():
                print(f"  Fold {fold_idx+1} AUC@{t}mo: {auc:.4f}")

        # Aggregate
        c_indices = [r["c_index"] for r in fold_results]
        mean_ci = np.mean(c_indices)
        std_ci = np.std(c_indices)

        print(f"\n{'='*60}")
        print(f"  AGGREGATE RESULTS")
        print(f"  C-index: {mean_ci:.4f} +/- {std_ci:.4f}")
        for t in [12, 36, 60]:
            aucs = [r["td_auc"].get(t, float("nan")) for r in fold_results]
            aucs = [a for a in aucs if not np.isnan(a)]
            if aucs:
                print(f"  AUC@{t}mo: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print(f"{'='*60}")

        summary = {
            "fold_results": fold_results,
            "mean_c_index": mean_ci,
            "std_c_index": std_ci,
            "n_patients": len(self.dataset),
            "n_folds": self.n_folds,
        }

        # Save summary
        import json
        with open(os.path.join(save_dir, "cv_summary.json"), "w") as f:
            # Convert non-serializable types
            json.dump(summary, f, indent=2, default=str)

        return summary
