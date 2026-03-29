"""
Training loop for the Coupling-Channel GNN.
"""

import os
import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from ..models.cox_loss import CoxPartialLikelihoodLoss
from .metrics import concordance_index


class SurvivalTrainer:
    """Train and evaluate coupling-channel GNN for survival prediction."""

    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=10, factor=0.5
        )
        self.loss_fn = CoxPartialLikelihoodLoss()

    def train_epoch(self, loader):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            hazard = self.model(batch)
            time = batch.y_time.squeeze()
            event = batch.y_event.squeeze()

            loss = self.loss_fn(hazard, time, event)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate model on a data loader.

        Returns:
            dict with 'loss', 'c_index'
        """
        self.model.eval()
        all_hazard, all_time, all_event = [], [], []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            hazard = self.model(batch)
            time = batch.y_time.squeeze()
            event = batch.y_event.squeeze()

            loss = self.loss_fn(hazard, time, event)
            total_loss += loss.item()
            n_batches += 1

            all_hazard.append(hazard.cpu())
            all_time.append(time.cpu())
            all_event.append(event.cpu())

        all_hazard = torch.cat(all_hazard)
        all_time = torch.cat(all_time)
        all_event = torch.cat(all_event)

        c_idx = concordance_index(all_hazard, all_time, all_event)

        return {
            "loss": total_loss / max(n_batches, 1),
            "c_index": c_idx,
        }

    def fit(self, train_dataset, val_dataset, save_dir=None):
        """Full training loop with early stopping.

        Args:
            train_dataset: PyG dataset for training
            val_dataset: PyG dataset for validation
            save_dir: optional directory to save best model

        Returns:
            dict with training history and best metrics
        """
        batch_size = self.config["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_c_index = 0.0
        best_model_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_c_index": []}

        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_c_index"].append(val_metrics["c_index"])

            self.scheduler.step(val_metrics["c_index"])

            if val_metrics["c_index"] > best_c_index:
                best_c_index = val_metrics["c_index"]
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val C-index: {val_metrics['c_index']:.4f} | "
                      f"LR: {lr:.2e}")

            if patience_counter >= self.config["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_model_state, os.path.join(save_dir, "best_model.pt"))

        return {
            "best_c_index": best_c_index,
            "history": history,
            "epochs_trained": len(history["train_loss"]),
        }
