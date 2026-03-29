#!/usr/bin/env python3
"""
Fine-tune AtlasTransformer V5 with DepMap pre-trained backbone.

Two-phase training:
  Phase 1 (frozen): Freeze backbone, train only readout MLP + cancer_type embeddings.
  Phase 2 (unfrozen): Unfreeze all, discriminative LR (backbone at 0.1× readout LR).

Compares against baseline (no pre-training) on the same folds for fair evaluation.

Usage:
    python3 -u -m gnn.scripts.train_atlas_transformer_v5_pretrained
    python3 -u -m gnn.scripts.train_atlas_transformer_v5_pretrained --skip-baseline
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use train_atlas_transformer_v6.py instead.")
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, MAX_NODES
from gnn.data.graph_schema import get_schema
from gnn.models.atlas_transformer_v5 import AtlasTransformerV5
from gnn.models.cox_loss import CoxPartialLikelihoodLoss
from gnn.training.metrics import concordance_index, time_dependent_auc
from gnn.config import GNN_RESULTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-block-layers", type=int, default=2)
    parser.add_argument("--n-cross-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1,
                        help="Backbone LR = lr * backbone-lr-factor in phase 2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to pretrained backbone checkpoint")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline (no pre-training) comparison")
    return parser.parse_args()


# Backbone parameter names: node_encoder, cancer_node_mod, age_node_mod,
# channel_pos_embed, block_layers, block_pool_attn, block_to_channel,
# cross_layers, channel_pool_attn
BACKBONE_PREFIXES = (
    "node_encoder", "cancer_node_mod", "age_node_mod", "channel_pos_embed",
    "block_layers", "block_pool_attn", "block_to_channel",
    "cross_layers", "channel_pool_attn",
)

READOUT_PREFIXES = ("readout", "sex_encoder", "atlas_skip")


def is_backbone_param(name):
    return any(name.startswith(p) for p in BACKBONE_PREFIXES)


def train_fold(model, train_loader, val_data, config, device,
               phase1_epochs=5, phase2_epochs=100, patience=15,
               backbone_lr_factor=0.1, pretrained=False):
    """Train one fold with optional two-phase schedule."""
    cox_loss = CoxPartialLikelihoodLoss()

    (val_nf, val_nm, val_cp, val_ct, val_clin, val_atlas,
     val_ef, val_bi, val_ci, val_t, val_e) = val_data

    best_c = 0.0
    best_state = None
    patience_counter = 0
    total_epochs = 0

    # Phase 1: frozen backbone (only if pretrained)
    if pretrained and phase1_epochs > 0:
        print(f"    Phase 1: frozen backbone ({phase1_epochs} epochs)", flush=True)

        # Freeze backbone
        for name, param in model.named_parameters():
            if is_backbone_param(name):
                param.requires_grad = False

        readout_params = [p for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(readout_params, lr=config["lr"], weight_decay=1e-4)

        for epoch in range(phase1_epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                nf, nm, cp, ct, clin, atlas, ef, bi, ci, t, e = [
                    b.to(device) for b in batch
                ]
                optimizer.zero_grad()
                hazard = model(nf, nm, cp, ct, clin, atlas, ef, bi, ci).squeeze()
                loss = cox_loss(hazard, t, e)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            model.eval()
            with torch.no_grad():
                val_hazard = model(
                    val_nf.to(device), val_nm.to(device), val_cp.to(device),
                    val_ct.to(device), val_clin.to(device), val_atlas.to(device),
                    val_ef.to(device), val_bi.to(device), val_ci.to(device),
                ).squeeze().cpu().numpy()
            val_hazard = np.nan_to_num(val_hazard, nan=0.0)
            c_idx = concordance_index(val_t.numpy(), val_e.numpy(), val_hazard)

            if c_idx > best_c:
                best_c = c_idx
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"      P1 Epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.4f} "
                  f"C={c_idx:.4f}", flush=True)
            total_epochs += 1

        # Unfreeze backbone
        for param in model.parameters():
            param.requires_grad = True

    # Phase 2: full model with discriminative LR
    phase_label = "Phase 2" if pretrained else "Training"
    print(f"    {phase_label}: full model ({phase2_epochs} epochs)", flush=True)

    if pretrained:
        backbone_params = [p for n, p in model.named_parameters() if is_backbone_param(n)]
        readout_params = [p for n, p in model.named_parameters() if not is_backbone_param(n)]
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": config["lr"] * backbone_lr_factor},
            {"params": readout_params, "lr": config["lr"]},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    for epoch in range(phase2_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            nf, nm, cp, ct, clin, atlas, ef, bi, ci, t, e = [
                b.to(device) for b in batch
            ]
            optimizer.zero_grad()
            hazard = model(nf, nm, cp, ct, clin, atlas, ef, bi, ci).squeeze()
            loss = cox_loss(hazard, t, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_hazard = model(
                val_nf.to(device), val_nm.to(device), val_cp.to(device),
                val_ct.to(device), val_clin.to(device), val_atlas.to(device),
                val_ef.to(device), val_bi.to(device), val_ci.to(device),
            ).squeeze().cpu().numpy()
        val_hazard = np.nan_to_num(val_hazard, nan=0.0)
        c_idx = concordance_index(val_t.numpy(), val_e.numpy(), val_hazard)

        if c_idx > best_c:
            best_c = c_idx
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(f"      Epoch {epoch+1:3d}: loss={epoch_loss/max(n_batches,1):.4f} "
                  f"C={c_idx:.4f} best={best_c:.4f}", flush=True)

        total_epochs += 1
        if patience_counter >= patience:
            print(f"      Early stop epoch {epoch+1}, best C={best_c:.4f}", flush=True)
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_c, best_state, total_epochs


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Find pretrained checkpoint
    if args.pretrained_path is None:
        default_path = os.path.join(GNN_RESULTS, "depmap_pretrain", "pretrained_backbone.pt")
        if os.path.exists(default_path):
            args.pretrained_path = default_path
        else:
            print(f"ERROR: No pretrained checkpoint found at {default_path}")
            print("Run pretrain_depmap.py first.")
            sys.exit(1)

    print(f"\nPretrained checkpoint: {args.pretrained_path}", flush=True)
    checkpoint = torch.load(args.pretrained_path, map_location="cpu", weights_only=False)
    pretrained_config = checkpoint["config"]
    print(f"  Pre-training val corr: {checkpoint.get('val_corr', 'N/A')}")
    print(f"  Pre-training epoch: {checkpoint.get('epoch', 'N/A')}")

    # Schema
    print("\nDiscovering graph schema...", flush=True)
    schema = get_schema()

    # Patient dataset
    ds = AtlasDataset()
    data = ds.build_v5_features(schema=schema)

    node_features = data["node_features"]
    node_masks = data["node_masks"]
    channel_pos_ids = data["channel_pos_ids"]
    atlas_sums = data["atlas_sums"]
    times = data["times"]
    events = data["events"]
    cancer_types = data["cancer_types"]
    ages = data["ages"]
    sexes = data["sexes"]
    edge_features = data["edge_features"]
    block_ids = data["block_ids"]
    channel_ids = data["channel_ids"]
    n_cancer_types = data["n_cancer_types"]

    clinical = torch.stack([ages, sexes], dim=-1)
    N = len(events)

    print(f"\nPatient dataset: {N} patients, {n_cancer_types} cancer types")

    # Model config: use pretrained config for backbone, override n_cancer_types
    config = dict(pretrained_config)
    config["n_cancer_types"] = n_cancer_types
    config.update({
        "lr": args.lr,
        "weight_decay": 1e-4,
        "epochs": args.phase1_epochs + args.phase2_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    })

    tag = "atlas_transformer_v5_pretrained"
    results_dir = os.path.join(GNN_RESULTS, tag)
    os.makedirs(results_dir, exist_ok=True)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    pretrained_results = []
    baseline_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), events.numpy())):
        print(f"\n{'='*60}", flush=True)
        print(f"  FOLD {fold}", flush=True)
        print(f"{'='*60}", flush=True)

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)

        train_ds = TensorDataset(
            node_features[train_idx], node_masks[train_idx],
            channel_pos_ids[train_idx], cancer_types[train_idx],
            clinical[train_idx], atlas_sums[train_idx],
            edge_features[train_idx], block_ids[train_idx],
            channel_ids[train_idx],
            times[train_idx], events[train_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)

        val_data = (
            node_features[val_idx], node_masks[val_idx],
            channel_pos_ids[val_idx], cancer_types[val_idx],
            clinical[val_idx], atlas_sums[val_idx],
            edge_features[val_idx], block_ids[val_idx],
            channel_ids[val_idx],
            times[val_idx], events[val_idx],
        )

        # --- Pretrained model ---
        print(f"\n  [PRETRAINED] Train={len(train_idx)}, Val={len(val_idx)}", flush=True)
        model_pt = AtlasTransformerV5(config).to(device)

        # Load pretrained backbone weights (skip mismatched keys like cancer_type embeddings)
        backbone_state = checkpoint["backbone"]
        model_state = model_pt.state_dict()

        loaded = 0
        skipped = 0
        for k, v in backbone_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model_pt.load_state_dict(model_state)

        if fold == 0:
            print(f"  Loaded {loaded} pretrained params, skipped {skipped}", flush=True)
            n_params = sum(p.numel() for p in model_pt.parameters())
            print(f"  Total params: {n_params:,}", flush=True)

        best_c, best_state, n_epochs = train_fold(
            model_pt, train_loader, val_data, config, device,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            patience=args.patience,
            backbone_lr_factor=args.backbone_lr_factor,
            pretrained=True,
        )

        # TD-AUC
        model_pt.load_state_dict(best_state)
        model_pt.to(device).eval()
        with torch.no_grad():
            val_hazard = model_pt(
                node_features[val_idx].to(device), node_masks[val_idx].to(device),
                channel_pos_ids[val_idx].to(device), cancer_types[val_idx].to(device),
                clinical[val_idx].to(device), atlas_sums[val_idx].to(device),
                edge_features[val_idx].to(device), block_ids[val_idx].to(device),
                channel_ids[val_idx].to(device),
            ).squeeze().cpu().numpy()
        td_auc = time_dependent_auc(val_hazard, times[val_idx].numpy(),
                                    events[val_idx].numpy(), [12, 36, 60])

        pretrained_results.append({
            "fold": fold, "c_index": best_c, "n_epochs": n_epochs, "td_auc": td_auc,
        })
        print(f"  [PRETRAINED] Fold {fold}: C={best_c:.4f}", flush=True)

        # Save fold model
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))

        # --- Baseline (no pre-training) ---
        if not args.skip_baseline:
            print(f"\n  [BASELINE] Train={len(train_idx)}, Val={len(val_idx)}", flush=True)
            model_bl = AtlasTransformerV5(config).to(device)

            best_c_bl, best_state_bl, n_epochs_bl = train_fold(
                model_bl, train_loader, val_data, config, device,
                phase1_epochs=0,
                phase2_epochs=args.phase2_epochs,
                patience=args.patience,
                pretrained=False,
            )

            baseline_results.append({
                "fold": fold, "c_index": best_c_bl, "n_epochs": n_epochs_bl,
            })
            print(f"  [BASELINE] Fold {fold}: C={best_c_bl:.4f}", flush=True)

    # Summary
    pt_cs = [r["c_index"] for r in pretrained_results]
    pt_mean = float(np.mean(pt_cs))
    pt_std = float(np.std(pt_cs))

    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  PRETRAINED:  C = {pt_mean:.4f} ± {pt_std:.4f}", flush=True)
    for r in pretrained_results:
        print(f"    Fold {r['fold']}: C={r['c_index']:.4f} ({r['n_epochs']} epochs)")

    if baseline_results:
        bl_cs = [r["c_index"] for r in baseline_results]
        bl_mean = float(np.mean(bl_cs))
        bl_std = float(np.std(bl_cs))
        print(f"\n  BASELINE:    C = {bl_mean:.4f} ± {bl_std:.4f}", flush=True)
        for r in baseline_results:
            print(f"    Fold {r['fold']}: C={r['c_index']:.4f} ({r['n_epochs']} epochs)")

        delta = pt_mean - bl_mean
        print(f"\n  DELTA: {delta:+.4f} ({'improved' if delta > 0 else 'no improvement'})")

    # Save results
    results = {
        "model": "AtlasTransformerV5 (DepMap pretrained)",
        "pretrained_checkpoint": args.pretrained_path,
        "pretrained_val_corr": checkpoint.get("val_corr"),
        "pretrained_results": pretrained_results,
        "pretrained_mean_c": pt_mean,
        "pretrained_std_c": pt_std,
        "baseline_results": baseline_results if baseline_results else None,
        "baseline_mean_c": float(np.mean([r["c_index"] for r in baseline_results])) if baseline_results else None,
        "n_patients": N,
        "n_cancer_types": n_cancer_types,
        "config": {k: v for k, v in config.items()
                   if not isinstance(v, (np.ndarray, torch.Tensor))},
        "args": vars(args),
    }
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
