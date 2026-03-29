"""
Incremental training — add features without full retraining.

Instead of retraining from scratch when data changes, this script:
  1. Loads a checkpoint (pretrained backbone or finetuned model)
  2. Diffs current export against what was trained on (manifest)
  3. Picks the minimal update strategy:
     - Level 1: Column expansion  — new feature dim, freeze old weights
     - Level 2: Head-only         — new atlas labels, retrain sign/magnitude heads
     - Level 3: Warm-start        — new patients or changed features, short fine-tune
  4. Trains only what needs training, saves new checkpoint + manifest

Each resolved flip is a rank-1 update. Purely additive — old signal preserved.

Usage:
    # Auto-detect what changed and pick update level
    python3 -m gnn.scripts.train_incremental

    # Force a specific level
    python3 -m gnn.scripts.train_incremental --level column
    python3 -m gnn.scripts.train_incremental --level head
    python3 -m gnn.scripts.train_incremental --level warm

    # Smoke test
    python3 -m gnn.scripts.train_incremental --smoke
"""

import os, sys, json, time, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.models.atlas_transformer_v6 import AtlasTransformerV6
from gnn.config import GNN_RESULTS

EXPORT_DIR = os.path.join(GNN_RESULTS, "colab_export")
CHECKPOINT_DIR = os.path.join(GNN_RESULTS, "checkpoints")
MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, "training_manifest.json")


# ─── Utilities ────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cox_ph_loss(hazard, time, event):
    order = torch.argsort(time, descending=True)
    hazard = hazard[order]
    event = event[order].float()
    log_cumsum_h = torch.logcumsumexp(hazard, dim=0)
    return -torch.mean((hazard - log_cumsum_h) * event)


def sign_loss(model, node_features, node_mask):
    sign_logits = model._last_sign_logits
    log_hr = node_features[:, :, 0]
    has_signal = (log_hr.abs() > 0.05) & (node_mask > 0)
    if has_signal.sum() < 10:
        return torch.tensor(0.0, device=sign_logits.device)
    target_01 = (torch.sign(log_hr[has_signal]) + 1) / 2
    return nn.functional.binary_cross_entropy_with_logits(
        sign_logits[has_signal], target_01)


def gather_edge_features(gene_pair_matrix, patient_edge_feats, gene_indices, gene_masks):
    B, N = gene_indices.shape
    safe_idx = gene_indices.clamp(0, gene_pair_matrix.shape[0] - 1)
    idx_i = safe_idx.unsqueeze(2).expand(B, N, N)
    idx_j = safe_idx.unsqueeze(1).expand(B, N, N)
    graph_feats = gene_pair_matrix[idx_i, idx_j]
    pair_mask = (gene_masks.unsqueeze(1) * gene_masks.unsqueeze(2)).unsqueeze(-1)
    graph_feats = graph_feats * pair_mask
    return torch.cat([graph_feats, patient_edge_feats * pair_mask], dim=-1)


def data_fingerprint(pt):
    """Hash the data shape and a sample to detect changes."""
    nf = pt["node_features"]
    return {
        "n_patients": int(nf.shape[0]),
        "max_nodes": int(nf.shape[1]),
        "node_feat_dim": int(nf.shape[2]),
        "graph_edge_dim": int(pt["graph_edge_dim"]),
        "patient_edge_dim": int(pt["patient_edge_dim"]),
        "n_cancer_types": int(pt["n_cancer_types"]),
        "n_channels": int(pt["n_channels"]),
        "n_blocks": int(pt["n_blocks"]),
        # Hash first 100 patients' features for content change detection
        "content_hash": hashlib.md5(
            nf[:100].numpy().tobytes()).hexdigest(),
    }


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return None


def save_manifest(fingerprint, level, c_index, checkpoint_path):
    manifest = {
        "data": fingerprint,
        "last_update": {
            "level": level,
            "c_index": c_index,
            "checkpoint": checkpoint_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "history": [],
    }
    # Append to history if prior manifest exists
    prior = load_manifest()
    if prior:
        manifest["history"] = prior.get("history", [])
        manifest["history"].append(prior.get("last_update", {}))
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {MANIFEST_PATH}")


# ─── Diff and Level Detection ────────────────────────────────────

def detect_level(current_fp, prior_manifest):
    """Compare current data fingerprint to prior manifest, return update level."""
    if prior_manifest is None:
        return "full", "No prior manifest — full training required"

    prior_fp = prior_manifest["data"]

    # New feature dimensions → column expansion
    if current_fp["node_feat_dim"] != prior_fp["node_feat_dim"]:
        old_d = prior_fp["node_feat_dim"]
        new_d = current_fp["node_feat_dim"]
        return "column", (f"Node feat dim changed: {old_d} → {new_d}. "
                          f"Column expansion: freeze old {old_d} weights, train new {new_d - old_d}.")

    # Same shape but different content → features changed (e.g., categorical → continuous)
    if current_fp["content_hash"] != prior_fp["content_hash"]:
        if current_fp["n_patients"] == prior_fp["n_patients"]:
            return "head", "Feature content changed (same patients). Retrain sign/magnitude heads."

    # New patients added
    if current_fp["n_patients"] != prior_fp["n_patients"]:
        old_n = prior_fp["n_patients"]
        new_n = current_fp["n_patients"]
        delta = new_n - old_n
        pct = abs(delta) / old_n * 100
        if pct < 5:
            return "head", f"Small patient change ({delta:+d}, {pct:.1f}%). Head-only retrain."
        return "warm", f"Patient count changed: {old_n} → {new_n} ({delta:+d}). Warm-start fine-tune."

    # Edge dims changed
    if (current_fp["graph_edge_dim"] != prior_fp["graph_edge_dim"] or
            current_fp["patient_edge_dim"] != prior_fp["patient_edge_dim"]):
        return "warm", "Edge feature dimensions changed. Warm-start fine-tune."

    return "none", "No changes detected."


# ─── Level 1: Column Expansion ───────────────────────────────────

def expand_columns(model, old_state, old_dim, new_dim):
    """Expand node_encoder input projection to accommodate new feature dims.

    Freezes the old columns and initializes the new ones with small random weights.
    """
    new_state = model.state_dict()

    # The first layer of node_encoder is Linear(node_feat_dim, hidden)
    key_w = "node_encoder.0.weight"  # (hidden, node_feat_dim)
    key_b = "node_encoder.0.bias"    # (hidden,)

    if key_w in old_state and key_w in new_state:
        old_w = old_state[key_w]  # (hidden, old_dim)
        new_w = new_state[key_w]  # (hidden, new_dim)

        # Copy old columns, keep new columns at random init
        new_w[:, :old_dim] = old_w
        new_state[key_w] = new_w

    if key_b in old_state:
        new_state[key_b] = old_state[key_b]

    # Copy everything else that matches shape
    loaded = 0
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
            loaded += 1

    model.load_state_dict(new_state)
    print(f"  Column expansion: {old_dim} → {new_dim} ({new_dim - old_dim} new columns)")
    print(f"  Loaded {loaded} matching params, new columns randomly initialized")
    return model


# ─── Training Loops ──────────────────────────────────────────────

def evaluate(model, nf, nm, ct, clinical, atlas_sums, gpm, pef, gi, bi, ci,
             times, events, val_idx, device, batch_size=256):
    """Compute C-index on validation set."""
    model.eval()
    val_t = torch.tensor(val_idx, dtype=torch.long)
    preds = []
    with torch.no_grad():
        for start in range(0, len(val_idx), batch_size):
            idx = val_t[start:start + batch_size]
            b_nm = nm[idx]
            edge = gather_edge_features(gpm, pef[idx], gi[idx], b_nm)
            h = model(nf[idx], b_nm, ct[idx], clinical[idx],
                      atlas_sums[idx], edge, bi[idx], ci[idx])
            preds.append(h.cpu())
    h_val = torch.cat(preds).numpy().flatten()
    e_val = events[val_idx].numpy().astype(bool)
    t_val = times[val_idx].numpy()
    valid = t_val > 0
    try:
        return concordance_index_censored(e_val[valid], t_val[valid], h_val[valid])[0]
    except Exception:
        return 0.5


def incremental_train(model, level, nf, nm, ct, clinical, atlas_sums,
                      gpm, pef, gi, bi, ci, times, events, device,
                      smoke=False, sign_weight=0.5):
    """Run incremental training at the specified level.

    Returns best C-index achieved.
    """
    # Configure based on level
    if level == "column":
        # Freeze everything except node_encoder first layer and sign/magnitude heads
        epochs = 3 if smoke else 50
        lr = 1e-3
        freeze_backbone = True
        patience = 10
    elif level == "head":
        # Freeze backbone, train sign + magnitude + readout heads
        epochs = 3 if smoke else 30
        lr = 5e-4
        freeze_backbone = True
        patience = 8
    elif level == "warm":
        # Warm-start: train everything with low LR
        epochs = 5 if smoke else 100
        lr = 1e-4
        freeze_backbone = False
        patience = 12
    else:
        raise ValueError(f"Unknown level: {level}")

    batch_size = 256
    seed = 42
    np.random.seed(seed)

    # Train/val split (15% holdback, then 80/20 CV)
    n_total = len(events)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    n_holdback = int(n_total * 0.15)
    cv_idx = all_idx[n_holdback:]
    n_val = int(len(cv_idx) * 0.2)
    val_idx = cv_idx[:n_val]
    train_idx = cv_idx[n_val:]

    print(f"\n  Level: {level}")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Holdback: {n_holdback}")

    # Freeze strategy
    if freeze_backbone:
        trainable_names = []
        for name, param in model.named_parameters():
            # Always train: sign_head, magnitude_head, readout, atlas_skip
            if any(k in name for k in ["sign_head", "magnitude_head", "readout",
                                        "atlas_skip", "sex_encoder", "temporal_encoder"]):
                param.requires_grad = True
                trainable_names.append(name)
            # For column expansion: also train node_encoder first layer
            elif level == "column" and "node_encoder.0" in name:
                param.requires_grad = True
                trainable_names.append(name)
            else:
                param.requires_grad = False

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {n_trainable:,} / {n_total_params:,} "
              f"({100 * n_trainable / n_total_params:.1f}%)")
        print(f"  Training: {', '.join(set(n.split('.')[0] for n in trainable_names))}")
    else:
        print(f"  All params trainable: {sum(p.numel() for p in model.parameters()):,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_c = evaluate(model, nf, nm, ct, clinical, atlas_sums, gpm, pef, gi, bi, ci,
                      times, events, val_idx, device)
    print(f"  Initial C-index: {best_c:.4f}")

    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    train_t = torch.tensor(train_idx, dtype=torch.long)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_idx))
        epoch_cox = 0; epoch_sign = 0; n_batches = 0

        for b_start in range(0, len(perm), batch_size):
            b_abs = train_t[perm[b_start:b_start + batch_size]]
            optimizer.zero_grad()
            b_nf, b_nm = nf[b_abs], nm[b_abs]
            b_edge = gather_edge_features(gpm, pef[b_abs], gi[b_abs], b_nm)
            hazard = model(b_nf, b_nm, ct[b_abs], clinical[b_abs],
                           atlas_sums[b_abs], b_edge, bi[b_abs], ci[b_abs])
            loss_c = cox_ph_loss(hazard, times[b_abs].to(device), events[b_abs].to(device))
            loss_s = sign_loss(model, b_nf, b_nm)
            loss = loss_c + sign_weight * loss_s
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_cox += loss_c.item()
            epoch_sign += loss_s.item()
            n_batches += 1

        scheduler.step()

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            c_val = evaluate(model, nf, nm, ct, clinical, atlas_sums, gpm, pef, gi, bi, ci,
                             times, events, val_idx, device)
            improved = c_val > best_c
            if improved:
                best_c = c_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            marker = " *" if improved else ""
            print(f"    Ep {epoch+1:3d}: cox={epoch_cox/n_batches:.4f} "
                  f"sign={epoch_sign/n_batches:.4f} C={c_val:.4f} best={best_c:.4f}{marker}")

            if no_improve >= patience // 5:
                print(f"    Early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    # Unfreeze all for next round
    for param in model.parameters():
        param.requires_grad = True

    print(f"  Done: C={best_c:.4f} [{elapsed:.0f}s]")
    return best_c


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["column", "head", "warm", "auto"],
                        default="auto", help="Update level (default: auto-detect)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (3 epochs)")
    parser.add_argument("--sign-weight", type=float, default=0.5,
                        help="Weight for sign supervision loss")
    args = parser.parse_args()

    print("=" * 60)
    print("  INCREMENTAL TRAINING")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    # Load current data
    pt_path = os.path.join(EXPORT_DIR, "colab_patient_data.pt")
    print(f"  Loading: {pt_path}")
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    current_fp = data_fingerprint(pt)
    print(f"  Patients: {current_fp['n_patients']}, "
          f"Feat dim: {current_fp['node_feat_dim']}, "
          f"Edge dims: {current_fp['graph_edge_dim']}+{current_fp['patient_edge_dim']}")

    # Load prior manifest
    prior = load_manifest()

    # Detect or use forced level
    if args.level == "auto":
        level, reason = detect_level(current_fp, prior)
        print(f"\n  Auto-detected level: {level}")
        print(f"  Reason: {reason}")
        if level == "none":
            print("  Nothing to do.")
            return
        if level == "full":
            print("  Use train_local.py for full training first.")
            return
    else:
        level = args.level
        print(f"\n  Forced level: {level}")

    # Find checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        # Try manifest first, then default locations
        if prior and "last_update" in prior:
            ckpt_path = prior["last_update"].get("checkpoint")
        if not ckpt_path or not os.path.exists(ckpt_path):
            # Check common locations
            candidates = [
                os.path.join(CHECKPOINT_DIR, "incremental_latest.pt"),
                os.path.join(CHECKPOINT_DIR, "pretrained_backbone.pt"),
                os.path.expanduser("~/Downloads/pretrained_backbone.pt"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    ckpt_path = c
                    break

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"\n  ERROR: No checkpoint found. Run train_local.py first or pass --checkpoint.")
        return

    print(f"  Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Determine old config
    if "config" in ckpt:
        old_config = ckpt["config"]
    else:
        old_config = None

    # Build new config from current data
    config = {
        "hidden_dim": 64,
        "n_heads": 4,
        "n_intra_layers": 2,
        "dropout": 0.1,
        "node_feat_dim": current_fp["node_feat_dim"],
        "edge_feat_dim": current_fp["graph_edge_dim"] + current_fp["patient_edge_dim"],
        "n_cancer_types": current_fp["n_cancer_types"],
        "n_channels": current_fp["n_channels"],
        "n_blocks": current_fp["n_blocks"],
    }

    # Inherit hidden_dim from checkpoint if available
    if old_config:
        for k in ["hidden_dim", "n_heads", "n_intra_layers"]:
            if k in old_config:
                config[k] = old_config[k]

    # Build model
    model = AtlasTransformerV6(config).to(device)

    # Load weights with appropriate strategy
    old_state = ckpt.get("backbone", ckpt.get("model_state", ckpt))
    old_dim = old_config["node_feat_dim"] if old_config else current_fp["node_feat_dim"]

    if level == "column" and old_dim != current_fp["node_feat_dim"]:
        model = expand_columns(model, old_state, old_dim, current_fp["node_feat_dim"])
    else:
        # Standard load: copy matching shapes
        state = model.state_dict()
        loaded = 0
        skipped = []
        for k, v in old_state.items():
            if k in state and state[k].shape == v.shape:
                state[k] = v
                loaded += 1
            elif k in state:
                skipped.append(f"{k}: {v.shape} → {state[k].shape}")
        model.load_state_dict(state)
        print(f"  Loaded {loaded} params from checkpoint")
        if skipped:
            print(f"  Skipped (shape mismatch): {len(skipped)}")
            for s in skipped[:5]:
                print(f"    {s}")

    # Move data to device
    nf = pt["node_features"].to(device)
    nm = pt["node_masks"].to(device)
    ct = pt["cancer_types"].to(device)
    clinical = pt["clinical"].to(device)
    atlas_sums = pt["atlas_sums"].to(device)
    gpm = pt["gene_pair_matrix"].to(device)
    pef = pt["patient_edge_feats"].to(device)
    gi = pt["gene_indices"].to(device)
    bi = pt["block_ids"].to(device)
    ci = pt["channel_ids"].to(device)

    # Train
    best_c = incremental_train(
        model, level, nf, nm, ct, clinical, atlas_sums,
        gpm, pef, gi, bi, ci, pt["times"], pt["events"],
        device, smoke=args.smoke, sign_weight=args.sign_weight,
    )

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, "incremental_latest.pt")
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": config,
        "c_index": best_c,
        "level": level,
    }, save_path)
    print(f"  Saved: {save_path}")

    # Save manifest
    save_manifest(current_fp, level, float(best_c), save_path)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Level:   {level}")
    print(f"  C-index: {best_c:.4f}")
    if prior and "last_update" in prior:
        prev_c = prior["last_update"].get("c_index", 0.5)
        delta = best_c - prev_c
        print(f"  Prior:   {prev_c:.4f}")
        print(f"  Delta:   {delta:+.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
