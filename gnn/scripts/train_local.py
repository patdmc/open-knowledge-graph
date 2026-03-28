"""
Local training script — runs on Apple Silicon (MPS) or CPU.

Equivalent to the Colab notebook but as a CLI script.
No Google Drive, no uploads — reads directly from gnn/results/colab_export/.

Usage:
    python3 -u -m gnn.scripts.train_local
    python3 -u -m gnn.scripts.train_local --skip-pretrain  # jump to fine-tuning
    python3 -u -m gnn.scripts.train_local --smoke           # quick 5-epoch test
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.models.atlas_transformer_v6 import AtlasTransformerV6
from gnn.models.essentiality_head import EssentialityHead, EssentialityLoss
from gnn.config import GNN_RESULTS

EXPORT_DIR = os.path.join(GNN_RESULTS, "colab_export")
CHECKPOINT_DIR = os.path.join(GNN_RESULTS, "checkpoints")


def get_device():
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device: MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("  Device: CPU")
        return torch.device("cpu")


def cox_ph_loss(hazard, time, event):
    """Cox partial likelihood loss (Breslow approximation)."""
    order = torch.argsort(time, descending=True)
    hazard = hazard[order]
    event = event[order].float()
    log_cumsum_h = torch.logcumsumexp(hazard, dim=0)
    return -torch.mean((hazard - log_cumsum_h) * event)


def sign_loss(model, node_features, node_mask):
    """Supervise per-mutation sign from atlas log_hr."""
    sign_logits = model._last_sign_logits
    log_hr = node_features[:, :, 0]

    has_signal = (log_hr.abs() > 0.05) & (node_mask > 0)
    if has_signal.sum() < 10:
        return torch.tensor(0.0, device=sign_logits.device)

    target_01 = (torch.sign(log_hr[has_signal]) + 1) / 2
    return nn.functional.binary_cross_entropy_with_logits(
        sign_logits[has_signal], target_01)


def gather_edge_features(gene_pair_matrix, patient_edge_feats, gene_indices, gene_masks):
    """Gather gene-pair + patient features into (B, N, N, D) tensor."""
    B, N = gene_indices.shape
    safe_idx = gene_indices.clamp(0, gene_pair_matrix.shape[0] - 1)
    idx_i = safe_idx.unsqueeze(2).expand(B, N, N)
    idx_j = safe_idx.unsqueeze(1).expand(B, N, N)
    graph_feats = gene_pair_matrix[idx_i, idx_j]
    pair_mask = (gene_masks.unsqueeze(1) * gene_masks.unsqueeze(2)).unsqueeze(-1)
    graph_feats = graph_feats * pair_mask
    return torch.cat([graph_feats, patient_edge_feats * pair_mask], dim=-1)


def apply_mutation_dropout(nf, nm, ef, ess, em, drop_min=0.1, drop_max=0.3):
    """Randomly mask mutations during pre-training."""
    B, N, _ = nf.shape
    nf, nm, ef, ess, em = nf.clone(), nm.clone(), ef.clone(), ess.clone(), em.clone()
    for b in range(B):
        real = nm[b].bool()
        n_real = real.sum().item()
        if n_real <= 1:
            continue
        drop_rate = np.random.uniform(drop_min, drop_max)
        n_drop = max(1, int(n_real * drop_rate))
        n_drop = min(n_drop, n_real - 1)
        real_indices = real.nonzero(as_tuple=True)[0]
        drop_indices = real_indices[torch.randperm(len(real_indices))[:n_drop]]
        nm[b, drop_indices] = 0.0
        nf[b, drop_indices] = 0.0
        ef[b, drop_indices, :] = 0.0
        ef[b, :, drop_indices] = 0.0
        em[b, drop_indices] = 0.0
    return nf, nm, ef, ess, em


# ─── Stage 1: DepMap Pre-training ─────────────────────────────────

def pretrain(device, smoke=False):
    """Pre-train backbone on DepMap essentiality."""
    print("\n" + "=" * 60)
    print("  STAGE 1: DepMap Pre-training")
    print("=" * 60)

    dm = torch.load(os.path.join(EXPORT_DIR, "colab_depmap_data.pt"),
                    map_location="cpu", weights_only=False)
    print(f"  Cell lines: {dm['node_features'].shape[0]}")
    print(f"  Valid essentiality targets: {int(dm['essentiality_masks'].sum())}")

    EPOCHS = 5 if smoke else 200
    LR = 1e-3
    BATCH = 32
    PATIENCE = 25
    SEED = 42

    config = {
        "hidden_dim": 64, "n_heads": 4, "n_intra_layers": 2, "dropout": 0.2,
        "node_feat_dim": dm["node_features"].shape[-1],
        "edge_feat_dim": dm["graph_edge_dim"] + dm["patient_edge_dim"],
        "n_cancer_types": dm["n_cancer_types"],
        "n_channels": dm["n_channels"],
        "n_blocks": dm["n_blocks"],
    }

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Train/val split
    N_cl = len(dm["cancer_types"])
    ct_np = dm["cancer_types"].numpy()
    ct_counts = Counter(ct_np)
    strat = np.array([ct if ct_counts[ct] >= 5 else -1 for ct in ct_np])
    train_idx, val_idx = train_test_split(
        np.arange(N_cl), test_size=0.2, random_state=SEED,
        stratify=strat if (strat == -1).sum() >= 2 else None)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    gpm = dm["gene_pair_matrix"].to(device)
    model = AtlasTransformerV6(config).to(device)
    head = EssentialityHead(config["hidden_dim"], dropout=0.2).to(device)
    loss_fn = EssentialityLoss(rank_weight=0.3)

    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Build simple batched loader
    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(
        dm["node_features"][train_idx], dm["node_masks"][train_idx],
        dm["cancer_types"][train_idx], dm["clinical"][train_idx],
        dm["gene_indices"][train_idx], dm["patient_edge_feats"][train_idx],
        dm["block_ids"][train_idx], dm["channel_ids"][train_idx],
        dm["essentiality"][train_idx], dm["essentiality_masks"][train_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    val_tensors = tuple(t[val_idx].to(device) for t in [
        dm["node_features"], dm["node_masks"], dm["cancer_types"], dm["clinical"],
        dm["gene_indices"], dm["patient_edge_feats"],
        dm["block_ids"], dm["channel_ids"],
        dm["essentiality"], dm["essentiality_masks"],
    ])

    best_val_loss = float("inf")
    best_corr = 0.0
    best_state = None
    patience_counter = 0

    print(f"\n  Training ({EPOCHS} epochs)...\n")
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train(); head.train()
        total_loss = 0; n_batches = 0

        for batch in train_loader:
            nf, nm, ct, clin, gi, pef, bi, ci, ess, em = [b.to(device) for b in batch]
            ef = gather_edge_features(gpm, pef, gi, nm)
            nf, nm, ef, ess, em = apply_mutation_dropout(nf, nm, ef, ess, em)

            optimizer.zero_grad()
            node_hidden, _ = model.encode(nf, nm, ct, clin, ef, bi, ci)
            predictions = head(node_hidden)
            loss, _ = loss_fn(predictions, ess, em)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

        scheduler.step()

        # Validate
        model.eval(); head.eval()
        with torch.no_grad():
            v_nf, v_nm, v_ct, v_clin, v_gi, v_pef, v_bi, v_ci, v_ess, v_em = val_tensors
            v_ef = gather_edge_features(gpm, v_pef, v_gi, v_nm)
            v_hidden, _ = model.encode(v_nf, v_nm, v_ct, v_clin, v_ef, v_bi, v_ci)
            v_pred = head(v_hidden)
            v_loss, v_metrics = loss_fn(v_pred, v_ess, v_em)
            valid = v_em.bool()
            if valid.sum() > 10:
                p = v_pred[valid].cpu().numpy()
                t = v_ess[valid].cpu().numpy()
                corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else 0.0
            else:
                corr = 0.0

        improved = v_loss.item() < best_val_loss
        if improved:
            best_val_loss = v_loss.item()
            best_corr = corr
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "config": config,
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "val_corr": best_corr,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or improved or patience_counter >= PATIENCE:
            marker = " *" if improved else ""
            print(f"  Epoch {epoch+1:3d}: train={total_loss/n_batches:.4f} "
                  f"val_mse={v_metrics['mse']:.4f} corr={corr:.3f} "
                  f"best={best_corr:.3f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"\n  Pre-training done: {best_state['epoch']} epochs, "
          f"corr={best_corr:.3f} [{elapsed:.0f}s]")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "pretrained_backbone.pt")
    torch.save(best_state, ckpt_path)
    print(f"  Saved: {ckpt_path}")
    return ckpt_path


# ─── Stage 2: Patient Fine-tuning (sign × magnitude + Cox) ───────

def finetune(device, pretrain_path=None, smoke=False):
    """Fine-tune on patient survival with sign × magnitude decomposition."""
    from sksurv.metrics import concordance_index_censored

    print("\n" + "=" * 60)
    print("  STAGE 2: Patient Fine-tuning (sign × magnitude + Cox)")
    print("=" * 60)

    pt = torch.load(os.path.join(EXPORT_DIR, "colab_patient_data.pt"),
                    map_location="cpu", weights_only=False)
    print(f"  Patients: {pt['node_features'].shape[0]}")
    print(f"  Node feat dim: {pt['node_features'].shape[-1]}")

    if pretrain_path and os.path.exists(pretrain_path):
        pretrain_ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        print(f"  Pre-trained: epoch {pretrain_ckpt['epoch']}, corr={pretrain_ckpt['val_corr']:.3f}")
    else:
        pretrain_ckpt = None
        print("  No pre-trained backbone — training from scratch")

    FT_EPOCHS = 5 if smoke else 200
    FT_LR = 3e-4
    FT_BATCH = 256
    FT_PATIENCE = 15
    PHASE1_EPOCHS = 0 if smoke else 5
    BACKBONE_LR_SCALE = 0.1
    N_FOLDS = 2 if smoke else 5
    SIGN_WEIGHT = 0.5
    SEED = 42

    config = {
        "hidden_dim": 64, "n_heads": 4, "n_intra_layers": 2, "dropout": 0.1,
        "node_feat_dim": pt["node_features"].shape[-1],
        "edge_feat_dim": pt["graph_edge_dim"] + pt["patient_edge_dim"],
        "n_cancer_types": pt["n_cancer_types"],
        "n_channels": pt["n_channels"],
        "n_blocks": pt["n_blocks"],
    }

    # Move to device
    nf = pt["node_features"].to(device)
    nm = pt["node_masks"].to(device)
    ct = pt["cancer_types"].to(device)
    clinical = pt["clinical"].to(device)
    atlas_sums = pt["atlas_sums"].to(device)
    times = pt["times"]
    events = pt["events"]
    gpm = pt["gene_pair_matrix"].to(device)
    pef = pt["patient_edge_feats"].to(device)
    gi = pt["gene_indices"].to(device)
    bi = pt["block_ids"].to(device)
    ci = pt["channel_ids"].to(device)

    # Holdback + CV
    np.random.seed(SEED)
    n_total = len(events)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    n_holdback = int(n_total * 0.15)
    holdback_idx = all_idx[:n_holdback]
    cv_idx = all_idx[n_holdback:]
    events_cv = events[cv_idx].numpy()
    print(f"  Patients: {n_total}, Holdback: {n_holdback}, CV: {len(cv_idx)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pretrained_results = []
    baseline_results = []

    def train_fold(model, train_idx, val_idx, use_pretrain=False):
        train_t = torch.tensor(train_idx, dtype=torch.long)
        val_t = torch.tensor(val_idx, dtype=torch.long)

        # Phase 1: frozen backbone
        if use_pretrain and PHASE1_EPOCHS > 0:
            for name, param in model.named_parameters():
                if any(k in name for k in ["node_encoder", "intra_block", "block_to_channel",
                                            "cross_channel", "channel_pool", "age_mod"]):
                    param.requires_grad = False
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt1 = torch.optim.Adam(trainable, lr=FT_LR, weight_decay=1e-4)

            for ep in range(PHASE1_EPOCHS):
                model.train()
                perm = np.random.permutation(len(train_idx))
                for b_start in range(0, len(perm), FT_BATCH):
                    b_abs = train_t[perm[b_start:b_start + FT_BATCH]]
                    opt1.zero_grad()
                    b_nf, b_nm = nf[b_abs], nm[b_abs]
                    b_edge = gather_edge_features(gpm, pef[b_abs], gi[b_abs], b_nm)
                    hazard = model(b_nf, b_nm, ct[b_abs], clinical[b_abs],
                                   atlas_sums[b_abs], b_edge, bi[b_abs], ci[b_abs])
                    loss_c = cox_ph_loss(hazard, times[b_abs].to(device), events[b_abs].to(device))
                    loss_s = sign_loss(model, b_nf, b_nm)
                    (loss_c + SIGN_WEIGHT * loss_s).backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt1.step()
            for param in model.parameters():
                param.requires_grad = True

        # Phase 2: full training
        backbone_params, readout_params = [], []
        for name, param in model.named_parameters():
            if any(k in name for k in ["node_encoder", "intra_block", "block_to_channel",
                                        "cross_channel", "channel_pool", "age_mod"]):
                backbone_params.append(param)
            else:
                readout_params.append(param)

        if use_pretrain:
            param_groups = [
                {"params": backbone_params, "lr": FT_LR * BACKBONE_LR_SCALE},
                {"params": readout_params, "lr": FT_LR},
            ]
        else:
            param_groups = [{"params": list(model.parameters()), "lr": FT_LR}]

        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_EPOCHS)

        best_c = 0.0
        no_improve = 0

        for epoch in range(FT_EPOCHS):
            model.train()
            perm = np.random.permutation(len(train_idx))
            epoch_cox = 0; epoch_sign = 0; n_batches = 0

            for b_start in range(0, len(perm), FT_BATCH):
                b_abs = train_t[perm[b_start:b_start + FT_BATCH]]
                optimizer.zero_grad()
                b_nf, b_nm = nf[b_abs], nm[b_abs]
                b_edge = gather_edge_features(gpm, pef[b_abs], gi[b_abs], b_nm)
                hazard = model(b_nf, b_nm, ct[b_abs], clinical[b_abs],
                               atlas_sums[b_abs], b_edge, bi[b_abs], ci[b_abs])
                loss_c = cox_ph_loss(hazard, times[b_abs].to(device), events[b_abs].to(device))
                loss_s = sign_loss(model, b_nf, b_nm)
                (loss_c + SIGN_WEIGHT * loss_s).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_cox += loss_c.item()
                epoch_sign += loss_s.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_preds = []
                    for v_start in range(0, len(val_idx), FT_BATCH):
                        v_abs = val_t[v_start:v_start + FT_BATCH]
                        v_edge = gather_edge_features(gpm, pef[v_abs], gi[v_abs], nm[v_abs])
                        h = model(nf[v_abs], nm[v_abs], ct[v_abs], clinical[v_abs],
                                  atlas_sums[v_abs], v_edge, bi[v_abs], ci[v_abs])
                        val_preds.append(h.cpu())
                    h_val = torch.cat(val_preds).numpy().flatten()

                e_val = events[val_idx].numpy().astype(bool)
                t_val = times[val_idx].numpy()
                valid = t_val > 0
                try:
                    c_val = concordance_index_censored(e_val[valid], t_val[valid], h_val[valid])[0]
                except Exception:
                    c_val = 0.5

                if c_val > best_c:
                    best_c = c_val
                    no_improve = 0
                else:
                    no_improve += 1

                if (epoch + 1) % 20 == 0:
                    print(f"      Ep {epoch+1:3d}: cox={epoch_cox/n_batches:.4f} "
                          f"sign={epoch_sign/n_batches:.4f} C={c_val:.4f} best={best_c:.4f}")

                if no_improve >= FT_PATIENCE // 5:
                    print(f"      Early stop ep {epoch+1}, C={best_c:.4f}")
                    break

        return best_c

    # Run folds
    t0 = time.time()
    for fold, (train_rel, val_rel) in enumerate(skf.split(cv_idx, events_cv)):
        train_idx = cv_idx[train_rel]
        val_idx = cv_idx[val_rel]

        # Pre-trained
        if pretrain_ckpt:
            print(f"\n=== Fold {fold} PRE-TRAINED ===")
            model_pt = AtlasTransformerV6(config).to(device)
            state = model_pt.state_dict()
            loaded = 0
            for k, v in pretrain_ckpt["backbone"].items():
                if k in state and state[k].shape == v.shape:
                    state[k] = v
                    loaded += 1
            model_pt.load_state_dict(state)
            print(f"    Loaded {loaded} params from pre-trained backbone")
            c_pt = train_fold(model_pt, train_idx, val_idx, use_pretrain=True)
            pretrained_results.append(c_pt)
            print(f"  Fold {fold} pre-trained: C={c_pt:.4f}")
            del model_pt

        # Baseline
        print(f"\n=== Fold {fold} BASELINE ===")
        model_bl = AtlasTransformerV6(config).to(device)
        c_bl = train_fold(model_bl, train_idx, val_idx, use_pretrain=False)
        baseline_results.append(c_bl)
        print(f"  Fold {fold} baseline: C={c_bl:.4f}")
        del model_bl

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Results
    bl_mean = np.mean(baseline_results)
    bl_std = np.std(baseline_results)
    print(f"\n{'=' * 60}")
    print(f"  Baseline:    {bl_mean:.4f} +/- {bl_std:.4f}  {[f'{c:.4f}' for c in baseline_results]}")
    if pretrained_results:
        pt_mean = np.mean(pretrained_results)
        pt_std = np.std(pretrained_results)
        delta = pt_mean - bl_mean
        print(f"  Pre-trained: {pt_mean:.4f} +/- {pt_std:.4f}  {[f'{c:.4f}' for c in pretrained_results]}")
        print(f"  Delta:       {delta:+.4f}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'=' * 60}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    results = {
        "baseline": {"mean": float(bl_mean), "std": float(bl_std),
                     "folds": [float(c) for c in baseline_results]},
        "time_minutes": elapsed / 60,
    }
    if pretrained_results:
        results["pretrained"] = {
            "mean": float(pt_mean), "std": float(pt_std),
            "folds": [float(c) for c in pretrained_results]}
        results["delta"] = float(delta)

    results_path = os.path.join(CHECKPOINT_DIR, "finetune_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip DepMap pre-training, go straight to fine-tuning")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (5 epochs, 2 folds)")
    args = parser.parse_args()

    print("=" * 60)
    print("  LOCAL TRAINING: sign × magnitude + Cox")
    print("=" * 60)

    device = get_device()

    ckpt_path = os.path.join(CHECKPOINT_DIR, "pretrained_backbone.pt")

    if not args.skip_pretrain:
        ckpt_path = pretrain(device, smoke=args.smoke)
    elif not os.path.exists(ckpt_path):
        print(f"  No checkpoint at {ckpt_path}, will train from scratch")
        ckpt_path = None

    finetune(device, pretrain_path=ckpt_path, smoke=args.smoke)


if __name__ == "__main__":
    main()
