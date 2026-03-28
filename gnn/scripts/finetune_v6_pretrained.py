"""
Fine-tune AtlasTransformerV6 with DepMap pre-trained backbone.

Compares pre-trained vs randomly initialized backbone on patient survival
using the same 5-fold CV protocol as train_atlas_transformer_v6.py.

Phase 1 (frozen backbone): Train only readout MLP + cancer_type embeddings
Phase 2 (unfrozen): All parameters, backbone at 0.1× readout LR

Usage:
    python3 -u -m gnn.scripts.finetune_v6_pretrained
    python3 -u -m gnn.scripts.finetune_v6_pretrained --phase1-epochs 5 --epochs 100
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, NODE_FEAT_DIM, MAX_NODES
from gnn.models.atlas_transformer_v6 import AtlasTransformerV6
from gnn.models.cox_sage import cox_ph_loss
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, GNN_CACHE, GNN_RESULTS
from gnn.data.block_assignments import load_block_assignments
from gnn.data.temporal import TemporalEstimator, year_features

RESULTS_DIR = os.path.join(GNN_RESULTS, "v6_pretrained_finetune")
PRETRAIN_DIR = os.path.join(GNN_RESULTS, "depmap_pretrain")


# =========================================================================
# Edge pipeline (same as train_atlas_transformer_v6.py)
# =========================================================================

def load_graph_data():
    from neo4j import GraphDatabase
    import networkx as nx
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    print("  Loading graph data...", flush=True)

    ppi_cache = os.path.join(GNN_CACHE, "string_ppi_edges_503.json")
    if os.path.exists(ppi_cache):
        with open(ppi_cache) as f:
            ppi_raw = json.load(f)
        G_ppi = nx.Graph()
        for e in ppi_raw:
            if isinstance(e, list):
                G_ppi.add_edge(e[0], e[1], weight=e[2] if len(e) > 2 else 0.4)
            else:
                G_ppi.add_edge(e['gene1'], e['gene2'], weight=e.get('score', 0.4))
    else:
        G_ppi = nx.Graph()
        with driver.session() as s:
            for r in s.run("MATCH (g1:Gene)-[r:PPI]-(g2:Gene) WHERE g1.name < g2.name "
                           "RETURN g1.name AS g1, g2.name AS g2, r.score AS score"):
                G_ppi.add_edge(r["g1"], r["g2"], weight=r["score"] or 0.4)

    ppi_dists = {}
    for comp in nx.connected_components(G_ppi):
        sub = G_ppi.subgraph(comp)
        for src, targets in dict(nx.all_pairs_shortest_path_length(sub)).items():
            for tgt, dist in targets.items():
                if src <= tgt:
                    ppi_dists[(src, tgt)] = dist

    channel_profiles = {}
    with driver.session() as s:
        for r in s.run("MATCH (g:Gene) WHERE g.channel_profile IS NOT NULL "
                       "RETURN g.name AS gene, g.channel_profile AS profile"):
            channel_profiles[r["gene"]] = np.array(r["profile"], dtype=np.float32)

    cooccurrence = {}
    with driver.session() as s:
        for r in s.run("MATCH (g1:Gene)-[r:COOCCURS]-(g2:Gene) WHERE g1.name < g2.name "
                       "RETURN g1.name AS g1, g2.name AS g2, r.count AS cnt"):
            cooccurrence[(r["g1"], r["g2"])] = r["cnt"] or 0
    max_cooccur = max(cooccurrence.values()) if cooccurrence else 1

    sl_pairs = set()
    with driver.session() as s:
        for r in s.run("MATCH (g1:Gene)-[:SL_PARTNER]-(g2:Gene) WHERE g1.name < g2.name "
                       "RETURN g1.name AS g1, g2.name AS g2"):
            sl_pairs.add((r["g1"], r["g2"]))

    driver.close()
    hub_set = set()
    for hubs in HUB_GENES.values():
        hub_set |= hubs

    return {
        'ppi_dists': ppi_dists, 'channel_profiles': channel_profiles,
        'cooccurrence': cooccurrence, 'max_cooccur': max_cooccur,
        'sl_pairs': sl_pairs, 'hub_set': hub_set,
    }


def compute_gene_pair_matrix(graph_data, gene_vocab):
    G = len(gene_vocab)
    GRAPH_EDGE_DIM = 10
    matrix = np.zeros((G, G, GRAPH_EDGE_DIM), dtype=np.float32)
    ppi_dists = graph_data['ppi_dists']
    profiles = graph_data['channel_profiles']
    cooccurrence = graph_data['cooccurrence']
    max_cooccur = graph_data['max_cooccur']
    sl_pairs = graph_data['sl_pairs']
    hub_set = graph_data['hub_set']
    DISCONNECTED = 10
    idx_to_gene = {i: g for g, i in gene_vocab.items()}

    for i in range(G):
        gi = idx_to_gene[i]
        matrix[i, i, 9] = 1.0
        for j in range(i + 1, G):
            gj = idx_to_gene[j]
            pair = (min(gi, gj), max(gi, gj))
            d = ppi_dists.get(pair, DISCONNECTED)
            matrix[i, j, 0] = matrix[j, i, 0] = d / DISCONNECTED
            matrix[i, j, 1] = matrix[j, i, 1] = 1.0 if d == 1 else 0.0
            matrix[i, j, 2] = matrix[j, i, 2] = 1.0 if d <= 3 else 0.0
            ch_i, ch_j = CHANNEL_MAP.get(gi), CHANNEL_MAP.get(gj)
            if ch_i and ch_j:
                matrix[i, j, 3] = matrix[j, i, 3] = 1.0 if ch_i == ch_j else 0.0
                matrix[i, j, 4] = matrix[j, i, 4] = 1.0 if ch_i != ch_j else 0.0
            cooccur = cooccurrence.get(pair, 0)
            if cooccur > 0:
                matrix[i, j, 5] = matrix[j, i, 5] = np.log1p(cooccur) / np.log1p(max_cooccur)
            pi, pj = profiles.get(gi), profiles.get(gj)
            if pi is not None and pj is not None:
                ni, nj = np.linalg.norm(pi), np.linalg.norm(pj)
                if ni > 0 and nj > 0:
                    matrix[i, j, 6] = matrix[j, i, 6] = np.dot(pi, pj) / (ni * nj)
            matrix[i, j, 7] = matrix[j, i, 7] = 1.0 if (gi in hub_set and gj in hub_set) else 0.0
            matrix[i, j, 8] = matrix[j, i, 8] = 1.0 if pair in sl_pairs else 0.0
    return matrix, GRAPH_EDGE_DIM


def compute_patient_edge_features(node_features_np):
    PATIENT_EDGE_DIM = 4
    nf = node_features_np if isinstance(node_features_np, np.ndarray) else node_features_np.numpy()
    B, N, _ = nf.shape
    feats = np.zeros((B, N, N, PATIENT_EDGE_DIM), dtype=np.float32)
    log_hr = np.abs(nf[:, :, 0])
    harmful = nf[:, :, 10] > 0.5
    protective = nf[:, :, 11] > 0.5
    feats[:, :, :, 0] = log_hr[:, :, None] * log_hr[:, None, :]
    feats[:, :, :, 1] = (harmful[:, :, None] & harmful[:, None, :]).astype(np.float32)
    feats[:, :, :, 2] = (protective[:, :, None] & protective[:, None, :]).astype(np.float32)
    feats[:, :, :, 3] = ((harmful[:, :, None] & protective[:, None, :]) |
                          (protective[:, :, None] & harmful[:, None, :])).astype(np.float32)
    return feats, PATIENT_EDGE_DIM


def gather_edge_features(gene_pair_matrix, patient_edge_feats, gene_indices, gene_masks):
    B, N = gene_indices.shape
    safe_idx = gene_indices.clamp(0, gene_pair_matrix.shape[0] - 1)
    idx_i = safe_idx.unsqueeze(2).expand(B, N, N)
    idx_j = safe_idx.unsqueeze(1).expand(B, N, N)
    graph_feats = gene_pair_matrix[idx_i, idx_j]
    pair_mask = (gene_masks.unsqueeze(1) * gene_masks.unsqueeze(2)).unsqueeze(-1)
    graph_feats = graph_feats * pair_mask
    return torch.cat([graph_feats, patient_edge_feats * pair_mask], dim=-1)


# =========================================================================
# Training helpers
# =========================================================================

def train_one_fold(model, config, device, train_idx, val_idx,
                   nf, nm, ct, clinical, atlas_sums, times, events,
                   gene_pair_matrix_t, patient_edge_feats_t, gene_indices_t,
                   block_ids_t, channel_ids_t,
                   args, phase1_epochs=0, backbone_lr_scale=0.1):
    """Train one fold with optional phased fine-tuning."""
    batch_size = args.batch_size

    train_t = torch.tensor(train_idx, dtype=torch.long)
    val_t = torch.tensor(val_idx, dtype=torch.long)

    if phase1_epochs > 0:
        # Phase 1: Freeze backbone, train only readout + embeddings
        backbone_params = set()
        for name, param in model.named_parameters():
            if any(k in name for k in ['node_encoder', 'intra_block', 'block_to_channel',
                                        'cross_channel', 'channel_pool', 'age_mod']):
                param.requires_grad = False
                backbone_params.add(name)

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt1 = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)

        print(f"    Phase 1: {sum(p.numel() for p in trainable):,} trainable params "
              f"(backbone frozen)")

        for epoch in range(phase1_epochs):
            model.train()
            perm = np.random.permutation(len(train_idx))
            for b_start in range(0, len(perm), batch_size):
                b_rel = perm[b_start:b_start + batch_size]
                b_abs = train_t[b_rel]
                opt1.zero_grad()
                b_edge = gather_edge_features(
                    gene_pair_matrix_t, patient_edge_feats_t[b_abs],
                    gene_indices_t[b_abs], nm[b_abs],
                )
                hazard = model(nf[b_abs], nm[b_abs], ct[b_abs],
                               clinical[b_abs], atlas_sums[b_abs],
                               b_edge, block_ids_t[b_abs], channel_ids_t[b_abs])
                loss = cox_ph_loss(hazard, times[b_abs].to(device), events[b_abs].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt1.step()

        # Unfreeze
        for param in model.parameters():
            param.requires_grad = True

    # Phase 2 (or only phase if phase1_epochs=0): Full training with discriminative LR
    backbone_params_list = []
    readout_params_list = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['node_encoder', 'intra_block', 'block_to_channel',
                                    'cross_channel', 'channel_pool', 'age_mod']):
            backbone_params_list.append(param)
        else:
            readout_params_list.append(param)

    if phase1_epochs > 0:
        # Discriminative LR for pre-trained backbone
        param_groups = [
            {'params': backbone_params_list, 'lr': args.lr * backbone_lr_scale},
            {'params': readout_params_list, 'lr': args.lr},
        ]
    else:
        # Baseline: uniform LR
        param_groups = [{'params': list(model.parameters()), 'lr': args.lr}]

    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_c = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(train_idx))
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(perm), batch_size):
            b_rel = perm[b_start:b_start + batch_size]
            b_abs = train_t[b_rel]
            optimizer.zero_grad()
            b_edge = gather_edge_features(
                gene_pair_matrix_t, patient_edge_feats_t[b_abs],
                gene_indices_t[b_abs], nm[b_abs],
            )
            hazard = model(nf[b_abs], nm[b_abs], ct[b_abs],
                           clinical[b_abs], atlas_sums[b_abs],
                           b_edge, block_ids_t[b_abs], channel_ids_t[b_abs])
            loss = cox_ph_loss(hazard, times[b_abs].to(device), events[b_abs].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = []
                for v_start in range(0, len(val_idx), batch_size):
                    v_end = min(v_start + batch_size, len(val_idx))
                    v_abs = val_t[v_start:v_end]
                    v_edge = gather_edge_features(
                        gene_pair_matrix_t, patient_edge_feats_t[v_abs],
                        gene_indices_t[v_abs], nm[v_abs],
                    )
                    h = model(nf[v_abs], nm[v_abs], ct[v_abs],
                              clinical[v_abs], atlas_sums[v_abs],
                              v_edge, block_ids_t[v_abs], channel_ids_t[v_abs])
                    val_preds.append(h.cpu())
                h_val = torch.cat(val_preds).numpy().flatten()

            e_val = events[val_idx].numpy().astype(bool)
            t_val = times[val_idx].numpy()
            valid = t_val > 0
            try:
                c = concordance_index_censored(e_val[valid], t_val[valid], h_val[valid])[0]
            except Exception:
                c = 0.5

            if c > best_c:
                best_c = c
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1:3d}: cox={epoch_loss/n_batches:.4f} "
                      f"C={c:.4f} best={best_c:.4f}", flush=True)

            if no_improve >= args.patience // 5:
                print(f"      Early stop epoch {epoch+1}, C-index: {best_c:.4f}")
                break

    return best_c, best_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--phase1-epochs', type=int, default=5,
                        help='Epochs with frozen backbone (pre-trained only)')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1,
                        help='LR multiplier for backbone in phase 2')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(PRETRAIN_DIR, "pretrained_backbone.pt"),
                        help='Path to pre-trained backbone checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check pre-trained checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Pre-trained checkpoint not found: {args.checkpoint}")
        print("Run pretrain_depmap.py first.")
        sys.exit(1)

    pretrain_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"Loaded pre-trained checkpoint: epoch {pretrain_ckpt['epoch']}, "
          f"val_corr={pretrain_ckpt['val_corr']:.3f}")

    # === Load patient data ===
    ds = AtlasDataset()
    data = ds.build_features()

    gene_vocab = {}
    for patient_genes in data['gene_names']:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    G = len(gene_vocab)
    print(f"\n  Gene vocabulary: {G} unique genes")

    gene_indices = np.zeros((len(data['gene_names']), MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long)

    gene_block, n_blocks, n_channels = load_block_assignments()
    channel_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}

    block_ids = np.full((len(data['gene_names']), MAX_NODES), n_blocks, dtype=np.int64)
    channel_ids = np.full((len(data['gene_names']), MAX_NODES), n_channels, dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT':
                info = gene_block.get(g)
                if info:
                    block_ids[b, s] = info['block_id']
                    channel_ids[b, s] = info['channel_id']
                else:
                    ch = CHANNEL_MAP.get(g)
                    if ch and ch in channel_idx:
                        channel_ids[b, s] = channel_idx[ch]

    block_ids_t = torch.tensor(block_ids, dtype=torch.long)
    channel_ids_t = torch.tensor(channel_ids, dtype=torch.long)

    print("  Computing gene-pair matrix...", flush=True)
    graph_data = load_graph_data()
    gene_pair_matrix, graph_edge_dim = compute_gene_pair_matrix(graph_data, gene_vocab)
    gene_pair_matrix_t = torch.tensor(gene_pair_matrix, dtype=torch.float32)

    patient_edge_feats, patient_edge_dim = compute_patient_edge_features(data['node_features'])
    patient_edge_feats_t = torch.tensor(patient_edge_feats, dtype=torch.float32)
    total_edge_dim = graph_edge_dim + patient_edge_dim
    print(f"  Edge dim: {total_edge_dim}")

    # Temporal features
    patient_ids = ds.clinical['patientId'].tolist()
    te = TemporalEstimator(ds.clinical)
    years, confs = te.estimate_all(patient_ids)
    temporal_feats = year_features(years, confs)
    temporal_t = torch.tensor(temporal_feats, dtype=torch.float32).to(device)

    nf = data['node_features'].to(device)
    nm = data['node_masks'].to(device)
    ct = data['cancer_types'].to(device)
    ages = data['ages'].to(device)
    sexes = data['sexes'].to(device)
    atlas_sums = data['atlas_sums'].to(device)
    times = data['times']
    events = data['events']
    n_cancer_types = data['n_cancer_types']

    clinical = torch.cat([ages.unsqueeze(-1), sexes.unsqueeze(-1), temporal_t], dim=-1)
    gene_pair_matrix_t = gene_pair_matrix_t.to(device)
    patient_edge_feats_t = patient_edge_feats_t.to(device)
    gene_indices_t = gene_indices_t.to(device)
    block_ids_t = block_ids_t.to(device)
    channel_ids_t = channel_ids_t.to(device)

    config = {
        'node_feat_dim': NODE_FEAT_DIM,
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'n_intra_layers': args.n_layers,
        'dropout': args.dropout,
        'n_cancer_types': n_cancer_types,
        'edge_feat_dim': total_edge_dim,
        'n_channels': n_channels,
        'n_blocks': n_blocks,
    }

    # === Holdback ===
    n_total = len(events)
    np.random.seed(args.seed)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    n_holdback = int(n_total * 0.15)
    holdback_idx = all_idx[:n_holdback]
    cv_idx = all_idx[n_holdback:]
    events_cv = events[cv_idx].numpy()

    print(f"\n  Patients: {n_total}, Holdback: {n_holdback}, CV: {len(cv_idx)}")

    # === Cross-validation: both pretrained and baseline ===
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    pretrained_results = []
    baseline_results = []

    for fold, (train_rel, val_rel) in enumerate(skf.split(cv_idx, events_cv)):
        train_idx = cv_idx[train_rel]
        val_idx = cv_idx[val_rel]

        # --- Pre-trained ---
        print(f"\n=== Fold {fold} — PRE-TRAINED ===")
        model_pt = AtlasTransformerV6(config).to(device)

        # Load pre-trained backbone weights (skip mismatched keys)
        backbone_state = pretrain_ckpt['backbone']
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
            print(f"    Loaded {loaded} params from pre-trained backbone "
                  f"(skipped {skipped} mismatched)")

        c_pt, _ = train_one_fold(
            model_pt, config, device, train_idx, val_idx,
            nf, nm, ct, clinical, atlas_sums, times, events,
            gene_pair_matrix_t, patient_edge_feats_t, gene_indices_t,
            block_ids_t, channel_ids_t, args,
            phase1_epochs=args.phase1_epochs,
            backbone_lr_scale=args.backbone_lr_scale,
        )
        pretrained_results.append(c_pt)
        print(f"  Fold {fold} pre-trained: C={c_pt:.4f}")

        # --- Baseline (random init) ---
        print(f"\n=== Fold {fold} — BASELINE ===")
        model_bl = AtlasTransformerV6(config).to(device)
        c_bl, _ = train_one_fold(
            model_bl, config, device, train_idx, val_idx,
            nf, nm, ct, clinical, atlas_sums, times, events,
            gene_pair_matrix_t, patient_edge_feats_t, gene_indices_t,
            block_ids_t, channel_ids_t, args,
            phase1_epochs=0,
        )
        baseline_results.append(c_bl)
        print(f"  Fold {fold} baseline: C={c_bl:.4f}")

    # === Summary ===
    pt_mean = np.mean(pretrained_results)
    pt_std = np.std(pretrained_results)
    bl_mean = np.mean(baseline_results)
    bl_std = np.std(baseline_results)
    delta = pt_mean - bl_mean

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING COMPARISON")
    print(f"{'='*60}")
    print(f"  Pre-trained: {pt_mean:.4f} +/- {pt_std:.4f}  {pretrained_results}")
    print(f"  Baseline:    {bl_mean:.4f} +/- {bl_std:.4f}  {baseline_results}")
    print(f"  Delta:       {delta:+.4f}")
    print(f"{'='*60}")

    results = {
        'model': 'AtlasTransformerV6',
        'task': 'DepMap pre-training fine-tune comparison',
        'pretrained': {
            'mean_c': float(pt_mean), 'std_c': float(pt_std),
            'folds': [float(c) for c in pretrained_results],
        },
        'baseline': {
            'mean_c': float(bl_mean), 'std_c': float(bl_std),
            'folds': [float(c) for c in baseline_results],
        },
        'delta': float(delta),
        'config': config,
        'args': vars(args),
        'pretrain_checkpoint': {
            'epoch': pretrain_ckpt['epoch'],
            'val_corr': pretrain_ckpt['val_corr'],
        },
        'n_patients': n_total,
        'n_holdback': n_holdback,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
