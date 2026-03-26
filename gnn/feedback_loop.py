"""
Graph-transformer feedback loop.

Each cycle:
  1. Train transformer on current graph
  2. Extract attention → precipitate as new edges/features
  3. Write back to Neo4j graph
  4. Rebuild PyG export
  5. Measure C-index delta
  6. Stop when delta < threshold (fixed point)

The graph accumulates knowledge. The transformer is temporary compute.

Usage:
    python3 -u -m gnn.feedback_loop                    # run full loop
    python3 -u -m gnn.feedback_loop --max-cycles 1     # single cycle
    python3 -u -m gnn.feedback_loop --precipitate-only  # skip training, just precipitate
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import GNN_CACHE, CHANNEL_NAMES, CHANNEL_MAP, HUB_GENES

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOOP_DIR = os.path.join(RESULTS_DIR, "feedback_loop")
GRAPH_PATH = os.path.join(GNN_CACHE, "hetero_graph.pt")

_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)


def _step(name, actual=None, elapsed=None):
    status = f"{actual:,}" if actual is not None else ""
    t = f"[{elapsed:.1f}s]" if elapsed is not None else ""
    print(f"  {name:.<50s} {status:>15s}  {t}", flush=True)


# =========================================================================
# STEP 1: Train transformer
# =========================================================================

def train_transformer(cycle, model_version="v2"):
    """Train transformer on current graph, return model path and C-index.

    For V3: delegates to the full training script (computes graph features).
    For V2: trains inline.

    If a trained model already exists for this cycle, loads it instead.
    """
    from gnn.data.atlas_dataset import AtlasDataset

    print(f"\n{'='*70}")
    print(f"  CYCLE {cycle}: TRAINING TRANSFORMER ({model_version})")
    print(f"{'='*70}\n")

    t0 = time.time()
    cycle_dir = os.path.join(LOOP_DIR, f"cycle_{cycle}")
    os.makedirs(cycle_dir, exist_ok=True)

    ds = AtlasDataset()
    data = ds.build_features()

    if model_version == "v3":
        return _train_or_load_v3(cycle, cycle_dir, data, ds, t0)
    else:
        return _train_v2(cycle, cycle_dir, data, t0)


def _train_or_load_v3(cycle, cycle_dir, data, ds, t0):
    """For V3: run the full training script, or load existing model."""
    import subprocess
    import json as json_mod
    from gnn.models.atlas_transformer_v3 import AtlasTransformerV3

    results_dir = os.path.join(RESULTS_DIR, "atlas_transformer_v3")
    results_json = os.path.join(results_dir, "results.json")
    model_path = os.path.join(results_dir, "fold_0", "best_model.pt")

    # If cycle > 0, retrain (graph has changed). Cycle 0 can reuse existing.
    if cycle == 0 and os.path.exists(model_path):
        print(f"  Loading existing V3 model (cycle 0)", flush=True)
    else:
        print(f"  Training V3 via training script...", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", "-m", "gnn.scripts.train_atlas_transformer_v3"],
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"V3 training failed with code {result.returncode}")

    # Load results
    with open(results_json) as f:
        results = json_mod.load(f)
    best_c = results['mean_c_index']
    config = results['config']

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = AtlasTransformerV3(config)
    model.load_state_dict(checkpoint)
    model.eval()

    # Compute graph features for precipitation (needed for forward pass)
    from gnn.scripts.train_atlas_transformer_v3 import (
        compute_graph_features, GRAPH_FEAT_DIM,
    )
    from gnn.scripts.expanded_graph_scorer import (
        load_expanded_channel_map, fetch_string_expanded,
        build_expanded_graph, compute_cooccurrence,
    )
    from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles
    from gnn.config import ANALYSIS_CACHE, HUB_GENES
    from collections import defaultdict
    import pandas as pd
    import networkx as nx

    expanded_cm = load_expanded_channel_map()
    expanded_genes = set(expanded_cm.keys())
    N = data['node_features'].shape[0]

    atlas_patient_ids = ds.clinical['patientId'].tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(atlas_patient_ids)}

    # Load patient-gene mutations from Neo4j (source of truth)
    from neo4j import GraphDatabase as _GD
    _driver = _GD.driver("bolt://localhost:7687", auth=("neo4j", "openknowledgegraph"))
    with _driver.session() as _s:
        _result = _s.run("""
            MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)
            RETURN p.id AS pid, g.name AS gene
        """)
        _neo4j_muts = [(r["pid"], r["gene"]) for r in _result]
    _driver.close()

    patient_genes_map = defaultdict(set)
    for pid, gene in _neo4j_muts:
        if gene in expanded_genes and pid in pid_to_idx:
            patient_genes_map[pid_to_idx[pid]].add(gene)

    ct_map = data['cancer_type_map']
    idx_to_ct = {v: k for k, v in ct_map.items()}
    ct_arr = data['cancer_types'].numpy()
    ct_per_patient = {i: idx_to_ct.get(int(ct_arr[i]), "Other") for i in range(N)}

    msk_genes = set()
    for genes in patient_genes_map.values():
        msk_genes |= genes
    msk_genes &= expanded_genes

    cooccurrence = compute_cooccurrence(patient_genes_map, ct_per_patient, min_count=10)

    # Load PPI from Neo4j instead of STRING API
    _driver2 = _GD.driver("bolt://localhost:7687", auth=("neo4j", "openknowledgegraph"))
    ppi_edges = []
    with _driver2.session() as _s2:
        _result2 = _s2.run("""
            MATCH (g1:Gene)-[r:PPI]-(g2:Gene)
            WHERE g1.name < g2.name
            RETURN g1.name AS gene1, g2.name AS gene2, r.score AS score
        """)
        for r in _result2:
            if r["gene1"] in msk_genes and r["gene2"] in msk_genes:
                ppi_edges.append((r["gene1"], r["gene2"], r["score"] or 0.4))
    _driver2.close()
    G, G_ppi = build_expanded_graph(expanded_cm, ppi_edges, cooccurrence)
    channel_profiles = compute_curated_profiles(G_ppi, expanded_cm)

    ppi_dists = {}
    for comp in nx.connected_components(G_ppi):
        sub = G_ppi.subgraph(comp)
        lengths = dict(nx.all_pairs_shortest_path_length(sub))
        for src, targets in lengths.items():
            for tgt, dist in targets.items():
                if src < tgt:
                    ppi_dists[(src, tgt)] = dist

    graph_features_np = compute_graph_features(
        patient_genes_map, G_ppi, expanded_cm, channel_profiles,
        ppi_dists, cooccurrence, ct_per_patient, N,
    )
    data['graph_features'] = torch.tensor(graph_features_np, dtype=torch.float32)

    # Copy model to cycle dir
    import shutil
    cycle_model_path = os.path.join(cycle_dir, "best_model.pt")
    shutil.copy2(model_path, cycle_model_path)

    print(f"  Cycle {0 if os.path.exists(model_path) else cycle} V3: "
          f"C = {best_c:.4f} [{time.time()-t0:.1f}s]")

    return model, data, best_c, cycle_model_path


def _train_v2(cycle, cycle_dir, data, t0):
    """Train V2 inline (no graph features needed)."""
    from gnn.models.atlas_transformer_v2 import AtlasTransformerV2
    from sklearn.model_selection import StratifiedKFold
    from sksurv.metrics import concordance_index_censored

    n = data['node_features'].shape[0]
    events_np = data['events'].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(np.arange(n), events_np))

    config = {
        'node_feat_dim': data['node_features'].shape[-1],
        'hidden_dim': 64,
        'dropout': 0.1,
        'n_cancer_types': data['n_cancer_types'],
        'n_heads': 4,
        'n_layers': 2,
    }
    model = AtlasTransformerV2(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    nf = data['node_features']
    nm = data['node_masks']
    cp = data['channel_pos_ids']
    ct = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    atlas = data['atlas_sums']
    times = data['times']
    events = data['events']
    clinical = torch.stack([ages, sexes], dim=-1)

    best_c = 0.0
    best_state = None
    no_improve = 0
    batch_size = 256

    for epoch in range(200):
        model.train()
        perm = np.random.permutation(train_idx)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(perm), batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            b_idx_t = torch.tensor(b_idx, dtype=torch.long)

            optimizer.zero_grad()
            hazard = model(
                nf[b_idx_t], nm[b_idx_t], cp[b_idx_t], ct[b_idx_t],
                clinical[b_idx_t], atlas[b_idx_t]
            )

            from gnn.models.cox_sage import cox_ph_loss
            loss = cox_ph_loss(hazard, times[b_idx_t], events[b_idx_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_t = torch.tensor(val_idx, dtype=torch.long)
                h_val = model(
                    nf[val_t], nm[val_t], cp[val_t], ct[val_t],
                    clinical[val_t], atlas[val_t]
                ).numpy()

            t_val = times[val_idx].numpy()
            e_val = events[val_idx].numpy().astype(bool)

            try:
                c = concordance_index_censored(e_val, t_val, h_val)[0]
            except Exception:
                c = 0.5

            if c > best_c:
                best_c = c
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}, "
                      f"val_C={c:.4f} (best={best_c:.4f})", flush=True)

            if no_improve >= 3:
                break

    model_path = os.path.join(cycle_dir, "best_model.pt")
    if best_state:
        torch.save(best_state, model_path)
        model.load_state_dict(best_state)

    print(f"  Cycle {cycle} V2: C = {best_c:.4f} [{time.time()-t0:.1f}s]")

    return model, data, best_c, model_path


# =========================================================================
# STEP 2: Extract attention (precipitate)
# =========================================================================

def precipitate(model, data, cycle, saved_n_ct=None):
    """Extract learned representations from transformer → graph-ready artifacts.

    Produces:
      - gene_gene_attention.csv: directed gene×gene attention edges
      - gene_self_attention.csv: per-gene self-attention weights
      - channel_channel_attention.npy: channel×channel matrix
      - patient_embeddings.npy: per-patient learned representations
      - cancer_type_gene_importance.csv: per-cancer-type gene importance
    """
    print(f"\n{'='*70}")
    print(f"  CYCLE {cycle}: PRECIPITATING ATTENTION → GRAPH")
    print(f"{'='*70}\n")

    t0 = time.time()
    cycle_dir = os.path.join(LOOP_DIR, f"cycle_{cycle}")
    os.makedirs(cycle_dir, exist_ok=True)

    model.eval()
    n_channels = len(CHANNEL_NAMES)

    if saved_n_ct is None:
        # Infer from model
        for name, param in model.named_parameters():
            if 'cancer_attn_bias' in name:
                saved_n_ct = param.shape[0]
                break
        if saved_n_ct is None:
            saved_n_ct = data['n_cancer_types']

    # Hook attention layers — works for both V2 and V3
    attn_store = {}
    is_v3 = hasattr(model.layers[0].attn, 'graph_attn_bias')

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x, cancer_type = input[0], input[1]
            # V3: (x, cancer_type, graph_context, attn_mask)
            # V2: (x, cancer_type, attn_mask)
            if is_v3:
                graph_context = input[2] if len(input) > 2 else None
                attn_mask = input[3] if len(input) > 3 else None
            else:
                graph_context = None
                attn_mask = input[2] if len(input) > 2 else None

            B, N, _ = x.shape
            q = module.q_proj(x).view(B, N, module.n_heads, module.head_dim).transpose(1, 2)
            k = module.k_proj(x).view(B, N, module.n_heads, module.head_dim).transpose(1, 2)

            # Cancer type modulates keys
            ct_key = module.cancer_key_mod(cancer_type)
            ct_key = ct_key.view(B, 1, module.n_heads, module.head_dim).transpose(1, 2)
            k = k + ct_key

            # V3: graph structure also modulates keys
            if is_v3 and graph_context is not None:
                g_key = module.graph_key_mod(graph_context)
                g_key = g_key.view(B, 1, module.n_heads, module.head_dim).transpose(1, 2)
                k = k + g_key

            attn = torch.matmul(q, k.transpose(-2, -1)) / module.scale

            # Cancer type attention bias
            ct_bias = module.cancer_attn_bias(cancer_type)
            attn = attn + ct_bias.unsqueeze(-1).unsqueeze(-1)

            # V3: graph structure attention bias
            if is_v3 and graph_context is not None:
                g_bias = module.graph_attn_bias(graph_context)
                attn = attn + g_bias.unsqueeze(-1).unsqueeze(-1)

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

            attn = torch.softmax(attn, dim=-1)
            attn_store[layer_idx] = attn.detach()

        return hook_fn

    for i, layer in enumerate(model.layers):
        layer.attn.register_forward_hook(make_hook(i))

    # --- Forward pass: collect gene×gene attention + embeddings ---
    batch_size = 512
    n_patients = data['node_features'].shape[0]
    n_batches = (n_patients + batch_size - 1) // batch_size

    nf = data['node_features']
    nm = data['node_masks']
    cp = data['channel_pos_ids']
    ct = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    atlas = data['atlas_sums']
    clinical = torch.stack([ages, sexes], dim=-1)

    # V3 needs graph_features
    graph_features = data.get('graph_features', None)

    # Gene name lookup from dataset
    gene_names = data.get('gene_names', None)

    # Accumulators
    gene_pair_sum = defaultdict(float)   # (gene_from, gene_to) → sum of attention
    gene_pair_count = defaultdict(int)
    gene_pair_cross = defaultdict(bool)  # is this a cross-channel edge?

    # Channel-level accumulators
    ch_sum = np.zeros((4, n_channels, n_channels), dtype=np.float64)
    ch_count = np.zeros((4, n_channels, n_channels), dtype=np.float64)

    # Per-cancer-type gene importance (sum of incoming attention per gene)
    ct_gene_importance = defaultdict(lambda: defaultdict(float))
    ct_gene_count = defaultdict(lambda: defaultdict(int))

    # Patient embeddings (from last layer hidden state)
    patient_embeddings = []

    with torch.no_grad():
        for b in range(n_batches):
            if b % 40 == 0:
                print(f"  Batch {b+1}/{n_batches}", flush=True)

            s = b * batch_size
            e = min(s + batch_size, n_patients)
            b_idx = slice(s, e)

            ct_clamped = ct[b_idx].clamp(max=saved_n_ct - 1)

            if is_v3 and graph_features is not None:
                out = model(nf[b_idx], nm[b_idx], cp[b_idx], ct_clamped,
                            clinical[b_idx], atlas[b_idx], graph_features[b_idx])
            else:
                out = model(nf[b_idx], nm[b_idx], cp[b_idx], ct_clamped,
                            clinical[b_idx], atlas[b_idx])

            # Get layer 1 attention (deeper layer)
            attn = attn_store[1].cpu().numpy()  # (B', heads, N, N)
            masks = nm[b_idx].numpy()
            cpids = cp[b_idx].numpy()
            cts = ct[b_idx].numpy()

            for i in range(attn.shape[0]):
                mask = masks[i]
                real_nodes = int(mask.sum())
                if real_nodes < 2:
                    continue

                ch_ids = cpids[i, :real_nodes] // 2
                ct_val = int(cts[i])

                # Get gene names for this patient's nodes
                patient_genes = []
                if gene_names is not None:
                    patient_genes = gene_names[s + i][:real_nodes]

                # Average across heads for gene-level
                attn_avg = attn[i, :, :real_nodes, :real_nodes].mean(axis=0)

                for qi in range(real_nodes):
                    for ki in range(real_nodes):
                        val = attn_avg[qi, ki]

                        # Channel-level (per head)
                        c_q, c_k = ch_ids[qi], ch_ids[ki]
                        for h in range(4):
                            ch_sum[h, c_q, c_k] += attn[i, h, qi, ki]
                            ch_count[h, c_q, c_k] += 1

                        # Gene-level (if we have names)
                        if patient_genes:
                            g_from = patient_genes[qi]
                            g_to = patient_genes[ki]
                            gene_pair_sum[(g_from, g_to)] += val
                            gene_pair_count[(g_from, g_to)] += 1
                            if c_q != c_k:
                                gene_pair_cross[(g_from, g_to)] = True

                            # Per-cancer-type gene importance (incoming attention)
                            ct_gene_importance[ct_val][g_to] += val
                            ct_gene_count[ct_val][g_to] += 1

    _step("Attention extraction", elapsed=time.time() - t0)

    # --- Save gene×gene attention ---
    t1 = time.time()
    rows = []
    for (g_from, g_to), total in gene_pair_sum.items():
        n_obs = gene_pair_count[(g_from, g_to)]
        rows.append({
            'from': g_from, 'to': g_to,
            'mean_attn': total / n_obs,
            'n_obs': n_obs,
            'cross_channel': gene_pair_cross.get((g_from, g_to), False),
        })
    gene_df = pd.DataFrame(rows)
    gene_path = os.path.join(cycle_dir, "gene_gene_attention.csv")
    gene_df.to_csv(gene_path, index=False)
    _step("Gene×gene edges", actual=len(gene_df), elapsed=time.time() - t1)

    # --- Save self-attention ---
    t1 = time.time()
    self_rows = []
    for (g_from, g_to), total in gene_pair_sum.items():
        if g_from == g_to:
            n_obs = gene_pair_count[(g_from, g_to)]
            self_rows.append({
                'gene': g_from,
                'self_attn': total / n_obs,
                'n_obs': n_obs,
            })
    self_df = pd.DataFrame(self_rows)
    self_path = os.path.join(cycle_dir, "gene_self_attention.csv")
    self_df.to_csv(self_path, index=False)
    _step("Self-attention values", actual=len(self_df), elapsed=time.time() - t1)

    # --- Save channel×channel matrix ---
    with np.errstate(divide='ignore', invalid='ignore'):
        ch_mean = np.where(ch_count > 0, ch_sum / ch_count, 0)
    ch_avg = ch_mean.mean(axis=0)
    np.save(os.path.join(cycle_dir, "channel_channel_attention.npy"), ch_avg)

    # --- Save per-cancer-type gene importance ---
    t1 = time.time()
    ct_rows = []
    inv_cancer_map = {v: k for k, v in data['cancer_type_map'].items()}
    for ct_val, gene_imp in ct_gene_importance.items():
        ct_name = inv_cancer_map.get(ct_val, f"CT_{ct_val}")
        for gene, total in gene_imp.items():
            n_obs = ct_gene_count[ct_val][gene]
            ct_rows.append({
                'cancer_type': ct_name,
                'gene': gene,
                'mean_importance': total / n_obs,
                'n_obs': n_obs,
            })
    ct_imp_df = pd.DataFrame(ct_rows)
    ct_imp_path = os.path.join(cycle_dir, "cancer_type_gene_importance.csv")
    ct_imp_df.to_csv(ct_imp_path, index=False)
    _step("Cancer-type gene importance", actual=len(ct_imp_df), elapsed=time.time() - t1)

    return {
        'gene_attention_path': gene_path,
        'self_attention_path': self_path,
        'cancer_gene_importance_path': ct_imp_path,
        'cycle_dir': cycle_dir,
    }


# =========================================================================
# STEP 3: Write back to Neo4j
# =========================================================================

def write_to_graph(precipitate_results, cycle, dry_run=False):
    """Write precipitated attention back to Neo4j as edges/properties.

    Uses GraphGateway — all writes are logged, never deleted.
    Set dry_run=True to preview without writing.
    """
    from gnn.data.graph_changelog import GraphGateway

    mode = "DRY RUN" if dry_run else "WRITING"
    print(f"\n{'='*70}")
    print(f"  CYCLE {cycle}: {mode} TO NEO4J")
    print(f"{'='*70}\n")

    t0 = time.time()
    gw = GraphGateway(dry_run=dry_run)

    try:
        # --- NEVER delete existing edges ---
        # All writes go through GraphGateway with provenance logging.

        # Snapshot current state before any changes
        n_before = gw.count_edges("ATTENDS_TO")
        print(f"  ATTENDS_TO before: {n_before:,}")

        # --- Load gene attention ---
        gene_df = pd.read_csv(precipitate_results['gene_attention_path'])
        if len(gene_df) == 0:
            print("  [SKIP] No gene attention edges to write (empty CSV)")
            gw.close()
            return 0, 0

        # Filter: mean_attn > 0.05 and n_obs >= 100
        edges = gene_df[(gene_df['mean_attn'] > 0.05) & (gene_df['n_obs'] >= 100)].copy()
        self_attn = edges[edges['from'] == edges['to']]
        cross_attn = edges[edges['from'] != edges['to']]

        # Compute asymmetry
        reverse_lookup = {}
        for _, row in cross_attn.iterrows():
            reverse_lookup[(row['to'], row['from'])] = row['mean_attn']

        attn_edges = []
        for _, row in cross_attn.iterrows():
            rev = reverse_lookup.get((row['from'], row['to']), row['mean_attn'])
            asym = row['mean_attn'] / (rev + 1e-8)
            attn_edges.append({
                'from': row['from'], 'to': row['to'],
                'weight': round(float(row['mean_attn']), 4),
                'asymmetry': round(float(asym), 4),
                'n_obs': int(row['n_obs']),
                'cross_channel': bool(row['cross_channel']),
                'cycle': cycle,
            })

        # MERGE via gateway (logged, never deletes)
        source = f"transformer_v3_cycle_{cycle}"
        n_attn = gw.merge_edges("ATTENDS_TO", attn_edges, source=source)
        _step("ATTENDS_TO edges merged", actual=n_attn, elapsed=time.time() - t0)

        # Self-attention as node properties
        t1 = time.time()
        self_updates = [{'name': row['from'],
                         'self_attn': round(float(row['mean_attn']), 4),
                         'self_attn_cycle': cycle}
                        for _, row in self_attn.iterrows()]
        gw.set_node_properties("Gene", "name", self_updates, source=source)

        # Self-loop edges
        self_edges = [{'from': row['from'], 'to': row['from'],
                       'weight': round(float(row['mean_attn']), 4),
                       'self_loop': True, 'cycle': cycle}
                      for _, row in self_attn.iterrows()]
        gw.merge_edges("ATTENDS_TO", self_edges, source=source)
        _step("Self-attention merged", actual=len(self_edges), elapsed=time.time() - t1)

        # --- MERGE GROUP_ATTENDS_TO from gene-level attention ---
        t1 = time.time()

        gene_edges = gene_df[(gene_df['mean_attn'] > 0.03) &
                             (gene_df['n_obs'] >= 50) &
                             (gene_df['from'] != gene_df['to'])].copy()

        # Build gene → group mapping (read-only query, no log needed)
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                      auth=("neo4j", "openknowledgegraph"))
        with driver.session() as s:
            result = s.run("""
                MATCH (mg:MutationGroup)-[:HAS_GENE]->(g:Gene)
                WHERE mg.n_patients >= 5
                RETURN mg.mutation_key AS key, collect(g.name) AS genes
            """)
            group_genes = {r['key']: set(r['genes']) for r in result}
        driver.close()

        gene_to_groups = defaultdict(set)
        for key, genes in group_genes.items():
            for g in genes:
                gene_to_groups[g].add(key)

        group_pair_sum = defaultdict(float)
        group_pair_count = defaultdict(int)

        for _, row in gene_edges.iterrows():
            g_from, g_to = row['from'], row['to']
            for grp_a in gene_to_groups.get(g_from, []):
                for grp_b in gene_to_groups.get(g_to, []):
                    if grp_a != grp_b:
                        group_pair_sum[(grp_a, grp_b)] += row['mean_attn']
                        group_pair_count[(grp_a, grp_b)] += 1

        ga_edges = []
        for (grp_a, grp_b), total in group_pair_sum.items():
            cnt = group_pair_count[(grp_a, grp_b)]
            mean_w = total / cnt
            if mean_w > 0.08:
                ga_edges.append({
                    'from': grp_a, 'to': grp_b,
                    'weight': round(mean_w, 4),
                    'cycle': cycle,
                })

        n_group = gw.merge_edges(
            "GROUP_ATTENDS_TO", ga_edges, source=source,
            match_from=("MutationGroup", "mutation_key"),
            match_to=("MutationGroup", "mutation_key"),
        )
        _step("GROUP_ATTENDS_TO edges merged", actual=n_group, elapsed=time.time() - t1)

        # Snapshot after
        n_after = gw.count_edges("ATTENDS_TO")
        print(f"  ATTENDS_TO after: {n_after:,} (delta: {n_after - n_before:+,})")

    finally:
        gw.close()

    return len(attn_edges), len(ga_edges)


# =========================================================================
# STEP 4: Rebuild PyG export
# =========================================================================

def rebuild_pyg(cycle):
    """Rebuild the PyG HeteroData from Neo4j."""
    print(f"\n{'='*70}")
    print(f"  CYCLE {cycle}: REBUILDING PyG GRAPH")
    print(f"{'='*70}\n")

    from gnn.data.graph_to_pyg import build_hetero_graph
    data = build_hetero_graph()
    return data


# =========================================================================
# STEP 5: Measure C-index (quick eval via Cox-SAGE)
# =========================================================================

def measure_c_index(cycle):
    """Quick C-index measurement using Cox-SAGE on rebuilt graph."""
    from sklearn.model_selection import StratifiedKFold
    from sksurv.metrics import concordance_index_censored
    from gnn.models.cox_sage import CoxSAGE, cox_ph_loss

    print(f"\n{'='*70}")
    print(f"  CYCLE {cycle}: MEASURING C-INDEX")
    print(f"{'='*70}\n")

    t0 = time.time()
    data = torch.load(GRAPH_PATH, weights_only=False)

    os_months = data['patient'].os_months.numpy()
    events = data['patient'].event.numpy()
    n_patients = len(os_months)
    valid_mask = os_months > 0
    valid_idx = np.where(valid_mask)[0]

    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    x_dict = {
        'gene': data['gene'].x,
        'patient': data['patient'].x,
        'mutation_group': data['mutation_group'].x,
    }

    # Single fold for speed
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(valid_idx, events[valid_idx]))
    train_patients = valid_idx[train_idx]
    val_patients = valid_idx[val_idx]

    train_mask = torch.zeros(n_patients, dtype=torch.bool)
    train_mask[train_patients] = True

    model = CoxSAGE(
        metadata=data.metadata(),
        gene_dim=data['gene'].x.shape[1],
        patient_dim=data['patient'].x.shape[1],
        group_dim=data['mutation_group'].x.shape[1],
        hidden=64, n_layers=2, dropout=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    time_t = torch.tensor(os_months, dtype=torch.float32)
    event_t = torch.tensor(events, dtype=torch.long)

    best_c = 0.0
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        hazard = model(x_dict, edge_index_dict)
        loss = cox_ph_loss(hazard[train_mask], time_t[train_mask], event_t[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                h_all = model(x_dict, edge_index_dict).numpy()
            h_val = h_all[val_patients]
            e_val = events[val_patients].astype(bool)
            t_val = os_months[val_patients]
            try:
                c = concordance_index_censored(e_val, t_val, h_val)[0]
            except Exception:
                c = 0.5
            best_c = max(best_c, c)

    print(f"  Cycle {cycle} GNN C-index: {best_c:.4f} [{time.time()-t0:.1f}s]")
    return best_c


# =========================================================================
# STEP 6: Convergence check
# =========================================================================

def check_convergence(history, threshold=0.005):
    """Check if the loop has converged (delta C < threshold)."""
    if len(history) < 2:
        return False

    delta = history[-1]['gnn_c'] - history[-2]['gnn_c']
    print(f"\n  Delta C-index: {delta:+.4f} (threshold: {threshold})")

    if abs(delta) < threshold:
        print(f"  CONVERGED — graph has absorbed available signal")
        return True
    return False


# =========================================================================
# MAIN LOOP
# =========================================================================

def run_loop(max_cycles=5, threshold=0.005, model_version="v2", dry_run=False):
    """Run the full feedback loop. Set dry_run=True to preview graph changes."""
    print(f"\n{'#'*70}")
    print(f"  GRAPH-TRANSFORMER FEEDBACK LOOP")
    print(f"  Max cycles: {max_cycles}, convergence threshold: {threshold}")
    print(f"  Transformer: {model_version}")
    print(f"{'#'*70}")

    os.makedirs(LOOP_DIR, exist_ok=True)
    history = []

    # Load existing history if resuming
    history_path = os.path.join(LOOP_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        print(f"\n  Resuming from cycle {len(history)}")

    start_cycle = len(history)

    for cycle in range(start_cycle, start_cycle + max_cycles):
        t_cycle = time.time()

        # 1. Train transformer
        model, data, transformer_c, model_path = train_transformer(cycle, model_version)

        # 2. Precipitate
        precip = precipitate(model, data, cycle)

        # 3. Write to Neo4j (or dry run)
        n_attn, n_group = write_to_graph(precip, cycle, dry_run=dry_run)

        # 4. Gap analysis — find weaknesses, generate new edges
        from gnn.gap_analysis_loop import run_gap_analysis
        gap_summary = run_gap_analysis(
            cycle=cycle, dry_run=dry_run, min_gap=0.015
        )
        n_gap_proposals = (
            gap_summary.get("n_prognostic_proposals", 0)
            + gap_summary.get("n_weight_adjustments", 0)
            + gap_summary.get("n_cooccur_proposals", 0)
        )

        # 5. Refresh all computed properties — graph must self-heal
        # Any denormalized values (log_hr, tissue_delta, channel_profile,
        # confidence, function, MutationGroup stats) recomputed from current
        # graph state. Stale properties = stale features = stuck model.
        from gnn.scripts.refresh_graph_properties import refresh_all
        refresh_summary = refresh_all(dry_run=dry_run)

        # 6. Rebuild PyG (reads refreshed state)
        rebuild_pyg(cycle)

        # 7. Measure GNN C-index
        gnn_c = measure_c_index(cycle)

        # Record
        entry = {
            'cycle': cycle,
            'transformer_c': transformer_c,
            'gnn_c': gnn_c,
            'n_attn_edges': n_attn,
            'n_group_edges': n_group,
            'n_gap_proposals': n_gap_proposals,
            'gap_profiles': gap_summary.get("n_gap_profiles", 0),
            'refresh_elapsed_s': refresh_summary.get("elapsed_s", 0),
            'time_s': time.time() - t_cycle,
        }
        history.append(entry)

        # Save history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Print progress
        print(f"\n{'='*70}")
        print(f"  CYCLE {cycle} SUMMARY")
        print(f"{'='*70}")
        print(f"  Transformer C:  {transformer_c:.4f}")
        print(f"  GNN C:          {gnn_c:.4f}")
        print(f"  ATTENDS_TO:     {n_attn:,} edges")
        print(f"  GROUP_ATTENDS:  {n_group:,} edges")
        print(f"  Gap proposals:  {n_gap_proposals:,} ({entry['gap_profiles']} CTs profiled)")
        print(f"  Time:           {entry['time_s']:.1f}s")

        if len(history) > 1:
            print(f"\n  History:")
            print(f"  {'Cycle':>6s} {'Transformer':>12s} {'GNN':>12s} {'Delta':>8s}")
            for h in history:
                delta = ""
                idx = history.index(h)
                if idx > 0:
                    delta = f"{h['gnn_c'] - history[idx-1]['gnn_c']:+.4f}"
                print(f"  {h['cycle']:6d} {h['transformer_c']:12.4f} "
                      f"{h['gnn_c']:12.4f} {delta:>8s}")

        # 6. Convergence check
        if check_convergence(history, threshold):
            break

    print(f"\n{'#'*70}")
    print(f"  FEEDBACK LOOP COMPLETE — {len(history)} cycles")
    print(f"{'#'*70}")

    return history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cycles', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.005)
    parser.add_argument('--model', default='v2', choices=['v2', 'v3'])
    parser.add_argument('--precipitate-only', action='store_true',
                        help='Load existing model and precipitate without training')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview graph changes without writing to Neo4j')
    args = parser.parse_args()

    if args.precipitate_only:
        # Load most recent model and precipitate
        from gnn.models.atlas_transformer_v2 import AtlasTransformerV2
        from gnn.data.atlas_dataset import AtlasDataset

        ds = AtlasDataset()
        data = ds.build_features()

        # Find most recent cycle
        cycle = 0
        while os.path.exists(os.path.join(LOOP_DIR, f"cycle_{cycle+1}")):
            cycle += 1

        model_path = os.path.join(LOOP_DIR, f"cycle_{cycle}", "best_model.pt")
        if not os.path.exists(model_path):
            # Fall back to V2 fold 0
            model_path = os.path.join(RESULTS_DIR, "atlas_transformer_v2",
                                      "fold_0", "best_model.pt")
            cycle = 0

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        saved_n_ct = checkpoint.get(
            'layers.0.attn.cancer_attn_bias.weight',
            next(v for k, v in checkpoint.items() if 'cancer_attn_bias' in k)
        ).shape[0]

        config = {
            'node_feat_dim': data['node_features'].shape[-1],
            'hidden_dim': 64, 'dropout': 0.0,
            'n_cancer_types': saved_n_ct, 'n_heads': 4, 'n_layers': 2,
        }
        model = AtlasTransformerV2(config)
        model.load_state_dict(checkpoint)

        precip = precipitate(model, data, cycle, saved_n_ct=saved_n_ct)
        write_to_graph(precip, cycle)
        rebuild_pyg(cycle)
        measure_c_index(cycle)
    else:
        run_loop(args.max_cycles, args.threshold, args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
