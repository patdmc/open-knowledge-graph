"""
Learning loop — the graph learns, the transformer discovers.

Each cycle:
  1. Measure: hierarchical graph scorer evaluates current graph
  2. Discover: transformer extracts attention → candidate edges
  3. Gate: score WITH candidates, measure delta
  4. Commit: delta positive → write through gateway. Negative → discard.
  5. Invalidate caches — schema may have changed
  6. Snapshot if WAL is large

DYNAMIC: Edge types and node properties are discovered from the graph
at each cycle. When new data sources are integrated between cycles,
the loop automatically picks up the new edge types and node properties.

Usage:
    python3 -u -m gnn.learning_loop                     # full loop
    python3 -u -m gnn.learning_loop --max-cycles 1      # single cycle
    python3 -u -m gnn.learning_loop --measure-only       # just measure
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOOP_DIR = os.path.join(RESULTS_DIR, "learning_loop")


def _banner(text, cycle=None):
    prefix = f"CYCLE {cycle}: " if cycle is not None else ""
    print(f"\n{'='*70}")
    print(f"  {prefix}{text}")
    print(f"{'='*70}\n", flush=True)


# =========================================================================
# STEP 1: Measure — hierarchical graph scorer baseline
# =========================================================================

def measure(scorer=None):
    """Score all patients with current graph state. Returns C-index + arrays."""
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import StratifiedKFold
    from gnn.models.hierarchical_scorer import HierarchicalScorer

    _banner("MEASURING GRAPH SCORER BASELINE")

    if scorer is None:
        scorer = HierarchicalScorer()
        scorer.load()

    scores, times, events, details = scorer.score_all_patients()
    valid = times > 0
    s, t, e = scores[valid], times[valid], events[valid]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_cs = []
    for fold, (_, val_idx) in enumerate(skf.split(np.arange(len(e)), e)):
        try:
            c = concordance_index_censored(e[val_idx].astype(bool), t[val_idx], s[val_idx])[0]
        except Exception:
            c = 0.5
        fold_cs.append(c)

    mean_c = float(np.mean(fold_cs))
    print(f"  Graph scorer C-index: {mean_c:.4f} ± {np.std(fold_cs):.4f}")
    return scorer, mean_c, scores, times, events, details


# =========================================================================
# STEP 2: Discover — transformer extracts candidate edges
# =========================================================================

def _discover_v5(cycle, schema):
    """Use V5 hierarchical transformer to discover candidate edges."""
    import torch
    from gnn.data.atlas_dataset import AtlasDataset
    from gnn.models.atlas_transformer_v5 import AtlasTransformerV5

    results_dir = os.path.join(RESULTS_DIR, "atlas_transformer_v5")
    model_path = os.path.join(results_dir, "fold_0", "best_model.pt")

    if not os.path.exists(model_path):
        return None

    results_json = os.path.join(results_dir, "results.json")
    with open(results_json) as f:
        config = json.load(f)['config']

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = AtlasTransformerV5(config)
    model.load_state_dict(checkpoint)
    model.eval()

    ds = AtlasDataset()
    data = ds.build_v5_features(schema=schema)

    return _extract_attention_from_v5(model, data, cycle)


def _discover_v3(cycle):
    """Fallback: use V3 transformer to discover candidate edges."""
    import torch
    from gnn.data.atlas_dataset import AtlasDataset
    from gnn.models.atlas_transformer_v3 import AtlasTransformerV3

    t0 = time.time()

    results_dir = os.path.join(RESULTS_DIR, "atlas_transformer_v3")
    results_json = os.path.join(results_dir, "results.json")
    model_path = os.path.join(results_dir, "fold_0", "best_model.pt")

    if not os.path.exists(model_path):
        return None

    with open(results_json) as f:
        results = json.load(f)
    config = results['config']

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = AtlasTransformerV3(config)
    model.load_state_dict(checkpoint)
    model.eval()

    ds = AtlasDataset()
    data = ds.build_features()

    # Compute graph features for V3
    from gnn.scripts.train_atlas_transformer_v3 import compute_graph_features
    from gnn.scripts.expanded_graph_scorer import (
        load_expanded_channel_map, fetch_string_expanded,
        build_expanded_graph, compute_cooccurrence,
    )
    from gnn.scripts.focused_multichannel_scorer import compute_curated_profiles
    from gnn.config import ANALYSIS_CACHE
    import networkx as nx

    expanded_cm = load_expanded_channel_map()
    expanded_genes = set(expanded_cm.keys())
    N = data['node_features'].shape[0]

    atlas_patient_ids = ds.clinical['patientId'].tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(atlas_patient_ids)}

    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False, usecols=["patientId", "gene.hugoGeneSymbol"],
    )
    mut_exp = mut[mut["gene.hugoGeneSymbol"].isin(expanded_genes)].copy()
    mut_exp["patient_idx"] = mut_exp["patientId"].map(pid_to_idx)
    mut_exp = mut_exp[mut_exp["patient_idx"].notna()]
    mut_exp["patient_idx"] = mut_exp["patient_idx"].astype(int)

    patient_genes_map = defaultdict(set)
    for _, row in mut_exp.iterrows():
        patient_genes_map[int(row["patient_idx"])].add(row["gene.hugoGeneSymbol"])

    ct_map = data['cancer_type_map']
    idx_to_ct = {v: k for k, v in ct_map.items()}
    ct_arr = data['cancer_types'].numpy()
    ct_per_patient = {i: idx_to_ct.get(int(ct_arr[i]), "Other") for i in range(N)}

    msk_genes = set()
    for genes in patient_genes_map.values():
        msk_genes |= genes
    msk_genes &= expanded_genes

    cooccurrence = compute_cooccurrence(patient_genes_map, ct_per_patient, min_count=10)
    ppi_edges = fetch_string_expanded(msk_genes)
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

    print(f"  V3 model loaded, data prepared [{time.time()-t0:.1f}s]", flush=True)

    return _extract_attention_from_v3(model, data, config, cycle)


def _extract_attention_from_v5(model, data, cycle):
    """Extract attention from V5 hierarchical transformer."""
    import torch

    gene_names = data['gene_names']
    n_patients = data['node_features'].shape[0]
    batch_size = 256  # smaller due to edge features memory

    gene_pair_sum = defaultdict(float)
    gene_pair_count = defaultdict(int)

    nf = data['node_features']
    nm = data['node_masks']
    cp = data['channel_pos_ids']
    ct = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    atlas = data['atlas_sums']
    clinical = torch.stack([ages, sexes], dim=-1)
    ef = data['edge_features']
    bi = data['block_ids']
    ci = data['channel_ids']

    n_ct = ct.max().item() + 1

    t0 = time.time()
    with torch.no_grad():
        for b_start in range(0, n_patients, batch_size):
            b_end = min(b_start + batch_size, n_patients)
            b_idx = slice(b_start, b_end)

            ct_clamped = ct[b_idx].clamp(max=n_ct - 1)
            _ = model(nf[b_idx], nm[b_idx], cp[b_idx], ct_clamped,
                      clinical[b_idx], atlas[b_idx],
                      ef[b_idx], bi[b_idx], ci[b_idx])

            # Get block-level attention from model
            attn = model._last_block_attn.cpu().numpy()
            masks = nm[b_idx].numpy()

            for i in range(attn.shape[0]):
                real_nodes = int(masks[i].sum())
                if real_nodes < 2:
                    continue

                patient_genes = gene_names[b_start + i][:real_nodes]
                attn_avg = attn[i, :, :real_nodes, :real_nodes].mean(axis=0)

                for qi in range(real_nodes):
                    for ki in range(real_nodes):
                        if qi == ki:
                            continue
                        g_from = patient_genes[qi]
                        g_to = patient_genes[ki]
                        if g_from and g_to and g_from != 'WT' and g_to != 'WT':
                            gene_pair_sum[(g_from, g_to)] += attn_avg[qi, ki]
                            gene_pair_count[(g_from, g_to)] += 1

            if (b_start // batch_size) % 20 == 0:
                print(f"  Batch {b_start//batch_size + 1}/"
                      f"{(n_patients + batch_size - 1)//batch_size}", flush=True)

    print(f"  V5 attention extracted [{time.time()-t0:.1f}s]", flush=True)
    return _build_candidates(gene_pair_sum, gene_pair_count, cycle, "v5")


def _extract_attention_from_v3(model, data, config, cycle):
    """Extract attention from V3 flat transformer."""
    import torch

    gene_names = data['gene_names']
    n_patients = data['node_features'].shape[0]
    batch_size = 512

    attn_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0]
            cancer_type = input[1]
            B, Nn, _ = x.shape
            q = module.q_proj(x).view(B, Nn, module.n_heads, module.head_dim).transpose(1, 2)
            k = module.k_proj(x).view(B, Nn, module.n_heads, module.head_dim).transpose(1, 2)

            ct_key = module.cancer_key_mod(cancer_type)
            ct_key = ct_key.view(B, 1, module.n_heads, module.head_dim).transpose(1, 2)
            k = k + ct_key

            if len(input) > 2 and input[2] is not None:
                g_key = module.graph_key_mod(input[2])
                g_key = g_key.view(B, 1, module.n_heads, module.head_dim).transpose(1, 2)
                k = k + g_key

            attn = torch.matmul(q, k.transpose(-2, -1)) / module.scale

            ct_bias = module.cancer_attn_bias(cancer_type)
            attn = attn + ct_bias.unsqueeze(-1).unsqueeze(-1)

            if len(input) > 2 and input[2] is not None:
                g_bias = module.graph_attn_bias(input[2])
                attn = attn + g_bias.unsqueeze(-1).unsqueeze(-1)

            if len(input) > 3 and input[3] is not None:
                attn = attn.masked_fill(input[3].unsqueeze(1).unsqueeze(2), float('-inf'))

            attn = torch.softmax(attn, dim=-1)
            attn_store[layer_idx] = attn.detach()
        return hook_fn

    for i, layer in enumerate(model.layers):
        layer.attn.register_forward_hook(make_hook(i))

    nf = data['node_features']
    nm = data['node_masks']
    cp = data['channel_pos_ids']
    ct = data['cancer_types']
    ages = data['ages']
    sexes = data['sexes']
    atlas = data['atlas_sums']
    clinical = torch.stack([ages, sexes], dim=-1)
    graph_features = data['graph_features']

    saved_n_ct = config['n_cancer_types']
    gene_pair_sum = defaultdict(float)
    gene_pair_count = defaultdict(int)

    t0 = time.time()
    with torch.no_grad():
        for b_start in range(0, n_patients, batch_size):
            b_end = min(b_start + batch_size, n_patients)
            b_idx = slice(b_start, b_end)

            ct_clamped = ct[b_idx].clamp(max=saved_n_ct - 1)
            cp_clamped = cp[b_idx].clamp(max=11)  # V3 trained with 6 channels (12 pos embeddings)
            out = model(nf[b_idx], nm[b_idx], cp_clamped, ct_clamped,
                        clinical[b_idx], atlas[b_idx], graph_features[b_idx])

            attn = attn_store[1].cpu().numpy()
            masks = nm[b_idx].numpy()

            for i in range(attn.shape[0]):
                real_nodes = int(masks[i].sum())
                if real_nodes < 2:
                    continue

                patient_genes = gene_names[b_start + i][:real_nodes]
                attn_avg = attn[i, :, :real_nodes, :real_nodes].mean(axis=0)

                for qi in range(real_nodes):
                    for ki in range(real_nodes):
                        if qi == ki:
                            continue
                        g_from = patient_genes[qi]
                        g_to = patient_genes[ki]
                        if g_from and g_to and g_from != 'WT' and g_to != 'WT':
                            gene_pair_sum[(g_from, g_to)] += attn_avg[qi, ki]
                            gene_pair_count[(g_from, g_to)] += 1

            if (b_start // batch_size) % 20 == 0:
                print(f"  Batch {b_start//batch_size + 1}/"
                      f"{(n_patients + batch_size - 1)//batch_size}", flush=True)

    print(f"  V3 attention extracted [{time.time()-t0:.1f}s]", flush=True)
    return _build_candidates(gene_pair_sum, gene_pair_count, cycle, "v3")


def _build_candidates(gene_pair_sum, gene_pair_count, cycle, model_version):
    """Build candidate edge list from aggregated attention."""
    candidates = []
    for (g_from, g_to), total in gene_pair_sum.items():
        n_obs = gene_pair_count[(g_from, g_to)]
        if n_obs < 100:
            continue
        mean_attn = total / n_obs
        if mean_attn < 0.05:
            continue

        ch_from = CHANNEL_MAP.get(g_from)
        ch_to = CHANNEL_MAP.get(g_to)

        candidates.append({
            'from': g_from,
            'to': g_to,
            'weight': round(float(mean_attn), 4),
            'n_obs': n_obs,
            'cross_channel': bool(ch_from != ch_to),
            'model': model_version,
        })

    print(f"  Candidate edges: {len(candidates)}", flush=True)

    os.makedirs(LOOP_DIR, exist_ok=True)
    cycle_dir = os.path.join(LOOP_DIR, f"cycle_{cycle}")
    os.makedirs(cycle_dir, exist_ok=True)
    pd.DataFrame(candidates).to_csv(
        os.path.join(cycle_dir, "candidate_edges.csv"), index=False)

    return candidates


def _v5_model_matches_schema(schema):
    """Check if the saved V5 model was trained on the current graph schema."""
    results_json = os.path.join(RESULTS_DIR, "atlas_transformer_v5", "results.json")
    if not os.path.exists(results_json):
        return False

    with open(results_json) as f:
        saved = json.load(f)

    saved_schema = saved.get('schema', {})
    return (saved_schema.get('edge_feature_dim') == schema.edge_feature_dim
            and saved_schema.get('node_extra_dim') == schema.node_extra_dim
            and saved_schema.get('n_channels') == schema.n_channels
            and saved_schema.get('n_blocks') == schema.n_blocks)


def _train_v5(schema):
    """Train V5 transformer on current graph topology."""
    import subprocess
    print("  Training V5 transformer on current graph topology...", flush=True)
    t0 = time.time()

    # cwd must be project root (parent of gnn/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, '-u', '-m', 'gnn.scripts.train_atlas_transformer_v5',
         '--epochs', '100', '--patience', '15', '--batch-size', '128',
         '--n-folds', '1'],  # single fold for speed in the loop
        capture_output=True, text=True, cwd=project_root,
    )

    if result.returncode != 0:
        print(f"  V5 training FAILED ({time.time()-t0:.0f}s):", flush=True)
        # Print last 20 lines of stderr
        for line in result.stderr.strip().split('\n')[-20:]:
            print(f"    {line}", flush=True)
        return False

    print(f"  V5 training complete [{time.time()-t0:.0f}s]", flush=True)
    # Print last 10 lines of stdout (summary)
    for line in result.stdout.strip().split('\n')[-10:]:
        print(f"    {line}", flush=True)
    return True


def discover(cycle, schema=None):
    """Run best available transformer, extract attention as candidate edges.

    If the V5 model doesn't exist or doesn't match the current schema,
    retrain it first. The graph changes each cycle — the transformer
    must see the current topology.
    """
    _banner("DISCOVERING NEW EDGES VIA TRANSFORMER", cycle)

    if schema is None:
        from gnn.data.graph_schema import get_schema
        schema = get_schema()

    # Check if V5 model exists AND matches current schema
    v5_path = os.path.join(RESULTS_DIR, "atlas_transformer_v5",
                            "fold_0", "best_model.pt")

    if os.path.exists(v5_path) and _v5_model_matches_schema(schema):
        print("  V5 model matches current schema", flush=True)
        return _discover_v5(cycle, schema)

    # V5 model is stale or missing — retrain
    if os.path.exists(v5_path):
        print("  V5 model schema mismatch — retraining on current graph", flush=True)
    else:
        print("  No V5 model — training on current graph", flush=True)

    if _train_v5(schema):
        return _discover_v5(cycle, schema)

    # Training failed — fall back to V3
    v3_path = os.path.join(RESULTS_DIR, "atlas_transformer_v3",
                            "fold_0", "best_model.pt")
    if os.path.exists(v3_path):
        print("  Falling back to V3 transformer", flush=True)
        return _discover_v3(cycle)

    print("  [SKIP] No transformer available.")
    return None


# =========================================================================
# STEP 3: Gate — test if candidates improve the score
# =========================================================================

def gate(scorer, candidates, baseline_c):
    """Test if candidate edges improve the graph scorer.

    Temporarily injects candidate ATTENDS_TO edges into the precomputed
    pairwise_edges dict, re-evaluates, and returns True if improvement
    exceeds threshold.
    """
    _banner("GATING — DO CANDIDATES IMPROVE THE SCORE?")

    if not candidates:
        print("  No candidates to test.")
        return False, 0.0

    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import StratifiedKFold

    # Temporarily inject candidates into precomputed pairwise_edges
    pw = scorer.gp.pairwise_edges
    originals = {}  # key → original entry (or None if didn't exist)
    injected = 0

    for cand in candidates:
        g1, g2 = cand['from'], cand['to']
        a, b = (g1, g2) if g1 < g2 else (g2, g1)
        key = f"{a}|{b}"

        # Save original
        if key not in originals:
            originals[key] = pw.get(key)

        # Inject: add ATTENDS_TO as multiplicative
        weight = float(cand['weight'])
        mult_increment = weight * 0.5  # matches ATTENDS_TO lambda in graph_precompute

        if key in pw:
            entry = pw[key]
            entry['multiplier'] = entry.get('multiplier', 0.0) + mult_increment
            entry['score'] = entry.get('score', 0.0) + mult_increment
            if 'ATTENDS_TO' not in entry.get('edge_types', []):
                entry.setdefault('edge_types', []).append('ATTENDS_TO')
        else:
            pw[key] = {
                'additive': 0.0,
                'multiplier': mult_increment,
                'score': mult_increment,
                'edge_types': ['ATTENDS_TO'],
                'cooccur_count': 0,
                'cooccur_by_ct': {},
            }
        injected += 1

    print(f"  Injected {injected} candidate edges into precomputed pairwise")

    # Re-evaluate
    scores, times, events, _ = scorer.score_all_patients()
    valid = times > 0
    s, t, e = scores[valid], times[valid], events[valid]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_cs = []
    for fold, (_, val_idx) in enumerate(skf.split(np.arange(len(e)), e)):
        try:
            c = concordance_index_censored(e[val_idx].astype(bool), t[val_idx], s[val_idx])[0]
        except Exception:
            c = 0.5
        fold_cs.append(c)

    new_c = float(np.mean(fold_cs))
    delta = new_c - baseline_c

    print(f"  Baseline C:  {baseline_c:.4f}")
    print(f"  With edges:  {new_c:.4f}")
    print(f"  Delta:       {delta:+.4f}")

    # Restore original pairwise_edges
    for key, orig in originals.items():
        if orig is None:
            pw.pop(key, None)
        else:
            pw[key] = orig

    accepted = delta > 0
    if accepted:
        print(f"\n  ACCEPTED — {delta:+.4f} improvement")
    else:
        print(f"\n  REJECTED — no improvement (delta={delta:+.4f})")

    return accepted, new_c


# =========================================================================
# STEP 4: Commit — write accepted edges through gateway
# =========================================================================

def commit(candidates, cycle):
    """Write accepted candidate edges to Neo4j through GraphGateway."""
    from gnn.data.graph_changelog import GraphGateway

    _banner("COMMITTING EDGES TO GRAPH", cycle)

    model_version = candidates[0].get('model', 'unknown') if candidates else 'unknown'

    gw = GraphGateway()
    try:
        before = gw.count_edges("ATTENDS_TO")
        print(f"  ATTENDS_TO before: {before:,}")

        edges = [{
            'from': c['from'],
            'to': c['to'],
            'weight': c['weight'],
            'n_obs': c['n_obs'],
            'cross_channel': c['cross_channel'],
            'cycle': cycle,
        } for c in candidates]

        n = gw.merge_edges("ATTENDS_TO", edges,
                           source=f"learning_loop_cycle_{cycle}",
                           source_detail=f"{len(edges)} edges, transformer_{model_version}")

        after = gw.count_edges("ATTENDS_TO")
        print(f"  ATTENDS_TO after: {after:,} (delta: {after - before:+,})")
    finally:
        gw.close()

    # Invalidate caches — graph has changed
    from gnn.data.graph_schema import GraphSchema
    schema = GraphSchema()
    schema.invalidate_cache()

    from gnn.data.graph_precompute import get_precomputed
    gp = get_precomputed()
    gp.invalidate()
    print("  Schema + precomputed caches invalidated (graph changed)")

    return n


# =========================================================================
# MAIN LOOP
# =========================================================================

def run_loop(max_cycles=5):
    """Run the learning loop."""
    from gnn.data.graph_changelog import should_snapshot, take_snapshot
    from gnn.data.graph_schema import get_schema

    print(f"\n{'#'*70}")
    print(f"  LEARNING LOOP — the graph learns, the transformer discovers")
    print(f"  Max cycles: {max_cycles}")
    print(f"{'#'*70}")

    os.makedirs(LOOP_DIR, exist_ok=True)
    history = []

    history_path = os.path.join(LOOP_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        print(f"\n  Resuming from cycle {len(history)}")

    start_cycle = len(history)

    for cycle in range(start_cycle, start_cycle + max_cycles):
        t_cycle = time.time()

        # Discover schema at start of each cycle — picks up new edge types
        schema = get_schema(force_refresh=True)
        print(f"\n  Schema: {len(schema.edge_types)} edge types, "
              f"{schema.edge_feature_dim} edge features, "
              f"{schema.node_extra_dim} node extra features")

        # 1. Measure baseline
        scorer, baseline_c, _, _, _, _ = measure()

        # 2. Discover candidate edges
        candidates = discover(cycle, schema=schema)
        if candidates is None or len(candidates) == 0:
            print("  No candidates discovered. Stopping.")
            break

        # 3. Gate — do candidates help?
        accepted, new_c = gate(scorer, candidates, baseline_c)

        # 4. Commit if accepted
        n_committed = 0
        if accepted:
            n_committed = commit(candidates, cycle)

        # Record
        entry = {
            'cycle': cycle,
            'baseline_c': baseline_c,
            'new_c': new_c,
            'delta': new_c - baseline_c,
            'accepted': accepted,
            'n_candidates': len(candidates),
            'n_committed': n_committed,
            'n_edge_types': len(schema.edge_types),
            'edge_feature_dim': schema.edge_feature_dim,
            'time_s': time.time() - t_cycle,
        }
        history.append(entry)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  CYCLE {cycle} SUMMARY")
        print(f"{'='*70}")
        print(f"  Baseline C:   {baseline_c:.4f}")
        print(f"  New C:        {new_c:.4f} ({new_c - baseline_c:+.4f})")
        print(f"  Accepted:     {accepted}")
        print(f"  Candidates:   {len(candidates)}")
        print(f"  Committed:    {n_committed}")
        print(f"  Edge types:   {len(schema.edge_types)}")
        print(f"  Time:         {entry['time_s']:.1f}s")

        if len(history) > 1:
            print(f"\n  History:")
            for h in history:
                status = "ACCEPTED" if h['accepted'] else "rejected"
                print(f"    Cycle {h['cycle']}: {h['baseline_c']:.4f} → "
                      f"{h['new_c']:.4f} ({h['delta']:+.4f}) [{status}]")

        # Snapshot if WAL is getting large
        if should_snapshot(threshold=500):
            take_snapshot(label=f"after_cycle_{cycle}")

        # Stop if rejected (no more signal to extract)
        if not accepted:
            print(f"\n  No improvement — graph has absorbed available signal.")
            break

    print(f"\n{'#'*70}")
    print(f"  LEARNING LOOP COMPLETE — {len(history)} cycles")
    print(f"{'#'*70}")
    return history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cycles', type=int, default=5)
    parser.add_argument('--measure-only', action='store_true')
    args = parser.parse_args()

    if args.measure_only:
        measure()
    else:
        run_loop(args.max_cycles)


if __name__ == "__main__":
    main()
