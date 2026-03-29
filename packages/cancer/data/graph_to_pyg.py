"""
Neo4j → PyTorch Geometric HeteroData export.

Exports the full graph structure for GNN training:
  Node types: Patient, Gene, MutationGroup
  Edge types: HAS_MUTATION, COOCCURS, PPI, ATTENDS_TO,
              MEMBER_OF, HAS_GENE, COUPLES, GROUP_ATTENDS_TO

Patient nodes carry survival labels (OS_MONTHS, event).
Gene nodes carry features (channel onehot, function, self_attn, etc).
Edge features carry weights (log_hr, attention weight, jaccard, etc).

Usage:
    python3 -u -m gnn.data.graph_to_pyg          # build and save
    python3 -u -m gnn.data.graph_to_pyg --check   # load and inspect
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neo4j import GraphDatabase
from torch_geometric.data import HeteroData

from gnn.config import (
    CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_TO_IDX,
    HUB_GENES, GENE_FUNCTION, GNN_CACHE,
)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")
SAVE_PATH = os.path.join(GNN_CACHE, "hetero_graph.pt")

_HUB_SET = set()
for hubs in HUB_GENES.values():
    _HUB_SET.update(hubs)

FUNC_MAP = {'GOF': 0, 'LOF': 1, 'context': 2}


def _step(name, actual=None, elapsed=None):
    status = f"{actual:,}" if actual is not None else ""
    t = f"[{elapsed:.1f}s]" if elapsed is not None else ""
    print(f"  {name:.<50s} {status:>15s}  {t}", flush=True)


def build_hetero_graph():
    """Export Neo4j graph to PyG HeteroData."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    data = HeteroData()

    print(f"\n{'='*70}")
    print("  EXPORTING NEO4J → PyG HeteroData")
    print(f"{'='*70}\n")
    t_total = time.time()

    # =============================================
    # 1. GENE NODES
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (g:Gene)
            RETURN g.name AS name, g.channel AS channel,
                   g.primary_channel AS primary_channel,
                   g.self_attn AS self_attn,
                   g.function AS function,
                   g.position AS position,
                   g.is_hub AS is_hub,
                   g.profile_entropy AS entropy
        """)
        genes = list(result)

    gene_idx = {}  # name -> index
    gene_feats = []
    for i, g in enumerate(genes):
        gene_idx[g['name']] = i

        feat = np.zeros(len(CHANNEL_NAMES) + 5, dtype=np.float32)
        # Channel one-hot (6 dims)
        ch = CHANNEL_MAP.get(g['name'])
        if ch and ch in CHANNEL_NAMES:
            feat[CHANNEL_NAMES.index(ch)] = 1.0

        # Function encoding
        func = GENE_FUNCTION.get(g['name'], 'context')
        feat[len(CHANNEL_NAMES)] = FUNC_MAP.get(func, 2) / 2.0

        # Hub flag
        feat[len(CHANNEL_NAMES) + 1] = 1.0 if g['name'] in _HUB_SET else 0.0

        # Self-attention
        feat[len(CHANNEL_NAMES) + 2] = g['self_attn'] or 0.0

        # Entropy
        feat[len(CHANNEL_NAMES) + 3] = g['entropy'] or 0.0

        # Position (centrality proxy)
        pos = g['position'] or 'leaf'
        feat[len(CHANNEL_NAMES) + 4] = 1.0 if pos == 'hub' else 0.0

        gene_feats.append(feat)

    data['gene'].x = torch.tensor(np.stack(gene_feats), dtype=torch.float32)
    data['gene'].name = [g['name'] for g in genes]
    _step("Gene nodes", actual=len(genes), elapsed=time.time() - t0)

    # =============================================
    # 2. PATIENT NODES
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (p:Patient)
            RETURN p.id AS pid, p.cancer_type AS cancer_type,
                   p.os_months AS os_months, p.event AS event,
                   p.tissue_delta AS tissue_delta,
                   p.mutation_key AS mutation_key
        """)
        patients = list(result)

    patient_idx = {}
    patient_feats = []
    patient_labels = []
    cancer_type_map = {}
    ct_idx = 0

    for i, p in enumerate(patients):
        patient_idx[p['pid']] = i

        ct = p['cancer_type'] or 'Unknown'
        if ct not in cancer_type_map:
            cancer_type_map[ct] = ct_idx
            ct_idx += 1

        feat = np.zeros(3, dtype=np.float32)
        feat[0] = cancer_type_map[ct] / max(ct_idx, 1)  # normalized cancer type
        feat[1] = (p['tissue_delta'] or 0.0)
        feat[2] = 1.0  # bias term

        patient_feats.append(feat)
        patient_labels.append({
            'os_months': p['os_months'] or 0.0,
            'event': p['event'] or 0,
        })

    data['patient'].x = torch.tensor(np.stack(patient_feats), dtype=torch.float32)
    data['patient'].os_months = torch.tensor(
        [l['os_months'] for l in patient_labels], dtype=torch.float32)
    data['patient'].event = torch.tensor(
        [l['event'] for l in patient_labels], dtype=torch.long)
    data['patient'].pid = [p['pid'] for p in patients]
    data['patient'].cancer_type_idx = torch.tensor(
        [cancer_type_map[p['cancer_type'] or 'Unknown'] for p in patients],
        dtype=torch.long)

    _step("Patient nodes", actual=len(patients), elapsed=time.time() - t0)
    print(f"    Cancer types: {ct_idx}")

    # =============================================
    # 3. MUTATION GROUP NODES
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (mg:MutationGroup)
            RETURN mg.mutation_key AS key, mg.n_patients AS n_patients,
                   mg.n_events AS n_events, mg.median_os AS median_os,
                   size(mg.gene_list) AS n_genes
        """)
        groups = list(result)

    group_idx = {}
    group_feats = []
    for i, g in enumerate(groups):
        group_idx[g['key']] = i
        feat = np.zeros(4, dtype=np.float32)
        feat[0] = min((g['n_patients'] or 0) / 100.0, 5.0)  # capped log-ish
        feat[1] = (g['n_events'] or 0) / max(g['n_patients'] or 1, 1)  # event_rate
        feat[2] = (g['median_os'] or 0) / 60.0  # normalize to ~5yr
        feat[3] = (g['n_genes'] or 0) / 10.0
        # NOTE: feat[1] (event_rate) and feat[2] (median_os) are recomputed
        # per fold in cox_sage.py to prevent target leakage
        group_feats.append(feat)

    data['mutation_group'].x = torch.tensor(np.stack(group_feats), dtype=torch.float32)
    _step("MutationGroup nodes", actual=len(groups), elapsed=time.time() - t0)

    # =============================================
    # 4. EDGES: HAS_MUTATION (Patient -> Gene)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (p:Patient)-[m:HAS_MUTATION]->(g:Gene)
            RETURN p.id AS pid, g.name AS gene,
                   m.log_hr AS log_hr, m.direction AS direction
        """)
        mut_edges = list(result)

    src, dst, edge_feat = [], [], []
    for e in mut_edges:
        pi = patient_idx.get(e['pid'])
        gi = gene_idx.get(e['gene'])
        if pi is not None and gi is not None:
            src.append(pi)
            dst.append(gi)
            d = FUNC_MAP.get(e['direction'] or 'context', 2) / 2.0
            edge_feat.append([e['log_hr'] or 0.0, d])

    data['patient', 'has_mutation', 'gene'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['patient', 'has_mutation', 'gene'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)
    _step("HAS_MUTATION edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 5. EDGES: COOCCURS (Gene -> Gene)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (a:Gene)-[r:COOCCURS]->(b:Gene)
            RETURN a.name AS from, b.name AS to,
                   r.count AS count
        """)
        cooc_edges = list(result)

    src, dst, edge_feat = [], [], []
    for e in cooc_edges:
        ai = gene_idx.get(e['from'])
        bi = gene_idx.get(e['to'])
        if ai is not None and bi is not None:
            src.append(ai)
            dst.append(bi)
            edge_feat.append([np.log1p(e['count'] or 0)])

    data['gene', 'cooccurs', 'gene'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['gene', 'cooccurs', 'gene'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)
    _step("COOCCURS edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 6. EDGES: PPI (Gene -> Gene)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (a:Gene)-[r:PPI]->(b:Gene)
            RETURN a.name AS from, b.name AS to, r.score AS score
        """)
        ppi_edges = list(result)

    src, dst, edge_feat = [], [], []
    for e in ppi_edges:
        ai = gene_idx.get(e['from'])
        bi = gene_idx.get(e['to'])
        if ai is not None and bi is not None:
            src.append(ai)
            dst.append(bi)
            edge_feat.append([e['score'] or 0.0])

    data['gene', 'ppi', 'gene'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['gene', 'ppi', 'gene'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)
    _step("PPI edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 7. EDGES: ATTENDS_TO (Gene -> Gene, directed)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (a:Gene)-[r:ATTENDS_TO]->(b:Gene)
            RETURN a.name AS from, b.name AS to,
                   r.weight AS weight, r.asymmetry AS asymmetry,
                   coalesce(r.self_loop, false) AS self_loop
        """)
        attn_edges = list(result)

    src, dst, edge_feat = [], [], []
    for e in attn_edges:
        ai = gene_idx.get(e['from'])
        bi = gene_idx.get(e['to'])
        if ai is not None and bi is not None:
            src.append(ai)
            dst.append(bi)
            edge_feat.append([
                e['weight'] or 0.0,
                e['asymmetry'] or 1.0,
                1.0 if e['self_loop'] else 0.0,
            ])

    data['gene', 'attends_to', 'gene'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['gene', 'attends_to', 'gene'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)
    _step("ATTENDS_TO edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 8. EDGES: MEMBER_OF (Patient -> MutationGroup)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (p:Patient)-[:MEMBER_OF]->(mg:MutationGroup)
            RETURN p.id AS pid, mg.mutation_key AS key
        """)
        member_edges = list(result)

    src, dst = [], []
    for e in member_edges:
        pi = patient_idx.get(e['pid'])
        gi = group_idx.get(e['key'])
        if pi is not None and gi is not None:
            src.append(pi)
            dst.append(gi)

    data['patient', 'member_of', 'mutation_group'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    _step("MEMBER_OF edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 9. EDGES: HAS_GENE (MutationGroup -> Gene)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (mg:MutationGroup)-[r:HAS_GENE]->(g:Gene)
            RETURN mg.mutation_key AS key, g.name AS gene,
                   r.direction AS direction
        """)
        hg_edges = list(result)

    src, dst, edge_feat = [], [], []
    for e in hg_edges:
        gi = group_idx.get(e['key'])
        gn = gene_idx.get(e['gene'])
        if gi is not None and gn is not None:
            src.append(gi)
            dst.append(gn)
            d = FUNC_MAP.get(e['direction'] or 'context', 2) / 2.0
            edge_feat.append([d])

    data['mutation_group', 'has_gene', 'gene'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['mutation_group', 'has_gene', 'gene'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)
    _step("HAS_GENE edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 10. EDGES: GROUP_ATTENDS_TO (MutationGroup -> MutationGroup)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (a:MutationGroup)-[r:GROUP_ATTENDS_TO]->(b:MutationGroup)
            RETURN a.mutation_key AS from, b.mutation_key AS to,
                   r.weight AS weight
        """)
        ga_edges = list(result)

    src, dst, edge_feat = [], [], []
    for e in ga_edges:
        ai = group_idx.get(e['from'])
        bi = group_idx.get(e['to'])
        if ai is not None and bi is not None:
            src.append(ai)
            dst.append(bi)
            edge_feat.append([e['weight'] or 0.0])

    data['mutation_group', 'group_attends_to', 'mutation_group'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['mutation_group', 'group_attends_to', 'mutation_group'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)
    _step("GROUP_ATTENDS_TO edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 11. EDGES: COUPLES (Gene -> Gene)
    # =============================================
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (a:Gene)-[r:COUPLES]->(b:Gene)
            RETURN a.name AS from, b.name AS to,
                   r.locality AS locality, r.pair_type AS pair_type
        """)
        couple_edges = list(result)

    loc_map = {'within': 0.0, 'cross': 1.0}
    pt_map = {'GOF_LOF': 0.0, 'GOF_GOF': 0.33, 'LOF_LOF': 0.67, 'other': 1.0}

    src, dst, edge_feat = [], [], []
    for e in couple_edges:
        ai = gene_idx.get(e['from'])
        bi = gene_idx.get(e['to'])
        if ai is not None and bi is not None:
            src.append(ai)
            dst.append(bi)
            edge_feat.append([
                loc_map.get(e['locality'], 0.5),
                pt_map.get(e['pair_type'], 1.0),
            ])

    data['gene', 'couples', 'gene'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['gene', 'couples', 'gene'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)
    _step("COUPLES edges", actual=len(src), elapsed=time.time() - t0)

    # =============================================
    # 12. REVERSE EDGES (so patient/group nodes receive messages)
    # =============================================
    t0 = time.time()

    # Gene -> Patient (reverse of HAS_MUTATION)
    hm = data['patient', 'has_mutation', 'gene'].edge_index
    data['gene', 'rev_has_mutation', 'patient'].edge_index = torch.stack([hm[1], hm[0]])
    data['gene', 'rev_has_mutation', 'patient'].edge_attr = \
        data['patient', 'has_mutation', 'gene'].edge_attr.clone()

    # MutationGroup -> Patient (reverse of MEMBER_OF)
    mo = data['patient', 'member_of', 'mutation_group'].edge_index
    data['mutation_group', 'rev_member_of', 'patient'].edge_index = torch.stack([mo[1], mo[0]])

    # Gene -> MutationGroup (reverse of HAS_GENE)
    hg = data['mutation_group', 'has_gene', 'gene'].edge_index
    data['gene', 'rev_has_gene', 'mutation_group'].edge_index = torch.stack([hg[1], hg[0]])
    data['gene', 'rev_has_gene', 'mutation_group'].edge_attr = \
        data['mutation_group', 'has_gene', 'gene'].edge_attr.clone()

    n_rev = (hm.shape[1] + mo.shape[1] + hg.shape[1])
    _step("Reverse edges (3 types)", actual=n_rev, elapsed=time.time() - t0)

    # =============================================
    # SUMMARY
    # =============================================
    print(f"\n{'='*70}")
    print("  GRAPH SUMMARY")
    print(f"{'='*70}\n")

    for ntype in data.node_types:
        print(f"  {ntype:20s} nodes: {data[ntype].x.shape[0]:>8,}  "
              f"features: {data[ntype].x.shape[1]}")

    print()
    for etype in data.edge_types:
        ei = data[etype].edge_index
        n_edges = ei.shape[1]
        has_attr = hasattr(data[etype], 'edge_attr') and data[etype].edge_attr is not None
        attr_dim = data[etype].edge_attr.shape[1] if has_attr else 0
        label = f"({etype[0]})-[{etype[1]}]->({etype[2]})"
        print(f"  {label:55s} edges: {n_edges:>8,}  attr: {attr_dim}")

    total_nodes = sum(data[nt].x.shape[0] for nt in data.node_types)
    total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    print(f"\n  Total: {total_nodes:,} nodes, {total_edges:,} edges")

    # Store metadata
    data.cancer_type_map = cancer_type_map
    data.gene_idx = gene_idx
    data.patient_idx = patient_idx
    data.group_idx = group_idx

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(data, SAVE_PATH)
    print(f"\n  Saved to {SAVE_PATH}")
    print(f"  Total export time: {time.time() - t_total:.1f}s")

    driver.close()
    return data


def check_graph():
    """Load and inspect saved graph."""
    data = torch.load(SAVE_PATH, weights_only=False)

    print(f"\n{'='*70}")
    print("  LOADED PyG HeteroData")
    print(f"{'='*70}\n")

    for ntype in data.node_types:
        print(f"  {ntype:20s} nodes: {data[ntype].x.shape[0]:>8,}  "
              f"features: {data[ntype].x.shape[1]}")

    print()
    for etype in data.edge_types:
        ei = data[etype].edge_index
        n_edges = ei.shape[1]
        label = f"({etype[0]})-[{etype[1]}]->({etype[2]})"
        print(f"  {label:55s} edges: {n_edges:>8,}")

    # Check patient labels
    n_events = data['patient'].event.sum().item()
    mean_os = data['patient'].os_months.mean().item()
    print(f"\n  Patient labels: {n_events:,} events, "
          f"mean OS {mean_os:.1f} months")

    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    if args.check:
        check_graph()
    else:
        build_hetero_graph()


if __name__ == "__main__":
    main()
