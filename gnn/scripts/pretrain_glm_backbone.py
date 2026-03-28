"""
Multi-task mechanistic pretraining for GLM backbone.

Architecture: 8 mechanistic edge-type predictors + COOCCURS as holdback exam.

COOCCURS is the union of all edge types. Each co-occurrence between two genes
should be explainable by one or more of the 8 mechanistic edge types:

  POSITIVE (attract — co-occur):
    1. PPI          — physical binding (STRING)
    2. TRANSPOSES   — same copy, different address (paralogs)
    3. BELONGS_TO   — same channel membership (channel co-damage)
    4. ANALOGOUS    — same edit mechanism (mutation type profiles)
    5. COUPLES      — functionally coupled (expression correlation)

  NEGATIVE (repel — mutually exclusive):
    6. SL_PARTNER   — compensatory, lethal if both lost (CRISPR)
    7. CONVERGES    — same delete at different addresses (ME)

  DIRECTIONAL:
    8. ENABLES      — removing A unleashes B (asymmetric lift)

  NOVEL:
    9. UNIQUE       — brand new mutation type, genuinely novel biology

  CO-DEPENDENCY:
   10. CO_ESSENTIAL — CRISPR co-dependency (DepMap), distinct from expression coupling

  RESIDUAL:
    UNKNOWN         — co-occurrences not explained by any of the 10 = needs investigation

COOCCURS accuracy measures how well the 10 explain all co-mutation patterns.
Like OS is for survival — the composite exam, not a training signal.

Usage:
    python3 -u -m gnn.scripts.pretrain_glm_backbone [--epochs 100]
    python3 -u -m gnn.scripts.pretrain_glm_backbone --quick  # fast test run
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, NODE_FEAT_DIM, MAX_NODES
from gnn.models.atlas_transformer_v6 import IntraBlockLayer, BlockToChannelPool
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, HUB_GENES, GNN_CACHE
from gnn.data.block_assignments import load_block_assignments

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "glm_pretrain",
)


# =========================================================================
# Backbone model (same layers that transfer to GLM)
# =========================================================================

class PretrainBackbone(nn.Module):
    """Backbone for mechanistic pretraining.

    Same architecture as GLM Levels 0-2:
      - Node encoder
      - Intra-block attention
      - Block → Channel pooling

    Plus task-specific heads for each pretraining objective.
    """

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']
        hidden = config['hidden_dim']
        dropout = config.get('dropout', 0.1)
        n_cancer_types = config['n_cancer_types']
        n_heads = config.get('n_heads', 4)
        n_intra_layers = config.get('n_intra_layers', 2)
        edge_feat_dim = config['edge_feat_dim']
        n_channels = config['n_channels']

        self.hidden = hidden
        self.n_channels = n_channels

        # === Backbone (transfers to GLM) ===
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.ct_sign = nn.Embedding(n_cancer_types, 1)
        nn.init.ones_(self.ct_sign.weight)

        self.age_mod = nn.Sequential(nn.Linear(1, hidden), nn.ELU())

        self.intra_block_layers = nn.ModuleList([
            IntraBlockLayer(hidden, n_heads, n_cancer_types, edge_feat_dim, dropout)
            for _ in range(n_intra_layers)
        ])

        self.block_to_channel = BlockToChannelPool(
            hidden, n_cancer_types, n_channels, dropout
        )

        # === Cross-channel attention (transfers to GLM router) ===
        self.cross_channel = CrossChannelLight(hidden, n_cancer_types, n_channels, dropout)

        # Channel pooling
        self.channel_pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # Clinical (minimal — just for forward compatibility)
        self.sex_encoder = nn.Linear(1, hidden // 4)
        self.atlas_skip = nn.Sequential(nn.Linear(1, hidden // 4), nn.ELU())
        self.temporal_encoder = nn.Sequential(nn.Linear(3, hidden // 4), nn.ELU())

        # === Task heads ===

        # Co-mutation: holdback exam, not trained — just evaluated
        self.comut_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, config['n_genes']),
        )

        # Channel co-damage: BELONGS_TO validation
        self.channel_damage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, n_channels),
        )

        # Per-edge-type gene pair projections
        # Each edge type gets its own projection so it can learn different
        # aspects of gene similarity/dissimilarity
        self.edge_type_names = [
            'PPI', 'TRANSPOSES', 'BELONGS_TO', 'ANALOGOUS',
            'COUPLES', 'SL_PARTNER', 'CONVERGES', 'ENABLES', 'UNIQUE',
            'CO_ESSENTIAL', 'UNKNOWN',
        ]
        self.edge_projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(hidden, hidden), nn.ELU())
            for name in self.edge_type_names
        })

        # Shared gene embedding (used by all edge-type heads)
        self.gene_embed_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )

        readout_in = hidden + hidden // 4 * 3
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_features, node_mask, cancer_type, clinical,
                atlas_sum, edge_features, block_ids, channel_ids):
        B, N, _ = node_features.shape
        pad_mask = (node_mask == 0)

        # Encode
        x = self.node_encoder(node_features)
        age = clinical[:, 0:1]
        age_scale = self.age_mod(age)
        x = x * (1 + age_scale.unsqueeze(1) * 0.1)

        # Intra-block attention
        block_mask = (block_ids.unsqueeze(1) != block_ids.unsqueeze(2))
        for layer in self.intra_block_layers:
            x, _ = layer(x, cancer_type, edge_features, block_mask, pad_mask)

        # Store mutation embeddings for gene-level tasks
        self._mutation_embeddings = x  # (B, N, hidden)

        # Block → Channel
        channel_tokens, channel_active = self.block_to_channel(
            x, cancer_type, channel_ids, pad_mask
        )

        # Cross-channel
        n_active = channel_active.float().sum(dim=1)
        if (n_active >= 2).any():
            channel_tokens = self.cross_channel(
                channel_tokens, cancer_type, channel_active
            )

        # Pool → patient embedding
        ch_attn = self.channel_pool_attn(channel_tokens).squeeze(-1)
        ch_attn = ch_attn.masked_fill(~channel_active, float('-inf'))
        ch_attn = F.softmax(ch_attn, dim=1).nan_to_num(0.0)
        patient_embed = (channel_tokens * ch_attn.unsqueeze(-1)).sum(dim=1)

        self._patient_embed = patient_embed
        self._channel_tokens = channel_tokens
        self._channel_active = channel_active

        return patient_embed

    def comut_logits(self):
        """Predict co-mutation probabilities from patient embedding."""
        return self.comut_head(self._patient_embed)

    def channel_damage_logits(self):
        """Predict channel activation pattern from patient embedding."""
        return self.channel_damage_head(self._patient_embed)

    def gene_pair_score(self, gene_i_embed, gene_j_embed, edge_type=None):
        """Score a gene pair for link prediction.

        If edge_type is specified, uses that edge type's projection head.
        Otherwise uses the shared gene_embed_proj.
        """
        if edge_type and edge_type in self.edge_projections:
            proj = self.edge_projections[edge_type]
            hi = proj(gene_i_embed)
            hj = proj(gene_j_embed)
        else:
            hi = self.gene_embed_proj(gene_i_embed)
            hj = self.gene_embed_proj(gene_j_embed)
        return (hi * hj).sum(dim=-1)


class CrossChannelLight(nn.Module):
    """Lightweight cross-channel interaction (same keys transfer to GLM router)."""

    def __init__(self, hidden, n_cancer_types, n_channels, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_channels = n_channels

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        self.ct_channel_bias = nn.Embedding(n_cancer_types, n_channels * n_channels)
        self.ct_key_mod = nn.Embedding(n_cancer_types, hidden)

        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, channel_tokens, cancer_type, channel_active):
        B, C, _ = channel_tokens.shape
        normed = self.norm(channel_tokens)

        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        ct_key = self.ct_key_mod(cancer_type).unsqueeze(1)
        k = k + ct_key

        attn = torch.bmm(q, k.transpose(1, 2)) / (self.hidden ** 0.5)

        ct_bias = self.ct_channel_bias(cancer_type).view(B, C, C)
        attn = attn + ct_bias * 0.1

        inactive = ~channel_active
        attn = attn.masked_fill(inactive.unsqueeze(1), float('-inf'))
        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        out = self.out_proj(out)

        channel_tokens = channel_tokens + out
        channel_tokens = channel_tokens + self.ffn(self.norm2(channel_tokens))
        return channel_tokens


# =========================================================================
# Pretraining data extraction
# =========================================================================

def load_pretraining_targets(gene_vocab, channel_idx):
    """Load all 10 pretraining targets from Neo4j.

    8 mechanistic edge types (training signal):
      + PPI, TRANSPOSES, BELONGS_TO, ANALOGOUS, COUPLES (positive/attract)
      - SL_PARTNER, CONVERGES (negative/repel)
      +/- ENABLES (directional)

    1 residual (UNKNOWN — unexplained co-occurrences, needs investigation)
    1 composite holdback (COOCCURS — the exam, not a training signal)

    Returns dict with pair lists for each edge type + comut matrix.
    """
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    n_genes = len(gene_vocab)
    idx_lookup = gene_vocab

    print("  Loading pretraining targets...", flush=True)
    t0 = time.time()

    # ---- COOCCURS (holdback exam — not used in loss) ----
    comut = np.zeros((n_genes, n_genes), dtype=np.float32)
    with driver.session() as s:
        result = s.run("""
            MATCH (g1:Gene)-[r:COOCCURS]-(g2:Gene)
            WHERE g1.name < g2.name
            RETURN g1.name AS g1, g2.name AS g2, r.count AS cnt
        """)
        n_comut = 0
        for rec in result:
            i = idx_lookup.get(rec['g1'])
            j = idx_lookup.get(rec['g2'])
            if i is not None and j is not None:
                val = np.log1p(rec['cnt'] or 0)
                comut[i, j] = comut[j, i] = val
                n_comut += 1
    print(f"    COOCCURS (holdback): {n_comut:,} pairs", flush=True)

    # ---- Edge type pair lists ----
    # sign is the TYPE DEFAULT — per-edge sign comes from correlation data when available
    # sign=0 means "direction varies per instance, read from edge properties"
    EDGE_TYPES = [
        {"name": "PPI",          "sign": +1, "rels": ["PPI"],                             "has_score": True},
        {"name": "TRANSPOSES",   "sign":  0, "rels": ["TRANSPOSES"],                      "has_score": False},
        {"name": "BELONGS_TO",   "sign": +1, "rels": ["BELONGS_TO"],                      "has_score": False},
        {"name": "ANALOGOUS",    "sign":  0, "rels": ["ANALOGOUS"],                       "has_score": False},
        {"name": "COUPLES",      "sign":  0, "rels": ["COUPLES", "CO_EXPRESSED", "CO_CNA"], "has_score": False},
        {"name": "SL_PARTNER",   "sign": -1, "rels": ["SL_PARTNER", "SYNTHETIC_LETHAL"],  "has_score": False},
        {"name": "CONVERGES",    "sign": -1, "rels": ["CONVERGES"],                       "has_score": False},
        {"name": "ENABLES",      "sign":  0, "rels": ["ENABLES", "ATTENDS_TO"],           "has_score": False},
        {"name": "UNIQUE",       "sign":  0, "rels": ["UNIQUE"],                          "has_score": False},
        {"name": "CO_ESSENTIAL", "sign":  0, "rels": ["CO_ESSENTIAL"],                    "has_score": True},
    ]

    edge_data = {}
    explained_pairs = set()  # track which gene pairs are explained

    for et in EDGE_TYPES:
        pairs = []
        name = et["name"]
        rels = et["rels"]

        if name == "BELONGS_TO":
            # Derived from CHANNEL_MAP, not a Neo4j Gene→Gene edge
            from gnn.config import CHANNEL_MAP
            from collections import defaultdict
            ch_genes = defaultdict(list)
            for gene, ch in CHANNEL_MAP.items():
                gi = idx_lookup.get(gene)
                if gi is not None:
                    ch_genes[ch].append(gi)
            for ch, gene_ids in ch_genes.items():
                for a in range(len(gene_ids)):
                    for b in range(a + 1, len(gene_ids)):
                        pairs.append((gene_ids[a], gene_ids[b]))
                        explained_pairs.add(
                            (min(gene_ids[a], gene_ids[b]),
                             max(gene_ids[a], gene_ids[b])))
        else:
            # Query all Neo4j rels that map to this canonical type
            for rel in rels:
                with driver.session() as s:
                    if et["has_score"]:
                        # Score property name varies: PPI uses 'score', CO_ESSENTIAL uses 'correlation'
                        score_prop = "correlation" if rel == "CO_ESSENTIAL" else "score"
                        result = s.run(f"""
                            MATCH (g1:Gene)-[r:{rel}]-(g2:Gene)
                            WHERE g1.name < g2.name
                            RETURN g1.name AS g1, g2.name AS g2,
                                   r.{score_prop} AS score
                        """)
                        for rec in result:
                            i = idx_lookup.get(rec['g1'])
                            j = idx_lookup.get(rec['g2'])
                            if i is not None and j is not None:
                                pairs.append((i, j, float(rec['score'] or 0.5)))
                                explained_pairs.add((min(i, j), max(i, j)))
                    else:
                        result = s.run(f"""
                            MATCH (g1:Gene)-[r:{rel}]-(g2:Gene)
                            WHERE g1.name < g2.name
                            RETURN g1.name AS g1, g2.name AS g2
                        """)
                        for rec in result:
                            i = idx_lookup.get(rec['g1'])
                            j = idx_lookup.get(rec['g2'])
                            if i is not None and j is not None:
                                pairs.append((i, j))
                                explained_pairs.add((min(i, j), max(i, j)))

        edge_data[name] = {
            "pairs": pairs,
            "sign": et["sign"],
        }
        rels_str = "+".join(rels) if rels else "CHANNEL_MAP"
        sign_str = '+' if et['sign'] > 0 else ('-' if et['sign'] < 0 else '?')
        print(f"    {name:>15s}: {len(pairs):>6,} pairs  (sign={sign_str}) ← {rels_str}",
              flush=True)

    # ---- UNKNOWN residual: COOCCURS pairs not explained by any of the 9 ----
    all_comut_pairs = set()
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if comut[i, j] > 0:
                all_comut_pairs.add((i, j))

    unknown_pairs = all_comut_pairs - explained_pairs
    edge_data["UNKNOWN"] = {
        "pairs": list(unknown_pairs),
        "sign": 0,  # unknown sign — needs investigation
    }

    pct_explained = 100 * len(explained_pairs & all_comut_pairs) / max(len(all_comut_pairs), 1)
    print(f"    {'UNKNOWN':>15s}: {len(unknown_pairs):>6,} unexplained co-occurrence pairs", flush=True)
    print(f"    COOCCURS decomposition: {pct_explained:.1f}% explained by 9 edge types", flush=True)

    # ---- Channel labels per gene ----
    channel_labels = np.zeros((n_genes, len(channel_idx)), dtype=np.float32)
    for gene, ch in CHANNEL_MAP.items():
        gi = idx_lookup.get(gene)
        ci = channel_idx.get(ch)
        if gi is not None and ci is not None:
            channel_labels[gi, ci] = 1.0

    driver.close()
    print(f"    Targets loaded [{time.time()-t0:.1f}s]", flush=True)

    return {
        'comut_matrix': comut,
        'edge_data': edge_data,
        'channel_labels': channel_labels,
        'n_explained': len(explained_pairs & all_comut_pairs),
        'n_total_comut': len(all_comut_pairs),
        'n_novel': len(novel_pairs),
    }


def build_patient_channel_targets(data, channel_idx):
    """For each patient, build channel co-damage target vector.

    Target: binary vector of which channels have mutations.
    """
    n_patients = len(data['gene_names'])
    n_channels = len(channel_idx)
    targets = np.zeros((n_patients, n_channels), dtype=np.float32)

    for p_idx, genes in enumerate(data['gene_names']):
        for g in genes:
            if g and g != '' and g != 'WT':
                ch = CHANNEL_MAP.get(g)
                ci = channel_idx.get(ch)
                if ci is not None:
                    targets[p_idx, ci] = 1.0

    return targets


# =========================================================================
# Loss functions
# =========================================================================

def comut_loss(logits, gene_indices, node_mask, comut_matrix, gene_vocab_size):
    """Masked co-mutation prediction loss.

    For each patient, mask one mutation and predict it from the rest.
    Uses the population co-mutation matrix as soft target.
    """
    B, N = gene_indices.shape
    device = logits.device

    # Pick a random real mutation to mask per patient
    real_mask = (node_mask > 0)
    n_real = real_mask.float().sum(dim=1)  # (B,)

    loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for b in range(B):
        real_idx = real_mask[b].nonzero(as_tuple=True)[0]
        if len(real_idx) < 2:
            continue

        # Random held-out gene
        held_out_pos = real_idx[torch.randint(len(real_idx), (1,))]
        held_out_gene = gene_indices[b, held_out_pos].item()

        # Soft target: co-mutation profile of held-out gene
        target = torch.tensor(comut_matrix[held_out_gene], device=device)
        target = target / (target.sum() + 1e-8)  # normalize to probability

        # Cross-entropy against the co-mutation profile
        log_probs = F.log_softmax(logits[b], dim=-1)
        loss = loss - (target * log_probs).sum()
        n_valid += 1

    return loss / max(n_valid, 1)


def channel_damage_loss(logits, targets):
    """Binary cross-entropy for channel co-damage prediction."""
    return F.binary_cross_entropy_with_logits(logits, targets)


def link_prediction_loss(model, mutation_embeddings, node_mask, gene_indices,
                         positive_pairs, n_negatives=5, sign=1.0,
                         edge_type=None):
    """Contrastive loss for edge-type-specific link prediction.

    Each edge type has its own projection head. Sign determines direction:
      sign=+1 (PPI, BELONGS_TO):
        positive pairs should be CLOSE — structural reinforcement
      sign=-1 (SL_PARTNER, CONVERGES):
        positive pairs should be FAR — compensatory opposition
      sign=0 (TRANSPOSES, ANALOGOUS, COUPLES, ENABLES, UNIQUE, CO_ESSENTIAL, UNKNOWN):
        per-edge sign from data — direction varies per instance
        no contrastive training unless per-edge sign is available
    """
    device = mutation_embeddings.device
    B, N, H = mutation_embeddings.shape

    if not positive_pairs:
        return torch.tensor(0.0, device=device)

    # Build gene embedding lookup: average across all patients where gene appears
    gene_embeds = {}
    for b in range(B):
        for s in range(N):
            if node_mask[b, s] > 0:
                gi = gene_indices[b, s].item()
                if gi not in gene_embeds:
                    gene_embeds[gi] = []
                gene_embeds[gi].append(mutation_embeddings[b, s])

    if len(gene_embeds) < 10:
        return torch.tensor(0.0, device=device)

    # Average embeddings per gene
    avg_embeds = {}
    for gi, embeds in gene_embeds.items():
        avg_embeds[gi] = torch.stack(embeds).mean(dim=0)

    available_genes = list(avg_embeds.keys())

    loss = torch.tensor(0.0, device=device)
    n_valid = 0
    margin = 0.5

    for gi, gj, *_ in positive_pairs:
        if gi not in avg_embeds or gj not in avg_embeds:
            continue

        pos_score = model.gene_pair_score(
            avg_embeds[gi].unsqueeze(0), avg_embeds[gj].unsqueeze(0),
            edge_type=edge_type,
        )

        # Negative samples
        for _ in range(n_negatives):
            neg_g = available_genes[np.random.randint(len(available_genes))]
            if neg_g == gi or neg_g == gj:
                continue
            neg_score = model.gene_pair_score(
                avg_embeds[gi].unsqueeze(0), avg_embeds[neg_g].unsqueeze(0),
                edge_type=edge_type,
            )
            # Margin loss: sign=+1 pushes pos>neg, sign=-1 pushes neg>pos
            loss = loss + F.relu(margin - sign * (pos_score - neg_score))
            n_valid += 1

    return loss / max(n_valid, 1)


# =========================================================================
# Training
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 10 epochs, small batch')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 10
        args.batch_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"{'='*70}")
    print(f"  GLM MULTI-TASK MECHANISTIC PRETRAINING")
    print(f"  11 edge-type predictors + COOCCURS holdback exam")
    print(f"  + PPI, TRANSPOSES, BELONGS_TO, ANALOGOUS, COUPLES, UNIQUE, CO_ESSENTIAL")
    print(f"  - SL_PARTNER, CONVERGES")
    print(f"  ? ENABLES (directional), UNKNOWN (residual)")
    print(f"  COOCCURS = composite exam | OS = survival exam")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # === Load data ===
    ds = AtlasDataset()
    data = ds.build_features()

    # Gene vocabulary
    gene_vocab = {}
    for patient_genes in data['gene_names']:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    n_genes = len(gene_vocab)
    print(f"\n  Gene vocabulary: {n_genes} unique genes")

    gene_indices = np.zeros((len(data['gene_names']), MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long).to(device)

    # Block/channel assignments
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

    block_ids_t = torch.tensor(block_ids, dtype=torch.long).to(device)
    channel_ids_t = torch.tensor(channel_ids, dtype=torch.long).to(device)

    # === Pairwise edge features (reuse V6 computation) ===
    print("  Loading graph data...", flush=True)
    from gnn.scripts.train_glm import load_graph_data, compute_gene_pair_matrix, \
        compute_patient_edge_features, gather_edge_features

    graph_data = load_graph_data()
    gene_pair_matrix, graph_edge_dim = compute_gene_pair_matrix(graph_data, gene_vocab)
    gene_pair_matrix_t = torch.tensor(gene_pair_matrix, dtype=torch.float32).to(device)

    patient_edge_feats, patient_edge_dim = compute_patient_edge_features(data['node_features'])
    patient_edge_feats_t = torch.tensor(patient_edge_feats, dtype=torch.float32).to(device)
    total_edge_dim = graph_edge_dim + patient_edge_dim

    # === Pretraining targets (9 edge types + COOCCURS holdback) ===
    targets = load_pretraining_targets(gene_vocab, channel_idx)
    comut_matrix = targets['comut_matrix']
    edge_data = targets['edge_data']

    # Channel co-damage targets per patient (BELONGS_TO validation)
    channel_targets = build_patient_channel_targets(data, channel_idx)
    channel_targets_t = torch.tensor(channel_targets, dtype=torch.float32).to(device)

    # Edge types with training signal (skip UNKNOWN — sign=0, needs investigation)
    trainable_edges = {k: v for k, v in edge_data.items()
                       if v['sign'] != 0 and len(v['pairs']) > 0}
    print(f"  Trainable edge types: {list(trainable_edges.keys())}")
    print(f"  UNKNOWN pairs (residual): {len(edge_data.get('UNKNOWN', {}).get('pairs', []))}")

    # === Data tensors ===
    nf = data['node_features'].to(device)
    nm = data['node_masks'].to(device)
    ct = data['cancer_types'].to(device)
    ages = data['ages'].to(device)
    sexes = data['sexes'].to(device)
    atlas_sums = data['atlas_sums'].to(device)
    n_cancer_types = data['n_cancer_types']

    from gnn.data.temporal import TemporalEstimator, year_features
    patient_ids = ds.clinical['patientId'].tolist()
    te = TemporalEstimator(ds.clinical)
    years, confs = te.estimate_all(patient_ids)
    temporal_feats = year_features(years, confs)
    temporal_t = torch.tensor(temporal_feats, dtype=torch.float32).to(device)

    clinical = torch.cat([
        ages.unsqueeze(-1), sexes.unsqueeze(-1), temporal_t
    ], dim=-1)

    # === Model ===
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
        'n_genes': n_genes,
    }

    model = PretrainBackbone(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Task loss weights (learned) — one per trainable edge type + channel
    n_tasks = len(trainable_edges) + 1  # +1 for channel co-damage
    task_names = ['channel'] + list(trainable_edges.keys())
    log_weights = nn.Parameter(torch.zeros(n_tasks, device=device))
    optimizer_w = torch.optim.Adam([log_weights], lr=1e-2)
    print(f"  Task weights: {task_names} ({n_tasks} tasks)")

    # === Train/val split (no survival involved) ===
    n_total = len(data['gene_names'])
    np.random.seed(args.seed)
    perm = np.random.permutation(n_total)
    n_val = int(n_total * 0.1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    batch_size = args.batch_size
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    best_metrics = {}

    print(f"\n  Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        ep_perm = np.random.permutation(len(train_idx))
        epoch_losses = defaultdict(float)
        n_batches = 0

        for b_start in range(0, len(ep_perm), batch_size):
            b_rel = ep_perm[b_start:b_start + batch_size]
            b_abs = torch.tensor(train_idx[b_rel], dtype=torch.long)

            optimizer.zero_grad()
            optimizer_w.zero_grad()

            # Forward
            batch_edge_feats = gather_edge_features(
                gene_pair_matrix_t, patient_edge_feats_t[b_abs],
                gene_indices_t[b_abs], nm[b_abs],
            )

            patient_embed = model(
                nf[b_abs], nm[b_abs], ct[b_abs],
                clinical[b_abs], atlas_sums[b_abs],
                batch_edge_feats,
                block_ids_t[b_abs], channel_ids_t[b_abs],
            )

            # === Channel co-damage (BELONGS_TO validation) ===
            ch_logits = model.channel_damage_logits()
            loss_channel = channel_damage_loss(ch_logits, channel_targets_t[b_abs])

            # === Edge-type link predictions (8 mechanistic types) ===
            task_losses = {'channel': loss_channel}

            for edge_name, edge_info in trainable_edges.items():
                pairs = edge_info['pairs']
                sign = edge_info['sign']
                n_sample = min(50, len(pairs))
                if n_sample == 0:
                    continue
                sampled = [pairs[i] for i in
                          np.random.choice(len(pairs), n_sample, replace=False)]
                loss_edge = link_prediction_loss(
                    model, model._mutation_embeddings, nm[b_abs],
                    gene_indices_t[b_abs], sampled,
                    n_negatives=3, sign=float(sign), edge_type=edge_name,
                )
                task_losses[edge_name] = loss_edge

            # === COOCCURS holdback (evaluated, not in loss) ===
            comut_logits = model.comut_logits()
            loss_comut = comut_loss(
                comut_logits, gene_indices_t[b_abs], nm[b_abs],
                comut_matrix, n_genes,
            )

            # Multi-task weighting (uncertainty-based)
            weights = torch.exp(-log_weights)
            total_loss = torch.tensor(0.0, device=device)
            for ti, tname in enumerate(task_names):
                if tname in task_losses:
                    total_loss = total_loss + weights[ti] * task_losses[tname] + log_weights[ti]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_w.step()

            for tname, tloss in task_losses.items():
                epoch_losses[tname] += tloss.item()
            epoch_losses['comut_holdback'] += loss_comut.item()
            epoch_losses['total'] += total_loss.item()
            n_batches += 1

        scheduler.step()

        # === Validation ===
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_losses = defaultdict(float)
                n_val_batches = 0

                for v_start in range(0, len(val_idx), batch_size):
                    v_end = min(v_start + batch_size, len(val_idx))
                    v_abs = torch.tensor(val_idx[v_start:v_end], dtype=torch.long)

                    v_edge = gather_edge_features(
                        gene_pair_matrix_t, patient_edge_feats_t[v_abs],
                        gene_indices_t[v_abs], nm[v_abs],
                    )
                    model(
                        nf[v_abs], nm[v_abs], ct[v_abs],
                        clinical[v_abs], atlas_sums[v_abs],
                        v_edge,
                        block_ids_t[v_abs], channel_ids_t[v_abs],
                    )

                    # COOCCURS holdback
                    v_comut = comut_loss(
                        model.comut_logits(), gene_indices_t[v_abs], nm[v_abs],
                        comut_matrix, n_genes,
                    )
                    v_channel = channel_damage_loss(
                        model.channel_damage_logits(), channel_targets_t[v_abs],
                    )

                    val_losses['comut_holdback'] += v_comut.item()
                    val_losses['channel'] += v_channel.item()
                    n_val_batches += 1

                # Use channel + comut holdback for model selection
                val_total = sum(val_losses[k] / n_val_batches for k in val_losses)

                if val_total < best_val_loss:
                    best_val_loss = val_total
                    best_epoch = epoch + 1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_metrics = {k: v / n_val_batches for k, v in val_losses.items()}

                # Compact logging
                w_vals = [torch.exp(-lw).item() for lw in log_weights]
                w_str = ' '.join(f'{v:.2f}' for v in w_vals)

                # Show per-edge-type training losses
                edge_strs = []
                for tname in task_names:
                    if tname in epoch_losses and epoch_losses[tname] > 0:
                        edge_strs.append(f"{tname}={epoch_losses[tname]/n_batches:.4f}")
                train_str = ' '.join(edge_strs)

                print(f"  Epoch {epoch+1:3d}: {train_str} "
                      f"| holdback_comut={val_losses['comut_holdback']/n_val_batches:.4f} "
                      f"val_ch={val_losses['channel']/n_val_batches:.4f} "
                      f"| w=[{w_str}]",
                      flush=True)

    # === Save backbone ===
    print(f"\n  Best epoch: {best_epoch}, val loss: {best_val_loss:.4f}")

    # Extract backbone state (only the parts that transfer to GLM)
    backbone_keys = [k for k in best_state.keys()
                     if not k.startswith('comut_head')
                     and not k.startswith('channel_damage_head')
                     and not k.startswith('gene_embed_proj')
                     and not k.startswith('edge_projections')]

    backbone_state = {k: best_state[k] for k in backbone_keys}

    save_path = os.path.join(RESULTS_DIR, "pretrained_backbone.pt")
    torch.save({
        'backbone': backbone_state,
        'edge_projections': {k: best_state[k] for k in best_state
                             if k.startswith('edge_projections')},
        'config': {
            'hidden_dim': args.hidden_dim,
            'n_heads': args.n_heads,
            'n_intra_layers': args.n_layers,
            'dropout': args.dropout,
            'node_feat_dim': NODE_FEAT_DIM,
            'edge_feat_dim': total_edge_dim,
            'n_cancer_types': n_cancer_types,
            'n_channels': n_channels,
            'n_blocks': n_blocks,
        },
        'epoch': best_epoch,
        'val_loss': best_val_loss,
        'val_metrics': best_metrics,
        'edge_types': task_names,
        'edge_signs': {k: v['sign'] for k, v in edge_data.items()},
        'decomposition': {
            'n_explained': targets['n_explained'],
            'n_total_comut': targets['n_total_comut'],
            'n_unique': targets['n_novel'],
            'pct_explained': 100 * targets['n_explained'] / max(targets['n_total_comut'], 1),
        },
    }, save_path)
    print(f"  Saved backbone to {save_path}")

    # === Save metrics ===
    edge_counts = {k: len(v['pairs']) for k, v in edge_data.items()}
    results = {
        'best_epoch': best_epoch,
        'val_loss': float(best_val_loss),
        'val_metrics': {k: float(v) for k, v in best_metrics.items()},
        'n_params': n_params,
        'edge_types': task_names,
        'edge_counts': edge_counts,
        'decomposition': {
            'n_explained': targets['n_explained'],
            'n_total_comut': targets['n_total_comut'],
            'n_unique': targets['n_novel'],
            'pct_explained': round(100 * targets['n_explained'] / max(targets['n_total_comut'], 1), 1),
        },
        'n_genes': n_genes,
        'n_comut_pairs': int((comut_matrix > 0).sum() // 2),
    }
    with open(os.path.join(RESULTS_DIR, "pretrain_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  PRETRAINING COMPLETE")
    print(f"  Best val metrics: {best_metrics}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
