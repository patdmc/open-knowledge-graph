"""
AtlasTransformer V3 — graph-informed context-conditioned attention.

V1: cancer type + age as bias terms on readout
V2: cancer type conditions attention edges, age conditions nodes
V3: graph-structural features condition everything

The graph tells the transformer about the PATTERN of mutations:
  - How close are mutated genes on PPI? (pairwise distances)
  - How many independent damage clusters? (connected components)
  - Are mutations hitting one pathway or spanning many? (cross-channel)
  - Is damage centrally located or peripheral? (hub degree)

These structural features are computed from the PPI/pathway graph and
fed as patient-level context that modulates both attention and readout.
The graph generates the dimensions. The transformer learns what they mean.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConditionedAttention(nn.Module):
    """Multi-head attention where cancer type + graph structure modulate Q-K."""

    def __init__(self, hidden, n_heads, n_cancer_types, graph_dim, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # Cancer type → attention bias per head
        self.cancer_attn_bias = nn.Embedding(n_cancer_types, n_heads)

        # Cancer type → key modulation
        self.cancer_key_mod = nn.Embedding(n_cancer_types, hidden)

        # Graph structure → attention bias per head
        # The graph pattern modulates which mutation pairs interact
        self.graph_attn_bias = nn.Sequential(
            nn.Linear(graph_dim, n_heads * 2),
            nn.ELU(),
            nn.Linear(n_heads * 2, n_heads),
        )

        # Graph structure → key modulation
        # The graph context shifts what each mutation "looks like"
        self.graph_key_mod = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.ELU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, cancer_type, graph_context, attn_mask=None):
        """
        x: (B, N, hidden)
        cancer_type: (B,) long
        graph_context: (B, graph_dim) — patient-level graph structural features
        attn_mask: (B, N) bool, True = ignore
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Cancer type modulates keys
        ct_key = self.cancer_key_mod(cancer_type).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + ct_key

        # Graph structure modulates keys — the mutation pattern context
        # shifts what each individual mutation means
        g_key = self.graph_key_mod(graph_context).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + g_key

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Cancer type attention bias
        ct_bias = self.cancer_attn_bias(cancer_type)
        attn = attn + ct_bias.unsqueeze(-1).unsqueeze(-1)

        # Graph structure attention bias — pattern determines which pairs matter
        g_bias = self.graph_attn_bias(graph_context)  # (B, n_heads)
        attn = attn + g_bias.unsqueeze(-1).unsqueeze(-1)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden)
        return self.out_proj(out)


class GraphConditionedLayer(nn.Module):
    def __init__(self, hidden, n_heads, n_cancer_types, graph_dim, dropout=0.1):
        super().__init__()
        self.attn = GraphConditionedAttention(hidden, n_heads, n_cancer_types, graph_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x, cancer_type, graph_context, attn_mask=None):
        x = x + self.attn(self.norm1(x), cancer_type, graph_context, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class AtlasTransformerV3(nn.Module):
    """Graph-informed atlas transformer.

    Per-mutation features come from the survival atlas (what each mutation does).
    Graph features come from PPI/pathway structure (how mutations relate to each other).
    Cancer type conditions the attention (context changes meaning).
    Age conditions the nodes (same mutation means different things at different ages).
    """

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']
        hidden = config['hidden_dim']
        dropout = config['dropout']
        n_cancer_types = config.get('n_cancer_types', 80)
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('n_layers', 2)
        graph_dim = config['graph_feat_dim']  # pairwise structural features
        self.mutation_only = config.get('mutation_only', False)

        # Age conditions each node
        self.age_node_mod = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ELU(),
        )

        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Channel positional embedding — disabled in mutation-only mode
        # (gene-level identity encoding drowns mutation-level signal)
        self.channel_pos_embed = nn.Embedding(12, hidden)

        # Graph feature encoder — compress graph structural features
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.graph_dim_out = hidden

        # Graph-conditioned transformer layers
        self.layers = nn.ModuleList([
            GraphConditionedLayer(hidden, n_heads, n_cancer_types, self.graph_dim_out, dropout)
            for _ in range(n_layers)
        ])

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        self.sex_encoder = nn.Linear(1, hidden // 4)
        self.atlas_skip = nn.Sequential(nn.Linear(1, hidden // 4), nn.ELU())

        # Graph features also go into readout — direct path for structural signal
        readout_in = hidden + hidden // 4 + hidden // 4 + self.graph_dim_out
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_features, node_mask, channel_pos_ids,
                cancer_type, clinical, atlas_sum, graph_features):
        """
        Args:
            node_features: (B, max_nodes, node_feat_dim) — per-mutation atlas features
            node_mask: (B, max_nodes) — 1 for real, 0 for pad
            channel_pos_ids: (B, max_nodes) — 0-11
            cancer_type: (B,) long
            clinical: (B, 2) — [age_z, sex]
            atlas_sum: (B, 1)
            graph_features: (B, graph_feat_dim) — pairwise structural features from PPI graph
        """
        B, N, _ = node_features.shape

        # Encode per-mutation nodes
        x = self.node_encoder(node_features)

        # Age conditions each node
        age = clinical[:, 0:1]
        age_mod = self.age_node_mod(age)
        x = x * (1 + age_mod.unsqueeze(1))

        # Channel-position embeddings — skip in mutation-only mode
        # Gene topology comes through graph_features instead
        if not self.mutation_only:
            x = x + self.channel_pos_embed(channel_pos_ids)

        # Encode graph structural features
        graph_context = self.graph_encoder(graph_features)

        # Graph-conditioned transformer
        attn_mask = (node_mask == 0)
        for layer in self.layers:
            x = layer(x, cancer_type, graph_context, attn_mask)

        # Attention pooling
        attn_w = self.pool_attn(x)
        attn_w = attn_w.masked_fill(attn_mask.unsqueeze(-1), float('-inf'))
        attn_w = F.softmax(attn_w, dim=1)
        patient_embed = (x * attn_w).sum(dim=1)

        sex = clinical[:, 1:2]
        sex_feat = self.sex_encoder(sex)
        atlas_feat = self.atlas_skip(atlas_sum)

        # Readout: transformer output + sex + atlas skip + graph structure (direct path)
        out = torch.cat([patient_embed, sex_feat, atlas_feat, graph_context], dim=-1)
        hazard = self.readout(out)

        return hazard
