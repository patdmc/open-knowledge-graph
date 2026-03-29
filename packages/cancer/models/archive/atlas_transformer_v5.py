"""
AtlasTransformer V5 — Hierarchical block-sparse attention from graph topology.

The graph IS the model. The transformer IS the hierarchical scorer v2.

Architecture matches the encoding hierarchy:
  Level 0: Self-attention within sub-pathway blocks (~5-20 mutations)
            Block assignments from PPI community detection.
            Edge features from ALL edge types bias attention directly.
  Level 1: Attention-pool blocks → channel tokens (6 channels)
  Level 2: Cross-channel attention on 6 channel tokens
  Level 3: Readout (channel tokens + clinical + atlas skip)

DYNAMIC: Edge types and node properties are discovered from the graph
at runtime. When new data sources add edge types, the model adapts —
just rebuild with the new schema and the edge_feature_dim grows.

The graph generates both the block structure AND the attention biases.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Attention layers ─────────────────────────────────────────────────

class EdgeInformedAttention(nn.Module):
    """Multi-head attention with pairwise edge features as attention bias.

    The graph tells the attention exactly how two mutations relate.
    Edge features → learned bias per head, added to Q·K scores.
    """

    def __init__(self, hidden, n_heads, n_edge_features, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # Edge features → attention bias per head
        # n_edge_features is dynamic — discovered from graph schema
        self.edge_bias = nn.Sequential(
            nn.Linear(n_edge_features, n_heads * 2),
            nn.ELU(),
            nn.Linear(n_heads * 2, n_heads),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, edge_features, attn_mask=None):
        """
        x: (B, N, hidden)
        edge_features: (B, N, N, n_edge_features) — pairwise graph edges
        attn_mask: (B, N, N) bool — True = block (don't attend)
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Edge bias: graph structure directly biases attention
        e_bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)  # (B, H, N, N)
        attn = attn + e_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # all-masked rows produce NaN
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden)
        return self.out_proj(out), attn


class BlockAttentionLayer(nn.Module):
    """Transformer layer with block-sparse edge-informed attention."""

    def __init__(self, hidden, n_heads, n_edge_features, dropout=0.1):
        super().__init__()
        self.attn = EdgeInformedAttention(hidden, n_heads, n_edge_features, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )

    def forward(self, x, edge_features, block_mask=None):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, edge_features, block_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class CrossChannelAttention(nn.Module):
    """Full attention on channel tokens — the metalanguage layer."""

    def __init__(self, hidden, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, C, _ = x.shape

        q = self.q_proj(x).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, C, self.hidden)
        return self.out_proj(out), attn


class CrossChannelLayer(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.1):
        super().__init__()
        self.attn = CrossChannelAttention(hidden, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


# ─── Main model ───────────────────────────────────────────────────────

class AtlasTransformerV5(nn.Module):
    """Hierarchical block-sparse transformer — the encoding hierarchy as architecture.

    Level 0: Edge-informed attention within sub-pathway blocks
    Level 1: Attention-pool blocks → channel tokens
    Level 2: Cross-channel attention on n_channels tokens
    Level 3: Readout

    Config must include:
        node_feat_dim: int — base atlas features (14) + graph node extras (dynamic)
        hidden_dim: int
        n_edge_features: int — from GraphSchema.edge_feature_dim (dynamic)
        n_channels: int — from GraphSchema.n_channels (dynamic)
        n_cancer_types: int
    """

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']
        hidden = config['hidden_dim']
        dropout = config.get('dropout', 0.1)
        n_cancer_types = config.get('n_cancer_types', 80)
        n_heads = config.get('n_heads', 4)
        n_block_layers = config.get('n_block_layers', 2)
        n_cross_layers = config.get('n_cross_layers', 1)
        n_channels = config['n_channels']
        n_edge_features = config['n_edge_features']

        self.n_channels = n_channels
        self.hidden = hidden
        self.config = config

        # ─── Node encoding ───
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Cancer type conditions nodes
        self.cancer_node_mod = nn.Embedding(n_cancer_types, hidden)

        # Age conditions nodes
        self.age_node_mod = nn.Sequential(nn.Linear(1, hidden), nn.ELU())

        # Channel positional embedding (2x channels for hub/leaf distinction)
        self.channel_pos_embed = nn.Embedding(n_channels * 2, hidden)

        # ─── Level 0: Block-local attention ───
        self.block_layers = nn.ModuleList([
            BlockAttentionLayer(hidden, n_heads, n_edge_features, dropout)
            for _ in range(n_block_layers)
        ])

        # ─── Level 1: Block → Channel pooling ───
        self.block_pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.block_to_channel = nn.Linear(hidden, hidden)

        # ─── Level 2: Cross-channel attention ───
        self.cross_layers = nn.ModuleList([
            CrossChannelLayer(hidden, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # ─── Level 3: Readout ───
        self.channel_pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        self.sex_encoder = nn.Linear(1, hidden // 4)
        self.atlas_skip = nn.Sequential(nn.Linear(1, hidden // 4), nn.ELU())

        readout_in = hidden + hidden // 4 + hidden // 4
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Store attention weights for precipitation
        self._last_block_attn = None
        self._last_cross_attn = None

    def _backbone(self, node_features, node_mask, channel_pos_ids,
                  cancer_type, clinical, edge_features, block_ids, channel_ids):
        """Run the shared backbone: node encoding → block attention → channel pooling → cross-channel.

        Returns:
            node_hidden: (B, N, hidden) — per-node representations after block attention
            patient_embed: (B, hidden) — pooled patient-level representation
            channel_tokens: (B, n_channels, hidden)
            pad_mask: (B, N) bool
        """
        B, N, _ = node_features.shape

        # ─── Encode nodes ───
        x = self.node_encoder(node_features)
        x = x + self.cancer_node_mod(cancer_type).unsqueeze(1)

        age = clinical[:, 0:1]
        x = x * (1 + self.age_node_mod(age).unsqueeze(1))

        x = x + self.channel_pos_embed(channel_pos_ids.clamp(max=self.n_channels * 2 - 1))

        # ─── Level 0: Block-local attention ───
        pad_mask = (node_mask == 0)
        block_mask = (block_ids.unsqueeze(1) != block_ids.unsqueeze(2))
        block_mask = block_mask | pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)

        for layer in self.block_layers:
            x, attn_w = layer(x, edge_features, block_mask)
        self._last_block_attn = attn_w

        node_hidden = x  # save per-node representations

        # ─── Level 1: Pool blocks → channels ───
        channel_tokens = torch.zeros(B, self.n_channels, self.hidden,
                                     device=x.device, dtype=x.dtype)
        channel_active = torch.zeros(B, self.n_channels, device=x.device)

        for ch_id in range(self.n_channels):
            ch_mask = (channel_ids == ch_id) & (~pad_mask)
            has_ch = ch_mask.any(dim=1)
            if not has_ch.any():
                continue

            channel_active[:, ch_id] = has_ch.float()

            pool_scores = self.block_pool_attn(x).squeeze(-1)
            pool_scores = pool_scores.masked_fill(~ch_mask, float('-inf'))
            pool_weights = F.softmax(pool_scores, dim=1).nan_to_num(0.0)
            pool_weights = pool_weights.masked_fill(~ch_mask, 0.0)

            ch_embed = (x * pool_weights.unsqueeze(-1)).sum(dim=1)
            channel_tokens[:, ch_id] = self.block_to_channel(ch_embed)

        # ─── Level 2: Cross-channel attention ───
        ch_pad_mask = (channel_active == 0)
        cross_mask = ch_pad_mask.unsqueeze(1) | ch_pad_mask.unsqueeze(2)

        for layer in self.cross_layers:
            channel_tokens, attn_w = layer(channel_tokens, cross_mask)
        self._last_cross_attn = attn_w

        # ─── Pool channels → patient embedding ───
        ch_pool_scores = self.channel_pool_attn(channel_tokens).squeeze(-1)
        ch_pool_scores = ch_pool_scores.masked_fill(ch_pad_mask, float('-inf'))
        ch_pool_weights = F.softmax(ch_pool_scores, dim=1).nan_to_num(0.0)
        ch_pool_weights = ch_pool_weights.masked_fill(ch_pad_mask, 0.0)
        patient_embed = (channel_tokens * ch_pool_weights.unsqueeze(-1)).sum(dim=1)

        return node_hidden, patient_embed, channel_tokens, pad_mask

    def encode(self, node_features, node_mask, channel_pos_ids,
               cancer_type, clinical, edge_features, block_ids, channel_ids):
        """Return backbone representations without the survival readout.

        Returns:
            node_hidden: (B, N, hidden) — per-node after block attention
            patient_embed: (B, hidden) — pooled patient-level embedding
        """
        node_hidden, patient_embed, _, _ = self._backbone(
            node_features, node_mask, channel_pos_ids,
            cancer_type, clinical, edge_features, block_ids, channel_ids,
        )
        return node_hidden, patient_embed

    def forward(self, node_features, node_mask, channel_pos_ids,
                cancer_type, clinical, atlas_sum,
                edge_features, block_ids, channel_ids):
        """
        Args:
            node_features: (B, N, node_feat_dim) — atlas + dynamic graph node properties
            node_mask: (B, N) — 1 real, 0 pad
            channel_pos_ids: (B, N) — 0 to 2*n_channels-1
            cancer_type: (B,) long
            clinical: (B, 2) — [age_z, sex]
            atlas_sum: (B, 1)
            edge_features: (B, N, N, n_edge_features) — dynamic pairwise graph edges
            block_ids: (B, N) long — sub-pathway block per mutation
            channel_ids: (B, N) long — channel per mutation (n_channels = unassigned)
        """
        _, patient_embed, _, _ = self._backbone(
            node_features, node_mask, channel_pos_ids,
            cancer_type, clinical, edge_features, block_ids, channel_ids,
        )

        sex_feat = self.sex_encoder(clinical[:, 1:2])
        atlas_feat = self.atlas_skip(atlas_sum)

        out = torch.cat([patient_embed, sex_feat, atlas_feat], dim=-1)
        return self.readout(out)

    def get_attention_maps(self):
        """Return last-computed attention weights for precipitation into graph."""
        return {
            'block_attn': self._last_block_attn,
            'cross_channel_attn': self._last_cross_attn,
        }

    @staticmethod
    def config_from_schema(schema, n_cancer_types=80):
        """Build model config directly from GraphSchema — no hardcoding needed."""
        from gnn.data.atlas_dataset import NODE_FEAT_DIM
        return {
            'node_feat_dim': NODE_FEAT_DIM + schema.node_extra_dim,
            'hidden_dim': 64,
            'dropout': 0.1,
            'n_cancer_types': n_cancer_types,
            'n_heads': 4,
            'n_block_layers': 2,
            'n_cross_layers': 1,
            'n_channels': schema.n_channels,
            'n_edge_features': schema.edge_feature_dim,
        }
