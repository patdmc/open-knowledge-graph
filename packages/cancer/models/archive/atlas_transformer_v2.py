"""
AtlasTransformer V2 — context-conditioned attention.

V1 bolted cancer type and age onto the readout as bias terms.
V2 puts them where they belong:
  - Cancer type conditions the EDGES (attention between mutations)
  - Age conditions the NODES (what each mutation means)

A TP53 mutation at 35 is a hereditary driver.
A TP53 mutation at 75 is accumulated damage.
TP53 × PIK3CA is multiplicative in colorectal but redundant in breast.

The context changes the meaning of the graph traversal.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContextConditionedAttention(nn.Module):
    """Multi-head attention where cancer type modulates Q-K interaction."""

    def __init__(self, hidden, n_heads, n_cancer_types, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # Cancer type → attention bias per head
        # Each cancer type learns which heads to activate
        self.cancer_attn_bias = nn.Embedding(n_cancer_types, n_heads)

        # Cancer type → key modulation
        # Shifts what each mutation "looks like" in cancer context
        self.cancer_key_mod = nn.Sequential(
            nn.Embedding(n_cancer_types, hidden),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, cancer_type, attn_mask=None):
        """
        x: (B, N, hidden)
        cancer_type: (B,) long
        attn_mask: (B, N) bool, True = ignore
        """
        B, N, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, n_heads, N, head_dim)

        # Cancer type modulates keys — shifts what mutations mean in context
        ct_key = self.cancer_key_mod(cancer_type)  # (B, hidden)
        ct_key = ct_key.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + ct_key  # broadcast over N

        # Standard scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, heads, N, N)

        # Cancer type attention bias per head
        ct_bias = self.cancer_attn_bias(cancer_type)  # (B, n_heads)
        attn = attn + ct_bias.unsqueeze(-1).unsqueeze(-1)  # broadcast over N×N

        # Mask padding
        if attn_mask is not None:
            # attn_mask: (B, N) True=ignore → expand to (B, 1, 1, N) for keys
            attn = attn.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden)
        return self.out_proj(out)


class ContextConditionedLayer(nn.Module):
    """Transformer layer with context-conditioned attention + FFN."""

    def __init__(self, hidden, n_heads, n_cancer_types, dropout=0.1):
        super().__init__()
        self.attn = ContextConditionedAttention(hidden, n_heads, n_cancer_types, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x, cancer_type, attn_mask=None):
        x = x + self.attn(self.norm1(x), cancer_type, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class AtlasTransformerV2(nn.Module):

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']
        hidden = config['hidden_dim']
        dropout = config['dropout']
        n_cancer_types = config.get('n_cancer_types', 80)
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('n_layers', 2)

        # Age conditions each node: project age into node space
        # Age modulates what each mutation means
        self.age_node_mod = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ELU(),
        )

        # Node embedding (mutation features + age modulation)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Channel positional embedding (6 channels × 2 positions = 12)
        self.channel_pos_embed = nn.Embedding(12, hidden)

        # Context-conditioned transformer layers
        self.layers = nn.ModuleList([
            ContextConditionedLayer(hidden, n_heads, n_cancer_types, dropout)
            for _ in range(n_layers)
        ])

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # Sex feature
        self.sex_encoder = nn.Linear(1, hidden // 4)

        # Atlas sum as skip connection
        self.atlas_skip = nn.Sequential(
            nn.Linear(1, hidden // 4),
            nn.ELU(),
        )

        # Readout — no separate cancer/age embeddings, they're inside the attention
        readout_in = hidden + hidden // 4 + hidden // 4
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_features, node_mask, channel_pos_ids,
                cancer_type, clinical, atlas_sum):
        """
        Args:
            node_features: (B, max_nodes, node_feat_dim)
            node_mask: (B, max_nodes) — 1 for real, 0 for pad
            channel_pos_ids: (B, max_nodes) — 0-11
            cancer_type: (B,) long
            clinical: (B, 2) — [age_z, sex]
            atlas_sum: (B, 1)
        """
        B, N, _ = node_features.shape

        # Encode nodes
        x = self.node_encoder(node_features)  # (B, N, hidden)

        # Age conditions each node — same age modulation applied to all nodes
        age = clinical[:, 0:1]  # (B, 1)
        age_mod = self.age_node_mod(age)  # (B, hidden)
        x = x * (1 + age_mod.unsqueeze(1))  # multiplicative gating: age scales node meaning

        # Channel-position embeddings
        x = x + self.channel_pos_embed(channel_pos_ids)

        # Padding mask (True = ignore)
        attn_mask = (node_mask == 0)

        # Context-conditioned transformer: cancer type modulates attention
        for layer in self.layers:
            x = layer(x, cancer_type, attn_mask)

        # Attention pooling
        attn_w = self.pool_attn(x)
        attn_w = attn_w.masked_fill(attn_mask.unsqueeze(-1), float('-inf'))
        attn_w = F.softmax(attn_w, dim=1)
        patient_embed = (x * attn_w).sum(dim=1)  # (B, hidden)

        # Sex
        sex = clinical[:, 1:2]  # (B, 1)
        sex_feat = self.sex_encoder(sex)

        # Atlas skip
        atlas_feat = self.atlas_skip(atlas_sum)

        # Readout — cancer type and age are already inside the transformer
        out = torch.cat([patient_embed, sex_feat, atlas_feat], dim=-1)
        hazard = self.readout(out)

        return hazard
