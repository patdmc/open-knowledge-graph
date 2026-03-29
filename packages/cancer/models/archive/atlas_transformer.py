"""
AtlasTransformer — transformer trained on the knowledge graph.

Instead of encoding raw mutation features, each patient is represented
as a set of nodes from the atlas graph:
  - Each mutation the patient carries gets a node
  - Node features come from the atlas: log(HR), channel, position, tier
  - Edges between nodes carry interaction weights from the taxonomy
  - Transformer attention learns how atlas-informed nodes combine

This is the theory doing the work: the graph structure IS the model.
The transformer just learns the residual that the atlas sum misses.

Input per patient:
  - Variable number of mutation nodes (padded to max_nodes)
  - Each node: [log_hr, channel_onehot(6), position_onehot(2), tier_onehot(3),
                 ci_width, n_patients, is_hub, is_harmful, is_protective]
  - Edge features: interaction type for known pairs

Architecture:
  - Node embedding MLP
  - Positional encoding by channel (not sequence position)
  - Small transformer (2 layers, 4 heads)
  - Attention pooling → readout with cancer type + age
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtlasTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']  # per-node atlas features
        hidden = config['hidden_dim']
        dropout = config['dropout']
        n_cancer_types = config.get('n_cancer_types', 80)
        max_nodes = config.get('max_nodes', 32)

        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Channel positional embedding (6 channels × 2 positions = 12)
        self.channel_pos_embed = nn.Embedding(12, hidden)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config.get('n_heads', 4),
                dim_feedforward=hidden * 2, dropout=dropout,
                batch_first=True,
            ),
            num_layers=config.get('n_layers', 2),
        )

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # Cancer type
        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # Clinical
        clinical_dim = config.get('clinical_dim', 2)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden // 4),
            nn.ELU(),
        )

        # Atlas sum as skip connection (1-dim: just the sum of log HRs)
        self.atlas_skip = nn.Sequential(
            nn.Linear(1, hidden // 4),
            nn.ELU(),
        )

        # Readout
        readout_in = hidden + hidden // 2 + hidden // 4 + hidden // 4
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self.max_nodes = max_nodes

    def forward(self, node_features, node_mask, channel_pos_ids,
                cancer_type, clinical, atlas_sum):
        """
        Args:
            node_features: (B, max_nodes, node_feat_dim) — atlas features per mutation
            node_mask: (B, max_nodes) — 1 for real nodes, 0 for padding
            channel_pos_ids: (B, max_nodes) — channel*2+position index (0-11)
            cancer_type: (B,) long
            clinical: (B, clinical_dim)
            atlas_sum: (B, 1) — simple sum of log HRs (atlas baseline)
        """
        B = node_features.size(0)

        # Encode nodes
        x = self.node_encoder(node_features)  # (B, max_nodes, hidden)

        # Add channel-position embeddings
        cp_embed = self.channel_pos_embed(channel_pos_ids)  # (B, max_nodes, hidden)
        x = x + cp_embed

        # Create attention mask for padding (True = ignore)
        attn_mask = (node_mask == 0)  # (B, max_nodes)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Attention pooling (masked)
        attn_w = self.pool_attn(x)  # (B, max_nodes, 1)
        attn_w = attn_w.masked_fill(attn_mask.unsqueeze(-1), float('-inf'))
        attn_w = F.softmax(attn_w, dim=1)
        patient_embed = (x * attn_w).sum(dim=1)  # (B, hidden)

        # Cancer type
        ct = self.cancer_embed(cancer_type)

        # Clinical
        clin = self.clinical_encoder(clinical)

        # Atlas skip
        atlas_feat = self.atlas_skip(atlas_sum)

        # Readout
        out = torch.cat([patient_embed, ct, clin, atlas_feat], dim=-1)
        hazard = self.readout(out)

        return hazard
