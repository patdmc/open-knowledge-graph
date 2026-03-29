"""
ChannelNetV8 — Interaction-aware architecture.

Same 12-node transformer as V7, plus an interaction branch that
encodes which specific cross-channel mutation pairs are co-present.

The interaction features are:
  - Binary indicators for each significant interaction pair
  - Scaled by log(HR) from Cox interaction scan
  - Projected through a small MLP before concat to readout

This lets the model directly leverage the 0/3 taxonomy:
  Type 0 (counteractive): negative log HR → protective signal
  Type 3 (multiplicative): positive log HR → compounding signal

The transformer still learns general cross-channel attention,
but the interaction branch gives it a shortcut for the strongest
known coupling effects.
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use channel_net_v6c.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F


N_NODES = 12


class ChannelNetV8(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config["feat_dim"]
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)
        n_interactions = config.get("n_interactions", 10)

        base_dim = config.get("base_feat_dim", 16)
        hotspot_dim = feat_dim - base_dim

        # --- Node encoders (same as V7) ---
        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        if hotspot_dim > 0:
            self.hotspot_encoder = nn.Sequential(
                nn.Linear(hotspot_dim, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.ELU(),
                nn.Dropout(dropout),
            )
            combine_dim = hidden
        else:
            self.hotspot_encoder = None
            combine_dim = hidden // 2

        self.combine = nn.Sequential(
            nn.Linear(combine_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # --- Positional embeddings ---
        self.node_pos = nn.Embedding(N_NODES, hidden)

        # --- Cross-channel transformer ---
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config["cross_channel_heads"],
                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
            ),
            num_layers=config.get("cross_channel_layers", 2),
        )

        # --- Attention pooling ---
        self.node_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Cancer type embedding ---
        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # --- Clinical encoder ---
        clinical_dim = config.get("clinical_dim", 2)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden // 4),
            nn.ELU(),
        )

        # --- Skip connection ---
        skip_dim = N_NODES * feat_dim
        self.skip_compress = nn.Sequential(
            nn.Linear(skip_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # --- Interaction branch (NEW in V8) ---
        # Input: n_interactions binary features, pre-scaled by log(HR)
        # Two paths: raw binary (model learns weights) + pre-scaled (prior from Cox)
        interact_input_dim = n_interactions * 2  # raw + scaled
        self.interaction_encoder = nn.Sequential(
            nn.Linear(interact_input_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ELU(),
        )

        # --- Readout (now includes interaction branch) ---
        readout_in = hidden + hidden // 2 + hidden // 4 + hidden + hidden // 4
        #             pool  + cancer_type + clinical   + skip  + interactions
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self.base_dim = base_dim
        self.hotspot_dim = hotspot_dim

    def forward(self, node_features, cancer_type, clinical,
                interactions=None, interaction_weights=None):
        """
        Args:
            node_features: (batch, 12, feat_dim)
            cancer_type: (batch,) long
            clinical: (batch, clinical_dim)
            interactions: (batch, n_interactions) binary
            interaction_weights: (n_interactions,) log(HR) for scaling
        """
        B = node_features.size(0)

        # --- Node encoding ---
        base = node_features[:, :, :self.base_dim]
        base_enc = self.base_encoder(base)

        if self.hotspot_encoder is not None and self.hotspot_dim > 0:
            hotspots = node_features[:, :, self.base_dim:]
            hs_enc = self.hotspot_encoder(hotspots)
            combined = torch.cat([base_enc, hs_enc], dim=-1)
        else:
            combined = base_enc

        x = self.combine(combined)
        pos = self.node_pos(torch.arange(N_NODES, device=x.device))
        x = x + pos.unsqueeze(0)
        x = self.transformer(x)

        # Attention pooling
        attn_w = self.node_attn(x)
        attn_w = F.softmax(attn_w, dim=1)
        patient_embed = (x * attn_w).sum(dim=1)

        # Cancer type
        ct_embed = self.cancer_embed(cancer_type)

        # Clinical
        clin = self.clinical_encoder(clinical)

        # Skip
        skip = self.skip_compress(node_features.view(B, -1))

        # --- Interaction branch ---
        if interactions is not None:
            # Scale by log(HR) for prior-informed features
            if interaction_weights is not None:
                w = interaction_weights.to(interactions.device)
                scaled = interactions * w.unsqueeze(0)
            else:
                scaled = interactions
            interact_input = torch.cat([interactions, scaled], dim=-1)
            interact_embed = self.interaction_encoder(interact_input)
        else:
            interact_embed = torch.zeros(B, self.readout[0].in_features -
                                          (patient_embed.size(1) + ct_embed.size(1) +
                                           clin.size(1) + skip.size(1)),
                                          device=node_features.device)

        # Readout
        out = torch.cat([patient_embed, ct_embed, clin, skip, interact_embed], dim=-1)
        hazard = self.readout(out)

        return hazard

    def get_attention_weights(self, node_features, cancer_type, clinical):
        B = node_features.size(0)
        base = node_features[:, :, :self.base_dim]
        base_enc = self.base_encoder(base)
        if self.hotspot_encoder is not None and self.hotspot_dim > 0:
            hotspots = node_features[:, :, self.base_dim:]
            hs_enc = self.hotspot_encoder(hotspots)
            combined = torch.cat([base_enc, hs_enc], dim=-1)
        else:
            combined = base_enc
        x = self.combine(combined)
        pos = self.node_pos(torch.arange(N_NODES, device=x.device))
        x = x + pos.unsqueeze(0)
        x = self.transformer(x)
        attn_w = self.node_attn(x)
        attn_w = F.softmax(attn_w, dim=1)
        return attn_w.squeeze(-1)

    def get_interaction_contribution(self, interactions, interaction_weights=None):
        """Return the interaction branch output for interpretability."""
        if interaction_weights is not None:
            w = interaction_weights.to(interactions.device)
            scaled = interactions * w.unsqueeze(0)
        else:
            scaled = interactions
        interact_input = torch.cat([interactions, scaled], dim=-1)
        return self.interaction_encoder(interact_input)
