"""
ChannelNetV7 — Hotspot mutation encoding.

Extends V5 architecture with per-mutation hotspot features.
Same 12-node transformer, but each node now carries:
  - Base features (mutation count, VAF, GOF/LOF balance)
  - Binary hotspot indicators (is TP53 R175H present? KRAS G12D?)

The transformer learns which specific mutations in which channels
interact to predict survival. This captures the 62-percentage-point
spread between KRAS G12R (+12% death) and MSH3 K383Rfs*32 (-24%)
that gene-level models average over.

Architecture identical to V5 except:
  - node_encoder input dim adapts to feat_dim (base + hotspots)
  - hotspot features pass through a separate projection before concat
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use channel_net_v6c.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F


N_NODES = 12  # 6 channels × 2 positions


class ChannelNetV7(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config["feat_dim"]  # base + hotspots (variable)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        base_dim = config.get("base_feat_dim", 16)
        hotspot_dim = feat_dim - base_dim

        # --- Separate projections for base and hotspot features ---
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

        # --- Combine to hidden dim ---
        self.combine = nn.Sequential(
            nn.Linear(combine_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # --- Positional embeddings for 12 nodes ---
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

        # --- Clinical encoder (age, sex, TMB if available) ---
        clinical_dim = config.get("clinical_dim", 2)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden // 4),
            nn.ELU(),
        )

        # --- Skip connection from raw features (compressed) ---
        skip_dim = N_NODES * feat_dim
        self.skip_compress = nn.Sequential(
            nn.Linear(skip_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # --- Readout ---
        readout_in = hidden + hidden // 2 + hidden // 4 + hidden
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self.base_dim = base_dim
        self.hotspot_dim = hotspot_dim

    def forward(self, node_features, cancer_type, clinical):
        """
        Args:
            node_features: (batch, 12, feat_dim)
            cancer_type: (batch,) long tensor
            clinical: (batch, clinical_dim)

        Returns:
            hazard: (batch, 1) log-hazard scores
        """
        B = node_features.size(0)

        # Split base and hotspot features
        base = node_features[:, :, :self.base_dim]
        base_enc = self.base_encoder(base)  # (B, 12, hidden//2)

        if self.hotspot_encoder is not None and self.hotspot_dim > 0:
            hotspots = node_features[:, :, self.base_dim:]
            hs_enc = self.hotspot_encoder(hotspots)  # (B, 12, hidden//2)
            combined = torch.cat([base_enc, hs_enc], dim=-1)  # (B, 12, hidden)
        else:
            combined = base_enc

        x = self.combine(combined)  # (B, 12, hidden)

        # Add positional embeddings
        pos = self.node_pos(torch.arange(N_NODES, device=x.device))
        x = x + pos.unsqueeze(0)

        # Transformer
        x = self.transformer(x)  # (B, 12, hidden)

        # Attention pooling
        attn_weights = self.node_attn(x)  # (B, 12, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        patient_embed = (x * attn_weights).sum(dim=1)  # (B, hidden)

        # Cancer type embedding
        ct_embed = self.cancer_embed(cancer_type)  # (B, hidden//2)

        # Clinical
        clin = self.clinical_encoder(clinical)  # (B, hidden//4)

        # Skip connection (compressed)
        skip = self.skip_compress(node_features.view(B, -1))  # (B, hidden)

        # Readout
        out = torch.cat([patient_embed, ct_embed, clin, skip], dim=-1)
        hazard = self.readout(out)

        return hazard

    def get_attention_weights(self, node_features, cancer_type, clinical):
        """Return attention pooling weights for interpretability."""
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

        attn_weights = self.node_attn(x)
        attn_weights = F.softmax(attn_weights, dim=1)

        return attn_weights.squeeze(-1)  # (B, 12)
