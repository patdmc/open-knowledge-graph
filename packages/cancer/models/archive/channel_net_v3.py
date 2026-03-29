"""
ChannelNetV3 — Adds escalation chain encoding to V2.

Architecture extends V2 with:
  1. 6-channel path (transformer) → channel embeddings
  2. 3-tier path (transformer) → tier embeddings
  3. 2-chain path (MLP + cross-attention) → chain embeddings
  4. Cancer type embedding + clinical covariates
  5. Skip: concat(all pooled, all raw features)
  6. Readout MLP → hazard

The chain path captures biological cascade progression:
  - Chain A: PI3K_Growth → DDR → Endocrine
  - Chain B: CellCycle → TissueArch → Immune
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use channel_net_v6c.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CrossChannelAttention
from ..data.channel_dataset_v3 import CHAIN_FEAT_DIM, N_CHAINS


class ChannelNetV3(nn.Module):

    def __init__(self, config):
        super().__init__()
        ch_feat_dim = config.get("channel_feat_dim", 17)
        chain_feat_dim = config.get("chain_feat_dim", CHAIN_FEAT_DIM)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        # --- Channel path (6 channels) ---
        self.ch_encoder = nn.Sequential(
            nn.Linear(ch_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.ch_transformer = CrossChannelAttention(
            hidden, num_heads=config["cross_channel_heads"],
            num_layers=config.get("cross_channel_layers", 2),
            dropout=dropout,
        )
        self.ch_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Tier path (3 tiers) ---
        self.tier_encoder = nn.Sequential(
            nn.Linear(ch_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.tier_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config["cross_channel_heads"],
                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
            ),
            num_layers=config.get("cross_channel_layers", 2),
        )
        self.tier_pos = nn.Embedding(3, hidden)
        self.tier_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Escalation chain path (2 chains) ---
        self.chain_encoder = nn.Sequential(
            nn.Linear(chain_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        # Cross-chain attention: let the two chains attend to each other
        self.chain_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config["cross_channel_heads"],
                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
            ),
            num_layers=1,  # lightweight — only 2 chains
        )
        self.chain_pos = nn.Embedding(N_CHAINS, hidden)
        self.chain_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        # --- Cancer type embedding ---
        self.cancer_embed = nn.Embedding(n_cancer_types, hidden // 2)

        # --- Clinical encoder ---
        self.clinical_encoder = nn.Sequential(
            nn.Linear(2, hidden // 4),
            nn.ReLU(),
        )

        # --- Skip connection + readout ---
        skip_raw = (6 * ch_feat_dim      # raw channel features
                    + 3 * ch_feat_dim     # raw tier features
                    + N_CHAINS * chain_feat_dim)  # raw chain features
        total_dim = (hidden               # ch_pooled
                     + hidden             # tier_pooled
                     + hidden             # chain_pooled
                     + hidden // 2        # cancer embed
                     + hidden // 4        # clinical
                     + skip_raw)

        self.readout = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        ch_feats = batch["channel_features"]     # (B, 6, 17)
        tier_feats = batch["tier_features"]       # (B, 3, 17)
        chain_feats = batch["chain_features"]     # (B, 2, 10)
        ct_idx = batch["cancer_type_idx"]         # (B,)
        age = batch["age"]                        # (B,)
        sex = batch["sex"]                        # (B,)
        B = ch_feats.shape[0]

        # --- Channel path ---
        h_ch = self.ch_encoder(ch_feats)                        # (B, 6, H)
        h_ch = self.ch_transformer(h_ch)                        # (B, 6, H)
        ch_scores = self.ch_attn(h_ch).squeeze(-1)              # (B, 6)
        ch_weights = F.softmax(ch_scores, dim=-1)
        ch_pooled = (ch_weights.unsqueeze(-1) * h_ch).sum(1)    # (B, H)

        # --- Tier path ---
        h_tier = self.tier_encoder(tier_feats)                  # (B, 3, H)
        pos_t = self.tier_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h_tier = self.tier_transformer(h_tier + pos_t)          # (B, 3, H)
        tier_scores = self.tier_attn(h_tier).squeeze(-1)        # (B, 3)
        tier_weights = F.softmax(tier_scores, dim=-1)
        tier_pooled = (tier_weights.unsqueeze(-1) * h_tier).sum(1)  # (B, H)

        # --- Escalation chain path ---
        h_chain = self.chain_encoder(chain_feats)               # (B, 2, H)
        pos_c = self.chain_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h_chain = self.chain_transformer(h_chain + pos_c)       # (B, 2, H)
        chain_scores = self.chain_attn(h_chain).squeeze(-1)     # (B, 2)
        chain_weights = F.softmax(chain_scores, dim=-1)
        chain_pooled = (chain_weights.unsqueeze(-1) * h_chain).sum(1)  # (B, H)

        # --- Cancer type ---
        cancer = self.cancer_embed(ct_idx)                      # (B, H//2)

        # --- Clinical ---
        clinical = self.clinical_encoder(
            torch.stack([age, sex.float()], dim=-1)             # (B, 2)
        )                                                        # (B, H//4)

        # --- Skip: raw features ---
        raw_ch = ch_feats.reshape(B, -1)                        # (B, 6*17)
        raw_tier = tier_feats.reshape(B, -1)                    # (B, 3*17)
        raw_chain = chain_feats.reshape(B, -1)                  # (B, 2*10)

        # --- Concat everything ---
        combined = torch.cat([
            ch_pooled, tier_pooled, chain_pooled,
            cancer, clinical,
            raw_ch, raw_tier, raw_chain,
        ], dim=-1)

        return self.readout(combined).squeeze(-1)               # (B,)
