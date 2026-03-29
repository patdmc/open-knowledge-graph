"""
ChannelNetV5 — Hub/leaf split: 12 nodes (6 channels × 2 positions).

Architecture:
  1. Encode 12 hub/leaf nodes -> transformer -> contextualized embeddings
  2. Attention pooling -> patient embedding
  3. Cancer type embedding + clinical covariates
  4. Skip: raw features
  5. Readout MLP -> hazard

The transformer can learn:
  - Hub-hub interactions across channels (upstream cascade)
  - Hub-leaf interactions within channels (pathway depth)
  - Leaf-leaf interactions (parallel downstream effects)
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use channel_net_v6c.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.channel_dataset_v5 import HUBLF_FEAT_DIM, N_NODES


class ChannelNetV5(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.get("hublf_feat_dim", HUBLF_FEAT_DIM)
        hidden = config["hidden_dim"]
        dropout = config["dropout"]
        n_cancer_types = config.get("n_cancer_types", 80)

        # --- 12-node encoder + transformer ---
        self.node_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.node_pos = nn.Embedding(N_NODES, hidden)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=config["cross_channel_heads"],
                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
            ),
            num_layers=config.get("cross_channel_layers", 2),
        )

        self.node_attn = nn.Sequential(
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
        skip_raw = N_NODES * feat_dim
        total_dim = hidden + hidden // 2 + hidden // 4 + skip_raw

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
        node_feats = batch["hublf_features"]      # (B, 12, 16)
        ct_idx = batch["cancer_type_idx"]          # (B,)
        age = batch["age"]                         # (B,)
        sex = batch["sex"]                         # (B,)
        B = node_feats.shape[0]

        # --- 12-node path ---
        h = self.node_encoder(node_feats)                       # (B, 12, H)
        pos = self.node_pos.weight.unsqueeze(0).expand(B, -1, -1)
        h = self.transformer(h + pos)                           # (B, 12, H)
        scores = self.node_attn(h).squeeze(-1)                  # (B, 12)
        weights = F.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(1)             # (B, H)

        # --- Cancer type ---
        cancer = self.cancer_embed(ct_idx)                      # (B, H//2)

        # --- Clinical ---
        clinical = self.clinical_encoder(
            torch.stack([age, sex.float()], dim=-1)
        )                                                        # (B, H//4)

        # --- Skip: raw features ---
        raw = node_feats.reshape(B, -1)                         # (B, 12*16)

        # --- Concat ---
        combined = torch.cat([pooled, cancer, clinical, raw], dim=-1)

        return self.readout(combined).squeeze(-1)                # (B,)
