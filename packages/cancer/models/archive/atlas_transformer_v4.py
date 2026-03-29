"""
AtlasTransformer V4 — treatment-conditional survival prediction.

V1: cancer type + age as bias terms on readout
V2: cancer type conditions attention edges, age conditions nodes
V3: graph-structural features condition everything
V4: treatment conditions attention + readout for counterfactual prediction

The key insight: historical survival data conflates biology with treatment
selection. By conditioning on treatment, we disentangle them. Running the
same patient through different treatments gives counterfactual predictions
— the delta IS the treatment recommendation.

Treatment conditions the model the same way cancer type does: by modulating
attention (which mutation pairs matter depends on treatment) and readout
(treatment main effects).
"""

from gnn.deprecation import deprecated  # DEPRECATED
deprecated(__file__, "Use atlas_transformer_v6.py instead.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TreatmentConditionedAttention(nn.Module):
    """Multi-head attention where cancer type, graph structure, AND treatment modulate Q-K."""

    def __init__(self, hidden, n_heads, n_cancer_types, graph_dim, treatment_dim, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # Cancer type conditioning (from V2/V3)
        self.cancer_attn_bias = nn.Embedding(n_cancer_types, n_heads)
        self.cancer_key_mod = nn.Embedding(n_cancer_types, hidden)

        # Graph structure conditioning (from V3)
        self.graph_attn_bias = nn.Sequential(
            nn.Linear(graph_dim, n_heads * 2), nn.ELU(),
            nn.Linear(n_heads * 2, n_heads),
        )
        self.graph_key_mod = nn.Sequential(
            nn.Linear(graph_dim, hidden), nn.ELU(),
        )

        # Treatment conditioning (NEW in V4)
        # Which mutation pairs matter depends on treatment:
        # - On immunotherapy, immune-related mutation pairs matter more
        # - On targeted therapy, the specific target pathway matters more
        self.treatment_key_mod = nn.Sequential(
            nn.Linear(treatment_dim, hidden), nn.ELU(),
        )
        self.treatment_attn_bias = nn.Sequential(
            nn.Linear(treatment_dim, n_heads * 2), nn.ELU(),
            nn.Linear(n_heads * 2, n_heads),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, cancer_type, graph_context, treatment_embed, attn_mask=None):
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Cancer type modulates keys
        ct_key = self.cancer_key_mod(cancer_type).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + ct_key

        # Graph structure modulates keys
        g_key = self.graph_key_mod(graph_context).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + g_key

        # Treatment modulates keys — what each mutation "looks like" depends on treatment
        t_key = self.treatment_key_mod(treatment_embed).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k + t_key

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Cancer type attention bias
        ct_bias = self.cancer_attn_bias(cancer_type)
        attn = attn + ct_bias.unsqueeze(-1).unsqueeze(-1)

        # Graph structure attention bias
        g_bias = self.graph_attn_bias(graph_context)
        attn = attn + g_bias.unsqueeze(-1).unsqueeze(-1)

        # Treatment attention bias — which mutation pairs interact depends on treatment
        t_bias = self.treatment_attn_bias(treatment_embed)
        attn = attn + t_bias.unsqueeze(-1).unsqueeze(-1)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden)
        return self.out_proj(out)


class TreatmentConditionedLayer(nn.Module):
    def __init__(self, hidden, n_heads, n_cancer_types, graph_dim, treatment_dim, dropout=0.1):
        super().__init__()
        self.attn = TreatmentConditionedAttention(
            hidden, n_heads, n_cancer_types, graph_dim, treatment_dim, dropout
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )

    def forward(self, x, cancer_type, graph_context, treatment_embed, attn_mask=None):
        x = x + self.attn(self.norm1(x), cancer_type, graph_context, treatment_embed, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class AtlasTransformerV4(nn.Module):
    """Treatment-conditional atlas transformer.

    Same architecture as V3, plus:
    - Treatment encoder: 11-dim binary vector → hidden embedding
    - Treatment conditions attention (which mutation pairs matter under this treatment)
    - Treatment in readout (direct path for treatment main effects)
    - Handles missing treatment fields via learned unknown tokens
    """

    # Treatment vector dimensions
    N_TREATMENT_DIMS = 11
    # [0] surgery, [1] radiation, [2] chemo, [3] endocrine, [4] targeted, [5] immuno
    # [6] platinum, [7] taxane, [8] anthracycline, [9] antimetabolite, [10] alkylating

    def __init__(self, config):
        super().__init__()
        node_feat_dim = config['node_feat_dim']
        hidden = config['hidden_dim']
        dropout = config['dropout']
        n_cancer_types = config.get('n_cancer_types', 80)
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('n_layers', 2)
        graph_dim = config['graph_feat_dim']
        treatment_dim = hidden  # treatment gets encoded to hidden dim

        # Age conditions each node
        self.age_node_mod = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(),
        )

        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Dropout(dropout),
        )

        # Channel positional embedding
        self.channel_pos_embed = nn.Embedding(12, hidden)

        # Graph feature encoder
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.graph_dim_out = hidden

        # Treatment encoder (NEW in V4)
        # Learned "unknown" token per treatment dimension — used when treatment_known_mask=0
        self.treatment_unknown = nn.Parameter(torch.zeros(self.N_TREATMENT_DIMS))
        nn.init.normal_(self.treatment_unknown, mean=0.5, std=0.1)

        self.treatment_encoder = nn.Sequential(
            nn.Linear(self.N_TREATMENT_DIMS, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )

        # Treatment-conditioned transformer layers
        self.layers = nn.ModuleList([
            TreatmentConditionedLayer(
                hidden, n_heads, n_cancer_types, self.graph_dim_out, treatment_dim, dropout
            )
            for _ in range(n_layers)
        ])

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

        self.sex_encoder = nn.Linear(1, hidden // 4)
        self.atlas_skip = nn.Sequential(nn.Linear(1, hidden // 4), nn.ELU())

        # Readout: transformer + sex + atlas + graph + treatment (direct paths)
        readout_in = hidden + hidden // 4 + hidden // 4 + self.graph_dim_out + hidden
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_treatment(self, treatment_vec, treatment_known_mask):
        """Encode treatment vector, replacing unknown dimensions with learned tokens."""
        # Where mask=0, use learned unknown token; where mask=1, use actual value
        filled = treatment_vec * treatment_known_mask + \
                 self.treatment_unknown.unsqueeze(0) * (1 - treatment_known_mask)
        return self.treatment_encoder(filled)

    def forward(self, node_features, node_mask, channel_pos_ids,
                cancer_type, clinical, atlas_sum, graph_features,
                treatment_vec, treatment_known_mask):
        """
        Args:
            node_features: (B, max_nodes, node_feat_dim)
            node_mask: (B, max_nodes) — 1 for real, 0 for pad
            channel_pos_ids: (B, max_nodes) — 0-11
            cancer_type: (B,) long
            clinical: (B, 2) — [age_z, sex]
            atlas_sum: (B, 1)
            graph_features: (B, graph_feat_dim)
            treatment_vec: (B, 11) — binary treatment flags
            treatment_known_mask: (B, 11) — 1=observed, 0=missing
        """
        B, N, _ = node_features.shape

        # Encode per-mutation nodes
        x = self.node_encoder(node_features)

        # Age conditions each node
        age = clinical[:, 0:1]
        age_mod = self.age_node_mod(age)
        x = x * (1 + age_mod.unsqueeze(1))

        # Channel-position embeddings
        x = x + self.channel_pos_embed(channel_pos_ids)

        # Encode graph structural features
        graph_context = self.graph_encoder(graph_features)

        # Encode treatment
        treatment_embed = self.encode_treatment(treatment_vec, treatment_known_mask)

        # Treatment-conditioned transformer
        attn_mask = (node_mask == 0)
        for layer in self.layers:
            x = layer(x, cancer_type, graph_context, treatment_embed, attn_mask)

        # Attention pooling
        attn_w = self.pool_attn(x)
        attn_w = attn_w.masked_fill(attn_mask.unsqueeze(-1), float('-inf'))
        attn_w = F.softmax(attn_w, dim=1)
        patient_embed = (x * attn_w).sum(dim=1)

        sex = clinical[:, 1:2]
        sex_feat = self.sex_encoder(sex)
        atlas_feat = self.atlas_skip(atlas_sum)

        # Readout: all context streams converge
        out = torch.cat([patient_embed, sex_feat, atlas_feat,
                         graph_context, treatment_embed], dim=-1)
        hazard = self.readout(out)

        return hazard

    def predict_counterfactual(self, node_features, node_mask, channel_pos_ids,
                                cancer_type, clinical, atlas_sum, graph_features,
                                treatment_options):
        """Run the same patient through multiple treatment options.

        Args:
            treatment_options: list of (treatment_vec, treatment_known_mask) tuples
                Each is (1, 11) shaped — one treatment scenario

        Returns:
            hazards: (n_options,) predicted log hazards for each treatment
        """
        hazards = []
        for t_vec, t_mask in treatment_options:
            h = self.forward(
                node_features, node_mask, channel_pos_ids,
                cancer_type, clinical, atlas_sum, graph_features,
                t_vec, t_mask,
            )
            hazards.append(h.squeeze())
        return torch.stack(hazards)
