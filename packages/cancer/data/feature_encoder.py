"""
Encode per-gene mutation features into node feature vectors.

Feature vector (18-dim per gene):
  [0:8]  - Mutation type one-hot (8 categories)
  [8]    - VAF (variant allele frequency)
  [9]    - Hub/leaf position (1=hub, 0=leaf, 0.5=unclassified)
  [10:12] - GOF/LOF (2-dim: [is_GOF, is_LOF])
  [12:18] - Channel index one-hot (6 channels)

Non-mutated genes get a zero vector for mutation features but retain
structural features (position, channel).
"""

import torch
import numpy as np
import pandas as pd

from ..config import (
    ALL_GENES, GENE_TO_IDX, GENE_TO_CHANNEL_IDX,
    GENE_POSITION, GENE_FUNCTION, NON_SILENT, TRUNCATING,
    CHANNEL_NAMES, N_GENES,
)


# Mutation type categories (order matters for one-hot encoding)
MUTATION_TYPES = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Splice_Region",
    "Nonstop_Mutation",
    "Translation_Start_Site",
]
MUTTYPE_TO_IDX = {mt: i for i, mt in enumerate(MUTATION_TYPES)}


class MutationFeatureEncoder:
    """Encode patient mutations into a fixed 92-node feature tensor."""

    def __init__(self):
        # Precompute structural features for all 92 genes
        # These are the same for every patient
        self._structural = self._precompute_structural()

    def _precompute_structural(self):
        """Build structural feature vectors (position + channel) for all genes."""
        feats = torch.zeros(N_GENES, 8)  # position(1) + GOF/LOF(2) + channel(6) = 9, but we use 8 for channel+pos+gof
        # Actually let's be precise about dims:
        # We'll store: hub_leaf(1) + gof_lof(2) + channel_onehot(6) = 9 structural dims
        feats = torch.zeros(N_GENES, 9)
        for gene in ALL_GENES:
            idx = GENE_TO_IDX[gene]
            # Hub/leaf
            pos = GENE_POSITION.get(gene, "unclassified")
            feats[idx, 0] = 1.0 if pos == "hub" else (0.0 if pos == "leaf" else 0.5)
            # GOF/LOF
            func = GENE_FUNCTION.get(gene, "context")
            feats[idx, 1] = 1.0 if func == "GOF" else 0.0
            feats[idx, 2] = 1.0 if func == "LOF" else 0.0
            # Channel one-hot
            ch_idx = GENE_TO_CHANNEL_IDX[gene]
            feats[idx, 3 + ch_idx] = 1.0
        return feats

    def encode_patient(self, patient_mutations):
        """Encode a single patient's mutations into node features.

        Args:
            patient_mutations: DataFrame with columns:
                gene.hugoGeneSymbol, mutationType, tumorAltCount, tumorRefCount

        Returns:
            x: (N_GENES, 18) float tensor — node features for all 92 genes
            mutated_mask: (N_GENES,) bool tensor — which genes are mutated
        """
        # Start with zeros for mutation features (8 mut_type + 1 VAF = 9 mutation dims)
        mut_feats = torch.zeros(N_GENES, 9)
        mutated = torch.zeros(N_GENES, dtype=torch.bool)

        if patient_mutations is not None and len(patient_mutations) > 0:
            # Group by gene, take the "worst" mutation if multiple
            for gene, group in patient_mutations.groupby("gene.hugoGeneSymbol"):
                if gene not in GENE_TO_IDX:
                    continue
                idx = GENE_TO_IDX[gene]
                mutated[idx] = True

                # Use the most damaging mutation for this gene
                row = self._select_driver_mutation(group)

                # Mutation type one-hot
                mt = row.get("mutationType", "")
                if mt in MUTTYPE_TO_IDX:
                    mut_feats[idx, MUTTYPE_TO_IDX[mt]] = 1.0

                # VAF
                alt = pd.to_numeric(row.get("tumorAltCount", 0), errors="coerce")
                ref = pd.to_numeric(row.get("tumorRefCount", 0), errors="coerce")
                if pd.notna(alt) and pd.notna(ref) and (alt + ref) > 0:
                    mut_feats[idx, 8] = alt / (alt + ref)

        # Concatenate: mutation features (9) + structural features (9) = 18
        x = torch.cat([mut_feats, self._structural], dim=1)  # (N_GENES, 18)
        return x, mutated

    def _select_driver_mutation(self, group):
        """From multiple mutations in the same gene, pick the most damaging."""
        # Priority: truncating > missense hotspot > any non-silent
        truncating = group[group["mutationType"].isin(TRUNCATING)]
        if len(truncating) > 0:
            return truncating.iloc[0]
        return group.iloc[0]
