"""
Cell Line Dataset — encode DepMap cell lines as AtlasDataset-compatible tensors.

Each cell line's mutations (from GDSC) become nodes with the same feature layout
as patient mutations. Essentiality scores (from DepMap CRISPR) serve as targets.

The backbone sees identical tensor shapes whether it's looking at a cell line
or a patient. Only the head differs: essentiality prediction vs survival.

Data sources:
  - depmap_dependency_cell_lines.csv: 1,186 cell lines × 502 genes (Chronos scores)
  - Model.csv: cell line metadata (OncotreeLineage → cancer_type mapping)
  - gdsc_cell_line_mutations.csv: 10,677 variant-level mutations across 926 cell lines
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, ALL_GENES
from gnn.data.atlas_dataset import (
    MAX_NODES, NODE_FEAT_DIM, MUTATION_ONLY,
    make_node_features, get_channel_pos_id, parse_position,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEPMAP_DIR = os.path.join(CACHE_DIR, "depmap")
GDSC_DIR = os.path.join(CACHE_DIR, "gdsc")

# Lineage → synthetic cancer type index mapping.
# These don't need to match patient cancer types exactly;
# they just need to give the cancer_type embedding something
# tissue-appropriate to condition on.
LINEAGE_TO_CANCER_TYPE = {
    "Lung": "Non-Small Cell Lung Cancer",
    "Breast": "Breast Cancer",
    "Bowel": "Colorectal Cancer",
    "Ovary/Fallopian Tube": "Ovarian Cancer",
    "Skin": "Melanoma",
    "CNS/Brain": "Glioma",
    "Pancreas": "Pancreatic Cancer",
    "Lymphoid": "Diffuse Large B-Cell Lymphoma",
    "Myeloid": "Acute Myeloid Leukemia",
    "Head and Neck": "Head and Neck Cancer",
    "Esophagus/Stomach": "Esophagogastric Cancer",
    "Kidney": "Renal Cell Carcinoma",
    "Liver": "Hepatocellular Carcinoma",
    "Bladder/Urinary Tract": "Bladder Cancer",
    "Uterus": "Endometrial Cancer",
    "Bone": "Bone Cancer",
    "Prostate": "Prostate Cancer",
    "Soft Tissue": "Soft Tissue Sarcoma",
    "Thyroid": "Thyroid Cancer",
    "Peripheral Nervous System": "Neuroblastoma",
    "Fibroblast": "Cancer of Unknown Primary",
    "Bile Duct": "Cholangiocarcinoma",
    "Cervix": "Cervical Cancer",
    "Vulva/Vagina": "Cervical Cancer",
    "Pleura": "Mesothelioma",
    "Adrenal Gland": "Adrenocortical Carcinoma",
    "Testis": "Germ Cell Tumor",
    "Ampulla of Vater": "Ampullary Carcinoma",
    "Eye": "Uveal Melanoma",
}


class CellLineDataset:
    """Convert DepMap cell lines to AtlasTransformer tensor format.

    Each cell line becomes a sample with:
      - node_features: (MAX_NODES, NODE_FEAT_DIM) from GDSC mutations
      - essentiality targets: Chronos scores for the cell line's mutated genes
      - Same edge_features, block_ids, channel_ids as patient data
    """

    def __init__(self):
        print("Loading DepMap + GDSC data...", flush=True)

        # DepMap essentiality scores
        self.dep = pd.read_csv(os.path.join(DEPMAP_DIR, "depmap_dependency_cell_lines.csv"))
        self.dep = self.dep.rename(columns={self.dep.columns[0]: "ModelID"})
        dep_genes = [c for c in self.dep.columns if c not in ("ModelID", "lineage")]
        self.dep_genes = set(dep_genes)
        print(f"  DepMap: {len(self.dep)} cell lines × {len(dep_genes)} genes", flush=True)

        # Model metadata
        self.model_meta = pd.read_csv(os.path.join(DEPMAP_DIR, "Model.csv"))
        self.name_to_model = dict(zip(self.model_meta["CellLineName"], self.model_meta["ModelID"]))

        # GDSC mutations
        self.gdsc_muts = pd.read_csv(os.path.join(GDSC_DIR, "gdsc_cell_line_mutations.csv"))
        self.gdsc_muts["ModelID"] = self.gdsc_muts["cell_line"].map(self.name_to_model)
        self.gdsc_muts = self.gdsc_muts.dropna(subset=["ModelID"])
        # Keep only genes in our project
        self.gdsc_muts = self.gdsc_muts[self.gdsc_muts["gene"].isin(set(ALL_GENES))]
        print(f"  GDSC mutations (mapped): {len(self.gdsc_muts)} across "
              f"{self.gdsc_muts['ModelID'].nunique()} cell lines", flush=True)

        # Cell lines with both mutations and essentiality
        dep_ids = set(self.dep["ModelID"])
        gdsc_ids = set(self.gdsc_muts["ModelID"])
        self.valid_ids = sorted(dep_ids & gdsc_ids)
        print(f"  Cell lines with mutations + essentiality: {len(self.valid_ids)}", flush=True)

    def build_features(self, cancer_type_map=None):
        """Build tensors in the same format as AtlasDataset.build_features().

        Args:
            cancer_type_map: dict mapping cancer type name → int index.
                If provided, maps lineages to existing patient cancer type indices.
                If None, builds its own mapping.

        Returns:
            dict with same keys as AtlasDataset plus essentiality targets.
        """
        print("Encoding cell lines as mutation-level nodes...", flush=True)

        # Load graph data for GOF/LOF
        gof_lof = self._load_gof_lof()

        # Index DepMap scores by ModelID
        dep_indexed = self.dep.set_index("ModelID")

        # Group GDSC mutations by ModelID
        mut_groups = self.gdsc_muts.groupby("ModelID")

        # Lineage lookup
        lineage_map = dict(zip(self.model_meta["ModelID"], self.model_meta["OncotreeLineage"]))
        sex_map = dict(zip(self.model_meta["ModelID"], self.model_meta["Sex"]))

        if cancer_type_map is None:
            cancer_type_map = {}
            ct_idx = 0
        else:
            cancer_type_map = dict(cancer_type_map)
            ct_idx = max(cancer_type_map.values()) + 1 if cancer_type_map else 0

        all_node_feats = []
        all_node_masks = []
        all_channel_pos_ids = []
        all_atlas_sums = []
        all_gene_names = []
        all_cancer_types = []
        all_ages = []
        all_sexes = []

        # Essentiality targets: per node, the Chronos score for that gene
        all_essentiality = []
        all_essentiality_masks = []

        skipped = 0

        for model_id in self.valid_ids:
            if model_id not in mut_groups.groups:
                skipped += 1
                continue

            muts = mut_groups.get_group(model_id)
            lineage = lineage_map.get(model_id, "Unknown")
            sex = sex_map.get(model_id, "Unknown")

            # Map lineage to cancer type
            ct_name = LINEAGE_TO_CANCER_TYPE.get(lineage, "Cancer of Unknown Primary")
            if ct_name not in cancer_type_map:
                cancer_type_map[ct_name] = ct_idx
                ct_idx += 1

            # Build node features for each mutation
            nodes = []
            cp_ids = []
            genes = []
            essentiality_scores = []
            essentiality_valid = []

            for _, row in muts.iterrows():
                gene = row["gene"]
                pc = row["mutation"] if row["mutation"] != "SNV" else None

                if gene not in CHANNEL_MAP:
                    continue

                # Node features: zero out patient-specific fields
                feat = make_node_features(
                    gene=gene,
                    pc=pc,
                    hr=1.0,  # no patient HR data
                    ci_width=1.0,
                    tier=2,  # default tier
                    n_with=100,
                    biallelic=False,
                    expression_context=None,
                    gof_lof=gof_lof.get(gene, 0),
                )
                # Zero out patient-specific features explicitly
                feat[0] = 0.0   # log_hr
                feat[1] = 0.0   # ci_width
                feat[12] = 0.0  # log_n_patients

                nodes.append(feat)
                cp_ids.append(get_channel_pos_id(gene) if not MUTATION_ONLY else 0)
                genes.append(gene)

                # Essentiality target for this gene
                if gene in self.dep_genes and model_id in dep_indexed.index:
                    score = dep_indexed.at[model_id, gene]
                    if pd.notna(score):
                        essentiality_scores.append(float(score))
                        essentiality_valid.append(1.0)
                    else:
                        essentiality_scores.append(0.0)
                        essentiality_valid.append(0.0)
                else:
                    essentiality_scores.append(0.0)
                    essentiality_valid.append(0.0)

            if len(nodes) == 0:
                skipped += 1
                continue

            n_nodes = len(nodes)

            # Truncate to MAX_NODES (keep diverse channel representation)
            if n_nodes > MAX_NODES:
                indices = list(range(n_nodes))
                np.random.shuffle(indices)
                indices = sorted(indices[:MAX_NODES])
                nodes = [nodes[i] for i in indices]
                cp_ids = [cp_ids[i] for i in indices]
                genes = [genes[i] for i in indices]
                essentiality_scores = [essentiality_scores[i] for i in indices]
                essentiality_valid = [essentiality_valid[i] for i in indices]
                n_nodes = MAX_NODES

            mask = [1] * n_nodes
            ess = list(essentiality_scores)
            ess_mask = list(essentiality_valid)

            # Pad to MAX_NODES
            while len(nodes) < MAX_NODES:
                nodes.append(np.zeros(NODE_FEAT_DIM, dtype=np.float32))
                cp_ids.append(0)
                genes.append("")
                mask.append(0)
                ess.append(0.0)
                ess_mask.append(0.0)

            all_node_feats.append(np.stack(nodes))
            all_node_masks.append(mask)
            all_channel_pos_ids.append(cp_ids)
            all_atlas_sums.append(0.0)  # no atlas sum for cell lines
            all_gene_names.append(genes)
            all_cancer_types.append(cancer_type_map[ct_name])
            all_ages.append(0.0)  # no patient age
            all_sexes.append(1.0 if sex == "Male" else 0.0)
            all_essentiality.append(ess)
            all_essentiality_masks.append(ess_mask)

        if skipped:
            print(f"  Skipped {skipped} cell lines (no valid mutations)", flush=True)

        n_samples = len(all_node_feats)
        print(f"  Final samples: {n_samples}", flush=True)

        node_feats = torch.tensor(np.stack(all_node_feats), dtype=torch.float32)
        node_masks = torch.tensor(all_node_masks, dtype=torch.float32)
        channel_pos_ids = torch.tensor(all_channel_pos_ids, dtype=torch.long)
        atlas_sums = torch.tensor(all_atlas_sums, dtype=torch.float32).unsqueeze(1)
        cancer_types = torch.tensor(all_cancer_types, dtype=torch.long)
        ages = torch.tensor(all_ages, dtype=torch.float32)
        sexes = torch.tensor(all_sexes, dtype=torch.float32)
        essentiality = torch.tensor(all_essentiality, dtype=torch.float32)
        essentiality_masks = torch.tensor(all_essentiality_masks, dtype=torch.float32)

        nodes_per = node_masks.sum(dim=1)
        valid_ess = essentiality_masks.sum()
        print(f"  Node features: {node_feats.shape}", flush=True)
        print(f"  Mean nodes/cell line: {nodes_per.mean():.1f}", flush=True)
        print(f"  Valid essentiality targets: {int(valid_ess.item())}", flush=True)
        print(f"  Mean essentiality (valid): {essentiality[essentiality_masks.bool()].mean():.3f}", flush=True)

        return {
            "node_features": node_feats,
            "node_masks": node_masks,
            "channel_pos_ids": channel_pos_ids,
            "atlas_sums": atlas_sums,
            "gene_names": all_gene_names,
            "cancer_types": cancer_types,
            "ages": ages,
            "sexes": sexes,
            "n_cancer_types": ct_idx,
            "cancer_type_map": cancer_type_map,
            # Pre-training targets
            "essentiality": essentiality,       # (N_samples, MAX_NODES)
            "essentiality_masks": essentiality_masks,  # (N_samples, MAX_NODES)
        }

    def build_v5_features(self, schema=None, cancer_type_map=None):
        """Build V5-compatible features with edge features and block/channel IDs.

        Same as AtlasDataset.build_v5_features() but for cell lines.
        """
        if schema is None:
            from gnn.data.graph_schema import get_schema
            schema = get_schema()

        base = self.build_features(cancer_type_map=cancer_type_map)

        gene_names = base["gene_names"]
        B = len(gene_names)
        N = MAX_NODES

        print(f"\nBuilding V5 features (schema: {schema.edge_feature_dim} edge dims, "
              f"{schema.node_extra_dim} node extra dims)...", flush=True)

        # Enriched node features
        all_extra = np.zeros((B, N, schema.node_extra_dim), dtype=np.float32)
        all_block_ids = np.full((B, N), schema.n_blocks, dtype=np.int64)
        all_channel_ids = np.full((B, N), schema.n_channels, dtype=np.int64)

        for b in range(B):
            for n in range(N):
                gene = gene_names[b][n]
                if gene and gene != "":
                    all_extra[b, n] = schema.get_node_extra_features(gene)
                    bid = schema.get_block_id(gene)
                    if bid >= 0:
                        all_block_ids[b, n] = bid
                    cid = schema.get_channel_id(gene)
                    if cid < schema.n_channels:
                        all_channel_ids[b, n] = cid

        atlas_feats = base["node_features"].numpy()
        enriched = np.concatenate([atlas_feats, all_extra], axis=-1)
        print(f"  Enriched node features: {enriched.shape}", flush=True)

        # Pairwise edge features
        all_edges = np.zeros((B, N, N, schema.edge_feature_dim), dtype=np.float32)

        for b in range(B):
            genes = gene_names[b]
            for i in range(N):
                gi = genes[i]
                if not gi or gi == "":
                    continue
                for j in range(i + 1, N):
                    gj = genes[j]
                    if not gj or gj == "":
                        continue
                    feat = schema.get_edge_feature_vector(gi, gj)
                    all_edges[b, i, j] = feat
                    all_edges[b, j, i] = feat

        # NaN guard
        for arr, name in [(all_edges, "edge"), (all_extra, "node extra")]:
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                print(f"  WARNING: {nan_count} NaN in {name} features, replacing with 0", flush=True)
                np.nan_to_num(arr, copy=False, nan=0.0)

        if np.isnan(all_extra).any():
            enriched = np.concatenate([atlas_feats, np.nan_to_num(all_extra, nan=0.0)], axis=-1)

        print(f"  Edge features: {all_edges.shape}", flush=True)

        result = dict(base)
        result["node_features"] = torch.tensor(enriched, dtype=torch.float32)
        result["edge_features"] = torch.tensor(all_edges, dtype=torch.float32)
        result["block_ids"] = torch.tensor(all_block_ids, dtype=torch.long)
        result["channel_ids"] = torch.tensor(all_channel_ids, dtype=torch.long)
        result["schema"] = schema
        return result

    def _load_gof_lof(self):
        """Load GOF/LOF from graph config cache."""
        from gnn.config import GENE_FUNCTION
        gof_lof = {}
        for gene, func in GENE_FUNCTION.items():
            if func == "GOF":
                gof_lof[gene] = 1
            elif func == "LOF":
                gof_lof[gene] = -1
        return gof_lof


if __name__ == "__main__":
    ds = CellLineDataset()
    data = ds.build_features()
    print("\nDone. Summary:")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} entries")
        elif isinstance(v, list):
            print(f"  {k}: list of {len(v)}")
        else:
            print(f"  {k}: {v}")
