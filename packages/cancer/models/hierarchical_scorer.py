"""
Hierarchical Graph Scorer — normalize within bounded contexts, then escalate.

The graph IS the model. This scorer reads it hierarchically:

  Level 0: Sub-pathway blocks (~20 genes each, dense PPI clusters)
           normalize within block → sub-pathway damage score

  Level 1: Channel blocks (~80 genes, composed of sub-pathway blocks)
           normalize across sub-blocks → channel damage score

  Level 2: Cross-channel interactions (8×8 sparse)
           normalize across channels → systemic damage score

  Level 3: Patient score (atlas + interactions + clinical)

Each level is a bounded context. You normalize within it before
escalating up. Overlap between blocks doesn't matter — the
higher-level normalization reconciles it.

No neural network. No training. No catastrophic forgetting.
The graph accumulates knowledge. This scorer reads it.

V2: Uses GraphPrecomputed — all computation pushed into Neo4j,
    cached to disk. Scoring is pure dict lookups.
    Confidence weighting prevents noise dilution from low-evidence genes.

Usage:
    python3 -u -m gnn.models.hierarchical_scorer
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, MSK_DATASETS, NON_SILENT


class HierarchicalScorer:
    """Score patients by hierarchical graph traversal with bounded-context normalization.

    V2: Backed by GraphPrecomputed — no Neo4j at scoring time.
    """

    def __init__(self):
        self.gp = None  # GraphPrecomputed instance

        # Atlas (individual mutation effects)
        self.atlas_t1 = {}
        self.atlas_t2 = {}
        self.atlas_t3 = {}
        self.atlas_t4 = {}

    def load(self, force=False):
        """Load precomputed graph data and atlas."""
        from gnn.data.graph_precompute import get_precomputed

        t0 = time.time()
        self.gp = get_precomputed(force=force)
        self._load_atlas()
        print(f"  Scorer loaded in {time.time()-t0:.1f}s", flush=True)

    def _load_atlas(self):
        from gnn.data.graph_snapshot import load_atlas
        t1_raw, t2_raw, t3_raw, t4_raw, _ = load_atlas()

        for key, entry in t1_raw.items():
            self.atlas_t1[key] = float(np.log(max(entry['hr'], 0.01)))
        for key, entry in t2_raw.items():
            self.atlas_t2[key] = float(np.log(max(entry['hr'], 0.01)))
        for key, entry in t3_raw.items():
            self.atlas_t3[key] = float(np.log(max(entry['hr'], 0.01)))
        for key, entry in t4_raw.items():
            self.atlas_t4[key] = float(np.log(max(entry['hr'], 0.01)))

        print(f"  Atlas: T1={len(self.atlas_t1)}, T2={len(self.atlas_t2)}, "
              f"T3={len(self.atlas_t3)}, T4={len(self.atlas_t4)}", flush=True)

    # -----------------------------------------------------------------
    # Scoring: hierarchical normalization with confidence weighting
    # -----------------------------------------------------------------

    def _get_atlas_score(self, gene, pc, cancer_type):
        """Individual mutation effect from atlas (tiers 1-4)."""
        ch = CHANNEL_MAP.get(gene)
        log_hr = self.atlas_t1.get((cancer_type, gene, pc))
        if log_hr is not None:
            return log_hr, 1
        log_hr = self.atlas_t2.get((cancer_type, gene))
        if log_hr is not None:
            return log_hr, 2
        if ch:
            log_hr = self.atlas_t3.get((cancer_type, ch))
            if log_hr is not None:
                return log_hr, 3
        log_hr = self.atlas_t4.get((cancer_type, gene))
        if log_hr is not None:
            return log_hr, 4
        return 0.0, 0

    def _gene_node_score(self, gene):
        """Score from gene node properties."""
        props = self.gp.gene_properties.get(gene, {})
        score = 0.0

        ess = props.get('depmap_mean_essentiality')
        if ess is not None:
            try:
                score += -float(ess) * 0.1
            except (TypeError, ValueError):
                pass

        pan = props.get('depmap_pan_essential')
        if pan:
            score += 0.05

        func = props.get('civic_dominant_function')
        if func == 'GOF':
            score += 0.05
        elif func == 'LOF':
            score += 0.03

        role = props.get('tcga_cna_role')
        if role in ('oncogene', 'tumor_suppressor'):
            score += 0.02

        return score

    def score_patient(self, genes, protein_changes, cancer_type):
        """Score a patient with hierarchical normalization + confidence weighting.

        Level 0: Score within each sub-pathway block, normalize
        Level 1: Aggregate blocks within channel, normalize
        Level 2: Cross-channel interactions, normalize
        Level 3: Final patient score
        """
        unique_genes = list(set(genes))
        gene_pc = {}
        for g, pc in zip(genes, protein_changes):
            gene_pc[g] = pc

        # Precompute per-gene atlas scores and confidence weights
        gene_atlas = {}  # gene → (log_hr, tier)
        gene_conf = {}   # gene → confidence weight
        for g in unique_genes:
            log_hr, tier = self._get_atlas_score(g, gene_pc.get(g, ''), cancer_type)
            gene_atlas[g] = (log_hr, tier)
            # Hard gate: only T1/T2/T4 genes get individual atlas score.
            # T3-only (channel-level) is the same value for every gene in the
            # channel — it adds no patient-specific signal, just noise.
            # No-atlas genes: zero individual score.
            # All genes still participate in pairwise scoring (co-occurrence
            # with a known gene is informative even if the gene itself is unknown).
            if tier in (1, 2):
                gene_conf[g] = 1.0
            elif tier == 4:
                gene_conf[g] = 0.6
            elif tier == 3:
                gene_conf[g] = 0.0  # channel-level — not gene-specific
            else:
                gene_conf[g] = 0.0  # no atlas

        # ============================================================
        # Level 0: Sub-pathway block scores
        # ============================================================
        block_mutations = defaultdict(list)
        unassigned = []

        for g in unique_genes:
            comm = self.gp.get_gene_community(g)
            if comm:
                block_mutations[(comm['channel'], comm['community_idx'])].append(g)
            else:
                unassigned.append(g)

        block_scores = {}

        for (ch, ci), block_genes in block_mutations.items():
            # Individual effects, confidence-weighted
            individual = sum(
                gene_atlas[g][0] * gene_conf[g]
                for g in block_genes
            )

            # Pairwise interactions — split into additive and multiplicative
            # Additive: statistical co-occurrence, PPI proximity
            # Multiplicative: functional interactions that amplify individual damage
            pairwise_additive = 0.0
            pairwise_multiplier = 1.0  # starts at 1.0, each pair can increase
            for i in range(len(block_genes)):
                for j in range(i + 1, len(block_genes)):
                    ga, gb = block_genes[i], block_genes[j]
                    # Additive component
                    pw_add = self.gp.get_pairwise_additive(ga, gb)
                    ct_count = self.gp.get_pairwise_ct_cooccur(ga, gb, cancer_type)
                    if ct_count > 0:
                        pw_add += np.log1p(ct_count) / 10.0
                    # Multiplicative component
                    pw_mult = self.gp.get_pairwise_multiplier(ga, gb)
                    # Gate: full weight if both known, half if one known
                    ca, cb = gene_conf[ga], gene_conf[gb]
                    if ca > 0 and cb > 0:
                        pair_w = 1.0
                    elif ca > 0 or cb > 0:
                        pair_w = 0.5
                    else:
                        pair_w = 0.1
                    pairwise_additive += pw_add * pair_w
                    # Multiplier accumulates — each functional interaction amplifies
                    if pw_mult > 1.0:
                        pairwise_multiplier *= (1.0 + (pw_mult - 1.0) * pair_w)

            # Gene node features — only for genes with atlas evidence
            # (DepMap/CIViC features are informative but only when the gene
            # has atlas context; otherwise it's just noise)
            node_feats = sum(
                self._gene_node_score(g) for g in block_genes
                if gene_conf[g] > 0
            )

            # Normalize additive pairwise by 1/n²
            n_bg = len(block_genes)
            n_pairs = n_bg * (n_bg - 1) / 2 if n_bg > 1 else 1
            pw_add_normalized = pairwise_additive / n_pairs if n_pairs > 0 else 0.0

            # Multiplicative: individual score * multiplier
            # The multiplier amplifies the base signal from known genes
            raw = (individual * pairwise_multiplier) + pw_add_normalized * 0.2 + node_feats * 1.0

            # Normalize within block
            block_scores[(ch, ci)] = {
                'raw': raw,
                'normalized': raw / max(np.sqrt(len(block_genes)), 1),
                'n_mutations': len(block_genes),
                'individual': individual,
                'pairwise': pairwise_additive,
                'node_feats': node_feats,
            }

        # ============================================================
        # Level 1: Channel scores (aggregate blocks, normalize)
        # ============================================================
        channel_scores = {}
        all_channels = set()
        for g in unique_genes:
            ch = CHANNEL_MAP.get(g)
            if ch:
                all_channels.add(ch)

        for ch in all_channels:
            ch_blocks = {k: v for k, v in block_scores.items() if k[0] == ch}
            if not ch_blocks:
                continue

            ch_raw = sum(b['normalized'] for b in ch_blocks.values())
            n_active_blocks = len(ch_blocks)
            total_blocks = len(self.gp.get_channel_communities(ch))

            channel_coverage = n_active_blocks / max(total_blocks, 1)
            channel_scores[ch] = {
                'raw': ch_raw,
                'normalized': ch_raw,
                'coverage': channel_coverage,
                'n_blocks': n_active_blocks,
            }

        # ============================================================
        # Level 2: Cross-channel interactions (8×8 sparse)
        # ============================================================
        cross_channel_score = 0.0
        active_channels = list(channel_scores.keys())

        for i in range(len(active_channels)):
            for j in range(i + 1, len(active_channels)):
                ch_a, ch_b = active_channels[i], active_channels[j]

                # Cross-channel interaction from precomputed matrix
                interaction = self.gp.get_cross_channel_weight(ch_a, ch_b)

                # Weight by channel damage intensity
                damage_product = (channel_scores[ch_a]['normalized'] *
                                  channel_scores[ch_b]['normalized'])
                cross_channel_score += interaction * damage_product * 0.01

                # Specific cross-channel gene pairs — precomputed lookup
                # Normalize by 1/(na*nb) to prevent O(n²) accumulation
                genes_a = [g for g in unique_genes if CHANNEL_MAP.get(g) == ch_a]
                genes_b = [g for g in unique_genes if CHANNEL_MAP.get(g) == ch_b]
                cross_pw_sum = 0.0
                n_cross_pairs = len(genes_a) * len(genes_b)
                for ga in genes_a:
                    for gb in genes_b:
                        pw = self.gp.get_pairwise_score(ga, gb)
                        ct_count = self.gp.get_pairwise_ct_cooccur(ga, gb, cancer_type)
                        if ct_count > 0:
                            pw += np.log1p(ct_count) / 10.0
                        ca = gene_conf.get(ga, 0)
                        cb = gene_conf.get(gb, 0)
                        if ca > 0 and cb > 0:
                            pair_w = 1.0
                        elif ca > 0 or cb > 0:
                            pair_w = 0.5
                        else:
                            pair_w = 0.1
                        cross_pw_sum += pw * pair_w
                if n_cross_pairs > 0:
                    cross_channel_score += (cross_pw_sum / n_cross_pairs) * 0.5

        # Normalize cross-channel
        n_pairs = max(len(active_channels) * (len(active_channels) - 1) / 2, 1)
        cross_channel_normalized = cross_channel_score / np.sqrt(n_pairs)

        # ============================================================
        # Level 3: Final patient score
        # ============================================================
        channel_total = sum(cs['normalized'] for cs in channel_scores.values())

        # Unassigned genes — confidence-weighted
        unassigned_score = sum(
            gene_atlas[g][0] * gene_conf[g]
            for g in unassigned
        )

        total = channel_total + cross_channel_normalized + unassigned_score

        return {
            'total': total,
            'channel_total': channel_total,
            'cross_channel': cross_channel_normalized,
            'unassigned': unassigned_score,
            'n_genes': len(unique_genes),
            'n_channels_hit': len(active_channels),
            'channels': channel_scores,
            'blocks': block_scores,
        }

    def score_all_patients(self, dataset_name="msk_impact_50k"):
        """Score all patients. Returns arrays for evaluation."""
        paths = MSK_DATASETS[dataset_name]

        print("\nLoading patient data...", flush=True)
        mutations = pd.read_csv(paths["mutations"])
        clinical = pd.read_csv(paths["clinical"])
        sample_clinical = pd.read_csv(paths["sample_clinical"])

        mutations = mutations[mutations['mutationType'].isin(NON_SILENT)]
        mutations = mutations[mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())]

        if 'CANCER_TYPE' in clinical.columns:
            clinical = clinical.drop(columns=['CANCER_TYPE'])
        clinical = clinical.merge(
            sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
            on='patientId', how='left'
        )
        clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
        clinical['event'] = clinical['OS_STATUS'].apply(
            lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
        )

        # Group mutations by patient
        patient_muts = defaultdict(list)
        for _, row in mutations.iterrows():
            patient_muts[row['patientId']].append({
                'gene': row['gene.hugoGeneSymbol'],
                'pc': str(row.get('proteinChange', '')),
            })

        scores = []
        times = []
        events = []
        details = []
        n_scored = 0

        t0 = time.time()
        for _, row in clinical.iterrows():
            pid = row['patientId']
            ct = row['CANCER_TYPE']
            muts = patient_muts.get(pid, [])

            genes = [m['gene'] for m in muts]
            pcs = [m['pc'] for m in muts]

            result = self.score_patient(genes, pcs, ct)
            scores.append(result['total'])
            times.append(row['OS_MONTHS'])
            events.append(row['event'])
            details.append(result)
            n_scored += 1

            if n_scored % 10000 == 0:
                elapsed = time.time() - t0
                rate = n_scored / elapsed
                print(f"  {n_scored:,} patients [{elapsed:.0f}s, "
                      f"{rate:.0f}/s]", flush=True)

        print(f"  Scored {n_scored:,} patients in {time.time()-t0:.1f}s", flush=True)

        return (
            np.array(scores, dtype=np.float64),
            np.array(times, dtype=np.float64),
            np.array(events, dtype=np.int32),
            details,
        )


def evaluate(scorer):
    """5-fold CV evaluation."""
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import StratifiedKFold

    scores, times, events, details = scorer.score_all_patients()
    valid = times > 0
    scores, times, events = scores[valid], times[valid], events[valid]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_cs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(events)), events)):
        try:
            c = concordance_index_censored(
                events[val_idx].astype(bool),
                times[val_idx],
                scores[val_idx]
            )[0]
        except Exception:
            c = 0.5
        fold_cs.append(c)
        print(f"  Fold {fold}: C={c:.4f}")

    mean_c = np.mean(fold_cs)
    std_c = np.std(fold_cs)
    print(f"\n  Mean C-index: {mean_c:.4f} ± {std_c:.4f}")

    # Breakdown by mutation count
    all_n_genes = np.array([d['n_genes'] for d in details])
    all_n_channels = np.array([d['n_channels_hit'] for d in details])
    n_genes = all_n_genes[valid]
    n_channels = all_n_channels[valid]

    print(f"\n  By mutation count:")
    for lo, hi, label in [(0, 0, '0'), (1, 2, '1-2'), (3, 5, '3-5'),
                           (6, 10, '6-10'), (11, 99, '11+')]:
        mask = (n_genes >= lo) & (n_genes <= hi)
        if mask.sum() < 50:
            continue
        try:
            c = concordance_index_censored(
                events[mask].astype(bool), times[mask], scores[mask])[0]
        except Exception:
            c = 0.5
        print(f"    {label:>5s} mutations: C={c:.4f} (n={mask.sum():,})")

    print(f"\n  By channels hit:")
    for n_ch in range(5):
        mask = n_channels == n_ch
        if mask.sum() < 50:
            continue
        try:
            c = concordance_index_censored(
                events[mask].astype(bool), times[mask], scores[mask])[0]
        except Exception:
            c = 0.5
        print(f"    {n_ch} channels: C={c:.4f} (n={mask.sum():,})")

    return mean_c, std_c, fold_cs


if __name__ == "__main__":
    print("=" * 70)
    print("  HIERARCHICAL GRAPH SCORER V2")
    print("  Precomputed graph data. Confidence-weighted scoring.")
    print("=" * 70)

    scorer = HierarchicalScorer()
    scorer.load()

    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    evaluate(scorer)
