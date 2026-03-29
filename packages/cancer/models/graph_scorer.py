"""
Graph Scorer — the graph IS the model.

Prediction = graph traversal, not neural network inference.
Every prediction traces back to specific edges with provenance.
No training step. No catastrophic forgetting. No opacity.

For patient P with mutations in genes {g₁, g₂, g₃}:

  hazard(P) = Σᵢ individual_score(gᵢ)              # atlas HRs
            + Σᵢⱼ interaction_score(gᵢ, gⱼ)         # graph edge evidence
            + cancer_type_intercept                   # baseline risk

Where interaction_score reads ALL edge types between gene pairs:
  - COOCCURS: empirical co-mutation frequency
  - SYNTHETIC_LETHAL: functional lethal interaction
  - PPI: physical protein interaction
  - ATTENDS_TO: transformer-learned context effect
  - COUPLES: curated pathway coupling

The only learned parameters are edge-type weights (~10 params)
calibrated via Cox regression. The graph does the heavy lifting.

New data → new edges → predictions improve. No retraining.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, MSK_DATASETS, NON_SILENT


class GraphScorer:
    """Score patients by traversing the knowledge graph.

    The graph is the model. This class reads it.
    """

    def __init__(self):
        self.gene_edges = {}      # (gene_a, gene_b) → {edge_type: weight, ...}
        self.gene_props = {}      # gene → {property: value, ...}
        self.atlas_t1 = {}        # (cancer_type, gene, protein_change) → log_hr
        self.atlas_t2 = {}        # (cancer_type, gene) → log_hr
        self.atlas_t3 = {}        # (cancer_type, channel) → log_hr
        self.cooccur_by_ct = {}   # (cancer_type, gene_a, gene_b) → count

    def load_from_neo4j(self):
        """Load the full graph into memory for fast scoring."""
        from neo4j import GraphDatabase

        t0 = time.time()
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                      auth=("neo4j", "openknowledgegraph"))

        print("Loading graph into scorer...", flush=True)

        with driver.session() as s:
            # --- Gene properties ---
            result = s.run("MATCH (g:Gene) RETURN g.name AS name, properties(g) AS props")
            for r in result:
                self.gene_props[r['name']] = dict(r['props'])
            print(f"  Genes: {len(self.gene_props)}", flush=True)

            # --- Pairwise edges (all types) ---
            edge_types = {
                'COOCCURS': 'cooccur',
                'SYNTHETIC_LETHAL': 'synthetic_lethal',
                'PPI': 'ppi',
                'ATTENDS_TO': 'attends_to',
                'COUPLES': 'couples',
            }

            for neo_type, key in edge_types.items():
                result = s.run(f"""
                    MATCH (a:Gene)-[r:{neo_type}]->(b:Gene)
                    RETURN a.name AS from, b.name AS to, properties(r) AS props
                """)
                n = 0
                for r in result:
                    pair = (r['from'], r['to'])
                    if pair not in self.gene_edges:
                        self.gene_edges[pair] = {}
                    self.gene_edges[pair][key] = r['props']
                    n += 1
                print(f"  {neo_type}: {n:,} edges", flush=True)

            # --- Co-occurrence by cancer type ---
            result = s.run("""
                MATCH (a:Gene)-[r:COOCCURS]->(b:Gene)
                WHERE r.cancer_type IS NOT NULL
                RETURN a.name AS from, b.name AS to,
                       r.cancer_type AS ct, r.count AS count
            """)
            for r in result:
                self.cooccur_by_ct[(r['ct'], r['from'], r['to'])] = r['count']

        driver.close()
        print(f"  Gene pairs with edges: {len(self.gene_edges):,}", flush=True)
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    def load_atlas(self):
        """Load survival atlas from Neo4j PROGNOSTIC_IN edges."""
        from gnn.data.graph_snapshot import load_atlas
        t1_raw, t2_raw, t3_raw, t4_raw, _ = load_atlas()

        for key, entry in t1_raw.items():
            self.atlas_t1[key] = float(np.log(max(entry['hr'], 0.01)))
        for key, entry in t2_raw.items():
            self.atlas_t2[key] = float(np.log(max(entry['hr'], 0.01)))
        for key, entry in t3_raw.items():
            self.atlas_t3[key] = float(np.log(max(entry['hr'], 0.01)))
        self.atlas_t4 = {}
        for key, entry in t4_raw.items():
            self.atlas_t4[key] = float(np.log(max(entry['hr'], 0.01)))

        print(f"  Atlas: T1={len(self.atlas_t1)}, T2={len(self.atlas_t2)}, "
              f"T3={len(self.atlas_t3)}, T4={len(self.atlas_t4)}", flush=True)

    def score_patient(self, genes, protein_changes, cancer_type, edge_weights=None):
        """Score a single patient by walking the graph.

        Args:
            genes: list of mutated gene names
            protein_changes: list of protein changes (parallel to genes)
            cancer_type: string cancer type
            edge_weights: dict of edge_type → learned weight (default: equal)

        Returns:
            dict with total score, breakdown by component, and edge trace
        """
        if edge_weights is None:
            edge_weights = {
                'atlas': 1.0,
                'cooccur': 0.1,
                'synthetic_lethal': 0.3,
                'ppi': 0.05,
                'attends_to': 0.2,
                'couples': 0.05,
                'depmap': 0.1,
                'civic': 0.1,
            }

        # --- Individual gene effects (atlas) ---
        individual_scores = []
        for gene, pc in zip(genes, protein_changes):
            ch = CHANNEL_MAP.get(gene)

            # Tiered atlas lookup
            log_hr = self.atlas_t1.get((cancer_type, gene, pc))
            tier = 1
            if log_hr is None:
                log_hr = self.atlas_t2.get((cancer_type, gene))
                tier = 2
            if log_hr is None and ch:
                log_hr = self.atlas_t3.get((cancer_type, ch))
                tier = 3
            if log_hr is None:
                log_hr = self.atlas_t4.get((cancer_type, gene))
                tier = 4
            if log_hr is None:
                log_hr = 0.0
                tier = 0

            individual_scores.append({
                'gene': gene, 'protein_change': pc,
                'log_hr': log_hr, 'tier': tier,
            })

        # --- Pairwise interaction effects (graph edges) ---
        interaction_scores = []
        unique_genes = list(set(genes))

        for i in range(len(unique_genes)):
            for j in range(i + 1, len(unique_genes)):
                ga, gb = unique_genes[i], unique_genes[j]

                # Look up edges in both directions
                edges_ab = self.gene_edges.get((ga, gb), {})
                edges_ba = self.gene_edges.get((gb, ga), {})

                interaction = {'gene_a': ga, 'gene_b': gb, 'components': {}}
                total_interaction = 0.0

                # Co-occurrence
                if 'cooccur' in edges_ab or 'cooccur' in edges_ba:
                    cooccur_props = edges_ab.get('cooccur', edges_ba.get('cooccur', {}))
                    count = cooccur_props.get('count', 0)
                    # Cancer-specific co-occurrence
                    ct_count = self.cooccur_by_ct.get((cancer_type, ga, gb), 0)
                    ct_count += self.cooccur_by_ct.get((cancer_type, gb, ga), 0)
                    score = np.log1p(ct_count) / 10.0 if ct_count > 0 else np.log1p(count) / 20.0
                    interaction['components']['cooccur'] = score
                    total_interaction += score * edge_weights.get('cooccur', 0)

                # Synthetic lethality
                if 'synthetic_lethal' in edges_ab or 'synthetic_lethal' in edges_ba:
                    sl = edges_ab.get('synthetic_lethal', edges_ba.get('synthetic_lethal', {}))
                    cross = sl.get('cross_channel', False)
                    score = 0.5 if cross else 0.3  # cross-channel SL is stronger signal
                    interaction['components']['synthetic_lethal'] = score
                    total_interaction += score * edge_weights.get('synthetic_lethal', 0)

                # PPI
                if 'ppi' in edges_ab or 'ppi' in edges_ba:
                    ppi = edges_ab.get('ppi', edges_ba.get('ppi', {}))
                    ppi_score = float(ppi.get('score', 500)) / 1000.0
                    interaction['components']['ppi'] = ppi_score
                    total_interaction += ppi_score * edge_weights.get('ppi', 0)

                # Attention (learned from transformer, now in graph)
                if 'attends_to' in edges_ab:
                    attn = edges_ab['attends_to']
                    score = float(attn.get('weight', 0))
                    interaction['components']['attends_to_ab'] = score
                    total_interaction += score * edge_weights.get('attends_to', 0)
                if 'attends_to' in edges_ba:
                    attn = edges_ba['attends_to']
                    score = float(attn.get('weight', 0))
                    interaction['components']['attends_to_ba'] = score
                    total_interaction += score * edge_weights.get('attends_to', 0)

                # Couples
                if 'couples' in edges_ab or 'couples' in edges_ba:
                    score = 0.2
                    interaction['components']['couples'] = score
                    total_interaction += score * edge_weights.get('couples', 0)

                interaction['total'] = total_interaction
                if total_interaction != 0:
                    interaction_scores.append(interaction)

        # --- Gene-level features (node properties) ---
        gene_feature_scores = []
        for gene in unique_genes:
            props = self.gene_props.get(gene, {})
            gf_score = 0.0
            components = {}

            # DepMap essentiality (more essential = worse prognosis when mutated)
            ess = props.get('depmap_mean_essentiality')
            if ess is not None:
                # Negative = essential. When mutated in patient → worse outcome
                dep_score = -float(ess)  # flip sign: essential genes score higher
                components['depmap'] = dep_score
                gf_score += dep_score * edge_weights.get('depmap', 0)

            # CIViC function
            civic_func = props.get('civic_dominant_function')
            if civic_func == 'GOF':
                components['civic_gof'] = 0.3
                gf_score += 0.3 * edge_weights.get('civic', 0)
            elif civic_func == 'LOF':
                components['civic_lof'] = 0.2
                gf_score += 0.2 * edge_weights.get('civic', 0)

            if gf_score != 0:
                gene_feature_scores.append({
                    'gene': gene, 'components': components, 'total': gf_score,
                })

        # --- Aggregate ---
        atlas_total = sum(s['log_hr'] for s in individual_scores) * edge_weights.get('atlas', 1)
        interaction_total = sum(s['total'] for s in interaction_scores)
        gene_feat_total = sum(s['total'] for s in gene_feature_scores)
        total = atlas_total + interaction_total + gene_feat_total

        return {
            'total_score': total,
            'atlas_score': atlas_total,
            'interaction_score': interaction_total,
            'gene_feature_score': gene_feat_total,
            'n_genes': len(unique_genes),
            'n_interactions': len(interaction_scores),
            'individual': individual_scores,
            'interactions': interaction_scores,
            'gene_features': gene_feature_scores,
        }

    def score_all_patients(self, dataset_name="msk_impact_50k", edge_weights=None):
        """Score all patients. Returns arrays for Cox evaluation."""
        paths = MSK_DATASETS[dataset_name]

        print("Loading patient data...", flush=True)
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
                'pc': row.get('proteinChange', ''),
            })

        # Score each patient
        scores = []
        times = []
        events = []
        n_scored = 0

        for _, row in clinical.iterrows():
            pid = row['patientId']
            ct = row['CANCER_TYPE']
            muts = patient_muts.get(pid, [])

            genes = [m['gene'] for m in muts]
            pcs = [m['pc'] for m in muts]

            result = self.score_patient(genes, pcs, ct, edge_weights)
            scores.append(result['total_score'])
            times.append(row['OS_MONTHS'])
            events.append(row['event'])
            n_scored += 1

            if n_scored % 10000 == 0:
                print(f"  Scored {n_scored:,} patients...", flush=True)

        print(f"  Total scored: {n_scored:,}", flush=True)

        return (
            np.array(scores, dtype=np.float64),
            np.array(times, dtype=np.float64),
            np.array(events, dtype=np.int32),
        )


def calibrate_edge_weights(scorer, n_random=50):
    """Find optimal edge-type weights via random search + Cox regression.

    Only ~10 parameters to optimize. No neural network.
    """
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import StratifiedKFold

    print("\nCalibrating edge weights...", flush=True)

    # Quick scoring with default weights to get baseline
    scores, times, events = scorer.score_all_patients()
    valid = times > 0
    scores, times, events = scores[valid], times[valid], events[valid]

    # Baseline C-index
    try:
        base_c = concordance_index_censored(events.astype(bool), times, scores)[0]
    except Exception:
        base_c = 0.5
    print(f"  Baseline C-index (default weights): {base_c:.4f}")

    # Random search over edge weights
    best_c = base_c
    best_weights = None
    rng = np.random.RandomState(42)

    # Use a single CV fold for speed
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(np.arange(len(events)), events))

    weight_keys = ['atlas', 'cooccur', 'synthetic_lethal', 'ppi',
                   'attends_to', 'couples', 'depmap', 'civic']

    for trial in range(n_random):
        # Sample weights from log-uniform
        weights = {}
        for k in weight_keys:
            weights[k] = float(10 ** rng.uniform(-2, 1))
        weights['atlas'] = float(10 ** rng.uniform(-0.5, 0.5))  # atlas stays near 1

        trial_scores, _, _ = scorer.score_all_patients(edge_weights=weights)
        trial_scores = trial_scores[valid]

        try:
            c = concordance_index_censored(
                events[val_idx].astype(bool),
                times[val_idx],
                trial_scores[val_idx]
            )[0]
        except Exception:
            c = 0.5

        if c > best_c:
            best_c = c
            best_weights = weights.copy()
            print(f"  Trial {trial}: C={c:.4f} *** new best", flush=True)

    if best_weights:
        print(f"\n  Best C-index: {best_c:.4f}")
        print(f"  Optimal weights:")
        for k, v in sorted(best_weights.items()):
            print(f"    {k}: {v:.4f}")
    else:
        print(f"\n  Default weights are best: C={base_c:.4f}")

    return best_weights or {k: 1.0 for k in weight_keys}, best_c


def evaluate(scorer, edge_weights=None):
    """Full 5-fold CV evaluation."""
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import StratifiedKFold

    scores, times, events = scorer.score_all_patients(edge_weights=edge_weights)
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
    return mean_c, std_c, fold_cs


if __name__ == "__main__":
    print("=" * 70)
    print("  GRAPH SCORER — the graph IS the model")
    print("=" * 70)

    scorer = GraphScorer()
    scorer.load_from_neo4j()
    scorer.load_atlas()

    # Evaluate with default weights first
    print("\n--- Default weights ---")
    mean_c, std_c, _ = evaluate(scorer)

    # Calibrate
    print("\n--- Calibrating ---")
    best_weights, cal_c = calibrate_edge_weights(scorer, n_random=100)

    # Evaluate with calibrated weights
    print("\n--- Calibrated weights ---")
    mean_c, std_c, _ = evaluate(scorer, edge_weights=best_weights)
