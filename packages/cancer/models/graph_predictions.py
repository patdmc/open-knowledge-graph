"""
Multi-target graph prediction — validate graph knowledge beyond survival.

Each prediction target has:
  - A holdback set (data never seen during graph construction for that target)
  - A contamination map (which edge types leak information about this target)
  - A scoring protocol

If optimizing these targets also improves C-index, the graph captures
causal structure. If one improves and C doesn't follow, we've found
where the causal model breaks.

Targets:
  1. Co-mutation: hold out one mutation per patient, predict from graph
  2. Cancer type: given mutations only, predict cancer type
  3. Drug response: given mutations, predict drug sensitivity (external validation)
  4. Synthetic lethality: graph SL edges vs DepMap essentiality (external)
  5. Survival C-index: existing target (for correlation tracking)

Usage:
    python3 -u -m gnn.models.graph_predictions --target co_mutation
    python3 -u -m gnn.models.graph_predictions --target cancer_type
    python3 -u -m gnn.models.graph_predictions --target all
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from gnn.config import CHANNEL_MAP, ANALYSIS_CACHE


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "graph_predictions")


def _neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver("bolt://localhost:7687",
                                auth=("neo4j", "openknowledgegraph"))


# ======================================================================
# Holdback management
# ======================================================================

# Which edge types are contaminated by each prediction target.
# Contaminated = built from the same data we're predicting on.
CONTAMINATION = {
    'co_mutation': {'COOCCURS'},
    'cancer_type': {'COOCCURS'},       # COOCCURS.cancer_type_id leaks cancer type
    'survival':    set(),               # atlas tiers handled separately via holdback
    'drug_response': set(),             # CIViC/GDSC = external
    'synthetic_lethality': set(),       # DepMap = external
}

# Edge types safe for each target (everything NOT contaminated)
CLEAN_EDGES = {
    target: {et for et in [
        'PPI', 'COUPLES', 'SYNTHETIC_LETHAL', 'CO_ESSENTIAL',
        'CO_EXPRESSED', 'CO_CNA', 'ATTENDS_TO',
        'HAS_RESISTANCE_EVIDENCE', 'HAS_SENSITIVITY_EVIDENCE',
        'COOCCURS',
    ] if et not in contaminated}
    for target, contaminated in CONTAMINATION.items()
}


class HoldbackSplit:
    """Manages train/holdback splits with contamination awareness."""

    def __init__(self, patient_ids, labels, n_folds=5, seed=42):
        self.patient_ids = np.array(patient_ids)
        self.labels = np.array(labels)
        self.n_folds = n_folds
        self.seed = seed

    def folds(self):
        """Yield (train_idx, holdback_idx) with stratification."""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.seed)
        for train_idx, hold_idx in skf.split(self.patient_ids, self.labels):
            yield train_idx, hold_idx


# ======================================================================
# Target 1: Co-mutation prediction
# ======================================================================

class CoMutationPredictor:
    """Hold out one mutation per patient, predict it from graph structure.

    Uses ONLY clean edges (no COOCCURS — that's built from the same patients).
    This tests whether PPI, COUPLES, SL, CO_ESSENTIAL etc. predict
    which genes co-mutate.

    Additionally: rebuild COOCCURS from training patients only, test on holdback.
    This is the honest test of COOCCURS signal.
    """

    def __init__(self):
        self.genes = sorted(CHANNEL_MAP.keys())
        self.gene_idx = {g: i for i, g in enumerate(self.genes)}
        self.n_genes = len(self.genes)

        # Graph-based gene-gene affinity (from clean edges only)
        self.clean_affinity = None       # (G, G) — from PPI, COUPLES, etc.
        self.cooccurs_affinity = None    # (G, G) — rebuilt from train patients

    def load_clean_edges(self):
        """Load gene-gene affinity from non-COOCCURS edges."""
        t0 = time.time()
        driver = _neo4j_driver()
        affinity = np.zeros((self.n_genes, self.n_genes), dtype=np.float32)

        clean = CLEAN_EDGES['co_mutation']
        with driver.session() as s:
            for etype in clean:
                result = s.run(f"""
                    MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                    WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
                      AND (r.deprecated IS NULL OR r.deprecated = false)
                    RETURN a.name AS g1, b.name AS g2, properties(r) AS props
                """)
                n = 0
                for r in result:
                    g1, g2 = r['g1'], r['g2']
                    if g1 not in self.gene_idx or g2 not in self.gene_idx:
                        continue
                    i, j = self.gene_idx[g1], self.gene_idx[g2]

                    # Weight by edge type
                    props = r['props'] or {}
                    w = _edge_weight(etype, props)
                    affinity[i, j] = max(affinity[i, j], w)
                    affinity[j, i] = max(affinity[j, i], w)
                    n += 1
                if n > 0:
                    print(f"    {etype}: {n:,} edges", flush=True)

        driver.close()
        self.clean_affinity = affinity
        print(f"  Clean affinity loaded [{time.time()-t0:.1f}s]")

    def build_cooccurs_from_train(self, patient_genes_train, cancer_types_train):
        """Rebuild COOCCURS affinity from training patients only.

        This is the honest version: holdback patients' mutations are not
        used to build co-occurrence scores.
        """
        cooccur = np.zeros((self.n_genes, self.n_genes), dtype=np.float32)

        for pid, genes in patient_genes_train.items():
            gene_list = [g for g in genes if g in self.gene_idx]
            for ii in range(len(gene_list)):
                for jj in range(ii + 1, len(gene_list)):
                    i = self.gene_idx[gene_list[ii]]
                    j = self.gene_idx[gene_list[jj]]
                    cooccur[i, j] += 1
                    cooccur[j, i] += 1

        # Log-scale normalize
        cooccur = np.log1p(cooccur)
        mx = cooccur.max()
        if mx > 0:
            cooccur /= mx

        self.cooccurs_affinity = cooccur

    def score_patient(self, known_genes, affinity_matrix):
        """Score all genes given patient's known mutations.

        For each candidate gene c, score = sum of affinity(c, known_gene)
        across all known_genes. Higher = more likely co-mutated.
        """
        known_idx = [self.gene_idx[g] for g in known_genes if g in self.gene_idx]
        if not known_idx:
            return np.zeros(self.n_genes)

        # Sum affinity to known genes
        scores = affinity_matrix[:, known_idx].sum(axis=1)
        return scores

    def evaluate(self, patient_genes, cancer_types, holdback_pids,
                 train_pids, use_cooccurs=False):
        """Hold-one-out evaluation on holdback patients.

        For each holdback patient with ≥2 mutations:
          - Hide one mutation at random
          - Score all genes from remaining mutations + graph
          - Measure rank of hidden gene

        Returns: MRR, Hits@10, Hits@50
        """
        t0 = time.time()

        if use_cooccurs:
            # Build COOCCURS from train only
            train_genes = {pid: patient_genes[pid] for pid in train_pids
                           if pid in patient_genes}
            train_cts = {pid: cancer_types[pid] for pid in train_pids
                         if pid in cancer_types}
            self.build_cooccurs_from_train(train_genes, train_cts)
            affinity = self.clean_affinity + self.cooccurs_affinity
            label = "clean + train-COOCCURS"
        else:
            affinity = self.clean_affinity
            label = "clean edges only"

        rng = np.random.RandomState(42)
        ranks = []
        n_eval = 0

        for pid in holdback_pids:
            if pid not in patient_genes:
                continue
            genes = [g for g in patient_genes[pid] if g in self.gene_idx]
            if len(genes) < 2:
                continue

            # Hold out one random mutation
            hold_idx = rng.randint(len(genes))
            held_out = genes[hold_idx]
            known = genes[:hold_idx] + genes[hold_idx + 1:]

            # Score all genes
            scores = self.score_patient(known, affinity)

            # Rank of held-out gene (1-indexed, lower = better)
            held_out_score = scores[self.gene_idx[held_out]]
            rank = int((scores > held_out_score).sum()) + 1
            ranks.append(rank)
            n_eval += 1

        if not ranks:
            return {'mrr': 0, 'hits_10': 0, 'hits_50': 0, 'n_eval': 0}

        ranks = np.array(ranks)
        mrr = float(np.mean(1.0 / ranks))
        hits_10 = float(np.mean(ranks <= 10))
        hits_50 = float(np.mean(ranks <= 50))
        median_rank = float(np.median(ranks))

        print(f"  Co-mutation ({label}):")
        print(f"    MRR: {mrr:.4f}")
        print(f"    Hits@10: {hits_10:.1%}")
        print(f"    Hits@50: {hits_50:.1%}")
        print(f"    Median rank: {median_rank:.0f} / {self.n_genes}")
        print(f"    N evaluated: {n_eval:,}")
        print(f"    [{time.time()-t0:.1f}s]")

        return {
            'mrr': mrr,
            'hits_10': hits_10,
            'hits_50': hits_50,
            'median_rank': median_rank,
            'n_eval': n_eval,
            'label': label,
        }


# ======================================================================
# Target 2: Cancer type prediction
# ======================================================================

class CancerTypePredictor:
    """Given a patient's mutations only, predict cancer type.

    Uses channel membership + clean edge structure to build a
    mutation-profile → cancer-type classifier. No COOCCURS (leaks cancer type).
    """

    def __init__(self):
        self.genes = sorted(CHANNEL_MAP.keys())
        self.gene_idx = {g: i for i, g in enumerate(self.genes)}

    def build_mutation_profile(self, genes):
        """Binary mutation vector over gene vocabulary."""
        profile = np.zeros(len(self.genes), dtype=np.float32)
        for g in genes:
            if g in self.gene_idx:
                profile[self.gene_idx[g]] = 1.0
        return profile

    def evaluate(self, patient_genes, cancer_types, holdback_pids, train_pids,
                 top_k_types=20):
        """Train simple classifier on train, evaluate on holdback.

        Uses logistic regression on mutation profiles — no graph edges needed
        for the classifier itself, but the gene vocabulary IS the graph.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import top_k_accuracy_score, accuracy_score
        t0 = time.time()

        # Filter to cancer types with enough samples
        ct_counts = defaultdict(int)
        for pid in train_pids:
            if pid in cancer_types:
                ct_counts[cancer_types[pid]] += 1
        valid_cts = {ct for ct, n in ct_counts.items() if n >= 50}

        # Build train set
        X_train, y_train = [], []
        for pid in train_pids:
            if pid not in patient_genes or pid not in cancer_types:
                continue
            ct = cancer_types[pid]
            if ct not in valid_cts:
                continue
            X_train.append(self.build_mutation_profile(patient_genes[pid]))
            y_train.append(ct)

        # Build holdback set
        X_hold, y_hold = [], []
        for pid in holdback_pids:
            if pid not in patient_genes or pid not in cancer_types:
                continue
            ct = cancer_types[pid]
            if ct not in valid_cts:
                continue
            X_hold.append(self.build_mutation_profile(patient_genes[pid]))
            y_hold.append(ct)

        if not X_train or not X_hold:
            return {'accuracy': 0, 'top5_accuracy': 0, 'n_eval': 0}

        X_train = np.array(X_train)
        X_hold = np.array(X_hold)
        y_train = np.array(y_train)
        y_hold = np.array(y_hold)

        # Logistic regression — mutation profile → cancer type
        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            multi_class='multinomial', n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_hold)
        accuracy = float(accuracy_score(y_hold, y_pred))

        # Top-5 accuracy
        y_proba = clf.predict_proba(X_hold)
        try:
            top5 = float(top_k_accuracy_score(y_hold, y_proba,
                                              k=min(5, len(valid_cts)),
                                              labels=clf.classes_))
        except ValueError:
            top5 = accuracy

        print(f"  Cancer type prediction (mutation profile → cancer type):")
        print(f"    Accuracy:     {accuracy:.1%}")
        print(f"    Top-5 acc:    {top5:.1%}")
        print(f"    N cancer types: {len(valid_cts)}")
        print(f"    N train: {len(X_train):,}, N holdback: {len(X_hold):,}")
        print(f"    [{time.time()-t0:.1f}s]")

        # Most informative genes per cancer type (top 5 by coefficient)
        top_genes_per_ct = {}
        for ci, ct in enumerate(clf.classes_):
            coefs = clf.coef_[ci]
            top_idx = np.argsort(coefs)[-5:][::-1]
            top_genes_per_ct[ct] = [
                (self.genes[idx], float(coefs[idx])) for idx in top_idx
            ]

        return {
            'accuracy': accuracy,
            'top5_accuracy': top5,
            'n_cancer_types': len(valid_cts),
            'n_train': len(X_train),
            'n_holdback': len(X_hold),
            'top_genes_per_ct': top_genes_per_ct,
        }


# ======================================================================
# Target 3: Synthetic lethality validation (external)
# ======================================================================

class SyntheticLethalityValidator:
    """Validate graph SL edges against DepMap essentiality data.

    If the graph says gene A is synthetic lethal with gene B, then
    cell lines with A-mutant should show B as more essential (lower
    CRISPR score) than cell lines with A-wildtype.

    This is naturally clean — SL edges and DepMap are independent sources.
    """

    def validate(self):
        """Check SL predictions against DepMap dependency data in graph."""
        t0 = time.time()
        driver = _neo4j_driver()

        results = []
        with driver.session() as s:
            # Get all SL edges with their genes
            sl_edges = s.run("""
                MATCH (a:Gene)-[r:SYNTHETIC_LETHAL]->(b:Gene)
                WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
                  AND (r.deprecated IS NULL OR r.deprecated = false)
                RETURN a.name AS g1, b.name AS g2, properties(r) AS props
            """)
            sl_pairs = [(r['g1'], r['g2'], dict(r['props'] or {}))
                        for r in sl_edges]

            # Get DepMap essentiality for these genes
            for g1, g2, props in sl_pairs:
                # Check if partner gene has DepMap data
                dep = s.run("""
                    MATCH (g:Gene {name: $gene})
                    RETURN g.depmap_mean_essentiality AS dep_mean,
                           g.depmap_min_essentiality AS dep_min,
                           g.depmap_n_lineages AS dep_n
                """, gene=g2).single()

                if dep and dep['dep_mean'] is not None:
                    results.append({
                        'gene_a': g1,
                        'gene_b': g2,
                        'sl_evidence': props.get('evidence_type', 'unknown'),
                        'b_depmap_mean': float(dep['dep_mean']),
                        'b_depmap_min': float(dep['dep_min'] or 0),
                        'b_depmap_lineages': int(dep['dep_n'] or 0),
                        'is_essential': float(dep['dep_mean']) < -0.5,
                    })

        driver.close()

        if not results:
            print("  SL validation: no DepMap data for SL partners")
            return {'n_validated': 0, 'agreement_rate': 0}

        n_essential = sum(1 for r in results if r['is_essential'])
        agreement = n_essential / len(results)

        print(f"  Synthetic lethality validation:")
        print(f"    SL pairs with DepMap data: {len(results)}")
        print(f"    Partner is essential: {n_essential} ({agreement:.1%})")
        print(f"    [{time.time()-t0:.1f}s]")

        return {
            'n_validated': len(results),
            'n_essential': n_essential,
            'agreement_rate': agreement,
            'details': results[:20],
        }


# ======================================================================
# Helpers
# ======================================================================

def _edge_weight(etype, props):
    """Convert edge properties to a single affinity weight."""
    if etype == 'PPI':
        return min(float(props.get('score', 500)) / 1000.0, 1.0)
    elif etype in ('CO_EXPRESSED', 'CO_CNA', 'CO_ESSENTIAL'):
        return abs(float(props.get('correlation', 0)))
    elif etype == 'COUPLES':
        return 0.5
    elif etype == 'SYNTHETIC_LETHAL':
        return 0.8
    elif etype == 'ATTENDS_TO':
        return float(props.get('weight', 0))
    elif etype in ('HAS_SENSITIVITY_EVIDENCE', 'HAS_RESISTANCE_EVIDENCE'):
        return 0.3
    return 0.1


def load_patient_data():
    """Load patient mutations + cancer types from MSK-IMPACT data."""
    t0 = time.time()

    mut = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_mutations.csv"),
        low_memory=False,
        usecols=["patientId", "gene.hugoGeneSymbol"],
    )
    clin = pd.read_csv(
        os.path.join(ANALYSIS_CACHE, "msk_impact_50k_2026_clinical.csv"),
        low_memory=False,
        usecols=["patientId", "CANCER_TYPE", "OS_MONTHS", "OS_STATUS"],
    )

    # Patient → set of mutated genes (within our vocabulary)
    vocab = set(CHANNEL_MAP.keys())
    patient_genes = defaultdict(set)
    for _, row in mut.iterrows():
        gene = row["gene.hugoGeneSymbol"]
        if gene in vocab:
            patient_genes[row["patientId"]].add(gene)

    # Patient → cancer type
    cancer_types = {}
    for _, row in clin.iterrows():
        ct = row.get("CANCER_TYPE")
        if pd.notna(ct):
            cancer_types[row["patientId"]] = ct

    # Patient → survival (for stratification)
    events = {}
    for _, row in clin.iterrows():
        status = row.get("OS_STATUS")
        if pd.notna(status):
            events[row["patientId"]] = 1 if "DECEASED" in str(status).upper() else 0

    print(f"  Loaded {len(patient_genes):,} patients with mutations, "
          f"{len(cancer_types):,} with cancer type [{time.time()-t0:.1f}s]")

    return dict(patient_genes), cancer_types, events


# ======================================================================
# Main orchestration
# ======================================================================

def run_predictions(targets=None, n_folds=5):
    """Run all prediction targets with proper holdback."""
    if targets is None:
        targets = ['co_mutation', 'cancer_type', 'synthetic_lethality']

    print("=" * 60)
    print("  MULTI-TARGET GRAPH PREDICTIONS")
    print("=" * 60)

    patient_genes, cancer_types, events = load_patient_data()

    # Patients with both mutations and cancer type
    valid_pids = sorted([pid for pid in patient_genes
                         if pid in cancer_types and pid in events])
    labels = np.array([events[pid] for pid in valid_pids])

    print(f"  Valid patients: {len(valid_pids):,}")

    holdback = HoldbackSplit(valid_pids, labels, n_folds=n_folds)
    all_results = {}

    for fold, (train_idx, hold_idx) in enumerate(holdback.folds()):
        train_pids = [valid_pids[i] for i in train_idx]
        hold_pids = [valid_pids[i] for i in hold_idx]

        print(f"\n--- Fold {fold + 1}/{n_folds} "
              f"(train={len(train_pids):,}, holdback={len(hold_pids):,}) ---")

        fold_results = {}

        # Co-mutation prediction
        if 'co_mutation' in targets:
            print("\n[Co-mutation prediction]")
            cmp = CoMutationPredictor()
            cmp.load_clean_edges()

            # Test 1: clean edges only (no COOCCURS at all)
            r_clean = cmp.evaluate(patient_genes, cancer_types,
                                   hold_pids, train_pids,
                                   use_cooccurs=False)

            # Test 2: clean edges + COOCCURS rebuilt from train only
            r_with_cooccurs = cmp.evaluate(patient_genes, cancer_types,
                                           hold_pids, train_pids,
                                           use_cooccurs=True)

            fold_results['co_mutation_clean'] = r_clean
            fold_results['co_mutation_with_cooccurs'] = r_with_cooccurs

        # Cancer type prediction
        if 'cancer_type' in targets:
            print("\n[Cancer type prediction]")
            ctp = CancerTypePredictor()
            r_ct = ctp.evaluate(patient_genes, cancer_types,
                                hold_pids, train_pids)
            fold_results['cancer_type'] = r_ct

        # SL validation (not fold-dependent — external data)
        if 'synthetic_lethality' in targets and fold == 0:
            print("\n[Synthetic lethality validation]")
            slv = SyntheticLethalityValidator()
            r_sl = slv.validate()
            fold_results['synthetic_lethality'] = r_sl

        # Accumulate
        for key, val in fold_results.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(val)

    # Summarize across folds
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ACROSS {n_folds} FOLDS")
    print(f"{'=' * 60}")

    summary = {}
    for key, fold_vals in all_results.items():
        if key == 'synthetic_lethality':
            summary[key] = fold_vals[0]  # not fold-dependent
            continue

        # Average numeric metrics across folds
        metrics = {}
        for metric in fold_vals[0]:
            vals = [fv[metric] for fv in fold_vals
                    if isinstance(fv.get(metric), (int, float))]
            if vals:
                metrics[f"{metric}_mean"] = float(np.mean(vals))
                metrics[f"{metric}_std"] = float(np.std(vals))

        summary[key] = metrics

        print(f"\n  {key}:")
        for mk, mv in sorted(metrics.items()):
            if mk.endswith('_mean'):
                base = mk.replace('_mean', '')
                std = metrics.get(f"{base}_std", 0)
                print(f"    {base}: {mv:.4f} ± {std:.4f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "predictions_report.json")

    # Clean for JSON
    def _clean(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_clean)

    print(f"\n  Results saved to {out_path}")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='all',
                        choices=['all', 'co_mutation', 'cancer_type',
                                 'synthetic_lethality'])
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()

    if args.target == 'all':
        targets = ['co_mutation', 'cancer_type', 'synthetic_lethality']
    else:
        targets = [args.target]

    run_predictions(targets=targets, n_folds=args.n_folds)


if __name__ == "__main__":
    main()
