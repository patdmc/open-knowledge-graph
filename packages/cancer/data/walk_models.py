"""
Walk models — runs survival models on cached walk features.

Separates data computation from model iteration:
  1. build_cache() — runs the graph walk once, saves features to parquet
  2. run_models() — loads cached features, runs Cox + GBM, compares

Usage:
    python3 -u -m gnn.data.walk_models build    # build cache (slow, once)
    python3 -u -m gnn.data.walk_models run      # run models (fast, iterate)
    python3 -u -m gnn.data.walk_models          # build if needed, then run
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_CACHE

CACHE_PATH = os.path.join(GNN_CACHE, "walk_features.parquet")


def build_cache():
    """Run the full graph walk pipeline and cache the result."""
    from gnn.data.neo4j_walk import Neo4jWalkEngine, _step
    from collections import defaultdict

    engine = Neo4jWalkEngine()
    try:
        t0 = time.time()

        # 1. Augment (skips if already done)
        engine.augment_graph()

        # 2. Map: group features
        group_df = engine.walk_groups()

        # 3. Reduce: patient features
        df = engine.walk_patients(group_df)
        df = df[df['OS_MONTHS'] > 0].copy()

        # 4. Affinity
        aff_df = engine.compute_group_affinity()
        if len(aff_df) > 0:
            aff_stats_a = aff_df.groupby('group_a').agg(
                n_neighbors=('jaccard', 'count'),
                mean_jacc=('jaccard', 'mean'),
            )
            aff_stats_b = aff_df.groupby('group_b').agg(
                n_neighbors=('jaccard', 'count'),
                mean_jacc=('jaccard', 'mean'),
            )
            aff_stats = pd.concat([aff_stats_a, aff_stats_b]).groupby(level=0).agg(
                n_neighbors=('n_neighbors', 'sum'),
                mean_jacc=('mean_jacc', 'mean'),
            )
            df = df.merge(aff_stats, left_on='mutation_key',
                          right_index=True, how='left')
            df['n_neighbors'] = df['n_neighbors'].fillna(0)
            df['mean_jacc'] = df['mean_jacc'].fillna(0.0)
        else:
            df['n_neighbors'] = 0
            df['mean_jacc'] = 0.0

        # 4b. Group attention neighborhood
        with engine.driver.session() as s:
            result = s.run("""
                MATCH (mg:MutationGroup)-[ga:GROUP_ATTENDS_TO]->(nb:MutationGroup)
                RETURN mg.mutation_key AS key,
                       count(nb) AS n_attn_neighbors,
                       avg(ga.weight) AS mean_attn_to_neighbors,
                       max(ga.weight) AS max_attn_to_neighbors,
                       avg(nb.median_os) AS attn_neighbor_median_os,
                       sum(ga.weight * nb.median_os) / sum(ga.weight) AS attn_weighted_os
            """)
            ga_rows = list(result)

        if ga_rows:
            ga_df = pd.DataFrame([dict(r) for r in ga_rows])
            df = df.merge(ga_df, left_on='mutation_key', right_on='key', how='left')
            df = df.drop(columns=['key'], errors='ignore')

        for col in ['n_attn_neighbors', 'mean_attn_to_neighbors',
                    'max_attn_to_neighbors', 'attn_neighbor_median_os',
                    'attn_weighted_os']:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = 0.0

        # Report NaN status
        print(f"\n{'='*70}")
        print("  DATA QUALITY CHECK")
        print(f"{'='*70}\n")
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"  Columns with NaN:")
            for col in nan_cols:
                n_nan = df[col].isna().sum()
                print(f"    {col}: {n_nan:,} ({100*n_nan/len(df):.1f}%)")
            # Fill remaining NaN
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(0.0)
            print(f"  Filled all remaining NaN with 0.0")
        else:
            print(f"  No NaN values found.")

        # Inf check
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            print(f"  Columns with Inf: {inf_cols}")
            for col in inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
            print(f"  Replaced Inf with 0.0")

        print(f"\n  Patients: {len(df):,}")
        print(f"  Features: {len(df.columns)}")

        # Save
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_parquet(CACHE_PATH, index=False)
        print(f"\n  Cached to {CACHE_PATH}")
        print(f"  Total build time: {time.time() - t0:.1f}s")

    finally:
        engine.close()


def run_models():
    """Load cached features and run survival models."""
    from sklearn.model_selection import StratifiedKFold
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as ci
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    print(f"\n{'='*70}")
    print("  LOADING CACHED WALK FEATURES")
    print(f"{'='*70}\n")

    df = pd.read_parquet(CACHE_PATH)
    print(f"  Patients: {len(df):,}")

    # Feature sets
    topo_features = [
        'atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_channels_severed',
        'GOF_LOF_within', 'GOF_LOF_cross',
        'GOF_GOF_cross', 'LOF_LOF_within',
        'path_PI3K_GOF_x_CC_LOF', 'path_PI3K_GOF_x_TA_LOF',
        'path_CC_GOF_x_DDR_LOF', 'path_anyGOF_x_Immune_LOF',
        'n_neighbors', 'mean_jacc',
    ]

    attn_features = [
        'attn_flow', 'attn_mean', 'attn_cross_flow', 'attn_max',
        'attn_max_asym', 'attn_density', 'self_attn_sum',
        'attn_max_driver_ratio',
        'n_attn_neighbors', 'mean_attn_to_neighbors',
        'max_attn_to_neighbors', 'attn_neighbor_median_os',
        'attn_weighted_os',
    ]

    all_features = topo_features + attn_features

    # Standardize numeric columns (make a copy for Cox)
    df_std = df.copy()
    for col in df_std.select_dtypes(include=[np.number]).columns:
        if col in ('OS_MONTHS', 'event'):
            continue
        std = df_std[col].std()
        if std > 0:
            df_std[col] = (df_std[col] - df_std[col].mean()) / std

    # Filter to valid features
    valid_all = [f for f in all_features if f in df_std.columns and df_std[f].std() > 0.001]
    valid_topo = [f for f in topo_features if f in df_std.columns and df_std[f].std() > 0.001]
    valid_attn = [f for f in attn_features if f in df_std.columns and df_std[f].std() > 0.001]

    print(f"  Topology features: {len(valid_topo)}")
    print(f"  Attention features: {len(valid_attn)}")
    print(f"  Total features: {len(valid_all)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def make_y(sub_df):
        return np.array(
            [(bool(e), t) for e, t in zip(sub_df['event'], sub_df['OS_MONTHS'])],
            dtype=[('event', bool), ('time', float)]
        )

    def run_cox(name, features, data):
        c_vals = []
        for fold, (ti, vi) in enumerate(skf.split(data, data['event'])):
            train, val = data.iloc[ti], data.iloc[vi]
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(train[features + ['OS_MONTHS', 'event']],
                    duration_col='OS_MONTHS', event_col='event')
            h = cph.predict_partial_hazard(val[features]).values.flatten()
            c = ci(val['OS_MONTHS'].values, -h, val['event'].values)
            c_vals.append(c)
        mean_c = np.mean(c_vals)
        std_c = np.std(c_vals)
        print(f"  {name:40s} C = {mean_c:.4f} +/- {std_c:.4f}")
        return mean_c, c_vals

    def run_gbm(name, features, data, return_imp=False):
        c_vals = []
        imp = np.zeros(len(features))
        for fold, (ti, vi) in enumerate(skf.split(data, data['event'])):
            train_x = data.iloc[ti][features].values
            val_x = data.iloc[vi][features].values
            train_y = make_y(data.iloc[ti])
            val_y = make_y(data.iloc[vi])

            gbm = GradientBoostingSurvivalAnalysis(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=50, subsample=0.8, random_state=42,
            )
            gbm.fit(train_x, train_y)
            pred = gbm.predict(val_x)
            c = concordance_index_censored(val_y['event'], val_y['time'], pred)[0]
            c_vals.append(c)
            imp += gbm.feature_importances_

        mean_c = np.mean(c_vals)
        std_c = np.std(c_vals)
        print(f"  {name:40s} C = {mean_c:.4f} +/- {std_c:.4f}")
        if return_imp:
            return mean_c, c_vals, imp / 5, features
        return mean_c, c_vals

    # ===== Run all models =====
    print(f"\n{'='*70}")
    print("  MODEL COMPARISON")
    print(f"{'='*70}\n")

    print("  --- Cox (linear) ---")
    cox_topo, _ = run_cox("Topology only", valid_topo, df_std)
    cox_all, _ = run_cox("Topology + Attention", valid_all, df_std)

    print("\n  --- Gradient Boosted Survival (non-linear) ---")
    gbm_topo, _ = run_gbm("Topology only", valid_topo, df_std)
    gbm_all, _, imp, imp_names = run_gbm("Topology + Attention", valid_all, df_std,
                                          return_imp=True)

    # Feature importance
    print(f"\n{'='*70}")
    print("  FEATURE IMPORTANCE (GBM, all features)")
    print(f"{'='*70}\n")

    imp_df = pd.DataFrame({'feature': imp_names, 'importance': imp})
    imp_df = imp_df.sort_values('importance', ascending=False)
    for _, row in imp_df.head(20).iterrows():
        bar = '#' * int(row['importance'] * 200)
        is_attn = '*' if row['feature'] in attn_features else ' '
        print(f"  {is_attn} {row['feature']:30s} {row['importance']:.4f}  {bar}")
    print(f"\n  * = attention-derived feature")

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Model':40s} {'Topo only':>12s} {'+ Attention':>12s} {'Lift':>8s}")
    print(f"  {'-'*72}")
    print(f"  {'Cox (linear)':40s} {cox_topo:12.4f} {cox_all:12.4f} {cox_all - cox_topo:+8.4f}")
    print(f"  {'GBM (non-linear)':40s} {gbm_topo:12.4f} {gbm_all:12.4f} {gbm_all - gbm_topo:+8.4f}")
    print(f"  {'-'*72}")
    print(f"  {'Non-linear lift over Cox':40s} {gbm_topo - cox_topo:+12.4f} {gbm_all - cox_all:+12.4f}")
    print(f"")
    print(f"  BASELINES:")
    print(f"  Atlas lookup (zero-param):    C = 0.577")
    print(f"  Directional walk (Python):    C = 0.636")
    print(f"  AtlasTransformer V1 (neural): C = 0.673")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', default='auto',
                        choices=['build', 'run', 'auto'])
    args = parser.parse_args()

    if args.action == 'build':
        build_cache()
    elif args.action == 'run':
        if not os.path.exists(CACHE_PATH):
            print(f"No cache found at {CACHE_PATH}. Run 'build' first.")
            return
        run_models()
    else:  # auto
        if not os.path.exists(CACHE_PATH):
            print("No cache found. Building...")
            build_cache()
        run_models()


if __name__ == "__main__":
    main()
