"""
Neo4j graph walk engine — batch map-reduce over patient equivalence groups.

Architecture:
  1. LOAD: Augment existing Neo4j graph with MutationGroups, COUPLES, atlas log_hr
  2. MAP:  Walk features per MutationGroup (one Cypher query, all groups)
  3. REDUCE: Join group features → patients, aggregate atlas_sum per patient
  4. COX: Run Cox regression on the joined feature matrix

All operations are batched via UNWIND. Nothing iterates 44K patients in Python.

Usage:
    python3 -u -m gnn.data.neo4j_walk
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from neo4j import GraphDatabase

from gnn.config import (
    CHANNEL_MAP, CHANNEL_NAMES, CHANNEL_TO_IDX,
    HUB_GENES, GENE_FUNCTION, GENE_POSITION,
    NON_SILENT, TRUNCATING, MSK_DATASETS,
    GNN_CACHE, ANALYSIS_CACHE, ALL_GENES,
)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")
BATCH_SIZE = 5000


def _step(name, expected=None, actual=None, elapsed=None):
    """Print a formatted step report."""
    status = ""
    if expected is not None and actual is not None:
        ok = "OK" if actual == expected else f"MISMATCH (expected {expected})"
        status = f"{actual:,} {ok}"
    elif actual is not None:
        status = f"{actual:,}"
    t = f"[{elapsed:.1f}s]" if elapsed is not None else ""
    print(f"  {name:.<50s} {status:>20s}  {t}", flush=True)


def _batched_write(session, query, rows, batch_size=BATCH_SIZE):
    """Execute a write query in UNWIND batches. Returns total processed."""
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        session.run(query, batch=batch)
        total += len(batch)
    return total


class Neo4jWalkEngine:
    """Neo4j-backed mutation graph walk with equivalence groups."""

    def __init__(self, uri=NEO4J_URI, auth=NEO4J_AUTH,
                 dataset_name="msk_impact_50k"):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.dataset_name = dataset_name
        self._atlas_t1 = {}
        self._atlas_t2 = {}
        self._atlas_t3 = {}

    def close(self):
        self.driver.close()

    # ==================================================================
    # PHASE 1: Augment graph with walk-specific structures
    # ==================================================================

    def augment_graph(self):
        """Add MutationGroups, COUPLES, attention edges, atlas log_hr to existing graph.

        Safe to run repeatedly — uses MERGE and idempotent operations.
        """
        print(f"\n{'='*70}")
        print("  AUGMENTING NEO4J GRAPH FOR WALK")
        print(f"{'='*70}\n")
        t_total = time.time()

        self._load_atlas()
        self._create_indexes()
        self._fix_gene_channels()
        self._add_couples_edges()
        self._add_attention_edges()
        self._add_atlas_log_hr()
        self._create_mutation_groups()
        self._add_group_attention_edges()
        self._validate()

        _step("Total augment time", elapsed=time.time() - t_total)

    def _load_atlas(self):
        """Load survival atlas from Neo4j PROGNOSTIC_IN edges."""
        t0 = time.time()
        from gnn.data.graph_snapshot import load_atlas
        t1, t2, t3, t4, _ = load_atlas()

        # Convert to log_hr format expected by this class
        for key, entry in t1.items():
            self._atlas_t1[key] = float(np.log(max(entry["hr"], 0.01)))
        for key, entry in t2.items():
            self._atlas_t2[key] = float(np.log(max(entry["hr"], 0.01)))
        for key, entry in t3.items():
            self._atlas_t3[key] = float(np.log(max(entry["hr"], 0.01)))
        self._atlas_t4 = {}
        for key, entry in t4.items():
            self._atlas_t4[key] = float(np.log(max(entry["hr"], 0.01)))

        total = len(self._atlas_t1) + len(self._atlas_t2) + len(self._atlas_t3) + len(self._atlas_t4)
        _step("Atlas loaded (from Neo4j)", actual=total, elapsed=time.time() - t0)

    def _atlas_log_hr(self, cancer_type, gene, protein_change):
        """Tiered atlas lookup: T1 (mutation) > T2 (gene) > T3 (channel) > T4 (imputed)."""
        key1 = (cancer_type, gene, protein_change)
        if key1 in self._atlas_t1:
            return self._atlas_t1[key1]
        key2 = (cancer_type, gene)
        if key2 in self._atlas_t2:
            return self._atlas_t2[key2]
        ch = CHANNEL_MAP.get(gene)
        if ch:
            key3 = (cancer_type, ch)
            if key3 in self._atlas_t3:
                return self._atlas_t3[key3]
        if key2 in self._atlas_t4:
            return self._atlas_t4[key2]
        return 0.0

    def _create_indexes(self):
        """Ensure indexes exist for walk queries."""
        t0 = time.time()
        indexes = [
            "CREATE INDEX mg_key IF NOT EXISTS FOR (mg:MutationGroup) ON (mg.mutation_key)",
            "CREATE INDEX patient_mk IF NOT EXISTS FOR (p:Patient) ON (p.mutation_key)",
        ]
        with self.driver.session() as s:
            for idx in indexes:
                s.run(idx)
        _step("Indexes created", actual=len(indexes), elapsed=time.time() - t0)

    def _fix_gene_channels(self):
        """Set the 6-channel `channel` property on Gene nodes from CHANNEL_MAP.

        The existing `primary_channel` uses the V6 8-channel system.
        The walk uses the 6-channel system from config.py. Fix at the source.
        """
        t0 = time.time()

        # Check if already done
        with self.driver.session() as s:
            r = s.run("MATCH (g:Gene) WHERE g.channel IS NOT NULL RETURN count(g) AS c")
            existing = r.single()['c']
            if existing > 0:
                _step("Gene.channel (6-ch, already set)", actual=existing,
                      elapsed=time.time() - t0)
                return

        rows = [{'gene': gene, 'channel': ch} for gene, ch in CHANNEL_MAP.items()]

        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   MATCH (g:Gene {name: row.gene})
                   SET g.channel = row.channel""",
                rows)

        _step("Gene.channel (6-ch) set", actual=n, elapsed=time.time() - t0)

    def _add_attention_edges(self):
        """Load transformer attention edges into Neo4j as directed ATTENDS_TO edges.

        Directed: TP53→KRAS ≠ KRAS→TP53. The weight encodes how much gene A's
        context changes the interpretation of gene B.

        Also loads self-attention as Gene.self_attn property.
        """
        t0 = time.time()

        # Check if already done
        with self.driver.session() as s:
            r = s.run("MATCH ()-[r:ATTENDS_TO]->() RETURN count(r) AS c")
            existing = r.single()['c']
            if existing > 0:
                _step("ATTENDS_TO edges (already exist)", actual=existing,
                      elapsed=time.time() - t0)
                return

        # Load from extracted attention data
        attn_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "attention_edges", "gene_gene_attention.csv"
        )
        if not os.path.exists(attn_path):
            print("  [SKIP] No attention data found. Run extract_attention.py first.")
            return

        attn_df = pd.read_csv(attn_path)

        # Directed edges: gene_from → gene_to with attention weight
        # Filter: mean_attn > 0.05 and n_obs >= 100
        edges = attn_df[(attn_df['mean_attn'] > 0.05) & (attn_df['n_obs'] >= 100)].copy()

        # Separate self-attention (from == to) and cross-attention
        self_attn = edges[edges['from'] == edges['to']]
        cross_attn = edges[edges['from'] != edges['to']]

        # Compute asymmetry ratio for each directed edge
        # For A→B: ratio = attn(A→B) / attn(B→A)
        reverse_lookup = {}
        for _, row in cross_attn.iterrows():
            reverse_lookup[(row['to'], row['from'])] = row['mean_attn']

        rows = []
        for _, row in cross_attn.iterrows():
            rev = reverse_lookup.get((row['from'], row['to']), row['mean_attn'])
            asym = row['mean_attn'] / (rev + 1e-8)
            rows.append({
                'from': row['from'], 'to': row['to'],
                'weight': round(float(row['mean_attn']), 4),
                'asymmetry': round(float(asym), 4),
                'n_obs': int(row['n_obs']),
                'cross_channel': bool(row['cross_channel']),
            })

        # Create edges
        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   MATCH (a:Gene {name: row.from}), (b:Gene {name: row.to})
                   CREATE (a)-[:ATTENDS_TO {
                       weight: row.weight,
                       asymmetry: row.asymmetry,
                       n_obs: row.n_obs,
                       cross_channel: row.cross_channel
                   }]->(b)""",
                rows)

        _step("ATTENDS_TO edges (directed)", actual=n, elapsed=time.time() - t0)

        # Self-attention as self-loop edges (not just a property — it's a real edge)
        t1 = time.time()
        self_rows = [{'gene': row['from'], 'weight': round(float(row['mean_attn']), 4)}
                     for _, row in self_attn.iterrows()]
        if self_rows:
            with self.driver.session() as s:
                _batched_write(s,
                    """UNWIND $batch AS row
                       MATCH (g:Gene {name: row.gene})
                       SET g.self_attn = row.weight
                       CREATE (g)-[:ATTENDS_TO {
                           weight: row.weight,
                           asymmetry: 1.0,
                           n_obs: 0,
                           cross_channel: false,
                           self_loop: true
                       }]->(g)""",
                    self_rows)
            _step("Gene self-loop ATTENDS_TO", actual=len(self_rows),
                  elapsed=time.time() - t1)

    def _add_couples_edges(self):
        """Add COUPLES edges between the 92 curated genes."""
        t0 = time.time()

        # Check if already done
        with self.driver.session() as s:
            r = s.run("MATCH ()-[r:COUPLES]->() RETURN count(r) AS c")
            existing = r.single()['c']
            if existing > 0:
                _step("COUPLES edges (already exist)", actual=existing,
                      elapsed=time.time() - t0)
                return

        # Build pairs from the 92 curated genes
        curated = [g for g in ALL_GENES if g in CHANNEL_MAP]
        rows = []
        for a, b in combinations(curated, 2):
            ch_a, ch_b = CHANNEL_MAP[a], CHANNEL_MAP[b]
            func_a = GENE_FUNCTION.get(a, 'context')
            func_b = GENE_FUNCTION.get(b, 'context')
            locality = 'within' if ch_a == ch_b else 'cross'

            dirs = sorted([func_a, func_b])
            if dirs == ['GOF', 'LOF']:
                pair_type = 'GOF_LOF'
            elif dirs == ['GOF', 'GOF']:
                pair_type = 'GOF_GOF'
            elif dirs == ['LOF', 'LOF']:
                pair_type = 'LOF_LOF'
            else:
                pair_type = 'other'

            rows.append({
                'a': a, 'b': b,
                'locality': locality, 'pair_type': pair_type,
            })

        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   MATCH (a:Gene {name: row.a}), (b:Gene {name: row.b})
                   CREATE (a)-[:COUPLES {locality: row.locality,
                                         pair_type: row.pair_type}]->(b)""",
                rows)

        _step("COUPLES edges created", expected=len(rows), actual=n,
              elapsed=time.time() - t0)

    def _add_atlas_log_hr(self):
        """Add log_hr property to HAS_MUTATION edges."""
        t0 = time.time()

        # Check if already done
        with self.driver.session() as s:
            r = s.run("MATCH ()-[m:HAS_MUTATION]->() WHERE m.log_hr IS NOT NULL "
                       "RETURN count(m) AS c")
            existing = r.single()['c']
            if existing > 0:
                _step("log_hr on mutations (already exist)", actual=existing,
                      elapsed=time.time() - t0)
                return

        # Fetch all mutation edges, compute full tiered log_hr in Python,
        # batch-update via elementId.
        print("  Fetching mutation edges for atlas lookup...", flush=True)
        with self.driver.session() as s:
            result = s.run(
                """MATCH (p:Patient)-[m:HAS_MUTATION]->(g:Gene)
                   RETURN elementId(m) AS eid, p.cancer_type AS ct,
                          g.name AS gene, m.protein_change AS pc""")
            records = list(result)

        print(f"  Computing log_hr for {len(records):,} mutations (T1>T2>T3)...",
              flush=True)

        updates = []
        for rec in records:
            log_hr = self._atlas_log_hr(
                rec['ct'], rec['gene'], rec['pc'] or '')
            updates.append({'eid': rec['eid'], 'log_hr': log_hr})

        # Batch update via elementId
        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   MATCH ()-[m:HAS_MUTATION]->()
                   WHERE elementId(m) = row.eid
                   SET m.log_hr = row.log_hr""",
                updates)

        nonzero = sum(1 for u in updates if u['log_hr'] != 0.0)
        _step("log_hr set (full T1>T2>T3)", actual=n, elapsed=time.time() - t0)
        _step("  non-zero log_hr", actual=nonzero)

    def _create_mutation_groups(self):
        """Create MutationGroup nodes and MEMBER_OF / HAS_GENE edges."""
        t0 = time.time()

        # Check if already done
        with self.driver.session() as s:
            r = s.run("MATCH (mg:MutationGroup) RETURN count(mg) AS c")
            existing = r.single()['c']
            if existing > 0:
                _step("MutationGroups (already exist)", actual=existing,
                      elapsed=time.time() - t0)
                return

        # Step 1: Get each patient's gene set (from 92 curated genes only)
        print("  Fetching patient gene sets...", flush=True)
        curated_set = set(CHANNEL_MAP.keys())

        with self.driver.session() as s:
            result = s.run(
                """MATCH (p:Patient)
                   OPTIONAL MATCH (p)-[m:HAS_MUTATION]->(g:Gene)
                   WHERE g.name IN $curated
                   RETURN p.id AS pid, p.cancer_type AS ct,
                          p.os_months AS os, p.event AS event,
                          collect(DISTINCT g.name) AS genes,
                          collect(DISTINCT m.direction) AS directions""",
                curated=list(curated_set))
            patient_data = list(result)

        print(f"  {len(patient_data):,} patients fetched", flush=True)

        # Step 2: Compute mutation_key per patient and group
        groups = defaultdict(list)  # mutation_key -> [patient_data]
        patient_keys = []  # (pid, mutation_key) for batch update

        for rec in patient_data:
            genes = sorted(rec['genes']) if rec['genes'] else []
            if genes:
                key = hashlib.sha256("|".join(genes).encode()).hexdigest()[:16]
            else:
                key = "WILDTYPE"
            groups[key].append(rec)
            patient_keys.append({'pid': rec['pid'], 'key': key})

        print(f"  {len(groups):,} equivalence groups", flush=True)

        # Step 3: Set mutation_key on Patient nodes
        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   MATCH (p:Patient {id: row.pid})
                   SET p.mutation_key = row.key""",
                patient_keys)
        _step("mutation_key set on patients", actual=n,
              elapsed=time.time() - t0)

        # Step 4: Create MutationGroup nodes
        group_rows = []
        for key, patients in groups.items():
            genes = sorted(patients[0]['genes']) if patients[0]['genes'] else []
            os_vals = [p['os'] for p in patients if p['os'] and p['os'] > 0]
            events = [p['event'] for p in patients if p['event'] is not None]
            group_rows.append({
                'key': key,
                'genes': genes,
                'n_patients': len(patients),
                'n_events': sum(1 for e in events if e == 1),
                'median_os': float(np.median(os_vals)) if os_vals else 0.0,
            })

        t1 = time.time()
        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   CREATE (mg:MutationGroup {
                       mutation_key: row.key,
                       gene_list: row.genes,
                       n_patients: row.n_patients,
                       n_events: row.n_events,
                       median_os: row.median_os
                   })""",
                group_rows)
        _step("MutationGroup nodes", actual=n, elapsed=time.time() - t1)

        # Step 5: MEMBER_OF edges (Patient -> MutationGroup)
        t1 = time.time()
        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   MATCH (p:Patient {id: row.pid}),
                         (mg:MutationGroup {mutation_key: row.key})
                   CREATE (p)-[:MEMBER_OF]->(mg)""",
                patient_keys)
        _step("MEMBER_OF edges", actual=n, elapsed=time.time() - t1)

        # Step 6: HAS_GENE edges (MutationGroup -> Gene with direction)
        t1 = time.time()
        gene_edges = []
        for key, patients in groups.items():
            genes = sorted(patients[0]['genes']) if patients[0]['genes'] else []
            for gene in genes:
                func = GENE_FUNCTION.get(gene, 'context')
                # For context genes, use majority direction from the group
                if func == 'context':
                    func = 'GOF'  # default for context at group level
                gene_edges.append({
                    'key': key, 'gene': gene, 'direction': func,
                })

        with self.driver.session() as s:
            n = _batched_write(s,
                """UNWIND $batch AS row
                   MATCH (mg:MutationGroup {mutation_key: row.key}),
                         (g:Gene {name: row.gene})
                   CREATE (mg)-[:HAS_GENE {direction: row.direction}]->(g)""",
                gene_edges)
        _step("HAS_GENE edges", actual=n, elapsed=time.time() - t1)

        # Summary stats
        sizes = [g['n_patients'] for g in group_rows]
        big = [g for g in group_rows if g['n_patients'] >= 10]
        big_patients = sum(g['n_patients'] for g in big)
        print(f"\n  Group size: mean {np.mean(sizes):.1f}, "
              f"median {np.median(sizes):.0f}, max {max(sizes)}", flush=True)
        print(f"  Groups N>=10: {len(big)} covering "
              f"{big_patients:,} patients ({100*big_patients/len(patient_keys):.1f}%)",
              flush=True)

    def _add_group_attention_edges(self):
        """Add ATTENDS_TO edges between MutationGroups.

        For two groups A and B, the attention weight is the mean of all
        directed gene-level attention edges from A's genes to B's genes.
        This captures how the transformer would attend between mutation clusters.

        Only creates edges where both groups have N>=5 patients and mean
        attention exceeds a threshold.
        """
        t0 = time.time()

        # Check if already done
        with self.driver.session() as s:
            r = s.run("""
                MATCH (:MutationGroup)-[r:GROUP_ATTENDS_TO]->(:MutationGroup)
                RETURN count(r) AS c
            """)
            existing = r.single()['c']
            if existing > 0:
                _step("GROUP_ATTENDS_TO edges (already exist)", actual=existing,
                      elapsed=time.time() - t0)
                return

        # Load gene-level attention into Python dict
        attn_lookup = {}
        with self.driver.session() as s:
            result = s.run("""
                MATCH (a:Gene)-[r:ATTENDS_TO]->(b:Gene)
                WHERE r.self_loop IS NULL
                RETURN a.name AS from, b.name AS to, r.weight AS weight
            """)
            for rec in result:
                attn_lookup[(rec['from'], rec['to'])] = rec['weight']

        if not attn_lookup:
            print("  [SKIP] No ATTENDS_TO edges to aggregate.", flush=True)
            return

        # Load groups with their gene lists (only groups with N >= 5)
        with self.driver.session() as s:
            result = s.run("""
                MATCH (mg:MutationGroup)
                WHERE mg.n_patients >= 5 AND size(mg.gene_list) > 0
                RETURN mg.mutation_key AS key, mg.gene_list AS genes
            """)
            groups = [(rec['key'], rec['genes']) for rec in result]

        _step("Groups for group-attention", actual=len(groups),
              elapsed=time.time() - t0)

        # Build inverted index: gene -> [group_keys]
        gene_to_groups = defaultdict(set)
        group_genes = {}
        for key, genes in groups:
            gs = frozenset(genes)
            group_genes[key] = gs
            for g in gs:
                gene_to_groups[g].add(key)

        # Compute group-level attention via shared gene attention
        # Only compare groups that share at least one gene with a gene in the other
        # (via ATTENDS_TO edges — if no gene in A attends to any gene in B, skip)
        t1 = time.time()
        rows = []
        n_compared = 0

        # Build reachable set per gene: which genes does this gene attend to?
        gene_attends_to = defaultdict(set)
        for (a, b) in attn_lookup:
            gene_attends_to[a].add(b)

        for key_a, genes_a in groups:
            # Find candidate groups: groups containing genes that A's genes attend to
            candidate_keys = set()
            for g in genes_a:
                for target in gene_attends_to.get(g, set()):
                    candidate_keys.update(gene_to_groups.get(target, set()))
            candidate_keys.discard(key_a)

            for key_b in candidate_keys:
                if key_a >= key_b:
                    continue  # avoid double-counting; we'll add both directions
                genes_b = group_genes[key_b]
                n_compared += 1

                # A→B: mean attention from A's genes to B's genes
                ab_sum, ab_n = 0.0, 0
                ba_sum, ba_n = 0.0, 0
                for ga in genes_a:
                    for gb in genes_b:
                        w_ab = attn_lookup.get((ga, gb), 0.0)
                        w_ba = attn_lookup.get((gb, ga), 0.0)
                        if w_ab > 0:
                            ab_sum += w_ab; ab_n += 1
                        if w_ba > 0:
                            ba_sum += w_ba; ba_n += 1

                ab_mean = ab_sum / ab_n if ab_n > 0 else 0.0
                ba_mean = ba_sum / ba_n if ba_n > 0 else 0.0

                # Only keep edges above threshold
                min_weight = 0.08
                if ab_mean >= min_weight:
                    rows.append({
                        'from': key_a, 'to': key_b,
                        'weight': round(ab_mean, 4),
                        'n_gene_pairs': ab_n,
                    })
                if ba_mean >= min_weight:
                    rows.append({
                        'from': key_b, 'to': key_a,
                        'weight': round(ba_mean, 4),
                        'n_gene_pairs': ba_n,
                    })

        print(f"  Compared {n_compared:,} group pairs", flush=True)

        if rows:
            with self.driver.session() as s:
                n = _batched_write(s,
                    """UNWIND $batch AS row
                       MATCH (a:MutationGroup {mutation_key: row.from}),
                             (b:MutationGroup {mutation_key: row.to})
                       CREATE (a)-[:GROUP_ATTENDS_TO {
                           weight: row.weight,
                           n_gene_pairs: row.n_gene_pairs
                       }]->(b)""",
                    rows)
            _step("GROUP_ATTENDS_TO edges", actual=n, elapsed=time.time() - t0)
        else:
            _step("GROUP_ATTENDS_TO edges", actual=0, elapsed=time.time() - t0)

    def _validate(self):
        """Cross-check counts."""
        t0 = time.time()
        print(f"\n  --- Validation ---", flush=True)
        with self.driver.session() as s:
            checks = {
                'Patient nodes': "MATCH (n:Patient) RETURN count(n) AS c",
                'MutationGroup nodes': "MATCH (n:MutationGroup) RETURN count(n) AS c",
                'MEMBER_OF edges': "MATCH ()-[r:MEMBER_OF]->() RETURN count(r) AS c",
                'HAS_GENE edges': "MATCH ()-[r:HAS_GENE]->() RETURN count(r) AS c",
                'COUPLES edges': "MATCH ()-[r:COUPLES]->() RETURN count(r) AS c",
                'ATTENDS_TO edges': "MATCH ()-[r:ATTENDS_TO]->() RETURN count(r) AS c",
                'GROUP_ATTENDS_TO edges': (
                    "MATCH (:MutationGroup)-[r:GROUP_ATTENDS_TO]->(:MutationGroup) "
                    "RETURN count(r) AS c"),
                'HAS_MUTATION w/ log_hr': (
                    "MATCH ()-[m:HAS_MUTATION]->() "
                    "WHERE m.log_hr IS NOT NULL RETURN count(m) AS c"),
                'Patients w/ mutation_key': (
                    "MATCH (p:Patient) "
                    "WHERE p.mutation_key IS NOT NULL RETURN count(p) AS c"),
                'Null os_months': (
                    "MATCH (p:Patient) "
                    "WHERE p.os_months IS NULL RETURN count(p) AS c"),
            }
            for name, query in checks.items():
                r = s.run(query)
                val = r.single()['c']
                _step(name, actual=val)
        _step("Validation", elapsed=time.time() - t0)

    # ==================================================================
    # PHASE 2: MAP — walk features per MutationGroup (one query)
    # ==================================================================

    def _load_attention_lookup(self):
        """Load ATTENDS_TO edges and Gene.self_attn into Python dicts for fast lookup."""
        t0 = time.time()
        self._attn_edges = {}  # (gene_from, gene_to) -> weight
        self._attn_asym = {}   # (gene_from, gene_to) -> asymmetry ratio
        self._self_attn = {}   # gene -> self_attn

        with self.driver.session() as s:
            result = s.run("""
                MATCH (a:Gene)-[r:ATTENDS_TO]->(b:Gene)
                RETURN a.name AS from, b.name AS to,
                       r.weight AS weight, r.asymmetry AS asym
            """)
            for rec in result:
                key = (rec['from'], rec['to'])
                self._attn_edges[key] = rec['weight']
                self._attn_asym[key] = rec['asym']

            result = s.run("""
                MATCH (g:Gene) WHERE g.self_attn IS NOT NULL
                RETURN g.name AS gene, g.self_attn AS self_attn
            """)
            for rec in result:
                self._self_attn[rec['gene']] = rec['self_attn']

        _step("Attention lookup loaded",
              actual=len(self._attn_edges),
              elapsed=time.time() - t0)

    def walk_groups(self):
        """Compute walk topology features per MutationGroup.

        Returns DataFrame with one row per group.
        All computed in a single Cypher query + Python pair-type computation.
        """
        print(f"\n{'='*70}")
        print("  MAP: Walk features per MutationGroup")
        print(f"{'='*70}\n")

        t0 = time.time()

        # Load attention lookup for feature computation
        self._load_attention_lookup()

        # Single query: get each group's gene names + directions
        # Channel lookup done in Python from CHANNEL_MAP (6-channel, not V6 8-channel)
        _hub_set = set()
        for hubs in HUB_GENES.values():
            _hub_set.update(hubs)

        with self.driver.session() as s:
            result = s.run("""
                MATCH (mg:MutationGroup)
                OPTIONAL MATCH (mg)-[hg:HAS_GENE]->(g:Gene)
                WITH mg, collect({gene: g.name, direction: hg.direction}) AS genes
                RETURN mg.mutation_key AS mutation_key,
                       mg.n_patients AS n_patients,
                       mg.n_events AS n_events,
                       mg.median_os AS median_os,
                       mg.gene_list AS gene_list,
                       genes
            """)
            rows = list(result)

        _step("Groups fetched", actual=len(rows), elapsed=time.time() - t0)

        # Compute features in Python using CHANNEL_MAP (6-channel)
        t1 = time.time()
        records = []
        for row in rows:
            raw_genes = row['genes']
            # Filter out null genes (WILDTYPE group), resolve channel in Python
            genes = []
            for g in raw_genes:
                if g['gene'] is None:
                    continue
                name = g['gene']
                genes.append({
                    'gene': name,
                    'channel': CHANNEL_MAP.get(name),  # 6-channel map
                    'direction': g['direction'],
                    'is_hub': name in _hub_set,
                })

            rec = {
                'mutation_key': row['mutation_key'],
                'n_patients': row['n_patients'],
                'n_events': row['n_events'],
                'median_os': row['median_os'],
                'n_genes': len(genes),
                'n_gof': sum(1 for g in genes if g['direction'] == 'GOF'),
                'n_lof': sum(1 for g in genes if g['direction'] == 'LOF'),
            }

            # Per-channel direction flags
            for ch in CHANNEL_NAMES:
                rec[f'ch_GOF_{ch}'] = int(any(
                    g['channel'] == ch and g['direction'] == 'GOF' for g in genes))
                rec[f'ch_LOF_{ch}'] = int(any(
                    g['channel'] == ch and g['direction'] == 'LOF' for g in genes))

            # Channels severed, hub count
            channels_hit = set(g['channel'] for g in genes if g['channel'])
            n_hub = sum(1 for g in genes if g['is_hub'])
            rec['n_channels_severed'] = len(channels_hit)
            rec['n_hub_hit'] = n_hub
            rec['hub_ratio'] = n_hub / max(len(genes), 1)
            total_dir = rec['n_gof'] + rec['n_lof']
            rec['gof_lof_ratio'] = rec['n_gof'] / max(total_dir, 1)

            # Pair types (combinations of this group's genes)
            pair_counts = defaultdict(int)
            cross_gof_lof_channels = set()
            for a, b in combinations(genes, 2):
                same_ch = a['channel'] == b['channel']
                locality = 'within' if same_ch else 'cross'
                dirs = sorted([a['direction'] or '', b['direction'] or ''])
                if dirs == ['GOF', 'LOF']:
                    pair_type = 'GOF_LOF'
                    if not same_ch and a['channel'] and b['channel']:
                        cross_gof_lof_channels.add(
                            tuple(sorted([a['channel'], b['channel']])))
                elif dirs == ['GOF', 'GOF']:
                    pair_type = 'GOF_GOF'
                elif dirs == ['LOF', 'LOF']:
                    pair_type = 'LOF_LOF'
                else:
                    pair_type = 'other'
                pair_counts[f'{pair_type}_{locality}'] += 1

            for pt in ['GOF_LOF_within', 'GOF_LOF_cross',
                        'GOF_GOF_within', 'GOF_GOF_cross',
                        'LOF_LOF_within', 'LOF_LOF_cross']:
                rec[pt] = pair_counts.get(pt, 0)
            rec['cross_gof_lof_channels'] = len(cross_gof_lof_channels)

            # Specific pathway flags
            rec['path_PI3K_GOF_x_CC_LOF'] = int(
                rec['ch_GOF_PI3K_Growth'] and rec['ch_LOF_CellCycle'])
            rec['path_PI3K_GOF_x_TA_LOF'] = int(
                rec['ch_GOF_PI3K_Growth'] and rec['ch_LOF_TissueArch'])
            rec['path_CC_GOF_x_DDR_LOF'] = int(
                rec['ch_GOF_CellCycle'] and rec['ch_LOF_DDR'])
            rec['path_anyGOF_x_Immune_LOF'] = int(
                rec['n_gof'] > 0 and rec['ch_LOF_Immune'])

            # --- Attention-derived features ---
            gene_names = [g['gene'] for g in genes]

            # 1. Total attention flow between this group's genes
            #    Sum of directed attention weights for all gene pairs
            attn_flow = 0.0
            attn_cross_flow = 0.0  # cross-channel only
            max_attn = 0.0
            max_asym = 0.0
            n_attn_edges = 0
            for a in genes:
                for b in genes:
                    if a['gene'] == b['gene']:
                        continue
                    key = (a['gene'], b['gene'])
                    w = self._attn_edges.get(key, 0.0)
                    if w > 0:
                        attn_flow += w
                        n_attn_edges += 1
                        max_attn = max(max_attn, w)
                        asym = self._attn_asym.get(key, 1.0)
                        max_asym = max(max_asym, asym)
                        if a['channel'] != b['channel']:
                            attn_cross_flow += w

            n_pairs = max(len(genes) * (len(genes) - 1), 1)
            rec['attn_flow'] = attn_flow
            rec['attn_mean'] = attn_flow / n_pairs
            rec['attn_cross_flow'] = attn_cross_flow
            rec['attn_max'] = max_attn
            rec['attn_max_asym'] = max_asym
            rec['attn_density'] = n_attn_edges / n_pairs

            # 2. Self-attention sum: how important are these genes individually
            self_attn_sum = sum(self._self_attn.get(g['gene'], 0.0)
                                for g in genes)
            rec['self_attn_sum'] = self_attn_sum

            # 3. Attention asymmetry: directional dominance
            #    Which direction is stronger — does this group have "drivers"
            #    that context other genes more than they are contextualized?
            out_attn = defaultdict(float)  # gene -> total outgoing attention
            in_attn = defaultdict(float)   # gene -> total incoming attention
            for a in genes:
                for b in genes:
                    if a['gene'] == b['gene']:
                        continue
                    w = self._attn_edges.get((a['gene'], b['gene']), 0.0)
                    out_attn[a['gene']] += w
                    in_attn[b['gene']] += w

            # Max out/in ratio: strongest "contextualizer" gene
            if gene_names:
                ratios = []
                for g in gene_names:
                    o = out_attn.get(g, 0.0)
                    i = in_attn.get(g, 0.0)
                    if i > 0.01:
                        ratios.append(o / i)
                rec['attn_max_driver_ratio'] = max(ratios) if ratios else 1.0
            else:
                rec['attn_max_driver_ratio'] = 1.0

            records.append(rec)

        group_df = pd.DataFrame(records)
        _step("Group features computed", actual=len(group_df),
              elapsed=time.time() - t1)

        return group_df

    # ==================================================================
    # PHASE 3: REDUCE — join group features to patients
    # ==================================================================

    def walk_patients(self, group_df):
        """Join group features to patients and add patient-specific features.

        Returns DataFrame with one row per patient.
        """
        print(f"\n{'='*70}")
        print("  REDUCE: Join group features to patients")
        print(f"{'='*70}\n")

        t0 = time.time()

        # Single query: patient-level atlas_sum + survival data
        with self.driver.session() as s:
            result = s.run("""
                MATCH (p:Patient)
                OPTIONAL MATCH (p)-[m:HAS_MUTATION]->(g:Gene)
                RETURN p.id AS patientId,
                       p.mutation_key AS mutation_key,
                       p.cancer_type AS cancer_type,
                       p.os_months AS OS_MONTHS,
                       p.event AS event,
                       p.tissue_delta AS tissue_delta,
                       coalesce(sum(m.log_hr), 0.0) AS atlas_sum
            """)
            patient_rows = list(result)

        patient_df = pd.DataFrame([dict(r) for r in patient_rows])
        _step("Patient atlas_sum computed", actual=len(patient_df),
              elapsed=time.time() - t0)

        # Join group features on mutation_key
        t1 = time.time()
        # Drop group-level stats that don't belong on patient rows
        group_cols = [c for c in group_df.columns
                      if c not in ('n_patients', 'n_events', 'median_os')]
        merged = patient_df.merge(group_df[group_cols],
                                   on='mutation_key', how='left')

        # Fill NaN for patients with no group match (shouldn't happen)
        for col in merged.select_dtypes(include=[np.number]).columns:
            merged[col] = merged[col].fillna(0)

        _step("Patient-group join", actual=len(merged),
              elapsed=time.time() - t1)

        return merged

    # ==================================================================
    # PHASE 4: GROUP AFFINITY (via shared Gene nodes in Cypher)
    # ==================================================================

    def compute_group_affinity(self, min_jaccard=0.3, max_neighbors=20):
        """Compute Jaccard affinity between MutationGroups via shared genes.

        Returns DataFrame of (group_a, group_b, jaccard, shared_genes).
        """
        print(f"\n{'='*70}")
        print("  GROUP AFFINITY (Jaccard via shared Gene nodes)")
        print(f"{'='*70}\n")

        t0 = time.time()

        # The full cross-join blows Neo4j memory. Instead, compute in Python
        # from the group gene lists (already in memory from walk_groups).
        # This is fast: ~14K groups, each with <10 genes.
        with self.driver.session() as s:
            result = s.run("""
                MATCH (mg:MutationGroup)
                WHERE size(mg.gene_list) > 0
                RETURN mg.mutation_key AS key,
                       mg.gene_list AS genes,
                       mg.n_patients AS n_patients
            """)
            groups = list(result)

        _step("Groups fetched for affinity", actual=len(groups),
              elapsed=time.time() - t0)

        # Cascaded affinity: avoid comparing groups that can't reach min_jaccard.
        #
        # Key insight: J(A,B) = |A∩B| / |A∪B| >= min_jaccard requires
        #   |A∩B| >= min_jaccard * (|A| + |B| - |A∩B|)
        #   |A∩B| * (1 + min_jaccard) >= min_jaccard * (|A| + |B|)
        #   |A∩B| >= min_jaccard * (|A| + |B|) / (1 + min_jaccard)
        #
        # So for J >= 0.3: shared >= 0.231 * (|A| + |B|)
        # If |A|=1, |B|=3: need shared >= 0.92 → 1. Possible (if they share the gene).
        # If |A|=1, |B|=4: need shared >= 1.15 → impossible with shared <= 1. SKIP.
        #
        # Also: groups with identical gene sets have J=1.0, handle first.

        t1 = time.time()
        gene_to_groups = defaultdict(set)
        group_genes = {}
        group_n = {}
        group_size = {}
        for g in groups:
            key = g['key']
            genes = frozenset(g['genes'])
            group_genes[key] = genes
            group_n[key] = g['n_patients']
            group_size[key] = len(genes)
            for gene in genes:
                gene_to_groups[gene].add(key)

        # Pre-compute: max partner size for each group size
        # shared >= min_jacc * (s1 + s2) / (1 + min_jacc), shared <= min(s1, s2)
        # So: min(s1, s2) >= min_jacc * (s1 + s2) / (1 + min_jacc)
        # For s1 <= s2: s1 * (1 + min_jacc) >= min_jacc * (s1 + s2)
        #   s1 + s1*min_jacc >= min_jacc*s1 + min_jacc*s2
        #   s1 >= min_jacc * s2
        #   s2 <= s1 / min_jacc
        # So a group of size s1 can only match groups up to size s1/min_jacc.
        max_partner_ratio = 1.0 / min_jaccard  # 3.33 for J >= 0.3

        aff_rows = []
        n_compared = 0
        n_skipped_size = 0

        for key, genes in group_genes.items():
            s1 = group_size[key]
            max_s2 = int(s1 * max_partner_ratio)

            # Candidates: groups sharing at least one gene
            candidates = set()
            for gene in genes:
                candidates.update(gene_to_groups[gene])
            candidates.discard(key)

            for cand in candidates:
                if key >= cand:  # avoid double-counting
                    continue

                s2 = group_size[cand]
                # Size filter: skip if impossible to reach min_jaccard
                if s2 > max_s2 or s1 > int(s2 * max_partner_ratio):
                    n_skipped_size += 1
                    continue

                cand_genes = group_genes[cand]
                shared = len(genes & cand_genes)

                # Shared threshold: need shared >= min_jacc * (s1+s2) / (1+min_jacc)
                min_shared = min_jaccard * (s1 + s2) / (1 + min_jaccard)
                if shared < min_shared:
                    n_compared += 1
                    continue

                jacc = shared / len(genes | cand_genes)
                n_compared += 1
                if jacc >= min_jaccard:
                    aff_rows.append({
                        'group_a': key, 'group_b': cand,
                        'jaccard': jacc, 'shared': shared,
                        'n_a': group_n.get(key, 0),
                        'n_b': group_n.get(cand, 0),
                    })

        print(f"  Compared: {n_compared:,}, skipped by size: {n_skipped_size:,}",
              flush=True)

        aff_df = pd.DataFrame(aff_rows) if aff_rows else pd.DataFrame()
        _step("Affinity pairs (J >= {:.1f})".format(min_jaccard),
              actual=len(aff_df), elapsed=time.time() - t0)

        if len(aff_df) > 0:
            print(f"  Mean Jaccard: {aff_df['jaccard'].mean():.3f}", flush=True)
            print(f"  Covering groups: {aff_df['group_a'].nunique()} + "
                  f"{aff_df['group_b'].nunique()}", flush=True)

        return aff_df

    # ==================================================================
    # FULL PIPELINE
    # ==================================================================

    def run(self):
        """Full pipeline: augment, map, reduce, Cox."""
        from sklearn.model_selection import StratifiedKFold
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index as ci

        t_total = time.time()

        # 1. Augment graph
        self.augment_graph()

        # 2. Map: group-level features
        group_df = self.walk_groups()

        # 3. Reduce: patient-level features
        df = self.walk_patients(group_df)
        df = df[df['OS_MONTHS'] > 0].copy()

        # 4. Group affinity
        aff_df = self.compute_group_affinity()

        # Add affinity stats per group
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

        # 4b. Group attention neighborhood features
        print(f"\n{'='*70}")
        print("  GROUP ATTENTION NEIGHBORHOOD")
        print(f"{'='*70}\n")

        t_ga = time.time()
        with self.driver.session() as s:
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
                df[col] = df[col].fillna(0.0)
            _step("Group attention features added", actual=len(ga_df),
                  elapsed=time.time() - t_ga)
        else:
            for col in ['n_attn_neighbors', 'mean_attn_to_neighbors',
                        'max_attn_to_neighbors', 'attn_neighbor_median_os',
                        'attn_weighted_os']:
                df[col] = 0.0
            _step("Group attention features (none)", actual=0,
                  elapsed=time.time() - t_ga)

        # 5. Cox regression
        print(f"\n{'='*70}")
        print("  COX REGRESSION")
        print(f"{'='*70}\n")

        # Standardize
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ('OS_MONTHS', 'event'):
                continue
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - df[col].mean()) / std

        features = [
            'atlas_sum', 'tissue_delta', 'n_gof', 'n_lof', 'n_channels_severed',
            'GOF_LOF_within', 'GOF_LOF_cross',
            'GOF_GOF_cross', 'LOF_LOF_within',
            'path_PI3K_GOF_x_CC_LOF', 'path_PI3K_GOF_x_TA_LOF',
            'path_CC_GOF_x_DDR_LOF', 'path_anyGOF_x_Immune_LOF',
            'n_neighbors', 'mean_jacc',
            # Attention-derived features (gene-level)
            'attn_flow', 'attn_mean', 'attn_cross_flow', 'attn_max',
            'attn_max_asym', 'attn_density', 'self_attn_sum',
            'attn_max_driver_ratio',
            # Attention-derived features (group-level)
            'n_attn_neighbors', 'mean_attn_to_neighbors',
            'max_attn_to_neighbors', 'attn_neighbor_median_os',
            'attn_weighted_os',
        ]
        valid = [f for f in features if f in df.columns and df[f].std() > 0.001]
        _step("Features for Cox", actual=len(valid))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        c_indices = []

        for fold, (ti, vi) in enumerate(skf.split(df, df['event'])):
            train, val = df.iloc[ti], df.iloc[vi]
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(train[valid + ['OS_MONTHS', 'event']],
                    duration_col='OS_MONTHS', event_col='event')
            h = cph.predict_partial_hazard(val[valid]).values.flatten()
            c = ci(val['OS_MONTHS'].values, -h, val['event'].values)
            c_indices.append(c)
            print(f"  Fold {fold}: C={c:.4f}", flush=True)

        mean_c = np.mean(c_indices)
        std_c = np.std(c_indices)

        # 5b. Gradient Boosted Survival (non-linear — can capture interactions)
        print(f"\n{'='*70}")
        print("  GRADIENT BOOSTED SURVIVAL (non-linear)")
        print(f"{'='*70}\n")

        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored

        # sksurv needs structured array for y
        def make_y(sub_df):
            return np.array(
                [(bool(e), t) for e, t in zip(sub_df['event'], sub_df['OS_MONTHS'])],
                dtype=[('event', bool), ('time', float)]
            )

        gb_c_indices = []
        gb_feature_imp = np.zeros(len(valid))

        for fold, (ti, vi) in enumerate(skf.split(df, df['event'])):
            train_x = df.iloc[ti][valid].values
            val_x = df.iloc[vi][valid].values
            train_y = make_y(df.iloc[ti])
            val_y = make_y(df.iloc[vi])

            gbm = GradientBoostingSurvivalAnalysis(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=50, subsample=0.8, random_state=42,
            )
            gbm.fit(train_x, train_y)

            pred = gbm.predict(val_x)
            c_val = concordance_index_censored(val_y['event'], val_y['time'], pred)
            gb_c_indices.append(c_val[0])
            gb_feature_imp += gbm.feature_importances_
            print(f"  Fold {fold}: C={c_val[0]:.4f}", flush=True)

        gb_mean = np.mean(gb_c_indices)
        gb_std = np.std(gb_c_indices)

        # Feature importance
        gb_feature_imp /= 5
        imp_df = pd.DataFrame({
            'feature': valid,
            'importance': gb_feature_imp
        }).sort_values('importance', ascending=False)

        print(f"\n  Top 15 features (GBM importance):")
        for _, row in imp_df.head(15).iterrows():
            bar = '#' * int(row['importance'] * 200)
            print(f"    {row['feature']:30s} {row['importance']:.4f}  {bar}",
                  flush=True)

        # Compare Cox vs GBM on attention features specifically
        # Run GBM without attention features to measure lift
        non_attn = [f for f in valid if not f.startswith('attn_')
                    and f not in ('self_attn_sum', 'n_attn_neighbors',
                                  'mean_attn_to_neighbors',
                                  'max_attn_to_neighbors',
                                  'attn_neighbor_median_os',
                                  'attn_weighted_os')]

        gb_no_attn = []
        for fold, (ti, vi) in enumerate(skf.split(df, df['event'])):
            train_x = df.iloc[ti][non_attn].values
            val_x = df.iloc[vi][non_attn].values
            train_y = make_y(df.iloc[ti])
            val_y = make_y(df.iloc[vi])

            gbm2 = GradientBoostingSurvivalAnalysis(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=50, subsample=0.8, random_state=42,
            )
            gbm2.fit(train_x, train_y)
            pred2 = gbm2.predict(val_x)
            c2 = concordance_index_censored(val_y['event'], val_y['time'], pred2)
            gb_no_attn.append(c2[0])

        gb_no_attn_mean = np.mean(gb_no_attn)
        attn_lift = gb_mean - gb_no_attn_mean

        print(f"\n  GBM without attention features: C={gb_no_attn_mean:.4f}")
        print(f"  GBM with attention features:    C={gb_mean:.4f}")
        print(f"  Attention edge lift:            +{attn_lift:.4f}")

        # Equivalence group analysis
        print(f"\n{'='*70}")
        print("  EQUIVALENCE GROUP ANALYSIS")
        print(f"{'='*70}\n")

        # Most common mutation combos
        group_survival = group_df[group_df['n_patients'] >= 20].sort_values(
            'n_patients', ascending=False).head(20)

        if len(group_survival) > 0:
            # Get gene_list from neo4j for readable names
            with self.driver.session() as s:
                result = s.run("""
                    MATCH (mg:MutationGroup)
                    WHERE mg.n_patients >= 20
                    RETURN mg.mutation_key AS key, mg.gene_list AS genes,
                           mg.n_patients AS n, mg.n_events AS events,
                           mg.median_os AS median_os
                    ORDER BY mg.n_patients DESC LIMIT 20
                """)
                top_groups = list(result)

            print(f"  {'Mutation Combo':40s} {'N':>5s} {'Events':>6s} "
                  f"{'MedOS':>6s}", flush=True)
            print(f"  {'-'*65}", flush=True)
            for g in top_groups:
                genes = g['genes'] or ['WILDTYPE']
                label = '+'.join(genes[:4])
                if len(genes) > 4:
                    label += f'+{len(genes)-4}more'
                print(f"  {label:40s} {g['n']:5d} {g['events']:6d} "
                      f"{g['median_os']:6.1f}", flush=True)

        # Final summary
        print(f"\n{'='*70}")
        print(f"  RESULTS")
        print(f"{'='*70}")
        print(f"  Cox (linear):                 C = {mean_c:.4f} +/- {std_c:.4f}")
        print(f"  GBM (non-linear):             C = {gb_mean:.4f} +/- {gb_std:.4f}")
        print(f"  GBM without attention edges:  C = {gb_no_attn_mean:.4f}")
        print(f"  Attention edge lift (GBM):    +{attn_lift:.4f}")
        print(f"  Total pipeline: {time.time() - t_total:.1f}s")
        print(f"")
        print(f"  BASELINES:")
        print(f"  Atlas lookup (zero-param):    C = 0.577")
        print(f"  Directional walk (Python):    C = 0.636")
        print(f"  AtlasTransformer V1 (neural): C = 0.673")


def main():
    engine = Neo4jWalkEngine()
    try:
        engine.run()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
