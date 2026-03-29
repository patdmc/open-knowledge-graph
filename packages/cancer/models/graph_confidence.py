"""
Graph confidence scorer — measures how well-supported the graph's claims are.

This is NOT about predicting patient outcomes. It's about measuring
internal consistency of the knowledge graph itself. Each edge is a claim
about biology. Confidence = how well do these claims support each other?

Five signals, computed purely from graph structure (no patient data):

1. Witness agreement: Do independent edge-type subgraphs produce
   consistent gene embeddings? PPI says A≈B, does COOCCURS agree?

2. Link prediction: Can the graph predict its own edges from context?
   High reconstruction = internally consistent structure.

3. Triangle closure: Edges participating in heterogeneous triangles
   (multiple edge types) are better supported than isolated edges.

4. Provenance diversity: How many independent evidence types support
   each edge? Single-source edges are low confidence.

5. Bootstrap stability: How much do gene embeddings shift when you
   drop 10% of edges? Stable = well-grounded. Volatile = fragile.

Usage:
    from gnn.models.graph_confidence import GraphConfidence
    gc = GraphConfidence()
    gc.load_graph()
    report = gc.full_report()
    print(report['composite_score'])
"""

import os
import json
import time
import numpy as np
from collections import defaultdict


def _neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver("bolt://localhost:7687",
                                auth=("neo4j", "openknowledgegraph"))


class GraphConfidence:
    """Measures internal consistency and support quality of the knowledge graph."""

    def __init__(self):
        self.genes = []          # ordered gene list
        self.gene_idx = {}       # gene name → index
        self.channels = {}       # gene → channel name
        self.blocks = {}         # gene → block_id

        # Per edge-type adjacency: {etype: {(i,j): props_dict}}
        self.typed_edges = defaultdict(dict)

        # Merged adjacency: {(i,j): [list of edge types]}
        self.edge_types_present = defaultdict(list)

        # All numeric features per edge: {(i,j): {etype: {prop: val}}}
        self.edge_features = defaultdict(dict)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_graph(self):
        """Pull full gene-gene graph from Neo4j."""
        t0 = time.time()
        driver = _neo4j_driver()

        with driver.session() as s:
            # Genes
            result = s.run("""
                MATCH (g:Gene)
                WHERE g.channel IS NOT NULL
                RETURN g.name AS name, g.channel AS channel
                ORDER BY g.channel, g.name
            """)
            for r in result:
                idx = len(self.genes)
                self.genes.append(r['name'])
                self.gene_idx[r['name']] = idx
                self.channels[r['name']] = r['channel']

            # Block assignments
            from gnn.data.block_assignments import load_block_assignments
            ba, _, _ = load_block_assignments()
            for gene, info in ba.items():
                if gene in self.gene_idx:
                    self.blocks[gene] = info['block_id']

            # All edges
            result = s.run("""
                MATCH (a:Gene)-[r]->(b:Gene)
                WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
                  AND (r.deprecated IS NULL OR r.deprecated = false)
                RETURN a.name AS g1, b.name AS g2, type(r) AS rtype,
                       properties(r) AS props
            """)
            n_edges = 0
            for r in result:
                g1, g2 = r['g1'], r['g2']
                if g1 not in self.gene_idx or g2 not in self.gene_idx:
                    continue
                i, j = self.gene_idx[g1], self.gene_idx[g2]
                # Canonical order
                a, b = min(i, j), max(i, j)
                etype = r['rtype']

                props = {}
                for k, v in (r['props'] or {}).items():
                    if isinstance(v, (int, float)) and k not in ('created_at', 'updated_at'):
                        props[k] = float(v)

                self.typed_edges[etype][(a, b)] = props
                if etype not in self.edge_types_present[(a, b)]:
                    self.edge_types_present[(a, b)].append(etype)
                self.edge_features[(a, b)][etype] = props
                n_edges += 1

        driver.close()
        n_pairs = len(self.edge_types_present)
        print(f"Loaded {len(self.genes)} genes, {n_edges} edges "
              f"({n_pairs} unique pairs) across "
              f"{len(self.typed_edges)} edge types [{time.time()-t0:.1f}s]")

    # ------------------------------------------------------------------
    # Signal 1: Witness agreement
    # ------------------------------------------------------------------

    def witness_agreement(self, n_dims=32, n_iter=10):
        """Do independent edge-type subgraphs produce consistent embeddings?

        For each edge type with enough coverage, run simple spectral embedding.
        Measure pairwise cosine similarity of gene embeddings across witnesses.
        High agreement = the different data sources see the same structure.

        Returns:
            per_pair: dict (etype_a, etype_b) → mean cosine similarity
            overall: float — mean agreement across all witness pairs
        """
        t0 = time.time()
        G = len(self.genes)

        # Only use edge types with meaningful coverage
        MIN_EDGES = 50
        eligible = {et: edges for et, edges in self.typed_edges.items()
                    if len(edges) >= MIN_EDGES}

        if len(eligible) < 2:
            return {'per_pair': {}, 'overall': 0.0}

        # Spectral embedding per edge type
        embeddings = {}
        for etype, edges in eligible.items():
            # Build adjacency
            adj = np.zeros((G, G), dtype=np.float32)
            for (i, j), props in edges.items():
                # Use strongest numeric property as weight, default 1.0
                w = 1.0
                for k, v in props.items():
                    if k in ('score', 'correlation', 'weight', 'count',
                             'abs_correlation'):
                        w = max(w, abs(v))
                adj[i, j] = w
                adj[j, i] = w

            # Normalized Laplacian embedding
            degree = adj.sum(axis=1)
            degree[degree == 0] = 1.0
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
            L_norm = np.eye(G) - D_inv_sqrt @ adj @ D_inv_sqrt

            # Eigenvectors of smallest eigenvalues (skip first = trivial)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
                # Take dims 1..n_dims+1 (skip constant eigenvector)
                k = min(n_dims + 1, G)
                emb = eigenvectors[:, 1:k]
                # Normalize rows
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb = emb / norms
                embeddings[etype] = emb
            except np.linalg.LinAlgError:
                continue

        # Pairwise cosine similarity between edge-type embeddings
        etypes = sorted(embeddings.keys())
        per_pair = {}
        sims = []
        for ii in range(len(etypes)):
            for jj in range(ii + 1, len(etypes)):
                ea, eb = etypes[ii], etypes[jj]
                # Only compare genes that appear in BOTH subgraphs
                genes_a = set()
                for (i, j) in self.typed_edges[ea]:
                    genes_a.add(i)
                    genes_a.add(j)
                genes_b = set()
                for (i, j) in self.typed_edges[eb]:
                    genes_b.add(i)
                    genes_b.add(j)
                shared = sorted(genes_a & genes_b)

                if len(shared) < 10:
                    continue

                emb_a = embeddings[ea][shared]
                emb_b = embeddings[eb][shared]

                # Cosine similarity per gene, averaged
                dots = (emb_a * emb_b).sum(axis=1)
                cos_sim = float(np.mean(dots))
                per_pair[(ea, eb)] = cos_sim
                sims.append(cos_sim)

        overall = float(np.mean(sims)) if sims else 0.0
        print(f"  Witness agreement: {overall:.3f} "
              f"({len(per_pair)} pairs) [{time.time()-t0:.1f}s]")

        return {'per_pair': per_pair, 'overall': overall}

    # ------------------------------------------------------------------
    # Signal 2: Link prediction (graph autoencoder)
    # ------------------------------------------------------------------

    def link_prediction(self, mask_frac=0.2, n_dims=32, n_trials=5):
        """Can the graph predict its own edges?

        Hold out mask_frac of edges, embed from the rest,
        score held-out vs random negatives. Returns AUC.
        High AUC = edges are predictable from context = internally consistent.
        """
        from sklearn.metrics import roc_auc_score
        t0 = time.time()
        G = len(self.genes)

        all_edges = list(self.edge_types_present.keys())
        all_edge_set = set(all_edges)

        aucs = []
        for trial in range(n_trials):
            rng = np.random.RandomState(42 + trial)
            n_mask = int(len(all_edges) * mask_frac)
            mask_idx = rng.choice(len(all_edges), n_mask, replace=False)
            mask_set = set(mask_idx)

            # Build adjacency from remaining edges
            adj = np.zeros((G, G), dtype=np.float32)
            for idx, (i, j) in enumerate(all_edges):
                if idx in mask_set:
                    continue
                n_types = len(self.edge_types_present[(i, j)])
                adj[i, j] = n_types
                adj[j, i] = n_types

            # Spectral embedding
            degree = adj.sum(axis=1)
            degree[degree == 0] = 1.0
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
            L_norm = np.eye(G) - D_inv_sqrt @ adj @ D_inv_sqrt

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
                k = min(n_dims + 1, G)
                emb = eigenvectors[:, 1:k]
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb = emb / norms
            except np.linalg.LinAlgError:
                continue

            # Score: cosine similarity for masked edges (positive) vs random (negative)
            positives = [all_edges[idx] for idx in mask_idx]
            # Random negatives: same count, not in edge set
            negatives = []
            while len(negatives) < len(positives):
                a = rng.randint(0, G)
                b = rng.randint(0, G)
                if a == b:
                    continue
                pair = (min(a, b), max(a, b))
                if pair not in all_edge_set:
                    negatives.append(pair)

            scores = []
            labels = []
            for (i, j) in positives:
                scores.append(float(np.dot(emb[i], emb[j])))
                labels.append(1)
            for (i, j) in negatives:
                scores.append(float(np.dot(emb[i], emb[j])))
                labels.append(0)

            try:
                auc = roc_auc_score(labels, scores)
                aucs.append(auc)
            except ValueError:
                pass

        mean_auc = float(np.mean(aucs)) if aucs else 0.5
        print(f"  Link prediction AUC: {mean_auc:.3f} "
              f"({n_trials} trials) [{time.time()-t0:.1f}s]")

        return {'auc': mean_auc, 'trials': aucs}

    # ------------------------------------------------------------------
    # Signal 3: Triangle closure (heterogeneous support)
    # ------------------------------------------------------------------

    def triangle_closure(self):
        """Count heterogeneous triangles per edge.

        An edge (A,B) participating in triangle A-B-C where A-B, B-C, A-C
        use different edge types is better supported than an isolated edge.

        Returns:
            per_edge: dict (i,j) → n_heterogeneous_triangles
            coverage: fraction of edges in at least one heterogeneous triangle
            mean_triangles: mean triangle count across all edges
        """
        t0 = time.time()

        # Build neighbor sets per gene
        neighbors = defaultdict(set)
        for (i, j) in self.edge_types_present:
            neighbors[i].add(j)
            neighbors[j].add(i)

        per_edge = {}
        for (i, j), etypes_ij in self.edge_types_present.items():
            shared_neighbors = neighbors[i] & neighbors[j]
            het_count = 0

            for k in shared_neighbors:
                ik = (min(i, k), max(i, k))
                jk = (min(j, k), max(j, k))

                etypes_ik = set(self.edge_types_present.get(ik, []))
                etypes_jk = set(self.edge_types_present.get(jk, []))
                etypes_ij_set = set(etypes_ij)

                # Heterogeneous = at least 2 different edge types across triangle
                all_types = etypes_ij_set | etypes_ik | etypes_jk
                if len(all_types) >= 2:
                    het_count += 1

            per_edge[(i, j)] = het_count

        n_edges = len(per_edge)
        n_in_triangle = sum(1 for v in per_edge.values() if v > 0)
        coverage = n_in_triangle / max(n_edges, 1)
        mean_tri = float(np.mean(list(per_edge.values()))) if per_edge else 0.0

        print(f"  Triangle closure: {coverage:.1%} of edges in heterogeneous "
              f"triangles, mean={mean_tri:.1f} [{time.time()-t0:.1f}s]")

        return {
            'per_edge': per_edge,
            'coverage': coverage,
            'mean_triangles': mean_tri,
        }

    # ------------------------------------------------------------------
    # Signal 4: Provenance diversity
    # ------------------------------------------------------------------

    def provenance_diversity(self):
        """How many independent evidence types support each edge?

        Returns:
            distribution: list of (n_types, count) — how many edges have 1, 2, 3... types
            per_edge: dict (i,j) → n_types
            mean_types: float
            low_confidence_edges: list of (gene_a, gene_b, types) with only 1 type
        """
        t0 = time.time()

        per_edge = {}
        dist = defaultdict(int)

        for (i, j), etypes in self.edge_types_present.items():
            n = len(etypes)
            per_edge[(i, j)] = n
            dist[n] += 1

        mean_types = float(np.mean(list(per_edge.values()))) if per_edge else 0.0

        # Low confidence: only 1 evidence type (excluding COOCCURS-only which is just stats)
        low_conf = []
        for (i, j), etypes in self.edge_types_present.items():
            if len(etypes) == 1:
                low_conf.append((self.genes[i], self.genes[j], etypes[0]))

        distribution = sorted(dist.items())

        print(f"  Provenance diversity: mean {mean_types:.2f} types/edge")
        for n_types, count in distribution:
            print(f"    {n_types} type(s): {count:,} edges "
                  f"({count/len(per_edge)*100:.1f}%)")
        print(f"  Low confidence (single source): {len(low_conf):,} edges "
              f"[{time.time()-t0:.1f}s]")

        return {
            'distribution': distribution,
            'per_edge': per_edge,
            'mean_types': mean_types,
            'low_confidence_edges': low_conf[:100],  # cap for memory
            'n_low_confidence': len(low_conf),
        }

    # ------------------------------------------------------------------
    # Signal 5: Bootstrap stability
    # ------------------------------------------------------------------

    def bootstrap_stability(self, n_bootstrap=10, drop_frac=0.1, n_dims=32):
        """How stable are gene embeddings when edges are randomly dropped?

        Genes with stable embeddings are well-grounded — many supporting edges.
        Genes with volatile embeddings depend on a few critical edges.

        Returns:
            per_gene: dict gene → mean_cosine_stability (1.0 = perfectly stable)
            overall: float — mean stability across all genes
            fragile_genes: list of (gene, stability) for bottom 20
        """
        t0 = time.time()
        G = len(self.genes)
        all_edges = list(self.edge_types_present.keys())

        # Base embedding (all edges)
        adj_full = np.zeros((G, G), dtype=np.float32)
        for (i, j), etypes in self.edge_types_present.items():
            w = len(etypes)
            adj_full[i, j] = w
            adj_full[j, i] = w

        def spectral_embed(adj_matrix):
            degree = adj_matrix.sum(axis=1)
            degree[degree == 0] = 1.0
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
            L_norm = np.eye(G) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
            k = min(n_dims + 1, G)
            emb = eigenvectors[:, 1:k]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return emb / norms

        try:
            base_emb = spectral_embed(adj_full)
        except np.linalg.LinAlgError:
            return {'per_gene': {}, 'overall': 0.0, 'fragile_genes': []}

        # Bootstrap: drop edges, re-embed, measure shift
        stability_scores = np.zeros((n_bootstrap, G))

        for b in range(n_bootstrap):
            rng = np.random.RandomState(b)
            n_drop = int(len(all_edges) * drop_frac)
            drop_idx = set(rng.choice(len(all_edges), n_drop, replace=False))

            adj_boot = np.zeros((G, G), dtype=np.float32)
            for idx, (i, j) in enumerate(all_edges):
                if idx in drop_idx:
                    continue
                w = len(self.edge_types_present[(i, j)])
                adj_boot[i, j] = w
                adj_boot[j, i] = w

            try:
                boot_emb = spectral_embed(adj_boot)
                # Handle sign ambiguity in eigenvectors: align signs
                for dim in range(boot_emb.shape[1]):
                    if np.dot(base_emb[:, dim], boot_emb[:, dim]) < 0:
                        boot_emb[:, dim] *= -1
                # Per-gene cosine similarity
                dots = (base_emb * boot_emb).sum(axis=1)
                stability_scores[b] = dots
            except np.linalg.LinAlgError:
                stability_scores[b] = 0.5

        mean_stability = stability_scores.mean(axis=0)  # (G,)
        per_gene = {self.genes[i]: float(mean_stability[i]) for i in range(G)}

        overall = float(mean_stability.mean())

        # Bottom 20 — most fragile
        sorted_genes = sorted(per_gene.items(), key=lambda x: x[1])
        fragile = sorted_genes[:20]

        print(f"  Bootstrap stability: {overall:.3f} overall")
        print(f"  Fragile genes (bottom 5):")
        for gene, stab in fragile[:5]:
            ch = self.channels.get(gene, '?')
            n_edges = sum(1 for (i, j) in self.edge_types_present
                          if self.gene_idx[gene] in (i, j))
            print(f"    {gene} ({ch}): stability={stab:.3f}, edges={n_edges}")
        print(f"  [{time.time()-t0:.1f}s]")

        return {
            'per_gene': per_gene,
            'overall': overall,
            'fragile_genes': fragile,
        }

    # ------------------------------------------------------------------
    # Signal 6: Subgraph coherence (per block)
    # ------------------------------------------------------------------

    def subgraph_coherence(self):
        """Measure Laplacian smoothness within each sub-pathway block.

        Low energy = edges within block tell a consistent story.
        High energy = contradictory or noisy edges.

        Returns:
            per_block: dict block_id → smoothness score (0-1, higher = better)
            overall: float — mean across blocks
            worst_blocks: list of (block_id, genes, smoothness)
        """
        t0 = time.time()

        # Group genes by block
        block_genes = defaultdict(list)
        for gene, bid in self.blocks.items():
            if gene in self.gene_idx:
                block_genes[bid].append(gene)

        per_block = {}
        for bid, genes in block_genes.items():
            if len(genes) < 3:
                continue

            indices = [self.gene_idx[g] for g in genes]
            idx_set = set(indices)
            n = len(indices)
            local_idx = {gi: li for li, gi in enumerate(indices)}

            # Build local adjacency
            adj = np.zeros((n, n), dtype=np.float32)
            for (i, j), etypes in self.edge_types_present.items():
                if i in idx_set and j in idx_set:
                    w = len(etypes)
                    adj[local_idx[i], local_idx[j]] = w
                    adj[local_idx[j], local_idx[i]] = w

            if adj.sum() == 0:
                per_block[bid] = 0.0
                continue

            # Laplacian
            degree = adj.sum(axis=1)
            L = np.diag(degree) - adj

            # Smoothness = x^T L x / x^T D x for first non-trivial eigenvector
            # Lower ratio = smoother signal = more coherent
            try:
                eigenvalues = np.linalg.eigvalsh(L)
                # Fiedler value (second smallest eigenvalue), normalized
                fiedler = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
                max_eval = eigenvalues[-1] if eigenvalues[-1] > 0 else 1.0
                # Invert: low Fiedler = well-connected = high coherence
                coherence = 1.0 - (fiedler / max_eval)
                per_block[bid] = float(coherence)
            except np.linalg.LinAlgError:
                per_block[bid] = 0.0

        overall = float(np.mean(list(per_block.values()))) if per_block else 0.0

        # Worst blocks
        sorted_blocks = sorted(per_block.items(), key=lambda x: x[1])
        worst = []
        for bid, score in sorted_blocks[:5]:
            genes = block_genes[bid]
            ch = self.channels.get(genes[0], '?') if genes else '?'
            worst.append((bid, ch, genes, score))

        print(f"  Subgraph coherence: {overall:.3f} overall "
              f"({len(per_block)} blocks) [{time.time()-t0:.1f}s]")

        return {
            'per_block': per_block,
            'overall': overall,
            'worst_blocks': worst,
        }

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def full_report(self):
        """Run all signals, return composite confidence score."""
        t0 = time.time()
        print("=" * 60)
        print("  GRAPH CONFIDENCE REPORT")
        print("=" * 60)

        witness = self.witness_agreement()
        link_pred = self.link_prediction()
        triangles = self.triangle_closure()
        provenance = self.provenance_diversity()
        stability = self.bootstrap_stability()
        coherence = self.subgraph_coherence()

        # Composite: weighted average of normalized signals
        # Each signal is scaled to [0, 1] where 1 = maximum confidence
        composite = (
            0.20 * witness['overall'] +                # witness agreement [-1,1] → use raw
            0.25 * link_pred['auc'] +                  # AUC [0.5, 1.0]
            0.15 * triangles['coverage'] +             # fraction [0, 1]
            0.15 * min(provenance['mean_types'] / 3.0, 1.0) +  # normalize to 3 types
            0.15 * stability['overall'] +              # cosine sim [0, 1]
            0.10 * coherence['overall']                # [0, 1]
        )

        print(f"\n{'=' * 60}")
        print(f"  COMPOSITE GRAPH CONFIDENCE: {composite:.3f}")
        print(f"{'=' * 60}")
        print(f"  Witness agreement:     {witness['overall']:.3f}  (×0.20)")
        print(f"  Link prediction AUC:   {link_pred['auc']:.3f}  (×0.25)")
        print(f"  Triangle coverage:     {triangles['coverage']:.3f}  (×0.15)")
        print(f"  Provenance diversity:  {provenance['mean_types']:.2f}  (×0.15)")
        print(f"  Bootstrap stability:   {stability['overall']:.3f}  (×0.15)")
        print(f"  Subgraph coherence:    {coherence['overall']:.3f}  (×0.10)")
        print(f"\n  Total time: {time.time()-t0:.1f}s")

        return {
            'composite_score': composite,
            'witness_agreement': witness,
            'link_prediction': link_pred,
            'triangle_closure': triangles,
            'provenance_diversity': provenance,
            'bootstrap_stability': stability,
            'subgraph_coherence': coherence,
            'n_genes': len(self.genes),
            'n_edge_pairs': len(self.edge_types_present),
            'n_edge_types': len(self.typed_edges),
        }

    def save_report(self, path=None):
        """Run full report and save to JSON."""
        if path is None:
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results", "graph_confidence")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, "confidence_report.json")

        report = self.full_report()

        # Strip large per-edge dicts for JSON
        serializable = {}
        for k, v in report.items():
            if isinstance(v, dict):
                clean = {}
                for kk, vv in v.items():
                    if kk in ('per_edge', 'per_gene', 'per_block',
                              'low_confidence_edges'):
                        if isinstance(vv, dict) and len(vv) > 100:
                            clean[kk] = f"<{len(vv)} entries>"
                        elif isinstance(vv, list) and len(vv) > 100:
                            clean[kk] = f"<{len(vv)} entries>"
                        else:
                            clean[kk] = vv
                    elif isinstance(vv, dict):
                        # Convert tuple keys to strings
                        clean[kk] = {str(kkk): vvv for kkk, vvv in vv.items()}
                    else:
                        clean[kk] = vv
                serializable[k] = clean
            else:
                serializable[k] = v

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"\n  Report saved to {path}")
        return report


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    gc = GraphConfidence()
    gc.load_graph()
    gc.save_report()


if __name__ == "__main__":
    main()
