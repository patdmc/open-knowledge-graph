#!/usr/bin/env python3
"""
Math knowledge graph: curried, composable math functions backed by
knowledge nodes.

Usage:
    from math_graph import MathGraph
    mg = MathGraph()

    # Direct call
    mg.sin(np.pi / 4)

    # Curry and compose
    f = mg.compose(mg.sin, mg.arcsin)  # identity
    f(0.5)  # 0.5

    # Pipe: left to right
    result = mg.pipe(0.5, mg.arcsin, mg.sin)  # 0.5

    # Look up knowledge
    mg.knowledge('sin')  # returns the YAML node
    mg.latex('sin')       # '\\sin(\\theta)'

    # Solve
    mg.solve('x**2 - 4', 'x')  # [-2, 2]
"""

import os
import math
import functools
import importlib
from pathlib import Path
from typing import Any, Callable, List

import yaml

try:
    import numpy as np
except ImportError:
    np = None

try:
    import sympy
except ImportError:
    sympy = None

try:
    import scipy
    import scipy.integrate
    import scipy.optimize
    import scipy.linalg
    import scipy.sparse
    import scipy.sparse.csgraph
    import scipy.spatial.distance
except ImportError:
    scipy = None

try:
    import networkx as nx
except ImportError:
    nx = None


MATH_DIR = Path(__file__).parent.parent


class MathGraph:
    """Curried, composable math backed by knowledge graph nodes."""

    def __init__(self):
        self._nodes = {}
        self._concepts = {}
        self._symbols = {}
        self._shared_lemmas = {}
        self._load_all()
        self._register_builtins()

    def _load_all(self):
        """Load all YAML nodes from math subdirectories."""
        for yaml_file in MATH_DIR.rglob("*.yaml"):
            with open(yaml_file) as f:
                node = yaml.safe_load(f)
            if node and 'id' in node:
                self._nodes[node['id']] = node
                # Index shared lemmas separately
                if node.get('type') == 'shared_lemma':
                    self._shared_lemmas[node['id']] = node
                # Index concepts
                for concept in node.get('concepts', []):
                    self._concepts[concept['id']] = concept
                    self._concepts[concept['id']]['_source'] = node['id']
                # Index symbols
                for symbol in node.get('symbols', []):
                    sid = symbol.get('name', symbol.get('symbol'))
                    self._symbols[sid] = symbol
                    self._symbols[sid]['_source'] = node['id']

    def _register_builtins(self):
        """Register numpy/scipy/sympy callables for each concept."""
        self._callables = {}

        # Trig
        if np:
            for name in ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                          'sinh', 'cosh', 'tanh', 'sqrt', 'abs', 'exp', 'log']:
                fn = getattr(np, name, None)
                if fn:
                    self._callables[name] = fn

        # Math module fallbacks
        for name in ['factorial']:
            fn = getattr(math, name, None)
            if fn:
                self._callables[name] = fn

        # Linear algebra
        if np:
            la = np.linalg
            for name in ['norm', 'det', 'inv', 'eig', 'svd', 'solve',
                          'matrix_rank']:
                fn = getattr(la, name, None)
                if fn:
                    self._callables[name] = fn
            self._callables['dot'] = np.dot
            self._callables['cross'] = np.cross
            self._callables['trace'] = np.trace
            self._callables['transpose'] = lambda A: A.T

        # Calculus (sympy)
        if sympy:
            self._callables['diff'] = sympy.diff
            self._callables['integrate'] = sympy.integrate
            self._callables['limit'] = sympy.limit
            self._callables['series'] = sympy.series
            self._callables['solve_eq'] = sympy.solve
            self._callables['simplify'] = sympy.simplify
            self._callables['expand'] = sympy.expand
            self._callables['factor'] = sympy.factor

        # Numerical integration / optimization (scipy)
        if scipy:
            self._callables['quad'] = scipy.integrate.quad
            self._callables['dblquad'] = scipy.integrate.dblquad
            self._callables['solve_ivp'] = scipy.integrate.solve_ivp
            self._callables['minimize'] = scipy.optimize.minimize
            self._callables['linprog'] = scipy.optimize.linprog
            self._callables['differential_evolution'] = scipy.optimize.differential_evolution

        # Advanced linear algebra (scipy)
        if scipy:
            self._callables['lu'] = scipy.linalg.lu
            self._callables['qr'] = lambda A: np.linalg.qr(A) if np else None
            self._callables['cholesky'] = lambda A: np.linalg.cholesky(A) if np else None
            self._callables['schur'] = scipy.linalg.schur
            self._callables['expm'] = scipy.linalg.expm
            self._callables['logm'] = scipy.linalg.logm
            self._callables['null_space'] = scipy.linalg.null_space
            self._callables['orth'] = scipy.linalg.orth
            self._callables['lstsq'] = lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0] if np else None
            self._callables['pinv'] = lambda A: np.linalg.pinv(A) if np else None
            self._callables['cond'] = lambda A: np.linalg.cond(A) if np else None
            self._callables['eigh'] = lambda A: np.linalg.eigh(A) if np else None
            self._callables['kron'] = np.kron if np else None

        # Number theory (sympy)
        if sympy:
            self._callables['isprime'] = sympy.isprime
            self._callables['factorint'] = sympy.factorint
            self._callables['totient'] = sympy.totient
            self._callables['gcd_sym'] = sympy.gcd
            self._callables['lcm_sym'] = sympy.lcm
            self._callables['binomial'] = sympy.binomial
            self._callables['fibonacci'] = sympy.fibonacci
            self._callables['nextprime'] = sympy.nextprime

        # Math module number theory
        self._callables['gcd'] = math.gcd
        self._callables['comb'] = math.comb
        self._callables['perm'] = math.perm

        # Graph theory (networkx)
        if nx:
            self._callables['Graph'] = nx.Graph
            self._callables['DiGraph'] = nx.DiGraph
            self._callables['shortest_path'] = nx.shortest_path
            self._callables['shortest_path_length'] = nx.shortest_path_length
            self._callables['connected_components'] = lambda G: list(nx.connected_components(G))
            self._callables['betweenness_centrality'] = nx.betweenness_centrality
            self._callables['pagerank'] = nx.pagerank
            self._callables['adjacency_matrix'] = lambda G: nx.adjacency_matrix(G).toarray()
            self._callables['laplacian_matrix'] = lambda G: nx.laplacian_matrix(G).toarray()
            self._callables['is_isomorphic'] = nx.is_isomorphic
            self._callables['minimum_spanning_tree'] = nx.minimum_spanning_tree
            self._callables['topological_sort'] = lambda G: list(nx.topological_sort(G))
            self._callables['density'] = nx.density
            self._callables['diameter'] = nx.diameter

        # Probability / statistics (scipy.stats)
        if scipy:
            _stats = __import__('scipy.stats', fromlist=['stats'])
            self._callables['norm_dist'] = _stats.norm
            self._callables['pearsonr'] = _stats.pearsonr
            self._callables['entropy'] = _stats.entropy
            self._callables['ttest_ind'] = _stats.ttest_ind
            self._callables['ks_2samp'] = _stats.ks_2samp

        # Computational geometry (scipy.spatial)
        if scipy:
            self._callables['ConvexHull'] = scipy.spatial.ConvexHull
            self._callables['Voronoi'] = scipy.spatial.Voronoi
            self._callables['Delaunay'] = scipy.spatial.Delaunay
            self._callables['KDTree'] = scipy.spatial.KDTree
            self._callables['cdist'] = scipy.spatial.distance.cdist
            self._callables['procrustes'] = scipy.spatial.procrustes

        # Differential equations (scipy.integrate)
        if scipy:
            self._callables['solve_bvp'] = scipy.integrate.solve_bvp

        # FFT
        if np:
            self._callables['fft'] = np.fft.fft
            self._callables['ifft'] = np.fft.ifft

        # Complex analysis (sympy)
        if sympy:
            self._callables['residue'] = sympy.residue
            self._callables['laplace_transform'] = sympy.laplace_transform

        # Combinatorics (sympy)
        if sympy:
            self._callables['catalan'] = sympy.catalan
            self._callables['bell'] = sympy.bell
            self._callables['subfactorial'] = sympy.subfactorial

        # Combinatorics (math)
        self._callables['factorial'] = math.factorial

    # === Core composition primitives ===

    @staticmethod
    def compose(*fns: Callable) -> Callable:
        """Right-to-left composition: compose(f, g)(x) = f(g(x))"""
        def composed(x):
            result = x
            for fn in reversed(fns):
                result = fn(result)
            return result
        return composed

    @staticmethod
    def pipe(x: Any, *fns: Callable) -> Any:
        """Left-to-right application: pipe(x, f, g) = g(f(x))"""
        result = x
        for fn in fns:
            result = fn(result)
        return result

    @staticmethod
    def curry(fn: Callable, *args) -> Callable:
        """Partial application."""
        return functools.partial(fn, *args)

    @staticmethod
    def map_over(fn: Callable) -> Callable:
        """Lift a scalar function to work over sequences."""
        if np:
            return np.vectorize(fn)
        return lambda xs: [fn(x) for x in xs]

    @staticmethod
    def reduce_with(op: Callable, init: Any = None) -> Callable:
        """Create a reducer from a binary operation."""
        if init is not None:
            return lambda xs: functools.reduce(op, xs, init)
        return lambda xs: functools.reduce(op, xs)

    # === Lookup ===

    def __getattr__(self, name: str):
        """Dynamic attribute access: mg.sin(x), mg.dot(a, b), etc."""
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._callables:
            return self._callables[name]
        raise AttributeError(f"Unknown math function: {name}")

    def knowledge(self, name: str) -> dict:
        """Look up the knowledge node for a concept."""
        if name in self._concepts:
            return self._concepts[name]
        if name in self._symbols:
            return self._symbols[name]
        # Search by name field
        for cid, c in self._concepts.items():
            if c.get('name', '').lower() == name.lower():
                return c
        return None

    def latex(self, name: str) -> str:
        """Get LaTeX notation for a concept."""
        node = self.knowledge(name)
        if node:
            return node.get('latex', f'\\text{{{name}}}')
        return f'\\text{{{name}}}'

    def action(self, name: str, lib: str = 'numpy') -> str:
        """Get the Python call string for a concept in a given library."""
        node = self.knowledge(name)
        if node:
            act = node.get('action', {})
            return act.get(lib, act.get('python', None))
        return None

    def curried(self, name: str) -> str:
        """Get the curried lambda string for a concept."""
        node = self.knowledge(name)
        if node:
            return node.get('curried', None)
        return None

    def inverse(self, name: str) -> str:
        """Get the inverse function name."""
        node = self.knowledge(name)
        if node:
            return node.get('inverse', node.get('inverse_of', None))
        return None

    def edges(self, node_id: str) -> list:
        """Get edges from a node."""
        node = self._nodes.get(node_id, {})
        return node.get('edges', [])

    def domains(self) -> list:
        """List all loaded domain nodes."""
        return list(self._nodes.keys())

    def concepts_in(self, domain: str) -> list:
        """List concepts belonging to a domain."""
        return [c for c in self._concepts.values()
                if c.get('_source', '').startswith(domain[:3].upper())
                or c.get('domain', '') == domain]

    # === Proof trees ===

    def proofs(self) -> list:
        """List all loaded proof nodes."""
        return [nid for nid, n in self._nodes.items()
                if n.get('type') == 'proof']

    def proof(self, proof_id: str) -> dict:
        """Get a proof node by id."""
        node = self._nodes.get(proof_id)
        if node and node.get('type') == 'proof':
            return node
        # Search by partial match
        for nid, n in self._nodes.items():
            if n.get('type') == 'proof' and proof_id in nid:
                return n
        return None

    def proof_tree(self, proof_id: str):
        """Build a networkx DiGraph of the proof's assertion dependency chain.

        Nodes: assertion ids (with attributes: type, statement, justification)
        Edges: depends_on links (internal) + references links (to knowledge graph)
        """
        if not nx:
            raise ImportError("networkx required for proof trees")
        p = self.proof(proof_id)
        if not p:
            raise KeyError(f"Proof not found: {proof_id}")

        G = nx.DiGraph()
        G.graph['name'] = p.get('name', proof_id)
        G.graph['target'] = p.get('target', {}).get('statement', '')
        G.graph['method'] = p.get('method', '')

        for a in p.get('assertions', []):
            aid = a['id']
            G.add_node(aid,
                       step=a.get('step', 0),
                       type=a.get('type', 'claim'),
                       statement=a.get('statement', ''),
                       justification=a.get('justification', ''))
            # Internal dependency edges
            for dep in a.get('depends_on', []):
                G.add_edge(dep, aid, relation='DEPENDS_ON')
            # Cross-references to knowledge graph
            for ref in a.get('references', []):
                ref_label = f'KG:{ref}'
                if ref_label not in G:
                    G.add_node(ref_label, type='knowledge_ref',
                               statement=self.latex(ref),
                               concept=ref)
                G.add_edge(ref_label, aid, relation='REFERENCES')

        # Add corollaries if present
        for cor in p.get('corollaries', []):
            cid = cor['id']
            G.add_node(cid,
                       type='corollary',
                       statement=cor.get('statement', ''),
                       justification=cor.get('justification', ''))
            for dep in cor.get('depends_on', []):
                G.add_edge(dep, cid, relation='DEPENDS_ON')

        return G

    def proof_chain(self, proof_id: str, target: str = None) -> list:
        """Return the ordered assertion chain from axioms/hypotheses to target.

        If target is None, uses the last assertion (conclusion).
        Returns list of assertion dicts in dependency order.
        """
        G = self.proof_tree(proof_id)
        if target is None:
            # Find the node with no outgoing DEPENDS_ON edges to other proof steps
            proof_nodes = [n for n, d in G.nodes(data=True)
                           if d.get('type') not in ('knowledge_ref',)]
            target = max(proof_nodes,
                         key=lambda n: G.nodes[n].get('step', 0))

        # All ancestors in topological order
        ancestors = nx.ancestors(G, target) | {target}
        subG = G.subgraph(ancestors)
        chain = []
        for nid in nx.topological_sort(subG):
            data = G.nodes[nid]
            if data.get('type') == 'knowledge_ref':
                continue
            chain.append({'id': nid, **data})
        return chain

    def proof_depth(self, proof_id: str) -> int:
        """Longest path in the proof DAG (logical depth)."""
        G = self.proof_tree(proof_id)
        # Filter to proof assertions only
        proof_nodes = [n for n, d in G.nodes(data=True)
                       if d.get('type') != 'knowledge_ref']
        subG = G.subgraph(proof_nodes)
        return nx.dag_longest_path_length(subG)

    def proof_foundations(self, proof_id: str) -> dict:
        """Return the axioms, hypotheses, definitions, and knowledge refs
        that the proof rests on (root nodes of the DAG)."""
        G = self.proof_tree(proof_id)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        foundations = {'hypotheses': [], 'definitions': [],
                       'axioms': [], 'knowledge_refs': [], 'other': []}
        for r in roots:
            data = G.nodes[r]
            t = data.get('type', '')
            if t == 'knowledge_ref':
                foundations['knowledge_refs'].append({
                    'id': r, 'concept': data.get('concept', ''),
                    'latex': data.get('statement', '')})
            elif t == 'hypothesis':
                foundations['hypotheses'].append({'id': r, **data})
            elif t == 'definition':
                foundations['definitions'].append({'id': r, **data})
            elif t == 'axiom':
                foundations['axioms'].append({'id': r, **data})
            else:
                foundations['other'].append({'id': r, **data})
        return foundations

    def cross_proof_graph(self) -> 'nx.DiGraph':
        """Build a graph connecting proofs to the knowledge concepts they use.

        Nodes: proof ids + concept ids
        Edges: PROVES (proof -> theorem), USES (proof -> concept)
        """
        if not nx:
            raise ImportError("networkx required")
        G = nx.DiGraph()
        for pid in self.proofs():
            p = self._nodes[pid]
            G.add_node(pid, type='proof', name=p.get('name', ''))

            # What does the proof prove?
            target = p.get('target', {})
            if target:
                G.add_edge(pid, pid + ':target',
                           relation='PROVES')
                G.add_node(pid + ':target', type='theorem',
                           statement=target.get('statement', ''))

            # What knowledge does it reference?
            all_refs = set()
            for a in p.get('assertions', []):
                for ref in a.get('references', []):
                    all_refs.add(ref)
            for ref in all_refs:
                if ref not in G:
                    G.add_node(ref, type='concept',
                               latex=self.latex(ref))
                G.add_edge(pid, ref, relation='USES')

            # Edge to other proofs/nodes
            for edge in p.get('edges', []):
                G.add_edge(pid, edge['to'],
                           relation=edge.get('relation', 'RELATED'))
        return G

    def print_proof(self, proof_id: str):
        """Pretty-print a proof's logical chain."""
        p = self.proof(proof_id)
        if not p:
            print(f"Proof not found: {proof_id}")
            return
        print(f"{'='*60}")
        print(f"PROOF: {p.get('name', proof_id)}")
        print(f"Target: {p.get('target', {}).get('statement', '?')}")
        print(f"Method: {p.get('method', '?')}")
        print(f"{'='*60}")
        chain = self.proof_chain(proof_id)
        for a in chain:
            tag = a.get('type', 'claim').upper()
            step = a.get('step', '?')
            print(f"\n  [{step}] ({tag}) {a['id']}")
            print(f"      {a.get('statement', '')}")
            print(f"      ∵ {a.get('justification', '')}")
        foundations = self.proof_foundations(proof_id)
        kg_refs = foundations.get('knowledge_refs', [])
        if kg_refs:
            print(f"\n  Knowledge graph refs: {[r['concept'] for r in kg_refs]}")
        print(f"\n  Logical depth: {self.proof_depth(proof_id)}")
        print(f"  QED: {p.get('qed', False)}")

    # === Deep proof traversal ===

    def concept_source(self, concept_id: str) -> str:
        """Which domain node does a concept belong to?"""
        c = self._concepts.get(concept_id)
        if c:
            return c.get('_source', None)
        return None

    def dependency_chain(self, node_id: str, depth: int = 10) -> list:
        """Follow REQUIRES edges from a domain node back to its foundations.

        Returns list of (node_id, relation, target_id) tuples.
        """
        chain = []
        visited = set()
        frontier = [node_id]
        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for edge in self.edges(nid):
                    target = edge.get('to', '')
                    rel = edge.get('relation', 'RELATED')
                    chain.append((nid, rel, target))
                    if target not in visited:
                        next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break
        return chain

    def deep_proof_tree(self, proof_id: str, kg_depth: int = 3):
        """Build proof DAG + extend knowledge_ref nodes into the knowledge graph.

        Each knowledge ref becomes a subgraph showing its domain, that domain's
        REQUIRES edges, and so on up to kg_depth hops.
        """
        if not nx:
            raise ImportError("networkx required")

        # Start with the proof tree
        G = self.proof_tree(proof_id)

        # Find all knowledge refs
        kg_nodes = [n for n, d in G.nodes(data=True)
                    if d.get('type') == 'knowledge_ref']

        visited_domains = set()
        for kg_node in kg_nodes:
            concept_id = G.nodes[kg_node].get('concept', '')
            source = self.concept_source(concept_id)
            if not source or source in visited_domains:
                # Still link concept to its domain
                if source and source not in G:
                    G.add_node(source, type='domain',
                               name=self._nodes.get(source, {}).get('name', source))
                if source:
                    G.add_edge(kg_node, source, relation='BELONGS_TO')
                continue

            # Add domain node
            visited_domains.add(source)
            domain_data = self._nodes.get(source, {})
            G.add_node(source, type='domain',
                       name=domain_data.get('name', source))
            G.add_edge(kg_node, source, relation='BELONGS_TO')

            # Follow REQUIRES/EXTENDS edges from this domain (forward only)
            FORWARD_RELS = {'REQUIRES', 'EXTENDS', 'USED_BY', 'REQUIRED_BY'}
            chain = self.dependency_chain(source, depth=kg_depth)
            for src, rel, tgt in chain:
                if rel not in FORWARD_RELS:
                    continue
                if tgt == source:
                    continue  # skip self-loops
                if src not in G:
                    src_data = self._nodes.get(src, {})
                    G.add_node(src, type='domain',
                               name=src_data.get('name', src))
                if tgt not in G:
                    tgt_data = self._nodes.get(tgt, {})
                    G.add_node(tgt, type='domain',
                               name=tgt_data.get('name', tgt))
                # Only add if it won't create a cycle
                G.add_edge(src, tgt, relation=rel)
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(src, tgt)

        return G

    def proof_requires(self, proof_id: str) -> dict:
        """What knowledge domains does this proof ultimately depend on?

        Returns {domain_id: [concept_ids used from that domain]}.
        """
        p = self.proof(proof_id)
        if not p:
            return {}

        # Collect all concept references
        all_refs = set()
        for a in p.get('assertions', []):
            all_refs.update(a.get('references', []))

        # Map concepts to domains, then follow domain edges
        domain_concepts = {}
        all_domains = set()
        for ref in all_refs:
            source = self.concept_source(ref)
            if source:
                domain_concepts.setdefault(source, []).append(ref)
                all_domains.add(source)

        # Transitively follow REQUIRES
        visited = set()
        frontier = list(all_domains)
        while frontier:
            nid = frontier.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for edge in self.edges(nid):
                if edge.get('relation') in ('REQUIRES', 'EXTENDS'):
                    tgt = edge['to']
                    if tgt not in visited:
                        frontier.append(tgt)
                        domain_concepts.setdefault(tgt, [])

        return domain_concepts

    def shared_foundations(self, proof_id_a: str, proof_id_b: str) -> dict:
        """Find knowledge domains shared between two proofs."""
        da = set(self.proof_requires(proof_id_a).keys())
        db = set(self.proof_requires(proof_id_b).keys())
        return {
            'shared': da & db,
            'only_a': da - db,
            'only_b': db - da,
        }

    def proof_for_concept(self, concept_id: str) -> list:
        """Find proofs that prove something about a concept or reference it."""
        results = []
        for pid in self.proofs():
            p = self._nodes[pid]
            # Check if concept appears in references
            for a in p.get('assertions', []):
                if concept_id in a.get('references', []):
                    results.append(pid)
                    break
        return results

    def chain_proofs(self, proof_ids: list):
        """Build a combined graph linking multiple proofs through shared concepts.

        If proof A uses concept X, and proof B proves concept X,
        the combined graph shows A depending on B.
        """
        if not nx:
            raise ImportError("networkx required")

        G = nx.DiGraph()
        # Add each proof as a cluster
        for pid in proof_ids:
            p = self.proof(pid)
            if not p:
                continue
            G.add_node(pid, type='proof',
                       name=p.get('name', pid),
                       method=p.get('method', ''))

            # What it proves
            target = p.get('target', {})
            target_id = f"{pid}:target"
            G.add_node(target_id, type='theorem',
                       statement=target.get('statement', ''))
            G.add_edge(pid, target_id, relation='PROVES')

            # What it uses
            all_refs = set()
            for a in p.get('assertions', []):
                all_refs.update(a.get('references', []))
            for ref in all_refs:
                if ref not in G:
                    G.add_node(ref, type='concept',
                               latex=self.latex(ref),
                               domain=self.concept_source(ref))
                G.add_edge(pid, ref, relation='USES')

            # Domain edges
            for domain_id in self.proof_requires(pid):
                if domain_id not in G:
                    dn = self._nodes.get(domain_id, {})
                    G.add_node(domain_id, type='domain',
                               name=dn.get('name', domain_id))
                G.add_edge(pid, domain_id, relation='REQUIRES_DOMAIN')

        # Now link proofs: if A uses concept C, and B's target is related to C
        for pid_a in proof_ids:
            pa = self.proof(pid_a)
            if not pa:
                continue
            refs_a = set()
            for a in pa.get('assertions', []):
                refs_a.update(a.get('references', []))

            for pid_b in proof_ids:
                if pid_a == pid_b:
                    continue
                pb = self.proof(pid_b)
                if not pb:
                    continue
                # Check if B proves something A references
                for edge in pb.get('edges', []):
                    target_domain = edge.get('to', '')
                    source_a = self.concept_source
                    # If A uses a concept from a domain that B proves from
                    for ref in refs_a:
                        src = self.concept_source(ref)
                        if src and src == target_domain:
                            G.add_edge(pid_a, pid_b,
                                       relation='DEPENDS_ON_PROOF',
                                       via=ref)

        return G

    # === Shared lemmas ===

    def shared_lemmas(self) -> list:
        """List all shared lemma IDs."""
        return list(self._shared_lemmas.keys())

    def shared_lemma(self, sl_id: str) -> dict:
        """Get a shared lemma by ID."""
        return self._shared_lemmas.get(sl_id)

    def lemma_users(self, sl_id: str) -> list:
        """Which proofs use a shared lemma?"""
        sl = self._shared_lemmas.get(sl_id, {})
        return [u['proof'] for u in sl.get('used_by', [])]

    def proof_shared_lemmas(self, proof_id: str) -> list:
        """Which shared lemmas does a proof use?"""
        results = []
        for sl_id, sl in self._shared_lemmas.items():
            for u in sl.get('used_by', []):
                if u.get('proof') == proof_id:
                    results.append(sl_id)
                    break
        return results

    # === Synthesis engine ===

    def _proof_concept_set(self, proof_id: str) -> set:
        """All concept IDs referenced by a proof."""
        p = self.proof(proof_id)
        if not p:
            return set()
        refs = set()
        for a in p.get('assertions', []):
            refs.update(a.get('references', []))
        return refs

    def _proof_domain_set(self, proof_id: str) -> set:
        """All domain IDs transitively required by a proof."""
        return set(self.proof_requires(proof_id).keys())

    def bridge_concepts(self) -> list:
        """Find concepts that connect otherwise-disjoint proof clusters.

        A bridge concept is referenced by proofs in different domains
        that share no other conceptual link. These are candidates for
        cross-domain synthesis.

        Returns list of (concept_id, proof_cluster_A, proof_cluster_B, gap_score)
        sorted by gap_score descending (higher = more disjoint = more interesting).
        """
        from collections import defaultdict
        # Map each concept to the proofs that reference it
        concept_proofs = defaultdict(set)
        for pid in self.proofs():
            for ref in self._proof_concept_set(pid):
                concept_proofs[ref].add(pid)

        # Map each proof to its domain set
        proof_domains = {pid: self._proof_domain_set(pid) for pid in self.proofs()}

        bridges = []
        for concept, pids in concept_proofs.items():
            if len(pids) < 2:
                continue
            pids = list(pids)
            for i, pa in enumerate(pids):
                for pb in pids[i+1:]:
                    # How disjoint are their domain dependencies?
                    da = proof_domains.get(pa, set())
                    db = proof_domains.get(pb, set())
                    if not da or not db:
                        continue
                    shared = da & db
                    total = da | db
                    # Gap score: fraction of domains NOT shared
                    # High gap = proofs are from different worlds but share this concept
                    gap = 1.0 - len(shared) / len(total) if total else 0
                    if gap > 0.3:  # meaningful gap
                        bridges.append((concept, pa, pb, gap))

        bridges.sort(key=lambda x: -x[3])
        return bridges

    def synthesis_candidates(self) -> list:
        """Find pairs of proven results that could combine into new theorems.

        Looks for:
        1. Proofs in different domains sharing a concept (bridge)
        2. Shared lemmas that connect proof clusters
        3. Domain edges that are traversed by one proof but not another
           despite both touching neighboring domains

        Returns list of synthesis opportunities as dicts.
        """
        from collections import defaultdict

        candidates = []

        # --- Strategy 1: Bridge concepts ---
        for concept, pa, pb, gap in self.bridge_concepts()[:20]:
            na = self.proof(pa).get('name', pa)
            nb = self.proof(pb).get('name', pb)
            da = self._proof_domain_set(pa)
            db = self._proof_domain_set(pb)
            only_a = da - db
            only_b = db - da

            candidates.append({
                'type': 'bridge_concept',
                'concept': concept,
                'concept_latex': self.latex(concept),
                'proof_a': pa,
                'proof_a_name': na,
                'proof_b': pb,
                'proof_b_name': nb,
                'gap_score': gap,
                'domains_only_a': [self._nodes.get(d, {}).get('name', d) for d in only_a],
                'domains_only_b': [self._nodes.get(d, {}).get('name', d) for d in only_b],
                'hypothesis': (
                    f"The concept '{concept}' bridges {na} (from {', '.join(list(only_a)[:2])}) "
                    f"and {nb} (from {', '.join(list(only_b)[:2])}). "
                    f"A theorem combining both proof techniques may exist."
                ),
            })

        # --- Strategy 2: Shared lemma bridges ---
        for sl_id, sl in self._shared_lemmas.items():
            users = [u['proof'] for u in sl.get('used_by', [])]
            if len(users) < 2:
                continue
            # Check if users are in different domain clusters
            for i, pa in enumerate(users):
                for pb in users[i+1:]:
                    da = self._proof_domain_set(pa)
                    db = self._proof_domain_set(pb)
                    if not da or not db:
                        continue
                    only_a = da - db
                    only_b = db - da
                    if len(only_a) >= 2 or len(only_b) >= 2:
                        na = self.proof(pa)
                        nb = self.proof(pb)
                        if not na or not nb:
                            continue
                        candidates.append({
                            'type': 'shared_lemma_bridge',
                            'lemma': sl_id,
                            'lemma_name': sl.get('name', ''),
                            'proof_a': pa,
                            'proof_a_name': na.get('name', pa),
                            'proof_b': pb,
                            'proof_b_name': nb.get('name', pb),
                            'domains_only_a': [self._nodes.get(d, {}).get('name', d) for d in only_a],
                            'domains_only_b': [self._nodes.get(d, {}).get('name', d) for d in only_b],
                            'hypothesis': (
                                f"Shared lemma '{sl.get('name', sl_id)}' connects "
                                f"{na.get('name', pa)} and {nb.get('name', pb)}. "
                                f"Their disjoint domains ({', '.join(list(only_a)[:2])} vs "
                                f"{', '.join(list(only_b)[:2])}) may yield a synthesis."
                            ),
                        })

        # --- Strategy 3: Untraversed domain edges ---
        # Find domain pairs (A, B) where:
        #   - A REQUIRES B (edge exists)
        #   - Some proof uses A but not B
        #   - Another proof uses B but not A
        # This means the domain edge is a potential bridge nobody has walked yet
        all_domain_edges = []
        for nid in self._nodes:
            for edge in self.edges(nid):
                if edge.get('relation') in ('REQUIRES', 'EXTENDS'):
                    all_domain_edges.append((nid, edge['to']))

        proof_domain_sets = {pid: self._proof_domain_set(pid) for pid in self.proofs()}
        for src, tgt in all_domain_edges:
            proofs_with_src = [p for p, ds in proof_domain_sets.items() if src in ds and tgt not in ds]
            proofs_with_tgt = [p for p, ds in proof_domain_sets.items() if tgt in ds and src not in ds]
            if proofs_with_src and proofs_with_tgt:
                pa = proofs_with_src[0]
                pb = proofs_with_tgt[0]
                na = self.proof(pa)
                nb = self.proof(pb)
                if na and nb:
                    candidates.append({
                        'type': 'unwalked_edge',
                        'domain_from': src,
                        'domain_from_name': self._nodes.get(src, {}).get('name', src),
                        'domain_to': tgt,
                        'domain_to_name': self._nodes.get(tgt, {}).get('name', tgt),
                        'proof_a': pa,
                        'proof_a_name': na.get('name', pa),
                        'proof_b': pb,
                        'proof_b_name': nb.get('name', pb),
                        'hypothesis': (
                            f"The edge {src} -> {tgt} connects domain of "
                            f"{na.get('name', pa)} to domain of {nb.get('name', pb)}, "
                            f"but no proof traverses both sides."
                        ),
                    })

        return candidates

    def print_synthesis_report(self):
        """Pretty-print synthesis candidates."""
        candidates = self.synthesis_candidates()
        by_type = {}
        for c in candidates:
            by_type.setdefault(c['type'], []).append(c)

        print(f"{'='*70}")
        print(f"SYNTHESIS CANDIDATES — {len(candidates)} opportunities found")
        print(f"{'='*70}")

        if 'bridge_concept' in by_type:
            print(f"\n--- Bridge Concepts (shared concept, disjoint domains) ---")
            for c in by_type['bridge_concept'][:10]:
                print(f"\n  [{c['concept']}] gap={c['gap_score']:.2f}")
                print(f"    {c['proof_a_name'][:40]}")
                print(f"      domains: {c['domains_only_a'][:3]}")
                print(f"    {c['proof_b_name'][:40]}")
                print(f"      domains: {c['domains_only_b'][:3]}")

        if 'shared_lemma_bridge' in by_type:
            print(f"\n--- Shared Lemma Bridges ---")
            for c in by_type['shared_lemma_bridge'][:10]:
                print(f"\n  [{c['lemma_name']}]")
                print(f"    {c['proof_a_name'][:40]} <-> {c['proof_b_name'][:40]}")
                print(f"    disjoint: {c['domains_only_a'][:3]} vs {c['domains_only_b'][:3]}")

        if 'unwalked_edge' in by_type:
            print(f"\n--- Unwalked Domain Edges ---")
            seen = set()
            for c in by_type['unwalked_edge'][:10]:
                key = (c['domain_from'], c['domain_to'])
                if key in seen:
                    continue
                seen.add(key)
                print(f"\n  {c['domain_from_name']} -> {c['domain_to_name']}")
                print(f"    {c['proof_a_name'][:40]} (has {c['domain_from_name'][:20]})")
                print(f"    {c['proof_b_name'][:40]} (has {c['domain_to_name'][:20]})")

    # === Convenience builders ===

    def derivative(self, expr_str: str, var: str = 'x'):
        """Symbolic derivative."""
        if not sympy:
            raise ImportError("sympy required for symbolic calculus")
        x = sympy.Symbol(var)
        expr = sympy.sympify(expr_str)
        return sympy.diff(expr, x)

    def integrate_expr(self, expr_str: str, var: str = 'x',
                       bounds: tuple = None):
        """Symbolic or definite integral."""
        if not sympy:
            raise ImportError("sympy required")
        x = sympy.Symbol(var)
        expr = sympy.sympify(expr_str)
        if bounds:
            return sympy.integrate(expr, (x, bounds[0], bounds[1]))
        return sympy.integrate(expr, x)

    def solve(self, expr_str: str, var: str = 'x'):
        """Solve equation (expr = 0)."""
        if not sympy:
            raise ImportError("sympy required")
        x = sympy.Symbol(var)
        return sympy.solve(sympy.sympify(expr_str), x)

    def taylor(self, expr_str: str, var: str = 'x',
               point: float = 0, order: int = 6):
        """Taylor series expansion."""
        if not sympy:
            raise ImportError("sympy required")
        x = sympy.Symbol(var)
        return sympy.series(sympy.sympify(expr_str), x, point, n=order)


if __name__ == "__main__":
    mg = MathGraph()
    print(f"Loaded {len(mg._nodes)} nodes, "
          f"{len(mg._concepts)} concepts, "
          f"{len(mg._symbols)} symbols")
    print(f"Domains: {mg.domains()}")

    # Demo: compose and pipe
    print(f"\nsin(pi/4) = {mg.sin(np.pi/4):.6f}")
    print(f"compose(sin, arcsin)(0.5) = {mg.compose(mg.sin, mg.arcsin)(0.5):.6f}")
    print(f"pipe(0.5, arcsin, sin) = {mg.pipe(0.5, mg.arcsin, mg.sin):.6f}")

    # Curry
    norm_l1 = mg.curry(np.linalg.norm, None)  # partial
    print(f"\nnorm([3,4]) = {mg.norm(np.array([3, 4])):.1f}")

    # Knowledge lookup
    print(f"\nlatex('sin') = {mg.latex('sin')}")
    print(f"action('sin', 'numpy') = {mg.action('sin', 'numpy')}")
    print(f"inverse('sin') = {mg.inverse('sin')}")

    # Symbolic
    if sympy:
        print(f"\nd/dx(x^3 + 2x) = {mg.derivative('x**3 + 2*x')}")
        print(f"integral(x^2) = {mg.integrate_expr('x**2')}")
        print(f"solve(x^2 - 4) = {mg.solve('x**2 - 4')}")
        print(f"taylor(sin(x)) = {mg.taylor('sin(x)')}")

    # Proofs
    print(f"\n{'='*60}")
    print(f"Proofs loaded: {mg.proofs()}")
    if nx and mg.proofs():
        mg.print_proof('PR01-sqrt2-irrational')

        # Cross-proof knowledge graph
        cpg = mg.cross_proof_graph()
        print(f"\nCross-proof graph: {cpg.number_of_nodes()} nodes, "
              f"{cpg.number_of_edges()} edges")
        for pid in mg.proofs():
            p = mg.proof(pid)
            f = mg.proof_foundations(pid)
            kg = [r['concept'] for r in f.get('knowledge_refs', [])]
            print(f"  {p.get('name', pid)}: depth={mg.proof_depth(pid)}, "
                  f"foundations={len(f['hypotheses'])}H+{len(f['definitions'])}D, "
                  f"kg_refs={kg}")
