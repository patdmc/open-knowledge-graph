#!/usr/bin/env python3
"""Visualize proof trees and the cross-proof knowledge graph.

Usage:
    python proof_viz.py                    # all proof trees + cross-proof graph
    python proof_viz.py PR01-sqrt2         # single proof tree
    python proof_viz.py --cross            # cross-proof graph only
"""

import sys
import os
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
from math_graph import MathGraph

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DOMAIN_COLORS = {
    'domain': '#7B1FA2',   # deep purple
}

# Colors by assertion type
TYPE_COLORS = {
    'axiom':          '#2196F3',  # blue
    'definition':     '#9C27B0',  # purple
    'hypothesis':     '#FF9800',  # orange
    'lemma':          '#4CAF50',  # green
    'claim':          '#78909C',  # gray
    'observation':    '#B0BEC5',  # light gray
    'theorem':        '#F44336',  # red
    'corollary':      '#E91E63',  # pink
    'contradiction':  '#FF5722',  # deep orange
    'knowledge_ref':  '#00BCD4',  # cyan
}

NODE_SHAPES = {
    'axiom': 's',       # square
    'definition': 'D',  # diamond
    'hypothesis': '^',  # triangle
    'theorem': '*',     # star
    'corollary': 'p',   # pentagon
    'contradiction': 'X',
    'knowledge_ref': 'h',  # hexagon
}


def wrap_label(text, width=25):
    """Wrap long labels."""
    if len(text) <= width:
        return text
    return '\n'.join(textwrap.wrap(text, width))


def draw_proof_tree(mg, proof_id, outdir='figures'):
    """Render a single proof's DAG."""
    os.makedirs(outdir, exist_ok=True)
    G = mg.proof_tree(proof_id)
    p = mg.proof(proof_id)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # Layout: topological layers
    proof_nodes = [n for n, d in G.nodes(data=True)
                   if d.get('type') != 'knowledge_ref']
    kg_nodes = [n for n, d in G.nodes(data=True)
                if d.get('type') == 'knowledge_ref']

    # Use graphviz layout if available, else spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception:
        # Layered layout from topological generations
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]['layer'] = layer
        pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
        # Flip so roots are at top
        for k in pos:
            pos[k] = (pos[k][0], -pos[k][1])

    # Draw edges
    dep_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('relation') == 'DEPENDS_ON']
    ref_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('relation') == 'REFERENCES']

    nx.draw_networkx_edges(G, pos, edgelist=dep_edges,
                           edge_color='#333333', arrows=True,
                           arrowstyle='->', arrowsize=15,
                           width=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=ref_edges,
                           edge_color='#00BCD4', arrows=True,
                           arrowstyle='->', arrowsize=10,
                           width=1.0, style='dashed', ax=ax)

    # Draw nodes by type
    for ntype, color in TYPE_COLORS.items():
        nodes_of_type = [n for n in G.nodes()
                         if G.nodes[n].get('type') == ntype]
        if not nodes_of_type:
            continue
        shape = NODE_SHAPES.get(ntype, 'o')
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                               node_color=color, node_size=600,
                               node_shape=shape, alpha=0.9, ax=ax)

    # Labels
    labels = {}
    for n, d in G.nodes(data=True):
        if d.get('type') == 'knowledge_ref':
            labels[n] = d.get('concept', n).replace('KG:', '')
        else:
            step = d.get('step', '')
            tag = d.get('type', '')[0].upper() if d.get('type') else ''
            labels[n] = f"{step}{tag}\n{n}"
    nx.draw_networkx_labels(G, pos, labels, font_size=7,
                            font_family='monospace', ax=ax)

    # Legend
    legend_patches = []
    for ntype in ['hypothesis', 'definition', 'axiom', 'claim', 'lemma',
                  'theorem', 'corollary', 'contradiction', 'knowledge_ref']:
        if any(G.nodes[n].get('type') == ntype for n in G.nodes()):
            legend_patches.append(
                mpatches.Patch(color=TYPE_COLORS[ntype], label=ntype))
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

    name = p.get('name', proof_id)
    ax.set_title(f"Proof Tree: {name}\nMethod: {p.get('method', '?')} | "
                 f"Depth: {mg.proof_depth(proof_id)}", fontsize=12)
    ax.axis('off')

    outpath = os.path.join(outdir, f"proof_{proof_id.replace('-','_')}.pdf")
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  → {outpath}")
    return outpath


def draw_cross_proof_graph(mg, outdir='figures'):
    """Render the graph connecting all proofs to knowledge concepts."""
    os.makedirs(outdir, exist_ok=True)
    G = mg.cross_proof_graph()

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor('white')

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    except Exception:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    # Classify nodes
    proof_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'proof']
    theorem_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'theorem']
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'concept']
    other_nodes = [n for n in G.nodes()
                   if n not in proof_nodes + theorem_nodes + concept_nodes]

    # Draw edges by type
    proves_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get('relation') == 'PROVES']
    uses_edges = [(u, v) for u, v, d in G.edges(data=True)
                  if d.get('relation') == 'USES']
    other_edges = [(u, v) for u, v, d in G.edges(data=True)
                   if d.get('relation') not in ('PROVES', 'USES')]

    nx.draw_networkx_edges(G, pos, edgelist=proves_edges,
                           edge_color='#F44336', width=2.0,
                           arrows=True, arrowstyle='->', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=uses_edges,
                           edge_color='#00BCD4', width=1.0,
                           style='dashed', arrows=True, arrowstyle='->', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=other_edges,
                           edge_color='#999999', width=0.8,
                           arrows=True, arrowstyle='->', ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=proof_nodes,
                           node_color='#FF9800', node_size=800,
                           node_shape='s', alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=theorem_nodes,
                           node_color='#F44336', node_size=600,
                           node_shape='*', alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes,
                           node_color='#00BCD4', node_size=400,
                           node_shape='h', alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                           node_color='#999999', node_size=300,
                           alpha=0.6, ax=ax)

    # Labels
    labels = {}
    for n, d in G.nodes(data=True):
        if d.get('type') == 'proof':
            labels[n] = d.get('name', n).split('(')[0].strip()[:30]
        elif d.get('type') == 'concept':
            labels[n] = n
        elif d.get('type') == 'theorem':
            labels[n] = wrap_label(d.get('statement', n)[:50])
        else:
            labels[n] = n[:20]
    nx.draw_networkx_labels(G, pos, labels, font_size=6,
                            font_family='monospace', ax=ax)

    legend_patches = [
        mpatches.Patch(color='#FF9800', label='Proof'),
        mpatches.Patch(color='#F44336', label='Theorem (target)'),
        mpatches.Patch(color='#00BCD4', label='Knowledge concept'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
    ax.set_title(f"Cross-Proof Knowledge Graph\n"
                 f"{len(proof_nodes)} proofs, {len(concept_nodes)} concepts, "
                 f"{G.number_of_edges()} edges", fontsize=13)
    ax.axis('off')

    outpath = os.path.join(outdir, 'cross_proof_graph.pdf')
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  → {outpath}")
    return outpath


def draw_deep_proof_tree(mg, proof_id, outdir='figures', kg_depth=2):
    """Render proof DAG extended into the knowledge graph.

    Proof steps → knowledge concepts → domain nodes → REQUIRES chain.
    Shows the full foundation a proof rests on.
    """
    os.makedirs(outdir, exist_ok=True)
    G = mg.deep_proof_tree(proof_id, kg_depth=kg_depth)
    p = mg.proof(proof_id)

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    fig.patch.set_facecolor('white')

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception:
        if nx.is_directed_acyclic_graph(G):
            for layer, nodes in enumerate(nx.topological_generations(G)):
                for node in nodes:
                    G.nodes[node]['layer'] = layer
            pos = nx.multipartite_layout(G, subset_key='layer',
                                         align='horizontal')
            for k in pos:
                pos[k] = (pos[k][0], -pos[k][1])
        else:
            pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    # Classify edges
    dep_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('relation') == 'DEPENDS_ON']
    ref_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('relation') == 'REFERENCES']
    belongs_edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get('relation') == 'BELONGS_TO']
    kg_edges = [(u, v) for u, v, d in G.edges(data=True)
                if d.get('relation') in ('REQUIRES', 'EXTENDS',
                                          'INVERSE_OF', 'USED_BY',
                                          'REQUIRED_BY', 'USES',
                                          'INSTANCE_OF', 'EXTENDS_TO')]

    nx.draw_networkx_edges(G, pos, edgelist=dep_edges,
                           edge_color='#333333', arrows=True,
                           arrowstyle='->', arrowsize=15,
                           width=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=ref_edges,
                           edge_color='#00BCD4', arrows=True,
                           arrowstyle='->', arrowsize=10,
                           width=1.0, style='dashed', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=belongs_edges,
                           edge_color='#7B1FA2', arrows=True,
                           arrowstyle='->', arrowsize=12,
                           width=1.2, style='dotted', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=kg_edges,
                           edge_color='#7B1FA2', arrows=True,
                           arrowstyle='->', arrowsize=10,
                           width=1.8, ax=ax)

    # Draw assertion nodes
    all_colors = {**TYPE_COLORS, **DOMAIN_COLORS}
    for ntype, color in all_colors.items():
        nodes_of_type = [n for n in G.nodes()
                         if G.nodes[n].get('type') == ntype]
        if not nodes_of_type:
            continue
        shape = NODE_SHAPES.get(ntype, 'o')
        size = 900 if ntype == 'domain' else 600
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                               node_color=color, node_size=size,
                               node_shape=shape, alpha=0.9, ax=ax)

    # Labels
    labels = {}
    for n, d in G.nodes(data=True):
        if d.get('type') == 'knowledge_ref':
            labels[n] = d.get('concept', n).replace('KG:', '')
        elif d.get('type') == 'domain':
            labels[n] = d.get('name', n)[:25]
        else:
            step = d.get('step', '')
            tag = d.get('type', '')[0].upper() if d.get('type') else ''
            labels[n] = f"{step}{tag}\n{n}"
    nx.draw_networkx_labels(G, pos, labels, font_size=6,
                            font_family='monospace', ax=ax)

    # Edge labels for KG edges
    kg_edge_labels = {}
    for u, v, d in G.edges(data=True):
        rel = d.get('relation', '')
        if rel in ('REQUIRES', 'EXTENDS', 'INVERSE_OF'):
            kg_edge_labels[(u, v)] = rel
    if kg_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, kg_edge_labels,
                                     font_size=5, font_color='#7B1FA2',
                                     ax=ax)

    # Legend
    legend_patches = []
    for ntype in ['hypothesis', 'definition', 'claim', 'lemma',
                  'theorem', 'contradiction', 'knowledge_ref', 'domain']:
        color = all_colors.get(ntype)
        if color and any(G.nodes[n].get('type') == ntype for n in G.nodes()):
            legend_patches.append(mpatches.Patch(color=color, label=ntype))
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

    # Foundation summary
    reqs = mg.proof_requires(proof_id)
    domain_names = [mg._nodes.get(d, {}).get('name', d)[:20]
                    for d in reqs if d in mg._nodes]
    name = p.get('name', proof_id)
    ax.set_title(
        f"Deep Proof Tree: {name}\n"
        f"Method: {p.get('method', '?')} | "
        f"Depth: {mg.proof_depth(proof_id)} | "
        f"Domains: {len(reqs)}\n"
        f"Foundations: {', '.join(domain_names[:6])}"
        f"{'...' if len(domain_names) > 6 else ''}",
        fontsize=11)
    ax.axis('off')

    outpath = os.path.join(outdir, f"deep_proof_{proof_id.replace('-','_')}.pdf")
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  → {outpath}")
    return outpath


def print_proof_foundations(mg):
    """Print foundation analysis for all proofs."""
    print(f"\n{'='*60}")
    print("PROOF FOUNDATION ANALYSIS")
    print(f"{'='*60}")
    for pid in mg.proofs():
        p = mg.proof(pid)
        reqs = mg.proof_requires(pid)
        name = p.get('name', pid)
        print(f"\n{name}")
        print(f"  Direct concept refs:")
        for domain, concepts in sorted(reqs.items()):
            if concepts:
                dname = mg._nodes.get(domain, {}).get('name', domain)
                print(f"    {dname}: {concepts}")
        print(f"  Transitive domain deps:")
        for domain in sorted(reqs.keys()):
            if not reqs[domain]:  # transitive only
                dname = mg._nodes.get(domain, {}).get('name', domain)
                print(f"    → {dname} (via REQUIRES chain)")

    # Shared foundations
    pids = mg.proofs()
    print(f"\n{'='*60}")
    print("SHARED FOUNDATIONS")
    print(f"{'='*60}")
    for i, a in enumerate(pids):
        for b in pids[i+1:]:
            shared = mg.shared_foundations(a, b)
            if shared['shared']:
                na = mg.proof(a).get('name', a)[:30]
                nb = mg.proof(b).get('name', b)[:30]
                names = [mg._nodes.get(d, {}).get('name', d)[:20]
                         for d in shared['shared']]
                print(f"  {na} ∩ {nb}:")
                print(f"    shared: {names}")


if __name__ == '__main__':
    mg = MathGraph()
    outdir = os.path.join(os.path.dirname(__file__), '..', 'figures')

    args = sys.argv[1:]
    if '--cross' in args:
        draw_cross_proof_graph(mg, outdir)
    elif '--deep' in args:
        remaining = [a for a in args if a != '--deep']
        if remaining:
            for pid in remaining:
                draw_deep_proof_tree(mg, pid, outdir)
        else:
            for pid in mg.proofs():
                draw_deep_proof_tree(mg, pid, outdir)
            print_proof_foundations(mg)
    elif '--foundations' in args:
        print_proof_foundations(mg)
    elif args:
        for pid in args:
            draw_proof_tree(mg, pid, outdir)
    else:
        # All proof trees + deep trees + cross-proof
        for pid in mg.proofs():
            draw_proof_tree(mg, pid, outdir)
            draw_deep_proof_tree(mg, pid, outdir)
        draw_cross_proof_graph(mg, outdir)
        print_proof_foundations(mg)
        print(f"\nGenerated {len(mg.proofs())} proof trees "
              f"+ {len(mg.proofs())} deep trees + cross-proof graph")
