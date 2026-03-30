"""
benchmark/decompose.py — Decompose logic from answers.

For each problem, captures the PROCESS (solver, graph structure, walk path)
separately from the RESULT (correct/incorrect). Then compares process
patterns between right and wrong answers within the same equivalence class.

Usage:
    python3 -m benchmark.decompose
    python3 -m benchmark.decompose --limit 50
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

SUITES_DIR = Path(__file__).parent / "suites"


def ast_depth(node):
    if node is None:
        return 0
    if hasattr(node, 'left'):
        return 1 + max(ast_depth(node.left), ast_depth(node.right))
    if hasattr(node, 'child'):
        return 1 + ast_depth(node.child)
    if hasattr(node, 'resolved'):
        return 1 + (ast_depth(node.resolved) if node.resolved else 0)
    return 1


def load_all_problems():
    problems = []
    seen = set()
    for suite in ['gsm8k.json', 'gsm8k_v2.json', 'gsm8k_500.json']:
        path = SUITES_DIR / suite
        if not path.exists():
            continue
        with open(path) as f:
            for c in json.load(f)['cases']:
                if c['id'] not in seen:
                    seen.add(c['id'])
                    problems.append(c)
    return problems


def decompose(problems):
    """Run each problem and capture logic + result separately."""
    from packages.core.interpret import compile, execute

    records = []
    for i, p in enumerate(problems):
        try:
            expected = float(p['answer'].replace(',', ''))
        except (ValueError, AttributeError):
            continue

        try:
            ast, ctx = compile(p['question'])
            answer = execute(ast) if ast else None
        except Exception:
            answer = None
            class _Ctx:
                equivalence_classes = []
                solver_used = 'error'
                problem_graph = None
                walk_result = None
                sentences = []
            ctx = _Ctx()

        ok = answer is not None and abs(answer - expected) < 0.01

        pg = getattr(ctx, 'problem_graph', None)
        wr = getattr(ctx, 'walk_result', None)

        record = {
            'id': p['id'],
            'classes': getattr(ctx, 'equivalence_classes', []) or ['unclassified'],
            'solver': getattr(ctx, 'solver_used', ''),
            'correct': ok,
            'answer': answer,
            'expected': expected,
            'n_clauses': len(getattr(ctx, 'sentences', [])),
            'has_rate': bool(pg and pg.nodes_with_per_unit()),
            'n_nodes': len(pg.nodes) if pg else 0,
            'n_entities': len(pg.entities) if pg else 0,
            'walk_conf': wr.confidence if wr else 0,
            'walk_path': wr.path if wr else '',
            'walk_used': wr.nodes_used if wr else 0,
            'walk_total': wr.nodes_total if wr else 0,
            'ast_depth': ast_depth(ast),
            # Logic signature: compact representation of what the solver DID
            'logic_sig': f"{getattr(ctx, 'solver_used', '')}|{'rate' if (pg and pg.nodes_with_per_unit()) else 'no_rate'}|d{ast_depth(ast)}|n{len(pg.nodes) if pg else 0}",
        }
        records.append(record)

        if (i + 1) % 100 == 0:
            c = sum(1 for r in records if r['correct'])
            print(f"  {i+1}/{len(problems)}  correct={c}/{i+1}", file=sys.stderr)

    return records


def analyze_classes(records):
    """Compare logic patterns between right and wrong within each class."""
    by_class = defaultdict(lambda: {'right': [], 'wrong': []})
    for r in records:
        for cls in r['classes']:
            bucket = 'right' if r['correct'] else 'wrong'
            by_class[cls][bucket].append(r)

    for cls in sorted(by_class.keys()):
        right = by_class[cls]['right']
        wrong = by_class[cls]['wrong']
        total = len(right) + len(wrong)
        rate = len(right) / total if total else 0
        print(f"\n{'='*60}")
        print(f"{cls}: {len(right)}/{total} ({rate:.0%})")
        print(f"{'='*60}")

        # Logic signature distribution
        for label, group in [("RIGHT", right), ("WRONG", wrong)]:
            if not group:
                continue
            sigs = defaultdict(int)
            for r in group:
                sigs[r['logic_sig']] += 1

            solvers = defaultdict(int)
            for r in group:
                solvers[r['solver']] += 1

            has_rate = sum(1 for r in group if r['has_rate'])
            has_walk = sum(1 for r in group if r['walk_conf'] > 0)
            avg_nodes = sum(r['n_nodes'] for r in group) / len(group)
            avg_depth = sum(r['ast_depth'] for r in group) / len(group)

            print(f"\n  {label} ({len(group)}):")
            print(f"    solvers: {dict(sorted(solvers.items(), key=lambda x: -x[1]))}")
            print(f"    has_rate: {has_rate}/{len(group)}, walk: {has_walk}/{len(group)}")
            print(f"    avg nodes: {avg_nodes:.1f}, avg ast_depth: {avg_depth:.1f}")

            # Top logic signatures
            top_sigs = sorted(sigs.items(), key=lambda x: -x[1])[:5]
            print(f"    top logic: {dict(top_sigs)}")

        # Cross-compare: what logic do right answers use that wrong ones don't?
        if right and wrong:
            r_sigs = set(r['logic_sig'] for r in right)
            w_sigs = set(r['logic_sig'] for r in wrong)
            only_right = r_sigs - w_sigs
            only_wrong = w_sigs - r_sigs
            if only_right:
                print(f"\n  Logic ONLY in correct: {only_right}")
            if only_wrong and len(only_wrong) <= 10:
                print(f"  Logic ONLY in wrong: {only_wrong}")

            # Show example wrong answers with their answer/expected ratio
            print(f"\n  Sample wrong answers:")
            for w in wrong[:3]:
                ratio = w['answer'] / w['expected'] if w['answer'] and w['expected'] else None
                ratio_str = f"{ratio:.3f}" if ratio else "N/A"
                print(f"    {w['id']}: got={w['answer']}, exp={w['expected']}, "
                      f"ratio={ratio_str}, solver={w['solver']}, "
                      f"nodes={w['n_nodes']}, walk={w['walk_path']}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    problems = load_all_problems()
    if args.limit:
        problems = problems[:args.limit]

    print(f"Decomposing {len(problems)} problems...", file=sys.stderr)
    records = decompose(problems)

    correct = sum(1 for r in records if r['correct'])
    print(f"\nTotal: {correct}/{len(records)} ({correct/len(records):.1%})")

    analyze_classes(records)

    # Save records for later comparison
    out_path = Path(__file__).parent / "results" / "decomposed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
