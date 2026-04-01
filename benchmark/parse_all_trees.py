"""
Parse ALL GSM8K training problems into computation trees.

Each GSM8K solution contains <<expr=result>> annotations that encode
the exact computation steps. We parse these into tree structures:
  - leaf_vals: numbers from the problem text
  - op: operation (add, subtract, multiply, divide)
  - children: indices of prior steps whose results are used
  - depth: distance from leaves

Output: benchmark/results/computation_trees_full.json
"""

import json
import re
import ast
import sys
from collections import Counter

# Pattern: <<48/2=24>> or <<100-50-30-15=5>>
STEP_RE = re.compile(r'<<(.+?)=([^>]+)>>')

# Final answer: #### 72
ANSWER_RE = re.compile(r'####\s*(.+)')

# Operators in order of precedence for detection
OP_MAP = {
    ast.Add: 'add',
    ast.Sub: 'subtract',
    ast.Mult: 'multiply',
    ast.Div: 'divide',
    ast.FloorDiv: 'divide',
    ast.Pow: 'power',
    ast.Mod: 'modulo',
}


def detect_op(expr_str: str) -> str:
    """Detect the root operation of an expression string."""
    # Clean up: GSM8K sometimes uses 'x' for multiply
    cleaned = expr_str.replace('x', '*').replace('×', '*').replace('÷', '/')
    try:
        tree = ast.parse(cleaned, mode='eval')
        body = tree.body
        if isinstance(body, ast.BinOp):
            return OP_MAP.get(type(body.op), 'unknown')
        if isinstance(body, ast.UnaryOp):
            return 'negate'
    except SyntaxError:
        pass

    # Fallback: regex-based detection (outermost operator)
    # Remove parenthesized subexpressions
    s = expr_str
    while '(' in s:
        s = re.sub(r'\([^()]*\)', '0', s)
    if '+' in s:
        return 'add'
    if '-' in s and not s.startswith('-'):
        return 'subtract'
    if any(c in s for c in ['*', 'x', '×']):
        return 'multiply'
    if any(c in s for c in ['/', '÷']):
        return 'divide'
    return 'unknown'


def get_leaf_values(expr_str: str) -> list[float]:
    """Extract all numeric literals from an expression (absolute values).

    We store absolute values because negative signs represent the subtract
    operation, not the number itself.  '100-50-30' has leaves [100, 50, 30],
    not [100, -50, -30].
    """
    nums = re.findall(r'\d+\.?\d*', expr_str)
    return [float(n) for n in nums]


def parse_solution(solution: str) -> dict | None:
    """Parse a GSM8K solution string into a computation tree."""
    steps = STEP_RE.findall(solution)
    if not steps:
        return None

    # Extract final answer
    answer_m = ANSWER_RE.search(solution)
    if not answer_m:
        return None

    answer_str = answer_m.group(1).strip().replace(',', '').replace('$', '')
    try:
        answer = float(answer_str)
    except ValueError:
        return None

    # Build nodes from steps
    nodes = []
    # Map result values to node indices for dependency tracking.
    # Use a list so we can handle duplicate result values (LIFO: most recent wins).
    result_index = []  # [(result_value, node_idx), ...]

    for i, (expr, result_str) in enumerate(steps):
        result_str = result_str.strip().replace(',', '')
        try:
            result = float(result_str)
        except ValueError:
            continue

        op = detect_op(expr)
        leaf_vals = get_leaf_values(expr)

        # Find children: which prior results appear as operands in this expression?
        # Match with tolerance for float precision.
        children = []
        remaining_leaves = list(leaf_vals)

        # Check most recent results first (LIFO) — closer steps are more likely refs
        for prev_result, prev_idx in reversed(result_index):
            # Try to match this prior result against a remaining leaf
            matched = False
            for j, lv in enumerate(remaining_leaves):
                if abs(lv - prev_result) < 0.001 or \
                   (prev_result != 0 and abs(lv / prev_result - 1) < 0.0001):
                    children.append(prev_idx)
                    remaining_leaves.pop(j)
                    matched = True
                    break

        # Clean leaf_vals to only include true leaves (not intermediate results)
        true_leaves = remaining_leaves

        node = {
            'idx': i,
            'op': op,
            'expr': expr.strip(),
            'result': result,
            'children': sorted(children),
            'leaf_vals': true_leaves,
            'depth': 0,
        }
        nodes.append(node)
        result_index.append((result, i))

    if not nodes:
        return None

    # Compute depths (BFS from leaves up)
    for node in nodes:
        if node['children']:
            node['depth'] = 1 + max(nodes[c]['depth'] for c in node['children']
                                     if c < len(nodes))

    root = nodes[-1]
    root_op = root['op']

    # Interior ops = all ops except root
    interior_ops = [n['op'] for n in nodes[:-1]] if len(nodes) > 1 else []

    return {
        'nodes': nodes,
        'root_idx': root['idx'],
        'root_op': root_op,
        'depth': max(n['depth'] for n in nodes),
        'n_steps': len(nodes),
    }


def build_inline_expr(steps: list[tuple[str, str]]) -> str | None:
    """Build a single inline expression by substituting intermediate results."""
    if not steps:
        return None

    # Build result→expr mapping for substitution
    exprs = {}
    for expr_str, result_str in steps:
        result_str = result_str.strip().replace(',', '')
        try:
            result = float(result_str)
        except ValueError:
            continue
        exprs[result] = expr_str.strip()

    # Start from last expression and substitute backwards
    final_expr = steps[-1][0].strip()
    # Substitute intermediate results with their expressions
    for i in range(len(steps) - 2, -1, -1):
        expr_str, result_str = steps[i]
        result_str = result_str.strip().replace(',', '')
        try:
            result = float(result_str)
        except ValueError:
            continue

        # Replace the result value in later expressions with (expr)
        result_int = int(result) if result == int(result) else None

        # Try exact float replacement, then int replacement
        for target in [str(result), str(result_int) if result_int is not None else None]:
            if target is None:
                continue
            # Use word boundary replacement to avoid partial matches
            pattern = re.escape(target)
            if re.search(r'(?<!\d)' + pattern + r'(?!\d)', final_expr):
                final_expr = re.sub(
                    r'(?<!\d)' + pattern + r'(?!\d)',
                    f'({expr_str.strip()})',
                    final_expr,
                    count=1
                )
                break

    return final_expr


def main():
    from datasets import load_dataset

    print("Loading GSM8K training set...")
    ds = load_dataset('openai/gsm8k', 'main', split='train')
    print(f"Loaded {len(ds)} problems")

    trees = []
    failures = []
    op_counts = Counter()
    depth_counts = Counter()

    for i, example in enumerate(ds):
        question = example['question']
        solution = example['answer']

        # Extract final answer
        answer_m = ANSWER_RE.search(solution)
        if not answer_m:
            failures.append({'gsm_idx': i, 'reason': 'no_answer_marker'})
            continue

        answer_str = answer_m.group(1).strip().replace(',', '').replace('$', '')
        try:
            answer = float(answer_str)
        except ValueError:
            failures.append({'gsm_idx': i, 'reason': 'bad_answer_value'})
            continue

        # Parse tree
        tree = parse_solution(solution)
        if tree is None:
            failures.append({'gsm_idx': i, 'reason': 'no_steps_parsed'})
            continue

        # Build inline expression
        steps = STEP_RE.findall(solution)
        inline_expr = build_inline_expr(steps)

        # Validate inline expression
        expr_valid = False
        if inline_expr:
            try:
                cleaned = inline_expr.replace(',', '').replace('x', '*').replace('×', '*').replace('÷', '/')
                # Fix implicit multiplication: "2(500)" → "2*(500)"
                cleaned = re.sub(r'(\d)\(', r'\1*(', cleaned)
                val = eval(cleaned)
                expr_valid = abs(val - answer) < 0.01
            except Exception:
                pass

        entry = {
            'id': f'gsm-train-{i:04d}',
            'gsm_idx': i,
            'question': question,
            'solution': solution,
            'answer': str(int(answer) if answer == int(answer) else answer),
            'root_op': tree['root_op'],
            'depth': tree['depth'],
            'n_steps': tree['n_steps'],
            'interior_ops': [n['op'] for n in tree['nodes'][:-1]],
            'tree': tree,
            'inline_expr': inline_expr,
            'expr_valid': expr_valid,
        }
        trees.append(entry)
        op_counts[tree['root_op']] += 1
        depth_counts[tree['depth']] += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} "
                  f"({len(trees)} trees, {len(failures)} failures)")

    # Save results
    out_path = 'benchmark/results/computation_trees_full.json'
    with open(out_path, 'w') as f:
        json.dump(trees, f, indent=2)
    print(f"\nSaved {len(trees)} trees to {out_path}")

    # Save failures
    fail_path = 'benchmark/results/tree_parse_failures.json'
    with open(fail_path, 'w') as f:
        json.dump(failures, f, indent=2)
    print(f"Saved {len(failures)} failures to {fail_path}")

    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Total problems: {len(ds)}")
    print(f"Trees parsed:   {len(trees)} ({100*len(trees)/len(ds):.1f}%)")
    print(f"Failures:       {len(failures)} ({100*len(failures)/len(ds):.1f}%)")
    print(f"Expr valid:     {sum(1 for t in trees if t['expr_valid'])} "
          f"({100*sum(1 for t in trees if t['expr_valid'])/len(trees):.1f}%)")

    print(f"\nRoot operations:")
    for op, cnt in op_counts.most_common():
        print(f"  {op}: {cnt} ({100*cnt/len(trees):.1f}%)")

    print(f"\nTree depths:")
    for d in sorted(depth_counts.keys()):
        cnt = depth_counts[d]
        print(f"  depth {d}: {cnt} ({100*cnt/len(trees):.1f}%)")

    # Hub+Scale distribution
    print(f"\nHub+Scale view:")
    add_sub = op_counts['add'] + op_counts['subtract']
    mul_div = op_counts['multiply'] + op_counts['divide']
    print(f"  Root is ADD/SUB: {add_sub} ({100*add_sub/len(trees):.1f}%)")
    print(f"  Root is MUL/DIV: {mul_div} ({100*mul_div/len(trees):.1f}%)")

    # Step count distribution
    step_counts = Counter(t['n_steps'] for t in trees)
    print(f"\nStep counts:")
    for s in sorted(step_counts.keys()):
        cnt = step_counts[s]
        print(f"  {s} steps: {cnt} ({100*cnt/len(trees):.1f}%)")


if __name__ == '__main__':
    main()
