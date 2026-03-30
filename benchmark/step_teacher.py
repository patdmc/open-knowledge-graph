"""
benchmark/step_teacher.py — Step-level teacher for the learning loop.

Uses Sonnet to decompose correct solutions into atomic arithmetic steps.
Caches decompositions so we only burn tokens once per problem.
Compares each candidate solver's steps against the correct decomposition.

Architecture:
  1. Sonnet decomposes correct solution → atomic steps (cached)
  2. Each candidate AST → atomic steps (mechanical, 0 tokens)
  3. Step matching → which steps each solver got right
  4. Step corrections → feed into graph learning

Constants are nodes. Equations are edges.
Intermediate values are a constant plus an equation.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

RESULTS_DIR = Path(__file__).parent / "results"
SUITES_DIR = Path(__file__).parent / "suites"
CACHE_DIR = RESULTS_DIR / "step_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def call_sonnet(prompt: str) -> str:
    """Call Sonnet via Claude CLI."""
    import subprocess
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text",
             "--model", "sonnet"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"    Sonnet error: {result.stderr[:100]}", file=sys.stderr)
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    Sonnet error: {e}", file=sys.stderr)
        return ""


def decompose_correct(question: str, answer: float, problem_id: str = "") -> list[dict]:
    """
    Ask Sonnet to decompose a word problem solution into atomic steps.
    Results are cached by problem_id.
    """
    # Check cache
    if problem_id:
        cache_path = CACHE_DIR / f"{problem_id}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    prompt = f"""Solve this word problem and show each arithmetic step.

PROBLEM: {question}
ANSWER: {answer}

Return ONLY a JSON array. Each element:
{{"step": 1, "description": "what this computes", "operation": "add|subtract|multiply|divide", "operands": [num1, num2], "result": number, "variable": "what this result represents"}}

Rules:
- Each step is ONE arithmetic operation on exactly TWO numbers
- Later steps can use results from earlier steps as operands
- The final step's result must equal {answer}
- "variable" names what the result represents (e.g., "total_cost", "baseballs_cost")
- Return ONLY the JSON array."""

    response = call_sonnet(prompt)
    if not response:
        return []

    # Parse JSON (handle markdown wrapping)
    steps = None
    try:
        steps = json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if m:
            try:
                steps = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        if steps is None:
            m = re.search(r'\[.*\]', response, re.DOTALL)
            if m:
                try:
                    steps = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

    if not isinstance(steps, list):
        return []

    # Validate: final result should match answer
    if steps and abs(steps[-1].get("result", 0) - answer) > 0.01:
        # Try again or accept with warning
        pass

    # Cache
    if problem_id:
        cache_path = CACHE_DIR / f"{problem_id}.json"
        with open(cache_path, 'w') as f:
            json.dump(steps, f, indent=2)

    return steps


def decompose_ast(ast, execute_fn) -> list[dict]:
    """Decompose a candidate AST into computation steps."""
    from packages.core.interpret import BinOp, Lit, Neg, Ref

    steps = []
    step_num = [0]

    def walk(node):
        if isinstance(node, Lit):
            return node.value
        if isinstance(node, Neg):
            child_val = walk(node.child)
            if child_val is not None:
                step_num[0] += 1
                steps.append({
                    "step": step_num[0],
                    "operation": "negate",
                    "operands": [child_val],
                    "result": -child_val,
                })
                return -child_val
            return None
        if isinstance(node, Ref):
            if node.resolved:
                return walk(node.resolved)
            return None
        if isinstance(node, BinOp):
            left_val = walk(node.left)
            right_val = walk(node.right)
            if left_val is not None and right_val is not None:
                try:
                    result = execute_fn(node)
                except Exception:
                    return None
                step_num[0] += 1
                steps.append({
                    "step": step_num[0],
                    "operation": node.op,
                    "operands": [left_val, right_val],
                    "result": result,
                })
                return result
            return None
        return None

    walk(ast)
    return steps


def match_steps(our_steps: list[dict], correct_steps: list[dict]) -> list[dict]:
    """Match our steps against correct steps. Returns enriched list."""
    results = []
    used_ours = set()

    for cs in correct_steps:
        c_op = cs.get("operation", "")
        c_ops = cs.get("operands", [])
        c_res = cs.get("result")

        match_info = {
            "correct_step": cs,
            "matched": False,
            "our_step": None,
            "partial": "",
        }

        for j, ms in enumerate(our_steps):
            if j in used_ours:
                continue

            m_op = ms.get("operation", "")
            m_ops = ms.get("operands", [])
            m_res = ms.get("result")

            # Operand match (order-independent for commutative ops)
            c_sorted = sorted(c_ops) if c_ops else []
            m_sorted = sorted(m_ops) if m_ops else []
            ops_match = (len(c_sorted) == len(m_sorted) and
                        all(abs(a - b) < 0.01 for a, b in zip(c_sorted, m_sorted)))

            op_match = c_op == m_op
            res_match = (c_res is not None and m_res is not None and
                        abs(c_res - m_res) < 0.01)

            if ops_match and (op_match or res_match):
                match_info["matched"] = True
                match_info["our_step"] = ms
                used_ours.add(j)
                break
            elif ops_match and not op_match:
                match_info["partial"] = f"wrong_op:{m_op}_should_be_{c_op}"
                match_info["our_step"] = ms

        results.append(match_info)

    return results


def run_step_teaching(limit=20):
    """
    For each wrong answer (one per equivalence class):
    1. Sonnet decomposes correct solution (cached)
    2. Each candidate's AST steps matched
    3. Step-level corrections generated
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from packages.core.interpret import compile, execute, parse, build_ast

    # Load problems
    problems = {}
    all_problems = []
    for suite in ['gsm8k.json', 'gsm8k_v2.json', 'gsm8k_500.json']:
        path = SUITES_DIR / suite
        if path.exists():
            with open(path) as f:
                for c in json.load(f)['cases']:
                    if c['id'] not in problems:
                        problems[c['id']] = c
                        all_problems.append(c)

    print(f"Running step-level teaching on {len(all_problems)} problems...\n")

    # First pass: find wrong answers
    wrong_by_class = {}
    t0 = time.monotonic()

    for p in all_problems:
        try:
            expected = float(p['answer'].replace(',', ''))
        except (ValueError, AttributeError):
            continue

        try:
            ast, ctx = compile(p['question'])
            answer = execute(ast) if ast else None
        except Exception:
            continue

        if answer is not None and abs(answer - expected) < 0.01:
            continue

        classes = getattr(ctx, 'equivalence_classes', []) or ['unclassified']
        for cls in classes:
            if cls not in wrong_by_class:
                wrong_by_class[cls] = {
                    'problem': p, 'expected': expected,
                    'answer': answer,
                }

    print(f"Found {len(wrong_by_class)} equivalence classes with wrong answers "
          f"({time.monotonic()-t0:.1f}s)")

    # Second pass: teach one per class
    step_corrections = []
    taught = 0
    step_totals = defaultdict(lambda: {"matched": 0, "total": 0})

    for cls, info in sorted(wrong_by_class.items()):
        if taught >= limit:
            break

        p = info['problem']
        expected = info['expected']

        print(f"\n{'='*50}")
        print(f"Class: {cls} | Problem: {p['id']}")
        print(f"Q: {p['question'][:80]}...")
        print(f"Expected: {expected}")

        # Sonnet decomposition (cached)
        correct_steps = decompose_correct(p['question'], expected, p['id'])
        if not correct_steps:
            print("  Sonnet failed")
            continue

        print(f"  Correct solution ({len(correct_steps)} steps):")
        for s in correct_steps:
            print(f"    Step {s.get('step','?')}: {s.get('description','')} "
                  f"= {s.get('result','?')} [{s.get('variable','')}]")

        # Re-compile to get candidate ASTs
        clauses, ctx2 = parse(p['question'])
        ast2 = build_ast(clauses, ctx2)

        if not hasattr(ctx2, '_candidate_asts'):
            print("  No candidates")
            continue

        best_match = None
        best_score = -1

        for c_ast, name, pri in ctx2._candidate_asts:
            our_steps = decompose_ast(c_ast, execute)
            matches = match_steps(our_steps, correct_steps)

            matched = sum(1 for m in matches if m['matched'])
            total = len(correct_steps)
            score = matched / total if total else 0
            val = execute(c_ast)
            ok = val is not None and abs(val - expected) < 0.01

            step_totals[name]["matched"] += matched
            step_totals[name]["total"] += total

            print(f"\n  {name} → {val} {'✓' if ok else '✗'} ({matched}/{total} steps)")
            for m in matches:
                cs = m['correct_step']
                sym = '✓' if m['matched'] else '✗'
                partial = f" ({m.get('partial', '')})" if m.get('partial') else ""
                print(f"    {sym} {cs.get('description','')}"
                      f" = {cs.get('result','?')}{partial}")

            if score > best_score:
                best_score = score
                best_match = (name, matches, val, ok)

            for m in matches:
                if not m['matched']:
                    cs = m['correct_step']
                    step_corrections.append({
                        'class': cls,
                        'problem_id': p['id'],
                        'solver': name,
                        'correct_step': cs,
                        'partial': m.get('partial', ''),
                        'our_step': m.get('our_step'),
                    })

        taught += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"STEP-LEVEL TEACHING SUMMARY")
    print(f"{'='*60}")
    print(f"Classes taught: {taught}")
    print(f"Step corrections: {len(step_corrections)}")

    print(f"\n{'Solver':<20} {'Matched':>8} {'Total':>6} {'StepRate':>8}")
    print(f"{'-'*44}")
    for solver in sorted(step_totals.keys(),
                         key=lambda s: -(step_totals[s]["matched"] /
                                         max(step_totals[s]["total"], 1))):
        s = step_totals[solver]
        rate = s["matched"] / s["total"] if s["total"] else 0
        print(f"{solver:<20} {s['matched']:>8} {s['total']:>6} {rate:>7.0%}")

    # Save
    out_path = RESULTS_DIR / "step_corrections.json"
    with open(out_path, 'w') as f:
        json.dump(step_corrections, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    run_step_teaching(limit=args.limit)
