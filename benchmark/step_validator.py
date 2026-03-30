"""
benchmark/step_validator.py — Validate candidate interpretations against
LLM-decomposed correct solutions.

Architecture:
  1. LLM (Haiku) decomposes the correct solution into atomic steps
  2. Each candidate AST is decomposed into its computation steps
  3. Steps are compared: which numbers, operations, and intermediate results match?
  4. Score: steps_correct / steps_total per candidate

This gives us confidence in the STRATEGY (interpretation), not the answer.
A candidate that gets 4/5 steps right but misses one division is better
than a candidate that gets 1/5 steps right but happens to land close.

The step decomposition feeds back into the graph — we learn which solver
gets which interpretation step right for each equivalence class.
"""

import json
import subprocess
import re
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

RESULTS_DIR = Path(__file__).parent / "results"
SUITES_DIR = Path(__file__).parent / "suites"


@dataclass
class StepMatch:
    """How well a candidate's steps match the correct solution."""
    solver: str
    answer: float
    correct_answer: float
    steps_matched: int
    steps_total: int
    step_details: list  # [(step_desc, matched: bool)]
    score: float  # steps_matched / steps_total


def call_haiku(prompt: str) -> str:
    """Call Claude Haiku for step decomposition."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text",
             "--model", "sonnet"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def decompose_solution(question: str, answer: float) -> list[dict]:
    """
    Ask Haiku to decompose a word problem solution into atomic steps.
    Returns list of {step, operation, operands, result}.
    """
    prompt = f"""Decompose this word problem solution into atomic arithmetic steps.

PROBLEM: {question}
ANSWER: {answer}

Return ONLY a JSON array. Each step has:
- "description": what this step computes in words
- "operation": one of "add", "subtract", "multiply", "divide"
- "operands": [number1, number2]
- "result": the result of this step

Example: [{{"description": "eggs per day minus eaten", "operation": "subtract", "operands": [16, 3], "result": 13}}]

Rules:
- Each step is ONE arithmetic operation on TWO numbers
- Results from earlier steps can be operands in later steps
- The final step's result must equal {answer}
- Return ONLY the JSON array, no other text."""

    response = call_haiku(prompt)
    if not response:
        return []

    try:
        steps = json.loads(response)
        if isinstance(steps, list):
            return steps
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown
    m = re.search(r'\[.*\]', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return []


def ast_to_steps(ast, execute_fn) -> list[dict]:
    """
    Decompose an AST into computation steps.
    Returns list of {operation, operands, result}.
    """
    from packages.core.interpret import BinOp, Lit, Neg, Ref

    steps = []

    def walk(node):
        if isinstance(node, Lit):
            return node.value
        if isinstance(node, Neg):
            child_val = walk(node.child)
            if child_val is not None:
                result = -child_val
                steps.append({
                    "operation": "negate",
                    "operands": [child_val],
                    "result": result,
                })
                return result
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
                steps.append({
                    "operation": node.op,
                    "operands": [left_val, right_val],
                    "result": result,
                })
                return result
            return None
        return None

    walk(ast)
    return steps


def match_steps(candidate_steps: list[dict], correct_steps: list[dict]) -> tuple[int, int, list]:
    """
    Match candidate computation steps against correct steps.
    Returns (matched, total, details).

    A step matches if:
    - Same operation AND same operands (order-independent for add/multiply)
    - OR same operands and same result (operation name might differ)
    """
    total = len(correct_steps)
    if total == 0:
        return 0, 0, []

    matched = 0
    details = []
    used = set()

    for cs in correct_steps:
        c_op = cs.get("operation", "")
        c_ops = sorted(cs.get("operands", []))
        c_res = cs.get("result")

        found = False
        for j, ms in enumerate(candidate_steps):
            if j in used:
                continue
            m_op = ms.get("operation", "")
            m_ops = sorted(ms.get("operands", []))
            m_res = ms.get("result")

            # Check operands match (within tolerance)
            ops_match = (len(c_ops) == len(m_ops) and
                        all(abs(a - b) < 0.01 for a, b in zip(c_ops, m_ops)))

            # Operation match or result match
            op_match = c_op == m_op
            res_match = (c_res is not None and m_res is not None and
                        abs(c_res - m_res) < 0.01)

            if ops_match and (op_match or res_match):
                found = True
                used.add(j)
                break

        if found:
            matched += 1
        details.append({
            "step": cs.get("description", f"{c_op}({c_ops})={c_res}"),
            "matched": found,
        })

    return matched, total, details


def validate_candidates(problem: dict, candidates: list, execute_fn) -> list[StepMatch]:
    """
    Validate all candidates for a problem against LLM-decomposed solution.

    candidates: list of (value, solver_name, priority, ...) tuples
    """
    expected = float(problem['answer'].replace(',', ''))

    # Get correct decomposition from Haiku
    correct_steps = decompose_solution(problem['question'], expected)
    if not correct_steps:
        return []

    results = []
    for entry in candidates:
        val, name = entry[0], entry[1]
        if val is None:
            continue

        # We don't have the AST here, just the answer
        # For now, score based on whether intermediate results match
        # (Full AST decomposition requires passing ASTs through)
        results.append(StepMatch(
            solver=name,
            answer=val,
            correct_answer=expected,
            steps_matched=0,
            steps_total=len(correct_steps),
            step_details=[],
            score=0.0,
        ))

    return results


def run_step_validation(limit=20):
    """
    Run step validation on missed-winner problems.
    These are problems where we had the right answer but picked wrong.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from packages.core.interpret import compile, execute, BinOp, Lit

    # Load candidate eval to find missed winners
    eval_path = RESULTS_DIR / "candidate_eval.json"
    if not eval_path.exists():
        print("Run eval_candidates.py first")
        return

    with open(eval_path) as f:
        eval_data = json.load(f)

    missed = eval_data.get("missed_winners", [])
    print(f"Validating step-level accuracy for {min(limit, len(missed))} "
          f"missed-winner problems...\n")

    # Load problems
    problems = {}
    for suite in ['gsm8k.json', 'gsm8k_v2.json', 'gsm8k_500.json']:
        path = SUITES_DIR / suite
        if path.exists():
            with open(path) as f:
                for c in json.load(f)['cases']:
                    problems[c['id']] = c

    results = []

    for i, m in enumerate(missed[:limit]):
        pid = m['id']
        p = problems.get(pid)
        if not p:
            continue

        expected = float(p['answer'].replace(',', ''))
        print(f"\n{'='*60}")
        print(f"Problem {pid}: expected={expected}")
        print(f"Q: {p['question'][:100]}...")

        # Get correct decomposition
        correct_steps = decompose_solution(p['question'], expected)
        if not correct_steps:
            print("  (Haiku failed to decompose)")
            continue

        print(f"  Correct steps ({len(correct_steps)}):")
        for s in correct_steps:
            print(f"    {s.get('description', '')}: "
                  f"{s.get('operation', '')}({s.get('operands', [])}) = {s.get('result', '?')}")

        # Re-compile to get ASTs
        ast, ctx = compile(p['question'])
        candidate_asts = getattr(ctx, '_candidate_asts', None)

        # If we don't have ASTs (already evaluated), re-get candidates
        all_cands = getattr(ctx, 'all_candidates', [])

        print(f"\n  Candidates:")
        for entry in all_cands:
            val, name = entry[0], entry[1]
            strat = entry[3] if len(entry) > 3 else 0
            ok = "✓" if val is not None and abs(val - expected) < 0.01 else "✗"
            print(f"    {ok} {name}: {val} (strategy={strat:.2f})")

        # For each candidate, decompose its AST and match steps
        # We need the ASTs — they're in _candidate_asts before compile deletes them
        # Re-compile to get them
        from packages.core.interpret import parse, build_ast
        clauses, ctx2 = parse(p['question'])
        ast2 = build_ast(clauses, ctx2)

        if hasattr(ctx2, '_candidate_asts'):
            print(f"\n  Step matching:")
            for c_ast, name, pri in ctx2._candidate_asts:
                c_steps = ast_to_steps(c_ast, execute)
                matched, total, details = match_steps(c_steps, correct_steps)
                score = matched / total if total else 0
                val = execute(c_ast)
                ok = "✓" if val is not None and abs(val - expected) < 0.01 else "✗"
                print(f"    {ok} {name}: {matched}/{total} steps matched "
                      f"(score={score:.0%}), answer={val}")
                for d in details:
                    m_str = "✓" if d['matched'] else "✗"
                    print(f"      {m_str} {d['step']}")

                results.append({
                    'problem_id': pid,
                    'solver': name,
                    'answer': val,
                    'expected': expected,
                    'correct': val is not None and abs(val - expected) < 0.01,
                    'steps_matched': matched,
                    'steps_total': total,
                    'step_score': score,
                })

    # Summary
    print(f"\n{'='*60}")
    print(f"STEP-LEVEL SUMMARY")
    print(f"{'='*60}")

    by_solver = defaultdict(lambda: {"total_steps": 0, "matched_steps": 0,
                                      "problems": 0, "correct": 0})
    for r in results:
        s = by_solver[r['solver']]
        s["total_steps"] += r['steps_total']
        s["matched_steps"] += r['steps_matched']
        s["problems"] += 1
        if r['correct']:
            s["correct"] += 1

    print(f"\n{'Solver':<20} {'Problems':>8} {'Correct':>8} "
          f"{'Steps':>10} {'StepRate':>8}")
    print(f"{'-'*56}")
    for solver in sorted(by_solver.keys(),
                         key=lambda s: -by_solver[s]["matched_steps"]):
        s = by_solver[solver]
        step_rate = s["matched_steps"] / s["total_steps"] if s["total_steps"] else 0
        print(f"{solver:<20} {s['problems']:>8} {s['correct']:>8} "
              f"{s['matched_steps']:>4}/{s['total_steps']:<4} {step_rate:>7.0%}")

    # Save
    out_path = RESULTS_DIR / "step_validation.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    run_step_validation(limit=args.limit)
