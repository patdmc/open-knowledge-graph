"""
benchmark/batch_decompose.py — Batch decompose correct solutions via Sonnet.

For every problem where no candidate is correct, ask Sonnet to
decompose the correct solution into atomic arithmetic steps.
Cache results so we only pay once per problem.

Then analyze: what operation patterns appear? What structures
do we need to learn?
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

RESULTS_DIR = Path(__file__).parent / "results"
SUITES_DIR = Path(__file__).parent / "suites"
CACHE_DIR = RESULTS_DIR / "step_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def call_sonnet(prompt: str) -> str:
    """Call Sonnet via Claude CLI."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text",
             "--model", "sonnet"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def decompose(question: str, answer: float, problem_id: str) -> list[dict]:
    """Decompose with caching."""
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
- Return ONLY the JSON array."""

    response = call_sonnet(prompt)
    if not response:
        return []

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

    # Cache
    with open(cache_path, 'w') as f:
        json.dump(steps, f, indent=2)

    return steps


def load_problems(suites=None):
    if suites is None:
        suites = ['gsm8k.json', 'gsm8k_v2.json', 'gsm8k_500.json']
    problems = []
    seen = set()
    for suite in suites:
        path = SUITES_DIR / suite
        if not path.exists():
            continue
        with open(path) as f:
            for c in json.load(f)['cases']:
                if c['id'] not in seen:
                    seen.add(c['id'])
                    problems.append(c)
    return problems


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from packages.core.interpret import compile, execute

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="Max problems to decompose per batch")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze cached decompositions")
    args = parser.parse_args()

    problems = load_problems()

    # Classify all problems
    correct_problems = []
    any_correct_problems = []
    no_correct = []
    t0 = time.monotonic()

    for p in problems:
        try:
            expected = float(p['answer'].replace(',', ''))
        except (ValueError, AttributeError):
            continue

        try:
            ast, ctx = compile(p['question'])
            answer = execute(ast) if ast else None
        except Exception:
            answer = None

        winner_ok = answer is not None and abs(answer - expected) < 0.01
        any_ok = False
        for entry in getattr(ctx, 'all_candidates', []):
            val = entry[0]
            if val is not None and abs(val - expected) < 0.01:
                any_ok = True
                break

        p['_winner_ok'] = winner_ok
        p['_any_ok'] = any_ok
        if winner_ok:
            correct_problems.append(p)
        elif any_ok:
            any_correct_problems.append(p)
        else:
            no_correct.append(p)

    elapsed = time.monotonic() - t0
    all_problems_valid = correct_problems + any_correct_problems + no_correct
    print(f"Winner correct: {len(correct_problems)}, "
          f"Any correct: {len(any_correct_problems)}, "
          f"None correct: {len(no_correct)} "
          f"({elapsed:.1f}s)")

    if not args.analyze_only:
        # Decompose ALL problems (correct and wrong)
        cached = sum(1 for p in all_problems_valid
                     if (CACHE_DIR / f"{p['id']}.json").exists())
        to_decompose = [p for p in all_problems_valid
                        if not (CACHE_DIR / f"{p['id']}.json").exists()]
        print(f"Already cached: {cached}, to decompose: {len(to_decompose)}")

        batch = to_decompose[:args.limit]
        print(f"\nDecomposing {len(batch)} problems via Sonnet...")

        for i, p in enumerate(batch):
            expected = float(p['answer'].replace(',', ''))
            steps = decompose(p['question'], expected, p['id'])
            status = f"{len(steps)} steps" if steps else "FAILED"
            print(f"  [{i+1}/{len(batch)}] {p['id']}: {status}")

    # --- Analyze all cached decompositions ---
    print(f"\n{'='*60}")
    print(f"ANALYZING CACHED DECOMPOSITIONS")
    print(f"{'='*60}")

    def analyze_group(group, label):
        steps_list = []
        n_with_steps = 0
        step_counts = Counter()
        op_sequences = Counter()
        op_counts = Counter()
        first_ops = Counter()

        for p in group:
            cache_path = CACHE_DIR / f"{p['id']}.json"
            if not cache_path.exists():
                continue
            with open(cache_path) as f:
                steps = json.load(f)
            if not steps:
                continue

            n_with_steps += 1
            step_counts[len(steps)] += 1
            ops = tuple(s.get("operation", "?") for s in steps)
            op_sequences[ops] += 1
            first_ops[steps[0].get("operation", "?")] += 1
            for s in steps:
                op_counts[s.get("operation", "?")] += 1
                steps_list.append(s)

        print(f"\n--- {label} ({n_with_steps}/{len(group)} cached) ---")

        print(f"  Step count distribution:")
        for n, count in sorted(step_counts.items()):
            bar = '█' * min(count, 60)
            print(f"    {n} steps: {count:>4} {bar}")

        print(f"  Operation frequency:")
        for op, count in op_counts.most_common():
            pct = count / len(steps_list) * 100 if steps_list else 0
            print(f"    {op:<12} {count:>5} ({pct:.0f}%)")

        print(f"  Top 15 operation sequences:")
        for seq, count in op_sequences.most_common(15):
            print(f"    {' → '.join(seq):<50} {count:>4}")

        print(f"  First operation:")
        for op, count in first_ops.most_common():
            print(f"    {op:<12} {count:>5}")

        return op_sequences, step_counts, op_counts

    correct_seqs, correct_steps, correct_ops = analyze_group(
        correct_problems, "CORRECT (winner right)")
    any_seqs, any_steps, any_ops = analyze_group(
        any_correct_problems, "ANY CORRECT (right answer available, wrong pick)")
    wrong_seqs, wrong_steps, wrong_ops = analyze_group(
        no_correct, "NO CORRECT (no candidate right)")

    # --- Compare: what sequences appear in WRONG but not CORRECT? ---
    print(f"\n{'='*60}")
    print(f"SEQUENCES WE CAN'T HANDLE (in wrong, not in correct)")
    print(f"{'='*60}")
    correct_set = set(correct_seqs.keys())
    for seq, count in wrong_seqs.most_common(30):
        if seq not in correct_set:
            print(f"  {' → '.join(seq):<50} {count:>4} problems")

    print(f"\n{'='*60}")
    print(f"SEQUENCES WE CAN HANDLE (in correct)")
    print(f"{'='*60}")
    for seq, count in correct_seqs.most_common(20):
        print(f"  {' → '.join(seq):<50} {count:>4} problems")

    # Save analysis
    analysis = {
        "total_no_correct": len(no_correct),
        "correct_seqs": {" → ".join(k): v for k, v in correct_seqs.most_common(50)},
        "wrong_seqs": {" → ".join(k): v for k, v in wrong_seqs.most_common(50)},
        "correct_ops": dict(correct_ops),
        "wrong_ops": dict(wrong_ops),
    }
    out_path = RESULTS_DIR / "decomposition_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
