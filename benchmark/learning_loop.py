"""
benchmark/learning_loop.py — Run the mechanical compiler learning loop.

Runs all problems through solve(), retains GraphSnapshots,
evaluates at the equivalence class level, diffs against baseline.

Usage:
    python -m benchmark.learning_loop
    python -m benchmark.learning_loop --diff benchmark/results/baseline_600.json
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SUITES_DIR = REPO_ROOT / "benchmark" / "suites"
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"


def load_all_problems() -> list[dict]:
    """Load all GSM8K problems from the three suites."""
    problems = []
    seen_ids = set()
    for suite_file in ["gsm8k.json", "gsm8k_v2.json", "gsm8k_500.json"]:
        path = SUITES_DIR / suite_file
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for case in data["cases"]:
            if case["id"] not in seen_ids:
                seen_ids.add(case["id"])
                problems.append(case)
    return problems


def run_eval() -> dict:
    """Run all problems, retain snapshots, return results."""
    from packages.core.interpret import solve
    from packages.core.problem_graph import (
        GraphSnapshot, clear_snapshots, retain, diagnose_failure,
        evaluate_run, get_snapshots,
    )
    clear_snapshots()
    problems = load_all_problems()
    print(f"Running {len(problems)} problems...")

    correct = 0
    total = 0
    class_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    failure_counts = defaultdict(int)
    solver_counts = defaultdict(int)

    t0 = time.monotonic()

    for i, prob in enumerate(problems):
        pid = prob["id"]
        question = prob["question"]
        expected_str = prob["answer"]

        try:
            expected = float(expected_str.replace(",", "").strip())
        except (ValueError, AttributeError):
            continue

        try:
            answer, ctx = solve(question)
        except Exception as e:
            answer = None
            # Create minimal ctx
            class _MinCtx:
                problem_graph = None
                walk_result = None
                solver_used = "error"
                equivalence_classes = []
            ctx = _MinCtx()

        is_correct = (answer is not None and abs(answer - expected) < 0.01)

        # Get equivalence classes
        eq_classes = getattr(ctx, 'equivalence_classes', [])
        if not eq_classes:
            eq_classes = ["unclassified"]

        # Build snapshot
        snap = GraphSnapshot(
            problem_id=pid,
            problem_text=question,
            graph=getattr(ctx, 'problem_graph', None),
            walk_result=getattr(ctx, 'walk_result', None),
            answer=answer,
            expected=expected,
            correct=is_correct,
            equivalence_classes=eq_classes,
            solver_used=getattr(ctx, 'solver_used', ''),
        )

        if not is_correct:
            snap.failure_type = diagnose_failure(snap)
            failure_counts[snap.failure_type] += 1

        retain(snap)

        if is_correct:
            correct += 1
        total += 1

        for cls in eq_classes:
            class_counts[cls]["total"] += 1
            if is_correct:
                class_counts[cls]["correct"] += 1

        solver_counts[getattr(ctx, 'solver_used', 'unknown')] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {i+1}/{len(problems)}  correct={correct}/{total}  "
                  f"({elapsed:.1f}s)")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Build class rates (sorted by rate)
    class_rates = {}
    for cls, counts in sorted(class_counts.items(),
                               key=lambda x: x[1]["correct"] / max(x[1]["total"], 1)):
        rate = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        class_rates[cls] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "rate": round(rate, 2),
        }

    # Sort failure types and solver counts by frequency
    failure_types = dict(sorted(failure_counts.items(),
                                 key=lambda x: -x[1]))
    solver_dist = dict(sorted(solver_counts.items(),
                                key=lambda x: -x[1]))

    result = {
        "date": time.strftime("%Y-%m-%d"),
        "total": total,
        "correct": correct,
        "pass_rate": round(correct / total, 3) if total > 0 else 0,
        "class_rates": class_rates,
        "failure_types": failure_types,
        "solver_distribution": solver_dist,
    }

    return result


def diff_results(current: dict, baseline: dict) -> str:
    """Diff current results against baseline at the set level."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"SET-LEVEL DIFF: {baseline['date']} → {current['date']}")
    lines.append(f"{'='*60}")
    lines.append(f"Total: {baseline['correct']}/{baseline['total']} "
                 f"→ {current['correct']}/{current['total']} "
                 f"(Δ {current['correct'] - baseline['correct']:+d})")
    lines.append(f"Pass rate: {baseline['pass_rate']:.1%} → {current['pass_rate']:.1%}")
    lines.append("")

    # Class-level diff
    lines.append(f"{'Class':<25} {'Baseline':>10} {'Current':>10} {'Delta':>8}")
    lines.append(f"{'-'*55}")
    all_classes = set(list(baseline.get("class_rates", {}).keys()) +
                      list(current.get("class_rates", {}).keys()))
    for cls in sorted(all_classes):
        b = baseline.get("class_rates", {}).get(cls, {"correct": 0, "total": 0, "rate": 0})
        c = current.get("class_rates", {}).get(cls, {"correct": 0, "total": 0, "rate": 0})
        delta = c["rate"] - b["rate"]
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        lines.append(f"{cls:<25} {b['correct']:>3}/{b['total']:<4} "
                     f"({b['rate']:.0%})  {c['correct']:>3}/{c['total']:<4} "
                     f"({c['rate']:.0%})  {marker}{delta:+.0%}")

    lines.append("")

    # Failure type diff
    lines.append(f"{'Failure Type':<25} {'Baseline':>10} {'Current':>10} {'Delta':>8}")
    lines.append(f"{'-'*55}")
    all_failures = set(list(baseline.get("failure_types", {}).keys()) +
                       list(current.get("failure_types", {}).keys()))
    for ft in sorted(all_failures, key=lambda x: -current.get("failure_types", {}).get(x, 0)):
        b_count = baseline.get("failure_types", {}).get(ft, 0)
        c_count = current.get("failure_types", {}).get(ft, 0)
        delta = c_count - b_count
        lines.append(f"{ft:<25} {b_count:>10} {c_count:>10} {delta:>+8}")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff", type=str, default=None,
                        help="Path to baseline JSON to diff against")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to this path")
    args = parser.parse_args()

    result = run_eval()

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {result['correct']}/{result['total']} "
          f"({result['pass_rate']:.1%})")
    print(f"{'='*60}")

    print(f"\nClass rates:")
    for cls, data in result["class_rates"].items():
        print(f"  {cls:<25} {data['correct']:>3}/{data['total']:<4} ({data['rate']:.0%})")

    print(f"\nFailure types:")
    for ft, count in result["failure_types"].items():
        print(f"  {ft:<25} {count}")

    print(f"\nSolver distribution:")
    for solver, count in result["solver_distribution"].items():
        print(f"  {solver:<25} {count}")

    # Diff against baseline
    if args.diff:
        baseline_path = Path(args.diff)
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            print(f"\n{diff_results(result, baseline)}")

    # Save
    save_path = args.save or str(RESULTS_DIR / f"eval_{result['date']}.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
