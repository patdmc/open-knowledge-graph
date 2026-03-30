"""
benchmark/learn.py — One learning phase of the determinism ratchet.

Architecture:
  1. Student runs all problems mechanically (0 LLM tokens)
  2. Validator scores parse coverage on all problems (0 LLM tokens)
  3. Teacher diagnoses high-coverage wrong answers (LLM tokens, batch)
  4. All corrections accumulated — not applied yet
  5. Corrections applied to the graph ALL AT ONCE
  6. Student re-runs all problems
  7. Set-level diff against pre-fix run

One learning phase is holistic: we re-evaluate the whole graph
after ALL fixes, not one fix at a time.

Usage:
    python3 -m benchmark.learn --phase direct --teach-limit 20
    python3 -m benchmark.learn --phase test    # no teacher, just calibrate
    python3 -m benchmark.learn --phase blind   # fresh 200 problems
"""

import json
import time
from collections import defaultdict
from pathlib import Path

SUITES_DIR = Path(__file__).parent / "suites"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_problems(suites=None):
    """Load problems from suite files."""
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


def run_student(problems):
    """Student runs all problems mechanically. Returns records."""
    from packages.core.interpret import compile, execute, parse
    from packages.core.validate_parse import validate_parse

    records = []
    t0 = time.monotonic()

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
            class _C:
                problem_graph = None
                walk_result = None
                solver_used = 'error'
                equivalence_classes = []
                sentences = []
                debug = []
            ctx = _C()

        ok = answer is not None and abs(answer - expected) < 0.01

        # Validator checks the student's work
        try:
            clauses, _ = parse(p['question'])
            v = validate_parse(p['question'], clauses, ctx)
            coverage = v.coverage
            issues = v.issues
        except Exception:
            coverage = 0.0
            issues = ['validation failed']

        records.append({
            'id': p['id'],
            'question': p['question'],
            'answer': answer,
            'expected': expected,
            'correct': ok,
            'solver': getattr(ctx, 'solver_used', ''),
            'classes': getattr(ctx, 'equivalence_classes', []) or ['unclassified'],
            'coverage': coverage,
            'issues': issues,
            'debug': getattr(ctx, 'debug', [])[:10],
        })

        if (i + 1) % 100 == 0:
            c = sum(1 for r in records if r['correct'])
            elapsed = time.monotonic() - t0
            print(f"  {i+1}/{len(problems)}  correct={c}/{i+1}  ({elapsed:.1f}s)")

    elapsed = time.monotonic() - t0
    correct = sum(1 for r in records if r['correct'])
    print(f"Student: {correct}/{len(records)} ({correct/len(records):.1%}) in {elapsed:.1f}s")
    return records


def summarize(records):
    """Produce set-level summary from records."""
    class_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    solver_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        for cls in r['classes']:
            class_counts[cls]["total"] += 1
            if r['correct']:
                class_counts[cls]["correct"] += 1
        solver_counts[r['solver']]["total"] += 1
        if r['correct']:
            solver_counts[r['solver']]["correct"] += 1

    class_rates = {}
    for cls, counts in sorted(class_counts.items(),
                               key=lambda x: x[1]["correct"] / max(x[1]["total"], 1)):
        rate = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        class_rates[cls] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "rate": round(rate, 3),
        }

    correct = sum(1 for r in records if r['correct'])
    return {
        "date": time.strftime("%Y-%m-%d"),
        "total": len(records),
        "correct": correct,
        "pass_rate": round(correct / len(records), 3) if records else 0,
        "class_rates": class_rates,
    }


def diff_summaries(before, after):
    """Set-level diff between two summaries."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"SET-LEVEL DIFF: before → after")
    lines.append(f"{'='*60}")
    lines.append(f"Total: {before['correct']}/{before['total']} "
                 f"→ {after['correct']}/{after['total']} "
                 f"(Δ {after['correct'] - before['correct']:+d})")
    lines.append("")

    lines.append(f"{'Class':<25} {'Before':>10} {'After':>10} {'Delta':>8}")
    lines.append(f"{'-'*55}")
    all_classes = set(list(before.get("class_rates", {}).keys()) +
                      list(after.get("class_rates", {}).keys()))
    for cls in sorted(all_classes):
        b = before.get("class_rates", {}).get(cls, {"correct": 0, "total": 0, "rate": 0})
        a = after.get("class_rates", {}).get(cls, {"correct": 0, "total": 0, "rate": 0})
        delta = a["rate"] - b["rate"]
        marker = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else " ")
        lines.append(f"{cls:<25} {b['correct']:>3}/{b['total']:<4} "
                     f"({b['rate']:.0%})  {a['correct']:>3}/{a['total']:<4} "
                     f"({a['rate']:.0%})  {marker}{delta:+.0%}")

    return "\n".join(lines)


def teach_batch(records, limit=20):
    """
    Teacher diagnoses high-coverage wrong answers.
    Returns accumulated corrections (not yet applied).
    """
    from benchmark.teacher import teach_from_wrong
    from packages.core.interpret import compile

    # Filter: high coverage, wrong answer
    targets = [r for r in records
                if not r['correct'] and r['coverage'] >= 0.9]

    # One representative per equivalence class.
    # NOT sorted by answer ratio — closeness of the number tells you
    # nothing about closeness of the process. A wildly off answer could
    # be one step from correct. A near-miss could be completely wrong logic.
    by_class = {}
    for r in targets:
        for cls in r['classes']:
            if cls not in by_class:
                by_class[cls] = r

    ordered = [(cls, (rep, 0)) for cls, rep in by_class.items()]

    all_corrections = []
    taught = 0

    for cls, (rep, closeness) in ordered:
        if taught >= limit:
            break

        pid = rep['id']
        print(f"  Teaching {cls} via {pid}...")

        # Re-compile for full context
        try:
            ast, ctx = compile(rep['question'])
        except Exception:
            continue

        corrections = teach_from_wrong(
            rep['question'], rep['answer'], rep['expected'],
            ctx, rep['id'],
        )

        for c in corrections:
            all_corrections.append({
                'class': cls,
                'problem_id': rep['id'],
                'type': c.type,
                'description': c.description,
                'details': c.details,
            })

        taught += 1
        print(f"    → {len(corrections)} corrections")

    print(f"\nTeacher produced {len(all_corrections)} corrections "
          f"from {taught} problems")
    return all_corrections


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["direct", "test", "blind"],
                        default="direct")
    parser.add_argument("--teach-limit", type=int, default=20)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if args.phase == "blind":
        # Fresh problems not in training set
        problems = load_problems(['gsm8k_blind.json'])
        if not problems:
            print("No blind suite found. Create benchmark/suites/gsm8k_blind.json first.")
            return
    else:
        problems = load_problems()

    # --- Step 1: Student runs all problems ---
    print(f"\n{'='*60}")
    print(f"PHASE: {args.phase.upper()}")
    print(f"{'='*60}")
    print(f"\nStep 1: Student solving {len(problems)} problems...")
    records = run_student(problems)
    before = summarize(records)

    # Print class rates
    print(f"\nClass rates:")
    for cls, data in before['class_rates'].items():
        print(f"  {cls:<25} {data['correct']:>3}/{data['total']:<4} ({data['rate']:.0%})")

    if args.phase == "test":
        # Test mode: just calibrate confidence, no teaching
        print(f"\nTest mode: calibrating confidence...")
        conf_buckets = defaultdict(lambda: [0, 0])
        for r in records:
            bucket = round(r['coverage'], 1)
            conf_buckets[bucket][0] += 1
            if r['correct']:
                conf_buckets[bucket][1] += 1
        print(f"\n{'Coverage':>8} {'Total':>6} {'Correct':>8} {'Rate':>6}")
        for cov in sorted(conf_buckets.keys()):
            total, correct = conf_buckets[cov]
            rate = correct / total if total else 0
            print(f"{cov:>8.1f} {total:>6} {correct:>8} {rate:>6.1%}")
        return

    if args.phase == "blind":
        # Blind mode: just measure, no teaching
        print(f"\nBlind evaluation complete. No teaching.")
        return

    # --- Step 2: Teacher diagnoses (direct mode only) ---
    print(f"\nStep 2: Teacher diagnosing wrong answers...")
    corrections = teach_batch(records, limit=args.teach_limit)

    # Save corrections
    corrections_path = RESULTS_DIR / f"corrections_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(corrections_path, 'w') as f:
        json.dump(corrections, f, indent=2)
    print(f"Corrections saved: {corrections_path}")

    # --- Step 3: Apply corrections to the graph ---
    # For now, corrections are accumulated but need manual review
    # before being applied to permanent graphs.
    # TODO: auto-apply verb_map corrections to learned_verb_ops.json
    # TODO: auto-apply unit_fix corrections to unit extraction rules
    print(f"\nStep 3: {len(corrections)} corrections accumulated.")
    print(f"  Types: ", end="")
    from collections import Counter
    types = Counter(c['type'] for c in corrections)
    print(dict(types))

    # --- Step 4: Re-run student (after corrections applied) ---
    # Since corrections aren't auto-applied yet, this step is manual.
    # When auto-apply is implemented, uncomment:
    # print(f"\nStep 4: Re-running student after fixes...")
    # records_after = run_student(problems)
    # after = summarize(records_after)
    # print(f"\n{diff_summaries(before, after)}")

    # Save summary
    save_path = args.save or str(RESULTS_DIR / f"phase_{args.phase}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump({
            'phase': args.phase,
            'summary': before,
            'n_corrections': len(corrections),
            'correction_types': dict(types),
        }, f, indent=2)
    print(f"\nPhase saved: {save_path}")


if __name__ == "__main__":
    main()
