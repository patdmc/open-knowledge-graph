"""
Evaluate multi-interpretation: for each problem, how many solvers
produce the correct answer? Which solver wins? Which SHOULD win?

Key insight: confidence should be in the STRATEGY (did we interpret
the language/variables/operators correctly?) not in the ANSWER.
"""

import json
import time
from pathlib import Path
from collections import defaultdict

SUITES_DIR = Path(__file__).parent / "suites"
RESULTS_DIR = Path(__file__).parent / "results"


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
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from packages.core.interpret import compile, execute

    problems = load_problems()
    print(f"Evaluating {len(problems)} problems with multi-interpretation...\n")

    winner_correct = 0
    any_correct = 0
    none_correct = 0
    total = 0

    # Track: for problems where winner is WRONG but another candidate is RIGHT
    missed_winners = []

    # Track solver accuracy
    solver_wins = defaultdict(lambda: {"picked": 0, "correct_when_picked": 0,
                                        "available": 0, "correct_when_available": 0})

    t0 = time.monotonic()

    for i, p in enumerate(problems):
        try:
            expected = float(p['answer'].replace(',', ''))
        except (ValueError, AttributeError):
            continue

        total += 1
        try:
            ast, ctx = compile(p['question'])
            answer = execute(ast) if ast else None
        except Exception:
            answer = None
            none_correct += 1
            continue

        candidates = getattr(ctx, 'all_candidates', [])
        winner_ok = answer is not None and abs(answer - expected) < 0.01

        if winner_ok:
            winner_correct += 1

        # Check all candidates (may have 3, 4, or 5 fields depending on version)
        correct_solvers = []
        for entry in candidates:
            val, name = entry[0], entry[1]
            pri = entry[2] if len(entry) > 2 else 0
            strat = entry[3] if len(entry) > 3 else 0.0
            blend = entry[4] if len(entry) > 4 else 0.0
            solver_wins[name]["available"] += 1
            if val is not None and abs(val - expected) < 0.01:
                correct_solvers.append((name, pri, val, strat, blend))
                solver_wins[name]["correct_when_available"] += 1

        # Track picked solver
        solver_used = getattr(ctx, 'solver_used', '')
        if solver_used:
            solver_wins[solver_used]["picked"] += 1
            if winner_ok:
                solver_wins[solver_used]["correct_when_picked"] += 1

        if correct_solvers:
            any_correct += 1
            if not winner_ok:
                missed_winners.append({
                    'id': p['id'],
                    'question': p['question'][:80],
                    'expected': expected,
                    'winner': (solver_used, answer),
                    'correct_solvers': [(n, v, s) for n, _, v, s, _ in correct_solvers],
                    'all_candidates': [(e[1], e[0], e[2] if len(e)>2 else 0,
                                        f"s={e[3]:.2f}" if len(e)>3 else "")
                                       for e in candidates],
                })
        else:
            if not winner_ok:
                none_correct += 1

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {i+1}/{len(problems)}  winner={winner_correct}  "
                  f"any={any_correct}  ({elapsed:.1f}s)")

    elapsed = time.monotonic() - t0

    print(f"\n{'='*60}")
    print(f"MULTI-INTERPRETATION RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Total problems:          {total}")
    print(f"Winner correct:          {winner_correct} ({winner_correct/total:.1%})")
    print(f"ANY candidate correct:   {any_correct} ({any_correct/total:.1%})")
    print(f"No candidate correct:    {none_correct} ({none_correct/total:.1%})")
    print(f"Missed (right answer,    {len(missed_winners)} "
          f"wrong winner):")

    print(f"\n{'Solver':<20} {'Picked':>7} {'Correct':>8} {'Pick%':>6}  "
          f"{'Available':>9} {'CorrectA':>9} {'Avail%':>7}")
    print(f"{'-'*72}")
    for solver in sorted(solver_wins.keys(),
                         key=lambda s: -solver_wins[s]["correct_when_available"]):
        s = solver_wins[solver]
        pick_rate = s["correct_when_picked"] / s["picked"] if s["picked"] else 0
        avail_rate = s["correct_when_available"] / s["available"] if s["available"] else 0
        print(f"{solver:<20} {s['picked']:>7} {s['correct_when_picked']:>8} "
              f"{pick_rate:>5.0%}  {s['available']:>9} "
              f"{s['correct_when_available']:>9} {avail_rate:>6.0%}")

    # Show missed winners (right answer available but wrong one picked)
    if missed_winners:
        print(f"\n{'='*60}")
        print(f"MISSED WINNERS: correct answer available but not picked")
        print(f"{'='*60}")
        for m in missed_winners[:20]:
            print(f"\n  {m['id']}: expected={m['expected']}")
            print(f"  Q: {m['question']}...")
            print(f"  Winner: {m['winner'][0]} → {m['winner'][1]}")
            print(f"  Correct: {m['correct_solvers']}")
            print(f"  All: {m['all_candidates']}")

    # Save full results
    out = {
        "total": total,
        "winner_correct": winner_correct,
        "any_correct": any_correct,
        "none_correct": none_correct,
        "missed_count": len(missed_winners),
        "solver_stats": {k: dict(v) for k, v in solver_wins.items()},
        "missed_winners": missed_winners,
    }
    out_path = RESULTS_DIR / "candidate_eval.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
