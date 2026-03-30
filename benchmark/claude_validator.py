"""
benchmark/claude_validator.py — Validate Claude answers against chain solver.

Architecture:
  1. Claude proposes an answer (fast oracle, ~95% accurate)
  2. Chain solver generates all valid computation graphs (exhaustive)
  3. Validator checks: does Claude's answer exist in our chain set?
     - YES → mechanically verified, accept with confidence
     - NO → either Claude hallucinated OR we're missing a chain

This is a proof checker. Claude provides the hypothesis,
the chain solver provides the proof.

Constants = nodes. Equations = edges.
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).parent / "results"
SUITES_DIR = Path(__file__).parent / "suites"


def load_problems():
    """Load all unique problems from test suites."""
    problems = []
    seen = set()
    for suite in ['gsm8k.json', 'gsm8k_v2.json', 'gsm8k_500.json']:
        path = SUITES_DIR / suite
        if path.exists():
            with open(path) as f:
                for c in json.load(f)['cases']:
                    if c['id'] not in seen:
                        seen.add(c['id'])
                        problems.append(c)
    return problems


def validate_answer(answer: float, candidates: list,
                    tolerance: float = 0.01) -> dict:
    """
    Check if a proposed answer exists in the candidate set.
    Returns validation result with chain details if found.
    """
    if answer is None:
        return {"verified": False, "reason": "null_answer"}

    matching_chains = []
    for entry in candidates:
        val = entry[0]
        if val is not None and abs(val - answer) < tolerance:
            matching_chains.append({
                "value": val,
                "solver": entry[1] if len(entry) > 1 else "unknown",
                "priority": entry[2] if len(entry) > 2 else 0,
            })

    if matching_chains:
        return {
            "verified": True,
            "n_matching_chains": len(matching_chains),
            "chains": matching_chains[:5],  # top 5
            "reason": "chain_proof_exists",
        }
    else:
        return {
            "verified": False,
            "reason": "no_chain_proof",
            "n_candidates": len(candidates),
        }


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from packages.core.interpret import compile, execute

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()

    problems = load_problems()[:args.limit]
    print(f"Validating {len(problems)} problems...\n")

    t0 = time.monotonic()
    results = {
        "verified_correct": 0,      # answer key answer found in chains
        "verified_wrong": 0,        # wrong answer verified (bug!)
        "unverified_correct": 0,    # correct answer but no chain proof
        "unverified_wrong": 0,      # wrong answer, no chain
        "total": 0,
    }

    rank_when_verified = Counter()

    for i, p in enumerate(problems):
        try:
            expected = float(p['answer'].replace(',', ''))
        except (ValueError, AttributeError):
            continue

        try:
            ast, ctx = compile(p['question'])
            our_answer = execute(ast) if ast else None
        except Exception:
            our_answer = None

        cands = getattr(ctx, 'all_candidates', [])
        results["total"] += 1

        # Validate the answer-key answer against our chains
        validation = validate_answer(expected, cands)

        if validation["verified"]:
            results["verified_correct"] += 1
            # What rank is it?
            for rank, entry in enumerate(cands, 1):
                val = entry[0]
                if val is not None and abs(val - expected) < 0.01:
                    rank_when_verified[rank] += 1
                    break
        else:
            results["unverified_correct"] += 1

        # Also check: if our winner is wrong, is it still "verified"?
        if our_answer is not None and abs(our_answer - expected) > 0.01:
            wrong_validation = validate_answer(our_answer, cands)
            if wrong_validation["verified"]:
                results["verified_wrong"] += 1

    elapsed = time.monotonic() - t0

    print(f"Results ({elapsed:.1f}s):")
    print(f"  Answer-key answer has chain proof: "
          f"{results['verified_correct']}/{results['total']} "
          f"({results['verified_correct']/results['total']:.1%})")
    print(f"  Answer-key answer has NO chain proof: "
          f"{results['unverified_correct']}/{results['total']} "
          f"({results['unverified_correct']/results['total']:.1%})")
    print(f"  Wrong answers also verified (noise): "
          f"{results['verified_wrong']}")

    # Rank distribution when verified
    print(f"\n  Rank of correct answer among candidates:")
    cumulative = 0
    total_v = sum(rank_when_verified.values())
    for rank in sorted(rank_when_verified.keys())[:10]:
        cumulative += rank_when_verified[rank]
        print(f"    Rank {rank:>3}: {rank_when_verified[rank]:>4} "
              f"(cumulative: {cumulative}/{total_v} = {cumulative/total_v:.1%})")

    remaining = total_v - cumulative
    if remaining > 0:
        print(f"    Rank >10: {remaining:>4}")

    # Save
    out_path = RESULTS_DIR / "claude_validation.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
