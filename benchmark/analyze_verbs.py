"""
benchmark/analyze_verbs.py — Analyze verb→operation misclassifications
using the chain training data.

For each clause in the training data, we know:
  1. What operation the classifier assigned (clause.operation)
  2. What operation the correct chain actually uses (from sonnet_ops)

This reveals which verbs are noise (setup) and which are misclassified.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    with open(RESULTS_DIR / "chain_training.json") as f:
        training = json.load(f)

    # For each problem, compare clause operations vs sonnet operations
    verb_actual = defaultdict(Counter)  # verb → {actual_op: count}
    verb_classified = defaultdict(Counter)  # verb → {classified_op: count}
    verb_examples = defaultdict(list)  # verb → [example texts]

    op_classified = Counter()  # what classifier says
    op_needed = Counter()  # what correct chain needs

    setup_verbs = Counter()  # verbs on clauses with no numbers (pure setup)
    has_verbs = Counter()  # "has/have" specifically

    mismatches = []

    for problem in training:
        sonnet_ops = problem.get("sonnet_ops", [])
        clauses = problem.get("clauses", [])

        # Count what classifier says
        for clause in clauses:
            op = clause.get("operation", "none")
            verb = clause.get("verb", "none")
            nums = clause.get("nums", [])
            text = clause.get("text", "")

            if op != "none":
                op_classified[op] += 1
                verb_classified[verb][op] += 1

            # Track verbs on clauses with exactly 1 number (likely setup)
            if len(nums) <= 1 and op != "none":
                verb_key = verb.split(":")[-1] if ":" in verb else verb
                if verb_key in ("has", "have", "hires", "getting", "make",
                                "earned", "pay", "bumps", "joins"):
                    has_verbs[f"{verb_key} → {op}"] += 1
                    verb_examples[verb_key].append(text[:80])

        # Count what correct chain needs
        for sop in sonnet_ops:
            op_needed[sop] += 1

    # Print analysis
    print("=" * 60)
    print("CLASSIFIER vs CORRECT CHAIN OPERATIONS")
    print("=" * 60)
    print(f"\n{'Operation':<12} {'Classified':>10} {'Needed':>10} {'Ratio':>10}")
    print("-" * 44)
    all_ops = set(list(op_classified.keys()) + list(op_needed.keys()))
    for op in sorted(all_ops):
        c = op_classified.get(op, 0)
        n = op_needed.get(op, 0)
        ratio = f"{c/n:.1%}" if n > 0 else "N/A"
        print(f"{op:<12} {c:>10} {n:>10} {ratio:>10}")

    print("\n" + "=" * 60)
    print("SUSPECT VERBS (single-number clauses classified as operations)")
    print("=" * 60)
    for verb_op, count in has_verbs.most_common(20):
        print(f"  {verb_op:<30} {count:>4}")
        verb_key = verb_op.split(" → ")[0]
        for ex in verb_examples[verb_key][:3]:
            print(f"    '{ex}'")

    # Now analyze per-verb: what does the classifier say vs what's correct?
    print("\n" + "=" * 60)
    print("PER-VERB CLASSIFICATION ACCURACY")
    print("=" * 60)

    # Group by verb stem
    verb_stem_ops = defaultdict(Counter)
    for problem in training:
        for clause in problem.get("clauses", []):
            verb = clause.get("verb", "none")
            op = clause.get("operation", "none")
            if op == "none":
                continue
            # Extract stem
            stem = verb.split(":")[-1] if ":" in verb else verb
            verb_stem_ops[stem][op] += 1

    print(f"\n{'Verb Stem':<20} {'→ add':>6} {'→ sub':>6} {'→ mul':>6} {'→ div':>6}")
    print("-" * 46)
    for stem, ops in sorted(verb_stem_ops.items(),
                             key=lambda x: -sum(x[1].values())):
        if sum(ops.values()) < 2:
            continue
        a = ops.get("add", 0)
        s = ops.get("subtract", 0)
        m = ops.get("multiply", 0)
        d = ops.get("divide", 0)
        print(f"{stem:<20} {a:>6} {s:>6} {m:>6} {d:>6}")

    # Key question: which verbs map to "add" but should be "setup"?
    print("\n" + "=" * 60)
    print("VERBS THAT MAP TO 'add' — SETUP OR REAL OPERATION?")
    print("=" * 60)

    for problem in training:
        clauses = problem.get("clauses", [])
        for clause in clauses:
            verb = clause.get("verb", "none")
            op = clause.get("operation", "none")
            text = clause.get("text", "")
            nums = clause.get("nums", [])

            if op == "add":
                stem = verb.split(":")[-1] if ":" in verb else verb
                # Is this a "state" verb or an "action" verb?
                is_state = any(w in text.lower() for w in
                             ["has ", "have ", "there are", "there is",
                              "weighs", "costs", "is ", "was ", "were "])
                if is_state:
                    role = clause.get("role", "?")
                    print(f"  STATE-as-add: [{role}] verb={stem}")
                    print(f"    '{text[:80]}'")
                    print(f"    nums={nums}")

    # Analysis: what fraction of "add" classifications are actually setup?
    print("\n" + "=" * 60)
    print("SETUP VS OPERATION ANALYSIS")
    print("=" * 60)

    state_as_add = 0
    action_as_add = 0

    state_patterns = ["has ", "have ", "there are", "there is",
                      "weighs", "costs ", "is ", "was ", "were ",
                      "contains", "holds"]

    for problem in training:
        for clause in problem.get("clauses", []):
            op = clause.get("operation", "none")
            text = clause.get("text", "").lower()

            if op == "add":
                is_state = any(p in text for p in state_patterns)
                if is_state:
                    state_as_add += 1
                else:
                    action_as_add += 1

    total_add = state_as_add + action_as_add
    print(f"  Total 'add' classifications: {total_add}")
    print(f"  State verbs (should be setup): {state_as_add} ({state_as_add/total_add:.0%})")
    print(f"  Action verbs (real operations): {action_as_add} ({action_as_add/total_add:.0%})")


if __name__ == "__main__":
    main()
