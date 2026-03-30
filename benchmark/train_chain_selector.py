"""
benchmark/train_chain_selector.py — Learn chain selection from training data.

Instead of perfecting the verb classifier, learn which chain features
predict the correct chain from 136 labeled examples.

Features per chain:
  - operation sequence (what ops does the chain use?)
  - number coverage (what fraction of problem numbers does it use?)
  - rate signal (does the text have "per/each/every" + the chain has multiply?)
  - comparison signal ("more than"/"less than" + chain has add/subtract?)
  - magnitude sanity (is the answer reasonable vs inputs?)
  - clause alignment (do the clause ops match the chain ops?)

This is a tiny logistic regression — no transformer needed.
Constants = nodes. Equations = edges. The weights are the edges.
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict

RESULTS_DIR = Path(__file__).parent / "results"


def extract_chain_features(chain_sig: str, clauses: list[dict],
                           problem_text: str, answer: float,
                           problem_nums: list[float]) -> dict:
    """Extract features from a chain signature + problem context."""
    text_lower = problem_text.lower()

    # Parse chain sig into operation sequence
    ops = re.findall(r'(add|subtract|multiply|divide)', chain_sig)
    op_counter = Counter(ops)

    features = {}

    # 1. Operation counts
    features['n_add'] = op_counter.get('add', 0)
    features['n_subtract'] = op_counter.get('subtract', 0)
    features['n_multiply'] = op_counter.get('multiply', 0)
    features['n_divide'] = op_counter.get('divide', 0)
    features['n_steps'] = len(ops)

    # 2. Rate signal: "per/each/every" in text AND chain has multiply
    has_rate = bool(re.search(r'\b(per|each|every)\b', text_lower))
    features['rate_x_multiply'] = 1.0 if (has_rate and op_counter.get('multiply', 0) > 0) else 0.0
    features['rate_no_multiply'] = 1.0 if (has_rate and op_counter.get('multiply', 0) == 0) else 0.0

    # 3. Comparison signal: "more than"/"less than"/"older"/"younger"
    has_comparison = bool(re.search(
        r'\b(more than|less than|fewer|older|younger|taller|shorter|'
        r'heavier|lighter|longer|shorter|bigger|smaller)\b', text_lower))
    features['comparison_x_addsub'] = 1.0 if (
        has_comparison and (op_counter.get('add', 0) + op_counter.get('subtract', 0)) > 0
    ) else 0.0

    # 4. "times"/"twice"/"triple"/"double" → multiply signal
    has_times = bool(re.search(
        r'\b(times|twice|thrice|triple|double|half)\b', text_lower))
    features['times_x_multiply'] = 1.0 if (
        has_times and op_counter.get('multiply', 0) > 0
    ) else 0.0
    features['times_x_divide'] = 1.0 if (
        has_times and op_counter.get('divide', 0) > 0
    ) else 0.0

    # 5. Fraction/percentage signal → multiply or divide
    has_fraction = bool(re.search(r'\b(\d+%|\d+/\d+)\b', text_lower))
    features['fraction_x_multiply'] = 1.0 if (
        has_fraction and op_counter.get('multiply', 0) > 0
    ) else 0.0
    features['fraction_x_divide'] = 1.0 if (
        has_fraction and op_counter.get('divide', 0) > 0
    ) else 0.0

    # 6. Dollar sign → often multiply (price * quantity)
    has_dollar = '$' in text_lower
    features['dollar_x_multiply'] = 1.0 if (
        has_dollar and op_counter.get('multiply', 0) > 0
    ) else 0.0

    # 7. "split"/"share"/"divide"/"distribute" → divide signal
    has_split = bool(re.search(
        r'\b(split|share|distribute|divided|equally)\b', text_lower))
    features['split_x_divide'] = 1.0 if (
        has_split and op_counter.get('divide', 0) > 0
    ) else 0.0

    # 8. Clause operation alignment
    clause_ops = [c.get('operation', 'none') for c in clauses
                  if c.get('operation', 'none') != 'none']
    clause_op_counter = Counter(clause_ops)
    # How many clause ops agree with chain ops?
    agreement = sum(min(op_counter.get(op, 0), clause_op_counter.get(op, 0))
                    for op in set(list(op_counter.keys()) + list(clause_op_counter.keys())))
    total_ops = max(len(ops), len(clause_ops), 1)
    features['clause_agreement'] = agreement / total_ops

    # 9. Magnitude sanity
    if problem_nums:
        max_input = max(abs(n) for n in problem_nums) if problem_nums else 1
        if max_input > 0:
            ratio = abs(answer) / max_input
            features['magnitude_sane'] = 1.0 if ratio < 100 else 0.0
        else:
            features['magnitude_sane'] = 1.0
    else:
        features['magnitude_sane'] = 1.0

    return features


def main():
    with open(RESULTS_DIR / "chain_training.json") as f:
        training = json.load(f)

    # Also need to run the chain solver to get all candidate chains
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from packages.core.interpret import compile, execute

    print(f"Training data: {len(training)} problems")

    # Collect features for correct and incorrect chains
    feature_names = None
    correct_features = []
    wrong_features = []

    for i, problem in enumerate(training):
        pid = problem['pid']
        answer = problem['answer']
        correct_sig = problem['correct_sig']
        clauses = problem.get('clauses', [])
        sonnet_ops = problem.get('sonnet_ops', [])

        # Reconstruct problem text from clauses
        text = " ".join(c.get('text', '') for c in clauses)
        nums = []
        for c in clauses:
            nums.extend(c.get('nums', []))

        # Features for the correct chain
        correct_feats = extract_chain_features(
            correct_sig, clauses, text, answer, nums)

        if feature_names is None:
            feature_names = sorted(correct_feats.keys())

        correct_features.append(correct_feats)

        # Generate some wrong chain signatures for contrast
        # We know the correct ops — generate chains with wrong ops
        all_ops = ['add', 'subtract', 'multiply', 'divide']
        for wrong_op in all_ops:
            # Replace first op in correct sig with wrong op
            if sonnet_ops:
                wrong_sig_ops = list(sonnet_ops)
                wrong_sig_ops[0] = wrong_op
                if wrong_sig_ops != list(sonnet_ops):
                    wrong_sig = f"{'('.join(wrong_sig_ops)}"  # rough sig
                    wrong_feats = extract_chain_features(
                        " ".join(wrong_sig_ops), clauses, text, answer, nums)
                    wrong_features.append(wrong_feats)

    print(f"Correct examples: {len(correct_features)}")
    print(f"Wrong examples: {len(wrong_features)}")

    # Simple analysis: for each feature, what's the average value
    # for correct vs wrong chains?
    print(f"\n{'Feature':<25} {'Correct':>8} {'Wrong':>8} {'Delta':>8} {'Signal':>8}")
    print("-" * 59)

    feature_signals = {}
    for feat in feature_names:
        correct_vals = [f[feat] for f in correct_features]
        wrong_vals = [f[feat] for f in wrong_features]

        c_mean = sum(correct_vals) / len(correct_vals) if correct_vals else 0
        w_mean = sum(wrong_vals) / len(wrong_vals) if wrong_vals else 0
        delta = c_mean - w_mean
        signal = abs(delta) / max(c_mean + w_mean, 0.01)

        feature_signals[feat] = (c_mean, w_mean, delta, signal)
        print(f"{feat:<25} {c_mean:>8.3f} {w_mean:>8.3f} {delta:>+8.3f} {signal:>8.3f}")

    # Rank features by signal strength
    print(f"\n{'='*60}")
    print(f"FEATURES RANKED BY DISCRIMINATIVE POWER")
    print(f"{'='*60}")
    for feat, (c, w, d, s) in sorted(feature_signals.items(),
                                       key=lambda x: -x[1][3]):
        direction = "correct>wrong" if d > 0 else "wrong>correct"
        print(f"  {feat:<25} signal={s:.3f} ({direction})")

    # Save feature weights (simple: delta as weight, sign as direction)
    weights = {}
    for feat, (c, w, d, s) in feature_signals.items():
        if s > 0.05:  # only keep meaningful signals
            weights[feat] = round(d, 4)

    out_path = RESULTS_DIR / "chain_selector_weights.json"
    with open(out_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"\nSaved {len(weights)} weights to {out_path}")


if __name__ == "__main__":
    main()
