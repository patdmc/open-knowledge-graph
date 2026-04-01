"""
Compare LLM accuracy on original vs nonsense-noun GSM8K problems.

Vanilla only — no graph, no solver. Pure LLM.
The delta between original and nonsense accuracy is the contamination
+ semantic shortcut signal.

Usage:
    python -m benchmark.scripts.nonsense_eval --model haiku
    python -m benchmark.scripts.nonsense_eval --model sonnet
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SUITES_DIR = REPO_ROOT / "benchmark" / "suites"
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"

SYSTEM_PREFIX = (
    "Answer the question concisely. Give ONLY the final numeric answer, "
    "no explanation, no units, just the number.\n\n"
)


def call_claude(prompt: str, model: str) -> tuple[str, int]:
    start = time.monotonic()
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model,
             "--output-format", "text", "--disable-slash-commands"],
            capture_output=True, text=True, timeout=60,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        if result.returncode != 0:
            return f"[ERROR: {result.stderr.strip()[:200]}]", elapsed
        return result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "[ERROR: timeout]", int((time.monotonic() - start) * 1000)
    except FileNotFoundError:
        return "[ERROR: claude CLI not found]", 0


def extract_number(text: str):
    text = text.lower().strip()
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        text = boxed.group(1)
    nums = re.findall(r'-?\$?\d[\d,]*\.?\d*', text)
    if nums:
        n = nums[-1].replace(',', '').lstrip('$')
        try:
            return float(n)
        except ValueError:
            pass
    return None


def score(answer: str, reference: str) -> bool:
    if answer.startswith("[ERROR:"):
        return False
    a_num = extract_number(answer)
    r_num = extract_number(reference)
    if a_num is not None and r_num is not None:
        return abs(a_num - r_num) < 0.01
    return answer.strip() == reference.strip()


def run_suite(suite_path: Path, model: str, label: str):
    with open(suite_path) as f:
        suite = json.load(f)

    cases = suite["cases"]
    results = []
    correct = 0
    total = len(cases)

    for i, case in enumerate(cases):
        prompt = SYSTEM_PREFIX + case["question"]
        answer, latency = call_claude(prompt, model)
        is_correct = score(answer, case["answer"])
        if is_correct:
            correct += 1

        mark = "+" if is_correct else "X"
        ans_short = answer[:30].replace("\n", " ")
        print(f"  [{i+1}/{total}] {label} {case['id']} {mark}  "
              f"ans={ans_short:<20} ref={case['answer']:<10} {latency}ms")

        results.append({
            "id": case["id"],
            "answer": answer,
            "reference": case["answer"],
            "correct": is_correct,
            "latency_ms": latency,
        })

    accuracy = correct / total if total else 0
    print(f"\n  {label}: {correct}/{total} = {accuracy:.1%}\n")
    return {"label": label, "accuracy": accuracy, "correct": correct,
            "total": total, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--suite", default="gsm8k",
                        help="Base suite name (will also run _nonsense version)")
    args = parser.parse_args()

    original_path = SUITES_DIR / f"{args.suite}.json"
    nonsense_path = SUITES_DIR / f"{args.suite}_nonsense.json"

    if not nonsense_path.exists():
        print(f"Nonsense suite not found: {nonsense_path}")
        print("Run: python benchmark/scripts/nonsense_nouns.py "
              f"benchmark/suites/{args.suite}.json")
        sys.exit(1)

    print(f"=== Nonsense Noun Contamination Test ===")
    print(f"Model: {args.model}")
    print(f"Suite: {args.suite} ({original_path.stat().st_size//1024}KB)")
    print()

    print("--- Original problems ---")
    original = run_suite(original_path, args.model, "ORIGINAL")

    print("--- Nonsense-noun problems ---")
    nonsense = run_suite(nonsense_path, args.model, "NONSENSE")

    delta = original["accuracy"] - nonsense["accuracy"]
    print("=" * 60)
    print(f"ORIGINAL accuracy:  {original['accuracy']:.1%} "
          f"({original['correct']}/{original['total']})")
    print(f"NONSENSE accuracy:  {nonsense['accuracy']:.1%} "
          f"({nonsense['correct']}/{nonsense['total']})")
    print(f"DELTA (cheat signal): {delta:+.1%}")
    print()

    if delta > 0.05:
        print(f">>> {args.model} scores {delta:.0%} worse on nonsense nouns.")
        print("    This is the contamination + semantic shortcut signal.")
    elif delta > 0.01:
        print(f">>> Small delta ({delta:.1%}). Marginal semantic dependence.")
    else:
        print(f">>> No meaningful delta. {args.model} appears to reason structurally.")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "model": args.model,
        "suite": args.suite,
        "original": original,
        "nonsense": nonsense,
        "delta": delta,
    }
    out_path = RESULTS_DIR / f"nonsense_eval_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
