"""
benchmark/teacher.py — LLM-as-teacher for the learning loop.

When the compiler gets a wrong answer, the LLM sees:
  - The problem text
  - Our parsed clauses (what we extracted)
  - Our logic (solver, graph nodes, walk path, AST)
  - Our answer vs the correct answer

The LLM produces structured corrections:
  - verb_map: "verb X should map to operation Y"
  - unit_fix: "phrase X should extract unit Y"
  - missing_value: "clause X contains value Y we missed"
  - operation_fix: "we multiplied but should have divided"
  - step_sequence: "the correct computation is A op B op C = D"

These corrections feed into the permanent graphs:
  - verb_map → language graph (learned_verb_ops.json)
  - unit_fix → unit extraction rules
  - step_sequence → equivalence class patterns

The LLM burns tokens ONCE per failure pattern.
The fix applies to the ENTIRE equivalence class forever.

Usage:
    from benchmark.teacher import teach_from_wrong
    corrections = teach_from_wrong(problem, our_answer, expected, ctx)
"""

import json
import subprocess
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class Correction:
    """A structured correction from the LLM teacher."""
    type: str           # verb_map, unit_fix, missing_value, operation_fix, step_sequence
    description: str    # human-readable
    details: dict = field(default_factory=dict)  # structured data for graph update


def build_teacher_prompt(problem_text: str, our_answer, expected,
                          clauses_debug: list[str], graph_debug: list[str],
                          solver_used: str, equivalence_classes: list[str]) -> str:
    """Build the prompt for the LLM teacher."""

    prompt = f"""You are analyzing why a mechanical word problem solver got the wrong answer.
Your job is to decompose BOTH solutions into steps and find where they diverge.

PROBLEM:
{problem_text}

OUR ANSWER: {our_answer}
CORRECT ANSWER: {expected}

OUR PROCESS:
- Solver used: {solver_used}
- Equivalence classes: {equivalence_classes}
- Debug trace:
{chr(10).join('  ' + d for d in (clauses_debug + graph_debug)[:20])}

IMPORTANT: Do NOT judge by how close the numbers are. A wildly off answer could be
one step from correct (missing a final division). A near-miss could be completely
wrong logic that happened to land close. Decompose the STEPS.

Respond with EXACTLY this JSON format:

{{
  "correct_steps": ["step 1: ...", "step 2: ...", "step 3: ..."],
  "our_steps": ["step 1: ...", "step 2: ..."],
  "divergence_point": "step N: description of where our logic diverged",
  "steps_correct": 0,
  "steps_total": 0,
  "corrections": [
    {{
      "type": "verb_map|unit_fix|missing_value|operation_fix|step_sequence",
      "key": "the word or phrase that triggered the error",
      "fix": "what it should map to or what we should extract",
      "reason": "why this fixes the class of problems, not just this one"
    }}
  ]
}}

Rules:
- Decompose the correct solution into atomic steps
- Decompose OUR solution into what steps the solver actually performed
- Identify the FIRST point of divergence — that's the bug
- Count how many steps we got right vs total needed
- Each correction should fix a CLASS of problems, not just this one
- "verb_map": a word maps to the wrong operation
- "unit_fix": we extracted the wrong unit from text
- "missing_value": we didn't extract a number or entity
- "operation_fix": wrong arithmetic direction
- "step_sequence": wrong order or missing chain step
- Return ONLY the JSON, no other text."""

    return prompt


def call_llm(prompt: str) -> str:
    """Call Claude CLI for teacher feedback."""
    import sys
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"      LLM error: {result.stderr[:200]}", file=sys.stderr)
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"      LLM timeout", file=sys.stderr)
        return ""
    except FileNotFoundError:
        print(f"      claude CLI not found", file=sys.stderr)
        return ""


def parse_teacher_response(response: str) -> dict:
    """Parse the structured JSON from the LLM teacher."""
    # Find JSON in response
    try:
        # Try direct parse first
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    m = re.search(r'\{[^{}]*"corrections"[^{}]*\[.*?\]\s*\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def teach_from_wrong(problem_text: str, our_answer, expected,
                      ctx, problem_id: str = "") -> list[Correction]:
    """
    Ask the LLM teacher to diagnose a wrong answer.

    Returns structured corrections that can feed back into the permanent graphs.
    """
    clauses_debug = getattr(ctx, 'debug', [])
    graph_debug = []
    if hasattr(ctx, 'problem_graph') and ctx.problem_graph:
        graph_debug = ctx.problem_graph.debug

    prompt = build_teacher_prompt(
        problem_text, our_answer, expected,
        clauses_debug, graph_debug,
        getattr(ctx, 'solver_used', ''),
        getattr(ctx, 'equivalence_classes', []),
    )

    response = call_llm(prompt)
    if not response:
        return []

    parsed = parse_teacher_response(response)
    if not parsed:
        return []

    corrections = []
    for c in parsed.get("corrections", []):
        corrections.append(Correction(
            type=c.get("type", "unknown"),
            description=c.get("reason", ""),
            details={
                "key": c.get("key", ""),
                "fix": c.get("fix", ""),
                "problem_id": problem_id,
                "our_error": parsed.get("our_error", ""),
                "correct_steps": parsed.get("correct_steps", []),
            },
        ))

    return corrections


def teach_batch(records: list[dict], limit: int = 10) -> list[dict]:
    """
    Run the LLM teacher on a batch of wrong answers.

    Prioritizes by equivalence class — teaches ONE problem per class,
    since the fix should apply to the whole class.

    Returns list of {problem_id, corrections, classes}.
    """
    from packages.core.interpret import compile, execute

    # Group wrong answers by class, pick one representative per class
    by_class = {}
    for r in records:
        if r['correct']:
            continue
        for cls in r['classes']:
            if cls not in by_class:
                by_class[cls] = r

    # Teach one per class, up to limit
    results = []
    taught = 0
    for cls, rep in sorted(by_class.items()):
        if taught >= limit:
            break

        pid = rep['id']
        print(f"  Teaching {cls} via {pid}...")

        # Re-compile to get full context
        ast, ctx = compile(rep.get('question', ''))

        # If we don't have the question in the record, skip
        # (we'd need to load from the suite)
        if not hasattr(ctx, 'debug'):
            continue

        corrections = teach_from_wrong(
            rep.get('question', ''),
            rep['answer'], rep['expected'],
            ctx, pid,
        )

        if corrections:
            results.append({
                'problem_id': pid,
                'class': cls,
                'corrections': [
                    {'type': c.type, 'description': c.description,
                     'details': c.details}
                    for c in corrections
                ],
            })
            taught += 1
            print(f"    → {len(corrections)} corrections")
        else:
            print(f"    → no corrections (LLM failed)")

    return results


def teach_from_decomposed(decomposed_path: str = None, limit: int = 10):
    """
    Load decomposed results and teach on wrong answers.

    This is the entry point for the direct learning mode.
    """
    if decomposed_path is None:
        decomposed_path = str(RESULTS_DIR / "decomposed.json")

    with open(decomposed_path) as f:
        records = json.load(f)

    # We need the question text — load from suites
    suites_dir = Path(__file__).parent / "suites"
    question_map = {}
    for suite in ['gsm8k.json', 'gsm8k_v2.json', 'gsm8k_500.json']:
        path = suites_dir / suite
        if path.exists():
            with open(path) as f:
                for c in json.load(f)['cases']:
                    question_map[c['id']] = c['question']

    # Attach questions to records
    for r in records:
        r['question'] = question_map.get(r['id'], '')

    wrong = [r for r in records if not r['correct'] and r['question']]
    print(f"Teaching on {min(limit, len(wrong))} wrong answers "
          f"(from {len(wrong)} total wrong)...")

    results = teach_batch(wrong, limit=limit)

    # Save
    out_path = RESULTS_DIR / "teacher_corrections.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} correction sets to {out_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10,
                        help="Max problems to teach on")
    parser.add_argument("--decomposed", type=str, default=None)
    args = parser.parse_args()

    results = teach_from_decomposed(args.decomposed, args.limit)

    # Print summary
    all_types = {}
    for r in results:
        for c in r['corrections']:
            t = c['type']
            all_types[t] = all_types.get(t, 0) + 1

    print(f"\nCorrection types: {all_types}")


if __name__ == "__main__":
    main()
