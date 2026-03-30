"""
benchmark/harness.py — AI test harness for knowledge-graph augmented inference.

Compares two conditions:
  A) Claude alone (vanilla prompt)
  B) Claude + language graph (graph-retrieved context prepended)

On standardized test suites. Measures:
  - Accuracy (exact match or judge-scored)
  - Input tokens (estimated from prompt length)
  - Output tokens (estimated from response length)
  - Latency (wall clock)

Uses Claude CLI (subprocess) — no API key or separate billing needed.
Future: migrate to claude_agent_sdk for programmatic access.

Usage:
    python -m benchmark.harness --suite disambiguation --model haiku
    python -m benchmark.harness --suite all --model sonnet --runs 3
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
SUITES_DIR = REPO_ROOT / "benchmark" / "suites"

MODEL_ALIASES = {
    "haiku":  "haiku",
    "sonnet": "sonnet",
    "opus":   "opus",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single test question."""
    id: str
    suite: str
    question: str
    reference_answer: str
    category: str = ""
    difficulty: str = ""            # easy, medium, hard
    graph_context: str = ""         # pre-computed graph retrieval for condition B
    metadata: dict = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of one model call on one test case."""
    test_id: str
    condition: str                  # "vanilla" or "graph"
    model: str
    answer: str
    correct: Optional[bool] = None
    input_chars: int = 0            # prompt character count
    output_chars: int = 0           # response character count
    input_tokens_est: int = 0       # estimated tokens (~4 chars/token)
    output_tokens_est: int = 0
    latency_ms: int = 0
    error: str = ""


@dataclass
class SuiteResult:
    """Aggregate results for one suite."""
    suite: str
    model: str
    n_cases: int = 0
    vanilla_accuracy: float = 0.0
    graph_accuracy: float = 0.0
    vanilla_mean_input_tokens: float = 0.0
    graph_mean_input_tokens: float = 0.0
    vanilla_mean_output_tokens: float = 0.0
    graph_mean_output_tokens: float = 0.0
    vanilla_mean_latency_ms: float = 0.0
    graph_mean_latency_ms: float = 0.0
    token_savings_ratio: float = 0.0
    accuracy_delta: float = 0.0
    runs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph context retrieval
# ---------------------------------------------------------------------------

def retrieve_graph_context(question: str, graph_backend: str = "full") -> str:
    """
    Query the knowledge graph for context relevant to the question.
    Returns a context string to prepend to the prompt.

    Backends:
      "full"      — language graph + math graph + structure extraction
      "language"   — language graph only (WordNet + homophones)
      "inference"  — embedding-based retrieval
    """
    if graph_backend == "full":
        return _retrieve_full_graph(question)
    elif graph_backend == "language":
        return _retrieve_language_graph(question)
    elif graph_backend == "inference":
        return _retrieve_inference_pipeline(question)
    else:
        return ""


def _retrieve_full_graph(question: str) -> str:
    """
    Full knowledge graph retrieval: language + math + structure.

    For math word problems, this:
    1. Extracts quantities and units (structured representation)
    2. Identifies operation type (rate, proportion, remainder, etc.)
    3. Provides relevant mathematical relationships from the graph
    4. Runs language disambiguation on ambiguous terms
    """
    parts = []

    # --- Layer 1: Structure extraction (problem decomposition) ---
    parts.extend(_extract_math_structure(question))

    # --- Layer 2: Language graph (disambiguation) ---
    lang_ctx = _retrieve_language_graph(question)
    if lang_ctx:
        # Strip the wrapper, we'll add our own
        for line in lang_ctx.split("\n"):
            if line.startswith("["):
                parts.append(line)

    # --- Layer 3: Math graph (relevant definitions/formulas) ---
    parts.extend(_retrieve_math_graph(question))

    if not parts:
        return ""

    return "GRAPH CONTEXT:\n" + "\n".join(parts) + "\n---\n"


def _extract_math_structure(question: str) -> list[str]:
    """
    Extract quantities, units, and relationships from a word problem.
    This is the graph doing what a compiler front-end does: parsing
    the unstructured text into structured representation.
    """
    import re
    parts = []

    # Extract all numbers with surrounding context
    quantities = []
    for m in re.finditer(
        r'(\$?\d[\d,]*\.?\d*)\s*(%|percent|per\s+\w+|each|every|'
        r'dollars?|cents?|hours?|minutes?|days?|weeks?|months?|years?|'
        r'miles?|km|feet|inches?|pounds?|kg|gallons?|liters?|'
        r'times?|pieces?|pairs?|sets?|groups?|boxes?|bags?|'
        r'dozen|score)?',
        question, re.IGNORECASE
    ):
        num_str = m.group(1).replace(',', '').lstrip('$')
        unit = m.group(2) or ""
        is_dollar = m.group(1).startswith('$')
        if is_dollar:
            unit = "dollars"
        try:
            num = float(num_str)
            quantities.append((num, unit.strip().lower()))
        except ValueError:
            pass

    if quantities:
        q_strs = [f"{n:g} {u}".strip() for n, u in quantities]
        parts.append(f"[QUANTITIES] {'; '.join(q_strs)}")

    # Identify operation patterns
    q_lower = question.lower()
    ops = []
    if any(w in q_lower for w in ["per ", "each ", "every ", "rate"]):
        ops.append("rate/unit-conversion")
    if any(w in q_lower for w in ["remain", "left over", "surplus", "difference"]):
        ops.append("subtraction/remainder")
    if any(w in q_lower for w in ["total", "altogether", "combined", "sum"]):
        ops.append("addition/total")
    if any(w in q_lower for w in ["times", "twice", "triple", "double", "multiply"]):
        ops.append("multiplication")
    if any(w in q_lower for w in ["split", "divide", "share", "distribute", "half"]):
        ops.append("division")
    if any(w in q_lower for w in ["percent", "%", "fraction", "ratio"]):
        ops.append("percentage/ratio")
    if any(w in q_lower for w in ["more than", "less than", "fewer", "greater"]):
        ops.append("comparison")
    if any(w in q_lower for w in ["profit", "cost", "price", "spend", "earn", "save"]):
        ops.append("financial")

    if ops:
        parts.append(f"[OPERATIONS] {', '.join(ops)}")

    # Count steps implied (number of sentences as proxy)
    sentences = [s.strip() for s in re.split(r'[.!?]', question) if s.strip()]
    if len(sentences) > 2:
        parts.append(f"[COMPLEXITY] {len(sentences)} steps implied")

    return parts


def _retrieve_math_graph(question: str) -> list[str]:
    """
    Retrieve relevant mathematical knowledge from the graph YAML definitions.
    """
    import yaml
    parts = []

    q_lower = question.lower()

    # Check if info theory concepts are relevant
    info_keywords = {"entropy", "information", "probability", "random",
                     "likely", "chance", "odds", "expected"}
    if info_keywords & set(q_lower.split()):
        try:
            it_path = REPO_ROOT / "packages" / "infotheory" / "IT01-information-theory.yaml"
            if it_path.exists():
                with open(it_path) as f:
                    it = yaml.safe_load(f)
                for concept in it.get("concepts", [])[:3]:
                    if any(k in q_lower for k in concept.get("name", "").lower().split()):
                        parts.append(
                            f"[MATH] {concept['name']}: {concept.get('latex', '')} "
                            f"— {concept.get('meaning', '')}"
                        )
        except Exception:
            pass

    # Rate/proportion: provide the relationship
    if "per " in q_lower or "each " in q_lower:
        parts.append("[FORMULA] total = rate × quantity; unit_price × count = cost")

    if "percent" in q_lower or "%" in q_lower:
        parts.append("[FORMULA] percentage: part/whole × 100; amount × (pct/100)")

    return parts


def _retrieve_language_graph(question: str) -> str:
    """
    Use the language graph to resolve ambiguity and provide structure.
    """
    try:
        from packages.core.language.check import check_text, get_polysemy
        from nltk.corpus import wordnet as wn
    except ImportError:
        try:
            from language_graph.check import check_text, get_polysemy
            from nltk.corpus import wordnet as wn
        except ImportError:
            return ""

    tokens = question.lower().split()
    context_parts = []

    # Check for homophones / ambiguity
    result = check_text(question)
    if result.has_moments():
        for m in result.moments:
            context_parts.append(f"[DISAMBIGUATION] {m.token}: {m.message}")

    # For content words: provide hypernym chain (type hierarchy)
    content_words = [t.strip(".,;:!?\"'") for t in tokens
                     if len(t) > 3 and get_polysemy(t.strip(".,;:!?\"'")) > 0]

    for word in content_words[:10]:
        synsets = wn.synsets(word)
        if not synsets:
            continue

        n_senses = len(synsets)
        if n_senses == 1:
            # Monosemous — scriptable, provide definition directly
            s = synsets[0]
            hyper_chain = []
            current = s
            for _ in range(3):
                hypers = current.hypernyms()
                if not hypers:
                    break
                current = hypers[0]
                hyper_chain.append(current.lemmas()[0].name())

            if hyper_chain:
                context_parts.append(
                    f"[RESOLVED] {word}: {s.definition()} "
                    f"(type: {' → '.join(hyper_chain)})"
                )
        elif n_senses > 5:
            # Highly polysemous — flag for LLM attention
            top_senses = [f"{s.pos()}: {s.definition()}" for s in synsets[:3]]
            context_parts.append(
                f"[AMBIGUOUS] {word} ({n_senses} senses): {'; '.join(top_senses)}"
            )

    if not context_parts:
        return ""

    return "GRAPH CONTEXT:\n" + "\n".join(context_parts) + "\n---\n"


def _retrieve_inference_pipeline(question: str) -> str:
    """
    Use the inference pipeline (embeddings → clusters → retrieval).
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        emb_path = REPO_ROOT / "inference" / "01-embeddings" / "embeddings.npy"
        meta_path = REPO_ROOT / "inference" / "01-embeddings" / "metadata.json"

        if not emb_path.exists():
            return ""

        embeddings = np.load(str(emb_path))
        with open(meta_path) as f:
            metadata = json.load(f)

        model = SentenceTransformer(metadata["model"])
        q_emb = model.encode([question], convert_to_numpy=True)[0]

        sims = embeddings @ q_emb / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])

        if top_sim < 0.3:
            return ""

        chunks_path = REPO_ROOT / "inference" / "00-chunks" / "chunks.jsonl"
        with open(chunks_path) as f:
            chunks = [json.loads(line) for line in f if line.strip()]

        if top_idx < len(chunks):
            chunk = chunks[top_idx]
            return (
                f"RETRIEVED CONTEXT (similarity={top_sim:.3f}):\n"
                f"[{chunk.get('section_title', '')}]\n"
                f"{chunk.get('text', '')[:1000]}\n---\n"
            )

    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# Claude CLI runner
# ---------------------------------------------------------------------------

def call_claude_cli(prompt: str, model: str = "sonnet",
                    max_tokens: int = 256) -> tuple[str, int]:
    """
    Call Claude via the CLI subprocess.
    Returns (response_text, latency_ms).

    Uses --print flag for non-interactive single-turn.
    No memory, no project context, no MCP — clean room.
    """
    start = time.monotonic()

    try:
        result = subprocess.run(
            [
                "claude",
                "-p", prompt,           # --print: non-interactive, stdout only
                "--model", model,
                "--output-format", "text",
                "--disable-slash-commands",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            return f"[ERROR: {result.stderr.strip()[:200]}]", elapsed_ms

        return result.stdout.strip(), elapsed_ms

    except subprocess.TimeoutExpired:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return "[ERROR: timeout]", elapsed_ms
    except FileNotFoundError:
        return "[ERROR: claude CLI not found]", 0


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def solve_graph_only(question: str) -> tuple[str, int]:
    """
    Attempt to solve the problem using only the knowledge graph.
    No LLM call. Pure structure extraction + deterministic computation.

    Returns (answer_str, latency_ms). Returns empty string if unsolvable.
    """
    start = time.monotonic()

    try:
        from packages.core.interpret import solve
        answer, problem = solve(question)

        elapsed = int((time.monotonic() - start) * 1000)

        if answer is not None and problem.confidence >= 0.5:
            # Format as integer if it's a whole number
            if answer == int(answer):
                return str(int(answer)), elapsed
            return str(round(answer, 2)), elapsed

        return "", elapsed

    except Exception:
        elapsed = int((time.monotonic() - start) * 1000)
        return "", elapsed


def run_case(case: TestCase, condition: str, model: str) -> RunResult:
    """Run one test case under one condition."""

    system_prefix = (
        "Answer the question concisely. Give ONLY the final numeric answer, "
        "no explanation, no units, just the number.\n\n"
    )

    if condition == "vanilla":
        prompt = system_prefix + case.question
        answer, latency_ms = call_claude_cli(prompt, model)
    elif condition == "graph":
        ctx = case.graph_context if case.graph_context else retrieve_graph_context(case.question)
        prompt = system_prefix + (ctx + "\n" if ctx else "") + case.question
        answer, latency_ms = call_claude_cli(prompt, model)
    elif condition == "graph_only":
        answer, latency_ms = solve_graph_only(case.question)
        prompt = ""
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Estimate tokens (~4 chars per token)
    input_chars = len(prompt)
    output_chars = len(answer)
    input_tokens_est = input_chars // 4
    output_tokens_est = output_chars // 4

    # Score
    correct = score_answer(answer, case.reference_answer) if answer else False

    error = ""
    if answer.startswith("[ERROR:"):
        error = answer
        correct = False

    return RunResult(
        test_id=case.id, condition=condition, model=model,
        answer=answer, correct=correct,
        input_chars=input_chars, output_chars=output_chars,
        input_tokens_est=input_tokens_est,
        output_tokens_est=output_tokens_est,
        latency_ms=latency_ms, error=error,
    )


def score_answer(answer: str, reference: str) -> bool:
    """Score an answer against reference. Flexible matching."""
    import re

    a = answer.lower().strip().rstrip(".")
    r = reference.lower().strip().rstrip(".")

    # Error responses are always wrong
    if a.startswith("[error:"):
        return False

    # Exact match
    if a == r:
        return True

    # Try numeric comparison first (handles "18", "$18", "18.0", "18 dollars")
    def extract_number(text: str):
        """Extract the final/primary number from text."""
        # Look for boxed answer (MATH benchmark)
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            text = boxed.group(1)
        # Strip currency, commas, units
        nums = re.findall(r'-?\$?\d[\d,]*\.?\d*', text)
        if nums:
            # Take the last number (usually the final answer)
            n = nums[-1].replace(',', '').lstrip('$')
            try:
                return float(n)
            except ValueError:
                pass
        return None

    a_num = extract_number(a)
    r_num = extract_number(r)
    if a_num is not None and r_num is not None:
        # Numeric match with small tolerance
        if abs(a_num - r_num) < 0.01:
            return True

    # Reference contained in answer (for short answers)
    if r in a:
        return True

    # Answer contained in reference
    if a in r and len(a) > 2:
        return True

    # Word-level near-match: handle inflection ("string" vs "stringed")
    a_words = a.split()
    r_words = r.split()
    if len(a_words) == len(r_words) and len(a_words) >= 1:
        matches = sum(1 for aw, rw in zip(a_words, r_words)
                      if aw == rw or aw.startswith(rw) or rw.startswith(aw))
        if matches == len(a_words):
            return True

    # Multiple choice: extract letter
    for prefix in ["(a)", "(b)", "(c)", "(d)", "a)", "b)", "c)", "d)",
                    "a.", "b.", "c.", "d."]:
        if a.startswith(prefix) and r.startswith(prefix):
            return a[:2] == r[:2]
        if a.startswith(prefix[0]) and r.startswith(prefix[0]) and len(a) <= 3:
            return a[0] == r[0]

    return False


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------

def load_suite(suite_name: str) -> list[TestCase]:
    """Load test cases from a suite JSON file."""
    suite_path = SUITES_DIR / f"{suite_name}.json"
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite not found: {suite_path}")

    with open(suite_path) as f:
        data = json.load(f)

    cases = []
    for item in data["cases"]:
        cases.append(TestCase(
            id=item["id"],
            suite=suite_name,
            question=item["question"],
            reference_answer=item["answer"],
            category=item.get("category", ""),
            difficulty=item.get("difficulty", ""),
            graph_context=item.get("graph_context", ""),
            metadata=item.get("metadata", {}),
        ))
    return cases


def list_suites() -> list[str]:
    """List available test suites."""
    if not SUITES_DIR.exists():
        return []
    return [p.stem for p in SUITES_DIR.glob("*.json")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(suite_name: str, model: str, n_runs: int = 1,
                  graph_backend: str = "language") -> SuiteResult:
    """Run a complete benchmark: all cases x both conditions x n_runs."""

    cases = load_suite(suite_name)
    result = SuiteResult(suite=suite_name, model=model, n_cases=len(cases))

    all_runs = []
    total = len(cases) * n_runs * 2
    done = 0

    for run_idx in range(n_runs):
        for case in cases:
            # Pre-compute graph context if not cached
            if not case.graph_context:
                case.graph_context = retrieve_graph_context(
                    case.question, graph_backend
                )

            # Run vanilla
            vanilla_result = run_case(case, "vanilla", model)
            all_runs.append(vanilla_result)
            done += 1
            v_mark = "+" if vanilla_result.correct else "X"
            v_ans = vanilla_result.answer[:40].replace("\n", " ")
            print(f"  [{done}/{total}] {case.id} vanilla {v_mark}  "
                  f"ans={v_ans:<20} ref={case.reference_answer:<10} "
                  f"{vanilla_result.latency_ms}ms  "
                  f"in={vanilla_result.input_tokens_est}tok")

            # Run graph-augmented
            graph_result = run_case(case, "graph", model)
            all_runs.append(graph_result)
            done += 1
            g_mark = "+" if graph_result.correct else "X"
            g_ans = graph_result.answer[:40].replace("\n", " ")
            ctx_chars = len(case.graph_context) if case.graph_context else 0
            print(f"  [{done}/{total}] {case.id} graph   {g_mark}  "
                  f"ans={g_ans:<20} ref={case.reference_answer:<10} "
                  f"{graph_result.latency_ms}ms  "
                  f"in={graph_result.input_tokens_est}tok  "
                  f"ctx={ctx_chars}ch")

    # Aggregate
    vanilla_runs = [r for r in all_runs if r.condition == "vanilla"]
    graph_runs = [r for r in all_runs if r.condition == "graph"]

    v_correct = sum(1 for r in vanilla_runs if r.correct)
    g_correct = sum(1 for r in graph_runs if r.correct)

    result.vanilla_accuracy = v_correct / len(vanilla_runs) if vanilla_runs else 0
    result.graph_accuracy = g_correct / len(graph_runs) if graph_runs else 0
    result.accuracy_delta = result.graph_accuracy - result.vanilla_accuracy

    result.vanilla_mean_input_tokens = (
        sum(r.input_tokens_est for r in vanilla_runs) / len(vanilla_runs)
        if vanilla_runs else 0
    )
    result.graph_mean_input_tokens = (
        sum(r.input_tokens_est for r in graph_runs) / len(graph_runs)
        if graph_runs else 0
    )
    result.vanilla_mean_output_tokens = (
        sum(r.output_tokens_est for r in vanilla_runs) / len(vanilla_runs)
        if vanilla_runs else 0
    )
    result.graph_mean_output_tokens = (
        sum(r.output_tokens_est for r in graph_runs) / len(graph_runs)
        if graph_runs else 0
    )

    result.vanilla_mean_latency_ms = (
        sum(r.latency_ms for r in vanilla_runs) / len(vanilla_runs)
        if vanilla_runs else 0
    )
    result.graph_mean_latency_ms = (
        sum(r.latency_ms for r in graph_runs) / len(graph_runs)
        if graph_runs else 0
    )

    if result.graph_mean_input_tokens > 0:
        result.token_savings_ratio = round(
            result.vanilla_mean_input_tokens / result.graph_mean_input_tokens, 2
        )

    result.runs = [asdict(r) for r in all_runs]

    return result


def print_result(result: SuiteResult):
    """Print benchmark results."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {result.suite} | Model: {result.model}")
    print(f"{'='*60}")
    print(f"  Cases: {result.n_cases}")
    print()
    print(f"  {'Metric':<25} {'Vanilla':<15} {'Graph':<15} {'Delta':<10}")
    print(f"  {'-'*65}")
    print(f"  {'Accuracy':<25} {result.vanilla_accuracy:.1%}{'':>9} "
          f"{result.graph_accuracy:.1%}{'':>9} "
          f"{result.accuracy_delta:+.1%}")
    print(f"  {'Avg input tokens':<25} {result.vanilla_mean_input_tokens:.0f}{'':>11} "
          f"{result.graph_mean_input_tokens:.0f}{'':>11} "
          f"{result.graph_mean_input_tokens - result.vanilla_mean_input_tokens:+.0f}")
    print(f"  {'Avg output tokens':<25} {result.vanilla_mean_output_tokens:.0f}{'':>11} "
          f"{result.graph_mean_output_tokens:.0f}{'':>11} "
          f"{result.graph_mean_output_tokens - result.vanilla_mean_output_tokens:+.0f}")
    print(f"  {'Avg latency (ms)':<25} {result.vanilla_mean_latency_ms:.0f}{'':>11} "
          f"{result.graph_mean_latency_ms:.0f}")
    print(f"  {'Input token ratio':<25} {'1.0x':<15} "
          f"{result.token_savings_ratio}x")

    # Show mismatches
    mismatches = []
    vanilla_map = {r["test_id"]: r for r in result.runs if r["condition"] == "vanilla"}
    graph_map = {r["test_id"]: r for r in result.runs if r["condition"] == "graph"}
    for tid in vanilla_map:
        v = vanilla_map[tid]
        g = graph_map.get(tid, {})
        if v.get("correct") != g.get("correct"):
            mismatches.append((tid, v, g))

    if mismatches:
        print(f"\n  Differences (vanilla vs graph):")
        for tid, v, g in mismatches:
            v_mark = "ok" if v.get("correct") else "X"
            g_mark = "ok" if g.get("correct") else "X"
            print(f"    {tid}: vanilla={v_mark} graph={g_mark}")
            print(f"      vanilla: {v.get('answer', '')[:80]}")
            print(f"      graph:   {g.get('answer', '')[:80]}")


def main():
    parser = argparse.ArgumentParser(description="AI benchmark harness")
    parser.add_argument("--suite", default="disambiguation",
                        help="Test suite name or 'all'")
    parser.add_argument("--model", default="sonnet",
                        help="Model: haiku, sonnet, opus")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per case")
    parser.add_argument("--backend", default="full",
                        help="Graph backend: full, language, inference")
    parser.add_argument("--list", action="store_true",
                        help="List available suites")
    args = parser.parse_args()

    if args.list:
        suites = list_suites()
        print(f"Available suites: {', '.join(suites) if suites else 'none'}")
        return

    model = MODEL_ALIASES.get(args.model, args.model)

    if args.suite == "all":
        suites = list_suites()
    else:
        suites = [args.suite]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify claude CLI is available
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
        print(f"Claude CLI: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: 'claude' CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        return

    for suite_name in suites:
        print(f"\nRunning {suite_name} ({model})...")
        result = run_benchmark(suite_name, model, args.runs, args.backend)
        print_result(result)

        # Save detailed results
        out_path = RESULTS_DIR / f"{suite_name}_{args.model}.json"
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
