"""
benchmark/harness.py — AI test harness for knowledge-graph augmented inference.

Compares two conditions:
  A) Claude alone (vanilla prompt)
  B) Claude + language graph (graph-retrieved context prepended)

On standardized test suites. Measures:
  - Accuracy (exact match or judge-scored)
  - Input tokens (context size)
  - Output tokens (response length)
  - Latency (wall clock)
  - Cost (estimated from token counts)

Usage:
    python -m gnn.benchmark.harness --suite disambiguation --model haiku
    python -m gnn.benchmark.harness --suite all --model sonnet --runs 3
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import anthropic

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "gnn" / "benchmark" / "results"
SUITES_DIR = REPO_ROOT / "gnn" / "benchmark" / "suites"

# Token pricing per 1M tokens (as of 2026-03)
PRICING = {
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":          {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":            {"input": 15.00, "output": 75.00},
}

MODEL_ALIASES = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
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
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
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
    vanilla_total_cost: float = 0.0
    graph_total_cost: float = 0.0
    vanilla_mean_latency_ms: float = 0.0
    graph_mean_latency_ms: float = 0.0
    token_savings_ratio: float = 0.0
    runs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph context retrieval
# ---------------------------------------------------------------------------

def retrieve_graph_context(question: str, graph_backend: str = "language") -> str:
    """
    Query the knowledge graph for context relevant to the question.

    This is the key function that determines what the graph adds.
    Strategies (plug in as backends):
      - "language": WordNet synonyms/hypernyms/frames for key terms
      - "inference": chunk retrieval from equivalence classes
      - "neo4j": live Neo4j traversal

    Returns a context string to prepend to the prompt.
    """
    if graph_backend == "language":
        return _retrieve_language_graph(question)
    elif graph_backend == "inference":
        return _retrieve_inference_pipeline(question)
    else:
        return ""


def _retrieve_language_graph(question: str) -> str:
    """
    Use the language graph to resolve ambiguity and provide structure.

    1. Tokenize the question
    2. For each token: check polysemy, retrieve synsets, hypernym chains
    3. For ambiguous tokens: retrieve disambiguation context
    4. For domain terms: retrieve frame evocations
    5. Return compact context string
    """
    try:
        from language_graph.check import check_text, get_polysemy
        from nltk.corpus import wordnet as wn
    except ImportError:
        return "[language graph unavailable]"

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

    for word in content_words[:10]:  # cap to avoid bloat
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
    Finds the most relevant equivalence class for the question.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        emb_path = REPO_ROOT / "inference" / "01-embeddings" / "embeddings.npy"
        meta_path = REPO_ROOT / "inference" / "01-embeddings" / "metadata.json"
        classes_path = REPO_ROOT / "inference" / "02-clusters" / "classes.json"

        if not emb_path.exists():
            return ""

        embeddings = np.load(str(emb_path))
        with open(meta_path) as f:
            metadata = json.load(f)
        with open(classes_path) as f:
            classes = json.load(f)

        # Embed the question
        model = SentenceTransformer(metadata["model"])
        q_emb = model.encode([question], convert_to_numpy=True)[0]

        # Find nearest chunk
        sims = embeddings @ q_emb / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])

        if top_sim < 0.3:
            return ""

        # Load the chunk text
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
# Model runner
# ---------------------------------------------------------------------------

def run_case(client: anthropic.Anthropic, case: TestCase, condition: str,
             model: str) -> RunResult:
    """Run one test case under one condition."""

    if condition == "vanilla":
        prompt = case.question
    elif condition == "graph":
        if case.graph_context:
            prompt = case.graph_context + "\n" + case.question
        else:
            # Retrieve on the fly
            ctx = retrieve_graph_context(case.question)
            prompt = ctx + "\n" + case.question if ctx else case.question
    else:
        raise ValueError(f"Unknown condition: {condition}")

    system_msg = (
        "Answer the question concisely. Give only the answer, "
        "no explanation unless asked."
    )

    start = time.monotonic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        answer = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Cost estimate
        pricing = PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        # Score
        correct = score_answer(answer, case.reference_answer)

        return RunResult(
            test_id=case.id, condition=condition, model=model,
            answer=answer, correct=correct,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=elapsed_ms, cost_usd=round(cost, 6),
        )

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return RunResult(
            test_id=case.id, condition=condition, model=model,
            answer="", correct=False, latency_ms=elapsed_ms,
            error=str(e),
        )


def score_answer(answer: str, reference: str) -> bool:
    """Score an answer against reference. Flexible matching."""
    a = answer.lower().strip().rstrip(".")
    r = reference.lower().strip().rstrip(".")

    # Exact match
    if a == r:
        return True

    # Reference contained in answer (for short answers)
    if r in a:
        return True

    # Answer contained in reference
    if a in r:
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
    """Run a complete benchmark: all cases × both conditions × n_runs."""

    client = anthropic.Anthropic()
    cases = load_suite(suite_name)

    result = SuiteResult(suite=suite_name, model=model, n_cases=len(cases))

    all_runs = []
    for run_idx in range(n_runs):
        for case in cases:
            # Pre-compute graph context if not cached
            if not case.graph_context:
                case.graph_context = retrieve_graph_context(
                    case.question, graph_backend
                )

            # Run vanilla
            vanilla_result = run_case(client, case, "vanilla", model)
            all_runs.append(vanilla_result)

            # Run graph-augmented
            graph_result = run_case(client, case, "graph", model)
            all_runs.append(graph_result)

    # Aggregate
    vanilla_runs = [r for r in all_runs if r.condition == "vanilla"]
    graph_runs = [r for r in all_runs if r.condition == "graph"]

    v_correct = sum(1 for r in vanilla_runs if r.correct)
    g_correct = sum(1 for r in graph_runs if r.correct)

    result.vanilla_accuracy = v_correct / len(vanilla_runs) if vanilla_runs else 0
    result.graph_accuracy = g_correct / len(graph_runs) if graph_runs else 0

    result.vanilla_mean_input_tokens = (
        sum(r.input_tokens for r in vanilla_runs) / len(vanilla_runs)
        if vanilla_runs else 0
    )
    result.graph_mean_input_tokens = (
        sum(r.input_tokens for r in graph_runs) / len(graph_runs)
        if graph_runs else 0
    )
    result.vanilla_mean_output_tokens = (
        sum(r.output_tokens for r in vanilla_runs) / len(vanilla_runs)
        if vanilla_runs else 0
    )
    result.graph_mean_output_tokens = (
        sum(r.output_tokens for r in graph_runs) / len(graph_runs)
        if graph_runs else 0
    )

    result.vanilla_total_cost = sum(r.cost_usd for r in vanilla_runs)
    result.graph_total_cost = sum(r.cost_usd for r in graph_runs)

    result.vanilla_mean_latency_ms = (
        sum(r.latency_ms for r in vanilla_runs) / len(vanilla_runs)
        if vanilla_runs else 0
    )
    result.graph_mean_latency_ms = (
        sum(r.latency_ms for r in graph_runs) / len(graph_runs)
        if graph_runs else 0
    )

    # Token savings: how much LESS context does graph need for same/better accuracy?
    if result.vanilla_mean_input_tokens > 0:
        result.token_savings_ratio = round(
            result.vanilla_mean_input_tokens / result.graph_mean_input_tokens, 2
        ) if result.graph_mean_input_tokens > 0 else 0

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
          f"{result.graph_accuracy - result.vanilla_accuracy:+.1%}")
    print(f"  {'Avg input tokens':<25} {result.vanilla_mean_input_tokens:.0f}{'':>11} "
          f"{result.graph_mean_input_tokens:.0f}{'':>11} "
          f"{result.graph_mean_input_tokens - result.vanilla_mean_input_tokens:+.0f}")
    print(f"  {'Avg output tokens':<25} {result.vanilla_mean_output_tokens:.0f}{'':>11} "
          f"{result.graph_mean_output_tokens:.0f}{'':>11} "
          f"{result.graph_mean_output_tokens - result.vanilla_mean_output_tokens:+.0f}")
    print(f"  {'Total cost ($)':<25} ${result.vanilla_total_cost:.4f}{'':>8} "
          f"${result.graph_total_cost:.4f}{'':>8}")
    print(f"  {'Avg latency (ms)':<25} {result.vanilla_mean_latency_ms:.0f}{'':>11} "
          f"{result.graph_mean_latency_ms:.0f}")
    print(f"  {'Token savings ratio':<25} {'1.0x':<15} "
          f"{result.token_savings_ratio}x")


def main():
    parser = argparse.ArgumentParser(description="AI benchmark harness")
    parser.add_argument("--suite", default="disambiguation",
                        help="Test suite name or 'all'")
    parser.add_argument("--model", default="haiku",
                        help="Model: haiku, sonnet, opus")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per case")
    parser.add_argument("--backend", default="language",
                        help="Graph backend: language, inference")
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

    for suite_name in suites:
        result = run_benchmark(suite_name, model, args.runs, args.backend)
        print_result(result)

        # Save detailed results
        out_path = RESULTS_DIR / f"{suite_name}_{args.model}.json"
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
