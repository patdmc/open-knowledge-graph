"""
Language graph loader.

Reads the compiled language graph (knowledge-graph/compiled/language.json)
and provides it to the lexer, parser, and resolver. All lexical knowledge
comes from the graph — the code is pure action.

The graph is loaded once and cached in memory.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

_GRAPH = None
_GRAPH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "knowledge-graph", "compiled"
)


@dataclass
class NotationRule:
    id: str
    symbol: Optional[str]
    position: str
    token_type: str
    unit: Optional[str]
    scale: Optional[float]
    operation: Optional[str]
    pattern: re.Pattern


@dataclass
class ReferenceRule:
    id: str
    resolves_to: str
    pattern: re.Pattern
    captures: Optional[str]
    multiplier_words: bool


@dataclass
class LanguageGraph:
    """Compiled language graph — all lexical knowledge in one place."""
    notations: list[NotationRule]
    number_words: dict[str, float]
    scale_multipliers: dict[str, float]
    unit_multipliers: dict[str, float]
    references: list[ReferenceRule]
    rate_words: set[str]
    pair_is_one_unit: set[str]
    default_pair_value: int
    stop_words: set[str]
    synset_to_op: dict[str, str]
    verb_index: dict[str, str]


def load() -> LanguageGraph:
    """Load and cache the compiled language graph."""
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    path = os.path.join(_GRAPH_DIR, "language.json")
    with open(path) as f:
        raw = json.load(f)

    notations = [
        NotationRule(
            id=n["id"],
            symbol=n["symbol"],
            position=n["position"],
            token_type=n["token_type"],
            unit=n.get("unit"),
            scale=n.get("scale"),
            operation=n.get("operation"),
            pattern=re.compile(n["pattern"]),
        )
        for n in raw["notations"]
    ]

    references = [
        ReferenceRule(
            id=r["id"],
            resolves_to=r["resolves_to"],
            pattern=re.compile(r["pattern"]),
            captures=r.get("captures"),
            multiplier_words=r.get("multiplier_words", False),
        )
        for r in raw["references"]
    ]

    rate_words = {r["word"] for r in raw["rate_indicators"]}

    _GRAPH = LanguageGraph(
        notations=notations,
        number_words=raw["number_words"],
        scale_multipliers=raw["scale_multipliers"],
        unit_multipliers=raw["unit_multipliers"],
        references=references,
        rate_words=rate_words,
        pair_is_one_unit=set(raw["pair_is_one_unit"]),
        default_pair_value=raw["default_pair_value"],
        stop_words=set(raw["stop_words"]),
        synset_to_op=raw["synset_operations"],
        verb_index=raw["verb_index"],
    )
    return _GRAPH


def reload():
    """Force reload (after recompilation)."""
    global _GRAPH
    _GRAPH = None
    return load()
