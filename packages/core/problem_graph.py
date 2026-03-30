"""
Ephemeral problem graph — built per word problem, walked to solve.

Each word problem creates its own temporary graph:
  - Nodes: quantities (numbers + units), entities (people/things)
  - Edges: operations (language graph) and conversions (math graph)

The graph links to permanent knowledge:
  - Language graph (resolve.py): verb → math operation
  - Math graph (units.py): unit → unit conversion

The walk from target backwards IS the computation.
No solver cascade. The path from target to known values produces the AST.

Architecture:
  1. Parse clauses → quantity nodes + entity nodes
  2. Link operations via language graph (verb edges)
  3. Link conversions via math graph (unit edges)
  4. Identify target from question
  5. Walk backward from target
  6. The walk produces the AST

Evaluation:
  Each ephemeral graph is retained as a GraphSnapshot with:
  - The graph itself (nodes, entities, target)
  - The walk result (AST, confidence, path)
  - The actual answer vs expected (when known)
  - Failure classification (equivalence class of process failure)

  This is the learning loop: failed graphs classify into process failures,
  fixes apply to the process (graph builder, walker), not to individual problems.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union

from packages.core.language.resolve import resolve_sentence
from packages.math.units import (
    lookup as unit_lookup, convert as unit_convert,
    UNIT_TO_HOURS, UNIT_TO_DOLLARS,
)


# =====================================================================
# Graph nodes
# =====================================================================

@dataclass
class QNode:
    """A quantity in the problem graph: a number with units and context."""
    id: str
    value: float
    unit: str = ""          # what it measures: "eggs", "dollars", "miles"
    per_unit: str = ""      # rate denominator: "day", "egg", "hour"
    entity: str = ""        # owning entity: "janet", "steve"
    operation: str = ""     # how this modifies its entity's state: add/subtract/multiply/divide
    is_rate: bool = False   # True if this is a conversion rate (has per_unit)
    clause_idx: int = 0
    source_text: str = ""


@dataclass
class EntityNode:
    """A named entity in the problem: a person, thing, or category."""
    name: str
    quantities: list[str] = field(default_factory=list)  # QNode ids


# =====================================================================
# Problem graph
# =====================================================================

@dataclass
class ProblemGraph:
    """Ephemeral graph for one word problem. Linked to permanent graphs."""
    nodes: dict[str, QNode] = field(default_factory=dict)  # id → QNode
    entities: dict[str, EntityNode] = field(default_factory=dict)
    target_unit: str = ""
    target_type: str = ""       # "money", "count", "difference"
    target_agg: str = ""        # "remaining", "total", "final"
    debug: list[str] = field(default_factory=list)

    def nodes_with_per_unit(self) -> list[QNode]:
        """All nodes that have a per_unit (potential conversion rates)."""
        return [n for n in self.nodes.values() if n.per_unit]

    def values_with_unit(self, unit: str, exclude_id: str = "") -> list[QNode]:
        """Nodes whose unit matches, optionally excluding one node."""
        result = []
        for n in self.nodes.values():
            if n.id == exclude_id:
                continue
            if not n.unit:
                continue
            if _units_match(n.unit, unit):
                result.append(n)
        return sorted(result, key=lambda n: n.clause_idx)

    def all_values(self) -> list[QNode]:
        """All quantity nodes, ordered by clause."""
        return sorted(self.nodes.values(), key=lambda n: n.clause_idx)


def _units_match(a: str, b: str) -> bool:
    """Check if two unit strings refer to the same thing (fuzzy)."""
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    # Prefix match (egg/eggs, dollar/dollars)
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and long.startswith(short[:3]):
        return True
    return False


# =====================================================================
# Stop words for entity detection
# =====================================================================

_ENTITY_STOP = {
    "the", "if", "she", "he", "how", "what", "in", "on", "at", "it",
    "one", "each", "they", "her", "his", "a", "an", "for", "but", "and",
    "or", "so", "yet", "from", "with", "by", "to", "of", "is", "are",
    "was", "were", "has", "had", "do", "does", "did", "will", "can",
    "may", "not", "then", "than", "when", "where", "there", "here",
    "this", "that", "some", "all", "many", "much", "most", "every",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "after", "before", "during", "since", "until", "while",
}


def _extract_entity(text: str) -> str:
    """Extract the primary entity (proper noun) from text."""
    for m in re.finditer(r'\b([A-Z][a-z]{2,})\b', text):
        name = m.group(1).lower()
        if name not in _ENTITY_STOP:
            return name
    return ""


# =====================================================================
# Build the problem graph from parsed clauses
# =====================================================================

def build_graph(clauses, target: dict) -> ProblemGraph:
    """
    Build an ephemeral problem graph from parsed clauses.

    Each clause contributes quantity nodes. Verbs become operation labels
    via the language graph. Units link to the math graph.
    """
    pg = ProblemGraph()
    pg.target_unit = target.get("unit", "")
    pg.target_type = target.get("question_type", "count")
    pg.target_agg = target.get("aggregation", "")

    # Track the "current" entity across clauses (pronouns inherit)
    current_entity = ""
    # Track which entity+unit combos have been seen (first = initial)
    seen_combos: set[str] = set()

    for ci, clause in enumerate(clauses):
        sent_lower = clause.text.lower()

        # Determine entity for this clause
        entity = _extract_entity(clause.text)
        if entity:
            current_entity = entity
        else:
            entity = current_entity

        # Ensure entity node exists
        if entity and entity not in pg.entities:
            pg.entities[entity] = EntityNode(name=entity)

        # Determine operation from verb (already resolved via language graph)
        op = clause.operation if clause.role == "operation" else ""

        # Extract quantity nodes from tokens
        for t in clause.tokens:
            if t.type not in ("number", "money", "fraction"):
                continue

            unit = t.unit or ("dollars" if t.type == "money" else "")
            per_unit = t.per_unit

            # Don't mark as rate in the graph builder.
            # The WALKER decides which node is the converting rate
            # based on target unit matching. "16 eggs per day" is a base
            # quantity. "$2 per egg" is a conversion rate. The difference
            # is which one the walker selects.

            # Is this the first occurrence of this entity+unit?
            combo = f"{entity}:{unit}"
            is_first = combo not in seen_combos
            if is_first and unit:
                seen_combos.add(combo)

            # Determine this node's operation
            node_op = ""
            if is_first:
                node_op = "initial"
            elif op:
                node_op = op

            nid = f"c{ci}_{t.position}"
            node = QNode(
                id=nid,
                value=t.value,
                unit=unit,
                per_unit=per_unit,
                entity=entity,
                operation=node_op,
                is_rate=False,  # walker decides, not builder
                clause_idx=ci,
                source_text=clause.text,
            )
            pg.nodes[nid] = node

            if entity and entity in pg.entities:
                pg.entities[entity].quantities.append(nid)

    pg.debug.append(f"graph: {len(pg.nodes)} nodes, {len(pg.entities)} entities")
    pg.debug.append(f"target: {pg.target_unit} ({pg.target_type})")
    return pg


# =====================================================================
# Walk backward from target
# =====================================================================

@dataclass
class WalkResult:
    """Result of a graph walk: AST + confidence."""
    ast: object  # ASTNode
    confidence: float  # 0-1: how complete/reliable the walk is
    path: str  # description of the walk path
    nodes_used: int  # how many graph nodes contributed
    nodes_total: int  # total non-rate nodes in graph


def walk(pg: ProblemGraph) -> Optional[WalkResult]:
    """
    Walk backward from the target to produce an AST with confidence.

    Returns WalkResult with AST and confidence score, or None.
    Confidence tells the caller whether to trust the walk or fall through.

    Algorithm:
      1. Target unit → find rate that converts TO target unit
      2. Rate has a source unit (per_unit) → collect values in that unit
      3. Accumulate source values (initial ± operations)
      4. Apply rate conversion → result
      5. Score confidence based on path completeness
    """
    from packages.core.interpret import Lit, BinOp

    target_unit = pg.target_unit
    if not target_unit:
        return None

    total_nodes = len(pg.all_values())

    # --- Step 1: Find rate conversion to target ---
    rate_node = _find_rate_to_target(pg, target_unit)
    source_values = []

    if rate_node:
        source_unit = rate_node.per_unit
        pg.debug.append(f"walk: rate {rate_node.value:g} {rate_node.unit}/{rate_node.per_unit}")

        # Validate: rate unit ≠ per_unit (self-reference = false rate)
        if rate_node.unit and rate_node.per_unit and \
                _units_match(rate_node.unit, rate_node.per_unit):
            pg.debug.append(f"walk: rejected self-referencing rate")
            rate_node = None
        else:
            # Collect source values matching the rate's per_unit
            source_values = _collect_source_values(pg, source_unit, rate_node)
            if not source_values:
                source_values = _collect_unmatched_values(pg, rate_node)

    if rate_node and source_values:
        source_ast = _accumulate(source_values, Lit, BinOp)
        if source_ast:
            # Decide operation direction: multiply or divide?
            # - If target asks "per each/per unit" → divide (total ÷ count)
            # - If target asks "how much/many total" → multiply (count × rate)
            # - Default: multiply (rate × source = target)
            op_direction = _infer_rate_direction(pg, rate_node, source_values)

            if op_direction == "divide":
                ast = BinOp("divide",
                            Lit(rate_node.value, f"total:{rate_node.value:g}"),
                            source_ast)
            else:
                ast = BinOp("multiply", source_ast,
                            Lit(rate_node.value, f"rate:{rate_node.value:g}"))

            n_used = len(source_values)
            n_with_ops = sum(1 for v in source_values
                             if v.operation in ("initial", "add", "subtract",
                                                 "multiply", "divide"))
            op_ratio = n_with_ops / n_used if n_used else 0
            coverage = (n_used + 1) / total_nodes if total_nodes else 0

            conf = min(0.95, op_ratio * 0.6 + coverage * 0.4)

            pg.debug.append(f"walk: source({n_used}) → {op_direction} rate → target "
                            f"[conf={conf:.2f}]")
            return WalkResult(
                ast=ast, confidence=conf,
                path=f"rate:{rate_node.unit}/{rate_node.per_unit}",
                nodes_used=n_used + 1, nodes_total=total_nodes)

    # --- Step 2b: Chain walk — resolve per_unit groups recursively ---
    chain_result = _try_chain_walk(pg, Lit, BinOp, total_nodes)
    if chain_result:
        return chain_result

    # --- Step 3: Direct accumulation on target unit ---
    target_values = pg.values_with_unit(target_unit)
    if target_values:
        ast = _accumulate(target_values, Lit, BinOp)
        if ast:
            n_used = len(target_values)
            n_with_ops = sum(1 for v in target_values
                             if v.operation in ("initial", "add", "subtract",
                                                 "multiply", "divide"))
            op_ratio = n_with_ops / n_used if n_used else 0
            coverage = n_used / total_nodes if total_nodes else 0

            conf = min(0.9, op_ratio * 0.5 + coverage * 0.3)

            pg.debug.append(f"walk: direct on {target_unit} ({n_used} nodes) "
                            f"[conf={conf:.2f}]")
            return WalkResult(
                ast=ast, confidence=conf,
                path=f"direct:{target_unit}",
                nodes_used=n_used, nodes_total=total_nodes)

    # No fallback — if we can't trace a path, we shouldn't guess
    return None


def _try_chain_walk(pg: ProblemGraph, Lit, BinOp, total_nodes: int) -> Optional[WalkResult]:
    """
    Chain walk: resolve per_unit groups into sub-expressions, then combine.

    When nodes share a per_unit, they form a chain:
    - Count node: unit matches per_unit (6 decks, per_unit=deck → count of decks)
    - Rate node: unit doesn't match per_unit (25 cards, per_unit=deck → cards per deck)
    → Multiply: count × rate = total (6 × 25 = 150 cards)

    Multiple chains at the same level get summed.
    Remaining ungrouped nodes apply as operations (subtract, divide, etc.).
    """
    from collections import defaultdict

    # Group nodes by per_unit
    pu_groups = defaultdict(list)  # per_unit → [QNode]
    ungrouped = []  # nodes without per_unit

    for n in pg.all_values():
        if n.per_unit:
            pu_groups[n.per_unit].append(n)
        else:
            ungrouped.append(n)

    # Find valid rate pairs: exactly one count + one rate per group
    chain_products = []
    used_nodes = set()

    for pu, nodes in pu_groups.items():
        if len(nodes) != 2:
            # Can't pair — move to ungrouped
            ungrouped.extend(nodes)
            continue

        # Identify count vs rate
        count_node = None
        rate_node = None
        for n in nodes:
            if n.unit and _units_match(n.unit, pu):
                count_node = n
            else:
                rate_node = n

        if count_node and rate_node:
            product = BinOp("multiply",
                            Lit(count_node.value, f"{count_node.value:g}"),
                            Lit(rate_node.value, f"{rate_node.value:g}"))
            chain_products.append(product)
            used_nodes.add(count_node.id)
            used_nodes.add(rate_node.id)
        else:
            # Both match or neither matches — not a valid pair
            ungrouped.extend(nodes)

    if not chain_products:
        return None

    # Sum chain products
    ast = chain_products[0]
    for p in chain_products[1:]:
        ast = BinOp("add", ast, p)

    # Apply ungrouped nodes as operations
    for n in ungrouped:
        if n.id in used_nodes:
            continue
        if n.operation == "subtract":
            ast = BinOp("subtract", ast, Lit(n.value, n.source_text[:20]))
        elif n.operation == "divide":
            ast = BinOp("divide", ast, Lit(n.value, n.source_text[:20]))
        elif n.operation == "add":
            ast = BinOp("add", ast, Lit(n.value, n.source_text[:20]))
        elif n.operation == "multiply":
            ast = BinOp("multiply", ast, Lit(n.value, n.source_text[:20]))
        elif n.operation == "initial":
            # Check context: if unit matches target, might be a final operation
            # like "keeps 50 cards" → subtract
            import re
            if re.search(r'\b(?:keep|kept|save|remain|left)\b',
                         n.source_text.lower()):
                ast = BinOp("subtract", ast, Lit(n.value, n.source_text[:20]))
            elif re.search(r'\b(?:equally|evenly|split|share|between|among)\b',
                           n.source_text.lower()):
                ast = BinOp("divide", ast, Lit(n.value, n.source_text[:20]))

    n_used = len(used_nodes) + sum(1 for n in ungrouped
                                    if n.id not in used_nodes
                                    and n.operation in ("subtract", "divide",
                                                        "add", "multiply"))
    coverage = n_used / total_nodes if total_nodes else 0
    conf = min(0.9, 0.5 + coverage * 0.4)

    pg.debug.append(f"walk: chain {len(chain_products)} groups, "
                    f"{len(ungrouped)} remaining [conf={conf:.2f}]")

    return WalkResult(
        ast=ast, confidence=conf,
        path=f"chain:{len(chain_products)}",
        nodes_used=n_used, nodes_total=total_nodes)


def _infer_rate_direction(pg: ProblemGraph, rate_node: QNode,
                          source_values: list[QNode]) -> str:
    """
    Infer whether to multiply or divide by the rate.

    - "per each" / "per gift" in question → DIVIDE (total ÷ count)
    - Source values are counts (small), rate is large total → DIVIDE
    - Rate is a price/unit ($X per item) → MULTIPLY
    - Default: MULTIPLY
    """
    import re

    # Check question target aggregation
    if pg.target_agg == "remaining":
        return "multiply"  # remainder problems use multiply

    # Check if question asks "per each" or "per [unit]"
    # (This would mean answer = rate ÷ source_sum)
    # The target_type from _extract_target already handles this

    # Heuristic: if rate value > sum of source values, it might be a total
    # that needs dividing (e.g., 144 inches ÷ 12 gifts = 12 per gift)
    source_sum = sum(v.value for v in source_values)
    if source_sum > 0 and rate_node.value > source_sum * 5:
        # Rate is much larger than sources — might be total ÷ count
        # But only if the rate's unit matches the target
        if _units_match(rate_node.unit, pg.target_unit):
            return "divide"

    return "multiply"


def _find_rate_to_target(pg: ProblemGraph, target_unit: str) -> Optional[QNode]:
    """
    Find the conversion rate node — the one that bridges to the target unit.

    A conversion rate has:
    - unit that matches the target unit (e.g., "dollars" when target is "dollars")
    - per_unit that is DIFFERENT from the target (e.g., "egg" — the source dimension)
    - It converts source units into target units

    "16 eggs per day" is NOT a conversion to dollars — eggs ≠ dollars.
    "$2 per egg" IS a conversion to dollars — dollars = target.
    """
    # Look through ALL nodes with per_unit (potential conversion rates)
    candidates = [n for n in pg.nodes.values() if n.per_unit]

    for node in candidates:
        # Unit must match target
        if not _units_match(node.unit, target_unit):
            continue
        # per_unit must be DIFFERENT from target (it's the source dimension)
        if _units_match(node.per_unit, target_unit):
            continue
        # Self-referencing rate check
        if node.unit and node.per_unit and _units_match(node.unit, node.per_unit):
            continue
        return node

    # Fallback: check via math graph dimension matching
    for node in candidates:
        if node.unit and target_unit:
            converted = unit_convert(1.0, node.unit, target_unit)
            if converted is not None and not _units_match(node.per_unit, target_unit):
                return node

    return None


def _collect_source_values(pg: ProblemGraph, source_unit: str,
                           rate_node: QNode) -> list[QNode]:
    """Collect all values that match the source unit, excluding the rate node."""
    result = []
    for n in sorted(pg.nodes.values(), key=lambda n: n.clause_idx):
        if n.id == rate_node.id:
            continue
        if n.unit and _units_match(n.unit, source_unit):
            result.append(n)
    return result


def _collect_unmatched_values(pg: ProblemGraph,
                              rate_node: QNode) -> list[QNode]:
    """
    Collect values that aren't the rate and don't match the target unit.

    These are "source" values when units aren't explicitly tagged
    on every token (common — "she eats 3" has no unit on 3).
    """
    result = []
    for n in sorted(pg.nodes.values(), key=lambda n: n.clause_idx):
        if n.id == rate_node.id:
            continue
        # Skip values whose unit IS the target (those are results, not sources)
        if n.unit and _units_match(n.unit, pg.target_unit):
            continue
        result.append(n)
    return result


def _accumulate(nodes: list[QNode], Lit, BinOp):
    """
    Accumulate quantity nodes into an AST.

    Initial nodes start the accumulator. Subsequent nodes apply their
    operation (add/subtract/multiply/divide) to the running total.
    """
    if not nodes:
        return None

    # Separate initial values from operations
    initials = [n for n in nodes if n.operation == "initial"]
    operations = [n for n in nodes if n.operation not in ("initial", "")]

    # Start with the first initial value
    if initials:
        ast = Lit(initials[0].value)
        # Multiple initials in same unit = sum them (parallel sources)
        for init_n in initials[1:]:
            ast = BinOp("add", ast, Lit(init_n.value))
    elif nodes:
        # No explicit initial — use first node
        ast = Lit(nodes[0].value)
        operations = nodes[1:]
    else:
        return None

    # Apply operations in clause order
    for node in operations:
        op = node.operation
        if op in ("add", "subtract", "multiply", "divide"):
            ast = BinOp(op, ast, Lit(node.value))
        # Unknown operation — skip (don't guess)

    return ast


# =====================================================================
# Graph snapshots — retained for evaluation and learning
# =====================================================================

@dataclass
class GraphSnapshot:
    """
    A retained ephemeral graph for evaluation.

    Every problem produces a snapshot. Correct and wrong answers both
    contribute to learning. Wrong answers learn from right answers in
    the same equivalence class: what about the right one's graph was
    different?
    """
    problem_id: str
    problem_text: str
    graph: ProblemGraph
    walk_result: Optional[WalkResult]
    answer: Optional[float]         # what we computed
    expected: Optional[float]       # correct answer (if known)
    correct: bool
    equivalence_classes: list[str]  # from classify.py
    failure_type: str = ""          # process failure class (if wrong)
    solver_used: str = ""           # which solver produced the answer


# Process failure taxonomy
FAILURE_TYPES = {
    "UNIT_EXTRACTION":    "Wrong unit/per_unit extracted from text",
    "FALSE_RATE":         "Non-rate value tagged as rate",
    "OPERATION_DIRECTION": "Multiply vs divide confusion on rate",
    "WRONG_VERB_OP":      "Verb resolved to wrong math operation",
    "MISSING_NODE":       "Value in text not extracted as graph node",
    "ORPHAN_NODE":        "Node extracted but no operation edge",
    "WRONG_ENTITY":       "Value assigned to wrong entity",
    "UNIT_MISMATCH":      "Unit chain doesn't connect target to sources",
    "PARTIAL_WALK":       "Walk found a path but missed values",
}


def diagnose_failure(snapshot: GraphSnapshot) -> str:
    """
    Classify WHY a graph walk failed.

    Looks at the graph structure to identify the process failure,
    not just that the answer was wrong.
    """
    if snapshot.correct:
        return ""

    pg = snapshot.graph
    wr = snapshot.walk_result

    if pg is None:
        return "MISSING_NODE"

    # No walk result at all
    if wr is None:
        # Check if there are rate nodes — if so, walk should have found them
        if pg.nodes_with_per_unit():
            return "UNIT_MISMATCH"
        return "MISSING_NODE"

    # Walk ran but answer was wrong
    # Check rate issues
    rates = pg.nodes_with_per_unit()
    for r in rates:
        # Rate per_unit doesn't match any source node's unit
        source_match = any(
            _units_match(n.unit, r.per_unit)
            for n in pg.all_values()
        )
        if not source_match:
            return "UNIT_EXTRACTION"

        # Rate unit == per_unit (self-reference, nonsense rate)
        if r.unit and r.per_unit and _units_match(r.unit, r.per_unit):
            return "FALSE_RATE"

    # Check operation direction: multiply vs divide confusion.
    # If we multiplied by rate r but should have divided (or vice versa),
    # then answer/expected ≈ r² or expected/answer ≈ r².
    if snapshot.expected is not None and snapshot.answer is not None:
        if snapshot.answer > 0 and snapshot.expected > 0:
            ratio = snapshot.answer / snapshot.expected
            if abs(ratio - 1) > 0.01:
                for r in rates:
                    if r.value > 0 and r.value != 1:
                        r_sq = r.value ** 2
                        # answer/expected ≈ r² → multiplied when should divide
                        if r_sq > 0.01 and abs(ratio / r_sq - 1) < 0.15:
                            return "OPERATION_DIRECTION"
                        # expected/answer ≈ r² → divided when should multiply
                        inv_ratio = 1 / ratio
                        if r_sq > 0.01 and abs(inv_ratio / r_sq - 1) < 0.15:
                            return "OPERATION_DIRECTION"

    # Check coverage
    if wr.nodes_used < wr.nodes_total * 0.5:
        return "PARTIAL_WALK"

    # Check for orphan nodes (no operation)
    orphans = [n for n in pg.all_values() if n.operation == ""]
    if orphans:
        return "ORPHAN_NODE"

    return "WRONG_VERB_OP"  # default: verb→op mapping was probably wrong


def compare_snapshots(wrong: GraphSnapshot,
                      right: GraphSnapshot) -> dict:
    """
    Compare a wrong answer to a right answer in the same equivalence class.

    Returns a dict describing what the right graph did differently.
    This is the learning signal: not just "what failed" but "what succeeded
    in a similar problem and why."
    """
    diff = {
        "wrong_id": wrong.problem_id,
        "right_id": right.problem_id,
        "shared_classes": list(
            set(wrong.equivalence_classes) & set(right.equivalence_classes)),
        "differences": [],
    }

    wg, rg = wrong.graph, right.graph

    # Compare node counts
    w_rates = len(wg.nodes_with_per_unit())
    r_rates = len(rg.nodes_with_per_unit())
    if w_rates != r_rates:
        diff["differences"].append(
            f"rate nodes: wrong={w_rates}, right={r_rates}")

    # Compare operation coverage
    w_ops = sum(1 for n in wg.all_values()
                if n.operation in ("add", "subtract", "multiply", "divide"))
    r_ops = sum(1 for n in rg.all_values()
                if n.operation in ("add", "subtract", "multiply", "divide"))
    w_total = len(wg.all_values())
    r_total = len(rg.all_values())
    w_ratio = w_ops / w_total if w_total else 0
    r_ratio = r_ops / r_total if r_total else 0
    if abs(w_ratio - r_ratio) > 0.2:
        diff["differences"].append(
            f"op coverage: wrong={w_ratio:.0%}, right={r_ratio:.0%}")

    # Compare entity detection
    if len(wg.entities) != len(rg.entities):
        diff["differences"].append(
            f"entities: wrong={len(wg.entities)}, right={len(rg.entities)}")

    # Compare walk paths
    ww, rw = wrong.walk_result, right.walk_result
    if ww and rw:
        if ww.confidence != rw.confidence:
            diff["differences"].append(
                f"walk confidence: wrong={ww.confidence:.2f}, right={rw.confidence:.2f}")

    # Compare solver used
    if wrong.solver_used != right.solver_used:
        diff["differences"].append(
            f"solver: wrong={wrong.solver_used}, right={right.solver_used}")

    return diff


# Global snapshot store — retained across problems in a benchmark run
_SNAPSHOTS: list[GraphSnapshot] = []


def retain(snapshot: GraphSnapshot):
    """Store a snapshot for later evaluation."""
    _SNAPSHOTS.append(snapshot)


def get_snapshots() -> list[GraphSnapshot]:
    """Get all retained snapshots."""
    return _SNAPSHOTS


def clear_snapshots():
    """Clear retained snapshots (between benchmark runs)."""
    _SNAPSHOTS.clear()


def evaluate_run() -> dict:
    """
    Evaluate all retained snapshots as equivalence classes.

    Returns a report: for each equivalence class, what's the pass rate,
    what are the common failure types, and what do right answers have
    that wrong answers don't.
    """
    from collections import defaultdict

    by_class = defaultdict(lambda: {"correct": [], "wrong": []})

    for snap in _SNAPSHOTS:
        for cls in snap.equivalence_classes:
            if cls == "sequential":
                continue  # skip fallback class
            bucket = "correct" if snap.correct else "wrong"
            by_class[cls][bucket].append(snap)

    report = {}
    for cls, groups in sorted(by_class.items()):
        n_right = len(groups["correct"])
        n_wrong = len(groups["wrong"])
        n_total = n_right + n_wrong

        # Failure type distribution
        failure_dist = defaultdict(int)
        for snap in groups["wrong"]:
            if not snap.failure_type:
                snap.failure_type = diagnose_failure(snap)
            failure_dist[snap.failure_type] += 1

        # Cross-compare: for each wrong answer, find the most similar
        # right answer and diff them
        comparisons = []
        for wrong in groups["wrong"][:5]:  # limit to 5 for readability
            best_right = None
            best_overlap = 0
            for right in groups["correct"]:
                overlap = len(
                    set(wrong.equivalence_classes) &
                    set(right.equivalence_classes))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_right = right
            if best_right:
                comparisons.append(
                    compare_snapshots(wrong, best_right))

        report[cls] = {
            "total": n_total,
            "correct": n_right,
            "wrong": n_wrong,
            "pass_rate": n_right / n_total if n_total else 0,
            "failure_types": dict(failure_dist),
            "comparisons": comparisons,
        }

    return report
