"""
Word problem schema classifier.

~36 equivalence classes from Kintsch & Greeno (1985) and extensions.
Once classified, the computation graph is deterministic — the uncertainty
is in classification, not computation.

Each schema defines:
  - slots: named positions in the computation graph
  - graph: how slots connect (which operations, which order)
  - signals: language patterns that identify this schema
  - unknown: which slot the question asks for

The class IS the graph. Constants = nodes. Equations = edges.

Schema families:
  ADDITIVE (14 types): Join, Separate, Part-Part-Whole, Compare
  MULTIPLICATIVE (10+ types): Equal Groups, Rate, Multiplicative Compare,
    Cartesian Product, Array/Area
  ALGEBRAIC (12+ types): Rate/Distance/Time, Work/Rate, Mixture, Age,
    Percent, Proportion
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Slot:
    """A named position in a schema's computation graph."""
    name: str           # e.g., "start", "change", "result", "rate", "quantity"
    value: Optional[float] = None
    unit: str = ""
    is_unknown: bool = False


@dataclass
class Schema:
    """A word problem schema — the computation graph template."""
    family: str         # "additive", "multiplicative", "algebraic"
    type: str           # "join_result", "rate_total", "compare_diff", etc.
    slots: dict = field(default_factory=dict)  # name → Slot
    compute: str = ""   # computation rule: "result = start + change"
    confidence: float = 0.0
    signals: list = field(default_factory=list)  # which patterns matched


# =====================================================================
# Schema definitions — each is a graph template
# =====================================================================

# Additive schemas
_ADDITIVE_SCHEMAS = {
    # JOIN: start + change = result
    "join_result": {
        "slots": ["start", "change", "result"],
        "compute": "start + change",
        "unknown": "result",
        "signals": [
            r'\b(?:gets?|got|receives?|received|finds?|found|earns?|earned)\b',
            r'\b(?:more|additional|another|extra)\b',
        ],
    },
    "join_change": {
        "slots": ["start", "change", "result"],
        "compute": "result - start",
        "unknown": "change",
        "signals": [
            r'\bhow many (?:more|additional)\b',
        ],
    },
    "join_start": {
        "slots": ["start", "change", "result"],
        "compute": "result - change",
        "unknown": "start",
        "signals": [
            r'\bhad some\b',
            r'\bnow (?:has|have)\b.*\bgot\b',
        ],
    },

    # SEPARATE: start - change = result
    "separate_result": {
        "slots": ["start", "change", "result"],
        "compute": "start - change",
        "unknown": "result",
        "signals": [
            r'\b(?:gives?|gave|loses?|lost|spends?|spent|eats?|ate)\b',
            r'\b(?:left|remain|remaining)\b',
        ],
    },
    "separate_change": {
        "slots": ["start", "change", "result"],
        "compute": "start - result",
        "unknown": "change",
        "signals": [
            r'\bhow many (?:did|were)\b.*\b(?:give|lose|eat|spend)\b',
        ],
    },

    # PART-PART-WHOLE
    "ppw_whole": {
        "slots": ["part1", "part2", "whole"],
        "compute": "part1 + part2",
        "unknown": "whole",
        "signals": [
            r'\b(?:total|altogether|in all|combined|together)\b',
            r'\band\b.*\bhow many\b',
        ],
    },
    "ppw_part": {
        "slots": ["part1", "part2", "whole"],
        "compute": "whole - part1",
        "unknown": "part2",
        "signals": [
            r'\brest\b|\bremainder\b|\bother\b',
            r'\b\d+\s+(?:are|were|is)\b.*\bhow many\b',
        ],
    },

    # COMPARE
    "compare_diff": {
        "slots": ["bigger", "smaller", "difference"],
        "compute": "bigger - smaller",
        "unknown": "difference",
        "signals": [
            r'\bhow (?:many|much) (?:more|fewer|less)\b',
            r'\bsave\b.*\bbuy\b',  # savings = comparison
        ],
    },
    "compare_bigger": {
        "slots": ["bigger", "smaller", "difference"],
        "compute": "smaller + difference",
        "unknown": "bigger",
        "signals": [
            r'\b(?:more|older|taller|heavier|longer|bigger) than\b',
        ],
    },
    "compare_smaller": {
        "slots": ["bigger", "smaller", "difference"],
        "compute": "bigger - difference",
        "unknown": "smaller",
        "signals": [
            r'\b(?:fewer|younger|shorter|lighter|shorter|smaller|less) than\b',
        ],
    },
}

# Multiplicative schemas
_MULTIPLICATIVE_SCHEMAS = {
    # EQUAL GROUPS: groups * per_group = total
    "equal_groups_total": {
        "slots": ["groups", "per_group", "total"],
        "compute": "groups * per_group",
        "unknown": "total",
        "signals": [
            r'\b(?:each|every|per)\b',
            r'\b\d+\s+(?:groups?|boxes?|bags?|packs?|packages?|rows?|sets?)\b',
        ],
    },
    "equal_groups_size": {
        "slots": ["groups", "per_group", "total"],
        "compute": "total / groups",
        "unknown": "per_group",
        "signals": [
            r'\bhow (?:many|much) (?:in|per|each)\b',
            r'\b(?:split|share|divide|distribute)\b.*\b(?:equally|evenly)\b',
        ],
    },
    "equal_groups_count": {
        "slots": ["groups", "per_group", "total"],
        "compute": "total / per_group",
        "unknown": "groups",
        "signals": [
            r'\bhow many (?:groups?|boxes?|bags?|packs?|trips?)\b',
        ],
    },

    # RATE: rate * quantity = total_cost
    "rate_total": {
        "slots": ["rate", "quantity", "total"],
        "compute": "rate * quantity",
        "unknown": "total",
        "signals": [
            r'\$\d+.*\b(?:per|each|every)\b',
            r'\b(?:per|each|every)\b.*\$\d+',
            r'\bcosts?\b.*\$',
        ],
    },
    "rate_quantity": {
        "slots": ["rate", "quantity", "total"],
        "compute": "total / rate",
        "unknown": "quantity",
        "signals": [
            r'\bhow many\b.*\bcan\b.*\b(?:buy|afford|get)\b',
        ],
    },

    # MULTIPLICATIVE COMPARISON: "N times as many"
    "mult_compare": {
        "slots": ["base", "multiplier", "result"],
        "compute": "base * multiplier",
        "unknown": "result",
        "signals": [
            r'\b(?:twice|double|triple|thrice)\b',
            r'\b\d+\s+times\s+(?:as|more|less)\b',
            r'\bhalf\b',
        ],
    },

    # RATE COMPARISON: two rates, find difference
    "rate_compare": {
        "slots": ["rate1", "qty1", "rate2", "qty2", "difference"],
        "compute": "(rate1 * qty1) - (rate2 * qty2)",
        "unknown": "difference",
        "signals": [
            r'\bsave\b',
            r'\bor\b.*\b(?:per|each)\b',
            r'\bcheaper\b|\bexpensive\b',
        ],
    },
}

# Algebraic schemas
_ALGEBRAIC_SCHEMAS = {
    # MULTI-STEP ADDITIVE: chain of joins/separates
    "chain_add_sub": {
        "slots": ["values", "result"],
        "compute": "sum(values) with signs",
        "unknown": "result",
        "signals": [
            r'\bthen\b.*\bthen\b',  # multiple sequential events
            r'\band\b.*\band\b.*\bhow many\b',
        ],
    },

    # PERCENT
    "percent_of": {
        "slots": ["base", "percent", "result"],
        "compute": "base * percent / 100",
        "unknown": "result",
        "signals": [
            r'\b\d+%\b',
            r'\b\d+\s+percent\b',
        ],
    },
    "percent_change": {
        "slots": ["original", "percent", "new"],
        "compute": "original * (1 + percent/100)",
        "unknown": "new",
        "signals": [
            r'\b(?:increase|decrease|raise|reduce)\b.*\b\d+%\b',
            r'\b\d+%\s+(?:more|less|increase|decrease|off|discount)\b',
        ],
    },

    # PROPORTION: a/b = c/d
    "proportion": {
        "slots": ["a", "b", "c", "d"],
        "compute": "a * d / b (or b * c / a)",
        "unknown": "d",
        "signals": [
            r'\bfor every\b',
            r'\bratio\b',
            r'\bif\b.*\bthen how\b',
        ],
    },

    # MULTI-STEP RATE: rate applied to computed quantity
    "rate_chain": {
        "slots": ["rate", "sub_quantity", "sub_result", "total"],
        "compute": "(sub_result) * rate or (sub_result) / rate",
        "unknown": "total",
        "signals": [
            r'\b(?:per|each|every)\b.*\b(?:total|altogether)\b',
        ],
    },
}

ALL_SCHEMAS = {}
ALL_SCHEMAS.update(_ADDITIVE_SCHEMAS)
ALL_SCHEMAS.update(_MULTIPLICATIVE_SCHEMAS)
ALL_SCHEMAS.update(_ALGEBRAIC_SCHEMAS)


def classify(text: str, clauses: list = None) -> list[tuple[str, float, list]]:
    """
    Classify a word problem text into schema types.

    Returns list of (schema_type, confidence, matched_signals)
    sorted by confidence descending.
    """
    text_lower = text.lower()
    matches = []

    for schema_type, schema_def in ALL_SCHEMAS.items():
        signals = schema_def.get("signals", [])
        matched = []
        for pattern in signals:
            if re.search(pattern, text_lower):
                matched.append(pattern)

        if matched:
            # Confidence = fraction of signals that matched
            confidence = len(matched) / len(signals) if signals else 0
            # Require at least 50% of signals to match (reduces noise)
            if confidence >= 0.5:
                matches.append((schema_type, confidence, matched))

    # Sort by confidence (most signals matched)
    matches.sort(key=lambda x: -x[1])
    return matches


def generate_interpretations(text: str, numbers: list[tuple]) -> list[tuple]:
    """
    Generate candidate answers by matching schemas to numbers.

    Uses clause structure and provenance to assign numbers to schema
    slots, not brute-force enumeration. The clause order + comparison
    words + rate patterns determine which numbers go where.

    numbers: list of (value, unit, per_unit, clause_idx, token_idx)
    Returns list of (value, schema_type, confidence, description)
    """
    matches = classify(text)
    if not matches:
        return []

    results = []
    seen = set()
    text_lower = text.lower()

    vals = [num[0] for num in numbers]
    units = [num[1] for num in numbers]
    per_units = [num[2] for num in numbers]
    clause_ids = [num[3] for num in numbers]
    n = len(vals)

    # Count real numbers (non-implicit)
    real_count = sum(1 for i in range(n) if clause_ids[i] >= 0)

    def _add(val, schema, conf, desc, nums_used=2):
        if val is None:
            return
        try:
            if abs(val) == float('inf') or val != val:
                return
        except (OverflowError, ValueError):
            return
        rkey = round(val, 4)
        if rkey in seen:
            return
        seen.add(rkey)
        # Boost confidence by coverage: schemas using more numbers rank higher
        coverage = min(nums_used / max(real_count, 1), 1.0)
        adjusted_conf = conf * (0.5 + 0.5 * coverage)
        results.append((val, schema, adjusted_conf, desc))

    # --- Identify number roles from language ---
    rates = []     # indices of numbers that are rates (have per_unit)
    quantities = []  # indices of numbers that are quantities

    for idx in range(n):
        if per_units[idx]:
            rates.append(idx)
        elif units[idx] in ("dollars", "cents"):
            rates.append(idx)  # prices are often rates
        else:
            quantities.append(idx)

    # --- Per-clause direction detection for comparisons ---
    # Split text into sentences and detect comparison direction per clause
    sentences = re.split(r'[.!?]+', text_lower)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Map each number to its clause's comparison direction
    # +1 = "more/older/taller", -1 = "less/younger/shorter", 0 = neutral
    _MORE_RE = re.compile(r'\b(?:more|older|taller|heavier|longer|bigger|greater|faster|wider|higher)\b')
    _LESS_RE = re.compile(r'\b(?:less|fewer|younger|shorter|lighter|smaller|slower|lower)\b')

    clause_direction = {}  # clause_idx → +1 or -1
    for ci, sent in enumerate(sentences):
        if _MORE_RE.search(sent):
            clause_direction[ci] = +1
        elif _LESS_RE.search(sent):
            clause_direction[ci] = -1

    for schema_type, confidence, signals in matches:
        schema_def = ALL_SCHEMAS.get(schema_type, {})

        # --- Compare schemas: use per-clause direction ---
        if schema_type in ("compare_bigger", "compare_smaller",
                            "compare_diff"):
            if n >= 2:
                # Find reference: the number in a clause with NO comparison
                # word (state clause like "Quinn is 30")
                ref_candidates = [i for i in range(n)
                                  if clause_ids[i] not in clause_direction]
                offset_candidates = [i for i in range(n)
                                     if clause_ids[i] in clause_direction]

                # Fallback: if no neutral clause found, largest = reference
                if not ref_candidates:
                    sorted_by_val = sorted(range(n), key=lambda i: -vals[i])
                    ref_candidates = [sorted_by_val[0]]
                    offset_candidates = [i for i in range(n)
                                         if i != sorted_by_val[0]]

                ref_idx = ref_candidates[0]

                # Apply offsets using each clause's own direction
                current = vals[ref_idx]
                desc_parts = [str(vals[ref_idx])]

                for oi in offset_candidates:
                    ci = clause_ids[oi]
                    direction = clause_direction.get(ci, +1)
                    current = current + direction * vals[oi]
                    sign = "+" if direction > 0 else "-"
                    desc_parts.append(f"{sign}{vals[oi]}")

                _add(current, "compare_chain", confidence,
                     f"compare_chain({'→'.join(desc_parts)})",
                     nums_used=len(desc_parts))

                # Also: simple pairwise for 2-number compare
                for oi in offset_candidates:
                    ci = clause_ids[oi]
                    direction = clause_direction.get(ci, +1)
                    _add(vals[ref_idx] + direction * vals[oi],
                         schema_type, confidence * 0.9,
                         f"{schema_type}({vals[ref_idx]}{'+' if direction > 0 else '-'}{vals[oi]})")

                # No brute-force fallback — if clause direction
                # doesn't resolve it, we don't guess

        # --- Separate/Join: use clause order ---
        elif schema_type in ("separate_result", "join_result"):
            if n >= 2:
                # Start with first number, apply changes in clause order
                start = vals[0]
                current = start
                for idx in range(1, n):
                    if schema_type == "separate_result":
                        current = current - vals[idx]
                    else:
                        current = current + vals[idx]
                _add(current, schema_type, confidence,
                     f"{schema_type}(sequential)")

                # No reversed fallback — clause order is the signal

        # --- PPW: sum parts ---
        elif schema_type == "ppw_whole":
            if n >= 2:
                total = sum(vals)
                _add(total, schema_type, confidence, f"ppw_whole(sum_all)",
                     nums_used=n)
                # No pairwise subsets — sum all or nothing

        elif schema_type == "ppw_part":
            if n >= 2:
                # Whole minus parts
                largest = max(vals)
                remainder = largest - sum(v for v in vals if v != largest)
                _add(remainder, schema_type, confidence,
                     f"ppw_part({largest}-rest)")

        # --- Rate: match rate to quantity via units ---
        elif schema_type in ("rate_total", "equal_groups_total"):
            if rates and quantities:
                # Rate * quantity — only provenance-matched pairs
                for ri in rates:
                    for qi in quantities:
                        result = vals[ri] * vals[qi]
                        _add(result, schema_type, confidence,
                             f"{schema_type}({vals[ri]}*{vals[qi]})")
            # No brute-force all-pairs multiply

        elif schema_type in ("rate_quantity", "equal_groups_size",
                              "equal_groups_count"):
            # Divide: largest by each smaller, or total by rate
            if n >= 2 and rates and quantities:
                for qi in quantities:
                    for ri in rates:
                        if vals[ri] != 0:
                            _add(vals[qi] / vals[ri], schema_type,
                                 confidence * 0.8,
                                 f"{schema_type}({vals[qi]}/{vals[ri]})")

        # --- Multiplicative comparison ---
        elif schema_type == "mult_compare":
            if n >= 2:
                # Find the multiplier (small number or "twice"=2, "half"=0.5)
                for i in range(n):
                    if vals[i] in (2, 3, 4, 5, 0.5, 0.25, 0.333333):
                        for j in range(n):
                            if i == j:
                                continue
                            _add(vals[i] * vals[j], schema_type, confidence,
                                 f"mult_compare({vals[i]}*{vals[j]})")
                            if vals[i] != 0:
                                _add(vals[j] / vals[i], schema_type,
                                     confidence * 0.8,
                                     f"mult_compare({vals[j]}/{vals[i]})")

        # --- Percent ---
        elif schema_type in ("percent_of", "percent_change"):
            for i in range(n):
                if 0 < vals[i] <= 100:  # likely a percentage
                    for j in range(n):
                        if i == j:
                            continue
                        _add(vals[j] * vals[i] / 100, "percent_of",
                             confidence, f"percent({vals[j]}*{vals[i]}%)")
                        _add(vals[j] * (1 + vals[i] / 100), "percent_change",
                             confidence, f"pct_change({vals[j]}+{vals[i]}%)")
                        _add(vals[j] * (1 - vals[i] / 100), "percent_change",
                             confidence, f"pct_change({vals[j]}-{vals[i]}%)")

        # --- Rate compare: parallel paths then subtract ---
        elif schema_type == "rate_compare":
            # Need 4+ numbers: rate1, qty1, rate2, qty2
            # Group by clause: numbers in same clause are rate-qty pairs
            by_clause = {}
            for idx in range(n):
                ci = clause_ids[idx]
                if ci not in by_clause:
                    by_clause[ci] = []
                by_clause[ci].append(idx)

            clause_products = []
            for ci, indices in sorted(by_clause.items()):
                if len(indices) >= 2:
                    # Multiply pair in same clause
                    product = vals[indices[0]] * vals[indices[1]]
                    clause_products.append(product)

            if len(clause_products) >= 2:
                _add(abs(clause_products[0] - clause_products[1]),
                     schema_type, confidence,
                     f"rate_compare({clause_products[0]}-{clause_products[1]})")

        # --- Multi-step compositions ---

        # Pattern: separate_then_rate
        # "Start with X, lose A, lose B, sell remainder at $R each"
        # = (X - A - B) * R
        # Key: the "price rate" (dollars/cents per unit) is the final
        # multiplier. Everything else is the subtraction chain.
        if schema_type == "separate_result" and n >= 3:
            # Price rates: have dollar/cent unit (the conversion factor)
            price_rates = [i for i in range(n)
                           if units[i] in ("dollars", "cents", "dollar")
                           or (per_units[i] and units[i] in
                               ("dollars", "cents", "dollar"))]
            # If no explicit price rate, use any rate that isn't first
            if not price_rates:
                price_rates = [i for i in rates if i != 0]

            # Chain: non-price, non-implicit numbers
            # Skip implicit numbers (clause_idx < 0) and multiplicative
            # identities (value=1 with no unit)
            chain = [i for i in range(n)
                     if i not in price_rates
                     and clause_ids[i] >= 0
                     and not (vals[i] == 1 and not units[i])]

            if chain and price_rates:
                remainder = vals[chain[0]]
                parts = [str(vals[chain[0]])]
                for qi in chain[1:]:
                    remainder -= vals[qi]
                    parts.append(f"-{vals[qi]}")
                for ri in price_rates:
                    result = remainder * vals[ri]
                    _add(result, "separate_then_rate", confidence,
                         f"({' '.join(parts)})*{vals[ri]}={result}",
                         nums_used=len(chain) + 1)

        # Pattern: join_then_rate (similar but adding)
        if schema_type == "join_result" and n >= 3:
            price_rates = [i for i in range(n)
                           if units[i] in ("dollars", "cents", "dollar")]
            if not price_rates:
                price_rates = [i for i in rates if i != 0]
            chain = [i for i in range(n)
                     if i not in price_rates
                     and clause_ids[i] >= 0
                     and not (vals[i] == 1 and not units[i])]
            if chain and price_rates:
                total = vals[chain[0]]
                for qi in chain[1:]:
                    total += vals[qi]
                for ri in price_rates:
                    result = total * vals[ri]
                    _add(result, "join_then_rate", confidence,
                         f"join_then_rate({total}*{vals[ri]})",
                         nums_used=len(chain) + 1)

        # Generic chain: (nums[0] op nums[1]) op2 nums[2]
        if n >= 3 and schema_type in ("separate_result", "join_result",
                                        "rate_total"):
            for op1_type in (schema_type,):
                inter = _compute_2slot(op1_type, vals[0], vals[1])
                if inter is None:
                    continue
                for inner_schema, inner_conf, _ in matches[:3]:
                    result = _compute_2slot(inner_schema, inter, vals[2])
                    if result is not None:
                        _add(result, f"{schema_type}+{inner_schema}",
                             confidence * 0.7,
                             f"chain({vals[0]}op{vals[1]})op{vals[2]}")
                    if n >= 4:
                        result3 = _compute_2slot(inner_schema, inter, vals[3])
                        if result3 is not None:
                            _add(result3, f"{schema_type}+{inner_schema}",
                                 confidence * 0.6,
                                 f"chain({vals[0]}op{vals[1]})op{vals[3]}")

    return results


def _compute_2slot(schema_type: str, a: float, b: float) -> Optional[float]:
    """
    Compute the result for a 2-input schema.
    The schema determines the operation.
    """
    try:
        # Map schema to operation
        if schema_type in ("join_result", "ppw_whole", "compare_bigger"):
            return a + b
        elif schema_type in ("separate_result", "ppw_part", "compare_diff",
                              "compare_smaller", "join_change", "join_start",
                              "separate_change"):
            return a - b
        elif schema_type in ("equal_groups_total", "rate_total",
                              "mult_compare", "percent_of", "percent_change",
                              "rate_chain"):
            return a * b
        elif schema_type in ("equal_groups_size", "equal_groups_count",
                              "rate_quantity", "proportion"):
            return a / b if b != 0 else None
        elif schema_type == "rate_compare":
            # This needs 4 inputs, handled in multi-step
            return a - b
        elif schema_type == "chain_add_sub":
            return a + b  # or a - b — both are valid
        else:
            return None
    except (OverflowError, ZeroDivisionError):
        return None
