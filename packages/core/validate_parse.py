"""
packages/core/validate_parse.py — Lexical coverage validator for word problems.

The student (solver) solves the problem.
The validator checks his work: does the parse account for every piece?

Checks:
  1. Number coverage: every number in the text has a parsed token
  2. Entity coverage: every named entity is detected
  3. Relationship coverage: relational phrases have corresponding edges
  4. Rate coverage: every "per X" is captured

Coverage score IS the confidence.
If you only captured 3 of 5 numbers, confidence ≤ 0.6.
Ground truth, not a heuristic.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ParseValidation:
    """What the validator found."""
    # Numbers
    text_numbers: list[float] = field(default_factory=list)
    parsed_numbers: list[float] = field(default_factory=list)
    missing_numbers: list[float] = field(default_factory=list)
    number_coverage: float = 0.0

    # Entities
    text_entities: list[str] = field(default_factory=list)
    parsed_entities: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    entity_coverage: float = 0.0

    # Relationships (multiplicative, comparative, percentage, etc.)
    text_relations: list[str] = field(default_factory=list)
    parsed_relations: list[str] = field(default_factory=list)
    missing_relations: list[str] = field(default_factory=list)
    relation_coverage: float = 0.0

    # Rates (per X)
    text_rates: list[str] = field(default_factory=list)
    parsed_rates: list[str] = field(default_factory=list)
    missing_rates: list[str] = field(default_factory=list)
    rate_coverage: float = 0.0

    # Overall
    coverage: float = 0.0
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ground truth extractors — what's actually in the text
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
}

_ENTITY_STOP = {
    "the", "if", "she", "he", "how", "what", "in", "on", "at", "it",
    "one", "each", "they", "her", "his", "a", "an", "for", "but", "and",
    "or", "so", "yet", "from", "with", "by", "to", "of", "is", "are",
    "was", "were", "has", "had", "do", "does", "did", "will", "can",
    "may", "then", "than", "when", "where", "there", "here",
    "this", "that", "some", "all", "many", "much", "most", "every",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "after", "before", "during", "since", "until", "while",
}

_RELATION_PATTERNS = [
    (r'\b\d+\s+times?\s+(?:as\s+)?(?:many|much)\b', 'multiplicative'),
    (r'\b(?:twice|double|triple)\s+(?:as\s+)?(?:many|much|the)\b', 'multiplicative'),
    (r'\bhalf\s+(?:as\s+)?(?:many|much|of|the)\b', 'fraction'),
    (r'\b\d+\s+(?:more|fewer|less)\s+than\b', 'comparative'),
    (r'\b\d+(?:\.\d+)?\s*%', 'percentage'),
    (r'\b(?:split|divide|share)\s+(?:equally|evenly)\b', 'division'),
]


def _text_numbers(text: str) -> list[float]:
    """Extract all numbers from raw text."""
    numbers = []
    seen_positions = set()

    # Money: $X.XX
    for m in re.finditer(r'\$(\d[\d,]*\.?\d*)', text):
        val = float(m.group(1).replace(',', ''))
        numbers.append(val)
        seen_positions.update(range(m.start(), m.end()))

    # Fractions: X/Y
    for m in re.finditer(r'(\d+)\s*/\s*(\d+)', text):
        if not any(p in seen_positions for p in range(m.start(), m.end())):
            num, den = int(m.group(1)), int(m.group(2))
            if den != 0:
                numbers.append(num / den)
                seen_positions.update(range(m.start(), m.end()))

    # Digit numbers
    for m in re.finditer(r'(?<!\$)\b(\d[\d,]*\.?\d*)\b', text):
        if not any(p in seen_positions for p in range(m.start(), m.end())):
            val = float(m.group(1).replace(',', ''))
            numbers.append(val)

    # Number words
    text_lower = text.lower()
    for word, val in _NUMBER_WORDS.items():
        if re.search(rf'\b{word}\b', text_lower):
            numbers.append(float(val))

    return numbers


def _text_entities(text: str) -> list[str]:
    """Extract named entities from raw text."""
    entities = []
    # Find capitalized words (not at sentence start)
    for m in re.finditer(r'(?<=[.!?]\s)([A-Z][a-z]{2,})\b|(?<=\s)([A-Z][a-z]{2,})\b', text):
        name = (m.group(1) or m.group(2)).lower()
        if name not in _ENTITY_STOP and name not in entities:
            entities.append(name)
    # Also check first word if it appears elsewhere
    first_match = re.match(r'([A-Z][a-z]{2,})\b', text)
    if first_match:
        name = first_match.group(1).lower()
        if name not in _ENTITY_STOP and name not in entities:
            # Count occurrences
            count = len(re.findall(rf'\b{re.escape(first_match.group(1))}\b', text))
            if count > 1:
                entities.append(name)
    return entities


def _text_relations(text: str) -> list[str]:
    """Extract relational phrases."""
    relations = []
    text_lower = text.lower()
    for pattern, rel_type in _RELATION_PATTERNS:
        if re.search(pattern, text_lower):
            relations.append(rel_type)
    return relations


def _text_rates(text: str) -> list[str]:
    """Extract rate phrases (per X, each X, every X)."""
    rates = []
    for m in re.finditer(r'\b(?:per|each|every)\s+(\w+)', text.lower()):
        word = m.group(1)
        if word not in {'and', 'or', 'the', 'a', 'an', 'of'}:
            rates.append(word)
    return rates


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_parse(text: str, clauses, ctx) -> ParseValidation:
    """
    Validate that the parse covers the problem text.

    The student solved the problem. Now check his work:
    did he account for every number, entity, relationship, and rate?
    """
    v = ParseValidation()

    # --- Number coverage ---
    v.text_numbers = _text_numbers(text)
    for clause in clauses:
        for t in clause.tokens:
            if t.type in ("number", "money", "fraction", "multiplier"):
                v.parsed_numbers.append(t.value)

    if v.text_numbers:
        parsed_set = list(v.parsed_numbers)  # allow duplicates
        for n in v.text_numbers:
            found = False
            for i, p in enumerate(parsed_set):
                if abs(n - p) < 0.01:
                    parsed_set.pop(i)
                    found = True
                    break
            if not found:
                v.missing_numbers.append(n)
        captured = len(v.text_numbers) - len(v.missing_numbers)
        v.number_coverage = captured / len(v.text_numbers)
    else:
        v.number_coverage = 1.0

    # --- Entity coverage ---
    v.text_entities = _text_entities(text)
    pg = getattr(ctx, 'problem_graph', None)
    if pg:
        v.parsed_entities = list(pg.entities.keys())
    for e in v.text_entities:
        if e not in v.parsed_entities:
            v.missing_entities.append(e)
    if v.text_entities:
        captured = len(v.text_entities) - len(v.missing_entities)
        v.entity_coverage = max(0, captured / len(v.text_entities))
    else:
        v.entity_coverage = 1.0

    # --- Relationship coverage ---
    v.text_relations = _text_relations(text)
    for clause in clauses:
        if clause.has_reference:
            if 'multiply' in (clause.ref_type or ''):
                v.parsed_relations.append('multiplicative')
            elif 'add' in (clause.ref_type or ''):
                v.parsed_relations.append('comparative')
            elif 'subtract' in (clause.ref_type or ''):
                v.parsed_relations.append('comparative')
        # Check if percentage was detected
        for t in clause.tokens:
            if t.type == 'fraction' and t.value < 1:
                v.parsed_relations.append('fraction')
    # Check if percentage solver would fire
    full_text = " ".join(c.text for c in clauses).lower()
    if re.search(r'\d+\s*%', full_text):
        v.parsed_relations.append('percentage')

    for rel in v.text_relations:
        if rel not in v.parsed_relations:
            v.missing_relations.append(rel)
    if v.text_relations:
        captured = len(v.text_relations) - len(v.missing_relations)
        v.relation_coverage = max(0, captured / len(v.text_relations))
    else:
        v.relation_coverage = 1.0

    # --- Rate coverage ---
    v.text_rates = _text_rates(text)
    if pg:
        for n in pg.nodes.values():
            if n.per_unit:
                v.parsed_rates.append(n.per_unit)
    for rate in v.text_rates:
        matched = any(
            rate[:3] == p[:3] if len(p) >= 3 and len(rate) >= 3
            else rate == p
            for p in v.parsed_rates
        )
        if not matched:
            v.missing_rates.append(rate)
    if v.text_rates:
        captured = len(v.text_rates) - len(v.missing_rates)
        v.rate_coverage = max(0, captured / len(v.text_rates))
    else:
        v.rate_coverage = 1.0

    # --- Overall coverage (weighted) ---
    # Numbers are the most important signal
    v.coverage = (
        v.number_coverage * 0.5 +
        v.entity_coverage * 0.2 +
        v.relation_coverage * 0.2 +
        v.rate_coverage * 0.1
    )

    # Build issues list
    if v.missing_numbers:
        v.issues.append(f"missing numbers: {v.missing_numbers}")
    if v.missing_entities:
        v.issues.append(f"missing entities: {v.missing_entities}")
    if v.missing_relations:
        v.issues.append(f"missing relations: {v.missing_relations}")
    if v.missing_rates:
        v.issues.append(f"missing rates: {v.missing_rates}")

    return v
