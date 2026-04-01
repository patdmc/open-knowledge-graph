"""
MapReduce word problem solver.

Architecture:
  MAP:    Each sentence → (role, value, unit) emissions
  REDUCE: Schema template + slot filling → answer

Every problem uses the same canonical variables:
  a, b, c, ... = extracted numbers assigned to schema slots

Every 2-step problem follows one of 4 patterns:
  Scale→Combine:  intermediate = a ⊗ b;  answer = intermediate ⊕ c
  Combine→Scale:  intermediate = a ⊕ b;  answer = intermediate ⊗ c
  Scale→Scale:    intermediate = a ⊗ b;  answer = intermediate ⊗ c
  Combine→Combine: intermediate = a ⊕ b; answer = intermediate ⊕ c

Where ⊗ = {*, /} and ⊕ = {+, -}

The MAP step is per-sentence (parallelizable, bounded).
The REDUCE step is deterministic (template application).
"""

import re
from dataclasses import dataclass, field
from itertools import permutations


@dataclass
class Emission:
    """One MAP emission from a sentence."""
    value: float
    role: str       # 'quantity', 'rate', 'total', 'change', 'part', 'base', 'factor'
    unit: str = ""
    per_unit: str = ""
    sentence_idx: int = 0
    confidence: float = 0.5


# =====================================================================
# MAP: sentence → emissions
# =====================================================================

# Role detection patterns
_RATE_PATTERN = re.compile(
    r'(?:per|an?|each|every)\s+\w+', re.I)
_TOTAL_PATTERN = re.compile(
    r'\b(?:total|altogether|combined|in all|all together)\b', re.I)
_CHANGE_PATTERN = re.compile(
    r'\b(?:gave|lost|spent|ate|sold|earned|found|got|received|bought|'
    r'added|removed|took|used|picked|collected|made|baked|broke|'
    r'dropped|threw|donated|drank|consumed|more|fewer|less|extra)\b', re.I)
_COMPARE_PATTERN = re.compile(
    r'\b(?:more than|fewer than|less than|as many as|times as)\b', re.I)
_MULT_PATTERN = re.compile(
    r'\b(?:twice|double|triple|half|quarter|third|'
    r'(\d+)\s+times)\b', re.I)
_QUESTION_PATTERN = re.compile(r'\?')
_LEFT_PATTERN = re.compile(
    r'\b(?:left|remain|still|after)\b', re.I)


def map_sentence(sentence: str, sentence_idx: int,
                 numbers: list[tuple[float, str, str]]) -> list[Emission]:
    """
    MAP function: extract (role, value) emissions from one sentence.

    numbers: [(value, unit, per_unit), ...] already extracted from this sentence
    """
    s = sentence.lower()
    emissions = []

    is_question = bool(_QUESTION_PATTERN.search(sentence))

    for value, unit, per_unit in numbers:
        # Determine role from context
        if per_unit or _RATE_PATTERN.search(s):
            role = 'rate'
            conf = 0.8
        elif _MULT_PATTERN.search(s):
            role = 'factor'
            conf = 0.8
        elif is_question:
            # Numbers in question sentence are usually the target or a constraint
            if _TOTAL_PATTERN.search(s):
                role = 'total'
                conf = 0.7
            elif _LEFT_PATTERN.search(s):
                role = 'remainder'
                conf = 0.7
            else:
                role = 'constraint'
                conf = 0.6
        elif _CHANGE_PATTERN.search(s):
            role = 'change'
            conf = 0.7
        elif _TOTAL_PATTERN.search(s):
            role = 'total'
            conf = 0.7
        else:
            role = 'quantity'
            conf = 0.5

        emissions.append(Emission(
            value=value,
            role=role,
            unit=unit,
            per_unit=per_unit,
            sentence_idx=sentence_idx,
            confidence=conf,
        ))

    return emissions


def map_problem(text: str, tokens: list) -> list[Emission]:
    """
    MAP the entire problem: split into sentences, extract per-sentence emissions.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    all_emissions = []

    for si, sentence in enumerate(sentences):
        sent_start = text.find(sentence)
        if sent_start < 0:
            sent_start = 0
        sent_end = sent_start + len(sentence)

        # Find which tokens belong to this sentence
        sent_numbers = []
        for tok in tokens:
            if sent_start <= tok.position < sent_end:
                sent_numbers.append((tok.value, tok.unit, tok.per_unit))

        emissions = map_sentence(sentence, si, sent_numbers)
        all_emissions.extend(emissions)

    return emissions if len(sentences) == 1 else all_emissions


# =====================================================================
# REDUCE: emissions + schema → answer
# =====================================================================

# The 4 canonical 2-step patterns
# Each pattern is: (step0_op_class, step1_op_class)
# where op_class is 'scale' (*/÷) or 'combine' (+/-)

def _apply_op(a: float, b: float, op: str) -> float | None:
    """Apply a single operation."""
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0:
            return None
        return a / b
    return None


def reduce_1step(values: list[float], text: str = "") -> list[tuple[float, str, float]]:
    """
    REDUCE for 1-step problems.
    Try all pairs with all 4 ops. Rank by schema match.
    """
    candidates = []
    t = text.lower()

    # Detect likely schema from text
    wants_total = bool(re.search(r'\b(?:total|altogether|combined|in all)\b', t))
    wants_left = bool(re.search(r'\b(?:left|remain|still)\b', t))
    wants_each = bool(re.search(r'\b(?:each|per|every)\b', t))
    wants_diff = bool(re.search(r'how (?:many|much) (?:more|fewer|less)', t))

    for i, a in enumerate(values):
        for j, b in enumerate(values):
            if i == j:
                continue
            for op in ['+', '-', '*', '/']:
                result = _apply_op(a, b, op)
                if result is None:
                    continue

                # Score based on schema match
                score = 0.3  # base
                desc = f'{a}{op}{b}'

                if op == '+' and wants_total:
                    score = 0.8
                    desc = f'COMBINE({a}+{b})'
                elif op == '-' and wants_left:
                    score = 0.8
                    desc = f'CHANGE({a}-{b})'
                elif op == '-' and wants_diff:
                    score = 0.8
                    desc = f'COMPARE({a}-{b})'
                elif op == '*' and wants_each:
                    score = 0.7
                    desc = f'VARY({a}*{b})'
                elif op == '/' and wants_each:
                    score = 0.7
                    desc = f'VARY({a}/{b})'

                # Prefer positive results
                if result < 0:
                    score *= 0.5
                # Prefer integer results
                if result == int(result):
                    score *= 1.1

                candidates.append((result, desc, score))

    # Multi-addend: sum all, sum all minus each
    if len(values) >= 3 and wants_total:
        total = sum(values)
        candidates.append((total, f'COMBINE(sum_all={total})', 0.8))

    candidates.sort(key=lambda x: -x[2])
    return candidates


def reduce_2step(values: list[float], text: str = "") -> list[tuple[float, str, float]]:
    """
    REDUCE for 2-step problems.

    Try all 4 canonical patterns × all op combinations × all value assignments.
    Rank by schema match + heuristics.
    """
    candidates = []
    t = text.lower()

    # Schema signals
    wants_total = bool(re.search(r'\b(?:total|altogether|combined|in all)\b', t))
    wants_left = bool(re.search(r'\b(?:left|remain|still)\b', t))
    has_rate = bool(re.search(r'\b(?:per|each|every|an hour|a day|a week)\b', t))
    has_mult = bool(re.search(r'\b(?:twice|double|triple|half|times)\b', t))
    has_percent = bool(re.search(r'\d+\s*%|percent', t))

    # 4 patterns × 4 op combos = 16 templates
    scale_ops = ['*', '/']
    combine_ops = ['+', '-']

    patterns = [
        ('S→C', scale_ops, combine_ops),    # Scale then Combine
        ('C→S', combine_ops, scale_ops),    # Combine then Scale
        ('S→S', scale_ops, scale_ops),      # Scale then Scale
        ('C→C', combine_ops, combine_ops),  # Combine then Combine
    ]

    # Try all permutations of values as (a, b, c)
    # a,b go into step 0; intermediate + c go into step 1
    unique_vals = list(set(values))
    if len(unique_vals) < 2:
        return reduce_1step(values, text)

    # For 3+ values, try all 3-element permutations
    if len(unique_vals) >= 3:
        perms = list(permutations(unique_vals, 3))
    else:
        # 2 values: one must be reused
        perms = []
        for a in unique_vals:
            for b in unique_vals:
                for c in unique_vals:
                    perms.append((a, b, c))

    # Cap permutations to avoid explosion
    if len(perms) > 120:
        perms = perms[:120]

    for pat_name, ops0, ops1 in patterns:
        for op0 in ops0:
            for op1 in ops1:
                for a, b, c in perms:
                    # Step 0: intermediate = a op0 b
                    intermediate = _apply_op(a, b, op0)
                    if intermediate is None:
                        continue

                    # Step 1: answer = intermediate op1 c
                    answer = _apply_op(intermediate, c, op1)
                    if answer is None:
                        continue

                    # Also try: answer = c op1 intermediate (for non-commutative ops)
                    answer_rev = _apply_op(c, intermediate, op1)

                    # Score
                    score = 0.3
                    desc = f'{pat_name}:({a}{op0}{b}){op1}{c}'

                    # Schema-based scoring
                    if pat_name == 'S→C' and wants_total and op1 == '+':
                        score = 0.7
                    elif pat_name == 'S→C' and wants_left and op1 == '-':
                        score = 0.7
                    elif pat_name == 'C→S' and has_rate:
                        score = 0.65
                    elif pat_name == 'S→S' and (has_rate or has_mult or has_percent):
                        score = 0.65

                    # Prefer positive, integer results
                    if answer is not None and answer > 0:
                        score *= 1.1
                    if answer is not None and answer == int(answer):
                        score *= 1.05

                    if answer is not None:
                        candidates.append((answer, desc, score))
                    if answer_rev is not None and answer_rev != answer:
                        candidates.append((answer_rev,
                                           f'{pat_name}:{c}{op1}({a}{op0}{b})',
                                           score * 0.9))

    # Also include 1-step candidates as fallback
    candidates.extend(reduce_1step(values, text))

    candidates.sort(key=lambda x: -x[2])

    # Deduplicate by value (keep highest score)
    seen = set()
    deduped = []
    for val, desc, score in candidates:
        # Round to avoid float precision issues
        key = round(val, 6)
        if key not in seen:
            seen.add(key)
            deduped.append((val, desc, score))

    return deduped


# =====================================================================
# Main solver
# =====================================================================

def solve(text: str, tokens: list = None) -> list[tuple[float, str, float]]:
    """
    MapReduce solver: MAP sentences → emissions, REDUCE via schema templates.

    tokens: pre-extracted NumberTokens (from hub_scale.extract_numbers + expand_conversions)
    """
    # Import extraction if tokens not provided
    if tokens is None:
        from hub_scale import extract_numbers, expand_conversions
        tokens = extract_numbers(text)
        tokens = expand_conversions(tokens, text)

    # MAP: get emissions from ORIGINAL tokens only (not expansion noise)
    # Use tokens with position > 0 (real text tokens) or high confidence expansions
    core_tokens = [t for t in tokens
                   if t.position > 0 or t.confidence >= 0.8]
    if not core_tokens:
        core_tokens = tokens

    emissions = map_problem(text, core_tokens)

    # Get unique non-zero values — keep it tight
    values = list(set(t.value for t in core_tokens if t.value != 0))
    if not values:
        return []

    # Cap value count to prevent combinatorial explosion
    # Keep the N most "interesting" values (from text, not injected)
    if len(values) > 8:
        # Prefer text-extracted over injected
        text_vals = [t.value for t in core_tokens
                     if t.position > 0 and t.value != 0]
        inject_vals = [t.value for t in core_tokens
                       if t.position == 0 and t.value != 0 and t.confidence >= 0.8]
        values = list(set(text_vals))[:8]
        # Add injected only if room
        for v in inject_vals:
            if v not in values and len(values) < 8:
                values.append(v)

    # Choose reducer based on value count
    if len(values) <= 3:
        candidates = reduce_1step(values, text)
    else:
        candidates = reduce_2step(values, text)

    return candidates[:50]  # top 50


def solve_oracle(text: str, answer: float,
                 tokens: list = None) -> dict:
    """
    Solve and report whether any candidate matches the answer.
    Returns debug info.
    """
    candidates = solve(text, tokens)

    correct_rank = -1
    for i, (val, desc, score) in enumerate(candidates):
        if abs(val - answer) < 0.01 or \
           (answer != 0 and abs(val / answer - 1) < 0.001):
            correct_rank = i
            break

    return {
        'correct': correct_rank >= 0,
        'rank': correct_rank,
        'n_candidates': len(candidates),
        'top_candidate': candidates[0] if candidates else None,
        'correct_desc': candidates[correct_rank][1] if correct_rank >= 0 else None,
    }
