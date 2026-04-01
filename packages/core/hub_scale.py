"""
Hub+Scale word problem solver.

Every computation tree reduces to two operations:
  ADD   — collapse a hub (combine same-unit values)
  SCALE — multiply by any factor (including -1 for subtract, 1/n for divide)

The classification problem is: for each number in a problem, is it a
scale factor (rate, multiplier, divisor) or a hub member (quantity that
gets added/subtracted with other quantities)?

Architecture:
  1. Extract numbers with provenance (value, unit, per_unit, clause, position)
  2. Classify each number as SCALE or HUB (the hard part — bounded ML)
  3. Build hub topology (which hub members group together)
  4. Execute (deterministic: collapse each hub, apply scale edges)
  5. Validate answer with LLM (does this make sense?)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NumberToken:
    """A number extracted from a word problem with its provenance."""
    value: float
    text: str           # original text ("$12", "50 minutes", "3")
    position: int       # char offset in problem text
    clause_idx: int     # which sentence (0-based)
    # Provenance
    unit: str = ""      # "dollars", "minutes", "eggs"
    per_unit: str = ""  # "hour", "day" — if set, this is a RATE
    # Role classification
    role: str = ""      # "scale" or "hub" (the decision)
    confidence: float = 0.0
    # Hub assignment
    hub_id: int = -1    # which hub this belongs to


@dataclass
class ScaleEdge:
    """A scale connection between hubs or within a computation."""
    factor: float       # what to multiply by
    source: str = ""    # description of why this scale exists
    is_inverse: bool = False  # True if this is 1/x (divide) or -1 (subtract)


@dataclass
class Hub:
    """A group of same-unit numbers that collapse via addition."""
    hub_id: int
    members: list = field(default_factory=list)  # list of NumberToken
    collapsed_value: Optional[float] = None
    unit: str = ""


# =====================================================================
# Number extraction
# =====================================================================

# Rate indicators: words that signal per-unit relationships
_RATE_AFTER = re.compile(
    r'^\s*(?:per|an?|each|every)\s+'
    r'(?:\w+\s+)*?'  # optional adjectives ("fresh duck")
    r'(second|hour|minute|day|week|month|year|pound|gallon|mile|liter|'
    r'piece|item|person|student|child|box|bag|pack|trip|game|'
    r'lawn|driveway|window|page|chapter|shirt|ticket|slice|'
    r'egg|apple|unit|serving|cup|load|towel|painting|flower|'
    r'kilogram|gram|meter|kilometer|ounce|yard|foot)',
    re.IGNORECASE
)

_RATE_BEFORE = re.compile(
    r'\b(?:per|each|every)\s*$',
    re.IGNORECASE
)

# Unit extraction: what comes right after the number
_UNIT_AFTER = re.compile(
    r'^\s*(dollars?|cents?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|'
    r'eggs?|apples?|clips?|stamps?|trees?|books?|people|students?|'
    r'children|friends?|balls?|hats?|boxes?|bags?|packs?|pieces?|'
    r'slices?|dogs?|cats?|miles?|feet|foot|meters?|inches?|pounds?|'
    r'yards?|ounces?|'
    r'kilograms?|grams?|liters?|milliliters?|gallons?|cups?|shirts?|tickets?|'
    r'lawns?|driveways?|windows?|pages?|paintings?|roses?|tulips?|'
    r'daisies?|flowers?|pizzas?|towels?|loads?|percent|'
    r'centimeters?|kilometers?|cm|kg|km|ml)\b',
    re.IGNORECASE
)

# Standalone percent sign (not a word char, so needs separate pattern)
_PERCENT_SIGN = re.compile(r'^\s*%')

_DOLLAR_BEFORE = re.compile(r'\$\s*$')

# Scale factor signals
_MULTIPLIER_CONTEXT = re.compile(
    r'\b(?:times|twice|double|triple|half|quarter|third)\b',
    re.IGNORECASE
)

_PERCENT_AFTER = re.compile(r'^\s*(%|percent)\b')

# Deterministic unit conversions — known edges in the language graph
# Maps (from_unit, to_unit) → scale factor
UNIT_CONVERSIONS = {
    ('second', 'minute'): 1/60,
    ('minute', 'second'): 60,
    ('minute', 'hour'): 1/60,
    ('hour', 'minute'): 60,
    ('hour', 'day'): 1/24,
    ('day', 'hour'): 24,
    ('day', 'week'): 1/7,
    ('week', 'day'): 7,
    ('week', 'month'): 1/4,
    ('month', 'week'): 4,
    ('day', 'month'): 1/30,
    ('month', 'day'): 30,
    ('month', 'year'): 1/12,
    ('year', 'month'): 12,
    ('week', 'year'): 1/52,
    ('year', 'week'): 52,
    ('year', 'day'): 365,
    ('day', 'year'): 1/365,
    ('second', 'hour'): 1/3600,
    ('hour', 'second'): 3600,
    ('cent', 'dollar'): 1/100,
    ('dollar', 'cent'): 100,
    ('inch', 'foot'): 1/12,
    ('foot', 'inch'): 12,
    ('foot', 'yard'): 1/3,
    ('yard', 'foot'): 3,
    ('ounce', 'pound'): 1/16,
    ('pound', 'ounce'): 16,
    ('centimeter', 'meter'): 1/100,
    ('meter', 'centimeter'): 100,
    ('meter', 'kilometer'): 1/1000,
    ('kilometer', 'meter'): 1000,
    ('gram', 'kilogram'): 1/1000,
    ('kilogram', 'gram'): 1000,
    ('dozen', 'unit'): 12,
    ('liter', 'milliliter'): 1000,
    ('milliliter', 'liter'): 1/1000,
}


def get_conversion(from_unit: str, to_unit: str) -> float | None:
    """Get deterministic conversion factor between units."""
    f = from_unit.rstrip('s')
    t = to_unit.rstrip('s')
    if f == t:
        return 1.0
    return UNIT_CONVERSIONS.get((f, t))


_WRITTEN_NUMBERS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
    'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
    'half': 0.5, 'quarter': 0.25, 'third': 1/3,
    'fourth': 0.25, 'fifth': 0.2, 'sixth': 1/6,
    'seventh': 1/7, 'eighth': 0.125, 'ninth': 1/9, 'tenth': 0.1,
    # Multiplier words
    'twice': 2, 'double': 2, 'doubled': 2, 'triple': 3, 'tripled': 3,
    'thrice': 3, 'quadrupled': 4,
    # Grouping words
    'dozen': 12, 'pair': 2, 'couple': 2, 'both': 2,
    # Fractional quantifiers
    'halfway': 0.5,
    # Coin values (in cents)
    'nickel': 5, 'dime': 10,
}

# Compound written numbers: "thirty five", "one hundred twenty three"
# Use word2number library for robust parsing
try:
    from word2number import w2n as _w2n

    # Pattern for multi-word number phrases
    _COMPOUND_NUM_WORDS = (
        r'\b(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
        r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
        r'eighty|ninety|hundred|thousand|million)\s*)+'
    )
    _COMPOUND_NUM_RE = re.compile(_COMPOUND_NUM_WORDS, re.IGNORECASE)

    def _parse_written_number(text: str) -> float | None:
        """Parse a written number phrase using word2number."""
        try:
            return float(_w2n.word_to_num(text.strip()))
        except (ValueError, IndexError):
            return None
except ImportError:
    _COMPOUND_NUM_RE = None

    def _parse_written_number(text: str) -> float | None:
        return None

_WRITTEN_NUM_RE = re.compile(
    r'\b(' + '|'.join(_WRITTEN_NUMBERS.keys()) + r')\b',
    re.IGNORECASE
)

# Rate from verb: "$15 to mow", "$7 to shovel"
_RATE_VERB = re.compile(
    r'^\s*(?:to|for)\s+(?:mow|shovel|wash|clean|paint|fix|cut|'
    r'make|build|cook|drive|walk|ride|run|swim|fly|ship|deliver)\b',
    re.IGNORECASE
)


def extract_numbers(text: str) -> list[NumberToken]:
    """Extract numbers from problem text with provenance."""
    tokens = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for ci, sentence in enumerate(sentences):
        # Find sentence offset in full text
        sent_start = text.find(sentence)
        if sent_start < 0:
            sent_start = 0

        # --- Fractions: "1/4", "3/8", mixed numbers "3 3/8" ---
        fraction_spans = set()
        for m in re.finditer(
                r'(\d+)\s+(\d+)\s*/\s*(\d+)|(\d+)\s*/\s*(\d+)', sentence):
            if m.group(1):  # mixed number: "3 3/8"
                whole = int(m.group(1))
                numer = int(m.group(2))
                denom = int(m.group(3))
                val = whole + numer / denom
            else:  # simple fraction: "1/4"
                numer = int(m.group(4))
                denom = int(m.group(5))
                val = numer / denom
            abs_pos = sent_start + m.start()
            num_end = sent_start + m.end()
            fraction_spans.add((m.start(), m.end()))

            after = text[num_end:].lower()
            unit = ""
            unit_m = _UNIT_AFTER.match(after)
            if unit_m:
                u = unit_m.group(1).lower().rstrip('s')
                unit = u

            tok = NumberToken(
                value=val, text=m.group(), position=abs_pos,
                clause_idx=ci, unit=unit, per_unit="",
            )
            tokens.append(tok)

        # --- Comma-separated integers: "1,200,000" ---
        comma_spans = set()
        for m in re.finditer(r'\b(\d{1,3}(?:,\d{3})+)\b', sentence):
            val = float(m.group().replace(',', ''))
            abs_pos = sent_start + m.start()
            num_end = sent_start + m.end()
            comma_spans.add((m.start(), m.end()))

            after = text[num_end:].lower()
            before = text[:abs_pos].lower()
            unit = ""
            if _DOLLAR_BEFORE.search(before):
                unit = "dollars"
            unit_m = _UNIT_AFTER.match(after)
            if unit_m and not unit:
                u = unit_m.group(1).lower().rstrip('s')
                unit = u

            tok = NumberToken(
                value=val, text=m.group(), position=abs_pos,
                clause_idx=ci, unit=unit, per_unit="",
            )
            tokens.append(tok)

        # --- Numbers glued to metric/abbreviated units: "10ml", "80cm", "15kg" ---
        # These have no word boundary between digit and unit letter, so the
        # standard \b digit regex misses them.
        glued_spans = set()
        _GLUED_UNIT_RE = re.compile(
            r'(\d+(?:\.\d+)?)\s*'
            r'(ml|cm[³²]?|kg|km|mg|mm|mph|kph|lb|oz|ft|hr|'
            r'sec|min|m|g|c|k|K|p)\b',
        )
        for m in _GLUED_UNIT_RE.finditer(sentence):
            # Avoid matching inside fractions/commas
            if any(s <= m.start() < e for s, e in fraction_spans):
                continue
            if any(s <= m.start() < e for s, e in comma_spans):
                continue
            val = float(m.group(1))
            abs_pos = sent_start + m.start()
            num_end = sent_start + m.end()
            glued_spans.add((m.start(), m.end()))

            # Map abbreviated unit to canonical form
            unit_abbr = m.group(2).lower().rstrip('²³')
            _ABBR_MAP = {
                'ml': 'milliliter', 'cm': 'centimeter', 'kg': 'kilogram',
                'km': 'kilometer', 'mg': 'milligram', 'mm': 'millimeter',
                'mph': 'mile', 'kph': 'kilometer', 'lb': 'pound',
                'oz': 'ounce', 'ft': 'foot', 'hr': 'hour',
                'sec': 'second', 'min': 'minute', 'm': 'meter',
                'g': 'gram', 'c': 'cent', 'k': 'dollar',
                'p': 'cent',
            }
            unit = _ABBR_MAP.get(unit_abbr, '')

            # "$Xk" means X thousand dollars
            before = text[:abs_pos].lower()
            if unit_abbr == 'k' and _DOLLAR_BEFORE.search(before):
                val = val * 1000
                unit = 'dollars'

            tok = NumberToken(
                value=val, text=m.group(), position=abs_pos,
                clause_idx=ci, unit=unit, per_unit="",
            )
            tokens.append(tok)

        # --- Ordinal numbers: "12th", "3rd", "1st", "2nd" ---
        ordinal_spans = set()
        for m in re.finditer(r'(\d+)(?:st|nd|rd|th)\b', sentence):
            if any(s <= m.start() < e for s, e in fraction_spans):
                continue
            if any(s <= m.start() < e for s, e in comma_spans):
                continue
            if any(s <= m.start() < e for s, e in glued_spans):
                continue
            val = float(m.group(1))
            abs_pos = sent_start + m.start()
            ordinal_spans.add((m.start(), m.end()))
            tok = NumberToken(
                value=val, text=m.group(), position=abs_pos,
                clause_idx=ci, unit='', per_unit="",
            )
            tokens.append(tok)

        # --- Dollar-dot amounts: "$.25" → 0.25 dollars ---
        dollar_dot_spans = set()
        for m in re.finditer(r'\$\.(\d+)', sentence):
            val = float('0.' + m.group(1))
            abs_pos = sent_start + m.start()
            dollar_dot_spans.add((m.start(), m.end()))
            tok = NumberToken(
                value=val, text=m.group(), position=abs_pos,
                clause_idx=ci, unit='dollars', per_unit="",
            )
            tokens.append(tok)

        # Digit numbers (skip positions already handled by fractions/commas/glued/ordinals)
        for m in re.finditer(r'\b(\d+\.?\d*)\b', sentence):
            # Skip if this position is inside a fraction or comma number span
            if any(s <= m.start() < e for s, e in fraction_spans):
                continue
            if any(s <= m.start() < e for s, e in comma_spans):
                continue
            if any(s <= m.start() < e for s, e in glued_spans):
                continue
            if any(s <= m.start() < e for s, e in ordinal_spans):
                continue
            if any(s <= m.start() < e for s, e in dollar_dot_spans):
                continue
            val = float(m.group())
            abs_pos = sent_start + m.start()
            num_end = sent_start + m.end()

            # Context windows
            before = text[:abs_pos].lower()
            after = text[num_end:].lower()

            # Detect unit
            unit = ""
            per_unit = ""

            # Dollar sign before number
            if _DOLLAR_BEFORE.search(before):
                unit = "dollars"

            # Unit after number (check % separately since it's not a word char)
            unit_m = _UNIT_AFTER.match(after)
            if not unit_m:
                unit_m = _PERCENT_SIGN.match(after)
            if unit_m:
                matched = unit_m.group().strip()
                if matched == '%' or matched.lower() == 'percent':
                    unit = "percent"
                else:
                    u = matched.lower().rstrip('s')
                    if not unit:  # don't overwrite dollar
                        unit = u

            # Rate: "per X" after the unit
            after_unit = after[unit_m.end():] if unit_m else after
            rate_m = _RATE_AFTER.match(after_unit)
            if rate_m:
                per_unit = rate_m.group(1).lower().rstrip('s')
            elif _RATE_AFTER.match(after):
                rate_m2 = _RATE_AFTER.match(after)
                if rate_m2:
                    per_unit = rate_m2.group(1).lower().rstrip('s')

            # Rate from verb: "$15 to mow" means $15/lawn
            if not per_unit:
                verb_rate = _RATE_VERB.match(after_unit if unit_m else after)
                if verb_rate:
                    per_unit = "task"

            # Rate before: "each $5"
            if _RATE_BEFORE.search(before) and not per_unit:
                per_unit = "each"

            tok = NumberToken(
                value=val,
                text=m.group(),
                position=abs_pos,
                clause_idx=ci,
                unit=unit,
                per_unit=per_unit,
            )
            tokens.append(tok)

        # Compound written numbers first ("thirty five", "one hundred twenty")
        # These take priority over single-word matches
        compound_spans = set()  # track positions to avoid double-counting
        if _COMPOUND_NUM_RE is not None:
            for m in _COMPOUND_NUM_RE.finditer(sentence):
                phrase = m.group().strip()
                # Skip single words (handled below with more context)
                if ' ' not in phrase:
                    continue
                val = _parse_written_number(phrase)
                if val is None:
                    continue
                abs_pos = sent_start + m.start()
                num_end = sent_start + m.end()
                compound_spans.add((m.start(), m.end()))

                after = text[num_end:].lower()
                unit = ""
                unit_m = _UNIT_AFTER.match(after)
                if unit_m:
                    u = unit_m.group(1).lower().rstrip('s')
                    if u == 'percent' or u == '%':
                        unit = "percent"
                    else:
                        unit = u

                tok = NumberToken(
                    value=val, text=phrase, position=abs_pos,
                    clause_idx=ci, unit=unit, per_unit="",
                )
                tokens.append(tok)

        # --- Hyphenated compound numbers: "forty-five", "twenty-three" ---
        _HYPHEN_TENS = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
        }
        _HYPHEN_ONES = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        }
        _HYPHEN_RE = re.compile(
            r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
            r'-(one|two|three|four|five|six|seven|eight|nine)\b',
            re.IGNORECASE
        )
        for m in _HYPHEN_RE.finditer(sentence):
            tens_word = m.group(1).lower()
            ones_word = m.group(2).lower()
            val = float(_HYPHEN_TENS[tens_word] + _HYPHEN_ONES[ones_word])
            abs_pos = sent_start + m.start()
            num_end = sent_start + m.end()
            compound_spans.add((m.start(), m.end()))

            after = text[num_end:].lower()
            unit = ""
            unit_m = _UNIT_AFTER.match(after)
            if unit_m:
                u = unit_m.group(1).lower().rstrip('s')
                if u == 'percent' or u == '%':
                    unit = "percent"
                else:
                    unit = u

            tok = NumberToken(
                value=val, text=m.group(), position=abs_pos,
                clause_idx=ci, unit=unit, per_unit="",
            )
            tokens.append(tok)

        # Single written-out numbers ("two hours", "half as many")
        for m in _WRITTEN_NUM_RE.finditer(sentence):
            # Skip if this position was already handled by compound match
            if any(s <= m.start() < e for s, e in compound_spans):
                continue

            word = m.group().lower()
            val = _WRITTEN_NUMBERS.get(word)
            if val is None:
                continue
            abs_pos = sent_start + m.start()
            num_end = sent_start + m.end()

            after = text[num_end:].lower()
            before = text[:abs_pos].lower()

            unit = ""
            per_unit = ""

            unit_m = _UNIT_AFTER.match(after)
            if unit_m:
                u = unit_m.group(1).lower().rstrip('s')
                if u == 'percent' or u == '%':
                    unit = "percent"
                else:
                    unit = u

            # Multiplier/scale words
            is_multiplier = word in ('half', 'quarter', 'third', 'fourth', 'fifth',
                                     'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                                     'twice', 'double', 'doubled', 'triple', 'tripled',
                                     'thrice', 'quadrupled', 'halfway')
            # Grouping words
            is_grouping = word in ('dozen', 'pair', 'couple', 'both')

            tok = NumberToken(
                value=val, text=word, position=abs_pos,
                clause_idx=ci, unit=unit, per_unit=per_unit,
            )
            # Pre-tag multiplier words as scale
            if is_multiplier or is_grouping:
                tok.role = 'scale'
                tok.confidence = 0.9
            tokens.append(tok)

    return tokens


# =====================================================================
# Conversion expansion (deterministic intermediate generation)
# =====================================================================

# Language patterns that imply multiplication/division by a specific factor
_MULTIPLIER_PHRASES = [
    # (pattern, factor, description)
    # "twice/double/doubled" patterns
    (re.compile(r'\btwice\s+(?:as\s+)?(?:many|much|long|far|fast|big|tall|heavy|old|large)', re.I),
     2.0, 'twice_as'),
    (re.compile(r'\bdoubled?\b', re.I), 2.0, 'double'),
    # "half" patterns
    (re.compile(r'\bhalf\s+(?:as\s+)?(?:many|much|long|far|fast|big|tall|heavy|old|large)', re.I),
     0.5, 'half_as'),
    (re.compile(r'\bhalf\s+(?:of\s+)?(?:the|that|this|his|her|their|its)', re.I),
     0.5, 'half_of'),
    (re.compile(r'\bhalfway\b', re.I), 0.5, 'halfway'),
    # "triple/three times/tripled"
    (re.compile(r'\b(?:tripled?|three\s+times)\b', re.I), 3.0, 'triple'),
    (re.compile(r'\bthree\s+times\s+(?:as\s+)?(?:many|much|long|far)', re.I),
     3.0, 'three_times_as'),
    # "quadrupled"
    (re.compile(r'\bquadrupled?\b', re.I), 4.0, 'quadruple'),
    # "four/five/... times"
    (re.compile(r'\bfour\s+times\b', re.I), 4.0, 'four_times'),
    (re.compile(r'\bfive\s+times\b', re.I), 5.0, 'five_times'),
    (re.compile(r'\bsix\s+times\b', re.I), 6.0, 'six_times'),
    (re.compile(r'\bseven\s+times\b', re.I), 7.0, 'seven_times'),
    (re.compile(r'\beight\s+times\b', re.I), 8.0, 'eight_times'),
    (re.compile(r'\bnine\s+times\b', re.I), 9.0, 'nine_times'),
    (re.compile(r'\bten\s+times\b', re.I), 10.0, 'ten_times'),
    # "quarter"
    (re.compile(r'\ba?\s*quarter\s+(?:of\s+)?(?:the|that|this|his|her|their|its)', re.I),
     0.25, 'quarter_of'),
    # "third of"
    (re.compile(r'\ba?\s*third\s+(?:of\s+)?(?:the|that|this|his|her|their|its)', re.I),
     1/3, 'third_of'),
    # "fifth of"
    (re.compile(r'\ba?\s*fifth\s+(?:of\s+)?(?:the|that|this|his|her|their|its|an?)', re.I),
     0.2, 'fifth_of'),
    # "tenth of"
    (re.compile(r'\ba?\s*tenth\s+(?:of\s+)?(?:the|that|this|his|her|their|its)', re.I),
     0.1, 'tenth_of'),
    # Fraction words: "three fourths", "three quarters", "two thirds", "nine tenths"
    (re.compile(r'\bthree\s*[-\s]?\s*fourths?\b', re.I), 0.75, 'three_fourths'),
    (re.compile(r'\bthree\s*[-\s]?\s*quarters?\b', re.I), 0.75, 'three_quarters'),
    (re.compile(r'\btwo\s*[-\s]?\s*thirds?\b', re.I), 2/3, 'two_thirds'),
    (re.compile(r'\bone\s*[-\s]?\s*thirds?\b', re.I), 1/3, 'one_third'),
    (re.compile(r'\bone\s*[-\s]?\s*fourths?\b', re.I), 0.25, 'one_fourth'),
    (re.compile(r'\bone\s*[-\s]?\s*quarters?\b', re.I), 0.25, 'one_quarter'),
    (re.compile(r'\bnine\s*[-\s]?\s*tenths?\b', re.I), 0.9, 'nine_tenths'),
    (re.compile(r'\bseven\s*[-\s]?\s*tenths?\b', re.I), 0.7, 'seven_tenths'),
    (re.compile(r'\bthree\s*[-\s]?\s*tenths?\b', re.I), 0.3, 'three_tenths'),
    (re.compile(r'\bone\s*[-\s]?\s*fifths?\b', re.I), 0.2, 'one_fifth'),
    (re.compile(r'\btwo\s*[-\s]?\s*fifths?\b', re.I), 0.4, 'two_fifths'),
    (re.compile(r'\bthree\s*[-\s]?\s*fifths?\b', re.I), 0.6, 'three_fifths'),
    (re.compile(r'\bfour\s*[-\s]?\s*fifths?\b', re.I), 0.8, 'four_fifths'),
    # "one Nth" ordinal fractions
    (re.compile(r'\bone\s*[-\s]?\s*sixths?\b', re.I), 1/6, 'one_sixth'),
    (re.compile(r'\bone\s*[-\s]?\s*sevenths?\b', re.I), 1/7, 'one_seventh'),
    (re.compile(r'\bone\s*[-\s]?\s*eighths?\b', re.I), 1/8, 'one_eighth'),
    (re.compile(r'\bone\s*[-\s]?\s*ninths?\b', re.I), 1/9, 'one_ninth'),
    (re.compile(r'\bone\s*[-\s]?\s*tenths?\b', re.I), 0.1, 'one_tenth'),
    # "two Nth" ordinal fractions
    (re.compile(r'\btwo\s*[-\s]?\s*sevenths?\b', re.I), 2/7, 'two_sevenths'),
    (re.compile(r'\bthree\s*[-\s]?\s*sevenths?\b', re.I), 3/7, 'three_sevenths'),
    (re.compile(r'\bfive\s*[-\s]?\s*sixths?\b', re.I), 5/6, 'five_sixths'),
    (re.compile(r'\bthree\s*[-\s]?\s*eighths?\b', re.I), 3/8, 'three_eighths'),
    (re.compile(r'\bfive\s*[-\s]?\s*eighths?\b', re.I), 5/8, 'five_eighths'),
    (re.compile(r'\bseven\s*[-\s]?\s*eighths?\b', re.I), 7/8, 'seven_eighths'),
    # "both" → 2
    (re.compile(r'\bboth\b', re.I), 2.0, 'both'),
    # "pair of" → 2
    (re.compile(r'\ba?\s*pair\s+of\b', re.I), 2.0, 'pair'),
    # "couple of" → 2
    (re.compile(r'\ba?\s*couple\s+of\b', re.I), 2.0, 'couple'),
    # "buy one get one" → 2
    (re.compile(r'\bbuy\s+one\s+get\s+one\b', re.I), 2.0, 'bogo'),
    # "one and a half" / "one and a half times" → 1.5
    (re.compile(r'\bone\s+and\s+a?\s*half\b', re.I), 1.5, 'one_and_half'),
    # "two and a half" → 2.5
    (re.compile(r'\btwo\s+and\s+a?\s*half\b', re.I), 2.5, 'two_and_half'),
    # "three and a half" → 3.5
    (re.compile(r'\bthree\s+and\s+a?\s*half\b', re.I), 3.5, 'three_and_half'),
]

# Implicit time/unit constants from context
# These use explicit "per/a/each/every UNIT" patterns
_IMPLICIT_CONVERSIONS = [
    # (condition_pattern, second_condition, inject_value, unit, per_unit, description)
    # "per year" + weekly context → 52
    (re.compile(r'\b(?:an?|per|each|every)\s+year\b', re.I),
     re.compile(r'\bweeks?\b', re.I),
     52.0, 'week', 'year', 'weeks_per_year'),
    # "per year" + monthly context → 12
    (re.compile(r'\b(?:an?|per|each|every)\s+year\b', re.I),
     re.compile(r'\bmonths?\b', re.I),
     12.0, 'month', 'year', 'months_per_year'),
    # "per year" + daily context → 365
    (re.compile(r'\b(?:an?|per|each|every)\s+year\b', re.I),
     re.compile(r'\bdays?\b', re.I),
     365.0, 'day', 'year', 'days_per_year'),
    # "per day" + hourly context → 24
    (re.compile(r'\b(?:an?|per|each|every)\s+day\b', re.I),
     re.compile(r'\bhours?\b', re.I),
     24.0, 'hour', 'day', 'hours_per_day'),
    # "per hour" + minute context → 60
    (re.compile(r'\b(?:an?|per|each|every)\s+hour\b', re.I),
     re.compile(r'\bminutes?\b', re.I),
     60.0, 'minute', 'hour', 'minutes_per_hour'),
    # "per minute" + second context → 60
    (re.compile(r'\b(?:an?|per|each|every)\s+minute\b', re.I),
     re.compile(r'\bseconds?\b', re.I),
     60.0, 'second', 'minute', 'seconds_per_minute'),
    # "per second" + minute context → 60
    (re.compile(r'\b(?:an?|per|each|every)\s+second\b', re.I),
     re.compile(r'\bminutes?\b', re.I),
     60.0, 'second', 'minute', 'seconds_per_minute_rev'),
    # "per second" + hour context → 3600
    (re.compile(r'\b(?:an?|per|each|every)\s+second\b', re.I),
     re.compile(r'\bhours?\b', re.I),
     3600.0, 'second', 'hour', 'seconds_per_hour'),
    # "per month" + daily context → 30
    (re.compile(r'\b(?:an?|per|each|every)\s+month\b', re.I),
     re.compile(r'\bdays?\b', re.I),
     30.0, 'day', 'month', 'days_per_month'),
    # "per week" + daily context → 7
    (re.compile(r'\b(?:an?|per|each|every)\s+week\b', re.I),
     re.compile(r'\bdays?\b', re.I),
     7.0, 'day', 'week', 'days_per_week'),
    # "per month" + weekly context → 4
    (re.compile(r'\b(?:an?|per|each|every)\s+month\b', re.I),
     re.compile(r'\bweeks?\b', re.I),
     4.0, 'week', 'month', 'weeks_per_month'),
    # "per foot" + inch context → 12
    (re.compile(r'\b(?:an?|per|each|every)\s+(?:foot|feet)\b', re.I),
     re.compile(r'\binch(?:es)?\b', re.I),
     12.0, 'inch', 'foot', 'inches_per_foot'),
    # "dozen" → 12
    (re.compile(r'\bdozen\b', re.I),
     None,  # no second condition needed
     12.0, 'unit', 'dozen', 'units_per_dozen'),
]

# Unit coexistence rules: when two units appear in the same problem,
# inject the conversion constant even without explicit "per" language
_UNIT_COEXISTENCE = [
    # (unit_a_pattern, unit_b_pattern, value, unit, per_unit, description)
    # hours + minutes → 60
    (re.compile(r'\bhours?\b', re.I),
     re.compile(r'\bminutes?\b', re.I),
     60.0, 'minute', 'hour', 'minutes_per_hour'),
    # minutes + seconds → 60
    (re.compile(r'\bminutes?\b', re.I),
     re.compile(r'\bseconds?\b', re.I),
     60.0, 'second', 'minute', 'seconds_per_minute'),
    # days + hours → 24
    (re.compile(r'\bdays?\b', re.I),
     re.compile(r'\bhours?\b', re.I),
     24.0, 'hour', 'day', 'hours_per_day'),
    # weeks + days → 7
    (re.compile(r'\bweeks?\b', re.I),
     re.compile(r'\bdays?\b', re.I),
     7.0, 'day', 'week', 'days_per_week'),
    # years + months → 12
    (re.compile(r'\byears?\b', re.I),
     re.compile(r'\bmonths?\b', re.I),
     12.0, 'month', 'year', 'months_per_year'),
    # years + weeks → 52
    (re.compile(r'\byears?\b', re.I),
     re.compile(r'\bweeks?\b', re.I),
     52.0, 'week', 'year', 'weeks_per_year'),
    # years + days → 365
    (re.compile(r'\byears?\b', re.I),
     re.compile(r'\bdays?\b', re.I),
     365.0, 'day', 'year', 'days_per_year'),
    # feet + inches → 12
    (re.compile(r'\b(?:feet|foot)\b', re.I),
     re.compile(r'\binch(?:es)?\b', re.I),
     12.0, 'inch', 'foot', 'inches_per_foot'),
    # months + days → 30
    (re.compile(r'\bmonths?\b', re.I),
     re.compile(r'\bdays?\b', re.I),
     30.0, 'day', 'month', 'days_per_month'),
    # months + weeks → 4
    (re.compile(r'\bmonths?\b', re.I),
     re.compile(r'\bweeks?\b', re.I),
     4.0, 'week', 'month', 'weeks_per_month'),
    # kilograms + grams → 1000
    (re.compile(r'\b(?:kilograms?|kg)\b', re.I),
     re.compile(r'\bgrams?\b', re.I),
     1000.0, 'gram', 'kilogram', 'grams_per_kilogram'),
    # kilometers + meters → 1000
    (re.compile(r'\b(?:kilometers?|km)\b', re.I),
     re.compile(r'\bmeters?\b', re.I),
     1000.0, 'meter', 'kilometer', 'meters_per_kilometer'),
    # meters + centimeters → 100
    (re.compile(r'\bmeters?\b', re.I),
     re.compile(r'\b(?:centimeters?|cm)\b', re.I),
     100.0, 'centimeter', 'meter', 'centimeters_per_meter'),
    # dollars + cents → 100
    (re.compile(r'\bdollars?\b|\$', re.I),
     re.compile(r'\bcents?\b', re.I),
     100.0, 'cent', 'dollar', 'cents_per_dollar'),
    # pounds + ounces → 16
    (re.compile(r'\bpounds?\b', re.I),
     re.compile(r'\bounces?\b', re.I),
     16.0, 'ounce', 'pound', 'ounces_per_pound'),
    # hours + seconds → 3600
    (re.compile(r'\bhours?\b', re.I),
     re.compile(r'\bseconds?\b', re.I),
     3600.0, 'second', 'hour', 'seconds_per_hour'),
    # liters + milliliters → 1000
    (re.compile(r'\bliters?\b', re.I),
     re.compile(r'\b(?:milliliters?|ml)\b', re.I),
     1000.0, 'milliliter', 'liter', 'ml_per_liter'),
    # yards + feet → 3
    (re.compile(r'\byards?\b', re.I),
     re.compile(r'\b(?:feet|foot)\b', re.I),
     3.0, 'foot', 'yard', 'feet_per_yard'),
]

# Percentage denominator: inject 100 when % or "percent"/"percentage" is mentioned
_PERCENT_CONTEXT = re.compile(r'\b(\d+)\s*(%|percent\b)', re.I)
_PERCENTAGE_WORD = re.compile(r'\b(?:percent(?:age)?)\b', re.I)


def expand_conversions(tokens: list[NumberToken], text: str) -> list[NumberToken]:
    """
    Deterministic expansion pass: convert language into intermediate tokens.

    Scans the problem text for implicit multipliers and unit conversions,
    then injects synthetic NumberTokens that make the computation explicit.

    Examples:
      "half as many" → injects token with value=0.5, role=scale
      "per hour" + "minutes" → injects token with value=60, role=scale
      "40%" in division → injects token with value=100, role=scale
    """
    text_lower = text.lower()
    injected = []

    # --- Multiplier phrases ---
    for pattern, factor, desc in _MULTIPLIER_PHRASES:
        m = pattern.search(text)
        if m:
            # Check if we already have this factor from written number extraction
            already_have = any(
                abs(t.value - factor) < 0.001 and t.role == 'scale'
                for t in tokens
            )
            if not already_have:
                tok = NumberToken(
                    value=factor,
                    text=desc,
                    position=m.start(),
                    clause_idx=0,  # approximate
                    unit='',
                    per_unit='',
                    role='scale',
                    confidence=0.85,
                )
                injected.append(tok)

    # --- Implicit unit conversions ---
    for entry in _IMPLICIT_CONVERSIONS:
        if len(entry) == 6:
            pat1, pat2, value, unit, per_unit, desc = entry
        else:
            continue

        if pat1.search(text):
            if pat2 is None or pat2.search(text):
                # Check we don't already have this conversion
                already_have = any(
                    abs(t.value - value) < 0.001
                    for t in tokens + injected
                )
                if not already_have:
                    tok = NumberToken(
                        value=value,
                        text=desc,
                        position=0,
                        clause_idx=0,
                        unit=unit,
                        per_unit=per_unit,
                        role='scale',
                        confidence=0.85,
                    )
                    injected.append(tok)

    # --- Unit coexistence rules ---
    # When both units of a pair appear in the text, inject conversion constant
    for pat_a, pat_b, value, unit, per_unit, desc in _UNIT_COEXISTENCE:
        if pat_a.search(text) and pat_b.search(text):
            already_have = any(
                abs(t.value - value) < 0.001
                for t in tokens + injected
            )
            if not already_have:
                tok = NumberToken(
                    value=value,
                    text=desc,
                    position=0,
                    clause_idx=0,
                    unit=unit,
                    per_unit=per_unit,
                    role='scale',
                    confidence=0.75,
                )
                injected.append(tok)

    # --- Coin values: "nickel" = 5 cents, "dime" = 10 cents, "quarter" = 25 cents ---
    _COIN_VALUES = [
        (re.compile(r'\bnickels?\b', re.I), 5.0, 'nickel_cents'),
        (re.compile(r'\bdimes?\b', re.I), 10.0, 'dime_cents'),
        (re.compile(r'\bquarters?\b', re.I), 25.0, 'quarter_cents'),
        (re.compile(r'\bpenni?e?s?\b', re.I), 1.0, 'penny_cents'),
    ]
    for pat, val, desc in _COIN_VALUES:
        if pat.search(text):
            if not any(abs(t.value - val) < 0.001 for t in tokens + injected):
                injected.append(NumberToken(
                    value=val, text=desc, position=0, clause_idx=0,
                    unit='cent', per_unit='', role='scale', confidence=0.8,
                ))

    # --- "half an hour" → 30 minutes, "quarter of an hour" → 15 minutes ---
    if re.search(r'\bhalf\s+(?:an?\s+)?hour\b', text_lower):
        if not any(abs(t.value - 30.0) < 0.001 for t in tokens + injected):
            injected.append(NumberToken(
                value=30.0, text='half_hour_minutes', position=0, clause_idx=0,
                unit='minute', per_unit='', role='scale', confidence=0.85,
            ))
    if re.search(r'\bquarter\s+(?:of\s+)?(?:an?\s+)?hour\b', text_lower):
        if not any(abs(t.value - 15.0) < 0.001 for t in tokens + injected):
            injected.append(NumberToken(
                value=15.0, text='quarter_hour_minutes', position=0, clause_idx=0,
                unit='minute', per_unit='', role='scale', confidence=0.85,
            ))

    # --- "per annum" / "annual" → year ---
    if re.search(r'\b(?:per\s+annum|annual(?:ly)?)\b', text_lower):
        if re.search(r'\bmonths?\b', text_lower):
            if not any(abs(t.value - 12.0) < 0.001 for t in tokens + injected):
                injected.append(NumberToken(
                    value=12.0, text='months_per_year', position=0, clause_idx=0,
                    unit='month', per_unit='year', role='scale', confidence=0.85,
                ))

    # --- Percentage denominator ---
    # Inject 100 when "%" or "percent"/"percentage" appears in text
    has_pct = _PERCENT_CONTEXT.search(text) or _PERCENTAGE_WORD.search(text)
    if has_pct:
        already_have = any(
            t.value == 100 and t.unit != 'percent'
            for t in tokens + injected
        )
        if not already_have:
            tok = NumberToken(
                value=100,
                text='percent_denom',
                position=0,
                clause_idx=0,
                unit='',
                per_unit='',
                role='scale',
                confidence=0.8,
            )
            injected.append(tok)

    # --- Percentage decimal form + complements ---
    # For any percentage token (e.g., 50 with unit=percent), inject:
    # - decimal equivalent (0.50)
    # - 1 + decimal (1.50) for "X% more/longer/bigger"
    # - 1 - decimal (0.50) for "X% less/shorter/smaller/off"
    for t in tokens:
        if t.unit == 'percent' and 0 < t.value <= 100:
            decimal_val = t.value / 100.0
            pct_variants = [
                (decimal_val, f'pct_decimal_{t.value}'),
                (1.0 + decimal_val, f'pct_plus_{t.value}'),
                (1.0 - decimal_val, f'pct_minus_{t.value}'),
            ]
            for pv, desc in pct_variants:
                if pv > 0 and not any(abs(tk.value - pv) < 0.001
                                       for tk in tokens + injected):
                    injected.append(NumberToken(
                        value=pv,
                        text=desc,
                        position=t.position, clause_idx=t.clause_idx,
                        unit='', per_unit='', role='scale', confidence=0.7,
                    ))

    # --- Fraction decomposition ---
    # When a fraction appears, inject its numerator and denominator
    # since computation trees decompose fractions into parts.
    all_tokens = tokens + injected
    all_values = set(t.value for t in all_tokens)

    # Known fraction → (numerator, denominator) map
    _FRAC_DECOMP = {
        0.5: (1, 2), 0.25: (1, 4), 0.2: (1, 5), 0.1: (1, 10),
        0.75: (3, 4), 2/3: (2, 3), 1/3: (1, 3),
        0.4: (2, 5), 0.6: (3, 5), 0.8: (4, 5),
        0.125: (1, 8), 0.375: (3, 8), 0.625: (5, 8), 0.875: (7, 8),
        1/6: (1, 6), 5/6: (5, 6),
        1/7: (1, 7), 2/7: (2, 7), 3/7: (3, 7),
        1/9: (1, 9), 2/9: (2, 9),
        1.5: (3, 2), 2.5: (5, 2), 3.5: (7, 2),
        0.3: (3, 10), 0.7: (7, 10), 0.9: (9, 10),
    }

    for t in all_tokens:
        numer = denom = None

        # Direct text-based decomposition: "3/7", "5/8", "1 3/8"
        frac_m = re.search(r'(\d+)\s*/\s*(\d+)', t.text)
        if frac_m:
            numer = int(frac_m.group(1))
            denom = int(frac_m.group(2))
        else:
            # Lookup in known fractions table
            for frac_val, (n, d) in _FRAC_DECOMP.items():
                if abs(t.value - frac_val) < 0.001:
                    numer, denom = n, d
                    break

        if numer is not None and denom is not None:
            for part in (float(numer), float(denom)):
                if part not in all_values:
                    injected.append(NumberToken(
                        value=part, text=f'frac_part_{int(part)}',
                        position=t.position, clause_idx=t.clause_idx,
                        unit='', per_unit='', role='scale', confidence=0.6,
                    ))
                    all_values.add(part)

    # --- Implicit 1 ---
    # 1.0 is needed as: percentage base (1+x%), "the rest" (1-fraction),
    # "next year" (+1), "each" (per 1 item), multiplicative identity.
    # Always inject if not already present.
    text_lower = text.lower()
    has_one = any(abs(t.value - 1.0) < 0.001 for t in tokens + injected)
    if not has_one:
        tok = NumberToken(
            value=1.0, text='implicit_one', position=0, clause_idx=0,
            unit='', per_unit='', role='scale', confidence=0.6,
        )
        injected.append(tok)

    # --- Percentage complement: 100 - pct_value ---
    # When we see "60%", inject 40 (= 100-60) as the complement.
    # Needed for problems like "60% drive, how many don't drive?"
    for t in tokens:
        if t.unit == 'percent' and 0 < t.value < 100:
            complement = 100 - t.value
            if not any(abs(tk.value - complement) < 0.001
                       for tk in tokens + injected):
                injected.append(NumberToken(
                    value=complement,
                    text=f'pct_complement_{t.value}',
                    position=t.position, clause_idx=t.clause_idx,
                    unit='', per_unit='', role='scale', confidence=0.65,
                ))

    # --- Broader time/unit context injection ---
    # Inject conversion constants when a time unit word appears,
    # even without an explicit "per X" or two-unit coexistence.
    _CONTEXT_INJECTIONS = [
        (r'\bweeks?\b', 7.0, 'day', 'week', 'days_per_week_ctx'),
        (r'\bhours?\b', 60.0, 'minute', 'hour', 'minutes_per_hour_ctx'),
        (r'\bminutes?\b', 60.0, 'second', 'minute', 'seconds_per_minute_ctx'),
        (r'\bdays?\b', 24.0, 'hour', 'day', 'hours_per_day_ctx'),
        (r'\bmonths?\b', 30.0, 'day', 'month', 'days_per_month_ctx'),
        (r'\byears?\b', 12.0, 'month', 'year', 'months_per_year_ctx'),
        (r'\byears?\b', 52.0, 'week', 'year', 'weeks_per_year_ctx'),
        (r'\byears?\b', 365.0, 'day', 'year', 'days_per_year_ctx'),
        (r'\b(?:feet|foot)\b', 12.0, 'inch', 'foot', 'inches_per_foot_ctx'),
        (r'\bdozens?\b', 12.0, 'unit', 'dozen', 'units_per_dozen_ctx'),
    ]
    for pat, value, unit, per_unit, desc in _CONTEXT_INJECTIONS:
        if re.search(pat, text_lower):
            if not any(abs(tk.value - value) < 0.001
                       for tk in tokens + injected):
                injected.append(NumberToken(
                    value=value, text=desc, position=0, clause_idx=0,
                    unit=unit, per_unit=per_unit,
                    role='scale', confidence=0.65,
                ))

    # --- Adverb time words: "daily", "weekly", "monthly" ---
    _ADVERB_INJECTIONS = [
        (r'\bdaily\b', [(7.0, 'days_per_week_daily'),
                        (30.0, 'days_per_month_daily'),
                        (365.0, 'days_per_year_daily')]),
        (r'\bweekly\b', [(7.0, 'days_per_week_weekly'),
                         (52.0, 'weeks_per_year_weekly'),
                         (4.0, 'weeks_per_month_weekly')]),
        (r'\bmonthly\b', [(12.0, 'months_per_year_monthly'),
                          (30.0, 'days_per_month_monthly'),
                          (4.0, 'weeks_per_month_monthly')]),
        (r'\bannual(?:ly)?\b', [(12.0, 'months_per_year_annual'),
                                (52.0, 'weeks_per_year_annual'),
                                (365.0, 'days_per_year_annual')]),
        (r'\bhourly\b', [(60.0, 'minutes_per_hour_hourly'),
                         (24.0, 'hours_per_day_hourly')]),
    ]
    for pat, vals in _ADVERB_INJECTIONS:
        if re.search(pat, text_lower):
            for value, desc in vals:
                if not any(abs(tk.value - value) < 0.001
                           for tk in tokens + injected):
                    injected.append(NumberToken(
                        value=value, text=desc, position=0, clause_idx=0,
                        unit='', per_unit='',
                        role='scale', confidence=0.6,
                    ))

    # --- Weekday/weekend constants ---
    if re.search(r'\bweekdays?\b', text_lower):
        if not any(abs(tk.value - 5.0) < 0.001 for tk in tokens + injected):
            injected.append(NumberToken(
                value=5.0, text='weekdays_count', position=0, clause_idx=0,
                unit='day', per_unit='week', role='scale', confidence=0.8,
            ))
    if re.search(r'\bweekends?\b', text_lower):
        if not any(abs(tk.value - 2.0) < 0.001 for tk in tokens + injected):
            injected.append(NumberToken(
                value=2.0, text='weekend_days', position=0, clause_idx=0,
                unit='day', per_unit='week', role='scale', confidence=0.8,
            ))
    # "Monday through Friday" → 5 weekdays
    if re.search(r'\bmonday\b.*\bfriday\b', text_lower):
        if not any(abs(tk.value - 5.0) < 0.001 for tk in tokens + injected):
            injected.append(NumberToken(
                value=5.0, text='mon_fri_count', position=0, clause_idx=0,
                unit='day', per_unit='week', role='scale', confidence=0.8,
            ))
    # "every morning/evening" + "week" → 7
    if re.search(r'\bevery\s+(?:morning|evening|night|day)\b', text_lower):
        if re.search(r'\bweeks?\b', text_lower):
            if not any(abs(tk.value - 7.0) < 0.001
                       for tk in tokens + injected):
                injected.append(NumberToken(
                    value=7.0, text='every_day_per_week', position=0,
                    clause_idx=0, unit='day', per_unit='week',
                    role='scale', confidence=0.7,
                ))

    # --- Pairwise derivation ---
    # Pre-compute single-step intermediates (product, quotient, sum,
    # difference) of all pairs of available values.  Many computation
    # tree "leaf" values are actually one-step derivations of extracted
    # numbers (e.g. "20% of 10" → 2, or "3 weeks" → 21 days).
    all_vals = set(tk.value for tk in tokens + injected)
    derived = set()
    avail_list = sorted(all_vals)
    for a in avail_list:
        for b in avail_list:
            derived.add(a * b)
            if b != 0:
                derived.add(a / b)
            derived.add(a + b)
            if a >= b:
                derived.add(a - b)
    # Only add values not already present
    for dv in derived:
        if dv not in all_vals and dv >= 0:
            injected.append(NumberToken(
                value=dv, text='derived', position=0, clause_idx=0,
                unit='', per_unit='', role='', confidence=0.3,
            ))

    return tokens + injected


# =====================================================================
# Role classification (the ML-augmented decision)
# =====================================================================

def classify_roles(tokens: list[NumberToken], text: str) -> list[NumberToken]:
    """
    Classify each number as 'scale' or 'hub'.

    Rules (deterministic, ratchetable):
      - Has per_unit → scale (it's a rate)
      - Is a percent → scale
      - Appears with "times/twice/double/half" → scale
      - Appears in enumeration (list of same-unit things) → hub
      - Default: hub (most numbers are quantities)

    Each rule has a confidence. When rules conflict, highest confidence wins.
    When no rule fires strongly, this is where an LLM can help.
    """
    text_lower = text.lower()

    # Pre-pass: find all rates and their per_units
    rate_per_units = set()
    rate_units = set()
    for tok in tokens:
        if tok.per_unit:
            rate_per_units.add(tok.per_unit)
            if tok.unit:
                rate_units.add(tok.unit)

    # Unit compatibility map (singular forms)
    _COMPAT = {
        'hour': {'hour', 'minute', 'min'},
        'minute': {'minute', 'min', 'hour'},
        'day': {'day'},
        'week': {'week', 'day'},
        'day': {'day', 'week'},
        'month': {'month', 'year'},
        'year': {'year', 'month'},
        'egg': {'egg'},
        'slice': {'slice', 'piece'},
        'lawn': {'lawn'},
        'person': {'person', 'student', 'child', 'friend'},
        'student': {'student', 'person', 'child'},
        'box': {'box', 'pack', 'bag'},
        'foot': {'foot', 'feet', 'inch'},
        'inch': {'inch', 'foot', 'feet'},
        'meter': {'meter', 'centimeter', 'kilometer'},
        'pound': {'pound', 'ounce', 'kilogram'},
        'ounce': {'ounce', 'pound'},
        'dollar': {'dollar', 'cent'},
        'cent': {'cent', 'dollar'},
        'each': set(),  # generic per-unit
        'task': set(),   # from verb rate detection
    }

    def units_compatible(u1, u2):
        """Check if two unit strings are compatible."""
        if u1 == u2:
            return True
        u1s = u1.rstrip('s')
        u2s = u2.rstrip('s')
        if u1s == u2s:
            return True
        return u2s in _COMPAT.get(u1s, set()) or u1s in _COMPAT.get(u2s, set())

    for tok in tokens:
        # Skip tokens already classified (e.g., written multipliers)
        if tok.role and tok.confidence >= 0.8:
            continue

        signals = []  # (role, confidence, reason)

        # --- Strong scale signals ---
        if tok.per_unit:
            signals.append(('scale', 0.9, f'has per_unit={tok.per_unit}'))

        if tok.unit == 'percent':
            signals.append(('scale', 0.9, 'is a percentage'))

        # Check IMMEDIATE context for multiplier language (narrow window)
        # Only match if the multiplier word is in the same clause as this number
        after_text_short = text_lower[tok.position + len(tok.text):
                                      tok.position + len(tok.text) + 20]
        before_text_short = text_lower[max(0, tok.position - 20):tok.position]
        if _MULTIPLIER_CONTEXT.search(after_text_short):
            signals.append(('scale', 0.8, 'multiplier after'))
        elif _MULTIPLIER_CONTEXT.search(before_text_short):
            signals.append(('scale', 0.8, 'multiplier before'))

        # "of" after number — only for fractions/percents: "20% of", "half of"
        # NOT for "48 of her friends" (that's a quantity)
        after_text = text_lower[tok.position + len(tok.text):]
        if re.match(r'\s+of\b', after_text):
            # Only scale if this looks like a fraction/percent
            if tok.unit == 'percent' or tok.value < 1 or tok.text in (
                    'half', 'third', 'quarter', 'double', 'triple'):
                signals.append(('scale', 0.7, '"fraction of" pattern'))

        # Rate companion: this number's unit matches a rate's per_unit
        # "50 minutes" matches "$12 per hour" (minute ~ hour)
        # "4 lawns" matches "$15 to mow" (lawn ~ task)
        if tok.unit and not tok.per_unit:
            for rpu in rate_per_units:
                if units_compatible(tok.unit, rpu):
                    signals.append(('scale', 0.75, f'unit {tok.unit} matches rate per_unit {rpu}'))
                    break

        # Bare number (no unit) with a rate present — likely a multiplier/count
        if not tok.unit and not tok.per_unit and rate_per_units:
            # Small integers near rates are often counts/multipliers
            if tok.value == int(tok.value) and 1 < tok.value <= 20:
                signals.append(('scale', 0.5, 'bare integer near rate'))

        # --- Strong hub signals ---
        # Enumeration: "26 pink, 15 green, 24 yellow"
        # Numbers that appear in a list with the same unit type
        if tok.unit and sum(1 for t in tokens if t.unit == tok.unit) >= 2:
            signals.append(('hub', 0.7, f'multiple {tok.unit} values'))

        # "there are/has/have N" pattern
        before_text = text_lower[max(0, tok.position - 30):tok.position]
        if re.search(r'\b(?:there (?:are|were|is)|(?:has|have|had))\s*$', before_text):
            signals.append(('hub', 0.6, 'existence statement'))

        # Numbers in the question sentence are often the target (hub)
        if '?' in text[tok.position:]:
            remaining = text[tok.position:]
            if '.' not in remaining.split('?')[0]:
                signals.append(('hub', 0.5, 'in question sentence'))

        # --- Default ---
        if not signals:
            signals.append(('hub', 0.3, 'default'))

        # Pick highest confidence signal
        best = max(signals, key=lambda s: s[1])
        tok.role = best[0]
        tok.confidence = best[1]

    return tokens


# =====================================================================
# Hub topology (which numbers group together)
# =====================================================================

def build_hubs(tokens: list[NumberToken]) -> tuple[list[Hub], list[ScaleEdge]]:
    """
    Group hub members into hubs and identify scale edges.

    Simple heuristic: numbers in the same clause with the same unit
    go in the same hub. Scale factors connect hubs.
    """
    hub_members = [t for t in tokens if t.role == 'hub']
    scale_factors = [t for t in tokens if t.role == 'scale']

    # Group hub members by unit (numbers with same unit collapse together)
    unit_groups = {}
    for t in hub_members:
        key = t.unit or 'unknown'
        if key not in unit_groups:
            unit_groups[key] = []
        unit_groups[key].append(t)

    hubs = []
    for i, (unit, members) in enumerate(unit_groups.items()):
        hub = Hub(hub_id=i, members=members, unit=unit)
        for m in members:
            m.hub_id = i
        hubs.append(hub)

    # If no grouping found, put all hub members in one hub
    if not hubs and hub_members:
        hub = Hub(hub_id=0, members=hub_members)
        for m in hub_members:
            m.hub_id = 0
        hubs.append(hub)

    # Scale edges from scale factors
    edges = []
    for sf in scale_factors:
        factor = sf.value
        if sf.unit == 'percent':
            factor = sf.value / 100.0
        edges.append(ScaleEdge(
            factor=factor,
            source=f"{sf.text} ({sf.unit}/{sf.per_unit})",
        ))

    return hubs, edges


# =====================================================================
# Execution (deterministic)
# =====================================================================

def execute(hubs: list[Hub], edges: list[ScaleEdge],
            text: str = "",
            all_tokens: list = None) -> list[tuple[float, str, float]]:
    """
    Execute the hub+scale computation.

    1. Collapse each hub (add its members)
    2. Apply scale edges between hubs
    3. Return candidate answers with descriptions

    Returns list of (value, description, confidence)
    """
    results = []
    _all_tokens = all_tokens or []

    # Step 1: collapse each hub
    for hub in hubs:
        if hub.members:
            hub.collapsed_value = sum(m.value for m in hub.members)

    # Detect subtraction from text
    text_lower = text.lower()
    has_subtract = bool(re.search(
        r'\b(?:left|remain|gave|lost|spent|ate|fewer|less|minus|subtract'
        r'|remove|took|difference)\b', text_lower))

    # Step 1b: apply unit conversions between hubs and scale edges
    # If a hub has unit "minute" and a scale edge came from "per hour",
    # inject a conversion factor
    all_tokens = []
    for h in hubs:
        all_tokens.extend(h.members)
    scale_tokens = [t for t in all_tokens]  # placeholder

    for hub in hubs:
        for edge in edges:
            # Parse the edge's rate unit from its source
            # source format: "12 (dollars/hour)"
            if '/' in edge.source:
                parts = edge.source.split('/')
                rate_per = parts[-1].strip().rstrip(')')
                conv = get_conversion(hub.unit, rate_per)
                if conv is not None and conv != 1.0:
                    # Apply conversion to hub's collapsed value
                    converted = hub.collapsed_value * conv if hub.collapsed_value else None
                    if converted is not None:
                        results.append((
                            converted * edge.factor,
                            f"convert({hub.unit}→{rate_per})*scale({hub.collapsed_value}*{conv}*{edge.factor})",
                            0.65
                        ))

    # --- Generate candidates ---

    # Pure hub collapse (no scaling)
    if not edges:
        for hub in hubs:
            if hub.collapsed_value is not None:
                results.append((
                    hub.collapsed_value,
                    f"hub_collapse({'+'.join(str(m.value) for m in hub.members)})",
                    0.5
                ))
        # Also try subtraction variants
        if has_subtract and len(hubs) == 1 and len(hubs[0].members) >= 2:
            members = hubs[0].members
            # First minus rest
            val = members[0].value
            for m in members[1:]:
                val -= m.value
            results.append((
                val,
                f"hub_subtract({members[0].value}-{'+'.join(str(m.value) for m in members[1:])})",
                0.5
            ))

    # Single hub + scale edges
    if len(hubs) == 1 and edges:
        hub_val = hubs[0].collapsed_value
        if hub_val is not None:
            for edge in edges:
                # Scale produces new value, add to original
                # Pattern: "48 clips in April, half as many in May → 48 + 24 = 72"
                scaled = hub_val * edge.factor
                results.append((
                    hub_val + scaled,
                    f"hub+scale({hub_val}+{hub_val}*{edge.factor})",
                    0.6
                ))
                # Scale up (direct)
                results.append((
                    scaled,
                    f"hub*scale({hub_val}*{edge.factor})",
                    0.55
                ))
                # Scale down (divide)
                if edge.factor != 0:
                    results.append((
                        hub_val / edge.factor,
                        f"hub/scale({hub_val}/{edge.factor})",
                        0.5
                    ))
                # Subtract scaled from original
                results.append((
                    hub_val - scaled,
                    f"hub-scale({hub_val}-{hub_val}*{edge.factor})",
                    0.5
                ))

        # Also: subtract within hub, THEN scale
        if has_subtract and len(hubs[0].members) >= 2:
            members = hubs[0].members
            remainder = members[0].value
            for m in members[1:]:
                remainder -= m.value
            for edge in edges:
                results.append((
                    remainder * edge.factor,
                    f"subtract_then_scale(({members[0].value}-...)*{edge.factor})",
                    0.6
                ))
                if edge.factor != 0:
                    results.append((
                        remainder / edge.factor,
                        f"subtract_then_divide(({members[0].value}-...)/{edge.factor})",
                        0.5
                    ))

    # Multi-hub: collapse each, then combine with scale
    if len(hubs) >= 2:
        # Each hub's collapsed value might scale by an edge
        hub_vals = [h.collapsed_value for h in hubs if h.collapsed_value is not None]
        if len(hub_vals) >= 2:
            # Add hub values
            results.append((
                sum(hub_vals),
                f"multi_hub_add({'+'.join(str(v) for v in hub_vals)})",
                0.4
            ))
            # Subtract
            results.append((
                hub_vals[0] - sum(hub_vals[1:]),
                f"multi_hub_sub({hub_vals[0]}-...)",
                0.4
            ))

        # Apply scale edges
        for edge in edges:
            for hv in hub_vals:
                results.append((
                    hv * edge.factor,
                    f"hub_scaled({hv}*{edge.factor})",
                    0.5
                ))

    # Chained scales (no hub — pure scaling)
    if not hubs and len(edges) >= 2:
        val = edges[0].factor
        for e in edges[1:]:
            val *= e.factor
        results.append((val, f"scale_chain({'*'.join(str(e.factor) for e in edges)})", 0.5))

    # Scale chain with unit conversion
    # When two scale factors have related but different units,
    # inject a conversion. E.g., $12/hour * 50 minutes → 12 * (50/60) = 10
    if len(edges) >= 2:
        # Find all scale tokens to check units
        scale_tokens_full = [t for t in _all_tokens if t.role == 'scale']
        for i, t1 in enumerate(scale_tokens_full):
            for j, t2 in enumerate(scale_tokens_full):
                if i >= j:
                    continue
                # Check if t1's per_unit needs conversion to match t2's unit
                if t1.per_unit and t2.unit:
                    conv = get_conversion(t2.unit, t1.per_unit)
                    if conv is not None and conv != 1.0:
                        # t1 is rate ($/hour), t2 is quantity (50 min)
                        # result = rate * quantity * conversion
                        result = t1.value * t2.value * conv
                        results.append((
                            result,
                            f"rate_convert({t1.value}*{t2.value}*{conv} [{t2.unit}→{t1.per_unit}])",
                            0.7
                        ))
                if t2.per_unit and t1.unit:
                    conv = get_conversion(t1.unit, t2.per_unit)
                    if conv is not None and conv != 1.0:
                        result = t2.value * t1.value * conv
                        results.append((
                            result,
                            f"rate_convert({t2.value}*{t1.value}*{conv} [{t1.unit}→{t2.per_unit}])",
                            0.7
                        ))

    # Mixed: scale factor as starting value, hub members subtract, another scale multiplies
    # Pattern: (scale1 - hub_members) * scale2
    # Example: (16 eggs/day - 3 eaten - 4 baked) * $2/egg = 18
    if len(edges) >= 2 and hubs:
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges):
                if i == j:
                    continue
                hub_val = hubs[0].collapsed_value if hubs else 0
                if hub_val is not None:
                    remainder = e1.factor - hub_val
                    results.append((
                        remainder * e2.factor,
                        f"({e1.factor}-hub)*{e2.factor}",
                        0.55
                    ))
                    results.append((
                        (e1.factor + hub_val) * e2.factor,
                        f"({e1.factor}+hub)*{e2.factor}",
                        0.5
                    ))

    return results


# =====================================================================
# Main solver
# =====================================================================

def solve(text: str) -> list[tuple[float, str, float]]:
    """
    Solve a word problem using the hub+scale architecture.

    Returns candidate answers sorted by confidence descending.
    """
    # 1. Extract numbers with provenance
    tokens = extract_numbers(text)

    # 1b. Expand implicit conversions (deterministic)
    tokens = expand_conversions(tokens, text)

    if not tokens:
        return []

    # 2. Classify each number as scale or hub
    tokens = classify_roles(tokens, text)

    # 3. Build hub topology
    hubs, edges = build_hubs(tokens)

    # 4. Execute (deterministic)
    candidates = execute(hubs, edges, text, all_tokens=tokens)

    # Sort by confidence descending
    candidates.sort(key=lambda x: -x[2])

    return candidates


def solve_debug(text: str) -> dict:
    """
    Solve with full debug output showing each layer's decisions.
    """
    tokens = extract_numbers(text)
    tokens = expand_conversions(tokens, text)
    tokens = classify_roles(tokens, text)
    hubs, edges = build_hubs(tokens)
    candidates = execute(hubs, edges, text, all_tokens=tokens)
    candidates.sort(key=lambda x: -x[2])

    return {
        'tokens': [
            {
                'value': t.value,
                'unit': t.unit,
                'per_unit': t.per_unit,
                'clause': t.clause_idx,
                'role': t.role,
                'confidence': t.confidence,
                'hub_id': t.hub_id,
            }
            for t in tokens
        ],
        'hubs': [
            {
                'id': h.hub_id,
                'unit': h.unit,
                'members': [m.value for m in h.members],
                'collapsed': h.collapsed_value,
            }
            for h in hubs
        ],
        'edges': [
            {'factor': e.factor, 'source': e.source}
            for e in edges
        ],
        'candidates': candidates[:10],
    }


# =====================================================================
# LLM-augmented solver
# =====================================================================

_LLM_PLAN_PROMPT = """You are a math word problem solver. I will give you a problem and the numbers extracted from it.

Your job: write the arithmetic expression that solves this problem using ONLY these two operations:
- ADD/SUBTRACT: combine same-unit quantities (use + and -)
- SCALE: multiply or divide to transform units (use * and /)

Rules:
- Use only the numbers provided (and constants like 100 for percentages)
- Show intermediate steps if the problem has multiple steps
- Write a Python expression that evaluates to the answer

Problem: {problem}

Numbers extracted:
{numbers}

Respond with ONLY a valid Python arithmetic expression (no variables, just numbers and +, -, *, /, parentheses). Example: (16 - 3 - 4) * 2
"""

_LLM_VALIDATE_PROMPT = """A math word problem solver produced candidate answers. Pick the most reasonable one.

Problem: {problem}

Candidates:
{candidates}

Respond with ONLY the numeric answer (just the number, nothing else).
"""


def solve_with_llm(text: str, llm_call=None) -> list[tuple[float, str, float]]:
    """
    Solve using deterministic extraction + LLM for layers 2, 3, 5.

    The LLM provides the computation PLAN (arithmetic expression).
    The deterministic engine evaluates it safely.

    llm_call: function(prompt: str) -> str
      Calls an LLM and returns the text response.
      If None, falls back to pure deterministic solve().
    """
    if llm_call is None:
        return solve(text)

    # Layer 1: deterministic number extraction
    tokens = extract_numbers(text)

    # Layer 1b: deterministic conversion expansion
    tokens = expand_conversions(tokens, text)

    # Layer 1c: deterministic role pre-classification
    tokens = classify_roles(tokens, text)

    # Layers 2+3: LLM provides the computation plan
    numbers_desc = "\n".join(
        f"  {t.value} (unit={t.unit or 'none'}, per_unit={t.per_unit or 'none'}, "
        f"clause {t.clause_idx})"
        for t in tokens
    )

    prompt = _LLM_PLAN_PROMPT.format(
        problem=text,
        numbers=numbers_desc,
    )

    llm_answer = None
    llm_expr = None
    try:
        response = llm_call(prompt)
        expr = response.strip()
        # Clean up: remove markdown, quotes, etc.
        if expr.startswith("```"):
            expr = "\n".join(expr.split("\n")[1:])
        if expr.endswith("```"):
            expr = "\n".join(expr.split("\n")[:-1])
        expr = expr.strip().strip('"').strip("'")

        # Layer 4: deterministic evaluation (safe eval — numbers and ops only)
        llm_expr = expr
        llm_answer = _safe_eval(expr)
    except Exception:
        pass

    # Also get deterministic candidates as fallback
    det_candidates = solve(text)

    # Combine: LLM plan answer (if valid) + deterministic candidates
    results = []
    if llm_answer is not None:
        results.append((llm_answer, f"llm_plan: {llm_expr}", 0.9))

    # Layer 5: if we have candidates, let LLM validate
    if det_candidates and llm_call and not llm_answer:
        top = det_candidates[:10]
        cand_desc = "\n".join(
            f"  {i+1}. {val:.4g}  ({desc})"
            for i, (val, desc, conf) in enumerate(top)
        )
        prompt = _LLM_VALIDATE_PROMPT.format(
            problem=text,
            candidates=cand_desc,
        )
        try:
            response = llm_call(prompt)
            val = float(response.strip().replace(',', ''))
            results.insert(0, (val, "llm_validated", 0.85))
        except (ValueError, TypeError):
            pass

    results.extend(det_candidates)
    return results


def _safe_eval(expr: str) -> float:
    """
    Safely evaluate an arithmetic expression.
    Only allows: numbers, +, -, *, /, (), **, and whitespace.
    """
    # Whitelist: digits, decimal points, operators, parens, whitespace
    cleaned = expr.strip()
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,]+$', cleaned):
        raise ValueError(f"Unsafe expression: {cleaned}")
    # Remove commas in numbers
    cleaned = cleaned.replace(',', '')
    return float(eval(cleaned))  # safe because whitelist above
