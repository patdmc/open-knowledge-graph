"""
Natural language → AST compiler for word problems.

Architecture:
  1. Lex:     text → tokens (numbers, units, verbs, references)
  2. Parse:   tokens → AST (tree of operations, not a flat chain)
  3. Resolve: bind references ("the remainder", "that number", "twice as many")
  4. Execute: walk AST bottom-up, evaluate deterministically

The AST respects order of operations because it IS order of operations.
Parentheses in arithmetic are serialized tree structure. The tree is the
canonical representation.

Usage:
    from packages.core.interpret import compile, execute

    ast, ctx = compile("Janet has 16 eggs per day. She eats 3. "
                       "She bakes with 4. She sells the rest at $2 each.")
    answer = execute(ast)  # → 18.0
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from packages.math.operations import OPERATIONS, resolve_verb


# =====================================================================
# Learned word→operation mappings (trained from GSM8K solutions)
# =====================================================================

_LEARNED_VERBS: dict[str, dict] = {}
_OP_SYMBOL_TO_ID = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}

# Names and pure nouns to exclude from learned mappings (they correlate
# with operations in training data but have no causal relationship)
_LEARNED_SKIP = {
    "adam", "andrew", "billy", "carl", "frank", "jack", "jake", "james",
    "janet", "jason", "jeff", "jerry", "jill", "john", "lisa", "mark",
    "maria", "martha", "mary", "michael", "peter", "rachel", "randy",
    "sam", "sarah", "tom",  # names
    "apple", "balloons", "balls", "bananas", "bars", "birds", "black",
    "blue", "bread", "cake", "cakes", "candies", "candy", "cards",
    "cats", "cheese", "chicken", "chocolate", "coffee", "cookies",
    "cream", "dog", "dogs", "ducks", "eggs", "fish", "flowers", "food",
    "fruits", "gas", "ice", "juice", "milk", "oranges", "pizza",
    "pizzas", "roses", "sandwiches", "seeds", "soda", "tomatoes",
    "toys", "water", "yellow",  # pure nouns
    "adult", "adults", "bill", "boys", "brother", "children",
    "customers", "family", "father", "friend", "friends", "girls",
    "kids", "mom", "mother", "sister", "student", "students",
    "people", "person", "players",  # people nouns
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "today",  # days
}

def _load_learned_verbs():
    """Load learned verb→operation mappings from JSON."""
    global _LEARNED_VERBS
    if _LEARNED_VERBS:
        return
    path = os.path.join(os.path.dirname(__file__), "..", "math",
                        "learned_verb_ops.json")
    try:
        with open(path) as f:
            raw = json.load(f)
        for word, info in raw.items():
            if word in _LEARNED_SKIP:
                continue
            if info["lift"] < 0.15:
                continue
            op_id = _OP_SYMBOL_TO_ID.get(info["op"])
            if op_id:
                _LEARNED_VERBS[word] = {"op": op_id, "lift": info["lift"],
                                         "n": info["n"]}
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def _learned_vote(sentence: str) -> tuple[str, str, float]:
    """
    Vote on the operation for a sentence using learned word mappings.
    Returns (operation_id, winning_word, total_lift) or ("", "", 0).
    """
    _load_learned_verbs()
    if not _LEARNED_VERBS:
        return "", "", 0.0

    words = set(re.findall(r'[a-z]+', sentence.lower()))
    votes: dict[str, float] = {}  # op_id → total lift
    best_word: dict[str, str] = {}  # op_id → highest-lift word

    for word in words:
        entry = _LEARNED_VERBS.get(word)
        if entry:
            op = entry["op"]
            lift = entry["lift"]
            votes[op] = votes.get(op, 0) + lift
            if op not in best_word or lift > _LEARNED_VERBS.get(
                    best_word[op], {}).get("lift", 0):
                best_word[op] = word

    if not votes:
        return "", "", 0.0

    winner = max(votes, key=votes.get)
    return winner, best_word.get(winner, ""), votes[winner]


# =====================================================================
# AST node types
# =====================================================================

@dataclass
class Lit:
    """Literal numeric value."""
    value: float
    label: str = ""      # descriptive label ("eggs_per_day", "$2")

    def __repr__(self):
        return f"{self.value:g}" + (f"({self.label})" if self.label else "")


@dataclass
class BinOp:
    """Binary operation: left OP right."""
    op: str              # "add", "subtract", "multiply", "divide"
    left: 'ASTNode'
    right: 'ASTNode'

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


@dataclass
class Ref:
    """Reference to a named variable (resolved during compilation)."""
    name: str            # "remainder", "that_number", "total"
    resolved: Optional['ASTNode'] = None

    def __repr__(self):
        if self.resolved:
            return f"&{self.name}={self.resolved}"
        return f"&{self.name}"


@dataclass
class Neg:
    """Unary negation."""
    child: 'ASTNode'

    def __repr__(self):
        return f"(-{self.child})"


ASTNode = Union[Lit, BinOp, Ref, Neg]


# =====================================================================
# Compilation context
# =====================================================================

@dataclass
class CompileContext:
    """State accumulated during compilation."""
    source: str
    variables: dict = field(default_factory=dict)   # name → ASTNode
    sentences: list = field(default_factory=list)    # parsed sentences
    ast: Optional[ASTNode] = None                    # final AST
    confidence: float = 0.0
    unsolved_reason: str = ""
    debug: list = field(default_factory=list)        # trace for debugging
    problem_graph: object = None                      # ProblemGraph (retained)
    walk_result: object = None                        # WalkResult (retained)
    solver_used: str = ""                             # which solver produced the answer
    equivalence_classes: list = field(default_factory=list)  # from classifier


# =====================================================================
# Lexer: extract tokens from text
# =====================================================================

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
}

MULTIPLIER_WORDS = {
    "half": 0.5, "quarter": 0.25, "third": 1/3,
    "twice": 2, "double": 2, "triple": 3, "quadruple": 4,
}

# Known unit conversion rates → dollars (world knowledge)
# ---------------------------------------------------------------------------
# Unit conversions — imported from math graph (packages/math/units.py)
#
# The conversion FACTORS are math knowledge (1 hour = 60 minutes).
# The LANGUAGE that maps "per hour" or "every 30 minutes" to a conversion
# is handled by the language graph (packages/core/language/resolve.py).
# ---------------------------------------------------------------------------
from packages.math.units import (
    UNIT_TO_DOLLARS, UNIT_TO_SECONDS, UNIT_TO_HOURS, UNIT_TO_MINUTES,
    UNIT_TO_DAYS, UNIT_TO_WEEKS,
    UNIT_TO_INCHES, UNIT_TO_FEET, UNIT_TO_MILES, UNIT_TO_CM,
)

# Quantity words (implicit multipliers)
# Note: "pair" excluded — too context-dependent ("pair of shorts" = 1 item)
UNIT_MULTIPLIERS = {
    "dozen": 12,
    "score": 20, "gross": 144,
    "couple": 2,
}

# Items where "a pair of X" = 1 unit (not 2 individual items)
# These naturally come in pairs — the pair IS the unit
_PAIR_IS_UNIT = {
    "shoes", "shoe", "socks", "sock", "pants", "pant",
    "shorts", "short", "gloves", "glove", "mittens", "mitten",
    "boots", "boot", "sandals", "sandal", "slippers", "slipper",
    "earrings", "earring", "glasses", "scissors", "tweezers",
    "jeans", "trousers", "leggings", "stockings",
}

FRACTION_PATTERN = re.compile(r'(\d+)\s*/\s*(\d+)')
MONEY_PATTERN = re.compile(r'\$(\d[\d,]*\.?\d*)')
NUMBER_PATTERN = re.compile(r'(?<!\d)(\d[\d,]*\.?\d*)(?!\d)')

# References to computed values
REFERENCE_PATTERNS = [
    (r'\b(?:the\s+)?remain(?:der|ing)\b', "remainder"),
    (r'\b(?:the\s+)?rest\b', "remainder"),
    (r'\bleft\s*over\b', "remainder"),
    (r'\bthat\s+(?:number|amount|many)\b', "previous"),
    (r'\bthis\s+(?:number|amount|many)\b', "previous"),
    (r'\bthe\s+(?:same\s+)?(?:number|amount)\b', "previous"),
    (r'\b(\d+)\s+times\s+(?:that|this|the)\b', "multiply_previous"),
    (r'\b(twice|double|triple|two|three|four|five|six|seven|eight|nine|ten)\s+times\s+(?:that|this|the)\b', "multiply_previous"),
    (r'\b(\d+)\s+times\s+(?:as\s+)?(?:many|much)\b', "multiply_previous"),
    (r'\b(twice|double|triple|two|three|four|five|six|seven|eight|nine|ten)\s+times\s+(?:as\s+)?(?:many|much)\b', "multiply_previous"),
    (r'\b(\d+)\s+more\s+than\b', "add_to_previous"),
    (r'\b(\d+)\s+(?:less|fewer)\s+than\b', "subtract_from_previous"),
]


@dataclass
class Token:
    """A lexed token from the source text."""
    type: str       # "number", "money", "fraction", "multiplier", "verb",
                    # "reference", "unit"
    value: float    # numeric value (or 0 for non-numeric)
    text: str       # original text
    position: int   # char position in source
    unit: str = ""      # what this number measures ("eggs", "dollars", "pounds")
    per_unit: str = ""  # if a rate: per what ("day", "egg", "comic_book")
    ref_type: str = ""  # for references: "remainder", "previous", etc.


def lex_sentence(sentence: str) -> list[Token]:
    """Extract tokens from a sentence."""
    tokens = []
    sent_lower = sentence.lower()

    # Money: $X.XX
    for m in MONEY_PATTERN.finditer(sentence):
        val = float(m.group(1).replace(',', ''))
        tokens.append(Token("money", val, m.group(), m.start(), unit="dollars"))

    # Fractions: X/Y
    for m in FRACTION_PATTERN.finditer(sentence):
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0:
            tokens.append(Token("fraction", num/den, m.group(), m.start()))

    # Multiplier words: half, twice, triple, etc.
    for word, val in MULTIPLIER_WORDS.items():
        for m in re.finditer(rf'\b{word}\b', sent_lower):
            tokens.append(Token("multiplier", val, word, m.start()))

    # Number words
    for word, val in NUMBER_WORDS.items():
        for m in re.finditer(rf'\b{word}\b', sent_lower):
            # Don't double-count if already captured as part of fraction/money
            if not any(abs(m.start() - t.position) < 3 for t in tokens):
                tokens.append(Token("number", val, word, m.start()))

    # Digit numbers (skip those already captured as money/fraction)
    for m in NUMBER_PATTERN.finditer(sentence):
        pos = m.start()
        if not any(abs(pos - t.position) < len(m.group()) + 1 for t in tokens):
            val = float(m.group(1).replace(',', ''))
            tokens.append(Token("number", val, m.group(), pos))

    # References
    for pattern, ref_type in REFERENCE_PATTERNS:
        for m in re.finditer(pattern, sent_lower):
            # Extract multiplier if present
            mult = 1.0
            if ref_type in ("multiply_previous", "add_to_previous",
                            "subtract_from_previous"):
                g = m.group(1) if m.lastindex else ""
                if g in MULTIPLIER_WORDS:
                    mult = MULTIPLIER_WORDS[g]
                elif g in NUMBER_WORDS:
                    mult = NUMBER_WORDS[g]
                elif g.isdigit():
                    mult = float(g)
            tokens.append(Token("reference", mult, m.group(), m.start(),
                               ref_type=ref_type))

    tokens.sort(key=lambda t: t.position)

    # --- Unit extraction ---
    # For each numeric token, find the noun it modifies and any rate context
    _assign_units(tokens, sent_lower)

    return tokens


# Words that are NOT units (stop words, verbs, adjectives)
_NOT_UNITS = {
    "a", "an", "the", "is", "are", "was", "were", "has", "have", "had",
    "to", "for", "of", "on", "in", "at", "by", "with", "from", "up",
    "and", "or", "but", "if", "then", "than", "that", "this", "it",
    "he", "she", "they", "his", "her", "its", "their", "him", "them",
    "be", "been", "being", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "shall", "must",
    "not", "no", "so", "as", "also", "just", "very", "too", "more",
    "most", "much", "many", "some", "any", "all", "each", "every",
    "per", "new", "old", "big", "small", "long", "short", "tall",
    "first", "second", "third", "last", "next",
}


def _find_head_noun(phrase: str) -> str:
    """
    Find the head noun of a phrase using the language graph (WordNet).

    "fresh duck egg" → "egg" (fresh=adj, duck=modifier, egg=head noun)
    "large red ball" → "ball"
    "hour" → "hour"

    Uses WordNet: if a word has noun synsets but no adjective synsets,
    it's likely the head noun. Take the LAST such word (English head-final).
    """
    try:
        from nltk.corpus import wordnet as wn
        words = phrase.strip().split()
        if len(words) == 1:
            return words[0]

        # Walk right-to-left (English heads are rightmost)
        for word in reversed(words):
            if word in _NOT_UNITS:
                continue
            noun_syns = wn.synsets(word, pos=wn.NOUN)
            adj_syns = wn.synsets(word, pos=wn.ADJ)
            # If it has noun senses, it's a candidate
            # If it ONLY has adj senses, skip it
            if noun_syns and not adj_syns:
                return word
            if noun_syns:
                # Has both noun and adj — still a candidate if it's last
                return word
        # Fallback: last word
        return words[-1]
    except Exception:
        # WordNet not available — fallback to last word
        return phrase.strip().split()[-1] if phrase.strip() else ""


def _assign_units(tokens: list[Token], sentence: str):
    """
    Assign unit nouns to numeric tokens based on surrounding context.

    "16 eggs per day" → value=16, unit="eggs", per_unit="day"
    "$2 per egg" → value=2, unit="dollars", per_unit="egg"
    "1/4 pound each" → value=0.25, unit="pound", per_unit inferred from subject
    """
    # Extract all words with positions
    word_spans = [(m.group(), m.start(), m.end())
                  for m in re.finditer(r'[a-z]+', sentence)]

    for token in tokens:
        if token.type not in ("number", "money", "fraction", "multiplier"):
            continue

        # Find the first noun AFTER this token
        token_end = token.position + len(token.text)
        for word, wstart, wend in word_spans:
            if wstart < token_end:
                continue
            if word in _NOT_UNITS:
                continue
            if word in NUMBER_WORDS or word in MULTIPLIER_WORDS:
                continue
            # Found a noun — assign as unit
            if not token.unit:
                token.unit = word
            break

        # Check for rate pattern: "per/each/every [noun]" after the unit
        # Use the language graph (WordNet) to find the head noun,
        # skipping adjectives and modifiers. "per fresh duck egg" → "egg"
        # Capture at most 4 words, stop at punctuation/conjunctions
        rate_match = re.search(
            rf'\b(?:per|each|every)\s+([a-z]+(?:\s+[a-z]+){{0,3}})',
            sentence[token_end:])
        if rate_match:
            # Trim at conjunctions/punctuation/verbs
            phrase = re.split(r'\b(?:and|or|but|she|he|they|who|that)\b',
                              rate_match.group(1))[0].strip()
            if phrase:
                per_word = _find_head_noun(phrase)
                if per_word and per_word not in _NOT_UNITS:
                    token.per_unit = per_word

        # "each" at end of clause = rate per the subject noun
        if re.search(r'\beach\s*[.,;!?]?\s*$', sentence[token_end:]):
            token.per_unit = "each"  # resolved later from context

        # Known unit conversions: "5 quarters" → value=5, unit="quarters"
        # The rate (0.25 $/quarter) is available via UNIT_TO_DOLLARS
        # "a dozen cookies" → value=12, unit="cookies"
        if token.unit in UNIT_MULTIPLIERS and token.type == "number":
            # Check what noun follows the multiplier
            # "3 pairs of shoes" → pair is the unit for shoes (no multiply)
            # "3 pairs of dice" → pair = 2 dice (multiply)
            real_unit = None
            for word, wstart, wend in word_spans:
                if wstart > token_end and word == token.unit:
                    for w2, ws2, we2 in word_spans:
                        if ws2 > wend and w2 not in _NOT_UNITS:
                            real_unit = w2
                            break
                    break

            # For clothing pairs, "pair" IS the unit — don't multiply
            if token.unit in ("pair", "pairs") and real_unit and \
               real_unit.lower() in _PAIR_IS_UNIT:
                token.unit = real_unit
            else:
                # "3 dozen cookies" → 3 × 12 = 36 cookies
                token.value = token.value * UNIT_MULTIPLIERS[token.unit]
                if real_unit:
                    token.unit = real_unit


# =====================================================================
# Parser: sentences → AST
# =====================================================================

@dataclass
class Clause:
    """A parsed sentence/clause with its semantic role."""
    text: str
    tokens: list[Token]
    role: str = ""          # "setup", "operation", "question"
    operation: str = ""     # verb → operation id
    verb: str = ""          # the triggering verb
    quantities: list = field(default_factory=list)  # Token values
    has_reference: bool = False
    ref_type: str = ""
    ref_multiplier: float = 1.0
    data_split: bool = False   # True if this question had data split out


def _classify_sentence(sentence: str, tokens: list[Token],
                       is_last: bool) -> Clause:
    """Classify a sentence and extract its semantic structure."""
    clause = Clause(text=sentence, tokens=tokens)
    sent_lower = sentence.lower()

    # Is this a question?
    if is_last and ("?" in sentence or
                    any(q in sent_lower for q in
                        ["how much", "how many", "what is", "what was",
                         "what will", "how old", "how far", "how long",
                         "find the", "calculate"])):
        clause.role = "question"
        # Still extract quantities from questions — some embed data
        clause.quantities = [t for t in tokens
                             if t.type in ("number", "money", "fraction")]
        return clause

    # Extract quantities (numbers and money)
    clause.quantities = [t for t in tokens
                         if t.type in ("number", "money", "fraction")]

    # Check for references
    refs = [t for t in tokens if t.type == "reference"]
    if refs:
        clause.has_reference = True
        clause.ref_type = refs[0].ref_type
        clause.ref_multiplier = refs[0].value

    # Identify the operation via verb matching
    # Priority: specific patterns first, then general verb lookup
    # Pattern: "sells/sold X for $Y" → the operation on the accumulator is multiply
    if re.search(r'\b(?:sells?|sold|makes?|earns?|charges?)\b.*\$', sent_lower):
        clause.operation = "multiply"
        clause.verb = "sells...for"
    elif re.search(r'\$.*\b(?:per|each|every)\b', sent_lower):
        clause.operation = "multiply"
        clause.verb = "at_rate"
    elif re.search(r'\b(?:per|each|every)\b.*\$', sent_lower):
        clause.operation = "multiply"
        clause.verb = "at_rate"
    else:
        # Resolve via language graph: synsets → operations
        # Greedy: phrases first, then individual words via WordNet
        from packages.core.language.resolve import resolve_sentence
        graph_result = resolve_sentence(sent_lower)
        if graph_result:
            clause.operation = graph_result.operation
            clause.verb = f"graph:{graph_result.source}:{graph_result.lemma}"
        else:
            # Fallback: curated verb index (will be deprecated as graph grows)
            from packages.math.operations import _VERB_INDEX
            for verb in sorted(_VERB_INDEX.keys(), key=lambda x: -len(x)):
                if " " in verb:
                    if verb in sent_lower:
                        clause.operation = _VERB_INDEX[verb]
                        clause.verb = verb
                        break
                else:
                    if re.search(rf'\b{re.escape(verb)}\b', sent_lower):
                        clause.operation = _VERB_INDEX[verb]
                        clause.verb = verb
                        break

            # Fallback 2: learned mappings (trained from GSM8K)
            if not clause.operation:
                learned_op, learned_word, learned_lift = _learned_vote(sent_lower)
                if learned_op and learned_lift >= 0.3:
                    clause.operation = learned_op
                    clause.verb = f"learned:{learned_word}"

    if not clause.operation and not clause.quantities:
        clause.role = "setup"
    elif not clause.operation:
        clause.role = "setup"
    else:
        clause.role = "operation"

    return clause


def parse(text: str) -> tuple[list[Clause], CompileContext]:
    """Parse text into classified clauses."""
    ctx = CompileContext(source=text)

    # Split into sentences
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                     if s.strip()]
    if not raw_sentences:
        ctx.unsolved_reason = "no sentences"
        return [], ctx

    clauses = []
    for i, sent in enumerate(raw_sentences):
        tokens = lex_sentence(sent)
        is_last = (i == len(raw_sentences) - 1)
        clause = _classify_sentence(sent, tokens, is_last)

        # Split "If [data], how many [question]?" into data + question
        # Guard: only split if previous clauses have very little numeric data
        # (otherwise the data is already parsed and splitting causes regressions)
        if clause.role == "question":
            prev_num_count = sum(
                len([t for t in c.tokens
                     if t.type in ("number", "money", "fraction")])
                for c in clauses)
            q_match = re.search(
                r'\b(how\s+(?:many|much)|what\s+(?:is|was|will))\b',
                sent, re.IGNORECASE)
            if q_match and q_match.start() > 10 and prev_num_count <= 1:
                # There's meaningful text before the question phrase
                # and previous clauses don't already carry the data
                data_part = sent[:q_match.start()].rstrip(', ')
                q_part = sent[q_match.start():]
                if data_part:
                    data_tokens = lex_sentence(data_part)
                    if data_tokens:
                        data_clause = _classify_sentence(
                            data_part, data_tokens, False)
                        clauses.append(data_clause)
                        ctx.sentences.append(data_part)
                        # Re-lex the question part
                        q_tokens = lex_sentence(q_part)
                        clause = _classify_sentence(q_part, q_tokens, True)
                        clause.data_split = True

        clauses.append(clause)
        ctx.sentences.append(sent)

    return clauses, ctx


# =====================================================================
# AST builder: clauses → tree
#
# Architecture: QUESTION FIRST
#   1. Parse the question → what unit/quantity is being asked for?
#   2. Find conversion clause → is there a rate between source and target?
#   3. Compute source quantity → accumulate from data clauses
#   4. Apply conversion → multiply source by rate to get target
# =====================================================================

# Common nouns that signal specific units
_MONEY_WORDS = {"dollar", "dollars", "money", "cost", "costs", "spend",
                "spent", "pay", "paid", "price", "earn", "earned",
                "profit", "income", "salary", "budget", "charge",
                "make", "makes", "made", "worth", "owe", "owes", "owed"}
_TIME_WORDS = {"hour", "hours", "minute", "minutes", "second", "seconds",
               "day", "days", "week", "weeks", "month", "months",
               "year", "years"}


def _extract_target(question: Clause) -> dict:
    """
    Parse the question to determine what is being asked for.

    Returns dict with:
      unit:      target noun ("dollars", "eggs", "elves", "cups", ...)
      question_type: "money", "count", "final_state", "difference", "rate"
      aggregation: "remaining", "total", "final", None
    """
    q = question.text.lower()
    words = set(re.findall(r'[a-z]+', q))
    target = {"unit": "", "question_type": "count", "aggregation": None}

    # --- Determine aggregation ---
    if any(w in q for w in ["left", "remain", "remaining", "still"]):
        target["aggregation"] = "remaining"
    elif any(w in q for w in ["total", "altogether", "combined", "all together",
                               "in all"]):
        target["aggregation"] = "total"
    elif any(w in q for w in ["final", "end", "now", "current", "currently"]):
        target["aggregation"] = "final"
    elif any(w in q for w in ["difference", "more than", "less than",
                               "fewer than"]):
        target["question_type"] = "difference"

    # --- Determine target unit ---
    # "How much in dollars" / "How much money"
    if words & _MONEY_WORDS or "how much" in q:
        target["question_type"] = "money"
        target["unit"] = "dollars"
        return target

    # "How many X" → X is the first NOUN after "many"
    # Skip function words (modals, pronouns, verbs) to find the real unit
    _SKIP_WORDS = {"can", "could", "will", "would", "shall", "should",
                   "may", "might", "must", "does", "did", "do", "is", "are",
                   "was", "were", "has", "have", "had",
                   "he", "she", "it", "they", "we", "you", "i",
                   "him", "her", "them", "us",
                   "be", "been", "being", "get", "got", "need",
                   "more", "still", "also", "then", "there"}
    m = re.search(r'how\s+many\s+([\w\s]+?)(?:\?|$)', q)
    if m:
        phrase_words = m.group(1).split()
        for w in phrase_words:
            if w not in _SKIP_WORDS and len(w) > 1:
                target["unit"] = w
                break
        return target

    # "What is the total/final X" → X
    m = re.search(r'what\s+(?:is|was|will\s+be)\s+(?:the\s+)?'
                  r'(?:total|final|new|current)?\s*(\w+)', q)
    if m:
        target["unit"] = m.group(1)
        return target

    # "How old/far/long" → specific unit
    if "how old" in q:
        target["unit"] = "years"
    elif "how far" in q:
        target["unit"] = "distance"
    elif "how long" in q:
        target["unit"] = "time"

    return target


def _find_rate_clause(clauses: list[Clause], target: dict) -> Optional[dict]:
    """
    Find a clause that provides a conversion rate to the target unit.

    Uses token-level unit data: any token with per_unit set is a rate.
    For money targets, also matches "sells...for $X" patterns.

    Returns dict with: rate_value, rate_op, clause_idx, rate_unit, per_unit
    """
    target_unit = target.get("unit", "")
    is_money = target.get("question_type") == "money"

    for i, clause in enumerate(clauses):
        if clause.role == "question":
            continue
        sent = clause.text.lower()

        # Look for tokens with per_unit set (rate tokens)
        rate_tokens = [t for t in clause.tokens
                       if t.per_unit and t.type in
                       ("number", "money", "fraction")]
        money_tokens = [t for t in clause.tokens if t.type == "money"]

        # Money target: "$X per [unit]" or "sells...for $X"
        if is_money:
            if money_tokens and rate_tokens:
                return {"rate_value": money_tokens[0].value,
                        "rate_op": "multiply",
                        "clause_idx": i,
                        "rate_unit": "dollars",
                        "per_unit": rate_tokens[0].per_unit}
            if money_tokens and re.search(
                    r'\b(?:sells?|sold|charges?|earns?)\b', sent):
                return {"rate_value": money_tokens[0].value,
                        "rate_op": "multiply",
                        "clause_idx": i,
                        "rate_unit": "dollars",
                        "per_unit": ""}

        # Count target: find rate whose unit matches target
        # e.g., target="cups", "2 cups per dozen" → rate
        if target_unit and rate_tokens:
            for rt in rate_tokens:
                if rt.unit and target_unit.startswith(rt.unit[:3]):
                    return {"rate_value": rt.value,
                            "rate_op": "multiply",
                            "clause_idx": i,
                            "rate_unit": rt.unit,
                            "per_unit": rt.per_unit}

    return None


def _try_unit_conversion(clauses: list[Clause], target: dict,
                         ctx: CompileContext) -> Optional[ASTNode]:
    """
    Handle problems where quantities need unit conversion before combining.

    Examples:
      "5 quarters and 2 dimes" → 5×0.25 + 2×0.10 = $1.45
      "3 feet and 6 inches" → 3×12 + 6 = 42 inches

    Only fires when tokens have known convertible units.
    """
    # Collect all numeric tokens with their units from ALL clauses
    # (questions often embed data: "If he has 8 quarters, 6 dimes...")
    all_tokens = []
    for clause in clauses:
        for t in clause.tokens:
            if t.type in ("number", "money", "fraction") and t.unit:
                all_tokens.append(t)

    if not all_tokens:
        return None

    # Determine which conversion table to use based on units present
    # Only actual coins need conversion — "dollars" and "$" are already dollars
    _COIN_UNITS = {"penny", "pennies", "nickel", "nickels",
                   "dime", "dimes", "quarter", "quarters"}
    coin_tokens = [t for t in all_tokens
                   if t.unit in _COIN_UNITS and t.type == "number"]
    target_is_cents = target.get("unit", "") in ("cents", "cent")
    target_is_money = target.get("question_type") == "money" or target_is_cents

    # Coin problems: "how much" (money) OR "how many can buy" (count)
    target_is_count = target.get("question_type") == "count"
    has_coins = coin_tokens and len(coin_tokens) >= 2

    if has_coins and (target_is_money or target_is_count):
        # Find if there's a per-item cost (for "how many can you buy" problems)
        item_cost = None
        for clause in clauses:
            sent = clause.text.lower()
            if re.search(r'\bcosts?\b', sent):
                # Check for money tokens first: "costs $0.05"
                for t in clause.tokens:
                    if t.type == "money":
                        item_cost = t.value
                        break
                # Check for coin-name costs: "cost a nickel"
                if item_cost is None:
                    cost_m = re.search(r'costs?\s+(?:a\s+)?(\w+)', sent)
                    if cost_m and cost_m.group(1) in UNIT_TO_DOLLARS:
                        item_cost = UNIT_TO_DOLLARS[cost_m.group(1)]
                # Check for number + "cents": "costs 55 cents"
                if item_cost is None:
                    cost_m = re.search(r'costs?\s+(\d+)\s*cents?', sent)
                    if cost_m:
                        item_cost = float(cost_m.group(1)) / 100
                if item_cost is not None:
                    break

        # For count-only questions (not money) without item_cost, skip
        if target_is_count and not target_is_money and item_cost is None:
            pass  # fall through
        else:
            # Convert all coins to dollars, sum them
            use_cents = target_is_cents
            multiplier = 100 if use_cents else 1
            terms = []
            for t in coin_tokens:
                rate = UNIT_TO_DOLLARS[t.unit]
                converted = rate * multiplier if use_cents else rate
                term = BinOp("multiply", Lit(t.value, f"{t.value:g}"),
                             Lit(converted,
                                 f"{t.unit}→{'cents' if use_cents else '$'}"))
                terms.append(term)

            ast = terms[0]
            for term in terms[1:]:
                ast = BinOp("add", ast, term)

            if target_is_count and item_cost is not None:
                # "how many can you buy" → total / cost_per_item
                ast = BinOp("divide", ast,
                            Lit(item_cost, f"cost:{item_cost:g}"))
            else:
                # "how much left" → subtract purchase cost
                for clause in clauses:
                    money = [t for t in clause.tokens if t.type == "money"]
                    cost_nums = [t for t in clause.tokens
                                 if t.type == "number"
                                 and t.unit in ("cents", "cent")
                                 and t not in coin_tokens]
                    sent = clause.text.lower()
                    if re.search(r'\b(?:buys?|bought|cost|costs|spend|spent|'
                                 r'pay|paid|price)\b', sent):
                        cost_tokens = cost_nums or money
                        if cost_tokens:
                            cost_val = cost_tokens[0].value
                            ast = BinOp("subtract", ast,
                                        Lit(cost_val, f"cost:{cost_val:g}"))
                            break

            ctx.debug.append(f"unit-conversion: {len(coin_tokens)} coins")
            return ast

    return None


# =====================================================================
# Algebra solver: variable chain substitution
# =====================================================================

# Stop words for entity extraction (not proper nouns)
_ALGEBRA_STOP = {
    "how", "what", "when", "where", "who", "why", "which",
    "the", "and", "but", "for", "not", "are", "was", "were", "has", "had",
    "she", "her", "his", "him", "they", "them", "its", "each", "every",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "after", "before", "then", "also", "both", "all", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten",
    "including", "because", "small", "large", "new", "old",
    "tree", "school", "farm", "farmer", "truck", "store",
    "christmas", "santa",
    "one", "two", "three", "first", "second", "third", "last",
    "there", "that", "this", "than", "with", "from", "into",
    "if", "on", "in", "at", "to", "by", "of", "or", "so",
}


@dataclass
class Relation:
    """A relationship between two named entities."""
    entity: str       # the entity being defined
    depends_on: str   # the entity it depends on
    op: str           # operation: "add", "subtract", "multiply", etc.
    value: float      # the numeric parameter


def _extract_entities(text: str) -> set[str]:
    """Extract named entities from text (capitalized proper nouns)."""
    entities = set()
    for m in re.finditer(r'\b([A-Z][a-z]{2,})\b', text):
        name = m.group(1).lower()
        if name not in _ALGEBRA_STOP:
            entities.add(name)
    return entities


def _find_entity_in_text(text: str, entities: set[str]) -> list[str]:
    """Find which entities are mentioned in a text span."""
    text_l = text.lower()
    return [e for e in entities if re.search(rf'\b{e}\b', text_l)]


def _extract_clause_relations(text: str, entities: set[str],
                              prev_entity: Optional[str] = None
                              ) -> list[Relation]:
    """Extract all relationships from a single clause/sub-clause.

    Strategy: find relationship keywords ("than", "twice"), then resolve
    subject entity (before keyword) and dependency entity (after keyword).
    """
    text_l = text.lower()
    results = []

    # Find entities mentioned in this clause and their positions
    ent_positions = []
    for e in entities:
        for m in re.finditer(rf'\b{e}\b', text_l):
            ent_positions.append((m.start(), e))
        # Also check possessives: "gissela's" → gissela
        for m in re.finditer(rf"\b{e}'s\b", text_l):
            ent_positions.append((m.start(), e))
    ent_positions.sort()

    if not ent_positions:
        return results

    def _entity_before(pos):
        """Closest entity BEFORE position."""
        for p, e in reversed(ent_positions):
            if p < pos:
                return e
        return ent_positions[0][1] if ent_positions else None

    def _entity_after(pos):
        """Closest entity AFTER position."""
        for p, e in ent_positions:
            if p > pos:
                return e
        return None

    # --- "N% more ... than" ---
    for m in re.finditer(r'(\d+)\s*%\s*more\b', text_l):
        pct = float(m.group(1))
        entity = _entity_before(m.start())
        # Find "than" after this match
        than_m = re.search(r'\bthan\b', text_l[m.end():])
        if than_m:
            dep = _entity_after(m.end() + than_m.start())
            if entity and dep and entity != dep:
                results.append(Relation(entity, dep, "percent_more", pct))

    # --- "N [words] more than twice/double" ---
    for m in re.finditer(
            r'(\d[\d,]*\.?\d*)\s+\w*\s*more\s+than\s+'
            r'(?:twice|double|two\s+times)\b',
            text_l):
        val = float(m.group(1).replace(',', ''))
        entity = _entity_before(m.start())
        dep = _entity_after(m.end())
        if not dep and prev_entity:
            dep = prev_entity  # "that" refers to previous entity
        if entity and dep and entity != dep:
            results.append(Relation(entity, dep, "twice_plus", val))

    # --- "N [words] more/older/taller/... than" ---
    if not results:  # don't double-match with twice_plus
        for m in re.finditer(
                r'(\d[\d,]*\.?\d*)\s+\w*\s*'
                r'(?:more|older|taller|heavier|larger|bigger|longer|higher|'
                r'greater|farther)\s+than\b',
                text_l):
            val = float(m.group(1).replace(',', ''))
            entity = _entity_before(m.start())
            dep = _entity_after(m.end())
            if entity and dep and entity != dep:
                results.append(Relation(entity, dep, "add", val))

    # --- "N [words] younger/less/shorter/... than" ---
    for m in re.finditer(
            r'(\d[\d,]*\.?\d*)\s+\w*\s*'
            r'(?:younger|less|shorter|lighter|smaller|fewer|lower|closer)'
            r'\s+than\b',
            text_l):
        val = float(m.group(1).replace(',', ''))
        entity = _entity_before(m.start())
        dep = _entity_after(m.end())
        if entity and dep and entity != dep:
            results.append(Relation(entity, dep, "subtract", val))

    # --- "twice/double [entity]" (without "more than") ---
    if not results:
        for m in re.finditer(r'\b(?:twice|double)\b', text_l):
            if 'more than' not in text_l[:m.start()].split('.')[-1]:
                entity = _entity_before(m.start())
                dep = _entity_after(m.end())
                if entity and dep and entity != dep:
                    results.append(Relation(entity, dep, "multiply", 2.0))

    # --- "half [of] [entity]" ---
    if not results:
        for m in re.finditer(r'\bhalf\b', text_l):
            entity = _entity_before(m.start())
            dep = _entity_after(m.end())
            if entity and dep and entity != dep:
                results.append(Relation(entity, dep, "divide", 2.0))

    return results


def _try_algebra(clauses: list[Clause],
                 ctx: CompileContext) -> Optional[ASTNode]:
    """
    Solve variable chain problems by substitution.

    Handles:
    - "X is N years older than Y" → X = Y + N
    - "X's truck can haul N more than Y's" → X = Y + N
    - "X has saved N% more than Y" → X = Y * (1 + N/100)
    - "X is N more than twice Y" → X = 2*Y + N
    - Combined/total constraints → remaining = total - sum(known)
    - "together" in question → sum all entities
    """
    full_text = " ".join(c.text for c in clauses)

    # --- Step 1: Extract named entities ---
    entities = _extract_entities(full_text)
    if len(entities) < 2:
        return None

    # --- Step 2: Extract relationships per-clause ---
    # Split compound clauses so "X older than Y, and Y younger than Z" both match
    relations: list[Relation] = []
    known: dict[str, float] = {}
    total_constraint: Optional[float] = None
    prev_entity: Optional[str] = None

    for clause in clauses:
        sent = clause.text
        sent_l = sent.lower()

        # Extract relationships from non-question clauses
        if clause.role != "question":
            # Split on ", and" to handle compound sentences
            sub_parts = re.split(r',\s*and\s+|\band\b', sent)

            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue

                # Find relationships in this sub-part
                rels = _extract_clause_relations(part, entities, prev_entity)
                for r in rels:
                    if not any(r2.entity == r.entity and r2.depends_on == r.depends_on
                               for r2 in relations):
                        relations.append(r)

                # Track most recently mentioned entity for pronoun resolution
                mentioned = _find_entity_in_text(part, entities)
                if mentioned:
                    prev_entity = mentioned[-1]

        # Extract given values from ALL clauses (including questions)
        mentioned = _find_entity_in_text(sent, entities)
        numbers = [t.value for t in clause.tokens
                   if t.type in ("number", "money", "fraction")]
        money = [t.value for t in clause.tokens if t.type == "money"]

        # Total/combined constraint
        if re.search(r'\b(?:total|combined|altogether)\b', sent_l) and numbers:
            total_constraint = max(numbers)
            continue

        # Given values — check at sub-clause level too
        # "Farmer Brown's farm is 200 acres, and Farmer Smith's farm is..."
        # Full clause has 2 entities, but sub-clause "Brown...200" has 1.
        if clause.role == "question":
            # "If Quinn is 30, how old is Trent?" — extract from conditional
            for m in re.finditer(r'(?:if\s+)?(\w+)\s+is\s+(\d+)', sent_l):
                name = m.group(1).lower()
                if name in entities:
                    known[name] = float(m.group(2))
        else:
            # Try sub-clause-level given value extraction
            sub_parts = re.split(r',\s*and\s+|\band\b', sent)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                part_l = part.lower()
                part_ents = _find_entity_in_text(part, entities)
                # Extract numbers from this sub-part
                part_nums = [float(m.group().replace(',', ''))
                             for m in re.finditer(r'\d[\d,]*\.?\d*', part)]
                part_money = [float(m.group(1).replace(',', ''))
                              for m in re.finditer(r'\$(\d[\d,]*\.?\d*)', part)]
                if len(part_ents) == 1 and (part_nums or part_money):
                    if not re.search(
                            r'\b(?:more\s+than|less\s+than|twice|double|'
                            r'half|times|%\s*more|younger\s+than|'
                            r'older\s+than)\b', part_l):
                        val = part_money[0] if part_money else part_nums[-1]
                        known[part_ents[0]] = val

    # No relationships → not an algebra problem
    if not relations:
        return None

    # --- Step 3: Identify leaf entities and validate givens ---
    defined_entities = {r.entity for r in relations}
    referenced_entities = {r.depends_on for r in relations}
    leaf_entities = referenced_entities - defined_entities
    known = {k: v for k, v in known.items() if k in leaf_entities}

    if not known:
        return None

    # --- Step 4: Substitute through chain ---
    for _ in range(10):
        progress = False
        for rel in relations:
            if rel.entity in known:
                continue
            if rel.depends_on not in known:
                continue

            base = known[rel.depends_on]
            if rel.op == "add":
                known[rel.entity] = base + rel.value
            elif rel.op == "subtract":
                known[rel.entity] = base - rel.value
            elif rel.op == "multiply":
                known[rel.entity] = base * rel.value
            elif rel.op == "divide":
                known[rel.entity] = base / rel.value if rel.value else 0
            elif rel.op == "twice_plus":
                known[rel.entity] = base * 2 + rel.value
            elif rel.op == "percent_more":
                known[rel.entity] = base * (1 + rel.value / 100)
            progress = True

        if not progress:
            break

    # --- Step 5: Handle total constraint ---
    if total_constraint is not None:
        unknown_ents = [e for e in entities if e not in known]
        if len(unknown_ents) == 1:
            known_sum = sum(known[e] for e in known if e in entities)
            known[unknown_ents[0]] = total_constraint - known_sum

    # --- Step 6: Find the target ---
    question = next((c for c in clauses if c.role == "question"), None)
    if not question:
        return None

    q_lower = question.text.lower()

    # "together" / "total" / "combined" → sum all entities
    if re.search(r'\b(?:together|total|combined|altogether|in all)\b', q_lower):
        if len(known) >= 2:
            result = sum(known.values())
            ctx.debug.append(f"algebra-sum: {known}")
            return Lit(result, f"sum={result:g}")

    # Find specific target entity in question
    target_entity = None
    for entity in known:
        if entity in q_lower:
            target_entity = entity

    if target_entity is None:
        for rel in reversed(relations):
            if rel.entity in known:
                target_entity = rel.entity
                break

    if target_entity and target_entity in known:
        result = known[target_entity]
        ctx.debug.append(f"algebra: {known}")
        return Lit(result, f"{target_entity}={result:g}")

    return None


@dataclass
class EqNode:
    """A node in the equation graph. Constants are known, unknowns are computed."""
    name: str        # identifier: "eggs_per_day", "c0_v0" (clause0_value0)
    value: Optional[float] = None   # known value (None if unknown)
    unit: str = ""
    per_unit: str = ""
    clause_idx: int = -1
    resolved_ast: Optional[ASTNode] = None  # how this was computed

    @property
    def known(self): return self.value is not None


@dataclass
class EqEdge:
    """An edge in the equation graph. Defines how target depends on sources."""
    target: str      # node name being defined
    sources: list    # node names it depends on
    op: str          # operation: "add", "subtract", "multiply", "divide"
    description: str = ""


def _try_equations(clauses: list[Clause],
                   ctx: CompileContext) -> Optional[ASTNode]:
    """
    General equation solver: constants are nodes, equations are edges.

    Every word problem is a graph:
      - Constants (given values) are leaf nodes
      - Equations (relationships) are directed edges
      - Intermediate results are internal nodes (constant + equation)
      - The answer is the root node

    Walk from known leaves → resolve internal nodes → reach target.
    Each internal node resolves to a sub-expression: its inputs
    combined by the edge operation. This is recursive substitution.

    The traversal accumulator: each node, as it's walked up,
    resolves to a complex statement. Chains union at resolution.
    """
    question = next((c for c in clauses if c.role == "question"), None)
    if not question:
        return None

    q_lower = question.text.lower()
    expanded = _expand_clauses(clauses)

    # --- Step 1: Build nodes (constants) ---
    nodes: dict[str, EqNode] = {}  # name → EqNode
    edges: list[EqEdge] = []

    # Extract every number as a node
    for ci, clause in enumerate(expanded):
        if clause.role == "question":
            # Still extract question numbers — some embed data
            for ti, t in enumerate(clause.tokens):
                if t.type in ("number", "money", "fraction", "multiplier"):
                    name = f"q_v{ti}"
                    nodes[name] = EqNode(name, t.value, t.unit, t.per_unit, ci)
            continue

        for ti, t in enumerate(clause.tokens):
            if t.type in ("number", "money", "fraction", "multiplier"):
                name = f"c{ci}_v{ti}"
                nodes[name] = EqNode(name, t.value, t.unit, t.per_unit, ci)

    if len(nodes) < 2:
        return None

    # --- Step 2: Build edges (equations) from clause structure ---

    # Group nodes by clause
    by_clause: dict[int, list[str]] = {}
    for name, node in nodes.items():
        by_clause.setdefault(node.clause_idx, []).append(name)

    # Identify setup clause (first clause with values, no operation verb)
    setup_nodes = []
    op_clauses = []  # (clause_idx, operation, node_names)
    rate_pairs = []  # (count_node, rate_node, per_unit)
    parallel_pairs = []  # (count_node, price_node)

    for ci, clause in enumerate(expanded):
        if clause.role == "question":
            continue
        clause_nodes = by_clause.get(ci, [])
        if not clause_nodes:
            continue

        sent_lower = clause.text.lower()

        # Rate pair detection: two values in same clause, one has per_unit
        if len(clause_nodes) == 2:
            n0, n1 = nodes[clause_nodes[0]], nodes[clause_nodes[1]]
            if n0.per_unit and not n1.per_unit:
                # n0 is rate, n1 is count (or vice versa)
                if n1.unit and _units_match_eq(n1.unit, n0.per_unit):
                    rate_pairs.append((clause_nodes[1], clause_nodes[0], n0.per_unit))
                else:
                    rate_pairs.append((clause_nodes[1], clause_nodes[0], n0.per_unit))
            elif n1.per_unit and not n0.per_unit:
                if n0.unit and _units_match_eq(n0.unit, n1.per_unit):
                    rate_pairs.append((clause_nodes[0], clause_nodes[1], n1.per_unit))
                else:
                    rate_pairs.append((clause_nodes[0], clause_nodes[1], n1.per_unit))

        # Parallel pair: count + money in same clause
        money_nodes = [n for n in clause_nodes if nodes[n].unit == "dollars"
                       or any(t.type == "money" for t in clause.tokens
                              if abs(t.value - nodes[n].value) < 0.01)]
        count_nodes = [n for n in clause_nodes if n not in money_nodes]
        if money_nodes and count_nodes:
            parallel_pairs.append((count_nodes[0], money_nodes[0]))

        # Classify by role
        if clause.role == "setup":
            setup_nodes.extend(clause_nodes)
        elif clause.role == "operation" and clause.operation:
            op_clauses.append((ci, clause.operation, clause_nodes))

    # --- Step 3: Build edges based on detected patterns ---

    # Pattern A: Parallel tracks (count × price pairs, then combine)
    if len(parallel_pairs) >= 2:
        product_nodes = []
        for count_name, price_name in parallel_pairs:
            prod_name = f"prod_{count_name}"
            nodes[prod_name] = EqNode(prod_name)
            edges.append(EqEdge(prod_name, [count_name, price_name],
                                "multiply", f"{count_name} × {price_name}"))
            product_nodes.append(prod_name)

        # Combine products
        is_difference = any(w in q_lower for w in
                           ["more than", "less than", "difference",
                            "how much more", "how much less"])
        if is_difference and len(product_nodes) == 2:
            result_name = "result"
            nodes[result_name] = EqNode(result_name)
            edges.append(EqEdge(result_name,
                                [product_nodes[1], product_nodes[0]],
                                "subtract", "difference of products"))
        else:
            result_name = "result"
            nodes[result_name] = EqNode(result_name)
            edges.append(EqEdge(result_name, product_nodes,
                                "add", "sum of products"))

        # Resolve the graph
        ast = _resolve_graph(nodes, edges, result_name)
        if ast:
            ctx.debug.append(f"eq-graph-parallel: {len(parallel_pairs)} pairs")
            return ast

    # Pattern B: Rate pairs (count × rate, then combine with remaining)
    if rate_pairs:
        product_nodes = []
        used_nodes = set()
        for count_name, rate_name, pu in rate_pairs:
            prod_name = f"prod_{count_name}"
            nodes[prod_name] = EqNode(prod_name)
            edges.append(EqEdge(prod_name, [count_name, rate_name],
                                "multiply", f"rate pair per {pu}"))
            product_nodes.append(prod_name)
            used_nodes.add(count_name)
            used_nodes.add(rate_name)

        # Sum rate products
        if len(product_nodes) > 1:
            sum_name = "rate_sum"
            nodes[sum_name] = EqNode(sum_name)
            edges.append(EqEdge(sum_name, product_nodes, "add", "sum of rate products"))
            current = sum_name
        else:
            current = product_nodes[0]

        # Apply remaining operations from op_clauses
        for ci, op, op_nodes in op_clauses:
            for n in op_nodes:
                if n not in used_nodes:
                    next_name = f"after_{n}"
                    nodes[next_name] = EqNode(next_name)
                    edges.append(EqEdge(next_name, [current, n],
                                        op, f"apply {op} {n}"))
                    current = next_name
                    used_nodes.add(n)

        ast = _resolve_graph(nodes, edges, current)
        if ast:
            ctx.debug.append(f"eq-graph-rate: {len(rate_pairs)} pairs")
            return ast

    # Pattern C: Accumulator (setup value + sequential operations)
    if setup_nodes and op_clauses:
        current = setup_nodes[0]
        used = {current}

        # For multi-value setup, sum them first
        if len(setup_nodes) > 1:
            sum_name = "setup_sum"
            nodes[sum_name] = EqNode(sum_name)
            edges.append(EqEdge(sum_name, setup_nodes, "add", "sum setup values"))
            current = sum_name
            used.update(setup_nodes)

        # Apply operations in order
        for ci, op, op_nodes in op_clauses:
            for n in op_nodes:
                if n not in used:
                    next_name = f"after_{n}"
                    nodes[next_name] = EqNode(next_name)
                    edges.append(EqEdge(next_name, [current, n],
                                        op, f"{op} {nodes[n].value:g}"))
                    current = next_name
                    used.add(n)

        ast = _resolve_graph(nodes, edges, current)
        if ast:
            ctx.debug.append(f"eq-graph-accum: {len(edges)} edges")
            return ast

    return None


def _try_chains(clauses: list[Clause],
                ctx: CompileContext) -> list[tuple[ASTNode, str]]:
    """
    Bottom-up chain solver: try all N-step computation chains.

    Chains evaluate UP from leaves to root. Each step combines
    two values (either leaf constants or prior step results)
    into an intermediate, building toward the answer.

    Depth adapts to problem size:
      2 values → 1 step (8 combos)
      3 values → 2 steps (~48 combos)
      4 values → 3 steps (~768 combos)
      5+ values → 3 steps on subsets

    Returns list of (ast, description) for each valid chain.
    """
    # Collect all numbers from clauses
    values = []  # (value, unit, per_unit, clause_idx, token_idx)
    for ci, clause in enumerate(clauses):
        for ti, t in enumerate(clause.tokens):
            if t.type in ("number", "money", "fraction", "multiplier"):
                values.append((t.value, t.unit, t.per_unit, ci, ti))

    # Inject implicit numbers from world knowledge
    full_text = " ".join(c.text.lower() for c in clauses)
    implicit = _get_implicit_numbers(full_text, values)
    values.extend(implicit)

    if len(values) < 2:
        return []

    nums = [v[0] for v in values]

    ops = ["multiply", "subtract", "add", "divide"]

    def apply_op(op, a, b):
        if op == "multiply": return a * b
        if op == "add": return a + b
        if op == "subtract": return a - b
        if op == "divide": return a / b if b != 0 else None
        return None

    results = []
    seen_values = set()

    def make_lit(idx):
        v = values[idx]
        return Lit(v[0], v[1])

    def add_result(ast, res):
        rkey = round(res, 4)
        if rkey in seen_values:
            return
        seen_values.add(rkey)
        results.append((ast, f"chain→{res}"))

    n = len(values)

    # 1-step: for 2-number problems
    if n >= 2:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for op in ops:
                    res = apply_op(op, nums[i], nums[j])
                    if res is not None and _is_finite(res):
                        ast = BinOp(op, make_lit(i), make_lit(j))
                        add_result(ast, res)

    # 2-step: combine pair → intermediate, then intermediate + any value
    # Allow reuse: step 2 can use a value already used in step 1
    # (e.g., "4 roses + 7 more = 11 dahlias; 4 + 11 = 15 total")
    if n >= 2:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for op1 in ops:
                    inter = apply_op(op1, nums[i], nums[j])
                    if inter is None or not _is_finite(inter):
                        continue
                    step1 = BinOp(op1, make_lit(i), make_lit(j))

                    for k in range(n):
                        for op2 in ops:
                            # Try both orderings for non-commutative ops
                            for left, right, left_ast, right_ast in [
                                (inter, nums[k], step1, make_lit(k)),
                                (nums[k], inter, make_lit(k), step1),
                            ]:
                                res = apply_op(op2, left, right)
                                if res is not None and _is_finite(res):
                                    ast = BinOp(op2, left_ast, right_ast)
                                    add_result(ast, res)

    # 3-step chains: step1 → step2 → step3
    # Allow reuse when n ≤ 4 values (tractable), require distinct for n > 4
    if n >= 2 and n <= 4:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for op1 in ops:
                    inter1 = apply_op(op1, nums[i], nums[j])
                    if inter1 is None or not _is_finite(inter1):
                        continue
                    step1 = BinOp(op1, make_lit(i), make_lit(j))

                    for k in range(n):
                        for op2 in ops:
                            for l1, r1, l1a, r1a in [
                                (inter1, nums[k], step1, make_lit(k)),
                                (nums[k], inter1, make_lit(k), step1),
                            ]:
                                inter2 = apply_op(op2, l1, r1)
                                if inter2 is None or not _is_finite(inter2):
                                    continue
                                step2 = BinOp(op2, l1a, r1a)

                                for m in range(n):
                                    for op3 in ops:
                                        for l2, r2, l2a, r2a in [
                                            (inter2, nums[m], step2, make_lit(m)),
                                            (nums[m], inter2, make_lit(m), step2),
                                        ]:
                                            res = apply_op(op3, l2, r2)
                                            if res is not None and _is_finite(res):
                                                ast = BinOp(op3, l2a, r2a)
                                                add_result(ast, res)

    # 3-step chains for 5-6 values: require distinct indices to stay tractable
    if 5 <= n <= 6:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for op1 in ops:
                    inter1 = apply_op(op1, nums[i], nums[j])
                    if inter1 is None or not _is_finite(inter1):
                        continue
                    step1 = BinOp(op1, make_lit(i), make_lit(j))

                    for k in range(n):
                        if k in (i, j):
                            continue
                        for op2 in ops:
                            for l1, r1, l1a, r1a in [
                                (inter1, nums[k], step1, make_lit(k)),
                                (nums[k], inter1, make_lit(k), step1),
                            ]:
                                inter2 = apply_op(op2, l1, r1)
                                if inter2 is None or not _is_finite(inter2):
                                    continue
                                step2 = BinOp(op2, l1a, r1a)

                                for m in range(n):
                                    if m in (i, j, k):
                                        continue
                                    for op3 in ops:
                                        for l2, r2, l2a, r2a in [
                                            (inter2, nums[m], step2, make_lit(m)),
                                            (nums[m], inter2, make_lit(m), step2),
                                        ]:
                                            res = apply_op(op3, l2, r2)
                                            if res is not None and _is_finite(res):
                                                ast = BinOp(op3, l2a, r2a)
                                                add_result(ast, res)

    # Also try parallel structure: (a op b) op3 (c op d)
    # Two independent pairs combined at the end
    if n >= 4:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for op1 in ops:
                    left_val = apply_op(op1, nums[i], nums[j])
                    if left_val is None or not _is_finite(left_val):
                        continue
                    left_ast = BinOp(op1, make_lit(i), make_lit(j))

                    for k in range(n):
                        if k in (i, j):
                            continue
                        for m in range(k + 1, n):
                            if m in (i, j):
                                continue
                            for op2 in ops:
                                right_val = apply_op(op2, nums[k], nums[m])
                                if right_val is None or not _is_finite(right_val):
                                    continue
                                right_ast = BinOp(op2, make_lit(k), make_lit(m))

                                for op3 in ops:
                                    res = apply_op(op3, left_val, right_val)
                                    if res is not None and _is_finite(res):
                                        ast = BinOp(op3, left_ast, right_ast)
                                        add_result(ast, res)

    return results


def _get_implicit_numbers(text: str, existing: list) -> list:
    """
    Inject implicit numbers from world knowledge.

    "5 quarters" → inject 25 (cents per quarter)
    "dozen cookies" → inject 12
    "per week" → inject 7 (days)
    "per year" → inject 12 (months) or 365 (days)
    "alphabet" → inject 26
    Ratio "5:9" → inject 9 if not already present
    """
    existing_vals = {v[0] for v in existing}
    implicit = []

    # Coin values: if we have quarters/dimes/nickels as units
    coin_vals = {"quarter": 25, "quarters": 25, "dime": 10, "dimes": 10,
                 "nickel": 5, "nickels": 5, "penny": 1, "pennies": 1}
    for coin, val in coin_vals.items():
        if coin in text and val not in existing_vals:
            implicit.append((val, "cents", "", -1, -1))
            existing_vals.add(val)

    # Time: week=7 days, year=12 months or 365 days
    if re.search(r'\bweek\b', text) and 7 not in existing_vals:
        implicit.append((7, "days", "", -1, -1))
        existing_vals.add(7)
    if re.search(r'\byear\b', text) and 12 not in existing_vals:
        implicit.append((12, "months", "", -1, -1))
        existing_vals.add(12)
    if re.search(r'\byear\b', text) and 365 not in existing_vals:
        implicit.append((365, "days", "", -1, -1))
        existing_vals.add(365)

    # Dozen
    if re.search(r'\bdozen\b', text) and 12 not in existing_vals:
        implicit.append((12, "", "", -1, -1))
        existing_vals.add(12)

    # Alphabet
    if re.search(r'\balphabet\b', text) and 26 not in existing_vals:
        implicit.append((26, "letters", "", -1, -1))
        existing_vals.add(26)

    # Score/points
    if re.search(r'\bscore\b', text) and 20 not in existing_vals:
        implicit.append((20, "", "", -1, -1))
        existing_vals.add(20)

    # Ratios: "5:9" or "5 to 9" — extract the second number if not present
    for m in re.finditer(r'(\d+)\s*:\s*(\d+)', text):
        v1, v2 = int(m.group(1)), int(m.group(2))
        if v1 not in existing_vals:
            implicit.append((v1, "", "", -1, -1))
            existing_vals.add(v1)
        if v2 not in existing_vals:
            implicit.append((v2, "", "", -1, -1))
            existing_vals.add(v2)

    # Percentage → decimal: if we have 25 (percent), inject 0.25
    for v, unit, _, _, _ in existing:
        if unit in ("percent", "%", "paycheck") or re.search(rf'\b{int(v) if v == int(v) else v}\s*%', text):
            decimal = v / 100
            if decimal not in existing_vals and 0 < decimal < 1:
                implicit.append((decimal, "", "", -1, -1))
                existing_vals.add(decimal)

    # Fraction complements: if we have 3/5 (0.6), inject 2/5 (0.4) and 1
    for v, unit, _, _, _ in existing:
        if 0 < v < 1:
            complement = 1 - v
            if complement not in existing_vals and complement > 0:
                implicit.append((complement, "", "", -1, -1))
                existing_vals.add(complement)
    if 1 not in existing_vals:
        # Inject 1 as the implicit whole
        implicit.append((1, "", "", -1, -1))
        existing_vals.add(1)

    # Unit conversions: feet→inches, hours→minutes, etc.
    if re.search(r'\bfeet\b|\bfoot\b', text) and re.search(r'\binch', text):
        if 12 not in existing_vals:
            implicit.append((12, "inches", "", -1, -1))
            existing_vals.add(12)
    if re.search(r'\bhour', text) and re.search(r'\bminute', text):
        if 60 not in existing_vals:
            implicit.append((60, "minutes", "", -1, -1))
            existing_vals.add(60)
    if re.search(r'\bday', text) and re.search(r'\bhour', text):
        if 24 not in existing_vals:
            implicit.append((24, "hours", "", -1, -1))
            existing_vals.add(24)
    if re.search(r'\bmonth', text) and re.search(r'\bday', text):
        if 30 not in existing_vals:
            implicit.append((30, "days", "", -1, -1))
            existing_vals.add(30)

    # For fractions extracted as decimals, inject the denominator
    # "a third" → 0.333, also inject 3
    for v, unit, _, _, _ in existing:
        if v > 0 and v < 1:
            # Check if 1/v is a clean integer
            recip = 1 / v
            if abs(recip - round(recip)) < 0.01 and round(recip) not in existing_vals:
                implicit.append((round(recip), "", "", -1, -1))
                existing_vals.add(round(recip))

    return implicit


def _is_finite(x):
    """Check if a number is finite and reasonable."""
    if x is None:
        return False
    try:
        return abs(x) < 1e12 and x == x  # not NaN
    except (TypeError, OverflowError):
        return False


def _units_match_eq(unit: str, per_unit: str) -> bool:
    """Check if a unit matches a per_unit for rate pairing."""
    if not unit or not per_unit:
        return False
    u = unit.lower().rstrip("s")
    p = per_unit.lower().rstrip("s")
    return u == p or u.startswith(p) or p.startswith(u)


def _resolve_graph(nodes: dict[str, EqNode], edges: list[EqEdge],
                   target: str) -> Optional[ASTNode]:
    """
    Resolve the equation graph from known nodes toward target.

    Each node resolves to an AST sub-expression. Known nodes become
    Lit nodes. Unknown nodes are computed by applying their edge
    operation to their resolved source nodes. This is recursive
    substitution — algebra IS recursion.
    """
    # Build edge lookup: target → edge
    edge_map: dict[str, EqEdge] = {}
    for e in edges:
        edge_map[e.target] = e

    resolved: dict[str, ASTNode] = {}

    def resolve(name: str, depth: int = 0) -> Optional[ASTNode]:
        if depth > 20:
            return None
        if name in resolved:
            return resolved[name]

        node = nodes.get(name)
        if node is None:
            return None

        # Known constant → leaf
        if node.known:
            ast = Lit(node.value, node.unit or "")
            resolved[name] = ast
            return ast

        # Unknown → resolve via edge
        edge = edge_map.get(name)
        if edge is None:
            return None

        # Resolve all sources
        source_asts = []
        for src in edge.sources:
            src_ast = resolve(src, depth + 1)
            if src_ast is None:
                return None
            source_asts.append(src_ast)

        if not source_asts:
            return None

        # Build AST from sources + operation
        if edge.op == "add" and len(source_asts) >= 2:
            ast = source_asts[0]
            for s in source_asts[1:]:
                ast = BinOp("add", ast, s)
        elif edge.op == "subtract" and len(source_asts) == 2:
            ast = BinOp("subtract", source_asts[0], source_asts[1])
        elif edge.op == "multiply" and len(source_asts) == 2:
            ast = BinOp("multiply", source_asts[0], source_asts[1])
        elif edge.op == "divide" and len(source_asts) == 2:
            ast = BinOp("divide", source_asts[0], source_asts[1])
        elif len(source_asts) == 1:
            ast = source_asts[0]
        else:
            # Fallback: chain with the operation
            ast = source_asts[0]
            for s in source_asts[1:]:
                ast = BinOp(edge.op, ast, s)

        resolved[name] = ast
        return ast

    return resolve(target)


def _expand_clauses(clauses: list[Clause]) -> list[Clause]:
    """Split compound sentences on 'and', 'then', commas.

    Sub-clauses inherit the parent's operation when they don't have one.
    "He gives 40 to A, 80 to B, 30 to C" → all three are 'subtract'.
    """
    expanded = []
    for clause in clauses:
        if clause.role == "question":
            expanded.append(clause)
            continue
        subparts = re.split(r'\band\b|,\s*then\b|\bthen\b|,\s*(?=[a-z0-9$])',
                            clause.text)
        if len(subparts) > 1:
            parent_op = clause.operation
            parent_verb = clause.verb
            for part in subparts:
                part = part.strip()
                if part:
                    tokens = lex_sentence(part)
                    sub = _classify_sentence(part, tokens, False)
                    # Inherit parent operation if sub-clause has none
                    if not sub.operation and parent_op:
                        sub.operation = parent_op
                        sub.verb = parent_verb
                        sub.role = "operation"
                    expanded.append(sub)
        else:
            expanded.append(clause)
    return expanded


def _compute_accumulator(clauses: list[Clause], skip_idx: int,
                         ctx: CompileContext) -> Optional[ASTNode]:
    """
    Compute a quantity by accumulating operations across clauses.
    Skips the rate clause (skip_idx) since that's the conversion, not source data.
    """
    accumulator: Optional[ASTNode] = None
    initial_val = None

    # Expand compound sentences
    expanded = _expand_clauses(
        [c for i, c in enumerate(clauses)
         if c.role != "question" and i != skip_idx])

    for clause in expanded:
        sent_lower = clause.text.lower()
        nums = [t for t in clause.tokens
                if t.type in ("number", "money", "fraction")]
        mults = [t for t in clause.tokens if t.type == "multiplier"]
        money = [t for t in clause.tokens if t.type == "money"]

        # No accumulator yet → first number is initial value
        if accumulator is None:
            if nums:
                initial_val = nums[0].value
                accumulator = Lit(initial_val)
                # If there are more numbers in this clause, check for rate
                # "16 eggs per day" → 16 is initial, "per day" is unit context
                # Don't chain additional numbers from setup clause
                continue
            elif money:
                initial_val = money[0].value
                accumulator = Lit(initial_val)
                continue
            else:
                continue

        # --- Pattern: fraction + consume verb ---
        if mults and any(re.search(rf'\b{w}\b', sent_lower) for w in
                         ["quit", "leave", "left", "place", "placed",
                          "remove", "removed", "eat", "ate", "eats",
                          "use", "used", "uses", "gave", "give", "gives",
                          "lost", "lose", "loses", "sold", "sell", "sells",
                          "spent", "spend", "spends", "took", "take",
                          "threw", "throw", "donated", "donate"]):
            fraction = mults[0].value
            consumed = BinOp("multiply", accumulator, Lit(fraction))
            accumulator = BinOp("subtract", accumulator, consumed)
            continue

        # --- Pattern: fraction + "of the remaining/rest" ---
        if mults and any(w in sent_lower for w in ["remaining", "rest"]):
            fraction = mults[0].value
            consumed = BinOp("multiply", accumulator, Lit(fraction))
            accumulator = BinOp("subtract", accumulator, consumed)
            continue

        # --- Pattern: "sells remainder for $X" → multiply ---
        if re.search(r'\b(?:sell|sold|earn|make|charge)\b', sent_lower):
            if money:
                accumulator = BinOp("multiply", accumulator,
                                    Lit(money[0].value,
                                        f"${money[0].value:g}"))
                continue

        # --- Pattern: subtract verbs with numbers ---
        if nums and any(re.search(rf'\b{w}\b', sent_lower) for w in
                        ["eat", "eats", "ate", "use", "used", "uses",
                         "bake", "baked", "bakes", "cook", "cooked",
                         "give", "gave", "gives", "lose", "lost", "loses",
                         "spend", "spent", "spends", "take", "took",
                         "remove", "removed", "throw", "threw",
                         "donate", "donated", "discard", "discarded",
                         "quit", "leave", "left", "break", "broke"]):
            for n in nums:
                accumulator = BinOp("subtract", accumulator,
                                    Lit(n.value, n.text))
            continue

        # --- Pattern: add verbs with numbers ---
        if nums and any(re.search(rf'\b{w}\b', sent_lower) for w in
                        ["add", "adds", "added", "hire", "hires", "hired",
                         "buy", "buys", "bought", "find", "finds", "found",
                         "get", "gets", "got", "receive", "received",
                         "collect", "collected", "join", "joined",
                         "more", "additional", "extra", "another",
                         "replace", "replaced", "gain", "gained",
                         "increase", "increased", "bring", "brought"]):
            for n in nums:
                accumulator = BinOp("add", accumulator,
                                    Lit(n.value, n.text))
            continue

        # --- Pattern: multiply verbs with numbers ---
        if nums and any(re.search(rf'\b{w}\b', sent_lower) for w in
                        ["times", "multiply", "per", "each", "every",
                         "rate"]):
            for n in nums:
                if n.value != initial_val:
                    accumulator = BinOp("multiply", accumulator,
                                        Lit(n.value, n.text))
            continue

        # --- Pattern: divide verbs with numbers ---
        if nums and any(re.search(rf'\b{w}\b', sent_lower) for w in
                        ["divide", "divided", "split", "share", "shared",
                         "distribute", "average"]):
            for n in nums:
                accumulator = BinOp("divide", accumulator,
                                    Lit(n.value, n.text))
            continue

        # --- Fallback: use learned vote for operation ---
        if nums:
            learned_op, _, lift = _learned_vote(sent_lower)
            if learned_op and lift >= 0.4:
                for n in nums:
                    accumulator = BinOp(learned_op, accumulator,
                                        Lit(n.value, n.text))
                continue

    return accumulator


# =====================================================================
# Entity-property-map solver
#
# Architecture:
#   1. Question → target unit
#   2. Chunk → one logical piece per entity/category
#   3. Interpret → entity: {property: value/rate} per chunk
#   4. Normalize → convert rates to common units
#   5. Compute → dimensional analysis from question to answer
#
# Example:
#   "Steve paints 3 boards every 30 minutes, paid $5/hour.
#    Joe paints 5 boards per hour, paid $7/hour.
#    How much will they get paid for painting 100 boards?"
#
#   → steve: {rate: 6 boards/hr, pay: $5/hr}
#     joe:   {rate: 5 boards/hr, pay: $7/hr}
#   → combined_rate: 11 boards/hr, combined_pay: $12/hr
#   → time = 100 boards / 11 boards/hr
#   → cost = time × $12/hr = $109.09
# =====================================================================

@dataclass
class EntityProp:
    """A property of an entity: a quantity with units."""
    value: float
    unit: str           # what it measures ("boards", "dollars")
    per_unit: str = ""  # if a rate, per what ("hour", "minute", "board")

    def __repr__(self):
        s = f"{self.value:g} {self.unit}"
        if self.per_unit:
            s += f"/{self.per_unit}"
        return s


def _normalize_rate(prop: EntityProp) -> EntityProp:
    """Normalize a rate to per-hour if it's a time rate, per-week if weekly."""
    if not prop.per_unit:
        return prop
    # Normalize time rates to per-hour
    if prop.per_unit in UNIT_TO_HOURS:
        factor = UNIT_TO_HOURS[prop.per_unit]
        if factor != 1:
            return EntityProp(prop.value / factor, prop.unit, "hour")
    # Normalize "per 30 minutes" → per hour
    return prop


def _try_entity_props(clauses: list[Clause], target: dict,
                      ctx: CompileContext) -> Optional[ASTNode]:
    """
    Entity-property-map solver.

    Chunks sentences by entity/category, builds property maps,
    then uses dimensional analysis to compute the answer.

    Currently handles:
    - Sum-then-divide: "6+4+2 gifts, 144 ribbon → how many per gift? → 144/12"
    - Sum-then-multiply: "2+4+12+1+2 per child × 3 children → 63"
    - Period scaling: "200+400+100 per month × 2 months → 1400"
    """
    question = next((c for c in clauses if c.role == "question"), None)
    if not question:
        return None

    q_lower = question.text.lower()
    non_q = [c for c in clauses if c.role != "question"]

    # Collect all numbers from ALL clauses (data can be in questions too)
    clause_nums: list[tuple[list, str]] = []  # (tokens, clause_text)
    for c in clauses:
        nums = [t for t in c.tokens if t.type in ("number", "money", "fraction")]
        if nums:
            clause_nums.append((nums, c.text.lower()))

    if not clause_nums:
        return None

    all_nums = [t for nums, _ in clause_nums for t in nums]
    q_nums = [t for t in question.tokens
              if t.type in ("number", "money", "fraction")]

    # --- Pattern: sum-then-divide ---
    # Q asks "per each/for each/per [unit]" → total_resource / count_of_items
    # gsm-0390: 6+4+2=12 gifts, 144 ribbon. "How many per each gift?" → 144/12
    # Guard: "each" alone isn't enough ("2 humps each" is a property, not division).
    #         Require "per each", "for each", or "per [noun]" in question.
    has_division_target = re.search(
        r'\b(?:per\s+each|for\s+each|per\s+\w+)\b', q_lower)
    if has_division_target and len(all_nums) >= 3:
        # The total resource is the outlier (much bigger than the others)
        sorted_nums = sorted(all_nums, key=lambda t: t.value)
        biggest = sorted_nums[-1]
        rest = sorted_nums[:-1]
        rest_sum = sum(n.value for n in rest)

        # Total must be bigger than the sum of the rest to be a resource
        if biggest.value > rest_sum and rest_sum > 0:
            ast = BinOp("divide", Lit(biggest.value), Lit(rest_sum))
            ctx.debug.append(
                f"entity-props: {biggest.value:g} / sum({rest_sum:g})")
            return ast

    # --- Pattern: sum-then-multiply ---
    # Multiple per-item quantities that need to be summed, then × count
    # gsm-0520: "2 skeins for hat, 4 for scarf, ... for 3 grandchildren"
    # gsm-0156: "200 bananas + 400 + 100 every month, order for 2 months"
    #
    # Detect: one clause has a single number + "each/every/for [N] [subjects]"
    # The rest have item-level quantities to sum.
    #
    # Strategy: find the multiplier by looking for a count associated with
    # the subjects/recipients (not items being counted).

    # Find a multiplier: number of subjects/recipients
    multiplier = None

    # Guard: don't look for multipliers if question asks for a difference
    is_difference_q = re.search(r'\b(?:difference|more than|less than|faster|slower)\b', q_lower)

    # Check question for a number (e.g., "how many for 2 months?")
    if not is_difference_q:
        for t in q_nums:
            # Numbers in questions that modify time/subject words
            m = re.search(rf'\b{int(t.value) if t.value == int(t.value) else t.value}\s+(\w+)',
                           q_lower)
            if m:
                word = m.group(1)
                if word in UNIT_TO_WEEKS or word in UNIT_TO_DAYS or word in UNIT_TO_HOURS:
                    # Time multiplier: "for 2 months"
                    multiplier = t.value
                    break

    # Check first non-question clause for a count of recipients
    if multiplier is None and not is_difference_q:
        for nums, sent_l in clause_nums:
            if len(nums) == 1 and re.search(
                    r'\b(?:for\s+(?:her|his|their)\s+\d|'
                    r'\d\s+(?:grandchild|child|student|friend|'
                    r'people|person|team|group|class))',
                    sent_l):
                multiplier = nums[0].value
                break

    if multiplier is not None and len(all_nums) >= 3:
        # Sum all numbers EXCEPT the multiplier
        item_sum = sum(n.value for n in all_nums if n.value != multiplier)
        if item_sum > 0:
            ast = BinOp("multiply", Lit(item_sum), Lit(multiplier))
            ctx.debug.append(
                f"entity-props: sum({item_sum:g}) × {multiplier:g}")
            return ast

    return None


# =====================================================================
# Percentage solver
#
# Handles problems where N% is applied to a base value.
# Three patterns:
#   "N% more than X"     → X * (1 + N/100)
#   "N% less/cheaper X"  → X * (1 - N/100)
#   "ate N% of X"        → remainder = X * (1 - N/100)
#   "N% of X"            → X * N/100
# =====================================================================

def _try_percentage(clauses: list[Clause], target: dict,
                    ctx: CompileContext) -> Optional[ASTNode]:
    """
    Percentage solver.

    Scans clauses for N% patterns, identifies the base value and direction,
    then computes. Handles multi-step percentage chains.
    """
    question = next((c for c in clauses if c.role == "question"), None)
    full_text = " ".join(c.text for c in clauses).lower()

    # Find all percentage values and their contexts
    pct_clauses = []
    for c in clauses:
        sent_lower = c.text.lower()
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', sent_lower)
        if pct_match:
            pct_val = float(pct_match.group(1))
            # Determine direction via language graph verb resolution
            # "more/increase" = add direction, "less/cheaper" = subtract direction
            from packages.core.language.resolve import resolve_sentence
            verb_result = resolve_sentence(sent_lower)
            if verb_result and verb_result.operation == "add":
                direction = "more"
            elif verb_result and verb_result.operation == "subtract":
                direction = "consumed"
            elif re.search(r'\b(?:cheaper|discount|off|lower|save)\b', sent_lower):
                # These are comparative adjectives, not verbs — language gap
                direction = "less"
            elif re.search(r'\b(?:more|additional|extra|higher|tax|markup)\b', sent_lower):
                direction = "more"
            else:
                direction = "of"

            # Find the base value: look for a number (not the %) in this or nearby clause
            base = None
            nums = [t for t in c.tokens
                    if t.type in ("number", "money") and t.value != pct_val]
            if nums:
                base = nums[0].value

            pct_clauses.append({
                "pct": pct_val, "direction": direction,
                "base": base, "clause": c, "text": sent_lower,
            })

    if not pct_clauses:
        return None

    # --- Pattern: single percentage applied to a known base ---
    # "X% more than $400" or "$100 is 40% cheaper"
    # Find base value from non-percentage clauses if not in same clause
    all_nums = []
    for c in clauses:
        for t in c.tokens:
            if t.type in ("number", "money"):
                all_nums.append(t.value)

    # Remove percentage values from the number pool
    pct_values = set(pc["pct"] for pc in pct_clauses)
    base_nums = [n for n in all_nums if n not in pct_values]

    # For each percentage clause without a base, find the right one
    for pc in pct_clauses:
        if pc["base"] is None and base_nums:
            # Look for "more/less than [noun]" → find that noun's value
            than_match = re.search(
                r'(?:more|less|cheaper|higher|lower)\s+than\s+(?:the\s+)?(\w+)',
                pc["text"])
            if than_match:
                ref_word = than_match.group(1)
                # Find the clause that defines this noun's value
                for c in clauses:
                    if ref_word in c.text.lower() and c is not pc["clause"]:
                        ref_nums = [t for t in c.tokens
                                    if t.type in ("number", "money")]
                        if ref_nums:
                            # Use the largest money/number value from that clause
                            pc["base"] = max(t.value for t in ref_nums)
                            break
            if pc["base"] is None:
                # Fallback: use the largest non-percentage number
                pc["base"] = max(base_nums)

    # --- Single percentage with base ---
    if len(pct_clauses) == 1 and pct_clauses[0]["base"] is not None:
        pc = pct_clauses[0]
        base = pc["base"]
        pct = pc["pct"]

        if pc["direction"] == "more":
            # "50% more than $400" → $400 * 1.5 = $600
            computed = BinOp("multiply", Lit(base),
                             Lit(1 + pct / 100))
            label = f"pct: {base:g} × (1 + {pct:g}%)"
        elif pc["direction"] in ("less", "consumed"):
            # "40% cheaper" → $100 * 0.6, or "ate 70%" → 20 * 0.3
            computed = BinOp("multiply", Lit(base),
                             Lit(1 - pct / 100))
            label = f"pct: {base:g} × (1 - {pct:g}%)"
        else:
            # "25% of $2000"
            computed = BinOp("multiply", Lit(base),
                             Lit(pct / 100))
            label = f"pct: {base:g} × {pct:g}%"

        # Check if question asks for total (both, together, all)
        q_lower = question.text.lower() if question else ""
        asks_total = bool(re.search(
            r'\b(?:both|total|altogether|together|all|combined|'
            r'how much (?:does|do|did|will).*cost)\b', q_lower))
        asks_remaining = bool(re.search(
            r'\b(?:left|remain|still)\b', q_lower))

        if asks_total and pc["direction"] in ("more", "less"):
            # "Both cars cost?" → base + computed
            ast = BinOp("add", Lit(base), computed)
            ctx.debug.append(f"{label} + base → total")
            return ast
        elif asks_remaining and pc["direction"] == "consumed":
            # "How many left?" → computed is already the remainder
            ctx.debug.append(label)
            return computed
        elif pc["direction"] == "more":
            # Check if question asks for ALL items (base + computed)
            # "how many fruits" when there are pears AND bananas
            # "how much does travel cost" when there are supplies AND tickets
            if asks_total or re.search(
                    r'\bhow (?:many|much)\s+(?:fruits?|items?|things?|'
                    r'(?:does|do|did|will).*(?:cost|spend|pay|need))\b',
                    q_lower):
                ast = BinOp("add", Lit(base), computed)
                ctx.debug.append(f"{label} + base → total")
                return ast
            ctx.debug.append(label)
            return computed
        else:
            ctx.debug.append(label)
            return computed

    # --- Multiple percentages: process sequentially ---
    if len(pct_clauses) >= 2:
        # Multi-step: each percentage might apply to a different base
        # Common pattern: "ate 70% of 20, ate 80% of 40, how many left?"
        results = []
        for pc in pct_clauses:
            if pc["base"] is None:
                continue
            base = pc["base"]
            pct = pc["pct"]
            if pc["direction"] in ("consumed", "less"):
                results.append(BinOp("multiply", Lit(base),
                                     Lit(1 - pct / 100)))
            elif pc["direction"] == "more":
                results.append(BinOp("multiply", Lit(base),
                                     Lit(1 + pct / 100)))
            else:
                results.append(BinOp("multiply", Lit(base),
                                     Lit(pct / 100)))

        if results:
            ast = results[0]
            for r in results[1:]:
                ast = BinOp("add", ast, r)
            ctx.debug.append(f"pct: {len(results)} percentage steps combined")
            return ast

    return None


def build_ast(clauses: list[Clause], ctx: CompileContext) -> Optional[ASTNode]:
    """
    Build an AST from parsed clauses.

    Architecture: CLASSIFY → DISPATCH
    1. What is the question asking for? (target unit)
    2. Classify the problem (language graph)
    3. Route to class-specific solvers
    4. Fallback to sequential accumulator
    """
    if not clauses:
        return None

    non_question = [c for c in clauses if c.role != "question"]
    if not non_question:
        ctx.unsolved_reason = "only questions, no content"
        return None

    # --- Step 1: What is the question asking for? ---
    question = next((c for c in clauses if c.role == "question"), None)
    target = _extract_target(question) if question else {
        "unit": "", "question_type": "count", "aggregation": None}

    ctx.debug.append(f"target: {target}")

    # --- Step 2: Classify the problem (language graph) ---
    from packages.core.language.classify import classify
    profile = classify(ctx.source)
    ctx.equivalence_classes = [c.cls for c in profile.classes if c.cls != 'sequential']
    ctx.debug.append(f"classes: {ctx.equivalence_classes}")

    # --- Step 3: Build ephemeral problem graph and walk backward ---
    from packages.core.problem_graph import build_graph, walk
    pg = build_graph(clauses, target)
    ctx.problem_graph = pg
    ctx.debug.extend(pg.debug)
    walk_result = walk(pg)
    ctx.walk_result = walk_result
    if walk_result:
        ctx.debug.extend(pg.debug[len(ctx.debug) - len(pg.debug):])
        ctx.debug.append(f"walk confidence: {walk_result.confidence:.2f}")

    # --- Unit conversion: coins→cents, time conversions, etc. ---
    unit_ast = _try_unit_conversion(clauses, target, ctx)
    if unit_ast:
        ctx.solver_used = "unit_conversion"
        return unit_ast

    # --- Multi-interpretation: try ALL solvers, collect candidates ---
    candidates = []  # (ast, solver_name, priority)

    # --- General equation solver (constants=nodes, equations=edges) ---
    eq_ast = _try_equations(clauses, ctx)
    if eq_ast:
        candidates.append((eq_ast, "equations", 4))

    # --- Named entity algebra (N equations, N unknowns) ---
    algebra_ast = _try_algebra(clauses, ctx)
    if algebra_ast:
        candidates.append((algebra_ast, "algebra", 10))

    if profile.has_class("percentage"):
        pct_ast = _try_percentage(clauses, target, ctx)
        if pct_ast:
            candidates.append((pct_ast, "percentage", 8))

    prop_ast = _try_entity_props(clauses, target, ctx)
    if prop_ast:
        candidates.append((prop_ast, "entity_props", 6))

    parallel_ast = _try_parallel_tracks(clauses, ctx)
    if parallel_ast:
        candidates.append((parallel_ast, "parallel_tracks", 7))

    ref_ast = _try_reference_chain(clauses, ctx)
    if ref_ast:
        candidates.append((ref_ast, "reference_chain", 5))

    rg_ast = _try_rate_groups(clauses, ctx)
    if rg_ast:
        candidates.append((rg_ast, "rate_groups", 7))

    rate = _find_rate_clause(clauses, target)
    if rate:
        qd_ast = _try_question_driven(clauses, target, ctx)
        if qd_ast:
            candidates.append((qd_ast, "question_driven", 4))

    seq_ast = _try_sequential(clauses, ctx)
    if seq_ast:
        candidates.append((seq_ast, "sequential", 2))

    # Last resort question-driven (without rate)
    if not rate:
        qd_ast = _try_question_driven(clauses, target, ctx)
        if qd_ast:
            candidates.append((qd_ast, "question_driven", 1))

    # Walk result as candidate
    if walk_result and walk_result.ast:
        walk_priority = int(walk_result.confidence * 10)
        candidates.append((walk_result.ast, "graph_walk", walk_priority))

    # --- Bottom-up chain solver: exhaustive search over all chains ---
    # Every possible computation graph over the problem's numbers.
    # ALL results go into the candidate pool — this is the real solver.
    # Selection is a separate problem.
    chain_results = _try_chains(clauses, ctx)
    if chain_results:
        for c_ast, desc in chain_results:
            candidates.append((c_ast, "chain", 0))
        ctx.debug.append(f"chains: {len(chain_results)} total")

    if not candidates:
        ctx.unsolved_reason = "no pattern matched"
        return None

    # Score each candidate by STRATEGY confidence, not static priority
    scored = []
    for c_ast, name, pri in candidates:
        strat_score = score_strategy(c_ast, clauses, ctx)
        # Blend: 70% strategy score, 30% solver prior (normalized 0-1)
        blended = strat_score * 0.7 + (pri / 10.0) * 0.3
        scored.append((c_ast, name, pri, strat_score, blended))

    scored.sort(key=lambda x: -x[4])
    best_ast, best_solver, _, best_strat, best_blend = scored[0]
    ctx.solver_used = best_solver
    # Store candidate metadata (ASTs evaluated lazily in compile())
    ctx._candidate_asts = [(ast, name, pri) for ast, name, pri, _, _ in scored]
    ctx._candidate_scores = [(name, pri, ss, bl) for _, name, pri, ss, bl in scored]
    ctx.debug.append(f"candidates: {[(n, f's={ss:.2f}', f'b={bl:.2f}') for _, n, _, ss, bl in scored]}")
    return best_ast


def _try_question_driven(clauses: list[Clause], target: dict,
                         ctx: CompileContext) -> Optional[ASTNode]:
    """
    Question-driven AST construction.

    1. Find conversion rate (if target is money or different unit)
    2. Compute source quantity from remaining clauses
    3. Apply conversion
    """
    # --- Step 2: Find conversion rate ---
    rate = _find_rate_clause(clauses, target)
    skip_idx = rate["clause_idx"] if rate else -1

    # --- Step 3: Compute source quantity ---
    source = _compute_accumulator(clauses, skip_idx, ctx)
    if source is None:
        return None

    # --- Step 4: Apply conversion ---
    if rate:
        ast = BinOp(rate["rate_op"], source,
                     Lit(rate["rate_value"], f"rate:{rate['rate_value']:g}"))
        ctx.debug.append(f"question-driven: source→convert "
                         f"(rate={rate['rate_value']:g})")
        return ast

    ctx.debug.append("question-driven: direct accumulator")
    return source


def _try_parallel_tracks(clauses: list[Clause],
                         ctx: CompileContext) -> Optional[ASTNode]:
    """
    Detect and build parallel tracks:
    multiple items × prices, summed together.
    """
    # Look for clauses that pair a count with a price
    # The count might be in a different sentence from the price
    # Strategy: first collect all counts and all prices, then pair them
    money_clauses = []

    # Extract counts from setup clauses (e.g., "bought 3 shorts, 3 pants, 3 shoes")
    counts = []
    for clause in clauses:
        if clause.role == "question":
            continue
        nums = [t for t in clause.tokens if t.type == "number"]
        money = [t for t in clause.tokens if t.type == "money"]
        if nums and not money:
            for n in nums:
                counts.append(n.value)

    # Extract prices from clauses with money
    prices = []
    for clause in clauses:
        if clause.role == "question":
            continue
        money = [t for t in clause.tokens if t.type == "money"]
        nums = [t for t in clause.tokens if t.type == "number"]
        for m in money:
            # If there's a meaningful count in the same clause (not "one"),
            # use it. "One pair costs $X" — "one" is a determiner, not a count.
            real_counts = [n for n in nums if n is not m and n.value > 1]
            if real_counts:
                money_clauses.append((real_counts[0].value, m.value))
            else:
                prices.append(m.value)

    # Pair remaining prices with counts (in order)
    if prices and counts:
        # Filter counts > 1 (skip "one" determiners)
        real_counts = [c for c in counts if c > 1]
        if not real_counts:
            real_counts = counts

        # If we have a single repeated count (e.g., "3 of each"), use it for all
        if len(set(real_counts)) == 1:
            shared_count = real_counts[0]
            for price in prices:
                money_clauses.append((shared_count, price))
        elif len(real_counts) >= len(prices):
            for i, price in enumerate(prices):
                money_clauses.append((real_counts[i], price))

    if len(money_clauses) >= 2:
        terms = [BinOp("multiply", Lit(count, f"{count:g}x"),
                        Lit(price, f"${price:g}"))
                 for count, price in money_clauses]

        # Check if question asks for a difference ("how much more/less")
        question = next((c for c in clauses if c.role == "question"), None)
        q_text = question.text.lower() if question else ""
        is_difference = any(w in q_text for w in
                            ["more than", "less than", "difference",
                             "how much more", "how much less",
                             "how much fewer"])

        if is_difference and len(terms) == 2:
            # Subtract smaller from larger (or second from first contextually)
            ast = BinOp("subtract", terms[1], terms[0])
            ctx.debug.append(f"parallel: 2 tracks → difference")
        else:
            ast = terms[0]
            for term in terms[1:]:
                ast = BinOp("add", ast, term)
            ctx.debug.append(f"parallel: {len(money_clauses)} tracks → sum")
        return ast

    return None


def _try_reference_chain(clauses: list[Clause],
                         ctx: CompileContext) -> Optional[ASTNode]:
    """
    Detect and build reference chains:
    "bought 4. Then three times that. Then 5 times the Tuesday number."
    Result: sum of all intermediate values.
    """
    # Look for explicit back-references: "three times that", "twice as many"
    # "remainder" is an accumulator concept, not a reference chain
    chain_ref_types = {"multiply_previous", "add_to_previous",
                       "subtract_from_previous", "previous"}
    ref_clauses = [c for c in clauses
                   if c.role == "operation" and c.has_reference
                   and c.ref_type in chain_ref_types]

    if not ref_clauses:
        return None

    # Find initial value from first non-question, non-ref clause
    initial = None
    for clause in clauses:
        if clause.role == "question" or clause.has_reference:
            continue
        nums = [t for t in clause.tokens if t.type == "number"]
        if nums:
            initial = nums[0].value
            break

    if initial is None:
        return None

    # Build chain: each step either references previous or is a new value
    values: list[ASTNode] = [Lit(initial)]
    previous: ASTNode = Lit(initial)

    for clause in clauses:
        if clause.role == "question":
            continue
        if clause is clauses[0] and not clause.has_reference:
            continue  # skip initial value clause

        if not clause.has_reference:
            # Non-reference clause with numbers → new independent value
            nums = [t for t in clause.tokens if t.type == "number"]
            if nums:
                new_val = Lit(nums[0].value)
                values.append(new_val)
                previous = new_val
            continue

        if clause.ref_type == "multiply_previous":
            mult = clause.ref_multiplier
            new_val = BinOp("multiply", previous, Lit(mult, f"×{mult:g}"))
            values.append(new_val)
            previous = new_val
        elif clause.ref_type == "add_to_previous":
            amt = clause.ref_multiplier
            new_val = BinOp("add", previous, Lit(amt))
            values.append(new_val)
            previous = new_val
        elif clause.ref_type == "subtract_from_previous":
            amt = clause.ref_multiplier
            new_val = BinOp("subtract", previous, Lit(amt))
            values.append(new_val)
            previous = new_val
        elif clause.ref_type in ("previous", "remainder"):
            # Operation on previous value with explicit operand
            op = clause.operation
            if op:
                for t in clause.tokens:
                    if t.type in ("number", "money", "fraction"):
                        new_val = BinOp(op, previous, Lit(t.value, t.text))
                        values.append(new_val)
                        previous = new_val
                        break

    # Question context: "how many total/after all" → sum all values
    # "how many now/left" → just the last value
    question = next((c for c in clauses if c.role == "question"), None)
    q_text = question.text.lower() if question else ""

    if any(w in q_text for w in ["total", "all", "altogether", "combined"]):
        # Sum all intermediate values
        ast = values[0]
        for v in values[1:]:
            ast = BinOp("add", ast, v)
        ctx.debug.append(f"reference chain: sum of {len(values)} values")
        return ast
    else:
        # Return last value
        ctx.debug.append(f"reference chain: last of {len(values)} values")
        return previous


def _try_rate_groups(clauses: list[Clause],
                     ctx: CompileContext) -> Optional[ASTNode]:
    """
    Rate-group solver: when tokens in a clause share a per_unit,
    multiply them within groups, sum the products, then apply
    remaining operations from other clauses.

    Example: "6 decks with 25 cards in each deck and 5 boxes with
    40 cards in each box. She keeps 50 and gives the rest to 10 students."
    → (6×25 + 5×40 - 50) / 10 = 30
    """
    from collections import defaultdict

    # Step 1: Find clauses with rate-grouped tokens
    rate_groups = defaultdict(list)  # per_unit → [(value, unit, clause_idx)]
    rate_clause_idxs = set()

    for i, clause in enumerate(clauses):
        if clause.role == "question":
            continue
        for t in clause.tokens:
            if t.per_unit and t.type in ("number", "money", "fraction"):
                rate_groups[t.per_unit].append((t.value, t.unit or "", i))
                rate_clause_idxs.add(i)

    # Only fire when groups have exactly 2 tokens AND one token's unit
    # matches the per_unit (it's the count of that container).
    # This distinguishes "6 decks with 25 cards per deck" (valid)
    # from "60 girls and 5 students per teacher" (invalid — neither is
    # a count of teachers).
    paired_groups = {}
    for pu, tokens in rate_groups.items():
        if len(tokens) != 2:
            continue
        # Check if at least one token's unit matches per_unit
        # (singularized: "decks" → "deck")
        def unit_matches_pu(unit_str, pu_str):
            if not unit_str or not pu_str:
                return False
            u = unit_str.lower().rstrip("s")
            p = pu_str.lower().rstrip("s")
            return u == p or u.startswith(p) or p.startswith(u)

        t0_match = unit_matches_pu(tokens[0][1], pu)
        t1_match = unit_matches_pu(tokens[1][1], pu)
        # Need exactly one match (count) and one non-match (content)
        if (t0_match and not t1_match) or (t1_match and not t0_match):
            paired_groups[pu] = tokens

    if not paired_groups:
        return None

    # Step 2: Multiply within each paired group
    products = []
    for per_unit, tokens in paired_groups.items():
        a, b = tokens[0][0], tokens[1][0]
        products.append(BinOp("multiply",
                              Lit(a, f"{a:g}"),
                              Lit(b, f"{b:g}")))

    # Sum the products
    accumulator = products[0]
    for p in products[1:]:
        accumulator = BinOp("add", accumulator, p)

    # Step 3: Apply remaining operations from non-rate clauses
    for i, clause in enumerate(clauses):
        if clause.role == "question":
            continue
        if i in rate_clause_idxs:
            continue

        sent_lower = clause.text.lower()
        nums = [t for t in clause.tokens
                if t.type in ("number", "money", "fraction")]
        if not nums:
            continue

        # Per-number operation: check local context for each number
        for n in nums:
            n_pos = sent_lower.find(n.text.lower()) if n.text else -1

            # Check if number follows a division marker
            is_division_target = False
            for div_word in ["equally", "evenly", "split", "share",
                             "between", "among", "distribute"]:
                dw_pos = sent_lower.find(div_word)
                if dw_pos >= 0 and n_pos > dw_pos:
                    is_division_target = True
                    break

            if is_division_target:
                accumulator = BinOp("divide", accumulator,
                                    Lit(n.value, n.text))
            elif any(w in sent_lower for w in
                     ["keep", "keeps", "kept", "subtract", "minus",
                      "lose", "lost", "spend", "spent",
                      "remove", "take", "took"]):
                accumulator = BinOp("subtract", accumulator,
                                    Lit(n.value, n.text))
            elif any(w in sent_lower for w in
                     ["give", "gave", "gives"]):
                accumulator = BinOp("subtract", accumulator,
                                    Lit(n.value, n.text))
            elif clause.operation:
                accumulator = BinOp(clause.operation, accumulator,
                                    Lit(n.value, n.text))
            else:
                accumulator = BinOp("subtract", accumulator,
                                    Lit(n.value, n.text))

    ctx.debug.append(
        f"rate_groups: {len(paired_groups)} groups "
        f"({', '.join(paired_groups.keys())})")
    return accumulator


def _try_sequential(clauses: list[Clause],
                    ctx: CompileContext) -> Optional[ASTNode]:
    """
    Sequential accumulator: start with a value, apply operations in order.
    Handles: subtract, add, multiply by rate, fractions ("a third quit").
    """
    # First clause with numbers defines the initial value.
    # If it also has an operation AND multiple numbers, the operation
    # applies to subsequent numbers in the same clause (not the initial).
    initial_val = None
    first_clause_idx = 0
    first_clause_extra_nums = []  # numbers after the initial in first clause

    for i, clause in enumerate(clauses):
        if clause.role == "question":
            continue
        nums = [t for t in clause.tokens
                if t.type in ("number", "money", "fraction")]
        if nums:
            initial_val = nums[0].value
            first_clause_idx = i
            # If first clause has an operation AND more numbers,
            # keep the operation for those extra numbers
            if clause.operation and len(nums) > 1:
                first_clause_extra_nums = nums[1:]
                # Don't strip the operation — we'll apply it to extra nums
            else:
                clause.role = "setup"
                clause.operation = ""
            break

    if initial_val is None:
        ctx.unsolved_reason = "no initial value"
        return None

    accumulator: ASTNode = Lit(initial_val)

    # If first clause has an operation with extra numbers, decide when to apply:
    # - "divide between N" / "split among N" → DEFER to end (it's aggregation)
    # - Other operations → apply immediately
    deferred_op = None
    if first_clause_extra_nums:
        first_clause = clauses[first_clause_idx]
        first_op = first_clause.operation
        first_text = first_clause.text.lower()
        if first_op:
            is_deferred = first_op == "divide" and any(
                w in first_text for w in
                ["between", "among", "equally", "evenly", "split", "share"])
            if is_deferred:
                # Save for after all other operations
                deferred_op = (first_op, first_clause_extra_nums)
            else:
                for n in first_clause_extra_nums:
                    accumulator = BinOp(first_op, accumulator,
                                        Lit(n.value, n.text))

    # Split compound sentences — sub-clauses inherit parent operation
    expanded_clauses = _expand_clauses(clauses[first_clause_idx + 1:])

    for clause in expanded_clauses:
        if clause.role == "question":
            # Extract "if [data]" from questions where data was already
            # split out by the parser (trailing conditions with numbers)
            if not clause.data_split:
                continue
            q_lower = clause.text.lower()
            if_match = re.search(r'\bif\s+', q_lower)
            if if_match:
                if_text = clause.text[if_match.start():]
                if_tokens = lex_sentence(if_text)
                if_nums = [t for t in if_tokens
                           if t.type in ("number", "money", "fraction")]
                if if_nums:
                    if_clause = _classify_sentence(if_text, if_tokens, False)
                    if if_clause.operation:
                        sent_lower = if_clause.text.lower()
                        op = if_clause.operation
                        nums = if_nums
                        mults = [t for t in if_tokens
                                 if t.type == "multiplier"]
                        # Fall through to process as data below
                    else:
                        continue
                else:
                    continue
            else:
                continue
        else:
            sent_lower = clause.text.lower()
            op = clause.operation
            nums = [t for t in clause.tokens
                    if t.type in ("number", "money", "fraction")]
            mults = [t for t in clause.tokens if t.type == "multiplier"]

        # --- Pattern: fraction + consume verb ---
        # "a third of the elves quit" → subtract(accumulator * 1/3)
        # "places a quarter of the pieces" → subtract(accumulator * 1/4)
        if mults and any(w in sent_lower for w in
                         ["quit", "leave", "left", "place", "places", "placed",
                          "remove", "removes", "removed",
                          "eat", "ate", "eats", "use", "used", "uses",
                          "gave", "give", "gives",
                          "lost", "lose", "loses",
                          "broke", "break", "breaks",
                          "sold", "sell", "sells",
                          "spent", "spend", "spends",
                          "took", "take", "takes",
                          "threw", "throw", "throws",
                          "donated", "donate", "donates"]):
            fraction = mults[0].value
            consumed = BinOp("multiply", accumulator, Lit(fraction))
            accumulator = BinOp("subtract", accumulator, consumed)
            continue

        # --- Pattern: fraction + "of the remaining/rest" ---
        # "a third of the remaining pieces" → subtract from what's left
        if mults and any(w in sent_lower for w in ["remaining", "rest", "left"]):
            fraction = mults[0].value
            consumed = BinOp("multiply", accumulator, Lit(fraction))
            accumulator = BinOp("subtract", accumulator, consumed)
            continue

        # --- Pattern: "sells remainder for $X per Y" → multiply ---
        if re.search(r'sell|sold|earn|make|charge', sent_lower):
            money = [t for t in clause.tokens if t.type == "money"]
            if money:
                accumulator = BinOp("multiply", accumulator,
                                    Lit(money[0].value, f"${money[0].value:g}"))
                continue

        # --- Pattern: explicit operation with number ---
        if op and nums:
            for n in nums:
                # Don't re-use the initial value as an operand
                if n.value == initial_val and clause == clauses[first_clause_idx]:
                    continue
                accumulator = BinOp(op, accumulator, Lit(n.value, n.text))
            continue

        # --- Pattern: explicit operation with multiplier ---
        if op and mults:
            accumulator = BinOp(op, accumulator, Lit(mults[0].value))
            continue

        # --- Pattern: "hires/adds N more" ---
        if nums and any(w in sent_lower for w in
                        ["hire", "hires", "hired", "add", "adds", "added",
                         "more", "additional", "extra", "another",
                         "replace", "replaces", "replaced"]):
            for n in nums:
                accumulator = BinOp("add", accumulator, Lit(n.value, n.text))
            continue

        # --- Pattern: consume/remove/take N ---
        # Catches verbs the language graph missed: "snuck", "stole", "lost"
        if nums and any(w in sent_lower for w in
                        ["snuck", "sneak", "sneaks",
                         "stole", "steal", "steals",
                         "lost", "lose", "loses",
                         "dropped", "drop", "drops",
                         "ruined", "ruin", "ruins",
                         "damaged", "damage", "damages",
                         "forgot", "forget", "forgets",
                         "wasted", "waste", "wastes"]):
            for n in nums:
                accumulator = BinOp("subtract", accumulator, Lit(n.value, n.text))
            continue

    # Apply deferred aggregation operation (e.g., "divide between 6")
    if deferred_op:
        d_op, d_nums = deferred_op
        for n in d_nums:
            accumulator = BinOp(d_op, accumulator, Lit(n.value, n.text))

    return accumulator


# =====================================================================
# Executor: AST → number
# =====================================================================

def execute(ast: ASTNode) -> Optional[float]:
    """Walk the AST bottom-up and compute the result."""
    if ast is None:
        return None

    if isinstance(ast, Lit):
        return ast.value

    if isinstance(ast, Neg):
        child = execute(ast.child)
        return -child if child is not None else None

    if isinstance(ast, Ref):
        if ast.resolved:
            return execute(ast.resolved)
        return None

    if isinstance(ast, BinOp):
        left = execute(ast.left)
        right = execute(ast.right)
        if left is None or right is None:
            return None

        op = OPERATIONS.get(ast.op)
        if op:
            try:
                return op.curry(left, right)
            except Exception:
                return None
        return None

    return None


# =====================================================================
# Confidence scoring
# =====================================================================

def _ast_numbers(node: ASTNode) -> list[float]:
    """Extract all literal values used in an AST."""
    if isinstance(node, Lit):
        return [node.value]
    if isinstance(node, Neg):
        return _ast_numbers(node.child)
    if isinstance(node, Ref):
        return _ast_numbers(node.resolved) if node.resolved else []
    if isinstance(node, BinOp):
        return _ast_numbers(node.left) + _ast_numbers(node.right)
    return []


def _ast_ops(node: ASTNode) -> list[str]:
    """Extract all operations in an AST."""
    if isinstance(node, BinOp):
        return [node.op] + _ast_ops(node.left) + _ast_ops(node.right)
    if isinstance(node, Neg):
        return _ast_ops(node.child)
    if isinstance(node, Ref) and node.resolved:
        return _ast_ops(node.resolved)
    return []


def score_strategy(ast: ASTNode, clauses: list[Clause],
                   ctx: CompileContext) -> float:
    """
    Score confidence in the STRATEGY, not the answer.

    Checks:
    1. Number coverage: does the AST account for all numbers in the problem?
    2. Operation alignment: does the op count match the clause structure?
    3. Structural sanity: no degenerate trees (single lit, div by 0)
    4. Completeness: did we use information from most clauses?

    Score is 0-1, where 1 means "this interpretation used the right
    pieces from the problem in a structurally sound way."
    """
    if ast is None:
        return 0.0

    score = 0.0

    # --- 1. Number coverage (0.4 weight) ---
    # How many of the problem's numbers did this AST use?
    problem_numbers = []
    for c in clauses:
        for t in c.tokens:
            if t.type in ("number", "money", "fraction", "multiplier"):
                problem_numbers.append(t.value)

    if not problem_numbers:
        num_score = 0.5  # no numbers to check
    else:
        ast_nums = _ast_numbers(ast)
        # Match each problem number to an AST number (allow duplicates)
        remaining = list(ast_nums)
        matched = 0
        for pn in problem_numbers:
            for i, an in enumerate(remaining):
                if abs(pn - an) < 0.01:
                    remaining.pop(i)
                    matched += 1
                    break
        num_score = matched / len(problem_numbers)

    # --- 2. Operation alignment (0.3 weight) → [0,1] ---
    # The AST should have roughly one op per operation clause
    n_op_clauses = sum(1 for c in clauses if c.role == "operation")
    ast_op_list = _ast_ops(ast)
    n_ast_ops = len(ast_op_list)

    if n_ast_ops == 0:
        op_score = 0.1
    else:
        # Expected ops: at least n_op_clauses, possibly more for multi-step
        expected = max(n_op_clauses, 1)
        ratio = n_ast_ops / expected
        if 0.5 <= ratio <= 2.0:
            op_score = 0.8
        elif 0.3 <= ratio <= 3.0:
            op_score = 0.5
        else:
            op_score = 0.2

    # --- 3. Structural sanity (0.2 weight) → [0,1] ---
    result = execute(ast)
    if result is None:
        sanity = 0.0
    else:
        sanity = 1.0
        try:
            if abs(result) == float('inf') or result != result:
                sanity = 0.0
        except (OverflowError, ValueError):
            sanity = 0.0
        if sanity > 0:
            # Negative penalty
            if result < 0:
                sanity *= 0.6
            # Magnitude: answer should be reasonable relative to inputs
            if problem_numbers:
                max_input = max(abs(n) for n in problem_numbers)
                if max_input > 0:
                    ratio = abs(result) / max_input
                    if ratio > 1000:
                        sanity *= 0.2  # wildly out of range
                    elif ratio > 100:
                        sanity *= 0.5  # suspicious
                    # Very tiny non-integer also suspicious
                    if result != 0 and abs(result) < 0.001:
                        sanity *= 0.5

    # --- 4. Completeness (0.1 weight) ---
    # Did we use numbers from multiple clauses, not just one?
    clauses_with_nums = sum(1 for c in clauses
                             if c.role != "question"
                             and any(t.type in ("number", "money", "fraction", "multiplier")
                                     for t in c.tokens))
    if clauses_with_nums <= 1:
        completeness = 0.5
    else:
        # How many clauses contributed numbers to the AST?
        clause_nums_used = 0
        remaining_ast = list(_ast_numbers(ast))
        for c in clauses:
            if c.role == "question":
                continue
            for t in c.tokens:
                if t.type in ("number", "money", "fraction", "multiplier"):
                    for j, an in enumerate(remaining_ast):
                        if abs(t.value - an) < 0.01:
                            remaining_ast.pop(j)
                            clause_nums_used += 1
                            break
                    break  # one match per clause is enough
        completeness = clause_nums_used / clauses_with_nums if clauses_with_nums else 0.5

    # All components are [0,1], weighted sum stays [0,1]
    score = (num_score * 0.4 + op_score * 0.3 +
             sanity * 0.2 + completeness * 0.1)
    return max(0.0, min(1.0, score))


def score_confidence(ast: ASTNode, clauses: list[Clause],
                     ctx: CompileContext) -> float:
    """Score confidence — delegates to strategy scorer."""
    return score_strategy(ast, clauses, ctx)


# =====================================================================
# Public API
# =====================================================================

def compile(text: str) -> tuple[Optional[ASTNode], CompileContext]:
    """
    Compile a word problem into an AST.
    Returns (ast, context) where ast is None if compilation fails.
    """
    clauses, ctx = parse(text)
    ast = build_ast(clauses, ctx)
    ctx.ast = ast
    ctx.confidence = score_confidence(ast, clauses, ctx)
    # Evaluate all candidate answers now that execute() is available
    if hasattr(ctx, '_candidate_asts') and ctx._candidate_asts:
        scores = getattr(ctx, '_candidate_scores', [])
        ctx.all_candidates = []
        for idx, (c_ast, name, pri) in enumerate(ctx._candidate_asts):
            try:
                val = execute(c_ast)
            except Exception:
                val = None
            strat = scores[idx][2] if idx < len(scores) else 0.0
            blend = scores[idx][3] if idx < len(scores) else 0.0
            ctx.all_candidates.append((val, name, pri, strat, blend))
        del ctx._candidate_asts
        if hasattr(ctx, '_candidate_scores'):
            del ctx._candidate_scores
    return ast, ctx


def solve(text: str) -> tuple[Optional[float], CompileContext]:
    """
    One-shot: compile + execute.
    Returns (answer, context) where answer is None if unsolvable.
    """
    ast, ctx = compile(text)
    answer = execute(ast)
    return answer, ctx


def ast_to_str(ast: ASTNode, indent: int = 0) -> str:
    """Pretty-print an AST."""
    prefix = "  " * indent
    if isinstance(ast, Lit):
        return f"{prefix}{ast.value:g}" + (f" ({ast.label})" if ast.label else "")
    if isinstance(ast, Neg):
        return f"{prefix}neg\n{ast_to_str(ast.child, indent+1)}"
    if isinstance(ast, Ref):
        if ast.resolved:
            return f"{prefix}&{ast.name}\n{ast_to_str(ast.resolved, indent+1)}"
        return f"{prefix}&{ast.name} (unresolved)"
    if isinstance(ast, BinOp):
        return (f"{prefix}{ast.op}\n"
                f"{ast_to_str(ast.left, indent+1)}\n"
                f"{ast_to_str(ast.right, indent+1)}")
    return f"{prefix}???"


# Backwards compatibility
def interpret(text):
    """Legacy wrapper — returns a duck-typed object matching old API."""
    ast, ctx = compile(text)

    class _Compat:
        def __init__(self):
            self.source = text
            self.confidence = ctx.confidence
            self.unsolved_reason = ctx.unsolved_reason
            self.quantities = []
            self.steps = []
            self.initial_value = None

    return _Compat()
