"""
Language Graph verb resolver.

Bridges the language graph (WordNet synsets) to math operations.
Instead of hardcoded verb lists in operations.py, this module resolves
verbs through the graph:

    "gives" → WordNet lemma "give" → synset "give.v.01" → DENOTES → subtract

The synset-to-operation mapping is the ONLY curated table. Everything else
comes from the graph: inflections, synonyms, entailments.

Architecture:
    1. Lemmatize the input word (WordNet morphology)
    2. Look up all synsets for the lemma
    3. Check synset → operation mapping (curated seed edges)
    4. If no direct hit, walk HYPERNYM_OF edges upward (graph traversal)
    5. Greedy: multi-word phrases checked first, then single words

This replaces the verb lists in operations.py. The lists become the seed
data for bootstrapping the graph — once synset mappings are in place, the
lists can be deleted.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

# NLTK data path — use project-local cache
_NLTK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "cache", "nltk_data")
if os.path.exists(_NLTK_DIR):
    import nltk
    nltk.data.path.insert(0, _NLTK_DIR)

try:
    from nltk.corpus import wordnet as wn
    _HAS_WORDNET = True
except Exception:
    _HAS_WORDNET = False


# ---------------------------------------------------------------------------
# Synset → math operation mapping (the curated seed edges)
#
# These are DENOTES_PREDICATE edges from Lang_Lexeme to Math_Operation.
# Format: {synset_offset_key: operation_id}
#
# Each entry says: "this meaning cluster maps to this math operation."
# All lemmas in the synset inherit the mapping automatically.
# ---------------------------------------------------------------------------

# We key by synset name (e.g., "give.v.01") for readability.
# At init time, we also index by offset for fast lookup.

_SYNSET_TO_OP: dict[str, str] = {
    # --- ADDITION: transfer TO the subject / increase ---
    "get.v.01": "add",           # come into possession of
    "receive.v.01": "add",       # get something
    "gain.v.01": "add",          # obtain
    "earn.v.01": "add",          # earn (receive as return)
    "collect.v.01": "add",       # accumulate
    "buy.v.01": "add",           # acquire by paying
    "find.v.01": "add",          # discover
    "add.v.01": "add",           # make an addition
    "increase.v.01": "add",      # make greater in size/amount
    "join.v.01": "add",          # become part of

    # --- SUBTRACTION: transfer FROM the subject / decrease ---
    "give.v.01": "subtract",     # transfer possession
    "lose.v.01": "subtract",     # fail to keep
    "spend.v.01": "subtract",    # pay out
    "use.v.01": "subtract",      # consume
    "eat.v.01": "subtract",      # consume food
    "remove.v.01": "subtract",   # take away
    "sell.v.01": "subtract",     # exchange for money (reduces inventory)
    "throw.v.01": "subtract",    # propel away
    "break.v.01": "subtract",    # destroy
    "leave.v.01": "subtract",    # go away from (reduces count)
    "donate.v.01": "subtract",   # give to charity
    "get_rid_of.v.01": "subtract", # dispose of
    "discard.v.01": "subtract",  # throw away

    # Consumption verbs — these use up materials (subtract from inventory)
    "cook.v.01": "subtract",    # prepare food (consumes ingredients)
    "bake.v.01": "subtract",    # cook in oven (consumes ingredients)
    "create_from_raw_material.v.01": "subtract",  # hypernym: consumes raw material

    # --- MULTIPLICATION: scaling / rates ---
    "multiply.v.01": "multiply", # arithmetic
    "double.v.01": "multiply",   # make twice as great

    # --- DIVISION: splitting / sharing ---
    "divide.v.01": "divide",     # separate into parts
    "share.v.01": "divide",      # use jointly
    "split.v.01": "divide",      # separate
    "distribute.v.01": "divide", # give to several
}

# Multi-word phrases that should be resolved before single words.
# These are idioms/phrasal verbs where the parts mean something different
# than the whole. Greedy matching: longest match wins.
_PHRASE_TO_OP: dict[str, str] = {
    "get rid of": "subtract",
    "gets rid of": "subtract",
    "got rid of": "subtract",
    "getting rid of": "subtract",
    "give away": "subtract",
    "gives away": "subtract",
    "gave away": "subtract",
    "throw away": "subtract",
    "throws away": "subtract",
    "threw away": "subtract",
    "gave him": "add",       # transfer TO: "his mom gave him 5 cookies"
    "gave her": "add",
    "gave them": "add",
    "less than": "subtract",
    "more than": "add",
    "left over": "subtract",
}

# Sorted by length (longest first) for greedy matching
_PHRASES_BY_LENGTH = sorted(_PHRASE_TO_OP.keys(), key=lambda x: -len(x))


# ---------------------------------------------------------------------------
# Structural patterns: phrases that emit relational equations, not just
# a single operation. These are "N times as many A as B" → A = N × B.
#
# Each pattern is a regex → template. The template says how the captured
# groups relate to each other via an operation.
#
# Format: (compiled_regex, operation, role_of_groups)
#   - groups are named: 'n' (number), 'subject', 'reference'
#   - operation: how subject relates to reference
#   - The equation produced: subject = reference × n (for 'multiply')
# ---------------------------------------------------------------------------

_STRUCTURAL_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    # "N times as many/much [stuff] as [ref]" → subject = N × ref
    # N can be a fraction (1/2), decimal, integer, or multiplier word
    (re.compile(
        r'(?P<n>\d+(?:\.\d+)?|half|twice|double|triple|thrice)'
        r'\s+times?\s+as\s+(?:many|much)\s+'
        r'(?:\w+\s+)*?as\s+(?:the\s+(?:number|amount)\s+of\s+)?'
        r'(?:\w+\s+)*?(?P<reference>[a-z]{2,})\b',
        re.I),
     'multiply', 0.95),

    # "twice/half/double/triple as many/much [stuff] as [ref]"
    (re.compile(
        r'(?P<n>twice|half|double|triple|thrice)'
        r'\s+as\s+(?:many|much)\s+'
        r'(?:\w+\s+)*?as\s+(?:the\s+(?:number|amount)\s+of\s+)?'
        r'(?:\w+\s+)*?(?P<reference>[a-z]{2,})\b',
        re.I),
     'multiply', 0.95),

    # "N more [stuff] than [ref]" → subject = ref + N
    (re.compile(
        r'(?P<n>\d+(?:\.\d+)?)\s+more\s+(?:\w+\s+)*?than\s+(?:\w+\s+)*?(?P<reference>\w+)',
        re.I),
     'add', 0.90),

    # "N fewer/less [stuff] than [ref]" → subject = ref - N
    (re.compile(
        r'(?P<n>\d+(?:\.\d+)?)\s+(?:fewer|less)\s+(?:\w+\s+)*?than\s+(?:\w+\s+)*?(?P<reference>\w+)',
        re.I),
     'subtract', 0.90),
]

# Word → number mapping for multiplier words in structural patterns
_MULT_WORDS = {
    'half': 0.5, 'twice': 2, 'double': 2,
    'triple': 3, 'thrice': 3, 'quarter': 0.25,
}


@dataclass
class StructuralMatch:
    """Result of matching a structural pattern in a sentence."""
    operation: str       # 'multiply', 'add', 'subtract'
    n: float             # the numeric parameter
    reference: str       # the reference entity/concept
    confidence: float
    pattern: str         # description of the matched pattern


def resolve_structural(sentence: str) -> Optional[StructuralMatch]:
    """
    Check if a sentence matches a structural relational pattern.

    These are patterns like "N times as many X as Y" that produce
    equations, not just operations. Returns the match with the
    highest confidence, or None.
    """
    sent_lower = sentence.lower()
    best: Optional[StructuralMatch] = None

    for pattern, op, conf in _STRUCTURAL_PATTERNS:
        m = pattern.search(sent_lower)
        if m:
            n_raw = m.group('n')
            if n_raw in _MULT_WORDS:
                n = _MULT_WORDS[n_raw]
            else:
                try:
                    n = float(n_raw)
                except ValueError:
                    continue

            ref = m.group('reference')
            match = StructuralMatch(
                operation=op,
                n=n,
                reference=ref,
                confidence=conf,
                pattern=m.group(0),
            )
            if best is None or match.confidence > best.confidence:
                best = match

    return best


@dataclass
class VerbResolution:
    """Result of resolving a verb through the language graph."""
    operation: str          # math operation id: "add", "subtract", etc.
    source: str             # how it was resolved: "synset", "phrase", "learned", "fallback"
    confidence: float       # 0-1 confidence
    synset: str = ""        # WordNet synset name if resolved via graph
    lemma: str = ""         # canonical lemma


# ---------------------------------------------------------------------------
# Init: build reverse index from synset offsets
# ---------------------------------------------------------------------------

_OFFSET_TO_OP: dict[int, str] = {}
_LEMMA_TO_OP: dict[str, tuple[str, str]] = {}  # lemma → (op, synset_name)

def _init_indices():
    """Build lookup indices from synset seed mappings."""
    if not _HAS_WORDNET:
        return

    for synset_name, op_id in _SYNSET_TO_OP.items():
        try:
            syn = wn.synset(synset_name)
        except Exception:
            continue

        _OFFSET_TO_OP[syn.offset()] = op_id

        # Index all lemmas in this synset
        for lemma in syn.lemmas():
            word = lemma.name().lower().replace("_", " ")
            # Don't overwrite if already mapped (first mapping wins)
            if word not in _LEMMA_TO_OP:
                _LEMMA_TO_OP[word] = (op_id, synset_name)

        # Also walk one level of ENTAILS (verb entailment)
        for entailed_syn in syn.entailments():
            for lemma in entailed_syn.lemmas():
                word = lemma.name().lower().replace("_", " ")
                if word not in _LEMMA_TO_OP:
                    _LEMMA_TO_OP[word] = (op_id, f"{synset_name}→entails→{entailed_syn.name()}")


_init_indices()


# ---------------------------------------------------------------------------
# Learned verb mappings (from GSM8K training)
# ---------------------------------------------------------------------------

_LEARNED_OPS: dict[str, tuple[str, float]] = {}  # word → (op_symbol, lift)

def _load_learned():
    """Load learned verb→op mappings from GSM8K training data."""
    import json
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "math", "learned_verb_ops.json")
    path = os.path.normpath(path)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        op_map = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
        for word, info in data.items():
            sym = info.get("op", "")
            lift = info.get("lift", 0)
            if sym in op_map and lift >= 0.3:
                _LEARNED_OPS[word.lower()] = (op_map[sym], lift)

_load_learned()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_verb(text: str) -> Optional[VerbResolution]:
    """
    Resolve a verb (word or phrase) to a math operation via the language graph.

    Resolution order (greedy, most specific first):
    1. Multi-word phrase lookup (idioms/phrasal verbs)
    2. WordNet synset mapping (graph edges)
    3. WordNet morphology → synset (lemmatized lookup)
    4. Learned verb mappings (GSM8K training data)

    Returns None if no resolution found.
    """
    text_lower = text.lower().strip()

    # 1. Phrase lookup (greedy: longest first)
    for phrase in _PHRASES_BY_LENGTH:
        if phrase in text_lower:
            return VerbResolution(
                operation=_PHRASE_TO_OP[phrase],
                source="phrase",
                confidence=0.95,
                lemma=phrase,
            )

    # 2. Direct lemma lookup from synset index
    if text_lower in _LEMMA_TO_OP:
        op, syn_name = _LEMMA_TO_OP[text_lower]
        return VerbResolution(
            operation=op,
            source="synset",
            confidence=0.9,
            synset=syn_name,
            lemma=text_lower,
        )

    # 3. WordNet morphological lookup (handles inflections)
    if _HAS_WORDNET:
        # Try all verb synsets for this word
        for syn in wn.synsets(text_lower, pos=wn.VERB):
            if syn.offset() in _OFFSET_TO_OP:
                return VerbResolution(
                    operation=_OFFSET_TO_OP[syn.offset()],
                    source="synset",
                    confidence=0.85,
                    synset=syn.name(),
                    lemma=text_lower,
                )

            # Walk hypernyms (one level) for broader match
            for hyper in syn.hypernyms():
                if hyper.offset() in _OFFSET_TO_OP:
                    return VerbResolution(
                        operation=_OFFSET_TO_OP[hyper.offset()],
                        source="synset_hypernym",
                        confidence=0.7,
                        synset=f"{syn.name()}→hypernym→{hyper.name()}",
                        lemma=text_lower,
                    )

    # 4. Learned mappings from GSM8K
    if text_lower in _LEARNED_OPS:
        op, lift = _LEARNED_OPS[text_lower]
        return VerbResolution(
            operation=op,
            source="learned",
            confidence=lift,
            lemma=text_lower,
        )

    return None


def resolve_sentence(sentence: str) -> Optional[VerbResolution]:
    """
    Resolve the dominant math operation in a sentence.

    Greedy: checks multi-word phrases first, then individual words.
    Returns the highest-confidence resolution found.
    """
    sent_lower = sentence.lower()

    # Phase 1: multi-word phrases (greedy, longest first)
    for phrase in _PHRASES_BY_LENGTH:
        if phrase in sent_lower:
            return VerbResolution(
                operation=_PHRASE_TO_OP[phrase],
                source="phrase",
                confidence=0.95,
                lemma=phrase,
            )

    # Phase 2: individual words, take highest confidence
    best: Optional[VerbResolution] = None
    words = re.findall(r'\b[a-z]+\b', sent_lower)
    for word in words:
        r = resolve_verb(word)
        if r and (best is None or r.confidence > best.confidence):
            best = r

    return best
