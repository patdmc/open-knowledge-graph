"""
Algebraic word problem solver.

Core insight: word problems are systems of equations.
Each sentence gives one equation. The question names the target.
Solve by substitution.

Uses the language graph (resolve.py) for verb→operation mapping
and interpret.py's lexer for number extraction. This solver adds
the algebraic layer: named variables, equation building, and
substitution.

Patterns learned from user walkthroughs:
  1. Chain substitution    — J=T+2, T=A+5, L=A+2; target J-L → 5
  2. Rate × quantity       — rate=12/60; earn=rate*50 → 10
  3. Define then combine   — A=48, M=A/2; total=A+M → 72
  4. Goal minus accum      — wallet=100, B=wallet/2; need=wallet-B-P-G
  5. Dimensional chain     — 3*2*2*52 (units cancel like fractions)
  6. Parallel groups + sum — 2*16 + 2*8 (count×rate per group, sum)
  7. Complement then scale — above_B = 1 - 0.40; answer = 60 * above_B
  8. Partition + complement— hit_1=100*2/5, hit_2=75*1/3; miss=175-sum
  9. Solve for missing     — known+x=total; x=total-known

Every word problem gives N variables and N formulas.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# =====================================================================
# Symbolic expression tree
# =====================================================================

@dataclass
class Sym:
    """A symbolic variable."""
    name: str
    def __repr__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, Sym) and self.name == other.name
    def __hash__(self):
        return hash(self.name)

@dataclass
class Num:
    """A numeric literal."""
    value: float
    def __repr__(self):
        v = self.value
        return str(int(v)) if v == int(v) else f"{v:g}"
    def __eq__(self, other):
        return isinstance(other, Num) and self.value == other.value
    def __hash__(self):
        return hash(self.value)

@dataclass
class BinExpr:
    """A binary expression: left op right."""
    op: str  # '+', '-', '*', '/'
    left: object
    right: object
    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

Expr = Sym | Num | BinExpr

@dataclass
class Equation:
    """One equation: lhs = rhs."""
    lhs: Expr
    rhs: Expr
    source: str = ""
    pattern: str = ""


# =====================================================================
# Language graph integration
# =====================================================================

def _resolve_operation(sentence: str) -> tuple[str, float, str]:
    """
    Resolve the math operation for a sentence via the language graph.

    Returns (operation, confidence, source) where operation is
    'add', 'subtract', 'multiply', 'divide', or ''.
    """
    try:
        from packages.core.language.resolve import resolve_sentence
        result = resolve_sentence(sentence)
        if result:
            return result.operation, result.confidence, f"{result.source}:{result.lemma}"
    except ImportError:
        pass

    # Fallback: basic verb detection without language graph
    sl = sentence.lower()

    # Phrase patterns first
    if any(p in sl for p in ['gave away', 'get rid of', 'threw away']):
        return 'subtract', 0.8, 'fallback:phrase'
    if any(p in sl for p in ['more than', 'in addition']):
        return 'add', 0.7, 'fallback:phrase'

    # Single verb patterns
    sub_verbs = ['gave', 'lost', 'spent', 'ate', 'sold', 'used', 'threw',
                 'donated', 'dropped', 'broke', 'removed', 'discarded',
                 'paid', 'returned', 'cooked', 'baked']
    add_verbs = ['got', 'received', 'found', 'earned', 'added', 'bought',
                 'picked', 'collected', 'gained', 'won', 'saved', 'grew']
    div_verbs = ['divided', 'shared', 'split', 'distributed']
    mul_verbs = ['doubled', 'tripled', 'multiplied']

    words = set(re.findall(r'\b[a-z]+\b', sl))
    for v in sub_verbs:
        if v in words:
            return 'subtract', 0.6, f'fallback:{v}'
    for v in add_verbs:
        if v in words:
            return 'add', 0.6, f'fallback:{v}'
    for v in div_verbs:
        if v in words:
            return 'divide', 0.6, f'fallback:{v}'
    for v in mul_verbs:
        if v in words:
            return 'multiply', 0.6, f'fallback:{v}'

    return '', 0.0, ''


def _lex_sentence(sentence: str) -> list:
    """
    Lex a sentence using interpret.py's lexer.

    Returns list of Token objects with value, unit, per_unit, type, etc.
    Falls back to basic extraction if interpret.py unavailable.
    """
    try:
        from packages.core.interpret import lex_sentence
        return lex_sentence(sentence)
    except ImportError:
        pass

    # Fallback: basic number extraction
    @dataclass
    class SimpleToken:
        type: str
        value: float
        text: str
        position: int
        unit: str = ""
        per_unit: str = ""
        ref_type: str = ""

    tokens = []
    # Dollar amounts
    for m in re.finditer(r'\$(\d[\d,]*\.?\d*)', sentence):
        v = float(m.group(1).replace(',', ''))
        tokens.append(SimpleToken('money', v, m.group(), m.start(), unit='dollars'))
    # Fractions
    for m in re.finditer(r'\b(\d+)\s*/\s*(\d+)\b', sentence):
        n, d = float(m.group(1)), float(m.group(2))
        if d != 0 and n < 100 and d < 100:
            tokens.append(SimpleToken('fraction', n/d, m.group(), m.start()))
    # Plain numbers
    captured = {t.position for t in tokens}
    for m in re.finditer(r'\b(\d[\d,]*\.?\d*)\b', sentence):
        if not any(abs(m.start() - p) < 3 for p in captured):
            v = float(m.group(1).replace(',', ''))
            tokens.append(SimpleToken('number', v, m.group(), m.start()))
    # Multiplier words
    mults = {'half': 0.5, 'twice': 2, 'double': 2, 'triple': 3, 'quarter': 0.25}
    for word, val in mults.items():
        for m in re.finditer(rf'\b{word}\b', sentence.lower()):
            tokens.append(SimpleToken('multiplier', val, word, m.start()))

    tokens.sort(key=lambda t: t.position)
    return tokens


# =====================================================================
# Entity detection
# =====================================================================

_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'had', 'have',
    'does', 'do', 'did', 'will', 'can', 'could', 'may', 'not', 'if',
    'she', 'he', 'it', 'they', 'her', 'his', 'how', 'what', 'many',
    'much', 'each', 'every', 'per', 'more', 'than', 'less', 'fewer',
    'all', 'total', 'altogether', 'together', 'in', 'on', 'at', 'to',
    'of', 'for', 'with', 'from', 'by', 'and', 'or', 'but', 'so',
    'then', 'there', 'here', 'this', 'that', 'some', 'now', 'still',
    'left', 'remain', 'after', 'before', 'between', 'among', 'old',
    'new', 'also', 'out', 'up', 'down', 'just', 'only', 'about',
    'into', 'over', 'since', 'until', 'which', 'while', 'during',
    'because', 'next', 'other', 'another', 'same', 'different',
    'own', 'few', 'several', 'both', 'either', 'neither', 'such',
    'no', 'any', 'enough', 'too', 'very', 'first', 'second', 'third',
    'last', 'once', 'twice', 'half', 'however', 'although', 'though',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'january', 'february', 'march', 'april',
    'may', 'june', 'july', 'august', 'september', 'october',
    'november', 'december', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'three', 'two', 'one',
}

_PRONOUNS = {'she', 'he', 'her', 'his', 'they', 'their', 'it', 'its'}


def _find_entity(sentence: str) -> str:
    """Find the first proper noun entity in a sentence."""
    for m in re.finditer(r'\b([A-Z][a-z]{2,})\b', sentence):
        name = m.group(1).lower()
        if name not in _STOP_WORDS:
            return name
    return ""


# Object nouns that should become concept variables (not entity state)
_CONCEPT_NOUNS = {
    # Food/grocery
    'apple', 'apples', 'orange', 'oranges', 'banana', 'bananas',
    'cookie', 'cookies', 'cake', 'cakes', 'pie', 'pies',
    'pizza', 'pizzas', 'sandwich', 'sandwiches', 'bread',
    'egg', 'eggs', 'milk', 'cheese', 'cheddar', 'meat',
    'chicken', 'fish', 'rice', 'flour', 'sugar', 'butter',
    'candy', 'candies', 'chocolate', 'cupcake', 'cupcakes',
    'muffin', 'muffins', 'donut', 'donuts', 'bagel', 'bagels',
    # Items
    'book', 'books', 'pen', 'pens', 'pencil', 'pencils',
    'shirt', 'shirts', 'shoe', 'shoes', 'toy', 'toys',
    'ball', 'balls', 'doll', 'dolls', 'card', 'cards',
    'flower', 'flowers', 'sticker', 'stickers', 'marble', 'marbles',
    'bead', 'beads', 'stamp', 'stamps', 'coin', 'coins',
    'ticket', 'tickets', 'gift', 'gifts', 'bottle', 'bottles',
    'box', 'boxes', 'bag', 'bags', 'basket', 'baskets',
    'binder', 'binders', 'page', 'pages', 'letter', 'letters',
    # Animals
    'cat', 'cats', 'dog', 'dogs', 'bird', 'birds',
    'horse', 'horses', 'cow', 'cows', 'sheep', 'pig', 'pigs',
    'goat', 'goats', 'chicken', 'chickens', 'duck', 'ducks',
    'fish', 'fishes', 'rabbit', 'rabbits', 'hamster', 'hamsters',
    # Structures
    'house', 'houses', 'room', 'rooms', 'car', 'cars',
    'truck', 'trucks', 'bike', 'bikes', 'boat', 'boats',
    'tree', 'trees', 'plant', 'plants', 'garden', 'gardens',
    # Containers/groups
    'dozen', 'pair', 'pack', 'set', 'pile', 'bunch', 'batch',
    'row', 'rows', 'group', 'groups', 'stack', 'stacks',
    # Money concepts
    'rent', 'salary', 'wage', 'tip', 'tips', 'tax', 'taxes',
    'discount', 'refund', 'deposit', 'payment', 'bill', 'bills',
    'groceries', 'supplies', 'materials',
    # Body of work
    'mile', 'miles', 'lap', 'laps', 'trip', 'trips',
    'game', 'games', 'match', 'matches', 'race', 'races',
    'class', 'classes', 'lesson', 'lessons', 'exam', 'exams',
    # Food prep
    'slice', 'slices', 'piece', 'pieces', 'serving', 'servings',
    'cup', 'cups', 'gallon', 'gallons', 'liter', 'liters',
    'pound', 'pounds', 'ounce', 'ounces', 'kilogram', 'kilograms',
    # Farm
    'hay', 'oats', 'carrots', 'bale', 'bales',
}


_UNIT_NOUNS = {
    'pound', 'pounds', 'ounce', 'ounces', 'kilogram', 'kilograms',
    'gram', 'grams', 'cup', 'cups', 'gallon', 'gallons',
    'liter', 'liters', 'dozen', 'pair', 'mile', 'miles',
    'inch', 'inches', 'foot', 'feet', 'yard', 'yards',
    'meter', 'meters', 'kilometer', 'kilometers',
}


def _find_concept(sentence: str, entity: str) -> str:
    """Find a concept noun in the sentence that should be a separate variable.

    Returns the concept noun (e.g., 'cheddar', 'cookies') or '' if
    the sentence is about the entity's state, not a separate concept.

    Skips unit nouns when they're used as units ("2 pounds of X" → X, not pounds).
    """
    sl = sentence.lower()
    words = re.findall(r'\b[a-z]+\b', sl)

    # "N unit of NOUN" pattern: skip the unit, prefer the NOUN after "of"
    of_m = re.search(r'\b(?:pounds?|ounces?|cups?|gallons?|dozen|'
                     r'pieces?|slices?|servings?|bags?|boxes?|packs?)\s+of\s+(\w+)', sl)
    if of_m:
        noun = of_m.group(1)
        if noun in _CONCEPT_NOUNS and noun != entity:
            return noun

    for w in words:
        if w in _CONCEPT_NOUNS and w != entity and w not in _UNIT_NOUNS:
            return w
    return ""


# =====================================================================
# Comparison pattern detection (regex, pre-language-graph)
# These are structural patterns that the verb resolver can't handle
# because they're about the RELATIONSHIP between two entities,
# not just a verb's meaning.
# =====================================================================

_CMP_MORE = re.compile(
    r'\b(\w+)\s+(?:is|was|has|had|gets?|got|earns?|earned|weighs?|'
    r'scored|can|runs?|ran|drives?|drove)\s+'
    r'(?:about\s+)?(\d+(?:\.\d+)?)\s+'
    r'(?:\w+\s+)*?'
    r'(?:more|older|taller|heavier|longer|wider|bigger|greater|'
    r'farther|higher|faster|richer|larger|deeper|thicker)\s+'
    r'(?:\w+\s+)*?'
    r'(?:than)\s+(\w+)', re.I)

_CMP_LESS = re.compile(
    r'\b(\w+)\s+(?:is|was|has|had|gets?|got)\s+'
    r'(?:about\s+)?(\d+(?:\.\d+)?)\s+'
    r'(?:\w+\s+)*?'
    r'(?:less|younger|shorter|lighter|smaller|fewer|slower|thinner)\s+'
    r'(?:\w+\s+)*?'
    r'(?:than)\s+(\w+)', re.I)

_CMP_TIMES = re.compile(
    r'\b(\w+)\s+\w+\s+'
    r'(?:about\s+)?(\d+(?:\.\d+)?)\s+times\s+'
    r'(?:as\s+\w+\s+(?:\w+\s+)?as|more\s+(?:\w+\s+)?than|that\s+of)\s+'
    r'(\w+)', re.I)

_CMP_MULT = re.compile(
    r'\b(twice|double|triple|thrice)\s+'
    r'(?:as\s+(?:many|much)\s+(?:\w+\s+)?as\s+|what\s+|that\s+of\s+|'
    r'the\s+\w+\s+(?:of\s+)?)'
    r'(\w+)', re.I)

_CMP_HALF = re.compile(
    r'\b(half)\s+'
    r'(?:as\s+(?:many|much)\s+(?:\w+\s+)?as\s+|of\s+(?:what\s+)?|'
    r'that\s+of\s+)'
    r'(\w+)', re.I)

# "N less/more than half/twice as many ... as Y"
_N_THAN_MULT = re.compile(
    r'(\d+(?:\.\d+)?)\s+(?:less|fewer)\s+than\s+'
    r'(half|twice|double|triple|thrice)\s+as\s+(?:many|much)\s+'
    r'(?:\w+\s+)*?as\s+(?:\w+\s+)*?(\w+)', re.I)

_N_MORE_THAN_MULT = re.compile(
    r'(\d+(?:\.\d+)?)\s+more\s+than\s+'
    r'(half|twice|double|triple|thrice)\s+as\s+(?:many|much)\s+'
    r'(?:\w+\s+)*?as\s+(?:\w+\s+)*?(\w+)', re.I)

# "half/twice as many" without explicit reference entity
_HALF_AS_MANY = re.compile(
    r'\b(half|twice|double|triple|thrice)\s+as\s+(?:many|much)\b', re.I)


def _is_entity(word: str) -> bool:
    return (len(word) >= 2 and
            word.lower() not in _STOP_WORDS and
            not word.isdigit())


# =====================================================================
# Sentence → Equation (using language graph + lexer)
# =====================================================================

def parse_clause(sentence: str, known_entities: set[str],
                 prior_equations: list[Equation],
                 current_entity: str = "",
                 frame: dict = None) -> tuple[list[Equation], str]:
    """
    Parse one clause into equations using the language graph.

    1. Lex the sentence (interpret.py) → tokens with values, units, rates
    2. Find entity (proper noun or pronoun → prior entity)
    3. Detect structural patterns (comparisons) via regex
    4. For non-comparison sentences, use language graph (resolve.py)
       to determine the operation, then build equation from tokens

    Frame context (from reading the whole problem first) helps:
    - Disambiguate number roles (is 20 a rate denominator or a count?)
    - Detect implicit rates ("8 pages in 20 minutes" → rate)
    - Name concept variables instead of piling on one entity
    """
    if frame is None:
        frame = {}
    equations = []
    s = sentence.strip()
    sl = s.lower()

    # --- Entity detection ---
    entity = _find_entity(s)
    if entity:
        current_entity = entity
    elif any(p in sl.split() for p in _PRONOUNS) and current_entity:
        entity = current_entity
    else:
        entity = current_entity

    # --- Structural patterns (comparisons between entities) ---
    # These are about RELATIONSHIPS, not verbs — language graph can't help here

    m = _CMP_MORE.search(s)
    if m and _is_entity(m.group(1)) and _is_entity(m.group(3)):
        x, n, y = m.group(1).lower(), float(m.group(2)), m.group(3).lower()
        equations.append(Equation(Sym(x), BinExpr('+', Sym(y), Num(n)),
                                  s, 'compare_more'))
        known_entities.update({x, y})
        return equations, x

    m = _CMP_LESS.search(s)
    if m and _is_entity(m.group(1)) and _is_entity(m.group(3)):
        x, n, y = m.group(1).lower(), float(m.group(2)), m.group(3).lower()
        equations.append(Equation(Sym(x), BinExpr('-', Sym(y), Num(n)),
                                  s, 'compare_less'))
        known_entities.update({x, y})
        return equations, x

    m = _CMP_TIMES.search(s)
    if m and _is_entity(m.group(1)) and _is_entity(m.group(3)):
        x, n, y = m.group(1).lower(), float(m.group(2)), m.group(3).lower()
        equations.append(Equation(Sym(x), BinExpr('*', Sym(y), Num(n)),
                                  s, 'compare_times'))
        known_entities.update({x, y})
        return equations, x

    # "twice/double/triple as many as Y" (with subject from context)
    m = _CMP_MULT.search(s)
    if m:
        mult_word = m.group(1).lower()
        y = m.group(2).lower()
        mult = {'twice': 2, 'double': 2, 'triple': 3, 'thrice': 3}[mult_word]
        subject = entity or current_entity
        if subject and _is_entity(y) and subject != y:
            equations.append(Equation(Sym(subject), BinExpr('*', Sym(y), Num(mult)),
                                      s, 'compare_mult'))
            known_entities.update({subject, y})
            return equations, subject

    m = _CMP_HALF.search(s)
    if m:
        y = m.group(2).lower()
        subject = entity or current_entity
        if subject and _is_entity(y) and subject != y:
            equations.append(Equation(Sym(subject), BinExpr('/', Sym(y), Num(2)),
                                      s, 'compare_half'))
            known_entities.update({subject, y})
            return equations, subject

    # "N less than half/twice as many ... as Y" compound pattern
    m = _N_THAN_MULT.search(s)
    if m:
        n = float(m.group(1))
        mult_word = m.group(2).lower()
        y = m.group(3).lower()
        mult = {'half': 0.5, 'twice': 2, 'double': 2,
                'triple': 3, 'thrice': 3}.get(mult_word, 1)
        subject = entity or current_entity
        if subject and _is_entity(y):
            # subject = y * mult - n
            rhs = BinExpr('-', BinExpr('*', Sym(y), Num(mult)), Num(n))
            equations.append(Equation(Sym(subject), rhs, s, 'n_less_than_mult'))
            known_entities.update({subject, y})
            return equations, subject

    m = _N_MORE_THAN_MULT.search(s)
    if m:
        n = float(m.group(1))
        mult_word = m.group(2).lower()
        y = m.group(3).lower()
        mult = {'half': 0.5, 'twice': 2, 'double': 2,
                'triple': 3, 'thrice': 3}.get(mult_word, 1)
        subject = entity or current_entity
        if subject and _is_entity(y):
            # subject = y * mult + n
            rhs = BinExpr('+', BinExpr('*', Sym(y), Num(mult)), Num(n))
            equations.append(Equation(Sym(subject), rhs, s, 'n_more_than_mult'))
            known_entities.update({subject, y})
            return equations, subject

    # "half/twice as many" without explicit reference
    m = _HALF_AS_MANY.search(s)
    if m and current_entity:
        mult_word = m.group(1).lower()
        mult = {'half': 0.5, 'twice': 2, 'double': 2,
                'triple': 3, 'thrice': 3}.get(mult_word, 1)
        # Create derived variable with time/context qualifier
        qualifier = ""
        for tw in ['january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november',
                    'december', 'morning', 'afternoon', 'evening', 'night',
                    'week', 'month', 'year', 'day']:
            if tw in sl:
                qualifier = tw
                break
        var_name = f"{current_entity}_{qualifier}" if qualifier else f"{current_entity}_2"
        if mult < 1:
            equations.append(Equation(
                Sym(var_name), BinExpr('/', Sym(current_entity), Num(int(1/mult))),
                s, 'half_as_many'))
        else:
            equations.append(Equation(
                Sym(var_name), BinExpr('*', Sym(current_entity), Num(mult)),
                s, 'mult_as_many'))
        known_entities.add(var_name)
        return equations, current_entity

    # --- Non-comparison: use language graph for operation ---
    tokens = _lex_sentence(s)
    numeric_tokens = [t for t in tokens
                      if t.type in ('number', 'money', 'fraction')]
    multiplier_tokens = [t for t in tokens if t.type == 'multiplier']
    reference_tokens = [t for t in tokens if t.type == 'reference']

    if not numeric_tokens and not multiplier_tokens and not reference_tokens:
        return equations, current_entity

    # Get the operation from language graph
    operation, op_confidence, op_source = _resolve_operation(s)

    # Map operation to arithmetic symbol
    op_to_sym = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
    op_sym = op_to_sym.get(operation, '')

    if not entity:
        entity = current_entity or 'result'

    known_entities.add(entity)

    # --- Concept variable detection ---
    # If the sentence mentions a concrete noun, use it as the variable
    # instead of piling everything onto the entity.
    # "He bought cheddar for $10" → cheddar = 10, not mark = 10
    concept = _find_concept(s, entity)
    var_name = concept if concept else entity

    # --- Build equation based on what we found ---

    # Handle references: "twice that many", "the rest", etc.
    if reference_tokens:
        ref = reference_tokens[0]
        if ref.ref_type == 'multiply_previous':
            # "twice/N times that many" → entity = entity_prev * N
            ref_var = f"{entity}_2"
            equations.append(Equation(
                Sym(ref_var), BinExpr('*', Sym(entity), Num(ref.value)),
                s, 'ref_multiply'))
            known_entities.add(ref_var)
            return equations, entity
        elif ref.ref_type in ('remainder', 'previous'):
            # "the rest" / "that number" — this gets resolved during substitution
            pass

    # "N unit of CONCEPT for $M" pattern: price assignment
    # The money is the value, the quantity is context
    money_tokens = [t for t in numeric_tokens if t.type == 'money']
    qty_tokens = [t for t in numeric_tokens if t.type != 'money']
    if money_tokens and qty_tokens and concept and 'for' in sl:
        # "bought 2 pounds of cheddar for $10" → cheddar = 10
        equations.append(Equation(Sym(concept), Num(money_tokens[0].value),
                                  s, 'price_assign'))
        known_entities.add(concept)
        return equations, entity

    # Rate tokens: tokens with per_unit
    rate_tokens = [t for t in numeric_tokens if t.per_unit]
    count_tokens = [t for t in numeric_tokens if not t.per_unit]

    # Frame-based implicit rate detection:
    # "Joy can read 8 pages in 20 minutes" → rate = 8/20
    if not rate_tokens and len(numeric_tokens) >= 2 and frame.get('has_rate'):
        denom = frame.get('rate_denominator')
        if denom:
            denom_val, denom_unit = denom
            # Find the number matching the rate denominator
            for i, t in enumerate(numeric_tokens):
                if abs(t.value - denom_val) < 0.01:
                    # This number is the denominator, the other is numerator
                    others = [nt for j, nt in enumerate(numeric_tokens) if j != i]
                    if others:
                        numerator = others[0]
                        rate_var = f"{var_name}_rate"
                        equations.append(Equation(
                            Sym(rate_var),
                            BinExpr('/', Num(numerator.value), Num(t.value)),
                            s, 'implicit_rate'))
                        known_entities.add(rate_var)
                        # If there are more numbers, they multiply with the rate
                        for extra in others[1:]:
                            equations.append(Equation(
                                Sym(var_name),
                                BinExpr('*', Sym(rate_var), Num(extra.value)),
                                s, 'rate_apply'))
                            known_entities.add(var_name)
                        return equations, entity

    if rate_tokens and count_tokens:
        # Rate × count pattern
        rate = rate_tokens[0]
        count = count_tokens[0]
        rhs = BinExpr('*', Num(count.value), Num(rate.value))
        # Additional counts multiply in
        for t in count_tokens[1:]:
            rhs = BinExpr('*', rhs, Num(t.value))
        equations.append(Equation(Sym(var_name), rhs, s, 'rate_count'))
        known_entities.add(var_name)
        return equations, entity

    # Multiple numbers with an operation
    if len(numeric_tokens) >= 2 and op_sym:
        has_prior = any(isinstance(eq.lhs, Sym) and eq.lhs.name == var_name
                        for eq in prior_equations)

        if has_prior and op_sym in ('-', '+'):
            # State change on existing variable
            for t in numeric_tokens:
                equations.append(Equation(
                    Sym(var_name),
                    BinExpr(op_sym, Sym(var_name), Num(t.value)),
                    s, f'state_{operation}'))
            return equations, entity
        else:
            # Two numbers combined
            rhs = Num(numeric_tokens[0].value)
            for t in numeric_tokens[1:]:
                rhs = BinExpr(op_sym, rhs, Num(t.value))
            equations.append(Equation(Sym(var_name), rhs, s, f'combine_{operation}'))
            known_entities.add(var_name)
            return equations, entity

    # Single number: assignment or state change
    if len(numeric_tokens) == 1:
        val = numeric_tokens[0].value
        has_prior_entity = any(isinstance(eq.lhs, Sym) and eq.lhs.name == entity
                               for eq in prior_equations)
        has_prior_concept = any(isinstance(eq.lhs, Sym) and eq.lhs.name == var_name
                                for eq in prior_equations)
        has_prior = has_prior_concept if concept else has_prior_entity

        if has_prior and op_sym in ('-', '+'):
            equations.append(Equation(
                Sym(var_name), BinExpr(op_sym, Sym(var_name), Num(val)),
                s, f'state_{operation}'))
        elif has_prior and op_sym == '*':
            equations.append(Equation(
                Sym(var_name), BinExpr('*', Sym(var_name), Num(val)),
                s, f'state_{operation}'))
        elif has_prior and op_sym == '/':
            equations.append(Equation(
                Sym(var_name), BinExpr('/', Sym(var_name), Num(val)),
                s, f'state_{operation}'))
        else:
            # Base assignment — use concept name if available
            equations.append(Equation(Sym(var_name), Num(val), s, 'assign'))
            known_entities.add(var_name)
        return equations, entity

    # Multiplier only (half, twice, etc.)
    if multiplier_tokens and not numeric_tokens:
        # "half/twice the price/amount of X" → concept = X * mult
        # A single clause can contain MULTIPLE such references:
        # "cream cheese for half the price of cheddar and cold cuts
        #  that cost twice the price of cheddar"
        _MULT_REF_RE = re.compile(
            r'(\w+(?:\s+\w+)?)\s+'
            r'(?:for|that\s+costs?|costing|at|worth)\s+'
            r'(half|twice|double|triple|thrice)\s+'
            r'(?:the\s+)?(?:price|cost|amount|number|value|size|weight|length)'
            r'(?:\s+of)?(?:\s+(?:the|a|an))?\s+(\w+)', re.I)
        found_mult_ref = False
        for m in _MULT_REF_RE.finditer(sl):
            subj_raw = m.group(1).strip().replace(' ', '_')
            mult_word = m.group(2).lower()
            ref_name = m.group(3).lower()
            mult = {'half': 0.5, 'twice': 2, 'double': 2,
                    'triple': 3, 'thrice': 3}.get(mult_word, 1)
            if ref_name in known_entities or ref_name in _CONCEPT_NOUNS:
                lhs_name = subj_raw if subj_raw != ref_name else f"{ref_name}_2"
                equations.append(Equation(
                    Sym(lhs_name), BinExpr('*', Sym(ref_name), Num(mult)),
                    s, 'mult_ref'))
                known_entities.add(lhs_name)
                found_mult_ref = True

        if found_mult_ref:
            return equations, entity

        # Simpler form: "half/twice the price of X" without explicit subject
        ref_m = re.search(
            r'(half|twice|double|triple|thrice)\s+'
            r'(?:the\s+)?(?:price|cost|amount|number|value|size|weight|length)'
            r'(?:\s+of)?(?:\s+(?:the|a|an))?\s+(\w+)', sl)
        if ref_m:
            mult_word = ref_m.group(1).lower()
            ref_name = ref_m.group(2).lower()
            mult = {'half': 0.5, 'twice': 2, 'double': 2,
                    'triple': 3, 'thrice': 3}.get(mult_word, 1)
            if ref_name in known_entities or ref_name in _CONCEPT_NOUNS:
                lhs_name = concept if concept and concept != ref_name else var_name
                if lhs_name == ref_name:
                    lhs_name = f"{lhs_name}_2"
                equations.append(Equation(
                    Sym(lhs_name), BinExpr('*', Sym(ref_name), Num(mult)),
                    s, 'mult_ref'))
                known_entities.add(lhs_name)
                return equations, entity

        # Fallback: state change on existing variable
        mult = multiplier_tokens[0].value
        has_prior = any(isinstance(eq.lhs, Sym) and eq.lhs.name == var_name
                        for eq in prior_equations)
        if has_prior:
            equations.append(Equation(
                Sym(var_name), BinExpr('*', Sym(var_name), Num(mult)),
                s, 'state_multiply'))
            return equations, entity

    return equations, current_entity


# =====================================================================
# Question → Target expression
# =====================================================================

_Q_DIFF = re.compile(
    r'how\s+(?:much\s+)?(?:older|younger|taller|shorter|heavier|lighter|'
    r'more|fewer|less|bigger|smaller|farther|faster|richer|longer|wider)\s+'
    r'(?:is|are|was|were|does?|did)\s+(\w+)\s+'
    r'(?:than|compared\s+to)\s+(\w+)', re.I)

_Q_HOW_MANY_DIFF = re.compile(
    r'how\s+(?:many|much)\s+(?:more|fewer|less)\s+'
    r'(?:\w+\s+)?'
    r'(?:does?|did|is|are|was|were)\s+(\w+)\s+'
    r'(?:have|has|had|earn|make|get|own)\s+'
    r'(?:than)\s+(\w+)', re.I)

_Q_LEFT = re.compile(
    r'how\s+(?:many|much)\s+(?:\w+\s+)?'
    r'(?:does?|did|is|are|was|were|will)\s+(\w+)\s+'
    r'(?:still\s+)?(?:have\s+)?(?:left|remain|need)', re.I)

_Q_TOTAL = re.compile(
    r'\b(?:total|altogether|combined|in\s+all|all\s+together)\b', re.I)

_Q_HOW_MANY = re.compile(
    r'how\s+(?:many|much)\s+\w+\s+'
    r'(?:does?|did|do|is|are|was|were|can|could|will|would)\s+(\w+)', re.I)

_Q_WHAT = re.compile(
    r'(?:what|how\s+much)\s+(?:is|was|are|were|does?|did|will)\s+'
    r'(?:the\s+)?(\w+)', re.I)


def _resolve_superlative(superlative: str, known_entities: set[str],
                         equations: list[Equation]) -> str:
    """Resolve 'youngest', 'oldest', 'smallest' etc. to an entity.

    For comparisons like A = B + 5, B = C + 3:
      - 'youngest'/'smallest' → the leaf entity (C, defined by nothing)
      - 'oldest'/'largest' → the root entity (A, not referenced by others)
    """
    # Which entities are defined in terms of other entities?
    defined_by = {}  # entity → set of entities it references
    for eq in equations:
        if isinstance(eq.lhs, Sym):
            refs = free_variables(eq.rhs) & known_entities
            if refs:
                defined_by[eq.lhs.name] = refs

    all_referenced = set()
    for refs in defined_by.values():
        all_referenced.update(refs)

    # Leaf = referenced by others but doesn't reference any entity itself
    leaves = (known_entities & all_referenced) - set(defined_by.keys())
    # Root = defines others but isn't referenced
    roots = set(defined_by.keys()) - all_referenced

    if superlative in ('youngest', 'smallest', 'shortest', 'lightest',
                       'least', 'fewest', 'lowest', 'cheapest'):
        return next(iter(leaves)) if leaves else ''
    elif superlative in ('oldest', 'largest', 'tallest', 'heaviest',
                         'most', 'highest', 'biggest'):
        return next(iter(roots)) if roots else ''
    return ''


def parse_target(question: str, known_entities: set[str],
                 all_equations: list[Equation]) -> Optional[Expr]:
    """Parse the question to identify the target expression."""
    q = question.strip()

    # Difference: "How much older is X than Y?"
    m = _Q_DIFF.search(q)
    if m:
        x = m.group(1).lower()
        y = m.group(2).lower()
        # Resolve superlatives: "the youngest" → entity with implied min
        if y == 'the':
            sup_m = re.search(r'than\s+the\s+(\w+)', q, re.I)
            if sup_m:
                sup = sup_m.group(1).lower()
                # Find the entity that's a leaf (not defined in terms of others)
                leaf = _resolve_superlative(sup, known_entities, all_equations)
                if leaf:
                    y = leaf
        if x in known_entities or y in known_entities:
            return BinExpr('-', Sym(x), Sym(y))

    m = _Q_HOW_MANY_DIFF.search(q)
    if m:
        return BinExpr('-', Sym(m.group(1).lower()), Sym(m.group(2).lower()))

    # Remainder: "How much does X have left?"
    m = _Q_LEFT.search(q)
    if m:
        x = m.group(1).lower()
        if x in known_entities:
            return Sym(x)

    # Total: "altogether" / "in all" / "total"
    if _Q_TOTAL.search(q):
        syms = []
        seen = set()
        for eq in all_equations:
            if isinstance(eq.lhs, Sym) and eq.lhs.name not in seen:
                syms.append(eq.lhs)
                seen.add(eq.lhs.name)
        if len(syms) >= 2:
            expr = BinExpr('+', syms[0], syms[1])
            for s in syms[2:]:
                expr = BinExpr('+', expr, s)
            return expr
        elif syms:
            return syms[0]

    # "How many X does Y verb?"
    m = _Q_HOW_MANY.search(q)
    if m:
        x = m.group(1).lower()
        if x in known_entities:
            return Sym(x)

    # "What is X?"
    m = _Q_WHAT.search(q)
    if m:
        x = m.group(1).lower()
        if x in known_entities:
            return Sym(x)

    # Fallback: proper noun in question
    for word in re.findall(r'\b([A-Z][a-z]+)\b', question):
        w = word.lower()
        if w in known_entities:
            return Sym(w)

    # Last resort: "total" → sum everything
    if re.search(r'\b(?:total|altogether)\b', q, re.I):
        syms = [eq.lhs for eq in all_equations if isinstance(eq.lhs, Sym)]
        if syms:
            expr = syms[0]
            for s in syms[1:]:
                expr = BinExpr('+', expr, s)
            return expr

    return None


# =====================================================================
# Substitution engine
# =====================================================================

def substitute(expr: Expr, var: str, replacement: Expr) -> Expr:
    if isinstance(expr, Sym):
        return replacement if expr.name == var else expr
    elif isinstance(expr, Num):
        return expr
    elif isinstance(expr, BinExpr):
        return BinExpr(expr.op,
                       substitute(expr.left, var, replacement),
                       substitute(expr.right, var, replacement))
    return expr

def evaluate(expr: Expr) -> Optional[float]:
    if isinstance(expr, Num):
        return expr.value
    elif isinstance(expr, Sym):
        return None
    elif isinstance(expr, BinExpr):
        left = evaluate(expr.left)
        right = evaluate(expr.right)
        if left is None or right is None:
            return None
        if expr.op == '+': return left + right
        if expr.op == '-': return left - right
        if expr.op == '*': return left * right
        if expr.op == '/': return left / right if right != 0 else None
    return None

def free_variables(expr: Expr) -> set[str]:
    if isinstance(expr, Sym):
        return {expr.name}
    elif isinstance(expr, Num):
        return set()
    elif isinstance(expr, BinExpr):
        return free_variables(expr.left) | free_variables(expr.right)
    return set()

def _to_linear(expr: Expr) -> Optional[dict[str, float]]:
    """Convert to linear form {var: coeff, '': constant}."""
    if isinstance(expr, Num):
        return {'': expr.value}
    elif isinstance(expr, Sym):
        return {expr.name: 1.0, '': 0.0}
    elif isinstance(expr, BinExpr):
        left = _to_linear(expr.left)
        right = _to_linear(expr.right)
        if left is None or right is None:
            return None
        if expr.op == '+':
            r = dict(left)
            for k, v in right.items():
                r[k] = r.get(k, 0.0) + v
            return r
        elif expr.op == '-':
            r = dict(left)
            for k, v in right.items():
                r[k] = r.get(k, 0.0) - v
            return r
        elif expr.op == '*':
            l_vars = {k for k in left if k != '' and left[k] != 0}
            r_vars = {k for k in right if k != '' and right[k] != 0}
            if not l_vars:
                c = left.get('', 0)
                return {k: v * c for k, v in right.items()}
            elif not r_vars:
                c = right.get('', 0)
                return {k: v * c for k, v in left.items()}
            return None
        elif expr.op == '/':
            r_vars = {k for k in right if k != '' and right[k] != 0}
            if not r_vars:
                c = right.get('', 0)
                if c == 0: return None
                return {k: v / c for k, v in left.items()}
            return None
    return None

def simplify(expr: Expr) -> Expr:
    """Simplify via linearization. Handles variable cancellation."""
    linear = _to_linear(expr)
    if linear is None:
        return expr
    vars_remaining = {k: v for k, v in linear.items()
                      if k != '' and abs(v) > 1e-10}
    if not vars_remaining:
        return Num(linear.get('', 0.0))
    if len(vars_remaining) == 1:
        var, coeff = next(iter(vars_remaining.items()))
        const = linear.get('', 0.0)
        base = (Sym(var) if abs(coeff - 1.0) < 1e-10
                else BinExpr('*', Num(coeff), Sym(var)))
        if abs(const) < 1e-10:
            return base
        return BinExpr('+', base, Num(const))
    return expr

def solve_system(equations: list[Equation], target: Expr,
                 max_iterations: int = 30) -> Optional[float]:
    """Solve by repeated substitution + simplification."""
    subs: dict[str, Expr] = {}
    for eq in equations:
        if not isinstance(eq.lhs, Sym):
            continue
        var = eq.lhs.name
        rhs_vars = free_variables(eq.rhs)
        if var in rhs_vars and var in subs:
            new_rhs = substitute(eq.rhs, var, subs[var])
            subs[var] = new_rhs
        else:
            subs[var] = eq.rhs

    current = target
    for _ in range(max_iterations):
        current = simplify(current)
        result = evaluate(current)
        if result is not None:
            return result
        vars_in_expr = free_variables(current)
        if not vars_in_expr:
            break
        substituted = False
        for var in list(vars_in_expr):
            if var in subs:
                current = substitute(current, var, subs[var])
                substituted = True
        if not substituted:
            break

    current = simplify(current)
    return evaluate(current)


# =====================================================================
# Main solver
# =====================================================================

def _extract_frame(text: str) -> dict:
    """
    Read the whole problem to establish the frame BEFORE parsing.

    A human reads the whole thing first. The question tells you:
    - What domain (money, time, distance, count)
    - What unit the answer needs (hours, dollars, pages)
    - What type of answer (total, difference, remainder, rate)
    - What entity is the focus

    This context flows backward to disambiguate numbers in setup sentences.
    """
    frame = {
        'domain': '',         # 'money', 'time', 'count', 'distance', 'weight'
        'target_unit': '',    # 'hours', 'dollars', 'pages', etc.
        'answer_type': '',    # 'total', 'difference', 'remainder', 'count', 'cost'
        'focus_entity': '',   # main entity the question is about
        'keywords': set(),    # important words from the whole problem
    }

    tl = text.lower()

    # Extract all meaningful words for domain detection
    frame['keywords'] = set(re.findall(r'\b[a-z]{3,}\b', tl))

    # Domain detection from keywords across the WHOLE problem
    if any(w in tl for w in ['$', 'dollar', 'price', 'cost', 'pay', 'spend',
                              'budget', 'earn', 'charge', 'money', 'cent']):
        frame['domain'] = 'money'
        frame['target_unit'] = 'dollars'
    elif any(w in tl for w in ['hour', 'minute', 'second', 'day', 'week',
                                'month', 'year', 'time']):
        frame['domain'] = 'time'
    elif any(w in tl for w in ['mile', 'kilometer', 'meter', 'feet', 'foot',
                                'inch', 'yard', 'distance', 'far']):
        frame['domain'] = 'distance'
    elif any(w in tl for w in ['pound', 'kilogram', 'ounce', 'gram',
                                'weigh', 'weight', 'heavy']):
        frame['domain'] = 'weight'
    elif any(w in tl for w in ['page', 'read', 'book', 'write', 'letter']):
        frame['domain'] = 'reading'
    else:
        frame['domain'] = 'count'

    # Find the question sentence for answer type
    q_match = re.search(r'[^.!?]*\?', text)
    if q_match:
        q = q_match.group().lower()
        if re.search(r'\b(?:total|altogether|combined|in all|all together)\b', q):
            frame['answer_type'] = 'total'
        elif re.search(r'\b(?:left|remain|still|need)\b', q):
            frame['answer_type'] = 'remainder'
        elif re.search(r'\b(?:more|fewer|less|older|younger|taller|shorter)\b.*\bthan\b', q):
            frame['answer_type'] = 'difference'
        elif re.search(r'\bhow\s+(?:long|many\s+hours|many\s+minutes)\b', q):
            frame['answer_type'] = 'duration'
            if 'hour' in q:
                frame['target_unit'] = 'hours'
            elif 'minute' in q:
                frame['target_unit'] = 'minutes'
        elif re.search(r'\bhow\s+much\b', q):
            frame['answer_type'] = 'amount'
        elif re.search(r'\bhow\s+many\b', q):
            frame['answer_type'] = 'count'

        # Target unit from question
        if not frame['target_unit']:
            unit_m = re.search(
                r'how\s+(?:many|much)\s+(\w+)', q)
            if unit_m:
                frame['target_unit'] = unit_m.group(1)

    # Rate detection: "X in/per/every Y" pattern across whole problem
    frame['has_rate'] = bool(re.search(
        r'\d+\s+\w+(?:\s+\w+)*?\s+(?:in|per|each|every|an?)\s+\d*\s*\w+', tl))

    # "in N minutes/hours" pattern (implicit rate denominator)
    in_time = re.search(r'\bin\s+(\d+)\s+(minutes?|hours?|seconds?|days?|weeks?)', tl)
    if in_time:
        frame['rate_denominator'] = (float(in_time.group(1)), in_time.group(2))
    else:
        frame['rate_denominator'] = None

    return frame


def solve(text: str) -> dict:
    """
    Read the whole problem first (like a human), then parse.

    1. Extract frame from full text (domain, target, answer type)
    2. Find and parse the question sentence
    3. Parse setup sentences with frame context
    4. Solve by substitution
    """
    # --- Step 0: Read the whole thing first ---
    frame = _extract_frame(text)

    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    clauses = []
    for s in raw_sentences:
        # Split on conjunctions that signal independent clause boundaries.
        # The comma before "and" is the structural marker — bare "and"
        # connects compound subjects/verbs/list items, not separate equations.
        parts = re.split(
            r'\s+(?:while|whereas)\s+'      # "while/whereas"
            r'|,\s*and\s+',                 # ", and" (comma = clause boundary)
            s)
        clauses.extend(p for p in parts if p and p.strip())

    known_entities: set[str] = set()
    all_equations: list[Equation] = []
    current_entity = ""
    question_sentence = None

    # --- Step 1: Find the question first ---
    setup_clauses = []
    for s in clauses:
        if '?' in s:
            question_sentence = s
            # Extract data from question sentence too
            pre_q = re.split(r',\s*(?=how\s|what\s|who\s|where\s|when\s)', s, flags=re.I)
            if len(pre_q) > 1:
                data_part = pre_q[0]
                data_part = re.sub(r'^(?:if|given\s+that|suppose|assuming)\s+',
                                   '', data_part, flags=re.I)
                setup_clauses.append(data_part)
        else:
            setup_clauses.append(s)

    # --- Step 2: Parse setup sentences with frame context ---
    for s in setup_clauses:
        eqs, current_entity = parse_clause(
            s, known_entities, all_equations, current_entity, frame)
        all_equations.extend(eqs)

    # --- Step 3: Extract numbers from question and create bridge equations ---
    if question_sentence:
        q_tokens = _lex_sentence(question_sentence)
        q_numbers = [t for t in q_tokens
                     if t.type in ('number', 'money', 'fraction')]

        # If question has numbers and we have a rate, apply rate to question number
        rate_eqs = [eq for eq in all_equations if eq.pattern == 'implicit_rate']
        if q_numbers and rate_eqs:
            rate_eq = rate_eqs[0]
            rate_var = rate_eq.lhs.name if isinstance(rate_eq.lhs, Sym) else 'rate'
            q_val = q_numbers[0].value

            # result_raw = quantity / rate (gives time in rate's unit)
            # or result_raw = quantity * rate (depends on context)
            raw_var = 'result_raw'
            # If rate is items/time, then time = items / rate
            all_equations.append(Equation(
                Sym(raw_var), BinExpr('/', Num(q_val), Sym(rate_var)),
                question_sentence, 'rate_apply_q'))
            known_entities.add(raw_var)

            # Unit conversion if needed
            if frame.get('target_unit') == 'hours' and frame.get('rate_denominator'):
                _, denom_unit = frame['rate_denominator']
                if 'minute' in denom_unit:
                    all_equations.append(Equation(
                        Sym('answer'), BinExpr('/', Sym(raw_var), Num(60)),
                        'unit conversion min→hr', 'unit_convert'))
                    known_entities.add('answer')
                elif 'second' in denom_unit:
                    all_equations.append(Equation(
                        Sym('answer'), BinExpr('/', Sym(raw_var), Num(3600)),
                        'unit conversion sec→hr', 'unit_convert'))
                    known_entities.add('answer')
            elif frame.get('target_unit') == 'minutes' and frame.get('rate_denominator'):
                _, denom_unit = frame['rate_denominator']
                if 'hour' in denom_unit:
                    all_equations.append(Equation(
                        Sym('answer'), BinExpr('*', Sym(raw_var), Num(60)),
                        'unit conversion hr→min', 'unit_convert'))
                    known_entities.add('answer')

    # --- Step 4: Parse target from question ---
    target = None
    if question_sentence:
        target = parse_target(question_sentence, known_entities, all_equations)

    # If we created an 'answer' variable from rate+question, prefer that
    if 'answer' in known_entities and target is None:
        target = Sym('answer')

    if target is None and known_entities:
        for eq in reversed(all_equations):
            if isinstance(eq.lhs, Sym):
                target = eq.lhs
                break

    if target is None:
        return {'answer': None, 'equations': all_equations,
                'target': None, 'debug': 'no target'}

    answer = solve_system(all_equations, target)
    return {'answer': answer, 'equations': all_equations,
            'target': target,
            'debug': f'{len(all_equations)} eqs, target={target}'}


def solve_oracle(text: str, expected: float) -> dict:
    """Solve and check against expected answer."""
    result = solve(text)
    answer = result['answer']
    correct = False
    if answer is not None and expected is not None:
        if abs(answer - expected) < 0.01:
            correct = True
        elif expected != 0 and abs(answer / expected - 1) < 0.001:
            correct = True
    return {
        'correct': correct,
        'answer': answer,
        'expected': expected,
        'n_equations': len(result['equations']),
        'target': str(result['target']),
        'equations': [f"{eq.lhs} = {eq.rhs}  [{eq.pattern}]"
                      for eq in result['equations']],
        'debug': result['debug'],
    }
