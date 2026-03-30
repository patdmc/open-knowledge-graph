"""
Math operation definitions — verb→operation mappings and curried arithmetic.

The action graph for mathematics. Each operation has:
  - verbs: natural language triggers that map to this operation
  - curry: the curried Python function
  - inverse: the inverse operation (for verification)
  - identity: the identity element

This is the math equivalent of the language graph's homophone sets:
verbs are addresses, operations are the resolved meaning.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Operation:
    """A mathematical operation with its natural language triggers."""
    id: str
    name: str
    verbs: list[str]          # NL triggers that resolve to this operation
    curry: Callable           # the executable function
    inverse_id: str = ""      # inverse operation for verification
    identity: float = 0.0     # identity element
    commutative: bool = False


# ---------------------------------------------------------------------------
# Arithmetic operations — curried, composable
# ---------------------------------------------------------------------------

OPERATIONS = {
    "add": Operation(
        id="add",
        name="Addition",
        verbs=[
            "add", "adds", "added", "plus", "more", "gain", "gains", "gained",
            "receive", "receives", "received", "get", "gets", "got",
            "earn", "earns", "earned", "collect", "collects", "collected",
            "buy", "buys", "bought", "find", "finds", "found",
            "join", "joins", "joined", "increase", "increases", "increased",
            "total", "altogether", "combined", "sum", "additional",
            "gave him", "gave her", "gave them",
        ],
        curry=lambda a, b: a + b,
        inverse_id="subtract",
        identity=0.0,
        commutative=True,
    ),

    "subtract": Operation(
        id="subtract",
        name="Subtraction",
        verbs=[
            "subtract", "minus", "less", "lose", "loses", "lost",
            "eat", "eats", "ate", "use", "uses", "used",
            "give", "gives", "gave", "spend", "spends", "spent",
            "remove", "removes", "removed", "take", "takes", "took",
            "discard", "discards", "discarded",
            "break", "breaks", "broke", "throw", "throws", "threw",
            "donate", "donates", "donated", "leave", "left",
            "fewer", "less than", "decrease", "decreases", "decreased",
            "remainder", "remaining", "left over",
            "get rid of", "gets rid of", "got rid of",
            "bake", "bakes", "baked",  # consumes ingredients
            "cook", "cooks", "cooked",
            "paint", "paints", "painted",  # uses up materials
        ],
        curry=lambda a, b: a - b,
        inverse_id="add",
        identity=0.0,
    ),

    "multiply": Operation(
        id="multiply",
        name="Multiplication",
        verbs=[
            "multiply", "times", "twice", "double", "triple", "quadruple",
            "per", "each", "every", "rate", "at",
        ],
        curry=lambda a, b: a * b,
        inverse_id="divide",
        identity=1.0,
        commutative=True,
    ),

    "divide": Operation(
        id="divide",
        name="Division",
        verbs=[
            "divide", "divides", "divided", "split", "splits",
            "share", "shares", "shared", "distribute", "distributes",
            "average", "half", "third", "quarter",
            "ratio",
        ],
        curry=lambda a, b: a / b if b != 0 else float('inf'),
        inverse_id="multiply",
        identity=1.0,
    ),

    "remainder": Operation(
        id="remainder",
        name="Modulo/Remainder",
        verbs=[
            "modulo", "mod",
        ],
        curry=lambda a, b: a % b if b != 0 else 0,
    ),

    "percent": Operation(
        id="percent",
        name="Percentage",
        verbs=[
            "percent", "%", "percentage", "discount", "tax", "tip",
            "markup", "markdown", "off", "increase by", "decrease by",
        ],
        curry=lambda whole, pct: whole * pct / 100,
    ),

    "power": Operation(
        id="power",
        name="Exponentiation",
        verbs=[
            "squared", "cubed", "power", "exponent",
        ],
        curry=lambda base, exp: base ** exp,
        identity=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Verb → Operation resolver
# ---------------------------------------------------------------------------

# Build reverse index: verb → operation_id
_VERB_INDEX: dict[str, str] = {}
for op_id, op in OPERATIONS.items():
    for verb in op.verbs:
        _VERB_INDEX[verb.lower()] = op_id


def resolve_verb(verb: str) -> Optional[Operation]:
    """Resolve a natural language verb to its mathematical operation."""
    v = verb.lower().strip()
    op_id = _VERB_INDEX.get(v)
    if op_id:
        return OPERATIONS[op_id]

    # Try partial match (verb is substring of a trigger or vice versa)
    for trigger, oid in _VERB_INDEX.items():
        if v in trigger or trigger in v:
            return OPERATIONS[oid]

    return None


def execute_chain(initial: float, steps: list[tuple[str, float]]) -> float:
    """
    Execute a chain of operations.

    steps: [(operation_id, operand), ...]
    Returns the final result.

    Example:
        execute_chain(16, [("subtract", 3), ("subtract", 4), ("multiply", 2)])
        → 18
    """
    result = initial
    for op_id, operand in steps:
        op = OPERATIONS.get(op_id)
        if op:
            result = op.curry(result, operand)
    return result
