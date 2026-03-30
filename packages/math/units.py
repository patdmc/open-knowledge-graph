"""
Math graph — unit conversion edges.

Each conversion is an edge: unit_A --CONVERTS_TO--> unit_B with a factor.
The graph is bidirectional: if A→B has factor f, then B→A has factor 1/f.

This is math knowledge — the WHAT. The language graph handles the HOW
(recognizing "per hour" or "every 30 minutes" in text).

Architecture:
    Define canonical units per dimension (seconds for time, inches for length).
    All units in a dimension convert through the canonical unit.
    To convert A→B: factor = canonical_value(A) / canonical_value(B).

    This means we only store ONE number per unit (its value in canonical units),
    not N² pairwise conversions. The graph computes the conversion on the fly.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Unit:
    """A unit of measurement — a node in the math graph."""
    name: str               # canonical name: "second", "inch", "dollar"
    aliases: list[str]      # other names: ["seconds", "sec", "secs", "s"]
    dimension: str          # what it measures: "time", "length", "currency"
    canonical_value: float  # value in canonical units for this dimension


# ---------------------------------------------------------------------------
# Dimension definitions — canonical unit + all units
# ---------------------------------------------------------------------------

_UNITS: list[Unit] = [
    # --- Time (canonical: second) ---
    Unit("millisecond", ["milliseconds", "ms"], "time", 0.001),
    Unit("second", ["seconds", "sec", "secs"], "time", 1),
    Unit("minute", ["minutes", "min", "mins"], "time", 60),
    Unit("hour", ["hours", "hr", "hrs"], "time", 3600),
    Unit("day", ["days"], "time", 86400),
    Unit("week", ["weeks"], "time", 604800),
    Unit("month", ["months"], "time", 2592000),     # 30 days
    Unit("year", ["years"], "time", 31536000),      # 365 days

    # --- Length: Imperial (canonical: inch) ---
    Unit("inch", ["inches", "in"], "length_imperial", 1),
    Unit("foot", ["feet", "ft"], "length_imperial", 12),
    Unit("yard", ["yards", "yd"], "length_imperial", 36),
    Unit("mile", ["miles", "mi"], "length_imperial", 63360),

    # --- Length: Metric (canonical: centimeter) ---
    Unit("millimeter", ["millimeters", "mm"], "length_metric", 0.1),
    Unit("centimeter", ["centimeters", "cm"], "length_metric", 1),
    Unit("meter", ["meters", "m"], "length_metric", 100),
    Unit("kilometer", ["kilometers", "km"], "length_metric", 100000),

    # --- Currency: USD (canonical: dollar) ---
    Unit("penny", ["pennies"], "currency_usd", 0.01),
    Unit("nickel", ["nickels"], "currency_usd", 0.05),
    Unit("dime", ["dimes"], "currency_usd", 0.10),
    Unit("quarter", ["quarters"], "currency_usd", 0.25),
    Unit("dollar", ["dollars"], "currency_usd", 1.00),

    # --- Quantity multipliers (canonical: 1) ---
    Unit("each", ["single", "one"], "quantity", 1),
    Unit("pair", ["pairs"], "quantity", 2),
    Unit("couple", [], "quantity", 2),
    Unit("dozen", ["dozens"], "quantity", 12),
    Unit("score", [], "quantity", 20),
    Unit("gross", [], "quantity", 144),
]

# ---------------------------------------------------------------------------
# Index: word → Unit (built once, O(1) lookup)
# ---------------------------------------------------------------------------

_WORD_TO_UNIT: dict[str, Unit] = {}
for _u in _UNITS:
    _WORD_TO_UNIT[_u.name] = _u
    for _alias in _u.aliases:
        _WORD_TO_UNIT[_alias] = _u

# Dimension → canonical value per word (for backward compat)
_DIMENSION_UNITS: dict[str, dict[str, float]] = {}
for _u in _UNITS:
    if _u.dimension not in _DIMENSION_UNITS:
        _DIMENSION_UNITS[_u.dimension] = {}
    _DIMENSION_UNITS[_u.dimension][_u.name] = _u.canonical_value
    for _alias in _u.aliases:
        _DIMENSION_UNITS[_u.dimension][_alias] = _u.canonical_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup(word: str) -> Optional[Unit]:
    """Look up a unit by name or alias."""
    return _WORD_TO_UNIT.get(word.lower())


def convert(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Convert a value from one unit to another.

    Returns None if units are in different dimensions (can't convert
    hours to inches).

    Example:
        convert(30, "minutes", "hours")  → 0.5
        convert(3, "feet", "inches")     → 36
        convert(2, "quarters", "dollars") → 0.50
    """
    src = lookup(from_unit)
    dst = lookup(to_unit)
    if not src or not dst:
        return None
    if src.dimension != dst.dimension:
        return None
    if dst.canonical_value == 0:
        return None
    return value * src.canonical_value / dst.canonical_value


def same_dimension(word_a: str, word_b: str) -> bool:
    """Check if two unit words are in the same dimension."""
    a = lookup(word_a)
    b = lookup(word_b)
    if not a or not b:
        return False
    return a.dimension == b.dimension


def is_unit(word: str) -> bool:
    """Check if a word is a known unit."""
    return word.lower() in _WORD_TO_UNIT


def dimension_of(word: str) -> Optional[str]:
    """Get the dimension of a unit word."""
    u = lookup(word)
    return u.dimension if u else None


# ---------------------------------------------------------------------------
# Backward-compatible dict accessors (for interpret.py migration)
# ---------------------------------------------------------------------------

def to_seconds(word: str) -> Optional[float]:
    """Get the value of a time unit in seconds."""
    return _DIMENSION_UNITS.get("time", {}).get(word.lower())


def to_hours(word: str) -> Optional[float]:
    """Get the value of a time unit in hours."""
    secs = to_seconds(word)
    if secs is None:
        return None
    return secs / 3600


def to_dollars(word: str) -> Optional[float]:
    """Get the value of a currency unit in dollars."""
    return _DIMENSION_UNITS.get("currency_usd", {}).get(word.lower())


def to_inches(word: str) -> Optional[float]:
    """Get the value of a length unit in inches."""
    return _DIMENSION_UNITS.get("length_imperial", {}).get(word.lower())


# ---------------------------------------------------------------------------
# Legacy dict exports (for gradual migration from interpret.py)
# These will be removed once interpret.py uses the API directly.
# ---------------------------------------------------------------------------

UNIT_TO_DOLLARS = _DIMENSION_UNITS.get("currency_usd", {})
UNIT_TO_SECONDS = _DIMENSION_UNITS.get("time", {})
UNIT_TO_HOURS = {k: v / 3600 for k, v in _DIMENSION_UNITS.get("time", {}).items()}
UNIT_TO_MINUTES = {k: v / 60 for k, v in _DIMENSION_UNITS.get("time", {}).items()}
UNIT_TO_DAYS = {k: v / 86400 for k, v in _DIMENSION_UNITS.get("time", {}).items()}
UNIT_TO_WEEKS = {k: v / 604800 for k, v in _DIMENSION_UNITS.get("time", {}).items()}
UNIT_TO_INCHES = _DIMENSION_UNITS.get("length_imperial", {})
UNIT_TO_FEET = {k: v / 12 for k, v in _DIMENSION_UNITS.get("length_imperial", {}).items()}
UNIT_TO_MILES = {k: v / 63360 for k, v in _DIMENSION_UNITS.get("length_imperial", {}).items()}
UNIT_TO_CM = _DIMENSION_UNITS.get("length_metric", {})
