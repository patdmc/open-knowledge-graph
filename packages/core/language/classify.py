"""
Problem classifier — language graph layer.

Reads the STRUCTURE of a word problem to determine its equivalence class
BEFORE any math happens. This is a language problem, not a math problem.

Each equivalence class maps to a computation template:
    "rate_chain"         → dimensional analysis
    "percentage"         → X * pct / 100
    "working_backwards"  → invert operations from known remainder
    "system_of_equations"→ constraint solver (N unknowns, N equations)
    "multiplicative_rel" → scaling relationships (twice, half, double)
    "additive_rel"       → offset relationships (N more/less than)
    "fraction_of_whole"  → part/whole reasoning
    "multi_entity"       → per-entity property maps, then combine
    "sequential"         → accumulate operations in order
    "comparison"         → compute two values, find difference

The classifier outputs a RANKED list of classes with confidence.
Multiple classes can apply (a problem can be both multi_entity AND percentage).
"""

import re
from dataclasses import dataclass, field


@dataclass
class Classification:
    """A problem's equivalence class with confidence."""
    cls: str            # equivalence class name
    confidence: float   # 0-1
    evidence: str       # what triggered this classification


@dataclass
class ProblemProfile:
    """Full classification of a word problem."""
    classes: list[Classification] = field(default_factory=list)
    entity_count: int = 0
    number_count: int = 0
    question_type: str = ""  # "count", "money", "difference", "remaining"

    @property
    def primary(self) -> str:
        """The highest-confidence class."""
        return self.classes[0].cls if self.classes else "unknown"

    def has_class(self, cls: str) -> bool:
        return any(c.cls == cls for c in self.classes)

    def confidence_of(self, cls: str) -> float:
        for c in self.classes:
            if c.cls == cls:
                return c.confidence
        return 0.0


# ---------------------------------------------------------------------------
# Entity detection
# ---------------------------------------------------------------------------

_STOP_NAMES = {
    "The", "If", "She", "He", "How", "What", "In", "On", "At", "It",
    "One", "Each", "They", "Her", "His", "A", "An", "For", "But", "And",
    "Or", "So", "Yet", "From", "With", "By", "To", "Of", "Is", "Are",
    "Was", "Were", "Has", "Had", "Do", "Does", "Did", "Will", "Can",
    "May", "Not", "Then", "Than", "When", "Where", "There", "Here",
    "This", "That", "Some", "All", "Many", "Much", "Most", "Every",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
}


def _count_entities(text: str) -> int:
    """Count distinct named entities (proper nouns) in text."""
    names = re.findall(r'\b[A-Z][a-z]+\b', text)
    unique = set(n for n in names if n not in _STOP_NAMES)
    return len(unique)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(text: str) -> ProblemProfile:
    """
    Classify a word problem into equivalence classes.

    Reads sentence structure, not numbers. This is language, not math.
    Returns a ranked list of applicable classes.
    """
    profile = ProblemProfile()
    t = text.lower()
    classes: list[Classification] = []

    # --- Count structural features ---
    profile.entity_count = _count_entities(text)
    profile.number_count = len(re.findall(r'\d+', text))

    # Detect question type
    if re.search(r'\b(?:difference|more than|less than)\b.*\?', t):
        profile.question_type = "difference"
    elif re.search(r'\b(?:left|remain|still)\b.*\?', t):
        profile.question_type = "remaining"
    elif re.search(r'\bhow\s+much\b.*\$|\bhow\s+many\s+dollars\b', t):
        profile.question_type = "money"
    elif re.search(r'\btotal\b.*\?|\baltogether\b.*\?', t):
        profile.question_type = "total"
    else:
        profile.question_type = "count"

    # --- PERCENTAGE ---
    pct_matches = re.findall(r'\d+\s*%', t)
    if pct_matches or re.search(r'\b(?:percent|discount|tax|tip|markup|off)\b', t):
        conf = min(0.9, 0.5 + 0.15 * len(pct_matches))
        classes.append(Classification(
            "percentage", conf,
            f"{len(pct_matches)} percentage markers"))

    # --- RATE CHAIN (dimensional analysis) ---
    rate_words = re.findall(r'\b(?:per|each|every|an?\s+hour|a\s+day|a\s+week|a\s+month)\b', t)
    if len(rate_words) >= 2:
        classes.append(Classification(
            "rate_chain", 0.85,
            f"{len(rate_words)} rate markers"))
    elif len(rate_words) == 1 and profile.number_count >= 4:
        classes.append(Classification(
            "rate_chain", 0.5,
            "1 rate marker + many numbers"))

    # --- WORKING BACKWARDS ---
    if re.search(r'\b(?:originally|at first|started? with|began? with|before)\b', t):
        classes.append(Classification(
            "working_backwards", 0.85,
            "backward-looking question word"))
    elif re.search(r'\bhow (?:much|many).*\b(?:was|were|did)\b.*\bbefore\b', t):
        classes.append(Classification(
            "working_backwards", 0.9,
            "how many...before pattern"))
    # Also: "the last N" with consuming verbs suggests working back
    if re.search(r'\bthe\s+(?:last|remaining|rest)\b.*\d+', t):
        if re.search(r'\b(?:ate|scavenged|consumed|took|used)\b.*\b(?:half|third|quarter)\b', t):
            classes.append(Classification(
                "working_backwards", 0.8,
                "sequential consumption with known remainder"))

    # --- MULTIPLICATIVE RELATION ---
    mult_markers = re.findall(
        r'\b(?:twice|triple|double|half|three\s+times|four\s+times|'
        r'two\s+times|five\s+times|ten\s+times|'
        r'as\s+(?:old|much|many|big|tall|fast|long)\s+as)\b', t)
    if mult_markers:
        classes.append(Classification(
            "multiplicative_rel", 0.7 + 0.1 * len(mult_markers),
            f"{len(mult_markers)} multiplicative markers"))

    # --- ADDITIVE RELATION ---
    add_rel = re.findall(
        r'\b\d+\s+(?:more|less|fewer|older|younger|taller|shorter|'
        r'heavier|lighter|longer|cheaper|expensive)\s+than\b', t)
    if add_rel:
        classes.append(Classification(
            "additive_rel", 0.7 + 0.1 * len(add_rel),
            f"{len(add_rel)} additive comparison markers"))

    # --- SYSTEM OF EQUATIONS ---
    has_total_constraint = bool(re.search(
        r'\b(?:total|altogether|combined|in all)\b', t))
    has_relationship = bool(re.search(
        r'\b(?:twice|more than|less than|as many|times)\b', t))
    has_two_unknowns = profile.entity_count >= 2 and re.search(
        r'\bhow many\b', t)
    if has_total_constraint and has_relationship and has_two_unknowns:
        classes.append(Classification(
            "system_of_equations", 0.8,
            "total constraint + relationship + unknowns"))
    elif re.search(r'\b(?:heads?|bumps?|legs?|wheels?)\b', t) and \
         re.search(r'\b(?:each|per)\b', t) and profile.entity_count >= 2:
        classes.append(Classification(
            "system_of_equations", 0.75,
            "count constraints with per-type properties"))

    # --- FRACTION OF WHOLE ---
    frac_markers = re.findall(
        r'\b(?:half|third|quarter|one-third|two-thirds|three-quarters|'
        r'one-fourth|three-fourths|one-half|a\s+third|a\s+quarter)\b', t)
    if frac_markers:
        classes.append(Classification(
            "fraction_of_whole", 0.6 + 0.1 * len(frac_markers),
            f"{len(frac_markers)} fraction markers"))

    # --- MULTI-ENTITY ---
    if profile.entity_count >= 3:
        classes.append(Classification(
            "multi_entity", 0.6 + 0.05 * profile.entity_count,
            f"{profile.entity_count} named entities"))
    elif profile.entity_count == 2:
        classes.append(Classification(
            "multi_entity", 0.4,
            "2 named entities"))

    # --- COMPARISON ---
    if re.search(r'\b(?:difference|how much (?:more|less|cheaper|faster))\b', t):
        classes.append(Classification(
            "comparison", 0.8,
            "comparison question"))

    # --- SEQUENTIAL (default/fallback) ---
    if profile.number_count >= 2:
        # Lower confidence — this is the fallback
        classes.append(Classification(
            "sequential", 0.3,
            f"{profile.number_count} numbers, default sequential"))

    # Sort by confidence (highest first)
    classes.sort(key=lambda c: -c.confidence)
    profile.classes = classes

    return profile
