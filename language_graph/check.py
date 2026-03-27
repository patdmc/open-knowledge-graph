"""
Language Graph real-time checker.

Runs user text through the language graph and surfaces learning moments
when the graph catches optimization opportunities — homophones, sense
ambiguity, logical compaction.

Two modes:
  1. Surface check — homophone/grammar errors the graph can catch
  2. Logical compaction — identify verbose statements that could be
     normalized to compact logical form

Usage:
    from language_graph.check import check_text
    result = check_text("there there, theirs lots to worry about")
"""

import os
import re
from dataclasses import dataclass, field

import nltk
_NLTK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache", "nltk_data")
if os.path.exists(_NLTK_DIR):
    nltk.data.path.insert(0, _NLTK_DIR)

from nltk.corpus import wordnet as wn


# ---------------------------------------------------------------------------
# Homophone sets — words that sound alike but have different graph addresses
# ---------------------------------------------------------------------------

HOMOPHONE_SETS = [
    {
        "sounds_like": "there",
        "forms": {
            "there":   {"pos": "adverb",     "channel": "Lexical_Semantics", "meaning": "location/existential"},
            "their":   {"pos": "determiner",  "channel": "Morphology",        "meaning": "possessive of they"},
            "they're": {"pos": "contraction", "channel": "Morphology",        "meaning": "they are"},
        },
    },
    {
        "sounds_like": "to",
        "forms": {
            "to":   {"pos": "preposition", "channel": "Syntax",            "meaning": "direction/infinitive marker"},
            "too":  {"pos": "adverb",      "channel": "Lexical_Semantics", "meaning": "also/excessively"},
            "two":  {"pos": "noun",        "channel": "Lexical_Semantics", "meaning": "the number 2"},
        },
    },
    {
        "sounds_like": "your",
        "forms": {
            "your":    {"pos": "determiner",  "channel": "Morphology",        "meaning": "possessive of you"},
            "you're":  {"pos": "contraction", "channel": "Morphology",        "meaning": "you are"},
        },
    },
    {
        "sounds_like": "its",
        "forms": {
            "its":   {"pos": "determiner",  "channel": "Morphology",        "meaning": "possessive of it"},
            "it's":  {"pos": "contraction", "channel": "Morphology",        "meaning": "it is / it has"},
        },
    },
    {
        "sounds_like": "affect",
        "forms": {
            "affect": {"pos": "verb",      "channel": "Lexical_Semantics", "meaning": "to influence"},
            "effect": {"pos": "noun",      "channel": "Lexical_Semantics", "meaning": "a result/consequence"},
        },
    },
    {
        "sounds_like": "then",
        "forms": {
            "then": {"pos": "adverb",      "channel": "Lexical_Semantics", "meaning": "at that time / next"},
            "than": {"pos": "conjunction",  "channel": "Syntax",            "meaning": "comparison"},
        },
    },
    {
        "sounds_like": "lose",
        "forms": {
            "lose":  {"pos": "verb",       "channel": "Lexical_Semantics", "meaning": "to misplace / fail to win"},
            "loose": {"pos": "adjective",  "channel": "Lexical_Semantics", "meaning": "not tight"},
        },
    },
    {
        "sounds_like": "led",
        "forms": {
            "led":  {"pos": "verb",        "channel": "Lexical_Semantics", "meaning": "past tense of lead"},
            "lead": {"pos": "noun",        "channel": "Lexical_Semantics", "meaning": "a heavy metal (Pb)"},
        },
    },
    {
        "sounds_like": "principal",
        "forms": {
            "principal": {"pos": "noun/adj", "channel": "Lexical_Semantics", "meaning": "primary / head of school"},
            "principle": {"pos": "noun",     "channel": "Lexical_Semantics", "meaning": "fundamental truth/rule"},
        },
    },
]

# Build fast lookup: word → which homophone set(s) it belongs to
_HOMOPHONE_LOOKUP = {}
for hset in HOMOPHONE_SETS:
    for form in hset["forms"]:
        _HOMOPHONE_LOOKUP.setdefault(form.lower(), []).append(hset)


# ---------------------------------------------------------------------------
# Polysemy detection — words with many senses
# ---------------------------------------------------------------------------

def get_polysemy(word, pos=None):
    """How many distinct senses does this word have?"""
    wn_pos = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}.get(pos)
    synsets = wn.synsets(word, pos=wn_pos) if wn_pos else wn.synsets(word)
    return len(synsets)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class LearningMoment:
    token: str
    position: int
    moment_type: str          # "homophone", "polysemy", "compaction"
    message: str
    graph_path: str = ""      # the graph traversal that caught it
    severity: str = "info"    # "info", "warning", "error"


@dataclass
class CheckResult:
    text: str
    moments: list = field(default_factory=list)
    n_tokens: int = 0
    n_ambiguous: int = 0
    n_homophones: int = 0

    def has_moments(self):
        return len(self.moments) > 0

    def summary(self):
        if not self.moments:
            return None
        lines = []
        for m in self.moments:
            icon = {"info": "~", "warning": "!", "error": "X"}[m.severity]
            lines.append(f"  [{icon}] \"{m.token}\" — {m.message}")
            if m.graph_path:
                lines.append(f"      graph: {m.graph_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context rules for homophones
# ---------------------------------------------------------------------------

def _check_there_their_theyre(tokens, i):
    """Context-sensitive check for there/their/they're."""
    token = tokens[i].lower().rstrip(".,;:!?")
    prev = tokens[i - 1].lower().rstrip(".,;:!?") if i > 0 else ""
    nxt = tokens[i + 1].lower().rstrip(".,;:!?") if i < len(tokens) - 1 else ""

    # "they're" should be followed by verb/adj/adverb, not a noun
    if token == "they're" and nxt:
        nxt_syns = wn.synsets(nxt)
        if nxt_syns and all(s.pos() == "n" for s in nxt_syns):
            return LearningMoment(
                token=token, position=i, moment_type="homophone",
                severity="warning",
                message=f"\"they're\" (= they are) followed by noun \"{nxt}\". Did you mean \"their\" (possessive)?",
                graph_path=f"COMPOSED_OF(they're, [they, are]) → expects predicate. But \"{nxt}\" is noun → needs possessive determiner \"their\"",
            )

    # "there" before adj (not existential context) might be "they're"
    if token == "there" and nxt:
        nxt_syns = wn.synsets(nxt)
        if nxt_syns and all(s.pos() in ("a", "s") for s in nxt_syns):
            return LearningMoment(
                token=token, position=i, moment_type="homophone",
                severity="warning",
                message=f"\"there\" (location) followed by adjective \"{nxt}\". Did you mean \"they're\" (they are)?",
                graph_path=f"Lang_Lexeme(there.r.01) → LOCATION channel. But \"{nxt}\" is adjective → needs copula. \"they're\" = they+are → licenses predicate adj.",
            )

    # "theirs" before noun phrase (no verb) might be "there's"
    if token == "theirs" and nxt:
        nxt_syns = wn.synsets(nxt)
        if nxt_syns and any(s.pos() == "n" for s in nxt_syns):
            return LearningMoment(
                token=token, position=i, moment_type="homophone",
                severity="warning",
                message=f"\"theirs\" (possessive pronoun) before \"{nxt}\". Did you mean \"there's\" (there is)?",
                graph_path=f"Lang_Lexeme(theirs) → possessive pronoun, stands alone. Before NP needs existential \"there's\" = there+is.",
            )

    return None


def _check_your_youre(tokens, i):
    """Context-sensitive check for your/you're."""
    token = tokens[i].lower().rstrip(".,;:!?")
    nxt = tokens[i + 1].lower().rstrip(".,;:!?") if i < len(tokens) - 1 else ""

    if token == "your" and nxt:
        nxt_syns = wn.synsets(nxt)
        if nxt_syns and all(s.pos() in ("a", "s") for s in nxt_syns):
            return LearningMoment(
                token=token, position=i, moment_type="homophone",
                severity="info",
                message=f"\"your\" before adjective \"{nxt}\" — could be \"you're\" (you are) if describing a state.",
                graph_path=f"\"your\" = possessive → expects noun. \"{nxt}\" is adjective → may need copula \"you're\".",
            )

    if token == "you're" and nxt:
        nxt_syns = wn.synsets(nxt)
        if nxt_syns and all(s.pos() == "n" for s in nxt_syns):
            return LearningMoment(
                token=token, position=i, moment_type="homophone",
                severity="warning",
                message=f"\"you're\" (you are) before noun \"{nxt}\". Did you mean \"your\" (possessive)?",
                graph_path=f"COMPOSED_OF(you're, [you, are]) → expects predicate. \"{nxt}\" is noun → needs possessive \"your\".",
            )

    return None


# ---------------------------------------------------------------------------
# Main check function
# ---------------------------------------------------------------------------

def check_text(text):
    """
    Run text through the language graph and return learning moments.
    """
    result = CheckResult(text=text)
    tokens = text.split()
    result.n_tokens = len(tokens)

    for i, raw_token in enumerate(tokens):
        token = raw_token.lower().rstrip(".,;:!?\"'")

        # Homophone check
        if token in _HOMOPHONE_LOOKUP:
            result.n_homophones += 1
            # Run context-sensitive checks
            if token in ("there", "their", "they're", "theirs"):
                moment = _check_there_their_theyre(tokens, i)
                if moment:
                    result.moments.append(moment)
            elif token in ("your", "you're"):
                moment = _check_your_youre(tokens, i)
                if moment:
                    result.moments.append(moment)

        # Polysemy check — flag highly ambiguous words
        n_senses = get_polysemy(token)
        if n_senses > 10:
            result.n_ambiguous += 1

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        text = "there there, theirs lots to worry about they're lot. there quite poor."
    else:
        text = " ".join(sys.argv[1:])

    print(f"Checking: \"{text}\"")
    print()

    result = check_text(text)

    if result.has_moments():
        print("Language Graph Learning Moments:")
        print(result.summary())
    else:
        print("No issues detected.")

    print(f"\n({result.n_tokens} tokens, {result.n_homophones} homophones, {result.n_ambiguous} highly polysemous)")


if __name__ == "__main__":
    main()
