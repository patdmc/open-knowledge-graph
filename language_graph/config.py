"""
Language Knowledge Graph — Configuration

Layer 0: foundational graph that encodes the structure of English.
Every other domain graph's nodes and edges are labeled with words —
this graph tells you what those words mean and how they combine.

Mirrors gnn/config.py pattern: channels, node types, edge types,
Neo4j connection, and derived lookups.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG_CACHE = os.path.join(ROOT, "language_graph", "data", "cache")
LANG_RESULTS = os.path.join(ROOT, "language_graph", "results")

os.makedirs(LANG_CACHE, exist_ok=True)
os.makedirs(LANG_RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Neo4j connection (shared instance with cancer graph)
# ---------------------------------------------------------------------------

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "openknowledgegraph")

# ---------------------------------------------------------------------------
# Graph identity
# ---------------------------------------------------------------------------

GRAPH_NAME = "language"
GRAPH_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Node types — neo4j labels
# ---------------------------------------------------------------------------

NODE_TYPES = {
    # Surface form nodes
    "Morpheme":     "Lang_Morpheme",
    "Lexeme":       "Lang_Lexeme",
    "WordForm":     "Lang_WordForm",
    "Frame":        "Lang_Frame",
    "Construction": "Lang_Construction",
    "Domain":       "Lang_Domain",
    # Logical form nodes — compact normal form for knowledge storage
    "Predicate":    "Lang_Predicate",     # CAUSE, INHIBIT, ENABLE, IS_A, HAS_PROPERTY
    "Proposition":  "Lang_Proposition",   # a complete logical statement: P(x,y) → Q(x)
    "Variable":     "Lang_Variable",      # bound/free variables: x, y, z
    "Constant":     "Lang_Constant",      # named entities resolved to graph nodes
    "Quantifier":   "Lang_Quantifier",    # ∀, ∃, ∄, most, some, few
    "Connective":   "Lang_Connective",    # ∧, ∨, →, ↔, ¬
}

# ---------------------------------------------------------------------------
# Edge types — 13 total, matching D11 schema
# ---------------------------------------------------------------------------

EDGE_TYPES = [
    # Morphological
    {"name": "COMPOSED_OF",             "directed": True,  "channel": "Morphology",
     "source": "Lexeme",  "target": "Morpheme",
     "sign": +1, "holdback": False},
    {"name": "INFLECTS_TO",             "directed": True,  "channel": "Morphology",
     "source": "Lexeme",  "target": "WordForm",
     "sign": +1, "holdback": False},

    # Lexical semantics
    {"name": "SYNONYMOUS",              "directed": False, "channel": "Lexical_Semantics",
     "source": "Lexeme",  "target": "Lexeme",
     "sign": +1, "holdback": False},
    {"name": "ANTONYMOUS",              "directed": False, "channel": "Lexical_Semantics",
     "source": "Lexeme",  "target": "Lexeme",
     "sign": -1, "holdback": False},
    {"name": "ENTAILS",                 "directed": True,  "channel": "Lexical_Semantics",
     "source": "Lexeme",  "target": "Lexeme",
     "sign": +1, "holdback": False},
    {"name": "PART_OF",                 "directed": True,  "channel": "Lexical_Semantics",
     "source": "Lexeme",  "target": "Lexeme",
     "sign": +1, "holdback": False},
    {"name": "HYPERNYM_OF",             "directed": True,  "channel": "Lexical_Semantics",
     "source": "Lexeme",  "target": "Lexeme",
     "sign": +1, "holdback": False},

    # Compositional semantics
    {"name": "EVOKES",                  "directed": True,  "channel": "Compositional_Semantics",
     "source": "Lexeme",  "target": "Frame",
     "sign": +1, "holdback": False},
    {"name": "SELECTIONAL_PREFERENCE",  "directed": True,  "channel": "Compositional_Semantics",
     "source": "Lexeme",  "target": "Frame",
     "sign": +1, "holdback": False},

    # Pragmatics / discourse
    {"name": "CAUSES",                  "directed": True,  "channel": "Pragmatics",
     "source": "Lexeme",  "target": "Lexeme",
     "sign": +1, "holdback": False},

    # Register / style
    {"name": "DOMAIN_SHIFTS",           "directed": True,  "channel": "Register_Style",
     "source": "Domain",  "target": "Lexeme",
     "sign": 0,  "holdback": False},

    # Syntax
    {"name": "CONSTRUCTIONAL_MEANING",  "directed": True,  "channel": "Syntax",
     "source": "Construction", "target": "Lexeme",
     "sign": 0,  "holdback": False},

    # Holdback / exam edge
    {"name": "COLLOCATES",              "directed": False, "channel": "Discourse",
     "source": "Lexeme",  "target": "Lexeme",
     "sign": 0,  "holdback": True},

    # --- Logical semantics edges ---

    # Surface → logical form bridge
    {"name": "DENOTES_PREDICATE",       "directed": True,  "channel": "Logical_Semantics",
     "source": "Lexeme",  "target": "Predicate",
     "sign": +1, "holdback": False},
    # "inhibit" → INHIBIT, "cause" → CAUSE, "mutate" → MODIFY

    {"name": "DENOTES_CONSTANT",        "directed": True,  "channel": "Logical_Semantics",
     "source": "Lexeme",  "target": "Constant",
     "sign": +1, "holdback": False},
    # "TP53" → constant:TP53, "BRCA1" → constant:BRCA1

    {"name": "DENOTES_QUANTIFIER",      "directed": True,  "channel": "Logical_Semantics",
     "source": "Lexeme",  "target": "Quantifier",
     "sign": +1, "holdback": False},
    # "all" → ∀, "some" → ∃, "no" → ∄

    # Proposition structure
    {"name": "HAS_PREDICATE",           "directed": True,  "channel": "Logical_Semantics",
     "source": "Proposition", "target": "Predicate",
     "sign": +1, "holdback": False},

    {"name": "HAS_ARGUMENT",            "directed": True,  "channel": "Logical_Semantics",
     "source": "Proposition", "target": "Constant",
     "sign": +1, "holdback": False},
    # P(x, y) → HAS_ARGUMENT → x, HAS_ARGUMENT → y (with role property)

    {"name": "BOUND_BY",                "directed": True,  "channel": "Logical_Semantics",
     "source": "Variable",  "target": "Quantifier",
     "sign": +1, "holdback": False},
    # x BOUND_BY ∀ means "for all x"

    # Proposition-to-proposition connectives
    {"name": "IMPLIES",                 "directed": True,  "channel": "Logical_Semantics",
     "source": "Proposition", "target": "Proposition",
     "sign": +1, "holdback": False},
    # P → Q (if P then Q)

    {"name": "CONTRADICTS",             "directed": False, "channel": "Logical_Semantics",
     "source": "Proposition", "target": "Proposition",
     "sign": -1, "holdback": False},
    # P ∧ Q = ⊥ (cannot both be true)

    {"name": "EQUIVALENT",              "directed": False, "channel": "Logical_Semantics",
     "source": "Proposition", "target": "Proposition",
     "sign": +1, "holdback": False},
    # P ↔ Q (same truth conditions, different surface forms)

    # Predicate hierarchy
    {"name": "PREDICATE_ENTAILS",       "directed": True,  "channel": "Logical_Semantics",
     "source": "Predicate", "target": "Predicate",
     "sign": +1, "holdback": False},
    # KILL entails DIE, INHIBIT entails REDUCE

    {"name": "PREDICATE_INVERSE",       "directed": False, "channel": "Logical_Semantics",
     "source": "Predicate", "target": "Predicate",
     "sign": -1, "holdback": False},
    # INHIBIT inverse of ACTIVATE, CAUSE inverse of PREVENT
]

EDGE_TYPE_NAMES = [e["name"] for e in EDGE_TYPES]
EDGE_TYPE_MAP = {e["name"]: e for e in EDGE_TYPES}

# ---------------------------------------------------------------------------
# Channels — 8 linguistic subsystems organized in 4 tiers
# Equivalent to cancer channels (DDR, PI3K, CellCycle, etc.)
# ---------------------------------------------------------------------------

CHANNELS = {
    # Tier 1: Formal Structure
    "Morphology": {
        "tier": "Formal_Structure",
        "description": "Word formation rules. Morphemes and inflection patterns.",
        "edge_types": ["COMPOSED_OF", "INFLECTS_TO"],
    },
    "Syntax": {
        "tier": "Formal_Structure",
        "description": "Sentence structure. Constructions and dependency patterns.",
        "edge_types": ["CONSTRUCTIONAL_MEANING"],
    },

    # Tier 2: Meaning
    "Lexical_Semantics": {
        "tier": "Meaning",
        "description": "Word-level meaning. Synonymy, antonymy, hypernymy networks.",
        "edge_types": ["SYNONYMOUS", "ANTONYMOUS", "ENTAILS", "PART_OF", "HYPERNYM_OF"],
    },
    "Compositional_Semantics": {
        "tier": "Meaning",
        "description": "How meanings combine. Frames, thematic roles, predicate-argument.",
        "edge_types": ["EVOKES", "SELECTIONAL_PREFERENCE"],
    },
    "Logical_Semantics": {
        "tier": "Meaning",
        "description": "Compact normal form for knowledge. Predicates, propositions, quantifiers, connectives. The bridge between natural language and graph edges.",
        "edge_types": [
            "DENOTES_PREDICATE", "DENOTES_CONSTANT", "DENOTES_QUANTIFIER",
            "HAS_PREDICATE", "HAS_ARGUMENT", "BOUND_BY",
            "IMPLIES", "CONTRADICTS", "EQUIVALENT",
            "PREDICATE_ENTAILS", "PREDICATE_INVERSE",
        ],
    },

    # Tier 3: Use
    "Pragmatics": {
        "tier": "Use",
        "description": "Meaning in context. Speech acts, implicature, causation.",
        "edge_types": ["CAUSES"],
    },
    "Discourse": {
        "tier": "Use",
        "description": "Text-level structure. Coherence, collocation, anaphora.",
        "edge_types": ["COLLOCATES"],
    },

    # Tier 4: Meta-Linguistic
    "Register_Style": {
        "tier": "Meta_Linguistic",
        "description": "How language varies by social/domain context.",
        "edge_types": ["DOMAIN_SHIFTS"],
    },
    "Etymology_History": {
        "tier": "Meta_Linguistic",
        "description": "How language changes over time. Borrowing, semantic drift.",
        "edge_types": [],
    },
}

CHANNEL_NAMES = sorted(CHANNELS.keys())
CHANNEL_TO_IDX = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
TIERS = sorted(set(v["tier"] for v in CHANNELS.values()))

# ---------------------------------------------------------------------------
# Data sources — ordered by priority
# ---------------------------------------------------------------------------

DATA_SOURCES = [
    {
        "name": "WordNet 3.1",
        "priority": 1,
        "license": "Princeton License (open)",
        "nodes": ["Lexeme"],
        "edges": ["SYNONYMOUS", "ANTONYMOUS", "ENTAILS", "PART_OF", "HYPERNYM_OF"],
        "url": "https://wordnet.princeton.edu/",
        "status": "pending",
    },
    {
        "name": "FrameNet 1.7",
        "priority": 2,
        "license": "CC-BY 3.0",
        "nodes": ["Frame"],
        "edges": ["EVOKES"],
        "url": "https://framenet.icsi.berkeley.edu/",
        "status": "pending",
    },
    {
        "name": "ConceptNet 5",
        "priority": 3,
        "license": "CC-BY-SA 4.0",
        "edges": ["CAUSES", "SYNONYMOUS", "ANTONYMOUS", "PART_OF"],
        "url": "https://conceptnet.io/",
        "status": "pending",
    },
    {
        "name": "Wiktionary (English)",
        "priority": 4,
        "license": "CC-BY-SA 3.0",
        "nodes": ["Morpheme", "WordForm"],
        "edges": ["COMPOSED_OF", "INFLECTS_TO"],
        "status": "pending",
    },
    {
        "name": "English Wikipedia (corpus statistics)",
        "priority": 5,
        "license": "CC-BY-SA",
        "edges": ["COLLOCATES"],
        "status": "pending",
    },
    {
        "name": "VerbNet",
        "priority": 6,
        "license": "Open",
        "edges": ["SELECTIONAL_PREFERENCE"],
        "status": "pending",
    },
]

# ---------------------------------------------------------------------------
# ID format: "language:{type}:{identifier}"
# ---------------------------------------------------------------------------


def make_node_id(node_type: str, identifier: str) -> str:
    return f"language:{node_type.lower()}:{identifier}"


def make_edge_id(source_id: str, edge_type: str, target_id: str) -> str:
    return f"{source_id}-[{edge_type}]->{target_id}"
