"""
Working memory for candidate language graph edges.

Candidate edges discovered from problem walkthroughs live here
until validated. An edge that travels (solves many problems) gets
promoted to the language graph. An edge that only solves one
problem is a heuristic, not an axiom — discard or refine.

Each candidate decomposes into up to three real edges:
  - language_cluster: words that trigger it (lang:clusters:*)
  - math_concept: the real operation it denotes (math:arithmetic:*)
  - unit_nodes: unit conversions involved (math:units:*)

The working memory candidate is the BRIDGE — it says "these words
mean this math." Once validated, the bridge becomes a DENOTES edge
in the language graph pointing to a node in the math graph.
"""

from dataclasses import dataclass, field


@dataclass
class CandidateEdge:
    """A candidate edge awaiting validation."""
    name: str
    pattern: str          # the language pattern, e.g. "N times as many A as B"
    interpretation: str   # what it means, e.g. "A = N × B"
    # --- Graph references (where this edge lives once promoted) ---
    language_cluster: str = ""   # lang:clusters:arithmetic/<cluster_id>
    math_concept: str = ""       # math:arithmetic:derived_operations/<concept_id>
    unit_node: str = ""          # math:units:<unit_file>
    # --- Validation ---
    examples: list[str] = field(default_factory=list)
    coverage: int = 0
    validated: bool = False


# =====================================================================
# Candidate edges from walkthrough sessions
#
# PROMOTED: edges that now exist as graph nodes in:
#   knowledge-graph/nodes/language/L01-word-clusters.yaml
#   knowledge-graph/nodes/math/arithmetic/AR01-derived-operations.yaml
#   knowledge-graph/nodes/math/units/U01-time.yaml
#   knowledge-graph/nodes/math/units/U02-length.yaml
#   knowledge-graph/nodes/math/units/U03-volume-mass.yaml
# =====================================================================

CANDIDATES: list[CandidateEdge] = [

    # --- Comparative relationships ---

    CandidateEdge(
        name="n_times_as_many",
        pattern="N times as many A as B",
        interpretation="A = N * B",
        language_cluster="comparative_times_words",
        math_concept="N01-arithmetic-operators/multiplication",
        examples=["#32 salt/flour", "#33 Ken/Johnson", "#48 Alan/Ben"],
        validated=True,
    ),

    CandidateEdge(
        name="n_more_than",
        pattern="N more A than B",
        interpretation="A = B + N",
        language_cluster="comparative_more_words",
        math_concept="N01-arithmetic-operators/addition",
        examples=["#20 stamps", "#47 Sam/Carlos"],
        validated=True,
    ),

    CandidateEdge(
        name="n_fewer_than",
        pattern="N fewer/less A than B",
        interpretation="A = B - N",
        language_cluster="comparative_less_words",
        math_concept="N01-arithmetic-operators/subtraction",
        examples=["#20 stamps", "#69 Tim/Martha"],
        validated=True,
    ),

    # --- Complement / remainder ---

    CandidateEdge(
        name="complement",
        pattern="the rest / remaining / left / not X",
        interpretation="unknown = total - known_parts",
        language_cluster="remainder_words",
        math_concept="complement",
        examples=["#34 B_and_above", "#43 coursework", "#27 files", "#99 stationery"],
        validated=True,
    ),

    CandidateEdge(
        name="percent_complement",
        pattern="N% are X, how many are not X",
        interpretation="not_X = total * (1 - N/100)",
        language_cluster="remainder_words",
        math_concept="percent_of",
        examples=["#34 grades", "#54 roses"],
        validated=True,
    ),

    # --- Rates and dimensional chains ---

    CandidateEdge(
        name="per_unit_rate",
        pattern="N A per B / each B / every B",
        interpretation="rate = N (unit_A / unit_B)",
        language_cluster="rate_words",
        math_concept="rate",
        examples=["#21 beetles/bird", "#38 candy/house", "#1 dollars/hour"],
        validated=True,
    ),

    CandidateEdge(
        name="dimensional_chain",
        pattern="rate_1 * rate_2 * ... until target unit",
        interpretation="multiply rates, units cancel like fractions",
        language_cluster="",  # pure math — no words trigger this
        math_concept="dimensional_chain",
        examples=["#21 jaguar->snake->bird->beetle", "#19 miles/trip * trips/week"],
        validated=True,
    ),

    # --- Round trip / multipliers ---

    CandidateEdge(
        name="round_trip",
        pattern="to and from / back and forth",
        interpretation="* 2 (one way becomes round trip)",
        language_cluster="round_trip_words",
        math_concept="round_trip",
        examples=["#18 Roque commute", "#19 Tim bike"],
        validated=True,
    ),

    CandidateEdge(
        name="frequency",
        pattern="N times a week/day/month",
        interpretation="* N (frequency multiplier)",
        language_cluster="frequency_words",
        math_concept="frequency_multiplier",
        examples=["#18 walks 3 times", "#45 six days a week"],
        validated=True,
    ),

    # --- Fractions of references ---

    CandidateEdge(
        name="fraction_of_reference",
        pattern="a third/half/quarter of what X verb",
        interpretation="result = X * fraction",
        language_cluster="fraction_words",
        math_concept="N01-arithmetic-operators/multiplication",
        examples=["#48 Ben=Laurie/3", "#35 Tommy=Lisa/2"],
        validated=True,
    ),

    CandidateEdge(
        name="sequential_fraction",
        pattern="spent fraction_1, then fraction_2 of the rest",
        interpretation="each fraction applies to remainder, not original",
        language_cluster="sequential_remainder_words",
        math_concept="sequential_fraction",
        examples=["#53 Leah: 1/7 then half the rest", "#56 Liza: 1/2 then 1/5 then 1/3 remaining"],
        validated=True,
    ),

    # --- Solve for missing ---

    CandidateEdge(
        name="solve_for_unknown",
        pattern="known + x = total / total - known = x",
        interpretation="unknown = total - sum(known_parts)",
        language_cluster="",  # pure math — structure triggers this
        math_concept="solve_for_unknown",
        examples=["#23 Ann's tops", "#31 car tunnel", "#51 pencils"],
        validated=True,
    ),

    # --- Partitioning ---

    CandidateEdge(
        name="threshold_partition",
        pattern="first N at rate_1, rest/thereafter at rate_2",
        interpretation="split at threshold, different rules per segment, sum results",
        language_cluster="partition_words",
        math_concept="threshold_partition",
        examples=["#39 ticket discount >10", "#46 download speed", "#25 Ralph first 100"],
        validated=True,
    ),

    # --- Geometric recurrence ---

    CandidateEdge(
        name="geometric_recurrence",
        pattern="each next has N times the previous",
        interpretation="sequence: a, a*r, a*r^2, ...; sum = a*(r^n-1)/(r-1)",
        language_cluster="",  # pure math — structure triggers this
        math_concept="geometric_sequence",
        examples=["#10 monster ships", "#42 earthquake buildings"],
        validated=True,
    ),

    # --- Unit conversions (now graph nodes in math:units:*) ---

    CandidateEdge(
        name="time_conversion",
        pattern="minutes<->hours, hours<->days, days<->weeks",
        interpretation="60 min=1hr, 24hr=1day, 7day=1wk, 52wk=1yr, 365day=1yr",
        language_cluster="",
        math_concept="",
        unit_node="math:units:time",
        examples=["#1 Weng min->hr", "#45 weeks->month", "#55 Leo 2hours=120min"],
        validated=True,
    ),

    CandidateEdge(
        name="length_conversion",
        pattern="feet<->inches, liters<->ml",
        interpretation="1ft=12in, 1L=1000mL, 1kg=1000g",
        language_cluster="",
        math_concept="",
        unit_node="math:units:length",
        examples=["#26 liters->ml", "#182 feet->inches"],
        validated=True,
    ),

    CandidateEdge(
        name="percent_to_fraction",
        pattern="N% -> N/100",
        interpretation="percent is parts per hundred",
        language_cluster="",
        math_concept="percent_of",
        examples=["#34 40%", "#43 30%/15%/25%"],
        validated=True,
    ),

    # --- Area ---

    CandidateEdge(
        name="area_rectangle",
        pattern="N by M (feet/inches/meters)",
        interpretation="area = N * M in square units",
        language_cluster="area_words",
        math_concept="N01-arithmetic-operators/multiplication",
        examples=["#50 fabric 4x6, 2x4, 16x12"],
        validated=True,
    ),

    # --- Ceiling division ---

    CandidateEdge(
        name="ceiling_division",
        pattern="how many containers/packs/groups of N for M items",
        interpretation="ceil(M / N) — can't buy fractional containers",
        language_cluster="container_words",
        math_concept="ceiling_division",
        examples=["#72 trail mix packs of 6", "#126 hotel rooms"],
        validated=True,
    ),

    # --- Ratio ---

    CandidateEdge(
        name="ratio",
        pattern="A:B = m:n / ratio of A to B is m:n",
        interpretation="A/B = m/n; if B=x then A=x*m/n",
        language_cluster="",
        math_concept="ratio",
        examples=["#16 Mike:Johnson 2:5", "#67 Elsa:Amalie 10:45"],
        validated=True,
    ),

    # --- Back-references ---

    CandidateEdge(
        name="pronoun_to_ratio",
        pattern="this same ratio / that rate / the same price",
        interpretation="back-reference to a previously established relationship",
        language_cluster="backreference_words",
        math_concept="",  # no fixed math concept — resolution is contextual
        examples=["#37 tea ratio", "#40 twice what Sara spent"],
        validated=True,
    ),

    CandidateEdge(
        name="ordinal_reference",
        pattern="the first/second subdivision, the year before",
        interpretation="ordinal maps to the Nth item in introduction order",
        language_cluster="ordinal_words",
        math_concept="",  # no fixed math concept — resolution is contextual
        examples=["#38 first/second subdivision", "#160 Tuesday/Wednesday/Thursday"],
        validated=True,
    ),

    # --- Compound references ---

    CandidateEdge(
        name="compound_reference",
        pattern="twice as many X as Y and Z combined",
        interpretation="X = 2 * (Y + Z)",
        language_cluster="comparative_times_words",
        math_concept="N01-arithmetic-operators/multiplication",
        examples=["#112 fish = 2*(cats+dogs)"],
        validated=True,
    ),

    # --- Percent more/less ---

    CandidateEdge(
        name="percent_more",
        pattern="N% more expensive / 50% more / increased by N%",
        interpretation="new = original * (1 + N/100)",
        language_cluster="percent_increase_words",
        math_concept="percent_change",
        examples=["#155 50% more expensive", "#63 20% less"],
        validated=True,
    ),

    CandidateEdge(
        name="percent_less",
        pattern="N% less / N% discount / N% off",
        interpretation="new = original * (1 - N/100)",
        language_cluster="percent_decrease_words",
        math_concept="percent_change",
        examples=["#63 20% less", "#148 45% discount", "#39 5% discount"],
        validated=True,
    ),

    # --- Counting/enumeration ---

    CandidateEdge(
        name="word_counting",
        pattern="how many letters in [word]",
        interpretation="count characters in the word",
        language_cluster="",  # meta-linguistic — no cluster
        math_concept="",      # no standard math concept
        examples=["#22 Grey=4 letters"],
        validated=True,
    ),

    CandidateEdge(
        name="dozen",
        pattern="a dozen / two dozen",
        interpretation="dozen = 12",
        language_cluster="dozen_words",
        math_concept="N01-arithmetic-operators/multiplication",
        examples=["#166 two dozen = 24"],
        validated=True,
    ),
]


def get_candidate(name: str) -> CandidateEdge | None:
    """Look up a candidate edge by name."""
    for c in CANDIDATES:
        if c.name == name:
            return c
    return None


def get_by_cluster(cluster_id: str) -> list[CandidateEdge]:
    """Find all candidates that use a given language cluster."""
    return [c for c in CANDIDATES if c.language_cluster == cluster_id]


def get_by_concept(concept_id: str) -> list[CandidateEdge]:
    """Find all candidates that map to a given math concept."""
    return [c for c in CANDIDATES if c.math_concept == concept_id]


def summarize() -> str:
    """Print summary of all candidate edges."""
    lines = []
    for c in CANDIDATES:
        status = "V" if c.validated else "?"
        cluster = c.language_cluster or "-"
        concept = c.math_concept or "-"
        lines.append(f"[{status}] {c.name}: {c.pattern}"
                     f"\n    lang: {cluster} -> math: {concept}"
                     f"  (examples: {len(c.examples)}, coverage: {c.coverage})")
    return "\n".join(lines)
