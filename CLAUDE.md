# Open Knowledge Graph

## What This Is

Open Knowledge Graph (OKG) is a framework for discovering, defining, and refining skills
through a graph-based learning system grounded in collective knowledge.

**A skill is an informed action** — an action grounded in evidence, validated against the
world, and attributed to its origins. Not a procedure or a recipe. An action that knows
why it works.

This definition derives directly from the theory of intelligence as the driven bounding of
uncertainty with knowledge-grounded action. A skill is the atomic unit of that: one
knowledge-grounded action. The skill graph is the complete set of informed actions available
to an entity.

A skill is not:
- An uninformed action (reflex, guess, behavior without evidence)
- Informed non-action (knowledge with no action potential — that is data, not a skill)

A skill is specifically the coupling: knowledge that enables action, action justified by knowledge.

## The Graph

Skills are nodes. Typed relations between skills are edges. The graph represents the complete
informed action space of the system — what it knows how to do, and how those actions relate.

**Edge types** (derived from the formal model):

| Relation | Meaning |
|---|---|
| `depends_on` | This skill requires another to have acted first |
| `recommends_before` | Evidence shows ordering reduces errors — soft, confidence-scored |
| `recommends_after` | Evidence shows this skill performs better following another |
| `enhanced_by` | Combined execution outperforms either skill independently |

The higher-order function space operates on the skill graph itself — discovering new skills,
refining existing ones, pruning those that fail to reduce uncertainty. These are also informed
actions, but their domain is the skill graph rather than the world. Same definition. Different world.

## What Grounds a Skill

Every skill declaration must carry:

- **Evidence** — real-world validation with attribution and provenance
- **Uncertainty region** — the domain in W where this skill reduces uncertainty
- **Activation conditions** — what signals in the world indicate this skill should act
- **Success criteria** — testable definition of what it means for uncertainty to be reduced
- **Relations** — how this skill connects to others in the graph

Confidence is never hand-authored. It is derived from inference on ranked sample sets drawn
from the equivalency class — commits/changes discovered via the skill's `discovery` parameters
— and validated against a held-back sample by comparing agent behavior to human ground truth.

## Skill Declaration Structure

Every skill is declared in a `SKILL.md` file with YAML frontmatter:

```yaml
---
name: skill-name
version: 1.0.0
type: skill          # skill | higher-order
description: One informed action, clearly stated
license: MIT

authors:
  - name: Author Name
    role: author
reviewers:
  - name: Reviewer Name
    role: reviewer

tags: []              # for discovery

capabilities:
  supports_dry_run: true | false
  supports_resume: true | false
  execution_model: deterministic | agentic | hybrid

triggers:
  - condition: "observable condition in W"
    intent: ["natural language expressions of this need"]
    priority: critical | high | medium | low
    reason: "why this condition calls for this action"
    confidence: ~    # runtime-derived, never hand-authored

graph:
  depends_on:
    - skill: skill-name
      required: true   # hard constraint — no confidence scored
  recommends_before:
    - skill: skill-name
      confidence: ~
      reason: ""
  recommends_after:
    - skill: skill-name
      confidence: ~
      reason: ""
  enhanced_by:
    - skills: [skill-a, skill-b]
      confidence: ~
      reason: ""

follows_patterns:
  - name: pattern-name
    reason: "why this skill needs this pattern"

context:
  mode: hybrid         # isolated | continuous | hybrid
  boundaries:
    repo_change: true
    time_gap_seconds: 1800
    domain_change: true
    explicit_keywords: ["new task", "switch to", "start fresh"]
  preserve:
    - auto_memory: true
    - skill_references: true
    - user_last_n_messages: 3
  clear:
    - previous_task_context: true
    - intermediate_results: true
    - cached_searches: true

success_criteria:
  - check: "testable condition"
    required: true | false

evidence_base:
  discovery:
    commit_message: []
    file_patterns: []
    diff_patterns: []
    label_patterns: []
    min_match_score: 0.7
  sample:
    strategy: hierarchical-ranked
    as_of: "YYYY-MM-DD"
    total: ~
    training: ~
    validation: ~
    ranked_breakdown:
      pattern_setters: ~
      high_quality: ~
      breadth: ~
      depth: ~
      low_quality: ~
  validation:
    last_run: ~
    run_by: ~
    samples_evaluated: ~
  patterns: []
  references: []

integrations:
  - name: tool-name
    type: cli | runtime | ci
    required: true | false
    version: ""
    check: ""

allowed_tools: [Bash, Read, Write, Edit, Grep, Glob]
---
```

## Authoring Rules

1. **Read `SKILL_BEST_PRACTICES.md` first** — identify which patterns apply before writing
2. **`follows_patterns` is progressive disclosure** — only declare what this skill actually needs
3. **Confidence is never hand-authored** — leave as `~` until the learning loop runs
4. **Evidence must have attribution** — no unattributed claims in `evidence_base`
5. **Success criteria must be testable** — if you cannot verify it, it is not a criterion

## Key Reference Files

- `SKILL_BEST_PRACTICES.md` — patterns available for `follows_patterns` declarations
- `THEORY.md` — the formal theory of intelligence this framework is grounded in

## Constraints

- Do not create files unless necessary — prefer editing existing skill declarations
- Do not add features beyond what was asked
- Always read existing skill files before suggesting modifications
