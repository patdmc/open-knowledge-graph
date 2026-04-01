# Contributing to the Open Knowledge Graph

You don't need to understand the whole graph. You need to add one honest node with honest edges.

## What you can contribute

| Type | What it is | Where it goes |
|------|-----------|---------------|
| **Knowledge node** | A domain with concepts, formulas, actions | `knowledge-graph/nodes/<domain>/<ID>.yaml` |
| **Proof** | A typed assertion chain proving a theorem | `knowledge-graph/nodes/math/proofs/PR<NN>-<name>.yaml` |
| **Shared lemma** | A reusable proof fragment used by multiple proofs | `knowledge-graph/nodes/math/proofs/shared/SL<NN>-<name>.yaml` |
| **Cross-domain edge** | A connection between existing nodes | Add to the `edges:` section of a knowledge node |

## Quick start: add a knowledge concept (10 minutes)

Find the right domain file. If your concept fits an existing domain, add it there. If not, create a new file.

```yaml
- id: your_concept_id          # lowercase_snake_case, unique across the graph
  name: "Human-Readable Name"
  latex: "\\text{the formula}"  # LaTeX notation
  meaning: "one sentence — what it IS, not how to compute it"
  action:
    numpy: "np.something(x)"   # optional — executable code
  curried: "lambda x: x"       # optional — composable callable
```

Rules:
- `id` must be unique. Search existing files first.
- `meaning` explains the concept to someone who can read the formula but doesn't know why it matters.
- `action` is optional but valuable — it makes the concept executable.
- Every concept you add becomes available to the synthesis engine. The more honest your edges, the better the synthesis.

## Quick start: add a proof (20 minutes)

Copy this template:

```yaml
id: PR<NN>-<short-name>
type: proof
domain: <primary domain>
name: "Theorem Name"
target: "<concept_id this proves>"
method: <inference type from PR00-proof-framework.yaml>
importance: "<one sentence — why this theorem matters>"

assertions:
  - step: 1
    id: hypotheses
    type: hypothesis
    statement: "Let ... be ..."
    justification: "given"
    references: [<concept_ids from knowledge nodes>]

  - step: 2
    id: step-name
    type: claim
    statement: "Then ..."
    justification: "by <reason>"
    depends_on: [hypotheses]
    references: [<concept_ids>]

  # ... more steps ...

  - step: N
    id: conclusion
    type: theorem
    statement: "Therefore ..."
    justification: "by steps above"
    depends_on: [<previous step ids>]

qed: true

proof_links:
  - target: PR<other>
    relation: USES | GENERALIZES | RELATED
    shared_concepts: [<what they share>]
```

### Assertion types

| Type | When to use |
|------|------------|
| `axiom` | Assumed true without proof |
| `definition` | Assigns meaning to a term |
| `hypothesis` | Assumed for this proof |
| `claim` | Intermediate step |
| `lemma` | Auxiliary result |
| `theorem` | The target — what you're proving |
| `contradiction` | False statement completing a contradiction proof |

### The three fields that matter most

1. **`depends_on`** — which earlier steps does this step use? This builds the DAG.
2. **`references`** — which concepts from the knowledge graph does this step invoke? This builds the cross-links.
3. **`justification`** — why does this step follow? Not a formal proof — a sentence a student can follow.

### If a proof step uses a fact that appears in other proofs

Don't inline it. Create a shared lemma:

```yaml
id: SL<NN>-<name>
type: shared_lemma
domain: <domain>
name: "Lemma Name"
statement: "<LaTeX>"
justification: "<why it's true — one sentence>"
references: [<concept_ids>]
used_by:
  - proof: PR<NN>-<name>
    step: <step-id>
```

Then reference it from your proof step's `justification` field. The synthesis engine uses shared lemmas to find bridges between proofs.

## Adding a new domain

Create a directory and a YAML file:

```
knowledge-graph/nodes/<your-domain>/
  <ID>-<name>.yaml
```

Use cross-domain edges to connect to existing nodes:

```yaml
cross_domain:
  - target: P01-probability-statistics
    relation: EXTENDS | REQUIRES | USES
    concepts: [<shared concept ids>]

edges:
  - from: <your-node-id>
    to: <existing-node-id>
    type: EXTENDS | REQUIRES | USES
    note: "why this connection exists"
```

Edge types:
- **REQUIRES** — your domain needs this to function (linear algebra REQUIRES arithmetic)
- **EXTENDS** — your domain generalizes this (information theory EXTENDS probability)
- **USES** — your domain references this without depending on it

## Quality checklist

Before submitting:

- [ ] Every `id` is unique (search the repo)
- [ ] Every `references` entry points to an existing concept `id`
- [ ] Every `depends_on` entry points to a step `id` earlier in the same proof
- [ ] `meaning` fields explain *what*, not *how*
- [ ] No orphan nodes — at least one edge connects to the existing graph
- [ ] LaTeX compiles (backslashes doubled in YAML strings)

## What happens after you contribute

The synthesis engine runs over the full graph and detects:
1. **Bridge concepts** — your concept appears in otherwise-disjoint proof clusters
2. **Shared lemma bridges** — your lemma connects proofs from different domains
3. **Unwalked edges** — your domain REQUIRES another, but no proof crosses both yet

Every node you add makes the graph hungrier. It creates new questions. Other students answer them.

## Philosophy

This is open infrastructure. The graph doesn't hallucinate — every edge is earned. Every assertion has a type, a justification, and dependencies. If you can't justify a step, don't add it. An honest gap is more valuable than a fake edge.

We are standing on the shoulders of giants. We will not charge others to stand here with us.
