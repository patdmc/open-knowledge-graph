# Math Knowledge Graph

Two layers:
- **Knowledge graph**: notation, definitions, theorems, relationships between concepts
- **Action graph**: Python function calls that compute each concept (numpy, scipy, sympy, math)

Each node carries both. The knowledge layer says what it means. The action layer says how to compute it.

## Domains
- `notation/` — lexical symbols, operators, conventions
- `algebra/` — equations, polynomials, linear algebra, matrices
- `geometry/` — points, lines, shapes, transforms, coordinate systems
- `trigonometry/` — trig functions, identities, unit circle
- `calculus/` — limits, derivatives, integrals, series
- `python/` — action nodes (library calls) that implement knowledge nodes

## Edge types
- `DENOTES` — notation → concept (symbol means this)
- `EQUIVALENT_TO` — same concept, different notation
- `REQUIRES` — prerequisite knowledge
- `INVERSE_OF` — inverse operations
- `GENERALIZES` — concept A is a special case of B
- `COMPUTES` — action node that implements a knowledge node
- `COMPOSES` — f(g(x)) composition
- `BELONGS_TO` — concept → domain
