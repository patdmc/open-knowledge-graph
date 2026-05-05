# Merge Notes: Papers 0-4 → Bounded Context Architecture

## Source files
- Paper 0: McCarthy2026_0_AllSoftwareIsAGraph.tex (489 lines)
- Paper 1: McCarthy2026_1_GraphNecessity.tex (1595 lines)
- Paper 2: McCarthy2026_2_Escalation.tex (1476 lines)
- Paper 3: McCarthy2026_3_Normalization.tex (1252 lines)
- Paper 4: McCarthy2026_4_Intelligence.tex (1505 lines)
- Total: 6,317 lines → target ~1,500-2,000 lines

## Target: McCarthy2026_BoundedContextArchitecture.tex

## What appears ONCE (in the Formal Framework section)

### Definitions (from multiple papers, deduplicated):
- World State Space (Paper 1, Def 1)
- Proposition and Knowledge Base (Paper 1, Def 2)
- Bounded Active Context C_n (Paper 1, Def 3)
- Growth and Curation Regimes (Paper 1, Def 4)
- Contextual Retrieval (Paper 1, Def 5)
- Factoring (Paper 1, Def 6)
- Information Preservation (Paper 1, Def 7)
- Encoding Hierarchy E(c,r) (Paper 2, Def 5 / Paper 4, Def 6)
- Delegation Overhead (Paper 2, Def 7)
- Escalation Policy (Paper 2, Def 8)
- Action Space and Entity (Paper 3, Def 4-5)
- Context Load (Paper 3, Def 8)
- Agency γ (Paper 4, Def 2)
- Action Feedback (Paper 4, Def 3)
- Information Gain (Paper 4, Def 4)
- Novelty Rate (Paper 4, Def 5)
- Free Context (Paper 4, Def 7)
- Higher-Order Function Space M (Paper 4, Def 8)
- M Efficiency η_M (Paper 4, Def 9)
- Intelligence I(E) = γ · η_M (Paper 4, Def 10)
- Persistence (Paper 4, Def 11)

### Remark on Selective Necessity (Paper 1, Remark 1)
Appears once in the framework section. All subsequent sections reference it.

## Section mapping

### Section 3: Graph Structure (from Paper 1)
KEEP:
- Proposition 1 (Graph Optimality from Info Preservation) with 5-option argument
- Theorem 1 (Graph Structure from Retention Dynamics) with O(n) lower bound
- Corollary: Edge Growth Dominates Node Growth
- Theorem: Contextual Retrieval Requires Adjacency Computation
- Physarum example (3 paragraphs)
- Catastrophic forgetting as prediction confirmed

CUT:
- Related work subsection (moves to unified Related Work)
- Forward references to Papers 2-4
- The "On limits of exhaustiveness" remark (good honesty, keep condensed)
- The full attention-as-graph discussion (keep the cost comparison, cut the detailed proof)

### Section 4: Escalation (from Paper 2)
KEEP:
- Theorem 2 (Convergence to Escalation at Scale) with crossover proof
- The crossover threshold k·α ≥ C_n/2
- One-directionality argument (escalation never converges to top-down)
- Circuit breaker pattern (condensed)
- Cell division illustration (condensed to 1 paragraph)

CUT:
- Full hierarchical top-down objection section (keep 1 paragraph)
- The biological level table (keep as example, not as claim)
- The detailed multi-page proof of φ → 1 (keep the result, condense the proof)
- Forward/backward references to other papers

### Section 5: Normalization (from Paper 3)
KEEP:
- Theorem (Normalization of Shared Propositions) with O(n·k) → O(n+k') proof
- Retention Criterion corollary
- K/A Inseparability corollary
- The Codd analogy (the precipitation metaphor)
- Catastrophic forgetting as empirical signature

CUT:
- The full neural network remark (condensed — point to catastrophic forgetting)
- The CP vs AP cost comparison (keep 1 paragraph)
- Redundant definitions

### Section 6: Persistence and Intelligence (from Paper 4)
KEEP:
- Theorem (Evaluation-Driven State Revision) — condensed, keep Parts I and II
- Encoding Permanence θ_i = 1 - ε/C_i (the cleanest result)
- Learning rate hierarchy derivation (1 page)
- Theorem (A/M Stratification) — condensed
- Theorem (Structural Invariants) with free-context exhaustion argument
- Corollary: I(E) = γ · η_M
- "Necessary but not sufficient" acknowledgment

CUT:
- The full Lyapunov analysis (keep the result, cite the mechanism)
- The detailed curation regime interaction (keep 1 paragraph)
- The measurement remark (keep the architectural class comparison, cut details)

### Section 7: Empirical (NEW — brief summary of Papers 5-7)
WRITE FRESH:
- Paper 5 result: channel count predicts survival, mutation count doesn't
- Paper 6 result: channels map to architectural tiers
- Paper 7 result: graph position predicts treatment response
- Evolutionary data: three epochs, top-down channels scaled with complexity
- ~2 pages

### Introduction (WRITE FRESH, drawing from Papers 0 and 1)
From Paper 0:
- The 11-item software graph enumeration (condensed to a list)
- "Software is a graph the same way water is wet"
- The speed-run argument (1 sentence)

From Paper 1:
- The two strategies and their crossover
- Physarum
- "The industry is slime-molding"
- The bounded context constraint statement

New:
- The unification claim: one constraint, four consequences
- The empirical hook: cancer genomics confirms the architecture

## What gets CUT entirely
- All "this is the Nth paper in a series" paragraphs (5 instances)
- All "Paper X established that..." cross-reference paragraphs (~15 instances)
- All redundant definition blocks (~30% of total content)
- The full "WTF" table from Paper 0 (keep the spirit, cut the table)
- Detailed cancer-channel opening paragraphs (appear in 4 of 5 papers)
- Redundant related work across all papers
