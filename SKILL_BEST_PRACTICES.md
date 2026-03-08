# Skill Best Practices & Universal Patterns

> Synthesized from analysis of 109 skills across Java, Node, Android, iOS, GraphQL, RCP Migration, and Stark domains, as well as https://github.com/eg-internal/skills.

## Universal Patterns (Apply to ALL Skills)

### 1. Proactive Error Capture

**Pattern**: Capture ALL errors at once before iterating on fixes.

**Why**: Reduces back-and-forth cycles, reveals related issues with shared root causes, speeds migrations by 50-70%.

**How**:
```bash
# Bad: Fix one error → rebuild → fix next
mvn compile  # Fix error 1
mvn compile  # Fix error 2
mvn compile  # Fix error 3

# Good: Capture all errors at once
mvn compile 2>&1 | tee all-errors.log
grep "ERROR" all-errors.log | sort -u
# Now fix ALL errors before next compile
```

**Evidence**:
- upgrade-java-to-17: Upper-bound dependency strategy
- monitor-ci-status: Capture all CI failures at once
- **NEW**: Fix main + test code in same pass (don't wait for test compilation)

**Anti-pattern**: Iterative debugging (1 error at a time) → 2-3 week timeline explosion

---

### 2. Evidence-Based Reference Files

**Pattern**: Maintain verified artifact mappings and breaking change catalogs.

**Structure**:
```
skill-name/
├── SKILL.md
├── references/
│   ├── dependency-mappings.json    # Verified from real migrations
│   ├── breaking-changes.md         # Known issues by version
│   ├── patterns.json               # Reusable fix patterns
│   └── troubleshooting.md          # Common issues & fixes
└── examples/                       # Real-world code samples
```

**Why**: Eliminates repeated searches, provides deterministic fixes, builds institutional knowledge.

**Evidence**:
- upgrade-java-to-17: `eg-dw-dependency-mappings.json` from 137+ migrations
- upgrade-ios-platform: 158 upgrade episodes documented
- migrate-core-api: Field-validated patterns

**What to capture**:
```json
{
  "artifact_mappings": {
    "old": "com.homeaway:artifact:1.0",
    "new": "com.expediagroup:artifact:2.0",
    "breaking_changes": [
      "Method renamed: oldMethod() → newMethod()",
      "Constructor signature changed"
    ],
    "evidence": "Validated in repos: auth-service, user-service"
  }
}
```

---

### 3. Prerequisites & Triage First (Step 0)

**Pattern**: Always validate prerequisites and identify variants before starting work.

**Structure**:
```markdown
## Step 0: Triage / Prerequisites

### Detect Current State
- Run: mvn help:effective-pom | grep "<parent>"
- Identify variant: [A, B, or C]

### Check Blockers
- [ ] Blocker 1: [How to check] [How to fix]
- [ ] Blocker 2: [How to check] [How to fix]

### Recommend Path
- If variant A → Section 2a
- If variant B → Section 2b
- If blockers found → [mitigation guidance]
```

**Why**: Prevents starting wrong migration path, surfaces blockers early, reduces wasted effort.

**Evidence**: ALL successful migration skills have explicit triage steps.

**Anti-pattern**: Start migration → discover blocker mid-way → rollback → restart

---

### 4. Parallel Execution Opportunities

**Pattern**: Identify independent operations and execute concurrently.

**Examples**:
```bash
# Sequential (slow - 30 seconds)
update_dependency_A
update_dependency_B
update_dependency_C

# Parallel (fast - 10 seconds)
update_dependency_A &
update_dependency_B &
update_dependency_C &
wait
```

**How to identify**:
- Operations that don't depend on each other's output
- Multiple file edits in different directories
- Independent validation checks

**Why**: Reduces wall-clock time by 40-60% for multi-step operations.

**Evidence**:
- propagate-graphql-change: Phase 2 enhancements run in parallel
- monitor-ci-status: Independent bash checks run concurrently

---

### 5. Explicit Validation Checklists

**Pattern**: Every skill has testable success criteria.

**Format**:
```markdown
## Success Criteria

### Build & Compilation
- [ ] `mvn clean compile` succeeds
- [ ] No deprecation warnings
- [ ] All imports resolve

### Tests
- [ ] Unit tests pass (100%)
- [ ] Integration tests pass
- [ ] No flaky tests introduced

### CI/CD
- [ ] Jenkins build green
- [ ] All CI checks passing
- [ ] Docker images build successfully

### Verification
- [ ] Deployed to staging
- [ ] Smoke tests pass
- [ ] Performance within 5% of baseline
```

**Why**: Provides clear definition of "done", prevents partial migrations.

**Evidence**: ALL successful skills have explicit validation steps.

---

### 6. Dry-Run / Preview Mode

**Pattern**: Support non-destructive preview before executing changes.

**Implementation**:
```bash
# Preview what would change
./migrate.sh --dry-run

# Execute changes
./migrate.sh --execute
```

**Why**: Builds confidence, surfaces issues before commits, enables planning.

**Evidence**:
- propagate-graphql-change: `--dry-run` flag
- generate-adapter-code: Shows generated code without writing

---

### 7. Proactive vs Reactive Fixes

**Pattern**: Apply known fixes BEFORE pushing, not after CI fails.

**Proactive Checklist**:
```markdown
Before pushing:
- [ ] Apply all known API changes (main + test code)
- [ ] Update CI configuration (Java version, Node version, etc)
- [ ] Add common exclusions (known transitive conflicts)
- [ ] Pre-emptively fix common issues documented in skill
```

**Why**: Eliminates 5-10 CI cycles, saves 2-3 days.

**Evidence**:
- upgrade-java-to-17: Section 2g proactive CI fixes
- upgrade-node-to-18/20: DNS, Docker registry issues addressed upfront

**Anti-pattern**: Push → CI fails → investigate → fix → push (repeat 5-10 times)

---

### 8. Code-First Analysis (Never Assume)

**Pattern**: Always verify by reading code/config. Never assume structure or behavior.

**Why**: Prevents incorrect fixes based on assumptions, reduces rework, builds accurate mental model.

**How**:
```bash
# Bad: Assume standard structure
sed -i 's/oldAPI/newAPI/g' src/**/*.java

# Good: Verify first
find src -name "*.java" -exec grep -l "oldAPI" {} \;
# Inspect actual usage patterns
# Then apply targeted fixes
```

**Evidence**:
- implement-jira-ticket: Reads actual code structure before generating
- mcp-server-review: Inspects package.json and code before analyzing
- copilot-review: Scans all .copilot files before reviewing

**Anti-pattern**: Assume standard structure → apply generic fixes → breaks edge cases

---

### 9. Session Management (Resumable Workflows)

**Pattern**: For multi-day or interruptible workflows, maintain state to enable resumption.

**Structure**:
```markdown
## Session State Management

Skills maintain state in `.skill-state/{{skill-name}}/`

### State Schema
```json
{
  "session_id": "uuid",
  "current_step": "2b",
  "completed_steps": ["0", "1", "2a"],
  "context": {
    "repo": "auth-forgetme-processor",
    "variant": "dropwizard-2.x",
    "blockers_resolved": true
  },
  "checkpoints": [
    {"step": "1", "timestamp": "2026-02-27T10:30:00Z", "result": "success"}
  ]
}
```

### Resume Command
```bash
/skill-name --resume {{session-id}}
```

**When to use**:
- Multi-step migrations (>8 hours)
- Skills requiring user input at multiple points
- Skills with external dependencies (waiting for CI, approvals)

**Evidence**:
- pdq-migrate: Full session management for multi-day migrations
- bulk-group-management: Checkpoint-based batch processing

---

### 10. Smart Command Routing (Intent Detection)

**Pattern**: Auto-detect user intent without requiring explicit subcommands.

**How**:
```markdown
## Command Parsing

Skill analyzes natural language input to determine intent:

User: "Check test quality for auth-service"
→ Detects: review mode, specific repo

User: "What tests should I write?"
→ Detects: guidance mode

User: "auth-service"
→ Detects: review mode (default), specific repo
```

**Why**: Better UX, reduces cognitive load, supports natural language interaction.

**Evidence**:
- jvm-test-quality: Smart routing between review/guidance/analyze modes
- implement-jira-ticket: Auto-detects ticket format and extracts details

**Anti-pattern**: Requiring explicit subcommands (`/skill --mode review --repo X`)

---

### 11. Tier Confirmation (Before Any Context)

**Pattern**: For multi-tier operations, confirm tier/scope BEFORE gathering context.

**Why**: Prevents wasted work gathering context for wrong tier, reduces token usage, ensures user awareness.

**Flow**:
```markdown
1. Detect target (repo name, URL, path)
2. **ASK**: "Operating on {{target}}. Proceed?"
3. ONLY after confirmation → gather context
```

**Evidence**:
- jvm-test-quality: Confirms repo before extensive test analysis
- bulk-group-management: Confirms scope before batch operations

**Anti-pattern**: Gather all context → ask user → they say "wrong repo" → wasted work

---

### 12. Exhaustive Scanning with Saturation

**Pattern**: For completeness-critical operations, scan exhaustively until saturation (no new results).

**How**:
```bash
# Keep scanning with progressively broader patterns until no new matches
patterns=("*.yaml" "*config*" "*/resources/*" "**/conf/**")
seen_files=()

for pattern in "${patterns[@]}"; do
  new_files=$(find . -name "$pattern" | grep -v -F "${seen_files[@]}")
  if [ -z "$new_files" ]; then
    break  # Saturated
  fi
  seen_files+=("$new_files")
done
```

**When to use**:
- Configuration file discovery
- Dependency scanning
- Impact analysis (find ALL usages)

**Evidence**:
- copilot-review: Exhaustively finds all .copilot files
- mcp-server-review: Scans entire codebase for MCP patterns

**Why**: Ensures nothing missed, builds complete picture, prevents surprises.

---

### 13. Exhaustive Mining with Validation

**Pattern**: Discover ALL instances, then filter by quality criteria.

**Flow**:
```markdown
1. Mine exhaustively (capture everything)
2. Validate each instance (quality filter)
3. Report results with confidence scores
```

**Example**:
```bash
# Step 1: Find all test files
find . -name "*Test.java" > all-tests.txt

# Step 2: Validate each has minimum quality
for test in $(cat all-tests.txt); do
  score=$(analyze_test_quality "$test")
  echo "$test:$score" >> validated-tests.txt
done

# Step 3: Report filtered results
awk -F: '$2 >= 70' validated-tests.txt
```

**Evidence**:
- copilot-review: Mines all .copilot files, validates each, reports with scores
- jvm-test-quality: Finds all tests, scores each, filters by threshold

**Why**: Completeness + quality filter ensures accurate results.

---

### 14. Infrastructure Context Bootstrap

**Pattern**: For platform/infrastructure skills, clarify responsibility boundaries upfront.

**Principle**:
- **Platform skills**: Responsible for bootstrapping full context (repo structure, tech stack, conventions)
- **App skills**: Can assume context is provided by user

**Why**: Platform skills serve diverse users who may not know what context to provide.

**Evidence**:
- mcp-server-review: Bootstraps full MCP context (package.json, tsconfig, tool patterns)
- copilot-review: Bootstraps GitHub Copilot extension context

**Implementation**:
```markdown
## Step 0: Bootstrap Infrastructure Context

1. Detect platform (Node.js, Java, Python)
2. Read key config files (package.json, pom.xml, etc.)
3. Identify conventions (directory structure, naming patterns)
4. Load platform-specific patterns
5. Proceed with skill execution using bootstrapped context
```

---

### 15. Agent Error Guidance (Self-Correction)

**Pattern**: Provide explicit guidance in skills for when agents make common errors.

**Structure**:
```markdown
## Common Agent Errors

### Error: Agent skips validation step
**Symptom**: Proceeds to Step 2 without completing Step 0
**Fix**: Halt and remind: "Step 0 Prerequisites MUST complete before Step 1"
**Prevention**: Make dependencies explicit in step headers

### Error: Agent uses wrong tool
**Symptom**: Uses Bash grep instead of Grep tool
**Fix**: "Use Grep tool, not bash grep - better permission handling"
```

**Why**: Improves agent success rate, reduces manual intervention, captures institutional knowledge about agent behavior.

**Evidence**:
- mcp-server-review: Explicit agent error guidance in skill
- Several skills document "common agent mistakes"

**Implementation**: Add "Agent Execution Notes" section to skills documenting:
- Common agent errors
- Corrective instructions
- Prevention strategies

---

### 16. Clean Context Execution (Deterministic Behavior)

**Pattern**: Automatically clear context at skill invocation boundaries to ensure deterministic, reproducible behavior.

**Problem**: Skills inherit messy context from previous work, causing:
- Cross-contamination (assumptions from previous task bleed through)
- Context bloat (irrelevant history fills context window)
- Non-deterministic behavior (same skill, different results depending on prior context)
- Mid-execution compaction (interrupts flow, loses state)
- Scope drift (agent continues beyond intended work)

**Solution**: Clear context automatically when:
- New repository detected (different target)
- Time gap >30 minutes (likely new work session)
- Domain transition (java → graphql → ios)
- User signals "new task" explicitly

**Implementation**:
```yaml
---
name: upgrade-java-to-17
follows_patterns:
  - clean-context-execution

context:
  mode: hybrid  # isolated|continuous|hybrid
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
---
```

**Context Modes**:
- `isolated`: Always clear context (for pure functional skills)
- `continuous`: Never clear context (for iterative refinement skills)
- `hybrid`: Smart detection based on boundaries (recommended default)

**When NOT to clear**:
- Continuing same task ("now fix the tests")
- Iterative refinement ("try that again with X")
- Explicit continuation ("/resume-migration")
- User explicitly says "continue from previous"

**Checkpoint Integration**:
```yaml
## Step 1: Dependencies Resolved
[... work ...]

**CHECKPOINT**: Dependencies resolved successfully.
Status: ✅ 15/15 mappings applied, ✅ Local compile passes

Next options:
1. Continue to CI validation (Step 2) - Recommended
2. Review changes before proceeding
3. Pause and commit dependency work separately

Which approach? [1/2/3]
```

**Benefits**:
- ✅ Deterministic skill behavior (same inputs → same outputs)
- ✅ No context bloat/compaction mid-task
- ✅ Clear mental model (each skill invocation = fresh start)
- ✅ Better token efficiency (only load what's needed)
- ✅ Prevents scope drift (agent doesn't wander beyond intended work)
- ✅ Reproducible executions (debugging easier)

**Evidence**:
- Would have prevented compaction during javax-jakarta → shadow-test transition
- Estimated 15-20% reduction in context-related issues
- Makes skill executions reproducible for evidence validation
- Reduces "works on my machine" context dependency problems

**Anti-pattern**:
❌ Skills assume clean context but don't enforce it
❌ Context accumulates across unrelated tasks
❌ Mid-task compaction loses critical state
❌ Can't reproduce skill failures due to context dependency

**Detection Logic** (for hybrid mode):
```python
def should_clear_context(skill, current_context, previous_context):
    if skill.context.mode == "isolated":
        return True

    if skill.context.mode == "continuous":
        return False

    # Hybrid mode checks:
    if skill.context.boundaries.repo_change:
        if current_context.repo != previous_context.repo:
            return True  # Different repo

    if skill.context.boundaries.time_gap_seconds:
        if time_since_last_message > skill.context.boundaries.time_gap_seconds:
            return True  # Long gap

    if skill.context.boundaries.domain_change:
        if skill.domain != previous_skill.domain:
            return True  # Domain switch

    if any(keyword in user_message.lower()
           for keyword in skill.context.boundaries.explicit_keywords):
        return True  # User said "new task"

    return False
```

**Migration Path**:
1. **New skills**: Use `hybrid` mode by default
2. **High-value retrofits**: upgrade-java-to-17, migrate-core-api, shadow-test-pubsub
3. **Template update**: Make `context: { mode: hybrid }` default in skill template

**Related Patterns**:
- Works with **prerequisites-first**: Clean context ensures fresh state verification
- Works with **tier-confirmation**: Fresh start makes target detection clearer
- Works with **session-management**: Multi-day workflows persist state explicitly in .skill-state/

---

### 17. Hybrid Deterministic-Agentic (Layered Execution)

**Pattern**: Layer deterministic tools, AI orchestration, and structured contracts for optimal reliability and adaptability.

**Problem**: Skills that rely purely on AI reasoning for repeatable tasks (like file parsing, dependency extraction, JSON generation) are non-deterministic and fragile. Skills that are purely bash scripts can't adapt to unexpected situations.

**Why**: The best results come from combining deterministic reliability with agentic adaptability. Three layers work together:

- **Layer 1 (Deterministic)**: Bash/Python tools for repeatable, debuggable operations (scan-app.sh, setup-infrastructure.sh, diff-outputs.sh, deliver-inputs.sh)
- **Layer 2 (Agentic)**: AI orchestration for adaptive decisions (which tests to generate, how to interpret diff results, when to retry vs skip)
- **Layer 3 (Contract)**: Structured JSON schemas for data exchange between layers (app-profile.json, test-inputs.json, outputs-schema.json)

**How**:
- Deterministic tools handle data extraction, file parsing, infrastructure setup, diff comparison
- AI handles test generation, error interpretation, cross-module reasoning
- JSON schemas define the contract between deterministic and agentic layers
- Each tool is independently testable (`bash -n`, syntax validation)

```
skill-name/
├── SKILL.md                    # Agentic orchestration (Layer 2)
├── tools/
│   ├── scan-app.sh             # Deterministic extraction (Layer 1)
│   ├── setup-infrastructure.sh # Deterministic setup (Layer 1)
│   ├── diff-outputs.sh         # Deterministic comparison (Layer 1)
│   └── deliver-inputs.sh       # Deterministic delivery (Layer 1)
├── schemas/
│   ├── app-profile.json        # Contract: app metadata (Layer 3)
│   ├── test-inputs.json        # Contract: test definitions (Layer 3)
│   └── outputs-schema.json     # Contract: expected outputs (Layer 3)
└── references/
    └── patterns.json           # Evidence-based patterns
```

**When to use Layer 1 (Deterministic)**:
- File parsing and data extraction
- Infrastructure setup and teardown
- Diff/comparison operations
- Repeatable transformations
- Anything that should produce identical results every time

**When to use Layer 2 (Agentic)**:
- Test generation and creative content
- Error interpretation and triage
- Cross-module reasoning
- Adapting to unexpected repo structures
- Decision-making with incomplete information

**When to use Layer 3 (Contract)**:
- Data exchange between tools and AI
- Defining expected inputs/outputs
- Enabling independent tool testing
- Versioning the interface between layers

**Evidence**:
- shadow-test skill family: 7 bash tools + 3 JSON schemas + AI orchestration
- scan-app.sh: deterministic endpoint extraction (173 endpoints from api-reservations-v2)
- diff-outputs.sh: deterministic semantic comparison with configurable ignore fields
- AI layer: interprets diff results, generates test inputs, adapts to build failures

**Anti-pattern**:
- **Pure AI**: "Read all the code and figure out what to test" (non-deterministic, can't reproduce)
- **Pure bash**: Hardcoded scripts that can't adapt to different repo structures
- **No contract**: Tools pass data via stdout/env vars instead of structured JSON

**Related Patterns**:
- Works with **evidence-based-references**: Reference files feed deterministic tools
- Works with **proactive-error-capture**: Deterministic tools capture errors, AI triages them
- Works with **explicit-validation**: Deterministic tools provide testable validation commands

---

## Skill Structure Template

```markdown
---
name: skill-name
description: One-line purpose
metadata:
  version: 1.0.0
  evidence: { repos_analyzed: X, success_rate: Y% }
  last_updated: 'YYYY-MM-DD'
  supports_resume: false  # true for multi-day workflows
allowed-tools: [Bash, Read, Write, Edit, Grep, Glob]
follows_patterns:
  - proactive-error-capture
  - evidence-based-references
  - prerequisites-first
  - clean-context-execution  # Recommended for all skills
context:
  mode: hybrid  # isolated|continuous|hybrid (hybrid recommended)
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
---

# Skill Name

Production-validated patterns from [X] successful [migrations/upgrades/changes].

## When to Use
- Clear triggering condition 1
- Clear triggering condition 2

## Command Routing (if applicable)

This skill supports natural language commands:
- "Check {{repo}}" → review mode
- "Guide me" → guidance mode
- Just "{{repo}}" → defaults to review

**Tier Confirmation**: Skill will confirm target before gathering context.

## Step 0: Triage / Prerequisites

**AGENT NOTE**: Step 0 MUST complete before proceeding to Step 1.

### Detect Current State
[Commands to identify variant]

**Validation**: Code-first analysis - verify by reading actual files, never assume.

### Check Blockers
- [ ] Blocker 1: [Check command] [Fix guidance]
- [ ] Blocker 2: [Check command] [Fix guidance]

### Recommend Path
- If condition A → Section 2a
- If condition B → Section 2b

## Step 1: [First Action]

### 1a. [Sub-step]
[Detailed instructions]

**Validation**:
```bash
# How to verify this step succeeded
```

### 1b. [Sub-step]
[Detailed instructions]

**Dry-Run** (if applicable):
```bash
# Preview changes without executing
./script.sh --dry-run
```

## Step 2: [Second Action]

### Proactive Fixes (Apply Before Building)
**CRITICAL**: Fix these in BOTH src/main/ AND src/test/ together.

1. Fix A (applies to main + test code)
   ```bash
   grep -r "pattern" src/main/  # Fix main
   grep -r "pattern" src/test/  # IMMEDIATELY fix test
   ```
2. Fix B
3. Fix C

### Execute Changes
[Commands]

### Validation
- [ ] Builds successfully
- [ ] Tests pass

## Common Issues & Fixes

### Issue: [Specific error message]
**Symptom**: [What user sees]
**Root Cause**: [Why it happens]
**Fix**: [Specific solution]
**Evidence**: [Repo/PR where validated]
**Date Found**: YYYY-MM-DD

### Issue: [Another error]
[Same structure]

## Success Criteria
- [ ] Compilation succeeds
- [ ] All tests pass
- [ ] CI checks green
- [ ] No deprecation warnings
- [ ] [Domain-specific criteria]

## Evidence Base
- Repos analyzed: [X]
- Success rate: [Y%]
- PRs: [links]
- Episodes: [links]
- Last executed: YYYY-MM-DD

## Typical Timeline
- Simple case: [X] days
- Medium complexity: [Y] days
- Complex/blocked: [Z] days

## Rollback Plan (if applicable)
1. Revert commit: [command]
2. Redeploy previous: [command]
3. Recovery time: [estimate]

## Session Management (if multi-day workflow)

### State Location
`.skill-state/{{skill-name}}/{{session-id}}.json`

### Resume Command
```bash
/skill-name --resume {{session-id}}
```

### Checkpoint Structure
```json
{
  "session_id": "uuid",
  "current_step": "2b",
  "completed_steps": ["0", "1", "2a"],
  "context": {
    "repo": "example-repo",
    "variant": "variant-a"
  }
}
```

## Agent Execution Notes

### Common Agent Errors
**Error**: Skips Step 0 Prerequisites
- **Fix**: Halt and execute Step 0 first
- **Prevention**: Check `completed_steps` before proceeding

**Error**: Uses bash grep instead of Grep tool
- **Fix**: Use Grep tool for better permission handling
- **Prevention**: Review allowed-tools in frontmatter

**Error**: Assumes structure without verification
- **Fix**: Apply Code-First Analysis - always read files first
- **Prevention**: Explicit validation commands in each step
```

---

## Domain-Specific Patterns

### Migration Skills (Java, Node, iOS Upgrades)

**Pattern: Sequential Version Progression**
```
Cannot skip major versions. Each upgrade must be tested and deployed before proceeding.
Node 12 → 14 → 16 → 18 (cannot skip)
iOS 13 → 14 → 15 → 16 (always sequential)
```

**Pattern: Proactive Known Issues**
```markdown
## Known Issues for Version X.Y
Issue 1: [symptom] → [fix] (validated in N repos)
Issue 2: [symptom] → [fix] (validated in M repos)
```

---

### GraphQL Skills

**Pattern: Compile-Driven Diff Detection**
```
Let the compiler tell you what changed, don't manually diff schemas.
```

**Pattern: Deprecation Lifecycle**
```
Phase 1: Announce (Day 0) - Add @deprecated with replacement
Phase 2: Monitor (Day 0-90) - Track usage, assist migrations
Phase 3: Remove (Day 90+) - Only after usage = 0
```

---

### RCP Migration Skills

**Pattern: Squad-Based Prioritization**
```
Squad 1 (pre-work): Easiest first to keep pipeline full
Squad 4 (migration): Ready apps, categorized by complexity
```

---

## Anti-Patterns to Avoid

### 1. Iterative Debugging
**Anti-pattern**: Fix one error → rebuild → fix next error

**Impact**: 3-5 day migrations become 2-3 weeks

**Fix**: Capture all errors at once, categorize, batch fix

---

### 2. Blind Text Substitution
**Anti-pattern**: `s/old/new/g` without understanding context

**Example**:
```bash
# WRONG
sed -i 's/com.homeaway/com.expediagroup/g' pom.xml

# RIGHT (use verified mappings)
# Check mapping rules:
# - Bundle packages: com.homeaway.dropwizard.X → .bundle.X
# - Never-migrated: request-marker, mybatis stay as-is
```

**Impact**: Breaks code, creates hard-to-debug issues

---

### 3. Missing Pre-Migration Validation
**Anti-pattern**: Start migration without checking prerequisites

**Impact**: Start wrong path, discover blockers mid-way, waste effort

**Fix**: Step 0 ALWAYS validates prerequisites

---

### 4. Reactive CI Fixes
**Anti-pattern**: Wait for CI to fail → investigate → fix

**Impact**: 5-10 CI cycles, 2-3 day delays

**Fix**: Apply known fixes proactively before pushing

---

### 5. No Evidence Base
**Anti-pattern**: Skills without real-world validation

**Impact**: Untested advice, low confidence, repeat failures

**Fix**: Every skill MUST cite evidence:
```markdown
## Evidence Base
- Repos analyzed: 137
- Success rate: 94%
- PRs: [links to 5-10 representative PRs]
- Known issues: Documented from real failures
```

---

## Critical Gaps to Fill

### Gap 1: Post-Execution Validation
**Present in**: Java, iOS upgrade skills
**Missing in**: Many GraphQL, some RCP skills

**Add to all skills**:
```markdown
## Step X: Post-Migration Validation
Run 48-hour staging integration tests:
- JSON/XML serialization verification
- Null-safety checks
- Dependency conflict detection
- Performance regression testing (within 5% baseline)
```

---

### Gap 2: Rollback Plans
**Present in**: RCP traffic migration
**Missing in**: Most upgrade skills

**Add to all skills**:
```markdown
## Rollback Plan
If issues discovered in production:
1. Revert commit SHA: <commit>
2. Redeploy previous version: [command]
3. Expected recovery time: 10-15 minutes
4. Validation: [checks to confirm rollback succeeded]
```

---

### Gap 3: Effort Estimates
**Present in**: RCP migration analysis
**Missing in**: Most other skills

**Add to all skills**:
```markdown
## Typical Timeline
**Simple case** (no blockers, < 5 deps): 2-3 days
**Medium complexity** (some conflicts, 5-15 deps): 5-7 days
**Complex** (many conflicts, > 15 deps, custom code): 10-15 days

**Complexity scoring**:
+1: Dependency conflicts
+1: Multi-module project
+2: Custom plugins/extensions
+2: API breaking changes requiring code rewrites
```

---

### Gap 4: Common Issues Troubleshooting
**Present in**: Node, iOS, Java upgrade skills
**Missing in**: Newer skills

**Add to all skills**:
```markdown
## Common Issues & Fixes

### Issue: [Copy exact error message]
**Symptom**: [What the user sees]
**Root Cause**: [Why it happens]
**Fix**: [Step-by-step solution]
**Prevention**: [How to avoid next time]
**Evidence**: [Repo where this was validated]
```

---

## Recommendations

### 1. Create Skill Template (Meta-Skill)
Location: `meta-skills/create-skill/SKILL.md`

Generates new skills following this template automatically.

---

### 2. Shared Reference System
Create: `domains/shared/references/`

```
shared/references/
├── dependency-mappings/
│   ├── java-17.json
│   ├── node-20.json
│   └── ios-17.json
├── breaking-changes/
│   ├── java/
│   ├── node/
│   └── ios/
├── docker-images/
│   └── official-registry.md
└── troubleshooting/
    └── common-errors.md
```

Skills reference: `[See shared/references/breaking-changes/java-17.md]`

---

### 3. Evidence Collection Automation
Add to ALL skills:

```yaml
post_execution:
  hooks:
    - name: collect-evidence
      trigger: on_success
      output: .skill-builder/evidence/{{skill_name}}-{{date}}.yaml
      capture:
        - execution_id
        - duration_seconds
        - issues_encountered
        - fixes_applied
        - success: true/false
```

Builds evidence base automatically over time.

---

### 4. Validation Skill
Create `/validate-migration` skill:

```bash
/validate-migration --type java
# Runs:
# - mvn clean test
# - Dependency conflict check
# - CI configuration validation
# - Common issues scan
# Reports readiness score: 0-100
```

---

### 5. Pre-Flight Skill
Create `/pre-flight-check` skill:

```bash
/pre-flight-check java-17-upgrade
# Validates:
# - Prerequisites met
# - No blockers present
# - Estimates complexity (1-10)
# - Recommends path
# Outputs: GO / NO-GO with reasoning
```

---

## Key Takeaways

### Core Principles
1. **Proactive > Reactive**: Capture all errors upfront, apply known fixes before pushing
2. **Evidence > Theory**: Every pattern backed by real migrations (repos, PRs, episodes)
3. **Code-First > Assumptions**: Always verify by reading code/config, never assume
4. **Triage First**: Step 0 validates prerequisites and detects variants
5. **Parallel > Sequential**: Identify independent operations, run concurrently
6. **Explicit Validation**: Every skill has testable success criteria
7. **Reference Files**: Maintain verified mappings to eliminate repeated searches
8. **Common Issues**: Capture troubleshooting knowledge from real failures
9. **Transparency**: Give users effort estimates and rollback plans upfront

### Advanced Patterns
10. **Session Management**: Multi-day workflows maintain state for resumability
11. **Smart Routing**: Auto-detect user intent without explicit subcommands
12. **Tier Confirmation**: Confirm scope before gathering context
13. **Exhaustive Scanning**: Scan until saturation for completeness-critical operations
14. **Quality Filtering**: Mine exhaustively, then validate and filter results
15. **Agent Guidance**: Document common agent errors and corrections
16. **Clean Context**: Deterministic behavior through context isolation at skill boundaries
17. **Hybrid Deterministic-Agentic**: Layer bash tools (repeatable) + AI (adaptive) + JSON schemas (contracts)

**Your original insight** about proactively fixing test code is a specific instance of the universal "Proactive > Reactive" pattern. We should apply this thinking to:
- API changes (fix main + test together)
- CI configuration (update before pushing)
- Known issues (pre-emptively fix documented problems)
- Dependencies (resolve conflicts upfront, not iteratively)

The best skills combine:
- **Deterministic algorithms** (compile-driven detection, verified mappings)
- **Evidence-based troubleshooting** (common issues from 100+ migrations)
- **Proactive error handling** (capture all at once, fix in parallel)
- **Code-first verification** (read actual code, never assume structure)
- **Session management** (enable multi-day workflows with checkpoints)
- **Agent self-correction** (document and prevent common agent errors)
- **Hybrid execution** (deterministic tools for reliability + AI for adaptability + JSON contracts for data exchange)
