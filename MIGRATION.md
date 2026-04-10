# Migrating to a New Machine

This document is for moving your full working state — four repos, work in progress, Claude Code memory and settings, personal configuration — from the current machine to a new one. It is written so that a future you (or the new Claude on the other machine) can follow the steps without having to reconstruct context.

There are three kinds of state to move:

1. **Git state** — the four repos, including uncommitted work in progress
2. **Claude state** — `~/.claude/` including global instructions, memory, and settings
3. **Operating state** — shell config, credentials, per-machine tooling

And one artifact to leave for the new Claude:

4. **A handoff briefing** — copy-paste prompt that orients the new Claude immediately

Order matters. Do git state first, Claude state second, operating state third, and the handoff briefing last.

---

## Step 1 — Git state (the four repos)

Current repo state as of this commit (check before moving):

| Repo | Branch (at time of writing) | Has uncommitted changes? |
|---|---|---|
| `~/open-knowledge-graph` | `main` | no (this commit is the last) |
| `~/meta-knowledge-graph` | `main` | no (cleaned up earlier this session) |
| `~/org-open-knowledge-graph` | `repro-scaffold` (off `fold-solver-dev`) | **yes — 19 modified solver files on top of `fold-solver-dev`** |
| `~/the-silence-of-our-times` | `master` | yes — `notes/structure.md` modified, pre-existing from before this session |

### 1a. Decide what to do with uncommitted work in org-open-knowledge-graph

The 19 modified solver files on `fold-solver-dev` in `org-open-knowledge-graph` are your active work-in-progress. Three options:

**Option A: commit them first** (cleanest).

```bash
cd ~/org-open-knowledge-graph
git checkout fold-solver-dev   # make sure you are on the right branch
git status                       # verify which files
git add packages/core/           # or more specific paths
git commit -m "WIP: solver state before machine migration"
```

**Option B: create a patch file** (if the work is not ready to commit).

```bash
cd ~/org-open-knowledge-graph
git diff fold-solver-dev > ~/solver-wip.patch
```

Then carry `~/solver-wip.patch` to the new machine and `git apply ~/solver-wip.patch` after cloning.

**Option C: git stash + stash bundle** (if you want to stash, push the stash as a branch, and pull it on the new side).

```bash
cd ~/org-open-knowledge-graph
git stash push -m "pre-migration wip"
git stash branch wip-stash     # moves the stash to its own branch you can push
git push -u origin wip-stash
```

On the new machine after cloning, `git checkout wip-stash` and keep working there.

Pick one. The author recommends Option A unless the work is truly unfinished; Option B is the safest fallback.

### 1b. Push everything to remote

```bash
cd ~/open-knowledge-graph && git push origin main
cd ~/meta-knowledge-graph && git push origin main
cd ~/org-open-knowledge-graph && git push origin fold-solver-dev && git push origin repro-scaffold
cd ~/the-silence-of-our-times && git push origin master
```

Check every push completes without errors. If any repo has branches you care about that are not listed above, push them too (`git branch -a` to list).

### 1c. On the new machine: clone everything

```bash
cd ~
git clone <your-fork-url>/open-knowledge-graph.git
git clone <your-fork-url>/meta-knowledge-graph.git
git clone <your-fork-url>/org-open-knowledge-graph.git
git clone <your-fork-url>/the-silence-of-our-times.git
```

Replace `<your-fork-url>` with the actual remote URL(s) — these may differ per repo. Use `git remote -v` on the old machine to look them up before you leave it behind.

Then check out the branches you were working on:

```bash
cd ~/org-open-knowledge-graph
git checkout fold-solver-dev      # or repro-scaffold, or whichever
```

Apply the patch from Step 1a Option B if that is the route you took:

```bash
git apply ~/solver-wip.patch
```

---

## Step 2 — Claude state (`~/.claude/`)

This is the part that captures the 1 GB of memory, plans, settings, and conversation history that makes Claude Code yours rather than a fresh install.

### 2a. What is in `~/.claude/`

From your current machine:

- **`CLAUDE.md`** (3 KB) — your global instructions. Critical. Every Claude session reads this.
- **`settings.json`** and **`settings.local.json`** (1 KB combined) — Claude Code settings.
- **`projects/`** (1 GB) — per-project memory directories. This is where your `MEMORY.md` files and all the individual memory notes live. Critical.
- **`plans/`** — persistent plans you may have made.
- **`plugins/`** — installed plugins.
- **`agents/`** (if present) — custom agents.
- **`backups/`** — local backups of some state.
- **`cache/`, `debug/`, `downloads/`, `paste-cache/`, `session-env/`, `sessions/`, `shell-snapshots/`, `tasks/`, `file-history/`, `history.jsonl`** — per-machine transient state. Do not move these.

### 2b. Rsync the parts worth moving

From the new machine (or from the old machine with the new machine as the destination, whichever is easier for you):

```bash
# Run from the old machine, pushing to the new one
rsync -av --progress \
  --exclude 'cache/' \
  --exclude 'debug/' \
  --exclude 'downloads/' \
  --exclude 'paste-cache/' \
  --exclude 'session-env/' \
  --exclude 'sessions/' \
  --exclude 'shell-snapshots/' \
  --exclude 'tasks/' \
  --exclude 'file-history/' \
  --exclude 'history.jsonl' \
  ~/.claude/ \
  new-machine:~/.claude/
```

If the new machine does not have SSH access from the old one, tar it instead and carry it:

```bash
cd ~
tar czf claude-state.tar.gz \
  --exclude '.claude/cache' \
  --exclude '.claude/debug' \
  --exclude '.claude/downloads' \
  --exclude '.claude/paste-cache' \
  --exclude '.claude/session-env' \
  --exclude '.claude/sessions' \
  --exclude '.claude/shell-snapshots' \
  --exclude '.claude/tasks' \
  --exclude '.claude/file-history' \
  --exclude '.claude/history.jsonl' \
  .claude
```

On the new machine:

```bash
cd ~
tar xzf claude-state.tar.gz
```

### 2c. Verify the memory migrated

On the new machine, check that the memory files and global instructions are in place:

```bash
cat ~/.claude/CLAUDE.md | head -20
ls ~/.claude/projects/-Users-patdmccarthy-open-knowledge-graph/memory/ | head
cat ~/.claude/projects/-Users-patdmccarthy-open-knowledge-graph/memory/MEMORY.md | head
```

You should see your global instructions and the memory index you expect to see. If any of these are empty, the migration did not take — re-run the rsync or tar step.

Note: the directory name `-Users-patdmccarthy-open-knowledge-graph` encodes the absolute path of the project. **If your username on the new machine is different**, or if you put the repos somewhere other than `~/open-knowledge-graph`, the project memory will not auto-link. You may need to rename the memory directory to match the new path. The encoding rule is: take the absolute path to the project, replace `/` with `-`, and that is the directory name.

---

## Step 3 — Operating state

Things that may need to move and are easy to forget:

- **SSH keys** for git remotes: `~/.ssh/`
- **Shell config**: `~/.zshrc`, `~/.bashrc`, `~/.zprofile`, `~/.bash_profile`
- **Python global config**: `~/.pythonrc`, `~/.config/pip/`
- **Git config**: `~/.gitconfig` (user name, email, aliases)
- **Neo4j credentials** if running a local Neo4j on the new machine (see `org-open-knowledge-graph/SETUP.md`)
- **API keys** and secrets: anywhere you store Anthropic API keys, HuggingFace tokens, cBioPortal credentials, etc. Check `~/.env` files, `~/.config/` directories, and shell environment variables (`env | grep -i 'KEY\|TOKEN\|SECRET'` on the old machine).
- **Licensed data**: if you have MSK-IMPACT or TCGA data downloaded locally (referenced by `gnn/config.py` paths), you need to re-download or copy it to the new machine. The data is large and licensed, so it is not in git.

Also check for gitignored state you actually care about:

```bash
# On the old machine, for each repo:
cd ~/org-open-knowledge-graph
find . -type d \( -name data -o -name cache -o -name results \) -not -path '*/\.*' 2>/dev/null
# Review output, copy any directories that have state you cannot regenerate
```

The `analysis/paralog_projection/data/paralogs.tsv` (380 MB, from the BioMart fetch in this session) is regenerable from `fetch_paralogs.py`, so you do not need to copy it — just re-run the fetch on the new machine per `NEXT_STEPS.md`.

---

## Step 4 — Handoff briefing for the new Claude

Copy the contents of this section into the first message you send to Claude on the new machine. It is written to be self-contained and to orient the new session immediately without context-stuffing.

---

### Copy-paste prompt starts below

```
Briefing for new machine session.

I just migrated my working state from another machine. Before you do anything, read these files in this order:

1. /Users/<me>/open-knowledge-graph/REPRODUCIBILITY_AUDIT.md — the state of reproducibility across the four repos I work on.
2. /Users/<me>/open-knowledge-graph/analysis/paralog_projection/NOTES.md — the theoretical development of "Paralog as Projection" and the follow-up encapsulation / K/A hierarchy / bridge paper ideas. This is ~1200 lines and captures the theoretical work of several sessions. Read all of it.
3. /Users/<me>/open-knowledge-graph/analysis/paralog_projection/NEXT_STEPS.md — reproducible setup instructions for the paralog projection analysis.
4. /Users/<me>/open-knowledge-graph/analysis/paralog_projection/README.md — short summary of the test.
5. /Users/<me>/.claude/projects/-Users-<me>-open-knowledge-graph/memory/MEMORY.md — the memory index for this project. The files it links to have my preferences, feedback, and project state. Follow relevant links.

Where I left off:

- The "Paralog as Projection" paper scaffold is committed in ~/open-knowledge-graph/analysis/paralog_projection/. Panel 1 (the dual-proximity handoff between linear chromosomal distance and 3D Hi-C contact across paralog divergence) is the gate for whether the paper exists. Panel 1 has not been run yet. Running it is the next concrete action.
- The fetch scripts exist and are parameterized. fetch_paralogs.py has been run once on the previous machine; the 382 MB output is regenerable and gitignored. The Hi-C and DepMap fetches have NOT been run; they are scripted but need to be executed here (the new machine has the hardware for the full Hi-C matrices, not just loop calls).
- The theoretical thread developed significantly in the previous session. The paper now has an explicit bridge to papers 1-4 (K/A inseparability as the mechanism forced by CAP under finite context), an encapsulation reading of channels (software-engineering literal, wrappers around bacterially-derived implementations), and a recursive K/A hierarchy framing where attention is the top-order unit managing the interface with the unbounded environment. NOTES.md has the full development.
- Critical rule from the last session: DO NOT name the levels of the K/A hierarchy in the paper. The level count and boundaries are what the analysis MEASURES, not what the theory ASSERTS. Let the data reveal the structure. NOTES.md has a dedicated section on this.
- The reproducibility audit found that org-open-knowledge-graph has hardcoded Neo4j credentials in 4+ scripts. This is flagged in its SETUP.md but not yet fixed. Do not attempt to fix it without explicit instruction — it is a targeted refactor that should be its own commit.

The next concrete action, when I give you go:

1. Create the venv and install requirements per ~/open-knowledge-graph/analysis/paralog_projection/NEXT_STEPS.md step 2.
2. Run fetch_paralogs.py (fast, ~5 min).
3. Verify the anchor case (RAD51C should have 6 paralogs across chromosomes 7/14/15/17/22, all Opisthokonta divergence) per the command in NEXT_STEPS.md step 3a.
4. Run fetch_hic.py for the Rao 2014 GSE63525 loop calls for GM12878, IMR90, K562 (fast, under 1 MB total).
5. Wait for my go to do anything with the full Hi-C contact matrices or DepMap data — those are multi-GB downloads and I want to decide the order explicitly.

Working style that you should already know from memory but in case you do not: short sentences, lead with insight, no warmup, no emojis, parens for quiet asides, emdashes for attention-commanding interjections, bold for emphasis (no color changes), and never Co-Authored-By Claude in git commits. Scripts should be parameterized. Always show me the section after editing so I can review. When corrected on anything, audit all related work for the same error and ask once how much should change.

Acknowledge you have read the files above by telling me the commit hashes of the most recent 3 commits in open-knowledge-graph, and summarizing in under 100 words what Panel 1 is supposed to show.
```

### Copy-paste prompt ends above

---

## Step 5 — First run on the new machine

After migration, the first thing to do on the new machine is verify the handoff worked. Tell Claude the prompt above (after replacing `<me>` with your actual username). Claude should respond with three recent commit hashes and a short description of Panel 1. If the response does not make sense, something did not migrate correctly — debug by re-verifying steps 1, 2, and 3 above.

Then run through the sanity checks:

```bash
cd ~/open-knowledge-graph
git log --oneline -5
cd analysis/paralog_projection
ls              # expect: README.md NOTES.md NEXT_STEPS.md requirements.txt fetch_*.py build_pair_table.py data/
cat NEXT_STEPS.md | head -50
```

If everything looks right, proceed with the first fetch per NEXT_STEPS.md.

---

## Notes for the author

- If you update the handoff prompt in Step 4 to include more context (or less), commit the update so the next migration uses the improved version.
- If the migration reveals state you did not think to move, add it to this document in the same commit that captures the fix.
- This document is itself reproducibility scaffolding — the kind of thing the reproducibility audit in REPRODUCIBILITY_AUDIT.md wants every subproject to have. If you find yourself doing something that belongs in this document, put it here rather than remembering it.
