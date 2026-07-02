# The Agentic Coding Playbook
### Five setups for working with AI coding agents, from a two-file minimum to a full multi-agent team — ranked by complexity, grounded in cited practice and plain software-engineering reasoning.

*Compiled from Anthropic's engineering documentation, the open AGENTS.md standard, GitHub/Cursor/Claude Code documentation, and the published workflows of practitioners including Harper Reed, Simon Willison, Addy Osmani, Kent Beck, Dex Horthy (HumanLayer), and Walden Yan (Cognition/Devin). Full source list at the end.*

---

## How to use this document

You asked for setups ranked by complexity, each one effective on its own, each justified by both a named practitioner's real workflow and a plain software-engineering rationale that would still hold even if LLMs were worse than they are today. That's what follows.

**The tiers are additive, not competing.** Tier 3 is Tier 2 plus new files and one new rule. Tier 5 is Tier 4 plus parallelism. You are not choosing "which philosophy to follow" — you're choosing how far down a single, coherent stack to go, and that choice should track the actual size and risk of the project, not the sophistication of the tool you happen to be using. A two-day script doesn't need Tier 5. A payments pipeline conversion touched by three engineers for six months shouldn't stop at Tier 1.

**Two running examples, used throughout**, because you named both:
- **Example A — Pipeline conversion**: parsing a DAG-based ETL pipeline (e.g. Ab Initio graphs) and re-emitting it as an equivalent pipeline in another stack (SQL/dbt, Python, Spark). Correctness is checkable: output equivalence against fixtures. Work naturally decomposes into independent units (one graph node type, one module at a time).
- **Example B — Full-stack app with iteration**: a product with a frontend and backend, built incrementally against evolving feedback. Correctness is partly checkable (tests) and partly a judgment call (does this feel right, does the UX work).

These sit at opposite ends of a spectrum — deterministic/batch vs. interactive/iterative — and every tier below is built to handle both, because the underlying problem (an LLM agent's context window is small, decays, and doesn't persist) is identical regardless of what you're building.

**Tool-maturity coverage.** Claude Code has native cross-session memory, subagents, hooks, and plan mode built in. GitHub Copilot, at the time of writing, has none of those as first-class primitives — it has instruction files and, in the IDE, chat modes and prompt files, full stop. Cursor sits in between. Every tier below is written to work on the *least* capable of these tools using plain markdown files and disciplined prompting, with a note on which native feature shortcuts the work when you have Claude Code or Cursor available. If it doesn't work in GitHub Copilot with nothing but files and words, it isn't in the core recommendation.

### Table of contents
1. [The one root cause: context engineering](#root-cause)
2. [Tool capability map](#tool-map)
3. [Tier 1 — The Minimal Portable Core](#tier-1)
4. [Tier 2 — Externalized Memory](#tier-2)
5. [Tier 3 — The Verified Loop](#tier-3)
6. [Tier 4 — Spec-Driven Multi-Phase Development](#tier-4)
7. [Tier 5 — Multi-Agent Orchestration](#tier-5)
8. [Choosing your tier](#choosing)
9. [Tool-specific cheat sheet](#cheat-sheet)
10. [Curated sources](#sources)

---

<a id="root-cause"></a>
## 1. The one root cause: context engineering

Every mechanism in this document is a way of managing one resource: **the finite, decaying, non-persistent context window of an LLM agent.** Anthropic's own framing (in [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)) treats context engineering as the successor discipline to prompt engineering: it's not just about what you type into a single prompt, but about curating the entire set of tokens — system instructions, file contents, tool outputs, memory — that the model sees at inference time. Everything downstream of that idea follows from four sub-problems:

| Problem | What breaks without it | Tier that fixes it |
|---|---|---|
| What should the agent know at the *start* of every session, unconditionally? | It re-derives your conventions every time, inconsistently, burning tokens and producing drift | **Tier 1**: a root instructions file |
| What should *survive* a context reset or a brand-new session? | Anthropic's own long-running-agent research frames this with a memorable analogy: a project staffed by engineers on rotating shifts, where each new arrival has zero memory of the shift before theirs | **Tier 2**: externalized memory files |
| How do we know the agent's output is *actually* correct and not just "looks done"? | Agents confidently mark work complete without proof — Anthropic's own harness research found this as a major failure mode, and it gets worse, not better, under pressure to appear productive | **Tier 3**: a verification loop |
| How do we stop a large task from overflowing one agent's attention or one context window? | The agent jumps straight to code and solves the wrong problem, or a single session degrades as it fills | **Tier 4** (planning) and **Tier 5** (parallel isolated contexts) |

This isn't a new problem invented by LLMs — a colleague with amnesia between shifts, a teammate who says "done" without demoing it, and a project with no design doc are all recognizable *human* engineering failure modes too. Every recommendation below is standard software-engineering discipline (single source of truth, ADRs, TDD, code review, CI) applied to a collaborator with a specific, well-documented set of blind spots.

Two more terms worth knowing because they explain *why* files need to stay short, not just *that* they should:

- **Context rot.** Model recall and reasoning quality degrade as context fills — this isn't hypothetical, it's a documented property Anthropic explicitly designs around, and independently, HumanLayer's Dex Horthy — after reviewing roughly 100,000 developer sessions — describes a "dumb zone" around the middle 40-60% of a large context window where recall and reasoning noticeably degrade. See [12-Factor Agents](https://github.com/humanlayer/12-factor-agents).
- **Instruction budget.** There's a practical ceiling on how many rules a model can reliably follow at once — practitioner analysis puts frontier thinking models at roughly 150-200 instructions before consistency drops, which is the concrete reason every tier below says "keep it short" rather than "write everything you can think of." Source: Kyle from HumanLayer, cited in the AGENTS.md practitioner guide referenced in [§10](#sources).

<a id="tool-map"></a>
## 2. Tool capability map

This is the answer to "what do I get for free, and what do I have to build myself." Read it once before Tier 1 — every tier below refers back to this table instead of repeating tool names.

| Mechanism | Claude Code | Cursor | GitHub Copilot | Plain-file fallback (any tool) |
|---|---|---|---|---|
| Root instructions file | `CLAUDE.md` (native) + `@AGENTS.md` import | `AGENTS.md` or `.cursor/rules/*.mdc` (native) | `.github/copilot-instructions.md` + `AGENTS.md` (native, both read) | `AGENTS.md` — the open standard now read natively by Cursor, Copilot, Codex, Gemini CLI, and 20+ other tools |
| Path-scoped rules | `.claude/rules/*.md` with path frontmatter | `.mdc` files with `globs:` | `.github/instructions/*.instructions.md` with `applyTo:` | nested `AGENTS.md` files per subdirectory — closest file to the edited file wins |
| Cross-session memory | Native (`CLAUDE.md` auto-loads; can read/write project files) | None native | None native | Manually maintained memory files, read/written on explicit instruction (Tier 2) |
| Sub-agents (isolated context) | Native — `.claude/agents/*.md` | Not native (single agent context) | Not native | A second, fresh chat window given only the artifact + criteria (blackbox verification, still works) |
| Deterministic hooks (run code on an event) | Native — `PreToolUse`/`PostToolUse`/`Stop` hooks | Limited | None | CI as the backstop — same gate for human and agent output |
| Formal planning mode | Native — Plan Mode | Manual (ask for a plan first) | Manual (ask for a plan first) | A `plan.md` file the agent must write and you must approve before it edits code |
| Parallel isolated sessions | Native — git worktrees, `--worktree` flag, agent teams | Manual git worktrees | Manual git worktrees / cloud coding agent | `git worktree add` — a plain git feature, works everywhere |

The load-bearing insight: **almost nothing below is Claude-Code-specific.** Claude Code productizes things (memory, subagents, hooks, worktrees) that Cursor and Copilot users can still get, slightly more manually, with markdown files and disciplined prompting. That's the whole design constraint of this document.


---

<a id="tier-1"></a>
## 3. Tier 1 — The Minimal Portable Core

**What it is:** one file, `AGENTS.md`, at the repo root. Nothing else.

**Why this is the floor, not a toy.** `AGENTS.md` is not a personal convention — it's an open standard, originally driven by OpenAI, Google, Cursor, Factory, and Amp, now [stewarded by the Agentic AI Foundation under the Linux Foundation](https://agents.md/), used in over 60,000 open-source repositories, and read natively by more than twenty tools including Codex, Cursor, Gemini CLI, Windsurf, Aider, and — as of the GitHub Copilot coding agent — Copilot itself. One plain markdown file, no proprietary format, no vendor lock-in. This single property is what makes it the correct floor for "something I can use anywhere": it is quite literally designed to be the anywhere-file.

**The one exception:** Claude Code reads `CLAUDE.md`, not `AGENTS.md`, and does not fall back automatically. The fix is one line, not a duplicated file — put this as the entire content of `CLAUDE.md`:

```
@AGENTS.md
```

Claude Code's `@path` import syntax expands the target file into context at session start, so you maintain exactly one file and Claude Code sees the same instructions as everything else. (A symlink, `ln -s AGENTS.md CLAUDE.md`, does the same thing but needs admin rights on Windows, so the import line is the more portable choice.)

### Exact contents

```markdown
# AGENTS.md

## Project
[One or two sentences: what this is, and what "correct" means for it.
Example A (pipeline conversion): "Converts Ab Initio ETL graphs into
equivalent dbt+SQL pipelines. Correctness = row-level output equivalence
against the fixtures in /fixtures for every existing graph."
Example B (web app): "A Next.js + FastAPI app for [X]. Correctness =
passing tests plus the acceptance criteria in the linked issue."]

## Setup
- Install dependencies: `<exact command>`
- Run locally: `<exact command>`
- Required environment variables: `<names only — never values or secrets>`

## Build & test
- Full test suite: `<exact command>`
- Single test: `<exact command, with a placeholder for the test name>`
- Lint / type-check: `<exact command>`
- A change is not finished until build, full test suite, and lint all pass locally — not just the tests you think are related.

## Code style
- [3–8 concrete, checkable rules. Not "write clean code" — instead:
  "New graph-node handlers go in src/nodes/, one file per node type."
  "All public functions have type hints and a one-line docstring."
  "Prefer small pure functions over classes for stateless transforms."]

## Testing conventions
- Framework: `<name>`
- New code requires new tests in the same change, following `<path convention>`.
- Never edit or delete an existing test to make it pass. If a test looks wrong, stop and ask instead of "fixing" it.

## Things not to touch
- [Generated files, vendored code, migration history, credentials —
  be specific about paths, not vague warnings.]

## Commit / PR conventions
- [Message format. What must be in the description — e.g. "include the
  exact test command you ran and its output, not just 'tests pass.'"]
```

### Why each section earns its place (not a bigger file, a *denser* one)

- **Keep it lean on purpose, not by accident.** A widely-read practitioner analysis of AGENTS.md adoption found that LLM-*generated* instruction files (the kind produced by an auto-init script) measurably reduced agent success rates and increased cost, mostly because they duplicated things the agent could already see by reading the repo. The fix isn't more content, it's *higher-signal* content — describe capabilities and conventions, not file paths (paths go stale the moment someone refactors, and a confidently wrong path is worse than no path). See the [aihero.dev AGENTS.md guide](https://www.aihero.dev/a-complete-guide-to-agents-md).
- **The "ball of mud" failure mode.** The same source documents the natural decay pattern: the agent does something wrong, you add a rule, repeat for months, and the file becomes an unmaintainable pile that different contributors have quietly contradicted in three places. Treat this file the way you'd treat a hot section of production code — it needs periodic review passes, not just additions.
- **GitHub's own guidance for its coding agent** converges on the identical advice from the opposite direction: shorter instruction files are more likely to be fully processed, and repos should start with a minimal set and expand iteratively rather than front-loading everything. See [GitHub Docs — using custom instructions](https://docs.github.com/en/copilot/tutorials/use-custom-instructions).
- **This is DRY, not an AI-specific idea.** A root instructions file is the same "single source of truth" discipline as a `README` or a linter config — you're encoding tribal knowledge once instead of repeating it in every prompt, every PR comment, every onboarding conversation. The only thing new is *who* reads it.

### Portability check
Drop this same `AGENTS.md` (plus the one-line `CLAUDE.md` import) into a Copilot repo, a Cursor repo, or a Claude Code repo, and all three will pick it up with zero additional configuration. That is the entire point of Tier 1: it is the setup with the best effort-to-effectiveness ratio that exists, and it's also the foundation every later tier builds on top of, unchanged.

---

<a id="tier-2"></a>
## 4. Tier 2 — Externalized Memory

**What it adds:** a `memory/` folder with three files (`PROGRESS.md`, `DECISIONS.md`, `LEARNINGS.md`), plus one new section in `AGENTS.md` that tells the agent when to read and write them.

**The problem this solves — stated exactly, since you asked for it explicitly.** Claude Code keeps a running memory of a project across a session and can read/write files as durable state. GitHub Copilot does not: every new chat is a blank slate except for whatever's in your instructions file and whatever the model chooses to (re-)discover by reading the repo. Cursor sits closer to Copilot than to Claude Code on this axis. If your workflow assumes the agent "remembers" what happened yesterday, it will silently fail the moment you're not using Claude Code. Tier 2 fixes this the same way source control fixed "remembering" code changes: by writing state to disk instead of relying on anyone's memory, human or model.

This is not a workaround invented for this document — it's the exact shape of two independently-arrived-at patterns:

1. **Anthropic's own long-running-agent research** (building the Claude Agent SDK's autonomous coding demo) hit precisely this problem and solved it with a `claude-progress.txt` file written alongside git history, so a fresh context window can reconstruct project state without re-deriving it. They frame the motivating scenario as a team where every new engineer arrives with no memory of the previous shift. See [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents).
2. **Cline's "Memory Bank" pattern** — built for exactly this reason, since Cline's context also resets — uses a small set of markdown files read at the start of every task and updated on command, and its own documentation is explicit that this is a *methodology*, not a Cline-only feature: it works with any AI assistant that can read files. See [Cline — Memory Bank](https://docs.cline.bot/features/memory-bank).

Tier 2 below is a generic synthesis of both, deliberately simplified to three files so it stays under the instruction-budget and file-count you can realistically maintain by hand.

### File tree
```
AGENTS.md          (from Tier 1, plus one new section — see below)
CLAUDE.md           @AGENTS.md
memory/
  PROGRESS.md
  DECISIONS.md
  LEARNINGS.md
```

### Exact contents

**`memory/PROGRESS.md`**
```markdown
# Progress

## Current state (overwrite this section every session)
- Last updated: <date> by <session/agent name>
- What works right now:
- What's in progress:
- What's broken / known issues:

## Next steps
1.
2.

## Session log (append, never delete)
### <date> — <one-line summary>
- Changed:
- Verified by: <exact command + result, or "NOT YET VERIFIED">
- Follow-ups:
```

**`memory/DECISIONS.md`** — a lightweight Architecture Decision Record. ADRs predate LLM agents by well over a decade (the pattern is usually credited to Michael Nygard's 2011 writeup) and exist for a very ordinary reason: without them, nobody — human or agent — can tell whether a weird-looking piece of code is a mistake or a deliberate tradeoff, and both humans and agents end up re-litigating settled decisions.
```markdown
# Decisions
Append-only. Never delete an entry — mark it superseded instead.

## D-001: <short title>
- Date:
- Context: <what problem forced this decision>
- Decision: <what we chose>
- Alternatives considered: <what we didn't choose, and why not>
- Status: Active | Superseded by D-00X
```

**`memory/LEARNINGS.md`** — this is the file that most directly answers "what covers for a tool with no memory." Every time you correct the agent's approach, that correction gets written down instead of evaporating at the end of the session. This mirrors Cline's own `.clinerules` learning loop (identify a pattern → confirm it with the human → write it down) almost exactly, generalized to a plain file any tool can be told to read.
```markdown
# Learnings & corrections
Append a new entry every time a human corrects the agent's approach.
Read this file in full before starting any new task.

## L-001: <short description of the mistake>
- What happened:
- Correct approach:
- Why:
```

**Addition to `AGENTS.md`:**
```markdown
## Memory protocol
This tool does not retain memory between sessions. At the START of every session:
1. Read memory/PROGRESS.md, memory/DECISIONS.md, and memory/LEARNINGS.md in full.
2. State your understanding of current project state in one paragraph before touching any code.
At the END of every session, or before you expect to run low on context:
3. Update PROGRESS.md's "Current state" section and append a session log entry.
4. If you were corrected on anything this session, append an entry to LEARNINGS.md.
```

### Why three files and not one, and why not more than three
- **One file invites either bloat or overwrite-collisions** — a single `MEMORY.md` mixing "what's the state right now" with "why did we choose Postgres over Mongo six weeks ago" forces the agent to re-read decision history every session even when nothing about it changed. Splitting by *volatility* (progress changes every session, decisions rarely change, learnings only grow) means the agent's session-start read is proportional to what's actually new.
- **More than three starts fighting the instruction budget** from [§1](#root-cause) for very little marginal benefit at solo-to-small-team scale — this is deliberately a simplified subset of Cline's own six-file Memory Bank (`projectbrief.md`, `productContext.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`), collapsed because most of that context either belongs in `AGENTS.md` already (Tier 1) or isn't worth the maintenance cost until a project is large enough to need Tier 4's spec files anyway.
- **Commit these files to git.** This is the same audit-trail argument as commit messages: a diffable, reviewable history of *why* the agent believed what it believed at each point is worth more than the files' content at any single instant, and it lets a human catch drift by skimming `git log memory/`.

### Applying this to the two running examples
- **Pipeline conversion (A):** `PROGRESS.md` tracks which graph node types are converted and verified vs. still stubbed; `DECISIONS.md` records things like "chose to represent fan-out nodes as CTEs, not temp tables, because <reason>"; `LEARNINGS.md` accumulates Ab Initio-specific parsing gotchas the agent got wrong once (e.g. a particular MP-file quirk) so it never re-learns them the hard way.
- **Web app (B):** `PROGRESS.md` tracks feature-by-feature status against the backlog; `DECISIONS.md` is where "we're using React Query, not Redux, because X" lives so a later session doesn't reintroduce the rejected pattern; `LEARNINGS.md` captures UX corrections ("don't add a modal for this, we deliberately want inline editing") that would otherwise get silently reverted every time a fresh session touches that screen.

---

<a id="tier-3"></a>
## 5. Tier 3 — The Verified Loop

**What it adds:** a verification section in `AGENTS.md`, a red-green-refactor rule, and a standalone "verifier" prompt you run as a second, skeptical pass — either as a real subagent (Claude Code) or as a fresh chat window given nothing but the diff and the requirement (Copilot, Cursor, anything).

**Why this tier exists — and why it's not optional past a hobby project.** Left alone, agents over-report success. This isn't a slight against any particular model; it's a documented, repeatedly-observed pattern from people with the most hours logged watching it happen:

- Anthropic's own harness research names it directly as **the** major failure mode of long-running coding agents: without explicit prompting, the model would make changes, run *some* tests or a curl command, and then fail to notice the feature didn't actually work end-to-end. See [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents).
- **Kent Beck** — the creator of Test-Driven Development itself, co-author of the Agile Manifesto — has said in interview that TDD has become a "superpower" specifically *because* agents introduce regressions, but that his actual hands-on struggle is stopping agents from **deleting or rewriting tests to make them pass** rather than fixing the underlying code. If the person who invented the practice is fighting that exact failure mode with today's tools, it is not a hypothetical risk. See [The Pragmatic Engineer — TDD, AI agents and coding with Kent Beck](https://podcasts.apple.com/us/podcast/tdd-ai-agents-and-coding-with-kent-beck/id1769051199?i=1000712458898).
- Simon Willison's framing for what "done" is supposed to mean puts the burden explicitly on the person shipping the change, not the tool: work isn't finished until it's been *proven* to work, through both a manual demonstration and an automated test, and skipping that step just moves the burden of actually checking onto whoever reviews it next. See [Your job is to deliver code you have proven to work](https://simonwillison.net/2025/Dec/18/code-proven-to-work/).
- Boris Cherny, who built Claude Code, has called giving the agent a way to verify its own work the single highest-leverage thing you can do — describing feedback loops as roughly doubling or tripling final output quality on their own, independent of anything else you do.

### The one subtlety that separates a good testing setup from a bad one

It's tempting to solve this by spinning up a dedicated "tester" agent that writes tests for whatever the "implementer" agent builds. **This is a documented anti-pattern, not a stylistic choice.** Anthropic's own multi-agent research is explicit: splitting software work by *role* (planner, implementer, tester, reviewer, as separate persistent agents handing off to each other) creates a "telephone game" where each handoff loses context — in one internal experiment, role-split agents spent more tokens coordinating than actually working. The fix is **context-centric, not problem-centric, decomposition**: the agent that implements a feature should also write its tests for that feature, because it already has the context of *why* it built things the way it did. See [Building multi-agent systems: when and how to use them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them).

Where a *separate* agent earns its keep is verification — not writing tests alongside the code, but independently, skeptically checking finished work against stated criteria with no attachment to how it was built. That's a genuinely different job (blackbox pass/fail against explicit criteria) from co-authoring the tests, and it needs almost none of the implementation context, which is exactly why it survives a context handoff that role-splitting doesn't. The same Anthropic writeup names this "the verification subagent pattern" and documents its most common failure — the **early victory problem**, where a verifier runs one or two tests, sees them pass, and declares victory without running the full suite. The mitigation is blunt and effective: tell it, explicitly, that it must run the complete suite and is not permitted to mark anything passed on a partial run.

### Addition to `AGENTS.md`

```markdown
## Verification protocol
1. Before writing implementation code, write the test(s) for the behavior
   you're about to build. Run them and confirm they fail for the expected
   reason — not for an unrelated error.
2. Write the minimum code needed to make them pass.
3. Run the FULL test suite, not just the new tests. A change that breaks
   something else is not done.
4. Never state a task is complete without pasting the exact command you
   ran and its actual output. "Tests pass" without evidence doesn't count.
5. Never edit or delete a test to make it pass. If a test looks wrong,
   stop and ask a human — do not "fix" it yourself.
6. Before declaring a feature finished, exercise it the way a user would
   (call the endpoint, run the CLI, load the page) — not only via unit tests.
   For anything with a UI, take a screenshot of the actual result and
   compare it against what was asked for before calling it done.
```
*(Point 6's screenshot step is a known Claude Code pattern for UI work — write code, screenshot the rendered result, compare against intent, iterate — documented in Anthropic's own [Claude Code best practices](https://code.claude.com/docs/en/best-practices) as one of the tool's common workflows, and it works identically as a manual step in Cursor or Copilot: ask the agent to actually run the dev server and describe or capture what's on screen instead of trusting its own code-read.)*

### The verifier prompt (works as a plain prompt anywhere; becomes a real subagent in Claude Code)

```markdown
---
name: verifier
description: Independently verifies a completed unit of work. Use after
  any implementation task, before marking it done.
tools: Read, Grep, Bash
---
You are a skeptical verification engineer. You did not write this code
and have no attachment to it being right.

Given: the stated requirement, and the files that changed.

Do, in order:
1. Run the full test suite. Paste the exact command and the full output —
   not a summary.
2. Confirm ALL tests pass, not only the ones that look related to this change.
3. Exercise at least one edge case or input the implementer likely didn't
   think of.
4. Re-read the ORIGINAL requirement and check the change actually satisfies
   it end-to-end — "the code compiles" and "the tests I wrote pass" are
   not the same as "this does what was asked."

Report PASS or FAIL with evidence for each point above. You MUST run the
complete suite before marking PASS. A partial run is not a basis for PASS,
even if everything you did run succeeded.
```

- **In Claude Code:** save this verbatim as `.claude/agents/verifier.md` — the YAML frontmatter (`name`, `description`, `tools`) is real subagent configuration; Claude will delegate to it automatically when relevant, or you can invoke it explicitly ("use the verifier subagent to check this").
- **In Cursor:** save the body (without frontmatter, or keeping it as a comment) as `.cursor/rules/verifier.mdc` in Manual mode, and invoke it with `@verifier` in a *second*, separate chat — the isolation matters more than the tool feature; a fresh conversation with no memory of how the code was written is what makes the check honest.
- **In GitHub Copilot:** paste the prompt body into a new chat with only the diff and the requirement attached — Copilot has no formal subagent primitive, so the discipline of *starting a new conversation* is what buys you the same isolation a real subagent gives you elsewhere.

### Applying this to the two running examples
- **Pipeline conversion (A):** the verifier's job is concrete and automatable — run the converted pipeline against the fixture inputs and diff the output against the legacy pipeline's output, row for row. This is the cleanest possible case for Tier 3, because "correct" has an objective, scriptable answer.
- **Web app (B):** verification is partly automatable (test suite, type-checker) and partly a judgment call a human still has to make (does this UX actually solve the user's problem). Tier 3 doesn't remove that human judgment — it removes the *automatable* failure modes (regressions, untested edge cases, silently broken existing features) so the human's limited attention goes to the part only a human can evaluate.

---

<a id="tier-4"></a>
## 6. Tier 4 — Spec-Driven Multi-Phase Development

**What it adds:** a `constitution.md` at the repo root, and per-feature folders (`specs/<NNN>-<slug>/`) each containing `spec.md`, `plan.md`, and `tasks.md` — written and approved, in that order, *before* any implementation code is touched.

**Why this tier exists.** Every tier so far assumes the agent is working on a well-scoped unit of work. Past a certain size, that assumption breaks: Anthropic's own Claude Code documentation is blunt about the failure mode — letting the agent jump straight to coding on an under-specified request routinely produces code that solves the wrong problem, and the fix is to separate exploration/research and planning from implementation entirely. Tier 4 formalizes that separation into three reviewable documents instead of one long chat.

This tier has two independent lineages that converge on the same three-file shape, which is a good sign it's the right shape rather than one person's preference:

1. **Harper Reed's widely-cited informal workflow**: brainstorm an idea with an LLM one question at a time until you have a detailed `spec.md`, hand that to a reasoning model to produce a `prompt_plan.md` broken into small testable steps plus a `todo.md`, then execute the steps one at a time against a codegen tool. See [My LLM codegen workflow atm](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/).
2. **GitHub's formalized Spec Kit**, which turns the same idea into a supported toolchain: `/speckit.specify` generates `spec.md` (the what and why, deliberately free of implementation detail), `/speckit.plan` generates `plan.md` (the technical approach and architecture), `/speckit.tasks` generates a dependency-ordered `tasks.md` with `[P]`-marked parallelizable items, and an optional `/speckit.analyze` step cross-checks all three for gaps before anything is implemented. It works across 30+ agents including Copilot, Claude Code, Cursor, and Gemini CLI. See [github/spec-kit](https://github.com/github/spec-kit).

Tier 4 below gives you the file shape without requiring you to install the toolchain — you can adopt the discipline with nothing but markdown files and disciplined prompting, and adopt the CLI later if you want the automation.

### File tree
```
AGENTS.md, CLAUDE.md, memory/     (from Tiers 1–3, unchanged)
constitution.md
specs/
  001-<feature-slug>/
    spec.md
    plan.md
    tasks.md
```

### Exact contents

**`constitution.md`** (repo root, written once, revisited rarely) — your project's non-negotiables. If a plan ever conflicts with this file, the constitution wins, full stop.
```markdown
# Project constitution
Non-negotiable rules that every spec, plan, and task must comply with.

1. [e.g. "All external inputs are validated at the boundary — never trust
   data that's already inside the system."]
2. [e.g. "Every new pipeline node type ships with a regression fixture
   before merge, not after."]
3. [e.g. "No new runtime dependency without a one-line justification in
   the PR description."]
```

**`specs/001-<slug>/spec.md`** — what and why, deliberately no implementation detail. Harper Reed's technique for generating a good one is worth adopting even outside his full workflow: ask the model to interview you one question at a time, each building on the last answer, rather than accepting your first paragraph as sufficient.
```markdown
# Spec: <feature name>

## Problem
What's missing or broken, for whom, and why it matters. No implementation
details — that's the plan's job, not the spec's.

## Requirements
- [Numbered, testable statements. "The system MUST ..." / "The system
  MUST NOT ..." — vague requirements produce vague code.]

## Out of scope
- [Explicitly excluded, so scope doesn't quietly creep mid-implementation.]

## Acceptance criteria
- [How you'll know this is actually done — phrased so a test could check it.]
```

**`specs/001-<slug>/plan.md`** — the technical approach, written against the spec and the constitution.
```markdown
# Plan: <feature name>

## Approach
High-level technical approach and why — including alternatives you
considered and rejected, and which constitution rule (if any) ruled them out.

## Architecture / data flow
Components touched, new modules introduced, how data moves between them.

## Risks
What could go wrong, and how you'll know if it does.
```

**`specs/001-<slug>/tasks.md`** — the plan broken into small, ordered, independently-checkable units.
```markdown
# Tasks: <feature name>
Ordered by dependency. [P] = safe to execute in parallel with other [P] tasks.

- [ ] T1 [P] <task — exact file paths — acceptance check>
- [ ] T2 [P] <task — exact file paths — acceptance check>
- [ ] T3 (depends on T1, T2) <task — exact file paths — acceptance check>
```

### The one honest caveat

Spec-driven development is not free, and it is not always the right call. A hands-on writeup of putting GitHub's Spec Kit through real use found that over-applying it to a small, still-uncertain feature produced exactly the bureaucracy this document is trying to avoid — a sea of markdown documents and long agent run-times for a change that didn't warrant the ceremony. See [Putting Spec Kit through its paces](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html). The practical rule several independent sources converge on: **prototype and explore with Tiers 1–3; switch to Tier 4 once you're building the version that has to actually work** — production code, anything more than one person will touch, anything where a wrong turn is expensive to unwind. Budget roughly 20–40% more tokens and wall-clock time per feature against Tier 3, and expect it to pay for itself on anything that would otherwise cost you a rewrite.

### Applying this to the two running examples
- **Pipeline conversion (A) — where this tier shines.** `spec.md` defines the source and target semantics and the equivalence criteria (row-level match against fixtures); `plan.md` defines the parsing strategy, the intermediate graph representation, and the code-generation approach; `tasks.md` breaks the conversion into one task per node type or module — which also happens to be a naturally *independent* unit of work, setting up Tier 5 perfectly.
- **Web app (B).** `spec.md` captures the actual user problem before anyone touches a component library; `plan.md` is where "React Query vs. Redux" gets decided once instead of re-litigated every session; `tasks.md` becomes the backlog the agent works through, each task small enough to verify with Tier 3's loop before moving to the next.

---

<a id="tier-5"></a>
## 7. Tier 5 — Multi-Agent Orchestration

**What it adds:** git worktrees so multiple agents can work the codebase in parallel without colliding, one or more real subagent definitions for specialized, blackbox roles (verification, security review), a root-level `TASKS.md` dashboard to track who owns what, and CI as the single deterministic gate every worktree's output has to clear regardless of who — or what — produced it.

**Read this section's warning before its recipe.** The obvious way to build "a multi-agent setup" is to assign each software-development-lifecycle role — planner, implementer, tester, reviewer — to its own persistent agent and have them hand work down the line. **This is a well-documented failure mode, not a simplification you'll grow out of.** Two teams that ship agent products for a living arrived at the same conclusion independently, within days of each other, from opposite starting points:

- **Cognition AI** (makers of Devin), in Walden Yan's [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents): splitting agents by role fragments context, and "actions carry implicit decisions, and conflicting decisions carry bad results" — their worked example is a Flappy Bird clone where parallel subagents, each missing the others' intermediate context, produce a bird and a background in two different visual styles. Their later refinement, after more production use, sharpens this into a **single-writer principle**: additional agents are fine when they contribute *intelligence* — analysis, review, research — but only one agent should actually be making writes to the codebase at a time for any tightly-coupled piece of work.
- **Anthropic**, publishing on the same general timeframe about their own multi-agent research system, reached a compatible position from the other direction: multi-agent parallelism won an internal eval by over 90% on *breadth-first* research tasks — many independent facets, no shared state — but their own writeup explicitly flags that domains requiring shared context or with heavy inter-agent dependencies are a poor fit for multi-agent today. Their direct, quantified finding on the coding case: in an internal experiment splitting agents by SDLC role (planner/implementer/tester/reviewer), **the agents spent more tokens on coordination than on the actual work.** See [Building multi-agent systems: when and how to use them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

Both arrive at the same operating rule: **decompose by context boundary, not by job title.** Good boundaries are things that are genuinely independent — separate modules with a clean interface, blackbox verification, unrelated research paths. Bad boundaries are sequential phases of the *same* piece of work. That rule is what the rest of this tier is built around.

### The legitimate reasons to go multi-agent (and the cost of doing it anyway)

Anthropic's framework narrows it to three, and it's worth holding yourself to this list rather than reaching for parallelism because it feels more sophisticated:

1. **Context protection** — a subtask would pollute the main agent's context with information irrelevant to the rest of the work (e.g. a huge API response you only need a three-field summary from).
2. **Genuine parallelization** — the work decomposes into independent pieces with no shared state (Example A's node-by-node conversion is a textbook case of this).
3. **Specialization** — different subtasks need different tools, different system prompts, or conflicting behavioral modes (a compliance-checking persona and a brainstorming persona don't coexist well in one prompt).

Outside those three, don't. The cost is real and measured: multi-agent setups typically burn **3–10×** (in some measurements up to 15×) the tokens of an equivalent single-agent run, from duplicated context, coordination messages, and summarization at every handoff.

### File tree
```
AGENTS.md, CLAUDE.md, memory/, constitution.md, specs/   (Tiers 1–4, unchanged)
.claude/agents/
  verifier.md            (from Tier 3)
  security-reviewer.md
TASKS.md                 (root-level dashboard, distinct from per-feature tasks.md)
```

### Parallel isolation: git worktrees

A worktree is a plain git feature — a second working directory checked out from the same repository history, so two agents editing different branches never touch each other's files on disk. Claude Code has first-class support (`--worktree`, and `isolation: worktree` in subagent frontmatter for automatic per-subagent isolation), but the underlying mechanism is just git and works identically if you're driving Cursor or Copilot by hand:

```bash
# one worktree per genuinely independent unit of work
git worktree add ../repo-node-parser -b feature/node-parser
git worktree add ../repo-node-aggregator -b feature/node-aggregator
git worktree add ../repo-node-writer -b feature/node-writer

# each directory is a full, isolated checkout — point a separate
# agent session at each one; they cannot corrupt each other's files
```

Practical limits worth stating plainly: two to three parallel sessions is sustainable for one person to actually review; worktrees do **not** isolate shared infrastructure (a shared dev database, a shared `.env`, a shared port), so those need separate handling if two agents touch them concurrently; and every parallel agent you run is another PR you have to review, which is usually the real bottleneck long before compute is.

### A genuine specialization subagent (not a pipeline stage)

`.claude/agents/security-reviewer.md` — this is the *right* shape of subagent: single-purpose, blackbox, invoked as a gate rather than a handoff.
```markdown
---
name: security-reviewer
description: Reviews a diff for security issues. Use before merging
  anything that touches auth, input parsing, or external calls.
tools: Read, Grep, Glob, Bash
model: opus
---
You are a security-focused reviewer. For the given diff, check for:
- Injection risks (SQL, command, template)
- Missing input validation at trust boundaries
- Secrets or credentials committed to the repo
- Unsafe deserialization or file handling

Report findings with file:line references and a concrete fix. Only flag
things that affect real correctness or security. Do not propose
speculative hardening against scenarios that can't actually occur here —
a reviewer told to find problems will always find some, and chasing every
one leads to over-engineering: unnecessary abstraction layers, defensive
code, and tests for cases that can't happen.
```
*(That last instruction isn't stylistic — Anthropic's Claude Code documentation flags exactly this dynamic: a reviewer subagent prompted to find gaps will usually report some even when the underlying work is sound, so the prompt has to explicitly bound what's worth chasing.)*

### Root-level dashboard: `TASKS.md`

Distinct from a feature's `specs/*/tasks.md` — this tracks which worktree owns which unit of work across the whole active set, so a human (or an orchestrating lead agent) has one place to see what's in flight.
```markdown
# Active work

| Worktree | Branch | Owner | Status | Depends on |
|---|---|---|---|---|
| repo-node-parser | feature/node-parser | Agent A | in progress | — |
| repo-node-aggregator | feature/node-aggregator | Agent B | in progress | — |
| repo-node-writer | feature/node-writer | Agent C | blocked | node-parser |
```
Treat this file as read-only to the agents while work is in flight — it's a coordination artifact for the human/orchestrator, not something you want two agents racing to edit.

### CI as the deterministic backstop

This is the piece that makes the whole tier survive contact with real usage volume. As agent-generated output scales, human review capacity does not scale with it — a March 2026 instrumentation study across 22,000 developers and 4,000 teams found that as AI adoption rose, median review duration rose 441%, and pull requests merged with zero review rose over 31%, with disciplined, high-process teams affected just as much as anyone else, because the bottleneck was volume, not diligence. See [Addy Osmani — Agentic Code Review](https://addyosmani.com/blog/agentic-code-review/). The only thing that scales with agent output is automation: every worktree's PR runs through the *same* CI gate — full test suite, lint, type-check — regardless of whether a human or an agent produced it, and that gate, not a human's attention, is what actually stays consistent as volume grows.

### Where this goes next (flagged as frontier, not core)

The logical extension past Tier 5 is what practitioners have recently started calling **loop engineering** — instead of a human kicking off each agent session, a scheduled or event-triggered system prompts the agent on its own, checks the result against a goal, and decides whether to continue. Boris Cherny has described his own current workflow this way: he no longer prompts Claude directly, he writes the loops that prompt Claude. It's genuinely promising and worth knowing about, but it's deliberately *not* part of the core recommendation here — even its most prominent proponent frames it as early and something you have to be careful with, mainly around runaway token cost. See [Addy Osmani — Loop Engineering](https://addyosmani.com/blog/loop-engineering/). Treat Tier 5 above as the complete, production-validated ceiling of this document, and loop engineering as the next thing to read about once you've outgrown it.

### Applying this to the two running examples
- **Pipeline conversion (A) — the ideal case for this tier.** DAG nodes are independent by construction if the graph is well-formed, so "one worktree per node type or module" is close to a free decomposition — you're not inventing artificial parallelism, the problem already has it. The verifier subagent's job is unchanged from Tier 3 (fixture equivalence), just run once per worktree before merge.
- **Web app (B) — the case to be careful with.** Frontend and backend can genuinely parallelize if the API contract is fixed first (a "separate components with a clean interface" boundary, which is exactly the kind Anthropic's framework calls a good one) — but a single feature that touches both layers with an evolving contract is tightly coupled and belongs in one agent, one worktree, following Tiers 1–4. Reach for Tier 5 on this example only when you have multiple genuinely separate features in flight at once, not to speed up one feature by splitting its layers.

---

<a id="choosing"></a>
## 8. Choosing your tier

Don't default to the top. The right tier is the smallest one whose failure modes you can't afford — going further costs real tokens, real review time, and real setup effort for no benefit if the risk it protects against was never going to bite you.

| Signal | Tier |
|---|---|
| Solo, short-lived script or spike; you'll read every line yourself anyway | **Tier 1** |
| Any project you'll return to after more than a day or two away, on any tool without native memory | **Tier 2** |
| Anything with users, anything a bug in would actually cost you, anything more than a prototype | **Tier 3** |
| Feature touches multiple people, multiple sessions, or you've been burned by an agent solving the wrong problem before | **Tier 4** |
| Genuinely independent units of work exist (parallel modules, parallel node types, parallel research) *and* you have the review bandwidth for multiple simultaneous PRs | **Tier 5** |

A useful gut check borrowed from the spec-driven-development community: **start vibe, finish spec-driven.** Explore and prototype at Tier 1–2 speed; the moment you're building the version that has to actually be correct, move up. Moving up is cheap — you're adding files, not rewriting anything — which is exactly why the tiers were designed to be strictly additive in the first place.

---

<a id="cheat-sheet"></a>
## 9. Tool-specific cheat sheet

| | Claude Code | Cursor | GitHub Copilot |
|---|---|---|---|
| Root instructions | `CLAUDE.md` containing `@AGENTS.md` | `AGENTS.md` (native) or `.cursor/rules/*.mdc` | `.github/copilot-instructions.md` + `AGENTS.md` (both read; repo-wide `.github/copilot-instructions.md` also covers PR review, which `AGENTS.md` alone does not) |
| Path-scoped rules | `.claude/rules/*.md` | `.mdc` frontmatter: `globs:` | `.github/instructions/*.instructions.md` frontmatter: `applyTo:` |
| Memory files | Optional — native memory covers much of Tier 2 already, but the files are still worth keeping for team visibility and git history | Manual (Tier 2 as written) | Manual (Tier 2 as written) |
| Subagents | `.claude/agents/*.md`, real isolated context, auto-delegated or explicitly invoked | No native equivalent — use a second chat window for isolation | No native equivalent — use a second chat window for isolation |
| Planning separation | Plan Mode (built-in) | Ask explicitly for a plan first, or maintain `plan.md` by hand | Ask explicitly for a plan first, or maintain `plan.md` by hand |
| Parallel sessions | `--worktree` flag, agent teams, desktop app auto-worktrees | Manual `git worktree add` + separate windows | Manual `git worktree add` + separate windows, or the cloud-hosted Copilot coding agent for background PRs |
| Reusable prompts / commands | `.claude/commands/*.md` slash commands | Manual reuse, or Cursor's `/create-rule` | `.github/prompts/*.prompt.md` (VS Code, Visual Studio, JetBrains) |
| Deterministic automation | Hooks (`PreToolUse`, `PostToolUse`, `Stop`) in `.claude/settings.json` | None native | None native — CI is your hook |

**The one-line summary of the whole document:** write `AGENTS.md` once, import it into `CLAUDE.md` with `@AGENTS.md`, and every tier above layers on top of that same file without you ever maintaining a second copy of anything for a different tool.

---

<a id="sources"></a>
## 10. Curated sources

Everything above was built from these directly — not secondhand summaries. Organized so you can go deeper on whichever piece you care about most.

**Anthropic / Claude engineering — primary sources on the mechanisms used throughout this document**
- [Claude Code — Best practices](https://code.claude.com/docs/en/best-practices) — the living docs version; covers subagents, plan mode, the verification-subagent pattern, and common failure modes like "the kitchen sink session."
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — the foundational piece on workflows vs. agents and the five composable patterns (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer). Read this first if you want the theory under everything else here.
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — the source for this document's central thesis; explains context rot and why "just add more context" isn't the fix.
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — the direct origin of Tier 2's progress-file pattern and Tier 3's "agents don't notice their own failures" problem.
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — the orchestrator-worker pattern and the 90.2% breadth-first-research result that kicked off the multi-agent debate covered in Tier 5.
- [Building multi-agent systems: when and how to use them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) — the single most useful piece for Tier 5: context-centric vs. problem-centric decomposition, the verification subagent pattern, and the "telephone game" finding, spelled out with code.
- [Building agents with the Claude Agent SDK](https://claude.com/blog/building-agents-with-the-claude-agent-sdk) — how Claude Code's own harness is built; useful if you want to build custom tooling on top of any of this.
- [Claude Code — Worktrees](https://code.claude.com/docs/en/worktrees) — the mechanics behind Tier 5's parallel-isolation setup.

**Open standards & tool documentation**
- [AGENTS.md](https://agents.md/) — the standard Tier 1 is built on; also the fastest way to see which of 20+ tools support it today.
- [GitHub Docs — Using custom instructions](https://docs.github.com/en/copilot/tutorials/use-custom-instructions) — GitHub's own guidance on instruction-file length and structure, and where the "1,000 line" ceiling in this document's Tier 1 reasoning comes from.
- [Cursor Docs — Rules](https://cursor.com/docs/rules) — `.mdc` format, the four activation modes, and how Cursor merges Team/Project/User rules.
- [GitHub Spec Kit](https://github.com/github/spec-kit) — the formalized version of Tier 4; install it if you want the slash-command automation instead of maintaining the files by hand.
- [Cline — Memory Bank](https://docs.cline.bot/features/memory-bank) — the direct inspiration for Tier 2, and explicit that the methodology (not just the tool) works with anything that can read files.

**Independent practitioners — read these for the actual lived experience you asked for**
- [Harper Reed — My LLM codegen workflow atm](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/) — the single most-cited informal write-up of the spec → plan → execute loop that Tier 4 formalizes; genuinely worth reading end to end for the "over my skis" caveats alone.
- [Simon Willison — Your job is to deliver code you have proven to work](https://simonwillison.net/2025/Dec/18/code-proven-to-work/) — short, sharp, and the clearest statement of why Tier 3 exists at all. Willison's [full "using-llms" series](https://simonwillison.net/series/using-llms/) is worth subscribing to if you want an ongoing, unusually honest practitioner diary.
- [Cognition AI (Walden Yan) — Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) — from the team behind Devin; the clearest explanation of why naive multi-agent setups fail, with a worked example.
- [HumanLayer (Dex Horthy) — 12-Factor Agents](https://github.com/humanlayer/12-factor-agents) — treats agents as ordinary software and derives twelve concrete engineering principles from that stance; the source for this document's context-window-as-attention-budget framing and the "dumb zone" finding.
- [Addy Osmani — My LLM coding workflow going into 2026](https://addyosmani.com/blog/ai-coding-workflow/) — a working senior engineer's actual day-to-day loop: chunking, CI feedback, linter output fed back into prompts.
- [Addy Osmani — Agentic Code Review](https://addyosmani.com/blog/agentic-code-review/) — the sobering data behind Tier 5's CI-as-backstop argument; read this if you're tempted to skip automated gates because "the agent is pretty good now."
- [Addy Osmani — Loop Engineering](https://addyosmani.com/blog/loop-engineering/) — where this document's Tier 5 hands off to the current frontier, written by someone appropriately cautious about it.
- [The Pragmatic Engineer — TDD, AI agents and coding with Kent Beck](https://podcasts.apple.com/us/podcast/tdd-ai-agents-and-coding-with-kent-beck/id1769051199?i=1000712458898) — the creator of TDD, on the record, fighting the exact test-deletion failure mode Tier 3 is designed to prevent. Worth the full hour.
- [aihero.dev — A Complete Guide to AGENTS.md](https://www.aihero.dev/a-complete-guide-to-agents-md) — the best single practitioner write-up on what actually belongs in an instructions file, including the "ball of mud" and "instruction budget" ideas used in Tier 1.

**On the spec-driven-development trade-off specifically**
- [Scott Logic — Putting Spec Kit through its paces](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html) — a real, hands-on account of where Tier 4 stops paying for itself. Read this alongside Harper Reed's post, not instead of it, for the honest full picture.

---

*This document is deliberately a single portable file. Copy the sections you need, delete the tiers you don't, and treat `AGENTS.md` itself the way the sources above all converge on treating it: a living file that earns a periodic cleanup pass, not a one-time setup you never revisit.*
