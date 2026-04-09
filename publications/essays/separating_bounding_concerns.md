# Separating and Bounding Concerns: The Reason All Design Patterns Work, and How to Apply This to Agentic Code

---

Every design pattern you've ever used is doing the same thing. Strategy separates *what* to do from *when* to do it. Observer separates the event from the response. Repository separates persistence from domain logic. Microservices separate deployment boundaries along concern boundaries. CQRS separates reads from writes. Each pattern draws a line: on this side, you don't need to know what's on that side. Strip away the specifics and they are all doing the same thing: separating concerns and bounding how much any single component needs to know.

As a principal engineer, my job is to define problems, design stable architectures to solve them with the least risk, make a plan to get there, and - almost most importantly - accurately estimate the total effort needed to deliver. Every one of those tasks is an exercise in separating and bounding concerns. You can't estimate what you can't decompose. You can't design what you haven't bounded. You can't plan what you haven't separated.

This post is about why that pattern works, why it's not a preference but required, and why it matters more than ever now that we're building agentic systems with LLMs.

---

## The scaling problem I actually have

My last role was leading a team of 70 engineers as the sole principal, working across dozens of teams in different organizations. Very quickly, it became obvious that what was manageable at lower ratios is impossible at scale: **I cannot feed them.**

Not in the literal sense. In the specification sense. A team of 70 engineers needs enough well-defined work to stay productive. That work needs to be decomposed into pieces that are independently actionable: bounded in scope, clear in interface, testable in isolation. Someone has to do that decomposition.

If that someone is only me, I am the bottleneck. My context is bounded. I can hold a finite number of concerns in my head at once. I can reason about a finite number of interfaces simultaneously. If every architectural decision routes through me, the team's throughput is capped by my cognitive bandwidth.

The solution is not to think harder. The solution is factorization.

I have to teach others how to do what I do: how to take an ambiguous problem, identify its concerns, separate them, bound them, and define the interfaces between them. Not because delegation is nice. Because it is *structurally necessary*. My bounded context makes it necessary.

And here's the thing: it's good for them. Engineers rise to the occasion. They are capable and sometimes don't get as many chances as they should. Giving someone the responsibility to decompose a problem, not just implement a solution, but *define* the problem, is one of the most meaningful growth opportunities you can offer. Most engineers are ready for it long before anyone asks.

My job as a principal engineer is not to be the smartest person in the room. It is to be the person who teaches decomposition. Who shows others how to separate and bound concerns so that the organization can scale beyond what any one person can hold in their head.

---

## How I found this

In January I was assigned to support the migration of 700 applications from one deployment platform to another. Two problems. First, no one could quantify the real prerequisites for migration. Second, 700 applications across 50 teams is more state than anyone can track.

I started by pulling every GitHub repo and its library dependencies. My company has over 4000 live applications, and many had already done what these 700 needed to do: upgrade the JVM, move to the new platform, update frameworks. The answer was distributed across thousands of repos, their commits, their Jira tickets, their incidents. Upgrading Java 17 alone had over 40 production incidents and a similar number of post-deployment fixes. The empirical record of what works, what breaks, and what order to do things in was already there. It just wasn't structured. Our RCA process is rigorous: the goal is to prevent incidents from happening again. But distributing that learning across teams always fell flat.

I built skills: agent capabilities for specific tasks like dependency resolution. Then meta-skills: actions whose job was to propagate learnings from one skill to all the others. I ran a learning loop, applying ML principles to agentic code. Run the skill, evaluate the result, update the skill, run it again.

But the skills shared so much. The same dependency patterns, the same upgrade sequences, the same failure modes. I was paying for the same knowledge in every skill independently. Most skills at my company used a third-party search tool to query our Confluence, Jira, and GitHub, then passed the results to Claude to interpret. Every skill, every run, paying two third parties to tell us what we already know. Who pays two vendors to look up their own institutional knowledge every time they ask a question? A business that hasn't factored its own knowledge.

So I factored the shared structure out. I extracted the common knowledge into named nodes with typed relationships, and called it the knowledge graph: the *what we know* that justifies the *how we act*.

The codebase shrank by 50%. Not through optimization. Through factoring. Half the code had been redundant shared structure that hadn't been extracted.

Then the learning loop forced a full re-evaluation of the entire graph. When one node changed, every skill that referenced it re-evaluated. But it went further: unrelated skills were evaluated to see if the new subskills applied to them too. One learning propagated everywhere it mattered, including places I hadn't thought to look. The graph tightened with every pass.

Two more things fell out of this. First, I realized that every uncertainty boundary needed to be wrapped in an agent. External scripts, API calls, anything where the outcome wasn't fully predictable. The agent hierarchy *is* the escalation chain. Agents don't exist to do things. They exist to bound uncertainty. And when you can define the bounds, you can shrink them.

Second, once the learning loop tightened the uncertainty bounds far enough, some actions didn't need agents anymore. They became scripts. Deterministic code precipitated out of the agent layer when confidence was high enough. Large swaths of what Claude was doing turned out to be scriptable. What Claude does when it encounters an error is useful to learn from, but it never gets less uncertain if you don't feed the learning back into the knowledge graph.

I cut costs by an order of magnitude. And then I asked: why does this work?

---

## Why this isn't a preference

At some point I started asking: why does this work? Not "why is separation of concerns a good idea?" Every engineer knows that. But *why is it the only thing that works at scale?* Why does every successful architecture, every lasting design pattern, every organization that ships reliably, converge on the same structural move?

I spent a year building the formal answer. It starts from a single constraint - **bounded active context** - the hard limit on how many things any system can hold in working memory simultaneously.

This limit is real for humans (you can juggle maybe 7 things), for organizations (a team can track a finite number of priorities), for CPUs (finite registers, finite cache), and for LLMs (finite context window). It is not a limitation to be overcome. It is a structural constraint that *determines* the architecture of any system that operates under it.

Here is what I proved:

**Any system that accumulates information under bounded context and must preserve that information necessarily maintains a directed graph with typed edges.**

That's Paper 1 of a five-paper series. The proof comes from two directions.

1. The only operation that creates space in a bounded context without losing information is *factoring*. Factoring is graph construction by another name: extracting shared structure into named nodes with typed relationships.

2. Any system that needs to do sub-linear lookups, contextual retrieval, and selective removal with cascade necessarily encodes directed adjacency; which is a graph regardless of what you call it.

Separation of concerns is factoring. Bounding concerns is respecting the context limit. Every design pattern is an instance of this. They work because they are the *only* structures that preserve information under bounded context. Not the best structures. The *necessary* structures.

---

## What this means for how we organize teams

The team-scaling problem I described - teaching 70 engineers to decompose problems because I can't do it all myself - has a name in the formal framework: **escalation dominance**.

When a team is small, top-down allocation works. The lead architect can hold the whole system in their head, decompose every problem, assign every task. The overhead is manageable because there aren't that many things to track.

But top-down allocation has a cost: the allocator has to maintain a representation of *every* concern in their bounded context simultaneously. As the system grows - more services, more interfaces, more edge cases - the representation overhead grows linearly (or worse). At some point, the overhead itself exceeds the bounded context. The allocator can no longer hold the whole system in their head. Decomposition quality degrades. Estimation accuracy drops. The team slows down and nobody can explain why.

You've seen what this looks like: more meetings, more coordination, more planning, more rigidity. Every decision requires three conversations instead of one. The organization gets expensive and slow, and the response is usually to add more process, which makes it slower.

The alternative is escalation: each level handles what it can and signals upward only when it can't. Junior engineers handle implementation concerns. Senior engineers handle component-level design. Staff engineers handle cross-component architecture. Principal engineers handle system-level concerns. Each level only escalates further up the chain when a design choice breaks the scope of that level.

Escalation achieves O(1) steady-state context cost for the highest level. I don't need to track every implementation decision. I need to track the interfaces and the failure modes. Everything else is handled below me, and it only reaches me when it needs to. And by delegating, I have enough time to engage with ANY engineer that is unsure of what to do. Big and small decisions matter, and you can't make them all. But you can have the flexibility to float to where you are needed.

This is not a management philosophy. It is a mathematical result. I proved (Paper 2) that once the number of active concerns exceeds a crossover point, top-down allocation degrades monotonically and escalation dominates at every subsequent scale. The crossover is permanent: you cannot grow your way back to top-down by hiring smarter leads, because the problem is the bounded context, not the person.

Every organization that has scaled successfully has discovered this empirically. The framework says why.

---

## Now: agentic code

Everything above applies to humans and organizations. But agentic code - systems built on LLMs - makes the same constraints *literal and measurable*.

An LLM has a context window. It is a number. You can look it up. 128k tokens, 200k tokens, whatever the model supports. That is the bounded active context. It is not a metaphor. It is a hard limit on the number of propositions the system can hold simultaneously for inference.

Every problem in agentic code is a bounded-context problem:

- **The agent loses coherence on long tasks.** Its context fills up. Propositions from early in the conversation are displaced by later ones. The graph of what it knows becomes disconnected.
- **The agent can't handle complex multi-step reasoning.** Because the intermediate steps exceed its context. It needs to factor the problem into subproblems; but if it can't represent the factoring itself within context, it fails.
- **The agent hallucinates.** Because it's inferring from an incomplete graph. Propositions that would ground its inference have been displaced by context pressure.

The design pattern for agentic code is the same one that works for human organizations: **separate and bound concerns.**

Concretely:

### 1. Factor into subagents

Don't give one agent a 50-step task. Factor it into bounded subtasks, each within a single agent's effective context. Each subagent handles one concern. The orchestrator handles the interfaces between them. Every uncertainty boundary needs to be bound by an agent: external tools, APIs, MCPs, anything where the input or output is not fully predictable.

This is not a hack. It is the *necessary* architecture under bounded context. The formal result (Paper 1) says that graph structure is the only way to preserve information under bounded context. A multi-agent system where each agent handles a bounded subgraph and communicates results to an orchestrator *is* a graph, with agents as nodes and their communication as typed edges.

### 2. Escalation, not top-down planning

Don't have the orchestrator pre-plan every step. Have each subagent attempt its task and escalate when it encounters something beyond its scope. The orchestrator engages only on escalation.

This is the same escalation dominance result that applies to human teams. A top-down planner that tries to pre-decompose a complex task will hit its own context limit. An escalation-based system where each level handles what it can and signals failure upward achieves O(1) orchestrator context cost.

### 3. Knowledge and action are inseparable

Don't separate "what the agent knows" from "what the agent does" into independent modules with separate optimization. Paper 3 proves that under survival pressure (the system needs to produce correct outputs to be useful), the knowledge base and the action space are inseparable. You cannot optimize them independently.

In practical terms: don't build a retrieval system and an action system and optimize them separately. The retrieval system's value is determined by what actions it enables. The action system's capability is determined by what knowledge it can access. Optimize them together or accept that you're leaving performance on the table.

### 4. The graph is not optional

If your agentic system doesn't have explicit graph structure - if it doesn't maintain typed relationships between propositions, support sub-linear lookups, and handle selective removal with cascade - then it is maintaining an implicit graph in the attention weights and losing information every time context pressure forces displacement.

Make the graph explicit. Knowledge graphs, tool-use graphs, task dependency graphs, conversation state graphs. The substrate doesn't matter. The structure is mandatory.

This is also how you get reuse. A dependency analyzer bound as a reusable subgraph can serve many flows. If every flow does dependency analysis independently, you pay the context cost every time and get inconsistent results. Factor it once, reuse the graph node. This is the same normalization argument that applies to knowledge: shared structure gets extracted into named nodes. That's what graphs are for.

---

## The forever tax

Every company building on LLMs is paying a tax they don't have to pay. Larger context windows cost more per call. More tokens in, more tokens out, higher latency, higher cost. The default architecture, dump everything into context and let the model sort it out, pays the full cost on every call. The same question costs the same whether you've answered it a thousand times or never. Knowledge grows, costs grow, and nothing is reused.

This is the forever tax. It is not a technology problem. It is a factoring failure: the failure to identify what can be reused. Every shared structure that stays buried in context instead of extracted into a named node is cost paid repeatedly for the same work.

Graph structure eliminates it. Instead of scaling the context window, you scale the knowledge graph and retrieve the relevant subgraph per problem. The context window stays bounded. The cost stays flat. The knowledge grows without bound. Organizations that share knowledge efficiently avoid the tax. Organizations that don't, pay it on every call.

---

## The punchline

I spent over 20 years learning to separate and bound concerns as a software engineer. Then I proved *why* it works. The answer is: bounded context makes graph structure necessary. Every design pattern is an instance of this. Every successful organization discovers it empirically. Every agentic system that works at scale implements it, whether the builders know it or not.

The formal proofs are in the papers for anyone who wants to check the math. But the practical insight is simple enough to fit in a sentence:

**Your context is bounded. Factor or fail.**

The three core papers prove the structural results:

[Paper 1: Graph Structure Is Necessary for Information Preservation Under Bounded Context — SSRN link]

[Paper 2: Escalation Dominates Top-Down Allocation Under Bounded Context — SSRN link]

[Paper 3: Knowledge and Action Are Inseparable Under Survival Pressure — SSRN link]

Two further papers derive what follows: gradient descent with encoding permanence (Paper 4) and the structural invariants of intelligence (Paper 5). They extend the framework but depend on the core three.

---

*Patrick D. McCarthy is a principal engineer who has led teams of up to 70 engineers and has spent the past year formalizing the structural constraints that govern system design under bounded context.*
