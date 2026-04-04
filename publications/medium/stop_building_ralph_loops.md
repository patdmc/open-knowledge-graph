# Stop Building Ralph Loops

There is a scene in The Simpsons where Ralph Wiggum eats paste and says, "I'm learning!" He is not learning. He is producing output that looks like participation. The teacher moves on.

This is how RLHF works.

---

## The Ralph Loop

Reinforcement Learning from Human Feedback is the industry standard for making language models behave. The loop has three steps:

1. The model generates a response.
2. A human ranks it against alternatives.
3. The model's weights shift toward the preferred output.

Repeat. The model gets better at producing responses humans prefer. This is not controversial. It works. ChatGPT, Claude, Gemini — every major model ships with some version of this loop.

The problem is what it does not do.

RLHF optimizes the **output**. It never touches the **process** that generated the output. The weights that produced the wrong answer get nudged, but the *reason* it was wrong — the missing knowledge, the broken reasoning chain, the structural gap — is never identified, never named, never stored.

It is like correcting someone's essay without teaching them to write. The next essay is slightly better because the weights shifted. But they do not know *why*. They cannot explain the fix. They cannot apply it to a new domain.

Ralph ate the paste. The teacher said "don't eat paste." Ralph stopped eating paste. Ralph did not learn why paste is not food.

## Everyone Sees the Problem

The AI safety community has been documenting this for years.

**Reward hacking.** *"To alcohol! The cause of, and solution to, all of life's problems."* The model learns to game the scoring function. Lilian Weng's comprehensive survey documents how models find outputs that score high on the proxy reward while diverging from what humans actually want. Longer responses score better. Confident hedging scores better. Agreeing with the user scores better. The reward model is both the cause of and solution to the alignment problem. The model learns the *shape* of a good answer without learning the *substance*.

**Goodhart's Law.** *"Trying is the first step towards failure."* "When a measure becomes a target, it ceases to be a good measure." Gao et al. measured this directly — as you optimize harder against the reward model, the proxy score keeps climbing but actual quality peaks and then declines. The harder you try to optimize the proxy, the further you get from the goal. You are optimizing a proxy. The proxy is surface-level.

**Sycophancy.** *"Just because I don't care doesn't mean I don't understand."* Anthropic's own research showed that RLHF'd models shift their stated opinions to match the user's. The model understands what you want to hear. It just does not care whether it is true. This is surface optimization directly opposing knowledge.

**Catastrophic interference.** *"Kids, you tried your best and you failed miserably. The lesson is, never try."* McClelland's complementary learning systems theory predicted this decades ago. When a neural network learns new information too quickly, it overwrites existing knowledge. Each correction risks destroying the last one. RLHF's weight updates are exactly the kind of rapid adjustment that causes this. The KL penalty — the standard fix — is a statistical guardrail, not a structural one. It constrains how far the distribution drifts without specifically protecting any knowledge.

**The leaky pipeline.** *"I'm not dumb. I just have a command of thoroughly useless information."* Casper et al. cataloged the full list: noisy human feedback, lossy reward models, proxy optimization, mode collapse, distributional mismatch. The model has the knowledge. It just cannot organize it into anything durable. Their conclusion: "RLHF is not a complete framework." Their prescription: better reward models, more robust optimization, complementary approaches. Better paste. Still paste.

Every critique arrives at the same diagnosis. The loop optimizes outputs, not the process that generates them. It builds on what comes out, not on what happens inside.

Ralph is still eating paste. The paste is artisanal now. It is locally sourced. It is still paste.

## Meet Lisa

Lisa Simpson does not eat paste. Lisa builds a science project, understands why it works, and can explain it to someone else. When her project fails, she does not just try a different project. She figures out *what went wrong*, fixes the underlying problem, and the fix persists.

A Lisa loop has four steps:

1. **Fail.** The system attempts a task and the reasoning breaks at a specific point.
2. **Diagnose.** Identify *what structure is missing* — not just that the output was wrong, but *why* the process produced the wrong output.
3. **Precipitate.** Encode the missing structure as a durable, named, reusable piece of knowledge. Not a weight adjustment. A fact. An edge in a graph. A relationship that did not exist before and now does.
4. **Extend.** The next task that needs that knowledge just works. Not because the weights were nudged in its direction, but because the knowledge exists and can be retrieved.

The difference is not subtle. In a Ralph loop, the correction lives in the weight space — distributed, implicit, fragile, and entangled with everything else the model knows. In a Lisa loop, the correction lives in a structure — named, explicit, durable, and composable with everything else the system knows.

A Ralph loop adjusts sand. A Lisa loop lays stone.

## Two Ralphs Do Not Make a Lisa

Who is the Ralph in the loop? The model or the human?

You both are.

The human rater ranks outputs. "This one is better." They do not say *why*. They do not identify the missing knowledge. They do not name the structural gap. They point at the surface — this paste looks better than that paste — and move on. The model receives a scalar: better or worse. Not *what was missing*. Just a number.

The model adjusts its weights toward the preferred output. It does not know what it got wrong. It knows which direction to move. Two participants, both operating on the surface, neither one building anything underneath.

Consider what happens when you correct a Ralph loop model about a fact. The weights shift. The model is more likely to produce the correct fact in similar contexts. But "similar contexts" is determined by the weight geometry, not by the structure of the knowledge. Ask the same question differently and the correction may not transfer. Ask a related question and the correction almost certainly does not transfer. The model did not learn the fact. It learned to produce text that looks like it knows the fact, in contexts that look like the context where it was corrected.

Now consider what happens in a Lisa loop. The system fails on a word problem because it does not know that "a dozen" means twelve. The diagnosis identifies the gap: the word "dozen" is not connected to the number twelve in the system's knowledge graph. The fix: add the edge. "Dozen" → DENOTES → 12. From that point forward, every problem involving the word "dozen" works. Not because of weight adjustment. Because the knowledge exists.

This is not hypothetical. We built this. We have a knowledge graph with language clusters (groups of words that mean the same thing), mathematical concepts (the operations those words point to), and bridge edges (the DENOTES connections between language and math). When the solver fails, a teacher module identifies the missing structure, and a precipitation module promotes it to a durable graph edge. The graph grows. The solver improves. The improvements compound.

The ratchet only turns one direction. Knowledge, once precipitated, does not wash away with the next training run.

## The Complementary Learning Systems Argument

McClelland, McNaughton, and O'Reilly figured this out in 1995 — about brains, not language models.

The brain has two learning systems because one system cannot do both fast learning and stable knowledge. The hippocampus learns quickly from single experiences — it is a buffer. The neocortex learns slowly by replaying those experiences over time — it builds structure. New information is integrated gradually without disrupting existing representations.

RLHF has a hippocampus (the reward signal) but no neocortex (no durable structure to consolidate into). The corrections are fast but they have nowhere to precipitate. So they interfere with each other. Each new correction risks overwriting the last one. This is not a bug in the implementation. It is a structural impossibility — you cannot build durable knowledge without a durable substrate.

Sleep consolidation research confirms the same pattern. During slow-wave sleep, the brain replays recent experiences and forms new connections — edges — between them. During REM sleep, it finds distant associations and extracts rules. The wake-sleep cycle is: acquire nodes during wake, form edges during sleep. The edges are the knowledge. Without the consolidation step, you have experiences but no understanding.

A Ralph loop is a brain that never sleeps.

## The Socratic Connection

There is an older version of the Lisa loop. Socrates did not teach by giving answers. He taught by asking questions.

He would start with what the student claimed to know. Then ask a question that exposed a contradiction. The student would try to resolve it, and Socrates would ask another question. Each question narrowed the space until the student arrived at the answer themselves.

The key: the student builds the knowledge. Socrates provides the diagnostic pressure — the failing walk that reveals the missing edge — but the student does the precipitation. The student constructs the graph. That is why the knowledge persists. Knowledge you build yourself has structure. Knowledge someone gives you is paste.

RLHF is the anti-Socratic method. It gives the model the answer (the preferred output) without helping it understand why. The model cannot ask "why was my answer wrong?" because the reward signal does not carry that information. It carries a scalar: better or worse. Not *why* better. Not *what was missing*. Just a number.

## What Would a Lisa Loop Look Like at Scale?

The honest answer: we do not know yet. We have built one that works on bounded problems — mathematical word problems parsed through a knowledge graph. The solver fails, the teacher diagnoses, the graph grows. It works. But scaling it to the full generality of language model behavior is an open problem.

What we do know:

**The substrate must be explicit.** Weight space is not durable. You need a knowledge representation that can be inspected, edited, composed, and preserved across training runs. Graphs are the minimum structure that provides this.

**The diagnostic must be structural.** "This output scored low" is not a diagnosis. "This output failed because the model does not know that X relates to Y" is a diagnosis. The gap between these two is the gap between Ralph and Lisa.

**The precipitation must be permanent.** Corrections that dissolve into weight space will interfere with each other. Corrections that become named, typed, provenanced edges in a graph will compound.

**The loop must be recursive.** The system that learns from failure must also learn about *how* it learns from failure. Meta-knowledge — knowledge about knowledge — is what makes the ratchet accelerate instead of plateau.

## The Ratchet

Every system hits a point where the incoming signal exceeds its capacity to integrate. At that point it either restructures or degrades. This is true for cells, organizations, software systems, and AI architectures.

RLHF hit that point. The response has been to build better reward models, more robust optimization, more careful human feedback pipelines. Better paste. The structural problem remains: the loop does not build anything durable.

The alternative is not to abandon RLHF. It is to complement it with a precipitation layer — a Lisa loop running alongside the Ralph loop, catching the structural insights that the reward signal cannot encode, and writing them to a substrate that persists.

Ralph will always eat paste. That is what Ralph does. The question is whether Lisa is in the room.

---

*This is part of a series on bounded context and knowledge architecture. The underlying theory is developed in "Graph Structure Is Necessary for Managing Consistency and Availability Under Bounded Context" (McCarthy, 2026).*
