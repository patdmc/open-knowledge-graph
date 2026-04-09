# What if Cancer Is Rational?

## A framework from seven years of watching treatments fail

---

Ten years ago, my wife was diagnosed with HR/PR+ breast cancer. We had just moved across the country, with a 2 year old and a newborn, to start a new job after the startup I was working at was acquired. A mammogram, then a biopsy, then reality. Our baby had never had formula before, but we needed to stop breastfeeding before treatment started. After staying up all night with him crying, he finally gave up and drank the formula. We thought this was going to be a 6 month ordeal. And it was... at least we thought it was.

A year after being declared cancer free, a new pain in my wife's neck led to an X-ray and a stage 4 diagnosis. The cancer had spread to vertebrae in her neck and she needed emergency surgery just to stabilize the bone. The first surgeon refused to do it. It was too risky. The second surgeon performed the kyphoplasty and we could momentarily take a breath. What followed was 7 years of treatments failing, trial drugs, proton therapy, and often driving to Houston, Dallas, or San Antonio from Austin, TX to get treatment. The doctors we worked with were generous with their time and their explanations. They sat with us, walked us through the mechanisms of each treatment, answered every question we had. My wife, Tricia Duryee, was a nationally known tech journalist and a member of the University of Oregon Journalism Hall of Fame. She was prepared. I spent time reading papers, not fully understanding them, but I had read enough papers in my life to get the gist and understand the theory of the case of each drug.

Treatment after treatment worked. And then stopped working.

Every time a drug failed, the explanation was the same: "The cancer evolved resistance." Our doctors explained this clearly and patiently. But I am a software engineer, someone who needs to understand the theory of the case, and the explanation never satisfied me. Not because the doctors were wrong, but because "cancer evolves resistance" is a description, not an explanation. It tells you what happens but not why it's so reliable, so fast, so seemingly inevitable.

My wife carried a rare RAD51C variant. One that was initially classified as a variant of uncertain significance, but was later confirmed as a high-risk driver, similar to BRCA. We watched the classification change in real time, from "possibly relevant" to "this is why." Knowing the genetic driver helped guide treatment choices, but it didn't answer the deeper question: why does the cancer always find a way around the drug?

We knew that immunotherapy treatments were producing durable cures in some cancers. Not in HR+ breast cancer, not for us. It was obvious it worked *differently*. It wasn't just killing cancer cells. The body was learning to treat the cancer cells for what they were: a serious problem that needed to be attacked by the immune system. And we knew the mechanism - blocking PD-L1, releasing the brakes on T-cells - but I kept asking a question they couldn't quite answer in structural terms: *why* did restoring immune surveillance work when killing the cells directly didn't?

My wife died. The question didn't.

---

## What I eventually understood

I am a principal engineer at a large technology company. As I have been building agentic systems, I have been developing a formal mathematical framework. A series of proofs about the architecture of intelligence, starting from a single constraint: bounded active context, the hard limit on how much any system can hold in mind at once.

The framework isn't about cancer. It is about knowledge representation, learning, uncertainty, and intelligence. But when I follow the logic far enough, it arrives at cancer on its own.

---

## Your cells are not solo operators

A cell in your body is not an independent organism. It is a participant in an organizational hierarchy: cell, tissue, organ, organism. Each level coordinates the level below. Your breast epithelial cells don't need to independently figure out what the organism's hormonal state is. They receive that information via estrogen receptors, paracrine signaling from neighboring cells, and direct cell-to-cell communication through gap junctions. The tissue tells the cell what the organism needs.

I call this the *escalation chain*. It's a term from engineering: when a problem exceeds what one level can handle, it escalates to the next level up. Your cells handle cell level problems. Tissue level problems escalate to tissue. Organ level problems escalate to the organ. And so on.

The escalation chain *extends what the cell can effectively know*. A cell connected to its tissue context has access to far more information than a cell operating alone. It doesn't need to represent the whole organism. It just needs to listen to the level above.

## What if cancer is what happens when the chain breaks?

When a cell loses its connection to the tissue: through mutation (like a BRCA or RAD51C variant that compromises DNA repair and destabilizes signaling), through inflammation, through physical displacement, through epigenetic silencing. It loses access to all that organizational information. Its effective world shrinks to what a single cell can observe: local nutrients, immediate mechanical environment, intracellular state.

The cell is not broken. The cell is now operating rationally within a much smaller world. And when your world shrinks to cell level, the *rational* things to do are:

- **Proliferate.** Cell division requires only local nutrients. No tissue level permission needed.
- **Switch to glycolysis.** Aerobic glycolysis (the Warburg effect) is the metabolic mode with the fewest dependencies on tissue level infrastructure. You don't need reliable oxygen delivery from a vascular system you can no longer coordinate with.
- **Move.** If local resources are scarce, explore. This is what we call metastasis. It is what happened when my wife's cancer spread to her spine.
- **Ignore immune signals.** The surface markers that identify you to the immune system are maintained by tissue level signaling. Without that signaling, they decay. The cell doesn't *decide* to evade the immune system. It just stops hearing the signals that kept it visible.

These are the *hallmarks of cancer*. The universal features that Hanahan and Weinberg catalogued in their landmark papers. Every cancer, from every tissue, converges on the same behaviors.

The standard explanation is that these are capabilities the cancer "acquires" through random mutation. The atavistic theory offers a different view: cancer is a reversion to ancient, single celled behavior: an old program reactivated. My framework says they are not acquired, and it is not reversion. They are *released*. They were always in the cell's repertoire, suppressed by tissue level coordination. When the chain breaks, they emerge. Not because an ancient program was reactivated, but because they are the only rational actions available to a cell operating alone.

## This is why treatments fail the way they do

Sit in enough oncology appointments, and I sat in hundreds, and you learn the pattern. Chemotherapy works, then it doesn't. Targeted therapy works, then it doesn't. A PARP inhibitor exploits the RAD51C vulnerability, until the cancer finds a workaround. The cancer "evolves resistance."

Our doctors explained each failure honestly and in detail. I am grateful for that. But now I can offer a structural reason *why* resistance is the default rather than the exception, and it's not because cancer is uniquely clever. It's because you are fighting an entity that is *optimizing correctly within its own constraints.*

Chemotherapy kills dividing cells. But the cancer cell is learning from every signal in its environment, including the drug. The surviving cells have already been selected for whatever trait lets them survive: slower division, drug efflux pumps, enhanced DNA repair. All of these are cell level capabilities. The cell doesn't need tissue level coordination to develop resistance. Resistance is *within the cell's action space.*

You are imposing selection pressure on a locally rational optimizer. Of course it adapts.

## This is why immunotherapy is different

Immunotherapy doesn't kill cancer cells. It restores a *relationship.*

When a checkpoint inhibitor blocks PD-L1 or CTLA-4, it re-establishes the connection between the cancer cell and the organism's immune surveillance system. It restores the escalation chain at the organism level. The immune system can see the tumor again. The organizational relationship is repaired.

This is why immunotherapy produces durable responses. Sometimes lasting years, sometimes permanent, in a way that chemotherapy almost never does. Chemotherapy changes the optimization landscape without changing the optimization regime. Immunotherapy changes the regime itself: the cell is no longer operating alone.

And this is why differentiation therapy - like ATRA in acute promyelocytic leukemia, which has a greater than 90% cure rate - is the most successful cancer treatment ever developed. ATRA doesn't kill the cancer cells. It *makes them normal cells again* by restoring their tissue level identity. It reconnects the chain at the most fundamental level.

The framework predicts that therapies which restore the chain should systematically outperform therapies that don't. That prediction aligns with where oncology is already heading: toward immunotherapy, toward differentiation therapy, toward microenvironment normalization. But for the most part, without a unified theoretical reason for *why* that direction is right.

## What this meant for us

Immunotherapy was out of reach for my wife's HR+ breast cancer. HR+ breast cancers are considered immunologically "cold". They don't provoke enough of an immune response for checkpoint inhibitors to amplify. In the language of the framework, the chain severance in HR+ breast cancer occurs primarily at the tissue level (loss of normal estrogen mediated tissue coordination), not at the organism immune level. Restoring immune surveillance doesn't help if the fundamental break is between the cell and its tissue identity.

What was needed was something closer to differentiation therapy. A way to restore the cell's connection to its breast tissue context, to re-establish the hormonal coordination that keeps breast epithelial cells in their proper role. Endocrine therapy (tamoxifen, aromatase inhibitors) attempts something like this by modulating the hormonal signaling channel. We supplemented baseline treatment with different aromatase inhibitors over the years, with varying success. But it addresses only one coupling channel, and when the cancer found routes around it, there was no therapy that could restore the broader tissue level relationship.

The framework says where to look: not at the cell, but at the coupling. Not at the mutation, but at the relationship that was lost. The question is not "how do we kill this cell?" but "how do we restore this cell's connection to the tissue it belongs to?"

## What this isn't

I am not a cancer researcher. I am not a biologist. I am a software engineer who built a mathematical framework about bounded cognition and followed the logic to a place I didn't expect.

This paper is speculative. It is a theoretical application of a formal framework to a biological domain. It does not prove anything about cancer. It generates structural predictions that happen to align with where clinical oncology is already moving, and it offers a formal reason for *why* that direction works.

I owe a debt to the oncologists, researchers, and trial coordinators who treated my wife with skill and compassion, and who took the time to explain their reasoning to a non-expert who asked too many questions. Their explanations are the foundation on which this framework was eventually built. I want to help these wonderful people, who know that every one of their patients is going to die but still show up for the fight every day. They are willing to try their hardest and fail over and over again. Knowing they will fail, knowing they will see the fear and tragedy first hand, and still showing up, still fighting. If anything, this is an attempt to give their clinical intuitions - that immunotherapy works differently, that differentiation therapy is underexplored, that cancer is fundamentally an organizational disease - a structural justification.

The testable predictions are in the paper. The proofs are in the paper. The engagement with existing cancer theory - somatic mutation theory, atavistic theory, tissue organization field theory - is in the paper. I offer it in the spirit of someone who needed to understand what happened, and who built the only tool he knew how to build: a formal model.

If you are a cancer researcher and this framework interests you, I am looking for collaborators who can evaluate whether the structural reasoning maps faithfully onto the biology. The predictions are specific enough to test. I would like to see them tested.

[Paper 6: Cancer as Escalation Chain Severance Under Bounded Context — SSRN link]

[Full five-paper series: The Architecture of Intelligence from Bounded Active Context — SSRN link]

---

*Patrick D. McCarthy is a principal engineer who has spent much of his career building and sharing institutional knowledge for his teams. In an effort to make that institutional knowledge actionable and explicit, a broader theory of the case emerged. These papers are an effort to describe the root of what was discovered in that process. If the foundations are solid, the implications are broad, and cancer treatment is something that will never completely leave my mind. I think about all the families that are dealing with what we dealt with and want to help in the best way that I can.*
