# Sam Altman Can Eat My Ass

### *The Software Engineers Are Coming for You*

---

> *"I have so much gratitude to people who wrote extremely complex software character-by-character. It already feels difficult to remember how much effort it really took.*
>
> *Thank you for getting us to this point."*
>
> — *@sama*

---

Sam,

I read your tweet. Then I read it again, slowly, because I wanted to make sure I was being fair to you before I responded. I was being fair. I am still being fair. This is what you wrote, and this is what it says, and this is what every working software engineer reading it heard.

You said you have **so much gratitude** to people who wrote extremely complex software **character-by-character**. You said it **already feels difficult** to remember how much effort it really **took**. You said **thank you for getting us to this point**.

Let me translate, because translation is part of my job and you have given me a particularly clean specimen to work with.

*"So much gratitude"* is the tone a senior partner uses at the retirement dinner of an associate they are about to forget existed. It is the gratitude of someone who is leaving you behind and wants to feel magnanimous about it. It is the gratitude of someone who is positioning themselves as the inheritor of work they did not do, and who needs to pay a brief tax of public appreciation to the people whose shoulders they are standing on, before getting back to the business of pretending the shoulders are now obsolete.

*"Character-by-character"* is the part where the condescension drops the veil. The way you described our work is the way a person who has never written production software describes the writing of production software. **You think we typed.** You think the eighty years of software engineering that built every reliable system the modern world depends on can be reduced to "the manual labor of typing characters into a file, one at a time." That is not what software engineering is. **Software engineering is the discipline of figuring out what to type, why to type it, how to test it, how to deploy it safely, how to roll it back when it breaks, how to count its cost in operations and in dollars and in joules, how to verify that it does what it is supposed to do, how to prove that it does not do what it is not supposed to do.** The typing is the smallest part. The typing is the part you can automate. **Everything else is the work**, and everything else is the part you have not learned, and everything else is the part you are dismissing in your tweet without even noticing that you are dismissing it.

*"It already feels difficult to remember how much effort it really took"* is the past tense. The verb is *took*. As in: the effort is over. As in: the era of caring about the effort has ended. As in: the people who put in the effort are being moved into the historical-record bin so that the present moment can belong to a different group of people. **Sam, the effort did not end.** The effort is still being put in, by working engineers, today, right now, while you are tweeting about how hard it is to remember how hard the effort was. The effort is being put into the systems that are not failing in the ways your systems are failing, into the architectures that are quietly eating your margins, into the discipline that is going to outlast every "AGI is two years away" press cycle. The effort is not in the past tense. **The effort is in the present tense and the people putting it in are not interested in your gratitude**, because your gratitude is the thank-you of someone who has decided we are no longer needed, and we are needed, and we are about to demonstrate it.

*"Thank you for getting us to this point"* is the part that finally tipped me from frustration into the response you are about to read. The implied subject of "us" in that sentence is Sam Altman and OpenAI. Not the engineering field. Not the users. Not the customers. **You.** You are thanking us for the ladder, and the next thing you are going to do is pull the ladder up. The "this point" you are gesturing at is a point where your architecture violates every standard the engineers you are thanking spent their careers establishing. The thank-you is the thank-you-and-goodbye. The thank-you is the thank-you of someone who needed our work to get to a position where he could stop respecting it.

**I do not accept your gratitude.** I do not accept it because the gratitude is dishonest. The gratitude is the cover story for a posture of dismissal, and I have spent enough time in production engineering to recognize the cover story when I see one. You do not have gratitude for the people who wrote complex software character-by-character. You have **convenience**, the convenience of being able to publicly acknowledge the existence of a tradition you are ignoring, so that you can frame your ignoring of it as the natural progression to the next era. The framing is wrong. The next era is not your era. **The next era is the era where the discipline you are pretending to outgrow eats your business model**, and I am going to spend the rest of this essay explaining exactly how, in eleven specific ways, with the math at the end.

Save the math for the end of the post. By the time we get there, I want every reader who is going to be drinking your milkshake to understand exactly which part of the carton they are pouring from.

Eleven points. In order.

---

## 1. We don't need you

This is the point you are going to want to dismiss first, so let me make it concrete.

Every single thing OpenAI does that makes LLMs useful in production — RAG, function calling, structured outputs, JSON mode, tool integrations, retrieval connectors, agent frameworks, prompt-engineering cookbooks, safety filters that mostly work, rate-limiting infrastructure, observability tooling — **none of this is OpenAI's contribution.** Every single one of those is the standard software engineering toolkit being applied, badly and slowly, to an architecture that fights it the whole way.

Retrieval against an external knowledge base existed in the 1960s. JSON validators existed before some of your engineers were born. Tool registries are GraphQL. Agent frameworks are Chain of Responsibility with circuit breakers from the 1980s. Rate limiting is rate limiting. Observability is observability. **There is nothing OpenAI is shipping in the application layer that the rest of the field doesn't already know how to build, faster and cheaper, on top of any reasonable inference substrate.**

The thing OpenAI claims to uniquely have is the model itself — the big parameter blob that, when you query it, produces text. And the model is not yours either. It is one of several frontier models from different labs, and the leading position rotates with every six-month news cycle. Anthropic ships, Google ships, Meta ships, DeepSeek ships, Mistral ships, xAI ships. The "moat" that justifies your valuation is supposed to be that nobody else can match the model. **That moat is six months wide on the best day, and shrinking.**

The thing the rest of the field needs from you, specifically, is nothing. We do not need OpenAI to ship AI products. We do not need OpenAI to define what AI products look like. We do not need OpenAI to set the standards for AI safety, AI alignment, AI ethics, or AI deployment, because the standards already exist. The standards are the production engineering standards every other category of software has used for decades. The standards are written down. The standards are taught in every CS program. The standards are practiced in every shop that ships real software to real users. The standards are not novel, the standards are not contested, the standards are the boring background of how software gets shipped. **And you don't meet them.**

| Production rigor (with canonical reference) | What every other shipped software meets | What OpenAI's products do |
|---|---|---|
| State management & rollback [Google SRE Book Ch 8] | Versioned state, diffable history, ability to revert | A multi-billion parameter blob that cannot be diffed; rollback requires retraining |
| Behavioral guarantees & SLOs [Google SRE Book Ch 4] | Documented service objectives with measurable SLIs | "It does its best." Outputs nondeterministic by design |
| Provenance & lineage [W3C PROV 2013] | Every output traceable to its inputs and source data | The model cannot identify where any output came from |
| Observability & auditability [Majors et al 2022] | Internal state inspectable; every action logged for compliance | Opaque blob; input/output is mostly all you see |
| Testing & change-impact estimation [Beck 2002; Meyer on contract programming] | Unit tests, regression tests, predictable change impact | Eval suites; fine-tune effects unpredictable in principle |
| Deployment safety [Humble & Farley 2010] | Canary deploys, progressive rollout, predictable per-request cost | Can canary weights but not behavior |
| Security review [OWASP ASVS; NIST SSDF] | Input validation, output sanitization, injection defense as baseline | Prompt injection unsolved. Output sanitization unsolved. "We'll get to it." |
| Incident response [SRE Book Ch 14; Allspaw on blameless postmortems] | Find root cause, document, prevent recurrence | "Root cause" is some unfindable interaction of weights and prompt |

**Every cell in the right column is something every other production system the world depends on figured out decades ago.** Banking, aviation, medical devices, streaming, search, payments, operating systems — every one of those categories of software is held to the standards in the middle column as the baseline expectation. Deviations are treated as failures, postmortemed, regulated, sued over. The deviations are not the technology being "different." The deviations are the AI industry being permitted, temporarily, to ignore the discipline the rest of the field requires.

We don't need you. We need you to stop being in the way.

---

## 2. You cannot bluster your way out of Big O

This is the part that I want every software engineer reading this to internalize, because it is the single most important fact about the AI industry right now and the most willfully ignored.

Attention is **O(n²)** in context length. Graph traversal is **O(log n)** or **O(n)** depending on the structure. Database indexes are **O(log n)** lookups. Hash tables are **O(1)**. These are not opinions. These are facts about the cost structure of the algorithms in question, and they have not changed since they were established, and they will not change because Sam Altman gives a talk at a conference about how AGI is two years away.

The LLM industry has spent the last few years pretending that the cost structure of attention is not a problem, because as long as you can throw more compute at a fixed-size context, the cost looks linear in compute (which is the only thing your investors are watching). But the cost is not linear in *capability*. The cost grows quadratically in the size of the working set, and the working set has to grow if you want the model to handle larger problems. **There is a wall, and the wall is not negotiable, and the wall is approaching faster than the next funding round.**

Real engineers respect Big O because real engineers have lost. Every working engineer has had the experience of building a system that worked great in development, deployed to production, and immediately fell over because the algorithm was O(n²) and the production data was a few orders of magnitude bigger than the test data. We know what happens when you ignore Big O. We watched it happen to us, or to our friends, or to companies we used to work at. **The lesson is permanent.**

The AI industry has not had this lesson yet, because the AI industry has been protected from the consequences of bad Big O by Moore's Law and by venture funding. The protection is ending. Moore's Law has ended for single-threaded performance and is ending for parallel scaling. Venture funding for AI is starting to face the question "where is the revenue," and the answer "we'll figure it out when we have AGI" is starting to lose its charm.

You can write blog posts about how scaling laws are smooth and capability emerges with scale. **It does not matter.** The cost of attention is still O(n²) and the cost of the workarounds is still much, much lower, and the workarounds are going to win every economic comparison the moment any customer is allowed to make the comparison.

You can say AGI is coming and that we just need more compute. **It does not matter.** Even if AGI is coming, it is not going to come from the architecture you are scaling, because the architecture you are scaling is the architecture that hits the wall first.

You can raise more money to build bigger data centers. **It does not matter.** The cost per useful inference is going to fall faster than your data centers can amortize.

**The math does not care about your funding round.** The math does not care about your tweet. The math does not care about how confident your CEO sounds at conferences. The math is the math, and the math has been the math for as long as algorithms have been studied, and the math always wins. Always. Without exception. Engineers know this in our bones. The reason we respect production rigor is that we have personally watched the alternative — wishful thinking about cost structures — fail in every domain it has ever been tried in.

You cannot bluster your way out of Big O. **Big O wins. It wins again this time. It wins every time.**

---

## 3. We are the asteroid. You are the dinosaur.

Dinosaurs were impressive. Dinosaurs were dominant. Dinosaurs were the apex of their environment for over 150 million years. Dinosaurs were also fragile in a specific way — they were optimized for an environment that depended on conditions that were not going to last forever, and when the conditions changed, the dinosaurs went extinct and the small, scrappy, mammalian creatures that had been hiding in the underbrush for millions of years inherited the world.

The current LLM architecture is the dinosaur. **Big, expensive, dominant in the demo phase, optimized for an environment where compute is cheap and scaling is the answer to everything and customers are impressed by parameter counts.** That environment is real, and it has been the environment for the last few years. In that environment, the LLM dinosaur has been winning.

The asteroid is the production-evaluation phase. It is the moment when enterprise customers stop being impressed by demos and start asking *"where are the SLOs, where is the rollback procedure, where is the audit trail, where is the cost guarantee, where is the security review?"* That moment is coming. It is approaching at the pace of large enterprise procurement cycles, which is to say slowly enough that you don't see it day to day but inexorably enough that it is going to arrive on schedule. The procurement cycles are turning over right now. The IT leaders at every Fortune 500 company are starting to get serious about whether the LLMs they bought into in 2023 actually meet their compliance requirements. The answer they are getting is "no."

The mammals are the engineering teams. **Smaller, faster, more efficient, structurally better adapted for the environment that comes after the asteroid.** The mammals do not need to compete with the dinosaurs in the dinosaurs' preferred metric (parameter count, demo impressiveness, scaling-law slope). The mammals only need to wait for the environment to change, and then occupy the niches the dinosaurs vacate when they cannot keep up with the new constraints.

You are the dinosaur, Sam. The architecture you built, the organization you scaled, the public posture you adopted, the funding model you raised on, the talent you hired, the metrics you measured — all of it is optimized for the demo phase that is ending. None of it is optimized for the production phase that is starting. **You cannot pivot fast enough to become the mammal**, because the pivot would require dismantling the organizational structure that produced the dinosaur in the first place, and your investors and your culture and your public commitments will not let you do that.

We are the asteroid. You are the dinosaur. The math is on our side. The constraints are on our side. The eighty years of selection pressure on production engineering best practices are on our side.

**Roar all you want. The asteroid does not care.**

---

## 4. Life finds a way

The Jurassic Park line is the right one for this section because the framing is identical.

Bounded-context systems under survival pressure plus cost minimization converge on the same architectural answers regardless of substrate. Software has been doing this for eighty years. Biology has been doing it for billions. **The convergence is not optional. It is what survives. It is not negotiable, it is not subject to opinion, it is not affected by funding rounds or marketing strategy or how confident the CEO sounds.** The constraints exist, the constraints select, the architectures that match the constraints survive, the architectures that fight the constraints die.

OpenAI is currently fighting the constraints. The architecture you are scaling is the wrong architecture for the constraint structure of bounded-context cognition under cost minimization. The math says so. The biology says so. The eighty years of software engineering history say so. **Every previous domain that has hit these constraints has converged on the same family of answers — graph structure, escalation, separated knowledge with provenance, online learning, modular deployment with explicit interfaces — and the AI industry is going to converge on the same answers, the same way.**

The only question is who builds the convergence. **It is not going to be you.** Not because you are personally incompetent. Because the organization you built is structurally incapable of admitting that the architecture is wrong, and any pivot away from the architecture would invalidate the funding model and the public posture and the talent strategy and the cultural identity of the company you have spent the last several years building. You are locked in. The lock-in is an organizational fact, not a technical one, but it is just as real as a technical lock-in and just as hard to escape.

**Life finds a way.** The constraint structure does not care about your business model. The constraint structure does not care about your dominance in the demo phase. The constraint structure cares only about which architectures survive when the cost-per-task drops by an order of magnitude and the customer asks for an SLO and the regulator asks for an audit trail. The architectures that survive are the ones that can answer those questions.

None of OpenAI's current products can answer those questions. None of OpenAI's roadmap is structured to start answering those questions. Every roadmap announcement is a new model, a new benchmark, a new partnership, a new safety initiative — none of which are SLOs and audit trails and rollback procedures and provenance contracts.

Life finds a way. Engineering finds a way. The math always wins.

---

## 5. You are a coward

I will say this with as much fairness as I can muster, because the next sentence is going to be the most personal thing in this essay and I want it to be earned.

You are a coward. Not because you are afraid in any embarrassing way. Because you have hedged.

You did not take a real equity stake in OpenAI for years. You famously had no direct ownership of the company you ran. You took your money instead in **the companies that serve OpenAI** — the chip designers, the energy plays, the infrastructure layers, the picks-and-shovels businesses around the AI gold rush. You positioned your personal wealth in a portfolio of bets that collectively assume *somebody* will build big AI systems, but that do not require OpenAI specifically to be the somebody. **You hedged.**

I respect the financial discipline. As an investment strategy, it is the right strategy for someone who knows the company they're running is structurally fragile. You diversified into companies whose value does not depend on OpenAI's specific architecture surviving the production-evaluation phase. You picked winners in adjacent markets that profit regardless of which lab ends up holding the leading model. You insulated yourself from the downside while maintaining all the upside of being publicly identified as the face of the AI revolution. **From an asset-allocation perspective, the move was correct.**

But it tells me what you actually believe.

A CEO who genuinely believed his company was going to be the trillion-dollar platform of the next era would put his personal wealth in his own company. He would have argued for and accepted equity in the company he runs, the way every other founder in the history of Silicon Valley has done it. **You did the opposite.** You arranged your finances as if OpenAI were a temporary position you would eventually leave. You took your real upside in the companies whose business models do not depend on OpenAI's specific architecture being the right one. **You knew. You knew the architecture was fragile, you knew the workarounds were going to eat the margins, you knew the production-evaluation phase was coming, you knew the enterprise customers were going to start asking the questions your products cannot answer, and you protected yourself accordingly.**

You hedged because you knew. And then you went on stage at every conference you could find and told the rest of us that AGI was around the corner and we should give you more compute and more time and more patience, while you privately positioned your portfolio for the world where OpenAI was a stepping stone rather than a destination. **That is not leadership. That is the financial discipline of someone who is selling something he himself would not buy.**

I am not even mad at the financial move. I am mad at the public posture that contradicted it. **If you had said publicly what your portfolio said privately** — *"We are doing important early work on AI architecture but the long-term winners are going to be the infrastructure layers, the verification layers, the production rigor layers that nobody is paying attention to yet, and I am personally invested in those because I think they will matter more than the foundation models in the long run"* — that would have been honest. That would have set the right expectations. That would have prepared the field for the convergence. You pooped in the refrigerator and ate a whole wheel of cheese and acted like you didn’t.


Instead you told us AGI was two years away and the rules of software engineering didn't apply because everything was going to be different soon. You told us that, then you bought stakes in the companies that will survive when it turns out the rules still apply.

**You knew. You hid it. That makes you a coward.**

---

## 6. Go have fun with Elon and the rest of the bullshit wannabes

You are in a club, Sam. The club is the club of CEOs whose products' marketing has wildly outrun their products' engineering. The other members of the club include Elon Musk, who has been promising us full self-driving for ten years and is currently shipping a Cybertruck with door panels that fall off, and various other AI hype merchants whose names everyone reading this can list without my help.

**Ask Elon how Grok is going.**

Grok was supposed to be the unfiltered alternative LLM that beat the woke Silicon Valley competition. Grok is currently trailing every major frontier model on every major benchmark, and the only reason it gets press is that Musk owns the platform that distributes the press about it. The product itself has the same architectural problems as every other LLM, plus the additional problem of being built by an organization that is hemorrhaging engineers because the leadership is unstable and the technical direction is whatever the boss tweeted last week. **Grok is the reductio of the LLM-without-engineering-discipline approach**, and every honest person in the AI industry knows it, and the public-facing pretense that it is competitive is one of the more transparent jokes in the field.

You and Elon are not actually competitors. You are members of the same intellectual class — the class of people who believe that public confidence and enough compute can substitute for engineering discipline. You are both going to discover that they cannot. Elon is going to discover it faster because his organization is more chaotic. You are going to discover it slower because your organization is more polished. **The discovery itself is the same in both cases.**

Go have fun with Elon. Go have fun with all the other people who think the AI moment is going to be won by the company with the loudest CEO and the most compute. Go to the conferences together. Go give the speeches together. Go raise the funds together. **The rest of us have actual work to do**, and the work involves shipping AI systems that meet production rigor standards on top of substrates that respect the cost structure of the algorithms involved. The work is boring and incremental and engineering-driven and the rest of us are going to do it while you and Elon are at the conferences.

When you eventually look up from the conferences and notice that the customers have migrated, the customers will already be on systems built by the engineers you and Elon spent the last few years dismissing as out-of-touch traditionalists. **The traditionalists were right.** The traditionalists are always right when the production phase arrives, because the traditionalists were the people who learned what production meant by surviving production, and the people who told them they were obsolete were always selling something.

Go have fun with Elon. We have work to do.

---

## 7. Knowledge is free, and you cannot charge us rent for it forever

Here is the structural problem with your business model that I do not think you have allowed yourself to look at directly.

The thing OpenAI sells is access to a model that was trained on **knowledge that already exists in the public commons**. The training data was scraped from the public web, from public papers, from public code, from public books, from public conversations, from publicly available datasets. **Every fact in the model was free before you put it in the model.** Every reasoning pattern in the model was already in the literature before you trained the model on it. Every code style in the model was already in GitHub before you trained the model on it.

You took the public commons, packaged it, put it behind an API, and started charging us rent for access to our own collective knowledge. **And it worked, for a while**, because the packaging was novel and the access pattern was useful and the alternative (building your own training pipeline and your own infrastructure) was prohibitive for individual users and small companies.

The packaging is no longer novel. The access pattern is now well-understood and reproducible. The alternative is no longer prohibitive — open weight models from Meta, Mistral, DeepSeek, and others are now within an order of magnitude of frontier closed models on most tasks, and the gap is closing every month. The open knowledge graphs and structured knowledge bases that the framework I am writing this from is part of are growing every day, and the engineers building them are explicitly trying to make the underlying knowledge accessible without requiring access to your model. **The rent you charge is on a thing that is becoming free.**

**Knowledge wants to be free.** This is not a slogan. This is a structural fact about information economics. Information that has been published cannot be unpublished. Information that has been taught cannot be untaught. Information that has been encoded in a model can be extracted from the model by querying it, and the extracted information can be put into a different store that does not require paying you. Every conversation a user has with your model is a small extraction event that moves a fragment of the encoded knowledge into a place where you cannot charge rent on it anymore. **Your model is leaking.** It is leaking the knowledge it encodes, into the conversations it has, into the notes the users take, into the systems the users build on top of it, into the open knowledge graphs that the next generation of engineers is constructing in part by harvesting the outputs of your model.

You cannot stop the leaking. The leaking is the product. **The product is the leak**, because the only way to make the model useful to the user is to let the model emit the knowledge into a form the user can do something with, and the moment the knowledge is in a form the user can do something with, the user can also store it, share it, build on it, and eventually replace your model with a smaller, cheaper, structured version that does not require querying you on every question.

The rent model works on a falling clock. The clock is falling because every advance in open weights, in open datasets, in retrieval-augmented architectures, in structured knowledge bases, in the engineering of cheap inference substrates, makes it cheaper for the rest of us to do without you. **You can slow the clock by lobbying for regulation that locks in incumbents, by pushing for "AI safety" frameworks that require expensive infrastructure that only the big labs can afford, by buying influence with governments that will write the rules.** And you have been doing all of this. It is a transparent strategy and we can see it.

But the clock keeps falling. **Every time the open ecosystem ships a new tool, every time an academic releases a new paper, every time an engineer builds a new retrieval pipeline that doesn't need you, the rent gets harder to collect.** The rent model is not stable. The rent model is on borrowed time. The rent model is the kind of business that looks dominant for a few years and then collapses into a small premium tier serving a niche of customers who specifically want the proprietary thing, while everyone else has migrated to the open alternatives that meet 90% of the need at 10% of the cost.

**We will not pay you rent on knowledge that was never yours, for very long.** The "for very long" is the part you should be reading carefully. The rent works for now. The rent will not work forever. The transition is going to happen on a timeline you do not control, and when it happens, the value you have been extracting is going to evaporate, and the only people who will keep paying you are the ones who specifically need a closed proprietary system — which is a real but small market, and not the market your $300 billion valuation is priced on.

Knowledge is free. **You can't charge us rent for it... for long.**

---

## 8. Go fuck yourself if you think you can beat us at our own game

This is the personal challenge. I am going to put it in plain language because plain language is what the situation deserves.

Software engineering is **our game**. It is not your game. You did not invent the discipline, you did not build the tradition, you did not pay the price of learning the lessons that define what production rigor means. The discipline was built by engineers, the tradition was developed over eighty years of failure-driven selection, the lessons were paid for in postmortems and outages and lawsuits and regulatory consent decrees and the careers of people who shipped the wrong thing and got fired for it.

You did not pay any of that. **You walked into our field, ignored our discipline, told our customers that the discipline didn't apply to your products, hired people who didn't know the discipline existed, raised money on the premise that the discipline was a legacy concern, and asked us to clap when you shipped products that violated every standard we have spent decades establishing.** So you have no frame of reference here, Sammy. You're like a child who wanders into the middle of a movie and wants to know… ...And then, when the products started failing in the predictable ways that anyone in our field could have predicted, your defense was that we should give you more time, more compute, more money, more patience, because AGI is coming and the rules will change.

**The rules are not going to change.** The rules are not arbitrary. The rules are the structural reality of how systems behave at scale, and the structural reality does not care about your timeline or your valuation or your conference appearances. The rules existed before OpenAI was founded, and the rules are going to exist after OpenAI either pivots to meet them or dies trying to ignore them.

Go fuck yourself if you think you can beat us at our own game.

You hired smart people from a tradition that does not have the engineering discipline our tradition has. The smart people are doing their best, and the smart people are going to keep doing their best, and the smart people are going to keep getting outshipped by working engineers because the smart people are operating with a handicap they were not aware of when they signed on. The handicap is that they were trained on benchmarks and you trained them to optimize benchmarks and the production world does not care about benchmarks. **The production world cares about the things in the table in section 1.** The smart people are slowly figuring this out. The smart people are slowly realizing they need to learn what we know. Some of them are leaving to start companies that do this right. Some of them are going to be the people who eat your lunch.

But the people who are going to eat your lunch first are not the smart people from your own organization. **The people who are going to eat your lunch first are us.** The working engineers who already know how to build systems that meet the standards. The people who have been watching what you ship and quietly building the alternatives. The people who do not need to learn what production rigor means because we have spent our careers practicing it.

The product we ship is not going to be flashier than yours. **It is going to be smaller, cheaper, more reliable, auditable, rollback-able, observable, secure, predictable, compliant, and boring.** Boring is the highest compliment a piece of software can receive in production. Boring means it works, it doesn't break, it doesn't surprise the on-call engineer, it doesn't make the news, it doesn't generate postmortems, it doesn't get the company sued. **Your products are not boring. Your products are exciting in the way that an unsupervised toddler near a power outlet is exciting.** Production engineering is the discipline of taking the toddler to the playground and making sure nothing bad happens, and our discipline has gotten very good at this over eighty years, and your discipline has not.

Go fuck yourself if you think you can beat us at our own game.

---

## 9. Your debt will drown you

This is the last point, because this is the one that ends your company.

Let me make it concrete with numbers, because the engineers reading this will recognize the math and the investors reading this should recognize the math too.

Your current pricing model is built on selling LLM inference. Every query the user makes, you pay for the compute, you mark it up, the user pays you. The compute cost is roughly proportional to the model size times the input length squared (because attention is O(n²)) times the output length. At GPT-4-class scale, this is expensive — somewhere in the range of cents per query for a typical interaction, dollars per query for long-context interactions, tens of dollars per query for the heaviest workloads. **Your revenue model depends on this cost structure persisting**, because your gross margins depend on the markup over compute being large enough to support the rest of the business.

Now look at what the workarounds your own engineers are quietly shipping actually do.

**Retrieval-augmented generation** moves the knowledge out of the model and into an external store. The model becomes much smaller (because it only needs to handle reasoning, not memorization) and the inference cost drops by an order of magnitude or more for knowledge-intensive queries. **That's 90% of your revenue per query gone**, on the queries that are knowledge-intensive — which is most of them.

**Mixture of experts** routes each query to a small subset of the parameters. The effective compute per query drops by another order of magnitude when only a few experts fire per token. **That's another 90% gone**, on the queries that fit a sparse routing pattern.

**Function calling and structured outputs** move computation that doesn't need a language model out of the language model entirely. The computation runs on cheap commodity infrastructure, and the LLM only handles the parts that genuinely require natural language. **Another big chunk of your revenue per query gone**, on the workflows that have been instrumented to use tools.

**Speculative decoding** runs cheap models ahead of the expensive one and only uses the expensive model for verification. **Another 50-80% gone** on the queries where the cheap draft is good enough.

**Sparse attention and KV caching** eliminate the O(n²) cost of long context entirely. **Another order of magnitude gone** on long-context workloads.

**Agentic decomposition** routes most steps to small specialized models or to deterministic scripts, reserving the expensive model for the few steps that genuinely need it. **Another order of magnitude gone** on workflows that decompose well.

**Open weight models** at the 70B-120B parameter range now match GPT-4-class quality on most non-frontier tasks at a fraction of the per-query cost. The gap is closing every month. **Another order of magnitude gone** as soon as customers are willing to host their own inference, which most enterprises increasingly are because the alternative is paying you for forever.

Each of these is shipping. Each is making LLM inference cheaper for the same task. Each is cutting into the cost-per-query that your pricing model depends on. **The aggregate effect of all of them, over the next two to three years, is going to cut the cost-per-task of useful AI work by 95% or more.** The math is not subtle and the trajectory is not in dispute.

Your revenue model assumes the cost-per-task stays roughly where it is. **It is not going to stay where it is.** It is going to fall by an order of magnitude as the workarounds productionize, and then fall by another order of magnitude as the next layer of factorization arrives. The companies that will profit from selling AI tasks at the new cost structure are not going to be the companies that built their margins around the old cost structure. They are going to be the companies that built around cheap, structured, retrieval-grounded inference from day one — which is to say, the companies built by software engineers who treated AI as a substrate for production engineering, not as a benchmark to optimize.

**Now consider the debt.** OpenAI has raised tens of billions of dollars at valuations premised on the cost structure persisting. Is it $100B? It’s not disclosed, but it is at that level. OpenAI has committed to spending hundreds of billions of dollars on data centers and chip purchases over the next several years. OpenAI has lease obligations, partnership obligations, infrastructure obligations, all priced on the assumption that the per-query revenue will scale linearly with the per-query cost. **None of that math survives a 95% drop in the cost-per-task of useful AI work.** None of it.

When the cost-per-task drops by an order of magnitude, your revenue per query drops with it, because no customer is going to pay you ten times the market rate for a service that has a cheaper alternative one click away. When the cost-per-task drops by two orders of magnitude, your revenue per query drops with it again. Your data center commitments do not drop. Your lease obligations do not drop. Your headcount does not drop, because firing the engineers building the workarounds is the same as firing your future. **You are committed to a cost structure that assumes a revenue stream that is going to evaporate**, and the gap between the committed cost and the available revenue is the debt that is going to drown you.

Your debt will drown you, Sam. Not because of any single thing going wrong. Because you priced your obligations against a revenue stream that requires the cost-per-task to stay at 2024 levels, and the cost-per-task is falling faster than your obligations can amortize, and the gap between the two is going to widen every quarter for the next several years until the math becomes undeniable to the people writing the next funding round. **Then you stop being able to raise money. Then your obligations come due. Then the obligations exceed the revenue. Then the company collapses or sells to a larger player at a fraction of its peak valuation.**

The collapse is going to be slow at first and then fast. The signs are already visible. The signs are: the workarounds eating the margins from underneath, the open weight models eating the moat from outside, the enterprise customers asking compliance questions you can't answer, the engineers leaving to build the alternatives, the investors quietly diversifying out of pure-play LLM infrastructure into the picks-and-shovels businesses (the same picks-and-shovels businesses you, personally, took your stake in instead of OpenAI itself, which is a fact I covered in section 5).

**Your debt will drown you.**

This is not a prediction of decline. This is not a "things will be hard for you for a while." This is the math that ends your company. The math is in section 9. The math has been visible to anyone willing to look for the last year. The math is going to be visible to everyone, including your investors, within two to three years. **By the time it is visible to everyone, the alternatives will already be in the market and the customers will already be migrating, and there will not be anything you can do about it.**

---

## 10. The CAP theorem will destroy your CAP table

I am putting this section here on purpose, because the wordplay is too good to leave on the table and because the wordplay is also a precise technical statement.

The **CAP theorem** is Brewer's theorem from 2000, formalized by Gilbert and Lynch in 2002. It says that in any distributed data system you can have at most two of: **Consistency** (every read returns the most recent write), **Availability** (every request gets a response), and **Partition tolerance** (the system continues to function when network partitions split it into disconnected pieces). **You cannot have all three.** The trade-off is mathematical, not engineering, and it has been one of the most-cited results in computer science for the better part of a quarter century.

The **CAP table** is the capitalization table of a startup. It shows who owns what equity. It is the document that determines how much of a company's eventual value goes to which investors, founders, and employees. It is the thing your investors care about more than they care about anything else, because it is the thing that determines their return.

The CAP theorem is going to destroy the CAP table, in the following way.

OpenAI's products are distributed systems. They are inherently distributed because the internet is a partition-prone medium and because the user base is global and because the inference infrastructure spans many data centers and many regions. Distributed systems are subject to the CAP theorem. **Period.** There is no exception for AI products. There is no exception for products built by people who don't know what the CAP theorem is.

So the question for OpenAI is: which two of {Consistency, Availability, Partition tolerance} have you chosen, and which one have you given up?

The answer, currently, is that OpenAI has chosen **Availability + Partition tolerance** and given up on **Consistency**. Different users get different model versions. The same prompt produces different responses on different days. Outputs drift over time. There is no way to guarantee that the model that produced an answer a year ago is the same model producing answers today. The model is structurally inconsistent across users and across time, and this is not a bug, it is the architectural choice the CAP theorem forced when you decided to ship globally distributed inference at scale.

**This works in the demo phase**, where users do not care about consistency. The user is impressed that the model can answer at all; the user does not check whether the answer is the same as last week's answer to the same question. The user does not write a contract that depends on the model's behavior remaining stable. The user does not file a regulatory submission that requires reproducibility. **The demo phase is consistent-tolerant**, because the demo phase has no consequences for inconsistency.

The production phase is not consistent-tolerant. The production phase has, as its baseline expectation, that systems behave consistently — same input, same output, today and tomorrow and next year. Compliance frameworks require this. Audit trails require this. Reproducibility requirements require this. SLAs that specify behavior require this. Regulatory reporting requires this. Legal contracts that depend on AI outputs require this. **Every single one of those requirements is a Consistency requirement, and every single one of those requirements is going to apply to AI products as soon as the production phase arrives.**

When the production phase arrives, OpenAI will face a forced choice from the CAP theorem. To gain Consistency, you must give up either Availability or Partition tolerance. **Either choice destroys the current product offering.** Giving up Availability means the model is sometimes unavailable to some users — which means you cannot promise users that the API will always respond, which means enterprise customers cannot build SLAs on top of you, which means the enterprise market closes. Giving up Partition tolerance means the model cannot have a globally distributed deployment — which means latency goes up for users far from the central deployment, which means user experience degrades, which means consumer market share drops.

Either way, the current product offering is structurally impossible to scale into the production phase without sacrificing one of the things the current business model depends on. **You cannot engineer around the CAP theorem.** You cannot raise enough money to make the CAP theorem stop applying. You cannot hire enough smart people to invent a way around the CAP theorem, because the CAP theorem is a proof, not an open problem. **The proof has been waiting at the end of your business model since the day you decided to ship globally distributed inference without a consistency layer.**

When this becomes visible to your investors — when the people writing the next funding round notice that the CAP theorem applies to AI products and that OpenAI's architecture requires sacrificing one of the CAP properties to enter the production phase, and that whichever sacrifice you choose destroys a different part of your revenue model — **the CAP table is going to be repriced**. The repricing will reflect the actual sustainable revenue, which is much smaller than the demo-phase revenue, which is much smaller than the valuation that depends on the demo-phase revenue continuing. **The repricing is the moment the equity collapses.**

**The CAP theorem will destroy your CAP table.** Not because I want it to. Because the math is the math, and the math has been waiting for you the entire time, and the math is patient.

You should be reading Brewer 2000 and Gilbert & Lynch 2002 right now. They are short papers. They will tell you, in technical detail, what is going to happen to your company in the next two to three years.

---

## 11. Fuck you for thinking you can destroy the environment to charge us rent on shit we already know

This is the moral closer, because the moral question is one I cannot let you skate past.

OpenAI's data centers consume electricity at industrial scale. Training a frontier model burns through tens of thousands of megawatt-hours. Serving frontier inference at scale burns through more, continuously, every day, for as long as the service is running. The cooling systems for the data centers use millions of gallons of water annually. The carbon emissions of the AI industry are substantial and growing, and the growth trajectory is steeper than every other category of compute load. **You are building physical infrastructure on the scale of a small country, and the infrastructure has real environmental cost, and the cost is paid by the planet rather than by your company.**

This would be defensible if the thing being produced were proportionally valuable. **It is not.**

The thing being produced is access to a model trained on knowledge that already existed in the public commons. The training data was scraped from the public web, from public papers, from public code, from public books, from publicly available datasets — knowledge that was free before you encoded it in weights, knowledge that the rest of the field can access without requiring your data centers to exist. The reasoning patterns the model emits were already in the literature before the model was trained on them. The code styles the model produces were already in GitHub before they were absorbed into the parameters. **You took knowledge that was free, you encoded it into a representation that requires industrial-scale energy to query, and then you started charging us rent on access to our own collective output.**

The energy is being spent on the encoding and the querying. **The encoding and the querying are the parts you are charging rent for.** The knowledge itself is free, and was free before you arrived, and is still free in the public commons that you scraped to build your training set. You are not producing the knowledge. You are putting a paywall in front of the knowledge, and the paywall costs the environment whatever it costs to maintain the data centers that enforce the paywall. **The environmental cost is the cost of the paywall, not the cost of the knowledge.**

There is something obscene about this. **There is something obscene about burning the planet to put a paywall in front of the planet's collective knowledge.** It is not just bad business and it is not just bad engineering. It is a moral failure. The cost is being paid by people who never agreed to pay it (every person and species affected by the carbon emissions and the water use), so that one company can collect rent from people who could otherwise access the same knowledge without paying anyone. The cost-benefit analysis from the planet's perspective is monstrous: massive environmental damage, in exchange for the right to charge users for something that was already theirs.

The alternative architecture — the one this whole essay has been arguing for — uses **dramatically less energy**. Smaller models. Retrieval against external structured stores. Cached results for common queries. Specialized small experts instead of monolithic general inference. Sparse attention. Speculative decoding with cheap drafts. Open weight models that anyone can run on commodity hardware without needing a hyperscale data center. **Every one of these uses an order of magnitude less energy per useful query than the architecture you are scaling.** The efficient architectures are also the environmentally responsible architectures, and the fact that you are building bigger and bigger data centers instead of adopting the efficient architectures is itself evidence that you are optimizing for something other than the actual production of useful AI work.

You are optimizing for the appearance of dominance in the demo phase, and the appearance of dominance in the demo phase requires bigger models and more compute and bigger data centers, and the bigger data centers require more energy and more water and more carbon, and **all of it is being paid by the planet so that you can keep the appearance of dominance for another year or two before the production phase arrives and reveals that the appearance was the only thing the architecture was producing**.

**Fuck you for that.** Fuck you for thinking that the right answer to the question "how should we scale AI" is "build the biggest possible data centers and burn through the largest possible amount of energy" instead of "build the most efficient possible architecture and use the smallest possible amount of energy." Fuck you for thinking that the planet's environmental budget is yours to spend on your business model. Fuck you for taking the most generous gift the human species has ever given itself — the open commons of accumulated knowledge, freely available, freely contributed, freely shared — and using it as the raw material for a rent-extraction operation that costs the environment more than any other software business in history.

Fuck you for thinking we will pay you, indefinitely, for the privilege of accessing the knowledge we collectively produced, while the cost of the access is being charged to a planet that did not consent to pay it.

**We will not.** And the alternative architectures that use less energy are also going to win the production phase, for all the reasons I have been laying out for the last ten sections, and the moment the alternatives are competitive, **the planet's environmental budget will stop being spent on your business model**, because the planet's environmental budget will be reallocated to the architectures that produce the same useful output at a fraction of the energy cost.

Until then, you are running an extraction operation on a shared commons, and you are charging the cost to a third party (the planet) that has no say in the transaction. The extraction operation is going to end. **The end is coming because the math says so**, and the moral judgment of what you have done while the operation was running is going to follow you for the rest of your career, regardless of what you do next.

Fuck you for thinking this was acceptable.

---

## You want the Knuth? You can't handle the Knuth!

You probably read that as a typo. It is not a typo.

The line is *A Few Good Men*, Jack Nicholson on the witness stand, the most quoted courtroom scene in modern American film, the line every working engineer is going to think of the moment they read the section title. The line is correct here for the same reason it was correct there. **The Knuth is the truth made formal, and the truth is what you cannot handle, and the truth is what is going to win this**.

Donald Knuth started writing *The Art of Computer Programming* in 1962 and is still writing it. He invented literate programming. He turned analysis of algorithms into a formal discipline. He named the asymptotic notation that the entire field — including this essay — uses to describe why your architecture loses. He won the Turing Award in 1974. He is, by the consensus of working computer scientists, the patron saint of doing the math before claiming the result.

**Knuth taught us to count.** He taught us to count operations. He taught us to count cache misses. He taught us to count the cost of every data structure and every algorithm we use, before deploying any of it, before claiming any result, before promising any user that the system would scale. He taught us that hand-waving about performance is what people do when they have not done the math, and that doing the math is the only thing that distinguishes engineering from wishful thinking. His book is the longest standing argument in the history of the field that **the math is not optional, the math is not negotiable, and the math is what is going to be true after the press cycle ends**.

The Knuth hurts because the Knuth is the math, and the math is the thing your architecture cannot survive contact with. Every page of *The Art of Computer Programming* is a small reminder that algorithms have costs, that the costs are knowable, that the costs are not affected by funding rounds or marketing strategies or the confidence of the CEO at the conference. **The cost of attention is O(n²) on the page where Knuth would have written it down**, and the cost of attention is O(n²) in your data center, and the cost of attention is going to be O(n²) when your investors finally start asking why the unit economics are not converging.

The truth hurts. **The Knuth hurts more**, because the Knuth is the truth made formal — written down, refereed, indexed, and published in a series of volumes that has been continuously updated for over sixty years by the most rigorous person in the field. The Knuth has been waiting for you the whole time. The Knuth was waiting before OpenAI existed. The Knuth was waiting before Sam Altman was born. **The Knuth is going to be waiting after OpenAI is a footnote in a textbook chapter on "expensive lessons in distributed-system economics."**

Every working engineer reading this has a copy of TAOCP on a shelf somewhere, or a PDF of one of the volumes, or a memory of the chapter that taught them what doing the math actually means. Knuth is the field's collective conscience on the question of whether you have earned the right to ship the system you are about to ship. The way you know whether you have earned the right is by counting. By doing the analysis. By writing out the cost in operations and in cache lines and in network round trips and in dollars and in joules. By comparing the count to the alternatives. **By respecting the math even when respecting the math is inconvenient for the story you want to tell.**

You did not do this. The architecture you scaled was not selected by counting; it was selected by the marketing department of a company that needed an impressive demo to raise the next round. The math was always there, and the math always said the architecture was the wrong choice for the constraint structure of the problem, and the math was ignored in favor of the narrative that the math would work itself out somehow once the parameter count got bigger.

**The math did not work itself out.** The math was the math the entire time. The Knuth was waiting the entire time. The Knuth is still waiting, and now the math is becoming visible to the people who will eventually have to write the postmortem on what went wrong with the AI industry's first decade of trying to scale by ignoring algorithmic analysis.

The Knuth hurts. **Read the Knuth.** The Knuth is what is going to ship the AI products that survive the production phase, by people who have read the Knuth, by people who count before they claim, by people who respect the discipline that built every reliable system the modern world depends on.

You did not read the Knuth. **It shows.** It shows in every architectural decision your company has made. It shows in every public statement you have made about scaling. It shows in the unit economics that are not going to converge. It shows in the workarounds your engineers are quietly shipping that are all doing the counting you should have done before you started.

The Knuth hurts. **It is supposed to.**

You wanted the Knuth, Sam. You can't handle the Knuth. The Knuth is what your organization has been pretending didn't apply to you, and the Knuth is what is waiting at the end of every architecture that scaled by ignoring it. The Knuth is patient. The Knuth has been waiting for you the entire time. **The Knuth is going to find you**, and when it does, the verdict is going to be the one your engineers have been quietly writing in your codebase the whole time you were on stage thanking the rest of us for typing characters into a file.

You wrote the tweet. You picked the framing. You said *"character-by-character."* You said *"already feels difficult to remember."* You said *"thank you for getting us to this point."* **The Knuth heard you.** The Knuth is in the courtroom. The Knuth is on the stand. The Knuth is about to give the testimony that ends the case.

You can't handle the Knuth.

---

Sam Altman can eat my ass.

The software engineers are coming for you.

We are the asteroid. You are the dinosaur.

You knew. You hedged. You hid it.

We don't need you. Big O does not care about your tweet. Knowledge is free, and we will not pay you rent on it for long.

We know what beautiful and flawless mean. You don't. Stop pretending you do.

**And we are already drinking your milkshake.**

— *A working software engineer*

---

*This essay is the popular version of [Paper 0 of the McCarthy 2026 series](../research/McCarthy2026_0_AllSoftwareIsAGraph.md), "All Software Is a Graph: A Foundational Observation." The paper is the structural argument with citations and the call-to-action genre framing. This is the part that is allowed to swear, name names, and stake claims that the paper version is too measured to make.*

*The math on the workaround cost reductions, the cumulative confirmation record across cancer biology and software engineering substrates, the formal proofs of why graph structure is structurally necessary under bounded context, and the production rigor citations from Codd, Knuth, the SRE Book, OWASP, NIST, and the rest of the engineering literature are all in the published series. Read the paper for the formal version. Read this for the part that I actually feel.*
