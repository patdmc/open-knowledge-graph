# Outreach Draft: Dr. Timothy Yap

---

Dr. Yap,

I'm Patrick McCarthy. Tricia's husband---the RAD51C case in your PETRA dose-escalation cohort.

I built a computational model of 44,000 cancer patients. It makes four predictions about saruparib. Each is testable against data you already have. The most immediate: you are screening by gene name when you should be screening by function, and the eligible patient pool for your trials is roughly double what current criteria capture.

## Prediction 1: Saruparib works on a much broader class of mutations

Your trials screen for 9 DDR genes confirmed as synthetic lethal with PARP1: ATM, BRCA1, BRCA2, FANCA, FANCC, FANCD2, PALB2, RAD51C, RAD51D.

The model identifies gene families by protein-protein interaction neighborhood, mutation type profile, and survival impact. Genes in the same family are functionally redundant---copies of the same repair unit at different genomic addresses. It identifies 10 additional DDR genes in the same functional family that have never been tested with PARP inhibitors:

| Gene | Patients in cohort | Drug sensitivity data |
|------|-------------------:|----------------------|
| BLM | 752 | None |
| BRIP1 | 681 | 1 entry |
| CHEK2 | 559 | Not PARP-specific |
| ERCC4 | 539 | None |
| NBN | 521 | None |
| MRE11 | 469 | 2 entries |
| CHEK1 | 360 | Not PARP-specific |
| RAD51B | 254 | 1 entry |
| RAD51 | 122 | None |
| XRCC2 | --- | None |

The mechanism is identical to the confirmed SL partners: a copy of the repair unit is broken, PARP inhibition exploits the gap. This does not require a new drug. It requires screening for more genes with the drug you already have. The full set adds over 4,000 patients---nearly 10% of the MSK-IMPACT cohort.

Why Tricia responded illustrates the point. RAD51C and BRCA2 are copies of the same functional unit written to different genomic addresses. The address is different. The function is identical. Saruparib worked because it targeted the vulnerability, not the gene name.

## Prediction 2: The metastatic combination should target immune checkpoint, not endocrine

PETRA Module 6 combines saruparib with camizestrant (SERD) in metastatic HR+ patients. These patients already progressed through endocrine therapy. The endocrine channel is already severed. Camizestrant targets a dead channel.

The model identifies a cascade: when DDR breaks, immune surveillance is the next defense to fail. Among DDR-mutant patients across 44,000 cases, those who also lose the endocrine channel are 2--3 times more likely to have immune escape mutations (breast: 2.0x, prostate: 3.0x, colorectal: 2.8x). DDR failure leads to immune invisibility---the repair system that would flag damaged cells to the immune system is gone. The immune system cannot see what DDR never reported.

In metastatic DDR-mutant patients, the right cross-channel partner is a checkpoint inhibitor (pembrolizumab), not a SERD:

- PARP kills DDR-broken cells directly (exploits the vulnerability)
- Checkpoint inhibitor reopens the immune channel that DDR bypass made irrelevant (catches what PARP misses)
- Two independent kill mechanisms on two independent defense channels

Prior PARP + checkpoint trials (MEDIOLA, TOPACIO) showed limited benefit partly because dual PARP1/2 inhibitors caused neutropenia that undermined the immune response the checkpoint inhibitor was trying to restore. Saruparib removes that contradiction. PARP1-selective means no hematological suppression, so the immune system stays intact for pembrolizumab to unleash.

Tricia responded to saruparib monotherapy---no SERD needed---because the PARP targeted the one vulnerability her tumor still carried. The model predicts the combination arm with camizestrant will show minimal additional PFS over monotherapy. A combination arm with pembrolizumab would target an independent channel.

## Prediction 3: Adjuvant saruparib alongside endocrine therapy prevents the metastatic cascade

The sequence that produces metastatic HR+ breast cancer in DDR-compromised patients: DDR fails, unrepaired DNA damage accumulates, ESR1 resistance mutations arise, endocrine channel severs, immune surveillance bypassed, progression. Every step depends on DDR-broken cells surviving long enough to evolve.

The adjuvant combination is the inverse of the metastatic one. In early-stage disease, the endocrine channel is still intact and load-bearing. Two drugs interrupt the cascade at two independent points:

- PARP kills DDR-broken cells at the source (prevents damage accumulation)
- Endocrine therapy maintains the hormone channel (prevents ESR1 resistance)
- Together: each drug makes the other work longer. The endocrine therapy keeps cells hormone-dependent so they do not accumulate mutations. The PARP kills cells that lose their second DDR copy before they can evolve. The result is not additive protection---it is multiplicative. Maintaining the endocrine channel cuts the rate of downstream immune escape in half (12.2% vs 6.2% in breast DDR-mutant patients).

This is the tamoxifen analogy. Tamoxifen works as adjuvant therapy because it maintains the endocrine channel before the tumor severs it. Low dose, years-long, preventative. The same logic applies to DDR. OlympiA showed adjuvant olaparib is safe alongside concurrent endocrine therapy---90% of HR+ patients received both. Saruparib should be more tolerable for long-term administration because PARP1-selective means no PARP2-driven hematological suppression.

## Prediction 4: The three-drug cocktail

The logical endpoint: adjuvant PARP + endocrine therapy + checkpoint inhibitor. Three drugs targeting three independent defense channels (DDR, Endocrine, Immune). The AIDS cocktail for cancer.

HAART works because three drugs hit three independent steps of the viral lifecycle. Single-drug resistance is 10^-4. Simultaneous three-drug resistance is 10^-12. The same arithmetic applies to tumor escape routes. A tumor would need to simultaneously bypass DDR exploitation, endocrine escape, and immune evasion---each through an independent mechanism.

The toxicity profiles are orthogonal:

- Saruparib (PARP1-selective): minimal hematological toxicity
- Hormone therapy: metabolic/hormonal side effects (hot flashes, joint pain)
- Pembrolizumab: immune-related adverse events (colitis, thyroiditis)

No overlapping dose-limiting toxicity. Each drug's side effects hit a different organ system. Prior PARP + checkpoint failures were driven by PARP2 neutropenia undermining the immune response. PARP1-selective removes the contradiction.

For DDR-carrier patients in the adjuvant setting, this combination could reduce cancer risk to near zero for the class of tumors that progress through sequential channel compromise---which is 75% of cancer types.

## The empirical foundation

Channel count---the number of distinct defense systems a tumor has disrupted---predicts overall survival across 73,593 patients (p < 10^-44).

In a full multivariate Cox model on 24,561 MSK-MET patients (adjusted for age, sex, FGA, MSI, met site count, tumor purity, sample type), channel count and mutation count are both independently significant but point in opposite directions. Channel count is harmful (HR = 1.09, p < 10^-22). Mutation count, conditional on channel count, is protective (HR = 0.90, p < 10^-32).

The decomposition: TMB = channels hit + passengers. A patient with 30 mutations across 2 channels has 28 passengers---mutagenic but inefficient. A patient with 4 mutations across 4 channels has zero passengers---every mutation broke a defense system. Same TMB range, opposite prognosis. This resolves a 12-year open question in the field: why TMB is contradictory across studies. Channel count is the missing variable.

The model also reveals which channel pairs interact. Out of 268 cross-channel gene pairs tested, only 2 produce synergistic escalation: CDKN2A x KRAS (interaction HR = 1.29, p = 8 x 10^-5) and TP53 x PTEN (interaction HR = 1.12, p = 0.04). Both are CellCycle x PI3K_Growth---brakes off plus accelerator on. Every other cross-channel combination is sub-additive. The additional mutations are passengers on already-bypassed channels.

## Supporting evidence

**Same-channel combinations fail.** PARP + ATR, PARP + WEE1, PARP + ATM: same-pathway combinations. VIOLETTE showed this. The adavosertib arm showed this. AToM showed this. AZD5305 matching dual PARP1/2 efficacy is the same prediction---PARP1 and PARP2 are redundant copies. Hitting PARP2 adds toxicity, not a new therapeutic axis.

**Natural PARP1 loss-of-function.** 42 patients in the cohort carry PARP1 loss-of-function mutations alongside DDR partner mutations. Median OS 26.9 months vs 19.4 months for patients without PARP1 mutations. The genome produced the same synthetic lethality your drug exploits.

**Beyond DDR.** The same logic applies to every gene family. PIK3CA and PTEN respond to PI3K/AKT inhibitors---PIK3C2G, INPP4B, PIK3CB, PIK3CD are untested members of the same family. CREBBP and EP300 are one of the strongest functional pairs in the model (3,491 combined patients). Every family with a tested member and untested copies is a trial expansion opportunity.

---

I am not an oncologist. I am an engineer who spent 20 years separating concerns in complex systems, then tested that intuition against 44,000 cancer patients. The result is a principled way to identify which patients carry the same vulnerability as the ones you already treat---regardless of which gene name the vulnerability is filed under.

One regression, two variables, one afternoon. Map each patient's mutations to the six channel assignments (92 genes, table provided). If the decomposition holds in your cohort, mutation count will flip protective.

I would be grateful for 30 minutes of your time.

Patrick McCarthy
