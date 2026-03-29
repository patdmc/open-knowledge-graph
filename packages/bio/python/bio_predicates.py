#!/usr/bin/env python3
"""
Biology predicate vocabulary for assertion-level knowledge graph.

Parallel to math PREDICATES in assertion_graph.py. Each predicate is a
molecular biology action or logical relation, detected by regex patterns
against biological text (papers, pathway descriptions, GO annotations).

The channel/passenger principle applies: the same predicate (e.g., "activates")
means different things in different pathway contexts. Domain context is carried
separately in the assertion signature, not baked into the predicate.

Usage:
    from bio_predicates import extract_bio_predicates, bio_signature
    preds = extract_bio_predicates("BRCA1 recruits RAD51 to resected DSB ends")
    sig = bio_signature(preds, context="DDR", direction="activating")
"""

import re
from typing import FrozenSet, Optional


# ── BIOLOGY PREDICATE PATTERNS ─────────────────────────────

BIOLOGY_PREDICATES = {
    # ── Enzymatic modifications ──────────────────────────
    "phosphorylates": [
        r"\bphosphorylat(?:e[sd]?|ion|ing)\b",
        r"\bkinase\s+activity\b",
    ],
    "dephosphorylates": [
        r"\bdephosphorylat(?:e[sd]?|ion|ing)\b",
        r"\bphosphatase\s+activity\b",
    ],
    "ubiquitinates": [
        r"\bubiquitinat(?:e[sd]?|ion|ing)\b",
        r"\bE3\s+(?:ubiquitin\s+)?ligase\b",
    ],
    "deubiquitinates": [
        r"\bdeubiquitinat(?:e[sd]?|ion|ing)\b",
        r"\bDUB\s+activity\b",
    ],
    "methylates": [
        r"\bmethylat(?:e[sd]?|ion|ing)\b(?!\s+CpG)",  # enzymatic, not DNA methylation
        r"\bmethyltransferase\b",
    ],
    "demethylates": [
        r"\bdemethylat(?:e[sd]?|ion|ing)\b",
        r"\bdemethylase\b",
    ],
    "acetylates": [
        r"\bacetylt(?:e[sd]?|ion|ing)\b",
        r"\bacetyltransferase\b",
        r"\bHAT\s+activity\b",
    ],
    "deacetylates": [
        r"\bdeacetylt(?:e[sd]?|ion|ing)\b",
        r"\bHDAC\b",
    ],
    "sumoylates": [
        r"\bsumoylat(?:e[sd]?|ion|ing)\b",
        r"\bSUMO\s+(?:conjugat|ligat)\b",
    ],
    "cleaves": [
        r"\bcleav(?:e[sd]?|age|ing)\b",
        r"\bproteas(?:e|olysis|olytic)\b",
        r"\bendopeptidase\b",
    ],

    # ── Physical interactions ────────────────────────────
    "binds": [
        r"\bbind(?:s|ing|bound)\b",
        r"\binteract(?:s|ing|ion)\s+with\b",
        r"\bassociat(?:e[sd]?|ion|ing)\s+with\b",
        r"\bcomplex(?:es)?\s+with\b",
    ],
    "recruits": [
        r"\brecruit(?:s|ed|ing|ment)\b",
        r"\btarget(?:s|ed|ing)\s+to\b",
    ],
    "scaffolds": [
        r"\bscaffold(?:s|ed|ing)\b",
        r"\bplatform\s+for\b",
        r"\bnucleation\s+(?:site|center)\b",
    ],
    "dimerizes": [
        r"\bdimeri[zs](?:e[sd]?|ation|ing)\b",
        r"\bhomodimer\b",
        r"\bheterodimer\b",
    ],
    "oligomerizes": [
        r"\boligomeri[zs](?:e[sd]?|ation|ing)\b",
        r"\bmultimer\b",
    ],

    # ── Regulatory actions ───────────────────────────────
    "activates": [
        r"\bactivat(?:e[sd]?|ion|ing|or)\b",
        r"\bstimulat(?:e[sd]?|ion|ing)\b",
        r"\bupstream\s+(?:activat|signal)\b",
    ],
    "inhibits": [
        r"\binhibit(?:s|ed|ing|ion|or)\b",
        r"\bsuppress(?:es|ed|ing|ion|or)\b",
        r"\bnegative(?:ly)?\s+regulat\b",
        r"\bantagoniz(?:e[sd]?|ing)\b",
    ],
    "stabilizes": [
        r"\bstabiliz(?:e[sd]?|ation|ing)\b",
        r"\bprevent(?:s|ed|ing)\s+degradation\b",
    ],
    "destabilizes": [
        r"\bdestabiliz(?:e[sd]?|ation|ing)\b",
        r"\bpromot(?:e[sd]?|ing)\s+degradation\b",
    ],
    "degrades": [
        r"\bdegrad(?:e[sd]?|ation|ing)\b",
        r"\bproteolytic\s+(?:cleavage|destruction)\b",
        r"\bturnover\b",
    ],

    # ── Transcriptional ──────────────────────────────────
    "transcribes": [
        r"\btranscri(?:b(?:e[sd]?|ing)|ption(?:al)?)\b",
        r"\bexpression\s+of\b",
        r"\bmRNA\s+(?:level|expression)\b",
    ],
    "represses": [
        r"\brepress(?:es|ed|ing|ion|or)\b",
        r"\bsilenc(?:e[sd]?|ing)\b",
        r"\btranscriptional(?:ly)?\s+(?:inactiv|silent)\b",
    ],
    "upregulates": [
        r"\bup-?regulat(?:e[sd]?|ion|ing)\b",
        r"\bincrease[sd]?\s+expression\b",
        r"\binduced?\s+expression\b",
    ],
    "downregulates": [
        r"\bdown-?regulat(?:e[sd]?|ion|ing)\b",
        r"\bdecrease[sd]?\s+expression\b",
        r"\breduced?\s+expression\b",
    ],

    # ── Localization ─────────────────────────────────────
    "translocates": [
        r"\btranslocat(?:e[sd]?|ion|ing)\b",
        r"\bshuttle[sd]?\s+(?:to|from|between)\b",
    ],
    "localizes": [
        r"\blocali[zs](?:e[sd]?|ation|ing)\b",
        r"\baccumulat(?:e[sd]?|ion|ing)\s+(?:at|in)\b",
    ],
    "exports": [
        r"\bnuclear\s+export\b",
        r"\bexport(?:s|ed|ing)\s+(?:from|to)\b",
    ],
    "imports": [
        r"\bnuclear\s+import\b",
        r"\bimport(?:s|ed|ing)\s+(?:from|to|into)\b",
    ],
    "secretes": [
        r"\bsecret(?:e[sd]?|ion|ing)\b",
        r"\bexocytos(?:is|ed)\b",
        r"\breleased?\s+(?:from|into|by)\b",
    ],

    # ── DNA/chromatin operations ─────────────────────────
    "repairs": [
        r"\brepair(?:s|ed|ing)?\b",
        r"\bDNA\s+(?:damage\s+)?repair\b",
        r"\b(?:homologous\s+)?recombination\b",
        r"\bNHEJ\b",
        r"\bBER\b",
        r"\bNER\b",
        r"\bMMR\b",
    ],
    "replicates": [
        r"\breplicat(?:e[sd]?|ion|ing)\b",
        r"\bDNA\s+synthesis\b",
        r"\breplication\s+fork\b",
    ],
    "remodels_chromatin": [
        r"\bchromatin\s+remodel(?:s|ed|ing|er)?\b",
        r"\bnucleosome\s+(?:repositioning|sliding|eviction)\b",
        r"\bSWI/SNF\b",
    ],
    "methylates_dna": [
        r"\bDNA\s+methylat(?:e[sd]?|ion|ing)\b",
        r"\bCpG\s+methylat\b",
        r"\bDNMT\b",
    ],
    "demethylates_dna": [
        r"\bDNA\s+demethylat(?:e[sd]?|ion|ing)\b",
        r"\bTET-mediated\b",
        r"\b5-?hmC\b",
    ],

    # ── Cell fate / signaling ────────────────────────────
    "arrests": [
        r"\b(?:cell\s+)?(?:cycle\s+)?arrest\b",
        r"\bG[12]/[SM]\s+(?:arrest|block|checkpoint)\b",
    ],
    "proliferates": [
        r"\bproliferat(?:e[sd]?|ion|ing)\b",
        r"\bcell\s+(?:division|growth)\b",
        r"\bmitogenic\b",
    ],
    "differentiates": [
        r"\bdifferentiat(?:e[sd]?|ion|ing)\b",
        r"\blineage\s+(?:commitment|specification)\b",
    ],
    "apoptosis": [
        r"\bapoptos(?:is|tic)\b",
        r"\bprogrammed\s+cell\s+death\b",
        r"\bcaspase\s+(?:activat|cleavage)\b",
    ],
    "senescence": [
        r"\bsenescen(?:ce|t)\b",
        r"\birreversible\s+(?:growth\s+)?arrest\b",
        r"\bSASP\b",
    ],
    "autophagy": [
        r"\bautophag(?:y|ic|osome)\b",
        r"\bself-?eat\b",
    ],

    # ── Immune-specific ──────────────────────────────────
    "presents_antigen": [
        r"\bantigen\s+present(?:ation|ing)\b",
        r"\bMHC\s+(?:class\s+)?[I]+\b",
        r"\bHLA\b.*\bpresent\b",
    ],
    "evades_immune": [
        r"\bimmune\s+(?:evasion|escape|checkpoint)\b",
        r"\bPD-?L1\b",
        r"\bimmune\s+suppress\b",
    ],

    # ── Logical / functional relations ───────────────────
    "requires": [
        r"\brequir(?:e[sd]?|ing|ement)\b",
        r"\bdepend(?:s|ent|ence)\s+on\b",
        r"\bnecessary\s+for\b",
    ],
    "sufficient_for": [
        r"\bsufficient\s+(?:for|to)\b",
    ],
    "compensates": [
        r"\bcompensat(?:e[sd]?|ion|ing|ory)\b",
        r"\bredundan(?:t|cy)\b",
        r"\bbackup\b",
        r"\bsynthetic\s+lethal\b",  # SL = loss of compensation
    ],
    "cooperates": [
        r"\bcooperat(?:e[sd]?|ion|ing)\b",
        r"\bsynerg(?:y|istic|ize)\b",
        r"\bepistatic\b",
    ],
    "antagonizes": [
        r"\bantagoniz(?:e[sd]?|ing|ism)\b",
        r"\bcounterbalance[sd]?\b",
        r"\bopposing\s+(?:effect|role|function)\b",
    ],
}

# ── BIOLOGY QUANTIFIERS ─────────────────────────────────

BIOLOGY_QUANTIFIERS = {
    "constitutive": [
        r"\bconstitutive(?:ly)?\b",
        r"\bbasal(?:ly)?\b",
        r"\balways\s+(?:active|on|expressed)\b",
    ],
    "conditional": [
        r"\bconditional(?:ly)?\b",
        r"\bin\s+response\s+to\b",
        r"\bupon\s+(?:stimulation|activation|damage)\b",
        r"\bwhen\s+(?:activated|phosphorylated|expressed)\b",
    ],
    "tissue_specific": [
        r"\btissue-?specific\b",
        r"\blineage-?(?:specific|restricted)\b",
        r"\bexpressed\s+in\s+(?:only\s+)?(?:the\s+)?(?:\w+\s+){0,2}(?:tissue|cell|organ)\b",
    ],
    "cell_cycle_dependent": [
        r"\bcell\s+cycle[- ]dependent\b",
        r"\b[GSM][12]?[- ](?:phase|specific)\b",
        r"\bmitotic\b",
    ],
}

# ── BIOLOGY CONTEXTS ─────────────────────────────────────
# These map to channels — the "tissue" that gives a predicate its meaning

BIOLOGY_CONTEXTS = {
    "ddr": [
        r"\bDNA\s+damage\b", r"\bDSB\b", r"\bSSB\b",
        r"\brepair\b", r"\bcheckpoint\b", r"\breplication\s+stress\b",
    ],
    "cell_cycle": [
        r"\bcell\s+cycle\b", r"\bmitos(?:is|tic)\b", r"\bcytokinesis\b",
        r"\bG[12]/[SM]\b", r"\bcheckpoint\b.*\bcycle\b",
    ],
    "growth_signaling": [
        r"\bgrowth\s+factor\b", r"\bPI3K\b", r"\bAKT\b", r"\bmTOR\b",
        r"\bMAPK\b", r"\bRAS\b", r"\bRAF\b", r"\bERK\b",
    ],
    "endocrine": [
        r"\bhormone\b", r"\bsteroid\b", r"\bestrogen\b", r"\bandrogen\b",
        r"\breceptor\s+(?:signaling|activation)\b",
    ],
    "immune": [
        r"\bimmun(?:e|ity|ological)\b", r"\bT\s+cell\b", r"\bB\s+cell\b",
        r"\bcytokine\b", r"\binflammation\b", r"\bantigen\b",
    ],
    "tissue_architecture": [
        r"\bcell\s+adhesion\b", r"\bextracellular\s+matrix\b",
        r"\btissue\s+(?:integrity|architecture|homeostasis)\b",
        r"\bWnt\b", r"\bNotch\b", r"\bHedgehog\b",
    ],
    "chromatin": [
        r"\bchromatin\b", r"\bhistone\b", r"\bnucleosome\b",
        r"\bepigenet(?:ic|ics)\b", r"\bSWI/SNF\b",
    ],
    "methylation": [
        r"\bDNA\s+methylat\b", r"\bCpG\b", r"\b5-?mC\b",
        r"\bDNMT\b", r"\bTET\b", r"\bIDH\b",
    ],
}


# ── EXTRACTION FUNCTIONS ─────────────────────────────────

def extract_bio_predicates(text: str) -> FrozenSet[str]:
    """Extract biology predicates from text."""
    found = set()
    for pred, patterns in BIOLOGY_PREDICATES.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.add(pred)
                break
    return frozenset(found)


def extract_bio_quantifiers(text: str) -> FrozenSet[str]:
    """Extract biology quantifiers (constitutive, conditional, etc.)."""
    found = set()
    for qtype, patterns in BIOLOGY_QUANTIFIERS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.add(qtype)
                break
    return frozenset(found)


def detect_context(text: str) -> Optional[str]:
    """Detect the biological context (channel) from text.

    Returns the most specific matching context, or None.
    This is the channel — the domain that gives predicates their meaning.
    """
    scores = {}
    for ctx, patterns in BIOLOGY_CONTEXTS.items():
        count = 0
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                count += 1
        if count > 0:
            scores[ctx] = count

    if not scores:
        return None
    return max(scores, key=scores.get)


def bio_signature(
    predicates: FrozenSet[str],
    context: Optional[str] = None,
    quantifiers: Optional[FrozenSet[str]] = None,
    direction: Optional[str] = None,
) -> str:
    """Create a canonical biology assertion signature.

    Three disambiguation axes (from the channel/passenger architecture):
    1. Domain context — which channel (DDR, immune, etc.)
    2. Quantifier — constitutive vs conditional vs tissue-specific
    3. Direction — activating vs inhibiting (not in math, critical in biology)

    Example: "DDR:conditional|phosphorylates+recruits|activating"
    """
    parts = []

    if context:
        parts.append(f"C:{context}")
    if quantifiers:
        parts.append("Q:" + "+".join(sorted(quantifiers)))
    if predicates:
        parts.append("P:" + "+".join(sorted(predicates)))
    if direction:
        parts.append(f"D:{direction}")

    return "|".join(parts) if parts else "atomic"
