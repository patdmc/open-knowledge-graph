"""
Download and cache all data sources for the Language Graph.

Data sources (in priority order):
  1. WordNet 3.1 — lexemes, semantic relations (~117K synsets, ~200K edges)
  2. FrameNet 1.7 — semantic frames, lexical units (~1.2K frames, ~13K LUs)
  3. ConceptNet 5 — causal/enrichment edges (future)
  4. Wiktionary — morphemes, inflections (future)
  5. Wikipedia corpus stats — collocation holdback edges (future)
  6. VerbNet — selectional preferences (future)

Usage:
    python -m language_graph.data.download_sources [--framenet]
"""

import os
import ssl
import sys

# Workaround for macOS Python SSL cert issues
ssl._create_default_https_context = ssl._create_unverified_context


def download_wordnet():
    """Download WordNet + Open Multilingual Wordnet via NLTK."""
    import nltk

    # Download to a project-local directory so it's portable
    nltk_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache", "nltk_data"
    )
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_data_dir)

    print(f"Downloading WordNet to {nltk_data_dir} ...")
    nltk.download("wordnet", download_dir=nltk_data_dir)
    nltk.download("omw-1.4", download_dir=nltk_data_dir)

    # Verify
    from nltk.corpus import wordnet as wn

    n_synsets = len(list(wn.all_synsets()))
    n_lemmas = sum(len(s.lemmas()) for s in wn.all_synsets())

    print(f"WordNet loaded: {n_synsets:,} synsets, {n_lemmas:,} lemma entries")
    print(f"  Nouns:      {len(list(wn.all_synsets('n'))):,}")
    print(f"  Verbs:      {len(list(wn.all_synsets('v'))):,}")
    print(f"  Adjectives: {len(list(wn.all_synsets('a'))):,}")
    print(f"  Adverbs:    {len(list(wn.all_synsets('r'))):,}")

    # Quick edge count preview
    n_hyper = sum(len(s.hypernyms()) for s in wn.all_synsets())
    n_mero = sum(
        len(s.part_meronyms()) + len(s.substance_meronyms()) + len(s.member_meronyms())
        for s in wn.all_synsets()
    )
    n_entail = sum(len(s.entailments()) for s in wn.all_synsets())
    n_antonym = sum(
        1 for s in wn.all_synsets() for l in s.lemmas() if l.antonyms()
    )

    print(f"\nRelation counts:")
    print(f"  Hypernym:   {n_hyper:,}")
    print(f"  Meronym:    {n_mero:,}")
    print(f"  Entailment: {n_entail:,}")
    print(f"  Antonym:    {n_antonym:,} (lemma-level)")

    return nltk_data_dir


def download_framenet():
    """Download FrameNet 1.7 via NLTK (for later use)."""
    import nltk

    nltk_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache", "nltk_data"
    )
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_data_dir)

    print(f"Downloading FrameNet to {nltk_data_dir} ...")
    nltk.download("framenet_v17", download_dir=nltk_data_dir)

    from nltk.corpus import framenet as fn

    n_frames = len(fn.frames())
    n_lus = len(fn.lus())
    print(f"FrameNet loaded: {n_frames:,} frames, {n_lus:,} lexical units")

    return nltk_data_dir


if __name__ == "__main__":
    download_wordnet()
    if "--framenet" in sys.argv:
        download_framenet()
    print("\nDone.")
