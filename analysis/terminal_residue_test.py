"""
Terminal residue prediction test.

Hypothesis: mutations near the C-terminal end of a protein have
extreme HR values (either very protective or very harmful) because
they sit at graph branch boundaries — the last thing evolution added,
either critical or vestigial.

For each gene with enough recurrent mutations:
  1. Parse amino acid position from proteinChange
  2. Normalize position to [0, 1] (N-terminal = 0, C-terminal = 1)
  3. Compare HR distribution at terminals vs interior
  4. Test whether position correlates with |log HR| (extremity)
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    CHANNEL_MAP, HUB_GENES, NON_SILENT, MSK_DATASETS,
)


def parse_position(protein_change):
    """Extract amino acid position number from proteinChange string."""
    if not isinstance(protein_change, str):
        return None
    # Match patterns like R175H, E545K, K267Rfs*9, X307_splice, etc.
    m = re.search(r'[A-Z*]?(\d+)', protein_change)
    if m:
        return int(m.group(1))
    return None


# Approximate protein lengths for key genes (from UniProt)
PROTEIN_LENGTHS = {
    'TP53': 393, 'KRAS': 189, 'BRAF': 766, 'PIK3CA': 1068,
    'PTEN': 403, 'APC': 2843, 'EGFR': 1210, 'BRCA1': 1863,
    'BRCA2': 3418, 'ATM': 3056, 'RB1': 928, 'SMAD4': 552,
    'NF1': 2818, 'CDH1': 882, 'ARID1A': 2285, 'FBXW7': 707,
    'CTNNB1': 781, 'PIK3R1': 724, 'STK11': 433, 'MAP3K1': 1512,
    'ERBB2': 1255, 'FGFR3': 806, 'ESR1': 595, 'AR': 919,
    'GATA3': 443, 'FOXA1': 472, 'MYC': 439, 'CDKN2A': 156,
    'JAK1': 1154, 'JAK2': 1132, 'B2M': 119, 'NOTCH1': 2555,
    'MSH6': 1360, 'MSH2': 934, 'MLH1': 756, 'POLE': 2286,
    'POLD1': 1107, 'SMAD2': 467, 'SMAD3': 425, 'NRAS': 189,
    'AKT1': 480, 'MTOR': 2549, 'ERBB3': 1342, 'FGFR2': 821,
    'FGFR1': 822, 'MET': 1390, 'AXIN1': 862, 'AXIN2': 843,
    'KMT2C': 4911, 'KMT2D': 5537, 'CREBBP': 2442, 'DNMT3A': 912,
    'TET2': 2002, 'IDH1': 414, 'BAP1': 729, 'CHEK2': 543,
    'PALB2': 1186, 'RAD51C': 376, 'CDK4': 303,
    'TGFBR2': 567, 'NOTCH2': 2471, 'NOTCH3': 2321, 'NOTCH4': 2003,
    'HLA-A': 365, 'HLA-B': 362, 'HLA-C': 366,
}


def get_position_type(gene):
    for ch, hubs in HUB_GENES.items():
        if gene in hubs:
            return "hub"
    return "leaf"


def main():
    paths = MSK_DATASETS["msk_impact_50k"]
    print("Loading...", flush=True)
    mutations = pd.read_csv(paths["mutations"])
    clinical = pd.read_csv(paths["clinical"])
    sample_clinical = pd.read_csv(paths["sample_clinical"])

    mutations = mutations[mutations['mutationType'].isin(NON_SILENT)]
    mutations = mutations[mutations['gene.hugoGeneSymbol'].isin(CHANNEL_MAP.keys())]

    clinical = clinical.merge(
        sample_clinical[['patientId', 'CANCER_TYPE']].drop_duplicates(),
        on='patientId', how='left'
    )
    clinical = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    clinical['event'] = clinical['OS_STATUS'].apply(
        lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0
    )
    clinical['age'] = pd.to_numeric(clinical.get('AGE_AT_DX', 60), errors='coerce').fillna(60)
    clinical['age_z'] = (clinical['age'] - clinical['age'].mean()) / (clinical['age'].std() + 1e-8)

    # Load atlas for per-mutation HRs
    atlas_path = os.path.join(os.path.dirname(__file__), "survival_atlas_full.csv")
    atlas = pd.read_csv(atlas_path)
    tier1 = atlas[atlas['tier'] == 1].copy()

    # Parse positions in atlas
    tier1['aa_pos'] = tier1['protein_change'].apply(parse_position)
    tier1 = tier1.dropna(subset=['aa_pos'])
    tier1['aa_pos'] = tier1['aa_pos'].astype(int)

    # Add protein length and normalized position
    tier1['protein_length'] = tier1['gene'].map(PROTEIN_LENGTHS)
    tier1 = tier1.dropna(subset=['protein_length'])
    tier1['norm_pos'] = tier1['aa_pos'] / tier1['protein_length']
    tier1['log_hr'] = np.log(tier1['hr'])
    tier1['abs_log_hr'] = tier1['log_hr'].abs()
    tier1['gene_position'] = tier1['gene'].map(get_position_type)

    print(f"Tier 1 mutations with position info: {len(tier1)}", flush=True)

    # ===================================================================
    # GLOBAL: position vs HR extremity
    # ===================================================================
    print(f"\n{'='*70}")
    print("POSITION vs HR EXTREMITY (all tier 1 mutations)")
    print(f"{'='*70}", flush=True)

    # Correlation: normalized position vs |log HR|
    r, p = stats.spearmanr(tier1['norm_pos'], tier1['abs_log_hr'])
    print(f"  norm_pos vs |log HR|: rho={r:+.3f}, p={p:.4f}")

    r2, p2 = stats.spearmanr(tier1['norm_pos'], tier1['log_hr'])
    print(f"  norm_pos vs log HR (direction): rho={r2:+.3f}, p={p2:.4f}")

    # Terminal vs interior
    terminal_thresh = 0.1  # last 10% of protein
    terminal = tier1[tier1['norm_pos'] > (1 - terminal_thresh)]
    n_terminal = tier1[(tier1['norm_pos'] < terminal_thresh)]
    interior = tier1[(tier1['norm_pos'] >= terminal_thresh) &
                      (tier1['norm_pos'] <= (1 - terminal_thresh))]

    print(f"\n  C-terminal (last 10%): {len(terminal)} mutations")
    print(f"    mean |log HR| = {terminal['abs_log_hr'].mean():.3f}")
    print(f"    mean log HR   = {terminal['log_hr'].mean():.3f}")
    print(f"    mean HR       = {terminal['hr'].mean():.3f}")

    print(f"  N-terminal (first 10%): {len(n_terminal)} mutations")
    print(f"    mean |log HR| = {n_terminal['abs_log_hr'].mean():.3f}")
    print(f"    mean log HR   = {n_terminal['log_hr'].mean():.3f}")
    print(f"    mean HR       = {n_terminal['hr'].mean():.3f}")

    print(f"  Interior (10-90%): {len(interior)} mutations")
    print(f"    mean |log HR| = {interior['abs_log_hr'].mean():.3f}")
    print(f"    mean log HR   = {interior['log_hr'].mean():.3f}")
    print(f"    mean HR       = {interior['hr'].mean():.3f}")

    # Mann-Whitney: are terminals more extreme?
    if len(terminal) >= 3 and len(interior) >= 3:
        u, p = stats.mannwhitneyu(terminal['abs_log_hr'], interior['abs_log_hr'],
                                    alternative='greater')
        print(f"\n  C-terminal more extreme than interior? U={u:.0f}, p={p:.4f}")

    # ===================================================================
    # PER-GENE: position vs HR
    # ===================================================================
    print(f"\n{'='*70}")
    print("PER-GENE: POSITION vs HR")
    print(f"{'='*70}", flush=True)

    genes_with_data = tier1.groupby('gene').filter(lambda x: len(x) >= 5)['gene'].unique()

    gene_results = []
    for gene in sorted(genes_with_data):
        gdf = tier1[tier1['gene'] == gene]
        if len(gdf) < 5:
            continue

        r, p = stats.spearmanr(gdf['norm_pos'], gdf['log_hr'])
        r_abs, p_abs = stats.spearmanr(gdf['norm_pos'], gdf['abs_log_hr'])

        # Terminal mutations for this gene
        g_terminal = gdf[gdf['norm_pos'] > 0.9]
        g_interior = gdf[gdf['norm_pos'] <= 0.9]

        ch = CHANNEL_MAP.get(gene, '?')
        pos = get_position_type(gene)
        plen = PROTEIN_LENGTHS.get(gene, '?')

        gene_results.append({
            'gene': gene,
            'channel': ch,
            'position': pos,
            'protein_length': plen,
            'n_mutations': len(gdf),
            'rho_pos_vs_loghr': r,
            'p_pos_vs_loghr': p,
            'rho_pos_vs_abs_loghr': r_abs,
            'p_pos_vs_abs_loghr': p_abs,
            'n_terminal': len(g_terminal),
            'n_interior': len(g_interior),
            'mean_hr_terminal': g_terminal['hr'].mean() if len(g_terminal) > 0 else np.nan,
            'mean_hr_interior': g_interior['hr'].mean() if len(g_interior) > 0 else np.nan,
        })

        sig = "*" if p < 0.05 else ""
        sig_abs = "*" if p_abs < 0.05 else ""
        print(f"  {gene:15s} ({ch}/{pos}, {plen}aa, n={len(gdf):2d}) "
              f"pos→HR: rho={r:+.3f}{sig:1s}  pos→|HR|: rho={r_abs:+.3f}{sig_abs:1s}",
              flush=True)

    # ===================================================================
    # DETAILED: mutations sorted by position for key genes
    # ===================================================================
    print(f"\n{'='*70}")
    print("KEY GENES: MUTATIONS BY POSITION")
    print(f"{'='*70}", flush=True)

    for gene in ['TP53', 'KRAS', 'PIK3CA', 'PTEN', 'BRAF', 'APC', 'EGFR',
                  'SMAD4', 'ARID1A', 'ESR1']:
        gdf = tier1[tier1['gene'] == gene].sort_values('aa_pos')
        if len(gdf) < 3:
            continue

        plen = PROTEIN_LENGTHS.get(gene, '?')
        print(f"\n  {gene} ({plen} aa):")
        for _, r in gdf.iterrows():
            zone = "C-TERM" if r['norm_pos'] > 0.9 else "N-TERM" if r['norm_pos'] < 0.1 else "      "
            direction = "HARMFUL " if r['hr'] > 1.1 else "PROTECT " if r['hr'] < 0.9 else "NEUTRAL "
            sig = "*" if r['p_value'] < 0.05 else " "
            print(f"    pos {r['aa_pos']:5.0f} ({r['norm_pos']:.2f}) {zone} "
                  f"{r['protein_change']:20s} HR={r['hr']:.2f} {direction}{sig} "
                  f"[{r['cancer_type']}]")

    # ===================================================================
    # Also test with gene-level (tier 2) — do C-terminal-heavy genes differ?
    # ===================================================================

    # For each mutation in the raw data, compute what fraction of a gene's
    # mutations are in the last 20% of the protein
    print(f"\n{'='*70}")
    print("C-TERMINAL MUTATION FRACTION vs GENE-LEVEL HR")
    print(f"{'='*70}", flush=True)

    mutations['aa_pos'] = mutations['proteinChange'].apply(parse_position)
    mutations['protein_length'] = mutations['gene.hugoGeneSymbol'].map(PROTEIN_LENGTHS)
    has_pos = mutations.dropna(subset=['aa_pos', 'protein_length'])
    has_pos['norm_pos'] = has_pos['aa_pos'] / has_pos['protein_length']

    gene_cterm_frac = (has_pos.groupby('gene.hugoGeneSymbol')
                       .apply(lambda df: (df['norm_pos'] > 0.8).mean())
                       .reset_index())
    gene_cterm_frac.columns = ['gene', 'cterm_fraction']

    # Merge with tier 2 atlas
    tier2 = atlas[atlas['tier'] == 2].copy()
    tier2 = tier2.merge(gene_cterm_frac, on='gene', how='inner')

    if len(tier2) > 10:
        r, p = stats.spearmanr(tier2['cterm_fraction'], tier2['hr'])
        print(f"  C-terminal mutation fraction vs gene-level HR: rho={r:+.3f}, p={p:.4f}")

    # Save
    gene_df = pd.DataFrame(gene_results)
    gene_df.to_csv(os.path.join(os.path.dirname(__file__), "terminal_residue_results.csv"),
                    index=False)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
