"""
Train bilinear edge-type model — 9D edge-type interaction matrix.

Instead of projecting N-dimensional patient data through a transformer into
a scalar hazard, we preserve the graph's 9 edge types as native dimensions.
For each mutation pair (i, j), the 9-dim edge vector e_ij encodes the
relationship through each edge type. The model learns a 9×9 interaction
matrix W per cancer type: score_ij = e_ij^T W e_ij. Patient risk is the
sum of scores across all mutation pairs.

Total parameters: 81 × n_cancer_types (~2,600 for 32 CTs).
Forward pass: sub-second. Training: minutes.

Usage:
    python3 -u -m gnn.scripts.train_bilinear_edge [--epochs 100]
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.data.atlas_dataset import AtlasDataset, MAX_NODES
from gnn.models.cox_sage import cox_ph_loss
from gnn.config import CHANNEL_MAP, CHANNEL_NAMES, ALL_GENES

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "bilinear_edge",
)

# The 9 edge types in the knowledge graph (order matters for W interpretation)
EDGE_TYPES = [
    # Original 9
    'PPI',
    'COUPLES',
    'SYNTHETIC_LETHAL',
    'CO_ESSENTIAL',
    'CO_EXPRESSED',
    'CO_CNA',
    'ATTENDS_TO',
    'HAS_SENSITIVITY_EVIDENCE',
    'HAS_RESISTANCE_EVIDENCE',
    # Projected drug edges
    'CO_SENSITIVE',
    'CO_RESISTANT',
    'DRUG_CONFLICT',
    # Projected tissue/biallelic edges
    'CO_TISSUE_EXPR',
    'CO_BIALLELIC',
    # Existing unused edges
    'ANALOGOUS',
    'CONVERGES',
    'TRANSPOSES',
    # Gene-pair attribute (not a Neo4j edge type)
    'SAME_STRAND',
]
N_EDGE_TYPES = len(EDGE_TYPES)

STRAND_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "gene_strand_data.json",
)

# Clinical covariates (MSK-MET 2021 full clinical file)
CLINICAL_COVARIATES = [
    'age',              # AGE_AT_SEQUENCING, normalized
    'sex',              # 1=male, 0=female
    'fga',              # fraction genome altered
    'msi_score',        # microsatellite instability score
    'met_site_count',   # number of metastatic sites
    'tumor_purity',     # tumor purity (0-100, normalized)
    'sample_type',      # 1=metastasis, 0=primary
    'tmb',              # tumor mutation burden (nonsynonymous)
]
N_CLINICAL = len(CLINICAL_COVARIATES)

FULL_CLINICAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "analysis", "cache", "msk_met_2021_full_clinical.csv",
)


# =========================================================================
# Load raw edge-type matrix from Neo4j
# =========================================================================

def load_raw_edge_matrix(gene_vocab):
    """Build (G, G, D) matrix — one dimension per edge type, no lossy projection.

    Each entry [i, j, k] is the raw weight of edge type k between genes i and j.
    D = N_EDGE_TYPES (currently 17).
    """
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))

    G = len(gene_vocab)
    matrix = np.zeros((G, G, N_EDGE_TYPES), dtype=np.float32)

    print("  Loading raw edge-type matrix from Neo4j...", flush=True)
    t0 = time.time()

    with driver.session() as s:
        for k, etype in enumerate(EDGE_TYPES):
            if etype == 'SAME_STRAND':
                continue  # populated below from strand data
            result = s.run(f"""
                MATCH (a:Gene)-[r:{etype}]->(b:Gene)
                WHERE a.channel IS NOT NULL AND b.channel IS NOT NULL
                  AND (r.deprecated IS NULL OR r.deprecated = false)
                RETURN a.name AS g1, b.name AS g2, properties(r) AS props
            """)
            n = 0
            for r in result:
                g1, g2 = r['g1'], r['g2']
                if g1 not in gene_vocab or g2 not in gene_vocab:
                    continue
                i, j = gene_vocab[g1], gene_vocab[g2]
                props = r['props'] or {}

                # Raw weight per edge type
                w = _raw_weight(etype, props)
                matrix[i, j, k] = max(matrix[i, j, k], w)
                matrix[j, i, k] = max(matrix[j, i, k], w)
                n += 1

            print(f"    {etype}: {n:,} edges", flush=True)

    driver.close()

    # Populate SAME_STRAND from strand data file
    strand_k = EDGE_TYPES.index('SAME_STRAND')
    if os.path.exists(STRAND_DATA_PATH):
        with open(STRAND_DATA_PATH) as f:
            strand_data = json.load(f)
        strand_lookup = {g: info['strand'] for g, info in strand_data.items()}
        n_same = 0
        for g1, idx1 in gene_vocab.items():
            s1 = strand_lookup.get(g1)
            if s1 is None:
                continue
            for g2, idx2 in gene_vocab.items():
                s2 = strand_lookup.get(g2)
                if s2 is None:
                    continue
                if s1 == s2:
                    matrix[idx1, idx2, strand_k] = 1.0
                    n_same += 1
        print(f"    SAME_STRAND: {n_same:,} pairs", flush=True)
    else:
        print(f"    SAME_STRAND: no strand data found, skipping", flush=True)

    # Self-loops: identity across all edge types (a gene is maximally related to itself)
    for i in range(G):
        matrix[i, i, :] = 1.0

    print(f"  Raw edge matrix: ({G}, {G}, {N_EDGE_TYPES}) [{time.time()-t0:.1f}s]")
    return matrix


def load_clinical_covariates(patient_ids):
    """Load patient-level clinical covariates from MSK-MET 2021 full clinical.

    Returns: (N_patients, N_CLINICAL) tensor, same order as patient_ids.
    """
    if not os.path.exists(FULL_CLINICAL_PATH):
        print(f"  WARNING: {FULL_CLINICAL_PATH} not found, skipping clinical covariates")
        return None

    df = pd.read_csv(FULL_CLINICAL_PATH)
    df = df.set_index('patientId')

    covs = np.zeros((len(patient_ids), N_CLINICAL), dtype=np.float32)

    for i, pid in enumerate(patient_ids):
        if pid not in df.index:
            continue
        row = df.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # age (from AGE_AT_SEQUENCING)
        age = pd.to_numeric(row.get('AGE_AT_SEQUENCING'), errors='coerce')
        covs[i, 0] = age if pd.notna(age) else 60.0

        # sex
        covs[i, 1] = 1.0 if row.get('SEX') == 'Male' else 0.0

        # FGA
        fga = pd.to_numeric(row.get('FRACTION_GENOME_ALTERED'), errors='coerce')
        covs[i, 2] = fga if pd.notna(fga) else 0.0

        # MSI score
        msi = pd.to_numeric(row.get('MSI_SCORE'), errors='coerce')
        covs[i, 3] = msi if pd.notna(msi) else 0.0

        # Met site count
        msc = pd.to_numeric(row.get('MET_SITE_COUNT'), errors='coerce')
        covs[i, 4] = msc if pd.notna(msc) else 0.0

        # Tumor purity
        tp = pd.to_numeric(row.get('TUMOR_PURITY'), errors='coerce')
        covs[i, 5] = tp / 100.0 if pd.notna(tp) else 0.4  # normalize to 0-1

        # Sample type (primary vs metastasis)
        covs[i, 6] = 1.0 if row.get('SAMPLE_TYPE') == 'Metastasis' else 0.0

        # TMB
        tmb = pd.to_numeric(row.get('TMB_NONSYNONYMOUS'), errors='coerce')
        covs[i, 7] = tmb if pd.notna(tmb) else 0.0

    # Normalize each column (z-score, except binary columns)
    for col_idx in [0, 2, 3, 4, 5, 7]:  # age, fga, msi, met_count, purity, tmb
        vals = covs[:, col_idx]
        mu = vals.mean()
        std = vals.std()
        if std > 1e-8:
            covs[:, col_idx] = (vals - mu) / std

    n_matched = sum(1 for pid in patient_ids if pid in df.index)
    print(f"  Clinical covariates: {n_matched}/{len(patient_ids)} patients matched")
    return torch.tensor(covs, dtype=torch.float32)


def _raw_weight(etype, props):
    """Extract raw weight from edge properties — no lossy projection."""
    if etype == 'PPI':
        return min(float(props.get('score', 500)) / 1000.0, 1.0)
    elif etype in ('CO_EXPRESSED', 'CO_CNA', 'CO_ESSENTIAL'):
        return abs(float(props.get('correlation', 0)))
    elif etype == 'COUPLES':
        return float(props.get('weight', 0.5))
    elif etype == 'SYNTHETIC_LETHAL':
        return 1.0
    elif etype == 'ATTENDS_TO':
        return float(props.get('weight', 0))
    elif etype == 'HAS_SENSITIVITY_EVIDENCE':
        return 1.0
    elif etype == 'HAS_RESISTANCE_EVIDENCE':
        return 1.0
    # Projected drug edges
    elif etype in ('CO_SENSITIVE', 'CO_RESISTANT', 'DRUG_CONFLICT'):
        return float(props.get('weight', 0))
    # Projected tissue/biallelic edges
    elif etype == 'CO_TISSUE_EXPR':
        return float(props.get('weight', 0))
    elif etype == 'CO_BIALLELIC':
        return float(props.get('weight', 0))
    # Existing unused edges
    elif etype == 'ANALOGOUS':
        return float(props.get('cosine_similarity', 0))
    elif etype == 'CONVERGES':
        return min(float(props.get('odds_ratio', 1)) / 10.0, 1.0)
    elif etype == 'TRANSPOSES':
        return 1.0
    return 0.0


# =========================================================================
# Bilinear edge-type model
# =========================================================================

class BilinearEdgeModel(nn.Module):
    """Bilinear interaction matrix per cancer type + clinical covariates.

    For each patient:
      1. Gather the D-dim edge vectors for all mutation pairs
      2. Compute e_ij^T W_ct e_ij for each pair
      3. Add clinical covariate linear term: β_ct · x_clinical
      4. Sum across pairs → scalar hazard

    Parameters: D×D×n_CT + D×n_CT + n_clinical×n_CT + n_CT
    """

    def __init__(self, n_edge_types, n_cancer_types, n_clinical=0):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.n_cancer_types = n_cancer_types
        self.n_clinical = n_clinical

        # Per-CT bilinear interaction matrix W (symmetric initialized)
        self.W = nn.Parameter(torch.zeros(n_cancer_types, n_edge_types, n_edge_types))
        nn.init.normal_(self.W, std=0.01)

        # Global bias per cancer type
        self.bias = nn.Parameter(torch.zeros(n_cancer_types))

        # Per-CT diagonal scaling
        self.diag_scale = nn.Parameter(torch.ones(n_cancer_types, n_edge_types) * 0.1)

        # Per-CT clinical covariate weights
        if n_clinical > 0:
            self.beta_clinical = nn.Parameter(torch.zeros(n_cancer_types, n_clinical))
            nn.init.normal_(self.beta_clinical, std=0.01)

    def forward(self, edge_feats, masks, cancer_types, clinical=None):
        """
        Args:
            edge_feats: (B, N, N, D) — raw edge-type vectors per mutation pair
            masks: (B, N) — 1 for real mutations, 0 for padding
            cancer_types: (B,) — cancer type indices
            clinical: (B, n_clinical) — patient-level clinical covariates (optional)

        Returns:
            hazard: (B,) — scalar hazard per patient
        """
        B, N, _, D = edge_feats.shape

        # Get per-patient W matrix
        W_ct = self.W[cancer_types]  # (B, D, D)

        # Symmetrize W
        W_sym = (W_ct + W_ct.transpose(-1, -2)) / 2

        # Add diagonal scaling
        diag = self.diag_scale[cancer_types]  # (B, D)
        W_sym = W_sym + torch.diag_embed(diag)

        # Pair mask
        pair_mask = masks.unsqueeze(1) * masks.unsqueeze(2)

        # Bilinear: e_ij^T W e_ij
        We = torch.einsum('bad,bnmd->bnma', W_sym, edge_feats)
        scores = (edge_feats * We).sum(dim=-1)

        # Mask and sum
        scores = scores * pair_mask
        hazard = scores.sum(dim=(1, 2)) + self.bias[cancer_types]

        # Clinical covariate linear term
        if clinical is not None and self.n_clinical > 0:
            beta = self.beta_clinical[cancer_types]  # (B, n_clinical)
            hazard = hazard + (beta * clinical).sum(dim=-1)

        return hazard

    def get_W_matrix(self, ct_idx):
        """Return the interpretable W matrix for a cancer type."""
        with torch.no_grad():
            W = self.W[ct_idx]
            W_sym = (W + W.T) / 2
            W_sym += torch.diag(self.diag_scale[ct_idx])
        return W_sym.cpu().numpy()


# =========================================================================
# Training
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--l2', type=float, default=0.01,
                        help='L2 regularization on W (encourages sparsity in interactions)')
    parser.add_argument('--dataset', type=str, default='msk_impact_50k',
                        choices=['msk_impact_50k', 'msk_met_2021'],
                        help='Dataset to train on')
    parser.add_argument('--clinical', action='store_true',
                        help='Include patient-level clinical covariates')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # === Load patient data ===
    print(f"Loading data (dataset={args.dataset})...", flush=True)
    ds = AtlasDataset(dataset_name=args.dataset)
    data = ds.build_features()

    # === Build gene vocabulary ===
    gene_vocab = {}
    for patient_genes in data['gene_names']:
        for g in patient_genes:
            if g and g != '' and g != 'WT' and g not in gene_vocab:
                gene_vocab[g] = len(gene_vocab)
    G = len(gene_vocab)
    print(f"  Gene vocabulary: {G} unique genes")

    # Map per-patient gene names → vocabulary indices
    gene_indices = np.zeros((len(data['gene_names']), MAX_NODES), dtype=np.int64)
    for b, patient_genes in enumerate(data['gene_names']):
        for s, g in enumerate(patient_genes):
            if g and g != '' and g != 'WT' and g in gene_vocab:
                gene_indices[b, s] = gene_vocab[g]
    gene_indices_t = torch.tensor(gene_indices, dtype=torch.long)

    # === Load raw 9D edge-type matrix ===
    cache_path = os.path.join(RESULTS_DIR, "raw_edge_matrix.npy")
    vocab_path = os.path.join(RESULTS_DIR, "gene_vocab.json")
    if os.path.exists(cache_path) and os.path.exists(vocab_path):
        with open(vocab_path) as f:
            cached_vocab = json.load(f)
        if len(cached_vocab) == G:
            print("  Loading cached raw edge matrix...", flush=True)
            edge_matrix = np.load(cache_path)
        else:
            edge_matrix = load_raw_edge_matrix(gene_vocab)
            np.save(cache_path, edge_matrix)
            with open(vocab_path, 'w') as f:
                json.dump(gene_vocab, f)
    else:
        edge_matrix = load_raw_edge_matrix(gene_vocab)
        np.save(cache_path, edge_matrix)
        with open(vocab_path, 'w') as f:
            json.dump(gene_vocab, f)

    edge_matrix_t = torch.tensor(edge_matrix, dtype=torch.float32).to(device)
    mem_mb = edge_matrix.nbytes / 1e6
    print(f"  Edge matrix: ({G}, {G}, {N_EDGE_TYPES}) = {mem_mb:.1f}MB")

    # === Unpack data ===
    nm = data['node_masks'].to(device)
    ct = data['cancer_types'].to(device)
    times = data['times']
    events = data['events']
    n_cancer_types = data['n_cancer_types']
    gene_indices_t = gene_indices_t.to(device)

    N_patients = len(events)
    print(f"  Patients: {N_patients}, Cancer types: {n_cancer_types}")

    # === Load clinical covariates ===
    use_clinical = args.clinical
    clinical_feats = None
    n_clin = 0
    if use_clinical:
        patient_ids = ds.clinical['patientId'].tolist()
        clinical_feats = load_clinical_covariates(patient_ids)
        if clinical_feats is not None:
            n_clin = clinical_feats.shape[1]
            clinical_feats = clinical_feats.to(device)
            print(f"  Clinical covariates: {n_clin} features")
        else:
            use_clinical = False

    # === Precompute per-patient edge features ===
    # Gather from the (G, G, 9) matrix using gene indices
    # This is the key operation: for each patient, get (32, 32, 9) sub-tensor
    print("  Precomputing patient edge features...", flush=True)
    t0 = time.time()

    # Do in batches to manage memory
    batch_gather = 4096
    patient_edges = torch.zeros(N_patients, MAX_NODES, MAX_NODES, N_EDGE_TYPES,
                                dtype=torch.float32, device=device)
    for start in range(0, N_patients, batch_gather):
        end = min(start + batch_gather, N_patients)
        idx = gene_indices_t[start:end]  # (batch, 32)
        safe_idx = idx.clamp(0, G - 1)
        idx_i = safe_idx.unsqueeze(2).expand(-1, MAX_NODES, MAX_NODES)
        idx_j = safe_idx.unsqueeze(1).expand(-1, MAX_NODES, MAX_NODES)
        patient_edges[start:end] = edge_matrix_t[idx_i, idx_j]

        # Mask padding
        m = nm[start:end]
        pair_mask = (m.unsqueeze(1) * m.unsqueeze(2)).unsqueeze(-1)
        patient_edges[start:end] *= pair_mask

    mem_edges = patient_edges.nelement() * 4 / 1e6
    print(f"  Patient edges: {patient_edges.shape} = {mem_edges:.0f}MB [{time.time()-t0:.1f}s]")

    # === Holdback ===
    np.random.seed(args.seed)
    all_idx = np.arange(N_patients)
    np.random.shuffle(all_idx)
    n_holdback = int(N_patients * 0.15)
    holdback_idx = all_idx[:n_holdback]
    cv_idx = all_idx[n_holdback:]
    print(f"\nHoldback: {n_holdback}, CV pool: {len(cv_idx)}")

    events_cv = events[cv_idx].numpy()

    # === Cross-validation ===
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    best_global_state = None
    best_global_c = 0.0

    for fold, (train_rel, val_rel) in enumerate(skf.split(cv_idx, events_cv)):
        print(f"\n=== Fold {fold} ===")

        train_idx = cv_idx[train_rel]
        val_idx = cv_idx[val_rel]

        model = BilinearEdgeModel(N_EDGE_TYPES, n_cancer_types, n_clinical=n_clin).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_c = 0.0
        best_state = None
        no_improve = 0
        batch_size = args.batch_size

        for epoch in range(args.epochs):
            model.train()
            perm = np.random.permutation(len(train_idx))
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, len(perm), batch_size):
                b_rel = perm[b_start:b_start + batch_size]
                b_abs = torch.tensor(train_idx[b_rel], dtype=torch.long, device=device)

                optimizer.zero_grad()

                clin_batch = clinical_feats[b_abs] if use_clinical else None
                hazard = model(
                    patient_edges[b_abs],
                    nm[b_abs],
                    ct[b_abs],
                    clinical=clin_batch,
                )

                loss = cox_ph_loss(
                    hazard, times[b_abs].to(device), events[b_abs].to(device)
                )

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_abs = torch.tensor(val_idx, dtype=torch.long, device=device)
                    clin_val = clinical_feats[val_abs] if use_clinical else None
                    h_val = model(
                        patient_edges[val_abs],
                        nm[val_abs],
                        ct[val_abs],
                        clinical=clin_val,
                    ).cpu().numpy().flatten()

                e_val = events[val_idx].numpy().astype(bool)
                t_val = times[val_idx].numpy()
                valid = t_val > 0
                try:
                    c = concordance_index_censored(
                        e_val[valid], t_val[valid], h_val[valid]
                    )[0]
                except Exception:
                    c = 0.5

                if c > best_c:
                    best_c = c
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1:4d}: loss={avg_loss:.4f} C={c:.4f} best={best_c:.4f}",
                          flush=True)

                if no_improve >= args.patience:
                    print(f"    Early stop epoch {epoch+1}, C-index: {best_c:.4f}")
                    break

        fold_results.append(best_c)
        print(f"  Fold {fold}: C-index = {best_c:.4f}")

        if best_c > best_global_c:
            best_global_c = best_c
            best_global_state = best_state

    # === Summary ===
    mean_c = np.mean(fold_results)
    std_c = np.std(fold_results)
    print(f"\n{'='*60}")
    print(f"  BILINEAR EDGE-TYPE RESULTS")
    print(f"  Mean C-index: {mean_c:.4f} +/- {std_c:.4f}")
    print(f"  Per-fold: {[f'{c:.4f}' for c in fold_results]}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}")

    # === Holdback evaluation ===
    model.load_state_dict(best_global_state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        h_abs = torch.tensor(holdback_idx, dtype=torch.long, device=device)
        clin_hb = clinical_feats[h_abs] if use_clinical else None
        h_pred = model(
            patient_edges[h_abs],
            nm[h_abs],
            ct[h_abs],
            clinical=clin_hb,
        ).cpu().numpy().flatten()

    e_hb = events[holdback_idx].numpy().astype(bool)
    t_hb = times[holdback_idx].numpy()
    valid_hb = t_hb > 0
    c_hb = concordance_index_censored(e_hb[valid_hb], t_hb[valid_hb], h_pred[valid_hb])[0]
    print(f"  Holdback C-index: {c_hb:.4f}")

    # === Per-CT holdback C-index ===
    ct_names = ds.ct_encoder.classes_ if hasattr(ds, 'ct_encoder') else None
    ct_hb = ct[holdback_idx].cpu().numpy()
    print(f"\n  Per-CT holdback C-index:")
    ct_results = {}
    for ct_idx_val in np.unique(ct_hb):
        mask_ct = ct_hb == ct_idx_val
        e_ct = e_hb[mask_ct]
        t_ct = t_hb[mask_ct]
        h_ct = h_pred[mask_ct]
        valid_ct = t_ct > 0
        n_pts = mask_ct.sum()
        n_events = e_ct.sum()
        if n_pts >= 20 and n_events >= 5:
            try:
                c_ct = concordance_index_censored(e_ct[valid_ct], t_ct[valid_ct], h_ct[valid_ct])[0]
                ct_name = ct_names[ct_idx_val] if ct_names is not None else f"CT_{ct_idx_val}"
                print(f"    {ct_name}: C={c_ct:.4f} (n={n_pts})")
                ct_results[ct_name] = {'c_index': c_ct, 'n': int(n_pts), 'events': int(n_events)}
            except Exception:
                pass

    # === Interpretability: print W matrices for top cancer types ===
    print(f"\n{'='*60}")
    print(f"  LEARNED W MATRICES (top cancer types)")
    print(f"{'='*60}")

    # Get top 5 CTs by patient count on holdback
    ct_counts = defaultdict(int)
    for c in ct_hb:
        ct_counts[c] += 1
    top_cts = sorted(ct_counts, key=ct_counts.get, reverse=True)[:5]

    for ct_idx_val in top_cts:
        ct_name = ct_names[ct_idx_val] if ct_names is not None else f"CT_{ct_idx_val}"
        W = model.get_W_matrix(ct_idx_val)
        print(f"\n  {ct_name}:")
        print(f"  {'':12s} " + " ".join(f"{et[:6]:>7s}" for et in EDGE_TYPES))
        for i, et_i in enumerate(EDGE_TYPES):
            vals = " ".join(f"{W[i,j]:7.3f}" for j in range(N_EDGE_TYPES))
            print(f"  {et_i[:12]:12s} {vals}")

    # === Clinical covariate coefficients ===
    if use_clinical:
        print(f"\n{'='*60}")
        print(f"  CLINICAL COVARIATE COEFFICIENTS (top cancer types)")
        print(f"{'='*60}")
        for ct_idx_val in top_cts:
            ct_name = ct_names[ct_idx_val] if ct_names is not None else f"CT_{ct_idx_val}"
            beta = model.beta_clinical[ct_idx_val].detach().cpu().numpy()
            print(f"\n  {ct_name}:")
            for k, cov_name in enumerate(CLINICAL_COVARIATES):
                print(f"    {cov_name:20s}: {beta[k]:+.4f}")

    # === Save results ===
    results = {
        'mean_c_index': float(mean_c),
        'std_c_index': float(std_c),
        'fold_results': [float(c) for c in fold_results],
        'holdback_c_index': float(c_hb),
        'n_params': n_params,
        'n_edge_types': N_EDGE_TYPES,
        'edge_types': EDGE_TYPES,
        'n_clinical': n_clin,
        'clinical_covariates': CLINICAL_COVARIATES if use_clinical else [],
        'dataset': args.dataset,
        'per_ct_holdback': ct_results,
        'args': vars(args),
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(best_global_state, os.path.join(RESULTS_DIR, 'best_model.pt'))

    # Save W matrices for all CTs
    W_all = {}
    for ct_idx_val in range(n_cancer_types):
        ct_name = ct_names[ct_idx_val] if ct_names is not None else f"CT_{ct_idx_val}"
        W_all[ct_name] = model.get_W_matrix(ct_idx_val).tolist()
    with open(os.path.join(RESULTS_DIR, 'W_matrices.json'), 'w') as f:
        json.dump(W_all, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
