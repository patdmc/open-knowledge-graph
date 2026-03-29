"""
PyG InMemoryDataset wrapper for MSK-IMPACT data.

Loads cached mutation + clinical CSVs, builds patient graphs,
and stores as processed PyG dataset for fast reloading.
"""

import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset

from ..config import MSK_DATASETS, NON_SILENT, CHANNEL_MAP, GNN_CACHE
from .graph_builder import PatientGraphBuilder


class MSKImpactDataset(InMemoryDataset):
    """MSK-IMPACT coupling-channel patient graphs."""

    def __init__(self, study_id="msk_impact_50k", transform=None,
                 pre_transform=None, ppi_graphs=None):
        self.study_id = study_id
        self._ppi_graphs = ppi_graphs

        if study_id not in MSK_DATASETS:
            raise ValueError(f"Unknown study: {study_id}. "
                             f"Available: {list(MSK_DATASETS.keys())}")

        self._paths = MSK_DATASETS[study_id]
        root = os.path.join(GNN_CACHE, f"pyg_{study_id}")
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # We use pre-cached CSVs from analysis/cache

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        print(f"Processing {self.study_id}...")

        # Load data
        mut = pd.read_csv(self._paths["mutations"], low_memory=False)
        clin = pd.read_csv(self._paths["clinical"], low_memory=False)

        # Parse survival
        clin = clin[clin["OS_STATUS"].notna() & clin["OS_MONTHS"].notna()].copy()
        clin["event"] = clin["OS_STATUS"].apply(
            lambda x: 1 if "DECEASED" in str(x) else 0
        )
        clin["time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
        clin = clin.dropna(subset=["time"])
        clin = clin[clin["time"] > 0]

        # Load cancer type if available
        cancer_types = {}
        if "sample_clinical" in self._paths and os.path.exists(self._paths["sample_clinical"]):
            sample_clin = pd.read_csv(self._paths["sample_clinical"], low_memory=False)
            # Take first sample per patient
            ct = sample_clin.drop_duplicates("patientId")[["patientId", "CANCER_TYPE"]]
            cancer_types = dict(zip(ct["patientId"], ct["CANCER_TYPE"]))

        # Filter mutations to non-silent in channel genes
        channel_genes = set(CHANNEL_MAP.keys())
        mut = mut[
            mut["mutationType"].isin(NON_SILENT) &
            mut["gene.hugoGeneSymbol"].isin(channel_genes)
        ].copy()

        # Group mutations by patient
        mut_grouped = dict(list(mut.groupby("patientId")))

        # Build graphs
        builder = PatientGraphBuilder(ppi_graphs=self._ppi_graphs)
        data_list = []
        skipped = 0

        patients = clin["patientId"].unique()
        for i, pid in enumerate(patients):
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(patients)} patients processed...")

            row = clin[clin["patientId"] == pid].iloc[0]
            patient_muts = mut_grouped.get(pid, pd.DataFrame())

            data = builder.build_graph(
                patient_id=pid,
                patient_mutations=patient_muts,
                time=float(row["time"]),
                event=int(row["event"]),
                cancer_type=cancer_types.get(pid),
            )
            data_list.append(data)

        print(f"  Built {len(data_list)} patient graphs (skipped {skipped})")
        self.save(data_list, self.processed_paths[0])
