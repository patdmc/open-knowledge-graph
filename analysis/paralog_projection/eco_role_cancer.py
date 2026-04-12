"""Cancer prevalence by ecological role across mammals.

Tests the framework prediction that 'nervous' species (prey) show elevated
cancer rates relative to 'calm' species (apex predators, megafauna) after
sample-size pooling.

Inputs: Vincze 2022 (cancer per species) + PanTHERIA (trophic, body mass).
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data" / "cross_mammal"

vincze = pd.read_csv(DATA / "vincze_2022_cancer.csv")
mam = vincze[vincze["Mammal"] == 1].copy()
mam["species_norm"] = mam["Species"].str.replace("_", " ")
mam["cancer_prev"] = mam["Neoplasia"] / mam["Sample"]

pan = pd.read_csv(DATA / "pantheria_raw.txt", sep="\t", na_values=["-999.00", "-999"])
pan = pan.rename(
    columns={
        "MSW05_Binomial": "species_norm",
        "5-1_AdultBodyMass_g": "body_g",
        "6-2_TrophicLevel": "trophic",
    }
)
pan = pan[["species_norm", "body_g", "trophic"]]

j = mam.merge(pan, on="species_norm", how="left")
# fall back to Vincze body mass when PanTHERIA missing
j["body_g"] = j["body_g"].fillna(j["FemaleMeanMass"] * 1000)
j["body_kg"] = j["body_g"] / 1000

print(f"Vincze mammals: {len(mam)}  |  joined with body mass: {j['body_kg'].notna().sum()}  |  with trophic: {j['trophic'].notna().sum()}")


def role(r):
    if pd.isna(r["body_kg"]) or pd.isna(r["trophic"]):
        return "unknown"
    if r["body_kg"] > 500:
        return "megafauna_unattackable"
    t = int(r["trophic"])
    if t == 3:
        return "apex_predator" if r["body_kg"] > 50 else "mesopredator"
    if t == 2:
        return "omnivore"
    if t == 1:
        return "large_herbivore" if r["body_kg"] > 50 else "small_prey"
    return "unknown"


j["eco_role"] = j.apply(role, axis=1)

print(f"\nRole assignments: {j['eco_role'].value_counts().to_dict()}")

# Group by role
g = j.groupby("eco_role").agg(
    n_species=("Species", "count"),
    total_sampled=("Sample", "sum"),
    total_neoplasia=("Neoplasia", "sum"),
    mean_per_species_prev=("cancer_prev", "mean"),
    median_per_species_prev=("cancer_prev", "median"),
)
g["pooled_prev"] = g["total_neoplasia"] / g["total_sampled"]
g = g.sort_values("pooled_prev", ascending=False).round(4)

print("\n=== Cancer prevalence by ecological role ===")
print(g.to_string())

print("\n=== Top 10 highest-cancer species ===")
top = j[["Species", "eco_role", "body_kg", "Sample", "Neoplasia", "cancer_prev"]].dropna(
    subset=["cancer_prev"]
).sort_values("cancer_prev", ascending=False).head(10)
print(top.to_string(index=False))

print("\n=== Bottom 10 lowest-cancer species (with sample >= 30) ===")
bot = (
    j[j["Sample"] >= 30][["Species", "eco_role", "body_kg", "Sample", "Neoplasia", "cancer_prev"]]
    .dropna(subset=["cancer_prev"])
    .sort_values("cancer_prev")
    .head(10)
)
print(bot.to_string(index=False))

print("\n=== Patrick's named species ===")
named = ["Loxodonta_africana", "Elephas_maximus", "Hippopotamus_amphibius",
         "Balaenoptera_acutorostrata", "Heterocephalus_glaber", "Mus_musculus",
         "Rattus_norvegicus", "Homo_sapiens", "Canis_lupus_familiaris",
         "Panthera_leo", "Orcinus_orca"]
hits = j[j["Species"].isin(named)][["Species", "eco_role", "body_kg", "Sample", "Neoplasia", "cancer_prev"]]
print(hits.to_string(index=False))
