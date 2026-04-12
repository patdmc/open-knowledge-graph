"""Cross-mammalian cancer regression with substrate-hardening proxy.

Tests whether mass-specific metabolic rate (and other nervousness/substrate
proxies) explains cancer prevalence above and beyond body mass and lifespan.

Boddy 2020's own PGLS found body_mass and lifespan p > 0.5 — null at the
two-axis level. Framework predicts the missing axis is substrate hardening.

Inputs:
  - Boddy 2020 supplementary_tables_V2.xlsx, sheet S2_LHtable (37 species)
  - AnAge anage_data.txt (metabolic rate, body temp by species)
  - PanTHERIA pantheria_raw.txt (social group size, trophic level by species)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent / "data" / "cross_mammal"


# ---------- Boddy 2020: cancer prevalence + body mass + lifespan ----------
boddy = pd.read_excel(DATA / "supplementary_tables_V2.xlsx", sheet_name="S2_LHtable")
boddy = boddy[
    [
        "common_name",
        "species_name",
        "order",
        "total_necropsies",
        "any_neoplasia",
        "any_malignant",
        "percent_any_neoplasia",
        "percent_any_malignant",
        "adult_mass_kg",
        "max_lifespan_yr",
    ]
].copy()
boddy["species_norm"] = boddy["species_name"].str.replace("_", " ")
print(f"Boddy 2020: {len(boddy)} species")


# ---------- AnAge: metabolic rate, body temperature ----------
anage = pd.read_csv(DATA / "anage_data.txt", sep="\t")
anage["species_norm"] = anage["Genus"] + " " + anage["Species"]
anage_keep = anage[
    [
        "species_norm",
        "Metabolic rate (W)",
        "Body mass (g)",
        "Temperature (K)",
        "Maximum longevity (yrs)",
    ]
].rename(
    columns={
        "Metabolic rate (W)": "MR_W",
        "Body mass (g)": "anage_mass_g",
        "Temperature (K)": "Tb_K",
        "Maximum longevity (yrs)": "anage_lifespan_yr",
    }
)


# ---------- PanTHERIA: social group size, trophic level ----------
pan = pd.read_csv(
    DATA / "pantheria_raw.txt", sep="\t", na_values=["-999.00", "-999"]
)
pan = pan.rename(
    columns={
        "MSW05_Binomial": "species_norm",
        "10-2_SocialGrpSize": "social_grp_size",
        "6-2_TrophicLevel": "trophic_level",
        "1-1_ActivityCycle": "activity_cycle",
        "5-1_AdultBodyMass_g": "pan_mass_g",
    }
)[["species_norm", "social_grp_size", "trophic_level", "activity_cycle", "pan_mass_g"]]


# ---------- Join ----------
df = boddy.merge(anage_keep, on="species_norm", how="left").merge(pan, on="species_norm", how="left")

# Compute mass-specific metabolic rate (W per gram of body)
df["MR_per_g"] = df["MR_W"] / df["anage_mass_g"]

# Coverage report
print("\n=== Join coverage ===")
for col in ["MR_W", "Tb_K", "social_grp_size", "trophic_level", "activity_cycle"]:
    n = df[col].notna().sum()
    print(f"  {col}: {n}/{len(df)} species")

print("\n=== Species with no AnAge metabolic rate ===")
print(df[df["MR_W"].isna()]["species_name"].tolist())


# ---------- Regression: weighted OLS ----------
def fit(formula, label):
    """Run a weighted OLS, return summary."""
    # Map log_X -> X for dropna check (the log columns get computed in build_design)
    base_cols = [c[4:] if c.startswith("log_") else c for c in formula]
    sub = df.dropna(subset=base_cols + [dv])
    if len(sub) < 5:
        print(f"\n{label}: SKIPPED (n={len(sub)} after dropna)")
        return None
    y = sub[dv]
    X = build_design(sub, formula)
    w = sub["total_necropsies"]
    model = sm.WLS(y, X, weights=w).fit()
    print(f"\n=== {label}  (n={len(sub)}) ===")
    print(model.summary().tables[1])
    print(f"  R²={model.rsquared:.3f}  adj R²={model.rsquared_adj:.3f}  AIC={model.aic:.1f}")
    return model


def build_design(sub, cols):
    X = pd.DataFrame({"intercept": 1.0}, index=sub.index)
    for c in cols:
        if c.startswith("log_"):
            base = c[4:]
            X[c] = np.log(sub[base].astype(float))
        else:
            X[c] = sub[c].astype(float)
    return X


for dv_label, dv in [("ANY NEOPLASIA", "percent_any_neoplasia"), ("MALIGNANT", "percent_any_malignant")]:
    print(f"\n\n############ DV = {dv_label} ############")

    # Replicate Boddy: body mass alone
    fit(["log_adult_mass_kg"], "Model 1: body mass alone")

    # Replicate Boddy: lifespan alone
    fit(["log_max_lifespan_yr"], "Model 2: lifespan alone")

    # Replicate Boddy: body mass + lifespan
    fit(["log_adult_mass_kg", "log_max_lifespan_yr"], "Model 3: body mass + lifespan (Boddy's null)")

    # NEW: + mass-specific metabolic rate
    fit(
        ["log_adult_mass_kg", "log_max_lifespan_yr", "log_MR_per_g"],
        "Model 4: + log mass-specific metabolic rate",
    )

    # NEW: + raw log MR_W (lets body mass absorb the body component separately)
    fit(
        ["log_adult_mass_kg", "log_max_lifespan_yr", "log_MR_W"],
        "Model 4b: + log raw MR_W (decoupled from body mass)",
    )

    # NEW: + body temperature
    fit(
        ["log_adult_mass_kg", "log_max_lifespan_yr", "Tb_K"],
        "Model 5: + body temperature",
    )

    # NEW: + social group size
    fit(
        ["log_adult_mass_kg", "log_max_lifespan_yr", "log_social_grp_size"],
        "Model 6: + log social group size",
    )

    # NEW: full nervousness model
    fit(
        ["log_adult_mass_kg", "log_max_lifespan_yr", "log_MR_per_g", "Tb_K"],
        "Model 7: + MR_per_g + Tb",
    )

    # NEW: full model with raw MR_W
    fit(
        ["log_adult_mass_kg", "log_max_lifespan_yr", "log_MR_W", "Tb_K"],
        "Model 7b: + raw log MR_W + Tb",
    )
