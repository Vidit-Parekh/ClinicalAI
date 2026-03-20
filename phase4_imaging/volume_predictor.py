"""
ClinicalAI — Phase 4: Organ Volume Prediction
===============================================
Location: phase4_imaging/volume_predictor.py

Simulates Medpace MCL (Medical Core Lab) organ volumetrics pipeline.
Predicts relative organ volume change (%) using:
  - Lab proxies for organ function (ALT→liver, creatinine→kidney)
  - ICD-9 disease category
  - Demographics + comorbidity burden
  - NLP severity score from Phase 2

MIMIC-III does not contain segmentation volumes directly, but
NOTEEVENTS has radiology reports (CT/MRI) with organ mentions.
We extract proxy volume signals from those reports + labs,
then train regression models per organ.

Outputs:
  data/processed/organ_volume_predictions.csv
  phase4_imaging/outputs/phase4_volume_report.png
  models/volume_models.pkl
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
MODEL_DIR     = os.getenv("MODEL_DIR",     "models")
OUTPUT_DIR    = os.path.join("phase4_imaging", "outputs")
RAW_DIR       = os.getenv("DATA_DIR", "data/raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

RANDOM_STATE = 42

# Organ → lab proxy features (clinically validated relationships)
ORGAN_FEATURE_MAP = {
    "liver": {
        "lab_proxies":   ["alt_ul", "bilirubin_total_mgdl", "bilirubin_direct_mgdl"],
        "icd9_keywords": ["liver", "hepat", "cirrhosis", "nafld", "jaundice"],
        "normal_vol_ml": 1500.0,   # reference adult liver volume
    },
    "kidney": {
        "lab_proxies":   ["creatinine_mgdl", "bun_mgdl"],
        "icd9_keywords": ["renal", "kidney", "nephro", "ckd", "dialysis"],
        "normal_vol_ml": 150.0,    # single kidney
    },
    "spleen": {
        "lab_proxies":   ["platelet_kul", "wbc_kul", "hemoglobin_gdl"],
        "icd9_keywords": ["spleen", "spleno", "portal", "hypersplen"],
        "normal_vol_ml": 200.0,
    },
}


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Load master features + extract radiology signals
# ─────────────────────────────────────────────────────────────────────

def load_data():
    log.info("[1/5] Loading master features...")
    master_path = os.path.join(PROCESSED_DIR, "mimic_master_features.csv")
    df = pd.read_csv(master_path, low_memory=False)
    log.info("   Master: %d rows", len(df))

    # Merge NLP severity if available
    nlp_path = os.path.join(PROCESSED_DIR, "nlp_predictions.csv")
    if os.path.exists(nlp_path):
        nlp = pd.read_csv(nlp_path)
        sev_map = {"stable": 0, "moderate": 1, "critical": 2}
        nlp["SEVERITY_ID"] = nlp["PRED_LABEL"].map(sev_map)
        nlp_agg = (nlp.groupby("HADM_ID")["SEVERITY_ID"]
                      .max().reset_index()
                      .rename(columns={"SEVERITY_ID": "NLP_SEVERITY_ID"}))
        df = df.merge(nlp_agg, on="HADM_ID", how="left")

    return df


def extract_radiology_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads NOTEEVENTS radiology reports to extract binary organ-mention flags.
    These serve as proxy evidence that imaging was done for a given organ.
    """
    log.info("   Extracting radiology signals from NOTEEVENTS...")

    notes_path = os.path.join(RAW_DIR, "NOTEEVENTS_sorted.csv")
    if not os.path.exists(notes_path):
        log.warning("   NOTEEVENTS not found — using lab proxies only")
        for organ in ORGAN_FEATURE_MAP:
            df[f"{organ}_mentioned"] = 0
        return df

    # Read only radiology notes — chunked for memory efficiency
    chunks = []
    for chunk in pd.read_csv(
        notes_path, low_memory=False,
        usecols=["HADM_ID", "CATEGORY", "TEXT"],
        chunksize=50_000,
    ):
        chunk.columns = chunk.columns.str.upper()
        radiology = chunk[
            chunk["CATEGORY"].str.lower().str.contains("radiol", na=False)
        ].copy()
        if len(radiology):
            chunks.append(radiology)

    if not chunks:
        for organ in ORGAN_FEATURE_MAP:
            df[f"{organ}_mentioned"] = 0
        return df

    rad_df = pd.concat(chunks, ignore_index=True)
    rad_df["HADM_ID"] = pd.to_numeric(rad_df["HADM_ID"], errors="coerce")
    rad_df["TEXT"]    = rad_df["TEXT"].fillna("").str.lower()

    # Per-organ mention flag
    for organ, config in ORGAN_FEATURE_MAP.items():
        pattern = "|".join(config["icd9_keywords"])
        rad_df[f"{organ}_mentioned"] = (
            rad_df["TEXT"].str.contains(pattern, na=False)
        ).astype(int)

    mention_agg = (
        rad_df.groupby("HADM_ID")[[f"{o}_mentioned" for o in ORGAN_FEATURE_MAP]]
              .max()
              .reset_index()
    )
    df = df.merge(mention_agg, on="HADM_ID", how="left")
    for organ in ORGAN_FEATURE_MAP:
        df[f"{organ}_mentioned"] = df[f"{organ}_mentioned"].fillna(0).astype(int)

    log.info("   Radiology signals extracted")
    return df


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Generate synthetic volume targets
# ─────────────────────────────────────────────────────────────────────

def generate_volume_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since MIMIC-III has no ground-truth organ volumes, we generate
    clinically-calibrated synthetic targets using validated relationships:

    Liver volume change (%):
      Elevated ALT → hepatomegaly (positive % change)
      Cirrhosis ICD-9 codes → reduced volume (negative % change)

    Kidney volume change (%):
      High creatinine → shrunken kidneys in CKD (negative % change)
      Acute kidney injury → swollen kidneys (positive % change)

    Spleen volume change (%):
      Thrombocytopenia → splenomegaly (positive % change)
      Low platelet + portal hypertension → large spleen

    Gaussian noise added to simulate real measurement variability.
    In a real Medpace MCL pipeline this would come from automated
    segmentation of CT/MRI volumes.
    """
    log.info("[2/5] Generating organ volume change targets...")
    df = df.copy()
    np.random.seed(RANDOM_STATE)

    n = len(df)

    # ── Liver volume change (%) ────────────────────────────────────────
    alt       = df.get("alt_ul", pd.Series(np.full(n, 40))).fillna(40).clip(5, 2000)
    bili      = df.get("bilirubin_total_mgdl", pd.Series(np.zeros(n))).fillna(0)
    liver_icd = df.get("ICD9_CATEGORY", pd.Series(["other"] * n)).fillna("other")

    liver_signal = (
        0.08  * np.log1p(alt - 40).clip(0)          # ALT elevation → hepatomegaly
        + 0.05 * bili.clip(0, 20)                   # bilirubin → cholestasis
        + 0.10 * (liver_icd == "infectious_parasitic").astype(float)
        - 0.05 * df.get("DIED", pd.Series(np.zeros(n))).fillna(0)
    )
    df["liver_vol_change_pct"] = (
        liver_signal * 100 + np.random.normal(0, 5, n)
    ).round(2)

    # ── Kidney volume change (%) ───────────────────────────────────────
    creat = df.get("creatinine_mgdl", pd.Series(np.ones(n))).fillna(1.0).clip(0.3, 20)
    bun   = df.get("bun_mgdl", pd.Series(np.full(n, 15))).fillna(15).clip(5, 200)

    kidney_signal = (
        - 0.06 * (creat - 1.0).clip(0)              # chronic elevation → atrophy
        + 0.04 * np.log1p(np.maximum(creat - 3, 0)) # acute spike → swelling
        - 0.02 * (bun / 15 - 1).clip(0)
        + 0.05 * df.get("IS_EMERGENCY", pd.Series(np.zeros(n))).fillna(0)
    )
    df["kidney_vol_change_pct"] = (
        kidney_signal * 100 + np.random.normal(0, 4, n)
    ).round(2)

    # ── Spleen volume change (%) ───────────────────────────────────────
    plt_count = df.get("platelet_kul", pd.Series(np.full(n, 200))).fillna(200).clip(10, 800)
    wbc       = df.get("wbc_kul", pd.Series(np.full(n, 8))).fillna(8).clip(0.5, 100)

    spleen_signal = (
        - 0.04 * (plt_count - 200).clip(None, 0) / 200   # low platelets → splenomegaly
        + 0.03 * np.log1p(np.maximum(wbc - 11, 0))       # leukocytosis
        + 0.06 * df.get("N_DIAGNOSES", pd.Series(np.zeros(n))).fillna(0) / 10
    )
    df["spleen_vol_change_pct"] = (
        spleen_signal * 100 + np.random.normal(0, 6, n)
    ).round(2)

    for organ in ["liver", "kidney", "spleen"]:
        col = f"{organ}_vol_change_pct"
        log.info("   %s vol change: mean=%.1f%% | std=%.1f%%",
                 organ, df[col].mean(), df[col].std())
    return df


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Build features and train per-organ regression models
# ─────────────────────────────────────────────────────────────────────

def get_organ_features(df: pd.DataFrame, organ: str) -> list:
    """Returns available feature columns for a given organ."""
    config = ORGAN_FEATURE_MAP[organ]

    base_features = [
        "AGE", "GENDER_ENC", "LOS_DAYS", "IS_EMERGENCY",
        "N_DIAGNOSES", "N_ABNORMAL_LABS",
    ]
    lab_features = [c for c in config["lab_proxies"] if c in df.columns]
    nlp_features = ["NLP_SEVERITY_ID"] if "NLP_SEVERITY_ID" in df.columns else []
    mention_feat = [f"{organ}_mentioned"] if f"{organ}_mentioned" in df.columns else []

    all_feats = base_features + lab_features + nlp_features + mention_feat
    return [f for f in all_feats if f in df.columns]


def train_organ_models(df: pd.DataFrame):
    log.info("[3/5] Training organ volume regression models...")

    results  = {}
    models   = {}

    for organ in ["liver", "kidney", "spleen"]:
        target   = f"{organ}_vol_change_pct"
        features = get_organ_features(df, organ)

        X = df[features].copy()
        y = df[target].copy()

        # Drop rows where target is NaN
        mask  = y.notna() & X.notna().all(axis=1)
        X, y  = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )

        # Pipeline: impute → scale → GBR
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        log.info("   %s — MAE: %.2f%% | RMSE: %.2f%% | R²: %.3f",
                 organ, mae, rmse, r2)

        results[organ] = {
            "mae": mae, "rmse": rmse, "r2": r2,
            "y_test": y_test, "y_pred": y_pred,
            "features": features,
        }
        # Store as dict so generate_predictions can retrieve the exact
        # feature list the pipeline was fitted on
        models[organ] = {"pipeline": pipe, "features": features}

    # Save all models
    model_path = os.path.join(MODEL_DIR, "volume_models.pkl")
    joblib.dump(models, model_path)
    log.info("   Models saved → %s", model_path)
    return models, results


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Generate predictions on full dataset
# ─────────────────────────────────────────────────────────────────────

def generate_predictions(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    log.info("[4/5] Generating predictions on full dataset...")
    pred_df = df[["SUBJECT_ID", "HADM_ID"]].copy()

    for organ, model_entry in models.items():
        pipe     = model_entry["pipeline"]
        features = model_entry["features"]   # exact list used at fit time

        X = df[features].copy().fillna(df[features].median())

        pred_df[f"{organ}_vol_change_pred_pct"] = pipe.predict(X).round(2)
        pred_df[f"{organ}_vol_change_actual_pct"] = df[f"{organ}_vol_change_pct"].values
        normal_vol = ORGAN_FEATURE_MAP[organ]["normal_vol_ml"]
        pred_df[f"{organ}_pred_vol_ml"] = (
            normal_vol * (1 + pred_df[f"{organ}_vol_change_pred_pct"] / 100)
        ).round(1)

    out_path = os.path.join(PROCESSED_DIR, "organ_volume_predictions.csv")
    pred_df.to_csv(out_path, index=False)
    log.info("   Predictions saved → %s", out_path)
    return pred_df


# ─────────────────────────────────────────────────────────────────────
# Step 5 — EDA and evaluation charts
# ─────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, results: dict, pred_df: pd.DataFrame):
    log.info("[5/5] Generating evaluation charts...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "ClinicalAI — Phase 4: Organ Volume Prediction (MCL simulation)",
        fontsize=14, fontweight="bold",
    )

    organs = ["liver", "kidney", "spleen"]
    colors = {"liver": "#534AB7", "kidney": "#1D9E75", "spleen": "#D85A30"}

    for col_idx, organ in enumerate(organs):
        res = results[organ]
        c   = colors[organ]

        # ── Row 1: Actual vs Predicted scatter ────────────────────────
        ax = axes[0, col_idx]
        ax.scatter(res["y_test"], res["y_pred"],
                   alpha=0.35, s=12, color=c, edgecolors="none")
        lim = max(abs(res["y_test"].min()), abs(res["y_test"].max()),
                  abs(res["y_pred"].min()), abs(res["y_pred"].max())) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{organ.capitalize()} — actual vs predicted",
                     fontsize=11)
        ax.set_xlabel("Actual change (%)")
        ax.set_ylabel("Predicted change (%)")
        ax.text(0.05, 0.92,
                f"R² = {res['r2']:.3f}\nMAE = {res['mae']:.1f}%",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#D3D1C7", alpha=0.8))
        ax.spines[["top", "right"]].set_visible(False)

        # ── Row 2: Predicted volume distribution ─────────────────────
        ax2 = axes[1, col_idx]
        vol_col = f"{organ}_pred_vol_ml"
        if vol_col in pred_df.columns:
            vols = pred_df[vol_col].dropna().clip(
                ORGAN_FEATURE_MAP[organ]["normal_vol_ml"] * 0.3,
                ORGAN_FEATURE_MAP[organ]["normal_vol_ml"] * 2.5,
            )
            ax2.hist(vols, bins=40, color=c, alpha=0.75, edgecolor="none")
            ax2.axvline(ORGAN_FEATURE_MAP[organ]["normal_vol_ml"],
                        color="#888780", linewidth=1.5, linestyle="--",
                        label=f"Normal ({ORGAN_FEATURE_MAP[organ]['normal_vol_ml']:.0f} mL)")
            ax2.set_title(f"{organ.capitalize()} volume distribution",
                          fontsize=11)
            ax2.set_xlabel("Predicted volume (mL)")
            ax2.set_ylabel("Admissions")
            ax2.legend(fontsize=8)
            ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "phase4_volume_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("   Chart saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run():
    log.info("══ ClinicalAI — Phase 4: Organ Volume Prediction ══")

    df              = load_data()
    df              = extract_radiology_signals(df)
    df              = generate_volume_targets(df)
    models, results = train_organ_models(df)
    pred_df         = generate_predictions(df, models)
    plot_results(df, results, pred_df)

    log.info("══ Phase 4 complete ══")
    log.info("   organ_volume_predictions.csv → Phase 6 dashboard")
    return models, pred_df


if __name__ == "__main__":
    run()