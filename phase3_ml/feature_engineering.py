"""
ClinicalAI — Phase 3: Feature Engineering
==========================================
Location: phase3_ml/feature_engineering.py

Reads:
  data/processed/mimic_master_features.csv  (Phase 1 — tabular)
  data/processed/nlp_predictions.csv        (Phase 2 — NLP severity scores)

Writes:
  data/processed/ml_features_train.csv
  data/processed/ml_features_val.csv
  data/processed/ml_features_test.csv

Target variable: DIED (in-hospital mortality)
  — binary outcome: 1 = died, 0 = survived

Key steps:
  1. Load & merge tabular + NLP features
  2. Select and engineer features
  3. Impute missing values
  4. Encode categoricals
  5. Stratified 80/10/10 split
  6. SMOTE oversampling on train only
  7. StandardScaler (fit on train, apply to val/test)
  8. Save splits
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import joblib
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
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET       = "DIED"

# ── Categorical columns to label-encode ──────────────────────────────
CAT_COLS = ["GENDER", "ADMISSION_TYPE", "INSURANCE", "ICD9_CATEGORY"]

# ── Numeric columns to use as features ───────────────────────────────
# Excludes leakage columns (DISCHARGE_LOCATION, HOSPITAL_EXPIRE_FLAG)
# and ID/timestamp columns
NUM_COLS = [
    "AGE", "LOS_DAYS", "ED_WAIT_HRS", "IS_EMERGENCY",
    "GENDER_ENC", "INSURANCE_ENC", "N_DIAGNOSES", "N_ABNORMAL_LABS",
    "hemoglobin_gdl", "platelet_kul", "wbc_kul", "hematocrit_pct",
    "mch_pg", "mcv_fl", "bilirubin_direct_mgdl", "bilirubin_total_mgdl",
    "creatinine_mgdl", "alt_ul", "glucose_mgdl", "bun_mgdl",
]

# ── NLP feature columns (merged from Phase 2) ─────────────────────────
NLP_COLS = ["NLP_SEVERITY_ID", "NLP_SEVERITY_STABLE", "NLP_SEVERITY_MODERATE",
            "NLP_SEVERITY_CRITICAL"]


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Load and merge
# ─────────────────────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    log.info("[1/7] Loading and merging features...")

    master_path = os.path.join(PROCESSED_DIR, "mimic_master_features.csv")
    nlp_path    = os.path.join(PROCESSED_DIR, "nlp_predictions.csv")

    master = pd.read_csv(master_path, low_memory=False)
    log.info("   Master features : %d rows × %d cols", *master.shape)

    # Merge NLP severity predictions if available
    if os.path.exists(nlp_path):
        nlp = pd.read_csv(nlp_path)

        # Aggregate per admission: take the most severe note label
        # (worst-case severity per admission is the clinically relevant signal)
        sev_map = {"stable": 0, "moderate": 1, "critical": 2}
        nlp["SEVERITY_ID"] = nlp["PRED_LABEL"].map(sev_map)

        nlp_agg = (
            nlp.groupby("HADM_ID")
               .agg(
                   NLP_SEVERITY_ID      = ("SEVERITY_ID",  "max"),
                   NLP_SEVERITY_STABLE  = ("PRED_LABEL",   lambda x: (x == "stable").mean()),
                   NLP_SEVERITY_MODERATE= ("PRED_LABEL",   lambda x: (x == "moderate").mean()),
                   NLP_SEVERITY_CRITICAL= ("PRED_LABEL",   lambda x: (x == "critical").mean()),
               )
               .reset_index()
        )
        master = master.merge(nlp_agg, on="HADM_ID", how="left")
        log.info("   NLP features merged — %d admissions matched",
                 nlp_agg["HADM_ID"].nunique())
    else:
        log.warning("   nlp_predictions.csv not found — skipping NLP features")
        for col in NLP_COLS:
            master[col] = np.nan

    log.info("   Merged shape: %d rows × %d cols", *master.shape)
    log.info("   Mortality rate: %.1f%%", master[TARGET].mean() * 100)
    return master


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Feature selection & engineering
# ─────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("[2/7] Engineering features...")
    df = df.copy()

    # ── Domain-driven engineered features ─────────────────────────────
    # These mimic what clinical scoring systems (APACHE, SOFA) use

    # Liver dysfunction proxy: ALT/AST ratio and bilirubin
    if "alt_ul" in df.columns and "bilirubin_total_mgdl" in df.columns:
        df["liver_score"] = (
            (df["alt_ul"].clip(0, 2000) / 40).clip(0, 10) +
            (df["bilirubin_total_mgdl"].clip(0, 30) / 1.2).clip(0, 10)
        ) / 2

    # Renal dysfunction proxy: creatinine + BUN
    if "creatinine_mgdl" in df.columns and "bun_mgdl" in df.columns:
        df["renal_score"] = (
            (df["creatinine_mgdl"].clip(0, 15) / 1.0).clip(0, 10) +
            (df["bun_mgdl"].clip(0, 150) / 20).clip(0, 10)
        ) / 2

    # Hematologic stress: low hemoglobin + high WBC
    if "hemoglobin_gdl" in df.columns and "wbc_kul" in df.columns:
        df["hematologic_score"] = (
            ((14 - df["hemoglobin_gdl"].clip(4, 18)) / 14).clip(0, 1) +
            (df["wbc_kul"].clip(0, 50) / 11).clip(0, 5)
        ) / 2

    # Emergency + high comorbidity burden = high risk flag
    df["high_risk_flag"] = (
        (df["IS_EMERGENCY"] == 1) & (df["N_DIAGNOSES"] >= 5)
    ).astype(int)

    # Age risk groups (clinical standard: <18, 18-64, 65-79, 80+)
    df["age_group"] = pd.cut(
        df["AGE"].clip(0, 90),
        bins=[0, 18, 65, 80, 91],
        labels=[0, 1, 2, 3],
        right=False,
    ).astype(float)

    log.info("   Engineered features: liver_score, renal_score, "
             "hematologic_score, high_risk_flag, age_group")
    return df


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Select final feature set
# ─────────────────────────────────────────────────────────────────────

def select_features(df: pd.DataFrame):
    log.info("[3/7] Selecting feature set...")

    engineered = ["liver_score", "renal_score", "hematologic_score",
                  "high_risk_flag", "age_group"]

    # Build final feature list from what actually exists in df
    all_candidates = NUM_COLS + NLP_COLS + engineered
    feature_cols   = [c for c in all_candidates if c in df.columns]

    # Remove target and obvious leakage columns
    leakage = [TARGET, "HOSPITAL_EXPIRE_FLAG", "DISCHARGE_LOCATION",
               "SUBJECT_ID", "HADM_ID", "ADMITTIME"]
    feature_cols = [c for c in feature_cols if c not in leakage]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    log.info("   Final features : %d columns", len(feature_cols))
    log.info("   Target balance : died=%d (%.1f%%) | survived=%d (%.1f%%)",
             y.sum(), y.mean()*100, (1-y).sum(), (1-y.mean())*100)
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Impute missing values
# ─────────────────────────────────────────────────────────────────────

def impute(X_train, X_val, X_test):
    log.info("[4/7] Imputing missing values (median strategy)...")
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=X_train.columns)
    X_val   = pd.DataFrame(imputer.transform(X_val),
                           columns=X_val.columns)
    X_test  = pd.DataFrame(imputer.transform(X_test),
                           columns=X_test.columns)

    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
    log.info("   Imputer saved → models/imputer.pkl")
    return X_train, X_val, X_test, imputer


# ─────────────────────────────────────────────────────────────────────
# Step 5 — Stratified split
# ─────────────────────────────────────────────────────────────────────

def split(X, y):
    log.info("[5/7] Stratified 80 / 10 / 10 split...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    log.info("   Train : %d  (died: %.1f%%)", len(y_train), y_train.mean()*100)
    log.info("   Val   : %d  (died: %.1f%%)", len(y_val),   y_val.mean()*100)
    log.info("   Test  : %d  (died: %.1f%%)", len(y_test),  y_test.mean()*100)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────
# Step 6 — SMOTE oversampling (train only)
# ─────────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train):
    log.info("[6/7] Applying SMOTE to training set...")
    before = y_train.value_counts().to_dict()
    smote  = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after  = pd.Series(y_res).value_counts().to_dict()
    log.info("   Before SMOTE: %s", before)
    log.info("   After  SMOTE: %s", after)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


# ─────────────────────────────────────────────────────────────────────
# Step 7 — Scale and save
# ─────────────────────────────────────────────────────────────────────

def scale_and_save(X_train, X_val, X_test, y_train, y_val, y_test):
    log.info("[7/7] Scaling and saving splits...")

    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                           columns=X_train.columns)
    X_val   = pd.DataFrame(scaler.transform(X_val),
                           columns=X_val.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # Save splits
    for split_name, X_s, y_s in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        X_s[TARGET] = y_s.values
        out = os.path.join(PROCESSED_DIR, f"ml_features_{split_name}.csv")
        X_s.to_csv(out, index=False)
        log.info("   Saved → %s", out)

    log.info("   Scaler saved → models/scaler.pkl")
    return scaler


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run_feature_engineering():
    log.info("══ ClinicalAI — Phase 3: Feature Engineering ══")

    df                      = load_and_merge()
    df                      = engineer_features(df)
    X, y, feature_cols      = select_features(df)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    X_train, X_val, X_test, _ = impute(X_train, X_val, X_test)
    X_train, y_train          = apply_smote(X_train, y_train)
    scale_and_save(X_train, X_val, X_test, y_train, y_val, y_test)

    log.info("══ Feature engineering complete ══")
    return feature_cols


if __name__ == "__main__":
    feature_cols = run_feature_engineering()
    log.info("Features ready. Run outcome_model.py to train XGBoost.")