"""
ClinicalAI — Phase 1: MIMIC-III Data Pipeline
==============================================
Location in project:  phase1_data_pipeline/data_pipeline.py

Uses all 5 real MIMIC-III-10k tables from Kaggle:
  PATIENTS.csv       demographics + mortality
  ADMISSIONS.csv     hospital stays + diagnoses
  LABEVENTS.csv      lab results (ITEMID coded)
  NOTEEVENTS.csv     clinical free-text notes
  DIAGNOSES_ICD.csv  ICD-9 diagnosis codes

Outputs (written to data/processed/):
  mimic_master_features.csv  → Phase 3 ML (outcome prediction)
  mimic_notes_nlp.csv        → Phase 2 NLP (BioBERT fine-tuning)

Configuration via .env file (see .env.template):
  DATA_DIR      = path to your MIMIC CSV folder   (data/raw/)
  PROCESSED_DIR = path for pipeline outputs        (data/processed/)

MIMIC ITEMID reference used here:
  51222 = Hemoglobin (g/dL)       51265 = Platelet Count (K/uL)
  51301 = WBC (K/uL)              51221 = Hematocrit (%)
  51248 = MCH (pg)                51250 = MCV (fL)
  50883 = Bilirubin Direct        50885 = Bilirubin Total (mg/dL)
  50912 = Creatinine (mg/dL)      50861 = ALT/SGPT (U/L)
  50931 = Glucose (mg/dL)         51006 = BUN (mg/dL)
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Load .env (falls back to defaults if .env not present) ───────────
load_dotenv()

DATA_DIR      = os.getenv("DATA_DIR",      "data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
EDA_DIR       = os.path.join("phase1_data_pipeline", "outputs")

for d in [PROCESSED_DIR, EDA_DIR]:
    os.makedirs(d, exist_ok=True)

# ── File map: MIMIC-III-10k Kaggle filenames ─────────────────────────
# The Kaggle MIMIC-III-10k dataset uses _sorted suffix on all files.
# Only change these values if your local filenames differ.
FILE_MAP = {
    "patients":   "PATIENTS_sorted.csv",
    "admissions": "ADMISSIONS_sorted.csv",
    "labevents":  "LABEVENTS_sorted.csv",
    "noteevents": "NOTEEVENTS_sorted.csv",
    "diagnoses":  "DIAGNOSES_ICD_sorted.csv",
}

# ── Lab ITEMIDs → friendly column names ──────────────────────────────
LAB_ITEM_MAP = {
    51222: "hemoglobin_gdl",
    51265: "platelet_kul",
    51301: "wbc_kul",
    51221: "hematocrit_pct",
    51248: "mch_pg",
    51250: "mcv_fl",
    50883: "bilirubin_direct_mgdl",
    50885: "bilirubin_total_mgdl",
    50912: "creatinine_mgdl",
    50861: "alt_ul",
    50931: "glucose_mgdl",
    51006: "bun_mgdl",
}

# ── ICD-9 first-digit → disease category ─────────────────────────────
# Source: CDC ICD-9-CM classification
ICD9_CATEGORY_MAP = {
    "0": "infectious_parasitic",   # 001–139
    "1": "neoplasm",               # 140–239
    "2": "endocrine_metabolic",    # 240–279
    "3": "blood_disorders",        # 280–289
    "4": "cardiovascular",         # 390–459  (also catches 400s)
    "5": "respiratory_digestive",  # 460–579
    "6": "genitourinary",          # 580–629
    "7": "musculoskeletal",        # 710–739
    "8": "injury_poisoning",       # 800–999
    "9": "other_conditions",
    "V": "supplementary_factors",  # V codes
    "E": "external_causes",        # E codes
}


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Load all 5 MIMIC tables
# ─────────────────────────────────────────────────────────────────────

def load_tables():
    log.info("══ ClinicalAI — Phase 1: MIMIC-III Pipeline ══")
    log.info("[1/5] Loading MIMIC-III tables from: %s", DATA_DIR)

    def read(key):
        path = os.path.join(DATA_DIR, FILE_MAP[key])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                f"  → Set DATA_DIR in your .env to your MIMIC folder.\n"
                f"  → Expected filename: {FILE_MAP[key]}"
            )
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.upper()
        return df

    patients   = read("patients")
    admissions = read("admissions")
    labs       = read("labevents")
    notes      = read("noteevents")
    diagnoses  = read("diagnoses")

    # Parse dates — errors="coerce" turns unparseable values to NaT safely
    for df, cols in [
        (patients,   ["DOB", "DOD", "DOD_HOSP"]),
        (admissions, ["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"]),
        (labs,       ["CHARTTIME"]),
        (notes,      ["CHARTTIME", "CHARTDATE"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure HADM_ID is numeric across all tables
    for df, col in [(admissions, "HADM_ID"), (labs, "HADM_ID"),
                    (notes, "HADM_ID"), (diagnoses, "HADM_ID")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("   PATIENTS      : %6d rows | expired: %d (%.0f%%)",
             len(patients), patients["EXPIRE_FLAG"].sum(), patients["EXPIRE_FLAG"].mean()*100)
    log.info("   ADMISSIONS    : %6d rows | types: %d | in-hosp deaths: %d",
             len(admissions), admissions["ADMISSION_TYPE"].nunique(),
             admissions["HOSPITAL_EXPIRE_FLAG"].sum())
    log.info("   LABEVENTS     : %6d rows | ITEMIDs: %d | abnormal flags: %d",
             len(labs), labs["ITEMID"].nunique(),
             (labs["FLAG"].str.lower() == "abnormal").sum())
    log.info("   NOTEEVENTS    : %6d rows | categories: %d",
             len(notes), notes["CATEGORY"].nunique())
    log.info("   DIAGNOSES_ICD : %6d rows | unique ICD9: %d",
             len(diagnoses), diagnoses["ICD9_CODE"].nunique())

    return patients, admissions, labs, notes, diagnoses


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Build master patient table (PATIENTS + ADMISSIONS join)
# ─────────────────────────────────────────────────────────────────────

def build_patient_table(patients, admissions):
    log.info("[2/5] Building master patient table...")

    adm = admissions.copy()
    pat = patients[["SUBJECT_ID", "GENDER", "DOB", "EXPIRE_FLAG"]].copy()

    df = adm.merge(pat, on="SUBJECT_ID", how="left")

    # Age at admission
    # MIMIC de-identification shifts DOB for patients aged >89 to a date
    # far in the past (e.g. 1800-xx-xx) causing int64 overflow on subtraction.
    # Fix: compute age using year difference on Python datetime objects,
    # which avoids the pandas int64 overflow entirely.
    def safe_age(admittime, dob):
        try:
            if pd.isnull(admittime) or pd.isnull(dob):
                return np.nan
            age = (admittime.year - dob.year
                   - ((admittime.month, admittime.day) < (dob.month, dob.day)))
            return float(age)
        except Exception:
            return np.nan

    df["AGE"] = df.apply(
        lambda r: safe_age(r["ADMITTIME"], r["DOB"]), axis=1
    )
    # MIMIC masks true age for patients >89 — clip to realistic range
    df["AGE"] = df["AGE"].clip(lower=0, upper=90)

    # Length of stay in days
    df["LOS_DAYS"] = (
        (df["DISCHTIME"] - df["ADMITTIME"]).dt.total_seconds() / 86400
    ).round(2)

    # ED wait time in hours (NaN for non-ED admissions — expected)
    df["ED_WAIT_HRS"] = (
        (df["ADMITTIME"] - df["EDREGTIME"]).dt.total_seconds() / 3600
    ).round(2)

    # Binary outcome: in-hospital death OR discharge to DEAD/EXPIRED
    df["DIED"] = (
        (df["HOSPITAL_EXPIRE_FLAG"] == 1) |
        (df["DISCHARGE_LOCATION"].str.upper().str.strip() == "DEAD/EXPIRED")
    ).astype(int)

    # Emergency admission flag
    df["IS_EMERGENCY"] = (
        df["ADMISSION_TYPE"].str.upper().str.strip() == "EMERGENCY"
    ).astype(int)

    # Gender binary encode (M=1, F=0) for ML
    df["GENDER_ENC"] = (df["GENDER"].str.upper().str.strip() == "M").astype(int)

    # Insurance label encode (proxy for socioeconomic status)
    le = LabelEncoder()
    df["INSURANCE_ENC"] = le.fit_transform(df["INSURANCE"].fillna("Unknown"))

    keep = [
        "SUBJECT_ID", "HADM_ID", "ADMITTIME",
        "AGE", "GENDER", "GENDER_ENC",
        "ADMISSION_TYPE", "IS_EMERGENCY",
        "INSURANCE", "INSURANCE_ENC",
        "DIAGNOSIS", "LOS_DAYS", "ED_WAIT_HRS",
        "HOSPITAL_EXPIRE_FLAG", "DIED",
        "DISCHARGE_LOCATION",
    ]
    df = df[keep].copy()

    log.info("   Master table   : %d rows × %d cols", *df.shape)
    log.info("   Avg age        : %.1f yrs", df["AGE"].dropna().mean())
    log.info("   Avg LOS        : %.1f days", df["LOS_DAYS"].dropna().mean())
    log.info("   Mortality rate : %.1f%%", df["DIED"].mean() * 100)
    log.info("   Emergency adm. : %.1f%%", df["IS_EMERGENCY"].mean() * 100)
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Process ICD-9 diagnoses → feature flags
# ─────────────────────────────────────────────────────────────────────

def process_diagnoses(diagnoses, master_df):
    log.info("[3/5] Processing ICD-9 diagnosis codes...")

    diag = diagnoses.copy()

    # Primary diagnosis: SEQ_NUM == 1
    primary = (
        diag[diag["SEQ_NUM"] == 1][["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]]
        .copy()
        .rename(columns={"ICD9_CODE": "PRIMARY_ICD9"})
    )

    # Comorbidity burden: total diagnoses per admission
    dx_count = (
        diag.groupby(["SUBJECT_ID", "HADM_ID"])
            .size()
            .reset_index(name="N_DIAGNOSES")
    )

    # ICD-9 category from first character of code
    def icd9_category(code):
        if pd.isna(code):
            return "unknown"
        c = str(code).strip()[0].upper()
        return ICD9_CATEGORY_MAP.get(c, "other")

    primary["ICD9_CATEGORY"] = primary["PRIMARY_ICD9"].apply(icd9_category)

    # Merge into master
    df = master_df.merge(
        primary[["HADM_ID", "PRIMARY_ICD9", "ICD9_CATEGORY"]],
        on="HADM_ID", how="left"
    )
    df = df.merge(dx_count[["HADM_ID", "N_DIAGNOSES"]], on="HADM_ID", how="left")
    df["N_DIAGNOSES"] = df["N_DIAGNOSES"].fillna(0).astype(int)

    log.info("   Primary ICD9 matched    : %d admissions", primary["HADM_ID"].nunique())
    log.info("   Avg diagnoses/admission : %.1f", df["N_DIAGNOSES"].mean())
    log.info("   ICD9 categories found   : %s",
             df["ICD9_CATEGORY"].value_counts().to_dict())
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Process lab events → wide feature table
# ─────────────────────────────────────────────────────────────────────

def process_labs(labs, master_df):
    print("\n[4/5] Processing lab events...")

    df = labs.copy()
    df = df[df["ITEMID"].isin(LAB_ITEM_MAP.keys())].copy()
    df["LAB_NAME"]   = df["ITEMID"].map(LAB_ITEM_MAP)
    df["VALUENUM"]   = pd.to_numeric(df["VALUENUM"], errors="coerce")
    df["IS_ABNORMAL"]= (df["FLAG"].str.lower() == "abnormal").astype(int)

    # Wide pivot — mean per admission
    pivot = df.pivot_table(
        index=["SUBJECT_ID", "HADM_ID"],
        columns="LAB_NAME",
        values="VALUENUM",
        aggfunc="mean"
    ).reset_index()
    pivot.columns.name = None

    # Abnormal lab count per admission
    abn = (df[df["IS_ABNORMAL"] == 1]
           .groupby(["SUBJECT_ID", "HADM_ID"])
           .size()
           .reset_index(name="N_ABNORMAL_LABS"))
    pivot = pivot.merge(abn, on=["SUBJECT_ID", "HADM_ID"], how="left")
def process_labs(labs, master_df):
    log.info("[4/5] Processing lab events...")

    df = labs.copy()
    df = df[df["ITEMID"].isin(LAB_ITEM_MAP.keys())].copy()
    df["LAB_NAME"]    = df["ITEMID"].map(LAB_ITEM_MAP)
    df["VALUENUM"]    = pd.to_numeric(df["VALUENUM"], errors="coerce")
    df["IS_ABNORMAL"] = (df["FLAG"].str.lower() == "abnormal").astype(int)

    # Pivot: long → wide (one row per patient-admission, one col per lab)
    # Using mean — for Phase 3 you may want first/last value instead
    pivot = df.pivot_table(
        index=["SUBJECT_ID", "HADM_ID"],
        columns="LAB_NAME",
        values="VALUENUM",
        aggfunc="mean"
    ).reset_index()
    pivot.columns.name = None

    # Count abnormal lab results per admission
    abn = (
        df[df["IS_ABNORMAL"] == 1]
        .groupby(["SUBJECT_ID", "HADM_ID"])
        .size()
        .reset_index(name="N_ABNORMAL_LABS")
    )
    pivot = pivot.merge(abn, on=["SUBJECT_ID", "HADM_ID"], how="left")
    pivot["N_ABNORMAL_LABS"] = pivot["N_ABNORMAL_LABS"].fillna(0).astype(int)

    # Impute missing lab values with median
    lab_cols = [c for c in pivot.columns
                if c not in ["SUBJECT_ID", "HADM_ID", "N_ABNORMAL_LABS"]]
    if lab_cols:
        imputer = SimpleImputer(strategy="median")
        pivot[lab_cols] = imputer.fit_transform(pivot[lab_cols])

    # Left join — patients without labs keep NaN (not dropped)
    merged = master_df.merge(pivot, on=["SUBJECT_ID", "HADM_ID"], how="left")

    log.info("   Lab features extracted  : %s", lab_cols)
    log.info("   Admissions with labs    : %d", pivot["HADM_ID"].nunique())
    log.info("   Total abnormal flags    : %d", pivot["N_ABNORMAL_LABS"].sum())
    return merged, lab_cols


# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Process notes → NLP-ready file
# ─────────────────────────────────────────────────────────────────────

def process_notes(notes):
    log.info("[5/5] Processing clinical notes...")

    df = notes.copy()

    # Drop notes flagged as errors by clinical staff
    before = len(df)
    df = df[df["ISERROR"].isna() | (df["ISERROR"] == 0)].copy()
    log.info("   Dropped %d error-flagged notes", before - len(df))

    # Strip MIMIC de-identification artifacts: [** ... **]
    # These are PHI placeholders inserted during anonymization
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r'\[\*\*.*?\*\*\]', '[REDACTED]', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df["TEXT_CLEAN"] = df["TEXT"].apply(clean_text)
    df["TEXT_LEN"]   = df["TEXT_CLEAN"].str.len()

    # Drop trivially short notes
    df = df[df["TEXT_LEN"] > 50].copy()

    # ── Keyword-based severity label ───────────────────────────────────
    # This is used as the training label for Phase 2 BioBERT fine-tuning.
    # Critical > moderate > stable (order matters — check critical first).
    critical_kw = [
        "arrest", "sepsis", "septic", "critical", "emergent",
        "deteriorat", "shock", "dnr", "intubat", "vfib", "v-fib",
        "cardiac arrest", "dead", "expired", "unresponsive",
        "code blue", "pressors", "vasopressor",
    ]
    moderate_kw = [
        "effusion", "infiltrate", "abnormal", "elevated", "concern",
        "monitor", "hypotension", "failure", "edema", "unstable",
        "tachycardia", "bradycardia", "fever", "infection",
    ]

    def label_severity(text):
        t = text.lower()
        if any(k in t for k in critical_kw):
            return "critical"
        elif any(k in t for k in moderate_kw):
            return "moderate"
        return "stable"

    df["SEVERITY_LABEL"] = df["TEXT_CLEAN"].apply(label_severity)

    # ── Consolidate MIMIC note categories ─────────────────────────────
    def group_category(cat):
        cat = str(cat).strip().lower()
        if "nurs"      in cat: return "Nursing"
        if "radiol"    in cat: return "Radiology"
        if "physician" in cat: return "Physician"
        if "discharge" in cat: return "Discharge"
        if "ecg"       in cat or "ekg" in cat: return "ECG"
        if "echo"      in cat: return "Echo"
        if "consult"   in cat: return "Consult"
        return "Other"

    df["NOTE_TYPE"] = df["CATEGORY"].apply(group_category)

    log.info("   Clean notes     : %d", len(df))
    log.info("   Note types      : %s", df["NOTE_TYPE"].value_counts().to_dict())
    log.info("   Severity dist   : %s", df["SEVERITY_LABEL"].value_counts().to_dict())
    log.info("   Avg note length : %.0f chars", df["TEXT_LEN"].mean())

    # Save NLP-ready file for Phase 2
    nlp_path = os.path.join(PROCESSED_DIR, "mimic_notes_nlp.csv")
    nlp_out  = df[[
        "ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE",
        "NOTE_TYPE", "DESCRIPTION", "TEXT_CLEAN",
        "SEVERITY_LABEL", "TEXT_LEN"
    ]].copy()
    nlp_out.to_csv(nlp_path, index=False)
    log.info("   Saved → %s", nlp_path)
    return df


# ─────────────────────────────────────────────────────────────────────
# EDA — 4-panel dashboard on real MIMIC data
# ─────────────────────────────────────────────────────────────────────

def run_eda(master_df, notes_df):
    log.info("Running EDA...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ClinicalAI — Phase 1 EDA  (MIMIC-III)",
                 fontsize=15, fontweight="bold")

    # ── Plot 1: Admission type vs mortality ──────────────────────────
    ax1 = axes[0, 0]
    mort = (master_df.groupby("ADMISSION_TYPE")["DIED"]
                     .agg(["mean", "count"])
                     .reset_index()
                     .sort_values("mean", ascending=True))
    colors = ["#9FE1CB" if m < 0.1 else "#FAC775" if m < 0.2 else "#F09595"
              for m in mort["mean"]]
    bars = ax1.barh(mort["ADMISSION_TYPE"], mort["mean"],
                    color=colors, edgecolor="none")
    for bar, (_, row) in zip(bars, mort.iterrows()):
        ax1.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f'{row["mean"]:.0%}  (n={int(row["count"])})',
                 va="center", fontsize=9)
    ax1.set_title("In-hospital mortality by admission type", fontsize=12)
    ax1.set_xlabel("Mortality rate")
    ax1.set_xlim(0, 0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Plot 2: Length of stay distribution ──────────────────────────
    ax2 = axes[0, 1]
    los = master_df["LOS_DAYS"].dropna().clip(0, 60)
    ax2.hist(los, bins=20, color="#534AB7", edgecolor="white", linewidth=0.5)
    ax2.axvline(los.mean(),   color="#D85A30", linewidth=1.5,
                linestyle="--", label=f"Mean: {los.mean():.1f}d")
    ax2.axvline(los.median(), color="#1D9E75", linewidth=1.5,
                linestyle="--", label=f"Median: {los.median():.1f}d")
    ax2.set_title("Length of stay distribution", fontsize=12)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Admissions")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Plot 3: Note severity breakdown ──────────────────────────────
    ax3 = axes[1, 0]
    sev_colors  = {"critical": "#E24B4A", "moderate": "#BA7517", "stable": "#1D9E75"}
    sev_counts  = notes_df["SEVERITY_LABEL"].value_counts()
    wedge_colors = [sev_colors.get(l, "#888780") for l in sev_counts.index]
    _, _, autotexts = ax3.pie(
        sev_counts.values,
        labels=sev_counts.index,
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax3.set_title("Note severity labels (keyword-based)", fontsize=12)

    # ── Plot 4: ICD-9 category distribution ──────────────────────────
    ax4 = axes[1, 1]
    if "ICD9_CATEGORY" in master_df.columns:
        icd_counts = master_df["ICD9_CATEGORY"].value_counts().head(8)
        ax4.bar(icd_counts.index, icd_counts.values,
                color="#1D9E75", edgecolor="none")
        ax4.set_title("Primary ICD-9 categories", fontsize=12)
        ax4.set_ylabel("Admissions")
        ax4.tick_params(axis="x", rotation=35, labelsize=9)
        ax4.spines[["top", "right"]].set_visible(False)
    else:
        ax4.axis("off")

    plt.tight_layout()
    eda_path = os.path.join(EDA_DIR, "phase1_eda_mimic.png")
    plt.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("   EDA chart saved → %s", eda_path)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load all 5 MIMIC tables
    patients, admissions, labs, notes, diagnoses = load_tables()

    # 2. Build master patient table (demographics + admission features)
    master = build_patient_table(patients, admissions)

    # 3. Enrich with ICD-9 diagnosis features
    master = process_diagnoses(diagnoses, master)

    # 4. Enrich with lab features (wide pivot)
    master, lab_cols = process_labs(labs, master)

    # 5. Process notes → NLP-ready CSV
    notes_clean = process_notes(notes)

    # 6. EDA charts
    run_eda(master, notes_clean)

    # 7. Save master feature table for Phase 3 ML
    master_path = os.path.join(PROCESSED_DIR, "mimic_master_features.csv")
    master.to_csv(master_path, index=False)
    log.info("   Saved → %s", master_path)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n── Output files ──────────────────────────────────────────────")
    log.info("  %s  → Phase 3 ML", master_path)
    log.info("  %s  → Phase 2 NLP",
             os.path.join(PROCESSED_DIR, "mimic_notes_nlp.csv"))
    log.info("  %s  → EDA chart",
             os.path.join(EDA_DIR, "phase1_eda_mimic.png"))

    log.info("\n── Master feature table preview ──────────────────────────────")
    preview_cols = ["SUBJECT_ID", "HADM_ID", "AGE", "GENDER",
                    "ADMISSION_TYPE", "LOS_DAYS", "N_DIAGNOSES",
                    "ICD9_CATEGORY", "DIED"]
    available = [c for c in preview_cols if c in master.columns]
    print(master[available].head(8).to_string(index=False))

    log.info("\n── Sample cleaned notes ──────────────────────────────────────")
    for _, row in notes_clean[["SUBJECT_ID", "NOTE_TYPE",
                                "SEVERITY_LABEL", "TEXT_CLEAN"]].head(3).iterrows():
        preview = row["TEXT_CLEAN"][:130].replace("\n", " ")
        log.info("  [PT %s | %s | %s]", row.SUBJECT_ID,
                 row.NOTE_TYPE, row.SEVERITY_LABEL)
        log.info("  %s...", preview)

    log.info("══ Phase 1 complete ══════════════════════════════════════════")