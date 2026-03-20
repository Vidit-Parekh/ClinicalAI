"""
ClinicalAI — Phase 6: Shared Data Loader
=========================================
Location: phase6_dashboard/components/data_loader.py

Centralised cached data loading for all dashboard pages.
Using @st.cache_data so each CSV is read once per session.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Resolve project root from this file's location — works regardless of
# which directory you run `streamlit run` from.
# Structure: .../ClinicalAI/phase6_dashboard/components/data_loader.py
#   _SCRIPT_DIR   → .../ClinicalAI/phase6_dashboard/components/
#   _PROJECT_ROOT → .../ClinicalAI/
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))


def _resolve(env_key: str, default: str) -> str:
    """
    Resolve a directory path from env or default.
    If the env value is relative (e.g. ../data/processed or data/processed),
    resolve it against the project root so it always points to the right place
    regardless of where `streamlit run` is called from.
    """
    raw = os.getenv(env_key, default)
    if os.path.isabs(raw):
        return raw                                  # already absolute — use as-is
    # Strip leading ../ — env values are written relative to project root
    clean = raw.lstrip("./").lstrip("../")
    return os.path.join(_PROJECT_ROOT, clean)


PROCESSED_DIR = _resolve("PROCESSED_DIR", "data/processed")
MODEL_DIR     = _resolve("MODEL_DIR",     "models")


@st.cache_data
def load_master() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "mimic_master_features.csv")
    df   = pd.read_csv(path, low_memory=False)
    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
    return df


@st.cache_data
def load_ml_predictions() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "ml_predictions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_nlp_predictions() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "nlp_predictions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_organ_volumes() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "organ_volume_predictions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_clinical_reports() -> pd.DataFrame:
    candidates = [
        os.path.join(_PROJECT_ROOT, "phase5_llm", "phase5_llm", "outputs", "clinical_reports.csv"),
        os.path.join(_PROJECT_ROOT, "phase5_llm", "outputs", "clinical_reports.csv"),
        os.path.join(_PROJECT_ROOT, "outputs", "clinical_reports.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, "shap_importance.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def risk_label(prob: float) -> str:
    if pd.isna(prob):
        return "Unknown"
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Moderate"
    return "Low"


RISK_COLORS = {
    "High":     "#E24B4A",
    "Moderate": "#BA7517",
    "Low":      "#1D9E75",
    "Unknown":  "#888780",
}

ORGAN_NORMALS = {
    "liver":  1500.0,
    "kidney": 150.0,
    "spleen": 200.0,
}