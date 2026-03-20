"""
ClinicalAI — Phase 6: Streamlit Analytics Dashboard
=====================================================
Location: phase6_dashboard/app.py

Entry point for the Streamlit multi-page app.
Run: streamlit run phase6_dashboard/app.py

Pages:
  01_patient_overview.py    — demographics, admission stats, mortality
  02_nlp_insights.py        — NLP severity predictions & note analysis
  03_outcome_prediction.py  — XGBoost predictions, SHAP, risk stratification
  04_organ_volumes.py        — organ volume predictions (MCL simulation)
  05_trial_reports.py        — auto-generated clinical insights reports
"""

import streamlit as st

st.set_page_config(
    page_title="ClinicalAI — Medpace Trial Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ClinicalAI")
    st.markdown("*Medpace-inspired Clinical Trial Analytics*")
    st.divider()
    st.caption("Powered by MIMIC-III · XGBoost · DistilBERT · T5")
    st.caption("Phases 1–5 pipeline outputs loaded automatically.")

# ── Landing page ──────────────────────────────────────────────────────
st.title("ClinicalAI — Clinical Trial Analytics Platform")
st.markdown(
    "Inspired by **Medpace IntelliPACE** and **MCL** — "
    "an end-to-end AI pipeline for accelerating clinical trials."
)

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Patients", "10,000")
    st.caption("MIMIC-III-10k cohort")

with col2:
    st.metric("Admissions", "12,911")
    st.caption("Hospital stays")

with col3:
    st.metric("Clinical Notes", "480,361")
    st.caption("NLP processed")

with col4:
    st.metric("Lab Events", "6.6M")
    st.caption("Across 677 ITEMIDs")

with col5:
    st.metric("ICD-9 Codes", "4,252")
    st.caption("Unique diagnoses")

st.divider()

st.markdown("### Navigate the platform")

cols = st.columns(5)
pages = [
    ("📊", "Patient Overview",     "Demographics, LOS, mortality rates by admission type"),
    ("🧠", "NLP Insights",         "DistilBERT severity predictions on 480k clinical notes"),
    ("🎯", "Outcome Prediction",   "XGBoost in-hospital mortality model + SHAP importance"),
    ("🫀", "Organ Volumes",        "MCL-style liver, kidney, spleen volume predictions"),
    ("📋", "Trial Reports",        "Auto-generated clinical insights from T5 LLM"),
]

for col, (icon, title, desc) in zip(cols, pages):
    with col:
        st.markdown(f"### {icon} {title}")
        st.caption(desc)

st.divider()
st.caption(
    "Data: MIMIC-III Clinical Database (Beth Israel Deaconess Medical Center). "
    "For research and portfolio demonstration purposes only."
)