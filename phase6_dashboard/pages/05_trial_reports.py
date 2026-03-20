"""
ClinicalAI — Page 5: Trial Reports
Location: phase6_dashboard/pages/05_trial_reports.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.data_loader import (
    load_clinical_reports, load_master, load_ml_predictions,
    risk_label, RISK_COLORS,
)

st.set_page_config(page_title="Trial Reports", layout="wide")
st.title("📋 Clinical Trial Reports")
st.caption(
    "Auto-generated plain-English clinical insights from the T5 fine-tuned summarization model. "
    "Simulates Medpace IntelliPACE trial progress reporting."
)

reports = load_clinical_reports()
master  = load_master()
pred    = load_ml_predictions()

if reports.empty:
    st.warning("clinical_reports.csv not found. Run phase5_llm/insights_generator.py first.")
    st.stop()

# Merge in mortality risk if available
if not pred.empty and "HADM_ID" in pred.columns:
    reports = reports.merge(
        pred[["HADM_ID", "PRED_PROB_DIED"]],
        on="HADM_ID", how="left",
    )
    reports["RISK_TIER"] = reports["PRED_PROB_DIED"].apply(risk_label)
else:
    reports["RISK_TIER"] = "Unknown"

# ── KPIs ──────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Reports generated", f"{len(reports):,}")
k2.metric("High risk patients",
          f"{(reports['RISK_TIER']=='High').sum():,}" if "RISK_TIER" in reports.columns else "N/A")
k3.metric("Died in cohort",
          f"{reports['DIED'].sum():,}" if "DIED" in reports.columns else "N/A")
k4.metric("Avg report length",
          f"{reports['clinical_report'].str.len().mean():.0f} chars")
st.divider()

# ── Filters ───────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    risk_filter = st.multiselect(
        "Filter by risk tier",
        ["High", "Moderate", "Low", "Unknown"],
        default=["High", "Moderate"],
    )
with col_f2:
    outcome_filter = st.selectbox(
        "Filter by outcome",
        ["All", "Died", "Survived"],
    )
with col_f3:
    search = st.text_input("Search diagnosis", placeholder="e.g. sepsis, pneumonia")

# Apply filters
filtered = reports.copy()
if risk_filter and "RISK_TIER" in filtered.columns:
    filtered = filtered[filtered["RISK_TIER"].isin(risk_filter)]
if outcome_filter == "Died" and "DIED" in filtered.columns:
    filtered = filtered[filtered["DIED"] == 1]
elif outcome_filter == "Survived" and "DIED" in filtered.columns:
    filtered = filtered[filtered["DIED"] == 0]
if search and "DIAGNOSIS" in filtered.columns:
    filtered = filtered[
        filtered["DIAGNOSIS"].str.lower().str.contains(search.lower(), na=False)
    ]

st.caption(f"Showing {len(filtered):,} reports")

# ── Report cards ──────────────────────────────────────────────────────
st.subheader("Clinical reports")
page_size = 10
page      = st.number_input("Page", min_value=1,
                             max_value=max(1, len(filtered) // page_size + 1),
                             value=1)
start = (page - 1) * page_size
end   = start + page_size

for _, row in filtered.iloc[start:end].iterrows():
    risk  = row.get("RISK_TIER", "Unknown")
    died  = "Died" if row.get("DIED", 0) == 1 else "Survived"
    diag  = str(row.get("DIAGNOSIS", "Unknown"))[:60]
    prob  = row.get("PRED_PROB_DIED", np.nan)
    prob_str = f"{prob:.1%}" if not pd.isna(prob) else "N/A"

    color = RISK_COLORS.get(risk, "#888780")
    hadm  = int(row['HADM_ID']) if pd.notna(row['HADM_ID']) else "N/A"
    label = f"HADM {hadm}  ·  {diag}  ·  {died}  ·  Risk: {risk} ({prob_str})"

    with st.expander(label):
        st.markdown(
            f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
            f"{row['clinical_report']}"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Summary analytics ─────────────────────────────────────────────────
st.divider()
st.subheader("Report analytics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Risk tier distribution in filtered cohort**")
    if "RISK_TIER" in filtered.columns:
        rt = filtered["RISK_TIER"].value_counts().reset_index()
        rt.columns = ["Risk tier", "Count"]
        fig = px.pie(
            rt, names="Risk tier", values="Count",
            color="Risk tier",
            color_discrete_map=RISK_COLORS,
            hole=0.4,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Top diagnoses in filtered cohort**")
    if "DIAGNOSIS" in filtered.columns:
        top_dx = (filtered["DIAGNOSIS"]
                          .value_counts()
                          .head(10)
                          .reset_index())
        top_dx.columns = ["Diagnosis", "Count"]
        fig2 = px.bar(
            top_dx.sort_values("Count"),
            x="Count", y="Diagnosis",
            orientation="h",
            color_discrete_sequence=["#534AB7"],
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)