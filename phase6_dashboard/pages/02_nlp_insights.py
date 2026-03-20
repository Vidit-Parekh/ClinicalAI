"""
ClinicalAI — Page 2: NLP Insights
Location: phase6_dashboard/pages/02_nlp_insights.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.data_loader import load_nlp_predictions, load_master

st.set_page_config(page_title="NLP Insights", layout="wide")
st.title("🧠 NLP Insights")
st.caption(
    "DistilBERT fine-tuned on 20k MIMIC clinical notes. "
    "Classifies each note as stable / moderate / critical severity."
)

nlp = load_nlp_predictions()
master = load_master()

if nlp.empty:
    st.warning("nlp_predictions.csv not found. Run phase2_nlp/train_biobert.py first.")
    st.stop()

SEV_COLORS = {"stable": "#1D9E75", "moderate": "#BA7517", "critical": "#E24B4A"}

# ── KPIs ──────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Notes classified",  f"{len(nlp):,}")
k2.metric("Critical notes",    f"{(nlp['PRED_LABEL']=='critical').mean():.1%}")
k3.metric("Moderate notes",    f"{(nlp['PRED_LABEL']=='moderate').mean():.1%}")
k4.metric("Stable notes",      f"{(nlp['PRED_LABEL']=='stable').mean():.1%}")
st.divider()

col1, col2 = st.columns(2)

# ── Severity distribution pie ─────────────────────────────────────────
with col1:
    st.subheader("Overall severity distribution")
    sev_counts = nlp["PRED_LABEL"].value_counts().reset_index()
    sev_counts.columns = ["severity", "count"]
    fig = px.pie(
        sev_counts, names="severity", values="count",
        color="severity",
        color_discrete_map=SEV_COLORS,
        hole=0.4,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ── Severity by note type ─────────────────────────────────────────────
with col2:
    st.subheader("Severity by note type")
    if "NOTE_TYPE" in nlp.columns:
        cross = (nlp.groupby(["NOTE_TYPE", "PRED_LABEL"])
                    .size()
                    .reset_index(name="count"))
        fig2 = px.bar(
            cross, x="NOTE_TYPE", y="count", color="PRED_LABEL",
            color_discrete_map=SEV_COLORS,
            barmode="stack",
        )
        fig2.update_layout(
            xaxis_title="", yaxis_title="Notes",
            legend_title="Severity",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Per-admission severity profile ────────────────────────────────────
st.subheader("Per-admission severity profile")
st.caption("Select an admission to see the note-level severity breakdown.")

sev_map = {"stable": 0, "moderate": 1, "critical": 2}
nlp["SEV_ID"] = nlp["PRED_LABEL"].map(sev_map)

adm_summary = (
    nlp.groupby("HADM_ID")
       .agg(
           n_notes          = ("PRED_LABEL", "count"),
           max_severity     = ("SEV_ID", "max"),
           critical_frac    = ("PRED_LABEL", lambda x: (x == "critical").mean()),
           moderate_frac    = ("PRED_LABEL", lambda x: (x == "moderate").mean()),
           stable_frac      = ("PRED_LABEL", lambda x: (x == "stable").mean()),
       )
       .reset_index()
)
adm_summary["max_label"] = adm_summary["max_severity"].map(
    {0: "stable", 1: "moderate", 2: "critical"}
)

# Merge with diagnosis for context
if "DIAGNOSIS" in master.columns:
    adm_summary = adm_summary.merge(
        master[["HADM_ID", "DIAGNOSIS", "DIED", "LOS_DAYS"]],
        on="HADM_ID", how="left",
    )

col3, col4 = st.columns(2)

with col3:
    st.subheader("Critical note fraction distribution")
    fig3 = px.histogram(
        adm_summary, x="critical_frac", nbins=30,
        color_discrete_sequence=["#E24B4A"],
    )
    fig3.update_layout(
        xaxis_title="Fraction of notes classified critical",
        yaxis_title="Admissions", bargap=0.05,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Max severity vs mortality")
    if "DIED" in adm_summary.columns:
        mort_sev = (adm_summary.groupby("max_label")["DIED"]
                               .mean()
                               .reset_index())
        mort_sev.columns = ["severity", "mortality_rate"]
        fig4 = px.bar(
            mort_sev, x="severity", y="mortality_rate",
            color="severity", color_discrete_map=SEV_COLORS,
            text=mort_sev["mortality_rate"].map("{:.1%}".format),
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(
            showlegend=False, yaxis_tickformat=".0%",
            xaxis_title="Peak note severity", yaxis_title="Mortality rate",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── Sample notes viewer ───────────────────────────────────────────────
st.subheader("Sample clinical notes by severity")
severity_filter = st.selectbox("Filter by severity", ["critical", "moderate", "stable"])
sample = nlp[nlp["PRED_LABEL"] == severity_filter].sample(
    min(5, len(nlp[nlp["PRED_LABEL"] == severity_filter])),
    random_state=42,
)
for _, row in sample.iterrows():
    hadm = int(row['HADM_ID']) if pd.notna(row['HADM_ID']) else "N/A"
    with st.expander(
        f"HADM {hadm} | {row.get('NOTE_TYPE','Note')} | {severity_filter.upper()}"
    ):
        st.write(str(row.get("TEXT_CLEAN", row.get("text", "N/A")))[:800] + "...")