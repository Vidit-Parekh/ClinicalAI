"""
ClinicalAI — Page 1: Patient Overview
Location: phase6_dashboard/pages/01_patient_overview.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.data_loader import load_master

st.set_page_config(page_title="Patient Overview", layout="wide")
st.title("📊 Patient Overview")
st.caption("Demographics, admission statistics, and mortality rates from MIMIC-III-10k.")

df = load_master()

# ── Top KPI row ───────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total admissions",    f"{len(df):,}")
k2.metric("Unique patients",      f"{df['SUBJECT_ID'].nunique():,}")
k3.metric("Avg age",              f"{df['AGE'].dropna().mean():.1f} yrs")
k4.metric("Avg LOS",              f"{df['LOS_DAYS'].dropna().mean():.1f} days")
k5.metric("Mortality rate",       f"{df['DIED'].mean():.1%}")
st.divider()

col1, col2 = st.columns(2)

# ── Admission type breakdown ──────────────────────────────────────────
with col1:
    st.subheader("Admission type distribution")
    adm_counts = df["ADMISSION_TYPE"].value_counts().reset_index()
    adm_counts.columns = ["type", "count"]
    fig = px.bar(
        adm_counts, x="type", y="count",
        color="type",
        color_discrete_sequence=["#534AB7", "#1D9E75", "#D85A30", "#888780"],
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Admissions",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ── Mortality by admission type ───────────────────────────────────────
with col2:
    st.subheader("Mortality rate by admission type")
    mort = (df.groupby("ADMISSION_TYPE")["DIED"]
              .agg(["mean", "count"])
              .reset_index()
              .sort_values("mean", ascending=True))
    mort.columns = ["type", "rate", "count"]
    colors = ["#1D9E75" if r < 0.10 else "#BA7517" if r < 0.20 else "#E24B4A"
              for r in mort["rate"]]
    fig2 = go.Figure(go.Bar(
        x=mort["rate"], y=mort["type"],
        orientation="h",
        marker_color=colors,
        text=[f"{r:.1%}" for r in mort["rate"]],
        textposition="outside",
    ))
    fig2.update_layout(xaxis_title="Mortality rate", yaxis_title="",
                       xaxis_tickformat=".0%",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

# ── Age distribution ──────────────────────────────────────────────────
with col3:
    st.subheader("Age distribution")
    fig3 = px.histogram(
        df.dropna(subset=["AGE"]), x="AGE", nbins=30,
        color_discrete_sequence=["#534AB7"],
    )
    fig3.update_layout(xaxis_title="Age (years)", yaxis_title="Admissions",
                       bargap=0.05, showlegend=False,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

# ── Length of stay distribution ───────────────────────────────────────
with col4:
    st.subheader("Length of stay distribution")
    los = df["LOS_DAYS"].dropna().clip(0, 60)
    fig4 = px.histogram(
        los, nbins=40, color_discrete_sequence=["#1D9E75"],
    )
    fig4.add_vline(x=los.mean(), line_dash="dash", line_color="#D85A30",
                   annotation_text=f"Mean: {los.mean():.1f}d")
    fig4.update_layout(xaxis_title="Days", yaxis_title="Admissions",
                       bargap=0.05, showlegend=False,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

# ── ICD-9 category breakdown ──────────────────────────────────────────
st.subheader("Primary ICD-9 disease categories")
if "ICD9_CATEGORY" in df.columns:
    icd = df["ICD9_CATEGORY"].value_counts().reset_index()
    icd.columns = ["category", "count"]
    fig5 = px.bar(
        icd.head(10), x="count", y="category",
        orientation="h",
        color="count",
        color_continuous_scale=["#CECBF6", "#534AB7"],
        text="count",
    )
    fig5.update_traces(textposition="outside")
    fig5.update_layout(
        yaxis=dict(autorange="reversed"), showlegend=False,
        xaxis_title="Admissions", yaxis_title="",
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── Insurance breakdown ───────────────────────────────────────────────
st.subheader("Insurance distribution")
if "INSURANCE" in df.columns:
    ins = df["INSURANCE"].value_counts().reset_index()
    ins.columns = ["insurance", "count"]
    fig6 = px.pie(
        ins, names="insurance", values="count",
        color_discrete_sequence=["#534AB7", "#1D9E75", "#D85A30", "#888780", "#FAC775"],
        hole=0.4,
    )
    fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig6, use_container_width=True)