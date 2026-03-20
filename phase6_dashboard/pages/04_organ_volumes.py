"""
ClinicalAI — Page 4: Organ Volumes
Location: phase6_dashboard/pages/04_organ_volumes.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.data_loader import load_organ_volumes, load_master, ORGAN_NORMALS

st.set_page_config(page_title="Organ Volumes", layout="wide")
st.title("🫀 Organ Volume Predictions")
st.caption(
    "Simulates Medpace MCL (Medical Core Lab) organ volumetrics pipeline. "
    "Predicted liver, kidney, and spleen volumes using lab proxies + GradientBoosting regression."
)

vol = load_organ_volumes()
master = load_master()

if vol.empty:
    st.warning("organ_volume_predictions.csv not found. Run phase4_imaging/volume_predictor.py first.")
    st.stop()

# Merge with master for context
merged = vol.merge(
    master[["HADM_ID", "DIAGNOSIS", "ICD9_CATEGORY", "DIED", "LOS_DAYS", "AGE"]],
    on="HADM_ID", how="left",
)

ORGANS = {
    "liver":  {"label": "Liver",  "color": "#534AB7", "normal": 1500.0, "unit": "mL"},
    "kidney": {"label": "Kidney", "color": "#1D9E75", "normal": 150.0,  "unit": "mL"},
    "spleen": {"label": "Spleen", "color": "#D85A30", "normal": 200.0,  "unit": "mL"},
}

# ── KPIs ──────────────────────────────────────────────────────────────
st.divider()
cols = st.columns(6)
for i, (organ, cfg) in enumerate(ORGANS.items()):
    vol_col = f"{organ}_pred_vol_ml"
    chg_col = f"{organ}_vol_change_pred_pct"
    if vol_col in vol.columns:
        cols[i*2].metric(
            f"{cfg['label']} avg vol",
            f"{vol[vol_col].mean():.0f} mL",
            f"Normal: {cfg['normal']:.0f} mL",
        )
        cols[i*2+1].metric(
            f"{cfg['label']} avg change",
            f"{vol[chg_col].mean():+.1f}%",
        )
st.divider()

# ── Volume distribution per organ ────────────────────────────────────
st.subheader("Predicted volume distributions vs normal reference")
fig = go.Figure()
for organ, cfg in ORGANS.items():
    vol_col = f"{organ}_pred_vol_ml"
    if vol_col not in vol.columns:
        continue
    vols = vol[vol_col].dropna().clip(cfg["normal"] * 0.3, cfg["normal"] * 2.5)
    fig.add_trace(go.Histogram(
        x=vols, name=cfg["label"],
        marker_color=cfg["color"], opacity=0.65,
        nbinsx=40,
    ))
    fig.add_vline(
        x=cfg["normal"], line_dash="dash",
        line_color=cfg["color"], opacity=0.8,
        annotation_text=f"{cfg['label']} normal ({cfg['normal']:.0f} mL)",
        annotation_position="top right",
    )

fig.update_layout(
    barmode="overlay",
    xaxis_title="Predicted volume (mL)",
    yaxis_title="Admissions",
    legend_title="Organ",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# ── Volume change by ICD-9 category ──────────────────────────────────
st.subheader("Volume change by disease category")
organ_sel = st.selectbox("Select organ", list(ORGANS.keys()),
                         format_func=lambda o: ORGANS[o]["label"])
chg_col = f"{organ_sel}_vol_change_pred_pct"

if chg_col in merged.columns and "ICD9_CATEGORY" in merged.columns:
    icd_vol = (merged.dropna(subset=[chg_col, "ICD9_CATEGORY"])
                     .groupby("ICD9_CATEGORY")[chg_col]
                     .mean()
                     .reset_index()
                     .sort_values(chg_col))
    icd_vol.columns = ["ICD9 category", "Mean volume change (%)"]
    colors = [ORGANS[organ_sel]["color"] if v >= 0 else "#E24B4A"
              for v in icd_vol["Mean volume change (%)"]]
    fig2 = px.bar(
        icd_vol, x="Mean volume change (%)", y="ICD9 category",
        orientation="h",
        color="Mean volume change (%)",
        color_continuous_scale=["#E24B4A", "#EEEDFE", ORGANS[organ_sel]["color"]],
        color_continuous_midpoint=0,
    )
    fig2.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Scatter: volume change vs LOS ────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{ORGANS[organ_sel]['label']} volume change vs length of stay")
    if "LOS_DAYS" in merged.columns and chg_col in merged.columns:
        scatter_df = merged[[chg_col, "LOS_DAYS", "DIED"]].dropna()
        scatter_df["Outcome"] = scatter_df["DIED"].map({0: "Survived", 1: "Died"})
        fig3 = px.scatter(
            scatter_df.sample(min(2000, len(scatter_df)), random_state=42),
            x="LOS_DAYS", y=chg_col,
            color="Outcome",
            color_discrete_map={"Survived": "#1D9E75", "Died": "#E24B4A"},
            opacity=0.5, size_max=6,
        )
        fig3.update_layout(
            xaxis_title="Length of stay (days)",
            yaxis_title=f"{ORGANS[organ_sel]['label']} volume change (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader(f"{ORGANS[organ_sel]['label']} volume change vs age")
    if "AGE" in merged.columns and chg_col in merged.columns:
        age_vol = (merged.dropna(subset=["AGE", chg_col])
                         .assign(age_group=lambda d: pd.cut(
                             d["AGE"].clip(0,90),
                             bins=[0,18,40,65,80,91],
                             labels=["<18","18-40","40-65","65-80","80+"],
                         ))
                         .groupby("age_group", observed=True)[chg_col]
                         .mean()
                         .reset_index())
        age_vol.columns = ["Age group", "Mean volume change (%)"]
        fig4 = px.bar(
            age_vol, x="Age group", y="Mean volume change (%)",
            color_discrete_sequence=[ORGANS[organ_sel]["color"]],
            text=age_vol["Mean volume change (%)"].map("{:+.1f}%".format),
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, use_container_width=True)