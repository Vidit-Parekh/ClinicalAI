"""
ClinicalAI — Page 3: Outcome Prediction
Location: phase6_dashboard/pages/03_outcome_prediction.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.data_loader import (
    load_ml_predictions, load_master, load_shap_importance,
    risk_label, RISK_COLORS,
)

st.set_page_config(page_title="Outcome Prediction", layout="wide")
st.title("🎯 Outcome Prediction")
st.caption(
    "XGBoost in-hospital mortality model trained on tabular features + NLP severity scores. "
    "SHAP values explain individual predictions."
)

pred = load_ml_predictions()
master = load_master()
shap_df = load_shap_importance()

if pred.empty:
    st.warning("ml_predictions.csv not found. Run phase3_ml/outcome_model.py first.")
    st.stop()

TARGET = "DIED"
y_true = pred[TARGET].values
y_prob = pred["PRED_PROB_DIED"].values
y_pred = pred["PRED_DIED"].values

# ── KPIs ──────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
st.divider()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Test patients",   f"{len(pred):,}")
k2.metric("ROC-AUC",         f"{roc_auc_score(y_true, y_prob):.3f}")
k3.metric("AUC-PR",          f"{average_precision_score(y_true, y_prob):.3f}")
k4.metric("F1 (died)",       f"{f1_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}")
k5.metric("Mortality (test)",f"{y_true.mean():.1%}")
st.divider()

col1, col2 = st.columns(2)

# ── ROC Curve ─────────────────────────────────────────────────────────
with col1:
    st.subheader("ROC curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color="#534AB7", width=2),
        fill="tozeroy", fillcolor="rgba(83,74,183,0.08)",
        name=f"XGBoost (AUC={roc_auc:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(color="#888780", dash="dash"), name="Random",
    ))
    fig.update_layout(
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        legend=dict(x=0.6, y=0.1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Confusion matrix ──────────────────────────────────────────────────
with col2:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig2 = px.imshow(
        cm_n,
        labels=dict(x="Predicted", y="Actual", color="Rate"),
        x=["Survived", "Died"],
        y=["Survived", "Died"],
        color_continuous_scale=["#EEEDFE", "#534AB7"],
        text_auto=".2f",
    )
    # Overlay raw counts
    for i in range(2):
        for j in range(2):
            fig2.add_annotation(
                x=j, y=i, text=f"n={cm[i,j]}",
                showarrow=False, font=dict(size=11, color="#888780"),
                yshift=-14,
            )
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ── SHAP feature importance ───────────────────────────────────────────
st.subheader("SHAP feature importance — top 15 predictors")
if not shap_df.empty:
    top15 = shap_df.head(15).sort_values("shap_mean")
    fig3 = px.bar(
        top15, x="shap_mean", y="feature",
        orientation="h",
        color="shap_mean",
        color_continuous_scale=["#CECBF6", "#534AB7"],
    )
    fig3.update_layout(
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("SHAP values not found. Install shap and re-run outcome_model.py.")

# ── Risk stratification table ─────────────────────────────────────────
st.subheader("Patient risk stratification")
pred_view = pred.copy()
pred_view["RISK"] = pred_view["PRED_PROB_DIED"].apply(risk_label)

risk_summary = (
    pred_view.groupby("RISK")
             .agg(n=("PRED_PROB_DIED", "count"),
                  avg_prob=("PRED_PROB_DIED", "mean"),
                  actual_mortality=(TARGET, "mean"))
             .reset_index()
)
risk_summary.columns = ["Risk tier", "Patients", "Avg predicted prob", "Actual mortality"]
risk_summary["Avg predicted prob"] = risk_summary["Avg predicted prob"].map("{:.1%}".format)
risk_summary["Actual mortality"]   = risk_summary["Actual mortality"].map("{:.1%}".format)
st.dataframe(risk_summary, use_container_width=True, hide_index=True)

# ── Individual patient lookup ─────────────────────────────────────────
st.subheader("Individual patient risk lookup")
st.caption("Browse high-risk predictions.")
high_risk = pred_view[pred_view["RISK"] == "High"].sort_values(
    "PRED_PROB_DIED", ascending=False
).head(20)

display_cols = ["PRED_PROB_DIED", "PRED_DIED", TARGET, "RISK"]
display_cols = [c for c in display_cols if c in high_risk.columns]
st.dataframe(
    high_risk[display_cols].rename(columns={
        "PRED_PROB_DIED": "Predicted probability",
        "PRED_DIED": "Predicted died",
        TARGET: "Actual died",
        "RISK": "Risk tier",
    }),
    use_container_width=True,
    hide_index=True,
)