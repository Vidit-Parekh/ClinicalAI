"""
ClinicalAI — Phase 3: ML Model Evaluation
==========================================
Location: phase3_ml/evaluate_ml.py

Reads:  data/processed/ml_predictions.csv
        models/xgboost_model.pkl
        models/shap_importance.csv  (if shap was installed)

Produces:
  phase3_ml/outputs/phase3_ml_report.png   (4-panel chart)
  phase3_ml/outputs/classification_report.txt

Run AFTER outcome_model.py has completed.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)
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
OUTPUT_DIR    = os.path.join("phase3_ml", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "DIED"


# ─────────────────────────────────────────────────────────────────────
# Load predictions
# ─────────────────────────────────────────────────────────────────────

def load_predictions():
    path = os.path.join(PROCESSED_DIR, "ml_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"  → Run outcome_model.py first."
        )
    df = pd.read_csv(path)
    log.info("Loaded %d predictions | mortality: %.1f%%",
             len(df), df[TARGET].mean() * 100)
    return df


# ─────────────────────────────────────────────────────────────────────
# 4-panel evaluation chart
# ─────────────────────────────────────────────────────────────────────

def plot_evaluation(df: pd.DataFrame):
    log.info("Generating evaluation charts...")

    y_true = df[TARGET].values
    y_prob = df["PRED_PROB_DIED"].values
    y_pred = df["PRED_DIED"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "ClinicalAI — Phase 3: XGBoost Mortality Prediction",
        fontsize=15, fontweight="bold",
    )

    # ── Plot 1: ROC Curve ─────────────────────────────────────────────
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#534AB7", linewidth=2,
             label=f"XGBoost  (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
             label="Random classifier")
    ax1.fill_between(fpr, tpr, alpha=0.08, color="#534AB7")
    ax1.set_title("ROC curve", fontsize=12)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Plot 2: Precision-Recall Curve ────────────────────────────────
    ax2 = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc         = average_precision_score(y_true, y_prob)
    baseline      = y_true.mean()
    ax2.plot(rec, prec, color="#1D9E75", linewidth=2,
             label=f"XGBoost  (AUC-PR = {auprc:.3f})")
    ax2.axhline(baseline, color="#888780", linewidth=1,
                linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax2.fill_between(rec, prec, alpha=0.08, color="#1D9E75")
    ax2.set_title("Precision-recall curve", fontsize=12)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Plot 3: Confusion Matrix ───────────────────────────────────────
    ax3 = axes[1, 0]
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_n, ax=ax3,
        annot=True, fmt=".2f",
        cmap=sns.light_palette("#534AB7", as_cmap=True),
        xticklabels=["survived", "died"],
        yticklabels=["survived", "died"],
        linewidths=0.5, linecolor="#D3D1C7",
        annot_kws={"size": 13},
        cbar_kws={"shrink": 0.8},
    )
    for i in range(2):
        for j in range(2):
            ax3.text(j + 0.5, i + 0.72, f"n={cm[i,j]}",
                     ha="center", fontsize=9, color="#888780")
    ax3.set_title("Confusion matrix (normalized)", fontsize=12)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    # ── Plot 4: SHAP Feature Importance ───────────────────────────────
    ax4 = axes[1, 1]
    shap_path = os.path.join(MODEL_DIR, "shap_importance.csv")
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path).head(15).sort_values("shap_mean")
        colors  = ["#534AB7" if v > shap_df["shap_mean"].median()
                   else "#AFA9EC" for v in shap_df["shap_mean"]]
        ax4.barh(shap_df["feature"], shap_df["shap_mean"],
                 color=colors, edgecolor="none")
        ax4.set_title("Top 15 features by SHAP importance", fontsize=12)
        ax4.set_xlabel("Mean |SHAP value|")
        ax4.spines[["top", "right"]].set_visible(False)
    else:
        # Fallback: XGBoost built-in feature importance
        model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
        if os.path.exists(model_path):
            model     = joblib.load(model_path)
            imp       = model.feature_importances_
            feat_cols = df.drop(
                columns=[TARGET, "PRED_PROB_DIED", "PRED_DIED", "CORRECT"],
                errors="ignore"
            ).columns.tolist()
            fi_df = pd.DataFrame({"feature": feat_cols, "importance": imp})
            fi_df = fi_df.nlargest(15, "importance").sort_values("importance")
            ax4.barh(fi_df["feature"], fi_df["importance"],
                     color="#534AB7", edgecolor="none")
            ax4.set_title("Top 15 features (XGBoost gain)", fontsize=12)
            ax4.set_xlabel("Feature importance (gain)")
            ax4.spines[["top", "right"]].set_visible(False)
        else:
            ax4.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "phase3_ml_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Chart saved → %s", out_path)

    # Save text report
    report = classification_report(
        y_true, y_pred,
        target_names=["survived", "died"],
        digits=4,
    )
    txt_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write("ClinicalAI — Phase 3 XGBoost Evaluation\n")
        f.write("=" * 45 + "\n\n")
        f.write(report)
        f.write(f"\nROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}\n")
        f.write(f"AUC-PR   : {average_precision_score(y_true, y_prob):.4f}\n")
    log.info("Text report saved → %s", txt_path)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("══ ClinicalAI — Phase 3: Evaluation ══")
    df = load_predictions()
    plot_evaluation(df)
    log.info("══ Evaluation complete ══")
    log.info("Outputs in: %s", OUTPUT_DIR)