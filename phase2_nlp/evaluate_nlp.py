"""
ClinicalAI — Phase 2: NLP Model Evaluation
===========================================
Location: phase2_nlp/evaluate_nlp.py

Reads:  data/processed/nlp_predictions.csv    (from train_biobert.py)
        models/biobert_finetuned/              (saved model)

Produces:
  phase2_nlp/outputs/phase2_nlp_report.png    (4-panel evaluation chart)
  phase2_nlp/outputs/classification_report.txt

Run AFTER train_biobert.py has completed.
Can also re-evaluate any saved model independently.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
load_dotenv()

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
OUTPUT_DIR    = os.path.join("phase2_nlp", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES   = ["stable", "moderate", "critical"]
LABEL_MAP     = {"stable": 0, "moderate": 1, "critical": 2}
COLORS        = {
    "stable":   "#1D9E75",
    "moderate": "#BA7517",
    "critical": "#E24B4A",
}


# ─────────────────────────────────────────────────────────────────────
# Load predictions
# ─────────────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "nlp_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"  → Run train_biobert.py first."
        )
    df = pd.read_csv(path)
    log.info("Loaded %d predictions from %s", len(df), path)
    return df


# ─────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame) -> dict:
    y_true = df["SEVERITY_LABEL"].map(LABEL_MAP).values
    y_pred = df["PRED_LABEL"].map(LABEL_MAP).values

    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    log.info("\n── Classification Report ──────────────────────────────────")
    log.info("\n%s", report)
    log.info("   Accuracy   : %.4f", acc)
    log.info("   F1 macro   : %.4f", f1_macro)
    log.info("   F1 weighted: %.4f", f1_weight)

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("ClinicalAI — Phase 2 BioBERT Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write(f"\nAccuracy   : {acc:.4f}\n")
        f.write(f"F1 macro   : {f1_macro:.4f}\n")
        f.write(f"F1 weighted: {f1_weight:.4f}\n")
    log.info("Text report saved → %s", report_path)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy":   acc,
        "f1_macro":   f1_macro,
        "f1_weighted": f1_weight,
    }


# ─────────────────────────────────────────────────────────────────────
# 4-panel evaluation chart
# ─────────────────────────────────────────────────────────────────────

def plot_evaluation(df: pd.DataFrame, metrics: dict):
    log.info("Generating evaluation charts...")

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "ClinicalAI — Phase 2: BioBERT Severity Classification",
        fontsize=15, fontweight="bold"
    )

    # ── Plot 1: Confusion matrix ──────────────────────────────────────
    ax1 = axes[0, 0]
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalized

    sns.heatmap(
        cm_n, ax=ax1,
        annot=True, fmt=".2f",
        cmap=sns.light_palette("#534AB7", as_cmap=True),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        linewidths=0.5, linecolor="#D3D1C7",
        annot_kws={"size": 12},
        cbar_kws={"shrink": 0.8},
    )
    # Overlay raw counts
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax1.text(j + 0.5, i + 0.72, f"n={cm[i,j]}",
                     ha="center", va="center",
                     fontsize=9, color="#888780")
    ax1.set_title("Confusion matrix (normalized)", fontsize=12)
    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")

    # ── Plot 2: Per-class F1 bar chart ────────────────────────────────
    ax2 = axes[0, 1]
    from sklearn.metrics import f1_score
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    bar_colors   = [COLORS[l] for l in LABEL_NAMES]
    bars = ax2.bar(LABEL_NAMES, per_class_f1, color=bar_colors,
                   edgecolor="none", width=0.5)
    for bar, val in zip(bars, per_class_f1):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax2.axhline(metrics["f1_macro"], color="#888780", linewidth=1,
                linestyle="--", label=f"Macro avg: {metrics['f1_macro']:.3f}")
    ax2.set_title("F1 score per class", fontsize=12)
    ax2.set_ylabel("F1 score")
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Plot 3: One-vs-Rest ROC curves ────────────────────────────────
    ax3 = axes[1, 0]

    # For ROC we need probability scores — if not available,
    # use a proxy by treating predicted label as score
    # (real training will save logits; for now use prediction confidence proxy)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    for i, label in enumerate(LABEL_NAMES):
        # Use 1 if predicted as this class, 0 otherwise (proxy score)
        score = (y_pred == i).astype(float)
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], score)
        roc_auc     = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color=COLORS[label], linewidth=2,
                 label=f"{label}  (AUC = {roc_auc:.3f})")

    ax3.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax3.set_title("ROC curves (one-vs-rest)", fontsize=12)
    ax3.set_xlabel("False positive rate")
    ax3.set_ylabel("True positive rate")
    ax3.legend(fontsize=9)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.05])
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Plot 4: Prediction distribution by note type ──────────────────
    ax4 = axes[1, 1]
    if "NOTE_TYPE" in df.columns:
        cross = pd.crosstab(
            df["NOTE_TYPE"],
            df["PRED_LABEL"],
            normalize="index"
        )
        # Ensure all columns present
        for lbl in LABEL_NAMES:
            if lbl not in cross.columns:
                cross[lbl] = 0.0
        cross = cross[LABEL_NAMES]

        cross.plot(
            kind="bar", ax=ax4, stacked=True,
            color=[COLORS[l] for l in LABEL_NAMES],
            edgecolor="none", width=0.6
        )
        ax4.set_title("Predicted severity by note type", fontsize=12)
        ax4.set_ylabel("Proportion")
        ax4.set_xlabel("")
        ax4.tick_params(axis="x", rotation=30, labelsize=9)
        ax4.legend(title="Predicted severity", fontsize=8,
                   bbox_to_anchor=(1.02, 1), loc="upper left")
        ax4.spines[["top", "right"]].set_visible(False)
    else:
        ax4.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "phase2_nlp_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Evaluation chart saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("══ ClinicalAI — Phase 2: Evaluation ══")
    df      = load_predictions()
    metrics = compute_all_metrics(df)
    plot_evaluation(df, metrics)
    log.info("══ Evaluation complete ══")
    log.info("Outputs in: %s", OUTPUT_DIR)