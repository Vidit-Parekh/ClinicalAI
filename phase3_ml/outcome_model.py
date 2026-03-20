"""
ClinicalAI — Phase 3: Outcome Prediction Model
================================================
Location: phase3_ml/outcome_model.py

Trains XGBoost to predict in-hospital mortality (DIED=1)
from tabular clinical features + NLP severity scores.

Pipeline:
  1. Load feature splits from feature_engineering.py
  2. Hyperparameter tuning via RandomizedSearchCV on val set
  3. Train final XGBoost on train+val with best params
  4. Generate predictions + SHAP explanations on test set
  5. Save model + predictions

Run time: ~2-5 min on CPU (no GPU needed for XGBoost tabular)
Expected AUC-ROC: 0.85-0.92 on MIMIC mortality prediction
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
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
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET       = "DIED"
RANDOM_STATE = 42
MODEL_PATH   = os.path.join(MODEL_DIR, "xgboost_model.pkl")


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Load splits
# ─────────────────────────────────────────────────────────────────────

def load_splits():
    log.info("[1/4] Loading feature splits...")
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(PROCESSED_DIR, f"ml_features_{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"  → Run feature_engineering.py first."
            )
        df = pd.read_csv(path)
        splits[name] = (df.drop(columns=[TARGET]), df[TARGET])
        log.info("   %s: %d rows | mortality %.1f%%",
                 name, len(df), df[TARGET].mean() * 100)

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Hyperparameter search
# ─────────────────────────────────────────────────────────────────────

def tune_hyperparams(X_train, y_train):
    """
    RandomizedSearchCV over key XGBoost hyperparameters.
    Uses 5-fold stratified CV to avoid data leakage.
    Optimises for ROC-AUC (best metric for imbalanced clinical outcomes).
    """
    log.info("[2/4] Hyperparameter tuning (RandomizedSearchCV)...")

    param_dist = {
        "n_estimators":       [200, 300, 500, 700],
        "max_depth":          [3, 4, 5, 6, 7],
        "learning_rate":      [0.01, 0.05, 0.1, 0.2],
        "subsample":          [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":   [0.6, 0.7, 0.8, 1.0],
        "min_child_weight":   [1, 3, 5, 7],
        "gamma":              [0, 0.1, 0.2, 0.5],
        "reg_alpha":          [0, 0.01, 0.1, 1.0],
        "reg_lambda":         [0.5, 1.0, 2.0, 5.0],
    }

    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",     # fast for large datasets
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=30,              # 30 random combinations
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    log.info("   Best CV AUC : %.4f", search.best_score_)
    log.info("   Best params : %s", search.best_params_)
    return search.best_params_


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Train final model
# ─────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, best_params):
    log.info("[3/4] Training final XGBoost model...")

    # Combine train + val for final training
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    # Scale pos_weight to handle any residual class imbalance
    pos_weight = (y_full == 0).sum() / (y_full == 1).sum()
    log.info("   pos_weight (class balance): %.2f", pos_weight)

    model = xgb.XGBClassifier(
        **best_params,
        objective    = "binary:logistic",
        eval_metric  = "auc",
        use_label_encoder = False,
        scale_pos_weight  = pos_weight,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
        tree_method  = "hist",
        early_stopping_rounds = 20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    joblib.dump(model, MODEL_PATH)
    log.info("   Model saved → %s", MODEL_PATH)
    return model


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Evaluate + SHAP + save predictions
# ─────────────────────────────────────────────────────────────────────

def evaluate_and_explain(model, X_test, y_test):
    log.info("[4/4] Evaluating on test set + SHAP explanations...")

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Core metrics
    auc    = roc_auc_score(y_test, y_prob)
    auprc  = average_precision_score(y_test, y_prob)

    log.info("   ROC-AUC  : %.4f", auc)
    log.info("   AUC-PR   : %.4f  (important for imbalanced outcomes)", auprc)
    log.info("\n%s", classification_report(
        y_test, y_pred,
        target_names=["survived", "died"],
        digits=4,
    ))

    # SHAP feature importance
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Mean absolute SHAP per feature
        shap_df = pd.DataFrame({
            "feature":    X_test.columns,
            "shap_mean":  np.abs(shap_values).mean(axis=0),
        }).sort_values("shap_mean", ascending=False)

        shap_path = os.path.join(MODEL_DIR, "shap_importance.csv")
        shap_df.to_csv(shap_path, index=False)
        log.info("   SHAP importance saved → %s", shap_path)
        log.info("   Top 5 features by SHAP:\n%s",
                 shap_df.head(5).to_string(index=False))

    except ImportError:
        log.warning("   shap not installed — skipping SHAP. pip install shap")
        shap_values = None

    # Save predictions CSV
    pred_df = X_test.copy()
    pred_df[TARGET]          = y_test.values
    pred_df["PRED_PROB_DIED"] = y_prob
    pred_df["PRED_DIED"]      = y_pred
    pred_df["CORRECT"]        = (y_pred == y_test.values).astype(int)

    pred_path = os.path.join(PROCESSED_DIR, "ml_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    log.info("   Predictions saved → %s", pred_path)

    return auc, auprc, y_prob, y_pred


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def train():
    log.info("══ ClinicalAI — Phase 3: XGBoost Outcome Model ══")

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="xgboost_mortality_prediction"):

        best_params = tune_hyperparams(X_train, y_train)
        mlflow.log_params(best_params)

        model = train_model(X_train, y_train, X_val, y_val, best_params)

        auc, auprc, y_prob, y_pred = evaluate_and_explain(
            model, X_test, y_test
        )
        mlflow.log_metric("test_roc_auc", auc)
        mlflow.log_metric("test_auc_pr",  auprc)
        mlflow.log_artifact(MODEL_PATH)

    log.info("══ Phase 3 training complete ══")
    log.info("   Run evaluate_ml.py to generate charts.")
    return model


if __name__ == "__main__":
    train()