"""
ClinicalAI — Phase 2: BioBERT Fine-Tuning
==========================================
Location: phase2_nlp/train_biobert.py

Fine-tunes Bio_ClinicalBERT on MIMIC clinical notes
for 3-class severity classification:
  0 = stable | 1 = moderate | 2 = critical

Uses HuggingFace Trainer API with:
  - Weighted loss to handle class imbalance
  - Early stopping on validation F1
  - Automatic GPU/MPS/CPU device selection
  - MLflow experiment tracking (optional)

Saves model to: models/biobert_finetuned/

Run time estimate:
  CPU  : ~20-40 min for 10k notes, 3 epochs
  GPU  : ~3-5 min
"""

import os
import logging
import numpy as np
import torch
import mlflow
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModel,                              # base encoder — loaded by nlp_pipeline
    AutoModelForSequenceClassification,     # adds 3-class head for fine-tuning
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)
import warnings
warnings.filterwarnings("ignore")

# Local imports
from nlp_pipeline import (
    run_pipeline, LABEL_MAP, ID_TO_LABEL, NUM_LABELS,
    SAMPLE_SIZE, MAX_LEN,
)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
load_dotenv()

MODEL_DIR     = os.getenv("MODEL_DIR",     "models")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "distilbert-base-uncased")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

SAVE_DIR    = os.path.join(MODEL_DIR, "distilbert_finetuned")
CHECKPOINTS = os.path.join(MODEL_DIR, "checkpoints")
os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(CHECKPOINTS, exist_ok=True)

# ── Hyperparameters — tuned for GTX 1650 (4 GB VRAM) + DistilBERT ───
# DistilBERT is 40% smaller than BERT → larger batches fit in VRAM.
# Gradient accumulation of 4 simulates an effective batch of 128
# without exceeding VRAM (32 × 4 = 128 effective batch size).
NUM_EPOCHS                 = 4       # DistilBERT needs one extra epoch vs BERT
LEARNING_RATE              = 3e-5    # slightly higher than BERT (2e-5) — DistilBERT converges faster
WARMUP_RATIO               = 0.1
WEIGHT_DECAY               = 0.01
BATCH_SIZE_TRAIN           = 32      # fits on GTX 1650 with MAX_LEN=128
BATCH_SIZE_EVAL            = 64
GRADIENT_ACCUMULATION_STEPS = 4     # effective batch = 32 × 4 = 128
EVAL_STRATEGY              = "epoch"
SAVE_STRATEGY              = "epoch"
LOAD_BEST_MODEL            = True
METRIC_FOR_BEST            = "eval_f1_macro"


# ─────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        log.info("Device: GPU (%s)", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        log.info("Device: Apple MPS (M-series chip)")
    else:
        device = "cpu"
        log.warning("Device: CPU — training will be slow. Consider a GPU runtime.")
    return device


# ─────────────────────────────────────────────────────────────────────
# Class-weighted loss (handles imbalanced clinical labels)
# ─────────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights to CrossEntropyLoss.
    This prevents the model from ignoring the minority 'critical' class.
    """

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Called by Trainer after each evaluation step.
    Returns macro F1 (best for imbalanced classes) + accuracy.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1_macro":  f1_score(labels, preds, average="macro",  zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────

def train():
    log.info("══ ClinicalAI — Phase 2: BioBERT Fine-Tuning ══")
    device = get_device()

    # ── Step 1: Build lazy datasets via nlp_pipeline ─────────────────
    # run_pipeline() loads tokenizer + splits data into lazy Datasets.
    # No upfront tokenization — tokenization fires per batch in __getitem__.
    train_ds, val_ds, test_ds, tokenizer, test_df = run_pipeline()

    # ── Step 2: Compute class weights ────────────────────────────────
    all_labels   = [item["labels"].item() for item in train_ds]
    classes      = np.unique(all_labels)
    weights      = compute_class_weight("balanced", classes=classes, y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    log.info("Class weights: %s", dict(zip(
        [ID_TO_LABEL[c] for c in classes], weights.round(3)
    )))

    # ── Step 3: Load Bio_ClinicalBERT for classification ─────────────
    # AutoModelForSequenceClassification wraps the same pretrained weights
    # as AutoModel, but adds a randomly-initialized linear classification
    # head on top of the [CLS] token output.
    # Architecture: Bio_ClinicalBERT encoder (frozen initially) → dropout
    #               → Linear(768, 3) → softmax → stable/moderate/critical
    log.info("Loading model for classification: %s", HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
        ignore_mismatched_sizes=True,   # replaces pretrain head with 3-class head
    )

    # ── Step 4: Training arguments ────────────────────────────────────
    # Note: transformers>=4.41 renamed evaluation_strategy → eval_strategy
    training_args = TrainingArguments(
        output_dir                   = CHECKPOINTS,
        num_train_epochs             = NUM_EPOCHS,
        per_device_train_batch_size  = BATCH_SIZE_TRAIN,
        per_device_eval_batch_size   = BATCH_SIZE_EVAL,
        gradient_accumulation_steps  = GRADIENT_ACCUMULATION_STEPS,
        learning_rate                = LEARNING_RATE,
        warmup_ratio                 = WARMUP_RATIO,
        weight_decay                 = WEIGHT_DECAY,
        eval_strategy                = EVAL_STRATEGY,
        save_strategy                = SAVE_STRATEGY,
        load_best_model_at_end       = LOAD_BEST_MODEL,
        metric_for_best_model        = METRIC_FOR_BEST,
        greater_is_better            = True,
        logging_dir                  = os.path.join(MODEL_DIR, "logs"),
        logging_steps                = 50,
        report_to                    = "none",
        fp16                         = torch.cuda.is_available(),
        seed                         = 42,
        dataloader_num_workers       = 0,
    )

    # ── Step 5: Trainer ───────────────────────────────────────────────
    trainer = WeightedTrainer(
        class_weights = class_weights,
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        compute_metrics = compute_metrics,
        callbacks     = [
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
            )
        ],
    )

    # ── Step 6: Train ─────────────────────────────────────────────────
    log.info("Starting fine-tuning — %d epochs, lr=%.0e, batch=%d",
             NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE_TRAIN)

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="distilbert_clinical_severity"):
        mlflow.log_params({
            "model":                 HF_MODEL_NAME,
            "epochs":                NUM_EPOCHS,
            "lr":                    LEARNING_RATE,
            "batch_size":            BATCH_SIZE_TRAIN,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size":  BATCH_SIZE_TRAIN * GRADIENT_ACCUMULATION_STEPS,
            "max_len":               MAX_LEN,
            "sample_size":           SAMPLE_SIZE,
            "device":                device,
        })

        train_result = trainer.train()
        log.info("Training complete — loss: %.4f", train_result.training_loss)

        # ── Step 7: Save model ────────────────────────────────────────
        trainer.save_model(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        log.info("Model saved → %s", SAVE_DIR)

        # ── Step 8: Evaluate on test set ──────────────────────────────
        log.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_ds)
        log.info("Test results: %s", {
            k: round(v, 4) for k, v in test_results.items()
        })

        # Full classification report
        test_preds  = trainer.predict(test_ds)
        pred_labels = np.argmax(test_preds.predictions, axis=-1)
        true_labels = [item["labels"].item() for item in test_ds]

        report = classification_report(
            true_labels, pred_labels,
            target_names=[ID_TO_LABEL[i] for i in range(NUM_LABELS)],
            digits=4,
        )
        log.info("\n── Classification Report ──────────────────────\n%s", report)

        mlflow.log_metric("test_f1_macro", test_results.get("eval_f1_macro", 0))
        mlflow.log_metric("test_accuracy", test_results.get("eval_accuracy", 0))

    # Save predictions for evaluate_nlp.py
    import pandas as pd
    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["PRED_LABEL_ID"] = pred_labels
    pred_df["PRED_LABEL"]    = pred_df["PRED_LABEL_ID"].map(ID_TO_LABEL)
    pred_df["CORRECT"]       = (pred_df["LABEL_ID"] == pred_df["PRED_LABEL_ID"]).astype(int)

    pred_path = os.path.join(os.getenv("PROCESSED_DIR", "data/processed"),
                             "nlp_predictions.csv")
    pred_df[["SUBJECT_ID", "HADM_ID", "NOTE_TYPE", "SEVERITY_LABEL",
             "PRED_LABEL", "CORRECT", "TEXT_CLEAN"]].to_csv(pred_path, index=False)
    log.info("Predictions saved → %s", pred_path)
    log.info("══ Phase 2 training complete ══")

    return trainer, pred_df


if __name__ == "__main__":
    trainer, pred_df = train()