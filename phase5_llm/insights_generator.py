"""
ClinicalAI — Phase 5: Clinical Insights LLM
=============================================
Location: phase5_llm/insights_generator.py

Builds a fine-tuned summarization model that generates plain-English
clinical trial progress reports from structured model outputs.

Two-stage pipeline:
  Stage A — prepare_training_data()
      Combines outputs from Phases 1-4 into (input, summary) pairs.
      Input  : structured JSON of patient features + model predictions
      Output : natural-language clinical insight paragraph

  Stage B — fine_tune() [optional — runs if GPU available]
      Fine-tunes facebook/bart-large-cnn on the clinical pairs.
      Falls back to zero-shot inference if GPU not available.

  Stage C — generate_reports()
      Generates a report for each admission and saves to CSV + txt.

Reads:
  data/processed/mimic_master_features.csv
  data/processed/ml_predictions.csv
  data/processed/nlp_predictions.csv
  data/processed/organ_volume_predictions.csv

Writes:
  data/processed/training_pairs.csv        (fine-tuning dataset)
  models/summarizer_model/                 (fine-tuned model)
  phase5_llm/outputs/clinical_reports.csv  (one report per admission)
  phase5_llm/outputs/sample_reports/       (txt files for top cases)
"""

import os
import json
import logging
import random
import pandas as pd
import numpy as np
import torch
import joblib
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
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
OUTPUT_DIR    = os.path.join("phase5_llm", "outputs")
REPORT_DIR    = os.path.join(OUTPUT_DIR, "sample_reports")

for d in [OUTPUT_DIR, REPORT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Base summarization model — BART is the standard for clinical summarization
HF_SUMM_MODEL  = "facebook/bart-large-cnn"
SAVE_DIR       = os.path.join(MODEL_DIR, "summarizer_model")
RANDOM_STATE   = 42
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 128


# ─────────────────────────────────────────────────────────────────────
# Stage A — Load and merge all phase outputs
# ─────────────────────────────────────────────────────────────────────

def load_all_outputs() -> pd.DataFrame:
    log.info("[1/5] Loading outputs from all phases...")

    master = pd.read_csv(
        os.path.join(PROCESSED_DIR, "mimic_master_features.csv"),
        low_memory=False,
    )
    log.info("   Master features : %d rows", len(master))

    # Merge ML mortality predictions
    ml_path = os.path.join(PROCESSED_DIR, "ml_predictions.csv")
    if os.path.exists(ml_path):
        ml = pd.read_csv(ml_path)[["PRED_PROB_DIED", "PRED_DIED"]].reset_index(drop=True)
        # ml_predictions has same row order as test split only — use outer merge via HADM_ID
        ml_full = pd.read_csv(ml_path, usecols=lambda c: c != "DIED")
        if "HADM_ID" in ml_full.columns:
            master = master.merge(
                ml_full[["HADM_ID", "PRED_PROB_DIED", "PRED_DIED"]],
                on="HADM_ID", how="left",
            )
        log.info("   ML predictions  : merged")

    # Merge NLP severity
    nlp_path = os.path.join(PROCESSED_DIR, "nlp_predictions.csv")
    if os.path.exists(nlp_path):
        nlp = pd.read_csv(nlp_path)
        sev_map = {"stable": 0, "moderate": 1, "critical": 2}
        nlp["SEV_ID"] = nlp["PRED_LABEL"].map(sev_map)
        nlp_agg = (
            nlp.groupby("HADM_ID")
               .agg(
                   NLP_MAX_SEVERITY  = ("SEV_ID",
                       lambda x: ["stable", "moderate", "critical"][int(x.max())]
                       if x.notna().any() else "stable"),
                   NLP_CRITICAL_FRAC = ("PRED_LABEL",
                       lambda x: (x == "critical").mean()),
                   N_NOTES           = ("PRED_LABEL", "count"),
               )
               .reset_index()
        )
        master = master.merge(nlp_agg, on="HADM_ID", how="left")
        log.info("   NLP predictions : merged")

    # Merge organ volume predictions
    vol_path = os.path.join(PROCESSED_DIR, "organ_volume_predictions.csv")
    if os.path.exists(vol_path):
        vol = pd.read_csv(vol_path)
        vol_cols = ["HADM_ID"] + [c for c in vol.columns
                                  if "pred_vol_ml" in c or "vol_change_pred" in c]
        master = master.merge(vol[vol_cols], on="HADM_ID", how="left")
        log.info("   Volume preds    : merged")

    log.info("   Final merged shape: %d rows × %d cols", *master.shape)
    return master


# ─────────────────────────────────────────────────────────────────────
# Stage A — Build structured input strings
# ─────────────────────────────────────────────────────────────────────

def build_input_text(row: pd.Series) -> str:
    """
    Converts a patient admission row into a structured clinical summary string.
    This is the INPUT to the summarization model.
    Format mirrors clinical trial case report forms (CRFs).
    """
    def fmt(val, decimals=1, suffix=""):
        if pd.isna(val):
            return "N/A"
        return f"{round(float(val), decimals)}{suffix}"

    # Mortality risk label
    prob = row.get("PRED_PROB_DIED", np.nan)
    if pd.isna(prob):
        risk_label = "unknown"
    elif prob >= 0.70:
        risk_label = "high"
    elif prob >= 0.40:
        risk_label = "moderate"
    else:
        risk_label = "low"

    parts = [
        f"Patient: {fmt(row.get('AGE'), 0)}yo {row.get('GENDER', 'unknown')}.",
        f"Admission: {row.get('ADMISSION_TYPE', 'N/A')} | LOS: {fmt(row.get('LOS_DAYS'), 1)} days.",
        f"Primary diagnosis: {row.get('DIAGNOSIS', 'N/A')}.",
        f"ICD-9 category: {row.get('ICD9_CATEGORY', 'N/A')} | Comorbidities: {fmt(row.get('N_DIAGNOSES', 0), 0)}.",
        f"Mortality risk: {risk_label} ({fmt(prob, 2, ' probability')}).",
        f"NLP note severity: {row.get('NLP_MAX_SEVERITY', 'N/A')} "
        f"(critical fraction: {fmt(row.get('NLP_CRITICAL_FRAC', 0), 2)}).",
        f"Labs — ALT: {fmt(row.get('alt_ul'))} U/L | "
        f"Creatinine: {fmt(row.get('creatinine_mgdl'), 2)} mg/dL | "
        f"Hemoglobin: {fmt(row.get('hemoglobin_gdl'), 1)} g/dL.",
        f"Organ volumes — "
        f"Liver: {fmt(row.get('liver_pred_vol_ml', np.nan), 0)} mL "
        f"({fmt(row.get('liver_vol_change_pred_pct', np.nan), 1, '%')} change) | "
        f"Kidney: {fmt(row.get('kidney_pred_vol_ml', np.nan), 0)} mL | "
        f"Spleen: {fmt(row.get('spleen_pred_vol_ml', np.nan), 0)} mL.",
        f"Outcome: {'Died' if row.get('DIED', 0) == 1 else 'Survived'}.",
    ]
    return " ".join(parts)


def build_target_summary(row: pd.Series) -> str:
    """
    Generates a template-based reference summary — the TARGET for fine-tuning.
    These are programmatically generated but clinically structured,
    similar to what a trial coordinator would write in a progress note.
    """
    age    = int(row.get("AGE", 0)) if not pd.isna(row.get("AGE")) else "unknown-age"
    gender = "male" if str(row.get("GENDER", "")).upper() == "M" else "female"
    dx     = str(row.get("DIAGNOSIS", "unspecified diagnosis")).lower()
    los    = row.get("LOS_DAYS", np.nan)
    died   = row.get("DIED", 0) == 1

    prob = row.get("PRED_PROB_DIED", np.nan)
    risk = "high" if not pd.isna(prob) and prob >= 0.70 else \
           "moderate" if not pd.isna(prob) and prob >= 0.40 else "low"

    nlp_sev  = str(row.get("NLP_MAX_SEVERITY", "stable"))
    crit_frac = row.get("NLP_CRITICAL_FRAC", 0)

    # Liver assessment
    liver_change = row.get("liver_vol_change_pred_pct", np.nan)
    if pd.isna(liver_change):
        liver_note = "Liver volume assessment unavailable."
    elif liver_change > 15:
        liver_note = f"Hepatomegaly indicated ({liver_change:+.1f}% predicted volume increase)."
    elif liver_change < -10:
        liver_note = f"Liver atrophy suggested ({liver_change:+.1f}% predicted volume decrease)."
    else:
        liver_note = "Liver volume within normal range."

    # Kidney assessment
    kidney_change = row.get("kidney_vol_change_pred_pct", np.nan)
    if pd.isna(kidney_change):
        kidney_note = "Renal volume assessment unavailable."
    elif kidney_change < -10:
        kidney_note = f"Renal atrophy consistent with chronic disease ({kidney_change:+.1f}%)."
    elif kidney_change > 10:
        kidney_note = f"Renal enlargement suggesting acute process ({kidney_change:+.1f}%)."
    else:
        kidney_note = "Kidney volume within expected range."

    outcome = "did not survive the admission" if died else \
              f"was discharged after {los:.1f} days" if not pd.isna(los) else "was discharged"

    summary = (
        f"A {age}-year-old {gender} admitted with {dx}. "
        f"Clinical note analysis classified the admission as {nlp_sev} severity "
        f"({'high' if crit_frac > 0.3 else 'low'} proportion of critical-flagged notes). "
        f"Mortality risk model predicted {risk} risk (probability {prob:.2f}). "
        f"{liver_note} {kidney_note} "
        f"The patient {outcome}."
    )
    return summary


# ─────────────────────────────────────────────────────────────────────
# Stage A — Prepare fine-tuning dataset
# ─────────────────────────────────────────────────────────────────────

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    log.info("[2/5] Building training pairs...")

    # Use admissions that have predictions from multiple phases
    has_ml  = df["PRED_PROB_DIED"].notna() if "PRED_PROB_DIED" in df.columns \
              else pd.Series(False, index=df.index)
    has_nlp = df["NLP_MAX_SEVERITY"].notna() if "NLP_MAX_SEVERITY" in df.columns \
              else pd.Series(False, index=df.index)
    has_vol = df["liver_pred_vol_ml"].notna() if "liver_pred_vol_ml" in df.columns \
              else pd.Series(False, index=df.index)

    # Prefer rows with all three, fall back to any subset
    rich_mask = has_ml & has_nlp & has_vol
    if rich_mask.sum() >= 500:
        sample_df = df[rich_mask].copy()
    else:
        sample_df = df.copy()
        log.warning("   Few fully-merged rows — using all %d admissions", len(sample_df))

    sample_df = sample_df.sample(
        min(len(sample_df), 5000), random_state=RANDOM_STATE
    ).reset_index(drop=True)

    sample_df["input_text"]  = sample_df.apply(build_input_text, axis=1)
    sample_df["target_text"] = sample_df.apply(build_target_summary, axis=1)

    # Save training pairs
    pairs = sample_df[["HADM_ID", "input_text", "target_text"]].copy()
    out   = os.path.join(PROCESSED_DIR, "training_pairs.csv")
    pairs.to_csv(out, index=False)

    log.info("   Training pairs : %d", len(pairs))
    log.info("   Saved → %s", out)

    # Show a sample
    sample = pairs.iloc[0]
    log.info("\n── Sample input ────────────────────────────────────────────")
    log.info("%s", sample["input_text"])
    log.info("── Sample target ───────────────────────────────────────────")
    log.info("%s", sample["target_text"])
    return pairs


# ─────────────────────────────────────────────────────────────────────
# Stage B — Dataset class for fine-tuning
# ─────────────────────────────────────────────────────────────────────

class ClinicalSummaryDataset(Dataset):
    """Lazy tokenization — same memory-safe pattern as Phase 2."""

    def __init__(self, inputs: list, targets: list, tokenizer):
        self.inputs    = inputs
        self.targets   = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        model_inputs = self.tokenizer(
            self.inputs[idx],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            self.targets[idx],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        model_inputs["labels"] = labels["input_ids"].squeeze(0)
        return {k: v.squeeze(0) for k, v in model_inputs.items()}


# ─────────────────────────────────────────────────────────────────────
# Stage B — Fine-tune BART
# ─────────────────────────────────────────────────────────────────────

def fine_tune(pairs: pd.DataFrame):
    """
    Fine-tunes facebook/bart-large-cnn on clinical (input, summary) pairs.
    Skipped automatically if no GPU is available — falls back to zero-shot.

    GTX 1650 (3.9GB VRAM) model choice:
      facebook/bart-base  (558MB) — OOMs even at batch=1 due to seq2seq overhead
      t5-small            ( 60MB) — fits comfortably, fast, good summarization quality
      t5-base             (250MB) — fits with gradient checkpointing + batch=1

    Set HF_SUMM_MODEL=facebook/bart-large-cnn in .env for larger GPUs (>=8GB).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = HF_SUMM_MODEL
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 5:
            # t5-small fits easily on GTX 1650 — 60MB weights
            model_name = "t5-small"
            log.info("   GTX 1650 (%.1fGB VRAM) — using t5-small (60MB)", vram_gb)
        elif vram_gb < 8:
            model_name = "t5-base"
            log.info("   GPU (%.1fGB VRAM) — using t5-base", vram_gb)
    else:
        log.warning("   No GPU — skipping fine-tuning, using template reports")
        return None, None

    log.info("[3/5] Fine-tuning %s on %d pairs...", model_name, len(pairs))

    # T5 requires "summarize: " prefix on inputs
    is_t5 = model_name.startswith("t5")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Gradient checkpointing: trades ~30% compute for ~40% VRAM saving
    model.gradient_checkpointing_enable()

    # 80/20 train/val split
    from sklearn.model_selection import train_test_split
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=0.20, random_state=RANDOM_STATE
    )

    # Add T5 task prefix if needed
    def add_prefix(texts):
        if is_t5:
            return ["summarize: " + t for t in texts]
        return texts

    train_ds = ClinicalSummaryDataset(
        add_prefix(train_pairs["input_text"].tolist()),
        train_pairs["target_text"].tolist(),
        tokenizer,
    )
    val_ds = ClinicalSummaryDataset(
        add_prefix(val_pairs["input_text"].tolist()),
        val_pairs["target_text"].tolist(),
        tokenizer,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir                   = SAVE_DIR,
        num_train_epochs             = 3,
        per_device_train_batch_size  = 2,        # safe for GTX 1650 3.9GB
        per_device_eval_batch_size   = 4,
        gradient_accumulation_steps  = 16,       # effective batch = 32
        learning_rate                = 5e-5,
        warmup_ratio                 = 0.1,
        weight_decay                 = 0.01,
        eval_strategy                = "epoch",
        save_strategy                = "epoch",
        load_best_model_at_end       = True,
        metric_for_best_model        = "eval_loss",
        greater_is_better            = False,
        predict_with_generate        = True,
        generation_max_length        = MAX_TARGET_LEN,
        fp16                         = torch.cuda.is_available(),
        gradient_checkpointing       = True,     # key memory saving for GTX 1650
        logging_steps                = 50,
        report_to                    = "none",
        seed                         = RANDOM_STATE,
        dataloader_num_workers       = 0,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collator,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    log.info("   Fine-tuned model saved → %s", SAVE_DIR)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# Stage C — Generate reports (fine-tuned OR zero-shot)
# ─────────────────────────────────────────────────────────────────────

def generate_reports(df: pd.DataFrame, model=None, tokenizer=None):
    """
    Generates a clinical insights report for every admission.
    If fine-tuned model is available, uses it.
    Otherwise falls back to the template-based target (zero-shot proxy).
    """
    log.info("[4/5] Generating clinical reports...")

    if model is not None and tokenizer is not None:
        log.info("   Using fine-tuned model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Detect if model is T5 (needs "summarize: " prefix)
        is_t5 = model.config.model_type.startswith("t5")

        reports = []
        batch_size = 8    # conservative for GTX 1650 during inference
        inputs = df.apply(build_input_text, axis=1).tolist()
        if is_t5:
            inputs = ["summarize: " + t for t in inputs]

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            enc   = tokenizer(
                batch,
                max_length=MAX_INPUT_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_TARGET_LEN,
                    num_beams=4,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            reports.extend(decoded)

            if (i // batch_size) % 10 == 0:
                log.info("   Generated %d / %d reports", min(i+batch_size, len(inputs)), len(inputs))
    else:
        log.info("   Using template-based reports (no fine-tuned model)")
        reports = df.apply(build_target_summary, axis=1).tolist()

    df = df.copy()
    df["clinical_report"] = reports

    # Save all reports
    out_cols = ["SUBJECT_ID", "HADM_ID", "DIAGNOSIS", "DIED", "clinical_report"]
    out_cols = [c for c in out_cols if c in df.columns]
    report_csv = os.path.join(OUTPUT_DIR, "clinical_reports.csv")
    df[out_cols].to_csv(report_csv, index=False)
    log.info("   Reports saved → %s", report_csv)

    # Save top-10 interesting cases as individual txt files
    save_sample_reports(df)
    return df


def save_sample_reports(df: pd.DataFrame):
    """Saves txt reports for high-risk, long-LOS, and critical-note cases."""
    cases = {}

    # Top 5 highest mortality risk
    if "PRED_PROB_DIED" in df.columns:
        cases["high_mortality_risk"] = (
            df.nlargest(5, "PRED_PROB_DIED")[["HADM_ID", "DIAGNOSIS",
                                               "PRED_PROB_DIED", "clinical_report"]]
        )

    # Top 5 longest LOS (trial burden)
    if "LOS_DAYS" in df.columns:
        cases["longest_los"] = (
            df.nlargest(5, "LOS_DAYS")[["HADM_ID", "DIAGNOSIS",
                                         "LOS_DAYS", "clinical_report"]]
        )

    for case_type, case_df in cases.items():
        path = os.path.join(REPORT_DIR, f"{case_type}.txt")
        with open(path, "w") as f:
            f.write(f"ClinicalAI — {case_type.replace('_', ' ').title()} Cases\n")
            f.write("=" * 60 + "\n\n")
            for _, row in case_df.iterrows():
                f.write(f"HADM_ID  : {row['HADM_ID']}\n")
                f.write(f"Diagnosis: {row.get('DIAGNOSIS', 'N/A')}\n")
                f.write(f"Report   :\n{row['clinical_report']}\n")
                f.write("-" * 60 + "\n\n")
        log.info("   Sample report saved → %s", path)


# ─────────────────────────────────────────────────────────────────────
# Stage C — Evaluation (ROUGE scores)
# ─────────────────────────────────────────────────────────────────────

def evaluate_summaries(pairs: pd.DataFrame, generated: list):
    log.info("[5/5] Evaluating generated summaries (ROUGE)...")
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        r1_list, r2_list, rl_list = [], [], []
        for ref, hyp in zip(pairs["target_text"], generated[:len(pairs)]):
            scores = scorer.score(ref, hyp)
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)

        log.info("   ROUGE-1 : %.3f", np.mean(r1_list))
        log.info("   ROUGE-2 : %.3f", np.mean(r2_list))
        log.info("   ROUGE-L : %.3f", np.mean(rl_list))

    except ImportError:
        log.warning("   rouge-score not installed — skipping ROUGE eval.")
        log.warning("   Install with: pip install rouge-score")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run():
    log.info("══ ClinicalAI — Phase 5: Clinical Insights LLM ══")

    df    = load_all_outputs()
    pairs = prepare_training_data(df)

    # Fine-tune if GPU available, else skip
    model, tokenizer = fine_tune(pairs)

    # Generate reports for all admissions
    df_with_reports = generate_reports(df, model, tokenizer)

    # Evaluate on training pairs (template targets as reference)
    if model is not None:
        sample_generated = df_with_reports["clinical_report"].tolist()
        evaluate_summaries(pairs, sample_generated)

    log.info("══ Phase 5 complete ══")
    log.info("   clinical_reports.csv → Phase 6 dashboard")
    return df_with_reports


if __name__ == "__main__":
    run()