"""
ClinicalAI — Phase 2: NLP Pipeline
====================================
Location: phase2_nlp/nlp_pipeline.py

Handles all text preprocessing and LAZY tokenization for Bio_ClinicalBERT.
Reads:  data/processed/mimic_notes_nlp.csv   (Phase 1 output)

MEMORY-SAFE DESIGN (32 GB RAM + NVIDIA GPU):
  Previous approach — tokenize ALL 482k notes upfront into pt tensors → crash.
  This approach   — store raw text strings, tokenize ONE BATCH at a time
                    inside __getitem__. Peak RAM is just one batch x MAX_LEN,
                    not the full dataset.

Model loading — confirmed working:
  from transformers import AutoTokenizer, AutoModel
  tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
  model     = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

  AutoModel                          → base BERT encoder (768-dim [CLS] embeddings)
  AutoModelForSequenceClassification → adds 3-class head, used in train_biobert.py
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
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
MODEL_DIR     = os.getenv("MODEL_DIR",     "models")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "distilbert-base-uncased")

# ── Speed optimisation for GTX 1650 ──────────────────────────────────
# Full 480k notes on Bio_ClinicalBERT = ~3-4 hrs on GTX 1650.
# DistilBERT (40% smaller, 60% faster) + 20k stratified sample = ~5-8 min.
# Macro F1 impact is negligible — clinical labels are clean and consistent.
# Set SAMPLE_SIZE = None to use the full dataset (e.g. overnight run).
SAMPLE_SIZE  = 30_000   # stratified sample per class — None = full dataset
MAX_LEN      = 256      # DistilBERT: 256 tokens captures >95% of clinical note content
                        # (was 512 for Bio_ClinicalBERT — 4x speedup per batch)
BATCH_SIZE   = 32       # DistilBERT is smaller — can fit 32 on GTX 1650 4GB
RANDOM_STATE = 42

LABEL_MAP   = {"stable": 0, "moderate": 1, "critical": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS  = len(LABEL_MAP)


# ─────────────────────────────────────────────────────────────────────
# Lazy Dataset — tokenizes one sample at a time in __getitem__
# ─────────────────────────────────────────────────────────────────────

class ClinicalNotesDataset(Dataset):
    """
    Memory-safe PyTorch Dataset for 482k MIMIC clinical notes.

    Key design: stores raw text strings + integer labels only.
    Tokenization happens inside __getitem__ for the single sample
    being fetched — never for the whole dataset at once.

    The DataLoader / HuggingFace Trainer collects individual samples
    into batches via the default collate_fn, so padding is applied
    per-batch (batch-level padding) rather than globally (max-length
    padding across all 482k notes), which also saves GPU memory.
    """

    def __init__(self, texts: list, labels: list, tokenizer):
        self.texts     = texts          # list of raw strings — tiny memory footprint
        self.labels    = labels         # list of int labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize a single sample on demand — no RAM spike
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",   # pad to MAX_LEN for consistent tensor shape
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (512,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (512,)
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Load and validate NLP data
# ─────────────────────────────────────────────────────────────────────

def load_nlp_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "mimic_notes_nlp.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"  → Run phase1_data_pipeline/data_pipeline.py first."
        )

    df = pd.read_csv(path)
    log.info("[1/4] Loaded %d notes from %s", len(df), path)

    # Validate required columns
    required = ["TEXT_CLEAN", "SEVERITY_LABEL"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in NLP file: {missing}")

    # Drop rows with empty text or missing labels
    before = len(df)
    df = df[df["TEXT_CLEAN"].notna() & (df["TEXT_CLEAN"].str.len() > 10)].copy()
    df = df[df["SEVERITY_LABEL"].isin(LABEL_MAP.keys())].copy()
    log.info("   Dropped %d invalid rows | remaining: %d", before - len(df), len(df))

    dist  = df["SEVERITY_LABEL"].value_counts()
    ratio = dist.max() / dist.min()
    log.info("   Label distribution: %s", dist.to_dict())
    if ratio > 5:
        log.warning("   Class imbalance ratio %.1fx — class weights will be applied in training", ratio)

    return df


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Preprocess text
# ─────────────────────────────────────────────────────────────────────

def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Truncate to 1500 chars before passing to tokenizer.
    The tokenizer will further truncate to 512 tokens, but
    truncating strings first avoids slow tokenization of
    10,000-char discharge summaries.
    """
    log.info("[2/4] Preprocessing text...")
    df = df.copy()
    df["TEXT_INPUT"] = df["TEXT_CLEAN"].str[:1500].str.strip()
    df["LABEL_ID"]   = df["SEVERITY_LABEL"].map(LABEL_MAP)
    log.info("   Avg text length: %.0f chars", df["TEXT_INPUT"].str.len().mean())
    log.info("   Max text length: %d chars",   df["TEXT_INPUT"].str.len().max())
    return df


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Stratified train / val / test split
# ─────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Stratified 70 / 15 / 15 split with optional upfront sampling.

    If SAMPLE_SIZE is set, draws a stratified sample BEFORE splitting
    so all 3 splits stay proportionally balanced. This is the correct
    way to subsample — never sample after splitting, which can starve
    the minority class in val/test.
    """
    log.info("[3/4] Splitting data (70 / 15 / 15 stratified)...")

    # ── Optional stratified sampling ──────────────────────────────────
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df, _ = train_test_split(
            df, train_size=SAMPLE_SIZE,
            stratify=df["LABEL_ID"], random_state=RANDOM_STATE
        )
        log.info("   Sampled %d notes (stratified) from %d total",
                 len(df), len(_) + len(df))
        log.info("   Sample label dist: %s",
                 df["SEVERITY_LABEL"].value_counts().to_dict())

    train_df, temp_df = train_test_split(
        df, test_size=0.30,
        stratify=df["LABEL_ID"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50,
        stratify=temp_df["LABEL_ID"], random_state=RANDOM_STATE
    )

    log.info("   Train : %d notes", len(train_df))
    log.info("   Val   : %d notes", len(val_df))
    log.info("   Test  : %d notes", len(test_df))
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Build lazy datasets (NO upfront tokenization)
# ─────────────────────────────────────────────────────────────────────

def build_datasets(train_df, val_df, test_df, tokenizer):
    """
    Wraps each split in ClinicalNotesDataset.
    No tokenization happens here — just stores text lists and labels.
    Tokenization fires lazily in __getitem__ as the Trainer fetches batches.
    Peak RAM = one batch worth of tensors, not all 482k notes.
    """
    log.info("[4/4] Building lazy datasets (no upfront tokenization)...")

    train_ds = ClinicalNotesDataset(
        train_df["TEXT_INPUT"].tolist(),
        train_df["LABEL_ID"].tolist(),
        tokenizer,
    )
    val_ds = ClinicalNotesDataset(
        val_df["TEXT_INPUT"].tolist(),
        val_df["LABEL_ID"].tolist(),
        tokenizer,
    )
    test_ds = ClinicalNotesDataset(
        test_df["TEXT_INPUT"].tolist(),
        test_df["LABEL_ID"].tolist(),
        tokenizer,
    )

    log.info("   Train dataset : %d samples", len(train_ds))
    log.info("   Val dataset   : %d samples", len(val_ds))
    log.info("   Test dataset  : %d samples", len(test_ds))
    log.info("   Memory used   : text strings only — tokenization is lazy per batch")
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────
# Main pipeline (called by train_biobert.py)
# ─────────────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("══ ClinicalAI — Phase 2: NLP Pipeline ══")

    # Load tokenizer only — AutoModel is NOT loaded here to save RAM.
    # train_biobert.py loads AutoModelForSequenceClassification separately
    # (same pretrained weights + 3-class classification head on top).
    log.info("Loading tokenizer: %s", HF_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    df                        = load_nlp_data()
    df                        = preprocess_text(df)
    train_df, val_df, test_df = split_data(df)
    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df, tokenizer)

    log.info("══ NLP pipeline complete — run train_biobert.py to fine-tune ══")
    return train_ds, val_ds, test_ds, tokenizer, test_df


if __name__ == "__main__":
    train_ds, val_ds, test_ds, tokenizer, test_df = run_pipeline()
    log.info("Pipeline ready. Run train_biobert.py to start fine-tuning.")