"""
Configuration for Approach 2 (Pipeline: Masker + Filler) Evaluation
====================================================================
5 models from HuggingFace:
  Maskers (Token Classification / NER):
    - Xyren2005/pii-ner-distilroberta      (DistilRoBERTa)
    - Xyren2005/pii-ner-roberta            (RoBERTa-base)
    - Xyren2005/pii-ner-encoder_deberta    (DeBERTa-v3-base)

  Fillers:
    - Xyren2005/pii-ner-filler_bart-base       (BART-base, Seq2Seq)
    - Xyren2005/pii-ner-filler_deberta-filler  (DeBERTa-v3, MLM)

6 evaluated pipelines = 3 maskers × 2 fillers
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Data paths ───────────────────────────────────────────────────────────────
# Reuse the same test split and eval examples from Seq2Seq
SEQ2SEQ_DIR      = os.path.join(BASE_DIR, "..", "Seq2Seq_model")
TEST_DATA_PATH   = os.path.join(SEQ2SEQ_DIR, "data_splits", "test.jsonl")
EVAL_EXAMPLES_PATH = os.path.join(SEQ2SEQ_DIR, "eval_examples.jsonl")

RESULTS_DIR      = os.path.join(BASE_DIR, "results")
RESULTS_JSON     = os.path.join(RESULTS_DIR, "evaluation_results.json")
RESULTS_TXT      = os.path.join(RESULTS_DIR, "evaluation_results_readable.txt")

# ─── Masker configs ────────────────────────────────────────────────────────────
MASKER_CONFIGS = {
    "distilroberta": {
        "model_id":    "Xyren2005/pii-ner-distilroberta",
        "base_model":  "distilroberta-base",
        "architecture": "RobertaForTokenClassification",
        "params_M":    82,
        "hf_f1":       0.9117,
        "description": "DistilRoBERTa-base fine-tuned for PII NER (6-layer)",
    },
    "roberta": {
        "model_id":    "Xyren2005/pii-ner-roberta",
        "base_model":  "roberta-base",
        "architecture": "RobertaForTokenClassification",
        "params_M":    125,
        "hf_f1":       None,
        "description": "RoBERTa-base fine-tuned for PII NER (12-layer)",
    },
    "deberta": {
        "model_id":    "Xyren2005/pii-ner-encoder_deberta",
        "base_model":  "microsoft/deberta-v3-base",
        "architecture": "DebertaV2ForTokenClassification",
        "params_M":    183,
        "hf_f1":       0.9372,
        "description": "DeBERTa-v3-base fine-tuned for PII NER (12-layer)",
    },
}

# ─── Filler configs ────────────────────────────────────────────────────────────
FILLER_CONFIGS = {
    "bart-base": {
        "model_id":    "Xyren2005/pii-ner-filler_bart-base",
        "base_model":  "facebook/bart-base",
        "architecture": "BartForConditionalGeneration",
        "filler_type": "seq2seq",
        "params_M":    139,
        "description": "BART-base Seq2Seq filler (encoder-decoder)",
        # Exact prefix used during training (from pipeline_maskfill/src/common.py)
        "prompt_prefix": "Replace PII placeholders with realistic fake entities: ",
    },
    "deberta-mlm": {
        "model_id":    "Xyren2005/pii-ner-filler_deberta-filler",
        "base_model":  "microsoft/deberta-v3-base",
        "architecture": "DebertaV2ForMaskedLM",
        "filler_type": "mlm",
        "params_M":    183,
        "description": "DeBERTa-v3 MLM filler (encoder-only, masked language model)",
        "prompt_prefix": "",
    },
}

# ─── All 6 combinations to evaluate ───────────────────────────────────────────
# Each entry: (masker_key, filler_key)
COMBOS = [
    ("distilroberta", "bart-base"),
    ("distilroberta", "deberta-mlm"),
    ("roberta",       "bart-base"),
    ("roberta",       "deberta-mlm"),
    ("deberta",       "bart-base"),
    ("deberta",       "deberta-mlm"),
]

# ─── Inference settings ────────────────────────────────────────────────────────
MAX_INPUT_LENGTH  = 256     # max tokens for masker / filler input
MAX_TARGET_LENGTH = 256     # max tokens for seq2seq generation
NUM_BEAMS         = 4       # beam search width for BART filler

# Batch size for masker and filler inference.
# Masker: batched NER forward pass (fast, linear speed-up).
# Filler (seq2seq): batched generate — significant speed-up over batch=1.
# Filler (MLM): batched forward pass through the encoder.
# Reduce if you get CUDA OOM; increase on a machine with more VRAM.
EVAL_BATCH_SIZE   = 4

# Limit test-set examples to keep runtime tractable (set None for full set)
TEST_SET_LIMIT    = 5000
