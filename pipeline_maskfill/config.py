# ==============================================================================
# config.py — Central configuration for the Two-Level PII Pipeline
# ==============================================================================
# All hyperparameters, model registries, entity mappings, and paths live here.
# Nothing is hardcoded elsewhere — change a model or param here, everything adapts.
# ==============================================================================

import os
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
DATA_CACHE_DIR = os.path.join(OUTPUT_DIR, "data_cache")

for d in (OUTPUT_DIR, LOG_DIR, EVAL_DIR, DATA_CACHE_DIR):
    os.makedirs(d, exist_ok=True)

# ── Seeds & Device ───────────────────────────────────────────────────────────
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_KAGGLE = os.path.exists("/kaggle/working")

# bf16 needs compute capability >= 8.0; T4 (7.5) / P100 (6.0) crash with bf16
BF16_OK = (torch.cuda.is_available()
           and torch.cuda.is_bf16_supported()
           and torch.cuda.get_device_capability()[0] >= 8)
FP16_OK = torch.cuda.is_available() and not BF16_OK

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ai4privacy/pii-masking-400k"  # ~500K samples on HuggingFace
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Quick mode — small subset for debugging
QUICK_N = 500

# ── 20 Native AI4Privacy Entity Types ────────────────────────────────────────
ENTITY_TYPES = [
    "GIVENNAME", "SURNAME", "TITLE", "GENDER", "SEX",
    "CITY", "STREET", "BUILDINGNUM", "ZIPCODE",
    "TELEPHONENUM", "EMAIL", "SOCIALNUM", "PASSPORTNUM",
    "DRIVERLICENSENUM", "IDCARDNUM", "TAXNUM", "CREDITCARDNUMBER",
    "DATE", "TIME", "AGE"
]

# We are using the native types directly, no mapping needed.
ENTITY_MAP = {t: t for t in ENTITY_TYPES}

# ── BIO Labels ───────────────────────────────────────────────────────────────
def build_bio_labels(entity_types):
    """Build BIO label scheme: O, B-PERSON, I-PERSON, B-LOC, I-LOC, ..."""
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = build_bio_labels(ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)

# ── Encoder (NER) Model Registry ─────────────────────────────────────────────
# Each encoder config contains everything needed to instantiate and train it.
ENCODER_REGISTRY = {
    "distilroberta": {
        "hf_name": "distilroberta-base",
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 5e-5,
        "epochs": 5,
        "max_length": 256,
        "grad_accum": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "needs_deberta_fix": False,
    },
    "roberta": {
        "hf_name": "roberta-base",
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "epochs": 5,
        "max_length": 256,
        "grad_accum": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "needs_deberta_fix": False,
    },
    "deberta": {
        "hf_name": "microsoft/deberta-v3-base",
        "batch_size": 8,
        "eval_batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 5,
        "max_length": 256,
        "grad_accum": 4,  # effective batch = 32
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "needs_deberta_fix": True,
    },
}

# ── Filler (Encoder-Decoder) Model Registry ──────────────────────────────────
FILLER_REGISTRY = {
    "bart-base": {
        "hf_name": "facebook/bart-base",
        "batch_size": 4,
        "eval_batch_size": 8,
        "learning_rate": 3e-5,
        "epochs": 3,
        "max_input_length": 256,
        "max_target_length": 256,
        "grad_accum": 4,   # effective batch = 16
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "use_qlora": False,  # 139M — fits full fine-tune on 16GB
        "prefix": "",        # BART doesn't use prefix in input
        "gen_max_tokens": 256,
        "gen_num_beams": 4,
    },
    "flan-t5": {
        "hf_name": "google/flan-t5-base",
        "batch_size": 4,
        "eval_batch_size": 8,
        "learning_rate": 3e-5,
        "epochs": 3,
        "max_input_length": 256,
        "max_target_length": 256,
        "grad_accum": 4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "use_qlora": True,   # 248M — needs QLoRA on consumer GPUs
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_targets": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
        "prefix": "Replace all personally identifiable information with realistic fake alternatives: ",
        "gen_max_tokens": 256,
        "gen_num_beams": 4,
    },
}

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_EVERY_N_STEPS = 50           # log training loss every N steps
SAMPLE_INFERENCE_COUNT = 5       # number of sample inferences after each epoch
EVAL_SAMPLES_FOR_DISPLAY = 20   # number of sample outputs to save during evaluation
