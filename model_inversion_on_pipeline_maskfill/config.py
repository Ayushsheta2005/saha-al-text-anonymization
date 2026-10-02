"""
config.py — settings for pipeline mask-fill model-inversion dataset generation.
"""
import os

# ── HuggingFace ──────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN    = os.environ.get("HF_TOKEN", os.getenv("HF_TOKEN"))
HF_USER     = "JALAPENO11"

# Output dataset repos (will be created if they don't exist)
HF_REPO_COMBO1 = f"{HF_USER}/pipeline-inversion-roberta-deberta"
HF_REPO_COMBO2 = f"{HF_USER}/pipeline-inversion-roberta-bart"

# ── Source dataset (local JSONL) ────────────────────────────────────────
# Each line: {"id": ..., "original_text": "...", ...}
SOURCE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_creation", "data", "english_entries_all.jsonl",
)
TARGET_N = 40_000      # how many sentences to take

# ── Model IDs ────────────────────────────────────────────────────────────
NER_MODEL_ID     = "Xyren2005/pii-ner-roberta"          # shared NER encoder

# Combo 1 — encoder + encoder (MLM filler)
MLM_FILLER_ID    = "Xyren2005/pii-ner-filler_deberta-filler"

# Combo 2 — encoder + decoder (seq2seq filler)
SEQ2SEQ_FILLER_ID = "Xyren2005/pii-ner-filler_bart-base"

# ── Inference knobs ───────────────────────────────────────────────────────
NER_MAX_LEN      = 256
MLM_MAX_LEN      = 256
S2S_MAX_LEN      = 256
GEN_MAX_TOKENS   = 80         # p95 output is ~41 words; 80 subword tokens is plenty
GEN_NUM_BEAMS    = 1          # greedy — 4× faster than beam=4, good enough for inversion data
NER_BATCH        = 64         # sentences per NER forward pass
MLM_BATCH        = 64         # sentences per MLM forward pass
S2S_BATCH        = 8          # BART KV-cache is large; keep small to avoid OOM on 4 GB GPU

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

COMBO1_FILE = os.path.join(OUTPUT_DIR, "combo1_roberta_deberta.jsonl")
COMBO2_FILE = os.path.join(OUTPUT_DIR, "combo2_roberta_bart.jsonl")

# ── MLM mask token (DeBERTa/all BERT-family models) ─────────────────────────
MLM_MASK_TOKEN = "[MASK]"

# ── BIO label set (must match training config of pii-ner-roberta) ─────────
ENTITY_TYPES = [
    "PERSON", "LOC", "ORG", "DATE", "PHONE", "EMAIL", "SSN",
    "CREDIT_CARD", "ADDRESS", "IP_ADDRESS", "IBAN", "PASSPORT",
    "DRIVER_LICENSE", "USERNAME", "URL", "MEDICAL", "ACCOUNT",
    "BUILDING", "POSTCODE",
]

def _build_bio(entity_types):
    labels = ["O"]
    for e in entity_types:
        labels += [f"B-{e}", f"I-{e}"]
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = _build_bio(ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)
