#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    TWO-LEVEL PII ANONYMIZATION PIPELINE — UNIFIED KAGGLE NOTEBOOK          ║
║                                                                            ║
║    Encoder (NER) → masks PII                                               ║
║    Filler (Seq2Seq or MLM) → fills with realistic replacements             ║
║                                                                            ║
║    All code in ONE file. Use the FLAGS below to control what runs.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage on Kaggle:
  1. Upload this file as a Kaggle notebook (or paste into a cell)
  2. Set the FLAGS section below
  3. Run the notebook

Training runs save checkpoints every epoch. If Kaggle disconnects, rerun
the same cell — it will resume from the last checkpoint automatically.

Models are pushed to HuggingFace Hub after training if PUSH_TO_HUB=True.
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 0: INSTALL DEPENDENCIES                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["datasets", "transformers", "accelerate", "peft", "bitsandbytes",
            "seqeval", "evaluate", "rouge-score", "sacrebleu", "bert-score"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: FLAGS — CHANGE THESE TO CONTROL WHAT RUNS                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── What to run ──────────────────────────────────────────────────────────────
TRAIN_ENCODER_DISTILROBERTA = True    # Train DistilRoBERTa NER
TRAIN_ENCODER_ROBERTA       = False   # Train RoBERTa NER
TRAIN_ENCODER_DEBERTA       = False   # Train DeBERTa-v3 NER
TRAIN_FILLER_BART           = True    # Train BART-BASE Seq2Seq filler
TRAIN_FILLER_DEBERTA        = False   # Train DeBERTa-v3 MLM filler
RUN_EVALUATION              = True    # Run full evaluation on test set

# ── Which models to evaluate (only used if RUN_EVALUATION=True) ──────────────
EVAL_ENCODER = "distilroberta"        # "distilroberta" | "roberta" | "deberta"
EVAL_FILLER  = "bart-base"            # "bart-base" | "deberta-filler"

# ── Quick mode (small data for debugging) ────────────────────────────────────
QUICK_MODE = False              # ← Set True only for local smoke-tests
QUICK_N    = 30

# ── HuggingFace Hub ──────────────────────────────────────────────────────────
PUSH_TO_HUB = True
HF_USERNAME = "Xyren2005"             # <-- CHANGE THIS
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN    = os.getenv("HF_TOKEN")  # <-- CHANGE THIS

# ── Weights & Biases ─────────────────────────────────────────────────────────
USE_WANDB     = True                  # Log training metrics to W&B
WANDB_PROJECT = "pii-anonymization"   # W&B project name
WANDB_ENTITY  = None                  # W&B team/username (None = default)
WANDB_API_KEY = "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"

import os
os.environ["HF_TOKEN"] = HF_TOKEN
from huggingface_hub import login
login(HF_TOKEN, add_to_git_credential=False)

import wandb
wandb.login(key=WANDB_API_KEY)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: CONFIG                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, gc, re, json, random, logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset

# ── Paths ────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    BASE_DIR = "/kaggle/working/pii_pipeline"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."

OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")
EVAL_DIR       = os.path.join(OUTPUT_DIR, "eval")
DATA_CACHE_DIR = os.path.join(OUTPUT_DIR, "data_cache")

for d in (OUTPUT_DIR, LOG_DIR, EVAL_DIR, DATA_CACHE_DIR):
    os.makedirs(d, exist_ok=True)

# ── Seeds & Device ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK = (torch.cuda.is_available()
           and torch.cuda.is_bf16_supported()
           and torch.cuda.get_device_capability()[0] >= 8)
FP16_OK = torch.cuda.is_available() and not BF16_OK

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "ai4privacy/open-pii-masking-500k-ai4privacy"
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10

# ── 21 Entity Types (aligned with Seq2Seq for fair comparison) ───────────────
ENTITY_TYPES = [
    "FULLNAME", "FIRST_NAME", "LAST_NAME", "ID_NUMBER", "PASSPORT", "SSN",
    "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "LOCATION", "ORGANIZATION",
    "ACCOUNT_NUM", "CREDIT_CARD", "ZIPCODE", "TITLE", "GENDER", "NUMBER",
    "OTHER_PII", "UNKNOWN",
]

# Map AI4Privacy's 20 fine-grained labels → our 21 coarse types
ENTITY_MAP = {
    "GIVENNAME": "FIRST_NAME", "SURNAME": "LAST_NAME",
    "TITLE": "TITLE", "GENDER": "GENDER", "SEX": "GENDER",
    "CITY": "LOCATION",
    "STREET": "ADDRESS", "BUILDINGNUM": "ADDRESS",
    "ZIPCODE": "ZIPCODE",
    "TELEPHONENUM": "PHONE",
    "EMAIL": "EMAIL",
    "SOCIALNUM": "SSN", "PASSPORTNUM": "PASSPORT",
    "DRIVERLICENSENUM": "ID_NUMBER", "IDCARDNUM": "ID_NUMBER", "TAXNUM": "ID_NUMBER",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "DATE": "DATE", "TIME": "TIME",
    "AGE": "NUMBER",
}

# ── BIO Labels ───────────────────────────────────────────────────────────────
def build_bio_labels(entity_types):
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = build_bio_labels(ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)

# ── Model Registries ─────────────────────────────────────────────────────────
ENCODER_REGISTRY = {
    "distilroberta": {
        # Small model (82M) — can afford larger batch; faster epochs
        "hf_name": "distilroberta-base",
        "batch_size": 32,          # ↑ from 16 — fits easily on T4
        "eval_batch_size": 64,     # ↑ from 32
        "learning_rate": 4e-5,     # ↑ slightly from 3e-5 — DistilRoBERTa handles it
        "epochs": 8,               # ↓ from 20 — early stopping will hit ~4-6 anyway
        "max_length": 192,         # ↓ from 256 — PII sentences are short; saves 30% memory
        "grad_accum": 1,           # ↓ from 2 — batch=32 already gives effective 32
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,      # ↑ from 0.06 — smoother warmup
        "needs_deberta_fix": False,
        "patience": 3,             # ↓ from 4 — proportional to 8 epochs
    },
    "roberta": {
        # Medium model (125M) — keep batch=16, bump eval throughput
        "hf_name": "roberta-base",
        "batch_size": 16,
        "eval_batch_size": 48,     # ↑ from 32 — faster validation
        "learning_rate": 2e-5,
        "epochs": 8,               # ↓ from 20
        "max_length": 192,         # ↓ from 256
        "grad_accum": 2,           # effective batch = 32
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,      # ↑ from 0.06
        "needs_deberta_fix": False,
        "patience": 3,             # ↓ from 4
    },
    "deberta": {
        # Large memory footprint — keep batch=8 but extend max_length (it helps DeBERTa)
        "hf_name": "microsoft/deberta-v3-base",
        "batch_size": 8,
        "eval_batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 8,               # ↓ from 20 — DeBERTa ~30min/epoch, 8 = 4hr max
        "max_length": 256,         # keep — DeBERTa benefits from full context
        "grad_accum": 4,           # effective batch = 32
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,      # ↑ from 0.06 — DeBERTa is more warmup-sensitive
        "needs_deberta_fix": True,
        "patience": 3,             # ↓ from 4
    },
}

FILLER_REGISTRY = {
    "bart-base": {
        "hf_name": "facebook/bart-base", "type": "seq2seq",
        "batch_size": 16,
        "eval_batch_size": 32,     # ↑ from 16 — BART eval is cheap
        "learning_rate": 3e-5,
        "epochs": 8,               # ↓ from 20
        "max_input_length": 192,   # ↑ from 128 — avoids truncating multi-span sentences
        "max_target_length": 192,  # ↑ from 128 — match input
        "grad_accum": 2,           # effective batch = 32
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "use_qlora": False, "prefix": "",
        "gen_max_tokens": 192,     # ↑ from 128 — match max_target_length
        "gen_num_beams": 4,        # keep — beam quality matters for realistic PII generation
        "patience": 3,             # ↓ from 4
    },
    "deberta-filler": {
        "hf_name": "microsoft/deberta-v3-base", "type": "mlm",
        "batch_size": 8,           # ↓ from 16 — DeBERTa at 256 tokens WILL OOM at batch=16 on T4
        "eval_batch_size": 16,     # ↓ from 32 — same OOM risk
        "learning_rate": 2e-5,     # ↓ from 5e-5 — DeBERTa is LR-sensitive; 5e-5 risks instability
        "epochs": 8,               # ↓ from 20
        "max_input_length": 256,
        "grad_accum": 4,           # ↑ from 2 — restores effective batch = 32
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "use_qlora": False, "needs_deberta_fix": True,
        "patience": 3,             # ↓ from 4
    },
}

LOG_EVERY_N_STEPS = 25
SAMPLE_INFERENCE_COUNT = 5
EVAL_SAMPLES_FOR_DISPLAY = 20

# ── Logging Setup ────────────────────────────────────────────────────────────
log = logging.getLogger("pipeline")
log.setLevel(logging.INFO)
if not log.handlers:
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"pipeline_{timestamp}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)
    log.info(f"Logging to: {log_file}")

log.info("=" * 70)
log.info("  TWO-LEVEL PII ANONYMIZATION PIPELINE")
log.info("=" * 70)
log.info(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"  GPU:    {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    n_gpu = torch.cuda.device_count()
    log.info(f"  GPUs:   {n_gpu}")
log.info(f"  BF16:   {BF16_OK},  FP16: {FP16_OK}")
log.info(f"  Kaggle: {IS_KAGGLE}")
log.info(f"  Quick:  {QUICK_MODE}")
log.info("=" * 70)

# ── W&B / Report config ──────────────────────────────────────────────────
if PUSH_TO_HUB:
    try:
        if IS_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        else:
            HF_TOKEN = os.environ.get("HF_TOKEN")
            
        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN)
            log.info("  HF Hub: Logged in successfully!")
        else:
            log.warning("  HF_TOKEN not found! Model push may fail with 401 Unauthorized.")
    except Exception as e:
        log.warning(f"  HF Hub Login failed: {e}")

if USE_WANDB:
    try:
        import wandb
        os.environ['WANDB_PROJECT'] = WANDB_PROJECT
        if WANDB_ENTITY: os.environ['WANDB_ENTITY'] = WANDB_ENTITY
        if WANDB_API_KEY:
            wandb.login(key=WANDB_API_KEY)
        REPORT_TO = ["wandb"]
        log.info(f"  W&B:    project={WANDB_PROJECT}, entity={WANDB_ENTITY}")
    except ImportError:
        log.warning("  W&B requested but wandb not installed. Falling back to no reporting.")
        REPORT_TO = ["none"]
else:
    REPORT_TO = ["none"]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: DATA LOADING & SPLITTING                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_ai4privacy() -> Dataset:
    log.info(f"Loading dataset: {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train")
    log.info(f"  Loaded {len(ds):,} examples")
    return ds

def language_stratified_split(ds: Dataset) -> Dict[str, Dataset]:
    log.info("Performing language-stratified split ...")
    df = ds.to_pandas()
    lang_col = None
    for col in ("language", "lang", "Language"):
        if col in ds.column_names:
            lang_col = col
            break
    if lang_col is None:
        lang_col = "__lang"
        df[lang_col] = "unknown"

    train_idx, val_enc_idx, val_fill_idx, test_idx = [], [], [], []
    for lang, grp in df.groupby(lang_col):
        idx = grp.index.tolist()
        random.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * TEST_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_val_enc = n_val // 2
        n_val_fill = n_val - n_val_enc
        test_idx.extend(idx[:n_test])
        val_enc_idx.extend(idx[n_test:n_test + n_val_enc])
        val_fill_idx.extend(idx[n_test + n_val_enc:n_test + n_val_enc + n_val_fill])
        train_idx.extend(idx[n_test + n_val:])
    for lst in (train_idx, val_enc_idx, val_fill_idx, test_idx):
        random.shuffle(lst)

    splits = {
        "train": ds.select(train_idx), "val_encoder": ds.select(val_enc_idx),
        "val_filler": ds.select(val_fill_idx), "test": ds.select(test_idx),
    }
    for name, split_ds in splits.items():
        log.info(f"  {name}: {len(split_ds):,}")
    return splits

def split_train_halves(train_ds: Dataset) -> Tuple[Dataset, Dataset]:
    n = len(train_ds)
    indices = list(range(n))
    random.shuffle(indices)
    mid = n // 2
    half_a, half_b = train_ds.select(indices[:mid]), train_ds.select(indices[mid:])
    log.info(f"  Half-A (NER): {len(half_a):,},  Half-B (Filler): {len(half_b):,}")
    return half_a, half_b

def quick_subsample(ds: Dataset, n: int = QUICK_N) -> Dataset:
    if len(ds) <= n: return ds
    return ds.select(random.sample(range(len(ds)), n))

def prepare_all_data():
    ds = load_ai4privacy()
    splits = language_stratified_split(ds)
    half_a, half_b = split_train_halves(splits["train"])
    if QUICK_MODE:
        log.info(f"  QUICK MODE: subsampling to {QUICK_N}")
        half_a = quick_subsample(half_a, QUICK_N)
        half_b = quick_subsample(half_b, QUICK_N)
        splits["val_encoder"] = quick_subsample(splits["val_encoder"], QUICK_N // 5)
        splits["val_filler"]  = quick_subsample(splits["val_filler"], QUICK_N // 5)
        splits["test"]        = quick_subsample(splits["test"], QUICK_N // 5)
    return {"half_a": half_a, "half_b": half_b,
            "val_encoder": splits["val_encoder"], "val_filler": splits["val_filler"],
            "test": splits["test"]}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: NER DATA PROCESSING (BIO alignment)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_tokens_and_labels(example: Dict) -> Tuple[List[str], List[str]]:
    if "tokens" in example and "ner_tags" in example:
        return example["tokens"], example["ner_tags"]
    text = example.get("source_text", example.get("text", ""))
    masks = example.get("privacy_mask", [])
    if not masks:
        tokens = text.split()
        return tokens, ["O"] * len(tokens)

    spans = sorted(masks, key=lambda m: m.get("start", m.get("offset", 0)))
    tokens, labels = [], []
    pos = 0
    for span in spans:
        start = span.get("start", span.get("offset", 0))
        end = span.get("end", start + span.get("length", len(span.get("value", ""))))
        label = span.get("label", span.get("entity_type", "O")).upper().replace(" ", "_")
        value = span.get("value", text[start:end])
        if pos < start:
            before = text[pos:start].split()
            tokens.extend(before); labels.extend(["O"] * len(before))
        entity_words = value.split()
        if entity_words:
            coarse = ENTITY_MAP.get(label, label)
            if coarse in ENTITY_TYPES:
                tokens.append(entity_words[0]); labels.append(f"B-{coarse}")
                for w in entity_words[1:]:
                    tokens.append(w); labels.append(f"I-{coarse}")
            else:
                tokens.extend(entity_words); labels.extend(["O"] * len(entity_words))
        pos = end
    if pos < len(text):
        remaining = text[pos:].split()
        tokens.extend(remaining); labels.extend(["O"] * len(remaining))
    return tokens, labels

def get_source_text(example: Dict) -> str:
    if "source_text" in example: return example["source_text"]
    if "text" in example: return example["text"]
    tokens, _ = extract_tokens_and_labels(example)
    return " ".join(tokens)

def tokenize_and_align_ner(examples, tokenizer, max_length=256):
    key = next(k for k in ("source_text", "text", "tokens") if k in examples)
    all_tokens, all_labels = [], []
    for i in range(len(examples[key])):
        ex = {k: v[i] for k, v in examples.items()}
        toks, labs = extract_tokens_and_labels(ex)
        all_tokens.append(toks); all_labels.append(labs)
    enc = tokenizer(all_tokens, truncation=True, max_length=max_length,
                    padding="max_length", is_split_into_words=True)
    aligned_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = enc.word_ids(batch_index=i)
        label_ids = []
        prev_wid = None
        for wid in word_ids:
            if wid is None: label_ids.append(-100)
            elif wid != prev_wid:
                lbl = labels[wid] if wid < len(labels) else "O"
                label_ids.append(LABEL2ID.get(lbl, 0))
            else:
                lbl = labels[wid] if wid < len(labels) else "O"
                if lbl.startswith("B-"): lbl = "I-" + lbl[2:]
                label_ids.append(LABEL2ID.get(lbl, 0))
            prev_wid = wid
        aligned_labels.append(label_ids)
    enc["labels"] = aligned_labels
    return enc

def create_filler_pair(example: Dict) -> Dict[str, str]:
    tokens, labels = extract_tokens_and_labels(example)
    masked_words, prev_entity = [], None
    for tok, lab in zip(tokens, labels):
        if lab == "O": masked_words.append(tok); prev_entity = None
        elif lab.startswith("B-"):
            etype = lab[2:]; masked_words.append(f"[{etype}]"); prev_entity = etype
        elif lab.startswith("I-") and prev_entity: pass
        else: masked_words.append(tok); prev_entity = None
    return {"input_text": f"Replace PII: {' '.join(masked_words)}",
            "target_text": " ".join(tokens)}

def tokenize_filler_pairs(examples, tokenizer, max_input_length=256, max_target_length=256):
    enc = tokenizer(examples["input_text"], max_length=max_input_length,
                    truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["target_text"], max_length=max_target_length,
                       truncation=True, padding="max_length")
    enc["labels"] = [[(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
                     for seq in labels["input_ids"]]
    return enc

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: ENCODER (NER) — Build, Train, Infer                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from transformers import (
    AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForTokenClassification, DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from seqeval.metrics import f1_score as seq_f1_score, classification_report as seq_classification_report

def fix_deberta_params(model):
    for name, param in model.named_parameters():
        if "LayerNorm" in name or "layernorm" in name:
            param.data = param.data.to(torch.float32)
    return model

def build_encoder(model_name: str):
    cfg = ENCODER_REGISTRY[model_name]
    hf_name = cfg["hf_name"]
    log.info(f"Building encoder: {model_name} ({hf_name})")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        hf_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        torch_dtype=torch.float32).float().to(DEVICE)
    if cfg.get("needs_deberta_fix"): fix_deberta_params(model)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  Parameters: {total:.1f}M")
    return model, tokenizer, cfg

def compute_ner_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        true_sent, pred_sent = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100: continue
            true_sent.append(ID2LABEL.get(int(l), "O"))
            pred_sent.append(ID2LABEL.get(int(p), "O"))
        true_labels.append(true_sent); pred_labels.append(pred_sent)
    return {"f1": seq_f1_score(true_labels, pred_labels, average="weighted")}

class NERSampleInferenceCallback(TrainerCallback):
    def __init__(self, sample_examples, tokenizer, n=SAMPLE_INFERENCE_COUNT):
        self.samples = sample_examples[:n]; self.tokenizer = tokenizer
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None: return
        model.eval()
        log.info(f"\n  ═══ Epoch {int(state.epoch)} — Sample NER Inferences ═══")
        for i, example in enumerate(self.samples):
            tokens, gold_labels = extract_tokens_and_labels(example)
            text = " ".join(tokens)
            enc = self.tokenizer(tokens, return_tensors="pt", truncation=True,
                                 max_length=256, is_split_into_words=True, padding=True).to(DEVICE)
            with torch.no_grad(): logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            word_ids = enc.word_ids()
            pred_labels = []
            prev_wid = None
            for j, wid in enumerate(word_ids):
                if wid is None: continue
                if wid != prev_wid: pred_labels.append(ID2LABEL.get(preds[j], "O"))
                prev_wid = wid
            min_len = min(len(tokens), len(gold_labels), len(pred_labels))
            gold_ents = [(tokens[k], gold_labels[k]) for k in range(min_len) if gold_labels[k] != "O"]
            pred_ents = [(tokens[k], pred_labels[k]) for k in range(min_len) if pred_labels[k] != "O"]
            mark = "✓" if gold_ents == pred_ents else "✗"
            log.info(f"  [{i+1}] {text[:100]}{'…' if len(text)>100 else ''}")
            log.info(f"       Gold: {gold_ents[:6]}"); log.info(f"       Pred: {pred_ents[:6]}  {mark}")

class FillerSampleInferenceCallback(TrainerCallback):
    def __init__(self, sample_examples, tokenizer, cfg, n=SAMPLE_INFERENCE_COUNT):
        self.samples = sample_examples[:n]; self.tokenizer = tokenizer; self.cfg = cfg
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None: return
        model.eval()
        log.info(f"\n  ═══ Epoch {int(state.epoch)} — Sample Filler Inferences ═══")
        for i, example in enumerate(self.samples):
            # For Seq2Seq, we already prepared masked texts, but let's just mask it dynamically using standard logic
            # actually we can just create the masked version using create_filler_pair
            pair = create_filler_pair(example)
            masked = pair["input_text"]
            gold = pair["target_text"]
            
            # Since create_filler_pair produces "Replace PII: ...", we should strip "Replace PII: " for MLM so it fits
            is_mlm = self.cfg.get("type", "seq2seq") == "mlm"
            if is_mlm and masked.startswith("Replace PII: "):
                masked = masked.replace("Replace PII: ", "", 1)
                
            pred = run_filler(masked, model, self.tokenizer, self.cfg)
            
            log.info(f"  [{i+1}] Masked : {masked[:100]}{'…' if len(masked)>100 else ''}")
            if is_mlm:
                log.info(f"       Pred     : {pred[:100]}{'…' if len(pred)>100 else ''}")
            else:
                log.info(f"       Gold (tgt): {gold[:100]}{'…' if len(gold)>100 else ''}")
                log.info(f"       Pred (gen): {pred[:100]}{'…' if len(pred)>100 else ''}")

class VerboseLoggingCallback(TrainerCallback):
    """Logs checkpoint saves, epoch boundaries, and eval results prominently."""
    def __init__(self, model_name, model_type="encoder"):
        self.model_name = model_name
        self.model_type = model_type
        self._epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        import time
        self._epoch_start_time = time.time()
        epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        log.info(f"\n{'━'*70}")
        log.info(f"  ▶ EPOCH {epoch}/{int(args.num_train_epochs)} STARTING  [{self.model_type.upper()}: {self.model_name}]")
        log.info(f"{'━'*70}")

    def on_epoch_end(self, args, state, control, **kwargs):
        import time
        epoch = int(state.epoch)
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        mins, secs = divmod(int(elapsed), 60)
        log.info(f"\n{'━'*70}")
        log.info(f"  ◼ EPOCH {epoch}/{int(args.num_train_epochs)} FINISHED  [{self.model_type.upper()}: {self.model_name}]  ⏱ {mins}m {secs}s")
        log.info(f"{'━'*70}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch else "?"
        log.info(f"\n  ┌──────────────────────────────────────────────┐")
        log.info(f"  │  📊 VALIDATION RESULTS — Epoch {epoch}               │")
        log.info(f"  ├──────────────────────────────────────────────┤")
        if metrics:
            for k, v in sorted(metrics.items()):
                if k.startswith("eval_"):
                    display_k = k.replace("eval_", "").upper()
                    if isinstance(v, float):
                        log.info(f"  │  {display_k:20s}: {v:.6f}           │")
                    else:
                        log.info(f"  │  {display_k:20s}: {v}           │")
        log.info(f"  └──────────────────────────────────────────────┘")

    def on_save(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else "?"
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        log.info(f"  💾 CHECKPOINT SAVED: {ckpt_dir}")
        log.info(f"     (epoch={epoch}, global_step={state.global_step}, best_metric={state.best_metric})")
        if PUSH_TO_HUB:
            log.info(f"     📤 Pushing checkpoint to HuggingFace Hub ...")
        else:
            log.info(f"     ⏭️  Hub push DISABLED — checkpoint saved locally only")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            epoch = f"{state.epoch:.2f}" if state.epoch else "?"
            lr = logs.get("learning_rate", "?")
            loss = logs.get("loss", "?")
            log.info(f"  📈 step={step} | epoch={epoch} | loss={loss:.4f} | lr={lr:.2e}" if isinstance(loss, float) and isinstance(lr, float)
                     else f"  📈 step={step} | epoch={epoch} | loss={loss} | lr={lr}")

def train_encoder(model_name: str, train_ds: Dataset, val_ds: Dataset):
    model, tokenizer, cfg = build_encoder(model_name)
    output_dir = os.path.join(OUTPUT_DIR, f"encoder_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Skip if already fully trained
    if os.path.exists(os.path.join(output_dir, "model.safetensors")):
        log.info(f"Encoder {model_name} already trained — loading ...")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForTokenClassification.from_pretrained(
            output_dir, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True).to(DEVICE)
        if cfg.get("needs_deberta_fix"): fix_deberta_params(model)
        return model, tokenizer, None

    patience = cfg.get("patience", 3)
    steps_per_epoch = max(1, len(train_ds) // (cfg["batch_size"] * cfg["grad_accum"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    # ── Verbose config banner ────────────────────────────────────────────
    log.info(f"\n{'─'*70}")
    log.info(f"  ENCODER CONFIG: {model_name}")
    log.info(f"{'─'*70}")
    log.info(f"  HF model       : {cfg['hf_name']}")
    log.info(f"  Train samples  : {len(train_ds):,}")
    log.info(f"  Val samples    : {len(val_ds):,}")
    log.info(f"  Max epochs     : {cfg['epochs']}")
    log.info(f"  Early stopping : patience={patience} (metric=f1, higher=better)")
    log.info(f"  Batch size     : {cfg['batch_size']} (x{cfg['grad_accum']} grad_accum = effective {cfg['batch_size'] * cfg['grad_accum']})")
    log.info(f"  Learning rate  : {cfg['learning_rate']}")
    log.info(f"  Weight decay   : {cfg['weight_decay']}")
    log.info(f"  Max length     : {cfg['max_length']}")
    log.info(f"  Warmup steps   : {warmup_steps} / {total_steps} total steps")
    log.info(f"  Steps/epoch    : {steps_per_epoch}")
    log.info(f"  Eval strategy  : every epoch")
    log.info(f"  Save strategy  : every epoch (keep best 2 checkpoints)")
    log.info(f"  Push to Hub    : {PUSH_TO_HUB}")
    log.info(f"  Log every      : {LOG_EVERY_N_STEPS} steps")
    log.info(f"  BF16={BF16_OK}, FP16={FP16_OK}")
    log.info(f"{'─'*70}")

    tok_fn = lambda ex: tokenize_and_align_ner(ex, tokenizer, cfg["max_length"])
    log.info(f"  Tokenizing train set ...")
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    log.info(f"  Tokenizing val set ...")
    val_tok = val_ds.map(tok_fn, batched=True, remove_columns=val_ds.column_names)
    log.info(f"  Tokenization complete. Train: {len(train_tok):,}, Val: {len(val_tok):,}")

    hub_id = f"{HF_USERNAME}/pii-ner-{model_name}" if PUSH_TO_HUB else None
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        bf16=BF16_OK, fp16=FP16_OK and not cfg.get("needs_deberta_fix", False), logging_steps=LOG_EVERY_N_STEPS,
        save_total_limit=2, report_to=REPORT_TO, seed=SEED,
        push_to_hub=False, run_name=f"enc-{model_name}",
    )
    log.info(f"  Checkpoint saved : every epoch (keep best 2)")
    log.info(f"  Validation eval  : every epoch")
    log.info(f"  HF Hub push      : only at the END (best model)")

    sample_examples = [val_ds[i] for i in range(min(SAMPLE_INFERENCE_COUNT, len(val_ds)))]
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        processing_class=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience),
                   NERSampleInferenceCallback(sample_examples, tokenizer),
                   VerboseLoggingCallback(model_name, model_type="encoder")],
    )

    # Auto-resume from checkpoint
    resume = None
    if os.path.isdir(output_dir):
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")])
        if ckpts:
            resume = True
            log.info(f"  ⟳ Resuming from checkpoint! Found: {ckpts}")
            
            # Clean up rogue scaler.pt if AMP is disabled (fixes HuggingFace crash)
            if not getattr(args, "fp16", False) and not getattr(args, "bf16", False):
                for ckpt in ckpts:
                    scaler_path = os.path.join(output_dir, ckpt, "scaler.pt")
                    if os.path.exists(scaler_path):
                        os.remove(scaler_path)
                        log.info(f"  🧹 Removed rogue scaler.pt from {ckpt} (AMP is disabled)")
        else:
            log.info(f"  Starting fresh training (no checkpoints found)")
    else:
        log.info(f"  Starting fresh training")

    log.info(f"  🚀 Training starts NOW ...")
    trainer.train(resume_from_checkpoint=resume)
    log.info(f"  Training complete! Saving model ...")
    trainer.save_model(output_dir); tokenizer.save_pretrained(output_dir)
    log.info(f"  ✓ Encoder {model_name} saved to {output_dir}")

    eval_result = trainer.evaluate()
    log.info(f"\n  ╔══════════════════════════════════════╗")
    log.info(f"  ║  ENCODER {model_name.upper():^20s} RESULTS  ║")
    log.info(f"  ╠══════════════════════════════════════╣")
    log.info(f"  ║  Final F1:   {eval_result.get('eval_f1', 0):.4f}              ║")
    log.info(f"  ║  Eval Loss:  {eval_result.get('eval_loss', 0):.4f}              ║")
    log.info(f"  ╚══════════════════════════════════════╝")
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_result, f, indent=2, default=str)
    log.info(f"  Results saved: {output_dir}/eval_results.json")

    # Push best model to HF Hub at the end
    if PUSH_TO_HUB:
        log.info(f"  📤 Pushing BEST model to HuggingFace Hub: {hub_id} ...")
        trainer.push_to_hub(commit_message=f"Best encoder {model_name} — F1={eval_result.get('eval_f1', 0):.4f}")
        log.info(f"  ✓ Model pushed to Hub!")
    return model, tokenizer, eval_result

def evaluate_filler_standalone(filler_name="bart-base", dataset=None):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, Trainer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
    import os
    
    BASE_DIR = "/kaggle/working/pii_pipeline/outputs" if os.path.exists("/kaggle/working") else "outputs"
    model_dir = f"{BASE_DIR}/filler_{filler_name}"
    
    if not os.path.exists(model_dir):
        print(f"❌ Cannot find model at {model_dir}")
        return
        
    log.info(f"\n{'═'*70}\n  STANDALONE EVALUATION: {filler_name.upper()}\n{'═'*70}")
    log.info(f"Loading {filler_name} from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    cfg = FILLER_REGISTRY[filler_name]
    is_seq2seq = cfg.get("type", "seq2seq") == "seq2seq"
    
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_dir).to(DEVICE)
        
    if dataset is None:
        try:
            dataset = splits["test"]
        except NameError:
            log.error("❌ Cannot find 'splits' variable in memory.")
            return

    log.info(f"Evaluating on {len(dataset)} samples...")

    if is_seq2seq:
        eval_pairs = dataset.map(create_filler_pair, remove_columns=dataset.column_names)
        tok_fn = lambda ex: tokenize_filler_pairs(ex, tokenizer, cfg["max_input_length"], cfg["max_target_length"])
        tok_ds = eval_pairs.map(tok_fn, batched=True, remove_columns=eval_pairs.column_names)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    else:
        # MLM eval requires specific masking which is handled mostly in train
        log.warning("Standalone MLM evaluation requires specific data mapping, skipping for simplicity.")
        return
        
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="outputs/tmp", per_device_eval_batch_size=cfg["eval_batch_size"], bf16=BF16_OK, fp16=FP16_OK, report_to=[]),
        eval_dataset=tok_ds,
        data_collator=data_collator,
    )
    
    log.info("Running Evaluation...")
    results = trainer.evaluate()
    log.info(f"✅ Standalone Eval Loss: {results.get('eval_loss', 0):.4f}")
    
    # Save to disk
    import json
    res_path = os.path.join(model_dir, "standalone_eval_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
        
    # Push to Hub
    if PUSH_TO_HUB:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = f"{HF_USERNAME}/pii-filler-{filler_name}"
            log.info(f"📤 Uploading standalone evaluation metrics to {repo_id}...")
            api.upload_file(
                path_or_fileobj=res_path,
                path_in_repo="standalone_eval_results.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add standalone evaluation results"
            )
        except Exception as e:
            log.warning(f"Failed to upload evaluation results to Hub: {e}")
            
    return results

def evaluate_encoder_standalone(encoder_name="distilroberta", dataset=None):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
    import os, json
    import numpy as np
    
    # 1. Paths
    BASE_DIR = "/kaggle/working/pii_pipeline/outputs" if os.path.exists("/kaggle/working") else "outputs"
    model_dir = f"{BASE_DIR}/encoder_{encoder_name}"
    
    if not os.path.exists(model_dir):
        print(f"❌ Cannot find model at {model_dir}")
        return
        
    log.info(f"\n{'═'*70}\n  STANDALONE EVALUATION: {encoder_name.upper()}\n{'═'*70}")
    log.info(f"Loading {encoder_name} from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)
    
    if dataset is None:
        try:
            dataset = splits["test"]
        except NameError:
            log.error("❌ Cannot find 'splits' variable in memory. Provide a valid dataset object.")
            return

    log.info(f"Evaluating on {len(dataset)} samples...")

    # 2. Tokenize
    cfg = ENCODER_REGISTRY[encoder_name]
    def tok_fn(ex): return tokenize_and_align_ner(ex, tokenizer, cfg["max_length"])
    tok_ds = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)

    # 3. Special Compute Metrics (gives full classification report)
    from seqeval.metrics import classification_report as seq_class_report
    def rich_ner_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        true_labels, pred_labels = [], []
        for pred_seq, label_seq in zip(preds, labels):
            true_sent, pred_sent = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100: continue
                true_sent.append(ID2LABEL.get(int(l), "O"))
                pred_sent.append(ID2LABEL.get(int(p), "O"))
            true_labels.append(true_sent); pred_labels.append(pred_sent)
        report = seq_class_report(true_labels, pred_labels)
        report_dict = seq_class_report(true_labels, pred_labels, output_dict=True)
        print("\n=== FULL CLASSIFICATION REPORT ===")
        print(report)
        base_metrics = compute_ner_metrics(eval_preds) # Return standard f1 too
        
        # Add the detailed dictionary logic explicitly to the trainer metrics
        if isinstance(report_dict, dict):
            for key, val in report_dict.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        base_metrics[f"eval_{key}_{k}"] = v
                else:
                    base_metrics[f"eval_{key}"] = val
        return base_metrics
        
    # 4. Evaluate
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="outputs/tmp", per_device_eval_batch_size=32, bf16=BF16_OK, fp16=FP16_OK, report_to=[]),
        eval_dataset=tok_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=rich_ner_metrics
    )
    
    log.info("Running Evaluation...")
    results = trainer.evaluate()
    log.info(f"✅ Standalone Eval F1: {results.get('eval_f1', 0):.4f}")
    
    # Save to disk
    res_path = os.path.join(model_dir, "standalone_eval_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
        
    # Push to Hub
    if PUSH_TO_HUB:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = f"{HF_USERNAME}/pii-ner-{encoder_name}"
            log.info(f"📤 Uploading standalone evaluation metrics to {repo_id}...")
            api.upload_file(
                path_or_fileobj=res_path,
                path_in_repo="standalone_eval_results.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add rich standalone NER evaluation results"
            )
        except Exception as e:
            log.warning(f"Failed to upload evaluation results to Hub: {e}")
            
    return results

def run_ner(text: str, model, tokenizer) -> Tuple[str, list]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
    with torch.no_grad(): logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    word_ids = enc.word_ids()

    words, tags = [], []
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        tag = ID2LABEL.get(preds[i], "O")
        if wid != prev_wid:
            raw = tokenizer.convert_ids_to_tokens(enc["input_ids"][0][i].item())
            raw = raw.replace("▁", "").replace("##", "").replace("Ġ", "")
            words.append(raw); tags.append(tag)
        else:
            cont = tokens[i].replace("▁", "").replace("##", "").replace("Ġ", "")
            if words: words[-1] += cont
        prev_wid = wid

    masked_words, entity_spans, prev_entity, current_entity_words = [], [], None, []
    for w, t in zip(words, tags):
        if t == "O":
            if current_entity_words and prev_entity:
                entity_spans.append((" ".join(current_entity_words), prev_entity)); current_entity_words = []
            masked_words.append(w); prev_entity = None
        elif t.startswith("B-"):
            if current_entity_words and prev_entity:
                entity_spans.append((" ".join(current_entity_words), prev_entity))
            etype = t[2:]; masked_words.append(f"[{etype}]"); current_entity_words = [w]; prev_entity = etype
        elif t.startswith("I-") and prev_entity: current_entity_words.append(w)
        else:
            if current_entity_words and prev_entity:
                entity_spans.append((" ".join(current_entity_words), prev_entity)); current_entity_words = []
            masked_words.append(w); prev_entity = None
    if current_entity_words and prev_entity:
        entity_spans.append((" ".join(current_entity_words), prev_entity))
    return " ".join(masked_words), entity_spans

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: FILLER — Build, Train, Infer                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_filler(model_name: str):
    cfg = FILLER_REGISTRY[model_name]
    hf_name = cfg["hf_name"]
    mtype = cfg.get("type", "seq2seq")
    log.info(f"Building filler: {model_name} ({hf_name}, type={mtype})")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model_cls = AutoModelForSeq2SeqLM if mtype == "seq2seq" else AutoModelForMaskedLM

    if cfg.get("use_qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16_OK else torch.float16,
            bnb_4bit_use_double_quant=True)
        model = model_cls.from_pretrained(hf_name, quantization_config=bnb_config, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if mtype == "seq2seq" else TaskType.CAUSAL_LM,
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], target_modules=cfg["lora_targets"])
        model = get_peft_model(model, lora_cfg)
    else:
        model = model_cls.from_pretrained(hf_name, torch_dtype=torch.float32).float().to(DEVICE)
        if cfg.get("needs_deberta_fix"): fix_deberta_params(model)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  Parameters: {total:.1f}M")
    return model, tokenizer

def train_filler(model_name: str, train_ds: Dataset, val_ds: Dataset):
    model, tokenizer = build_filler(model_name)
    cfg = FILLER_REGISTRY[model_name]
    mtype = cfg.get("type", "seq2seq")
    output_dir = os.path.join(OUTPUT_DIR, f"filler_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "model.safetensors")) or \
       os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        log.info(f"Filler {model_name} already trained — skipping.")
        return model, tokenizer

    patience = cfg.get("patience", 3)

    # ── Verbose config banner ────────────────────────────────────────────
    log.info(f"\n{'─'*70}")
    log.info(f"  FILLER CONFIG: {model_name} (type={mtype})")
    log.info(f"{'─'*70}")
    log.info(f"  HF model       : {cfg['hf_name']}")
    log.info(f"  Train samples  : {len(train_ds):,}")
    log.info(f"  Val samples    : {len(val_ds):,}")
    log.info(f"  Max epochs     : {cfg['epochs']}")
    log.info(f"  Early stopping : patience={patience} (metric=eval_loss, lower=better)")
    log.info(f"  Batch size     : {cfg['batch_size']} (x{cfg['grad_accum']} grad_accum = effective {cfg['batch_size'] * cfg['grad_accum']})")
    log.info(f"  Learning rate  : {cfg['learning_rate']}")
    log.info(f"  Weight decay   : {cfg['weight_decay']}")
    if mtype == 'seq2seq':
        log.info(f"  Max input len  : {cfg['max_input_length']}")
        log.info(f"  Max target len : {cfg['max_target_length']}")
        log.info(f"  Gen beams      : {cfg.get('gen_num_beams', 4)}")
        log.info(f"  Gen max tokens : {cfg.get('gen_max_tokens', 256)}")
    else:
        log.info(f"  Max input len  : {cfg['max_input_length']}")
        log.info(f"  MLM probability: 0.15")
    log.info(f"  Eval strategy  : every epoch")
    log.info(f"  Save strategy  : every epoch (keep best 2 checkpoints)")
    log.info(f"  Push to Hub    : {PUSH_TO_HUB}")
    log.info(f"  QLoRA          : {cfg.get('use_qlora', False)}")
    log.info(f"  BF16={BF16_OK}, FP16={FP16_OK}")
    log.info(f"{'─'*70}")

    if mtype == "seq2seq":
        log.info(f"  Creating filler pairs (masked → original) ...")
        train_pairs = train_ds.map(create_filler_pair, remove_columns=train_ds.column_names)
        val_pairs = val_ds.map(create_filler_pair, remove_columns=val_ds.column_names)
        log.info(f"  Tokenizing filler pairs ...")
        tok_fn = lambda ex: tokenize_filler_pairs(ex, tokenizer,
                                                   cfg["max_input_length"], cfg["max_target_length"])
        train_tok = train_pairs.map(tok_fn, batched=True, remove_columns=train_pairs.column_names)
        val_tok = val_pairs.map(tok_fn, batched=True, remove_columns=val_pairs.column_names)
        log.info(f"  Tokenization complete. Train: {len(train_tok):,}, Val: {len(val_tok):,}")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
        total_steps = max(1, len(train_tok) // (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        steps_per_epoch = max(1, len(train_tok) // (cfg["batch_size"] * cfg["grad_accum"]))
        log.info(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}")

        hub_id = f"{HF_USERNAME}/pii-filler-{model_name}" if PUSH_TO_HUB else None
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir, num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["eval_batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
            warmup_steps=warmup_steps,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            predict_with_generate=False, bf16=BF16_OK, fp16=FP16_OK and not cfg.get("needs_deberta_fix", False),
            logging_steps=LOG_EVERY_N_STEPS, save_total_limit=2, report_to=REPORT_TO, seed=SEED,
            push_to_hub=False, run_name=f"fill-{model_name}",
        )
        log.info(f"  Checkpoint saved : every epoch (keep best 2)")
        log.info(f"  Validation eval  : every epoch")
        log.info(f"  HF Hub push      : only at the END (best model)")
        trainer_cls = Seq2SeqTrainer
    elif mtype == "mlm":
        log.info(f"  Tokenizing for MLM ...")
        def mlm_preprocess(examples):
            return tokenizer(examples["source_text"], truncation=True, max_length=cfg["max_input_length"])
        train_tok = train_ds.map(mlm_preprocess, batched=True, remove_columns=train_ds.column_names)
        val_tok = val_ds.map(mlm_preprocess, batched=True, remove_columns=val_ds.column_names)
        log.info(f"  Tokenization complete. Train: {len(train_tok):,}, Val: {len(val_tok):,}")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        total_steps = max(1, len(train_tok) // (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        steps_per_epoch = max(1, len(train_tok) // (cfg["batch_size"] * cfg["grad_accum"]))
        log.info(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}")

        hub_id = f"{HF_USERNAME}/pii-filler-{model_name}" if PUSH_TO_HUB else None
        args = TrainingArguments(
            output_dir=output_dir, num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["eval_batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
            warmup_steps=warmup_steps,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            bf16=BF16_OK, fp16=FP16_OK and not cfg.get("needs_deberta_fix", False),
            logging_steps=LOG_EVERY_N_STEPS, save_total_limit=2, report_to=REPORT_TO, seed=SEED,
            push_to_hub=False, run_name=f"fill-{model_name}",
        )
        log.info(f"  Checkpoint saved : every epoch (keep best 2)")
        log.info(f"  Validation eval  : every epoch")
        log.info(f"  HF Hub push      : only at the END (best model)")
        trainer_cls = Trainer

    sample_examples = [val_ds[i] for i in range(min(SAMPLE_INFERENCE_COUNT, len(val_ds)))]
    trainer = trainer_cls(model=model, args=args, train_dataset=train_tok,
                          eval_dataset=val_tok, processing_class=tokenizer, data_collator=data_collator,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=patience),
                                     FillerSampleInferenceCallback(sample_examples, tokenizer, cfg),
                                     VerboseLoggingCallback(model_name, model_type="filler")])
    resume = None
    if os.path.isdir(output_dir):
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")])
        if ckpts:
            resume = True
            log.info(f"  ⟳ Resuming from checkpoint! Found: {ckpts}")
            
            # Clean up rogue scaler.pt if AMP is disabled (fixes HuggingFace crash)
            if not getattr(args, "fp16", False) and not getattr(args, "bf16", False):
                for ckpt in ckpts:
                    scaler_path = os.path.join(output_dir, ckpt, "scaler.pt")
                    if os.path.exists(scaler_path):
                        os.remove(scaler_path)
                        log.info(f"  🧹 Removed rogue scaler.pt from {ckpt} (AMP is disabled)")
        else:
            log.info(f"  Starting fresh training (no checkpoints found)")
    else:
        log.info(f"  Starting fresh training")

    log.info(f"  🚀 Training starts NOW ...")
    trainer.train(resume_from_checkpoint=resume)
    log.info(f"  Training complete! Saving model ...")
    trainer.save_model(output_dir); tokenizer.save_pretrained(output_dir)
    log.info(f"  ✓ Filler {model_name} saved to {output_dir}")

    # Push best model to HF Hub at the end
    if PUSH_TO_HUB:
        log.info(f"  📤 Pushing BEST filler model to HuggingFace Hub: {hub_id} ...")
        trainer.push_to_hub(commit_message=f"Best filler {model_name}")
        log.info(f"  ✓ Filler model pushed to Hub!")
    return model, tokenizer

def run_filler(masked_text: str, model, tokenizer, cfg: dict) -> str:
    mtype = cfg.get("type", "seq2seq")
    if mtype == "seq2seq":
        prefix = cfg.get("prefix", "")
        prompt = f"{prefix}Replace PII placeholders with realistic entities: {masked_text}"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=cfg.get("max_input_length", 256), padding=True).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=cfg.get("gen_max_tokens", 256),
                                     num_beams=cfg.get("gen_num_beams", 4), do_sample=False)
        out_ids = torch.clamp(out_ids, 0, len(tokenizer) - 1)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)
    elif mtype == "mlm":
        filled_text = masked_text
        for etype in ENTITY_TYPES:
            tag = f"[{etype}]"
            if tag in filled_text:
                filled_text = filled_text.replace(tag, f"{tokenizer.mask_token} {tokenizer.mask_token}")
        inputs = tokenizer(filled_text, return_tensors="pt").to(model.device)
        with torch.no_grad(): outputs = model(**inputs)
        token_logits = outputs.logits[0]
        mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        predicted_ids = inputs.input_ids[0].clone()
        for idx in mask_idx: predicted_ids[idx] = token_logits[idx].argmax()
        return tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()
    return ""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: PIPELINE — Combined NER→Mask→Fill                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def anonymize(text, enc_model, enc_tok, fill_model, fill_tok, fill_cfg):
    masked_text, entities = run_ner(text, enc_model, enc_tok)
    if not entities: return text
    return run_filler(masked_text, fill_model, fill_tok, fill_cfg)

def batch_anonymize(texts, enc_model, enc_tok, fill_model, fill_tok, fill_cfg, desc="Anonymizing"):
    from tqdm.auto import tqdm
    anonymized, masked_texts, all_entities = [], [], []
    for text in tqdm(texts, desc=desc):
        masked, entities = run_ner(text, enc_model, enc_tok)
        masked_texts.append(masked); all_entities.append(entities)
        anon = run_filler(masked, fill_model, fill_tok, fill_cfg) if entities else text
        anonymized.append(anon)
    return anonymized, masked_texts, all_entities

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8: EVALUATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_entity_leakage(originals, anonymized):
    total_ent, leaked_ent = 0, 0
    for orig, anon in zip(originals, anonymized):
        anon_lower = anon.lower()
        for entity in re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', orig):
            if len(entity) > 2:
                total_ent += 1
                if entity.lower() in anon_lower: leaked_ent += 1
    return {"entity_leak_rate": round(leaked_ent / max(total_ent, 1) * 100, 2),
            "leaked": leaked_ent, "total": total_ent}

def compute_utility_metrics(originals, anonymized):
    import evaluate as hf_evaluate
    import numpy as np
    results = {}
    rouge_m = hf_evaluate.load("rouge")
    rouge = rouge_m.compute(predictions=anonymized, references=originals)
    results["rouge"] = {k: round(v, 4) for k, v in rouge.items() if isinstance(v, float)}
    bleu_m = hf_evaluate.load("sacrebleu")
    bleu = bleu_m.compute(predictions=anonymized, references=[[r] for r in originals])
    results["bleu"] = round(bleu["score"], 2)
    try:
        bsm = hf_evaluate.load("bertscore")
        bs = bsm.compute(predictions=anonymized, references=originals, lang="en")
        results["bertscore_f1"] = round(float(np.mean([float(x) for x in bs["f1"]])), 4)
    except Exception as e:
        log.warning(f"BERTScore failed: {e}"); results["bertscore_f1"] = 0.0
        
    try:
        wer_m = hf_evaluate.load("wer")
        wer = wer_m.compute(predictions=anonymized, references=originals)
        results["wer"] = round(wer, 4)
    except Exception as e:
        log.warning(f"WER failed: {e}"); results["wer"] = 0.0

    # Length preservation ratio
    orig_lens = [len(x.split()) for x in originals if x.strip()]
    anon_lens = [len(x.split()) for x in anonymized if x.strip()]
    mean_orig = np.mean(orig_lens) if orig_lens else 1.0
    mean_anon = np.mean(anon_lens) if anon_lens else 1.0
    results["length_ratio"] = round(mean_anon / mean_orig, 4)

    return results

def evaluate_pipeline(originals, anonymized, enc_name, fill_name):
    combo = f"{enc_name}+{fill_name}"
    log.info(f"\n{'═'*70}\n  EVALUATING: {combo}  ({len(originals)} examples)\n{'═'*70}")
    
    leakage = compute_entity_leakage(originals, anonymized)
    utility = compute_utility_metrics(originals, anonymized)
    results = {"pipeline": combo, **leakage, **utility}
    
    # Extract concise output for logs
    log.info(f"  Entity Leak Rate: {leakage['entity_leak_rate']}%  ({leakage['leaked']}/{leakage['total']})")
    log.info(f"  ROUGE-L: {utility.get('rouge', {}).get('rougeL', 0):.4f}")
    log.info(f"  BLEU:    {utility.get('bleu', 0):.2f}")
    log.info(f"  BERTScr: {utility.get('bertscore_f1', 0):.4f}")
    log.info(f"  WER:     {utility.get('wer', 0.0):.4f} (Word Error Rate)")
    log.info(f"  Len Ratio: {utility.get('length_ratio', 0.0):.2f} (Preservation of original text size)")
    
    os.makedirs(EVAL_DIR, exist_ok=True)
    res_path = os.path.join(EVAL_DIR, f"results_{combo}.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"  Results saved: {res_path}")
    
    # Save samples
    n = min(EVAL_SAMPLES_FOR_DISPLAY, len(originals))
    sample_path = os.path.join(EVAL_DIR, f"samples_{combo}.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"[{i+1}] ORIG: {originals[i]}\n     ANON: {anonymized[i]}\n\n")

    # Push the end-to-end results to BOTH HuggingFace repositories!
    if PUSH_TO_HUB:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            enc_repo = f"{HF_USERNAME}/pii-ner-{enc_name}"
            fill_repo = f"{HF_USERNAME}/pii-filler-{fill_name}"
            
            for repo, rname in [(enc_repo, fill_name), (fill_repo, enc_name)]:
                log.info(f"  📤 Uploading MIX pipeline metrics & samples to Hub ({repo})...")
                api.upload_file(path_or_fileobj=res_path, path_in_repo="pipeline_eval_results.json",
                                repo_id=repo, repo_type="model", 
                                commit_message=f"Add Full Pipeline Eval (used alongside {rname})")
                api.upload_file(path_or_fileobj=sample_path, path_in_repo="pipeline_samples.txt",
                                repo_id=repo, repo_type="model", 
                                commit_message=f"Add Pipeline Sample Outputs")
                                
        except Exception as e:
            log.warning(f"  Failed to upload pipeline metrics/samples to Hub: {e}")

    return results

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: MAIN EXECUTION — Controlled by FLAGS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__" or IS_KAGGLE or True:  # Always runs in notebook

    # ── Load data once (shared by all models) ────────────────────────────────
    any_training = any([TRAIN_ENCODER_DISTILROBERTA, TRAIN_ENCODER_ROBERTA,
                        TRAIN_ENCODER_DEBERTA, TRAIN_FILLER_BART,
                        TRAIN_FILLER_DEBERTA, RUN_EVALUATION])
    if any_training:
        splits = prepare_all_data()

    # ── Train Encoders ───────────────────────────────────────────────────────
    if TRAIN_ENCODER_DISTILROBERTA:
        log.info("\n" + "="*70 + "\n  TRAINING ENCODER: distilroberta\n" + "="*70)
        train_encoder("distilroberta", splits["half_a"], splits["val_encoder"])
        evaluate_encoder_standalone("distilroberta", splits["test"])
        cleanup_gpu()

    if TRAIN_ENCODER_ROBERTA:
        log.info("\n" + "="*70 + "\n  TRAINING ENCODER: roberta\n" + "="*70)
        train_encoder("roberta", splits["half_a"], splits["val_encoder"])
        evaluate_encoder_standalone("roberta", splits["test"])
        cleanup_gpu()

    if TRAIN_ENCODER_DEBERTA:
        log.info("\n" + "="*70 + "\n  TRAINING ENCODER: deberta\n" + "="*70)
        train_encoder("deberta", splits["half_a"], splits["val_encoder"])
        evaluate_encoder_standalone("deberta", splits["test"])
        cleanup_gpu()

    # ── Train Fillers ────────────────────────────────────────────────────────
    if TRAIN_FILLER_BART:
        log.info("\n" + "="*70 + "\n  TRAINING FILLER: bart-base (Seq2Seq)\n" + "="*70)
        train_filler("bart-base", splits["half_b"], splits["val_filler"])
        evaluate_filler_standalone("bart-base", splits["test"])
        cleanup_gpu()

    if TRAIN_FILLER_DEBERTA:
        log.info("\n" + "="*70 + "\n  TRAINING FILLER: deberta-filler (MLM)\n" + "="*70)
        train_filler("deberta-filler", splits["half_b"], splits["val_filler"])
        cleanup_gpu()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if RUN_EVALUATION:
        log.info("\n" + "="*70 + f"\n  EVALUATING: {EVAL_ENCODER} + {EVAL_FILLER}\n" + "="*70)
        enc_model, enc_tok, _ = train_encoder(EVAL_ENCODER, splits["half_a"], splits["val_encoder"])
        fill_model, fill_tok  = train_filler(EVAL_FILLER, splits["half_b"], splits["val_filler"])
        fill_cfg = FILLER_REGISTRY[EVAL_FILLER]
        n_test = min(len(splits["test"]), 200 if QUICK_MODE else len(splits["test"]))
        originals = [get_source_text(splits["test"][i]) for i in range(n_test)]
        log.info(f"  Running pipeline on {n_test} test examples ...")
        anonymized, _, _ = batch_anonymize(originals, enc_model, enc_tok, fill_model, fill_tok, fill_cfg)
        evaluate_pipeline(originals, anonymized, EVAL_ENCODER, EVAL_FILLER)
        
        log.info(f"\n  Running standalone NER metrics on {EVAL_ENCODER} ...")
        evaluate_encoder_standalone(encoder_name=EVAL_ENCODER, dataset=splits["test"])
        
        cleanup_gpu()

    log.info("\n  ✓ All requested tasks complete!")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BONUS CELL: Manual HF Hub Push (Kaggle Specific)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Run this cell if trainer.push_to_hub() fails or if you want to push checkpoints.

def manual_hf_push(model_folder_name="encoder_distilroberta", hf_repo_name="pii-ner-distilroberta"):
    from huggingface_hub import HfApi, login
    import os
    
    # Authenticate
    print("Logging into Hugging Face...")
    login(HF_TOKEN)
    
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{hf_repo_name}"
    
    # The path based on your Kaggle structure
    local_dir = f"/kaggle/working/pii_pipeline/outputs/{model_folder_name}"
    
    # Fallback to local path if running outside of exactly that Kaggle path
    if not os.path.exists(local_dir):
         local_dir = f"outputs/{model_folder_name}"
            
    if not os.path.exists(local_dir):
        print(f"❌ Error: Could not find model directory at {local_dir}")
        print("Please check the 'model_folder_name' parameter matches exactly.")
        return
    
    print(f"Creating repo: {repo_id} (if it doesn't exist)...")
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    print(f"🚀 Pushing contents of {local_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message="Manual push from Kaggle",
        ignore_patterns=["checkpoint-*", "logs/", "*.pt"] # Ignored checkpoints, but keeping JSON/TXT evaluations!
    )
    print("✅ Successfully pushed to HuggingFace Hub!")

# Uncomment and adjust the names below to manually push
# manual_hf_push(model_folder_name="encoder_distilroberta", hf_repo_name="pii-ner-distilroberta")
# manual_hf_push(model_folder_name="filler_bart-base", hf_repo_name="pii-filler-bart-base")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BONUS CELL: Standalone Encoder Evaluation                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Run this cell if you want to evaluate an already trained Encoder directly.

# encoder_name = "distilroberta"
# print(f"Loading '{encoder_name}' for standalone evaluation...")
# evaluate_encoder_standalone(encoder_name, splits["test"])
# manual_hf_push(model_folder_name=f"encoder_{encoder_name}", hf_repo_name=f"pii-ner-{encoder_name}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BONUS CELL: Standalone Filler Evaluation                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Run this cell if you want to evaluate an already trained Filler directly.

# filler_name = "bart-base"
# print(f"Loading '{filler_name}' for standalone evaluation...")
# evaluate_filler_standalone(filler_name, splits["test"])
# manual_hf_push(model_folder_name=f"filler_{filler_name}", hf_repo_name=f"pii-filler-{filler_name}")
