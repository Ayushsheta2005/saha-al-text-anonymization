"""
tab_finetune.py — Fine-tune roberta + deberta-mlm on the TAB dataset
=====================================================================
Phase 1: Fine-tune the RoBERTa NER masker on TAB with 15% AI4Privacy rehearsal
Phase 2: Fine-tune the DeBERTa MLM filler on TAB legal text
Phase 3: Evaluate the fine-tuned pipeline on TAB test set

Designed for Kaggle T4 GPU. Convert to .ipynb with:
    pip install jupytext && jupytext --to notebook tab_finetune.py
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: INSTALL & IMPORTS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# # 🔒 TAB Fine-Tuning: roberta + deberta-mlm
# **Goal**: Fine-tune our AI4Privacy-trained PII pipeline on the
# [Text Anonymization Benchmark (TAB)](https://huggingface.co/datasets/ildpil/text-anonymization-benchmark)
# to improve performance on real-world ECHR legal documents.
#
# **Strategy**:
# 1. Fine-tune the RoBERTa NER masker with TAB labels mapped to AI4Privacy's label set
# 2. Mix 15% AI4Privacy data to prevent catastrophic forgetting
# 3. Fine-tune the DeBERTa-v3 MLM filler on legal text
# 4. Evaluate and compare against the zero-shot baseline

# %%
import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["transformers", "datasets", "seqeval", "evaluate", "accelerate",
            "rouge_score", "sacrebleu", "bert_score"]:
    try:
        __import__(pkg.replace("-", "_").replace(" ", ""))
    except ImportError:
        install(pkg)

# %%
import os, gc, re, json, random, logging, sys, time, math
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForTokenClassification, AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForTokenClassification, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback,
)
from seqeval.metrics import (
    f1_score as seq_f1_score,
    classification_report as seq_classification_report,
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: CONFIGURATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# ## ⚙️ Configuration
# All tunable hyperparameters and flags are here.

# %%
# ── Flags ─────────────────────────────────────────────────────────────────────
PHASE_1_MASKER  = True    # Fine-tune RoBERTa NER masker on TAB
PHASE_2_FILLER  = True    # Fine-tune DeBERTa MLM filler on TAB legal text
PHASE_3_EVAL    = True    # Evaluate fine-tuned pipeline on TAB test set
PUSH_TO_HUB     = True    # Push fine-tuned models to HuggingFace Hub
DRY_RUN          = False   # Just print data stats, don't train

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_USERNAME      = "Xyren2005"
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN         = os.getenv("HF_TOKEN")
MASKER_CHECKPOINT = "Xyren2005/pii-ner-roberta"        # Pre-trained on AI4Privacy
FILLER_CHECKPOINT = "Xyren2005/pii-ner-filler_deberta-filler"  # Pre-trained on AI4Privacy

# ── Paths ─────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")
BASE_DIR  = "/kaggle/working/tab_finetune" if IS_KAGGLE else "outputs_tab"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HF Login ──────────────────────────────────────────────────────────────────
if PUSH_TO_HUB and HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in to HuggingFace Hub")

# ── Seeds & Device ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
FP16_OK = torch.cuda.is_available() and not BF16_OK

# ── 21 Entity Types (MUST match the original pipeline exactly) ────────────────
ENTITY_TYPES = [
    "FULLNAME", "FIRST_NAME", "LAST_NAME", "ID_NUMBER", "PASSPORT", "SSN",
    "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "LOCATION", "ORGANIZATION",
    "ACCOUNT_NUM", "CREDIT_CARD", "ZIPCODE", "TITLE", "GENDER", "NUMBER",
    "OTHER_PII", "UNKNOWN",
]

def build_bio_labels(entity_types):
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = build_bio_labels(ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)  # 43

# ── TAB → AI4Privacy Label Mapping ────────────────────────────────────────────
TAB_TO_AI4P = {
    "PERSON":   "FULLNAME",
    "LOC":      "LOCATION",
    "ORG":      "ORGANIZATION",
    "DATETIME": "DATE",
    "CODE":     "ID_NUMBER",
    "QUANTITY": "NUMBER",
    "DEM":      "OTHER_PII",
    "MISC":     "OTHER_PII",
}

# ── AI4Privacy ENTITY_MAP (for rehearsal data) ─────────────────────────────────
AI4P_ENTITY_MAP = {
    "GIVENNAME": "FIRST_NAME", "SURNAME": "LAST_NAME",
    "TITLE": "TITLE", "GENDER": "GENDER", "SEX": "GENDER",
    "CITY": "LOCATION", "STREET": "ADDRESS", "BUILDINGNUM": "ADDRESS",
    "ZIPCODE": "ZIPCODE", "TELEPHONENUM": "PHONE", "EMAIL": "EMAIL",
    "SOCIALNUM": "SSN", "PASSPORTNUM": "PASSPORT",
    "DRIVERLICENSENUM": "ID_NUMBER", "IDCARDNUM": "ID_NUMBER", "TAXNUM": "ID_NUMBER",
    "CREDITCARDNUMBER": "CREDIT_CARD", "DATE": "DATE", "TIME": "TIME", "AGE": "NUMBER",
}

# ── Masker Fine-Tuning Config (Phase 1) ──────────────────────────────────────
MASKER_CFG = {
    "batch_size":      8,       # TAB paragraphs are longer → smaller batch
    "eval_batch_size":  16,
    "learning_rate":   1e-5,    # Lower LR to preserve AI4Privacy knowledge
    "epochs":          8,
    "max_length":      512,     # Legal paragraphs need more context
    "grad_accum":      4,       # Effective batch = 32
    "weight_decay":    0.01,
    "warmup_ratio":    0.10,
    "patience":        2,
    "rehearsal_ratio":  0.15,   # 15% AI4Privacy data mixed in
}

# ── Filler Fine-Tuning Config (Phase 2) ──────────────────────────────────────
FILLER_CFG = {
    "batch_size":      8,
    "eval_batch_size":  16,
    "learning_rate":   1e-5,
    "epochs":          5,
    "max_length":      256,
    "grad_accum":      4,
    "weight_decay":    0.01,
    "warmup_ratio":    0.10,
    "patience":        2,
    "mlm_probability": 0.15,
}

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger("tab_finetune")
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

log.info("=" * 70)
log.info("  TAB FINE-TUNING: roberta + deberta-mlm")
log.info("=" * 70)
log.info(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"  GPU:    {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
log.info(f"  BF16={BF16_OK}, FP16={FP16_OK}")
log.info("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: DATA LOADING — TAB Dataset                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# ## 📦 Data Loading
# Load TAB dataset, convert character-level spans to word-level BIO tags.

# %%
def load_tab_as_ner(split="train"):
    """
    Load TAB dataset and convert to word-level NER format compatible with
    the existing pipeline's 21 entity types.

    Returns a HuggingFace Dataset with columns: tokens, ner_tags, doc_id
    """
    log.info(f"Loading TAB dataset (split={split}) ...")
    ds = load_dataset("ildpil/text-anonymization-benchmark")[split]

    # Deduplicate: keep first annotator per doc
    seen = set()
    unique_docs = []
    for ex in ds:
        if ex["doc_id"] not in seen:
            seen.add(ex["doc_id"])
            unique_docs.append(ex)
    log.info(f"  {len(unique_docs)} unique documents")

    all_tokens, all_tags, all_doc_ids = [], [], []

    for doc in unique_docs:
        text = doc["text"]
        mentions = doc["entity_mentions"]

        # Filter to DIRECT/QUASI only
        pii_mentions = [
            m for m in mentions
            if m.get("identifier_type", "") in {"DIRECT", "QUASI"}
        ]

        # Split into paragraphs
        paragraphs = text.split("\n\n")
        char_cursor = 0

        for para_raw in paragraphs:
            para = para_raw.strip()
            if not para or len(para) < 20:
                char_cursor += len(para_raw) + 2
                continue

            # Find paragraph position in full text
            para_start = text.find(para, char_cursor)
            if para_start < 0:
                char_cursor += len(para_raw) + 2
                continue
            para_end = para_start + len(para)

            # Get mentions within this paragraph
            para_mentions = []
            for m in pii_mentions:
                if m["start_offset"] >= para_start and m["end_offset"] <= para_end:
                    para_mentions.append({
                        **m,
                        "start_offset": m["start_offset"] - para_start,
                        "end_offset":   m["end_offset"]   - para_start,
                    })

            # Convert paragraph → tokens + BIO tags using character offsets
            tokens, tags = _char_spans_to_bio(para, para_mentions)

            if tokens and len(tokens) <= 600:  # Skip extremely long paragraphs
                all_tokens.append(tokens)
                all_tags.append(tags)
                all_doc_ids.append(doc["doc_id"])

            char_cursor = para_end

    log.info(f"  {len(all_tokens)} paragraph segments extracted")

    # Count entity stats
    entity_counts = defaultdict(int)
    for tags in all_tags:
        for t in tags:
            if t.startswith("B-"):
                entity_counts[t[2:]] += 1
    log.info("  Entity distribution:")
    for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        log.info(f"    {etype:<15} {count:>5}")

    return Dataset.from_dict({
        "tokens":  all_tokens,
        "ner_tags": all_tags,
        "doc_id":  all_doc_ids,
    })


def _char_spans_to_bio(text: str, mentions: list) -> Tuple[List[str], List[str]]:
    """
    Convert character-level entity mentions to word-level BIO tags.
    Maps TAB entity types → AI4Privacy entity types using TAB_TO_AI4P.
    """
    # Tokenize text into words with character positions
    words = []
    word_starts = []
    word_ends = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        words.append(text[i:j])
        word_starts.append(i)
        word_ends.append(j)
        i = j

    if not words:
        return [], []

    # Sort mentions by start offset, de-overlap
    sorted_m = sorted(mentions, key=lambda m: m["start_offset"])
    deoverlapped = []
    prev_end = -1
    for m in sorted_m:
        if m["start_offset"] >= prev_end:
            deoverlapped.append(m)
            prev_end = m["end_offset"]

    # Assign BIO tags to each word
    tags = ["O"] * len(words)

    for m in deoverlapped:
        m_start = m["start_offset"]
        m_end   = m["end_offset"]
        tab_type = m.get("entity_type", "MISC")
        ai4p_type = TAB_TO_AI4P.get(tab_type, "OTHER_PII")

        if ai4p_type not in ENTITY_TYPES:
            ai4p_type = "OTHER_PII"

        first_word = True
        for w_idx in range(len(words)):
            # Word overlaps with mention if word_start < m_end AND word_end > m_start
            if word_starts[w_idx] < m_end and word_ends[w_idx] > m_start:
                if first_word:
                    tags[w_idx] = f"B-{ai4p_type}"
                    first_word = False
                else:
                    tags[w_idx] = f"I-{ai4p_type}"

    return words, tags


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: AI4Privacy Rehearsal Data                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %%
def load_ai4privacy_rehearsal(n_samples: int) -> Dataset:
    """
    Load a subset of AI4Privacy data for rehearsal (anti-forgetting).
    Converts to the same tokens/ner_tags format as TAB data.
    """
    log.info(f"Loading AI4Privacy rehearsal data ({n_samples} samples) ...")
    ds = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy", split="train")

    # Sample randomly
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    subset = ds.select(indices)

    all_tokens, all_tags, all_doc_ids = [], [], []

    for i in range(len(subset)):
        ex = subset[i]
        tokens, labels = _extract_ai4p_tokens_and_labels(ex)
        if tokens and len(tokens) <= 300:
            all_tokens.append(tokens)
            all_tags.append(labels)
            all_doc_ids.append(f"ai4p_{i}")

    log.info(f"  {len(all_tokens)} AI4Privacy samples for rehearsal")
    return Dataset.from_dict({
        "tokens":  all_tokens,
        "ner_tags": all_tags,
        "doc_id":  all_doc_ids,
    })


def _extract_ai4p_tokens_and_labels(example):
    """Extract tokens and BIO labels from an AI4Privacy example."""
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
        end   = span.get("end", start + span.get("length", len(span.get("value", ""))))
        label = span.get("label", span.get("entity_type", "O")).upper().replace(" ", "_")
        value = span.get("value", text[start:end])
        if pos < start:
            before = text[pos:start].split()
            tokens.extend(before); labels.extend(["O"] * len(before))
        entity_words = value.split()
        if entity_words:
            coarse = AI4P_ENTITY_MAP.get(label, label)
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: NER TOKENIZATION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %%
def tokenize_and_align_ner(examples, tokenizer, max_length=512):
    """Tokenize pre-split tokens and align BIO labels to subword tokens."""
    all_tokens = examples["tokens"]
    all_labels = examples["ner_tags"]

    enc = tokenizer(
        all_tokens, truncation=True, max_length=max_length,
        padding="max_length", is_split_into_words=True,
    )

    aligned_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = enc.word_ids(batch_index=i)
        label_ids = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_wid:
                lbl = labels[wid] if wid < len(labels) else "O"
                label_ids.append(LABEL2ID.get(lbl, 0))
            else:
                lbl = labels[wid] if wid < len(labels) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(LABEL2ID.get(lbl, 0))
            prev_wid = wid
        aligned_labels.append(label_ids)
    enc["labels"] = aligned_labels
    return enc


def compute_ner_metrics(eval_preds):
    """Compute seqeval F1 for NER evaluation."""
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
    f1 = seq_f1_score(true_labels, pred_labels, average="weighted")
    report = seq_classification_report(true_labels, pred_labels)
    print("\n=== NER Classification Report ===")
    print(report)
    return {"f1": f1}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: PHASE 1 — Fine-Tune Masker (NER)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# ## 🎯 Phase 1: Fine-Tune NER Masker on TAB
# - Start from pre-trained `Xyren2005/pii-ner-roberta`
# - Map TAB entity types → AI4Privacy's 21 labels
# - Mix 15% AI4Privacy data to prevent catastrophic forgetting

# %%
def phase1_finetune_masker():
    """Fine-tune the RoBERTa NER masker on TAB data."""
    log.info("\n" + "=" * 70)
    log.info("  PHASE 1: Fine-Tune NER Masker on TAB")
    log.info("=" * 70)

    # ── Load TAB data ──
    tab_train = load_tab_as_ner(split="train")
    tab_val   = load_tab_as_ner(split="validation")
    log.info(f"  TAB train: {len(tab_train)}, TAB val: {len(tab_val)}")

    # ── Load AI4Privacy rehearsal ──
    n_rehearsal = int(len(tab_train) * MASKER_CFG["rehearsal_ratio"] / (1 - MASKER_CFG["rehearsal_ratio"]))
    ai4p_rehearsal = load_ai4privacy_rehearsal(n_rehearsal)
    log.info(f"  AI4Privacy rehearsal: {len(ai4p_rehearsal)} samples "
             f"({MASKER_CFG['rehearsal_ratio']*100:.0f}% of total)")

    # ── Combine ──
    combined_train = concatenate_datasets([tab_train, ai4p_rehearsal]).shuffle(seed=SEED)
    log.info(f"  Combined training set: {len(combined_train)} "
             f"(TAB={len(tab_train)}, AI4P={len(ai4p_rehearsal)})")

    if DRY_RUN:
        log.info("  DRY RUN — skipping training")
        # Print some samples
        for i in range(min(3, len(tab_train))):
            tokens = tab_train[i]["tokens"][:20]
            tags   = tab_train[i]["ner_tags"][:20]
            log.info(f"\n  Sample {i}: {' '.join(tokens)}")
            entities = [(t, tag) for t, tag in zip(tokens, tags) if tag != "O"]
            log.info(f"  Entities: {entities}")
        return None, None

    # ── Load pre-trained masker ──
    log.info(f"  Loading pre-trained masker: {MASKER_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MASKER_CHECKPOINT, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MASKER_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    ).float().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  Model parameters: {total_params:.1f}M")

    # ── Tokenize ──
    log.info("  Tokenizing training data ...")
    tok_fn = lambda ex: tokenize_and_align_ner(ex, tokenizer, MASKER_CFG["max_length"])
    train_tok = combined_train.map(tok_fn, batched=True, remove_columns=combined_train.column_names)
    val_tok   = tab_val.map(tok_fn, batched=True, remove_columns=tab_val.column_names)
    log.info(f"  Tokenized: train={len(train_tok)}, val={len(val_tok)}")

    # ── Training ──
    output_dir = os.path.join(OUTPUT_DIR, "masker_roberta_tab")
    os.makedirs(output_dir, exist_ok=True)

    total_steps = max(1, len(train_tok) // (MASKER_CFG["batch_size"] * MASKER_CFG["grad_accum"])) * MASKER_CFG["epochs"]
    warmup_steps = int(total_steps * MASKER_CFG["warmup_ratio"])

    log.info(f"\n  {'─'*60}")
    log.info(f"  MASKER TRAINING CONFIG")
    log.info(f"  {'─'*60}")
    log.info(f"  Base checkpoint : {MASKER_CHECKPOINT}")
    log.info(f"  Train samples   : {len(train_tok)}")
    log.info(f"  Val samples     : {len(val_tok)}")
    log.info(f"  Epochs          : {MASKER_CFG['epochs']}")
    log.info(f"  Batch size      : {MASKER_CFG['batch_size']} x{MASKER_CFG['grad_accum']} = {MASKER_CFG['batch_size'] * MASKER_CFG['grad_accum']}")
    log.info(f"  Learning rate   : {MASKER_CFG['learning_rate']}")
    log.info(f"  Max length      : {MASKER_CFG['max_length']}")
    log.info(f"  Total steps     : {total_steps}, Warmup: {warmup_steps}")
    log.info(f"  Early stopping  : patience={MASKER_CFG['patience']}")
    log.info(f"  {'─'*60}\n")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=MASKER_CFG["epochs"],
        per_device_train_batch_size=MASKER_CFG["batch_size"],
        per_device_eval_batch_size=MASKER_CFG["eval_batch_size"],
        gradient_accumulation_steps=MASKER_CFG["grad_accum"],
        learning_rate=MASKER_CFG["learning_rate"],
        weight_decay=MASKER_CFG["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=BF16_OK, fp16=FP16_OK,
        logging_steps=25,
        save_total_limit=2,
        report_to=["none"],
        seed=SEED,
        run_name="masker-roberta-tab",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=MASKER_CFG["patience"])],
    )

    # Resume from checkpoint if available
    resume = None
    if os.path.isdir(output_dir):
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")])
        if ckpts:
            resume = True
            log.info(f"  ↻ Resuming from checkpoint: {ckpts[-1]}")

    log.info("  🚀 Starting masker fine-tuning ...")
    trainer.train(resume_from_checkpoint=resume)

    log.info("  Saving fine-tuned masker ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"  ✅ Masker saved to {output_dir}")

    # Final evaluation
    log.info("  Running final evaluation on TAB validation ...")
    results = trainer.evaluate()
    log.info(f"  Final F1: {results.get('eval_f1', 0):.4f}")

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    if PUSH_TO_HUB:
        hub_id = f"{HF_USERNAME}/pii-ner-roberta-tab"
        log.info(f"  📤 Pushing to Hub: {hub_id}")
        trainer.push_to_hub(commit_message="Fine-tuned RoBERTa masker on TAB")

    return model, tokenizer


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: PHASE 2 — Fine-Tune Filler (MLM)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# ## 📝 Phase 2: Fine-Tune DeBERTa MLM Filler on Legal Text
# - Continue training on unmasked ECHR text
# - Standard MLM with 15% random masking
# - Teaches the filler "legalese" vocabulary

# %%
def fix_deberta_params(model):
    """Fix DeBERTa LayerNorm params for float32 stability."""
    for name, param in model.named_parameters():
        if "LayerNorm" in name or "layernorm" in name:
            param.data = param.data.to(torch.float32)
    return model


def phase2_finetune_filler():
    """Fine-tune the DeBERTa MLM filler on TAB legal text."""
    log.info("\n" + "=" * 70)
    log.info("  PHASE 2: Fine-Tune DeBERTa MLM Filler on TAB Legal Text")
    log.info("=" * 70)

    # ── Load TAB text ──
    log.info("  Loading TAB documents for MLM training ...")
    ds = load_dataset("ildpil/text-anonymization-benchmark")

    # Deduplicate train + validation
    all_texts = []
    seen = set()
    for split in ["train", "validation"]:
        for ex in ds[split]:
            if ex["doc_id"] not in seen:
                seen.add(ex["doc_id"])
                all_texts.append(ex["text"])

    log.info(f"  {len(all_texts)} unique documents loaded")

    # Chunk documents into manageable segments
    segments = []
    for text in all_texts:
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Skip very short paragraphs
                segments.append(para)
    log.info(f"  {len(segments)} text segments for MLM training")

    if DRY_RUN:
        log.info("  DRY RUN — skipping training")
        for s in segments[:3]:
            log.info(f"  Sample: {s[:150]}...")
        return None, None

    # ── Load pre-trained filler ──
    log.info(f"  Loading pre-trained filler: {FILLER_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(FILLER_CHECKPOINT)
    model = AutoModelForMaskedLM.from_pretrained(
        FILLER_CHECKPOINT,
        torch_dtype=torch.float32,
    ).float().to(DEVICE)
    fix_deberta_params(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  Model parameters: {total_params:.1f}M")

    # ── Create dataset ──
    # Split 90/10 for train/val
    random.shuffle(segments)
    n_val = max(50, int(len(segments) * 0.10))
    train_texts = segments[:-n_val]
    val_texts   = segments[-n_val:]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds   = Dataset.from_dict({"text": val_texts})

    log.info(f"  MLM train: {len(train_ds)}, MLM val: {len(val_ds)}")

    # ── Tokenize ──
    def mlm_tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True,
            max_length=FILLER_CFG["max_length"],
            padding="max_length",
        )

    train_tok = train_ds.map(mlm_tokenize, batched=True, remove_columns=["text"])
    val_tok   = val_ds.map(mlm_tokenize, batched=True, remove_columns=["text"])

    # ── Training ──
    output_dir = os.path.join(OUTPUT_DIR, "filler_deberta_tab")
    os.makedirs(output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True,
        mlm_probability=FILLER_CFG["mlm_probability"],
    )

    total_steps = max(1, len(train_tok) // (FILLER_CFG["batch_size"] * FILLER_CFG["grad_accum"])) * FILLER_CFG["epochs"]
    warmup_steps = int(total_steps * FILLER_CFG["warmup_ratio"])

    log.info(f"\n  {'─'*60}")
    log.info(f"  FILLER TRAINING CONFIG")
    log.info(f"  {'─'*60}")
    log.info(f"  Base checkpoint : {FILLER_CHECKPOINT}")
    log.info(f"  Train samples   : {len(train_tok)}")
    log.info(f"  Val samples     : {len(val_tok)}")
    log.info(f"  Epochs          : {FILLER_CFG['epochs']}")
    log.info(f"  Batch size      : {FILLER_CFG['batch_size']} x{FILLER_CFG['grad_accum']} = {FILLER_CFG['batch_size'] * FILLER_CFG['grad_accum']}")
    log.info(f"  Learning rate   : {FILLER_CFG['learning_rate']}")
    log.info(f"  MLM prob        : {FILLER_CFG['mlm_probability']}")
    log.info(f"  {'─'*60}\n")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=FILLER_CFG["epochs"],
        per_device_train_batch_size=FILLER_CFG["batch_size"],
        per_device_eval_batch_size=FILLER_CFG["eval_batch_size"],
        gradient_accumulation_steps=FILLER_CFG["grad_accum"],
        learning_rate=FILLER_CFG["learning_rate"],
        weight_decay=FILLER_CFG["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=BF16_OK, fp16=FP16_OK,
        logging_steps=25,
        save_total_limit=2,
        report_to=["none"],
        seed=SEED,
        run_name="filler-deberta-tab",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=FILLER_CFG["patience"])],
    )

    resume = None
    if os.path.isdir(output_dir):
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")])
        if ckpts:
            resume = True
            log.info(f"  ↻ Resuming from checkpoint: {ckpts[-1]}")

    log.info("  🚀 Starting filler fine-tuning ...")
    trainer.train(resume_from_checkpoint=resume)

    log.info("  Saving fine-tuned filler ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"  ✅ Filler saved to {output_dir}")

    results = trainer.evaluate()
    log.info(f"  Final MLM loss: {results.get('eval_loss', 0):.4f}")

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    if PUSH_TO_HUB:
        hub_id = f"{HF_USERNAME}/pii-filler-deberta-tab"
        log.info(f"  📤 Pushing to Hub: {hub_id}")
        trainer.push_to_hub(commit_message="Fine-tuned DeBERTa filler on TAB legal text")

    return model, tokenizer


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8: PHASE 3 — Full Evaluation on TAB Test Set                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# ## 📊 Phase 3: Full Evaluation on TAB Test Set
# Loads fine-tuned masker + filler from HF Hub (or local), runs the full
# pipeline, and computes **all metrics**: entity recall, leakage rate,
# BERTScore, ROUGE-1/2/L, word accuracy — matching `evaluate_tab.py`.

# %%
# ── Install extra eval packages if needed ──
for _pkg in ["rouge_score", "bert_score", "sacrebleu"]:
    try:
        __import__(_pkg.replace("-", "_"))
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])

from rouge_score import rouge_scorer as _rouge_scorer_mod
import bert_score as _bert_score_mod

# %%
# ── Correct HF model IDs (as pushed by trainer.push_to_hub) ──
HF_MASKER_ID = f"{HF_USERNAME}/masker_roberta_tab"
HF_FILLER_ID = f"{HF_USERNAME}/filler_deberta_tab"

def _resolve_model(hf_id: str, local_dir: str, model_label: str):
    """Try HF Hub first, fall back to local dir."""
    from huggingface_hub import repo_exists
    try:
        exists = repo_exists(hf_id, token=HF_TOKEN if HF_TOKEN else None)
    except Exception:
        exists = False
    if exists:
        log.info(f"  ✅ {model_label}: HF Hub → {hf_id}")
        return "hub", hf_id
    elif os.path.isdir(local_dir) and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(local_dir)
    ):
        log.info(f"  ✅ {model_label}: local → {local_dir}")
        return "local", local_dir
    else:
        log.error(f"  ❌ {model_label}: not found at HF ({hf_id}) or local ({local_dir})")
        return None, None


def run_ner(text: str, model, tokenizer) -> Tuple[str, list]:
    """Run NER masker; return (masked_text, [(entity_text, etype), ...])."""
    tokens = text.split()
    if not tokens:
        return text, []
    enc = tokenizer(
        tokens, return_tensors="pt", truncation=True,
        max_length=512, is_split_into_words=True, padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    word_ids = enc.word_ids()
    word_tags, prev_wid = [], None
    for j, wid in enumerate(word_ids):
        if wid is None: continue
        if wid != prev_wid:
            word_tags.append(ID2LABEL.get(preds[j], "O"))
        prev_wid = wid

    masked_words, entity_spans = [], []
    prev_entity, cur_words = None, []
    for w, t in zip(tokens, word_tags[:len(tokens)]):
        if t == "O":
            if cur_words and prev_entity:
                entity_spans.append((" ".join(cur_words), prev_entity)); cur_words = []
            masked_words.append(w); prev_entity = None
        elif t.startswith("B-"):
            if cur_words and prev_entity:
                entity_spans.append((" ".join(cur_words), prev_entity))
            etype = t[2:]; masked_words.append(f"[{etype}]")
            cur_words = [w]; prev_entity = etype
        elif t.startswith("I-") and prev_entity:
            cur_words.append(w)
        else:
            if cur_words and prev_entity:
                entity_spans.append((" ".join(cur_words), prev_entity)); cur_words = []
            masked_words.append(w); prev_entity = None
    if cur_words and prev_entity:
        entity_spans.append((" ".join(cur_words), prev_entity))
    return " ".join(masked_words), entity_spans


def run_filler(masked_text: str, model, tokenizer) -> str:
    """Replace [ENTITY] placeholders via MLM inference."""
    filled = masked_text
    for etype in ENTITY_TYPES:
        tag = f"[{etype}]"
        if tag in filled:
            filled = filled.replace(tag, f"{tokenizer.mask_token} {tokenizer.mask_token}")
    if tokenizer.mask_token not in filled:
        return masked_text
    inputs = tokenizer(filled, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    pred_ids = inputs.input_ids[0].clone()
    for idx in mask_idx:
        pred_ids[idx] = logits[idx].argmax()
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


def phase3_evaluate():
    """Full evaluation: entity recall, leakage, BERTScore, ROUGE, word accuracy."""
    log.info("\n" + "=" * 70)
    log.info("  PHASE 3: Full Evaluation on TAB Test Set")
    log.info("=" * 70)

    # ── Resolve models ──
    masker_local = os.path.join(OUTPUT_DIR, "masker_roberta_tab")
    filler_local  = os.path.join(OUTPUT_DIR, "filler_deberta_tab")
    masker_src, masker_path = _resolve_model(HF_MASKER_ID, masker_local, "Masker")
    filler_src,  filler_path  = _resolve_model(HF_FILLER_ID,  filler_local,  "Filler")
    if masker_path is None or filler_path is None:
        log.error("  ❌ Cannot evaluate — models not found. Finish training first.")
        return

    hf_kw_m = {"token": HF_TOKEN} if masker_src == "hub" and HF_TOKEN else {}
    hf_kw_f = {"token": HF_TOKEN} if filler_src  == "hub" and HF_TOKEN else {}

    masker_tok = AutoTokenizer.from_pretrained(masker_path, **hf_kw_m)
    masker_model = AutoModelForTokenClassification.from_pretrained(
        masker_path, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True, **hf_kw_m,
    ).to(DEVICE)
    masker_model.eval()

    filler_tok = AutoTokenizer.from_pretrained(filler_path, **hf_kw_f)
    filler_model = AutoModelForMaskedLM.from_pretrained(filler_path, **hf_kw_f).to(DEVICE)
    fix_deberta_params(filler_model)
    filler_model.eval()
    log.info("  ✅ Models loaded successfully\n")

    # ── Load TAB test set ──
    tab_test = load_tab_as_ner(split="test")
    log.info(f"  TAB test: {len(tab_test)} paragraphs")

    # ── Run pipeline + collect data ──
    rouge = _rouge_scorer_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    TAB_TO_PLACEHOLDER = {
        "FULLNAME": "FULLNAME", "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION",
        "DATE": "DATE", "ID_NUMBER": "ID_NUMBER", "NUMBER": "NUMBER",
        "OTHER_PII": "OTHER_PII",
    }

    originals, anonymized, gold_masked_list = [], [], []
    total_gold, total_detected = 0, 0
    per_type_gold = defaultdict(int)
    per_type_detected = defaultdict(int)
    leaked_entities = defaultdict(int)
    word_correct, word_total = 0, 0
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    samples_display = []

    for i in range(len(tab_test)):
        tokens    = tab_test[i]["tokens"]
        gold_tags = tab_test[i]["ner_tags"]
        text      = " ".join(tokens)

        # Build gold-masked text (replace entities with [PLACEHOLDER])
        gold_masked_words = list(tokens)
        j = 0
        gold_entities = []
        while j < len(gold_tags):
            if gold_tags[j].startswith("B-"):
                etype = gold_tags[j][2:]
                entity_words = [tokens[j]]
                k = j + 1
                while k < len(gold_tags) and gold_tags[k] == f"I-{etype}":
                    entity_words.append(tokens[k]); k += 1
                gold_entities.append((" ".join(entity_words), etype))
                placeholder = f"[{TAB_TO_PLACEHOLDER.get(etype, etype)}]"
                gold_masked_words[j] = placeholder
                for m in range(j + 1, k):
                    gold_masked_words[m] = ""
                j = k
            else:
                j += 1
        gold_masked = " ".join(w for w in gold_masked_words if w)

        # Run masker
        masked_text, pred_entities = run_ner(text, masker_model, masker_tok)

        # Entity recall & leakage
        masked_lower = masked_text.lower()
        for entity_text, etype in gold_entities:
            total_gold += 1
            per_type_gold[etype] += 1
            if entity_text.lower() not in masked_lower:
                total_detected += 1
                per_type_detected[etype] += 1
            else:
                leaked_entities[entity_text.lower()] += 1

        # Run filler
        if pred_entities:
            anon_text = run_filler(masked_text, filler_model, filler_tok)
        else:
            anon_text = text

        originals.append(text)
        anonymized.append(anon_text)
        gold_masked_list.append(gold_masked)

        # ROUGE (anon vs gold_masked — utility preservation)
        r = rouge.score(gold_masked, anon_text)
        rouge1_scores.append(r["rouge1"].fmeasure)
        rouge2_scores.append(r["rouge2"].fmeasure)
        rougeL_scores.append(r["rougeL"].fmeasure)

        # Word accuracy (anon vs original non-entity words)
        orig_words = text.split()
        anon_words  = anon_text.split()
        for ow, aw in zip(orig_words, anon_words):
            word_total += 1
            if ow.lower() == aw.lower():
                word_correct += 1

        if i < 5:
            samples_display.append((text, gold_masked, masked_text, anon_text))

    # ── BERTScore ──
    log.info("  Computing BERTScore (this may take ~1 min) ...")
    _P, _R, _F = _bert_score_mod.score(
        anonymized, originals,
        lang="en", model_type="distilbert-base-uncased", verbose=False,
    )
    bert_p  = float(_P.mean()) * 100
    bert_r  = float(_R.mean()) * 100
    bert_f1 = float(_F.mean()) * 100

    # ── Aggregate metrics ──
    overall_recall   = total_detected / max(total_gold, 1)
    leakage_rate     = 100.0 * (total_gold - total_detected) / max(total_gold, 1)
    word_acc         = 100.0 * word_correct / max(word_total, 1)
    avg_rouge1       = 100.0 * sum(rouge1_scores) / max(len(rouge1_scores), 1)
    avg_rouge2       = 100.0 * sum(rouge2_scores) / max(len(rouge2_scores), 1)
    avg_rougeL       = 100.0 * sum(rougeL_scores) / max(len(rougeL_scores), 1)
    top_leaked = sorted(leaked_entities.items(), key=lambda x: -x[1])[:5]

    # ── Print full report ──
    log.info(f"\n  {'═'*68}")
    log.info(f"  FULL EVALUATION REPORT — Fine-Tuned roberta + deberta-mlm on TAB")
    log.info(f"  Masker : {masker_path}")
    log.info(f"  Filler : {filler_path}")
    log.info(f"  {'═'*68}")

    log.info(f"\n  ── Masker Entity Recall ──")
    log.info(f"  Overall Masker Recall : {overall_recall:.4f}  ({total_detected}/{total_gold})")
    log.info(f"  Per entity type:")
    for etype in sorted(per_type_gold.keys()):
        recall = per_type_detected[etype] / max(per_type_gold[etype], 1)
        log.info(f"    {etype:<16} {recall:.4f}  ({per_type_detected[etype]}/{per_type_gold[etype]})")

    log.info(f"\n  ── Privacy Leakage ──")
    log.info(f"  Entity Leakage Rate   : {leakage_rate:.4f}%")
    log.info(f"  Top leaked entities:")
    for ent, cnt in top_leaked:
        log.info(f"    '{ent}'  count={cnt}")

    log.info(f"\n  ── Utility Metrics ──")
    log.info(f"  BERTScore P    : {bert_p:.4f}")
    log.info(f"  BERTScore R    : {bert_r:.4f}")
    log.info(f"  BERTScore F1   : {bert_f1:.4f}")
    log.info(f"  ROUGE-1        : {avg_rouge1:.4f}")
    log.info(f"  ROUGE-2        : {avg_rouge2:.4f}")
    log.info(f"  ROUGE-L        : {avg_rougeL:.4f}")
    log.info(f"  Word Accuracy  : {word_acc:.4f}%")

    log.info(f"\n  ── vs Zero-Shot Baseline (roberta + deberta-mlm) ──")
    log.info(f"  {'Metric':<25} {'Baseline':>10} {'Fine-Tuned':>12} {'Δ':>8}")
    log.info(f"  {'─'*60}")
    baseline = {
        "Masker Recall":    0.7760, "Leakage Rate%":  18.43,
        "BERTScore F1":    88.63,  "ROUGE-1":         0.00,
        "ROUGE-2":          0.00,  "ROUGE-L":         0.00,
        "Word Accuracy%":  32.08,
    }
    finetuned = {
        "Masker Recall":   overall_recall, "Leakage Rate%": leakage_rate,
        "BERTScore F1":   bert_f1,         "ROUGE-1":       avg_rouge1,
        "ROUGE-2":         avg_rouge2,      "ROUGE-L":       avg_rougeL,
        "Word Accuracy%": word_acc,
    }
    higher_is_better = {"Masker Recall", "BERTScore F1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Word Accuracy%"}
    for metric in baseline:
        b = baseline[metric]; f = finetuned[metric]
        delta = f - b
        good = delta > 0 if metric in higher_is_better else delta < 0
        arrow = ("↑" if good else "↓") if abs(delta) > 0.001 else "="
        log.info(f"  {metric:<25} {b:>10.4f} {f:>12.4f} {arrow} {abs(delta):>6.4f}")

    log.info(f"\n  ── Per-Type Recall vs Baseline ──")
    log.info(f"  {'Type':<20} {'Baseline':>10} {'Fine-Tuned':>12} {'Δ':>8}")
    log.info(f"  {'─'*55}")
    base_per_type = {
        "FULLNAME": 0.9709, "LOCATION": 0.8727, "DATE": 0.8162,
        "ORGANIZATION": 0.6821, "ID_NUMBER": 0.6216,
        "NUMBER": 0.2368, "OTHER_PII": 0.0738,
    }
    for etype in sorted(per_type_gold.keys()):
        b  = base_per_type.get(etype, 0.0)
        ft = per_type_detected[etype] / max(per_type_gold[etype], 1)
        delta = ft - b
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        log.info(f"  {etype:<20} {b:>10.4f} {ft:>12.4f} {arrow} {abs(delta):>6.4f}")

    log.info(f"\n  ── Sample Outputs ──")
    for idx, (orig, gold_m, pred_m, anon) in enumerate(samples_display):
        log.info(f"\n  Sample {idx+1}:")
        log.info(f"    Original    : {orig[:200]}")
        log.info(f"    Gold masked : {gold_m[:200]}")
        log.info(f"    Pred masked : {pred_m[:200]}")
        log.info(f"    Anonymized  : {anon[:200]}")

    # ── Save JSON results ──
    eval_results = {
        "masker_path": masker_path, "filler_path": filler_path,
        "n_test_paragraphs": len(tab_test),
        "overall_masker_recall": round(overall_recall, 4),
        "entity_leakage_rate_pct": round(leakage_rate, 4),
        "bert_score_p": round(bert_p, 4),
        "bert_score_r": round(bert_r, 4),
        "bert_score_f1": round(bert_f1, 4),
        "rouge1": round(avg_rouge1, 4),
        "rouge2": round(avg_rouge2, 4),
        "rougeL": round(avg_rougeL, 4),
        "word_accuracy_pct": round(word_acc, 4),
        "per_type_recall": {
            et: round(per_type_detected[et] / max(per_type_gold[et], 1), 4)
            for et in per_type_gold
        },
        "top_leaked_entities": top_leaked,
    }
    res_path = os.path.join(OUTPUT_DIR, "tab_finetuned_full_eval.json")
    with open(res_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    log.info(f"\n  📄 Full results saved: {res_path}")
    log.info(f"  {'═'*68}")

    # ── Upload results to HF Hub as dataset card / file ──
    if PUSH_TO_HUB and HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            _api = HfApi(token=HF_TOKEN)
            # Push JSON
            _api.upload_file(
                path_or_fileobj=res_path,
                path_in_repo="tab_finetuned_full_eval.json",
                repo_id=f"{HF_USERNAME}/pii-tab-eval-results",
                repo_type="dataset",
                commit_message="TAB fine-tuned pipeline evaluation results",
                create_pr=False,
            )
            log.info(f"  📤 Results uploaded to HF Hub: {HF_USERNAME}/pii-tab-eval-results")
        except Exception as _e:
            log.warning(f"  ⚠️  HF upload failed (non-critical): {_e}")

    # ── Also save a human-readable .txt report ──
    txt_path = res_path.replace(".json", ".txt")
    with open(txt_path, "w") as _f:
        _f.write("TAB Fine-Tuned Pipeline — Full Evaluation Report\n")
        _f.write("=" * 60 + "\n")
        import datetime as _dt
        _f.write(f"Run at: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _f.write(f"Masker: {masker_path}\nFiller: {filler_path}\n")
        _f.write("=" * 60 + "\n\n")
        _f.write(f"Overall Masker Recall : {overall_recall:.4f}\n")
        _f.write(f"Entity Leakage Rate   : {leakage_rate:.4f}%\n")
        _f.write(f"BERTScore F1          : {bert_f1:.4f}\n")
        _f.write(f"ROUGE-1               : {avg_rouge1:.4f}\n")
        _f.write(f"ROUGE-2               : {avg_rouge2:.4f}\n")
        _f.write(f"ROUGE-L               : {avg_rougeL:.4f}\n")
        _f.write(f"Word Accuracy         : {word_acc:.4f}%\n\n")
        _f.write("Per-Type Recall:\n")
        for etype in sorted(per_type_gold.keys()):
            r = per_type_detected[etype] / max(per_type_gold[etype], 1)
            _f.write(f"  {etype:<16} {r:.4f}\n")
    log.info(f"  📄 Human-readable report: {txt_path}")
    log.info(f"  💡 Download from Kaggle Output tab → tab_finetuned_full_eval.txt")

    return eval_results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: MAIN EXECUTION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# ## 🚀 Run All Phases

# %%
def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__" or IS_KAGGLE or True:

    if PHASE_1_MASKER:
        phase1_finetune_masker()
        cleanup_gpu()

    if PHASE_2_FILLER:
        phase2_finetune_filler()
        cleanup_gpu()

    if PHASE_3_EVAL:
        phase3_evaluate()
        cleanup_gpu()

    log.info("\n  ✅ All phases complete!")

