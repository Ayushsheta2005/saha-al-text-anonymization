# ==============================================================================
# data.py — Data loading, splitting, and preprocessing
# ==============================================================================
# Handles the AI4Privacy 500K dataset:
#   1. Load from HuggingFace
#   2. Language-stratified 80/10/10 split
#   3. Half-A (NER encoder) / Half-B (Filler) split of train set
#   4. Tokenization & BIO alignment for NER
#   5. Masked→Real text pair creation for the Filler
# ==============================================================================

import os
import json
import random
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset

from config import (
    SEED, DATASET_NAME, DATA_CACHE_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    ENTITY_TYPES, ENTITY_MAP, LABEL2ID, ID2LABEL,
    QUICK_N,
)

random.seed(SEED)
np.random.seed(SEED)

log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_ai4privacy() -> Dataset:
    """Load AI4Privacy PII masking dataset from HuggingFace."""
    log.info(f"Loading dataset: {DATASET_NAME} ...")
    ds = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy", split="train")
    log.info(f"  Loaded {len(ds):,} examples")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Language-Stratified Splitting
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_lang_col(ds: Dataset) -> Optional[str]:
    """Find the language column in the dataset."""
    for col in ("language", "lang", "Language"):
        if col in ds.column_names:
            return col
    return None


def language_stratified_split(ds: Dataset) -> Dict[str, Dataset]:
    """
    Split dataset into train (80%), val (10%), test (10%) — language-stratified.
    Each language is represented proportionally in every split.

    Returns dict with keys: 'train', 'val_encoder', 'val_filler', 'test'
    The val set is further split 50/50 for encoder and filler validation.
    """
    log.info("Performing language-stratified split ...")
    df = ds.to_pandas()
    lang_col = _detect_lang_col(ds)
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
        n_train = n - n_test - n_val

        test_idx.extend(idx[:n_test])
        val_enc_idx.extend(idx[n_test:n_test + n_val_enc])
        val_fill_idx.extend(idx[n_test + n_val_enc:n_test + n_val_enc + n_val_fill])
        train_idx.extend(idx[n_test + n_val:])

    # Shuffle all splits
    for lst in (train_idx, val_enc_idx, val_fill_idx, test_idx):
        random.shuffle(lst)

    splits = {
        "train": ds.select(train_idx),
        "val_encoder": ds.select(val_enc_idx),
        "val_filler": ds.select(val_fill_idx),
        "test": ds.select(test_idx),
    }

    log.info(f"  Train:       {len(splits['train']):,}")
    log.info(f"  Val Encoder: {len(splits['val_encoder']):,}")
    log.info(f"  Val Filler:  {len(splits['val_filler']):,}")
    log.info(f"  Test:        {len(splits['test']):,}")

    # Log language distribution
    for split_name, split_ds in splits.items():
        split_df = split_ds.to_pandas()
        if lang_col in split_df.columns:
            lang_counts = split_df[lang_col].value_counts()
            log.info(f"  {split_name} — {len(lang_counts)} languages, "
                     f"top: {dict(lang_counts.head(5))}")

    return splits


def split_train_halves(train_ds: Dataset) -> Tuple[Dataset, Dataset]:
    """
    Split the train set into Half-A (NER encoder) and Half-B (Filler).

    Half-A: used to train the NER encoder on (source_text → entity_labels)
    Half-B: used to train the filler on (masked_text → real_text)

    The privacy wall: encoder never sees Half-B's entities, filler never sees Half-A's.
    """
    n = len(train_ds)
    indices = list(range(n))
    random.shuffle(indices)
    mid = n // 2

    half_a = train_ds.select(indices[:mid])
    half_b = train_ds.select(indices[mid:])

    log.info(f"  Half-A (NER):    {len(half_a):,} examples")
    log.info(f"  Half-B (Filler): {len(half_b):,} examples")

    return half_a, half_b


def quick_subsample(ds: Dataset, n: int = QUICK_N) -> Dataset:
    """Subsample dataset for quick debugging runs."""
    if len(ds) <= n:
        return ds
    return ds.select(random.sample(range(len(ds)), n))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Extract Tokens + BIO Labels from AI4Privacy examples
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tokens_and_labels(example: Dict) -> Tuple[List[str], List[str]]:
    """
    Parse an AI4Privacy example into (word_tokens, bio_labels).

    Handles the privacy_mask field which contains entity span annotations.
    Maps fine-grained labels (FIRSTNAME, CITY, ...) to coarse types (PERSON, LOC, ...).
    """
    # If pre-tokenized
    if "tokens" in example and "ner_tags" in example:
        return example["tokens"], example["ner_tags"]

    text = example.get("source_text", example.get("text", ""))
    masks = example.get("privacy_mask", [])

    if not masks:
        tokens = text.split()
        return tokens, ["O"] * len(tokens)

    # Sort spans by position
    spans = sorted(masks, key=lambda m: m.get("start", m.get("offset", 0)))
    tokens, labels = [], []
    pos = 0

    for span in spans:
        start = span.get("start", span.get("offset", 0))
        end = span.get("end", start + span.get("length", len(span.get("value", ""))))
        label = span.get("label", span.get("entity_type", "O")).upper().replace(" ", "_")
        value = span.get("value", text[start:end])

        # Non-entity text before this span
        if pos < start:
            before = text[pos:start].split()
            tokens.extend(before)
            labels.extend(["O"] * len(before))

        # Entity tokens
        entity_words = value.split()
        if entity_words:
            coarse = ENTITY_MAP.get(label, label)
            if coarse in ENTITY_TYPES:
                tokens.append(entity_words[0])
                labels.append(f"B-{coarse}")
                for w in entity_words[1:]:
                    tokens.append(w)
                    labels.append(f"I-{coarse}")
            else:
                # Unknown entity type — treat as non-entity
                tokens.extend(entity_words)
                labels.extend(["O"] * len(entity_words))

        pos = end

    # Remaining text after last span
    if pos < len(text):
        remaining = text[pos:].split()
        tokens.extend(remaining)
        labels.extend(["O"] * len(remaining))

    return tokens, labels


def get_source_text(example: Dict) -> str:
    """Get the raw source text string from an example."""
    if "source_text" in example:
        return example["source_text"]
    if "text" in example:
        return example["text"]
    tokens, _ = extract_tokens_and_labels(example)
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NER Data Preparation (Tokenize + Align BIO labels)
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_and_align_ner(examples, tokenizer, max_length=256):
    """
    Tokenize text and align BIO labels to subword tokens.

    Handles the subword→word alignment: first subword gets the word's label,
    continuation subwords get I- (if entity) or inherit. Padding/special tokens get -100.
    """
    # Collect all tokens and labels
    key = next(k for k in ("source_text", "text", "tokens") if k in examples)
    all_tokens, all_labels = [], []

    for i in range(len(examples[key])):
        ex = {k: v[i] for k, v in examples.items()}
        toks, labs = extract_tokens_and_labels(ex)
        all_tokens.append(toks)
        all_labels.append(labs)

    # Tokenize with word-level pre-splitting
    enc = tokenizer(
        all_tokens, truncation=True, max_length=max_length,
        padding="max_length", is_split_into_words=True,
    )

    # Align labels to subword tokens
    aligned_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = enc.word_ids(batch_index=i)
        label_ids = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)  # special tokens
            elif wid != prev_wid:
                # First subword of a new word — use the word's label
                lbl = labels[wid] if wid < len(labels) else "O"
                label_ids.append(LABEL2ID.get(lbl, 0))
            else:
                # Continuation subword — use I- variant
                lbl = labels[wid] if wid < len(labels) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(LABEL2ID.get(lbl, 0))
            prev_wid = wid
        aligned_labels.append(label_ids)

    enc["labels"] = aligned_labels
    return enc


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Filler Data Preparation (masked_text → real_text pairs)
# ═══════════════════════════════════════════════════════════════════════════════

def create_filler_pair(example: Dict) -> Dict[str, str]:
    """
    Create a (masked_text → real_text) pair for training the filler.

    Input:  "Replace PII: [PERSON] from [LOC] called at [PHONE]."
    Target: "Ayush Sheta from Gujarat called at +91 98765 43210."

    The target is the REAL text from Half-B (NOT Faker-generated).
    This preserves natural language patterns and cultural coherence.
    """
    tokens, labels = extract_tokens_and_labels(example)

    # Build masked version: replace entity spans with [TYPE] placeholders
    masked_words = []
    prev_entity = None
    for tok, lab in zip(tokens, labels):
        if lab == "O":
            masked_words.append(tok)
            prev_entity = None
        elif lab.startswith("B-"):
            etype = lab[2:]
            masked_words.append(f"[{etype}]")
            prev_entity = etype
        elif lab.startswith("I-") and prev_entity:
            # Continuation of entity — skip (already have placeholder)
            pass
        else:
            masked_words.append(tok)
            prev_entity = None

    masked_text = " ".join(masked_words)
    original_text = " ".join(tokens)

    return {
        "input_text": f"Replace PII placeholders with realistic entities: {masked_text}",
        "target_text": original_text,
    }


def tokenize_filler_pairs(examples, tokenizer, max_input_length=256, max_target_length=256):
    """Tokenize input/target pairs for the filler seq2seq model."""
    enc = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )
    # Replace pad token ids with -100 so they're ignored in loss
    enc["labels"] = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in labels["input_ids"]
    ]
    return enc


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Data Preparation Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_all_data(quick: bool = False) -> Dict:
    """
    Full data preparation pipeline:
    1. Load AI4Privacy from HuggingFace
    2. Language-stratified split
    3. Split train into Half-A / Half-B

    Returns dict with all splits ready for training.
    """
    # Check cache
    cache_file = os.path.join(DATA_CACHE_DIR, "splits_info.json")
    ds = load_ai4privacy()

    # Split
    splits = language_stratified_split(ds)
    half_a, half_b = split_train_halves(splits["train"])

    if quick:
        log.info(f"  QUICK MODE: subsampling to {QUICK_N} per split")
        half_a = quick_subsample(half_a, QUICK_N)
        half_b = quick_subsample(half_b, QUICK_N)
        splits["val_encoder"] = quick_subsample(splits["val_encoder"], QUICK_N // 5)
        splits["val_filler"] = quick_subsample(splits["val_filler"], QUICK_N // 5)
        splits["test"] = quick_subsample(splits["test"], QUICK_N // 5)

    result = {
        "half_a": half_a,
        "half_b": half_b,
        "val_encoder": splits["val_encoder"],
        "val_filler": splits["val_filler"],
        "test": splits["test"],
    }

    # Save split info
    info = {k: len(v) for k, v in result.items()}
    with open(cache_file, "w") as f:
        json.dump(info, f, indent=2)
    log.info(f"  Split info saved: {cache_file}")

    return result
