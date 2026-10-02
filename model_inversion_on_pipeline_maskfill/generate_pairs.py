#!/usr/bin/env python3
"""
generate_pairs.py
=================
Generates two (original, anonymized) parallel datasets for model-inversion
attack research on the pipeline mask-fill anonymisation approach.

Combo 1 — encoder + encoder (MLM filler):
  Stage 1: pii-ner-roberta  → BIO token classification → identify PII spans
  Stage 2: pii-ner-filler_deberta-filler (MLM) → fill-mask each entity span
           with a single [MASK] token, predict replacement

Combo 2 — encoder + decoder (seq2seq filler):
  Stage 1: pii-ner-roberta  → same as above
  Stage 2: pii-ner-filler_bart-base (seq2seq) → replace spans with [TYPE]
           placeholders, decode anonymized sentence

Source:  ai4privacy/pii-masking-400k  (English, 40 000 examples)
Output:  output/combo1_roberta_deberta.jsonl
         output/combo2_roberta_bart.jsonl

Run:
    python generate_pairs.py [--combo {1,2,both}] [--limit N]
"""

import gc
import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

# ── project config ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NER_MODEL_ID, MLM_FILLER_ID, SEQ2SEQ_FILLER_ID,
    SOURCE_FILE, TARGET_N,
    NER_MAX_LEN, MLM_MAX_LEN, S2S_MAX_LEN,
    GEN_MAX_TOKENS, GEN_NUM_BEAMS,
    NER_BATCH, MLM_BATCH, S2S_BATCH,
    OUTPUT_DIR, COMBO1_FILE, COMBO2_FILE,
    MLM_MASK_TOKEN, ID2LABEL,
)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "generate_pairs.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NER_CACHE_FILE = os.path.join(OUTPUT_DIR, "ner_intermediate.jsonl")
log.info(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_english_sentences(n: int) -> list[dict]:
    """Load n sentences from the local english_entries_all.jsonl.
    Each row has 'original_text' and 'id' fields.
    """
    log.info(f"Loading local source: {SOURCE_FILE}")
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    records = []
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = row.get("original_text") or row.get("source_text") or row.get("text") or ""
            if txt.strip():
                records.append({"source_text": txt.strip(), "id": str(row.get("id", len(records)))})
            if len(records) >= n:
                break

    log.info(f"  Loaded {len(records):,} sentences from local file")
    if len(records) < n:
        log.warning(f"  Only found {len(records):,} < requested {n:,}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
# NER — shared for both combos
# ═══════════════════════════════════════════════════════════════════════════

def load_ner_model():
    log.info(f"Loading NER model: {NER_MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(NER_MODEL_ID, use_fast=True)
    mdl = AutoModelForTokenClassification.from_pretrained(NER_MODEL_ID)
    mdl.eval().to(DEVICE)
    # Prefer label mapping from the saved model config if available
    if hasattr(mdl.config, "id2label") and mdl.config.id2label:
        id2lbl = {int(k): v for k, v in mdl.config.id2label.items()}
    else:
        id2lbl = ID2LABEL
    log.info(f"  NER labels: {len(id2lbl)} classes")
    return mdl, tok, id2lbl


@torch.no_grad()
def ner_predict_batch(
    texts: list[str],
    model,
    tokenizer,
    id2lbl: dict,
    max_len: int = NER_MAX_LEN,
) -> list[list[tuple]]:
    """
    Batch NER prediction. Returns list of (word, bio_tag) lists, one per text.
    """
    batch_words = [t.split() for t in texts]
    enc = tokenizer(
        batch_words,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        is_split_into_words=True,
        padding=True,
    ).to(DEVICE)

    logits = model(**enc).logits          # (B, seq_len, num_labels)
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

    results = []
    for i, words in enumerate(batch_words):
        word_ids = enc.word_ids(batch_index=i)
        word_tags: dict[int, str] = {}
        for token_idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid not in word_tags:
                word_tags[wid] = id2lbl.get(pred_ids[i][token_idx], "O")
        results.append([(words[wid], word_tags.get(wid, "O")) for wid in range(len(words))])
    return results


@torch.no_grad()
def ner_predict(text: str, model, tokenizer, id2lbl: dict, max_len: int = NER_MAX_LEN):
    """Single-sentence wrapper kept for compatibility."""
    return ner_predict_batch([text], model, tokenizer, id2lbl, max_len)[0]


def extract_entity_spans(word_tags: list[tuple]) -> list[dict]:
    """
    Collapse consecutive B-/I- tokens of the same type into spans.
    Returns list of dicts with keys: start_word, end_word (exclusive),
    entity_type, surface_text.
    """
    spans = []
    i = 0
    while i < len(word_tags):
        word, tag = word_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(word_tags) and word_tags[j][1] == f"I-{etype}":
                j += 1
            spans.append({
                "start_word": i,
                "end_word":   j,
                "entity_type": etype,
                "surface":    " ".join(w for w, _ in word_tags[i:j]),
            })
            i = j
        else:
            i += 1
    return spans


# ═══════════════════════════════════════════════════════════════════════════
# Combo 1 — MLM filler (DeBERTa)
# ═══════════════════════════════════════════════════════════════════════════

def load_mlm_filler():
    log.info(f"Loading MLM filler: {MLM_FILLER_ID}")
    tok = AutoTokenizer.from_pretrained(MLM_FILLER_ID, use_fast=True)
    mdl = AutoModelForMaskedLM.from_pretrained(MLM_FILLER_ID)
    mdl.eval().to(DEVICE)
    return mdl, tok


@torch.no_grad()
def fill_masks_deberta_batch(
    masked_texts: list[str],
    model,
    tokenizer,
    max_len: int = MLM_MAX_LEN,
) -> list[str]:
    """
    Batch MLM fill. For each sentence, predicts top-1 token at every [MASK]
    position and returns the reconstructed sentences.
    """
    mask_token = tokenizer.mask_token
    mask_id    = tokenizer.mask_token_id

    enc = tokenizer(
        masked_texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,
    ).to(DEVICE)

    logits = model(**enc).logits          # (B, seq_len, vocab_size)
    pred_ids = logits.argmax(dim=-1).cpu()  # (B, seq_len)
    input_ids = enc["input_ids"].cpu()     # (B, seq_len)

    outputs = []
    for i, text in enumerate(masked_texts):
        if mask_token not in text:
            outputs.append(text)
            continue
        result = text
        mask_positions = (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].tolist()
        for pos in mask_positions:
            replacement = tokenizer.decode([pred_ids[i, pos].item()]).strip()
            result = result.replace(mask_token, replacement, 1)
        outputs.append(result)
    return outputs


@torch.no_grad()
def fill_masks_deberta(
    masked_text: str,
    model,
    tokenizer,
    max_len: int = MLM_MAX_LEN,
) -> str:
    """Single-sentence wrapper kept for compatibility."""
    return fill_masks_deberta_batch([masked_text], model, tokenizer, max_len)[0]


# ── Pre-masking helpers (run during NER phase, before filler is loaded) ────

def _build_masked_mlm(word_tags: list, spans: list, mask_token: str) -> str:
    """Build [MASK]-substituted string for the DeBERTa MLM filler."""
    words = [w for w, _ in word_tags]
    masked = list(words)
    offset = 0
    for span in spans:
        s = span["start_word"] + offset
        e = span["end_word"]   + offset
        masked[s:e] = [mask_token]
        offset -= (span["end_word"] - span["start_word"] - 1)
    return " ".join(masked)


def _build_masked_s2s(word_tags: list, spans: list) -> str:
    """Build [TYPE]-placeholder prompt for the BART seq2seq filler."""
    words = [w for w, _ in word_tags]
    masked = list(words)
    offset = 0
    for span in spans:
        s = span["start_word"] + offset
        e = span["end_word"]   + offset
        masked[s:e] = [f"[{span['entity_type']}]"]
        offset -= (span["end_word"] - span["start_word"] - 1)
    return f"Replace PII placeholders with realistic fake entities: {' '.join(masked)}"


def free_model(model):
    """Delete a model from memory and free the GPU cache."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase1_ner(
    sentences: list,
    ner_model,
    ner_tok,
    id2lbl: dict,
    mlm_mask_token: str = MLM_MASK_TOKEN,
    batch_size: int = NER_BATCH,
) -> list:
    """
    Phase 1: run NER on every sentence; cache (masked_mlm, masked_s2s) to
    NER_CACHE_FILE.  Returns the full list of intermediate records.
    Resumable: skips rows already in NER_CACHE_FILE.
    """
    done_ids = load_done_ids(NER_CACHE_FILE)
    log.info(f"Phase 1 NER — {len(sentences):,} sentences | already cached: {len(done_ids):,}")

    # Re-load records that were cached in a previous run
    cached = []
    if os.path.exists(NER_CACHE_FILE):
        with open(NER_CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cached.append(json.loads(line))

    errors = 0
    pending = [
        (i, ex) for i, ex in enumerate(sentences)
        if (ex.get("id") or f"row_{i}") not in done_ids
    ]
    with tqdm(total=len(sentences), initial=len(done_ids), desc="Phase1-NER") as pbar:
        for chunk_start in range(0, len(pending), batch_size):
            chunk = pending[chunk_start: chunk_start + batch_size]
            srcs  = [ex["source_text"] for _, ex in chunk]
            try:
                all_word_tags = ner_predict_batch(srcs, ner_model, ner_tok, id2lbl)
            except Exception as exc:
                log.warning(f"  NER batch failed: {exc} — falling back to per-sample")
                all_word_tags = []
                for src in srcs:
                    try:
                        all_word_tags.append(ner_predict(src, ner_model, ner_tok, id2lbl))
                    except Exception as e2:
                        log.warning(f"    per-sample also failed: {e2}")
                        all_word_tags.append([])
                        errors += 1

            for (i, ex), word_tags in zip(chunk, all_word_tags):
                uid = ex.get("id") or f"row_{i}"
                src = ex["source_text"]
                try:
                    spans      = extract_entity_spans(word_tags)
                    masked_mlm = _build_masked_mlm(word_tags, spans, mlm_mask_token) if spans else src
                    masked_s2s = (
                        _build_masked_s2s(word_tags, spans)
                        if spans
                        else f"Replace PII placeholders with realistic fake entities: {src}"
                    )
                except Exception as exc:
                    log.warning(f"  span-build row {i} failed: {exc}")
                    masked_mlm = masked_s2s = src
                    errors += 1

                record = {
                    "id": uid, "original": src,
                    "masked_mlm": masked_mlm, "masked_s2s": masked_s2s,
                }
                append_jsonl(NER_CACHE_FILE, record)
                cached.append(record)
            pbar.update(len(chunk))

    log.info(f"Phase 1 NER done. Errors: {errors}  Cache: {NER_CACHE_FILE}")
    return cached


# ═══════════════════════════════════════════════════════════════════════════
# Combo 2 — seq2seq filler (BART)
# ═══════════════════════════════════════════════════════════════════════════

def load_seq2seq_filler():
    log.info(f"Loading seq2seq filler: {SEQ2SEQ_FILLER_ID}")
    tok = AutoTokenizer.from_pretrained(SEQ2SEQ_FILLER_ID, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_FILLER_ID)
    mdl.eval().to(DEVICE)
    return mdl, tok


_BART_OUTPUT_PREFIX = "Placeholders with realistic fake entities: "

def _strip_bart_prefix(text: str) -> str:
    """The BART model was trained to echo a prefix in its output; strip it."""
    if text.startswith(_BART_OUTPUT_PREFIX):
        return text[len(_BART_OUTPUT_PREFIX):]
    return text

def _bart_generate_safe(bart_model, bart_tok, prompts: list, device) -> list[str]:
    """
    Generate with BART. If a batch causes OOM, split it in half and retry
    recursively until batch size reaches 1. This avoids silently writing
    original (un-anonymized) text on OOM.
    Also strips the model's training-time output prefix from every result.
    """
    if not prompts:
        return []
    enc = bart_tok(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=S2S_MAX_LEN,
        padding=True,
    ).to(device)
    try:
        out_ids = bart_model.generate(
            **enc,
            max_new_tokens=GEN_MAX_TOKENS,
            num_beams=GEN_NUM_BEAMS,
        )
        return [_strip_bart_prefix(bart_tok.decode(ids, skip_special_tokens=True)) for ids in out_ids]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            # Single sentence still OOMs — move it to CPU for this one call
            log.warning("  Single-sentence BART OOM — running on CPU for this sample")
            enc_cpu = {k: v.cpu() for k, v in enc.items()}
            bart_model.cpu()
            out_ids = bart_model.generate(**enc_cpu, max_new_tokens=GEN_MAX_TOKENS,
                                          num_beams=GEN_NUM_BEAMS)
            result = [_strip_bart_prefix(bart_tok.decode(ids, skip_special_tokens=True)) for ids in out_ids]
            bart_model.to(device)
            torch.cuda.empty_cache()
            return result
        mid = len(prompts) // 2
        left  = _bart_generate_safe(bart_model, bart_tok, prompts[:mid], device)
        right = _bart_generate_safe(bart_model, bart_tok, prompts[mid:], device)
        return left + right




# ═══════════════════════════════════════════════════════════════════════════
# Resumable writer helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_done_ids(path: str) -> set:
    """Return set of source IDs already written to the output file."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row.get("id", ""))
            except json.JSONDecodeError:
                pass
    return done


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main generation loops
# ═══════════════════════════════════════════════════════════════════════════

def run_combo1(ner_results: list, out_file: str, batch_size: int = MLM_BATCH):
    """
    Phase 2a: DeBERTa MLM filler — batched.
    GPU only holds the MLM model; NER model has already been freed.
    """
    log.info("=" * 60)
    log.info("COMBO 1 — roberta-NER + DeBERTa-MLM filler")
    log.info("=" * 60)

    mlm_model, mlm_tok = load_mlm_filler()
    done_ids = load_done_ids(out_file)
    log.info(f"  Already done: {len(done_ids):,}  Remaining: {len(ner_results) - len(done_ids):,}")

    errors = 0
    pending = [item for item in ner_results if item["id"] not in done_ids]
    with tqdm(total=len(ner_results), initial=len(done_ids), desc="Combo1-fill") as pbar:
        for chunk_start in range(0, len(pending), batch_size):
            chunk = pending[chunk_start: chunk_start + batch_size]
            try:
                anons = fill_masks_deberta_batch(
                    [item["masked_mlm"] for item in chunk], mlm_model, mlm_tok
                )
            except Exception as exc:
                log.warning(f"  MLM batch failed: {exc} — falling back to originals")
                anons = [item["original"] for item in chunk]
                errors += len(chunk)

            for item, anon in zip(chunk, anons):
                append_jsonl(out_file, {
                    "id":         item["id"],
                    "original":   item["original"],
                    "anonymized": anon,
                    "combo":      "roberta-ner+deberta-mlm",
                })
            pbar.update(len(chunk))

    free_model(mlm_model)
    log.info(f"Combo1 done. Errors: {errors}  Output: {out_file}")


def run_combo2(ner_results: list, out_file: str, batch_size: int = S2S_BATCH):
    """
    Phase 2b: BART seq2seq filler — batched.
    GPU only holds the BART model; NER and MLM models have already been freed.
    """
    log.info("=" * 60)
    log.info("COMBO 2 — roberta-NER + BART seq2seq filler")
    log.info("=" * 60)

    bart_model, bart_tok = load_seq2seq_filler()
    done_ids = load_done_ids(out_file)
    log.info(f"  Already done: {len(done_ids):,}  Remaining: {len(ner_results) - len(done_ids):,}")

    pending = [item for item in ner_results if item["id"] not in done_ids]
    with tqdm(total=len(ner_results), initial=len(done_ids), desc="Combo2-fill") as pbar:
        for chunk_start in range(0, len(pending), batch_size):
            chunk   = pending[chunk_start: chunk_start + batch_size]
            prompts = [item["masked_s2s"] for item in chunk]
            anons = _bart_generate_safe(bart_model, bart_tok, prompts, DEVICE)

            for item, anon in zip(chunk, anons):
                append_jsonl(out_file, {
                    "id":         item["id"],
                    "original":   item["original"],
                    "anonymized": anon,
                    "combo":      "roberta-ner+bart-seq2seq",
                })
            pbar.update(len(chunk))

    free_model(bart_model)
    log.info(f"Combo2 done. Output: {out_file}")


# ═══════════════════════════════════════════════════════════════════════════
# Upload to HuggingFace
# ═══════════════════════════════════════════════════════════════════════════

def upload_to_hub(jsonl_path: str, repo_id: str, hf_token: str):
    """Convert JSONL → HuggingFace Dataset and push_to_hub."""
    from datasets import Dataset
    from huggingface_hub import login

    log.info(f"Uploading {jsonl_path} → {repo_id} …")
    login(token=hf_token, add_to_git_credential=False)

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        log.warning("  No records to upload, skipping.")
        return

    ds = Dataset.from_list(records)
    ds.push_to_hub(
        repo_id,
        token=hf_token,
        private=False,
        commit_message=f"Add {len(records):,} (original, anonymized) pairs",
    )
    log.info(f"  Uploaded {len(records):,} rows → https://huggingface.co/datasets/{repo_id}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combo", choices=["1", "2", "both"], default="both",
                   help="Which pipeline combo to run (default: both)")
    p.add_argument("--limit", type=int, default=TARGET_N,
                   help=f"Max number of sentences (default: {TARGET_N})")
    p.add_argument("--upload", action="store_true",
                   help="Upload finished datasets to HuggingFace after generation")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (falls back to HF_TOKEN env var)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from config import HF_TOKEN, HF_REPO_COMBO1, HF_REPO_COMBO2
    hf_token = args.hf_token or HF_TOKEN

    # ── 1. Source sentences ─────────────────────────────────────────────
    sentences = load_english_sentences(args.limit)

    # ── 2. Phase 1 — NER (one model on GPU) ────────────────────────────
    # Run NER over all sentences, persist masked strings to NER_CACHE_FILE,
    # then free the GPU before the filler is loaded.
    ner_model, ner_tok, id2lbl = load_ner_model()
    ner_results = phase1_ner(sentences, ner_model, ner_tok, id2lbl)
    free_model(ner_model)
    del ner_model, ner_tok
    log.info(f"GPU freed after NER. Processing {len(ner_results):,} records.")

    # ── 3. Phase 2a — Combo 1 fill (DeBERTa MLM) ───────────────────────
    if args.combo in ("1", "both"):
        run_combo1(ner_results, COMBO1_FILE)
        if args.upload:
            upload_to_hub(COMBO1_FILE, HF_REPO_COMBO1, hf_token)

    # ── 4. Phase 2b — Combo 2 fill (BART seq2seq) ──────────────────────
    if args.combo in ("2", "both"):
        run_combo2(ner_results, COMBO2_FILE)
        if args.upload:
            upload_to_hub(COMBO2_FILE, HF_REPO_COMBO2, hf_token)

    log.info("All done.")


if __name__ == "__main__":
    main()
