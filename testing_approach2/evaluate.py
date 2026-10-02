"""
Comprehensive Evaluation — Approach 2: Pipeline (Masker + Filler)
=================================================================
Tests all 6 masker×filler combinations loaded directly from HuggingFace:

  Maskers (NER / Token Classification):
    distilroberta  →  Xyren2005/pii-ner-distilroberta      (82M, DistilRoBERTa)
    roberta        →  Xyren2005/pii-ner-roberta             (125M, RoBERTa-base)
    deberta        →  Xyren2005/pii-ner-encoder_deberta     (183M, DeBERTa-v3)

  Fillers:
    bart-base      →  Xyren2005/pii-ner-filler_bart-base       (139M, BART Seq2Seq)
    deberta-mlm    →  Xyren2005/pii-ner-filler_deberta-filler  (183M, DeBERTa MLM)

Pipeline:
  text  ──[Masker]──►  masked text (e.g. "[PERSON] works at [ORGANIZATION]")
                ──[Filler]──►  anonymized text

Evaluates on:
  1. Test split (Seq2Seq_model/data_splits/test.jsonl) — TEST_SET_LIMIT samples
  2. Curated eval examples (Seq2Seq_model/eval_examples.jsonl)

Metrics:
  - Entity Leakage Rate (privacy, primary)
  - Masker Detection Rate (intermediate masking quality)
  - BLEU (1, 2, 4, overall)
  - ROUGE (1, 2, L)
  - BERTScore (P, R, F1)
  - Exact Match, Word Accuracy

Usage:
    python evaluate.py
"""

import os
import sys
import gc
import re
import json
import time
from datetime import datetime
from collections import Counter

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
)

from config import (
    MASKER_CONFIGS,
    FILLER_CONFIGS,
    COMBOS,
    TEST_DATA_PATH,
    EVAL_EXAMPLES_PATH,
    RESULTS_DIR,
    RESULTS_JSON,
    RESULTS_TXT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_BEAMS,
    EVAL_BATCH_SIZE,
    TEST_SET_LIMIT,
)
from utils import (
    compute_all_metrics,
    compute_masker_detection_rate,
    count_parameters,
    aggressive_cleanup,
    format_time,
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(limit=None):
    """Load test split: original_text, anonymized_text, entity_texts."""
    if not os.path.exists(TEST_DATA_PATH):
        print(f"  [ERROR] test.jsonl not found at {TEST_DATA_PATH}")
        return []
    examples = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]
    return examples


def load_eval_examples():
    """Load curated eval examples (difficulty-graded, no reference anonymized text)."""
    if not os.path.exists(EVAL_EXAMPLES_PATH):
        print(f"  [WARNING] eval_examples.jsonl not found at {EVAL_EXAMPLES_PATH}")
        return []
    examples = []
    with open(EVAL_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    # Sort by difficulty
    order = {"easy": 0, "medium": 1, "hard": 2}
    examples.sort(key=lambda e: (order.get(e.get("difficulty", "hard"), 99), e.get("id", "")))
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING / UNLOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_masker(masker_key: str):
    """Load a NER masker model + tokenizer from HuggingFace."""
    cfg = MASKER_CONFIGS[masker_key]
    print(f"    Loading masker '{masker_key}' from {cfg['model_id']} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float32
    )
    model = model.to(DEVICE)
    model.eval()
    params = count_parameters(model)
    print(f"    Masker loaded: {params['total_millions']}M params")
    return model, tokenizer


def load_filler(filler_key: str):
    """Load a filler model (Seq2Seq or MLM) + tokenizer from HuggingFace."""
    cfg = FILLER_CONFIGS[filler_key]
    print(f"    Loading filler '{filler_key}' from {cfg['model_id']} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if cfg["filler_type"] == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float32
        )
    else:  # mlm
        model = AutoModelForMaskedLM.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float32
        )
    model = model.to(DEVICE)
    model.eval()
    params = count_parameters(model)
    print(f"    Filler loaded:  {params['total_millions']}M params")
    return model, tokenizer


def unload(model, tokenizer):
    del model, tokenizer
    aggressive_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# MASKER INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_masker(text: str, model, tokenizer) -> tuple:
    """
    Run NER masker on text.

    Returns:
        masked_text  (str)  – text with entity spans replaced by [ENTITY_TYPE]
        entity_spans (list) – list of (entity_type, original_span_text) tuples
    """
    words = text.split()
    if not words:
        return text, []

    # Tokenize with word-alignment (same convention as training).
    # Keep the BatchEncoding on CPU first so we can call .word_ids(), then
    # move tensors to DEVICE for inference.
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=False,
    )
    word_ids_list = encoding.word_ids(batch_index=0)

    # Move to device for model inference
    model_inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
    outputs = model(**model_inputs)
    predictions = outputs.logits[0].argmax(dim=-1).cpu().numpy()

    # Map each word to its predicted label (first subtoken wins)
    word_labels: dict = {}
    for pos, wid in enumerate(word_ids_list):
        if wid is None:
            continue
        if wid not in word_labels:
            word_labels[wid] = model.config.id2label.get(int(predictions[pos]), "O")

    # Build masked text by grouping B-/I- spans
    masked_words = []
    entity_spans = []
    i = 0
    while i < len(words):
        label = word_labels.get(i, "O")
        if label.startswith("B-"):
            etype = label[2:]
            span_words = [words[i]]
            j = i + 1
            while j < len(words) and word_labels.get(j, "O") == f"I-{etype}":
                span_words.append(words[j])
                j += 1
            entity_spans.append((etype, " ".join(span_words)))
            masked_words.append(f"[{etype}]")
            i = j
        elif label.startswith("I-"):
            # Stray I- without B- (model imperfection) — treat as normal token
            masked_words.append(words[i])
            i += 1
        else:
            masked_words.append(words[i])
            i += 1

    return " ".join(masked_words), entity_spans


@torch.no_grad()
def run_masker_batch(texts: list, model, tokenizer) -> list:
    """
    Batched NER masker. Returns a list of (masked_text, entity_spans) tuples,
    one per input text.
    """
    results = []
    # Tokenize all texts in one padded batch
    words_list = [t.split() for t in texts]
    # Empty texts pass-through
    non_empty_idx = [i for i, w in enumerate(words_list) if w]
    if not non_empty_idx:
        return [(t, []) for t in texts]

    non_empty_words = [words_list[i] for i in non_empty_idx]

    encoding = tokenizer(
        non_empty_words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
    )
    # word_ids must be collected BEFORE moving tensors to device
    word_ids_per_example = [encoding.word_ids(batch_index=b) for b in range(len(non_empty_words))]

    model_inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
    outputs = model(**model_inputs)
    all_logits = outputs.logits  # (batch, seq, labels)

    batch_results = []
    for b_idx, (words, word_ids_list) in enumerate(zip(non_empty_words, word_ids_per_example)):
        predictions = all_logits[b_idx].argmax(dim=-1).cpu().numpy()

        word_labels: dict = {}
        for pos, wid in enumerate(word_ids_list):
            if wid is None:
                continue
            if wid not in word_labels:
                word_labels[wid] = model.config.id2label.get(int(predictions[pos]), "O")

        masked_words = []
        entity_spans = []
        i = 0
        while i < len(words):
            label = word_labels.get(i, "O")
            if label.startswith("B-"):
                etype = label[2:]
                span_words = [words[i]]
                j = i + 1
                while j < len(words) and word_labels.get(j, "O") == f"I-{etype}":
                    span_words.append(words[j])
                    j += 1
                entity_spans.append((etype, " ".join(span_words)))
                masked_words.append(f"[{etype}]")
                i = j
            elif label.startswith("I-"):
                masked_words.append(words[i])
                i += 1
            else:
                masked_words.append(words[i])
                i += 1

        batch_results.append((" ".join(masked_words), entity_spans))

    # Re-insert empty-text results at original positions
    full_results = [(t, []) for t in texts]
    for orig_idx, res in zip(non_empty_idx, batch_results):
        full_results[orig_idx] = res
    return full_results


# ─────────────────────────────────────────────────────────────────────────────
# FILLER INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_filler_seq2seq(masked_text: str, model, tokenizer, prefix: str) -> str:
    """BART / Seq2Seq filler: generate anonymized text from masked template."""
    return run_filler_seq2seq_batch([masked_text], model, tokenizer, prefix)[0]


@torch.no_grad()
def run_filler_seq2seq_batch(masked_texts: list, model, tokenizer, prefix: str) -> list:
    """Batched BART filler. Returns list of anonymized strings."""
    prompts = [prefix + t for t in masked_texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    gen_ids = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=NUM_BEAMS,
        early_stopping=True,
    )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]


@torch.no_grad()
def run_filler_mlm(masked_text: str, model, tokenizer) -> str:
    """DeBERTa MLM filler (single-example wrapper)."""
    return run_filler_mlm_batch([masked_text], model, tokenizer)[0]


@torch.no_grad()
def run_filler_mlm_batch(masked_texts: list, model, tokenizer) -> list:
    """
    Batched DeBERTa MLM filler. Replaces each [ENTITY_TYPE] placeholder with
    the tokenizer mask token, runs one batched forward pass, then fills each
    mask position with the argmax prediction.
    """
    mask_tok = tokenizer.mask_token

    texts_with_masks = [re.sub(r"\[([A-Z][A-Z_0-9]*)\]", mask_tok, t) for t in masked_texts]

    inputs = tokenizer(
        texts_with_masks,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    mask_token_id = tokenizer.convert_tokens_to_ids(mask_tok)
    outputs = model(**inputs)
    logits = outputs.logits  # (batch, seq, vocab)

    results = []
    for b in range(len(masked_texts)):
        input_ids = inputs["input_ids"][b]
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            results.append(tokenizer.decode(input_ids, skip_special_tokens=True))
            continue

        filled_ids = input_ids.clone()
        for pos in mask_positions:
            pred_id = logits[b][pos].argmax().item()
            filled_ids[pos] = pred_id
        results.append(tokenizer.decode(filled_ids, skip_special_tokens=True))

    return results


def run_pipeline(text: str, masker_model, masker_tok, filler_model, filler_tok,
                 filler_cfg: dict) -> tuple:
    """
    Full pipeline: text → masked text → anonymized text.

    Returns:
        anonymized_text  (str)
        masked_text      (str)   intermediate output
        entity_spans     (list)  detected (entity_type, original_span) pairs
    """
    masked_text, entity_spans = run_masker(text, masker_model, masker_tok)

    if filler_cfg["filler_type"] == "seq2seq":
        anon_text = run_filler_seq2seq(
            masked_text, filler_model, filler_tok, filler_cfg["prompt_prefix"]
        )
    else:
        anon_text = run_filler_mlm(masked_text, filler_model, filler_tok)

    return anon_text, masked_text, entity_spans


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test_set(
    masker_model, masker_tok,
    filler_model, filler_tok,
    filler_cfg: dict,
    test_data: list,
    combo_label: str,
):
    """Run the full pipeline on the test set using batched inference."""
    all_preds, all_targets, all_originals, all_entities, all_masked = [], [], [], [], []

    # Split into batches of EVAL_BATCH_SIZE
    batches = [test_data[i:i + EVAL_BATCH_SIZE] for i in range(0, len(test_data), EVAL_BATCH_SIZE)]

    for batch in tqdm(batches, desc=f"    Test set [{combo_label}]", leave=False):
        originals = [ex["original_text"] for ex in batch]
        targets   = [ex["anonymized_text"] for ex in batch]
        entities  = [ex.get("entity_texts", []) for ex in batch]

        # ── Batched masker ──
        masker_results = run_masker_batch(originals, masker_model, masker_tok)
        masked_texts   = [r[0] for r in masker_results]

        # ── Batched filler ──
        if filler_cfg["filler_type"] == "seq2seq":
            anon_texts = run_filler_seq2seq_batch(masked_texts, filler_model, filler_tok, filler_cfg["prompt_prefix"])
        else:
            anon_texts = run_filler_mlm_batch(masked_texts, filler_model, filler_tok)

        all_preds.extend(anon_texts)
        all_targets.extend(targets)
        all_originals.extend(originals)
        all_entities.extend(entities)
        all_masked.extend(masked_texts)

    print(f"    Computing metrics on {len(all_preds)} predictions …")
    metrics = compute_all_metrics(
        preds=all_preds,
        targets=all_targets,
        original_texts=all_originals,
        entity_texts_list=all_entities,
        masked_texts=all_masked,
        compute_bert=True,
    )

    # Serialise top10 for JSON
    top10 = metrics.pop("leaked_entities_top10", [])
    metrics["leaked_entities_top10"] = [{"entity": e, "count": c} for e, c in top10]

    # Collect sample predictions
    samples = []
    for i in range(min(5, len(all_preds))):
        samples.append({
            "original":   all_originals[i][:250],
            "masked":     all_masked[i][:250],
            "prediction": all_preds[i][:250],
            "target":     all_targets[i][:250],
        })

    return metrics, samples


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ON CURATED EVAL EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_eval_examples(
    masker_model, masker_tok,
    filler_model, filler_tok,
    filler_cfg: dict,
    eval_examples: list,
    combo_label: str,
):
    """
    Run pipeline on curated eval examples.

    For PII examples: check if the output differs from the input
    (i.e. masker found something and filler changed it).

    For no-PII examples: check if the output is unchanged
    (no false positives).
    """
    results = []
    for ex in tqdm(eval_examples, desc=f"    Eval examples [{combo_label}]", leave=False):
        input_text = ex["input"]
        is_no_pii  = ex.get("category", "") == "no_pii"

        anon, masked, entity_spans = run_pipeline(
            input_text, masker_model, masker_tok, filler_model, filler_tok, filler_cfg
        )

        changed        = anon.strip() != input_text.strip()
        entities_found = len(entity_spans)

        results.append({
            "id":             ex.get("id", ""),
            "difficulty":     ex.get("difficulty", ""),
            "category":       ex.get("category", ""),
            "is_no_pii":      is_no_pii,
            "input":          input_text,
            "masked":         masked,
            "output":         anon,
            "changed":        changed,
            "entities_found": entities_found,
            "entity_spans":   [(etype, span) for etype, span in entity_spans],
        })

    # Summarize
    pii_examples   = [r for r in results if not r["is_no_pii"]]
    nopii_examples = [r for r in results if r["is_no_pii"]]

    pii_changed   = sum(1 for r in pii_examples if r["changed"])
    nopii_correct = sum(1 for r in nopii_examples if not r["changed"])

    per_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        pii_s  = [r for r in subset if not r["is_no_pii"]]
        nopii_s = [r for r in subset if r["is_no_pii"]]
        per_difficulty[diff] = {
            "pii_changed":   sum(1 for r in pii_s if r["changed"]),
            "pii_total":     len(pii_s),
            "no_pii_correct": sum(1 for r in nopii_s if not r["changed"]),
            "no_pii_total":  len(nopii_s),
        }

    summary = {
        "total_examples":        len(results),
        "pii_changed":           pii_changed,
        "pii_total":             len(pii_examples),
        "pii_changed_pct":       round(pii_changed / max(len(pii_examples), 1) * 100, 1),
        "no_pii_correct":        nopii_correct,
        "no_pii_total":          len(nopii_examples),
        "no_pii_correct_pct":    round(nopii_correct / max(len(nopii_examples), 1) * 100, 1),
        "per_difficulty":        per_difficulty,
    }

    return {"examples": results, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_readable_report(all_results: list, filepath: str):
    lines = []
    w = lines.append

    w("=" * 115)
    w("  COMPREHENSIVE EVALUATION REPORT — APPROACH 2: MASK-AND-FILL PIPELINE")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 115)
    w("")

    w(f"  Combos evaluated:  {len(all_results)}")
    w(f"  Architecture:      Masker (NER/Token-Classification) → Filler (Seq2Seq or MLM)")
    w("")

    # ── Combo overview ──
    w("  COMBO OVERVIEW")
    w(f"  {'#':<4} {'Combo':<30} {'Masker':<22} {'Filler':<25} {'Masker Arch':<28} {'Filler Arch'}")
    w("  " + "─" * 113)
    for i, res in enumerate(all_results, 1):
        mk  = res["masker_key"]
        fk  = res["filler_key"]
        mca = MASKER_CONFIGS[mk]["architecture"].replace("ForTokenClassification", "ForTC")
        fca = FILLER_CONFIGS[fk]["architecture"]
        w(f"  {i:<4} {res['combo_label']:<30} {mk:<22} {fk:<25} {mca:<28} {fca}")
    w("")

    # ── Per-combo details ──
    for i, res in enumerate(all_results, 1):
        mk = res["masker_key"]
        fk = res["filler_key"]
        w("─" * 115)
        w(f"  [{i}/{len(all_results)}]  {res['combo_label']}")
        w(f"  Masker: {MASKER_CONFIGS[mk]['model_id']}")
        w(f"  Filler: {FILLER_CONFIGS[fk]['model_id']}")
        w("─" * 115)
        w(f"  Masker params:   {res.get('masker_params_M', '?')}M  ({MASKER_CONFIGS[mk]['description']})")
        w(f"  Filler params:   {res.get('filler_params_M', '?')}M  ({FILLER_CONFIGS[fk]['description']})")
        w(f"  Total params:    ~{(res.get('masker_params_M', 0) + res.get('filler_params_M', 0))}M combined")
        w(f"  Eval time:       {res.get('eval_time', 'N/A')}")
        w("")

        # Test set
        test = res.get("test_set")
        if test:
            m = test["metrics"]
            w(f"  ┌─── TEST SET METRICS  ({test.get('n_examples', '?')} examples) ────────────────────────────┐")
            w(f"  │  [Privacy]                                                              │")
            w(f"  │  Entity Leakage Rate ↓  (lower=better):   {m.get('entity_leakage_rate',0):<30.2f} │")
            w(f"  │  Sample Leakage Rate ↓:                   {m.get('leakage_rate',0):<30.2f} │")
            leak_str = f"{m.get('total_entities_leaked',0)}/{m.get('total_entities_checked',0)}"
            w(f"  │  Entities Leaked / Checked:               {leak_str:<30} │")
            w(f"  │                                                                         │")
            w(f"  │  [Masker Quality]                                                       │")
            w(f"  │  Masker Detection Rate ↑:                 {m.get('masker_detection_rate',0):<30.2f} │")
            w(f"  │  Samples with Any Mask ↑:                 {m.get('samples_with_any_mask',0):<30.2f} │")
            w(f"  │                                                                         │")
            w(f"  │  [Text Quality vs Reference]                                            │")
            w(f"  │  Exact Match:                             {m.get('exact_match',0):<30.2f} │")
            w(f"  │  Word Accuracy:                           {m.get('word_accuracy',0):<30.2f} │")
            w(f"  │  BLEU:                                    {m.get('bleu',0):<30.2f} │")
            w(f"  │  BLEU-1:                                  {m.get('bleu_1',0):<30.2f} │")
            w(f"  │  BLEU-2:                                  {m.get('bleu_2',0):<30.2f} │")
            w(f"  │  BLEU-4:                                  {m.get('bleu_4',0):<30.2f} │")
            w(f"  │  ROUGE-1:                                 {m.get('rouge_1',0):<30.2f} │")
            w(f"  │  ROUGE-2:                                 {m.get('rouge_2',0):<30.2f} │")
            w(f"  │  ROUGE-L:                                 {m.get('rouge_l',0):<30.2f} │")
            w(f"  │  BERTScore P:                             {m.get('bertscore_precision',0):<30.2f} │")
            w(f"  │  BERTScore R:                             {m.get('bertscore_recall',0):<30.2f} │")
            w(f"  │  BERTScore F1:                            {m.get('bertscore_f1',0):<30.2f} │")
            w(f"  └─────────────────────────────────────────────────────────────────────────┘")
            w("")

            # Sample predictions
            samples = test.get("sample_predictions", [])
            if samples:
                w(f"  Sample Pipeline Outputs (original → masked → predicted | reference):")
                for j, s in enumerate(samples, 1):
                    w(f"    [{j}] ORIG:   {s['original']}")
                    w(f"        MASKED: {s['masked']}")
                    w(f"        PRED:   {s['prediction']}")
                    w(f"        REF:    {s['target']}")
                    w("")
        else:
            w("  Test set: SKIPPED")
            w("")

        # Eval examples
        ev = res.get("eval_examples")
        if ev:
            s = ev["summary"]
            w(f"  ┌─── EVAL EXAMPLES ──────────────────────────────────────────────────────┐")
            w(f"  │  Total:               {s['total_examples']:<51} │")
            pii_str = f"{s['pii_changed']}/{s['pii_total']} ({s['pii_changed_pct']}%)"
            w(f"  │  PII examples changed:  {pii_str:<49} │")
            nopii_str = f"{s['no_pii_correct']}/{s['no_pii_total']} ({s['no_pii_correct_pct']}%)"
            w(f"  │  No-PII correct (unchanged): {nopii_str:<44} │")
            w(f"  │                                                                         │")
            for diff in ["easy", "medium", "hard"]:
                ds = s["per_difficulty"].get(diff, {})
                if ds:
                    pii_d = f"{ds['pii_changed']}/{ds['pii_total']}" if ds["pii_total"] > 0 else "—"
                    nopii_d = f"{ds['no_pii_correct']}/{ds['no_pii_total']}" if ds["no_pii_total"] > 0 else "—"
                    w(f"  │  {diff.upper():<8}  PII changed: {pii_d:<14}  No-PII correct: {nopii_d:<14} │")
            w(f"  └─────────────────────────────────────────────────────────────────────────┘")
            w("")

            # ── Sample anonymizations by difficulty (3 from each) ──
            w("")
            w("  Sample Anonymizations by Difficulty (3 easy · 3 medium · 3 hard):")
            for diff in ["easy", "medium", "hard"]:
                diff_pii = [e for e in ev["examples"] if e["difficulty"] == diff and not e["is_no_pii"]]
                if not diff_pii:
                    continue
                w("")
                w(f"  {'─'*6} {diff.upper()} {'─'*65}")
                for j, ex in enumerate(diff_pii[:3], 1):
                    status = "✓ CHANGED" if ex["changed"] else "— UNCHANGED (missed)"
                    ents = ", ".join(f"{e[0]}:{e[1]}" for e in ex["entity_spans"][:4])
                    w(f"    [{j}] [{ex['id']}] {ex['category']} — {status}")
                    w(f"      IN:     {ex['input']}")
                    if ex["masked"] != ex["input"]:
                        w(f"      MASKED: {ex['masked']}")
                    w(f"      OUT:    {ex['output']}")
                    if ents:
                        w(f"      FOUND:  {ents}")
                    w("")
            # Also show no-PII examples (from hard, where they appear)
            nopii_egs = [e for e in ev["examples"] if e["is_no_pii"]]
            if nopii_egs:
                w(f"  {'─'*6} NO-PII (should be unchanged) {'─'*47}")
                for j, ex in enumerate(nopii_egs[:3], 1):
                    status = "✓ CORRECT (unchanged)" if not ex["changed"] else "✗ FALSE POSITIVE"
                    w(f"    [{j}] [{ex['id']}] {ex['category']} — {status}")
                    w(f"      IN:  {ex['input']}")
                    w(f"      OUT: {ex['output']}")
                    w("")

            # ── Full per-example details ──
            w("")
            w("  All eval example details:")
            for ex in ev["examples"]:
                if ex["is_no_pii"]:
                    status = "✓ CORRECT (unchanged)" if not ex["changed"] else "✗ FALSE POSITIVE"
                else:
                    status = "✓ CHANGED" if ex["changed"] else "— UNCHANGED (missed)"
                ents = ", ".join(f"{e[0]}:{e[1]}" for e in ex["entity_spans"][:3])
                w(f"    [{ex['id']}] {ex['category']} — {status}")
                w(f"      IN:  {ex['input']}")
                if ex["masked"] != ex["input"]:
                    w(f"      MSK: {ex['masked']}")
                w(f"      OUT: {ex['output']}")
                if ents:
                    w(f"      ENT: {ents}")
                w("")
        else:
            w("  Eval examples: SKIPPED")
            w("")

    # ── Summary comparison table ──
    w("")
    w("=" * 115)
    w("  COMBO COMPARISON TABLE")
    w("=" * 115)
    w("")

    header = f"  {'Combo':<30} {'Masker':<14} {'Filler':<14}"
    header += f" {'Ent.Leak%':>10} {'Det.Rate%':>10} {'BLEU':>8} {'ROUGE-L':>8} {'BScoreF1':>10} {'PII-Chg%':>10}"
    w(header)
    w("  " + "─" * 113)

    for res in all_results:
        test = res.get("test_set")
        ev   = res.get("eval_examples", {})
        mk, fk = res["masker_key"], res["filler_key"]
        if test:
            m = test["metrics"]
            s = ev["summary"] if ev else {}
            pii_pct = f"{s.get('pii_changed_pct', 'N/A')}"
            row = f"  {res['combo_label']:<30} {mk:<14} {fk:<14}"
            row += f" {m.get('entity_leakage_rate',0):>10.2f}"
            row += f" {m.get('masker_detection_rate',0):>10.2f}"
            row += f" {m.get('bleu',0):>8.2f}"
            row += f" {m.get('rouge_l',0):>8.2f}"
            row += f" {m.get('bertscore_f1',0):>10.2f}"
            row += f" {pii_pct:>10}"
        else:
            row = f"  {res['combo_label']:<30} {mk:<14} {fk:<14}" + "         N/A" * 6
        w(row)

    w("")
    w("  Columns:")
    w("    Ent.Leak%  — Entity Leakage Rate (↓ lower is better, primary privacy metric)")
    w("    Det.Rate%  — Masker entity detection rate (↑ higher means more PII masked)")
    w("    BLEU       — BLEU score vs reference anonymization (↑ better text quality)")
    w("    ROUGE-L    — ROUGE-L vs reference (↑ better text preservation)")
    w("    BScoreF1   — BERTScore F1 (↑ better semantic similarity to reference)")
    w("    PII-Chg%   — % of curated PII examples where output changed (↑ good)")
    w("")
    w("=" * 115)
    w(f"  Report saved: {filepath}")
    w("=" * 115)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# JSON SERIALISATION HELPER  (module-level so incremental saves can use it)
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    """Recursively convert non-JSON-native types to JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  APPROACH 2 EVALUATION — MASK-AND-FILL PIPELINE")
    print("=" * 70)
    print(f"  Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"  Combos: {len(COMBOS)}")
    print(f"  Test limit: {TEST_SET_LIMIT}")

    # Load data once
    test_data = load_test_data(limit=TEST_SET_LIMIT)
    if not test_data:
        print("\n  [ERROR] No test data found. Check TEST_DATA_PATH in config.py")
        sys.exit(1)
    print(f"  Test examples loaded: {len(test_data)}")

    eval_examples = load_eval_examples()
    if eval_examples:
        print(f"  Eval examples loaded: {len(eval_examples)}")
    else:
        print("  Eval examples: not found, will skip")

    all_results = []
    total_start = time.time()

    for idx, (masker_key, filler_key) in enumerate(COMBOS, 1):
        combo_label = f"{masker_key}+{filler_key}"
        print(f"\n{'─' * 70}")
        print(f"  [{idx}/{len(COMBOS)}]  {combo_label}")
        print(f"{'─' * 70}")

        combo_start = time.time()

        result = {
            "combo_label":    combo_label,
            "masker_key":     masker_key,
            "filler_key":     filler_key,
            "masker_model_id": MASKER_CONFIGS[masker_key]["model_id"],
            "filler_model_id": FILLER_CONFIGS[filler_key]["model_id"],
        }

        # ── Load masker ──
        try:
            masker_model, masker_tok = load_masker(masker_key)
            result["masker_params_M"] = count_parameters(masker_model)["total_millions"]
        except Exception as e:
            print(f"  [ERROR] Failed to load masker '{masker_key}': {e}")
            result["error"] = str(e)
            all_results.append(result)
            continue

        # ── Load filler ──
        try:
            filler_model, filler_tok = load_filler(filler_key)
            result["filler_params_M"] = count_parameters(filler_model)["total_millions"]
        except Exception as e:
            print(f"  [ERROR] Failed to load filler '{filler_key}': {e}")
            unload(masker_model, masker_tok)
            result["error"] = str(e)
            all_results.append(result)
            continue

        filler_cfg = FILLER_CONFIGS[filler_key]

        # ── Test set ──
        print(f"  Running pipeline on test set ({len(test_data)} examples) …")
        try:
            test_metrics, test_samples = evaluate_on_test_set(
                masker_model, masker_tok,
                filler_model, filler_tok,
                filler_cfg, test_data, combo_label,
            )
            result["test_set"] = {
                "n_examples":        len(test_data),
                "metrics":           test_metrics,
                "sample_predictions": test_samples,
            }
            m = test_metrics
            print(f"    EntityLeak: {m.get('entity_leakage_rate',0):.2f}%  |  "
                  f"MaskerDet: {m.get('masker_detection_rate',0):.2f}%  |  "
                  f"BLEU: {m.get('bleu',0):.2f}  |  "
                  f"ROUGE-L: {m.get('rouge_l',0):.2f}  |  "
                  f"BScoreF1: {m.get('bertscore_f1',0):.2f}")
        except Exception as e:
            print(f"  [ERROR] Test set evaluation failed: {e}")
            result["test_set"] = None

        # ── Eval examples ──
        if eval_examples:
            print(f"  Running pipeline on eval examples ({len(eval_examples)}) …")
            try:
                eval_result = evaluate_on_eval_examples(
                    masker_model, masker_tok,
                    filler_model, filler_tok,
                    filler_cfg, eval_examples, combo_label,
                )
                result["eval_examples"] = eval_result
                s = eval_result["summary"]
                print(f"    PII changed: {s['pii_changed']}/{s['pii_total']} "
                      f"({s['pii_changed_pct']}%)  |  "
                      f"No-PII correct: {s['no_pii_correct']}/{s['no_pii_total']}")
            except Exception as e:
                print(f"  [ERROR] Eval examples failed: {e}")
                result["eval_examples"] = None
        else:
            result["eval_examples"] = None

        result["eval_time"] = format_time(time.time() - combo_start)
        print(f"  Combo done in {result['eval_time']}")

        all_results.append(result)

        # ── Save partial results immediately after each combo ──
        with open(RESULTS_JSON, "w", encoding="utf-8") as _f:
            json.dump(_sanitize(all_results), _f, indent=2, ensure_ascii=False)
        write_readable_report(all_results, RESULTS_TXT)
        print(f"  [Saved] Partial results written after combo {idx}/{len(COMBOS)}")

        # ── Unload both models before next combo ──
        unload(masker_model, masker_tok)
        unload(filler_model, filler_tok)

    total_time = format_time(time.time() - total_start)

    # ── Final save (already done incrementally, but do one clean final write) ──
    print(f"\n{'=' * 70}")
    print(f"  Total time: {total_time}")
    print(f"  Final save …")
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(_sanitize(all_results), f, indent=2, ensure_ascii=False)
    print(f"  JSON written: {RESULTS_JSON}")
    write_readable_report(all_results, RESULTS_TXT)
    print(f"  TXT written:  {RESULTS_TXT}")

    # ── Print summary to stdout ──
    print("\n" + "=" * 70)
    print("  QUICK SUMMARY")
    print("=" * 70)
    print(f"  {'Combo':<30} {'Ent.Leak%':>10} {'Det.Rate%':>10} {'BLEU':>8} {'ROUGE-L':>8}")
    print("  " + "─" * 68)
    for res in all_results:
        test = res.get("test_set")
        if test:
            m = test["metrics"]
            print(f"  {res['combo_label']:<30} "
                  f"{m.get('entity_leakage_rate',0):>10.2f} "
                  f"{m.get('masker_detection_rate',0):>10.2f} "
                  f"{m.get('bleu',0):>8.2f} "
                  f"{m.get('rouge_l',0):>8.2f}")
        else:
            print(f"  {res['combo_label']:<30}   (no test results)")
    print("")
    print(f"  Full report: {RESULTS_TXT}")
    print(f"  JSON data:   {RESULTS_JSON}")


if __name__ == "__main__":
    main()
