"""
evaluate_tab.py  —  testing_approach1
======================================
Evaluate all Seq2Seq encoder-decoder models from Seq2Seq_model/checkpoints2/
on the Text Anonymization Benchmark (TAB) (Pilán et al., 2022).

Models evaluated (distilbart excluded — too large for GPU):
  • t5-efficient-tiny   (~16M params)
  • t5-small            (~60M params)
  • flan-t5-small       (~77M params)
  • bart-base           (~139M params)

Dataset : ildpil/text-anonymization-benchmark  (test split, 127 documents)
Metric target : gold-masked text — DIRECT/QUASI entity spans replaced by
                [PLACEHOLDER] tokens using the same char-offset annotations.

TAB entity → placeholder map:
  PERSON   → [FULLNAME]       LOC      → [LOCATION]
  ORG      → [ORGANIZATION]   DATETIME → [DATE]
  CODE     → [ID_NUMBER]      DEM      → [OTHER_PII]
  QUANTITY → [NUMBER]         MISC     → [OTHER_PII]

Key metrics:
  • Gold entity recall  — fraction of gold DIRECT/QUASI spans that no longer
                          appear verbatim in the model output (privacy success)
  • BLEU-1/2/4, ROUGE-1/2/L, BERTScore P/R/F1
  • Entity & sample leakage rate

Usage:
    python evaluate_tab.py
"""

import os
import sys
import gc
import json
import time
import math
from collections import defaultdict
from datetime import datetime

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Add Seq2Seq_model/ to path so we can reuse its modules ────────────────────
SEQ2SEQ_DIR = os.path.join(os.path.dirname(__file__), "..", "Seq2Seq_model")
sys.path.insert(0, os.path.abspath(SEQ2SEQ_DIR))

from config import MODEL_CONFIGS          # all model hparam dicts
from utils import compute_all_metrics, format_time

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
RESULTS_JSON    = os.path.join(RESULTS_DIR, "tab_evaluation_results.json")
RESULTS_TXT     = os.path.join(RESULTS_DIR, "tab_evaluation_results_readable.txt")

# Checkpoints are in Seq2Seq_model/checkpoints2/
CHECKPOINTS_DIR = os.path.join(SEQ2SEQ_DIR, "checkpoints2")

# Models to evaluate (distilbart excluded — too large)
MODELS_TO_EVAL = ["t5-efficient-tiny", "t5-small", "flan-t5-small", "bart-base"]

# TAB data settings
TAB_HF_ID            = "ildpil/text-anonymization-benchmark"
CONFIDENTIAL_TYPES   = {"DIRECT", "QUASI"}
TAB_SEGMENT_LIMIT    = 500   # max paragraph segments to evaluate
MAX_INPUT_LENGTH     = 128
MAX_TARGET_LENGTH    = 128
EVAL_BATCH_SIZE      = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Entity type → placeholder used in gold-masked text
TAB_TYPE_TO_PLACEHOLDER = {
    "PERSON":   "FULLNAME",
    "LOC":      "LOCATION",
    "ORG":      "ORGANIZATION",
    "DATETIME": "DATE",
    "CODE":     "ID_NUMBER",
    "DEM":      "OTHER_PII",
    "QUANTITY": "NUMBER",
    "MISC":     "OTHER_PII",
}

# ─────────────────────────────────────────────────────────────────────────────
# TAB DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _build_gold_masked(text: str, mentions: list) -> str:
    """
    Replace DIRECT/QUASI mention spans with [PLACEHOLDER] tokens
    using character offsets.  Works right-to-left to keep earlier
    offsets valid after replacement.
    """
    # Sort by start offset, de-overlap
    sorted_m = sorted(mentions, key=lambda m: m["start_offset"])
    clean: list = []
    prev_end = -1
    for m in sorted_m:
        if m["start_offset"] >= prev_end:
            clean.append(m)
            prev_end = m["end_offset"]

    for m in reversed(clean):
        label = TAB_TYPE_TO_PLACEHOLDER.get(m.get("entity_type", "MISC"), "UNKNOWN")
        text = text[: m["start_offset"]] + f"[{label}]" + text[m["end_offset"] :]
    return text


def load_tab_segments(split: str = "test", limit: int = TAB_SEGMENT_LIMIT) -> list:
    """
    Download TAB from HuggingFace, deduplicate by doc_id, split into paragraphs,
    return only paragraphs containing ≥1 DIRECT/QUASI entity.

    Each dict has:
        original_text  (str)
        gold_masked    (str)   — reference output for metrics
        entity_texts   (list)  — span_text strings of confidential entities
        gold_mentions  (list)  — mention dicts with paragraph-relative offsets
        doc_id         (str)
        para_idx       (int)
    """
    print(f"Loading TAB from HuggingFace ({TAB_HF_ID}) …")
    ds = load_dataset(TAB_HF_ID)[split]

    seen: set = set()
    unique_docs = []
    for ex in ds:
        if ex["doc_id"] not in seen:
            seen.add(ex["doc_id"])
            unique_docs.append(ex)
    print(f"  {len(unique_docs)} unique documents in '{split}' split.")

    segments: list = []
    for doc in unique_docs:
        if len(segments) >= limit:
            break

        full_text = doc["text"]
        all_mentions = doc["entity_mentions"]   # already a Python list

        char_cursor = 0
        for p_idx, para_raw in enumerate(full_text.split("\n\n")):
            if len(segments) >= limit:
                break

            para = para_raw.strip()
            if not para:
                char_cursor += len(para_raw) + 2
                continue

            para_start = full_text.find(para, char_cursor)
            if para_start < 0:
                char_cursor += len(para_raw) + 2
                continue
            para_end = para_start + len(para)

            # Filter mentions that are fully inside this paragraph and confidential
            para_mentions = []
            for m in all_mentions:
                if (
                    m["identifier_type"] in CONFIDENTIAL_TYPES
                    and m["start_offset"] >= para_start
                    and m["end_offset"] <= para_end
                ):
                    para_mentions.append({
                        **m,
                        "start_offset": m["start_offset"] - para_start,
                        "end_offset":   m["end_offset"]   - para_start,
                    })

            if not para_mentions:
                char_cursor += len(para_raw) + 2
                continue

            # Truncate to MAX_INPUT_LENGTH words
            words = para.split()
            if len(words) > MAX_INPUT_LENGTH:
                para = " ".join(words[:MAX_INPUT_LENGTH])
                para_mentions = [
                    pm for pm in para_mentions if pm["end_offset"] <= len(para)
                ]
                if not para_mentions:
                    char_cursor += len(para_raw) + 2
                    continue

            segments.append({
                "original_text": para,
                "gold_masked":   _build_gold_masked(para, para_mentions),
                "entity_texts":  [m["span_text"] for m in para_mentions],
                "gold_mentions": para_mentions,
                "doc_id":        doc["doc_id"],
                "para_idx":      p_idx,
            })
            char_cursor += len(para_raw) + 2

    print(f"  {len(segments)} paragraph segments with ≥1 DIRECT/QUASI entity "
          f"(limit={limit}).")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING / UNLOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_key: str):
    """Load a fine-tuned Seq2Seq model from its checkpoint."""
    cfg        = MODEL_CONFIGS[model_key]
    model_name = cfg["model_name"]
    ckpt_path  = os.path.join(CHECKPOINTS_DIR, model_key, "best_model.pt")

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model weights from {ckpt_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    gc.collect()

    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# GOLD-ENTITY RECALL
# ─────────────────────────────────────────────────────────────────────────────

def compute_entity_recall(predictions: list, segments: list) -> dict:
    """
    Entity recall = fraction of gold DIRECT/QUASI spans whose span_text
    is no longer present verbatim in the model's output.

    Returns overall recall + per TAB entity type breakdown.
    """
    per_type_det   = defaultdict(int)
    per_type_total = defaultdict(int)
    total_det = total_count = 0

    for pred, seg in zip(predictions, segments):
        pred_lower = pred.lower()
        for m in seg["gold_mentions"]:
            span  = m["span_text"].lower().strip()
            etype = m.get("entity_type", "MISC")
            per_type_total[etype] += 1
            total_count += 1
            if span not in pred_lower:
                per_type_det[etype] += 1
                total_det += 1

    overall = total_det / total_count if total_count else 0.0
    per_type = {
        et: round(per_type_det[et] / per_type_total[et], 4)
        for et in per_type_total
    }
    return {
        "overall_recall":   round(overall, 4),
        "per_type_recall":  per_type,
        "detected_count":   total_det,
        "total_count":      total_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCHED INFERENCE ON TAB SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_batch(
    model, tokenizer, prefix: str, texts: list
) -> list:
    """Tokenize a batch and generate predictions."""
    prompts = [prefix + t for t in texts]
    enc = tokenizer(
        prompts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    gen_ids = model.generate(
        input_ids=enc["input_ids"].to(DEVICE),
        attention_mask=enc["attention_mask"].to(DEVICE),
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
    )
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


@torch.no_grad()
def evaluate_on_tab(
    model, tokenizer, prefix: str,
    tab_data: list, model_key: str,
) -> tuple:
    """
    Run batched inference over all TAB segments.

    Returns (metrics, entity_recall, sample_predictions).
    """
    all_preds     = []
    all_targets   = []
    all_originals = []
    all_entities  = []

    batches = [
        tab_data[i: i + EVAL_BATCH_SIZE]
        for i in range(0, len(tab_data), EVAL_BATCH_SIZE)
    ]

    for batch in tqdm(batches, desc=f"    TAB [{model_key}]", leave=False):
        originals = [s["original_text"] for s in batch]
        targets   = [s["gold_masked"]   for s in batch]
        entities  = [s["entity_texts"]  for s in batch]

        preds = run_inference_batch(model, tokenizer, prefix, originals)

        all_preds.extend(preds)
        all_targets.extend(targets)
        all_originals.extend(originals)
        all_entities.extend(entities)

    print(f"    Computing metrics on {len(all_preds)} predictions …")

    metrics = compute_all_metrics(
        preds=all_preds,
        targets=all_targets,
        original_texts=all_originals,
        entity_texts_list=all_entities,
        compute_bert=True,
    )
    top10 = metrics.pop("leaked_entities_top10", [])
    metrics["leaked_entities_top10"] = [{"entity": e, "count": c} for e, c in top10]

    entity_recall = compute_entity_recall(all_preds, tab_data[:len(all_preds)])

    samples = []
    for i in range(min(5, len(all_preds))):
        samples.append({
            "doc_id":      tab_data[i]["doc_id"],
            "original":    all_originals[i][:300],
            "gold_masked": all_targets[i][:300],
            "prediction":  all_preds[i][:300],
        })

    return metrics, entity_recall, samples


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(all_results: list, tab_data: list, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = []
    W = 90

    def hr(c="═"): lines.append(c * W)
    def blank():   lines.append("")

    # ── Header ────────────────────────────────────────────────────────────────
    hr("╔" + "═" * (W - 2) + "╗")
    lines.append("║" + "  TAB EVALUATION — APPROACH 1: SEQ2SEQ ENCODER-DECODERS".center(W-2) + "║")
    lines.append("║" + "  Text Anonymization Benchmark (Pilán et al., 2022)".center(W-2) + "║")
    lines.append("║" + "  ECHR Court Decisions · Zero-shot Transfer".center(W-2) + "║")
    hr("╚" + "═" * (W - 2) + "╝")
    blank()
    lines.append(f"  Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Dataset        : {TAB_HF_ID}  (test split)")
    lines.append(f"  Segments eval  : {len(tab_data)}")
    lines.append(f"  Gold entities  : {sum(len(s['gold_mentions']) for s in tab_data)}"
                 f"   (DIRECT + QUASI only)")
    lines.append(f"  Metric target  : gold-masked text (entity spans → [PLACEHOLDER])")
    lines.append(f"  Device         : {DEVICE}")
    blank()
    lines.append("  NOTE: All models were fine-tuned on AI4Privacy synthetic data.")
    lines.append("  These results measure zero-shot cross-domain transfer to real legal text.")
    blank()

    # ── Summary table ─────────────────────────────────────────────────────────
    hr()
    lines.append("  RESULTS SUMMARY")
    hr()
    blank()
    hdr = f"  {'Model':<22} {'Params':>7} {'Exact':>7} {'Word%':>7} {'BLEU-4':>7} {'ROUGE-L':>8} {'BERTScore':>10} {'Recall':>8} {'Leak%':>7}"
    lines.append(hdr)
    lines.append("  " + "─" * (W - 2))
    for r in all_results:
        m  = r["metrics"]
        er = r["entity_recall"]
        lines.append(
            f"  {r['model_key']:<22} {r.get('params_M','?'):>6}M"
            f" {m.get('exact_match',0):>7.3f}"
            f" {m.get('word_accuracy',0):>7.3f}"
            f" {m.get('bleu4',0):>7.3f}"
            f" {m.get('rougeL',0):>8.3f}"
            f" {m.get('bertscore_f1',0):>10.3f}"
            f" {er['overall_recall']:>8.3f}"
            f" {m.get('sample_leakage_rate',0)*100:>6.1f}%"
        )
    blank()

    # ── Per-model detail ───────────────────────────────────────────────────────
    hr()
    lines.append("  DETAILED RESULTS PER MODEL")
    hr()

    for r in all_results:
        blank()
        hr("─")
        lines.append(f"  Model    : {r['model_key']}  ({r['model_name']})")
        lines.append(f"  Params   : {r.get('params_M','?')}M  |  Ckpt: {r.get('ckpt_size_mb','?')} MB")
        lines.append(f"  Prefix   : {repr(r.get('prefix',''))}")
        lines.append(f"  Runtime  : {r.get('runtime_str','N/A')}")
        hr("─")
        blank()

        er = r["entity_recall"]
        lines.append("  ── Gold Entity Recall (TAB annotations) ──")
        lines.append(f"    Overall   : {er['overall_recall']:.4f}"
                     f"  ({er['detected_count']}/{er['total_count']} entities removed from output)")
        lines.append("    Per type  :")
        for etype, rec in sorted(er["per_type_recall"].items(), key=lambda x: -x[1]):
            lines.append(f"      {etype:<12}  {rec:.4f}")

        blank()
        m = r["metrics"]
        lines.append("  ── Anonymization Quality (vs gold-masked) ──")
        lines.append(f"    BLEU-1    : {m.get('bleu1',0):.4f}")
        lines.append(f"    BLEU-2    : {m.get('bleu2',0):.4f}")
        lines.append(f"    BLEU-4    : {m.get('bleu4',0):.4f}")
        lines.append(f"    ROUGE-1   : {m.get('rouge1',0):.4f}")
        lines.append(f"    ROUGE-2   : {m.get('rouge2',0):.4f}")
        lines.append(f"    ROUGE-L   : {m.get('rougeL',0):.4f}")
        lines.append(f"    BERTScore P: {m.get('bertscore_p',0):.4f}")
        lines.append(f"    BERTScore R: {m.get('bertscore_r',0):.4f}")
        lines.append(f"    BERTScore F1:{m.get('bertscore_f1',0):.4f}")
        lines.append(f"    Exact Match: {m.get('exact_match',0):.4f}")
        lines.append(f"    Word Acc   : {m.get('word_accuracy',0):.4f}")

        blank()
        lines.append("  ── Privacy Leakage ──")
        lines.append(f"    Entity Leakage Rate : {m.get('entity_leakage_rate',0):.4f}")
        lines.append(f"    Sample Leakage Rate : {m.get('sample_leakage_rate',0):.4f}")
        top10 = m.get("leaked_entities_top10", [])
        if top10:
            lines.append("    Top leaked entities :")
            for entry in top10[:5]:
                lines.append(f"      {entry['entity']!r:30s}  count={entry['count']}")

        blank()
        lines.append("  ── Sample Outputs ──")
        for i, s in enumerate(r.get("samples", [])[:3], 1):
            lines.append(f"  Sample {i}  [doc_id={s.get('doc_id','')}]")
            lines.append(f"    Original    : {s['original'][:200]}")
            lines.append(f"    Gold masked : {s.get('gold_masked','N/A')[:200]}")
            lines.append(f"    Prediction  : {s['prediction'][:200]}")
            blank()

    # ── Cross-domain notes ─────────────────────────────────────────────────────
    hr()
    lines.append("  CROSS-DOMAIN NOTES")
    hr()
    blank()
    lines.append("  Training domain : AI4Privacy — short synthetic sentences, ~50 fine-grained types")
    lines.append("  Test domain     : TAB / ECHR  — long legal paragraphs, 8 coarse entity types")
    blank()
    lines.append("  Domain gap indicators to watch for:")
    lines.append("    • Low PERSON recall — legal text uses titled names (Mr, Judge, Dr ...)")
    lines.append("    • Low ORG recall    — court/institution names overlap with non-PII words")
    lines.append("    • High DATETIME recall — dates are surface-similar across domains")
    lines.append("    • Models with a task prefix tend to be more robust (flan-t5 family)")
    blank()

    hr()
    lines.append("  END OF REPORT")
    hr()

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON SANITISER
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    if isinstance(obj, dict):   return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Approach 1 — Seq2Seq TAB Evaluation")
    print(f"  Device : {DEVICE}" + (
        f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print("=" * 60)

    # ── Check which models actually have checkpoints ───────────────────────────
    available = []
    for key in MODELS_TO_EVAL:
        ckpt = os.path.join(CHECKPOINTS_DIR, key, "best_model.pt")
        if os.path.exists(ckpt):
            available.append(key)
        else:
            print(f"  [SKIP] {key} — checkpoint not found at {ckpt}")
    if not available:
        print("No checkpoints found. Exiting.")
        return
    print(f"\n  Models to evaluate: {available}\n")

    # ── Load TAB segments ──────────────────────────────────────────────────────
    tab_data = load_tab_segments(split="test", limit=TAB_SEGMENT_LIMIT)
    if not tab_data:
        print("No TAB segments loaded. Exiting.")
        return
    total_gold = sum(len(s["gold_mentions"]) for s in tab_data)
    print(f"\n  {len(tab_data)} segments  |  {total_gold} gold DIRECT/QUASI entities\n")

    all_results = []

    for model_key in available:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}  ({cfg['model_name']})")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_key)

        # Count params
        total_p = sum(p.numel() for p in model.parameters())
        params_M = round(total_p / 1e6, 1)

        ckpt_path = os.path.join(CHECKPOINTS_DIR, model_key, "best_model.pt")
        ckpt_mb   = round(os.path.getsize(ckpt_path) / 1e6, 1)

        t0 = time.time()
        metrics, entity_recall, samples = evaluate_on_tab(
            model, tokenizer, cfg["prefix"], tab_data, model_key
        )
        elapsed = time.time() - t0

        print(f"  ✓ Done in {format_time(elapsed)}")
        print(f"    Entity recall  : {entity_recall['overall_recall']:.3f}")
        print(f"    BERTScore F1   : {metrics.get('bertscore_f1', 0):.3f}")
        print(f"    Sample leak    : {metrics.get('sample_leakage_rate', 0):.3f}")

        result = {
            "model_key":     model_key,
            "model_name":    cfg["model_name"],
            "prefix":        cfg["prefix"],
            "params_M":      params_M,
            "ckpt_size_mb":  ckpt_mb,
            "runtime_sec":   round(elapsed, 1),
            "runtime_str":   format_time(elapsed),
            "n_segments":    len(tab_data),
            "n_gold_entities": total_gold,
            "metrics":       metrics,
            "entity_recall": entity_recall,
            "samples":       samples,
        }
        all_results.append(result)

        # ── Incremental save ──────────────────────────────────────────────────
        write_report(all_results, tab_data, RESULTS_TXT)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(_sanitize(all_results), f, indent=2, ensure_ascii=False)
        print(f"  Saved ({len(all_results)}/{len(available)} models)")

        unload_model(model, tokenizer)

    # ── Final save ─────────────────────────────────────────────────────────────
    write_report(all_results, tab_data, RESULTS_TXT)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(_sanitize(all_results), f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_results)} models evaluated")
    print(f"  Report : {RESULTS_TXT}")
    print(f"  JSON   : {RESULTS_JSON}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
