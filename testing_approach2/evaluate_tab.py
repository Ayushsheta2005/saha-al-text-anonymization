"""
evaluate_tab.py
===============
Evaluate all 6 masker+filler pipelines on the Text Anonymization Benchmark (TAB).

Dataset : Pilán et al. 2022 — ECHR court decisions with expert PII annotations
HF ID   : ildpil/text-anonymization-benchmark  (test split = 127 unique docs)

Key differences vs. evaluate.py (AI4Privacy test set)
------------------------------------------------------
• Gold annotations are character-level spans, NOT a pre-anonymized target text.
• Documents are long legal texts (avg 4 768 chars); we segment by paragraph so
  the masker can process them within MAX_INPUT_LENGTH tokens.
• Extra metric: masker entity recall — fraction of gold DIRECT/QUASI entities
  actually detected and masked by the NER model.
• No gold "filler" output exists — BLEU/ROUGE/BERTScore are computed against
  the *gold-masked text* (entities replaced by [ENTITY_TYPE] placeholders).
• Cross-domain evaluation: models were trained on AI4Privacy synthetic data,
  tested here on real-world European court decisions.

TAB entity-type → placeholder mapping
--------------------------------------
  PERSON   → [FULLNAME]       LOC      → [LOCATION]
  ORG      → [ORGANIZATION]   DATETIME → [DATE]
  CODE     → [ID_NUMBER]      DEM      → [OTHER_PII]
  QUANTITY → [NUMBER]         MISC     → [OTHER_PII]
"""

import os
import re
import sys
import json
import time
import gc
import math
from collections import defaultdict

import torch
from tqdm import tqdm
from datasets import load_dataset

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MASKER_CONFIGS,
    FILLER_CONFIGS,
    COMBOS,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_BEAMS,
    EVAL_BATCH_SIZE,
    RESULTS_DIR,
)
from utils import (
    compute_all_metrics,
    format_time,
    aggressive_cleanup,
)
from evaluate import (
    DEVICE,
    load_masker,
    load_filler,
    unload,
    run_masker_batch,
    run_filler_seq2seq_batch,
    run_filler_mlm_batch,
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB-SPECIFIC CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TAB_RESULTS_JSON = os.path.join(RESULTS_DIR, "tab_evaluation_results.json")
TAB_RESULTS_TXT  = os.path.join(RESULTS_DIR, "tab_evaluation_results_readable.txt")

# identifier_type values considered confidential PII in the TAB annotation scheme
CONFIDENTIAL_ID_TYPES = {"DIRECT", "QUASI"}

# TAB entity-type → NER placeholder token (matches the masker's label vocabulary)
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

# Maximum paragraph segments to evaluate (keep runtime tractable)
TAB_SEGMENT_LIMIT = 500

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _build_gold_masked_text(text: str, mentions: list) -> str:
    """
    Replace every DIRECT/QUASI mention span in *text* with its [PLACEHOLDER].

    Uses character-level offsets.  Processes sorted, non-overlapping mentions
    from right-to-left so earlier offsets stay valid.
    """
    # Sort ascending by start offset, then de-overlap
    sorted_m = sorted(mentions, key=lambda m: m["start_offset"])
    deoverlapped = []
    prev_end = -1
    for m in sorted_m:
        if m["start_offset"] >= prev_end:
            deoverlapped.append(m)
            prev_end = m["end_offset"]

    # Replace right-to-left to keep earlier offsets valid
    for m in reversed(deoverlapped):
        etype = m.get("entity_type", "MISC")
        label = TAB_TYPE_TO_PLACEHOLDER.get(etype, "UNKNOWN")
        placeholder = f"[{label}]"
        text = text[: m["start_offset"]] + placeholder + text[m["end_offset"] :]

    return text


def load_tab_segments(split: str = "test", limit: int = TAB_SEGMENT_LIMIT) -> list:
    """
    Download the TAB dataset from HuggingFace, deduplicate by doc_id (keep the
    first annotator), split each document into paragraphs, and return the subset
    of paragraphs that contain at least one DIRECT or QUASI entity mention.

    Each returned dict has:
        original_text   (str)  – paragraph text
        gold_masked     (str)  – paragraph with DIRECT/QUASI spans → [PLACEHOLDER]
        entity_texts    (list) – list of span_text strings for DIRECT/QUASI entities
        gold_mentions   (list) – raw mention dicts (paragraph-relative offsets)
        doc_id          (str)
        para_idx        (int)
    """
    print("Loading TAB dataset from HuggingFace …")
    ds = load_dataset("ildpil/text-anonymization-benchmark")[split]

    # One annotation per doc (first occurrence of each doc_id)
    seen_docs: set = set()
    unique_docs = []
    for ex in ds:
        if ex["doc_id"] not in seen_docs:
            seen_docs.add(ex["doc_id"])
            unique_docs.append(ex)
    print(f"  {len(unique_docs)} unique documents in {split} split.")

    segments = []
    for doc in unique_docs:
        if len(segments) >= limit:
            break

        full_text = doc["text"]
        all_mentions = doc["entity_mentions"]  # already a Python list

        # Split document into paragraphs
        raw_paras = full_text.split("\n\n")
        char_cursor = 0
        for p_idx, para_raw in enumerate(raw_paras):
            if len(segments) >= limit:
                break

            para = para_raw.strip()
            if not para:
                char_cursor += len(para_raw) + 2  # +2 for "\n\n"
                continue

            # Find the exact char offset of this paragraph in the full text
            para_start = full_text.find(para, char_cursor)
            if para_start < 0:
                char_cursor += len(para_raw) + 2
                continue
            para_end = para_start + len(para)

            # Filter DIRECT/QUASI mentions fully contained in this paragraph
            para_mentions = []
            for m in all_mentions:
                if (
                    m["identifier_type"] in CONFIDENTIAL_ID_TYPES
                    and m["start_offset"] >= para_start
                    and m["end_offset"] <= para_end
                ):
                    # Adjust offsets to be paragraph-relative
                    pm = {**m,
                          "start_offset": m["start_offset"] - para_start,
                          "end_offset":   m["end_offset"]   - para_start}
                    para_mentions.append(pm)

            # Only keep paragraphs that have at least one confidential entity
            if not para_mentions:
                char_cursor += len(para_raw) + 2
                continue

            # Truncate paragraph to MAX_INPUT_LENGTH words so the masker sees the
            # same content that fits in its context window
            words = para.split()
            if len(words) > MAX_INPUT_LENGTH:
                para = " ".join(words[:MAX_INPUT_LENGTH])
                para_end = para_start + len(para)
                # Re-filter mentions that still fit in truncated para
                para_mentions = [
                    pm for pm in para_mentions if pm["end_offset"] <= len(para)
                ]
                if not para_mentions:
                    char_cursor += len(para_raw) + 2
                    continue

            gold_masked = _build_gold_masked_text(para, para_mentions)
            entity_texts = [m["span_text"] for m in para_mentions]

            segments.append({
                "original_text": para,
                "gold_masked":   gold_masked,
                "entity_texts":  entity_texts,
                "gold_mentions": para_mentions,
                "doc_id":        doc["doc_id"],
                "para_idx":      p_idx,
            })

            char_cursor += len(para_raw) + 2

    print(f"  {len(segments)} paragraph segments with ≥1 DIRECT/QUASI entity "
          f"(limit={limit}).")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# MASKER RECALL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_masker_recall(masked_texts: list, segments: list) -> dict:
    """
    For each segment, compute how many gold DIRECT/QUASI entities the masker
    detected (i.e. their span_text no longer appears in masked_text).

    Returns:
        overall_recall   (float) – macro-averaged across all segments
        per_type_recall  (dict)  – {entity_type: recall}
        detected_count   (int)
        total_count      (int)
    """
    per_type_detected = defaultdict(int)
    per_type_total    = defaultdict(int)
    total_detected = 0
    total_count    = 0

    for masked, seg in zip(masked_texts, segments):
        masked_lower = masked.lower()
        for m in seg["gold_mentions"]:
            span   = m["span_text"].lower().strip()
            etype  = m.get("entity_type", "MISC")
            per_type_total[etype] += 1
            total_count += 1
            # "detected" = span text is no longer present verbatim in masked output
            if span not in masked_lower:
                per_type_detected[etype] += 1
                total_detected += 1

    overall = total_detected / total_count if total_count else 0.0
    per_type = {
        etype: per_type_detected[etype] / per_type_total[etype]
        for etype in per_type_total
    }
    return {
        "overall_recall": round(overall, 4),
        "per_type_recall": {k: round(v, 4) for k, v in per_type.items()},
        "detected_count": total_detected,
        "total_count":    total_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCHED TAB EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_tab(
    masker_model, masker_tok,
    filler_model, filler_tok,
    filler_cfg: dict,
    tab_data: list,
    combo_label: str,
) -> tuple:
    """
    Run masker + filler on all TAB segments using batched inference.

    Returns (metrics_dict, masker_recall_dict, samples_list).
    """
    all_preds     = []
    all_targets   = []   # gold_masked text used as reference for BLEU/ROUGE
    all_originals = []
    all_entities  = []
    all_masked    = []

    batches = [
        tab_data[i: i + EVAL_BATCH_SIZE]
        for i in range(0, len(tab_data), EVAL_BATCH_SIZE)
    ]

    for batch in tqdm(batches, desc=f"    TAB [{combo_label}]", leave=False):
        originals = [seg["original_text"] for seg in batch]
        targets   = [seg["gold_masked"]   for seg in batch]
        entities  = [seg["entity_texts"]  for seg in batch]

        # Batched masker
        masker_results = run_masker_batch(originals, masker_model, masker_tok)
        masked_texts   = [r[0] for r in masker_results]

        # Batched filler
        if filler_cfg["filler_type"] == "seq2seq":
            anon_texts = run_filler_seq2seq_batch(
                masked_texts, filler_model, filler_tok, filler_cfg["prompt_prefix"]
            )
        else:
            anon_texts = run_filler_mlm_batch(masked_texts, filler_model, filler_tok)

        all_preds.extend(anon_texts)
        all_targets.extend(targets)
        all_originals.extend(originals)
        all_entities.extend(entities)
        all_masked.extend(masked_texts)

    print(f"    Computing metrics on {len(all_preds)} predictions …")

    # ── Standard metrics (against gold_masked as reference) ──
    metrics = compute_all_metrics(
        preds=all_preds,
        targets=all_targets,
        original_texts=all_originals,
        entity_texts_list=all_entities,
        masked_texts=all_masked,
        compute_bert=True,
    )
    top10 = metrics.pop("leaked_entities_top10", [])
    metrics["leaked_entities_top10"] = [{"entity": e, "count": c} for e, c in top10]

    # ── Masker recall (TAB-specific) ──
    masker_recall = compute_masker_recall(all_masked, tab_data[:len(all_masked)])

    # ── Sample predictions ──
    samples = []
    for i in range(min(5, len(all_preds))):
        samples.append({
            "doc_id":     tab_data[i]["doc_id"],
            "original":   all_originals[i][:300],
            "gold_masked": all_targets[i][:300],
            "masked":     all_masked[i][:300],
            "prediction": all_preds[i][:300],
        })

    return metrics, masker_recall, samples


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_tab_report(all_results: list, tab_data: list, filepath: str) -> None:
    """Write a human-readable TAB evaluation report to *filepath*."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    lines = []
    W = 90

    def hr(char="═"): lines.append(char * W)
    def section(title): hr(); lines.append(f"  {title}"); hr()
    def blank(): lines.append("")

    # ── Header ────────────────────────────────────────────────────────────────
    hr("╔" + "═" * (W - 2) + "╗")
    lines.append("║" + "  TAB CROSS-DOMAIN EVALUATION REPORT".center(W - 2) + "║")
    lines.append("║" + "  Text Anonymization Benchmark (Pilán et al., 2022)".center(W - 2) + "║")
    lines.append("║" + "  ECHR Court Decisions · Zero-shot Transfer".center(W - 2) + "║")
    hr("╚" + "═" * (W - 2) + "╝")
    blank()

    lines.append(f"  Dataset        : ildpil/text-anonymization-benchmark (test split)")
    lines.append(f"  Segments eval  : {len(tab_data)}")
    lines.append(f"  Entity types   : PERSON, LOC, ORG, DATETIME, CODE, DEM, QUANTITY, MISC")
    lines.append(f"  Confidential   : identifier_type ∈ {{DIRECT, QUASI}}")
    lines.append(f"  Target for BLEU: gold-masked text (DIRECT/QUASI spans → [PLACEHOLDER])")
    blank()

    lines.append("  NOTE: These models were trained on AI4Privacy synthetic data.")
    lines.append("  TAB results measure zero-shot cross-domain transfer to real legal text.")
    blank()

    # ── Summary table ─────────────────────────────────────────────────────────
    section("RESULTS SUMMARY")
    blank()

    col_w = [30, 8, 8, 8, 8, 10, 11, 10]
    headers = ["Combo", "Exact", "Word%", "BLEU-4", "ROUGE-L", "BERTScore", "Mask Rec.", "Leak%"]
    row_fmt = "  {:<30} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}"
    lines.append(row_fmt.format(*headers))
    lines.append("  " + "─" * (W - 2))

    for r in all_results:
        m = r["metrics"]
        mr = r["masker_recall"]
        name = f"{r['masker_key']} + {r['filler_key']}"
        lines.append(row_fmt.format(
            name,
            f"{m.get('exact_match', 0):.3f}",
            f"{m.get('word_accuracy', 0):.3f}",
            f"{m.get('bleu_4', 0):.3f}",
            f"{m.get('rouge_l', 0):.3f}",
            f"{m.get('bertscore_f1', 0):.3f}",
            f"{mr['overall_recall']:.3f}",
            f"{m.get('sample_leakage_rate', 0)*100:.1f}%",
        ))
    blank()

    # ── Per-combo detailed results ─────────────────────────────────────────────
    section("DETAILED RESULTS PER PIPELINE COMBO")

    for r in all_results:
        blank()
        hr("─")
        lines.append(f"  Combo  : {r['masker_key']} + {r['filler_key']}")
        lines.append(f"  Masker : {MASKER_CONFIGS[r['masker_key']]['description']}")
        lines.append(f"  Filler : {FILLER_CONFIGS[r['filler_key']]['description']}")
        lines.append(f"  Runtime: {r.get('runtime_str', 'N/A')}")
        hr("─")

        m  = r["metrics"]
        mr = r["masker_recall"]

        blank()
        lines.append("  ── Masker Entity Recall (TAB gold annotations) ──")
        lines.append(f"    Overall recall : {mr['overall_recall']:.4f}"
                     f"  ({mr['detected_count']}/{mr['total_count']} entities masked)")
        lines.append("    Per entity type:")
        for etype, rec in sorted(mr["per_type_recall"].items(), key=lambda x: -x[1]):
            lines.append(f"      {etype:<12} {rec:.4f}")

        blank()
        lines.append("  ── Anonymization Metrics (prediction vs gold-masked) ──")
        lines.append(f"    Exact Match    : {m.get('exact_match', 0):.4f}")
        lines.append(f"    Word Accuracy  : {m.get('word_accuracy', 0):.4f}")
        lines.append(f"    BLEU-1         : {m.get('bleu_1', 0):.4f}")
        lines.append(f"    BLEU-2         : {m.get('bleu_2', 0):.4f}")
        lines.append(f"    BLEU-4         : {m.get('bleu_4', 0):.4f}")
        lines.append(f"    ROUGE-1        : {m.get('rouge_1', 0):.4f}")
        lines.append(f"    ROUGE-2        : {m.get('rouge_2', 0):.4f}")
        lines.append(f"    ROUGE-L        : {m.get('rouge_l', 0):.4f}")
        lines.append(f"    BERTScore P    : {m.get('bertscore_precision', 0):.4f}")
        lines.append(f"    BERTScore R    : {m.get('bertscore_recall', 0):.4f}")
        lines.append(f"    BERTScore F1   : {m.get('bertscore_f1', 0):.4f}")

        blank()
        lines.append("  ── Privacy Leakage ──")
        lines.append(f"    Entity Leakage Rate    : {m.get('entity_leakage_rate', 0):.4f}")
        lines.append(f"    Sample Leakage Rate    : {m.get('sample_leakage_rate', 0):.4f}")
        top10 = m.get("leaked_entities_top10", [])
        if top10:
            lines.append("    Top leaked entities:")
            for entry in top10[:5]:
                lines.append(f"      {entry['entity']!r:30s}  count={entry['count']}")

        blank()
        lines.append("  ── Sample Outputs ──")
        for i, s in enumerate(r.get("samples", [])[:3], 1):
            lines.append(f"  Sample {i} [doc_id={s.get('doc_id','')}]")
            lines.append(f"    Original    : {s['original'][:200]}")
            lines.append(f"    Gold masked : {s.get('gold_masked', 'N/A')[:200]}")
            lines.append(f"    Masked      : {s['masked'][:200]}")
            lines.append(f"    Prediction  : {s['prediction'][:200]}")
            blank()

    # ── Cross-domain observations ─────────────────────────────────────────────
    section("CROSS-DOMAIN OBSERVATION NOTES")
    blank()
    lines.append("  The TAB dataset uses real European Court of Human Rights (ECHR) decisions.")
    lines.append("  Models were trained exclusively on AI4Privacy synthetic text. Key differences:")
    blank()
    lines.append("  • Domain: AI4Privacy = short synthetic sentences; TAB = long legal paragraphs")
    lines.append("  • Entity types: AI4Privacy has ~50 fine-grained types; TAB uses 8 coarse types")
    lines.append("  • TAB DATETIME ≈ AI4Privacy DATE; TAB CODE ≈ AI4Privacy ID_NUMBER")
    lines.append("  • TAB PERSON spans often include titles (Mr, Ms) which may confuse the masker")
    lines.append("  • TAB entities tend to repeat across a document (coreference) — the first")
    lines.append("    masking may prevent the filler from being consistent across mentions")
    blank()

    lines.append("  Masker recall < pipeline recall — some entities not masked but not leaked")
    lines.append("  because the filler may rewrite text enough to obscure them naturally.")
    blank()

    # ── Footer ─────────────────────────────────────────────────────────────────
    hr()
    lines.append("  END OF TAB EVALUATION REPORT")
    hr()

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON SERIALISER
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load TAB segments ──────────────────────────────────────────────────────
    tab_data = load_tab_segments(split="test", limit=TAB_SEGMENT_LIMIT)
    if not tab_data:
        print("No TAB segments loaded. Exiting.")
        return

    # Count gold entities for summary
    total_gold = sum(len(s["gold_mentions"]) for s in tab_data)
    print(f"\nReady to evaluate on {len(tab_data)} segments  "
          f"({total_gold} gold DIRECT/QUASI entities)\n")

    all_results = []

    for masker_key, filler_key in COMBOS:
        combo_label = f"{masker_key}+{filler_key}"
        print(f"\n{'='*60}")
        print(f"  Combo: {combo_label}")
        print(f"{'='*60}")

        # Load models
        print("  Loading masker …")
        masker_model, masker_tok = load_masker(masker_key)

        print("  Loading filler …")
        filler_model, filler_tok = load_filler(filler_key)
        filler_cfg = FILLER_CONFIGS[filler_key]

        # Run evaluation
        t0 = time.time()
        metrics, masker_recall, samples = evaluate_on_tab(
            masker_model, masker_tok,
            filler_model, filler_tok,
            filler_cfg,
            tab_data,
            combo_label,
        )
        elapsed = time.time() - t0

        print(f"  ✓ Done in {format_time(elapsed)}")
        print(f"    Masker recall   : {masker_recall['overall_recall']:.3f}")
        print(f"    BERTScore F1    : {metrics.get('bertscore_f1', 0):.3f}")
        print(f"    Sample leak rate: {metrics.get('sample_leakage_rate', 0):.3f}")

        result = {
            "masker_key":    masker_key,
            "filler_key":    filler_key,
            "combo_label":   combo_label,
            "runtime_sec":   round(elapsed, 1),
            "runtime_str":   format_time(elapsed),
            "n_segments":    len(tab_data),
            "n_gold_entities": total_gold,
            "metrics":       metrics,
            "masker_recall": masker_recall,
            "samples":       samples,
        }
        all_results.append(result)

        # ── Incremental save ──────────────────────────────────────────────────
        write_tab_report(all_results, tab_data, TAB_RESULTS_TXT)
        with open(TAB_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(_sanitize(all_results), f, indent=2, ensure_ascii=False)
        print(f"  Results saved ({len(all_results)}/{len(COMBOS)} combos)")

        # ── Unload models ─────────────────────────────────────────────────────
        unload(masker_model, masker_tok)
        unload(filler_model, filler_tok)
        aggressive_cleanup()

    # ── Final save ─────────────────────────────────────────────────────────────
    write_tab_report(all_results, tab_data, TAB_RESULTS_TXT)
    with open(TAB_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(_sanitize(all_results), f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  TAB EVALUATION COMPLETE — {len(all_results)} combos")
    print(f"  Report : {TAB_RESULTS_TXT}")
    print(f"  JSON   : {TAB_RESULTS_JSON}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
