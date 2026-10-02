#!/usr/bin/env python3
"""
evaluate_both.py — Unified Evaluation of Both Adversarially Hardened Pipeline Fillers
========================================================================================
Evaluates BOTH hardened fillers side-by-side:

  Combo 1: RoBERTa NER  →  DeBERTa-MLM filler (JALAPENO11/pii-filler-deberta-adv-hardened)
  Combo 2: RoBERTa NER  →  BART seq2seq filler  (JALAPENO11/pii-filler-bart-adv-hardened)

Metrics computed for each combo:
  ──────────────────────────────────────────────────────────────────────────
  Privacy metrics:
    • ELR  — Entity Leakage Rate: fraction of gold PII entities that literally
             appear in the anonymized output. Lower = better privacy.
    • ERR  — Entity Recovery Rate: fraction of original tokens recovered by the
             frozen BART inverter from the anonymized text.
             Lower = better robustness against model-inversion attack.

  Utility metrics:
    • BLEU-4  — 4-gram overlap between anonymized output and gold anonymized ref.
    • ROUGE-L — Longest common subsequence F1 vs gold anonymized reference.
    • Semantic similarity (BERTScore) — optional, skipped if bert_score not available.

  Structural metrics:
    • Length ratio  — |anon| / |original| (token count ratio, 1.0 = same length)
    • [MASK] fill rate — fraction of all [MASK] tokens that were successfully replaced
                         (Combo 1 only)

Outputs:
  results/eval_report_both.json     — full JSON with all numbers
  results/eval_report_both.txt      — human-readable detailed report
  results/samples_combo1.txt        — 20 sample anonymizations for Combo 1
  results/samples_combo2.txt        — 20 sample anonymizations for Combo 2

Usage:
    python evaluate_both.py
    python evaluate_both.py --n_test 500 --batch 1 --beams 2
    python evaluate_both.py --n_test 200 --skip_inverter   # fast quality-only run
"""

import argparse
import gc
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ADV_EVAL_FILE,
    COMBO1_INVERTER_ID,
    COMBO2_INVERTER_ID,
    COMBO1_CKPT_DIR,
    COMBO2_CKPT_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)
from dataset import load_jsonl

# ── Constants ────────────────────────────────────────────────────────────────
NER_MASKER_ID = "Xyren2005/pii-ner-roberta"

# Hardened model IDs on HuggingFace (primary source; local ckpt used as fallback)
COMBO1_FILLER_BASE_ID     = "Xyren2005/pii-ner-filler_deberta-filler"
COMBO2_FILLER_BASE_ID     = "Xyren2005/pii-ner-filler_bart-base"
COMBO1_HARDENED_HF_ID     = "JALAPENO11/pii-filler-deberta-adv-hardened"
COMBO2_HARDENED_HF_ID     = "JALAPENO11/pii-filler-bart-adv-hardened"

# Pre-hardening baselines (from inversion experiment logs)
BASELINE_ERR_C1 = 0.7580   # 75.80%
BASELINE_ERR_C2 = 0.7683   # 76.83%
BASELINE_ELR_C1 = 0.0250   # 2.50%
BASELINE_ELR_C2 = 0.0133   # 1.33%

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--n_test",        type=int,   default=None,  help="Limit eval samples")
p.add_argument("--n_samples",     type=int,   default=20,    help="Sample anonymizations to show per combo")
p.add_argument("--batch",         type=int,   default=4,     help="Inference batch size")
p.add_argument("--beams",         type=int,   default=4,     help="Beam search width")
p.add_argument("--device",        default="cuda")
p.add_argument("--skip_inverter", action="store_true",       help="Skip ERR computation (faster)")
p.add_argument("--combo",         type=int,   default=0,     choices=[0, 1, 2],
               help="0 = both (default), 1 = combo1 only, 2 = combo2 only")
args = p.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
RESULTS_PATH = Path(RESULTS_DIR)
RESULTS_PATH.mkdir(exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  METRIC HELPERS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu4(hyp: str, ref: str) -> float:
    """Clipped 4-gram BLEU with brevity penalty."""
    h, r = hyp.lower().split(), ref.lower().split()
    if not h:
        return 0.0
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r) / len(h))
    scores = []
    for n in range(1, 5):
        hn, rn = _ngrams(h, n), _ngrams(r, n)
        if not hn:
            scores.append(0.0)
            continue
        m = sum(min(hn[k], rn[k]) for k in hn)
        scores.append(m / sum(hn.values()))
    if any(s == 0 for s in scores):
        return 0.0
    return bp * math.exp(sum(math.log(s + 1e-8) for s in scores) / 4)


def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 based on longest common subsequence."""
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if h[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m if m > 0 else 0.0
    rec  = lcs / n if n > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)


def corpus_bleu4(hyps: list[str], refs: list[str]) -> float:
    return round(sum(sentence_bleu4(h, r) for h, r in zip(hyps, refs))
                 / max(len(hyps), 1) * 100, 2)


def corpus_rougel(hyps: list[str], refs: list[str]) -> float:
    return round(sum(rouge_l(h, r) for h, r in zip(hyps, refs))
                 / max(len(hyps), 1) * 100, 2)


def entity_leakage_rate(preds: list[str], entity_lists: list[list[str]]) -> float:
    """Fraction of gold PII entities that appear verbatim in the prediction."""
    leaked = total = 0
    for pred, entities in zip(preds, entity_lists):
        for ent in entities:
            if not ent.strip():
                continue
            total += 1
            if ent.lower() in pred.lower():
                leaked += 1
    return round(leaked / max(total, 1), 4)


def entity_recovery_rate(inv_preds: list[str], originals: list[str]) -> float:
    """
    Token-level ERR: fraction of meaningful tokens from the original text that
    the inverter successfully recovered from the anonymized output.
    Only tokens with > 2 characters are counted (skip stopwords/punctuation).
    """
    exact = total = 0
    for pred, ref in zip(inv_preds, originals):
        tokens = [t for t in ref.lower().split() if len(t) > 2]
        total += len(tokens)
        for t in tokens:
            if t in pred.lower():
                exact += 1
    return round(exact / max(total, 1), 4)


def avg_length_ratio(hyps: list[str], refs: list[str]) -> float:
    """Average token-length ratio prediction/reference. 1.0 = same length."""
    ratios = [len(h.split()) / max(len(r.split()), 1) for h, r in zip(hyps, refs)]
    return round(sum(ratios) / max(len(ratios), 1), 3)


def mask_fill_rate(original_masked: list[str], filled: list[str]) -> float:
    """
    For Combo 1 (DeBERTa-MLM): what fraction of [MASK] tokens were replaced?
    A [MASK] token is considered 'filled' if it does NOT appear in the output.
    """
    total_masks = sum(t.count("[MASK]") for t in original_masked)
    remaining   = sum(t.count("[MASK]") for t in filled)
    if total_masks == 0:
        return 1.0
    return round(1.0 - remaining / total_masks, 4)


def try_bertscore(hyps: list[str], refs: list[str]) -> float | None:
    """Return corpus-level BERTScore F1 if bert_score package available, else None."""
    try:
        from bert_score import score as btscore  # type: ignore
        _, _, F1 = btscore(hyps, refs, lang="en", verbose=False)
        return round(F1.mean().item() * 100, 2)
    except ImportError:
        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL LOADING                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_ner_masker(device):
    print(f"  Loading NER masker: {NER_MASKER_ID}")
    tok   = AutoTokenizer.from_pretrained(NER_MASKER_ID, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(NER_MASKER_ID)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


def load_hardened_combo1(device):
    """
    Load hardened DeBERTa-MLM filler.
    Priority: local best_model.pt → local final_model.pt → HF JALAPENO11/pii-filler-deberta-adv-hardened
    Tokenizer always from HF (the hardened model repo contains the fine-tuned weights
    but the tokenizer architecture is identical to the base model).
    """
    print(f"  Loading hardened DeBERTa filler: {COMBO1_HARDENED_HF_ID}")
    tok = AutoTokenizer.from_pretrained(COMBO1_FILLER_BASE_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"

    # Try to load from HF directly (weights were pushed there after training)
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            COMBO1_HARDENED_HF_ID, torch_dtype=torch.float32
        )
        print(f"    ✓ Loaded from HuggingFace: {COMBO1_HARDENED_HF_ID}")
    except Exception as e:
        print(f"    HF load failed ({e}). Trying local checkpoint …")
        model = AutoModelForMaskedLM.from_pretrained(
            COMBO1_FILLER_BASE_ID, torch_dtype=torch.float32
        )
        for ckpt_name in ("best_model.pt", "final_model.pt"):
            ckpt_path = os.path.join(COMBO1_CKPT_DIR, ckpt_name)
            if os.path.exists(ckpt_path):
                ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                print(f"    ✓ Loaded local checkpoint: {ckpt_path}")
                break
        else:
            print("    WARNING: No local checkpoint found. Using unhardened base weights.")

    model.eval().to(device)
    return model, tok


def load_hardened_combo2(device):
    """
    Load hardened BART seq2seq filler.
    Priority: HF JALAPENO11/pii-filler-bart-adv-hardened → local checkpoint
    """
    print(f"  Loading hardened BART filler: {COMBO2_HARDENED_HF_ID}")
    tok = AutoTokenizer.from_pretrained(COMBO2_FILLER_BASE_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            COMBO2_HARDENED_HF_ID, torch_dtype=torch.float32
        )
        print(f"    ✓ Loaded from HuggingFace: {COMBO2_HARDENED_HF_ID}")
    except Exception as e:
        print(f"    HF load failed ({e}). Trying local checkpoint …")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            COMBO2_FILLER_BASE_ID, torch_dtype=torch.float32
        )
        for ckpt_name in ("best_model.pt", "final_model.pt"):
            ckpt_path = os.path.join(COMBO2_CKPT_DIR, ckpt_name)
            if os.path.exists(ckpt_path):
                ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                print(f"    ✓ Loaded local checkpoint: {ckpt_path}")
                break
        else:
            print("    WARNING: No local checkpoint found. Using unhardened base weights.")

    model.eval().to(device)
    return model, tok


def load_frozen_inverter(inverter_id: str, device):
    print(f"  Loading frozen inverter: {inverter_id}")
    tok   = AutoTokenizer.from_pretrained(inverter_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        inverter_id, torch_dtype=torch.float16
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE HELPERS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def run_ner_mask(texts: list[str], ner_model, ner_tok, device,
                 combo: int) -> list[str]:
    """
    Run RoBERTa NER masker on free-form text.
    combo=1: replaces entity spans with [MASK]
    combo=2: replaces entity spans with [TYPE] and prepends instruction prefix
    """
    masked = []
    for text in tqdm(texts, desc=f"  NER masking (combo {combo})", leave=False, ncols=90):
        words = text.split()
        if not words:
            masked.append(text)
            continue
        enc = ner_tok(
            words,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            is_split_into_words=True,
            padding=True,
        ).to(device)
        logits   = ner_model(**enc).logits
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
        word_ids = enc.word_ids(batch_index=0)

        id2lbl: dict[int, str] = ner_model.config.id2label
        word_tags: dict[int, str] = {}
        for tok_idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_tags:
                word_tags[wid] = id2lbl.get(pred_ids[tok_idx], "O")

        result, i = [], 0
        while i < len(words):
            tag = word_tags.get(i, "O")
            if tag.startswith("B-"):
                etype = tag[2:]
                j = i + 1
                while j < len(words) and word_tags.get(j, "O") == f"I-{etype}":
                    j += 1
                result.append("[MASK]" if combo == 1 else f"[{etype}]")
                i = j
            else:
                result.append(words[i])
                i += 1

        out = " ".join(result)
        if combo == 2:
            out = "Replace PII placeholders with realistic fake entities: " + out
        masked.append(out)
    return masked


@torch.no_grad()
def run_filler_combo1(masked_texts: list[str], filler, filler_tok,
                      batch_size: int, device) -> list[str]:
    """DeBERTa-MLM: predict top-1 token at each [MASK] position."""
    outputs = []
    mask_id = filler_tok.mask_token_id
    for i in tqdm(range(0, len(masked_texts), batch_size),
                  desc="  Filler (DeBERTa-MLM)", unit="batch", ncols=90):
        batch = masked_texts[i: i + batch_size]
        enc = filler_tok(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_LENGTH,
        ).to(device)
        logits   = filler(**enc).logits          # (B, T, V)
        pred_ids = logits.argmax(dim=-1).cpu()   # (B, T)
        in_ids   = enc["input_ids"].cpu()

        for j, text in enumerate(batch):
            mask_positions = (in_ids[j] == mask_id).nonzero(as_tuple=True)[0].tolist()
            result = text
            for pos in mask_positions:
                tok_str = filler_tok.decode([pred_ids[j, pos].item()]).strip()
                result  = result.replace("[MASK]", tok_str or "[UNK]", 1)
            outputs.append(result)
    return outputs


@torch.no_grad()
def run_filler_combo2(masked_texts: list[str], filler, filler_tok,
                      batch_size: int, num_beams: int, device) -> list[str]:
    """BART seq2seq filler: beam-search decode."""
    outputs = []
    for i in tqdm(range(0, len(masked_texts), batch_size),
                  desc="  Filler (BART)", unit="batch", ncols=90):
        batch = masked_texts[i: i + batch_size]
        enc = filler_tok(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_LENGTH,
        ).to(device)
        gen = filler.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=MAX_TARGET_LENGTH,
            num_beams=num_beams,
        )
        outputs.extend(filler_tok.batch_decode(gen, skip_special_tokens=True))
    return outputs


@torch.no_grad()
def run_inverter(anon_texts: list[str], inverter, inverter_tok,
                 batch_size: int, num_beams: int, device) -> list[str]:
    """Frozen BART inverter: attempt to recover original text from anonymized."""
    outputs = []
    for i in tqdm(range(0, len(anon_texts), batch_size),
                  desc="  Inverter (BART)", unit="batch", ncols=90):
        batch = anon_texts[i: i + batch_size]
        enc = inverter_tok(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_LENGTH,
        ).to(device)
        gen = inverter.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_length=MAX_TARGET_LENGTH,
            num_beams=num_beams,
        )
        outputs.extend(inverter_tok.batch_decode(gen, skip_special_tokens=True))
    return outputs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PER-COMBO EVALUATION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_combo(
    combo_id: int,
    eval_data: list[dict],
    filler,
    filler_tok,
    inverter,
    inverter_tok,
    ner_model,
    ner_tok,
    device,
    batch_size: int,
    num_beams: int,
    skip_inverter: bool,
    n_samples: int,
) -> dict:
    """
    Full evaluation pipeline for one combo.
    Returns a dict with all metric values + samples list.
    """
    originals    = [r["original"]           for r in eval_data]
    gold_anon    = [r["anonymized"]         for r in eval_data]
    entity_lists = [r.get("entity_texts", []) for r in eval_data]
    masked_field = "masked_mlm" if combo_id == 1 else "masked_s2s"
    gold_masked  = [r.get(masked_field, r.get("masked_mlm", "")) for r in eval_data]

    # ── PART A: quality on pre-computed masked forms ─────────────────────
    print(f"\n  ── Part A: Quality (pre-computed masked inputs) ──")
    if combo_id == 1:
        preds_A = run_filler_combo1(gold_masked, filler, filler_tok, batch_size, device)
    else:
        preds_A = run_filler_combo2(gold_masked, filler, filler_tok, batch_size, num_beams, device)

    elr_A    = entity_leakage_rate(preds_A, entity_lists)
    bleu4_A  = corpus_bleu4(preds_A, gold_anon)
    rouge_A  = corpus_rougel(preds_A, gold_anon)
    exact_A  = round(sum(p.strip() == r.strip() for p, r in zip(preds_A, gold_anon))
                     / max(len(preds_A), 1), 4)
    lenr_A   = avg_length_ratio(preds_A, gold_anon)
    print(f"    ELR          : {elr_A:.4f}   (lower = better)")
    print(f"    BLEU-4       : {bleu4_A:.2f}")
    print(f"    ROUGE-L      : {rouge_A:.2f}")
    print(f"    Exact Match  : {exact_A:.4f}")
    print(f"    Length Ratio : {lenr_A:.3f}   (1.0 = same length as gold)")

    fillrate_A = None
    if combo_id == 1:
        fillrate_A = mask_fill_rate(gold_masked, preds_A)
        print(f"    Mask Fill %  : {fillrate_A:.4f}   (fraction of [MASK] tokens replaced)")

    bertscore_A = try_bertscore(preds_A, gold_anon)
    if bertscore_A is not None:
        print(f"    BERTScore F1 : {bertscore_A:.2f}")

    # ── PART B: end-to-end adversarial robustness ─────────────────────────
    print(f"\n  ── Part B: End-to-End Adversarial Robustness ──")
    print("    Step 1/3 — Live NER masking on originals …")
    masked_live = run_ner_mask(originals, ner_model, ner_tok, device, combo_id)

    print("    Step 2/3 — Hardened filler …")
    if combo_id == 1:
        anon_B = run_filler_combo1(masked_live, filler, filler_tok, batch_size, device)
    else:
        anon_B = run_filler_combo2(masked_live, filler, filler_tok, batch_size, num_beams, device)

    elr_B   = entity_leakage_rate(anon_B, entity_lists)
    bleu4_B = corpus_bleu4(anon_B, gold_anon)
    rouge_B = corpus_rougel(anon_B, gold_anon)
    lenr_B  = avg_length_ratio(anon_B, originals)

    err_B   = None
    inv_preds = None
    if not skip_inverter:
        print("    Step 3/3 — Frozen inverter attack on anonymized outputs …")
        inv_preds = run_inverter(anon_B, inverter, inverter_tok, batch_size, num_beams, device)
        err_B     = entity_recovery_rate(inv_preds, originals)
        print(f"    ERR (inverter recovery): {err_B:.4f}  ← lower = more robust")
    else:
        print("    Step 3/3 — [SKIPPED] (--skip_inverter flag set)")

    print(f"    ELR (live pipeline) : {elr_B:.4f}")
    print(f"    BLEU-4 vs gold      : {bleu4_B:.2f}")
    print(f"    ROUGE-L vs gold     : {rouge_B:.2f}")
    print(f"    Length ratio        : {lenr_B:.3f}   (vs original len)")

    fillrate_B = None
    if combo_id == 1:
        fillrate_B = mask_fill_rate(masked_live, anon_B)
        print(f"    Mask Fill %         : {fillrate_B:.4f}")

    # ── Samples ───────────────────────────────────────────────────────────
    n = min(n_samples, len(originals))
    idxs = random.sample(range(len(originals)), n)
    samples = []
    for i in idxs:
        inv_out = inv_preds[i] if inv_preds is not None else "(inverter skipped)"
        samples.append({
            "original":  originals[i],
            "masked":    masked_live[i],
            "anonymized": anon_B[i],
            "gold_anon":  gold_anon[i],
            "inv_recovered": inv_out,
        })

    return {
        "part_A": {
            "description": "Quality on pre-computed masked inputs (adv_eval.jsonl)",
            "ELR":         elr_A,
            "BLEU4":       bleu4_A,
            "ROUGEL":      rouge_A,
            "ExactMatch":  exact_A,
            "LengthRatio": lenr_A,
            "MaskFillRate": fillrate_A,
            "BERTScoreF1": bertscore_A,
        },
        "part_B": {
            "description": "End-to-end live pipeline robustness",
            "ERR":         err_B,
            "ELR":         elr_B,
            "BLEU4":       bleu4_B,
            "ROUGEL":      rouge_B,
            "LengthRatio": lenr_B,
            "MaskFillRate": fillrate_B,
        },
        "samples": samples,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REPORT FORMATTERS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def fmt_metric(label: str, val, suffix: str = "", baseline=None, lower_better=True) -> str:
    if val is None:
        return f"    {label:<28}: N/A"
    v_str = f"{val:.4f}" if isinstance(val, float) else str(val)
    line  = f"    {label:<28}: {v_str}{suffix}"
    if baseline is not None:
        delta = val - baseline
        sign  = "+" if delta >= 0 else ""
        indicator = "↓ better" if (lower_better and delta < 0) else ("↑ better" if (not lower_better and delta > 0) else "—")
        line += f"  (baseline {baseline:.4f}, Δ={sign}{delta:.4f} {indicator})"
    return line


def write_text_report(
    out_path: Path,
    n_samples_total: int,
    res1: dict | None,
    res2: dict | None,
    eval_combos: list[int],
    timestamp: str,
) -> str:
    sep = "=" * 80

    lines = [
        sep,
        "  ADVERSARIAL HARDENING — UNIFIED EVALUATION REPORT",
        f"  Generated : {timestamp}",
        f"  Eval set  : {n_samples_total:,} samples from adv_eval.jsonl",
        sep,
        "",
        "  HARDENED MODELS",
        f"    Combo 1 : {COMBO1_HARDENED_HF_ID}  (DeBERTa-MLM filler)",
        f"    Combo 2 : {COMBO2_HARDENED_HF_ID}  (BART seq2seq filler)",
        f"    NER     : {NER_MASKER_ID}  (shared, frozen)",
        "",
        "  METRIC GUIDE",
        "    ELR  = Entity Leakage Rate — fraction of gold PII entities appearing",
        "           verbatim in anonymized output. Lower = better privacy.",
        "    ERR  = Entity Recovery Rate — fraction of original tokens recovered by",
        "           frozen BART inverter. Lower = better attack resistance.",
        "    BLEU-4 / ROUGE-L — n-gram / LCS overlap vs gold anonymized text.",
        "           Higher = output closer to expected anonymization.",
        "    Length Ratio — token length of output / token length of reference.",
        "           1.0 = same length as gold; <1.0 = shorter output.",
        "    Mask Fill Rate — fraction of [MASK] tokens that were filled",
        "           (Combo 1 only). 1.0 = all masks replaced.",
        "",
    ]

    for combo_id, res in [(1, res1), (2, res2)]:
        if combo_id not in eval_combos or res is None:
            continue
        A, B = res["part_A"], res["part_B"]
        filler_id = COMBO1_HARDENED_HF_ID if combo_id == 1 else COMBO2_HARDENED_HF_ID
        inv_id    = COMBO1_INVERTER_ID    if combo_id == 1 else COMBO2_INVERTER_ID
        b_err     = BASELINE_ERR_C1 if combo_id == 1 else BASELINE_ERR_C2
        b_elr     = BASELINE_ELR_C1 if combo_id == 1 else BASELINE_ELR_C2

        lines += [
            sep,
            f"  COMBO {combo_id} — {'DeBERTa-MLM filler' if combo_id==1 else 'BART seq2seq filler'}",
            f"  Filler   : {filler_id}",
            f"  Inverter : {inv_id}  (frozen attacker)",
            sep,
            "",
            "  PART A — Anonymization Quality (pre-computed masked forms)",
            "  ─────────────────────────────────────────────────────────",
            fmt_metric("ELR (entity leakage)", A["ELR"],
                       suffix="", baseline=b_elr, lower_better=True),
            fmt_metric("BLEU-4", A["BLEU4"], suffix=" / 100",
                       baseline=None, lower_better=False),
            fmt_metric("ROUGE-L", A["ROUGEL"], suffix=" / 100",
                       baseline=None, lower_better=False),
            fmt_metric("Exact Match", A["ExactMatch"],
                       baseline=None, lower_better=False),
            fmt_metric("Length Ratio (vs gold)", A["LengthRatio"],
                       baseline=None, lower_better=False),
        ]
        if A["MaskFillRate"] is not None:
            lines.append(fmt_metric("Mask Fill Rate", A["MaskFillRate"],
                                    baseline=None, lower_better=False))
        if A["BERTScoreF1"] is not None:
            lines.append(fmt_metric("BERTScore F1", A["BERTScoreF1"],
                                    suffix=" / 100", baseline=None, lower_better=False))

        lines += [
            "",
            "  PART B — End-to-End Adversarial Robustness (live pipeline)",
            "  ─────────────────────────────────────────────────────────────",
        ]

        if B["ERR"] is not None:
            lines.append(fmt_metric("ERR  (inverter attack)", B["ERR"],
                                    baseline=b_err, lower_better=True))
            delta_err = B["ERR"] - b_err
            reduction = abs(delta_err) / b_err * 100
            lines.append(f"    {'ERR reduction':28}: {reduction:.1f}%  ({b_err*100:.1f}% → {B['ERR']*100:.1f}%)")
        else:
            lines.append("    ERR (inverter attack)       : N/A  (--skip_inverter)")

        lines += [
            fmt_metric("ELR  (live pipeline)", B["ELR"],
                       baseline=b_elr, lower_better=True),
            fmt_metric("BLEU-4 vs gold", B["BLEU4"], suffix=" / 100",
                       baseline=None, lower_better=False),
            fmt_metric("ROUGE-L vs gold", B["ROUGEL"], suffix=" / 100",
                       baseline=None, lower_better=False),
            fmt_metric("Length Ratio (vs orig)", B["LengthRatio"],
                       baseline=None, lower_better=False),
        ]
        if B["MaskFillRate"] is not None:
            lines.append(fmt_metric("Mask Fill Rate", B["MaskFillRate"],
                                    baseline=None, lower_better=False))
        lines.append("")

    # Side-by-side summary table
    if 1 in eval_combos and 2 in eval_combos and res1 and res2:
        lines += [
            sep,
            "  SIDE-BY-SIDE SUMMARY",
            sep,
            f"  {'Metric':<30} {'Combo 1 (DeBERTa)':>18} {'Combo 2 (BART)':>15} {'Winner':>10}",
            "  " + "-" * 76,
        ]

        def row(label, v1, v2, lower_better=True, suffix=""):
            if v1 is None or v2 is None:
                return f"  {label:<30} {'N/A':>18} {'N/A':>15} {'—':>10}"
            v1s = f"{v1:.4f}{suffix}" if isinstance(v1, float) else str(v1)
            v2s = f"{v2:.4f}{suffix}" if isinstance(v2, float) else str(v2)
            if lower_better:
                winner = "Combo 1" if v1 <= v2 else "Combo 2"
            else:
                winner = "Combo 1" if v1 >= v2 else "Combo 2"
            return f"  {label:<30} {v1s:>18} {v2s:>15} {winner:>10}"

        B1, B2 = res1["part_B"], res2["part_B"]
        A1, A2 = res1["part_A"], res2["part_A"]
        lines += [
            row("ERR ↓ (robustness, Part B)", B1["ERR"], B2["ERR"], lower_better=True),
            row("ELR ↓ (leakage, Part B)",    B1["ELR"], B2["ELR"], lower_better=True),
            row("BLEU-4 ↑ (quality, Part A)", A1["BLEU4"], A2["BLEU4"], lower_better=False, suffix=""),
            row("ROUGE-L ↑ (quality, Part A)", A1["ROUGEL"], A2["ROUGEL"], lower_better=False, suffix=""),
            row("ELR ↓ (leakage, Part A)",    A1["ELR"], A2["ELR"], lower_better=True),
            "",
            "  Pre-hardening baselines (for reference):",
            f"  {'ERR baseline':30} {'75.80%':>18} {'76.83%':>15}",
            f"  {'ELR baseline':30} {'2.50%':>18} {'1.33%':>15}",
        ]

    lines += ["", sep]
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


def write_samples_file(out_path: Path, combo_id: int, samples: list[dict]):
    filler_id = COMBO1_HARDENED_HF_ID if combo_id == 1 else COMBO2_HARDENED_HF_ID
    sep   = "─" * 80

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"SAMPLE ANONYMIZATIONS — COMBO {combo_id}  "
                f"({'DeBERTa-MLM' if combo_id==1 else 'BART seq2seq'})\n")
        f.write(f"Model: {filler_id}\n")
        f.write(f"Total shown: {len(samples)}\n")
        f.write("=" * 80 + "\n\n")
        for k, s in enumerate(samples, 1):
            f.write(sep + "\n")
            f.write(f"  Sample {k:>2}/{len(samples)}\n")
            f.write(sep + "\n")
            f.write(f"  ORIGINAL    : {s['original']}\n")
            f.write(f"  MASKED      : {s['masked']}\n")
            f.write(f"  ANONYMIZED  : {s['anonymized']}\n")
            f.write(f"  GOLD ANON   : {s['gold_anon']}\n")
            f.write(f"  INV RECOVER : {s['inv_recovered']}\n")
            f.write("\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    timestamp = datetime.now().isoformat(timespec="seconds")
    sep = "=" * 80

    run_combo1 = args.combo in (0, 1)
    run_combo2 = args.combo in (0, 2)
    eval_combos = []
    if run_combo1:
        eval_combos.append(1)
    if run_combo2:
        eval_combos.append(2)

    print(sep)
    print("  ADVERSARIAL HARDENING — UNIFIED EVALUATION")
    print(f"  Combos     : {eval_combos}")
    print(f"  Device     : {DEVICE}")
    print(f"  Batch size : {args.batch}   Beams: {args.beams}")
    print(f"  Samples    : {args.n_samples} per combo")
    print(f"  Skip inv.  : {args.skip_inverter}")
    print(sep)

    # ── Load eval data ────────────────────────────────────────────────────
    print("\n[1/5] Loading eval data …")
    eval_data = load_jsonl(ADV_EVAL_FILE)
    if args.n_test:
        eval_data = eval_data[: args.n_test]
    print(f"  Eval samples: {len(eval_data):,}")

    # ── Load shared NER masker ─────────────────────────────────────────────
    print("\n[2/5] Loading shared NER masker …")
    ner_model, ner_tok = load_ner_masker(DEVICE)

    res1 = res2 = None

    # ══════════════════ COMBO 1 ═══════════════════════════════════════════
    if run_combo1:
        print(f"\n[3/5] Loading Combo 1 models …")
        filler1, ftok1 = load_hardened_combo1(DEVICE)
        inv1, itok1 = None, None
        if not args.skip_inverter:
            inv1, itok1 = load_frozen_inverter(COMBO1_INVERTER_ID, DEVICE)

        print(f"\n{'─'*80}")
        print(f"  EVALUATING COMBO 1 — DeBERTa-MLM Filler")
        print(f"{'─'*80}")

        res1 = evaluate_combo(
            combo_id=1,
            eval_data=eval_data,
            filler=filler1, filler_tok=ftok1,
            inverter=inv1, inverter_tok=itok1,
            ner_model=ner_model, ner_tok=ner_tok,
            device=DEVICE,
            batch_size=args.batch,
            num_beams=args.beams,
            skip_inverter=args.skip_inverter,
            n_samples=args.n_samples,
        )

        # Free GPU memory before next combo
        del filler1, ftok1
        if inv1 is not None:
            del inv1, itok1
        gc.collect()
        torch.cuda.empty_cache()

    # ══════════════════ COMBO 2 ═══════════════════════════════════════════
    if run_combo2:
        print(f"\n[4/5] Loading Combo 2 models …")
        filler2, ftok2 = load_hardened_combo2(DEVICE)
        inv2, itok2 = None, None
        if not args.skip_inverter:
            inv2, itok2 = load_frozen_inverter(COMBO2_INVERTER_ID, DEVICE)

        print(f"\n{'─'*80}")
        print(f"  EVALUATING COMBO 2 — BART seq2seq Filler")
        print(f"{'─'*80}")

        res2 = evaluate_combo(
            combo_id=2,
            eval_data=eval_data,
            filler=filler2, filler_tok=ftok2,
            inverter=inv2, inverter_tok=itok2,
            ner_model=ner_model, ner_tok=ner_tok,
            device=DEVICE,
            batch_size=args.batch,
            num_beams=args.beams,
            skip_inverter=args.skip_inverter,
            n_samples=args.n_samples,
        )

        del filler2, ftok2
        if inv2 is not None:
            del inv2, itok2
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save report ────────────────────────────────────────────────────────
    print(f"\n[5/5] Writing reports …")

    out_json  = RESULTS_PATH / "eval_report_both.json"
    out_txt   = RESULTS_PATH / "eval_report_both.txt"
    out_s1    = RESULTS_PATH / "samples_combo1.txt"
    out_s2    = RESULTS_PATH / "samples_combo2.txt"

    full_report = {
        "timestamp":   timestamp,
        "n_samples":   len(eval_data),
        "combos_run":  eval_combos,
        "skip_inverter": args.skip_inverter,
        "hardened_models": {
            "combo1": COMBO1_HARDENED_HF_ID,
            "combo2": COMBO2_HARDENED_HF_ID,
        },
        "baselines": {
            "ERR_pre_hardening_combo1": BASELINE_ERR_C1,
            "ERR_pre_hardening_combo2": BASELINE_ERR_C2,
            "ELR_pre_hardening_combo1": BASELINE_ELR_C1,
            "ELR_pre_hardening_combo2": BASELINE_ELR_C2,
        },
    }
    if res1:
        r1_copy = {k: v for k, v in res1.items() if k != "samples"}
        full_report["combo1"] = r1_copy
    if res2:
        r2_copy = {k: v for k, v in res2.items() if k != "samples"}
        full_report["combo2"] = r2_copy

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"  JSON  → {out_json}")

    report_text = write_text_report(
        out_txt, len(eval_data), res1, res2, eval_combos, timestamp
    )
    print(f"  TXT   → {out_txt}")
    print("\n" + report_text)

    if res1 and res1.get("samples"):
        write_samples_file(out_s1, 1, res1["samples"])
        print(f"  Samples Combo 1 → {out_s1}")

    if res2 and res2.get("samples"):
        write_samples_file(out_s2, 2, res2["samples"])
        print(f"  Samples Combo 2 → {out_s2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
