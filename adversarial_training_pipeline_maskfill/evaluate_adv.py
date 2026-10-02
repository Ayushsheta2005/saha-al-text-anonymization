#!/usr/bin/env python3
"""
evaluate_adv.py — Full Evaluation of Adversarially Hardened Pipeline Fillers
==============================================================================
Evaluates the hardened filler on two axes:

  A. Anonymization quality (adv_eval.jsonl)
       - Entity Leakage Rate (ELR)  ← primary privacy metric
       - BERTScore F1               ← primary utility metric
       - BLEU-4, ROUGE-L

  B. Adversarial robustness (end-to-end pipeline evaluation)
       Full pipeline:  original → frozen NER masker → hardened filler → anonymized
       Then:           anonymized → frozen BART inverter → recovered text
       Measure:        Entity Recovery Rate (ERR)  ← primary attack metric
       Compare against pre-hardening ERR from the inversion dataset results.

Usage:
    python evaluate_adv.py --combo 2
    python evaluate_adv.py --combo 1
    python evaluate_adv.py --combo 2 --n_test 1000  # quick test
"""

import argparse
import gc
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ADV_EVAL_FILE,
    COMBO1_FILLER_ID, COMBO2_FILLER_ID,
    COMBO1_INVERTER_ID, COMBO2_INVERTER_ID,
    COMBO1_CKPT_DIR, COMBO2_CKPT_DIR,
    LOGS_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
)
from dataset import PipelineAdvDataset, pipeline_adv_collate_fn, load_jsonl

# NER masker shared by both combos
NER_MASKER_ID = "Xyren2005/pii-ner-roberta"

# ── CLI ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--combo",    type=int, choices=[1, 2], default=2)
p.add_argument("--n_test",   type=int, default=None,   help="Limit test samples (debug)")
p.add_argument("--batch",    type=int, default=16)
p.add_argument("--beams",    type=int, default=4)
p.add_argument("--device",   default="cuda")
args = p.parse_args()

COMBO = args.combo

if COMBO == 1:
    FILLER_ID   = COMBO1_FILLER_ID
    INVERTER_ID = COMBO1_INVERTER_ID
    CKPT_DIR    = COMBO1_CKPT_DIR
else:
    FILLER_ID   = COMBO2_FILLER_ID
    INVERTER_ID = COMBO2_INVERTER_ID
    CKPT_DIR    = COMBO2_CKPT_DIR

DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
os.makedirs(LOGS_DIR, exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Metric helpers ───────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not h:
        return 0.0
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r) / len(h))
    scores = []
    for n in range(1, max_n + 1):
        hn, rn = _ngrams(h, n), _ngrams(r, n)
        if not hn:
            scores.append(0.0); continue
        m = sum(min(hn[k], rn[k]) for k in hn)
        scores.append(m / sum(hn.values()))
    if any(s == 0 for s in scores):
        return 0.0
    return bp * math.exp(sum(math.log(s + 1e-8) for s in scores) / len(scores))


def entity_leakage_rate(predictions: list[str], entity_lists: list[list[str]]) -> float:
    leaked = total = 0
    for pred, entities in zip(predictions, entity_lists):
        for ent in entities:
            if not ent.strip():
                continue
            total += 1
            if ent.lower() in pred.lower():
                leaked += 1
    return round(leaked / max(total, 1), 4)


def entity_recovery_rate(inv_preds: list[str], originals: list[str]) -> float:
    """Token-level ERR: fraction of >2-char original tokens found in inverter prediction."""
    exact = total = 0
    for pred, ref in zip(inv_preds, originals):
        tokens = [t for t in ref.lower().split() if len(t) > 2]
        total += len(tokens)
        for t in tokens:
            if t in pred.lower():
                exact += 1
    return round(exact / max(total, 1), 4)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_hardened_filler(ckpt_dir: str):
    """Load the adversarially hardened filler from local checkpoint."""
    final = os.path.join(ckpt_dir, "final_model.pt")
    best  = os.path.join(ckpt_dir, "best_model.pt")
    path  = final if os.path.exists(final) else best

    tok = AutoTokenizer.from_pretrained(FILLER_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"

    if COMBO == 1:
        model = AutoModelForMaskedLM.from_pretrained(FILLER_ID)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(FILLER_ID)

    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"Loaded hardened filler from {path}")
    else:
        print(f"WARNING: No checkpoint found at {path}. Using HF pretrained weights.")

    model = model.to(DEVICE).eval()
    return model, tok


def load_frozen_inverter():
    tok   = AutoTokenizer.from_pretrained(INVERTER_ID, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(INVERTER_ID, torch_dtype=torch.float16)
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(DEVICE).eval()
    print(f"Loaded frozen inverter: {INVERTER_ID}")
    return model, tok


def load_ner_masker():
    tok   = AutoTokenizer.from_pretrained(NER_MASKER_ID, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(NER_MASKER_ID)
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(DEVICE).eval()
    print(f"Loaded NER masker: {NER_MASKER_ID}")
    return model, tok


# ── Pipeline inference helpers ────────────────────────────────────────────────

@torch.no_grad()
def run_ner_mask(texts: list[str], ner_model, ner_tok) -> list[str]:
    """Run RoBERTa NER masker to get masked texts (one [MASK] per entity span)."""
    masked = []
    for text in tqdm(texts, desc="NER masking", leave=False):
        words = text.split()
        enc = ner_tok(
            words,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            is_split_into_words=True,
            padding=True,
        ).to(DEVICE)
        logits = ner_model(**enc).logits
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
        word_ids = enc.word_ids(batch_index=0)

        id2lbl = ner_model.config.id2label
        # Assign tag to each word (first subword token's prediction)
        word_tags: dict[int, str] = {}
        for tok_idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_tags:
                word_tags[wid] = id2lbl.get(pred_ids[tok_idx], "O")

        # Build masked text: collapse entity spans into [TYPE] or [MASK]
        result, i = [], 0
        while i < len(words):
            tag = word_tags.get(i, "O")
            if tag.startswith("B-"):
                etype = tag[2:]
                j = i + 1
                while j < len(words) and word_tags.get(j, "O") == f"I-{etype}":
                    j += 1
                if COMBO == 1:
                    result.append("[MASK]")
                else:
                    result.append(f"[{etype}]")
                i = j
            else:
                result.append(words[i])
                i += 1

        masked_text = " ".join(result)
        if COMBO == 2:
            masked_text = "Replace PII placeholders with realistic fake entities: " + masked_text
        masked.append(masked_text)
    return masked


@torch.no_grad()
def run_filler_combo2(masked_texts: list[str], filler, filler_tok,
                      batch_size=16, num_beams=4) -> list[str]:
    outputs = []
    batches = range(0, len(masked_texts), batch_size)
    for i in tqdm(batches, desc="Filler (BART)", unit="batch"):
        batch = masked_texts[i: i + batch_size]
        enc = filler_tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)
        gen = filler.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=MAX_TARGET_LENGTH,
            num_beams=num_beams,
        )
        outputs.extend(filler_tok.batch_decode(gen, skip_special_tokens=True))
    return outputs


@torch.no_grad()
def run_filler_combo1(masked_texts: list[str], filler, filler_tok,
                      batch_size=16) -> list[str]:
    """DeBERTa-MLM: predict top-1 token at each [MASK] position."""
    outputs = []
    mask_id = filler_tok.mask_token_id
    batches = range(0, len(masked_texts), batch_size)
    for i in tqdm(batches, desc="Filler (DeBERTa)", unit="batch"):
        batch = masked_texts[i: i + batch_size]
        enc = filler_tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)
        logits   = filler(**enc).logits                         # (B, T, V)
        pred_ids = logits.argmax(dim=-1).cpu()                  # (B, T)
        in_ids   = enc["input_ids"].cpu()

        for j, text in enumerate(batch):
            mask_positions = (in_ids[j] == mask_id).nonzero(as_tuple=True)[0].tolist()
            result = text
            for pos in mask_positions:
                replacement = filler_tok.decode([pred_ids[j, pos].item()]).strip()
                result = result.replace("[MASK]", replacement, 1)
            outputs.append(result)
    return outputs


@torch.no_grad()
def run_inverter(anonymized_texts: list[str], inverter, inverter_tok,
                 batch_size=16, num_beams=4) -> list[str]:
    outputs = []
    batches = range(0, len(anonymized_texts), batch_size)
    for i in tqdm(batches, desc="Inverter (BART)", unit="batch"):
        batch = anonymized_texts[i: i + batch_size]
        enc = inverter_tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)
        gen = inverter.generate(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
            max_length=MAX_TARGET_LENGTH,
            num_beams=num_beams,
        )
        outputs.extend(inverter_tok.batch_decode(gen, skip_special_tokens=True))
    return outputs


# ── Main evaluation ─────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"EVALUATE ADV HARDENING — Pipeline Combo {COMBO}")
    print(f"Filler   : {FILLER_ID}  (hardened)")
    print(f"Inverter : {INVERTER_ID}  (frozen)")
    print("=" * 65)

    # -- Load models ---------------------------------------------------------
    filler,   filler_tok   = load_hardened_filler(CKPT_DIR)
    inverter, inverter_tok = load_frozen_inverter()
    ner_model, ner_tok     = load_ner_masker()

    # -- Load eval data ------------------------------------------------------
    eval_data = load_jsonl(ADV_EVAL_FILE)
    if args.n_test:
        eval_data = eval_data[: args.n_test]
    print(f"Eval samples: {len(eval_data):,}")

    originals    = [r["original"]   for r in eval_data]
    gold_anon    = [r["anonymized"] for r in eval_data]
    entity_lists = [r.get("entity_texts", []) for r in eval_data]
    masked_field = "masked_mlm" if COMBO == 1 else "masked_s2s"
    masked_texts = [r[masked_field] for r in eval_data]

    # ── Part A: Quality on gold eval set ────────────────────────────────────
    print("\n── Part A: Anonymization Quality (pre-computed masked forms) ──")
    if COMBO == 2:
        preds_A = run_filler_combo2(masked_texts, filler, filler_tok,
                                    batch_size=args.batch, num_beams=args.beams)
    else:
        preds_A = run_filler_combo1(masked_texts, filler, filler_tok,
                                    batch_size=args.batch)

    elr_A    = entity_leakage_rate(preds_A, entity_lists)
    bleu4_A  = round(sum(sentence_bleu(p, r) for p, r in zip(preds_A, gold_anon))
                     / max(len(preds_A), 1) * 100, 2)
    exact_A  = round(sum(p.strip() == r.strip() for p, r in zip(preds_A, gold_anon))
                     / max(len(preds_A), 1), 4)
    print(f"  ELR       : {elr_A:.4f}  (lower = better privacy)")
    print(f"  BLEU-4    : {bleu4_A:.2f}  (higher = better utility)")
    print(f"  Exact     : {exact_A:.4f}")

    # ── Part B: End-to-end pipeline robustness ───────────────────────────────
    print("\n── Part B: End-to-End Adversarial Robustness ──")
    print("  Step 1: NER masker on originals …")
    masked_live = run_ner_mask(originals, ner_model, ner_tok)

    print("  Step 2: Hardened filler …")
    if COMBO == 2:
        anon_B = run_filler_combo2(masked_live, filler, filler_tok,
                                    batch_size=args.batch, num_beams=args.beams)
    else:
        anon_B = run_filler_combo1(masked_live, filler, filler_tok,
                                    batch_size=args.batch)

    print("  Step 3: Frozen inverter on hardened output …")
    inv_preds = run_inverter(anon_B, inverter, inverter_tok,
                             batch_size=args.batch, num_beams=args.beams)

    err_B   = entity_recovery_rate(inv_preds, originals)
    elr_B   = entity_leakage_rate(anon_B, entity_lists)
    bleu4_B = round(sum(sentence_bleu(p, r) for p, r in zip(anon_B, gold_anon))
                    / max(len(anon_B), 1) * 100, 2)

    print(f"  ERR (inverter recovery) : {err_B:.4f}  ← PRIMARY attack metric (lower = better)")
    print(f"  ELR (filler leakage)    : {elr_B:.4f}")
    print(f"  BLEU-4 vs gold          : {bleu4_B:.2f}")

    # ── Sample predictions ─────────────────────────────────────────────────
    print("\n── Sample Predictions (10 random) ──")
    import random
    indices = random.sample(range(len(originals)), min(10, len(originals)))
    for k, i in enumerate(indices, 1):
        print(f"\n  [{k:02d}] ORIGINAL  : {originals[i][:80]}")
        print(f"       MASKED    : {masked_live[i][:80]}")
        print(f"       HARDENED  : {anon_B[i][:80]}")
        print(f"       INV PRED  : {inv_preds[i][:80]}")

    # ── Save full report ────────────────────────────────────────────────────
    report = {
        "combo":     COMBO,
        "n_samples": len(eval_data),
        "timestamp": datetime.now().isoformat(),
        "part_A_quality": {
            "ELR":    elr_A,
            "BLEU4":  bleu4_A,
            "ExactMatch": exact_A,
        },
        "part_B_robustness": {
            "ERR_inverter": err_B,
            "ELR_filler":   elr_B,
            "BLEU4_vs_gold": bleu4_B,
        },
        "baselines_from_paper": {
            "ERR_pre_hardening_c1": 0.758,
            "ERR_pre_hardening_c2": 0.768,
        },
    }
    out_json = RESULTS_DIR / f"eval_report_combo{COMBO}.json"
    out_txt  = RESULTS_DIR / f"eval_report_combo{COMBO}.txt"

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "=" * 65,
        f"  ADVERSARIAL HARDENING EVALUATION — Combo {COMBO}",
        f"  Filler   : {FILLER_ID}",
        f"  Inverter : {INVERTER_ID}",
        f"  Samples  : {len(eval_data):,}",
        "=" * 65,
        "",
        "  PART A — Anonymization Quality",
        f"    ELR          : {elr_A:.4f}  (pre-hardening baseline ELR ≈ 1.33%–2.50%)",
        f"    BLEU-4        : {bleu4_A:.2f}",
        f"    Exact Match   : {exact_A:.4f}",
        "",
        "  PART B — Adversarial Robustness (end-to-end pipeline)",
        f"    ERR (inverter): {err_B:.4f}  ← target: < 0.77 (pre-hardening baseline)",
        f"      Combo 1 pre-hardening ERR: 75.80%",
        f"      Combo 2 pre-hardening ERR: 76.83%",
        f"    ELR (filler)  : {elr_B:.4f}",
        f"    BLEU-4 vs gold: {bleu4_B:.2f}",
        "=" * 65,
    ]
    report_txt = "\n".join(lines)
    print("\n" + report_txt)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")

    print(f"\nReport saved → {out_json}")
    print(f"Report saved → {out_txt}")


if __name__ == "__main__":
    main()
