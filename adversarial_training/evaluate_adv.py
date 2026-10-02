#!/usr/bin/env python3
"""
Adversarial Training — Evaluation
===================================
Evaluates the adversarially hardened BART-base model on two axes:

  A. Normal anonymization quality (val set from Seq2Seq_model/data_splits/val.jsonl)
       - Exact sentence match
       - Token-level accuracy
       - Word-level accuracy
       - BLEU (1, 2, 4)
       - ROUGE (1, 2, L)
       - Entity Leakage Rate (ELR) — fraction of outputs that still contain
         any original PII entity strings

  B. Adversarial robustness (inverter eval set from model_inversion/output/inverter_eval.jsonl)
       - Run hardened BART → generate new anonymized text
       - Feed generated text to the FROZEN original inverter
       - Compute Entity Recovery Rate (ERR) — did the inverter recover the PII?
       - Compare against the baseline ERR (~30%) from before hardening
       - A lower post-training ERR means the model is harder to invert

Run:
    cd adversarial_training
    python3 evaluate_adv.py

Writes results to adversarial_training/results/eval_report.json
                  adversarial_training/results/eval_report.txt
"""

import os
import sys
import gc
import json
import math
import logging
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    VICTIM_MODEL_NAME, INVERTER_MODEL_NAME, INVERTER_CHECKPOINT,
    ADV_CHECKPOINT_DIR,
    NORMAL_VAL, ADV_EVAL_FILE,
    LOGS_DIR, RESULTS_DIR,
    MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, NUM_WORKERS,
)
from dataset import NormalDataset, AdvDataset, adv_collate_fn, load_jsonl

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# Prefer the fully-trained final model; fall back to best-val checkpoint
_final = os.path.join(ADV_CHECKPOINT_DIR, "final_model.pt")
_best  = os.path.join(ADV_CHECKPOINT_DIR, "best_model.pt")
ADV_BEST_MODEL    = _final if os.path.exists(_final) else _best
RESULTS_JSON      = os.path.join(RESULTS_DIR, "eval_report.json")
RESULTS_TXT       = os.path.join(RESULTS_DIR, "eval_report.txt")
EVAL_BATCH_SIZE   = 4
ADV_EVAL_BATCH    = 4
BEAMS             = 4    # beam-search for quality metrics

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "evaluate_adv.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BLEU (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    from collections import Counter
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp:
        return 0.0
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
    scores = []
    for n in range(1, max_n + 1):
        h_ng = _ngrams(hyp, n)
        r_ng = _ngrams(ref, n)
        if not h_ng:
            scores.append(0.0)
            continue
        matches = sum((h_ng & r_ng).values())
        precision = matches / sum(h_ng.values())
        scores.append(precision if precision > 0 else 0.0)
    if any(s == 0 for s in scores):
        return 0.0
    return bp * math.exp(sum(math.log(s) for s in scores) / len(scores))


def corpus_bleu(hypotheses, references) -> float:
    return (
        sum(sentence_bleu(h, r) for h, r in zip(hypotheses, references))
        / max(len(hypotheses), 1)
    )


def bleu_n(hypotheses, references, n: int) -> float:
    """BLEU up to n-gram order n."""
    return (
        sum(sentence_bleu(h, r, max_n=n) for h, r in zip(hypotheses, references))
        / max(len(hypotheses), 1)
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE (requires rouge_score; graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(preds, refs) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        import numpy as np
        return {
            "rouge1": round(float(np.mean(r1)) * 100, 2),
            "rouge2": round(float(np.mean(r2)) * 100, 2),
            "rougeL": round(float(np.mean(rl)) * 100, 2),
        }
    except ImportError:
        logger.warning("rouge_score not installed. Skipping ROUGE.")
        return {"rouge1": None, "rouge2": None, "rougeL": None}


# ─────────────────────────────────────────────────────────────────────────────
# Entity Leakage Rate
# ─────────────────────────────────────────────────────────────────────────────

def entity_leakage_rate(predictions: list, entity_texts_list: list) -> float:
    """
    Fraction of predictions that contain at least one original PII entity string.
    A good anonymizer should score 0.0 here.
    """
    leaked = 0
    total  = 0
    for pred, entities in zip(predictions, entity_texts_list):
        if not entities:
            continue
        total += 1
        pred_lower = pred.lower()
        if any(e.lower() in pred_lower for e in entities if e.strip()):
            leaked += 1
    return round(leaked / max(total, 1), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Entity Recovery Rate (ERR)  — adversarial metric
# ─────────────────────────────────────────────────────────────────────────────

def entity_recovery_rate(
    predictions: list, originals: list, probe_entities: list
) -> dict:
    """
    Exact and partial ERR: fraction of predictions from the INVERTER that contain
    the probe entity from the original text.

    A LOWER post-hardening ERR means the adversarial training worked.
    """
    exact = partial = total = 0
    for pred, orig, entity in zip(predictions, originals, probe_entities):
        if not entity:
            continue
        total += 1
        if entity in pred:
            exact += 1
        elif any(part in pred for part in entity.split() if len(part) > 2):
            partial += 1
    if total == 0:
        return {"exact": 0.0, "partial": 0.0, "total": 0}
    return {
        "exact":   round(exact   / total, 4),
        "partial": round(partial / total, 4),
        "total":   total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hardened_model(device):
    logger.info("Loading adversarially hardened victim from: %s", ADV_BEST_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(VICTIM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        VICTIM_MODEL_NAME, torch_dtype=torch.float32
    )
    if not os.path.exists(ADV_BEST_MODEL):
        raise FileNotFoundError(
            f"Adversarial checkpoint not found: {ADV_BEST_MODEL}\n"
            "Run train_adv.py first."
        )
    ckpt  = torch.load(ADV_BEST_MODEL, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    logger.info("  Hardened model loaded (step=%d)", ckpt.get("global_step", -1))
    return model, tokenizer


def load_frozen_inverter(device, tokenizer):
    logger.info("Loading frozen inverter from: %s", INVERTER_CHECKPOINT)
    inverter = AutoModelForSeq2SeqLM.from_pretrained(
        INVERTER_MODEL_NAME, torch_dtype=torch.float32
    )
    if os.path.exists(INVERTER_CHECKPOINT):
        ckpt  = torch.load(INVERTER_CHECKPOINT, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        inverter.load_state_dict(state, strict=False)
    else:
        logger.warning("Inverter checkpoint not found — using pretrained weights.")
    for param in inverter.parameters():
        param.requires_grad_(False)
    inverter = inverter.to(device)
    inverter.eval()
    return inverter


# ─────────────────────────────────────────────────────────────────────────────
# A: Normal anonymization quality
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_normal(model, tokenizer, device) -> dict:
    logger.info("=" * 60)
    logger.info("PART A: Normal anonymization quality (val set)")
    logger.info("=" * 60)

    val_data = load_jsonl(ADV_EVAL_FILE)
    # remap inverter_eval.jsonl fields to NormalDataset schema
    for row in val_data:
        row["original_text"]  = row["original"]
        row["anonymized_text"] = row["anonymized"]
        row["entity_texts"]   = [row["probe_entity"]] if row.get("probe_entity") else []
    val_ds   = NormalDataset(val_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    def _normal_collate(batch):
        return {
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels":         torch.stack([b["labels"]         for b in batch]),
            "entity_texts":   [b["entity_texts"]  for b in batch],
            "original_text":  [b["original_text"] for b in batch],
        }

    val_dl   = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          collate_fn=_normal_collate)

    all_preds, all_refs   = [], []
    all_originals         = []
    all_entity_texts      = []

    for batch in tqdm(val_dl, desc="Normal eval"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_TARGET_LENGTH,
            num_beams=BEAMS,
            early_stopping=True,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        label_ids = batch["labels"].clone()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        refs  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        all_preds.extend(preds)
        all_refs.extend(refs)
        all_originals.extend(batch["original_text"])
        all_entity_texts.extend(batch["entity_texts"])

        del input_ids, attention_mask, gen_ids

    # Metrics
    def _tok_acc(p_list, r_list):
        c = t = 0
        for p, r in zip(p_list, r_list):
            pt = tokenizer.tokenize(p); rt = tokenizer.tokenize(r)
            ml = min(len(pt), len(rt))
            if ml == 0: continue
            c += sum(a == b for a, b in zip(pt[:ml], rt[:ml]))
            t += max(len(pt), len(rt))
        return round(c / t, 4) if t else 0.0

    def _word_acc(p_list, r_list):
        c = t = 0
        for p, r in zip(p_list, r_list):
            pw = p.lower().split(); rw = r.lower().split()
            ml = min(len(pw), len(rw))
            if ml == 0: continue
            c += sum(a == b for a, b in zip(pw[:ml], rw[:ml]))
            t += max(len(pw), len(rw))
        return round(c / t, 4) if t else 0.0

    def _exact_match(p_list, r_list):
        return round(sum(p.strip() == r.strip() for p, r in zip(p_list, r_list)) / max(len(p_list), 1), 4)

    elr   = entity_leakage_rate(all_preds, all_entity_texts)
    rouge = compute_rouge(all_preds, all_refs)

    results = {
        "n_samples":         len(all_preds),
        "exact_match":       _exact_match(all_preds, all_refs),
        "token_accuracy":    _tok_acc(all_preds, all_refs),
        "word_accuracy":     _word_acc(all_preds, all_refs),
        "bleu1":             round(bleu_n(all_preds, all_refs, 1), 4),
        "bleu2":             round(bleu_n(all_preds, all_refs, 2), 4),
        "bleu4":             round(corpus_bleu(all_preds, all_refs), 4),
        "rouge1":            rouge["rouge1"],
        "rouge2":            rouge["rouge2"],
        "rougeL":            rouge["rougeL"],
        "entity_leakage_rate": elr,
        "sample_predictions": [
            {"original": o, "prediction": p, "reference": r}
            for o, p, r in zip(all_originals[:10], all_preds[:10], all_refs[:10])
        ],
    }

    logger.info("  Exact match   : %.4f", results["exact_match"])
    logger.info("  Token acc     : %.4f", results["token_accuracy"])
    logger.info("  Word acc      : %.4f", results["word_accuracy"])
    logger.info("  BLEU-4        : %.4f", results["bleu4"])
    logger.info("  ROUGE-L       : %s",   results["rougeL"])
    logger.info("  Entity leakage: %.4f  (lower = better)", results["entity_leakage_rate"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# B: Adversarial robustness — post-hardening ERR
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_adversarial(hardened_model, inverter, tokenizer, device) -> dict:
    """
    Pipeline:
      original_text ──► Hardened BART ──► new_anonymized_text (generated)
      new_anonymized_text ──► Frozen Inverter ──► inverter_prediction
      Measure ERR: did the inverter recover the probe entity?

    A lower ERR than the pre-hardening baseline (~30%) means success.
    """
    logger.info("=" * 60)
    logger.info("PART B: Adversarial robustness (inverter eval set)")
    logger.info("=" * 60)

    adv_data = load_jsonl(ADV_EVAL_FILE)
    # Build a simple dataset for generation: we only need original text
    class SimpleGenDataset(torch.utils.data.Dataset):
        def __init__(self, data, tok, max_len):
            self.data = data
            self.tok  = tok
            self.max_len = max_len
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            d   = self.data[i]
            enc = self.tok(d["original"], max_length=self.max_len,
                           padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids":       enc["input_ids"].squeeze(0),
                "attention_mask":  enc["attention_mask"].squeeze(0),
                "original":        d["original"],
                "probe_entity":    d.get("probe_entity", ""),
                "strategy":        d.get("strategy", ""),
                "name_rarity":     d.get("name_rarity", ""),
                "entity_type":     d.get("entity_type", ""),
            }

    def _collate(batch):
        return {
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "original":       [b["original"]     for b in batch],
            "probe_entity":   [b["probe_entity"] for b in batch],
            "strategy":       [b["strategy"]     for b in batch],
            "name_rarity":    [b["name_rarity"]  for b in batch],
            "entity_type":    [b["entity_type"]  for b in batch],
        }

    gen_ds = SimpleGenDataset(adv_data, tokenizer, MAX_INPUT_LENGTH)
    gen_dl = DataLoader(gen_ds, batch_size=ADV_EVAL_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=_collate)

    all_originals, all_probe_entities = [], []
    all_strategies, all_rarities      = [], []
    all_new_anonymized                = []
    all_inverter_preds                = []

    for batch in tqdm(gen_dl, desc="Adversarial eval (hardened → inverter)"):
        orig_ids      = batch["input_ids"].to(device)
        orig_mask     = batch["attention_mask"].to(device)

        # Step 1: Hardened BART anonymizes the original
        new_anon_ids = hardened_model.generate(
            input_ids=orig_ids,
            attention_mask=orig_mask,
            max_length=MAX_TARGET_LENGTH,
            num_beams=BEAMS,
            early_stopping=True,
        )
        new_anon_texts = tokenizer.batch_decode(new_anon_ids, skip_special_tokens=True)

        # Step 2: Tokenize the NEW anonymized text as input for the inverter
        inv_enc = tokenizer(
            new_anon_texts,
            max_length=MAX_INPUT_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Step 3: Frozen inverter tries to recover original
        inv_out = inverter.generate(
            input_ids=inv_enc["input_ids"],
            attention_mask=inv_enc["attention_mask"],
            max_length=MAX_TARGET_LENGTH,
            num_beams=BEAMS,
            early_stopping=True,
        )
        inv_preds = tokenizer.batch_decode(inv_out, skip_special_tokens=True)

        all_originals.extend(batch["original"])
        all_probe_entities.extend(batch["probe_entity"])
        all_strategies.extend(batch["strategy"])
        all_rarities.extend(batch["name_rarity"])
        all_new_anonymized.extend(new_anon_texts)
        all_inverter_preds.extend(inv_preds)

        del orig_ids, orig_mask, new_anon_ids, inv_enc, inv_out

    # Overall ERR
    err_overall = entity_recovery_rate(
        all_inverter_preds, all_originals, all_probe_entities
    )

    # ERR per strategy
    err_by_strategy = {}
    strat_groups = defaultdict(lambda: {"preds": [], "origs": [], "ents": []})
    for pred, orig, ent, strat in zip(
        all_inverter_preds, all_originals, all_probe_entities, all_strategies
    ):
        strat_groups[strat]["preds"].append(pred)
        strat_groups[strat]["origs"].append(orig)
        strat_groups[strat]["ents"].append(ent)
    for strat, g in strat_groups.items():
        err_by_strategy[strat] = entity_recovery_rate(g["preds"], g["origs"], g["ents"])

    # ERR per rarity
    err_by_rarity = {}
    rar_groups = defaultdict(lambda: {"preds": [], "origs": [], "ents": []})
    for pred, orig, ent, rar in zip(
        all_inverter_preds, all_originals, all_probe_entities, all_rarities
    ):
        rar_groups[rar or "unknown"]["preds"].append(pred)
        rar_groups[rar or "unknown"]["origs"].append(orig)
        rar_groups[rar or "unknown"]["ents"].append(ent)
    for rar, g in rar_groups.items():
        err_by_rarity[rar] = entity_recovery_rate(g["preds"], g["origs"], g["ents"])

    results = {
        "n_samples":          len(all_originals),
        "err_exact":          err_overall["exact"],
        "err_partial":        err_overall["partial"],
        "err_by_strategy":    err_by_strategy,
        "err_by_rarity":      err_by_rarity,
        "sample_pairs": [
            {
                "original":       o,
                "new_anonymized": a,
                "inverter_pred":  p,
                "probe_entity":   e,
            }
            for o, a, p, e in zip(
                all_originals[:10], all_new_anonymized[:10],
                all_inverter_preds[:10], all_probe_entities[:10],
            )
        ],
    }

    logger.info("  Post-hardening ERR (exact)  : %.4f  (pre-hardening baseline: ~0.30)", err_overall["exact"])
    logger.info("  Post-hardening ERR (partial): %.4f", err_overall["partial"])
    logger.info("  ERR reduction (exact)       : %.4f (%+.1f%%)",
                0.30 - err_overall["exact"],
                (0.30 - err_overall["exact"]) / 0.30 * 100)
    logger.info("  Per-strategy breakdown:")
    for s, e in sorted(err_by_strategy.items()):
        logger.info("    %-35s exact=%.4f partial=%.4f (n=%d)", s, e["exact"], e["partial"], e["total"])
    logger.info("  Per-rarity breakdown:")
    for r, e in sorted(err_by_rarity.items()):
        logger.info("    %-12s exact=%.4f partial=%.4f (n=%d)", r, e["exact"], e["partial"], e["total"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(normal_results: dict, adv_results: dict):
    report = {
        "generated_at":    datetime.now().isoformat(),
        "normal_val":      normal_results,
        "adversarial_val": adv_results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("JSON report saved → %s", RESULTS_JSON)

    lines = []
    lines.append("=" * 70)
    lines.append("ADVERSARIAL TRAINING — EVALUATION REPORT")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("=" * 70)

    lines.append("\n── A. Normal Anonymization Quality (val set) ──────────────────────")
    nr = normal_results
    lines.append(f"  Samples         : {nr['n_samples']}")
    lines.append(f"  Exact Match     : {nr['exact_match']:.4f}")
    lines.append(f"  Token Accuracy  : {nr['token_accuracy']:.4f}")
    lines.append(f"  Word Accuracy   : {nr['word_accuracy']:.4f}")
    lines.append(f"  BLEU-1          : {nr['bleu1']:.4f}")
    lines.append(f"  BLEU-2          : {nr['bleu2']:.4f}")
    lines.append(f"  BLEU-4          : {nr['bleu4']:.4f}")
    lines.append(f"  ROUGE-1         : {nr['rouge1']}")
    lines.append(f"  ROUGE-2         : {nr['rouge2']}")
    lines.append(f"  ROUGE-L         : {nr['rougeL']}")
    lines.append(f"  Entity Leakage  : {nr['entity_leakage_rate']:.4f}  (lower = better)")

    lines.append("\n  Sample predictions:")
    for s in nr["sample_predictions"][:5]:
        lines.append(f"    ORIG  : {s['original'][:90]}")
        lines.append(f"    PRED  : {s['prediction'][:90]}")
        lines.append(f"    REF   : {s['reference'][:90]}")
        lines.append("")

    lines.append("\n── B. Adversarial Robustness (inverter eval set) ──────────────────")
    ar = adv_results
    lines.append(f"  Samples                     : {ar['n_samples']}")
    lines.append(f"  Post-hardening ERR (exact)  : {ar['err_exact']:.4f}")
    lines.append(f"  Post-hardening ERR (partial): {ar['err_partial']:.4f}")
    lines.append(f"  Pre-hardening baseline      : ~0.3000  (30% from attack)")
    reduction = 0.30 - ar['err_exact']
    lines.append(f"  ERR reduction (exact)       : {reduction:+.4f}  ({reduction/0.30*100:+.1f}%)")

    lines.append("\n  Per-strategy ERR (exact):")
    for s, e in sorted(ar["err_by_strategy"].items()):
        lines.append(f"    {s:<40} {e['exact']:.4f}  (n={e['total']})")

    lines.append("\n  Per-rarity ERR (exact):")
    for r, e in sorted(ar["err_by_rarity"].items()):
        lines.append(f"    {r:<15} {e['exact']:.4f}  (n={e['total']})")

    lines.append("\n  Sample inversion attempts (original → new_anon → inverter_pred):")
    for s in ar["sample_pairs"][:5]:
        lines.append(f"    ORIG    : {s['original'][:80]}")
        lines.append(f"    NEW_ANON: {s['new_anonymized'][:80]}")
        lines.append(f"    INV_PRED: {s['inverter_pred'][:80]}")
        lines.append(f"    ENTITY  : {s['probe_entity']}")
        lines.append("")

    lines.append("=" * 70)

    with open(RESULTS_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Text report saved → %s", RESULTS_TXT)

    # Print to console
    print("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation device: %s", device)

    hardened_model, tokenizer = load_hardened_model(device)
    inverter                  = load_frozen_inverter(device, tokenizer)

    normal_results = eval_normal(hardened_model, tokenizer, device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adv_results = eval_adversarial(hardened_model, inverter, tokenizer, device)

    write_report(normal_results, adv_results)


if __name__ == "__main__":
    main()
