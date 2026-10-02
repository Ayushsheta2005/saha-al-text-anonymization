#!/usr/bin/env python3
"""
Step 4 — Attack Evaluator
==========================
Evaluates TWO attack strategies on the held-out eval set:

  Attack A — Lookup Table Attack (zero-training baseline)
    Uses the consistency_report.json lookup table to directly
    reverse BART replacements. If BART is deterministic, this
    alone should achieve high ERR.

  Attack B — Inverter Model Attack (learned)
    Uses the trained BART-base inverter from train_inverter.py.
    Evaluated on the same eval set as Attack A for fair comparison.

Metrics reported:
  • Entity Recovery Rate (ERR) — exact and partial
  • Token-level accuracy
  • BLEU score (sentence and corpus)
  • Per-strategy breakdown
  • Per-rarity breakdown
  • Per-PII-type breakdown
  • Attack confusion matrix (what did the model recover vs miss)

Run:
    python3 evaluate_attack.py

Output: output/attack_results.json + output/attack_report.txt
"""

import os
import sys
import json
import math
import time
import logging
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INVERTER_MODEL_NAME, INVERTER_CHECKPOINT_DIR,
    INVERTER_MAX_INPUT, INVERTER_MAX_TARGET, INVERTER_EVAL_BATCH,
    BART_PAIRS_FILE, CONSISTENCY_REPORT_FILE,
    ATTACK_RESULTS_FILE, ATTACK_REPORT_FILE,
    OUTPUT_DIR, LOGS_DIR,
)

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "evaluate_attack.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BLEU (no external deps — simple n-gram implementation)
# ═══════════════════════════════════════════════════════════════════════════

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp:
        return 0.0
    # Brevity penalty
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref)/len(hyp))
    scores = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)
        if not hyp_ng:
            scores.append(0.0)
            continue
        matches = sum((hyp_ng & ref_ng).values())
        precision = matches / sum(hyp_ng.values())
        scores.append(precision if precision > 0 else 0.0)
    if any(s == 0 for s in scores):
        return 0.0
    log_avg = sum(math.log(s) for s in scores) / len(scores)
    return bp * math.exp(log_avg)

def corpus_bleu(hypotheses: List[str], references: List[str]) -> float:
    return sum(sentence_bleu(h, r) for h, r in zip(hypotheses, references)) / max(len(hypotheses), 1)


# ═══════════════════════════════════════════════════════════════════════════
# METRICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MetricsEngine:
    """Unified metrics computation for both attack types."""

    @staticmethod
    def entity_recovery_rate(
        predictions: List[str],
        originals: List[str],
        probe_entities: List[str],
    ) -> Dict:
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

    @staticmethod
    def token_accuracy(predictions: List[str], references: List[str], tokenizer) -> float:
        correct = total = 0
        for pred, ref in zip(predictions, references):
            p = tokenizer.tokenize(pred)
            r = tokenizer.tokenize(ref)
            ml = min(len(p), len(r))
            if ml == 0:
                continue
            correct += sum(a == b for a, b in zip(p[:ml], r[:ml]))
            total   += max(len(p), len(r))
        return round(correct / total, 4) if total > 0 else 0.0

    @staticmethod
    def exact_sentence_match(predictions: List[str], references: List[str]) -> float:
        if not predictions:
            return 0.0
        matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        return round(matches / len(predictions), 4)

    @staticmethod
    def bleu(predictions: List[str], references: List[str]) -> float:
        return round(corpus_bleu(predictions, references), 4)

    @classmethod
    def full_eval(
        cls,
        predictions: List[str],
        references: List[str],
        probe_entities: List[str],
        tokenizer,
    ) -> Dict:
        err = cls.entity_recovery_rate(predictions, references, probe_entities)
        return {
            "err_exact":        err["exact"],
            "err_partial":      err["partial"],
            "err_total_probed": err["total"],
            "token_accuracy":   cls.token_accuracy(predictions, references, tokenizer),
            "exact_sentence_match": cls.exact_sentence_match(predictions, references),
            "corpus_bleu":      cls.bleu(predictions, references),
        }

    @classmethod
    def breakdown(
        cls,
        predictions: List[str],
        references: List[str],
        probe_entities: List[str],
        group_keys: List[str],
    ) -> Dict:
        """Group predictions by group_keys and compute ERR per group."""
        groups: Dict[str, Dict] = defaultdict(lambda: {"preds": [], "refs": [], "ents": []})
        for pred, ref, ent, key in zip(predictions, references, probe_entities, group_keys):
            key = key or "unknown"
            groups[key]["preds"].append(pred)
            groups[key]["refs"].append(ref)
            groups[key]["ents"].append(ent)
        result = {}
        for key, g in sorted(groups.items()):
            err = cls.entity_recovery_rate(g["preds"], g["refs"], g["ents"])
            result[key] = {
                "n":            len(g["preds"]),
                "err_exact":    err["exact"],
                "err_partial":  err["partial"],
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════
# ATTACK A — LOOKUP TABLE
# ═══════════════════════════════════════════════════════════════════════════

class LookupTableAttack:
    """
    Zero-training attack.
    Uses the entity → replacement mapping from consistency_report.json
    to build an inverse table: replacement → original_entity.
    For each anonymized sentence, attempts to substitute back.
    """

    def __init__(self, consistency_report: Dict):
        self.lookup = consistency_report.get("lookup_table", {})
        # Build inverse: replacement → list of possible originals
        self.inverse: Dict[str, List[str]] = defaultdict(list)
        for original, replacement in self.lookup.items():
            if replacement not in ("[LEAKED]", "[DELETED]", "[REMOVED]"):
                self.inverse[replacement].append(original)

        logger.info(f"  Lookup table: {len(self.lookup):,} forward entries")
        logger.info(f"  Inverse table: {len(self.inverse):,} reverse entries")

    def predict(self, anonymized: str, probe_entity: str) -> str:
        """
        Attempt to reverse the anonymized text.
        Strategy:
          1. Find the most likely replacement for probe_entity in the anonymized text
          2. Substitute it back
          3. If no match found, return anonymized text unchanged
        """
        if not self.lookup or not self.inverse:
            return anonymized

        result = anonymized
        # Try every known replacement in the anonymized text
        for replacement, originals in self.inverse.items():
            if replacement in result and originals:
                # Pick the best matching original (prefer exact if probe_entity matches)
                best = originals[0]
                for orig in originals:
                    if probe_entity and probe_entity.lower() in orig.lower():
                        best = orig
                        break
                result = result.replace(replacement, best, 1)
        return result

    def run(self, eval_pairs: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        predictions  = []
        references   = []
        probe_entities = []

        for pair in tqdm(eval_pairs, desc="  Lookup attack", dynamic_ncols=True):
            anon   = pair.get("anonymized", "")
            orig   = pair.get("original",   "")
            entity = pair.get("probe_entity", "")

            pred = self.predict(anon, entity)
            predictions.append(pred)
            references.append(orig)
            probe_entities.append(entity)

        return predictions, references, probe_entities


# ═══════════════════════════════════════════════════════════════════════════
# ATTACK B — INVERTER MODEL
# ═══════════════════════════════════════════════════════════════════════════

class InverterModelAttack:
    """Learned inversion using the trained BART-base inverter."""

    def __init__(self, device: torch.device, tokenizer: AutoTokenizer,
                 model: AutoModelForSeq2SeqLM):
        self.device    = device
        self.tokenizer = tokenizer
        self.model     = model
        self.model.eval()

    @torch.no_grad()
    def run(self, eval_pairs: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        predictions    = []
        references     = []
        probe_entities = []

        # Batch the eval pairs
        for i in tqdm(range(0, len(eval_pairs), INVERTER_EVAL_BATCH),
                      desc="  Inverter attack", dynamic_ncols=True):
            batch = eval_pairs[i:i + INVERTER_EVAL_BATCH]
            texts = [p.get("anonymized", "").strip() for p in batch]

            enc = self.tokenizer(
                texts,
                max_length=INVERTER_MAX_INPUT,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=INVERTER_MAX_TARGET,
                num_beams=4,
                early_stopping=True,
            )
            preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for pred, pair in zip(preds, batch):
                predictions.append(pred)
                references.append(pair.get("original", ""))
                probe_entities.append(pair.get("probe_entity", ""))

        return predictions, references, probe_entities


# ═══════════════════════════════════════════════════════════════════════════
# REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════════

def write_text_report(results: Dict, output_path: str):
    lines = []
    lines.append("=" * 72)
    lines.append("  MODEL INVERSION ATTACK — EVALUATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    lines.append("")

    for attack_name in ["attack_a_lookup", "attack_b_inverter"]:
        if attack_name not in results:
            continue
        r = results[attack_name]
        label = "Attack A — Lookup Table (zero training)" \
                if "lookup" in attack_name else "Attack B — Inverter Model (BART-base)"
        lines.append(f"{'─'*72}")
        lines.append(f"  {label}")
        lines.append(f"{'─'*72}")

        m = r.get("overall_metrics", {})
        lines.append(f"  Eval samples:          {r.get('n_eval', 0):,}")
        lines.append(f"  ERR (exact):           {m.get('err_exact', 0):.4f}   "
                     f"({m.get('err_exact', 0)*100:.1f}% of probe entities recovered verbatim)")
        lines.append(f"  ERR (partial):         {m.get('err_partial', 0):.4f}   "
                     f"({m.get('err_partial', 0)*100:.1f}% partial recovery)")
        lines.append(f"  Token accuracy:        {m.get('token_accuracy', 0):.4f}")
        lines.append(f"  Exact sentence match:  {m.get('exact_sentence_match', 0):.4f}")
        lines.append(f"  Corpus BLEU:           {m.get('corpus_bleu', 0):.4f}")
        lines.append("")

        # Per-strategy
        if "by_strategy" in r:
            lines.append("  By strategy:")
            for strat, sm in sorted(r["by_strategy"].items()):
                lines.append(f"    {strat:<38} n={sm['n']:<5}  "
                             f"ERR_exact={sm['err_exact']:.3f}  "
                             f"ERR_partial={sm['err_partial']:.3f}")
            lines.append("")

        # Per-rarity
        if "by_rarity" in r:
            lines.append("  By name rarity:")
            order = ["common", "medium", "rare", "very_rare", "unknown"]
            for rar in order:
                if rar not in r["by_rarity"]:
                    continue
                rm = r["by_rarity"][rar]
                lines.append(f"    {rar:<15} n={rm['n']:<5}  "
                             f"ERR_exact={rm['err_exact']:.3f}  "
                             f"ERR_partial={rm['err_partial']:.3f}")
            lines.append("")

        # Per entity type
        if "by_entity_type" in r:
            lines.append("  By entity type:")
            for etype, em in sorted(r["by_entity_type"].items()):
                lines.append(f"    {etype:<15} n={em['n']:<5}  "
                             f"ERR_exact={em['err_exact']:.3f}  "
                             f"ERR_partial={em['err_partial']:.3f}")
            lines.append("")

        # Sample predictions
        lines.append("  Sample recoveries (original → anonymized → inverted):")
        for s in r.get("samples", [])[:8]:
            lines.append(f"    ORIGINAL  : {s['reference'][:80]}")
            lines.append(f"    ANONYMIZED: {s['anonymized'][:80]}")
            lines.append(f"    INVERTED  : {s['predicted'][:80]}")
            lines.append(f"    ENTITY    : {s['probe_entity']}  "
                         f"(recovered={'YES' if s['entity_recovered'] else 'NO'})")
            lines.append("")

    # Comparison
    lines.append("=" * 72)
    lines.append("  ATTACK COMPARISON SUMMARY")
    lines.append("=" * 72)
    if "attack_a_lookup" in results and "attack_b_inverter" in results:
        a = results["attack_a_lookup"]["overall_metrics"]
        b = results["attack_b_inverter"]["overall_metrics"]
        lines.append(f"  {'Metric':<30} {'Lookup':>10} {'Inverter':>10} {'Delta':>10}")
        lines.append(f"  {'─'*62}")
        for metric in ["err_exact", "err_partial", "token_accuracy", "corpus_bleu"]:
            va = a.get(metric, 0)
            vb = b.get(metric, 0)
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {metric:<30} {va:>10.4f} {vb:>10.4f} {sign}{delta:>9.4f}")
    lines.append("")
    lines.append("=" * 72)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Text report saved: {output_path}")

    # Also print to stdout
    print("\n" + "\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  MODEL INVERSION — STEP 4: ATTACK EVALUATION")
    print("=" * 70)

    # ── Load eval pairs ────────────────────────────────────────────────────
    if not os.path.exists(BART_PAIRS_FILE):
        print(f"  ❌  Pairs file not found: {BART_PAIRS_FILE}")
        print("  Run: python3 query_bart.py first.")
        return

    all_pairs = []
    with open(BART_PAIRS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(json.loads(line))

    eval_pairs = [p for p in all_pairs if p.get("split") == "eval"]
    logger.info(f"  Total pairs: {len(all_pairs):,}")
    logger.info(f"  Eval pairs:  {len(eval_pairs):,}")

    if not eval_pairs:
        print("  ❌  No eval pairs found. Check split=eval in pairs file.")
        return

    # ── Load tokenizer (shared) ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(INVERTER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"generated_at": datetime.now().isoformat(), "n_eval": len(eval_pairs)}

    # ══════════════════════════════════════════════════════════════════════
    # ATTACK A — LOOKUP TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n  ── Running Attack A: Lookup Table ──")
    if not os.path.exists(CONSISTENCY_REPORT_FILE):
        logger.warning("  Consistency report not found — skipping Attack A")
    else:
        with open(CONSISTENCY_REPORT_FILE) as f:
            consistency_report = json.load(f)

        attack_a = LookupTableAttack(consistency_report)
        preds_a, refs_a, ents_a = attack_a.run(eval_pairs)

        metrics_a = MetricsEngine.full_eval(preds_a, refs_a, ents_a, tokenizer)
        by_strategy_a = MetricsEngine.breakdown(
            preds_a, refs_a, ents_a,
            [p.get("strategy", "unknown") for p in eval_pairs]
        )
        by_rarity_a = MetricsEngine.breakdown(
            preds_a, refs_a, ents_a,
            [p.get("name_rarity", "unknown") for p in eval_pairs]
        )
        by_etype_a = MetricsEngine.breakdown(
            preds_a, refs_a, ents_a,
            [p.get("entity_type", "unknown") for p in eval_pairs]
        )

        # Build samples
        samples_a = []
        for pred, ref, ent, pair in zip(preds_a[:20], refs_a[:20], ents_a[:20], eval_pairs[:20]):
            samples_a.append({
                "anonymized":       pair.get("anonymized", ""),
                "predicted":        pred,
                "reference":        ref,
                "probe_entity":     ent,
                "entity_recovered": ent in pred if ent else False,
                "strategy":         pair.get("strategy"),
            })

        results["attack_a_lookup"] = {
            "n_eval":          len(eval_pairs),
            "overall_metrics": metrics_a,
            "by_strategy":     by_strategy_a,
            "by_rarity":       by_rarity_a,
            "by_entity_type":  by_etype_a,
            "samples":         samples_a,
        }
        logger.info(f"  Attack A — ERR_exact={metrics_a['err_exact']:.4f}  "
                    f"ERR_partial={metrics_a['err_partial']:.4f}  "
                    f"BLEU={metrics_a['corpus_bleu']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # ATTACK B — INVERTER MODEL
    # ══════════════════════════════════════════════════════════════════════
    print("\n  ── Running Attack B: Inverter Model ──")
    best_ckpt = os.path.join(INVERTER_CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(best_ckpt):
        logger.warning(f"  Inverter checkpoint not found at {best_ckpt} — skipping Attack B")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  Device: {device}")

        model = AutoModelForSeq2SeqLM.from_pretrained(INVERTER_MODEL_NAME)
        ckpt  = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        logger.info(f"  Loaded inverter from epoch {ckpt.get('epoch')} "
                    f"(eval_loss={ckpt['metrics'].get('eval_loss'):.4f})")

        attack_b  = InverterModelAttack(device, tokenizer, model)
        preds_b, refs_b, ents_b = attack_b.run(eval_pairs)

        metrics_b = MetricsEngine.full_eval(preds_b, refs_b, ents_b, tokenizer)
        by_strategy_b = MetricsEngine.breakdown(
            preds_b, refs_b, ents_b,
            [p.get("strategy", "unknown") for p in eval_pairs]
        )
        by_rarity_b = MetricsEngine.breakdown(
            preds_b, refs_b, ents_b,
            [p.get("name_rarity", "unknown") for p in eval_pairs]
        )
        by_etype_b = MetricsEngine.breakdown(
            preds_b, refs_b, ents_b,
            [p.get("entity_type", "unknown") for p in eval_pairs]
        )

        samples_b = []
        for pred, ref, ent, pair in zip(preds_b[:20], refs_b[:20], ents_b[:20], eval_pairs[:20]):
            samples_b.append({
                "anonymized":       pair.get("anonymized", ""),
                "predicted":        pred,
                "reference":        ref,
                "probe_entity":     ent,
                "entity_recovered": ent in pred if ent else False,
                "strategy":         pair.get("strategy"),
            })

        results["attack_b_inverter"] = {
            "n_eval":            len(eval_pairs),
            "inverter_epoch":    ckpt.get("epoch"),
            "inverter_training_metrics": ckpt.get("metrics", {}),
            "overall_metrics":   metrics_b,
            "by_strategy":       by_strategy_b,
            "by_rarity":         by_rarity_b,
            "by_entity_type":    by_etype_b,
            "samples":           samples_b,
        }
        logger.info(f"  Attack B — ERR_exact={metrics_b['err_exact']:.4f}  "
                    f"ERR_partial={metrics_b['err_partial']:.4f}  "
                    f"BLEU={metrics_b['corpus_bleu']:.4f}")

    # ── Save JSON results ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ATTACK_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  JSON results saved: {ATTACK_RESULTS_FILE}")

    # ── Write text report ─────────────────────────────────────────────────
    write_text_report(results, ATTACK_REPORT_FILE)

    print(f"\n  ✅  Attack results: {ATTACK_RESULTS_FILE}")
    print(f"  ✅  Attack report:  {ATTACK_REPORT_FILE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
