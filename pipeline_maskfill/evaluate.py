# ==============================================================================
# evaluate.py — Comprehensive Evaluation Framework
# ==============================================================================
# Evaluates combined pipeline output across privacy, utility, and robustness.
#
# Privacy:  Entity Leak Rate, Token Leak Rate, CRR (2/3/4-gram), Per-Entity Leak
# Utility:  BERTScore F1, BLEU, ROUGE-1/2/L
# NER:      Entity-level P/R/F1 (seqeval)
# Output:   JSON results, comparison table, Pareto plot, sample outputs
# ==============================================================================

import os
import re
import json
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np

from config import OUTPUT_DIR, EVAL_DIR, EVAL_SAMPLES_FOR_DISPLAY

log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# Privacy Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_entity_leakage(originals: List[str], anonymized: List[str]) -> Dict:
    """
    Compute entity leak rate: fraction of original PII entities that appear
    verbatim in the anonymized output.

    Uses case-insensitive substring matching on capitalized multi-word spans.
    """
    total_entities, leaked_entities = 0, 0
    total_tokens, leaked_tokens = 0, 0

    for orig, anon in zip(originals, anonymized):
        anon_lower = anon.lower()

        # Token-level: any capitalized word > 2 chars
        for word in orig.split():
            if len(word) > 2 and word[0].isupper():
                total_tokens += 1
                if word.lower() in anon_lower:
                    leaked_tokens += 1

        # Entity-level: capitalized multi-word spans
        for entity in re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', orig):
            if len(entity) > 2:
                total_entities += 1
                if entity.lower() in anon_lower:
                    leaked_entities += 1

    return {
        "entity_leak_rate": round(leaked_entities / max(total_entities, 1) * 100, 2),
        "token_leak_rate": round(leaked_tokens / max(total_tokens, 1) * 100, 2),
        "leaked_entities": leaked_entities,
        "total_entities": total_entities,
        "leaked_tokens": leaked_tokens,
        "total_tokens": total_tokens,
    }


def compute_per_entity_leakage(originals: List[str], anonymized: List[str]) -> Dict:
    """Compute leak rate broken down by entity type via regex heuristics."""
    patterns = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        "PHONE": r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        "URL": r'https?://[^\s]+',
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        "NUMBER": r'\b\d{5,}\b',
        "NAME": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
    }
    results = {}
    for etype, pattern in patterns.items():
        total, leaked = 0, 0
        for orig, anon in zip(originals, anonymized):
            matches = re.findall(pattern, orig)
            anon_lower = anon.lower()
            for m in matches:
                total += 1
                if m.lower() in anon_lower:
                    leaked += 1
        if total > 0:
            results[etype] = {
                "leaked": leaked, "total": total,
                "rate": round(leaked / total * 100, 1),
            }
    return results


def compute_crr(originals: List[str], anonymized: List[str], n_grams: int = 3) -> Dict:
    """
    Contextual Re-identification Risk: measures how many identifying n-grams
    from originals survive in the anonymized text.
    """
    risk, total = 0, 0
    for orig, anon in zip(originals, anonymized):
        orig_words = orig.split()
        anon_lower = anon.lower()
        for i in range(len(orig_words) - n_grams + 1):
            ngram = " ".join(orig_words[i:i + n_grams]).lower()
            # Only count n-grams containing a capitalized word
            if any(w[0].isupper() for w in orig_words[i:i + n_grams] if w):
                total += 1
                if ngram in anon_lower:
                    risk += 1
    return {
        "crr": round(risk / max(total, 1) * 100, 2),
        "identifying_ngrams_leaked": risk,
        "total_identifying_ngrams": total,
    }


def compute_multi_crr(originals: List[str], anonymized: List[str]) -> Dict:
    """CRR at 2, 3, and 4-gram granularities."""
    return {
        f"crr_{n}gram": compute_crr(originals, anonymized, n)["crr"]
        for n in [2, 3, 4]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_utility_metrics(originals: List[str], anonymized: List[str]) -> Dict:
    """Compute ROUGE, BLEU, and BERTScore."""
    import evaluate as hf_evaluate

    results = {}

    # ROUGE
    log.info("  Computing ROUGE ...")
    rouge_m = hf_evaluate.load("rouge")
    rouge = rouge_m.compute(predictions=anonymized, references=originals)
    results["rouge"] = {k: round(v, 4) for k, v in rouge.items() if isinstance(v, float)}

    # BLEU
    log.info("  Computing BLEU ...")
    bleu_m = hf_evaluate.load("sacrebleu")
    bleu = bleu_m.compute(predictions=anonymized, references=[[r] for r in originals])
    results["bleu"] = round(bleu["score"], 2)

    # BERTScore
    log.info("  Computing BERTScore ...")
    try:
        bsm = hf_evaluate.load("bertscore")
        bs = bsm.compute(predictions=anonymized, references=originals, lang="en")
        bs_f1_scores = [float(x) for x in bs["f1"]]
        results["bertscore_f1"] = round(float(np.mean(bs_f1_scores)), 4)
        results["bertscore_f1_std"] = round(float(np.std(bs_f1_scores)), 4)
    except Exception as e:
        log.warning(f"  BERTScore failed: {e}")
        results["bertscore_f1"] = 0.0

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Length Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_length_stats(originals: List[str], anonymized: List[str]) -> Dict:
    """Analyze length distribution of original vs anonymized text."""
    orig_lens = [len(o.split()) for o in originals]
    anon_lens = [len(a.split()) for a in anonymized]
    ratios = [al / max(ol, 1) for ol, al in zip(orig_lens, anon_lens)]
    return {
        "orig_mean_words": round(float(np.mean(orig_lens)), 1),
        "anon_mean_words": round(float(np.mean(anon_lens)), 1),
        "ratio_mean": round(float(np.mean(ratios)), 3),
        "ratio_std": round(float(np.std(ratios)), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 37 Curated Edge-Case Examples
# ═══════════════════════════════════════════════════════════════════════════════

CURATED_EXAMPLES = [
    # EASY (10)
    {"id": "e01", "cat": "name", "diff": "easy", "text": "Please contact John regarding the project update."},
    {"id": "e02", "cat": "name", "diff": "easy", "text": "Maria has submitted the report on time."},
    {"id": "e03", "cat": "location", "diff": "easy", "text": "I live in London and work remotely."},
    {"id": "e04", "cat": "location", "diff": "easy", "text": "The office is located in Chicago."},
    {"id": "e05", "cat": "number", "diff": "easy", "text": "My account number is 74829361."},
    {"id": "e06", "cat": "number", "diff": "easy", "text": "Please reference ticket number 55123."},
    {"id": "e07", "cat": "email", "diff": "easy", "text": "You can reach me at sarah.jones@gmail.com for details."},
    {"id": "e08", "cat": "name", "diff": "easy", "text": "Thank you, David, for your quick response."},
    {"id": "e09", "cat": "date", "diff": "easy", "text": "The appointment is scheduled for 15/07/2025."},
    {"id": "e10", "cat": "location", "diff": "easy", "text": "She moved to Berlin last year."},
    # MEDIUM (12)
    {"id": "m01", "cat": "multi", "diff": "medium", "text": "Dear Michael Thompson, your invoice has been processed."},
    {"id": "m02", "cat": "multi", "diff": "medium", "text": "Jessica Parker can be contacted at jessica.parker@outlook.com."},
    {"id": "m03", "cat": "multi", "diff": "medium", "text": "Robert Chen from San Francisco submitted the application."},
    {"id": "m04", "cat": "multi", "diff": "medium", "text": "Hello Priya Sharma, we need to discuss your account 48291037."},
    {"id": "m05", "cat": "multi", "diff": "medium", "text": "Please call Ahmed Hassan at +44 7911 123456 to confirm."},
    {"id": "m06", "cat": "multi", "diff": "medium", "text": "The meeting between Lisa Wong and James Miller was rescheduled."},
    {"id": "m07", "cat": "multi", "diff": "medium", "text": "Send the verification code 83921 to admin@techcorp.io."},
    {"id": "m08", "cat": "multi", "diff": "medium", "text": "Emily Rodriguez was born on 23/04/1992 per our records."},
    {"id": "m09", "cat": "multi", "diff": "medium", "text": "The property at 742 Evergreen Terrace, Springfield has ID 90210."},
    {"id": "m10", "cat": "case", "diff": "medium", "text": "my name is alex morgan and i live in seattle."},
    {"id": "m11", "cat": "case", "diff": "medium", "text": "CONTACT EMMA WATSON AT EMMA.WATSON@YAHOO.COM IMMEDIATELY."},
    {"id": "m12", "cat": "case", "diff": "medium", "text": "jOHN sMITH lives in nEW yORK and his email is John@Gmail.Com."},
    # HARD (15)
    {"id": "h01", "cat": "dense", "diff": "hard", "text": "Dr. Samantha Clarke from Boston General Hospital can be reached at samantha.clarke@bgh.org or +1 617 555 0192 regarding patient file 2847193."},
    {"id": "h02", "cat": "dense", "diff": "hard", "text": "Hi, I'm Rajesh Kumar. My employee ID is EMP-78432, my email is rajesh.k@infosys.com, and I work at the Bangalore office."},
    {"id": "h03", "cat": "dense", "diff": "hard", "text": "Transfer $5,000 from account 9821-4573-0012 to Maria Gonzalez (maria.g@bankmail.com) at 45 Oak Street, Miami, FL 33101."},
    {"id": "h04", "cat": "long", "diff": "hard", "text": "Following up on our conversation, Daniel Kim mentioned that the project deadline is 31/12/2025. His colleague, Sophie Martin, suggested we consult the client, Nakamura Industries, before proceeding. You can reach Daniel at daniel.kim@company.co or Sophie at +33 6 12 34 56 78."},
    {"id": "h05", "cat": "informal", "diff": "hard", "text": "yo hit up mike at mike99@hotmail.com or txt him at 555-867-5309 hes in LA rn"},
    {"id": "h06", "cat": "typos", "diff": "hard", "text": "Plese contcat Jonh Smtih at jonh.smith@gmal.com abuot the accont 73829."},
    {"id": "h07", "cat": "multilang", "diff": "hard", "text": "The visa for François Müller-Björkström was processed at the São Paulo consulate on 08/11/2024."},
    {"id": "h08", "cat": "embedded", "diff": "hard", "text": "Username: alex_chen_1995, Password reset email sent to alexchen@protonmail.com, last login from IP 192.168.1.42."},
    {"id": "h09", "cat": "ambiguous", "diff": "hard", "text": "Apple hired Jordan from Amazon. Jordan's first day in Cupertino is March 15th."},
    {"id": "h10", "cat": "no_pii", "diff": "hard", "text": "The weather forecast predicts rain tomorrow with temperatures around 15 degrees."},
    {"id": "h11", "cat": "no_pii", "diff": "hard", "text": "Please review the quarterly report and submit your feedback by Friday."},
    {"id": "h12", "cat": "repeated", "diff": "hard", "text": "Call Sarah. Sarah's number is 555-0147. Tell Sarah that Sarah's appointment is confirmed."},
    {"id": "h13", "cat": "tabular", "diff": "hard", "text": "Name: Wei Zhang, DOB: 12/03/1988, SSN: 123-45-6789, Address: 88 Pine Road, Austin, TX 73301."},
    {"id": "h14", "cat": "conversational", "diff": "hard", "text": "Hey, it's Tom. Package goes to 1520 Maple Avenue, Portland. Zip 97201, phone 503-555-0198."},
    {"id": "h15", "cat": "edge", "diff": "hard", "text": "Patient ID: P-2024-08173, Room 42B, admitted on 01/15/2024 by Dr. Ananya Patel, contact: ananya.p@hospital.org."},
]


def run_curated_eval(anonymize_fn, model_label: str = "pipeline") -> Dict:
    """Run curated examples and compute per-difficulty/per-category breakdown."""
    from tqdm.auto import tqdm

    log.info(f"\n{'═' * 65}")
    log.info(f"  CURATED EVALUATION ({len(CURATED_EXAMPLES)} examples) — {model_label}")
    log.info(f"{'═' * 65}")

    results_by_diff = {"easy": [], "medium": [], "hard": []}
    pii_ok, pii_tot = 0, 0
    nopii_ok, nopii_tot = 0, 0

    for ex in tqdm(CURATED_EXAMPLES, desc=f"Curated eval ({model_label})", leave=False):
        out = anonymize_fn(ex["text"])

        if ex["cat"] == "no_pii":
            ok = (ex["text"].strip() == out.strip())
            status = "CORRECT" if ok else "FALSE_POSITIVE"
            nopii_tot += 1
            if ok:
                nopii_ok += 1
        else:
            changed = (ex["text"].strip() != out.strip())
            status = "CHANGED" if changed else "UNCHANGED"
            pii_tot += 1
            if changed:
                pii_ok += 1

        results_by_diff[ex["diff"]].append({
            "id": ex["id"], "cat": ex["cat"], "status": status,
            "input": ex["text"], "output": out,
        })

    # Per-difficulty stats
    per_diff = {}
    for diff, items in results_by_diff.items():
        ok = sum(1 for r in items if r["status"] in ("CHANGED", "CORRECT"))
        per_diff[diff] = {"total": len(items), "success": ok,
                          "rate": round(ok / max(len(items), 1) * 100, 1)}

    summary = {
        "pii_rate": round(pii_ok / max(pii_tot, 1) * 100, 1),
        "nopii_rate": round(nopii_ok / max(nopii_tot, 1) * 100, 1),
        "per_difficulty": per_diff,
    }

    # Log results
    log.info(f"  PII anonymized: {pii_ok}/{pii_tot} ({summary['pii_rate']}%)")
    log.info(f"  No-PII correct: {nopii_ok}/{nopii_tot} ({summary['nopii_rate']}%)")
    for diff, stats in per_diff.items():
        log.info(f"  {diff.capitalize()}: {stats['success']}/{stats['total']} ({stats['rate']}%)")

    # Log individual results
    log.info(f"\n  Detailed Results:")
    for diff in ["easy", "medium", "hard"]:
        for r in results_by_diff[diff]:
            log.info(f"    [{r['id']}] {r['status']:>14} | {r['cat']}")
            log.info(f"      IN:  {r['input'][:120]}")
            log.info(f"      OUT: {r['output'][:120]}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Full Evaluation Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_pipeline(
    originals: List[str],
    anonymized: List[str],
    anonymize_fn,
    encoder_name: str,
    filler_name: str,
) -> Dict:
    """
    Run all evaluations on a pipeline combination.

    Args:
        originals: Original texts (from test set)
        anonymized: Anonymized texts (pipeline output)
        anonymize_fn: Callable for curated eval
        encoder_name: Name of encoder model
        filler_name: Name of filler model

    Returns:
        Comprehensive results dict
    """
    combo_name = f"{encoder_name}+{filler_name}"
    n = len(originals)

    log.info(f"\n{'═' * 70}")
    log.info(f"  EVALUATING PIPELINE: {combo_name}  ({n} examples)")
    log.info(f"{'═' * 70}")

    # Privacy
    log.info("\n  ── Privacy Metrics ──")
    leakage = compute_entity_leakage(originals, anonymized)
    per_entity = compute_per_entity_leakage(originals, anonymized)
    crr = compute_crr(originals, anonymized)
    multi_crr = compute_multi_crr(originals, anonymized)

    # Utility
    log.info("\n  ── Utility Metrics ──")
    utility = compute_utility_metrics(originals, anonymized)

    # Length
    length = compute_length_stats(originals, anonymized)

    # Curated eval
    curated = run_curated_eval(anonymize_fn, combo_name)

    # Assemble results
    results = {
        "pipeline": combo_name,
        "encoder": encoder_name,
        "filler": filler_name,
        "n_examples": n,
        "leakage": leakage,
        "per_entity_leakage": per_entity,
        "crr": crr,
        "multi_crr": multi_crr,
        **utility,
        "length_stats": length,
        "curated": curated,
    }

    # Print summary table
    _print_summary(results)

    # Save sample outputs
    _save_sample_outputs(originals, anonymized, combo_name)

    # Save results JSON
    os.makedirs(EVAL_DIR, exist_ok=True)
    results_path = os.path.join(EVAL_DIR, f"results_{combo_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\n  Results saved: {results_path}")

    return results


def _print_summary(results: Dict):
    """Print a formatted summary table of all metrics."""
    log.info(f"\n  ┌─── RESULTS SUMMARY: {results['pipeline']} ───────────────────┐")
    log.info(f"  │  {'METRIC':<35} {'VALUE':>12} │")
    log.info(f"  │  {'─' * 49} │")

    # Privacy
    log.info(f"  │  {'Entity Leak Rate ↓':<35} {results['leakage']['entity_leak_rate']:>11.2f}% │")
    log.info(f"  │  {'Token Leak Rate ↓':<35} {results['leakage']['token_leak_rate']:>11.2f}% │")
    log.info(f"  │  {'Leaked / Total Entities':<35} {results['leakage']['leaked_entities']:>5} / {results['leakage']['total_entities']:<5} │")
    log.info(f"  │  {'CRR (3-gram) ↓':<35} {results['crr']['crr']:>11.2f}% │")
    log.info(f"  │  {'─' * 49} │")

    # Utility
    rouge = results.get("rouge", {})
    log.info(f"  │  {'ROUGE-1 ↑':<35} {rouge.get('rouge1', 0):>11.4f} │")
    log.info(f"  │  {'ROUGE-2 ↑':<35} {rouge.get('rouge2', 0):>11.4f} │")
    log.info(f"  │  {'ROUGE-L ↑':<35} {rouge.get('rougeL', 0):>11.4f} │")
    log.info(f"  │  {'BLEU ↑':<35} {results.get('bleu', 0):>11.2f} │")
    log.info(f"  │  {'BERTScore F1 ↑':<35} {results.get('bertscore_f1', 0):>11.4f} │")
    log.info(f"  │  {'─' * 49} │")

    # Length
    ls = results.get("length_stats", {})
    log.info(f"  │  {'Avg orig words':<35} {ls.get('orig_mean_words', 0):>11.1f} │")
    log.info(f"  │  {'Avg anon words':<35} {ls.get('anon_mean_words', 0):>11.1f} │")
    log.info(f"  │  {'Length ratio':<35} {ls.get('ratio_mean', 0):>11.3f} │")
    log.info(f"  └─────────────────────────────────────────────────────┘")

    # Per-entity breakdown
    per_ent = results.get("per_entity_leakage", {})
    if per_ent:
        log.info(f"\n  Per-Entity-Type Leakage:")
        log.info(f"  {'Entity':<15} {'Leaked':>8} {'Total':>8} {'Rate%':>8}")
        log.info(f"  {'─' * 41}")
        for etype, info in sorted(per_ent.items(), key=lambda x: -x[1].get("rate", 0)):
            log.info(f"  {etype:<15} {info['leaked']:>8} {info['total']:>8} {info['rate']:>7.1f}%")


def _save_sample_outputs(originals, anonymized, combo_name):
    """Save sample input/output pairs for manual inspection."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    n = min(EVAL_SAMPLES_FOR_DISPLAY, len(originals))
    path = os.path.join(EVAL_DIR, f"samples_{combo_name}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Sample Outputs — {combo_name}\n")
        f.write(f"{'=' * 80}\n\n")
        for i in range(n):
            f.write(f"[{i+1}] Original:\n  {originals[i]}\n")
            f.write(f"    Anonymized:\n  {anonymized[i]}\n\n")

    log.info(f"  Sample outputs saved: {path}")
