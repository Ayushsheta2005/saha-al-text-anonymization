"""
Inference Script for Adversarially Hardened BART-base PII Anonymizer
=====================================================================
Loads the adversarially fine-tuned BART checkpoint and supports:
  - Interactive sentence-by-sentence anonymization
  - Batch evaluation on curated eval_examples.jsonl
  - Optional side-by-side comparison with the original (unhardened) model
  - Results saved to results/inference_results.jsonl

Checkpoint priority (in order):
  1. final_model.pt  — end-of-training weights (model only, lightweight)
  2. best_model.pt   — best validation checkpoint (full optimizer state)
  3. latest_checkpoint.pt — most recent periodic checkpoint
"""

import os
import sys
import json
import gc
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    VICTIM_MODEL_NAME,
    VICTIM_CHECKPOINT,
    ADV_CHECKPOINT_DIR,
    RESULTS_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_EXAMPLES_FILE = os.path.join(
    os.path.dirname(__file__), "..", "Seq2Seq_model", "eval_examples.jsonl"
)
EVAL_EXAMPLES_FILE = os.path.normpath(EVAL_EXAMPLES_FILE)

ADV_CHECKPOINTS = {
    "final":   os.path.join(ADV_CHECKPOINT_DIR, "final_model.pt"),
    "best":    os.path.join(ADV_CHECKPOINT_DIR, "best_model.pt"),
    "latest":  os.path.join(ADV_CHECKPOINT_DIR, "latest_checkpoint.pt"),
}


# ============================================================
# DISCOVER AVAILABLE CHECKPOINTS
# ============================================================

def discover_checkpoints() -> list[dict]:
    """
    Return a list of dicts for available adversarial checkpoints,
    ordered by priority (final > best > latest).
    """
    found = []
    priority_order = ["final", "best", "latest"]

    for key in priority_order:
        path = ADV_CHECKPOINTS[key]
        if not os.path.exists(path):
            continue

        info = {
            "key":   key,
            "path":  path,
            "label": f"bart-base-adv [{key}]",
            "epoch": None,
            "global_step": None,
            "best_val_loss": None,
            "final_metrics": None,
        }

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            info["epoch"]          = ckpt.get("epoch")
            info["global_step"]    = ckpt.get("global_step")
            info["best_val_loss"]  = ckpt.get("best_val_loss")
            info["final_metrics"]  = ckpt.get("final_metrics") or ckpt.get("val_metrics")
            del ckpt
            gc.collect()
        except Exception as e:
            print(f"  [WARNING] Could not read metadata from {path}: {e}")

        found.append(info)

    return found


# ============================================================
# LOAD MODEL
# ============================================================

def load_adv_model(checkpoint_info: dict, device: torch.device):
    """Load the adversarially hardened BART model from a checkpoint."""
    print(f"\n  Loading tokenizer: {VICTIM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(VICTIM_MODEL_NAME)

    print(f"  Loading base architecture: {VICTIM_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        VICTIM_MODEL_NAME, torch_dtype=torch.float32
    )

    print(f"  Loading adversarial weights from: {checkpoint_info['path']}")
    ckpt = torch.load(checkpoint_info["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    gc.collect()

    model = model.to(device)
    model.eval()
    print(f"  Model ready on {device}.")
    return model, tokenizer


def load_original_model(device: torch.device):
    """Load the original (pre-hardening) BART model for comparison."""
    if not os.path.exists(VICTIM_CHECKPOINT):
        print(f"  [WARNING] Original checkpoint not found: {VICTIM_CHECKPOINT}")
        return None, None

    print(f"\n  Loading original (unhardened) model for comparison...")
    tokenizer = AutoTokenizer.from_pretrained(VICTIM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        VICTIM_MODEL_NAME, torch_dtype=torch.float32
    )
    ckpt = torch.load(VICTIM_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    gc.collect()
    model = model.to(device)
    model.eval()
    print(f"  Original model ready.")
    return model, tokenizer


# ============================================================
# GENERATE PREDICTION
# ============================================================

@torch.no_grad()
def anonymize(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    num_beams: int = 4,
    max_length: int = MAX_TARGET_LENGTH,
) -> str:
    """Run inference on a single input text. BART uses no task prefix."""
    inputs = tokenizer(
        text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ============================================================
# BATCH EVALUATION ON CURATED EXAMPLES
# ============================================================

def load_eval_examples() -> list[dict]:
    if not os.path.exists(EVAL_EXAMPLES_FILE):
        print(f"  [ERROR] eval_examples.jsonl not found at: {EVAL_EXAMPLES_FILE}")
        return []
    examples = []
    with open(EVAL_EXAMPLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_batch_evaluation(
    adv_model,
    tokenizer,
    device: torch.device,
    checkpoint_label: str,
    num_beams: int = 4,
    difficulty_filter: str = None,
    orig_model=None,
):
    """
    Run the adversarial model on all curated eval examples.
    If orig_model is provided, shows a side-by-side diff column.
    """
    examples = load_eval_examples()
    if not examples:
        return

    if difficulty_filter:
        examples = [e for e in examples if e["difficulty"] == difficulty_filter]
        if not examples:
            print(f"  No examples found for difficulty '{difficulty_filter}'.")
            return

    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    examples.sort(key=lambda e: (difficulty_order.get(e["difficulty"], 99), e["id"]))

    compare_mode = orig_model is not None
    total = len(examples)
    print(f"\n{'=' * 110}")
    print(f"  BATCH EVALUATION — {checkpoint_label}")
    print(f"  Examples: {total}  |  Beam width: {num_beams}"
          + ("  |  COMPARE MODE (adv vs original)" if compare_mode else ""))
    if difficulty_filter:
        print(f"  Filter: {difficulty_filter.upper()} only")
    print(f"{'=' * 110}")

    results = []
    current_difficulty = None
    start_time = time.time()

    for i, example in enumerate(examples, 1):
        if example["difficulty"] != current_difficulty:
            current_difficulty = example["difficulty"]
            label = current_difficulty.upper()
            marker = {"easy": "EASY", "medium": "MEDIUM", "hard": "HARD"}.get(current_difficulty, "?")
            print(f"\n  {'─' * 106}")
            print(f"  [{marker}]")
            print(f"  {'─' * 106}")

        adv_output  = anonymize(adv_model, tokenizer, example["input"], device, num_beams=num_beams)
        orig_output = anonymize(orig_model, tokenizer, example["input"], device, num_beams=num_beams) \
                      if compare_mode else None

        adv_changed  = adv_output.strip()  != example["input"].strip()
        orig_changed = orig_output.strip() != example["input"].strip() if compare_mode else None
        is_no_pii    = example.get("category") == "no_pii"

        if is_no_pii:
            adv_status  = "CORRECT (no change)" if not adv_changed  else "FALSE POSITIVE"
            orig_status = "CORRECT (no change)" if not orig_changed else "FALSE POSITIVE"
        else:
            adv_status  = "CHANGED" if adv_changed  else "UNCHANGED"
            orig_status = "CHANGED" if orig_changed else "UNCHANGED"

        print(f"\n  [{example['id']}] {example['category']}")
        print(f"    Notes : {example['notes']}")
        print(f"    INPUT : {example['input']}")
        if compare_mode:
            print(f"    ORIG  : {orig_output}  [{orig_status}]")
        print(f"    ADV   : {adv_output}  [{adv_status}]")

        result = {
            "id":           example["id"],
            "difficulty":   example["difficulty"],
            "category":     example["category"],
            "input":        example["input"],
            "adv_output":   adv_output,
            "adv_changed":  adv_changed,
            "is_no_pii":    is_no_pii,
            "notes":        example["notes"],
        }
        if compare_mode:
            result["orig_output"]  = orig_output
            result["orig_changed"] = orig_changed
        results.append(result)

    elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────────
    pii_examples   = [r for r in results if not r["is_no_pii"]]
    no_pii_examples = [r for r in results if r["is_no_pii"]]

    adv_changed_count  = sum(1 for r in pii_examples if r["adv_changed"])
    pii_total          = len(pii_examples)
    no_pii_correct     = sum(1 for r in no_pii_examples if not r["adv_changed"])
    no_pii_total       = len(no_pii_examples)

    orig_changed_count = sum(1 for r in pii_examples if r.get("orig_changed")) if compare_mode else None

    diff_stats = {}
    for r in results:
        d = r["difficulty"]
        if d not in diff_stats:
            diff_stats[d] = {"total": 0, "adv_changed": 0, "orig_changed": 0,
                             "no_pii_total": 0, "no_pii_correct": 0}
        diff_stats[d]["total"] += 1
        if r["is_no_pii"]:
            diff_stats[d]["no_pii_total"] += 1
            if not r["adv_changed"]:
                diff_stats[d]["no_pii_correct"] += 1
        else:
            if r["adv_changed"]:
                diff_stats[d]["adv_changed"] += 1
            if compare_mode and r.get("orig_changed"):
                diff_stats[d]["orig_changed"] += 1

    print(f"\n{'=' * 110}")
    print(f"  SUMMARY — {checkpoint_label}")
    print(f"{'=' * 110}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total:.2f}s per example)")
    print()
    if compare_mode:
        print(f"  PII anonymized:  ADV {adv_changed_count}/{pii_total} "
              f"({100*adv_changed_count/max(pii_total,1):.0f}%)  |  "
              f"ORIG {orig_changed_count}/{pii_total} "
              f"({100*orig_changed_count/max(pii_total,1):.0f}%)")
    else:
        print(f"  PII anonymized:  {adv_changed_count}/{pii_total} "
              f"({100*adv_changed_count/max(pii_total,1):.0f}%)")
    if no_pii_total > 0:
        print(f"  No-PII correct:  {no_pii_correct}/{no_pii_total} "
              f"({100*no_pii_correct/max(no_pii_total,1):.0f}%)")
    print()

    header = f"  {'Difficulty':<12} {'Adv Anon':<20}"
    if compare_mode:
        header += f" {'Orig Anon':<20}"
    header += f" {'No-PII OK':<20}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for d in ["easy", "medium", "hard"]:
        if d not in diff_stats:
            continue
        s = diff_stats[d]
        pii_in_d = s["total"] - s["no_pii_total"]
        adv_str  = f"{s['adv_changed']}/{pii_in_d}" if pii_in_d > 0 else "—"
        orig_str = f"{s['orig_changed']}/{pii_in_d}" if pii_in_d > 0 else "—"
        nopii_str = f"{s['no_pii_correct']}/{s['no_pii_total']}" if s["no_pii_total"] > 0 else "—"
        row = f"  {d:<12} {adv_str:<20}"
        if compare_mode:
            row += f" {orig_str:<20}"
        row += f" {nopii_str:<20}"
        print(row)
    print(f"{'=' * 110}")

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{difficulty_filter}" if difficulty_filter else ""
    output_path = os.path.join(RESULTS_DIR, f"inference_eval{suffix}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Results saved to: {output_path}")

    return results


# ============================================================
# BATCH INFERENCE ON VALIDATION / TEST JSONL
# ============================================================

def run_val_set(
    adv_model,
    tokenizer,
    device: torch.device,
    val_file: str,
    checkpoint_label: str,
    num_beams: int = 4,
    orig_model=None,
):
    """
    Run the adversarial model over every example in a JSONL val file.
    Expected fields per line: 'original_text' (input), 'anonymized_text' (gold reference).
    Outputs predictions to results/ and prints a live progress bar with metrics.
    """
    if not os.path.exists(val_file):
        print(f"  [ERROR] Val file not found: {val_file}")
        return

    print(f"\n  Loading val file: {val_file}")
    examples = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    total = len(examples)
    compare_mode = orig_model is not None
    print(f"  Examples: {total}  |  Beam: {num_beams}"
          + ("  |  COMPARE MODE" if compare_mode else ""))

    # Lazy-import metrics
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer as rouge_lib
        smoother = SmoothingFunction().method1
        rouge_eval = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
        compute_metrics = True
    except ImportError:
        print("  [INFO] nltk/rouge_score not found — skipping BLEU/ROUGE (predictions still saved)")
        compute_metrics = False

    results       = []
    bleu_scores   = []
    rougeL_scores = []
    exact_matches = 0
    leakage_count = 0  # entity strings present in output unchanged

    start_time = time.time()
    print()
    print(f"  {'Idx':>6}  {'BLEU-4':>7}  {'ROUGE-L':>8}  {'Elapsed':>8}  Preview (first 80 chars)")
    print("  " + "─" * 90)

    for idx, ex in enumerate(examples, 1):
        original  = ex.get("original_text", ex.get("input", ""))
        reference = ex.get("anonymized_text", ex.get("target", ""))
        entities  = [e["text"] for e in ex.get("entities", [])]

        adv_out  = anonymize(adv_model, tokenizer, original, device, num_beams=num_beams)
        orig_out = anonymize(orig_model, tokenizer, original, device, num_beams=num_beams) \
                   if compare_mode else None

        # ── Metrics ────────────────────────────────────────────────────────
        bleu4    = None
        rougel   = None
        if compute_metrics and reference:
            ref_tokens  = reference.split()
            hyp_tokens  = adv_out.split()
            bleu4  = sentence_bleu([ref_tokens], hyp_tokens,
                                   weights=(0.25,0.25,0.25,0.25),
                                   smoothing_function=smoother)
            rougel = rouge_eval.score(reference, adv_out)["rougeL"].fmeasure
            bleu_scores.append(bleu4)
            rougeL_scores.append(rougel)

        if adv_out.strip() == reference.strip():
            exact_matches += 1

        # Entity leakage: any original entity still verbatim in adv output
        leaked = [e for e in entities if e and e.lower() in adv_out.lower()]
        if leaked:
            leakage_count += 1

        # ── Save result ─────────────────────────────────────────────────────
        record = {
            "id":          ex.get("id", idx),
            "original":    original,
            "reference":   reference,
            "adv_output":  adv_out,
            "bleu4":       round(bleu4,  4) if bleu4  is not None else None,
            "rougeL":      round(rougel, 4) if rougel is not None else None,
            "leaked_entities": leaked,
        }
        if compare_mode:
            record["orig_output"] = orig_out
        results.append(record)

        # ── Progress line every 10 examples ─────────────────────────────────
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start_time
            avg_bleu  = sum(bleu_scores)  / len(bleu_scores)  if bleu_scores  else 0.0
            avg_rouge = sum(rougeL_scores)/ len(rougeL_scores)if rougeL_scores else 0.0
            eta_str   = f"{elapsed * (total - idx) / idx:.0f}s" if idx < total else "done"
            preview   = adv_out[:80].replace("\n", " ")
            print(f"  {idx:>6}/{total}  {avg_bleu:>7.4f}  {avg_rouge*100:>7.2f}%  "
                  f"{elapsed:>6.0f}s/eta {eta_str:<6}  {preview}")

    elapsed = time.time() - start_time

    # ── Aggregate metrics ────────────────────────────────────────────────────
    n = len(results)
    avg_bleu   = sum(bleu_scores)   / len(bleu_scores)   if bleu_scores   else None
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else None
    exact_pct  = 100.0 * exact_matches / n if n else 0.0
    elr        = round(leakage_count / n, 4) if n else 0.0

    print(f"\n{'=' * 90}")
    print(f"  RESULTS — {checkpoint_label}")
    print(f"{'=' * 90}")
    print(f"  Examples        : {n}")
    print(f"  Time            : {elapsed:.1f}s  ({elapsed/n:.2f}s per sample)")
    if avg_bleu   is not None: print(f"  BLEU-4 (avg)    : {avg_bleu:.4f}")
    if avg_rougeL is not None: print(f"  ROUGE-L (avg)   : {avg_rougeL*100:.2f}%")
    print(f"  Exact match     : {exact_matches}/{n}  ({exact_pct:.2f}%)")
    print(f"  Entity leak     : {leakage_count}/{n}  ({elr:.4f})")
    print(f"{'=' * 90}")

    # ── Save predictions ─────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_tag   = checkpoint_label.split("[")[-1].rstrip("]") if "[" in checkpoint_label else "adv"
    pred_path  = os.path.join(RESULTS_DIR, f"val_predictions_{ckpt_tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Predictions saved to : {pred_path}")

    # Also save a summary JSON
    summary = {
        "checkpoint":   checkpoint_label,
        "val_file":     val_file,
        "num_examples": n,
        "avg_bleu4":    round(avg_bleu, 4)    if avg_bleu   is not None else None,
        "avg_rougeL":   round(avg_rougeL, 4)  if avg_rougeL is not None else None,
        "exact_match_pct": round(exact_pct, 4),
        "entity_leakage_rate": elr,
        "time_seconds": round(elapsed, 1),
    }
    summary_path = os.path.join(RESULTS_DIR, f"val_summary_{ckpt_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to     : {summary_path}")

    return results


# ============================================================
# SELECT CHECKPOINT (interactive)
# ============================================================

def select_checkpoint(checkpoints: list[dict]) -> dict:
    print("\n" + "=" * 80)
    print("  AVAILABLE ADVERSARIAL CHECKPOINTS")
    print("=" * 80)
    print(f"  {'#':<4} {'Type':<10} {'Step':<10} {'Epoch':<8} {'Best Val Loss':<18} {'Metrics'}")
    print("  " + "─" * 78)

    for i, info in enumerate(checkpoints, 1):
        step_str  = str(info["global_step"]) if info["global_step"] is not None else "N/A"
        epoch_str = str(info["epoch"])       if info["epoch"]       is not None else "N/A"
        loss_str  = f"{info['best_val_loss']:.4f}" if info["best_val_loss"] is not None else "N/A"
        metrics_str = ""
        if info["final_metrics"]:
            m = info["final_metrics"]
            parts = []
            if "val_L1" in m:
                parts.append(f"L1={m['val_L1']:.3f}")
            if "val_L2" in m:
                parts.append(f"L2={m['val_L2']:.1f}")
            if "tok_acc" in m:
                parts.append(f"TokAcc={m['tok_acc']:.3f}")
            if "leakage_rate" in m:
                parts.append(f"Leak={m['leakage_rate']:.3f}")
            metrics_str = "  |  " + "  ".join(parts) if parts else ""
        print(f"  {i:<4} {info['key']:<10} {step_str:<10} {epoch_str:<8} {loss_str:<18}{metrics_str}")

    print("  " + "─" * 78)

    while True:
        try:
            user_input = input(f"\n  Select checkpoint (1-{len(checkpoints)}) or 'q' to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)
        if user_input.lower() == 'q':
            print("  Exiting.")
            sys.exit(0)
        try:
            idx = int(user_input)
            if 1 <= idx <= len(checkpoints):
                return checkpoints[idx - 1]
            print(f"  Invalid number. Enter 1-{len(checkpoints)}.")
        except ValueError:
            print(f"  Invalid input. Enter a number or 'q'.")


# ============================================================
# INTERACTIVE LOOP
# ============================================================

def interactive_loop(
    adv_model,
    tokenizer,
    device: torch.device,
    checkpoint_label: str,
    orig_model=None,
):
    """Main REPL: user types text, gets anonymized output."""
    compare_mode = orig_model is not None
    print("\n" + "=" * 80)
    print(f"  INFERENCE — {checkpoint_label}")
    if compare_mode:
        print("  Compare mode: ON (showing original model output alongside)")
    print("=" * 80)
    print("  Commands:")
    print("    q              — quit")
    print("    switch         — change checkpoint")
    print("    beam N         — set beam width (default: 4)")
    print("    compare        — toggle original model comparison")
    print("    eval           — run all curated eval examples")
    print("    eval easy      — run only easy examples")
    print("    eval medium    — run only medium examples")
    print("    eval hard      — run only hard examples")
    print("=" * 80)

    num_beams = 4

    while True:
        try:
            user_input = input("\n  INPUT > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            return "quit"

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == 'q':
            return "quit"

        if cmd == 'switch':
            return "switch"

        if cmd == 'compare':
            compare_mode = not compare_mode
            state = "ON" if compare_mode else "OFF"
            if compare_mode and orig_model is None:
                print("  [INFO] Original model not loaded. Restart with --compare flag.")
                compare_mode = False
            else:
                print(f"  Compare mode: {state}")
            continue

        if cmd.startswith('beam'):
            try:
                num_beams = int(user_input.split()[1])
                print(f"  Beam width set to {num_beams}")
            except (IndexError, ValueError):
                print("  Usage: beam N  (e.g., beam 4)")
            continue

        if cmd.startswith('eval'):
            parts = cmd.split()
            difficulty_filter = None
            if len(parts) > 1 and parts[1] in ("easy", "medium", "hard"):
                difficulty_filter = parts[1]
            run_batch_evaluation(
                adv_model, tokenizer, device, checkpoint_label,
                num_beams=num_beams,
                difficulty_filter=difficulty_filter,
                orig_model=orig_model if compare_mode else None,
            )
            continue

        # ── Run inference ──
        adv_out = anonymize(adv_model, tokenizer, user_input, device, num_beams=num_beams)
        if compare_mode and orig_model is not None:
            orig_out = anonymize(orig_model, tokenizer, user_input, device, num_beams=num_beams)
            print(f"  ORIG > {orig_out}")
        print(f"  ADV  > {adv_out}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive inference for adversarially hardened BART-base"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also load original (unhardened) model for side-by-side comparison"
    )
    parser.add_argument(
        "--checkpoint", choices=["final", "best", "latest"],
        help="Directly select a checkpoint type without the interactive menu"
    )
    parser.add_argument(
        "--beam", type=int, default=4,
        help="Beam search width (default: 4)"
    )
    parser.add_argument(
        "--val", action="store_true",
        help="Run batch inference on benchmark validation set and save predictions"
    )
    parser.add_argument(
        "--val-file",
        default=None,
        help="Path to a custom JSONL val file (overrides default benchmark val set)"
    )
    parser.add_argument(
        "--val-batch", type=int, default=1,
        help="Batch size for val inference (default: 1; increase if VRAM allows)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ADVERSARIAL BART — PII ANONYMIZER INFERENCE")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  Device: CPU")

    # Discover checkpoints
    checkpoints = discover_checkpoints()
    if not checkpoints:
        print(f"\n  [ERROR] No adversarial checkpoints found in: {ADV_CHECKPOINT_DIR}")
        print("  Train the model first with: python train_adv.py")
        sys.exit(1)

    print(f"\n  Found {len(checkpoints)} checkpoint(s).")

    # Load original model if requested (do it once, outside the switch loop)
    orig_model = orig_tokenizer = None
    if args.compare:
        orig_model, orig_tokenizer = load_original_model(device)

    # ── Val-set batch inference mode ──────────────────────────────────────
    if args.val or args.val_file:
        # Pick checkpoint
        if args.checkpoint:
            matching = [c for c in checkpoints if c["key"] == args.checkpoint]
            if not matching:
                print(f"  [ERROR] Checkpoint '{args.checkpoint}' not available.")
                sys.exit(1)
            selected = matching[0]
        else:
            selected = checkpoints[0]   # default: final
            print(f"  No --checkpoint specified; using: {selected['key']}")

        model, tokenizer = load_adv_model(selected, device)

        val_file = args.val_file or os.path.join(
            os.path.dirname(__file__), "..", "benchmark", "data", "validation.jsonl"
        )
        val_file = os.path.normpath(val_file)
        run_val_set(
            model, tokenizer, device,
            val_file=val_file,
            checkpoint_label=selected["label"],
            num_beams=args.beam,
            orig_model=orig_model,
        )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n  Done.")
        sys.exit(0)

    # Main loop — supports 'switch' to reload a different checkpoint
    loaded_model = loaded_tokenizer = None

    while True:
        # ── Select checkpoint ──────────────────────────────────────────────
        if args.checkpoint:
            # Command-line shortcut: pick directly
            matching = [c for c in checkpoints if c["key"] == args.checkpoint]
            if not matching:
                print(f"  [ERROR] Checkpoint type '{args.checkpoint}' not available.")
                print(f"  Available: {[c['key'] for c in checkpoints]}")
                sys.exit(1)
            selected = matching[0]
            args.checkpoint = None   # only auto-select on first pass
        else:
            selected = select_checkpoint(checkpoints)

        # ── Cleanup previous model ─────────────────────────────────────────
        if loaded_model is not None:
            del loaded_model, loaded_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Load model ─────────────────────────────────────────────────────
        loaded_model, loaded_tokenizer = load_adv_model(selected, device)

        # ── Run interactive loop ───────────────────────────────────────────
        result = interactive_loop(
            loaded_model,
            loaded_tokenizer,
            device,
            selected["label"],
            orig_model=orig_model,
        )

        if result == "quit":
            break
        # result == "switch" → loop back to checkpoint selection

    # Cleanup
    if loaded_model is not None:
        del loaded_model, loaded_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if orig_model is not None:
        del orig_model, orig_tokenizer
        gc.collect()

    print("\n  Goodbye!")


if __name__ == "__main__":
    main()
