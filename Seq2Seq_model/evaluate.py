"""
Comprehensive Evaluation Script for Seq2Seq PII Anonymization
==============================================================
Automatically discovers all trained models from checkpoints/,
evaluates each on:
  1. eval_examples.jsonl  (curated difficulty-graded examples)
  2. test set             (held-out split from training data)

Evaluates smaller models first, then larger ones.

Computes ALL metrics:
  - Exact Match, Word Accuracy
  - BLEU (1, 2, 4, overall)
  - ROUGE (1, 2, L)
  - BERTScore (P, R, F1)
  - Entity Leakage Rate

Reports model sizes:
  - Parameter count (total & trainable)
  - Checkpoint file size on disk

Writes results to:
  - evaluation_results.json          (structured, machine-readable)
  - evaluation_results_readable.txt  (formatted, human-readable)

Usage:
    python evaluate.py
"""

import os
import sys
import gc
import json
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from config import (
    MODEL_CONFIGS,
    CHECKPOINTS_DIR,
    DATA_SPLITS_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)
from dataset import AnonymizationDataset, load_split_data
from utils import (
    compute_all_metrics,
    compute_token_accuracy,
    compute_word_level_accuracy,
    count_parameters,
    format_time,
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_EXAMPLES_FILE = os.path.join(BASE_DIR, "eval_examples.jsonl")
RESULTS_JSON = os.path.join(BASE_DIR, "evaluation_results2.json")
RESULTS_TXT = os.path.join(BASE_DIR, "evaluation_results_readable2.txt")

EVAL_BATCH_SIZE = 16
NUM_WORKERS = 2


# ============================================================
# MODEL SIZE INFO (approximate params in millions — for sorting)
# ============================================================
# These are approximate pre-trained parameter counts so we can sort
# models smallest→largest WITHOUT loading them into memory first.

MODEL_SIZE_MILLIONS = {
    "t5-efficient-tiny":    16,    # ~16M params
    "t5-small":             60,    # ~60M params
    "flan-t5-small":        77,    # ~77M params
    "distilbart":           230,   # ~230M params
    "bart-base":            139,   # ~139M params
    "flan-t5-base-qlora":   248,   # ~248M base (only LoRA adapters trained)
}


def get_checkpoint_file_size_mb(checkpoint_path):
    """Return checkpoint file size in MB, or None if file doesn't exist."""
    if os.path.exists(checkpoint_path):
        return round(os.path.getsize(checkpoint_path) / (1024 * 1024), 1)
    return None


def format_params(millions):
    """Format parameter count nicely."""
    if millions >= 1000:
        return f"{millions / 1000:.1f}B"
    return f"{millions}M"


# ============================================================
# DISCOVER TRAINED MODELS (sorted smallest → largest)
# ============================================================

def discover_trained_models():
    """
    Scan checkpoints/ for models that have a best_model.pt file.
    Returns list sorted by model size (smallest first).
    """
    trained = []
    if not os.path.exists(CHECKPOINTS_DIR):
        return trained

    for model_key in os.listdir(CHECKPOINTS_DIR):
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, model_key)
        best_path = os.path.join(checkpoint_dir, "best_model.pt")

        if not os.path.isdir(checkpoint_dir) or not os.path.exists(best_path):
            continue
        if model_key not in MODEL_CONFIGS:
            continue

        config = MODEL_CONFIGS[model_key]
        info = {
            "model_key": model_key,
            "model_name": config["model_name"],
            "prefix": config.get("prefix", ""),
            "use_qlora": config.get("use_qlora", False),
            "checkpoint_path": best_path,
            "checkpoint_dir": checkpoint_dir,
            "approx_size_millions": MODEL_SIZE_MILLIONS.get(model_key, 999),
            "checkpoint_size_mb": get_checkpoint_file_size_mb(best_path),
        }

        # Read best_val_loss from history or checkpoint
        hist_path = os.path.join(checkpoint_dir, "training_history.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    info["best_val_loss"] = json.load(f).get("best_val_loss")
            except Exception:
                info["best_val_loss"] = None
        else:
            info["best_val_loss"] = None

        trained.append(info)

    # Sort by approximate model size — smallest first
    trained.sort(key=lambda m: m["approx_size_millions"])

    return trained


# ============================================================
# LOAD MODEL FOR INFERENCE
# ============================================================

def load_model(model_info, device):
    """Load a trained model and tokenizer from checkpoint."""
    config = MODEL_CONFIGS[model_info["model_key"]]
    model_name = config["model_name"]
    use_qlora = config.get("use_qlora", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        adapter_path = os.path.join(model_info["checkpoint_dir"], "lora_adapter")
        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print(f"    [WARNING] No LoRA adapter found — using base model")
            model = base_model
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        ckpt = torch.load(model_info["checkpoint_path"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt
        gc.collect()
        model = model.to(device)

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free model from memory."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# EVAL EXAMPLES EVALUATION
# ============================================================

def load_eval_examples():
    """Load curated eval_examples.jsonl."""
    if not os.path.exists(EVAL_EXAMPLES_FILE):
        print(f"  [WARNING] {EVAL_EXAMPLES_FILE} not found — skipping eval examples.")
        return []
    examples = []
    with open(EVAL_EXAMPLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


@torch.no_grad()
def evaluate_on_examples(model, tokenizer, prefix, device, examples):
    """
    Run inference on curated eval examples.
    Returns a dict with per-example results and summary stats.
    """
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    examples.sort(key=lambda e: (difficulty_order.get(e["difficulty"], 99), e["id"]))

    results = []
    for ex in tqdm(examples, desc="    Eval examples", leave=False):
        input_text = prefix + ex["input"]
        inputs = tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        output_ids = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        changed = output.strip() != ex["input"].strip()
        is_no_pii = ex["category"] == "no_pii"

        results.append({
            "id": ex["id"],
            "difficulty": ex["difficulty"],
            "category": ex["category"],
            "notes": ex["notes"],
            "input": ex["input"],
            "output": output,
            "changed": changed,
            "is_no_pii": is_no_pii,
        })

    # Compute summary
    pii_examples = [r for r in results if not r["is_no_pii"]]
    no_pii_examples = [r for r in results if r["is_no_pii"]]

    pii_changed = sum(1 for r in pii_examples if r["changed"])
    no_pii_correct = sum(1 for r in no_pii_examples if not r["changed"])

    # Per-difficulty breakdown
    diff_stats = {}
    for r in results:
        d = r["difficulty"]
        if d not in diff_stats:
            diff_stats[d] = {"pii_total": 0, "pii_changed": 0,
                             "no_pii_total": 0, "no_pii_correct": 0}
        if r["is_no_pii"]:
            diff_stats[d]["no_pii_total"] += 1
            if not r["changed"]:
                diff_stats[d]["no_pii_correct"] += 1
        else:
            diff_stats[d]["pii_total"] += 1
            if r["changed"]:
                diff_stats[d]["pii_changed"] += 1

    summary = {
        "total_examples": len(results),
        "pii_anonymized": f"{pii_changed}/{len(pii_examples)}",
        "pii_anonymized_pct": round(100 * pii_changed / max(len(pii_examples), 1), 1),
        "no_pii_correct": f"{no_pii_correct}/{len(no_pii_examples)}" if no_pii_examples else "N/A",
        "no_pii_correct_pct": round(100 * no_pii_correct / max(len(no_pii_examples), 1), 1) if no_pii_examples else None,
        "per_difficulty": diff_stats,
    }

    return {"examples": results, "summary": summary}


# ============================================================
# TEST SET EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_on_test_set(model, tokenizer, prefix, device):
    """
    Evaluate on the full test split.
    Returns (loss, metrics_dict, sample_predictions).
    """
    test_path = os.path.join(DATA_SPLITS_DIR, "test.jsonl")
    if not os.path.exists(test_path):
        print(f"  [WARNING] {test_path} not found — skipping test set evaluation.")
        return None, None, None

    test_data = load_split_data(test_path)
    print(f"    Test set: {len(test_data)} samples")

    original_texts = [d["original_text"] for d in test_data]
    entity_texts = [d.get("entity_texts", []) for d in test_data]

    test_dataset = AnonymizationDataset(
        test_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix, augmentor=None,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for batch in tqdm(test_loader, desc="    Test set", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        total_loss += loss.item()
        n_batches += 1

        # Generate predictions
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_TARGET_LENGTH,
            num_beams=1,
            do_sample=False,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        label_ids = labels.clone()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        targets = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = total_loss / max(n_batches, 1)

    # Align original_texts / entity_texts to however many predictions we got
    orig_aligned = original_texts[:len(all_preds)]
    ent_aligned = entity_texts[:len(all_preds)]

    print(f"    Computing metrics on {len(all_preds)} predictions …")
    metrics = compute_all_metrics(
        preds=all_preds,
        targets=all_targets,
        original_texts=orig_aligned,
        entity_texts_list=ent_aligned,
        compute_bert=True,
    )

    # Serialise leaked_entities_top10 for JSON
    top10 = metrics.pop("leaked_entities_top10", [])
    metrics["leaked_entities_top10"] = [{"entity": e, "count": c} for e, c in top10]

    # Sample predictions
    samples = []
    for i in range(min(5, len(all_preds))):
        samples.append({
            "original": orig_aligned[i][:200],
            "prediction": all_preds[i][:200],
            "target": all_targets[i][:200],
        })

    return avg_loss, metrics, samples


# ============================================================
# FORMATTED REPORT WRITER
# ============================================================

def write_readable_report(all_results, filepath):
    """Write a nicely formatted human-readable .txt report."""
    lines = []
    w = lines.append  # shorthand

    w("=" * 110)
    w("  COMPREHENSIVE EVALUATION REPORT — SEQ2SEQ PII ANONYMIZATION")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 110)
    w("")

    n_models = len(all_results)
    w(f"  Models evaluated: {n_models}")
    w(f"  Evaluation order: smallest → largest (by parameter count)")
    w("")

    # ── Model overview table ──
    w("  MODEL OVERVIEW")
    w(f"  {'#':<4} {'Model':<25} {'HuggingFace ID':<35} {'Params':<12} {'Ckpt Size':<12} {'Type'}")
    w("  " + "─" * 100)
    for i, res in enumerate(all_results, 1):
        params_str = f"{res.get('params_total_millions', '?')}M"
        ckpt_str = f"{res.get('checkpoint_size_mb', '?')} MB"
        qlora_str = "QLoRA" if res.get("use_qlora") else "Full"
        w(f"  {i:<4} {res['model_key']:<25} {res['model_name']:<35} {params_str:<12} {ckpt_str:<12} {qlora_str}")
    w("")

    # ── Per-model sections ──
    for i, res in enumerate(all_results, 1):
        key = res["model_key"]
        w("─" * 110)
        w(f"  [{i}/{n_models}]  {key}  ({res['model_name']})")
        w("─" * 110)
        params_total = res.get('params_total_millions', '?')
        params_train = res.get('params_trainable_millions', '?')
        ckpt_mb = res.get('checkpoint_size_mb', '?')
        w(f"  Model size:    {params_total}M parameters total, {params_train}M trainable")
        w(f"  Checkpoint:    {ckpt_mb} MB on disk")
        w(f"  Type:          {'QLoRA (4-bit base + LoRA adapters)' if res.get('use_qlora') else 'Full fine-tuning'}")
        w(f"  Best val loss: {res.get('best_val_loss', 'N/A')}")
        w(f"  Eval time:     {res.get('eval_time', 'N/A')}")
        w("")

        # -- Test set metrics --
        test = res.get("test_set")
        if test:
            m = test["metrics"]
            w(f"  ┌─── TEST SET METRICS ───────────────────────────────────────┐")
            w(f"  │  Loss:                {test['loss']:<38.4f} │")
            w(f"  │  Exact Match:         {m['exact_match']:<38.2f} │")
            w(f"  │  Word Accuracy:       {m['word_accuracy']:<38.2f} │")
            w(f"  │                                                           │")
            w(f"  │  BLEU:                {m['bleu']:<38.2f} │")
            w(f"  │  BLEU-1:              {m['bleu1']:<38.2f} │")
            w(f"  │  BLEU-2:              {m['bleu2']:<38.2f} │")
            w(f"  │  BLEU-4:              {m['bleu4']:<38.2f} │")
            w(f"  │                                                           │")
            w(f"  │  ROUGE-1:             {m['rouge1']:<38.2f} │")
            w(f"  │  ROUGE-2:             {m['rouge2']:<38.2f} │")
            w(f"  │  ROUGE-L:             {m['rougeL']:<38.2f} │")
            w(f"  │                                                           │")
            w(f"  │  BERTScore P:         {m.get('bertscore_p', 0):<38.2f} │")
            w(f"  │  BERTScore R:         {m.get('bertscore_r', 0):<38.2f} │")
            w(f"  │  BERTScore F1:        {m.get('bertscore_f1', 0):<38.2f} │")
            w(f"  │                                                           │")
            w(f"  │  Leakage Rate:        {m.get('leakage_rate', 0):<38.2f} │")
            w(f"  │  Entity Leak Rate:    {m.get('entity_leakage_rate', 0):<38.2f} │")
            leak_str = f"{m.get('total_entities_leaked', 0)}/{m.get('total_entities_checked', 0)}"
            w(f"  │  Entities Leaked:     {leak_str:<38} │")
            w(f"  └───────────────────────────────────────────────────────────┘")
            w("")

            # Sample predictions
            samples = test.get("sample_predictions", [])
            if samples:
                w(f"  Sample predictions:")
                for j, s in enumerate(samples, 1):
                    w(f"    [{j}] ORIG: {s['original']}")
                    w(f"        PRED: {s['prediction']}")
                    w(f"        TRUE: {s['target']}")
                    w("")
        else:
            w(f"  Test set: SKIPPED (test.jsonl not found)")
            w("")

        # -- Eval examples --
        eval_ex = res.get("eval_examples")
        if eval_ex:
            summary = eval_ex["summary"]
            w(f"  ┌─── EVAL EXAMPLES ─────────────────────────────────────────┐")
            w(f"  │  Total examples:      {summary['total_examples']:<38} │")
            pii_str = f"{summary['pii_anonymized']} ({summary['pii_anonymized_pct']}%)"
            w(f"  │  PII anonymized:      {pii_str:<38} │")
            nopii_str = summary['no_pii_correct']
            if summary.get('no_pii_correct_pct') is not None:
                nopii_str += f" ({summary['no_pii_correct_pct']}%)"
            w(f"  │  No-PII correct:      {nopii_str:<38} │")
            w(f"  │                                                           │")

            for diff in ["easy", "medium", "hard"]:
                ds = summary["per_difficulty"].get(diff)
                if ds:
                    pii_s = f"{ds['pii_changed']}/{ds['pii_total']}" if ds["pii_total"] > 0 else "—"
                    nopii_s = f"{ds['no_pii_correct']}/{ds['no_pii_total']}" if ds["no_pii_total"] > 0 else "—"
                    w(f"  │  {diff.upper():<8}  PII: {pii_s:<12}  No-PII: {nopii_s:<14} │")

            w(f"  └───────────────────────────────────────────────────────────┘")
            w("")

            # Per-example details
            w(f"  Eval example details:")
            for ex in eval_ex["examples"]:
                if ex["is_no_pii"]:
                    status = "✓ CORRECT" if not ex["changed"] else "✗ FALSE POSITIVE"
                else:
                    status = "✓ CHANGED" if ex["changed"] else "— UNCHANGED"
                w(f"    [{ex['id']}] {ex['category']} — {status}")
                w(f"      IN:  {ex['input']}")
                w(f"      OUT: {ex['output']}")
                w("")
        else:
            w(f"  Eval examples: SKIPPED (eval_examples.jsonl not found)")
            w("")

    # ── Comparison table ──
    w("")
    w("=" * 110)
    w("  MODEL COMPARISON TABLE  (sorted smallest → largest)")
    w("=" * 110)
    w("")

    # Header
    h = f"  {'Model':<25} {'Size':>8}"
    for col in ["Loss", "Exact%", "WordAcc%", "BLEU", "ROUGE-L", "BERTScF1", "Leak%"]:
        h += f" {col:>10}"
    w(h)
    w("  " + "─" * 108)

    for res in all_results:
        size_str = f"{res.get('params_total_millions', '?')}M"
        test = res.get("test_set")
        if test:
            m = test["metrics"]
            row = f"  {res['model_key']:<25} {size_str:>8}"
            row += f" {test['loss']:>10.4f}"
            row += f" {m['exact_match']:>10.2f}"
            row += f" {m['word_accuracy']:>10.2f}"
            row += f" {m['bleu']:>10.2f}"
            row += f" {m['rougeL']:>10.2f}"
            row += f" {m.get('bertscore_f1', 0):>10.2f}"
            row += f" {m.get('entity_leakage_rate', 0):>10.2f}"
        else:
            row = f"  {res['model_key']:<25} {size_str:>8}" + "        N/A" * 7
        w(row)

    w("")

    # Eval examples comparison
    w(f"  {'Model':<25} {'Size':>8} {'PII Anon%':>10} {'NoPII Corr%':>12}")
    w("  " + "─" * 57)
    for res in all_results:
        size_str = f"{res.get('params_total_millions', '?')}M"
        eval_ex = res.get("eval_examples")
        if eval_ex:
            s = eval_ex["summary"]
            pii_pct = f"{s['pii_anonymized_pct']:.1f}"
            nopii_pct = f"{s['no_pii_correct_pct']:.1f}" if s.get("no_pii_correct_pct") is not None else "N/A"
        else:
            pii_pct = nopii_pct = "N/A"
        w(f"  {res['model_key']:<25} {size_str:>8} {pii_pct:>10} {nopii_pct:>12}")

    w("")
    w("=" * 110)
    w(f"  Report saved to: {filepath}")
    w("=" * 110)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  SEQ2SEQ PII ANONYMIZATION — COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  Device: CPU")

    # Discover models (sorted smallest → largest)
    trained_models = discover_trained_models()
    if not trained_models:
        print("\n  [ERROR] No trained models found in checkpoints/")
        print(f"  Looked in: {CHECKPOINTS_DIR}")
        print("  Train a model first with: python train.py")
        sys.exit(1)

    print(f"\n  Found {len(trained_models)} trained model(s) (smallest → largest):")
    for info in trained_models:
        loss_s = f"{info['best_val_loss']:.4f}" if info.get("best_val_loss") else "N/A"
        qlora = " [QLoRA]" if info["use_qlora"] else ""
        size = format_params(info["approx_size_millions"])
        ckpt_s = f"{info['checkpoint_size_mb']} MB" if info['checkpoint_size_mb'] else "?"
        print(f"    • {info['model_key']:<25} ~{size:>5}  ckpt={ckpt_s:<10}  val_loss={loss_s}{qlora}")

    # Load eval examples once
    eval_examples = load_eval_examples()
    if eval_examples:
        print(f"  Eval examples loaded: {len(eval_examples)}")
    else:
        print(f"  Eval examples: not found, will skip")

    # Evaluate each model (smallest first)
    all_results = []
    total_start = time.time()

    for idx, model_info in enumerate(trained_models, 1):
        model_key = model_info["model_key"]
        approx_size = format_params(model_info["approx_size_millions"])
        print(f"\n{'─' * 70}")
        print(f"  [{idx}/{len(trained_models)}] Evaluating: {model_key}  (~{approx_size} params)")
        print(f"{'─' * 70}")

        model_start = time.time()

        result = {
            "model_key": model_key,
            "model_name": model_info["model_name"],
            "prefix": model_info["prefix"],
            "use_qlora": model_info["use_qlora"],
            "best_val_loss": model_info.get("best_val_loss"),
            "approx_size_millions": model_info["approx_size_millions"],
            "checkpoint_size_mb": model_info["checkpoint_size_mb"],
        }

        # Load model
        print(f"  Loading model …")
        model, tokenizer = load_model(model_info, device)
        params = count_parameters(model)
        result["params_total_millions"] = params["total_millions"]
        result["params_trainable_millions"] = params["trainable_millions"]
        print(f"  Size: {params['total_millions']}M params total, "
              f"{params['trainable_millions']}M trainable, "
              f"checkpoint {model_info['checkpoint_size_mb'] or '?'} MB on disk")

        # ── 1. Test set evaluation ──
        print(f"  Evaluating on test set …")
        test_loss, test_metrics, test_samples = evaluate_on_test_set(
            model, tokenizer, model_info["prefix"], device,
        )
        if test_metrics is not None:
            result["test_set"] = {
                "loss": test_loss,
                "metrics": test_metrics,
                "sample_predictions": test_samples,
            }
            print(f"    Loss: {test_loss:.4f}  |  BLEU: {test_metrics['bleu']:.2f}  |  "
                  f"ROUGE-L: {test_metrics['rougeL']:.2f}  |  "
                  f"BERTScore F1: {test_metrics.get('bertscore_f1', 0):.2f}  |  "
                  f"Leak: {test_metrics.get('entity_leakage_rate', 0):.2f}%")
        else:
            result["test_set"] = None

        # ── 2. Eval examples ──
        if eval_examples:
            print(f"  Evaluating on curated examples …")
            eval_result = evaluate_on_examples(
                model, tokenizer, model_info["prefix"], device, eval_examples,
            )
            result["eval_examples"] = eval_result
            s = eval_result["summary"]
            print(f"    PII anonymized: {s['pii_anonymized']} ({s['pii_anonymized_pct']}%)  |  "
                  f"No-PII correct: {s['no_pii_correct']}")
        else:
            result["eval_examples"] = None

        model_time = time.time() - model_start
        result["eval_time"] = format_time(model_time)
        print(f"  Done in {format_time(model_time)}")

        all_results.append(result)

        # Unload model before next one
        unload_model(model, tokenizer)

    total_time = time.time() - total_start

    # ── Save JSON results ──
    print(f"\n{'=' * 70}")
    print(f"  Saving results …")

    json_output = {
        "timestamp": datetime.now().isoformat(),
        "total_eval_time": format_time(total_time),
        "device": str(device),
        "models_evaluated": len(all_results),
        "evaluation_order": "smallest_to_largest",
        "results": all_results,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {RESULTS_JSON}")

    # ── Save readable report ──
    report = write_readable_report(all_results, RESULTS_TXT)
    print(f"  TXT:  {RESULTS_TXT}")

    # ── Print comparison table to console ──
    print("\n")
    # Extract just the comparison table from the report
    lines = report.split("\n")
    in_table = False
    for line in lines:
        if "MODEL COMPARISON TABLE" in line:
            in_table = True
        if in_table:
            print(line)

    print(f"\n  Total evaluation time: {format_time(total_time)}")
    print(f"  Done!\n")


if __name__ == "__main__":
    main()
