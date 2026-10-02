"""
Interactive Inference Script for Seq2Seq PII Anonymization
==========================================================
- Scans the checkpoints/ folder for trained models
- Shows model info (best val loss, metrics, etc.)
- Lets you pick a model and type input sentences
- Displays the anonymized output
- Batch evaluation on curated examples (eval_examples.jsonl)
"""

import os
import sys
import json
import gc
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from config import MODEL_CONFIGS, CHECKPOINTS_DIR, MAX_TARGET_LENGTH

# Path to the curated evaluation examples
EVAL_EXAMPLES_FILE = os.path.join(os.path.dirname(__file__), "eval_examples.jsonl")


# ============================================================
# DISCOVER TRAINED MODELS
# ============================================================

def discover_trained_models() -> list[dict]:
    """
    Scan checkpoints/ for models that have a best_model.pt file.
    Returns a list of dicts with model info.
    """
    trained = []

    if not os.path.exists(CHECKPOINTS_DIR):
        return trained

    for model_key in sorted(os.listdir(CHECKPOINTS_DIR)):
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, model_key)
        best_path = os.path.join(checkpoint_dir, "best_model.pt")

        if not os.path.isdir(checkpoint_dir) or not os.path.exists(best_path):
            continue

        # Must be a known model from config
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
            "best_val_loss": None,
            "final_metrics": None,
        }

        # Load training history if available
        hist_path = os.path.join(checkpoint_dir, "training_history.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    hist = json.load(f)
                info["best_val_loss"] = hist.get("best_val_loss")
                info["final_metrics"] = hist.get("final_metrics")
            except Exception:
                pass

        # Fallback: read val loss from checkpoint metadata
        if info["best_val_loss"] is None:
            try:
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                info["best_val_loss"] = ckpt.get("best_val_loss")
                del ckpt
                gc.collect()
            except Exception:
                pass

        trained.append(info)

    return trained


# ============================================================
# LOAD MODEL FOR INFERENCE
# ============================================================

def load_model_for_inference(model_info: dict, device: torch.device):
    """Load a trained model and tokenizer from checkpoint."""
    model_key = model_info["model_key"]
    config = MODEL_CONFIGS[model_key]
    model_name = config["model_name"]
    use_qlora = config.get("use_qlora", False)

    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        print(f"  Loading QLoRA model: {model_name}")
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
            print(f"  LoRA adapter loaded from {adapter_path}")
        else:
            print(f"  [WARNING] No LoRA adapter found. Using base model.")
            model = base_model
    else:
        print(f"  Loading model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )

        # Load trained weights
        print(f"  Loading trained weights from checkpoint...")
        ckpt = torch.load(
            model_info["checkpoint_path"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt
        gc.collect()

        model = model.to(device)

    model.eval()
    print(f"  ✓ Model ready for inference!")
    return model, tokenizer


# ============================================================
# GENERATE PREDICTION
# ============================================================

@torch.no_grad()
def anonymize(
    model,
    tokenizer,
    text: str,
    prefix: str,
    device: torch.device,
    num_beams: int = 4,
    max_length: int = MAX_TARGET_LENGTH,
) -> str:
    """Run inference on a single input text."""
    input_text = prefix + text
    inputs = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,
    )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return prediction


# ============================================================
# BATCH EVALUATION ON CURATED EXAMPLES
# ============================================================

def load_eval_examples(filepath: str = EVAL_EXAMPLES_FILE) -> list[dict]:
    """Load evaluation examples from the JSONL file."""
    if not os.path.exists(filepath):
        print(f"  [ERROR] Eval examples file not found: {filepath}")
        return []
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_batch_evaluation(
    model,
    tokenizer,
    prefix: str,
    device: torch.device,
    model_key: str,
    num_beams: int = 4,
    difficulty_filter: str = None,
):
    """
    Run the model on all curated eval examples and display a formatted report.

    Args:
        difficulty_filter: if set, only run examples of that difficulty
                           ("easy", "medium", "hard"). None = run all.
    """
    examples = load_eval_examples()
    if not examples:
        return

    # Apply difficulty filter
    if difficulty_filter:
        examples = [e for e in examples if e["difficulty"] == difficulty_filter]
        if not examples:
            print(f"  No examples found for difficulty '{difficulty_filter}'.")
            return

    # Group by difficulty for organized output
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    examples.sort(key=lambda e: (difficulty_order.get(e["difficulty"], 99), e["id"]))

    total = len(examples)
    print(f"\n{'=' * 100}")
    print(f"  BATCH EVALUATION — {model_key}")
    print(f"  Examples: {total}  |  Beam width: {num_beams}")
    if difficulty_filter:
        print(f"  Filter: {difficulty_filter.upper()} only")
    print(f"{'=' * 100}")

    results = []
    current_difficulty = None
    passed = 0    # output differs from input (PII was changed)
    unchanged = 0 # output identical to input

    start_time = time.time()

    for i, example in enumerate(examples, 1):
        # Print difficulty header when it changes
        if example["difficulty"] != current_difficulty:
            current_difficulty = example["difficulty"]
            label = current_difficulty.upper()
            emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(current_difficulty, "⚪")
            print(f"\n  {'─' * 96}")
            print(f"  {emoji} {label} EXAMPLES")
            print(f"  {'─' * 96}")

        # Run inference
        output = anonymize(
            model, tokenizer, example["input"], prefix, device, num_beams=num_beams,
        )

        changed = output.strip() != example["input"].strip()
        if changed:
            passed += 1
        else:
            unchanged += 1

        status = "✓ CHANGED" if changed else "— UNCHANGED"
        is_no_pii = example["category"] == "no_pii"

        # For no-PII examples, unchanged is the CORRECT behavior
        if is_no_pii:
            status = "✓ CORRECT (no PII)" if not changed else "✗ FALSE POSITIVE"

        # Print result
        print(f"\n  [{example['id']}] {example['category']} — {status}")
        print(f"    Notes:  {example['notes']}")
        print(f"    INPUT:  {example['input']}")
        print(f"    OUTPUT: {output}")

        results.append({
            "id": example["id"],
            "difficulty": example["difficulty"],
            "category": example["category"],
            "input": example["input"],
            "output": output,
            "changed": changed,
            "is_no_pii": is_no_pii,
            "notes": example["notes"],
        })

    elapsed = time.time() - start_time

    # ---- Summary ----
    # Count no-PII examples separately
    no_pii_examples = [r for r in results if r["is_no_pii"]]
    pii_examples = [r for r in results if not r["is_no_pii"]]

    pii_changed = sum(1 for r in pii_examples if r["changed"])
    pii_total = len(pii_examples)
    no_pii_correct = sum(1 for r in no_pii_examples if not r["changed"])
    no_pii_total = len(no_pii_examples)

    # Per-difficulty breakdown
    diff_stats = {}
    for r in results:
        d = r["difficulty"]
        if d not in diff_stats:
            diff_stats[d] = {"total": 0, "changed": 0, "no_pii_total": 0, "no_pii_correct": 0}
        diff_stats[d]["total"] += 1
        if r["is_no_pii"]:
            diff_stats[d]["no_pii_total"] += 1
            if not r["changed"]:
                diff_stats[d]["no_pii_correct"] += 1
        elif r["changed"]:
            diff_stats[d]["changed"] += 1

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY — {model_key}")
    print(f"{'=' * 100}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total:.2f}s per example)")
    print()
    print(f"  PII examples:    {pii_changed}/{pii_total} anonymized "
          f"({100*pii_changed/max(pii_total,1):.0f}%)")
    if no_pii_total > 0:
        print(f"  No-PII examples: {no_pii_correct}/{no_pii_total} correctly unchanged "
              f"({100*no_pii_correct/max(no_pii_total,1):.0f}%)")
    print()
    print(f"  {'Difficulty':<12} {'Anonymized':<20} {'No-PII Correct':<20}")
    print(f"  {'─'*12} {'─'*20} {'─'*20}")
    for d in ["easy", "medium", "hard"]:
        if d in diff_stats:
            s = diff_stats[d]
            pii_in_d = s["total"] - s["no_pii_total"]
            anon_str = f"{s['changed']}/{pii_in_d}" if pii_in_d > 0 else "—"
            nopii_str = f"{s['no_pii_correct']}/{s['no_pii_total']}" if s["no_pii_total"] > 0 else "—"
            print(f"  {d:<12} {anon_str:<20} {nopii_str:<20}")
    print(f"{'=' * 100}")

    # ---- Save results to file ----
    output_dir = os.path.join(CHECKPOINTS_DIR, model_key)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "eval_examples_results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Results saved to: {output_path}")

    return results


# ============================================================
# INTERACTIVE UI
# ============================================================

def select_model(trained_models: list[dict]) -> dict:
    """Display trained models and let the user pick one."""
    print("\n" + "=" * 80)
    print("  TRAINED MODELS")
    print("=" * 80)
    print(f"  {'#':<4} {'Model Key':<25} {'HuggingFace ID':<35} {'Best Val Loss'}")
    print("  " + "-" * 78)

    for i, info in enumerate(trained_models, 1):
        loss_str = f"{info['best_val_loss']:.4f}" if info["best_val_loss"] else "N/A"
        qlora_tag = " [QLoRA]" if info["use_qlora"] else ""
        model_id = info["model_name"] + qlora_tag
        print(f"  {i:<4} {info['model_key']:<25} {model_id:<35} {loss_str}")

    # Show extra metrics if available
    print()
    for i, info in enumerate(trained_models, 1):
        metrics = info.get("final_metrics")
        if metrics:
            print(f"  [{i}] {info['model_key']}:")
            print(f"      Word Acc: {metrics.get('word_accuracy', 'N/A')}%  |  "
                  f"BLEU: {metrics.get('bleu', 'N/A')}  |  "
                  f"ROUGE-L: {metrics.get('rougeL', 'N/A')}  |  "
                  f"BERTScore F1: {metrics.get('bertscore_f1', 'N/A')}  |  "
                  f"Leakage: {metrics.get('entity_leakage_rate', 'N/A')}%")

    print("  " + "-" * 78)

    while True:
        try:
            user_input = input(f"\n  Select a model (1-{len(trained_models)}) or 'q' to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

        if user_input.lower() == 'q':
            print("  Exiting.")
            sys.exit(0)

        try:
            idx = int(user_input)
            if 1 <= idx <= len(trained_models):
                return trained_models[idx - 1]
            else:
                print(f"  Invalid number. Enter 1-{len(trained_models)}.")
        except ValueError:
            print(f"  Invalid input. Enter a number or 'q'.")


def interactive_loop(model, tokenizer, prefix: str, device: torch.device, model_key: str):
    """Main loop: user types input, sees anonymized output."""
    print("\n" + "=" * 80)
    print(f"  INFERENCE MODE — {model_key}")
    print("=" * 80)
    print("  Type a sentence containing PII to anonymize it.")
    print("  Commands:")
    print("    'q'             — quit")
    print("    'switch'        — change model")
    print("    'beam N'        — set beam width (e.g., beam 4)")
    print("    'eval'          — run all curated eval examples")
    print("    'eval easy'     — run only easy examples")
    print("    'eval medium'   — run only medium examples")
    print("    'eval hard'     — run only hard examples")
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

        if user_input.lower() == 'q':
            return "quit"

        if user_input.lower() == 'switch':
            return "switch"

        if user_input.lower().startswith('beam'):
            try:
                num_beams = int(user_input.split()[1])
                print(f"  Beam width set to {num_beams}")
            except (IndexError, ValueError):
                print(f"  Usage: beam N (e.g., beam 4)")
            continue

        if user_input.lower().startswith('eval'):
            parts = user_input.lower().split()
            difficulty_filter = None
            if len(parts) > 1 and parts[1] in ("easy", "medium", "hard"):
                difficulty_filter = parts[1]
            run_batch_evaluation(
                model, tokenizer, prefix, device, model_key,
                num_beams=num_beams, difficulty_filter=difficulty_filter,
            )
            continue

        # Run inference
        output = anonymize(model, tokenizer, user_input, prefix, device, num_beams=num_beams)

        print(f"  OUTPUT> {output}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  SEQ2SEQ PII ANONYMIZATION — INTERACTIVE INFERENCE")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  Device: CPU")

    # Find trained models
    trained_models = discover_trained_models()
    if not trained_models:
        print("\n  [ERROR] No trained models found in checkpoints/")
        print(f"  Looked in: {CHECKPOINTS_DIR}")
        print("  Train a model first with: python train.py")
        sys.exit(1)

    print(f"\n  Found {len(trained_models)} trained model(s).")

    # Main loop: select model → inference → optionally switch
    loaded_model = None
    loaded_tokenizer = None

    while True:
        selected = select_model(trained_models)

        # Cleanup previous model
        if loaded_model is not None:
            del loaded_model, loaded_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load selected model
        loaded_model, loaded_tokenizer = load_model_for_inference(selected, device)

        # Run interactive loop
        result = interactive_loop(
            loaded_model, loaded_tokenizer,
            selected["prefix"], device, selected["model_key"],
        )

        if result == "quit":
            break
        # result == "switch" → loop back to model selection

    # Cleanup
    if loaded_model is not None:
        del loaded_model, loaded_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n  Goodbye!")


if __name__ == "__main__":
    main()
