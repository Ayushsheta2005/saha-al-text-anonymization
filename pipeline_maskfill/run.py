# ==============================================================================
# run.py — CLI Entry Point for the Two-Level PII Pipeline
# ==============================================================================
# Usage:
#   python run.py train-encoder --model distilroberta [--quick]
#   python run.py train-encoder --model all [--quick]
#   python run.py train-filler  --model bart-base [--quick]
#   python run.py train-filler  --model all [--quick]
#   python run.py evaluate --encoder distilroberta --filler bart-base [--quick]
#   python run.py evaluate --all [--quick]
#   python run.py prepare-data [--quick] [--verify]
# ==============================================================================

import os
import sys
import argparse
import logging
import json
from datetime import datetime

import torch
import numpy as np
import random

# ── Setup path ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    SEED, DEVICE, OUTPUT_DIR, LOG_DIR, EVAL_DIR,
    BF16_OK, FP16_OK,
    ENCODER_REGISTRY, FILLER_REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(name: str = "pipeline") -> logging.Logger:
    """
    Set up dual logging: console + timestamped file.
    All training/eval output is captured in both.
    """
    log = logging.getLogger("pipeline")
    log.setLevel(logging.INFO)

    # Prevent duplicate handlers on re-runs
    if log.handlers:
        return log

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    # File handler — timestamped
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(file_handler)

    log.info(f"Logging to: {log_file}")
    return log


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_config(log, args):
    """Log the full run configuration."""
    log.info("=" * 70)
    log.info("  TWO-LEVEL PII ANONYMIZATION PIPELINE")
    log.info("=" * 70)
    log.info(f"  Command:    {' '.join(sys.argv)}")
    log.info(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Device:     {DEVICE}")
    if torch.cuda.is_available():
        log.info(f"  GPU:        {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    log.info(f"  BF16:       {BF16_OK}")
    log.info(f"  FP16:       {FP16_OK}")
    log.info(f"  Seed:       {SEED}")
    log.info(f"  Quick mode: {args.quick}")
    log.info(f"  Output dir: {OUTPUT_DIR}")
    log.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_prepare_data(args):
    """Prepare and verify the dataset splits."""
    log = setup_logging("prepare_data")
    set_seed()
    log_config(log, args)

    from data import prepare_all_data

    splits = prepare_all_data(quick=args.quick)

    if args.verify:
        log.info("\n  ── Verification ──")
        for name, ds in splits.items():
            log.info(f"  {name}: {len(ds):,} examples, columns: {ds.column_names}")

        # Check disjointness of Half-A and Half-B
        log.info("\n  Checking Half-A / Half-B disjointness ...")
        a_ids = set(range(len(splits["half_a"])))
        b_ids = set(range(len(splits["half_b"])))
        log.info(f"  Half-A: {len(a_ids):,} examples")
        log.info(f"  Half-B: {len(b_ids):,} examples")
        log.info(f"  ✓ Splits prepared and verified")


def cmd_train_encoder(args):
    """Train one or all NER encoder models."""
    log = setup_logging(f"encoder_{args.model}")
    set_seed()
    log_config(log, args)

    from data import prepare_all_data
    from encoder import train_encoder, cleanup_gpu

    splits = prepare_all_data(quick=args.quick)

    models_to_train = (
        list(ENCODER_REGISTRY.keys()) if args.model == "all"
        else [args.model]
    )

    for model_name in models_to_train:
        if model_name not in ENCODER_REGISTRY:
            log.error(f"Unknown encoder: {model_name}. Available: {list(ENCODER_REGISTRY.keys())}")
            continue

        log.info(f"\n{'═' * 70}")
        log.info(f"  TRAINING ENCODER: {model_name}")
        log.info(f"{'═' * 70}")

        model, tokenizer, eval_results = train_encoder(
            model_name=model_name,
            train_ds=splits["half_a"],
            val_ds=splits["val_encoder"],
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.hub_model_id else f"ner-{model_name}",
            hub_token=args.hub_token,
        )

        if eval_results:
            log.info(f"\n  Encoder {model_name} — Final NER F1: {eval_results.get('eval_f1', 0):.4f}")

        cleanup_gpu()
        log.info(f"  ✓ Encoder {model_name} complete")


def cmd_train_filler(args):
    """Train one or all filler models."""
    log = setup_logging(f"filler_{args.model}")
    set_seed()
    log_config(log, args)

    from data import prepare_all_data
    from filler import train_filler

    splits = prepare_all_data(quick=args.quick)

    models_to_train = (
        list(FILLER_REGISTRY.keys()) if args.model == "all"
        else [args.model]
    )

    for model_name in models_to_train:
        if model_name not in FILLER_REGISTRY:
            log.error(f"Unknown filler: {model_name}. Available: {list(FILLER_REGISTRY.keys())}")
            continue

        log.info(f"\n{'═' * 70}")
        log.info(f"  TRAINING FILLER: {model_name}")
        log.info(f"{'═' * 70}")

        model, tokenizer, eval_results = train_filler(
            model_name=model_name,
            train_ds=splits["half_b"],
            val_ds=splits["val_filler"],
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.hub_model_id else f"filler-{model_name}",
            hub_token=args.hub_token,
        )

        if eval_results:
            log.info(f"\n  Filler {model_name} — Final eval loss: "
                     f"{eval_results.get('eval_loss', 'N/A')}")

        log.info(f"  ✓ Filler {model_name} complete")


def cmd_evaluate(args):
    """Evaluate a pipeline combo (encoder + filler) or all combos."""
    log = setup_logging(f"eval_{args.encoder}_{args.filler}" if not args.all else "eval_all")
    set_seed()
    log_config(log, args)

    from data import prepare_all_data, get_source_text
    from encoder import train_encoder, cleanup_gpu
    from filler import train_filler
    from pipeline import anonymize, batch_anonymize
    from evaluate import evaluate_pipeline

    splits = prepare_all_data(quick=args.quick)

    # Determine combos
    if args.all:
        combos = [
            (enc, fill)
            for enc in ENCODER_REGISTRY
            for fill in FILLER_REGISTRY
        ]
    else:
        combos = [(args.encoder, args.filler)]

    all_results = []

    for enc_name, fill_name in combos:
        log.info(f"\n{'═' * 70}")
        log.info(f"  EVALUATING: {enc_name} + {fill_name}")
        log.info(f"{'═' * 70}")

        # Load encoder
        enc_model, enc_tok, _ = train_encoder(
            enc_name, splits["half_a"], splits["val_encoder"])
        # Load filler
        fill_model, fill_tok, _ = train_filler(
            fill_name, splits["half_b"], splits["val_filler"])
        fill_cfg = FILLER_REGISTRY[fill_name]

        # Prepare test data
        n_test = min(len(splits["test"]), 200 if args.quick else len(splits["test"]))
        originals = [get_source_text(splits["test"][i]) for i in range(n_test)]

        # Anonymize
        log.info(f"\n  Running pipeline on {n_test} test examples ...")
        anonymized, masked_texts, all_entities = batch_anonymize(
            originals, enc_model, enc_tok, fill_model, fill_tok, fill_cfg,
            desc=f"Pipeline ({enc_name}+{fill_name})",
        )

        # Define anonymize function for curated eval
        anon_fn = lambda text: anonymize(
            text, enc_model, enc_tok, fill_model, fill_tok, fill_cfg)

        # Full evaluation
        results = evaluate_pipeline(
            originals, anonymized, anon_fn, enc_name, fill_name)
        all_results.append(results)

        cleanup_gpu()

    # If multiple combos, print comparison table
    if len(all_results) > 1:
        _print_comparison_table(log, all_results)

    log.info("\n  ✓ Evaluation complete")


def _print_comparison_table(log, results_list):
    """Print a side-by-side comparison table for all pipeline combos."""
    log.info(f"\n{'═' * 90}")
    log.info(f"  PIPELINE COMPARISON TABLE")
    log.info(f"{'═' * 90}")

    header = f"  {'Pipeline':<25} {'Leak%':>8} {'CRR%':>8} {'BLEU':>8} {'ROUGE-L':>8} {'BERTScr':>8}"
    log.info(header)
    log.info(f"  {'─' * 67}")

    for r in results_list:
        name = r["pipeline"]
        leak = r["leakage"]["entity_leak_rate"]
        crr = r["crr"]["crr"]
        bleu = r.get("bleu", 0)
        rouge_l = r.get("rouge", {}).get("rougeL", 0)
        bert = r.get("bertscore_f1", 0)
        log.info(f"  {name:<25} {leak:>7.2f}% {crr:>7.2f}% {bleu:>8.2f} {rouge_l:>8.4f} {bert:>8.4f}")

    log.info(f"{'═' * 90}")

    # Save comparison
    comparison_path = os.path.join(EVAL_DIR, "comparison_table.json")
    with open(comparison_path, "w") as f:
        json.dump(results_list, f, indent=2, default=str)
    log.info(f"  Comparison saved: {comparison_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Parser
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-Level PII Anonymization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py prepare-data --verify
  python run.py train-encoder --model distilroberta --quick
  python run.py train-filler --model bart-base --quick
  python run.py evaluate --encoder distilroberta --filler bart-base --quick
  python run.py evaluate --all
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── prepare-data ──
    p_data = subparsers.add_parser("prepare-data",
        help="Load and split the AI4Privacy dataset")
    p_data.add_argument("--quick", action="store_true",
        help="Use small subset for debugging")
    p_data.add_argument("--verify", action="store_true",
        help="Print verification info about splits")

    # ── train-encoder ──
    p_enc = subparsers.add_parser("train-encoder",
        help="Train a NER encoder model")
    p_enc.add_argument("--model", required=True,
        choices=list(ENCODER_REGISTRY.keys()) + ["all"],
        help="Encoder model name (or 'all')")
    p_enc.add_argument("--quick", action="store_true",
        help="Quick mode: small data subset")
    p_enc.add_argument("--push_to_hub", action="store_true",
        help="Push model to huggingface hub after training")
    p_enc.add_argument("--hub_model_id", type=str,
        help="Huggingface Model ID (e.g. username/my-ner-model)")
    p_enc.add_argument("--hub_token", type=str,
        help="HuggingFace API token")

    # ── train-filler ──
    p_fill = subparsers.add_parser("train-filler",
        help="Train a filler encoder-decoder/MLM model")
    p_fill.add_argument("--model", required=True,
        choices=list(FILLER_REGISTRY.keys()) + ["all"],
        help="Filler model name (or 'all')")
    p_fill.add_argument("--quick", action="store_true",
        help="Quick mode: small data subset")
    p_fill.add_argument("--push_to_hub", action="store_true",
        help="Push model to huggingface hub after training")
    p_fill.add_argument("--hub_model_id", type=str,
        help="Huggingface Model ID (e.g. username/my-filler-model)")
    p_fill.add_argument("--hub_token", type=str,
        help="HuggingFace API token")

    # ── evaluate ──
    p_eval = subparsers.add_parser("evaluate",
        help="Evaluate a pipeline combination")
    p_eval.add_argument("--encoder", default="distilroberta",
        choices=list(ENCODER_REGISTRY.keys()),
        help="Encoder model to use")
    p_eval.add_argument("--filler", default="bart-base",
        choices=list(FILLER_REGISTRY.keys()),
        help="Filler model to use")
    p_eval.add_argument("--all", action="store_true",
        help="Evaluate all trained combinations")
    p_eval.add_argument("--quick", action="store_true",
        help="Quick mode: fewer test examples")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch
    commands = {
        "prepare-data": cmd_prepare_data,
        "train-encoder": cmd_train_encoder,
        "train-filler": cmd_train_filler,
        "evaluate": cmd_evaluate,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
