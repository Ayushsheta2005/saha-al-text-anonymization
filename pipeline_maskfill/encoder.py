# ==============================================================================
# encoder.py — NER Encoder: training, inference, and evaluation
# ==============================================================================
# Trains any registered encoder model (DistilRoBERTa, RoBERTa, DeBERTa-v3)
# for BIO token classification to detect and mask PII entities.
#
# Features:
#   - Sample inferences logged after each epoch
#   - Per-entity-type NER metrics
#   - Checkpoint save/resume
#   - DeBERTa fp32 parameter fix for AMP compatibility
# ==============================================================================

import os
import gc
import logging
import random
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    f1_score as seq_f1_score,
    classification_report as seq_classification_report,
)

from config import (
    SEED, DEVICE, OUTPUT_DIR, LOG_DIR,
    BF16_OK, FP16_OK,
    ENCODER_REGISTRY, NUM_LABELS, LABEL2ID, ID2LABEL,
    LOG_EVERY_N_STEPS, SAMPLE_INFERENCE_COUNT,
)
from data import (
    extract_tokens_and_labels, tokenize_and_align_ner, get_source_text,
)

log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# DeBERTa Fix
# ═══════════════════════════════════════════════════════════════════════════════

def fix_deberta_params(model):
    """
    DeBERTa-v3 uses non-standard LayerNorm parameter names (gamma/beta instead
    of weight/bias). This causes AMP fp16 instability. Cast them to fp32.
    """
    for name, param in model.named_parameters():
        if "LayerNorm" in name or "layernorm" in name:
            param.data = param.data.to(torch.float32)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Build Encoder Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_encoder(model_name: str) -> Tuple:
    """
    Build an encoder model + tokenizer from the registry.

    Args:
        model_name: Key from ENCODER_REGISTRY (e.g., 'distilroberta', 'roberta', 'deberta')

    Returns:
        (model, tokenizer, config_dict)
    """
    if model_name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {model_name}. "
                         f"Available: {list(ENCODER_REGISTRY.keys())}")

    cfg = ENCODER_REGISTRY[model_name]
    hf_name = cfg["hf_name"]

    log.info(f"Building encoder: {model_name} ({hf_name})")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        hf_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )

    if cfg.get("needs_deberta_fix"):
        log.info("  Applying DeBERTa fp32 LayerNorm fix")
        fix_deberta_params(model)

    model = model.float().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f"  Parameters: {total_params:.1f}M total, {train_params:.1f}M trainable")

    return model, tokenizer, cfg


# ═══════════════════════════════════════════════════════════════════════════════
# NER Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ner_metrics(eval_preds):
    """Compute entity-level F1 during training (seqeval)."""
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        true_sent, pred_sent = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_sent.append(ID2LABEL.get(int(l), "O"))
            pred_sent.append(ID2LABEL.get(int(p), "O"))
        true_labels.append(true_sent)
        pred_labels.append(pred_sent)

    f1 = seq_f1_score(true_labels, pred_labels, average="weighted")
    return {"f1": f1}


def compute_ner_metrics_detailed(eval_preds):
    """Extended NER metrics with per-entity-type breakdown."""
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        true_sent, pred_sent = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_sent.append(ID2LABEL.get(int(l), "O"))
            pred_sent.append(ID2LABEL.get(int(p), "O"))
        true_labels.append(true_sent)
        pred_labels.append(pred_sent)

    f1 = seq_f1_score(true_labels, pred_labels, average="weighted")
    report = seq_classification_report(true_labels, pred_labels,
                                        output_dict=True, zero_division=0)

    # Pretty-print the report
    log.info("  ┌─── Per-Entity NER Performance ───────────────────────┐")
    log.info(f"  │  {'Entity':<20} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Sup':>7} │")
    log.info(f"  │  {'─' * 48} │")
    for etype in sorted(report.keys()):
        if etype in ("micro avg", "macro avg", "weighted avg", "samples avg"):
            continue
        d = report[etype]
        log.info(f"  │  {etype:<20} {d['precision']:>7.3f} {d['recall']:>7.3f} "
                 f"{d['f1-score']:>7.3f} {d['support']:>7.0f} │")
    for avg in ("micro avg", "macro avg", "weighted avg"):
        if avg in report:
            d = report[avg]
            log.info(f"  │  {avg:<20} {d['precision']:>7.3f} {d['recall']:>7.3f} "
                     f"{d['f1-score']:>7.3f} {d['support']:>7.0f} │")
    log.info("  └─────────────────────────────────────────────────────┘")

    return {"f1": f1, "per_entity": report}


# ═══════════════════════════════════════════════════════════════════════════════
# Sample Inference Callback
# ═══════════════════════════════════════════════════════════════════════════════

class NERSampleInferenceCallback(TrainerCallback):
    """
    After each epoch, run inference on a few samples and log the results.
    Shows: input text → gold entities vs predicted entities, with ✓/✗ markers.
    """
    def __init__(self, sample_examples, tokenizer, n_samples=SAMPLE_INFERENCE_COUNT):
        self.samples = sample_examples[:n_samples]
        self.tokenizer = tokenizer
        self.n_samples = n_samples

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        epoch = int(state.epoch)
        log.info(f"\n  ═══ Epoch {epoch} — Sample NER Inferences ═══")

        for i, example in enumerate(self.samples):
            tokens, gold_labels = extract_tokens_and_labels(example)
            text = " ".join(tokens)

            # Run inference
            enc = self.tokenizer(
                tokens, return_tensors="pt", truncation=True,
                max_length=256, is_split_into_words=True, padding=True,
            ).to(DEVICE)

            with torch.no_grad():
                logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            word_ids = enc.word_ids()

            # Reconstruct word-level predictions
            pred_labels = []
            prev_wid = None
            for j, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev_wid:
                    pred_labels.append(ID2LABEL.get(preds[j], "O"))
                prev_wid = wid

            # Align lengths
            min_len = min(len(tokens), len(gold_labels), len(pred_labels))

            # Build compact display
            gold_entities = [(tokens[k], gold_labels[k]) for k in range(min_len)
                              if gold_labels[k] != "O"]
            pred_entities = [(tokens[k], pred_labels[k]) for k in range(min_len)
                              if pred_labels[k] != "O"]

            match = (gold_entities == pred_entities)
            mark = "✓" if match else "✗"

            log.info(f"  [{i+1}] Input: {text[:120]}{'…' if len(text) > 120 else ''}")
            log.info(f"       Gold: {gold_entities[:8]}")
            log.info(f"       Pred: {pred_entities[:8]}  {mark}")

        log.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# Train Encoder
# ═══════════════════════════════════════════════════════════════════════════════

def train_encoder(
    model_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    hub_token: str | None = None,
    sample_ds: Dataset = None,
) -> Tuple:
    """
    Full training pipeline for a NER encoder.

    Args:
        model_name: Key from ENCODER_REGISTRY
        train_ds: Half-A training data
        val_ds: Encoder validation data
        sample_ds: Optional dataset for sample inferences (defaults to val_ds)

    Returns:
        (model, tokenizer, eval_results)
    """
    model, tokenizer, cfg = build_encoder(model_name)
    output_dir = os.path.join(OUTPUT_DIR, f"encoder_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Check if already trained
    saved_model = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(saved_model):
        log.info(f"Encoder {model_name} already trained — loading from {output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForTokenClassification.from_pretrained(
            output_dir, num_labels=NUM_LABELS,
            id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(DEVICE)
        if cfg.get("needs_deberta_fix"):
            fix_deberta_params(model)
        return model, tokenizer, None

    log.info(f"Training encoder: {model_name}")
    log.info(f"  Train size: {len(train_ds):,}")
    log.info(f"  Val size:   {len(val_ds):,}")
    log.info(f"  Output dir: {output_dir}")

    # Tokenize datasets
    log.info("  Tokenizing NER data ...")
    tok_fn = lambda ex: tokenize_and_align_ner(ex, tokenizer, cfg["max_length"])
    train_tokenized = train_ds.map(tok_fn, batched=True,
                                    remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(tok_fn, batched=True,
                                remove_columns=val_ds.column_names)

    # Training args
    total_steps = max(1, len(train_tokenized) // (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=BF16_OK,
        fp16=FP16_OK,
        logging_steps=LOG_EVERY_N_STEPS,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=SEED,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
    )

    # Prepare sample inference callback
    if sample_ds is None:
        sample_ds = val_ds
    sample_examples = [sample_ds[i] for i in range(min(SAMPLE_INFERENCE_COUNT, len(sample_ds)))]
    sample_cb = NERSampleInferenceCallback(sample_examples, tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_ner_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            sample_cb,
        ],
    )

    # Determine if we should resume
    resume_from_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            resume_from_checkpoint = True
            log.info(f"  Found existing checkpoint in {output_dir}, enabling resume!")

    # Train!
    log.info("  Starting training ...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"  Model saved to: {output_dir}")

    # Detailed evaluation
    log.info("  Running detailed NER evaluation ...")
    eval_result = trainer.evaluate()
    log.info(f"  Final NER F1: {eval_result.get('eval_f1', 0):.4f}")

    # Per-entity breakdown
    eval_preds = trainer.predict(val_tokenized)
    detailed = compute_ner_metrics_detailed((eval_preds.predictions, eval_preds.label_ids))
    eval_result["per_entity"] = detailed.get("per_entity", {})

    # Save eval results
    import json
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_result, f, indent=2, default=str)

    return model, tokenizer, eval_result


# ═══════════════════════════════════════════════════════════════════════════════
# NER Inference — produce masked text
# ═══════════════════════════════════════════════════════════════════════════════

def run_ner(text: str, model, tokenizer) -> Tuple[str, list]:
    """
    Run NER inference on a text string.

    Returns:
        (masked_text, entity_spans)
        masked_text: text with PII replaced by [TYPE] tags
        entity_spans: list of (entity_text, entity_type) tuples detected
    """
    # Tokenize
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=256, padding=True,
    ).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    word_ids = enc.word_ids()

    # Merge subwords back to words + tags
    words, tags = [], []
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        tag = ID2LABEL.get(preds[i], "O")
        if wid != prev_wid:
            # Get the raw token text
            raw = tokenizer.convert_ids_to_tokens(enc["input_ids"][0][i].item())
            # Clean up subword markers
            raw = raw.replace("▁", "").replace("##", "").replace("Ġ", "")
            words.append(raw)
            tags.append(tag)
        else:
            # Continuation subword — append to current word
            continuation = tokens[i].replace("▁", "").replace("##", "").replace("Ġ", "")
            if words:
                words[-1] += continuation
        prev_wid = wid

    # Build masked text and collect entities
    masked_words = []
    entity_spans = []
    prev_entity = None
    current_entity_words = []

    for w, t in zip(words, tags):
        if t == "O":
            # Flush current entity if any
            if current_entity_words and prev_entity:
                entity_spans.append((" ".join(current_entity_words), prev_entity))
                current_entity_words = []
            masked_words.append(w)
            prev_entity = None
        elif t.startswith("B-"):
            # Flush previous entity
            if current_entity_words and prev_entity:
                entity_spans.append((" ".join(current_entity_words), prev_entity))
            etype = t[2:]
            masked_words.append(f"[{etype}]")
            current_entity_words = [w]
            prev_entity = etype
        elif t.startswith("I-") and prev_entity:
            current_entity_words.append(w)
        else:
            if current_entity_words and prev_entity:
                entity_spans.append((" ".join(current_entity_words), prev_entity))
                current_entity_words = []
            masked_words.append(w)
            prev_entity = None

    # Flush final entity
    if current_entity_words and prev_entity:
        entity_spans.append((" ".join(current_entity_words), prev_entity))

    masked_text = " ".join(masked_words)
    return masked_text, entity_spans


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _find_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory for resume."""
    if not os.path.isdir(output_dir):
        return None
    import re
    ckpts = [d for d in os.listdir(output_dir)
             if os.path.isdir(os.path.join(output_dir, d))
             and re.match(r"checkpoint-\d+$", d)]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    ckpt_path = os.path.join(output_dir, ckpts[-1])
    log.info(f"  Resuming from checkpoint: {ckpt_path}")
    return ckpt_path


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
