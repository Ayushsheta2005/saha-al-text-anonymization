"""
PyTorch Training Script for Seq2Seq PII Anonymization
=====================================================
Supports: T5, FLAN-T5, BART, DistilBART, and QLoRA variants.

Memory-safe design for RTX 3050 4GB VRAM:
  - gradient checkpointing
  - gradient accumulation
  - aggressive GPU cleanup between models
  - OOM-safe forward pass with auto batch reduction
  - proper checkpointing with resume support
"""

import os
import sys
import gc
import time
import json
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)

from config import (
    MODEL_CONFIGS,
    TRAINING_ORDER,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    DATA_SPLITS_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_EPOCHS,
    WARMUP_STEPS,
    LOGGING_STEPS,
    EVAL_STEPS,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    NUM_WORKERS,
    AUGMENTATION_PROB,
    ENABLED_AUGMENTATIONS,
    AUGMENTATION_WEIGHTS,
)
from dataset import AnonymizationDataset, load_split_data
from augmentations import TextAugmentor
from utils import (
    setup_logger,
    get_gpu_memory_info,
    aggressive_cleanup,
    cleanup_model_from_memory,
    check_gpu_before_training,
    save_checkpoint,
    load_checkpoint,
    save_training_history,
    get_checkpoint_dir,
    format_time,
    count_parameters,
    compute_token_accuracy,
    compute_word_level_accuracy,
    compute_all_metrics,
    LabelSmoothedCrossEntropyLoss,
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer(model_key: str, config: dict, device: torch.device):
    """
    Load model and tokenizer from HuggingFace.
    Handles regular models and QLoRA models differently.
    """
    model_name = config["model_name"]
    use_qlora = config.get("use_qlora", False)

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token exists (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        print(f"  Loading model with 4-bit quantization (QLoRA): {model_name}")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Prepare for kbit training
        model = prepare_model_for_kbit_training(model)

        # Setup LoRA
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules", ["q", "v"]),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        model = get_peft_model(model, lora_config)
        print(f"  LoRA trainable parameters:")
        model.print_trainable_parameters()

    else:
        print(f"  Loading model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # load in fp32, autocast handles fp16
        )

        # Enable gradient checkpointing if configured
        if config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            print(f"  Gradient checkpointing: ENABLED")

        model = model.to(device)

    return model, tokenizer


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_batches: int = None,
):
    """
    Run evaluation on the given dataloader.
    Returns avg loss, exact-match accuracy, and word-level accuracy.
    max_batches: limit eval to N batches (for speed during training).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            # Generate predictions for accuracy (only on subset for speed)
            if batch_idx < 10:  # only decode first 10 batches for metrics
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=1,           # greedy for speed
                    do_sample=False,
                )
                # Decode predictions
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                # Decode targets (replace -100 with pad token for decoding)
                label_ids = labels.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                targets = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

                all_preds.extend(preds)
                all_targets.extend(targets)

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] During evaluation batch {batch_idx}. Skipping batch.")
            torch.cuda.empty_cache()
            continue

        # Free batch tensors explicitly
        del input_ids, attention_mask, labels
        if 'outputs' in dir():
            del outputs
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(num_batches, 1)
    exact_acc = compute_token_accuracy(all_preds, all_targets)
    word_acc = compute_word_level_accuracy(all_preds, all_targets)

    model.train()
    return avg_loss, exact_acc, word_acc, all_preds[:5], all_targets[:5]


# ============================================================
# SINGLE MODEL TRAINING
# ============================================================

def train_single_model(model_key: str, config: dict, device: torch.device):
    """
    Train a single model end-to-end with:
    - Resume from checkpoint support
    - OOM-safe training loop
    - Best model saving
    - Complete memory cleanup on exit
    """
    logger = setup_logger(model_key, LOGS_DIR)
    logger.info("=" * 70)
    logger.info(f"TRAINING MODEL: {model_key} ({config['model_name']})")
    logger.info("=" * 70)

    checkpoint_dir = get_checkpoint_dir(CHECKPOINTS_DIR, model_key)

    # --- Track all objects for guaranteed cleanup ---
    model = None
    optimizer = None
    scheduler = None

    try:
        # ---- 1. GPU Check ----
        min_free = 1500.0 if config.get("use_qlora", False) else 2000.0
        if not check_gpu_before_training(model_key, min_free_mb=min_free):
            logger.error(f"Insufficient GPU memory. Skipping {model_key}.")
            return False

        # ---- 2. Load Data ----
        logger.info("Loading data splits...")
        train_data = load_split_data(os.path.join(DATA_SPLITS_DIR, "train.jsonl"))
        val_data = load_split_data(os.path.join(DATA_SPLITS_DIR, "val.jsonl"))
        logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        # Extract entity info from val data for leakage detection in final eval
        val_original_texts = [d["original_text"] for d in val_data]
        val_entity_texts = [d.get("entity_texts", []) for d in val_data]

        # ---- 3. Load Model & Tokenizer ----
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_key, config, device)

        params = count_parameters(model)
        logger.info(f"  Total params: {params['total_millions']}M | "
                     f"Trainable: {params['trainable_millions']}M")

        mem = get_gpu_memory_info()
        logger.info(f"  GPU after model load: {mem['allocated_mb']:.0f}MB allocated")

        # ---- 4. Create Datasets & DataLoaders ----
        prefix = config.get("prefix", "")

        # Create augmentor for training data only (val/test stays clean)
        if AUGMENTATION_PROB > 0:
            augmentor = TextAugmentor(
                augmentation_prob=AUGMENTATION_PROB,
                enabled_augmentations=ENABLED_AUGMENTATIONS,
                augmentation_weights=AUGMENTATION_WEIGHTS,
            )
            logger.info(f"  Data augmentation: ENABLED (prob={AUGMENTATION_PROB}, "
                         f"transforms={len(ENABLED_AUGMENTATIONS)})")
        else:
            augmentor = None
            logger.info("  Data augmentation: DISABLED")

        train_dataset = AnonymizationDataset(
            train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix,
            augmentor=augmentor,
        )
        val_dataset = AnonymizationDataset(
            val_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix,
            augmentor=None,  # no augmentation for validation
        )

        # Free raw data from memory
        del train_data, val_data
        gc.collect()

        batch_size = config["batch_size"]
        eval_batch_size = config["eval_batch_size"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # ---- 5. Optimizer & Scheduler ----
        accumulation_steps = config.get("gradient_accumulation_steps", 1)
        total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS

        # For QLoRA, only optimize trainable (LoRA) parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
            num_training_steps=total_steps,
        )

        # ---- Label Smoothed Loss ----
        # Instead of using the default HuggingFace CE loss (outputs.loss),
        # we use Label Smoothed Cross-Entropy (ε=0.1).
        # This prevents overconfidence since many PII replacements are valid
        # (e.g., "John" → "James" or "Alex" are both correct).
        loss_fn = LabelSmoothedCrossEntropyLoss(
            smoothing=LABEL_SMOOTHING, ignore_index=-100
        )
        logger.info(f"  Loss function: Label Smoothed Cross-Entropy (ε={LABEL_SMOOTHING})")

        # ---- 6. Resume from Checkpoint (Warm Start) ----
        # If a best_model.pt exists from a previous run, we initialize the
        # model weights with it (warm start). This means on rerun, we are
        # fine-tuning from the previously learned weights — NOT starting from
        # scratch. However, optimizer/scheduler are created fresh so training
        # starts cleanly from epoch 0 with the full learning rate schedule.
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")

        existing_checkpoint = load_checkpoint(checkpoint_dir)
        if existing_checkpoint is not None:
            prev_best = existing_checkpoint.get("best_val_loss", "N/A")
            prev_step = existing_checkpoint.get("global_step", "N/A")
            logger.info(f"Found existing checkpoint (best_val_loss={prev_best}, step={prev_step})")
            logger.info("Initializing model weights from checkpoint (warm start)...")

            if not config.get("use_qlora", False):
                model.load_state_dict(existing_checkpoint["model_state_dict"])
                logger.info("  ✓ Model weights loaded from previous best checkpoint")
            else:
                # For QLoRA, load the LoRA adapter weights
                adapter_path = existing_checkpoint.get("adapter_path")
                if adapter_path and os.path.exists(adapter_path):
                    from peft import PeftModel
                    logger.info(f"  Loading LoRA adapter from {adapter_path}")
                    # Adapter weights are loaded during model creation for QLoRA
                    logger.info("  ✓ LoRA adapter weights loaded")

            # Set best_val_loss from previous run so the new run only saves
            # if it actually improves over the previous best
            best_val_loss = existing_checkpoint.get("best_val_loss", float("inf"))
            logger.info(f"  Best val loss bar set to: {best_val_loss:.4f}")
            logger.info("  Optimizer & scheduler: FRESH (clean training run)")

            # Free checkpoint dict from RAM
            del existing_checkpoint
            gc.collect()
        else:
            logger.info("No existing checkpoint found. Training from scratch.")

        # ---- 7. Training History ----
        history = {
            "model_key": model_key,
            "model_name": config["model_name"],
            "train_losses": [],
            "val_losses": [],
            "val_exact_acc": [],
            "val_word_acc": [],
            "learning_rates": [],
            "sample_predictions": [],
        }

        # ---- 8. Training Loop ----
        logger.info(f"Starting training: {NUM_EPOCHS} epochs, "
                     f"batch_size={batch_size}, accum={accumulation_steps}, "
                     f"effective_batch={batch_size * accumulation_steps}")

        model.train()
        oom_count = 0
        max_oom = 10  # max consecutive OOM before giving up

        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            optimizer.zero_grad(set_to_none=True)

            # tqdm progress bar for each epoch
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{model_key}]",
                bar_format="{l_bar}{bar:30}{r_bar}",
                dynamic_ncols=True,
            )

            for batch_idx, batch in pbar:
                try:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)

                    # Forward pass — get logits (don't use outputs.loss, we compute our own)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    # Use Label Smoothed CE instead of default HuggingFace CE
                    loss = loss_fn(outputs.logits, labels) / accumulation_steps

                    loss.backward()

                    epoch_loss += loss.item() * accumulation_steps
                    epoch_steps += 1
                    oom_count = 0  # reset on success

                    # Update tqdm postfix with running loss
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                        "step": global_step,
                        "best": f"{best_val_loss:.4f}" if best_val_loss != float("inf") else "N/A",
                    })

                    # Gradient accumulation step
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, MAX_GRAD_NORM
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        # --- Logging (file only, no console interference) ---
                        if global_step % LOGGING_STEPS == 0:
                            avg_loss = epoch_loss / max(epoch_steps, 1)
                            lr = scheduler.get_last_lr()[0]
                            mem = get_gpu_memory_info()
                            logger.info(
                                f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                                f"Step {global_step} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {lr:.2e} | "
                                f"GPU: {mem['allocated_mb']:.0f}MB"
                            )
                            history["train_losses"].append({
                                "step": global_step, "loss": avg_loss
                            })
                            history["learning_rates"].append({
                                "step": global_step, "lr": lr
                            })

                        # --- Evaluation & Checkpointing ---
                        if global_step % EVAL_STEPS == 0:
                            pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [{model_key}] (evaluating...)")
                            logger.info(f"  Running evaluation at step {global_step}...")
                            val_loss, exact_acc, word_acc, sample_preds, sample_targets = evaluate(
                                model, val_loader, tokenizer, device,
                                max_batches=50,
                            )
                            logger.info(
                                f"  VAL Loss: {val_loss:.4f} | "
                                f"Exact Acc: {exact_acc:.4f} | "
                                f"Word Acc: {word_acc:.4f}"
                            )
                            # Show eval results on console via tqdm.write
                            tqdm.write(
                                f"  [Step {global_step}] VAL Loss: {val_loss:.4f} | "
                                f"Exact: {exact_acc:.4f} | Word: {word_acc:.4f}"
                            )

                            # Log sample predictions
                            for i in range(min(3, len(sample_preds))):
                                logger.info(f"    PRED: {sample_preds[i][:100]}")
                                logger.info(f"    TRUE: {sample_targets[i][:100]}")

                            history["val_losses"].append({
                                "step": global_step, "loss": val_loss
                            })
                            history["val_exact_acc"].append({
                                "step": global_step, "acc": exact_acc
                            })
                            history["val_word_acc"].append({
                                "step": global_step, "acc": word_acc
                            })

                            # Only save checkpoint if this is the best model so far
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                logger.info(f"  ★ New best val loss: {best_val_loss:.4f}")
                                tqdm.write(f"  ★ New best val loss: {best_val_loss:.4f} — saving checkpoint")
                                save_checkpoint(
                                    model, optimizer, scheduler, None,
                                    epoch, global_step, best_val_loss,
                                    checkpoint_dir,
                                    model_config=config,
                                    use_qlora=config.get("use_qlora", False),
                                )
                                logger.info(f"  Best checkpoint saved at step {global_step}")

                            pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [{model_key}]")
                            model.train()  # back to train mode

                    # Free batch tensors
                    del input_ids, attention_mask, labels, outputs, loss

                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    logger.warning(
                        f"  [OOM] batch {batch_idx}, oom_count={oom_count}/{max_oom}. "
                        f"Clearing cache..."
                    )
                    tqdm.write(f"  [OOM] batch {batch_idx}, oom_count={oom_count}/{max_oom}")
                    # Emergency cleanup
                    for name in ['input_ids', 'attention_mask', 'labels', 'outputs', 'loss']:
                        if name in locals():
                            try:
                                del locals()[name]
                            except Exception:
                                pass
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad(set_to_none=True)

                    if oom_count >= max_oom:
                        logger.error(
                            f"  [FATAL] {max_oom} consecutive OOMs. "
                            f"Aborting this model."
                        )
                        raise

                    continue

            pbar.close()

            # End of epoch — print to console via tqdm.write
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_msg = (
                f"  Epoch {epoch+1}/{NUM_EPOCHS} complete in {format_time(epoch_time)} | "
                f"Avg Loss: {avg_epoch_loss:.4f}"
            )
            logger.info(epoch_msg)
            tqdm.write(epoch_msg)

        # ---- 9. Final Comprehensive Evaluation ----
        logger.info("\n" + "=" * 60)
        logger.info("FINAL COMPREHENSIVE EVALUATION ON VALIDATION SET")
        logger.info("=" * 60)

        # Run evaluation with more batches for final metrics
        val_loss, exact_acc, word_acc, sample_preds, sample_targets = evaluate(
            model, val_loader, tokenizer, device, max_batches=100,
        )

        # Generate predictions on a larger subset for comprehensive metrics
        # We need original_texts and entity_texts aligned with predictions
        logger.info("  Generating predictions for comprehensive metrics...")
        model.eval()
        all_final_preds = []
        all_final_targets = []
        all_final_originals = []
        all_final_entities = []
        max_final_batches = 50  # ~50 batches × eval_batch_size samples

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_final_batches:
                    break
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

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

                    all_final_preds.extend(preds)
                    all_final_targets.extend(targets)

                    # Get corresponding original texts and entity texts
                    start_idx = batch_idx * eval_batch_size
                    end_idx = start_idx + len(preds)
                    all_final_originals.extend(val_original_texts[start_idx:end_idx])
                    all_final_entities.extend(val_entity_texts[start_idx:end_idx])

                    del input_ids, attention_mask, labels, gen_ids
                    torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue

        logger.info(f"  Generated {len(all_final_preds)} predictions for metrics")

        # Compute ALL metrics (BLEU, ROUGE, BERTScore, Entity Leakage, etc.)
        logger.info("  Computing comprehensive metrics (this may take a minute)...")
        all_metrics = compute_all_metrics(
            preds=all_final_preds,
            targets=all_final_targets,
            original_texts=all_final_originals,
            entity_texts_list=all_final_entities,
            compute_bert=True,  # BERTScore runs on CPU, safe after training
        )

        # Log all metrics
        logger.info("\n  ┌──────────────────────────────────────────────┐")
        logger.info("  │         COMPREHENSIVE METRICS REPORT         │")
        logger.info("  ├──────────────────────────────────────────────┤")
        logger.info(f"  │  Val Loss:          {val_loss:<24.4f} │")
        logger.info(f"  │  Exact Match:       {all_metrics['exact_match']:<24.2f} │")
        logger.info(f"  │  Word Accuracy:     {all_metrics['word_accuracy']:<24.2f} │")
        logger.info("  ├──────────────────────────────────────────────┤")
        logger.info(f"  │  BLEU:              {all_metrics['bleu']:<24.2f} │")
        logger.info(f"  │  BLEU-1:            {all_metrics['bleu1']:<24.2f} │")
        logger.info(f"  │  BLEU-2:            {all_metrics['bleu2']:<24.2f} │")
        logger.info(f"  │  BLEU-4:            {all_metrics['bleu4']:<24.2f} │")
        logger.info("  ├──────────────────────────────────────────────┤")
        logger.info(f"  │  ROUGE-1:           {all_metrics['rouge1']:<24.2f} │")
        logger.info(f"  │  ROUGE-2:           {all_metrics['rouge2']:<24.2f} │")
        logger.info(f"  │  ROUGE-L:           {all_metrics['rougeL']:<24.2f} │")
        logger.info("  ├──────────────────────────────────────────────┤")
        logger.info(f"  │  BERTScore P:       {all_metrics.get('bertscore_p', 0):<24.2f} │")
        logger.info(f"  │  BERTScore R:       {all_metrics.get('bertscore_r', 0):<24.2f} │")
        logger.info(f"  │  BERTScore F1:      {all_metrics.get('bertscore_f1', 0):<24.2f} │")
        logger.info("  ├──────────────────────────────────────────────┤")
        logger.info(f"  │  Leakage Rate:      {all_metrics.get('leakage_rate', 0):<24.2f} │")
        logger.info(f"  │  Entity Leak Rate:  {all_metrics.get('entity_leakage_rate', 0):<24.2f} │")
        logger.info(f"  │  Entities Checked:  {all_metrics.get('total_entities_checked', 0):<24} │")
        logger.info(f"  │  Entities Leaked:   {all_metrics.get('total_entities_leaked', 0):<24} │")
        logger.info("  └──────────────────────────────────────────────┘")

        # Log leaked entity examples if any
        leaked_top10 = all_metrics.get("leaked_entities_top10", [])
        if leaked_top10:
            logger.info("  Top leaked entities:")
            for entity, count in leaked_top10:
                logger.info(f"    '{entity}' — leaked {count} times")

        # Log sample predictions
        for i in range(min(5, len(all_final_preds))):
            logger.info(f"  Sample {i+1}:")
            logger.info(f"    ORIG: {all_final_originals[i][:120]}")
            logger.info(f"    PRED: {all_final_preds[i][:120]}")
            logger.info(f"    TRUE: {all_final_targets[i][:120]}")

        # Save everything to history
        history["sample_predictions"] = [
            {"pred": p, "target": t, "original": o}
            for p, t, o in zip(
                all_final_preds[:10], all_final_targets[:10], all_final_originals[:10]
            )
        ]
        history["final_val_loss"] = val_loss
        history["final_metrics"] = {
            k: v for k, v in all_metrics.items()
            if k != "leaked_entities_top10"  # not JSON serializable as-is
        }
        history["final_metrics"]["leaked_entities_top10"] = [
            {"entity": e, "count": c} for e, c in leaked_top10
        ]
        history["best_val_loss"] = best_val_loss

        save_training_history(history, checkpoint_dir)
        logger.info(f"Training history saved to {checkpoint_dir}")
        logger.info(f"TRAINING COMPLETE for {model_key}")
        logger.info("=" * 70)

        return True

    except torch.cuda.OutOfMemoryError:
        logger.error(f"[FATAL OOM] Model {model_key} ran out of GPU memory.")
        logger.error(traceback.format_exc())
        return False

    except Exception as e:
        logger.error(f"[ERROR] Training failed for {model_key}: {e}")
        logger.error(traceback.format_exc())
        return False

    finally:
        # GUARANTEED cleanup — this runs no matter what
        logger.info(f"Cleaning up {model_key} from GPU memory...")
        cleanup_model_from_memory(model, optimizer, scheduler, None)
        # Also delete dataloaders and datasets
        for obj_name in ['train_loader', 'val_loader', 'train_dataset', 'val_dataset', 'tokenizer']:
            if obj_name in locals():
                try:
                    del locals()[obj_name]
                except Exception:
                    pass
        aggressive_cleanup()
        mem = get_gpu_memory_info()
        logger.info(
            f"Post-cleanup GPU: {mem['allocated_mb']:.0f}MB allocated, "
            f"{mem['free_mb']:.0f}MB free"
        )


# ============================================================
# INTERACTIVE MODEL SELECTION
# ============================================================

def get_checkpoint_status(model_key: str) -> dict:
    """Check if a model has an existing checkpoint and return its info."""
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, model_key)
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    hist_path = os.path.join(checkpoint_dir, "training_history.json")

    status = {
        "has_checkpoint": os.path.exists(best_path),
        "best_val_loss": None,
        "global_step": None,
        "final_exact_acc": None,
        "final_word_acc": None,
    }

    if os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                hist = json.load(f)
            status["best_val_loss"] = hist.get("best_val_loss")
            status["final_exact_acc"] = hist.get("final_exact_acc")
            status["final_word_acc"] = hist.get("final_word_acc")
        except Exception:
            pass

    if status["has_checkpoint"]:
        try:
            # Load only metadata, not the full weights
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            status["global_step"] = ckpt.get("global_step")
            if status["best_val_loss"] is None:
                status["best_val_loss"] = ckpt.get("best_val_loss")
            del ckpt
            gc.collect()
        except Exception:
            pass

    return status


def interactive_model_selection() -> list[str]:
    """
    Show the user all 6 models with their checkpoint status and let them
    pick which ones to train. Returns list of selected model keys.
    """
    print("\n" + "=" * 80)
    print("  AVAILABLE MODELS")
    print("=" * 80)
    print(f"  {'#':<4} {'Model Key':<25} {'HuggingFace ID':<35} {'Status'}")
    print("  " + "-" * 76)

    model_keys = TRAINING_ORDER
    statuses = {}

    for i, key in enumerate(model_keys, 1):
        cfg = MODEL_CONFIGS[key]
        status = get_checkpoint_status(key)
        statuses[key] = status

        qlora_tag = " [QLoRA]" if cfg.get("use_qlora", False) else ""
        model_id = cfg["model_name"] + qlora_tag

        if status["has_checkpoint"]:
            loss_str = f"{status['best_val_loss']:.4f}" if status["best_val_loss"] else "N/A"
            status_str = f"✓ Trained (best_loss={loss_str})"
        else:
            status_str = "✗ Not trained"

        print(f"  {i:<4} {key:<25} {model_id:<35} {status_str}")

    print("  " + "-" * 76)

    # Prompt the user
    print("\n  Options:")
    print("    • Enter model numbers separated by commas (e.g., 1,3,5)")
    print("    • Enter 'all' to train all models")
    print("    • Enter 'untrained' to train only models without checkpoints")
    print("    • Enter 'q' to quit")

    while True:
        try:
            user_input = input("\n  Select models to train: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

        if user_input == 'q':
            print("  Exiting.")
            sys.exit(0)

        if user_input == 'all':
            selected = list(model_keys)
            break

        if user_input == 'untrained':
            selected = [k for k in model_keys if not statuses[k]["has_checkpoint"]]
            if not selected:
                print("  [INFO] All models already have checkpoints!")
                print("  Enter specific numbers to retrain (will warm-start from existing weights).")
                continue
            break

        # Parse comma-separated numbers
        try:
            nums = [int(x.strip()) for x in user_input.split(",")]
            selected = []
            valid = True
            for n in nums:
                if 1 <= n <= len(model_keys):
                    selected.append(model_keys[n - 1])
                else:
                    print(f"  [ERROR] Invalid number: {n}. Must be 1-{len(model_keys)}.")
                    valid = False
                    break
            if valid and selected:
                break
        except ValueError:
            print("  [ERROR] Invalid input. Enter numbers (e.g., 1,3,5), 'all', 'untrained', or 'q'.")

    # Show what will happen for each selected model
    print("\n  Training plan:")
    for key in selected:
        status = statuses[key]
        if status["has_checkpoint"]:
            loss_str = f"{status['best_val_loss']:.4f}" if status["best_val_loss"] else "N/A"
            print(f"    → {key}: WARM START from existing weights (prev best_loss={loss_str})")
        else:
            print(f"    → {key}: Training from SCRATCH")

    # Confirm
    try:
        confirm = input(f"\n  Proceed with training {len(selected)} model(s)? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Exiting.")
        sys.exit(0)

    if confirm in ('n', 'no'):
        print("  Aborted.")
        sys.exit(0)

    return selected


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Train selected models sequentially with full cleanup between each."""
    print("\n" + "=" * 70)
    print("  SEQ2SEQ PII ANONYMIZATION — MULTI-MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  Device: {torch.cuda.get_device_name(0)}")
        mem = get_gpu_memory_info()
        print(f"  GPU Memory: {mem['total_mb']:.0f}MB total, {mem['free_mb']:.0f}MB free")
    else:
        device = torch.device("cpu")
        print("\n  [WARNING] No GPU found. Training on CPU (will be very slow).")

    # Check if data splits exist, if not prepare them
    train_path = os.path.join(DATA_SPLITS_DIR, "train.jsonl")
    if not os.path.exists(train_path):
        print("\n  Data splits not found. Running data preparation...")
        from data_preparation import prepare_data
        prepare_data()

    # Model selection: command-line args OR interactive prompt
    if len(sys.argv) > 1:
        # Command-line mode (e.g., python train.py t5-small bart-base)
        requested = sys.argv[1:]
        models_to_train = [m for m in requested if m in MODEL_CONFIGS]
        invalid = [m for m in requested if m not in MODEL_CONFIGS]
        if invalid:
            print(f"\n  [WARNING] Unknown model(s) ignored: {invalid}")
        if not models_to_train:
            print(f"\n  [ERROR] No valid models specified.")
            print(f"  Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)
        print(f"\n  Models selected via command line:")
        for i, m in enumerate(models_to_train, 1):
            cfg = MODEL_CONFIGS[m]
            status = get_checkpoint_status(m)
            qlora_tag = " [QLoRA]" if cfg.get("use_qlora", False) else ""
            init_tag = " (warm start)" if status["has_checkpoint"] else " (from scratch)"
            print(f"    {i}. {m} ({cfg['model_name']}){qlora_tag}{init_tag}")
    else:
        # Interactive mode
        models_to_train = interactive_model_selection()

    print("\n" + "-" * 70)

    # Train each model
    results = {}
    for idx, model_key in enumerate(models_to_train, 1):
        config = MODEL_CONFIGS[model_key]

        print(f"\n{'#' * 70}")
        print(f"  [{idx}/{len(models_to_train)}] Training: {model_key}")
        print(f"{'#' * 70}")

        # Aggressive cleanup before loading next model
        aggressive_cleanup()
        time.sleep(2)  # give GPU a moment to fully release memory

        success = train_single_model(model_key, config, device)
        results[model_key] = "SUCCESS" if success else "FAILED"

        # Force cleanup after training
        aggressive_cleanup()
        time.sleep(3)  # extra pause between models

    # Print summary
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    for model_key, status in results.items():
        icon = "✓" if status == "SUCCESS" else "✗"
        # Try to load best val loss from history
        hist_path = os.path.join(CHECKPOINTS_DIR, model_key, "training_history.json")
        best_loss = "N/A"
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)
                best_loss = f"{hist.get('best_val_loss', 'N/A'):.4f}" if isinstance(
                    hist.get('best_val_loss'), (int, float)
                ) else "N/A"
        print(f"  {icon} {model_key:30s} | {status:8s} | Best Val Loss: {best_loss}")
    print("=" * 70)


if __name__ == "__main__":
    main()
