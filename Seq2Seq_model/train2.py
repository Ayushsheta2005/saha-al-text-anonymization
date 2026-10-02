"""
Kaggle Training Script for Seq2Seq PII Anonymization
=====================================================
Simplified training pipeline tuned for Kaggle T4x2 (15 GB VRAM each).

Differences from train.py (RTX 3050 4GB):
  - Larger batch sizes (more VRAM available)
  - DataParallel for 2× T4 GPUs
  - Test set evaluation after training
  - No aggressive memory management hacks (plenty of VRAM)

Usage:
    python train2.py                    # interactive model selection
    python train2.py t5-small bart-base # command-line selection
    python train2.py all                # train every model
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
    TRAINING_ORDER,
    AUGMENTATION_PROB,
    ENABLED_AUGMENTATIONS,
    AUGMENTATION_WEIGHTS,
)
from dataset import AnonymizationDataset, load_split_data
from augmentations import TextAugmentor
from utils import (
    setup_logger,
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
# KAGGLE MODEL CONFIGS  (larger batches than RTX 3050)
# ============================================================

NUM_WORKERS = 4  # Kaggle has 4 CPU cores

MODEL_CONFIGS = {
    "t5-efficient-tiny": {
        "model_name": "google/t5-efficient-tiny",
        "model_type": "t5",
        "batch_size": 32,
        "eval_batch_size": 64,
        "learning_rate": 3e-4,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
        "prefix": "anonymize: ",
        "use_qlora": False,
    },
    "t5-small": {
        "model_name": "google-t5/t5-small",
        "model_type": "t5",
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 3e-4,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 2,
        "prefix": "anonymize: ",
        "use_qlora": False,
    },
    "flan-t5-small": {
        "model_name": "google/flan-t5-small",
        "model_type": "t5",
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 3e-4,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 2,
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
        "use_qlora": False,
    },
    "bart-base": {
        "model_name": "facebook/bart-base",
        "model_type": "bart",
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 2e-5,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 2,
        "prefix": "",
        "use_qlora": False,
    },
    "distilbart": {
        "model_name": "sshleifer/distilbart-cnn-6-6",
        "model_type": "bart",
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 2e-5,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 2,
        "prefix": "",
        "use_qlora": False,
    },
    # "flan-t5-base-qlora": {
    #     "model_name": "google/flan-t5-base",
    #     "model_type": "t5",
    #     "batch_size": 16,
    #     "eval_batch_size": 32,
    #     "learning_rate": 2e-4,
    #     "gradient_checkpointing": True,
    #     "gradient_accumulation_steps": 2,
    #     "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
    #     "use_qlora": True,
    #     "lora_r": 16,
    #     "lora_alpha": 32,
    #     "lora_dropout": 0.05,
    #     "lora_target_modules": ["q", "v"],
    # },
}


# ============================================================
# HELPERS
# ============================================================

def get_device():
    """Return the torch device and number of GPUs."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2
            print(f"  GPU {i}: {name} ({mem:.0f} MB)")
        return device, num_gpus
    print("  [WARNING] No GPU detected — training on CPU.")
    return torch.device("cpu"), 0


def unwrap(model):
    """Return the underlying model if wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer(config: dict, device: torch.device, num_gpus: int):
    """Load model + tokenizer. Wraps in DataParallel when >1 GPU and not QLoRA."""
    model_name = config["model_name"]
    use_qlora = config.get("use_qlora", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
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
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules", ["q", "v"]),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # QLoRA uses device_map="auto" — do NOT wrap in DataParallel
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        if config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        model = model.to(device)
        if num_gpus > 1:
            model = nn.DataParallel(model)
            print(f"  DataParallel: enabled across {num_gpus} GPUs")

    return model, tokenizer


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, max_batches=None):
    """Quick evaluation — returns (avg_loss, exact_acc, word_acc, sample_preds, sample_targets)."""
    model.eval()
    raw = unwrap(model)
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for idx, batch in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        total_loss += loss.item()
        n_batches += 1

        # Decode first few batches for accuracy metrics
        if idx < 10:
            gen_ids = raw.generate(
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
    exact_acc = compute_token_accuracy(all_preds, all_targets)
    word_acc = compute_word_level_accuracy(all_preds, all_targets)

    model.train()
    return avg_loss, exact_acc, word_acc, all_preds[:5], all_targets[:5]


# ============================================================
# FULL EVALUATION (val or test — all metrics)
# ============================================================

@torch.no_grad()
def full_evaluation(model, dataloader, tokenizer, device, eval_batch_size,
                    original_texts, entity_texts, split_name, logger):
    """
    Generate predictions on the entire split and compute comprehensive metrics
    (BLEU, ROUGE, BERTScore, entity leakage).
    Returns (avg_loss, metrics_dict, preds, targets, originals).
    """
    logger.info(f"Running full evaluation on {split_name} set …")
    model.eval()
    raw = unwrap(model)

    total_loss, n_batches = 0.0, 0
    all_preds, all_targets, all_originals, all_entities = [], [], [], []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  Eval {split_name}", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        total_loss += loss.item()
        n_batches += 1

        gen_ids = raw.generate(
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

        start = batch_idx * eval_batch_size
        end = start + len(preds)
        all_originals.extend(original_texts[start:end])
        all_entities.extend(entity_texts[start:end])

    avg_loss = total_loss / max(n_batches, 1)

    logger.info(f"  {split_name}: generated {len(all_preds)} predictions — computing metrics …")
    metrics = compute_all_metrics(
        preds=all_preds,
        targets=all_targets,
        original_texts=all_originals,
        entity_texts_list=all_entities,
        compute_bert=True,
    )

    # Log summary
    logger.info(f"  {split_name} Loss: {avg_loss:.4f}")
    for key in ["exact_match", "word_accuracy", "bleu", "rouge1", "rougeL",
                 "bertscore_f1", "entity_leakage_rate"]:
        logger.info(f"  {key}: {metrics.get(key, 0):.2f}")

    # Print to console
    tqdm.write(f"\n  ── {split_name.upper()} RESULTS ──")
    tqdm.write(f"  Loss: {avg_loss:.4f}  |  Exact: {metrics['exact_match']:.2f}%  |  "
               f"Word Acc: {metrics['word_accuracy']:.2f}%")
    tqdm.write(f"  BLEU: {metrics['bleu']:.2f}  |  ROUGE-L: {metrics['rougeL']:.2f}  |  "
               f"BERTScore F1: {metrics.get('bertscore_f1', 0):.2f}")
    tqdm.write(f"  Entity Leakage: {metrics.get('entity_leakage_rate', 0):.2f}% "
               f"({metrics.get('total_entities_leaked', 0)}/{metrics.get('total_entities_checked', 0)})")

    # Sample predictions
    for i in range(min(3, len(all_preds))):
        logger.info(f"  Sample {i+1}:  PRED={all_preds[i][:120]}  |  TRUE={all_targets[i][:120]}")

    return avg_loss, metrics, all_preds, all_targets, all_originals


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_single_model(model_key, config, device, num_gpus):
    """Train a single model end-to-end. Returns True on success."""
    logger = setup_logger(model_key, LOGS_DIR)
    logger.info("=" * 70)
    logger.info(f"TRAINING: {model_key}  ({config['model_name']})")
    logger.info("=" * 70)

    checkpoint_dir = get_checkpoint_dir(CHECKPOINTS_DIR, model_key)

    try:
        # ── 1. Data ──────────────────────────────────────────────
        train_data = load_split_data(os.path.join(DATA_SPLITS_DIR, "train.jsonl"))
        val_data   = load_split_data(os.path.join(DATA_SPLITS_DIR, "val.jsonl"))
        test_data  = load_split_data(os.path.join(DATA_SPLITS_DIR, "test.jsonl"))
        logger.info(f"  Data: train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

        val_original_texts  = [d["original_text"] for d in val_data]
        val_entity_texts    = [d.get("entity_texts", []) for d in val_data]
        test_original_texts = [d["original_text"] for d in test_data]
        test_entity_texts   = [d.get("entity_texts", []) for d in test_data]

        # ── 2. Model & Tokenizer ─────────────────────────────────
        model, tokenizer = load_model_and_tokenizer(config, device, num_gpus)
        params = count_parameters(unwrap(model))
        logger.info(f"  Params: {params['total_millions']}M total, {params['trainable_millions']}M trainable")

        # ── 3. Datasets & Loaders ────────────────────────────────
        prefix = config.get("prefix", "")
        augmentor = None
        if AUGMENTATION_PROB > 0:
            augmentor = TextAugmentor(
                augmentation_prob=AUGMENTATION_PROB,
                enabled_augmentations=ENABLED_AUGMENTATIONS,
                augmentation_weights=AUGMENTATION_WEIGHTS,
            )

        train_dataset = AnonymizationDataset(train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix, augmentor=augmentor)
        val_dataset   = AnonymizationDataset(val_data,   tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix, augmentor=None)
        test_dataset  = AnonymizationDataset(test_data,  tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix, augmentor=None)

        bs      = config["batch_size"]
        eval_bs = config["eval_batch_size"]

        train_loader = DataLoader(train_dataset, batch_size=bs,      shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=eval_bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_dataset,  batch_size=eval_bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # ── 4. Optimizer & Scheduler ─────────────────────────────
        accum_steps = config.get("gradient_accumulation_steps", 1)
        total_steps = (len(train_loader) // accum_steps) * NUM_EPOCHS

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
                                                    num_training_steps=total_steps)
        loss_fn = LabelSmoothedCrossEntropyLoss(smoothing=LABEL_SMOOTHING, ignore_index=-100)

        # ── 5. Resume / Warm Start ───────────────────────────────
        best_val_loss = float("inf")
        global_step = 0

        ckpt = load_checkpoint(checkpoint_dir)
        if ckpt is not None:
            prev_loss = ckpt.get("best_val_loss", "N/A")
            logger.info(f"  Found checkpoint (best_val_loss={prev_loss}) — warm-starting weights")
            if not config.get("use_qlora", False):
                unwrap(model).load_state_dict(ckpt["model_state_dict"])
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            del ckpt
            gc.collect()
        else:
            logger.info("  No checkpoint — training from scratch")

        # ── 6. History ───────────────────────────────────────────
        history = {
            "model_key": model_key,
            "model_name": config["model_name"],
            "environment": "kaggle",
            "batch_size": bs,
            "effective_batch_size": bs * accum_steps,
            "train_losses": [],
            "val_losses": [],
            "val_exact_acc": [],
            "val_word_acc": [],
            "learning_rates": [],
        }

        # ── 7. Training Loop ────────────────────────────────────
        effective_bs = bs * accum_steps * max(num_gpus, 1)
        logger.info(f"  Training: {NUM_EPOCHS} epochs, batch={bs}, accum={accum_steps}, effective={effective_bs}")

        model.train()

        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                        bar_format="{l_bar}{bar:30}{r_bar}", dynamic_ncols=True)

            for batch_idx, batch in pbar:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fn(outputs.logits, labels) / accum_steps
                loss.backward()

                epoch_loss += loss.item() * accum_steps
                epoch_steps += 1

                avg_loss = epoch_loss / epoch_steps
                pbar.set_postfix(loss=f"{avg_loss:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.1e}",
                                 step=global_step,
                                 best=f"{best_val_loss:.4f}" if best_val_loss != float("inf") else "N/A",
                                 refresh=False)

                # Accumulation step
                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # Log to file
                    if global_step % LOGGING_STEPS == 0:
                        lr = scheduler.get_last_lr()[0]
                        logger.info(f"  Epoch {epoch+1} | Step {global_step} | Loss {avg_loss:.4f} | LR {lr:.2e}")
                        history["train_losses"].append({"step": global_step, "loss": avg_loss})
                        history["learning_rates"].append({"step": global_step, "lr": lr})

                    # Eval & Checkpoint
                    if global_step % EVAL_STEPS == 0:
                        val_loss, exact_acc, word_acc, s_preds, s_targets = evaluate(
                            model, val_loader, tokenizer, device, max_batches=50)

                        logger.info(f"  [Eval step {global_step}] val_loss={val_loss:.4f}  "
                                    f"exact={exact_acc:.4f}  word={word_acc:.4f}")
                        tqdm.write(f"  [Step {global_step}] Val Loss: {val_loss:.4f} | "
                                   f"Exact: {exact_acc:.4f} | Word: {word_acc:.4f}")

                        history["val_losses"].append({"step": global_step, "loss": val_loss})
                        history["val_exact_acc"].append({"step": global_step, "acc": exact_acc})
                        history["val_word_acc"].append({"step": global_step, "acc": word_acc})

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            tqdm.write(f"  ★ New best val loss: {best_val_loss:.4f} — saving")
                            logger.info(f"  ★ New best val loss: {best_val_loss:.4f}")
                            save_checkpoint(
                                unwrap(model), optimizer, scheduler, None,
                                epoch, global_step, best_val_loss,
                                checkpoint_dir, model_config=config,
                                use_qlora=config.get("use_qlora", False),
                            )

                        model.train()

            pbar.close()
            epoch_time = time.time() - epoch_start
            tqdm.write(f"  Epoch {epoch+1} done in {format_time(epoch_time)} — avg loss: {epoch_loss / max(epoch_steps, 1):.4f}")

        # ── 8. Final Evaluation — Val ────────────────────────────
        val_loss, val_metrics, val_preds, val_targets, val_originals = full_evaluation(
            model, val_loader, tokenizer, device, eval_bs,
            val_original_texts, val_entity_texts, "validation", logger)

        # ── 9. Final Evaluation — Test ───────────────────────────
        test_loss, test_metrics, test_preds, test_targets, test_originals = full_evaluation(
            model, test_loader, tokenizer, device, eval_bs,
            test_original_texts, test_entity_texts, "test", logger)

        # ── 10. Save History ─────────────────────────────────────
        def _serialise_leaked(metrics_dict):
            """Make leaked_entities_top10 JSON-friendly."""
            top10 = metrics_dict.pop("leaked_entities_top10", [])
            metrics_dict["leaked_entities_top10"] = [{"entity": e, "count": c} for e, c in top10]
            return metrics_dict

        history["best_val_loss"] = best_val_loss
        history["final_val_loss"] = val_loss
        history["final_metrics"] = _serialise_leaked(val_metrics)
        history["test_loss"] = test_loss
        history["test_metrics"] = _serialise_leaked(test_metrics)

        history["sample_predictions"] = [
            {"pred": p, "target": t, "original": o}
            for p, t, o in zip(val_preds[:10], val_targets[:10], val_originals[:10])
        ]
        history["test_sample_predictions"] = [
            {"pred": p, "target": t, "original": o}
            for p, t, o in zip(test_preds[:10], test_targets[:10], test_originals[:10])
        ]

        save_training_history(history, checkpoint_dir)
        logger.info(f"History saved to {checkpoint_dir}")

        # Print val vs test comparison
        tqdm.write(f"\n  {'Metric':<25} {'Val':<12} {'Test':<12}")
        tqdm.write(f"  {'─'*25} {'─'*12} {'─'*12}")
        for m in ["exact_match", "word_accuracy", "bleu", "rougeL", "bertscore_f1", "entity_leakage_rate"]:
            v = history["final_metrics"].get(m, 0)
            t = history["test_metrics"].get(m, 0)
            tqdm.write(f"  {m:<25} {v:<12.2f} {t:<12.2f}")

        logger.info(f"DONE — {model_key}")
        return True

    except Exception as e:
        logger.error(f"Training failed for {model_key}: {e}")
        logger.error(traceback.format_exc())
        tqdm.write(f"  [ERROR] {model_key}: {e}")
        return False

    finally:
        # Basic cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# CHECKPOINT STATUS HELPER
# ============================================================

def get_checkpoint_status(model_key):
    """Return dict with has_checkpoint / best_val_loss."""
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, model_key)
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    hist_path = os.path.join(ckpt_dir, "training_history.json")

    status = {"has_checkpoint": os.path.exists(best_path), "best_val_loss": None}

    if os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                status["best_val_loss"] = json.load(f).get("best_val_loss")
        except Exception:
            pass

    if status["has_checkpoint"] and status["best_val_loss"] is None:
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            status["best_val_loss"] = ckpt.get("best_val_loss")
            del ckpt
        except Exception:
            pass

    return status


# ============================================================
# INTERACTIVE MODEL SELECTION
# ============================================================

def interactive_model_selection():
    """Show models, let user pick. Returns list of model keys."""
    model_keys = TRAINING_ORDER

    print("\n" + "=" * 80)
    print("  AVAILABLE MODELS (Kaggle config)")
    print("=" * 80)
    print(f"  {'#':<4} {'Key':<25} {'HuggingFace ID':<35} {'Status'}")
    print("  " + "-" * 76)

    statuses = {}
    for i, key in enumerate(model_keys, 1):
        cfg = MODEL_CONFIGS[key]
        st = get_checkpoint_status(key)
        statuses[key] = st

        qlora = " [QLoRA]" if cfg.get("use_qlora") else ""
        if st["has_checkpoint"]:
            loss_s = f"{st['best_val_loss']:.4f}" if st["best_val_loss"] else "N/A"
            tag = f"✓ Trained (loss={loss_s})"
        else:
            tag = "✗ Not trained"
        print(f"  {i:<4} {key:<25} {cfg['model_name']}{qlora:<35} {tag}")

    print("  " + "-" * 76)
    print("  Enter numbers (e.g. 1,3,5) | 'all' | 'untrained' | 'q' to quit")

    while True:
        try:
            inp = input("\n  Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

        if inp == "q":
            sys.exit(0)
        if inp == "all":
            selected = list(model_keys)
            break
        if inp == "untrained":
            selected = [k for k in model_keys if not statuses[k]["has_checkpoint"]]
            if not selected:
                print("  All models already trained. Pick numbers to retrain.")
                continue
            break
        try:
            nums = [int(x.strip()) for x in inp.split(",")]
            selected = []
            for n in nums:
                if 1 <= n <= len(model_keys):
                    selected.append(model_keys[n - 1])
                else:
                    print(f"  Invalid: {n}")
                    break
            else:
                if selected:
                    break
        except ValueError:
            print("  Invalid input.")

    # Confirm
    print("\n  Will train:")
    for key in selected:
        st = statuses[key]
        mode = "warm start" if st["has_checkpoint"] else "from scratch"
        print(f"    → {key} ({mode})")

    try:
        ans = input(f"\n  Proceed ({len(selected)} model(s))? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
    if ans in ("n", "no"):
        sys.exit(0)

    return selected


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  SEQ2SEQ PII ANONYMIZATION — KAGGLE TRAINING PIPELINE")
    print("=" * 70)

    device, num_gpus = get_device()

    # Ensure data splits exist
    if not os.path.exists(os.path.join(DATA_SPLITS_DIR, "train.jsonl")):
        print("  Data splits not found — running data preparation …")
        from data_preparation import prepare_data
        prepare_data()

    # Model selection
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        if requested == ["all"]:
            models_to_train = list(TRAINING_ORDER)
        else:
            models_to_train = [m for m in requested if m in MODEL_CONFIGS]
            bad = [m for m in requested if m not in MODEL_CONFIGS]
            if bad:
                print(f"  Unknown models ignored: {bad}")
            if not models_to_train:
                print(f"  No valid models. Available: {list(MODEL_CONFIGS.keys())}")
                sys.exit(1)
        print("  Selected:", ", ".join(models_to_train))
    else:
        models_to_train = interactive_model_selection()

    # Train each model
    results = {}
    for idx, model_key in enumerate(models_to_train, 1):
        config = MODEL_CONFIGS[model_key]
        print(f"\n{'#' * 70}")
        print(f"  [{idx}/{len(models_to_train)}] {model_key}")
        print(f"{'#' * 70}")

        ok = train_single_model(model_key, config, device, num_gpus)
        results[model_key] = "SUCCESS" if ok else "FAILED"

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)

    # Summary
    print("\n" + "=" * 90)
    print("  TRAINING SUMMARY")
    print("=" * 90)
    print(f"  {'Model':<25} {'Status':<10} {'Val Loss':<12} {'Test Loss':<12} {'Test BLEU':<12} {'Leak%'}")
    print("  " + "-" * 80)

    for key, status in results.items():
        icon = "✓" if status == "SUCCESS" else "✗"
        val_loss = test_loss = bleu = leak = "N/A"
        hist_path = os.path.join(CHECKPOINTS_DIR, key, "training_history.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    h = json.load(f)
                bl = h.get("best_val_loss")
                val_loss = f"{bl:.4f}" if isinstance(bl, (int, float)) else "N/A"
                tl = h.get("test_loss")
                test_loss = f"{tl:.4f}" if isinstance(tl, (int, float)) else "N/A"
                tm = h.get("test_metrics", {})
                b = tm.get("bleu")
                bleu = f"{b:.2f}" if isinstance(b, (int, float)) else "N/A"
                e = tm.get("entity_leakage_rate")
                leak = f"{e:.2f}%" if isinstance(e, (int, float)) else "N/A"
            except Exception:
                pass
        print(f"  {icon} {key:<23} {status:<10} {val_loss:<12} {test_loss:<12} {bleu:<12} {leak}")

    print("=" * 90)


if __name__ == "__main__":
    main()
