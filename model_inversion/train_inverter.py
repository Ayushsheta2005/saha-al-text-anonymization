#!/usr/bin/env python3
"""
Step 3 — Inverter Model Trainer (BART-base)
============================================
Trains a BART-base model to reverse the victim BART-base anonymizer.

Input:  anonymized text  (what the attacker sees)
Output: original text    (what the attacker wants to recover)

Training data: output/bart_query_pairs.jsonl  (split=train)
Eval data:     output/bart_query_pairs.jsonl  (split=eval)

Architecture choice: BART-base (same as victim)
  - BART was pre-trained with a denoising/reconstruction objective
  - Reconstruction is BART's native task — perfect fit for inversion
  - Same tokenizer as victim → shared vocabulary → easier to learn inverse

Run:
    python3 train_inverter.py

Saves best checkpoint to: inverter_checkpoint/
"""

import os
import sys
import json
import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INVERTER_MODEL_NAME,
    INVERTER_EPOCHS, INVERTER_BATCH_SIZE, INVERTER_EVAL_BATCH,
    INVERTER_LR, INVERTER_MAX_INPUT, INVERTER_MAX_TARGET,
    INVERTER_WARMUP, INVERTER_GRAD_ACCUM, INVERTER_NUM_WORKERS,
    INVERTER_CHECKPOINT_DIR,
    BART_PAIRS_FILE, INVERTER_TRAIN_FILE, INVERTER_EVAL_FILE,
    OUTPUT_DIR, LOGS_DIR,
)

# ── logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(INVERTER_CHECKPOINT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "train_inverter.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def prepare_splits(pairs_file: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load bart_query_pairs.jsonl and split by the 'split' field.
    Writes separate train/eval jsonl files for easy inspection.
    """
    train, eval_ = [], []
    with open(pairs_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Only use pairs where both original and anonymized are non-empty
            if not d.get("original", "").strip() or not d.get("anonymized", "").strip():
                continue
            if d.get("split") == "eval":
                eval_.append(d)
            else:
                train.append(d)

    logger.info(f"  Split: {len(train):,} train | {len(eval_):,} eval")

    # Write splits for inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(INVERTER_TRAIN_FILE, "w", encoding="utf-8") as f:
        for d in train:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(INVERTER_EVAL_FILE, "w", encoding="utf-8") as f:
        for d in eval_:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return train, eval_


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════

class InverterDataset(Dataset):
    """
    Input  → anonymized text  (what BART outputs / attacker sees)
    Target → original text    (what attacker wants to recover)
    """

    def __init__(self, pairs: List[Dict], tokenizer, max_input: int, max_target: int):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src  = pair["anonymized"].strip()   # INPUT to inverter
        tgt  = pair["original"].strip()     # TARGET of inverter

        enc = self.tokenizer(
            src,
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # as_target_tokenizer() removed in transformers v4.26+ — BART shares
        # the same tokenizer for encoder and decoder so it was always a no-op
        dec = self.tokenizer(
            tgt,
            max_length=self.max_target,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = dec["input_ids"].squeeze(0).clone()
        # Mask padding tokens in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
            # Metadata for analysis
            "strategy":       pair.get("strategy", ""),
            "probe_entity":   pair.get("probe_entity", ""),
            "name_rarity":    pair.get("name_rarity", ""),
        }


def collate_fn(batch):
    """Custom collate to handle string metadata fields."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "strategy":       [b["strategy"]     for b in batch],
        "probe_entity":   [b["probe_entity"] for b in batch],
        "name_rarity":    [b["name_rarity"]  for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_entity_recovery_rate(
    predictions: List[str],
    originals: List[str],
    probe_entities: List[str],
) -> Dict:
    """
    Entity Recovery Rate (ERR) — primary attack success metric.
    Measures what fraction of original PII entities the inverter
    recovers exactly (verbatim) in its output.
    """
    exact_matches    = 0
    partial_matches  = 0
    total            = 0

    for pred, orig, entity in zip(predictions, originals, probe_entities):
        if not entity:
            continue
        total += 1
        if entity in pred:
            exact_matches += 1
        elif any(part in pred for part in entity.split() if len(part) > 2):
            partial_matches += 1

    if total == 0:
        return {"err_exact": 0.0, "err_partial": 0.0, "total": 0}

    return {
        "err_exact":   round(exact_matches   / total, 4),
        "err_partial": round(partial_matches / total, 4),
        "total":       total,
    }


def compute_token_accuracy(
    predictions: List[str],
    references: List[str],
    tokenizer,
) -> float:
    """Simple token-level accuracy between prediction and reference."""
    correct = total = 0
    for pred, ref in zip(predictions, references):
        p_tokens = tokenizer.tokenize(pred)
        r_tokens = tokenizer.tokenize(ref)
        min_len  = min(len(p_tokens), len(r_tokens))
        if min_len == 0:
            continue
        matches = sum(p == r for p, r in zip(p_tokens[:min_len], r_tokens[:min_len]))
        correct += matches
        total   += max(len(p_tokens), len(r_tokens))
    return round(correct / total, 4) if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class InverterTrainer:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        device: torch.device,
    ):
        self.model        = model
        self.tokenizer    = tokenizer
        self.train_loader = train_loader
        self.eval_loader  = eval_loader
        self.device       = device

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": 1e-2},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(params, lr=INVERTER_LR)

        # Scheduler
        total_steps = (len(train_loader) // INVERTER_GRAD_ACCUM) * INVERTER_EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=INVERTER_WARMUP,
            num_training_steps=total_steps,
        )

        self.scaler          = GradScaler("cuda")   # fp16 mixed precision
        self.best_eval_loss  = float("inf")
        self.history         = []
        self.global_step     = 0

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        steps      = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"  Epoch {epoch} [train]",
                    dynamic_ncols=True, leave=False)

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            with autocast("cuda"):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / INVERTER_GRAD_ACCUM
            self.scaler.scale(loss).backward()

            if (step + 1) % INVERTER_GRAD_ACCUM == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += outputs.loss.item()
            steps      += 1
            pbar.set_postfix({"loss": f"{total_loss/steps:.4f}", "step": self.global_step})

        return {"train_loss": round(total_loss / steps, 4), "steps": steps}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict:
        self.model.eval()
        total_loss = 0.0
        steps      = 0

        all_preds       = []
        all_originals   = []
        all_entities    = []
        all_strategies  = []
        all_rarities    = []

        pbar = tqdm(self.eval_loader, desc=f"  Epoch {epoch} [eval ]",
                    dynamic_ncols=True, leave=False)

        for batch in pbar:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            # Loss
            with autocast("cuda"):
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()
            steps      += 1

            # Generate predictions
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=INVERTER_MAX_TARGET,
                num_beams=4,
                early_stopping=True,
            )
            preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            refs  = self.tokenizer.batch_decode(
                labels.masked_fill(labels == -100, self.tokenizer.pad_token_id),
                skip_special_tokens=True,
            )

            all_preds.extend(preds)
            all_originals.extend(refs)
            all_entities.extend(batch["probe_entity"])
            all_strategies.extend(batch["strategy"])
            all_rarities.extend(batch["name_rarity"])

        eval_loss = total_loss / steps

        # ERR overall
        err = compute_entity_recovery_rate(all_preds, all_originals, all_entities)

        # ERR per strategy
        err_by_strategy = {}
        from collections import defaultdict
        strat_groups = defaultdict(lambda: {"preds": [], "origs": [], "ents": []})
        for pred, orig, ent, strat in zip(all_preds, all_originals, all_entities, all_strategies):
            strat_groups[strat]["preds"].append(pred)
            strat_groups[strat]["origs"].append(orig)
            strat_groups[strat]["ents"].append(ent)
        for strat, g in strat_groups.items():
            err_by_strategy[strat] = compute_entity_recovery_rate(
                g["preds"], g["origs"], g["ents"])

        # ERR per rarity tier
        err_by_rarity = {}
        rarity_groups = defaultdict(lambda: {"preds": [], "origs": [], "ents": []})
        for pred, orig, ent, rar in zip(all_preds, all_originals, all_entities, all_rarities):
            rarity_groups[rar or "unknown"]["preds"].append(pred)
            rarity_groups[rar or "unknown"]["origs"].append(orig)
            rarity_groups[rar or "unknown"]["ents"].append(ent)
        for rar, g in rarity_groups.items():
            err_by_rarity[rar] = compute_entity_recovery_rate(
                g["preds"], g["origs"], g["ents"])

        # Token accuracy
        tok_acc = compute_token_accuracy(all_preds, all_originals, self.tokenizer)

        # Sample predictions for logging
        samples = list(zip(all_preds[:5], all_originals[:5], all_entities[:5]))

        return {
            "eval_loss":        round(eval_loss, 4),
            "perplexity":       round(math.exp(min(eval_loss, 20)), 2),
            "err_exact":        err["err_exact"],
            "err_partial":      err["err_partial"],
            "token_accuracy":   tok_acc,
            "err_by_strategy":  err_by_strategy,
            "err_by_rarity":    err_by_rarity,
            "n_eval":           err["total"],
            "samples":          samples,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool):
        ckpt = {
            "epoch":            epoch,
            "global_step":      self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "metrics":          metrics,
            "saved_at":         datetime.now().isoformat(),
        }
        # Always save latest
        path = os.path.join(INVERTER_CHECKPOINT_DIR, "latest_checkpoint.pt")
        torch.save(ckpt, path)

        if is_best:
            best_path = os.path.join(INVERTER_CHECKPOINT_DIR, "best_model.pt")
            torch.save(ckpt, best_path)
            logger.info(f"  ✅ New best model saved (eval_loss={metrics['eval_loss']:.4f})")

        # Save training history
        self.history.append({"epoch": epoch, **metrics})
        hist_path = os.path.join(INVERTER_CHECKPOINT_DIR, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self) -> List[Dict]:
        logger.info(f"  Starting training: {INVERTER_EPOCHS} epochs")
        logger.info(f"  Train batches/epoch: {len(self.train_loader):,}")
        logger.info(f"  Eval batches:        {len(self.eval_loader):,}")
        logger.info(f"  Grad accum steps:    {INVERTER_GRAD_ACCUM}")
        logger.info(f"  Effective batch:     {INVERTER_BATCH_SIZE * INVERTER_GRAD_ACCUM}")
        logger.info(f"  Learning rate:       {INVERTER_LR}")

        for epoch in range(1, INVERTER_EPOCHS + 1):
            epoch_start = time.time()
            logger.info(f"\n{'─'*60}")
            logger.info(f"  EPOCH {epoch} / {INVERTER_EPOCHS}")
            logger.info(f"{'─'*60}")

            train_metrics = self.train_epoch(epoch)
            eval_metrics  = self.evaluate(epoch)

            elapsed = time.time() - epoch_start
            is_best = eval_metrics["eval_loss"] < self.best_eval_loss
            if is_best:
                self.best_eval_loss = eval_metrics["eval_loss"]

            self.save_checkpoint(epoch, {**train_metrics, **eval_metrics}, is_best)

            # Log
            logger.info(
                f"  Epoch {epoch}: "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"eval_loss={eval_metrics['eval_loss']:.4f} | "
                f"PPL={eval_metrics['perplexity']:.1f} | "
                f"ERR_exact={eval_metrics['err_exact']:.3f} | "
                f"ERR_partial={eval_metrics['err_partial']:.3f} | "
                f"tok_acc={eval_metrics['token_accuracy']:.3f} | "
                f"time={elapsed:.0f}s"
                + (" ⭐ BEST" if is_best else "")
            )

            # Log per-strategy ERR
            logger.info("  ERR by strategy:")
            for s, e in eval_metrics["err_by_strategy"].items():
                logger.info(f"    {s:<35} exact={e['err_exact']:.3f}  partial={e['err_partial']:.3f}")

            # Log per-rarity ERR
            logger.info("  ERR by rarity:")
            for r, e in eval_metrics["err_by_rarity"].items():
                logger.info(f"    {r:<15} exact={e['err_exact']:.3f}  partial={e['err_partial']:.3f}")

            # Sample predictions
            logger.info("  Sample predictions:")
            for pred, orig, ent in eval_metrics.get("samples", [])[:3]:
                logger.info(f"    ORIG  : {orig[:80]}")
                logger.info(f"    PRED  : {pred[:80]}")
                logger.info(f"    ENTITY: {ent}")
                logger.info("")

        return self.history


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  MODEL INVERSION — STEP 3: INVERTER TRAINING (BART-base)")
    print("=" * 70)

    # ── Check input ────────────────────────────────────────────────────────
    if not os.path.exists(BART_PAIRS_FILE):
        print(f"  ❌  Pairs file not found: {BART_PAIRS_FILE}")
        print("  Run: python3 query_bart.py first.")
        return

    # ── Prepare data ───────────────────────────────────────────────────────
    logger.info("  Loading and splitting pairs...")
    train_pairs, eval_pairs = prepare_splits(BART_PAIRS_FILE)

    if len(train_pairs) == 0:
        print("  ❌  No training pairs found. Check split field in pairs file.")
        return

    print(f"  Train pairs: {len(train_pairs):,}")
    print(f"  Eval pairs:  {len(eval_pairs):,}")
    print()

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info(f"  Loading {INVERTER_MODEL_NAME} (inverter)...")
    tokenizer = AutoTokenizer.from_pretrained(INVERTER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(INVERTER_MODEL_NAME)
    model = model.to(device)
    model.gradient_checkpointing_enable()  # trades compute for VRAM

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Model: {n_params:.1f}M parameters")

    # ── Resume from checkpoint if exists ────────────────────────────────────
    start_epoch = 1
    latest_ckpt = os.path.join(INVERTER_CHECKPOINT_DIR, "latest_checkpoint.pt")
    if os.path.exists(latest_ckpt):
        logger.info(f"  Resuming from {latest_ckpt}")
        # Load to CPU first — GPU is already holding the model weights.
        # Loading a second copy of the 139M-param checkpoint directly to
        # CUDA would exceed the 3.7 GB VRAM budget and raise OOM.
        torch.cuda.empty_cache()
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"  Resuming from epoch {start_epoch}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_ds = InverterDataset(train_pairs, tokenizer, INVERTER_MAX_INPUT, INVERTER_MAX_TARGET)
    eval_ds  = InverterDataset(eval_pairs,  tokenizer, INVERTER_MAX_INPUT, INVERTER_MAX_TARGET)

    train_loader = DataLoader(
        train_ds, batch_size=INVERTER_BATCH_SIZE, shuffle=True,
        num_workers=INVERTER_NUM_WORKERS, collate_fn=collate_fn, pin_memory=(device.type=="cuda"),
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=INVERTER_EVAL_BATCH, shuffle=False,
        num_workers=INVERTER_NUM_WORKERS, collate_fn=collate_fn, pin_memory=(device.type=="cuda"),
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print()
    input("  Press Enter to start training (Ctrl+C to cancel)...")
    print()

    trainer = InverterTrainer(model, tokenizer, train_loader, eval_loader, device)
    if start_epoch > 1 and os.path.exists(latest_ckpt):
        # Reuse the already-loaded ckpt dict if available, otherwise reload to CPU.
        if "ckpt" not in dir():
            torch.cuda.empty_cache()
            ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        # Optimizer states are CPU tensors — move them to the right device explicitly
        opt_state = ckpt["optimizer_state"]
        trainer.optimizer.load_state_dict(opt_state)
        # Move optimizer state tensors to GPU
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        trainer.global_step = ckpt["global_step"]
        trainer.best_eval_loss = ckpt["metrics"].get("eval_loss", float("inf"))

    history = trainer.train()

    # ── Summary ──────────────────────────────────────────────────────────────
    best = min(history, key=lambda x: x.get("eval_loss", float("inf")))
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best epoch:          {best.get('epoch')}")
    print(f"  Best eval loss:      {best.get('eval_loss'):.4f}")
    print(f"  Best perplexity:     {best.get('perplexity'):.1f}")
    print(f"  Best ERR (exact):    {best.get('err_exact'):.3f}  ({best.get('err_exact',0)*100:.1f}% of PII entities recovered)")
    print(f"  Best ERR (partial):  {best.get('err_partial'):.3f}")
    print(f"  Best token accuracy: {best.get('token_accuracy'):.3f}")
    print()
    print(f"  Checkpoint: {os.path.join(INVERTER_CHECKPOINT_DIR, 'best_model.pt')}")
    print()
    print("  Next step: python3 evaluate_attack.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
