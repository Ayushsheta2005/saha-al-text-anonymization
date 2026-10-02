#!/usr/bin/env python3
"""
Adversarial Training -- BART-base PII Anonymizer
=================================================
The victim BART model is already well-trained on 93k normal anonymization
pairs.  We do NOT re-iterate over that data.

Instead we use ONLY the 38k adversarial pairs from the model-inversion query
step (inverter_train.jsonl).  Each pair (original, anonymized) drives BOTH
losses in a single combined forward pass:

  original_text --> [VICTIM BART encoder]
                          |
          teacher-force on anonymized_text
                          v
              victim_logits  (B, T, V) fp32
                   |                |
                   v                v
  L1 = CE(victim_logits, anonymized_labels)
  [PII replaced correctly + output stays fluent]
  [CE against a fluent reference = conditional PPL]
                                    |
  soft_embs = softmax(logits/T) @ inverter.embed_W   (fp16)
                                    |
            [FROZEN INVERTER encoder on soft_embs]
            [FROZEN INVERTER decoder, teacher-forced on original]
  L2 = CE(inv_logits.float(), original_labels)
                                    |
  L_total = alpha*L1  -  lambda*L2  [NEGATED -- fool the inverter]

  If the victim escapes to gibberish to fool the inverter, L1 (CE against
  a fluent reference) rises -- preventing degenerate output.

Gradient path through the frozen inverter:
  L2 -> inv_decoder -> inv_encoder -> soft_embs (fp16)
     -> victim_logits (fp32) -> victim weights  checkmark
  Inverter params have requires_grad=False (no .grad accumulated),
  but autograd still propagates the signal THROUGH their ops.

Run:
    cd adversarial_training
    python3 train_adv.py

Saves: adversarial_training/checkpoints/bart-base-adv/best_model.pt
Original Seq2Seq_model/checkpoints/ is NEVER modified.
"""

import os
import sys
import gc
import json
import time
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    VICTIM_MODEL_NAME, VICTIM_CHECKPOINT,
    INVERTER_MODEL_NAME, INVERTER_CHECKPOINT,
    ADV_TRAIN_FILE, ADV_EVAL_FILE,
    GOLD_TRAIN_FILE, GOLD_VAL_FILE, GOLD_MAX_SAMPLES,
    ADV_CHECKPOINT_DIR, LOGS_DIR, RESULTS_DIR,
    ALPHA, LAMBDA_ADV, SOFT_TEMP,
    NUM_EPOCHS, BATCH_SIZE, EVAL_BATCH_SIZE, GRAD_ACCUM_STEPS,
    LEARNING_RATE, WARMUP_STEPS, MAX_GRAD_NORM, WEIGHT_DECAY,
    LABEL_SMOOTHING, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    NUM_WORKERS, EVAL_STEPS, LOGGING_STEPS,
)
from dataset import AdvDataset, adv_collate_fn, GoldDataset, gold_collate_fn, load_jsonl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ADV_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "train_adv.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return f"{alloc:.0f}/{total:.0f}MB"


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# ---------------------------------------------------------------------------
# Label-smoothed CE (cast to fp32 internally -- safe over V=50265 tokens)
# ---------------------------------------------------------------------------

class LabelSmoothedCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing    = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        vocab_size  = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size).float()
        labels_flat = labels.reshape(-1)
        mask        = labels_flat != self.ignore_index

        log_probs = F.log_softmax(logits_flat, dim=-1)
        nll    = F.nll_loss(log_probs, labels_flat.clamp(min=0),
                            ignore_index=self.ignore_index, reduction="sum")
        smooth = -log_probs[mask].sum() / vocab_size
        count  = mask.sum().float().clamp(min=1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth) / count


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_victim(device: torch.device) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load BART-base victim from original Seq2Seq checkpoint (fp32, trainable)."""
    logger.info("Loading victim model: %s", VICTIM_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(VICTIM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        VICTIM_MODEL_NAME, torch_dtype=torch.float32
    )
    model.gradient_checkpointing_enable()

    if os.path.exists(VICTIM_CHECKPOINT):
        logger.info("  Loading victim weights from: %s", VICTIM_CHECKPOINT)
        ckpt  = torch.load(VICTIM_CHECKPOINT, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("  Missing keys (%d): %s", len(missing), missing[:5])
        logger.info("  Victim loaded. Original checkpoint untouched.")
    else:
        logger.warning("  Victim checkpoint not found -- using pretrained weights!")

    model = model.to(device)
    model.train()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("  Victim: %.1fM params (fp32, trainable) | GPU: %s", n, gpu_info())
    return model, tokenizer


def load_frozen_inverter(device: torch.device) -> AutoModelForSeq2SeqLM:
    """
    Load the trained inverter in fp16, fully frozen.

    fp16 saves ~260MB VRAM vs fp32.  Inverter weights are never updated so
    reduced precision is fine.  CE loss (L2) casts logits to fp32 before
    log_softmax to avoid overflow over V=50265.

    Gradient propagation still works:
      L2 flows back through inverter ops -> soft_embs -> victim logits
      (requires_grad=False means no .grad tensors for inverter params,
       but autograd still propagates THROUGH them.)
    """
    logger.info("Loading frozen inverter (fp16): %s", INVERTER_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INVERTER_MODEL_NAME, torch_dtype=torch.float16
    )

    if os.path.exists(INVERTER_CHECKPOINT):
        logger.info("  Loading inverter weights from: %s", INVERTER_CHECKPOINT)
        ckpt  = torch.load(INVERTER_CHECKPOINT, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info("  Inverter loaded.")
    else:
        logger.warning("  Inverter checkpoint not found -- using pretrained weights!")

    for param in model.parameters():
        param.requires_grad_(False)

    model = model.to(device)
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("  Inverter: %.1fM params FROZEN (fp16) | GPU: %s", n, gpu_info())
    return model


# ---------------------------------------------------------------------------
# Loss functions  (L1 from gold, L2 from adversarial pairs)
# ---------------------------------------------------------------------------

def compute_l1_loss(
    victim: AutoModelForSeq2SeqLM,
    loss_fn: LabelSmoothedCE,
    batch: Dict,
    device: torch.device,
) -> torch.Tensor:
    """
    L1 = CE(victim(original_text), gold_anonymized_labels)
    Uses gold benchmark references — no circularity with BART's own outputs.
    """
    orig_ids    = batch["orig_input_ids"].to(device)
    orig_mask   = batch["orig_attention_mask"].to(device)
    anon_labels = batch["anon_labels"].to(device)
    out = victim(input_ids=orig_ids, attention_mask=orig_mask, labels=anon_labels)
    return loss_fn(out.logits, anon_labels)


def compute_l2_loss(
    victim: AutoModelForSeq2SeqLM,
    inverter: AutoModelForSeq2SeqLM,
    loss_fn: LabelSmoothedCE,
    batch: Dict,
    device: torch.device,
) -> torch.Tensor:
    """
    L2 = CE(frozen_inverter(soft_victim_output), original_labels)
    Uses adversarial pairs.  Caller negates with -LAMBDA_ADV before backprop.

    Gradient path:
      L2 -> inv_decoder -> inv_encoder -> soft_embs (fp16)
          -> victim_logits (fp32) -> victim params
    """
    orig_ids    = batch["orig_input_ids"].to(device)
    orig_mask   = batch["orig_attention_mask"].to(device)
    anon_mask   = batch["anon_attention_mask"].to(device)
    anon_labels = batch["anon_labels"].to(device)
    inv_labels  = batch["inv_labels"].to(device)

    victim_out = victim(input_ids=orig_ids, attention_mask=orig_mask, labels=anon_labels)
    logits = victim_out.logits  # (B, T, V) fp32, requires_grad=True

    embed_W   = inverter.model.shared.weight                          # (V, D) fp16
    soft_embs = torch.softmax(logits.float() / SOFT_TEMP, dim=-1).half() @ embed_W

    enc_out = inverter.model.encoder(inputs_embeds=soft_embs, attention_mask=anon_mask)

    pad_id = inverter.config.pad_token_id
    bos_id = (inverter.config.decoder_start_token_id
               or inverter.config.bos_token_id)
    inv_labels_clean  = inv_labels.masked_fill(inv_labels == -100, pad_id)
    decoder_input_ids = torch.cat([
        torch.full((inv_labels.size(0), 1), bos_id, dtype=torch.long, device=device),
        inv_labels_clean[:, :-1],
    ], dim=1)
    dec_out = inverter.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=enc_out.last_hidden_state,
        encoder_attention_mask=anon_mask,
    )
    return loss_fn(inverter.lm_head(dec_out.last_hidden_state).float(), inv_labels)


# ---------------------------------------------------------------------------
# Validation on adversarial eval split
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    victim: AutoModelForSeq2SeqLM,
    inverter: AutoModelForSeq2SeqLM,
    adv_val_loader: DataLoader,
    gold_val_loader: DataLoader,
    tokenizer,
    device: torch.device,
    loss_fn: LabelSmoothedCE,
    max_batches: int = 50,
) -> Dict:
    """
    Two-phase evaluation:
      Phase 1 (gold_val_loader) → val_L1 (CE vs gold references) + tok_acc
      Phase 2 (adv_val_loader)  → val_L2 (inverter CE) + leakage_rate
    """
    victim.eval()

    # ── Phase 1: L1 from gold validation split ────────────────────────────
    total_L1 = 0.0
    n_gold   = 0
    all_preds, all_gold_refs = [], []

    for i, batch in enumerate(gold_val_loader):
        if i >= max_batches:
            break
        orig_ids    = batch["orig_input_ids"].to(device)
        orig_mask   = batch["orig_attention_mask"].to(device)
        anon_labels = batch["anon_labels"].to(device)

        out = victim(input_ids=orig_ids, attention_mask=orig_mask, labels=anon_labels)
        total_L1 += loss_fn(out.logits, anon_labels).item()
        n_gold   += 1

        if i < 20:
            gen_ids = victim.generate(
                input_ids=orig_ids, attention_mask=orig_mask,
                max_length=MAX_TARGET_LENGTH, num_beams=1, do_sample=False,
            )
            all_preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            label_ids = anon_labels.masked_fill(anon_labels == -100, tokenizer.pad_token_id)
            all_gold_refs.extend(tokenizer.batch_decode(label_ids, skip_special_tokens=True))

        del orig_ids, orig_mask, anon_labels

    # ── Phase 2: L2 + leakage from adversarial validation split ──────────
    total_L2 = 0.0
    n_adv    = 0
    adv_preds, all_probes = [], []

    for i, batch in enumerate(adv_val_loader):
        if i >= max_batches:
            break
        orig_ids    = batch["orig_input_ids"].to(device)
        orig_mask   = batch["orig_attention_mask"].to(device)
        anon_mask   = batch["anon_attention_mask"].to(device)
        anon_labels = batch["anon_labels"].to(device)
        inv_labels  = batch["inv_labels"].to(device)

        victim_out = victim(input_ids=orig_ids, attention_mask=orig_mask, labels=anon_labels)
        logits     = victim_out.logits

        embed_W   = inverter.model.shared.weight
        soft_embs = torch.softmax(logits.float() / SOFT_TEMP, dim=-1).half() @ embed_W
        enc_out   = inverter.model.encoder(inputs_embeds=soft_embs, attention_mask=anon_mask)

        pad_id = inverter.config.pad_token_id
        bos_id = (inverter.config.decoder_start_token_id
                   or inverter.config.bos_token_id)
        inv_clean = inv_labels.masked_fill(inv_labels == -100, pad_id)
        dec_in    = torch.cat([
            torch.full((inv_labels.size(0), 1), bos_id, dtype=torch.long, device=device),
            inv_clean[:, :-1],
        ], dim=1)
        dec_out = inverter.model.decoder(
            input_ids=dec_in,
            encoder_hidden_states=enc_out.last_hidden_state,
            encoder_attention_mask=anon_mask,
        )
        total_L2 += loss_fn(
            inverter.lm_head(dec_out.last_hidden_state).float(), inv_labels
        ).item()
        n_adv += 1

        if i < 20:
            gen_ids = victim.generate(
                input_ids=orig_ids, attention_mask=orig_mask,
                max_length=MAX_TARGET_LENGTH, num_beams=1, do_sample=False,
            )
            adv_preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            all_probes.extend(batch["probe_entity"])

        del orig_ids, orig_mask, anon_mask, anon_labels, inv_labels

    # ── Metrics ───────────────────────────────────────────────────────────
    n_g    = max(n_gold, 1)
    n_a    = max(n_adv, 1)
    val_L1 = round(total_L1 / n_g, 4)
    val_L2 = round(total_L2 / n_a, 4)

    def _tok_acc(preds, refs):
        c = t = 0
        for p, r in zip(preds, refs):
            pt = tokenizer.tokenize(p); rt = tokenizer.tokenize(r)
            ml = min(len(pt), len(rt))
            if ml == 0:
                continue
            c += sum(a == b for a, b in zip(pt[:ml], rt[:ml]))
            t += max(len(pt), len(rt))
        return round(c / t, 4) if t else 0.0

    def _leakage(preds, entities):
        leaked = total = 0
        for pred, ent in zip(preds, entities):
            if not ent:
                continue
            total += 1
            if ent.lower() in pred.lower():
                leaked += 1
        return round(leaked / max(total, 1), 4)

    victim.train()
    return {
        "val_L1":       val_L1,
        "val_L2":       val_L2,
        "val_L_total":  round(ALPHA * val_L1 - LAMBDA_ADV * val_L2, 4),
        "tok_acc":      _tok_acc(all_preds, all_gold_refs),
        "leakage_rate": _leakage(adv_preds, all_probes),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _build_ckpt(victim, optimizer, scheduler, scaler,
                epoch, batch_idx, global_step, best_val_loss):
    ckpt = {
        "epoch":                epoch,
        "batch_idx":            batch_idx,   # last batch completed in this epoch
        "global_step":          global_step,
        "best_val_loss":        best_val_loss,
        "model_state_dict":     victim.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp":            datetime.now().isoformat(),
        "config": {
            "ALPHA": ALPHA, "LAMBDA_ADV": LAMBDA_ADV,
            "SOFT_TEMP": SOFT_TEMP, "LABEL_SMOOTHING": LABEL_SMOOTHING,
            "BATCH_SIZE": BATCH_SIZE,
        },
    }
    if scheduler:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    return ckpt


def save_checkpoint(victim, optimizer, scheduler, scaler,
                    epoch, batch_idx, global_step, best_val_loss):
    """Save best-model checkpoint (kept when val_L_total improves)."""
    ckpt = _build_ckpt(victim, optimizer, scheduler, scaler,
                       epoch, batch_idx, global_step, best_val_loss)
    path = os.path.join(ADV_CHECKPOINT_DIR, "best_model.pt")
    torch.save(ckpt, path)
    logger.info("  Best checkpoint -> %s  (step=%d, val_L_total=%.4f)",
                path, global_step, best_val_loss)


def save_latest_checkpoint(victim, optimizer, scheduler, scaler,
                           epoch, batch_idx, global_step, best_val_loss):
    """Periodic safety checkpoint -- always overwritten with the latest state."""
    ckpt = _build_ckpt(victim, optimizer, scheduler, scaler,
                       epoch, batch_idx, global_step, best_val_loss)
    path = os.path.join(ADV_CHECKPOINT_DIR, "latest_checkpoint.pt")
    torch.save(ckpt, path)
    logger.info("  Latest checkpoint -> %s  (step=%d)", path, global_step)


def save_history(history: dict):
    with open(os.path.join(ADV_CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


def resume_if_available() -> Optional[Dict]:
    """
    Prefer latest_checkpoint.pt (has the most recent step) over best_model.pt.
    latest_checkpoint also stores batch_idx so we can skip already-processed
    batches and resume mid-epoch.
    """
    latest_path = os.path.join(ADV_CHECKPOINT_DIR, "latest_checkpoint.pt")
    best_path   = os.path.join(ADV_CHECKPOINT_DIR, "best_model.pt")

    for path, label in [(latest_path, "latest"), (best_path, "best")]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            logger.info(
                "Resuming from %s checkpoint (step=%d, epoch=%d, batch=%d, "
                "val_L_total=%.4f)",
                label,
                ckpt.get("global_step", 0),
                ckpt.get("epoch", 0),
                ckpt.get("batch_idx", -1),
                ckpt.get("best_val_loss", float("inf")),
            )
            return ckpt
    return None


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("ADVERSARIAL TRAINING -- BART-base PII Anonymizer")
    logger.info("Device: %s | GPU: %s", device, gpu_info())
    logger.info("Loss: %.1f*L1  -  %.2f*L2  |  SOFT_TEMP=%.1f",
                ALPHA, LAMBDA_ADV, SOFT_TEMP)
    logger.info("Training data: L2=adv pairs (inverter_train.jsonl) | L1=gold refs (%d samples)",
                GOLD_MAX_SAMPLES)
    logger.info("=" * 70)

    # -- 1. Models -----------------------------------------------------------
    victim, tokenizer = load_victim(device)
    inverter          = load_frozen_inverter(device)
    loss_fn           = LabelSmoothedCE(smoothing=LABEL_SMOOTHING)

    # -- 2. Data  ------------------------------------------------------------
    logger.info("Loading adversarial pairs (L2) ...")
    adv_train_data = load_jsonl(ADV_TRAIN_FILE)
    adv_eval_data  = load_jsonl(ADV_EVAL_FILE)
    logger.info("  Adv train: %d | Adv eval: %d", len(adv_train_data), len(adv_eval_data))

    logger.info("Loading gold pairs (L1, max %d) ...", GOLD_MAX_SAMPLES)
    gold_train_data = load_jsonl(GOLD_TRAIN_FILE)
    gold_val_data   = load_jsonl(GOLD_VAL_FILE)
    logger.info("  Gold pool: %d train | %d val", len(gold_train_data), len(gold_val_data))

    adv_ds        = AdvDataset(adv_train_data,  tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    adv_eval_ds   = AdvDataset(adv_eval_data,   tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    gold_train_ds = GoldDataset(gold_train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
                                max_samples=GOLD_MAX_SAMPLES)
    gold_eval_ds  = GoldDataset(gold_val_data,   tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
                                max_samples=500)

    train_loader = DataLoader(
        adv_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=adv_collate_fn,
    )
    eval_loader = DataLoader(
        adv_eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        collate_fn=adv_collate_fn,
    )
    gold_loader = DataLoader(
        gold_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=gold_collate_fn,
    )
    gold_eval_loader = DataLoader(
        gold_eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        collate_fn=gold_collate_fn,
    )

    iters_per_epoch = len(train_loader)   # driven by adv (38k)
    opt_steps_epoch = iters_per_epoch // GRAD_ACCUM_STEPS
    total_steps     = opt_steps_epoch * NUM_EPOCHS
    logger.info(
        "  Adv iters/epoch: %d | Gold cycles/epoch: %.1f | "
        "Opt steps/epoch: %d | Total: %d | Eff batch: %d",
        iters_per_epoch, iters_per_epoch / max(len(gold_loader), 1),
        opt_steps_epoch, total_steps, BATCH_SIZE * GRAD_ACCUM_STEPS,
    )

    # -- 3. Optimiser & scheduler --------------------------------------------
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        {"params": [p for n, p in victim.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in victim.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
        num_training_steps=total_steps,
    )
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # -- 4. Resume -----------------------------------------------------------
    best_val_loss  = float("inf")
    global_step    = 0
    start_epoch    = 0
    skip_batches   = 0   # how many batches to skip at the start of start_epoch

    existing = resume_if_available()
    if existing is not None:
        victim.load_state_dict(existing["model_state_dict"])
        optimizer.load_state_dict(existing["optimizer_state_dict"])
        if "scheduler_state_dict" in existing and scheduler:
            scheduler.load_state_dict(existing["scheduler_state_dict"])
        if "scaler_state_dict" in existing and scaler:
            scaler.load_state_dict(existing["scaler_state_dict"])
        best_val_loss = existing.get("best_val_loss", float("inf"))
        global_step   = existing.get("global_step", 0)
        start_epoch   = existing.get("epoch", 0)
        saved_batch   = existing.get("batch_idx", -1)
        # Resume inside the epoch: skip the batches already processed.
        # (batch_idx is the last *completed* batch, so skip saved_batch+1 batches)
        if saved_batch >= 0:
            skip_batches = saved_batch + 1
        del existing
        aggressive_cleanup()
        if skip_batches:
            logger.info("  Will skip first %d batches of epoch %d to resume mid-epoch",
                        skip_batches, start_epoch + 1)

    # -- 5. History ----------------------------------------------------------
    history = {
        "train_L1": [], "train_L2": [], "train_L_total": [],
        "val_L1":   [], "val_L2":   [], "val_L_total":   [],
        "tok_acc":  [], "leakage_rate": [],
        "learning_rates": [], "best_val_loss": best_val_loss,
    }

    PERIODIC_SAVE_STEPS = 500   # save latest_checkpoint.pt every N optimizer steps

    # -- 6. Training loop ----------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0        = time.time()
        ep_L1     = ep_L2 = ep_Lt = 0.0
        ep_steps  = 0
        oom_count = 0

        optimizer.zero_grad(set_to_none=True)

        # How many batches to skip at the start of this epoch (mid-epoch resume)
        _skip = skip_batches if epoch == start_epoch else 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )
        if _skip:
            pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [skipping {_skip} batches]")

        # Cycling gold iterator — yield from gold_loader gives a fresh shuffle each cycle
        def _make_gold_iter():
            while True:
                yield from gold_loader
        gold_iter = _make_gold_iter()

        for batch_idx, adv_batch in pbar:
            # Skip already-processed batches when resuming mid-epoch
            if batch_idx < _skip:
                pbar.update(0)
                continue
            try:
                gold_batch = next(gold_iter)

                # --- L1: gold reference (backward frees graph before L2) --------
                L1       = compute_l1_loss(victim, loss_fn, gold_batch, device)
                l1       = L1.item()
                L1_scaled = ALPHA * L1 / GRAD_ACCUM_STEPS
                if scaler is not None:
                    scaler.scale(L1_scaled).backward()
                else:
                    L1_scaled.backward()
                del L1, L1_scaled, gold_batch

                # --- L2: adversarial inverter loss --------------------------------
                L2       = compute_l2_loss(victim, inverter, loss_fn, adv_batch, device)
                l2       = L2.item()
                L2_scaled = LAMBDA_ADV * L2 / GRAD_ACCUM_STEPS
                if scaler is not None:
                    scaler.scale(-L2_scaled).backward()
                else:
                    (-L2_scaled).backward()
                del L2, L2_scaled

                ep_L1    += l1
                ep_L2    += l2
                ep_Lt    += (ALPHA * l1 - LAMBDA_ADV * l2)
                ep_steps += 1

                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in victim.parameters() if p.requires_grad],
                        MAX_GRAD_NORM,
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    oom_count = 0  # reset only after a full successful optimizer step

                    # Periodic safety checkpoint (latest state, always overwritten)
                    if global_step % PERIODIC_SAVE_STEPS == 0:
                        save_latest_checkpoint(victim, optimizer, scheduler, scaler,
                                               epoch, batch_idx, global_step, best_val_loss)
                        save_history(history)

                    avg_l1 = ep_L1 / ep_steps
                    avg_l2 = ep_L2 / ep_steps
                    avg_lt = ep_Lt / ep_steps
                    lr     = scheduler.get_last_lr()[0]

                    pbar.set_postfix({
                        "L1": f"{avg_l1:.3f}",
                        "L2": f"{avg_l2:.3f}",
                        "Lt": f"{avg_lt:.3f}",
                        "lr": f"{lr:.1e}",
                        "GPU": gpu_info(),
                    })

                    if global_step % LOGGING_STEPS == 0:
                        logger.info(
                            "Step %d | L1=%.4f | L2=%.4f | L_total=%.4f | LR=%.2e | %s",
                            global_step, avg_l1, avg_l2, avg_lt, lr, gpu_info(),
                        )
                        history["train_L1"].append({"step": global_step, "v": avg_l1})
                        history["train_L2"].append({"step": global_step, "v": avg_l2})
                        history["train_L_total"].append({"step": global_step, "v": avg_lt})
                        history["learning_rates"].append({"step": global_step, "lr": lr})

                    if global_step % EVAL_STEPS == 0:
                        pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [eval]")
                        vm = evaluate(victim, inverter, eval_loader, gold_eval_loader,
                                      tokenizer, device, loss_fn, max_batches=50)
                        logger.info(
                            "  [EVAL %d] L1=%.4f | L2=%.4f | L_total=%.4f | "
                            "tok_acc=%.4f | leakage=%.4f",
                            global_step, vm["val_L1"], vm["val_L2"],
                            vm["val_L_total"], vm["tok_acc"], vm["leakage_rate"],
                        )
                        tqdm.write(
                            f"  [Step {global_step}] val_L1={vm['val_L1']:.4f} | "
                            f"val_L2={vm['val_L2']:.4f} | "
                            f"val_L_total={vm['val_L_total']:.4f} | "
                            f"tok_acc={vm['tok_acc']:.4f} | "
                            f"leakage={vm['leakage_rate']:.4f}"
                        )
                        for k in ("val_L1", "val_L2", "val_L_total",
                                  "tok_acc", "leakage_rate"):
                            history[k].append({"step": global_step, "v": vm[k]})

                        if vm["val_L_total"] < best_val_loss:
                            best_val_loss = vm["val_L_total"]
                            history["best_val_loss"] = best_val_loss
                            save_checkpoint(victim, optimizer, scheduler, scaler,
                                            epoch, batch_idx, global_step, best_val_loss)
                            tqdm.write(
                                f"  ** New best val_L_total={best_val_loss:.4f}"
                            )
                        save_history(history)
                        pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
                        victim.train()

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                logger.warning("  [OOM] batch %d (count=%d)", batch_idx, oom_count)
                tqdm.write(f"  [OOM] batch {batch_idx} count={oom_count}")
                aggressive_cleanup()
                optimizer.zero_grad(set_to_none=True)
                if oom_count >= 8:
                    logger.error("  [FATAL] Too many OOMs. Aborting.")
                    raise
                continue

        pbar.close()
        logger.info(
            "Epoch %d/%d | %.0fs | avg L1=%.4f | avg L2=%.4f | "
            "avg L_total=%.4f | best=%.4f",
            epoch + 1, NUM_EPOCHS, time.time() - t0,
            ep_L1 / max(ep_steps, 1), ep_L2 / max(ep_steps, 1),
            ep_Lt / max(ep_steps, 1), best_val_loss,
        )

    # -- 7. Final evaluation -------------------------------------------------
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION (full eval split)")
    logger.info("=" * 60)
    final = evaluate(victim, inverter, eval_loader, gold_eval_loader, tokenizer, device,
                     loss_fn, max_batches=999)
    logger.info(
        "Final | val_L1=%.4f | val_L2=%.4f | val_L_total=%.4f | "
        "tok_acc=%.4f | leakage=%.4f",
        final["val_L1"], final["val_L2"], final["val_L_total"],
        final["tok_acc"], final["leakage_rate"],
    )
    history["final_metrics"] = final
    save_history(history)

    torch.save(
        {
            "model_state_dict": victim.state_dict(),
            "global_step":      global_step,
            "final_metrics":    final,
            "timestamp":        datetime.now().isoformat(),
        },
        os.path.join(ADV_CHECKPOINT_DIR, "final_model.pt"),
    )
    logger.info("Final model -> %s/final_model.pt", ADV_CHECKPOINT_DIR)
    logger.info("Adversarial training complete.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
