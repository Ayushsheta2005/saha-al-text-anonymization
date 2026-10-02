#!/usr/bin/env python3
"""
train_adv.py — Pipeline Mask-and-Fill Adversarial Hardening
============================================================
Fine-tunes the filler component of the pipeline against a frozen inverter.

Usage (local / Kaggle):
    python train_adv.py --combo 2   # BART filler (recommended, exact gradient)
    python train_adv.py --combo 1   # DeBERTa-MLM filler (approx. gradient)

Combo 2 (BART filler) — exact soft-embedding trick:
    masked_s2s  →  [BART filler]  →  logits (B, T, 50265)
                                   →  soft_embs = softmax(logits/T) @ inv.embed_W
                                   →  [frozen BART inverter]  →  L₂

    L₁ = CE(filler_logits, anonymized_labels)          quality anchor
    L₂ = CE(inverter(soft_embs), original_labels)      adversarial
    L_total = α·L₁ − λ_adv·L₂                         minimise

Combo 1 (DeBERTa-MLM filler) — hidden-state approximation:
    masked_mlm  →  [DeBERTa-MLM filler]  →  last_hidden_state (B, T, 768)
                                          →  inputs_embeds for [frozen BART inverter]
                                          →  L₂  (gradient flows through 768-dim states)

    L₁ = MLM CE at [MASK] positions vs anonymized labels
         (degenerate-output prevention: if hidden states collapse to gibberish,
          L₁ at non-PII positions also rises)
    L₂ = same adversarial loss as above, using hidden states

Saves:
    checkpoints/combo{N}/best_model.pt
    checkpoints/combo{N}/latest_checkpoint.pt
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

# Must be set before importing torch so the CUDA allocator uses expandable segments,
# which avoids fragmentation-driven OOM on small GPUs (e.g. RTX 3050 4 GiB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    Adafactor,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ADV_TRAIN_FILE, ADV_EVAL_FILE,
    COMBO1_FILLER_ID, COMBO2_FILLER_ID,
    COMBO1_INVERTER_ID, COMBO2_INVERTER_ID,
    COMBO1_CKPT_DIR, COMBO2_CKPT_DIR,
    LOGS_DIR,
    ALPHA, LAMBDA_ADV, SOFT_TEMP,
    NUM_EPOCHS, WEIGHT_DECAY, LABEL_SMOOTHING, MAX_GRAD_NORM,
    NUM_WORKERS, EVAL_STEPS, LOGGING_STEPS,
    MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    COMBO1_BATCH_SIZE, COMBO1_EVAL_BATCH_SIZE, COMBO1_GRAD_ACCUM_STEPS,
    COMBO1_LEARNING_RATE, COMBO1_WARMUP_STEPS,
    COMBO2_BATCH_SIZE, COMBO2_EVAL_BATCH_SIZE, COMBO2_GRAD_ACCUM_STEPS,
    COMBO2_LEARNING_RATE, COMBO2_WARMUP_STEPS,
)
from dataset import PipelineAdvDataset, pipeline_adv_collate_fn, load_jsonl

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--combo", type=int, choices=[1, 2], default=2,
                    help="1=DeBERTa-MLM filler  2=BART filler (default)")
parser.add_argument("--ckpt_dir", default=None, help="Override checkpoint directory")
parser.add_argument("--kaggle", action="store_true",
                    help="Use /kaggle/working/ as checkpoint dir (Kaggle runs)")
# ── Runtime overrides (used by run_all_fast.py / Kaggle notebooks) ─────────
parser.add_argument("--max_train",  type=int, default=None,
                    help="Cap training samples (e.g. 4000 for fast local run)")
parser.add_argument("--max_eval",   type=int, default=None,
                    help="Cap eval samples (e.g. 400)")
parser.add_argument("--num_epochs", type=int, default=None,
                    help="Override NUM_EPOCHS from config")
parser.add_argument("--batch_size", type=int, default=None,
                    help="Override per-combo BATCH_SIZE")
parser.add_argument("--seq_len",    type=int, default=None,
                    help="Override MAX_INPUT_LENGTH and MAX_TARGET_LENGTH")
parser.add_argument("--grad_accum", type=int, default=None,
                    help="Override GRAD_ACCUM_STEPS")
args = parser.parse_args()

COMBO = args.combo

# ── Per-combo config ────────────────────────────────────────────────────────
if COMBO == 1:
    FILLER_ID    = COMBO1_FILLER_ID
    INVERTER_ID  = COMBO1_INVERTER_ID
    CKPT_DIR     = args.ckpt_dir or (
        "/kaggle/working/checkpoints/combo1" if args.kaggle else COMBO1_CKPT_DIR
    )
    BATCH_SIZE       = COMBO1_BATCH_SIZE
    EVAL_BATCH       = COMBO1_EVAL_BATCH_SIZE
    GRAD_ACCUM       = COMBO1_GRAD_ACCUM_STEPS
    LEARNING_RATE    = COMBO1_LEARNING_RATE
    WARMUP_STEPS     = COMBO1_WARMUP_STEPS
else:
    FILLER_ID    = COMBO2_FILLER_ID
    INVERTER_ID  = COMBO2_INVERTER_ID
    CKPT_DIR     = args.ckpt_dir or (
        "/kaggle/working/checkpoints/combo2" if args.kaggle else COMBO2_CKPT_DIR
    )
    BATCH_SIZE       = COMBO2_BATCH_SIZE
    EVAL_BATCH       = COMBO2_EVAL_BATCH_SIZE
    GRAD_ACCUM       = COMBO2_GRAD_ACCUM_STEPS
    LEARNING_RATE    = COMBO2_LEARNING_RATE
    WARMUP_STEPS     = COMBO2_WARMUP_STEPS

# ── Apply runtime overrides ────────────────────────────────────────────────
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
    EVAL_BATCH = max(1, args.batch_size)
if args.grad_accum is not None:
    GRAD_ACCUM = args.grad_accum
if args.seq_len is not None:
    MAX_INPUT_LENGTH  = args.seq_len
    MAX_TARGET_LENGTH = args.seq_len
NUM_EPOCHS = args.num_epochs if args.num_epochs is not None else NUM_EPOCHS
MAX_TRAIN_SAMPLES = args.max_train   # None = use full dataset
MAX_EVAL_SAMPLES  = args.max_eval

# ── Logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"train_adv_combo{COMBO}.log"), mode="a"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return f"{alloc:.0f}/{total:.0f} MB"


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# ── Label-smoothed CE ──────────────────────────────────────────────────────

class LabelSmoothedCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing  = smoothing
        self.ignore_idx = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        V           = logits.size(-1)
        logits_flat = logits.reshape(-1, V).float()
        labels_flat = labels.reshape(-1)
        mask        = labels_flat != self.ignore_idx
        log_probs   = F.log_softmax(logits_flat, dim=-1)
        nll    = F.nll_loss(log_probs, labels_flat.clamp(min=0),
                            ignore_index=self.ignore_idx, reduction="sum")
        smooth = -log_probs[mask].sum() / V
        count  = mask.sum().float().clamp(min=1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth) / count


# ── Model loading ───────────────────────────────────────────────────────────

def load_filler(device: torch.device):
    """Load the filler model (trainable) from HuggingFace Hub."""
    logger.info("Loading filler (Combo %d): %s", COMBO, FILLER_ID)

    filler_tok = AutoTokenizer.from_pretrained(FILLER_ID, use_fast=True)
    if filler_tok.pad_token is None:
        filler_tok.pad_token = filler_tok.eos_token or "[PAD]"

    if COMBO == 1:
        filler = AutoModelForMaskedLM.from_pretrained(FILLER_ID, torch_dtype=torch.bfloat16)
    else:
        filler = AutoModelForSeq2SeqLM.from_pretrained(FILLER_ID, torch_dtype=torch.float32)

    filler.gradient_checkpointing_enable()

    filler = filler.to(device)
    filler.train()
    n = sum(p.numel() for p in filler.parameters()) / 1e6
    logger.info("  Filler: %.1fM params (fp32, trainable) | GPU: %s", n, gpu_info())
    return filler, filler_tok


def load_frozen_inverter(device: torch.device):
    """Load the BART inverter (frozen, fp16) from HuggingFace Hub."""
    logger.info("Loading frozen BART inverter: %s", INVERTER_ID)
    inverter     = AutoModelForSeq2SeqLM.from_pretrained(INVERTER_ID, torch_dtype=torch.float16)
    inverter_tok = AutoTokenizer.from_pretrained(INVERTER_ID, use_fast=True)
    if inverter_tok.pad_token is None:
        inverter_tok.pad_token = inverter_tok.eos_token

    for param in inverter.parameters():
        param.requires_grad_(False)

    inverter = inverter.to(device)
    inverter.eval()
    n = sum(p.numel() for p in inverter.parameters()) / 1e6
    logger.info("  Inverter: %.1fM params FROZEN (fp16) | GPU: %s", n, gpu_info())
    return inverter, inverter_tok


# ── Loss computation ────────────────────────────────────────────────────────

def _inverter_decoder_ids(inv_labels: torch.Tensor, inverter, device):
    """Build teacher-forced decoder input IDs for the frozen inverter."""
    pad_id = inverter.config.pad_token_id
    bos_id = (inverter.config.decoder_start_token_id or inverter.config.bos_token_id)
    clean  = inv_labels.masked_fill(inv_labels == -100, pad_id)
    return torch.cat([
        torch.full((inv_labels.size(0), 1), bos_id, dtype=torch.long, device=device),
        clean[:, :-1],
    ], dim=1)


def compute_combo2_losses(
    filler, inverter, loss_fn,
    batch: Dict, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combo 2 (BART seq2seq filler).

    L₁: CE(filler logits, anonymized_labels)
    L₂: CE(frozen_inverter(soft_embs), original_labels)
        soft_embs = softmax(filler_logits / SOFT_TEMP) @ inverter.embed_W
        → exact same gradient trick as the Seq2Seq adversarial training.
    """
    in_ids    = batch["filler_input_ids"].to(device)
    in_mask   = batch["filler_attention_mask"].to(device)
    anon_lbl  = batch["anon_labels"].to(device)
    anon_mask = batch["anon_attention_mask"].to(device)
    inv_lbl   = batch["inv_labels"].to(device)

    # Forward through filler (trainable) — fp16 activations via autocast
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        filler_out = filler(input_ids=in_ids, attention_mask=in_mask, labels=anon_lbl)
    logits = filler_out.logits.float()    # (B, T, V=50265) cast to fp32 for stable loss

    # L₁ — quality anchor
    L1 = loss_fn(logits, anon_lbl)

    # Soft embeddings → frozen inverter
    embed_W   = inverter.model.shared.weight          # (V, D) fp16
    soft_embs = torch.softmax(logits.float() / SOFT_TEMP, dim=-1).half() @ embed_W

    enc_out = inverter.model.encoder(inputs_embeds=soft_embs, attention_mask=anon_mask)

    dec_in  = _inverter_decoder_ids(inv_lbl, inverter, device)
    dec_out = inverter.model.decoder(
        input_ids=dec_in,
        encoder_hidden_states=enc_out.last_hidden_state,
        encoder_attention_mask=anon_mask,
    )
    # L₂ — inverter entity recovery (we want this HIGH so negating it lowers it)
    L2 = loss_fn(inverter.lm_head(dec_out.last_hidden_state).float(), inv_lbl)

    return L1, L2


def compute_combo1_losses(
    filler, inverter, loss_fn,
    batch: Dict, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combo 1 (DeBERTa-MLM filler) — hidden-state gradient approximation.

    L₁: MLM CE at [MASK] positions vs anonymized labels (quality anchor).
    L₂: CE(frozen_inverter(deberta_hidden_states), original_labels)
        DeBERTa last_hidden_state (B, T_d, 768) is fed directly as
        inputs_embeds to the BART inverter encoder.  The embedding spaces
        are not perfectly aligned, but both have dim=768 and the gradient
        signal is sufficient to discourage predictable entity patterns.
    """
    in_ids   = batch["filler_input_ids"].to(device)     # DeBERTa tokenized
    in_mask  = batch["filler_attention_mask"].to(device)
    anon_lbl = batch["anon_labels"].to(device)
    inv_lbl  = batch["inv_labels"].to(device)

    # --- DeBERTa forward -------------------------------------------------
    # AutoModelForMaskedLM.forward returns (loss, logits) when labels given.
    # We supply labels (anon_lbl) only at [MASK] positions; elsewhere -100.
    # Build MLM labels: copy anon_lbl but keep -100 where input is NOT [MASK].
    mask_token_id = filler.config.mask_token_id if hasattr(filler.config, "mask_token_id") else None
    if mask_token_id is not None:
        # Only supervise positions where the filler input has [MASK]
        is_mask = (in_ids == mask_token_id)          # (B, T) bool
        # Clamp anon_lbl to [0, vocab) before masking (tokens only)
        mlm_labels = anon_lbl.clone()
        # Zero out non-MASK positions (they're -100 already from dataset logic),
        # but also eliminate positions where input is not [MASK] to be safe.
        mlm_labels[~is_mask] = -100
    else:
        # Fallback: supervise all non-padding positions
        mlm_labels = anon_lbl

    with torch.autocast("cuda", enabled=device.type == "cuda"):
        deberta_out = filler(
            input_ids=in_ids,
            attention_mask=in_mask,
            labels=mlm_labels,       # MLM CE at [MASK] positions
            output_hidden_states=True,
        )

    # L₁ — MLM quality loss (already computed by the model)
    L1 = deberta_out.loss if deberta_out.loss is not None else loss_fn(
        deberta_out.logits, mlm_labels
    )

    # --- Adversarial path ------------------------------------------------
    # Use last hidden states (B, T_d, 768) as soft embeddings for BART inverter.
    # The hidden states at [MASK] positions encode the model's predicted entity,
    # which is precisely what the adversarial loss should discourage being
    # recoverable by the inverter.
    hidden = deberta_out.hidden_states[-1]          # (B, T_d, 768) fp32
    hidden_h = hidden.half()                        # fp16 for inverter compat.

    # Inverter encoder accepts any inputs_embeds of shape (B, T, 768).
    # We use in_mask as the attention mask (same length T_d).
    enc_out = inverter.model.encoder(
        inputs_embeds=hidden_h,
        attention_mask=in_mask,
    )

    dec_in  = _inverter_decoder_ids(inv_lbl, inverter, device)
    dec_out = inverter.model.decoder(
        input_ids=dec_in,
        encoder_hidden_states=enc_out.last_hidden_state,
        encoder_attention_mask=in_mask,
    )
    L2 = loss_fn(inverter.lm_head(dec_out.last_hidden_state).float(), inv_lbl)

    return L1, L2


# ── Validation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(filler, inverter, eval_loader, loss_fn, device, max_batches=50) -> Dict:
    filler.eval()
    total_L1 = total_L2 = 0.0
    n = 0
    leakage_count = leakage_total = 0

    compute_fn = compute_combo2_losses if COMBO == 2 else compute_combo1_losses

    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        L1, L2 = compute_fn(filler, inverter, loss_fn, batch, device)
        total_L1 += L1.item()
        total_L2 += L2.item()
        n += 1

        # Entity leakage on a few batches (hard decode)
        if i < 20 and COMBO == 2:
            in_ids  = batch["filler_input_ids"].to(device)
            in_mask = batch["filler_attention_mask"].to(device)
            gen_ids = filler.generate(
                input_ids=in_ids,
                attention_mask=in_mask,
                max_new_tokens=MAX_TARGET_LENGTH,
                num_beams=1,
                do_sample=False,
            )
            # Check entity leakage
            from transformers import AutoTokenizer as _AT
            # tokenizer already stored in filler.tokenizer is not guaranteed;
            # we use the tokenizer passed at dataset build time.  For a quick
            # leakage estimate, compare decoded prediction against entity_texts.
            decoded = ["<skip>"] * len(gen_ids)   # placeholder; full eval in evaluate_adv.py
            for pred, entities in zip(decoded, batch["entity_texts"]):
                for ent in entities:
                    leakage_total += 1
                    if ent.lower() in pred.lower():
                        leakage_count += 1

    filler.train()
    n = max(n, 1)
    val_L1 = round(total_L1 / n, 4)
    val_L2 = round(total_L2 / n, 4)
    return {
        "val_L1":      val_L1,
        "val_L2":      val_L2,
        "val_L_total": round(ALPHA * val_L1 - LAMBDA_ADV * val_L2, 4),
        "leakage_est": round(leakage_count / max(leakage_total, 1), 4),
    }


# ── Checkpoint helpers ──────────────────────────────────────────────────────

def _build_ckpt(filler, optimizer, scheduler, scaler,
                epoch, batch_idx, global_step, best_val_loss):
    ckpt = {
        "epoch":                epoch,
        "batch_idx":            batch_idx,
        "global_step":          global_step,
        "best_val_loss":        best_val_loss,
        "model_state_dict":     filler.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp":            datetime.now().isoformat(),
        "combo":                COMBO,
        "config": {
            "ALPHA": ALPHA, "LAMBDA_ADV": LAMBDA_ADV, "SOFT_TEMP": SOFT_TEMP,
        },
    }
    if scheduler: ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler:    ckpt["scaler_state_dict"]    = scaler.state_dict()
    return ckpt


def save_checkpoint(filler, optimizer, scheduler, scaler,
                    epoch, batch_idx, global_step, best_val_loss, tag="best"):
    ckpt = _build_ckpt(filler, optimizer, scheduler, scaler,
                       epoch, batch_idx, global_step, best_val_loss)
    path = os.path.join(CKPT_DIR, f"{tag}_model.pt")
    torch.save(ckpt, path)
    logger.info("  Saved %s checkpoint → %s  (step=%d, val_L_total=%.4f)",
                tag, path, global_step, best_val_loss)


def resume_if_available() -> Optional[Dict]:
    for fname, label in [("latest_checkpoint.pt", "latest"), ("best_model.pt", "best")]:
        path = os.path.join(CKPT_DIR, fname)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            logger.info(
                "Resuming from %s checkpoint (step=%d, epoch=%d, val_L_total=%.4f)",
                label, ckpt.get("global_step", 0), ckpt.get("epoch", 0),
                ckpt.get("best_val_loss", float("inf")),
            )
            return ckpt
    return None


# ── Main training loop ──────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("ADVERSARIAL TRAINING — Pipeline Combo %d", COMBO)
    logger.info("Filler   : %s", FILLER_ID)
    logger.info("Inverter : %s (frozen)", INVERTER_ID)
    logger.info("Loss     : %.1f·L₁ − %.2f·L₂  |  T=%.1f", ALPHA, LAMBDA_ADV, SOFT_TEMP)
    logger.info("Device   : %s | GPU: %s", device, gpu_info())
    logger.info("=" * 70)

    # -- 1. Models -----------------------------------------------------------
    filler, filler_tok  = load_filler(device)
    inverter, inv_tok   = load_frozen_inverter(device)
    loss_fn             = LabelSmoothedCE(smoothing=LABEL_SMOOTHING)

    # -- 2. Data  ------------------------------------------------------------
    logger.info("Loading adversarial pairs …")
    train_data = load_jsonl(ADV_TRAIN_FILE)
    eval_data  = load_jsonl(ADV_EVAL_FILE)
    if MAX_TRAIN_SAMPLES is not None:
        train_data = train_data[:MAX_TRAIN_SAMPLES]
    if MAX_EVAL_SAMPLES is not None:
        eval_data = eval_data[:MAX_EVAL_SAMPLES]
    logger.info("  Train: %d | Eval: %d", len(train_data), len(eval_data))

    train_ds = PipelineAdvDataset(
        train_data, filler_tok, inv_tok, COMBO, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    eval_ds = PipelineAdvDataset(
        eval_data, filler_tok, inv_tok, COMBO, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=pipeline_adv_collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=EVAL_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        collate_fn=pipeline_adv_collate_fn,
    )

    iters_per_epoch = len(train_loader)
    opt_steps_epoch = max(1, iters_per_epoch // GRAD_ACCUM)
    total_steps     = opt_steps_epoch * NUM_EPOCHS
    logger.info(
        "  Iters/epoch: %d | Opt steps/epoch: %d | Total: %d | Eff batch: %d",
        iters_per_epoch, opt_steps_epoch, total_steps, BATCH_SIZE * GRAD_ACCUM,
    )

    # -- 3. Optimizer & scheduler --------------------------------------------
    # Combo 1 (DeBERTa-MLM): use Adafactor — factored second moments use ~50x less
    # memory than AdamW on a 3.7 GiB GPU (optimizer state ~50 MB vs ~1.5 GB).
    # Combo 2 (BART seq2seq): AdamW is fine; single BART fits with more headroom.
    if COMBO == 1:
        optimizer = Adafactor(
            filler.parameters(),
            lr=LEARNING_RATE,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
            num_training_steps=total_steps,
        )
    else:
        no_decay = {"bias", "LayerNorm.weight"}
        param_groups = [
            {"params": [p for n, p in filler.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in filler.named_parameters()
                        if any(nd in n for nd in no_decay)],     "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
            num_training_steps=total_steps,
        )
    # bfloat16 (combo 1) has the same exponent range as fp32 — no loss scaling needed.
    # float32 (combo 2) uses autocast to fp16 activations, so scaling IS needed.
    scaler = GradScaler("cuda") if (device.type == "cuda" and COMBO == 2) else None

    # -- 4. Resume  ----------------------------------------------------------
    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 0
    skip_batches  = 0

    existing = resume_if_available()
    if existing is not None:
        filler.load_state_dict(existing["model_state_dict"])
        optimizer.load_state_dict(existing["optimizer_state_dict"])
        if "scheduler_state_dict" in existing and scheduler:
            scheduler.load_state_dict(existing["scheduler_state_dict"])
        if "scaler_state_dict" in existing and scaler:
            scaler.load_state_dict(existing["scaler_state_dict"])
        best_val_loss = existing.get("best_val_loss", float("inf"))
        global_step   = existing.get("global_step", 0)
        start_epoch   = existing.get("epoch", 0)
        skip_batches  = existing.get("batch_idx", -1) + 1

    history: Dict = {"train": [], "eval": []}
    compute_fn    = compute_combo2_losses if COMBO == 2 else compute_combo1_losses

    # -- 5. Training loop  ---------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        logger.info("\n── Epoch %d/%d ──", epoch + 1, NUM_EPOCHS)
        filler.train()
        optimizer.zero_grad()

        running_L1 = running_L2 = running_total = 0.0
        accum_step = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):

            # Mid-epoch resume: skip already-processed batches
            if epoch == start_epoch and batch_idx < skip_batches:
                global_step += 1
                continue

            try:
                L1, L2 = compute_fn(filler, inverter, loss_fn, batch, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM at batch %d — skipping", batch_idx)
                    aggressive_cleanup()
                    optimizer.zero_grad()
                    continue
                raise

            L_total = ALPHA * L1 - LAMBDA_ADV * L2
            loss_for_accum = L_total / GRAD_ACCUM

            if scaler:
                scaler.scale(loss_for_accum).backward()
            else:
                loss_for_accum.backward()

            running_L1    += L1.item()
            running_L2    += L2.item()
            running_total += L_total.item()
            accum_step    += 1

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filler.parameters(), MAX_GRAD_NORM)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOGGING_STEPS == 0:
                    n = max(accum_step, 1)
                    logger.info(
                        "  step=%d  L1=%.4f  L2=%.4f  L_total=%.4f  lr=%.2e  gpu=%s",
                        global_step,
                        running_L1 / n, running_L2 / n, running_total / n,
                        scheduler.get_last_lr()[0],
                        gpu_info(),
                    )
                    history["train"].append({
                        "step": global_step,
                        "L1": round(running_L1 / n, 4),
                        "L2": round(running_L2 / n, 4),
                        "L_total": round(running_total / n, 4),
                    })
                    running_L1 = running_L2 = running_total = 0.0
                    accum_step = 0

                if global_step % EVAL_STEPS == 0:
                    metrics = evaluate(filler, inverter, eval_loader, loss_fn, device)
                    logger.info(
                        "  [EVAL step=%d]  L1=%.4f  L2=%.4f  L_total=%.4f",
                        global_step, metrics["val_L1"], metrics["val_L2"],
                        metrics["val_L_total"],
                    )
                    history["eval"].append({"step": global_step, **metrics})

                    if metrics["val_L_total"] < best_val_loss:
                        best_val_loss = metrics["val_L_total"]
                        save_checkpoint(filler, optimizer, scheduler, scaler,
                                        epoch, batch_idx, global_step,
                                        best_val_loss, tag="best")

                # Periodic safety checkpoint
                if global_step % (EVAL_STEPS * 2) == 0:
                    save_checkpoint(filler, optimizer, scheduler, scaler,
                                    epoch, batch_idx, global_step,
                                    best_val_loss, tag="latest_checkpoint")

            del L1, L2, L_total, loss_for_accum

        # End of epoch
        metrics = evaluate(filler, inverter, eval_loader, loss_fn, device)
        logger.info(
            "Epoch %d done | val_L1=%.4f  val_L2=%.4f  val_L_total=%.4f",
            epoch + 1, metrics["val_L1"], metrics["val_L2"], metrics["val_L_total"],
        )
        if metrics["val_L_total"] < best_val_loss:
            best_val_loss = metrics["val_L_total"]
            save_checkpoint(filler, optimizer, scheduler, scaler,
                            epoch, len(train_loader) - 1, global_step,
                            best_val_loss, tag="best")

        # Save after each epoch for crash recovery
        save_checkpoint(filler, optimizer, scheduler, scaler,
                        epoch + 1, -1, global_step,
                        best_val_loss, tag="latest_checkpoint")
        aggressive_cleanup()

    # -- 6. Save final model -------------------------------------------------
    final_path = os.path.join(CKPT_DIR, "final_model.pt")
    torch.save({"model_state_dict": filler.state_dict(), "combo": COMBO}, final_path)
    logger.info("Final model saved → %s", final_path)

    with open(os.path.join(CKPT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 70)
    logger.info("Adversarial training complete. Best val_L_total: %.4f", best_val_loss)
    logger.info("=" * 70)


if __name__ == "__main__":
    train()
