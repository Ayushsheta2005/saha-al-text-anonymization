"""
Adversarial Training — Central Configuration
=============================================
Trains a hardened BART-base PII anonymizer using ONLY the adversarial pairs
from the model-inversion query step (inverter_train.jsonl).

The model is already well-trained on 93k normal pairs.  We do NOT re-iterate
over that data.  Instead, each adversarial pair (original, anonymized) drives
BOTH losses in a single forward pass:

    L₁ = CE(Victim(original), anonymized_labels)
         → keeps PII replacement correct AND enforces output fluency
           (CE against a fluent reference = conditional PPL;
            if the model outputs gibberish to fool the inverter, L₁ rises)

    L₂ = CE(FrozenInverter(soft_victim_output), original_labels)
         → inverter's entity-recovery loss; NEGATED in the total

    L_total = ALPHA · L₁  −  LAMBDA_ADV · L₂

No separate normal-data loop.  No cycling two loaders.
38k adv pairs, batch=4 → ~9,500 iterations/epoch (vs 46k before).

The frozen inverter is the one trained in model_inversion/.
Original Seq2Seq checkpoints are NEVER touched.
"""

import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ── Input data ──────────────────────────────────────────────────────────────
# Adversarial pairs — (original PII text, BART-anonymized text)
# produced by model_inversion/query_bart.py
ADV_TRAIN_FILE = os.path.join(PROJECT_DIR, "model_inversion", "output", "inverter_train.jsonl")
ADV_EVAL_FILE  = os.path.join(PROJECT_DIR, "model_inversion", "output", "inverter_eval.jsonl")

# Normal val — used ONLY in evaluate_adv.py Part A to check for forgetting
NORMAL_VAL = os.path.join(PROJECT_DIR, "Seq2Seq_model", "data_splits", "val.jsonl")

# Gold references (benchmark — LLM-synthesised + reviewed, independent of BART)
# Used for L1 loss during adversarial training (avoids circularity).
GOLD_TRAIN_FILE   = os.path.join(PROJECT_DIR, "benchmark", "data", "train.jsonl")
GOLD_VAL_FILE     = os.path.join(PROJECT_DIR, "benchmark", "data", "validation.jsonl")
GOLD_MAX_SAMPLES  = 10_000   # subset of 113k; cycles ~3.8× per adv epoch

# ── Victim model (BART-base anonymizer — to be adversarially hardened) ──────
VICTIM_MODEL_NAME = "facebook/bart-base"
# Starting weights: best checkpoint from original Seq2Seq training (read-only)
VICTIM_CHECKPOINT = os.path.join(
    PROJECT_DIR, "Seq2Seq_model", "checkpoints", "bart-base", "best_model.pt"
)

# ── Frozen adversary (inverter — never updated) ──────────────────────────────
# Loaded in fp16 to save ~260MB VRAM (frozen → precision not critical for fwd pass;
# CE loss casts back to fp32 internally before log_softmax over vocab).
INVERTER_MODEL_NAME = "facebook/bart-base"   # same arch → same vocab & embed dim
INVERTER_CHECKPOINT = os.path.join(
    PROJECT_DIR, "model_inversion", "inverter_checkpoint", "best_model.pt"
)

# ── Outputs (original checkpoints are untouched) ─────────────────────────────
ADV_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "bart-base-adv")
LOGS_DIR           = os.path.join(BASE_DIR, "logs")
RESULTS_DIR        = os.path.join(BASE_DIR, "results")

# ── Combined loss weights ────────────────────────────────────────────────────
# L_total = ALPHA · L₁  −  LAMBDA_ADV · L₂
ALPHA      = 1.0   # L₁ weight — never drop below 1.0 (preserves anonymization quality)
LAMBDA_ADV = 0.5   # L₂ weight — start conservative; raise if quality stays high

# Softmax temperature on victim logits before the soft-embedding matmul.
# 1.0 = standard; >1 gives smoother gradients through the inverter.
SOFT_TEMP = 1.0

# ── Training hyperparameters ─────────────────────────────────────────────────
NUM_EPOCHS       = 4
BATCH_SIZE       = 2     # adv pairs per step; halved from 4 to avoid VRAM fragmentation
                         # RTX 3050 4GB: victim fp32 (0.52GB) + inverter fp16 (0.26GB)
                         #   + AdamW on victim (1.04GB) + activations batch=2 (~0.3GB) ≈ 2.1GB
                         # The key tensor: softmax(logits/T) is (B,T,V) fp32;
                         # at B=2,T=128,V=50265 that's ~51MB vs ~102MB at B=4.
EVAL_BATCH_SIZE  = 4
GRAD_ACCUM_STEPS = 8     # effective batch = 2×8 = 16  (unchanged)
                         # iterations: 38,032 / 2 = ~19,016 per epoch
                         # optimiser steps/epoch: ~2,377  (unchanged)
LEARNING_RATE    = 1e-5  # fine-tuning LR (model already trained — small updates only)
WARMUP_STEPS     = 100
MAX_GRAD_NORM    = 1.0
WEIGHT_DECAY     = 0.01
LABEL_SMOOTHING  = 0.1   # same as original training

MAX_INPUT_LENGTH  = 128
MAX_TARGET_LENGTH = 128
NUM_WORKERS       = 2

# ── Eval / checkpoint schedule ────────────────────────────────────────────────
EVAL_STEPS    = 500   # run val every N optimiser steps
LOGGING_STEPS = 50    # log training losses every N optimiser steps
SAVE_STEPS    = 500   # save checkpoint every N steps (if val improved)

# ── Checkpointing / logging ───────────────────────────────────────────────────
EVAL_STEPS    = 500   # run validation every N optimiser steps
LOGGING_STEPS = 100   # log training loss every N optimiser steps
SAVE_STEPS    = 500   # save checkpoint every N optimiser steps (if improved)
