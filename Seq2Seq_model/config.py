"""
Configuration for all 6 Seq2Seq models.
Each model has carefully tuned batch sizes and memory settings for RTX 3050 4GB VRAM.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "data_creation", "output", "anonymized_dataset_final.jsonl")
DATA_SPLITS_DIR = os.path.join(BASE_DIR, "data_splits")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints2")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ============================================================
# DATA SPLIT RATIOS
# ============================================================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# ============================================================
# COMMON TRAINING SETTINGS
# ============================================================
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
NUM_EPOCHS = 3
WARMUP_STEPS = 500
LOGGING_STEPS = 100
EVAL_STEPS = 500          # evaluate every N steps
SAVE_STEPS = 500          # checkpoint every N steps
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1     # Label smoothing factor (0.0 = no smoothing, 0.1 = recommended)
NUM_WORKERS = 2           # dataloader workers

# ============================================================
# DATA AUGMENTATION SETTINGS
# ============================================================
# On-the-fly augmentations applied ONLY to training data.
# Makes the model robust to case variations, typos, whitespace noise, etc.
AUGMENTATION_PROB = 0.35          # probability of augmenting any given sample (0.0 = off)
ENABLED_AUGMENTATIONS = [         # which augmentations to use
    "lowercase",                  # "my name is john smith"
    "uppercase",                  # "MY NAME IS JOHN SMITH"
    "title_case",                 # "My Name Is John Smith"
    "random_case",                # "mY nAmE iS jOhN sMiTh"
    "swap_case",                  # "mY NAME IS jOHN sMITH"
    "remove_punctuation",         # strips .,;:!? etc.
    "typo",                       # keyboard-adjacent char replacements
    "whitespace_noise",           # extra/missing spaces
]
AUGMENTATION_WEIGHTS = {          # relative sampling weights (higher = more likely)
    "lowercase": 3.0,             # most important — user input is often all lowercase
    "uppercase": 1.5,             # ALL CAPS input (forms, SMS)
    "title_case": 1.5,            # Title Case (headers, forms)
    "random_case": 1.0,           # edge case robustness
    "swap_case": 0.5,             # rare but adds diversity
    "remove_punctuation": 1.0,    # messy user input
    "typo": 1.5,                  # very common in real user input
    "whitespace_noise": 1.0,      # sloppy spacing
}

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================
# Each config has memory-safe settings for RTX 3050 4GB VRAM.
# gradient_checkpointing = True saves ~30-40% VRAM at cost of speed.
# fp16 = True halves memory usage.
# accumulation_steps simulates larger effective batch size.

MODEL_CONFIGS = {
    "t5-efficient-tiny": {
        "model_name": "google/t5-efficient-tiny",
        "model_type": "t5",            # t5 family
        "batch_size": 8,
        "eval_batch_size": 16,
        "learning_rate": 3e-4,
        "fp16": False,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 2,  # effective batch = 16
        "prefix": "anonymize: ",
        "use_qlora": False,
    },

    "t5-small": {
        "model_name": "google-t5/t5-small",
        "model_type": "t5",
        "batch_size": 4,
        "eval_batch_size": 8,
        "learning_rate": 3e-4,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,  # effective batch = 16
        "prefix": "anonymize: ",
        "use_qlora": False,
    },

    "flan-t5-small": {
        "model_name": "google/flan-t5-small",
        "model_type": "t5",
        "batch_size": 4,
        "eval_batch_size": 8,
        "learning_rate": 3e-4,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,  # effective batch = 16
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
        "use_qlora": False,
    },

    "bart-base": {
        "model_name": "facebook/bart-base",
        "model_type": "bart",
        "batch_size": 2,
        "eval_batch_size": 4,
        "learning_rate": 2e-5,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 8,  # effective batch = 16
        "prefix": "",                      # BART doesn't use prefix
        "use_qlora": False,
    },

    "distilbart": {
        "model_name": "sshleifer/distilbart-cnn-6-6",
        "model_type": "bart",
        "batch_size": 4,
        "eval_batch_size": 8,
        "learning_rate": 2e-5,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,  # effective batch = 16
        "prefix": "",
        "use_qlora": False,
    },

    "flan-t5-base-qlora": {
        "model_name": "google/flan-t5-base",
        "model_type": "t5",
        "batch_size": 4,
        "eval_batch_size": 8,
        "learning_rate": 2e-4,
        "fp16": False,                      # fp16 on top of 4-bit base
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,  # effective batch = 16
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
        "use_qlora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q", "v"],
    },
}

# Order of training (smallest to largest to catch OOM early on smaller models)
TRAINING_ORDER = [
    "t5-efficient-tiny",
    "t5-small",
    "flan-t5-small",
    "distilbart",
    "bart-base",
    "flan-t5-base-qlora",
]
