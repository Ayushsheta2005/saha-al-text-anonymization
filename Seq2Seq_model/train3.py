"""
Kaggle Training Script with PII-Aware Loss
============================================
Same as train2.py but uses a novel PII-Aware Loss function:

Standard CE Problem:
    Target says "John" → "James", model outputs "David".
    Standard CE penalizes this — even though "David" is a valid anonymization.

PII-Aware Loss Solution:
    For NON-PII tokens:  standard label-smoothed CE (must match target exactly)
    For PII tokens:      α × label-smoothed CE   (weak push toward target — for structure)
                       + β × anti-leakage penalty (strong push AWAY from input token)

    This means the model is:
      - Strongly penalized for copying original PII (leakage)
      - Weakly guided toward the target replacement (for type consistency)
      - NOT harshly punished for choosing a different valid replacement

Saves to checkpoints2/ (separate from train2.py's checkpoints/).

Usage:
    python train3.py                    # interactive model selection
    python train3.py t5-small bart-base # command-line selection
    python train3.py all                # train every model
"""

import os
import sys
import gc
import time
import json
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)

from config import (
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
from dataset import load_split_data
from augmentations import TextAugmentor
from utils import (
    setup_logger,
    save_training_history,
    format_time,
    count_parameters,
    compute_token_accuracy,
    compute_word_level_accuracy,
    compute_all_metrics,
)


# ============================================================
# PATHS — separate from train2.py
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS2_DIR = os.path.join(BASE_DIR, "checkpoints2")
LOGS2_DIR = os.path.join(BASE_DIR, "logs2")


# ============================================================
# KAGGLE MODEL CONFIGS (same as train2.py)
# ============================================================

NUM_WORKERS = 4

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
    "flan-t5-base-qlora": {
        "model_name": "google/flan-t5-base",
        "model_type": "t5",
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 2,
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
        "use_qlora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q", "v"],
    },
}


# ============================================================
# PII-AWARE DATASET
# ============================================================
# Returns a pii_mask alongside input_ids/labels so the loss knows
# which target positions correspond to PII replacements.

class PIIAwareDataset(Dataset):
    """
    Dataset that also produces:
      - original_token_ids: tokenized original text (no prefix), padded
      - pii_mask:           1 at target positions that correspond to PII, 0 otherwise

    The pii_mask is built using CHARACTER OFFSETS:
      1. Find each entity's character span in the ORIGINAL text via str.find()
      2. Tokenize the ANONYMIZED text with return_offsets_mapping to get
         each token's (start_char, end_char) in the anonymized string
      3. Use a word-level alignment between original and anonymized to map
         entity character positions → anonymized character positions
      4. Mark target tokens whose character spans overlap with PII regions

    This avoids the token-shift problem when PII replacements have different
    numbers of subword tokens (e.g., "John" → "Elara Vance").
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
        prefix: str = "",
        augmentor=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data)

    def _find_entity_char_spans(self, text: str, entity_texts: list) -> list:
        """
        Find all (start_char, end_char) spans of entities in the text.
        Handles multiple occurrences and overlapping entities.
        Returns list of (start, end) tuples in the text.
        """
        spans = []
        for entity in entity_texts:
            if not entity or len(entity) < 1:
                continue
            # Find all occurrences of this entity in the text
            search_start = 0
            while True:
                idx = text.find(entity, search_start)
                if idx == -1:
                    # Try case-insensitive as fallback
                    idx = text.lower().find(entity.lower(), search_start)
                    if idx == -1:
                        break
                spans.append((idx, idx + len(entity)))
                search_start = idx + 1  # allow overlapping matches
        return spans

    def _build_pii_mask_for_target(self, anonymized_text: str, original_text: str,
                                    entity_texts: list, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a binary mask over TARGET (label) tokens using character offsets.

        Approach:
          1. Find entity character spans in ORIGINAL text
          2. Build word-level alignment: split both texts into words,
             identify which word indices in the original contain PII
          3. Map those word indices to the ANONYMIZED text's word positions
          4. Tokenize anonymized text with offset_mapping to get token→char mapping
          5. Mark target tokens that fall within PII word boundaries

        This is robust to different-length PII replacements because we align
        at the word level (which preserves non-PII word order) rather than
        at the token level (which shifts when token counts differ).
        """
        seq_len = label_ids.size(0)
        pii_mask = torch.zeros(seq_len, dtype=torch.float32)

        if not entity_texts:
            return pii_mask

        # ── Step 1: Find entity character spans in the ORIGINAL text ──
        orig_entity_spans = self._find_entity_char_spans(original_text, entity_texts)
        if not orig_entity_spans:
            return pii_mask

        # ── Step 2: Identify which WORD indices in original contain PII ──
        # Split original into words with their character positions
        orig_words = original_text.split()
        orig_word_char_starts = []
        pos = 0
        for w in orig_words:
            idx = original_text.find(w, pos)
            orig_word_char_starts.append(idx)
            pos = idx + len(w)

        # Mark which word indices overlap with entity spans
        pii_word_indices = set()
        for word_idx, word in enumerate(orig_words):
            word_start = orig_word_char_starts[word_idx]
            word_end = word_start + len(word)
            for ent_start, ent_end in orig_entity_spans:
                # Check overlap
                if word_start < ent_end and word_end > ent_start:
                    pii_word_indices.add(word_idx)
                    break

        if not pii_word_indices:
            return pii_mask

        # ── Step 3: Map PII word indices to ANONYMIZED text character spans ──
        # The key insight: non-PII words appear in the same order in both texts.
        # We align by walking through both word lists simultaneously:
        #   - non-PII words in original correspond 1:1 to words in anonymized
        #   - PII words in original correspond to "replacement spans" in anonymized
        #
        # We identify the character regions in anonymized_text that correspond
        # to PII replacements by finding the boundaries between non-PII words.

        anon_words = anonymized_text.split()
        anon_word_char_starts = []
        pos = 0
        for w in anon_words:
            idx = anonymized_text.find(w, pos)
            anon_word_char_starts.append(idx)
            pos = idx + len(w)

        # Build list of non-PII word texts and their positions in original
        non_pii_anchors = []  # (word_text, orig_word_idx)
        for i, w in enumerate(orig_words):
            if i not in pii_word_indices:
                non_pii_anchors.append((w, i))

        # Find PII character spans in anonymized text by identifying gaps
        # between matched non-PII anchors
        anon_pii_char_spans = []

        # Match non-PII anchors greedily in the anonymized word list
        anon_idx = 0
        prev_anon_end_char = 0  # character position after last matched non-PII word
        anchor_idx = 0
        first_pii_word_idx = min(pii_word_indices) if pii_word_indices else len(orig_words)

        # If PII starts before first non-PII anchor, mark from beginning
        # Walk through anchors and find their positions in anonymized text
        matched_anon_positions = []  # (anchor_idx_in_list, anon_word_idx, orig_word_idx)

        for a_idx, (anchor_word, orig_w_idx) in enumerate(non_pii_anchors):
            # Find this anchor word in anonymized words starting from anon_idx
            found = False
            for j in range(anon_idx, len(anon_words)):
                if anon_words[j] == anchor_word:
                    matched_anon_positions.append((a_idx, j, orig_w_idx))
                    anon_idx = j + 1
                    found = True
                    break
            if not found:
                # Anchor word not found (maybe augmentation changed it) — skip
                continue

        # Now identify PII spans in anonymized text as the gaps between anchors
        # Gap before first anchor
        if matched_anon_positions:
            first_match_anon_idx = matched_anon_positions[0][1]
            first_match_orig_idx = matched_anon_positions[0][2]
            # If there are PII words before the first anchor in original
            if any(i < first_match_orig_idx for i in pii_word_indices):
                if first_match_anon_idx > 0:
                    # PII region: from start to first anchor
                    span_end = anon_word_char_starts[first_match_anon_idx]
                    anon_pii_char_spans.append((0, span_end))

            # Gaps between consecutive anchors
            for k in range(len(matched_anon_positions) - 1):
                _, curr_anon_idx, curr_orig_idx = matched_anon_positions[k]
                _, next_anon_idx, next_orig_idx = matched_anon_positions[k + 1]

                # Check if there are PII words between these two anchors in original
                has_pii_between = any(
                    curr_orig_idx < i < next_orig_idx for i in pii_word_indices
                )
                if has_pii_between and next_anon_idx > curr_anon_idx + 1:
                    # PII region: from end of current anchor to start of next anchor
                    span_start = anon_word_char_starts[curr_anon_idx] + len(anon_words[curr_anon_idx])
                    span_end = anon_word_char_starts[next_anon_idx]
                    anon_pii_char_spans.append((span_start, span_end))

            # Gap after last anchor
            last_match_anon_idx = matched_anon_positions[-1][1]
            last_match_orig_idx = matched_anon_positions[-1][2]
            if any(i > last_match_orig_idx for i in pii_word_indices):
                if last_match_anon_idx < len(anon_words) - 1:
                    span_start = anon_word_char_starts[last_match_anon_idx] + len(anon_words[last_match_anon_idx])
                    anon_pii_char_spans.append((span_start, len(anonymized_text)))
        else:
            # No anchors matched — entire text might be PII (rare edge case)
            # Fall back: mark everything as PII
            anon_pii_char_spans.append((0, len(anonymized_text)))

        if not anon_pii_char_spans:
            return pii_mask

        # ── Step 4: Tokenize anonymized text with offset_mapping ──
        try:
            anon_enc = self.tokenizer(
                anonymized_text,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping = anon_enc["offset_mapping"].squeeze(0)  # (seq_len, 2)
        except Exception:
            # Fallback if offset_mapping not supported: use entity subsequence matching
            return self._build_pii_mask_fallback(anonymized_text, entity_texts, label_ids)

        # ── Step 5: Mark tokens whose char spans overlap with PII regions ──
        for tok_idx in range(min(seq_len, offset_mapping.size(0))):
            if label_ids[tok_idx] == -100:  # skip padding
                continue
            tok_start = offset_mapping[tok_idx, 0].item()
            tok_end = offset_mapping[tok_idx, 1].item()
            if tok_start == 0 and tok_end == 0:  # special token
                continue
            # Check if this token overlaps with any PII span
            for pii_start, pii_end in anon_pii_char_spans:
                if tok_start < pii_end and tok_end > pii_start:
                    pii_mask[tok_idx] = 1.0
                    break

        return pii_mask

    def _build_pii_mask_fallback(self, anonymized_text: str, entity_texts: list,
                                  label_ids: torch.Tensor) -> torch.Tensor:
        """
        Fallback PII mask when offset_mapping is not available.
        Uses entity subsequence matching in token space.
        """
        seq_len = label_ids.size(0)
        pii_mask = torch.zeros(seq_len, dtype=torch.float32)

        anon_tokens = self.tokenizer(
            anonymized_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze()

        for entity in entity_texts:
            if not entity or len(entity) < 2:
                continue
            ent_ids = self.tokenizer.encode(entity, add_special_tokens=False)
            if not ent_ids:
                continue
            ent_len = len(ent_ids)
            min_len = min(len(anon_tokens), seq_len)
            for start in range(min_len - ent_len + 1):
                if anon_tokens[start:start + ent_len].tolist() == ent_ids:
                    for j in range(start, min(start + ent_len, seq_len)):
                        if label_ids[j] != -100:
                            pii_mask[j] = 1.0

        return pii_mask

    def __getitem__(self, idx):
        entry = self.data[idx]

        original_text = entry["original_text"]
        anonymized_text = entry["anonymized_text"]
        entity_texts = entry.get("entity_texts", [])

        # Augmentation (training only)
        if self.augmentor is not None:
            original_text, anonymized_text = self.augmentor(original_text, anonymized_text)

        # Tokenize input (with prefix)
        input_text = self.prefix + original_text
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        labels = self.tokenizer(
            anonymized_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        # Tokenize original (without prefix) — needed for anti-leakage penalty
        orig_no_prefix = self.tokenizer(
            original_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Build PII mask using character-offset approach
        pii_mask = self._build_pii_mask_for_target(
            anonymized_text, original_text, entity_texts, label_ids
        )

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": label_ids,
            "original_token_ids": orig_no_prefix["input_ids"].squeeze(),
            "pii_mask": pii_mask,
        }


# ============================================================
# PII-AWARE LOSS
# ============================================================

class PIIAwareLoss(nn.Module):
    """
    PII-Aware Loss for Seq2Seq Anonymization.

    For NON-PII token positions (pii_mask == 0):
        Standard label-smoothed cross-entropy against the target.
        The model MUST reproduce these tokens exactly.

    For PII token positions (pii_mask == 1):
        α × label-smoothed CE      — weak structural guidance toward target
        + β × anti-leakage penalty  — strong penalty for outputting the original token

    The anti-leakage penalty is:  -log(1 - P(original_token))
        If the model assigns HIGH probability to the original PII token,
        this penalty is LARGE → pushes the model away from copying.
        If the model assigns LOW probability to the original token,
        this penalty is SMALL → no punishment.

    Args:
        smoothing:     label smoothing factor for CE (default 0.1)
        alpha:         CE weight for PII positions (default 0.3 — weak)
        beta:          anti-leakage penalty weight (default 2.0 — strong)
        ignore_index:  token ID to ignore (default -100)
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        alpha: float = 0.3,
        beta: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        original_token_ids: torch.Tensor,
        pii_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:             (B, T, V) — model output logits
            labels:             (B, T)    — target token IDs (-100 for padding)
            original_token_ids: (B, T)    — original input token IDs (for leakage check)
            pii_mask:           (B, T)    — 1.0 for PII positions, 0.0 for non-PII

        Returns:
            Scalar loss
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten
        logits_flat = logits.view(-1, vocab_size).float()  # (B*T, V)
        labels_flat = labels.view(-1)                       # (B*T,)
        orig_flat = original_token_ids.view(-1)             # (B*T,)
        pii_flat = pii_mask.view(-1)                        # (B*T,)

        # Non-padding mask
        non_pad = labels_flat != self.ignore_index
        if non_pad.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Log probabilities
        log_probs = F.log_softmax(logits_flat, dim=-1)  # (B*T, V)
        probs = torch.exp(log_probs)                     # (B*T, V)

        # ── Standard label-smoothed CE (for all positions) ──
        safe_labels = labels_flat.clamp(min=0)
        nll_loss = F.nll_loss(log_probs, safe_labels, reduction='none')  # (B*T,)
        smooth_loss = -log_probs.sum(dim=-1) / vocab_size                # (B*T,)
        ce_loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss  # (B*T,)

        # ── Anti-leakage penalty: -log(1 - P(original_token)) ──
        # Only applies at PII positions
        safe_orig = orig_flat.clamp(min=0)
        # P(original_token) at each position
        p_orig = probs.gather(1, safe_orig.unsqueeze(1)).squeeze(1)  # (B*T,)
        # Clamp to avoid log(0)
        anti_leak = -torch.log(1.0 - p_orig.clamp(max=0.999))  # (B*T,)

        # ── Combine ──
        # Non-PII positions: full CE, no leakage penalty
        # PII positions:     α × CE + β × anti-leakage
        is_pii = (pii_flat > 0.5) & non_pad
        is_non_pii = (pii_flat <= 0.5) & non_pad

        loss = torch.zeros_like(ce_loss)

        # Non-PII: standard CE (weight 1.0)
        loss[is_non_pii] = ce_loss[is_non_pii]

        # PII: reduced CE + anti-leakage
        loss[is_pii] = self.alpha * ce_loss[is_pii] + self.beta * anti_leak[is_pii]

        # Average over non-padding positions
        total_loss = loss[non_pad].mean()

        return total_loss


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


def get_checkpoint_dir(model_key: str) -> str:
    """Get checkpoint directory inside checkpoints2/."""
    path = os.path.join(CHECKPOINTS2_DIR, model_key)
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint_v2(
    model, optimizer, scheduler,
    epoch, global_step, best_val_loss,
    checkpoint_dir, model_config=None,
    use_qlora=False,
):
    """Save the best model checkpoint to checkpoints2/."""
    from datetime import datetime
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_config": model_config,
        "loss_type": "pii_aware",
        "timestamp": datetime.now().isoformat(),
    }

    if use_qlora:
        adapter_path = os.path.join(checkpoint_dir, "lora_adapter")
        model.save_pretrained(adapter_path)
        checkpoint["adapter_path"] = adapter_path
    else:
        checkpoint["model_state_dict"] = model.state_dict()

    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save(checkpoint, best_path)
    return best_path


def load_checkpoint_v2(checkpoint_dir: str):
    """Load checkpoint from checkpoints2/."""
    path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


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
# EVALUATION (uses standard CE loss for comparability)
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

        # Use model's built-in CE for val loss (comparable across runs)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        total_loss += loss.item()
        n_batches += 1

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


@torch.no_grad()
def full_evaluation(model, dataloader, tokenizer, device, eval_batch_size,
                    original_texts, entity_texts, split_name, logger):
    """Full evaluation with all metrics (BLEU, ROUGE, BERTScore, leakage)."""
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

    for key in ["exact_match", "word_accuracy", "bleu", "rouge1", "rougeL",
                 "bertscore_f1", "entity_leakage_rate"]:
        logger.info(f"  {key}: {metrics.get(key, 0):.2f}")

    tqdm.write(f"\n  ── {split_name.upper()} RESULTS ──")
    tqdm.write(f"  Loss: {avg_loss:.4f}  |  Exact: {metrics['exact_match']:.2f}%  |  "
               f"Word Acc: {metrics['word_accuracy']:.2f}%")
    tqdm.write(f"  BLEU: {metrics['bleu']:.2f}  |  ROUGE-L: {metrics['rougeL']:.2f}  |  "
               f"BERTScore F1: {metrics.get('bertscore_f1', 0):.2f}")
    tqdm.write(f"  Entity Leakage: {metrics.get('entity_leakage_rate', 0):.2f}% "
               f"({metrics.get('total_entities_leaked', 0)}/{metrics.get('total_entities_checked', 0)})")

    for i in range(min(3, len(all_preds))):
        logger.info(f"  Sample {i+1}:  PRED={all_preds[i][:120]}  |  TRUE={all_targets[i][:120]}")

    return avg_loss, metrics, all_preds, all_targets, all_originals


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_single_model(model_key, config, device, num_gpus):
    """Train a single model with PII-Aware Loss. Returns True on success."""
    os.makedirs(LOGS2_DIR, exist_ok=True)
    logger = setup_logger(f"{model_key}_v3", LOGS2_DIR)
    logger.info("=" * 70)
    logger.info(f"TRAINING (PII-Aware Loss): {model_key}  ({config['model_name']})")
    logger.info("=" * 70)

    checkpoint_dir = get_checkpoint_dir(model_key)

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

        # Training uses PII-aware dataset (returns pii_mask + original_token_ids)
        train_dataset = PIIAwareDataset(train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix, augmentor=augmentor)
        # Val/test use standard dataset (evaluation uses model's built-in loss)
        from dataset import AnonymizationDataset
        val_dataset   = AnonymizationDataset(val_data,   tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix, augmentor=None)
        test_dataset  = AnonymizationDataset(test_data,  tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix, augmentor=None)

        bs      = config["batch_size"]
        eval_bs = config["eval_batch_size"]

        train_loader = DataLoader(train_dataset, batch_size=bs,      shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=eval_bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_dataset,  batch_size=eval_bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # ── 4. Optimizer, Scheduler, Loss ────────────────────────
        accum_steps = config.get("gradient_accumulation_steps", 1)
        total_steps = (len(train_loader) // accum_steps) * NUM_EPOCHS

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
                                                    num_training_steps=total_steps)

        # ★ THE KEY DIFFERENCE: PII-Aware Loss instead of standard CE
        loss_fn = PIIAwareLoss(
            smoothing=LABEL_SMOOTHING,
            alpha=0.3,    # weak CE on PII positions (don't force exact target match)
            beta=2.0,     # strong anti-leakage penalty (push away from original PII)
            ignore_index=-100,
        )
        logger.info(f"  Loss: PIIAwareLoss(smoothing={LABEL_SMOOTHING}, alpha=0.3, beta=2.0)")

        # ── 5. Resume / Warm Start ───────────────────────────────
        best_val_loss = float("inf")
        global_step = 0

        ckpt = load_checkpoint_v2(checkpoint_dir)
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
            "loss_type": "pii_aware",
            "loss_params": {"smoothing": LABEL_SMOOTHING, "alpha": 0.3, "beta": 2.0},
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
                original_ids   = batch["original_token_ids"].to(device)
                pii_mask       = batch["pii_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # ★ PII-Aware Loss
                loss = loss_fn(outputs.logits, labels, original_ids, pii_mask) / accum_steps
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

                    # Log
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
                            save_checkpoint_v2(
                                unwrap(model), optimizer, scheduler,
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# CHECKPOINT STATUS
# ============================================================

def get_checkpoint_status(model_key):
    ckpt_dir = os.path.join(CHECKPOINTS2_DIR, model_key)
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
    model_keys = TRAINING_ORDER

    print("\n" + "=" * 80)
    print("  AVAILABLE MODELS — PII-Aware Loss (saves to checkpoints2/)")
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

    print("\n  Will train (PII-Aware Loss):")
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
    print("  SEQ2SEQ PII ANONYMIZATION — PII-AWARE LOSS TRAINING")
    print("  Saves to: checkpoints2/")
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
        print(f"  [{idx}/{len(models_to_train)}] {model_key} (PII-Aware Loss)")
        print(f"{'#' * 70}")

        ok = train_single_model(model_key, config, device, num_gpus)
        results[model_key] = "SUCCESS" if ok else "FAILED"

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)

    # Summary
    print("\n" + "=" * 90)
    print("  TRAINING SUMMARY (PII-Aware Loss → checkpoints2/)")
    print("=" * 90)
    print(f"  {'Model':<25} {'Status':<10} {'Val Loss':<12} {'Test Loss':<12} {'Test BLEU':<12} {'Leak%'}")
    print("  " + "-" * 80)

    for key, status in results.items():
        icon = "✓" if status == "SUCCESS" else "✗"
        val_loss = test_loss = bleu = leak = "N/A"
        hist_path = os.path.join(CHECKPOINTS2_DIR, key, "training_history.json")
        if (os.path.exists(hist_path)):
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
