"""
Utility functions for Seq2Seq training pipeline.
- GPU memory management (aggressive cleanup to prevent OOM)
- Logging setup
- Metric computation
- Checkpoint save/load helpers
"""

import gc
import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime


# ============================================================
# GPU MEMORY MANAGEMENT — CRITICAL FOR 4GB VRAM
# ============================================================

def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    free = total - reserved
    return {
        "total_mb": round(total, 1),
        "reserved_mb": round(reserved, 1),
        "allocated_mb": round(allocated, 1),
        "free_mb": round(free, 1),
    }


def aggressive_cleanup():
    """
    Aggressively free GPU and CPU memory.
    Call this BETWEEN model trainings to ensure the previous model
    is fully evicted from VRAM before the next one loads.
    """
    # Force garbage collection multiple times
    gc.collect()
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Another round after CUDA cleanup
    gc.collect()


def cleanup_model_from_memory(model, optimizer=None, scheduler=None, scaler=None):
    """
    Completely remove a model and its optimizer/scheduler from memory.
    This is the nuclear option — ensures nothing lingers in VRAM.
    """
    # Delete optimizer first (it holds references to model params)
    if optimizer is not None:
        del optimizer
    if scheduler is not None:
        del scheduler
    if scaler is not None:
        del scaler

    # Move model to CPU first, then delete
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del model

    aggressive_cleanup()

    mem = get_gpu_memory_info()
    print(f"  [CLEANUP] GPU after cleanup: {mem['allocated_mb']:.0f}MB allocated, "
          f"{mem['free_mb']:.0f}MB free / {mem['total_mb']:.0f}MB total")


def check_gpu_before_training(model_name: str, min_free_mb: float = 2000.0) -> bool:
    """Check if we have enough free GPU memory to start training."""
    mem = get_gpu_memory_info()
    print(f"\n  [GPU CHECK] Before loading '{model_name}':")
    print(f"    Allocated: {mem['allocated_mb']:.0f}MB | "
          f"Free: {mem['free_mb']:.0f}MB | Total: {mem['total_mb']:.0f}MB")

    if mem["free_mb"] < min_free_mb:
        print(f"  [WARNING] Only {mem['free_mb']:.0f}MB free. "
              f"Need at least {min_free_mb:.0f}MB. Running aggressive cleanup...")
        aggressive_cleanup()
        mem = get_gpu_memory_info()
        if mem["free_mb"] < min_free_mb:
            print(f"  [ERROR] Still only {mem['free_mb']:.0f}MB free after cleanup. "
                  f"Skipping {model_name}.")
            return False
    return True


# ============================================================
# LOGGING
# ============================================================

def setup_logger(model_name: str, log_dir: str) -> logging.Logger:
    """Create a logger that writes to a log file only (no console).
    
    Console output during training is handled by tqdm exclusively.
    Using a StreamHandler here would conflict with tqdm's progress bar,
    causing the bar to be reprinted on every logger.info() call.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent bubbling to root logger

    # Remove existing handlers (in case of re-runs)
    logger.handlers.clear()

    # File handler only — no console handler
    log_file = os.path.join(log_dir, f"{model_name}.log")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)

    # Format
    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)

    logger.addHandler(fh)

    return logger


# ============================================================
# LABEL SMOOTHED CROSS-ENTROPY LOSS
# ============================================================

class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Label Smoothed Cross-Entropy Loss for Seq2Seq models.
    
    Why use this for PII anonymization?
    ------------------------------------
    Standard CE loss assigns 100% probability to the single "correct" target token.
    But in PII replacement, MANY replacements are valid:
      - "John" → "James", "Alex", "David" are all correct
      - "10:17" → "11:30", "09:45", "14:00" are all correct
    
    Label smoothing distributes a small fraction (ε) of probability across ALL tokens
    in the vocabulary, preventing the model from being overconfident about one specific
    replacement. This leads to better generalization.
    
    Used by: Original T5 paper, BART paper, most modern seq2seq models.
    
    Args:
        smoothing (float): Label smoothing factor. 0.0 = standard CE, 0.1 = recommended.
        ignore_index (int): Token ID to ignore in loss (typically -100 for padding).
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size) — model output logits
            labels: (batch_size, seq_len) — target token IDs, with -100 for padding
        Returns:
            Scalar loss value
        """
        vocab_size = logits.size(-1)
        
        # Cast to float32 to prevent fp16 overflow.
        # log_softmax produces large negative values for unlikely tokens (e.g., -65500).
        # Summing ~32k of these overflows fp16 max range (~65504) → -inf → nan.
        logits_flat = logits.view(-1, vocab_size).float()
        labels_flat = labels.view(-1)
        
        # Create mask for non-padding tokens
        non_pad_mask = labels_flat != self.ignore_index
        
        # Get log probabilities
        log_probs = F.log_softmax(logits_flat, dim=-1)
        
        # Standard NLL loss component (1 - ε) on the correct token
        nll_loss = F.nll_loss(
            log_probs, 
            labels_flat.clamp(min=0),  # clamp -100 to 0 to avoid index error
            reduction='none'
        )
        
        # Smooth loss component (ε / V) spread across all tokens
        smooth_loss = -log_probs.sum(dim=-1) / vocab_size
        
        # Combine: (1 - ε) * NLL + ε * smooth
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        # Apply padding mask and average
        loss = loss[non_pad_mask].mean()
        
        return loss


# ============================================================
# METRICS
# ============================================================

def compute_token_accuracy(preds: list[str], targets: list[str]) -> float:
    """
    Compute simple exact-match accuracy at the sentence level.
    Returns fraction of predictions that exactly match the target.
    """
    if not preds:
        return 0.0
    correct = sum(1 for p, t in zip(preds, targets) if p.strip() == t.strip())
    return correct / len(preds)


def compute_word_level_accuracy(preds: list[str], targets: list[str]) -> float:
    """
    Compute word-level accuracy: avg fraction of matching words per sample.
    More forgiving than exact match.
    """
    if not preds:
        return 0.0

    accuracies = []
    for p, t in zip(preds, targets):
        pred_words = p.strip().split()
        target_words = t.strip().split()
        if not target_words:
            continue
        matches = sum(1 for pw, tw in zip(pred_words, target_words) if pw == tw)
        accuracies.append(matches / max(len(target_words), 1))

    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def compute_bleu(preds: list[str], targets: list[str]) -> dict:
    """
    BLEU Score (Bilingual Evaluation Understudy).
    
    Measures n-gram overlap between predictions and targets.
    Standard metric for machine translation & text rewriting.
    
    Returns BLEU-1, BLEU-2, BLEU-4, and overall BLEU score.
    
    For PII anonymization:
      - High BLEU means the model preserves most of the original structure
      - Non-PII parts should match exactly → high n-gram overlap
    """
    import sacrebleu
    
    if not preds or not targets:
        return {"bleu": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}
    
    try:
        # sacrebleu expects: hypotheses as list[str], references as list[list[str]]
        # Overall BLEU (uses 1-4 grams with brevity penalty)
        bleu = sacrebleu.corpus_bleu(preds, [targets])
        
        # bleu.precisions = [unigram%, bigram%, trigram%, 4gram%]
        return {
            "bleu": round(bleu.score, 2),
            "bleu1": round(bleu.precisions[0], 2),
            "bleu2": round(bleu.precisions[1], 2),
            "bleu4": round(bleu.precisions[3], 2),
        }
    except Exception as e:
        print(f"  [WARN] BLEU computation failed: {e}")
        return {"bleu": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}


def compute_rouge(preds: list[str], targets: list[str]) -> dict:
    """
    ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation).
    
    - ROUGE-1: Unigram overlap (individual word matching)
    - ROUGE-2: Bigram overlap (two-word phrase matching)  
    - ROUGE-L: Longest Common Subsequence (captures sentence structure)
    
    For PII anonymization:
      - ROUGE-L is most important — it measures whether the overall sentence
        structure is preserved even when specific PII tokens change.
    """
    from rouge_score import rouge_scorer
    
    if not preds or not targets:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        r1_scores, r2_scores, rl_scores = [], [], []
        for pred, target in zip(preds, targets):
            scores = scorer.score(target, pred)
            r1_scores.append(scores['rouge1'].fmeasure)
            r2_scores.append(scores['rouge2'].fmeasure)
            rl_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": round(np.mean(r1_scores) * 100, 2),
            "rouge2": round(np.mean(r2_scores) * 100, 2),
            "rougeL": round(np.mean(rl_scores) * 100, 2),
        }
    except Exception as e:
        print(f"  [WARN] ROUGE computation failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore(preds: list[str], targets: list[str]) -> dict:
    """
    BERTScore: Semantic similarity using contextual BERT embeddings.
    
    Unlike BLEU/ROUGE which count exact word matches, BERTScore computes
    cosine similarity between BERT token embeddings of pred vs target.
    
    This captures cases where the meaning is preserved even if wording differs:
      Target: "Contact James at 555-1234"
      Pred:   "Contact James at 555-1234"   → BERTScore ≈ 1.0
      Pred:   "Reach James on 555-1234"     → BERTScore ≈ 0.95 (BLEU would be lower)
    
    For PII anonymization:
      - Measures if the overall meaning/structure is semantically preserved
      - More forgiving than BLEU for paraphrased outputs
    
    Note: Uses 'distilbert-base-uncased' to keep GPU memory low.
    """
    try:
        from bert_score import score as bert_score_fn
        
        if not preds or not targets:
            return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}
        
        # Use distilbert to save memory (model is ~260MB vs ~440MB for bert-base)
        P, R, F1 = bert_score_fn(
            preds, targets,
            model_type="distilbert-base-uncased",
            num_layers=5,
            batch_size=32,
            verbose=False,
            device="cpu",  # Run on CPU to avoid GPU memory conflicts during training
        )
        
        return {
            "bertscore_p": round(P.mean().item() * 100, 2),
            "bertscore_r": round(R.mean().item() * 100, 2),
            "bertscore_f1": round(F1.mean().item() * 100, 2),
        }
    except Exception as e:
        print(f"  [WARN] BERTScore computation failed: {e}")
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}


def compute_entity_leakage(
    preds: list[str],
    original_texts: list[str],
    entity_texts_list: list[list[str]],
) -> dict:
    """
    Entity Leakage Rate: THE most important privacy metric.
    
    Checks if any original PII entities leaked into the model's predictions.
    If the model outputs "Meet Yohann at Koratagere" when the original had
    "Yohann" and "Koratagere" as PII entities, that's a PRIVACY FAILURE.
    
    Returns:
      - leakage_rate: fraction of samples where at least 1 entity leaked
      - entity_leakage_rate: fraction of individual entities that leaked
      - leaked_entities: list of (entity, count) showing which entities leaked most
    
    Lower = better. Ideal = 0.0 (no leakage).
    """
    if not preds or not entity_texts_list:
        return {"leakage_rate": 0.0, "entity_leakage_rate": 0.0, "leaked_entities": []}
    
    samples_with_leakage = 0
    total_entities = 0
    leaked_entities_count = 0
    leaked_entities_examples = []
    
    for pred, orig, entities in zip(preds, original_texts, entity_texts_list):
        if not entities:
            continue
        
        pred_lower = pred.lower()
        sample_leaked = False
        
        for entity in entities:
            if not entity or len(entity) < 2:  # skip very short entities (single chars)
                continue
            
            total_entities += 1
            
            # Check if the original entity appears in the prediction
            if entity.lower() in pred_lower:
                leaked_entities_count += 1
                sample_leaked = True
                leaked_entities_examples.append(entity)
        
        if sample_leaked:
            samples_with_leakage += 1
    
    # Count most common leaked entities
    from collections import Counter
    leaked_counter = Counter(leaked_entities_examples).most_common(10)
    
    return {
        "leakage_rate": round(samples_with_leakage / max(len(preds), 1) * 100, 2),
        "entity_leakage_rate": round(leaked_entities_count / max(total_entities, 1) * 100, 2),
        "total_entities_checked": total_entities,
        "total_entities_leaked": leaked_entities_count,
        "leaked_entities_top10": leaked_counter,
    }


def compute_all_metrics(
    preds: list[str],
    targets: list[str],
    original_texts: list[str] = None,
    entity_texts_list: list[list[str]] = None,
    compute_bert: bool = True,
) -> dict:
    """
    Compute ALL evaluation metrics in one call.
    
    Args:
        preds: model predictions (anonymized text)
        targets: ground truth anonymized text
        original_texts: original text with real PII (for leakage check)
        entity_texts_list: list of entity texts per sample (for leakage check)
        compute_bert: whether to compute BERTScore (slow, skip during training)
    
    Returns dict with all metrics.
    """
    metrics = {}
    
    # 1. Exact Match & Word Accuracy (fast)
    metrics["exact_match"] = round(compute_token_accuracy(preds, targets) * 100, 2)
    metrics["word_accuracy"] = round(compute_word_level_accuracy(preds, targets) * 100, 2)
    
    # 2. BLEU (fast)
    bleu_scores = compute_bleu(preds, targets)
    metrics.update(bleu_scores)
    
    # 3. ROUGE (fast)
    rouge_scores = compute_rouge(preds, targets)
    metrics.update(rouge_scores)
    
    # 4. BERTScore (slow — only for final eval, not during training)
    if compute_bert:
        bert_scores = compute_bertscore(preds, targets)
        metrics.update(bert_scores)
    
    # 5. Entity Leakage (fast, but needs entity data)
    if original_texts is not None and entity_texts_list is not None:
        leakage = compute_entity_leakage(preds, original_texts, entity_texts_list)
        metrics["leakage_rate"] = leakage["leakage_rate"]
        metrics["entity_leakage_rate"] = leakage["entity_leakage_rate"]
        metrics["total_entities_checked"] = leakage["total_entities_checked"]
        metrics["total_entities_leaked"] = leakage["total_entities_leaked"]
        metrics["leaked_entities_top10"] = leakage.get("leaked_entities_top10", [])
    
    return metrics


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def get_checkpoint_dir(checkpoints_base: str, model_name: str) -> str:
    """Get the checkpoint directory for a specific model."""
    path = os.path.join(checkpoints_base, model_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    checkpoint_dir: str,
    model_config: dict = None,
    use_qlora: bool = False,
):
    """
    Save the BEST model checkpoint only.
    Overwrites previous best — so each model subdirectory has exactly 1 .pt file.
    
    Structure:
        checkpoints/
            t5-efficient-tiny/
                best_model.pt
                training_history.json
            t5-small/
                best_model.pt
                training_history.json
            ...
    """
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_config": model_config,
        "timestamp": datetime.now().isoformat(),
    }

    # For QLoRA, save only the adapter weights (much smaller)
    if use_qlora:
        adapter_path = os.path.join(checkpoint_dir, "lora_adapter")
        model.save_pretrained(adapter_path)
        checkpoint["adapter_path"] = adapter_path
    else:
        checkpoint["model_state_dict"] = model.state_dict()

    checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Single file per model — always overwritten with the best
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save(checkpoint, best_path)

    return best_path


def load_checkpoint(checkpoint_dir: str) -> dict | None:
    """
    Load the best model checkpoint. Returns None if no checkpoint exists.
    
    Uses map_location="cpu" intentionally:
      - Loads weights into RAM first (cheap, plenty of space)
      - Then model.load_state_dict() copies them to GPU, overwriting in-place
      - Avoids having 2 copies of weights on GPU simultaneously (would OOM on 4GB)
    """
    path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def save_training_history(history: dict, checkpoint_dir: str):
    """Save training metrics history as JSON."""
    path = os.path.join(checkpoint_dir, "training_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# ============================================================
# MISC
# ============================================================

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def count_parameters(model) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": round(total / 1e6, 2),
        "trainable_millions": round(trainable / 1e6, 2),
    }
