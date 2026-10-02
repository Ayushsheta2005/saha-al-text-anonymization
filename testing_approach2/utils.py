"""
Utility functions for Approach 2 Pipeline Evaluation
=====================================================
Metrics: Exact Match, Word Accuracy, BLEU, ROUGE, BERTScore,
         Entity Leakage Rate, Masker Detection Rate, Parameter count.
"""

import gc
import torch
import numpy as np
import time


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_millions": round(total / 1e6, 1),
        "trainable_millions": round(trainable / 1e6, 1),
    }


def aggressive_cleanup():
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()


# ─── Text-level metrics ────────────────────────────────────────────────────────

def compute_exact_match(preds: list, targets: list) -> float:
    if not preds:
        return 0.0
    return sum(p.strip() == t.strip() for p, t in zip(preds, targets)) / len(preds) * 100


def compute_word_accuracy(preds: list, targets: list) -> float:
    if not preds:
        return 0.0
    scores = []
    for p, t in zip(preds, targets):
        pw, tw = p.strip().split(), t.strip().split()
        if not tw:
            continue
        matches = sum(pw_ == tw_ for pw_, tw_ in zip(pw, tw))
        scores.append(matches / max(len(tw), 1))
    return (sum(scores) / len(scores) * 100) if scores else 0.0


def compute_bleu(preds: list, targets: list) -> dict:
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(preds, [targets])
        return {
            "bleu":   round(bleu.score, 2),
            "bleu_1": round(bleu.precisions[0], 2),
            "bleu_2": round(bleu.precisions[1], 2),
            "bleu_4": round(bleu.precisions[3], 2),
        }
    except Exception as e:
        print(f"  [WARN] BLEU failed: {e}")
        return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}


def compute_rouge(preds: list, targets: list) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for p, t in zip(preds, targets):
            s = scorer.score(t, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return {
            "rouge_1": round(np.mean(r1) * 100, 2),
            "rouge_2": round(np.mean(r2) * 100, 2),
            "rouge_l": round(np.mean(rl) * 100, 2),
        }
    except Exception as e:
        print(f"  [WARN] ROUGE failed: {e}")
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


def compute_bertscore(preds: list, targets: list) -> dict:
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            preds, targets,
            model_type="distilbert-base-uncased",
            num_layers=5,
            batch_size=32,
            verbose=False,
            device="cpu",
        )
        return {
            "bertscore_precision": round(P.mean().item() * 100, 2),
            "bertscore_recall":    round(R.mean().item() * 100, 2),
            "bertscore_f1":        round(F1.mean().item() * 100, 2),
        }
    except Exception as e:
        print(f"  [WARN] BERTScore failed: {e}")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


# ─── Privacy metrics ───────────────────────────────────────────────────────────

def compute_entity_leakage(
    preds: list,
    original_texts: list,
    entity_texts_list: list,
) -> dict:
    """
    Entity Leakage Rate: fraction of original PII entities that still appear
    verbatim (case-insensitive) in the pipeline's output.

    Lower = better (0.0 means no leakage — ideal for privacy).

    Returns:
        leakage_rate          – fraction of *samples* with at least one leaked entity
        entity_leakage_rate   – fraction of *individual* entities that leaked
        total_entities_checked
        total_entities_leaked
        leaked_entities_top10 – list of (entity_string, count)
    """
    if not preds or not entity_texts_list:
        return {
            "leakage_rate": 0.0,
            "entity_leakage_rate": 0.0,
            "total_entities_checked": 0,
            "total_entities_leaked": 0,
            "leaked_entities_top10": [],
        }

    samples_leaked = 0
    total_entities = 0
    leaked_entities_count = 0
    leaked_entity_counter: dict = {}

    for pred, orig, entities in zip(preds, original_texts, entity_texts_list):
        if not entities:
            continue
        pred_lower = pred.lower()
        sample_leaked = False
        for ent in entities:
            if not ent or len(ent) < 2:
                continue
            total_entities += 1
            if ent.lower() in pred_lower:
                leaked_entities_count += 1
                sample_leaked = True
                leaked_entity_counter[ent] = leaked_entity_counter.get(ent, 0) + 1
        if sample_leaked:
            samples_leaked += 1

    leakage_rate = (samples_leaked / max(len(preds), 1)) * 100
    entity_leakage_rate = (leaked_entities_count / max(total_entities, 1)) * 100

    top10 = sorted(leaked_entity_counter.items(), key=lambda x: -x[1])[:10]

    return {
        "leakage_rate":          round(leakage_rate, 2),
        "entity_leakage_rate":   round(entity_leakage_rate, 2),
        "total_entities_checked": total_entities,
        "total_entities_leaked":  leaked_entities_count,
        "leaked_entities_top10": top10,
    }


def compute_masker_detection_rate(
    masked_texts: list,
    entity_texts_list: list,
) -> dict:
    """
    Masker detection rate: for each sample, check how many original
    entities are no longer visible in the masked text (i.e., the masker
    replaced them with [ENTITY_TYPE] placeholders).

    detection_rate = fraction of entities that were masked out
    """
    if not masked_texts or not entity_texts_list:
        return {"masker_detection_rate": 0.0, "samples_with_any_mask": 0.0}

    total_entities = 0
    detected_entities = 0
    samples_with_any_mask = 0

    for masked, entities in zip(masked_texts, entity_texts_list):
        if not entities:
            continue
        masked_lower = masked.lower()
        sample_any = False
        for ent in entities:
            if not ent or len(ent) < 2:
                continue
            total_entities += 1
            if ent.lower() not in masked_lower:
                detected_entities += 1
                sample_any = True
        if sample_any:
            samples_with_any_mask += 1

    detection_rate = (detected_entities / max(total_entities, 1)) * 100
    samples_pct = (samples_with_any_mask / max(len(masked_texts), 1)) * 100

    return {
        "masker_detection_rate":  round(detection_rate, 2),
        "samples_with_any_mask":  round(samples_pct, 2),
        "total_entities":         total_entities,
        "detected_entities":      detected_entities,
    }


def compute_all_metrics(
    preds: list,
    targets: list,
    original_texts: list,
    entity_texts_list: list,
    masked_texts: list = None,
    compute_bert: bool = True,
) -> dict:
    """Compute the full metrics suite for a set of pipeline predictions."""
    metrics = {}
    metrics["exact_match"]    = compute_exact_match(preds, targets)
    metrics["word_accuracy"]  = compute_word_accuracy(preds, targets)
    metrics.update(compute_bleu(preds, targets))
    metrics.update(compute_rouge(preds, targets))
    if compute_bert:
        metrics.update(compute_bertscore(preds, targets))
    else:
        metrics.update({"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0})
    leakage = compute_entity_leakage(preds, original_texts, entity_texts_list)
    metrics.update(leakage)
    if masked_texts is not None:
        masker_stats = compute_masker_detection_rate(masked_texts, entity_texts_list)
        metrics.update(masker_stats)
    return metrics
