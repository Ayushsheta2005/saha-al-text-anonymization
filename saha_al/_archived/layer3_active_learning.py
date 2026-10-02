"""
Layer 3 — Active Learning (Bootstrap NER + Uncertainty Scoring)
Optional layer that:
  1. Trains a lightweight spaCy NER model on gold-standard data.
  2. Scores un-annotated entries by prediction uncertainty.
  3. Re-sorts YELLOW and RED queues so the most uncertain entries are shown first.
"""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

try:
    import spacy
    from spacy.training import Example
    from spacy.util import minibatch, compounding
except ImportError:
    spacy = None

from saha_al.config import (
    GOLD_STANDARD_PATH,
    YELLOW_QUEUE_PATH,
    RED_QUEUE_PATH,
    BOOTSTRAP_MODEL_DIR,
    BOOTSTRAP_INITIAL_SIZE,
    RETRAIN_EVERY_N,
    UNCERTAINTY_THRESHOLD,
    MODEL_VERSIONS_PATH,
)
from saha_al.utils.io_helpers import read_jsonl, write_jsonl, append_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Layer3] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Train a bootstrap NER model from gold standard
# ─────────────────────────────────────────────────────────────────────
def _gold_to_training_data(gold_entries: list) -> list:
    """
    Convert gold-standard entries to spaCy training format:
      [(text, {"entities": [(start, end, label), ...]}), ...]
    """
    training_data = []
    for entry in gold_entries:
        text = entry.get("original_text", "")
        entities = entry.get("entities", [])
        spacy_ents = []
        for ent in entities:
            start = ent.get("start")
            end = ent.get("end")
            label = ent.get("entity_type", "UNKNOWN")
            if start is not None and end is not None and label != "UNKNOWN":
                spacy_ents.append((start, end, label))
        if text.strip():
            training_data.append((text, {"entities": spacy_ents}))
    return training_data


def train_bootstrap_model(
    gold_path: str = None,
    output_dir: str = None,
    n_iter: int = 10,
    drop: float = 0.35,
) -> str | None:
    """
    Train a lightweight spaCy NER model on gold-standard data.
    
    Returns the path to the saved model, or None if insufficient data.
    """
    if spacy is None:
        log.error("spaCy not installed — cannot train bootstrap model.")
        return None

    gold_path = gold_path or GOLD_STANDARD_PATH
    output_dir = output_dir or BOOTSTRAP_MODEL_DIR

    gold_entries = read_jsonl(gold_path)
    if len(gold_entries) < BOOTSTRAP_INITIAL_SIZE:
        log.warning(
            f"Only {len(gold_entries)} gold entries — need {BOOTSTRAP_INITIAL_SIZE} "
            f"before first bootstrap training. Skipping."
        )
        return None

    log.info(f"Training bootstrap model on {len(gold_entries)} gold entries")
    training_data = _gold_to_training_data(gold_entries)

    if not training_data:
        log.warning("No valid training data after conversion.")
        return None

    # Create a blank NER model
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    # Add labels
    labels = set()
    for _, annotations in training_data:
        for _, _, label in annotations.get("entities", []):
            labels.add(label)
    for label in labels:
        ner.add_label(label)

    # Train
    optimizer = nlp.begin_training()
    for epoch in range(n_iter):
        random.shuffle(training_data)
        losses = {}
        batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            examples = []
            for text, annots in batch:
                try:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annots)
                    examples.append(example)
                except Exception:
                    continue
            if examples:
                nlp.update(examples, sgd=optimizer, drop=drop, losses=losses)
        log.info(f"  Epoch {epoch + 1}/{n_iter}  loss={losses.get('ner', 0):.4f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"bootstrap_{timestamp}")
    nlp.to_disk(model_path)
    log.info(f"Model saved to: {model_path}")

    # Log version
    append_jsonl(MODEL_VERSIONS_PATH, {
        "model_path": model_path,
        "trained_at": timestamp,
        "gold_entries": len(gold_entries),
        "labels": sorted(labels),
        "n_iter": n_iter,
    })

    return model_path


# ─────────────────────────────────────────────────────────────────────
#  Uncertainty scoring
# ─────────────────────────────────────────────────────────────────────
def compute_uncertainty_scores(
    entries: list,
    model_path: str = None,
) -> list:
    """
    Score each entry by the bootstrap model's prediction uncertainty.
    Uncertainty = 1 − (avg confidence of predicted entities).
    Entries with no predictions get uncertainty = 1.0.
    
    Returns entries with added 'uncertainty_score' field.
    """
    if spacy is None:
        log.error("spaCy not installed — cannot score uncertainty.")
        return entries

    if model_path is None:
        # Find latest bootstrap model
        if os.path.exists(BOOTSTRAP_MODEL_DIR):
            subdirs = sorted(
                [
                    d
                    for d in os.listdir(BOOTSTRAP_MODEL_DIR)
                    if os.path.isdir(os.path.join(BOOTSTRAP_MODEL_DIR, d))
                ]
            )
            if subdirs:
                model_path = os.path.join(BOOTSTRAP_MODEL_DIR, subdirs[-1])

    if not model_path or not os.path.exists(model_path):
        log.warning("No bootstrap model available — assigning default uncertainty.")
        for entry in entries:
            entry["uncertainty_score"] = 0.50
        return entries

    log.info(f"Loading bootstrap model from: {model_path}")
    nlp = spacy.load(model_path)

    for entry in entries:
        text = entry.get("original_text", "")
        if not text.strip():
            entry["uncertainty_score"] = 0.0
            continue
        try:
            doc = nlp(text)
            if doc.ents:
                # spaCy blank models don't have confidence scores on entities by default,
                # so we approximate uncertainty based on entity count difference
                # between bootstrap and pre-annotation
                pre_count = len(entry.get("entities", []))
                boot_count = len(doc.ents)
                diff = abs(pre_count - boot_count)
                max_count = max(pre_count, boot_count, 1)
                entry["uncertainty_score"] = round(diff / max_count, 4)
            else:
                # No entities predicted → high uncertainty if pre-annotation found some
                if entry.get("entities"):
                    entry["uncertainty_score"] = 0.90
                else:
                    entry["uncertainty_score"] = 0.10
        except Exception as exc:
            log.error(f"Scoring entry {entry.get('entry_id')}: {exc}")
            entry["uncertainty_score"] = 0.50

    return entries


# ─────────────────────────────────────────────────────────────────────
#  Re-sort queues by uncertainty
# ─────────────────────────────────────────────────────────────────────
def resort_queues(
    model_path: str = None,
    yellow_path: str = None,
    red_path: str = None,
):
    """
    Load YELLOW and RED queues, score uncertainty, re-sort descending
    (most uncertain first), and rewrite the queue files.
    """
    yellow_path = yellow_path or YELLOW_QUEUE_PATH
    red_path = red_path or RED_QUEUE_PATH

    for queue_path, queue_name in [(yellow_path, "YELLOW"), (red_path, "RED")]:
        entries = read_jsonl(queue_path)
        if not entries:
            log.info(f"{queue_name} queue is empty — skipping.")
            continue

        log.info(f"Scoring {len(entries)} {queue_name} entries...")
        entries = compute_uncertainty_scores(entries, model_path)

        # Sort: highest uncertainty first
        entries.sort(key=lambda x: x.get("uncertainty_score", 0), reverse=True)
        write_jsonl(queue_path, entries)
        log.info(f"{queue_name} queue re-sorted by uncertainty.")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer 3 — Active Learning")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train bootstrap NER model")
    train_p.add_argument("--gold", type=str, default=None)
    train_p.add_argument("--output", type=str, default=None)
    train_p.add_argument("--n-iter", type=int, default=10)

    score_p = sub.add_parser("score", help="Score queues by uncertainty & re-sort")
    score_p.add_argument("--model", type=str, default=None)
    score_p.add_argument("--yellow", type=str, default=None)
    score_p.add_argument("--red", type=str, default=None)

    args = parser.parse_args()

    if args.command == "train":
        train_bootstrap_model(
            gold_path=args.gold,
            output_dir=args.output,
            n_iter=args.n_iter,
        )
    elif args.command == "score":
        resort_queues(
            model_path=args.model,
            yellow_path=args.yellow,
            red_path=args.red,
        )
    else:
        parser.print_help()
