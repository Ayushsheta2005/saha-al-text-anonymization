"""
Layer 2 — Confidence-Based Routing
Reads pre_annotated.jsonl and routes each entry into:
  GREEN  – high confidence, auto-approvable
  YELLOW – medium confidence, needs human review
  RED    – low confidence or disagreement, needs expert review
"""

import logging
from datetime import datetime

from saha_al.config import (
    PRE_ANNOTATED_PATH,
    GREEN_QUEUE_PATH,
    YELLOW_QUEUE_PATH,
    RED_QUEUE_PATH,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)
from saha_al.utils.io_helpers import read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Layer2] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Classification logic
# ─────────────────────────────────────────────────────────────────────
def classify_entry(entry: dict) -> str:
    """
    Classify a pre-annotated entry into GREEN / YELLOW / RED.
    
    Rules:
      GREEN  → all entities have agreement="full" or "single_source" with
               confidence ≥ CONFIDENCE_HIGH, and no UNKNOWN types.
      RED    → any entity has agreement="partial" with conflicting types,
               OR any entity has confidence < CONFIDENCE_MEDIUM,
               OR more than 30% of entities are UNKNOWN.
      YELLOW → everything in between.
    """
    entities = entry.get("entities", [])

    # No entities? Probably short text — mark GREEN
    if not entities:
        return "GREEN"

    confidences = [e.get("confidence", 0) for e in entities]
    agreements = [e.get("agreement", "single_source") for e in entities]
    types = [e.get("entity_type", "UNKNOWN") for e in entities]

    min_conf = min(confidences) if confidences else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    unknown_ratio = types.count("UNKNOWN") / len(types) if types else 0
    partial_count = agreements.count("partial")

    # RED conditions
    if min_conf < CONFIDENCE_MEDIUM:
        return "RED"
    if unknown_ratio > 0.30:
        return "RED"
    if partial_count >= 2:
        return "RED"

    # GREEN conditions
    if min_conf >= CONFIDENCE_HIGH and partial_count == 0 and unknown_ratio == 0:
        return "GREEN"

    # Otherwise YELLOW
    return "YELLOW"


def add_routing_metadata(entry: dict, queue: str) -> dict:
    """Add routing metadata to an entry."""
    entry["routing"] = {
        "queue": queue,
        "routed_at": datetime.now().isoformat(),
        "auto_approved": queue == "GREEN",
    }
    return entry


# ─────────────────────────────────────────────────────────────────────
#  Main batch runner
# ─────────────────────────────────────────────────────────────────────
def main(
    input_path: str = None,
    green_path: str = None,
    yellow_path: str = None,
    red_path: str = None,
):
    """
    Route all pre-annotated entries into queues.
    Overwrites existing queue files.
    """
    input_path = input_path or PRE_ANNOTATED_PATH
    green_path = green_path or GREEN_QUEUE_PATH
    yellow_path = yellow_path or YELLOW_QUEUE_PATH
    red_path = red_path or RED_QUEUE_PATH

    log.info(f"Reading pre-annotated data from: {input_path}")
    entries = read_jsonl(input_path)
    log.info(f"Loaded {len(entries)} entries")

    green, yellow, red = [], [], []

    for entry in entries:
        queue = classify_entry(entry)
        entry = add_routing_metadata(entry, queue)
        if queue == "GREEN":
            green.append(entry)
        elif queue == "YELLOW":
            yellow.append(entry)
        else:
            red.append(entry)

    # Write queues
    write_jsonl(green_path, green)
    write_jsonl(yellow_path, yellow)
    write_jsonl(red_path, red)

    total = len(entries)
    log.info(f"Routing complete:")
    log.info(f"  GREEN:  {len(green):>6} ({100*len(green)/total:.1f}%)" if total else "  GREEN: 0")
    log.info(f"  YELLOW: {len(yellow):>6} ({100*len(yellow)/total:.1f}%)" if total else "  YELLOW: 0")
    log.info(f"  RED:    {len(red):>6} ({100*len(red)/total:.1f}%)" if total else "  RED: 0")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer 2 — Routing")
    parser.add_argument("--input", type=str, default=None, help="Pre-annotated JSONL path")
    parser.add_argument("--green", type=str, default=None, help="Green queue output path")
    parser.add_argument("--yellow", type=str, default=None, help="Yellow queue output path")
    parser.add_argument("--red", type=str, default=None, help="Red queue output path")
    args = parser.parse_args()

    main(
        input_path=args.input,
        green_path=args.green,
        yellow_path=args.yellow,
        red_path=args.red,
    )
