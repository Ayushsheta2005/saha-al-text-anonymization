#!/usr/bin/env python3
"""
scripts/partition_queue.py
Split annotation_queue.jsonl into 3 annotator-specific slices.

Usage:
    python3 -m saha_al.scripts.partition_queue [--overlap 0.1]

Output:
    data/queue_A1.jsonl  — exclusive slice for Annotator 1
    data/queue_A2.jsonl  — exclusive slice for Annotator 2
    data/queue_A3.jsonl  — exclusive slice for Annotator 3
    data/queue_overlap.jsonl  — shared overlap slice (10% by default), all 3 annotate

The overlap set is used to calculate Inter-Annotator Agreement (IAA).
"""

import os
import sys
import argparse
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from saha_al.config import ANNOTATION_QUEUE_PATH, DATA_DIR
from saha_al.utils.io_helpers import read_jsonl, write_jsonl

ANNOTATORS = ["A1", "A2", "A3"]


def partition(overlap_fraction: float = 0.10):
    queue = read_jsonl(ANNOTATION_QUEUE_PATH)
    total = len(queue)

    overlap_n = math.ceil(total * overlap_fraction)
    exclusive_n = total - overlap_n

    # Overlap goes to the BEGINNING so it's the first thing each annotator sees
    overlap_slice = queue[:overlap_n]
    exclusive = queue[overlap_n:]

    per_annotator = math.ceil(exclusive_n / len(ANNOTATORS))

    print(f"Total queue:      {total:,}")
    print(f"Overlap slice:    {overlap_n:,}  ({overlap_fraction*100:.0f}%)")
    print(f"Exclusive total:  {exclusive_n:,}")
    print(f"Per annotator:    ~{per_annotator:,}")

    for i, code in enumerate(ANNOTATORS):
        start = i * per_annotator
        end = min(start + per_annotator, exclusive_n)
        # Each annotator gets the overlap set first, then their exclusive slice
        annotator_queue = overlap_slice + exclusive[start:end]
        out_path = os.path.join(DATA_DIR, f"queue_{code}.jsonl")
        write_jsonl(out_path, annotator_queue)
        print(f"  → {out_path}  ({len(annotator_queue):,} entries)")

    # Also save the standalone overlap file for IAA analysis
    overlap_path = os.path.join(DATA_DIR, "queue_overlap.jsonl")
    write_jsonl(overlap_path, overlap_slice)
    print(f"\nOverlap set saved to: {overlap_path}")
    print("\nDone! Each annotator should open layer4_app.py and point ANNOTATION_QUEUE_PATH")
    print("to their own queue_A1/A2/A3.jsonl file (or use the --annotator CLI flag).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition annotation queue for 3 annotators.")
    parser.add_argument("--overlap", type=float, default=0.10,
                        help="Fraction of entries all 3 annotators share (for IAA). Default: 0.10")
    args = parser.parse_args()
    partition(args.overlap)
