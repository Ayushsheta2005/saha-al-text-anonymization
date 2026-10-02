"""
Data Preparation Script
-----------------------
Loads the anonymized_dataset_final.jsonl and splits it into train/val/test sets.
Saves splits as separate JSONL files for reproducibility.
"""

import json
import os
import random
from collections import Counter

from config import (
    DATASET_PATH,
    DATA_SPLITS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def load_dataset(path: str) -> list[dict]:
    """Load JSONL dataset, filtering out entries with empty text."""
    data = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                orig = entry.get("original_text", "").strip()
                anon = entry.get("anonymized_text", "").strip()
                # Skip entries where either field is empty or they are identical
                # (identical means no anonymization happened — useless for training)
                if orig and anon and orig != anon:
                    # Extract entity texts for leakage detection
                    entity_texts = [
                        e.get("text", "").strip()
                        for e in entry.get("entities", [])
                        if e.get("text", "").strip()
                    ]
                    data.append({
                        "entry_id": entry.get("entry_id", line_num),
                        "original_text": orig,
                        "anonymized_text": anon,
                        "num_entities": len(entry.get("entities", [])),
                        "entity_texts": entity_texts,
                    })
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
                print(f"  [WARN] Skipping malformed JSON at line {line_num}")

    print(f"  Loaded {len(data)} valid entries, skipped {skipped}")
    return data


def split_dataset(data: list[dict], train_r: float, val_r: float, test_r: float, seed: int):
    """Shuffle and split data into train/val/test."""
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def save_split(data: list[dict], filepath: str):
    """Save a data split as JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_stats(name: str, data: list[dict]):
    """Print statistics about a data split."""
    if not data:
        print(f"  {name}: 0 entries")
        return

    lengths_in = [len(d["original_text"].split()) for d in data]
    lengths_out = [len(d["anonymized_text"].split()) for d in data]
    entity_counts = [d["num_entities"] for d in data]

    print(f"  {name}:")
    print(f"    Entries       : {len(data)}")
    print(f"    Avg input len : {sum(lengths_in)/len(lengths_in):.1f} words")
    print(f"    Avg output len: {sum(lengths_out)/len(lengths_out):.1f} words")
    print(f"    Avg entities  : {sum(entity_counts)/len(entity_counts):.1f}")
    print(f"    Max input len : {max(lengths_in)} words")
    print(f"    Max output len: {max(lengths_out)} words")


def prepare_data():
    """Main function: load, split, save, and print stats."""
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Check if splits already exist
    train_path = os.path.join(DATA_SPLITS_DIR, "train.jsonl")
    val_path = os.path.join(DATA_SPLITS_DIR, "val.jsonl")
    test_path = os.path.join(DATA_SPLITS_DIR, "test.jsonl")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("\n[INFO] Data splits already exist. Loading existing splits...")
        train_data = load_split(train_path)
        val_data = load_split(val_path)
        test_data = load_split(test_path)
    else:
        print(f"\n[1/3] Loading dataset from: {DATASET_PATH}")
        data = load_dataset(DATASET_PATH)

        print(f"\n[2/3] Splitting data ({TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO})...")
        train_data, val_data, test_data = split_dataset(
            data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
        )

        print(f"\n[3/3] Saving splits to: {DATA_SPLITS_DIR}")
        os.makedirs(DATA_SPLITS_DIR, exist_ok=True)
        save_split(train_data, train_path)
        save_split(val_data, val_path)
        save_split(test_data, test_path)

    print("\n--- Dataset Statistics ---")
    print_stats("Train", train_data)
    print_stats("Val  ", val_data)
    print_stats("Test ", test_data)
    print("=" * 60)

    return train_data, val_data, test_data


def load_split(filepath: str) -> list[dict]:
    """Load a previously saved JSONL split."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    prepare_data()
