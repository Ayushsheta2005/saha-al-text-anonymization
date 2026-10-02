import json
import os
import random
import re
from collections import Counter

GOLD_CUTOFF = 36000


def normalize_entity_type(ent_type):
    if not isinstance(ent_type, str) or not ent_type.strip():
        return "UNKNOWN"
    return ent_type.strip().upper()


def find_non_overlapping_span(text, token, occupied_spans):
    """Find the first non-overlapping token span in text."""
    if not token:
        return None
    start = 0
    while True:
        start = text.find(token, start)
        if start == -1:
            return None
        end = start + len(token)
        overlap = any(not (end <= o_start or start >= o_end) for o_start, o_end in occupied_spans)
        if not overlap:
            return start, end
        start += 1


def prepare_dataset(input_file="anonymized_dataset_final.jsonl", output_dir="data"):
    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Formatting records...")

    all_entries = []
    type_distribution = Counter()
    type_warning_count = 0
    span_warning_count = 0

    for idx, line in enumerate(lines):
        record = json.loads(line)

        entry_id = record.get("entry_id", idx)
        new_id = f"sample_{entry_id:05d}"

        orig_text = record.get("original_text", "")
        anon_text = record.get("anonymized_text", "")

        entities = []
        occupied_spans = []

        for ent in record.get("entities", []):
            text = ent.get("text", "")
            ent_type = normalize_entity_type(ent.get("type", "UNKNOWN"))
            type_distribution[ent_type] += 1
            if ent_type == "UNKNOWN":
                type_warning_count += 1

            if not text:
                span_warning_count += 1
                continue

            start = ent.get("start")
            end = ent.get("end")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(orig_text)
                and orig_text[start:end] == text
                and not any(not (end <= o_start or start >= o_end) for o_start, o_end in occupied_spans)
            ):
                span = (start, end)
            else:
                span = find_non_overlapping_span(orig_text, text, occupied_spans)

            if span is None:
                span_warning_count += 1
                # Keep invalid entity instead of dropping it, to preserve entity counts.
                entities.append({
                    "text": text,
                    "type": ent_type,
                    "start": -1,
                    "end": -1,
                    "invalid": True,
                })
            else:
                start, end = span
                occupied_spans.append((start, end))
                entities.append({
                    "text": text,
                    "type": ent_type,
                    "start": start,
                    "end": end,
                })

        formatted_record = {
            "id": new_id,
            "original_text": orig_text,
            "anonymized_text": anon_text,
            "entities": entities,
        }
        all_entries.append(formatted_record)

    if type_distribution:
        print("\nEntity type distribution:")
        for ent_type, count in type_distribution.most_common():
            print(f"  {ent_type}: {count}")

    if type_warning_count > 0:
        print(f"[WARNING] {type_warning_count} entities normalized to UNKNOWN type")
    if span_warning_count > 0:
        print(f"[WARNING] {span_warning_count} entities could not be aligned with safe spans and were marked invalid")

    print(f"\nSplitting dataset via index cutoff (GOLD_CUTOFF = {GOLD_CUTOFF})...")
    gold_entries = all_entries[:GOLD_CUTOFF]
    augmented_entries = all_entries[GOLD_CUTOFF:]

    random.seed(42)
    random.shuffle(gold_entries)

    total_gold = len(gold_entries)
    train_end = int(0.8 * total_gold)
    val_end = int(0.9 * total_gold)

    train_gold = gold_entries[:train_end]
    gold_val = gold_entries[train_end:val_end]
    gold_test = gold_entries[val_end:]

    train_split = train_gold + augmented_entries

    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "train": train_split,
        "validation": gold_val,
        "test": gold_test,
    }

    for split_name, data in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

    print("\n--- SANITY STATS ---")
    print(f"Train size: {len(train_split)} ({len(train_gold)} gold + {len(augmented_entries)} augmented)")
    print(f"Val size:   {len(gold_val)} (gold only)")
    print(f"Test size:  {len(gold_test)} (gold only)")

    train_ids = set(r["id"] for r in train_split)
    test_ids = set(r["id"] for r in gold_test)
    assert len(train_ids & test_ids) == 0, "Leakage detected!"
    print("[OK] Test set is strictly segregated from train set.")
    print("--------------------\n")

    print("[INFO] Dataset lacks source_id/is_gold metadata; index-based fallback splitting used.")
    print("Dataset preparation complete!")


if __name__ == "__main__":
    prepare_dataset()
