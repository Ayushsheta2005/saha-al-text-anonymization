"""
I/O Helpers
JSONL read / write / append, backup, and dataset id tracking.
"""

import json
import os
import shutil
from datetime import datetime


def read_jsonl(path: str) -> list:
    """Read a .jsonl file, return list of dicts. Empty list on missing file."""
    if not os.path.exists(path):
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def write_jsonl(path: str, data: list) -> None:
    """Overwrite / create a .jsonl file from list of dicts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_jsonl(path: str, entry: dict) -> None:
    """Append a single dict as one JSON line."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def backup_gold_standard(gold_path: str, backup_dir: str) -> str | None:
    """
    Copy the gold-standard file to backup_dir with a timestamp suffix.
    Returns the backup path, or None if the gold file doesn't exist.
    """
    if not os.path.exists(gold_path):
        return None
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(gold_path)
    name, ext = os.path.splitext(base)
    backup_path = os.path.join(backup_dir, f"{name}_{timestamp}{ext}")
    shutil.copy2(gold_path, backup_path)
    return backup_path


def get_annotated_ids(gold_path: str) -> set:
    """Return a set of entry_ids already present in the gold standard."""
    entries = read_jsonl(gold_path)
    return {e.get("entry_id") for e in entries if "entry_id" in e}


def count_lines(path: str) -> int:
    """Count lines in a file efficiently."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)
