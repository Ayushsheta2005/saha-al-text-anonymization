#!/usr/bin/env python3
"""
upload_to_hub.py
================
Standalone script to upload the two generated JSONL files to HuggingFace.
Run this after generate_pairs.py has finished (or partially finished).

Usage:
    python upload_to_hub.py [--combo {1,2,both}] [--hf-token TOKEN]
"""

import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    HF_TOKEN,
    HF_REPO_COMBO1,
    HF_REPO_COMBO2,
    COMBO1_FILE,
    COMBO2_FILE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        log.error(f"File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def upload(base_jsonl: str, repo_id: str, hf_token: str):
    """Upload train/val/test splits as a DatasetDict."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import login

    login(token=hf_token, add_to_git_credential=False)

    prefix = base_jsonl.replace(".jsonl", "")
    splits = {}
    for split in ("train", "val", "test"):
        path = f"{prefix}_{split}.jsonl"
        rows = _read_jsonl(path)
        if not rows:
            log.warning(f"  Skipping empty split: {path}")
            continue
        splits[split] = Dataset.from_list(rows)
        log.info(f"  {split}: {len(rows):,} rows")

    if not splits:
        log.error("No splits found — nothing to upload.")
        return

    dd = DatasetDict(splits)
    dd.push_to_hub(
        repo_id,
        token=hf_token,
        private=False,
        commit_message="Upload train/val/test splits — pipeline mask-fill inversion dataset",
    )
    log.info(f"  Done → https://huggingface.co/datasets/{repo_id}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combo", choices=["1", "2", "both"], default="both")
    p.add_argument("--hf-token", default=None)
    args = p.parse_args()

    token = args.hf_token or HF_TOKEN
    if not token:
        log.error("HuggingFace token not provided. Use --hf-token or set HF_TOKEN env var.")
        sys.exit(1)

    if args.combo in ("1", "both"):
        upload(COMBO1_FILE, HF_REPO_COMBO1, token)

    if args.combo in ("2", "both"):
        upload(COMBO2_FILE, HF_REPO_COMBO2, token)


if __name__ == "__main__":
    main()
