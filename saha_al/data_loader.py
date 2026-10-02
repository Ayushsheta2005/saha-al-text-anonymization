"""
Data Loader for SAHA-AL Pipeline
Pulls directly from huggingface dataset: ai4privacy/open-pii-masking-500k-ai4privacy
Filters to English entries and populates the annotation_queue.
"""

import os
import json
import logging
from tqdm import tqdm
from datasets import load_dataset
from saha_al.config import HF_DATASET_NAME, ANNOTATION_QUEUE_PATH, GOLD_STANDARD_PATH
from saha_al.utils.io_helpers import write_jsonl, append_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

def load_annotation_queue(limit=None):
    log.info(f"Loading dataset {HF_DATASET_NAME} from HuggingFace...")
    dataset = load_dataset(HF_DATASET_NAME, split="train")

    log.info("Filtering for English (language == 'en')...")
    en_dataset = dataset.filter(lambda x: x["language"] == "en")
    log.info(f"Found {len(en_dataset)} English entries.")

    if limit is not None:
        en_dataset = en_dataset.select(range(min(limit, len(en_dataset))))
        log.info(f"Limited to {limit} entries.")

    queue_entries = []
    gold_entries = []
    
    # We will use row index as entry_id
    for idx, row in enumerate(tqdm(en_dataset, desc="Building annotation queue")):
        # row structure: 
        # { 'source_text':..., 'masked_text':..., 'privacy_mask':..., 'language':..., ...}
        
        # privacy_mask is already a list of dicts in huggingface dataset
        privacy_mask = row.get("privacy_mask", [])
        if not isinstance(privacy_mask, list):
            privacy_mask = []
            
        entry = {
            "entry_id": idx,
            "original_text": row.get("source_text", ""),
            "masked_text": row.get("masked_text", ""),
            "entities": privacy_mask,
            "language": row.get("language", ""),
            "locale": row.get("locale", ""),
            "routing": {"queue": "FLAT_QUEUE"} # To preserve downstream compatibility
        }
        
        # Auto-Accept Zero-PII sentences
        if len(privacy_mask) == 0:
            entry["anonymized_text"] = entry["original_text"]
            import datetime
            entry["timestamp"] = datetime.datetime.now().isoformat()
            gold_entries.append(entry)
        else:
            queue_entries.append(entry)

    # Make output dirs
    os.makedirs(os.path.dirname(ANNOTATION_QUEUE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(GOLD_STANDARD_PATH), exist_ok=True)
    
    write_jsonl(ANNOTATION_QUEUE_PATH, queue_entries)
    
    if gold_entries:
        append_jsonl(GOLD_STANDARD_PATH, gold_entries)
        
    log.info(f"Successfully wrote {len(queue_entries)} entries to {ANNOTATION_QUEUE_PATH}")
    log.info(f"Auto-accepted {len(gold_entries)} zero-PII entries directly to {GOLD_STANDARD_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load AI4Privacy dataset to annotation queue")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N entries")
    args = parser.parse_args()

    load_annotation_queue(limit=args.limit)
