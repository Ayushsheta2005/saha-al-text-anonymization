"""
Layer 1 — Pre-Annotation Engine
Reads original_data.jsonl, runs spaCy NER + regex patterns, merges detections,
generates Faker replacement candidates, and writes pre_annotated.jsonl.
"""

import json
import os
import sys
import logging
from datetime import datetime

try:
    import spacy
except ImportError:
    spacy = None

from saha_al.config import (
    ORIGINAL_DATA_PATH,
    PRE_ANNOTATED_PATH,
    SPACY_MODEL,
    SPACY_TO_SAHA,
    ENTITY_TYPES,
    TITLES,
    GENDERS,
)
from saha_al.utils.regex_patterns import run_all_patterns
from saha_al.utils.merger import merge_detections
from saha_al.utils.faker_replacements import generate_replacements
from saha_al.utils.io_helpers import read_jsonl, write_jsonl, append_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Layer1] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  spaCy NER
# ─────────────────────────────────────────────────────────────────────
_nlp = None


def _get_nlp():
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is None:
        if spacy is None:
            raise RuntimeError("spaCy is not installed. Run: pip install spacy")
        log.info(f"Loading spaCy model: {SPACY_MODEL}")
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def run_spacy_ner(text: str) -> list:
    """
    Run spaCy NER on the text. Return list of entity dicts mapped to
    our SAHA taxonomy. Entities with no mapping are skipped.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        saha_type = SPACY_TO_SAHA.get(ent.label_)
        if saha_type is None:
            continue  # skip non-PII types
        entities.append({
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "entity_type": saha_type,
            "source": "spacy",
            "spacy_label": ent.label_,
            "priority": 5,
            "confidence": round(0.75 + 0.10 * (1 if ent.label_ in ("PERSON", "GPE", "ORG") else 0), 3),
        })
    return entities


# ─────────────────────────────────────────────────────────────────────
#  Build anonymized text from entity list + replacements
# ─────────────────────────────────────────────────────────────────────
def build_anonymized_text(original_text: str, entities: list, replacement_map: dict) -> str:
    """
    Replace detected PII spans with Faker replacements.
    Process entities from right to left to preserve character offsets.
    """
    # Sort by start descending so replacements don't shift earlier offsets
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0), reverse=True)
    result = original_text
    for ent in sorted_ents:
        start = ent.get("start", 0)
        end = ent.get("end", 0)
        original_pii = ent.get("text", "")
        replacement = replacement_map.get(original_pii)
        if replacement:
            result = result[:start] + replacement + result[end:]
    return result


# ─────────────────────────────────────────────────────────────────────
#  Pre-annotate a single entry
# ─────────────────────────────────────────────────────────────────────
def preannotate_entry(entry: dict) -> dict:
    """
    Full pre-annotation pipeline for one entry:
      1. spaCy NER
      2. Regex patterns
      3. Merge & deduplicate
      4. Generate Faker replacements
      5. Build anonymized_text
      6. Return annotated entry
    """
    entry_id = entry.get("entry_id")
    original_text = entry.get("original_text", "")

    if not original_text.strip():
        return {
            "entry_id": entry_id,
            "original_text": original_text,
            "anonymized_text": original_text,
            "entities": [],
            "replacements": {},
            "metadata": {
                "preannotation_timestamp": datetime.now().isoformat(),
                "spacy_count": 0,
                "regex_count": 0,
                "merged_count": 0,
            },
        }

    # Step 1: spaCy
    spacy_ents = run_spacy_ner(original_text)

    # Step 2: Regex
    regex_ents = run_all_patterns(original_text)

    # Step 3: Merge
    merged = merge_detections(spacy_ents, regex_ents)

    # Step 4: Generate replacement candidates
    replacement_map = {}
    for ent in merged:
        pii_text = ent.get("text", "")
        etype = ent.get("entity_type", "UNKNOWN")
        if pii_text and pii_text not in replacement_map:
            candidates = generate_replacements(etype, pii_text, n=3)
            # Pick the first candidate as default
            replacement_map[pii_text] = candidates[0] if candidates else f"[{etype}]"
            ent["replacement_candidates"] = candidates

    # Step 5: Build anonymized text
    anonymized_text = build_anonymized_text(original_text, merged, replacement_map)

    return {
        "entry_id": entry_id,
        "original_text": original_text,
        "anonymized_text": anonymized_text,
        "entities": merged,
        "replacements": replacement_map,
        "metadata": {
            "preannotation_timestamp": datetime.now().isoformat(),
            "spacy_count": len(spacy_ents),
            "regex_count": len(regex_ents),
            "merged_count": len(merged),
            "agreement_counts": {
                "full": sum(1 for e in merged if e.get("agreement") == "full"),
                "partial": sum(1 for e in merged if e.get("agreement") == "partial"),
                "single_source": sum(1 for e in merged if e.get("agreement") == "single_source"),
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────
#  Main batch runner
# ─────────────────────────────────────────────────────────────────────
def main(
    input_path: str = None,
    output_path: str = None,
    limit: int = None,
    resume: bool = True,
):
    """
    Run Layer 1 pre-annotation on all entries.
    
    Args:
        input_path:  Override for ORIGINAL_DATA_PATH
        output_path: Override for PRE_ANNOTATED_PATH
        limit:       Process only first N entries (for testing)
        resume:      Skip already pre-annotated entry_ids
    """
    input_path = input_path or ORIGINAL_DATA_PATH
    output_path = output_path or PRE_ANNOTATED_PATH

    log.info(f"Reading original data from: {input_path}")
    entries = read_jsonl(input_path)
    log.info(f"Loaded {len(entries)} entries")

    if limit:
        entries = entries[:limit]
        log.info(f"Limited to {limit} entries")

    # Resume: skip already processed
    done_ids = set()
    if resume and os.path.exists(output_path):
        existing = read_jsonl(output_path)
        done_ids = {e.get("entry_id") for e in existing}
        log.info(f"Resuming — {len(done_ids)} entries already processed")

    # Pre-load spaCy
    _get_nlp()

    processed = 0
    errors = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(entries, desc="Layer 1: Pre-annotation")
    except ImportError:
        iterator = entries

    for entry in iterator:
        eid = entry.get("entry_id")
        if eid in done_ids:
            continue
        try:
            result = preannotate_entry(entry)
            append_jsonl(output_path, result)
            processed += 1
        except Exception as exc:
            errors += 1
            log.error(f"Entry {eid}: {exc}")
            continue

    log.info(f"Layer 1 complete. Processed: {processed}, Errors: {errors}")
    log.info(f"Output: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer 1 — Pre-Annotation")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL path")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N entries")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already processed entries")
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
