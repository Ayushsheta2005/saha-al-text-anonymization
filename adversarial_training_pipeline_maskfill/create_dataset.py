#!/usr/bin/env python3
"""
create_dataset.py
=================
Builds the adversarial training and evaluation datasets for pipeline
mask-fill adversarial hardening.

Sources
-------
  data_creation/data/english_entries_all.jsonl
      Fields: id, original_text, masked_text  (AI4Privacy format [LABEL_N]),
              privacy_mask (list of entity spans)
  data_creation/output/anonymized_dataset_final.jsonl
      Fields: entry_id, original_text, anonymized_text

Join key: english_entries_all.id == anonymized_dataset_final.entry_id

Strategy
--------
  • Skip the first 40,000 rows (already consumed by the model-inversion
    dataset generation via generate_pairs.py in model_inversion_on_pipeline_maskfill/).
  • From the remaining rows, collect the first 45,000 records that have
    at least one PII entity (masked_text ≠ original_text).
  • Shuffle (seed=42) → split 40,000 train / 5,000 eval.

Output fields per row
---------------------
  id           : source record ID (int)
  original     : original PII-bearing text
  masked_mlm   : masked text for DeBERTa-MLM filler
                 [LABEL_N] → [MASK]  (one [MASK] per placeholder)
  masked_s2s   : masked text for BART seq2seq filler
                 [LABEL_N] → [COARSE_TYPE] e.g. [PERSON], [DATE]
                 prepended with the standard filler prompt
  anonymized   : gold anonymized text (from human-annotation pipeline)
  entity_texts : list of original PII strings from privacy_mask

Output files
------------
  adversarial_training_pipeline_maskfill/output/adv_train.jsonl   (40k rows)
  adversarial_training_pipeline_maskfill/output/adv_eval.jsonl    (5k rows)

Usage
-----
  cd adversarial_training_pipeline_maskfill
  python create_dataset.py [--total 45000] [--train 40000] [--skip 40000]
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

SOURCE_FILE    = PROJECT_DIR / "data_creation" / "data" / "english_entries_all.jsonl"
ANONYMIZED_FILE = PROJECT_DIR / "data_creation" / "output" / "anonymized_dataset_final.jsonl"
OUTPUT_DIR     = BASE_DIR / "output"

TRAIN_FILE = OUTPUT_DIR / "adv_train.jsonl"
EVAL_FILE  = OUTPUT_DIR / "adv_eval.jsonl"

# ── AI4Privacy label  →  pipeline entity type mapping ─────────────────────
# Pipeline entity types (from model_inversion_on_pipeline_maskfill/config.py):
#   PERSON, LOC, ORG, DATE, PHONE, EMAIL, SSN, CREDIT_CARD, ADDRESS,
#   IP_ADDRESS, IBAN, PASSPORT, DRIVER_LICENSE, USERNAME, URL, MEDICAL,
#   ACCOUNT, BUILDING, POSTCODE
AI4PRIV_TO_PIPELINE: dict[str, str] = {
    # Person / demographics
    "GIVENNAME":           "PERSON",
    "SURNAME":             "PERSON",
    "MIDDLENAME":          "PERSON",
    "TITLE":               "PERSON",
    "GENDER":              "PERSON",
    "SEX":                 "PERSON",
    "AGE":                 "PERSON",
    # Time
    "DATE":                "DATE",
    "TIME":                "DATE",
    # Contact
    "TELEPHONENUM":        "PHONE",
    "PHONE":               "PHONE",
    "EMAIL":               "EMAIL",
    "URL":                 "URL",
    # Location
    "CITY":                "LOC",
    "STATE":               "LOC",
    "COUNTRY":             "LOC",
    "COUNTY":              "LOC",
    "VILLAGE":             "LOC",
    "STREET":              "ADDRESS",
    "BUILDINGNUM":         "BUILDING",
    "ZIPCODE":             "POSTCODE",
    "POSTCODE":            "POSTCODE",
    # Identity documents
    "IDCARDNUM":           "SSN",
    "SOCIALNUM":           "SSN",
    "TAXNUM":              "SSN",
    "DOCNUMPLACEHOLDER":   "SSN",
    "PASSPORTNUM":         "PASSPORT",
    "DRIVERLICENSENUM":    "DRIVER_LICENSE",
    "CREDITCARDNUMBER":    "CREDIT_CARD",
    # Digital / accounts
    "USERNAME":            "USERNAME",
    "ACCOUNTNAME":         "USERNAME",
    "IPADDRESS":           "IP_ADDRESS",
    "IP":                  "IP_ADDRESS",
    "IBAN":                "ACCOUNT",
    "ACCOUNTNUMPLACEHOLDER": "ACCOUNT",
    "POLICYNUMPLACEHOLDER":  "ACCOUNT",
    # Organisation / misc structured
    "ORGANISATIONPLACEHOLDER": "ORG",
    "LANGUAGEPLACEHOLDER":     "ORG",
    "JOBTITLEPLACEHOLDER":     "ORG",
    # Medical / generic
    "AMOUNTPLACEHOLDER":       "MEDICAL",
    "CURRENCYPLACEHOLDER":     "MEDICAL",
    "HEIGHTPLACEHOLDER":       "MEDICAL",
    "WEIGHTPLACEHOLDER":       "MEDICAL",
    "MEDICAL":                 "MEDICAL",
}

# Pattern:  [LABELNAME_INDEX]  e.g. [GIVENNAME_1], [ORGANISATIONPLACEHOLDER_14]
PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z0-9]*)_(\d+)\]")

# The exact prompt the BART filler was trained with
BART_PROMPT = "Replace PII placeholders with realistic fake entities: "


# ── Conversion helpers ─────────────────────────────────────────────────────

def to_masked_mlm(masked_text: str) -> str:
    """Replace every [LABEL_N] with [MASK] (DeBERTa-MLM filler format)."""
    return PLACEHOLDER_RE.sub("[MASK]", masked_text)


def to_masked_s2s(masked_text: str) -> str:
    """
    Replace every [LABEL_N] with the coarse pipeline entity type tag,
    then prepend the BART filler training prompt.

    e.g.  "Meet [GIVENNAME_1] on [DATE_1]"
      →   "Replace PII placeholders with realistic fake entities: Meet [PERSON] on [DATE]"
    """
    def _replace(m: re.Match) -> str:
        label  = m.group(1)
        coarse = AI4PRIV_TO_PIPELINE.get(label, "ENTITY")
        return f"[{coarse}]"

    replaced = PLACEHOLDER_RE.sub(_replace, masked_text)
    return f"{BART_PROMPT}{replaced}"


def has_pii(masked_text: str) -> bool:
    """Return True iff the masked_text contains at least one placeholder."""
    return bool(PLACEHOLDER_RE.search(masked_text))


def extract_entity_texts(privacy_mask: list) -> list[str]:
    """Return the list of original entity string values from privacy_mask."""
    if not privacy_mask:
        return []
    return [span["value"] for span in privacy_mask if span.get("value")]


# ── Data loading ───────────────────────────────────────────────────────────

def load_anonymized_lookup(path: Path) -> dict[int, str]:
    """Build {entry_id → anonymized_text} lookup from anonymized_dataset_final.jsonl."""
    print(f"Loading anonymized lookup from {path} …")
    lookup: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = row.get("entry_id")
            txt = row.get("anonymized_text", "").strip()
            if eid is not None and txt:
                lookup[int(eid)] = txt
    print(f"  Loaded {len(lookup):,} anonymized entries")
    return lookup


def iter_source_records(
    path: Path,
    skip: int,
    max_pii: int,
    anon_lookup: dict[int, str],
) -> list[dict]:
    """
    Stream english_entries_all.jsonl.
    - Skips the first `skip` rows (already used by inversion dataset).
    - Collects up to `max_pii` records that have ≥1 PII entity AND have a
      matching anonymized text.
    - Returns list of raw dicts ready for processing.
    """
    print(f"Streaming {path} (skipping first {skip:,} rows) …")
    collected: list[dict] = []
    skipped   = 0
    total_seen = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_seen += 1

            if skipped < skip:
                skipped += 1
                continue

            if len(collected) >= max_pii:
                break

            row = json.loads(line)
            rec_id      = int(row.get("id", -1))
            original    = (row.get("original_text") or row.get("text") or "").strip()
            masked_text = (row.get("masked_text") or "").strip()

            # Skip records without PII placeholders
            if not original or not masked_text or not has_pii(masked_text):
                continue

            # Skip records without a gold anonymized counterpart
            if rec_id not in anon_lookup:
                continue

            collected.append({
                "id":          rec_id,
                "original":    original,
                "masked_text": masked_text,
                "anonymized":  anon_lookup[rec_id],
                "privacy_mask": row.get("privacy_mask", []),
            })

    print(f"  Scanned {total_seen:,} rows total | collected {len(collected):,} PII records")
    return collected


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build pipeline adversarial training dataset")
    parser.add_argument("--total",     type=int, default=45_000, help="Total PII records to collect")
    parser.add_argument("--train",     type=int, default=40_000, help="Number of training rows")
    parser.add_argument("--skip",      type=int, default=40_000,
                        help="Rows to skip from english_entries_all.jsonl (default: 40k, "
                             "already consumed by the model-inversion dataset)")
    parser.add_argument("--seed",      type=int, default=42,     help="Random seed for shuffle/split")
    args = parser.parse_args()

    assert args.train < args.total, "--train must be less than --total"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load anonymized gold lookup ─────────────────────────────────────
    anon_lookup = load_anonymized_lookup(ANONYMIZED_FILE)

    # ── 2. Stream + filter source records ─────────────────────────────────
    records = iter_source_records(SOURCE_FILE, args.skip, args.total, anon_lookup)

    if len(records) < args.total:
        print(f"WARNING: only found {len(records):,}/{args.total:,} PII records. "
              "Consider decreasing --skip or --total.")

    # ── 3. Shuffle ──────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[: args.total]          # trim to exactly total (may already be ≤total)

    # ── 4. Build final rows ────────────────────────────────────────────────
    print("Building output rows …")
    rows: list[dict] = []
    skipped_build = 0
    for r in records:
        m_text = r["masked_text"]
        anon   = r["anonymized"]

        # Basic quality filters
        if len(r["original"].split()) < 3:
            skipped_build += 1; continue
        if anon.strip() == r["original"].strip():
            skipped_build += 1; continue

        masked_mlm = to_masked_mlm(m_text)
        masked_s2s = to_masked_s2s(m_text)

        # After conversion masked_mlm should still differ from original
        if masked_mlm.strip() == r["original"].strip():
            skipped_build += 1; continue

        rows.append({
            "id":           r["id"],
            "original":     r["original"],
            "masked_mlm":   masked_mlm,
            "masked_s2s":   masked_s2s,
            "anonymized":   anon,
            "entity_texts": extract_entity_texts(r["privacy_mask"]),
        })

    if skipped_build:
        print(f"  Skipped {skipped_build} rows during quality check")
    print(f"  Final rows after quality filter: {len(rows):,}")

    if len(rows) < args.train:
        print(f"ERROR: fewer rows ({len(rows):,}) than requested train size ({args.train:,}).")
        sys.exit(1)

    # ── 5. Split ────────────────────────────────────────────────────────────
    train_rows = rows[: args.train]
    eval_rows  = rows[args.train :]

    # ── 6. Write output ─────────────────────────────────────────────────────
    def write_jsonl(path: Path, data: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(data):,} rows → {path}")

    print("Writing output files …")
    write_jsonl(TRAIN_FILE, train_rows)
    write_jsonl(EVAL_FILE,  eval_rows)

    # ── 7. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dataset creation complete")
    print(f"  Train : {len(train_rows):,} rows  →  {TRAIN_FILE.name}")
    print(f"  Eval  : {len(eval_rows):,} rows  →  {EVAL_FILE.name}")

    # Quick sanity check on a sample row
    sample = train_rows[0]
    print("\nSample row:")
    print(f"  id        : {sample['id']}")
    print(f"  original  : {sample['original'][:80]}")
    print(f"  masked_mlm: {sample['masked_mlm'][:80]}")
    print(f"  masked_s2s: {sample['masked_s2s'][:80]}")
    print(f"  anonymized: {sample['anonymized'][:80]}")
    print(f"  entities  : {sample['entity_texts'][:5]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
