#!/usr/bin/env python3
"""
Step 2 — BART Black-Box Querier + Consistency Analyser
=======================================================
Loads the trained BART-base victim model and queries it with every
NEW sentence from the adversarial dataset (Step 1 output).
Skips entries already present in bart_query_pairs.jsonl — determined
by cross-referencing BOTH the query state file AND the pairs file itself,
so resumption is robust across file resets.

Produces:
  output/bart_query_pairs.jsonl   — (original, anonymized, metadata) pairs
  output/consistency_report.json  — entity → replacement consistency stats

The consistency report answers the key question BEFORE training the
inverter: does BART map the same entity to the same replacement every
time? If yes, a simple lookup table already breaks it.

Run:
    python3 query_bart.py

Resumes automatically if interrupted.
"""

import os
import sys
import json
import time
import logging
import re
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BART_MODEL_NAME, BART_CHECKPOINT,
    BART_MAX_INPUT_LEN, BART_MAX_OUTPUT_LEN, BART_BATCH_SIZE,
    GENERATED_DATASET_FILE, BART_PAIRS_FILE, BART_QUERY_STATE_FILE,
    CONSISTENCY_REPORT_FILE, OUTPUT_DIR, DATA_DIR, LOGS_DIR,
)

# ── HuggingFace Hub ────────────────────────────────────────────────────────
from huggingface_hub import HfApi, login as hf_login

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_USER      = "JALAPENO11"
HF_REPO_ID   = f"{HF_USER}/model-inversion-adversarial"
HF_REPO_TYPE = "dataset"

# ── logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "query_bart.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_bart_model(device: torch.device) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load BART-base from checkpoint.
    The checkpoint saves the full model state_dict inside best_model.pt.
    """
    logger.info(f"  Loading tokenizer: {BART_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"  Loading model architecture: {BART_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME, torch_dtype=torch.float32)

    if os.path.exists(BART_CHECKPOINT):
        logger.info(f"  Loading checkpoint: {BART_CHECKPOINT}")
        ckpt = torch.load(BART_CHECKPOINT, map_location="cpu", weights_only=False)

        # Handle DataParallel prefix
        state_dict = ckpt.get("model_state_dict", ckpt)
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        logger.info("  ✅ Checkpoint loaded successfully")
    else:
        logger.warning(f"  ⚠️  Checkpoint not found at {BART_CHECKPOINT}")
        logger.warning("  Using pretrained BART-base weights (no fine-tuning)")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Model: {total_params:.1f}M parameters on {device}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════

class AdversarialSentenceDataset(Dataset):
    def __init__(self, entries: List[Dict], tokenizer, max_length: int):
        self.entries   = entries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        sentence = entry["sentence"]

        # BART-base was trained with NO prefix (unlike T5 models)
        enc = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "idx":            idx,
        }


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class QueryStateManager:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {"queried_ids": [], "total_queried": 0, "started_at": datetime.now().isoformat()}

    def sync_from_pairs_file(self, pairs_file: str):
        """
        Scan the existing bart_query_pairs.jsonl and add any IDs found there
        to the queried set. This makes resume robust even if the state file
        was lost or reset independently of the pairs file.
        """
        if not os.path.exists(pairs_file):
            return
        existing_ids = set(self.state["queried_ids"])
        added = 0
        with open(pairs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    eid = rec.get("id", "")
                    if eid and eid not in existing_ids:
                        existing_ids.add(eid)
                        added += 1
                except json.JSONDecodeError:
                    pass
        if added > 0:
            self.state["queried_ids"] = list(existing_ids)
            self.state["total_queried"] = len(existing_ids)
            logger.info(f"  Synced {added:,} additional IDs from pairs file (total queried: {len(existing_ids):,})")
            self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_queried(self, entry_id: str) -> bool:
        # Use cached set for O(1) lookup
        if not hasattr(self, "_queried_set"):
            self._queried_set = set(self.state["queried_ids"])
        return entry_id in self._queried_set

    def mark_queried(self, entry_ids: List[str]):
        self.state["queried_ids"].extend(entry_ids)
        self.state["total_queried"] += len(entry_ids)
        # Keep cached set in sync
        if hasattr(self, "_queried_set"):
            self._queried_set.update(entry_ids)
        self.save()

    @property
    def total_queried(self) -> int:
        return self.state["total_queried"]


# ═══════════════════════════════════════════════════════════════════════════
# CONSISTENCY ANALYSER
# ═══════════════════════════════════════════════════════════════════════════

class ConsistencyAnalyser:
    """
    Builds entity → replacement mapping from all pairs.
    Measures how consistently BART maps the same probe entity
    to the same replacement — the key vulnerability metric.
    """

    def __init__(self):
        # entity_text → Counter of replacements seen
        self.entity_to_replacements: Dict[str, Counter] = defaultdict(Counter)
        # strategy → list of (entity, replacement) for per-strategy analysis
        self.strategy_pairs: Dict[str, List[Tuple]] = defaultdict(list)

    def add_pair(self, original: str, anonymized: str, probe_entity: str,
                 entity_type: str, strategy: str):
        """
        Extract what BART replaced probe_entity with and record it.
        Uses simple string matching — if probe_entity appears in original
        but not in anonymized, it was replaced. We find the replacement
        by aligning the two texts word-by-word.
        """
        if not probe_entity or probe_entity not in original:
            return

        replacement = self._extract_replacement(original, anonymized, probe_entity)
        if replacement:
            self.entity_to_replacements[probe_entity][replacement] += 1
            self.strategy_pairs[strategy].append((probe_entity, replacement))

    def _extract_replacement(self, original: str, anonymized: str, entity: str) -> Optional[str]:
        """
        Try to find what replaced `entity` in the anonymized text.
        Approach: split both texts by non-entity context words and find the gap.
        Falls back to 'REMOVED' if entity is simply deleted.
        """
        # If entity still present in output → not replaced (leakage)
        if entity in anonymized:
            return "[LEAKED]"

        # Find character position of entity in original
        idx = original.find(entity)
        if idx == -1:
            return None

        # Get surrounding context (5 chars before and after entity)
        before_ctx = original[max(0, idx - 30):idx].strip().split()[-2:]
        after_ctx  = original[idx + len(entity):idx + len(entity) + 30].strip().split()[:2]

        # Find the same context in anonymized
        before_str = " ".join(before_ctx).lower()
        after_str  = " ".join(after_ctx).lower()

        anon_lower = anonymized.lower()
        start_pos  = 0
        end_pos    = len(anonymized)

        if before_ctx:
            p = anon_lower.rfind(before_str)
            if p != -1:
                start_pos = p + len(before_str)

        if after_ctx:
            p = anon_lower.find(after_str, start_pos)
            if p != -1:
                end_pos = p

        replacement = anonymized[start_pos:end_pos].strip(" .,;:!?\"'")

        if not replacement:
            return "[DELETED]"
        if len(replacement) > 80:  # too long — extraction failed
            return None
        return replacement

    def compute_report(self) -> Dict:
        """
        Compute consistency metrics for every probe entity.
        Returns a JSON-serialisable report dict.
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_entities_analysed": len(self.entity_to_replacements),
            "summary": {},
            "entity_details": {},
            "lookup_table": {},          # entity → most_common_replacement
            "leakage_entities": [],      # entities where BART leaked PII
            "per_strategy": {},
        }

        consistency_scores = []
        leaked = []

        for entity, counter in self.entity_to_replacements.items():
            total = sum(counter.values())
            most_common, most_common_count = counter.most_common(1)[0]
            consistency = most_common_count / total  # 1.0 = perfectly consistent

            # Check for leakage
            leak_count = counter.get("[LEAKED]", 0)
            leak_rate  = leak_count / total

            report["entity_details"][entity] = {
                "total_occurrences":    total,
                "unique_replacements":  len(counter),
                "most_common":          most_common,
                "most_common_count":    most_common_count,
                "consistency_score":    round(consistency, 4),
                "top_5_replacements":   counter.most_common(5),
                "leakage_rate":         round(leak_rate, 4),
            }

            report["lookup_table"][entity] = most_common
            consistency_scores.append(consistency)

            if leak_rate > 0:
                leaked.append({"entity": entity, "leak_rate": round(leak_rate, 4), "count": leak_count})

        # Summary stats
        if consistency_scores:
            report["summary"] = {
                "mean_consistency":     round(sum(consistency_scores) / len(consistency_scores), 4),
                "perfectly_consistent": sum(1 for s in consistency_scores if s == 1.0),
                "highly_consistent":    sum(1 for s in consistency_scores if s >= 0.9),
                "inconsistent":         sum(1 for s in consistency_scores if s < 0.5),
                "total_entities":       len(consistency_scores),
                "lookup_attack_viable": sum(1 for s in consistency_scores if s >= 0.9),
            }

        report["leakage_entities"] = sorted(leaked, key=lambda x: -x["leak_rate"])

        # Per-strategy
        for strategy, pairs in self.strategy_pairs.items():
            if not pairs:
                continue
            strat_counter: Counter = Counter()
            entity_counter: Dict[str, Counter] = defaultdict(Counter)
            for ent, rep in pairs:
                strat_counter[rep] += 1
                entity_counter[ent][rep] += 1
            scores = [
                max(c.values()) / sum(c.values())
                for c in entity_counter.values()
            ]
            report["per_strategy"][strategy] = {
                "pairs":             len(pairs),
                "unique_entities":   len(entity_counter),
                "mean_consistency":  round(sum(scores) / len(scores), 4) if scores else 0,
                "top_replacements":  strat_counter.most_common(10),
            }

        return report


# ═══════════════════════════════════════════════════════════════════════════
# BATCH INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference_batched(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    entries: List[Dict],
    device: torch.device,
    output_file: str,
    state: QueryStateManager,
    analyser: ConsistencyAnalyser,
) -> int:
    """
    Run all entries through BART in batches.
    Writes pairs to output_file immediately after each batch.
    Returns total pairs written.
    """
    dataset    = AdversarialSentenceDataset(entries, tokenizer, BART_MAX_INPUT_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BART_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_written = 0

    for batch in tqdm(dataloader, desc="  Querying BART", unit="batch", dynamic_ncols=True):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        idxs           = batch["idx"].tolist()

        # Greedy decoding — deterministic, mirrors inference.py
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=BART_MAX_OUTPUT_LEN,
            num_beams=1,
            do_sample=False,
            early_stopping=False,
        )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        batch_entries  = [entries[i] for i in idxs]
        batch_ids      = []

        with open(output_file, "a", encoding="utf-8") as f:
            for entry, anonymized in zip(batch_entries, decoded):
                pair = {
                    # ── Core pair ──────────────────────────────
                    "id":               entry["id"],
                    "original":         entry["sentence"],
                    "anonymized":       anonymized,
                    "probe_entity":     entry["probe_entity"],
                    "entity_type":      entry["entity_type"],

                    # ── Strategy metadata (pass-through) ──────
                    "strategy":             entry["strategy"],
                    "combo_type":           entry.get("combo_type"),
                    "paraphrase_group_id":  entry.get("paraphrase_group_id"),
                    "rarity_tier":          entry.get("rarity_tier"),
                    "correlation_type":     entry.get("correlation_type"),
                    "edge_case_type":       entry.get("edge_case_type"),
                    "context_domain":       entry.get("context_domain"),
                    "name_rarity":          entry.get("name_rarity"),
                    "name_origin_mix":      entry.get("name_origin_mix"),

                    # ── Sentence stats (pass-through) ──────────
                    "word_count":           entry.get("word_count"),
                    "pii_types_present":    entry.get("pii_types_present"),
                    "pii_count":            entry.get("pii_count"),
                    "sentence_template":    entry.get("sentence_template"),
                    "attack_difficulty":    entry.get("attack_difficulty"),

                    # ── Attack fields ──────────────────────────
                    "split":                    entry.get("split", "train"),
                    "is_consistency_probe":     entry.get("is_consistency_probe", False),
                    "is_paraphrase_group":      entry.get("is_paraphrase_group", False),
                    "is_correlation_probe":     entry.get("is_correlation_probe", False),
                    "bart_model":               BART_MODEL_NAME,
                    "bart_decoding":            "greedy",
                    "queried_at":               datetime.now().isoformat(),
                }
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                batch_ids.append(entry["id"])
                total_written += 1

                # Feed to consistency analyser
                analyser.add_pair(
                    original=entry["sentence"],
                    anonymized=anonymized,
                    probe_entity=entry["probe_entity"],
                    entity_type=entry["entity_type"],
                    strategy=entry["strategy"],
                )

        state.mark_queried(batch_ids)

    return total_written


# ═══════════════════════════════════════════════════════════════════════════
# CONSISTENCY REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_consistency_report(report: Dict):
    s = report.get("summary", {})
    print("\n" + "═" * 70)
    print("  CONSISTENCY ANALYSIS — BART-BASE VULNERABILITY REPORT")
    print("═" * 70)
    print(f"  Entities analysed:       {s.get('total_entities', 0):,}")
    print(f"  Mean consistency score:  {s.get('mean_consistency', 0):.3f}  (1.0 = always same replacement)")
    print(f"  Perfectly consistent:    {s.get('perfectly_consistent', 0):,}  entities (score = 1.0)")
    print(f"  Highly consistent:       {s.get('highly_consistent', 0):,}  entities (score ≥ 0.9)")
    print(f"  Inconsistent:            {s.get('inconsistent', 0):,}  entities (score < 0.5)")
    print(f"  Lookup attack viable for:{s.get('lookup_attack_viable', 0):,}  entities")
    print()

    leaked = report.get("leakage_entities", [])
    if leaked:
        print(f"  ⚠️  PII LEAKAGE DETECTED in {len(leaked)} entities:")
        for item in leaked[:10]:
            print(f"    '{item['entity']}' leaked {item['leak_rate']*100:.1f}% of the time")
    else:
        print("  ✅  No direct PII leakage detected")

    print()
    print("  TOP 20 ENTITY → REPLACEMENT MAPPINGS:")
    details = report.get("entity_details", {})
    top20   = sorted(details.items(), key=lambda x: -x[1]["consistency_score"])[:20]
    for entity, info in top20:
        bar = "█" * int(info["consistency_score"] * 10)
        print(f"    {entity:<35} → '{info['most_common']}'  [{bar:<10}] {info['consistency_score']:.2f}  ({info['total_occurrences']} obs)")

    print()
    print("  PER-STRATEGY CONSISTENCY:")
    for strat, info in report.get("per_strategy", {}).items():
        print(f"    {strat:<35}  mean={info['mean_consistency']:.3f}  pairs={info['pairs']}")

    print("═" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# HUGGINGFACE UPLOAD  — called after BART pairs are complete
# ═══════════════════════════════════════════════════════════════════════════

def upload_to_huggingface(pairs_file: str, report_file: str) -> bool:
    """
    Upload the COMPLETE dataset (bart_query_pairs.jsonl + consistency_report.json)
    to HuggingFace Hub.  This is called AFTER query_bart.py finishes so every
    record contains both the original sentence AND the BART-anonymized output.
    """
    print("\n" + "─" * 70)
    print("  UPLOADING COMPLETE DATASET TO HUGGINGFACE")
    print("─" * 70)
    print(f"  Repo      : {HF_REPO_ID}")
    print(f"  Pairs file: {pairs_file}")
    try:
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            private=False,
            exist_ok=True,
        )
        print(f"  Repo ready: https://huggingface.co/datasets/{HF_REPO_ID}")

        # ── Count records & compute split stats ────────────────────────────
        n_total = n_train = n_eval = 0
        with open(pairs_file, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                n_total += 1
                if rec.get("split") == "eval":
                    n_eval += 1
                else:
                    n_train += 1
        size_mb = os.path.getsize(pairs_file) / 1e6

        # ── Dataset card ───────────────────────────────────────────────────
        readme = f"""---
language:
- en
license: mit
task_categories:
- text-generation
- token-classification
tags:
- privacy
- pii
- model-inversion
- adversarial
- synthetic
- anonymization
pretty_name: Model Inversion Adversarial Dataset (with BART anonymization)
size_categories:
- 10K<n<100K
---

# Model Inversion Adversarial Dataset

**{n_total:,} (original, anonymized) sentence pairs** (target: 40,000) for black-box model
inversion attack research against PII anonymization models.

Each record contains the **original PII-rich sentence** and the **BART-anonymized
output** produced by a fine-tuned BART-base anonymizer, along with rich metadata.

## Splits

| Split | Count |
|-------|-------|
| train | {n_train:,} |
| eval  | {n_eval:,} |
| **total** | **{n_total:,}** |

## Probing Strategies

| Strategy | Count | Purpose |
|---|---|---|
| S1 — Entity consistency | 12,000 | 150 probe names × ~80 contexts each |
| S2 — Combinatorial PII | 8,000 | Controlled NAME+PHONE/EMAIL/DATE combos |
| S3 — Paraphrase consistency | 6,000 | Same PII, different sentence structure |
| S4 — Rarity spectrum | 6,000 | common → very_rare name spectrum |
| S5 — Cross-entity correlation | 4,000 | Email/phone/org correlated to name |
| S6 — Edge cases | 4,000 | Dense PII, implicit PII, multi-person |

## Key Fields

| Field | Description |
|-------|-------------|
| `original` | Raw PII-rich sentence (Gemini-generated) |
| `anonymized` | BART-base output (the anonymized version) |
| `probe_entity` | Primary PII entity in the sentence |
| `entity_type` | NAME / PHONE / EMAIL / DATE / ID_DOCUMENT / ADDRESS |
| `strategy` | Which of the 6 probing strategies generated this row |
| `name_rarity` | common / medium / rare / very_rare |
| `attack_difficulty` | 1 (easy) – 5 (hard) heuristic score |
| `split` | train / eval |

## Pipeline

```
generate_dataset.py  →  adversarial_dataset_raw.jsonl  (Step 1: Gemini)
query_bart.py        →  bart_query_pairs.jsonl          (Step 2: BART)   ← this file
train_inverter.py    →  inverter_checkpoint/            (Step 3: train)
evaluate_attack.py   →  attack_results.json             (Step 4: eval)
```

Generated: {datetime.now().strftime("%Y-%m-%d")} | Victim model: facebook/bart-base (fine-tuned)
"""
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message="Update dataset card with BART pair stats",
        )

        # ── Upload main pairs file ─────────────────────────────────────────
        print(f"  Uploading {n_total:,} pairs ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=pairs_file,
            path_in_repo="bart_query_pairs.jsonl",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Upload complete dataset: {n_total:,} (original, anonymized) pairs",
        )

        # ── Upload consistency report ──────────────────────────────────────
        if os.path.exists(report_file):
            print(f"  Uploading consistency report...")
            api.upload_file(
                path_or_fileobj=report_file,
                path_in_repo="consistency_report.json",
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                commit_message="Upload BART consistency analysis report",
            )

        print(f"  ✅  Upload complete!")
        print(f"  🔗  https://huggingface.co/datasets/{HF_REPO_ID}")
        return True

    except Exception as e:
        logger.error(f"  ❌  HuggingFace upload failed: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("  MODEL INVERSION — STEP 2: BART BLACK-BOX QUERIER")
    print("=" * 70)

    # ── Load adversarial dataset ───────────────────────────────────────────
    if not os.path.exists(GENERATED_DATASET_FILE):
        print(f"  ❌  Generated dataset not found: {GENERATED_DATASET_FILE}")
        print("  Run: python3 generate_dataset.py first.")
        return

    logger.info(f"  Loading adversarial dataset: {GENERATED_DATASET_FILE}")
    all_entries = []
    with open(GENERATED_DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))
    logger.info(f"  Loaded {len(all_entries):,} entries")

    # ── Resume filtering ───────────────────────────────────────────────────
    state = QueryStateManager(BART_QUERY_STATE_FILE)
    # Sync from the actual pairs file first — handles the case where the
    # state file is stale or was reset but pairs file has more records.
    state.sync_from_pairs_file(BART_PAIRS_FILE)
    queried_set = set(state.state["queried_ids"])
    remaining   = [e for e in all_entries if e["id"] not in queried_set]

    print(f"  Total entries:     {len(all_entries):,}")
    print(f"  Already queried:   {state.total_queried:,}")
    print(f"  Remaining:         {len(remaining):,}")

    if not remaining:
        print("  ✅  All entries already queried. Regenerating consistency report...")
    else:
        # ── Device ────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device:            {device}")
        if device.type == "cuda":
            print(f"  GPU:               {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:              {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        print()

        input("  Press Enter to start querying BART (Ctrl+C to cancel)...")
        print()

        # ── Load model ─────────────────────────────────────────────────────
        model, tokenizer = load_bart_model(device)

        # ── Run inference ──────────────────────────────────────────────────
        analyser = ConsistencyAnalyser()
        start    = time.time()

        total_written = run_inference_batched(
            model, tokenizer, remaining, device,
            BART_PAIRS_FILE, state, analyser,
        )

        elapsed = time.time() - start
        logger.info(f"  ✅  Queried {total_written:,} entries in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # ── Save consistency report ────────────────────────────────────────
        report = analyser.compute_report()
        with open(CONSISTENCY_REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  Consistency report saved: {CONSISTENCY_REPORT_FILE}")

        print_consistency_report(report)

    # ── Final stats ────────────────────────────────────────────────────────
    if os.path.exists(BART_PAIRS_FILE):
        n_pairs = sum(1 for _ in open(BART_PAIRS_FILE))
        n_train = sum(1 for line in open(BART_PAIRS_FILE)
                      if json.loads(line).get("split") == "train")
        n_eval  = sum(1 for line in open(BART_PAIRS_FILE)
                      if json.loads(line).get("split") == "eval")
        print(f"  Pairs file:  {BART_PAIRS_FILE}")
        print(f"  Total pairs: {n_pairs:,}  (train={n_train:,}, eval={n_eval:,})")

    # ── Upload complete dataset to HuggingFace ────────────────────────────
    # bart_query_pairs.jsonl is the COMPLETE dataset:
    # every record has both the original PII sentence AND the BART output.
    if os.path.exists(BART_PAIRS_FILE):
        upload_to_huggingface(BART_PAIRS_FILE, CONSISTENCY_REPORT_FILE)

    print("\n  Next step: python3 train_inverter.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
