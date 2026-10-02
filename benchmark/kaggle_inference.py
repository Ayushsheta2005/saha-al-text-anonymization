"""
SAHA-AL Benchmark: Kaggle Inference Script
============================================
Generates predictions.jsonl from trained seq2seq checkpoints.

Handles two checkpoint formats:
  A) Raw .pt files (as in JALAPENO11/pii_identification_and_anonymisations)
  B) Standard HuggingFace model directories (config.json + pytorch_model.bin)

Usage on Kaggle:
  1. Upload this script + test.jsonl to your Kaggle notebook
  2. Set CHECKPOINT_PATH to your model checkpoint
  3. Run all cells
  4. Download the output predictions.jsonl
"""

import json
import os
import re
import torch
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these for your setup
# ══════════════════════════════════════════════════════════════════

# --- Dataset ---
DATASET_SOURCE = "huggingface"  # "huggingface" or "local"
HF_DATASET_ID = "huggingbahl21/saha-al"
LOCAL_TEST_PATH = "/kaggle/input/saha-al-test/test.jsonl"  # if DATASET_SOURCE="local"

# --- Model Checkpoints ---
# List of all available checkpoints with their configs
# Using HF repo for checkpoints: JALAPENO11/pii_identification_and_anonymisations
CHECKPOINT_CONFIGS = [
    {
        "name": "t5-efficient-tiny-pii",
        "base_model": "google/t5-efficient-tiny",
        "checkpoint_filename": "checkpoints_pii_aware_loss/t5-efficient-tiny/best_model.pt",
        "prefix": "anonymize: "
    },
    {
        "name": "t5-small-pii",
        "base_model": "t5-small",
        "checkpoint_filename": "checkpoints_pii_aware_loss/t5-small/best_model.pt",
        "prefix": "anonymize: "
    },
    {
        "name": "flan-t5-small-pii",
        "base_model": "google/flan-t5-small",
        "checkpoint_filename": "checkpoints_pii_aware_loss/flan-t5-small/best_model.pt",
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: "
    },
    {
        "name": "bart-base-pii",
        "base_model": "facebook/bart-base",
        "checkpoint_filename": "checkpoints_pii_aware_loss/bart-base/best_model.pt",
        "prefix": ""
    },
    {
        "name": "distilbart-pii",
        "base_model": "sshleifer/distilbart-cnn-6-6",
        "checkpoint_filename": "checkpoints_pii_aware_loss/distilbart/best_model.pt",
        "prefix": ""
    },
]

# --- Inference ---
BATCH_SIZE = 16
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128
NUM_BEAMS = 4
SUMMARY_FILE = "inference_summary.json"


def normalize_entity_type(ent_type):
    if not isinstance(ent_type, str) or not ent_type.strip():
        return "UNKNOWN"
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", ent_type.strip().upper())
    return cleaned if cleaned else "UNKNOWN"


def find_non_overlapping_span(text, token, occupied_spans):
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


def format_input(record):
    text = record.get("original_text", "")
    entities = sorted(record.get("entities", []), key=lambda x: x.get("start", 0))
    inserted_spans = []
    formatted = text
    safe_entities = []

    for ent in entities:
        ent_text = ent.get("text", "")
        ent_type = normalize_entity_type(ent.get("type", "UNKNOWN"))
        if not ent_text:
            continue

        start = ent.get("start")
        end = ent.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end > len(text)
            or text[start:end] != ent_text
            or any(not (end <= o_start or start >= o_end) for o_start, o_end in inserted_spans)
        ):
            span = find_non_overlapping_span(text, ent_text, inserted_spans)
            if span is None:
                print(f"[WARN] Could not align entity {ent_text!r} for record {record.get('id')}")
                continue
            start, end = span

        if any(not (end <= o_start or start >= o_end) for o_start, o_end in inserted_spans):
            continue

        inserted_spans.append((start, end))
        safe_entities.append({"start": start, "end": end, "type": ent_type, "text": ent_text})

    for ent in sorted(safe_entities, key=lambda x: x["start"], reverse=True):
        formatted = (
            formatted[: ent["start"]]
            + f"<{ent['type']}> "
            + ent["text"]
            + f" </{ent['type']}>"
            + formatted[ent["end"] :]
        )

    return formatted


# ══════════════════════════════════════════════════════════════════
# LOAD TEST DATA
# ══════════════════════════════════════════════════════════════════

print("=" * 50)
print("SAHA-AL Benchmark Inference")
print("=" * 50)

if DATASET_SOURCE == "huggingface":
    from datasets import load_dataset
    print(f"Loading test split from HuggingFace: {HF_DATASET_ID}")
    dataset = load_dataset(HF_DATASET_ID)
    test_records = list(dataset["test"])
    print(f"Loaded {len(test_records)} test records from HuggingFace.")
else:
    print(f"Loading test split from local: {LOCAL_TEST_PATH}")
    with open(LOCAL_TEST_PATH, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]
    print(f"Loaded {len(test_records)} test records.")

# ══════════════════════════════════════════════════════════════════
# RUN INFERENCE FOR ALL CHECKPOINTS
# ══════════════════════════════════════════════════════════════════

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

inference_summary = []
for config in CHECKPOINT_CONFIGS:
    print(f"\n{'=' * 30} Processing {config['name']} {'=' * 30}")
    
    BASE_MODEL = config["base_model"]
    MODEL_PREFIX = config["prefix"]
    OUTPUT_FILE = f"predictions_{config['name']}.jsonl"
    
    checkpoint_path = hf_hub_download(repo_id="JALAPENO11/pii_identification_and_anonymisations", filename=config["checkpoint_filename"])
    
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Checkpoint loaded (strict=True).")
        except RuntimeError as e:
            print(f"Strict load failed: {e}")
            print("Retrying with strict=False...")
            result = model.load_state_dict(state_dict, strict=False)
            print(f"  Missing keys: {len(result.missing_keys)}")
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")
    else:
        print("No checkpoint found, using base model.")
    
    model = model.to(device)
    model.eval()
    print(f"Model ready. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # --- Run Inference ---
    print(f"Running inference on {len(test_records)} records...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Prefix: '{MODEL_PREFIX}' {'(none)' if not MODEL_PREFIX else ''}")
    print(f"  Beams: {NUM_BEAMS}")
    
    results = []
    num_failed = 0
    
    for batch_start in tqdm(range(0, len(test_records), BATCH_SIZE), desc="Inference"):
        batch_records = test_records[batch_start:batch_start + BATCH_SIZE]
        
        input_texts = [
            MODEL_PREFIX + format_input(rec)
            for rec in batch_records
        ]
        
        try:
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_OUTPUT_LENGTH,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    early_stopping=True,
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
            
            for rec, anon_text in zip(batch_records, decoded):
                results.append({
                    "id": rec["id"],
                    "anonymized_text": anon_text
                })
        
        except Exception as e:
            num_failed += len(batch_records)
            if num_failed <= BATCH_SIZE * 3:
                print(f"  [WARN] Batch failed: {e}")
            for rec in batch_records:
                results.append({
                    "id": rec["id"],
                    "anonymized_text": rec["original_text"]
                })
    
    # --- Save Predictions ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    inference_summary.append({
        "model_name": config["name"],
        "base_model": BASE_MODEL,
        "checkpoint_filename": config["checkpoint_filename"],
        "output_file": OUTPUT_FILE,
        "num_records": len(results),
        "num_failed": num_failed,
        "failed_rate": round(num_failed / len(results) * 100, 2) if results else 0.0,
    })

    print(f"Done! {len(results)} predictions saved to {OUTPUT_FILE}")
    if num_failed > 0:
        print(f"  ({num_failed} records failed, fell back to original text)")

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(inference_summary, f, indent=2)

print(f"\n{'=' * 50}")
print("All models processed!")
print(f"Inference summary saved to: {SUMMARY_FILE}")
print("Download the predictions_*.jsonl files and evaluate locally with:")
print("  python benchmark_eval.py --gold data/test.jsonl --pred predictions_*.jsonl")
print(f"{'=' * 50}")