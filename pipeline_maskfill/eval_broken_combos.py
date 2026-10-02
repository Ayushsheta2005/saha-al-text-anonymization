#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  PII PIPELINE OOM-RESCUE EVALUATION
  Runs evaluation for specific broken combos and correctly manages VRAM
  so BERTScore doesn't trigger a silent Out-Of-Memory (OOM) error.
═══════════════════════════════════════════════════════════════════════════
"""

import os, gc, json, argparse, warnings
import torch
import numpy as np

warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForSeq2SeqLM, 
    AutoModelForMaskedLM
)
import evaluate as hf_evaluate
from tqdm.auto import tqdm

# --- Configuration ---
HF_USERNAME = "Xyren2005"
DATASET_NAME = "ai4privacy/open-pii-masking-500k-ai4privacy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_DIR = "eval_results_fixed"

os.makedirs(EVAL_DIR, exist_ok=True)

# The combinations that crashed BERTScore
BROKEN_COMBOS = [
    ("deberta", "bart-base"),
    ("deberta", "deberta-filler"),
    ("distilroberta", "deberta-filler")
]

ENTITY_TYPES = [
    "FULLNAME", "FIRST_NAME", "LAST_NAME", "ID_NUMBER", "PASSPORT", "SSN",
    "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "LOCATION", "ORGANIZATION",
    "ACCOUNT_NUM", "CREDIT_CARD", "ZIPCODE", "TITLE", "GENDER", "NUMBER",
    "OTHER_PII", "UNKNOWN",
]

ENTITY_MAP = {
    "GIVENNAME": "FIRST_NAME", "SURNAME": "LAST_NAME",
    "TITLE": "TITLE", "GENDER": "GENDER", "SEX": "GENDER",
    "CITY": "LOCATION", "STREET": "ADDRESS", "BUILDINGNUM": "ADDRESS",
    "ZIPCODE": "ZIPCODE", "TELEPHONENUM": "PHONE", "EMAIL": "EMAIL",
    "SOCIALNUM": "SSN", "PASSPORTNUM": "PASSPORT",
    "DRIVERLICENSENUM": "ID_NUMBER", "IDCARDNUM": "ID_NUMBER", "TAXNUM": "ID_NUMBER",
    "CREDITCARDNUMBER": "CREDIT_CARD", "DATE": "DATE", "TIME": "TIME",
    "AGE": "NUMBER",
}

def build_bio_labels():
    labels = ["O"]
    for e in ENTITY_TYPES:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL_BASE = build_bio_labels()

def extract_tokens_and_labels(example):
    text = example.get("source_text", example.get("text", ""))
    masks = example.get("privacy_mask", [])
    if not masks:
        tokens = text.split()
        return tokens, ["O"] * len(tokens)

    spans = sorted(masks, key=lambda m: m.get("start", m.get("offset", 0)))
    tokens, labels = [], []
    pos = 0
    for span in spans:
        start = span.get("start", span.get("offset", 0))
        end = span.get("end", start + span.get("length", len(span.get("value", ""))))
        label = span.get("label", span.get("entity_type", "O")).upper().replace(" ", "_")
        value = span.get("value", text[start:end])
        if pos < start:
            before = text[pos:start].split()
            tokens.extend(before); labels.extend(["O"] * len(before))
        entity_words = value.split()
        if entity_words:
            coarse = ENTITY_MAP.get(label, label)
            if coarse in ENTITY_TYPES:
                tokens.append(entity_words[0]); labels.append(f"B-{coarse}")
                for w in entity_words[1:]:
                    tokens.append(w); labels.append(f"I-{coarse}")
            else:
                tokens.extend(entity_words); labels.extend(["O"] * len(entity_words))
        pos = end
    if pos < len(text):
        remaining = text[pos:].split()
        tokens.extend(remaining); labels.extend(["O"] * len(remaining))
    return tokens, labels

def run_ner(text: str, model, tokenizer, id2label) -> str:
    words = text.split()
    if not words: return text, []
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    word_ids = enc.word_ids()
    pred_labels = []
    prev_wid = None
    for j, wid in enumerate(word_ids):
        if wid is None: continue
        if wid != prev_wid:
            pred_labels.append(id2label.get(preds[j], "O"))
        prev_wid = wid
    pred_labels = pred_labels[:len(words)]
    if len(pred_labels) < len(words):
        pred_labels.extend(["O"] * (len(words) - len(pred_labels)))
    
    masked_words, entities = [], []
    i = 0
    while i < len(words):
        lab = pred_labels[i]
        if lab != "O":
            etype = lab.replace("B-", "").replace("I-", "")
            span_words = [words[i]]
            start_i = i
            i += 1
            while i < len(words) and pred_labels[i] == f"I-{etype}":
                span_words.append(words[i])
                i += 1
            val = " ".join(span_words)
            entities.append((val, etype, start_i, i))
            masked_words.append(f"[{etype}]")
        else:
            masked_words.append(words[i])
            i += 1
    return " ".join(masked_words), entities

def run_filler(masked_text: str, model, tokenizer, is_seq2seq: bool) -> str:
    if is_seq2seq:
        prompt = f"Replace PII placeholders with realistic entities: {masked_text}"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=192, num_beams=4, do_sample=False)
        out_ids = torch.clamp(out_ids, 0, len(tokenizer) - 1)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)
    else: 
        filled_text = masked_text
        for etype in ENTITY_TYPES:
            tag = f"[{etype}]"
            if tag in filled_text:
                filled_text = filled_text.replace(tag, f"{tokenizer.mask_token} {tokenizer.mask_token}")
        inputs = tokenizer(filled_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad(): outputs = model(**inputs)
        token_logits = outputs.logits[0]
        mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        predicted_ids = inputs.input_ids[0].clone()
        for idx in mask_idx: predicted_ids[idx] = token_logits[idx].argmax()
        return tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()

def evaluate_metrics(dataset_subset, originals, anonymized, combo):
    print(f"\n[METRICS] Calculating metrics for {combo}...")
    
    # 1. Leakage
    total_ent, leaked_ent = 0, 0
    for i, anon in enumerate(anonymized):
        anon_lower = anon.lower()
        tokens, labels = extract_tokens_and_labels(dataset_subset[i])
        gold_entities = []
        current_ent = []
        for tok, lab in zip(tokens, labels):
            if lab.startswith("B-"):
                if current_ent: gold_entities.append(" ".join(current_ent))
                current_ent = [tok]
            elif lab.startswith("I-") and current_ent:
                current_ent.append(tok)
            else:
                if current_ent: gold_entities.append(" ".join(current_ent))
                current_ent = []
        if current_ent: gold_entities.append(" ".join(current_ent))
        
        for entity in set(gold_entities):
            if len(entity) > 2:
                total_ent += 1
                if entity.lower() in anon_lower: 
                    leaked_ent += 1
    
    leakage_rate = round(leaked_ent / max(total_ent, 1) * 100, 2)
    
    # 2. ROUGE & BLEU
    rouge = hf_evaluate.load("rouge").compute(predictions=anonymized, references=originals)
    bleu = hf_evaluate.load("sacrebleu").compute(predictions=anonymized, references=[[r] for r in originals])
    
    try:
        wer = hf_evaluate.load("wer").compute(predictions=anonymized, references=originals)
    except:
        wer = 0.0

    # 3. BERTScore (Safe mode)
    try:
        print("    -> Running BERTScore (this normally causes OOM if VRAM isn't empty)")
        # Force BERTScore to run on GPU if available, but since we cleared VRAM it will survive
        bs = hf_evaluate.load("bertscore").compute(
            predictions=anonymized, 
            references=originals, 
            lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        bertscore_f1 = round(float(np.mean([float(x) for x in bs["f1"]])), 4)
    except Exception as e:
        print(f"    -> BERTScore failed even after clearing VRAM: {e}")
        bertscore_f1 = 0.0

    results = {
        "pipeline": combo,
        "entity_leak_rate": leakage_rate,
        "leaked_count": leaked_ent,
        "total_entities": total_ent,
        "rougeL": round(rouge["rougeL"], 4),
        "bleu": round(bleu["score"], 2),
        "bertscore_f1": bertscore_f1,
        "wer": round(wer, 4)
    }
    
    # Save
    res_path = os.path.join(EVAL_DIR, f"results_{combo}.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    -> Results saved to {res_path}")
    print(f"    -> BERTScore: {bertscore_f1} | BLEU: {results['bleu']} | Leak: {leakage_rate}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10, help="Number of samples to run quickly")
    parser.add_argument("--all", action="store_true", help="Run 5000 test set instead of sample")
    
    # Check if running in a Jupyter/Colab notebook
    import sys
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        args = parser.parse_args([]) # Ignore Jupyter's magic arguments
        args.sample = 10             # Change this to 5000 to run full test in notebook
        args.all = False
    else:
        args = parser.parse_args()

    n_samples = 5000 if args.all else args.sample
    print(f"Loading {DATASET_NAME} (evaluating {n_samples} examples)...")
    
    ds = load_dataset(DATASET_NAME, split="train")
    
    # Replicate test split deterministically
    import pandas as pd
    df = ds.to_pandas()
    lang_col = "lang" if "lang" in ds.column_names else "language" if "language" in ds.column_names else "__lang"
    if lang_col not in df.columns: df[lang_col] = "unknown"
    
    test_idx = []
    np.random.seed(42); random.seed(42)
    for lang, grp in df.groupby(lang_col):
        idx = grp.index.tolist()
        random.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * 0.10))
        test_idx.extend(idx[:n_test])
    random.shuffle(test_idx)
    test_dataset = ds.select(test_idx)
    
    eval_subset = test_dataset.select(range(min(n_samples, len(test_dataset))))
    originals = [ex.get("source_text", ex.get("text", "")) for ex in eval_subset]

    for enc_key, fill_key in BROKEN_COMBOS:
        combo = f"{enc_key}+{fill_key}"
        print(f"\n{'='*70}\n[COMBINATION] {combo}\n{'='*70}")
        
        # 1. LOAD MODELS
        enc_repo = f"{HF_USERNAME}/pii-ner-encoder_deberta" if enc_key == "deberta" else f"{HF_USERNAME}/pii-ner-{enc_key}"
        fill_repo = f"{HF_USERNAME}/pii-ner-filler_{fill_key}"
        is_seq2seq = "bart" in fill_key
        
        print(f"Loading Masker: {enc_repo}")
        enc_tok = AutoTokenizer.from_pretrained(enc_repo, use_fast=True)
        enc_model = AutoModelForTokenClassification.from_pretrained(enc_repo).eval().to(DEVICE)
        enc_id2label = {int(k): v for k, v in enc_model.config.id2label.items()}
        
        print(f"Loading Filler: {fill_repo}")
        fill_tok = AutoTokenizer.from_pretrained(fill_repo, use_fast=True)
        if is_seq2seq:
            fill_model = AutoModelForSeq2SeqLM.from_pretrained(fill_repo).eval().to(DEVICE)
        else:
            fill_model = AutoModelForMaskedLM.from_pretrained(fill_repo).eval().to(DEVICE)
        
        # 2. RUN INFERENCE
        anonymized = []
        for text in tqdm(originals, desc="Generating"):
            masked, entities = run_ner(text, enc_model, enc_tok, enc_id2label)
            anon = run_filler(masked, fill_model, fill_tok, is_seq2seq) if entities else text
            anonymized.append(anon)
            
        # 3. CRITICAL OOM FIX: UNLOAD MODELS BEFORE METRICS
        # BERTScore loads 'roberta-large'. If we keep our pipeline models in GPU RAM, it crashes silently!
        print("\nFreeing GPU memory to prepare for BERTScore calculations...")
        del enc_model
        del fill_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 4. COMPUTE METRICS
        evaluate_metrics(eval_subset, originals, anonymized, combo)

if __name__ == "__main__":
    main()
