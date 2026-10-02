#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    PII ANONYMIZATION PIPELINE — EVALUATION MATRIX NOTEBOOK                 ║
║                                                                            ║
║    This notebook rigidly loads pre-trained models exclusively from their   ║
║    remote Hugging Face Hub repositories (Xyren2005) and structurally       ║
║    evaluates them via seqeval, semantic matching, and robust leak logic.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 0: INSTALL DEPENDENCIES                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["datasets", "transformers", "accelerate",
            "seqeval", "evaluate", "rouge-score", "sacrebleu", "bert-score"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: CREDENTIALS & HARDCODED MODELS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

HF_USERNAME = "Xyren2005"
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN    = os.getenv("HF_TOKEN")
WANDB_API_KEY = "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"

MODELS_ENCODER = {
    "distilroberta": f"{HF_USERNAME}/pii-ner-distilroberta",
    "roberta": f"{HF_USERNAME}/pii-ner-roberta",
    "deberta": f"{HF_USERNAME}/pii-ner-encoder_deberta",
}

MODELS_FILLER = {
    "bart-base": f"{HF_USERNAME}/pii-ner-filler_bart-base",
    "deberta-filler": f"{HF_USERNAME}/pii-ner-filler_deberta-filler",
}

import os
os.environ["HF_TOKEN"] = HF_TOKEN
from huggingface_hub import login
login(HF_TOKEN, add_to_git_credential=False)

try:
    import wandb
    os.environ['WANDB_PROJECT'] = "pii-anonymization-evaluation"
    wandb.login(key=WANDB_API_KEY)
except:
    pass

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: PIPELINE CONFIGURATION & DATA SPLITING                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, gc, re, json, random, logging
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import torch
from datasets import load_dataset, Dataset

# Hardware & Determinism SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK = (torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8)
FP16_OK = torch.cuda.is_available() and not BF16_OK

EVAL_DIR = "eval_results"
os.makedirs(EVAL_DIR, exist_ok=True)

# All possible entity types used across different legacy checkpoints
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

# Base labels (Ground Truth alignment uses this, but models will use their own configs)
def build_bio_labels(entity_types):
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL_BASE = build_bio_labels(ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)

# Stratified Dataloader
DATASET_NAME = "ai4privacy/open-pii-masking-500k-ai4privacy"
print(f"Loading {DATASET_NAME} for explicit evaluation testing...")
ds = load_dataset(DATASET_NAME, split="train")

def generate_test_split(ds: Dataset) -> Dataset:
    print("Performing deterministic generation of test partition (Seed=42)...")
    df = ds.to_pandas()
    lang_col = None
    for col in ("language", "lang", "Language"):
        if col in ds.column_names:
            lang_col = col
            break
    if lang_col is None:
        lang_col = "__lang"
        df[lang_col] = "unknown"

    test_idx = []
    # Ratio mimics exact 500k pipeline notebook
    TEST_RATIO = 0.10
    for lang, grp in df.groupby(lang_col):
        idx = grp.index.tolist()
        random.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * TEST_RATIO))
        test_idx.extend(idx[:n_test])
    
    random.shuffle(test_idx)
    test_ds = ds.select(test_idx)
    print(f"Test split populated with {len(test_ds):,} strictly unseen evaluation rows.")
    return test_ds

test_dataset = generate_test_split(ds)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: PIPELINE FUNCTIONS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_tokens_and_labels(example: Dict) -> Tuple[List[str], List[str]]:
    if "tokens" in example and "ner_tags" in example:
        return example["tokens"], example["ner_tags"]
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

def get_source_text(example: Dict) -> str:
    if "source_text" in example: return example["source_text"]
    if "text" in example: return example["text"]
    tokens, _ = extract_tokens_and_labels(example)
    return " ".join(tokens)

def run_ner(text: str, model, tokenizer, id2label: dict) -> Tuple[str, List[Tuple[str, str, int, int]]]:
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
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256, padding=True).to(model.device)
        with torch.no_grad():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out_ids = model.generate(**enc, max_new_tokens=192,
                                         num_beams=4, do_sample=False)
        out_ids = torch.clamp(out_ids, 0, len(tokenizer) - 1)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)
    else: # MLM
        filled_text = masked_text
        for etype in ENTITY_TYPES:
            tag = f"[{etype}]"
            if tag in filled_text:
                filled_text = filled_text.replace(tag, f"{tokenizer.mask_token} {tokenizer.mask_token}")
        inputs = tokenizer(filled_text, return_tensors="pt").to(model.device)
        with torch.no_grad(): outputs = model(**inputs)
        token_logits = outputs.logits[0]
        mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        predicted_ids = inputs.input_ids[0].clone()
        for idx in mask_idx: predicted_ids[idx] = token_logits[idx].argmax()
        return tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()

def batch_anonymize(texts, enc_model, enc_tok, fill_model, fill_tok, is_seq2seq, id2label, desc="Anonymizing"):
    from tqdm.auto import tqdm
    anonymized, masked_texts, all_entities = [], [], []
    for text in tqdm(texts, desc=desc[:20]):
        masked, entities = run_ner(text, enc_model, enc_tok, id2label)
        masked_texts.append(masked); all_entities.append(entities)
        anon = run_filler(masked, fill_model, fill_tok, is_seq2seq) if entities else text
        anonymized.append(anon)
    return anonymized, masked_texts, all_entities

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: EVALUATION HELPER FUNCTIONS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from seqeval.metrics import f1_score as seq_f1_score, classification_report
import evaluate as hf_evaluate

def compute_entity_leakage(dataset_subset, anonymized):
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
                    
    return {"entity_leak_rate": round(leaked_ent / max(total_ent, 1) * 100, 2),
            "leaked_count": leaked_ent, "total_entities": total_ent}

def compute_utility_metrics(originals, anonymized):
    results = {}
    rouge = hf_evaluate.load("rouge").compute(predictions=anonymized, references=originals)
    results["rouge"] = {k: round(v, 4) for k, v in rouge.items() if isinstance(v, float)}
    bleu = hf_evaluate.load("sacrebleu").compute(predictions=anonymized, references=[[r] for r in originals])
    results["bleu"] = round(bleu["score"], 2)
    try:
        bs = hf_evaluate.load("bertscore").compute(predictions=anonymized, references=originals, lang="en")
        results["bertscore_f1"] = round(float(np.mean([float(x) for x in bs["f1"]])), 4)
    except: results["bertscore_f1"] = 0.0
    try:
        wer = hf_evaluate.load("wer").compute(predictions=anonymized, references=originals)
        results["wer"] = round(wer, 4)
    except: results["wer"] = 0.0

    orig_lens = [len(x.split()) for x in originals if x.strip()]
    anon_lens = [len(x.split()) for x in anonymized if x.strip()]
    mean_orig = np.mean(orig_lens) if orig_lens else 1.0
    mean_anon = np.mean(anon_lens) if anon_lens else 1.0
    results["length_ratio"] = round(mean_anon / mean_orig, 4)
    return results

def evaluate_pipeline(dataset_subset, originals, anonymized, enc_name, fill_name):
    combo = f"{enc_name}+{fill_name}"
    print(f"\n{'═'*70}\n  EVALUATING: {combo}  ({len(originals)} examples)\n{'═'*70}")
    
    leakage = compute_entity_leakage(dataset_subset, anonymized)
    utility = compute_utility_metrics(originals, anonymized)
    results = {"pipeline": combo, **leakage, **utility}
    
    print(f"  Entity Leak Rate: {leakage['entity_leak_rate']}%  ({leakage['leaked_count']}/{leakage['total_entities']})")
    print(f"  ROUGE-L: {utility.get('rouge', {}).get('rougeL', 0):.4f}")
    print(f"  BLEU:    {utility.get('bleu', 0):.2f}")
    print(f"  BERTScr: {utility.get('bertscore_f1', 0):.4f}")
    print(f"  WER:     {utility.get('wer', 0.0):.4f} (Word Error Rate)")
    print(f"  Len Ratio: {utility.get('length_ratio', 0.0):.2f} (Preservation of original text size)")
    
    res_path = os.path.join(EVAL_DIR, f"results_{combo}.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {res_path}")

def force_evaluate_encoder_standalone(enc_key, repo_id, dataset):
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    print(f"Evaluating Encoder Native Logic for: {enc_key}")
    try:
        tokenizer = hub_retry(AutoTokenizer.from_pretrained, repo_id, use_fast=True)
        model = hub_retry(AutoModelForTokenClassification.from_pretrained, 
            repo_id, ignore_mismatched_sizes=False).eval().to(DEVICE)
    except Exception as e:
        print(f"Failed to load {repo_id}: {e}")
        return
        
    id2label = {int(k): v for k, v in model.config.id2label.items()}
        
    true_labels, pred_labels = [], []
    for i in range(min(len(dataset), 5000)):
        ex = dataset[i]
        tokens, gold_labs = extract_tokens_and_labels(ex)
        if not tokens: continue
        text = " ".join(tokens)
        enc = tokenizer(tokens, return_tensors="pt", truncation=True, max_length=256, is_split_into_words=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        word_ids = enc.word_ids()
        plabs = []
        prev_wid = None
        for j, wid in enumerate(word_ids):
            if wid is None: continue
            if wid != prev_wid: plabs.append(id2label.get(preds[j], "O"))
            prev_wid = wid
        min_len = min(len(gold_labs), len(plabs))
        true_labels.append(gold_labs[:min_len])
        pred_labels.append(plabs[:min_len])
        
    f1 = seq_f1_score(true_labels, pred_labels, average="weighted")
    report = classification_report(true_labels, pred_labels)
    print(f"\n{enc_key.upper()} STANDALONE REPORT (Subset)")
    print(f"Overall Weighted F1: {f1:.4f}")
    print(report)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: EXECUTION MATRIX                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# N_TEST evaluates 5,000 test examples to prevent the 3x2 matrix from exceeding Kaggle's 12-hour limit.
# 5000 is statistically robust enough to get a definitive performance grade!
N_TEST = min(5000, len(test_dataset))

# Retry wrapper to defeat random Kaggle/HF API HTTP timeouts
import time
def hub_retry(func, *args, retries=5, delay=5, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1: raise e
            print(f"    [Network Drop Detected] Retrying download in {delay}s... ({e})")
            time.sleep(delay)
eval_subset = test_dataset.select(range(N_TEST))
originals = [get_source_text(eval_subset[i]) for i in range(N_TEST)]

print("="*60)
print(f"Executing Matrix Combinations directly from HF ({N_TEST} examples)")
print("="*60)

for enc_key, enc_repo in MODELS_ENCODER.items():
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    print(f"\n[{enc_key.upper()}] Loading Encoder...")
    enc_tok = hub_retry(AutoTokenizer.from_pretrained, enc_repo, use_fast=True)
    enc_model = hub_retry(AutoModelForTokenClassification.from_pretrained, 
        enc_repo, ignore_mismatched_sizes=False).eval().to(DEVICE)
        
    enc_id2label = {int(k): v for k, v in enc_model.config.id2label.items()}

    # Trigger standalone metrics for this encoder
    force_evaluate_encoder_standalone(enc_key, enc_repo, eval_subset)

    for fill_key, fill_repo in MODELS_FILLER.items():
        from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM
        print(f"\n   [{fill_key.upper()}] Loading Filler...")
        is_seq2seq = "bart" in fill_key
        fill_tok = hub_retry(AutoTokenizer.from_pretrained, fill_repo, use_fast=True)
        if is_seq2seq:
            fill_model = hub_retry(AutoModelForSeq2SeqLM.from_pretrained, fill_repo).eval().to(DEVICE)
        else:
            fill_model = hub_retry(AutoModelForMaskedLM.from_pretrained, fill_repo).eval().to(DEVICE)
            
        anonymized, _, _ = batch_anonymize(originals, enc_model, enc_tok, fill_model, fill_tok, is_seq2seq, id2label=enc_id2label, desc=f"{enc_key}+{fill_key}")
        evaluate_pipeline(eval_subset, originals, anonymized, enc_key, fill_key)

        del fill_model
        del fill_tok
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    del enc_model
    del enc_tok
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("\n🎉 Full Standalone Hub Evaluation Matrix Finished!")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BONUS CELL: Download Results as ZIP (Kaggle Specific)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Run this cell to automatically zip all JSON matrices locally so you can click to download!

# import shutil
# from IPython.display import FileLink
# shutil.make_archive("evaluation_metrics", "zip", "eval_results")
# print("Finished Zipping! Click the link below to download:")
# display(FileLink("evaluation_metrics.zip"))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BONUS CELL: Push Results to a dedicated Hugging Face Dataset              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Run this cell to officially log all metrics permanently inside a Dataset repo.

# from huggingface_hub import HfApi
# api = HfApi()
# DATASET_REPO = f"{HF_USERNAME}/pii-evaluation-metrics"
# print(f"Creating/Locating dataset repository: {DATASET_REPO}")
# api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
# print("Blasting all JSON metrics to the hub...")
# api.upload_folder(
#     folder_path="eval_results",
#     repo_id=DATASET_REPO,
#     repo_type="dataset",
#     commit_message="Add fresh evaluation matrix from Kaggle Notebook"
# )
# print("✅ Successfully backed up your matrix!")
