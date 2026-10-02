#!/usr/bin/env python3
"""
Generates two Kaggle training notebooks:
  - combo1_inverter_roberta_deberta.ipynb
  - combo2_inverter_roberta_bart.ipynb

Each notebook includes:
  - Training with checkpointing on 2×T4 GPUs
  - Comprehensive evaluation (BLEU 1/2/4, ROUGE 1/2/L, ERR exact/partial,
    token accuracy, exact sentence match, perplexity)
  - Reconstruction samples table (20 examples)
  - Detailed formatted text report
  - Model upload to HuggingFace
"""
import json, os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── helpers ──────────────────────────────────────────────────────────────────
def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

def md(src):
    return {"cell_type": "markdown", "id": None, "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": None,
            "metadata": {}, "outputs": [], "source": src}

# ── cells shared by both notebooks ───────────────────────────────────────────

INSTALL = """\
%%capture
!pip install -q --upgrade transformers datasets accelerate evaluate sacrebleu rouge_score huggingface_hub
"""

IMPORTS = """\
import os, json, numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

print("torch       :", torch.__version__)
print("CUDA        :", torch.cuda.is_available())
print("GPU count   :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory/1024**3:.1f} GiB")
"""

HF_LOGIN = """\
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)
print("Logged in to HuggingFace ✓")
"""

LOAD_DATA = """\
dataset = load_dataset(DATASET_ID)
print(dataset)
print("\\nSample train row:")
print(json.dumps(dataset["train"][0], indent=2, ensure_ascii=False))
"""

TOKENIZE = """\
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def preprocess(batch):
    # INPUT  = anonymized text  (what remains after PII removal)
    # TARGET = original text    (what the inverter tries to recover)
    inputs = tokenizer(
        batch["anonymized"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
    )
    targets = tokenizer(
        text_target=batch["original"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
)
print(tokenized)
"""

MODEL_INIT = """\
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {BASE_MODEL}  ({total_params/1e6:.1f} M parameters)")

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)
"""

METRICS = """\
import math, re
from collections import Counter

bleu_metric  = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

# ── lightweight helpers (no extra deps) ──────────────────────────────────────

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _sentence_bleu(hyp, ref, max_n=4):
    h, r = hyp.lower().split(), ref.lower().split()
    if not h:
        return 0.0
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r)/len(h))
    scores = []
    for n in range(1, max_n+1):
        hn, rn = _ngrams(h,n), _ngrams(r,n)
        if not hn:
            scores.append(0.0); continue
        m = sum(min(hn[k], rn[k]) for k in hn)
        scores.append(m/sum(hn.values()))
    if any(s == 0 for s in scores):
        return 0.0
    return bp * math.exp(sum(math.log(s) for s in scores)/len(scores))

def _token_accuracy(preds, refs, tok):
    correct = total = 0
    for p, r in zip(preds, refs):
        pt, rt = tok.tokenize(p), tok.tokenize(r)
        ml = min(len(pt), len(rt))
        if ml == 0: continue
        correct += sum(a==b for a,b in zip(pt[:ml], rt[:ml]))
        total   += max(len(pt), len(rt))
    return round(correct/total, 4) if total else 0.0

def _word_accuracy(preds, refs):
    correct = total = 0
    for p, r in zip(preds, refs):
        pw, rw = p.lower().split(), r.lower().split()
        ml = min(len(pw), len(rw))
        if ml == 0: continue
        correct += sum(a==b for a,b in zip(pw[:ml],rw[:ml]))
        total   += max(len(pw), len(rw))
    return round(correct/total, 4) if total else 0.0

def _err(preds, refs):
    \"\"\"Entity Recovery Rate — treat every token as a potential entity.\"\"\"
    exact = partial = total = 0
    for pred, ref in zip(preds, refs):
        tokens = [t for t in ref.lower().split() if len(t) > 2]
        total += len(tokens)
        for t in tokens:
            if t in pred.lower():
                exact += 1
    return (
        round(exact/total, 4) if total else 0.0,
        total,
    )

# ── used by Trainer during training ──────────────────────────────────────────
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace out-of-range / padding token IDs before decoding.
    # predict_with_generate pads short sequences with -100 (or very large ints
    # on some Transformers versions) which overflows the fast Rust tokenizer.
    preds  = np.where((preds  < 0) | (preds  >= tokenizer.vocab_size), tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    bleu  = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
    )["score"]
    rouge = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    return {
        "bleu"  : round(bleu, 4),
        "rougeL": round(rouge["rougeL"], 4),
        "rouge1": round(rouge["rouge1"], 4),
    }
"""

TRAINING_ARGS = """\
os.makedirs(CKPT_DIR, exist_ok=True)

# Estimate warmup steps from ratio (30k train rows, eff. batch=128)
_steps_per_epoch = math.ceil(len(tokenized["train"]) / (PER_DEVICE_BATCH * 2 * GRAD_ACCUM))
_total_steps     = _steps_per_epoch * NUM_EPOCHS
_warmup_steps    = max(1, int(_total_steps * WARMUP_RATIO))
print(f"Steps/epoch: {_steps_per_epoch}  Total: {_total_steps}  Warmup: {_warmup_steps}")

training_args = Seq2SeqTrainingArguments(
    output_dir                  = CKPT_DIR,
    num_train_epochs            = NUM_EPOCHS,

    # ── batching ─────────────────────────────────────────────────────────────
    per_device_train_batch_size = PER_DEVICE_BATCH,   # 32 × 2 GPUs = 64 / step
    per_device_eval_batch_size  = EVAL_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,          # eff. batch = 128

    # ── optimiser ────────────────────────────────────────────────────────────
    learning_rate               = LR,
    warmup_steps                = _warmup_steps,
    weight_decay                = WEIGHT_DECAY,
    label_smoothing_factor      = LABEL_SMOOTH,

    # ── mixed precision + multi-GPU ──────────────────────────────────────────
    fp16                        = True,
    ddp_find_unused_parameters  = False,

    # ── generation during eval ───────────────────────────────────────────────
    predict_with_generate       = True,
    generation_max_length       = MAX_TARGET_LEN,
    generation_num_beams        = NUM_BEAMS_EVAL,

    # ── checkpointing ────────────────────────────────────────────────────────
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "bleu",
    greater_is_better           = True,
    save_total_limit            = SAVE_TOTAL,

    # ── logging ──────────────────────────────────────────────────────────────
    logging_steps               = 100,
    report_to                   = "none",
    dataloader_num_workers      = 4,
)

trainer = Seq2SeqTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = tokenized["train"],
    eval_dataset     = tokenized["val"],
    processing_class = tokenizer,
    data_collator    = data_collator,
    compute_metrics  = compute_metrics,
    callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
)
"""

TRAIN = """\
# Resume automatically from the latest checkpoint (safe on Kaggle restarts)
last_ckpt = None
if os.path.isdir(CKPT_DIR):
    ckpts = sorted(
        [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    )
    if ckpts:
        last_ckpt = os.path.join(CKPT_DIR, ckpts[-1])
        print(f"Resuming from: {last_ckpt}")

trainer.train(resume_from_checkpoint=last_ckpt)
"""

TEST_EVAL = """\
# ── Full evaluation on held-out test set ─────────────────────────────────────
print("\\n=== Full evaluation on test set ===")
raw_test_results = trainer.evaluate(eval_dataset=tokenized["test"])
eval_loss   = raw_test_results.get("eval_loss", float("nan"))
perplexity  = math.exp(min(eval_loss, 20))
print(f"  eval_loss   : {eval_loss:.4f}")
print(f"  perplexity  : {perplexity:.2f}")
print(f"  BLEU        : {raw_test_results.get('eval_bleu', 0):.4f}")
print(f"  ROUGE-1     : {raw_test_results.get('eval_rouge1', 0):.4f}")
print(f"  ROUGE-L     : {raw_test_results.get('eval_rougeL', 0):.4f}")

# ── Generate all predictions on test set for extended metrics ────────────────
print("\\nGenerating test-set predictions for extended metrics…")
model.eval()
all_preds, all_refs, all_anons = [], [], []

test_rows = dataset["test"]
gen_pipeline = __import__("transformers").pipeline(
    "text2text-generation",
    model=trainer.model, tokenizer=tokenizer,
    device=0, batch_size=64,
)
for row in test_rows:
    all_anons.append(row["anonymized"])
    all_refs.append(row["original"])

outputs = gen_pipeline(
    all_anons,
    max_new_tokens=MAX_TARGET_LEN,
    num_beams=NUM_BEAMS_EVAL,
    batch_size=64,
)
all_preds = [o[0]["generated_text"].strip() for o in outputs]

# ── Compute extended metrics ─────────────────────────────────────────────────
bleu_scores = {
    "BLEU-1": round(bleu_metric.compute(predictions=all_preds,
        references=[[r] for r in all_refs], max_order=1)["score"], 2),
    "BLEU-2": round(bleu_metric.compute(predictions=all_preds,
        references=[[r] for r in all_refs], max_order=2)["score"], 2),
    "BLEU-4": round(bleu_metric.compute(predictions=all_preds,
        references=[[r] for r in all_refs], max_order=4)["score"], 2),
}
bleu_corpus = round(
    sum(_sentence_bleu(p,r) for p,r in zip(all_preds,all_refs)) / len(all_preds), 4
)
rouge_scores = rouge_metric.compute(predictions=all_preds, references=all_refs)
tok_acc   = _token_accuracy(all_preds, all_refs, tokenizer)
word_acc  = _word_accuracy(all_preds, all_refs)
exact     = round(sum(p.strip()==r.strip() for p,r in zip(all_preds,all_refs))/len(all_preds),4)
err_exact, err_total = _err(all_preds, all_refs)

test_metrics = {
    "n_test_samples"      : len(all_preds),
    "eval_loss"           : round(eval_loss, 4),
    "perplexity"          : round(perplexity, 2),
    "BLEU-1"              : bleu_scores["BLEU-1"],
    "BLEU-2"              : bleu_scores["BLEU-2"],
    "BLEU-4"              : bleu_scores["BLEU-4"],
    "corpus_BLEU"         : bleu_corpus,
    "ROUGE-1"             : round(rouge_scores["rouge1"], 4),
    "ROUGE-2"             : round(rouge_scores["rouge2"], 4),
    "ROUGE-L"             : round(rouge_scores["rougeL"], 4),
    "token_accuracy"      : tok_acc,
    "word_accuracy"       : word_acc,
    "exact_sentence_match": exact,
    "ERR_exact"           : err_exact,
    "ERR_total_tokens_probed": err_total,
}
print("\\n=== Extended Test Metrics ===")
for k, v in test_metrics.items():
    print(f"  {k:<30}: {v}")
"""

UPLOAD = """\
trainer.push_to_hub(
    repo_id        = OUTPUT_REPO,
    token          = HF_TOKEN,
    commit_message = "Fine-tuned BART PII inverter",
    private        = False,
)
print(f"\\nModel uploaded → https://huggingface.co/{OUTPUT_REPO}")
"""

SAMPLES = """\
import textwrap, random

print("=" * 90)
print("  RECONSTRUCTION SAMPLES (20 random test examples)")
print("=" * 90)

indices = random.sample(range(len(all_preds)), min(20, len(all_preds)))
for rank, i in enumerate(indices, 1):
    anon = all_anons[i]
    pred = all_preds[i]
    ref  = all_refs[i]
    exact_match = "✅" if pred.strip() == ref.strip() else "❌"
    bleu_i = round(_sentence_bleu(pred, ref), 4)
    print(f"\\n── Sample {rank:02d} ─────────────────────────────────────────────────────────────────")
    print(f"  INPUT  (anonymized) : {textwrap.fill(anon, 80, subsequent_indent=' '*22)}")
    print(f"  PREDICTED (original): {textwrap.fill(pred, 80, subsequent_indent=' '*22)}")
    print(f"  REFERENCE (original): {textwrap.fill(ref,  80, subsequent_indent=' '*22)}")
    print(f"  Sentence BLEU: {bleu_i:.4f}  |  Exact match: {exact_match}")
"""

REPORT = """\
from datetime import datetime

report_lines = []
sep = "=" * 72
thin = "-" * 72

report_lines += [
    sep,
    "  MODEL INVERSION ATTACK — PIPELINE MASK-FILL",
    f"  {COMBO_LABEL}",
    f"  Dataset  : {DATASET_ID}",
    f"  Inverter : facebook/bart-base → fine-tuned",
    f"  Reported : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    sep,
    "",
    "  EVALUATION METRICS  (test set, n={:,})".format(test_metrics["n_test_samples"]),
    thin,
    f"  {'Metric':<32} {'Value':>12}",
    thin,
]
metric_order = [
    ("eval_loss"          , "Eval Loss"),
    ("perplexity"         , "Perplexity"),
    ("BLEU-1"             , "BLEU-1"),
    ("BLEU-2"             , "BLEU-2"),
    ("BLEU-4"             , "BLEU-4"),
    ("corpus_BLEU"        , "Corpus BLEU"),
    ("ROUGE-1"            , "ROUGE-1"),
    ("ROUGE-2"            , "ROUGE-2"),
    ("ROUGE-L"            , "ROUGE-L"),
    ("token_accuracy"     , "Token Accuracy"),
    ("word_accuracy"      , "Word Accuracy"),
    ("exact_sentence_match","Exact Sentence Match"),
    ("ERR_exact"          , "Entity Recovery Rate (ERR)"),
]
for key, label in metric_order:
    val = test_metrics.get(key, "—")
    report_lines.append(f"  {label:<32} {val:>12}")

report_lines += [
    thin,
    "",
    "  METRIC DESCRIPTIONS",
    thin,
    "  BLEU-1/2/4      : n-gram precision vs reference (higher = closer to original)",
    "  ROUGE-1/2/L     : recall-oriented overlap (higher = better reconstruction)",
    "  Token Accuracy  : fraction of sub-word tokens matching reference at same position",
    "  Word Accuracy   : fraction of words matching reference at same position",
    "  Exact Match     : fraction of sentences reconstructed verbatim",
    "  ERR (exact)     : fraction of reference word-tokens found in prediction",
    "                    A HIGH ERR means the inverter successfully recovers",
    "                    the original text — this is the PRIMARY attack metric.",
    "  Perplexity      : exp(eval_loss) — model confidence (lower = better fit)",
    "",
    "  INTERPRETATION",
    thin,
    "  For a model inversion ATTACK, higher BLEU/ROUGE/ERR = stronger attack.",
    "  For an anonymisation DEFENCE, we want these numbers to be LOW on an",
    "  adversary's inverter, indicating the anonymised output is hard to reverse.",
    sep,
]

report_text = "\\n".join(report_lines)
print(report_text)

# Save to file
report_path = f"/kaggle/working/{OUTPUT_REPO.replace('/','_')}_eval_report.txt"
with open(report_path, "w") as f:
    f.write(report_text + "\\n")
metrics_path = f"/kaggle/working/{OUTPUT_REPO.replace('/','_')}_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"\\nReport saved  → {report_path}")
print(f"Metrics JSON  → {metrics_path}")
"""

# ── notebook factory ─────────────────────────────────────────────────────────

def make_notebook(title, subtitle, dataset_id, output_repo, combo_label):

    CONFIG = f"""\
# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_ID     = "{dataset_id}"
BASE_MODEL     = "facebook/bart-base"          # BART used as inverter backbone
OUTPUT_REPO    = "{output_repo}"
COMBO_LABEL    = "{combo_label}"

# Sequence lengths  (PII sentences are ~20-60 words; 128 tokens is ample)
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 128

# 2 × T4 (15 GiB each).
# bart-base in fp16 ≈ 300 MB / GPU → lots of headroom for large batches.
PER_DEVICE_BATCH = 32     # per GPU → 64 rows/step across 2 GPUs
GRAD_ACCUM       = 2      # effective batch size = 128
EVAL_BATCH       = 64
LR               = 5e-5
NUM_EPOCHS       = 5
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01
LABEL_SMOOTH     = 0.1
NUM_BEAMS_EVAL   = 4      # beam search for eval / inference
SAVE_TOTAL       = 3      # keep at most 3 checkpoints

CKPT_DIR = "/kaggle/working/checkpoints"
"""

    TITLE_MD = f"""\
# {title}

**Task:** Model-inversion attack on the PII pipeline mask-fill anonymiser  
**Dataset:** `{dataset_id}` (30 k train / 5 k val / 5 k test)  
**Inverter:** `facebook/bart-base` fine-tuned to reconstruct original text from anonymised text  
**Setup:** {subtitle}  
**Output model:** `{output_repo}`

---
The pipeline anonymiser is:

1. **RoBERTa NER** — detect PII spans and replace with `[TYPE]` placeholders  
2. **Filler** — fill each placeholder with a synthetic fake entity

The inverter learns the reverse mapping: *anonymised → original*.
"""

    cells = [
        md(TITLE_MD),
        code(INSTALL),
        code(CONFIG),
        code(IMPORTS),
        md("## 1 · Login to HuggingFace\n\nAdd your HF write token as a Kaggle secret named `HF_TOKEN`."),
        code(HF_LOGIN),
        md("## 2 · Load Dataset"),
        code(LOAD_DATA),
        md("## 3 · Tokenise\n\n`anonymized` → encoder input, `original` → decoder target."),
        code(TOKENIZE),
        md("## 4 · Load Model"),
        code(MODEL_INIT),
        md("## 5 · Metrics\n\nComputes BLEU, ROUGE-L, and helper functions for extended evaluation."),
        code(METRICS),
        md("## 6 · Training Arguments + Trainer\n\nFully utilises both T4 GPUs via HuggingFace `accelerate` / DDP."),
        code(TRAINING_ARGS),
        md("## 7 · Train\n\nAutomatically resumes from the latest checkpoint on Kaggle restarts."),
        code(TRAIN),
        md("## 8 · Test-set Evaluation\n\n"
           "Computes all metrics:\n"
           "- **BLEU-1 / 2 / 4, Corpus BLEU**\n"
           "- **ROUGE-1 / 2 / L**\n"
           "- **Token accuracy, Word accuracy**\n"
           "- **Exact sentence match**\n"
           "- **ERR (Entity Recovery Rate)** — primary attack metric\n"
           "- **Perplexity**"),
        code(TEST_EVAL),
        md("## 9 · Reconstruction Samples\n\n20 random test examples showing anonymized → predicted vs reference."),
        code(SAMPLES),
        md("## 10 · Detailed Evaluation Report"),
        code(REPORT),
        md("## 11 · Upload Model to HuggingFace"),
        code(UPLOAD),
    ]

    return nb(cells)


# ── write notebooks ───────────────────────────────────────────────────────────

notebooks = [
    {
        "filename"   : "combo1_inverter_roberta_deberta.ipynb",
        "title"      : "PII Model-Inversion — Combo 1: RoBERTa-NER + DeBERTa-MLM filler",
        "subtitle"   : "2 × T4 (15 GiB each), fp16, DDP",
        "dataset"    : "JALAPENO11/pipeline-inversion-roberta-deberta",
        "repo"       : "JALAPENO11/pii-inverter-roberta-deberta",
        "combo_label": "Combo 1 — RoBERTa NER + DeBERTa MLM filler",
    },
    {
        "filename"   : "combo2_inverter_roberta_bart.ipynb",
        "title"      : "PII Model-Inversion — Combo 2: RoBERTa-NER + BART seq2seq filler",
        "subtitle"   : "2 × T4 (15 GiB each), fp16, DDP",
        "dataset"    : "JALAPENO11/pipeline-inversion-roberta-bart",
        "repo"       : "JALAPENO11/pii-inverter-roberta-bart",
        "combo_label": "Combo 2 — RoBERTa NER + BART seq2seq filler",
    },
]

for spec in notebooks:
    notebook = make_notebook(
        title       = spec["title"],
        subtitle    = spec["subtitle"],
        dataset_id  = spec["dataset"],
        output_repo = spec["repo"],
        combo_label = spec["combo_label"],
    )
    path = os.path.join(OUT, spec["filename"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Written: {path}")

# ── Local RTX 3050 notebook (Combo 2) ────────────────────────────────────────

LOCAL_CONFIG = """\
# ─── Configuration ─────────────────────────────────────────────────────────────
DATASET_ID     = "JALAPENO11/pipeline-inversion-roberta-bart"
BASE_MODEL     = "facebook/bart-base"
OUTPUT_REPO    = "JALAPENO11/pii-inverter-roberta-bart"
COMBO_LABEL    = "Combo 2 — RoBERTa NER + BART seq2seq filler"

MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 128

# ── RTX 3050 (4 GB VRAM) ──────────────────────────────────────────────────────
# bart-base fp16 ≈ 300 MB; activations + optimiser eat the rest.
# batch=2 + accum=32 → effective batch=64, fits in ~3.5 GB with grad checkpointing.
PER_DEVICE_BATCH = 2
GRAD_ACCUM       = 32     # eff. batch = 64
EVAL_BATCH       = 4      # generation is memory-heavy; keep small
LR               = 5e-5
NUM_EPOCHS       = 5
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01
LABEL_SMOOTH     = 0.1
NUM_BEAMS_EVAL   = 1      # greedy for per-epoch val (fast); beam=4 at final test only
SAVE_TOTAL       = 2

CKPT_DIR = "./checkpoints"
"""

LOCAL_LOGIN = """\
from huggingface_hub import login

HF_TOKEN = ""    # ← paste your HF write token here
assert HF_TOKEN, "Set HF_TOKEN before running!"
login(token=HF_TOKEN, add_to_git_credential=False)
print("Logged in ✓")
"""

LOCAL_IMPORTS = """\
import os, json, math, numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

print("torch  :", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU    : {p.name}  {p.total_memory/1024**3:.1f} GiB")
"""

LOCAL_MODEL = """\
import gc
torch.cuda.empty_cache(); gc.collect()

model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# Gradient checkpointing: recomputes activations during backward instead of
# storing them — halves activation memory at ~20% extra compute cost.
model.gradient_checkpointing_enable()
model.config.use_cache = False   # must be False while grad checkpointing is on

total_params = sum(p.numel() for p in model.parameters())
print(f"Model     : {BASE_MODEL}  ({total_params/1e6:.1f} M params)")
print("Grad ckpt : enabled  (saves ~50% activation VRAM)")

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)
"""

LOCAL_TRAIN_ARGS = """\
os.makedirs(CKPT_DIR, exist_ok=True)

_steps_per_epoch = math.ceil(len(tokenized["train"]) / (PER_DEVICE_BATCH * GRAD_ACCUM))
_total_steps     = _steps_per_epoch * NUM_EPOCHS
_warmup_steps    = max(1, int(_total_steps * WARMUP_RATIO))
print(f"Steps/epoch    : {_steps_per_epoch}")
print(f"Total steps    : {_total_steps}")
print(f"Warmup steps   : {_warmup_steps}")
print(f"Effective batch: {PER_DEVICE_BATCH} × {GRAD_ACCUM} = {PER_DEVICE_BATCH*GRAD_ACCUM}")

training_args = Seq2SeqTrainingArguments(
    output_dir                  = CKPT_DIR,
    num_train_epochs            = NUM_EPOCHS,

    # ── batching (4 GB budget) ────────────────────────────────────────────────
    per_device_train_batch_size = PER_DEVICE_BATCH,
    per_device_eval_batch_size  = EVAL_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,

    # ── optimiser ─────────────────────────────────────────────────────────────
    learning_rate               = LR,
    warmup_steps                = _warmup_steps,
    weight_decay                = WEIGHT_DECAY,
    label_smoothing_factor      = LABEL_SMOOTH,

    # ── memory ────────────────────────────────────────────────────────────────
    fp16                        = True,
    gradient_checkpointing      = True,

    # ── generation during per-epoch val (greedy = 4× faster) ─────────────────
    predict_with_generate       = True,
    generation_max_length       = MAX_TARGET_LEN,
    generation_num_beams        = NUM_BEAMS_EVAL,   # 1 = greedy

    # ── checkpointing ─────────────────────────────────────────────────────────
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "bleu",
    greater_is_better           = True,
    save_total_limit            = SAVE_TOTAL,

    # ── logging ───────────────────────────────────────────────────────────────
    logging_steps               = 50,
    report_to                   = "none",
    dataloader_num_workers      = 0,
)

trainer = Seq2SeqTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = tokenized["train"],
    eval_dataset     = tokenized["val"],
    processing_class = tokenizer,
    data_collator    = data_collator,
    compute_metrics  = compute_metrics,
    callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
)
"""

LOCAL_TEST_EVAL = """\
print("\\n=== Full evaluation on test set ===")
raw_test_results = trainer.evaluate(eval_dataset=tokenized["test"])
eval_loss   = raw_test_results.get("eval_loss", float("nan"))
perplexity  = math.exp(min(eval_loss, 20))
print(f"  eval_loss : {eval_loss:.4f}")
print(f"  perplexity: {perplexity:.2f}")
print(f"  BLEU      : {raw_test_results.get('eval_bleu', 0):.4f}")
print(f"  ROUGE-1   : {raw_test_results.get('eval_rouge1', 0):.4f}")
print(f"  ROUGE-L   : {raw_test_results.get('eval_rougeL', 0):.4f}")

# Re-enable cache for generation
model.config.use_cache = True

print("\\nGenerating predictions on test set (beam=4)…")
all_anons, all_refs = [], []
for row in dataset["test"]:
    all_anons.append(row["anonymized"])
    all_refs.append(row["original"])

gen_pipeline = __import__("transformers").pipeline(
    "text2text-generation",
    model=trainer.model, tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    batch_size=EVAL_BATCH,
)
outputs   = gen_pipeline(all_anons, max_new_tokens=MAX_TARGET_LEN,
                         num_beams=4, batch_size=EVAL_BATCH)
all_preds = [o[0]["generated_text"].strip() for o in outputs]

bleu_scores = {
    "BLEU-1": round(bleu_metric.compute(predictions=all_preds,
        references=[[r] for r in all_refs], max_order=1)["score"], 2),
    "BLEU-2": round(bleu_metric.compute(predictions=all_preds,
        references=[[r] for r in all_refs], max_order=2)["score"], 2),
    "BLEU-4": round(bleu_metric.compute(predictions=all_preds,
        references=[[r] for r in all_refs], max_order=4)["score"], 2),
}
bleu_corpus  = round(sum(_sentence_bleu(p,r) for p,r in zip(all_preds,all_refs))/len(all_preds),4)
rouge_scores = rouge_metric.compute(predictions=all_preds, references=all_refs)
tok_acc   = _token_accuracy(all_preds, all_refs, tokenizer)
word_acc  = _word_accuracy(all_preds, all_refs)
exact     = round(sum(p.strip()==r.strip() for p,r in zip(all_preds,all_refs))/len(all_preds),4)
err_exact, err_total = _err(all_preds, all_refs)

test_metrics = {
    "n_test_samples"       : len(all_preds),
    "eval_loss"            : round(eval_loss, 4),
    "perplexity"           : round(perplexity, 2),
    "BLEU-1"               : bleu_scores["BLEU-1"],
    "BLEU-2"               : bleu_scores["BLEU-2"],
    "BLEU-4"               : bleu_scores["BLEU-4"],
    "corpus_BLEU"          : bleu_corpus,
    "ROUGE-1"              : round(rouge_scores["rouge1"], 4),
    "ROUGE-2"              : round(rouge_scores["rouge2"], 4),
    "ROUGE-L"              : round(rouge_scores["rougeL"], 4),
    "token_accuracy"       : tok_acc,
    "word_accuracy"        : word_acc,
    "exact_sentence_match" : exact,
    "ERR_exact"            : err_exact,
    "ERR_total_tokens_probed": err_total,
}
print("\\n=== Extended Test Metrics ===")
for k, v in test_metrics.items():
    print(f"  {k:<32}: {v}")
"""

LOCAL_REPORT = """\
from datetime import datetime
import textwrap, random

# ── 20 reconstruction samples ─────────────────────────────────────────────────
print("=" * 90)
print("  RECONSTRUCTION SAMPLES (20 random test examples)")
print("=" * 90)
indices = random.sample(range(len(all_preds)), min(20, len(all_preds)))
for rank, i in enumerate(indices, 1):
    anon, pred, ref = all_anons[i], all_preds[i], all_refs[i]
    match  = "✅" if pred.strip() == ref.strip() else "❌"
    bleu_i = round(_sentence_bleu(pred, ref), 4)
    print(f"\\n── Sample {rank:02d} {'─'*68}")
    print(f"  INPUT  : {textwrap.fill(anon, 80, subsequent_indent=' '*11)}")
    print(f"  PRED   : {textwrap.fill(pred, 80, subsequent_indent=' '*11)}")
    print(f"  REF    : {textwrap.fill(ref,  80, subsequent_indent=' '*11)}")
    print(f"  BLEU: {bleu_i:.4f}  |  Exact: {match}")

# ── formatted report ──────────────────────────────────────────────────────────
sep, thin = "=" * 72, "-" * 72
report_lines = [
    sep,
    "  MODEL INVERSION ATTACK — PIPELINE MASK-FILL",
    f"  {COMBO_LABEL}",
    f"  Dataset  : {DATASET_ID}",
    f"  Inverter : facebook/bart-base fine-tuned",
    f"  Hardware : RTX 3050 (~4 GB VRAM), fp16, gradient checkpointing",
    f"  Reported : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    sep, "",
    "  EVALUATION METRICS  (test set, n={:,})".format(test_metrics["n_test_samples"]),
    thin, f"  {'Metric':<32} {'Value':>12}", thin,
]
for key, label in [
    ("eval_loss","Eval Loss"),("perplexity","Perplexity"),
    ("BLEU-1","BLEU-1"),("BLEU-2","BLEU-2"),("BLEU-4","BLEU-4"),
    ("corpus_BLEU","Corpus BLEU"),("ROUGE-1","ROUGE-1"),("ROUGE-2","ROUGE-2"),
    ("ROUGE-L","ROUGE-L"),("token_accuracy","Token Accuracy"),
    ("word_accuracy","Word Accuracy"),("exact_sentence_match","Exact Sentence Match"),
    ("ERR_exact","Entity Recovery Rate (ERR)"),
]:
    report_lines.append(f"  {label:<32} {test_metrics.get(key,'—'):>12}")

report_lines += [
    thin, "",
    "  ERR (primary attack metric): fraction of reference tokens found in prediction.",
    "  HIGH ERR = strong attack. LOW ERR = anonymisation is robust.",
    sep,
]
report_text = "\\n".join(report_lines)
print("\\n" + report_text)

report_path  = f"./{OUTPUT_REPO.replace('/','_')}_eval_report.txt"
metrics_path = f"./{OUTPUT_REPO.replace('/','_')}_metrics.json"
with open(report_path,  "w") as f: f.write(report_text + "\\n")
with open(metrics_path, "w") as f: json.dump(test_metrics, f, indent=2)
print(f"\\nReport  → {report_path}")
print(f"Metrics → {metrics_path}")
"""

LOCAL_UPLOAD = """\
trainer.push_to_hub(
    repo_id        = OUTPUT_REPO,
    token          = HF_TOKEN,
    commit_message = "Fine-tuned BART PII inverter — combo2, RTX 3050",
    private        = False,
)
print(f"\\nModel uploaded → https://huggingface.co/{OUTPUT_REPO}")
"""

local_cells = [
    md("# PII Model-Inversion — Combo 2 (RTX 3050, ~4 GB VRAM)\n\n"
       "**Task:** Model-inversion attack — reconstruct original text from anonymised text  \n"
       "**Dataset:** `JALAPENO11/pipeline-inversion-roberta-bart` (30 k train / 5 k val / 5 k test)  \n"
       "**Inverter:** `facebook/bart-base` fine-tuned  \n"
       "**Hardware:** RTX 3050 ~4 GB VRAM — fp16, gradient checkpointing, batch=2 + accum=32  \n"
       "**Effective batch size:** 64  \n"
       "**Output model:** `JALAPENO11/pii-inverter-roberta-bart`"),
    code("%%capture\n!pip install -q --upgrade transformers datasets accelerate evaluate sacrebleu rouge_score huggingface_hub"),
    code(LOCAL_CONFIG),
    code(LOCAL_IMPORTS),
    md("## 1 · Login to HuggingFace\n\nPaste your HF **write** token at https://huggingface.co/settings/tokens."),
    code(LOCAL_LOGIN),
    md("## 2 · Load Dataset"),
    code("dataset = load_dataset(DATASET_ID)\nprint(dataset)\nprint(dataset['train'][0])"),
    md("## 3 · Tokenise\n\n`anonymized` → encoder input, `original` → decoder target."),
    code(TOKENIZE),
    md("## 4 · Load Model\n\nGradient checkpointing enabled — halves activation VRAM at ~20% speed cost."),
    code(LOCAL_MODEL),
    md("## 5 · Metrics"),
    code(METRICS),
    md("## 6 · Training Arguments + Trainer\n\n"
       "Key 4 GB settings: `fp16=True`, `gradient_checkpointing=True`, "
       "`per_device_train_batch_size=2`, `gradient_accumulation_steps=32`, "
       "`generation_num_beams=1` (greedy, 4× faster per-epoch eval)."),
    code(LOCAL_TRAIN_ARGS),
    md("## 7 · Train\n\nAuto-resumes from the latest checkpoint if interrupted."),
    code(TRAIN),
    md("## 8 · Test-set Evaluation\n\n"
       "Computes BLEU-1/2/4, ROUGE-1/2/L, token accuracy, word accuracy, "
       "exact sentence match, ERR, and perplexity. Final generation uses beam=4."),
    code(LOCAL_TEST_EVAL),
    md("## 9 · Reconstruction Samples + Detailed Report"),
    code(LOCAL_REPORT),
    md("## 10 · Upload Model to HuggingFace"),
    code(LOCAL_UPLOAD),
]

local_nb = nb(local_cells)
local_path = os.path.join(OUT, "combo2_inverter_roberta_bart_rtx3050.ipynb")
with open(local_path, "w", encoding="utf-8") as f:
    json.dump(local_nb, f, indent=1, ensure_ascii=False)
print(f"Written: {local_path}")

