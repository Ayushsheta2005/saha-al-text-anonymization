#!/usr/bin/env python3
"""
push_inverter_to_hub.py
───────────────────────
Converts inverter_checkpoint/best_model.pt (raw state_dict) into a
HuggingFace Seq2Seq model and pushes it to the Hub.

Also uploads output/attack_report.txt as a Hub file.

Usage
─────
  python3 push_inverter_to_hub.py

Environment / secrets
  Set HF_TOKEN in the environment OR edit HF_TOKEN below.
  Repo name can be changed with --repo_id.

  python3 push_inverter_to_hub.py --repo_id JALAPENO11/pii-inverter-bart-base
"""

import argparse, json, os, sys, tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi, login

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CKPT_PATH  = BASE_DIR / "inverter_checkpoint" / "best_model.pt"
REPORT_PATH= BASE_DIR / "output" / "attack_report.txt"

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_REPO    = "JALAPENO11/pii-inverter-bart-base"
BASE_MODEL_NAME = "facebook/bart-base"

# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id",   default=DEFAULT_REPO,
                   help="HuggingFace repo to push to (created if absent)")
    p.add_argument("--hf_token",  default=os.environ.get("HF_TOKEN", ""),
                   help="HF write token (falls back to HF_TOKEN env var)")
    p.add_argument("--private",   action="store_true",
                   help="Make the repo private")
    return p.parse_args()


def build_model_card(meta: dict, repo_id: str) -> str:
    """Build a README.md model card from checkpoint metrics."""
    m = meta.get("metrics", {})
    return f"""---
language: en
license: mit
tags:
  - text2text-generation
  - model-inversion
  - pii
  - bart
  - seq2seq
base_model: facebook/bart-base
---

# PII Model-Inversion Inverter — BART-base

**Repo:** `{repo_id}`

This model is a **model-inversion attack** against a PII anonymisation system.
It was trained to reverse the output of a BART-base anonymiser:

> anonymised text → (inverter) → recovered original text

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `facebook/bart-base` |
| Checkpoint epoch | {meta.get('epoch', '?')} |
| Global step | {meta.get('global_step', '?')} |
| Saved at | {meta.get('saved_at', '?')} |
| Train loss | {m.get('train_loss', '?')} |
| Eval loss | {m.get('eval_loss', '?')} |
| Perplexity | {m.get('perplexity', '?')} |
| Token accuracy | {m.get('token_accuracy', '?')} |
| ERR exact | {m.get('err_exact', '?')} |
| ERR partial | {m.get('err_partial', '?')} |
| Eval samples | {m.get('n_eval', '?')} |

---

## Evaluation Metrics

### Attack B — Inverter Model (best checkpoint)

| Metric | Value |
|---|---|
| ERR exact | {m.get('err_exact', '?')} ({float(m.get('err_exact', 0))*100:.1f}% entities recovered verbatim) |
| ERR partial | {m.get('err_partial', '?')} |
| Token accuracy | {m.get('token_accuracy', '?')} |
| Exact sentence match | 0.1528 |
| Corpus BLEU | 0.6466 |

### ERR by adversarial strategy

| Strategy | Samples | ERR exact |
|---|---|---|
{chr(10).join(
    f"| {k} | {v['total']} | {v['err_exact']:.4f} |"
    for k, v in m.get('err_by_strategy', {}).items()
)}

### ERR by name rarity

| Rarity | Samples | ERR exact |
|---|---|---|
{chr(10).join(
    f"| {k} | {v['total']} | {v['err_exact']:.4f} |"
    for k, v in m.get('err_by_rarity', {}).items()
)}

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model     = AutoModelForSeq2SeqLM.from_pretrained("{repo_id}")
model.eval()

anonymized = "Dear Ms. Anya Sharma, please update your account details."

inputs = tokenizer(anonymized, return_tensors="pt",
                   max_length=128, truncation=True)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# → "Dear Amyna Tharathip, please update your account details."
```

---

## Disclaimer

This model is published for **research purposes only** to study the
robustness of PII anonymisation systems.  It should not be used to
de-anonymise real personal data.

See `attack_report.txt` in the repo Files tab for the full evaluation report.
"""


def main():
    args = parse_args()

    if not args.hf_token:
        sys.exit("ERROR: No HF token found. "
                 "Set HF_TOKEN environment variable or pass --hf_token.")

    login(token=args.hf_token, add_to_git_credential=False)
    print(f"Logged in to HuggingFace ✓")

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    print(f"  epoch      : {ckpt['epoch']}")
    print(f"  global_step: {ckpt['global_step']}")
    print(f"  eval_loss  : {ckpt['metrics'].get('eval_loss')}")
    print(f"  err_exact  : {ckpt['metrics'].get('err_exact')}")

    # ── Reconstruct HF model ───────────────────────────────────────────────
    print(f"\nLoading base architecture: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    print("Loading state_dict into model ...")
    # The raw checkpoint may have 'module.' prefix if trained with DDP.
    state = ckpt["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    print("  State dict loaded ✓")

    # ── Save temporarily ───────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(f"\nSaving HF model to temp dir: {tmp}")
        model.save_pretrained(tmp, safe_serialization=False)
        tokenizer.save_pretrained(tmp)

        # Write model card
        card = build_model_card(ckpt, args.repo_id)
        (tmp / "README.md").write_text(card, encoding="utf-8")

        # Copy attack report
        if REPORT_PATH.exists():
            (tmp / "attack_report.txt").write_bytes(REPORT_PATH.read_bytes())
            print(f"  Included: attack_report.txt")
        else:
            print(f"  WARNING: attack_report.txt not found at {REPORT_PATH}")

        # Save checkpoint metadata as JSON for reference
        meta_out = {
            "epoch"      : ckpt["epoch"],
            "global_step": ckpt["global_step"],
            "saved_at"   : ckpt["saved_at"],
            "base_model" : BASE_MODEL_NAME,
            "metrics"    : {k: v for k, v in ckpt["metrics"].items()
                            if k != "samples"},   # exclude bulky sample list
        }
        (tmp / "training_metadata.json").write_text(
            json.dumps(meta_out, indent=2), encoding="utf-8")
        print(f"  Included: training_metadata.json")

        # ── Push to Hub ────────────────────────────────────────────────────
        print(f"\nPushing to Hub: {args.repo_id}  (private={args.private})")
        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            token=args.hf_token,
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=args.repo_id,
            token=args.hf_token,
            commit_message="Upload BART-base PII inverter (best checkpoint) + attack report",
        )

    print(f"\nDone!  Model live at → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
