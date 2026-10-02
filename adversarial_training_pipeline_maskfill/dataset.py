"""
dataset.py — PyTorch Dataset classes for Pipeline Adversarial Training
======================================================================
Two dataset classes:

  PipelineAdvDataset (combo=1)
      Wraps the adv_train/eval JSONL files for the DeBERTa-MLM filler.
      Input:  masked_mlm  (text with [MASK] tokens at entity positions)
      Target: anonymized  (gold anonymized text — L1 quality anchor)
      Inv-target: original (text the frozen inverter must FAIL to recover)

  PipelineAdvDataset (combo=2)
      Same wrapping, but input is masked_s2s (prompt + [TYPE]-tagged text)
      passed to the BART seq2seq filler.

Both classes expose the same __getitem__ signature so train_adv.py can
use a single collate_fn regardless of combo.

Fields returned per sample
--------------------------
  filler_input_ids        (B, T)  — tokenized masked text (filler input)
  filler_attention_mask   (B, T)
  anon_labels             (B, T)  — tokenized anonymized (L1 teacher-forcing target)
  anon_attention_mask     (B, T)  — used as attention mask for soft embeds fed to inverter
  inv_labels              (B, T)  — tokenized original (inverter decoder target, negated)
  entity_texts            list[str]  — original PII strings (ERR evaluation only)
  original_text           str        — for logging
"""

import json
import os
import sys
import torch
from torch.utils.data import Dataset


# ── Helpers ────────────────────────────────────────────────────────────────

def _hf_download_dataset(local_path: str) -> None:
    """Pull the missing JSONL file from HuggingFace Hub into local_path."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import HF_DATASET_REPO  # imported here to avoid circular deps

    filename = os.path.basename(local_path)   # adv_train.jsonl or adv_eval.jsonl
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"[dataset] '{filename}' not found locally — downloading from "
          f"huggingface.co/datasets/{HF_DATASET_REPO} …")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=os.path.dirname(local_path),
        )
        # hf_hub_download may place the file in a subdirectory; move if needed.
        if os.path.abspath(downloaded) != os.path.abspath(local_path):
            import shutil
            shutil.move(downloaded, local_path)
        print(f"[dataset] Downloaded → {local_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download '{filename}' from HuggingFace ({HF_DATASET_REPO}). "
            f"Either set HF_TOKEN env var or run create_dataset.py first.\n"
            f"Original error: {exc}"
        ) from exc


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        _hf_download_dataset(path)
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Main dataset ───────────────────────────────────────────────────────────

class PipelineAdvDataset(Dataset):
    """
    Universal dataset for pipeline adversarial training.

    Parameters
    ----------
    data        : list of dicts from adv_train.jsonl / adv_eval.jsonl
    filler_tokenizer  : tokenizer for the filler model
                        Combo 1 → DeBERTa tokenizer
                        Combo 2 → BART tokenizer
    inverter_tokenizer: tokenizer for the frozen BART inverter
                        Both combos → BART tokenizer
    combo       : 1 or 2  (selects masked_mlm vs masked_s2s as filler input)
    max_input_length  : max tokens for filler input
    max_target_length : max tokens for decoder labels
    """

    def __init__(
        self,
        data: list[dict],
        filler_tokenizer,
        inverter_tokenizer,
        combo: int,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        assert combo in (1, 2), "combo must be 1 or 2"
        # Drop rows where any required field is missing
        self.data = [
            d for d in data
            if d.get("original", "").strip()
            and d.get("anonymized", "").strip()
            and d.get("masked_mlm" if combo == 1 else "masked_s2s", "").strip()
        ]
        self.filler_tok    = filler_tokenizer
        self.inverter_tok  = inverter_tokenizer
        self.combo         = combo
        self.max_in_len    = max_input_length
        self.max_tgt_len   = max_target_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]

        masked_field = "masked_mlm" if self.combo == 1 else "masked_s2s"
        masked_text  = row[masked_field].strip()
        anonymized   = row["anonymized"].strip()
        original     = row["original"].strip()

        # ── Filler encoder input: masked text ─────────────────────────────
        enc_in = self.filler_tok(
            masked_text,
            max_length=self.max_in_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Filler decoder target: anonymized text (L1 teacher-forcing) ───
        # For Combo 1 (MLM), these are used as MLM labels at [MASK] positions.
        # For Combo 2 (seq2seq), these are the decoder output labels.
        # Also serves as the attention mask reference for the soft embeddings
        # fed to the frozen inverter encoder.
        enc_anon = self.filler_tok(
            anonymized,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        anon_labels = enc_anon["input_ids"].squeeze(0).clone()
        anon_labels[anon_labels == self.filler_tok.pad_token_id] = -100
        anon_attention_mask = enc_anon["attention_mask"].squeeze(0)

        # ── Inverter decoder target: original text (L2 — negated loss) ────
        # Tokenised with the BART inverter's own tokenizer so the inverter
        # decoder can be teacher-forced during the adversarial forward pass.
        enc_orig_inv = self.inverter_tok(
            original,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inv_labels = enc_orig_inv["input_ids"].squeeze(0).clone()
        inv_labels[inv_labels == self.inverter_tok.pad_token_id] = -100

        return {
            # Filler
            "filler_input_ids":       enc_in["input_ids"].squeeze(0),
            "filler_attention_mask":  enc_in["attention_mask"].squeeze(0),
            # L1 labels (filler decoder target)
            "anon_labels":            anon_labels,
            "anon_attention_mask":    anon_attention_mask,
            # L2 labels (inverter decoder target — negated)
            "inv_labels":             inv_labels,
            # Metadata (strings — handled by collate_fn)
            "entity_texts":  row.get("entity_texts", []),
            "original_text": original,
        }


# ── Collate function ────────────────────────────────────────────────────────

def pipeline_adv_collate_fn(batch: list[dict]) -> dict:
    """Stack tensors; keep string/list metadata as plain Python lists."""
    return {
        "filler_input_ids":      torch.stack([b["filler_input_ids"]      for b in batch]),
        "filler_attention_mask": torch.stack([b["filler_attention_mask"] for b in batch]),
        "anon_labels":           torch.stack([b["anon_labels"]           for b in batch]),
        "anon_attention_mask":   torch.stack([b["anon_attention_mask"]   for b in batch]),
        "inv_labels":            torch.stack([b["inv_labels"]            for b in batch]),
        "entity_texts":          [b["entity_texts"]  for b in batch],
        "original_text":         [b["original_text"] for b in batch],
    }
