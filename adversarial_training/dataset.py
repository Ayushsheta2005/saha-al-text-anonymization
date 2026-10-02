"""
Datasets for Adversarial Training
===================================
Two dataset classes:

  NormalDataset   — loads Seq2Seq_model/data_splits/{train,val}.jsonl
                    Fields: original_text → anonymized_text
                    Same format as the original Seq2Seq training.

  AdvDataset      — loads model_inversion/output/inverter_{train,eval}.jsonl
                    Fields: original → anonymized
                    These are the (PII text, BART-anonymized) pairs from the
                    adversarial querying step.  Used to compute the adversarial
                    loss: we feed the victim's soft output to the frozen inverter
                    and want to maximise the inverter's error.
"""

import json
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(filepath: str) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Normal task dataset  (original anonymization training data)
# ─────────────────────────────────────────────────────────────────────────────

class NormalDataset(Dataset):
    """
    Wraps the standard PII anonymization training/validation splits.

    Each sample has:
      input  : original_text  (text with real PII)
      target : anonymized_text (text with synthetic PII replacements)

    No prefix for BART-base (unlike T5 models).
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
        prefix: str = "",
    ):
        self.data             = data
        self.tokenizer        = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix           = prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        input_text  = self.prefix + entry["original_text"]
        target_text = entry["anonymized_text"]

        # Tokenise input (encoder)
        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenise target (decoder labels)
        dec = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
            # kept for entity-leakage evaluation
            "entity_texts":   entry.get("entity_texts", []),
            "original_text":  entry["original_text"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial dataset  (model-inversion queried pairs)
# ─────────────────────────────────────────────────────────────────────────────

class AdvDataset(Dataset):
    """
    Wraps the adversarial (original, anonymized) pairs from the model-inversion
    querying step (inverter_train.jsonl / inverter_eval.jsonl).

    During adversarial training:
      • 'original'   — fed to the VICTIM BART encoder (same as normal task)
      • 'anonymized' — the target for the VICTIM BART decoder (teacher-forcing)
                       and ALSO the reference for the soft-embedding passed to
                       the frozen inverter encoder.
      • 'original'   — the teacher-forcing target for the frozen INVERTER decoder;
                       we want the inverter to FAIL here (loss is negated).

    Each sample exposes both tokenisations so that train_adv.py can build the
    adversarial loss path without additional tokenisation calls.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        # Keep only samples where both original and anonymized are non-empty
        self.data = [
            d for d in data
            if d.get("original", "").strip() and d.get("anonymized", "").strip()
        ]
        self.tokenizer         = tokenizer
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        original   = pair["original"].strip()
        anonymized = pair["anonymized"].strip()

        # ── Victim encoder input: original PII text ──────────────────────────
        enc_orig = self.tokenizer(
            original,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Victim decoder target: anonymized text (teacher-forcing) ─────────
        # These labels are also used to build the attention mask for the soft
        # embeddings fed to the inverter encoder.
        enc_anon = self.tokenizer(
            anonymized,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        anon_labels = enc_anon["input_ids"].squeeze(0).clone()
        anon_labels[anon_labels == self.tokenizer.pad_token_id] = -100

        # ── Inverter decoder target: original text (teacher-forcing) ─────────
        # We want the inverter to be BAD at predicting this → we negate its loss.
        enc_orig_inv = self.tokenizer(
            original,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        orig_inv_labels = enc_orig_inv["input_ids"].squeeze(0).clone()
        orig_inv_labels[orig_inv_labels == self.tokenizer.pad_token_id] = -100

        # Attention mask for the anonymized sequence
        # (1 for real tokens, 0 for padding) — used when running inverter encoder
        anon_attention_mask = enc_anon["attention_mask"].squeeze(0)

        return {
            # Victim encoder
            "orig_input_ids":      enc_orig["input_ids"].squeeze(0),
            "orig_attention_mask": enc_orig["attention_mask"].squeeze(0),
            # Victim decoder target (anonymized labels — for L_normal in adv pass,
            # and to derive soft embeddings for the inverter)
            "anon_labels":         anon_labels,
            "anon_attention_mask": anon_attention_mask,
            # Inverter decoder target (original labels — negated in loss)
            "inv_labels":          orig_inv_labels,
            # Metadata
            "probe_entity":   pair.get("probe_entity", ""),
            "entity_type":    pair.get("entity_type", ""),
            "strategy":       pair.get("strategy", ""),
            "name_rarity":    pair.get("name_rarity", ""),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom collate for AdvDataset (metadata fields are strings)
# ─────────────────────────────────────────────────────────────────────────────

def adv_collate_fn(batch: list[dict]) -> dict:
    """Stack tensors, keep string metadata as lists."""
    return {
        "orig_input_ids":      torch.stack([b["orig_input_ids"]      for b in batch]),
        "orig_attention_mask": torch.stack([b["orig_attention_mask"] for b in batch]),
        "anon_labels":         torch.stack([b["anon_labels"]         for b in batch]),
        "anon_attention_mask": torch.stack([b["anon_attention_mask"] for b in batch]),
        "inv_labels":          torch.stack([b["inv_labels"]          for b in batch]),
        "probe_entity":        [b["probe_entity"]  for b in batch],
        "entity_type":         [b["entity_type"]   for b in batch],
        "strategy":            [b["strategy"]      for b in batch],
        "name_rarity":         [b["name_rarity"]   for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gold dataset  (benchmark pairs — LLM-synthesised + reviewed, NOT from BART)
# ─────────────────────────────────────────────────────────────────────────────

import random as _random


class GoldDataset(Dataset):
    """
    Wraps benchmark/data/train.jsonl (original_text → anonymized_text).
    These are LLM-synthesised and human-reviewed pairs independent of the
    victim BART model, providing clean L1 anchor references.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
        max_samples=None,
    ):
        self.data = [
            d for d in data
            if d.get("original_text", "").strip() and d.get("anonymized_text", "").strip()
        ]
        if max_samples and len(self.data) > max_samples:
            _random.seed(42)
            self.data = _random.sample(self.data, max_samples)
        self.tokenizer         = tokenizer
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        enc = self.tokenizer(
            entry["original_text"].strip(),
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            entry["anonymized_text"].strip(),
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "orig_input_ids":      enc["input_ids"].squeeze(0),
            "orig_attention_mask": enc["attention_mask"].squeeze(0),
            "anon_labels":         labels,
        }


def gold_collate_fn(batch: list[dict]) -> dict:
    return {
        "orig_input_ids":      torch.stack([b["orig_input_ids"]      for b in batch]),
        "orig_attention_mask": torch.stack([b["orig_attention_mask"] for b in batch]),
        "anon_labels":         torch.stack([b["anon_labels"]         for b in batch]),
    }
