"""
PyTorch Dataset for Seq2Seq PII Anonymization.
Handles tokenization for both T5-family and BART-family models.
Supports on-the-fly text augmentation for training robustness.
"""

import json
import torch
from torch.utils.data import Dataset

from augmentations import TextAugmentor


class AnonymizationDataset(Dataset):
    """
    Dataset that reads JSONL split files and tokenizes on-the-fly.
    On-the-fly tokenization keeps RAM usage low for 120K+ entries.
    
    Args:
        augmentor: optional TextAugmentor instance. If provided, augmentations
            are applied on-the-fly during __getitem__. Should only be used
            for training — NOT for validation/test sets.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
        prefix: str = "",
        augmentor: TextAugmentor = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Input: original text (with PII) — model learns to anonymize this
        input_text = entry["original_text"]
        # Target: anonymized text (with fake PII)
        target_text = entry["anonymized_text"]

        # Apply augmentation (only during training, augmentor is None for val/test)
        if self.augmentor is not None:
            input_text, target_text = self.augmentor(input_text, target_text)

        # Add prefix after augmentation (prefix should stay clean)
        input_text = self.prefix + input_text

        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it's ignored in loss
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": label_ids,
        }


def load_split_data(filepath: str) -> list[dict]:
    """Load a JSONL split file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
