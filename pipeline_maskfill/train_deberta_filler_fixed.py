#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  FIXED DEBERTA-MLM FILLER TRAINING
  Trains DeBERTa to fill *only* PII spans, rather than random 15% of text.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer
)

# Configuration
MODEL_NAME = "microsoft/deberta-v3-base"
DATASET_NAME = "ai4privacy/open-pii-masking-500k-ai4privacy"
OUTPUT_DIR = "outputs/deberta-mlm-targeted-filler"
EPOCHS = 8
BATCH_SIZE = 8
MAX_LENGTH = 256
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("1. Loading dataset...")
    # Load just a subset for demonstration/speed if needed, or the full train split
    ds = load_dataset(DATASET_NAME, split="train")
    
    # Simple split for training/eval
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds["train"].select(range(min(len(ds["train"]), 100000)))  # Subsample for faster training loop
    val_ds = ds["test"].select(range(min(len(ds["test"]), 5000)))

    print(f"2. Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)

    # DeBERTa v3 fp16 layernorm fix
    for name, param in model.named_parameters():
        if "LayerNorm" in name or "layernorm" in name:
            param.data = param.data.to(torch.float32)

    def targeted_masking_preprocess(examples):
        """
        Locates the exact PII spans using character offsets, tokenizes the text, 
        and replaces *only* the PII tokens with <mask>. Sets all non-PII labels to -100.
        """
        batch_input_ids = []
        batch_labels = []
        
        # Process each example
        for i in range(len(examples["source_text"])):
            text = examples["source_text"][i]
            masks = examples["privacy_mask"][i]
            
            # 1. Tokenize the sentence normally
            encoding = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length")
            input_ids = encoding["input_ids"]
            labels = [-100] * MAX_LENGTH
            
            # 2. Find which token indices correspond to PII characters
            pii_char_indices = set()
            for mask in masks:
                start = mask["start"]
                end = mask["end"]
                # Mark these characters as "PII"
                for char_idx in range(start, end):
                    pii_char_indices.add(char_idx)
            
            # 3. Map characters -> tokens and apply masks
            for token_idx in range(len(input_ids)):
                # Ignore special tokens and padding
                if input_ids[token_idx] in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    continue
                    
                span = encoding.token_to_chars(token_idx)
                if span is None:
                    continue
                    
                # If this token overlaps with our PII characters, mask it!
                if span.start in pii_char_indices or (span.end - 1) in pii_char_indices:
                    # Save the original token for the loss calculation
                    labels[token_idx] = input_ids[token_idx]
                    # Replace the input token with <mask>
                    input_ids[token_idx] = tokenizer.mask_token_id
                    
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            
        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": [tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length")["attention_mask"] for text in examples["source_text"]]
        }

    print("3. Preprocessing with targeted PII masks...")
    train_tok = train_ds.map(targeted_masking_preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(targeted_masking_preprocess, batched=True, remove_columns=val_ds.column_names)

    # Note: We do NOT use DataCollatorForLanguageModeling here. 
    # Our preprocessing function already created the exact labels and masked the input_ids.
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        logging_steps=50,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        # No special collator needed; dicts have input_ids, labels, attention_mask
    )

    print("4. Starting Targeted Training...")
    trainer.train()
    
    print(f"5. Saving fixed model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
