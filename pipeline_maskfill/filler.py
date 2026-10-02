# ==============================================================================
# filler.py — Encoder-Decoder Filler: training, inference
# ==============================================================================
# Trains BART-BASE or Flan-T5-base to fill masked PII placeholders with
# natural, coherent replacements learned from real text (Half-B).
#
# Features:
#   - Full fine-tuning (BART) or QLoRA (Flan-T5)
#   - Sample inferences after each epoch
#   - Checkpoint save/resume
# ==============================================================================

import os
import gc
import logging
import random
import re
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from config import (
    SEED, DEVICE, OUTPUT_DIR,
    BF16_OK, FP16_OK,
    FILLER_REGISTRY,
    LOG_EVERY_N_STEPS, SAMPLE_INFERENCE_COUNT,
)
from data import create_filler_pair, tokenize_filler_pairs

log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# Build Filler Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_filler(model_name: str) -> Tuple:
    """
    Build a filler model + tokenizer from the registry.
    """
    if model_name not in FILLER_REGISTRY:
        raise ValueError(f"Unknown filler: {model_name}. "
                         f"Available: {list(FILLER_REGISTRY.keys())}")

    cfg = FILLER_REGISTRY[model_name]
    hf_name = cfg["hf_name"]
    mtype = cfg.get("type", "seq2seq")

    log.info(f"Building filler: {model_name} ({hf_name})")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = AutoModelForSeq2SeqLM if mtype == "seq2seq" else AutoModelForMaskedLM

    if cfg.get("use_qlora"):
        log.info("  Using QLoRA (4-bit NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16_OK else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = model_cls.from_pretrained(
            hf_name, quantization_config=bnb_config, device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if mtype == "seq2seq" else TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_targets"],
        )
        model = get_peft_model(model, lora_config)
    else:
        log.info("  Full fine-tuning")
        model = model_cls.from_pretrained(hf_name).to(DEVICE)

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Sample Inference Callback
# ═══════════════════════════════════════════════════════════════════════════════

class FillerSampleInferenceCallback(TrainerCallback):
    """
    After each epoch, run inference on sample masked texts and log:
      Input (masked) → Generated output vs Gold target
    """
    def __init__(self, sample_examples, tokenizer, cfg, n_samples=SAMPLE_INFERENCE_COUNT):
        self.samples = sample_examples[:n_samples]
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.n_samples = n_samples

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        epoch = int(state.epoch)
        log.info(f"\n  ═══ Epoch {epoch} — Sample Filler Inferences ═══")

        for i, sample in enumerate(self.samples):
            pair = create_filler_pair(sample)
            input_text = pair["input_text"]
            gold = pair["target_text"]

            enc = self.tokenizer(
                input_text, return_tensors="pt", truncation=True,
                max_length=self.cfg["max_input_length"], padding=True,
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=self.cfg["gen_max_tokens"],
                    num_beams=min(2, self.cfg["gen_num_beams"]),
                    do_sample=False,
                )
            pred = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

            inp_short = input_text[:150] + ("…" if len(input_text) > 150 else "")
            gold_short = gold[:150] + ("…" if len(gold) > 150 else "")
            pred_short = pred[:150] + ("…" if len(pred) > 150 else "")

            mark = "✓" if pred.strip() != gold.strip() else "≈"
            log.info(f"  [{i+1}] Input: {inp_short}")
            log.info(f"       Gold:  {gold_short}")
            log.info(f"       Pred:  {pred_short}  {mark}")

        log.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# Train Filler
# ═══════════════════════════════════════════════════════════════════════════════

def train_filler(
    model_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    hub_token: str | None = None,
):
    """
    Train an Encoder-Decoder (Seq2Seq) or Encoder (MLM) filler.
    """
    log.info(f"Preparing filler training for: {model_name}")
    model, tokenizer = build_filler(model_name)
    cfg = FILLER_REGISTRY[model_name]
    
    mtype = cfg.get("type", "seq2seq")
    output_dir = os.path.join(OUTPUT_DIR, "checkpoints", f"filler_{model_name}")

    if mtype == "seq2seq":
        def seq2seq_preprocess(examples):
            inputs = [f"Fill PII: {text}" for text in examples["masked_text"]]
            targets = examples["source_text"]
            
            model_inputs = tokenizer(
                inputs, max_length=cfg["max_source_length"], truncation=True
            )
            labels = tokenizer(
                targets, max_length=cfg["max_target_length"], truncation=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_dataset = train_ds.map(seq2seq_preprocess, batched=True, remove_columns=train_ds.column_names)
        val_dataset = val_ds.map(seq2seq_preprocess, batched=True, remove_columns=val_ds.column_names)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100
        )

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=cfg["epochs"],
            predict_with_generate=True,
            bf16=BF16_OK,
            fp16=FP16_OK,
            logging_steps=50,
            report_to="none",
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_token=hub_token,
        )
        trainer_cls = Seq2SeqTrainer
        
    elif mtype == "mlm":
        def mlm_preprocess(examples):
            return tokenizer(
                examples["source_text"], 
                truncation=True, 
                max_length=128
            )
            
        train_dataset = train_ds.map(mlm_preprocess, batched=True, remove_columns=train_ds.column_names)
        val_dataset = val_ds.map(mlm_preprocess, batched=True, remove_columns=val_ds.column_names)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=cfg["epochs"],
            bf16=BF16_OK,
            fp16=FP16_OK,
            logging_steps=50,
            report_to="none",
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_token=hub_token,
        )
        trainer_cls = Trainer
        
    resume_from_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            resume_from_checkpoint = True
            log.info(f"  Found existing checkpoint in {output_dir}, enabling resume!")

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    log.info(f"  Starting training ...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"  Model saved to: {output_dir}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Filler Inference — fill masked text
# ═══════════════════════════════════════════════════════════════════════════════

def run_filler(
    masked_text: str,
    model, 
    tokenizer,
    cfg: dict,
) -> str:
    """
    Run the filler model to anonymize text.
    Handles both Seq2Seq and MLM inferences.
    """
    mtype = cfg.get("type", "seq2seq")

    if mtype == "seq2seq":
        prompt = f"Fill PII: {masked_text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.get("generation_max_length", 128),
                num_beams=4,
                do_sample=False,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
    elif mtype == "mlm":
        from config import ENTITY_TYPES
        
        filled_text = masked_text
        for etype in ENTITY_TYPES:
            tag = f"[{etype}]"
            if tag in filled_text:
                mlm_masks = f"{tokenizer.mask_token} {tokenizer.mask_token}"
                filled_text = filled_text.replace(tag, mlm_masks)
                
        inputs = tokenizer(filled_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        token_logits = outputs.logits[0]
        mask_token_index = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        
        predicted_ids = inputs.input_ids[0].clone()
        for idx in mask_token_index:
            predicted_ids[idx] = token_logits[idx].argmax()
            
        return tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Load Pre-Trained Filler
# ═══════════════════════════════════════════════════════════════════════════════

def _load_trained_filler(model_name: str, output_dir: str) -> Tuple:
    """Load a previously trained filler from disk."""
    cfg = FILLER_REGISTRY[model_name]
    hf_name = cfg["hf_name"]

    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    if cfg.get("use_qlora"):
        from peft import PeftModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16_OK else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForSeq2SeqLM.from_pretrained(
            hf_name, quantization_config=bnb_config, device_map="auto",
        )
        model = PeftModel.from_pretrained(base, output_dir)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            output_dir, device_map="auto",
        )

    model.eval()
    return model, tokenizer, None


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _find_checkpoint(output_dir: str) -> Optional[str]:
    """Find latest checkpoint for resume."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir)
             if os.path.isdir(os.path.join(output_dir, d))
             and re.match(r"checkpoint-\d+$", d)]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    ckpt_path = os.path.join(output_dir, ckpts[-1])
    log.info(f"  Resuming from checkpoint: {ckpt_path}")
    return ckpt_path
