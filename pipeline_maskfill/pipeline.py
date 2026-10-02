# ==============================================================================
# pipeline.py — Combined Two-Stage Inference Pipeline
# ==============================================================================
# Stage 1: NER Encoder detects and masks PII
# Stage 2: Filler Encoder-Decoder generates realistic replacements
# ==============================================================================

import logging
from typing import List, Tuple
from tqdm.auto import tqdm

from encoder import run_ner
from filler import run_filler

log = logging.getLogger("pipeline")


def anonymize(
    text: str,
    encoder_model, encoder_tokenizer,
    filler_model, filler_tokenizer,
    filler_cfg: dict = None,
) -> str:
    """
    Full anonymization pipeline:
      1. NER detects PII → masked text with [TYPE] placeholders
      2. Filler generates realistic fake replacements

    Args:
        text: Original text containing PII
        encoder_model/tokenizer: Trained NER encoder
        filler_model/tokenizer: Trained filler encoder-decoder
        filler_cfg: Filler config dict (for prefix, gen params)

    Returns:
        Anonymized text with PII replaced by realistic fakes
    """
    # Stage 1: Detect & Mask
    masked_text, entities = run_ner(text, encoder_model, encoder_tokenizer)

    # If no entities detected, return original text unchanged
    if not entities:
        return text

    # Stage 2: Fill
    anonymized = run_filler(masked_text, filler_model, filler_tokenizer, filler_cfg)

    return anonymized


def batch_anonymize(
    texts: List[str],
    encoder_model, encoder_tokenizer,
    filler_model, filler_tokenizer,
    filler_cfg: dict = None,
    desc: str = "Anonymizing",
) -> Tuple[List[str], List[str], List[list]]:
    """
    Anonymize a batch of texts with progress bar.

    Returns:
        (anonymized_texts, masked_texts, all_entities)
    """
    anonymized, masked_texts, all_entities = [], [], []

    for text in tqdm(texts, desc=desc):
        masked, entities = run_ner(text, encoder_model, encoder_tokenizer)
        masked_texts.append(masked)
        all_entities.append(entities)

        if entities:
            anon = run_filler(masked, filler_model, filler_tokenizer, filler_cfg)
        else:
            anon = text  # no PII detected, return as-is

        anonymized.append(anon)

    return anonymized, masked_texts, all_entities
