# Fine-Tuning roberta + deberta-mlm on the TAB Dataset

## Background & Current State

Your pipeline has two stages trained **exclusively on AI4Privacy synthetic data**:

| Component | Model | Trained On | Current TAB Performance |
|-----------|-------|-----------|------------------------|
| **Masker** (NER) | `roberta-base` fine-tuned for token classification | AI4Privacy 500k (21 BIO labels) | Masker Recall = **0.776** |
| **Filler** (MLM) | `deberta-v3-base` fine-tuned for masked LM | AI4Privacy 500k | BERTScore F1 = **93.23**, Word Acc = **32.08%** |

On the AI4Privacy test set (in-domain), this combo achieves **1.33% leakage**, **89.76 BERTScore F1**. On TAB (out-of-domain ECHR legal text), masker recall drops to 0.776 and BLEU/ROUGE collapse to 0.000 due to the massive domain gap.

---

## The Core Problem: Label Taxonomy Mismatch

This is the **single most critical issue** and the first thing to decide.

| AI4Privacy (your current labels) | TAB Labels |
|----------------------------------|------------|
| 21 entity types: `FIRST_NAME`, `LAST_NAME`, `FULLNAME`, `EMAIL`, `PHONE`, `SSN`, `PASSPORT`, `CREDIT_CARD`, `ZIPCODE`, `ADDRESS`, `ID_NUMBER`, `ACCOUNT_NUM`, `DATE`, `TIME`, `LOCATION`, `ORGANIZATION`, `TITLE`, `GENDER`, `NUMBER`, `OTHER_PII`, `UNKNOWN` | 8 entity types: `PERSON`, `CODE`, `LOC`, `ORG`, `DEM`, `DATETIME`, `QUANTITY`, `MISC` |

> [!IMPORTANT]
> **You cannot naively fine-tune the RoBERTa masker on TAB labels.** The model currently produces 43 BIO labels (21×2 + O). If you retrain on 8 TAB types, the model will forget ALL the fine-grained AI4Privacy labels (catastrophic forgetting), and the filler stage — which depends on seeing `[FIRST_NAME]`, `[LOCATION]`, etc. — will break entirely.

### Recommended Solution: Map TAB → AI4Privacy labels

We convert TAB's 8 coarse entity types into your existing 21 fine-grained BIO labels:

```
TAB PERSON   → FULLNAME        (can heuristically split into FIRST_NAME/LAST_NAME)
TAB LOC      → LOCATION
TAB ORG      → ORGANIZATION
TAB DATETIME → DATE            (or TIME if the span is a time expression)
TAB CODE     → ID_NUMBER
TAB QUANTITY → NUMBER
TAB DEM      → OTHER_PII       (demographic attributes like "Turkish", "male")
TAB MISC     → OTHER_PII
```

This way the masker keeps its full 43-label vocabulary and just learns to better detect entities in legal text.

---

## Proposed Fine-Tuning Strategy

### Phase 1: Fine-Tune the Masker (NER) on TAB

**What**: Continue training `Xyren2005/pii-ner-roberta` (your already AI4Privacy-trained model) on TAB data.

**How the data is prepared**:
1. Load TAB from HuggingFace: `ildpil/text-anonymization-benchmark`
2. For each document, extract text + entity_mentions (character-level spans with `identifier_type` ∈ {DIRECT, QUASI})
3. Split documents into paragraphs (same as [evaluate_tab.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/testing_approach2/evaluate_tab.py) does)
4. Convert character-level spans → word-level BIO tags using the mapping above
5. Use TAB's official train/dev/test split (1014/127/127 docs)

**Training Config** (continue from your checkpoint, NOT from scratch):
- **Base model**: `Xyren2005/pii-ner-roberta` (your HF checkpoint)
- **Learning rate**: `1e-5` (lower than original `2e-5` to avoid overwriting AI4Privacy knowledge)
- **Epochs**: 5-8 with early stopping (patience=2)
- **Batch size**: 8 (TAB paragraphs are longer than AI4Privacy sentences)
- **Max length**: 512 (legal paragraphs are much longer; 192 is too short)
- **Gradient accumulation**: 4 (effective batch = 32)
- **Weight decay**: 0.01
- **Warmup**: 10% of steps

**Expected improvement**: Masker recall should jump from **0.776 → 0.90+**, particularly for DATETIME (0.816→0.95+), ORG (0.682→0.85+), CODE (0.622→0.80+), and DEM/MISC (currently near 0).

---

### Phase 2: Fine-Tune the Filler (MLM) on TAB

**What**: Continue training `Xyren2005/pii-ner-filler_deberta-filler` on unmasked ECHR legal text using standard MLM.

**Why**: The DeBERTa MLM filler currently produces replacements based on AI4Privacy casual/synthetic language patterns. Exposure to legal text will teach it "legalese" vocabulary (e.g., "applicant", "respondent State", "Article 3 of the Convention").

**How**:
1. Take all TAB training documents (full unmasked text)
2. Chunk into 256-token segments
3. Continue MLM training with 15% random masking (standard DataCollatorForLanguageModeling)
4. Lower LR = `1e-5`, 3-5 epochs

**Expected improvement**: The filler will produce more natural legal-domain replacements instead of casual names/locations.

---

### Phase 3: Re-evaluate on TAB Test Set

Use the **existing** [evaluate_tab.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/testing_approach2/evaluate_tab.py) script — no changes needed. Simply point it at the fine-tuned model checkpoints.

---

## Key Considerations & Potential Problems

### 1. TAB's PERSON spans include titles
TAB annotates "Mr Benham" as a single PERSON span. Your masker was trained to see "Mr" as a TITLE and "Benham" as LAST_NAME. The label mapping script will map the entire "Mr Benham" span to FULLNAME, which is correct and your masker already supports FULLNAME via B-FULLNAME/I-FULLNAME tags.

### 2. Coreference Consistency
TAB documents mention the same entity 10-15 times. Even after fine-tuning, the filler will treat each `[MASK]` independently. This is a **post-processing problem**, not a fine-tuning problem. A simple entity-memory dictionary (first occurrence → reuse that fake name) would solve it, but is out of scope for this fine-tuning phase.

### 3. Data Size
TAB train split has **1,014 documents** with ~1,442 tokens avg = ~300-500 paragraphs with DIRECT/QUASI entities. This is small compared to AI4Privacy's 500k, but for _continued fine-tuning_ (not from scratch), this is actually sufficient — legal NER papers show significant gains with even 500-1000 annotated examples when starting from a pre-trained checkpoint.

### 4. Mixed Training (Optional but Recommended)
To further guard against catastrophic forgetting, we can mix a small portion of AI4Privacy data (e.g., 10-20%) into each training batch. This is analogous to "replay" in continual learning.

---

## Proposed Changes

### TAB Fine-Tuning Scripts

#### [NEW] [tab_finetune_masker.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/tab-finetuning/tab_finetune_masker.py)
- Loads TAB dataset from HuggingFace
- Converts TAB entity annotations → BIO labels using AI4Privacy label vocabulary
- Paragraph segmentation with character→word offset alignment
- Loads pre-trained `Xyren2005/pii-ner-roberta` and continues training
- Saves fine-tuned checkpoint locally and optionally pushes to HF Hub

#### [NEW] [tab_finetune_filler.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/tab-finetuning/tab_finetune_filler.py)
- Loads TAB training documents (full text)
- Chunks into 256-token segments
- Loads pre-trained `Xyren2005/pii-ner-filler_deberta-filler` and continues MLM training
- Saves fine-tuned checkpoint

#### [NEW] [tab_evaluate_finetuned.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/tab-finetuning/tab_evaluate_finetuned.py)
- Identical logic to existing [evaluate_tab.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/testing_approach2/evaluate_tab.py) but points to fine-tuned checkpoints
- Generates a comparable report for before/after comparison

---

## Verification Plan

### Automated Tests
1. **Data pipeline sanity check**: Run the label-mapping function on 5 TAB documents and print the BIO tags — manually verify they align with the gold entity spans
   ```bash
   python tab-finetuning/tab_finetune_masker.py --dry-run --limit 5
   ```

2. **Fine-tuning convergence**: Monitor training loss — it should decrease smoothly without spiking (which would indicate catastrophic forgetting)

3. **TAB evaluation comparison**: Run the evaluation script on both the original and fine-tuned models:
   ```bash
   # Original (already done, baseline in tab_evaluation_results_readable.txt)
   # Fine-tuned
   python tab-finetuning/tab_evaluate_finetuned.py
   ```
   Expected: Masker recall ↑ from 0.776 to 0.90+, Word Accuracy ↑ from 32% to 50%+

### Manual Verification
- After fine-tuning, run the masker on 3-5 TAB test paragraphs and visually inspect:
  - Are PERSON names correctly detected?
  - Are DATETIME spans fully captured (not just the year)?
  - Are ORG names (e.g. "European Court of Human Rights") properly masked?
- **User action needed**: Please confirm whether you will run this on Kaggle (with GPU) or locally, so we can adjust batch sizes accordingly.
