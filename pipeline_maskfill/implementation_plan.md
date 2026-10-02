# Two-Level PII Pipeline — Implementation Plan

> Working directory: `/home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/`

## Goal

Build a **two-stage Encoder + Encoder-Decoder pipeline** for PII anonymization trained on the AI4Privacy 500K dataset. The pipeline uses:
- **Stage 1 (Encoder/NER):** Detects & masks PII entities in text
- **Stage 2 (Encoder-Decoder/Filler):** Replaces masked spans with natural, coherent fake entities

All models configurable via CLI flags. Comprehensive logging, intermediate inference samples, and evaluation built-in.

---

## The Faker Problem — Resolved

> [!IMPORTANT]
> **We do NOT use Faker-generated targets for training the Filler.** Faker produces culturally/contextually mismatched replacements ("James Lothbrok from Delhi") that destroy utility. Instead:
>
> - **Half-A** trains the **NER Encoder** on `(source_text, entity_labels)` pairs
> - **Half-B** trains the **Filler** on `(masked_text → real_text_from_half_B)` pairs
>
> The model learns from **real language patterns** — co-occurring names, locations, dates — producing natural fills like "Aditya Gaur from Shimla" instead of Faker's "James Lothbrok from Delhi."
>
> **Privacy wall intact:** NER only knows Half-A entities. Filler only knows Half-B entities. At inference on test data, the filler generates **new** entities from neither half.
>
> AB testing (Faker targets vs. real-text targets) can be added later to quantify the utility difference.

---

## Encoder (NER) Model Exploration

### Candidates

| Model | Params | Architecture | Strengths | Weaknesses | Best For |
|-------|--------|-------------|-----------|------------|----------|
| **`distilroberta-base`** | 82M | 6-layer distilled RoBERTa | Fast, lightweight, good baseline | Lower NER recall on rare entities | Speed baseline, resource-constrained |
| **`roberta-base`** | 125M | 12-layer RoBERTa, dynamic masking | Strong contextual understanding, proven NER backbone | No cross-attention innovations | Solid mid-tier comparison |
| **`microsoft/deberta-v3-base`** | 184M | Disentangled attention + enhanced mask decoder | **SOTA on token classification**, best sub-word handling | Slower, AMP quirks (gamma/beta naming), higher VRAM | Best expected NER performance |

### Why These Three?

- **Size gradient:** 82M → 125M → 184M lets you plot **NER F1 vs model size** — a clean analysis for the report
- **Architecture diversity:** Distillation (DistilRoBERTa) vs standard (RoBERTa) vs innovations (DeBERTa disentangled attention) — each represents a different design philosophy
- **DeBERTa will likely win** because its disentangled attention separately encodes content and position — critical for NER where *position within an entity span* matters (B- vs I- tags)

### What Could Give Better Encoder Results?

1. **Recall-weighted focal loss** instead of standard CE — rare entity types (PASSPORT, IBAN) get missed by standard training. Focal loss `(1−p)^γ · CE` with `γ=2.0` and per-class weights inversely proportional to frequency forces the model to pay attention to rare PII types
2. **Lowered classification threshold** — Default argmax picks the highest-probability label. Instead, at inference: if `max(non-O_probs) > 0.3`, classify as that entity type. This trades precision for recall — and **recall is what matters for privacy** (missed PII = leaked PII)
3. **NER Ensemble (post-baseline)** — Union of predictions from all 3 encoders. If *any* model flags a span, it gets masked. Maximizes recall at cost of some over-masking

---

## Encoder-Decoder (Filler) Model Exploration

### Candidates

| Model | Params | Architecture | Strengths | Weaknesses | Best For |
|-------|--------|-------------|-----------|------------|----------|
| **`facebook/bart-base`** | 139M | Denoising autoencoder, bidirectional encoder + autoregressive decoder | **Best in your Seq2Seq experiments** (0.98% leak rate), natural text generation | No instruction tuning, needs careful prefix design | Primary filler — proven winner |
| **`google/flan-t5-base`** | 248M | T5 with instruction tuning on 1800+ tasks | Follows instructions well, strong zero-shot | Larger, P100 bf16 issues (needs fp32 LoRA workaround) | Instruction-following filler |

### Why BART-BASE is the Primary Choice?

From your Seq2Seq experiments, BART-BASE achieved:
- **0.98% entity leak rate** (best of all 5 models)
- **94.92 BERTScore F1** (near-identical to baseline)
- No hallucination/repetition issues (unlike DistilBART)

It's pre-trained as a **denoising autoencoder** — literally trained to reconstruct corrupted text — which is exactly what our filler does (fill in `[PERSON]` slots).

### Why Flan-T5-base as Second?

- Instruction-tuned: responds well to prompts like "Replace PII placeholders with realistic fake entities"
- Different architecture family: validates findings aren't BART-specific
- But needs QLoRA (4-bit NF4) to fit in VRAM — while BART-BASE can be full fine-tuned

### What Could Give Better Filler Results?

1. **Entity-type-aware prefix** — Instead of generic "Replace PII placeholders:", use: `"Fill [PERSON], [LOC], [PHONE] placeholders: ..."`. This tells the decoder exactly which types to generate
2. **Constrained decoding** — At generation time, if the model is filling a `[PHONE]` slot, constrain the output distribution to digit-heavy tokens. Not for training, just for inference
3. **Multi-turn filling (post-baseline)** — Fill one entity type at a time: first all `[PERSON]`, then `[LOC]`, then `[PHONE]`. Each pass has a simpler task. Slower but potentially more accurate
4. **Temperature tuning** — Higher temperature (0.7–0.9) for entity diversity vs. greedy (0.0) for consistency. Can be exposed as a CLI flag

---

## Proposed Changes

### Project Structure

#### [NEW] [config.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/config.py)

Central configuration — all hyperparameters, model registries, paths. Contains:
- `ENCODER_REGISTRY` — dict mapping model names to HuggingFace IDs + per-model hyperparams
- `FILLER_REGISTRY` — dict mapping model names to HuggingFace IDs + per-model hyperparams  
- Data split ratios, training params, generation params
- Logging config

#### [NEW] [data.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/data.py)

Data loading & splitting:
- Load AI4Privacy 500K from HuggingFace
- Language-stratified 80/10/10 split (reuse logic from existing `common.py`)
- Train set → Half-A / Half-B split
- Val set → 5% encoder val / 5% filler val
- BIO label construction, entity type mapping
- `prepare_ner_data()` — tokenize & align for NER
- `prepare_filler_data()` — create `(masked_text → real_text)` pairs from Half-B

#### [NEW] [encoder.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/encoder.py)

NER encoder training & inference:
- `build_encoder(model_name)` — loads any registered encoder
- `train_encoder(model, ...)` — trains with:
  - Validation after every epoch
  - Training loss logging every 50 steps
  - **3 sample inferences after each epoch** (input → predicted entities vs gold)
  - NER F1 early stopping
  - DeBERTa `fix_deberta_params()` handling
- `run_ner_inference(text, model, tokenizer)` — detect entities, return masked text
- Per-entity-type metrics logged to file

#### [NEW] [filler.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/filler.py)

Encoder-decoder filler training & inference:
- `build_filler(model_name)` — loads any registered filler (with QLoRA for Flan-T5)
- `train_filler(model, ...)` — trains with:
  - Validation after every epoch
  - Training loss logging every 50 steps
  - **5 sample inferences after each epoch** (masked_input → generated_output vs gold)
  - BERTScore on val set as checkpoint metric
- `run_filler_inference(masked_text, model, tokenizer)` — generate anonymized text

#### [NEW] [pipeline.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/pipeline.py)

Combined two-stage inference:
- `anonymize(text, encoder, filler)` — NER → mask → fill
- Batch inference for evaluation

#### [NEW] [evaluate.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/evaluate.py)

Evaluation framework:
- **Privacy metrics:** Entity Leak Rate, Token Leak Rate, CRR (2/3/4-gram), Per-Entity-Type Leak
- **Utility metrics:** BERTScore F1, BLEU, ROUGE-1/2/L
- **NER-specific:** Entity-level Precision, Recall, F1 (via `seqeval`)
- **Filler-specific:** Perplexity, Entity Type Accuracy
- **Robustness:** Difficulty-stratified leak rate (short/medium/long texts)
- Comparison table output (console + `.txt` + `.json`)
- 37 curated edge-case examples (reuse from existing pipeline)
- Pareto front plot (Privacy vs Utility)

#### [NEW] [run.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/run.py)

**CLI entry point** — everything controlled via flags:

```bash
# Train a specific encoder
python run.py train-encoder --model distilroberta

# Train a specific filler
python run.py train-filler --model bart-base

# Run full pipeline evaluation for a combo
python run.py evaluate --encoder distilroberta --filler bart-base

# Train all encoders
python run.py train-encoder --model all

# Quick mode (small data subset for debugging)
python run.py train-encoder --model distilroberta --quick

# Evaluate all trained combos
python run.py evaluate --all
```

Implemented with `argparse` subcommands.

#### [NEW] [requirements.txt](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta/requirements.txt)

Pinned dependencies.

---

## Logging Strategy

### Training Logs (every run creates a timestamped log file)

```
outputs/
  logs/
    distilroberta_encoder_20260405_143022.log
    bart-base_filler_20260405_150512.log
```

### What Gets Logged

| Event | Frequency | Content |
|-------|-----------|---------|
| **Config dump** | Start of run | All hyperparameters, model name, data sizes |
| **Training loss** | Every 50 steps | `Step 150/3000 | loss=0.342 | lr=2.8e-5` |
| **Epoch summary** | Every epoch | Train loss, val loss, val F1 / val BERTScore |
| **Sample inferences** | Every epoch | 3–5 input→output examples, annotated with correctness |
| **NER classification report** | End of training | Per-entity P/R/F1 table |
| **Checkpoint saved** | Every epoch | Path, file size |
| **Evaluation results** | End of evaluation | Full metrics table |

### Sample Inference Logging (Encoder — after each epoch)

```
═══ Epoch 2/5 — Sample NER Inferences ═══
[1] Input:  "Contact Ayush Sheta at ayush@email.com about the project."
    Gold:   [B-PERSON I-PERSON] O [B-EMAIL] O O O
    Pred:   [B-PERSON I-PERSON] O [B-EMAIL] O O O  ✓
[2] Input:  "Meeting at 742 Oak Street, Mumbai on 15/03/2025."
    Gold:   O O [B-ADDRESS I-ADDRESS I-ADDRESS] O [B-DATE]
    Pred:   O O [B-ADDRESS I-ADDRESS] O O [B-DATE]  ✗ (missed I-ADDRESS)
[3] Input:  "The weather is nice today."
    Gold:   O O O O O
    Pred:   O O O O O  ✓ (no false positives)
```

### Sample Inference Logging (Filler — after each epoch)

```
═══ Epoch 1/3 — Sample Filler Inferences ═══
[1] Input:  "Replace PII: [PERSON] from [LOC] called at [PHONE]."
    Gold:   "Priya Sharma from Bangalore called at +91 98765 43210."
    Pred:   "Rahul Verma from Chennai called at +91 99887 12345."  ✓ (different but valid)
[2] Input:  "Replace PII: Dear [PERSON], your account [ACCOUNT] has been updated."
    Gold:   "Dear Amit Kumar, your account 4829103756 has been updated."
    Pred:   "Dear Sunita Patel, your account 7623891045 has been updated."  ✓
```

---

## Evaluation Metrics — Detailed Breakdown

### Privacy Metrics (↓ = better)

| Metric | What It Measures | How It's Computed |
|--------|-----------------|-------------------|
| **Entity Leak Rate** | % of original PII appearing verbatim in output | Case-insensitive substring match of capitalised entities |
| **Token Leak Rate** | % of capitalised tokens that survive | Per-token check |
| **CRR (2/3/4-gram)** | Contextual re-identification risk | % of identifying n-grams from original that survive in output |
| **Per-Entity-Type Leak** | Which PII types leak most | Regex-based entity detection + comparison |

### Utility Metrics (↑ = better)

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **BERTScore F1** | Semantic similarity (contextual embeddings) | **Primary utility metric** — tolerates valid alternative PII replacements |
| **BLEU** | N-gram precision with brevity penalty | Standard MT metric, sensitive to exact wording |
| **ROUGE-L** | Longest common subsequence | Captures structural preservation |

### NER-Specific Metrics (reported separately for encoder)

| Metric | What It Measures |
|--------|-----------------|
| **Entity-level F1** (seqeval) | Strict span-matching — both boundary and type must be correct |
| **Per-entity P/R/F1** | Which entity types the model struggles with |
| **NER Recall** | **Critical for privacy** — missed PII = leaked PII |

### Filler-Specific Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Entity Type Accuracy** | Does `[PERSON]` get filled with a name (not a number)? |
| **Length Ratio** | `len(output) / len(input)` — should be ~1.0 |

### Output Format

Evaluation produces:
- `outputs/eval/results.json` — Machine-readable full metrics
- `outputs/eval/comparison_table.txt` — Human-readable table
- `outputs/eval/pareto_plot.png` — Privacy (leak rate) vs Utility (BERTScore) scatter
- `outputs/eval/per_entity_breakdown.png` — Bar chart of leak rates by entity type
- `outputs/eval/sample_outputs.txt` — 20 example input→output pairs for manual inspection

---

## Ideas to Make the Pipeline Better (Post-Baseline)

### Idea 1: Two-Pass Verification
After the filler produces output, run the NER encoder **again** on the output. If it detects any original PII that leaked through, re-mask and re-fill. This is a cheap post-processing step.

### Idea 2: Encoder Ensemble
Take the union of predictions from all 3 encoders. If any model flags a span as PII, mask it. Maximizes recall — the most important NER metric for privacy.

### Idea 3: PII-Aware Filler Loss
Port the anti-leakage penalty from the Seq2Seq experiments:
`L = α·CE + β·(-log(1 - p(original_token)))` — penalises the filler for copying original PII.

### Idea 4: Faker vs Real-Text AB Test
Train two fillers — one on `(masked → real_text)`, one on `(masked → faker_text)`. Compare BERTScore and naturalness to quantify the Faker problem.

### Idea 5: Confidence-Based Masking
Use NER confidence scores to determine masking aggressiveness. High confidence → mask with type tag. Low confidence → mask with generic `[PII]`.

---

## Verification Plan

### Automated Tests

1. **Quick-mode smoke test**
   ```bash
   cd /home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill_sheta
   python run.py train-encoder --model distilroberta --quick
   python run.py train-filler --model bart-base --quick
   python run.py evaluate --encoder distilroberta --filler bart-base --quick
   ```
   **Expected:** Runs without errors, produces metrics JSON and sample outputs. Quick mode uses ~500 examples.

2. **Data split validation**
   ```bash
   python run.py prepare-data --verify
   ```
   **Expected:** Prints split sizes, confirms all languages represented in each split, confirms Half-A/Half-B are disjoint, confirms no data leakage between train/val/test.

### Manual Verification

1. **Check log files** — After a training run, open the log file in `outputs/logs/` and verify:
   - Config dump is present at the top
   - Training loss is logged every 50 steps
   - Sample inferences appear after each epoch
   - Epoch-level validation metrics are recorded

2. **Inspect sample outputs** — After evaluation, open `outputs/eval/sample_outputs.txt` and manually check 5–10 examples: are the PII replacements natural? Is non-PII text preserved?

3. **Review metric tables** — Compare Entity Leak Rate and BERTScore F1 against the Seq2Seq baseline (0.98% leak, 94.92 BERTScore). The pipeline result may be higher leak (due to NER bottleneck) but should have comparable BERTScore.
