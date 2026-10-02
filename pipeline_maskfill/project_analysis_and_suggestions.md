    # PII Anonymization — Project Analysis & Suggestions for the Two-Level Pipeline Task

    ## Project Understanding Summary

    Your project explores **two fundamentally different neural approaches** to privacy-preserving text anonymization:

    1. **Direct Seq2Seq Rewriting** (`Seq2Seq_model/`) — Already completed. Best result: `bart-base` + PII-Aware Loss → **0.98% entity leak, 94.92 BERTScore F1**.
    2. **Pipeline Mask-and-Fill** (`pipeline_maskfill/`) — Four increasingly sophisticated models (Baseline → DP-Guided → Rephraser → Semantic Paraphraser). Trained on a custom ~36K gold + ~120K augmented dataset from the SAHA-AL annotation pipeline.

    ### Your New Task (Two-Level Pipeline on ai4privacy 500K)

    You now want to build a **cleaner, more controlled two-stage pipeline** using the full **AI4Privacy 500K dataset**:

    | Stage | Role | Models |
    |-------|------|--------|
    | **Encoder (Masker/Censor)** | NER — detect & mask PII | DistilRoBERTa, RoBERTa, DeBERTa-v3 |
    | **Encoder-Decoder (Filler)** | Generate fake replacements for masked spans | BART-BASE (+ optionally Flan-T5) |

    **Data Split:** 80/10/10 (train/val/test), language-stratified. Train set further split in half: Half-A for encoders, Half-B (reversed) for encoder-decoders. This enforces a strict **privacy wall**.

    **Evaluation:** All `num_encoders × num_encoder_decoders` combos evaluated on Entity Leak Rate, BERTScore, BLEU, ROUGE, etc.

    ---

    ## Suggestions and Recommendations

    ### 1. Model Choices — Refinements

    #### Encoders (NER / Masking)
    Your choices (DistilRoBERTa, RoBERTa, DeBERTa-v3) are solid. A few thoughts:

    - **DeBERTa-v3-base** will almost certainly be your best NER model — it consistently dominates token classification benchmarks. Your earlier pipeline already uses it successfully.
    - **RoBERTa-base** is a strong mid-tier choice.
    - **DistilRoBERTa** gives you a useful small-model data point for the analysis (speed vs. accuracy trade-off).

    > [!TIP]
    > Consider adding **XLM-RoBERTa-base** if you want cross-lingual results. The ai4privacy dataset is multilingual, and XLM-R would let you report multilingual NER performance — a strong differentiator for the final report.

    #### Encoder-Decoders (Filling / Unmasking)
    - **BART-BASE** is the right primary choice — it was your best Seq2Seq model.
    - **Flan-T5-base** is the right second choice. Yes, it doubles your combos from 3 to 6, but 6 combos is still very manageable and the comparison is extremely valuable.

    > [!IMPORTANT]
    > **Use Flan-T5-base, not Flan-T5-small.** The fair comparison with BART-BASE requires similar model sizes (both ~140M params). Flan-T5-small (77M) would be unfairly disadvantaged. If you're worried about compute, just use QLoRA for Flan-T5.

    ### 2. Training Strategy Improvements

    #### Half-A / Half-B Split — Critical Refinement
    Your plan to split the 80% train set into two halves is correct for the privacy wall. However:

    > [!WARNING]
    > **Do NOT reverse Half-B for the encoder-decoder.** The encoder-decoder needs **(masked_text → anonymized_text)** pairs. "Reversing" makes no sense here since the task is generating fake fills, not reconstructing originals. Instead:
    > - **Half-A:** Train encoder NER models on [(source_text, entity_labels)](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill/src/eval_all.py#657-660) pairs.
    > - **Half-B:** Create [(masked_text → fake-filled_text)](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill/src/eval_all.py#657-660) pairs using the entity annotations in the data, then train the encoder-decoder. The key is that the encoder-decoder never sees the original PII from Half-A.

    #### Encoder-Decoder Input Format
    For the filling model, use the format already proven in your pipeline:
    ```
    Input:  "Replace PII placeholders: Hello [PERSON] from [LOC], call [PHONE]"
    Target: "Hello James Wilson from Portland, call 555-0193"
    ```
    The entity type tags (`[PERSON]`, `[LOC]`) are critical — they tell the filler *what kind* of fake to generate.

    #### Validation Split Strategy
    Your 5-5 split of the 10% val is fine. Make sure:
    - **Encoder val (5%):** Used for NER F1 early stopping.
    - **Encoder-decoder val (5%):** Used for seq2seq loss / BERTScore early stopping.

    ### 3. Metrics — What to Run

    #### Core Metrics (Must Have)
    | Metric | What it Measures | Direction |
    |--------|-----------------|-----------|
    | **Entity Leak Rate** | % of original PII appearing verbatim in output | ↓ |
    | **BERTScore F1** | Semantic similarity (tolerates valid PII alternatives) | ↑ |
    | **BLEU** | N-gram overlap with reference | ↑ |
    | **ROUGE-L** | Longest common subsequence overlap | ↑ |
    | **NER F1 (per-entity)** | Encoder's detection quality | ↑ |

    #### Advanced Metrics (Strongly Recommended)
    | Metric | What it Measures | Why It's Important |
    |--------|-----------------|-------------------|
    | **CRR (Contextual Re-identification Risk)** | Whether surrounding n-grams survive intact | Key for showing the *pipeline* leaks contextual clues even after entity replacement |
    | **Per-Entity-Type Leak Rate** | Which entity types leak most | Shows where the NER bottleneck is (names vs. emails vs. dates) |
    | **Difficulty-Stratified Leakage** | Leak rate by text length (short/medium/long) | Reveals robustness differences |
    | **Perplexity (PPL) of output** | How "natural" the generated text sounds | Use a separate LM (e.g., GPT-2) to score. Important for filler quality |
    | **Entity Type Accuracy** | Does the filler produce the *right kind* of entity? | E.g., replacing `[PERSON]` with a name, not a phone number |

    #### Novel Metrics (Would Set Your Work Apart)
    | Metric | Description |
    |--------|-------------|
    | **NER Recall at Pipeline Level** | Measure what fraction of ground-truth PII spans the encoder catches on the *test set* — this is the fundamental bottleneck for pipeline approaches |
    | **Downstream Task Utility** | Train a sentiment classifier on the anonymized output, compare accuracy to training on original data. This directly proves the anonymized data is still *useful* |
    | **Entity Consistency Score** | Same entity → same replacement across a document? Important for coherence |

    ### 4. Ideas & Innovations (Beyond Baseline)

    These can be added *after* the baseline results are in:

    #### A. Cascaded NER Ensemble (Low Effort, High Impact)
    Instead of a single encoder, use a **union of predictions** from all 3 encoders. If *any* encoder detects a span as PII, mask it. This maximizes recall at the cost of some precision — but for privacy, recall is far more important than precision.

    ```
    Final_mask = DistilRoBERTa_mask ∪ RoBERTa_mask ∪ DeBERTa_mask
    ```

    This gives you a 7th "ensemble" row in your results table with likely the lowest leak rate.

    #### B. Confidence-Weighted Masking
    Instead of binary mask/don't-mask, use the NER model's confidence:
    - **High confidence (>0.9):** Mask with entity type tag `[PERSON]`
    - **Medium confidence (0.5–0.9):** Mask with generic `[PII]` tag
    - **Low confidence (<0.5):** Leave unchanged

    This lets the filler handle uncertain spans more carefully.

    #### C. Two-Pass Filling
    1. First pass: Fill with synthetic entities using the encoder-decoder.
    2. Second pass: Run the same NER encoder on the *output* to check if any original PII leaked. If detected, re-mask and re-fill.

    This is a cheap post-processing step that can significantly reduce leak rates.

    #### D. PII-Aware Loss for the Filler (Port from Seq2Seq)
    You already have a proven PII-Aware Loss from the Seq2Seq work. Port the anti-leakage penalty to the encoder-decoder filler:

    ```
    L_filler = α·L_CE + β·L_anti-leak
    ```

    This penalises the filler if it assigns high probability to any token from the original PII in Half-A (which it shouldn't even have seen, but the base model may have memorized from pre-training).

    #### E. Cross-Paradigm Comparison Table
    Your most powerful contribution is the **direct comparison**: same dataset, same test set, but:
    - Seq2Seq (bart-base, 0.98% leak) vs.
    - Pipeline (best encoder + best filler, ?% leak)

    This answers RQ2 from your report definitively.

    ### 5. Practical Suggestions

    #### Compute Efficiency
    - Use **QLoRA** for both BART-BASE and Flan-T5-base fillers. Your pipeline already does this successfully.
    - NER models (encoder-only) are much cheaper — full fine-tuning is fine for DistilRoBERTa and even RoBERTa-base.
    - DeBERTa-v3-base may need gradient checkpointing on a 4GB GPU, but fits easily on Kaggle T4/P100.

    #### Language Stratification
    Your [language_stratified_split()](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill/src/common.py#306-331) in [common.py](file:///home/ayush/Desktop/sem6/inlp/project/pii_identification_and_replacement/pipeline_maskfill/src/common.py) (lines 306–330) already implements this. Reuse it, but extend it for the 80/10/10 split with separate encoder/decoder halves.

    #### Checkpointing
    Follow the same fault-tolerant checkpointing pattern from `pipeline_maskfill` — save every 500 steps, `save_total_limit=2`, auto-restore from Kaggle input mounts.

    ### 6. Summary of Recommended Final Setup

    | Component | Choice | Justification |
    |-----------|--------|---------------|
    | **Encoders** | DistilRoBERTa, RoBERTa-base, DeBERTa-v3-base | Varying sizes, SOTA |
    | **Fillers** | BART-BASE, Flan-T5-base (QLoRA) | Best two from Seq2Seq experiments |
    | **Combos** | 3 × 2 = **6 pipelines** + 1 ensemble | Manageable, clean comparison |
    | **Core Metrics** | Entity Leak Rate, BERTScore F1, BLEU, ROUGE-L, NER F1 | Standard + proven |
    | **Advanced Metrics** | CRR, Per-Entity Leak, PPL, Downstream Utility | Differentiators |
    | **Innovation (Post-Baseline)** | NER Ensemble, Two-Pass Filling, PII-Aware Filler Loss | Each adds a new row/column to analysis |

    ---

    > [!IMPORTANT]
    > **Key Strategic Point:** If the pipeline results are *worse* than your 0.98% Seq2Seq leak rate, that's actually a valid finding — it justifies your hypothesis from the task notes that creating a new dataset and using a single encoder-decoder is better. If they're *better*, you've found something interesting about the power of modular architectures. Either way, the comparison is the contribution.
