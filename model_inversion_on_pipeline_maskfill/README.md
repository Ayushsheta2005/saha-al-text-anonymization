# Model-Inversion Dataset for Pipeline Mask-Fill Anonymisation

This folder generates two **model-inversion attack** datasets targeting the
pipeline mask-fill anonymisation approach.  Each dataset contains
**40 000 (original sentence, anonymized output)** pairs that can be used to
train a neural inverter that attempts to recover the original PII from the
anonymized text.

---

## Pipeline Combos

| Combo | NER encoder | Filler | Architecture |
|-------|-------------|--------|--------------|
| 1 | [`pii-ner-roberta`](https://huggingface.co/Xyren2005/pii-ner-roberta) | [`pii-ner-filler_deberta-filler`](https://huggingface.co/Xyren2005/pii-ner-filler_deberta-filler) | encoder → encoder (MLM) |
| 2 | [`pii-ner-roberta`](https://huggingface.co/Xyren2005/pii-ner-roberta) | [`pii-ner-filler_bart-base`](https://huggingface.co/Xyren2005/pii-ner-filler_bart-base) | encoder → encoder-decoder (seq2seq) |

### How the pipeline works

**Stage 1 — NER (shared)**
`pii-ner-roberta` (RoBERTa fine-tuned for token classification) labels every
word with a BIO tag:  `O`, `B-PERSON`, `I-PERSON`, `B-DATE`, …

**Stage 2 — Fill (differs per combo)**

*Combo 1 (MLM):*
Each detected entity span is replaced by a single `[MASK]` token.
`pii-ner-filler_deberta-filler` (DeBERTa fine-tuned as a masked-LM) predicts
the top-1 replacement for every `[MASK]` in one forward pass.

*Combo 2 (seq2seq):*
Each entity span is replaced by a `[ENTITY_TYPE]` placeholder
(e.g. `[PERSON]`, `[DATE]`).  The prompt
`"Replace PII placeholders with realistic fake entities: <masked text>"`
is fed to `pii-ner-filler_bart-base` (BART fine-tuned for seq2seq generation)
which decodes the anonymized sentence.

---

## Source Data

[`ai4privacy/pii-masking-400k`](https://huggingface.co/datasets/ai4privacy/pii-masking-400k)
— English split, first 40 000 sentences selected.

---

## Output Datasets on HuggingFace

| Dataset | Combo |
|---------|-------|
| [`JALAPENO11/pipeline-inversion-roberta-deberta`](https://huggingface.co/datasets/JALAPENO11/pipeline-inversion-roberta-deberta) | Combo 1 |
| [`JALAPENO11/pipeline-inversion-roberta-bart`](https://huggingface.co/datasets/JALAPENO11/pipeline-inversion-roberta-bart) | Combo 2 |

Each dataset has columns: `id`, `original`, `anonymized`, `combo`.

---

## Running

```bash
cd model_inversion_on_pipeline_maskfill

# Generate both datasets (resumes automatically if interrupted)
python generate_pairs.py --combo both

# Upload to HuggingFace after generation
python upload_to_hub.py --combo both

# Or do both in one command
python generate_pairs.py --combo both --upload
```

Run only one combo:
```bash
python generate_pairs.py --combo 1   # roberta + DeBERTa MLM
python generate_pairs.py --combo 2   # roberta + BART
```

Progress checkpoints are written to `output/` every example (append-only JSONL).
If the script is killed it will skip already-processed rows on the next run.

---

## File Structure

```
model_inversion_on_pipeline_maskfill/
├── README.md
├── config.py             # model IDs, paths, HF tokens, BIO labels
├── generate_pairs.py     # main generation script
├── upload_to_hub.py      # standalone HF upload
└── output/
    ├── combo1_roberta_deberta.jsonl   ← generated pairs (Combo 1)
    ├── combo2_roberta_bart.jsonl      ← generated pairs (Combo 2)
    └── generate_pairs.log
```
