# SAHA-AL: PII Anonymization Benchmark

**SAHA-AL** is a comprehensive dataset and benchmark for training and evaluating text anonymization systems. The task requires models to identify Personally Identifiable Information (PII) within a given text and replace it with realistic, non-identifying proxies—preserving the semantic utility and context of the original text.

## 1. Task Description

**Given text containing PII, generate realistic, anonymized text.**

Models must locate sensitive entities (e.g., names, locations, document numbers, phone numbers) and replace them contextually. The ultimate goal is zero data leakage without degrading the grammatical correctness or downstream utility of the sentence.

## 2. Dataset Construction

SAHA-AL features a 4-layer robust annotation pipeline:

1. **Pre-Annotation**: Generating high-recall rough masking.
2. **Confidence Routing**: Directing low-confidence examples to human verification.
3. **Active Learning**: Model refinement on newly clustered anomalies.
4. **Human Review**: Comprehensive multi-annotator review for complex context inferences.

## 3. Data Quality

Our pipeline was validated with rigorous inter-annotator agreements over a 60-day review period by a 4-annotator team, featuring:

- **$\kappa=0.83$**: Entity boundary agreement.
- **87.4%**: Type-label agreement.
- **26 sec/entry**: Weighted average human processing time using confidence routing.

## 4. Entity Types

The dataset features 21 fine-grained entity types, providing significantly richer entity diversity than typical binary sensitive/non-sensitive datasets:

- `FULLNAME`, `FIRSTNAME`, `LASTNAME`
- `EMAIL`, `PHONE`
- `SSN`, `ID_NUMBER`, `CREDIT_CARD`, `IBAN`, `ACCOUNT`
- `ADDRESS`, `LOC`, `ORG`
- `DATE`
- `IP_ADDRESS`, `URL`, `USERNAME`, `PASSWORD`
- *(and more)*

## 5. Dataset Structure

The dataset comprises JSONL structures formatted closely to the HuggingFace standard:

```json
{
  "id": "sample_00000",
  "original_text": "John lives in New York.",
  "anonymized_text": "David lives in Chicago.",
  "entities": [
    {"text": "John", "type": "FIRSTNAME", "start": 0, "end": 4},
    {"text": "New York", "type": "LOC", "start": 14, "end": 22}
  ]
}
```

## 6. Splits

To ensure complete isolation of the evaluation subset, the dataset is strictly separated. Any augmented training data is derived mechanically from gold data; augmented entries are forced to follow their gold "source_id" directly into the training set, avoiding test leakage.

- **Train**: ~114k records (Gold + Augmented)
- **Validation**: ~3.6k records (Gold only)
- **Test**: ~3.6k records (Gold only) *[Frozen Benchmark]*

## 7. Metrics

We evaluate systems along three conflicting axes with strict execution metrics:


| Metric           | Exact Definition                                                                                                                                            |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ELR**          | Fraction of entities in `entities` list whose `text` field appears as a case-insensitive substring in the predicted `anonymized_text`. Computed per-entity. |
| **BERTScore F1** | Computed via `bert_score`. Uses pinned `distilbert-base-uncased` (or `microsoft/deberta-xlarge-mnli`).                                                      |
| **CRR-3**        | Fraction of capitalized 3-grams from `original_text` that appear in the prediction. (Capitalized = at least one token starts with uppercase).               |


## 8. Benchmark Protocol

To appear on the Leaderboard, models must:

1. **Zero Contamination**: Do not train or tune on the test set (`data/test.jsonl`).
2. **Standard Evaluation**: Evaluate your JSONL predictions using the provided portable `benchmark_eval.py`.
3. **Format**: Your predictions must follow this format:
  `{"id": "sample_00000", "anonymized_text": "David lives in Chicago"}`

## 9. Leaderboard

*Measured on the frozen `data/test.jsonl` split (3,600 records, 9,271 gold entities).*

### Heuristic / rule-based baselines


| Model           | ELR ↓  | BERTScore F1 ↑ | CRR-3 ↓ |
| --------------- | ------ | -------------- | ------- |
| **Regex+Faker** | 83.49% | 98.13          | 92.53   |
| **spaCy+Faker** | 26.70% | 91.84          | 40.62   |
| **Presidio**    | 33.86% | 90.02          | 50.33   |


### Fine-tuned seq2seq (this repository)

Evaluated with `benchmark/benchmark_eval.py` (`--gold data/test.jsonl`, BERTScore model `distilbert-base-uncased`). JSON summaries: `benchmark/Results/eval_predictions_*.json`; predictions: `benchmark/predictions/predictions_*.jsonl`.


| Model                       | ELR ↓ | BERTScore F1 ↑ | CRR-3 ↓ |
| --------------------------- | ----- | -------------- | ------- |
| **BART-base + PII**         | 0.93% | 92.74          | 34.62   |
| **Flan-T5-small + PII**     | 0.99% | 92.47          | 34.94   |
| **DistilBART + PII**        | 1.23% | 86.34          | 34.69   |
| **T5-small + PII**          | 1.54% | 92.59          | 35.27   |
| **T5-efficient-tiny + PII** | 4.14% | 92.57          | 37.06   |


*Notes:* Gold `entities[].type` is `UNKNOWN` for this split, so per-type ELR is not reported here; ELR, CRR-3, and BERTScore still use entity surface strings and full texts. One test row has empty `original_text` and is omitted from BERTScore only (see script output).

*(Run `python benchmark/benchmark_eval.py --gold data/test.jsonl --pred <predictions.jsonl> --print-types` to reproduce; add `--summary-file` to write JSON.)*

## 10. Cross-Dataset Transfer (TAB)

We recommend benchmarking your top zero-shot/pretrained model directly against the [Text Anonymization Benchmark (TAB)](https://github.com/NorskRegnesentral/text-anonymization-benchmark) without fine-tuning to quantify the synthetic-to-real transfer domain shift.

## Limitations

- Source texts rely on synthetic/LLM-generated prompts mirroring real patterns to preserve privacy.
- Currently English-only contexts.
- Specialized document types (e.g., highly unstructured logs) may fall out of domain.

