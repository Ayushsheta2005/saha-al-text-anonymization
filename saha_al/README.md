# SAHA-AL: Semi-Automatic Human-Augmented Active Learning Annotation Pipeline

SAHA-AL is a streamlined "Human-in-the-Loop" annotation tool built to construct a **Gold Standard parallel corpus** for privacy-preserving text anonymization. It is designed to work natively with the `ai4privacy/open-pii-masking-500k-ai4privacy` dataset.

## The Objective
Our final output must be pairs of `(original_text, anonymized_text)`, where `anonymized_text` contains realistic, synthetic fake data in place of all Personally Identifiable Information (PII) present in `original_text`, while retaining perfect sentence structure, fluency, and semantic meaning.

Since the AI4Privacy dataset already guarantees accurate detection of PII spans (through `privacy_mask` and `span_labels`), generating this parallel corpus becomes a **fill-in-the-blank text rewriting task** for human annotators, rather than a Named Entity Recognition (NER) detection task.

## New Pipeline Architecture 

1. **Loader:** `data_loader.py` fetches the AI4Privacy HF dataset, filters by language, and writes directly to an `annotation_queue.jsonl`.
2. **Annotation Interface:** A custom Streamlit application (`layer4_app.py`) parses the queue, highlights the pre-labelled entities, and provides a text box where the annotator writes an anonymized paragraph.
3. **Gold Standard Verification:** Leakage checks verify that no original PII text leaked into the anonymized text.
4. **Augmentation:** Script `augmentation.py` applies Template Filling, Entity Swapping, and EDA to multiply the hard-earned Gold Standard annotations.

## Running the Pipeline

### 1. Build the Annotation Queue
Pull data from huggingface and prepare the JSONLines queue for Streamlit:
```bash
python -m saha_al.data_loader --limit 1000
```
This loads 1,000 entries into `data/annotation_queue.jsonl`.

### 2. Start Annotating
Launch the Streamlit interface:
```bash
streamlit run saha_al/layer4_app.py
```
* **Read** the original text with PII highlighted in yellow.
* **Observe** the `Reference Masked Text` for structure.
* **Write** the fully coherent, synthetically generated anonymized sentence in the text box.
* **Accept** to save to `data/gold_standard.jsonl`.

### 3. Augment Data (Optional)
Run peer-reviewed data augmentation strategies over your accepted Gold Standard corpus to multiply your data footprint for model training.
```bash
python -m saha_al.augmentation --strategy all
```

## Directory Structure
```text
saha_al/
├── _archived/                 # Old spaCy NER & Active Learning detection files
├── data/                      # Queue, Gold Standard, Skipped, Augmentations
├── utils/
│   ├── quality_checks.py      # Leakage / Length ratio checks
│   ├── faker_replacements.py  # Synthetic generation for data augmentation
│   ├── entity_types.py        # Mappings of HF entity types to visual colors
│   └── io_helpers.py          # robust JSONL reader/writer
├── config.py                  # Paths and thresholds
├── data_loader.py             # Pulls AI4Privacy directly from HF
├── layer4_app.py              # The Streamlit text-rewriter application
└── augmentation.py            # Augmentation logic targeting gold_standard
```
