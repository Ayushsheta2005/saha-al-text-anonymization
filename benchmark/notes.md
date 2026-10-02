# SAHA-AL Benchmarking Journey: Start to End

This document summarizes exactly what we did to transform the SAHA-AL project from a raw dataset pipeline into a formal, research-grade benchmark on Hugging Face.

## 1. Goal Realization
We realized that simply having a dataset and a model is an *experiment*, but standardizing it so that anyone can compare their own models turns it into a **benchmark**. This is the leap required for top-tier publication (e.g., NeurIPS). 

## 2. Standardizing the Dataset Schema
The raw `anonymized_dataset_final.jsonl` contained loose entities. We wrote `prepare_dataset.py` to:
- Normalize the keys (`id`, `original_text`, `anonymized_text`, `entities`).
- Compute explicit `start` and `end` character indices for every PII entity within the `original_text`. This ensures systems can map spans perfectly during evaluation.

## 3. Creating the Official Benchmark Splits
To prevent data contamination and ensure fair comparisons across future research, we deterministically split the 120k records into:
- **Train (80%)**: `data/train.jsonl`
- **Validation (10%)**: `data/validation.jsonl`
- **Test (10%)**: `data/test.jsonl` (This is the *frozen* benchmark evaluation set).

## 4. Extracting a Standalone Evaluation Tool
The old `eval_all.py` was too tightly coupled to loading specific PyTorch/PEFT models from training. A true benchmark needs a portable evaluator. We wrote `benchmark_eval.py` to be standalone. It takes any JSONL prediction file and compares it against `test.jsonl`, generating:
- **Entity Leakage Rate (ELR)**: Measures privacy leakage (lower is better).
- **Contextual Re-identification Risk (CRR3)**: Measures risk of context inference (lower is better).
- **BERTScore (F1)**: Measures semantic preservation and utility (higher is better).

## 5. Formalizing the Hugging Face Dataset Card
We wrote a massive `README.md` that serves as the official Hugging Face Dataset Card. It contains:
- The exact task definition.
- Explanations of the dataset structure and splits.
- The official benchmark protocol (zero test-set contamination).
- A markdown Leaderboard to track baseline models (Regex, spaCy, Pipeline, Seq2Seq).

## 6. Going Live on Hugging Face
We wrote `push_to_hf.py`, authorized it with a Hugging Face write token, and successfully compiled all the data into ultra-fast Apache Parquet format. 
- You successfully uploaded the entire `huggingbahl21/saha-al` dataset.
- The evaluation script and dataset card are now hosted on the hub.

## 7. What is Left? (The Benchmark Baselines)
The final step left to put this in a paper is **populating the leaderboard with both baseline and model numbers**.
You have now generated and evaluated the baseline predictions on `data/test.jsonl`:
- **Regex+Faker** → ELR: 83.49%, BERTScore F1: 98.13, CRR-3: 92.53
- **spaCy+Faker** → ELR: 26.70%, BERTScore F1: 91.84, CRR-3: 40.62
- **Presidio** → ELR: 33.86%, BERTScore F1: 90.02, CRR-3: 50.33

The remaining step is to generate and evaluate your own model predictions, then add those scores to the leaderboard.

Running `python benchmark_eval.py --pred model_predictions.jsonl` for your model output will give you the final metrics to populate the leaderboard.
