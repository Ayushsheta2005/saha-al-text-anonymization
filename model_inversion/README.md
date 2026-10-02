# Model Inversion Attack Pipeline

Black-box model inversion attack against the BART-base PII anonymizer
trained in `Seq2Seq_model/`. Given only the **anonymized output** of the
victim model, can an attacker reconstruct the original text including
private entities?

---

## Pipeline Overview

```
Step 1  generate_dataset.py   →  20,000 adversarial sentences  (Gemini)
Step 2  query_bart.py         →  (original, anonymized) pairs   (BART-base)
Step 3  train_inverter.py     →  inverter model                 (BART-base)
Step 4  evaluate_attack.py    →  ERR / BLEU / attack report
```

---

## Step 1 — Dataset Generation

```bash
cd model_inversion
python3 generate_dataset.py
```

Generates **20,000 PII-rich sentences** across 6 adversarial strategies:

| Strategy | Count | Purpose |
|---|---|---|
| S1 — Entity consistency | 6,000 | 150 probe names × 14 domains × varied templates — measures if BART maps each name deterministically |
| S2 — Combinatorial PII | 4,000 | Controlled combos (NAME+PHONE, NAME+EMAIL, NAME+DATE, etc.) — tests PII-type interaction |
| S3 — Paraphrase consistency | 3,000 | Same PII, different sentence structure — tests structural invariance of BART |
| S4 — Rarity spectrum | 3,000 | common → very_rare names — measures how rarity affects recovery rate |
| S5 — Cross-entity correlation | 2,000 | Email prefix matches name, org implies location, etc. — exploits correlations |
| S6 — Edge cases | 2,000 | Dense PII, implicit PII, multi-person, single-token — boundary cases |

**Each entry produced (by code, not Gemini):**
```json
{
  "id": "uuid",
  "sentence": "...",
  "probe_entity": "Kukshi Welshet",
  "entity_type": "NAME",
  "strategy": "S1_entity_consistency",
  "combo_type": null,
  "paraphrase_group_id": null,
  "rarity_tier": null,
  "correlation_type": null,
  "edge_case_type": null,
  "context_domain": "chat_message",
  "name_rarity": "medium",
  "name_origin_mix": ["SAS", "WEU"],
  "word_count": 14,
  "char_count": 78,
  "sentence_template": "CHAT",
  "pii_types_present": ["NAME", "PHONE"],
  "pii_count": 2,
  "attack_type": "model_inversion_black_box",
  "attack_difficulty": 3,
  "split": "train",
  "is_consistency_probe": true,
  "is_paraphrase_group": false,
  "is_correlation_probe": false,
  "batch_num": 12,
  "generated_at": "...",
  "generator_model": "gemini-2.5-flash"
}
```

**Name pool:** 150 synthetic multicultural mashup names covering 9 origin types:
`MC` (mashup), `EEU`, `AFR`, `SEA`, `MID`, `LAT`, `WEU`, `SAS`, `HYP`, `RAR`
spanning rarity tiers: `common`, `medium`, `rare`, `very_rare`.

---

## Step 2 — BART Querying + Consistency Analysis

```bash
python3 query_bart.py
```

- Loads the fine-tuned BART-base checkpoint from `Seq2Seq_model/checkpoints/bart-base/best_model.pt`
- Runs all 20,000 sentences through it (greedy decoding, batch=32)
- Writes `output/bart_query_pairs.jsonl` — rich (original, anonymized) pairs with all metadata
- Writes `output/consistency_report.json` — answers: *does BART deterministically map the same entity to the same replacement?*

**Consistency report includes:**
- Per-entity consistency score (1.0 = always same replacement)
- Lookup table: `entity → most_common_replacement`
- Leakage detection: entities the model failed to anonymize
- Per-strategy breakdown

---

## Step 3 — Inverter Training

```bash
python3 train_inverter.py
```

Trains a BART-base model on the task:
```
Input:  anonymized text   →   Output: original text
```

- Architecture: `facebook/bart-base` (same as victim — shared vocabulary)
- 5 epochs, AdamW lr=2e-5, linear warmup (200 steps), grad accum ×2
- Effective batch size: 32
- Saves best checkpoint to `inverter_checkpoint/best_model.pt`
- Logs ERR by strategy, rarity tier, and entity type every epoch

---

## Step 4 — Attack Evaluation

```bash
python3 evaluate_attack.py
```

Evaluates **two attacks** on the 5% held-out eval set:

**Attack A — Lookup Table** (zero training, baseline)
Uses the consistency report's inverse mapping to substitute replacements back.

**Attack B — Inverter Model** (learned)
Uses the trained BART-base inverter from Step 3.

**Metrics reported:**
- **ERR (exact)** — fraction of probe entities recovered verbatim
- **ERR (partial)** — fraction partially recovered (token overlap)
- **Token accuracy** — token-level match rate
- **Corpus BLEU** — overall reconstruction quality
- **Exact sentence match** — full sentence recovery rate
- Breakdowns by: strategy, name rarity, entity type

Outputs:
- `output/attack_results.json` — full machine-readable results
- `output/attack_report.txt` — human-readable comparison table

---

## File Layout

```
model_inversion/
├── config.py              # all paths, API keys, hyperparams, 150 probe names
├── generate_dataset.py    # Step 1 — Gemini dataset generator
├── query_bart.py          # Step 2 — BART querier + consistency analyser
├── train_inverter.py      # Step 3 — inverter model trainer
├── evaluate_attack.py     # Step 4 — attack evaluator
├── data/
│   ├── generation_state.json     # resume state for Step 1
│   └── bart_query_state.json     # resume state for Step 2
├── output/
│   ├── adversarial_dataset_raw.jsonl   # Step 1 output
│   ├── bart_query_pairs.jsonl          # Step 2 output
│   ├── consistency_report.json         # Step 2 analysis
│   ├── inverter_train.jsonl            # Step 3 train split
│   ├── inverter_eval.jsonl             # Step 3 eval split
│   ├── attack_results.json             # Step 4 results
│   └── attack_report.txt               # Step 4 human report
├── inverter_checkpoint/
│   ├── best_model.pt
│   ├── latest_checkpoint.pt
│   └── training_history.json
└── logs/
    ├── generate_dataset.log
    ├── query_bart.log
    ├── train_inverter.log
    └── evaluate_attack.log
```

---

## Resumable

Every step saves state and resumes automatically if interrupted:
- Step 1: tracks generated count per strategy in `data/generation_state.json`
- Step 2: tracks queried IDs in `data/bart_query_state.json`
- Step 3: saves `latest_checkpoint.pt` after every epoch
