# SAHA-AL — Text Anonymization: Benchmark, Attack, and Defense

Anonymizing text is usually treated as a tagging problem: find the PII, replace it, ship it.
This project treats it as an adversarial one. It contributes a benchmark, a model that leads
it, an **attack that breaks that model**, and a **defense that closes the attack**.

| | |
|---|---|
| **Benchmark** | SAHA-AL — 120K records, 21 entity types, frozen 3.6K test split |
| **Best model** | BART-base, **0.93% entity leakage** vs 33.86% for Microsoft Presidio |
| **Attack** | Black-box model inversion recovers **~30% of PII** from anonymized output alone |
| **Defense** | Adversarial hardening drops recovery to **0.42% — a 98.6% reduction** |

---

## 1. The benchmark

Anonymization has three objectives that pull against each other, so it is scored on three
axes simultaneously:

| Metric | Definition | Direction |
|---|---|---|
| **ELR** | Entity Leakage Rate — fraction of gold entities still present in the output | lower |
| **BERTScore F1** | Semantic fidelity to the reference anonymization | higher |
| **CRR-3** | Capitalized 3-gram retention — residual surface overlap | lower |

A model that deletes every proper noun wins on ELR and destroys BERTScore. A model that
copies the input wins BERTScore and leaks everything. The benchmark exists to make that
tradeoff explicit.

**Construction**: 4-stage pipeline — high-recall pre-annotation, confidence routing to
human review, active learning on clustered anomalies, multi-pass review. Entity-boundary
agreement **κ = 0.83**, type-label agreement 87.4%.

**Splits**: train 113,133 / validation 3,600 / test 3,600 (frozen, 9,271 gold entities).
Augmented rows follow their gold `source_id` into training so augmentation cannot straddle
the split boundary.

### Leaderboard

Measured on the frozen test split with `benchmark/benchmark_eval.py`.

| Model | ELR ↓ | BERTScore F1 ↑ | CRR-3 ↓ |
|---|---|---|---|
| Regex + Faker | 83.49% | 98.13 | 92.53 |
| Presidio (Microsoft) | 33.86% | 90.02 | 50.33 |
| spaCy + Faker | 26.70% | 91.84 | 40.62 |
| **BART-base + PII** | **0.93%** | **92.74** | 34.62 |
| Flan-T5-small + PII | 0.99% | 92.47 | 34.94 |
| T5-small + PII | 1.54% | 92.59 | 35.27 |
| DistilBART + PII | 1.23% | 86.34 | 34.69 |
| T5-efficient-tiny + PII | 4.14% | 92.57 | 37.06 |

Regex+Faker is instructive: the highest BERTScore on the board and the worst leakage by a
wide margin. It preserves the text almost perfectly, which is precisely the failure.

**Cross-dataset transfer** to the Text Anonymization Benchmark (TAB), zero-shot:
77.7% entity recall, BERTScore F1 93.27 — quantifying the synthetic-to-real domain shift.

---

## 2. The attack

Low ELR means the *surface form* of each entity is gone. It does not mean the information
is gone. If the anonymizer maps a given name to the same replacement every time, or if its
substitutions correlate with the input, the mapping is learnable.

**Threat model**: black-box. The attacker sees only `anonymize(x)`, never the model weights
or the original `x`. The attacker may query the model freely.

**Method**: 20,000 adversarial probes across six strategies, each isolating a different
leak channel.

| Strategy | n | What it probes |
|---|---|---|
| S1 — entity consistency | 6,000 | Does the same name map deterministically? |
| S2 — combinatorial PII | 4,000 | Do PII types interact (NAME+PHONE, NAME+EMAIL)? |
| S3 — paraphrase consistency | 3,000 | Is the mapping invariant to sentence structure? |
| S4 — rarity spectrum | 3,000 | Do rare names survive where common ones don't? |
| S5 — cross-entity correlation | 2,000 | Does an email prefix betray the name? |
| S6 — edge cases | 2,000 | Dense, implicit, multi-person, single-token PII |

A seq2seq inverter is then trained on `(anonymized → original)` pairs and scored by **ERR**
(Entity Recovery Rate).

**Result: ERR ≈ 0.30.** Roughly a third of the entities a 0.93%-ELR anonymizer had
"removed" are reconstructible from its output alone. **ELR is not a privacy guarantee.**

---

## 3. The defense

Adversarial hardening against the inverter, re-evaluated on the attack's own eval set
(n = 1,918):

```
Pre-hardening  ERR (exact)   ~0.3000
Post-hardening ERR (exact)    0.0042      -98.6%
Post-hardening ERR (partial)  0.0021
```

Per-strategy, after hardening:

| Strategy | ERR | n |
|---|---|---|
| S1 entity consistency | 0.0000 | 556 |
| S2 combinatorial PII | 0.0024 | 424 |
| S3 paraphrase consistency | 0.0038 | 261 |
| S4 rarity spectrum | 0.0064 | 311 |
| S5 cross-entity correlation | 0.0056 | 180 |
| **S6 edge cases** | **0.0161** | 186 |

Deterministic mapping — the most exploitable channel — is closed completely. **Edge cases
remain 4× more vulnerable than the mean**, and that is the honest open problem: dense,
implicit and multi-person PII is where residual recoverable signal lives.

Anonymization quality is preserved through hardening (entity leakage 0.0695 on the
validation set, BLEU-4 0.615, ROUGE-L 75.66) — the defense does not buy privacy by
destroying utility.

---

## Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env            # set GEMINI_API_KEY, HF_TOKEN

python benchmark/check_split_integrity.py --data benchmark/data
python benchmark/benchmark_eval.py --gold benchmark/data/test.jsonl \
       --pred benchmark/predictions/predictions_bart.jsonl --print-types

cd model_inversion                # attack: 4 resumable stages
python generate_dataset.py && python query_bart.py
python train_inverter.py && python evaluate_attack.py
```

Every attack stage checkpoints and resumes from interruption.

---

## Known limitations

Stated rather than buried, because a benchmark's credibility is its failure disclosure.

- **Entity types are not realized in the released splits.** All 286,542 train / 9,271 test
  entities carry `type: UNKNOWN`. The 21-type taxonomy exists in the annotation schema but
  is absent from the published JSONL, so **per-type ELR cannot currently be computed** —
  which is the breakdown a practitioner most wants. Restoring types is the highest-value
  next change.
- **Split overlap.** `check_split_integrity.py` reports 7 texts shared train↔test, 5
  train↔validation, 2 test↔validation, plus one `entry_id` collision per pair. At 0.19% of
  test this does not move the leaderboard, but a frozen split should be exactly disjoint.
- **Intra-split duplicates**: 990 in train, 36 in test, 24 in validation.
- **Synthetic source text.** Documents are LLM-generated to mirror real patterns without
  exposing real PII. The TAB transfer result quantifies the resulting domain shift; it does
  not eliminate it.
- **English only.**
- **The attack targets one victim.** ERR ≈ 0.30 is measured against the BART-base
  anonymizer. Whether the other four seq2seq models leak comparably is untested.

## Layout

```
benchmark/              dataset, frozen splits, evaluation harness, baselines
model_inversion/        the attack: generation -> query -> inverter -> evaluation
adversarial_training/   the defense, and its evaluation report
tab-finetuning/         cross-dataset transfer to TAB
saha_al/                annotation pipeline (routing, active learning, augmentation)
pipeline_maskfill/      mask-and-fill alternative to end-to-end seq2seq
Seq2Seq_model/          the victim anonymizer
```

## References

- Pilán et al., *The Text Anonymization Benchmark (TAB)* (2022)
- Microsoft Presidio — PII detection baseline
- Carlini et al., *Extracting Training Data from Large Language Models* (2021)
- Song & Raghunathan, *Information Leakage in Embedding Models* (2020)
