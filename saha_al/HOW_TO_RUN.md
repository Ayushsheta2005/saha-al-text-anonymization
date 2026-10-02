# 🚀 HOW TO RUN — SAHA-AL Pipeline

> **Complete step-by-step guide** to set up, run, and use every component of the SAHA-AL annotation pipeline.
>
> Estimated time: **~15 minutes setup**, then pipeline runs incrementally.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Step 1 — Pre-Annotation (Layer 1)](#4-step-1--pre-annotation-layer-1)
5. [Step 2 — Confidence Routing (Layer 2)](#5-step-2--confidence-routing-layer-2)
6. [Step 3 — Annotation Interface (Layer 4)](#6-step-3--annotation-interface-layer-4)
7. [Step 4 — Active Learning (Layer 3)](#7-step-4--active-learning-layer-3)
8. [Step 5 — Data Augmentation (Phase 2)](#8-step-5--data-augmentation-phase-2)
9. [Step 6 — Export Reports](#9-step-6--export-reports)
10. [Full Pipeline — One-Shot Run](#10-full-pipeline--one-shot-run)
11. [Configuration Reference](#11-configuration-reference)
12. [CLI Flags Reference](#12-cli-flags-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [FAQ](#14-faq)

---

## 1. Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10+ | 3.11 or 3.12 |
| **RAM** | 4 GB | 8 GB (spaCy `en_core_web_lg` loads ~560 MB) |
| **Disk** | 2 GB free | 5 GB free (models + augmented data) |
| **OS** | Linux / macOS / WSL2 | Ubuntu 22.04+ |
| **GPU** | Not required | Optional (for `en_core_web_trf`) |

### Software

- Python 3.10 or higher
- `pip` (comes with Python)
- `git` (for cloning)
- A modern web browser (for Streamlit UI — Chrome / Firefox recommended)

---

## 2. Environment Setup

### 2.1 Clone / Navigate to Project

```bash
cd /home/ayush/Desktop/sem6/inlp/PROJECT
```

### 2.2 Create a Virtual Environment

```bash
# Create venv
python3 -m venv .venv

# Activate it
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows (PowerShell)
# .venv\Scripts\activate.bat # Windows (CMD)
```

> ⚠️ **Always activate the venv** before running any pipeline command. If you see `externally-managed-environment` errors, you forgot this step.

### 2.3 Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r saha_al/requirements.txt
```

This installs:

| Package | Version | Why |
|---------|---------|-----|
| `streamlit` | ≥ 1.28.0 | Annotation web interface (Layer 4) |
| `spacy` | ≥ 3.7.0 | NER engine + bootstrap training |
| `faker` | ≥ 20.0.0 | Multi-locale PII replacement generation |
| `tqdm` | ≥ 4.65.0 | Progress bars for batch CLI scripts |
| `pandas` | ≥ 2.0.0 | Statistics, data manipulation |
| `nltk` | ≥ 3.8.0 | WordNet synonyms for EDA augmentation (optional fallback built-in) |

### 2.4 Download spaCy Language Model

```bash
# Standard CPU model (~560 MB) — REQUIRED
python -m spacy download en_core_web_lg

# GPU-accelerated transformer model (~500 MB) — OPTIONAL, faster but needs GPU
# python -m spacy download en_core_web_trf
```

If you use `en_core_web_trf`, update `config.py`:
```python
SPACY_MODEL = "en_core_web_trf"   # instead of "en_core_web_lg"
```

### 2.5 (Optional) Download NLTK WordNet Data

Only needed if you want richer synonyms in EDA augmentation. The pipeline has a built-in synonym dictionary as fallback.

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2.6 Verify Installation

```bash
# Quick sanity check — all imports should work
python -c "
import spacy
import faker
import streamlit
import tqdm
import pandas
print('✅ All packages installed successfully')
print(f'   spaCy: {spacy.__version__}')
print(f'   Faker: {faker.__version__}')
print(f'   Streamlit: {streamlit.__version__}')
print(f'   Pandas: {pandas.__version__}')
"

# Verify spaCy model
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print(f'✅ spaCy model loaded: {nlp.meta[\"name\"]}')"

# Verify SAHA-AL imports
python -c "from saha_al.config import ENTITY_TYPES; print(f'✅ SAHA-AL config loaded: {len(ENTITY_TYPES)} entity types')"
```

Expected output:
```
✅ All packages installed successfully
   spaCy: 3.7.x
   Faker: 2x.x.x
   Streamlit: 1.3x.x
   Pandas: 2.x.x
✅ spaCy model loaded: core_web_lg
✅ SAHA-AL config loaded: 21 entity types
```

---

## 3. Data Preparation

### 3.1 Source Data Location

The source dataset should be at:
```
saha_al/data/original_data.jsonl
```

If it's not there, create a symlink:
```bash
# From the PROJECT root
ln -sf /home/ayush/Desktop/sem6/inlp/PROJECT/original_data.jsonl saha_al/data/original_data.jsonl
```

### 3.2 Verify Source Data

```bash
# Check file exists and count entries
wc -l saha_al/data/original_data.jsonl
# Expected: 120333 saha_al/data/original_data.jsonl

# Preview first entry
head -1 saha_al/data/original_data.jsonl | python -m json.tool
```

Expected schema per line:
```json
{
    "entry_id": 1,
    "original_text": "My name is John Smith and my SSN is 123-45-6789..."
}
```

### 3.3 Directory Structure (Auto-Created)

The pipeline auto-creates needed directories, but you can pre-create them:

```bash
mkdir -p saha_al/data/queues
mkdir -p saha_al/data/backups
mkdir -p saha_al/models/bootstrap
mkdir -p saha_al/logs
```

---

## 4. Step 1 — Pre-Annotation (Layer 1)

**What it does:** Runs spaCy NER + 23 regex patterns on every entry, merges detections, generates Faker replacements, and builds anonymized text.

**Script:** `saha_al/scripts/run_layer1.py`  
**Core module:** `saha_al/layer1_preannotation.py`

### Test Run (First 100 Entries)

```bash
python saha_al/scripts/run_layer1.py --limit 100
```

Expected output:
```
Loading spaCy model: en_core_web_lg...
Processing entries: 100%|██████████| 100/100 [00:45<00:00,  2.22it/s]
✅ Pre-annotated 100 entries → saha_al/data/pre_annotated.jsonl
   Entities found: ~487 (avg 4.9/entry)
   Sources: spaCy=312, regex=264, merged=487
```

### Full Run (All 120K Entries)

```bash
python saha_al/scripts/run_layer1.py
```

> ⏱️ **Estimated time:** 12–18 hours for 120K entries on CPU (spaCy is the bottleneck).
>
> 💡 **Resume-safe:** If you stop and restart, it skips already-processed entries. Use `--no-resume` to force re-processing.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | `data/original_data.jsonl` | Custom input JSONL path |
| `--output PATH` | `data/pre_annotated.jsonl` | Custom output path |
| `--limit N` | None (all) | Process only first N entries |
| `--no-resume` | Resume enabled | Force re-processing of all entries |

### Verify Output

```bash
# Count pre-annotated entries
wc -l saha_al/data/pre_annotated.jsonl

# Preview first entry
head -1 saha_al/data/pre_annotated.jsonl | python -m json.tool | head -40

# Quick entity stats
python -c "
import json
entities = 0
with open('saha_al/data/pre_annotated.jsonl') as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        entities += len(entry.get('entities', []))
        if i >= 99: break
print(f'First 100 entries: {entities} entities ({entities/100:.1f} avg/entry)')
"
```

---

## 5. Step 2 — Confidence Routing (Layer 2)

**What it does:** Reads `pre_annotated.jsonl` and triages every entry into GREEN / YELLOW / RED queues based on confidence scores and agreement levels.

**Script:** `saha_al/scripts/run_layer2.py`  
**Core module:** `saha_al/layer2_routing.py`

### Run Routing

```bash
python saha_al/scripts/run_layer2.py
```

Expected output:
```
Routing pre-annotated entries...
100%|██████████| 120333/120333 [00:12<00:00, 9876.54it/s]

✅ Routing complete:
   🟢 GREEN:  67,412 entries (56.0%) → data/queues/green_queue.jsonl
   🟡 YELLOW: 38,201 entries (31.7%) → data/queues/yellow_queue.jsonl
   🔴 RED:    14,720 entries (12.3%) → data/queues/red_queue.jsonl
```

### Routing Rules

| Queue | Condition | Human Effort |
|-------|-----------|-------------|
| 🟢 GREEN | min confidence ≥ 0.85, no partial agreement, no UNKNOWN types | Auto-approved (spot-check only) |
| 🟡 YELLOW | Medium confidence, minor ambiguity | Quick human review |
| 🔴 RED | Any confidence < 0.50, or >30% UNKNOWN, or ≥2 partial disagreements | Expert review required |

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | `data/pre_annotated.jsonl` | Input pre-annotated JSONL |
| `--green PATH` | `data/queues/green_queue.jsonl` | Green queue output |
| `--yellow PATH` | `data/queues/yellow_queue.jsonl` | Yellow queue output |
| `--red PATH` | `data/queues/red_queue.jsonl` | Red queue output |

### Verify Output

```bash
wc -l saha_al/data/queues/*.jsonl
# Expected: three files with counts summing to total pre-annotated entries
```

---

## 6. Step 3 — Annotation Interface (Layer 4)

**What it does:** Launches a full-featured Streamlit web app for human annotators to review, correct, and approve entries.

**Script:** `saha_al/layer4_app.py` (runs directly with Streamlit)

### Launch the Interface

```bash
streamlit run saha_al/layer4_app.py
```

This opens a browser at `http://localhost:8501`.

> 💡 **Custom port:** `streamlit run saha_al/layer4_app.py --server.port 8080`
>
> 💡 **LAN access:** `streamlit run saha_al/layer4_app.py --server.address 0.0.0.0` (allows other machines on your network to access it)

### Interface Walkthrough

#### A. Sidebar Setup

1. **Enter your name** in the "Annotator Name" field (tracked in metadata)
2. **Select queue** — Start with 🟡 YELLOW (most impactful); GREEN is for spot-checks
3. Click **"Load Queue"** to begin

#### B. Annotating an Entry

Each entry shows:
- **Original text** with color-coded PII highlights (21 unique colors per entity type)
- **Entity review table** — one row per detected entity with:
  - Detected text span
  - Entity type (editable dropdown — all 21 types)
  - Confidence score (color-coded badge: 🟢 ≥ 0.85, 🟡 ≥ 0.50, 🔴 < 0.50)
  - Faker replacement (editable text field)
  - Delete button (remove false positives)
- **Anonymized text preview** — live preview of the final output

#### C. Actions

| Button | What it does | When to use |
|--------|-------------|-------------|
| ✅ **Accept** | Saves entry to `gold_standard.jsonl` with current entities + replacements | Everything looks correct |
| 🚩 **Flag** | Saves to `flagged.jsonl` with a reason | Confusing entry, unsure about types, needs expert |
| ⏭️ **Skip** | Saves to `skipped.jsonl` | Garbage text, not worth annotating |
| 🔄 **Regenerate** | Generates fresh Faker replacements for all entities | Current replacements look unrealistic |

#### D. Statistics Tab

- Entity type distribution bar chart
- Annotator contribution breakdown
- Queue progress (how many reviewed vs. remaining)
- Annotation rate (entries/hour)

#### E. Flagged Review Tab

- View all flagged entries with reasons
- Re-annotate and accept, or confirm the flag

### Annotation Tips

1. **Start with YELLOW queue** — these are the most informative for the model
2. **Spend ~15–30 seconds per entry** — don't overthink it
3. **Fix entity types first, then replacements** — type accuracy matters more
4. **Flag liberally** — better to flag and revisit than accept a bad annotation
5. **Check the anonymized preview** — if it reads naturally, the entry is good
6. **Auto-backup** happens every 50 entries — you won't lose work

### Stop / Resume

- Just close the browser tab or Ctrl+C the terminal
- Next time you launch, your progress is preserved (gold standard is append-only)
- The queue automatically skips entries you've already annotated

---

## 7. Step 4 — Active Learning (Layer 3)

**What it does:** Trains a lightweight NER model on your gold annotations, then re-scores the queues so the most uncertain (= most valuable) entries appear first.

**Script:** `saha_al/scripts/run_bootstrap_train.py`  
**Core module:** `saha_al/layer3_active_learning.py`

### 7.1 Train Bootstrap Model

> ⚠️ **Requires ≥ 200 gold-standard entries.** Run this after you've annotated at least 200 entries in the Streamlit interface.

```bash
python saha_al/scripts/run_bootstrap_train.py train
```

Expected output:
```
Loading gold standard (847 entries)...
Converting to spaCy training format...
Training bootstrap NER (10 iterations)...
Epoch  1/10 — Loss: 245.32
Epoch  2/10 — Loss: 189.17
...
Epoch 10/10 — Loss: 42.08
✅ Model saved → models/bootstrap/bootstrap_20260308_143021/
   Trained on: 847 entries, 4,129 entities
```

### 7.2 Re-Score Queues by Uncertainty

```bash
python saha_al/scripts/run_bootstrap_train.py score
```

Expected output:
```
Loading bootstrap model: models/bootstrap/bootstrap_20260308_143021/
Scoring YELLOW queue (38,201 entries)...
100%|██████████| 38201/38201 [02:34<00:00, 247.12it/s]
Scoring RED queue (14,720 entries)...
100%|██████████| 14720/14720 [00:58<00:00, 253.79it/s]
✅ Queues re-sorted by uncertainty (highest uncertainty first)
   YELLOW: avg uncertainty = 0.31
   RED:    avg uncertainty = 0.58
```

Now when you reopen the Streamlit interface, the **most uncertain entries appear first** — maximizing the value of each annotation.

### CLI Options

**`train` subcommand:**

| Flag | Default | Description |
|------|---------|-------------|
| `--gold PATH` | `data/gold_standard.jsonl` | Gold standard input |
| `--output DIR` | `models/bootstrap/` | Model output directory |
| `--n-iter N` | 10 | Training epochs |

**`score` subcommand:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model DIR` | Latest in `models/bootstrap/` | Model directory to use |
| `--yellow PATH` | `data/queues/yellow_queue.jsonl` | Yellow queue to re-score |
| `--red PATH` | `data/queues/red_queue.jsonl` | Red queue to re-score |

### When to Retrain

| Trigger | Action |
|---------|--------|
| After first 200 gold entries | First training run |
| Every 500 new gold entries | Retrain (`train`) + re-score (`score`) |
| If annotation feels "too easy" | Retrain — model has probably improved |

---

## 8. Step 5 — Data Augmentation (Phase 2)

**What it does:** Expands the gold standard using three peer-reviewed augmentation strategies to create a much larger training set.

**Script:** `saha_al/scripts/run_augmentation.py`  
**Core module:** `saha_al/augmentation.py`

> ⚠️ **Run this after annotation is complete** (or at checkpoints). Only verified gold entries are augmented.

### 8.1 Run All Three Strategies (Recommended)

```bash
python saha_al/scripts/run_augmentation.py --strategy all
```

Expected output:
```
Loading gold standard (10,000 entries)...

═══ Entity Swap (Dai & Adel, COLING 2020) ═══
Generating ×4 variants per entry...
100%|██████████| 10000/10000 [05:23<00:00, 30.94it/s]
✅ Entity Swap: 40,000 variants generated

═══ Template Fill (Anaby-Tavor et al., AAAI 2020) ═══
Building entity pools from gold standard...
   Pools: FULLNAME=8421, EMAIL=6234, PHONE=7891, SSN=3245, ...
Filling 40 templates (×5 multiplier)...
100%|██████████| 50000/50000 [01:12<00:00, 694.44it/s]
✅ Template Fill: 50,000 entries generated

═══ EDA — Easy Data Augmentation (Wei & Zou, EMNLP 2019) ═══
Generating ×3 variants per entry (α=0.1)...
100%|██████████| 10000/10000 [02:45<00:00, 60.61it/s]
✅ EDA: 30,000 variants generated

═══ Summary ═══
   Gold standard:     10,000
   Entity Swap:      +40,000
   Template Fill:    +50,000
   EDA:              +30,000
   ─────────────────────────
   Augmented total:   120,000 → data/augmented_data.jsonl
   Training set:      130,000 → data/training_data.jsonl (gold + augmented, shuffled)
```

### 8.2 Run Individual Strategies

```bash
# Entity Swap only — custom multiplier
python saha_al/scripts/run_augmentation.py --strategy swap
python saha_al/scripts/run_augmentation.py --strategy swap --multiplier 6

# Template Fill only — custom count
python saha_al/scripts/run_augmentation.py --strategy template
python saha_al/scripts/run_augmentation.py --strategy template --count 20000

# EDA only — custom multiplier and alpha
python saha_al/scripts/run_augmentation.py --strategy eda
python saha_al/scripts/run_augmentation.py --strategy eda --multiplier 5 --alpha 0.15
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | **Required** | `swap`, `template`, `eda`, or `all` |
| `--multiplier N` | 4 (swap) / 3 (eda) | Number of variants per gold entry |
| `--count N` | 5000 | Number of template-filled entries to generate |
| `--alpha F` | 0.1 | EDA word-change fraction (0.0–0.5) |
| `--output PATH` | `data/augmented_data.jsonl` | Output path for augmented data |

### Strategy Summary

| Strategy | What changes | What stays | Multiplier | Speed |
|----------|-------------|------------|-----------|-------|
| **Entity Swap** | All PII values (new Faker) | Sentence structure exactly | ×4 | ~31 entries/sec |
| **Template Fill** | Entire sentence (from templates) | Entity format patterns | ×5 | ~694 entries/sec |
| **EDA** | Non-PII context words | All PII spans exactly | ×3 | ~61 entries/sec |

### Verify Output

```bash
# Check augmented data
wc -l saha_al/data/augmented_data.jsonl
# Expected: ~120,000 lines (for 10K gold)

# Check training data (gold + augmented)
wc -l saha_al/data/training_data.jsonl
# Expected: ~130,000 lines

# Preview an augmented entry
head -1 saha_al/data/augmented_data.jsonl | python -m json.tool | head -30

# Check strategy distribution
python -c "
import json
from collections import Counter
c = Counter()
with open('saha_al/data/augmented_data.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        strategy = entry.get('metadata', {}).get('augmentation', {}).get('strategy', 'unknown')
        c[strategy] += 1
for k, v in c.most_common():
    print(f'  {k}: {v:,}')
print(f'  Total: {sum(c.values()):,}')
"
```

---

## 9. Step 6 — Export Reports

**What it does:** Generates a comprehensive JSON report + human-readable Markdown summary of annotation statistics.

**Script:** `saha_al/scripts/export_report_stats.py`

### Run Report Export

```bash
python saha_al/scripts/export_report_stats.py
```

Expected output:
```
Generating report...
   Gold standard: 10,000 entries
   Total entities: 48,923
   Annotators: 4
   Flagged: 127
   Skipped: 43

✅ Reports saved:
   JSON:     logs/report_20260308.json
   Markdown: logs/report_20260308.md
```

### Report Contents

The generated report includes:
- **Dataset overview** — total entries, entity counts, avg entities/entry
- **Entity type distribution** — count + percentage per type
- **Annotator contributions** — entries per annotator
- **Queue source breakdown** — how many came from GREEN / YELLOW / RED
- **Quality metrics** — warnings count, flagged ratio
- **Timeline** — first/last annotation timestamps

---

## 10. Step 7 — Dataset Analysis & Charts

**What it does:** Generates 10 publication-ready PNG charts, 5 LaTeX table snippets, a Markdown summary report, and a JSON statistics dump.

**Script:** `saha_al/scripts/dataset_analysis.py`

### Run Dataset Analysis

```bash
python saha_al/scripts/dataset_analysis.py
```

Expected output:
```
══════════════════════════════════════════════════
  SAHA-AL Dataset Analysis
══════════════════════════════════════════════════

Loading data...
  ✓ original: 120,333 entries
  ✓ pre_annotated: 120,333 entries
  ✓ gold: 10,000 entries
  ✓ augmented: 130,000 entries
  ...

Computing statistics...
Generating charts...
✅ Charts saved to: saha_al/reports/figures
Generating LaTeX tables...
✅ LaTeX tables saved to: saha_al/reports/tables
Generating Markdown report...
✅ Markdown report saved to: saha_al/reports/dataset_analysis_report.md
Saving statistics JSON...
✅ Stats JSON saved to: saha_al/reports/dataset_statistics.json
```

### Generated Files

| File | Description |
|------|-------------|
| `reports/figures/entity_type_distribution.png` | Bar chart of entity type counts |
| `reports/figures/entity_type_pie.png` | Pie chart of top 10 types |
| `reports/figures/routing_distribution.png` | GREEN / YELLOW / RED bar chart |
| `reports/figures/confidence_histogram.png` | Confidence score histogram |
| `reports/figures/agreement_breakdown.png` | Agreement level pie chart |
| `reports/figures/text_length_distribution.png` | Text length histogram |
| `reports/figures/pii_density.png` | PII density histogram |
| `reports/figures/entities_per_entry.png` | Entities-per-entry histogram |
| `reports/figures/cooccurrence_heatmap.png` | Entity type co-occurrence heatmap |
| `reports/figures/detection_source.png` | spaCy vs regex vs both pie |
| `reports/tables/table_dataset_overview.tex` | LaTeX: Dataset overview |
| `reports/tables/table_entity_distribution.tex` | LaTeX: Entity distribution |
| `reports/tables/table_routing.tex` | LaTeX: Routing stats |
| `reports/tables/table_augmentation.tex` | LaTeX: Augmentation breakdown |
| `reports/tables/table_text_stats.tex` | LaTeX: Text statistics |
| `reports/dataset_analysis_report.md` | Full Markdown report |
| `reports/dataset_statistics.json` | Raw stats as JSON |

### Options

```bash
# Custom output directory
python saha_al/scripts/dataset_analysis.py --output-dir reports/v2/

# Skip chart generation (text reports only)
python saha_al/scripts/dataset_analysis.py --no-charts
```

### Using LaTeX Tables in Your Report

Include generated tables directly:

```latex
\input{saha_al/reports/tables/table_dataset_overview.tex}
\input{saha_al/reports/tables/table_entity_distribution.tex}
\input{saha_al/reports/tables/table_routing.tex}
\input{saha_al/reports/tables/table_augmentation.tex}
\input{saha_al/reports/tables/table_text_stats.tex}
```

---

## 11. Full Pipeline — One-Shot Run

Copy-paste this block to run the **entire pipeline** from scratch:

```bash
# ── Setup ──
cd /home/ayush/Desktop/sem6/inlp/PROJECT
source .venv/bin/activate

# ── Phase 1: Annotation Pipeline ──

# Step 1: Pre-annotate (test first, then full)
python saha_al/scripts/run_layer1.py --limit 500      # 🧪 Test on 500
python saha_al/scripts/run_layer1.py                    # 🚀 Full 120K (12-18 hrs)

# Step 2: Route into queues
python saha_al/scripts/run_layer2.py

# Step 3: Launch annotation UI
streamlit run saha_al/layer4_app.py
# >>> Annotate entries in the browser (target: 200+ for first bootstrap)
# >>> Press Ctrl+C when done for this session

# Step 4: Train bootstrap model (after ≥200 gold entries)
python saha_al/scripts/run_bootstrap_train.py train

# Step 5: Re-sort queues by uncertainty
python saha_al/scripts/run_bootstrap_train.py score

# Step 3 again: Continue annotating (now with uncertainty-sorted queues)
streamlit run saha_al/layer4_app.py
# >>> Repeat Steps 4-5 every ~500 new annotations

# ── Phase 2: Augmentation ──

# Step 6: Augment gold standard
python saha_al/scripts/run_augmentation.py --strategy all

# Step 7: Export final report
python saha_al/scripts/export_report_stats.py

# Step 8: Generate dataset analysis (charts + LaTeX tables)
python saha_al/scripts/dataset_analysis.py
```

### Pipeline Lifecycle

```
Day 1:     Setup + Layer 1 (let run overnight)
Day 2:     Layer 2 routing + start annotating (YELLOW queue)
Day 2–7:   Annotate ~1,500/day across 4 annotators
Day 3:     First bootstrap train (after 200+ gold)
Day 5:     Retrain + re-score (after 1,000+ gold)
Day 7:     Retrain + re-score (after 2,500+ gold)
Day 10:    Annotation complete (~10K gold entries)
Day 10:    Run augmentation → 130K training set
Day 10:    Export report → submit
```

---

## 11. Configuration Reference

All configuration lives in `saha_al/config.py`. Key settings:

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `ORIGINAL_DATA_PATH` | `data/original_data.jsonl` | Source text data |
| `PRE_ANNOTATED_PATH` | `data/pre_annotated.jsonl` | Layer 1 output |
| `GREEN_QUEUE_PATH` | `data/queues/green_queue.jsonl` | Auto-approved entries |
| `YELLOW_QUEUE_PATH` | `data/queues/yellow_queue.jsonl` | Human review queue |
| `RED_QUEUE_PATH` | `data/queues/red_queue.jsonl` | Expert review queue |
| `GOLD_STANDARD_PATH` | `data/gold_standard.jsonl` | Final annotated dataset |
| `AUGMENTED_DATA_PATH` | `data/augmented_data.jsonl` | Augmentation output |
| `TRAINING_DATA_PATH` | `data/training_data.jsonl` | Gold + augmented combined |
| `BOOTSTRAP_MODEL_DIR` | `models/bootstrap/` | Saved NER models |

### Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_HIGH` | `0.85` | Minimum for GREEN queue |
| `CONFIDENCE_MEDIUM` | `0.50` | Below this → RED queue |
| `BOOTSTRAP_INITIAL_SIZE` | `200` | Min gold entries before first bootstrap |
| `RETRAIN_EVERY_N` | `500` | Retrain interval |
| `UNCERTAINTY_THRESHOLD` | `0.40` | High-uncertainty cutoff |

### Augmentation

| Variable | Default | Description |
|----------|---------|-------------|
| `AUG_ENTITY_SWAP_MULTIPLIER` | `4` | Variants per entry (Entity Swap) |
| `AUG_TEMPLATE_FILL_COUNT` | `5000` | Template-filled entries to generate |
| `AUG_EDA_MULTIPLIER` | `3` | Variants per entry (EDA) |
| `AUG_EDA_ALPHA` | `0.1` | Word-change fraction for EDA |

### spaCy Model

| Variable | Default | Description |
|----------|---------|-------------|
| `SPACY_MODEL` | `en_core_web_lg` | spaCy pipeline name |

> To switch to GPU-accelerated model: change to `en_core_web_trf`

---

## 12. CLI Flags Reference

### `run_layer1.py`

```
usage: run_layer1.py [-h] [--input PATH] [--output PATH] [--limit N] [--no-resume]

Run Layer 1 Pre-Annotation

optional arguments:
  --input PATH    Input JSONL path (default: data/original_data.jsonl)
  --output PATH   Output JSONL path (default: data/pre_annotated.jsonl)
  --limit N       Process only first N entries
  --no-resume     Don't skip already processed entries
```

### `run_layer2.py`

```
usage: run_layer2.py [-h] [--input PATH] [--green PATH] [--yellow PATH] [--red PATH]

Run Layer 2 Routing

optional arguments:
  --input PATH    Pre-annotated JSONL path (default: data/pre_annotated.jsonl)
  --green PATH    Green queue output path
  --yellow PATH   Yellow queue output path
  --red PATH      Red queue output path
```

### `run_bootstrap_train.py`

```
usage: run_bootstrap_train.py [-h] {train,score} ...

Layer 3 Bootstrap Training & Scoring

subcommands:
  train    Train bootstrap NER model
    --gold PATH      Gold standard input (default: data/gold_standard.jsonl)
    --output DIR     Model output directory (default: models/bootstrap/)
    --n-iter N       Training epochs (default: 10)

  score    Score queues by uncertainty & re-sort
    --model DIR      Model directory (default: latest in models/bootstrap/)
    --yellow PATH    Yellow queue to re-score
    --red PATH       Red queue to re-score
```

### `run_augmentation.py`

```
usage: run_augmentation.py [-h] --strategy {swap,template,eda,all}
                           [--multiplier N] [--count N] [--alpha F]
                           [--output PATH]

Run data augmentation on gold-standard entries

required arguments:
  --strategy      Augmentation strategy: swap, template, eda, or all

optional arguments:
  --multiplier N  Variants per entry (swap: 4, eda: 3)
  --count N       Template-filled entries to generate (default: 5000)
  --alpha F       EDA word-change fraction (default: 0.1)
  --output PATH   Output path (default: data/augmented_data.jsonl)
```

### `export_report_stats.py`

```
usage: export_report_stats.py [-h] [--output-dir DIR]

Export JSON + Markdown report

optional arguments:
  --output-dir DIR   Output directory (default: logs/)
```

### `dataset_analysis.py`

```
usage: dataset_analysis.py [-h] [--output-dir DIR] [--no-charts]

SAHA-AL Dataset Analysis & Statistics Generator

optional arguments:
  --output-dir DIR   Output directory for charts, tables, reports (default: reports/)
  --no-charts        Skip chart generation (text reports only)
```

**Generated:** 10 PNG charts, 5 LaTeX tables, Markdown report, JSON stats.

---

## 13. Troubleshooting

### Common Issues

#### ❌ `ModuleNotFoundError: No module named 'saha_al'`

**Cause:** Python can't find the package. You need to run from the PROJECT root.

**Fix:**
```bash
cd /home/ayush/Desktop/sem6/inlp/PROJECT
source .venv/bin/activate
python saha_al/scripts/run_layer1.py   # ✅ run from PROJECT root
```

#### ❌ `OSError: [E050] Can't find model 'en_core_web_lg'`

**Cause:** spaCy model not installed.

**Fix:**
```bash
source .venv/bin/activate
python -m spacy download en_core_web_lg
```

#### ❌ `externally-managed-environment` error on pip install

**Cause:** Running pip outside a virtual environment on Debian/Ubuntu 23+.

**Fix:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r saha_al/requirements.txt
```

#### ❌ `FileNotFoundError: data/original_data.jsonl`

**Cause:** Source data file missing or wrong path.

**Fix:**
```bash
ls -la saha_al/data/original_data.jsonl
# If missing:
ln -sf /home/ayush/Desktop/sem6/inlp/PROJECT/original_data.jsonl saha_al/data/original_data.jsonl
```

#### ❌ Streamlit shows "No entries in queue"

**Cause:** Layer 2 hasn't been run yet, or queue files are empty.

**Fix:** Run Layer 1 then Layer 2 first:
```bash
python saha_al/scripts/run_layer1.py --limit 500
python saha_al/scripts/run_layer2.py
streamlit run saha_al/layer4_app.py
```

#### ❌ `ValueError: need at least 200 gold entries for bootstrap training`

**Cause:** Not enough annotated entries yet.

**Fix:** Continue annotating in the Streamlit interface until you have ≥200 gold entries, then retry.

#### ❌ Layer 1 is very slow (< 1 entry/sec)

**Cause:** spaCy `en_core_web_lg` is CPU-bound.

**Fix options:**
1. Use `--limit` to process in batches
2. Switch to `en_core_web_trf` with GPU (edit `config.py`)
3. Let it run overnight — it's resume-safe

#### ❌ Augmentation produces 0 entries

**Cause:** `gold_standard.jsonl` is empty or doesn't exist.

**Fix:** Ensure you have annotated entries before running augmentation:
```bash
wc -l saha_al/data/gold_standard.jsonl
# Should be > 0
```

#### ❌ Port 8501 already in use (Streamlit)

**Cause:** Another Streamlit instance is running.

**Fix:**
```bash
# Kill existing Streamlit
pkill -f streamlit

# Or use a different port
streamlit run saha_al/layer4_app.py --server.port 8502
```

---

## 14. FAQ

**Q: Can multiple annotators work simultaneously?**
A: Yes. Each annotator launches their own Streamlit instance (use different ports: `--server.port 8502`, `8503`, etc.). The gold standard uses append-only JSONL, so concurrent writes are safe. Each annotator enters their name in the sidebar.

**Q: How long does the full pipeline take?**
A: Setup ~15 min. Layer 1 (120K entries) ~12-18 hrs. Layer 2 ~2 min. Annotation ~2.5 weeks (4 annotators, ~10K entries). Augmentation ~10-15 min. Total wall time: ~2.5 weeks (annotation is the bottleneck).

**Q: What if I need to re-run Layer 1 after changing regex patterns?**
A: Use `--no-resume` to force reprocessing:
```bash
python saha_al/scripts/run_layer1.py --no-resume
```

**Q: Can I run augmentation on a partial gold standard?**
A: Yes. Run it at any checkpoint (e.g., after 1K entries). The augmented data scales linearly with gold size. You can re-run it later with more gold entries.

**Q: How do I change the augmentation multipliers?**
A: Either via CLI flags (`--multiplier 6`) or by editing `config.py`:
```python
AUG_ENTITY_SWAP_MULTIPLIER = 6   # was 4
AUG_EDA_MULTIPLIER = 5           # was 3
```

**Q: Do I need a GPU?**
A: No. The entire pipeline runs on CPU. A GPU only speeds up spaCy with `en_core_web_trf` (~3x faster for Layer 1) and bootstrap training (~2x faster for Layer 3).

**Q: What's the minimum viable run?**
A: For a quick demo:
```bash
python saha_al/scripts/run_layer1.py --limit 100
python saha_al/scripts/run_layer2.py
streamlit run saha_al/layer4_app.py
```
Annotate ~50 entries, then run augmentation to show the full pipeline works.

**Q: Where are backups stored?**
A: `saha_al/data/backups/gold_standard_YYYYMMDD_HHMMSS.jsonl` — auto-created every 50 annotations. Manual backup via the sidebar button in Streamlit.

---

> **Questions?** Check `README.md` for architecture details, or inspect `config.py` for all configurable parameters.
