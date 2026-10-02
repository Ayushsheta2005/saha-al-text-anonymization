#!/usr/bin/env python3
"""
SAHA-AL Dataset Analysis & Statistics Generator
═══════════════════════════════════════════════════════════════════════
Generates charts, tables, and statistics for the project report.

Outputs:
  - Entity type distribution (bar chart + pie chart)
  - Source dataset statistics table
  - Routing distribution (GREEN/YELLOW/RED)
  - Confidence score distribution (histogram)
  - Agreement level breakdown
  - Per-annotator contribution chart
  - Augmentation expansion summary
  - Entity co-occurrence heatmap
  - Text length distribution
  - PII density distribution
  - LaTeX-ready table snippets (.tex files)

Usage:
    python scripts/dataset_analysis.py
    python scripts/dataset_analysis.py --input data/gold_standard.jsonl
    python scripts/dataset_analysis.py --output-dir reports/figures/
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

from saha_al.config import (
    ORIGINAL_DATA_PATH,
    PRE_ANNOTATED_PATH,
    GOLD_STANDARD_PATH,
    AUGMENTED_DATA_PATH,
    TRAINING_DATA_PATH,
    GREEN_QUEUE_PATH,
    YELLOW_QUEUE_PATH,
    RED_QUEUE_PATH,
    FLAGGED_PATH,
    SKIPPED_PATH,
    ANNOTATION_LOG_PATH,
    BASE_DIR,
    ENTITY_TYPES,
)
from saha_al.utils.io_helpers import read_jsonl, count_lines


# ─── Color palette for entity types ────────────────────────────────
ENTITY_COLORS = {
    "FULLNAME": "#FF6B6B", "FIRST_NAME": "#FF8E8E", "LAST_NAME": "#FFA5A5",
    "ID_NUMBER": "#4ECDC4", "PASSPORT": "#3BAEA0", "SSN": "#2EA495",
    "PHONE": "#F7DC6F", "EMAIL": "#F0B27A", "ADDRESS": "#82E0AA",
    "DATE": "#85C1E9", "TIME": "#A9CCE3", "LOCATION": "#BB8FCE",
    "ORGANIZATION": "#D7BDE2", "ACCOUNT_NUMBER": "#F9E79F",
    "CREDIT_CARD": "#FADBD8", "ZIPCODE": "#D5F5E3", "TITLE": "#D6DBDF",
    "GENDER": "#E8DAEF", "NUMBER": "#FCF3CF", "OTHER_PII": "#F2D7D5",
    "UNKNOWN": "#BDC3C7",
}
QUEUE_COLORS = {"GREEN": "#27ae60", "YELLOW": "#f39c12", "RED": "#e74c3c"}


def _safe_count(path):
    """Count entries; return 0 if file missing."""
    try:
        return count_lines(path)
    except Exception:
        return 0


# ═════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════
def load_all_data():
    """Load all available pipeline data files."""
    data = {}
    data["original"] = read_jsonl(ORIGINAL_DATA_PATH)
    data["pre_annotated"] = read_jsonl(PRE_ANNOTATED_PATH)
    data["gold"] = read_jsonl(GOLD_STANDARD_PATH)
    data["augmented"] = read_jsonl(AUGMENTED_DATA_PATH)
    data["training"] = read_jsonl(TRAINING_DATA_PATH)
    data["green"] = read_jsonl(GREEN_QUEUE_PATH)
    data["yellow"] = read_jsonl(YELLOW_QUEUE_PATH)
    data["red"] = read_jsonl(RED_QUEUE_PATH)
    data["flagged"] = read_jsonl(FLAGGED_PATH)
    data["skipped"] = read_jsonl(SKIPPED_PATH)
    data["logs"] = read_jsonl(ANNOTATION_LOG_PATH)
    return data


# ═════════════════════════════════════════════════════════════════════
#  STATISTICS COMPUTATION
# ═════════════════════════════════════════════════════════════════════
def compute_statistics(data: dict) -> dict:
    """Compute all report statistics from loaded data."""
    stats = {}

    # ── Dataset sizes ──
    stats["total_original"] = len(data["original"])
    stats["total_pre_annotated"] = len(data["pre_annotated"])
    stats["total_gold"] = len(data["gold"])
    stats["total_augmented"] = len(data["augmented"])
    stats["total_training"] = len(data["training"])
    stats["total_flagged"] = len(data["flagged"])
    stats["total_skipped"] = len(data["skipped"])

    # ── Queue sizes ──
    stats["green_count"] = len(data["green"])
    stats["yellow_count"] = len(data["yellow"])
    stats["red_count"] = len(data["red"])
    stats["total_routed"] = stats["green_count"] + stats["yellow_count"] + stats["red_count"]

    # ── Entity statistics from best available data ──
    # Use gold > pre_annotated > original in priority
    entities_source = data["gold"] if data["gold"] else data["pre_annotated"]
    source_label = "gold" if data["gold"] else "pre_annotated"

    entity_type_counts = Counter()
    entity_source_counts = Counter()
    agreement_counts = Counter()
    confidences = []
    entities_per_entry = []
    text_lengths = []
    pii_densities = []

    for entry in entities_source:
        ents = entry.get("entities", [])
        text = entry.get("original_text", "")
        entities_per_entry.append(len(ents))
        text_lengths.append(len(text))

        total_pii_chars = sum(len(e.get("text", "")) for e in ents)
        pii_densities.append(total_pii_chars / max(len(text), 1))

        for ent in ents:
            entity_type_counts[ent.get("entity_type", "UNKNOWN")] += 1
            entity_source_counts[ent.get("source", "unknown")] += 1
            agreement_counts[ent.get("agreement", "unknown")] += 1
            conf = ent.get("confidence")
            if conf is not None:
                confidences.append(conf)

    stats["entity_type_counts"] = dict(entity_type_counts.most_common())
    stats["entity_source_counts"] = dict(entity_source_counts)
    stats["agreement_counts"] = dict(agreement_counts)
    stats["total_entities"] = sum(entity_type_counts.values())
    stats["avg_entities_per_entry"] = (
        sum(entities_per_entry) / len(entities_per_entry) if entities_per_entry else 0
    )
    stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0
    stats["confidences"] = confidences
    stats["entities_per_entry"] = entities_per_entry
    stats["text_lengths"] = text_lengths
    stats["pii_densities"] = pii_densities
    stats["entities_source_label"] = source_label

    # ── Annotator stats (from gold) ──
    annotator_counts = Counter()
    for entry in data["gold"]:
        ann = entry.get("metadata", {}).get("annotator", "auto")
        annotator_counts[ann] += 1
    stats["annotator_counts"] = dict(annotator_counts)

    # ── Queue source in gold ──
    queue_source = Counter()
    for entry in data["gold"]:
        q = entry.get("metadata", {}).get("source_queue", "UNKNOWN")
        queue_source[q] += 1
    stats["gold_queue_source"] = dict(queue_source)

    # ── Augmentation breakdown ──
    aug_strategy_counts = Counter()
    for entry in data["augmented"]:
        strategy = entry.get("metadata", {}).get("augmentation", {}).get("strategy", "unknown")
        aug_strategy_counts[strategy] += 1
    stats["augmentation_breakdown"] = dict(aug_strategy_counts)

    # ── Entity co-occurrence ──
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for entry in entities_source:
        types_in_entry = set(e.get("entity_type", "UNKNOWN") for e in entry.get("entities", []))
        for t1 in types_in_entry:
            for t2 in types_in_entry:
                cooccurrence[t1][t2] += 1
    stats["cooccurrence"] = {k: dict(v) for k, v in cooccurrence.items()}

    # ── Quality warnings ──
    total_warnings = 0
    warning_types = Counter()
    for entry in data["gold"]:
        warnings = entry.get("metadata", {}).get("warnings", [])
        total_warnings += len(warnings)
        for w in warnings:
            warning_types[w.get("check", "unknown")] += 1
    stats["total_warnings"] = total_warnings
    stats["warning_types"] = dict(warning_types)

    # ── Text length stats ──
    if text_lengths:
        stats["avg_text_length"] = sum(text_lengths) / len(text_lengths)
        stats["min_text_length"] = min(text_lengths)
        stats["max_text_length"] = max(text_lengths)
        stats["median_text_length"] = sorted(text_lengths)[len(text_lengths) // 2]
    else:
        stats["avg_text_length"] = 0
        stats["min_text_length"] = 0
        stats["max_text_length"] = 0
        stats["median_text_length"] = 0

    return stats


# ═════════════════════════════════════════════════════════════════════
#  CHART GENERATION
# ═════════════════════════════════════════════════════════════════════
def generate_charts(stats: dict, output_dir: str):
    """Generate all charts as PNG files."""
    if not HAS_MPL:
        print("⚠️  matplotlib not installed — skipping chart generation.")
        print("   Install: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 150,
        "figure.figsize": (10, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    _chart_entity_distribution(stats, output_dir)
    _chart_entity_pie(stats, output_dir)
    _chart_routing_distribution(stats, output_dir)
    _chart_confidence_histogram(stats, output_dir)
    _chart_agreement_breakdown(stats, output_dir)
    _chart_text_length_distribution(stats, output_dir)
    _chart_pii_density(stats, output_dir)
    _chart_entities_per_entry(stats, output_dir)
    _chart_annotator_contributions(stats, output_dir)
    _chart_augmentation_breakdown(stats, output_dir)
    _chart_cooccurrence_heatmap(stats, output_dir)
    _chart_detection_source(stats, output_dir)

    print(f"✅ Charts saved to: {output_dir}")


def _chart_entity_distribution(stats, out):
    """Horizontal bar chart of entity type counts."""
    counts = stats["entity_type_counts"]
    if not counts:
        return
    types = list(counts.keys())
    values = list(counts.values())
    colors = [ENTITY_COLORS.get(t, "#BDC3C7") for t in types]

    fig, ax = plt.subplots(figsize=(10, max(6, len(types) * 0.4)))
    bars = ax.barh(types, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_title("Entity Type Distribution", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "entity_type_distribution.png"), bbox_inches="tight")
    plt.close()


def _chart_entity_pie(stats, out):
    """Pie chart of top entity types."""
    counts = stats["entity_type_counts"]
    if not counts:
        return

    top_n = 10
    items = list(counts.items())[:top_n]
    other = sum(v for _, v in list(counts.items())[top_n:])
    if other > 0:
        items.append(("OTHER", other))

    labels = [t for t, _ in items]
    sizes = [v for _, v in items]
    colors = [ENTITY_COLORS.get(t, "#BDC3C7") for t in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.85
    )
    for text in autotexts:
        text.set_fontsize(8)
    ax.set_title("Entity Type Proportions (Top 10)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "entity_type_pie.png"), bbox_inches="tight")
    plt.close()


def _chart_routing_distribution(stats, out):
    """Bar chart of GREEN / YELLOW / RED routing."""
    queues = ["GREEN", "YELLOW", "RED"]
    counts = [stats.get("green_count", 0), stats.get("yellow_count", 0), stats.get("red_count", 0)]
    if sum(counts) == 0:
        return
    colors = [QUEUE_COLORS[q] for q in queues]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(queues, counts, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    ax.set_ylabel("Number of Entries")
    ax.set_title("Confidence-Based Routing Distribution", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, counts):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "routing_distribution.png"), bbox_inches="tight")
    plt.close()


def _chart_confidence_histogram(stats, out):
    """Histogram of entity confidence scores."""
    confs = stats.get("confidences", [])
    if not confs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confs, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(x=0.85, color="#27ae60", linestyle="--", linewidth=1.5, label="HIGH threshold (0.85)")
    ax.axvline(x=0.50, color="#e74c3c", linestyle="--", linewidth=1.5, label="MEDIUM threshold (0.50)")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Entity Confidence Score Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confidence_histogram.png"), bbox_inches="tight")
    plt.close()


def _chart_agreement_breakdown(stats, out):
    """Pie chart of agreement levels."""
    agr = stats.get("agreement_counts", {})
    if not agr:
        return
    agr_colors = {"full": "#27ae60", "partial": "#f39c12", "single_source": "#3498db", "unknown": "#bdc3c7"}

    labels = list(agr.keys())
    sizes = list(agr.values())
    colors = [agr_colors.get(l, "#bdc3c7") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.set_title("Detection Agreement Levels", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "agreement_breakdown.png"), bbox_inches="tight")
    plt.close()


def _chart_text_length_distribution(stats, out):
    """Histogram of text lengths."""
    lengths = stats.get("text_lengths", [])
    if not lengths:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.axvline(x=stats["avg_text_length"], color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Mean: {stats['avg_text_length']:.0f} chars")
    ax.set_xlabel("Text Length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Text Length Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "text_length_distribution.png"), bbox_inches="tight")
    plt.close()


def _chart_pii_density(stats, out):
    """Histogram of PII density (% of text that is PII)."""
    densities = stats.get("pii_densities", [])
    if not densities:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([d * 100 for d in densities], bins=40, color="#e67e22", edgecolor="white", alpha=0.85)
    avg_density = sum(densities) / len(densities) * 100
    ax.axvline(x=avg_density, color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"Mean: {avg_density:.1f}%")
    ax.set_xlabel("PII Density (%)")
    ax.set_ylabel("Count")
    ax.set_title("PII Character Density per Entry", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pii_density.png"), bbox_inches="tight")
    plt.close()


def _chart_entities_per_entry(stats, out):
    """Histogram of entities per entry."""
    epe = stats.get("entities_per_entry", [])
    if not epe:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    max_val = min(max(epe) + 1, 20) if epe else 10
    ax.hist(epe, bins=range(0, max_val + 1), color="#1abc9c", edgecolor="white", alpha=0.85)
    avg_epe = sum(epe) / len(epe)
    ax.axvline(x=avg_epe, color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"Mean: {avg_epe:.1f}")
    ax.set_xlabel("Number of Entities")
    ax.set_ylabel("Number of Entries")
    ax.set_title("Entities per Entry Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "entities_per_entry.png"), bbox_inches="tight")
    plt.close()


def _chart_annotator_contributions(stats, out):
    """Bar chart of per-annotator contributions."""
    ann = stats.get("annotator_counts", {})
    if not ann:
        return

    names = list(ann.keys())
    counts = list(ann.values())
    colors = plt.cm.Set2.colors[:len(names)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1)
    ax.set_ylabel("Entries Annotated")
    ax.set_title("Annotator Contributions", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "annotator_contributions.png"), bbox_inches="tight")
    plt.close()


def _chart_augmentation_breakdown(stats, out):
    """Bar chart of augmentation strategies."""
    aug = stats.get("augmentation_breakdown", {})
    if not aug:
        return

    strategies = list(aug.keys())
    counts = list(aug.values())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"][:len(strategies)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, counts, color=colors, edgecolor="white", linewidth=1, width=0.5)
    ax.set_ylabel("Generated Entries")
    ax.set_title("Augmentation Strategy Breakdown", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "augmentation_breakdown.png"), bbox_inches="tight")
    plt.close()


def _chart_cooccurrence_heatmap(stats, out):
    """Entity type co-occurrence heatmap."""
    cooc = stats.get("cooccurrence", {})
    if not cooc or not HAS_NP:
        return

    # Only show types that actually appear
    types = sorted(set(cooc.keys()))
    if len(types) < 2:
        return

    matrix = np.zeros((len(types), len(types)))
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            matrix[i, j] = cooc.get(t1, {}).get(t2, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 0.6), max(6, len(types) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(types)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(types, fontsize=8)
    ax.set_title("Entity Type Co-occurrence Heatmap", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "cooccurrence_heatmap.png"), bbox_inches="tight")
    plt.close()


def _chart_detection_source(stats, out):
    """Pie chart of detection sources (spacy vs regex vs both)."""
    src = stats.get("entity_source_counts", {})
    if not src:
        return

    labels = list(src.keys())
    sizes = list(src.values())
    colors = {"spacy": "#3498db", "regex": "#e67e22", "both": "#27ae60", "augmentation": "#9b59b6"}
    c = [colors.get(l, "#bdc3c7") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=c, autopct="%1.1f%%", startangle=140)
    ax.set_title("Entity Detection Source", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "detection_source.png"), bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  LaTeX TABLE GENERATION
# ═════════════════════════════════════════════════════════════════════
def generate_latex_tables(stats: dict, output_dir: str):
    """Generate LaTeX table snippets for the report."""
    os.makedirs(output_dir, exist_ok=True)

    _latex_dataset_overview(stats, output_dir)
    _latex_entity_distribution(stats, output_dir)
    _latex_routing_table(stats, output_dir)
    _latex_augmentation_table(stats, output_dir)
    _latex_text_stats(stats, output_dir)

    print(f"✅ LaTeX tables saved to: {output_dir}")


def _latex_dataset_overview(stats, out):
    """Generate dataset overview table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Dataset Overview}",
        r"\label{tab:dataset-overview}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\midrule",
        f"Source entries & {stats['total_original']:,} \\\\",
        f"Pre-annotated entries & {stats['total_pre_annotated']:,} \\\\",
        f"Gold-standard entries & {stats['total_gold']:,} \\\\",
        f"Augmented entries & {stats['total_augmented']:,} \\\\",
        f"Training set (gold + aug) & {stats['total_training']:,} \\\\",
        f"Flagged entries & {stats['total_flagged']:,} \\\\",
        f"Skipped entries & {stats['total_skipped']:,} \\\\",
        r"\midrule",
        f"Total entities (gold) & {stats['total_entities']:,} \\\\",
        f"Avg entities/entry & {stats['avg_entities_per_entry']:.2f} \\\\",
        f"Avg confidence & {stats['avg_confidence']:.3f} \\\\",
        f"Entity types & {len(stats['entity_type_counts'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(out, "table_dataset_overview.tex"), "w") as f:
        f.write("\n".join(lines))


def _latex_entity_distribution(stats, out):
    """Generate entity type distribution table."""
    counts = stats["entity_type_counts"]
    total = stats["total_entities"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Entity Type Distribution}",
        r"\label{tab:entity-distribution}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Entity Type} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]
    for etype, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        lines.append(f"\\texttt{{{etype}}} & {count:,} & {pct:.1f}\\% \\\\")
    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total:,}}} & \\textbf{{100.0\\%}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(os.path.join(out, "table_entity_distribution.tex"), "w") as f:
        f.write("\n".join(lines))


def _latex_routing_table(stats, out):
    """Generate routing distribution table."""
    total = stats["total_routed"] or 1
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Confidence-Based Routing Distribution}",
        r"\label{tab:routing}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Queue} & \textbf{Action} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
        f"\\textcolor{{green}}{{GREEN}} & Auto-approved & {stats['green_count']:,} & {stats['green_count']/total*100:.1f}\\% \\\\",
        f"\\textcolor{{orange}}{{YELLOW}} & Human review & {stats['yellow_count']:,} & {stats['yellow_count']/total*100:.1f}\\% \\\\",
        f"\\textcolor{{red}}{{RED}} & Expert review & {stats['red_count']:,} & {stats['red_count']/total*100:.1f}\\% \\\\",
        r"\midrule",
        f"\\textbf{{Total}} & & \\textbf{{{total:,}}} & \\textbf{{100.0\\%}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(out, "table_routing.tex"), "w") as f:
        f.write("\n".join(lines))


def _latex_augmentation_table(stats, out):
    """Generate augmentation summary table."""
    aug = stats["augmentation_breakdown"]
    gold = stats["total_gold"]
    total_aug = stats["total_augmented"]
    total_train = stats["total_training"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Data Augmentation Summary}",
        r"\label{tab:augmentation}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{Reference} & \textbf{Generated} & \textbf{Multiplier} \\",
        r"\midrule",
        f"Entity Swap & Dai \\& Adel, COLING 2020 & {aug.get('entity_swap', 0):,} & $\\times 4$ \\\\",
        f"Template Fill & Anaby-Tavor et al., AAAI 2020 & {aug.get('template_fill', 0):,} & $\\times 5$ \\\\",
        f"EDA & Wei \\& Zou, EMNLP 2019 & {aug.get('eda', 0):,} & $\\times 3$ \\\\",
        r"\midrule",
        f"Gold standard & — & {gold:,} & $\\times 1$ \\\\",
        f"\\textbf{{Total augmented}} & & \\textbf{{{total_aug:,}}} & $\\times {total_aug/max(gold,1):.0f}$ \\\\",
        f"\\textbf{{Training set}} & & \\textbf{{{total_train:,}}} & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(out, "table_augmentation.tex"), "w") as f:
        f.write("\n".join(lines))


def _latex_text_stats(stats, out):
    """Generate text statistics table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Text and Entity Statistics}",
        r"\label{tab:text-stats}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\midrule",
        f"Mean text length & {stats['avg_text_length']:.0f} chars \\\\",
        f"Median text length & {stats['median_text_length']} chars \\\\",
        f"Min text length & {stats['min_text_length']} chars \\\\",
        f"Max text length & {stats['max_text_length']} chars \\\\",
        r"\midrule",
        f"Mean entities/entry & {stats['avg_entities_per_entry']:.2f} \\\\",
        f"Mean confidence & {stats['avg_confidence']:.3f} \\\\",
        f"Mean PII density & {sum(stats['pii_densities'])/max(len(stats['pii_densities']),1)*100:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(out, "table_text_stats.tex"), "w") as f:
        f.write("\n".join(lines))


# ═════════════════════════════════════════════════════════════════════
#  MARKDOWN REPORT
# ═════════════════════════════════════════════════════════════════════
def generate_markdown_report(stats: dict, output_dir: str):
    """Generate a human-readable Markdown summary."""
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append("# SAHA-AL Dataset Analysis Report")
    lines.append(f"\n> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Dataset Overview\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Source entries | {stats['total_original']:,} |")
    lines.append(f"| Pre-annotated | {stats['total_pre_annotated']:,} |")
    lines.append(f"| Gold standard | {stats['total_gold']:,} |")
    lines.append(f"| Augmented | {stats['total_augmented']:,} |")
    lines.append(f"| Training set | {stats['total_training']:,} |")
    lines.append(f"| Flagged | {stats['total_flagged']:,} |")
    lines.append(f"| Skipped | {stats['total_skipped']:,} |")

    lines.append(f"\n## Entity Statistics (from {stats['entities_source_label']})\n")
    lines.append(f"- **Total entities:** {stats['total_entities']:,}")
    lines.append(f"- **Avg entities/entry:** {stats['avg_entities_per_entry']:.2f}")
    lines.append(f"- **Avg confidence:** {stats['avg_confidence']:.3f}")
    lines.append(f"- **Entity types in use:** {len(stats['entity_type_counts'])}")

    lines.append(f"\n### Entity Type Distribution\n")
    lines.append("| Entity Type | Count | % |")
    lines.append("|-------------|-------|---|")
    total = stats["total_entities"]
    for etype, count in sorted(stats["entity_type_counts"].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        lines.append(f"| `{etype}` | {count:,} | {pct:.1f}% |")

    lines.append(f"\n## Routing Distribution\n")
    lines.append(f"| Queue | Count | % |")
    lines.append(f"|-------|-------|---|")
    tr = stats["total_routed"] or 1
    lines.append(f"| 🟢 GREEN | {stats['green_count']:,} | {stats['green_count']/tr*100:.1f}% |")
    lines.append(f"| 🟡 YELLOW | {stats['yellow_count']:,} | {stats['yellow_count']/tr*100:.1f}% |")
    lines.append(f"| 🔴 RED | {stats['red_count']:,} | {stats['red_count']/tr*100:.1f}% |")

    lines.append(f"\n## Text Statistics\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Avg length | {stats['avg_text_length']:.0f} chars |")
    lines.append(f"| Median length | {stats['median_text_length']} chars |")
    lines.append(f"| Min length | {stats['min_text_length']} chars |")
    lines.append(f"| Max length | {stats['max_text_length']} chars |")
    pii_d = stats["pii_densities"]
    avg_d = sum(pii_d) / len(pii_d) * 100 if pii_d else 0
    lines.append(f"| Avg PII density | {avg_d:.1f}% |")

    if stats["augmentation_breakdown"]:
        lines.append(f"\n## Augmentation Breakdown\n")
        lines.append(f"| Strategy | Count |")
        lines.append(f"|----------|-------|")
        for s, c in stats["augmentation_breakdown"].items():
            lines.append(f"| {s} | {c:,} |")

    if stats["annotator_counts"]:
        lines.append(f"\n## Annotator Contributions\n")
        lines.append(f"| Annotator | Entries |")
        lines.append(f"|-----------|---------|")
        for a, c in sorted(stats["annotator_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"| {a} | {c:,} |")

    report_path = os.path.join(output_dir, "dataset_analysis_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Markdown report saved to: {report_path}")


# ═════════════════════════════════════════════════════════════════════
#  JSON DUMP
# ═════════════════════════════════════════════════════════════════════
def save_stats_json(stats: dict, output_dir: str):
    """Save raw statistics as JSON for other tools to consume."""
    os.makedirs(output_dir, exist_ok=True)
    # Remove non-serializable lists (too large)
    export = {k: v for k, v in stats.items()
              if k not in ("confidences", "entities_per_entry", "text_lengths", "pii_densities")}
    path = os.path.join(output_dir, "dataset_statistics.json")
    with open(path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"✅ Stats JSON saved to: {path}")


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SAHA-AL Dataset Analysis & Statistics Generator"
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for charts, tables, and reports")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation (text reports only)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(BASE_DIR, "reports")
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")

    print("═" * 60)
    print("  SAHA-AL Dataset Analysis")
    print("═" * 60)
    print()

    print("Loading data...")
    data = load_all_data()

    # Print quick summary of what's available
    for name, entries in data.items():
        if entries:
            print(f"  ✓ {name}: {len(entries):,} entries")
        else:
            print(f"  ✗ {name}: (empty / not found)")
    print()

    print("Computing statistics...")
    stats = compute_statistics(data)
    print(f"  Total entities: {stats['total_entities']:,}")
    print(f"  Entity types: {len(stats['entity_type_counts'])}")
    print(f"  Avg entities/entry: {stats['avg_entities_per_entry']:.2f}")
    print()

    # Generate all outputs
    if not args.no_charts:
        print("Generating charts...")
        generate_charts(stats, figures_dir)

    print("Generating LaTeX tables...")
    generate_latex_tables(stats, tables_dir)

    print("Generating Markdown report...")
    generate_markdown_report(stats, output_dir)

    print("Saving statistics JSON...")
    save_stats_json(stats, output_dir)

    print()
    print("═" * 60)
    print(f"  All outputs saved to: {output_dir}")
    print("═" * 60)


if __name__ == "__main__":
    main()
