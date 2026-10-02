#!/usr/bin/env python3
"""
Export report statistics from the gold standard and annotation logs.
Generates a summary JSON + human-readable Markdown report.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saha_al.config import (
    GOLD_STANDARD_PATH,
    ANNOTATION_LOG_PATH,
    FLAGGED_PATH,
    SKIPPED_PATH,
    GREEN_QUEUE_PATH,
    YELLOW_QUEUE_PATH,
    RED_QUEUE_PATH,
    BASE_DIR,
)
from saha_al.utils.io_helpers import read_jsonl, count_lines


def generate_report(output_dir: str = None) -> dict:
    """Generate a comprehensive report dict."""
    output_dir = output_dir or os.path.join(BASE_DIR, "logs")

    # Load data
    gold = read_jsonl(GOLD_STANDARD_PATH)
    logs = read_jsonl(ANNOTATION_LOG_PATH)
    flagged = read_jsonl(FLAGGED_PATH)
    skipped = read_jsonl(SKIPPED_PATH)

    # Entity type distribution
    entity_counts = Counter()
    total_entities = 0
    for entry in gold:
        for ent in entry.get("entities", []):
            entity_counts[ent.get("entity_type", "UNKNOWN")] += 1
            total_entities += 1

    # Annotator stats
    annotator_counts = Counter()
    for entry in gold:
        ann = entry.get("metadata", {}).get("annotator", "unknown")
        annotator_counts[ann] += 1

    # Queue source stats
    queue_source = Counter()
    for entry in gold:
        q = entry.get("metadata", {}).get("source_queue", "UNKNOWN")
        queue_source[q] += 1

    # Quality warnings
    total_warnings = sum(
        entry.get("metadata", {}).get("quality_warnings", 0) for entry in gold
    )

    # Action distribution from logs
    action_counts = Counter()
    for log_entry in logs:
        action_counts[log_entry.get("action", "UNKNOWN")] += 1

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "gold_standard_entries": len(gold),
            "total_entities_annotated": total_entities,
            "flagged_entries": len(flagged),
            "skipped_entries": len(skipped),
            "total_quality_warnings": total_warnings,
            "total_log_actions": len(logs),
        },
        "queue_sizes": {
            "green": count_lines(GREEN_QUEUE_PATH),
            "yellow": count_lines(YELLOW_QUEUE_PATH),
            "red": count_lines(RED_QUEUE_PATH),
        },
        "entity_type_distribution": dict(entity_counts.most_common()),
        "annotator_contributions": dict(annotator_counts.most_common()),
        "source_queue_distribution": dict(queue_source.most_common()),
        "action_distribution": dict(action_counts.most_common()),
    }

    # Write JSON report
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"report_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Write Markdown report
    md_path = os.path.join(output_dir, f"report_{timestamp}.md")
    md = _build_markdown(report)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"Reports saved to:\n  {json_path}\n  {md_path}")
    return report


def _build_markdown(report: dict) -> str:
    """Build a human-readable Markdown report."""
    s = report["summary"]
    lines = [
        "# SAHA-AL Annotation Report",
        f"_Generated: {report['generated_at']}_",
        "",
        "## Summary",
        f"- **Gold standard entries:** {s['gold_standard_entries']}",
        f"- **Total entities annotated:** {s['total_entities_annotated']}",
        f"- **Flagged entries:** {s['flagged_entries']}",
        f"- **Skipped entries:** {s['skipped_entries']}",
        f"- **Quality warnings:** {s['total_quality_warnings']}",
        "",
        "## Queue Sizes",
    ]
    for q, cnt in report["queue_sizes"].items():
        lines.append(f"- {q.upper()}: {cnt}")

    lines.extend(["", "## Entity Type Distribution", "| Type | Count |", "|------|-------|"])
    for etype, cnt in report["entity_type_distribution"].items():
        lines.append(f"| {etype} | {cnt} |")

    lines.extend(["", "## Annotator Contributions", "| Annotator | Count |", "|-----------|-------|"])
    for ann, cnt in report["annotator_contributions"].items():
        lines.append(f"| {ann} | {cnt} |")

    lines.extend(["", "## Actions", "| Action | Count |", "|--------|-------|"])
    for act, cnt in report["action_distribution"].items():
        lines.append(f"| {act} | {cnt} |")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Annotation Report")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()
    generate_report(output_dir=args.output_dir)
