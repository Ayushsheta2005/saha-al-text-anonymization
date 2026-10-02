"""
Inter-Annotator Agreement (IAA)
Computes entity-level agreement between two annotators' outputs.
Used for quality reporting / midterm stats.
"""


def compute_entity_level_agreement(
    entries_a: list[dict],
    entries_b: list[dict],
    key: str = "entry_id",
) -> dict:
    """
    Compute entity-level precision, recall, F1 between two annotator sets.
    
    Each entry is expected to have:
      - entry_id   (or whatever `key` is)
      - entities    list of dicts with {start, end, entity_type}
    
    Returns dict with precision, recall, f1, matched, total_a, total_b.
    """
    # Index entries by key
    idx_b = {e.get(key): e for e in entries_b}

    matched = 0
    total_a = 0
    total_b = 0

    for entry_a in entries_a:
        eid = entry_a.get(key)
        entry_b = idx_b.get(eid)
        if entry_b is None:
            total_a += len(entry_a.get("entities", []))
            continue

        ents_a = entry_a.get("entities", [])
        ents_b = entry_b.get("entities", [])
        total_a += len(ents_a)
        total_b += len(ents_b)

        # Build set of (start, end, entity_type) for comparison
        set_a = {
            (e.get("start"), e.get("end"), e.get("entity_type"))
            for e in ents_a
        }
        set_b = {
            (e.get("start"), e.get("end"), e.get("entity_type"))
            for e in ents_b
        }
        matched += len(set_a & set_b)

    # Count total_b for entries only in B
    seen_a_ids = {e.get(key) for e in entries_a}
    for entry_b in entries_b:
        if entry_b.get(key) not in seen_a_ids:
            total_b += len(entry_b.get("entities", []))

    precision = matched / total_a if total_a else 0.0
    recall = matched / total_b if total_b else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": matched,
        "total_a": total_a,
        "total_b": total_b,
    }
