"""
Entity Detection Merger
Combines spaCy NER and Regex detections into a unified entity list.
Resolves overlapping spans, assigns agreement levels for routing.
"""


def merge_detections(spacy_entities: list, regex_entities: list) -> list:
    """
    Merge entity lists from spaCy and Regex.
    
    Conflict resolution:
      - Identical spans + type → "full" agreement (highest confidence wins)
      - Overlapping but different span/type → "partial" (higher priority wins)
      - No overlap → "single_source"
    
    Returns a deduplicated, sorted list of entity dicts.
    """
    # Deep-copy to avoid mutating originals
    spacy_ents = [dict(e) for e in spacy_entities]
    regex_ents = [dict(e) for e in regex_entities]

    for e in spacy_ents:
        e.setdefault("source", "spacy")
        e.setdefault("priority", 5)
        e.setdefault("confidence", 0.85)

    for e in regex_ents:
        e.setdefault("source", "regex")

    combined = spacy_ents + regex_ents
    # Sort by start position ascending, then by span length descending (longer first)
    combined.sort(key=lambda x: (x.get("start", 0), -(x.get("end", 0) - x.get("start", 0))))

    merged = []
    used = set()

    for i, e1 in enumerate(combined):
        if i in used:
            continue

        # Collect all entities overlapping with e1
        cluster = [e1]
        for j, e2 in enumerate(combined):
            if j <= i or j in used:
                continue
            if _spans_overlap(e1, e2):
                cluster.append(e2)
                used.add(j)

        used.add(i)

        if len(cluster) == 1:
            e1["agreement"] = "single_source"
            merged.append(e1)
        else:
            merged.append(_resolve_cluster(cluster))

    # Sort final list by start position
    merged.sort(key=lambda x: x.get("start", 0))
    return merged


def _resolve_cluster(cluster: list) -> dict:
    """Resolve a cluster of overlapping entity detections."""
    sources = set(e.get("source") for e in cluster)
    types = set(e.get("entity_type") for e in cluster)
    spans = set((e.get("start"), e.get("end")) for e in cluster)

    # Pick the best entity by priority, then confidence
    best = max(cluster, key=lambda x: (x.get("priority", 0), x.get("confidence", 0)))
    best = dict(best)  # copy

    if len(sources) >= 2 and len(types) == 1 and len(spans) == 1:
        # Both sources agree on everything
        best["agreement"] = "full"
        best["source"] = "both"
        best["confidence"] = max(e.get("confidence", 0) for e in cluster)
    elif len(sources) >= 2:
        # Both sources detected something, but they disagree
        best["agreement"] = "partial"
        best["alternatives"] = [
            {
                "text": e.get("text"),
                "start": e.get("start"),
                "end": e.get("end"),
                "entity_type": e.get("entity_type"),
                "source": e.get("source"),
            }
            for e in cluster if e is not best
        ]
    else:
        # Same source had multiple overlapping matches (e.g., two regex patterns)
        best["agreement"] = "single_source"

    return best


def _spans_overlap(e1: dict, e2: dict) -> bool:
    """Check if two character-offset spans overlap."""
    s1, e1_end = e1.get("start", 0), e1.get("end", 0)
    s2, e2_end = e2.get("start", 0), e2.get("end", 0)
    return s1 < e2_end and s2 < e1_end
