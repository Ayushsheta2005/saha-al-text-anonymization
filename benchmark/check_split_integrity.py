#!/usr/bin/env python
"""Verify the frozen benchmark splits are disjoint.

A leaderboard is only as trustworthy as the guarantee that nobody trained on
the test set. This checks that guarantee directly rather than assuming the
split script got it right, and is meant to run in CI on every dataset change.

    python benchmark/check_split_integrity.py --data benchmark/data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def norm(rec: dict) -> str:
    return " ".join(rec.get("original_text", "").split()).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/data")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any overlap (use in CI)")
    args = ap.parse_args()

    d = Path(args.data)
    splits = {n: load(d / f"{n}.jsonl") for n in ("train", "test", "validation")}
    for n, rows in splits.items():
        print(f"{n:<12} {len(rows):>7,} records")

    texts = {n: Counter(norm(r) for r in rows) for n, rows in splits.items()}
    ids = {n: Counter(r.get("entry_id") for r in rows) for n, rows in splits.items()}

    problems = 0
    print()
    for a, b in (("train", "test"), ("train", "validation"), ("test", "validation")):
        shared_text = set(texts[a]) & set(texts[b])
        shared_id = set(ids[a]) & set(ids[b])
        pct = 100 * len(shared_text) / max(1, len(splits[b]))
        flag = "  <-- LEAK" if shared_text or shared_id else ""
        print(f"{a:>5} n {b:<11} text={len(shared_text):>4} ({pct:.2f}% of {b})  "
              f"entry_id={len(shared_id):>4}{flag}")
        problems += len(shared_text) + len(shared_id)

    print()
    for n, c in texts.items():
        dupes = sum(v - 1 for v in c.values() if v > 1)
        if dupes:
            print(f"{n}: {dupes:,} duplicate original_text within the split")
            problems += dupes

    # Per-type coverage: a benchmark advertising 21 entity types cannot report
    # per-type leakage if the frozen split labels everything UNKNOWN.
    print()
    for n, rows in splits.items():
        t = Counter(e.get("type", "UNKNOWN") for r in rows for e in r.get("entities", []))
        unknown = t.get("UNKNOWN", 0)
        total = sum(t.values())
        if total:
            print(f"{n:<12} {total:>7,} entities, {len(t)} distinct types, "
                  f"{100*unknown/total:.1f}% UNKNOWN")

    if problems:
        print(f"\n{problems} integrity problem(s) found.")
        return 1 if args.strict else 0
    print("\nsplits are disjoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
