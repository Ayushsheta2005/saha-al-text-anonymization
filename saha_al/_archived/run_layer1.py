#!/usr/bin/env python3
"""Run Layer 1 — Pre-Annotation on all original data."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saha_al.layer1_preannotation import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Layer 1 Pre-Annotation")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL path")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N entries")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already processed entries")
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
