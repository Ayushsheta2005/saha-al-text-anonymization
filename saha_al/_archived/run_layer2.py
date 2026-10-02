#!/usr/bin/env python3
"""Run Layer 2 — Route pre-annotated entries into GREEN/YELLOW/RED queues."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saha_al.layer2_routing import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Layer 2 Routing")
    parser.add_argument("--input", type=str, default=None, help="Pre-annotated JSONL path")
    parser.add_argument("--green", type=str, default=None, help="Green queue output path")
    parser.add_argument("--yellow", type=str, default=None, help="Yellow queue output path")
    parser.add_argument("--red", type=str, default=None, help="Red queue output path")
    args = parser.parse_args()

    main(
        input_path=args.input,
        green_path=args.green,
        yellow_path=args.yellow,
        red_path=args.red,
    )
