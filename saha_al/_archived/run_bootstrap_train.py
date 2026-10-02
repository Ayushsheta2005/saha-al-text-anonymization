#!/usr/bin/env python3
"""Run Layer 3 — Train bootstrap NER model from gold standard."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saha_al.layer3_active_learning import train_bootstrap_model, resort_queues

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer 3 Bootstrap Training & Scoring")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train bootstrap NER model")
    train_p.add_argument("--gold", type=str, default=None)
    train_p.add_argument("--output", type=str, default=None)
    train_p.add_argument("--n-iter", type=int, default=10)

    score_p = sub.add_parser("score", help="Score queues by uncertainty & re-sort")
    score_p.add_argument("--model", type=str, default=None)
    score_p.add_argument("--yellow", type=str, default=None)
    score_p.add_argument("--red", type=str, default=None)

    args = parser.parse_args()

    if args.command == "train":
        train_bootstrap_model(
            gold_path=args.gold,
            output_dir=args.output,
            n_iter=args.n_iter,
        )
    elif args.command == "score":
        resort_queues(
            model_path=args.model,
            yellow_path=args.yellow,
            red_path=args.red,
        )
    else:
        parser.print_help()
