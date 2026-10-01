#!/usr/bin/env python
# Copyright 2026 Juwon Hwang, Hanyang University
# Licensed under the Apache License, Version 2.0 (see LICENSE-RLCC).

"""Build a fully confidence-sorted (descending) RLCC training dataset.

Unlike build_rlcc_dataset.py (which splits into a
confident/deferred bucket at a fixed threshold), this sorts every example by
its RLCR confidence, highest first, and shuffles ties so rows that share the
same confidence value aren't left in their original dataset order.

Confidence values only ever affect row order here -- the "prompt" column that
actually gets tokenized and shown to the model is built later (in
dataset_processing.py) from "problem"/"question" alone, so no confidence
information leaks into the model's input.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk


def read_confidence(row, column):
    value = row[column]
    while isinstance(value, list):
        if not value:
            return 0.0
        value = value[0]
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--confidence_column", type=str, required=True)
    parser.add_argument(
        "--keep_columns",
        type=str,
        default="problem,answer,source,domain,llama8b_solve_rate,id",
        help="Comma-separated columns to keep from the input dataset (plus the confidence column itself).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_from_disk(str(args.input_path))
    confidences = [read_confidence(row, args.confidence_column) for row in dataset]

    rng = random.Random(args.seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)  # randomize tie order before the stable sort below
    order.sort(key=lambda i: confidences[i], reverse=True)

    keep_columns = [c.strip() for c in args.keep_columns.split(",") if c.strip()]
    drop_columns = [c for c in dataset.column_names if c not in keep_columns]

    ordered = dataset.select(order)
    ordered = ordered.remove_columns(drop_columns)
    ordered = ordered.add_column("rlcc_confidence", [confidences[i] for i in order])
    ordered = ordered.add_column("rlcc_original_position", order)
    # rlcc_phase is kept for schema parity with build_rlcc_dataset.py; it has no
    # effect on training (rl_runner.py only uses rlcc_preordered_dataset_path).
    # with a fully sorted dataset instead, so the column is filled with a constant
    # for bookkeeping/inspection only -- it is not read by the training path.
    ordered = ordered.add_column("rlcc_phase", ["sorted"] * len(ordered))

    args.output_path.mkdir(parents=True, exist_ok=True)
    ordered.save_to_disk(str(args.output_path))

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "confidence_column": args.confidence_column,
        "total": len(ordered),
        "ordering": "descending_confidence_ties_shuffled",
        "seed": args.seed,
    }
    (args.output_path / "rlcc_order_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
