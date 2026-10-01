#!/usr/bin/env python
"""Build the RLCC (grouped, repeated-curriculum) RLAA training dataset.

RLCC = split into N groups, sort each group by descending confidence
(--ascending flips this to build the hard-first anti-curriculum ablation
instead, for comparison against RLCC).

Unlike build_rlcc_single_sweep.py (one single descending sweep over all
examples), this:
  1. Shuffles all examples randomly (removes any original dataset order bias).
  2. Splits the shuffled examples into N contiguous groups of roughly equal size.
  3. Sorts each group independently by confidence (descending by default = RLCC;
     ties keep their post-shuffle random order, since the sort is stable).
  4. Concatenates the sorted groups: sorted(group_1) + sorted(group_2) + ...

The result is N repeated easy->hard sweeps instead of one long sweep, so the
model revisits high- and low-confidence examples N times spread across
training instead of only once at the very start/end.

Confidence values only ever affect row order here -- the "prompt" column that
actually gets tokenized and shown to the model is built later (in
dataset_processing.py) from "problem"/"question" alone, so no confidence
information leaks into the model's input.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


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
    parser.add_argument("--num_groups", type=int, required=True)
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort each group ascending (0.0 -> 1.0, hardest first) instead of the default descending.",
    )
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
    shuffled = list(range(len(dataset)))
    rng.shuffle(shuffled)

    n = args.num_groups
    total = len(shuffled)
    group_boundaries = [round(i * total / n) for i in range(n + 1)]
    groups = [shuffled[group_boundaries[i]:group_boundaries[i + 1]] for i in range(n)]

    keep_columns = [c.strip() for c in args.keep_columns.split(",") if c.strip()]
    drop_columns = [c for c in dataset.column_names if c not in keep_columns]

    group_datasets = []
    group_sizes = []
    for group_idx, group_indices in enumerate(groups):
        group_indices_sorted = sorted(group_indices, key=lambda i: confidences[i], reverse=not args.ascending)
        group_ds = dataset.select(group_indices_sorted)
        group_ds = group_ds.remove_columns(drop_columns)
        group_ds = group_ds.add_column("rlcc_confidence", [confidences[i] for i in group_indices_sorted])
        group_ds = group_ds.add_column("rlcc_original_position", group_indices_sorted)
        group_ds = group_ds.add_column("rlcc_group", [group_idx] * len(group_ds))
        group_ds = group_ds.add_column("rlcc_phase", ["sorted"] * len(group_ds))
        group_datasets.append(group_ds)
        group_sizes.append(len(group_ds))

    ordered = concatenate_datasets(group_datasets)

    args.output_path.mkdir(parents=True, exist_ok=True)
    ordered.save_to_disk(str(args.output_path))

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "confidence_column": args.confidence_column,
        "total": len(ordered),
        "num_groups": n,
        "group_sizes": group_sizes,
        "method": "rlcc_ascending_ablation" if args.ascending else "rlcc",
        "ordering": "shuffled_then_grouped_ascending_confidence_per_group" if args.ascending else "shuffled_then_grouped_descending_confidence_per_group",
        "seed": args.seed,
    }
    (args.output_path / "rlcc_order_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
