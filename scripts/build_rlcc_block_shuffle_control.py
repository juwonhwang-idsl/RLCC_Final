#!/usr/bin/env python
"""Build the RLCC block-shuffle control ordering (Reviewer 2 W2).

Takes an already-built RLCC curriculum dataset (e.g. the default K=3
descending-confidence ordering from build_rlcc_dataset.py) and destroys the
GLOBAL easy->hard trajectory while preserving LOCAL within-block confidence
homogeneity:

  1. Split the curriculum-ordered sequence into consecutive blocks of
     `--block_size` rows (default 32 -- see module docstring caveat below on
     what this block size does and does not correspond to).
  2. NEVER reorder rows within a block -- each block's example order and
     confidence composition is copied verbatim from the input ordering.
  3. Shuffle the *list of blocks* with a fixed seed, then concatenate.

Result: within any 32-row neighborhood, the set and order of confidence
values is byte-identical to the input ordering (same low within-neighborhood
variance), but the block-to-block progression across the whole sequence is
randomized, so any global easy->hard trend or per-group restart pattern is
destroyed.

Note on block_size=32: gradient_accumulation_steps=32 in the RLCC training
configs equals num_generations=32, so RepeatSampler's batch_size works out to
generation_batch_size // num_generations == 1 -- one optimizer step consumes
exactly one unique training example (confirmed via training_args.bin /
trainer_state.json: global_step == training example count, 1:1). So a
"32-example block" here means 32 *consecutive rows in the curriculum
sequence* (~32 consecutive optimizer steps), not one gradient-accumulation
window's worth of distinct examples.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="An already-built curriculum dataset, e.g. "
        "persist/rlcc-bigmathdigits-qwen3-1p7B-ckpt5000-n3/combined",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for shuffling block order. Deliberately distinct from the "
        "curriculum-build seed (0) used by build_rlcc_dataset.py, to avoid "
        "confusion between 'which shuffle' when reading configs later.",
    )
    parser.add_argument(
        "--no_partial_block_in_shuffle",
        action="store_true",
        help="If total rows is not a multiple of block_size, keep the final "
        "partial block fixed at the end instead of including it in the "
        "shuffle pool (default: included in the shuffle like any other block).",
    )
    args = parser.parse_args()

    dataset = load_from_disk(str(args.input_path))
    total = len(dataset)

    block_ranges = [(i, min(i + args.block_size, total)) for i in range(0, total, args.block_size)]
    full_blocks = [b for b in block_ranges if b[1] - b[0] == args.block_size]
    partial_blocks = [b for b in block_ranges if b[1] - b[0] != args.block_size]
    assert len(partial_blocks) <= 1, f"expected at most 1 partial block, got {partial_blocks}"

    rng = random.Random(args.seed)
    if args.no_partial_block_in_shuffle:
        blocks = full_blocks[:]
        rng.shuffle(blocks)
        blocks = blocks + partial_blocks  # fixed at the end, unshuffled
    else:
        blocks = full_blocks + partial_blocks
        rng.shuffle(blocks)

    indices = []
    block_id_per_row = []
    for block_idx, (start, end) in enumerate(blocks):
        indices.extend(range(start, end))
        block_id_per_row.extend([block_idx] * (end - start))

    assert len(indices) == total
    assert sorted(indices) == list(range(total)), "block shuffle must be a permutation, not lose/duplicate rows"

    ordered = dataset.select(indices)
    ordered = ordered.add_column("block_shuffle_original_position", indices)
    ordered = ordered.add_column("block_shuffle_block_id", block_id_per_row)

    args.output_path.mkdir(parents=True, exist_ok=True)
    ordered.save_to_disk(str(args.output_path))

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "total": total,
        "block_size": args.block_size,
        "num_full_blocks": len(full_blocks),
        "last_block_size": (partial_blocks[0][1] - partial_blocks[0][0]) if partial_blocks else args.block_size,
        "num_blocks_total": len(blocks),
        "partial_block_included_in_shuffle": not args.no_partial_block_in_shuffle,
        "method": "rlcc_block_shuffle_control",
        "ordering": "blocks_of_input_ordering_shuffled_intra_block_order_preserved",
        "seed": args.seed,
    }
    (args.output_path / "rlcc_order_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
