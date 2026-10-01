#!/usr/bin/env bash
# Copyright 2026 Juwon Hwang, Hanyang University
# Licensed under the Apache License, Version 2.0 (see LICENSE-RLCC).

# Run an eval config (held-out or OOD) with the shared evaluation/ codebase.
# (Both training-rl/ and training-rlcc/ checkpoints are evaluated the same way --
# evaluation.py does not depend on which codebase trained the model.)

set -euo pipefail
CONFIG="$1"             # e.g. configs/eval_configs/E01-E02-E03-A01-5models-1.7B-bigmath-heldout.json
GPU="${2:-0}"

cd evaluation
CUDA_VISIBLE_DEVICES="$GPU" python evaluation.py --config "../$CONFIG"

# Example:
#   bash run_eval.sh configs/eval_configs/K01-rlcc-k1-heldout.json 0
