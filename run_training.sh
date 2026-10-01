#!/usr/bin/env bash
# Copyright 2026 Juwon Hwang, Hanyang University
# Licensed under the ISC License (see LICENSE-RLCC).

# Example single-GPU launch commands. Pick the codebase that matches the config:
#   configs/rl/...    -> training-rl/   (RLVR, RLCR -- no curriculum loading)
#   configs/rlcc/...  -> training-rlcc/ (RLCC, ascending, K-sweep, difficulty-signal, etc.
#                        -- requires rlcc_preordered_dataset_path support)
# Always match the config directory to its paired codebase.

set -euo pipefail
CONFIG="$1"            # e.g. configs/rl/E01-rlvr-1.7B-bigmath.yaml
CODEBASE="$2"           # training-rl or training-rlcc
GPU="${3:-0}"
PORT="${4:-29500}"

cd "$CODEBASE"
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
CUDA_VISIBLE_DEVICES="$GPU" \
python -m accelerate.commands.launch --num_processes 1 rl_runner.py --config "../$CONFIG"

# Example:
#   bash run_training.sh configs/rl/E01-rlvr-1.7B-bigmath.yaml training-rl 0
#   bash run_training.sh configs/rlcc/E03-rlcc-k3-1.7B-bigmath.yaml training-rlcc 0
