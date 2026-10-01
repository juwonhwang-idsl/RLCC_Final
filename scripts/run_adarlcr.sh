#!/usr/bin/env bash
# Copyright 2026 Juwon Hwang, Hanyang University
# Licensed under the ISC License (see LICENSE-RLCC).

# R05 (AdaRFT difficulty sampler + RLCR reward, Qwen3-1.7B).
#
# Adapted from the A100 original: that version launched via
# `accelerate launch --config_file configs/a100-3090-match/accelerate.yaml`, a file that is not
# part of this release. Since that config only disabled DeepSpeed/FSDP for a single-GPU run
# (confirmed by ACCELERATE_USE_DEEPSPEED=false / ACCELERATE_USE_FSDP=false / deepspeed: null in
# the training config), this script instead uses the same env-var single-process launch pattern
# as run_training.sh, which is equivalent and matches how this file's own imports
# (training-rlcc/rl_runner.py's logger/logger_setup/model_init) are normally invoked.

set -euo pipefail
GPU="${1:?Usage: bash scripts/run_adarlcr.sh GPU [extra trainer arguments, e.g. --max_steps 2]}"
shift 1
PORT="${RLCC_MASTER_PORT:-29507}"

cd training-rlcc
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
python -m accelerate.commands.launch --num_processes 1 rl_runner_adarlcr.py \
  --config "../configs/rlcc/R05-rlcc-adarlcr-1.7B-bigmath.yaml" "$@"

# Example:
#   bash run_adarlcr.sh 0
#   bash run_adarlcr.sh 0 --max_steps 2   # smoke test
