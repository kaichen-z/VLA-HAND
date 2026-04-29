#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${CONFIG:-vitra/configs/vlm_distill_gigahands_cognition.json}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --standalone \
    scripts/train_vlm_distill.py \
    --config "${CONFIG}" \
    "$@"
