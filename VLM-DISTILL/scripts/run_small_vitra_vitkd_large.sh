#!/usr/bin/env bash
set -euo pipefail

export PATH="${CONDA_ENV_BIN:-/home/hannahyao24/miniconda3/envs/myproject/bin}:$PATH"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

torchrun --nproc_per_node=1 --standalone \
  scripts/train_vlm_distill.py \
  --config vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands_large.json
