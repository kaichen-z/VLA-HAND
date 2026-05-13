#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
REPO="${REPO:-/home/chonghej/scratch/chonghej/VLA-HAND}"
PYTHON="${PYTHON:-/scratch/chonghej/conda_envs/vitra/bin/python}"
OUT="${OUT:-runs/tactile_dps_opentouch_encoder_v1_smoke}"
TRAIN_CACHE="${TRAIN_CACHE:-${REPO}/runs/online_touch_editor_large/cache_train}"
TEST_CACHE="${TEST_CACHE:-${REPO}/runs/online_touch_editor_large/cache_test}"

cd "${REPO}"

rm -rf "${OUT}"

"${PYTHON}" scripts/train_tactile_measurement_model.py \
  --train_cache "${TRAIN_CACHE}" \
  --test_cache "${TEST_CACHE}" \
  --output_dir "${OUT}" \
  --max_train_samples 256 \
  --max_test_samples 64 \
  --batch_size 32 \
  --num_workers 0 \
  --encoder_steps 100 \
  --forward_steps 100 \
  --eval_every 50 \
  --horizon 8

"${PYTHON}" scripts/evaluate_tactile_dps_replay.py \
  --cache_root "${TEST_CACHE}" \
  --checkpoint "${OUT}/best.pt" \
  --output_path "${OUT}/eval_smoke.json" \
  --max_samples 64 \
  --batch_size 32 \
  --num_workers 0 \
  --num_guidance_steps 5 \
  --guidance_lr 0.03 \
  --edit_indices 3 5 \
  --ablations matched shuffled_touch zero_touch stats_only
