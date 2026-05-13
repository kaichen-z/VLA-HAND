#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-tactile_dps_opentouch_encoder_v1}"
GPU="${GPU:-1}"
REPO="${REPO:-/home/chonghej/scratch/chonghej/VLA-HAND}"
PYTHON="${PYTHON:-/scratch/chonghej/conda_envs/vitra/bin/python}"
OUT="${OUT:-runs/tactile_dps_opentouch_encoder_v1}"
TRAIN_CACHE="${TRAIN_CACHE:-${REPO}/runs/online_touch_editor_large/cache_train}"
TEST_CACHE="${TEST_CACHE:-${REPO}/runs/online_touch_editor_large/cache_test}"

tmux new-session -d -s "${SESSION}" "cd ${REPO} && \
  export CUDA_VISIBLE_DEVICES=${GPU} && \
  mkdir -p ${OUT} && \
  ${PYTHON} scripts/train_tactile_measurement_model.py \
    --train_cache ${TRAIN_CACHE} \
    --test_cache ${TEST_CACHE} \
    --output_dir ${OUT} \
    --batch_size 512 \
    --num_workers 4 \
    --encoder_steps 30000 \
    --forward_steps 60000 \
    --eval_every 2000 \
    --horizon 8 \
    2>&1 | tee ${OUT}/train.log && \
  ${PYTHON} scripts/evaluate_tactile_dps_replay.py \
    --cache_root ${TEST_CACHE} \
    --checkpoint ${OUT}/best.pt \
    --output_path ${OUT}/eval_1k.json \
    --max_samples 1000 \
    --batch_size 128 \
    --num_workers 4 \
    --num_guidance_steps 10 \
    --guidance_lr 0.03 \
    --edit_indices 3 5 \
    --ablations matched shuffled_touch zero_touch stats_only \
    2>&1 | tee ${OUT}/eval_1k.log && \
  ${PYTHON} scripts/evaluate_tactile_dps_replay.py \
    --cache_root ${TEST_CACHE} \
    --checkpoint ${OUT}/best.pt \
    --output_path ${OUT}/eval_high_contact_1k.json \
    --max_samples 1000 \
    --subset high_contact \
    --batch_size 128 \
    --num_workers 4 \
    --num_guidance_steps 10 \
    --guidance_lr 0.03 \
    --edit_indices 3 5 \
    --ablations matched shuffled_touch zero_touch stats_only \
    2>&1 | tee ${OUT}/eval_high_contact_1k.log"

echo "Started tmux session ${SESSION} on GPU ${GPU}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs: ${REPO}/${OUT}/train.log"
