#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/scratch/chonghej/conda_envs/vitra/bin/python}"
GPU="${GPU:-0}"

CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned}"
DATA_MIX="${DATA_MIX:-gigahands_real_test}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a VITRA checkpoint directory or .pt file.}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${REPO_ROOT}/checkpoints/vitra-vla-3b.pt}"

LABEL="${LABEL:-trained}"
BASELINE_LABEL="${BASELINE_LABEL:-base}"
NUM_EVAL_CLIPS="${NUM_EVAL_CLIPS:-1495}"
EVAL_SAMPLE_STRATEGY="${EVAL_SAMPLE_STRATEGY:-middle_per_episode}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/gigahands_cleaned_eval/${LABEL}_vs_${BASELINE_LABEL}_${NUM_EVAL_CLIPS}clips}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" tools/evaluate_gigahands_stage1.py \
  --config "${CONFIG}" \
  --dataset_root "${DATASET_ROOT}" \
  --data_mix "${DATA_MIX}" \
  --checkpoint "${CHECKPOINT}" \
  --label "${LABEL}" \
  --baseline_checkpoint "${BASELINE_CHECKPOINT}" \
  --baseline_label "${BASELINE_LABEL}" \
  --num_eval_clips "${NUM_EVAL_CLIPS}" \
  --eval_sample_strategy "${EVAL_SAMPLE_STRATEGY}" \
  --no_videos \
  --output_dir "${OUTPUT_DIR}"

echo "Metrics written to ${OUTPUT_DIR}/metrics_comparison.json"
