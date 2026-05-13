#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/scratch/chonghej/conda_envs/vitra/bin/python}"
GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT}/vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json}"
STUDENT_CONFIG="${STUDENT_CONFIG:-${ROOT}/vitra/configs/finetune_distill_all_cam0_keypoints_mano.json}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT}/checkpoints/vitra-vla-3b.pt}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${BASE_CHECKPOINT}}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a finetune-distill student checkpoint directory or weights.pt file.}"
NUM_EVAL_CLIPS="${NUM_EVAL_CLIPS:-200}"
EVAL_SAMPLE_STRATEGY="${EVAL_SAMPLE_STRATEGY:-middle_per_episode}"
LABEL="${LABEL:-finetune_distill_student}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/finetune_distill_all_cam0_keypoints_mano/eval/${LABEL}_${NUM_EVAL_CLIPS}clips}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/evaluate_gigahands_stage1.py \
  --config "${BASE_CONFIG}" \
  --dataset_root "${DATASET_ROOT}" \
  --data_mix gigahands_real_test \
  --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
  --teacher_config "${BASE_CONFIG}" \
  --teacher_label base_vitra_teacher \
  --base_checkpoint "${BASE_CHECKPOINT}" \
  --base_config_file "${BASE_CONFIG}" \
  --base_label base3b \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_config "${STUDENT_CONFIG}" \
  --label "${LABEL}" \
  --num_eval_clips "${NUM_EVAL_CLIPS}" \
  --eval_sample_strategy "${EVAL_SAMPLE_STRATEGY}" \
  --num_ddim_steps 10 \
  --cfg_scale 5.0 \
  --no_videos \
  --output_dir "${OUTPUT_DIR}"

echo "Metrics written to ${OUTPUT_DIR}/metrics.json"
