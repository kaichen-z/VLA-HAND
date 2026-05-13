#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/scratch/chonghej/conda_envs/vitra/bin/python}"
CONFIG="${CONFIG:-vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano_smoke.json}"
TEACHER_CONFIG="${TEACHER_CONFIG:-vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-runs/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_train/checkpoints/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_stage1_TB2_B2_bf16True/checkpoints/epoch=0-step=140000.ckpt}"
INIT_DIR="${INIT_DIR:-runs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano_smoke/init_student}"
GPU="${GPU:-7}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${GPU}"

cd "${ROOT}"

"${PYTHON}" scripts/create_finetune_distill_student.py \
  --student_config "${CONFIG}" \
  --teacher_config "${TEACHER_CONFIG}" \
  --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
  --output_dir "${INIT_DIR}" \
  --force

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 --standalone \
  scripts/train_finetune_distill.py \
  --config "${CONFIG}"

CHECKPOINT="${ROOT}/runs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano_smoke/checkpoints/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano_smoke_TB1_B1_bf16True/checkpoints/epoch=0-step=20.ckpt"
OUTPUT_DIR="${ROOT}/runs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano_smoke/eval/smoke_5clips"

"${PYTHON}" tools/evaluate_gigahands_stage1.py \
  --config "${ROOT}/${TEACHER_CONFIG}" \
  --dataset_root "${ROOT}/datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned" \
  --data_mix gigahands_real_test \
  --teacher_checkpoint "${ROOT}/${TEACHER_CHECKPOINT}" \
  --teacher_config "${ROOT}/${TEACHER_CONFIG}" \
  --teacher_label teacher_step140000 \
  --base_checkpoint "${ROOT}/checkpoints/vitra-vla-3b.pt" \
  --base_config_file "${ROOT}/${TEACHER_CONFIG}" \
  --base_label base3b \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_config "${ROOT}/${CONFIG}" \
  --label step140000_joint_kd_smoke \
  --num_eval_clips 5 \
  --eval_sample_strategy middle_per_episode \
  --num_ddim_steps 10 \
  --cfg_scale 5.0 \
  --no_videos \
  --output_dir "${OUTPUT_DIR}"

echo "Smoke checkpoint: ${CHECKPOINT}"
echo "Smoke metrics: ${OUTPUT_DIR}/metrics.json"
