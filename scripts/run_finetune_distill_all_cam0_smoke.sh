#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/scratch/chonghej/conda_envs/vitra/bin/python}"
CONFIG="${CONFIG:-vitra/configs/finetune_distill_all_cam0_keypoints_mano_smoke.json}"
TEACHER_CONFIG="${TEACHER_CONFIG:-vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-./checkpoints/vitra-vla-3b.pt}"
INIT_DIR="${INIT_DIR:-runs/finetune_distill_all_cam0_keypoints_mano_smoke/init_student}"
GPU="${GPU:-0}"

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
