#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/scratch/chonghej/conda_envs/vitra/bin/python}"
CONFIG="${CONFIG:-vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano.json}"
TEACHER_CONFIG="${TEACHER_CONFIG:-vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-runs/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_train/checkpoints/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_stage1_TB2_B2_bf16True/checkpoints/epoch=0-step=140000.ckpt}"
INIT_DIR="${INIT_DIR:-runs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano/init_student}"
GPU="${GPU:-7}"
NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-50000}"
SESSION="${SESSION:-finetune_distill_step140000_joint_kd_all_cam0_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/runs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano/launch_logs"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

mkdir -p "${LOG_DIR}"

CMD="cd ${ROOT} && \
export TOKENIZERS_PARALLELISM=false WANDB_MODE=${WANDB_MODE:-offline} CUDA_VISIBLE_DEVICES=${GPU} NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
${PYTHON} scripts/create_finetune_distill_student.py \
  --student_config ${CONFIG} \
  --teacher_config ${TEACHER_CONFIG} \
  --teacher_checkpoint ${TEACHER_CHECKPOINT} \
  --output_dir ${INIT_DIR} \
  --force && \
${PYTHON} -m torch.distributed.run --nproc_per_node=${NPROC} --standalone \
  scripts/train_finetune_distill.py \
  --config ${CONFIG} \
  --batch_size ${BATCH_SIZE} \
  --total_batch_size ${TOTAL_BATCH_SIZE} \
  --max_steps ${MAX_STEPS} \
  --save_steps ${SAVE_STEPS} 2>&1 | tee ${LOG_FILE}"

tmux new-session -d -s "${SESSION}" "${CMD}"
echo "session=${SESSION}"
echo "gpu=${GPU}"
echo "nproc=${NPROC}"
echo "batch_size=${BATCH_SIZE}"
echo "total_batch_size=${TOTAL_BATCH_SIZE}"
echo "max_steps=${MAX_STEPS}"
echo "save_steps=${SAVE_STEPS}"
echo "log=${LOG_FILE}"
