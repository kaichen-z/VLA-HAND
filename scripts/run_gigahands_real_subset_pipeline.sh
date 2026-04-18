#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
GIGAHANDS_ROOT="${GIGAHANDS_ROOT:-${DATA_ROOT}/gigahands_real}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/vitra_gigahands_real_subset}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/gigahands_real_subset_train}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/runs/gigahands_real_subset_eval}"
CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_gigahands_real_subset.json}"
GPUS="${GPUS:-0,1,3,4}"
NPROC="${NPROC:-4}"
NUM_TRAIN="${NUM_TRAIN:-20}"
NUM_TEST="${NUM_TEST:-5}"
MIN_FRAMES="${MIN_FRAMES:-32}"
CAMERA="${CAMERA:-brics-odroid-011_cam0}"
CANDIDATE_POOL_FACTOR="${CANDIDATE_POOL_FACTOR:-4}"
REQUIRE_VIDEO_EXISTS="${REQUIRE_VIDEO_EXISTS:-0}"
STAGE="${STAGE:-help}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

download_metadata() {
  mkdir -p "${GIGAHANDS_ROOT}"
  cd "${GIGAHANDS_ROOT}"
  wget -c https://g-ad09a0.56197.5898.data.globus.org/annotations_v2.jsonl
  wget -c https://g-852369.56197.5898.data.globus.org/multiview_camera_video_map.csv
}

download_hand_poses() {
  mkdir -p "${GIGAHANDS_ROOT}"
  cd "${GIGAHANDS_ROOT}"
  wget -c https://g-ad09a0.56197.5898.data.globus.org/hand_poses.tar.gz
  tar -xzf hand_poses.tar.gz
}

prepare_subset() {
  python tools/prepare_gigahands_real_subset.py \
    --gigahands_root "${GIGAHANDS_ROOT}" \
    --num_train "${NUM_TRAIN}" \
    --num_test "${NUM_TEST}" \
    --min_frames "${MIN_FRAMES}" \
    --prefer_camera "${CAMERA}" \
    --require_both_hands_valid \
    --prefer_bimanual_motion \
    --candidate_pool_factor "${CANDIDATE_POOL_FACTOR}" \
    --output_manifest "${GIGAHANDS_ROOT}/subset_manifest.json" \
    --output_video_list "${GIGAHANDS_ROOT}/needed_videos.txt" \
    $(if [[ "${REQUIRE_VIDEO_EXISTS}" == "1" ]]; then echo "--require_video_exists"; fi)
  make_unique_video_list
  echo "Download the RGB videos listed in ${GIGAHANDS_ROOT}/needed_videos_unique.txt into ${GIGAHANDS_ROOT}/multiview_rgb_vids before running STAGE=convert."
}

make_unique_video_list() {
  python tools/verify_required_videos.py \
    --root "${GIGAHANDS_ROOT}" \
    --video_list "${GIGAHANDS_ROOT}/needed_videos.txt" \
    --unique_output "${GIGAHANDS_ROOT}/needed_videos_unique.txt"
}

verify_videos() {
  python tools/verify_required_videos.py \
    --root "${GIGAHANDS_ROOT}" \
    --video_list "${GIGAHANDS_ROOT}/needed_videos.txt" \
    --unique_output "${GIGAHANDS_ROOT}/needed_videos_unique.txt" \
    --fail_on_missing
}

convert_subset() {
  verify_videos
  python data/preprocessing/convert_gigahands_to_vitra_stage1.py \
    --gigahands_root "${GIGAHANDS_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --input_layout full \
    --subset_manifest "${GIGAHANDS_ROOT}/subset_manifest.json" \
    --split all \
    --camera auto \
    --dataset_name_prefix gigahands_real \
    --write_video
}

calculate_stats() {
  python vitra/datasets/calculate_statistics.py \
    --dataset_folder "${OUTPUT_ROOT}" \
    --dataset_name gigahands_real_train \
    --num_workers 0 \
    --batch_size 16 \
    --save_folder "${OUTPUT_ROOT}/Annotation/statistics"
}

train_smoke() {
  CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE="${WANDB_MODE:-offline}" \
  torchrun --nproc_per_node="${NPROC}" --standalone scripts/train.py \
    --config "${CONFIG}" \
    --data_mix gigahands_real_train \
    --batch_size 1 \
    --total_batch_size "${NPROC}" \
    --max_steps 5 \
    --num_workers 0 \
    --no_save_checkpoint
}

train_small() {
  CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE="${WANDB_MODE:-offline}" \
  torchrun --nproc_per_node="${NPROC}" --standalone scripts/train.py \
    --config "${CONFIG}" \
    --data_mix gigahands_real_train \
    --batch_size 1 \
    --total_batch_size "${NPROC}" \
    --max_steps "${MAX_STEPS:-500}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --num_workers "${NUM_WORKERS:-2}"
}

eval_before() {
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python tools/evaluate_gigahands_stage1.py \
    --config "${CONFIG}" \
    --dataset_root "${OUTPUT_ROOT}" \
    --data_mix gigahands_real_test \
    --checkpoint none \
    --num_eval_clips "${NUM_EVAL_CLIPS:-5}" \
    --output_dir "${EVAL_ROOT}/before"
}

eval_after() {
  : "${CHECKPOINT:?Set CHECKPOINT to a checkpoint directory or weights.pt path.}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python tools/evaluate_gigahands_stage1.py \
    --config "${CONFIG}" \
    --dataset_root "${OUTPUT_ROOT}" \
    --data_mix gigahands_real_test \
    --checkpoint "${CHECKPOINT}" \
    --num_eval_clips "${NUM_EVAL_CLIPS:-5}" \
    --output_dir "${EVAL_ROOT}/after"
}

case "${STAGE}" in
  download_metadata) download_metadata ;;
  download_hand_poses) download_hand_poses ;;
  prepare) prepare_subset ;;
  make_unique_video_list) make_unique_video_list ;;
  verify_videos) verify_videos ;;
  convert) convert_subset ;;
  stats) calculate_stats ;;
  train_smoke) train_smoke ;;
  train) train_small ;;
  eval_before) eval_before ;;
  eval_after) eval_after ;;
  *)
    cat <<EOF
Usage:
  STAGE=download_metadata bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=download_hand_poses bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=prepare bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=make_unique_video_list bash scripts/run_gigahands_real_subset_pipeline.sh
  # Download or place the RGB videos listed in \${GIGAHANDS_ROOT}/needed_videos_unique.txt under \${GIGAHANDS_ROOT}/multiview_rgb_vids.
  STAGE=verify_videos bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=convert bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=stats bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=train_smoke bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=train bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=eval_before bash scripts/run_gigahands_real_subset_pipeline.sh
  CHECKPOINT=/path/to/checkpoint STAGE=eval_after bash scripts/run_gigahands_real_subset_pipeline.sh

Resolved defaults:
  REPO_ROOT=${REPO_ROOT}
  DATA_ROOT=${DATA_ROOT}
  GIGAHANDS_ROOT=${GIGAHANDS_ROOT}
  OUTPUT_ROOT=${OUTPUT_ROOT}
  RUN_ROOT=${RUN_ROOT}
EOF
    ;;
esac
