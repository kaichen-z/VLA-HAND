#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
GIGAHANDS_ROOT="${GIGAHANDS_ROOT:-${DATA_ROOT}/gigahands_real}"
LINKED_OUTPUT_ROOT="${LINKED_OUTPUT_ROOT:-${DATA_ROOT}/vitra_gigahands_real_large_keypoints_linked}"
UNDISTORTED_OUTPUT_ROOT="${UNDISTORTED_OUTPUT_ROOT:-${DATA_ROOT}/vitra_gigahands_real_large_keypoints_undistorted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LINKED_OUTPUT_ROOT}}"
MANIFEST="${MANIFEST:-${GIGAHANDS_ROOT}/subset_manifest_large_keypoints.json}"
VIDEO_LIST="${VIDEO_LIST:-${GIGAHANDS_ROOT}/needed_videos_large_keypoints.txt}"
UNIQUE_VIDEO_LIST="${UNIQUE_VIDEO_LIST:-${GIGAHANDS_ROOT}/needed_videos_large_keypoints_unique.txt}"
CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_gigahands_real_large_keypoints_vitra3b_linked.json}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/gigahands_real_large_keypoints_vitra3b_linked_train}"
VITRA_BASE_CHECKPOINT="${VITRA_BASE_CHECKPOINT:-${REPO_ROOT}/checkpoints/vitra-vla-3b.pt}"
HF_CACHE_SEARCH_ROOT="${HF_CACHE_SEARCH_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/hf_cache}"

GPUS="${GPUS:-1,2}"
NPROC="${NPROC:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((NPROC * BATCH_SIZE))}"
NUM_TRAIN="${NUM_TRAIN:-500}"
NUM_TEST="${NUM_TEST:-100}"
MIN_FRAMES="${MIN_FRAMES:-32}"
CAMERA="${CAMERA:-brics-odroid-011_cam0}"
CANDIDATE_POOL_FACTOR="${CANDIDATE_POOL_FACTOR:-20}"
REQUIRE_VIDEO_EXISTS="${REQUIRE_VIDEO_EXISTS:-1}"
REQUIRE_VIDEO_FRAME_COUNT="${REQUIRE_VIDEO_FRAME_COUNT:-1}"
REQUIRE_KEYPOINTS="${REQUIRE_KEYPOINTS:-1}"
MAX_STEPS="${MAX_STEPS:-65000}"
SAVE_STEPS="${SAVE_STEPS:-4000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-4h}"
STAGE="${STAGE:-help}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

ensure_vitra_base_checkpoint() {
  if [[ -f "${VITRA_BASE_CHECKPOINT}" || -L "${VITRA_BASE_CHECKPOINT}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${VITRA_BASE_CHECKPOINT}")"
  local candidate=""
  local search_roots=("${HF_CACHE_SEARCH_ROOT}")
  if [[ -n "${HF_HOME:-}" ]]; then
    search_roots=("${HF_HOME}" "${HF_CACHE_SEARCH_ROOT}")
  fi
  for root in "${search_roots[@]}"; do
    candidate="$(find "${root}" \
      -path "*/models--VITRA-VLA--VITRA-VLA-3B/snapshots/*/checkpoints/vitra-vla-3b.pt" \
      \( -type f -o -type l \) -print -quit 2>/dev/null || true)"
    if [[ -n "${candidate}" ]]; then
      break
    fi
  done
  if [[ -n "${candidate}" ]]; then
    ln -s "${candidate}" "${VITRA_BASE_CHECKPOINT}"
    return
  fi
  echo "Could not find VITRA base checkpoint. Set VITRA_BASE_CHECKPOINT or download VITRA-VLA/VITRA-VLA-3B first." >&2
  return 1
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
    --output_manifest "${MANIFEST}" \
    --output_video_list "${VIDEO_LIST}" \
    $(if [[ "${REQUIRE_KEYPOINTS}" == "1" ]]; then echo "--require_keypoints"; fi) \
    $(if [[ "${REQUIRE_VIDEO_EXISTS}" == "1" ]]; then echo "--require_video_exists"; fi) \
    $(if [[ "${REQUIRE_VIDEO_FRAME_COUNT}" == "1" ]]; then echo "--require_video_frame_count"; fi)
  make_unique_video_list
}

make_unique_video_list() {
  python tools/verify_required_videos.py \
    --root "${GIGAHANDS_ROOT}" \
    --video_list "${VIDEO_LIST}" \
    --unique_output "${UNIQUE_VIDEO_LIST}"
}

verify_videos() {
  python tools/verify_required_videos.py \
    --root "${GIGAHANDS_ROOT}" \
    --video_list "${VIDEO_LIST}" \
    --unique_output "${UNIQUE_VIDEO_LIST}" \
    --fail_on_missing
}

convert_undistorted() {
  verify_videos
  python data/preprocessing/convert_gigahands_to_vitra_stage1.py \
    --gigahands_root "${GIGAHANDS_ROOT}" \
    --output_root "${UNDISTORTED_OUTPUT_ROOT}" \
    --input_layout full \
    --subset_manifest "${MANIFEST}" \
    --split all \
    --camera auto \
    --dataset_name_prefix gigahands_real \
    --write_video \
    --undistort \
    --clean_output
}

convert_linked() {
  verify_videos
  python data/preprocessing/convert_gigahands_to_vitra_stage1.py \
    --gigahands_root "${GIGAHANDS_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --input_layout full \
    --subset_manifest "${MANIFEST}" \
    --split all \
    --camera auto \
    --dataset_name_prefix gigahands_real \
    --clean_output
  rm -rf "${OUTPUT_ROOT}/Video/GigaHands_root"
  mkdir -p "${OUTPUT_ROOT}/Video"
  ln -s "${GIGAHANDS_ROOT}/multiview_rgb_vids" "${OUTPUT_ROOT}/Video/GigaHands_root"
}

calculate_keypoint_stats() {
  python vitra/datasets/calculate_statistics.py \
    --dataset_folder "${OUTPUT_ROOT}" \
    --dataset_name gigahands_real_train \
    --action_type keypoints \
    --num_workers 0 \
    --batch_size 16 \
    --save_folder "${OUTPUT_ROOT}/Annotation/statistics"
}

train_smoke() {
  ensure_vitra_base_checkpoint
  CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE="${WANDB_MODE:-offline}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  torchrun --nproc_per_node="${NPROC}" --standalone scripts/train.py \
    --config "${CONFIG}" \
    --data_mix gigahands_real_train \
    --batch_size "${BATCH_SIZE}" \
    --total_batch_size "${TOTAL_BATCH_SIZE}" \
    --max_steps "${SMOKE_STEPS:-5}" \
    --num_workers 0 \
    --no_save_checkpoint
}

train_timed() {
  ensure_vitra_base_checkpoint
  mkdir -p "${RUN_ROOT}/launch_logs"
  CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE="${WANDB_MODE:-offline}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  timeout "${TRAIN_TIMEOUT}" torchrun --nproc_per_node="${NPROC}" --standalone scripts/train.py \
    --config "${CONFIG}" \
    --data_mix gigahands_real_train \
    --batch_size "${BATCH_SIZE}" \
    --total_batch_size "${TOTAL_BATCH_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --num_workers "${NUM_WORKERS}"
}

train_timed_background() {
  mkdir -p "${RUN_ROOT}/launch_logs"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_path="${RUN_ROOT}/launch_logs/train_timed_${timestamp}.log"
  local pid_path="${RUN_ROOT}/launch_logs/train_timed_${timestamp}.pid"
  nohup env \
    STAGE=train_timed \
    REPO_ROOT="${REPO_ROOT}" \
    DATA_ROOT="${DATA_ROOT}" \
    GIGAHANDS_ROOT="${GIGAHANDS_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    MANIFEST="${MANIFEST}" \
    VIDEO_LIST="${VIDEO_LIST}" \
    UNIQUE_VIDEO_LIST="${UNIQUE_VIDEO_LIST}" \
    CONFIG="${CONFIG}" \
    RUN_ROOT="${RUN_ROOT}" \
    VITRA_BASE_CHECKPOINT="${VITRA_BASE_CHECKPOINT}" \
    HF_CACHE_SEARCH_ROOT="${HF_CACHE_SEARCH_ROOT}" \
    GPUS="${GPUS}" \
    NPROC="${NPROC}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE}" \
    MAX_STEPS="${MAX_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    TRAIN_TIMEOUT="${TRAIN_TIMEOUT}" \
    HF_HOME="${HF_HOME:-}" \
    WANDB_MODE="${WANDB_MODE:-offline}" \
    PATH="${PATH}" \
    bash "${BASH_SOURCE[0]}" >"${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_path}"
  ln -sfn "$(basename "${log_path}")" "${RUN_ROOT}/launch_logs/latest_train_timed.log"
  ln -sfn "$(basename "${pid_path}")" "${RUN_ROOT}/launch_logs/latest_train_timed.pid"
  echo "Started background timed training."
  echo "PID: ${pid}"
  echo "Log: ${log_path}"
  echo "PID file: ${pid_path}"
  echo "GPUS=${GPUS} NPROC=${NPROC} TRAIN_TIMEOUT=${TRAIN_TIMEOUT} SAVE_STEPS=${SAVE_STEPS}"
}

case "${STAGE}" in
  ensure_vitra_base_checkpoint) ensure_vitra_base_checkpoint ;;
  prepare) prepare_subset ;;
  make_unique_video_list) make_unique_video_list ;;
  verify_videos) verify_videos ;;
  convert_linked) convert_linked ;;
  convert_undistorted) convert_undistorted ;;
  stats_keypoints) calculate_keypoint_stats ;;
  prepare_all) prepare_subset; convert_linked; calculate_keypoint_stats ;;
  train_smoke) train_smoke ;;
  train_timed) train_timed ;;
  train_timed_background) train_timed_background ;;
  train_12h) train_timed ;;
  *)
    cat <<EOF
Usage:
  STAGE=prepare bash scripts/run_gigahands_real_large_keypoints_pipeline.sh
  STAGE=convert_linked bash scripts/run_gigahands_real_large_keypoints_pipeline.sh
  STAGE=convert_undistorted bash scripts/run_gigahands_real_large_keypoints_pipeline.sh
  STAGE=stats_keypoints bash scripts/run_gigahands_real_large_keypoints_pipeline.sh
  STAGE=train_smoke bash scripts/run_gigahands_real_large_keypoints_pipeline.sh
  STAGE=train_timed bash scripts/run_gigahands_real_large_keypoints_pipeline.sh
  STAGE=train_timed_background bash scripts/run_gigahands_real_large_keypoints_pipeline.sh

Defaults:
  GPU devices: ${GPUS}
  train timeout: ${TRAIN_TIMEOUT}
  save steps: ${SAVE_STEPS}
  train/test clips: ${NUM_TRAIN}/${NUM_TEST}
  linked output dataset: ${OUTPUT_ROOT}
  undistorted output dataset: ${UNDISTORTED_OUTPUT_ROOT}
  config: ${CONFIG}
EOF
    ;;
esac
