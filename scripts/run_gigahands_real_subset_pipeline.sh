#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
GIGAHANDS_ROOT="${GIGAHANDS_ROOT:-${DATA_ROOT}/gigahands_real}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/vitra_gigahands_real_subset}"
UNDISTORTED_OUTPUT_ROOT="${UNDISTORTED_OUTPUT_ROOT:-${DATA_ROOT}/vitra_gigahands_real_subset_undistorted}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/gigahands_real_subset_train}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/runs/gigahands_real_subset_eval}"
CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_gigahands_real_subset.json}"
UNDISTORTED_CONFIG="${UNDISTORTED_CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_gigahands_real_subset_vitra3b_undistorted.json}"
UNDISTORTED_RUN_ROOT="${UNDISTORTED_RUN_ROOT:-${REPO_ROOT}/runs/gigahands_real_subset_vitra3b_undistorted_train}"
UNDISTORTED_EVAL_ROOT="${UNDISTORTED_EVAL_ROOT:-${REPO_ROOT}/runs/gigahands_real_subset_vitra3b_undistorted_eval}"
VITRA_BASE_CHECKPOINT="${VITRA_BASE_CHECKPOINT:-${REPO_ROOT}/checkpoints/vitra-vla-3b.pt}"
HF_CACHE_SEARCH_ROOT="${HF_CACHE_SEARCH_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/hf_cache}"
GPUS="${GPUS:-1,3}"
NPROC="${NPROC:-2}"
NUM_TRAIN="${NUM_TRAIN:-20}"
NUM_TEST="${NUM_TEST:-5}"
MIN_FRAMES="${MIN_FRAMES:-32}"
CAMERA="${CAMERA:-brics-odroid-011_cam0}"
CANDIDATE_POOL_FACTOR="${CANDIDATE_POOL_FACTOR:-4}"
REQUIRE_VIDEO_EXISTS="${REQUIRE_VIDEO_EXISTS:-0}"
STAGE="${STAGE:-help}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

clean_generated() {
  rm -rf \
    "${OUTPUT_ROOT}" \
    "${UNDISTORTED_OUTPUT_ROOT}" \
    "${REPO_ROOT}/runs/gigahands_real_subset_train" \
    "${REPO_ROOT}/runs/gigahands_real_subset_eval" \
    "${REPO_ROOT}/runs/gigahands_real_subset_vitra3b_train" \
    "${REPO_ROOT}/runs/gigahands_real_subset_vitra3b_eval" \
    "${UNDISTORTED_RUN_ROOT}" \
    "${UNDISTORTED_EVAL_ROOT}"
}

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

convert_subset_undistorted() {
  verify_videos
  python data/preprocessing/convert_gigahands_to_vitra_stage1.py \
    --gigahands_root "${GIGAHANDS_ROOT}" \
    --output_root "${UNDISTORTED_OUTPUT_ROOT}" \
    --input_layout full \
    --subset_manifest "${GIGAHANDS_ROOT}/subset_manifest.json" \
    --split all \
    --camera auto \
    --dataset_name_prefix gigahands_real \
    --write_video \
    --undistort \
    --clean_output
}

calculate_stats() {
  python vitra/datasets/calculate_statistics.py \
    --dataset_folder "${OUTPUT_ROOT}" \
    --dataset_name gigahands_real_train \
    --num_workers 0 \
    --batch_size 16 \
    --save_folder "${OUTPUT_ROOT}/Annotation/statistics"
}

calculate_stats_undistorted() {
  python vitra/datasets/calculate_statistics.py \
    --dataset_folder "${UNDISTORTED_OUTPUT_ROOT}" \
    --dataset_name gigahands_real_train \
    --num_workers 0 \
    --batch_size 16 \
    --save_folder "${UNDISTORTED_OUTPUT_ROOT}/Annotation/statistics"
}

train_smoke() {
  ensure_vitra_base_checkpoint
  CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE="${WANDB_MODE:-offline}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
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
  ensure_vitra_base_checkpoint
  CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE="${WANDB_MODE:-offline}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  torchrun --nproc_per_node="${NPROC}" --standalone scripts/train.py \
    --config "${CONFIG}" \
    --data_mix gigahands_real_train \
    --batch_size 1 \
    --total_batch_size "${NPROC}" \
    --max_steps "${MAX_STEPS:-500}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --num_workers "${NUM_WORKERS:-2}"
}

train_smoke_undistorted() {
  CONFIG="${UNDISTORTED_CONFIG}" train_smoke
}

train_undistorted() {
  CONFIG="${UNDISTORTED_CONFIG}" SAVE_STEPS="${SAVE_STEPS:-500}" train_small
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

eval_undistorted() {
  : "${CHECKPOINT:?Set CHECKPOINT to a checkpoint directory or weights.pt path.}"
  ensure_vitra_base_checkpoint
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}" python tools/evaluate_gigahands_stage1.py \
    --config "${UNDISTORTED_CONFIG}" \
    --dataset_root "${UNDISTORTED_OUTPUT_ROOT}" \
    --data_mix gigahands_real_test \
    --checkpoint "${CHECKPOINT}" \
    --label "${LABEL:-trained}" \
    --baseline_checkpoint "${BASELINE_CHECKPOINT:-${VITRA_BASE_CHECKPOINT}}" \
    --baseline_label "${BASELINE_LABEL:-base}" \
    --num_eval_clips "${NUM_EVAL_CLIPS:-5}" \
    --eval_sample_strategy "${EVAL_SAMPLE_STRATEGY:-middle_per_episode}" \
    --output_dir "${UNDISTORTED_EVAL_ROOT}" \
    --rgb_overlay_videos \
    --undistort_overlay_frames \
    --keypoint_overlay_debug \
    --mano_param_overlay_debug
}

case "${STAGE}" in
  download_metadata) download_metadata ;;
  download_hand_poses) download_hand_poses ;;
  clean_generated) clean_generated ;;
  ensure_vitra_base_checkpoint) ensure_vitra_base_checkpoint ;;
  prepare) prepare_subset ;;
  make_unique_video_list) make_unique_video_list ;;
  verify_videos) verify_videos ;;
  convert) convert_subset ;;
  convert_undistorted) convert_subset_undistorted ;;
  stats) calculate_stats ;;
  stats_undistorted) calculate_stats_undistorted ;;
  train_smoke) train_smoke ;;
  train_smoke_undistorted) train_smoke_undistorted ;;
  train) train_small ;;
  train_undistorted) train_undistorted ;;
  eval_before) eval_before ;;
  eval_after) eval_after ;;
  eval_undistorted) eval_undistorted ;;
  *)
    cat <<EOF
Usage:
  STAGE=download_metadata bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=download_hand_poses bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=clean_generated bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=ensure_vitra_base_checkpoint bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=prepare bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=make_unique_video_list bash scripts/run_gigahands_real_subset_pipeline.sh
  # Download or place the RGB videos listed in \${GIGAHANDS_ROOT}/needed_videos_unique.txt under \${GIGAHANDS_ROOT}/multiview_rgb_vids.
  STAGE=verify_videos bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=convert bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=convert_undistorted bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=stats bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=stats_undistorted bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=train_smoke bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=train_smoke_undistorted bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=train bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=train_undistorted bash scripts/run_gigahands_real_subset_pipeline.sh
  STAGE=eval_before bash scripts/run_gigahands_real_subset_pipeline.sh
  CHECKPOINT=/path/to/checkpoint STAGE=eval_after bash scripts/run_gigahands_real_subset_pipeline.sh
  CHECKPOINT=/path/to/checkpoint STAGE=eval_undistorted bash scripts/run_gigahands_real_subset_pipeline.sh

Resolved defaults:
  REPO_ROOT=${REPO_ROOT}
  DATA_ROOT=${DATA_ROOT}
  GIGAHANDS_ROOT=${GIGAHANDS_ROOT}
  OUTPUT_ROOT=${OUTPUT_ROOT}
  UNDISTORTED_OUTPUT_ROOT=${UNDISTORTED_OUTPUT_ROOT}
  UNDISTORTED_CONFIG=${UNDISTORTED_CONFIG}
  UNDISTORTED_RUN_ROOT=${UNDISTORTED_RUN_ROOT}
  UNDISTORTED_EVAL_ROOT=${UNDISTORTED_EVAL_ROOT}
  VITRA_BASE_CHECKPOINT=${VITRA_BASE_CHECKPOINT}
  HF_CACHE_SEARCH_ROOT=${HF_CACHE_SEARCH_ROOT}
  GPUS=${GPUS}
  NPROC=${NPROC}
EOF
    ;;
esac
