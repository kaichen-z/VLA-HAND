#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
GIGAHANDS_ROOT="${GIGAHANDS_ROOT:-${DATA_ROOT}/gigahands_real}"

export REPO_ROOT
export DATA_ROOT
export GIGAHANDS_ROOT
export LINKED_OUTPUT_ROOT="${LINKED_OUTPUT_ROOT:-${DATA_ROOT}/vitra_gigahands_real_all_cam0_keypoints_mano_linked}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${LINKED_OUTPUT_ROOT}}"
export MANIFEST="${MANIFEST:-${GIGAHANDS_ROOT}/subset_manifest_all_cam0_keypoints_mano.json}"
export VIDEO_LIST="${VIDEO_LIST:-${GIGAHANDS_ROOT}/needed_videos_all_cam0_keypoints_mano.txt}"
export UNIQUE_VIDEO_LIST="${UNIQUE_VIDEO_LIST:-${GIGAHANDS_ROOT}/needed_videos_all_cam0_keypoints_mano_unique.txt}"
export CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json}"
export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_train}"
export PYTHON_BIN="${PYTHON_BIN:-/scratch/chonghej/conda_envs/vitra/bin/python}"

export NUM_TRAIN="${NUM_TRAIN:-0}"
export NUM_TEST="${NUM_TEST:-0}"
export SELECT_ALL="${SELECT_ALL:-1}"
export TEST_RATIO="${TEST_RATIO:-0.05}"
export CAMERA_SCOPE="${CAMERA_SCOPE:-all_cam0}"
export CAMERA="${CAMERA:-brics-odroid-001_cam0}"
export REQUIRE_BOTH_HANDS_VALID="${REQUIRE_BOTH_HANDS_VALID:-0}"
export PREFER_BIMANUAL_MOTION="${PREFER_BIMANUAL_MOTION:-0}"
export REQUIRE_KEYPOINTS="${REQUIRE_KEYPOINTS:-1}"
export REQUIRE_REAL_KEYPOINTS="${REQUIRE_REAL_KEYPOINTS:-0}"
export REQUIRE_VIDEO_FRAME_COUNT="${REQUIRE_VIDEO_FRAME_COUNT:-0}"
export KEYPOINTS_SOURCE="${KEYPOINTS_SOURCE:-mano}"
export ALLOW_MANO_KEYPOINT_FALLBACK="${ALLOW_MANO_KEYPOINT_FALLBACK:-0}"
export MIN_FRAMES="${MIN_FRAMES:-32}"
export SEED="${SEED:-42}"
export STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-2048}"
export STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-8}"

exec bash "${SCRIPT_DIR}/run_gigahands_real_large_keypoints_pipeline.sh" "$@"
