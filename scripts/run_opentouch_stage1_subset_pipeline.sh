#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
OPENTOUCH_ROOT="${OPENTOUCH_ROOT:-${DATA_ROOT}/opentouch_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/vitra_opentouch_keypoint}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/opentouch_keypoint_subset}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/runs/opentouch_keypoint_subset_eval}"
CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_opentouch_keypoint_subset.json}"
GPUS="${GPUS:-0,1,3,4}"
NPROC="${NPROC:-4}"
MIN_FRAMES="${MIN_FRAMES:-17}"
MAX_FILES="${MAX_FILES:-1}"
MAX_CLIPS="${MAX_CLIPS:-20}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
STAGE="${STAGE:-help}"

# From https://github.com/OpenTouch-MIT/opentouch/blob/main/scripts/download_data.sh
OPENTOUCH_GDRIVE_IDS=(
  "1EjMOzs45devBo0TqhuhZTT_Ll7HZ1lrW"
  "1fAmmieSr0yFm7ldhW7Smld7jUxBCw8fu"
  "1cUhgYbredkIRswanUiM5uDixiFLq4WCC"
  "1PCzWMJxtbD2HJLCl2WFzOIB-5RN3X81G"
  "1jFlYmCFb6GldbPJ-zLzJSCY-BKipdjPE"
  "1reSqa8v8RaY2kZXLw0_g7Amvq7lJl6Cu"
  "1atXpcctoHs4dbXhyAAO9EY88D2f1JYfT"
  "1Z3b-I6BMPgNlpiKw8gISkUi3VULUtLFN"
  "1u-6WGn3eMQJe3eh6lCFahlIcEVmkULna"
  "17wF0aBIH6RRtRGRaXeiI-Y4Lh5bnDFBL"
  "1KICpqtfmbnKhgHi-CIR9XAp24TE1945M"
  "1vkl6wat_dgF5NQs9QVDfCyJGyjEjd2FW"
  "1BbKU5vSH-wOrCnOjRWNe3H7niJP_uJrb"
  "1GCX4mAgCvOvmIQ0uXotqpoNdXgYzp4ki"
  "1rxsLWGw_diPvRnALxOYakCIweG90O28I"
  "1zAYfcMt2hqcG1bPtCOAkWu0zsd6lfvrX"
  "1tQh21z8KRxYHsh69dW6VcSw5Wux67R6_"
  "1jeA1bEit-tDQpfwt3NmTeC8iwM6I1qiE"
  "1UT5htydKCfBCO57On-mRJRz7mSi57K4u"
  "1h9Bl8CTGJWvU2XPr93fptBTpB2cwYgwq"
  "1SAbxWQZDEyTZ-ESVi9G5bxEc7ov-EO28"
  "1jKyVNsi7fsofSho_xoRi0Kgqem4zrk5F"
  "11LQ28c6jPhNfiu9fPDu5diruNUCa0bGM"
  "1dwlVYtBfyNUHg7Qxnxa_iYBYPCcn9VeX"
  "1X4-MS7Qodhtmn6zcY9a5cMq02eDLvOJq"
  "1VAKXJPO4j_40hpqslNJ4_WbgWfKaGLQC"
  "1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ"
)

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ensure_gdown() {
  if command -v gdown >/dev/null 2>&1; then
    return
  fi
  python -m pip install --user gdown
}

download_id() {
  local id="$1"
  mkdir -p "${OPENTOUCH_ROOT}"
  cd "${OPENTOUCH_ROOT}"
  gdown "${id}"
}

download_first_available() {
  ensure_gdown
  if [[ -n "${OPENTOUCH_GDRIVE_ID:-}" ]]; then
    download_id "${OPENTOUCH_GDRIVE_ID}"
    return
  fi
  local failures=0
  for id in "${OPENTOUCH_GDRIVE_IDS[@]}"; do
    echo "Trying OpenTouch Google Drive file ${id}"
    if download_id "${id}"; then
      echo "Downloaded one OpenTouch file successfully."
      return
    fi
    failures=$((failures + 1))
    echo "Skipping unavailable OpenTouch file ${id}"
  done
  echo "No OpenTouch Google Drive files were downloadable; ${failures} attempts failed." >&2
  return 1
}

download_one() {
  download_first_available
}

download_all() {
  ensure_gdown
  local failures=0
  for id in "${OPENTOUCH_GDRIVE_IDS[@]}"; do
    echo "Trying OpenTouch Google Drive file ${id}"
    if ! download_id "${id}"; then
      failures=$((failures + 1))
      echo "Skipping unavailable OpenTouch file ${id}"
    fi
  done
  cd "${OPENTOUCH_ROOT}"
  if [[ -f final_annotations.zip ]]; then
    unzip -o final_annotations.zip
  fi
  if [[ "${failures}" -gt 0 ]]; then
    echo "OpenTouch download completed with ${failures} skipped files."
  fi
}

verify_raw() {
  python - <<PY
from pathlib import Path
import json

root = Path("${OPENTOUCH_ROOT}")
hdf5 = sorted([*root.rglob("*.h5"), *root.rglob("*.hdf5")]) if root.exists() else []
print(json.dumps({
    "root": str(root),
    "hdf5_files": len(hdf5),
    "examples": [str(path.relative_to(root)) for path in hdf5[:10]],
}, indent=2))
if not hdf5:
    raise SystemExit(1)
PY
}

convert_subset() {
  verify_raw
  python data/preprocessing/convert_opentouch_to_vitra_stage1.py \
    --opentouch_root "${OPENTOUCH_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --min_frames "${MIN_FRAMES}" \
    --train_ratio "${TRAIN_RATIO}" \
    --max_files "${MAX_FILES}" \
    --max_clips "${MAX_CLIPS}" \
    --write_video
}

calculate_stats() {
  python vitra/datasets/calculate_statistics.py \
    --dataset_folder "${OUTPUT_ROOT}" \
    --dataset_name opentouch_keypoint_train \
    --action_type keypoints \
    --num_workers 0 \
    --batch_size 16 \
    --save_folder "${OUTPUT_ROOT}/Annotation/statistics"
}

train_smoke() {
  CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE="${WANDB_MODE:-offline}" \
  torchrun --nproc_per_node="${NPROC}" --standalone scripts/train.py \
    --config "${CONFIG}" \
    --data_mix opentouch_keypoint_train \
    --batch_size 1 \
    --total_batch_size "${NPROC}" \
    --max_steps "${MAX_STEPS:-5}" \
    --num_workers 0
}

eval_before() {
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python tools/evaluate_gigahands_stage1.py \
    --config "${CONFIG}" \
    --dataset_root "${OUTPUT_ROOT}" \
    --data_mix opentouch_keypoint_test \
    --checkpoint none \
    --num_eval_clips "${NUM_EVAL_CLIPS:-5}" \
    --output_dir "${EVAL_ROOT}/before"
}

eval_after() {
  : "${CHECKPOINT:?Set CHECKPOINT to a checkpoint directory or weights.pt path.}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python tools/evaluate_gigahands_stage1.py \
    --config "${CONFIG}" \
    --dataset_root "${OUTPUT_ROOT}" \
    --data_mix opentouch_keypoint_test \
    --checkpoint "${CHECKPOINT}" \
    --num_eval_clips "${NUM_EVAL_CLIPS:-5}" \
    --output_dir "${EVAL_ROOT}/after"
}

case "${STAGE}" in
  download_one) download_one ;;
  download_all) download_all ;;
  verify_raw) verify_raw ;;
  convert) convert_subset ;;
  stats) calculate_stats ;;
  train_smoke) train_smoke ;;
  eval_before) eval_before ;;
  eval_after) eval_after ;;
  *)
    cat <<EOF
Usage:
  STAGE=download_one bash scripts/run_opentouch_stage1_subset_pipeline.sh
  STAGE=download_all bash scripts/run_opentouch_stage1_subset_pipeline.sh
  STAGE=verify_raw bash scripts/run_opentouch_stage1_subset_pipeline.sh
  STAGE=convert bash scripts/run_opentouch_stage1_subset_pipeline.sh
  STAGE=stats bash scripts/run_opentouch_stage1_subset_pipeline.sh
  STAGE=train_smoke bash scripts/run_opentouch_stage1_subset_pipeline.sh
  STAGE=eval_before bash scripts/run_opentouch_stage1_subset_pipeline.sh
  CHECKPOINT=/path/to/checkpoint STAGE=eval_after bash scripts/run_opentouch_stage1_subset_pipeline.sh

Defaults:
  REPO_ROOT=${REPO_ROOT}
  DATA_ROOT=${DATA_ROOT}
  OPENTOUCH_ROOT=${OPENTOUCH_ROOT}
  OUTPUT_ROOT=${OUTPUT_ROOT}
  MAX_FILES=${MAX_FILES}
  MAX_CLIPS=${MAX_CLIPS}
EOF
    ;;
esac
