#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/chonghej/scratch/chonghej/VLA-HAND}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/chonghej/conda_envs/vitra/bin/python}"
GPU="${GPU:-7}"

OPENTOUCH_ROOT="${OPENTOUCH_ROOT:-${REPO_ROOT}/datasets/opentouch_raw}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-${OPENTOUCH_ROOT}/final_annotations}"
LABELS_PATH="${LABELS_PATH:-${OPENTOUCH_ROOT}/final_annotations_merged.csv}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/vitra_opentouch_keypoint_full_text_aligned}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/online_touch_editor_large}"
CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_opentouch_keypoint_subset.json}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/runs/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_train/checkpoints/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_stage1_TB2_B2_bf16True/checkpoints/epoch=0-step=140000.ckpt}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
TEST_SAMPLES="${TEST_SAMPLES:-10000}"
EDITOR_STEPS="${EDITOR_STEPS:-5000}"
EDITOR_BATCH_SIZE="${EDITOR_BATCH_SIZE:-256}"
EDIT_START_IDX="${EDIT_START_IDX:-3}"
DDIM_STEPS="${DDIM_STEPS:-10}"
CFG_SCALE="${CFG_SCALE:-5.0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RUN_ROOT

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"

if [[ ! -f "${DATASET_ROOT}/Annotation/opentouch_keypoint_train/episode_frame_index.npz" ]]; then
  echo "[1/8] Merge OpenTouch annotations"
  "${PYTHON_BIN}" data/preprocessing/merge_opentouch_annotations.py \
    --annotations_dir "${ANNOTATIONS_DIR}" \
    --output_path "${LABELS_PATH}"

  echo "[2/8] Convert full OpenTouch dataset"
  "${PYTHON_BIN}" data/preprocessing/convert_opentouch_to_vitra_stage1.py \
    --opentouch_root "${OPENTOUCH_ROOT}" \
    --output_root "${DATASET_ROOT}" \
    --labels_path "${LABELS_PATH}" \
    --min_frames 17 \
    --train_ratio 0.8 \
    --write_video \
    --require_labels
else
  echo "[1/8] Dataset already exists at ${DATASET_ROOT}; skipping raw annotation merge."
  echo "[2/8] Dataset already exists at ${DATASET_ROOT}; skipping conversion."
fi

echo "[3/8] Calculate training statistics"
if [[ ! -f "${DATASET_ROOT}/Annotation/statistics/opentouch_keypoint_train_keypoints_statistics.json" ]]; then
  "${PYTHON_BIN}" vitra/datasets/calculate_statistics.py \
    --dataset_folder "${DATASET_ROOT}" \
    --dataset_name opentouch_keypoint_train \
    --action_type keypoints \
    --num_workers 0 \
    --batch_size 32 \
    --save_folder "${DATASET_ROOT}/Annotation/statistics"
else
  echo "Statistics already exist; skipping."
fi

echo "[4/8] Cache VITRA base actions for train/test"
if [[ ! -f "${RUN_ROOT}/cache_train/summary.json" ]]; then
  "${PYTHON_BIN}" scripts/cache_touch_editor_base_actions.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --dataset_root "${DATASET_ROOT}" \
    --data_mix opentouch_keypoint_train \
    --statistics_dataset_name opentouch_keypoint_train \
    --cache_root "${RUN_ROOT}/cache_train" \
    --sample_mode random \
    --max_samples "${TRAIN_SAMPLES}" \
    --seed 42 \
    --edit_start_idx "${EDIT_START_IDX}" \
    --num_ddim_steps "${DDIM_STEPS}" \
    --cfg_scale "${CFG_SCALE}"
else
  echo "Train cache already exists; skipping."
fi

if [[ ! -f "${RUN_ROOT}/cache_test/summary.json" ]]; then
  "${PYTHON_BIN}" scripts/cache_touch_editor_base_actions.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --dataset_root "${DATASET_ROOT}" \
    --data_mix opentouch_keypoint_test \
    --statistics_dataset_name opentouch_keypoint_train \
    --cache_root "${RUN_ROOT}/cache_test" \
    --sample_mode random \
    --max_samples "${TEST_SAMPLES}" \
    --seed 43 \
    --edit_start_idx "${EDIT_START_IDX}" \
    --num_ddim_steps "${DDIM_STEPS}" \
    --cfg_scale "${CFG_SCALE}"
else
  echo "Test cache already exists; skipping."
fi

echo "[5/8] Train editor variants"
if [[ ! -f "${RUN_ROOT}/editor_matched/latest.pt" ]]; then
  "${PYTHON_BIN}" scripts/train_touch_editor.py \
    --cache_root "${RUN_ROOT}/cache_train" \
    --output_dir "${RUN_ROOT}/editor_matched" \
    --batch_size "${EDITOR_BATCH_SIZE}" \
    --max_steps "${EDITOR_STEPS}" \
    --device cuda
fi

if [[ ! -f "${RUN_ROOT}/editor_zero_touch/latest.pt" ]]; then
  "${PYTHON_BIN}" scripts/train_touch_editor.py \
    --cache_root "${RUN_ROOT}/cache_train" \
    --output_dir "${RUN_ROOT}/editor_zero_touch" \
    --batch_size "${EDITOR_BATCH_SIZE}" \
    --max_steps "${EDITOR_STEPS}" \
    --touch_ablation zero_touch \
    --device cuda
fi

if [[ ! -f "${RUN_ROOT}/editor_high_contact/latest.pt" ]]; then
  "${PYTHON_BIN}" scripts/train_touch_editor.py \
    --cache_root "${RUN_ROOT}/cache_train" \
    --output_dir "${RUN_ROOT}/editor_high_contact" \
    --batch_size "${EDITOR_BATCH_SIZE}" \
    --max_steps "${EDITOR_STEPS}" \
    --contact_subset high_contact \
    --high_contact_quantile 0.75 \
    --device cuda
fi

echo "[6/8] Evaluate editor variants and ablations"
for editor in matched zero_touch high_contact; do
  for ablation in matched zero_touch shuffled_touch future_touch_oracle; do
    "${PYTHON_BIN}" scripts/evaluate_touch_guided_actions.py \
      --cache_root "${RUN_ROOT}/cache_test" \
      --touch_editor_checkpoint "${RUN_ROOT}/editor_${editor}/latest.pt" \
      --output_path "${RUN_ROOT}/eval_${editor}_${ablation}.json" \
      --batch_size 256 \
      --fps 8 \
      --edit_times 0.33 \
      --ablation "${ablation}" \
      --device cuda
  done
done

echo "[7/8] Summarize eval JSON files"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import os

run_root = Path(os.environ.get("RUN_ROOT", "runs/online_touch_editor_large"))
rows = []
for path in sorted(run_root.glob("eval_*.json")):
    payload = json.loads(path.read_text())
    rows.append({
        "file": path.name,
        "base_mse": payload.get("base_mse"),
        "edit_mse": payload.get("edit_3_mse") or payload.get("edit_2_mse") or payload.get("edit_1_mse"),
        "improvement_pct": payload.get("edit_3_improvement_pct") or payload.get("edit_2_improvement_pct") or payload.get("edit_1_improvement_pct"),
        "prefix_change_l2": payload.get("edit_3_prefix_change_l2") or payload.get("edit_2_prefix_change_l2") or payload.get("edit_1_prefix_change_l2"),
        "touch_valid_rate": payload.get("touch_valid_rate"),
    })
(run_root / "eval_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY

echo "[8/8] Done. Outputs are under ${RUN_ROOT}"
