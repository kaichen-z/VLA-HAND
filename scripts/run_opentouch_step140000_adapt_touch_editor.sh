#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/chonghej/scratch/chonghej/VLA-HAND}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/chonghej/conda_envs/vitra/bin/python}"
GPU="${GPU:-7}"
NPROC="${NPROC:-1}"
STAGE="${STAGE:-smoke}"
SESSION="${SESSION:-opentouch_step140000_adapt_touch_editor}"

CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_opentouch_keypoint_from_step140000.json}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_ROOT}/runs/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_train/checkpoints/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_stage1_TB2_B2_bf16True/checkpoints/epoch=0-step=140000.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/vitra_opentouch_keypoint_full_text_aligned}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/opentouch_step140000_adapt_touch_editor}"
ADAPT_OUTPUT_ROOT="${ADAPT_OUTPUT_ROOT:-${RUN_ROOT}/adapt_checkpoints}"
ADAPT_LOG_ROOT="${ADAPT_LOG_ROOT:-${RUN_ROOT}/adapt_logs}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval}"
BASE_CONFIG="${CONFIG}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${RUN_ROOT}/runtime_config.json}"

SMOKE_STEPS="${SMOKE_STEPS:-20}"
SMOKE_SAVE_STEPS="${SMOKE_SAVE_STEPS:-20}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-16}"
SMOKE_TEST_SAMPLES="${SMOKE_TEST_SAMPLES:-16}"
SMOKE_EDITOR_STEPS="${SMOKE_EDITOR_STEPS:-20}"
SMOKE_EVAL_SAMPLES="${SMOKE_EVAL_SAMPLES:-8}"

LARGE_STEPS="${LARGE_STEPS:-10000}"
LARGE_SAVE_STEPS="${LARGE_SAVE_STEPS:-5000}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
TEST_SAMPLES="${TEST_SAMPLES:-10000}"
EDITOR_STEPS="${EDITOR_STEPS:-5000}"
EDITOR_BATCH_SIZE="${EDITOR_BATCH_SIZE:-256}"

PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
TOTAL_BATCH="${TOTAL_BATCH:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EDIT_START_IDX="${EDIT_START_IDX:-3}"
DDIM_STEPS="${DDIM_STEPS:-10}"
CFG_SCALE="${CFG_SCALE:-5.0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/chonghej/scratch/chonghej/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/chonghej/scratch/chonghej/hf_cache/transformers}"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}" "${ADAPT_OUTPUT_ROOT}" "${ADAPT_LOG_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

prepare_runtime_config() {
  require_file "${BASE_CONFIG}"
  require_file "${SOURCE_CHECKPOINT}/weights.pt"
  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${SOURCE_CHECKPOINT}" "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

source_config = Path(sys.argv[1])
runtime_config = Path(sys.argv[2])
source_checkpoint = sys.argv[3]
dataset_root = sys.argv[4]

config = json.loads(source_config.read_text(encoding="utf-8"))
config["pretrain_path"] = source_checkpoint
config.setdefault("train_dataset", {})["data_root_dir"] = dataset_root
runtime_config.parent.mkdir(parents=True, exist_ok=True)
runtime_config.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
print(f"Runtime config: {runtime_config}")
PY
  CONFIG="${RUNTIME_CONFIG}"
}

latest_checkpoint() {
  local root="$1"
  "${PYTHON_BIN}" - "${root}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pattern = re.compile(r"step=(\d+)\.ckpt$")
best = None
for path in root.rglob("epoch=*-step=*.ckpt"):
    match = pattern.search(path.name)
    if not match:
        continue
    item = (int(match.group(1)), str(path))
    if best is None or item[0] > best[0]:
        best = item
if best is not None:
    print(best[1])
PY
}

preflight() {
  prepare_runtime_config
  require_file "${CONFIG}"
  require_file "${SOURCE_CHECKPOINT}/weights.pt"
  require_file "${DATASET_ROOT}/Annotation/opentouch_keypoint_train/episode_frame_index.npz"
  require_file "${DATASET_ROOT}/Annotation/opentouch_keypoint_test/episode_frame_index.npz"
  require_file "${DATASET_ROOT}/Annotation/statistics/opentouch_keypoint_train_keypoints_statistics.json"
"${PYTHON_BIN}" - "${CONFIG}" "${DATASET_ROOT}" <<'PY'
from pathlib import Path
import sys
from vitra.utils.config_utils import load_config
from scripts.cache_touch_editor_base_actions import build_dataset

config = load_config(sys.argv[1])
root = Path(sys.argv[2])
dataset = build_dataset(config, root, "opentouch_keypoint_train", statistics_dataset_name="opentouch_keypoint_train")
sample = dataset.episodic_dataset_core.__getitem__(0)
sample = dataset.episodic_dataset_core.transform_trajectory(sample.copy(), normalization=True)
assert sample["action_list"].shape == (16, 192), sample["action_list"].shape
assert sample["current_state"].shape == (212,), sample["current_state"].shape
assert "image_list" in sample
print({"dataset_len": len(dataset), "action_shape": sample["action_list"].shape, "state_shape": sample["current_state"].shape})
PY
}

train_adapt() {
  local task_name="$1"
  local steps="$2"
  local save_steps="$3"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC}" --standalone \
    scripts/train.py \
    --config "${CONFIG}" \
    --task_name "${task_name}" \
    --output_root "${ADAPT_OUTPUT_ROOT}" \
    --log_root "${ADAPT_LOG_ROOT}" \
    --batch_size "${PER_GPU_BATCH}" \
    --total_batch_size "${TOTAL_BATCH}" \
    --max_steps "${steps}" \
    --save_steps "${save_steps}" \
    --num_workers "${NUM_WORKERS}"
}

eval_vitra_pair() {
  local adapted_checkpoint="$1"
  local tag="$2"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  "${PYTHON_BIN}" scripts/evaluate_vitra_gt_actions.py \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --config "${CONFIG}" \
    --dataset_root "${DATASET_ROOT}" \
    --data_mix opentouch_keypoint_test \
    --statistics_dataset_name opentouch_keypoint_train \
    --output_path "${EVAL_ROOT}/${tag}_source_step140000.json" \
    --max_samples "${SMOKE_EVAL_SAMPLES}" \
    --num_ddim_steps "${DDIM_STEPS}" \
    --cfg_scale "${CFG_SCALE}"
  "${PYTHON_BIN}" scripts/evaluate_vitra_gt_actions.py \
    --checkpoint "${adapted_checkpoint}" \
    --config "${CONFIG}" \
    --dataset_root "${DATASET_ROOT}" \
    --data_mix opentouch_keypoint_test \
    --statistics_dataset_name opentouch_keypoint_train \
    --output_path "${EVAL_ROOT}/${tag}_adapted.json" \
    --max_samples "${SMOKE_EVAL_SAMPLES}" \
    --num_ddim_steps "${DDIM_STEPS}" \
    --cfg_scale "${CFG_SCALE}"
}

cache_actions() {
  local checkpoint="$1"
  local cache_train="$2"
  local cache_test="$3"
  local train_samples="$4"
  local test_samples="$5"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  if [[ ! -f "${cache_train}/summary.json" ]]; then
    "${PYTHON_BIN}" scripts/cache_touch_editor_base_actions.py \
      --checkpoint "${checkpoint}" \
      --config "${CONFIG}" \
      --dataset_root "${DATASET_ROOT}" \
      --data_mix opentouch_keypoint_train \
      --statistics_dataset_name opentouch_keypoint_train \
      --cache_root "${cache_train}" \
      --sample_mode random \
      --max_samples "${train_samples}" \
      --seed 42 \
      --edit_start_idx "${EDIT_START_IDX}" \
      --num_ddim_steps "${DDIM_STEPS}" \
      --cfg_scale "${CFG_SCALE}"
  fi
  if [[ ! -f "${cache_test}/summary.json" ]]; then
    "${PYTHON_BIN}" scripts/cache_touch_editor_base_actions.py \
      --checkpoint "${checkpoint}" \
      --config "${CONFIG}" \
      --dataset_root "${DATASET_ROOT}" \
      --data_mix opentouch_keypoint_test \
      --statistics_dataset_name opentouch_keypoint_train \
      --cache_root "${cache_test}" \
      --sample_mode random \
      --max_samples "${test_samples}" \
      --seed 43 \
      --edit_start_idx "${EDIT_START_IDX}" \
      --num_ddim_steps "${DDIM_STEPS}" \
      --cfg_scale "${CFG_SCALE}"
  fi
}

train_editors() {
  local cache_train="$1"
  local editor_root="$2"
  local steps="$3"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  if [[ ! -f "${editor_root}/gated_full_contrastive/latest.pt" ]]; then
    "${PYTHON_BIN}" scripts/train_touch_editor.py \
      --cache_root "${cache_train}" \
      --output_dir "${editor_root}/gated_full_contrastive" \
      --batch_size "${EDITOR_BATCH_SIZE}" \
      --max_steps "${steps}" \
      --editor_type tactile_gated \
      --condition_mode full \
      --context_dropout_prob 0.3 \
      --hand_scope right \
      --lambda_dev 1.0 \
      --lambda_delta 0.01 \
      --lambda_smooth 0.05 \
      --lambda_shuffle_zero 0.1 \
      --lambda_zero_zero 0.05 \
      --lambda_margin 0.05 \
      --shuffle_margin 0.05 \
      --lambda_touch_gate 0.01 \
      --contact_weighting observed_delta \
      --device cuda
  fi
  if [[ ! -f "${editor_root}/gated_no_base_contrastive/latest.pt" ]]; then
    "${PYTHON_BIN}" scripts/train_touch_editor.py \
      --cache_root "${cache_train}" \
      --output_dir "${editor_root}/gated_no_base_contrastive" \
      --batch_size "${EDITOR_BATCH_SIZE}" \
      --max_steps "${steps}" \
      --editor_type tactile_gated \
      --condition_mode no_base \
      --context_dropout_prob 0.3 \
      --hand_scope right \
      --lambda_dev 1.0 \
      --lambda_delta 0.01 \
      --lambda_smooth 0.05 \
      --lambda_shuffle_zero 0.1 \
      --lambda_zero_zero 0.05 \
      --lambda_margin 0.05 \
      --shuffle_margin 0.05 \
      --lambda_touch_gate 0.01 \
      --contact_weighting observed_delta \
      --device cuda
  fi
  if [[ ! -f "${editor_root}/zero_touch_control/latest.pt" ]]; then
    "${PYTHON_BIN}" scripts/train_touch_editor.py \
      --cache_root "${cache_train}" \
      --output_dir "${editor_root}/zero_touch_control" \
      --batch_size "${EDITOR_BATCH_SIZE}" \
      --max_steps "${steps}" \
      --editor_type tactile_gated \
      --condition_mode no_base \
      --touch_ablation zero_touch \
      --hand_scope right \
      --lambda_dev 1.0 \
      --lambda_delta 0.01 \
      --lambda_smooth 0.05 \
      --device cuda
  fi
}

eval_editors() {
  local cache_test="$1"
  local editor_root="$2"
  local eval_tag="$3"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  for editor in gated_full_contrastive gated_no_base_contrastive zero_touch_control; do
    for ablation in matched shuffled_touch zero_touch future_touch_oracle; do
      "${PYTHON_BIN}" scripts/evaluate_touch_guided_actions.py \
        --cache_root "${cache_test}" \
        --touch_editor_checkpoint "${editor_root}/${editor}/latest.pt" \
        --output_path "${EVAL_ROOT}/${eval_tag}_${editor}_${ablation}.json" \
        --batch_size 256 \
        --fps 8 \
        --edit_times 0.33 \
        --ablation "${ablation}" \
        --hand_scope right \
        --device cuda
    done
  done
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json
root = Path("${EVAL_ROOT}")
rows = []
for path in sorted(root.glob("${eval_tag}_*.json")):
    payload = json.loads(path.read_text())
    rows.append({
        "file": path.name,
        "base_mse": payload.get("base_mse"),
        "edit_1_mse": payload.get("edit_1_mse"),
        "edit_1_improvement_pct": payload.get("edit_1_improvement_pct"),
        "edit_1_right_edit_mse": payload.get("edit_1_right_edit_mse"),
        "edit_1_right_improvement_pct": payload.get("edit_1_right_improvement_pct"),
        "edit_1_prefix_change_l2": payload.get("edit_1_prefix_change_l2"),
        "edit_1_touch_gate_mean": payload.get("edit_1_touch_gate_mean"),
    })
(root / "${eval_tag}_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY
}

run_smoke() {
  preflight
  train_adapt "opentouch_keypoint_from_step140000_smoke" "${SMOKE_STEPS}" "${SMOKE_SAVE_STEPS}"
  local ckpt
  ckpt="$(latest_checkpoint "${ADAPT_OUTPUT_ROOT}")"
  require_dir "${ckpt}"
  echo "Smoke adapted checkpoint: ${ckpt}"
  eval_vitra_pair "${ckpt}" "smoke"
  cache_actions "${ckpt}" "${RUN_ROOT}/smoke_cache_train" "${RUN_ROOT}/smoke_cache_test" "${SMOKE_TRAIN_SAMPLES}" "${SMOKE_TEST_SAMPLES}"
  EDITOR_BATCH_SIZE="${SMOKE_TRAIN_SAMPLES}" train_editors "${RUN_ROOT}/smoke_cache_train" "${RUN_ROOT}/smoke_editors" "${SMOKE_EDITOR_STEPS}"
  eval_editors "${RUN_ROOT}/smoke_cache_test" "${RUN_ROOT}/smoke_editors" "smoke_editor"
}

run_large_inner() {
  preflight
  train_adapt "opentouch_keypoint_from_step140000_large" "${LARGE_STEPS}" "${LARGE_SAVE_STEPS}"
  local ckpt
  ckpt="$(latest_checkpoint "${ADAPT_OUTPUT_ROOT}")"
  require_dir "${ckpt}"
  echo "Large adapted checkpoint: ${ckpt}"
  SMOKE_EVAL_SAMPLES="${TEST_SAMPLES}" eval_vitra_pair "${ckpt}" "large"
  cache_actions "${ckpt}" "${RUN_ROOT}/large_cache_train" "${RUN_ROOT}/large_cache_test" "${TRAIN_SAMPLES}" "${TEST_SAMPLES}"
  train_editors "${RUN_ROOT}/large_cache_train" "${RUN_ROOT}/large_editors" "${EDITOR_STEPS}"
  eval_editors "${RUN_ROOT}/large_cache_test" "${RUN_ROOT}/large_editors" "large_editor"
}

case "${STAGE}" in
  preflight)
    preflight
    ;;
  smoke)
    run_smoke
    ;;
  large)
    require_file "$(command -v tmux)"
    log_path="${RUN_ROOT}/${SESSION}.log"
    tmux new-session -d -s "${SESSION}" "cd '${REPO_ROOT}' && STAGE=large_inner GPU='${GPU}' NPROC='${NPROC}' RUN_ROOT='${RUN_ROOT}' CONFIG='${CONFIG}' SOURCE_CHECKPOINT='${SOURCE_CHECKPOINT}' DATASET_ROOT='${DATASET_ROOT}' PYTHON_BIN='${PYTHON_BIN}' bash '${BASH_SOURCE[0]}' 2>&1 | tee '${log_path}'"
    echo "Launched tmux session ${SESSION}. Log: ${log_path}"
    ;;
  large_inner)
    run_large_inner
    ;;
  *)
    echo "Unsupported STAGE=${STAGE}; use preflight, smoke, large, or large_inner." >&2
    exit 1
    ;;
esac
