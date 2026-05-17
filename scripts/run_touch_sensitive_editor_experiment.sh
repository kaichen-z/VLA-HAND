#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/chonghej/scratch/chonghej/VLA-HAND}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/chonghej/conda_envs/vitra/bin/python}"
GPU="${GPU:-7}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/touch_sensitive_editor}"
DEFAULT_CACHE_ROOT="${REPO_ROOT}/runs/online_touch_editor_large"
LEGACY_CACHE_ROOT="/home/chonghej/scratch/chonghej/vla_touch/runs/online_touch_editor_large"
TRAIN_CACHE="${TRAIN_CACHE:-${DEFAULT_CACHE_ROOT}/cache_train}"
TEST_CACHE="${TEST_CACHE:-${DEFAULT_CACHE_ROOT}/cache_test}"

if [[ ! -f "${TRAIN_CACHE}/summary.json" && -f "${LEGACY_CACHE_ROOT}/cache_train/summary.json" ]]; then
  TRAIN_CACHE="${LEGACY_CACHE_ROOT}/cache_train"
fi
if [[ ! -f "${TEST_CACHE}/summary.json" && -f "${LEGACY_CACHE_ROOT}/cache_test/summary.json" ]]; then
  TEST_CACHE="${LEGACY_CACHE_ROOT}/cache_test"
fi

TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
TEST_SAMPLES="${TEST_SAMPLES:-10000}"
EDITOR_STEPS="${EDITOR_STEPS:-10000}"
EDITOR_BATCH_SIZE="${EDITOR_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
PRETRAINED_TOUCH_ENCODER_CHECKPOINT="${PRETRAINED_TOUCH_ENCODER_CHECKPOINT:-${REPO_ROOT}/checkpoints/opentouch-vp2t-encoder-best/epoch_280.pt}"
if [[ ! -f "${PRETRAINED_TOUCH_ENCODER_CHECKPOINT}" ]]; then
  PRETRAINED_TOUCH_ENCODER_CHECKPOINT="${REPO_ROOT}/runs/opentouch_official_encoder_train/logs/opentouch_encoder_restart_20260515_085839/checkpoints/epoch_280.pt"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"

if [[ ! -f "${TRAIN_CACHE}/summary.json" || ! -f "${TEST_CACHE}/summary.json" ]]; then
  echo "Missing touch-editor cache."
  echo "Expected train cache: ${TRAIN_CACHE}/summary.json"
  echo "Expected test cache: ${TEST_CACHE}/summary.json"
  echo "Run scripts/run_online_touch_editor_large.sh first, or pass TRAIN_CACHE/TEST_CACHE."
  exit 1
fi

train_editor() {
  local variant="$1"
  shift
  local out="${RUN_ROOT}/editor_${variant}"
  if [[ -f "${out}/latest.pt" ]]; then
    echo "Editor ${variant} already exists; skipping train."
    return
  fi
  "${PYTHON_BIN}" scripts/train_touch_editor.py \
    --cache_root "${TRAIN_CACHE}" \
    --output_dir "${out}" \
    --batch_size "${EDITOR_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_steps "${EDITOR_STEPS}" \
    --device cuda \
    "$@"
}

eval_editor() {
  local variant="$1"
  local ablation="$2"
  local max_sample_args=()
  local hand_scope="both"
  if [[ "${variant}" == gated_* ]]; then
    hand_scope="right"
  fi
  if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
    max_sample_args=(--max_samples "${EVAL_MAX_SAMPLES}")
  fi
  "${PYTHON_BIN}" scripts/evaluate_touch_guided_actions.py \
    --cache_root "${TEST_CACHE}" \
    --touch_editor_checkpoint "${RUN_ROOT}/editor_${variant}/latest.pt" \
    --output_path "${RUN_ROOT}/eval_${variant}_${ablation}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    "${max_sample_args[@]}" \
    --hand_scope "${hand_scope}" \
    --fps 8 \
    --edit_times 0.33 \
    --ablation "${ablation}" \
    --device cuda
}

echo "[1/4] Train baseline residual full editor"
train_editor baseline_residual_full \
  --editor_type residual \
  --condition_mode full

echo "[2/4] Train tactile-sensitive gated variants"
train_editor gated_right_dropout \
  --editor_type tactile_gated \
  --condition_mode full \
  --hand_scope right \
  --context_dropout_prob 0.30 \
  --contact_weighting observed_delta \
  --lambda_touch_gate 0.01

train_editor gated_right_contrastive \
  --editor_type tactile_gated \
  --condition_mode full \
  --hand_scope right \
  --context_dropout_prob 0.30 \
  --contact_weighting observed_delta \
  --lambda_shuffle_margin 1.0 \
  --shuffle_margin 0.03 \
  --lambda_zero_delta 1.0 \
  --lambda_touch_gate 0.01

train_editor gated_no_base_contrastive \
  --editor_type tactile_gated \
  --condition_mode no_base \
  --hand_scope right \
  --context_dropout_prob 0.30 \
  --contact_weighting observed_delta \
  --lambda_shuffle_margin 1.0 \
  --shuffle_margin 0.03 \
  --lambda_zero_delta 1.0 \
  --lambda_touch_gate 0.01

train_editor pretrained_right_contrastive \
  --editor_type pretrained_tactile_gated \
  --condition_mode full \
  --hand_scope right \
  --context_dropout_prob 0.30 \
  --pretrained_touch_encoder_checkpoint "${PRETRAINED_TOUCH_ENCODER_CHECKPOINT}" \
  --pretrained_touch_hand right \
  --freeze_pretrained_touch_encoder true \
  --contact_weighting observed_delta \
  --lambda_shuffle_margin 1.0 \
  --shuffle_margin 0.03 \
  --lambda_zero_delta 1.0 \
  --lambda_touch_gate 0.01

echo "[3/4] Evaluate variants"
for variant in baseline_residual_full gated_right_dropout gated_right_contrastive gated_no_base_contrastive pretrained_right_contrastive; do
  for ablation in matched shuffled_touch zero_touch future_touch_oracle; do
    eval_editor "${variant}" "${ablation}"
  done
done

echo "[4/4] Summarize tactile sensitivity"
"${PYTHON_BIN}" scripts/summarize_touch_sensitive_editor.py \
  --run_root "${RUN_ROOT}" \
  --output_path "${RUN_ROOT}/sensitivity_summary.json"

echo "Done. Outputs are under ${RUN_ROOT}"
