#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-runs/vlm_distill_gigahands_cognition/checkpoints/vlm_distill_gigahands_cognition_base3b_TB2_B1_bf16True/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/vlm_distill_gigahands_cognition/action_eval/checkpoint_sweep_20clips}"
NUM_EVAL_CLIPS="${NUM_EVAL_CLIPS:-20}"

for STEP in 1000 2000 3000 4000 5000; do
    CHECKPOINT="${RUN_DIR}/epoch=0-step=${STEP}.ckpt"
    if [[ ! -e "${CHECKPOINT}/weights.pt" ]]; then
        echo "Skipping missing checkpoint: ${CHECKPOINT}" >&2
        continue
    fi

    OUTPUT_ROOT="${OUTPUT_ROOT}/step${STEP}" \
        bash scripts/evaluate_vlm_distill_gigahands.sh \
        "${CHECKPOINT}" \
        --num_eval_clips "${NUM_EVAL_CLIPS}"
done
