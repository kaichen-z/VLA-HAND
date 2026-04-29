#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <distilled-checkpoint-dir-or-weights.pt> [extra evaluate_gigahands_stage1.py args...]" >&2
    exit 2
fi

CHECKPOINT="$1"
shift

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-runs/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_train/checkpoints/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_stage1_TB1_B1_bf16True/checkpoints/epoch=0-step=28000.ckpt/weights.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-./checkpoints/vitra-vla-3b.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/vlm_distill_gigahands_cognition/action_eval/manual}"

python tools/evaluate_gigahands_stage1.py \
    --config vitra/configs/human_pretrain_gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked.json \
    --dataset_root datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked \
    --data_mix gigahands_real_test \
    --checkpoint "${CHECKPOINT}" \
    --label distilled \
    --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
    --teacher_label gigahands_teacher \
    --base_checkpoint "${BASE_CHECKPOINT}" \
    --base_label base3b \
    --num_eval_clips 20 \
    --output_dir "${OUTPUT_ROOT}/triad" \
    --no_videos \
    "$@"
