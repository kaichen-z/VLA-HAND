# VLM Encoder Student Teacher-Action Results

## Setup

- Run: `vlm_distill_encoder_student_gigahands_teacher_action_normalized`.
- Config: `vitra/configs/vlm_distill_encoder_student_gigahands_teacher_action_normalized.json`.
- Student: DINOv2-base vision encoder + DistilBERT text encoder + fusion projection to a 2304-d VITRA cognition feature, initialized from the 10k cognition-only encoder student.
- Trainable modules: action head, FOV encoder, DistilBERT text encoder, and student fusion projection.
- Teacher: GigaHands BRICS camera 0 VITRA checkpoint `epoch=0-step=28000.ckpt`.
- Dataset: `datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked`.
- Eval split: `gigahands_real_test`, sequential clips `0-19`.
- Eval protocol: 20 clips, 320 valid bimanual frames, DDIM 10 steps, CFG scale `5.0`.
- Metric: normalized action MSE; lower is better.

## Checkpoints

Weights are local artifacts and are not committed:

```text
VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=500.ckpt/weights.pt
VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=1000.ckpt/weights.pt
VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=1500.ckpt/weights.pt
VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=2000.ckpt/weights.pt
```

## Action Eval Sweep

| Step | action_mse | left_action_mse | right_action_mse | dual_hand_action_mse | vlm_cognition_mse | vlm_cognition_cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.621532 | 0.568480 | 0.742795 | 0.621532 | 0.00009503 | 0.996043 |
| 1000 | 0.430019 | 0.392736 | 0.515237 | 0.430019 | 0.00008925 | 0.996274 |
| 1500 | 0.409036 | 0.399437 | 0.430975 | 0.409036 | 0.00008157 | 0.996596 |
| 2000 | 0.446638 | 0.392757 | 0.569795 | 0.446638 | 0.00007798 | 0.996745 |

Best retained action eval:

```text
epoch=0-step=1500.ckpt
```

Latest retained checkpoint:

```text
epoch=0-step=2000.ckpt
```

## Baselines on the Same 20 Clips

| Model | Checkpoint | action_mse | left_action_mse | right_action_mse | dual_hand_action_mse |
| --- | --- | ---: | ---: | ---: | ---: |
| Base VITRA-VLA-3B | `./checkpoints/vitra-vla-3b.pt` | 16.234415 | 3.682127 | 44.925354 | 16.234415 |
| GigaHands teacher | `runs/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_train/checkpoints/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_stage1_TB1_B1_bf16True/checkpoints/epoch=0-step=28000.ckpt/weights.pt` | 0.131295 | 0.140411 | 0.110458 | 0.131295 |
| Best student step 1500 | `VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=1500.ckpt` | 0.409036 | 0.399437 | 0.430975 | 0.409036 |
| Latest student step 2000 | `VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=2000.ckpt` | 0.446638 | 0.392757 | 0.569795 | 0.446638 |

## Relative Results

- Step 1500 improves over base VITRA-VLA-3B by `15.825379` absolute action MSE, a `97.48%` reduction on this 20-clip eval.
- Step 1500 is `0.277741` action MSE above the GigaHands teacher on the same clips.
- Step 2000 improves over base VITRA-VLA-3B by `15.787777` absolute action MSE, a `97.25%` reduction on this 20-clip eval.
- Step 2000 is `0.315343` action MSE above the GigaHands teacher on the same clips.

## Source Artifacts

```text
runs/vlm_distill_encoder_student_gigahands_teacher_action_normalized/action_eval/epoch=0-step=500.ckpt/metrics.json
runs/vlm_distill_encoder_student_gigahands_teacher_action_normalized/action_eval/epoch=0-step=1000.ckpt/metrics.json
runs/vlm_distill_encoder_student_gigahands_teacher_action_normalized/action_eval/epoch=0-step=1500.ckpt/metrics.json
runs/vlm_distill_encoder_student_gigahands_teacher_action_normalized/action_eval/epoch=0-step=2000.ckpt/metrics.json
```
