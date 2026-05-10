# Distilled VLM Models and Results

This document summarizes the locally evaluated VLM distillation runs for VLA-HAND on GigaHands BRICS001 camera 0. All reported action metrics are normalized MSE values, so lower is better. Cognition metrics measure alignment between the distilled/student VLM cognition feature and the GigaHands teacher cognition feature.

## Training and Testing Data

The distillation experiments use `datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked`, a GigaHands real-data subset restricted to BRICS001 camera 0 clips linked with hand keypoints. The training split is `gigahands_real_train`, with `action_type` set to `keypoints` and statistics loaded from `gigahands_real_train`.

Testing uses the matching `gigahands_real_test` split from the same dataset root. The local upload package describes the same structure: `Annotation/gigahands_real_train/` for training annotations, `Annotation/gigahands_real_test/` for testing annotations, `Annotation/statistics/` for split statistics, and `Video/GigaHands_root/` for the RGB videos referenced by annotations.

The evaluated subsets differ by run:

- Triad evaluations use 5 test clips, giving 80 valid bimanual frames.
- Most checkpoint sweeps use 20 test clips, giving 320 valid bimanual frames.
- The encoder-student cognition eval also includes a 100-random-clip run with 1513 valid frames and 1481 bimanual frames.

## Model Setups

**Base3B cognition-distilled VLM** starts from `./checkpoints/vitra-vla-3b.pt`. It keeps the Base3B VITRA/VLM layout and freezes the vision encoder while training the VLM-side representation to imitate the GigaHands teacher cognition feature. This setup gets very strong cognition alignment but only a small action improvement.

**Base3B cognition + ground-truth action** continues from the Base3B cognition-distilled checkpoint at step 5000. It adds ground-truth action supervision with `action_distill_target=gt`. In the evaluated runs, this damages cognition alignment and does not produce a useful action improvement.

**Encoder-student cognition model** replaces the large VLM-side encoder with a smaller student layout: DINOv2-base vision encoder, DistilBERT text encoder, and a fusion projection into the 2304-dimensional VITRA cognition feature. It trains with weighted cognition distillation. The model learns teacher-like cognition features, but the action head remains ineffective without explicit action training.

**Encoder-student + teacher-action normalized model** starts from the 10k-step encoder-student cognition checkpoint. It trains the action head, FOV encoder, DistilBERT text encoder, and student fusion projection with a normalized loss against teacher actions (`action_distill_target=teacher`). This is the best evaluated distilled VLM action model.

**Small VITRA ViTKD model** uses a smaller VITRA student trained with ViTKD-style losses: cognition loss, shallow mimic loss on layers 0 and 1, and deep-generation loss on the final layer. Direct deep mimic is disabled. The run improves cognition alignment over time but leaves action prediction broken in the evaluated path.

## Best Checkpoints Overview

This table shows the most useful comparison point for each model family. `gap_vs_teacher` is `student action_mse - teacher action_mse` on the same test subset.

| Model family | Representative checkpoint | Eval clips / frames | student_action | teacher_action | gap_vs_teacher | cognition cosine | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Base3B cognition-distilled VLM | step1000 triad | 5 / 80 | 15.561666 | 0.119079 | 15.442587 | 0.997294 | Good cognition, weak action |
| Base3B cognition + GT action | step3000 triad | 5 / 80 | 17.245607 | 0.119079 | 17.126528 | 0.726112 | Action and cognition both weak |
| Encoder-student cognition | step10000, 100 random clips | 100 / 1513 valid, 1481 bimanual | 51.858463 | 0.160412 | 51.698051 | 0.995944 | Good cognition, action failed |
| Encoder-student + teacher-action normalized | step1500 | 20 / 320 | 0.409036 | 0.131295 | 0.277741 | 0.996596 | BEST action result |
| Small VITRA ViTKD | step5000 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 0.997453 | Best ViTKD cognition, action failed |

## Teacher Action Reference

Teacher rows are repeated by eval subset because the tested clips differ across runs.

| Eval subset | Clips / frames | teacher_action_mse | teacher_left_mse | teacher_right_mse | teacher_dual_hand_mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Triad eval | 5 / 80 | 0.119079 | 0.135317 | 0.081963 | 0.119079 |
| 20-clip eval | 20 / 320 | 0.131295 | 0.140411 | 0.110458 | 0.131295 |
| 100-random-clip eval | 100 / 1513 valid, 1481 bimanual | 0.160412 | 0.179639 | 0.115824 | 0.160311 |

## Detailed Metrics by Model Family

### Base3B Distillation Runs

- Layout: Base VITRA-VLA-3B student; the cognition-only run freezes the vision encoder.
- Objectives: cognition-only distillation for the first row, then ground-truth action loss after cognition distillation for the GT-action rows.

| Run | Eval clips / frames | student_action | teacher_action | gap_vs_teacher | left_action_mse | right_action_mse | dual_hand_action_mse | vlm_cognition_mse | vlm_cognition_cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cognition-only, step1000 triad | 5 / 80 | 15.561666 | 0.119079 | 15.442587 | 2.508740 | 45.396923 | 15.561666 | 0.00007131 | 0.997294 |
| Cognition + GT action, step1000 triad | 5 / 80 | 22.014793 | 0.119079 | 21.895714 | 11.471155 | 46.114548 | 22.014793 | 0.00845045 | 0.760031 |
| Cognition + GT action, step3000 triad | 5 / 80 | 17.245607 | 0.119079 | 17.126528 | 4.795616 | 45.702736 | 17.245607 | 0.01001881 | 0.726112 |

### Encoder-Student Cognition

- Layout: DINOv2-base vision encoder, DistilBERT text encoder, and 2304-dimensional fusion projection.
- Objective: weighted cognition feature distillation; no explicit action training in the trained row.

| Run | Eval clips / frames | student_action | teacher_action | gap_vs_teacher | left_action_mse | right_action_mse | dual_hand_action_mse | vlm_cognition_mse | vlm_cognition_cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untrained 20 clips | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.01420629 | 0.015645 |
| step10000, 100 random clips | 100 / 1513 valid, 1481 bimanual | 51.858463 | 0.160412 | 51.698051 | 51.866955 | 51.838783 | 51.840664 | 0.00009851 | 0.995944 |

### Encoder-Student + Teacher-Action Normalized

- Layout: encoder student with trainable action head, FOV encoder, DistilBERT text encoder, and student fusion projection.
- Objective: normalized teacher-action loss from the 10k-step encoder-student cognition checkpoint.

| Checkpoint | Eval clips / frames | student_action | teacher_action | gap_vs_teacher | left_action_mse | right_action_mse | dual_hand_action_mse | vlm_cognition_mse | vlm_cognition_cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step500 | 20 / 320 | 0.621532 | 0.131295 | 0.490237 | 0.568480 | 0.742795 | 0.621532 | 0.00009503 | 0.996043 |
| step1000 | 20 / 320 | 0.430019 | 0.131295 | 0.298724 | 0.392736 | 0.515237 | 0.430019 | 0.00008925 | 0.996274 |
| step1500 BEST | 20 / 320 | 0.409036 | 0.131295 | 0.277741 | 0.399437 | 0.430975 | 0.409036 | 0.00008157 | 0.996596 |
| step2000 | 20 / 320 | 0.446638 | 0.131295 | 0.315343 | 0.392757 | 0.569795 | 0.446638 | 0.00007798 | 0.996745 |

### Small VITRA ViTKD

- Layout: Small VITRA student.
- Objective: cognition loss, shallow mimic on layers 0 and 1, and final-layer deep-generation loss; direct deep mimic is disabled.

| Checkpoint | Eval clips / frames | student_action | teacher_action | gap_vs_teacher | left_action_mse | right_action_mse | dual_hand_action_mse | vlm_cognition_mse | vlm_cognition_cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step500 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00013460 | 0.994415 |
| step1000 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00013654 | 0.994318 |
| step1500 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00009837 | 0.995892 |
| step2000 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00010702 | 0.995529 |
| step2500 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00008443 | 0.996477 |
| step3000 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00008246 | 0.996556 |
| step3500 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00012176 | 0.994917 |
| step4000 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00006920 | 0.997112 |
| step4500 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00007033 | 0.997066 |
| step5000 | 20 / 320 | 52.131599 | 0.131295 | 52.000304 | 52.191635 | 51.994396 | 52.131599 | 0.00006116 | 0.997453 |

## Overall Takeaways

The experiments separate cognition alignment from action quality. Several students reach excellent cognition cosine similarity around 0.996 to 0.997, but most still fail to produce good actions. The best result is the encoder-student + teacher-action normalized checkpoint at step1500: student `action_mse=0.409036` versus teacher `action_mse=0.131295`, for a `gap_vs_teacher=0.277741`.

The Base3B cognition-only run shows that matching the teacher cognition feature is not sufficient for action quality. The Base3B ground-truth action continuation weakens cognition alignment and does not recover strong actions. The Small VITRA ViTKD run is the clearest example of this mismatch: cognition alignment becomes very strong by step5000, but action MSE stays at 52.131599 across all evaluated checkpoints.

## Source Artifacts

- Dataset description: `/scratch/hannahyao24/hf_upload/vlm-distill-gigahands-brics001-cam0/README.md`
- Base training data config: `vitra/configs/human_pretrain_gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked.json`
- Distillation configs: `vitra/configs/vlm_distill*.json`
- Existing encoder-student action summary: `vlm_encoder_student_teacher_action_results.md`
- Metrics: `runs/*/action_eval/**/metrics.json`
