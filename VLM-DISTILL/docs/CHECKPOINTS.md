# Student Checkpoint Index

Checkpoint weights are local artifacts under `runs/` or `VLM-DISTILL/models/`. They are too large for normal git storage and are ignored by git.

## Encoder Student

Config:

```text
vitra/configs/vlm_distill_encoder_student_gigahands.json
```

Copied config:

```text
VLM-DISTILL/configs/vlm_distill_encoder_student_gigahands.json
```

Final checkpoint:

```text
runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt
```

Files:

```text
runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt/weights.pt
runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt/meta.json
```

Local retained checkpoints:

```text
epoch=0-step=8000.ckpt   weights.pt ~727 MB
epoch=0-step=9000.ckpt   weights.pt ~727 MB
epoch=0-step=10000.ckpt  weights.pt ~727 MB
```

Training log:

```text
runs/vlm_distill_encoder_student_gigahands/launch_logs/train_10k_tmux_20260427_0053.log
```

Summary:

- Student: DINOv2-base + DistilBERT encoder student.
- Objective: cognition-token MSE against the GigaHands VITRA teacher.
- Trained to: `10000` steps.
- Final logged loss: about `8e-05`.
- Final 100-random-clip action eval: `action_mse=51.858463`; this is poor because the action head was not trained with the student feature.

## ViTKD-Style Small Student

Config:

```text
vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json
```

Copied config:

```text
VLM-DISTILL/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json
```

Final checkpoint:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/checkpoints/vlm_distill_small_vitra_vitkd_full_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=5000.ckpt
```

Files:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/checkpoints/vlm_distill_small_vitra_vitkd_full_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=5000.ckpt/weights.pt
runs/vlm_distill_small_vitra_vitkd_full_gigahands/checkpoints/vlm_distill_small_vitra_vitkd_full_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=5000.ckpt/meta.json
```

Local retained checkpoints:

```text
epoch=0-step=3000.ckpt  weights.pt ~1.8 GB
epoch=0-step=3500.ckpt  weights.pt ~1.8 GB
epoch=0-step=4000.ckpt  weights.pt ~1.8 GB
epoch=0-step=4500.ckpt  weights.pt ~1.8 GB
epoch=0-step=5000.ckpt  weights.pt ~1.8 GB
```

Training log:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/train_5k_20260427_012738.log
```

Final action eval:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/action_eval/epoch=0-step=5000.ckpt/metrics.json
```

Summary:

- Student: small PaliGemma-style student.
- Objective: ViTKD-style cognition, shallow mimic, and deep generation losses.
- Trained to: `5000` steps.
- Final feature alignment to teacher:
  - `vlm_cognition_mse`: about `6.1e-05`
  - `vlm_cognition_cosine`: about `0.997`
- Action MSE remained poor in the 20-clip eval because the action head was not trainable in that run.

## Teacher-Action-Normalized Encoder Student

Config:

```text
vitra/configs/vlm_distill_encoder_student_gigahands_teacher_action_normalized.json
```

Copied config:

```text
VLM-DISTILL/configs/vlm_distill_encoder_student_gigahands_teacher_action_normalized.json
```

Latest checkpoint:

```text
VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=2000.ckpt
```

Files:

```text
VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=2000.ckpt/weights.pt
VLM-DISTILL/models/vlm_distill_encoder_student_gigahands_teacher_action_normalized_TB2_B1_bf16True/checkpoints/epoch=0-step=2000.ckpt/meta.json
```

Local retained checkpoints:

```text
epoch=0-step=500.ckpt   weights.pt ~694 MB
epoch=0-step=1000.ckpt  weights.pt ~694 MB
epoch=0-step=1500.ckpt  weights.pt ~694 MB
epoch=0-step=2000.ckpt  weights.pt ~694 MB
```

Training log:

```text
VLM-DISTILL/models/training_logs/vlm_distill_encoder_student_gigahands_teacher_action_normalized_20260428_000421.log
```

Final action eval:

```text
runs/vlm_distill_encoder_student_gigahands_teacher_action_normalized/action_eval/epoch=0-step=2000.ckpt/metrics.json
```

Best retained action eval:

```text
runs/vlm_distill_encoder_student_gigahands_teacher_action_normalized/action_eval/epoch=0-step=1500.ckpt/metrics.json
```

Summary:

- Student: DINOv2-base + DistilBERT encoder student initialized from the 10k cognition-only run.
- Objective: normalized cognition alignment plus teacher-action distillation.
- Trained to: `2000` steps.
- Best retained 20-clip action MSE: `0.409036` at step `1500`.
- Latest 20-clip action MSE: `0.446638` at step `2000`.
- Latest feature alignment to teacher:
  - `vlm_cognition_mse`: `0.00007798`
  - `vlm_cognition_cosine`: `0.996745`

## Teacher Checkpoint

Most distillation configs use this teacher:

```text
runs/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_train/checkpoints/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_stage1_TB1_B1_bf16True/checkpoints/epoch=0-step=28000.ckpt/weights.pt
```

Base VITRA checkpoint:

```text
checkpoints/vitra-vla-3b.pt
```

## Git Policy

Do not add `weights.pt`, `optimizer.pt`, or full `runs/` directories to git. Use this file as the tracked index and keep the actual artifacts local or upload them to an artifact store.
