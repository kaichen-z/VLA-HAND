# VLM-DISTILL

This folder collects the VLM/student distillation work for VLA-HAND without moving the original source files.

The files here are copied from their original repo locations so the distillation code, configs, tests, and notes are easy to inspect together. The original training entrypoints still live under `scripts/`, `vitra/`, and `tests/`; this folder is an organized snapshot, not a replacement Python package.

## Student Models

### Encoder student

- Model file: `models/vitra_encoder_student.py`
- Main config: `configs/vlm_distill_encoder_student_gigahands.json`
- Student design: DINOv2 vision encoder + DistilBERT text encoder + fusion projection to a 2304-d VITRA conditioning feature.
- Training objective: cognition-token distillation from the GigaHands VITRA teacher.
- Final local checkpoint: step 10000.

### Teacher-action-normalized encoder student

- Model file: `models/vitra_encoder_student.py`
- Main config: `configs/vlm_distill_encoder_student_gigahands_teacher_action_normalized.json`
- Student design: same encoder student, initialized from the step 10000 cognition-only checkpoint, with the DiT action head trainable.
- Training objective: normalized cognition alignment plus teacher-action distillation.
- Latest local checkpoint: step 2000.
- Best retained 20-clip action eval: step 1500, `action_mse=0.409036`.
- Latest 20-clip action eval: step 2000, `action_mse=0.446638`.

### ViTKD-style small student

- Model file: `models/vitra_small_paligemma_student.py`
- Main config: `configs/vlm_distill_small_vitra_vitkd_full_gigahands.json`
- Student design: smaller PaliGemma-style student.
- Training objective: ViTKD-style cognition, shallow mimic, and deep generation losses.
- Final local checkpoint: step 5000.

## Core Files

- `scripts/train_vlm_distill.py`: shared distillation trainer.
- `scripts/run_vlm_backbone_distill_gigahands.sh`: generic distillation launch wrapper.
- `scripts/run_small_vitra_vitkd_short_ablations.sh`: ViTKD ablation sweep.
- `scripts/evaluate_vlm_distill_gigahands.sh`: teacher/base/student action evaluation wrapper.
- `configs/`: copied distillation and teacher configs.
- `tests/test_vlm_distill.py`: unit tests for distillation behavior.
- `docs/CHECKPOINTS.md`: local checkpoint index.
- `docs/encoder_student_teacher_action_results.md`: teacher-action encoder student evaluation notes.
- `docs/vitra_dit_vlm_distillation_results.md`: VITRA DiT/VLM distillation result notes.
- `docs/vlm_distilled_models_summary.md`: distilled model comparison summary.
- `docs/ViTKD.pdf`: reference paper copy.

## Training Commands

Run encoder-student cognition distillation from the repo root:

```bash
NPROC_PER_NODE=1 \
CONFIG=vitra/configs/vlm_distill_encoder_student_gigahands.json \
bash scripts/run_vlm_backbone_distill_gigahands.sh
```

Run the ViTKD-style small student:

```bash
CUDA_VISIBLE_DEVICES=7 \
torchrun --nproc_per_node=1 --standalone \
  scripts/train_vlm_distill.py \
  --config vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json
```

Run the current teacher-action-normalized encoder student:

```bash
CUDA_VISIBLE_DEVICES=7 \
torchrun --nproc_per_node=1 --standalone \
  scripts/train_vlm_distill.py \
  --config vitra/configs/vlm_distill_encoder_student_gigahands_teacher_action_normalized.json
```

Evaluate a student checkpoint:

```bash
bash scripts/evaluate_vlm_distill_gigahands.sh \
  runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt
```

## Important Note

Checkpoint weights are local artifacts under `runs/` or `VLM-DISTILL/models/` and are ignored by git. See `docs/CHECKPOINTS.md` for exact local paths.

The latest teacher-action-normalized student metrics are summarized in `docs/encoder_student_teacher_action_results.md`.
