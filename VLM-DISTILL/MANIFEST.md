# VLM-DISTILL Manifest

This manifest maps the copied files in `VLM-DISTILL/` back to their original repo locations.

## Docs

| Copied file | Original file |
| --- | --- |
| `README.md` | new distillation index |
| `MANIFEST.md` | new file inventory |
| `docs/CHECKPOINTS.md` | new checkpoint index |
| `docs/encoder_student_teacher_action_results.md` | `vlm_encoder_student_teacher_action_results.md` |
| `docs/vitra_dit_vlm_distillation_results.md` | `vitra_dit_vlm_distillation_results.md` |
| `docs/vlm_distilled_models_summary.md` | `vlm_distilled_models_summary.md` |
| `docs/ViTKD.pdf` | `ViTKD.pdf` |

## Scripts

| Copied file | Original file |
| --- | --- |
| `scripts/train_vlm_distill.py` | `scripts/train_vlm_distill.py` |
| `scripts/run_vlm_backbone_distill_gigahands.sh` | `scripts/run_vlm_backbone_distill_gigahands.sh` |
| `scripts/run_small_vitra_vitkd_large.sh` | `scripts/run_small_vitra_vitkd_large.sh` |
| `scripts/run_small_vitra_vitkd_short_ablations.sh` | `scripts/run_small_vitra_vitkd_short_ablations.sh` |
| `scripts/evaluate_vlm_distill_gigahands.sh` | `scripts/evaluate_vlm_distill_gigahands.sh` |
| `scripts/evaluate_vlm_distill_cognition_sweep_gigahands.sh` | `scripts/evaluate_vlm_distill_cognition_sweep_gigahands.sh` |

## Models

| Copied file | Original file |
| --- | --- |
| `models/vitra_encoder_student.py` | `vitra/models/vla/vitra_encoder_student.py` |
| `models/vitra_small_paligemma_student.py` | `vitra/models/vla/vitra_small_paligemma_student.py` |

## Tests

| Copied file | Original file |
| --- | --- |
| `tests/test_vlm_distill.py` | `tests/test_vlm_distill.py` |

## Configs

| Copied file | Original file |
| --- | --- |
| `configs/human_pretrain_gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked.json` | `vitra/configs/human_pretrain_gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked.json` |
| `configs/human_pretrain_gigahands_real_full_keypoints_vitra3b_linked.json` | `vitra/configs/human_pretrain_gigahands_real_full_keypoints_vitra3b_linked.json` |
| `configs/human_pretrain_gigahands_real_keypoints_smoke.json` | `vitra/configs/human_pretrain_gigahands_real_keypoints_smoke.json` |
| `configs/vlm_distill_encoder_student_gigahands.json` | `vitra/configs/vlm_distill_encoder_student_gigahands.json` |
| `configs/vlm_distill_encoder_student_gigahands_action.json` | `vitra/configs/vlm_distill_encoder_student_gigahands_action.json` |
| `configs/vlm_distill_encoder_student_gigahands_joint_normalized.json` | `vitra/configs/vlm_distill_encoder_student_gigahands_joint_normalized.json` |
| `configs/vlm_distill_gigahands_cognition.json` | `vitra/configs/vlm_distill_gigahands_cognition.json` |
| `configs/vlm_distill_gigahands_cognition_action_gt.json` | `vitra/configs/vlm_distill_gigahands_cognition_action_gt.json` |
| `configs/vlm_distill_small_vitra_bad_direct_deep_mimic_gigahands.json` | `vitra/configs/vlm_distill_small_vitra_bad_direct_deep_mimic_gigahands.json` |
| `configs/vlm_distill_small_vitra_cognition_only_gigahands.json` | `vitra/configs/vlm_distill_small_vitra_cognition_only_gigahands.json` |
| `configs/vlm_distill_small_vitra_deep_gen_only_gigahands.json` | `vitra/configs/vlm_distill_small_vitra_deep_gen_only_gigahands.json` |
| `configs/vlm_distill_small_vitra_shallow_only_gigahands.json` | `vitra/configs/vlm_distill_small_vitra_shallow_only_gigahands.json` |
| `configs/vlm_distill_small_vitra_vitkd_full_gigahands.json` | `vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json` |
| `configs/vlm_distill_small_vitra_vitkd_full_gigahands_large.json` | `vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands_large.json` |
| `configs/vlm_distill_stage2_action_head_only_strict.json` | `vitra/configs/vlm_distill_stage2_action_head_only_strict.json` |
| `configs/vlm_distill_stage2_action_head_plus_adapters.json` | `vitra/configs/vlm_distill_stage2_action_head_plus_adapters.json` |
| `configs/vlm_distill_stage2_action_head_plus_adapters_normalized.json` | `vitra/configs/vlm_distill_stage2_action_head_plus_adapters_normalized.json` |
| `configs/vlm_distill_stage2_action_head_plus_vlm_small_lr_normalized.json` | `vitra/configs/vlm_distill_stage2_action_head_plus_vlm_small_lr_normalized.json` |

## Excluded Artifacts

The following are intentionally not copied:

- `runs/**/weights.pt`
- `runs/**/optimizer.pt`
- `datasets/`
- wandb run binaries
- Python cache directories

See `docs/CHECKPOINTS.md` for local checkpoint paths.
