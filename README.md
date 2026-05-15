# VLA-HAND

This repository contains our VITRA-based hand-action prediction, compression, and
test-time editing experiments.

The current code supports four related workflows:

- GigaHands fine-tuning and evaluation for VITRA future hand-action prediction.
- VITRA student distillation for faster VLA backend and diffusion-head inference.
- OpenTouch action editing with a learned tactile-conditioned residual editor.
- Test-time diffusion editing / replanning with polynomial guidance and tactile DPS-style guidance.

The model predicts future hand actions, not future RGB frames. In the GigaHands setup,
the action target is a 16-step chunk in VITRA's `keypoints` action interface, using
converted `keypoints_3d_mano` targets.

## Core Setup

The VITRA stage-1 interface used by these experiments is:

| Field | Shape | Meaning |
| --- | ---: | --- |
| RGB image | image | current egocentric frame |
| instruction | text | task language |
| current state | `[212]` | current hand/action state used by VITRA |
| action chunk | `[16, 192]` | predicted future hand-action trajectory |
| action mask | `[16, 192]` | valid target dimensions |

Important default inference settings:

- action horizon: `16` steps
- action dimension: `192`
- DDIM steps: `10`
- CFG scale: `5.0`

## Data

### GigaHands

The cleaned converted GigaHands dataset is released as:

```text
LeoJiangOR/vitra-gigahands-allcam0-keypoints-mano-cleaned
```

It contains converted VITRA annotations, train/test frame indices, cleanup reports,
manifests, and normalization statistics. The annotation splits are stored as:

- `gigahands_real_train_annotations.tar.zst`
- `gigahands_real_test_annotations.tar.zst`

The RGB videos are not stored in GitHub. The converted dataset expects the original
GigaHands RGB videos under:

```text
Video/GigaHands_root
```

For local use, link this path to the extracted GigaHands `multiview_rgb_vids` directory.

```bash
cd /path/to/VLA-HAND

huggingface-cli download \
  --repo-type dataset \
  LeoJiangOR/vitra-gigahands-allcam0-keypoints-mano-cleaned \
  --local-dir datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned_archives

mkdir -p datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned
tar --zstd -xf \
  datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned_archives/gigahands_real_train_annotations.tar.zst \
  -C datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned
tar --zstd -xf \
  datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned_archives/gigahands_real_test_annotations.tar.zst \
  -C datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned

mkdir -p datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned/Video
ln -s /path/to/multiview_rgb_vids \
  datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned/Video/GigaHands_root
```

Cleaned dataset size:

| Split | Samples | Episodes |
| --- | ---: | ---: |
| train | 7,043,667 | 28,368 |
| test | 368,585 | 1,495 |

The cleaned split removes frame-level samples whose annotation frame id points past the
actual decodable mp4 length.

### OpenTouch

The converted OpenTouch VITRA-format dataset for tactile action editing is released as:

```text
LeoJiangOR/opentouch-vitra-stage1-keypoint-full
```

The current release uses the text-aligned conversion. The converter matches every
OpenTouch clip to its annotation by `ts_start/ts_end`, uses the original `description`
as the VITRA instruction, and fails with `--require_labels` if any clip has no matching
language annotation.

Each cached editor sample stores:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `a_base` | `[16, 192]` | frozen VITRA action prediction |
| `a_target` | `[16, 192]` | OpenTouch converted future action target |
| `current_state` | `[212]` | current hand/action state |
| `touch_pressure` | `[16, 2, 16, 16]` | two-hand tactile pressure maps |
| `touch_mask` | `[16, 2]` | valid tactile hand mask |

The local text-aligned dataset path used by the current scripts is:

```text
datasets/vitra_opentouch_keypoint_full_text_aligned
```

Download:

```bash
cd /path/to/VLA-HAND
huggingface-cli download \
  --repo-type dataset \
  LeoJiangOR/opentouch-vitra-stage1-keypoint-full \
  --local-dir datasets/vitra_opentouch_keypoint_full_text_aligned
```

## GigaHands Evaluation

Run the cleaned GigaHands test evaluation:

```bash
GPU=0 \
CHECKPOINT=/path/to/epoch=0-step=140000.ckpt \
BASELINE_CHECKPOINT=/path/to/vitra-vla-3b.pt \
bash scripts/evaluate_gigahands_cleaned.sh
```

Default evaluation:

- dataset: `datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned`
- split: `gigahands_real_test`
- sample strategy: `middle_per_episode`
- clips: `1495`
- videos: disabled, metrics only

Output:

```text
runs/gigahands_cleaned_eval/<label>_vs_<baseline>_<clips>clips/metrics_comparison.json
```

Main metrics:

- `action_mse`: masked MSE over all valid action dimensions
- `left_action_mse`: masked MSE over valid left-hand dimensions
- `right_action_mse`: masked MSE over valid right-hand dimensions
- `dual_hand_action_mse`: masked MSE over frames where both hands are valid

Current cleaned-test result:

| Metric | Base VITRA | Step 140000 | Relative improvement |
| --- | ---: | ---: | ---: |
| `action_mse` | 16.1358 | 0.4061 | 97.48% |
| `left_action_mse` | 3.4089 | 0.4763 | 86.03% |
| `right_action_mse` | 45.2258 | 0.2456 | 99.46% |
| `dual_hand_action_mse` | 16.1358 | 0.4061 | 97.48% |

This is an in-distribution GigaHands train/test split result. It does not measure
open-world single-image inference quality, hand mesh rendering quality, future RGB
prediction, or overlay alignment.

## Distillation

The distillation code compresses VITRA into a smaller student:

- DINOv2-base vision encoder
- DistilBERT text encoder
- 6-layer `DiT-B-6L` action head

Launch scripts:

```bash
bash scripts/run_finetune_distill_all_cam0_full_tmux.sh
bash scripts/run_finetune_distill_step140000_joint_kd_full_tmux.sh
```

The two evaluated schemes are:

| Scheme | Teacher | Objective |
| --- | --- | --- |
| old student | original VITRA-3B | VLM feature distillation + GT diffusion action loss |
| joint-KD student | step-140000 finetuned VITRA | VLM feature distillation + GT action loss + action-head KD |

Cleaned GigaHands test result:

| Model | Action MSE | Left MSE | Right MSE | Dual-hand MSE |
| --- | ---: | ---: | ---: | ---: |
| Base VITRA-3B | 16.1358 | 3.4089 | 45.2258 | 16.1358 |
| Finetuned VITRA step140000 | 0.4061 | 0.4763 | 0.2456 | 0.4061 |
| Old distilled student | 0.3778 | 0.4301 | 0.2583 | 0.3778 |
| New joint-KD student | 0.4532 | 0.5367 | 0.2624 | 0.4532 |

Measured model-side inference speed:

| Model | VLA backend ms | Diffusion ms | Backend + diffusion ms | Total speedup |
| --- | ---: | ---: | ---: | ---: |
| Base VITRA-3B | 74.10 | 73.71 | 147.81 | 1.00x |
| Old distilled student | 8.49 | 44.60 | 53.09 | 2.78x |
| New joint-KD student | 8.47 | 44.52 | 52.99 | 2.79x |

Full report:

```text
doc/distillation_and_test_time_guidance_report.md
```

## OpenTouch Action Editing

The action editor is a learned residual model. VITRA first generates a 16-step action
chunk at time `0s`. At approximately `0.33s`, tactile observations are available, and
the editor changes only the unexecuted suffix of the chunk.

The editor is trained on cached frozen VITRA predictions:

```text
residual_target = a_target - a_base
a_edit = a_base + editable_suffix_mask * residual
```

It receives VITRA's base action, current state, action mask, chunk phase, and causal
tactile history up to the edit time. It is not allowed to modify the already executed
prefix.

Run the full OpenTouch editor pipeline:

```bash
GPU=7 \
DATASET_ROOT=$PWD/datasets/vitra_opentouch_keypoint_full_text_aligned \
bash scripts/run_online_touch_editor_large.sh
```

This script can also convert raw OpenTouch if `datasets/vitra_opentouch_keypoint_full_text_aligned`
does not already exist, but the released converted dataset is the intended sharing path.

To adapt the step-140000 GigaHands-finetuned VITRA checkpoint to OpenTouch first, then
train tactile editors on cached predictions from the adapted checkpoint, run:

```bash
GPU=7 \
SOURCE_CHECKPOINT=/path/to/epoch=0-step=140000.ckpt \
STAGE=large \
bash scripts/run_opentouch_step140000_adapt_touch_editor.sh
```

The launcher performs:

1. preflight checks for the text-aligned OpenTouch dataset and action/statistics files
2. VITRA OpenTouch adaptation initialized from the GigaHands `step140000` checkpoint
3. source-vs-adapted VITRA evaluation on OpenTouch test samples
4. cached train/test action generation for editor learning
5. tactile editor training and matched/shuffled/zero/future-touch evaluations

Use `STAGE=smoke` for a small end-to-end sanity check before launching the full tmux
run.

Current 10,000-sample OpenTouch test-cache result:

| Editor / Evaluation | Base MSE | Edited MSE | Improvement |
| --- | ---: | ---: | ---: |
| matched editor + matched touch | 0.7610 | 0.2080 | 72.67% |
| matched editor + shuffled touch | 0.7610 | 0.2099 | 72.42% |
| matched editor + zero touch | 0.7610 | 0.6217 | 18.31% |
| high-contact editor + matched touch | 0.7610 | 0.3170 | 58.34% |
| high-contact editor + shuffled touch | 0.7610 | 0.3176 | 58.26% |
| zero-touch editor + zero touch | 0.7610 | 0.2040 | 73.20% |

For these evaluations, `prefix_change_l2 = 0.0`, so the editor preserves the already
executed prefix.

Important caveat: matched touch and shuffled touch are almost identical. The editor
reduces action MSE, but the current result is weak evidence for sample-specific tactile
causality. It likely learns a strong residual prior from frozen VITRA predictions toward
OpenTouch targets.

## Diffusion Editing

### Polynomial Replanning

The polynomial replanning toy evaluates whether diffusion guidance can reduce explicit
region-constraint violation during action generation. The setup is:

- action chunk: `[16, 192]`
- guided dimensions: `[51, 52]`
- replanning points: `K=5` and `K=10`
- prefix clamping: actions before `K` are fixed
- DDIM steps: `10`
- CFG scale: `5.0`

Run:

```bash
python scripts/inference_guided_replanning_toy.py \
  --regions_json configs/polynomial_guidance_example.json \
  --guidance_scale 2.0 \
  --save_dir outputs/replanning_guidance/debug_seed0
```

Moderate guidance reduced violation in the fair same-prefix / same-noise ablation.
Very large guidance scales destabilized the trajectory. Prefix error stayed `0.0`.

Representative result:

| Setting | Unguided violation | Guided violation | Runtime slowdown |
| --- | ---: | ---: | ---: |
| K=5, scale=2 | 0.543184 | 0.456102 | about 3.0x |
| K=10, scale=2 | 0.823132 | 0.772271 | about 2.5x |

### Tactile DPS-Style Editing

The tactile DPS experiment trains a tactile encoder and an action-conditioned tactile
forward model on OpenTouch replay cache. The learned mapping is:

```text
(current_state, candidate_action_segment, chunk_phase) -> tactile representation
```

At test time, the tactile loss is used as a diffusion guidance term during
prefix-clamped DDIM replanning. This is different from the residual action editor: DPS
does not directly learn `tactile -> action residual`; it learns `action -> tactile`, then
uses gradients to regenerate the unexecuted suffix.

Smoke test:

```bash
GPU=0 bash scripts/run_tactile_dps_smoke.sh
```

Full run:

```bash
GPU=0 bash scripts/run_tactile_dps_full_tmux.sh
```

For diffusion-time tactile replanning, the replay cache must include cached VITRA
`action_features`. Regenerate the cache with the current
`scripts/cache_touch_editor_base_actions.py`, then run:

```bash
python scripts/evaluate_tactile_dps_diffusion_replanning.py \
  --cache_root runs/online_touch_editor_large/cache_test \
  --measurement_checkpoint runs/tactile_dps_opentouch_encoder_v1/best.pt \
  --vla_config vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json \
  --vla_checkpoint checkpoints/vitra-vla-3b.pt \
  --output_path runs/tactile_dps_diffusion_replanning/eval.json \
  --max_samples 256
```

Measurement model result:

| Metric | Value |
| --- | ---: |
| encoder stats MSE | 0.0000963 |
| forward embedding MSE | 0.0001434 |
| forward stats MSE | 0.0005048 |

Replay editing result is currently very small:

| Split | Base MSE | Guided MSE |
| --- | ---: | ---: |
| all samples | 0.7176356267 | 0.7176356078 |
| high-contact samples | 0.7513158663 | 0.7513158556 |

This means the measurement model can fit tactile summaries, but the earlier action-space
replay edit did not yet produce a meaningful action-space improvement. The diffusion
replanning script is the intended DPS path going forward.

Diffusion-only guided regeneration latency:

| Model | K=5 guided regenerate | K=10 guided regenerate |
| --- | ---: | ---: |
| Base VITRA | 484 ms | 357 ms |
| joint-KD student | 310 ms | 245 ms |

## Useful Files

- `tools/prepare_gigahands_real_subset.py`: convert GigaHands to VITRA stage-1 format
- `tools/clean_gigahands_linked_dataset.py`: remove invalid video-frame samples
- `tools/evaluate_gigahands_stage1.py`: evaluate base vs finetuned checkpoints
- `scripts/evaluate_gigahands_cleaned.sh`: reproducible cleaned-test evaluation
- `scripts/run_online_touch_editor_large.sh`: OpenTouch residual action-editor pipeline
- `scripts/run_opentouch_step140000_adapt_touch_editor.sh`: OpenTouch adaptation plus tactile-editor pipeline from the GigaHands step-140000 checkpoint
- `scripts/train_touch_editor.py`: train the learned residual editor
- `scripts/evaluate_touch_guided_actions.py`: evaluate editor ablations
- `scripts/inference_guided_replanning_toy.py`: polynomial guided replanning toy
- `scripts/evaluate_tactile_dps_diffusion_replanning.py`: tactile DPS diffusion replanning evaluation
- `scripts/run_tactile_dps_smoke.sh`: tactile DPS smoke test
- `scripts/run_tactile_dps_full_tmux.sh`: tactile DPS full tmux launcher
- `doc/distillation_and_test_time_guidance_report.md`: distillation and guidance report
- `doc/test-time-guidance-dps/dps_style_tactile_guided_action_diffusion.md`: tactile DPS design note
