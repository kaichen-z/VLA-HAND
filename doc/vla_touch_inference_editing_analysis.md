# kaichen-z/VLA-HAND Touch Editing Analysis

## Repository Snapshot

- Original analysis clone: `vla_touch`
- Remote: `https://github.com/kaichen-z/VLA-HAND`
- Branch/commit inspected: `main` at `c49dc8ea1ca5668fca6dd06d46fe5e28991bfeca`
- Clone size: about `20M`

This fork is not fully self-contained at the inspected commit. Several scripts and tests reference files that are not present in the Git tree:

- missing `scripts/cache_touch_editor_base_actions.py`
- missing `vitra/touch_editor/cache_utils.py`

Because of those missing files, some documented cache-building and pseudo-pair workflows cannot run from this commit without recovery from another branch or commit.

Local recovery status:

- Recovered the two missing files and migrated them into this repository.
- Verified `tests/test_touch_editor.py`: `29 passed`.
- Verified `tests/test_opentouch_stage1_converter.py`: `9 passed`.
- Note: the upstream `.gitignore` contains `*cache*`, so the recovered files are ignored by default unless force-added or the ignore rule is changed.

## High-Level Method

The inference-time editing method is a learned residual action editor, not diffusion-time gradient guidance.

The intended runtime flow is:

```text
image + instruction + current hand state
  -> frozen VITRA
  -> A_base: [B, 16, 192] normalized action chunk

A_base + current state + touch history + action/future masks
  -> ResidualTouchEditor
  -> DeltaA: [B, 16, 192]

A_edit = A_base + future_mask * DeltaA
```

The editor changes only the unexecuted future suffix. It does not regenerate the action chunk through VITRA diffusion, and it does not backpropagate through the diffusion model at inference time.

## Core Files

### `vitra/touch_editor/model.py`

Defines `ResidualTouchEditor`.

Important inputs:

- `a_base`: frozen VITRA action, `[B, T, 192]`
- `current_state`: current hand state, `[B, 212]`
- `current_state_mask`: valid state mask, `[B, 212]`
- `touch_pressure`: tactile pressure, `[B, H, 2, 16, 16]`
- `touch_mask`: valid tactile hand mask, `[B, H, 2]`
- `chunk_phase`: normalized timestep phase
- `future_mask`: editable suffix mask, `[B, T, 192]`
- `action_mask`: valid action dimensions, `[B, T, 192]`

Architecture:

- tactile encoder: small Conv2D stack over the two-hand `16 x 16` pressure maps
- feature concat per timestep:
  - raw base action
  - masked base action
  - action mask
  - future mask
  - current state
  - current state mask
  - touch feature
  - touch-valid flag
  - phase
- temporal model: Transformer encoder
- output: residual action delta `[B, T, 192]`

### `vitra/touch_editor/guidance.py`

Implements inference-time editing.

Key functions:

- `seconds_to_chunk_index(seconds, fps, chunk_len)`
- `apply_touch_editor_once(...)`
- `apply_touch_guidance_schedule(...)`

At `fps=8`, the documented default edit times map to:

```text
0.33s -> round(0.33 * 8) = 3
0.66s -> round(0.66 * 8) = 5
```

The schedule is sequential:

```text
A_0 = A_base
A_1 = edit(A_0, start_idx=3)
A_2 = edit(A_1, start_idx=5)
```

Each edit builds `future_mask[:, edit_start_idx:]`, so earlier timesteps stay exactly equal to the input action for that edit.

### `vitra/touch_editor/losses.py`

Training loss:

```text
A_edit = A_base + future_mask * action_mask * DeltaA
residual_target = A_target - A_base

L_total =
  L_residual
  + lambda_dev * L_dev
  + lambda_delta * L_delta
  + lambda_smooth * L_smooth
  + lambda_mask * L_mask
```

Where:

- `L_residual`: masked MSE between `DeltaA` and `residual_target`
- `L_demo`: tracked but not directly included in total
- `L_dev`: penalizes deviation from `A_base`
- `L_delta`: regularizes residual magnitude
- `L_smooth`: regularizes temporal residual changes
- `L_mask`: penalizes nonzero residuals outside valid action dimensions

Script defaults in `scripts/train_touch_editor.py`:

```text
lambda_dev = 0.1
lambda_delta = 0.01
lambda_smooth = 0.05
lambda_mask = 1.0
```

### `vitra/touch_editor/dataset.py`

Loads cached `.npz` samples.

Required cache fields:

```text
a_base: [T, 192]
a_target: [T, 192]
residual_target: [T, 192]
action_mask: [T, 192]
current_state: [212]
current_state_mask: [212]
touch_pressure: [H, 2, 16, 16]
touch_mask: [H, 2]
future_mask: [T, 192]
```

Optional metadata includes `chunk_phase`, `edit_start_idx`, action/touch timestamps, and alignment validity.

### `scripts/train_touch_editor.py`

Trains the residual editor from cached frozen-VLA samples.

Verified:

```bash
python scripts/train_touch_editor.py --help
```

works at the inspected commit.

### `scripts/evaluate_touch_guided_actions.py`

Evaluates whether the trained editor improves action MSE relative to the cached target.

It reports, for each edit step:

- base MSE
- edited MSE
- improvement percentage
- left/right MSE breakdown
- delta norm
- smoothness
- valid editable dimensions

It supports ablations:

- `matched`
- `zero_touch`
- `shuffled_touch`
- `random_pair`
- `no_touch`

Verified:

```bash
python scripts/evaluate_touch_guided_actions.py --help
```

works at the inspected commit.

### `scripts/inference_human_prediction.py`

Contains the runtime hook for one-shot inference.

If `--touch_editor_checkpoint` and `--touch_data_path` are passed:

1. VITRA predicts `norm_action`.
2. Touch payload is loaded from `.npz`.
3. `apply_touch_guidance_schedule(...)` edits normalized actions.
4. Edited normalized action is denormalized and visualized.

Expected touch payload:

```text
touch_pressure: [H, 2, 16, 16]
touch_mask: [H, 2]
```

## OpenTouch Role

OpenTouch provides tactile pressure signals and right-hand landmark supervision.

The converter in `data/preprocessing/convert_opentouch_to_vitra_stage1.py` does the following:

- reads OpenTouch HDF5 clips
- extracts RGB JPEG frames
- extracts right-hand landmarks; left hand is optional/dummy
- extracts left/right tactile pressure maps
- aligns tactile timestamps to video frame timestamps
- writes VITRA-format episodes under `datasets/vitra_opentouch_keypoint`

The tactile payload stored per episode includes:

```text
touch_pressure: [T, 2, 16, 16]
touch_mask: [T, 2]
video_timestamps
touch_timestamps
touch_aligned_indices
touch_aligned_timestamps
touch_alignment_valid
```

The OpenTouch stage-1 action target is keypoint-based and mainly right-hand supervised. The README also describes a pseudo-pair route where OpenTouch supplies tactile input, while the action target is matched from official GigaHands examples.

Local OpenTouch data status:

- Raw path: `datasets/opentouch_raw`
- Downloaded raw files: `26` HDF5 files plus `final_annotation.zip`
- Raw size: about `20G`
- Extracted annotations: `25` CSV files under `datasets/opentouch_raw/final_annotations`
- Small converted subset: `datasets/vitra_opentouch_keypoint`
- Converted subset result: `20` episodes, `1869` frames, `0` conversion errors
- Statistics written to `datasets/vitra_opentouch_keypoint/Annotation/statistics/opentouch_keypoint_train_keypoints_statistics.json`

The small converted subset is only a pipeline check. It is not yet the full OpenTouch training set.

## Method Critique

The core concern is the training target. The editor is trained to predict a residual:

```text
residual_target = A_target - A_base
```

For direct OpenTouch conversion, `A_target` comes from OpenTouch right-hand landmarks converted into VITRA keypoint-style action targets. This gives tactile-contact supervision, but it is mostly right-hand supervised and is not the same as a real online corrective-action label.

For pseudo-paired data, OpenTouch supplies touch observations while the target action can come from matched GigaHands clips. That gives a way to train an editor without real paired "wrong action -> tactile feedback -> corrected action" demonstrations, but the matching is heuristic and may teach dataset correlation instead of true tactile correction.

The strongest way to frame this method is:

- It is a learned residual post-processor over VITRA actions.
- It can be useful if tactile signals reliably predict future hand/action corrections.
- It is weaker than true online editing unless the train/test setup contains real paired tactile failure/correction data.
- Evaluation should include ablations such as matched touch, zero touch, shuffled touch, random-pair touch, and no-touch. If matched touch is not clearly better than shuffled/random touch, the editor is probably exploiting action priors rather than touch information.

## Repro Commands

```bash
cd /path/to/VLA-HAND

/scratch/chonghej/conda_envs/vitra/bin/python -m pytest tests/test_touch_editor.py -q
/scratch/chonghej/conda_envs/vitra/bin/python -m pytest tests/test_opentouch_stage1_converter.py -q

STAGE=verify_raw bash scripts/run_opentouch_stage1_subset_pipeline.sh
MAX_FILES=1 MAX_CLIPS=20 STAGE=convert bash scripts/run_opentouch_stage1_subset_pipeline.sh

PYTHONPATH=$PWD \
  /scratch/chonghej/conda_envs/vitra/bin/python vitra/datasets/calculate_statistics.py \
  --dataset_folder datasets/vitra_opentouch_keypoint \
  --dataset_name opentouch_keypoint_train \
  --action_type keypoints \
  --num_workers 0 \
  --batch_size 16 \
  --save_folder datasets/vitra_opentouch_keypoint/Annotation/statistics
```

## Comparison With Our Diffusion Guidance

### Their method

```text
Type: learned residual editor
When applied: after VITRA produces A_base
Conditioning: touch pressure + current state + base action + masks
Computation: one small editor forward pass
Output: A_edit = A_base + masked DeltaA
Training required: yes
Inference backward through VITRA: no
```

Strengths:

- likely much faster than guided diffusion
- naturally latency-aware because it edits only the future suffix
- can use real tactile signal
- does not require rerunning diffusion or backpropagating through DiT

Weaknesses:

- needs trained editor and cache data
- depends on OpenTouch-style tactile input at inference time
- quality depends on whether `A_target` is meaningful
- current repo is incomplete for cache generation

### Our current method

```text
Type: explicit test-time diffusion guidance
When applied: inside DDIM sampling / regeneration
Conditioning: polynomial constraint in action space
Computation: DiT forward + autograd guidance per DDIM step
Output: regenerated action chunk or suffix
Training required: no
Inference backward through VITRA: yes
```

Strengths:

- no extra editor training needed
- can impose arbitrary differentiable constraints
- edits diffusion generation directly

Weaknesses:

- slower because guidance requires autograd through DiT
- harder to meet real-time next-action latency
- toy polynomial constraint is not grounded in real tactile feedback

## Integration Implications

The most promising integration direction is not to replace our guidance with her method directly, but to treat her editor as a fast learned suffix controller:

```text
VITRA / finetuned VITRA -> A_base
fast touch/editor module -> A_edit for future suffix
optional diffusion guidance -> only for slower high-risk correction
```

If we want to reuse her approach, the first engineering task should be reconstructing the missing cache utilities:

- `build_future_mask(action_mask, edit_start_idx)`
- `chunk_phase(chunk_len)`
- `cache_touch_editor_base_actions.py`
  - load frozen VITRA
  - run `predict_action`
  - extract target/action/touch windows
  - write `.npz` cache records

After that, a minimal smoke test can use synthetic cache records to verify training/eval without downloading OpenTouch.

## Practical Recommendation

For reporting:

- Describe her method as a learned OpenTouch-conditioned residual action editor.
- Be explicit that it edits normalized VITRA action chunks after generation, not the diffusion denoising trajectory.
- Mention that main currently appears incomplete because cache-building utilities referenced by README/tests are missing.

For future work:

1. recover missing cache files from her local branch or earlier commit
2. run synthetic-cache smoke training
3. train/evaluate editor on OpenTouch or pseudo-paired OpenTouch-GigaHands cache
4. benchmark latency against our diffusion-only guided regenerate
