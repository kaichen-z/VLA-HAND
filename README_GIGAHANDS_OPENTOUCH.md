# GigaHands and OpenTouch Pretraining Workflows for VITRA

This fork adds two Stage-1 human-hand pretraining paths on top of VITRA:

- **GigaHands real subset**: MANO-style left/right hand supervision plus RGB videos.
- **OpenTouch keypoint subset**: RGB plus right-hand landmarks from OpenTouch HDF5 files; the right hand is supervised and the left hand is dummy/masked.

The repository does not include raw data, converted datasets, checkpoints, or demo videos. Keep all downloaded data under `datasets/` or set the environment variables shown below.

## Installation

```bash
git clone <your-vitra-fork-url>
cd VITRA
conda create -n vitra python=3.10 -y
conda activate vitra
pip install -e .
```

Before training the 3B base model, make sure your Hugging Face account has access to `google/paligemma2-3b-mix-224`.

The scripts use repo-relative defaults:

```text
datasets/
  gigahands_real/
  vitra_gigahands_real_subset/
  opentouch_raw/
  vitra_opentouch_keypoint/
runs/
```

You can override locations with `DATA_ROOT`, `GIGAHANDS_ROOT`, `OPENTOUCH_ROOT`, `OUTPUT_ROOT`, `RUN_ROOT`, `GPUS`, and `NPROC`.

## GigaHands Real Subset

### 1. Download metadata and hand poses

```bash
STAGE=download_metadata bash scripts/run_gigahands_real_subset_pipeline.sh
STAGE=download_hand_poses bash scripts/run_gigahands_real_subset_pipeline.sh
```

Expected layout:

```text
datasets/gigahands_real/
  annotations_v2.jsonl
  multiview_camera_video_map.csv
  hand_poses.tar.gz
  hand_poses/ or extracted scene folders
```

### 2. Build a small subset manifest

```bash
NUM_TRAIN=20 NUM_TEST=5 STAGE=prepare bash scripts/run_gigahands_real_subset_pipeline.sh
```

This writes:

```text
datasets/gigahands_real/subset_manifest.json
datasets/gigahands_real/needed_videos.txt
datasets/gigahands_real/needed_videos_unique.txt
```

`subset_manifest.json` controls the selected clips. `needed_videos_unique.txt` lists the exact RGB mp4 files needed for conversion.

### 3. Download or place the required RGB videos

Download the official GigaHands RGB videos and place the extracted `multiview_rgb_vids/` directory under `datasets/gigahands_real/`.

Expected layout:

```text
datasets/gigahands_real/
  multiview_rgb_vids/
    <session>/
      <camera>/
        <camera>_<timestamp>.mp4
```

For subset experiments, the exact files required by the current manifest are listed in `needed_videos_unique.txt`. The repository does not store RGB videos; it only verifies that the required files are present before conversion.

Verify all RGB videos are present:

```bash
STAGE=verify_videos bash scripts/run_gigahands_real_subset_pipeline.sh
```

### 4. Convert, compute statistics, and train

```bash
STAGE=convert bash scripts/run_gigahands_real_subset_pipeline.sh
STAGE=stats bash scripts/run_gigahands_real_subset_pipeline.sh
STAGE=train_smoke bash scripts/run_gigahands_real_subset_pipeline.sh
```

For a longer small run:

```bash
MAX_STEPS=500 STAGE=train bash scripts/run_gigahands_real_subset_pipeline.sh
```

Converted output:

```text
datasets/vitra_gigahands_real_subset/
  Annotation/gigahands_real_train/
  Annotation/gigahands_real_test/
  Annotation/statistics/
  Video/GigaHands_root/
```

### 5. Export demo/evaluation videos

Before training:

```bash
STAGE=eval_before bash scripts/run_gigahands_real_subset_pipeline.sh
```

After training:

```bash
CHECKPOINT=runs/gigahands_real_subset_train/checkpoints/<checkpoint-dir-or-weights.pt> \
STAGE=eval_after bash scripts/run_gigahands_real_subset_pipeline.sh
```

Outputs:

```text
runs/gigahands_real_subset_eval/
  before/metrics.json
  before/videos/*.mp4
  after/metrics.json
  after/videos/*.mp4
```

The demo videos compare target action traces against predicted action traces.

## OpenTouch Keypoint Subset

### 1. Download OpenTouch HDF5 files

OpenTouch is hosted on Google Drive and may hit quota limits. The script tries official file IDs and skips unavailable files.

Download the first available file:

```bash
STAGE=download_one bash scripts/run_opentouch_stage1_subset_pipeline.sh
```

Or try all official files:

```bash
STAGE=download_all bash scripts/run_opentouch_stage1_subset_pipeline.sh
```

You can also manually place `.h5` or `.hdf5` files under:

```text
datasets/opentouch_raw/
```

Verify raw HDF5 files:

```bash
STAGE=verify_raw bash scripts/run_opentouch_stage1_subset_pipeline.sh
```

### 2. Convert, compute statistics, and train

```bash
STAGE=convert bash scripts/run_opentouch_stage1_subset_pipeline.sh
STAGE=stats bash scripts/run_opentouch_stage1_subset_pipeline.sh
STAGE=train_smoke bash scripts/run_opentouch_stage1_subset_pipeline.sh
```

To build the contact-rich subset for touch-editor work, pass OpenTouch labels and enable keyword filtering:

```bash
LABELS_PATH=datasets/opentouch_raw/final_annotations/<labels-file>.csv \
FILTER_CONTACT_KEYWORDS=1 \
STAGE=convert bash scripts/run_opentouch_stage1_subset_pipeline.sh
```

If OpenTouch tactile timestamps are separate from RGB/state timestamps, the converter aligns each video frame to the nearest tactile frame. Override the default half-frame tolerance when needed:

```bash
TOUCH_ALIGNMENT_TOLERANCE=0.02 \
STAGE=convert bash scripts/run_opentouch_stage1_subset_pipeline.sh
```

The default retained verbs are:

```text
grasp grip hold press push pull lift place insert remove open close
```

The converter writes `opentouch_contact_subset_manifest.jsonl` and preserves tactile tensors under each episode's `opentouch` payload:

```text
touch_pressure: T x 2 x 16 x 16
touch_mask:     T x 2
```

The payload also records timestamp alignment metadata:

```text
video_timestamps
touch_timestamps
touch_aligned_indices
touch_aligned_timestamps
touch_alignment_valid
```

Cache frozen VLA-Hand base actions on the converted OpenTouch subset:

```bash
python scripts/cache_touch_editor_base_actions.py \
  --checkpoint LeoJiangOR/vitra-gigahands-keypoints-step140000 \
  --dataset_root datasets/vitra_opentouch_keypoint \
  --data_mix opentouch_keypoint_train \
  --cache_root runs/touch_editor_cache \
  --random_edit_start
```

The cache script uses the same future action window as VITRA and stores aligned touch windows plus action/touch timestamp metadata. The residual editor training entrypoint expects cached frozen-VLA samples containing `A_base`, VITRA-format `A_target`, action masks, current state, future edit masks, and touch windows:

```bash
python scripts/train_touch_editor.py \
  --cache_root runs/touch_editor_cache \
  --output_dir runs/touch_editor
```

Evaluate touch guidance at real second offsets inside each cached VITRA chunk:

```bash
python scripts/evaluate_touch_guided_actions.py \
  --cache_root runs/touch_editor_cache \
  --touch_editor_checkpoint runs/touch_editor/latest.pt \
  --output_path runs/touch_editor_eval/metrics.json \
  --fps 8 \
  --edit_times 0.33 0.66
```

This reports masked normalized-action MSE for the frozen base action and for the sequential touch-edited actions. The edit indices are computed from actual FPS, so at `fps=8` the default edits land at chunk indices `3` and `5`.

Evaluate frozen VITRA against official GigaHands/VITRA ground-truth actions separately:

```bash
python scripts/evaluate_vitra_gt_actions.py \
  --checkpoint LeoJiangOR/vitra-gigahands-keypoints-step140000 \
  --dataset_root datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked \
  --data_mix gigahands_real_test \
  --statistics_dataset_name gigahands_real_train \
  --output_path runs/vitra_gt_eval_gigahands_test/metrics.json
```

This target is the normalized GigaHands/VITRA `action_list`, not OpenTouch landmarks. GigaHands does not include tactile pressure, so this baseline should be reported separately from OpenTouch touch-editor metrics.

Two-stage residual editor training:

```bash
# Stage 1: learn the VITRA residual prior from official GigaHands targets.
python scripts/cache_touch_editor_base_actions.py \
  --checkpoint LeoJiangOR/vitra-gigahands-keypoints-step140000 \
  --dataset_root datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked \
  --data_mix gigahands_real_train \
  --statistics_dataset_name gigahands_real_train \
  --cache_root runs/touch_editor_cache_gigahands_train \
  --touch_mode zeros \
  --random_edit_start

python scripts/train_touch_editor.py \
  --cache_root runs/touch_editor_cache_gigahands_train \
  --output_dir runs/touch_editor_stage1_gigahands

# Stage 2 option A: fine-tune with real OpenTouch pressure and OpenTouch-derived targets.
python scripts/train_touch_editor.py \
  --cache_root runs/touch_editor_cache \
  --output_dir runs/touch_editor_stage2_opentouch \
  --init_checkpoint runs/touch_editor_stage1_gigahands/latest.pt
```

If the action supervision should always be official GigaHands GT, build pseudo-paired
OpenTouch-to-GigaHands samples instead. In this cache, OpenTouch supplies only the
tactile input and the target is a matched official GigaHands action chunk:

```bash
python scripts/build_opentouch_gigahands_pseudo_pairs.py \
  --checkpoint LeoJiangOR/vitra-gigahands-keypoints-step140000 \
  --opentouch_dataset_root datasets/vitra_opentouch_keypoint \
  --opentouch_data_mix opentouch_keypoint_train \
  --gigahands_dataset_root datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked \
  --gigahands_data_mix gigahands_real_train \
  --gigahands_statistics_dataset_name gigahands_real_train \
  --cache_root runs/touch_editor_cache_opentouch_gigahands_matched \
  --candidate_pool_size 4096 \
  --random_edit_start

python scripts/train_touch_editor.py \
  --cache_root runs/touch_editor_cache_opentouch_gigahands_matched \
  --output_dir runs/touch_editor_stage2_opentouch_gigahands_matched \
  --init_checkpoint runs/touch_editor_stage1_gigahands/latest.pt
```

The pseudo-paired cache writes `target_source=gigahands_matched`, keeps
`touch_source=opentouch`, and sets `residual_target = A_gigahands_gt - A_base`.
Matching uses contact verb overlap when available, then normalized trajectory phase
and current-state/keypoint distance. The converted OpenTouch action target is not
used as Stage 2 supervision in this path.

The cache stores both `a_target` and `residual_target = a_target - a_base`. Training minimizes the explicit residual loss on the editable future suffix:

```text
L_total = L_residual + 0.01 L_delta + 0.05 L_smooth + 1.0 L_mask
```

For one-shot inference visualization, provide a saved tactile `.npz` payload with:

```text
touch_pressure: H x 2 x 16 x 16
touch_mask:     H x 2
```

Then pass the trained editor and touch payload to the inference script:

```bash
python scripts/inference_human_prediction.py \
  --config_path vitra/configs/human_pretrain_gigahands_real_full_keypoints_vitra3b_linked.json \
  --model_path <vitra-weights.pt-or-checkpoint-dir> \
  --image_path <input-image> \
  --use_right \
  --touch_editor_checkpoint runs/touch_editor/latest.pt \
  --touch_data_path <touch-window.npz> \
  --touch_edit_times 0.33 0.66 \
  --fps 8
```

The inference hook edits normalized VITRA actions before denormalization:

```text
A_edit = A_base + future_mask * action_mask * DeltaA
```

Converted output:

```text
datasets/vitra_opentouch_keypoint/
  Annotation/opentouch_keypoint_train/
  Annotation/opentouch_keypoint_test/
  Annotation/statistics/
  Video/OpenTouch_root/
```

The OpenTouch path uses:

```text
action_type=keypoints
right hand supervised
left hand dummy/masked
```

### 3. Export demo/evaluation videos

```bash
STAGE=eval_before bash scripts/run_opentouch_stage1_subset_pipeline.sh

CHECKPOINT=runs/opentouch_keypoint_subset/checkpoints/<checkpoint-dir-or-weights.pt> \
STAGE=eval_after bash scripts/run_opentouch_stage1_subset_pipeline.sh
```

Outputs are written under:

```text
runs/opentouch_keypoint_subset_eval/
```

## Common Issues

- **Missing GigaHands videos**: run `STAGE=verify_videos`; download or place the missing paths under `datasets/gigahands_real/multiview_rgb_vids/`.
- **OpenTouch Google Drive quota**: rerun `download_one`, set `OPENTOUCH_GDRIVE_ID=<id>`, or manually download `.hdf5` files into `datasets/opentouch_raw/`.
- **Missing statistics**: run `STAGE=stats` before training with normalization enabled.
- **PaliGemma access errors**: request access to `google/paligemma2-3b-mix-224` and authenticate with Hugging Face.
- **GPU memory**: reduce `NPROC`, `batch_size`, or `MAX_STEPS` for smoke tests.

## Quick Smoke Commands

GigaHands after RGB files are present:

```bash
STAGE=verify_videos bash scripts/run_gigahands_real_subset_pipeline.sh
STAGE=convert bash scripts/run_gigahands_real_subset_pipeline.sh
STAGE=stats bash scripts/run_gigahands_real_subset_pipeline.sh
STAGE=train_smoke bash scripts/run_gigahands_real_subset_pipeline.sh
```

OpenTouch after at least one HDF5 file is present:

```bash
STAGE=verify_raw bash scripts/run_opentouch_stage1_subset_pipeline.sh
STAGE=convert bash scripts/run_opentouch_stage1_subset_pipeline.sh
STAGE=stats bash scripts/run_opentouch_stage1_subset_pipeline.sh
STAGE=train_smoke bash scripts/run_opentouch_stage1_subset_pipeline.sh
```
