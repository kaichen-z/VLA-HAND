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
