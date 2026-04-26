# VLA-HAND GigaHands Fine-Tuning

This repository contains the code used to fine-tune VITRA-VLA on converted GigaHands egocentric hand-action data and evaluate the resulting checkpoint against the base VITRA model.

The current GigaHands experiment is a VITRA stage-1 future hand-action prediction setup:

- input: current egocentric RGB frame, language instruction, current hand state, hand-state mask, and FOV
- output: a 16-step future hand-action chunk
- target representation: `keypoints_3d_mano` converted into VITRA `action_type=keypoints`
- metric: masked action-space MSE, not image/video reconstruction error

## Released Data

The cleaned converted dataset is released as a Hugging Face Dataset:

`LeoJiangOR/vitra-gigahands-allcam0-keypoints-mano-cleaned`

It contains the converted VITRA annotations, train/test frame indices, cleanup reports, manifests, and normalization statistics. To avoid pushing tens of thousands of small files, the annotation splits are released as two archives:

- `gigahands_real_train_annotations.tar.zst`
- `gigahands_real_test_annotations.tar.zst`

The RGB videos are not stored in GitHub. The released dataset expects the original GigaHands RGB videos to be available under:

`Video/GigaHands_root`

For local use, this can be a symlink to the extracted GigaHands `multiview_rgb_vids` directory.

## Download The Cleaned Dataset

Install the Hugging Face CLI if needed, then download and extract the converted annotations:

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
```

Then link the original GigaHands RGB videos:

```bash
mkdir -p datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned/Video
ln -s /path/to/multiview_rgb_vids \
  datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned/Video/GigaHands_root
```

The cleaned split removes frame-level samples whose annotation frame id points past the actual decodable mp4 length.

Cleaned dataset size:

- train samples: `7,043,667`
- test samples: `368,585`
- train episodes: `28,368`
- test episodes: `1,495`

## Evaluation

Use the cleaned evaluation script:

```bash
GPU=0 \
CHECKPOINT=/path/to/epoch=0-step=140000.ckpt \
BASELINE_CHECKPOINT=/path/to/vitra-vla-3b.pt \
bash scripts/evaluate_gigahands_cleaned.sh
```

By default this runs:

- dataset: `datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned`
- split: `gigahands_real_test`
- sample strategy: `middle_per_episode`
- clips: `1495`
- videos: disabled, metrics only

The output file is:

`runs/gigahands_cleaned_eval/<label>_vs_<baseline>_<clips>clips/metrics_comparison.json`

The main metrics are:

- `action_mse`: masked MSE over all valid hand-action dimensions
- `left_action_mse`: masked MSE over valid left-hand dimensions
- `right_action_mse`: masked MSE over valid right-hand dimensions
- `dual_hand_action_mse`: masked MSE over frames where both hands are valid

These metrics compare predicted future `keypoints_3d_mano` action chunks against the converted GigaHands ground truth in the same action space.

## Current Cleaned-Test Result

Using the step-140000 checkpoint on the cleaned full GigaHands test split:

| Metric | Base VITRA | Step 140000 | Relative improvement |
| --- | ---: | ---: | ---: |
| `action_mse` | 16.1358 | 0.4061 | 97.48% |
| `left_action_mse` | 3.4089 | 0.4763 | 86.03% |
| `right_action_mse` | 45.2258 | 0.2456 | 99.46% |
| `dual_hand_action_mse` | 16.1358 | 0.4061 | 97.48% |

This result measures in-distribution GigaHands future hand-action prediction. It does not measure open-world single-image inference quality, hand reconstruction quality, RGB future-frame prediction, or visual overlay alignment.

## Useful Scripts

- `tools/prepare_gigahands_real_subset.py`: convert GigaHands annotations/videos into VITRA stage-1 format
- `tools/clean_gigahands_linked_dataset.py`: remove samples with invalid video frame indices
- `tools/evaluate_gigahands_stage1.py`: evaluate base vs finetuned checkpoints
- `scripts/download_gigahands_cleaned_dataset.sh`: download and extract the released cleaned annotations
- `scripts/evaluate_gigahands_cleaned.sh`: reproducible cleaned-test evaluation entrypoint
- `scripts/run_gigahands_real_all_cam0_keypoints_mano_pipeline.sh`: all-cam0 keypoints-MANO data preparation pipeline

## Notes

The model predicts future hand actions, not future video frames. In the current GigaHands setup, both prediction and ground truth are represented as `keypoints_3d_mano` action chunks.
