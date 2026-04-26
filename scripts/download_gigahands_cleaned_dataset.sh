#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HF_REPO="${HF_REPO:-LeoJiangOR/vitra-gigahands-allcam0-keypoints-mano-cleaned}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${REPO_ROOT}/datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned_archives}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned}"
GIGAHANDS_VIDEO_ROOT="${GIGAHANDS_VIDEO_ROOT:-}"

mkdir -p "${ARCHIVE_DIR}" "${DATASET_ROOT}"

huggingface-cli download \
  --repo-type dataset \
  "${HF_REPO}" \
  --local-dir "${ARCHIVE_DIR}"

tar --zstd -xf "${ARCHIVE_DIR}/gigahands_real_train_annotations.tar.zst" -C "${DATASET_ROOT}"
tar --zstd -xf "${ARCHIVE_DIR}/gigahands_real_test_annotations.tar.zst" -C "${DATASET_ROOT}"

if [[ -n "${GIGAHANDS_VIDEO_ROOT}" ]]; then
  mkdir -p "${DATASET_ROOT}/Video"
  ln -sfn "${GIGAHANDS_VIDEO_ROOT}" "${DATASET_ROOT}/Video/GigaHands_root"
  echo "Linked videos: ${DATASET_ROOT}/Video/GigaHands_root -> ${GIGAHANDS_VIDEO_ROOT}"
else
  echo "Set GIGAHANDS_VIDEO_ROOT=/path/to/multiview_rgb_vids to link RGB videos."
fi

echo "Dataset annotations are ready at ${DATASET_ROOT}"
