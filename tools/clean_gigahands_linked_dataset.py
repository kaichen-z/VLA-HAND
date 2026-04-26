"""Clean converted GigaHands linked datasets by filtering undecodable frames.

The converted VITRA dataset stores per-sample frame ids in
Annotation/<split>/episode_frame_index.npz. Some released RGB videos are
shorter than the annotation frame ids. This script creates a cleaned linked
dataset root where frame-index rows whose RGB frame cannot exist are removed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


def _video_len(video_path: str) -> tuple[str, int | None, str | None]:
    try:
        from decord import VideoReader

        reader = VideoReader(video_path)
        return video_path, len(reader), None
    except Exception as exc:  # pragma: no cover - depends on local video files
        return video_path, None, repr(exc)


def _link_or_copy(src: Path, dst: Path, *, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if src.is_symlink():
        target = os.readlink(src)
        os.symlink(target, dst)
        return
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _copy_top_level(src_root: Path, dst_root: Path, *, hardlink: bool) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)

    for item in src_root.iterdir():
        if item.name in {"Annotation", "Video"}:
            continue
        dst = dst_root / item.name
        if item.is_dir():
            if not dst.exists():
                shutil.copytree(item, dst, copy_function=os.link if hardlink else shutil.copy2)
        else:
            _link_or_copy(item, dst, hardlink=hardlink)

    src_video = src_root / "Video" / "GigaHands_root"
    dst_video = dst_root / "Video" / "GigaHands_root"
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    if not dst_video.exists() and not dst_video.is_symlink():
        if src_video.is_symlink():
            os.symlink(os.readlink(src_video), dst_video)
        else:
            os.symlink(src_video, dst_video)

    src_stats = src_root / "Annotation" / "statistics"
    dst_stats = dst_root / "Annotation" / "statistics"
    if src_stats.exists() and not dst_stats.exists():
        shutil.copytree(src_stats, dst_stats, copy_function=os.link if hardlink else shutil.copy2)


def _load_episode(split_dir: Path, episode_id: str) -> dict[str, Any]:
    path = split_dir / "episodic_annotations" / f"{episode_id}.npy"
    arr = np.load(path, allow_pickle=True)
    return arr.item()


def _frame_count_map(video_paths: list[Path], workers: int) -> tuple[dict[str, int], dict[str, str]]:
    lengths: dict[str, int] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_video_len, str(path)) for path in video_paths]
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            video_path, length, error = future.result()
            if length is None:
                failures[video_path] = error or "unknown error"
            else:
                lengths[video_path] = length
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"  checked {done}/{total} videos")
    return lengths, failures


def _clean_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    *,
    hardlink: bool,
    workers: int,
) -> dict[str, Any]:
    src_split_dir = src_root / "Annotation" / split
    dst_split_dir = dst_root / "Annotation" / split
    dst_anno_dir = dst_split_dir / "episodic_annotations"
    dst_anno_dir.mkdir(parents=True, exist_ok=True)

    idx_path = src_split_dir / "episode_frame_index.npz"
    idx = np.load(idx_path, allow_pickle=True)
    index_frame_pair = idx["index_frame_pair"]
    index_to_episode_id = idx["index_to_episode_id"]

    print(f"[{split}] loading {len(index_to_episode_id)} episodes")
    episode_meta: list[dict[str, Any]] = []
    unique_videos: dict[str, Path] = {}
    for old_ep_idx, episode_id in enumerate(index_to_episode_id.tolist()):
        epi = _load_episode(src_split_dir, episode_id)
        video_rel = epi["video_name"]
        video_path = src_root / "Video" / "GigaHands_root" / video_rel
        unique_videos[str(video_path)] = video_path
        episode_meta.append(
            {
                "old_ep_idx": old_ep_idx,
                "episode_id": episode_id,
                "video_path": str(video_path),
                "video_decode_frame": np.asarray(epi["video_decode_frame"], dtype=np.int64),
            }
        )

    print(f"[{split}] checking {len(unique_videos)} unique videos")
    video_lengths, video_failures = _frame_count_map(list(unique_videos.values()), workers)

    old_to_new: dict[int, int] = {}
    kept_episode_ids: list[str] = []
    per_episode_report: list[dict[str, Any]] = []
    valid_by_episode: dict[int, np.ndarray] = {}

    for meta in episode_meta:
        old_ep_idx = int(meta["old_ep_idx"])
        decode_table = meta["video_decode_frame"]
        video_path = meta["video_path"]
        video_len = video_lengths.get(video_path)

        if video_len is None:
            valid_frame_mask = np.zeros(len(decode_table), dtype=bool)
        else:
            valid_frame_mask = (decode_table >= 0) & (decode_table < video_len)

        invalid_frames = int((~valid_frame_mask).sum())
        valid_by_episode[old_ep_idx] = valid_frame_mask
        if bool(valid_frame_mask.any()):
            old_to_new[old_ep_idx] = len(kept_episode_ids)
            kept_episode_ids.append(str(meta["episode_id"]))
            src_anno = src_split_dir / "episodic_annotations" / f"{meta['episode_id']}.npy"
            dst_anno = dst_anno_dir / src_anno.name
            _link_or_copy(src_anno, dst_anno, hardlink=hardlink)

        if invalid_frames or video_len is None:
            per_episode_report.append(
                {
                    "episode_id": str(meta["episode_id"]),
                    "video_path": video_path,
                    "annotation_frames": int(len(decode_table)),
                    "video_frames": None if video_len is None else int(video_len),
                    "invalid_annotation_frames": invalid_frames,
                    "video_error": video_failures.get(video_path),
                }
            )

    kept_pairs: list[np.ndarray] = []
    removed_sample_count = 0
    for old_ep_idx, frame_id in index_frame_pair:
        valid_frame_mask = valid_by_episode[int(old_ep_idx)]
        keep = 0 <= int(frame_id) < len(valid_frame_mask) and bool(valid_frame_mask[int(frame_id)])
        if keep:
            kept_pairs.append(np.asarray([old_to_new[int(old_ep_idx)], int(frame_id)], dtype=np.int64))
        else:
            removed_sample_count += 1

    if kept_pairs:
        clean_index_frame_pair = np.stack(kept_pairs, axis=0).astype(np.int64)
    else:
        clean_index_frame_pair = np.zeros((0, 2), dtype=np.int64)
    clean_index_to_episode_id = np.asarray(kept_episode_ids, dtype=index_to_episode_id.dtype)

    np.savez_compressed(
        dst_split_dir / "episode_frame_index.npz",
        index_frame_pair=clean_index_frame_pair,
        index_to_episode_id=clean_index_to_episode_id,
    )

    for name in ["conversion_report.json"]:
        src = src_split_dir / name
        if src.exists():
            _link_or_copy(src, dst_split_dir / name, hardlink=hardlink)

    report = {
        "split": split,
        "source": str(src_split_dir),
        "destination": str(dst_split_dir),
        "episodes_before": int(len(index_to_episode_id)),
        "episodes_after": int(len(clean_index_to_episode_id)),
        "samples_before": int(len(index_frame_pair)),
        "samples_after": int(len(clean_index_frame_pair)),
        "samples_removed": int(removed_sample_count),
        "videos_checked": int(len(unique_videos)),
        "video_open_failures": int(len(video_failures)),
        "episodes_with_invalid_frames": int(len(per_episode_report)),
        "invalid_episode_report": per_episode_report,
    }
    (dst_split_dir / "cleanup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=Path, required=True)
    parser.add_argument("--dst_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["gigahands_real_train", "gigahands_real_test"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking annotation files.")
    args = parser.parse_args()

    if args.dst_root.exists():
        raise FileExistsError(f"Destination already exists: {args.dst_root}")

    hardlink = not args.copy
    _copy_top_level(args.src_root, args.dst_root, hardlink=hardlink)

    reports = []
    for split in args.splits:
        reports.append(
            _clean_split(
                args.src_root,
                args.dst_root,
                split,
                hardlink=hardlink,
                workers=args.workers,
            )
        )

    summary = {
        "source_root": str(args.src_root),
        "destination_root": str(args.dst_root),
        "hardlinked_annotations": hardlink,
        "splits": reports,
    }
    (args.dst_root / "cleanup_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
