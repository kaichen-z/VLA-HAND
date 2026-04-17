"""Convert OpenTouch HDF5 clips into VITRA Stage-1 keypoint episode format."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_INSTRUCTION = "Perform the demonstrated right-hand touch interaction."


@dataclass(frozen=True)
class OpenTouchClip:
    h5_path: Path
    clip_key: str
    group_path: str
    instruction: str


def clean_instruction(text: str | None) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not text:
        text = DEFAULT_INSTRUCTION
    if not text.endswith("."):
        text += "."
    return text


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def load_labels(labels_path: str | Path | None) -> dict[str, str]:
    if labels_path is None:
        return {}
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels_path does not exist: {labels_path}")

    if labels_path.suffix.lower() == ".json":
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(key): clean_instruction(value if isinstance(value, str) else _row_to_text(value)) for key, value in payload.items()}
        if isinstance(payload, list):
            return _rows_to_labels(payload)

    if labels_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _rows_to_labels(rows)

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        return _rows_to_labels(list(csv.DictReader(handle)))


def _rows_to_labels(rows: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        clip_key = first_present(row, ["clip_id", "clip", "id", "video_id", "episode_id", "group", "group_path"])
        if clip_key is None:
            continue
        labels[str(clip_key)] = clean_instruction(_row_to_text(row))
    return labels


def _row_to_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)
    direct = first_present(row, ["instruction", "text", "description", "caption", "task", "label"])
    if direct:
        return str(direct)
    parts = [
        str(row[key])
        for key in ("action", "object", "grip_type", "surface", "material")
        if key in row and row[key] not in (None, "")
    ]
    return " ".join(parts)


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def discover_clips(opentouch_root: str | Path, labels: dict[str, str] | None = None, max_files: int | None = None, max_clips: int | None = None) -> list[OpenTouchClip]:
    opentouch_root = Path(opentouch_root)
    labels = labels or {}
    h5_paths = sorted(
        path
        for pattern in ("*.h5", "*.hdf5")
        for path in opentouch_root.rglob(pattern)
    )
    if max_files is not None:
        h5_paths = h5_paths[:max_files]

    clips: list[OpenTouchClip] = []
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as handle:
            for group_path, group in iter_clip_groups(handle):
                clip_key = Path(group_path).name
                instruction = labels.get(clip_key) or labels.get(group_path) or labels.get(h5_path.stem)
                clips.append(
                    OpenTouchClip(
                        h5_path=h5_path,
                        clip_key=clip_key,
                        group_path=group_path,
                        instruction=clean_instruction(instruction),
                    )
                )
                if max_clips is not None and len(clips) >= max_clips:
                    return clips
    return clips


def iter_clip_groups(handle: h5py.File):
    required = {"rgb_images_jpeg", "right_hand_landmarks"}

    def visitor(name: str, obj: h5py.Group) -> None:
        if isinstance(obj, h5py.Group) and required.issubset(set(obj.keys())):
            groups.append((name, obj))

    groups: list[tuple[str, h5py.Group]] = []
    handle.visititems(visitor)
    return groups


def jpeg_item_to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.void):
        return bytes(value)
    arr = np.asarray(value)
    if arr.dtype.kind in {"S", "O"} and arr.shape == ():
        return bytes(arr.item())
    return arr.astype(np.uint8).tobytes()


def decode_jpegs_to_video(jpeg_values: list[Any], destination: Path, fps: float = 30.0) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Writing OpenTouch video requires opencv-python/cv2") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    width = height = 0
    try:
        for value in jpeg_values:
            encoded = np.frombuffer(jpeg_item_to_bytes(value), dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to decode JPEG frame for {destination}")
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(destination), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {destination}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    if width == 0 or height == 0:
        raise ValueError(f"No frames were written for {destination}")
    return width, height


def infer_image_size(jpeg_values: list[Any]) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Inferring OpenTouch image size requires opencv-python/cv2") from exc
    encoded = np.frombuffer(jpeg_item_to_bytes(jpeg_values[0]), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode first JPEG frame")
    height, width = frame.shape[:2]
    return width, height


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(raw, dtype=np.float32)
    if landmarks.ndim == 2 and landmarks.shape[1] == 63:
        landmarks = landmarks.reshape(landmarks.shape[0], 21, 3)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise ValueError(f"Expected right_hand_landmarks shape (T,21,3) or (T,63), got {landmarks.shape}")
    return landmarks


def identity_rotations(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    global_orient = np.repeat(np.eye(3, dtype=np.float32)[None, ...], frame_count, axis=0)
    hand_pose = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], frame_count * 15, axis=0).reshape(frame_count, 15, 3, 3)
    return global_orient, hand_pose


def build_side_from_landmarks(landmarks: np.ndarray, extrinsics: np.ndarray, kept_frames: np.ndarray) -> dict[str, np.ndarray]:
    frame_count = len(landmarks)
    global_orient, hand_pose = identity_rotations(frame_count)
    transl_worldspace = landmarks[:, 0, :].astype(np.float32)
    R_w2c = extrinsics[:, :3, :3]
    t_w2c = extrinsics[:, :3, 3]
    transl_camspace = (R_w2c @ transl_worldspace[..., None])[..., 0] + t_w2c
    joints_camspace = (R_w2c @ landmarks.transpose(0, 2, 1)).transpose(0, 2, 1) + t_w2c[:, None, :]
    return {
        "beta": np.zeros(10, dtype=np.float32),
        "global_orient_worldspace": global_orient,
        "global_orient_camspace": (R_w2c @ global_orient).astype(np.float32),
        "hand_pose": hand_pose,
        "transl_worldspace": transl_worldspace,
        "transl_camspace": transl_camspace.astype(np.float32),
        "joints_worldspace": landmarks.astype(np.float32),
        "joints_camspace": joints_camspace.astype(np.float32),
        "kept_frames": kept_frames.astype(bool),
    }


def build_dummy_side(frame_count: int) -> dict[str, np.ndarray]:
    landmarks = np.zeros((frame_count, 21, 3), dtype=np.float32)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], frame_count, axis=0)
    return build_side_from_landmarks(landmarks, extrinsics, np.zeros(frame_count, dtype=bool))


def build_intrinsics(width: int, height: int) -> np.ndarray:
    intrinsics = np.eye(3, dtype=np.float32)
    focal = float(max(width, height))
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0
    return intrinsics


def build_episode(source: OpenTouchClip, opentouch_root: Path, output_root: Path, write_video: bool) -> tuple[str, dict[str, Any]]:
    with h5py.File(source.h5_path, "r") as handle:
        group = handle[source.group_path]
        jpeg_ds = group["rgb_images_jpeg"]
        landmarks = normalize_landmarks(group["right_hand_landmarks"][()])
        frame_count = min(len(jpeg_ds), len(landmarks))
        timestamps = group["timestamps"][()] if "timestamps" in group else np.arange(frame_count, dtype=np.float64) / 30.0
        if len(timestamps) > 1:
            deltas = np.diff(np.asarray(timestamps[:frame_count], dtype=np.float64))
            fps = float(1.0 / np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 30.0
        else:
            fps = 30.0
        pressure = group["right_pressure"][()] if "right_pressure" in group else None
        jpeg_values = [jpeg_ds[idx] for idx in range(frame_count)]

    if pressure is not None:
        frame_count = min(frame_count, len(pressure))
    landmarks = landmarks[:frame_count]
    jpeg_values = jpeg_values[:frame_count]
    timestamps = np.asarray(timestamps[:frame_count], dtype=np.float64)
    if pressure is not None:
        pressure = np.asarray(pressure[:frame_count], dtype=np.float32)

    rel_session = safe_name(source.h5_path.relative_to(opentouch_root).with_suffix("").as_posix())
    rel_video_path = Path(rel_session) / f"{safe_name(source.clip_key)}.mp4"
    output_video_path = output_root / "Video" / "OpenTouch_root" / rel_video_path
    if write_video:
        width, height = decode_jpegs_to_video(jpeg_values, output_video_path, fps=fps)
    else:
        width, height = infer_image_size(jpeg_values)

    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], frame_count, axis=0)
    kept_frames = np.isfinite(landmarks).all(axis=(1, 2))
    right = build_side_from_landmarks(landmarks, extrinsics, kept_frames)
    left = build_dummy_side(frame_count)
    episode_id = f"OpenTouch_{safe_name(source.h5_path.stem)}_{safe_name(source.clip_key)}_ep_000000"
    instruction = clean_instruction(source.instruction)
    episode = {
        "video_name": rel_video_path.as_posix(),
        "video_decode_frame": np.arange(frame_count, dtype=np.int64),
        "intrinsics": build_intrinsics(width, height),
        "extrinsics": extrinsics,
        "anno_type": "right",
        "text": {
            "left": [],
            "right": [(instruction, (0, frame_count))],
        },
        "text_rephrase": {
            "left": [],
            "right": [[instruction]],
        },
        "left": left,
        "right": right,
        "opentouch": {
            "h5_path": str(source.h5_path),
            "group_path": source.group_path,
            "timestamps": timestamps,
        },
    }
    if pressure is not None:
        episode["opentouch"]["right_pressure"] = pressure
    return episode_id, episode


def split_sources(sources: list[OpenTouchClip], train_ratio: float, seed: int) -> dict[str, list[OpenTouchClip]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    shuffled = list(sources)
    random.Random(seed).shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_ratio))
    return {
        "opentouch_keypoint_train": shuffled[:train_count],
        "opentouch_keypoint_test": shuffled[train_count:],
    }


def convert_opentouch_to_vitra_stage1(
    opentouch_root: str | Path,
    output_root: str | Path,
    labels_path: str | Path | None = None,
    min_frames: int = 17,
    min_valid_ratio: float = 0.9,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_files: int | None = None,
    max_clips: int | None = None,
    write_video: bool = True,
) -> dict[str, Any]:
    opentouch_root = Path(opentouch_root)
    output_root = Path(output_root)
    labels = load_labels(labels_path)
    sources = discover_clips(opentouch_root, labels, max_files=max_files, max_clips=max_clips)
    splits = split_sources(sources, train_ratio=train_ratio, seed=seed)

    report: dict[str, Any] = {
        "num_hdf5_files_seen": len({str(source.h5_path) for source in sources}),
        "num_clips_seen": len(sources),
        "num_episodes_written": 0,
        "num_frames_written": 0,
        "skipped_short_clip": 0,
        "skipped_low_valid_ratio": 0,
        "errors": [],
        "format": "VITRA Stage-1 keypoints",
        "source": "OpenTouch",
        "right_hand_supervised": True,
        "left_hand_dummy_masked": True,
        "video_written": bool(write_video),
    }
    (output_root / "Annotation" / "statistics").mkdir(parents=True, exist_ok=True)

    for dataset_name, split_sources_list in splits.items():
        if not split_sources_list:
            continue
        annotation_root = output_root / "Annotation" / dataset_name
        label_root = annotation_root / "episodic_annotations"
        label_root.mkdir(parents=True, exist_ok=True)
        index_frame_pair: list[tuple[int, int]] = []
        index_to_episode_id: list[str] = []

        for source in split_sources_list:
            try:
                episode_id, episode = build_episode(source, opentouch_root, output_root, write_video=write_video)
            except Exception as exc:
                report["errors"].append({"source": str(source.h5_path), "group": source.group_path, "error": str(exc)})
                continue
            frame_count = len(episode["video_decode_frame"])
            if frame_count < min_frames:
                report["skipped_short_clip"] += 1
                continue
            valid_ratio = float(np.mean(episode["right"]["kept_frames"]))
            if valid_ratio < min_valid_ratio:
                report["skipped_low_valid_ratio"] += 1
                continue

            episode_index = len(index_to_episode_id)
            index_to_episode_id.append(episode_id)
            index_frame_pair.extend((episode_index, frame_id) for frame_id in range(frame_count))
            np.save(label_root / f"{episode_id}.npy", episode, allow_pickle=True)
            report["num_episodes_written"] += 1
            report["num_frames_written"] += frame_count

        np.savez(
            annotation_root / "episode_frame_index.npz",
            index_frame_pair=np.asarray(index_frame_pair, dtype=np.int64),
            index_to_episode_id=np.asarray(index_to_episode_id),
        )
        with (annotation_root / "conversion_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    with (output_root / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentouch_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path)
    parser.add_argument("--min_frames", type=int, default=17)
    parser.add_argument("--min_valid_ratio", type=float, default=0.9)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--max_clips", type=int)
    parser.add_argument("--write_video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_opentouch_to_vitra_stage1(
        opentouch_root=args.opentouch_root,
        output_root=args.output_root,
        labels_path=args.labels_path,
        min_frames=args.min_frames,
        min_valid_ratio=args.min_valid_ratio,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_files=args.max_files,
        max_clips=args.max_clips,
        write_video=args.write_video,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
