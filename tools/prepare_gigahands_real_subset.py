"""Select a small real GigaHands subset for VITRA Stage-1 fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def clean_instruction(text: str) -> str:
    text = " ".join(str(text).replace("\n", " ").split()).strip()
    if not text or text.lower() in {"none", "buggy"}:
        text = "Perform the demonstrated hand activity."
    if not text.endswith("."):
        text += "."
    return text


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_sequence_id(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def sequence_aliases(sequence: str) -> list[str]:
    aliases = [sequence]
    if sequence.isdigit():
        aliases.append(str(int(sequence)))
        aliases.append(sequence.zfill(3))
    return list(dict.fromkeys(aliases))


def load_video_map(map_path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        rows: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in reader:
            normalized = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}
            scene = first_present(normalized, ["scene", "session", "session_name", "sequence_name"])
            sequence = normalize_sequence_id(first_present(normalized, ["sequence", "seqid", "seq_id", "sequence_id", "id"]))
            if not scene or not sequence:
                continue
            for alias in sequence_aliases(sequence):
                rows.setdefault((scene, alias), []).append(normalized)
        return rows


def first_present(row: dict[str, str], names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value
    return default


def infer_camera(row: dict[str, str], prefer_camera: str) -> str:
    camera = first_present(row, ["camera", "camera_name", "cam", "cam_name", "view"])
    return camera or prefer_camera


def infer_video_path(root: Path, scene: str, sequence_id: str, camera: str, map_rows: list[dict[str, str]]) -> Path:
    for row in map_rows:
        raw_camera_path = row.get(camera, "")
        if raw_camera_path:
            path = Path(raw_camera_path)
            if path.is_absolute():
                return path
            if len(path.parts) > 0 and path.parts[0] == "multiview_rgb_vids":
                return root / path
            return root / "multiview_rgb_vids" / scene / path

        raw_path = first_present(row, ["video_path", "path", "file", "filename", "video", "rgb_video"])
        if raw_path:
            path = Path(raw_path)
            return path if path.is_absolute() else root / path

    video_root = root / "multiview_rgb_vids" / scene / camera
    matches = sorted(video_root.glob(f"*{sequence_id}*.mp4"))
    if matches:
        return matches[0]
    return video_root / f"{camera}_{sequence_id}.mp4"


def hand_params_valid(params_path: Path, start: int, end: int, require_both_hands_valid: bool) -> tuple[bool, float]:
    if not params_path.exists():
        return False, 0.0
    raw = load_json(params_path)
    valid_sides = []
    motions = []
    for side in ("left", "right"):
        if side not in raw:
            valid_sides.append(False)
            motions.append(0.0)
            continue
        side_data = raw[side]
        poses = np.asarray(side_data.get("poses", []), dtype=np.float32)
        transl = np.asarray(side_data.get("Th", side_data.get("transl", [])), dtype=np.float32)
        side_end = min(end, len(poses), len(transl))
        if poses.ndim != 2 or transl.ndim != 2 or side_end <= start:
            valid_sides.append(False)
            motions.append(0.0)
            continue
        valid = np.isfinite(poses[start:side_end]).all() and np.isfinite(transl[start:side_end]).all()
        motion = float(np.linalg.norm(transl[side_end - 1] - transl[start])) if valid else 0.0
        valid_sides.append(bool(valid))
        motions.append(motion)
    if require_both_hands_valid:
        return all(valid_sides), min(motions)
    return any(valid_sides), max(motions)


def iter_annotation_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_candidates(
    gigahands_root: Path,
    min_frames: int,
    prefer_camera: str,
    require_both_hands_valid: bool,
    prefer_bimanual_motion: bool,
    target_candidates: int | None = None,
) -> list[dict[str, Any]]:
    annotations_path = gigahands_root / "annotations_v2.jsonl"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")
    hand_poses_root = gigahands_root / "hand_poses"
    if not hand_poses_root.exists():
        raise FileNotFoundError(f"Missing hand_poses directory: {hand_poses_root}")

    video_map = load_video_map(gigahands_root / "multiview_camera_video_map.csv")
    candidates = []
    for row in iter_annotation_rows(annotations_path):
        raw_text = row.get("rewritten_annotation") or row.get("clarify_annotation") or ""
        if isinstance(raw_text, list):
            raw_text = raw_text[0] if raw_text else ""
        if str(raw_text).strip().lower() in {"none", "buggy"}:
            continue
        scene = str(row.get("scene", "")).strip()
        sequence_id = normalize_sequence_id(row.get("sequence", row.get("sequence_id", "")))
        if not scene or not sequence_id:
            continue

        start = max(int(row.get("start_frame_id", row.get("start_frame", 0)) or 0), 0)
        raw_end = int(row.get("end_frame_id", row.get("end_frame", -1)) or -1)
        params_path = gigahands_root / "hand_poses" / scene / "params" / f"{sequence_id}.json"
        frame_count = infer_frame_count(params_path)
        end = min(raw_end + 1 if raw_end >= 0 else frame_count, frame_count)
        if end - start < min_frames:
            continue

        valid, bimanual_motion = hand_params_valid(params_path, start, end, require_both_hands_valid)
        if not valid:
            continue

        map_rows = video_map.get((scene, sequence_id), [])
        camera = choose_camera(gigahands_root, scene, sequence_id, prefer_camera, map_rows)
        video_path = infer_video_path(gigahands_root, scene, sequence_id, camera, map_rows)
        clip = {
            "clip_id": f"{scene}_{sequence_id}",
            "scene": scene,
            "sequence_id": sequence_id,
            "camera": camera,
            "instruction": clean_instruction(str(raw_text)),
            "start_frame": start,
            "end_frame": end,
            "num_frames": end - start,
            "bimanual_motion": bimanual_motion,
            "params_path": str(params_path.relative_to(gigahands_root)),
            "camera_path": str((gigahands_root / "hand_poses" / scene / "optim_params.txt").relative_to(gigahands_root)),
            "video_path": str(video_path.relative_to(gigahands_root)) if not video_path.is_absolute() or video_path.is_relative_to(gigahands_root) else str(video_path),
        }
        candidates.append(clip)
        if target_candidates is not None and len(candidates) >= target_candidates:
            break

    if prefer_bimanual_motion:
        candidates.sort(key=lambda item: (-float(item["bimanual_motion"]), item["scene"], item["sequence_id"]))
    else:
        candidates.sort(key=lambda item: (item["scene"], item["sequence_id"]))
    return candidates


def infer_frame_count(params_path: Path) -> int:
    if not params_path.exists():
        return 0
    raw = load_json(params_path)
    counts = []
    for side in ("left", "right"):
        if side in raw and "poses" in raw[side]:
            counts.append(len(raw[side]["poses"]))
    return min(counts) if counts else 0


def choose_camera(root: Path, scene: str, sequence_id: str, prefer_camera: str, map_rows: list[dict[str, str]]) -> str:
    for row in map_rows:
        if row.get(prefer_camera):
            return prefer_camera
    existing = root / "multiview_rgb_vids" / scene / prefer_camera
    if existing.exists():
        return prefer_camera
    for row in map_rows:
        camera = infer_camera(row, prefer_camera)
        if camera == prefer_camera:
            return camera
    for row in map_rows:
        camera = infer_camera(row, "")
        if camera:
            return camera
        for key, value in row.items():
            if key.startswith("brics-odroid-") and value:
                return key
    camera_dirs = sorted((root / "multiview_rgb_vids" / scene).glob("*"))
    camera_dirs = [path for path in camera_dirs if path.is_dir()]
    if camera_dirs:
        return camera_dirs[0].name
    return prefer_camera


def prepare_real_subset(
    gigahands_root: str | Path,
    num_train: int,
    num_test: int,
    min_frames: int,
    prefer_camera: str,
    require_both_hands_valid: bool,
    prefer_bimanual_motion: bool,
    output_manifest: str | Path,
    output_video_list: str | Path,
    candidate_pool_factor: int = 4,
) -> dict[str, Any]:
    gigahands_root = Path(gigahands_root)
    output_manifest = Path(output_manifest)
    output_video_list = Path(output_video_list)

    candidates = build_candidates(
        gigahands_root=gigahands_root,
        min_frames=min_frames,
        prefer_camera=prefer_camera,
        require_both_hands_valid=require_both_hands_valid,
        prefer_bimanual_motion=prefer_bimanual_motion,
        target_candidates=max(num_train + num_test, (num_train + num_test) * candidate_pool_factor),
    )
    selected = candidates[: num_train + num_test]
    train = selected[:num_train]
    test = selected[num_train : num_train + num_test]
    for clip in train:
        clip["split"] = "train"
        clip["clip_id"] = f"train_{clip['clip_id']}"
    for clip in test:
        clip["split"] = "test"
        clip["clip_id"] = f"test_{clip['clip_id']}"

    payload = {
        "version": 1,
        "source_root": str(gigahands_root),
        "selection": {
            "num_train": num_train,
            "num_test": num_test,
            "min_frames": min_frames,
            "prefer_camera": prefer_camera,
            "require_both_hands_valid": require_both_hands_valid,
            "prefer_bimanual_motion": prefer_bimanual_motion,
            "candidate_pool_factor": candidate_pool_factor,
        },
        "splits": {
            "train": [clip["clip_id"] for clip in train],
            "test": [clip["clip_id"] for clip in test],
        },
        "clips": train + test,
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_video_list.parent.mkdir(parents=True, exist_ok=True)
    output_video_list.write_text("\n".join(clip["video_path"] for clip in train + test) + "\n", encoding="utf-8")

    return {
        "candidates": len(candidates),
        "selected_train": len(train),
        "selected_test": len(test),
        "manifest": str(output_manifest),
        "needed_videos": str(output_video_list),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gigahands_root", type=Path, required=True)
    parser.add_argument("--num_train", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--min_frames", type=int, default=32)
    parser.add_argument("--prefer_camera", default="brics-odroid-011_cam0")
    parser.add_argument("--require_both_hands_valid", action="store_true")
    parser.add_argument("--prefer_bimanual_motion", action="store_true")
    parser.add_argument("--candidate_pool_factor", type=int, default=4)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--output_video_list", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_real_subset(
        gigahands_root=args.gigahands_root,
        num_train=args.num_train,
        num_test=args.num_test,
        min_frames=args.min_frames,
        prefer_camera=args.prefer_camera,
        require_both_hands_valid=args.require_both_hands_valid,
        prefer_bimanual_motion=args.prefer_bimanual_motion,
        output_manifest=args.output_manifest,
        output_video_list=args.output_video_list,
        candidate_pool_factor=args.candidate_pool_factor,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
