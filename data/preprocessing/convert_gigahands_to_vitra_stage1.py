"""Convert GigaHands demo/full data into VITRA Stage-1 human episode format."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


CAMERA_DTYPE = [
    ("cam_id", int),
    ("width", int),
    ("height", int),
    ("fx", float),
    ("fy", float),
    ("cx", float),
    ("cy", float),
    ("k1", float),
    ("k2", float),
    ("p1", float),
    ("p2", float),
    ("cam_name", "<U64"),
    ("qvecw", float),
    ("qvecx", float),
    ("qvecy", float),
    ("qvecz", float),
    ("tvecx", float),
    ("tvecy", float),
    ("tvecz", float),
]


@dataclass(frozen=True)
class SourceEpisode:
    sequence_root: Path
    params_path: Path
    keypoints_path: Path | None
    video_path: Path
    camera_path: Path
    instruction: str
    sequence_name: str
    sequence_id: str
    start_frame: int = 0
    end_frame: int | None = None
    split: str = "train"
    camera: str | None = None


@dataclass(frozen=True)
class VideoWritePlan:
    source: Path
    destination: Path
    intrinsics_raw: np.ndarray
    intrinsics_used: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    undistort: bool


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float32,
    )


def read_camera(camera_path: Path, camera_name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
    params = np.loadtxt(camera_path, dtype=CAMERA_DTYPE)
    params = np.atleast_1d(params)
    if camera_name == "auto":
        row = params[0]
    else:
        matches = [row for row in params if str(row["cam_name"]) == camera_name]
        if not matches:
            raise ValueError(f"Camera '{camera_name}' not found in {camera_path}")
        row = matches[0]

    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = row["fx"]
    intrinsics[1, 1] = row["fy"]
    intrinsics[0, 2] = row["cx"]
    intrinsics[1, 2] = row["cy"]

    qvec = np.asarray([row["qvecw"], row["qvecx"], row["qvecy"], row["qvecz"]], dtype=np.float32)
    tvec = np.asarray([row["tvecx"], row["tvecy"], row["tvecz"]], dtype=np.float32)
    rot = qvec2rotmat(-qvec)
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = rot
    extrinsics[:3, 3] = tvec
    dist = np.asarray([row["k1"], row["k2"], row["p1"], row["p2"]], dtype=np.float32)
    return intrinsics, extrinsics, (int(row["width"]), int(row["height"])), dist


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    flat = axis_angle.reshape(-1, 3)
    return R.from_rotvec(flat).as_matrix().reshape(*axis_angle.shape[:-1], 3, 3).astype(np.float32)


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


def find_keypoints_path(sequence_root: Path, sequence_id: str) -> Path | None:
    aliases = [sequence_id]
    if str(sequence_id).isdigit():
        aliases.extend([str(int(sequence_id)), str(sequence_id).zfill(3)])
    aliases = list(dict.fromkeys(aliases))
    candidates = []
    for alias in aliases:
        candidates.extend(
            [
                sequence_root / "keypoints_3d_mano" / f"{alias}.json",
                sequence_root / "keypoints_3d_mano" / alias,
                sequence_root / "keypoints_3d" / f"{alias}.json",
                sequence_root / "keypoints_3d" / alias,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_video_path(sequence_root: Path, camera: str) -> Path | None:
    rgb_root = sequence_root / "rgb_vid"
    if not rgb_root.exists():
        return None
    camera_dirs = sorted(path for path in rgb_root.iterdir() if path.is_dir())
    if camera != "auto":
        exact = rgb_root / camera
        if exact.exists():
            camera_dirs = [exact]
        else:
            camera_dirs = [path for path in camera_dirs if camera in path.name]
    for camera_dir in camera_dirs:
        videos = sorted(camera_dir.glob("*.mp4"))
        if videos:
            return videos[0]
    return None


def find_instruction(video_path: Path, fallback: str) -> str:
    text_files = sorted(video_path.parent.glob("*.txt"))
    for text_file in text_files:
        lines = [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        # GigaHands demo camera folders include timestamp sidecars named *.txt.
        # Those lines are frame ids, not language captions.
        if all(line.startswith("frame_") for line in lines):
            continue
        if len(lines) > 5:
            continue
        return clean_instruction(" ".join(lines))
    return clean_instruction(fallback.replace("_", " ").replace("-", " "))


def resolve_demo_hand_pose_root(root: Path) -> Path:
    hand_pose_root = root / "hand_pose"
    if not hand_pose_root.exists():
        nested = sorted(root.glob("*/hand_pose"))
        if nested:
            hand_pose_root = nested[0]
    if not hand_pose_root.exists():
        raise FileNotFoundError(f"Demo layout requires a hand_pose directory under {root}")
    return hand_pose_root


def summarize_demo_layout(root: Path, camera: str) -> dict[str, int]:
    hand_pose_root = resolve_demo_hand_pose_root(root)
    summary = {
        "num_sequences_seen": 0,
        "skipped_missing_params": 0,
        "skipped_missing_video": 0,
        "skipped_missing_camera": 0,
    }
    for sequence_root in sorted(path for path in hand_pose_root.iterdir() if path.is_dir()):
        summary["num_sequences_seen"] += 1
        params_root = sequence_root / "params"
        if not params_root.exists() or not list(params_root.glob("*.json")):
            summary["skipped_missing_params"] += 1
        if not (sequence_root / "optim_params.txt").exists():
            summary["skipped_missing_camera"] += 1
        if find_video_path(sequence_root, camera) is None:
            summary["skipped_missing_video"] += 1
    return summary


def discover_demo_episodes(root: Path, camera: str) -> list[SourceEpisode]:
    hand_pose_root = resolve_demo_hand_pose_root(root)

    episodes: list[SourceEpisode] = []
    for sequence_root in sorted(path for path in hand_pose_root.iterdir() if path.is_dir()):
        params_root = sequence_root / "params"
        camera_path = sequence_root / "optim_params.txt"
        if not params_root.exists() or not camera_path.exists():
            continue
        video_path = find_video_path(sequence_root, camera)
        if video_path is None:
            continue
        for params_path in sorted(params_root.glob("*.json")):
            sequence_id = params_path.stem
            episodes.append(
                SourceEpisode(
                    sequence_root=sequence_root,
                    params_path=params_path,
                    keypoints_path=find_keypoints_path(sequence_root, sequence_id),
                    video_path=video_path,
                    camera_path=camera_path,
                    instruction=find_instruction(video_path, sequence_root.name),
                    sequence_name=sequence_root.name,
                    sequence_id=sequence_id,
                )
            )
    return episodes


def discover_full_episodes(root: Path, camera: str) -> list[SourceEpisode]:
    annotations_path = root / "annotations_v2.jsonl"
    hand_poses_root = root / "hand_poses"
    if not annotations_path.exists() or not hand_poses_root.exists():
        raise FileNotFoundError(
            "Full layout requires hand_poses/ and annotations_v2.jsonl under the GigaHands root"
        )

    episodes: list[SourceEpisode] = []
    with annotations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row.get("clarify_annotation", "")
            if str(text).strip() in {"None", "Buggy"}:
                continue
            rewrites = row.get("rewritten_annotation") or []
            instruction = clean_instruction(rewrites[0] if rewrites else text)
            sequence = row["sequence"][0] if isinstance(row.get("sequence"), list) else row.get("sequence")
            scene = row.get("scene")
            if not scene or not sequence:
                continue
            sequence_id = str(sequence)
            sequence_root = hand_poses_root / str(scene)
            params_path = sequence_root / "params" / f"{sequence_id}.json"
            camera_path = sequence_root / "optim_params.txt"
            video_path = find_full_video(root, str(scene), sequence_id, camera)
            if video_path is None:
                continue
            episodes.append(
                SourceEpisode(
                    sequence_root=sequence_root,
                    params_path=params_path,
                    keypoints_path=find_keypoints_path(sequence_root, sequence_id),
                    video_path=video_path,
                    camera_path=camera_path,
                    instruction=instruction,
                    sequence_name=str(scene),
                    sequence_id=sequence_id,
                    start_frame=int(row.get("start_frame_id", 0)),
                    end_frame=None if int(row.get("end_frame_id", -1)) == -1 else int(row["end_frame_id"]) + 1,
                    camera=video_path.parent.name,
                )
            )
    return episodes


def discover_manifest_episodes(root: Path, manifest_path: Path, split: str) -> list[SourceEpisode]:
    manifest = load_json(manifest_path)
    clips = manifest.get("clips", [])
    if not isinstance(clips, list):
        raise ValueError("subset_manifest must contain a list field named 'clips'")

    episodes: list[SourceEpisode] = []
    for clip in clips:
        clip_split = str(clip.get("split", "train"))
        if split != "all" and clip_split != split:
            continue
        scene = str(clip.get("scene", clip.get("sequence_name", "")))
        sequence_id = str(clip.get("sequence_id", clip.get("sequence", "")))
        if not scene or not sequence_id:
            raise ValueError(f"Manifest clip is missing scene or sequence_id: {clip}")

        params_path = resolve_manifest_path(root, clip.get("params_path") or f"hand_poses/{scene}/params/{sequence_id}.json")
        camera_path = resolve_manifest_path(root, clip.get("camera_path") or f"hand_poses/{scene}/optim_params.txt")
        video_path = resolve_manifest_path(root, clip.get("video_path"))
        sequence_root = root / "hand_poses" / scene
        instruction = clean_instruction(str(clip.get("instruction", scene)))
        episodes.append(
            SourceEpisode(
                sequence_root=sequence_root,
                params_path=params_path,
                keypoints_path=find_keypoints_path(sequence_root, sequence_id),
                video_path=video_path,
                camera_path=camera_path,
                instruction=instruction,
                sequence_name=scene,
                sequence_id=sequence_id,
                start_frame=int(clip.get("start_frame", 0)),
                end_frame=None if clip.get("end_frame") is None else int(clip["end_frame"]),
                split=clip_split,
                camera=str(clip.get("camera", video_path.parent.name)),
            )
        )
    return episodes


def resolve_manifest_path(root: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Manifest clip is missing a required path")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def find_full_video(root: Path, scene: str, sequence_id: str, camera: str) -> Path | None:
    video_root = root / "multiview_rgb_vids" / scene
    if not video_root.exists():
        return None
    camera_dirs = sorted(path for path in video_root.iterdir() if path.is_dir())
    if camera != "auto":
        exact = video_root / camera
        if exact.exists():
            camera_dirs = [exact]
        else:
            camera_dirs = [path for path in camera_dirs if camera in path.name]
    for camera_dir in camera_dirs:
        videos = sorted(camera_dir.glob(f"*{sequence_id}*.mp4"))
        if videos:
            return videos[0]
    return None


def load_hand_params(params_path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = load_json(params_path)
    return {
        side: {key: np.asarray(value, dtype=np.float32) for key, value in raw[side].items()}
        for side in ("left", "right")
    }


def slice_hand_params(params: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    sliced: dict[str, np.ndarray] = {}
    for key, value in params.items():
        if value.ndim > 0 and value.shape[0] >= end:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def normalize_pose_fields(params: dict[str, np.ndarray], frame_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poses = np.asarray(params.get("poses"), dtype=np.float32)
    if poses.ndim != 2:
        raise ValueError("MANO params must contain a 2D poses array")
    if "Rh" in params:
        global_axis_angle = np.asarray(params["Rh"], dtype=np.float32)
        hand_axis_angle = poses[:, 3:48] if poses.shape[1] >= 48 else poses[:, :45]
    elif poses.shape[1] >= 48:
        global_axis_angle = poses[:, :3]
        hand_axis_angle = poses[:, 3:48]
    elif poses.shape[1] == 45:
        global_axis_angle = np.zeros((frame_count, 3), dtype=np.float32)
        hand_axis_angle = poses
    else:
        raise ValueError(f"Unsupported MANO poses dimension: {poses.shape}")

    transl = np.asarray(params.get("Th", params.get("transl", np.zeros((frame_count, 3)))), dtype=np.float32)
    shapes = np.asarray(params.get("shapes", np.zeros((frame_count, 10))), dtype=np.float32)
    if shapes.ndim == 1:
        beta = shapes[:10]
    else:
        beta = np.nanmean(shapes[:, :10], axis=0)
    return global_axis_angle[:frame_count], hand_axis_angle[:frame_count], transl[:frame_count], beta.astype(np.float32)


def load_keypoint_fallback(path: Path | None, frame_count: int) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    if path.is_dir():
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            return None
        path = json_files[0]
    raw = load_json(path)
    if isinstance(raw, dict) and "left" in raw and "right" in raw:
        left = np.asarray(raw["left"], dtype=np.float32)[:frame_count]
        right = np.asarray(raw["right"], dtype=np.float32)[:frame_count]
        return {"left": left.reshape(left.shape[0], 21, 3), "right": right.reshape(right.shape[0], 21, 3)}
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 126:
        arr = arr.reshape(arr.shape[0], 42, 3)
    if arr.ndim == 3 and arr.shape[1] >= 42:
        arr = arr[:frame_count]
        return {"left": arr[:, :21, :], "right": arr[:, 21:42, :]}
    return None


def build_hand_dict(
    params: dict[str, np.ndarray],
    joints_worldspace: np.ndarray | None,
    extrinsics: np.ndarray,
) -> dict[str, np.ndarray]:
    frame_count = len(extrinsics)
    global_axis_angle, hand_axis_angle, transl_worldspace, beta = normalize_pose_fields(params, frame_count)
    global_orient_worldspace = axis_angle_to_matrix(global_axis_angle)
    hand_pose = axis_angle_to_matrix(hand_axis_angle.reshape(frame_count, 15, 3))

    if joints_worldspace is None:
        joints_worldspace = np.repeat(transl_worldspace[:, None, :], 21, axis=1)
    joints_worldspace = np.asarray(joints_worldspace, dtype=np.float32)[:frame_count]

    R_w2c = extrinsics[:, :3, :3]
    t_w2c = extrinsics[:, :3, 3]
    global_orient_camspace = R_w2c @ global_orient_worldspace
    transl_camspace = (R_w2c @ transl_worldspace[..., None])[..., 0] + t_w2c
    joints_camspace = (R_w2c @ joints_worldspace.transpose(0, 2, 1)).transpose(0, 2, 1) + t_w2c[:, None, :]
    kept_frames = np.isfinite(transl_worldspace).all(axis=1) & np.isfinite(hand_axis_angle).all(axis=1)

    return {
        "beta": beta.astype(np.float32),
        "global_orient_worldspace": global_orient_worldspace.astype(np.float32),
        "global_orient_camspace": global_orient_camspace.astype(np.float32),
        "hand_pose": hand_pose.astype(np.float32),
        "transl_worldspace": transl_worldspace.astype(np.float32),
        "transl_camspace": transl_camspace.astype(np.float32),
        "joints_worldspace": joints_worldspace.astype(np.float32),
        "joints_camspace": joints_camspace.astype(np.float32),
        "kept_frames": kept_frames.astype(bool),
    }


def choose_main_hand(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> str:
    left_motion = float(np.linalg.norm(left["transl_worldspace"][-1] - left["transl_worldspace"][0]))
    right_motion = float(np.linalg.norm(right["transl_worldspace"][-1] - right["transl_worldspace"][0]))
    return "left" if left_motion > right_motion else "right"


def undistorted_intrinsics(intrinsics: np.ndarray, dist: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--undistort requires opencv-python/cv2 to be installed") from exc

    new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(intrinsics, dist, image_size, alpha=1)
    return new_intrinsics.astype(np.float32)


def metadata_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_camera_metadata(
    *,
    camera_name: str,
    camera_path: Path,
    source_video_path: Path,
    intrinsics_raw: np.ndarray,
    intrinsics_used: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    undistorted: bool,
    root: Path,
) -> dict[str, Any]:
    return {
        "name": camera_name,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "intrinsics": np.asarray(intrinsics_used, dtype=np.float32),
        "original_intrinsics": np.asarray(intrinsics_raw, dtype=np.float32),
        "distortion": np.asarray(distortion, dtype=np.float32),
        "undistorted": bool(undistorted),
        "camera_path": metadata_path(camera_path, root),
        "source_video_path": metadata_path(source_video_path, root),
    }


def write_video_from_plan(plan: VideoWritePlan) -> bool:
    source = plan.source
    destination = plan.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not plan.undistort:
        shutil.copy2(source, destination)
        return False

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--undistort requires opencv-python/cv2 to be installed") from exc

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for undistortion: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destination), fourcc, fps, (width, height))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.undistort(frame, plan.intrinsics_raw, plan.distortion, None, plan.intrinsics_used))
    finally:
        cap.release()
        writer.release()
    return True


def build_episode(source: SourceEpisode, output_root: Path, gigahands_root: Path, camera: str, write_video: bool, undistort: bool) -> tuple[str, dict[str, Any], VideoWritePlan | None]:
    params = load_hand_params(source.params_path)
    total_frames = min(len(params["left"]["poses"]), len(params["right"]["poses"]))
    start = max(source.start_frame, 0)
    end = min(source.end_frame if source.end_frame is not None else total_frames, total_frames)
    frame_count = end - start
    sliced_params = {side: slice_hand_params(params[side], start, end) for side in ("left", "right")}

    camera_name = source.camera or (source.video_path.parent.name if camera == "auto" else camera)
    intrinsics, extrinsic, image_size, dist = read_camera(source.camera_path, camera_name)
    rel_video_path = Path(source.sequence_name) / camera_name / source.video_path.name
    output_video_path = output_root / "Video" / "GigaHands_root" / rel_video_path
    should_undistort = bool(write_video and undistort)
    intrinsics_used = undistorted_intrinsics(intrinsics, dist, image_size) if should_undistort else intrinsics
    video_plan = None
    if write_video:
        video_plan = VideoWritePlan(
            source=source.video_path,
            destination=output_video_path,
            intrinsics_raw=intrinsics.astype(np.float32),
            intrinsics_used=intrinsics_used.astype(np.float32),
            distortion=dist.astype(np.float32),
            image_size=image_size,
            undistort=should_undistort,
        )
    extrinsics = np.repeat(extrinsic[None, ...], frame_count, axis=0).astype(np.float32)

    keypoints = load_keypoint_fallback(source.keypoints_path, total_frames)
    if keypoints is not None:
        keypoints = {side: value[start:end] for side, value in keypoints.items()}

    left = build_hand_dict(sliced_params["left"], None if keypoints is None else keypoints.get("left"), extrinsics)
    right = build_hand_dict(sliced_params["right"], None if keypoints is None else keypoints.get("right"), extrinsics)
    clip_name = safe_name(source.video_path.stem)
    episode_id = (
        f"GigaHands_{safe_name(source.sequence_name)}_{safe_name(source.sequence_id)}_"
        f"{safe_name(camera_name)}_{clip_name}_f{start:06d}_{end:06d}_ep_000000"
    )
    instruction = source.instruction
    camera_metadata = build_camera_metadata(
        camera_name=camera_name,
        camera_path=source.camera_path,
        source_video_path=source.video_path,
        intrinsics_raw=intrinsics,
        intrinsics_used=intrinsics_used,
        distortion=dist,
        image_size=image_size,
        undistorted=should_undistort,
        root=gigahands_root,
    )
    episode = {
        "video_name": str(rel_video_path),
        "video_decode_frame": np.arange(start, end, dtype=np.int64),
        "intrinsics": intrinsics_used.astype(np.float32),
        "extrinsics": extrinsics,
        "camera": camera_metadata,
        "anno_type": choose_main_hand(left, right),
        "text": {
            "left": [(instruction, (0, frame_count))],
            "right": [(instruction, (0, frame_count))],
        },
        "text_rephrase": {
            "left": [[instruction]],
            "right": [[instruction]],
        },
        "left": left,
        "right": right,
    }
    return episode_id, episode, video_plan


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def init_report(camera: str, undistort: bool) -> dict[str, Any]:
    return {
        "num_sequences_seen": 0,
        "num_annotations_seen": 0,
        "num_episodes_written": 0,
        "num_frames_written": 0,
        "skipped_missing_video": 0,
        "skipped_missing_params": 0,
        "skipped_missing_camera": 0,
        "skipped_short_clip": 0,
        "skipped_low_valid_ratio": 0,
        "used_mano_joints": 0,
        "used_keypoint_fallback_joints": 0,
        "camera": camera,
        "undistorted": False,
        "undistorted_requested": bool(undistort),
        "undistorted_written": False,
        "num_undistorted_videos": 0,
        "num_raw_copied_videos": 0,
        "video_written": False,
        "errors": [],
    }


def convert_gigahands_to_vitra(
    gigahands_root: str | Path,
    output_root: str | Path,
    input_layout: str = "demo",
    camera: str = "brics-odroid-011_cam0",
    subset_manifest: str | Path | None = None,
    split: str = "all",
    dataset_name_prefix: str = "gigahands",
    min_frames: int = 17,
    min_valid_ratio: float = 0.9,
    write_video: bool = True,
    undistort: bool = False,
    clean_output: bool = False,
) -> dict[str, Any]:
    gigahands_root = Path(gigahands_root)
    output_root = Path(output_root)
    if clean_output:
        shutil.rmtree(output_root / "Annotation", ignore_errors=True)
        shutil.rmtree(output_root / "Video" / "GigaHands_root", ignore_errors=True)
    if input_layout == "demo":
        demo_summary = summarize_demo_layout(gigahands_root, camera)
        sources = discover_demo_episodes(gigahands_root, camera)
    elif input_layout == "full":
        demo_summary = {}
        if subset_manifest is not None:
            sources = discover_manifest_episodes(gigahands_root, Path(subset_manifest), split)
        else:
            sources = discover_full_episodes(gigahands_root, camera)
    else:
        raise ValueError("--input_layout must be 'demo' or 'full'")

    report = init_report(camera, undistort)
    report.update(demo_summary)
    report["num_annotations_seen"] = report["num_sequences_seen"] if input_layout == "demo" else len(sources)
    (output_root / "Annotation" / "statistics").mkdir(parents=True, exist_ok=True)
    if subset_manifest is not None:
        shutil.copy2(subset_manifest, output_root / "subset_manifest.json")

    datasets: dict[str, dict[str, Any]] = {}
    for source in sources:
        dataset_name = dataset_name_for_source(input_layout, dataset_name_prefix, source.split)
        dataset_state = datasets.setdefault(dataset_name, {"pairs": [], "ids": [], "id_set": set()})
        annotation_root = output_root / "Annotation" / dataset_name
        label_root = annotation_root / "episodic_annotations"
        label_root.mkdir(parents=True, exist_ok=True)

        if not source.params_path.exists():
            report["skipped_missing_params"] += 1
            continue
        if not source.video_path.exists():
            report["skipped_missing_video"] += 1
            continue
        if not source.camera_path.exists():
            report["skipped_missing_camera"] += 1
            continue
        try:
            episode_id, episode, video_plan = build_episode(source, output_root, gigahands_root, camera, write_video, undistort)
        except ValueError as exc:
            if "Camera" in str(exc):
                report["skipped_missing_camera"] += 1
            else:
                report["errors"].append({"source": str(source.params_path), "error": str(exc)})
            continue
        except Exception as exc:
            report["errors"].append({"source": str(source.params_path), "error": str(exc)})
            continue

        frame_count = len(episode["video_decode_frame"])
        if frame_count < min_frames:
            report["skipped_short_clip"] += 1
            continue
        valid_ratio = min(
            float(np.mean(episode["left"]["kept_frames"])),
            float(np.mean(episode["right"]["kept_frames"])),
        )
        if valid_ratio < min_valid_ratio:
            report["skipped_low_valid_ratio"] += 1
            continue

        episode_index = len(dataset_state["ids"])
        if episode_id in dataset_state["id_set"]:
            report["errors"].append({"source": str(source.params_path), "error": f"Duplicate episode_id: {episode_id}"})
            continue

        if video_plan is not None:
            try:
                did_undistort = write_video_from_plan(video_plan)
            except Exception as exc:
                report["errors"].append({"source": str(source.video_path), "error": str(exc)})
                continue
            report["video_written"] = True
            if did_undistort:
                report["num_undistorted_videos"] += 1
            else:
                report["num_raw_copied_videos"] += 1
            report["undistorted_written"] = report["undistorted_written"] or did_undistort

        dataset_state["id_set"].add(episode_id)
        dataset_state["ids"].append(episode_id)
        dataset_state["pairs"].extend((episode_index, frame_id) for frame_id in range(frame_count))
        np.save(label_root / f"{episode_id}.npy", episode, allow_pickle=True)
        report["num_episodes_written"] += 1
        report["num_frames_written"] += frame_count
        if source.keypoints_path is None:
            report["used_mano_joints"] += 1
        else:
            report["used_keypoint_fallback_joints"] += 1

    report["undistorted"] = bool(report["num_undistorted_videos"] > 0 and report["num_raw_copied_videos"] == 0)

    for dataset_name, dataset_state in datasets.items():
        annotation_root = output_root / "Annotation" / dataset_name
        annotation_root.mkdir(parents=True, exist_ok=True)
        np.savez(
            annotation_root / "episode_frame_index.npz",
            index_frame_pair=np.asarray(dataset_state["pairs"], dtype=np.int64),
            index_to_episode_id=np.asarray(dataset_state["ids"]),
        )
        with (annotation_root / "conversion_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    with (output_root / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def dataset_name_for_source(input_layout: str, dataset_name_prefix: str, split: str) -> str:
    if input_layout == "demo" or dataset_name_prefix == "gigahands":
        return "gigahands"
    return f"{dataset_name_prefix}_{split}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gigahands_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--input_layout", choices=["demo", "full"], default="demo")
    parser.add_argument("--camera", default="brics-odroid-011_cam0")
    parser.add_argument("--subset_manifest", type=Path)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--dataset_name_prefix", default="gigahands")
    parser.add_argument("--min_frames", type=int, default=17)
    parser.add_argument("--min_valid_ratio", type=float, default=0.9)
    parser.add_argument("--write_video", action="store_true")
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--clean_output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_gigahands_to_vitra(
        gigahands_root=args.gigahands_root,
        output_root=args.output_root,
        input_layout=args.input_layout,
        camera=args.camera,
        subset_manifest=args.subset_manifest,
        split=args.split,
        dataset_name_prefix=args.dataset_name_prefix,
        min_frames=args.min_frames,
        min_valid_ratio=args.min_valid_ratio,
        write_video=args.write_video,
        undistort=args.undistort,
        clean_output=args.clean_output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
