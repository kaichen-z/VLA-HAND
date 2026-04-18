"""Evaluate VITRA Stage-1 predictions on a converted GigaHands split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

FINGER_BASES = np.asarray(
    [
        [-0.050, 0.018, 0.010],
        [-0.025, 0.052, 0.000],
        [0.000, 0.060, 0.000],
        [0.025, 0.052, 0.000],
        [0.050, 0.038, 0.000],
    ],
    dtype=np.float32,
)
FINGER_DIRS = np.asarray(
    [
        [-0.55, 0.78, 0.12],
        [-0.20, 0.98, 0.02],
        [0.00, 1.00, 0.00],
        [0.22, 0.97, 0.02],
        [0.45, 0.88, 0.08],
    ],
    dtype=np.float32,
)
FINGER_LENGTHS = np.asarray(
    [
        [0.040, 0.032, 0.026, 0.020],
        [0.050, 0.034, 0.025, 0.018],
        [0.055, 0.037, 0.027, 0.020],
        [0.050, 0.034, 0.025, 0.018],
        [0.042, 0.028, 0.021, 0.016],
    ],
    dtype=np.float32,
)


def compute_action_metrics(prediction: np.ndarray, target: np.ndarray, action_masks: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 3 or prediction.shape[-1] % 2 != 0:
        raise ValueError("prediction and target must be shaped [N, T, D] with even D")
    half = prediction.shape[-1] // 2
    left_err = (prediction[..., :half] - target[..., :half]) ** 2
    right_err = (prediction[..., half:] - target[..., half:]) ** 2
    all_err = np.concatenate([left_err, right_err], axis=-1)
    if action_masks.shape == prediction.shape:
        left_mask = action_masks[..., :half]
        right_mask = action_masks[..., half:]
        left_frame_mask = left_mask.any(axis=-1)
        right_frame_mask = right_mask.any(axis=-1)
        dual_frame_mask = left_frame_mask & right_frame_mask
        dual_mask = np.concatenate([left_mask, right_mask], axis=-1) & dual_frame_mask[..., None]
        any_mask = action_masks
    elif action_masks.shape[:2] == prediction.shape[:2] and action_masks.shape[-1] == 2:
        left_frame_mask = action_masks[..., 0]
        right_frame_mask = action_masks[..., 1]
        dual_frame_mask = left_frame_mask & right_frame_mask
        left_mask = left_frame_mask
        right_mask = right_frame_mask
        dual_mask = dual_frame_mask
        any_mask = left_frame_mask | right_frame_mask
    else:
        raise ValueError("action_masks must be shaped [N, T, 2] or match prediction shape [N, T, D]")

    left_mse = masked_mean(left_err, left_mask)
    right_mse = masked_mean(right_err, right_mask)
    dual_mse = masked_mean(all_err, dual_mask)

    return {
        "action_mse": masked_mean(all_err, any_mask),
        "left_action_mse": left_mse,
        "right_action_mse": right_mse,
        "dual_hand_action_mse": dual_mse,
        "valid_frame_count": int((left_frame_mask | right_frame_mask).sum()),
        "bimanual_frame_count": int(dual_frame_mask.sum()),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def evaluate_predictions(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, float]:
    return compute_action_metrics(np.stack(predictions), np.stack(targets), np.stack(masks))


def hand_frame_masks(action_mask: np.ndarray, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    action_mask = np.asarray(action_mask, dtype=bool)
    if action_mask.shape[-1] == 2:
        return action_mask[:, 0], action_mask[:, 1]
    half = action_dim // 2
    return action_mask[:, :half].any(axis=-1), action_mask[:, half:action_dim].any(axis=-1)


def to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr.astype(dtype) if dtype is not None else arr


def unnormalize_padded_actions(normalizer: Any, actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    action_dim = normalizer.action_mean.shape[0]
    return normalizer.unnormalize_action(actions[:, :action_dim])


def split_state_122(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    return state[:51], state[61:112]


def split_state_beta_122(state: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] < 122:
        raise ValueError(f"Expected 122-dim GigaHands state, got shape {state.shape}")
    return (state[:51], state[51:61]), (state[61:112], state[112:122])


def recon_traj_from_actions(state: np.ndarray, rel_action: np.ndarray) -> np.ndarray:
    t_cur = np.asarray(state[:3], dtype=np.float32)
    R_cur = R.from_euler("xyz", state[3:6]).as_matrix()
    pose_cur = R.from_euler("xyz", state[6:51].reshape(15, 3)).as_matrix()
    traj = [np.asarray(state[:51], dtype=np.float32)]
    for action in np.asarray(rel_action, dtype=np.float32):
        t_next = t_cur + action[:3]
        R_next = R.from_euler("xyz", action[3:6]).as_matrix() @ R_cur
        pose_next = R.from_euler("xyz", action[6:51].reshape(15, 3)).as_matrix()
        traj.append(
            np.concatenate(
                [
                    t_next,
                    R.from_matrix(R_next).as_euler("xyz", degrees=False),
                    R.from_matrix(pose_next).as_euler("xyz", degrees=False).reshape(-1),
                ]
            ).astype(np.float32)
        )
        t_cur, R_cur, pose_cur = t_next, R_next, pose_next
    return np.stack(traj)


def canonical_hand_joints(hand_pose_euler: np.ndarray, is_left: bool) -> np.ndarray:
    pose = np.asarray(hand_pose_euler, dtype=np.float32).reshape(15, 3)
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[0] = 0.0
    joint_cursor = 0
    for finger_idx, chain in enumerate(FINGER_CHAINS):
        base = FINGER_BASES[finger_idx].copy()
        direction = FINGER_DIRS[finger_idx].copy()
        if is_left:
            base[0] *= -1
            direction[0] *= -1
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        joints[chain[1]] = base
        local_rot = np.eye(3, dtype=np.float32)
        for segment_idx, joint_id in enumerate(chain[2:]):
            if joint_cursor < len(pose):
                local_rot = local_rot @ R.from_euler("xyz", pose[joint_cursor]).as_matrix().astype(np.float32)
            bend = local_rot @ direction
            joints[joint_id] = joints[chain[1 + segment_idx]] + bend * FINGER_LENGTHS[finger_idx, segment_idx + 1]
            joint_cursor += 1
    return joints


def hand_traj_to_joints(traj: np.ndarray, is_left: bool) -> np.ndarray:
    out = []
    for frame in np.asarray(traj, dtype=np.float32):
        local = canonical_hand_joints(frame[6:51], is_left=is_left)
        R_root = R.from_euler("xyz", frame[3:6]).as_matrix().astype(np.float32)
        out.append((R_root @ local.T).T + frame[:3])
    return np.stack(out).astype(np.float32)


def traj_to_mano_labels(traj: np.ndarray, beta: np.ndarray) -> dict[str, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    return {
        "transl_worldspace": traj[:, :3],
        "global_orient_worldspace": R.from_euler("xyz", traj[:, 3:6]).as_matrix().astype(np.float32),
        "hand_pose": R.from_euler("xyz", traj[:, 6:51].reshape(-1, 3)).as_matrix().reshape(len(traj), 15, 3, 3).astype(np.float32),
        "beta": np.asarray(beta, dtype=np.float32),
    }


def mano_vertices_from_labels(mano: Any, labels: dict[str, np.ndarray], is_left: bool, device: str) -> np.ndarray:
    import torch

    wrist_worldspace = labels["transl_worldspace"].reshape(-1, 1, 3).astype(np.float32)
    wrist_orientation = labels["global_orient_worldspace"].astype(np.float32)
    pose = labels["hand_pose"].astype(np.float32)
    beta = labels["beta"].astype(np.float32)
    num_frames = pose.shape[0]

    beta_torch = torch.from_numpy(beta).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1)
    pose_torch = torch.from_numpy(pose).to(device=device, dtype=torch.float32)
    global_rot_placeholder = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(num_frames, 1, 1, 1)

    with torch.no_grad():
        mano_out = mano(betas=beta_torch, hand_pose=pose_torch, global_orient=global_rot_placeholder)
    verts = mano_out.vertices.detach().cpu().numpy()
    joints = mano_out.joints.detach().cpu().numpy()

    if is_left:
        verts[:, :, 0] *= -1
        joints[:, :, 0] *= -1

    verts_worldspace = (
        wrist_orientation @ (verts - joints[:, 0][:, None]).transpose(0, 2, 1)
    ).transpose(0, 2, 1) + wrist_worldspace
    return verts_worldspace.astype(np.float32)


def normalize_mesh_sets(mesh_sets: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    pts = np.concatenate(
        [arr.reshape(-1, 3) for sides in mesh_sets.values() for arr in sides.values()],
        axis=0,
    )
    center = pts.mean(axis=0)
    scale = float(np.percentile(np.linalg.norm(pts - center, axis=1), 95))
    scale = max(scale, 1e-3)
    return {
        label: {side: (arr - center) / scale for side, arr in sides.items()}
        for label, sides in mesh_sets.items()
    }


def resolve_mano_model_path(model_path: Path) -> Path:
    path = Path(model_path).expanduser()
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    repo_relative = repo_root / path
    return repo_relative if repo_relative.exists() else path


def validate_mano_model_path(model_path: Path) -> Path:
    path = resolve_mano_model_path(model_path)
    required = path if path.is_file() else path / "MANO_RIGHT.pkl"
    if not required.exists():
        raise FileNotFoundError(
            "MANO mesh videos need the official MANO_RIGHT.pkl file. "
            f"Put it at {path / 'MANO_RIGHT.pkl'} or pass --mano_model_path to the directory/file you already have."
        )
    return path


def load_mano_model(model_path: Path) -> tuple[Any, np.ndarray, str]:
    import torch
    from libs.models.mano_wrapper import MANO

    resolved_path = validate_mano_model_path(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano = MANO(model_path=str(resolved_path)).to(device).eval()
    return mano, np.asarray(mano.faces, dtype=np.int32), device


def normalize_motion_sets(motion_sets: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    all_points = []
    for sides in motion_sets.values():
        all_points.extend([sides["left"], sides["right"]])
    pts = np.concatenate([arr.reshape(-1, 3) for arr in all_points], axis=0)
    center = pts.mean(axis=0)
    scale = float(np.percentile(np.linalg.norm(pts - center, axis=1), 95))
    scale = max(scale, 1e-3)
    normalized = {}
    for label, sides in motion_sets.items():
        normalized[label] = {
            side: (arr - center) / scale
            for side, arr in sides.items()
        }
    return normalized


def project_motion_points(points: np.ndarray, origin: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    x0, y0 = origin
    width, height = size
    x = points[..., 0] - 0.45 * points[..., 2]
    y = -points[..., 1] - 0.22 * points[..., 2]
    px = x0 + width * (0.5 + 0.36 * x)
    py = y0 + height * (0.52 + 0.36 * y)
    return np.stack([px, py], axis=-1).astype(np.int32)


def draw_hand_skeleton(
    canvas: np.ndarray,
    joints: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    import cv2

    pts = project_motion_points(joints, origin, size)
    for chain in FINGER_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for idx, point in enumerate(pts):
        radius = 4 if idx == 0 else 3
        cv2.circle(canvas, tuple(point), radius, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (origin[0] + 12, origin[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_motion_panel(
    canvas: np.ndarray,
    motion_sets: dict[str, dict[str, np.ndarray]],
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (58, 58, 58), 1)
    cv2.putText(canvas, title, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    mid = x0 + width // 2
    cv2.line(canvas, (mid, y0), (mid, y0 + height), (48, 48, 48), 1)
    left_origin = (x0, y0)
    right_origin = (mid, y0)
    panel_size = (width // 2, height)
    for label, sides in motion_sets.items():
        color = colors.get(label.lower(), (220, 220, 220))
        draw_hand_skeleton(canvas, sides["left"][frame_idx], left_origin, panel_size, color, f"{label} left")
        draw_hand_skeleton(canvas, sides["right"][frame_idx], right_origin, panel_size, color, f"{label} right")


def draw_palm_trails(
    canvas: np.ndarray,
    motion_sets: dict[str, dict[str, np.ndarray]],
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    colors: dict[str, tuple[int, int, int]],
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (58, 58, 58), 1)
    cv2.putText(canvas, "Palm trajectory", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    for label, sides in motion_sets.items():
        color = colors.get(label.lower(), (220, 220, 220))
        for side in ("left", "right"):
            palms = sides[side][: frame_idx + 1, 0, :]
            pts = project_motion_points(palms, origin, size)
            if len(pts) > 1:
                cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
            if len(pts):
                cv2.circle(canvas, tuple(pts[-1]), 4, color, -1, cv2.LINE_AA)


def draw_mano_mesh(
    canvas: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    face_stride: int = 7,
) -> None:
    import cv2

    pts = project_motion_points(verts, origin, size)
    for tri in faces[::face_stride]:
        tri_pts = pts[np.asarray(tri, dtype=np.int32)]
        cv2.polylines(canvas, [tri_pts], True, color, 1, cv2.LINE_AA)
    sampled = pts[::20]
    for point in sampled:
        cv2.circle(canvas, tuple(point), 1, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (origin[0] + 12, origin[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_mano_mesh_panel(
    canvas: np.ndarray,
    mesh_sets: dict[str, dict[str, np.ndarray]],
    faces: np.ndarray,
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (58, 58, 58), 1)
    cv2.putText(canvas, title, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    mid = x0 + width // 2
    cv2.line(canvas, (mid, y0), (mid, y0 + height), (48, 48, 48), 1)
    panel_size = (width // 2, height)
    for label, sides in mesh_sets.items():
        color = colors.get(label.lower(), (220, 220, 220))
        draw_mano_mesh(canvas, sides["left"][frame_idx], faces, (x0, y0), panel_size, color, f"{label} left")
        draw_mano_mesh(canvas, sides["right"][frame_idx], faces, (mid, y0), panel_size, color, f"{label} right")


def draw_mesh_wrist_trails(
    canvas: np.ndarray,
    mesh_sets: dict[str, dict[str, np.ndarray]],
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    colors: dict[str, tuple[int, int, int]],
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (58, 58, 58), 1)
    cv2.putText(canvas, "Mesh center trajectory", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    for label, sides in mesh_sets.items():
        color = colors.get(label.lower(), (220, 220, 220))
        for side in ("left", "right"):
            centers = sides[side][: frame_idx + 1].mean(axis=1)
            pts = project_motion_points(centers, origin, size)
            if len(pts) > 1:
                cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
            if len(pts):
                cv2.circle(canvas, tuple(pts[-1]), 4, color, -1, cv2.LINE_AA)


def write_mano_motion_video(
    image: np.ndarray,
    instruction: str,
    mesh_sets: dict[str, dict[str, np.ndarray]],
    faces: np.ndarray,
    output_path: Path,
    fps: float = 6.0,
) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    input_panel = cv2.resize(image, (448, 336))
    if input_panel.shape[-1] == 3:
        input_panel = cv2.cvtColor(input_panel, cv2.COLOR_RGB2BGR)
    mesh_sets = normalize_mesh_sets(mesh_sets)
    num_frames = min(arr.shape[0] for sides in mesh_sets.values() for arr in sides.values())
    width, height = 1280, 720
    colors = {
        "gt": (245, 245, 245),
        "base": (80, 130, 255),
        "step500": (80, 220, 120),
        "prediction": (80, 220, 120),
        "trained": (80, 220, 120),
    }
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame_idx in range(num_frames):
            canvas = np.full((height, width, 3), 24, dtype=np.uint8)
            canvas[82:418, 36:484] = input_panel
            cv2.putText(canvas, "Predicted MANO Hand Mesh", (36, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"future step {frame_idx}/{num_frames - 1}", (36, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
            draw_wrapped_text(canvas, instruction, (36, 455), max_width=448, color=(225, 225, 225), scale=0.52)
            draw_legend(canvas, [(label, colors.get(label.lower(), (220, 220, 220))) for label in mesh_sets], origin=(36, 650))
            draw_mano_mesh_panel(
                canvas,
                mesh_sets,
                faces,
                frame_idx,
                origin=(520, 86),
                size=(720, 420),
                title="MANO mesh, normalized camera coordinates",
                colors=colors,
            )
            draw_mesh_wrist_trails(canvas, mesh_sets, frame_idx, origin=(520, 560), size=(720, 120), colors=colors)
            writer.write(canvas)
    finally:
        writer.release()
    return output_path.exists()


def project_vertices_to_image(vertices: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    z = vertices[:, 2]
    valid = z > 1e-4
    z_safe = np.where(valid, z, 1.0)
    u = intrinsics[0, 0] * vertices[:, 0] / z_safe + intrinsics[0, 2]
    v = intrinsics[1, 1] * vertices[:, 1] / z_safe + intrinsics[1, 2]
    return np.stack([u, v], axis=-1).astype(np.int32), valid


def draw_projected_mesh(
    frame_rgb: np.ndarray,
    left_vertices: np.ndarray,
    right_vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    label: str,
    face_stride: int = 3,
) -> np.ndarray:
    import cv2

    frame = np.asarray(frame_rgb).copy()
    height, width = frame.shape[:2]
    overlay = frame.copy()
    side_specs = [
        ("left", left_vertices, (40, 220, 255)),
        ("right", right_vertices, (80, 255, 120)),
    ]
    for side, vertices, color in side_specs:
        points, valid = project_vertices_to_image(vertices, intrinsics)
        margin = 120
        for tri in faces[::face_stride]:
            tri = np.asarray(tri, dtype=np.int32)
            if not np.all(valid[tri]):
                continue
            tri_points = points[tri]
            if (
                np.all(tri_points[:, 0] < -margin)
                or np.all(tri_points[:, 0] > width + margin)
                or np.all(tri_points[:, 1] < -margin)
                or np.all(tri_points[:, 1] > height + margin)
            ):
                continue
            cv2.polylines(overlay, [tri_points], True, color, 1, cv2.LINE_AA)

        visible = valid & (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        if np.any(visible):
            center = points[visible].mean(axis=0).astype(np.int32)
            cv2.circle(overlay, tuple(center), 6, color, -1, cv2.LINE_AA)
            cv2.putText(overlay, side, tuple(center + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
    cv2.rectangle(frame, (12, 12), (300, 52), (0, 0, 0), -1)
    cv2.putText(frame, label, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    return frame


def read_video_frames_at(video_path: Path, frame_indices: np.ndarray, fallback_frame: np.ndarray) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    if not cap.isOpened():
        return [np.asarray(fallback_frame).copy() for _ in frame_indices]
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            else:
                frames.append(np.asarray(fallback_frame).copy())
    finally:
        cap.release()
    return frames


def resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(np.asarray(frame), size, interpolation=cv2.INTER_AREA)


def transcode_h264_in_place(path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None or not path.exists():
        return
    tmp_path = path.with_name(f"{path.stem}.h264_tmp{path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()


def write_rgb_overlay_video(
    frames_rgb: list[np.ndarray],
    mesh_sets: dict[str, dict[str, np.ndarray]],
    faces: np.ndarray,
    intrinsics: np.ndarray,
    output_path: Path,
    label: str,
    fps: float = 8.0,
) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        sides = mesh_sets[label]
        num_frames = min(len(frames_rgb), sides["left"].shape[0], sides["right"].shape[0])
        for frame_idx in range(num_frames):
            overlay = draw_projected_mesh(
                frames_rgb[frame_idx],
                sides["left"][frame_idx],
                sides["right"][frame_idx],
                faces,
                intrinsics,
                label=label,
            )
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    transcode_h264_in_place(output_path)
    return output_path.exists()


def write_rgb_overlay_grid_video(
    frames_rgb: list[np.ndarray],
    mesh_sets: dict[str, dict[str, np.ndarray]],
    faces: np.ndarray,
    intrinsics: np.ndarray,
    output_path: Path,
    labels: list[str],
    instruction: str,
    fps: float = 8.0,
) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_size = (640, 360)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    num_frames = min(len(frames_rgb), *(mesh_sets[label]["left"].shape[0] for label in labels))
    try:
        for frame_idx in range(num_frames):
            raw = resize_rgb(frames_rgb[frame_idx], panel_size)
            cv2.rectangle(raw, (12, 12), (190, 52), (0, 0, 0), -1)
            cv2.putText(raw, "input RGB", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)

            panels = [raw]
            for label in labels:
                overlay = draw_projected_mesh(
                    frames_rgb[frame_idx],
                    mesh_sets[label]["left"][frame_idx],
                    mesh_sets[label]["right"][frame_idx],
                    faces,
                    intrinsics,
                    label=label,
                )
                panels.append(resize_rgb(overlay, panel_size))
            while len(panels) < 4:
                panels.append(np.zeros_like(raw))
            grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4])])
            draw_wrapped_text(grid, instruction, (18, 690), max_width=1220, color=(245, 245, 245), scale=0.55)
            writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    transcode_h264_in_place(output_path)
    return output_path.exists()


def write_hand_motion_video(
    image: np.ndarray,
    instruction: str,
    motion_sets: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    fps: float = 6.0,
) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    input_panel = cv2.resize(image, (448, 336))
    if input_panel.shape[-1] == 3:
        input_panel = cv2.cvtColor(input_panel, cv2.COLOR_RGB2BGR)
    motion_sets = normalize_motion_sets(motion_sets)
    num_frames = min(arr.shape[0] for sides in motion_sets.values() for arr in sides.values())
    width, height = 1280, 720
    colors = {
        "gt": (245, 245, 245),
        "base": (80, 130, 255),
        "step500": (80, 220, 120),
        "prediction": (80, 220, 120),
        "trained": (80, 220, 120),
    }
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame_idx in range(num_frames):
            canvas = np.full((height, width, 3), 24, dtype=np.uint8)
            canvas[82:418, 36:484] = input_panel
            cv2.putText(canvas, "Predicted Hand Motion", (36, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"future step {frame_idx}/{num_frames - 1}", (36, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
            draw_wrapped_text(canvas, instruction, (36, 455), max_width=448, color=(225, 225, 225), scale=0.52)
            draw_legend(canvas, [(label, colors.get(label.lower(), (220, 220, 220))) for label in motion_sets], origin=(36, 650))
            draw_motion_panel(canvas, motion_sets, frame_idx, origin=(520, 86), size=(720, 420), title="3D hand skeleton, normalized camera coordinates", colors=colors)
            draw_palm_trails(canvas, motion_sets, frame_idx, origin=(520, 560), size=(720, 120), colors=colors)
            writer.write(canvas)
    finally:
        writer.release()
    return output_path.exists()


def write_action_comparison_video(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray | dict[str, np.ndarray],
    action_mask: np.ndarray,
    output_path: Path,
    instruction: str = "",
    clip_metrics: dict[str, dict[str, float]] | None = None,
    fps: float = 6.0,
) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    image = np.asarray(image)
    if image.ndim != 3:
        return False
    target = np.asarray(target, dtype=np.float32)
    if isinstance(prediction, dict):
        predictions = {str(label): np.asarray(value, dtype=np.float32) for label, value in prediction.items()}
    else:
        predictions = {"prediction": np.asarray(prediction, dtype=np.float32)}
    action_mask = np.asarray(action_mask, dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_panel = cv2.resize(image, (448, 336))
    if input_panel.shape[-1] == 3:
        input_panel = cv2.cvtColor(input_panel, cv2.COLOR_RGB2BGR)
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    half = target.shape[-1] // 2
    component_dims = [0, 1, min(6, half - 1), half, half + 1, min(half + 6, target.shape[-1] - 1)]
    colors = {
        "target": (245, 245, 245),
        "base": (80, 130, 255),
        "before": (80, 130, 255),
        "trained": (80, 220, 120),
        "after": (80, 220, 120),
        "step500": (80, 220, 120),
        "prediction": (80, 220, 120),
    }
    try:
        for frame_idx in range(target.shape[0]):
            canvas = np.full((height, width, 3), 28, dtype=np.uint8)
            canvas[84:420, 36:484] = input_panel
            cv2.putText(canvas, "GigaHands Stage-1 Test Demo", (36, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2)
            cv2.putText(canvas, f"clip frame {frame_idx + 1}/{target.shape[0]}", (36, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1)
            draw_wrapped_text(canvas, instruction, (36, 455), max_width=448, color=(225, 225, 225), scale=0.52)
            draw_metric_cards(canvas, clip_metrics or {}, origin=(36, 570), width=448)

            legend_items = [("target", colors["target"])]
            for label in predictions:
                legend_items.append((label, colors.get(label.lower(), (80, 220, 120))))
            draw_legend(canvas, legend_items, origin=(520, 44))

            draw_norm_plot(
                canvas,
                target,
                predictions,
                action_mask,
                frame_idx,
                origin=(520, 88),
                size=(720, 220),
                title="Action magnitude, target vs prediction",
                colors=colors,
            )
            draw_error_plot(
                canvas,
                target,
                predictions,
                action_mask,
                frame_idx,
                origin=(520, 342),
                size=(720, 160),
                title="Per-frame masked squared error",
                colors=colors,
            )
            draw_component_plot(
                canvas,
                target,
                predictions,
                component_dims,
                frame_idx,
                origin=(520, 548),
                size=(720, 120),
                title="Selected normalized action components",
                colors=colors,
            )
            writer.write(canvas)
    finally:
        writer.release()
    return output_path.exists()


def draw_wrapped_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    color: tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    import cv2

    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = origin
    for line in lines[:4]:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(26 * scale) + 8


def draw_metric_cards(canvas: np.ndarray, metrics: dict[str, dict[str, float]], origin: tuple[int, int], width: int) -> None:
    import cv2

    x, y = origin
    cv2.putText(canvas, "Masked action MSE", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if not metrics:
        return
    card_w = max(120, width // max(len(metrics), 1) - 8)
    for idx, (label, values) in enumerate(metrics.items()):
        cx = x + idx * (card_w + 8)
        cv2.rectangle(canvas, (cx, y), (cx + card_w, y + 74), (48, 48, 48), -1)
        cv2.rectangle(canvas, (cx, y), (cx + card_w, y + 74), (92, 92, 92), 1)
        cv2.putText(canvas, label, (cx + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(canvas, f"{values.get('action_mse', 0.0):.3f}", (cx + 10, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)


def draw_legend(canvas: np.ndarray, legend_items: list[tuple[str, tuple[int, int, int]]], origin: tuple[int, int]) -> None:
    import cv2

    x, y = origin
    for label, color in legend_items:
        cv2.line(canvas, (x, y), (x + 26, y), color, 3)
        cv2.putText(canvas, label, (x + 34, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1)
        x += 150


def draw_norm_plot(
    canvas: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    action_mask: np.ndarray,
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    series: dict[str, np.ndarray] = {"target L": hand_norm_series(target, action_mask, left=True), "target R": hand_norm_series(target, action_mask, left=False)}
    for label, pred in predictions.items():
        series[f"{label} L"] = hand_norm_series(pred, action_mask, left=True)
        series[f"{label} R"] = hand_norm_series(pred, action_mask, left=False)
    draw_series_panel(canvas, series, frame_idx, origin, size, title, colors)


def draw_error_plot(
    canvas: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    action_mask: np.ndarray,
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    series = {}
    for label, pred in predictions.items():
        err = (pred - target) ** 2
        mask = action_mask if action_mask.shape == target.shape else np.repeat(action_mask, target.shape[-1] // action_mask.shape[-1], axis=-1)
        values = np.zeros(target.shape[0], dtype=np.float32)
        for idx in range(target.shape[0]):
            values[idx] = float(err[idx][mask[idx]].mean()) if np.any(mask[idx]) else 0.0
        series[label] = values
    draw_series_panel(canvas, series, frame_idx, origin, size, title, colors)


def draw_component_plot(
    canvas: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    dims: list[int],
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    series = {"target": target[:, dims].mean(axis=1)}
    for label, pred in predictions.items():
        series[label] = pred[:, dims].mean(axis=1)
    draw_series_panel(canvas, series, frame_idx, origin, size, title, colors)


def draw_series_panel(
    canvas: np.ndarray,
    series: dict[str, np.ndarray],
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (62, 62, 62), 1)
    cv2.putText(canvas, title, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)
    values = np.concatenate([np.asarray(value, dtype=np.float32) for value in series.values()])
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    usable = max(frame_idx + 1, 2)
    for grid_idx in range(1, 4):
        y = y0 + grid_idx * height // 4
        cv2.line(canvas, (x0, y), (x0 + width, y), (44, 44, 44), 1)
    for label, values_i in series.items():
        color = colors.get(label.lower().split()[0], (180, 180, 180))
        points = series_points(values_i[:usable], x0, y0, width, height, lo, hi)
        cv2.polylines(canvas, [points], False, color, 2)
    cursor_x = x0 + 8 + int((width - 16) * frame_idx / max(len(next(iter(series.values()))) - 1, 1))
    cv2.line(canvas, (cursor_x, y0), (cursor_x, y0 + height), (120, 120, 120), 1)


def hand_norm_series(actions: np.ndarray, action_mask: np.ndarray, left: bool) -> np.ndarray:
    half = actions.shape[-1] // 2
    hand_slice = slice(0, half) if left else slice(half, actions.shape[-1])
    values = np.linalg.norm(actions[:, hand_slice], axis=-1)
    left_mask, right_mask = hand_frame_masks(action_mask, actions.shape[-1])
    mask = left_mask if left else right_mask
    return np.where(mask, values, 0.0).astype(np.float32)


def series_points(values: np.ndarray, x0: int, y0: int, width: int, height: int, lo: float, hi: float) -> np.ndarray:
    xs = np.linspace(x0 + 8, x0 + width - 8, len(values))
    ys = y0 + height - 8 - ((values - lo) / (hi - lo)) * (height - 16)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_eval_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_on_cuda(value: Any, dtype: Any):
    import torch

    if torch.is_tensor(value):
        return value.detach().to(device="cuda", dtype=dtype)[None]
    return torch.tensor(value, dtype=dtype, device="cuda")[None]


def build_eval_dataset(args: argparse.Namespace, config: dict[str, Any]):
    from vitra.datasets.dataset import FrameDataset

    dataset_name = resolve_single_dataset_name(args.data_mix)
    dataset = FrameDataset(
        dataset_folder=str(args.dataset_root),
        dataset_name=dataset_name,
        action_future_window_size=config.get("fwd_pred_next_n", 16) - 1,
        augmentation=False,
        normalization=True,
        processor=None,
        load_images=True,
        **dataset_kwargs_from_config(config),
    )
    if getattr(dataset.episodic_dataset_core, "data_statistics", None) is not None:
        dataset.episodic_dataset_core.set_global_data_statistics(dataset.episodic_dataset_core.data_statistics)
    return dataset


def load_eval_model(config: dict[str, Any], checkpoint: str):
    from vitra.models.vla_builder import build_vla, load_vla_checkpoint

    model = build_vla(configs=config)
    if checkpoint != "none":
        checkpoint_path = Path(checkpoint)
        weights_path = checkpoint_path / "weights.pt" if checkpoint_path.is_dir() else checkpoint_path
        model = load_vla_checkpoint(model, str(weights_path))
    return model.eval().cuda()


def select_eval_indices(dataset, args: argparse.Namespace) -> list[int]:
    core = dataset.episodic_dataset_core
    if args.eval_sample_strategy == "sequential":
        return list(range(min(args.num_eval_clips, len(dataset))))

    index_frame_pair = np.asarray(core.index_frame_pair)
    selected: list[int] = []
    for episode_idx in range(len(core.index_to_episode_id)):
        rows = np.where(index_frame_pair[:, 0] == episode_idx)[0]
        if len(rows) == 0:
            continue
        if args.eval_sample_strategy == "first_per_episode":
            selected.append(int(rows[0]))
        elif args.eval_sample_strategy == "middle_per_episode":
            selected.append(int(rows[len(rows) // 2]))
        else:
            raise ValueError(f"Unknown eval sample strategy: {args.eval_sample_strategy}")
        if len(selected) >= args.num_eval_clips:
            break
    return selected


def collect_predictions(args: argparse.Namespace, model, dataset) -> dict[str, Any]:
    import torch

    set_eval_seed(args.seed)
    predictions: list[np.ndarray] = []
    unnormalized_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    unnormalized_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    data_ids: list[int] = []

    eval_indices = select_eval_indices(dataset, args)
    with torch.no_grad():
        for data_id in eval_indices:
            raw_sample = dataset.episodic_dataset_core.__getitem__(data_id)
            sample = dataset.episodic_dataset_core.transform_trajectory(raw_sample.copy(), normalization=True)
            current_state = tensor_on_cuda(sample["current_state"], torch.float32)
            current_state_mask = tensor_on_cuda(sample["current_state_mask"], torch.bool)
            action_mask = tensor_on_cuda(sample["action_mask"], torch.bool)
            fov = tensor_on_cuda(sample["fov"], torch.float32)
            pred = model.predict_action(
                image=sample["image_list"][-1],
                instruction=sample["instruction"],
                current_state=current_state,
                current_state_mask=current_state_mask,
                action_mask_torch=action_mask,
                fov=fov,
                sample_times=1,
                num_ddim_steps=args.num_ddim_steps,
                cfg_scale=args.cfg_scale,
            )[0]
            predictions.append(pred.astype(np.float32))
            unnormalized_predictions.append(
                unnormalize_padded_actions(dataset.episodic_dataset_core.gaussian_normalizer, pred)
            )
            targets.append(to_numpy(sample["action_list"], np.float32))
            unnormalized_targets.append(to_numpy(raw_sample["action_list"], np.float32))
            masks.append(to_numpy(sample["action_mask"], bool))
            states.append(to_numpy(raw_sample["current_state"], np.float32))
            data_ids.append(int(data_id))
            samples.append(
                {
                    "image": np.asarray(sample["image_list"][-1]),
                    "instruction": str(sample["instruction"]),
                    "intrinsics": to_numpy(sample["intrinsics"], np.float32),
                }
            )

    metrics = evaluate_predictions(predictions, targets, masks)
    return {
        "predictions": predictions,
        "unnormalized_predictions": unnormalized_predictions,
        "targets": targets,
        "unnormalized_targets": unnormalized_targets,
        "masks": masks,
        "states": states,
        "data_ids": data_ids,
        "samples": samples,
        "metrics": metrics,
    }


def evaluate_checkpoint(args: argparse.Namespace, config: dict[str, Any], dataset, checkpoint: str) -> dict[str, Any]:
    import gc
    import torch

    model = load_eval_model(config, checkpoint)
    try:
        return collect_predictions(args, model, dataset)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def metric_comparison(before: dict[str, float], after: dict[str, float]) -> dict[str, dict[str, float]]:
    delta = {}
    relative_improvement = {}
    for key, before_value in before.items():
        after_value = after.get(key)
        if not isinstance(before_value, (int, float)) or not isinstance(after_value, (int, float)):
            continue
        delta[key] = float(after_value - before_value)
        relative_improvement[key] = 0.0 if abs(before_value) < 1e-12 else float((before_value - after_value) / before_value)
    return {"delta_after_minus_before": delta, "relative_improvement": relative_improvement}


def write_demo_videos(
    output_dir: Path,
    result_by_label: dict[str, dict[str, Any]],
    primary_label: str,
) -> None:
    video_dir = output_dir / "videos"
    primary = result_by_label[primary_label]
    for idx, sample in enumerate(primary["samples"]):
        target = primary["targets"][idx]
        mask = primary["masks"][idx]
        predictions = {label: result["predictions"][idx] for label, result in result_by_label.items()}
        clip_metrics = {
            label: compute_action_metrics(
                result["predictions"][idx][None],
                target[None],
                mask[None],
            )
            for label, result in result_by_label.items()
        }
        write_action_comparison_video(
            image=sample["image"],
            target=target,
            prediction=predictions,
            action_mask=mask,
            output_path=video_dir / f"clip_{idx:04d}.mp4",
            instruction=sample["instruction"],
            clip_metrics=clip_metrics,
        )


def write_hand_motion_videos(
    output_dir: Path,
    result_by_label: dict[str, dict[str, Any]],
    primary_label: str,
) -> None:
    motion_dir = output_dir / "hand_motion_videos"
    primary = result_by_label[primary_label]
    for idx, sample in enumerate(primary["samples"]):
        state_left, state_right = split_state_122(primary["states"][idx])
        target = primary["unnormalized_targets"][idx]
        motion_sets: dict[str, dict[str, np.ndarray]] = {
            "gt": {
                "left": hand_traj_to_joints(recon_traj_from_actions(state_left, target[:, :51]), is_left=True),
                "right": hand_traj_to_joints(recon_traj_from_actions(state_right, target[:, 51:102]), is_left=False),
            }
        }
        for label, result in result_by_label.items():
            pred = result["unnormalized_predictions"][idx]
            motion_sets[label] = {
                "left": hand_traj_to_joints(recon_traj_from_actions(state_left, pred[:, :51]), is_left=True),
                "right": hand_traj_to_joints(recon_traj_from_actions(state_right, pred[:, 51:102]), is_left=False),
            }
        write_hand_motion_video(
            image=sample["image"],
            instruction=sample["instruction"],
            motion_sets=motion_sets,
            output_path=motion_dir / f"clip_{idx:04d}_hand_motion.mp4",
        )


def write_mano_motion_videos(
    output_dir: Path,
    result_by_label: dict[str, dict[str, Any]],
    primary_label: str,
    mano_model_path: Path,
) -> None:
    mano, faces, device = load_mano_model(mano_model_path)
    motion_dir = output_dir / "mano_motion_videos"
    primary = result_by_label[primary_label]
    for idx, sample in enumerate(primary["samples"]):
        (state_left, beta_left), (state_right, beta_right) = split_state_beta_122(primary["states"][idx])
        target = primary["unnormalized_targets"][idx]
        mesh_sets: dict[str, dict[str, np.ndarray]] = {
            "gt": {
                "left": mano_vertices_from_labels(
                    mano,
                    traj_to_mano_labels(recon_traj_from_actions(state_left, target[:, :51]), beta_left),
                    is_left=True,
                    device=device,
                ),
                "right": mano_vertices_from_labels(
                    mano,
                    traj_to_mano_labels(recon_traj_from_actions(state_right, target[:, 51:102]), beta_right),
                    is_left=False,
                    device=device,
                ),
            }
        }
        for label, result in result_by_label.items():
            pred = result["unnormalized_predictions"][idx]
            mesh_sets[label] = {
                "left": mano_vertices_from_labels(
                    mano,
                    traj_to_mano_labels(recon_traj_from_actions(state_left, pred[:, :51]), beta_left),
                    is_left=True,
                    device=device,
                ),
                "right": mano_vertices_from_labels(
                    mano,
                    traj_to_mano_labels(recon_traj_from_actions(state_right, pred[:, 51:102]), beta_right),
                    is_left=False,
                    device=device,
                ),
            }
        write_mano_motion_video(
            image=sample["image"],
            instruction=sample["instruction"],
            mesh_sets=mesh_sets,
            faces=faces,
            output_path=motion_dir / f"clip_{idx:04d}_mano_motion.mp4",
        )


def sample_video_context(dataset, idx: int, num_frames: int) -> dict[str, Any]:
    core = dataset.episodic_dataset_core
    data_id = int(idx)
    episode_index, frame_id = core.index_frame_pair[data_id]
    episode_id = str(core.index_to_episode_id[int(episode_index)])
    epi, _, _ = core._load_or_cache_episode(episode_id)
    dataset_name = episode_id.split("_")[0]
    video_path = Path(core._resolve_video_path(dataset_name, epi["video_name"]))
    max_frame = len(epi["video_decode_frame"]) - 1
    local_frame_ids = np.arange(int(frame_id), int(frame_id) + num_frames).clip(0, max_frame)
    video_frame_ids = epi["video_decode_frame"][local_frame_ids]
    return {
        "episode_id": episode_id,
        "frame_id": int(frame_id),
        "video_path": video_path,
        "video_frame_ids": video_frame_ids,
    }


def build_mano_mesh_sets(
    result_by_label: dict[str, dict[str, Any]],
    primary_label: str,
    idx: int,
    mano: Any,
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    primary = result_by_label[primary_label]
    (state_left, beta_left), (state_right, beta_right) = split_state_beta_122(primary["states"][idx])
    target = primary["unnormalized_targets"][idx]
    mesh_sets: dict[str, dict[str, np.ndarray]] = {
        "gt": {
            "left": mano_vertices_from_labels(
                mano,
                traj_to_mano_labels(recon_traj_from_actions(state_left, target[:, :51]), beta_left),
                is_left=True,
                device=device,
            ),
            "right": mano_vertices_from_labels(
                mano,
                traj_to_mano_labels(recon_traj_from_actions(state_right, target[:, 51:102]), beta_right),
                is_left=False,
                device=device,
            ),
        }
    }
    for label, result in result_by_label.items():
        pred = result["unnormalized_predictions"][idx]
        mesh_sets[label] = {
            "left": mano_vertices_from_labels(
                mano,
                traj_to_mano_labels(recon_traj_from_actions(state_left, pred[:, :51]), beta_left),
                is_left=True,
                device=device,
            ),
            "right": mano_vertices_from_labels(
                mano,
                traj_to_mano_labels(recon_traj_from_actions(state_right, pred[:, 51:102]), beta_right),
                is_left=False,
                device=device,
            ),
        }
    return mesh_sets


def write_rgb_overlay_videos(
    output_dir: Path,
    result_by_label: dict[str, dict[str, Any]],
    primary_label: str,
    dataset,
    mano_model_path: Path,
) -> None:
    mano, faces, device = load_mano_model(mano_model_path)
    overlay_dir = output_dir / "rgb_overlay_videos"
    primary = result_by_label[primary_label]
    labels = ["gt", *result_by_label.keys()]
    for idx, sample in enumerate(primary["samples"]):
        mesh_sets = build_mano_mesh_sets(result_by_label, primary_label, idx, mano, device)
        num_frames = min(arr.shape[0] for sides in mesh_sets.values() for arr in sides.values())
        context = sample_video_context(dataset, primary["data_ids"][idx], num_frames)
        frames = read_video_frames_at(context["video_path"], context["video_frame_ids"], sample["image"])
        intrinsics = sample["intrinsics"]

        write_rgb_overlay_grid_video(
            frames_rgb=frames,
            mesh_sets=mesh_sets,
            faces=faces,
            intrinsics=intrinsics,
            output_path=overlay_dir / f"clip_{idx:04d}_rgb_overlay_grid.mp4",
            labels=labels[:3],
            instruction=sample["instruction"],
        )
        for label in labels:
            write_rgb_overlay_video(
                frames_rgb=frames,
                mesh_sets=mesh_sets,
                faces=faces,
                intrinsics=intrinsics,
                output_path=overlay_dir / f"clip_{idx:04d}_{label}_rgb_overlay.mp4",
                label=label,
            )


def run_model_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.mano_motion_videos or args.rgb_overlay_videos:
        args.mano_model_path = validate_mano_model_path(args.mano_model_path)

    config = load_config(args.config)
    config["train_dataset"]["data_root_dir"] = str(args.dataset_root)
    config["train_dataset"]["data_mix"] = args.data_mix

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_eval_dataset(args, config)

    if args.baseline_checkpoint is not None:
        before = evaluate_checkpoint(args, config, dataset, args.baseline_checkpoint)
        after = evaluate_checkpoint(args, config, dataset, args.checkpoint)
        comparison = {
            "before_label": args.baseline_label,
            "after_label": args.label,
            "before_checkpoint": args.baseline_checkpoint,
            "after_checkpoint": args.checkpoint,
            "before": before["metrics"],
            "after": after["metrics"],
            **metric_comparison(before["metrics"], after["metrics"]),
        }
        (output_dir / f"metrics_{args.baseline_label}.json").write_text(json.dumps(before["metrics"], indent=2), encoding="utf-8")
        (output_dir / f"metrics_{args.label}.json").write_text(json.dumps(after["metrics"], indent=2), encoding="utf-8")
        (output_dir / "metrics_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        (output_dir / "metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        if not args.no_videos:
            write_demo_videos(output_dir, {args.baseline_label: before, args.label: after}, args.label)
        if args.hand_motion_videos:
            write_hand_motion_videos(output_dir, {args.baseline_label: before, args.label: after}, args.label)
        if args.mano_motion_videos:
            write_mano_motion_videos(
                output_dir,
                {args.baseline_label: before, args.label: after},
                args.label,
                args.mano_model_path,
            )
        if args.rgb_overlay_videos:
            write_rgb_overlay_videos(
                output_dir,
                {args.baseline_label: before, args.label: after},
                args.label,
                dataset,
                args.mano_model_path,
            )
        return comparison

    result = evaluate_checkpoint(args, config, dataset, args.checkpoint)
    metrics = result["metrics"]
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.no_videos:
        write_demo_videos(output_dir, {args.label: result}, args.label)
    if args.hand_motion_videos:
        write_hand_motion_videos(output_dir, {args.label: result}, args.label)
    if args.mano_motion_videos:
        write_mano_motion_videos(output_dir, {args.label: result}, args.label, args.mano_model_path)
    if args.rgb_overlay_videos:
        write_rgb_overlay_videos(output_dir, {args.label: result}, args.label, dataset, args.mano_model_path)
    return metrics


def resolve_single_dataset_name(data_mix: str) -> str:
    if data_mix == "gigahands_real_train":
        return "gigahands_real_train"
    if data_mix == "gigahands_real_test":
        return "gigahands_real_test"
    return data_mix


def dataset_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    train_dataset = config.get("train_dataset", {})
    return {
        "action_type": train_dataset.get("action_type", "angle"),
        "use_rel": train_dataset.get("use_rel", False),
        "rel_mode": train_dataset.get("rel_mode", "step"),
        "clip_len": train_dataset.get("clip_len", None),
        "state_mask_prob": train_dataset.get("state_mask_prob", 0.0),
        "target_image_height": train_dataset.get("target_image_height", 224),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--data_mix", default="gigahands_real_test")
    parser.add_argument("--checkpoint", default="none")
    parser.add_argument("--label", default="trained")
    parser.add_argument("--baseline_checkpoint", default=None)
    parser.add_argument("--baseline_label", default="base")
    parser.add_argument("--num_eval_clips", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_sample_strategy",
        choices=["sequential", "first_per_episode", "middle_per_episode"],
        default="sequential",
    )
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument("--hand_motion_videos", action="store_true")
    parser.add_argument("--mano_motion_videos", action="store_true")
    parser.add_argument("--rgb_overlay_videos", action="store_true")
    parser.add_argument("--mano_model_path", type=Path, default=Path("weights/mano"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metrics = run_model_evaluation(args)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
