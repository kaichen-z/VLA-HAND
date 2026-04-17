"""Evaluate VITRA Stage-1 predictions on a converted GigaHands split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_action_metrics(prediction: np.ndarray, target: np.ndarray, action_masks: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 3 or prediction.shape[-1] % 2 != 0:
        raise ValueError("prediction and target must be shaped [N, T, D] with even D")
    if action_masks.shape[:2] != prediction.shape[:2] or action_masks.shape[-1] != 2:
        raise ValueError("action_masks must be shaped [N, T, 2]")

    half = prediction.shape[-1] // 2
    left_err = (prediction[..., :half] - target[..., :half]) ** 2
    right_err = (prediction[..., half:] - target[..., half:]) ** 2
    left_mask = action_masks[..., 0]
    right_mask = action_masks[..., 1]
    dual_mask = left_mask & right_mask

    left_mse = masked_mean(left_err, left_mask)
    right_mse = masked_mean(right_err, right_mask)
    dual_err = np.concatenate([left_err, right_err], axis=-1)
    dual_mse = masked_mean(dual_err, dual_mask)
    any_mask = left_mask | right_mask
    all_err = np.concatenate([left_err, right_err], axis=-1)

    return {
        "action_mse": masked_mean(all_err, any_mask),
        "left_action_mse": left_mse,
        "right_action_mse": right_mse,
        "dual_hand_action_mse": dual_mse,
        "valid_frame_count": int(any_mask.sum()),
        "bimanual_frame_count": int(dual_mask.sum()),
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


def write_action_comparison_video(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    action_mask: np.ndarray,
    output_path: Path,
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
    prediction = np.asarray(prediction, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_panel = cv2.resize(image, (320, 240))
    if input_panel.shape[-1] == 3:
        input_panel = cv2.cvtColor(input_panel, cv2.COLOR_RGB2BGR)
    width, height = 960, 360
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    half = target.shape[-1] // 2
    dims = [
        ("left dx", 0, (255, 80, 80)),
        ("left dy", 1, (255, 140, 80)),
        ("right dx", half, (80, 160, 255)),
        ("right dy", half + 1, (80, 220, 255)),
    ]
    try:
        for frame_idx in range(target.shape[0]):
            canvas = np.full((height, width, 3), 245, dtype=np.uint8)
            canvas[:240, :320] = input_panel
            cv2.putText(canvas, "Input RGB", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
            cv2.putText(canvas, "Solid=GT  Dashed=Prediction", (360, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
            cv2.putText(canvas, f"frame {frame_idx + 1}/{target.shape[0]}", (360, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
            draw_action_plot(canvas, target, prediction, action_mask, dims, frame_idx, origin=(360, 85), size=(560, 230))
            writer.write(canvas)
    finally:
        writer.release()
    return output_path.exists()


def draw_action_plot(
    canvas: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    action_mask: np.ndarray,
    dims: list[tuple[str, int, tuple[int, int, int]]],
    frame_idx: int,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (210, 210, 210), 1)
    values = np.concatenate([target[:, [dim for _, dim, _ in dims]], prediction[:, [dim for _, dim, _ in dims]]], axis=1)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    usable = max(frame_idx + 1, 2)
    for label, dim, color in dims:
        gt_points = series_points(target[:usable, dim], x0, y0, width, height, lo, hi)
        pred_points = series_points(prediction[:usable, dim], x0, y0, width, height, lo, hi)
        cv2.polylines(canvas, [gt_points], False, color, 2)
        for start in range(0, len(pred_points) - 1, 2):
            cv2.line(canvas, tuple(pred_points[start]), tuple(pred_points[start + 1]), color, 1)
        y = y0 + 20 + 22 * dims.index((label, dim, color))
        cv2.putText(canvas, label, (x0 + width + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    active = "L" if action_mask[frame_idx, 0] else "-"
    active += "R" if action_mask[frame_idx, 1] else "-"
    cv2.putText(canvas, f"active hands: {active}", (x0, y0 + height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)


def series_points(values: np.ndarray, x0: int, y0: int, width: int, height: int, lo: float, hi: float) -> np.ndarray:
    xs = np.linspace(x0 + 8, x0 + width - 8, len(values))
    ys = y0 + height - 8 - ((values - lo) / (hi - lo)) * (height - 16)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_model_evaluation(args: argparse.Namespace) -> dict[str, float]:
    import torch

    from vitra.datasets.dataset import FrameDataset
    from vitra.models.vla_builder import build_vla, load_vla_checkpoint

    config = load_config(args.config)
    config["train_dataset"]["data_root_dir"] = str(args.dataset_root)
    config["train_dataset"]["data_mix"] = args.data_mix

    model = build_vla(configs=config)
    if args.checkpoint != "none":
        checkpoint_path = Path(args.checkpoint)
        weights_path = checkpoint_path / "weights.pt" if checkpoint_path.is_dir() else checkpoint_path
        model = load_vla_checkpoint(model, str(weights_path))
    model = model.eval().cuda()

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

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = min(args.num_eval_clips, len(dataset))
    with torch.no_grad():
        for idx in range(limit):
            sample = dataset.episodic_dataset_core.__getitem__(idx)
            sample = dataset.episodic_dataset_core.transform_trajectory(sample, normalization=True)
            current_state = torch.tensor(sample["current_state"], dtype=torch.float32, device="cuda")[None]
            current_state_mask = torch.tensor(sample["current_state_mask"], dtype=torch.bool, device="cuda")[None]
            action_mask = torch.tensor(sample["action_mask"], dtype=torch.bool, device="cuda")[None]
            fov = torch.tensor(sample["fov"], dtype=torch.float32, device="cuda")[None]
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
            targets.append(np.asarray(sample["action_list"], dtype=np.float32))
            masks.append(np.asarray(sample["action_mask"], dtype=bool))
            if not args.no_videos:
                write_action_comparison_video(
                    image=sample["image_list"][-1],
                    target=targets[-1],
                    prediction=predictions[-1],
                    action_mask=masks[-1],
                    output_path=video_dir / f"clip_{idx:04d}.mp4",
                )

    metrics = evaluate_predictions(predictions, targets, masks)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
    parser.add_argument("--num_eval_clips", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--no_videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_model_evaluation(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
