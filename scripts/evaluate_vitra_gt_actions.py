#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_touch_editor_base_actions import (
    DEFAULT_LOCAL_CONFIG,
    build_dataset,
    load_frozen_vitra,
    resolve_checkpoint_and_config,
    tensor_on_cuda,
)
from vitra.utils.config_utils import load_config


DEFAULT_DATASET_ROOT = Path("datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked")
DEFAULT_CHECKPOINT = "LeoJiangOR/vitra-gigahands-keypoints-step140000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen VITRA predictions against official VITRA/GigaHands ground-truth action_list targets."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="VITRA checkpoint path/dir or Hugging Face repo id.")
    parser.add_argument("--config", default=None, help=f"Config path. Defaults to {DEFAULT_LOCAL_CONFIG} for local checkpoints.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--data_mix", default="gigahands_real_test")
    parser.add_argument("--statistics_dataset_name", default="gigahands_real_train")
    parser.add_argument("--output_path", type=Path, default=Path("runs/vitra_gt_eval_gigahands_test/metrics.json"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sample_times", type=int, default=1)
    return parser.parse_args()


def to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr.astype(dtype) if dtype is not None else arr


def masked_sse_and_count(diff: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if diff.shape != mask.shape:
        raise ValueError(f"diff/mask shape mismatch: {diff.shape} vs {mask.shape}")
    return float((diff * diff * mask).sum()), float(mask.sum())


def split_hand_metrics(diff: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if diff.shape != mask.shape:
        raise ValueError(f"diff/mask shape mismatch: {diff.shape} vs {mask.shape}")
    if diff.shape[-1] % 2 != 0:
        return {}
    half = diff.shape[-1] // 2
    left_sse, left_count = masked_sse_and_count(diff[..., :half], mask[..., :half])
    right_sse, right_count = masked_sse_and_count(diff[..., half:], mask[..., half:])
    return {
        "left_action_mse": left_sse / max(left_count, 1.0),
        "right_action_mse": right_sse / max(right_count, 1.0),
        "left_valid_scalar_count": int(left_count),
        "right_valid_scalar_count": int(right_count),
    }


def per_step_sse_and_count(diff: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if diff.shape != mask.shape or diff.ndim != 2:
        raise ValueError(f"Expected diff/mask shaped [T,D], got {diff.shape} and {mask.shape}")
    return (diff * diff * mask).sum(axis=-1).astype(np.float64), mask.sum(axis=-1).astype(np.float64)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VITRA predict_action currently requires CUDA for this evaluator.")

    weights_path, config_path, hf_repo_id = resolve_checkpoint_and_config(args.checkpoint, args.config)
    config = load_config(str(config_path))
    config.setdefault("train_dataset", {})
    config["train_dataset"]["data_root_dir"] = str(args.dataset_root)
    config["train_dataset"]["data_mix"] = args.data_mix
    config["train_dataset"]["action_type"] = "keypoints"
    config["train_dataset"]["statistics_dataset_name"] = args.statistics_dataset_name

    dataset = build_dataset(config, args.dataset_root, args.data_mix)
    model = load_frozen_vitra(config, weights_path)

    dataset_len = len(dataset)
    start = max(0, int(args.start_index))
    stop = dataset_len if args.max_samples is None else min(dataset_len, start + int(args.max_samples))
    if start >= stop:
        raise ValueError(f"Empty evaluation range: start_index={start}, stop={stop}, dataset_len={dataset_len}")

    total_sse = 0.0
    total_count = 0.0
    hand_sse = {"left": 0.0, "right": 0.0}
    hand_count = {"left": 0.0, "right": 0.0}
    per_step_sse: np.ndarray | None = None
    per_step_count: np.ndarray | None = None
    action_shape: list[int] | None = None

    with torch.no_grad():
        for data_id in tqdm(range(start, stop), desc="evaluating VITRA GT actions"):
            raw_sample = dataset.episodic_dataset_core.__getitem__(data_id)
            sample = dataset.episodic_dataset_core.transform_trajectory(raw_sample.copy(), normalization=True)
            current_state = tensor_on_cuda(sample["current_state"], torch.float32)
            current_state_mask = tensor_on_cuda(sample["current_state_mask"], torch.bool)
            action_mask = tensor_on_cuda(sample["action_mask"], torch.bool)
            fov = tensor_on_cuda(sample["fov"], torch.float32)
            prediction = model.predict_action(
                image=sample["image_list"][-1],
                instruction=sample["instruction"],
                current_state=current_state,
                current_state_mask=current_state_mask,
                action_mask_torch=action_mask,
                fov=fov,
                sample_times=args.sample_times,
                num_ddim_steps=args.num_ddim_steps,
                cfg_scale=args.cfg_scale,
            )[0]

            target = to_numpy(sample["action_list"], np.float32)
            mask = to_numpy(sample["action_mask"], bool)
            pred = to_numpy(prediction, np.float32)
            if pred.shape != target.shape:
                raise ValueError(f"Prediction/target shape mismatch at data_id={data_id}: {pred.shape} vs {target.shape}")
            action_shape = list(target.shape)
            diff = pred - target

            sse, count = masked_sse_and_count(diff, mask)
            total_sse += sse
            total_count += count

            if target.shape[-1] % 2 == 0:
                half = target.shape[-1] // 2
                left_sse, left_count = masked_sse_and_count(diff[..., :half], mask[..., :half])
                right_sse, right_count = masked_sse_and_count(diff[..., half:], mask[..., half:])
                hand_sse["left"] += left_sse
                hand_count["left"] += left_count
                hand_sse["right"] += right_sse
                hand_count["right"] += right_count

            step_sse, step_count = per_step_sse_and_count(diff, mask)
            if per_step_sse is None:
                per_step_sse = np.zeros_like(step_sse, dtype=np.float64)
                per_step_count = np.zeros_like(step_count, dtype=np.float64)
            per_step_sse += step_sse
            per_step_count += step_count

    metrics: dict[str, Any] = {
        "target_definition": "official VITRA/GigaHands normalized keypoint action_list",
        "prediction_definition": "frozen VITRA predict_action output",
        "action_mse": total_sse / max(total_count, 1.0),
        "valid_scalar_count": int(total_count),
        "num_samples": int(stop - start),
        "dataset_len": int(dataset_len),
        "start_index": int(start),
        "dataset_root": str(args.dataset_root),
        "data_mix": args.data_mix,
        "statistics_dataset_name": args.statistics_dataset_name,
        "checkpoint": str(args.checkpoint),
        "weights_path": str(weights_path),
        "config_path": str(config_path),
        "hf_repo_id": hf_repo_id,
        "num_ddim_steps": args.num_ddim_steps,
        "cfg_scale": args.cfg_scale,
        "sample_times": args.sample_times,
        "action_shape": action_shape,
    }
    if action_shape is not None and action_shape[-1] % 2 == 0:
        metrics.update(
            {
                "left_action_mse": hand_sse["left"] / max(hand_count["left"], 1.0),
                "right_action_mse": hand_sse["right"] / max(hand_count["right"], 1.0),
                "left_valid_scalar_count": int(hand_count["left"]),
                "right_valid_scalar_count": int(hand_count["right"]),
            }
        )
    if per_step_sse is not None and per_step_count is not None:
        metrics["per_step_action_mse"] = (per_step_sse / np.maximum(per_step_count, 1.0)).tolist()
        metrics["per_step_valid_scalar_count"] = per_step_count.astype(np.int64).tolist()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
