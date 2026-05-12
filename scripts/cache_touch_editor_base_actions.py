#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.datasets.dataset import FrameDataset  # noqa: E402
from vitra.models.vla_builder import build_vla, load_vla_checkpoint  # noqa: E402
from vitra.touch_editor.alignment import clipped_future_window_indices  # noqa: E402
from vitra.touch_editor.cache_utils import build_future_mask, chunk_phase  # noqa: E402


DEFAULT_LOCAL_CONFIG = "vitra/configs/human_pretrain_gigahands_real_full_keypoints_vitra3b_linked.json"
DEFAULT_FALLBACK_CONFIG = "vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json"


def tensor_on_cuda(value: Any, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device="cuda", dtype=dtype).unsqueeze(0)


def resolve_checkpoint_and_config(
    checkpoint: str | Path,
    config: str | Path | None = None,
    *,
    list_repo_files_fn: Callable | None = None,
    hf_hub_download_fn: Callable | None = None,
) -> tuple[Path, Path, str | None]:
    checkpoint_path = Path(str(checkpoint))
    if checkpoint_path.exists():
        weights_path = checkpoint_path / "weights.pt" if checkpoint_path.is_dir() else checkpoint_path
        config_path = Path(config) if config is not None else Path(DEFAULT_LOCAL_CONFIG)
        return weights_path, config_path, None

    checkpoint_str = str(checkpoint)
    if "/" not in checkpoint_str:
        weights_path = checkpoint_path / "weights.pt" if checkpoint_path.suffix == "" else checkpoint_path
        config_path = Path(config) if config is not None else Path(DEFAULT_LOCAL_CONFIG)
        return weights_path, config_path, None

    if list_repo_files_fn is None or hf_hub_download_fn is None:
        from huggingface_hub import hf_hub_download, list_repo_files

        list_repo_files_fn = list_repo_files
        hf_hub_download_fn = hf_hub_download

    files = list_repo_files_fn(checkpoint_str)
    weight_candidates = [name for name in files if name.endswith("weights.pt")]
    if not weight_candidates:
        weight_candidates = [name for name in files if name.endswith(".pt")]
    if not weight_candidates:
        raise ValueError(f"No .pt or weights.pt file found in Hugging Face repo {checkpoint_str}")

    if config is None:
        config_candidates = [name for name in files if name.startswith("config/") and name.endswith(".json")]
        if not config_candidates:
            config_candidates = [name for name in files if name.endswith(".json")]
        if not config_candidates:
            raise ValueError(f"No config JSON found in Hugging Face repo {checkpoint_str}")
        config_file = sorted(config_candidates)[0]
        config_path = Path(hf_hub_download_fn(repo_id=checkpoint_str, filename=config_file))
    else:
        config_path = Path(config)

    weight_file = sorted(weight_candidates)[0]
    weights_path = Path(hf_hub_download_fn(repo_id=checkpoint_str, filename=weight_file))
    return weights_path, config_path, checkpoint_str


def build_dataset(
    config: dict[str, Any],
    dataset_root: Path,
    data_mix: str,
    *,
    statistics_dataset_name: str | None = None,
) -> FrameDataset:
    train_cfg = config.get("train_dataset", {})
    dataset = FrameDataset(
        dataset_folder=str(dataset_root),
        dataset_name=data_mix,
        action_future_window_size=config.get("fwd_pred_next_n", 16) - 1,
        augmentation=False,
        normalization=True,
        processor=None,
        load_images=True,
        action_type=train_cfg.get("action_type", "keypoints"),
        use_rel=train_cfg.get("use_rel", False),
        clip_len=train_cfg.get("clip_len"),
        state_mask_prob=0.0,
        statistics_dataset_name=statistics_dataset_name or train_cfg.get("statistics_dataset_name", data_mix),
    )
    core = dataset.episodic_dataset_core
    if not hasattr(core, "gaussian_normalizer") and getattr(core, "data_statistics", None) is not None:
        core.set_global_data_statistics(core.data_statistics)
    return dataset


def load_frozen_vitra(config: dict[str, Any], weights_path: str | Path):
    model = build_vla(config)
    model = load_vla_checkpoint(model, str(weights_path))
    model = model.cuda().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def choose_edit_start(chunk_len: int, args: argparse.Namespace, rng: random.Random) -> int:
    if getattr(args, "random_edit_start", False):
        return rng.randint(1, max(1, int(chunk_len) - 1))
    return max(0, min(int(getattr(args, "edit_start_idx", 0)), int(chunk_len)))


def select_data_ids(
    dataset_len: int,
    start_index: int = 0,
    max_samples: int | None = None,
    sample_mode: str = "sequential",
    seed: int = 42,
) -> list[int]:
    start = max(0, int(start_index))
    if start >= int(dataset_len):
        return []
    candidates = list(range(start, int(dataset_len)))
    if max_samples is not None:
        max_samples = max(0, int(max_samples))
    if sample_mode == "sequential":
        stop = len(candidates) if max_samples is None else min(len(candidates), max_samples)
        return candidates[:stop]
    if sample_mode == "random":
        rng = random.Random(seed)
        if max_samples is None or max_samples >= len(candidates):
            rng.shuffle(candidates)
            return candidates
        return rng.sample(candidates, max_samples)
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def get_episode_and_frame(dataset: FrameDataset, data_id: int):
    core = dataset.episodic_dataset_core
    if getattr(core, "data_index", None) is not None:
        episode_id, frame_id = core.data_index[int(data_id)]
        episode = core.episodic_set[episode_id]
        return episode, int(frame_id), str(episode_id)
    if getattr(core, "index_frame_pair", None) is not None and getattr(core, "index_to_episode_id", None) is not None:
        episode_index, frame_id = core.index_frame_pair[int(data_id)]
        episode_id = str(core.index_to_episode_id[int(episode_index)])
        if hasattr(core, "_load_or_cache_episode"):
            episode, *_ = core._load_or_cache_episode(episode_id)
        else:
            episode = np.load(Path(core.label_folder) / f"{episode_id}.npy", allow_pickle=True).item()
        return episode, int(frame_id), episode_id
    raw_sample = core.__getitem__(int(data_id))
    episode = raw_sample.get("episode") or raw_sample.get("episodic_data") or raw_sample
    frame_id = int(raw_sample.get("frame_id", raw_sample.get("frame_idx", 0)))
    episode_id = str(raw_sample.get("episode_id", int(data_id)))
    return episode, frame_id, episode_id


def _window_from_array(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(array)[indices]


def _touch_window_from_episode(
    episode: dict[str, Any],
    frame_indices: np.ndarray,
    *,
    touch_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chunk_len = len(frame_indices)
    touch = episode.get("opentouch")
    if touch is None:
        if touch_mode != "zeros":
            raise KeyError("Episode has no opentouch payload; pass touch_mode zeros for non-touch datasets")
        return (
            np.zeros((chunk_len, 2, 16, 16), dtype=np.float32),
            np.zeros((chunk_len, 2), dtype=bool),
            np.full((chunk_len,), -1, dtype=np.int64),
            np.full((chunk_len,), np.nan, dtype=np.float64),
            np.zeros((chunk_len,), dtype=bool),
            np.full((chunk_len,), np.nan, dtype=np.float64),
        )

    pressure = _window_from_array(touch["touch_pressure"], frame_indices).astype(np.float32)
    mask = _window_from_array(touch["touch_mask"], frame_indices).astype(bool)
    aligned_indices = _window_from_array(
        touch.get("touch_aligned_indices", np.full(len(touch["touch_pressure"]), -1, dtype=np.int64)),
        frame_indices,
    ).astype(np.int64)
    aligned_timestamps = _window_from_array(
        touch.get("touch_aligned_timestamps", np.full(len(touch["touch_pressure"]), np.nan, dtype=np.float64)),
        frame_indices,
    ).astype(np.float64)
    alignment_valid = _window_from_array(
        touch.get("touch_alignment_valid", np.ones(len(touch["touch_pressure"]), dtype=bool)),
        frame_indices,
    ).astype(bool)
    video_timestamps = _window_from_array(
        touch.get("video_timestamps", np.full(len(touch["touch_pressure"]), np.nan, dtype=np.float64)),
        frame_indices,
    ).astype(np.float64)
    return pressure, mask, aligned_indices, aligned_timestamps, alignment_valid, video_timestamps


def observed_touch_stats(
    touch_pressure: np.ndarray,
    touch_mask: np.ndarray,
    edit_start_idx: int,
) -> tuple[int, float, float]:
    observed_len = max(1, min(int(edit_start_idx) + 1, int(touch_pressure.shape[0])))
    observed_pressure = np.asarray(touch_pressure[:observed_len], dtype=np.float32)
    observed_mask = np.asarray(touch_mask[:observed_len], dtype=bool)
    valid = observed_mask[..., None, None]
    valid_count = int(valid.sum()) * int(observed_pressure.shape[-1]) * int(observed_pressure.shape[-2])
    if valid_count <= 0:
        return observed_len, 0.0, 0.0
    contact_score = float((np.abs(observed_pressure) * valid).sum() / valid_count)

    endpoint_mask = observed_mask[0] & observed_mask[observed_len - 1]
    endpoint_valid = endpoint_mask[None, :, None, None]
    endpoint_count = int(endpoint_mask.sum()) * int(observed_pressure.shape[-1]) * int(observed_pressure.shape[-2])
    if endpoint_count <= 0:
        contact_delta = 0.0
    else:
        contact_delta = float(
            (np.abs(observed_pressure[observed_len - 1] - observed_pressure[0]) * endpoint_valid).sum()
            / endpoint_count
        )
    return observed_len, contact_score, contact_delta


def build_cache_record(
    *,
    a_base: np.ndarray,
    sample: dict[str, Any],
    episode: dict[str, Any],
    frame_id: int,
    edit_start_idx: int,
    touch_mode: str = "require",
) -> dict[str, np.ndarray]:
    a_base = np.asarray(a_base, dtype=np.float32)
    a_target = np.asarray(sample["action_list"], dtype=np.float32)
    action_mask = np.asarray(sample["action_mask"], dtype=bool)
    if a_base.shape != a_target.shape:
        raise ValueError(f"a_base/a_target shape mismatch: {a_base.shape} vs {a_target.shape}")
    if action_mask.shape != a_target.shape:
        raise ValueError(f"action_mask/a_target shape mismatch: {action_mask.shape} vs {a_target.shape}")

    chunk_len = int(a_target.shape[0])
    episode_len = len(episode.get("video_decode_frame", episode.get("extrinsics", np.arange(frame_id + chunk_len))))
    action_frame_indices, oob = clipped_future_window_indices(frame_id, chunk_len, episode_len)
    touch_pressure, touch_mask, touch_indices, touch_timestamps, touch_valid, action_timestamps = _touch_window_from_episode(
        episode,
        action_frame_indices,
        touch_mode=touch_mode,
    )
    if oob.any():
        touch_mask[oob] = False
        touch_valid[oob] = False

    future_mask = build_future_mask(action_mask, edit_start_idx)
    observed_touch_len, observed_touch_contact_score, observed_touch_contact_delta = observed_touch_stats(
        touch_pressure,
        touch_mask,
        edit_start_idx,
    )
    record = {
        "a_base": a_base,
        "a_target": a_target,
        "residual_target": (a_target - a_base).astype(np.float32),
        "action_mask": action_mask,
        "current_state": np.asarray(sample["current_state"], dtype=np.float32),
        "current_state_mask": np.asarray(sample["current_state_mask"], dtype=bool),
        "touch_pressure": touch_pressure,
        "touch_mask": touch_mask,
        "future_mask": future_mask,
        "edit_start_idx": np.asarray(int(edit_start_idx), dtype=np.int64),
        "chunk_phase": chunk_phase(chunk_len),
        "action_frame_indices": action_frame_indices.astype(np.int64),
        "action_timestamps": action_timestamps.astype(np.float64),
        "touch_aligned_indices": touch_indices.astype(np.int64),
        "touch_aligned_timestamps": touch_timestamps.astype(np.float64),
        "touch_alignment_valid": touch_valid.astype(bool),
        "observed_touch_len": np.asarray(observed_touch_len, dtype=np.int64),
        "observed_touch_contact_score": np.asarray(observed_touch_contact_score, dtype=np.float32),
        "observed_touch_contact_delta": np.asarray(observed_touch_contact_delta, dtype=np.float32),
        "target_source": np.asarray("opentouch_derived"),
        "target_dataset": np.asarray("opentouch"),
        "touch_source": np.asarray("opentouch" if touch_mode != "zeros" else "zeros"),
    }
    return record


def save_record(path: Path, record: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **record)
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen VITRA base actions for touch-editor training.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--data_mix", default="opentouch_keypoint_train")
    parser.add_argument("--statistics_dataset_name", default=None)
    parser.add_argument("--cache_root", type=Path, default=Path("runs/touch_editor_cache"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--sample_mode", choices=("sequential", "random"), default="sequential")
    parser.add_argument("--edit_start_idx", type=int, default=10)
    parser.add_argument("--random_edit_start", action="store_true")
    parser.add_argument("--touch_mode", choices=("require", "zeros"), default="require")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from vitra.utils.config_utils import load_config

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Caching frozen VITRA actions requires CUDA because predict_action uses CUDA internally.")
    rng = random.Random(args.seed)
    weights_path, config_path, hf_repo_id = resolve_checkpoint_and_config(args.checkpoint, args.config)
    config = load_config(str(config_path))
    dataset = build_dataset(
        config,
        args.dataset_root,
        args.data_mix,
        statistics_dataset_name=args.statistics_dataset_name,
    )
    model = load_frozen_vitra(config, weights_path)

    data_ids = select_data_ids(
        dataset_len=len(dataset),
        start_index=args.start_index,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    if not data_ids:
        raise ValueError(
            f"Empty cache selection: start_index={args.start_index}, max_samples={args.max_samples}, "
            f"sample_mode={args.sample_mode}, dataset_len={len(dataset)}"
        )
    args.cache_root.mkdir(parents=True, exist_ok=True)
    np.save(args.cache_root / "selected_data_ids.npy", np.asarray(data_ids, dtype=np.int64))

    written = 0
    with torch.no_grad():
        for data_id in tqdm(data_ids, desc="caching touch-editor base actions"):
            raw_sample = dataset.episodic_dataset_core.__getitem__(data_id)
            sample = dataset.episodic_dataset_core.transform_trajectory(raw_sample.copy(), normalization=True)
            current_state = tensor_on_cuda(sample["current_state"], torch.float32)
            current_state_mask = tensor_on_cuda(sample["current_state_mask"], torch.bool)
            action_mask = tensor_on_cuda(sample["action_mask"], torch.bool)
            fov = tensor_on_cuda(sample["fov"], torch.float32)
            a_base = model.predict_action(
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
            episode, frame_id, episode_id = get_episode_and_frame(dataset, data_id)
            edit_start_idx = choose_edit_start(len(sample["action_list"]), args, rng)
            record = build_cache_record(
                a_base=a_base,
                sample=sample,
                episode=episode,
                frame_id=frame_id,
                edit_start_idx=edit_start_idx,
                touch_mode=args.touch_mode,
            )
            save_record(
                args.cache_root / f"sample_{data_id:08d}.npz",
                record,
                {
                    "data_id": int(data_id),
                    "episode_id": episode_id,
                    "frame_id": int(frame_id),
                    "edit_start_idx": int(edit_start_idx),
                    "checkpoint": str(args.checkpoint),
                    "weights_path": str(weights_path),
                    "config_path": str(config_path),
                    "hf_repo_id": hf_repo_id,
                    "touch_mode": args.touch_mode,
                },
            )
            written += 1

    summary = {
        "cache_root": str(args.cache_root),
        "num_samples": int(written),
        "start_index": int(max(0, int(args.start_index))),
        "stop_index": int(max(data_ids) + 1),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "selected_data_ids_path": str(args.cache_root / "selected_data_ids.npy"),
        "dataset_root": str(args.dataset_root),
        "data_mix": args.data_mix,
        "checkpoint": str(args.checkpoint),
        "weights_path": str(weights_path),
        "config_path": str(config_path),
        "hf_repo_id": hf_repo_id,
        "target_source": "opentouch_derived",
        "touch_source": "opentouch" if args.touch_mode != "zeros" else "zeros",
    }
    (args.cache_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
