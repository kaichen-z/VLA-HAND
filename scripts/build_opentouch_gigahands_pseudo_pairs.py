#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_touch_editor_base_actions import (  # noqa: E402
    build_cache_record,
    choose_edit_start,
    get_episode_and_frame,
    load_frozen_vitra,
    resolve_checkpoint_and_config,
    save_record,
    tensor_on_cuda,
)
from vitra.datasets.dataset import FrameDataset  # noqa: E402
from vitra.touch_editor.cache_utils import build_future_mask  # noqa: E402
from vitra.touch_editor.pseudo_pair import (  # noqa: E402
    DEFAULT_CONTACT_VERBS,
    MatchFeature,
    extract_contact_verbs,
    find_best_match,
    intersect_action_masks,
    normalized_phase,
)
from vitra.utils.data_utils import read_dataset_statistics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache OpenTouch tactile samples with pseudo-paired official GigaHands action targets."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opentouch_dataset_root", type=Path, default=Path("datasets/vitra_opentouch_keypoint"))
    parser.add_argument("--opentouch_data_mix", default="opentouch_keypoint_train")
    parser.add_argument("--opentouch_statistics_dataset_name", default="opentouch_keypoint_train")
    parser.add_argument(
        "--gigahands_dataset_root",
        type=Path,
        default=Path("datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked"),
    )
    parser.add_argument("--gigahands_data_mix", default="gigahands_real_train")
    parser.add_argument("--gigahands_statistics_dataset_name", default="gigahands_real_train")
    parser.add_argument("--shared_statistics_path", type=Path, default=None)
    parser.add_argument("--cache_root", type=Path, default=Path("runs/touch_editor_cache_opentouch_gigahands_matched"))
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--candidate_start_index", type=int, default=0)
    parser.add_argument("--candidate_pool_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edit_start_idx", type=int, default=10)
    parser.add_argument("--random_edit_start", action="store_true")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--match_contact_verbs", nargs="+", default=list(DEFAULT_CONTACT_VERBS))
    parser.add_argument("--w_task", type=float, default=3.0)
    parser.add_argument("--w_phase", type=float, default=1.0)
    parser.add_argument("--w_state", type=float, default=1.0)
    return parser.parse_args()


def shared_statistics_path(args: argparse.Namespace) -> Path:
    if args.shared_statistics_path is not None:
        return args.shared_statistics_path
    return (
        args.gigahands_dataset_root
        / "Annotation"
        / "statistics"
        / f"{args.gigahands_statistics_dataset_name}_keypoints_statistics.json"
    )


def build_dataset(
    config: dict[str, Any],
    dataset_root: Path,
    data_mix: str,
    *,
    statistics_dataset_name: str,
    shared_stats: dict[str, np.ndarray],
    load_images: bool,
) -> FrameDataset:
    train_cfg = config.get("train_dataset", {})
    dataset = FrameDataset(
        dataset_folder=str(dataset_root),
        dataset_name=data_mix,
        action_future_window_size=config.get("fwd_pred_next_n", 16) - 1,
        augmentation=False,
        normalization=True,
        processor=None,
        load_images=load_images,
        action_type="keypoints",
        use_rel=train_cfg.get("use_rel", False),
        clip_len=train_cfg.get("clip_len"),
        state_mask_prob=0.0,
        statistics_dataset_name=statistics_dataset_name,
    )
    dataset.episodic_dataset_core.data_statistics = shared_stats
    dataset.episodic_dataset_core.set_global_data_statistics(shared_stats)
    return dataset


def transformed_sample(dataset: FrameDataset, data_id: int) -> dict[str, Any]:
    raw_sample = dataset.episodic_dataset_core.__getitem__(data_id)
    return dataset.episodic_dataset_core.transform_trajectory(raw_sample.copy(), normalization=True)


def make_match_feature(
    dataset: FrameDataset,
    data_id: int,
    *,
    contact_verbs: list[str],
) -> MatchFeature:
    sample = transformed_sample(dataset, data_id)
    episode, frame_id, episode_id = get_episode_and_frame(dataset, data_id)
    episode_len = len(episode.get("video_decode_frame", episode.get("extrinsics", [])))
    label_text = str(episode.get("opentouch", {}).get("label_text", ""))
    matched_keyword = str(episode.get("opentouch", {}).get("matched_keyword", ""))
    verb_text = " ".join([str(sample["instruction"]), label_text, matched_keyword])
    return MatchFeature(
        data_id=int(data_id),
        episode_id=episode_id,
        frame_id=int(frame_id),
        instruction=str(sample["instruction"]),
        phase=normalized_phase(int(frame_id), int(episode_len)),
        state=np.asarray(sample["current_state"], dtype=np.float32),
        state_mask=np.asarray(sample["current_state_mask"], dtype=bool),
        action_target=np.asarray(sample["action_list"], dtype=np.float32),
        action_mask=np.asarray(sample["action_mask"], dtype=bool),
        contact_verbs=extract_contact_verbs(verb_text, contact_verbs),
    )


def build_gigahands_candidate_pool(
    dataset: FrameDataset,
    *,
    start_index: int,
    pool_size: int,
    contact_verbs: list[str],
) -> list[MatchFeature]:
    start = max(0, int(start_index))
    stop = min(len(dataset), start + int(pool_size))
    if start >= stop:
        raise ValueError(f"Empty GigaHands candidate range: start={start}, stop={stop}, dataset_len={len(dataset)}")
    candidates: list[MatchFeature] = []
    for data_id in tqdm(range(start, stop), desc="building GigaHands match pool"):
        candidates.append(make_match_feature(dataset, data_id, contact_verbs=contact_verbs))
    return candidates


def replace_target_with_gigahands_match(
    record: dict[str, np.ndarray],
    *,
    matched: MatchFeature,
    source: MatchFeature | None = None,
    match: Any | None = None,
) -> dict[str, np.ndarray]:
    a_base = np.asarray(record["a_base"], dtype=np.float32)
    a_target = np.asarray(matched.action_target, dtype=np.float32)
    action_mask = intersect_action_masks(record["action_mask"], matched.action_mask)
    edit_start_idx = int(np.asarray(record["edit_start_idx"]).item())
    record = dict(record)
    record["a_target"] = a_target
    record["residual_target"] = (a_target - a_base).astype(np.float32)
    record["action_mask"] = action_mask.astype(bool)
    record["future_mask"] = build_future_mask(action_mask, edit_start_idx)
    record["target_source"] = np.asarray("gigahands_matched")
    record["target_dataset"] = np.asarray("gigahands")
    record["touch_source"] = np.asarray("opentouch")
    record["matched_gigahands_data_id"] = np.asarray(matched.data_id, dtype=np.int64)
    record["matched_episode_id"] = np.asarray(matched.episode_id)
    record["matched_gigahands_frame_id"] = np.asarray(matched.frame_id, dtype=np.int64)
    record["matched_phase"] = np.asarray(matched.phase, dtype=np.float32)
    record["matched_instruction"] = np.asarray(matched.instruction)
    record["matched_contact_verbs"] = np.asarray(matched.contact_verbs, dtype=str)
    if source is not None:
        record["source_data_id"] = np.asarray(source.data_id, dtype=np.int64)
        record["source_episode_id"] = np.asarray(source.episode_id)
        record["source_frame_id"] = np.asarray(source.frame_id, dtype=np.int64)
        record["source_phase"] = np.asarray(source.phase, dtype=np.float32)
        record["source_instruction"] = np.asarray(source.instruction)
        record["source_contact_verbs"] = np.asarray(source.contact_verbs, dtype=str)
    if match is not None:
        record["match_score"] = np.asarray(match.score, dtype=np.float32)
        record["match_task_cost"] = np.asarray(match.task_cost, dtype=np.float32)
        record["match_phase_cost"] = np.asarray(match.phase_cost, dtype=np.float32)
        record["match_state_cost"] = np.asarray(match.state_cost, dtype=np.float32)
    return record


def main() -> None:
    from vitra.utils.config_utils import load_config

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Caching frozen VITRA actions requires CUDA because VITRA predict_action uses CUDA internally.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    weights_path, config_path, hf_repo_id = resolve_checkpoint_and_config(args.checkpoint, args.config)
    config = load_config(str(config_path))
    stats_path = shared_statistics_path(args)
    shared_stats = read_dataset_statistics(str(stats_path))

    opentouch_dataset = build_dataset(
        config,
        args.opentouch_dataset_root,
        args.opentouch_data_mix,
        statistics_dataset_name=args.opentouch_statistics_dataset_name,
        shared_stats=shared_stats,
        load_images=True,
    )
    gigahands_dataset = build_dataset(
        config,
        args.gigahands_dataset_root,
        args.gigahands_data_mix,
        statistics_dataset_name=args.gigahands_statistics_dataset_name,
        shared_stats=shared_stats,
        load_images=False,
    )
    candidates = build_gigahands_candidate_pool(
        gigahands_dataset,
        start_index=args.candidate_start_index,
        pool_size=args.candidate_pool_size,
        contact_verbs=args.match_contact_verbs,
    )
    model = load_frozen_vitra(config, weights_path)

    start_index = max(0, int(args.start_index))
    stop_index = len(opentouch_dataset) if args.max_samples is None else min(len(opentouch_dataset), start_index + int(args.max_samples))
    if start_index >= stop_index:
        raise ValueError(
            f"Empty OpenTouch cache range: start_index={start_index}, stop_index={stop_index}, dataset_len={len(opentouch_dataset)}"
        )

    written = 0
    match_scores: list[float] = []
    with torch.no_grad():
        for data_id in tqdm(range(start_index, stop_index), desc="caching pseudo-paired samples"):
            sample = transformed_sample(opentouch_dataset, data_id)
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
            episode, frame_id, episode_id = get_episode_and_frame(opentouch_dataset, data_id)
            edit_start_idx = choose_edit_start(len(sample["action_list"]), args, rng)
            source_feature = make_match_feature(opentouch_dataset, data_id, contact_verbs=args.match_contact_verbs)
            match = find_best_match(
                source_feature,
                candidates,
                w_task=args.w_task,
                w_phase=args.w_phase,
                w_state=args.w_state,
            )
            matched = candidates[match.index]
            record = build_cache_record(
                a_base=a_base,
                sample=sample,
                episode=episode,
                frame_id=frame_id,
                edit_start_idx=edit_start_idx,
                touch_mode="require",
            )
            record = replace_target_with_gigahands_match(record, matched=matched, source=source_feature, match=match)
            metadata = {
                "source_data_id": int(data_id),
                "source_episode_id": episode_id,
                "source_frame_id": int(frame_id),
                "source_instruction": str(sample["instruction"]),
                "source_contact_verbs": list(source_feature.contact_verbs),
                "target_source": "gigahands_matched",
                "target_data_id": int(matched.data_id),
                "target_episode_id": matched.episode_id,
                "target_frame_id": int(matched.frame_id),
                "target_instruction": matched.instruction,
                "target_contact_verbs": list(matched.contact_verbs),
                "match_score": float(match.score),
                "match_task_cost": float(match.task_cost),
                "match_phase_cost": float(match.phase_cost),
                "match_state_cost": float(match.state_cost),
                "edit_start_idx": int(edit_start_idx),
            }
            save_record(args.cache_root / f"sample_{data_id:08d}.npz", record, metadata)
            match_scores.append(float(match.score))
            written += 1

    summary = {
        "cache_root": str(args.cache_root),
        "num_samples": written,
        "start_index": start_index,
        "stop_index": stop_index,
        "opentouch_dataset_root": str(args.opentouch_dataset_root),
        "opentouch_data_mix": args.opentouch_data_mix,
        "opentouch_statistics_dataset_name": args.opentouch_statistics_dataset_name,
        "gigahands_dataset_root": str(args.gigahands_dataset_root),
        "gigahands_data_mix": args.gigahands_data_mix,
        "gigahands_candidate_start_index": int(args.candidate_start_index),
        "gigahands_candidate_pool_size": len(candidates),
        "checkpoint": str(args.checkpoint),
        "weights_path": str(weights_path),
        "config_path": str(config_path),
        "hf_repo_id": hf_repo_id,
        "shared_statistics_path": str(stats_path),
        "target_source": "gigahands_matched",
        "touch_source": "opentouch",
        "match_score_mean": float(np.mean(match_scores)) if match_scores else None,
        "match_score_max": float(np.max(match_scores)) if match_scores else None,
    }
    (args.cache_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
