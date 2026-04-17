#!/usr/bin/env python3
"""Analyze whether hand-prediction metadata is single-hand or bimanual.

The script reads metadata only. It does not decode videos or run the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_THRESHOLDS_CM = (0.5, 1.0, 2.0)


def _empty_counter(dataset_name: str) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "episodes": 0,
        "frames_sampled": 0,
        "schema_dual_hand_frames": 0,
        "valid_dual_hand_frames": 0,
        "supervised_dual_hand_frames": 0,
        "semantic_dual_hand_frames": 0,
        "left_only_valid_frames": 0,
        "right_only_valid_frames": 0,
        "none_valid_frames": 0,
        "active_dual_hand_frames_by_threshold_cm": {str(t): 0 for t in DEFAULT_THRESHOLDS_CM},
    }


def _with_ratios(summary: dict[str, Any]) -> dict[str, Any]:
    total = summary["frames_sampled"]
    fields = [
        "schema_dual_hand_frames",
        "valid_dual_hand_frames",
        "supervised_dual_hand_frames",
        "semantic_dual_hand_frames",
        "left_only_valid_frames",
        "right_only_valid_frames",
        "none_valid_frames",
    ]
    for field in fields:
        summary[field.replace("_frames", "_ratio")] = summary[field] / total if total else 0.0
    summary["active_dual_hand_ratio_by_threshold_cm"] = {
        k: v / total if total else 0.0
        for k, v in summary["active_dual_hand_frames_by_threshold_cm"].items()
    }
    return summary


def _sample_indices(total: int, sample_count: int) -> np.ndarray:
    if sample_count < 0 or sample_count >= total:
        return np.arange(total, dtype=np.int64)
    if sample_count == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.linspace(0, total - 1, sample_count, dtype=np.int64)


def _is_non_none_text(text: Any) -> bool:
    if text is None:
        return False
    text = str(text).strip()
    return bool(text) and text.lower().rstrip(".") != "none"


def _has_matching_text(text_entries: Any, frame_id: int) -> bool:
    if not isinstance(text_entries, (list, tuple)):
        return False
    for entry in text_entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        text, frame_range = entry[0], entry[1]
        if not _is_non_none_text(text):
            continue
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            continue
        start, end = int(frame_range[0]), int(frame_range[1])
        if start <= frame_id < end:
            return True
    return False


def _hand_positions_from_episode(epi: dict[str, Any], hand: str) -> np.ndarray:
    hand_data = epi.get(hand, {})
    if "transl_worldspace" in hand_data:
        return np.asarray(hand_data["transl_worldspace"], dtype=np.float32)
    if "joints_worldspace" in hand_data:
        joints = np.asarray(hand_data["joints_worldspace"], dtype=np.float32)
        if joints.ndim == 3 and joints.shape[1] >= 18:
            return joints[:, [0, 2, 5, 9, 13, 17], :].mean(axis=1)
        if joints.ndim == 3:
            return joints.mean(axis=1)
    return np.zeros((0, 3), dtype=np.float32)


def _local_motion_meters(positions: np.ndarray, frame_id: int, window: int, valid: np.ndarray | None = None) -> float:
    if positions.ndim != 2 or len(positions) == 0:
        return 0.0
    start = max(0, frame_id - window)
    end = min(len(positions) - 1, frame_id + window)
    indices = np.arange(start, end + 1)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        indices = indices[valid[indices]]
    if len(indices) < 2:
        return 0.0
    segment = positions[indices]
    return float(np.linalg.norm(segment.max(axis=0) - segment.min(axis=0)))


def _window_all_valid(valid: np.ndarray, frame_id: int, future_window: int) -> bool:
    valid = np.asarray(valid, dtype=bool)
    end = frame_id + future_window
    if frame_id < 0 or end >= len(valid):
        return False
    return bool(valid[frame_id : end + 1].all())


def _summarize_vitra_sample(
    dataset_name: str,
    episode_id: str,
    epi: dict[str, Any],
    frame_id: int,
    future_window: int,
    thresholds_cm: Iterable[float],
) -> dict[str, Any]:
    has_left = "left" in epi
    has_right = "right" in epi
    schema_dual = has_left and has_right

    left_valid_arr = np.asarray(epi.get("left", {}).get("kept_frames", []), dtype=bool)
    right_valid_arr = np.asarray(epi.get("right", {}).get("kept_frames", []), dtype=bool)
    left_valid = bool(frame_id < len(left_valid_arr) and left_valid_arr[frame_id])
    right_valid = bool(frame_id < len(right_valid_arr) and right_valid_arr[frame_id])
    valid_dual = left_valid and right_valid

    supervised_dual = (
        _window_all_valid(left_valid_arr, frame_id, future_window)
        and _window_all_valid(right_valid_arr, frame_id, future_window)
    )

    text = epi.get("text", {})
    semantic_dual = _has_matching_text(text.get("left"), frame_id) and _has_matching_text(text.get("right"), frame_id)

    left_positions = _hand_positions_from_episode(epi, "left")
    right_positions = _hand_positions_from_episode(epi, "right")
    left_motion = _local_motion_meters(left_positions, frame_id, future_window, left_valid_arr)
    right_motion = _local_motion_meters(right_positions, frame_id, future_window, right_valid_arr)
    active_by_threshold = {
        str(t): bool(left_motion > t / 100.0 and right_motion > t / 100.0)
        for t in thresholds_cm
    }

    return {
        "dataset": dataset_name,
        "episode_id": episode_id,
        "frame_id": frame_id,
        "schema_dual_hand": schema_dual,
        "valid_dual_hand": valid_dual,
        "supervised_dual_hand": supervised_dual,
        "semantic_dual_hand": semantic_dual,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "left_motion_m": left_motion,
        "right_motion_m": right_motion,
        **{f"active_dual_hand_{t}cm": active for t, active in active_by_threshold.items()},
    }


def _update_summary(summary: dict[str, Any], sample: dict[str, Any], thresholds_cm: Iterable[float]) -> None:
    summary["frames_sampled"] += 1
    if sample["schema_dual_hand"]:
        summary["schema_dual_hand_frames"] += 1
    if sample["valid_dual_hand"]:
        summary["valid_dual_hand_frames"] += 1
    if sample["supervised_dual_hand"]:
        summary["supervised_dual_hand_frames"] += 1
    if sample["semantic_dual_hand"]:
        summary["semantic_dual_hand_frames"] += 1
    if sample["left_valid"] and not sample["right_valid"]:
        summary["left_only_valid_frames"] += 1
    elif sample["right_valid"] and not sample["left_valid"]:
        summary["right_only_valid_frames"] += 1
    elif not sample["left_valid"] and not sample["right_valid"]:
        summary["none_valid_frames"] += 1
    for threshold in thresholds_cm:
        key = str(threshold)
        if sample[f"active_dual_hand_{key}cm"]:
            summary["active_dual_hand_frames_by_threshold_cm"][key] += 1


def _episode_path(label_root: Path, episode_id: str) -> Path:
    episode_id = episode_id[:-4] if episode_id.endswith(".npy") else episode_id
    return label_root / f"{episode_id}.npy"


def analyze_vitra_format_dataset(
    dataset_root: str | Path,
    dataset_names: list[str],
    sample_frames_per_dataset: int = 5000,
    future_window: int = 15,
    thresholds_cm: Iterable[float] = DEFAULT_THRESHOLDS_CM,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    thresholds_cm = tuple(float(t) for t in thresholds_cm)
    result = {"datasets": {}, "samples": []}

    for dataset_name in dataset_names:
        annotation_root = dataset_root / "Annotation" / dataset_name
        index_path = annotation_root / "episode_frame_index.npz"
        label_root = annotation_root / "episodic_annotations"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing VITRA episode index: {index_path}")
        if not label_root.exists():
            raise FileNotFoundError(f"Missing VITRA episode labels: {label_root}")

        index_data = np.load(index_path, allow_pickle=True)
        index_frame_pair = index_data["index_frame_pair"]
        index_to_episode_id = index_data["index_to_episode_id"]
        selected_indices = _sample_indices(len(index_frame_pair), sample_frames_per_dataset)

        summary = _empty_counter(dataset_name)
        episode_cache: dict[str, dict[str, Any]] = {}
        seen_episodes = set()
        for sample_idx in selected_indices:
            episode_index, frame_id = index_frame_pair[int(sample_idx)]
            episode_id = str(index_to_episode_id[int(episode_index)])
            seen_episodes.add(episode_id)
            if episode_id not in episode_cache:
                episode_cache[episode_id] = np.load(_episode_path(label_root, episode_id), allow_pickle=True).item()
            sample = _summarize_vitra_sample(
                dataset_name,
                episode_id,
                episode_cache[episode_id],
                int(frame_id),
                future_window,
                thresholds_cm,
            )
            _update_summary(summary, sample, thresholds_cm)
            result["samples"].append(sample)

        summary["episodes"] = len(seen_episodes)
        result["datasets"][dataset_name] = _with_ratios(summary)

    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _raw_positions(params: dict[str, Any], hand: str) -> np.ndarray:
    hand_data = params.get(hand, {})
    if "Th" in hand_data:
        return np.asarray(hand_data["Th"], dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def analyze_gigahands_raw_demo(
    gigahands_raw_root: str | Path,
    future_window: int = 15,
    thresholds_cm: Iterable[float] = DEFAULT_THRESHOLDS_CM,
) -> dict[str, Any]:
    gigahands_raw_root = Path(gigahands_raw_root)
    thresholds_cm = tuple(float(t) for t in thresholds_cm)
    hand_pose_root = gigahands_raw_root / "hand_pose"
    if not hand_pose_root.exists():
        raise FileNotFoundError(f"Missing GigaHands demo hand_pose root: {hand_pose_root}")

    result = {"datasets": {}, "samples": []}
    summary = _empty_counter("gigahands_raw_demo")

    for params_path in sorted(hand_pose_root.glob("*/params/*.json")):
        sequence = params_path.parents[1].name
        params = _load_json(params_path)
        left_positions = _raw_positions(params, "left")
        right_positions = _raw_positions(params, "right")
        frame_count = min(len(left_positions), len(right_positions))
        if frame_count == 0:
            continue

        summary["episodes"] += 1
        left_valid = np.ones(frame_count, dtype=bool)
        right_valid = np.ones(frame_count, dtype=bool)
        for frame_id in range(frame_count):
            left_motion = _local_motion_meters(left_positions, frame_id, future_window, left_valid)
            right_motion = _local_motion_meters(right_positions, frame_id, future_window, right_valid)
            sample = {
                "dataset": "gigahands_raw_demo",
                "episode_id": f"{sequence}_{params_path.stem}",
                "frame_id": frame_id,
                "schema_dual_hand": "left" in params and "right" in params,
                "valid_dual_hand": True,
                "supervised_dual_hand": _window_all_valid(left_valid, frame_id, future_window)
                and _window_all_valid(right_valid, frame_id, future_window),
                "semantic_dual_hand": False,
                "left_valid": True,
                "right_valid": True,
                "left_motion_m": left_motion,
                "right_motion_m": right_motion,
                **{
                    f"active_dual_hand_{threshold}cm": bool(
                        left_motion > threshold / 100.0 and right_motion > threshold / 100.0
                    )
                    for threshold in thresholds_cm
                },
            }
            _update_summary(summary, sample, thresholds_cm)
            result["samples"].append(sample)

    result["datasets"]["gigahands_raw_demo"] = _with_ratios(summary)
    return result


def _merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {"datasets": {}, "samples": []}
    for result in results:
        merged["datasets"].update(result.get("datasets", {}))
        merged["samples"].extend(result.get("samples", []))
    return merged


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_outputs(
    result: dict[str, Any],
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
) -> None:
    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, default=_json_default))

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        samples = result.get("samples", [])
        fieldnames = sorted({key for sample in samples for key in sample.keys()})
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

    if output_md:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown_report(result))


def render_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Hand Usage Analysis",
        "",
        "| Dataset | Episodes | Frames sampled | Schema dual-hand | Valid dual-hand | Supervised dual-hand | Semantic dual-hand | Active dual-hand @1cm | Left-only | Right-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, summary in sorted(result.get("datasets", {}).items()):
        total = max(1, summary["frames_sampled"])
        active_1cm = summary["active_dual_hand_frames_by_threshold_cm"].get("1.0", 0)
        lines.append(
            "| {dataset} | {episodes} | {frames} | {schema:.1%} | {valid:.1%} | {supervised:.1%} | {semantic:.1%} | {active:.1%} | {left:.1%} | {right:.1%} |".format(
                dataset=dataset,
                episodes=summary["episodes"],
                frames=summary["frames_sampled"],
                schema=summary["schema_dual_hand_frames"] / total,
                valid=summary["valid_dual_hand_frames"] / total,
                supervised=summary["supervised_dual_hand_frames"] / total,
                semantic=summary["semantic_dual_hand_frames"] / total,
                active=active_1cm / total,
                left=summary["left_only_valid_frames"] / total,
                right=summary["right_only_valid_frames"] / total,
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `schema_dual_hand` means both `left` and `right` slots exist.",
            "- `valid_dual_hand` means both hands are valid at the sampled frame.",
            "- `supervised_dual_hand` means both hands stay valid through the requested future action window.",
            "- `semantic_dual_hand` means both hands have non-None text active at the sampled frame.",
            "- `active_dual_hand` uses local left/right hand displacement over the temporal window.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=None, help="VITRA-format dataset root.")
    parser.add_argument("--dataset_names", nargs="*", default=[], help="VITRA-format dataset names to analyze.")
    parser.add_argument("--gigahands_raw_root", type=Path, default=None, help="Raw GigaHands demo root.")
    parser.add_argument("--sample_frames_per_dataset", type=int, default=5000)
    parser.add_argument("--future_window", type=int, default=15)
    parser.add_argument("--thresholds_cm", nargs="*", type=float, default=list(DEFAULT_THRESHOLDS_CM))
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    if args.dataset_root and args.dataset_names:
        results.append(
            analyze_vitra_format_dataset(
                dataset_root=args.dataset_root,
                dataset_names=args.dataset_names,
                sample_frames_per_dataset=args.sample_frames_per_dataset,
                future_window=args.future_window,
                thresholds_cm=args.thresholds_cm,
            )
        )
    if args.gigahands_raw_root:
        results.append(
            analyze_gigahands_raw_demo(
                gigahands_raw_root=args.gigahands_raw_root,
                future_window=args.future_window,
                thresholds_cm=args.thresholds_cm,
            )
        )
    if not results:
        raise SystemExit("Provide --dataset_root with --dataset_names and/or --gigahands_raw_root.")

    result = _merge_results(results)
    write_outputs(result, args.output_json, args.output_csv, args.output_md)
    print(render_markdown_report(result))


if __name__ == "__main__":
    main()
