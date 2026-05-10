#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.touch_editor.dataset import TouchEditorCacheDataset
from vitra.touch_editor.guidance import apply_touch_guidance_schedule, load_touch_editor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 0.33s/0.66s touch-guided edits on cached VITRA actions.")
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--touch_editor_checkpoint", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=Path("runs/touch_editor_eval/metrics.json"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--edit_times", type=float, nargs="+", default=[0.33, 0.66])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ablation",
        choices=("matched", "zero_touch", "shuffled_touch", "random_pair", "no_touch"),
        default="matched",
        help="Which pseudo-pair/touch ablation to evaluate.",
    )
    parser.add_argument("--random_pair_cache_root", type=Path, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=1)
    parser.add_argument("--subset", choices=("all", "high_contact"), default="all")
    parser.add_argument("--high_contact_quantile", type=float, default=0.75)
    parser.add_argument("--zero_touch", action="store_true", help="Deprecated alias for --ablation zero_touch.")
    return parser.parse_args()


def masked_sse_and_count(value: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(value.dtype)
    return float((value.square() * mask).sum().detach().cpu()), float(mask.sum().detach().cpu())


def touch_contact_score(touch_pressure: torch.Tensor, touch_mask: torch.Tensor) -> float:
    valid = touch_mask.to(touch_pressure.dtype)[..., None, None]
    denom = valid.sum() * touch_pressure.shape[-1] * touch_pressure.shape[-2]
    if float(denom) <= 0.0:
        return 0.0
    return float((touch_pressure.abs() * valid).sum().item() / float(denom))


def high_contact_path_set(
    dataset: TouchEditorCacheDataset,
    *,
    quantile: float,
) -> tuple[set[str], float]:
    scores: list[tuple[str, float]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        scores.append((item["path"], touch_contact_score(item["touch_pressure"], item["touch_mask"])))
    if not scores:
        return set(), 0.0
    threshold = torch.quantile(torch.tensor([score for _, score in scores], dtype=torch.float32), float(quantile)).item()
    selected = {path for path, score in scores if score >= threshold}
    return selected, float(threshold)


def filter_batch_by_paths(batch: dict[str, object], selected_paths: set[str]) -> dict[str, object] | None:
    paths = [str(path) for path in batch["path"]]
    keep = torch.tensor([path in selected_paths for path in paths], dtype=torch.bool)
    if not bool(keep.any()):
        return None
    out: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == keep.shape:
            out[key] = value[keep]
        elif key == "path":
            out[key] = [path for path, should_keep in zip(value, keep.tolist()) if should_keep]
        else:
            out[key] = value
    return out


def hand_slices(action_dim: int) -> dict[str, slice]:
    half = action_dim // 2
    return {"left": slice(0, half), "right": slice(half, action_dim)}


def add_mse(totals: dict[str, float], prefix: str, diff: torch.Tensor, mask: torch.Tensor) -> None:
    sse, count = masked_sse_and_count(diff, mask)
    totals[f"{prefix}_sse"] = totals.get(f"{prefix}_sse", 0.0) + sse
    totals[f"{prefix}_count"] = totals.get(f"{prefix}_count", 0.0) + count


def add_hand_mse(totals: dict[str, float], prefix: str, diff: torch.Tensor, mask: torch.Tensor) -> None:
    add_mse(totals, prefix, diff, mask)
    for hand, hand_slice in hand_slices(diff.shape[-1]).items():
        add_mse(totals, f"{prefix}_{hand}", diff[..., hand_slice], mask[..., hand_slice])


def l2_timestep_sum_and_count(value: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(value.dtype)
    timestep_valid = mask.any(dim=-1)
    norms = ((value * mask).square().sum(dim=-1).sqrt() * timestep_valid.to(value.dtype)).sum()
    return float(norms.detach().cpu()), float(timestep_valid.sum().detach().cpu())


def add_delta_stats(totals: dict[str, float], prefix: str, delta: torch.Tensor, editable: torch.Tensor) -> None:
    delta_sum, delta_count = l2_timestep_sum_and_count(delta, editable)
    totals[f"{prefix}_delta_norm_sum"] = totals.get(f"{prefix}_delta_norm_sum", 0.0) + delta_sum
    totals[f"{prefix}_delta_norm_count"] = totals.get(f"{prefix}_delta_norm_count", 0.0) + delta_count

    if delta.shape[1] <= 1:
        return
    smooth_mask = editable[:, 1:] * editable[:, :-1]
    smooth_sum, smooth_count = l2_timestep_sum_and_count(delta[:, 1:] - delta[:, :-1], smooth_mask)
    totals[f"{prefix}_smoothness_sum"] = totals.get(f"{prefix}_smoothness_sum", 0.0) + smooth_sum
    totals[f"{prefix}_smoothness_count"] = totals.get(f"{prefix}_smoothness_count", 0.0) + smooth_count


def shuffle_touch(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(batch["touch_pressure"].shape[0])
    if batch_size <= 1:
        return batch["touch_pressure"], batch["touch_mask"]
    order = torch.roll(torch.arange(batch_size, device=batch["touch_pressure"].device), shifts=1)
    return batch["touch_pressure"][order], batch["touch_mask"][order]


def randomize_targets_from_batch(batch: dict[str, torch.Tensor]) -> None:
    batch_size = int(batch["a_target"].shape[0])
    if batch_size <= 1:
        return
    order = torch.roll(torch.arange(batch_size, device=batch["a_target"].device), shifts=1)
    batch["a_target"] = batch["a_target"][order]
    batch["action_mask"] = batch["action_mask"][order]


def load_random_pair_targets(
    random_dataset: TouchEditorCacheDataset,
    *,
    start: int,
    batch_size: int,
    offset: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    masks = []
    for idx in range(batch_size):
        item = random_dataset[(start + idx + offset) % len(random_dataset)]
        targets.append(item["a_target"])
        masks.append(item["action_mask"])
    return torch.stack(targets).to(device), torch.stack(masks).to(device)


def finalize_metrics(totals: dict[str, float], edit_count: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for edit_idx in range(1, edit_count + 1):
        prefix = f"edit_{edit_idx}"
        for scope in ("", "_left", "_right"):
            base_key = f"{prefix}_base{scope}"
            edit_key = f"{prefix}_edit{scope}"
            base_mse = totals.get(f"{base_key}_sse", 0.0) / max(totals.get(f"{base_key}_count", 0.0), 1.0)
            edit_mse = totals.get(f"{edit_key}_sse", 0.0) / max(totals.get(f"{edit_key}_count", 0.0), 1.0)
            label = f"{prefix}{scope}"
            metrics[f"{label}_base_mse"] = base_mse
            metrics[f"{label}_edit_mse"] = edit_mse
            metrics[f"{label}_improvement_pct"] = (base_mse - edit_mse) / base_mse * 100.0 if base_mse > 0 else 0.0

        metrics[f"{prefix}_mse"] = metrics[f"{prefix}_edit_mse"]
        metrics[f"{prefix}_valid_editable_dims"] = totals.get(f"{prefix}_valid_editable_dims", 0.0)
        metrics[f"{prefix}_delta_norm"] = totals.get(f"{prefix}_delta_norm_sum", 0.0) / max(
            totals.get(f"{prefix}_delta_norm_count", 0.0), 1.0
        )
        metrics[f"{prefix}_smoothness"] = totals.get(f"{prefix}_smoothness_sum", 0.0) / max(
            totals.get(f"{prefix}_smoothness_count", 0.0), 1.0
        )
    metrics["base_mse"] = metrics.get("edit_1_base_mse", 0.0)
    return metrics


def main() -> None:
    args = parse_args()
    if args.zero_touch:
        args.ablation = "zero_touch"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = TouchEditorCacheDataset(args.cache_root)
    selected_paths: set[str] | None = None
    high_contact_threshold: float | None = None
    if args.subset == "high_contact":
        selected_paths, high_contact_threshold = high_contact_path_set(dataset, quantile=args.high_contact_quantile)
    random_dataset = TouchEditorCacheDataset(args.random_pair_cache_root) if args.random_pair_cache_root is not None else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    editor = load_touch_editor(args.touch_editor_checkpoint, device=args.device)

    totals: dict[str, float] = {"touch_valid_sum": 0.0, "touch_valid_count": 0.0}
    edit_indices: list[list[int]] = []
    sample_offset = 0
    with torch.no_grad():
        selected_sample_count = 0
        for batch in tqdm(loader, desc=f"evaluating touch guidance ({args.ablation}, {args.subset})"):
            if selected_paths is not None:
                batch = filter_batch_by_paths(batch, selected_paths)
                if batch is None:
                    continue
            batch = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in batch.items()}
            batch_size = int(batch["a_base"].shape[0])
            selected_sample_count += batch_size
            if args.ablation == "random_pair":
                if random_dataset is None:
                    randomize_targets_from_batch(batch)
                else:
                    batch["a_target"], batch["action_mask"] = load_random_pair_targets(
                        random_dataset,
                        start=sample_offset,
                        batch_size=batch_size,
                        offset=args.shuffle_seed,
                        device=args.device,
                    )

            touch_pressure = batch["touch_pressure"]
            touch_mask = batch["touch_mask"]
            if args.ablation in {"zero_touch", "no_touch"}:
                touch_pressure = torch.zeros_like(touch_pressure)
                touch_mask = torch.zeros_like(touch_mask)
            elif args.ablation == "shuffled_touch":
                touch_pressure, touch_mask = shuffle_touch(batch)

            totals["touch_valid_sum"] += float(touch_mask.to(torch.float32).sum().detach().cpu())
            totals["touch_valid_count"] += float(touch_mask.numel())

            result = apply_touch_guidance_schedule(
                editor=editor,
                a_base=batch["a_base"],
                current_state=batch["current_state"],
                current_state_mask=batch["current_state_mask"],
                touch_pressure=touch_pressure,
                touch_mask=touch_mask,
                action_mask=batch["action_mask"],
                fps=args.fps,
                edit_times=args.edit_times,
            )
            edit_indices.append(result.edit_indices)
            for idx, (a_edit, delta, future_mask) in enumerate(
                zip(result.a_history, result.deltas, result.future_masks), start=1
            ):
                prefix = f"edit_{idx}"
                a_edit = a_edit.to(args.device)
                delta = delta.to(args.device)
                editable = future_mask.to(args.device).to(batch["a_base"].dtype)
                add_hand_mse(totals, f"{prefix}_base", batch["a_base"] - batch["a_target"], editable)
                add_hand_mse(totals, f"{prefix}_edit", a_edit - batch["a_target"], editable)
                add_delta_stats(totals, prefix, delta, editable)
                totals[f"{prefix}_valid_editable_dims"] = totals.get(f"{prefix}_valid_editable_dims", 0.0) + float(
                    editable.sum().detach().cpu()
                )
            sample_offset += batch_size

    metrics = finalize_metrics(totals, len(args.edit_times))
    metrics.update(
        {
            "num_samples": len(dataset),
            "selected_num_samples": selected_sample_count,
            "cache_root": str(args.cache_root),
            "touch_editor_checkpoint": str(args.touch_editor_checkpoint),
            "fps": args.fps,
            "edit_times": args.edit_times,
            "edit_indices": edit_indices[0] if edit_indices else [],
            "ablation": args.ablation,
            "zero_touch": args.ablation in {"zero_touch", "no_touch"},
            "subset": args.subset,
            "high_contact_quantile": args.high_contact_quantile,
            "high_contact_threshold": high_contact_threshold,
            "random_pair_cache_root": str(args.random_pair_cache_root) if args.random_pair_cache_root is not None else None,
            "touch_valid_rate": totals["touch_valid_sum"] / max(totals["touch_valid_count"], 1.0),
        }
    )
    args.output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
