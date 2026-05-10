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
    return parser.parse_args()


def masked_sse_and_count(value: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(value.dtype)
    return float((value.square() * mask).sum().detach().cpu()), float(mask.sum().detach().cpu())


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = TouchEditorCacheDataset(args.cache_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    editor = load_touch_editor(args.touch_editor_checkpoint, device=args.device)

    totals: dict[str, dict[str, float]] = {"base": {"sse": 0.0, "count": 0.0}}
    for idx, _ in enumerate(args.edit_times, start=1):
        totals[f"edit_{idx}"] = {"sse": 0.0, "count": 0.0}

    edit_indices: list[list[int]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating touch guidance"):
            batch = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in batch.items()}
            action_mask = batch["action_mask"]
            base_sse, base_count = masked_sse_and_count(batch["a_base"] - batch["a_target"], action_mask)
            totals["base"]["sse"] += base_sse
            totals["base"]["count"] += base_count

            result = apply_touch_guidance_schedule(
                editor=editor,
                a_base=batch["a_base"],
                current_state=batch["current_state"],
                current_state_mask=batch["current_state_mask"],
                touch_pressure=batch["touch_pressure"],
                touch_mask=batch["touch_mask"],
                action_mask=action_mask,
                fps=args.fps,
                edit_times=args.edit_times,
            )
            edit_indices.append(result.edit_indices)
            for idx, a_edit in enumerate(result.a_history, start=1):
                a_edit = a_edit.to(args.device)
                sse, count = masked_sse_and_count(a_edit - batch["a_target"], action_mask)
                key = f"edit_{idx}"
                totals[key]["sse"] += sse
                totals[key]["count"] += count

    metrics = {}
    for key, payload in totals.items():
        metrics[f"{key}_mse"] = payload["sse"] / max(payload["count"], 1.0)
    metrics.update(
        {
            "num_samples": len(dataset),
            "cache_root": str(args.cache_root),
            "touch_editor_checkpoint": str(args.touch_editor_checkpoint),
            "fps": args.fps,
            "edit_times": args.edit_times,
            "edit_indices": edit_indices[0] if edit_indices else [],
        }
    )
    args.output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
