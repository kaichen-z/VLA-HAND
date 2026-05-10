#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Design 1 touch ablation metrics and tactile gains.")
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--zero_touch", type=Path, required=True)
    parser.add_argument("--shuffled_touch", type=Path, required=True)
    parser.add_argument("--no_touch", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def gain(matched: float, baseline: float) -> tuple[float, float]:
    abs_gain = baseline - matched
    pct_gain = abs_gain / baseline * 100.0 if baseline > 0 else 0.0
    return abs_gain, pct_gain


def main() -> None:
    args = parse_args()
    metrics = {
        "matched": load_metrics(args.matched),
        "zero_touch": load_metrics(args.zero_touch),
        "shuffled_touch": load_metrics(args.shuffled_touch),
        "no_touch": load_metrics(args.no_touch),
    }
    summary: dict[str, object] = {
        "inputs": {name: str(path) for name, path in vars(args).items() if name != "output_path"},
        "metrics": metrics,
        "touch_specific_gains": {},
    }
    edit_times = metrics["matched"].get("edit_times", [])
    for idx, _ in enumerate(edit_times, start=1):
        matched_mse = float(metrics["matched"][f"edit_{idx}_edit_mse"])
        gains = {}
        for baseline_name in ("shuffled_touch", "zero_touch", "no_touch"):
            baseline_mse = float(metrics[baseline_name][f"edit_{idx}_edit_mse"])
            abs_gain, pct_gain = gain(matched_mse, baseline_mse)
            gains[f"touch_gain_vs_{baseline_name}"] = abs_gain
            gains[f"touch_gain_pct_vs_{baseline_name}"] = pct_gain
        summary["touch_specific_gains"][f"edit_{idx}"] = gains

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
