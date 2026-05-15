#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


ABLATIONS = ("matched", "shuffled_touch", "zero_touch", "future_touch_oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize touch-sensitive editor ablations by variant.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_edit_mse(metrics: dict[str, object]) -> float:
    for key in ("edit_1_edit_mse", "edit_2_edit_mse", "edit_3_edit_mse"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError("No edit MSE key found")


def first_base_mse(metrics: dict[str, object]) -> float:
    for key in ("edit_1_base_mse", "edit_2_base_mse", "edit_3_base_mse", "base_mse"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError("No base MSE key found")


def first_prefix_change(metrics: dict[str, object]) -> float:
    for key in ("edit_1_prefix_change_l2", "edit_2_prefix_change_l2", "edit_3_prefix_change_l2"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def first_touch_gate(metrics: dict[str, object]) -> float:
    for key in ("edit_1_touch_gate_mean", "edit_2_touch_gate_mean", "edit_3_touch_gate_mean"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def summarize_variant(run_root: Path, variant: str) -> dict[str, object]:
    payloads = {}
    for ablation in ABLATIONS:
        path = run_root / f"eval_{variant}_{ablation}.json"
        if path.exists():
            payloads[ablation] = load_json(path)
    if "matched" not in payloads:
        raise FileNotFoundError(f"Missing matched eval for variant {variant}")

    matched = first_edit_mse(payloads["matched"])
    shuffled = first_edit_mse(payloads["shuffled_touch"]) if "shuffled_touch" in payloads else None
    zero = first_edit_mse(payloads["zero_touch"]) if "zero_touch" in payloads else None
    oracle = first_edit_mse(payloads["future_touch_oracle"]) if "future_touch_oracle" in payloads else None
    base = first_base_mse(payloads["matched"])
    summary: dict[str, object] = {
        "variant": variant,
        "base_mse": base,
        "matched_mse": matched,
        "shuffled_touch_mse": shuffled,
        "zero_touch_mse": zero,
        "future_touch_oracle_mse": oracle,
        "matched_improvement_pct": (base - matched) / base * 100.0 if base > 0 else 0.0,
        "matched_vs_shuffled_gap": (shuffled - matched) if shuffled is not None else None,
        "matched_vs_shuffled_gap_pct": ((shuffled - matched) / shuffled * 100.0) if shuffled and shuffled > 0 else None,
        "matched_vs_zero_gap": (zero - matched) if zero is not None else None,
        "prefix_change_l2": first_prefix_change(payloads["matched"]),
        "touch_gate_mean": first_touch_gate(payloads["matched"]),
    }
    return summary


def discover_variants(run_root: Path) -> list[str]:
    variants = []
    for path in sorted(run_root.glob("eval_*_matched.json")):
        name = path.name.removeprefix("eval_").removesuffix("_matched.json")
        variants.append(name)
    return variants


def main() -> None:
    args = parse_args()
    variants = discover_variants(args.run_root)
    summary = {
        "run_root": str(args.run_root),
        "variants": [summarize_variant(args.run_root, variant) for variant in variants],
    }
    output_path = args.output_path or args.run_root / "sensitivity_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
