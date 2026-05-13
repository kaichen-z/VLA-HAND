#!/usr/bin/env python3
"""Initialize a compressed finetune-distill student from a base VITRA checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.utils.config_utils import load_config


def resolve_weights_path(path: Path) -> Path:
    return path / "weights.pt" if path.is_dir() else path


def parse_layer_map(value: str) -> list[int]:
    layer_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layer_ids:
        raise ValueError("layer_map must contain at least one layer id")
    return layer_ids


def _shape(value: Any) -> tuple[int, ...] | None:
    return tuple(value.shape) if hasattr(value, "shape") else None


def _same_shape(left: Any, right: Any) -> bool:
    return _shape(left) == _shape(right)


def copy_mapped_action_weights(
    teacher_state: dict[str, Any],
    student_state: dict[str, Any],
    layer_map: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {
        "copied_matching": [],
        "copied_blocks": [],
        "skipped_missing": [],
        "skipped_shape_mismatch": [],
    }
    output = dict(student_state)
    block_prefix = (
        "act_model.net.blocks."
        if any(key.startswith("act_model.net.blocks.") for key in student_state)
        else "net.blocks."
    )

    for key, student_value in student_state.items():
        if key.startswith(block_prefix):
            continue
        teacher_value = teacher_state.get(key)
        if teacher_value is None:
            report["skipped_missing"].append(key)
            continue
        if not _same_shape(teacher_value, student_value):
            report["skipped_shape_mismatch"].append(
                {"key": key, "teacher": _shape(teacher_value), "student": _shape(student_value)}
            )
            continue
        output[key] = teacher_value.detach().clone() if hasattr(teacher_value, "detach") else copy.deepcopy(teacher_value)
        report["copied_matching"].append(key)

    for student_idx, teacher_idx in enumerate(layer_map):
        teacher_prefix = f"{block_prefix}{teacher_idx}."
        student_prefix = f"{block_prefix}{student_idx}."
        copied_keys = []
        for teacher_key, teacher_value in teacher_state.items():
            if not teacher_key.startswith(teacher_prefix):
                continue
            suffix = teacher_key[len(teacher_prefix) :]
            student_key = f"{student_prefix}{suffix}"
            if student_key not in student_state:
                report["skipped_missing"].append(student_key)
                continue
            if not _same_shape(teacher_value, student_state[student_key]):
                report["skipped_shape_mismatch"].append(
                    {
                        "key": student_key,
                        "teacher": _shape(teacher_value),
                        "student": _shape(student_state[student_key]),
                    }
                )
                continue
            output[student_key] = teacher_value.detach().clone() if hasattr(teacher_value, "detach") else copy.deepcopy(teacher_value)
            copied_keys.append(student_key)
        report["copied_blocks"].append(
            {"student_block": student_idx, "teacher_block": teacher_idx, "num_tensors": len(copied_keys)}
        )

    return output, report


def cpu_state_dict(module) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def create_student_checkpoint(
    student_config_path: Path,
    teacher_config_path: Path,
    teacher_checkpoint: Path,
    output_dir: Path,
    layer_map: list[int],
    force: bool = False,
) -> dict[str, Any]:
    weights_path = output_dir / "weights.pt"
    if weights_path.exists() and not force:
        return {"weights_path": str(weights_path), "skipped_existing": True}

    student_config = load_config(str(student_config_path))
    teacher_config = load_config(str(teacher_config_path))

    teacher = build_vla(teacher_config)
    teacher = load_vla_checkpoint(teacher, str(resolve_weights_path(teacher_checkpoint)))
    teacher_action_state = teacher.act_model.state_dict()

    student = build_vla(student_config)
    student_action_state = student.act_model.state_dict()
    copied_action_state, report = copy_mapped_action_weights(teacher_action_state, student_action_state, layer_map)
    student.act_model.load_state_dict(copied_action_state, strict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state_dict(student), weights_path)
    torch.save(cpu_state_dict(student.act_model), output_dir / "action_head.pt")
    (output_dir / "student_config.json").write_text(json.dumps(student_config, indent=2), encoding="utf-8")
    report.update(
        {
            "student_config": str(student_config_path),
            "teacher_config": str(teacher_config_path),
            "teacher_checkpoint": str(teacher_checkpoint),
            "weights_path": str(weights_path),
            "action_head_path": str(output_dir / "action_head.pt"),
            "layer_map": layer_map,
        }
    )
    (output_dir / "init_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_config", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layer_map", default="0,2,4,6,8,10")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = create_student_checkpoint(
        student_config_path=args.student_config,
        teacher_config_path=args.teacher_config,
        teacher_checkpoint=args.teacher_checkpoint,
        output_dir=args.output_dir,
        layer_map=parse_layer_map(args.layer_map),
        force=args.force,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
