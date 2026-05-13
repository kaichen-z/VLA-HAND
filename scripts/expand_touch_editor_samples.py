#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.touch_editor.cache_utils import build_future_mask
from scripts.cache_touch_editor_base_actions import observed_touch_stats


def parse_edit_start_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("At least one edit start index is required.")
    return sorted(set(indices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a Design 1 touch-editor cache by cloning samples across edit_start_idx values."
    )
    parser.add_argument("--input_cache_root", type=Path, required=True)
    parser.add_argument("--output_cache_root", type=Path, required=True)
    parser.add_argument(
        "--edit_start_indices",
        type=parse_edit_start_indices,
        default=parse_edit_start_indices("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"),
        help="Comma-separated edit_start_idx values to materialize.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar_to_python(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.tolist()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def assert_design1_record(record: dict[str, np.ndarray], path: Path) -> None:
    target_source = str(scalar_to_python(record.get("target_source", "")))
    if target_source != "opentouch_derived":
        raise ValueError(f"{path} is not a Design 1 OpenTouch cache record: target_source={target_source!r}")
    forbidden = [key for key in record if key.startswith("matched") or "gigahands" in key.lower()]
    if forbidden:
        raise ValueError(f"{path} contains pseudo-pair/GigaHands metadata: {forbidden}")


def expand_record(record: dict[str, np.ndarray], edit_start_idx: int, source_path: Path) -> dict[str, np.ndarray]:
    out = dict(record)
    out["residual_target"] = (np.asarray(out["a_target"], dtype=np.float32) - np.asarray(out["a_base"], dtype=np.float32)).astype(np.float32)
    out["future_mask"] = build_future_mask(out["action_mask"], edit_start_idx).astype(np.float32)
    out["edit_start_idx"] = np.asarray(edit_start_idx, dtype=np.int64)
    observed_len, contact_score, contact_delta = observed_touch_stats(out["touch_pressure"], out["touch_mask"], edit_start_idx)
    out["observed_touch_len"] = np.asarray(observed_len, dtype=np.int64)
    out["observed_touch_contact_score"] = np.asarray(contact_score, dtype=np.float32)
    out["observed_touch_contact_delta"] = np.asarray(contact_delta, dtype=np.float32)
    out["source_cache_path"] = np.asarray(str(source_path))
    if "edit_start_idx" in record:
        out["source_edit_start_idx"] = np.asarray(int(np.asarray(record["edit_start_idx"]).item()), dtype=np.int64)
    return out


def read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def write_sample(path: Path, record: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    np.savez(path, **record)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_cache_root.rglob("*.npz"))
    if not input_paths:
        raise FileNotFoundError(f"No .npz cache files found under {args.input_cache_root}")
    if args.output_cache_root.exists() and any(args.output_cache_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_cache_root} already exists and is non-empty; pass --overwrite to replace files.")
    args.output_cache_root.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for source_idx, source_path in enumerate(tqdm(input_paths, desc="expanding touch-editor cache")):
        record = load_npz(source_path)
        assert_design1_record(record, source_path)
        chunk_len = int(np.asarray(record["a_target"]).shape[0])
        metadata = read_sidecar(source_path)
        for edit_start_idx in args.edit_start_indices:
            if edit_start_idx < 0 or edit_start_idx > chunk_len:
                skipped += 1
                continue
            expanded = expand_record(record, edit_start_idx, source_path)
            out_name = f"sample_{source_idx:08d}_edit{edit_start_idx:02d}.npz"
            out_metadata = {
                **metadata,
                "source_cache_path": str(source_path),
                "source_cache_index": source_idx,
                "source_edit_start_idx": int(scalar_to_python(record.get("edit_start_idx", -1))),
                "edit_start_idx": int(edit_start_idx),
                "expanded_from_design1_cache": True,
            }
            write_sample(args.output_cache_root / out_name, expanded, out_metadata)
            written += 1

    summary = {
        "cache_root": str(args.output_cache_root),
        "input_cache_root": str(args.input_cache_root),
        "source_samples": len(input_paths),
        "num_samples": written,
        "skipped_out_of_range": skipped,
        "edit_start_indices": args.edit_start_indices,
        "target_source": "opentouch_derived",
        "touch_source": "opentouch",
        "expansion": "edit_start_idx",
    }
    source_summary = args.input_cache_root / "summary.json"
    if source_summary.exists():
        summary["source_summary"] = json.loads(source_summary.read_text(encoding="utf-8"))
    (args.output_cache_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
