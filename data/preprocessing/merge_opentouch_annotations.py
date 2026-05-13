#!/usr/bin/env python
"""Merge extracted OpenTouch annotation CSV files into one converter label file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merged_rows(annotations_dir: str | Path) -> list[dict[str, Any]]:
    annotations_dir = Path(annotations_dir)
    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations_dir does not exist: {annotations_dir}")
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(annotations_dir.glob("*.csv")):
        source_stem = csv_path.stem
        for row in read_rows(csv_path):
            merged = dict(row)
            clip_id = str(merged.get("clip_id") or merged.get("id") or "").strip()
            if "::" not in clip_id and clip_id:
                merged["clip_id"] = f"{source_stem}::{clip_id}"
            else:
                merged["clip_id"] = clip_id
            merged["annotation_file"] = csv_path.name
            rows.append(merged)
    return rows


def write_rows(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if "clip_id" in fieldnames:
        fieldnames.insert(0, fieldnames.pop(fieldnames.index("clip_id")))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merged_rows(args.annotations_dir)
    write_rows(rows, args.output_path)
    print(f"Wrote {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
