"""Verify that a GigaHands RGB subset video list exists under a dataset root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_video_list(video_list: str | Path) -> list[str]:
    path = Path(video_list)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_required_videos(
    root: str | Path,
    video_list: str | Path,
    unique_output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    listed = load_video_list(video_list)
    unique = sorted(dict.fromkeys(listed))
    missing_paths = [path for path in unique if not (root / path).exists()]
    present = len(unique) - len(missing_paths)

    if unique_output is not None:
        unique_path = Path(unique_output)
        unique_path.parent.mkdir(parents=True, exist_ok=True)
        unique_path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")

    return {
        "root": str(root),
        "video_list": str(video_list),
        "listed": len(listed),
        "needed": len(unique),
        "present": present,
        "missing": len(missing_paths),
        "missing_paths": missing_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--video_list", type=Path, required=True)
    parser.add_argument("--unique_output", type=Path)
    parser.add_argument("--fail_on_missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_required_videos(args.root, args.video_list, unique_output=args.unique_output)
    print(json.dumps(report, indent=2))
    if args.fail_on_missing and report["missing"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
