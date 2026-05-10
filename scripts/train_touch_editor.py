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
from vitra.touch_editor.losses import touch_editor_loss
from vitra.touch_editor.model import ResidualTouchEditor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the residual OpenTouch editor from cached VLA predictions.")
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/touch_editor"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="Optional editor checkpoint to initialize from.")
    parser.add_argument("--lambda_dev", type=float, default=0.0, help="Deprecated duplicate residual-size penalty; prefer lambda_delta.")
    parser.add_argument("--lambda_delta", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = TouchEditorCacheDataset(args.cache_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    model = ResidualTouchEditor().to(args.device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    metrics_path = args.output_dir / "metrics.jsonl"
    step = 0
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        progress = tqdm(total=args.max_steps, desc="training touch editor")
        while step < args.max_steps:
            for batch in loader:
                batch = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in batch.items()}
                delta = model(
                    batch["a_base"],
                    batch["current_state"],
                    batch["current_state_mask"],
                    batch["touch_pressure"],
                    batch["touch_mask"],
                    batch["chunk_phase"],
                    batch["future_mask"],
                    batch["action_mask"],
                )
                losses = touch_editor_loss(
                    batch["a_base"],
                    batch["a_target"],
                    delta,
                    batch["action_mask"],
                    batch["future_mask"],
                    residual_target=batch["residual_target"],
                    lambda_dev=args.lambda_dev,
                    lambda_delta=args.lambda_delta,
                    lambda_smooth=args.lambda_smooth,
                    lambda_mask=args.lambda_mask,
                )
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                record = {key: float(value.detach().cpu()) for key, value in losses.items() if key.startswith("loss")}
                record["step"] = step
                metrics_file.write(json.dumps(record) + "\n")
                if step % 100 == 0:
                    torch.save({"model": model.state_dict(), "step": step, "args": serializable_args(args)}, args.output_dir / "latest.pt")
                step += 1
                progress.update(1)
                if step >= args.max_steps:
                    break
        progress.close()
    torch.save({"model": model.state_dict(), "step": step, "args": serializable_args(args)}, args.output_dir / "latest.pt")


if __name__ == "__main__":
    main()
