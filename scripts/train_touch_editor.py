#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
    parser.add_argument("--lambda_dev", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--contact_subset", choices=("all", "high_contact"), default="all")
    parser.add_argument("--high_contact_quantile", type=float, default=0.75)
    parser.add_argument(
        "--touch_ablation",
        choices=("none", "zero_touch"),
        default="none",
        help="Optionally ablate tactile input during training. Use zero_touch for a no-touch editor.",
    )
    return parser.parse_args()


def _score_cache_path(path: Path) -> float:
    with np.load(path, allow_pickle=False) as payload:
        contact_score = float(np.asarray(payload.get("observed_touch_contact_score", 0.0)).item())
        contact_delta = float(np.asarray(payload.get("observed_touch_contact_delta", 0.0)).item())
    return max(contact_score, contact_delta)


def filter_dataset_by_observed_contact(
    dataset: TouchEditorCacheDataset,
    *,
    quantile: float,
) -> TouchEditorCacheDataset:
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    scored_paths = [(path, _score_cache_path(Path(path))) for path in dataset.paths]
    if not scored_paths:
        return dataset
    threshold = torch.quantile(torch.tensor([score for _, score in scored_paths], dtype=torch.float32), float(quantile)).item()
    filtered_paths = [path for path, score in scored_paths if score >= threshold]
    filtered = object.__new__(TouchEditorCacheDataset)
    filtered.cache_root = dataset.cache_root
    filtered.paths = filtered_paths
    return filtered


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def apply_touch_training_ablation(
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    touch_ablation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if touch_ablation == "zero_touch":
        return torch.zeros_like(touch_pressure), torch.zeros_like(touch_mask)
    return touch_pressure, touch_mask


def causal_touch_history_from_batch(
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    edit_start_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only touch observations available at each sample's edit time.

    ``edit_start_idx`` is inclusive: editing at index 3 may use tactile
    observations from indices 0, 1, 2, and 3, but never indices 4+.
    """
    if touch_pressure.ndim != 5:
        raise ValueError(f"touch_pressure must be [B,T,2,16,16], got {tuple(touch_pressure.shape)}")
    if touch_mask.ndim != 3:
        raise ValueError(f"touch_mask must be [B,T,2], got {tuple(touch_mask.shape)}")
    if touch_pressure.shape[:2] != touch_mask.shape[:2]:
        raise ValueError("touch_pressure and touch_mask batch/time dimensions must match")
    if touch_pressure.shape[0] == 0:
        return touch_pressure, touch_mask

    edit_start_idx = torch.as_tensor(edit_start_idx, device=touch_pressure.device, dtype=torch.long).reshape(-1)
    if edit_start_idx.numel() == 1 and touch_pressure.shape[0] > 1:
        edit_start_idx = edit_start_idx.expand(touch_pressure.shape[0])
    if edit_start_idx.numel() != touch_pressure.shape[0]:
        raise ValueError(
            f"edit_start_idx must have batch size {touch_pressure.shape[0]}, got {edit_start_idx.numel()}"
        )

    max_time = int(touch_pressure.shape[1])
    observed_lens = (edit_start_idx.clamp(min=0, max=max_time - 1) + 1).clamp(min=1, max=max_time)
    max_observed_len = int(observed_lens.max().item())
    history = touch_pressure[:, :max_observed_len].clone()
    history_mask = touch_mask[:, :max_observed_len].clone()
    time = torch.arange(max_observed_len, device=touch_pressure.device)[None, :]
    valid_history = time < observed_lens[:, None]
    history = history * valid_history[:, :, None, None, None].to(history.dtype)
    history_mask = history_mask & valid_history[:, :, None]
    return history, history_mask


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = TouchEditorCacheDataset(args.cache_root)
    if args.contact_subset == "high_contact":
        dataset = filter_dataset_by_observed_contact(dataset, quantile=args.high_contact_quantile)
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
                touch_pressure, touch_mask = apply_touch_training_ablation(
                    batch["touch_pressure"],
                    batch["touch_mask"],
                    args.touch_ablation,
                )
                touch_pressure, touch_mask = causal_touch_history_from_batch(
                    touch_pressure,
                    touch_mask,
                    batch["edit_start_idx"],
                )
                delta = model(
                    batch["a_base"],
                    batch["current_state"],
                    batch["current_state_mask"],
                    touch_pressure,
                    touch_mask,
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
