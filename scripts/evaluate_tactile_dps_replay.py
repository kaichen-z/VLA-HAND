#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.guidance.tactile_dps import TactileDPSGuidance
from vitra.guidance.tactile_forward_model import TactileEncoder, TactileForwardModel
from vitra.guidance.tactile_losses import suffix_action_mask
from vitra.guidance.tactile_replay_dataset import TactileReplayCacheDataset, move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tactile DPS-style replay on cached OpenTouch/VITRA actions.")
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--subset", choices=("all", "high_contact"), default="all")
    parser.add_argument("--high_contact_quantile", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--edit_indices", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--num_guidance_steps", type=int, default=10)
    parser.add_argument("--guidance_lr", type=float, default=0.03)
    parser.add_argument("--lambda_embed", type=float, default=1.0)
    parser.add_argument("--lambda_stats", type=float, default=0.25)
    parser.add_argument("--lambda_prior", type=float, default=0.01)
    parser.add_argument("--lambda_boundary", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--touch_scale", type=float, default=None)
    parser.add_argument("--ablations", nargs="+", default=["matched", "shuffled_touch", "zero_touch", "stats_only"])
    return parser.parse_args()


def load_measurement_checkpoint(path: Path, device: str) -> tuple[TactileEncoder, TactileForwardModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    encoder = TactileEncoder(
        embed_dim=int(ckpt_args.get("embed_dim", 64)),
        stat_dim=int(ckpt_args.get("stat_dim", 8)),
        hidden_dim=int(ckpt_args.get("encoder_hidden_dim", 128)),
    ).to(device)
    forward_model = TactileForwardModel(
        action_dim=192,
        state_dim=212,
        embed_dim=int(ckpt_args.get("embed_dim", 64)),
        stat_dim=int(ckpt_args.get("stat_dim", 8)),
        hidden_dim=int(ckpt_args.get("forward_hidden_dim", 256)),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    forward_model.load_state_dict(checkpoint["forward_model"])
    encoder.eval()
    forward_model.eval()
    for module in (encoder, forward_model):
        for param in module.parameters():
            param.requires_grad_(False)
    return encoder, forward_model, ckpt_args


def shuffled_touch(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = int(batch["touch_pressure"].shape[0])
    if bsz <= 1:
        return batch["touch_pressure"], batch["touch_mask"]
    order = torch.roll(torch.arange(bsz, device=batch["touch_pressure"].device), shifts=1)
    return batch["touch_pressure"][order], batch["touch_mask"][order]


def ablated_touch(batch: dict[str, torch.Tensor], ablation: str) -> tuple[torch.Tensor, torch.Tensor]:
    if ablation == "matched" or ablation == "stats_only":
        return batch["touch_pressure"], batch["touch_mask"]
    if ablation == "shuffled_touch":
        return shuffled_touch(batch)
    if ablation == "zero_touch":
        return torch.zeros_like(batch["touch_pressure"]), torch.zeros_like(batch["touch_mask"])
    raise ValueError(f"Unknown ablation: {ablation}")


def masked_sse_count(diff: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(diff.dtype)
    return float((diff.square() * mask).sum().detach().cpu()), float(mask.sum().detach().cpu())


def prefix_change_l2(a_edit: torch.Tensor, a_base: torch.Tensor, action_mask: torch.Tensor, first_edit_idx: int) -> float:
    prefix_mask = torch.zeros_like(action_mask, dtype=torch.float32)
    prefix_mask[:, :first_edit_idx] = action_mask[:, :first_edit_idx].to(torch.float32)
    denom = prefix_mask.any(dim=-1).sum().clamp_min(1)
    return float((((a_edit - a_base) * prefix_mask).square().sum(dim=-1).sqrt().sum() / denom).detach().cpu())


def apply_guidance_schedule(
    *,
    encoder: TactileEncoder,
    forward_model: TactileForwardModel,
    batch: dict[str, torch.Tensor],
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    edit_indices: list[int],
    num_steps: int,
    guidance_lr: float,
    gradient_clip_norm: float,
    lambda_embed: float,
    lambda_stats: float,
    lambda_prior: float,
    lambda_boundary: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    a_work = batch["a_base"].detach()
    traces: list[dict[str, float]] = []
    for edit_idx in edit_indices:
        old_action = a_work.detach()
        for step in range(num_steps):
            x_in = a_work.detach().requires_grad_(True)
            guidance = TactileDPSGuidance(
                tactile_encoder=encoder,
                tactile_forward_model=forward_model,
                current_state=batch["current_state"],
                touch_pressure=touch_pressure,
                touch_mask=touch_mask,
                chunk_phase=batch["chunk_phase"],
                edit_start_idx=edit_idx,
                action_mask=batch["action_mask"],
                old_action=old_action,
                lambda_embed=lambda_embed,
                lambda_stats=lambda_stats,
                lambda_prior=lambda_prior,
                lambda_boundary=lambda_boundary,
            )
            grad, loss, metrics = guidance.gradient(x_in, x_in)
            if gradient_clip_norm > 0:
                norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1)
                grad = grad * (float(gradient_clip_norm) / (norm + 1e-6)).clamp(max=1.0)
            editable = suffix_action_mask(batch["action_mask"], edit_idx).to(a_work.dtype)
            next_action = x_in.detach() - float(guidance_lr) * grad.detach()
            a_work = next_action * editable + old_action * (1.0 - editable)
            if step == num_steps - 1:
                trace = {
                    "edit_idx": float(edit_idx),
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad.detach().flatten(1).norm(dim=1).mean().cpu()),
                }
                for key, value in metrics.items():
                    trace[key] = float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else float(value)
                traces.append(trace)
    return a_work.detach(), traces


def evaluate_ablation(
    *,
    encoder: TactileEncoder,
    forward_model: TactileForwardModel,
    loader: DataLoader,
    args: argparse.Namespace,
    ablation: str,
) -> dict[str, Any]:
    totals = {
        "base_sse": 0.0,
        "edit_sse": 0.0,
        "count": 0.0,
        "prefix_change_l2_sum": 0.0,
        "prefix_change_l2_count": 0.0,
        "latency_sum": 0.0,
        "latency_count": 0.0,
    }
    traces: list[dict[str, float]] = []
    first_edit_idx = min(args.edit_indices)
    for batch in tqdm(loader, desc=f"eval tactile DPS ({ablation})"):
        batch = move_batch_to_device(batch, args.device)
        touch_pressure, touch_mask = ablated_touch(batch, ablation)
        touch_scale = float(args.touch_scale)
        touch_pressure = touch_pressure / touch_scale
        lambda_embed = 0.0 if ablation == "stats_only" else args.lambda_embed
        start = time.perf_counter()
        a_edit, batch_traces = apply_guidance_schedule(
            encoder=encoder,
            forward_model=forward_model,
            batch=batch,
            touch_pressure=touch_pressure,
            touch_mask=touch_mask,
            edit_indices=list(args.edit_indices),
            num_steps=args.num_guidance_steps,
            guidance_lr=args.guidance_lr,
            gradient_clip_norm=args.gradient_clip_norm,
            lambda_embed=lambda_embed,
            lambda_stats=args.lambda_stats,
            lambda_prior=args.lambda_prior,
            lambda_boundary=args.lambda_boundary,
        )
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        latency = time.perf_counter() - start
        eval_mask = suffix_action_mask(batch["action_mask"], first_edit_idx)
        base_sse, count = masked_sse_count(batch["a_base"] - batch["a_target"], eval_mask)
        edit_sse, _ = masked_sse_count(a_edit - batch["a_target"], eval_mask)
        totals["base_sse"] += base_sse
        totals["edit_sse"] += edit_sse
        totals["count"] += count
        totals["prefix_change_l2_sum"] += prefix_change_l2(a_edit, batch["a_base"], batch["action_mask"], first_edit_idx) * int(batch["a_base"].shape[0])
        totals["prefix_change_l2_count"] += int(batch["a_base"].shape[0])
        totals["latency_sum"] += latency
        totals["latency_count"] += 1
        traces.extend(batch_traces[:2])

    base_mse = totals["base_sse"] / max(totals["count"], 1.0)
    edit_mse = totals["edit_sse"] / max(totals["count"], 1.0)
    return {
        "ablation": ablation,
        "base_mse": base_mse,
        "guided_mse": edit_mse,
        "improvement_pct": (base_mse - edit_mse) / base_mse * 100.0 if base_mse > 0 else 0.0,
        "valid_action_dims": totals["count"],
        "prefix_change_l2": totals["prefix_change_l2_sum"] / max(totals["prefix_change_l2_count"], 1.0),
        "latency_ms_per_batch": totals["latency_sum"] / max(totals["latency_count"], 1.0) * 1000.0,
        "example_traces": traces[:8],
    }


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    high_contact_quantile = args.high_contact_quantile if args.subset == "high_contact" else None
    dataset = TactileReplayCacheDataset(args.cache_root, max_samples=args.max_samples, high_contact_quantile=high_contact_quantile)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    encoder, forward_model, ckpt_args = load_measurement_checkpoint(args.checkpoint, args.device)
    if args.touch_scale is None:
        args.touch_scale = float(ckpt_args.get("touch_scale", 1000.0))
    results = [
        evaluate_ablation(
            encoder=encoder,
            forward_model=forward_model,
            loader=loader,
            args=args,
            ablation=ablation,
        )
        for ablation in args.ablations
    ]
    summary = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_args": ckpt_args,
        "num_samples": len(dataset),
        "results": results,
    }
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
