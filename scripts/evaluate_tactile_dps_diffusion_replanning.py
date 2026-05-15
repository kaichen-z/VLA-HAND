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
from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.utils.config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tactile DPS guidance inside VITRA DDIM replanning.")
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--measurement_checkpoint", type=Path, required=True)
    parser.add_argument("--vla_config", type=Path, required=True)
    parser.add_argument("--vla_checkpoint", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--subset", choices=("all", "high_contact"), default="all")
    parser.add_argument("--high_contact_quantile", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_start_frac", type=float, default=0.0)
    parser.add_argument("--guidance_end_frac", type=float, default=1.0)
    parser.add_argument("--guidance_grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_embed", type=float, default=1.0)
    parser.add_argument("--lambda_stats", type=float, default=0.25)
    parser.add_argument("--lambda_prior", type=float, default=0.01)
    parser.add_argument("--lambda_boundary", type=float, default=0.05)
    parser.add_argument("--touch_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ablations", nargs="+", default=["matched", "shuffled_touch", "zero_touch", "stats_only"])
    return parser.parse_args()


def resolve_weights_path(path: Path) -> Path:
    return path / "weights.pt" if path.is_dir() else path


def sync(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_vla_for_diffusion_replanning(config_path: Path, checkpoint_path: Path, device: str):
    config = load_config(str(config_path))
    model = build_vla(config)
    model = load_vla_checkpoint(model, str(resolve_weights_path(checkpoint_path)))
    model = model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config


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
    if ablation in {"matched", "stats_only"}:
        return batch["touch_pressure"], batch["touch_mask"]
    if ablation == "shuffled_touch":
        return shuffled_touch(batch)
    if ablation == "zero_touch":
        return torch.zeros_like(batch["touch_pressure"]), torch.zeros_like(batch["touch_mask"])
    raise ValueError(f"Unknown ablation: {ablation}")


def make_prefix_tensors(a_base: torch.Tensor, action_mask: torch.Tensor, edit_start_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_mask = torch.zeros_like(action_mask, dtype=a_base.dtype)
    fixed_mask[:, : int(edit_start_idx)] = action_mask[:, : int(edit_start_idx)].to(a_base.dtype)
    return a_base, fixed_mask


def build_tactile_guidance(
    *,
    encoder: TactileEncoder,
    forward_model: TactileForwardModel,
    batch: dict[str, torch.Tensor],
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    edit_start_idx: int,
    lambda_embed: float,
    lambda_stats: float,
    lambda_prior: float,
    lambda_boundary: float,
) -> TactileDPSGuidance:
    return TactileDPSGuidance(
        tactile_encoder=encoder,
        tactile_forward_model=forward_model,
        current_state=batch["current_state"],
        touch_pressure=touch_pressure,
        touch_mask=touch_mask,
        chunk_phase=batch["chunk_phase"],
        edit_start_idx=int(edit_start_idx),
        action_mask=batch["action_mask"],
        old_action=batch["a_base"],
        lambda_embed=lambda_embed,
        lambda_stats=lambda_stats,
        lambda_prior=lambda_prior,
        lambda_boundary=lambda_boundary,
    )


def suffix_mask_per_sample(action_mask: torch.Tensor, edit_start_idx: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(action_mask, dtype=torch.bool)
    for row, start in enumerate(edit_start_idx.detach().cpu().tolist()):
        out[row, int(start) :] = action_mask[row, int(start) :]
    return out


def masked_sse_count(diff: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(diff.dtype)
    return float((diff.square() * mask).sum().detach().cpu()), float(mask.sum().detach().cpu())


def prefix_error(a_edit: torch.Tensor, a_base: torch.Tensor, action_mask: torch.Tensor, edit_start_idx: torch.Tensor) -> float:
    prefix = torch.zeros_like(action_mask, dtype=torch.bool)
    for row, start in enumerate(edit_start_idx.detach().cpu().tolist()):
        prefix[row, : int(start)] = action_mask[row, : int(start)]
    if not prefix.any():
        return 0.0
    return float(((a_edit - a_base).abs() * prefix.to(a_edit.dtype)).max().detach().cpu())


def sample_replanned_action(
    *,
    model,
    batch: dict[str, torch.Tensor],
    guidance_fn: TactileDPSGuidance | None,
    edit_start_idx: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    fixed_actions, fixed_mask = make_prefix_tensors(batch["a_base"], batch["action_mask"], edit_start_idx)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    sync(args.device)
    start = time.perf_counter()
    action_np, trace = model.sample_action_from_condition(
        action_features=batch["action_features"],
        current_state=batch["current_state"],
        current_state_mask=batch["current_state_mask"],
        action_mask_torch=batch["action_mask"].to(torch.float32),
        sample_times=1,
        num_ddim_steps=args.num_ddim_steps,
        cfg_scale=args.cfg_scale,
        guidance_fn=guidance_fn,
        guidance_scale=args.guidance_scale if guidance_fn is not None else 0.0,
        guidance_start_frac=args.guidance_start_frac,
        guidance_end_frac=args.guidance_end_frac,
        guidance_grad_clip=args.guidance_grad_clip,
        return_guidance_trace=guidance_fn is not None,
        fixed_actions=fixed_actions,
        fixed_action_mask=fixed_mask,
        return_replan_trace=True,
    )
    sync(args.device)
    elapsed = time.perf_counter() - start
    return torch.as_tensor(action_np, device=batch["a_base"].device, dtype=batch["a_base"].dtype), trace or [], elapsed


def evaluate_group(
    *,
    model,
    encoder: TactileEncoder,
    forward_model: TactileForwardModel,
    batch: dict[str, torch.Tensor],
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    edit_start_idx: int,
    ablation: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], float, float]:
    group_size = int(batch["a_base"].shape[0])
    unguided, _, unguided_elapsed = sample_replanned_action(
        model=model,
        batch=batch,
        guidance_fn=None,
        edit_start_idx=edit_start_idx,
        args=args,
        seed=seed,
    )
    guidance = build_tactile_guidance(
        encoder=encoder,
        forward_model=forward_model,
        batch=batch,
        touch_pressure=touch_pressure,
        touch_mask=touch_mask,
        edit_start_idx=edit_start_idx,
        lambda_embed=0.0 if ablation == "stats_only" else args.lambda_embed,
        lambda_stats=args.lambda_stats,
        lambda_prior=args.lambda_prior,
        lambda_boundary=args.lambda_boundary,
    )
    guided, trace, guided_elapsed = sample_replanned_action(
        model=model,
        batch=batch,
        guidance_fn=guidance,
        edit_start_idx=edit_start_idx,
        args=args,
        seed=seed,
    )
    return unguided, guided, trace[:2], unguided_elapsed / max(group_size, 1), guided_elapsed / max(group_size, 1)


def batch_subset(batch: dict[str, Any], rows: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    row_list = rows.detach().cpu().tolist()
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(batch["a_base"]):
            out[key] = value[rows]
        elif isinstance(value, list) and len(value) == len(batch["a_base"]):
            out[key] = [value[i] for i in row_list]
        else:
            out[key] = value
    return out


def evaluate_ablation(
    *,
    model,
    encoder: TactileEncoder,
    forward_model: TactileForwardModel,
    loader: DataLoader,
    args: argparse.Namespace,
    ablation: str,
) -> dict[str, Any]:
    totals = {
        "base_sse": 0.0,
        "unguided_sse": 0.0,
        "guided_sse": 0.0,
        "count": 0.0,
        "prefix_error_max": 0.0,
        "unguided_latency_sum": 0.0,
        "guided_latency_sum": 0.0,
        "latency_count": 0.0,
    }
    traces: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"diffusion DPS ({ablation})")):
        batch = move_batch_to_device(batch, args.device)
        touch_pressure, touch_mask = ablated_touch(batch, ablation)
        if args.touch_scale:
            touch_pressure = touch_pressure / float(args.touch_scale)
        edit_indices = batch["edit_start_idx"].reshape(-1).to(torch.long)
        eval_mask = suffix_mask_per_sample(batch["action_mask"], edit_indices)
        base_sse, count = masked_sse_count(batch["a_base"] - batch["a_target"], eval_mask)
        totals["base_sse"] += base_sse
        totals["count"] += count

        guided_full = torch.empty_like(batch["a_base"])
        unguided_full = torch.empty_like(batch["a_base"])
        for edit_idx in sorted({int(item) for item in edit_indices.detach().cpu().tolist()}):
            rows = torch.where(edit_indices == edit_idx)[0]
            sub = batch_subset(batch, rows)
            sub_touch_pressure = touch_pressure[rows]
            sub_touch_mask = touch_mask[rows]
            unguided, guided, trace, unguided_latency, guided_latency = evaluate_group(
                model=model,
                encoder=encoder,
                forward_model=forward_model,
                batch=sub,
                touch_pressure=sub_touch_pressure,
                touch_mask=sub_touch_mask,
                edit_start_idx=edit_idx,
                ablation=ablation,
                args=args,
                seed=int(args.seed + batch_idx * 997 + edit_idx),
            )
            unguided_full[rows] = unguided
            guided_full[rows] = guided
            traces.extend(trace)
            totals["unguided_latency_sum"] += unguided_latency * len(rows)
            totals["guided_latency_sum"] += guided_latency * len(rows)
            totals["latency_count"] += len(rows)

        unguided_sse, _ = masked_sse_count(unguided_full - batch["a_target"], eval_mask)
        guided_sse, _ = masked_sse_count(guided_full - batch["a_target"], eval_mask)
        totals["unguided_sse"] += unguided_sse
        totals["guided_sse"] += guided_sse
        totals["prefix_error_max"] = max(
            totals["prefix_error_max"],
            prefix_error(guided_full, batch["a_base"], batch["action_mask"], edit_indices),
        )

    base_mse = totals["base_sse"] / max(totals["count"], 1.0)
    unguided_mse = totals["unguided_sse"] / max(totals["count"], 1.0)
    guided_mse = totals["guided_sse"] / max(totals["count"], 1.0)
    return {
        "ablation": ablation,
        "base_mse": base_mse,
        "unguided_replan_mse": unguided_mse,
        "guided_replan_mse": guided_mse,
        "guided_vs_unguided_delta": guided_mse - unguided_mse,
        "guided_vs_base_improvement_pct": (base_mse - guided_mse) / base_mse * 100.0 if base_mse > 0 else 0.0,
        "valid_action_dims": totals["count"],
        "prefix_error_max": totals["prefix_error_max"],
        "unguided_latency_ms_per_sample": totals["unguided_latency_sum"] / max(totals["latency_count"], 1.0) * 1000.0,
        "guided_latency_ms_per_sample": totals["guided_latency_sum"] / max(totals["latency_count"], 1.0) * 1000.0,
        "example_traces": traces[:8],
    }


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    high_contact_quantile = args.high_contact_quantile if args.subset == "high_contact" else None
    dataset = TactileReplayCacheDataset(
        args.cache_root,
        max_samples=args.max_samples,
        high_contact_quantile=high_contact_quantile,
        require_action_features=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    model, vla_config = load_vla_for_diffusion_replanning(args.vla_config, args.vla_checkpoint, args.device)
    encoder, forward_model, ckpt_args = load_measurement_checkpoint(args.measurement_checkpoint, args.device)
    if args.touch_scale is None:
        args.touch_scale = float(ckpt_args.get("touch_scale", 1000.0))

    results = [
        evaluate_ablation(
            model=model,
            encoder=encoder,
            forward_model=forward_model,
            loader=loader,
            args=args,
            ablation=ablation,
        )
        for ablation in args.ablations
    ]
    by_name = {item["ablation"]: item for item in results}
    matched = by_name.get("matched", {})
    shuffled = by_name.get("shuffled_touch", {})
    summary = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_args": ckpt_args,
        "vla_action_dim": int(vla_config["action_model"]["action_dim"]),
        "num_samples": len(dataset),
        "results": results,
        "matched_vs_shuffled_guided_gap": (
            float(shuffled["guided_replan_mse"] - matched["guided_replan_mse"])
            if matched and shuffled
            else None
        ),
    }
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
