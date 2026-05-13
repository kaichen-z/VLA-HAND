#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vitra.guidance.metrics import compute_region_metrics, relative_l2, smoothness
from vitra.guidance.plotting import plot_guidance_loss_curve
from vitra.guidance.polynomial_guidance import (
    PolynomialGuidanceConfig,
    PolynomialRegionLoss,
    load_polynomial_guidance_config,
    tau_to_index,
)
from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.utils.config_utils import load_config


def resolve_weights_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "weights.pt" if path.is_dir() else path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config_path: Path, checkpoint: str | None):
    config = load_config(str(config_path))
    model = build_vla(config)
    if checkpoint and checkpoint != "none":
        model = load_vla_checkpoint(model, str(resolve_weights_path(checkpoint)))
    return model.eval().cuda(), config


def make_default_tensors(config: dict, mask_start: int, mask_end: int):
    state_dim = int(config.get("state_encoder", {}).get("state_dim", 212))
    action_dim = int(config["action_model"]["action_dim"])
    chunk_size = int(config.get("fwd_pred_next_n", 16))
    current_state = torch.zeros(1, state_dim, device="cuda", dtype=torch.float32)
    current_state_mask = torch.ones(1, state_dim, device="cuda", dtype=torch.bool)
    action_mask = torch.zeros(1, chunk_size, action_dim, device="cuda", dtype=torch.float32)
    action_mask[:, :, max(0, mask_start) : min(action_dim, mask_end)] = 1.0
    fov = torch.zeros(1, 2, device="cuda", dtype=torch.float32)
    return current_state, current_state_mask, action_mask, fov


def filter_config_for_suffix(config: PolynomialGuidanceConfig, start_idx: int, horizon: int) -> PolynomialGuidanceConfig:
    regions = [region for region in config.regions if tau_to_index(region.tau, horizon) >= start_idx]
    return PolynomialGuidanceConfig(
        guide_dims=list(config.guide_dims),
        regions=regions,
        loss_type=config.loss_type,
        eps=config.eps,
        temporal_mask=config.temporal_mask,
    )


def run_predict(
    model,
    image,
    instruction,
    current_state,
    current_state_mask,
    action_mask,
    fov,
    args,
    guidance_fn=None,
    fixed_actions=None,
    fixed_action_mask=None,
):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.predict_action(
        image=image,
        instruction=instruction,
        current_state=current_state,
        current_state_mask=current_state_mask,
        action_mask_torch=action_mask,
        fov=fov,
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
        fixed_action_mask=fixed_action_mask,
        return_replan_trace=fixed_actions is not None,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if guidance_fn is not None or fixed_actions is not None:
        action, trace = output
    else:
        action, trace = output, None
    return action[0].astype(np.float32), trace or [], elapsed


def make_prefix_tensors(action_norm: np.ndarray, action_mask: torch.Tensor, k: int):
    fixed_actions = torch.from_numpy(action_norm[None]).to(device=action_mask.device, dtype=action_mask.dtype)
    fixed_mask = torch.zeros_like(action_mask)
    fixed_mask[:, :k, :] = action_mask[:, :k, :]
    return fixed_actions, fixed_mask


def prefix_error(candidate: np.ndarray, reference: np.ndarray, action_mask: np.ndarray, k: int) -> float:
    mask = action_mask[0, :k, :].astype(np.float32)
    if mask.sum() == 0:
        return 0.0
    return float(np.max(np.abs(candidate[:k] - reference[:k]) * mask))


def boundary_jump(action: np.ndarray, k: int, dims: list[int]) -> float:
    if k <= 0 or k >= action.shape[0]:
        return 0.0
    return float(np.linalg.norm(action[k, dims] - action[k - 1, dims]))


def plot_replanning_trajectory(initial, replan_k5, replan_k10, config, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    dims = list(config.guide_dims)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(initial[:, dims[0]], initial[:, dims[1]], "o-", label="initial", alpha=0.75)
    ax.plot(replan_k5[:, dims[0]], replan_k5[:, dims[1]], "o-", label="replan K=5", alpha=0.75)
    ax.plot(replan_k10[:, dims[0]], replan_k10[:, dims[1]], "o-", label="replan K=10", alpha=0.75)

    all_xy = np.concatenate([initial[:, dims], replan_k5[:, dims], replan_k10[:, dims]], axis=0)
    lo = all_xy.min(axis=0) - 1.0
    hi = all_xy.max(axis=0) + 1.0
    xs = np.linspace(lo[0], hi[0], 180)
    ys = np.linspace(lo[1], hi[1], 180)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    for region in config.regions:
        c = np.asarray(region.center, dtype=np.float64).reshape(2)
        Q = np.asarray(region.Q, dtype=np.float64).reshape(2, 2)
        du = flat - c
        p = np.sum((du @ Q.T) * du, axis=-1).reshape(grid_x.shape) - float(region.radius2)
        ax.contour(grid_x, grid_y, p, levels=[0.0], linewidths=1.5)
        ax.text(c[0], c[1], region.name or f"tau={region.tau}", fontsize=8)

    ax.axvline(0, color="black", alpha=0.1)
    ax.set_xlabel(f"action dim {dims[0]}")
    ax.set_ylabel(f"action dim {dims[1]}")
    ax.set_title("Guided replanning trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_region_violation_bars(metrics: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["initial", "replan_k5", "replan_k10"]
    values = [metrics[label]["violation_mean"] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("mean region violation")
    ax.set_title("region_violation mean")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VITRA prefix-clamped guided replanning toy inference.")
    parser.add_argument("--config", type=Path, default=Path("vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json"))
    parser.add_argument("--checkpoint", default="checkpoints/vitra-vla-3b.pt")
    parser.add_argument("--image_path", type=Path, default=Path("examples/0002.jpg"))
    parser.add_argument("--instruction", default="Left hand: None. Right hand: Pick up the phone on the table.")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--regions_json", type=Path, default=Path("configs/polynomial_guidance_example.json"))
    parser.add_argument("--guide_dims", type=int, nargs=2, default=None)
    parser.add_argument("--guidance_scale", type=float, default=0.1)
    parser.add_argument("--guidance_start_frac", type=float, default=0.0)
    parser.add_argument("--guidance_end_frac", type=float, default=1.0)
    parser.add_argument("--guidance_grad_clip", type=float, default=5.0)
    parser.add_argument("--replan_indices", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_start", type=int, default=51)
    parser.add_argument("--mask_end", type=int, default=102)
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/replanning_guidance/debug_seed0"))
    args = parser.parse_args()

    if args.replan_indices != [5, 10]:
        raise ValueError("V0 report script expects --replan_indices 5 10.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    guidance_config = load_polynomial_guidance_config(args.regions_json, guide_dims=args.guide_dims)
    model, config = load_model(args.config, args.checkpoint)
    for param in model.parameters():
        param.requires_grad_(False)

    image = Image.open(args.image_path).convert("RGB")
    current_state, current_state_mask, action_mask, fov = make_default_tensors(
        config,
        mask_start=args.mask_start,
        mask_end=args.mask_end,
    )
    action_mask_np = action_mask.detach().cpu().numpy()

    set_seed(args.seed)
    initial, initial_trace, initial_time = run_predict(
        model,
        image,
        args.instruction,
        current_state,
        current_state_mask,
        action_mask,
        fov,
        args,
    )

    k5 = args.replan_indices[0]
    k5_config = filter_config_for_suffix(guidance_config, k5, initial.shape[0])
    fixed_actions, fixed_mask = make_prefix_tensors(initial, action_mask, k5)
    set_seed(args.seed + 5)
    replan_k5, trace_k5, time_k5 = run_predict(
        model,
        image,
        args.instruction,
        current_state,
        current_state_mask,
        action_mask,
        fov,
        args,
        guidance_fn=PolynomialRegionLoss(k5_config),
        fixed_actions=fixed_actions,
        fixed_action_mask=fixed_mask,
    )

    k10 = args.replan_indices[1]
    k10_config = filter_config_for_suffix(guidance_config, k10, initial.shape[0])
    fixed_actions, fixed_mask = make_prefix_tensors(replan_k5, action_mask, k10)
    set_seed(args.seed + 10)
    replan_k10, trace_k10, time_k10 = run_predict(
        model,
        image,
        args.instruction,
        current_state,
        current_state_mask,
        action_mask,
        fov,
        args,
        guidance_fn=PolynomialRegionLoss(k10_config),
        fixed_actions=fixed_actions,
        fixed_action_mask=fixed_mask,
    )

    dims = list(guidance_config.guide_dims)
    region_metrics = {
        "initial": compute_region_metrics(initial, guidance_config),
        "replan_k5": compute_region_metrics(replan_k5, guidance_config),
        "replan_k10": compute_region_metrics(replan_k10, guidance_config),
    }
    metrics = {
        "region_violation": region_metrics,
        "prefix_error": {
            "k5_vs_initial": prefix_error(replan_k5, initial, action_mask_np, k5),
            "k10_vs_replan_k5": prefix_error(replan_k10, replan_k5, action_mask_np, k10),
        },
        "suffix_delta": {
            "k5_suffix_relative_l2": relative_l2(replan_k5[k5:], initial[k5:]),
            "k10_suffix_relative_l2": relative_l2(replan_k10[k10:], replan_k5[k10:]),
        },
        "smoothness": {
            "initial": smoothness(initial),
            "replan_k5": smoothness(replan_k5),
            "replan_k10": smoothness(replan_k10),
            "boundary_jump_k5": boundary_jump(replan_k5, k5, dims),
            "boundary_jump_k10": boundary_jump(replan_k10, k10, dims),
        },
        "timing": {
            "initial_sec": initial_time,
            "replan_k5_sec": time_k5,
            "replan_k10_sec": time_k10,
            "total_sec": initial_time + time_k5 + time_k10,
            "num_ddim_steps": args.num_ddim_steps,
            "cfg_scale": args.cfg_scale,
        },
        "config": {
            "guide_dims": dims,
            "guidance_scale": args.guidance_scale,
            "replan_indices": args.replan_indices,
            "mask_start": args.mask_start,
            "mask_end": args.mask_end,
            "seed": args.seed,
        },
    }

    np.save(args.save_dir / "initial_action_norm.npy", initial)
    np.save(args.save_dir / "replan_k5_action_norm.npy", replan_k5)
    np.save(args.save_dir / "replan_k10_action_norm.npy", replan_k10)
    (args.save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.save_dir / "initial_trace.json").write_text(json.dumps(initial_trace, indent=2), encoding="utf-8")
    (args.save_dir / "replan_trace_k5.json").write_text(json.dumps(trace_k5, indent=2), encoding="utf-8")
    (args.save_dir / "replan_trace_k10.json").write_text(json.dumps(trace_k10, indent=2), encoding="utf-8")
    plot_replanning_trajectory(initial, replan_k5, replan_k10, guidance_config, args.save_dir / "trajectory_replanning_xy.png")
    plot_region_violation_bars(region_metrics, args.save_dir / "region_violation_bars.png")
    plot_guidance_loss_curve(trace_k5, args.save_dir / "guidance_loss_curve_k5.png")
    plot_guidance_loss_curve(trace_k10, args.save_dir / "guidance_loss_curve_k10.png")

    print(json.dumps(metrics, indent=2))
    print(f"Outputs written to {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
