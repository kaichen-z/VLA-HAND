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
from vitra.guidance.plotting import plot_guidance_loss_curve, plot_trajectory_xy
from vitra.guidance.polynomial_guidance import PolynomialRegionLoss, load_polynomial_guidance_config
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


def make_default_tensors(config: dict, sample_times: int, mask_start: int, mask_end: int):
    state_dim = int(config.get("state_encoder", {}).get("state_dim", 212))
    action_dim = int(config["action_model"]["action_dim"])
    chunk_size = int(config.get("fwd_pred_next_n", 16))
    current_state = torch.zeros(1, state_dim, device="cuda", dtype=torch.float32)
    current_state_mask = torch.ones(1, state_dim, device="cuda", dtype=torch.bool)
    action_mask = torch.zeros(1, chunk_size, action_dim, device="cuda", dtype=torch.float32)
    action_mask[:, :, max(0, mask_start) : min(action_dim, mask_end)] = 1.0
    fov = torch.zeros(1, 2, device="cuda", dtype=torch.float32)
    return current_state, current_state_mask, action_mask, fov


def run_predict(model, image, instruction, current_state, current_state_mask, action_mask, fov, args, guidance_fn=None):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.predict_action(
        image=image,
        instruction=instruction,
        current_state=current_state,
        current_state_mask=current_state_mask,
        action_mask_torch=action_mask,
        fov=fov,
        sample_times=args.sample_times,
        num_ddim_steps=args.num_ddim_steps,
        cfg_scale=args.cfg_scale,
        guidance_fn=guidance_fn,
        guidance_scale=args.guidance_scale if guidance_fn is not None else 0.0,
        guidance_start_frac=args.guidance_start_frac,
        guidance_end_frac=args.guidance_end_frac,
        guidance_grad_clip=args.guidance_grad_clip,
        return_guidance_trace=guidance_fn is not None,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if guidance_fn is not None:
        action, trace = output
    else:
        action, trace = output, None
    return action, trace, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VITRA polynomial-region inference-time guidance.")
    parser.add_argument("--config", type=Path, default=Path("vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json"))
    parser.add_argument("--checkpoint", default="checkpoints/vitra-vla-3b.pt")
    parser.add_argument("--image_path", type=Path, default=Path("examples/0002.jpg"))
    parser.add_argument("--instruction", default="Left hand: None. Right hand: Pick up the phone on the table.")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--regions_json", type=Path, default=Path("configs/polynomial_guidance_example.json"))
    parser.add_argument("--guide_dims", type=int, nargs=2, default=None)
    parser.add_argument("--guidance_scale", type=float, default=0.1)
    parser.add_argument("--guidance_start_frac", type=float, default=0.0)
    parser.add_argument("--guidance_end_frac", type=float, default=1.0)
    parser.add_argument("--guidance_grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_start", type=int, default=51)
    parser.add_argument("--mask_end", type=int, default=102)
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/poly_guidance/debug_seed0"))
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    guidance_config = load_polynomial_guidance_config(args.regions_json, guide_dims=args.guide_dims)
    guidance_fn = PolynomialRegionLoss(guidance_config)
    model, config = load_model(args.config, args.checkpoint)
    for param in model.parameters():
        param.requires_grad_(False)

    image = Image.open(args.image_path).convert("RGB")
    current_state, current_state_mask, action_mask, fov = make_default_tensors(
        config,
        sample_times=args.sample_times,
        mask_start=args.mask_start,
        mask_end=args.mask_end,
    )

    set_seed(args.seed)
    baseline, _, baseline_time = run_predict(
        model,
        image,
        args.instruction,
        current_state,
        current_state_mask,
        action_mask,
        fov,
        args,
        guidance_fn=None,
    )
    set_seed(args.seed)
    guided, trace, guided_time = run_predict(
        model,
        image,
        args.instruction,
        current_state,
        current_state_mask,
        action_mask,
        fov,
        args,
        guidance_fn=guidance_fn,
    )

    baseline_norm = baseline[0].astype(np.float32)
    guided_norm = guided[0].astype(np.float32)
    baseline_metrics = compute_region_metrics(baseline_norm, guidance_config)
    guided_metrics = compute_region_metrics(guided_norm, guidance_config)
    comparison = {
        "delta_violation_mean": guided_metrics["violation_mean"] - baseline_metrics["violation_mean"],
        "relative_l2": relative_l2(guided_norm, baseline_norm),
        "smoothness_baseline": smoothness(baseline_norm),
        "smoothness_guided": smoothness(guided_norm),
        "smoothness_ratio": smoothness(guided_norm) / (smoothness(baseline_norm) + 1e-8),
        "max_abs_guided_action": float(np.abs(guided_norm).max()),
    }
    metrics = {
        "baseline": baseline_metrics,
        "guided": guided_metrics,
        "comparison": comparison,
        "timing": {
            "baseline_sec": baseline_time,
            "guided_sec": guided_time,
            "guided_over_baseline": guided_time / max(baseline_time, 1e-8),
            "num_ddim_steps": args.num_ddim_steps,
            "cfg_scale": args.cfg_scale,
        },
    }

    np.save(args.save_dir / "baseline_action_norm.npy", baseline_norm)
    np.save(args.save_dir / "guided_action_norm.npy", guided_norm)
    (args.save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.save_dir / "guidance_trace.json").write_text(json.dumps(trace or [], indent=2), encoding="utf-8")
    plot_trajectory_xy(baseline_norm, guided_norm, guidance_config, args.save_dir / "trajectory_xy.png")
    plot_guidance_loss_curve(trace or [], args.save_dir / "guidance_loss_curve.png")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs written to {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
