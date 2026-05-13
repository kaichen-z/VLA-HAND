#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vitra.guidance.metrics import compute_region_metrics, relative_l2
from vitra.guidance.plotting import plot_guidance_loss_curve
from vitra.guidance.polynomial_guidance import (
    PolynomialGuidanceConfig,
    PolynomialRegionLoss,
    load_polynomial_guidance_config,
    tau_to_index,
)
from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.utils.config_utils import load_config


MODEL_SPECS = {
    "base_vitra3b": {
        "config": "vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json",
        "checkpoint": "checkpoints/vitra-vla-3b.pt",
    },
    "joint_kd_student": {
        "config": "vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano.json",
        "checkpoint": (
            "runs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano/checkpoints/"
            "finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano_TB2_B1_bf16True/"
            "checkpoints/epoch=0-step=50000.ckpt/weights.pt"
        ),
    },
}


def resolve_weights_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "weights.pt" if path.is_dir() else path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn):
    sync()
    start = time.perf_counter()
    output = fn()
    sync()
    return output, time.perf_counter() - start


def load_model(config_path: Path, checkpoint: str | Path | None):
    config = load_config(str(config_path))
    model = build_vla(config)
    if checkpoint and str(checkpoint) != "none":
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


def plot_trajectory(initial, unguided_by_k, guided_by_k, config, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    dims = list(config.guide_dims)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(initial[:, dims[0]], initial[:, dims[1]], "o-", label="initial", alpha=0.7)
    for k, action in unguided_by_k.items():
        ax.plot(action[:, dims[0]], action[:, dims[1]], "o--", label=f"unguided K={k}", alpha=0.65)
    for k, action in guided_by_k.items():
        ax.plot(action[:, dims[0]], action[:, dims[1]], "o-", label=f"guided K={k}", alpha=0.8)

    all_actions = [initial, *unguided_by_k.values(), *guided_by_k.values()]
    all_xy = np.concatenate([action[:, dims] for action in all_actions], axis=0)
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

    ax.set_xlabel(f"action dim {dims[0]}")
    ax.set_ylabel(f"action dim {dims[1]}")
    ax.set_title("Diffusion-only guided replanning")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_sample(
    model,
    action_features,
    current_state,
    current_state_mask,
    action_mask,
    fov,
    args,
    guidance_fn=None,
    fixed_actions=None,
    fixed_action_mask=None,
):
    def _call():
        return model.sample_action_from_condition(
            action_features=action_features,
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

    output, elapsed = timed_call(_call)
    if guidance_fn is not None or fixed_actions is not None:
        action, trace = output
    else:
        action, trace = output, []
    return action[0].astype(np.float32), trace or [], elapsed


def summarize_rows(rows: list[dict]) -> dict:
    summary = {}
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        summary[model_name] = {}
        for k in sorted({row["k"] for row in model_rows}):
            k_rows = [row for row in model_rows if row["k"] == k]
            summary[model_name][f"k{k}"] = {
                "n": len(k_rows),
                "encode_time_ms": float(np.mean([row["encode_time_ms"] for row in k_rows])),
                "initial_diffusion_time_ms": float(np.mean([row["initial_diffusion_time_ms"] for row in k_rows])),
                "unguided_diffusion_only_time_ms": float(
                    np.mean([row["unguided_diffusion_only_time_ms"] for row in k_rows])
                ),
                "guided_diffusion_only_time_ms": float(
                    np.mean([row["guided_diffusion_only_time_ms"] for row in k_rows])
                ),
                "guided_over_unguided_ratio": float(
                    np.mean([row["guided_diffusion_only_time_ms"] / row["unguided_diffusion_only_time_ms"] for row in k_rows])
                ),
                "violation_mean_unguided": float(np.mean([row["violation_mean_unguided"] for row in k_rows])),
                "violation_mean_guided": float(np.mean([row["violation_mean_guided"] for row in k_rows])),
                "violation_delta": float(np.mean([row["violation_delta"] for row in k_rows])),
                "improvement_rate": float(np.mean([row["violation_delta"] < 0 for row in k_rows])),
                "max_prefix_error_guided": float(np.max([row["prefix_error_guided"] for row in k_rows])),
            }
    return summary


def run_model(model_name: str, args, guidance_config: PolynomialGuidanceConfig, image: Image.Image) -> dict:
    spec = MODEL_SPECS[model_name]
    model, config = load_model(Path(spec["config"]), spec["checkpoint"])
    for param in model.parameters():
        param.requires_grad_(False)

    current_state, current_state_mask, action_mask, fov = make_default_tensors(
        config,
        mask_start=args.mask_start,
        mask_end=args.mask_end,
    )
    action_mask_np = action_mask.detach().cpu().numpy()
    action_features, encode_time = timed_call(
        lambda: model.encode_action_condition(
            image=image,
            instruction=args.instruction,
            current_state=current_state,
            current_state_mask=current_state_mask,
            fov=fov,
        )
    )

    rows = []
    trace_dir = args.save_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    seed0_initial = None
    seed0_unguided = {}
    seed0_guided = {}

    for seed in args.seeds:
        set_seed(seed)
        initial, initial_trace, initial_time = run_sample(
            model,
            action_features,
            current_state,
            current_state_mask,
            action_mask,
            fov,
            args,
        )
        if seed == args.seeds[0]:
            seed0_initial = initial

        for k in args.replan_indices:
            suffix_config = filter_config_for_suffix(guidance_config, k, initial.shape[0])
            fixed_actions, fixed_mask = make_prefix_tensors(initial, action_mask, k)

            set_seed(seed + k)
            unguided, trace_unguided, unguided_time = run_sample(
                model,
                action_features,
                current_state,
                current_state_mask,
                action_mask,
                fov,
                args,
                fixed_actions=fixed_actions,
                fixed_action_mask=fixed_mask,
            )
            set_seed(seed + k)
            guided, trace_guided, guided_time = run_sample(
                model,
                action_features,
                current_state,
                current_state_mask,
                action_mask,
                fov,
                args,
                guidance_fn=PolynomialRegionLoss(suffix_config),
                fixed_actions=fixed_actions,
                fixed_action_mask=fixed_mask,
            )

            unguided_metrics = compute_region_metrics(unguided, suffix_config)
            guided_metrics = compute_region_metrics(guided, suffix_config)
            row = {
                "model": model_name,
                "seed": seed,
                "k": k,
                "encode_time_ms": encode_time * 1000.0,
                "initial_diffusion_time_ms": initial_time * 1000.0,
                "unguided_diffusion_only_time_ms": unguided_time * 1000.0,
                "guided_diffusion_only_time_ms": guided_time * 1000.0,
                "violation_mean_unguided": unguided_metrics["violation_mean"],
                "violation_mean_guided": guided_metrics["violation_mean"],
                "violation_delta": guided_metrics["violation_mean"] - unguided_metrics["violation_mean"],
                "prefix_error_unguided": prefix_error(unguided, initial, action_mask_np, k),
                "prefix_error_guided": prefix_error(guided, initial, action_mask_np, k),
                "suffix_relative_l2_guided_vs_unguided": relative_l2(guided[k:], unguided[k:]),
            }
            rows.append(row)

            if seed == args.seeds[0]:
                seed0_unguided[k] = unguided
                seed0_guided[k] = guided
                (trace_dir / f"{model_name}_seed{seed}_k{k}_unguided.json").write_text(
                    json.dumps(trace_unguided, indent=2),
                    encoding="utf-8",
                )
                (trace_dir / f"{model_name}_seed{seed}_k{k}_guided.json").write_text(
                    json.dumps(trace_guided, indent=2),
                    encoding="utf-8",
                )
                plot_guidance_loss_curve(
                    trace_guided,
                    args.save_dir / f"{model_name}_seed{seed}_k{k}_guidance_loss.png",
                )

    if seed0_initial is not None:
        plot_trajectory(
            seed0_initial,
            seed0_unguided,
            seed0_guided,
            guidance_config,
            args.save_dir / f"{model_name}_seed{args.seeds[0]}_trajectory.png",
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cached diffusion-only VITRA guided replanning.")
    parser.add_argument("--models", default="base_vitra3b,joint_kd_student")
    parser.add_argument("--image_path", type=Path, default=Path("examples/0002.jpg"))
    parser.add_argument("--instruction", default="Left hand: None. Right hand: Pick up the phone on the table.")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--regions_json", type=Path, default=Path("configs/polynomial_guidance_example.json"))
    parser.add_argument("--guide_dims", type=int, nargs=2, default=None)
    parser.add_argument("--temporal_mask", choices=["none", "tail", "target"], default="target")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--guidance_start_frac", type=float, default=0.0)
    parser.add_argument("--guidance_end_frac", type=float, default=1.0)
    parser.add_argument("--guidance_grad_clip", type=float, default=5.0)
    parser.add_argument("--replan_indices", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--mask_start", type=int, default=51)
    parser.add_argument("--mask_end", type=int, default=102)
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/replanning_guidance/diffusion_only_base_vs_distill"))
    args = parser.parse_args()

    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in selected_models if item not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model names {unknown}; available: {sorted(MODEL_SPECS)}")
    if args.replan_indices != [5, 10]:
        raise ValueError("This report expects --replan_indices 5 10.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    guidance_config = load_polynomial_guidance_config(args.regions_json, guide_dims=args.guide_dims)
    guidance_config.temporal_mask = args.temporal_mask
    image = Image.open(args.image_path).convert("RGB")

    rows = []
    for model_name in selected_models:
        result = run_model(model_name, args, guidance_config, image)
        rows.extend(result["rows"])

    report = {
        "setup": {
            "timing_scope": "cached action feature; replan timings are diffusion-only action-head sampling",
            "models": selected_models,
            "num_ddim_steps": args.num_ddim_steps,
            "cfg_scale": args.cfg_scale,
            "guidance_scale": args.guidance_scale,
            "guidance_grad_clip": args.guidance_grad_clip,
            "temporal_mask": guidance_config.temporal_mask,
            "guide_dims": list(guidance_config.guide_dims),
            "replan_indices": args.replan_indices,
            "seeds": args.seeds,
        },
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    (args.save_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Outputs written to {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
