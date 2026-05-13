from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class QuadraticRegion:
    tau: float
    center: Sequence[float]
    Q: Sequence[Sequence[float]]
    radius2: float = 1.0
    weight: float = 1.0
    name: str = ""


@dataclass
class PolynomialGuidanceConfig:
    guide_dims: Sequence[int]
    regions: list[QuadraticRegion]
    loss_type: str = "relu_squared"
    eps: float = 1e-6
    temporal_mask: str = "tail"


def tau_to_index(tau: float, T: int) -> int:
    idx = int(round(float(tau) * float(T - 1)))
    return max(0, min(T - 1, idx))


def _as_region(item: dict[str, Any]) -> QuadraticRegion:
    return QuadraticRegion(
        tau=float(item["tau"]),
        center=item["center"],
        Q=item["Q"],
        radius2=float(item.get("radius2", 1.0)),
        weight=float(item.get("weight", 1.0)),
        name=str(item.get("name", "")),
    )


def load_polynomial_guidance_config(path: str | Path, guide_dims: Sequence[int] | None = None) -> PolynomialGuidanceConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PolynomialGuidanceConfig(
        guide_dims=list(guide_dims if guide_dims is not None else data["guide_dims"]),
        regions=[_as_region(item) for item in data["regions"]],
        loss_type=str(data.get("loss_type", "relu_squared")),
        eps=float(data.get("eps", 1e-6)),
        temporal_mask=str(data.get("temporal_mask", "tail")),
    )


class PolynomialRegionLoss:
    def __init__(self, config: PolynomialGuidanceConfig):
        if len(config.guide_dims) != 2:
            raise ValueError("PolynomialRegionLoss V0 supports exactly two guide dimensions.")
        if config.loss_type not in {"relu_squared", "approx_distance"}:
            raise ValueError(f"Unknown loss_type: {config.loss_type}")
        if config.temporal_mask not in {"none", "tail", "target"}:
            raise ValueError(f"Unknown temporal_mask: {config.temporal_mask}")
        self.config = config

    def _region_loss(self, pred_xstart: torch.Tensor, region: QuadraticRegion):
        B, T, _ = pred_xstart.shape
        device = pred_xstart.device
        dtype = pred_xstart.dtype
        dims = torch.tensor(self.config.guide_dims, device=device, dtype=torch.long)
        idx = tau_to_index(region.tau, T)
        u = pred_xstart[:, idx, :].index_select(dim=-1, index=dims)
        c = torch.tensor(region.center, device=device, dtype=dtype).view(1, 2)
        Q = torch.tensor(region.Q, device=device, dtype=dtype).view(2, 2)
        radius2 = torch.tensor(region.radius2, device=device, dtype=dtype)

        du = u - c
        Qdu = du @ Q.T
        p = (du * Qdu).sum(dim=-1) - radius2
        violation = F.relu(p)
        if self.config.loss_type == "relu_squared":
            loss = violation.pow(2).mean()
        else:
            grad_p = 2.0 * Qdu
            grad_norm2 = grad_p.pow(2).sum(dim=-1).clamp_min(self.config.eps)
            loss = (violation.pow(2) / grad_norm2).mean()
        return idx, p, violation, loss

    def __call__(self, pred_xstart: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total_loss = pred_xstart.new_tensor(0.0)
        metrics: dict[str, torch.Tensor] = {}
        for region in self.config.regions:
            idx, p, violation, loss = self._region_loss(pred_xstart, region)
            weighted = float(region.weight) * loss
            total_loss = total_loss + weighted
            name = region.name or f"tau_{region.tau}"
            metrics[f"{name}/p_mean"] = p.detach().mean()
            metrics[f"{name}/violation_mean"] = violation.detach().mean()
            metrics[f"{name}/success_rate"] = (p.detach() <= 0).float().mean()
            metrics[f"{name}/idx"] = torch.tensor(idx, device=pred_xstart.device)
        metrics["loss_poly"] = total_loss.detach()
        return total_loss, metrics

    def _apply_temporal_mask(self, grad: torch.Tensor, idx: int) -> torch.Tensor:
        if self.config.temporal_mask == "none":
            return grad
        mask = torch.zeros_like(grad)
        if self.config.temporal_mask == "tail":
            mask[:, idx:, :] = 1
        elif self.config.temporal_mask == "target":
            mask[:, idx : idx + 1, :] = 1
        return grad * mask

    def gradient(self, x_in: torch.Tensor, pred_xstart: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        total_loss = pred_xstart.new_tensor(0.0)
        total_grad = torch.zeros_like(x_in)
        metrics: dict[str, torch.Tensor] = {}
        for region_idx, region in enumerate(self.config.regions):
            idx, p, violation, loss = self._region_loss(pred_xstart, region)
            weighted = float(region.weight) * loss
            retain_graph = region_idx != len(self.config.regions) - 1
            grad_j = torch.autograd.grad(
                weighted,
                x_in,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )[0]
            total_grad = total_grad + self._apply_temporal_mask(grad_j, idx)
            total_loss = total_loss + weighted.detach()
            name = region.name or f"tau_{region.tau}"
            metrics[f"{name}/p_mean"] = p.detach().mean()
            metrics[f"{name}/violation_mean"] = violation.detach().mean()
            metrics[f"{name}/success_rate"] = (p.detach() <= 0).float().mean()
            metrics[f"{name}/idx"] = torch.tensor(idx, device=pred_xstart.device)
        metrics["loss_poly"] = total_loss.detach()
        return total_grad, total_loss, metrics


class CFGAwareGuidanceWrapper:
    def __init__(self, base_guidance_fn, original_batch_size: int, using_cfg: bool):
        self.base_guidance_fn = base_guidance_fn
        self.original_batch_size = int(original_batch_size)
        self.using_cfg = bool(using_cfg)

    def _slice(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[: self.original_batch_size] if self.using_cfg else tensor

    def __call__(self, pred_xstart: torch.Tensor):
        loss, metrics = self.base_guidance_fn(self._slice(pred_xstart))
        metrics = dict(metrics)
        metrics["cfg/original_batch_size"] = torch.tensor(
            self.original_batch_size,
            device=pred_xstart.device,
            dtype=torch.long,
        )
        return loss, metrics

    def gradient(self, x_in: torch.Tensor, pred_xstart: torch.Tensor):
        pred_cond = self._slice(pred_xstart)
        if hasattr(self.base_guidance_fn, "_region_loss") and hasattr(self.base_guidance_fn, "_apply_temporal_mask"):
            grad = torch.zeros_like(x_in)
            loss = pred_xstart.new_tensor(0.0)
            metrics: dict[str, torch.Tensor] = {}
            regions = self.base_guidance_fn.config.regions
            for region_idx, region in enumerate(regions):
                idx, p, violation, loss_j = self.base_guidance_fn._region_loss(pred_cond, region)
                weighted = float(region.weight) * loss_j
                retain_graph = region_idx != len(regions) - 1
                grad_j = torch.autograd.grad(
                    weighted,
                    x_in,
                    retain_graph=retain_graph,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                grad = grad + self.base_guidance_fn._apply_temporal_mask(grad_j, idx)
                loss = loss + weighted.detach()
                name = region.name or f"tau_{region.tau}"
                metrics[f"{name}/p_mean"] = p.detach().mean()
                metrics[f"{name}/violation_mean"] = violation.detach().mean()
                metrics[f"{name}/success_rate"] = (p.detach() <= 0).float().mean()
                metrics[f"{name}/idx"] = torch.tensor(idx, device=pred_xstart.device)
            metrics["loss_poly"] = loss.detach()
        else:
            loss, metrics = self.base_guidance_fn(pred_cond)
            grad = torch.autograd.grad(loss, x_in, retain_graph=False, create_graph=False)[0]
        metrics = dict(metrics)
        metrics["cfg/original_batch_size"] = torch.tensor(
            self.original_batch_size,
            device=pred_xstart.device,
            dtype=torch.long,
        )
        return grad, loss, metrics
