from __future__ import annotations

import torch


def masked_mean_square(
    value: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = mask.to(value.dtype)
    if sample_weight is not None:
        weight = sample_weight.to(value.device, dtype=value.dtype).reshape(-1, *([1] * (value.ndim - 1)))
        mask = mask * weight
    return (value.square() * mask).sum() / mask.sum().clamp_min(eps)


def editable_mask(action_mask: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    return action_mask.to(future_mask.dtype) * future_mask.to(future_mask.dtype)


def hand_scope_action_mask(action_mask: torch.Tensor, hand_scope: str) -> torch.Tensor:
    """Restrict an action mask to left, right, or both hand dimensions."""
    if hand_scope == "both":
        return action_mask
    if hand_scope not in {"left", "right"}:
        raise ValueError(f"Unsupported hand_scope: {hand_scope}")
    if action_mask.ndim < 1:
        raise ValueError("action_mask must have at least one dimension")
    half_dim = int(action_mask.shape[-1]) // 2
    scoped = torch.zeros_like(action_mask, dtype=torch.bool)
    if hand_scope == "left":
        scoped[..., :half_dim] = action_mask[..., :half_dim].to(torch.bool)
    else:
        scoped[..., half_dim:] = action_mask[..., half_dim:].to(torch.bool)
    return scoped


def zero_delta_loss(delta: torch.Tensor, action_mask: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    return masked_mean_square(delta, editable_mask(action_mask, future_mask))


def touch_editor_loss(
    a_base: torch.Tensor,
    a_target: torch.Tensor,
    delta: torch.Tensor,
    action_mask: torch.Tensor,
    future_mask: torch.Tensor,
    residual_target: torch.Tensor | None = None,
    lambda_dev: float = 0.0,
    lambda_delta: float = 0.01,
    lambda_smooth: float = 0.05,
    lambda_mask: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    valid = action_mask.to(delta.dtype)
    editable = editable_mask(action_mask, future_mask).to(delta.dtype)
    a_edit = a_base + editable * delta
    if residual_target is None:
        residual_target = a_target - a_base

    loss_residual = masked_mean_square(delta - residual_target, editable, sample_weight=sample_weight)
    loss_demo = masked_mean_square(a_edit - a_target, editable, sample_weight=sample_weight)
    loss_dev = masked_mean_square(a_edit - a_base, editable, sample_weight=sample_weight)
    loss_delta = masked_mean_square(delta, editable, sample_weight=sample_weight)
    if delta.shape[1] > 1:
        smooth_mask = editable[:, 1:] * editable[:, :-1]
        loss_smooth = masked_mean_square(delta[:, 1:] - delta[:, :-1], smooth_mask, sample_weight=sample_weight)
    else:
        loss_smooth = delta.new_tensor(0.0)
    loss_mask = masked_mean_square(delta, 1.0 - valid, sample_weight=sample_weight)

    total = (
        loss_residual
        + lambda_dev * loss_dev
        + lambda_delta * loss_delta
        + lambda_smooth * loss_smooth
        + lambda_mask * loss_mask
    )
    return {
        "loss": total,
        "loss_residual": loss_residual,
        "loss_demo": loss_demo,
        "loss_dev": loss_dev,
        "loss_delta": loss_delta,
        "loss_smooth": loss_smooth,
        "loss_mask": loss_mask,
        "a_edit": a_edit,
    }
