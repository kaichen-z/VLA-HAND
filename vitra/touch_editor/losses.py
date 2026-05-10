from __future__ import annotations

import torch


def masked_mean_square(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(value.dtype)
    return (value.square() * mask).sum() / mask.sum().clamp_min(eps)


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
) -> dict[str, torch.Tensor]:
    valid = action_mask.to(delta.dtype)
    future = future_mask.to(delta.dtype)
    editable = future * valid
    a_edit = a_base + editable * delta
    if residual_target is None:
        residual_target = a_target - a_base

    loss_residual = masked_mean_square(delta - residual_target, editable)
    loss_demo = masked_mean_square(a_edit - a_target, editable)
    loss_dev = masked_mean_square(a_edit - a_base, editable)
    loss_delta = masked_mean_square(delta, editable)
    if delta.shape[1] > 1:
        smooth_mask = editable[:, 1:] * editable[:, :-1]
        loss_smooth = masked_mean_square(delta[:, 1:] - delta[:, :-1], smooth_mask)
    else:
        loss_smooth = delta.new_tensor(0.0)
    loss_mask = masked_mean_square(delta, 1.0 - valid)

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
