from __future__ import annotations

import torch


def _valid_step_mask(touch_mask: torch.Tensor) -> torch.Tensor:
    if touch_mask.ndim != 3:
        raise ValueError(f"touch_mask must be [B,T,2], got {tuple(touch_mask.shape)}")
    return torch.any(touch_mask, dim=-1)


def build_tactile_stats(touch_pressure: torch.Tensor, touch_mask: torch.Tensor) -> torch.Tensor:
    """Build interpretable per-step tactile statistics from OpenTouch pressure maps.

    Output channels are:
      valid_left, valid_right, mean_abs_left, mean_abs_right,
      max_abs_left, max_abs_right, delta_mean_left, delta_mean_right.
    """
    if touch_pressure.ndim != 5:
        raise ValueError(f"touch_pressure must be [B,T,2,16,16], got {tuple(touch_pressure.shape)}")
    if touch_mask.ndim != 3:
        raise ValueError(f"touch_mask must be [B,T,2], got {tuple(touch_mask.shape)}")
    if touch_pressure.shape[:3] != touch_mask.shape:
        raise ValueError("touch_pressure and touch_mask batch/time/channel dimensions must match")

    pressure = touch_pressure.abs()
    valid = touch_mask.to(dtype=touch_pressure.dtype)
    pressure = pressure * valid[..., None, None]
    mean_abs = pressure.mean(dim=(-1, -2))
    max_abs = pressure.amax(dim=(-1, -2))
    base = touch_pressure[:, :1]
    base_valid = touch_mask[:, :1] & touch_mask
    delta = (touch_pressure - base).abs() * base_valid[..., None, None].to(touch_pressure.dtype)
    delta_mean = delta.mean(dim=(-1, -2))
    return torch.cat([valid, mean_abs, max_abs, delta_mean], dim=-1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if mask.ndim == pred.ndim - 1:
        mask = mask[..., None]
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.expand_as(pred).sum().clamp_min(1.0)
    return ((pred - target).square() * mask).sum() / denom


def touch_valid_step_mask(touch_mask: torch.Tensor) -> torch.Tensor:
    return _valid_step_mask(touch_mask)


def suffix_action_mask(action_mask: torch.Tensor, edit_start_idx: int) -> torch.Tensor:
    if action_mask.ndim != 3:
        raise ValueError(f"action_mask must be [B,T,D], got {tuple(action_mask.shape)}")
    mask = torch.zeros_like(action_mask, dtype=torch.float32)
    k = max(0, min(int(edit_start_idx), int(action_mask.shape[1])))
    mask[:, k:] = action_mask[:, k:].to(torch.float32)
    return mask
