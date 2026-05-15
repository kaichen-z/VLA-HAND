from __future__ import annotations

from dataclasses import dataclass

import torch

from .tactile_forward_model import TactileEncoder, TactileForwardModel
from .tactile_losses import build_tactile_stats, masked_mse, suffix_action_mask, touch_valid_step_mask


@dataclass
class TactileDPSConfig:
    edit_start_idx: int = 3
    horizon: int | None = None
    lambda_embed: float = 1.0
    lambda_stats: float = 0.25
    lambda_prior: float = 0.01
    lambda_boundary: float = 0.05


class TactileDPSGuidance:
    """DPS-style guidance loss for tactile-conditioned action regeneration."""

    def __init__(
        self,
        *,
        tactile_encoder: TactileEncoder,
        tactile_forward_model: TactileForwardModel,
        current_state: torch.Tensor,
        touch_pressure: torch.Tensor,
        touch_mask: torch.Tensor,
        chunk_phase: torch.Tensor,
        edit_start_idx: int,
        action_mask: torch.Tensor,
        old_action: torch.Tensor | None = None,
        horizon: int | None = None,
        lambda_embed: float = 1.0,
        lambda_stats: float = 0.25,
        lambda_prior: float = 0.01,
        lambda_boundary: float = 0.05,
    ) -> None:
        self.tactile_encoder = tactile_encoder
        self.tactile_forward_model = tactile_forward_model
        self.current_state = current_state
        self.touch_pressure = touch_pressure
        self.touch_mask = touch_mask
        self.chunk_phase = chunk_phase
        self.edit_start_idx = int(edit_start_idx)
        self.action_mask = action_mask
        self.old_action = old_action
        self.horizon = horizon
        self.lambda_embed = float(lambda_embed)
        self.lambda_stats = float(lambda_stats)
        self.lambda_prior = float(lambda_prior)
        self.lambda_boundary = float(lambda_boundary)

    def _slice(self, value: torch.Tensor) -> torch.Tensor:
        k = max(0, min(self.edit_start_idx, int(value.shape[1])))
        end = value.shape[1] if self.horizon is None else min(value.shape[1], k + int(self.horizon))
        return value[:, k:end]

    def _loss(self, pred_xstart: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = pred_xstart.device
        dtype = pred_xstart.dtype
        current_state = self.current_state.to(device=device, dtype=dtype)
        touch_pressure = self.touch_pressure.to(device=device, dtype=dtype)
        touch_mask = self.touch_mask.to(device=device, dtype=torch.bool)
        chunk_phase = self.chunk_phase.to(device=device, dtype=dtype)
        if chunk_phase.ndim == 1:
            chunk_phase = chunk_phase[None].expand(pred_xstart.shape[0], -1)

        action_segment = self._slice(pred_xstart)
        target_touch = self._slice(touch_pressure)
        target_mask = self._slice(touch_mask)
        target_phase = self._slice(chunk_phase)
        target_stats = build_tactile_stats(target_touch, target_mask)
        valid = touch_valid_step_mask(target_mask)

        target = self.tactile_encoder(target_touch, target_mask)
        pred = self.tactile_forward_model(current_state, action_segment, target_phase)
        embed_loss = masked_mse(pred["embedding"], target["embedding"].detach(), valid)
        stats_loss = masked_mse(pred["stats"], target_stats, valid)
        loss = self.lambda_embed * embed_loss + self.lambda_stats * stats_loss

        prior_loss = pred_xstart.new_tensor(0.0)
        if self.old_action is not None and self.lambda_prior > 0:
            old_action = self.old_action.to(device=device, dtype=dtype)
            editable = suffix_action_mask(self.action_mask.to(device=device, dtype=torch.bool), self.edit_start_idx)
            prior_loss = masked_mse(pred_xstart, old_action, editable.any(dim=-1))
            loss = loss + self.lambda_prior * prior_loss

        boundary_loss = pred_xstart.new_tensor(0.0)
        if self.lambda_boundary > 0 and 0 < self.edit_start_idx < pred_xstart.shape[1]:
            boundary_loss = (pred_xstart[:, self.edit_start_idx] - pred_xstart[:, self.edit_start_idx - 1]).square().mean()
            loss = loss + self.lambda_boundary * boundary_loss

        metrics = {
            "embed_loss": embed_loss.detach(),
            "stats_loss": stats_loss.detach(),
            "prior_loss": prior_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
            "touch_valid_rate": valid.float().mean().detach(),
        }
        return loss, metrics

    def __call__(self, pred_xstart: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._loss(pred_xstart)

    def gradient(
        self,
        x_in: torch.Tensor,
        pred_xstart: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = self._loss(pred_xstart)
        grad = torch.autograd.grad(loss, x_in, retain_graph=False, create_graph=False, allow_unused=False)[0]
        prefix_mask = suffix_action_mask(self.action_mask.to(device=grad.device, dtype=torch.bool), self.edit_start_idx)
        if prefix_mask.shape[0] != grad.shape[0]:
            full_mask = torch.zeros_like(grad, dtype=torch.bool)
            full_mask[: prefix_mask.shape[0]] = prefix_mask
            prefix_mask = full_mask
        grad = grad * prefix_mask.to(dtype=grad.dtype)
        return grad, loss.detach(), metrics
