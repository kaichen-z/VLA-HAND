from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTouchEditor(nn.Module):
    """Small residual editor conditioned on touch history and base actions."""

    def __init__(
        self,
        action_dim: int = 192,
        state_dim: int = 212,
        touch_feature_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.touch_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, touch_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.input_proj = nn.Linear(action_dim * 4 + state_dim * 2 + touch_feature_dim + 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        a_base: torch.Tensor,
        current_state: torch.Tensor,
        current_state_mask: torch.Tensor,
        touch_pressure: torch.Tensor,
        touch_mask: torch.Tensor,
        chunk_phase: torch.Tensor,
        future_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return residual action delta shaped [B, T, action_dim]."""
        bsz, chunk_len, _ = a_base.shape
        touch = self._align_touch_to_chunk(touch_pressure, chunk_len)
        aligned_touch_mask = self._align_touch_mask_to_chunk(touch_mask, chunk_len)
        touch = touch * aligned_touch_mask[:, :, :, None, None].to(touch.dtype)
        touch_features = self.touch_encoder(touch.reshape(bsz * chunk_len, 2, 16, 16)).reshape(bsz, chunk_len, -1)
        state_mask_float = current_state_mask.to(current_state.dtype)
        masked_state = current_state * state_mask_float
        state_features = masked_state[:, None, :].expand(-1, chunk_len, -1)
        state_mask_features = state_mask_float[:, None, :].expand(-1, chunk_len, -1)
        phase = self._format_phase(chunk_phase, bsz, chunk_len, a_base.device)
        action_mask_float = action_mask.to(a_base.dtype)
        future_mask_float = future_mask.to(a_base.dtype)
        touch_valid = aligned_touch_mask.any(dim=-1).to(a_base.dtype)[..., None]
        features = torch.cat(
            [
                a_base,
                a_base * action_mask_float,
                action_mask_float,
                future_mask_float,
                state_features,
                state_mask_features,
                touch_features,
                touch_valid,
                phase,
            ],
            dim=-1,
        )
        hidden = self.temporal(self.input_proj(features))
        return self.out(hidden)

    @staticmethod
    def _align_touch_to_chunk(touch_pressure: torch.Tensor, chunk_len: int) -> torch.Tensor:
        if touch_pressure.shape[1] == chunk_len:
            return touch_pressure
        if touch_pressure.shape[1] > chunk_len:
            return touch_pressure[:, -chunk_len:]
        pad_len = chunk_len - touch_pressure.shape[1]
        pad = touch_pressure[:, :1].expand(-1, pad_len, -1, -1, -1)
        return torch.cat([pad, touch_pressure], dim=1)

    @staticmethod
    def _align_touch_mask_to_chunk(touch_mask: torch.Tensor, chunk_len: int) -> torch.Tensor:
        if touch_mask.shape[1] == chunk_len:
            return touch_mask
        if touch_mask.shape[1] > chunk_len:
            return touch_mask[:, -chunk_len:]
        pad_len = chunk_len - touch_mask.shape[1]
        pad = touch_mask[:, :1].expand(-1, pad_len, -1)
        return torch.cat([pad, touch_mask], dim=1)

    @staticmethod
    def _format_phase(chunk_phase: torch.Tensor, bsz: int, chunk_len: int, device: torch.device) -> torch.Tensor:
        phase = chunk_phase.to(device)
        if phase.ndim == 0:
            phase = phase[None, None].expand(bsz, chunk_len)
        elif phase.ndim == 1:
            if phase.shape[0] == chunk_len:
                phase = phase[None, :].expand(bsz, -1)
            else:
                phase = phase[:, None].expand(-1, chunk_len)
        return phase[..., None].to(torch.float32)
