from __future__ import annotations

import torch
import torch.nn as nn

from .tactile_losses import build_tactile_stats


def _make_transformer(hidden_dim: int, num_heads: int, num_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class TactileEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        stat_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.stat_dim = int(stat_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.frame_proj = nn.Linear(64, hidden_dim)
        self.mask_proj = nn.Linear(2, hidden_dim)
        self.temporal = _make_transformer(hidden_dim, num_heads=num_heads, num_layers=num_layers)
        self.embed_head = nn.Linear(hidden_dim, embed_dim)
        self.stat_head = nn.Linear(hidden_dim, stat_dim)

    def forward(self, touch_pressure: torch.Tensor, touch_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if touch_pressure.ndim != 5:
            raise ValueError(f"touch_pressure must be [B,T,2,16,16], got {tuple(touch_pressure.shape)}")
        if touch_mask.ndim != 3:
            raise ValueError(f"touch_mask must be [B,T,2], got {tuple(touch_mask.shape)}")
        bsz, horizon = touch_pressure.shape[:2]
        touch_mask_float = touch_mask.to(device=touch_pressure.device, dtype=touch_pressure.dtype)
        masked_touch = touch_pressure * touch_mask_float[..., None, None]
        hidden = self.cnn(masked_touch.reshape(bsz * horizon, 2, 16, 16)).flatten(1)
        hidden = self.frame_proj(hidden).reshape(bsz, horizon, -1)
        hidden = hidden + self.mask_proj(touch_mask_float)
        hidden = self.temporal(hidden)
        return {
            "embedding": self.embed_head(hidden),
            "stats": self.stat_head(hidden),
        }


class TactileForwardModel(nn.Module):
    def __init__(
        self,
        action_dim: int = 192,
        state_dim: int = 212,
        embed_dim: int = 64,
        stat_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        self.stat_dim = int(stat_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.phase_proj = nn.Linear(1, hidden_dim)
        self.temporal = _make_transformer(hidden_dim, num_heads=num_heads, num_layers=num_layers)
        self.embed_head = nn.Linear(hidden_dim, embed_dim)
        self.stat_head = nn.Linear(hidden_dim, stat_dim)

    def forward(
        self,
        current_state: torch.Tensor,
        action_segment: torch.Tensor,
        chunk_phase: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if current_state.ndim != 2:
            raise ValueError(f"current_state must be [B,S], got {tuple(current_state.shape)}")
        if action_segment.ndim != 3:
            raise ValueError(f"action_segment must be [B,T,D], got {tuple(action_segment.shape)}")
        bsz, horizon = action_segment.shape[:2]
        if chunk_phase.ndim == 1:
            chunk_phase = chunk_phase[None].expand(bsz, -1)
        if chunk_phase.shape[:2] != (bsz, horizon):
            raise ValueError(f"chunk_phase must be [B,T], got {tuple(chunk_phase.shape)} for B={bsz}, T={horizon}")
        state = self.state_proj(current_state)[:, None, :]
        hidden = self.action_proj(action_segment) + state + self.phase_proj(chunk_phase[..., None].to(action_segment.dtype))
        hidden = self.temporal(hidden)
        return {
            "embedding": self.embed_head(hidden),
            "stats": self.stat_head(hidden),
        }


@torch.no_grad()
def encode_tactile_targets(
    encoder: TactileEncoder,
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    encoded = encoder(touch_pressure, touch_mask)
    encoded["manual_stats"] = build_tactile_stats(touch_pressure, touch_mask)
    return encoded
