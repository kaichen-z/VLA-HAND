from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenTouchCNNetEmbedding(nn.Module):
    """OpenTouch-compatible tactile encoder for 16x16 pressure sequences."""

    def __init__(self, emb_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.gru = nn.GRU(128, 120, num_layers=2, bidirectional=True)
        self.projection = nn.Linear(240, emb_dim)

    @staticmethod
    def _normalize_input(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim == 5:
            bsz, horizon, channels, height, width = x.shape
            if channels != 1:
                raise ValueError(f"OpenTouch tactile input must have one channel, got {channels}")
            x = x.squeeze(2)
        elif x.ndim == 4:
            bsz, horizon, height, width = x.shape
        else:
            raise ValueError(f"OpenTouch tactile input must be [B,T,1,16,16] or [B,T,16,16], got {tuple(x.shape)}")
        if (height, width) != (16, 16):
            raise ValueError(f"OpenTouch tactile input must be 16x16, got {(height, width)}")
        return x.reshape(bsz * horizon, 1, height, width), bsz, horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, bsz, horizon = self._normalize_input(x)
        feats = self.cnn(flat)
        seq = feats.view(bsz, horizon, -1).transpose(0, 1)
        self.gru.flatten_parameters()
        out, _ = self.gru(seq)
        combined = torch.cat([out[-1, :, :120], out[0, :, 120:]], dim=1)
        return self.projection(F.relu(combined))


def _load_opentouch_tactile_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported OpenTouch checkpoint format: {checkpoint_path}")
    tactile_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        if clean_key.startswith("tactile."):
            tactile_state[clean_key[len("tactile."):]] = value
    if not tactile_state:
        raise ValueError(f"No tactile.* weights found in OpenTouch checkpoint: {checkpoint_path}")
    return tactile_state


def load_opentouch_tactile_encoder(
    checkpoint_path: str | Path,
    *,
    embed_dim: int = 64,
) -> OpenTouchCNNetEmbedding:
    encoder = OpenTouchCNNetEmbedding(emb_dim=embed_dim)
    encoder.load_state_dict(_load_opentouch_tactile_state_dict(checkpoint_path), strict=True)
    return encoder


class ResidualTouchEditor(nn.Module):
    """Small residual editor conditioned on touch history and base actions."""

    def __init__(
        self,
        action_dim: int = 192,
        state_dim: int = 212,
        touch_feature_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        condition_mode: str = "full",
    ) -> None:
        super().__init__()
        if condition_mode not in {"full", "no_base", "touch_only"}:
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.touch_feature_dim = touch_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.condition_mode = condition_mode
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
        self.input_proj = nn.Linear(
            self._feature_dim(action_dim, state_dim, touch_feature_dim, condition_mode),
            hidden_dim,
        )
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
        feature_parts = []
        if self.condition_mode == "full":
            feature_parts.extend([a_base, a_base * action_mask_float])
        feature_parts.extend([action_mask_float, future_mask_float])
        if self.condition_mode in {"full", "no_base"}:
            feature_parts.extend([state_features, state_mask_features])
        feature_parts.extend([touch_features, touch_valid, phase])
        features = torch.cat(feature_parts, dim=-1)
        hidden = self.temporal(self.input_proj(features))
        return self.out(hidden)

    @staticmethod
    def _feature_dim(action_dim: int, state_dim: int, touch_feature_dim: int, condition_mode: str) -> int:
        dim = action_dim * 2 + touch_feature_dim + 2
        if condition_mode == "full":
            dim += action_dim * 2 + state_dim * 2
        elif condition_mode == "no_base":
            dim += state_dim * 2
        elif condition_mode != "touch_only":
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        return dim

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


class TactileGatedResidualEditor(nn.Module):
    """Residual editor with a separate tactile branch and learned tactile gate."""

    def __init__(
        self,
        action_dim: int = 192,
        state_dim: int = 212,
        touch_feature_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        condition_mode: str = "full",
        context_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if condition_mode not in {"full", "no_base", "touch_only"}:
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        if not 0.0 <= float(context_dropout_prob) <= 1.0:
            raise ValueError("context_dropout_prob must be between 0 and 1")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.touch_feature_dim = touch_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.condition_mode = condition_mode
        self.context_dropout_prob = float(context_dropout_prob)
        self.last_diagnostics: dict[str, torch.Tensor | float] = {}

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
        self.touch_proj = nn.Linear(touch_feature_dim + 2, hidden_dim)
        touch_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.touch_temporal = nn.TransformerEncoder(touch_layer, num_layers=max(1, num_layers))

        self.context_proj = nn.Linear(
            self._context_feature_dim(action_dim, state_dim, condition_mode),
            hidden_dim,
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.prior_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim))
        self.touch_head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, action_dim))
        self.gate_head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

    @staticmethod
    def _context_feature_dim(action_dim: int, state_dim: int, condition_mode: str) -> int:
        dim = action_dim * 2 + 1
        if condition_mode == "full":
            dim += action_dim * 2 + state_dim * 2
        elif condition_mode == "no_base":
            dim += state_dim * 2
        elif condition_mode != "touch_only":
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        return dim

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
        bsz, chunk_len, _ = a_base.shape
        a_base_ctx, current_state_ctx, base_keep, state_keep = self._apply_context_dropout(a_base, current_state)
        context = self._context_features(
            a_base_ctx,
            current_state_ctx,
            current_state_mask,
            chunk_phase,
            future_mask,
            action_mask,
        )
        context_hidden = self.context_proj(context)
        touch_hidden, touch_key_padding_mask = self._touch_tokens(touch_pressure, touch_mask, chunk_len)
        attended_touch, _ = self.cross_attention(
            query=context_hidden,
            key=touch_hidden,
            value=touch_hidden,
            key_padding_mask=touch_key_padding_mask,
            need_weights=False,
        )
        fused = torch.cat([context_hidden, attended_touch], dim=-1)
        prior_delta = self.prior_head(context_hidden)
        touch_delta = self.touch_head(fused)
        touch_gate = self.gate_head(fused)
        valid_touch = (~touch_key_padding_mask).any(dim=-1).to(touch_gate.dtype).view(bsz, 1, 1)
        touch_gate = touch_gate * valid_touch
        delta = prior_delta + touch_gate * touch_delta
        self.last_diagnostics = {
            "prior_delta": prior_delta.detach(),
            "touch_delta": touch_delta.detach(),
            "touch_gate": touch_gate.detach(),
            "base_keep_rate": float(base_keep.detach().mean().cpu()),
            "state_keep_rate": float(state_keep.detach().mean().cpu()),
        }
        return delta

    def _apply_context_dropout(
        self,
        a_base: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.context_dropout_prob <= 0.0:
            base_keep = torch.ones((a_base.shape[0], 1, 1), device=a_base.device, dtype=a_base.dtype)
            state_keep = torch.ones((current_state.shape[0], 1), device=current_state.device, dtype=current_state.dtype)
            return a_base, current_state, base_keep, state_keep
        base_keep = (
            torch.rand((a_base.shape[0], 1, 1), device=a_base.device, dtype=a_base.dtype) >= self.context_dropout_prob
        ).to(a_base.dtype)
        state_keep = (
            torch.rand((current_state.shape[0], 1), device=current_state.device, dtype=current_state.dtype)
            >= self.context_dropout_prob
        ).to(current_state.dtype)
        return a_base * base_keep, current_state * state_keep, base_keep, state_keep

    def _context_features(
        self,
        a_base: torch.Tensor,
        current_state: torch.Tensor,
        current_state_mask: torch.Tensor,
        chunk_phase: torch.Tensor,
        future_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, chunk_len, _ = a_base.shape
        state_mask_float = current_state_mask.to(current_state.dtype)
        masked_state = current_state * state_mask_float
        state_features = masked_state[:, None, :].expand(-1, chunk_len, -1)
        state_mask_features = state_mask_float[:, None, :].expand(-1, chunk_len, -1)
        phase = ResidualTouchEditor._format_phase(chunk_phase, bsz, chunk_len, a_base.device)
        action_mask_float = action_mask.to(a_base.dtype)
        future_mask_float = future_mask.to(a_base.dtype)
        feature_parts = []
        if self.condition_mode == "full":
            feature_parts.extend([a_base, a_base * action_mask_float])
        feature_parts.extend([action_mask_float, future_mask_float])
        if self.condition_mode in {"full", "no_base"}:
            feature_parts.extend([state_features, state_mask_features])
        feature_parts.append(phase)
        return torch.cat(feature_parts, dim=-1)

    def _touch_tokens(
        self,
        touch_pressure: torch.Tensor,
        touch_mask: torch.Tensor,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(touch_pressure.shape[0])
        touch = ResidualTouchEditor._align_touch_to_chunk(touch_pressure, chunk_len)
        aligned_touch_mask = ResidualTouchEditor._align_touch_mask_to_chunk(touch_mask, chunk_len)
        touch = touch * aligned_touch_mask[:, :, :, None, None].to(touch.dtype)
        features = self.touch_encoder(touch.reshape(bsz * chunk_len, 2, 16, 16)).reshape(bsz, chunk_len, -1)
        touch_valid = aligned_touch_mask.any(dim=-1).to(touch.dtype)[..., None]
        touch_count = aligned_touch_mask.to(touch.dtype).sum(dim=-1, keepdim=True) / max(1, aligned_touch_mask.shape[-1])
        tokens = self.touch_proj(torch.cat([features, touch_valid, touch_count], dim=-1))
        key_padding_mask = ~aligned_touch_mask.any(dim=-1)
        all_invalid = key_padding_mask.all(dim=1)
        if bool(all_invalid.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False
            tokens = tokens.clone()
            tokens[all_invalid, 0] = 0.0
        return self.touch_temporal(tokens), key_padding_mask


class PretrainedTactileGatedResidualEditor(nn.Module):
    """Gated residual editor using a frozen OpenTouch tactile embedding."""

    def __init__(
        self,
        action_dim: int = 192,
        state_dim: int = 212,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        condition_mode: str = "full",
        context_dropout_prob: float = 0.0,
        pretrained_touch_encoder_checkpoint: str | Path | None = None,
        pretrained_touch_encoder: nn.Module | None = None,
        pretrained_touch_embed_dim: int = 64,
        pretrained_touch_hand: str = "right",
        freeze_pretrained_touch_encoder: bool = True,
    ) -> None:
        super().__init__()
        if condition_mode not in {"full", "no_base", "touch_only"}:
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        if pretrained_touch_hand not in {"left", "right"}:
            raise ValueError(f"pretrained_touch_hand must be 'left' or 'right', got {pretrained_touch_hand}")
        if not 0.0 <= float(context_dropout_prob) <= 1.0:
            raise ValueError("context_dropout_prob must be between 0 and 1")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.condition_mode = condition_mode
        self.context_dropout_prob = float(context_dropout_prob)
        self.pretrained_touch_embed_dim = int(pretrained_touch_embed_dim)
        self.pretrained_touch_hand = pretrained_touch_hand
        self.freeze_pretrained_touch_encoder = bool(freeze_pretrained_touch_encoder)
        self.last_diagnostics: dict[str, torch.Tensor | float] = {}

        if pretrained_touch_encoder is not None:
            self.pretrained_tactile_encoder = pretrained_touch_encoder
        elif pretrained_touch_encoder_checkpoint is not None:
            self.pretrained_tactile_encoder = load_opentouch_tactile_encoder(
                pretrained_touch_encoder_checkpoint,
                embed_dim=self.pretrained_touch_embed_dim,
            )
        else:
            self.pretrained_tactile_encoder = OpenTouchCNNetEmbedding(emb_dim=self.pretrained_touch_embed_dim)
        if self.freeze_pretrained_touch_encoder:
            for param in self.pretrained_tactile_encoder.parameters():
                param.requires_grad = False
            self.pretrained_tactile_encoder.eval()

        self.context_proj = nn.Linear(
            TactileGatedResidualEditor._context_feature_dim(action_dim, state_dim, condition_mode),
            hidden_dim,
        )
        self.pretrained_touch_proj = nn.Sequential(
            nn.Linear(self.pretrained_touch_embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.context_temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=max(1, num_layers),
        )
        self.prior_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim))
        self.touch_head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, action_dim))
        self.gate_head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

    def train(self, mode: bool = True) -> "PretrainedTactileGatedResidualEditor":
        super().train(mode)
        if self.freeze_pretrained_touch_encoder:
            self.pretrained_tactile_encoder.eval()
        return self

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
        a_base_ctx, current_state_ctx, base_keep, state_keep = self._apply_context_dropout(a_base, current_state)
        context = self._context_features(
            a_base_ctx,
            current_state_ctx,
            current_state_mask,
            chunk_phase,
            future_mask,
            action_mask,
        )
        context_hidden = self.context_temporal(self.context_proj(context))
        touch_embedding, touch_valid = self._pretrained_touch_embedding(touch_pressure, touch_mask)
        touch_input = torch.cat([touch_embedding, touch_valid[:, None].to(touch_embedding.dtype)], dim=-1)
        touch_hidden = self.pretrained_touch_proj(touch_input)[:, None, :].expand(-1, a_base.shape[1], -1)
        fused = torch.cat([context_hidden, touch_hidden], dim=-1)
        prior_delta = self.prior_head(context_hidden)
        touch_delta = self.touch_head(fused)
        touch_gate = self.gate_head(fused) * touch_valid[:, None, None].to(a_base.dtype)
        delta = prior_delta + touch_gate * touch_delta
        self.last_diagnostics = {
            "prior_delta": prior_delta.detach(),
            "touch_delta": touch_delta.detach(),
            "touch_gate": touch_gate.detach(),
            "pretrained_touch_embedding": touch_embedding.detach(),
            "touch_valid": touch_valid.detach(),
            "base_keep_rate": float(base_keep.detach().mean().cpu()),
            "state_keep_rate": float(state_keep.detach().mean().cpu()),
        }
        return delta

    def _apply_context_dropout(
        self,
        a_base: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.context_dropout_prob <= 0.0:
            base_keep = torch.ones((a_base.shape[0], 1, 1), device=a_base.device, dtype=a_base.dtype)
            state_keep = torch.ones((current_state.shape[0], 1), device=current_state.device, dtype=current_state.dtype)
            return a_base, current_state, base_keep, state_keep
        base_keep = (
            torch.rand((a_base.shape[0], 1, 1), device=a_base.device, dtype=a_base.dtype) >= self.context_dropout_prob
        ).to(a_base.dtype)
        state_keep = (
            torch.rand((current_state.shape[0], 1), device=current_state.device, dtype=current_state.dtype)
            >= self.context_dropout_prob
        ).to(current_state.dtype)
        return a_base * base_keep, current_state * state_keep, base_keep, state_keep

    def _context_features(
        self,
        a_base: torch.Tensor,
        current_state: torch.Tensor,
        current_state_mask: torch.Tensor,
        chunk_phase: torch.Tensor,
        future_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, chunk_len, _ = a_base.shape
        state_mask_float = current_state_mask.to(current_state.dtype)
        masked_state = current_state * state_mask_float
        state_features = masked_state[:, None, :].expand(-1, chunk_len, -1)
        state_mask_features = state_mask_float[:, None, :].expand(-1, chunk_len, -1)
        phase = ResidualTouchEditor._format_phase(chunk_phase, bsz, chunk_len, a_base.device)
        action_mask_float = action_mask.to(a_base.dtype)
        future_mask_float = future_mask.to(a_base.dtype)
        feature_parts = []
        if self.condition_mode == "full":
            feature_parts.extend([a_base, a_base * action_mask_float])
        feature_parts.extend([action_mask_float, future_mask_float])
        if self.condition_mode in {"full", "no_base"}:
            feature_parts.extend([state_features, state_mask_features])
        feature_parts.append(phase)
        return torch.cat(feature_parts, dim=-1)

    def _pretrained_touch_embedding(
        self,
        touch_pressure: torch.Tensor,
        touch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if touch_pressure.ndim != 5:
            raise ValueError(f"touch_pressure must be [B,T,C,16,16], got {tuple(touch_pressure.shape)}")
        if touch_mask.ndim != 3:
            raise ValueError(f"touch_mask must be [B,T,C], got {tuple(touch_mask.shape)}")
        if touch_pressure.shape[:3] != touch_mask.shape[:3]:
            raise ValueError("touch_pressure and touch_mask batch/time/channel dimensions must match")
        hand_index = 0 if self.pretrained_touch_hand == "left" else 1
        if touch_pressure.shape[2] == 1:
            hand_index = 0
        elif hand_index >= touch_pressure.shape[2]:
            raise ValueError(f"touch_pressure has {touch_pressure.shape[2]} channels, cannot select {self.pretrained_touch_hand}")
        selected_touch = touch_pressure[:, :, hand_index:hand_index + 1]
        selected_mask = touch_mask[:, :, hand_index]
        selected_touch = selected_touch * selected_mask[:, :, None, None, None].to(selected_touch.dtype)
        valid = selected_mask.any(dim=1)
        if self.freeze_pretrained_touch_encoder:
            with torch.no_grad():
                embedding = self.pretrained_tactile_encoder(selected_touch)
        else:
            embedding = self.pretrained_tactile_encoder(selected_touch)
        embedding = F.normalize(embedding, dim=-1)
        embedding = embedding * valid[:, None].to(embedding.dtype)
        return embedding, valid
