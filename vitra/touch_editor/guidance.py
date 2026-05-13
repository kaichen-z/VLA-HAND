from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from vitra.touch_editor.model import ResidualTouchEditor


@dataclass
class TouchGuidanceResult:
    a_edit: torch.Tensor
    a_history: list[torch.Tensor]
    deltas: list[torch.Tensor]
    edit_indices: list[int]
    future_masks: list[torch.Tensor]


def seconds_to_chunk_index(seconds: float, fps: float, chunk_len: int) -> int:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    idx = int(round(float(seconds) * float(fps)))
    return max(0, min(int(chunk_len), idx))


def load_touch_editor(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    *,
    action_dim: int = 192,
    state_dim: int = 212,
) -> ResidualTouchEditor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model = ResidualTouchEditor(
        action_dim=int(args.get("action_dim", action_dim)),
        state_dim=int(args.get("state_dim", state_dim)),
        touch_feature_dim=int(args.get("touch_feature_dim", 128)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        num_layers=int(args.get("num_layers", 2)),
        condition_mode=str(args.get("condition_mode", "full")),
    ).to(device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return model.eval()


def make_future_mask(action_mask: torch.Tensor, edit_start_idx: int) -> torch.Tensor:
    if action_mask.ndim != 3:
        raise ValueError(f"action_mask must be shaped [B,T,D], got {tuple(action_mask.shape)}")
    edit_start_idx = max(0, min(int(edit_start_idx), int(action_mask.shape[1])))
    future_mask = torch.zeros_like(action_mask, dtype=torch.float32)
    future_mask[:, edit_start_idx:] = action_mask[:, edit_start_idx:].to(torch.float32)
    return future_mask


def make_chunk_phase(batch_size: int, chunk_len: int, device: torch.device) -> torch.Tensor:
    if chunk_len <= 1:
        phase = torch.zeros((chunk_len,), dtype=torch.float32, device=device)
    else:
        phase = torch.linspace(0.0, 1.0, chunk_len, dtype=torch.float32, device=device)
    return phase[None].expand(batch_size, -1)


def _as_batch(value: torch.Tensor, expected_ndim: int) -> torch.Tensor:
    if value.ndim == expected_ndim - 1:
        return value.unsqueeze(0)
    if value.ndim == expected_ndim:
        return value
    raise ValueError(f"Expected tensor with {expected_ndim - 1} or {expected_ndim} dims, got {tuple(value.shape)}")


def _match_batch(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] == 1:
        return value.expand(batch_size, *value.shape[1:])
    raise ValueError(f"{name} batch size {value.shape[0]} does not match action batch size {batch_size}")


def _touch_history_until(
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    edit_start_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    history_len = max(1, min(int(edit_start_idx) + 1, int(touch_pressure.shape[1])))
    return touch_pressure[:, :history_len], touch_mask[:, :history_len]


@torch.no_grad()
def apply_touch_editor_once(
    editor: ResidualTouchEditor,
    a_base: torch.Tensor,
    current_state: torch.Tensor,
    current_state_mask: torch.Tensor,
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    action_mask: torch.Tensor,
    edit_start_idx: int,
    use_full_touch_window: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one residual edit to the unexecuted suffix of a normalized VITRA chunk."""
    device = next(editor.parameters()).device
    a_base = _as_batch(a_base, 3).to(device=device, dtype=torch.float32)
    current_state = _as_batch(current_state, 2).to(device=device, dtype=torch.float32)
    current_state_mask = _as_batch(current_state_mask, 2).to(device=device, dtype=torch.bool)
    touch_pressure = _as_batch(touch_pressure, 5).to(device=device, dtype=torch.float32)
    touch_mask = _as_batch(touch_mask, 3).to(device=device, dtype=torch.bool)
    action_mask = _as_batch(action_mask, 3).to(device=device, dtype=torch.bool)

    batch_size = int(a_base.shape[0])
    current_state = _match_batch(current_state, batch_size, "current_state")
    current_state_mask = _match_batch(current_state_mask, batch_size, "current_state_mask")
    touch_pressure = _match_batch(touch_pressure, batch_size, "touch_pressure")
    touch_mask = _match_batch(touch_mask, batch_size, "touch_mask")
    action_mask = _match_batch(action_mask, batch_size, "action_mask")
    if a_base.shape[1:] != action_mask.shape[1:]:
        raise ValueError(f"a_base/action_mask shape mismatch: {tuple(a_base.shape)} vs {tuple(action_mask.shape)}")

    edit_start_idx = max(0, min(int(edit_start_idx), int(a_base.shape[1])))
    future_mask = make_future_mask(action_mask, edit_start_idx).to(device)
    if use_full_touch_window:
        touch_window, touch_window_mask = touch_pressure, touch_mask
    else:
        touch_window, touch_window_mask = _touch_history_until(touch_pressure, touch_mask, edit_start_idx)
    chunk_phase = make_chunk_phase(a_base.shape[0], a_base.shape[1], device)
    delta = editor(
        a_base=a_base,
        current_state=current_state,
        current_state_mask=current_state_mask,
        touch_pressure=touch_window,
        touch_mask=touch_window_mask,
        chunk_phase=chunk_phase,
        future_mask=future_mask,
        action_mask=action_mask,
    )
    a_edit = a_base + future_mask.to(delta.dtype) * delta
    return a_edit, delta, future_mask


@torch.no_grad()
def apply_touch_guidance_schedule(
    editor: ResidualTouchEditor,
    a_base: torch.Tensor,
    current_state: torch.Tensor,
    current_state_mask: torch.Tensor,
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    fps: float,
    edit_times: Iterable[float] = (0.33, 0.66),
    use_full_touch_window: bool = False,
) -> TouchGuidanceResult:
    """Apply sequential touch edits at second-based offsets within a VITRA chunk."""
    a_work = _as_batch(a_base, 3)
    chunk_len = int(a_work.shape[1])
    a_history: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    future_masks: list[torch.Tensor] = []
    edit_indices: list[int] = []
    a_edit = a_base
    for edit_time in edit_times:
        edit_idx = seconds_to_chunk_index(edit_time, fps, chunk_len)
        a_edit, delta, future_mask = apply_touch_editor_once(
            editor=editor,
            a_base=a_edit,
            current_state=current_state,
            current_state_mask=current_state_mask,
            touch_pressure=touch_pressure,
            touch_mask=touch_mask,
            action_mask=action_mask,
            edit_start_idx=edit_idx,
            use_full_touch_window=use_full_touch_window,
        )
        a_history.append(a_edit.detach().cpu())
        deltas.append(delta.detach().cpu())
        future_masks.append(future_mask.detach().cpu())
        edit_indices.append(edit_idx)
    return TouchGuidanceResult(
        a_edit=a_edit,
        a_history=a_history,
        deltas=deltas,
        edit_indices=edit_indices,
        future_masks=future_masks,
    )
