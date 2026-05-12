from __future__ import annotations

import numpy as np


def build_future_mask(action_mask, edit_start_idx: int) -> np.ndarray:
    """Return an editable suffix mask with the same shape as ``action_mask``."""
    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"action_mask must be shaped [T,D], got {mask.shape}")
    edit_start_idx = max(0, min(int(edit_start_idx), int(mask.shape[0])))
    future_mask = np.zeros(mask.shape, dtype=np.float32)
    future_mask[edit_start_idx:] = mask[edit_start_idx:].astype(np.float32)
    return future_mask


def chunk_phase(chunk_len: int) -> np.ndarray:
    """Normalized phase for each action timestep in a chunk."""
    chunk_len = int(chunk_len)
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}")
    if chunk_len == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, chunk_len, dtype=np.float32)
