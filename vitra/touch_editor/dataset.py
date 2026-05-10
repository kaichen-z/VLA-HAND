from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TouchEditorCacheDataset(Dataset):
    """Loads cached frozen-VLA samples for residual touch-editor training.

    Each ``.npz`` sample must contain:
      - a_base: [T, 192]
      - a_target: [T, 192]
      - residual_target: [T, 192]
      - action_mask: [T, 192]
      - current_state: [212]
      - current_state_mask: [212]
      - touch_pressure: [H, 2, 16, 16]
      - touch_mask: [H, 2]
      - future_mask: [T, 192]

    Optional keys:
      - chunk_phase: scalar or [T]
      - edit_start_idx: scalar
      - action_frame_indices/action_timestamps/touch alignment metadata
    """

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self.paths = sorted(self.cache_root.rglob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No touch-editor .npz cache files found under {self.cache_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as payload:
            sample = {key: payload[key] for key in payload.files}
        out = {
            "a_base": torch.tensor(sample["a_base"], dtype=torch.float32),
            "a_target": torch.tensor(sample.get("a_target", sample.get("a_demo")), dtype=torch.float32),
            "action_mask": torch.tensor(sample["action_mask"], dtype=torch.bool),
            "current_state": torch.tensor(sample["current_state"], dtype=torch.float32),
            "current_state_mask": torch.tensor(sample.get("current_state_mask", np.ones_like(sample["current_state"], dtype=bool)), dtype=torch.bool),
            "touch_pressure": torch.tensor(sample["touch_pressure"], dtype=torch.float32),
            "touch_mask": torch.tensor(sample["touch_mask"], dtype=torch.bool),
            "path": str(path),
        }
        if "residual_target" in sample:
            out["residual_target"] = torch.tensor(sample["residual_target"], dtype=torch.float32)
        else:
            out["residual_target"] = out["a_target"] - out["a_base"]
        if "future_mask" in sample:
            out["future_mask"] = torch.tensor(sample["future_mask"], dtype=torch.float32)
        elif "edit_mask" in sample:
            out["future_mask"] = torch.tensor(sample["edit_mask"], dtype=torch.float32)
        else:
            out["future_mask"] = torch.ones_like(out["a_base"])
        if "chunk_phase" in sample:
            out["chunk_phase"] = torch.tensor(sample["chunk_phase"], dtype=torch.float32)
        else:
            out["chunk_phase"] = torch.linspace(0.0, 1.0, out["a_base"].shape[0], dtype=torch.float32)
        if "edit_start_idx" in sample:
            out["edit_start_idx"] = torch.tensor(sample["edit_start_idx"], dtype=torch.long)
        else:
            first_future = torch.where(out["future_mask"].any(dim=-1))[0]
            out["edit_start_idx"] = first_future[0].to(torch.long) if len(first_future) else torch.tensor(out["a_base"].shape[0], dtype=torch.long)
        for key in ("action_frame_indices", "touch_aligned_indices"):
            if key in sample:
                out[key] = torch.tensor(sample[key], dtype=torch.long)
        for key in ("action_timestamps", "touch_aligned_timestamps"):
            if key in sample:
                out[key] = torch.tensor(sample[key], dtype=torch.float64)
        if "touch_alignment_valid" in sample:
            out["touch_alignment_valid"] = torch.tensor(sample["touch_alignment_valid"], dtype=torch.bool)
        return out
