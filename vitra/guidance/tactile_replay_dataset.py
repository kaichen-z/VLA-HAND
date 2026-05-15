from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TactileReplayCacheDataset(Dataset):
    """OpenTouch replay cache used for tactile guidance experiments."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        max_samples: int | None = None,
        high_contact_quantile: float | None = None,
        require_action_features: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.require_action_features = bool(require_action_features)
        paths = sorted(self.cache_root.rglob("*.npz"))
        if high_contact_quantile is not None:
            paths = self._filter_high_contact(paths, high_contact_quantile)
        if max_samples is not None:
            paths = paths[: max(0, int(max_samples))]
        if not paths:
            raise FileNotFoundError(f"No .npz replay cache files found under {self.cache_root}")
        self.paths = paths

    @staticmethod
    def _contact_score(path: Path) -> float:
        with np.load(path, allow_pickle=False) as payload:
            if "observed_touch_contact_score" in payload:
                score = float(np.asarray(payload["observed_touch_contact_score"]).item())
            else:
                pressure = np.asarray(payload["touch_pressure"], dtype=np.float32)
                mask = np.asarray(payload["touch_mask"], dtype=bool)
                valid = mask[..., None, None]
                denom = float(valid.sum() * pressure.shape[-1] * pressure.shape[-2])
                score = float((np.abs(pressure) * valid).sum() / denom) if denom > 0 else 0.0
        return score

    @classmethod
    def _filter_high_contact(cls, paths: list[Path], quantile: float) -> list[Path]:
        if not paths:
            return paths
        scores = np.asarray([cls._contact_score(path) for path in paths], dtype=np.float32)
        threshold = float(np.quantile(scores, float(quantile)))
        return [path for path, score in zip(paths, scores) if float(score) >= threshold]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as payload:
            sample = {key: payload[key] for key in payload.files}
        if self.require_action_features and "action_features" not in sample:
            raise KeyError(
                f"{path} does not contain action_features. Regenerate the replay cache with "
                "scripts/cache_touch_editor_base_actions.py before diffusion DPS replanning."
            )
        out = {
            "a_base": torch.tensor(sample["a_base"], dtype=torch.float32),
            "a_target": torch.tensor(sample["a_target"], dtype=torch.float32),
            "action_mask": torch.tensor(sample["action_mask"], dtype=torch.bool),
            "current_state": torch.tensor(sample["current_state"], dtype=torch.float32),
            "current_state_mask": torch.tensor(sample.get("current_state_mask", np.ones_like(sample["current_state"], dtype=bool)), dtype=torch.bool),
            "touch_pressure": torch.tensor(sample["touch_pressure"], dtype=torch.float32),
            "touch_mask": torch.tensor(sample["touch_mask"], dtype=torch.bool),
            "future_mask": torch.tensor(sample.get("future_mask", np.ones_like(sample["a_base"], dtype=np.float32)), dtype=torch.float32),
            "chunk_phase": torch.tensor(sample.get("chunk_phase", np.linspace(0.0, 1.0, sample["a_base"].shape[0])), dtype=torch.float32),
            "edit_start_idx": torch.tensor(sample.get("edit_start_idx", 0), dtype=torch.long),
            "path": str(path),
        }
        if "action_features" in sample:
            action_features = np.asarray(sample["action_features"], dtype=np.float32)
            while action_features.ndim > 2 and action_features.shape[0] == 1:
                action_features = action_features[0]
            out["action_features"] = torch.tensor(action_features, dtype=torch.float32)
        return out


def move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
