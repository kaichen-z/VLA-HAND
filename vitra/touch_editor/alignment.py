from __future__ import annotations

import numpy as np


def median_positive_delta(timestamps: np.ndarray) -> float | None:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 2:
        return None
    deltas = np.diff(timestamps)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return None
    return float(np.median(deltas))


def default_alignment_tolerance(
    target_timestamps: np.ndarray,
    source_timestamps: np.ndarray,
) -> float:
    deltas = [
        value
        for value in (
            median_positive_delta(target_timestamps),
            median_positive_delta(source_timestamps),
        )
        if value is not None
    ]
    if not deltas:
        return 1.0 / 60.0
    return 0.5 * max(deltas)


def nearest_timestamp_indices(
    target_timestamps: np.ndarray,
    source_timestamps: np.ndarray,
    tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each target timestamp to the nearest source timestamp within tolerance."""
    target = np.asarray(target_timestamps, dtype=np.float64)
    source = np.asarray(source_timestamps, dtype=np.float64)
    if target.ndim != 1 or source.ndim != 1:
        raise ValueError("target_timestamps and source_timestamps must be 1D")
    if source.size == 0:
        return np.full(target.shape, -1, dtype=np.int64), np.zeros(target.shape, dtype=bool)
    if tolerance is None:
        tolerance = default_alignment_tolerance(target, source)

    order = np.argsort(source)
    sorted_source = source[order]
    positions = np.searchsorted(sorted_source, target)
    indices = np.full(target.shape, -1, dtype=np.int64)
    valid = np.zeros(target.shape, dtype=bool)
    for i, (timestamp, pos) in enumerate(zip(target, positions)):
        candidates = []
        if pos < sorted_source.size:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates or not np.isfinite(timestamp):
            continue
        best_pos = min(candidates, key=lambda candidate: abs(sorted_source[candidate] - timestamp))
        if abs(sorted_source[best_pos] - timestamp) <= tolerance:
            indices[i] = int(order[best_pos])
            valid[i] = True
    return indices, valid


def align_touch_to_timestamps(
    touch_pressure: np.ndarray,
    touch_mask: np.ndarray,
    target_timestamps: np.ndarray,
    touch_timestamps: np.ndarray,
    tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align tactile pressure/mask rows to target frame timestamps."""
    pressure = np.asarray(touch_pressure, dtype=np.float32)
    mask = np.asarray(touch_mask, dtype=bool)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)
    touch_timestamps = np.asarray(touch_timestamps, dtype=np.float64)
    if pressure.ndim != 4 or pressure.shape[1:] != (2, 16, 16):
        raise ValueError(f"touch_pressure must be shaped [T,2,16,16], got {pressure.shape}")
    if mask.ndim != 2 or mask.shape[1] != 2:
        raise ValueError(f"touch_mask must be shaped [T,2], got {mask.shape}")
    if pressure.shape[0] != mask.shape[0]:
        raise ValueError("touch_pressure and touch_mask length mismatch")
    if pressure.shape[0] != touch_timestamps.shape[0]:
        raise ValueError("touch_pressure and touch_timestamps length mismatch")

    indices, valid = nearest_timestamp_indices(target_timestamps, touch_timestamps, tolerance)
    aligned_pressure = np.zeros((len(target_timestamps), 2, 16, 16), dtype=np.float32)
    aligned_mask = np.zeros((len(target_timestamps), 2), dtype=bool)
    if valid.any():
        aligned_pressure[valid] = pressure[indices[valid]]
        aligned_mask[valid] = mask[indices[valid]]
    aligned_mask &= valid[:, None]
    return aligned_pressure, aligned_mask, indices, valid


def clipped_future_window_indices(frame_id: int, chunk_len: int, episode_len: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.arange(int(frame_id), int(frame_id) + int(chunk_len), dtype=np.int64)
    if episode_len <= 0:
        raise ValueError("episode_len must be positive")
    oob = (raw < 0) | (raw >= episode_len)
    return raw.clip(0, episode_len - 1), oob
