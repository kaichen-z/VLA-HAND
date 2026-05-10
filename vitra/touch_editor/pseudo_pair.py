from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_CONTACT_VERBS = (
    "grasp",
    "grip",
    "hold",
    "press",
    "push",
    "pull",
    "lift",
    "place",
    "insert",
    "remove",
    "open",
    "close",
)


@dataclass(frozen=True)
class MatchFeature:
    data_id: int
    episode_id: str
    frame_id: int
    instruction: str
    phase: float
    state: np.ndarray
    state_mask: np.ndarray
    action_target: np.ndarray
    action_mask: np.ndarray
    contact_verbs: tuple[str, ...]


@dataclass(frozen=True)
class MatchResult:
    index: int
    score: float
    task_cost: float
    phase_cost: float
    state_cost: float


def extract_contact_verbs(text: str, verbs: Iterable[str] = DEFAULT_CONTACT_VERBS) -> tuple[str, ...]:
    lower = str(text).lower()
    found = []
    for verb in verbs:
        pattern = rf"\b{re.escape(verb.lower())}(?:s|ed|ing)?\b"
        if re.search(pattern, lower):
            found.append(verb.lower())
    return tuple(sorted(set(found)))


def normalized_phase(frame_id: int, episode_len: int) -> float:
    if episode_len <= 1:
        return 0.0
    return float(np.clip(frame_id / float(episode_len - 1), 0.0, 1.0))


def masked_state_cost(
    source_state: np.ndarray,
    source_mask: np.ndarray,
    target_state: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    source = np.nan_to_num(np.asarray(source_state, dtype=np.float32))
    target = np.nan_to_num(np.asarray(target_state, dtype=np.float32))
    valid = np.asarray(source_mask, dtype=bool) & np.asarray(target_mask, dtype=bool)
    if source.shape != target.shape:
        raise ValueError(f"state shape mismatch: {source.shape} vs {target.shape}")
    if valid.shape != source.shape:
        raise ValueError(f"state mask shape mismatch: {valid.shape} vs {source.shape}")
    if not valid.any():
        return 1.0
    diff = source[valid] - target[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def task_match_cost(source_verbs: Iterable[str], target_verbs: Iterable[str]) -> float:
    source = set(source_verbs)
    target = set(target_verbs)
    if not source or not target:
        return 0.5
    return 0.0 if source & target else 1.0


def score_match(
    source: MatchFeature,
    target: MatchFeature,
    *,
    w_task: float = 3.0,
    w_phase: float = 1.0,
    w_state: float = 1.0,
) -> MatchResult:
    task_cost = task_match_cost(source.contact_verbs, target.contact_verbs)
    phase_cost = abs(float(source.phase) - float(target.phase))
    state_cost = masked_state_cost(source.state, source.state_mask, target.state, target.state_mask)
    score = w_task * task_cost + w_phase * phase_cost + w_state * state_cost
    return MatchResult(index=-1, score=float(score), task_cost=float(task_cost), phase_cost=float(phase_cost), state_cost=float(state_cost))


def find_best_match(
    source: MatchFeature,
    candidates: list[MatchFeature],
    *,
    w_task: float = 3.0,
    w_phase: float = 1.0,
    w_state: float = 1.0,
) -> MatchResult:
    if not candidates:
        raise ValueError("At least one GigaHands candidate is required for pseudo-pair matching")
    best: MatchResult | None = None
    for index, candidate in enumerate(candidates):
        result = score_match(source, candidate, w_task=w_task, w_phase=w_phase, w_state=w_state)
        result = MatchResult(
            index=index,
            score=result.score,
            task_cost=result.task_cost,
            phase_cost=result.phase_cost,
            state_cost=result.state_cost,
        )
        if best is None or result.score < best.score:
            best = result
    assert best is not None
    return best


def intersect_action_masks(source_action_mask: np.ndarray, target_action_mask: np.ndarray) -> np.ndarray:
    source = np.asarray(source_action_mask, dtype=bool)
    target = np.asarray(target_action_mask, dtype=bool)
    if source.shape != target.shape:
        raise ValueError(f"action mask shape mismatch: {source.shape} vs {target.shape}")
    return source & target
