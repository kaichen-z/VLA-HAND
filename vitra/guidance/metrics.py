from __future__ import annotations

from typing import Any

import numpy as np

from .polynomial_guidance import PolynomialGuidanceConfig, tau_to_index


def _region_p(action: np.ndarray, config: PolynomialGuidanceConfig, region) -> tuple[int, float, float]:
    T = action.shape[0]
    idx = tau_to_index(region.tau, T)
    u = action[idx, list(config.guide_dims)].astype(np.float64)
    c = np.asarray(region.center, dtype=np.float64).reshape(2)
    Q = np.asarray(region.Q, dtype=np.float64).reshape(2, 2)
    du = u - c
    p = float(du @ Q.T @ du - float(region.radius2))
    return idx, p, max(p, 0.0)


def compute_region_metrics(
    action: np.ndarray,
    guidance_config: PolynomialGuidanceConfig,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    per_region = {}
    violations = []
    successes = []
    for region in guidance_config.regions:
        idx, p, violation = _region_p(action, guidance_config, region)
        name = region.name or f"tau_{region.tau}"
        success = p <= tolerance
        per_region[name] = {
            "idx": idx,
            "p": p,
            "violation": violation,
            "success": bool(success),
        }
        violations.append(violation)
        successes.append(success)
    return {
        "violation_mean": float(np.mean(violations)) if violations else 0.0,
        "violation_max": float(np.max(violations)) if violations else 0.0,
        "success_all": bool(all(successes)) if successes else True,
        "per_region": per_region,
    }


def smoothness(action: np.ndarray) -> float:
    if action.shape[0] < 2:
        return 0.0
    return float(((action[1:] - action[:-1]) ** 2).sum(axis=-1).mean())


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))
