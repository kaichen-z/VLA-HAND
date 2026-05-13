from __future__ import annotations

from pathlib import Path

import numpy as np

from .polynomial_guidance import PolynomialGuidanceConfig, tau_to_index


def plot_trajectory_xy(
    baseline: np.ndarray,
    guided: np.ndarray,
    config: PolynomialGuidanceConfig,
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dims = list(config.guide_dims)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(baseline[:, dims[0]], baseline[:, dims[1]], "o-", label="baseline", alpha=0.75)
    ax.plot(guided[:, dims[0]], guided[:, dims[1]], "o-", label="guided", alpha=0.75)

    all_xy = np.concatenate([baseline[:, dims], guided[:, dims]], axis=0)
    lo = all_xy.min(axis=0) - 1.0
    hi = all_xy.max(axis=0) + 1.0
    xs = np.linspace(lo[0], hi[0], 180)
    ys = np.linspace(lo[1], hi[1], 180)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    for region in config.regions:
        c = np.asarray(region.center, dtype=np.float64).reshape(2)
        Q = np.asarray(region.Q, dtype=np.float64).reshape(2, 2)
        du = flat - c
        p = np.sum((du @ Q.T) * du, axis=-1).reshape(grid_x.shape) - float(region.radius2)
        ax.contour(grid_x, grid_y, p, levels=[0.0], linewidths=1.5)
        idx = tau_to_index(region.tau, baseline.shape[0])
        ax.scatter(guided[idx, dims[0]], guided[idx, dims[1]], s=90, marker="x")
        ax.text(c[0], c[1], region.name or f"tau={region.tau}", fontsize=8)

    ax.set_xlabel(f"action dim {dims[0]}")
    ax.set_ylabel(f"action dim {dims[1]}")
    ax.set_title("Polynomial guidance trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_guidance_loss_curve(trace: list[dict], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active = [item for item in trace if item.get("guidance_active")]
    xs = [item["step_id"] for item in active]
    ys = [item.get("loss", 0.0) for item in active]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("DDIM step id")
    ax.set_ylabel("polynomial loss")
    ax.set_title("Guidance loss during active DDIM steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
