from __future__ import annotations

import importlib.util

import numpy as np
import torch

from vitra.guidance.tactile_dps import TactileDPSGuidance
from vitra.guidance.tactile_forward_model import TactileEncoder, TactileForwardModel
from vitra.guidance.tactile_losses import build_tactile_stats


def test_build_tactile_stats_returns_masked_pressure_features():
    touch = torch.zeros((1, 4, 2, 16, 16), dtype=torch.float32)
    mask = torch.tensor([[[True, False], [True, True], [False, True], [True, True]]])
    touch[:, 0, 0] = 1.0
    touch[:, 1, 0] = 2.0
    touch[:, 1, 1] = -3.0
    touch[:, 2, 1] = 4.0
    touch[:, 3, 0] = 5.0
    touch[:, 3, 1] = 6.0

    stats = build_tactile_stats(touch, mask)

    assert stats.shape == (1, 4, 8)
    assert torch.allclose(stats[0, :, 0], torch.tensor([1.0, 1.0, 0.0, 1.0]))
    assert torch.allclose(stats[0, :, 1], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(stats[0, :, 2], torch.tensor([1.0, 2.0, 0.0, 5.0]))
    assert torch.allclose(stats[0, :, 3], torch.tensor([0.0, 3.0, 4.0, 6.0]))
    assert torch.allclose(stats[0, :, 6], torch.tensor([0.0, 1.0, 0.0, 4.0]))


def test_tactile_encoder_and_forward_model_shapes():
    encoder = TactileEncoder(embed_dim=16, stat_dim=8, hidden_dim=32, num_layers=1)
    forward = TactileForwardModel(action_dim=12, state_dim=7, embed_dim=16, stat_dim=8, hidden_dim=32, num_layers=1)
    touch = torch.randn((2, 5, 2, 16, 16), dtype=torch.float32)
    touch_mask = torch.ones((2, 5, 2), dtype=torch.bool)
    state = torch.randn((2, 7), dtype=torch.float32)
    action = torch.randn((2, 5, 12), dtype=torch.float32)
    phase = torch.linspace(0.0, 1.0, 5)[None].expand(2, -1)

    encoded = encoder(touch, touch_mask)
    pred = forward(state, action, phase)

    assert encoded["embedding"].shape == (2, 5, 16)
    assert encoded["stats"].shape == (2, 5, 8)
    assert pred["embedding"].shape == (2, 5, 16)
    assert pred["stats"].shape == (2, 5, 8)


def test_tactile_dps_guidance_masks_fixed_prefix_gradient():
    encoder = TactileEncoder(embed_dim=8, stat_dim=8, hidden_dim=16, num_layers=1)
    forward = TactileForwardModel(action_dim=6, state_dim=4, embed_dim=8, stat_dim=8, hidden_dim=16, num_layers=1)
    guidance = TactileDPSGuidance(
        tactile_encoder=encoder,
        tactile_forward_model=forward,
        current_state=torch.zeros((1, 4), dtype=torch.float32),
        touch_pressure=torch.ones((1, 6, 2, 16, 16), dtype=torch.float32),
        touch_mask=torch.ones((1, 6, 2), dtype=torch.bool),
        chunk_phase=torch.linspace(0.0, 1.0, 6)[None],
        edit_start_idx=3,
        action_mask=torch.ones((1, 6, 6), dtype=torch.bool),
        lambda_embed=1.0,
        lambda_stats=1.0,
    )
    x_t = torch.randn((1, 6, 6), dtype=torch.float32, requires_grad=True)
    pred_xstart = x_t * 1.0

    grad, loss, metrics = guidance.gradient(x_t, pred_xstart)

    assert grad.shape == x_t.shape
    assert loss.ndim == 0
    assert metrics["embed_loss"].ndim == 0
    assert torch.allclose(grad[:, :3], torch.zeros_like(grad[:, :3]))
    assert torch.isfinite(grad[:, 3:]).all()


def test_cfg_wrapper_keeps_tactile_dps_gradient_on_conditional_half_only():
    from vitra.guidance.polynomial_guidance import CFGAwareGuidanceWrapper

    batch_size = 2
    encoder = TactileEncoder(embed_dim=8, stat_dim=8, hidden_dim=16, num_layers=1)
    forward = TactileForwardModel(action_dim=6, state_dim=4, embed_dim=8, stat_dim=8, hidden_dim=16, num_layers=1)
    guidance = TactileDPSGuidance(
        tactile_encoder=encoder,
        tactile_forward_model=forward,
        current_state=torch.zeros((batch_size, 4), dtype=torch.float32),
        touch_pressure=torch.ones((batch_size, 6, 2, 16, 16), dtype=torch.float32),
        touch_mask=torch.ones((batch_size, 6, 2), dtype=torch.bool),
        chunk_phase=torch.linspace(0.0, 1.0, 6)[None].expand(batch_size, -1),
        edit_start_idx=3,
        action_mask=torch.ones((batch_size, 6, 6), dtype=torch.bool),
        lambda_embed=1.0,
        lambda_stats=1.0,
    )
    wrapped = CFGAwareGuidanceWrapper(guidance, original_batch_size=batch_size, using_cfg=True)
    x_t = torch.randn((batch_size * 2, 6, 6), dtype=torch.float32, requires_grad=True)
    pred_xstart = x_t * 1.0

    grad, loss, metrics = wrapped.gradient(x_t, pred_xstart)

    assert grad.shape == x_t.shape
    assert loss.ndim == 0
    assert metrics["cfg/original_batch_size"].item() == batch_size
    assert torch.allclose(grad[:batch_size, :3], torch.zeros_like(grad[:batch_size, :3]))
    assert grad[:batch_size, 3:].abs().sum() > 0
    assert torch.allclose(grad[batch_size:], torch.zeros_like(grad[batch_size:]))


def test_tactile_replay_dataset_requires_cached_action_features_for_diffusion(tmp_path):
    from vitra.guidance.tactile_replay_dataset import TactileReplayCacheDataset

    np.savez(
        tmp_path / "sample_00000000.npz",
        a_base=np.zeros((4, 6), dtype=np.float32),
        a_target=np.zeros((4, 6), dtype=np.float32),
        action_mask=np.ones((4, 6), dtype=bool),
        current_state=np.zeros((7,), dtype=np.float32),
        current_state_mask=np.ones((7,), dtype=bool),
        touch_pressure=np.zeros((4, 2, 16, 16), dtype=np.float32),
        touch_mask=np.ones((4, 2), dtype=bool),
        chunk_phase=np.linspace(0.0, 1.0, 4, dtype=np.float32),
        edit_start_idx=np.asarray(2, dtype=np.int64),
    )

    dataset = TactileReplayCacheDataset(tmp_path, require_action_features=True)

    try:
        _ = dataset[0]
    except KeyError as exc:
        assert "action_features" in str(exc)
    else:
        raise AssertionError("diffusion DPS replay should require cached action_features")


def test_tactile_replay_dataset_squeezes_cached_action_feature_batch_axis(tmp_path):
    from vitra.guidance.tactile_replay_dataset import TactileReplayCacheDataset

    np.savez(
        tmp_path / "sample_00000000.npz",
        a_base=np.zeros((4, 6), dtype=np.float32),
        a_target=np.zeros((4, 6), dtype=np.float32),
        action_mask=np.ones((4, 6), dtype=bool),
        current_state=np.zeros((7,), dtype=np.float32),
        current_state_mask=np.ones((7,), dtype=bool),
        touch_pressure=np.zeros((4, 2, 16, 16), dtype=np.float32),
        touch_mask=np.ones((4, 2), dtype=bool),
        chunk_phase=np.linspace(0.0, 1.0, 4, dtype=np.float32),
        edit_start_idx=np.asarray(2, dtype=np.int64),
        action_features=np.zeros((1, 1, 8), dtype=np.float32),
    )

    item = TactileReplayCacheDataset(tmp_path, require_action_features=True)[0]

    assert item["action_features"].shape == (1, 8)


def test_tactile_dps_diffusion_replanning_script_assets_are_present():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/evaluate_tactile_dps_diffusion_replanning.py"
    assert script.exists()

    spec = importlib.util.spec_from_file_location("evaluate_tactile_dps_diffusion_replanning", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    for name in [
        "make_prefix_tensors",
        "build_tactile_guidance",
        "evaluate_ablation",
        "load_vla_for_diffusion_replanning",
    ]:
        assert hasattr(module, name)
