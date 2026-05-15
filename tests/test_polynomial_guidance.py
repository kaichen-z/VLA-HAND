import torch


def test_tau_to_index_clamps_and_rounds_normalized_action_time():
    from vitra.guidance.polynomial_guidance import tau_to_index

    assert tau_to_index(0.33, 16) == 5
    assert tau_to_index(0.66, 16) == 10
    assert tau_to_index(1.0, 16) == 15
    assert tau_to_index(-1.0, 16) == 0
    assert tau_to_index(2.0, 16) == 15


def test_polynomial_region_loss_is_zero_inside_and_positive_outside():
    from vitra.guidance.polynomial_guidance import (
        PolynomialGuidanceConfig,
        PolynomialRegionLoss,
        QuadraticRegion,
    )

    guidance = PolynomialRegionLoss(
        PolynomialGuidanceConfig(
            guide_dims=[0, 1],
            regions=[
                QuadraticRegion(
                    tau=1.0,
                    center=[0.0, 0.0],
                    Q=[[1.0, 0.0], [0.0, 1.0]],
                    radius2=1.0,
                    name="target",
                )
            ],
        )
    )

    inside = torch.zeros(1, 4, 3)
    outside = torch.zeros(1, 4, 3)
    outside[:, -1, :2] = torch.tensor([2.0, 0.0])

    inside_loss, inside_metrics = guidance(inside)
    outside_loss, outside_metrics = guidance(outside)

    assert torch.allclose(inside_loss, torch.tensor(0.0))
    assert outside_loss > inside_loss
    assert inside_metrics["target/success_rate"].item() == 1.0
    assert outside_metrics["target/success_rate"].item() == 0.0


def test_cfg_aware_guidance_wrapper_only_scores_conditional_half():
    from vitra.guidance.polynomial_guidance import (
        CFGAwareGuidanceWrapper,
        PolynomialGuidanceConfig,
        PolynomialRegionLoss,
        QuadraticRegion,
    )

    base = PolynomialRegionLoss(
        PolynomialGuidanceConfig(
            guide_dims=[0, 1],
            regions=[
                QuadraticRegion(
                    tau=0.0,
                    center=[0.0, 0.0],
                    Q=[[1.0, 0.0], [0.0, 1.0]],
                    radius2=1.0,
                )
            ],
        )
    )
    wrapped = CFGAwareGuidanceWrapper(base, original_batch_size=1, using_cfg=True)

    pred_xstart = torch.zeros(2, 4, 3)
    pred_xstart[0, 0, :2] = torch.tensor([0.0, 0.0])
    pred_xstart[1, 0, :2] = torch.tensor([10.0, 0.0])

    loss, metrics = wrapped(pred_xstart)

    assert torch.allclose(loss, torch.tensor(0.0))
    assert metrics["cfg/original_batch_size"].item() == 1


def test_temporal_tail_mask_keeps_region_gradient_from_changing_prefix():
    from vitra.guidance.polynomial_guidance import (
        PolynomialGuidanceConfig,
        PolynomialRegionLoss,
        QuadraticRegion,
    )

    guidance = PolynomialRegionLoss(
        PolynomialGuidanceConfig(
            guide_dims=[0, 1],
            regions=[
                QuadraticRegion(
                    tau=0.66,
                    center=[0.0, 0.0],
                    Q=[[1.0, 0.0], [0.0, 1.0]],
                    radius2=0.01,
                )
            ],
            temporal_mask="tail",
        )
    )
    x_in = torch.zeros(1, 4, 3, requires_grad=True)
    pred_xstart = x_in * 1.0
    pred_xstart[:, 2, :2] = pred_xstart[:, 2, :2] + torch.tensor([2.0, 0.0])

    grad, loss, metrics = guidance.gradient(x_in, pred_xstart)

    assert loss > 0
    assert metrics["tau_0.66/idx"].item() == 2
    assert torch.allclose(grad[:, :2, :], torch.zeros_like(grad[:, :2, :]))
    assert grad[:, 2:, :].abs().sum() > 0


def test_compute_region_metrics_reports_guidance_improvement():
    import numpy as np
    from vitra.guidance.metrics import compute_region_metrics
    from vitra.guidance.polynomial_guidance import PolynomialGuidanceConfig, QuadraticRegion

    config = PolynomialGuidanceConfig(
        guide_dims=[0, 1],
        regions=[
            QuadraticRegion(
                tau=1.0,
                center=[0.0, 0.0],
                Q=[[1.0, 0.0], [0.0, 1.0]],
                radius2=1.0,
            )
        ],
    )
    baseline = np.zeros((4, 3), dtype=np.float32)
    guided = np.zeros((4, 3), dtype=np.float32)
    baseline[-1, :2] = np.asarray([3.0, 0.0], dtype=np.float32)

    baseline_metrics = compute_region_metrics(baseline, config)
    guided_metrics = compute_region_metrics(guided, config)

    assert baseline_metrics["success_all"] is False
    assert guided_metrics["success_all"] is True
    assert guided_metrics["violation_mean"] < baseline_metrics["violation_mean"]


def test_vitra_sampling_interfaces_expose_replanning_guidance_hooks_only():
    import inspect

    from vitra.models.action_model.diffusion_policy import DiffusionPolicy
    from vitra.models.action_model.gaussian_diffusion import GaussianDiffusion
    from vitra.models.vla.vitra_paligemma import VITRA_Paligemma

    sample_sig = inspect.signature(DiffusionPolicy.sample)
    predict_sig = inspect.signature(VITRA_Paligemma.predict_action)

    for name in [
        "guidance_fn",
        "guidance_scale",
        "guidance_start_frac",
        "guidance_end_frac",
        "guidance_grad_clip",
        "return_guidance_trace",
        "fixed_actions",
        "fixed_action_mask",
        "return_replan_trace",
    ]:
        assert name in sample_sig.parameters
        assert name in predict_sig.parameters

    assert hasattr(GaussianDiffusion, "ddim_sample_loop_replanning_guided")
    assert not hasattr(GaussianDiffusion, "ddim_sample_loop_velocity_guided")


def test_one_shot_polynomial_guidance_entrypoint_is_removed():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inference_polynomial_guidance.py"
    config = root / "configs/polynomial_guidance_example.json"

    assert config.exists()
    assert not script.exists()


def test_diffusion_policy_non_cfg_branch_uses_action_features_as_condition():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "vitra/models/action_model/diffusion_policy.py").read_text(encoding="utf-8")

    assert "else:\n            z = action_features" in text
