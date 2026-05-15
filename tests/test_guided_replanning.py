import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_test_diffusion():
    from vitra.models.action_model.gaussian_diffusion import (
        GaussianDiffusion,
        LossType,
        ModelMeanType,
        ModelVarType,
        get_named_beta_schedule,
    )

    return GaussianDiffusion(
        betas=get_named_beta_schedule("squaredcos_cap_v2", 8),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


class ZeroStartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


class IdentityStartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, **kwargs):
        return x


def test_replanning_interfaces_are_exposed():
    from vitra.models.action_model.diffusion_policy import DiffusionPolicy
    from vitra.models.action_model.gaussian_diffusion import GaussianDiffusion
    from vitra.models.vla.vitra_encoder_student import VITRA_EncoderStudent
    from vitra.models.vla.vitra_paligemma import VITRA_Paligemma

    sample_sig = inspect.signature(DiffusionPolicy.sample)
    predict_sig = inspect.signature(VITRA_Paligemma.predict_action)
    student_forward_sig = inspect.signature(VITRA_EncoderStudent._forward_act_model)

    for name in ["fixed_actions", "fixed_action_mask", "return_replan_trace"]:
        assert name in sample_sig.parameters
        assert name in predict_sig.parameters
        assert name in student_forward_sig.parameters

    assert hasattr(GaussianDiffusion, "ddim_sample_loop_replanning_guided")
    assert not hasattr(GaussianDiffusion, "ddim_sample_loop_velocity_guided")


def test_diffusion_policy_rejects_one_shot_guidance_without_fixed_prefix():
    from vitra.guidance.polynomial_guidance import (
        PolynomialGuidanceConfig,
        PolynomialRegionLoss,
        QuadraticRegion,
    )
    from vitra.models.action_model.diffusion_policy import DiffusionPolicy

    policy = DiffusionPolicy(
        token_size=8,
        model_type="DiT-T",
        in_channels=4,
        future_action_window_size=3,
        use_state=None,
        diffusion_steps=8,
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
                )
            ],
        )
    )

    with torch.no_grad():
        try:
            policy.sample(
                action_features=torch.zeros(1, 1, 8),
                cfg_scale=1.0,
                current_state=None,
                current_state_mask=None,
                use_ddim=True,
                num_ddim_steps=2,
                action_masks=torch.ones(1, 4, 4),
                guidance_fn=guidance,
                guidance_scale=1.0,
            )
        except ValueError as exc:
            assert "replanning" in str(exc).lower()
        else:
            raise AssertionError("one-shot guidance without fixed_actions should be rejected")


def test_cached_diffusion_only_interfaces_are_exposed():
    from vitra.models.vla.vitra_encoder_student import VITRA_EncoderStudent
    from vitra.models.vla.vitra_paligemma import VITRA_Paligemma

    for cls in [VITRA_Paligemma, VITRA_EncoderStudent]:
        assert hasattr(cls, "encode_action_condition")
        assert hasattr(cls, "sample_action_from_condition")

        encode_sig = inspect.signature(cls.encode_action_condition)
        sample_sig = inspect.signature(cls.sample_action_from_condition)
        for name in ["image", "instruction", "current_state", "current_state_mask", "fov"]:
            assert name in encode_sig.parameters
        for name in ["action_features", "action_mask_torch", "fixed_actions", "fixed_action_mask"]:
            assert name in sample_sig.parameters


def test_replanning_sampler_preserves_fixed_prefix():
    diffusion = make_test_diffusion()
    model = ZeroStartModel()
    fixed = torch.zeros(1, 4, 3)
    fixed[:, :2, :] = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    fixed_mask = torch.zeros_like(fixed)
    fixed_mask[:, :2, :] = 1.0
    noise = torch.randn_like(fixed)
    prefix_noise = torch.randn_like(fixed)

    sample, trace = diffusion.ddim_sample_loop_replanning_guided(
        model,
        fixed.shape,
        noise=noise,
        clip_denoised=False,
        device=torch.device("cpu"),
        eta=0.0,
        fixed_xstart=fixed,
        fixed_mask=fixed_mask,
        prefix_noise=prefix_noise,
        guidance_fn=None,
        guidance_scale=0.0,
    )

    assert torch.allclose(sample[:, :2, :], fixed[:, :2, :], atol=1e-6)
    assert torch.allclose(sample[:, 2:, :], torch.zeros_like(sample[:, 2:, :]), atol=1e-5)
    assert trace
    assert trace[-1]["fixed_prefix_error_max"] < 1e-6


def test_replanning_guidance_gradient_does_not_change_fixed_prefix():
    from vitra.guidance.polynomial_guidance import (
        PolynomialGuidanceConfig,
        PolynomialRegionLoss,
        QuadraticRegion,
    )

    diffusion = make_test_diffusion()
    model = IdentityStartModel()
    fixed = torch.zeros(1, 4, 4)
    fixed[:, :2, :] = 3.0
    fixed_mask = torch.zeros_like(fixed)
    fixed_mask[:, :2, :] = 1.0
    guidance = PolynomialRegionLoss(
        PolynomialGuidanceConfig(
            guide_dims=[0, 1],
            regions=[
                QuadraticRegion(
                    tau=0.66,
                    center=[10.0, 0.0],
                    Q=[[1.0, 0.0], [0.0, 1.0]],
                    radius2=0.01,
                    name="future_target",
                )
            ],
            temporal_mask="tail",
        )
    )

    sample, trace = diffusion.ddim_sample_loop_replanning_guided(
        model,
        fixed.shape,
        noise=torch.randn_like(fixed),
        clip_denoised=False,
        device=torch.device("cpu"),
        eta=0.0,
        fixed_xstart=fixed,
        fixed_mask=fixed_mask,
        prefix_noise=torch.randn_like(fixed),
        guidance_fn=guidance,
        guidance_scale=0.25,
        guidance_grad_clip=1.0,
    )

    active = [item for item in trace if item["guidance_active"]]
    assert active
    assert torch.allclose(sample[:, :2, :], fixed[:, :2, :], atol=1e-6)
    assert max(item["fixed_grad_abs_max"] for item in active) == 0.0


def test_replanning_report_script_assets_are_present():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inference_guided_replanning_toy.py"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    for flag in [
        "--replan_indices",
        "--guidance_scale",
        "--num_ddim_steps",
        "--guide_dims",
        "prefix_error",
        "region_violation",
    ]:
        assert flag in text


def test_diffusion_only_replanning_report_script_assets_are_present():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inference_guided_replanning_diffusion_only.py"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    for term in [
        "encode_action_condition",
        "sample_action_from_condition",
        "--models",
        "base_vitra3b",
        "joint_kd_student",
        "diffusion_only",
        "encode_time",
    ]:
        assert term in text
