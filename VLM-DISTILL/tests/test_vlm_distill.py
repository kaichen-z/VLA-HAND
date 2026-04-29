import json
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

from scripts.train_vlm_distill import RunningLossNormalizer, cognition_distill_metrics, save_student_checkpoint, vitkd_distill_losses
from vitra.models.vla.vitra_encoder_student import VITRA_EncoderStudent
from vitra.models.vla.vitra_paligemma import VITRA_Paligemma
from vitra.utils.config_utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_eval_module():
    path = REPO_ROOT / "tools" / "evaluate_gigahands_stage1.py"
    spec = importlib.util.spec_from_file_location("evaluate_gigahands_stage1_for_vlm_distill_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"use_cache": True})()
        self.vision_tower = nn.Linear(2, 2)
        self.language_model = nn.Linear(2, 2)
        self.input_embeddings = nn.Embedding(8, 2)

    def get_input_embeddings(self):
        return self.input_embeddings


def make_dummy_vitra():
    model = VITRA_Paligemma.__new__(VITRA_Paligemma)
    nn.Module.__init__(model)
    model.backbone = DummyBackbone()
    model.act_model = nn.Linear(2, 2)
    model.fov_encoder = nn.Linear(2, 2)
    model.vlm_state_encoder = nn.Linear(2, 2)
    model.cognition_token = nn.Parameter(torch.zeros(2))
    model.use_state = "VLM"
    model.use_fov = True
    return model


def trainable_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def make_dummy_encoder_student():
    config = {
        "train_dataset": {"action_type": "keypoints"},
        "state_encoder": {"state_dim": 212},
        "fwd_pred_next_n": 2,
        "loss_type": "human",
        "student_vlm": {
            "use_dummy": True,
            "dummy_hidden_size": 16,
            "student_output_size": 32,
            "fusion_hidden_size": 24,
            "freeze_vision_encoder": True,
            "include_state_in_fusion": True,
        },
    }
    action_model = {"model_type": "DiT-T", "token_size": 32, "action_dim": 138}
    return VITRA_EncoderStudent(
        configs=config,
        train_setup_configs={"freeze_option": "encoder_student_cognition"},
        act_model_configs=action_model,
        fwd_pred_next_n=2,
        repeated_diffusion_steps=1,
        use_state="DiT",
        use_fov=True,
    )


def test_vlm_distill_default_trainable_mask_freezes_vision_and_action():
    model = make_dummy_vitra()

    model.trainable_params_setup_for_vlm_distill("freeze_vision_encoder")

    names = trainable_names(model)
    assert "backbone.vision_tower.weight" not in names
    assert "backbone.vision_tower.bias" not in names
    assert "act_model.weight" not in names
    assert "act_model.bias" not in names
    assert "backbone.language_model.weight" in names
    assert "backbone.input_embeddings.weight" in names
    assert "cognition_token" in names
    assert "fov_encoder.weight" in names
    assert "vlm_state_encoder.weight" in names
    assert model.backbone.config.use_cache is False


def test_vlm_distill_action_head_only_strict_trains_only_action_head():
    model = make_dummy_vitra()

    model.trainable_params_setup_for_vlm_distill("action_head_only_strict")

    assert trainable_names(model) == {"act_model.weight", "act_model.bias"}
    assert model.backbone.config.use_cache is False


def test_vlm_distill_action_head_plus_adapters_freezes_backbone():
    model = make_dummy_vitra()

    model.trainable_params_setup_for_vlm_distill("action_head_plus_adapters")

    names = trainable_names(model)
    assert "act_model.weight" in names
    assert "act_model.bias" in names
    assert "cognition_token" in names
    assert "fov_encoder.weight" in names
    assert "vlm_state_encoder.weight" in names
    assert "backbone.language_model.weight" not in names
    assert "backbone.input_embeddings.weight" not in names
    assert "backbone.vision_tower.weight" not in names


def test_vlm_distill_action_head_plus_vlm_small_lr_keeps_vision_frozen():
    model = make_dummy_vitra()

    model.trainable_params_setup_for_vlm_distill("action_head_plus_vlm_small_lr")

    names = trainable_names(model)
    assert "act_model.weight" in names
    assert "backbone.language_model.weight" in names
    assert "backbone.input_embeddings.weight" in names
    assert "cognition_token" in names
    assert "fov_encoder.weight" in names
    assert "vlm_state_encoder.weight" in names
    assert "backbone.vision_tower.weight" not in names


def test_vlm_distill_checkpoint_round_trip_temp_dir(tmp_path):
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(1.25)
        model.bias.fill_(-0.5)

    checkpoint_dir = save_student_checkpoint(model, tmp_path, global_step=1, epoch=0)
    loaded = torch.load(checkpoint_dir / "weights.pt", map_location="cpu")

    restored = nn.Linear(3, 2)
    restored.load_state_dict(loaded)
    x = torch.ones(1, 3)
    assert torch.allclose(model(x), restored(x))

    meta = json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta == {"epoch": 0, "global_step": 1}


def test_vlm_distill_cognition_metrics_are_finite():
    student = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    metrics = cognition_distill_metrics(student, teacher)

    assert set(metrics) == {"vlm_cognition_mse", "vlm_cognition_cosine"}
    assert torch.isfinite(metrics["vlm_cognition_mse"])
    assert torch.isfinite(metrics["vlm_cognition_cosine"])
    assert torch.allclose(metrics["vlm_cognition_cosine"], torch.tensor(1.0))


def test_running_loss_normalizer_initializes_and_clamps():
    normalizer = RunningLossNormalizer(decay=0.5, eps=1e-8)

    first_norm, first_ema = normalizer.normalize("cognition", torch.tensor(2.0))
    second_norm, second_ema = normalizer.normalize("cognition", torch.tensor(4.0))
    tiny_norm, tiny_ema = normalizer.normalize("action", torch.tensor(0.0))

    assert torch.allclose(first_ema, torch.tensor(2.0))
    assert torch.allclose(first_norm, torch.tensor(1.0))
    assert torch.allclose(second_ema, torch.tensor(3.0))
    assert torch.allclose(second_norm, torch.tensor(4.0 / 3.0))
    assert torch.allclose(tiny_ema, torch.tensor(0.0))
    assert torch.isfinite(tiny_norm)


def test_encoder_student_cognition_shape_and_frozen_vision():
    model = make_dummy_encoder_student()
    batch = 2
    feature = model(
        torch.zeros(batch, 3, 224, 224),
        torch.ones(batch, 5, dtype=torch.long),
        attention_mask=torch.ones(batch, 5, dtype=torch.bool),
        current_state=torch.zeros(batch, 212),
        current_state_mask=torch.ones(batch, 212, dtype=torch.bool),
        fov=torch.zeros(batch, 2),
        mode="vlm_cognition",
    )

    names = trainable_names(model)
    assert feature.shape == (batch, 32)
    assert "vision_encoder.proj.weight" not in names
    assert "text_encoder.proj.weight" in names
    assert "fusion.0.weight" in names
    assert "cognition_projection.weight" in names


def test_vitkd_distill_losses_use_masked_features():
    student_features = {
        "cognition": torch.ones(2, 4),
        "shallow_features": [torch.ones(2, 3, 4), torch.ones(2, 3, 4) * 2],
        "deep_feature": torch.ones(2, 3, 4),
        "deep_generated": torch.ones(2, 3, 4) * 3,
        "deep_generation_mask": torch.tensor([[True, False, False], [False, True, False]]),
        "token_mask": torch.tensor([[True, True, False], [True, True, True]]),
    }
    teacher_features = {
        "cognition": torch.zeros(2, 4),
        "shallow_features": [torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)],
        "deep_feature": torch.zeros(2, 3, 4),
        "token_mask": torch.tensor([[True, True, False], [True, True, True]]),
    }

    losses, metrics = vitkd_distill_losses(student_features, teacher_features, {})

    assert set(losses) == {"cognition", "shallow_mimic", "deep_generation"}
    assert torch.allclose(losses["cognition"], torch.tensor(1.0))
    assert torch.allclose(losses["shallow_mimic"], torch.tensor(2.5))
    assert torch.allclose(losses["deep_generation"], torch.tensor(9.0))
    assert "vitkd_deep_mask_ratio" in metrics


def test_vlm_distill_config_has_comparison_checkpoints():
    config = load_config("vitra/configs/vlm_distill_gigahands_cognition.json")

    assert config["student_init_checkpoint"] == "./checkpoints/vitra-vla-3b.pt"
    assert config["action_eval"]["baseline_checkpoint"] == "./checkpoints/vitra-vla-3b.pt"
    assert config["action_eval"]["num_eval_clips"] == 20
    assert "epoch=0-step=28000.ckpt" in config["teacher_checkpoint"]
    assert Path(config["teacher_checkpoint"]).name == "weights.pt"


def test_vlm_distill_stage2_configs_encode_ablation_order():
    expected = {
        "vlm_distill_stage2_action_head_only_strict.json": ("action_head_only_strict", "action_only"),
        "vlm_distill_stage2_action_head_plus_adapters.json": ("action_head_plus_adapters", "action_only"),
        "vlm_distill_stage2_action_head_plus_adapters_normalized.json": ("action_head_plus_adapters", "normalized"),
        "vlm_distill_stage2_action_head_plus_vlm_small_lr_normalized.json": (
            "action_head_plus_vlm_small_lr",
            "normalized",
        ),
    }

    for filename, (freeze_option, loss_mode) in expected.items():
        config = load_config(f"vitra/configs/{filename}")
        assert config["distill_train_setup"]["freeze_option"] == freeze_option
        assert config["distill_loss_mode"] == loss_mode
        assert config["action_eval"]["num_eval_clips"] == 20


def test_small_vitra_vitkd_configs_encode_ablation_order():
    expected = {
        "vlm_distill_small_vitra_cognition_only_gigahands.json": (True, False, False, False),
        "vlm_distill_small_vitra_shallow_only_gigahands.json": (True, True, False, False),
        "vlm_distill_small_vitra_deep_gen_only_gigahands.json": (True, False, True, False),
        "vlm_distill_small_vitra_vitkd_full_gigahands.json": (True, True, True, False),
        "vlm_distill_small_vitra_bad_direct_deep_mimic_gigahands.json": (True, False, False, True),
    }
    for filename, flags in expected.items():
        config = load_config(f"vitra/configs/{filename}")
        vitkd = config["vitkd"]
        assert config["vla_name"] == "VITRA_SmallPaliGemmaStudent"
        assert config["distill_loss_mode"] == "vitkd"
        assert config["action_eval"]["num_eval_clips"] == 20
        assert (
            vitkd["use_cognition_loss"],
            vitkd["use_shallow_mimic_loss"],
            vitkd["use_deep_generation_loss"],
            vitkd["use_direct_deep_mimic_loss"],
        ) == flags
        assert config["save_steps"] == 500


def test_encoder_student_config_uses_separate_teacher_and_student():
    config = load_config("vitra/configs/vlm_distill_encoder_student_gigahands.json")

    assert config["vla_name"] == "VITRA_EncoderStudent"
    assert config["teacher_config"].endswith("human_pretrain_gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked.json")
    assert config["student_vlm"]["freeze_vision_encoder"] is True
    assert config["student_vlm"]["text_encoder_name"] == "distilbert-base-uncased"
    assert config["student_vlm"]["student_output_size"] == 2304
    assert config["action_model"]["model_type"] == "DiT-S"


def test_triad_summary_includes_required_action_feature_and_probe_keys():
    module = load_eval_module()

    summary = module.triad_summary(
        teacher_checkpoint="teacher.pt",
        base_checkpoint="base.pt",
        distilled_checkpoint="distilled.pt",
        teacher_action_metrics={"action_mse": 0.1},
        base_action_metrics={"action_mse": 0.4},
        distilled_action_metrics={"action_mse": 0.2},
        base_feature_alignment={"vlm_cognition_mse": 3.0, "vlm_cognition_cosine": 0.2},
        distilled_feature_alignment={"vlm_cognition_mse": 1.0, "vlm_cognition_cosine": 0.8},
        data_ids=[7, 9],
        base_vs_teacher_action={"action_mse": 0.3},
        distilled_vs_teacher_action={"action_mse": 0.15},
    )

    assert summary["data_ids"] == [7, 9]
    assert "action_metrics" in summary["teacher"]
    assert "action_metrics" in summary["base3b"]
    assert "action_metrics" in summary["distilled"]
    assert "feature_alignment_to_teacher" in summary["base3b"]
    assert "feature_alignment_to_teacher" in summary["distilled"]
    assert summary["base3b"]["action_metrics_vs_teacher_action"]["action_mse"] == 0.3
    assert summary["distilled"]["action_metrics_vs_teacher_action"]["action_mse"] == 0.15
    assert summary["base_3b_student_probe"]["checkpoint"] == "base.pt"
    assert summary["deltas"]["distilled_vs_base3b_action"]["delta_after_minus_before"]["action_mse"] == -0.2


def test_triad_evaluation_reuses_selected_data_ids(monkeypatch, tmp_path):
    module = load_eval_module()
    calls = {"eval_indices": [], "feature_indices": []}

    monkeypatch.setattr(module, "load_config", lambda _: {"train_dataset": {}})
    monkeypatch.setattr(module, "build_eval_dataset", lambda args, config: object())
    monkeypatch.setattr(module, "select_eval_indices", lambda dataset, args: [3, 5])

    def fake_evaluate_checkpoint(args, config, dataset, checkpoint, eval_indices=None):
        calls["eval_indices"].append((checkpoint, list(eval_indices)))
        return {"metrics": {"action_mse": float(len(calls["eval_indices"]))}, "samples": [], "targets": [], "masks": []}

    def fake_feature_alignment(args, teacher_config, candidate_config, dataset, eval_indices, candidate_checkpoint, teacher_checkpoint):
        calls["feature_indices"].append((candidate_checkpoint, teacher_checkpoint, list(eval_indices)))
        return {"vlm_cognition_mse": 1.0, "vlm_cognition_cosine": 0.5}

    monkeypatch.setattr(module, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(module, "feature_alignment_to_teacher", fake_feature_alignment)

    args = Namespace(
        mano_motion_videos=False,
        rgb_overlay_videos=False,
        config="child.json",
        dataset_root=tmp_path,
        data_mix="gigahands_real_test",
        output_dir=tmp_path,
        teacher_checkpoint="teacher.pt",
        base_checkpoint="base.pt",
        checkpoint="distilled.pt",
        no_videos=True,
        hand_motion_videos=False,
        teacher_label="teacher",
        base_label="base3b",
        label="distilled",
    )

    summary = module.run_model_evaluation(args)

    assert summary["data_ids"] == [3, 5]
    assert calls["eval_indices"] == [
        ("teacher.pt", [3, 5]),
        ("base.pt", [3, 5]),
        ("distilled.pt", [3, 5]),
    ]
    assert calls["feature_indices"] == [
        ("base.pt", "teacher.pt", [3, 5]),
        ("distilled.pt", "teacher.pt", [3, 5]),
    ]
