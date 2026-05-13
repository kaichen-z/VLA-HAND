import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trainable_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def make_dummy_encoder_student(action_model_type: str = "DiT-T"):
    from vitra.models.vla.vitra_encoder_student import VITRA_EncoderStudent

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
    action_model = {"model_type": action_model_type, "token_size": 32, "action_dim": 138}
    return VITRA_EncoderStudent(
        configs=config,
        train_setup_configs={"freeze_option": "encoder_student_joint"},
        act_model_configs=action_model,
        fwd_pred_next_n=2,
        repeated_diffusion_steps=1,
        use_state="DiT",
        use_fov=True,
    )


def test_encoder_student_joint_mode_outputs_vitra_sized_cognition_and_trainable_mask():
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
    assert any(name.startswith("act_model.") for name in names)


def test_dit_b_6l_student_action_head_is_registered():
    from vitra.models.action_model.diffusion_policy import DiT_models

    assert "DiT-B-6L" in DiT_models
    model = make_dummy_encoder_student(action_model_type="DiT-B-6L")
    assert model.act_model.net.x_embedder is not None


def test_joint_loss_combines_normalized_feature_and_action_terms():
    train_finetune_distill = load_script("train_finetune_distill")
    normalizer = train_finetune_distill.RunningLossNormalizer(decay=0.5, eps=1e-8)

    total, metrics = train_finetune_distill.combine_joint_losses(
        feature_loss=torch.tensor(2.0),
        action_gt_loss=torch.tensor(4.0),
        action_kd_loss=torch.tensor(8.0),
        normalizer=normalizer,
        feature_loss_weight=1.0,
        action_loss_weight=1.0,
        action_kd_loss_weight=1.0,
    )

    assert torch.allclose(total, torch.tensor(3.0))
    assert torch.allclose(metrics["feature_loss_norm"], torch.tensor(1.0))
    assert torch.allclose(metrics["action_gt_loss_norm"], torch.tensor(1.0))
    assert torch.allclose(metrics["action_kd_loss_norm"], torch.tensor(1.0))
    assert torch.allclose(metrics["total_loss"], total)


def test_student_init_layer_map_copies_expected_dit_blocks():
    create_student = load_script("create_finetune_distill_student")
    teacher_state = {
        "act_model.net.blocks.0.weight": torch.tensor([1.0]),
        "act_model.net.blocks.2.weight": torch.tensor([2.0]),
        "act_model.net.blocks.4.weight": torch.tensor([4.0]),
        "act_model.net.final_layer.weight": torch.tensor([9.0]),
    }
    student_state = {
        "act_model.net.blocks.0.weight": torch.tensor([0.0]),
        "act_model.net.blocks.1.weight": torch.tensor([0.0]),
        "act_model.net.blocks.2.weight": torch.tensor([0.0]),
        "act_model.net.final_layer.weight": torch.tensor([0.0]),
    }

    copied, report = create_student.copy_mapped_action_weights(
        teacher_state,
        student_state,
        layer_map=[0, 2, 4],
    )

    assert torch.equal(copied["act_model.net.blocks.0.weight"], torch.tensor([1.0]))
    assert torch.equal(copied["act_model.net.blocks.1.weight"], torch.tensor([2.0]))
    assert torch.equal(copied["act_model.net.blocks.2.weight"], torch.tensor([4.0]))
    assert torch.equal(copied["act_model.net.final_layer.weight"], torch.tensor([9.0]))
    assert report["copied_blocks"] == [
        {"student_block": 0, "teacher_block": 0, "num_tensors": 1},
        {"student_block": 1, "teacher_block": 2, "num_tensors": 1},
        {"student_block": 2, "teacher_block": 4, "num_tensors": 1},
    ]


def test_finetune_distill_configs_use_base_vitra_teacher_not_step140000():
    config_path = REPO_ROOT / "vitra/configs/finetune_distill_all_cam0_keypoints_mano.json"
    smoke_path = REPO_ROOT / "vitra/configs/finetune_distill_all_cam0_keypoints_mano_smoke.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))

    assert config["teacher_checkpoint"] == "./checkpoints/vitra-vla-3b.pt"
    assert "140000" not in json.dumps(config)
    assert config["vla_name"] == "VITRA_EncoderStudent"
    assert config["action_model"]["model_type"] == "DiT-B-6L"
    assert config["train_dataset"]["data_mix"] == "gigahands_real_train"
    assert config["eval_dataset"]["data_mix"] == "gigahands_real_test"
    assert config["feature_loss_weight"] == 1.0
    assert config["action_loss_weight"] == 1.0
    assert smoke["parent"] == "vitra/configs/finetune_distill_all_cam0_keypoints_mano.json"
    assert smoke["trainer"]["max_steps"] == 20


def test_full_tmux_launch_uses_gpu_0_2_and_infrequent_checkpoints():
    script = (REPO_ROOT / "scripts/run_finetune_distill_all_cam0_full_tmux.sh").read_text(encoding="utf-8")

    assert 'GPUS="${GPUS:-0,2}"' in script
    assert 'NPROC="${NPROC:-2}"' in script
    assert 'SAVE_STEPS="${SAVE_STEPS:-20000}"' in script
    assert "NCCL_P2P_DISABLE=1" in script
    assert "NCCL_IB_DISABLE=1" in script
    assert "tmux new-session" in script


def test_step140000_joint_kd_configs_use_gigahands_teacher_and_action_kd():
    config_path = REPO_ROOT / "vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano.json"
    smoke_path = REPO_ROOT / "vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano_smoke.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))

    assert "step=140000.ckpt" in config["teacher_checkpoint"]
    assert config["teacher_config"] == "vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json"
    assert config["vla_name"] == "VITRA_EncoderStudent"
    assert config["action_model"]["model_type"] == "DiT-B-6L"
    assert config["train_dataset"]["data_root_dir"] == "datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned"
    assert config["action_kd_loss_weight"] == 1.0
    assert config["save_steps"] == 50000
    assert config["max_saved_checkpoints"] == 1
    assert smoke["parent"] == "vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano.json"
    assert smoke["trainer"]["max_steps"] == 20


def test_step140000_joint_kd_full_tmux_uses_gpu7_and_final_checkpoint_only():
    script = (REPO_ROOT / "scripts/run_finetune_distill_step140000_joint_kd_full_tmux.sh").read_text(encoding="utf-8")

    assert 'GPU="${GPU:-7}"' in script
    assert 'NPROC="${NPROC:-1}"' in script
    assert 'SAVE_STEPS="${SAVE_STEPS:-50000}"' in script
    assert "finetune_distill_step140000_joint_kd_all_cam0" in script
    assert "tmux new-session" in script


def test_action_kd_loss_uses_same_noisy_action_and_teacher_student_cognition():
    train_finetune_distill = load_script("train_finetune_distill")

    source = (REPO_ROOT / "scripts/train_finetune_distill.py").read_text(encoding="utf-8")
    assert "compute_action_kd_loss(" in source
    assert "teacher_cognition_repeated" in source
    assert "student_cognition_repeated" in source
    assert "eps_teacher = teacher.act_model.net" in source
    assert "eps_student = student_module.act_model.net" in source
    assert "noise = torch.randn_like(actions_repeated)" in source
    assert "timestep = torch.randint" in source

    student_eps = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    teacher_eps = torch.tensor([[[0.0, 1.0], [3.0, 7.0]]])
    mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    loss = train_finetune_distill.masked_mse(student_eps, teacher_eps, mask)

    assert torch.allclose(loss, torch.tensor((1.0 + 4.0 + 4.0) / 3.0))


def test_distill_trainer_skips_ddp_init_sync_for_identical_rank_checkpoints():
    script = (REPO_ROOT / "scripts/train_finetune_distill.py").read_text(encoding="utf-8")

    assert "torch.cuda.set_device(local_rank)" in script
    assert "dist.barrier(device_ids=[local_rank])" not in script
    assert "broadcast_buffers=False" in script
    assert "init_sync=bool(config.get(\"training\", {}).get(\"ddp_init_sync\", False))" in script


def test_eval_tool_exposes_triad_summary_for_distilled_student():
    path = REPO_ROOT / "tools/evaluate_gigahands_stage1.py"
    spec = importlib.util.spec_from_file_location("evaluate_gigahands_stage1_for_distill_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    summary = module.triad_summary(
        teacher_checkpoint="base_vitra.pt",
        base_checkpoint="base_vitra.pt",
        distilled_checkpoint="student.ckpt",
        teacher_action_metrics={"action_mse": 0.2},
        base_action_metrics={"action_mse": 0.2},
        distilled_action_metrics={"action_mse": 0.1},
        base_feature_alignment={"vlm_cognition_mse": 0.0, "vlm_cognition_cosine": 1.0},
        distilled_feature_alignment={"vlm_cognition_mse": 0.5, "vlm_cognition_cosine": 0.7},
        data_ids=[1, 2],
    )

    assert summary["teacher"]["checkpoint"] == "base_vitra.pt"
    assert summary["distilled"]["checkpoint"] == "student.ckpt"
    assert "distilled_vs_base3b_action" in summary["deltas"]
    assert summary["data_ids"] == [1, 2]


def test_paligemma_image_token_error_path_uses_model_config():
    text = (REPO_ROOT / "vitra/models/vla/vitra_paligemma.py").read_text(encoding="utf-8")

    assert "self.config.image_token_index" not in text
    assert "self.model.config.image_token_index" in text


def test_hand_collator_preserves_raw_inputs_for_teacher_reprocessing():
    from vitra.utils.data_utils import PaddedCollatorForHandPrediction

    collator = PaddedCollatorForHandPrediction(model_max_length=8, pad_token_id=0)

    def instance(idx: int):
        return {
            "pixel_values": torch.ones(1, 3, 4, 4) * idx,
            "input_ids": torch.tensor([1, 2 + idx], dtype=torch.long),
            "labels": None,
            "actions": torch.zeros(2, 138),
            "action_masks": torch.ones(2, 138),
            "current_state_mask": torch.ones(212),
            "current_state": torch.zeros(212),
            "fov": torch.zeros(2),
            "intrinsics": torch.eye(3),
            "raw_image": f"image-{idx}",
            "instruction": f"instruction-{idx}",
        }

    batch = collator([instance(0), instance(1)])

    assert batch["raw_images"] == ["image-0", "image-1"]
    assert batch["instructions"] == ["instruction-0", "instruction-1"]
