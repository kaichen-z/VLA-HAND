"""
VLM cognition-token distillation for VITRA.

By default this trains only the student VLM-side representation to imitate a
frozen teacher cognition token. A separate Stage 2 config can add a supervised
action loss against ground-truth actions while keeping the teacher objective
unchanged.
"""

import argparse
import copy
import faulthandler
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.train import get_fsdp_wrap_policy_and_checkpointing
from vitra.datasets.materialize import get_vla_dataset_and_collator
from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.training import VLAMetrics
from vitra.training.fsdp import VLAFSDPStrategy, distributed_barrier, move_to_device
from vitra.utils import set_global_seed, setup_seed
from vitra.utils.config_utils import load_config
from vitra.utils.overwatch import initialize_overwatch


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

overwatch = initialize_overwatch(__name__)


def distributed_barrier_for_device(device_id: int) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def posix_to_str(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: posix_to_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [posix_to_str(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_vitra_weights(model, checkpoint: str | None, description: str):
    if checkpoint is None:
        return model
    checkpoint_path = Path(checkpoint)
    weights_path = checkpoint_path / "weights.pt" if checkpoint_path.is_dir() else checkpoint_path
    if not weights_path.exists():
        raise FileNotFoundError(f"{description} checkpoint not found: {weights_path}")
    return load_vla_checkpoint(model, str(weights_path))


def prepare_teacher_inputs(teacher, batch: dict[str, Any], device_id: int) -> dict[str, torch.Tensor]:
    if "raw_images" not in batch or "instructions" not in batch:
        return {
            "pixel_values": batch["pixel_values"],
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "current_state_mask": batch["current_state_mask"],
            "current_state": batch["current_state"],
            "fov": batch["fov"],
        }

    instructions = []
    for instruction in batch["instructions"]:
        text = str(instruction)
        instructions.append(text if text.startswith("<image>") else "<image>" + text)
    model_inputs = teacher.processor(
        text=instructions,
        images=batch["raw_images"],
        return_tensors="pt",
        padding=True,
    )
    model_inputs = model_inputs.to(device=torch.device("cuda", device_id))
    return {
        "pixel_values": model_inputs["pixel_values"],
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs.get(
            "attention_mask",
            torch.ones_like(model_inputs["input_ids"], dtype=torch.bool),
        ).to(torch.bool),
        "current_state_mask": batch["current_state_mask"],
        "current_state": batch["current_state"],
        "fov": batch["fov"],
    }


def freeze_teacher(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def count_trainable_groups(model) -> dict[str, int]:
    groups = {
        "vision": 0,
        "language_or_backbone": 0,
        "action_head": 0,
        "cognition_token": 0,
        "fov_encoder": 0,
        "state_encoder": 0,
        "student_vision_encoder": 0,
        "student_text_encoder": 0,
        "student_fusion_projection": 0,
        "other": 0,
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if "vision_tower" in name:
            groups["vision"] += n
        elif "act_model" in name:
            groups["action_head"] += n
        elif "cognition_token" in name:
            groups["cognition_token"] += n
        elif "fov_encoder" in name:
            groups["fov_encoder"] += n
        elif "vlm_state_encoder" in name:
            groups["state_encoder"] += n
        elif "vision_encoder" in name:
            groups["student_vision_encoder"] += n
        elif "text_encoder" in name:
            groups["student_text_encoder"] += n
        elif "fusion" in name or "cognition_projection" in name or "student_state_encoder" in name:
            groups["student_fusion_projection"] += n
        elif "vitkd_" in name:
            groups["student_fusion_projection"] += n
        elif "backbone" in name:
            groups["language_or_backbone"] += n
        else:
            groups["other"] += n
    groups["total"] = sum(groups.values())
    return groups


def cognition_distill_metrics(student_cognition: torch.Tensor, teacher_cognition: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher_cognition = teacher_cognition.detach()
    student_float = student_cognition.float()
    teacher_float = teacher_cognition.float()
    return {
        "vlm_cognition_mse": F.mse_loss(student_float, teacher_float),
        "vlm_cognition_cosine": F.cosine_similarity(student_float, teacher_float, dim=-1).mean(),
    }


def masked_feature_mse(student_feature: torch.Tensor, teacher_feature: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = token_mask.bool()
    per_token = (student_feature.float() - teacher_feature.detach().float()).pow(2).mean(dim=-1)
    denom = mask.float().sum().clamp_min(1.0)
    return (per_token * mask.float()).sum() / denom


def vitkd_distill_losses(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    vitkd_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    losses = {}
    metrics = {}
    token_mask = student_features["token_mask"].bool() & teacher_features["token_mask"].bool()

    if vitkd_config.get("use_cognition_loss", True):
        cognition_metrics = cognition_distill_metrics(student_features["cognition"], teacher_features["cognition"])
        losses["cognition"] = cognition_metrics["vlm_cognition_mse"]
        metrics.update({k: v.detach() for k, v in cognition_metrics.items()})
        metrics["cognition_loss"] = losses["cognition"].detach()

    if vitkd_config.get("use_shallow_mimic_loss", True):
        shallow_losses = []
        for idx, (student_layer, teacher_layer) in enumerate(
            zip(student_features["shallow_features"], teacher_features["shallow_features"])
        ):
            layer_loss = masked_feature_mse(student_layer, teacher_layer, token_mask)
            shallow_losses.append(layer_loss)
            metrics[f"vitkd_shallow_layer_{idx}_loss"] = layer_loss.detach()
        losses["shallow_mimic"] = torch.stack(shallow_losses).mean()
        metrics["vitkd_shallow_mimic_loss"] = losses["shallow_mimic"].detach()

    if vitkd_config.get("use_deep_generation_loss", True):
        deep_mask = student_features["deep_generation_mask"].bool() & token_mask
        losses["deep_generation"] = masked_feature_mse(
            student_features["deep_generated"],
            teacher_features["deep_feature"],
            deep_mask,
        )
        metrics["vitkd_deep_generation_loss"] = losses["deep_generation"].detach()
        metrics["vitkd_deep_mask_ratio"] = deep_mask.float().mean().detach()

    if vitkd_config.get("use_direct_deep_mimic_loss", False):
        losses["direct_deep_mimic"] = masked_feature_mse(
            student_features["deep_feature"],
            teacher_features["deep_feature"],
            token_mask,
        )
        metrics["vitkd_direct_deep_mimic_loss"] = losses["direct_deep_mimic"].detach()

    if not losses:
        raise RuntimeError("distill_loss_mode='vitkd' requires at least one enabled ViTKD loss.")
    return losses, metrics


class RunningLossNormalizer:
    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = float(decay)
        self.eps = float(eps)
        self.ema: dict[str, torch.Tensor] = {}

    def update(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        value = loss.detach().float()
        if name not in self.ema:
            self.ema[name] = value
        else:
            self.ema[name] = self.decay * self.ema[name] + (1.0 - self.decay) * value
        return self.ema[name]

    def normalize(self, name: str, loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ema = self.update(name, loss).to(device=loss.device)
        denom = torch.clamp(ema, min=self.eps)
        return loss / denom, ema


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step=(\d+)\.ckpt$", path.name)
    return int(match.group(1)) if match else -1


def prune_old_checkpoints(run_dir: Path, keep_last: int | None) -> None:
    if keep_last is None or keep_last <= 0:
        return
    checkpoints_root = run_dir / "checkpoints"
    checkpoint_dirs = sorted(checkpoints_root.glob("epoch=*-step=*.ckpt"), key=checkpoint_step)
    stale_dirs = checkpoint_dirs[:-keep_last]
    for stale_dir in stale_dirs:
        shutil.rmtree(stale_dir)
        overwatch.info(f"Pruned old distilled student checkpoint {stale_dir}")


def resolve_loss_weights(variant: dict[str, Any]) -> tuple[float, float]:
    cognition_weight = variant.get("cognition_loss_weight", variant.get("distill_loss_weight", 1.0))
    action_weight = variant.get("action_loss_weight", 0.0)
    return float(cognition_weight), float(action_weight)


def resolve_loss_mode(variant: dict[str, Any]) -> str:
    mode = variant.get("distill_loss_mode", "weighted")
    valid_modes = {"weighted", "action_only", "normalized", "vitkd"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported distill_loss_mode={mode!r}; supported modes: {sorted(valid_modes)}")
    return mode


def resolve_action_distill_target(variant: dict[str, Any]) -> str:
    target = variant.get("action_distill_target", "gt")
    valid_targets = {"gt", "teacher"}
    if target not in valid_targets:
        raise ValueError(
            f"Unsupported action_distill_target={target!r}; supported targets: {sorted(valid_targets)}"
        )
    return target


def extract_teacher_cognition_and_actions(
    teacher,
    teacher_inputs: dict[str, torch.Tensor],
    action_masks: torch.Tensor,
    variant: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    action_eval_config = variant.get("action_eval", {})
    cfg_scale = variant.get("teacher_action_cfg_scale", action_eval_config.get("cfg_scale", 5.0))
    use_ddim = variant.get("teacher_action_use_ddim", True)
    num_ddim_steps = variant.get("teacher_action_num_ddim_steps", action_eval_config.get("num_ddim_steps", 10))

    output_hs, inputs_masks = teacher.prepare_vlm_features(
        pixel_values=teacher_inputs["pixel_values"],
        input_ids=teacher_inputs["input_ids"],
        attention_mask=teacher_inputs.get("attention_mask"),
        current_state_mask=teacher_inputs.get("current_state_mask"),
        current_state=teacher_inputs.get("current_state"),
        fov=teacher_inputs.get("fov"),
        use_cache=False,
    )
    teacher_cognition = teacher.extract_cognition_token(output_hs, inputs_masks).squeeze(1)
    teacher_actions, _ = teacher._forward_act_model(
        vlm_features=output_hs,
        attention_mask=inputs_masks,
        action_masks=action_masks,
        current_state=teacher_inputs.get("current_state"),
        current_state_mask=teacher_inputs.get("current_state_mask"),
        mode="eval",
        repeated_diffusion_steps=1,
        cfg_scale=cfg_scale,
        use_ddim=use_ddim,
        num_ddim_steps=num_ddim_steps,
    )
    teacher_actions = teacher_actions * action_masks.to(dtype=teacher_actions.dtype, device=teacher_actions.device)
    return teacher_cognition.detach(), teacher_actions.detach()


def save_student_checkpoint(
    student,
    run_dir: Path,
    global_step: int,
    epoch: int,
    max_saved_checkpoints: int | None = None,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"epoch={epoch}-step={global_step}.ckpt"
    if overwatch.is_rank_zero():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(student, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, save_policy, optim_policy):
            state_dict = student.state_dict()
    else:
        state_dict = student.state_dict()

    distributed_barrier()
    if overwatch.is_rank_zero():
        meta_state = {"epoch": epoch, "global_step": global_step}
        with open(checkpoint_dir / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta_state, handle)
        torch.save(state_dict, checkpoint_dir / "weights.pt")
        overwatch.info(f"Saved distilled student checkpoint to {checkpoint_dir}")
        prune_old_checkpoints(run_dir, max_saved_checkpoints)
    distributed_barrier()
    return checkpoint_dir


def maybe_run_action_eval(config: dict[str, Any], checkpoint_dir: Path) -> None:
    eval_config = config.get("action_eval", {})
    if not eval_config.get("enabled", False):
        return
    if not overwatch.is_rank_zero():
        return

    output_root = Path(eval_config.get("output_dir", checkpoint_dir / "action_eval")) / checkpoint_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "tools/evaluate_gigahands_stage1.py",
        "--config",
        str(eval_config.get("config", config.get("action_eval_config", config.get("config", "")))),
        "--dataset_root",
        str(eval_config["dataset_root"]),
        "--data_mix",
        str(eval_config.get("data_mix", "gigahands_real_test")),
        "--checkpoint",
        str(checkpoint_dir),
        "--label",
        str(eval_config.get("label", "distilled")),
        "--num_eval_clips",
        str(eval_config.get("num_eval_clips", 5)),
        "--output_dir",
        str(output_root),
        "--num_ddim_steps",
        str(eval_config.get("num_ddim_steps", 10)),
        "--cfg_scale",
        str(eval_config.get("cfg_scale", 5.0)),
        "--seed",
        str(eval_config.get("seed", config.get("seed", 42))),
        "--eval_sample_strategy",
        str(eval_config.get("eval_sample_strategy", "sequential")),
        "--no_videos",
    ]
    if eval_config.get("teacher_reference_checkpoint") and eval_config.get("baseline_checkpoint"):
        cmd.extend(
            [
                "--teacher_checkpoint",
                str(eval_config["teacher_reference_checkpoint"]),
                "--teacher_label",
                str(eval_config.get("teacher_reference_label", "gigahands_teacher")),
                "--base_checkpoint",
                str(eval_config["baseline_checkpoint"]),
                "--base_label",
                str(eval_config.get("baseline_label", "base3b")),
            ]
        )
        if eval_config.get("teacher_config"):
            cmd.extend(["--teacher_config", str(eval_config["teacher_config"])])
        if eval_config.get("baseline_config"):
            cmd.extend(["--base_config_file", str(eval_config["baseline_config"])])
        if eval_config.get("checkpoint_config"):
            cmd.extend(["--checkpoint_config", str(eval_config["checkpoint_config"])])
    elif eval_config.get("baseline_checkpoint"):
        cmd.extend(
            [
                "--baseline_checkpoint",
                str(eval_config["baseline_checkpoint"]),
                "--baseline_label",
                str(eval_config.get("baseline_label", "base3b")),
            ]
        )

    overwatch.info(f"Running action-aware eval: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)


def update_configs(configs, args):
    if args["task_name"] is not None:
        configs["task_name"] = args["task_name"]
    if args.get("no_save_checkpoint"):
        configs["save_checkpoint"] = False
    if args.get("disable_action_eval"):
        configs.setdefault("action_eval", {})["enabled"] = False
    configs.setdefault("save_checkpoint", True)
    configs.setdefault("num_workers", None)
    configs.setdefault("prefetch_factor", None)

    configs["use_bf16"] = args["use_bf16"] if args["use_bf16"] is not None else configs.get("use_bf16", False)

    for key in ("output_root", "log_root", "cache_root"):
        configs[key] = Path(args[key]) if args.get(key) is not None else Path(configs[key])
    configs["cache_root"] = configs["cache_root"] / configs["model"]

    for k, v in args.items():
        if k in {"config", "no_save_checkpoint", "disable_action_eval"}:
            continue
        if k == "trainer":
            for sub_k, sub_v in v.items():
                if sub_v is not None:
                    configs["trainer"][sub_k] = sub_v
        elif v is not None:
            configs[k] = v

    return configs


def build_dataloader(variant, processor, batch_size, resume_epoch=0, resume_step=0, grad_accumulation_steps=1):
    worker_init_fn = set_global_seed(variant["seed"], get_worker_init_fn=True)
    vla_dataset, collator, batch_sampler = get_vla_dataset_and_collator(
        variant["train_dataset"]["data_root_dir"],
        variant["train_dataset"]["data_mix"],
        augmentation=variant["train_dataset"]["augmentation"],
        shard_num=dist.get_world_size(),
        shard_index=dist.get_rank(),
        seed=variant["seed"],
        future_action_window_size=variant["fwd_pred_next_n"] - 1,
        processor=processor,
        batch_size=batch_size,
        normalization=variant["train_dataset"].get("normalization", True),
        flip_augmentation=variant["train_dataset"].get("flip_augmentation", 1.0),
        set_none_ratio=variant["train_dataset"].get("set_none_ratio", 0.0),
        action_type=variant["train_dataset"].get("action_type", "angle"),
        use_rel=variant["train_dataset"].get("use_rel", False),
        rel_mode=variant["train_dataset"].get("rel_mode", "step"),
        clip_len=variant["train_dataset"].get("clip_len", None),
        state_mask_prob=variant["train_dataset"].get("state_mask_prob", 0.1),
        target_image_height=variant["train_dataset"].get("target_image_height", 224),
        statistics_dataset_name=variant["train_dataset"].get("statistics_dataset_name", None),
    )

    batch_sampler.set_epoch(resume_epoch, resume_step * grad_accumulation_steps)
    setup_seed(variant["seed"], rank=dist.get_rank())

    num_workers = variant["num_workers"] if variant["num_workers"] is not None else variant["train_dataset"]["num_workers"]
    prefetch_factor = (
        variant["prefetch_factor"]
        if variant["prefetch_factor"] is not None
        else variant["train_dataset"].get("prefetch_factor", None)
    )
    if num_workers == 0 or prefetch_factor == 0:
        prefetch_factor = None

    dataloader = DataLoader(
        vla_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
        pin_memory=num_workers > 0,
    )
    return vla_dataset, dataloader


def experiment(variant):
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_mode = os.getenv("WANDB_MODE", "").lower()
    if wandb_api_key is not None:
        wandb.login(key=wandb_api_key)
    elif wandb_mode not in {"offline", "disabled"}:
        raise ValueError("Please set WANDB_API_KEY or WANDB_MODE=offline/disabled.")

    os.makedirs(variant["log_root"], exist_ok=True)
    os.makedirs(variant["output_root"], exist_ok=True)
    os.makedirs(variant["cache_root"], exist_ok=True)

    run_id = f"{variant['task_name']}_TB{variant['total_batch_size']}_B{variant['batch_size']}_bf16{variant['use_bf16']}"
    checkpoint_dir = Path(variant["output_root"]) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    copied_variant = posix_to_str(copy.deepcopy(variant))
    if overwatch.rank() == 0:
        with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(copied_variant, handle, indent=2)
        print(json.dumps(copied_variant, indent=2))

    distributed_barrier_for_device(device_id)

    teacher_variant = load_config(variant["teacher_config"]) if variant.get("teacher_config") else variant
    teacher_variant = copy.deepcopy(teacher_variant)
    teacher_variant["use_bf16"] = variant["use_bf16"]
    overwatch.info("Loading frozen distillation teacher", ctx_level=1)
    teacher = build_vla(configs=teacher_variant)
    teacher = load_vitra_weights(teacher, variant["teacher_checkpoint"], "Teacher")
    teacher = freeze_teacher(teacher).cuda()
    if hasattr(teacher, "model"):
        teacher.model.use_bf16 = variant["use_bf16"]
    teacher.use_bf16 = variant["use_bf16"]

    overwatch.info("Loading trainable distillation student", ctx_level=1)
    student = build_vla(configs=variant)
    student = load_vitra_weights(student, variant.get("student_init_checkpoint"), "Student init")
    student = student.train()
    freeze_option = variant.get("distill_train_setup", {}).get("freeze_option", "freeze_vision_encoder")
    student.trainable_params_setup_for_vlm_distill(freeze_option)
    if hasattr(student, "model"):
        student.model.use_bf16 = variant["use_bf16"]
    student.use_bf16 = variant["use_bf16"]

    trainable_groups = count_trainable_groups(student)
    if overwatch.rank() == 0:
        overwatch.info(f"Student trainable parameter groups: {trainable_groups}")

    processor = student.processor
    batch_size = variant["batch_size"]
    total_batch_size = variant["total_batch_size"]

    training_strategy = VLAFSDPStrategy(
        vla=student,
        device_id=overwatch.local_rank(),
        stage="vlm-distill",
        epochs=variant["trainer"]["max_epochs"],
        max_steps=variant["trainer"]["max_steps"],
        global_batch_size=total_batch_size,
        per_device_batch_size=batch_size,
        learning_rate=variant["trainer"]["learning_rate"],
        weight_decay=variant["trainer"]["weight_decay"],
        max_grad_norm=variant["trainer"]["gradient_clip_val"],
        lr_scheduler_type=variant["trainer"]["lr_scheduler_type"],
        warmup_ratio=variant["trainer"]["warmup_ratio"],
        enable_gradient_checkpointing=variant["trainer"]["enable_gradient_checkpointing"],
        enable_mixed_precision_training=variant["trainer"]["enable_mixed_precision_training"],
        reduce_in_full_precision=variant["trainer"]["reduce_in_full_precision"],
        action_model_learning_rate=variant["trainer"].get("action_model_learning_rate", None),
        action_model_weight_decay=variant["trainer"].get("action_model_weight_decay", None),
        sharding_strategy=variant["trainer"].get("sharding_strategy", "shard-grad-op"),
        cognition_token_weight_decay=variant["trainer"].get("cognition_token_weight_decay", True),
        llm_freeze_step=variant["trainer"].get("llm_freeze_step", 0),
        move_word_embedding_to_action_model=variant["trainer"].get("move_word_embedding_to_action_head", False),
        optimizer_betas=variant["trainer"].get("optimizer_betas", (0.9, 0.999)),
    )

    vla_dataset, dataloader = build_dataloader(
        variant,
        processor,
        batch_size,
        grad_accumulation_steps=training_strategy.grad_accumulation_steps,
    )

    auto_wrap_policy, checkpointing_policy = get_fsdp_wrap_policy_and_checkpointing(variant["trainer"])
    training_strategy.run_setup(
        run_dir=checkpoint_dir,
        n_train_examples=len(vla_dataset),
        auto_wrap_policy_modules=auto_wrap_policy,
        checkpointing_policy_modules=checkpointing_policy,
    )
    student = training_strategy.vla

    trackers = [] if os.getenv("WANDB_MODE", "").lower() == "disabled" else ["wandb"]
    metrics = VLAMetrics(
        trackers,
        hparams=copied_variant,
        run_id=run_id,
        run_dir=checkpoint_dir,
        wandb_project=variant["wandb_project"],
        wandb_entity=variant["wandb_entity"],
    )
    metrics.commit(
        **{
            f"trainable_params_{name}": torch.tensor(
                count,
                dtype=torch.float32,
                device=torch.device("cuda", device_id),
            )
            for name, count in trainable_groups.items()
        }
    )

    max_steps = variant["trainer"]["max_steps"]
    cognition_loss_weight, action_loss_weight = resolve_loss_weights(variant)
    loss_mode = resolve_loss_mode(variant)
    action_distill_target = resolve_action_distill_target(variant)
    action_loss_required = loss_mode in {"action_only", "normalized"} or action_loss_weight > 0.0
    normalizer_config = variant.get("loss_normalization", {})
    loss_normalizer = RunningLossNormalizer(
        decay=normalizer_config.get("ema_decay", 0.99),
        eps=normalizer_config.get("eps", 1e-8),
    )
    vitkd_config = variant.get("vitkd", {})
    save_checkpoint = variant.get("save_checkpoint", True)
    save_steps = variant["save_steps"]
    max_saved_checkpoints = variant.get("max_saved_checkpoints")

    status = metrics.get_status()
    progress_total = max_steps if max_steps is not None else variant["trainer"]["max_epochs"] * math.ceil(
        len(dataloader) / training_strategy.grad_accumulation_steps
    )
    train_idx = 0
    with tqdm(total=progress_total, desc=status, leave=False, disable=not overwatch.is_rank_zero()) as progress:
        for epoch in range(variant["trainer"]["max_epochs"]):
            student.train()
            training_strategy.optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                batch = move_to_device(batch, device_id)
                input_ids = batch["input_ids"]
                rgb = batch["pixel_values"]
                attention_mask = batch["attention_mask"]
                current_state_mask = batch["current_state_mask"]
                current_state = batch["current_state"]
                fov = batch["fov"]
                action_labels = batch.get("actions")
                action_masks = batch.get("action_masks")

                teacher_inputs = prepare_teacher_inputs(teacher, batch, device_id)
                teacher_action_targets = None
                if loss_mode == "vitkd":
                    shallow_layers = vitkd_config.get("shallow_layers", [0, 1])
                    deep_layer = vitkd_config.get("deep_layer", -1)
                    with torch.no_grad():
                        teacher_features = teacher.extract_vitkd_features(
                            **teacher_inputs,
                            shallow_layers=shallow_layers,
                            deep_layer=deep_layer,
                        )
                    student_features = student(
                        rgb,
                        input_ids,
                        attention_mask=attention_mask,
                        current_state_mask=current_state_mask,
                        current_state=current_state,
                        fov=fov,
                        mode="vitkd_features",
                        shallow_layers=shallow_layers,
                        deep_layer=deep_layer,
                    )
                    vitkd_losses, loss_metrics = vitkd_distill_losses(
                        student_features,
                        teacher_features,
                        vitkd_config,
                    )
                    vlm_cognition_mse = vitkd_losses.get("cognition")
                else:
                    with torch.no_grad():
                        if action_loss_required and action_distill_target == "teacher":
                            if action_masks is None:
                                raise KeyError("Teacher action distillation requested, but batch is missing 'action_masks'.")
                            teacher_cognition, teacher_action_targets = extract_teacher_cognition_and_actions(
                                teacher=teacher,
                                teacher_inputs=teacher_inputs,
                                action_masks=action_masks,
                                variant=variant,
                            )
                        else:
                            teacher_cognition = teacher.extract_cognition_features(**teacher_inputs)
                    student_cognition = student(
                        rgb,
                        input_ids,
                        attention_mask=attention_mask,
                        current_state_mask=current_state_mask,
                        current_state=current_state,
                        fov=fov,
                        mode="vlm_cognition",
                    )

                    distill_metrics = cognition_distill_metrics(student_cognition, teacher_cognition)
                    vlm_cognition_mse = distill_metrics["vlm_cognition_mse"]
                    loss_metrics = {k: v.detach() for k, v in distill_metrics.items()}
                    loss_metrics["cognition_loss"] = vlm_cognition_mse.detach()

                action_loss = None
                if action_loss_required:
                    if action_masks is None:
                        raise KeyError("Action loss requested, but batch is missing 'action_masks'.")
                    if action_distill_target == "gt":
                        if action_labels is None:
                            raise KeyError("GT action loss requested, but batch is missing 'actions'.")
                        action_targets = action_labels
                    else:
                        action_targets = teacher_action_targets
                        if action_targets is None:
                            with torch.no_grad():
                                _, action_targets = extract_teacher_cognition_and_actions(
                                    teacher=teacher,
                                    teacher_inputs=teacher_inputs,
                                    action_masks=action_masks,
                                    variant=variant,
                                )
                        loss_metrics["vlm_teacher_action_target_mean"] = action_targets.float().mean().detach()

                    action_prediction = student(
                        rgb,
                        input_ids,
                        attention_mask=attention_mask,
                        action_labels=action_targets,
                        action_masks=action_masks,
                        current_state_mask=current_state_mask,
                        current_state=current_state,
                        fov=fov,
                        mode="train",
                    )
                    action_loss = action_prediction["loss"]
                    loss_metrics["action_loss"] = action_loss.detach()
                    loss_metrics[f"vlm_action_loss_{action_distill_target}"] = action_loss.detach()

                    for key, value in action_prediction.items():
                        if key == "loss":
                            continue
                        if isinstance(value, torch.Tensor):
                            loss_metrics[f"vlm_action_{key}"] = value.detach()

                if loss_mode == "action_only":
                    if action_loss is None:
                        raise RuntimeError("distill_loss_mode='action_only' requires action loss.")
                    loss = action_loss
                elif loss_mode == "normalized":
                    if action_loss is None:
                        raise RuntimeError("distill_loss_mode='normalized' requires action loss.")
                    cognition_loss_norm, ema_cognition_loss = loss_normalizer.normalize("cognition", vlm_cognition_mse)
                    action_loss_norm, ema_action_loss = loss_normalizer.normalize("action", action_loss)
                    loss = cognition_loss_norm + action_loss_norm
                    loss_metrics["ema_cognition_loss"] = ema_cognition_loss.detach()
                    loss_metrics["ema_action_loss"] = ema_action_loss.detach()
                    loss_metrics["cognition_loss_norm"] = cognition_loss_norm.detach()
                    loss_metrics["action_loss_norm"] = action_loss_norm.detach()
                elif loss_mode == "vitkd":
                    normalized_terms = []
                    for name, raw_loss in vitkd_losses.items():
                        normalized_loss, ema_loss = loss_normalizer.normalize(f"vitkd_{name}", raw_loss)
                        normalized_terms.append(normalized_loss)
                        loss_metrics[f"ema_vitkd_{name}_loss"] = ema_loss.detach()
                        loss_metrics[f"vitkd_{name}_loss_norm"] = normalized_loss.detach()
                    loss = torch.stack(normalized_terms).sum()
                else:
                    loss = cognition_loss_weight * vlm_cognition_mse
                    if action_loss is not None:
                        loss = loss + action_loss_weight * action_loss

                loss_metrics["total_loss"] = loss.detach()
                loss_metrics["vlm_total_loss"] = loss.detach()
                logged_action_weight = 1.0 if loss_mode in {"action_only", "normalized"} else action_loss_weight
                logged_cognition_weight = 0.0 if loss_mode == "action_only" else cognition_loss_weight
                if loss_mode == "normalized":
                    logged_cognition_weight = 1.0
                loss_metrics["vlm_cognition_loss_weight"] = torch.tensor(logged_cognition_weight, device=loss.device)
                loss_metrics["vlm_action_loss_weight"] = torch.tensor(logged_action_weight, device=loss.device)
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f"Non-finite distillation loss at global_step={metrics.global_step}, batch_idx={batch_idx}"
                    )

                metrics.commit(loss=loss.detach(), **loss_metrics)
                (loss / training_strategy.grad_accumulation_steps).backward()

                if (train_idx + 1) % training_strategy.grad_accumulation_steps == 0:
                    training_strategy.clip_grad_norm()
                    training_strategy.optimizer.step()
                    training_strategy.lr_scheduler.step()
                    training_strategy.optimizer.zero_grad()

                    lr_values = training_strategy.lr_scheduler.get_last_lr()
                    lr_dict = {
                        "action_decay_lr": torch.tensor(lr_values[2] if len(lr_values) > 2 else lr_values[0]),
                    }
                    metrics.commit(
                        update_step_time=True,
                        global_step=metrics.global_step + 1,
                        epoch=epoch,
                        lr=lr_values[0],
                        **lr_dict,
                    )
                    status = metrics.push()

                    terminate = max_steps is not None and metrics.global_step >= max_steps
                    should_save = save_checkpoint and (terminate or metrics.global_step % save_steps == 0)
                    if should_save:
                        saved_dir = save_student_checkpoint(
                            student,
                            checkpoint_dir,
                            metrics.global_step,
                            epoch,
                            max_saved_checkpoints=max_saved_checkpoints,
                        )
                        maybe_run_action_eval(variant, saved_dir)

                    if terminate:
                        metrics.finalize()
                        distributed_barrier_for_device(device_id)
                        dist.destroy_process_group()
                        return

                train_idx += 1
                progress.set_description(status)
                progress.update()

    metrics.finalize()
    distributed_barrier_for_device(device_id)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="VITRA VLM cognition-token distillation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher_config", default=None)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--log_root", default=None)
    parser.add_argument("--cache_root", default=None)
    parser.add_argument("--use_bf16", default=None, action="store_true")
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--total_batch_size", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--prefetch_factor", default=None, type=int)
    parser.add_argument("--save_steps", default=None, type=int)
    parser.add_argument("--max_saved_checkpoints", default=None, type=int)
    parser.add_argument("--cognition_loss_weight", default=None, type=float)
    parser.add_argument("--action_loss_weight", default=None, type=float)
    parser.add_argument("--action_distill_target", default=None, type=str)
    parser.add_argument("--distill_loss_mode", default=None, type=str)
    parser.add_argument("--no_save_checkpoint", default=False, action="store_true")
    parser.add_argument("--disable_action_eval", default=False, action="store_true")

    global_names = set(vars(parser.parse_known_args()[0]).keys())
    trainer_parser = parser.add_argument_group("trainer")
    trainer_parser.add_argument("--strategy", default=None, type=str)
    trainer_parser.add_argument("--gradient_clip_val", default=None, type=float)
    trainer_parser.add_argument("--max_steps", default=None, type=int)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    temp_args = vars(parser.parse_args())
    args = {}
    trainer_args = {}
    for key, value in temp_args.items():
        if key in global_names:
            args[key] = value
        elif key in trainer_names:
            trainer_args[key] = value
    args["trainer"] = trainer_args
    return args


if __name__ == "__main__":
    faulthandler.enable()
    args = parse_args()
    configs = load_config(args["config"])
    configs = update_configs(configs, args)

    dist_backend = os.environ.get("DIST_BACKEND", "nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        init_kwargs = {"backend": dist_backend}
        if dist_backend == "nccl" and torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device(f"cuda:{torch.cuda.current_device()}")
        dist.init_process_group(**init_kwargs)

    experiment(configs)
