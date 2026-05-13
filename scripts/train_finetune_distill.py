#!/usr/bin/env python3
"""Joint finetune + VITRA feature distillation for compressed GigaHands students."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.datasets.materialize import get_vla_dataset_and_collator
from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.utils.config_utils import load_config


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
faulthandler.enable()


def debug_log(message: str) -> None:
    if os.environ.get("FINETUNE_DISTILL_DEBUG", "1") == "0":
        return
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    print(f"[finetune-distill][rank={rank} local_rank={local_rank}] {message}", flush=True)


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
        ema = self.update(name, loss).to(loss.device)
        return loss / torch.clamp(ema, min=self.eps), ema


def combine_joint_losses(
    feature_loss: torch.Tensor,
    action_gt_loss: torch.Tensor,
    action_kd_loss: torch.Tensor | None,
    normalizer: RunningLossNormalizer,
    feature_loss_weight: float,
    action_loss_weight: float,
    action_kd_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature_loss_norm, ema_feature = normalizer.normalize("feature", feature_loss)
    action_gt_loss_norm, ema_action_gt = normalizer.normalize("action_gt", action_gt_loss)
    total = float(feature_loss_weight) * feature_loss_norm + float(action_loss_weight) * action_gt_loss_norm
    metrics = {
        "feature_loss": feature_loss.detach(),
        "action_gt_loss": action_gt_loss.detach(),
        "action_loss": action_gt_loss.detach(),
        "feature_loss_norm": feature_loss_norm.detach(),
        "action_gt_loss_norm": action_gt_loss_norm.detach(),
        "action_loss_norm": action_gt_loss_norm.detach(),
        "ema_feature_loss": ema_feature.detach(),
        "ema_action_gt_loss": ema_action_gt.detach(),
        "ema_action_loss": ema_action_gt.detach(),
    }
    if action_kd_loss is not None and float(action_kd_loss_weight) != 0.0:
        action_kd_loss_norm, ema_action_kd = normalizer.normalize("action_kd", action_kd_loss)
        total = total + float(action_kd_loss_weight) * action_kd_loss_norm
        metrics.update(
            {
                "action_kd_loss": action_kd_loss.detach(),
                "action_kd_loss_norm": action_kd_loss_norm.detach(),
                "ema_action_kd_loss": ema_action_kd.detach(),
            }
        )
    else:
        zero = action_gt_loss.detach() * 0.0
        metrics.update(
            {
                "action_kd_loss": zero,
                "action_kd_loss_norm": zero,
                "ema_action_kd_loss": zero,
            }
        )
    metrics["total_loss"] = total.detach()
    return total, metrics


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=prediction.dtype, device=prediction.device)
    square_delta = (prediction.float() - target.float()) ** 2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return square_delta.sum() / denom


def repeat_for_diffusion(
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    teacher_cognition: torch.Tensor,
    student_cognition: torch.Tensor,
    current_state: torch.Tensor,
    current_state_mask: torch.Tensor,
    repeated: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = actions.shape[0]
    actions_repeated = actions.unsqueeze(0).repeat(repeated, 1, 1, 1).view(
        batch_size * repeated,
        actions.shape[1],
        actions.shape[2],
    )
    masks_repeated = action_masks.unsqueeze(0).repeat(repeated, 1, 1, 1).view(
        batch_size * repeated,
        action_masks.shape[1],
        action_masks.shape[2],
    )
    teacher_cognition_repeated = teacher_cognition.unsqueeze(0).repeat(repeated, 1, 1).view(
        batch_size * repeated,
        1,
        teacher_cognition.shape[-1],
    )
    student_cognition_repeated = student_cognition.unsqueeze(0).repeat(repeated, 1, 1).view(
        batch_size * repeated,
        1,
        student_cognition.shape[-1],
    )
    state_repeated = current_state.unsqueeze(0).repeat(repeated, 1, 1).view(
        batch_size * repeated,
        1,
        current_state.shape[-1],
    )
    state_mask_repeated = current_state_mask.unsqueeze(0).repeat(repeated, 1, 1).view(
        batch_size * repeated,
        1,
        current_state_mask.shape[-1],
    )
    return (
        actions_repeated,
        masks_repeated,
        teacher_cognition_repeated,
        student_cognition_repeated,
        state_repeated,
        state_mask_repeated,
    )


def compute_action_kd_loss(
    teacher,
    student_module,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    teacher_cognition: torch.Tensor,
    student_cognition: torch.Tensor,
    current_state: torch.Tensor,
    current_state_mask: torch.Tensor,
    repeated: int,
    use_bf16: bool,
) -> torch.Tensor:
    (
        actions_repeated,
        masks_repeated,
        teacher_cognition_repeated,
        student_cognition_repeated,
        state_repeated,
        state_mask_repeated,
    ) = repeat_for_diffusion(
        actions,
        action_masks,
        teacher_cognition,
        student_cognition,
        current_state,
        current_state_mask,
        repeated,
    )

    diffusion = student_module.act_model.diffusion
    noise = torch.randn_like(actions_repeated)
    timestep = torch.randint(0, diffusion.num_timesteps, (actions_repeated.size(0),), device=actions_repeated.device)
    x_t = diffusion.q_sample(actions_repeated, timestep, noise)
    x_t = x_t * masks_repeated
    x_in = torch.cat([x_t, masks_repeated], dim=2)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            eps_teacher = teacher.act_model.net(
                x_in,
                timestep,
                teacher_cognition_repeated,
                state_repeated,
                state_mask_repeated,
            )
        eps_teacher = eps_teacher.float()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        eps_student = student_module.act_model.net(
            x_in,
            timestep,
            student_cognition_repeated,
            state_repeated,
            state_mask_repeated,
        )
    eps_student = eps_student.float()
    return masked_mse(eps_student, eps_teacher, masks_repeated)


def setup_distributed() -> tuple[int, int, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    torch.cuda.set_device(local_rank)
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=os.environ.get("DIST_BACKEND", "nccl"))
    return world_size, local_rank, torch.device(f"cuda:{local_rank}"), distributed


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def resolve_weights_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "weights.pt" if path.is_dir() else path


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def prepare_teacher_inputs(teacher, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
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
    ).to(device)
    return {
        "pixel_values": model_inputs["pixel_values"],
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs.get("attention_mask", torch.ones_like(model_inputs["input_ids"])).to(torch.bool),
        "current_state_mask": batch["current_state_mask"],
        "current_state": batch["current_state"],
        "fov": batch["fov"],
    }


def build_dataloader(config: dict[str, Any], processor, world_size: int, rank: int):
    dataset_cfg = config["train_dataset"]
    train_cfg = config["training"]
    dataset, collator, _ = get_vla_dataset_and_collator(
        dataset_cfg["data_root_dir"],
        dataset_cfg.get("data_mix", "gigahands_real_train"),
        augmentation=dataset_cfg.get("augmentation", False),
        shard_num=1,
        shard_index=0,
        seed=config.get("seed", 42),
        future_action_window_size=config.get("fwd_pred_next_n", 16) - 1,
        processor=processor,
        batch_size=train_cfg.get("batch_size_per_gpu", config.get("batch_size", 1)),
        normalization=dataset_cfg.get("normalization", True),
        flip_augmentation=dataset_cfg.get("flip_augmentation", 1.0),
        set_none_ratio=dataset_cfg.get("set_none_ratio", 0.0),
        action_type=dataset_cfg.get("action_type", "keypoints"),
        use_rel=dataset_cfg.get("use_rel", False),
        rel_mode=dataset_cfg.get("rel_mode", "step"),
        clip_len=dataset_cfg.get("clip_len"),
        state_mask_prob=dataset_cfg.get("state_mask_prob", 0.0),
        target_image_height=dataset_cfg.get("target_image_height", 224),
        statistics_dataset_name=dataset_cfg.get("statistics_dataset_name", "gigahands_real_train"),
    )
    max_train_samples = int(train_cfg.get("max_train_samples", 0) or 0)
    if max_train_samples > 0:
        dataset = Subset(dataset, range(min(max_train_samples, len(dataset))))

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.get("seed", 42))
    else:
        sampler = RandomSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size_per_gpu", config.get("batch_size", 1)),
        sampler=sampler,
        collate_fn=collator,
        num_workers=dataset_cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    ), sampler


def write_csv_row(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def cpu_state_dict(module) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def save_checkpoint(student, output_dir: Path, step: int, epoch: int, metrics: dict[str, float], keep_last: int) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"epoch={epoch}-step={step}.ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    module = student.module if isinstance(student, DistributedDataParallel) else student
    torch.save(cpu_state_dict(module), checkpoint_dir / "weights.pt")
    (checkpoint_dir / "meta.json").write_text(
        json.dumps({"epoch": epoch, "global_step": step, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    if keep_last > 0:
        checkpoints = sorted((output_dir / "checkpoints").glob("epoch=*-step=*.ckpt"), key=lambda p: p.stat().st_mtime)
        for stale in checkpoints[:-keep_last]:
            shutil.rmtree(stale)
    return checkpoint_dir


def train(config: dict[str, Any]) -> None:
    world_size, local_rank, device, distributed = setup_distributed()
    rank = int(os.environ.get("RANK", "0"))
    torch.manual_seed(int(config.get("seed", 42)) + rank)
    debug_log(f"initialized distributed={distributed} world_size={world_size} device={device}")

    teacher_config = load_config(str(config["teacher_config"]))
    teacher_config["train_dataset"]["data_root_dir"] = config["train_dataset"]["data_root_dir"]
    debug_log("building teacher")
    teacher = build_vla(teacher_config)
    debug_log("loading teacher checkpoint")
    teacher = load_vla_checkpoint(teacher, str(resolve_weights_path(config["teacher_checkpoint"])))
    debug_log("freezing/moving teacher")
    teacher = freeze_model(teacher).to(device)
    teacher.use_bf16 = bool(config.get("use_bf16", True))
    if hasattr(teacher, "model"):
        teacher.model.use_bf16 = bool(config.get("use_bf16", True))

    debug_log("building student")
    student = build_vla(config)
    if config.get("student_init_checkpoint"):
        debug_log("loading student init checkpoint")
        student = load_vla_checkpoint(student, str(resolve_weights_path(config["student_init_checkpoint"])))
    debug_log("moving student")
    student = student.to(device).train()
    student.use_bf16 = bool(config.get("use_bf16", True))
    if distributed:
        debug_log("wrapping student with DistributedDataParallel")
        student = DistributedDataParallel(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
            init_sync=bool(config.get("training", {}).get("ddp_init_sync", False)),
        )
        debug_log("wrapped student with DistributedDataParallel")

    trainable_params = [param for param in student.parameters() if param.requires_grad]
    train_cfg = config["training"]
    debug_log(f"creating optimizer with trainable_param_tensors={len(trainable_params)}")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("lr", config["trainer"].get("learning_rate", 1e-5))),
        weight_decay=float(train_cfg.get("weight_decay", config["trainer"].get("weight_decay", 0.1))),
    )
    debug_log("building dataloader")
    dataloader, sampler = build_dataloader(config, student.module.processor if isinstance(student, DistributedDataParallel) else student.processor, world_size, rank)
    debug_log("built dataloader")

    output_dir = Path(config["output_root"]) / f"{config['task_name']}_TB{config['total_batch_size']}_B{config['batch_size']}_bf16{config['use_bf16']}"
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    normalizer = RunningLossNormalizer(
        decay=config.get("loss_normalization", {}).get("ema_decay", 0.99),
        eps=config.get("loss_normalization", {}).get("eps", 1e-8),
    )
    max_steps = int(config["trainer"].get("max_steps", train_cfg.get("max_steps", 1000)))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    save_steps = int(config.get("save_steps", train_cfg.get("save_every", 20000)))
    log_every = int(train_cfg.get("log_every", 50))
    keep_last = int(config.get("max_saved_checkpoints", 3))
    feature_weight = float(config.get("feature_loss_weight", 1.0))
    action_weight = float(config.get("action_loss_weight", 1.0))
    action_kd_weight = float(config.get("action_kd_loss_weight", 0.0))
    use_bf16 = bool(config.get("use_bf16", True))
    csv_path = output_dir / "logs" / "loss_curve.csv"
    student_module = student.module if isinstance(student, DistributedDataParallel) else student

    global_step = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    last_metrics: dict[str, float] = {}
    while global_step < max_steps:
        if distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            start = time.time()
            batch = move_to_device(batch, device)
            teacher_inputs = prepare_teacher_inputs(teacher, batch, device)
            with torch.no_grad():
                teacher_cognition = teacher.extract_cognition_features(**teacher_inputs)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                student_cognition = student(
                    batch["pixel_values"],
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    current_state_mask=batch["current_state_mask"],
                    current_state=batch["current_state"],
                    fov=batch["fov"],
                    mode="vlm_cognition",
                )
                _, action_prediction = student_module._forward_act_model(
                    student_cognition,
                    action_labels=batch["actions"],
                    action_masks=batch["action_masks"],
                    current_state=batch["current_state"],
                    current_state_mask=batch["current_state_mask"],
                    mode="train",
                    repeated_diffusion_steps=student_module.repeated_diffusion_steps,
                )
                feature_loss = F.mse_loss(student_cognition.float(), teacher_cognition.detach().float())
                action_gt_loss = action_prediction["loss"] if isinstance(action_prediction, dict) else action_prediction
                action_kd_loss = None
                if action_kd_weight != 0.0:
                    action_kd_loss = compute_action_kd_loss(
                        teacher=teacher,
                        student_module=student_module,
                        actions=batch["actions"],
                        action_masks=batch["action_masks"],
                        teacher_cognition=teacher_cognition.detach(),
                        student_cognition=student_cognition,
                        current_state=batch["current_state"],
                        current_state_mask=batch["current_state_mask"],
                        repeated=student_module.repeated_diffusion_steps,
                        use_bf16=use_bf16,
                    )
                total_loss, loss_metrics = combine_joint_losses(
                    feature_loss,
                    action_gt_loss,
                    action_kd_loss,
                    normalizer,
                    feature_loss_weight=feature_weight,
                    action_loss_weight=action_weight,
                    action_kd_loss_weight=action_kd_weight,
                )

            if not torch.isfinite(total_loss).all():
                raise RuntimeError(f"Non-finite loss at step={global_step} batch={batch_idx}")
            (total_loss / grad_accum_steps).backward()

            if (global_step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(train_cfg.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            last_metrics = {
                "step": float(global_step),
                "feature_loss": float(loss_metrics["feature_loss"].detach().cpu()),
                "action_loss": float(loss_metrics["action_loss"].detach().cpu()),
                "action_gt_loss": float(loss_metrics["action_gt_loss"].detach().cpu()),
                "action_kd_loss": float(loss_metrics["action_kd_loss"].detach().cpu()),
                "feature_loss_norm": float(loss_metrics["feature_loss_norm"].detach().cpu()),
                "action_loss_norm": float(loss_metrics["action_loss_norm"].detach().cpu()),
                "action_gt_loss_norm": float(loss_metrics["action_gt_loss_norm"].detach().cpu()),
                "action_kd_loss_norm": float(loss_metrics["action_kd_loss_norm"].detach().cpu()),
                "total_loss": float(loss_metrics["total_loss"].detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds_per_step": time.time() - start,
                "gpu_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            }
            if is_rank0() and (global_step == 1 or global_step % log_every == 0):
                write_csv_row(csv_path, last_metrics)
                print(json.dumps(last_metrics), flush=True)
            if is_rank0() and (global_step % save_steps == 0 or global_step >= max_steps):
                checkpoint_dir = save_checkpoint(student, output_dir, global_step, epoch, last_metrics, keep_last)
                print(json.dumps({"saved_checkpoint": str(checkpoint_dir)}, indent=2), flush=True)
            if global_step >= max_steps:
                break
        epoch += 1

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(str(args.config))
    if args.max_steps is not None:
        config.setdefault("trainer", {})["max_steps"] = args.max_steps
    if args.save_steps is not None:
        config["save_steps"] = args.save_steps
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
        config.setdefault("training", {})["batch_size_per_gpu"] = args.batch_size
    if args.total_batch_size is not None:
        config["total_batch_size"] = args.total_batch_size
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
