#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.touch_editor.dataset import TouchEditorCacheDataset
from vitra.touch_editor.losses import hand_scope_action_mask, editable_mask, masked_mean_square, touch_editor_loss, zero_delta_loss
from vitra.touch_editor.model import (
    PretrainedTactileGatedResidualEditor,
    ResidualTouchEditor,
    TactileGatedResidualEditor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the residual OpenTouch editor from cached VLA predictions.")
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/touch_editor"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="Optional editor checkpoint to initialize from.")
    parser.add_argument("--editor_type", choices=("residual", "tactile_gated", "pretrained_tactile_gated"), default="residual")
    parser.add_argument("--condition_mode", choices=("full", "no_base", "touch_only"), default="full")
    parser.add_argument("--action_dim", type=int, default=192)
    parser.add_argument("--state_dim", type=int, default=212)
    parser.add_argument("--touch_feature_dim", type=int, default=128)
    parser.add_argument("--pretrained_touch_encoder_checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained_touch_encoder_config", type=Path, default=None)
    parser.add_argument("--pretrained_touch_embed_dim", type=int, default=64)
    parser.add_argument("--pretrained_touch_hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--freeze_pretrained_touch_encoder",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--context_dropout_prob", type=float, default=0.0)
    parser.add_argument("--hand_scope", choices=("both", "left", "right"), default="both")
    parser.add_argument("--lambda_dev", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--negative_touch_loss", choices=("none", "zero_delta"), default="none")
    parser.add_argument("--lambda_shuffle_zero", type=float, default=0.0)
    parser.add_argument("--lambda_zero_zero", type=float, default=0.0)
    parser.add_argument("--lambda_zero_delta", dest="lambda_zero_zero", type=float)
    parser.add_argument("--lambda_margin", type=float, default=0.0)
    parser.add_argument("--lambda_shuffle_margin", dest="lambda_margin", type=float)
    parser.add_argument("--shuffle_margin", type=float, default=0.05)
    parser.add_argument("--lambda_touch_gate", type=float, default=0.0)
    parser.add_argument("--contact_weighting", choices=("none", "observed_score", "observed_delta"), default="none")
    parser.add_argument("--contact_subset", choices=("all", "high_contact"), default="all")
    parser.add_argument("--high_contact_quantile", type=float, default=0.75)
    parser.add_argument(
        "--touch_ablation",
        choices=("none", "zero_touch"),
        default="none",
        help="Optionally ablate tactile input during training. Use zero_touch for a no-touch editor.",
    )
    args = parser.parse_args()
    if args.lambda_zero_zero is None:
        args.lambda_zero_zero = 0.0
    if args.lambda_margin is None:
        args.lambda_margin = 0.0
    return args


def build_touch_editor(args: argparse.Namespace) -> ResidualTouchEditor | TactileGatedResidualEditor | PretrainedTactileGatedResidualEditor:
    common = {
        "action_dim": args.action_dim,
        "state_dim": args.state_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "condition_mode": args.condition_mode,
    }
    if args.editor_type == "residual":
        return ResidualTouchEditor(**common, touch_feature_dim=args.touch_feature_dim)
    if args.editor_type == "tactile_gated":
        return TactileGatedResidualEditor(
            **common,
            touch_feature_dim=args.touch_feature_dim,
            num_heads=args.num_heads,
            context_dropout_prob=args.context_dropout_prob,
        )
    if args.editor_type == "pretrained_tactile_gated":
        if args.pretrained_touch_encoder_checkpoint is None and args.init_checkpoint is None:
            raise ValueError("--pretrained_touch_encoder_checkpoint is required for pretrained_tactile_gated training")
        return PretrainedTactileGatedResidualEditor(
            **common,
            num_heads=args.num_heads,
            context_dropout_prob=args.context_dropout_prob,
            pretrained_touch_encoder_checkpoint=args.pretrained_touch_encoder_checkpoint,
            pretrained_touch_embed_dim=args.pretrained_touch_embed_dim,
            pretrained_touch_hand=args.pretrained_touch_hand,
            freeze_pretrained_touch_encoder=args.freeze_pretrained_touch_encoder,
        )
    raise ValueError(f"Unsupported editor_type: {args.editor_type}")


def _score_cache_path(path: Path) -> float:
    with np.load(path, allow_pickle=False) as payload:
        contact_score = float(np.asarray(payload.get("observed_touch_contact_score", 0.0)).item())
        contact_delta = float(np.asarray(payload.get("observed_touch_contact_delta", 0.0)).item())
    return max(contact_score, contact_delta)


def filter_dataset_by_observed_contact(
    dataset: TouchEditorCacheDataset,
    *,
    quantile: float,
) -> TouchEditorCacheDataset:
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    scored_paths = [(path, _score_cache_path(Path(path))) for path in dataset.paths]
    if not scored_paths:
        return dataset
    threshold = torch.quantile(torch.tensor([score for _, score in scored_paths], dtype=torch.float32), float(quantile)).item()
    filtered_paths = [path for path, score in scored_paths if score >= threshold]
    filtered = object.__new__(TouchEditorCacheDataset)
    filtered.cache_root = dataset.cache_root
    filtered.paths = filtered_paths
    return filtered


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def apply_touch_training_ablation(
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    touch_ablation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if touch_ablation == "zero_touch":
        return torch.zeros_like(touch_pressure), torch.zeros_like(touch_mask)
    return touch_pressure, touch_mask


def shuffled_touch_for_training(
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(touch_pressure.shape[0])
    if batch_size <= 1:
        return touch_pressure, touch_mask
    order = torch.roll(torch.arange(batch_size, device=touch_pressure.device), shifts=1)
    return touch_pressure[order], touch_mask[order]


def causal_touch_history_from_batch(
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
    edit_start_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only touch observations available at each sample's edit time.

    ``edit_start_idx`` is inclusive: editing at index 3 may use tactile
    observations from indices 0, 1, 2, and 3, but never indices 4+.
    """
    if touch_pressure.ndim != 5:
        raise ValueError(f"touch_pressure must be [B,T,2,16,16], got {tuple(touch_pressure.shape)}")
    if touch_mask.ndim != 3:
        raise ValueError(f"touch_mask must be [B,T,2], got {tuple(touch_mask.shape)}")
    if touch_pressure.shape[:2] != touch_mask.shape[:2]:
        raise ValueError("touch_pressure and touch_mask batch/time dimensions must match")
    if touch_pressure.shape[0] == 0:
        return touch_pressure, touch_mask

    edit_start_idx = torch.as_tensor(edit_start_idx, device=touch_pressure.device, dtype=torch.long).reshape(-1)
    if edit_start_idx.numel() == 1 and touch_pressure.shape[0] > 1:
        edit_start_idx = edit_start_idx.expand(touch_pressure.shape[0])
    if edit_start_idx.numel() != touch_pressure.shape[0]:
        raise ValueError(
            f"edit_start_idx must have batch size {touch_pressure.shape[0]}, got {edit_start_idx.numel()}"
        )

    max_time = int(touch_pressure.shape[1])
    observed_lens = (edit_start_idx.clamp(min=0, max=max_time - 1) + 1).clamp(min=1, max=max_time)
    max_observed_len = int(observed_lens.max().item())
    history = touch_pressure[:, :max_observed_len].clone()
    history_mask = touch_mask[:, :max_observed_len].clone()
    time = torch.arange(max_observed_len, device=touch_pressure.device)[None, :]
    valid_history = time < observed_lens[:, None]
    history = history * valid_history[:, :, None, None, None].to(history.dtype)
    history_mask = history_mask & valid_history[:, :, None]
    return history, history_mask


def run_touch_editor(
    model: ResidualTouchEditor,
    batch: dict[str, torch.Tensor],
    touch_pressure: torch.Tensor,
    touch_mask: torch.Tensor,
) -> torch.Tensor:
    return model(
        batch["a_base"],
        batch["current_state"],
        batch["current_state_mask"],
        touch_pressure,
        touch_mask,
        batch["chunk_phase"],
        batch["future_mask"],
        batch["action_mask"],
    )


def shuffled_margin_loss(
    *,
    a_base: torch.Tensor,
    a_target: torch.Tensor,
    delta: torch.Tensor,
    action_mask: torch.Tensor,
    future_mask: torch.Tensor,
    matched_demo_loss: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    editable = editable_mask(action_mask, future_mask).to(delta.dtype)
    a_edit = a_base + editable * delta
    shuffled_demo = masked_mean_square(a_edit - a_target, editable)
    return torch.relu(delta.new_tensor(float(margin)) + matched_demo_loss.detach() - shuffled_demo)


def apply_hand_scope_to_batch(batch: dict[str, torch.Tensor], hand_scope: str) -> dict[str, torch.Tensor]:
    if hand_scope == "both":
        return batch
    batch = dict(batch)
    batch["action_mask"] = hand_scope_action_mask(batch["action_mask"], hand_scope)
    batch["future_mask"] = batch["future_mask"].to(batch["action_mask"].dtype) * batch["action_mask"].to(batch["future_mask"].dtype)
    return batch


def contact_sample_weight(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    key = "observed_touch_contact_delta" if mode == "observed_delta" else "observed_touch_contact_score"
    if key not in batch:
        return None
    raw = batch[key].to(dtype=torch.float32).reshape(-1)
    if raw.numel() == 0:
        return None
    centered = raw / raw.detach().mean().clamp_min(1e-6)
    return centered.clamp(min=0.25, max=4.0)


def touch_gate_regularization(model: torch.nn.Module, action_mask: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    diagnostics = getattr(model, "last_diagnostics", {})
    gate = diagnostics.get("touch_gate") if isinstance(diagnostics, dict) else None
    if not torch.is_tensor(gate):
        return action_mask.new_tensor(0.0, dtype=torch.float32)
    editable = editable_mask(action_mask, future_mask).to(gate.dtype)
    timestep_valid = editable.any(dim=-1, keepdim=True).to(gate.dtype)
    return masked_mean_square(1.0 - gate, timestep_valid)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = TouchEditorCacheDataset(args.cache_root)
    if args.contact_subset == "high_contact":
        dataset = filter_dataset_by_observed_contact(dataset, quantile=args.high_contact_quantile)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    model = build_touch_editor(args).to(args.device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Touch editor has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    metrics_path = args.output_dir / "metrics.jsonl"
    step = 0
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        progress = tqdm(total=args.max_steps, desc="training touch editor")
        while step < args.max_steps:
            for batch in loader:
                batch = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in batch.items()}
                batch = apply_hand_scope_to_batch(batch, args.hand_scope)
                touch_pressure, touch_mask = apply_touch_training_ablation(
                    batch["touch_pressure"],
                    batch["touch_mask"],
                    args.touch_ablation,
                )
                touch_pressure, touch_mask = causal_touch_history_from_batch(
                    touch_pressure,
                    touch_mask,
                    batch["edit_start_idx"],
                )
                delta = run_touch_editor(model, batch, touch_pressure, touch_mask)
                sample_weight = contact_sample_weight(batch, args.contact_weighting)
                losses = touch_editor_loss(
                    batch["a_base"],
                    batch["a_target"],
                    delta,
                    batch["action_mask"],
                    batch["future_mask"],
                    residual_target=batch["residual_target"],
                    lambda_dev=args.lambda_dev,
                    lambda_delta=args.lambda_delta,
                    lambda_smooth=args.lambda_smooth,
                    lambda_mask=args.lambda_mask,
                    sample_weight=sample_weight,
                )
                total_loss = losses["loss"]
                if args.lambda_touch_gate > 0:
                    losses["loss_touch_gate"] = touch_gate_regularization(model, batch["action_mask"], batch["future_mask"])
                    total_loss = total_loss + args.lambda_touch_gate * losses["loss_touch_gate"]
                if args.negative_touch_loss == "zero_delta" or args.lambda_shuffle_zero > 0 or args.lambda_margin > 0 or args.lambda_zero_zero > 0:
                    if args.lambda_shuffle_zero > 0 or args.lambda_margin > 0:
                        shuffled_pressure, shuffled_mask = shuffled_touch_for_training(touch_pressure, touch_mask)
                        shuffled_delta = run_touch_editor(model, batch, shuffled_pressure, shuffled_mask)
                        losses["loss_shuffle_zero"] = zero_delta_loss(
                            shuffled_delta,
                            batch["action_mask"],
                            batch["future_mask"],
                        )
                        total_loss = total_loss + args.lambda_shuffle_zero * losses["loss_shuffle_zero"]
                        if args.lambda_margin > 0:
                            losses["loss_shuffle_margin"] = shuffled_margin_loss(
                                a_base=batch["a_base"],
                                a_target=batch["a_target"],
                                delta=shuffled_delta,
                                action_mask=batch["action_mask"],
                                future_mask=batch["future_mask"],
                                matched_demo_loss=losses["loss_demo"],
                                margin=args.shuffle_margin,
                            )
                            total_loss = total_loss + args.lambda_margin * losses["loss_shuffle_margin"]
                    if args.lambda_zero_zero > 0:
                        zero_delta = run_touch_editor(
                            model,
                            batch,
                            torch.zeros_like(touch_pressure),
                            torch.zeros_like(touch_mask),
                        )
                        losses["loss_zero_touch_zero"] = zero_delta_loss(
                            zero_delta,
                            batch["action_mask"],
                            batch["future_mask"],
                        )
                        total_loss = total_loss + args.lambda_zero_zero * losses["loss_zero_touch_zero"]
                losses["loss"] = total_loss
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                record = {key: float(value.detach().cpu()) for key, value in losses.items() if key.startswith("loss")}
                record["step"] = step
                metrics_file.write(json.dumps(record) + "\n")
                if step % 100 == 0:
                    torch.save({"model": model.state_dict(), "step": step, "args": serializable_args(args)}, args.output_dir / "latest.pt")
                step += 1
                progress.update(1)
                if step >= args.max_steps:
                    break
        progress.close()
    torch.save({"model": model.state_dict(), "step": step, "args": serializable_args(args)}, args.output_dir / "latest.pt")


if __name__ == "__main__":
    main()
