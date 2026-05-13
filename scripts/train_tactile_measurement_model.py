#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitra.guidance.tactile_forward_model import TactileEncoder, TactileForwardModel
from vitra.guidance.tactile_losses import build_tactile_stats, masked_mse, touch_valid_step_mask
from vitra.guidance.tactile_replay_dataset import TactileReplayCacheDataset, move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OpenTouch tactile encoder and action-conditioned forward model.")
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--test_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--encoder_steps", type=int, default=30000)
    parser.add_argument("--forward_steps", type=int, default=60000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--stat_dim", type=int, default=8)
    parser.add_argument("--encoder_hidden_dim", type=int, default=128)
    parser.add_argument("--forward_hidden_dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--touch_scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def slice_from_edit(
    batch: dict[str, torch.Tensor],
    horizon: int,
    *,
    touch_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = []
    touch = []
    masks = []
    phase = []
    chunk_len = int(batch["a_target"].shape[1])
    for i, edit_idx in enumerate(batch["edit_start_idx"].reshape(-1).tolist()):
        k = max(0, min(int(edit_idx), chunk_len - 1))
        end = min(chunk_len, k + int(horizon))
        if end - k < int(horizon):
            k = max(0, end - int(horizon))
        actions.append(batch["a_target"][i, k:end])
        touch.append(batch["touch_pressure"][i, k:end] / float(touch_scale))
        masks.append(batch["touch_mask"][i, k:end])
        phase.append(batch["chunk_phase"][i, k:end])
    return torch.stack(actions), torch.stack(touch), torch.stack(masks), torch.stack(phase)


@torch.no_grad()
def evaluate(
    encoder: TactileEncoder,
    forward_model: TactileForwardModel,
    loader: DataLoader,
    *,
    device: str,
    horizon: int,
    touch_scale: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    encoder.eval()
    forward_model.eval()
    totals = {"encoder_stats_sse": 0.0, "forward_embed_sse": 0.0, "forward_stats_sse": 0.0, "count": 0.0}
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        action, touch, touch_mask, phase = slice_from_edit(batch, horizon, touch_scale=touch_scale)
        stats = build_tactile_stats(touch, touch_mask)
        valid = touch_valid_step_mask(touch_mask)
        target = encoder(touch, touch_mask)
        pred = forward_model(batch["current_state"], action, phase)
        valid_count = float(valid.sum().detach().cpu())
        totals["encoder_stats_sse"] += float(masked_mse(target["stats"], stats, valid).detach().cpu()) * max(valid_count, 1.0)
        totals["forward_embed_sse"] += float(masked_mse(pred["embedding"], target["embedding"], valid).detach().cpu()) * max(valid_count, 1.0)
        totals["forward_stats_sse"] += float(masked_mse(pred["stats"], stats, valid).detach().cpu()) * max(valid_count, 1.0)
        totals["count"] += max(valid_count, 1.0)
    count = max(totals["count"], 1.0)
    return {
        "encoder_stats_mse": totals["encoder_stats_sse"] / count,
        "forward_embed_mse": totals["forward_embed_sse"] / count,
        "forward_stats_mse": totals["forward_stats_sse"] / count,
    }


def save_checkpoint(
    output_dir: Path,
    name: str,
    encoder: TactileEncoder,
    forward_model: TactileForwardModel,
    args: argparse.Namespace,
    metrics: dict[str, float],
    step: int,
) -> None:
    path = output_dir / name
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "forward_model": forward_model.state_dict(),
            "args": serializable_args(args),
            "metrics": metrics,
            "step": int(step),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = TactileReplayCacheDataset(args.train_cache, max_samples=args.max_train_samples)
    test_ds = TactileReplayCacheDataset(args.test_cache, max_samples=args.max_test_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    train_iter = cycle(train_loader)

    encoder = TactileEncoder(
        embed_dim=args.embed_dim,
        stat_dim=args.stat_dim,
        hidden_dim=args.encoder_hidden_dim,
    ).to(args.device)
    forward_model = TactileForwardModel(
        action_dim=192,
        state_dim=212,
        embed_dim=args.embed_dim,
        stat_dim=args.stat_dim,
        hidden_dim=args.forward_hidden_dim,
    ).to(args.device)
    enc_opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    fwd_opt = torch.optim.AdamW(forward_model.parameters(), lr=args.lr, weight_decay=1e-4)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_score = float("inf")

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        encoder.train()
        progress = tqdm(range(1, args.encoder_steps + 1), desc="train tactile encoder")
        for step in progress:
            batch = move_batch_to_device(next(train_iter), args.device)
            _, touch, touch_mask, _ = slice_from_edit(batch, args.horizon, touch_scale=args.touch_scale)
            stats = build_tactile_stats(touch, touch_mask)
            valid = touch_valid_step_mask(touch_mask)
            pred = encoder(touch, touch_mask)
            loss = masked_mse(pred["stats"], stats, valid)
            enc_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            enc_opt.step()
            if step % args.eval_every == 0 or step == args.encoder_steps:
                metrics = evaluate(
                    encoder,
                    forward_model,
                    test_loader,
                    device=args.device,
                    horizon=args.horizon,
                    touch_scale=args.touch_scale,
                    max_batches=4,
                )
                row = {"phase": "encoder", "step": step, "loss": float(loss.detach().cpu()), **metrics}
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
                progress.set_postfix(loss=float(loss.detach().cpu()))

        encoder.eval()
        progress = tqdm(range(1, args.forward_steps + 1), desc="train tactile forward")
        for step in progress:
            batch = move_batch_to_device(next(train_iter), args.device)
            action, touch, touch_mask, phase = slice_from_edit(batch, args.horizon, touch_scale=args.touch_scale)
            stats = build_tactile_stats(touch, touch_mask)
            valid = touch_valid_step_mask(touch_mask)
            with torch.no_grad():
                target = encoder(touch, touch_mask)
            pred = forward_model(batch["current_state"], action, phase)
            embed_loss = masked_mse(pred["embedding"], target["embedding"], valid)
            stats_loss = masked_mse(pred["stats"], stats, valid)
            loss = embed_loss + 0.25 * stats_loss
            fwd_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forward_model.parameters(), 1.0)
            fwd_opt.step()
            if step % args.eval_every == 0 or step == args.forward_steps:
                metrics = evaluate(
                    encoder,
                    forward_model,
                    test_loader,
                    device=args.device,
                    horizon=args.horizon,
                    touch_scale=args.touch_scale,
                    max_batches=None,
                )
                row = {
                    "phase": "forward",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "embed_loss": float(embed_loss.detach().cpu()),
                    "stats_loss": float(stats_loss.detach().cpu()),
                    **metrics,
                }
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
                score = metrics["forward_embed_mse"] + 0.25 * metrics["forward_stats_mse"]
                if score < best_score:
                    best_score = score
                    save_checkpoint(args.output_dir, "best.pt", encoder, forward_model, args, metrics, step)
                progress.set_postfix(loss=float(loss.detach().cpu()), score=score)

    final_metrics = evaluate(
        encoder,
        forward_model,
        test_loader,
        device=args.device,
        horizon=args.horizon,
        touch_scale=args.touch_scale,
        max_batches=None,
    )
    save_checkpoint(args.output_dir, "last.pt", encoder, forward_model, args, final_metrics, args.forward_steps)
    summary = {
        "args": serializable_args(args),
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "final_metrics": final_metrics,
        "best_score": best_score,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "last_checkpoint": str(args.output_dir / "last.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
