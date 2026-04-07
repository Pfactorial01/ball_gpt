#!/usr/bin/env python3
"""
Phase 3 (ball_gpt): train decoder-only GPT on char-level match corpus.

  source .venv/bin/activate
  PYTHONPATH=. python scripts/train.py

Checkpoints: data/checkpoints/ckpt.pt (best val loss)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import DataLoader

from ball_gpt.dataset import build_datasets
from ball_gpt.model import GPT, GPTConfig


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    if step < warmup_steps:
        return lr_max * float(step + 1) / float(warmup_steps)
    if step >= max_steps:
        return lr_min
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_min + coeff * (lr_max - lr_min)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> float:
    model.eval()
    loss_acc = 0.0
    n = 0
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        xb = xb.to(device)
        yb = yb.to(device)
        _, loss = model(xb, yb)
        loss_acc += loss.item()
        n += 1
    model.train()
    return loss_acc / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="ball_gpt: train nano-GPT on match corpus")
    ap.add_argument("--corpus", default=os.path.join(_ROOT, "data", "corpus", "pl2526.txt"))
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8, help="Must divide --n-embd")
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr-max", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=50, help="Val loss batches (subsample)")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--amp", action="store_true", help="Use torch.cuda.amp")
    ap.add_argument(
        "--checkpoint-dir",
        default=os.path.join(_ROOT, "data", "checkpoints"),
    )
    ap.add_argument("--resume", default=None, help="Path to ckpt.pt to warm-start weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Device: {device}")

    train_ds, val_ds, tok = build_datasets(
        args.corpus,
        block_size=args.block_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"Parameters: {model.get_num_params() / 1e6:.2f}M (non-embedding)")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr_max, betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "ckpt.pt")
    cfg_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "dropout": cfg.dropout,
            },
            f,
            indent=2,
        )

    best_val = float("inf")
    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    while step < args.max_steps:
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr_max, args.lr_min)
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb = xb.to(device)
        yb = yb.to(device)

        optim.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

        step += 1
        if step % 50 == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step}/{args.max_steps}  train_loss={loss.item():.4f}  lr={lr:.2e}  ({dt:.0f}s)")

        if step % args.eval_interval == 0 or step == args.max_steps:
            vloss = estimate_loss(model, val_loader, device, max_batches=args.eval_batches)
            print(f"  >> val_loss={vloss:.4f}")
            if vloss < best_val:
                best_val = vloss
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "val_loss": vloss,
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"  >> saved {ckpt_path} (best val so far)")

    print(f"Done. Best val loss ~ {best_val:.4f}. Checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
