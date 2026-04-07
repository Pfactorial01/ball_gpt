#!/usr/bin/env python3
"""
ball_gpt Phase 2: build CharTokenizer + train/val CharLMDataset from data/corpus/pl2526.txt.

Saves vocabulary to data/prepared/vocab.json for training/inference later.

Run from repo root:
  PYTHONPATH=. python3 scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ball_gpt.dataset import build_datasets
from torch.utils.data import DataLoader


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare LM dataset (Phase 2)")
    ap.add_argument(
        "--corpus",
        default=os.path.join(_ROOT, "data", "corpus", "pl2526.txt"),
        help="Path to pl2526.txt",
    )
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--vocab-out",
        default=os.path.join(_ROOT, "data", "prepared", "vocab.json"),
        help="Where to save CharTokenizer JSON",
    )
    ap.add_argument("--batch-size", type=int, default=16, help="Sanity-check DataLoader only")
    args = ap.parse_args()

    train_ds, val_ds, tok = build_datasets(
        args.corpus,
        block_size=args.block_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.vocab_out), exist_ok=True)
    tok.save(args.vocab_out)

    print("Corpus:", os.path.abspath(args.corpus))
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Block size: {args.block_size}")
    print(f"Train samples (chunks): {len(train_ds)}")
    print(f"Val samples (chunks):   {len(val_ds)}")
    print(f"Saved vocab to {args.vocab_out}")

    # Quick DataLoader smoke test
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    x, y = next(iter(train_loader))
    print(f"Batch x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")
    assert x.shape == y.shape
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
