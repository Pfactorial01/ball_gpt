"""ball_gpt — PyTorch Dataset: next-character prediction on contiguous chunks of text."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ball_gpt.tokenizer import CharTokenizer


def load_corpus_lines(path: str | Path) -> list[str]:
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def prepare_train_val(
    lines: list[str],
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[str, str]:
    """Shuffle match lines, split into train/val *text* blobs (newline-separated)."""
    rng = random.Random(seed)
    order = list(lines)
    rng.shuffle(order)
    n_train = int(len(order) * train_ratio)
    train_lines = order[:n_train]
    val_lines = order[n_train:]
    train_text = "\n".join(train_lines) + "\n"
    val_text = "\n".join(val_lines) + "\n"
    return train_text, val_text


class CharLMDataset(Dataset):
    """
    Contiguous sliding windows over encoded text: predicts next character at each position.

    For each index ``i``, returns ``x = ids[i:i+block_size]``, ``y = ids[i+1:i+block_size+1]``.
    """

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        block_size: int,
    ):
        if len(text) < block_size + 1:
            raise ValueError(
                f"Text length {len(text)} must be >= block_size + 1 ({block_size + 1})"
            )
        self.block_size = block_size
        self.ids = tokenizer.encode(text)
        if len(self.ids) != len(text):
            raise RuntimeError("Tokenizer must be round-trip safe for this text")
        self._len = len(self.ids) - block_size

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_datasets(
    corpus_path: str | Path,
    *,
    block_size: int = 512,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[CharLMDataset, CharLMDataset, CharTokenizer]:
    """
    Load corpus file, build tokenizer on **full** text (so val chars are in vocab),
    return train and validation CharLMDatasets.
    """
    lines = load_corpus_lines(corpus_path)
    full_text = "\n".join(lines) + "\n"
    tokenizer = CharTokenizer(full_text)
    train_text, val_text = prepare_train_val(lines, train_ratio=train_ratio, seed=seed)
    train_ds = CharLMDataset(train_text, tokenizer, block_size)
    val_ds = CharLMDataset(val_text, tokenizer, block_size)
    return train_ds, val_ds, tokenizer
