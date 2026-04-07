"""ball_gpt — character-level tokenizer for match-line corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharTokenizer:
    """Maps each distinct Unicode character in the training text to an integer id."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Cannot build tokenizer from empty text")
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stoi": self.stoi}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = payload["stoi"]
        obj = object.__new__(cls)
        obj.stoi = stoi
        obj.itos = {int(i): ch for ch, i in stoi.items()}
        obj.vocab_size = len(obj.stoi)
        return obj
