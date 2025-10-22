from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _TextStrategy:  # pragma: no cover - deterministic stub
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    min_size: int = 0
    max_size: int | None = None

    def example(self) -> str:
        if not self.alphabet:
            base = "a"
        else:
            base = self.alphabet[0]
        length = max(self.min_size, 1)
        if self.max_size is not None:
            length = min(length, self.max_size or length)
        return base * length


def text(alphabet: str = "abcdefghijklmnopqrstuvwxyz", min_size: int = 0, max_size: int | None = None) -> _TextStrategy:
    return _TextStrategy(alphabet=alphabet, min_size=min_size, max_size=max_size)
