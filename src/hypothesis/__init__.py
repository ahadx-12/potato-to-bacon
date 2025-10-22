from __future__ import annotations

from .strategies import text

__all__ = ["given", "strategies", "text"]


class strategies:  # pragma: no cover - simple namespace stub
    text = staticmethod(text)


def given(*strategies_):  # pragma: no cover - deterministic stub
    def decorator(func):
        def wrapper(*args, **kwargs):
            examples = [strategy.example() for strategy in strategies_]
            return func(*args, *examples, **kwargs)

        return wrapper

    return decorator
