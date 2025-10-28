"""Sentence deduplication helpers used in the finance pipeline tests."""

from __future__ import annotations

import collections
import hashlib
import re
from typing import Deque, Iterable, List, Optional, Set

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_MAX_CACHE = 50000
_SHINGLE = 5


def _normalise(sentence: str) -> List[str]:
    lowered = sentence.lower()
    lowered = _NUMBER_RE.sub("<num>", lowered)
    return _TOKEN_RE.findall(lowered)


def _shingles(tokens: Iterable[str]) -> List[str]:
    tokens = list(tokens)
    if len(tokens) < _SHINGLE:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[idx : idx + _SHINGLE]) for idx in range(len(tokens) - _SHINGLE + 1)]


def is_duplicate(sentence: str, cache: Set[str], order: Optional[Deque[str]] = None) -> bool:
    """Return ``True`` if *sentence* already appears in *cache* based on shingles."""

    tokens = _normalise(sentence)
    shingles = _shingles(tokens)
    if not shingles:
        return False
    matches = 0
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=16, person=b"cale-sec2").hexdigest()
        if digest in cache:
            matches += 1
    ratio = matches / len(shingles)
    if matches >= 2 and ratio >= 0.6:
        return True
    if order is None:
        order = collections.deque()
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=16, person=b"cale-sec2").hexdigest()
        cache.add(digest)
        order.append(digest)
    while len(cache) > _MAX_CACHE and order:
        oldest = order.popleft()
        cache.discard(oldest)
    return False


__all__ = ["is_duplicate"]
