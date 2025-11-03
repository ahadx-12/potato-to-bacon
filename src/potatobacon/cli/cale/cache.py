"""Disk-backed caches for CALE CLI workflows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

from ...cale.finance.docio import Doc, DocBlock
from ...cale.embed import LegalEmbedder

CACHE_VERSION = "v1"


def _stable_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _timestamp_ns(path: Path) -> int:
    stat = path.stat()
    return getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))


def _doc_to_payload(doc: Doc) -> dict[str, Any]:
    return {
        "src_path": doc.src_path,
        "doc_kind": doc.doc_kind,
        "blocks": [
            {
                "kind": block.kind,
                "text": block.text,
                "table": block.table,
                "meta": block.meta,
            }
            for block in doc.blocks
        ],
    }


def _doc_from_payload(payload: dict[str, Any]) -> Doc:
    blocks: List[DocBlock] = []
    for raw in payload.get("blocks", []):
        blocks.append(
            DocBlock(
                kind=str(raw.get("kind", "paragraph")),
                text=str(raw.get("text", "")),
                table=raw.get("table"),
                meta=dict(raw.get("meta", {})),
            )
        )
    return Doc(
        src_path=str(payload.get("src_path", "")),
        doc_kind=str(payload.get("doc_kind", "OTHER")),
        blocks=blocks,
    )


class CacheManager:
    """Helper that persists manifest, document and embedding caches."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        manifest_ttl_seconds: int = 6 * 60 * 60,
    ) -> None:
        self.root = Path(root) if root else Path("data/cache")
        self.manifest_ttl = int(max(600, manifest_ttl_seconds))
        self.manifest_dir = self.root / "manifest"
        self.docs_dir = self.root / "docs"
        self.embed_dir = self.root / "embeddings"
        for directory in (self.root, self.manifest_dir, self.docs_dir, self.embed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest caching
    # ------------------------------------------------------------------
    def manifest_key(
        self, tickers: Iterable[str], event_date: str, user_agent: str
    ) -> str:
        payload = {
            "tickers": sorted({ticker.upper() for ticker in tickers}),
            "event_date": event_date,
            "ua": user_agent,
            "version": CACHE_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return _stable_hash(blob)

    def load_manifest(
        self, key: str, *, read_cache: bool = True
    ) -> Optional[list[dict[str, Any]]]:
        if not read_cache:
            return None
        path = self.manifest_dir / f"{CACHE_VERSION}-{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            path.unlink(missing_ok=True)
            return None
        created = float(data.get("created_at", 0.0))
        if created and time.time() - created > self.manifest_ttl:
            path.unlink(missing_ok=True)
            return None
        entries = data.get("entries")
        if not isinstance(entries, list):
            return None
        return entries

    def save_manifest(self, key: str, entries: list[dict[str, Any]]) -> Path:
        path = self.manifest_dir / f"{CACHE_VERSION}-{key}.json"
        payload = {
            "created_at": time.time(),
            "entries": entries,
        }
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))
        return path

    # ------------------------------------------------------------------
    # Document caching
    # ------------------------------------------------------------------
    def _doc_cache_path(self, key: str) -> Path:
        return self.docs_dir / f"{CACHE_VERSION}-{key}.json"

    def _doc_key(self, src_path: Path) -> str:
        payload = json.dumps(
            {
                "path": str(src_path.resolve()),
                "mtime": _timestamp_ns(src_path),
                "size": src_path.stat().st_size,
                "version": CACHE_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return _stable_hash(payload)

    def load_doc(
        self, path: str | Path, *, read_cache: bool = True
    ) -> Optional[Doc]:
        src = Path(path)
        if not read_cache:
            return None
        key = self._doc_key(src)
        cache_path = self._doc_cache_path(key)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text())
        except Exception:
            cache_path.unlink(missing_ok=True)
            return None
        stored_path = payload.get("source_path")
        stored_mtime = payload.get("source_mtime_ns")
        stored_size = payload.get("source_size")
        if (
            stored_path != str(src.resolve())
            or stored_mtime != _timestamp_ns(src)
            or stored_size != src.stat().st_size
        ):
            cache_path.unlink(missing_ok=True)
            return None
        doc_payload = payload.get("doc")
        if not isinstance(doc_payload, dict):
            cache_path.unlink(missing_ok=True)
            return None
        try:
            return _doc_from_payload(doc_payload)
        except Exception:
            cache_path.unlink(missing_ok=True)
            return None

    def save_doc(self, path: str | Path, doc: Doc) -> Path:
        src = Path(path)
        key = self._doc_key(src)
        cache_path = self._doc_cache_path(key)
        payload = {
            "source_path": str(src.resolve()),
            "source_mtime_ns": _timestamp_ns(src),
            "source_size": src.stat().st_size,
            "doc": _doc_to_payload(doc),
        }
        cache_path.write_text(json.dumps(payload, sort_keys=True, indent=2))
        return cache_path

    # ------------------------------------------------------------------
    # Embedding caching
    # ------------------------------------------------------------------
    def _embed_key(self, text: str) -> str:
        normalized = text.strip().encode("utf-8")
        return _stable_hash(normalized)

    def load_embedding(self, text: str, *, read_cache: bool = True) -> Optional[np.ndarray]:
        if not read_cache:
            return None
        key = self._embed_key(text)
        path = self.embed_dir / f"{CACHE_VERSION}-{key}.npy"
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def save_embedding(self, text: str, vector: np.ndarray) -> Path:
        key = self._embed_key(text)
        path = self.embed_dir / f"{CACHE_VERSION}-{key}.npy"
        np.save(path, np.asarray(vector, dtype=np.float32))
        return path


@dataclass
class CachingLegalEmbedder:
    """Wrapper around :class:`LegalEmbedder` that persists embeddings to disk."""

    inner: LegalEmbedder
    cache: CacheManager
    read_cache: bool = True

    def embed_rule(self, rule: Any) -> np.ndarray:  # type: ignore[override]
        key_text = "|".join(
            [
                getattr(rule, "subject", ""),
                getattr(rule, "modality", ""),
                getattr(rule, "action", ""),
                getattr(rule, "text", ""),
            ]
        )
        cached = self.cache.load_embedding(key_text, read_cache=self.read_cache)
        if cached is not None:
            return cached
        vector = self.inner.embed_rule(rule)
        self.cache.save_embedding(key_text, vector)
        return vector

    def embed_phrase(self, phrase: str) -> np.ndarray:  # type: ignore[override]
        cached = self.cache.load_embedding(phrase, read_cache=self.read_cache)
        if cached is not None:
            return cached
        vector = self.inner.embed_phrase(phrase)
        self.cache.save_embedding(phrase, vector)
        return vector

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
