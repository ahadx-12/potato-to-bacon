"""Anchor and numbering utilities for finance covenant extraction tests."""

from __future__ import annotations

import re
from typing import Dict, Optional

from .docio import Doc

_SECTION_PATTERNS = [
    re.compile(r"^(Article|Section)\s+[IVXLC\d][\.-]?\d*(?:\([a-z]\))?", re.I),
    re.compile(r"^\s*\d+\.\d+(?:\([a-z]\))?", re.I),
]


def _extract_section_num(title: str) -> Optional[str]:
    for pattern in _SECTION_PATTERNS:
        match = pattern.match(title.strip())
        if match:
            return match.group(0).strip()
    return None


def _ensure_cache(doc: Doc) -> None:
    if hasattr(doc, "_authority_cache"):
        return
    block_info = []
    current_title = ""
    current_anchor = None
    current_num = None
    counter = 1
    for idx, block in enumerate(doc.blocks):
        if block.kind == "header" and block.text.strip():
            current_title = block.text.strip()
            current_anchor = block.meta.get("html_id") or block.meta.get("html_name") or f"sec-{counter:04d}"
            current_num = _extract_section_num(current_title)
            counter += 1
        block_info.append(
            {
                "anchor": current_anchor or f"sec-{counter:04d}",
                "section_title": current_title,
                "section_num": current_num,
            }
        )
    doc._authority_cache = block_info  # type: ignore[attr-defined]


def link_block(doc: Doc, block_idx: int) -> Dict[str, Optional[str]]:
    """Return anchor metadata for ``block_idx`` within ``doc``."""

    _ensure_cache(doc)
    block_idx = max(0, min(block_idx, len(doc.blocks) - 1))
    return dict(doc._authority_cache[block_idx])  # type: ignore[attr-defined]


def link_range(doc: Doc, start_block: int, end_block: int) -> Dict[str, Optional[str]]:
    """Return anchor metadata for a block range."""

    _ensure_cache(doc)
    start_block = max(0, min(start_block, len(doc.blocks) - 1))
    return dict(doc._authority_cache[start_block])  # type: ignore[attr-defined]


__all__ = ["link_block", "link_range"]
