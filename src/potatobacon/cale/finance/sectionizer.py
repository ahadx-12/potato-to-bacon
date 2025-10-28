"""Section selection heuristics for covenant extraction tests."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .docio import Doc, DocBlock


@dataclass
class Section:
    """Candidate covenant section derived from headers and nearby blocks."""

    key: str
    title: str
    score: float
    start_block: int
    end_block: int
    anchor: Optional[str]
    doc_kind: str


class _SectionizerConfig:
    def __init__(self, cfg: Dict[str, object]) -> None:
        section_cfg = cfg.get("sectionizer", {}) if isinstance(cfg, dict) else {}
        if not isinstance(section_cfg, dict):
            section_cfg = {}
        self.positive = [_compile(pattern) for pattern in section_cfg.get("positive_keywords", [])]
        self.metric = [_compile(pattern) for pattern in section_cfg.get("metric_keywords", [])]
        self.negative = [_compile(pattern) for pattern in section_cfg.get("negative_keywords", [])]
        self.window_sizes = section_cfg.get("window_sizes", [3, 8, 20])
        weights = section_cfg.get("weights", {}) if isinstance(section_cfg.get("weights"), dict) else {}
        self.w_header = float(weights.get("header_match", 1.0))
        self.w_metric = float(weights.get("metric_density", 0.6))
        self.w_negative = float(weights.get("negative_penalty", 0.9))
        doc_boosts = weights.get("doc_kind_boost", {}) if isinstance(weights.get("doc_kind_boost"), dict) else {}
        self.doc_boosts = {str(key): float(value) for key, value in doc_boosts.items()}
        self.min_score = float(section_cfg.get("min_score", 0.9))


def _compile(pattern: object) -> re.Pattern[str]:
    if isinstance(pattern, re.Pattern):
        return pattern
    return re.compile(str(pattern), re.I)


def _count(patterns: Iterable[re.Pattern[str]], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+", text)))


def _score_header(block: DocBlock, doc: Doc, window: List[DocBlock], cfg: _SectionizerConfig) -> float:
    header_hits = _count(cfg.positive, block.text)
    header_score = header_hits / _token_count(block.text)

    window_text = " ".join(part.text for part in window if part.text)
    metric_hits = _count(cfg.metric, window_text)
    metric_score = metric_hits / _token_count(window_text) if window_text else 0.0
    negative_hits = _count(cfg.negative, window_text)
    negative_score = negative_hits / _token_count(window_text) if window_text else 0.0

    doc_boost = cfg.doc_boosts.get(doc.doc_kind, cfg.doc_boosts.get("default", 0.0))

    score = (
        cfg.w_header * header_score
        + cfg.w_metric * metric_score
        - cfg.w_negative * negative_score
        + doc_boost
    )
    return score


def _hash_key(src_path: str, title: str, idx: int) -> str:
    seed = f"{src_path}|{title}|{idx}".encode("utf-8", "ignore")
    return hashlib.blake2b(seed, digest_size=12, person=b"cale-sec2").hexdigest()


def find_sections(doc: Doc, cfg_dict: Optional[Dict[str, object]] = None) -> List[Section]:
    """Return ranked sections using configuration driven heuristics."""

    if cfg_dict is None:
        from tools.finance_extract import CFG as _CFG  # lazy import to avoid cycle

        cfg_dict = _CFG

    cfg = _SectionizerConfig(cfg_dict)
    sections: List[Section] = []
    for idx, block in enumerate(doc.blocks):
        if block.kind != "header" or not block.text.strip():
            continue
        best_score = -math.inf
        best_end = idx
        window_blocks: List[DocBlock] = []
        for window_size in cfg.window_sizes:
            end_idx = min(len(doc.blocks), idx + 1 + window_size)
            window = doc.blocks[idx + 1 : end_idx]
            score = _score_header(block, doc, window, cfg)
            if score > best_score:
                best_score = score
                best_end = end_idx
                window_blocks = window
        if best_score < cfg.min_score:
            continue
        anchor = block.meta.get("html_id") or block.meta.get("html_name")
        key = _hash_key(doc.src_path, block.text, idx)
        sections.append(
            Section(
                key=key,
                title=block.text,
                score=float(best_score),
                start_block=idx,
                end_block=best_end,
                anchor=anchor,
                doc_kind=doc.doc_kind,
            )
        )
    sections.sort(key=lambda sec: (-sec.score, sec.start_block))
    return sections


__all__ = ["Section", "find_sections"]
