"""Utilities for reading and normalizing finance documents for CALE tests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


BlockKind = Literal["paragraph", "table", "header"]
DocKind = Literal["10-K", "10-Q", "INDENTURE", "CREDIT_AGREEMENT", "OTHER"]


@dataclass
class DocBlock:
    """Normalized block of text or table extracted from a filing."""

    kind: BlockKind
    text: str = ""
    table: Optional[List[List[str]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Doc:
    """Container describing a parsed document."""

    src_path: str
    doc_kind: DocKind
    blocks: List[DocBlock]


_HEADER_RE = re.compile(r"^\s*(Article|Section)\s+[IVXLC\d][\.-]?\d*(?:\([a-z]\))?", re.I)
_ASCII_TABLE_RE = re.compile(r"[|+].*[|+]|\s{2,}\S+\s{2,}\S+")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_ALLCAPS_RE = re.compile(r"^[A-Z0-9 .,&'()\-]{5,}$")
_DOC_KIND_PATTERNS = [
    (re.compile(r"(?i)indenture"), "INDENTURE"),
    (re.compile(r"(?i)credit agreement"), "CREDIT_AGREEMENT"),
]
_FORM_INFER = [
    (re.compile(r"10-k", re.I), "10-K"),
    (re.compile(r"10-q", re.I), "10-Q"),
]


class _HTMLDocParser(HTMLParser):
    """Small HTML parser that collects blocks and metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks: List[DocBlock] = []
        self._stack: List[str] = []
        self._current_text: List[str] = []
        self._current_block: Optional[DocBlock] = None
        self._current_cell_text: List[str] = []
        self._table_rows: List[List[str]] = []
        self._row: List[str] = []
        self._in_header: Optional[str] = None
        self._header_meta: Dict[str, Any] = {}

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:  # type: ignore[override]
        attr_dict = {key.lower(): val for key, val in attrs}
        self._stack.append(tag)
        if tag in {"p", "div", "li"}:
            self._flush_text()
            self._current_block = DocBlock("paragraph", meta={"html_tag": tag, **attr_dict})
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_text()
            self._in_header = tag
            meta = {"html_tag": tag}
            if "id" in attr_dict:
                meta["html_id"] = attr_dict["id"]
            if "name" in attr_dict:
                meta["html_name"] = attr_dict["name"]
            self._header_meta = meta
            self._current_block = DocBlock("header", meta=meta)
        elif tag == "table":
            self._flush_text()
            self._table_rows = []
            self._current_block = DocBlock("table", table=[], meta={**attr_dict})
        elif tag == "tr":
            if self._current_block and self._current_block.kind == "table":
                self._row = []
        elif tag in {"td", "th"}:
            self._current_cell_text = []
        elif tag == "br":
            self._current_text.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"p", "div", "li"}:
            self._finalize_text_block("paragraph")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._in_header == tag:
            self._finalize_text_block("header")
            self._in_header = None
            self._header_meta = {}
        elif tag == "table":
            if self._current_block and self._current_block.kind == "table":
                self._current_block.table = [row[:] for row in self._table_rows]
                self.blocks.append(self._current_block)
            self._current_block = None
            self._table_rows = []
        elif tag == "tr":
            if self._current_block and self._current_block.kind == "table":
                self._table_rows.append(self._row[:])
                self._row = []
        elif tag in {"td", "th"}:
            text = _normalize_text("".join(self._current_cell_text))
            self._row.append(text)
            self._current_cell_text = []
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._current_block and self._current_block.kind == "table" and self._stack and self._stack[-1] in {"td", "th"}:
            self._current_cell_text.append(data)
        else:
            self._current_text.append(data)

    def _flush_text(self) -> None:
        text = _normalize_text("".join(self._current_text))
        if text and self._current_block and self._current_block.kind in {"paragraph", "header"}:
            self._current_block.text = text
            self.blocks.append(self._current_block)
            self._current_block = None
        self._current_text = []

    def _finalize_text_block(self, kind: BlockKind) -> None:
        text = _normalize_text("".join(self._current_text))
        if not text:
            self._current_text = []
            self._current_block = None
            return
        if self._current_block:
            self._current_block.kind = kind
            self._current_block.text = text
            if kind == "header" and self._header_meta:
                self._current_block.meta.update(self._header_meta)
            self.blocks.append(self._current_block)
        else:
            self.blocks.append(DocBlock(kind, text=text))
        self._current_text = []
        self._current_block = None


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\xa0", " "))
    return text.strip()


def _infer_doc_kind(path: Path, text: str) -> DocKind:
    for pattern, doc_kind in _DOC_KIND_PATTERNS:
        if pattern.search(text):
            return doc_kind
    for pattern, doc_kind in _FORM_INFER:
        if pattern.search(path.name):
            return doc_kind  # type: ignore[return-value]
    return "OTHER"


def _split_ascii_table(lines: List[str]) -> List[List[str]]:
    cleaned = [line.rstrip("\n") for line in lines]
    if not cleaned:
        return []
    if any("|" in line for line in cleaned):
        rows = []
        for line in cleaned:
            parts = [part.strip() for part in line.strip("|").split("|")]
            rows.append(parts)
        return rows
    positions: List[int] = []
    for line in cleaned:
        for match in _MULTI_SPACE_RE.finditer(line):
            start = match.start()
            if start not in positions:
                positions.append(start)
    positions.sort()
    rows = []
    for line in cleaned:
        cols: List[str] = []
        last = 0
        for pos in positions:
            cols.append(line[last:pos].strip())
            last = pos
        cols.append(line[last:].strip())
        rows.append([col for col in cols if col])
    return rows


def _parse_ascii_document(text: str) -> List[DocBlock]:
    blocks: List[DocBlock] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if _ASCII_TABLE_RE.search(line):
            start = i
            table_lines = [lines[i]]
            i += 1
            while i < len(lines) and (lines[i].strip() and _ASCII_TABLE_RE.search(lines[i])):
                table_lines.append(lines[i])
                i += 1
            table = _split_ascii_table(table_lines)
            blocks.append(DocBlock("table", table=table, meta={"line_start": start + 1, "line_end": i}))
            continue
        if _ALLCAPS_RE.match(line) or _HEADER_RE.match(line):
            blocks.append(
                DocBlock(
                    "header",
                    text=_normalize_text(line),
                    meta={"line_no": i + 1},
                )
            )
            i += 1
            continue
        para_lines = [lines[i]]
        i += 1
        while i < len(lines) and lines[i].strip() and not _ALLCAPS_RE.match(lines[i]) and not _ASCII_TABLE_RE.search(lines[i]):
            para_lines.append(lines[i])
            i += 1
        paragraph = _normalize_text(" ".join(para_lines))
        blocks.append(DocBlock("paragraph", text=paragraph, meta={"line_no": i}))
    return blocks


def load_doc(path: str | Path) -> Doc:
    """Load a filing from ``path`` into a :class:`Doc` structure."""

    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() in {".html", ".htm"}:
        parser = _HTMLDocParser()
        parser.feed(raw)
        text_preview = " ".join(block.text for block in parser.blocks if block.text)[:5000]
        doc_kind = _infer_doc_kind(p, text_preview)
        return Doc(src_path=str(p), doc_kind=doc_kind, blocks=parser.blocks)
    doc_kind = _infer_doc_kind(p, raw)
    blocks = _parse_ascii_document(raw)
    return Doc(src_path=str(p), doc_kind=doc_kind, blocks=blocks)


__all__ = ["Doc", "DocBlock", "load_doc"]
