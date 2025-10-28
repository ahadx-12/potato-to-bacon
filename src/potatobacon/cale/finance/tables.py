"""Table flattening helpers for finance covenant extraction tests."""

from __future__ import annotations

import re
from typing import Iterator, List, Sequence, Tuple, TypedDict

from .docio import DocBlock


class TableCellMeta(TypedDict):
    row_header: str
    col_header: str
    row_index: int
    col_index: int


_HEADER_UNIT_RE = re.compile(r"\b(?:x|%|percent|million|billion|mm|bn|\$)\b", re.I)
_ALLCAPS_RE = re.compile(r"^[A-Z0-9 .,&'()\-]{3,}$")
_BORDER_RE = re.compile(r"^[+\-=\s]+$")


def _is_header_row(row: Sequence[str]) -> bool:
    score = 0
    for cell in row:
        if not cell:
            continue
        if _HEADER_UNIT_RE.search(cell):
            score += 1
        if _ALLCAPS_RE.match(cell.strip()):
            score += 1
    return score >= max(1, len([c for c in row if c.strip()]) // 2)


def _normalise(cell: str) -> str:
    cell = re.sub(r"\s+", " ", cell or "").strip()
    return cell


def _clean_rows(rows: List[List[str]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for row in rows:
        if not any(cell.strip() for cell in row):
            continue
        joined = "".join(row).strip()
        if _BORDER_RE.match(joined):
            continue
        cleaned.append(row)
    return cleaned


def _extract_headers(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    if not rows:
        return [], []
    rows = _clean_rows(rows)
    if not rows:
        return [], []
    header: List[str] = [_normalise(cell) for cell in rows[0]]
    body: List[List[str]] = [rows[idx] for idx in range(1, len(rows)) if any(rows[idx])]
    if not _is_header_row(rows[0]) and body:
        header = [_normalise(cell) for cell in rows[0]]
    return header, body


def flatten(block: DocBlock) -> Iterator[Tuple[str, TableCellMeta]]:
    """Yield sentence-like strings from a table :class:`DocBlock`."""

    if block.kind != "table" or not block.table:
        return iter(())
    rows = [[_normalise(cell) for cell in row] for row in block.table]
    header, body = _extract_headers(rows)
    if not header:
        return iter(())
    col_headers = header
    for row_index, row in enumerate(body, start=1):
        if not any(cell.strip() for cell in row):
            continue
        row_header = row[0].strip() if row and row[0].strip() else f"Row {row_index}"
        for col_index, cell in enumerate(row[1:], start=1):
            col_header = col_headers[col_index] if col_index < len(col_headers) else f"Column {col_index}"
            value = cell.strip()
            if not value:
                continue
            sentence = f"{row_header} {col_header}: {value}".strip()
            meta: TableCellMeta = {
                "row_header": row_header,
                "col_header": col_header,
                "row_index": row_index,
                "col_index": col_index,
            }
            yield sentence, meta


__all__ = ["flatten", "TableCellMeta"]
