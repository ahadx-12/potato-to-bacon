"""API router serving tax-law analytics for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/law/tax", tags=["tax-law"])

OUT_DIR = Path("out")
SECTION_METRICS = OUT_DIR / "section_metrics.jsonl"
PAIRS_SCORED = OUT_DIR / "pairs_scored.jsonl"
GRAPH_PATH = OUT_DIR / "graph.json"
TIME_SERIES = OUT_DIR / "time_series.json"


def _load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    results: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise HTTPException(status_code=500, detail=f"Malformed JSON at {path}") from exc


@router.get("/sections")
def list_sections(page: int = Query(1, ge=1), page_size: int = Query(50, le=200)) -> Dict[str, object]:
    """Return a paginated list of section-level metrics."""

    data = _load_jsonl(SECTION_METRICS)
    total = len(data)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "results": data[start:end],
    }


@router.get("/pairs")
def list_pairs(limit: int = Query(100, ge=1, le=1000)) -> Dict[str, object]:
    """Return the top-N scored statute/regulation pairs."""

    data = _load_jsonl(PAIRS_SCORED, limit=limit)
    return {
        "count": len(data),
        "results": data,
    }


@router.get("/graph")
def graph() -> Dict[str, object]:
    """Return the cross-reference graph if present."""

    payload = _load_json(GRAPH_PATH)
    if not payload:
        payload = {"nodes": [], "edges": []}
    return payload


@router.get("/summary")
def summary() -> Dict[str, object]:
    """Return roll-up metrics for the dashboard."""

    sections = _load_jsonl(SECTION_METRICS)
    pairs = _load_jsonl(PAIRS_SCORED, limit=2000)
    timeseries = _load_json(TIME_SERIES)
    top_sections = sorted(sections, key=lambda item: item.get("policy_flaw", 0), reverse=True)[:10]
    return {
        "sections_total": len(sections),
        "pairs_total": len(pairs),
        "top_sections": top_sections,
        "time_series": timeseries,
    }
