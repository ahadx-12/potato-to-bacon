#!/usr/bin/env python3
"""Enrich the SEC manifest with risk labels and quality flags."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DISTRESSED_WATCHLIST: Sequence[str] = ("CVNA", "UPST", "KSS", "CCL", "RIVN", "AA")
CONTROL_WATCHLIST: Sequence[str] = ("AAPL", "MSFT", "JNJ", "PG", "COST", "ADBE")

DEFAULT_LEVERAGE_THRESHOLD = 4.0
MANIFEST_PATH = Path("reports/realworld/manifest.csv")
MANIFEST_HEADERS: Sequence[str] = (
    "ticker",
    "form",
    "filed",
    "path",
    "accession",
    "cik",
    "md5",
    "label",
    "quality_flag",
)


def _load_leverage_data(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    leverage: Dict[str, float] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = row.get("ticker")
            metric = row.get("net_debt_to_ebitda") or row.get("leverage")
            if not ticker or metric is None:
                continue
            try:
                leverage[ticker.upper()] = float(metric)
            except ValueError:
                continue
    return leverage


def _iter_manifest_rows(manifest_path: Path) -> Iterable[Dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {header: row.get(header, "") for header in MANIFEST_HEADERS}


def _apply_quality(row: Dict[str, str], reason: str) -> None:
    existing = [flag for flag in (row.get("quality_flag") or "").split(";") if flag]
    if reason and reason not in existing:
        existing.append(reason)
    row["quality_flag"] = ";".join(existing) if existing else ""


def _classify_row(
    row: Dict[str, str], leverage: Dict[str, float], threshold: float
) -> Dict[str, str]:
    ticker = (row.get("ticker") or "").upper()
    current_label = row.get("label", "")
    reasons: List[str] = []

    if ticker in DISTRESSED_WATCHLIST:
        row["label"] = "distressed"
        reasons.append("watchlist")
    elif ticker in CONTROL_WATCHLIST:
        row["label"] = "control"
        reasons.append("watchlist")
    elif ticker in leverage:
        if leverage[ticker] >= threshold:
            row["label"] = "distressed"
            reasons.append("leverage-high")
        else:
            row["label"] = "control"
            reasons.append("leverage-low")
    elif not current_label:
        row["label"] = "unknown"
        reasons.append("insufficient-data")

    for reason in reasons:
        _apply_quality(row, reason)

    if not reasons and current_label:
        _apply_quality(row, "preserved-label")

    if not row.get("quality_flag"):
        row["quality_flag"] = "unreviewed"

    return row


def update_manifest(manifest_path: Path, leverage_csv: Path | None, threshold: float) -> Path:
    leverage = _load_leverage_data(leverage_csv) if leverage_csv else {}
    rows = [_classify_row(dict(row), leverage, threshold) for row in _iter_manifest_rows(manifest_path)]

    if not rows:
        return manifest_path

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_HEADERS))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to the manifest CSV to enrich.",
    )
    parser.add_argument(
        "--leverage",
        type=Path,
        default=None,
        help="Optional CSV containing leverage metrics (ticker, net_debt_to_ebitda).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_LEVERAGE_THRESHOLD,
        help="Threshold above which leverage indicates distress.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path: Path = args.manifest
    leverage_csv: Path | None = args.leverage
    updated = update_manifest(manifest_path, leverage_csv, args.threshold)
    print(str(updated.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
