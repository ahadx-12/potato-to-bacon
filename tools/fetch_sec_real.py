#!/usr/bin/env python3
"""Download a small cohort of real SEC filings for CALE smoke tests."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from sec_edgar_downloader import Downloader
from tqdm import tqdm

DISTRESSED: Sequence[str] = ("CVNA", "UPST", "KSS", "CCL", "RIVN", "AA")
CONTROL: Sequence[str] = ("AAPL", "MSFT", "JNJ", "PG", "COST", "ADBE")

VALID_SUFFIXES = {".htm", ".html", ".txt"}


def _iter_downloaded_forms(root: Path, ticker: str, form: str) -> Iterable[Tuple[Path, Path]]:
    base = root / "sec-edgar-filings" / ticker.upper() / form
    if not base.exists():
        return []
    rows: List[Tuple[Path, Path]] = []
    for filing_dir in base.iterdir():
        if not filing_dir.is_dir():
            continue
        for file_path in filing_dir.iterdir():
            if file_path.suffix.lower() not in VALID_SUFFIXES:
                continue
            rows.append((filing_dir, file_path))
    return sorted(rows, key=lambda item: item[1].stat().st_mtime)


def _filed_date(filing_dir: Path, file_path: Path) -> str:
    stem = filing_dir.name.split("_")[0]
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(stem, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")


def _try_download(dl: Downloader, ticker: str, form: str) -> None:
    try:
        dl.get(form, ticker, limit=1)
    except Exception:
        # Downloader surfaces HTTP errors as exceptions; we treat them as misses.
        return


def _collect_rows(out_dir: Path, tickers: Sequence[str]) -> List[List[str]]:
    dl = Downloader(
        company_name="CALE-Research",
        email_address="cale@example.com",
        download_folder=str(out_dir),
    )
    rows: List[List[str]] = []
    for ticker in tqdm(tickers, desc="tickers", leave=False):
        for form in ("10-Q", "10-K"):
            _try_download(dl, ticker, form)
            candidates = list(_iter_downloaded_forms(out_dir, ticker, form))
            if not candidates:
                continue
            filing_dir, file_path = candidates[-1]
            filed = _filed_date(filing_dir, file_path)
            rows.append([ticker.upper(), form, filed, str(file_path.resolve())])
            break
    return rows


def main() -> int:
    output_root = Path("data/sec_real")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[List[str]] = []
    manifest_rows.extend(_collect_rows(output_root, DISTRESSED))
    manifest_rows.extend(_collect_rows(output_root, CONTROL))

    manifest_path = Path("reports/realworld/manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "form", "filed", "path"])
        writer.writerows(manifest_rows)

    print(str(manifest_path.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
