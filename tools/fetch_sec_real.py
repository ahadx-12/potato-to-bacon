#!/usr/bin/env python3
"""Download a cohort of real SEC filings for CALE smoke tests.

This helper now keeps a rolling manifest of previously-downloaded filings so
we can compare longitudinal changes in the source documents. The manifest is
augmented with richer metadata for each accession.
"""

from __future__ import annotations

import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from sec_edgar_downloader import Downloader
from tqdm import tqdm

DISTRESSED: Sequence[str] = ("CVNA", "UPST", "KSS", "CCL", "RIVN", "AA")
CONTROL: Sequence[str] = ("AAPL", "MSFT", "JNJ", "PG", "COST", "ADBE")

VALID_SUFFIXES = {".htm", ".html", ".txt"}

MAX_ACCESSIONS_PER_FORM = 3
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0
RETRY_INITIAL_DELAY = 1.0

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

DEFAULT_QUALITY = "unreviewed"


def _iter_downloaded_forms(root: Path, ticker: str, form: str) -> Iterable[Tuple[Path, Path]]:
    base = root / "sec-edgar-filings" / ticker.upper() / form
    if not base.exists():
        return []
    rows: List[Tuple[Path, Path]] = []
    for filing_dir in base.iterdir():
        if not filing_dir.is_dir():
            continue
        valid_docs = sorted(
            (
                file_path
                for file_path in filing_dir.iterdir()
                if file_path.suffix.lower() in VALID_SUFFIXES
            ),
            key=lambda candidate: candidate.name,
        )
        if not valid_docs:
            continue
        rows.append((filing_dir, valid_docs[0]))
    return sorted(rows, key=lambda item: item[0].stat().st_mtime, reverse=True)


def _filed_date(filing_dir: Path, file_path: Path) -> str:
    stem = filing_dir.name.split("_")[0]
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(stem, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")


def _extract_accession(filing_dir: Path) -> str:
    name = filing_dir.name
    if "_" in name:
        parts = name.split("_")
    else:
        parts = [name]
    for part in reversed(parts):
        stripped = part.replace("-", "")
        if stripped.isdigit() and len(stripped) >= 6:
            return part
    return name


def _extract_cik(accession: str) -> str:
    digits = "".join(ch for ch in accession if ch.isdigit())
    if len(digits) >= 10:
        return digits[:10]
    return digits


def _calc_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _try_download(dl: Downloader, ticker: str, form: str, limit: int) -> None:
    delay = RETRY_INITIAL_DELAY
    for attempt in range(RETRY_ATTEMPTS):
        try:
            dl.get(form, ticker, limit=limit)
            return
        except Exception as exc:  # pragma: no cover - defensive network handling
            if attempt == RETRY_ATTEMPTS - 1:
                tqdm.write(
                    f"failed to download {ticker} {form} after {RETRY_ATTEMPTS} attempts: {exc}"
                )
                return
            time.sleep(delay)
            delay *= RETRY_BACKOFF


def _collect_rows(out_dir: Path, tickers: Sequence[str]) -> List[Dict[str, str]]:
    dl = Downloader(
        company_name="CALE-Research",
        email_address="cale@example.com",
        download_folder=str(out_dir),
    )
    rows: List[Dict[str, str]] = []
    for ticker in tqdm(tickers, desc="tickers", leave=False):
        for form in ("10-Q", "10-K"):
            _try_download(dl, ticker, form, limit=MAX_ACCESSIONS_PER_FORM)
            candidates = list(_iter_downloaded_forms(out_dir, ticker, form))
            if not candidates:
                continue
            for filing_dir, file_path in candidates[:MAX_ACCESSIONS_PER_FORM]:
                filed = _filed_date(filing_dir, file_path)
                accession = _extract_accession(filing_dir)
                cik = _extract_cik(accession)
                rows.append(
                    {
                        "ticker": ticker.upper(),
                        "form": form,
                        "filed": filed,
                        "path": str(file_path.resolve()),
                        "accession": accession,
                        "cik": cik,
                        "md5": _calc_md5(file_path),
                        "label": "",
                        "quality_flag": DEFAULT_QUALITY,
                    }
                )
            break
    return rows


def _load_existing_manifest(manifest_path: Path) -> Tuple[List[Dict[str, str]], Set[Tuple[str, str, str, str]]]:
    rows: List[Dict[str, str]] = []
    keys: Set[Tuple[str, str, str, str]] = set()
    if not manifest_path.exists():
        return rows, keys

    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for incoming in reader:
            row: Dict[str, str] = {header: incoming.get(header, "") for header in MANIFEST_HEADERS}
            if not row.get("quality_flag"):
                row["quality_flag"] = DEFAULT_QUALITY
            key = (
                row.get("ticker", ""),
                row.get("form", ""),
                row.get("accession", ""),
                row.get("path", ""),
            )
            rows.append(row)
            keys.add(key)
    return rows, keys


def _merge_manifest(
    manifest_path: Path, new_rows: Sequence[Dict[str, str]]
) -> List[Dict[str, str]]:
    existing_rows, existing_keys = _load_existing_manifest(manifest_path)
    for row in new_rows:
        key = (row.get("ticker", ""), row.get("form", ""), row.get("accession", ""), row.get("path", ""))
        if key in existing_keys:
            continue
        existing_rows.append(row)
        existing_keys.add(key)
    return existing_rows


def main() -> int:
    output_root = Path("data/sec_real")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    manifest_rows.extend(_collect_rows(output_root, DISTRESSED))
    manifest_rows.extend(_collect_rows(output_root, CONTROL))

    manifest_path = Path("reports/realworld/manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    merged_rows = _merge_manifest(manifest_path, manifest_rows)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_HEADERS))
        writer.writeheader()
        writer.writerows(merged_rows)

    print(str(manifest_path.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
