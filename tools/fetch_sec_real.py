#!/usr/bin/env python3
"""Download and maintain a balanced cohort of real SEC filings.

Phase-2 requirements introduce incremental fetching, richer issuer metadata,
rate-limit friendly retry tracking, and cohort balancing controls. The helper
keeps a manifest of downloaded filings and is able to operate in an offline
simulation mode to validate logic when network access is not available."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, MutableMapping, Sequence, Set, Tuple

import pandas as pd

VALID_SUFFIXES = {".htm", ".html", ".txt", ".xml"}
DEFAULT_MAX_ACCESSIONS_PER_FORM = 8
DEFAULT_FORMS = ("10-K", "10-Q", "8-K", "EX-10.1")
DEFAULT_SINCE = "2021-01-01"
DEFAULT_MANIFEST_PATH = Path("data/sec_real/manifest.csv")
DEFAULT_ISSUER_METADATA_PATH = Path("data/sec_real/issuer_metadata.csv")
DEFAULT_LEDGER_PATH = Path("data/sec_real/retry_ledger.json")
DEFAULT_SUMMARY_PATH = Path("data/sec_real/manifest_summary.csv")
DEFAULT_OUTPUT_ROOT = Path("data/sec_real/raw")
DEFAULT_SOURCE = "sec-edgar"
DEFAULT_DOWNLOAD_STATUS = "ok"
DEFAULT_QUALITY = "unreviewed"
SEC_HEADERS = {
    "User-Agent": "CALE-Research support@cale.example",  # pragma: allowlist secret
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
RETRY_INITIAL_DELAY = 1.0
RETRY_BACKOFF = 2.0
MAX_LEDGER_ATTEMPTS = 5

DISTRESSED_WATCHLIST: Sequence[str] = (
    "CVNA",
    "UPST",
    "KSS",
    "CCL",
    "RIVN",
    "AA",
    "FRCB",
    "AMC",
    "BHC",
    "ALLY",
    "DFS",
    "COIN",
    "RUN",
    "NKLA",
    "HOOD",
    "NKTR",
    "CHWY",
    "BIDU",
    "TSLA",
    "RIDEQ",
    "BBBYQ",
    "MSTR",
    "SI",
    "HASI",
    "MARA",
    "NCLH",
    "TDOC",
    "WBD",
    "DOCU",
    "SNAP",
)

INVESTMENT_GRADE_WATCHLIST: Sequence[str] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "ORCL",
    "IBM",
    "TSM",
    "NVDA",
    "JNJ",
    "PG",
    "KO",
    "PEP",
    "MCD",
    "HON",
    "RTX",
    "LMT",
    "CAT",
    "MMM",
    "UNP",
    "UPS",
    "FDX",
    "V",
    "MA",
    "BAC",
    "JPM",
    "GS",
    "BLK",
    "ADBE",
    "INTC",
    "CSCO",
    "TXN",
    "QCOM",
    "NKE",
    "COST",
    "HD",
    "LOW",
    "TGT",
    "WMT",
    "SBUX",
    "INTU",
    "CRM",
    "PYPL",
    "AVGO",
    "ADP",
    "BK",
    "EL",
)

SIMULATION_YEARS: Sequence[int] = tuple(range(2018, datetime.utcnow().year + 1))
SIMULATION_FORMS: Sequence[str] = DEFAULT_FORMS
SIMULATION_FILES_ROOT = Path("data/sec_real/simulated_docs")

MANIFEST_HEADERS: Sequence[str] = (
    "ticker",
    "issuer",
    "form",
    "filed",
    "year",
    "path",
    "accession",
    "cik",
    "md5",
    "label",
    "quality_flag",
    "rating_proxy",
    "sic",
    "naics",
    "download_status",
    "source",
    "downloaded_at",
)


@dataclass
class FilingRecord:
    ticker: str
    issuer: str
    form: str
    filed: str
    year: str
    path: str
    accession: str
    cik: str
    md5: str
    label: str
    quality_flag: str
    rating_proxy: str
    sic: str
    naics: str
    download_status: str
    source: str
    downloaded_at: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "ticker": self.ticker,
            "issuer": self.issuer,
            "form": self.form,
            "filed": self.filed,
            "year": self.year,
            "path": self.path,
            "accession": self.accession,
            "cik": self.cik,
            "md5": self.md5,
            "label": self.label,
            "quality_flag": self.quality_flag,
            "rating_proxy": self.rating_proxy,
            "sic": self.sic,
            "naics": self.naics,
            "download_status": self.download_status,
            "source": self.source,
            "downloaded_at": self.downloaded_at,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help="Only request filings filed on or after this ISO date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--forms",
        default=",".join(DEFAULT_FORMS),
        help="Comma separated list of SEC forms to download.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to store downloaded filings.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the manifest CSV that should be created/updated.",
    )
    parser.add_argument(
        "--issuer-metadata",
        type=Path,
        default=DEFAULT_ISSUER_METADATA_PATH,
        help="Path to a CSV cache of issuer metadata (SIC/NAICS/rating proxy).",
    )
    parser.add_argument(
        "--max-per-form",
        type=int,
        default=DEFAULT_MAX_ACCESSIONS_PER_FORM,
        help="Maximum accessions to download per form (8-10 recommended).",
    )
    parser.add_argument(
        "--retry-ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Path to a JSON ledger tracking failed download attempts.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Optional path to export manifest summary aggregations.",
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help="Optional newline or CSV file containing extra tickers to monitor.",
    )
    parser.add_argument(
        "--extra-tickers",
        default="",
        help="Comma separated list of additional tickers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, no network requests are issued and a simulated manifest is generated.",
    )
    parser.add_argument(
        "--min-ig",
        type=int,
        default=40,
        help="Minimum number of investment grade issuers to retain in the cohort.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling in dry-run mode.",
    )
    return parser.parse_args(argv)


def _normalize_forms(forms: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(forms, str):
        parts = [segment.strip().upper() for segment in forms.split(",") if segment.strip()]
    else:
        parts = [segment.strip().upper() for segment in forms if segment.strip()]
    unique: List[str] = []
    seen: Set[str] = set()
    for form in parts:
        if form not in seen:
            unique.append(form)
            seen.add(form)
    return tuple(unique)


def _load_existing_manifest(manifest_path: Path) -> Tuple[List[FilingRecord], Set[str]]:
    rows: List[FilingRecord] = []
    md5s: Set[str] = set()
    if not manifest_path.exists():
        return rows, md5s
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for incoming in reader:
            record = FilingRecord(
                ticker=incoming.get("ticker", ""),
                issuer=incoming.get("issuer", ""),
                form=incoming.get("form", ""),
                filed=incoming.get("filed", ""),
                year=incoming.get("year", ""),
                path=incoming.get("path", ""),
                accession=incoming.get("accession", ""),
                cik=incoming.get("cik", ""),
                md5=incoming.get("md5", ""),
                label=incoming.get("label", ""),
                quality_flag=incoming.get("quality_flag", DEFAULT_QUALITY),
                rating_proxy=incoming.get("rating_proxy", ""),
                sic=incoming.get("sic", ""),
                naics=incoming.get("naics", ""),
                download_status=incoming.get("download_status", DEFAULT_DOWNLOAD_STATUS),
                source=incoming.get("source", DEFAULT_SOURCE),
                downloaded_at=incoming.get("downloaded_at", ""),
            )
            rows.append(record)
            if record.md5:
                md5s.add(record.md5)
    return rows, md5s


def _write_manifest(manifest_path: Path, rows: Sequence[FilingRecord]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_HEADERS))
        writer.writeheader()
        for record in rows:
            writer.writerow(record.as_dict())


def _read_tickers_from_file(path: Path) -> Sequence[str]:
    if not path or not path.exists():
        return []
    tickers: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for candidate in re.split(r"[,\s]", line.strip()):
                if not candidate:
                    continue
                tickers.append(candidate.upper())
    return tickers


def _load_retry_ledger(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    return data  # type: ignore[return-value]


def _write_retry_ledger(path: Path, ledger: MutableMapping[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)


def _ledger_key(ticker: str, form: str) -> str:
    return f"{ticker.upper()}::{form.upper()}"


def _should_attempt_download(ledger: MutableMapping[str, Dict[str, object]], ticker: str, form: str) -> bool:
    key = _ledger_key(ticker, form)
    if key not in ledger:
        return True
    entry = ledger[key]
    attempts = int(entry.get("attempts", 0))
    last_attempt = float(entry.get("last_attempt", 0.0))
    wait = RETRY_INITIAL_DELAY * (RETRY_BACKOFF**attempts)
    if attempts >= MAX_LEDGER_ATTEMPTS:
        return False
    return (time.time() - last_attempt) >= wait


def _update_ledger(
    ledger: MutableMapping[str, Dict[str, object]],
    ticker: str,
    form: str,
    success: bool,
    error: str | None = None,
) -> None:
    key = _ledger_key(ticker, form)
    if success:
        ledger.pop(key, None)
        return
    entry = ledger.setdefault(key, {})
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["last_attempt"] = time.time()
    if error:
        entry["last_error"] = error


def _ensure_downloader(output_root: Path):
    try:
        from sec_edgar_downloader import Downloader  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "sec_edgar_downloader is required for live runs; install it or use --dry-run"
        ) from exc

    return Downloader(
        company_name="CALE-Research",
        email_address="support@cale.example",
        download_folder=str(output_root),
    )


def _iter_downloaded_forms(root: Path, ticker: str, form: str) -> Iterator[Tuple[Path, Path]]:
    base = root / "sec-edgar-filings" / ticker.upper() / form
    if not base.exists():
        return iter(())
    for filing_dir in sorted(base.iterdir(), key=lambda item: item.name, reverse=True):
        if not filing_dir.is_dir():
            continue
        candidates = sorted(
            (
                file_path
                for file_path in filing_dir.iterdir()
                if file_path.suffix.lower() in VALID_SUFFIXES and file_path.is_file()
            ),
            key=lambda candidate: candidate.name,
        )
        if not candidates:
            continue
        yield filing_dir, candidates[0]


def _filed_date(filing_dir: Path, fallback_file: Path) -> str:
    stem = filing_dir.name.split("_")[0]
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(stem, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.fromtimestamp(fallback_file.stat().st_mtime).strftime("%Y-%m-%d")


def _extract_accession(filing_dir: Path) -> str:
    parts = filing_dir.name.split("_")
    for part in reversed(parts):
        stripped = part.replace("-", "")
        if stripped.isdigit() and len(stripped) >= 6:
            return part
    return filing_dir.name


def _extract_cik(accession: str) -> str:
    digits = "".join(ch for ch in accession if ch.isdigit())
    if len(digits) >= 10:
        return digits[:10]
    return digits


def _calc_md5(file_path: Path) -> str:
    import hashlib

    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _determine_label(ticker: str) -> Tuple[str, str]:
    ticker = ticker.upper()
    if ticker in {item.upper() for item in DISTRESSED_WATCHLIST}:
        return "distressed", "watchlist-distressed"
    if ticker in {item.upper() for item in INVESTMENT_GRADE_WATCHLIST}:
        return "control", "watchlist-ig"
    return "unknown", "unrated"


def _load_company_reference() -> Dict[str, Dict[str, str]]:
    reference_path = Path("data/sec/company_tickers.json")
    if not reference_path.exists():
        return {}
    with reference_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping: Dict[str, Dict[str, str]] = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper()
        if not ticker:
            continue
        mapping[ticker] = {
            "cik": str(entry.get("cik_str", "")),
            "issuer": str(entry.get("title", "")),
        }
    return mapping


def _load_issuer_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    metadata: Dict[str, Dict[str, str]] = {}
    for row in df.to_dict("records"):
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        metadata[ticker] = {
            "issuer": str(row.get("issuer", "")),
            "cik": str(row.get("cik", "")),
            "sic": str(row.get("sic", "")),
            "naics": str(row.get("naics", "")),
            "rating_proxy": str(row.get("rating_proxy", "")),
        }
    return metadata


def _write_issuer_metadata(path: Path, metadata: Dict[str, Dict[str, str]]) -> None:
    if not metadata:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for ticker, info in sorted(metadata.items()):
        records.append(
            {
                "ticker": ticker,
                "issuer": info.get("issuer", ""),
                "cik": info.get("cik", ""),
                "sic": info.get("sic", ""),
                "naics": info.get("naics", ""),
                "rating_proxy": info.get("rating_proxy", ""),
            }
        )
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)


def _fetch_remote_metadata(cik: str) -> Tuple[str, str]:
    """Attempt to fetch SIC/NAICS metadata from the SEC submissions API."""

    if not cik:
        return "", ""
    try:
        import httpx
    except ModuleNotFoundError:  # pragma: no cover - httpx is an install-time dependency
        return "", ""

    cik_str = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
    try:
        with httpx.Client(headers=SEC_HEADERS, timeout=10.0) as client:
            response = client.get(url)
            if response.status_code != 200:
                return "", ""
            payload = response.json()
    except Exception:  # pragma: no cover - network failure tolerant
        return "", ""

    sic = str(payload.get("sic", "")) if isinstance(payload, dict) else ""
    naics = ""
    if isinstance(payload, dict):
        entity_info = payload.get("entityInfo") or {}
        if isinstance(entity_info, dict):
            naics = str(entity_info.get("naics", ""))
    return sic, naics


def _collect_metadata(
    ticker: str,
    reference: Dict[str, Dict[str, str]],
    cache: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    ticker_upper = ticker.upper()
    if ticker_upper in cache:
        return cache[ticker_upper]
    entry = reference.get(ticker_upper, {})
    issuer = entry.get("issuer", ticker_upper)
    cik = entry.get("cik", "")
    sic, naics = _fetch_remote_metadata(cik)
    label, rating_proxy = _determine_label(ticker_upper)
    if not rating_proxy or rating_proxy == "unrated":
        rating_proxy = cache.get(ticker_upper, {}).get("rating_proxy", "unrated")
    cache[ticker_upper] = {
        "issuer": issuer,
        "cik": cik,
        "sic": sic,
        "naics": naics,
        "rating_proxy": rating_proxy,
        "label": label,
    }
    return cache[ticker_upper]


def _collect_live_rows(
    args: argparse.Namespace,
    forms: Sequence[str],
    tickers: Sequence[str],
    existing_rows: List[FilingRecord],
    existing_md5: Set[str],
) -> List[FilingRecord]:
    output_root: Path = args.out
    output_root.mkdir(parents=True, exist_ok=True)

    downloader = _ensure_downloader(output_root)
    since = args.since if args.since else None
    ledger = _load_retry_ledger(Path(args.retry_ledger))
    reference = _load_company_reference()
    metadata_cache = _load_issuer_metadata(Path(args.issuer_metadata))

    new_rows: List[FilingRecord] = []
    for ticker in tickers:
        ticker_upper = ticker.upper()
        for form in forms:
            if not _should_attempt_download(ledger, ticker_upper, form):
                continue
            try:
                downloader.get(form, ticker_upper, amount=args.max_per_form, after=since)
                _update_ledger(ledger, ticker_upper, form, success=True)
            except Exception as exc:  # pragma: no cover - defensive network handling
                _update_ledger(ledger, ticker_upper, form, success=False, error=str(exc))
                continue
            for filing_dir, file_path in _iter_downloaded_forms(output_root, ticker_upper, form):
                filed_date = _filed_date(filing_dir, file_path)
                if since and filed_date < since:
                    continue
                accession = _extract_accession(filing_dir)
                cik = _extract_cik(accession)
                md5 = _calc_md5(file_path)
                if md5 in existing_md5:
                    continue
                metadata = _collect_metadata(ticker_upper, reference, metadata_cache)
                label = metadata.get("label", "unknown")
                rating_proxy = metadata.get("rating_proxy", "unrated")
                year = filed_date[:4]
                downloaded_at = datetime.utcnow().isoformat(timespec="seconds")
                record = FilingRecord(
                    ticker=ticker_upper,
                    issuer=metadata.get("issuer", ticker_upper),
                    form=form,
                    filed=filed_date,
                    year=year,
                    path=str(file_path.resolve()),
                    accession=accession,
                    cik=cik or metadata.get("cik", ""),
                    md5=md5,
                    label=label,
                    quality_flag=DEFAULT_QUALITY,
                    rating_proxy=rating_proxy,
                    sic=metadata.get("sic", ""),
                    naics=metadata.get("naics", ""),
                    download_status=DEFAULT_DOWNLOAD_STATUS,
                    source=DEFAULT_SOURCE,
                    downloaded_at=downloaded_at,
                )
                existing_md5.add(md5)
                new_rows.append(record)
    _write_retry_ledger(Path(args.retry_ledger), ledger)
    _write_issuer_metadata(Path(args.issuer_metadata), metadata_cache)
    return new_rows


def _create_simulated_document(path: Path, ticker: str, form: str, year: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(f"{ticker}-{form}-{year}-{seed}")
    sentences = [
        f"{ticker} filed a {form} for fiscal year {year}.",
        f"Revenue growth remained {random.randint(-10, 25)}%.",
        f"Leverage ratio updated to {round(random.uniform(0.5, 6.5), 2)}x.",
        f"Management commentary mentions {random.choice(['liquidity', 'capital allocation', 'supply chain'])}.",
        f"Credit agreement covenants {random.choice(['tightened', 'loosened', 'remained stable'])} by lenders.",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(sentences))


def _collect_simulated_rows(
    args: argparse.Namespace,
    forms: Sequence[str],
    tickers: Sequence[str],
    existing_rows: List[FilingRecord],
    existing_md5: Set[str],
) -> List[FilingRecord]:
    seed = args.seed
    random.seed(seed)
    reference = _load_company_reference()
    metadata_cache = _load_issuer_metadata(Path(args.issuer_metadata))
    since_year = int((args.since or DEFAULT_SINCE)[:4])
    rows: List[FilingRecord] = []

    for ticker in tickers:
        metadata = _collect_metadata(ticker, reference, metadata_cache)
        label = metadata.get("label", "unknown")
        rating_proxy = metadata.get("rating_proxy", "unrated")
        issuer = metadata.get("issuer", ticker)
        cik = metadata.get("cik", "")
        sic = metadata.get("sic", "")
        naics = metadata.get("naics", "")
        years = [year for year in SIMULATION_YEARS if year >= since_year]
        random.shuffle(years)
        for year in years[: args.max_per_form]:
            for form in forms:
                filename = f"{ticker}_{form}_{year}.txt"
                document_path = SIMULATION_FILES_ROOT / ticker / str(year) / filename
                _create_simulated_document(document_path, ticker, form, year, seed)
                md5 = _calc_md5(document_path)
                if md5 in existing_md5:
                    continue
                accession = f"{ticker}-{form}-{year}-{md5[:8]}"
                filed = f"{year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                downloaded_at = datetime.utcnow().isoformat(timespec="seconds")
                record = FilingRecord(
                    ticker=ticker,
                    issuer=issuer,
                    form=form,
                    filed=filed,
                    year=str(year),
                    path=str(document_path.resolve()),
                    accession=accession,
                    cik=cik,
                    md5=md5,
                    label=label,
                    quality_flag=DEFAULT_QUALITY,
                    rating_proxy=rating_proxy,
                    sic=sic,
                    naics=naics,
                    download_status="offline-simulated",
                    source="offline-simulation",
                    downloaded_at=downloaded_at,
                )
                existing_md5.add(md5)
                rows.append(record)
    _write_issuer_metadata(Path(args.issuer_metadata), metadata_cache)
    return rows


def _build_ticker_list(
    args: argparse.Namespace, min_ig: int = 40
) -> Tuple[Sequence[str], Sequence[str]]:
    base_distressed = [ticker.upper() for ticker in DISTRESSED_WATCHLIST]
    base_ig = [ticker.upper() for ticker in INVESTMENT_GRADE_WATCHLIST]

    extra_file = _read_tickers_from_file(Path(args.tickers_file)) if args.tickers_file else []
    extra_inline = [ticker.strip().upper() for ticker in args.extra_tickers.split(",") if ticker.strip()]

    ig_set = dict.fromkeys(base_ig + extra_file + extra_inline)
    distressed_set = dict.fromkeys(base_distressed + extra_file)

    ig_tickers = list(ig_set.keys())
    distressed_tickers = list(distressed_set.keys())

    if len(ig_tickers) < min_ig:
        raise RuntimeError(
            f"Investment grade cohort too small ({len(ig_tickers)} < {min_ig}); add tickers via CLI."
        )
    return distressed_tickers, ig_tickers


def _merge_rows(existing: List[FilingRecord], incoming: List[FilingRecord]) -> List[FilingRecord]:
    merged = list(existing)
    existing_keys = {
        (row.ticker, row.form, row.accession, row.path): idx
        for idx, row in enumerate(merged)
    }
    for record in incoming:
        key = (record.ticker, record.form, record.accession, record.path)
        if key in existing_keys:
            merged[existing_keys[key]] = record
        else:
            merged.append(record)
    merged.sort(key=lambda item: (item.filed, item.ticker, item.form, item.accession))
    return merged


def _summarize_manifest(manifest_path: Path, summary_path: Path | None = None) -> pd.DataFrame:
    if not manifest_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(manifest_path)
    if df.empty:
        return df
    for column in ("label", "issuer", "year", "form", "rating_proxy"):
        if column not in df.columns:
            df[column] = ""
    summary = (
        df.groupby(["label", "issuer", "year", "form", "rating_proxy"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return summary


def _emit_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No filings captured; summary unavailable.")
        return
    print("Manifest Summary (by class, issuer, year, form, rating_proxy):")
    print(summary.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    forms = _normalize_forms(args.forms)
    distressed_tickers, ig_tickers = _build_ticker_list(args, min_ig=args.min_ig)
    all_tickers = list(dict.fromkeys(list(distressed_tickers) + list(ig_tickers)))

    manifest_path = Path(args.manifest)
    existing_rows, existing_md5 = _load_existing_manifest(manifest_path)

    if args.dry_run:
        new_rows = _collect_simulated_rows(args, forms, all_tickers, existing_rows, existing_md5)
    else:
        new_rows = _collect_live_rows(args, forms, all_tickers, existing_rows, existing_md5)

    if not new_rows:
        print("No new filings were added. Manifest left unchanged.")
        summary = _summarize_manifest(manifest_path, Path(args.summary))
        _emit_summary(summary)
        return 0

    merged_rows = _merge_rows(existing_rows, new_rows)
    _write_manifest(manifest_path, merged_rows)
    summary = _summarize_manifest(manifest_path, Path(args.summary))
    _emit_summary(summary)
    print(str(manifest_path.resolve()))

    if args.dry_run:
        print("offline simulation completed; counts reflect synthetic data")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
