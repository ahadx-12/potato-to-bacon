"""Fetch United States Tax Court opinions from CourtListener."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

COURTLISTENER_ENDPOINT = "https://www.courtlistener.com/api/rest/v3/opinions/"
RAW_DIR = Path("us_tax_corpus/raw/tax_court_json")
PARSED_PATH = Path("us_tax_corpus/parsed/cases.jsonl")
MANIFEST_PATH = Path("us_tax_corpus/manifests/sources.csv")


@dataclass
class OpinionRecord:
    identifier: str
    citation: str
    date: str
    url: str
    md5: str
    text: str
    outcome: Optional[str]
    snapshot_date: str
    layer: str = "case"
    version: str = "latest"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def fetch_page(url: str, session: Optional[requests.Session] = None) -> dict:
    sess = session or requests.Session()
    response = sess.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def iter_opinions(limit: int, session: Optional[requests.Session] = None) -> Iterator[OpinionRecord]:
    next_url = f"{COURTLISTENER_ENDPOINT}?court=tax&order_by=-date_decided"
    snapshot = dt.datetime.utcnow().date().isoformat()
    count = 0
    while next_url and count < limit:
        payload = fetch_page(next_url, session=session)
        RAW_DIR.joinpath(f"page_{count}.json").write_text(json.dumps(payload), encoding="utf-8")
        for result in payload.get("results", []):
            if count >= limit:
                break
            text = result.get("plain_text") or ""
            if not text:
                continue
            md5 = compute_md5(text)
            identifier = f"case:USTC:{result.get('id')}"
            outcome = result.get("judges") or result.get("decision")
            citation = result.get("citation") or result.get("case_name", "")
            yield OpinionRecord(
                identifier=identifier,
                citation=citation,
                date=result.get("date_decided") or "",
                url=result.get("absolute_url") or next_url,
                md5=md5,
                text=text,
                outcome=outcome,
                snapshot_date=snapshot,
            )
            count += 1
        next_url = payload.get("next")
        time.sleep(0.2)


def write_records(records: Iterable[OpinionRecord]) -> int:
    count = 0
    with PARSED_PATH.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")
            count += 1
    return count


def update_manifest(records: Iterable[OpinionRecord]) -> None:
    is_new = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(["doc_id", "source", "url", "md5", "snapshot_date", "layer", "status"])
        for record in records:
            writer.writerow(
                [
                    record.identifier,
                    "tax_court",
                    record.url,
                    record.md5,
                    record.snapshot_date,
                    record.layer,
                    "parsed",
                ]
            )


def download_and_parse(limit: int = 200, session: Optional[requests.Session] = None) -> int:
    ensure_dirs()
    records = list(iter_opinions(limit=limit, session=session))
    write_records(records)
    update_manifest(records)
    logger.info("Captured %s Tax Court opinions", len(records))
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Tax Court opinions")
    parser.add_argument("--limit", type=int, default=200, help="Opinion count to fetch")
    args = parser.parse_args()
    download_and_parse(limit=args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
