"""Fetch Title 26 eCFR data using the public API."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

ECFR_API = "https://www.ecfr.gov/api/versioner/v1/full/2024-01-01/title-26.json"
RAW_DIR = Path("us_tax_corpus/raw/ecfr_title26")
PARSED_PATH = Path("us_tax_corpus/parsed/cfr_sections.jsonl")
MANIFEST_PATH = Path("us_tax_corpus/manifests/sources.csv")


@dataclass
class RegulationRecord:
    identifier: str
    section: str
    heading: str
    text: str
    md5: str
    url: str
    snapshot_date: str
    layer: str = "cfr"
    version: str = "latest"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def fetch_json(session: Optional[requests.Session] = None) -> dict:
    sess = session or requests.Session()
    logger.info("Downloading eCFR Title 26 JSON from %s", ECFR_API)
    response = sess.get(ECFR_API, timeout=60)
    response.raise_for_status()
    return response.json()


def iter_sections(node: dict, chapter: Optional[str] = None) -> Iterator[RegulationRecord]:
    node_type = node.get("type")
    label = node.get("label", "").strip()
    heading = node.get("heading", "").strip()
    content = node.get("content", "")
    citation = node.get("identifier", label)
    if node_type == "section" and content:
        text = content if isinstance(content, str) else "\n".join(content)
        md5 = compute_md5(text)
        section_id = f"cfr:26:{citation or label}"
        yield RegulationRecord(
            identifier=section_id,
            section=citation or label,
            heading=heading or label,
            text=text,
            md5=md5,
            url=ECFR_API,
            snapshot_date=dt.datetime.utcnow().date().isoformat(),
        )
    for child in node.get("children", []):
        yield from iter_sections(child, chapter=label or chapter)


def write_records(records: Iterable[RegulationRecord]) -> int:
    count = 0
    with PARSED_PATH.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")
            count += 1
    return count


def update_manifest(records: Iterable[RegulationRecord]) -> None:
    is_new = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(["doc_id", "source", "url", "md5", "snapshot_date", "layer", "status"])
        for record in records:
            writer.writerow(
                [
                    record.identifier,
                    "ecfr_title26",
                    record.url,
                    record.md5,
                    record.snapshot_date,
                    record.layer,
                    "parsed",
                ]
            )


def download_and_parse(session: Optional[requests.Session] = None) -> int:
    ensure_dirs()
    data = fetch_json(session=session)
    raw_path = RAW_DIR / "title26.json"
    raw_path.write_text(json.dumps(data)[:100000], encoding="utf-8")
    records = list(iter_sections(data))
    write_records(records)
    update_manifest(records)
    logger.info("Parsed %s eCFR sections", len(records))
    return len(records)


def main() -> None:
    argparse.ArgumentParser(description="Fetch eCFR Title 26 JSON").parse_args()
    download_and_parse()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
