"""Download and normalize Title 26 of the U.S. Code from the OLRC USLM feed."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)

USLM_ZIP_URL = "https://uscode.house.gov/download/releasepoints/uscode/mostrecent/xml/USCODE-title26.zip"
RAW_DIR = Path("us_tax_corpus/raw/usc_title26_xml")
PARSED_PATH = Path("us_tax_corpus/parsed/usc_sections.jsonl")
MANIFEST_PATH = Path("us_tax_corpus/manifests/sources.csv")


@dataclass
class SectionRecord:
    """Normalized representation of a USC section."""

    identifier: str
    section: str
    heading: str
    text: str
    md5: str
    url: str
    snapshot_date: str
    layer: str = "usc"
    version: str = "latest"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_zip(session: Optional[requests.Session] = None) -> bytes:
    sess = session or requests.Session()
    logger.info("Downloading Title 26 USLM zip from %s", USLM_ZIP_URL)
    response = sess.get(USLM_ZIP_URL, timeout=60)
    response.raise_for_status()
    return response.content


def parse_sections(xml_bytes: bytes, source_url: str, snapshot: str) -> Iterator[SectionRecord]:
    root = ET.fromstring(xml_bytes)
    ns = {"uslm": "http://xml.house.gov/schemas/uslm/1.0"}
    for section in root.findall(".//uslm:section", ns):
        identifier = section.get("identifier") or ""
        heading = (section.findtext("uslm:heading", default="", namespaces=ns) or "").strip()
        number = (section.findtext("uslm:num", default="", namespaces=ns) or "").strip()
        raw_text_parts: List[str] = []
        for para in section.findall(".//uslm:content//uslm:p", ns):
            text = " ".join(para.itertext()).strip()
            if text:
                raw_text_parts.append(text)
        text = "\n".join(raw_text_parts)
        if not text:
            continue
        md5 = compute_md5(text)
        identifier_norm = identifier or f"usc:26:{number}" if number else "usc:26:unknown"
        yield SectionRecord(
            identifier=identifier_norm,
            section=number,
            heading=heading,
            text=text,
            md5=md5,
            url=source_url,
            snapshot_date=snapshot,
        )


def write_records(records: Iterable[SectionRecord]) -> int:
    count = 0
    with PARSED_PATH.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")
            count += 1
    return count


def update_manifest(records: Iterable[SectionRecord]) -> None:
    is_new = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(["doc_id", "source", "url", "md5", "snapshot_date", "layer", "status"])
        for record in records:
            writer.writerow(
                [
                    record.identifier,
                    "usc_title26",
                    record.url,
                    record.md5,
                    record.snapshot_date,
                    record.layer,
                    "parsed",
                ]
            )


def download_and_parse(session: Optional[requests.Session] = None) -> int:
    ensure_dirs()
    snapshot = dt.datetime.utcnow().date().isoformat()
    content = fetch_zip(session=session)
    manifest_url = USLM_ZIP_URL
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        count = 0
        parsed_records: List[SectionRecord] = []
        for name in zf.namelist():
            if not name.lower().endswith(".xml"):
                continue
            xml_bytes = zf.read(name)
            records = list(parse_sections(xml_bytes, manifest_url, snapshot))
            parsed_records.extend(records)
            count += write_records(records)
        update_manifest(parsed_records)
    raw_path = RAW_DIR / f"title26_{snapshot}.zip"
    raw_path.write_bytes(content)
    logger.info("Wrote %s sections to %s", count, PARSED_PATH)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Title 26 USLM data")
    parser.add_argument("--session-cache", help="Path to persist the downloaded zip", default=None)
    args = parser.parse_args()
    if args.session_cache:
        cache_path = Path(args.session_cache)
        if cache_path.exists():
            logger.info("Loading cached zip from %s", cache_path)
            content = cache_path.read_bytes()
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                snapshot = dt.datetime.utcnow().date().isoformat()
                records = []
                for name in zf.namelist():
                    if not name.lower().endswith(".xml"):
                        continue
                    records.extend(parse_sections(zf.read(name), USLM_ZIP_URL, snapshot))
                write_records(records)
                update_manifest(records)
                return
    download_and_parse()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
