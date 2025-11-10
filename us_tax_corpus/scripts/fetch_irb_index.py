"""Crawl the IRS Internal Revenue Bulletin index and normalize guidance documents."""
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
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

IRB_INDEX_URL = "https://www.irs.gov/irb"
RAW_DIR = Path("us_tax_corpus/raw/irb_html")
PARSED_PATH = Path("us_tax_corpus/parsed/irb_docs.jsonl")
MANIFEST_PATH = Path("us_tax_corpus/manifests/sources.csv")


@dataclass
class IRBDocument:
    identifier: str
    title: str
    url: str
    publication_date: str
    md5: str
    text: str
    snapshot_date: str
    layer: str = "irb"
    version: str = "latest"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def fetch_url(url: str, session: Optional[requests.Session] = None) -> str:
    sess = session or requests.Session()
    response = sess.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_issue(issue_url: str, session: Optional[requests.Session] = None) -> Iterator[IRBDocument]:
    html = fetch_url(issue_url, session=session)
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title").get_text(strip=True) if soup.title else issue_url
    snapshot = dt.datetime.utcnow().date().isoformat()
    for anchor in soup.select("a"):
        href = anchor.get("href")
        text = anchor.get_text(strip=True)
        if not href or not text:
            continue
        if "rev" in text.lower() or "notice" in text.lower() or "announcement" in text.lower():
            doc_url = requests.compat.urljoin(issue_url, href)
            doc_text = fetch_url(doc_url, session=session)
            md5 = compute_md5(doc_text)
            identifier = f"irb:{snapshot}:{md5[:12]}"
            yield IRBDocument(
                identifier=identifier,
                title=text,
                url=doc_url,
                publication_date=title,
                md5=md5,
                text=doc_text,
                snapshot_date=snapshot,
            )
            time.sleep(0.2)


def crawl_index(limit: int = 5, session: Optional[requests.Session] = None) -> List[IRBDocument]:
    html = fetch_url(IRB_INDEX_URL, session=session)
    RAW_DIR.joinpath("index.html").write_text(html, encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    issue_links = [requests.compat.urljoin(IRB_INDEX_URL, a.get("href")) for a in soup.select("a") if a.get("href")]
    documents: List[IRBDocument] = []
    for issue_url in issue_links[:limit]:
        try:
            documents.extend(parse_issue(issue_url, session=session))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to parse %s: %s", issue_url, exc)
        time.sleep(0.5)
    return documents


def write_records(records: Iterable[IRBDocument]) -> int:
    count = 0
    with PARSED_PATH.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")
            count += 1
    return count


def update_manifest(records: Iterable[IRBDocument]) -> None:
    is_new = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(["doc_id", "source", "url", "md5", "snapshot_date", "layer", "status"])
        for record in records:
            writer.writerow(
                [
                    record.identifier,
                    "irb",
                    record.url,
                    record.md5,
                    record.snapshot_date,
                    record.layer,
                    "parsed",
                ]
            )


def download_and_parse(limit: int = 5, session: Optional[requests.Session] = None) -> int:
    ensure_dirs()
    documents = crawl_index(limit=limit, session=session)
    write_records(documents)
    update_manifest(documents)
    logger.info("Captured %s IRB documents", len(documents))
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch IRS IRB guidance")
    parser.add_argument("--limit", type=int, default=5, help="Number of IRB issues to crawl")
    args = parser.parse_args()
    download_and_parse(limit=args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
