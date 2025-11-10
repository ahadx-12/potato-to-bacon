"""Link citations across the U.S. tax law corpus."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

PARSED_DIR = Path("us_tax_corpus/parsed")
CROSSWALKS_PATH = Path("us_tax_corpus/manifests/crosswalks.csv")

CITATION_PATTERNS: Dict[str, re.Pattern[str]] = {
    "usc": re.compile(r"26\s+U\.S\.C\.\s+§?\s*(?P<section>\d+[A-Za-z0-9\-]*)"),
    "cfr": re.compile(r"26\s+C\.F\.R\.\s+§?\s*(?P<section>[0-9]+\.[0-9A-Za-z\-]+)"),
    "irb": re.compile(r"IRB\s+(?P<year>\d{4})-(?P<issue>\d{2})"),
    "case": re.compile(r"(?P<case>\d+\s+T\.C\.\s+\d+)")
}


@dataclass
class CrosswalkEdge:
    source_id: str
    target_id: str
    relation: str
    confidence: float

    def to_row(self) -> List[str]:
        return [self.source_id, self.target_id, self.relation, f"{self.confidence:.3f}"]


def load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_doc_id(layer: str, match: re.Match[str]) -> Optional[str]:
    if layer == "usc":
        section = match.group("section").replace(" ", "")
        return f"usc:26:§{section}"
    if layer == "cfr":
        section = match.group("section")
        return f"cfr:26:{section}"
    if layer == "irb":
        year = match.group("year")
        issue = match.group("issue")
        return f"irb:{year}-{issue}"
    if layer == "case":
        return f"case:USTC:{match.group('case').replace(' ', '')}"
    return None


def extract_citations(text: str) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for layer, pattern in CITATION_PATTERNS.items():
        for match in pattern.finditer(text):
            target = normalize_doc_id(layer, match)
            if target:
                edges.append((layer, target))
    return edges


def build_crosswalk(confidence: float = 0.95) -> List[CrosswalkEdge]:
    parsed_files = {
        "usc": PARSED_DIR / "usc_sections.jsonl",
        "cfr": PARSED_DIR / "cfr_sections.jsonl",
        "irb": PARSED_DIR / "irb_docs.jsonl",
        "case": PARSED_DIR / "cases.jsonl",
    }
    edges: List[CrosswalkEdge] = []
    for layer, path in parsed_files.items():
        for row in load_jsonl(path):
            source = row.get("identifier") or row.get("section")
            if not source or "text" not in row:
                continue
            for target_layer, target_id in extract_citations(str(row["text"])):
                relation = f"{layer}->{target_layer}"
                edges.append(CrosswalkEdge(str(source), target_id, relation, confidence))
    return edges


def write_crosswalk(edges: Iterable[CrosswalkEdge]) -> int:
    CROSSWALKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CROSSWALKS_PATH.exists()
    with CROSSWALKS_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(["source_id", "target_id", "relation", "confidence"])
        count = 0
        for edge in edges:
            writer.writerow(edge.to_row())
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Link citations across corpus layers")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    edges = build_crosswalk(confidence=args.confidence)
    count = write_crosswalk(edges)
    logger.info("Recorded %s cross-layer edges", count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
