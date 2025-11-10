"""Simulate the CALE-LAW tax pipeline on a compact in-memory dataset.

The script mirrors the multi-phase ingestion/metrics workflow but substitutes
real downloads with a deterministic fixture so the full stack can be exercised
in constrained environments.  Generated artifacts live under ``us_tax_corpus``
and ``out`` matching the production layout.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from potatobacon.law.ambiguity_entropy import normalized_entropy
from potatobacon.law.contradiction_score import (
    ContradictionFeatures,
    contradiction_probability,
)
from potatobacon.law.flaw_score import flaw_score, policy_flaw_score
from potatobacon.law.impact_weight import impact_weight
from potatobacon.law.judicial_uncertainty import judicial_uncertainty
from potatobacon.law.network_fragility import compute_network_scores
from us_tax_corpus.scripts.link_citations import build_crosswalk

logger = logging.getLogger(__name__)
PARSED_DIR = ROOT / "us_tax_corpus" / "parsed"
MANIFEST_DIR = ROOT / "us_tax_corpus" / "manifests"
RAW_DIR = ROOT / "us_tax_corpus" / "raw"
LOG_DIR = ROOT / "us_tax_corpus" / "logs"
OUT_DIR = ROOT / "out"

USC_PATH = PARSED_DIR / "usc_sections.jsonl"
CFR_PATH = PARSED_DIR / "cfr_sections.jsonl"
IRB_PATH = PARSED_DIR / "irb_docs.jsonl"
CASE_PATH = PARSED_DIR / "cases.jsonl"
SOURCES_CSV = MANIFEST_DIR / "sources.csv"
CROSSWALK_CSV = MANIFEST_DIR / "crosswalks.csv"

PAIRS_SCORED = OUT_DIR / "pairs_scored.jsonl"
SECTION_METRICS = OUT_DIR / "section_metrics.jsonl"
GRAPH_JSON = OUT_DIR / "graph.json"
TIMESERIES_JSON = OUT_DIR / "time_series.json"
SUMMARY_JSON = OUT_DIR / "summary_stub.json"

SNAPSHOT = dt.date.today().isoformat()


@dataclass
class SampleRecord:
    identifier: str
    heading: str
    text: str
    url: str
    layer: str
    jurisdiction: str
    section: str
    tags: Sequence[str]
    outcomes: Sequence[float]
    circuits: Sequence[str]

    def to_payload(self) -> Dict[str, object]:
        """Transform the sample data into the canonical JSONL row."""

        md5 = hashlib.md5(self.text.encode("utf-8")).hexdigest()
        return {
            "identifier": self.identifier,
            "id": self.identifier,
            "heading": self.heading,
            "section": self.section,
            "text": self.text,
            "md5": md5,
            "url": self.url,
            "snapshot_date": SNAPSHOT,
            "layer": self.layer,
            "jurisdiction": self.jurisdiction,
            "tags": list(self.tags),
            "outcomes": list(self.outcomes),
            "circuits": list(self.circuits),
        }


def ensure_scaffold() -> None:
    """Create directory scaffolding expected by the tests and pipeline."""

    for directory in (PARSED_DIR, MANIFEST_DIR, RAW_DIR, LOG_DIR, OUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for stem in ("usc_title26_xml", "ecfr_title26", "irb_html", "tax_court_json"):
        (RAW_DIR / stem).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.unlink(missing_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def seed_ingestion() -> Dict[str, List[Dict[str, object]]]:
    """Populate the parsed directory with a compact representative dataset."""

    statute_text = (
        "Section 1 imposes a tax on taxable income. "
        "Notwithstanding any other provision, individuals with income under 20,000 "
        "are taxed at 10 percent. See 26 C.F.R. §1.1-1 for computation."
    )
    reg_text = (
        "26 C.F.R. §1.1-1 explains that the tax on taxable income shall be computed "
        "using progressive brackets. The rule does not apply to amounts not exceeding 15,000."
    )
    reg_alt_text = (
        "26 C.F.R. §1.36B-2 addresses premium tax credits and references 26 U.S.C. §36B."
    )
    irb_text = (
        "Notice 2023-05 clarifies coordination between 26 U.S.C. §1 and 26 C.F.R. §1.1-1."
    )
    case_one_text = (
        "In Smith v. Commissioner, 10 T.C. 123, the court held that 26 U.S.C. §1 requires "
        "inclusion of bonus income despite taxpayer arguments."
    )
    case_two_text = (
        "Johnson v. Commissioner, 11 T.C. 456, analyzed 26 U.S.C. §36B and coordinated "
        "regulations including 26 C.F.R. §1.36B-2."
    )

    records: Dict[str, List[Dict[str, object]]] = {
        "usc": [
            SampleRecord(
                identifier="usc:26:§1",
                heading="Tax imposed",
                text=statute_text,
                url="https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section1",
                layer="usc",
                jurisdiction="federal",
                section="1",
                tags=("income", "individual"),
                outcomes=(0.6, 0.8),
                circuits=("Tax Court", "Tax Court"),
            ).to_payload(),
            SampleRecord(
                identifier="usc:26:§36B",
                heading="Refundable credit for coverage",
                text="Section 36B provides premium tax credits and references related regulations.",
                url="https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section36B",
                layer="usc",
                jurisdiction="federal",
                section="36B",
                tags=("credits", "health"),
                outcomes=(0.4, 0.5, 0.7),
                circuits=("Tax Court", "9th Cir.", "9th Cir."),
            ).to_payload(),
        ],
        "cfr": [
            SampleRecord(
                identifier="cfr:26:1.1-1",
                heading="Income tax on individuals",
                text=reg_text,
                url="https://www.ecfr.gov/current/title-26/section-1.1-1",
                layer="cfr",
                jurisdiction="federal",
                section="1.1-1",
                tags=("income", "computations"),
                outcomes=(0.3, 0.5, 0.7),
                circuits=("Tax Court", "Tax Court", "8th Cir."),
            ).to_payload(),
            SampleRecord(
                identifier="cfr:26:1.36B-2",
                heading="Premium tax credit eligibility",
                text=reg_alt_text,
                url="https://www.ecfr.gov/current/title-26/section-1.36B-2",
                layer="cfr",
                jurisdiction="federal",
                section="1.36B-2",
                tags=("credits", "health"),
                outcomes=(0.45, 0.55),
                circuits=("Tax Court", "9th Cir."),
            ).to_payload(),
        ],
        "irb": [
            SampleRecord(
                identifier="irb:2023-05:notice",
                heading="Notice 2023-05",
                text=irb_text,
                url="https://www.irs.gov/pub/irs-irbs/irb23-05.pdf",
                layer="irb",
                jurisdiction="federal",
                section="Notice 2023-05",
                tags=("notice", "coordination"),
                outcomes=(0.5,),
                circuits=("IRS",),
            ).to_payload(),
        ],
        "case": [
            SampleRecord(
                identifier="case:USTC:10TC123",
                heading="Smith v. Commissioner",
                text=case_one_text,
                url="https://ustaxcourt.gov/Smith_v_Commissioner",
                layer="case",
                jurisdiction="Tax Court",
                section="10 T.C. 123",
                tags=("income", "bonus"),
                outcomes=(0.2, 0.6, 0.9),
                circuits=("Tax Court", "Tax Court", "Tax Court"),
            ).to_payload(),
            SampleRecord(
                identifier="case:USTC:11TC456",
                heading="Johnson v. Commissioner",
                text=case_two_text,
                url="https://ustaxcourt.gov/Johnson_v_Commissioner",
                layer="case",
                jurisdiction="Tax Court",
                section="11 T.C. 456",
                tags=("credits", "coverage"),
                outcomes=(0.3, 0.55, 0.65),
                circuits=("Tax Court", "9th Cir.", "9th Cir."),
            ).to_payload(),
        ],
    }

    write_jsonl(USC_PATH, records["usc"])
    write_jsonl(CFR_PATH, records["cfr"])
    write_jsonl(IRB_PATH, records["irb"])
    write_jsonl(CASE_PATH, records["case"])

    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SOURCES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_id", "source", "url", "md5", "snapshot_date", "layer", "status"])
        for layer, layer_records in records.items():
            for record in layer_records:
                writer.writerow(
                    [
                        record["identifier"],
                        layer,
                        record["url"],
                        record["md5"],
                        record["snapshot_date"],
                        record["layer"],
                        "parsed",
                    ]
                )

    logger.info("Seeded parsed fixtures: %s", {layer: len(rows) for layer, rows in records.items()})
    return records


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity using simple word frequency vectors."""

    def vectorize(text: str) -> Dict[str, float]:
        tokens = [token.lower() for token in text.split() if token]
        vector: Dict[str, float] = {}
        for token in tokens:
            vector[token] = vector.get(token, 0.0) + 1.0
        return vector

    vec_a = vectorize(text_a)
    vec_b = vectorize(text_b)
    if not vec_a or not vec_b:
        return 0.0
    intersection = set(vec_a) & set(vec_b)
    dot = sum(vec_a[token] * vec_b[token] for token in intersection)
    norm_a = sum(value ** 2 for value in vec_a.values()) ** 0.5
    norm_b = sum(value ** 2 for value in vec_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


NEGATION_TERMS = {"not", "notwithstanding", "except"}


def negation_overlap(text_a: str, text_b: str) -> float:
    """Return a normalized count of negation/control terms appearing in both texts."""

    tokens_a = {token.strip(".,;:").lower() for token in text_a.split()}
    tokens_b = {token.strip(".,;:").lower() for token in text_b.split()}
    overlap = NEGATION_TERMS & tokens_a & tokens_b
    if not NEGATION_TERMS:
        return 0.0
    return len(overlap) / len(NEGATION_TERMS)


def numeric_conflict(text_a: str, text_b: str) -> float:
    """Heuristic numeric conflict detection based on mismatched thresholds."""

    import re

    numbers_a = [float(match.group()) for match in re.finditer(r"\d+(?:\.\d+)?", text_a)]
    numbers_b = [float(match.group()) for match in re.finditer(r"\d+(?:\.\d+)?", text_b)]
    if not numbers_a or not numbers_b:
        return 0.0
    difference = min(abs(a - b) for a in numbers_a for b in numbers_b)
    return 1.0 if difference >= 5 else difference / 5.0


LAYER_LABELS = {
    "usc": "statute",
    "cfr": "regulation",
    "irb": "guidance",
    "case": "case",
}


def map_crosswalk_edges(edges: Sequence[Tuple[str, str, str, float]]) -> Dict[str, List[str]]:
    """Return mapping from document id to cited identifiers."""

    adjacency: Dict[str, List[str]] = {}
    for source, target, _relation, _confidence in edges:
        adjacency.setdefault(source, []).append(target)
    return adjacency


def compute_pairs(
    records: Dict[str, List[Dict[str, object]]],
    edges: Sequence[Tuple[str, str, str, float]],
) -> List[Dict[str, object]]:
    """Score cross-layer pairs using simplified feature extraction."""

    pairs: List[Dict[str, object]] = []
    adjacency = map_crosswalk_edges(edges)
    for statute in records["usc"]:
        for regulation in records["cfr"]:
            shared = set(adjacency.get(statute["identifier"], [])) & set(
                adjacency.get(regulation["identifier"], [])
            )
            citation_union = set(adjacency.get(statute["identifier"], [])) | set(
                adjacency.get(regulation["identifier"], [])
            )
            features = ContradictionFeatures(
                cosine_similarity=cosine_similarity(statute["text"], regulation["text"]),
                negation_overlap=negation_overlap(statute["text"], regulation["text"]),
                numeric_conflict=numeric_conflict(statute["text"], regulation["text"]),
                shared_citations=list(shared),
                citation_total=max(1, len(citation_union)),
                layer_a=LAYER_LABELS["usc"],
                layer_b=LAYER_LABELS["cfr"],
                newer_date=dt.date.fromisoformat(statute["snapshot_date"]),
                older_date=dt.date.fromisoformat(regulation["snapshot_date"]),
            )
            probability = contradiction_probability(features)
            pairs.append(
                {
                    "source": statute["identifier"],
                    "target": regulation["identifier"],
                    "score": round(probability, 6),
                    "snippet_a": statute["text"][:200],
                    "snippet_b": regulation["text"][:200],
                    "shared_citations": list(shared),
                    "cosine_similarity": features.cosine_similarity,
                    "negation_overlap": features.negation_overlap,
                    "numeric_conflict": features.numeric_conflict,
                }
            )
    pairs.sort(key=lambda item: item["score"], reverse=True)
    return pairs


def compute_section_metrics(
    records: Dict[str, List[Dict[str, object]]],
    edges: Sequence[Tuple[str, str, str, float]],
    pair_scores: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Aggregate per-section metrics from pair scores and case links."""

    metrics: List[Dict[str, object]] = []
    edge_map = map_crosswalk_edges(edges)
    inverse_edges: Dict[str, List[str]] = {}
    for source, targets in edge_map.items():
        for target in targets:
            inverse_edges.setdefault(target, []).append(source)

    graph_edges = [(source, target) for source, target, _relation, _confidence in edges]
    network_scores = compute_network_scores(graph_edges)

    pair_lookup: Dict[str, List[float]] = {}
    for pair in pair_scores:
        pair_lookup.setdefault(pair["source"], []).append(pair["score"])
        pair_lookup.setdefault(pair["target"], []).append(pair["score"])

    for layer, docs in records.items():
        for doc in docs:
            related_cases = [
                records_by_id[source]
                for source in inverse_edges.get(doc["identifier"], [])
                if source in records_by_id
            ]
            interpretations: Dict[str, int] = {}
            outcome_values: List[float] = []
            circuits: List[str] = []
            for case in related_cases:
                for tag in case.get("tags", []):
                    interpretations[tag] = interpretations.get(tag, 0) + 1
                outcome_values.extend(case.get("outcomes", []))
                circuits.extend(case.get("circuits", []))

            ambiguity = normalized_entropy(interpretations.values()) if interpretations else 0.0
            judicial = judicial_uncertainty(
                outcome_values,
                split_circuits=len(set(circuits)),
                total_circuits=max(1, len(circuits)),
            ) if outcome_values else 0.0
            contradiction_avg = sum(pair_lookup.get(doc["identifier"], [0.0])) / max(
                1, len(pair_lookup.get(doc["identifier"], []))
            )
            fragility = network_scores.get(doc["identifier"], 0.0)
            reach = len(edge_map.get(doc["identifier"], [])) + len(inverse_edges.get(doc["identifier"], []))
            stakes = 0.7 if "income" in doc.get("tags", []) else 0.5
            impact = impact_weight(reach, stakes)
            flaw = flaw_score(ambiguity, contradiction_avg, judicial, fragility)
            policy = policy_flaw_score(flaw, impact)

            metrics.append(
                {
                    "identifier": doc["identifier"],
                    "layer": layer,
                    "heading": doc.get("heading"),
                    "url": doc.get("url"),
                    "snapshot_date": doc.get("snapshot_date"),
                    "ambiguity": round(ambiguity, 6),
                    "contradiction": round(contradiction_avg, 6),
                    "judicial_uncertainty": round(judicial, 6),
                    "fragility": round(fragility, 6),
                    "impact": round(impact, 6),
                    "flaw": round(flaw, 6),
                    "policy_flaw": round(policy, 6),
                }
            )
    metrics.sort(key=lambda item: item["policy_flaw"], reverse=True)
    return metrics


records_by_id: Dict[str, Dict[str, object]] = {}


def load_records(records: Dict[str, List[Dict[str, object]]]) -> None:
    """Populate the global id lookup for downstream helpers."""

    records_by_id.clear()
    for docs in records.values():
        for doc in docs:
            records_by_id[doc["identifier"]] = doc


def write_pairs(pairs: Iterable[Dict[str, object]]) -> None:
    PAIRS_SCORED.unlink(missing_ok=True)
    with PAIRS_SCORED.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_section_metrics(metrics: Iterable[Dict[str, object]]) -> None:
    SECTION_METRICS.unlink(missing_ok=True)
    with SECTION_METRICS.open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_graph(edges: Sequence[Tuple[str, str, str, float]]) -> None:
    nodes = []
    seen = set()
    for layer_docs in records_by_id.values():
        if layer_docs["identifier"] not in seen:
            nodes.append({
                "id": layer_docs["identifier"],
                "layer": layer_docs.get("layer"),
                "heading": layer_docs.get("heading"),
            })
            seen.add(layer_docs["identifier"])

    payload = {
        "nodes": nodes,
        "edges": [
            {"source": source, "target": target, "relation": relation, "confidence": confidence}
            for source, target, relation, confidence in edges
        ],
    }
    GRAPH_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_time_series(metrics: Sequence[Dict[str, object]]) -> None:
    # For the simulated run we expose a single snapshot; full pipeline would append.
    aggregate_flaw = sum(item["flaw"] for item in metrics) / max(1, len(metrics))
    payload = {
        "snapshot": SNAPSHOT,
        "D": round(aggregate_flaw, 4),
        "kappa": round(1.0 - min(1.0, aggregate_flaw), 4),
        "notes": "Simulated snapshot; replace with longitudinal data after full ingestion.",
    }
    TIMESERIES_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_stub(metrics: Sequence[Dict[str, object]], pairs: Sequence[Dict[str, object]]) -> None:
    payload = {
        "generated_at": SNAPSHOT,
        "sections_total": len(metrics),
        "pairs_total": len(pairs),
        "top_sections": metrics[:5],
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_edge_tuple(edge) -> Tuple[str, str, str, float]:
    return edge.source_id, edge.target_id, edge.relation, float(edge.confidence)


def simulate_pipeline() -> None:
    ensure_scaffold()
    records = seed_ingestion()
    load_records(records)

    # Build crosswalk on the synthetic texts.
    CROSSWALK_CSV.unlink(missing_ok=True)
    edges = [format_edge_tuple(edge) for edge in build_crosswalk(confidence=0.97)]
    if edges:
        with CROSSWALK_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_id", "target_id", "relation", "confidence"])
            writer.writerows([[e[0], e[1], e[2], f"{e[3]:.3f}"] for e in edges])
    logger.info("Crosswalk edges generated: %d", len(edges))

    pair_scores = compute_pairs(records, edges)
    section_metrics = compute_section_metrics(records, edges, pair_scores)

    write_pairs(pair_scores)
    write_section_metrics(section_metrics)
    write_graph(edges)
    write_time_series(section_metrics)
    write_summary_stub(section_metrics, pair_scores)

    logger.info(
        "Simulated pipeline complete — sections=%d pairs=%d",
        len(section_metrics),
        len(pair_scores),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the CALE-LAW tax pipeline")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    simulate_pipeline()


if __name__ == "__main__":
    main()
