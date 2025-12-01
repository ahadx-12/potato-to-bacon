from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from potatobacon.cale.bootstrap import CALEServices
from potatobacon.cale.suggest import AmendmentSuggester
from potatobacon.cale.types import LegalRule
from potatobacon.storage import load_manifest, save_manifest, latest_manifest_hash


@dataclass
class LawSource:
    id: str
    text: str
    jurisdiction: str = "Unknown"
    statute: str = ""
    section: str = ""
    enactment_year: int = 2000


def _serialize_rule(rule: LegalRule) -> Dict[str, Any]:
    payload = asdict(rule)
    # Convert numpy arrays to lists for JSON serialisation
    for key, value in list(payload.items()):
        if hasattr(value, "tolist"):
            payload[key] = value.tolist()
    return payload


def _hydrate_rules(services: CALEServices, sources: Iterable[LawSource]) -> List[LegalRule]:
    rules: List[LegalRule] = []
    for source in sources:
        metadata = {
            "jurisdiction": source.jurisdiction,
            "statute": source.statute or source.id,
            "section": source.section or "",
            "enactment_year": int(source.enactment_year),
            "id": source.id,
        }
        try:
            parsed = services.parser.parse(source.text, metadata)
        except ValueError:
            synthetic_text = f"{source.text.strip()} MUST apply"
            parsed = services.parser.parse(synthetic_text, metadata)
        rules.append(services.feature_engine.populate(parsed))
    return rules


def _merge_sources(existing: List[Dict[str, Any]], new_sources: Iterable[LawSource]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {src.get("id"): dict(src) for src in existing}
    for source in new_sources:
        merged[source.id] = {
            "id": source.id,
            "text": source.text,
            "jurisdiction": source.jurisdiction,
            "statute": source.statute,
            "section": source.section,
            "enactment_year": source.enactment_year,
        }
    return list(merged.values())


def load_latest_law_manifest(domain: str | None = None) -> Dict[str, Any] | None:
    manifest_hash = latest_manifest_hash(domain)
    if manifest_hash is None:
        return None
    try:
        return load_manifest(manifest_hash)
    except FileNotFoundError:
        return None


def ingest_sources(
    services: CALEServices,
    domain: str,
    sources: List[LawSource],
    replace_existing: bool = False,
) -> Dict[str, Any]:
    """Parse raw law sources into a manifest and refresh CALE services."""

    existing_manifest = load_latest_law_manifest(domain)
    existing_sources = existing_manifest.get("sources", []) if existing_manifest else []

    combined_sources = _merge_sources([] if replace_existing else existing_sources, sources)
    rules = _hydrate_rules(services, [LawSource(**src) if isinstance(src, dict) else src for src in combined_sources])

    manifest_payload = {
        "type": "law_manifest",
        "domain": domain,
        "sources": combined_sources,
        "rules": [_serialize_rule(rule) for rule in rules],
    }
    manifest_hash = save_manifest(manifest_payload)

    # Refresh the services corpus and suggester to use the new manifest
    services.corpus = rules
    services.suggester = AmendmentSuggester(
        rule_corpus=rules,
        embedder=services.embedder,
        ccs_calculator=services.calculator,
        predicate_mapper=services.mapper,
    )

    return {
        "manifest_hash": manifest_hash,
        "rules_count": len(rules),
        "domain": domain,
        "sources_ingested": [src["id"] for src in combined_sources],
    }

