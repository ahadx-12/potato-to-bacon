from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from potatobacon.tariff.models import TariffOverlayResultModel


@dataclass(frozen=True)
class _OverlayRule:
    overlay_name: str
    hts_prefixes: tuple[str, ...]
    additional_rate: float
    reason: str
    requires_review: bool = False
    stop_optimization: bool = False
    origin_countries: tuple[str, ...] = ()
    import_countries: tuple[str, ...] = ()
    match_level: str = ""  # exact_8digit | heading_fallback | ""


def _data_root(base_path: str | None = None) -> Path:
    if base_path:
        return Path(base_path)
    return Path(__file__).resolve().parents[3] / "data" / "overlays"


def _load_overlay_payload(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        entries = payload.get("overlays") or []
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _normalize_country(value: str | None) -> str | None:
    return value.upper() if isinstance(value, str) and value.strip() else None


def _normalize_numeric(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits


def _normalize_text(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum())
    return cleaned.upper()


def _normalize_rate_percent(value: Any) -> float:
    """Normalize overlay rates to percentage points.

    Data feeds may encode rates as either `25` (percentage points) or
    `0.25` (fraction). This helper standardizes to percentage points.
    """
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < rate <= 1.0:
        return rate * 100.0
    return rate


@lru_cache(maxsize=None)
def _load_overlay_rules(base_path: str | None) -> tuple[_OverlayRule, ...]:
    base_dir = _data_root(base_path)
    rules: list[_OverlayRule] = []

    # Section 232 — always load from sample
    for path in sorted(base_dir.glob("section232_sample.json")):
        for entry in _load_overlay_payload(path):
            prefixes = tuple(str(prefix) for prefix in entry.get("hts_prefixes", []))
            rules.append(
                _OverlayRule(
                    overlay_name=str(entry.get("overlay_name") or entry.get("name") or "unknown"),
                    hts_prefixes=prefixes,
                    additional_rate=_normalize_rate_percent(entry.get("additional_rate") or 0.0),
                    reason=str(entry.get("reason") or "unspecified"),
                    requires_review=bool(entry.get("requires_review", False)),
                    stop_optimization=bool(entry.get("stop_optimization", False)),
                    origin_countries=tuple(
                        _normalize_country(country)
                        for country in entry.get("origin_countries", [])
                        if _normalize_country(country)
                    ),
                    import_countries=tuple(
                        _normalize_country(country)
                        for country in entry.get("import_countries", [])
                        if _normalize_country(country)
                    ),
                )
            )

    # Section 301 — prefer subheading-level (8-digit), fall back to heading-level
    subheading_path = base_dir / "section301_subheading.json"
    heading_path = base_dir / "section301_sample.json"
    loaded_301_prefixes: set[str] = set()

    if subheading_path.exists():
        for entry in _load_overlay_payload(subheading_path):
            prefixes = tuple(str(prefix) for prefix in entry.get("hts_prefixes", []))
            for pfx in prefixes:
                loaded_301_prefixes.add(_normalize_numeric(pfx)[:4])
            rules.append(
                _OverlayRule(
                    overlay_name=str(entry.get("overlay_name") or entry.get("name") or "unknown"),
                    hts_prefixes=prefixes,
                    additional_rate=_normalize_rate_percent(entry.get("additional_rate") or 0.0),
                    reason=str(entry.get("reason") or "unspecified"),
                    requires_review=bool(entry.get("requires_review", False)),
                    stop_optimization=bool(entry.get("stop_optimization", False)),
                    origin_countries=tuple(
                        _normalize_country(country)
                        for country in entry.get("origin_countries", [])
                        if _normalize_country(country)
                    ),
                    import_countries=tuple(
                        _normalize_country(country)
                        for country in entry.get("import_countries", [])
                        if _normalize_country(country)
                    ),
                    match_level=str(entry.get("match_level", "exact_8digit")),
                )
            )

    # Load heading-level 301 as fallback for chapters not covered at 8-digit
    if heading_path.exists():
        for entry in _load_overlay_payload(heading_path):
            prefixes = tuple(str(prefix) for prefix in entry.get("hts_prefixes", []))
            # Only add heading-level rules for headings not already covered by 8-digit
            uncovered = [
                pfx for pfx in prefixes
                if _normalize_numeric(pfx)[:4] not in loaded_301_prefixes
            ]
            if not uncovered and loaded_301_prefixes:
                continue
            use_prefixes = tuple(uncovered) if uncovered else prefixes
            rules.append(
                _OverlayRule(
                    overlay_name=str(entry.get("overlay_name") or entry.get("name") or "unknown"),
                    hts_prefixes=use_prefixes,
                    additional_rate=_normalize_rate_percent(entry.get("additional_rate") or 0.0),
                    reason=str(entry.get("reason") or "unspecified"),
                    requires_review=bool(entry.get("requires_review", False)),
                    stop_optimization=bool(entry.get("stop_optimization", False)),
                    origin_countries=tuple(
                        _normalize_country(country)
                        for country in entry.get("origin_countries", [])
                        if _normalize_country(country)
                    ),
                    import_countries=tuple(
                        _normalize_country(country)
                        for country in entry.get("import_countries", [])
                        if _normalize_country(country)
                    ),
                    match_level="heading_fallback",
                )
            )

    rules.sort(key=lambda rule: (rule.overlay_name, rule.additional_rate, rule.reason))
    return tuple(rules)



def _candidate_codes(
    explicit_hts_code: str | None,
    *,
    facts: Mapping[str, Any] | None,
    active_codes: Iterable[str],
) -> list[str]:
    candidates: list[str] = []
    if explicit_hts_code:
        candidates.append(str(explicit_hts_code))
    fact_candidates = []
    if facts:
        for key in ("hts_code", "hts_heading", "hts"):
            value = facts.get(key)
            if isinstance(value, str):
                fact_candidates.append(value)
    candidates.extend(fact_candidates)
    for code in active_codes:
        candidates.append(code)
        if code.startswith("HTS_"):
            candidates.append(code[4:])
    normalized: list[str] = []
    for code in candidates:
        normalized_code = _normalize_text(str(code))
        if normalized_code and normalized_code not in normalized:
            normalized.append(normalized_code)
    return normalized


def _matches_prefix(code: str, prefix: str) -> bool:
    normalized_prefix_text = _normalize_text(prefix)
    normalized_prefix_digits = _normalize_numeric(prefix)
    code_digits = _normalize_numeric(code)
    if normalized_prefix_digits and code_digits:
        if code_digits.startswith(normalized_prefix_digits):
            return True
    if normalized_prefix_text and code.startswith(normalized_prefix_text):
        return True
    return False


def evaluate_overlays(
    *,
    facts: Mapping[str, Any] | None,
    active_codes: Sequence[str] | None,
    origin_country: str | None = None,
    import_country: str | None = None,
    hts_code: str | None = None,
    data_root: str | None = None,
) -> list[TariffOverlayResultModel]:
    """Return applied tariff overlays for a scenario.

    Overlays are matched using HTS prefixes and optional origin/import guards.
    """

    normalized_origin = _normalize_country(origin_country)
    normalized_import = _normalize_country(import_country)
    facts = facts or {}
    if normalized_origin is None:
        for key, value in facts.items():
            if key.startswith("origin_country_") and value:
                normalized_origin = _normalize_country(key.split("origin_country_")[-1])
                break
        if normalized_origin is None:
            raw_origin = facts.get("origin_country_raw")
            normalized_origin = _normalize_country(raw_origin if isinstance(raw_origin, str) else None)

    codes = _candidate_codes(hts_code, facts=facts, active_codes=active_codes or [])
    overlays: list[TariffOverlayResultModel] = []
    for rule in _load_overlay_rules(data_root):
        if rule.origin_countries and (normalized_origin is None or normalized_origin not in rule.origin_countries):
            continue
        if rule.import_countries and (normalized_import is None or normalized_import not in rule.import_countries):
            continue
        if not rule.hts_prefixes:
            continue
        if not any(_matches_prefix(code, prefix) for code in codes for prefix in rule.hts_prefixes):
            continue
        overlays.append(
            TariffOverlayResultModel(
                overlay_name=rule.overlay_name,
                applies=True,
                additional_rate=rule.additional_rate,
                reason=rule.reason,
                requires_review=rule.requires_review,
                stop_optimization=rule.stop_optimization,
                match_level=rule.match_level,
            )
        )

    overlays.sort(key=lambda item: (item.overlay_name, item.additional_rate))
    return overlays


def effective_duty_rate(duty_rate: float | None, overlays: Sequence[TariffOverlayResultModel]) -> float:
    base = duty_rate if duty_rate is not None else 0.0
    additive = sum(item.additional_rate for item in overlays if item.applies)
    return base + additive
