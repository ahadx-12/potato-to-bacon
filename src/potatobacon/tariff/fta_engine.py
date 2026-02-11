"""FTA (Free Trade Agreement) preference engine.

Evaluates whether a product qualifies for preferential duty treatment
under applicable FTA programs based on origin country, HTS code, and
product characteristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProductSpecificRule:
    """Product-specific rule within an FTA program."""

    hts_prefixes: tuple[str, ...]
    description: str
    rvc_threshold_pct: float
    additional_requirements: tuple[str, ...]


@dataclass(frozen=True)
class FTAProgram:
    """Single FTA or preference program."""

    program_id: str
    name: str
    partner_countries: tuple[str, ...]
    import_country: str
    preference_type: str  # "FTA", "GSP"
    default_preference_pct: float
    requires_certificate_of_origin: bool
    rvc_threshold_pct: float
    tariff_shift_required: bool
    effective_date: str
    status: str
    product_specific_rules: tuple[ProductSpecificRule, ...] = ()
    excluded_hts_prefixes: tuple[str, ...] = ()
    covered_hts_prefixes: tuple[str, ...] = ()  # empty = all covered
    notes: str = ""


@dataclass(frozen=True)
class FTAEligibilityResult:
    """Result of evaluating FTA eligibility for a single program."""

    program_id: str
    program_name: str
    eligible: bool
    preference_pct: float  # 0-100; 100 = full duty elimination
    reasons: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    rvc_threshold_pct: float
    tariff_shift_required: bool
    requires_certificate_of_origin: bool
    product_specific_rule_applied: str | None = None


@dataclass(frozen=True)
class FTALookupResult:
    """Result of checking all applicable FTA programs."""

    eligible_programs: tuple[FTAEligibilityResult, ...]
    ineligible_programs: tuple[FTAEligibilityResult, ...]
    best_program: FTAEligibilityResult | None
    best_preference_pct: float
    has_eligible_program: bool


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "overlays" / "fta_preferences.json"


def _normalize_hts(code: str) -> str:
    return "".join(ch for ch in str(code) if ch.isdigit())


def _normalize_country(code: str) -> str:
    return code.strip().upper()


def _matches_prefix(hts_digits: str, prefix: str) -> bool:
    prefix_digits = "".join(ch for ch in prefix if ch.isdigit())
    if not prefix_digits or not hts_digits:
        return False
    return hts_digits.startswith(prefix_digits)


def _evaluate_single_program(
    program: FTAProgram,
    hts_code: str,
    origin_country: str,
    import_country: str,
    facts: Mapping[str, Any],
) -> FTAEligibilityResult:
    """Evaluate eligibility under a single FTA program."""
    hts_digits = _normalize_hts(hts_code)
    origin_norm = _normalize_country(origin_country)
    import_norm = _normalize_country(import_country)

    reasons: list[str] = []
    missing: list[str] = []

    # Check program status
    if program.status != "active":
        return FTAEligibilityResult(
            program_id=program.program_id,
            program_name=program.name,
            eligible=False,
            preference_pct=0.0,
            reasons=("Program is not active",),
            missing_requirements=(),
            rvc_threshold_pct=program.rvc_threshold_pct,
            tariff_shift_required=program.tariff_shift_required,
            requires_certificate_of_origin=program.requires_certificate_of_origin,
        )

    # Check import country
    if program.import_country and import_norm != _normalize_country(program.import_country):
        return FTAEligibilityResult(
            program_id=program.program_id,
            program_name=program.name,
            eligible=False,
            preference_pct=0.0,
            reasons=(f"Import country {import_norm} not covered by {program.program_id}",),
            missing_requirements=(),
            rvc_threshold_pct=program.rvc_threshold_pct,
            tariff_shift_required=program.tariff_shift_required,
            requires_certificate_of_origin=program.requires_certificate_of_origin,
        )

    # Check origin country is a partner
    if origin_norm not in program.partner_countries:
        return FTAEligibilityResult(
            program_id=program.program_id,
            program_name=program.name,
            eligible=False,
            preference_pct=0.0,
            reasons=(f"Origin country {origin_norm} is not a {program.program_id} partner",),
            missing_requirements=(),
            rvc_threshold_pct=program.rvc_threshold_pct,
            tariff_shift_required=program.tariff_shift_required,
            requires_certificate_of_origin=program.requires_certificate_of_origin,
        )

    # Check if HTS is excluded (GSP has excluded prefixes)
    if program.excluded_hts_prefixes and hts_digits:
        if any(_matches_prefix(hts_digits, pfx) for pfx in program.excluded_hts_prefixes):
            return FTAEligibilityResult(
                program_id=program.program_id,
                program_name=program.name,
                eligible=False,
                preference_pct=0.0,
                reasons=(f"HTS {hts_code} is excluded from {program.program_id}",),
                missing_requirements=(),
                rvc_threshold_pct=program.rvc_threshold_pct,
                tariff_shift_required=program.tariff_shift_required,
                requires_certificate_of_origin=program.requires_certificate_of_origin,
            )

    # Check if product is covered (some FTAs only cover specific HTS)
    if program.covered_hts_prefixes and hts_digits:
        if not any(_matches_prefix(hts_digits, pfx) for pfx in program.covered_hts_prefixes):
            return FTAEligibilityResult(
                program_id=program.program_id,
                program_name=program.name,
                eligible=False,
                preference_pct=0.0,
                reasons=(f"HTS {hts_code} is not covered by {program.program_id}",),
                missing_requirements=(),
                rvc_threshold_pct=program.rvc_threshold_pct,
                tariff_shift_required=program.tariff_shift_required,
                requires_certificate_of_origin=program.requires_certificate_of_origin,
            )

    reasons.append(f"Origin {origin_norm} is a {program.program_id} partner country")

    # Find product-specific rule if applicable
    psr_applied: str | None = None
    rvc_threshold = program.rvc_threshold_pct
    additional_reqs: list[str] = []

    for psr in program.product_specific_rules:
        if any(_matches_prefix(hts_digits, pfx) for pfx in psr.hts_prefixes):
            psr_applied = psr.description
            rvc_threshold = psr.rvc_threshold_pct
            additional_reqs = list(psr.additional_requirements)
            reasons.append(f"Product-specific rule applies: {psr.description}")
            break

    # Check RVC if available in facts
    rvc_value = facts.get("origin_rvc_build_down") or facts.get("origin_rvc_build_up")
    if rvc_threshold > 0 and rvc_value is not None:
        if float(rvc_value) >= rvc_threshold:
            reasons.append(f"RVC {rvc_value}% meets {rvc_threshold}% threshold")
        else:
            missing.append(f"RVC {rvc_value}% below {rvc_threshold}% threshold")
    elif rvc_threshold > 0:
        missing.append(f"RVC documentation required (threshold: {rvc_threshold}%)")

    # Check tariff shift requirement
    if program.tariff_shift_required:
        has_shift = facts.get("origin_tariff_shift")
        if has_shift:
            reasons.append("Tariff shift requirement met")
        elif has_shift is False:
            missing.append("Tariff shift requirement not met")
        else:
            missing.append("Tariff shift documentation required")

    # Check certificate of origin
    if program.requires_certificate_of_origin:
        has_cert = facts.get("has_certificate_of_origin", False)
        if not has_cert:
            missing.append(f"Certificate of origin required for {program.program_id}")

    # Check additional requirements from product-specific rules
    for req in additional_reqs:
        fact_val = facts.get(req)
        if fact_val is False:
            missing.append(f"Additional requirement failed: {req}")
        elif not fact_val:
            missing.append(f"Additional requirement documentation needed: {req}")

    # Determine eligibility - eligible if origin matches and no hard blockers
    # Missing documentation doesn't block eligibility; it flags requirements
    # Only explicit failures ("failed", "not met", "below") are hard blockers
    hard_blockers = [m for m in missing if "not met" in m.lower() or "failed" in m.lower() or "below" in m.lower()]
    eligible = len(hard_blockers) == 0
    preference_pct = program.default_preference_pct if eligible else 0.0

    return FTAEligibilityResult(
        program_id=program.program_id,
        program_name=program.name,
        eligible=eligible,
        preference_pct=preference_pct,
        reasons=tuple(reasons),
        missing_requirements=tuple(missing),
        rvc_threshold_pct=rvc_threshold,
        tariff_shift_required=program.tariff_shift_required,
        requires_certificate_of_origin=program.requires_certificate_of_origin,
        product_specific_rule_applied=psr_applied,
    )


class FTAPreferenceEngine:
    """Evaluates FTA preference eligibility across all available programs."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        path = Path(data_path) if data_path else _default_data_path()
        self._programs = _load_programs(path)

    @property
    def programs(self) -> tuple[FTAProgram, ...]:
        return self._programs

    def evaluate(
        self,
        hts_code: str,
        origin_country: str,
        import_country: str = "US",
        facts: Mapping[str, Any] | None = None,
    ) -> FTALookupResult:
        """Evaluate eligibility across all FTA programs."""
        facts = facts or {}

        eligible: list[FTAEligibilityResult] = []
        ineligible: list[FTAEligibilityResult] = []

        for program in self._programs:
            result = _evaluate_single_program(
                program, hts_code, origin_country, import_country, facts
            )
            if result.eligible:
                eligible.append(result)
            else:
                ineligible.append(result)

        eligible.sort(key=lambda r: (-r.preference_pct, r.program_id))
        ineligible.sort(key=lambda r: r.program_id)

        best = eligible[0] if eligible else None
        best_pct = best.preference_pct if best else 0.0

        return FTALookupResult(
            eligible_programs=tuple(eligible),
            ineligible_programs=tuple(ineligible),
            best_program=best,
            best_preference_pct=best_pct,
            has_eligible_program=bool(eligible),
        )

    def find_programs_for_country(self, origin_country: str) -> list[FTAProgram]:
        """Find all FTA programs that cover a given origin country."""
        origin_norm = _normalize_country(origin_country)
        return [p for p in self._programs if origin_norm in p.partner_countries and p.status == "active"]


def _load_programs(path: Path) -> tuple[FTAProgram, ...]:
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ()

    raw = payload.get("programs", [])
    if not isinstance(raw, list):
        return ()

    programs: list[FTAProgram] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        psr_list: list[ProductSpecificRule] = []
        for rule in entry.get("product_specific_rules", []):
            if isinstance(rule, dict):
                psr_list.append(
                    ProductSpecificRule(
                        hts_prefixes=tuple(str(p) for p in rule.get("hts_prefixes", [])),
                        description=str(rule.get("description", "")),
                        rvc_threshold_pct=float(rule.get("rvc_threshold_pct", 0.0)),
                        additional_requirements=tuple(
                            str(r) for r in rule.get("additional_requirements", [])
                        ),
                    )
                )

        programs.append(
            FTAProgram(
                program_id=str(entry.get("program_id", "")),
                name=str(entry.get("name", "")),
                partner_countries=tuple(
                    _normalize_country(c) for c in entry.get("partner_countries", [])
                ),
                import_country=_normalize_country(str(entry.get("import_country", "US"))),
                preference_type=str(entry.get("preference_type", "FTA")),
                default_preference_pct=float(entry.get("default_preference_pct", 0.0)),
                requires_certificate_of_origin=bool(entry.get("requires_certificate_of_origin", True)),
                rvc_threshold_pct=float(entry.get("rvc_threshold_pct", 0.0)),
                tariff_shift_required=bool(entry.get("tariff_shift_required", False)),
                effective_date=str(entry.get("effective_date", "")),
                status=str(entry.get("status", "active")),
                product_specific_rules=tuple(psr_list),
                excluded_hts_prefixes=tuple(
                    str(p) for p in entry.get("excluded_hts_prefixes", [])
                ),
                covered_hts_prefixes=tuple(
                    str(p) for p in entry.get("covered_hts_prefixes", [])
                ),
                notes=str(entry.get("notes", "")),
            )
        )
    return tuple(sorted(programs, key=lambda p: p.program_id))


@lru_cache(maxsize=1)
def get_fta_engine(data_path: str | None = None) -> FTAPreferenceEngine:
    """Return a cached FTAPreferenceEngine instance."""
    return FTAPreferenceEngine(data_path)
