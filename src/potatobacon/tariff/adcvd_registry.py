"""AD/CVD (Antidumping / Countervailing Duty) order registry.

Loads active AD/CVD orders from ``data/overlays/adcvd_orders.json`` and
provides lookup by HTS code and origin country.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ADCVDOrder:
    """Single antidumping or countervailing duty order."""

    order_id: str
    order_type: str  # "AD" or "CVD"
    product_description: str
    hts_prefixes: tuple[str, ...]
    origin_countries: tuple[str, ...]
    duty_rate_pct: float
    effective_date: str
    status: str
    case_number: str
    federal_register_citation: str
    scope_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ADCVDOrderMatch:
    """Single AD/CVD order match with confidence metadata."""

    order: ADCVDOrder
    confidence: str  # "high" (8+ digit match), "high" (6+ digit), "medium" (4-digit), or "low"
    matched_prefix: str
    note: str


@dataclass(frozen=True)
class ADCVDLookupResult:
    """Result of an AD/CVD lookup for a specific HTS code + origin."""

    ad_orders: tuple[ADCVDOrder, ...]
    cvd_orders: tuple[ADCVDOrder, ...]
    total_ad_rate: float
    total_cvd_rate: float
    combined_rate: float
    has_exposure: bool
    confidence: str = "none"  # "high", "low", or "none"
    confidence_note: str = ""
    order_matches: tuple[ADCVDOrderMatch, ...] = ()


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "overlays" / "adcvd_orders.json"


def _full_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "overlays" / "adcvd_orders_full.json"


def _normalize_hts(code: str) -> str:
    """Strip dots/spaces and return digits only."""
    return "".join(ch for ch in str(code) if ch.isdigit())


def _normalize_country(code: str) -> str:
    return code.strip().upper()


def _matches_prefix(hts_digits: str, prefix: str) -> bool:
    prefix_digits = "".join(ch for ch in prefix if ch.isdigit())
    if not prefix_digits or not hts_digits:
        return False
    return hts_digits.startswith(prefix_digits)


class ADCVDRegistry:
    """Registry of active AD/CVD orders with lookup by HTS + origin."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        if data_path:
            self._orders = _load_orders(Path(data_path))
        else:
            # Try full database first, fall back to original
            full = _full_data_path()
            if full.exists():
                self._orders = _load_orders(full)
            else:
                self._orders = _load_orders(_default_data_path())

    @property
    def orders(self) -> tuple[ADCVDOrder, ...]:
        return self._orders

    def lookup(
        self,
        hts_code: str,
        origin_country: str,
    ) -> ADCVDLookupResult:
        """Find applicable AD/CVD orders for a given HTS code and origin country."""
        hts_digits = _normalize_hts(hts_code)
        origin_norm = _normalize_country(origin_country)

        ad_orders: list[ADCVDOrder] = []
        cvd_orders: list[ADCVDOrder] = []
        order_matches: list[ADCVDOrderMatch] = []

        for order in self._orders:
            if order.status != "active":
                continue
            if origin_norm not in order.origin_countries:
                continue
            matched_pfx: str | None = None
            for pfx in order.hts_prefixes:
                if _matches_prefix(hts_digits, pfx):
                    matched_pfx = pfx
                    break
            if matched_pfx is None:
                continue

            # Determine confidence based on prefix specificity
            pfx_digits = _normalize_hts(matched_pfx)
            if len(pfx_digits) >= 8:
                confidence = "high"
                note = f"Exact 8-digit match ({matched_pfx})"
            elif len(pfx_digits) >= 6:
                confidence = "high"
                note = f"Matched at {len(pfx_digits)}-digit level ({matched_pfx})"
            elif len(pfx_digits) >= 4:
                confidence = "medium"
                note = (
                    f"Heading-level match ({matched_pfx}, {len(pfx_digits)} digits) "
                    "— verify product is within order scope"
                )
            else:
                confidence = "low"
                note = (
                    f"Broad HTS prefix match ({matched_pfx}, {len(pfx_digits)} digits) "
                    "— verify product is within order scope"
                )

            match = ADCVDOrderMatch(
                order=order,
                confidence=confidence,
                matched_prefix=matched_pfx,
                note=note,
            )
            order_matches.append(match)

            if order.order_type == "AD":
                ad_orders.append(order)
            elif order.order_type == "CVD":
                cvd_orders.append(order)

        total_ad = sum(o.duty_rate_pct for o in ad_orders)
        total_cvd = sum(o.duty_rate_pct for o in cvd_orders)

        # Overall confidence is the lowest of any match
        if not order_matches:
            overall_confidence = "none"
            confidence_note = ""
        elif any(m.confidence == "low" for m in order_matches):
            overall_confidence = "low"
            low_notes = [m.note for m in order_matches if m.confidence == "low"]
            confidence_note = "; ".join(low_notes)
        elif any(m.confidence == "medium" for m in order_matches):
            overall_confidence = "medium"
            med_notes = [m.note for m in order_matches if m.confidence == "medium"]
            confidence_note = "; ".join(med_notes)
        else:
            overall_confidence = "high"
            confidence_note = "All matches at 6+ digit specificity"

        return ADCVDLookupResult(
            ad_orders=tuple(sorted(ad_orders, key=lambda o: o.order_id)),
            cvd_orders=tuple(sorted(cvd_orders, key=lambda o: o.order_id)),
            total_ad_rate=total_ad,
            total_cvd_rate=total_cvd,
            combined_rate=total_ad + total_cvd,
            has_exposure=bool(ad_orders or cvd_orders),
            confidence=overall_confidence,
            confidence_note=confidence_note,
            order_matches=tuple(order_matches),
        )

    def lookup_by_hts(self, hts_code: str) -> list[ADCVDOrder]:
        """Find all AD/CVD orders matching an HTS prefix regardless of origin."""
        hts_digits = _normalize_hts(hts_code)
        matched: list[ADCVDOrder] = []
        for order in self._orders:
            if order.status != "active":
                continue
            if any(_matches_prefix(hts_digits, pfx) for pfx in order.hts_prefixes):
                matched.append(order)
        return sorted(matched, key=lambda o: o.order_id)


def _load_orders(path: Path) -> tuple[ADCVDOrder, ...]:
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ()

    raw_orders = payload.get("orders", [])
    if not isinstance(raw_orders, list):
        return ()

    orders: list[ADCVDOrder] = []
    for entry in raw_orders:
        if not isinstance(entry, dict):
            continue
        orders.append(
            ADCVDOrder(
                order_id=str(entry.get("order_id", "")),
                order_type=str(entry.get("type", "AD")).upper(),
                product_description=str(entry.get("product_description", "")),
                hts_prefixes=tuple(str(p) for p in entry.get("hts_prefixes", [])),
                origin_countries=tuple(
                    _normalize_country(c) for c in entry.get("origin_countries", [])
                ),
                duty_rate_pct=float(entry.get("duty_rate_pct", 0.0)),
                effective_date=str(entry.get("effective_date", "")),
                status=str(entry.get("status", "active")),
                case_number=str(entry.get("case_number", "")),
                federal_register_citation=str(entry.get("federal_register_citation", "")),
                scope_keywords=tuple(
                    str(kw).lower() for kw in entry.get("scope_keywords", [])
                ),
            )
        )
    return tuple(sorted(orders, key=lambda o: (o.order_type, o.order_id)))


@lru_cache(maxsize=1)
def get_adcvd_registry(data_path: str | None = None) -> ADCVDRegistry:
    """Return a cached ADCVDRegistry instance."""
    return ADCVDRegistry(data_path)
