"""AD/CVD (Antidumping / Countervailing Duty) order registry.

Loads active AD/CVD orders and provides lookup by HTS code and origin country.
Includes optional keyword-based scope matching for imperfect HTS declarations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class ADCVDOrder:
    order_id: str
    order_type: str
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
    order: ADCVDOrder
    confidence: str
    matched_prefix: str
    note: str


@dataclass(frozen=True)
class ADCVDLookupResult:
    ad_orders: tuple[ADCVDOrder, ...]
    cvd_orders: tuple[ADCVDOrder, ...]
    total_ad_rate: float
    total_cvd_rate: float
    combined_rate: float
    has_exposure: bool
    confidence: str = "none"
    confidence_note: str = ""
    order_matches: tuple[ADCVDOrderMatch, ...] = ()


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "overlays" / "adcvd_orders.json"


def _full_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "overlays" / "adcvd_orders_full.json"


def _normalize_hts(code: str) -> str:
    return "".join(ch for ch in str(code) if ch.isdigit())


def _normalize_country(code: str) -> str:
    return code.strip().upper()


def _matches_prefix(hts_digits: str, prefix: str) -> bool:
    prefix_digits = _normalize_hts(prefix)
    if not prefix_digits or not hts_digits:
        return False
    return hts_digits.startswith(prefix_digits)


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


class ADCVDRegistry:
    def __init__(self, data_path: str | Path | None = None) -> None:
        if data_path:
            self._orders = _load_orders(Path(data_path))
        else:
            full = _full_data_path()
            if full.exists():
                self._orders = _load_orders(full)
            else:
                self._orders = _load_orders(_default_data_path())

    @property
    def orders(self) -> tuple[ADCVDOrder, ...]:
        return self._orders

    def lookup(self, hts_code: str, origin_country: str, description: str | None = None) -> ADCVDLookupResult:
        hts_digits = _normalize_hts(hts_code)
        origin_norm = _normalize_country(origin_country)
        desc_tokens = _tokens(description)

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

            keyword_hits = 0
            if desc_tokens and order.scope_keywords:
                for keyword in order.scope_keywords:
                    kw_tokens = _tokens(keyword)
                    if kw_tokens and kw_tokens.issubset(desc_tokens):
                        keyword_hits += 1
                    elif kw_tokens and (kw_tokens & desc_tokens):
                        keyword_hits += 1
            keyword_hits = min(keyword_hits, len(order.scope_keywords))

            if matched_pfx is None and keyword_hits < 2:
                continue

            if matched_pfx is None:
                confidence = "medium"
                note = f"Keyword-only scope match ({keyword_hits} keyword hits)"
            else:
                pfx_digits = _normalize_hts(matched_pfx)
                if len(pfx_digits) >= 8:
                    confidence = "high"
                    note = f"Exact 8-digit match ({matched_pfx})"
                elif len(pfx_digits) >= 6:
                    confidence = "high"
                    note = f"Matched at {len(pfx_digits)}-digit level ({matched_pfx})"
                elif len(pfx_digits) >= 4:
                    confidence = "medium"
                    note = f"Heading-level match ({matched_pfx}, {len(pfx_digits)} digits)"
                else:
                    confidence = "low"
                    note = f"Broad HTS prefix match ({matched_pfx}, {len(pfx_digits)} digits)"
                if keyword_hits >= 2:
                    confidence = "high"
                    note = f"{note}; keyword support ({keyword_hits} hits)"

            order_matches.append(
                ADCVDOrderMatch(
                    order=order,
                    confidence=confidence,
                    matched_prefix=matched_pfx or "",
                    note=note,
                )
            )
            if order.order_type == "AD":
                ad_orders.append(order)
            elif order.order_type == "CVD":
                cvd_orders.append(order)

        total_ad = sum(item.duty_rate_pct for item in ad_orders)
        total_cvd = sum(item.duty_rate_pct for item in cvd_orders)

        if not order_matches:
            overall_confidence = "none"
            confidence_note = ""
        elif any(item.confidence == "low" for item in order_matches):
            overall_confidence = "low"
            confidence_note = "; ".join(item.note for item in order_matches if item.confidence == "low")
        elif any(item.confidence == "medium" for item in order_matches):
            overall_confidence = "medium"
            confidence_note = "; ".join(item.note for item in order_matches if item.confidence == "medium")
        else:
            overall_confidence = "high"
            confidence_note = "All matches supported by 6+ digit HTS and/or scope keywords"

        return ADCVDLookupResult(
            ad_orders=tuple(sorted(ad_orders, key=lambda item: item.order_id)),
            cvd_orders=tuple(sorted(cvd_orders, key=lambda item: item.order_id)),
            total_ad_rate=total_ad,
            total_cvd_rate=total_cvd,
            combined_rate=total_ad + total_cvd,
            has_exposure=bool(ad_orders or cvd_orders),
            confidence=overall_confidence,
            confidence_note=confidence_note,
            order_matches=tuple(order_matches),
        )

    def lookup_by_hts(self, hts_code: str) -> list[ADCVDOrder]:
        hts_digits = _normalize_hts(hts_code)
        matched: list[ADCVDOrder] = []
        for order in self._orders:
            if order.status != "active":
                continue
            if any(
                _matches_prefix(hts_digits, pfx) or _normalize_hts(pfx).startswith(hts_digits)
                for pfx in order.hts_prefixes
            ):
                matched.append(order)
        return sorted(matched, key=lambda item: item.order_id)


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
                hts_prefixes=tuple(str(item) for item in entry.get("hts_prefixes", [])),
                origin_countries=tuple(_normalize_country(item) for item in entry.get("origin_countries", [])),
                duty_rate_pct=float(entry.get("duty_rate_pct", 0.0)),
                effective_date=str(entry.get("effective_date", "")),
                status=str(entry.get("status", "active")),
                case_number=str(entry.get("case_number", "")),
                federal_register_citation=str(entry.get("federal_register_citation", "")),
                scope_keywords=tuple(str(item).lower() for item in entry.get("scope_keywords", [])),
            )
        )
    return tuple(sorted(orders, key=lambda item: (item.order_type, item.order_id)))


@lru_cache(maxsize=1)
def get_adcvd_registry(data_path: str | None = None) -> ADCVDRegistry:
    return ADCVDRegistry(data_path)
