from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


def parse_duty_rate(raw: Any) -> float | None:
    """Parse a duty rate expressed as a percentage string or number."""

    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        raise ValueError(f"Unsupported duty rate type: {type(raw)}")

    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    if cleaned in {"free", "0", "0%"}:
        return 0.0
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace("percent", "").strip()
    return float(cleaned)


@dataclass
class TariffLine:
    source_id: str
    hts_code: str
    description: str
    duty_rate: float | None
    effective_date: str
    chapter: str
    heading: str
    note_id: Optional[str]
    source_ref: str
    guard_tokens: List[str]
    jurisdiction: str = "US"
    subject: str = "import_duty"
    statute: str = "HTSUS"
    rule_type: str = "TARIFF"
    modality: str = "OBLIGE"
    action: Optional[str] = None
    rate_applies: bool = True

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TariffLine":
        try:
            duty_rate = parse_duty_rate(payload.get("duty_rate"))
        except Exception as exc:  # pragma: no cover - defensive validation
            raise ValueError(f"Invalid duty rate for {payload.get('source_id')}: {exc}") from exc

        return cls(
            source_id=str(payload["source_id"]),
            hts_code=str(payload["hts_code"]),
            description=str(payload["description"]),
            duty_rate=duty_rate,
            effective_date=str(payload.get("effective_date") or "2025-01-01"),
            chapter=str(payload.get("chapter") or str(payload.get("hts_code"))[:2]),
            heading=str(payload.get("heading") or str(payload.get("hts_code"))[:4]),
            note_id=payload.get("note_id"),
            source_ref=str(payload.get("source_ref") or payload.get("statute") or "HTSUS Extract"),
            guard_tokens=list(payload.get("guard_tokens") or []),
            jurisdiction=str(payload.get("jurisdiction") or "US"),
            subject=str(payload.get("subject") or "import_duty"),
            statute=str(payload.get("statute") or "HTSUS"),
            rule_type=str(payload.get("rule_type") or "TARIFF"),
            modality=str(payload.get("modality") or "OBLIGE"),
            action=payload.get("action"),
            rate_applies=bool(payload.get("rate_applies", True)),
        )

    def citation(self) -> Dict[str, Any]:
        return {
            "statute": self.statute,
            "chapter": self.chapter,
            "heading": self.heading,
            "note_id": self.note_id,
            "source_ref": self.source_ref,
        }


@dataclass
class TariffNote:
    note_id: str
    text: str
    chapter: str
    heading: Optional[str]
    guard_tokens: List[str]
    source_ref: str
    statute: str = "HTSUS"
    modality: str = "PERMIT"
    action: Optional[str] = None
    effective_date: str = "2025-01-01"
    rate_applies: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TariffNote":
        return cls(
            note_id=str(payload["note_id"]),
            text=str(payload["text"]),
            chapter=str(payload.get("chapter") or ""),
            heading=payload.get("heading"),
            guard_tokens=list(payload.get("guard_tokens") or []),
            source_ref=str(payload.get("source_ref") or payload.get("statute") or "HTSUS Note"),
            statute=str(payload.get("statute") or "HTSUS"),
            modality=str(payload.get("modality") or "PERMIT"),
            action=payload.get("action"),
            effective_date=str(payload.get("effective_date") or "2025-01-01"),
            rate_applies=bool(payload.get("rate_applies", False)),
        )

    def citation(self) -> Dict[str, Any]:
        return {
            "statute": self.statute,
            "chapter": self.chapter,
            "heading": self.heading,
            "note_id": self.note_id,
            "source_ref": self.source_ref,
        }
