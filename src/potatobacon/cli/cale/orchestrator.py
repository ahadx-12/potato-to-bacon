"""High level orchestration for the CALE validation CLI."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import click

from ...cale.bootstrap import CALEServices, build_services
from ...cale.engine import CALEEngine
from ...cale.finance.docio import Doc, load_doc
from ...cale.finance.numeric import extract_numeric_covenants
from ...cale.finance.sectionizer import find_sections
from .cache import CacheManager, CachingLegalEmbedder

try:  # pragma: no cover - available during repo development
    from tools.sec_fetch import (
        TICKER_TO_CIK,
        ensure_filing_html,
        load_submissions,
        pick_last_10k_10q_before,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - fallback when packaged
    raise RuntimeError(
        "SEC helper utilities are unavailable. Ensure the repository root is on sys.path."
    ) from exc

try:  # pragma: no cover - available during repo development
    from tools.finance_extract import extract_pairs_from_html
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "Finance extraction helpers are unavailable. Ensure the repository root is on sys.path."
    ) from exc

LOGGER = logging.getLogger(__name__)

DEFAULT_TICKERS = ("MPW", "BBBY", "UPST", "AAPL", "MSFT", "JNJ")
DEFAULT_USER_AGENT = "CALE-CLI/0.9 (contact: you@example.com)"


@dataclass
class FilingManifestEntry:
    ticker: str
    cik: str
    form: str
    filed: str
    accession: str
    primary_document: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "form": self.form,
            "filed": self.filed,
            "accession": self.accession,
            "primary_document": self.primary_document,
            "path": self.path,
        }


@dataclass
class FinanceSectionSummary:
    title: str
    score: float
    anchor: Optional[str]
    doc_kind: str
    covenant_count: int
    sample_covenants: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "score": self.score,
            "anchor": self.anchor,
            "doc_kind": self.doc_kind,
            "covenant_count": self.covenant_count,
            "sample_covenants": self.sample_covenants,
        }


@dataclass
class LawConflictSummary:
    obligation: str
    permission: str
    analysis: Dict[str, Any]
    numeric_signals: List[Dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation": self.obligation,
            "permission": self.permission,
            "analysis": self.analysis,
            "numeric_signals": self.numeric_signals,
        }


class CALEValidator:
    """End-to-end helper that wires together SEC fetch, finance extraction and CALE."""

    def __init__(
        self,
        *,
        tickers: Sequence[str] | None = None,
        event_date: Optional[str] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: CacheManager | None = None,
        read_cache: bool = True,
        services: CALEServices | None = None,
    ) -> None:
        self.tickers = tuple(t.upper() for t in (tickers or DEFAULT_TICKERS))
        self.event_date = event_date or date.today().isoformat()
        self.user_agent = user_agent
        self.cache = cache or CacheManager()
        base_services = services or build_services()
        caching_embedder = CachingLegalEmbedder(
            base_services.embedder, cache=self.cache, read_cache=read_cache
        )
        base_services.embedder = caching_embedder
        base_services.feature_engine.embedder = caching_embedder
        if hasattr(base_services.suggester, "embedder"):
            base_services.suggester.embedder = caching_embedder
        self.services = base_services
        self.engine = CALEEngine(services=base_services)
        self.read_cache = read_cache

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _manifest_key(self) -> str:
        return self.cache.manifest_key(self.tickers, self.event_date, self.user_agent)

    def _fetch_manifest(self) -> List[FilingManifestEntry]:
        entries: List[FilingManifestEntry] = []
        for ticker in self.tickers:
            cik = TICKER_TO_CIK.get(ticker.upper())
            if not cik:
                LOGGER.warning("Skipping unknown ticker %s", ticker)
                continue
            submissions = load_submissions(cik, ua=self.user_agent)
            rows = pick_last_10k_10q_before(submissions, self._event_date())
            for row in rows:
                path = ensure_filing_html(cik, row["acc"], row["prim"], ua=self.user_agent)
                if not path:
                    LOGGER.debug("No filing downloaded for %s %s", ticker, row)
                    continue
                entries.append(
                    FilingManifestEntry(
                        ticker=ticker,
                        cik=cik,
                        form=row.get("form", ""),
                        filed=row.get("date", self.event_date),
                        accession=row.get("acc", ""),
                        primary_document=row.get("prim", ""),
                        path=str(path),
                    )
                )
        entries.sort(key=lambda item: item.filed, reverse=True)
        return entries

    def _event_date(self) -> date:
        try:
            return date.fromisoformat(self.event_date)
        except ValueError:
            return date.today()

    def load_manifest(self) -> List[FilingManifestEntry]:
        key = self._manifest_key()
        cached = self.cache.load_manifest(key, read_cache=self.read_cache)
        if cached is not None:
            LOGGER.info("Loaded manifest from cache (%d entries)", len(cached))
            return [FilingManifestEntry(**entry) for entry in cached]
        entries = self._fetch_manifest()
        self.cache.save_manifest(key, [entry.to_dict() for entry in entries])
        LOGGER.info("Fetched %d filings from SEC", len(entries))
        return entries

    # ------------------------------------------------------------------
    # Filing analysis
    # ------------------------------------------------------------------
    def _load_document(self, entry: FilingManifestEntry) -> Doc:
        cached = self.cache.load_doc(entry.path, read_cache=self.read_cache)
        if cached is not None:
            return cached
        doc = load_doc(entry.path)
        self.cache.save_doc(entry.path, doc)
        return doc

    def _section_summaries(
        self, doc: Doc, *, limit: int
    ) -> List[FinanceSectionSummary]:
        sections = find_sections(doc)
        results: List[FinanceSectionSummary] = []
        for section in sections[:limit]:
            window = doc.blocks[section.start_block : section.end_block]
            text = " ".join(block.text for block in window if block.text)
            covenants = extract_numeric_covenants(text)
            results.append(
                FinanceSectionSummary(
                    title=section.title,
                    score=round(section.score, 3),
                    anchor=section.anchor,
                    doc_kind=section.doc_kind,
                    covenant_count=len(covenants),
                    sample_covenants=covenants[:3],
                )
            )
        return results

    def _rule_payload(
        self,
        entry: FilingManifestEntry,
        sentence: str,
        role: str,
        index: int,
    ) -> dict[str, Any]:
        filed_year = int(entry.filed.split("-")[0]) if entry.filed else datetime.now(UTC).year
        statute = f"{entry.form or 'SEC Filing'}"
        section = f"{role}-{index:03d}"
        return {
            "id": f"{entry.ticker}-{entry.form}-{role}-{index:03d}",
            "text": sentence,
            "jurisdiction": "US.Federal",
            "statute": statute,
            "section": section,
            "enactment_year": filed_year,
        }

    def _law_conflicts(
        self,
        entry: FilingManifestEntry,
        pairs: Sequence[tuple[str, str]],
        *,
        limit: int,
    ) -> List[LawConflictSummary]:
        rng = random.Random(hash((entry.ticker, entry.accession)) & 0xFFFFFFFF)
        conflicts: List[LawConflictSummary] = []
        analyses: List[tuple[float, LawConflictSummary]] = []
        for idx, (obligation, permission) in enumerate(pairs):
            rule1 = self._rule_payload(entry, obligation, "OBL", idx)
            rule2 = self._rule_payload(entry, permission, "PERM", idx)
            analysis = self.engine.suggest(rule1, rule2)
            numeric = extract_numeric_covenants(obligation)
            summary = LawConflictSummary(
                obligation=obligation.strip(),
                permission=permission.strip(),
                analysis=analysis,
                numeric_signals=numeric,
            )
            # Introduce small deterministic jitter to avoid tie bias.
            jitter = rng.uniform(-1e-3, 1e-3)
            analyses.append((analysis.get("conflict_intensity", 0.0) + jitter, summary))
        analyses.sort(key=lambda item: item[0], reverse=True)
        for _, summary in analyses[:limit]:
            conflicts.append(summary)
        return conflicts

    def analyse_filing(
        self,
        entry: FilingManifestEntry,
        *,
        section_limit: int,
        pair_limit: int,
    ) -> dict[str, Any]:
        doc = self._load_document(entry)
        sections = self._section_summaries(doc, limit=section_limit)
        try:
            html = Path(entry.path).read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            LOGGER.warning("Failed to read HTML for %s: %s", entry.path, exc)
            pairs: List[tuple[str, str]] = []
        else:
            pairs = extract_pairs_from_html(html)
        conflicts = self._law_conflicts(entry, pairs, limit=pair_limit) if pairs else []
        return {
            "ticker": entry.ticker,
            "form": entry.form,
            "filed": entry.filed,
            "path": entry.path,
            "sections": [section.to_dict() for section in sections],
            "pairs_considered": len(pairs),
            "law_conflicts": [conflict.to_dict() for conflict in conflicts],
        }

    def generate_report(
        self,
        *,
        section_limit: int = 3,
        pair_limit: int = 5,
        manifest_limit: Optional[int] = None,
    ) -> dict[str, Any]:
        manifest = self.load_manifest()
        if manifest_limit is not None:
            manifest = manifest[: int(manifest_limit)]
        if not manifest:
            raise click.ClickException("No filings available for validation.")
        filings: List[dict[str, Any]] = []
        conflict_scores: List[float] = []
        max_conflict: tuple[float, dict[str, Any]] | None = None
        for entry in manifest:
            filing = self.analyse_filing(
                entry,
                section_limit=section_limit,
                pair_limit=pair_limit,
            )
            filings.append(filing)
            for conflict in filing["law_conflicts"]:
                ci = float(conflict.get("analysis", {}).get("conflict_intensity", 0.0))
                conflict_scores.append(ci)
                if max_conflict is None or ci > max_conflict[0]:
                    max_conflict = (
                        ci,
                        {
                            "ticker": filing["ticker"],
                            "form": filing["form"],
                            "filed": filing["filed"],
                            "obligation": conflict.get("obligation", ""),
                            "permission": conflict.get("permission", ""),
                        },
                    )
        avg_conflict = sum(conflict_scores) / len(conflict_scores) if conflict_scores else 0.0
        summary = {
            "filings_processed": len(filings),
            "pairs_analyzed": int(sum(filing["pairs_considered"] for filing in filings)),
            "avg_conflict_intensity": round(avg_conflict, 4),
        }
        if max_conflict:
            summary["max_conflict"] = {
                "score": round(max_conflict[0], 4),
                **max_conflict[1],
            }
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "parameters": {
                "tickers": list(self.tickers),
                "event_date": self.event_date,
                "user_agent": self.user_agent,
                "section_limit": section_limit,
                "pair_limit": pair_limit,
                "manifest_limit": manifest_limit,
            },
            "summary": summary,
            "manifest": [entry.to_dict() for entry in manifest],
            "filings": filings,
        }

    def write_report(self, report: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        LOGGER.info("Report written to %s", path)
