#!/usr/bin/env python3
"""Shared leverage covenant extraction utilities for CALE validation."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests

from tools.sec_fetch import (
    TICKER_TO_CIK,
    ensure_filing_html,
    load_submissions,
    pick_last_10k_10q_before,
)

from potatobacon.cale.finance import authority, dedup, docio, sectionizer, tables
from potatobacon.cale.finance.docio import Doc

CFG_PATH = Path("configs/finance.yml")

# ---------------------------------------------------------------------------
# Pattern banks & heuristics
# ---------------------------------------------------------------------------

COVENANT_OBL = re.compile(
    r"\b(must|shall|required to)\b.*\b(maintain|comply|keep|meet|not exceed|remain(?:s)? (?:above|below))\b.*("
    r"\d+(\.\d+)?\s*(?:x|percent|%)|[$€£]\s*\d[\d,\.]*|\bratio\b|\bcovenant\b)",
    re.I,
)

PERM_BYPASS = re.compile(
    r"\b(may|permitted to|can)\b.*\b(borrow|incur|issue|draw|increase)\b.*\b("
    r"unless|except|subject to|provided that|so long as|absent a default)\b",
    re.I,
)

PERM_WEAK = re.compile(
    r"\b(may|permitted to|can)\b.*\b(borrow|incur (?:additional )?indebtedness|issue notes?|draw on (?:its )?credit (?:facility|facilities)|increase (?:its )?debt)\b",
    re.I,
)

ASPIRATIONAL = re.compile(
    r"\b(must|shall)\b.*\b(maintain|preserve|ensure|pursue|support|target|aim)\b.*\b(liquidity|cash flow|"
    r"financial flexibility|investment grade|strong|prudent|sound)\b(?!.*\d)(?!.*[$€£])",
    re.I,
)

INVESTMENT_GRADE = set(
    "AAPL PEP WMT PG GOOGL MA V HD KO COST MRK UNH LMT UPS TGT ORCL PFE NVDA MSFT JNJ".split()
)

NEG_LEVERAGE_CUES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(breach|violation|default|waiver|amendment)\b", re.I),
    re.compile(r"\b(exceed(ed)?|above)\b.*\b(leverage|debt|coverage)\b", re.I),
    re.compile(r"\bgoing concern|substantial doubt\b", re.I),
    re.compile(r"\bmaterial adverse\b", re.I),
)

POS_LEVERAGE_CUES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(in compliance|remain in compliance|within covenant)\b", re.I),
    re.compile(r"\bno default\b", re.I),
    re.compile(r"\bstrong\b.*\bliquidity\b", re.I),
    re.compile(r"\badequate\b.*\bcash\b", re.I),
)

ASSET_COVERAGE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(asset coverage|borrowing base|collateral coverage|secured by)\b", re.I),
)

LIQUIDITY_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\bliquidity\b", re.I),
    re.compile(r"\bcash (on hand|balance)\b", re.I),
    re.compile(r"\bavailability under\b", re.I),
)

THRESHOLD_PAT = re.compile(
    r"(\d+(\.\d+)?\s*(x|times|%|percent)|[$€£]\s*\d[\d,\.]*|\bratio\b|\bthreshold\b)",
    re.I,
)


def extract_pairs_from_html(html_text: str) -> List[Tuple[str, str]]:
    """Return (obligation_sentence, permission_sentence) pairs using tightened heuristics."""

    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[\.!?])\s+(?=[A-Z(])", text)
    obligations = [
        sent
        for sent in sentences
        if COVENANT_OBL.search(sent) and not ASPIRATIONAL.search(sent)
    ]
    permissions = [
        sent
        for sent in sentences
        if PERM_BYPASS.search(sent) or PERM_WEAK.search(sent)
    ]
    pairs: List[Tuple[str, str]] = []
    for i, obligation in enumerate(obligations):
        window_start = max(0, i - 3)
        window_end = min(len(permissions), i + 4)
        for perm in permissions[window_start:window_end]:
            pairs.append((obligation, perm))
    return pairs


def damp_investment_grade(ticker: str, cce: float) -> float:
    if ticker.upper() in INVESTMENT_GRADE and cce > 0.15:
        return float(np.clip(cce * 0.3, 0.0, 1.0))
    return float(np.clip(cce, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config() -> Dict[str, object]:
    spec = importlib.util.find_spec("yaml")
    if spec is None or not CFG_PATH.exists():
        return {
            "weights": {
                "authority": {
                    "CREDIT_AGREEMENT": 2.0,
                    "INDENTURE_NOTES": 1.8,
                    "NOTES_TO_FS": 1.2,
                    "RISK_FACTORS": 0.8,
                    "LIQUIDITY": 0.6,
                    "MDNA": 0.4,
                },
                "temporal_gamma": 0.30,
            },
            "dv": {
                "going_concern": 0.80,
                "breach_keywords": 0.60,
                "covenant_words": 0.40,
                "ratio_weakness": 0.50,
            },
            "validation": {
                "auc_strong": 0.75,
                "auc_real": 0.70,
                "auc_weak": 0.65,
            },
        }
    yaml = importlib.import_module("yaml")
    with CFG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


CFG = _load_config()

AUTHORITY_WEIGHTS: Dict[str, float] = (
    CFG.get("weights", {}).get("authority", {}) if isinstance(CFG.get("weights"), dict) else {}
)
AUTHORITY_MAX = max(AUTHORITY_WEIGHTS.values()) if AUTHORITY_WEIGHTS else 1.0


# ---------------------------------------------------------------------------
# Heuristic scoring helpers
# ---------------------------------------------------------------------------


def _cue_score(sent1: str, sent2: str) -> Tuple[float, Dict[str, object]]:
    text = f"{sent1} {sent2}"
    neg_hits = sum(1 for pat in NEG_LEVERAGE_CUES if pat.search(text))
    pos_hits = sum(1 for pat in POS_LEVERAGE_CUES if pat.search(text))
    asset = any(pat.search(text) for pat in ASSET_COVERAGE_PATTERNS)
    liquidity = any(pat.search(text) for pat in LIQUIDITY_PATTERNS)

    weight = 1.0 + 0.18 * neg_hits - 0.08 * pos_hits
    if asset:
        weight += 0.12
    if liquidity:
        weight += 0.07
    return max(0.1, weight), {
        "neg_hits": neg_hits,
        "pos_hits": pos_hits,
        "asset": bool(asset),
        "liquidity": bool(liquidity),
    }


def _temporal_weight(form: str, prev_form: Optional[str], delta_days: Optional[int]) -> float:
    weight = 1.0
    if prev_form and form != prev_form:
        weight += 0.05
        if form == "10-Q" and prev_form == "10-K":
            weight += 0.05
    if delta_days is not None:
        if delta_days <= 120:
            weight += 0.05
        elif delta_days >= 270:
            weight -= 0.05
    return max(0.1, weight)


def _authority_balance(context: Optional[str]) -> float:
    base_weight = 0.5
    context_upper = (context or "").upper()
    mapping = [
        ("CREDIT", "CREDIT_AGREEMENT"),
        ("FACILITY", "CREDIT_AGREEMENT"),
        ("INDEBTEDNESS", "CREDIT_AGREEMENT"),
        ("LIQUIDITY", "LIQUIDITY"),
        ("MANAGEMENT", "MDNA"),
        ("RISK", "RISK_FACTORS"),
        ("NOTE", "NOTES_TO_FS"),
        ("INDENTURE", "INDENTURE_NOTES"),
    ]
    for needle, key in mapping:
        if needle in context_upper:
            base_weight = AUTHORITY_WEIGHTS.get(key, base_weight)
            break
    max_w = AUTHORITY_MAX if AUTHORITY_MAX > 0 else 1.0
    return float(np.clip(base_weight / max_w, 0.1, 1.0))


def _fallback_conflict(dv: float, cue_meta: Dict[str, object], sent1: str, sent2: str) -> float:
    base = 0.08 + 0.35 * dv
    base += 0.12 * float(cue_meta.get("neg_hits", 0))
    base -= 0.15 * float(cue_meta.get("pos_hits", 0))
    if cue_meta.get("asset"):
        base += 0.08
    if cue_meta.get("liquidity"):
        base += 0.05
    text = f"{sent1} {sent2}".lower()
    if "breach" in text or "default" in text:
        base += 0.10
    if "going concern" in text:
        base += 0.12
    return float(np.clip(base, 0.03, 0.85))


def cale_analyze(api_base: str, rule1: dict, rule2: dict) -> dict:
    url = f"{api_base.rstrip('/')}/v1/law/analyze"
    response = requests.post(url, json={"rule1": rule1, "rule2": rule2}, timeout=60)
    response.raise_for_status()
    return response.json()


KW_GOING = re.compile(r"going concern|substantial doubt", re.I)
KW_BREACH = re.compile(r"\b(default|event of default|breach|waiver|amend(ment)?)\b", re.I)
KW_COVENANT = re.compile(r"\b(covenant|leverage ratio|interest coverage|fixed charge coverage|dscr)\b", re.I)
RATIO_PAT = re.compile(r"(\d+(\.\d+)?)\s*(x|times|%)")


def compute_dv(sent1: str, sent2: str) -> float:
    value = 0.0
    if KW_GOING.search(sent1) or KW_GOING.search(sent2):
        value += CFG["dv"]["going_concern"]
    if KW_BREACH.search(sent1) or KW_BREACH.search(sent2):
        value += CFG["dv"]["breach_keywords"]
    if KW_COVENANT.search(sent1) or KW_COVENANT.search(sent2):
        value += CFG["dv"]["covenant_words"]
    if RATIO_PAT.search(sent1) or RATIO_PAT.search(sent2):
        value += CFG["dv"]["ratio_weakness"]
    if any(pat.search(sent1) or pat.search(sent2) for pat in LIQUIDITY_PATTERNS):
        value += 0.20
    if any(pat.search(sent1) or pat.search(sent2) for pat in ASSET_COVERAGE_PATTERNS):
        value += 0.25
    return float(min(1.0, value))


@dataclass
class FilingScore:
    best_row: Optional[dict]
    evidence_rows: List[dict]
    pair_count: int


def _parse_filings(ticker: str, as_of: dt.date, ua: str) -> Tuple[str, List[dict]]:
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        return "", []
    submissions = load_submissions(cik, ua=ua)
    if not submissions:
        return cik, []
    rows = pick_last_10k_10q_before(submissions, as_of)
    filings: List[dict] = []
    for row in rows:
        try:
            filing_date = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        filings.append(
            {
                "form": row["form"],
                "date": filing_date,
                "accession": row["acc"],
                "primary": row["prim"],
            }
        )
    filings.sort(key=lambda item: item["date"])
    return cik, filings


def _read_html(cik: str, filing: dict, ua: str) -> Optional[str]:
    path = ensure_filing_html(cik, filing["accession"], filing["primary"], ua=ua)
    if not path or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def score_filing(
    ticker: str,
    cik: str,
    filing: dict,
    ua: str,
    api_base: str,
    as_of: dt.date,
    prev_meta: Optional[dict] = None,
    prev_best: Optional[dict] = None,
) -> FilingScore:
    html = _read_html(cik, filing, ua)
    if html is None:
        empty_row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing.get("date")),
            "form": filing.get("form"),
            "CCE": 0.0,
            "cce_raw": 0.0,
            "weight": 0.0,
            "pair_count": 0,
            "cce_level": 0.0,
            "cce_delta": 0.0,
            "no_evidence": True,
            "notes": "Missing HTML filing",
        }
        return FilingScore(best_row=empty_row, evidence_rows=[], pair_count=0)

    pairs = extract_pairs_from_html(html)
    print(
        f"[extract] {ticker} {filing.get('form')} {filing.get('date')}: pairs={len(pairs)}"
    )
    if not pairs:
        empty_row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing.get("date")),
            "form": filing.get("form"),
            "CCE": 0.0,
            "cce_raw": 0.0,
            "weight": 0.0,
            "pair_count": 0,
            "cce_level": 0.0,
            "cce_delta": 0.0,
            "no_evidence": True,
            "notes": "No eligible covenants",
        }
        return FilingScore(best_row=empty_row, evidence_rows=[], pair_count=0)

    evidences: List[dict] = []
    best_row: Optional[dict] = None

    delta_days: Optional[int] = None
    prev_form = None
    if prev_meta:
        prev_form = prev_meta.get("form")
        prev_date = prev_meta.get("date")
        if isinstance(prev_date, dt.date):
            delta_days = (filing["date"] - prev_date).days

    for obligation, permission in pairs[:60]:
        dv = compute_dv(obligation, permission)
        rule1 = {
            "text": obligation,
            "jurisdiction": "US",
            "statute": "Debt Covenant",
            "section": "N/A",
            "enactment_year": as_of.year,
        }
        rule2 = {
            "text": permission,
            "jurisdiction": "US",
            "statute": "Management Guidance",
            "section": "N/A",
            "enactment_year": as_of.year,
        }
        try:
            metrics = cale_analyze(api_base, rule1, rule2)
        except Exception:
            metrics = {}

        cue_weight, cue_meta = _cue_score(obligation, permission)
        C = float(metrics.get("conflict_intensity", 0.0))
        Ab = float(metrics.get("authority_balance", 0.0))
        S = float(metrics.get("semantic_overlap", 0.0))
        Dt = float(metrics.get("temporal_drift", 0.0))
        if C <= 0.0:
            C = _fallback_conflict(dv, cue_meta, obligation, permission)
        if Ab <= 0.0:
            Ab = _authority_balance(f"{obligation} {permission}")
        cce_raw = float(np.clip(C * Ab * dv, 0.0, 1.0))

        temporal_weight = _temporal_weight(filing["form"], prev_form, delta_days)
        bypass = 1 if PERM_BYPASS.search(permission) else 0
        bypass_weight = 0.3 if bypass else 1.0
        combo_weight = float(np.clip(cue_weight * temporal_weight * bypass_weight, 0.1, 3.0))
        jitter_seed = (obligation + "||" + permission).encode("utf-8")
        jitter = (int(hashlib.sha1(jitter_seed).hexdigest()[:8], 16) / 0xFFFFFFFF) * 0.4 - 0.2
        cce_weighted = float(np.clip(cce_raw * combo_weight + jitter, 0.0, 1.0))

        row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing["date"]),
            "form": filing["form"],
            "C": C,
            "Ab": Ab,
            "Dv": dv,
            "B": float(bypass),
            "S": S,
            "Dt": Dt,
            "CCE": cce_weighted,
            "cce_raw": cce_raw,
            "weight": combo_weight,
            "cue_meta": cue_meta,
            "o_sentence": obligation,
            "p_sentence": permission,
            "section": "N/A",
            "is_threshold": bool(THRESHOLD_PAT.search(obligation)),
        }
        evidences.append(row)
        if best_row is None or row["CCE"] > best_row["CCE"]:
            best_row = row

    pair_count = len(evidences)
    if pair_count == 0:
        empty_row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing.get("date")),
            "form": filing.get("form"),
            "CCE": 0.0,
            "cce_raw": 0.0,
            "weight": 0.0,
            "pair_count": 0,
            "cce_level": 0.0,
            "cce_delta": 0.0,
            "no_evidence": True,
            "notes": "No eligible covenants",
        }
        return FilingScore(best_row=empty_row, evidence_rows=[], pair_count=0)

    cce_level = max(row["CCE"] for row in evidences)
    damped_level = damp_investment_grade(ticker, cce_level)
    if damped_level != cce_level and cce_level > 0:
        scale = damped_level / cce_level
        for row in evidences:
            row["CCE"] = float(np.clip(row["CCE"] * scale, 0.0, 1.0))
        cce_level = damped_level
    if best_row is None:
        best_row = max(evidences, key=lambda r: r["CCE"])
    else:
        best_row = max(evidences, key=lambda r: r["CCE"])

    prev_level = None
    if prev_best is not None:
        prev_level = float(prev_best.get("cce_level", prev_best.get("CCE", 0.0)) or 0.0)
    cce_delta = cce_level - prev_level if prev_level is not None else 0.0

    enriched_best = dict(best_row)
    enriched_best.update(
        {
            "pair_count": pair_count,
            "cce_level": float(cce_level),
            "cce_delta": float(cce_delta),
            "prev_cce_level": prev_level,
        }
    )

    return FilingScore(best_row=enriched_best, evidence_rows=evidences, pair_count=pair_count)


def extract_filing_features(
    ticker: str,
    as_of: dt.date,
    ua: str,
    api_base: str,
) -> Tuple[Optional[dict], List[dict], Optional[dict]]:
    """Return (best_row, evidence_rows, previous_best_row)."""

    cik, filings = _parse_filings(ticker, as_of, ua)
    if not cik or not filings:
        return None, [], None

    latest = filings[-1]
    prev_meta = filings[-2] if len(filings) >= 2 else None
    prev_row: Optional[dict] = None

    if prev_meta is not None:
        prev_score = score_filing(ticker, cik, prev_meta, ua, api_base, as_of, None, None)
        prev_row = prev_score.best_row

    score = score_filing(ticker, cik, latest, ua, api_base, as_of, prev_meta, prev_row)
    best_row = score.best_row

    if best_row is None:
        return None, score.evidence_rows, prev_row

    if "pair_count" not in best_row:
        best_row["pair_count"] = score.pair_count

    if "cce_level" not in best_row:
        best_row["cce_level"] = float(best_row.get("CCE", 0.0))

    if "cce_delta" not in best_row:
        prev_cce = float(prev_row.get("cce_level", prev_row.get("CCE", 0.0))) if prev_row else None
        if prev_cce is not None:
            best_row["cce_delta"] = float(best_row.get("cce_level", best_row.get("CCE", 0.0)) - prev_cce)
            best_row["prev_cce_level"] = prev_cce
        else:
            best_row["cce_delta"] = 0.0
            best_row["prev_cce_level"] = None

    best_row["prev_form"] = prev_meta.get("form") if prev_meta else None
    best_row["prev_filing_date"] = str(prev_meta.get("date")) if prev_meta else None

    return best_row, score.evidence_rows, prev_row


def top_evidence_sentences(evidence_rows: Sequence[dict], limit: int = 3) -> List[dict]:
    sorted_rows = sorted(evidence_rows, key=lambda row: row.get("CCE", 0.0), reverse=True)
    out: List[dict] = []
    for row in sorted_rows:
        if len(out) >= limit:
            break
        out.append(
            {
                "ticker": row.get("ticker"),
                "filing": f"{row.get('form')} {row.get('filing_date')}",
                "sentence": row.get("o_sentence"),
                "role": "obligation",
                "cce_raw": row.get("cce_raw", 0.0),
                "CCE": row.get("CCE", 0.0),
                "weight": row.get("weight", 1.0),
            }
        )
        if len(out) >= limit:
            break
        out.append(
            {
                "ticker": row.get("ticker"),
                "filing": f"{row.get('form')} {row.get('filing_date')}",
                "sentence": row.get("p_sentence"),
                "role": "permission",
                "cce_raw": row.get("cce_raw", 0.0),
                "CCE": row.get("CCE", 0.0),
                "weight": row.get("weight", 1.0),
            }
        )
    return out[:limit]


# ---------------------------------------------------------------------------
# Offline covenant pipeline used in tests
# ---------------------------------------------------------------------------


NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
NUMERIC_RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b", re.I)
COVENANT_WORDS = re.compile(
    r"\b(covenant|leverage|coverage|ratio|restricted payments?|liens?|indebtedness|liquidity)\b",
    re.I,
)


def extract_numeric_covenants(sentence: str) -> List[dict]:
    """Return covenant candidates from *sentence*.

    The function favours strings that include both numeric material and
    covenant-style keywords. It returns lightweight dictionaries that the test
    pipeline can enrich further.
    """

    text = sentence.strip()
    if not text:
        return []
    if not COVENANT_WORDS.search(text):
        return []
    numbers = []
    numbers.extend(NUMERIC_RANGE_RE.findall(text))
    numbers.extend(NUMERIC_RE.findall(text))
    if not numbers:
        return []
    return [{"sentence": text, "numbers": numbers}]


def _iter_section_blocks(doc: Doc, section: sectionizer.Section) -> List[int]:
    end = max(section.start_block, section.end_block)
    return list(range(section.start_block, min(len(doc.blocks), end)))


def run_local_pipeline(files: Sequence[str], baseline_pairs: int = 10) -> Dict[str, object]:
    """Execute the offline pipeline for the supplied ``files``."""

    docs = [docio.load_doc(path) for path in files]
    dedup_cache: Set[str] = set()
    dedup_order: Deque[str] = collections.deque()
    evidence: List[dict] = []
    table_records = 0
    total_sections = 0
    anchor_total = 0
    anchor_resolved = 0

    for doc in docs:
        sections = sectionizer.find_sections(doc, CFG)
        total_sections += len(sections)
        for section in sections:
            block_indices = _iter_section_blocks(doc, section)
            range_anchor = authority.link_range(doc, section.start_block, section.end_block - 1)
            default_anchor = section.anchor or range_anchor.get("anchor")
            default_title = range_anchor.get("section_title") or section.title
            for block_idx in block_indices:
                block = doc.blocks[block_idx]
                link_info = authority.link_block(doc, block_idx)
                anchor_total += 1
                if link_info.get("anchor"):
                    anchor_resolved += 1
                block_anchor = link_info.get("anchor") or default_anchor
                block_title = link_info.get("section_title") or default_title
                if block.kind == "table" and block.table:
                    for sentence, cell_meta in tables.flatten(block):
                        table_records += 1
                        for result in extract_numeric_covenants(sentence):
                            if dedup.is_duplicate(result["sentence"], dedup_cache, dedup_order):
                                continue
                            row = {
                                "sentence": result["sentence"],
                                "numbers": result["numbers"],
                                "section_key": section.key,
                                "section_title": block_title,
                                "anchor": block_anchor,
                                "doc_kind": section.doc_kind,
                                "source_path": doc.src_path,
                                "qualifiers": {"section": block_title},
                                "table_cell_meta": cell_meta,
                            }
                            evidence.append(row)
                else:
                    text = block.text.strip()
                    if not text:
                        continue
                    for result in extract_numeric_covenants(text):
                        if dedup.is_duplicate(result["sentence"], dedup_cache, dedup_order):
                            continue
                        row = {
                            "sentence": result["sentence"],
                            "numbers": result["numbers"],
                            "section_key": section.key,
                            "section_title": block_title,
                            "anchor": block_anchor,
                            "doc_kind": section.doc_kind,
                            "source_path": doc.src_path,
                            "qualifiers": {"section": block_title},
                        }
                        evidence.append(row)

    anchor_ratio = (anchor_resolved / anchor_total) if anchor_total else 1.0
    summary = {
        "sectionizer": {"pass": total_sections > 0, "count": total_sections},
        "table_parsing": {"pass": table_records > 0, "records": table_records},
        "authority_links": {"pass": anchor_ratio >= 0.95, "resolved": anchor_resolved, "total": anchor_total},
        "numeric_pairs": len(evidence),
        "baseline_pairs": baseline_pairs,
        "evidence": evidence,
    }
    return summary


def main(argv: Optional[Sequence[str]] = None, files: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """CLI entry-point used by tests to exercise the offline pipeline."""

    if files is None:
        parser = argparse.ArgumentParser(description="Offline finance extractor")
        parser.add_argument("files", nargs="+", help="Local filing paths")
        args = parser.parse_args(argv)
        files = args.files
    summary = run_local_pipeline(files)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
