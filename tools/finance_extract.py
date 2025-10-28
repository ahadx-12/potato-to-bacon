#!/usr/bin/env python3
"""Shared leverage covenant extraction utilities for CALE validation."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests

CFG_PATH = Path("configs/finance.yml")

# ---------------------------------------------------------------------------
# Pattern banks & heuristics
# ---------------------------------------------------------------------------

TARGET_SECTIONS = [
    "LIQUIDITY AND CAPITAL RESOURCES",
    "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "RISK FACTORS",
    "INDEBTEDNESS",
    "CREDIT FACILITIES",
    "GOING CONCERN",
]

OBLIGATION_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(must|shall|required to|agree(s)? to)\b.*\b(maintain|comply|keep|meet|reduce|not exceed)\b", re.I),
    re.compile(r"\b(covenant|indenture)\b.*\b(require(s|d)?|obligate(s|d)?)\b", re.I),
    re.compile(r"\b(is subject to|remains subject to)\b.*\b(leverage|coverage|debt)\b", re.I),
)

PERMISSION_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(may|can|is permitted to|are allowed to)\b.*\b(borrow|incur|issue|draw|access)\b", re.I),
    re.compile(r"\b(avail(able)?|availability)\b.*\b(facility|revolver|credit line)\b", re.I),
    re.compile(r"\b(borrowing capacity|borrowing availability)\b", re.I),
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

EXC_TERMS = ("unless", "except", "subject to", "provided that", "waiver", "amend", "amendment")
HEAD_RE = re.compile(r"^\s*<[^>]+>\s*$")

KW_GOING = re.compile(r"going concern|substantial doubt", re.I)
KW_BREACH = re.compile(r"\b(default|event of default|breach|waiver|amend(ment)?)\b", re.I)
KW_COVENANT = re.compile(r"\b(covenant|leverage ratio|interest coverage|fixed charge coverage|dscr)\b", re.I)
RATIO_PAT = re.compile(r"(\d+(\.\d+)?)\s*(x|times|%)")

SEC_BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives"
LOCAL_SEC_ROOT = Path("data/sec")


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
# Filing + text utilities
# ---------------------------------------------------------------------------


class _LocalResponse:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.status_code = 200

    def json(self) -> Dict[str, object]:
        with self._path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def text(self) -> str:
        return self._path.read_text(encoding="utf-8")


def _local_sec_path(url: str) -> Optional[Path]:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    path = parsed.path
    if path.endswith("company_tickers.json"):
        return LOCAL_SEC_ROOT / "company_tickers.json"
    if "/submissions/CIK" in path:
        name = path.split("/submissions/")[-1]
        return LOCAL_SEC_ROOT / "submissions" / name
    if "/edgar/data/" in path:
        suffix = path.split("/edgar/data/")[-1]
        return LOCAL_SEC_ROOT / "edgar" / "data" / suffix
    return None


def sec_get(url: str, ua: str, params=None, max_retries: int = 3, sleep: float = 0.5):
    local_path = _local_sec_path(url)
    if local_path and local_path.exists():
        return _LocalResponse(local_path)

    last_exc: Optional[Exception] = None
    response: Optional[requests.Response] = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={"User-Agent": ua}, params=params, timeout=30)
        except Exception as exc:  # pragma: no cover - network fallback
            last_exc = exc
            response = None
        else:
            if response.status_code == 200:
                return response
        time.sleep(sleep * (attempt + 1))

    local_path = _local_sec_path(url)
    if local_path and local_path.exists():
        return _LocalResponse(local_path)

    if response is not None:
        response.raise_for_status()
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url}")


_CIK_CACHE: Dict[str, str] = {}


def get_cik_map(ua: str) -> Dict[str, str]:
    if _CIK_CACHE:
        return _CIK_CACHE
    url = "https://www.sec.gov/files/company_tickers.json"
    data = sec_get(url, ua).json()
    mapping: Dict[str, str] = {}
    for row in data.values():
        mapping[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    _CIK_CACHE.update(mapping)
    return mapping


def fetch_submissions(cik: str, ua: str) -> dict:
    return sec_get(f"{SEC_BASE}/submissions/CIK{cik}.json", ua).json()


def filings_before(sub: dict, cutoff: dt.date, forms: Sequence[str] = ("10-K", "10-Q")) -> List[dict]:
    recent = sub.get("filings", {}).get("recent", {})
    out: List[dict] = []
    for form, date_str, accession, primary in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        try:
            date = dt.date.fromisoformat(str(date_str))
        except Exception:
            continue
        if form in forms and date < cutoff:
            out.append({"form": form, "date": date, "accession": accession, "primary": primary})
    out.sort(key=lambda row: row["date"])
    return out


def fetch_primary_doc(cik: str, accession: str, primary: str, ua: str) -> str:
    accession_nodash = accession.replace("-", "")
    url = f"{ARCHIVES}/edgar/data/{int(cik)}/{accession_nodash}/{primary}"
    return sec_get(url, ua).text


def strip_html_to_lines(html: str) -> List[str]:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p>", "\n", html)
    text = re.sub(r"(?is)<.*?>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and not HEAD_RE.match(ln)]


def find_sections(lines: Iterable[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current
        if current and buffer:
            sections.setdefault(current, []).extend(buffer)
        buffer = []

    for line in lines:
        is_heading = (line.isupper() and len(line) < 140) or any(target in line.upper() for target in TARGET_SECTIONS)
        if is_heading:
            upper = line.upper()
            matched = None
            for candidate in TARGET_SECTIONS:
                if candidate in upper:
                    matched = candidate
                    break
            flush()
            current = matched
        else:
            if current:
                buffer.append(line)
    flush()
    return sections


def sentence_split(text: str) -> List[str]:
    out: List[str] = []
    for chunk in re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", text):
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
    return out


@dataclass
class Clause:
    sentence: str
    section: str
    modality: float
    is_obl: bool
    is_perm: bool


def modality_scalar(sentence: str) -> float:
    lower = sentence.lower()
    if "must not" in lower or "shall not" in lower:
        return -1.0
    if "may not" in lower:
        return -0.3
    if "must" in lower or "shall" in lower or "required to" in lower:
        return 1.0
    if "may" in lower or "can" in lower:
        return 0.3
    return 0.0


def extract_clauses(sections: Dict[str, List[str]]) -> List[Clause]:
    clauses: List[Clause] = []
    for sec, lines in sections.items():
        text = " ".join(lines)
        for sent in sentence_split(text):
            is_obl = any(pat.search(sent) for pat in OBLIGATION_PATTERNS)
            is_perm = any(pat.search(sent) for pat in PERMISSION_PATTERNS)
            if not (is_obl or is_perm):
                continue
            clauses.append(Clause(sent, sec, modality_scalar(sent), is_obl, is_perm))
    return clauses


def pairs_from_clauses(clauses: Sequence[Clause], max_pairs: int = 60) -> List[Tuple[Clause, Clause, int]]:
    pairs: List[Tuple[Clause, Clause, int]] = []
    for rule_obl in clauses:
        if not rule_obl.is_obl:
            continue
        for rule_perm in clauses:
            if not rule_perm.is_perm:
                continue
            if rule_obl.section != rule_perm.section:
                continue
            bypass_flag = 1 if any(term in rule_perm.sentence.lower() for term in EXC_TERMS) else 0
            pairs.append((rule_obl, rule_perm, bypass_flag))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def cale_analyze(api_base: str, rule1: dict, rule2: dict) -> dict:
    url = f"{api_base.rstrip('/')}/v1/law/analyze"
    response = requests.post(url, json={"rule1": rule1, "rule2": rule2}, timeout=60)
    response.raise_for_status()
    return response.json()


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
    return min(1.0, value)


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


def _authority_balance(section: str) -> float:
    section_upper = (section or "").upper()
    mapping = [
        ("CREDIT", "CREDIT_AGREEMENT"),
        ("INDEBTEDNESS", "CREDIT_AGREEMENT"),
        ("LIQUIDITY", "LIQUIDITY"),
        ("MANAGEMENT", "MDNA"),
        ("RISK", "RISK_FACTORS"),
        ("NOTE", "NOTES_TO_FS"),
        ("INDENTURE", "INDENTURE_NOTES"),
    ]
    weight = 0.5
    for needle, key in mapping:
        if needle in section_upper:
            weight = AUTHORITY_WEIGHTS.get(key, weight)
            break
    max_w = AUTHORITY_MAX if AUTHORITY_MAX > 0 else 1.0
    return float(np.clip(weight / max_w, 0.1, 1.0))


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


@dataclass
class FilingScore:
    best_row: Optional[dict]
    evidence_rows: List[dict]


def score_filing(
    ticker: str,
    cik: str,
    filing: dict,
    ua: str,
    api_base: str,
    as_of: dt.date,
    prev_meta: Optional[dict] = None,
) -> FilingScore:
    html = fetch_primary_doc(cik, filing["accession"], filing["primary"], ua)
    lines = strip_html_to_lines(html)
    sections = find_sections(lines)
    clauses = extract_clauses(sections)
    pairs = pairs_from_clauses(clauses)
    evidences: List[dict] = []
    best_row: Optional[dict] = None

    delta_days = None
    prev_form = None
    if prev_meta:
        prev_form = prev_meta.get("form")
        prev_date = prev_meta.get("date")
        if isinstance(prev_date, dt.date):
            delta_days = (filing["date"] - prev_date).days

    for clause_obl, clause_perm, bypass in pairs:
        dv = compute_dv(clause_obl.sentence, clause_perm.sentence)
        rule1 = {
            "text": clause_obl.sentence,
            "jurisdiction": "US",
            "statute": "Debt Covenant",
            "section": clause_obl.section,
            "enactment_year": as_of.year,
        }
        rule2 = {
            "text": clause_perm.sentence,
            "jurisdiction": "US",
            "statute": "Management Guidance",
            "section": clause_perm.section,
            "enactment_year": as_of.year,
        }
        try:
            metrics = cale_analyze(api_base, rule1, rule2)
        except Exception:
            continue

        cue_weight, cue_meta = _cue_score(clause_obl.sentence, clause_perm.sentence)
        C = float(metrics.get("conflict_intensity", 0.0))
        Ab = float(metrics.get("authority_balance", 0.0))
        S = float(metrics.get("semantic_overlap", 0.0))
        Dt = float(metrics.get("temporal_drift", 0.0))
        if C <= 0.0:
            C = _fallback_conflict(dv, cue_meta, clause_obl.sentence, clause_perm.sentence)
        if Ab <= 0.0:
            Ab = _authority_balance(clause_obl.section)
        cce_raw = float(np.clip(C * Ab * dv, 0.0, 1.0))

        temporal_weight = _temporal_weight(filing["form"], prev_form, delta_days)
        bypass_weight = 0.3 if bypass else 1.0
        combo_weight = float(np.clip(cue_weight * temporal_weight * bypass_weight, 0.1, 3.0))
        jitter_seed = (clause_obl.sentence + "||" + clause_perm.sentence).encode("utf-8")
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
            "o_sentence": clause_obl.sentence,
            "p_sentence": clause_perm.sentence,
            "section": clause_obl.section,
        }
        evidences.append(row)
        if best_row is None or row["CCE"] > best_row["CCE"]:
            best_row = row

    return FilingScore(best_row=best_row, evidence_rows=evidences)


def extract_filing_features(
    ticker: str,
    as_of: dt.date,
    ua: str,
    api_base: str,
) -> Tuple[Optional[dict], List[dict], Optional[dict]]:
    """Return (best_row, evidence_rows, previous_best_row)."""

    cik_map = get_cik_map(ua)
    cik = cik_map.get(ticker.upper())
    if not cik:
        return None, [], None

    submissions = fetch_submissions(cik, ua)
    filings = filings_before(submissions, as_of)
    if not filings:
        return None, [], None

    latest = filings[-1]
    prev_meta = filings[-2] if len(filings) >= 2 else None
    prev_row: Optional[dict] = None

    if prev_meta is not None:
        prev_score = score_filing(ticker, cik, prev_meta, ua, api_base, as_of, None)
        prev_row = prev_score.best_row

    score = score_filing(ticker, cik, latest, ua, api_base, as_of, prev_meta)
    best_row = score.best_row

    if best_row is None:
        return None, score.evidence_rows, prev_row

    prev_cce = float(prev_row["CCE"]) if (prev_row is not None and "CCE" in prev_row) else None
    if prev_cce is not None:
        best_row["cce_delta"] = float(best_row["CCE"] - prev_cce)
        best_row["prev_cce"] = prev_cce
    else:
        best_row["cce_delta"] = 0.0
        best_row["prev_cce"] = None

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

