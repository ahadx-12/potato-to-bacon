#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rapid CALE event-study validator for leverage/debt conflicts."""

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from urllib.parse import urlsplit

try:  # optional dependency
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - fallback if sklearn missing
    roc_auc_score = None  # type: ignore[assignment]

try:  # optional dependency
    from scipy import stats
except Exception:  # pragma: no cover - fallback if scipy missing
    stats = None  # type: ignore[assignment]

import importlib
import importlib.util

CFG_PATH = Path("configs/finance.yml")

TARGET_SECTIONS = [
    "LIQUIDITY AND CAPITAL RESOURCES",
    "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "RISK FACTORS",
    "INDEBTEDNESS",
    "CREDIT FACILITIES",
    "GOING CONCERN",
]

OBLIGATION_PAT = re.compile(r"\b(must|shall)\b.*\b(maintain|comply|keep|meet)\b", re.I)
PERMISSION_PAT = re.compile(r"\b(may|permit|allowed to)\b.*\b(borrow|incur|issue|leverage|indebtedness|notes?)\b", re.I)
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
def _load_config() -> Dict[str, any]:
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


# ---------------------------------------------------------------------------
# Filing + text utilities
# ---------------------------------------------------------------------------
class _LocalResponse:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.status_code = 200

    def json(self) -> Dict[str, any]:
        with self._path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def text(self) -> str:
        return self._path.read_text(encoding="utf-8")


def _local_sec_path(url: str) -> Optional[Path]:
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
        except Exception as exc:
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


def get_cik_map(ua: str) -> Dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = sec_get(url, ua).json()
    mapping: Dict[str, str] = {}
    for row in data.values():
        mapping[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return mapping


def fetch_submissions(cik: str, ua: str) -> dict:
    return sec_get(f"{SEC_BASE}/submissions/CIK{cik}.json", ua).json()


def filings_before(sub: dict, cutoff: dt.date, forms=("10-K", "10-Q")) -> List[dict]:
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


def find_sections(lines: List[str]) -> Dict[str, List[str]]:
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
    if "must" in lower or "shall" in lower:
        return 1.0
    if "may" in lower:
        return 0.3
    return 0.0


def extract_clauses(sections: Dict[str, List[str]]) -> List[Clause]:
    clauses: List[Clause] = []
    for sec, lines in sections.items():
        text = " ".join(lines)
        for sent in sentence_split(text):
            is_obl = bool(OBLIGATION_PAT.search(sent))
            is_perm = bool(PERMISSION_PAT.search(sent))
            if not (is_obl or is_perm):
                continue
            clauses.append(Clause(sent, sec, modality_scalar(sent), is_obl, is_perm))
    return clauses


def pairs_from_clauses(clauses: List[Clause], max_pairs: int = 60) -> List[Tuple[Clause, Clause, int]]:
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


# ---------------------------------------------------------------------------
# CALE API bridge
# ---------------------------------------------------------------------------
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
    return min(1.0, value)


# ---------------------------------------------------------------------------
# Feature extraction per filing
# ---------------------------------------------------------------------------
def filing_score(ticker: str, as_of: dt.date, ua: str, api_base: str) -> Tuple[Optional[float], Optional[dict], List[dict]]:
    try:
        cik_map = get_cik_map(ua)
    except Exception:
        return None, None, []
    cik = cik_map.get(ticker.upper())
    if not cik:
        return None, None, []

    submissions = fetch_submissions(cik, ua)
    filings = filings_before(submissions, as_of)
    if not filings:
        return None, None, []

    latest = filings[-1]
    html = fetch_primary_doc(cik, latest["accession"], latest["primary"], ua)
    lines = strip_html_to_lines(html)
    sections = find_sections(lines)
    clauses = extract_clauses(sections)
    pairs = pairs_from_clauses(clauses)
    if not pairs:
        return None, None, []

    evidences: List[dict] = []
    best_row: Optional[dict] = None

    for clause_obl, clause_perm, bypass in pairs:
        dv = compute_dv(clause_obl.sentence, clause_perm.sentence)
        r1 = {
            "text": clause_obl.sentence,
            "jurisdiction": "US",
            "statute": "Debt Covenant",
            "section": clause_obl.section,
            "enactment_year": as_of.year,
        }
        r2 = {
            "text": clause_perm.sentence,
            "jurisdiction": "US",
            "statute": "Management Guidance",
            "section": clause_perm.section,
            "enactment_year": as_of.year,
        }
        try:
            metrics = cale_analyze(api_base, r1, r2)
        except Exception:
            continue
        C = float(metrics.get("conflict_intensity", 0.0))
        Ab = float(metrics.get("authority_balance", 0.0))
        S = float(metrics.get("semantic_overlap", 0.0))
        Dt = float(metrics.get("temporal_drift", 0.0))
        cce = max(0.0, min(1.0, C * Ab * dv))
        row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(latest["date"]),
            "form": latest["form"],
            "C": C,
            "Ab": Ab,
            "Dv": dv,
            "B": float(bypass),
            "S": S,
            "Dt": Dt,
            "CCE": cce,
            "o_sentence": clause_obl.sentence,
            "p_sentence": clause_perm.sentence,
            "section": clause_obl.section,
        }
        evidences.append(row)
        if best_row is None or cce > best_row["CCE"]:
            best_row = row

    if best_row is None:
        return None, None, evidences

    return best_row["CCE"], best_row, evidences


# ---------------------------------------------------------------------------
# Metrics + verdict helpers
# ---------------------------------------------------------------------------
def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if roc_auc_score is not None:
        return float(roc_auc_score(y_true, scores))
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[y_true == 1]
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.mean() - (n_pos - 1) / 2.0) / n_neg)


def welch_pvalue(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan")
    if stats is not None:
        return float(stats.ttest_ind(sample_a, sample_b, equal_var=False).pvalue)
    mean_a, mean_b = float(sample_a.mean()), float(sample_b.mean())
    var_a, var_b = float(sample_a.var(ddof=1)), float(sample_b.var(ddof=1))
    n_a, n_b = len(sample_a), len(sample_b)
    numerator = mean_a - mean_b
    denominator = math.sqrt(var_a / n_a + var_b / n_b)
    if denominator == 0.0:
        return float("nan")
    t_stat = numerator / denominator
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a**2) / ((n_a**2) * (n_a - 1)) + (var_b**2) / ((n_b**2) * (n_b - 1))
    dof = df_num / df_den if df_den != 0 else min(n_a, n_b) - 1
    # two-tailed p-value using survival function approximation
    # via complementary error function for normal approximation
    return float(2.0 * 0.5 * math.erfc(abs(t_stat) / math.sqrt(2)))


def verdict_for_auc(auc: float) -> str:
    strong = float(CFG.get("validation", {}).get("auc_strong", 0.75))
    real = float(CFG.get("validation", {}).get("auc_real", 0.70))
    weak = float(CFG.get("validation", {}).get("auc_weak", 0.65))
    if auc >= strong:
        return "🟢 STRONG — proceed to backtest immediately"
    if auc >= real:
        return "🟡 REAL — expand sample & tune"
    if auc >= weak:
        return "🟠 WEAK — iterate parsing/authority/ΔCCE"
    return "🔴 NO SIGNAL — fix features and re-test"


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--controls-csv", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--user-agent", required=True)
    parser.add_argument("--out-dir", default="reports/leverage_alpha")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_csv)
    controls = pd.read_csv(args.controls_csv)

    rows: List[dict] = []
    evidence: List[dict] = []

    for _, row in events.iterrows():
        ticker = str(row["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(row["event_date"]))
        cce, feature_row, ev_rows = filing_score(ticker, as_of, args.user_agent, args.api_base)
        evidence.extend(ev_rows)
        if feature_row is None:
            continue
        rows.append({**feature_row, "label": 1})

    for _, row in controls.iterrows():
        ticker = str(row["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(row["as_of_date"]))
        cce, feature_row, ev_rows = filing_score(ticker, as_of, args.user_agent, args.api_base)
        evidence.extend(ev_rows)
        if feature_row is None:
            continue
        rows.append({**feature_row, "label": 0})

    if not rows:
        print("No rows extracted. Check SEC rate limits, API availability, and CSV inputs.")
        return

    df = pd.DataFrame(rows)
    feature_path = out_dir / "event_scores.csv"
    df.to_csv(feature_path, index=False)

    evidence_path = out_dir / "top_pairs.json"
    with evidence_path.open("w", encoding="utf-8") as handle:
        json.dump(evidence[:50], handle, indent=2)

    y = df["label"].to_numpy(dtype=float)
    scores = df["CCE"].to_numpy(dtype=float)
    auc = compute_auc(y, scores)

    distressed = df[df.label == 1]["CCE"].to_numpy(dtype=float)
    control = df[df.label == 0]["CCE"].to_numpy(dtype=float)
    p_value = welch_pvalue(distressed, control)

    print("\n=== CALE Event Study (CCE Baseline) ===")
    print(f"N(distressed)={len(distressed)}, mean CCE={float(distressed.mean()) if len(distressed) else float('nan'):.3f}")
    print(f"N(control)   ={len(control)}, mean CCE={float(control.mean()) if len(control) else float('nan'):.3f}")
    print(f"AUC={auc:.3f}   Verdict: {verdict_for_auc(auc)}")
    print(f"Welch t-test p-value={p_value:.4f}")
    print(f"Saved features → {feature_path}")
    print(f"Saved evidence → {evidence_path}")
    print("Note: Research tool only — not investment advice.")


if __name__ == "__main__":  # pragma: no cover
    random.seed(42)
    np.random.seed(42)
    main()
