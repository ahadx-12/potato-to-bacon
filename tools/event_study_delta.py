#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, math, time, argparse, datetime as dt, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score  # fallback: if unavailable, we compute AUC manually
from scipy import stats

# ---------- Config load ----------
def load_yaml_config(path="configs/finance.yml"):
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        # Minimal defaults if yaml missing
        return {
            "weights": {
                "authority": {
                    "CREDIT_AGREEMENT": 2.0,
                    "INDENTURE_NOTES": 1.8,
                    "NOTES_TO_FS": 1.2,
                    "RISK_FACTORS": 0.8,
                    "LIQUIDITY": 0.6,
                    "MDNA": 0.4
                }
            },
            "dv": {
                "going_concern": 0.8,
                "breach_keywords": 0.6,
                "covenant_words": 0.4,
                "ratio_weakness": 0.5
            },
            "validation": {
                "auc_strong": 0.75,
                "auc_real": 0.70,
                "auc_weak": 0.65
            },
            "ablation": {
                "enable_logistic": True,
                "l2_reg": 0.5,
                "max_iter": 2000,
                "lr": 0.05,
                "seed": 42,
                "feature_set": ["C","Ab","Dv","B","dCCE","S","Dt"]
            }
        }

CFG = load_yaml_config()

# ---------- SEC + parsing (copied/adapted from event_study.py) ----------
SEC_BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives"
LOCAL_SEC_ROOT = Path("data/sec")

TARGET_SECTIONS = [
    "LIQUIDITY AND CAPITAL RESOURCES",
    "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "RISK FACTORS",
    "INDEBTEDNESS",
    "CREDIT FACILITIES",
    "GOING CONCERN"
]

OBLIGATION_PAT = re.compile(r"\b(must|shall)\b.*\b(maintain|comply|keep|meet)\b", re.I)
PERMISSION_PAT = re.compile(r"\b(may|permit|allowed to)\b.*\b(borrow|incur|issue|leverage|indebtedness|notes?)\b", re.I)
EXC_TERMS = ("unless","except","subject to","provided that","waiver","amend","amendment")
HEAD_RE = re.compile(r"^\s*<[^>]+>\s*$")
KW_GOING = re.compile(r"going concern|substantial doubt", re.I)
KW_BREACH = re.compile(r"\b(default|event of default|breach|waiver|amend(ment)?)\b", re.I)
KW_COVENANT = re.compile(r"\b(covenant|leverage ratio|interest coverage|fixed charge coverage|dscr)\b", re.I)
RATIO_PAT = re.compile(r"(\d+(\.\d+)?)\s*(x|times|%)")

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


def sec_get(url: str, ua: str, params=None, max_retries=3, sleep=0.5):
    local_path = _local_sec_path(url)
    if local_path and local_path.exists():
        return _LocalResponse(local_path)

    last_exc: Optional[Exception] = None
    response: Optional[requests.Response] = None
    for i in range(max_retries):
        try:
            response = requests.get(url, headers={"User-Agent": ua}, params=params, timeout=30)
        except Exception as exc:
            last_exc = exc
            response = None
        else:
            if response.status_code == 200:
                return response
        time.sleep(sleep * (i+1))

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
    r = sec_get(url, ua)
    data = r.json()
    out = {}
    for _, row in data.items():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out

def fetch_submissions(cik: str, ua: str) -> dict:
    r = sec_get(f"{SEC_BASE}/submissions/CIK{cik}.json", ua)
    return r.json()

def filings_before(sub: dict, cutoff: dt.date, forms=("10-K","10-Q")) -> List[dict]:
    recent = sub.get("filings", {}).get("recent", {})
    dates = recent.get("filingDate", [])
    fm = recent.get("form", [])
    acc = recent.get("accessionNumber", [])
    prim = recent.get("primaryDocument", [])
    rows = []
    for f, d, a, p in zip(fm, dates, acc, prim):
        try:
            ddate = dt.date.fromisoformat(d)
        except:
            continue
        if f in forms and ddate < cutoff:
            rows.append({"form": f, "date": ddate, "accession": a, "primary": p})
    rows.sort(key=lambda x: x["date"])
    return rows

def fetch_primary_doc(cik: str, accession: str, primary: str, ua: str) -> str:
    no_dash = accession.replace("-", "")
    url = f"{ARCHIVES}/edgar/data/{int(cik)}/{no_dash}/{primary}"
    r = sec_get(url, ua)
    return r.text

def strip_html_to_lines(html: str) -> List[str]:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p>", "\n", html)
    text = re.sub(r"(?is)<.*?>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and not HEAD_RE.match(ln)]

def find_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections, cur, buf = {}, None, []
    def flush():
        nonlocal buf, cur
        if cur and buf:
            sections.setdefault(cur, []).extend(buf)
        buf = []
    for ln in lines:
        is_heading = (ln.isupper() and len(ln) < 140) or any(h in ln.upper() for h in TARGET_SECTIONS)
        if is_heading:
            up = ln.upper()
            matched = None
            for target in TARGET_SECTIONS:
                if target in up:
                    matched = target
                    break
            if matched:
                flush(); cur = matched
            else:
                flush(); cur = None
        else:
            if cur: buf.append(ln)
    flush()
    return sections

def sentence_split(text: str) -> List[str]:
    out = []
    for chunk in re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", text):
        c = chunk.strip()
        if c:
            out.append(c)
    return out

@dataclass
class Clause:
    sentence: str
    section: str
    modality: float
    is_obl: bool
    is_perm: bool

def modality_scalar(s: str) -> float:
    s2 = s.lower()
    if "must not" in s2 or "shall not" in s2: return -1.0
    if "must" in s2 or "shall" in s2:       return +1.0
    if "may not" in s2:                      return -0.3
    if "may" in s2:                          return +0.3
    return 0.0

def extract_clauses(sections: Dict[str, List[str]]) -> List[Clause]:
    clauses = []
    for sec, lines in sections.items():
        text = " ".join(lines)
        for sent in sentence_split(text):
            is_obl = bool(OBLIGATION_PAT.search(sent))
            is_perm = bool(PERMISSION_PAT.search(sent))
            if not (is_obl or is_perm): continue
            clauses.append(Clause(sent, sec, modality_scalar(sent), is_obl, is_perm))
    return clauses

def pairs_from_clauses(clauses: List[Clause], max_pairs=60) -> List[Tuple[Clause, Clause, int]]:
    pairs = []
    for o in clauses:
        if not o.is_obl: continue
        for p in clauses:
            if not p.is_perm: continue
            if o.section != p.section: continue
            # Bypass flag (B) if permission has explicit exception terms
            B = 1 if any(term in p.sentence.lower() for term in EXC_TERMS) else 0
            pairs.append((o, p, B))
            if len(pairs) >= max_pairs: return pairs
    return pairs

# ---------- CALE API ----------
def cale_analyze(api_base: str, r1: dict, r2: dict) -> dict:
    url = f"{api_base.rstrip('/')}/v1/law/analyze"
    r = requests.post(url, headers={"Content-Type":"application/json"}, json={"rule1": r1, "rule2": r2}, timeout=60)
    r.raise_for_status()
    return r.json()

def compute_dv(sent_o: str, sent_p: str) -> float:
    v = 0.0
    if KW_GOING.search(sent_o) or KW_GOING.search(sent_p):   v += CFG["dv"]["going_concern"]
    if KW_BREACH.search(sent_o) or KW_BREACH.search(sent_p): v += CFG["dv"]["breach_keywords"]
    if KW_COVENANT.search(sent_o) or KW_COVENANT.search(sent_p): v += CFG["dv"]["covenant_words"]
    if RATIO_PAT.search(sent_o) or RATIO_PAT.search(sent_p): v += CFG["dv"]["ratio_weakness"]
    return min(1.0, v)

def filing_score(ticker: str, as_of: dt.date, ua: str, api_base: str) -> Tuple[Optional[float], Optional[dict], List[dict]]:
    """Return (best_CCE, best_feature_row, evidence_list)."""
    try:
        cik_map = get_cik_map(ua)
    except Exception as e:
        return None, None, []
    cik = cik_map.get(ticker.upper())
    if not cik: return None, None, []

    sub = fetch_submissions(cik, ua)
    flist = filings_before(sub, as_of, forms=("10-K","10-Q"))
    if not flist: return None, None, []

    filing = flist[-1]  # latest before date
    html = fetch_primary_doc(cik, filing["accession"], filing["primary"], ua)
    lines = strip_html_to_lines(html)
    secs = find_sections(lines)
    clauses = extract_clauses(secs)
    pairs = pairs_from_clauses(clauses, max_pairs=60)
    if not pairs: return None, None, []

    evidences = []
    best = None
    for o, p, B in pairs:
        dv = compute_dv(o.sentence, p.sentence)
        r1 = {"text": o.sentence, "jurisdiction": "US", "statute": "Debt Covenant", "section": o.section, "enactment_year": as_of.year}
        r2 = {"text": p.sentence, "jurisdiction": "US", "statute": "Management Guidance", "section": p.section, "enactment_year": as_of.year}
        try:
            mets = cale_analyze(api_base, r1, r2)
        except Exception:
            continue
        C  = float(mets.get("conflict_intensity", 0.0))
        Ab = float(mets.get("authority_balance", 0.0))
        S  = float(mets.get("semantic_overlap", 0.0))
        Dt = float(mets.get("temporal_drift", 0.0))
        cce = max(0.0, min(1.0, C * Ab * dv))
        row = {"ticker": ticker, "as_of": str(as_of), "filing_date": str(filing["date"]), "form": filing["form"],
               "C": C, "Ab": Ab, "Dv": dv, "B": float(B), "S": S, "Dt": Dt, "CCE": cce,
               "o_sentence": o.sentence, "p_sentence": p.sentence, "section": o.section}
        evidences.append(row)
        if best is None or cce > best["CCE"]:
            best = row
    if best is None:
        return None, None, evidences
    return best["CCE"], best, evidences

def previous_filing_score(ticker: str, as_of: dt.date, ua: str, api_base: str) -> Optional[float]:
    """Return best CCE from the filing BEFORE the latest (i.e., second most recent)."""
    try:
        cik_map = get_cik_map(ua)
    except Exception:
        return None
    cik = cik_map.get(ticker.upper())
    if not cik: return None
    sub = fetch_submissions(cik, ua)
    flist = filings_before(sub, as_of, forms=("10-K","10-Q"))
    if len(flist) < 2: return None
    prev = flist[-2]
    html = fetch_primary_doc(cik, prev["accession"], prev["primary"], ua)
    lines = strip_html_to_lines(html)
    secs = find_sections(lines)
    clauses = extract_clauses(secs)
    pairs = pairs_from_clauses(clauses, max_pairs=60)
    if not pairs: return None
    best_cce = 0.0
    for o, p, B in pairs:
        dv = compute_dv(o.sentence, p.sentence)
        r1 = {"text": o.sentence, "jurisdiction": "US", "statute": "Debt Covenant", "section": o.section, "enactment_year": as_of.year}
        r2 = {"text": p.sentence, "jurisdiction": "US", "statute": "Management Guidance", "section": p.section, "enactment_year": as_of.year}
        try:
            mets = cale_analyze(api_base, r1, r2)
        except Exception:
            continue
        C  = float(mets.get("conflict_intensity", 0.0))
        Ab = float(mets.get("authority_balance", 0.0))
        cce = max(0.0, min(1.0, C * Ab * dv))
        if cce > best_cce: best_cce = cce
    return best_cce

# ---------- Logistic combiner (dependency-free) ----------
class Logistic:
    def __init__(self, l2=0.0, lr=0.05, max_iter=2000, seed=42):
        self.l2 = l2; self.lr = lr; self.max_iter = max_iter; self.rng = random.Random(seed)
        self.w = None

    @staticmethod
    def _sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w = np.zeros(d)
        lr = self.lr
        for it in range(self.max_iter):
            z = X @ self.w
            p = self._sigmoid(z)
            grad = (X.T @ (p - y))/n + self.l2 * self.w
            self.w -= lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w
        return self._sigmoid(z)

    def stderr(self, X: np.ndarray) -> np.ndarray:
        # approximate std errors from Hessian diagonal
        p = self.predict_proba(X)
        W = p*(1-p)
        XtWX = X.T @ (X * W[:,None]) + self.l2*np.eye(X.shape[1])
        try:
            cov = np.linalg.inv(XtWX)
            return np.sqrt(np.diag(cov))
        except Exception:
            return np.full(X.shape[1], np.nan)

# ---------- Main validator ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--controls-csv", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--user-agent", required=True)
    ap.add_argument("--out-dir", default="reports/leverage_alpha")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    events = pd.read_csv(args.events_csv)
    controls = pd.read_csv(args.controls_csv)

    rows = []
    evidence_all = []

    # Distressed
    for _, r in events.iterrows():
        tkr = str(r["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(r["event_date"]))
        cce, feat, ev = filing_score(tkr, as_of, args.user_agent, args.api_base)
        if cce is None: continue
        prev = previous_filing_score(tkr, as_of, args.user_agent, args.api_base)
        dCCE = (cce - prev) if (prev is not None) else 0.0
        has_prev = 0 if (prev is None) else 1
        feat_row = {
            "ticker": tkr, "as_of": str(as_of), "label": 1,
            "C": feat["C"], "Ab": feat["Ab"], "Dv": feat["Dv"], "B": feat["B"], "S": feat["S"], "Dt": feat["Dt"],
            "CCE": feat["CCE"], "dCCE": dCCE, "has_prev": has_prev
        }
        rows.append(feat_row)
        # keep top pair evidence for explainability
        evidence_all.append({k:feat[k] for k in ["ticker","as_of","filing_date","form","section","o_sentence","p_sentence","CCE"]})

    # Controls
    for _, r in controls.iterrows():
        tkr = str(r["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(r["as_of_date"]))
        cce, feat, ev = filing_score(tkr, as_of, args.user_agent, args.api_base)
        if cce is None: continue
        prev = previous_filing_score(tkr, as_of, args.user_agent, args.api_base)
        dCCE = (cce - prev) if (prev is not None) else 0.0
        has_prev = 0 if (prev is None) else 1
        feat_row = {
            "ticker": tkr, "as_of": str(as_of), "label": 0,
            "C": feat["C"], "Ab": feat["Ab"], "Dv": feat["Dv"], "B": feat["B"], "S": feat["S"], "Dt": feat["Dt"],
            "CCE": feat["CCE"], "dCCE": dCCE, "has_prev": has_prev
        }
        rows.append(feat_row)
        evidence_all.append({k:feat[k] for k in ["ticker","as_of","filing_date","form","section","o_sentence","p_sentence","CCE"]})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows produced — check API availability, SEC UA, or CSVs.")
        return

    # Save raw features
    features_csv = os.path.join(args.out_dir, "event_scores_delta.csv")
    df.to_csv(features_csv, index=False)

    # Baseline AUC on CCE
    y = df["label"].values
    scores_base = df["CCE"].values
    try:
        auc_base = roc_auc_score(y, scores_base)
    except Exception:
        # manual AUC if sklearn unavailable
        order = np.argsort(scores_base)
        ranks = np.empty_like(order); ranks[order] = np.arange(len(scores_base))
        pos_ranks = ranks[y==1]
        n1 = (y==1).sum(); n0 = (y==0).sum()
        auc_base = (pos_ranks.mean() - (n1-1)/2) / n0

    # t-test on means
    d = df[df.label==1]["CCE"].values
    c = df[df.label==0]["CCE"].values
    t_p = stats.ttest_ind(d, c, equal_var=False).pvalue if (len(d)>1 and len(c)>1) else float('nan')

    # Logistic features
    feats = CFG["ablation"]["feature_set"]
    X_list = []
    for _, r in df.iterrows():
        v = []
        for f in feats:
            v.append(float(r[f]) if f in r else 0.0)
        # Add bias term
        v.insert(0, 1.0)
        X_list.append(v)
    X = np.array(X_list, dtype=float)
    ybin = y.astype(float)

    # Standardize non-bias columns
    Xm = X.copy()
    mu = Xm[:,1:].mean(axis=0); sd = Xm[:,1:].std(axis=0) + 1e-9
    Xm[:,1:] = (Xm[:,1:] - mu)/sd

    # Fit logistic
    logi = Logistic(l2=CFG["ablation"]["l2_reg"], lr=CFG["ablation"]["lr"], max_iter=CFG["ablation"]["max_iter"], seed=CFG["ablation"]["seed"])
    logi.fit(Xm, ybin)
    probs = logi.predict_proba(Xm)
    try:
        auc_log = roc_auc_score(y, probs)
    except Exception:
        order = np.argsort(probs); ranks = np.empty_like(order); ranks[order] = np.arange(len(probs))
        pos_ranks = ranks[y==1]; n1 = (y==1).sum(); n0 = (y==0).sum()
        auc_log = (pos_ranks.mean() - (n1-1)/2) / n0

    # Std errors (approx)
    se = logi.stderr(Xm)

    # Save coefficients
    coef_path = os.path.join(args.out_dir, "logistic_coeffs.json")
    with open(coef_path, "w") as f:
        names = ["bias"] + feats
        json.dump({"names": names, "weights": list(map(float, logi.w)), "stderr": list(map(float, se)), "mu": mu.tolist(), "sd": sd.tolist()}, f, indent=2)

    # Save evidence pairs
    pairs_path = os.path.join(args.out_dir, "top_pairs_delta.json")
    with open(pairs_path, "w") as f:
        json.dump(evidence_all[:50], f, indent=2)

    # Verdicts
    auc_s = CFG["validation"]["auc_strong"]; auc_r = CFG["validation"]["auc_real"]; auc_w = CFG["validation"]["auc_weak"]
    def bucket(auc):
        if auc >= auc_s: return "🟢 STRONG — proceed to backtest immediately"
        if auc >= auc_r: return "🟡 REAL — expand sample & tune"
        if auc >= auc_w: return "🟠 WEAK — iterate parsing/authority/ΔCCE"
        return "🔴 NO SIGNAL — fix features and re-test"

    print("\n=== CALE ΔCCE + Logistic Combiner Event Study ===")
    print(f"N(distressed)={int((df.label==1).sum())}, mean CCE={np.mean(d) if len(d)>0 else float('nan'):.3f}")
    print(f"N(control)   ={int((df.label==0).sum())}, mean CCE={np.mean(c) if len(c)>0 else float('nan'):.3f}")
    print(f"Baseline CCE AUC: {auc_base:.3f}   Verdict: {bucket(auc_base)}")
    print(f"Logistic  AUC:    {auc_log:.3f}   Verdict: {bucket(auc_log)}")
    print(f"T-test (CCE means) p-value: {t_p:.4f}")
    print("\nLogistic weights (bias + " + ", ".join(CFG['ablation']['feature_set']) + "):")
    names = ["bias"] + CFG["ablation"]["feature_set"]
    for i, n in enumerate(names):
        se_i = float(se[i]) if i < len(se) else float('nan')
        print(f"  {n:>6}: {float(logi.w[i]): .4f}  (± {se_i:.4f})")
    print(f"\nSaved features: {features_csv}")
    print(f"Saved logistic coeffs: {coef_path}")
    print(f"Saved top evidence pairs: {pairs_path}")
    print("Note: Research tool only — not investment advice.")


if __name__ == "__main__":
    random.seed(CFG.get("ablation", {}).get("seed", 42))
    np.random.seed(CFG.get("ablation", {}).get("seed", 42))
    main()
