# tools/sec_fetch.py
import json, os, re, time, pathlib, datetime as dt
from typing import Dict, List, Optional, Tuple
import requests

DATA_ROOT = pathlib.Path("data/sec")
SUBMISSIONS_DIR = DATA_ROOT / "submissions"
EDGAR_DIR = DATA_ROOT / "edgar" / "data"

# Hard-coded CIKs for our 40 tickers (zero-padded to 10)
TICKER_TO_CIK: Dict[str, str] = {
    # Distressed (20)
    "BBBY": "0000886158",
    "MPW": "0001287865",
    "UPST": "0001647639",
    "RVLV": "0001649739",
    "CHK":  "0000895126",
    "SABR": "0001597033",
    "AAL":  "0000006201",
    "CVNA": "0001690820",
    "DISH": "0001001082",
    "HTZ":  "0001657853",
    "PENN": "0000921738",
    "BYND": "0001655210",
    "FIVE": "0001177609",
    "SDC":  "0001668010",
    "RCL":  "0000884887",
    "ABR":  "0001253986",
    "PAA":  "0001070423",
    "RIDE": "0001754586",
    "TUP":  "0001008654",
    "RIVN": "0001874178",
    # Controls (20)
    "AAPL": "0000320193",
    "PEP":  "0000077476",
    "WMT":  "0000104169",
    "PG":   "0000080424",
    "GOOGL":"0001652044",
    "MA":   "0001141391",
    "V":    "0001403161",
    "HD":   "0000354950",
    "KO":   "0000021344",
    "COST": "0000909832",
    "MRK":  "0000310158",
    "UNH":  "0000731766",
    "LMT":  "0000936468",
    "UPS":  "0001090727",
    "TGT":  "0000027419",
    "ORCL": "0001341439",
    "PFE":  "0000078003",
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "JNJ":  "0000200406",
}

SEC_BASE = "https://data.sec.gov"
UA_DEFAULT = "CALE-Research/0.4 (contact: you@example.com)"

def ensure_dirs():
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)

def _get(url: str, ua: str) -> Optional[requests.Response]:
    try:
        return requests.get(url, headers={"User-Agent": ua}, timeout=30)
    except Exception:
        return None

def load_submissions(cik: str, ua: str = UA_DEFAULT) -> Optional[dict]:
    ensure_dirs()
    path = SUBMISSIONS_DIR / f"CIK{cik}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    resp = _get(url, ua)
    if resp and resp.ok:
        path.write_text(resp.text)
        return resp.json()
    return None

def pick_last_10k_10q_before(sub: dict, event_date: dt.date) -> List[dict]:
    if not sub or "filings" not in sub or "recent" not in sub["filings"]:
        return []
    r = sub["filings"]["recent"]
    forms = r.get("form", [])
    dates = r.get("filingDate", [])
    accs  = r.get("accessionNumber", [])
    prims = r.get("primaryDocument", [])
    rows = []
    for form, d, acc, prim in zip(forms, dates, accs, prims):
        if form not in ("10-K", "10-Q"):
            continue
        try:
            fdate = dt.datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            continue
        if fdate < event_date:
            rows.append({"form": form, "date": d, "acc": acc, "prim": prim})
    # keep last 10-K and last 10-Q
    last_10k = max([x for x in rows if x["form"]=="10-K"], key=lambda x: x["date"], default=None)
    last_10q = max([x for x in rows if x["form"]=="10-Q"], key=lambda x: x["date"], default=None)
    out = [x for x in (last_10k, last_10q) if x]
    # sort by date desc
    return sorted(out, key=lambda x: x["date"], reverse=True)

def ensure_filing_html(cik: str, acc: str, prim: str, ua: str = UA_DEFAULT) -> Optional[pathlib.Path]:
    """
    Save primary document to: data/sec/edgar/data/{cik}/{acc_no_digits}/{prim}
    """
    ensure_dirs()
    acc_nodash = acc.replace("-", "")
    tgt_dir = EDGAR_DIR / cik / acc_nodash
    tgt_dir.mkdir(parents=True, exist_ok=True)
    tgt_path = tgt_dir / prim
    if tgt_path.exists():
        return tgt_path
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{prim}"
    resp = _get(url, ua)
    if resp and resp.ok and len(resp.text) > 2000:  # crude sanity
        tgt_path.write_text(resp.text, encoding="utf-8", errors="ignore")
        time.sleep(0.2)
        return tgt_path
    return None

def fetch_latest_before(ticker: str, event_date: str, ua: str = UA_DEFAULT) -> List[pathlib.Path]:
    """
    Returns list of local HTML paths for latest 10-K and 10-Q before event_date.
    """
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        return []
    sub = load_submissions(cik, ua=ua)
    try:
        edate = dt.datetime.strptime(event_date, "%Y-%m-%d").date()
    except Exception:
        edate = dt.date(2100,1,1)
    rows = pick_last_10k_10q_before(sub, edate)
    out = []
    for row in rows:
        p = ensure_filing_html(cik, row["acc"], row["prim"], ua=ua)
        if p:
            out.append(p)
    return out
