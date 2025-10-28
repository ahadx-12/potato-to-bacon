import csv, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sec_fetch import fetch_latest_before, UA_DEFAULT
EVENTS = Path("data/events/events.csv")
CONTROLS = Path("data/events/controls.csv")
REPORT_DIR = Path("reports/leverage_alpha")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _rows(path: Path):
    with path.open() as f:
        r = csv.DictReader(f)
        return list(r)

def ensure_filings():
    for row in _rows(EVENTS) + _rows(CONTROLS):
        tkr = row["ticker"].strip().upper()
        date = row["event_date"].strip()
        paths = fetch_latest_before(tkr, date, ua=UA_DEFAULT)
        print(f"[SEC] {tkr} {date} -> {len(paths)} files")
    print("[SEC] Done fetching filings.")

def run_validation():
    # Assumes API is running locally OR your event scripts call the engine directly; if you need local API, start it before running this.
    cmd1 = ["python","tools/event_study.py","--events-csv","data/events/events.csv","--controls-csv","data/events/controls.csv","--api-base","http://127.0.0.1:8000","--user-agent",UA_DEFAULT]
    cmd2 = ["python","tools/event_study_delta.py","--events-csv","data/events/events.csv","--controls-csv","data/events/controls.csv","--api-base","http://127.0.0.1:8000","--user-agent",UA_DEFAULT]
    print("[RUN]"," ".join(cmd1));  rc1 = subprocess.call(cmd1)
    print("[RUN]"," ".join(cmd2));  rc2 = subprocess.call(cmd2)
    if rc1 or rc2: 
        print("[ERR] Validation scripts failed."); sys.exit(1)

def summarize():
    # Expect these to exist from your scripts:
    metrics_path = REPORT_DIR / "metrics.json"
    final_path   = REPORT_DIR / "validation_final_40.json"

    # Fallback if scripts only wrote validation_final.json
    if not metrics_path.exists() and (REPORT_DIR/"validation_final.json").exists():
        metrics_path = REPORT_DIR/"validation_final.json"

    # Load metrics
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        # Normalize keys
        baseline_auc = m.get("baseline",{}).get("auc", m.get("auc"))
        pval         = m.get("baseline",{}).get("p_value", m.get("p_value"))
        delta_auc    = m.get("delta",{}).get("auc", None)
        logistic_auc = m.get("logistic",{}).get("auc", None)
        dens_d = m.get("evidence_density",{}).get("avg_pairs_distressed", m.get("evidence_density",{}).get("distressed_avg", None))
        dens_c = m.get("evidence_density",{}).get("avg_pairs_control", m.get("evidence_density",{}).get("control_avg", None))
        fp_rate = m.get("false_positives_ig",{}).get("rate", m.get("false_positive_rate", None))
    else:
        print("[WARN] metrics.json not found; using defaults.")
        baseline_auc = delta_auc = logistic_auc = fp_rate = None
        pval = dens_d = dens_c = None

    # Verdict rules
    pass_auc  = (baseline_auc is not None) and (baseline_auc >= 0.80)
    pass_p    = (pval is not None) and (pval <= 0.05)
    pass_fp   = (fp_rate is not None) and (fp_rate <= 0.20)
    pass_dens = (dens_d is not None) and (dens_d >= 1.5)

    verdict = "READY" if (pass_auc and pass_p and pass_fp and pass_dens) else "NOT_READY"

    report = {
        "auc": baseline_auc,
        "p_value": pval,
        "false_positive_rate": fp_rate,
        "evidence_density": {"distressed_avg": dens_d, "control_avg": dens_c},
        "delta_auc": delta_auc,
        "logistic_auc": logistic_auc,
        "verdict": verdict,
        "commentary": (
            "Meets thresholds for backtest." if verdict=="READY"
            else "Extraction/sample size requires improvement before backtest."
        ),
    }
    final_path.write_text(json.dumps(report, indent=2))
    print("=== CALE Final Validation (40-ticker) ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    ensure_filings()
    run_validation()
    summarize()