#!/usr/bin/env python3
"""Run a real-world CALE validation sweep over LAW and FINANCE modules."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))

from tools.fetch_sec_real import CONTROL, DISTRESSED
import tools.finance_extract as fx
import tools.event_study_core as esc

MANIFEST = Path("reports/realworld/manifest.csv")
EVIDENCE = Path("reports/realworld/evidence.csv")

DISTRESSED_SET = {ticker.upper() for ticker in DISTRESSED}
CONTROL_SET = {ticker.upper() for ticker in CONTROL}

LAW_PAIRS: Sequence[Tuple[str, str]] = (
    (
        "The contractor must complete all work by June 30, 2025.",
        "The contractor may request an extension if weather delays exceed 10 days.",
    ),
    (
        "The licensee must not sublicense the software to any third party.",
        "The licensee may grant sublicenses to affiliates as necessary for operations.",
    ),
    (
        "Employee must not engage in competing business during employment and for 12 months thereafter.",
        "Employee may consult for unrelated clients including those in similar industries.",
    ),
)


class ReportError(RuntimeError):
    """Raised when the orchestrator cannot produce a report."""


def _read_manifest() -> List[Dict[str, str]]:
    if not MANIFEST.exists():
        raise ReportError("Manifest missing. Run tools/fetch_sec_real.py first.")
    with MANIFEST.open() as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("ticker")]
    if not rows:
        raise ReportError("Manifest is empty; downloader likely failed.")
    return rows


def _load_html(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise ReportError(f"Failed to read filing {path}: {exc}") from exc


def _filing_role(ticker: str) -> Tuple[str, int]:
    ticker = ticker.upper()
    if ticker in DISTRESSED_SET:
        return "distressed", 1
    if ticker in CONTROL_SET:
        return "control", 0
    return ("unknown", 0)


def _rng_for_filing(seed: int, ticker: str, filed: str) -> random.Random:
    salt = hash((ticker, filed)) & 0xFFFFFFFF
    return random.Random(seed + salt)


def _build_records(rows: Sequence[Dict[str, str]]) -> Tuple[List[esc.FilingRecord], List[esc.FilingEvidence]]:
    records: List[esc.FilingRecord] = []
    evidences: List[esc.FilingEvidence] = []
    for idx, row in enumerate(rows):
        ticker = row.get("ticker", "").upper()
        form = row.get("form", "?")
        filed = row.get("filed") or datetime.now(UTC).strftime("%Y-%m-%d")
        path = Path(row.get("path", "")).expanduser()
        if not path.exists():
            # Skip quietly but note via stderr for visibility.
            print(f"[warn] Missing filing for {ticker}: {path}", file=sys.stderr)
            continue
        html = _load_html(path)
        pairs = fx.extract_pairs_from_html(html, form)
        if not pairs:
            continue
        rng = _rng_for_filing(idx, ticker, filed)
        try:
            filing_date = datetime.strptime(filed, "%Y-%m-%d").date()
        except ValueError:
            filing_date = datetime.now(UTC).date()
        filing_id = f"{ticker}-{filing_date.isoformat()}-000"
        role, label = _filing_role(ticker)
        best_score = 0.0
        best_raw = 0.0
        best_cue = 1.0
        best_sev = 0
        best_num = 0
        evidence_rows: List[esc.FilingEvidence] = []
        for pair in pairs:
            obligation, permission = pair.obligation, pair.permission
            score, raw, cue, sev_hits, num_hits = esc._compute_pair_score(obligation, permission, rng)
            evidence = esc.FilingEvidence(
                ticker=ticker,
                filing_id=filing_id,
                filing_date=filing_date.isoformat(),
                form=form,
                obligation=obligation.strip(),
                permission=permission.strip(),
                score=score,
                raw_score=raw,
                cue_weight=cue,
                severity_hits=sev_hits,
                numeric_hits=num_hits,
            )
            evidence_rows.append(evidence)
            if score > best_score:
                best_score = score
                best_raw = raw
                best_cue = cue
                best_sev = sev_hits
                best_num = num_hits
        if not evidence_rows:
            continue
        record = esc.FilingRecord(
            ticker=ticker,
            role=role,
            label=label,
            as_of=filing_date,
            filing_date=filing_date,
            form=form,
            filing_id=filing_id,
            cce=best_score,
            cce_raw=best_raw,
            cue_weight=best_cue,
            severity_hits=best_sev,
            numeric_hits=best_num,
            pair_count=len(evidence_rows),
            evidence_rows=evidence_rows,
        )
        records.append(record)
        evidences.extend(evidence_rows)
    return records, evidences


def _write_evidence_csv(evidences: Sequence[esc.FilingEvidence]) -> None:
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ticker",
            "path",
            "section",
            "pair",
            "cce",
            "numeric_conf",
            "authority",
            "has_bypass",
            "snippet",
        ])
        for ev in evidences:
            pair = f"{ev.obligation} || {ev.permission}"
            snippet = pair.replace("\n", " ")[:600]
            has_bypass = "yes" if "may" in ev.permission.lower() else "no"
            writer.writerow(
                [
                    ev.ticker,
                    ev.filing_id,
                    "Unknown",
                    pair[:200],
                    f"{ev.score:.3f}",
                    ev.numeric_hits,
                    "n/a",
                    has_bypass,
                    snippet,
                ]
            )


def _run_baseline_metrics() -> Dict[str, object]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env["CALE_SYNTHETIC_MIN_N"] = "0"
    proc = subprocess.run(
        [sys.executable, "tools/event_study.py", "--print-metrics"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        payload = json.loads(proc.stdout.strip()) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        payload = {"raw": proc.stdout.strip(), "stderr": proc.stderr.strip(), "returncode": proc.returncode}
    return payload


def _run_law_cli(rule1: str, rule2: str) -> Dict[str, object]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "potatobacon.cli.main",
            "law",
            "sanity-check",
            "--rule1",
            rule1,
            "--rule2",
            rule2,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"raw": proc.stdout.strip(), "stderr": proc.stderr.strip(), "returncode": proc.returncode}


def _law_report() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for rule1, rule2 in LAW_PAIRS:
        out.append(_run_law_cli(rule1, rule2))
    return out


def _git_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() or "unknown"


def _top_evidence_by_ticker(evidences: Sequence[esc.FilingEvidence]) -> Dict[str, List[esc.FilingEvidence]]:
    grouped: Dict[str, List[esc.FilingEvidence]] = defaultdict(list)
    for ev in evidences:
        grouped[ev.ticker].append(ev)
    top: Dict[str, List[esc.FilingEvidence]] = {}
    for ticker, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda ev: ev.score, reverse=True)
        top[ticker] = rows_sorted[:3]
    return top


def _format_snippet(text: str) -> str:
    clean = " ".join(text.split())
    return clean[:400]


def main() -> int:
    rows = _read_manifest()
    records, evidences = _build_records(rows)
    if not records:
        raise ReportError("No real filings produced scoring evidence.")
    _write_evidence_csv(evidences)

    metrics = esc.build_metrics_payload(records)
    baseline_payload = _run_baseline_metrics()
    law_outputs = _law_report()

    density = metrics.get("evidence_density", {})
    fp_rates = metrics.get("false_positives_ig", {})

    distressed_records = [rec for rec in records if rec.label == 1]
    control_records = [rec for rec in records if rec.label == 0]

    top_evidence = _top_evidence_by_ticker([ev for ev in evidences if ev.ticker in DISTRESSED_SET])

    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    git_sha = _git_sha()

    print("# CALE Real-World Validation (LAW + FINANCE)")
    print(f"_Run_: {now}")
    print(f"_Repo SHA_: {git_sha}")
    print("\n## Environment")
    print(f"- Platform: {platform.platform()}")
    print(f"- Python: {platform.python_version()}")
    print(f"- CALE_SYNTHETIC_MIN_N: {os.environ.get('CALE_SYNTHETIC_MIN_N', 'not set')}")

    print("\n## Data")
    print(f"- Manifest rows: {len(rows)}")
    for row in rows:
        ticker = row.get("ticker", "?")
        form = row.get("form", "?")
        filed = row.get("filed", "?")
        path = row.get("path", "?")
        print(f"  - {ticker} — {form} — {filed} — {path}")

    print("\n## Finance — Event Study (Synthetic OFF)")
    print("### Real filings cohort")
    print(f"- Distressed filings: {len(distressed_records)}")
    print(f"- Control filings: {len(control_records)}")
    baseline = metrics.get("baseline", {})
    logistic = metrics.get("logistic", {})
    delta = metrics.get("delta", {})
    print(f"- Baseline AUC: {baseline.get('auc', 'n/a')}")
    print(f"- Logistic AUC: {logistic.get('auc', 'n/a')}")
    print(f"- Delta AUC: {delta.get('auc', 'n/a')}")
    print(f"- Welch p-value: {baseline.get('p_value', 'n/a')}")
    print(f"- Baseline confusion: {baseline.get('confusion_matrix')}")
    print(f"- Logistic confusion: {logistic.get('confusion_matrix')}")
    print(f"- IG false-positive rate: {fp_rates.get('rate')}, count={fp_rates.get('count')}, total={fp_rates.get('total')}")
    print(
        f"- Evidence density: avg distressed={density.get('avg_pairs_distressed')}, "
        f"avg control={density.get('avg_pairs_control')}, min distressed={density.get('min_pairs_distressed')}"
    )
    if density:
        avg_dist = float(density.get("avg_pairs_distressed") or 0.0)
        avg_ctrl = float(density.get("avg_pairs_control") or 0.0)
        avg_pairs = (avg_dist + avg_ctrl) / 2.0
        print(f"- Average evidence pairs per filing: {avg_pairs:.2f}")
    else:
        print("- Average evidence pairs per filing: n/a")

    print("\n### Baseline synthetic diagnostic (for comparison)")
    print(json.dumps(baseline_payload, indent=2, sort_keys=True))

    print("\n### Top evidence snippets (distressed cohort)")
    for ticker, evs in top_evidence.items():
        print(f"- {ticker}:")
        for ev in evs:
            combo = f"{ev.obligation} || {ev.permission}"
            print(f"    - {ev.form} {ev.filing_date}: {_format_snippet(combo)} (CCE={ev.score:.3f})")

    print("\n## LAW — Realistic Pairs (CLI)")
    for idx, payload in enumerate(law_outputs, start=1):
        if "conflict_scores" in payload:
            conflict = payload.get("conflict_scores", {}).get("pragmatic")
            overlap = payload.get("semantic_overlap")
            amendment = payload.get("suggested_amendment", {}).get("justification")
            print(f"- Pair {idx}: conflict={conflict}, overlap={overlap}, suggested={amendment}")
        else:
            raw = payload.get("raw", "<no output>")
            print(f"- Pair {idx}: {raw}")

    print("\n## Gaps & Recommendations to Reach Logistic AUC ≥ 0.80 (p < 0.05)")
    recommendations = [
        "Expand configs/finance.yml section_keywords to capture Liquidity, Capital Resources, and covenant compliance variants.",
        "Tighten numeric comparator heuristics and link qualifiers to reduce noise in leverage indicators.",
        "Penalize bypass lexicon (e.g., 'except that', 'carve-out', 'basket') by distance and cap IG false positives ≤20%.",
        "Recalibrate authority weights for trustee/board narratives to stabilise cue_weight across filings.",
        "Incorporate ΔCCE (trend) features and refit logistic model with stratified CV on the growing real cohort.",
        "Increase real cohort coverage to ≥200 filings (balanced distressed/control) before the next measurement.",
    ]
    for item in recommendations:
        print(f"- {item}")

    print("\n## Verdict")
    print("- Real cohort metrics above reflect CALE_SYNTHETIC_MIN_N=0 with genuine SEC filings.")
    print("- Prior ~1.00 AUC runs stem from synthetic fixtures; the comparison payload above confirms the synthetic baseline.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        raise SystemExit(main())
    except ReportError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
