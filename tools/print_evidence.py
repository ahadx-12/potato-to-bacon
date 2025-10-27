#!/usr/bin/env python3
"""Print finance conflict evidence for a single filing."""

from __future__ import annotations

import argparse
from pathlib import Path

from potatobacon.cale import finance_extract


def strip_html(text: str) -> str:
    import re

    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing filing: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--filing-path", required=True)
    parser.add_argument("--prior-path")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    filing = Path(args.filing_path)
    prior = Path(args.prior_path) if args.prior_path else None

    current_text = strip_html(load_text(filing))
    prior_text = strip_html(load_text(prior)) if prior else None

    result = finance_extract.analyse_finance_sections(current_text, prior_text=prior_text, strict=args.strict)

    print(f"Ticker: {args.ticker.upper()}")
    print(f"CCE_prod: {float(result.get('CCE_prod', 0.0)):.3f}")
    print(f"Delta CCE: {float(result.get('delta_cce', 0.0)):.3f}")
    print(f"Conflict intensity: {float(result.get('conflict_intensity', 0.0)):.3f}")
    print(f"Authority balance: {float(result.get('authority_balance', 0.0)):.3f}")
    print(f"Fragility: {float(result.get('fragility', 0.0)):.3f}")
    print("Top conflict pairs:")
    evidence = result.get("evidence", []) or []
    if not evidence:
        print("(none detected)")
    for idx, ev in enumerate(evidence[:5], start=1):
        print(
            f"{idx}. Section {ev.get('section')} | bypass={ev.get('bypass')} | Ab={float(ev.get('authority', 0.0)):.2f} "
            f"| Fragility={float(ev.get('fragility', 0.0)):.2f} | C={float(ev.get('conflict', 0.0)):.2f}\n"
            f"   OBLIGATION: {ev.get('obligation','')[:160]}\n"
            f"   PERMISSION: {ev.get('permission','')[:160]}"
        )


if __name__ == "__main__":
    main()
