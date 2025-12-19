#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from potatobacon.tariff.e2e_runner import TariffE2ERunner


def main() -> None:
    parser = argparse.ArgumentParser(description="CALE-TARIFF end-to-end harness")
    parser.add_argument("--mode", choices=["engine", "http"], default="engine")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--law-context", default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out", dest="out", default=None)
    parser.add_argument("--dataset", dest="dataset", default=Path("tests/data/e2e_pilot_pack.json"))
    args = parser.parse_args()

    runner = TariffE2ERunner(
        dataset_path=Path(args.dataset),
        mode=args.mode,
        seed=args.seed,
        api_base=args.api_base,
        api_key=args.api_key,
        law_context_override=args.law_context,
        output_path=Path(args.out) if args.out else None,
    )
    result = runner.run()

    counts = result.counts
    print(f"Report written to: {result.report_path}")
    print(
        " ".join(
            [
                f"OK_OPTIMIZED={counts.get('OK_OPTIMIZED',0)}",
                f"OK_BASELINE_ONLY={counts.get('OK_BASELINE_ONLY',0)}",
                f"INSUFFICIENT_INPUTS={counts.get('INSUFFICIENT_INPUTS',0)}",
                f"INSUFFICIENT_RULE_COVERAGE={counts.get('INSUFFICIENT_RULE_COVERAGE',0)}",
                f"ERROR={counts.get('ERROR',0)}",
            ]
        )
    )
    print(
        f"Determinism={'PASS' if result.determinism.passed else 'FAIL'} payload stability {result.determinism.payload_match_rate:.1%}"
    )
    print(f"Proof replay pass rate {result.proof_replay_pass_rate:.1%}")
    if result.sku_results:
        top = max(
            [res for res in result.sku_results if res.annual_savings is not None],
            key=lambda r: r.annual_savings or 0,
            default=None,
        )
        if top:
            print(f"Top annual savings: {top.sku_id} -> {top.annual_savings}")


if __name__ == "__main__":
    main()
