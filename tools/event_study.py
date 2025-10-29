#!/usr/bin/env python3
"""Offline CALE event study using SEC fixtures and pandas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))

import tools.event_study_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    core.configure_cli(parser)
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Echo the JSON metrics payload after the run",
    )
    args = parser.parse_args()

    metrics = core.run_event_study(args)
    if args.print_metrics:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

