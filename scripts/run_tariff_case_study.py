from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from potatobacon.api.security import set_rate_limit
from potatobacon.api.app import app
from potatobacon.proofs.canonical import canonical_json
from potatobacon.tariff.case_study import run_case_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic tariff case study and capture artifacts.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports/case_studies")
    args = parser.parse_args()

    os.environ.setdefault("CALE_API_KEYS", "case-study-key")
    os.environ.setdefault("CALE_RATE_LIMIT_PER_MINUTE", "100")
    set_rate_limit(100)

    output_dir = Path(args.output_dir)
    with TestClient(app, headers={"X-API-Key": "case-study-key"}) as client:
        result = run_case_study(client, seed=args.seed, output_dir=output_dir, timestamp=args.timestamp)

    print(canonical_json(result))


if __name__ == "__main__":
    main()
