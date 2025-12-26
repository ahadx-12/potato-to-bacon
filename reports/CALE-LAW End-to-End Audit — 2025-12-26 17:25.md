# CALE-LAW End-to-End Audit — 2025-12-26 17:25

## Overview
- Test run: python -m pytest -q --maxfail=1 --disable-warnings
- Result: PASS
- Notes: Dynamic lever policy gating + deterministic runner updates.

## Coverage %
- Not measured in this run (pytest-cov optional in scripts/run_tests.py).

## Latency
- Not measured in this run.

## Provenance Samples
- Not sampled in this run.

## Graph Samples
- Not sampled in this run.

## Fix List
- Hardened dynamic lever generation with policy gating, deterministic ordering, and documentation levers.
- Added deterministic test runner with optional coverage.
- Added tests for dynamic lever determinism and documentation behavior.

## Next Tasks
- Run coverage-enabled tests via scripts/run_tests.py in CI.
- Collect latency summaries and provenance samples during full system runs.
