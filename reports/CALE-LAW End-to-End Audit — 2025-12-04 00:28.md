# CALE-LAW End-to-End Audit — 2025-12-04 00:28

## Overview
- Exercised secured API flows via FastAPI TestClient against `/v1/version`, `/v1/law/analyze`, `/v1/manifest/bulk_ingest`, and `/v1/manifest/ingest_pdf`.
- Added realistic tax/privacy fixtures and validated ingestion of both text bundles and PDF statutes.
- Rate-limit enforcement verified with a 2s window and per-route counters.

## Coverage %
- Focused system slice; pytest subset (4 tests) passed. Full coverage run not executed in this pass.

## Latency
- Request latencies not sampled (TestClient synchronous harness). Consider instrumenting timing hooks in follow-up runs.

## Provenance Samples
- Bulk ingest manifest hashes observed: `87fce4d89765408b010cbc45b1804bbab96fc36672e3521dcc8aaf5090e78501`.
- PDF ingest manifest hash observed: `15d27a86a5a75ecd48b7ccd4355ea9a88ca8bad4ecfb37040e03740fb1083ed7`.

## Graph Samples
- Scenario metrics after ingest reported active rule lists with entropy/kappa/value_estimate fields populated; dependency graphs not visualized in this run.

## Fix List
- Rate-limit window now derives from `CALE_RATE_WINDOW_SEC` to allow short test windows.
- Added statutory fixtures (US IRC §61, Irish Section 110, Cayman exempt rules, AML privacy) plus ensured PDF statute carries explicit modalities.

## Next Tasks
- Run full pytest suite with coverage to quantify regression surface.
- Capture latency metrics during system tests and emit dependency/provenance graphs for audit trails.
- Consider cleaning generated manifests between runs to keep `out/system-tests` noise low in long-lived environments.
