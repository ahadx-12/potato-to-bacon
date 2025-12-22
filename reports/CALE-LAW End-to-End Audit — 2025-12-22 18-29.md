# CALE-LAW End-to-End Audit — 2025-12-22 18:29 UTC

## Overview
- Ingested HTS US 2025 slice from structured chapter extracts and wired it into the tariff context registry.
- Attached duty/citation metadata to tariff PolicyAtoms and surfaced provenance through dossier and suggest responses.
- Added deterministic fixtures, manifests, and coverage-backed tests for the new HTS slice pipeline.

## Coverage %
- pytest --cov run reported ~83% overall coverage for src/potatobacon (see test run log).

## Latency
- System tests executed locally without explicit latency sampling; future runs should capture median endpoint timings for tariff dossier/suggest flows.

## Provenance Samples
- HTS_ELECTRONICS_SIGNAL_LOW_VOLT → heading 8544, note HD8544_NOTE_VOLTAGE, duty 1.0%, source HTSUS 2025 low-voltage cables.
- HTS_6404_11_90 → heading 6404, footwear textile/rubber >50% contact, duty 37.5%, source HTSUS 2025 Ch64 sample.

## Graph Samples
- Dependency/provenance graph export not generated in this run; retain latest solver traces for follow-up visualization.

## Fix List
- Replaced hard-coded HTS atoms with data-driven ingestion (chapters 64/73/85 slice) and emitted a new HTS_US_2025_SLICE manifest.
- Enriched tariff provenance builders to include HTS citations/metadata and refreshed DUTY_RATES from ingested data.
- Added regression tests for manifest hash stability, citation metadata, and end-to-end dossier coverage of the new slice context.

## Next Tasks
- Extend ingestion to additional HTS chapters and map more note-derived conditions.
- Capture latency metrics during system runs and persist representative dependency graphs.
- Expand provenance sampling in proofs to include hashed note text and manifest references.
