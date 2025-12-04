# CALE-LAW End-to-End Audit — 2025-12-04 00:23

## Overview
System arbitrage sync/async transparency checks with deterministic seeds and manifest ingest.

## Coverage %
System scope only (tests/system); broader coverage not measured in this run.

## Latency
Not captured for this ad-hoc audit (TestClient in-process).

## Seed Reproducibility
Same seed reproducible: yes (golden scenario + score).

## Provenance Samples
[
  {
    "step": 1,
    "jurisdiction": "US",
    "rule_id": "US_TAX",
    "type": "STATUTE",
    "role": "obligation",
    "summary": "US corporations must pay corporate income tax on domestic profits.",
    "atom_id": "US_TAX_atom_0",
    "urn": "urn:law:us:irc:s11",
    "citations": [
      "IRC 11"
    ],
    "effective_date": null
  },
  {
    "step": 2,
    "jurisdiction": "IE",
    "rule_id": "IE_TAX",
    "type": "STATUTE",
    "role": "permission",
    "summary": "Irish entities may claim R&D credits reducing effective tax.",
    "atom_id": "IE_TAX_atom_1",
    "urn": "urn:law:ie:tca:s766",
    "citations": [
      "TCA 766"
    ],
    "effective_date": null
  },
  {
    "step": 3,
    "jurisdiction": "KY",
    "rule_id": "KY_TAX",
    "type": "STATUTE",
    "role": "obligation",
    "summary": "Kentucky allows deductions for manufacturing investments. MUST apply",
    "atom_id": "KY_TAX_atom_2",
    "urn": "urn:law:ky:kyrev:s141",
    "citations": [
      "KYREV 141"
    ],
    "effective_date": null
  }
]

## Graph Samples
{
  "nodes": [
    {
      "id": "US_TAX",
      "jurisdiction": "US",
      "label": "IRC 11",
      "urn": "urn:law:us:irc:s11",
      "citations": [
        "IRC 11"
      ]
    },
    {
      "id": "IE_TAX",
      "jurisdiction": "IE",
      "label": "TCA 766",
      "urn": "urn:law:ie:tca:s766",
      "citations": [
        "TCA 766"
      ]
    }
  ],
  "edges": [
    {
      "from_id": "US_TAX",
      "to_id": "IE_TAX",
      "relation": "sequence"
    }
  ]
}

## Schema Gaps
- None observed in this run; new dossier fields were already exercised in ArbitrageDossierModel and response payloads.

## Fix List
- Added deterministic arbitrage scoring components and provenance metadata.
- Added system tests for sync and async hunts verifying ArbitrageDossier v2 fields.

## Next Tasks
- Track latency metrics across runs and surface in reports.
- Expand coverage to additional domains beyond tax for arbitrage hunts.