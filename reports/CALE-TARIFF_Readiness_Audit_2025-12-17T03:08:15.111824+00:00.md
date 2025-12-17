# CALE-TARIFF Readiness Audit — 2025-12-17T03:08:15.111824+00:00

## Executive summary
- Law context: **HTS_US_DEMO_2025**
- OK rate: **67.5%**, Errors: **7.5%**
- Top savings SKUs: RW-FOOT-024: $1780890.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact), RW-FOOT-038: $931500.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact), RW-FOOT-012: $852150.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
- Determinism & proof replay: **PASS**

## Coverage snapshot
- fastener: 19 SKUs
- footwear: 8 SKUs

## Evidence quality summary
- Facts with evidence: 45 / 326
- Snippets captured: 45
- Evidence requested: **True**

## Proof replay integrity summary
- Determinism check: **PASS**
- Proof payload hash stability: 100.0%

## Risk distribution summary
- A: 19
- C: 8

## Known limitations
- NO_CANDIDATES: 10
- Errors: 3
- Gaps: limited electronics/textile coverage; no AD/CVD or origin logic; mutation library narrow.

## Top 10 improvements
- **Expand parser coverage for electronics/textile SKUs** — Impact: High; Effort: Medium; Dependencies: Add category keywords + atoms for PCB housings and apparel blends
- **Broaden mutation library beyond footwear/fasteners** — Impact: High; Effort: Medium; Dependencies: Design defensible mutations for electronics enclosures and apparel
- **Integrate origin/FTA & exclusion logic** — Impact: Medium; Effort: High; Dependencies: Requires origin data ingestion + new rule atoms
- **Persist structured BOM ingestion (CSV/JSON)** — Impact: Medium; Effort: Medium; Dependencies: Add upload/parse pipeline and schema validation
- **Add AD/CVD + ruling/precedent integration** — Impact: High; Effort: High; Dependencies: Link to rulings corpus and maintain versioned citations
- **Improve evidence density and snippet extraction** — Impact: Medium; Effort: Low; Dependencies: More robust keyword windows + BOM parsing
- **Strengthen mutation feasibility constraints** — Impact: Medium; Effort: Medium; Dependencies: Capture engineering constraints + cost models
- **Add seed-aware replay CLI and archived manifests** — Impact: Medium; Effort: Low; Dependencies: Reuse proof payload hash + manifest snapshot
- **Expose batch audit metrics via API** — Impact: Low; Effort: Low; Dependencies: Wrap readiness_eval outputs in JSON endpoint
- **Increase category recall for ambiguous gadgets** — Impact: Medium; Effort: Low; Dependencies: Fallback heuristics + ML embeddings for unknown SKUs

## Top 10 SKUs by annual savings
RW-FOOT-024: $1780890.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-038: $931500.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-012: $852150.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-020: $807300.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-028: $569250.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-005: $478170.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-009: $360180.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-001: $324990.00 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FAST-003: $44000.00 (Switch material from steel to aluminum to qualify for lower-duty classification)
RW-FAST-004: $17248.00 (Switch material from steel to aluminum to qualify for lower-duty classification)

