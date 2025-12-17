# CALE-TARIFF Readiness Audit — 2025-12-17T22:31:10.380001+00:00

## Executive summary
- Law context: **HTS_US_DEMO_2025**
- OK rate: **55.6%**, Errors: **0.0%**
- Top savings SKUs: RW-FOOT-006: $68547182.26 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact), RW-FOOT-015: $68531910.29 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact), RW-FOOT-004: $59900584.84 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
- Determinism & proof replay: **PASS**

## Coverage snapshot
- apparel_textile: 20 SKUs
- electronics: 20 SKUs
- fastener: 30 SKUs
- footwear: 20 SKUs
- unknown: 0 SKUs

## Category scorecard
| Category | Processed | OK rate | NO_CANDIDATES | Errors | Avg annual savings | Avg risk | Load-bearing evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apparel_textile | 20 | 0.0% | 20 | 0 | n/a | n/a | 0.0% |
| electronics | 20 | 0.0% | 20 | 0 | n/a | n/a | 0.0% |
| fastener | 30 | 100.0% | 0 | 0 | $1879197.59 | 12.3 | 84.4% |
| footwear | 20 | 100.0% | 0 | 0 | $27161334.84 | 63.0 | 8.8% |
| unknown | 0 | 0.0% | 0 | 0 | n/a | n/a | 0.0% |

## Evidence quality summary
- Facts with evidence: 639 / 1382
- Snippets captured: 639
- Evidence requested: **True**
- Evidence coverage: 46.2%
- Load-bearing evidence: 48.8%

## Origin and AD/CVD overview
- Origin provided: 32.2%
- Requires origin data: 37.8%
- AD/CVD possible: 7.8%

## Proof replay integrity summary
- Determinism check: **PASS**
- Proof payload hash stability: 100.0%

## Risk distribution summary
- A: 30
- C: 20

## Known limitations
- Highest NO_CANDIDATES: apparel_textile, electronics (20)
- Origin missing rate: 67.8%
- AD/CVD possible rate: 7.8%

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
RW-FOOT-006: $68547182.26 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-015: $68531910.29 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-004: $59900584.84 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-001: $46528653.23 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-018: $39430976.26 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-007: $31743861.43 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-003: $31236495.52 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-019: $30482255.10 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-008: $26386075.89 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)
RW-FOOT-016: $22438237.06 (Add >50% felt/textile overlay on outsole to make textile dominant ground contact)

