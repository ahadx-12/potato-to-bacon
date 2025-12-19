# CALE-TARIFF Readiness Audit — 2025-12-19T21:04:51.599088+00:00

## Executive summary
- Law context: **HTS_US_DEMO_2025**
- OK rate: **86.7%**, Errors: **0.0%**
- Top savings SKUs: RW-FOOT-006: $68547182.26 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.), RW-FOOT-015: $68531910.29 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.), RW-FOOT-003: $31236495.52 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
- Determinism & proof replay: **PASS**

## Coverage snapshot
- apparel_textile: 18 SKUs
- electronics: 22 SKUs
- fastener: 30 SKUs
- footwear: 20 SKUs
- unknown: 0 SKUs

## Category scorecard
| Category | Processed | Optimized | Baseline-only | Insufficient inputs | Insufficient rules | Errors | Avg annual savings | Avg risk | Load-bearing evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apparel_textile | 18 | 6 | 12 | 0 | 0 | 0 | $2921062.64 | 30.0 | 77.8% |
| electronics | 22 | 22 | 0 | 0 | 0 | 0 | $977981.26 | 25.0 | 75.8% |
| fastener | 30 | 30 | 0 | 0 | 0 | 0 | $1879197.59 | 35.0 | 84.4% |
| footwear | 20 | 20 | 0 | 0 | 0 | 0 | $15938477.34 | 39.0 | 8.8% |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | 0.0% |

## Evidence quality summary
- Facts with evidence: 1209 / 2522
- Snippets captured: 1209
- Evidence requested: **True**
- Evidence coverage: 47.9%
- Load-bearing evidence: 60.3%

## Origin and AD/CVD overview
- Origin provided: 62.2%
- Requires origin data: 37.8%
- AD/CVD possible: 7.8%

## Proof replay integrity summary
- Determinism check: **PASS**
- Proof payload hash stability: 100.0%

## Risk distribution summary
- A: 22
- B: 56

## Known limitations
- Highest baseline-only: apparel_textile (12)
- Origin missing rate: 37.8%
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
RW-FOOT-006: $68547182.26 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-015: $68531910.29 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-003: $31236495.52 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-008: $26386075.89 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-020: $22240762.10 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-002: $18624354.49 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-009: $17351986.62 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-014: $15794220.28 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-011: $11200682.54 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)
RW-FOOT-017: $9614978.95 (Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.)

