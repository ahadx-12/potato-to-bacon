# TEaaS Implementation Roadmap (Executable)

## Current State

- 21 sample HTS lines (footwear, fasteners, electronics, apparel)
- 2,050 full chapter lines (Ch39 Plastics, Ch84 Machinery, Ch87 Vehicles, Ch90 Optical, Ch94 Furniture)
- Z3 solver pipeline: PolicyAtom -> check_scenario -> DutyResult
- Proof engine: SHA-256 hashed, canonically serialized
- Overlay system: Section 232/301 surcharges
- Origin engine: USMCA RVC computation
- Fact compiler: ProductSpec -> boolean facts
- Tenant model: Built but not wired
- API: 14 tariff routes already live

## Build Order

### Sprint 1: Data Expansion + Wiring (THIS SESSION)

**1.1 Expand HTS atom coverage to high-value import categories**
- Add Chapter 61/62 (Apparel/Textiles) - $120B/yr US imports
- Add Chapter 85 (Electronics) - $400B/yr US imports
- Add Chapter 87 (Vehicles/Auto Parts) - $370B/yr US imports
- Add Chapter 44 (Wood/Furniture) - $30B/yr US imports
- Add Chapter 29/38 (Chemicals/Pharma) - $150B/yr US imports
- Add Chapter 95 (Toys/Sporting Goods) - $40B/yr US imports
- Each with guard tokens, duty rates, and optimization mutations

**1.2 Build mutation derivation engine**
- Replace hardcoded felt logic in apply_mutations with Z3-driven entailments
- Build a MutationEngine that discovers optimization paths from atom structure
- For each product category, define valid mutation candidates automatically

**1.3 Wire tenant isolation into tariff routes**
- Add tenant_id to proof records
- Scope SKU storage per tenant
- Add /v1/auth/whoami endpoint

**1.4 Build TEaaS orchestration endpoint**
- POST /v1/teaas/analyze - Single endpoint that:
  - Takes product description + BOM
  - Compiles facts
  - Runs baseline classification
  - Discovers and tests mutation candidates
  - Returns full dossier with proof hash

### Sprint 2: Dashboard + Monitoring (NEXT SESSION)
### Sprint 3: Production Hardening (FUTURE)
