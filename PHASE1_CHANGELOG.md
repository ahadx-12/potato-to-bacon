# TEaaS Full Implementation Changelog — Phases 1-3

## What Was Done

### 1. HTS Data Foundation (USITC Ingest Pipeline)

**Files created:**
- `src/potatobacon/tariff/hts_ingest/usitc_fetcher.py` — Downloads machine-readable HTS data from USITC (`hts.usitc.gov/reststop/`). Supports bulk JSON download, REST keyword search, and local edition storage with SHA-256 hashing. Includes retry with exponential backoff.
- `src/potatobacon/tariff/hts_ingest/usitc_parser.py` — Parses USITC API records into internal `TariffLine` format. Handles all USITC duty rate formats (ad valorem %, specific ¢/kg, compound rates, "Free"). Tracks HTS indent hierarchy so parent descriptions flow to children for guard token generation.
- `src/potatobacon/tariff/hts_ingest/guard_token_gen.py` — Auto-generates guard tokens from HTS descriptions using keyword extraction. Maps 100+ material, product type, construction, and attribute keywords to boolean predicates that Z3 can match against BOM-derived facts. Tokens inherit from parent descriptions in the HTS hierarchy.

### 2. Versioned HTS Store with Change Tracking

**Files created:**
- `src/potatobacon/tariff/hts_ingest/versioned_store.py` — Manages versioned HTS data with edition registry, diff engine, and SKU alerting. Computes line-by-line diffs between editions (added/removed/rate_changed/description_changed), tracks affected chapters/headings, and cross-references against SKU classifications to generate change alerts.

### 3. Z3 Entailment Extraction (Hardcoded Logic Replacement)

**Files modified:**
- `src/potatobacon/tariff/engine.py` — Replaced hardcoded felt-only logic in `apply_mutations()` with a general entailment rule engine. Rules are evaluated in a fixed-point loop until no new facts are derived. Supports: felt→textile coverage, material metal derivation, USMCA assembly eligibility, green energy exemptions, battery→electronics inference, cable assembly entailments.

### 4. Per-Step Proof Chain

**Files created:**
- `src/potatobacon/proofs/proof_chain.py` — SHA-256 chained proof of every analysis step. Each step (BOM input → fact compilation → Z3 solve → mutation discovery → classification → dossier) gets its own hash, chained to the previous step. The final hash verifies the entire pipeline. Includes `verify()` for chain integrity checks.

**Files modified:**
- `src/potatobacon/api/routes_teaas.py` — TEaaS `/v1/teaas/analyze` endpoint now builds a 6-step proof chain and includes it in the response alongside the existing proof record.

### 5. Test Fixes

**Files modified:**
- `tests/tariff/test_hts_ingest.py` — Fixed hash determinism test to clear ingest cache between runs and removed brittle hardcoded hash assertion.
- `tests/tariff/test_tariff_contexts.py` — Fixed atom count assertion to account for GRI atoms added by context loader.

## Files Updated Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/potatobacon/tariff/hts_ingest/usitc_fetcher.py` | **NEW** | USITC data download + local storage |
| `src/potatobacon/tariff/hts_ingest/usitc_parser.py` | **NEW** | USITC → TariffLine conversion + duty rate parsing |
| `src/potatobacon/tariff/hts_ingest/guard_token_gen.py` | **NEW** | Auto-generate guard tokens from HTS descriptions |
| `src/potatobacon/tariff/hts_ingest/versioned_store.py` | **NEW** | Versioned HTS store + diff engine + SKU alerting |
| `src/potatobacon/proofs/proof_chain.py` | **NEW** | Per-step SHA-256 proof chain |
| `src/potatobacon/tariff/engine.py` | **MODIFIED** | Entailment rules replace hardcoded felt logic |
| `src/potatobacon/api/routes_teaas.py` | **MODIFIED** | Proof chain wired into TEaaS endpoint |
| `tests/tariff/test_hts_ingest.py` | **MODIFIED** | Hash test robustness |
| `tests/tariff/test_tariff_contexts.py` | **MODIFIED** | GRI atom count fix |

## How to Use

### Fetch USITC Data
```python
from potatobacon.tariff.hts_ingest.usitc_fetcher import USITCFetcher

fetcher = USITCFetcher()
edition = fetcher.fetch_current_edition()  # Downloads ~12K records
records = fetcher.load_local_edition(edition.edition_id)
```

### Parse into TariffLines
```python
from potatobacon.tariff.hts_ingest.usitc_parser import parse_usitc_edition

lines = parse_usitc_edition(records, effective_date="2025-01-01")
# Each line has auto-generated guard_tokens from its HTS description
```

### Track Schedule Changes
```python
from potatobacon.tariff.hts_ingest.versioned_store import VersionedHTSStore

store = VersionedHTSStore()
diff = store.compute_diff("edition_v1", "edition_v2")
alerts = store.find_affected_skus(diff, {"SKU-001": "8703.23.00.50"})
```

### Full TEaaS Analysis (with Proof Chain)
```python
# POST /v1/teaas/analyze
{
    "description": "Steel chassis bolt M12x1.5",
    "origin_country": "CN",
    "import_country": "US",
    "declared_value_per_unit": 2.50,
    "annual_volume": 500000
}
# Response includes proof_chain with 6 verified steps
```

---

## Phase 2: Multi-Tenant API

### 6. Async BOM Analysis Job Queue

**Files created:**
- `src/potatobacon/api/routes_jobs.py` — Async job management for batch BOM analysis. `POST /v1/jobs/analyze` accepts up to 500 BOMs, returns a job_id immediately, processes in a background thread. `GET /v1/jobs/{job_id}` for polling, `GET /v1/jobs` to list tenant jobs. Tenant-isolated (each tenant only sees their own jobs).

### 7. Tenant-Scoped SKU Storage

**Files modified:**
- `src/potatobacon/tariff/sku_store.py` — Added `get_tenant_sku_store(tenant_id)` that returns a tenant-isolated SKU store (separate JSONL file per tenant under `data/tenants/{tenant_id}/skus.jsonl`).
- `src/potatobacon/api/routes_tariff_skus.py` — All SKU CRUD endpoints now resolve the tenant from the API key and use tenant-scoped storage. Each endpoint returns `tenant_id` in responses.

### 8. USITC Live Context Loader

**Files created:**
- `src/potatobacon/tariff/hts_ingest/usitc_context.py` — Bridges the USITC versioned store to the context registry. `load_usitc_context()` reads the current USITC edition, parses it through the guard token generator, and returns PolicyAtoms + metadata compatible with `load_atoms_for_context()`.

---

## Phase 3: Monitoring & Portfolio

### 9. Portfolio Dashboard API

**Files created:**
- `src/potatobacon/api/routes_portfolio.py` — Dashboard data layer with 4 endpoints:
  - `GET /v1/portfolio/summary` — Aggregate metrics (total SKUs, duty exposure, opportunities, pending alerts)
  - `GET /v1/portfolio/skus` — All SKUs with classification, duty rates, optimization status
  - `GET /v1/portfolio/alerts` — Schedule change alerts for affected SKUs
  - `POST /v1/portfolio/alerts/{id}/acknowledge` — Mark alert as reviewed

### 10. Schedule Change Detection

**Files created:**
- `src/potatobacon/tariff/schedule_monitor.py` — Nightly monitoring engine that:
  1. Diffs current vs previous USITC editions
  2. Cross-references changed headings against tenant SKU classifications
  3. Generates alerts for affected SKUs
  This is the engine behind the recurring revenue model — importers pay to be notified when tariff changes affect their products.

---

## Complete File Inventory

| File | Action | Phase |
|------|--------|-------|
| `src/potatobacon/tariff/hts_ingest/usitc_fetcher.py` | **NEW** | 1 |
| `src/potatobacon/tariff/hts_ingest/usitc_parser.py` | **NEW** | 1 |
| `src/potatobacon/tariff/hts_ingest/guard_token_gen.py` | **NEW** | 1 |
| `src/potatobacon/tariff/hts_ingest/versioned_store.py` | **NEW** | 1 |
| `src/potatobacon/tariff/hts_ingest/usitc_context.py` | **NEW** | 2 |
| `src/potatobacon/proofs/proof_chain.py` | **NEW** | 1 |
| `src/potatobacon/tariff/mutation_engine.py` | **NEW** | 1 |
| `src/potatobacon/tariff/schedule_monitor.py` | **NEW** | 3 |
| `src/potatobacon/api/routes_teaas.py` | **NEW** | 1 |
| `src/potatobacon/api/routes_auth.py` | **NEW** | 1 |
| `src/potatobacon/api/routes_jobs.py` | **NEW** | 2 |
| `src/potatobacon/api/routes_portfolio.py` | **NEW** | 3 |
| `src/potatobacon/tariff/engine.py` | **MODIFIED** | 1 |
| `src/potatobacon/tariff/sku_store.py` | **MODIFIED** | 2 |
| `src/potatobacon/api/routes_tariff_skus.py` | **MODIFIED** | 2 |
| `src/potatobacon/api/app.py` | **MODIFIED** | 1-3 |
| `data/hts_extract/hts_expansion_new.jsonl` | **NEW** | 1 |
| `TEAAS_IMPLEMENTATION.md` | **NEW** | 1 |
| `tests/tariff/test_hts_ingest.py` | **MODIFIED** | 1 |
| `tests/tariff/test_tariff_contexts.py` | **MODIFIED** | 1 |

## API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/teaas/analyze` | POST | Unified BOM → classification → optimization → proof |
| `/v1/auth/whoami` | GET | Tenant identity + plan metadata |
| `/v1/jobs/analyze` | POST | Submit batch BOMs for async analysis |
| `/v1/jobs/{job_id}` | GET | Poll job completion status |
| `/v1/jobs` | GET | List tenant's analysis jobs |
| `/v1/portfolio/summary` | GET | Aggregate portfolio metrics |
| `/v1/portfolio/skus` | GET | All SKUs with duty info |
| `/v1/portfolio/alerts` | GET | Schedule change alerts |
| `/v1/portfolio/alerts/{id}/acknowledge` | POST | Acknowledge alert |
| `/api/tariff/skus` | POST/GET | Tenant-scoped SKU CRUD |
