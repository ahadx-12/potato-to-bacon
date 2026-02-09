# TEaaS (Tariff Engineering-as-a-Service) Product Roadmap

## Product Vision

A SaaS platform where importers upload product BOMs, receive optimized HTS
classifications with cryptographic proof, and monitor their portfolio as tariff
schedules change.  Every analysis is formal-verification-backed, auditable, and
defensible in a CBP audit.

---

## Architecture Overview

```
                        ┌─────────────────────────────┐
                        │      TEaaS Dashboard        │
                        │  (React SPA - /ui/)          │
                        └──────────┬──────────────────┘
                                   │ HTTPS
                        ┌──────────▼──────────────────┐
                        │      FastAPI Backend         │
                        │  /v1/tariff/*  /v1/proofs/*  │
                        │  /v1/law/*     /v1/auth/*    │
                        ├──────────────────────────────┤
                        │  Tenant Isolation Layer      │
                        │  API Key → Tenant Scope      │
                        ├──────────────────────────────┤
                        │  CALE Engine + Z3 Solver     │
                        │  PolicyAtom Pipeline          │
                        │  Proof Engine (SHA-256)       │
                        ├──────────────────────────────┤
                        │  HTS Data Layer              │
                        │  Overlay Rules (232/301)     │
                        │  Origin Engine (USMCA)       │
                        └──────────┬──────────────────┘
                                   │
                        ┌──────────▼──────────────────┐
                        │  Storage                     │
                        │  SQLite (dev) / PostgreSQL   │
                        │  Proof Store (immutable)     │
                        │  Evidence Store (S3/local)   │
                        └─────────────────────────────┘
```

## Deployment Target

- **Platform:** Railway.app (current Procfile already configured)
- **Container:** Docker (Dockerfile exists, needs slimming)
- **Database:** SQLite for MVP, PostgreSQL for production
- **Frontend:** Static React SPA served from /app/web/ via FastAPI StaticFiles mount
- **Domain:** Custom domain via Railway (e.g., app.caletariff.com)

---

## Phase 1: Foundation (THIS SESSION)

**Goal:** Strip physics code, rebrand, establish tenant model, clean API surface.

### 1A. Remove Physics Dead Weight
- Delete: validation/, codegen/, semantics/, core/, units/, parser/dsl_parser.py,
  parser/latex_parser.py, parser/transformers.py, dashboard/tax_dashboard.py, models.py
- Remove: routes_units.py
- Strip from app.py: translate, validate, schema, codegen, manifest endpoints
- Strip physics imports and models from app.py
- Remove physics dependencies from pyproject.toml (sympy, pint, lark)
- Update Dockerfile: remove PyTorch install, slim down

### 1B. Rebrand
- pyproject.toml: name = "caletariff", description updated
- app.py: title = "CALE Tariff API"
- Dockerfile: updated

### 1C. Tenant Isolation Model
- New file: src/potatobacon/api/tenants.py
- API key → tenant_id mapping
- Tenant-scoped proof store and asset persistence
- Rate limiting foundation

### 1D. Clean API Surface for TEaaS
- Keep all /v1/tariff/* routes (already production-shaped)
- Keep /v1/proofs/* routes
- Keep /v1/law-contexts/* routes
- Keep /api/law/arbitrage/* routes (asset management)
- Keep manifest bulk ingest + PDF ingest endpoints
- Add /v1/auth/whoami endpoint for tenant info
- Add /v1/tariff/portfolio endpoint (list all SKUs for a tenant)

---

## Phase 2: Data Pipeline (NEXT SESSION)

**Goal:** Real HTS data ingestion, live tariff schedule monitoring.

### 2A. USITC HTS Data Feed
- Ingest full HTS schedule from USITC public data (hts.usitc.gov)
- Parse HTS chapters into PolicyAtoms automatically
- Build duty rate lookup from General/Special/Column 2 rates
- Store as versioned manifests with hash chain

### 2B. Federal Register Monitor
- Watch for tariff schedule amendments via Federal Register API
- Detect when duty rates change, overlays update, or new exclusions appear
- Trigger re-analysis of affected tenant SKUs
- Push notifications via webhook

### 2C. Enhanced Overlay Engine
- Expand beyond Section 232/301 samples to full overlay corpus
- Add AD/CVD (anti-dumping/countervailing duty) overlays
- Add FTA preference overlays (USMCA, KORUS, etc.)
- Each overlay carries its own PolicyAtoms and proof chain

---

## Phase 3: Dashboard MVP (FUTURE SESSION)

**Goal:** Professional importer-facing web UI.

### Design Language
- Clean, corporate. Think Bloomberg Terminal meets Stripe Dashboard.
- Primary: Deep navy (#0F172A). Accent: Emerald (#10B981) for savings.
  Red (#EF4444) for violations. Amber (#F59E0B) for review-needed.
- Font: Inter for UI, JetBrains Mono for codes/hashes.
- Dense information display. No wasted space.

### Core Views
1. **Portfolio Dashboard** - All SKUs with duty rates, savings potential, status
2. **SKU Dossier View** - Deep dive: baseline vs optimized, provenance, proof
3. **Optimization Studio** - Interactive scenario mutation with live Z3 recalc
4. **Audit Pack Generator** - One-click CBP audit defense dossier (PDF export)
5. **Alert Feed** - Regulatory changes affecting your portfolio

---

## Phase 4: Production Hardening (FUTURE)

- PostgreSQL migration
- Redis caching for Z3 results
- Solver pool (replace global _Z3_LOCK)
- Background job queue (Celery/ARQ) for batch analysis
- Stripe billing integration
- SOC 2 compliance documentation

---

## What Gets Deleted (Phase 1)

```
src/potatobacon/validation/          # Physics guardrails
src/potatobacon/codegen/             # Physics code gen
src/potatobacon/semantics/           # Physics canonicalization
src/potatobacon/core/                # Physics units/dimensions
src/potatobacon/units/               # Physics unit algebra
src/potatobacon/parser/dsl_parser.py # Physics DSL parsing
src/potatobacon/parser/latex_parser.py
src/potatobacon/parser/transformers.py
src/potatobacon/dashboard/           # Old physics dashboard
src/potatobacon/models.py            # Physics equation models
src/potatobacon/api/routes_units.py  # Physics unit API
```

## What Stays

```
src/potatobacon/tariff/              # Core tariff engine (ALL files)
src/potatobacon/law/                 # Legal analysis + Z3 solver
src/potatobacon/cale/                # Rule parsing + conflict checker
src/potatobacon/proofs/              # Cryptographic proof engine
src/potatobacon/api/routes_tariff*.py  # All 12 tariff API routes
src/potatobacon/api/routes_proofs.py
src/potatobacon/api/routes_law_contexts.py
src/potatobacon/extract/             # Obligation/rule extraction
src/potatobacon/text/                # Document text processing
src/potatobacon/sdk/                 # API client SDK
src/potatobacon/cli/                 # CLI tools
src/potatobacon/persistence.py       # Asset/job persistence
src/potatobacon/observability.py     # Logging/tracing
src/potatobacon/storage.py           # Manifest storage
```
