# What Needs to Be Done Next

## Our Goal

We are building a **general tariff engineer**, not a tariff calculator.

The distinction is fundamental:

- A **tariff calculator** accepts a product description or HTS code, looks up the rate, and returns a number. Passive. Static.
- A **tariff engineer** takes a company's complete product portfolio (their BOMs), identifies every legal lever available to reduce their duty burden, surfaces those as ranked, actionable savings opportunities, and tells them exactly what to do to capture each one.

**Our system must, for any product a company imports, answer:**
1. What is your real total duty exposure? (base rate + Section 232 + Section 301 + AD/CVD, net of FTA preferences and active exclusions)
2. Where are the legal savings opportunities?
3. What type of engineering is required to capture each one? (documentation, product change, supply chain shift, etc.)
4. How much is each opportunity worth annually?
5. What evidence does the company need to provide?
6. What is the legal basis for each recommendation?

A real tariff engineer produces this as a portfolio-level report — not a dossier of rates.

---

## What Was Done in This Sprint

### Core Architecture Established

1. **`engineering_opportunity.py`** — The fundamental output unit.
   Every finding the engine produces is a `TariffEngineeringOpportunity`, which includes:
   - Opportunity type (RECLASSIFICATION, PRODUCT_ENGINEERING, TRADE_LANE, AD_CVD_EXPOSURE, FTA_UTILIZATION, EXCLUSION_FILING, DOCUMENTATION)
   - Baseline and optimized duty rates
   - Annual savings estimate (if value/volume known)
   - Specific action items
   - Evidence requirements
   - Legal citations
   - Risk grade and confidence level

2. **`bom_engineering_report.py`** — The portfolio-level deliverable.
   `BOMEngineeringReport` aggregates per-SKU findings into:
   - Executive summary: total annual duty exposure, achievable savings, opportunity counts by type
   - All opportunities ranked by annual savings (risk findings separate)
   - Quick wins: documentation-only, available immediately
   - Per-SKU detailed duty breakdown with full layer disclosure

3. **`routes_engineering.py`** — The primary API interface.
   - `POST /v1/engineering/analyze-sku`: spot-check a single product
   - `POST /v1/engineering/analyze-bom`: upload a full BOM (CSV/JSON/XLSX), receive a complete engineering report

4. **`category_taxonomy.py`** expanded from 5 to 17 categories covering all major HTS sections:
   - Electronics (ch. 84-85, 90)
   - Machinery (ch. 84-85)
   - Automotive (ch. 87)
   - Steel/metals (ch. 72-73, 74-83)
   - Aluminum (ch. 76)
   - Apparel (ch. 61-62), Footwear (ch. 64), Textiles (ch. 50-63)
   - Plastics (ch. 39), Rubber (ch. 40)
   - Furniture (ch. 94)
   - Medical/Optical (ch. 90)
   - Chemicals/Pharma (ch. 28-38)
   - Agricultural/Food (ch. 1-24)
   - Wood/Paper (ch. 44-49)
   - Consumer Goods (ch. 93, 95-96)

   Added `chapters_for_description(description) -> List[int]` — routes any free-text description to candidate HTS chapters.

---

## What Needs to Be Done Next

These are ordered by impact, not by difficulty.

---

### 1. HTS Schedule Completeness (CRITICAL)

**The problem:** The default tariff context (`HTS_US_2025_SLICE`) only covers 3 chapters: 64 (footwear), 73 (steel articles), 85 (electronics). Any product outside these chapters will receive no classification and no mutations.

**What's needed:**
- Load all 98 HTS chapters into a default context, or at minimum the top 20 by import value
- Available data files: `data/hts_extract/full_chapters/` contains ch39, ch84, ch87, ch90, ch94 — wire these into a `HTS_US_2025_FULL` context
- For production: integrate the USITC HTSA live API to pull current HTS data on demand (`data/live/` directory already exists)
- The `context_registry.py` and `context_loader.py` are ready — they just need more chapters loaded

**Files to change:**
- `src/potatobacon/tariff/context_registry.py` — register HTS_US_2025_FULL context
- `data/contexts/` — create or extend the full context manifest
- `scripts/` — add a data pipeline script to ingest all chapters from USITC

---

### 2. HTS Text Search and Classification Engine (CRITICAL FOR GENERAL USE)

**The problem:** When a product doesn't match any chapter by keyword, and no HTS hint is given, the engine falls back to the full atom set and produces nothing useful. A real tariff engineer would search the HTS schedule text to find candidate headings.

**What's needed:** An HTS text search module that:
1. Takes a product description
2. Searches HTS heading descriptions (the legal text in the schedule)
3. Returns candidate headings with similarity scores
4. Passes those candidates to the Z3 engine as hints

**Implementation approach:**
- Store HTS heading descriptions in a searchable index (sqlite full-text search or simple TF-IDF)
- `POST /v1/engineering/classify` — takes a description, returns top-5 HTS heading candidates with rationale
- The `analyze-sku` and `analyze-bom` endpoints call this automatically when no hts_hint is provided

**Files to create:**
- `src/potatobacon/tariff/hts_search.py`
- `data/hts_search_index/` — pre-built index from all chapter JSONL files

---

### 3. GRI Engine (General Rules of Interpretation)

**The problem:** HTS classification is determined by 6 legally binding rules (GRI 1-6). Without modeling these rules, the engine cannot classify composite products, multi-material products, sets, or unfinished articles correctly.

**What's needed:** A `GRIEngine` that applies:
- **GRI 1**: Classification by heading text and chapter/section notes (already partially done via atoms/guards)
- **GRI 2(a)**: Unfinished/incomplete articles classified as the finished article
- **GRI 2(b)**: Mixtures/composites — route to GRI 3
- **GRI 3(a)**: Most specific heading wins
- **GRI 3(b)**: Essential character (primary material/component)
- **GRI 3(c)**: Last in numerical order among equally specific
- **GRI 6**: Subheading classification uses same rules

**Why it matters:** A steel bolt partially coated in nylon is NOT just "a steel bolt." GRI determines whether it's steel articles or plastics. The engine cannot produce defensible reclassification opportunities without this.

**Files to create:**
- `src/potatobacon/tariff/gri_engine.py`

---

### 4. Substantial Transformation Analysis for FTA Origin

**The problem:** FTA utilization (USMCA, KORUS, etc.) requires the product to satisfy "rules of origin" — usually either a tariff shift test (the inputs come from a different chapter than the finished good) or a regional value content test. The current FTA engine checks country of origin but does not verify origin qualification.

**What's needed:**
- `src/potatobacon/tariff/origin_rules.py` — tariff shift rules per FTA and HTS heading
- Takes: BOM line items with their origin countries and HTS codes
- Returns: whether the finished product qualifies for FTA preference, and what changes would make it qualify

**Why it matters:** Telling an importer "you could use USMCA" without checking origin qualification is noise. We need to say: "your product currently fails the tariff shift test because component X (HTS 7318) is sourced from CN, but if you sourced it from MX instead, the finished product would qualify for USMCA and you'd save $X."

---

### 5. AD/CVD Scope Analysis

**The problem:** The AD/CVD registry matches orders by HTS prefix and keyword. But AD/CVD scope is legally complex — a product might share an HTS code with covered merchandise but not be within scope (or vice versa). The current confidence levels (high/medium/low) are not sufficient for a production system.

**What's needed:**
- Detailed scope text for each AD/CVD order (from Commerce Department scope determinations)
- A scope matching engine that compares product characteristics against scope language
- Clear distinction between "HTS prefix match" and "confirmed within scope"
- Connection to the Commerce Department's scope ruling database

**Files to change:**
- `data/overlays/adcvd_orders_full.json` — add scope_text field with full legal scope language
- `src/potatobacon/tariff/adcvd_registry.py` — add scope_analysis() method
- `src/potatobacon/tariff/adcvd_scope.py` — new scope matching engine

---

### 6. Binding Ruling Integration

**The problem:** Every reclassification and product engineering recommendation should be backed by CBP binding rulings that confirm the same fact pattern was accepted. Without this, recommendations are legally untested.

**What's needed:**
- A database of CBP binding rulings (HQ and NY series) — publicly available from CBP's CROSS database
- A ruling search module that finds relevant rulings for a product + HTS code pair
- Surface matching rulings in the `legal_basis` field of each opportunity

**Files to create:**
- `src/potatobacon/tariff/ruling_search.py`
- `data/rulings/` — indexed ruling excerpts (start with the most-cited ~2,000 rulings)

---

### 7. Portfolio Risk Scoring

**The problem:** Beyond savings, a tariff engineer identifies compliance risks. Companies can be audited and assessed back duties plus interest plus penalties (up to 4x the unpaid duty). We need to surface not just savings opportunities but risk exposures.

**What's needed:**
- A risk scoring model that looks across the portfolio and flags:
  - High-confidence AD/CVD exposure that isn't being paid
  - Products with Section 232/301 exposure that aren't in the declared HTS codes
  - Incorrect origin claims (products from China declared as ROW)
  - Products likely to trigger CBP audit based on value/duty rate profile

**Files to create:**
- `src/potatobacon/tariff/risk_scorer.py`
- Add `risk_findings` section to `BOMEngineeringReport`

---

### 8. Company Context / Intake System

**The problem:** To produce a complete engineering report, the system needs company-level context that doesn't come from the BOM alone:
- Which FTAs does the company actively use?
- What countries do they source from?
- Do they have existing binding rulings?
- What is their audit history?
- What are their supply chain constraints (cannot change origin, cannot change materials, etc.)?

**What's needed:**
- A company intake form / onboarding flow that captures this context
- A `CompanyProfile` model that shapes the engineering analysis
  - Suppresses trade lane opportunities if origin is fixed
  - Suppresses product engineering if product is certified/validated
  - Flags FTAs the company is already using vs. not using

**Files to create:**
- `src/potatobacon/tariff/company_profile.py`
- `POST /v1/engineering/company-profile` endpoint

---

### 9. Production HTS Rate Store

**The problem:** The current rate store covers a small subset of HTS codes. For a general tariff engineer, every HTS code must return an accurate base rate.

**What's needed:**
- Full USITC HTSA data ingested into the rate store (currently ~12,000 tariff lines)
- Rate store should be updated quarterly (HTS schedule changes each January)
- Special column rates (FTA preferential rates) for all partner countries

**Files to change:**
- `src/potatobacon/tariff/rate_store.py` — production-grade implementation
- `scripts/ingest_usitc.py` — data pipeline to ingest from USITC API

---

### 10. Output Formats: PDF Report and Excel Workbook

**The problem:** Tariff engineering engagements are delivered as professional reports, not API JSON responses. The output needs to be presentable to a company's CFO and legal team.

**What's needed:**
- PDF generation for `BOMEngineeringReport` — executive summary on page 1, opportunity detail in appendix
- Excel workbook with per-SKU duty breakdown, opportunity rankings, and implementation timeline
- Both should be branded and formatted for client delivery

**Files to create:**
- `src/potatobacon/reporting/pdf_generator.py`
- `src/potatobacon/reporting/excel_generator.py`
- `POST /v1/engineering/export/pdf`
- `POST /v1/engineering/export/xlsx`

---

## Immediate Next Steps (Do These First)

1. **Expand HTS context coverage** — Load ch39, ch84, ch87, ch90, ch94 into a `HTS_US_2025_FULL` context that `analyze-bom` uses by default. This makes the engine work for 5x more product types immediately.

2. **Write integration tests for `analyze-sku` and `analyze-bom`** — Use real product examples: a steel pipe from China (should trigger 232 + potential AD/CVD), a cotton shirt from Mexico (should surface USMCA preference), a USB cable (should classify in ch. 85 and find 301 exposure).

3. **Build the HTS text search index** — From the existing JSONL chapter files, build a simple keyword index. This is the fastest path to handling arbitrary products.

4. **Add the first golden-dataset test suite** — 20 real products, known correct HTS codes, known duty rates, known savings opportunities. Every code change must pass this suite.

5. **Implement `POST /v1/engineering/company-profile`** — Without knowing supply chain constraints, every recommendation must be qualified with "if feasible." With a company profile, recommendations become specific.

---

## Architecture Principle to Keep in Mind

Every opportunity must be:
- **Legal**: supported by GRI rules, chapter notes, CBP rulings, or treaty text
- **Specific**: says exactly what change to make, not "consider reclassification"
- **Evidenced**: lists the exact documents needed to support it
- **Valued**: quantifies the annual savings if volume and value are known
- **Risk-graded**: A (minimal), B (moderate), C (professional review required)

We do not recommend shortcuts, evasion, or anything that requires misrepresentation to CBP.
Every recommendation assumes the company will accurately describe their product to customs.
