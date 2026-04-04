# CLAUDE.md — Project Instructions for AI Agents

## What This Project Is

**CaleTariff / PotatoBacon** is a **Tariff Engineering-as-a-Service (TEaaS)** platform.

It is NOT a tariff calculator. It is a tariff **engineer**.

The distinction matters:
- A **tariff calculator** takes an HTS code, looks up a rate, returns a number. Passive, static, useless beyond lookup.
- A **tariff engineer** takes a company's entire Bill of Materials (BOM), analyzes every SKU against the full US tariff schedule, discovers every legal lever to reduce duty burden, and produces a ranked, actionable optimization plan with dollar savings, implementation steps, evidence requirements, legal citations, and risk grades.

## The Goal

Build a system where a company can upload their BOM (CSV/XLSX/JSON of SKUs with descriptions, HTS codes, origin countries, declared values, annual volumes) and receive back:

1. **Total duty exposure** — base rate + Section 232 + Section 301 + AD/CVD, net of FTA preferences and exclusions, for every SKU
2. **Optimization opportunities** — ranked by annual savings, each with:
   - What type of engineering is needed (documentation, product change, supply chain shift, reclassification)
   - Specific action items to capture the savings
   - Evidence/documents the company must provide
   - Legal basis (GRI rules, CBP rulings, treaty text)
   - Risk grade (A/B/C) and confidence level
   - Implementation cost and timeline
3. **Portfolio-level strategies** — cross-SKU strategies that group related opportunities (e.g., "shift 14 SKUs to USMCA" as one strategy, not 14 separate findings)
4. **Scenario comparison** — side-by-side what-if analysis showing current state vs. each optimization path
5. **Constraint-aware optimization** — respect company constraints (can't change origin, can't modify product, budget limits) and find the optimal set of strategies given those constraints

## Tech Stack

- **Language**: Python 3.11+
- **Core solver**: Z3 (SMT solver) for formal tariff classification and mutation verification
- **API**: FastAPI
- **Data**: HTS schedule data in JSONL, overlay data (232/301/AD-CVD) in JSON
- **Proofs**: Cryptographic proof chain for every optimization recommendation
- **Database**: PostgreSQL via SQLAlchemy + Alembic
- **Queue**: Celery + Redis for async batch processing
- **Output**: PDF reports (ReportLab), Excel workbooks (openpyxl)

## Key Architecture

```
src/potatobacon/
├── tariff/              # Core tariff engineering engine
│   ├── optimizer.py             # Z3-backed single-SKU optimization
│   ├── strategy_synthesizer.py  # Cross-portfolio strategy grouping
│   ├── scenario_comparator.py   # What-if scenario analysis
│   ├── engineering_opportunity.py # Core opportunity model
│   ├── bom_engineering_report.py # Portfolio-level deliverable
│   ├── mutation_engine.py       # Z3-driven mutation discovery
│   ├── strategy_optimizer.py    # Origin shift + reclassification strategies
│   ├── portfolio_optimizer.py   # Portfolio-level optimization
│   ├── fta_engine.py            # FTA preference evaluation
│   ├── gri_engine.py            # General Rules of Interpretation
│   ├── origin_rules.py          # Rules of origin / substantial transformation
│   ├── adcvd_registry.py        # AD/CVD order matching
│   ├── adcvd_scope.py           # AD/CVD scope analysis
│   ├── hts_search.py            # HTS text search and classification
│   ├── levers.py                # Lever library (deterministic optimization pathways)
│   ├── company_profile.py       # Company context and constraints
│   ├── risk_scorer.py           # Portfolio risk scoring
│   ├── overlays.py              # Section 232/301 overlay engine
│   ├── duty_calculator.py       # Duty rate computation
│   └── ...
├── api/                 # FastAPI routes
├── law/                 # Z3 solver integration
├── proofs/              # Cryptographic proof engine
└── ...
```

## Current State (as of April 2026)

The system has strong foundations but operates more as an **advanced calculator with engineering aspirations** than a true optimization engine. What exists:

**Working well:**
- Z3-backed classification and mutation verification
- Per-SKU duty calculation with overlay support (232/301)
- Mutation engine that discovers fact patches to reduce duty
- Engineering opportunity model with full metadata
- BOM engineering report with portfolio summary
- Strategy synthesizer (groups opportunities into named strategies)
- Scenario comparator (side-by-side what-if)
- FTA engine, GRI engine, origin rules
- AD/CVD registry and scope analysis
- HTS text search
- PDF and Excel export
- API endpoints for single-SKU and BOM analysis
- Cryptographic proof chain

**Critical gaps for true tariff engineering with optimization:**
- No constraint-based portfolio optimizer (knapsack/ILP for "given $X budget, which strategies maximize savings?")
- No dependency modeling between opportunities (can't do both reclassification AND origin shift on same SKU)
- No implementation sequencing optimizer (which strategies to do first given dependencies and payback periods)
- Limited BOM-level optimization (component substitution, assembly location, value engineering)
- No iterative what-if simulation (change one thing, re-run entire portfolio)
- No supply chain network modeling (where to source, where to assemble, given tariff landscape)

## Development Guidelines

1. **Every optimization must be legal** — supported by GRI rules, chapter notes, CBP rulings, or treaty text. Never recommend evasion or misrepresentation.
2. **Every recommendation must be specific** — "Reclassify under 8544.42.20 per GRI 3(b)" not "consider reclassification."
3. **Every recommendation must be evidenced** — list exact documents needed.
4. **Every recommendation must be valued** — dollar savings if value/volume known.
5. **Every recommendation must be risk-graded** — A (minimal), B (moderate), C (professional review required).
6. **Optimization means constraint satisfaction** — not just listing opportunities, but finding the optimal *set* of actions given budget, timeline, and supply chain constraints.
7. **Tests are mandatory** — every new module needs tests. Run `pytest tests/` to verify.
8. **Proofs are mandatory** — every optimization path must produce a verifiable proof.

## Running Tests

```bash
# Run all tests
pytest tests/ -q

# Run tariff-specific tests
pytest tests/tariff/ -q

# Run with coverage
pytest tests/ --cov=potatobacon --cov-report=term-missing
```

## Branch Convention

Development happens on feature branches prefixed with `claude/`. Current optimization work is on `claude/tariff-engineer-optimization-BQLFV`.
