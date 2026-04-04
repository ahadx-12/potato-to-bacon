# Master Plan: From Tariff Calculator to Tariff Engineer

## The Problem We're Solving

Companies importing goods into the US face a complex, layered tariff system:
- **Base duty rates** (0-50%+) determined by HTS classification
- **Section 232** surcharges (25% on steel, 10-25% on aluminum)
- **Section 301** tariffs (7.5-100% on Chinese-origin goods, escalating in 2025-2026)
- **AD/CVD duties** (antidumping/countervailing, sometimes 200%+)
- **FTA preferences** (USMCA, KORUS, etc. — can reduce or eliminate base duty)
- **Exclusions** (product-specific relief from 232/301)

A typical importer with 500 SKUs is overpaying by 15-40% because they:
1. Don't know all the tariff layers hitting each product
2. Don't realize some products are misclassified (lower-rate heading exists)
3. Don't claim FTA preferences they're entitled to
4. Don't know about product engineering moves that shift classification
5. Don't optimize their supply chain routing for tariff efficiency

**A tariff engineer solves all five.** Not by listing rates — by producing an optimized action plan.

---

## Where We Are Today

### What Works
| Component | Status | Notes |
|-----------|--------|-------|
| Z3 classification solver | ✅ Solid | Formal verification of HTS classification |
| Mutation engine | ✅ Solid | Discovers fact patches that reduce duty |
| Engineering opportunity model | ✅ Solid | Full metadata per finding |
| Strategy synthesizer | ✅ Solid | Groups opportunities into named strategies |
| Scenario comparator | ✅ Solid | Side-by-side what-if tables |
| BOM engineering report | ✅ Solid | Portfolio-level summary with executive numbers |
| FTA engine | ✅ Solid | Evaluates FTA eligibility per program |
| GRI engine | ✅ Solid | General Rules of Interpretation |
| AD/CVD registry + scope | ✅ Solid | Order matching and scope analysis |
| HTS text search | ✅ Solid | Keyword-based heading search |
| Lever library | ✅ Solid | Deterministic optimization pathways with constraints |
| Origin rules | ✅ Solid | Substantial transformation analysis |
| Company profile | ✅ Solid | Supply chain constraints, existing FTAs |
| Risk scorer | ✅ Solid | Portfolio-level compliance risk |
| PDF/Excel export | ✅ Solid | Client-deliverable reports |
| Proof chain | ✅ Solid | Cryptographic verification of every recommendation |

### What's Missing: The Optimization Layer

The system finds opportunities. It does NOT optimize across them. This is the difference between a tariff **analyst** and a tariff **engineer**.

Current behavior:
```
Input: 200 SKUs
Output: 47 opportunities totaling $2.1M potential savings
         (listed individually, unranked beyond dollar value)
```

Target behavior:
```
Input: 200 SKUs + company constraints (budget: $150K, timeline: 6 months, 
       can't change supplier for SKU-001 through SKU-050, MX assembly plant 
       has capacity for 30 additional SKUs)

Output: Optimal implementation plan:
  Phase 1 (Month 1-2, cost: $12K): Documentation quick wins
    → 8 SKUs, $340K annual savings, zero risk
  Phase 2 (Month 2-4, cost: $45K): USMCA preference capture  
    → 22 SKUs shifted to MX assembly, $890K annual savings
  Phase 3 (Month 4-6, cost: $80K): Reclassification program
    → 6 SKUs, $410K annual savings, requires binding rulings
  
  Total: $137K investment → $1.64M annual savings
  ROI: 12x first year
  Strategies deferred (budget/timeline): 11 opportunities worth $460K
```

---

## The Master Plan: 6 Phases

### Phase 1: Constraint-Based Portfolio Optimizer (THE CORE)

**Goal:** Given a set of opportunities and company constraints, find the optimal subset and sequencing that maximizes net savings.

**This is the single most important feature.** Everything else in the system exists to feed this optimizer.

#### 1.1 Opportunity Dependency Graph

Opportunities interact. You can't do both "reclassify SKU-100 under heading 8544" AND "shift SKU-100 origin to MX for USMCA" independently — the reclassification might change whether USMCA applies. The system must model these dependencies.

**Data model:**
```python
class OpportunityDependency:
    """Relationship between two opportunities."""
    opportunity_a_id: str
    opportunity_b_id: str
    relationship: DependencyType  # MUTEX, ENABLES, ENHANCES, CONFLICTS
    explanation: str

class DependencyType(str, Enum):
    MUTEX = "mutex"          # Can only pick one (same SKU, different strategies)
    ENABLES = "enables"      # A must be done before B is possible
    ENHANCES = "enhances"    # Doing A increases savings from B
    CONFLICTS = "conflicts"  # Doing A reduces savings from B
```

**Implementation:**
- Build a dependency graph from the opportunity set
- Detect conflicts: two opportunities targeting the same SKU with different HTS codes → MUTEX
- Detect enablers: FTA preference requires origin shift to qualify → ENABLES
- Detect enhancers: reclassification reduces base rate, which also reduces 301 surcharge → ENHANCES

**File:** `src/potatobacon/tariff/opportunity_graph.py`

#### 1.2 Integer Linear Programming (ILP) Optimizer

The core optimization is a variant of the **knapsack problem** with dependencies:
- Each opportunity has a value (annual savings) and a cost (implementation cost + time)
- Constraints: budget limit, timeline limit, supply chain constraints, mutual exclusions
- Objective: maximize net annual savings subject to constraints

**Implementation approach:**
```python
class PortfolioConstraints:
    """Company-provided constraints for optimization."""
    max_implementation_budget: float | None       # USD cap
    max_timeline_months: int | None               # Calendar months
    fixed_origin_skus: set[str]                   # SKUs where origin cannot change
    fixed_hts_skus: set[str]                      # SKUs where classification cannot change
    fixed_product_skus: set[str]                   # SKUs where product cannot be modified
    available_assembly_countries: list[str]         # Countries where company has/can get assembly capacity
    assembly_capacity: dict[str, int]              # Country → max additional SKUs
    max_risk_grade: str                            # "A", "B", or "C" — max acceptable risk
    require_binding_ruling_first: bool             # If True, exclude opportunities needing CBP ruling
    excluded_opportunity_types: set[OpportunityType]  # Types the company won't consider

class OptimizedPortfolioPlan:
    """The optimizer's output — the actual engineering plan."""
    selected_opportunities: list[TariffEngineeringOpportunity]
    deferred_opportunities: list[TariffEngineeringOpportunity]  # Good but didn't fit constraints
    rejected_opportunities: list[RejectedOpportunity]           # Why each was excluded
    implementation_phases: list[ImplementationPhase]             # Sequenced execution plan
    total_implementation_cost: float
    total_annual_savings: float
    net_first_year_savings: float
    roi_multiple: float
    payback_months: float
    constraint_utilization: ConstraintUtilization  # How much of each constraint was used
```

**Solver options (in order of preference):**
1. **scipy.optimize.milp** — already in our dependency tree via numpy/scipy, supports mixed-integer linear programming
2. **Google OR-Tools** — more powerful, handles complex constraints natively
3. **Z3 Optimize** — we already have Z3, its optimization mode handles this perfectly

**Recommendation: Use Z3 Optimize** since Z3 is already a core dependency. Z3's optimization mode (`z3.Optimize()`) supports:
- Integer variables (select/don't-select each opportunity)
- Linear constraints (budget, timeline, capacity)
- Mutual exclusion constraints (at-most-one from a MUTEX group)
- Precedence constraints (if B selected, A must also be selected)
- Maximize objective (total net savings)

**File:** `src/potatobacon/tariff/portfolio_ilp_optimizer.py`

#### 1.3 Implementation Sequencing

Once the optimizer selects the best set of opportunities, they must be sequenced into phases:

**Sequencing rules:**
1. Documentation-only opportunities go first (zero risk, immediate payback)
2. FTA preference capture next (low risk, high value, mainly paperwork)
3. Reclassification with binding ruling (medium timeline, requires CBP interaction)
4. Product engineering changes (longer timeline, requires supplier coordination)
5. Supply chain shifts (longest timeline, highest cost, highest savings)

**Within each phase:**
- Respect dependency ordering (ENABLES relationships)
- Parallelize independent opportunities
- Group by shared implementation (all USMCA filings together, all lab tests together)

**File:** `src/potatobacon/tariff/implementation_sequencer.py`

#### 1.4 Sensitivity Analysis

For each selected strategy, compute:
- **Break-even volume**: at what annual volume does this opportunity become worthwhile?
- **Rate sensitivity**: if tariff rates change by ±5pp, does this still make sense?
- **FX sensitivity**: for origin shifts, how does currency fluctuation affect total landed cost?
- **Regulatory risk**: probability that an exclusion expires, a new AD/CVD order is issued, etc.

**File:** `src/potatobacon/tariff/sensitivity_analyzer.py`

---

### Phase 2: BOM-Level Optimization (Component Engineering)

**Goal:** Go deeper than SKU-level. Analyze the Bill of Materials at the component level to find optimization opportunities within the product structure.

#### 2.1 BOM Tree Parser

A BOM is hierarchical: a finished product is made of subassemblies, which are made of components, which are made of raw materials. Tariff classification depends on:
- The finished product's essential character (GRI 3(b))
- Which country performs "substantial transformation"
- Component origin mix (for FTA rules of origin / regional value content)

**Data model:**
```python
class BOMNode:
    """A node in the BOM tree."""
    part_number: str
    description: str
    hts_code: str | None
    origin_country: str
    unit_cost: float
    quantity_per_parent: float
    material_type: str | None
    children: list[BOMNode]  # Subcomponents

class BOMTree:
    """Full BOM tree for a finished product."""
    finished_product: BOMNode
    total_bom_cost: float
    origin_value_breakdown: dict[str, float]  # Country → % of total value
    material_weight_breakdown: dict[str, float]  # Material → % of total weight
```

**File:** `src/potatobacon/tariff/bom_tree.py`

#### 2.2 Component Substitution Engine

For each component in the BOM:
1. What if this component came from a different origin? (affects FTA qualification)
2. What if this component used a different material? (affects HTS classification)
3. What if this component was sourced domestically? (eliminates its duty burden)

This generates **component-level mutations** that feed into the portfolio optimizer.

**Key optimization:** Regional Value Content (RVC) for FTA qualification is calculated from the BOM. If a product is at 60% regional content but needs 75% for USMCA, the engine should identify which components to re-source to cross the threshold at minimum cost.

**File:** `src/potatobacon/tariff/component_optimizer.py`

#### 2.3 Assembly Location Optimizer

Where final assembly happens determines:
- Country of origin (for tariff purposes)
- Whether substantial transformation occurs (for FTA qualification)
- Which overlay tariffs apply (301 is origin-specific)

Given a BOM and a set of possible assembly locations, find the optimal assembly location that minimizes total landed cost (duty + logistics + labor).

**File:** `src/potatobacon/tariff/assembly_optimizer.py`

---

### Phase 3: Iterative Simulation Engine

**Goal:** Enable "what-if" analysis where changing one parameter re-runs the entire portfolio and shows cascading effects.

#### 3.1 Simulation Runner

```python
class SimulationScenario:
    """A what-if scenario to evaluate."""
    scenario_name: str
    mutations: dict[str, Any]  # Parameter changes
    # Examples:
    #   {"tariff_rate_301_cn": 60}  — "what if 301 goes to 60%?"
    #   {"origin_shift": {"SKU-100": "MX"}}  — "what if we move SKU-100 to Mexico?"
    #   {"new_exclusion": {"hts": "8544.42", "origin": "CN"}}  — "what if this exclusion is granted?"

class SimulationResult:
    """Full portfolio re-analysis under a scenario."""
    scenario: SimulationScenario
    portfolio_summary: PortfolioSummary
    changed_skus: list[SKUDelta]  # Only SKUs whose analysis changed
    new_opportunities: list[TariffEngineeringOpportunity]
    lost_opportunities: list[TariffEngineeringOpportunity]
    net_savings_delta: float
```

**Use cases:**
- "What happens to our portfolio if Section 301 tariffs on China go to 60%?"
- "What if we move all Chinese-origin electronics assembly to Vietnam?"
- "What if the steel 232 exclusion we applied for gets granted?"
- "What if USMCA is renegotiated and auto parts RVC goes to 85%?"

**File:** `src/potatobacon/tariff/simulation_engine.py`

#### 3.2 Tariff Landscape Monitor

Track changes in the tariff landscape and automatically re-evaluate portfolios:
- New AD/CVD orders (Federal Register monitoring)
- Section 301 list changes
- Exclusion grants/expirations
- FTA renegotiations
- HTS schedule updates (annual)

**File:** `src/potatobacon/tariff/landscape_monitor.py`

---

### Phase 4: Supply Chain Network Optimization

**Goal:** Model the full supply chain network and optimize for total landed cost including tariffs.

#### 4.1 Network Model

```python
class SupplyChainNode:
    """A node in the supply chain network."""
    node_id: str
    node_type: NodeType  # SUPPLIER, MANUFACTURER, ASSEMBLER, WAREHOUSE, PORT
    country: str
    capabilities: list[str]  # What operations this node can perform
    capacity: int            # Units per month
    unit_cost: float         # Per-unit processing cost
    lead_time_days: int

class SupplyChainEdge:
    """A logistics link between nodes."""
    from_node: str
    to_node: str
    transport_cost_per_unit: float
    transit_time_days: int
    transport_mode: str  # SEA, AIR, TRUCK, RAIL

class SupplyChainNetwork:
    """Full supply chain network for optimization."""
    nodes: list[SupplyChainNode]
    edges: list[SupplyChainEdge]
    products: list[BOMTree]
    demand: dict[str, int]  # Product → annual demand
```

#### 4.2 Total Landed Cost Optimizer

Minimize: `material_cost + manufacturing_cost + logistics_cost + duty_cost`

Subject to:
- Capacity constraints at each node
- Lead time requirements
- Quality/certification requirements
- Tariff rules (origin determination, FTA qualification, substantial transformation)
- Company constraints (approved suppliers, existing contracts)

This is a **network flow optimization** problem. Use Z3 or OR-Tools to solve.

**File:** `src/potatobacon/tariff/landed_cost_optimizer.py`

---

### Phase 5: Intelligence & Learning

**Goal:** Make the engine smarter over time.

#### 5.1 Outcome Tracking

Track which recommendations were implemented and their actual savings:
```python
class ImplementationOutcome:
    opportunity_id: str
    implemented: bool
    actual_savings: float | None
    actual_cost: float | None
    actual_timeline_days: int | None
    complications: list[str]
    cbp_ruling_outcome: str | None  # For reclassification
```

Use outcomes to:
- Calibrate savings estimates (are we over/under-estimating?)
- Calibrate implementation costs
- Identify which opportunity types have highest success rates
- Build a knowledge base of what works

**File:** `src/potatobacon/tariff/outcome_tracker.py`

#### 5.2 Pattern Library

Build a searchable library of proven optimization patterns:
```
Pattern: "Textile-dominant composite → felt overlay reclassification"
  Applicable when: Product has mixed materials, textile >50% by weight
  Legal basis: GRI 3(b), CBP Ruling NY N123456
  Success rate: 87% (based on 23 implementations)
  Typical savings: 12-18 percentage points
  Typical timeline: 4-8 weeks
  Required evidence: Lab test showing material composition by weight
```

**File:** `src/potatobacon/tariff/pattern_library.py`

#### 5.3 Tariff Rate Forecasting

Use historical data to forecast tariff rate changes:
- Section 301 escalation schedule (published, deterministic)
- AD/CVD sunset review outcomes (probabilistic)
- Exclusion expiration/renewal likelihood
- FTA renegotiation impact

Factor forecasted rates into the optimization to make forward-looking recommendations.

**File:** `src/potatobacon/tariff/rate_forecaster.py`

---

### Phase 6: Production Hardening

#### 6.1 Performance Optimization
- Parallelize Z3 solver across SKUs (current global lock is the bottleneck)
- Cache classification results for identical product profiles
- Pre-compute opportunity sets for common product categories
- Incremental re-analysis when a single SKU changes

#### 6.2 Data Pipeline
- Automated USITC HTS schedule ingestion (quarterly)
- Federal Register AD/CVD order monitoring (daily)
- Section 301 list change tracking
- Exclusion grant/expiration tracking

#### 6.3 Confidence & Quality Gates
- Per-recommendation confidence score based on data completeness
- Portfolio-level quality score
- Automatic flagging of recommendations that need human review
- Audit trail for every recommendation (already have proof chain)

#### 6.4 Multi-Country Support
- Extend beyond US imports
- EU tariff schedule (TARIC)
- UK Global Tariff
- Canadian Customs Tariff
- Support cross-border optimization (e.g., import to US via Canada under USMCA)

---

## Implementation Priority & Sequencing

```
Phase 1 (IMMEDIATE — this is the optimization core)
├── 1.1 Opportunity dependency graph          ← Week 1
├── 1.2 ILP portfolio optimizer (Z3 Optimize) ← Week 1-2
├── 1.3 Implementation sequencer              ← Week 2
└── 1.4 Sensitivity analysis                  ← Week 2-3

Phase 2 (SHORT TERM — component-level depth)
├── 2.1 BOM tree parser                       ← Week 3
├── 2.2 Component substitution engine         ← Week 3-4
└── 2.3 Assembly location optimizer           ← Week 4

Phase 3 (MEDIUM TERM — simulation and monitoring)
├── 3.1 Simulation runner                     ← Week 5
└── 3.2 Tariff landscape monitor              ← Week 6

Phase 4 (LONGER TERM — full supply chain)
├── 4.1 Network model                         ← Week 7-8
└── 4.2 Total landed cost optimizer           ← Week 8-9

Phase 5 (ONGOING — intelligence)
├── 5.1 Outcome tracking                      ← Week 9
├── 5.2 Pattern library                       ← Week 10
└── 5.3 Rate forecasting                      ← Week 10-11

Phase 6 (ONGOING — production)
├── 6.1 Performance optimization              ← Continuous
├── 6.2 Data pipeline                         ← Continuous
├── 6.3 Quality gates                         ← Continuous
└── 6.4 Multi-country                         ← Future
```

---

## Key Design Decisions

### 1. Z3 Optimize as the Portfolio Solver

We already depend on Z3 for classification verification. Z3's `Optimize` context supports:
- Boolean decision variables (include opportunity or not)
- Linear arithmetic constraints (budget ≤ $X, timeline ≤ Y months)
- Soft constraints with weights (prefer high-confidence opportunities)
- Pareto optimization (multi-objective: maximize savings AND minimize risk)

This avoids adding a new dependency and keeps the entire optimization stack in one solver.

### 2. Opportunity as the Atomic Unit

Everything flows through `TariffEngineeringOpportunity`. The dependency graph, optimizer, sequencer, and simulation engine all operate on opportunities. This keeps the architecture clean — the discovery engines (mutation, FTA, AD/CVD, etc.) produce opportunities, the optimization layer consumes them.

### 3. Constraints Drive the Engineering

The difference between "here are opportunities" and "here is your plan" is constraints. Without constraints, you get a wish list. With constraints, you get an engineering plan. The `PortfolioConstraints` model is as important as the optimizer itself.

### 4. Proofs All the Way Down

Every optimization recommendation is backed by a cryptographic proof. This isn't just nice-to-have — it's essential for:
- Legal defensibility (we recommended X because of formal verification Y)
- Audit trail (CBP can verify our classification logic)
- Reproducibility (re-run with same inputs, get same outputs)

### 5. Separation of Discovery and Optimization

Discovery engines (mutation_engine, fta_engine, adcvd_registry, etc.) find individual opportunities. The optimization layer (portfolio_ilp_optimizer, implementation_sequencer) selects and sequences them. This separation means we can improve discovery without touching optimization, and vice versa.

---

## Success Metrics

1. **Coverage**: % of HTS chapters the engine can analyze (target: 95%+)
2. **Accuracy**: % of recommendations that survive professional customs broker review (target: 90%+)
3. **Optimization gap**: savings found by optimizer vs. simple "sort by value" (target: 15%+ improvement)
4. **Constraint satisfaction**: % of generated plans that respect all stated constraints (target: 100%)
5. **Time to report**: minutes from BOM upload to complete optimization plan (target: <5 min for 500 SKUs)
6. **Recommendation specificity**: % of recommendations with specific HTS codes, legal citations, and action items (target: 100%)

---

## What This Is NOT

- NOT a compliance tool (we don't file entries or interact with CBP)
- NOT a trade finance tool (no duty drawback calculation, no bonding)
- NOT a logistics optimizer (we optimize for tariff cost, not shipping speed)
- NOT a product designer (we suggest engineering changes, we don't design products)
- NOT legal advice (every high-complexity recommendation says "requires professional review")

We are the **analytical brain** that tells importers where their money is going and how to legally keep more of it. The optimization layer is what makes this an engineer, not a calculator.
