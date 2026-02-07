# CALE-LAW / CALE-TARIFF: Deep Architectural Audit & Strategic Pivot Analysis

**Prepared by:** Senior Technical Partner, Deep Tech Due Diligence
**Date:** 2026-02-07
**Scope:** Full codebase review of `potato-to-bacon` (CALE-LAW + CALE-TARIFF)
**Classification:** Confidential - Investment Committee Only

---

## PHASE 1: THE ARCHITECTURAL AUDIT

### 1.1 Executive Summary

This is a **legitimate neuro-symbolic system** -- not vaporware. The Z3 integration is real, the logic pipeline is structurally sound, and the tariff optimization demo produces verifiable results. However, the system is currently a **"research prototype with production aspirations"** rather than production infrastructure. The gap between those two states is where both the risk and the opportunity live.

**Verdict: Strong Toy approaching Infrastructure.** The bones are right. The math is honest. The engineering needs hardening.

---

### 1.2 The Solver Core: `solver_z3.py` (275 lines)

**File:** `src/potatobacon/law/solver_z3.py`

#### How PolicyAtoms Translate to Z3 Constraints

The translation is straightforward propositional logic. Each `PolicyAtom` carries:
- A **guard**: a list of string literals (e.g., `"token"` or `"~token"`)
- An **outcome**: a modality/action pair (`OBLIGE`, `FORBID`, `PERMIT`)

The compilation pipeline (`compile_atoms_to_z3`, lines 156-167) does this:

```
guard = ["felt_covering_gt_50", "~surface_contact_rubber_gt_50"]
  --> And(Bool("felt_covering_gt_50"), Not(Bool("surface_contact_rubber_gt_50")))

outcome = {modality: "FORBID", action: "import_at_37_5"}
  --> Not(Bool("import_at_37_5"))
```

Each atom becomes an **implication**: `Guard => Outcome`. These implications are asserted into a Z3 `Solver` or `Optimize` instance. A scenario is a mapping `{variable_name: True/False}` that fixes fact variables, and `check_scenario` determines SAT/UNSAT.

#### Honest Assessment

**Is this true symbolic execution?** No. It is **propositional satisfiability checking** over Boolean domains. There are no quantifiers, no arithmetic constraints, no first-order logic, no SMT theories beyond pure Boolean. Every "rule" is reduced to `AND(literals) => literal`.

**Is that bad?** Not necessarily. Propositional SAT is computationally tractable (Z3 is overkill but correct), and the domain (legal rules as Boolean guards) maps cleanly. The architecture is using Z3 as a **correctness oracle**, not for its full power.

**Where it's fragile:**
- `check_scenario` (lines 214-240) creates a **new Solver per guard evaluation** inside a loop. For N atoms, this is O(N) fresh solver instantiations per call. Z3 solver creation isn't free -- this will not scale past ~1000 atoms.
- The thread lock (`_Z3_LOCK`, line 33) serializes all Z3 access. Under concurrent API load, this becomes a bottleneck.
- The atom cache (`_ATOM_CACHE`, line 32) is a global dict with no eviction policy. In a long-running server, this is a memory leak.

**Where it's solid:**
- The `analyze_scenario` function (lines 243-274) uses Z3's `assert_and_track` for UNSAT core extraction. This is the correct Z3 idiom and provides genuine explanatory power for contradictions.
- The `memoized_optimize` pattern (lines 170-187) clones a base `Optimize` instance by replaying assertions. This is a clever way to avoid recompilation.
- Guard/outcome separation is clean. The `PolicyAtom` dataclass is a well-designed intermediate representation.

---

### 1.3 The Optimization Engine: `arbitrage_hunter.py` (471 lines)

**File:** `src/potatobacon/law/arbitrage_hunter.py`

#### How Scenario Mutation Works

The `ArbitrageHunter` is **not a genetic algorithm**. It is a **random bit-flip fuzzer** with metric-guided ranking:

1. **Seed Construction** (`_optimise_seed`, lines 90-102): Extract known facts from constraints, producing a base scenario.
2. **Mutation** (`_mutate`, lines 104-110): Pick `N` random keys from the scenario dict and flip their Boolean values. No crossover, no fitness-proportionate selection, no population dynamics.
3. **Evaluation Loop** (lines 335-349): For each seed, generate `fuzz_budget` mutations (2-6 depending on risk tolerance), compute `ScenarioMetrics` for each, collect all candidates.
4. **Selection** (line 351): Sort candidates by `score` descending, take top 5.

#### Honest Assessment

**Is the "Genetic Algorithm" robust?** There is no genetic algorithm. The mutation is pure random perturbation with no selection pressure feeding back into the mutation operator. It is closer to **random search with greedy selection**. For a small Boolean space (< 20 variables), this is actually fine -- the space is small enough that random sampling covers it adequately. For 100+ variables, this will miss structured optima.

**Where it's fragile:**
- The fuzz budget is hardcoded by risk tolerance string (`"low"` = 2, `"medium"` = 4, `"high"` = 6). These are small numbers. With 20 Boolean variables, the search space is 2^20 = ~1M points. Sampling 5-30 of them is cosmetically random, not systematic.
- The `_provenance_chain` method (lines 123-259) has a **hardcoded set of required IDs** (`{"KY_ES_TEST", "IE_TRADING_VS_PASSIVE", "US_GILTI_2026"}`, line 235). This is a demo artifact leaking into production code.
- The jurisdiction boost (lines 357-360) applies a linear `value_boost = 1.0 + 0.05 * (span - 1)` which has no theoretical justification. It's a tuning knob masquerading as a formula.

**Where it's solid:**
- The separation between `ArbitrageHunter` (search) and `compute_scenario_metrics` (evaluation) is architecturally clean. You can swap search strategies without touching the scoring.
- The `ArbitrageDossier` output structure with provenance chains and dependency graphs is production-quality data modeling.
- The `ProvenanceStep` and `DependencyGraph` models (in `arbitrage_models.py`) are well-designed for audit trails -- this is the kind of output regulators and lawyers actually need.

---

### 1.4 The Metrics System: `cale_metrics.py` (281 lines)

**File:** `src/potatobacon/law/cale_metrics.py`

#### The "Regulatory Alpha" Formula (The Math Check)

The composite score is computed at lines 193-196:

```python
value_term = value_estimate ** alpha
entropy_term = entropy ** beta
risk_term = 1.0 - risk
score = value_term * entropy_term * risk_term
```

**Decomposition:**

| Component | Definition | Range |
|-----------|-----------|-------|
| `value_estimate` | Spread between top-2 outcome probabilities | [0, 1] |
| `entropy` | Normalized Shannon entropy of outcome distribution | [0, 1] |
| `risk` | `max(enforcement, ambiguity, treaty_mismatch) / 3` | [0, 1] |
| `alpha`, `beta` | Tuning exponents (default 1.0) | User-specified |

**Critique:**

1. **The score rewards high entropy AND high value spread simultaneously.** This is mathematically contradictory. High entropy means outcomes are uniformly distributed (no clear winner), which means the spread between top-2 should be small. The only regime where both are high is with 3+ outcomes where one dominates slightly -- a narrow band. In practice, `score` will be dominated by whichever term is smallest, making `alpha` and `beta` nearly irrelevant at defaults of 1.0.

2. **The `value_estimate` heuristic** (lines 140-146) is the gap between first and second probability. This measures **decisiveness**, not economic value. A scenario where Rule A fires with 60% and Rule B with 40% gets `value = 0.2`. A scenario where Rule A fires with 99% gets `value = 0.99`. The metric conflates "certainty of outcome" with "value of arbitrage opportunity." True arbitrage value should measure the **delta in economic consequence** between outcomes, not their probability gap.

3. **The `risk` term** (lines 179-191) is a max-then-average hybrid:
   ```python
   risk = max(risk, sum(risk_components.values()) / len(risk_components))
   ```
   This means risk is always >= the mean of its components, but could be as high as any single component. The aggregation is ad-hoc -- there's no covariance modeling, no tail-risk weighting.

4. **The contradiction probability** (`_local_contradiction_probability`, lines 97-119) is empirically grounded: it actually flips bits and checks SAT/UNSAT. This is the most defensible metric in the system. However, it only checks the first 5 scenario variables (line 112: `[:5]`), which is a silent truncation that could miss contradictions in deeper variables.

5. **The Cohen's Kappa proxy** (lines 207-211) is correctly implemented as `(observed - expected) / (1 - expected)` but is clamped to `[0, 1]`, discarding the informative negative range (where agreement is worse than chance). Negative kappa values would indicate systematic disagreement between rules, which is exactly what you'd want to detect in legal conflict analysis.

6. **Tax liability modeling** (lines 154-173): The `effective_tax_rate_base` formula `0.25 + (1.0 - dominant) * 0.5` is a linear interpolation that maps "certainty of dominant outcome" to a tax rate range of [0.25, 0.75]. This is not grounded in any tax code or treaty -- it's a placeholder heuristic. The `gross_income` default falls back to `max(len(scenario), 1) * 100000.0` which is arbitrary.

**Verdict on Metrics:** The entropy and contradiction metrics are mathematically real and defensible. The "Regulatory Alpha" composite score is a **vanity metric** -- it combines individually meaningful signals into a product that loses interpretability. The value and tax components are placeholders, not production-grade financial models.

---

### 1.5 The Tariff Experiments

#### Converse Felt-Sole Hack (`exp_tariff_converse.py`)

**Claim:** Apply 51% felt covering to a Converse shoe's outer sole to reclassify from 37.5% duty to 3.0% duty (92% savings).

**Assessment:** This is a **real tariff engineering strategy** based on HTS classification rules. The US Harmonized Tariff Schedule does differentiate footwear by sole material composition, and felt-sole reclassification has been used in actual trade practice (see: the "Converse customs ruling" from the early 2000s). The code demonstrates the system can:

1. Define a baseline scenario with Boolean facts (`upper_material_textile`, `outer_sole_material_rubber_or_plastics`, etc.)
2. Apply a mutation (`felt_covering_gt_50: True`)
3. Trigger derived logic (`apply_mutations` in `engine.py:97-104` automatically sets `surface_contact_textile_gt_50 = True` and `surface_contact_rubber_gt_50 = False`)
4. Recompute the duty rate via Z3 constraint evaluation
5. Generate a cryptographic proof of the before/after analysis

The experiment test (`test_exp_tariff_converse.py`) asserts specific duty rates (37.5 baseline, 3.0 optimized), confirming the system produces deterministic, auditable results.

**Limitation:** The "derived logic" (felt => textile > 50%) is hardcoded in `apply_mutations`, not derived from the Z3 model. A true neuro-symbolic system would infer these entailments from the rule graph. Currently it's a hand-coded business rule.

#### Tesla Bolt Experiment (`exp_tariff_tesla_bolt.py`)

This experiment is minimal -- it tests material-based classification mutations (steel vs. aluminum) for a chassis bolt. It's a proof-of-concept, not a showcase.

---

### 1.6 Supporting Architecture

#### The Proof Engine (`proofs/engine.py`)
This is genuinely well-designed. Every tariff computation is:
1. Serialized to a canonical JSON form
2. Hashed with SHA-256 to produce a `proof_id`
3. Separately hashed for a `proof_payload_hash`
4. Persisted to a `ProofStore`

This creates an **immutable audit trail** where any tariff optimization can be independently verified. For regulated industries, this is infrastructure gold.

#### The Overlay System (`tariff/overlays.py`)
Handles Section 232 (steel/aluminum) and Section 301 (China tariffs) as additive rate adjustments. The system correctly:
- Loads overlay rules from JSON data files
- Matches HTS prefixes to determine applicability
- Supports `stop_optimization` flags where overlays make optimization infeasible
- Applies overlays additively to the base duty rate

#### The Origin Engine (`tariff/origin_engine.py`)
Implements USMCA Chapter 4 rules with:
- Tariff shift testing (heading/subheading changes)
- Regional Value Content (RVC) calculation (build-down and build-up methods)
- Substantial transformation determination
- Conflict intensity scoring using CALE's symbolic checker

This is the most mature module in the codebase and closest to production readiness.

#### The CALE Symbolic Checker (`cale/symbolic.py`)
Correctly models OBLIGE/FORBID/PERMIT as mutual constraints:
- `OBLIGE(action)` => `must(action) AND act(action)`
- `FORBID(action)` => `forb(action) AND NOT(act(action))`
- `PERMIT(action)` => `perm(action)`

With domain axioms: `perm == NOT(forb)` and `must => NOT(forb)`. This is a correct deontic logic encoding for a propositional fragment.

---

### 1.7 The Brutal Truth Summary

| Dimension | Grade | Notes |
|-----------|-------|-------|
| **Z3 Integration** | B+ | Real, correct, but using 5% of Z3's power. Boolean-only. |
| **Architecture** | A- | Clean separation of concerns. PolicyAtom -> Z3 -> Metrics -> Dossier pipeline is sound. |
| **Search/Optimization** | C | Random fuzzing, not a real optimizer. Works only because the spaces are tiny. |
| **Metrics/Scoring** | B- | Entropy and contradiction are real. Composite score is a vanity metric. Tax modeling is placeholder. |
| **Proof System** | A | SHA-256 hashed, canonically serialized, persisted. Audit-grade. |
| **Tariff Domain** | A- | Real HTS logic, real overlays, real origin rules. The Converse hack is commercially viable. |
| **Scalability** | D | Global locks, O(N) solver instantiation, no eviction on caches. |
| **Test Coverage** | B+ | 100+ test files, experiment validation, system tests. |
| **Production Readiness** | C+ | FastAPI endpoints exist but the solver won't survive concurrent load. |

**Is this a "Toy" or "Infrastructure"?**

It's a **toy that contains real infrastructure inside it**. The proof engine, the overlay system, the origin analysis, and the PolicyAtom IR are all production-grade components. The solver core needs scale engineering, the optimizer needs a real algorithm, and the metrics need to be split into "mathematically defensible" (entropy, contradiction) vs. "business heuristic" (score, value).

---

## PHASE 2: SIX MONETIZATION PATHWAYS

Based on the architecture observed: **Logic Engine (Z3) + Scenario Mutation + Cryptographic Proofs + Domain-Specific Atoms**, here are six concrete product pivots.

---

### CASH FLOW BUSINESSES (Revenue in 90 days)

#### 1. Tariff Engineering-as-a-Service (TEaaS)

**What it is:** A SaaS platform where importers upload their product BOMs and receive optimized HTS classification with cryptographic proof of the analysis chain.

**Why the architecture supports it:** The existing `run_tariff_hack` pipeline already produces a `TariffDossierModel` with baseline vs. optimized duty rates, provenance chains, and SHA-256 proof hashes. The overlay system handles Section 232/301 complexities. The origin engine computes USMCA RVC.

**Revenue model:** Per-SKU analysis ($500-2000/SKU for one-time classification, $100/month for monitoring as tariff schedules change). A mid-size importer with 500 SKUs = $250K-$1M one-time + $50K/month recurring.

**Competitive edge:** The proof system. No other tariff classification tool produces a cryptographically verifiable audit trail. When CBP audits hit (and they do), having a machine-generated, hash-verified dossier is worth 10x the subscription fee in avoided penalties.

**Build gap:** The HTS ingest pipeline needs real USITC data feeds (currently working from manifests). The `apply_mutations` logic needs to derive entailments from the Z3 model rather than hardcoding them. The API needs rate limiting and tenant isolation. Estimated: 3-4 months to MVP.

---

#### 2. Sanctions & Export Control Screening Engine

**What it is:** Use the PolicyAtom + Z3 pipeline to model EAR/ITAR export control rules and screen transactions against them. When a transaction involves Country X + Technology Y + End-Use Z, the solver determines whether an export license is required, which exceptions might apply, and what the contradiction risk is.

**Why the architecture supports it:** Export control rules have the same logical structure as tariff rules: `IF (conditions) THEN (obligation/prohibition)`. The `SymbolicConflictChecker` can detect when two export control rules give contradictory guidance (e.g., "License Exception TMP permits temporary exports" vs. "Entity List prohibits all exports to Entity X"). The proof engine provides the audit trail required by BIS.

**Revenue model:** Per-transaction screening ($5-50/transaction for high-volume exporters). A defense contractor screening 10,000 transactions/year = $500K/year. Enterprise license: $100K-500K/year.

**Competitive edge:** Current export control tools (Visual Compliance, Descartes) do keyword matching against denied party lists. They don't model the logical structure of the regulations. A Z3-backed system can answer "Is there ANY combination of license exceptions that makes this transaction legal?" -- a question no keyword tool can answer.

**Build gap:** Need to ingest EAR Part 774 (Commerce Control List) and ITAR Part 121 (Munitions List) as PolicyAtoms. The Boolean guard structure maps directly. The contradiction checker needs to handle jurisdiction conflicts (US vs. EU dual-use regulations). Estimated: 4-6 months to MVP.

---

#### 3. Transfer Pricing Audit Defense Generator

**What it is:** Multinational corporations structure intercompany transactions to minimize global tax. When audited, they need to prove their pricing is "arm's length." Use CALE to model transfer pricing rules across jurisdictions, identify contradiction risks between local TP regulations, and generate audit-ready documentation with provenance chains.

**Why the architecture supports it:** The `ArbitrageHunter` was literally designed for multi-jurisdiction tax analysis. The `ScenarioMetrics` already compute per-jurisdiction effective tax rates and treaty mismatch risk. The provenance chain shows which rules from which jurisdictions were considered.

**Revenue model:** Per-entity TP documentation ($50K-200K per entity per year for Big 4 quality documentation). Target: mid-market companies ($500M-$5B revenue) with 10-50 intercompany entities = $500K-$10M per engagement.

**Competitive edge:** Big 4 firms charge $300K+ for TP documentation and it takes 3-6 months. An automated system that generates the initial analysis with provenance and contradiction scoring in hours, with a human review layer on top, could deliver 80% of the value at 20% of the cost.

**Build gap:** The tax rate modeling in `cale_metrics.py` is currently a placeholder heuristic. Need real OECD TP Guidelines and country-specific TP rules ingested as PolicyAtoms. The `gross_income` and `blended_tax_rate` calculations need to use actual financial data. Estimated: 6-9 months to MVP.

---

### MOONSHOT BUSINESSES (Billion-dollar potential)

#### 4. Universal Regulatory Compliance Compiler

**What it is:** A platform that ingests regulatory text (FDA, EPA, SEC, OSHA, any regulator) and compiles it into PolicyAtoms, then provides a "regulatory API" where companies can query: "Given my product/operation with facts {X, Y, Z}, what are my obligations, prohibitions, permissions, and contradictions?"

**Why the architecture supports it:** The full pipeline already exists: `RuleParser` (text -> `LegalRule`) -> `build_policy_atoms_from_rules` (`LegalRule` -> `PolicyAtom`) -> `compile_atoms_to_z3` (`PolicyAtom` -> Z3 constraints) -> `check_scenario` (scenario -> SAT/UNSAT + active rules). The `FeatureEngine` in `cale/embed.py` produces interpretive, situational, temporal, and jurisdictional vectors for each rule.

**The $1B insight:** Every compliance department in every Fortune 500 company is essentially doing manual SAT solving. They have rules, they have facts, and they need to know: "Are we compliant? Where are the conflicts? What do we need to change?" This is the same computation CALE performs, just at a different scale.

**Build gap:** The `RuleParser` currently handles a narrow syntactic pattern (subject + modality + action + conditions). Real regulatory text is far more complex (nested conditionals, cross-references, temporal triggers, numerical thresholds). This is where LLM-to-Logic pipelines become essential (see Phase 3). Need to move from Boolean to SMT (arithmetic, bit-vectors, arrays) for regulations involving numerical thresholds. Estimated: 18-24 months to a general-purpose platform.

---

#### 5. Autonomous Treaty Arbitrage for Digital Assets

**What it is:** A system that continuously monitors global tax treaty networks, identifies arbitrage opportunities for digital asset taxation (staking rewards, DeFi yields, NFT royalties), and generates optimized jurisdictional structures with proof-of-compliance.

**Why the architecture supports it:** The `ArbitrageHunter` was built for exactly this use case. The multi-jurisdiction provenance chain, the contradiction detection between treaty obligations, and the cryptographic proof system are all directly applicable. The `ScenarioMetrics` already model staking reward taxation (see `context.get("gross_staking_rewards")` in `cale_metrics.py:154`).

**The $1B insight:** The global crypto/DeFi space has ~$2T in assets operating across 195 jurisdictions with wildly inconsistent tax treatment. Every DAO, protocol, and institutional crypto fund needs to optimize its jurisdictional structure. The current approach is hiring Big 4 firms at $1M+/year. An automated system that monitors treaty changes, re-runs the arbitrage analysis, and alerts when structures need adjustment is a recurring revenue machine.

**Build gap:** Need real treaty text ingestion (OECD Model Tax Convention, bilateral DTTs). The solver needs to handle temporal logic (treaty provisions with sunset clauses, grandfathering rules). The `value_components` need to use real tax rates and treaty withholding rates, not heuristics. Need regulatory approval/legal opinion coverage for jurisdictional recommendations. Estimated: 12-18 months to MVP for a specific corridor (e.g., US-Ireland-Cayman).

---

#### 6. Formal Verification Layer for AI-Generated Contracts

**What it is:** As LLMs increasingly draft legal contracts, use CALE as a **formal verification backend** that checks contracts for internal contradictions, conflicts with applicable law, and unintended obligations. The LLM generates the contract text; CALE parses it into PolicyAtoms and proves it's logically consistent.

**Why the architecture supports it:** The `SymbolicConflictChecker` already detects when two rules produce contradictory outcomes under shared preconditions. The `analyze_scenario` function extracts UNSAT cores showing exactly which clauses conflict. The proof engine provides a verifiable certificate that the contract was checked.

**The $1B insight:** The legal tech market is racing toward AI-generated contracts, but nobody trusts them yet. The missing piece is formal verification -- a mathematical guarantee that the contract doesn't contradict itself or violate applicable regulations. CALE can be that guarantee layer, sitting between the LLM and the user: "The AI drafted this contract. CALE verified it has 0 internal contradictions and 0 conflicts with [selected regulatory corpus]."

**Build gap:** The parser needs to handle full contract language (representations, warranties, covenants, conditions precedent), not just rule sentences. Need to integrate with LLM APIs (take contract text, generate PolicyAtoms via LLM, verify via Z3). The conflict checker needs to handle quantitative terms (payment amounts, deadlines, interest rates) via SMT arithmetic theories. Estimated: 12-18 months to MVP.

---

## PHASE 3: THE UNIVERSAL LOGIC ENGINE ROADMAP

### 3.1 Ingestion: PDF to PolicyAtom Automatically

The current pipeline has a gap between raw documents and `PolicyAtom`. Here's how to close it:

#### Architecture: LLM-to-Logic Pipeline

```
PDF/HTML Document
    |
    v
[Document Segmenter] -- split into sections, detect tables/lists
    |
    v
[LLM Extraction Layer] -- GPT-4/Claude extracts structured rules
    |                      Output: {subject, modality, action, conditions}
    v
[Validation Layer] -- CALE Parser validates structure
    |                  Z3 checks consistency of extracted rules
    v
[Human Review Queue] -- flag low-confidence extractions
    |
    v
PolicyAtom Registry (versioned, hashed)
```

#### Key Engineering Decisions

1. **LLM prompt structure:** Use few-shot prompting with the existing `LegalRule` schema as the target format. The LLM should output `{text, subject, modality, action, conditions[], jurisdiction, statute, section}`. The `PredicateMapper` already canonicalizes arbitrary condition text to slugs.

2. **Confidence gating:** After LLM extraction, re-parse the output through `RuleParser`. If the parser succeeds, high confidence. If it fails (can't find modality, can't extract action), flag for human review. Use the `FeatureEngine` embedding distance to detect when a new rule is semantically far from existing corpus rules.

3. **Incremental ingestion:** Use the manifest hash system (already in `solver_z3.py`) to detect when new rules are added. Only recompile affected atoms. The `_ATOM_CACHE` keyed on manifest hash already supports this pattern.

4. **Table extraction:** The `cale/finance/tables.py` module already handles financial table extraction. Extend this to tariff schedule tables (HTS data is fundamentally tabular) and regulatory rate tables.

#### What to Build First
- LLM extraction endpoint: takes raw text, returns candidate `LegalRule` objects
- Confidence scorer: uses parser success + embedding distance to rate extraction quality
- Human review UI: display extracted rules, allow corrections, retrain few-shot examples
- Versioned atom registry: track which rules came from which document version

---

### 3.2 Scale: Handling 100,000 Rules Without Z3 Timing Out

The current architecture has three scaling bottlenecks. Here's how to address each:

#### Bottleneck 1: O(N) Solver Instantiation in `check_scenario`

**Current:** For each atom, `check_scenario` creates a fresh `Solver`, copies all assertions, and checks if the guard's negation is SAT. This is O(N) solver constructions.

**Solution: Incremental Solving.** Z3's `Solver` supports `push()`/`pop()` for incremental constraint addition. Instead of creating N solvers:

```python
solver = Solver()
# Add all scenario facts and implications once
for atom in atoms:
    solver.push()
    solver.add(Not(atom.z3_guard))
    if solver.check() == unsat:
        active_atoms.append(atom)  # guard is necessarily true
    solver.pop()
```

This reuses the same solver state, avoiding reconstruction. Expected speedup: 10-50x for large atom sets.

#### Bottleneck 2: Global Thread Lock

**Current:** `_Z3_LOCK` serializes all Z3 access across threads.

**Solution: Solver Pool.** Create a pool of Z3 `Solver` instances, each pre-loaded with the base assertion set (from `memoized_optimize`). Threads check out a solver, use it, and return it. No global lock needed -- each solver is thread-local.

For truly concurrent workloads, consider Z3's `ParOr` tactic or partition the atom space by jurisdiction and solve in parallel.

#### Bottleneck 3: No Rule Partitioning

**Current:** Every scenario check loads ALL atoms and evaluates all guards.

**Solution: Jurisdiction + Subject Partitioning.** Build an index:

```
{
  "US:tariff:footwear": [atom_1, atom_2, ...],
  "EU:tariff:textiles": [atom_3, atom_4, ...],
  "US:tax:income":      [atom_5, atom_6, ...],
}
```

Scenario facts determine which partitions are relevant. Only load and check atoms from relevant partitions. For 100K rules across 50 jurisdictions, this reduces the effective N from 100K to ~2K per query.

#### Bottleneck 4: Z3 Itself

For very large rule sets (100K+), Z3's SAT solver may genuinely struggle. Mitigation strategies:

1. **Pre-filter with BDD (Binary Decision Diagrams):** Use a lightweight BDD library to quickly prune atoms whose guards are trivially false given the scenario. Only pass ambiguous atoms to Z3.
2. **Tiered solving:** Use fast heuristic checks first (simple Python boolean evaluation of guards), escalate to Z3 only for atoms that pass the heuristic filter and need formal verification.
3. **Caching solved scenarios:** Many queries will share fact subsets. Cache partial results keyed on fact subsets to avoid redundant solving.

---

### 3.3 UI/UX: The "God Mode" Dashboard

#### Core Dashboard Views

**View 1: Regulatory Landscape Map**

A force-directed graph where:
- **Nodes** = PolicyAtoms, sized by activation frequency, colored by jurisdiction
- **Edges** = logical dependencies (guard overlap, outcome conflicts)
- **Highlights** = Red glow on nodes with high contradiction probability, Green glow on nodes in the golden scenario

Interactive: Click a node to see its full provenance chain. Drag to rearrange. Filter by jurisdiction, modality, date range.

**View 2: Scenario Explorer**

A spreadsheet-like interface where:
- **Rows** = Boolean fact variables (from all atom guards)
- **Columns** = Scenarios (baseline, optimized, mutations 1-N)
- **Cells** = True/False with color coding (green = contributes to optimization, red = contributes to higher cost)
- **Footer row** = Computed duty rate, entropy, contradiction probability, composite score

Interactive: Click any cell to flip it and see real-time recalculation of all metrics. The Z3 solve happens on keystroke with debouncing.

**View 3: Arbitrage Dossier**

A legal-brief-style document view showing:
- Executive summary with headline savings
- Side-by-side baseline vs. optimized scenarios
- Provenance chain as a timeline with statute citations
- Dependency graph visualization
- Risk flags with explanations
- Cryptographic proof verification badge (click to see hash chain)
- PDF export button for client delivery

**View 4: Contradiction Heat Map**

A matrix where:
- **Rows and Columns** = PolicyAtoms
- **Cell color** = Conflict intensity between the two atoms (from `SymbolicConflictChecker`)
- **Red clusters** = Groups of mutually contradictory rules
- **Drill-down** = Click a cell to see the Z3 UNSAT core explaining the contradiction

**View 5: Monitoring & Alerts**

- Feed of regulatory changes (new rules ingested, rules modified, rules repealed)
- Impact analysis: "Rule X changed. 47 of your SKUs are affected. 3 optimizations are now invalid."
- Automated re-run of scenario analysis with before/after comparison
- Compliance calendar: upcoming deadlines triggered by temporal clauses in rules

#### Technology Stack Recommendation

- **Frontend:** React + D3.js for graph visualizations, AG Grid for scenario explorer
- **Backend:** The existing FastAPI app (20+ routes already built)
- **Real-time:** WebSocket connection for live scenario recalculation
- **Export:** Use the existing proof engine to generate PDF dossiers with embedded hash verification

---

### 3.4 Priority Roadmap

| Quarter | Milestone | Key Deliverable |
|---------|-----------|-----------------|
| Q1 | **Solver Hardening** | Incremental solving, solver pool, jurisdiction partitioning. Target: 10K atoms, <500ms per check_scenario. |
| Q1 | **TEaaS MVP** | Real USITC HTS data feed, tenant isolation, per-SKU pricing. Launch with 3 pilot importers. |
| Q2 | **LLM Extraction Pipeline** | PDF->PolicyAtom pipeline with confidence gating and human review. Target: 80% auto-extraction accuracy. |
| Q2 | **Dashboard v1** | Scenario Explorer + Arbitrage Dossier views. Ship to TEaaS pilot customers. |
| Q3 | **Export Control Module** | EAR Part 774 ingestion, transaction screening API. Target: 3 defense/aerospace pilot customers. |
| Q3 | **Dashboard v2** | Regulatory Landscape Map + Contradiction Heat Map. |
| Q4 | **Scale Engineering** | BDD pre-filter, scenario caching, concurrent solver pool. Target: 100K atoms, <2s per query. |
| Q4 | **Treaty Arbitrage Beta** | US-Ireland-Cayman corridor with real treaty data. Target: 5 crypto fund pilots. |

---

## APPENDIX: KEY FILE REFERENCE

| File | Lines | Role in Architecture |
|------|-------|---------------------|
| `src/potatobacon/law/solver_z3.py` | 275 | Z3 constraint compilation, SAT checking |
| `src/potatobacon/law/arbitrage_hunter.py` | 471 | Scenario search and ranking |
| `src/potatobacon/law/cale_metrics.py` | 281 | Entropy, contradiction, composite scoring |
| `src/potatobacon/law/arbitrage_models.py` | 80 | Pydantic output models |
| `src/potatobacon/law/contradiction_score.py` | 65 | Pairwise contradiction probability |
| `src/potatobacon/law/ambiguity_entropy.py` | 23 | Normalized Shannon entropy |
| `src/potatobacon/law/features_hierarchy.py` | 74 | Precedence, directness, temporal priority |
| `src/potatobacon/cale/symbolic.py` | 145 | Deontic logic conflict checker |
| `src/potatobacon/cale/parser.py` | 320 | Rule text -> LegalRule parser |
| `src/potatobacon/cale/types.py` | 169 | LegalRule, ConflictAnalysis datatypes |
| `src/potatobacon/tariff/engine.py` | ~350 | Tariff hack orchestration |
| `src/potatobacon/tariff/optimizer.py` | ~250 | Candidate mutation optimization |
| `src/potatobacon/tariff/overlays.py` | 185 | Section 232/301 overlay application |
| `src/potatobacon/tariff/origin_engine.py` | 348 | USMCA origin analysis + RVC |
| `src/potatobacon/proofs/engine.py` | 233 | Cryptographic proof generation |
| `tests/experiments/exp_tariff_converse.py` | ~30 | Converse felt-sole tariff hack |
| `tests/experiments/exp_tariff_tesla_bolt.py` | ~20 | Tesla bolt material optimization |
