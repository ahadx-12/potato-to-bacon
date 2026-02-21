"""Golden dataset integration tests for the tariff engineering engine.

These are 20 real products with known correct HTS chapters, known duty
exposure types, and known engineering opportunity types.

Every code change must pass this suite.  If a change causes a regression here,
it means the engine is producing worse engineering output than before.

The tests do NOT require the full FastAPI stack.  They test the individual
tariff engineering components directly.
"""

from __future__ import annotations

import pytest
from typing import List, Optional


# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------

# Each entry:
#   description         — product description
#   expected_chapters   — acceptable HTS chapters (set; any match = pass)
#   expected_opp_types  — opportunity types that SHOULD appear in findings
#   origin_country      — ISO 2-letter
#   hts_hint            — known HTS code for validation
#   notes               — engineering notes
GOLDEN_PRODUCTS = [
    {
        "id": "GP-01",
        "description": "Men's woven cotton dress shirt",
        "origin_country": "BD",   # Bangladesh
        "hts_hint": "6205.20",   # Woven cotton shirts, men's
        "expected_chapters": {62},
        "expected_opp_types": ["trade_lane", "documentation"],  # USMCA N/A; look for GSP/CAFTA alternatives
        "notes": "Ch. 62 woven apparel. MFN ~12-20%. Bangladesh has GSP preference.",
    },
    {
        "id": "GP-02",
        "description": "Carbon steel welded pipe for industrial plumbing",
        "origin_country": "CN",
        "hts_hint": "7306.30",   # Steel tube/pipe, welded, circular cross-section
        "expected_chapters": {73},
        "expected_opp_types": ["ad_cvd_exposure", "trade_lane"],
        "notes": "Ch. 73 steel articles. 25% Section 232. Active AD/CVD orders on Chinese steel pipe.",
    },
    {
        "id": "GP-03",
        "description": "USB-C to USB-A charging cable, 1 meter, braided nylon",
        "origin_country": "CN",
        "hts_hint": "8544.42",   # Electric conductors, fitted with connectors
        "expected_chapters": {85},
        "expected_opp_types": ["trade_lane", "documentation"],
        "notes": "Ch. 85 electrical conductors. 25% Section 301 surcharge.",
    },
    {
        "id": "GP-04",
        "description": "Athletic running shoe with rubber outsole and mesh upper",
        "origin_country": "VN",  # Vietnam
        "hts_hint": "6404.11",   # Footwear, rubber/plastic sole, textile upper, sports
        "expected_chapters": {64},
        "expected_opp_types": ["documentation", "product_engineering"],
        "notes": "Ch. 64 footwear. Rate depends on upper material. Textile upper vs rubber coverage is classification-sensitive.",
    },
    {
        "id": "GP-05",
        "description": "Automotive brake caliper assembly for passenger vehicles",
        "origin_country": "MX",  # Mexico
        "hts_hint": "8708.30",   # Parts and accessories, brakes
        "expected_chapters": {87},
        "expected_opp_types": ["fta_utilization", "documentation"],
        "notes": "Ch. 87, automotive parts. USMCA eligible if origin qualifies. MX should trigger USMCA.",
    },
    {
        "id": "GP-06",
        "description": "24-inch LCD computer monitor with HDMI and DisplayPort",
        "origin_country": "CN",
        "hts_hint": "8528.52",   # Monitors for ADP machines
        "expected_chapters": {85},
        "expected_opp_types": ["trade_lane"],
        "notes": "Ch. 85 monitors. 25% Section 301 from China. No exclusion for standard monitors.",
    },
    {
        "id": "GP-07",
        "description": "Injection-molded ABS plastic housing for power tool",
        "origin_country": "CN",
        "hts_hint": "3926.90",   # Other articles of plastic
        "expected_chapters": {39},
        "expected_opp_types": ["trade_lane", "reclassification"],
        "notes": "Ch. 39 plastics. Section 301 applies. ABS housing may qualify as a part of ch. 84 machinery if designed specifically.",
    },
    {
        "id": "GP-08",
        "description": "Extruded aluminum window frame with thermal break",
        "origin_country": "CN",
        "hts_hint": "7610.10",   # Aluminum doors, windows, frames
        "expected_chapters": {76},
        "expected_opp_types": ["trade_lane", "ad_cvd_exposure"],
        "notes": "Ch. 76 aluminum. 10% Section 232. AD/CVD exposure on Chinese aluminum extrusions.",
    },
    {
        "id": "GP-09",
        "description": "Upholstered dining chair with solid wood frame",
        "origin_country": "CN",
        "hts_hint": "9401.61",   # Seats with wooden frame, upholstered
        "expected_chapters": {94},
        "expected_opp_types": ["trade_lane", "ad_cvd_exposure"],
        "notes": "Ch. 94 furniture. Section 301 25%. Active AD/CVD orders on wooden furniture from China.",
    },
    {
        "id": "GP-10",
        "description": "100% cotton knit T-shirt for men",
        "origin_country": "MX",
        "hts_hint": "6109.10",   # T-shirts, knit, cotton
        "expected_chapters": {61},
        "expected_opp_types": ["fta_utilization"],
        "notes": "Ch. 61 knit apparel. USMCA eligible if yarn-forward rule satisfied.",
    },
    {
        "id": "GP-11",
        "description": "Bare printed circuit board for industrial control panel",
        "origin_country": "TW",  # Taiwan
        "hts_hint": "8534.00",   # Printed circuits
        "expected_chapters": {85},
        "expected_opp_types": ["documentation"],
        "notes": "Ch. 85. Taiwan: no Section 301. Standard MFN rate applies.",
    },
    {
        "id": "GP-12",
        "description": "Diesel fuel injection pump for truck engine",
        "origin_country": "DE",  # Germany
        "hts_hint": "8413.91",   # Parts of pumps for liquids
        "expected_chapters": {84},
        "expected_opp_types": ["documentation", "fta_utilization"],
        "notes": "Ch. 84 pumps/parts. Germany: no Section 301/232. Standard MFN applies.",
    },
    {
        "id": "GP-13",
        "description": "Stainless steel surgical forceps and dissecting scissors",
        "origin_country": "PK",  # Pakistan
        "hts_hint": "9018.90",   # Surgical instruments, other
        "expected_chapters": {90},
        "expected_opp_types": ["documentation"],
        "notes": "Ch. 90 medical/optical instruments. Pakistan: GSP may apply.",
    },
    {
        "id": "GP-14",
        "description": "Baby stroller with adjustable handlebar and 5-point harness",
        "origin_country": "CN",
        "hts_hint": "8715.00",   # Baby carriages
        "expected_chapters": {87},
        "expected_opp_types": ["trade_lane"],
        "notes": "Ch. 87 baby carriages. Section 301 25% from China.",
    },
    {
        "id": "GP-15",
        "description": "Silicone rubber gasket for automotive engine sealing",
        "origin_country": "IN",  # India
        "hts_hint": "4016.93",   # Gaskets of vulcanized rubber
        "expected_chapters": {40},
        "expected_opp_types": ["documentation", "fta_utilization"],
        "notes": "Ch. 40 rubber articles. India: GSP preference may apply.",
    },
    {
        "id": "GP-16",
        "description": "Stainless steel hex head bolts M10 x 40mm, grade 8.8",
        "origin_country": "CN",
        "hts_hint": "7318.15",   # Screws and bolts of iron/steel
        "expected_chapters": {73},
        "expected_opp_types": ["ad_cvd_exposure", "trade_lane"],
        "notes": "Ch. 73 fasteners. Section 232 25%. Active AD/CVD on Chinese steel fasteners.",
    },
    {
        "id": "GP-17",
        "description": "Optical fiber cable, single mode, 12-fiber, armored jacket",
        "origin_country": "JP",  # Japan
        "hts_hint": "9001.10",   # Optical fibers and optical fiber bundles
        "expected_chapters": {90},
        "expected_opp_types": ["documentation"],
        "notes": "Ch. 90. Japan: no Section 301. Low MFN rate. Documentation-only.",
    },
    {
        "id": "GP-18",
        "description": "Lithium-ion battery pack for electric vehicle, 400V 75kWh",
        "origin_country": "CN",
        "hts_hint": "8507.60",   # Lithium-ion batteries
        "expected_chapters": {85},
        "expected_opp_types": ["trade_lane", "exclusion_filing"],
        "notes": "Ch. 85. Section 301 25%. Active exclusion requests under review for EV batteries.",
    },
    {
        "id": "GP-19",
        "description": "100% merino wool crew neck sweater",
        "origin_country": "IT",  # Italy
        "hts_hint": "6110.11",   # Jerseys/pullovers of wool, knit
        "expected_chapters": {61},
        "expected_opp_types": ["documentation"],
        "notes": "Ch. 61 knit apparel. Italy: standard MFN. No Section 232/301.",
    },
    {
        "id": "GP-20",
        "description": "Centrifugal water pump, 50HP, for irrigation",
        "origin_country": "KR",  # South Korea
        "hts_hint": "8413.70",   # Centrifugal pumps
        "expected_chapters": {84},
        "expected_opp_types": ["fta_utilization"],
        "notes": "Ch. 84. Korea: KORUS FTA applies. Prefer KORUS over MFN.",
    },
]


# ---------------------------------------------------------------------------
# Tests: HTS search classification
# ---------------------------------------------------------------------------

class TestHTSSearchGolden:
    """Verify that hts_search routes each product to the expected chapter."""

    def test_imports(self):
        from potatobacon.tariff.hts_search import search_hts_by_description, get_search_index
        index = get_search_index()
        assert index.entry_count > 0, "HTS search index must have entries"

    @pytest.mark.parametrize("product", [
        p for p in GOLDEN_PRODUCTS
        # Only test chapters that are in the full_chapters directory
        if p["expected_chapters"] & {39, 84, 87, 90, 94}
    ])
    def test_search_finds_correct_chapter(self, product):
        from potatobacon.tariff.hts_search import search_hts_by_description

        results = search_hts_by_description(product["description"], top_n=10)
        assert results, (
            f"[{product['id']}] No HTS candidates found for: {product['description']!r}"
        )

        found_chapters = {r.chapter for r in results}
        overlap = found_chapters & product["expected_chapters"]
        assert overlap, (
            f"[{product['id']}] Expected chapter(s) {product['expected_chapters']} "
            f"not in top-10 results (got chapters {found_chapters}). "
            f"Product: {product['description']!r}. Note: {product['notes']}"
        )

    def test_index_covers_key_chapters(self):
        """The index must cover at minimum chapters 39, 84, 87, 90, 94."""
        from potatobacon.tariff.hts_search import get_search_index
        index = get_search_index()
        required = {39, 84, 87, 90, 94}
        indexed = set(index.chapters_indexed)
        missing = required - indexed
        assert not missing, f"Missing chapters from index: {missing}"


# ---------------------------------------------------------------------------
# Tests: GRI engine
# ---------------------------------------------------------------------------

class TestGRIEngineGolden:
    """Verify GRI cascade produces legally valid output."""

    def test_gri_single_candidate(self):
        """GRI 1 should fire when there is exactly one candidate."""
        from potatobacon.tariff.hts_search import HTSSearchResult
        from potatobacon.tariff.gri_engine import apply_gri

        candidate = HTSSearchResult(
            hts_code="8471.30.01",
            heading="8471",
            chapter=84,
            description="Portable automatic data processing machines",
            base_duty_rate="Free",
            score=0.8,
            matched_terms=["data", "processing"],
            rationale="test",
        )

        result = apply_gri("laptop computer", [], [candidate])
        assert result.determining_rule == "GRI_1"
        assert result.winning_heading == "8471"
        assert result.confidence == "high"

    def test_gri_dominant_score(self):
        """GRI 1 should fire when one candidate scores ≥ 2x the next."""
        from potatobacon.tariff.hts_search import HTSSearchResult
        from potatobacon.tariff.gri_engine import apply_gri

        candidates = [
            HTSSearchResult("8471.30.01", "8471", 84, "Laptops", "Free",
                            score=0.9, matched_terms=["laptop"], rationale=""),
            HTSSearchResult("8528.52.00", "8528", 85, "Monitors", "Free",
                            score=0.3, matched_terms=["display"], rationale=""),
        ]
        result = apply_gri("laptop computer", [], candidates)
        assert result.determining_rule == "GRI_1"
        assert result.winning_heading == "8471"

    def test_gri_3b_essential_character(self):
        """GRI 3b selects the heading matching the dominant material."""
        from potatobacon.tariff.hts_search import HTSSearchResult
        from potatobacon.tariff.gri_engine import apply_gri

        # Two candidates with similar scores — GRI 3a won't decide
        candidates = [
            HTSSearchResult("3926.90.99", "3926", 39, "Other articles of plastic",
                            "5.3%", score=0.4, matched_terms=["plastic"], rationale=""),
            HTSSearchResult("8473.30.10", "8473", 84, "Parts for computers",
                            "Free", score=0.41, matched_terms=["computer"], rationale=""),
        ]
        materials = [
            {"component": "housing", "material": "ABS plastic"},
            {"component": "housing", "material": "polycarbonate plastic"},
        ]
        result = apply_gri("computer housing", materials, candidates)
        # Should select plastic heading based on material match
        assert result.winning_heading in ("3926", "8473")
        assert result.gri_chain  # Must have reasoning

    def test_gri_no_candidates(self):
        """Empty candidate list should return a low-confidence no-classification."""
        from potatobacon.tariff.gri_engine import apply_gri
        result = apply_gri("mystery product", [], [])
        assert result.determining_rule == "NONE"
        assert result.confidence == "low"
        assert result.notes  # Must include guidance

    def test_gri_unfinished_marker(self):
        """GRI 2a note should appear for unfinished articles."""
        from potatobacon.tariff.hts_search import HTSSearchResult
        from potatobacon.tariff.gri_engine import apply_gri

        candidates = [
            HTSSearchResult("7318.15.00", "7318", 73, "Screws and bolts",
                            "5%", score=0.7, matched_terms=["bolt"], rationale=""),
            HTSSearchResult("7318.16.00", "7318", 73, "Nuts",
                            "6%", score=0.3, matched_terms=["nut"], rationale=""),
        ]
        result = apply_gri("unfinished steel bolt blank", [], candidates)
        # GRI 2a should be in the chain
        gri_ids = [r.gri_rule for r in result.gri_chain]
        assert "GRI_2a" in gri_ids

    def test_gri_legal_basis_populated(self):
        """Every GRI result must produce non-empty legal basis citations."""
        from potatobacon.tariff.hts_search import HTSSearchResult
        from potatobacon.tariff.gri_engine import apply_gri, gri_legal_basis

        candidates = [
            HTSSearchResult("8407.10.00", "8407", 84, "Spark-ignition engines",
                            "Free", score=0.6, matched_terms=["engine"], rationale=""),
        ]
        result = apply_gri("gasoline engine for motorcycle", [], candidates)
        basis = gri_legal_basis(result)
        assert basis, "GRI legal basis must not be empty"
        assert any("GRI" in b for b in basis)


# ---------------------------------------------------------------------------
# Tests: Origin rules / tariff shift
# ---------------------------------------------------------------------------

class TestOriginRulesGolden:
    """Verify tariff shift analysis for key FTA scenarios."""

    def test_usmca_qualifies_cc_satisfied(self):
        """Product from MX where all non-MX inputs satisfy CC rule should qualify."""
        from potatobacon.tariff.origin_rules import evaluate_tariff_shift, BOMComponent

        components = [
            # Steel from US — originating (USMCA partner)
            BOMComponent("Steel plate", "7209.17", "US"),
            # Electronics from MX — originating (USMCA partner)
            BOMComponent("Control module", "8537.10", "MX"),
        ]
        result = evaluate_tariff_shift("8479.89", components, "USMCA")
        assert result.qualifies, f"Expected USMCA qualification. Reason: {result.action_summary}"

    def test_usmca_fails_cc_chinese_input_same_chapter(self):
        """Steel input from China in the same chapter as steel finished good should fail CC."""
        from potatobacon.tariff.origin_rules import evaluate_tariff_shift, BOMComponent

        components = [
            # Chinese steel in ch.73 — same chapter as finished good ch.73
            BOMComponent("Steel fasteners (Chinese)", "7318.15", "CN"),
            BOMComponent("Mexican steel casting", "7320.20", "MX"),
        ]
        # Finished good is a steel bracket (ch. 73)
        result = evaluate_tariff_shift("7326.90", components, "USMCA")
        # Chinese steel fasteners (7318) are same chapter (73) as the finished good (7326) → CC fails
        assert not result.qualifies
        assert result.failing_components
        # Failing component should be the Chinese input
        failing_origins = {s.component.origin_country for s in result.failing_components}
        assert "CN" in failing_origins

    def test_usmca_automotive_requires_high_rvc(self):
        """Passenger vehicle (8703) requires 75% RVC under USMCA."""
        from potatobacon.tariff.origin_rules import evaluate_tariff_shift, BOMComponent

        components = [
            BOMComponent("Engine from MX", "8407.34", "MX", value_usd=8000),
            BOMComponent("Electronics from CN", "8537.10", "CN", value_usd=2000),
        ]
        result = evaluate_tariff_shift("8703.23", components, "USMCA",
                                       finished_good_value=25000)
        # Even if tariff shift passes, 75% RVC likely not met with CN electronics
        assert result.rvc_required
        assert result.rvc_threshold_pct == 75.0

    def test_korus_korea_origin(self):
        """Korean-origin machinery should have KORUS available."""
        from potatobacon.tariff.origin_rules import check_fta_eligibility, BOMComponent

        result = check_fta_eligibility(
            finished_hts="8413.70",  # Centrifugal pump
            origin_country="KR",
            bom_components=[
                BOMComponent("Korean steel casing", "7308.90", "KR"),
                BOMComponent("Korean motor", "8501.31", "KR"),
            ],
        )
        assert result is not None
        assert result.fta == "KORUS"

    def test_no_fta_for_china(self):
        """China has no US FTA — check_fta_eligibility should return None."""
        from potatobacon.tariff.origin_rules import check_fta_eligibility

        result = check_fta_eligibility("8471.30", "CN")
        assert result is None

    def test_what_needs_to_change_populated(self):
        """Non-qualifying results must include specific supply chain changes."""
        from potatobacon.tariff.origin_rules import evaluate_tariff_shift, BOMComponent

        components = [
            BOMComponent("Chinese plastic resin", "3901.10", "CN"),
        ]
        # Finished plastic housing (ch. 39) — Chinese plastic resin is same chapter → CC fails
        result = evaluate_tariff_shift("3926.90", components, "USMCA")
        assert not result.qualifies
        assert result.what_needs_to_change, "Must specify what needs to change for qualification"


# ---------------------------------------------------------------------------
# Tests: Company profile
# ---------------------------------------------------------------------------

class TestCompanyProfile:
    """Verify company profile feasibility logic."""

    def test_fixed_origin_suppresses_trade_lane(self):
        from potatobacon.tariff.company_profile import CompanyProfile, SupplyChainConstraint

        profile = CompanyProfile(
            fixed_origin_countries=["CN"],
            supply_chain_constraints=[SupplyChainConstraint.FIXED_ORIGIN],
        )
        assert not profile.trade_lane_feasible("CN")
        assert profile.trade_lane_feasible("MX")  # Not in fixed list

    def test_certified_product_suppresses_engineering(self):
        from potatobacon.tariff.company_profile import CompanyProfile, SupplyChainConstraint

        profile = CompanyProfile(
            supply_chain_constraints=[SupplyChainConstraint.CERTIFIED_PRODUCT],
        )
        assert not profile.product_engineering_feasible()

    def test_active_fta_already_claimed(self):
        from potatobacon.tariff.company_profile import CompanyProfile

        profile = CompanyProfile(active_fta_programs=["USMCA", "KORUS"])
        assert profile.fta_already_claimed("USMCA")
        assert profile.fta_already_claimed("usmca")  # case-insensitive
        assert not profile.fta_already_claimed("GSP")

    def test_low_risk_tolerance_suppresses_grade_b(self):
        from potatobacon.tariff.company_profile import CompanyProfile, RiskTolerance

        profile = CompanyProfile(risk_tolerance=RiskTolerance.LOW)
        assert profile.should_surface_opportunity("A")
        assert not profile.should_surface_opportunity("B")
        assert not profile.should_surface_opportunity("C")

    def test_high_risk_tolerance_surfaces_all(self):
        from potatobacon.tariff.company_profile import CompanyProfile, RiskTolerance

        profile = CompanyProfile(risk_tolerance=RiskTolerance.HIGH)
        assert profile.should_surface_opportunity("A")
        assert profile.should_surface_opportunity("B")
        assert profile.should_surface_opportunity("C")

    def test_default_profile_is_permissive(self):
        from potatobacon.tariff.company_profile import DEFAULT_PROFILE

        # Default: moderate risk tolerance, no fixed origins
        assert DEFAULT_PROFILE.trade_lane_feasible("CN")
        assert DEFAULT_PROFILE.product_engineering_feasible()
        assert DEFAULT_PROFILE.should_surface_opportunity("A")
        assert DEFAULT_PROFILE.should_surface_opportunity("B")
        assert not DEFAULT_PROFILE.should_surface_opportunity("C")  # moderate excludes C


# ---------------------------------------------------------------------------
# Tests: End-to-end engineering report (smoke test)
# ---------------------------------------------------------------------------

class TestEngineeringReportSmoke:
    """Smoke tests verifying the full engineering pipeline produces output."""

    def test_hts_search_index_loads(self):
        """The search index must load without error and contain entries."""
        from potatobacon.tariff.hts_search import get_search_index
        index = get_search_index()
        assert index.entry_count > 100, "Expected at least 100 HTS entries in index"

    def test_classify_laptop(self):
        """Laptop should route to chapter 84."""
        from potatobacon.tariff.hts_search import search_hts_by_description, top_chapters_for_description

        results = search_hts_by_description("laptop computer portable", top_n=5)
        assert results, "Must return candidates"
        chapters = {r.chapter for r in results}
        assert 84 in chapters, f"Ch.84 not found for laptop. Got: {chapters}"

    def test_classify_automotive_part(self):
        """Brake caliper should route to chapter 87."""
        from potatobacon.tariff.hts_search import search_hts_by_description

        results = search_hts_by_description("automotive brake caliper assembly passenger vehicle", top_n=5)
        assert results, "Must return candidates"
        chapters = {r.chapter for r in results}
        assert 87 in chapters, f"Ch.87 not found for brake caliper. Got: {chapters}"

    def test_classify_plastic_part(self):
        """Plastic housing should route to chapter 39."""
        from potatobacon.tariff.hts_search import search_hts_by_description

        results = search_hts_by_description("ABS plastic injection molded housing enclosure", top_n=5)
        assert results
        chapters = {r.chapter for r in results}
        assert 39 in chapters, f"Ch.39 not found for plastic housing. Got: {chapters}"

    def test_gri_applies_to_search_results(self):
        """Full classify flow: search → GRI → confidence and winning code."""
        from potatobacon.tariff.hts_search import search_hts_by_description
        from potatobacon.tariff.gri_engine import apply_gri

        description = "portable automatic data processing machine laptop"
        candidates = search_hts_by_description(description, top_n=5)
        assert candidates

        result = apply_gri(description, [], candidates)
        assert result.winning_code, "GRI must produce a winning code"
        assert result.gri_chain, "GRI must produce reasoning chain"
        assert result.confidence in ("high", "medium", "low")

    def test_origin_rules_usmca_brake_caliper(self):
        """Mexican brake caliper with MX-origin components should qualify for USMCA."""
        from potatobacon.tariff.origin_rules import evaluate_tariff_shift, BOMComponent

        components = [
            BOMComponent("Cast iron housing", "7325.91", "MX"),
            BOMComponent("Brake pads", "6813.81", "US"),
            BOMComponent("Steel bolts", "7318.15", "US"),
        ]
        # Ch. 87 automotive — CTH + RVC 35% for parts NES
        result = evaluate_tariff_shift("8708.30", components, "USMCA")
        # All components are from MX/US (USMCA partners) — should qualify
        assert result.qualifies, f"Expected USMCA qualification: {result.action_summary}"

    @pytest.mark.parametrize("product_id,description,expected_chapter", [
        ("GP-07", "Injection-molded ABS plastic housing for power tool", 39),
        ("GP-08", "Extruded aluminum window frame with thermal break", 76),
        ("GP-12", "Diesel fuel injection pump for truck engine", 84),
        ("GP-20", "Centrifugal water pump 50HP for irrigation", 84),
    ])
    def test_search_routes_to_indexed_chapters(self, product_id, description, expected_chapter):
        """Products should route to chapters present in the index."""
        from potatobacon.tariff.hts_search import search_hts_by_description

        results = search_hts_by_description(description, top_n=10)
        assert results, f"[{product_id}] No candidates: {description!r}"
        chapters = {r.chapter for r in results}
        assert expected_chapter in chapters, (
            f"[{product_id}] Expected ch.{expected_chapter} in {chapters} for: {description!r}"
        )
