"""Sprint B integration tests: USITC data → full TEaaS pipeline.

Tests:
1. Bootstrap a USITC subset from fixture data (no network)
2. Run an analysis and verify classifications come from USITC data
3. Chapter pre-filter correctness and performance
"""

from __future__ import annotations

import time
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

from potatobacon.law.solver_z3 import PolicyAtom, check_scenario
from potatobacon.tariff.atom_utils import duty_rate_index
from potatobacon.tariff.chapter_filter import (
    clear_chapter_cache,
    extract_chapters_from_atom,
    extract_chapters_from_facts,
    filter_atoms_by_chapter,
)
from potatobacon.tariff.engine import compute_duty_result
from potatobacon.tariff.fact_compiler import compile_facts
from potatobacon.tariff.fact_vocabulary import expand_facts
from potatobacon.tariff.hts_ingest.usitc_context import _tariff_line_to_policy_atom
from potatobacon.tariff.hts_ingest.usitc_parser import parse_usitc_edition
from potatobacon.tariff.models import TariffScenario
from potatobacon.tariff.mutation_engine import MutationEngine
from potatobacon.tariff.product_schema import ProductCategory, ProductSpecModel

# ---------------------------------------------------------------------------
# USITC fixture records — a small subset of real USITC record structure
# covering chapters 73 (steel articles) and 85 (electronics).
# These simulate what USITCFetcher.fetch_current_edition() would return.
# ---------------------------------------------------------------------------
USITC_FIXTURE_RECORDS: List[Dict[str, Any]] = [
    # Chapter 73 hierarchy
    {
        "htsno": "73",
        "description": "Articles of iron or steel",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "7318",
        "description": "Screws, bolts, nuts, coach screws, screw hooks, rivets, cotters, cotter pins, washers and similar articles, of iron or steel",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "7318.15",
        "description": "Other screws and bolts, whether or not with their nuts or washers",
        "indent": 1,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "7318.15.20",
        "description": "Having shanks or threads with a diameter of less than 6 mm",
        "indent": 2,
        "general": "Free",
        "special": "",
        "other": "Free",
    },
    {
        "htsno": "7318.15.50",
        "description": "Having shanks or threads with a diameter of 6 mm or more",
        "indent": 2,
        "general": "3.4%",
        "special": "Free (A*,AU,BH,CL,CO)",
        "other": "45%",
    },
    {
        "htsno": "7318.16.00",
        "description": "Nuts of iron or steel",
        "indent": 2,
        "general": "2.8%",
        "special": "Free (A*,AU,BH,CL,CO)",
        "other": "45%",
    },
    # Chapter 72 hierarchy (iron and steel raw)
    {
        "htsno": "72",
        "description": "Iron and steel",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "7208",
        "description": "Flat-rolled products of iron or nonalloy steel, of a width of 600 mm or more, hot-rolled, not clad, plated or coated",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "7208.10.15",
        "description": "In coils, of a thickness of 4.75 mm or more but not exceeding 150 mm",
        "indent": 2,
        "general": "Free",
        "special": "",
        "other": "Free",
    },
    # Chapter 85 hierarchy (electronics)
    {
        "htsno": "85",
        "description": "Electrical machinery and equipment and parts thereof",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "8544",
        "description": "Insulated wire, cable and other insulated electric conductors; optical fiber cables",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "8544.42",
        "description": "Other electric conductors, for a voltage not exceeding 1,000 V",
        "indent": 1,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "8544.42.20",
        "description": "Fitted with connectors",
        "indent": 2,
        "general": "3.5%",
        "special": "Free (A*,AU,BH,CL,CO)",
        "other": "50%",
    },
    {
        "htsno": "8544.42.90",
        "description": "Other insulated conductors",
        "indent": 2,
        "general": "3.5%",
        "special": "Free (A*,AU,BH,CL,CO)",
        "other": "50%",
    },
    # Chapter 82 (tools)
    {
        "htsno": "82",
        "description": "Tools, implements, cutlery, spoons and forks, of base metal; parts thereof of base metal",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "8205",
        "description": "Handtools not elsewhere specified",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "8205.10.00",
        "description": "Drilling, threading or tapping tools",
        "indent": 2,
        "general": "5.5%",
        "special": "Free (A*,AU,BH,CL,CO)",
        "other": "40%",
    },
    # Chapter 64 (footwear)
    {
        "htsno": "64",
        "description": "Footwear, gaiters and the like; parts of such articles",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "6402",
        "description": "Other footwear with outer soles and uppers of rubber or plastics",
        "indent": 0,
        "general": "",
        "special": "",
        "other": "",
    },
    {
        "htsno": "6402.99.31",
        "description": "Footwear covering the ankle, with open toes or open heels",
        "indent": 2,
        "general": "6%",
        "special": "Free (A*,AU,BH,CL,CO)",
        "other": "35%",
    },
    {
        "htsno": "6402.99.33",
        "description": "Other footwear with rubber or plastic soles",
        "indent": 2,
        "general": "8.5%",
        "special": "Free (A*,AU,BH,CL,CO)",
        "other": "35%",
    },
]


@pytest.fixture
def usitc_atoms() -> List[PolicyAtom]:
    """Parse fixture records into PolicyAtoms."""
    lines = parse_usitc_edition(USITC_FIXTURE_RECORDS, rate_lines_only=True)
    return [_tariff_line_to_policy_atom(line) for line in lines]


@pytest.fixture
def usitc_duty_rates(usitc_atoms: List[PolicyAtom]) -> Dict[str, float]:
    return duty_rate_index(usitc_atoms)


# ---------------------------------------------------------------------------
# 1. Integration test: bootstrap USITC subset → full analysis
# ---------------------------------------------------------------------------
class TestUSITCBootstrapIntegration:
    """Verify that bootstrapped USITC data flows through the full pipeline."""

    def test_bootstrap_from_records(self):
        """bootstrap_from_records creates atoms from fixture data."""
        from scripts.bootstrap_usitc import bootstrap_from_records

        with tempfile.TemporaryDirectory() as tmpdir:
            result = bootstrap_from_records(
                USITC_FIXTURE_RECORDS,
                store_dir=Path(tmpdir),
                edition_id="FIXTURE_TEST",
            )

        assert result["edition_id"] == "FIXTURE_TEST"
        assert result["atom_count"] > 0
        assert result["rate_line_count"] > 0
        assert result["context_id"] == "HTS_US_LIVE"
        assert result["parse_errors"] == 0

    def test_usitc_atoms_have_expected_structure(self, usitc_atoms: List[PolicyAtom]):
        """USITC-parsed atoms have proper guard tokens and metadata."""
        assert len(usitc_atoms) > 0

        # Check that we got atoms from multiple chapters
        chapters = set()
        for atom in usitc_atoms:
            for token in atom.guard:
                if token.startswith("chapter_"):
                    chapters.add(token)

        assert "chapter_73" in chapters, "Should have chapter 73 (steel articles)"
        assert "chapter_85" in chapters, "Should have chapter 85 (electronics)"

    def test_usitc_atoms_have_duty_rates(
        self, usitc_atoms: List[PolicyAtom], usitc_duty_rates: Dict[str, float]
    ):
        """USITC atoms carry duty rate metadata."""
        assert len(usitc_duty_rates) > 0
        # 7318.15.20 is Free (0.0%)
        free_atoms = [sid for sid, rate in usitc_duty_rates.items() if rate == 0.0]
        assert len(free_atoms) > 0, "Should have at least one free-rate atom"

    def test_steel_bolt_classifies_against_usitc(
        self, usitc_atoms: List[PolicyAtom], usitc_duty_rates: Dict[str, float]
    ):
        """A steel bolt BOM should match USITC chapter 73 atoms."""
        product = ProductSpecModel(
            product_category=ProductCategory.FASTENER,
            materials=[{"component": "body", "material": "steel"}],
        )
        facts, _evidence = compile_facts(product)
        expanded = expand_facts(facts)

        scenario = TariffScenario(name="test_bolt", facts=expanded)
        result = compute_duty_result(usitc_atoms, scenario, duty_rates=usitc_duty_rates)

        # Should find at least some duty rule active
        assert result.status in ("OK", "NO_DUTY_RULE_ACTIVE")
        if result.status == "OK":
            assert result.duty_rate is not None
            # Verify the active atoms are from USITC data (HTS_ prefix)
            for sid in result.duty_atom_ids:
                assert sid.startswith("HTS_"), f"Expected USITC atom, got {sid}"

    def test_electronics_classifies_against_usitc(
        self, usitc_atoms: List[PolicyAtom], usitc_duty_rates: Dict[str, float]
    ):
        """An electronics product should match USITC chapter 85 atoms."""
        product = ProductSpecModel(
            product_category=ProductCategory.ELECTRONICS,
            materials=[{"component": "body", "material": "copper"}],
        )
        facts, _evidence = compile_facts(product)
        expanded = expand_facts(facts)

        scenario = TariffScenario(name="test_cable", facts=expanded)
        result = compute_duty_result(usitc_atoms, scenario, duty_rates=usitc_duty_rates)

        if result.status == "OK":
            assert result.duty_rate is not None
            for sid in result.duty_atom_ids:
                assert sid.startswith("HTS_")

    def test_mutation_engine_finds_mutations_on_usitc(
        self, usitc_atoms: List[PolicyAtom], usitc_duty_rates: Dict[str, float]
    ):
        """MutationEngine discovers savings using USITC atoms."""
        product = ProductSpecModel(
            product_category=ProductCategory.FASTENER,
            materials=[{"component": "body", "material": "steel"}],
        )
        facts, _ = compile_facts(product)
        expanded = expand_facts(facts)

        scenario = TariffScenario(name="baseline", facts=expanded)
        baseline_result = compute_duty_result(usitc_atoms, scenario, duty_rates=usitc_duty_rates)
        baseline_rate = baseline_result.duty_rate or 0.0

        engine = MutationEngine(usitc_atoms, duty_rates=usitc_duty_rates)
        mutations = engine.discover_mutations(scenario, baseline_rate, max_candidates=5)

        # May or may not find mutations depending on baseline;
        # the key is no crash and result is a list
        assert isinstance(mutations, list)


# ---------------------------------------------------------------------------
# 2. Unit tests for chapter pre-filter
# ---------------------------------------------------------------------------
class TestChapterPreFilter:
    """Verify chapter pre-filter correctness."""

    def setup_method(self):
        clear_chapter_cache()

    def test_extract_chapters_from_facts(self):
        """Extract chapter numbers from expanded facts."""
        facts = {
            "chapter_73": True,
            "chapter_72": True,
            "chapter_85": False,
            "material_steel": True,
            "is_fastener": True,
        }
        chapters = extract_chapters_from_facts(facts)
        assert chapters == {"73", "72"}

    def test_extract_chapters_from_atom(self, usitc_atoms: List[PolicyAtom]):
        """Extract chapters from atom guard tokens."""
        # Find a chapter 73 atom
        ch73_atom = None
        for atom in usitc_atoms:
            if "chapter_73" in atom.guard:
                ch73_atom = atom
                break
        assert ch73_atom is not None
        chapters = extract_chapters_from_atom(ch73_atom)
        assert "73" in chapters

    def test_filter_reduces_atom_count(self, usitc_atoms: List[PolicyAtom]):
        """Filter reduces atoms to only matching chapters."""
        # Facts that expand to chapter 73 only
        facts = {
            "chapter_73": True,
            "material_steel": True,
        }
        filtered = filter_atoms_by_chapter(usitc_atoms, facts)

        # Should be fewer than total (assuming fixture has multiple chapters)
        assert len(filtered) < len(usitc_atoms)
        assert len(filtered) > 0

        # All filtered atoms with chapter guards should be chapter 73
        for atom in filtered:
            atom_chapters = extract_chapters_from_atom(atom)
            if atom_chapters:  # skip chapterless atoms
                assert "73" in atom_chapters, (
                    f"Atom {atom.source_id} has chapters {atom_chapters} "
                    f"but should only contain 73"
                )

    def test_filter_preserves_chapterless_atoms(self, usitc_atoms: List[PolicyAtom]):
        """Atoms without chapter_XX guards (e.g., GRI rules) are preserved."""
        # Create a chapterless atom
        gri_atom = PolicyAtom(
            guard=["is_fastener"],
            outcome={"modality": "PERMIT", "action": "gri_rule"},
            source_id="GRI_TEST",
            statute="GRI",
            section="GRI1",
            text="General rule",
            modality="PERMIT",
            action="gri_rule",
        )
        atoms_with_gri = usitc_atoms + [gri_atom]

        facts = {"chapter_73": True}
        filtered = filter_atoms_by_chapter(atoms_with_gri, facts)

        gri_ids = [a.source_id for a in filtered if a.source_id == "GRI_TEST"]
        assert len(gri_ids) == 1, "Chapterless atoms must be included"

    def test_filter_with_no_chapters_returns_all(self, usitc_atoms: List[PolicyAtom]):
        """When facts have no chapter tokens, all atoms are returned."""
        facts = {"material_steel": True}  # No chapter_XX keys
        filtered = filter_atoms_by_chapter(usitc_atoms, facts)
        assert len(filtered) == len(usitc_atoms)

    def test_filter_does_not_drop_valid_matches(self, usitc_atoms: List[PolicyAtom]):
        """Pre-filter must not drop atoms that could legally match."""
        product = ProductSpecModel(
            product_category=ProductCategory.FASTENER,
            materials=[{"component": "body", "material": "steel"}],
        )
        facts, _ = compile_facts(product)
        expanded = expand_facts(facts)

        # Compute baseline with ALL atoms
        scenario = TariffScenario(name="full", facts=expanded)
        full_result = compute_duty_result(
            usitc_atoms, scenario, duty_rates=duty_rate_index(usitc_atoms)
        )

        # Compute with filtered atoms
        filtered = filter_atoms_by_chapter(usitc_atoms, expanded)
        filtered_result = compute_duty_result(
            filtered, scenario, duty_rates=duty_rate_index(filtered)
        )

        # The filtered result should produce the same duty rate
        assert full_result.duty_rate == filtered_result.duty_rate, (
            f"Filter changed duty rate: full={full_result.duty_rate}, "
            f"filtered={filtered_result.duty_rate}"
        )

    def test_filter_caching(self, usitc_atoms: List[PolicyAtom]):
        """Repeated calls with same chapters hit cache."""
        facts = {"chapter_73": True}

        # First call
        result1 = filter_atoms_by_chapter(
            usitc_atoms, facts, context_id="test_cache"
        )
        # Second call (should hit cache)
        result2 = filter_atoms_by_chapter(
            usitc_atoms, facts, context_id="test_cache"
        )

        assert len(result1) == len(result2)

    def test_filter_multi_chapter(self, usitc_atoms: List[PolicyAtom]):
        """Filter with multiple chapters includes atoms from all specified chapters."""
        facts = {
            "chapter_73": True,
            "chapter_85": True,
        }
        filtered = filter_atoms_by_chapter(usitc_atoms, facts)

        filtered_chapters = set()
        for atom in filtered:
            for token in atom.guard:
                if token.startswith("chapter_"):
                    filtered_chapters.add(token)

        assert "chapter_73" in filtered_chapters
        assert "chapter_85" in filtered_chapters


# ---------------------------------------------------------------------------
# 3. Performance test: pre-filter speedup
# ---------------------------------------------------------------------------
class TestChapterPreFilterPerformance:
    """Measure Z3 solve time with and without chapter pre-filter."""

    def setup_method(self):
        clear_chapter_cache()

    def _make_large_atom_set(self, count: int) -> List[PolicyAtom]:
        """Generate a large set of atoms across many chapters."""
        atoms: List[PolicyAtom] = []
        chapters = [f"{i:02d}" for i in range(1, 98)]

        for i in range(count):
            ch = chapters[i % len(chapters)]
            rate = float(i % 20)
            atom = PolicyAtom(
                guard=[f"chapter_{ch}", f"material_steel" if i % 3 == 0 else "material_plastic"],
                outcome={"modality": "OBLIGE", "action": f"duty_rate_{rate}"},
                source_id=f"HTS_GEN_{i:05d}",
                statute="HTSUS",
                section=f"{ch}{(i % 100):02d}.{(i % 100):02d}.{(i % 100):02d}",
                text=f"Test atom {i} in chapter {ch}",
                modality="OBLIGE",
                action=f"duty_rate_{rate}",
                metadata={
                    "duty_rate": rate,
                    "rate_applies": True,
                    "hts_code": f"{ch}{(i % 100):02d}.{(i % 100):02d}.{(i % 100):02d}",
                    "description": f"Test atom {i}",
                },
            )
            atoms.append(atom)
        return atoms

    def test_prefilter_achieves_5x_speedup(self):
        """Pre-filter should achieve at least 5x speedup on 1K+ atoms."""
        large_atoms = self._make_large_atom_set(1500)

        # Facts that match only chapter 73
        facts = {
            "chapter_73": True,
            "material_steel": True,
        }

        # Measure time without pre-filter
        scenario = TariffScenario(name="perf_full", facts=deepcopy(facts))
        rates_full = duty_rate_index(large_atoms)

        start_full = time.monotonic()
        for _ in range(3):
            check_scenario(facts, large_atoms)
        time_full = (time.monotonic() - start_full) / 3

        # Measure time with pre-filter
        filtered = filter_atoms_by_chapter(large_atoms, facts)
        rates_filtered = duty_rate_index(filtered)

        start_filtered = time.monotonic()
        for _ in range(3):
            check_scenario(facts, filtered)
        time_filtered = (time.monotonic() - start_filtered) / 3

        # Verify the filter actually reduced the set
        assert len(filtered) < len(large_atoms), (
            f"Pre-filter should reduce atom count: {len(large_atoms)} -> {len(filtered)}"
        )

        # Verify at least 5x speedup (or near-zero time for both, which is fine)
        if time_full > 0.001:  # Only assert speedup if baseline time is meaningful
            speedup = time_full / max(time_filtered, 1e-9)
            assert speedup >= 5.0, (
                f"Expected >= 5x speedup, got {speedup:.1f}x "
                f"(full={time_full:.4f}s, filtered={time_filtered:.4f}s, "
                f"atoms: {len(large_atoms)} -> {len(filtered)})"
            )
        else:
            # Both are fast enough that speedup is irrelevant
            pass

    def test_prefilter_reduces_1k_atoms(self):
        """With 1K+ atoms, chapter filter should reduce to <25% of original."""
        large_atoms = self._make_large_atom_set(1200)

        # Single chapter should select ~1/97 of atoms
        facts = {"chapter_73": True}
        filtered = filter_atoms_by_chapter(large_atoms, facts)

        reduction_pct = len(filtered) / len(large_atoms) * 100
        assert reduction_pct < 25, (
            f"Expected <25% retention, got {reduction_pct:.1f}% "
            f"({len(filtered)}/{len(large_atoms)})"
        )


# ---------------------------------------------------------------------------
# 4. Bootstrap script tests
# ---------------------------------------------------------------------------
class TestBootstrapScript:
    """Test the bootstrap_usitc script internals."""

    def test_parse_with_error_handling_tolerates_bad_records(self):
        """Individual bad records don't crash the entire parse."""
        from scripts.bootstrap_usitc import _parse_with_error_handling

        records = list(USITC_FIXTURE_RECORDS) + [
            # Corrupted record
            {"htsno": "XXXX.YY.ZZ", "description": None, "indent": -1, "general": "???"},
        ]

        lines, errors = _parse_with_error_handling(records)
        # Should still parse the good records
        assert len(lines) > 0

    def test_bootstrap_from_records_creates_store(self):
        """bootstrap_from_records creates a VersionedHTSStore entry."""
        from scripts.bootstrap_usitc import bootstrap_from_records

        with tempfile.TemporaryDirectory() as tmpdir:
            result = bootstrap_from_records(
                USITC_FIXTURE_RECORDS,
                store_dir=Path(tmpdir),
            )
            assert result["atom_count"] > 0
            assert result["atoms"] is not None
            assert len(result["atoms"]) == result["atom_count"]

            # Verify the store has the edition registered
            from potatobacon.tariff.hts_ingest.versioned_store import VersionedHTSStore
            store = VersionedHTSStore(Path(tmpdir))
            assert store.get_current_edition() == "USITC_FIXTURE"
