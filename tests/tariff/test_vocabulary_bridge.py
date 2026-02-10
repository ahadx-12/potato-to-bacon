"""End-to-end tests for the Sprint A vocabulary bridge.

Proves that:
1. Facts compiled from BOMs include chapter/category tokens
2. Guard tokens from USITC descriptions include fact-compiler synonyms
3. Expanded facts can satisfy USITC-style guard clauses
4. The mutation engine uses vocabulary-aware gap computation
5. The full pipeline (compile -> expand -> Z3 match) works end-to-end
"""

from __future__ import annotations

import pytest

from potatobacon.tariff.fact_compiler import compile_facts
from potatobacon.tariff.fact_mapper import (
    can_satisfy_guard,
    compute_fact_gap,
    facts_to_guard_tokens,
    guard_tokens_to_required_facts,
)
from potatobacon.tariff.fact_vocabulary import (
    CATEGORY_TO_CHAPTERS,
    ENTAILMENTS,
    MATERIAL_CHAPTER_TOKENS,
    expand_facts,
    get_entailed_tokens,
    get_synonyms,
)
from potatobacon.tariff.hts_ingest.guard_token_gen import generate_guard_tokens
from potatobacon.tariff.product_schema import (
    MaterialBreakdown,
    ProductCategory,
    ProductSpecModel,
)


# ---------------------------------------------------------------------------
# 1. Synonym resolution
# ---------------------------------------------------------------------------
class TestSynonyms:
    def test_fastener_synonym(self):
        syns = get_synonyms("is_fastener")
        assert "product_type_fastener" in syns
        assert "is_fastener" in syns

    def test_apparel_synonym(self):
        syns = get_synonyms("product_type_apparel_textile")
        assert "product_type_apparel" in syns

    def test_unknown_token_returns_self(self):
        syns = get_synonyms("unknown_token_xyz")
        assert syns == frozenset({"unknown_token_xyz"})


# ---------------------------------------------------------------------------
# 2. Entailment resolution
# ---------------------------------------------------------------------------
class TestEntailments:
    def test_chassis_bolt_entails_fastener(self):
        entailed = get_entailed_tokens("product_type_chassis_bolt")
        assert "product_type_fastener" in entailed
        assert "is_fastener" in entailed

    def test_steel_entails_metal(self):
        entailed = get_entailed_tokens("material_steel")
        assert "material_metal" in entailed

    def test_stainless_steel_transitive(self):
        entailed = get_entailed_tokens("material_stainless_steel")
        assert "material_steel" in entailed
        assert "material_metal" in entailed

    def test_fiber_cotton_dominant_entails_cotton_and_textile(self):
        entailed = get_entailed_tokens("fiber_cotton_dominant")
        assert "fiber_cotton" in entailed
        assert "material_textile" in entailed


# ---------------------------------------------------------------------------
# 3. Fact compiler emits chapter/category tokens
# ---------------------------------------------------------------------------
class TestFactCompilerBridge:
    def test_fastener_emits_chapter_tokens(self):
        product = ProductSpecModel(
            product_category=ProductCategory.FASTENER,
            materials=[MaterialBreakdown(component="body", material="steel")],
            origin_country="CN",
        )
        facts, evidence = compile_facts(product)

        # Must emit chapter tokens for fastener
        assert facts.get("chapter_73") is True
        assert facts.get("category_steel_article") is True

        # Must emit synonym tokens
        assert facts.get("product_type_fastener") is True
        assert facts.get("is_fastener") is True

        # Must emit material-derived chapter tokens
        assert facts.get("chapter_72") is True  # steel -> chapter 72/73

    def test_electronics_emits_chapter_tokens(self):
        product = ProductSpecModel(
            product_category=ProductCategory.ELECTRONICS,
            materials=[MaterialBreakdown(component="housing", material="plastic")],
            origin_country="CN",
        )
        facts, _ = compile_facts(product)

        assert facts.get("chapter_85") is True
        assert facts.get("category_electronics") is True
        assert facts.get("chapter_39") is True  # plastic -> chapter 39

    def test_apparel_emits_chapter_and_synonym(self):
        product = ProductSpecModel(
            product_category=ProductCategory.APPAREL_TEXTILE,
            materials=[MaterialBreakdown(component="body", material="textile")],
            origin_country="CN",
        )
        facts, _ = compile_facts(product)

        assert facts.get("chapter_61") is True
        assert facts.get("chapter_62") is True
        assert facts.get("product_type_apparel") is True
        assert facts.get("product_type_apparel_textile") is True


# ---------------------------------------------------------------------------
# 4. Guard token gen emits fact-compiler-compatible tokens
# ---------------------------------------------------------------------------
class TestGuardTokenGenBridge:
    def test_fastener_description_emits_is_fastener(self):
        tokens = generate_guard_tokens(
            htsno="7318.15.2061",
            description="Steel bolts for vehicle frames",
            parent_descriptions=["Articles of iron or steel"],
            indent=2,
            chapter="73",
        )
        assert "product_type_fastener" in tokens
        assert "is_fastener" in tokens
        assert "product_type_chassis_bolt" in tokens
        assert "material_steel" in tokens

    def test_apparel_description_emits_apparel_textile(self):
        tokens = generate_guard_tokens(
            htsno="6201.92.2051",
            description="Woven garment of cotton",
            parent_descriptions=[],
            indent=2,
            chapter="62",
        )
        assert "product_type_apparel" in tokens
        assert "product_type_apparel_textile" in tokens
        assert "textile_woven" in tokens

    def test_coated_description_emits_coating_or_lamination(self):
        tokens = generate_guard_tokens(
            htsno="6210.50.3000",
            description="Coated textile garment",
            parent_descriptions=[],
            indent=2,
            chapter="62",
        )
        assert "has_coating" in tokens
        assert "has_coating_or_lamination" in tokens


# ---------------------------------------------------------------------------
# 5. expand_facts bridges the gap
# ---------------------------------------------------------------------------
class TestExpandFacts:
    def test_expands_fastener_synonyms(self):
        facts = {"is_fastener": True, "material_steel": True, "product_category": "fastener"}
        expanded = expand_facts(facts)

        assert expanded["product_type_fastener"] is True
        assert expanded["material_metal"] is True
        assert expanded["chapter_73"] is True

    def test_expands_chapter_from_category(self):
        facts = {"product_category": "electronics", "product_type_electronics": True}
        expanded = expand_facts(facts)
        assert expanded.get("chapter_85") is True
        assert expanded.get("category_electronics") is True

    def test_expands_material_chapters(self):
        facts = {"material_aluminum": True}
        expanded = expand_facts(facts)
        assert expanded.get("chapter_76") is True
        assert expanded.get("category_aluminum") is True
        assert expanded.get("material_metal") is True


# ---------------------------------------------------------------------------
# 6. Fact mapper: gap computation respects vocabulary
# ---------------------------------------------------------------------------
class TestFactMapper:
    def test_no_gap_when_synonym_present(self):
        """is_fastener should satisfy a guard requiring product_type_fastener."""
        facts = {"is_fastener": True, "material_steel": True}
        guard = ["product_type_fastener", "material_steel"]
        gap = compute_fact_gap(facts, guard)
        assert gap == {}, f"Expected no gap but got: {gap}"

    def test_no_gap_when_entailment_satisfied(self):
        """product_type_chassis_bolt entails product_type_fastener."""
        facts = {"product_type_chassis_bolt": True, "material_steel": True}
        guard = ["product_type_fastener", "material_steel"]
        gap = compute_fact_gap(facts, guard)
        assert gap == {}, f"Expected no gap but got: {gap}"

    def test_gap_detected_when_missing(self):
        facts = {"material_steel": True}
        guard = ["product_type_fastener", "material_steel"]
        gap = compute_fact_gap(facts, guard)
        assert "product_type_fastener" in gap

    def test_can_satisfy_guard_with_synonyms(self):
        facts = {"is_fastener": True, "material_steel": True}
        assert can_satisfy_guard(facts, ["product_type_fastener", "material_steel"])

    def test_guard_tokens_to_required(self):
        required = guard_tokens_to_required_facts(["material_steel", "\u00acmaterial_aluminum"])
        assert required == {"material_steel": True, "material_aluminum": False}


# ---------------------------------------------------------------------------
# 7. End-to-end: compiled facts match USITC-style guard tokens
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_steel_bolt_matches_usitc_fastener_guard(self):
        """A steel bolt BOM should satisfy a USITC atom with
        guard = ['chapter_73', 'material_steel', 'product_type_fastener']."""
        product = ProductSpecModel(
            product_category=ProductCategory.FASTENER,
            materials=[MaterialBreakdown(component="body", material="steel")],
            origin_country="CN",
            import_country="US",
        )
        facts, _ = compile_facts(product)
        expanded = expand_facts(facts)

        # Simulate a USITC-sourced atom guard
        usitc_guard = ["chapter_73", "material_steel", "product_type_fastener"]
        assert can_satisfy_guard(expanded, usitc_guard), (
            f"Steel bolt facts should satisfy USITC guard {usitc_guard}. "
            f"Missing: {compute_fact_gap(expanded, usitc_guard)}"
        )

    def test_electronics_enclosure_matches_usitc_guard(self):
        """An electronics enclosure BOM should match USITC chapter 85 atoms."""
        product = ProductSpecModel(
            product_category=ProductCategory.ELECTRONICS,
            materials=[MaterialBreakdown(component="housing", material="plastic")],
            origin_country="CN",
            import_country="US",
            is_enclosure_or_housing=True,
        )
        facts, _ = compile_facts(product)
        expanded = expand_facts(facts)

        usitc_guard = ["chapter_85", "category_electronics", "material_plastic"]
        assert can_satisfy_guard(expanded, usitc_guard)

    def test_woven_cotton_apparel_matches_usitc_guard(self):
        """Woven cotton apparel should match USITC chapter 62 atoms."""
        product = ProductSpecModel(
            product_category=ProductCategory.APPAREL_TEXTILE,
            materials=[MaterialBreakdown(component="body", material="textile")],
            origin_country="CN",
            import_country="US",
            is_woven=True,
            fiber_cotton_pct=85.0,
        )
        facts, _ = compile_facts(product)
        expanded = expand_facts(facts)

        usitc_guard = ["chapter_62", "product_type_apparel", "textile_woven", "fiber_cotton_dominant"]
        assert can_satisfy_guard(expanded, usitc_guard), (
            f"Woven cotton apparel should satisfy USITC guard {usitc_guard}. "
            f"Missing: {compute_fact_gap(expanded, usitc_guard)}"
        )

    def test_facts_to_guard_tokens_covers_all(self):
        """facts_to_guard_tokens should return all satisfiable tokens."""
        product = ProductSpecModel(
            product_category=ProductCategory.FASTENER,
            materials=[MaterialBreakdown(component="body", material="steel")],
            origin_country="CN",
        )
        facts, _ = compile_facts(product)
        tokens = facts_to_guard_tokens(facts)

        # Should include direct facts, synonyms, entailments, and chapters
        assert "material_steel" in tokens
        assert "product_type_fastener" in tokens
        assert "is_fastener" in tokens
        assert "material_metal" in tokens
        assert "chapter_73" in tokens
        assert "category_steel_article" in tokens

    def test_guard_gen_and_compiler_share_vocabulary(self):
        """Guard tokens generated from HTS description should overlap
        significantly with tokens from compiled facts for the same product."""
        # Simulate guard tokens from a USITC steel bolt entry
        guard_tokens = set(generate_guard_tokens(
            htsno="7318.15.2061",
            description="Steel bolts for vehicle frames",
            parent_descriptions=["Articles of iron or steel"],
            indent=2,
            chapter="73",
        ))

        # Compile facts for a steel bolt BOM
        product = ProductSpecModel(
            product_category=ProductCategory.FASTENER,
            materials=[MaterialBreakdown(component="body", material="steel")],
            origin_country="CN",
        )
        facts, _ = compile_facts(product)
        fact_tokens = facts_to_guard_tokens(facts)

        # There must be significant overlap
        overlap = guard_tokens & fact_tokens
        assert len(overlap) >= 3, (
            f"Expected at least 3 overlapping tokens, got {len(overlap)}: {overlap}\n"
            f"Guard tokens: {guard_tokens}\n"
            f"Fact tokens: {fact_tokens}"
        )
        # Specifically, these must overlap
        assert "material_steel" in overlap
        assert "product_type_fastener" in overlap
