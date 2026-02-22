"""Tests for the AD/CVD scope analysis engine.

A tariff engineer doesn't just check if an HTS code prefix matches an AD/CVD
order.  They analyze whether the PRODUCT is within scope of the order's legal
definition.  These tests verify that the scope engine correctly:

  - Identifies products that are clearly within scope (SCOPE_CONFIRMED)
  - Flags products where scope is uncertain (HTS_PREFIX_ONLY, SCOPE_LIKELY)
  - Detects products likely outside scope (SCOPE_EXCLUSION_POSSIBLE, SCOPE_EXCLUDED)
  - Provides legally grounded guidance and recommended actions
"""

from __future__ import annotations

import pytest
from typing import List


class TestScopeConfidence:
    """Test scope confidence assignment for different product profiles."""

    def test_carbon_steel_pipe_within_scope(self):
        """Carbon steel welded pipe should be SCOPE_CONFIRMED against pipe orders."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        result = analyze_scope(
            product_description="Carbon steel welded pipe for industrial plumbing, schedule 40",
            order_id="A-570-001",
            order_type="AD",
            order_scope_text="Carbon steel welded pipe and tube from China",
            order_scope_keywords=["pipe", "carbon steel", "welded"],
            hts_prefix_matched=True,
        )

        assert result.scope_confidence in (
            ScopeConfidence.SCOPE_CONFIRMED,
            ScopeConfidence.SCOPE_LIKELY,
        ), f"Expected CONFIRMED or LIKELY, got: {result.scope_confidence}"
        assert result.inclusion_score > result.exclusion_score
        assert result.requires_scope_ruling is False or result.scope_confidence == ScopeConfidence.SCOPE_CONFIRMED
        assert result.recommended_actions

    def test_stainless_steel_excluded_from_carbon_steel_order(self):
        """Stainless steel is frequently excluded from carbon steel AD/CVD orders."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        result = analyze_scope(
            product_description="Stainless steel 316L seamless tube for food processing",
            order_id="A-570-001",
            order_type="AD",
            order_scope_text="Carbon steel welded and seamless pipe from China",
            order_scope_keywords=["pipe", "carbon steel", "welded", "seamless"],
            hts_prefix_matched=True,
        )

        assert result.scope_confidence in (
            ScopeConfidence.SCOPE_EXCLUDED,
            ScopeConfidence.SCOPE_EXCLUSION_POSSIBLE,
        ), f"Stainless should be excluded from carbon steel order, got: {result.scope_confidence}"
        assert result.exclusion_score > 0
        assert result.requires_scope_ruling is True

    def test_hts_prefix_only_returns_correct_confidence(self):
        """When only HTS prefix matches and no keyword hits, return HTS_PREFIX_ONLY."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        # Generic product with no strong scope signals
        result = analyze_scope(
            product_description="Industrial component for machinery",
            order_id="A-570-999",
            order_type="AD",
            order_scope_text="Steel widgets from Country X",
            order_scope_keywords=[],
            hts_prefix_matched=True,
        )

        assert result.scope_confidence == ScopeConfidence.HTS_PREFIX_ONLY
        assert result.requires_scope_ruling is True

    def test_aluminum_extrusion_within_scope(self):
        """Extruded aluminum window frame should be within extrusion order scope."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        result = analyze_scope(
            product_description="Extruded aluminum window frame with thermal break",
            order_id="A-570-extrusion",
            order_type="AD",
            order_scope_text="Aluminum extrusions from China",
            order_scope_keywords=["aluminum", "extrusion", "extruded"],
            hts_prefix_matched=True,
        )

        assert result.scope_confidence in (
            ScopeConfidence.SCOPE_CONFIRMED,
            ScopeConfidence.SCOPE_LIKELY,
        )
        assert result.inclusion_score > 0

    def test_wooden_furniture_within_scope(self):
        """Upholstered wood chair should be within furniture order scope."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        result = analyze_scope(
            product_description="Upholstered dining chair with solid wood frame",
            order_id="A-570-furniture",
            order_type="AD",
            order_scope_text="Wooden bedroom furniture from China",
            order_scope_keywords=["wood", "furniture", "upholstered"],
            hts_prefix_matched=True,
        )

        assert result.scope_confidence != ScopeConfidence.SCOPE_EXCLUDED

    def test_medical_grade_product_outside_scope(self):
        """Medical grade materials are often excluded from commercial orders."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        result = analyze_scope(
            product_description="Medical grade stainless steel 316L surgical tubing",
            order_id="A-570-001",
            order_type="AD",
            order_scope_text="Carbon steel pipe from China",
            order_scope_keywords=["pipe", "carbon steel"],
            hts_prefix_matched=True,
        )

        # Both stainless + medical grade should be strong exclusion signals
        assert result.exclusion_score >= 1.0
        assert result.scope_confidence in (
            ScopeConfidence.SCOPE_EXCLUDED,
            ScopeConfidence.SCOPE_EXCLUSION_POSSIBLE,
        )


class TestScopeGuidance:
    """Test that scope guidance is actionable and legally grounded."""

    def test_scope_confirmed_has_no_ruling_required(self):
        """If scope is confirmed, we don't ask for a ruling — we know the answer."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        result = analyze_scope(
            product_description="Carbon steel welded pipe 2 inch schedule 40",
            order_id="A-570-001",
            order_type="AD",
            order_scope_text="Carbon steel welded pipe",
            order_scope_keywords=["carbon steel", "pipe", "welded"],
            hts_prefix_matched=True,
        )

        if result.scope_confidence == ScopeConfidence.SCOPE_CONFIRMED:
            assert result.requires_scope_ruling is False
            assert any("confirm" in a.lower() for a in result.recommended_actions)

    def test_exclusion_possible_requires_ruling(self):
        """Scope exclusion possibility always needs a formal ruling."""
        from potatobacon.tariff.adcvd_scope import analyze_scope, ScopeConfidence

        result = analyze_scope(
            product_description="Stainless steel 304 mechanical tubing",
            order_id="A-570-001",
            order_type="AD",
            order_scope_text="Carbon steel standard pipe",
            order_scope_keywords=["carbon steel", "pipe"],
            hts_prefix_matched=True,
        )

        if result.scope_confidence in (
            ScopeConfidence.SCOPE_EXCLUSION_POSSIBLE,
            ScopeConfidence.SCOPE_EXCLUDED,
        ):
            assert result.requires_scope_ruling is True
            assert result.legal_notes

    def test_all_results_have_determination_text(self):
        """Every scope result must have a plain-English determination."""
        from potatobacon.tariff.adcvd_scope import analyze_scope

        for description in [
            "Carbon steel pipe",
            "Stainless steel tube",
            "Generic industrial component",
        ]:
            result = analyze_scope(
                product_description=description,
                order_id="A-TEST",
                order_type="AD",
                order_scope_text="Test scope",
                order_scope_keywords=[],
                hts_prefix_matched=True,
            )
            assert result.scope_determination, f"Missing determination for: {description}"
            assert result.recommended_actions

    def test_legal_citations_present(self):
        """All scope results must reference relevant legal authority."""
        from potatobacon.tariff.adcvd_scope import analyze_scope

        result = analyze_scope(
            product_description="Steel pipe from China",
            order_id="A-570-001",
            order_type="AD",
            order_scope_text="Steel pipe scope",
            order_scope_keywords=["steel", "pipe"],
            hts_prefix_matched=True,
        )
        assert result.legal_notes, "Must include legal citations"
        assert any("351" in n or "731" in n or "570" in n for n in result.legal_notes)


class TestBatchScopeAnalysis:
    """Test batch scope analysis across multiple orders."""

    def test_batch_analysis_returns_only_relevant_orders(self):
        """Batch analysis should skip orders with no keyword or HTS match."""
        from potatobacon.tariff.adcvd_scope import analyze_scope_for_orders

        orders = [
            {
                "order_id": "A-570-pipe",
                "type": "AD",
                "product_description": "Steel pipe",
                "scope_keywords": ["steel", "pipe"],
            },
            {
                "order_id": "A-570-furniture",
                "type": "AD",
                "product_description": "Wooden furniture",
                "scope_keywords": ["furniture", "wood"],
            },
        ]

        # Pipe product — should only match pipe order
        results = analyze_scope_for_orders(
            product_description="Carbon steel welded pipe for plumbing",
            orders=orders,
            hts_matched_order_ids={"A-570-pipe"},
        )
        order_ids = {r.order_id for r in results}
        assert "A-570-pipe" in order_ids
        # Furniture order should not match (no keyword hit, no HTS match)
        assert "A-570-furniture" not in order_ids

    def test_batch_with_no_matches_returns_empty(self):
        """A product with no HTS or keyword match should return no results."""
        from potatobacon.tariff.adcvd_scope import analyze_scope_for_orders

        orders = [
            {
                "order_id": "A-570-shoe",
                "type": "AD",
                "product_description": "Athletic footwear",
                "scope_keywords": ["shoe", "footwear"],
            },
        ]

        results = analyze_scope_for_orders(
            product_description="Laptop computer",
            orders=orders,
            hts_matched_order_ids=set(),
        )
        assert results == []


class TestScopeConfidenceMapping:
    """Test the confidence → legacy adcvd_confidence mapping."""

    def test_scope_confirmed_maps_to_high(self):
        from potatobacon.tariff.adcvd_scope import (
            ScopeConfidence,
            scope_confidence_to_adcvd_confidence,
        )
        assert scope_confidence_to_adcvd_confidence(ScopeConfidence.SCOPE_CONFIRMED) == "high"

    def test_hts_prefix_only_maps_to_low(self):
        from potatobacon.tariff.adcvd_scope import (
            ScopeConfidence,
            scope_confidence_to_adcvd_confidence,
        )
        assert scope_confidence_to_adcvd_confidence(ScopeConfidence.HTS_PREFIX_ONLY) == "low"

    def test_scope_excluded_maps_to_none(self):
        from potatobacon.tariff.adcvd_scope import (
            ScopeConfidence,
            scope_confidence_to_adcvd_confidence,
        )
        assert scope_confidence_to_adcvd_confidence(ScopeConfidence.SCOPE_EXCLUDED) == "none"
