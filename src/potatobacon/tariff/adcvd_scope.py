"""AD/CVD scope analysis engine.

AD/CVD orders are defined by a "scope" — the legal description of which products
are covered.  A product with the same HTS code as an AD/CVD order may or may
not be within scope.  Scope is determined by:

  1. The legal scope language in the order (Federal Register notice)
  2. The product's physical characteristics (dimensions, composition, end use)
  3. Commerce Department scope rulings (precedent for how scope is interpreted)

A tariff CALCULATOR checks: "does the HTS code start with a covered prefix?" If
yes, it reports AD/CVD exposure.  This is wrong in two directions:
  - False positives: a product sharing an HTS code but clearly outside scope
  - False negatives: a product within scope but declared under a different code

A tariff ENGINEER runs scope analysis:
  1. Match the product against the scope language (keyword + characteristic match)
  2. Flag indicators that the product is WITHIN scope (confirm exposure)
  3. Flag indicators that the product is OUTSIDE scope (exclusion from risk)
  4. Surface scope rulings that support the analysis
  5. Tell the company whether to seek a scope ruling from Commerce

This module implements that analysis.  It supplements the HTS prefix matching
in adcvd_registry.py with a text-based scope analysis layer.

Scope confidence levels:
  SCOPE_CONFIRMED     : Multiple scope indicators match — high confidence exposure
  SCOPE_LIKELY        : Several indicators match — recommend scope review
  HTS_PREFIX_ONLY     : Only HTS prefix matched — uncertainty is high
  SCOPE_EXCLUSION_POSSIBLE : Indicators suggest product may be outside scope
  SCOPE_EXCLUDED      : Strong indicators product is outside scope (document it)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Confidence enum
# ---------------------------------------------------------------------------

class ScopeConfidence(str, Enum):
    """How confidently a product is within scope of an AD/CVD order."""

    SCOPE_CONFIRMED           = "scope_confirmed"
    SCOPE_LIKELY              = "scope_likely"
    HTS_PREFIX_ONLY           = "hts_prefix_only"
    SCOPE_EXCLUSION_POSSIBLE  = "scope_exclusion_possible"
    SCOPE_EXCLUDED            = "scope_excluded"


# ---------------------------------------------------------------------------
# Scope feature models
# ---------------------------------------------------------------------------

@dataclass
class ScopeFeature:
    """A single feature that contributes to scope determination."""

    feature_type: str       # "inclusion" or "exclusion"
    matched_term: str       # The term from scope language that matched
    source: str             # "scope_text", "scope_keywords", "characteristic"
    weight: float           # How strongly this feature signals scope match (0-1)
    note: str               # Human-readable explanation


@dataclass
class ADCVDScopeResult:
    """Scope analysis result for a single product against a single AD/CVD order."""

    order_id: str
    order_type: str                    # "AD" or "CVD"
    product_description: str           # Scope language from the order
    scope_confidence: ScopeConfidence

    # Feature matches
    inclusion_features: List[ScopeFeature] = field(default_factory=list)
    exclusion_features: List[ScopeFeature] = field(default_factory=list)
    inclusion_score: float = 0.0
    exclusion_score: float = 0.0

    # Guidance
    scope_determination: str = ""      # Plain-English summary
    recommended_actions: List[str] = field(default_factory=list)
    legal_notes: List[str] = field(default_factory=list)
    requires_scope_ruling: bool = False


# ---------------------------------------------------------------------------
# Scope term databases
# ---------------------------------------------------------------------------

# Scope EXCLUSION indicators: if the product matches these, it may be outside scope.
# These are standard Commerce Department exclusion language patterns.
_SCOPE_EXCLUSION_INDICATORS: List[Tuple[str, str, float]] = [
    # (pattern, note, weight)

    # Stainless steel excluded from carbon steel orders
    (r"\bstainless\b", "Stainless steel is frequently excluded from carbon/mild steel orders", 0.7),
    (r"\bstainless\s+steel\b", "Stainless steel specifically excluded from carbon steel AD/CVD orders", 0.85),

    # Mechanical tubing vs standard pipe (often excluded)
    (r"\bmechanical\s+tub", "Mechanical tubing often excluded from standard pipe/tube orders", 0.6),

    # Specialty coatings / grades excluded from standard product orders
    (r"\bgalvanized\b", "Galvanized products sometimes excluded from bare steel orders", 0.45),
    (r"\bepoxy.{0,20}coat", "Specialty-coated products may be outside bare steel scope", 0.4),

    # Custom / bespoke products excluded from commodity orders
    (r"\bcustom.{0,20}fabricat", "Custom-fabricated products may be outside commodity scope", 0.5),

    # Finished articles (assembled into products) excluded from component orders
    (r"\bfully\s+assembl", "Fully assembled articles may be outside component/part scope", 0.55),
    (r"\bfinished\s+goods?\b", "Finished goods sometimes excluded from intermediate scope", 0.45),

    # Medical/pharmaceutical exclusions
    (r"\bmedical\s+grade\b", "Medical-grade materials often excluded from commercial-grade orders", 0.6),
    (r"\bpharmaceutical\b", "Pharmaceutical materials often have separate scope", 0.5),

    # Aerospace / defense exclusions
    (r"\baerospace\b", "Aerospace-grade materials often excluded from commercial scope", 0.55),
    (r"\bmilitary\s+spec\b", "Military-specification materials often excluded", 0.6),
]

# Scope INCLUSION indicators: if the product matches these, it is likely within scope.
_SCOPE_INCLUSION_INDICATORS: List[Tuple[str, str, float]] = [
    # Material match (general)
    (r"\bcarbon\s+steel\b", "Carbon steel material matches typical steel order scope", 0.65),
    (r"\bmild\s+steel\b", "Mild steel matches standard steel order scope", 0.6),
    (r"\bsteel\s+pipe\b", "Steel pipe directly within pipe order scope", 0.75),
    (r"\bsteel\s+tube\b", "Steel tube within typical tube order scope", 0.7),
    (r"\bwelded\s+pipe\b", "Welded pipe within pipe order scope", 0.75),
    (r"\bseamless\s+pipe\b", "Seamless pipe within pipe order scope", 0.7),

    # Aluminum scope indicators
    (r"\bextruded\s+aluminum\b", "Extruded aluminum within aluminum extrusion order scope", 0.8),
    (r"\baluminum\s+extrusion\b", "Aluminum extrusion within scope", 0.8),

    # Fastener scope indicators
    (r"\bhex\s+head\s+bolt", "Hex head bolts within typical fastener order scope", 0.75),
    (r"\bsteel\s+bolt\b", "Steel bolts within fastener order scope", 0.7),
    (r"\bsteel\s+screw\b", "Steel screws within fastener order scope", 0.7),
    (r"\bsteel\s+nut\b", "Steel nuts within fastener order scope", 0.65),

    # Furniture scope
    (r"\bwooden\s+furniture\b", "Wooden furniture within furniture order scope", 0.75),
    (r"\bwood\s+furniture\b", "Wood furniture within scope", 0.75),
    (r"\bupholstered\b.*\bwood\b|\bwood\b.*\bupholstered\b",
     "Upholstered wood furniture within residential furniture scope", 0.7),

    # Tire scope
    (r"\bpneumatic\s+tire\b", "Pneumatic tires within tire order scope", 0.75),
    (r"\bpassenger\s+vehicle\s+tire\b", "Passenger vehicle tires within scope", 0.8),

    # Chemical / circumvention
    (r"\bglyphosate\b", "Glyphosate within glyphosate order scope", 0.9),
    (r"\bcitric\s+acid\b", "Citric acid within citric acid order scope", 0.85),
    (r"\bsodium\s+glutamate\b", "Monosodium glutamate within MSG scope", 0.9),
]

# Characteristic-to-scope mappings: product facts that are strong scope indicators.
_CHARACTERISTIC_SCOPE_SIGNALS: Dict[str, Tuple[str, float]] = {
    # fact_key → (description, weight)
    "steel_carbon":         ("Carbon steel product — within steel order scope", 0.7),
    "steel_stainless":      ("Stainless steel — likely excluded from carbon steel orders", 0.7),
    "aluminum_extruded":    ("Extruded aluminum — within aluminum extrusion order scope", 0.8),
    "wood_frame":           ("Wood-frame product — within wood furniture order scope", 0.65),
    "origin_CN":            ("Chinese origin — must confirm within each active CN order scope", 0.5),
    "weld_type_welded":     ("Welded product — within welded pipe/tube order scope", 0.7),
    "weld_type_seamless":   ("Seamless product — within seamless pipe/tube order scope", 0.7),
    "surface_galvanized":   ("Galvanized surface — may be excluded from bare steel orders", 0.5),
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _tokenize(text: str) -> Set[str]:
    """Lowercase, normalize, and split text into tokens."""
    lowered = _NON_ALPHA.sub(" ", text.lower())
    return {t for t in lowered.split() if len(t) >= 2}


# ---------------------------------------------------------------------------
# Core scope analysis
# ---------------------------------------------------------------------------

def analyze_scope(
    *,
    product_description: str,
    order_id: str,
    order_type: str,
    order_scope_text: str,
    order_scope_keywords: List[str],
    product_characteristics: Optional[Dict[str, bool]] = None,
    hts_prefix_matched: bool = False,
) -> ADCVDScopeResult:
    """Analyze whether a product is within the scope of an AD/CVD order.

    Args:
        product_description: Free-text product description.
        order_id: AD/CVD case number.
        order_type: "AD" or "CVD".
        order_scope_text: Full legal scope text from the order (Federal Register).
        order_scope_keywords: Key terms extracted from the scope definition.
        product_characteristics: Optional dict of product facts
            (e.g. {"steel_carbon": True, "aluminum_extruded": False}).
        hts_prefix_matched: Whether the product's HTS code matches the order's
            HTS prefix list (the simpler lookup layer).

    Returns:
        ADCVDScopeResult with confidence level and feature breakdown.
    """
    product_desc = product_description.lower()
    scope_text = order_scope_text.lower() if order_scope_text else ""
    characteristics = product_characteristics or {}

    inclusion_features: List[ScopeFeature] = []
    exclusion_features: List[ScopeFeature] = []

    # ------------------------------------------------------------------
    # 1. Scope keyword match (order's curated keyword list)
    # ------------------------------------------------------------------
    product_tokens = _tokenize(product_description)
    for kw in order_scope_keywords:
        kw_tokens = _tokenize(kw)
        if kw_tokens and kw_tokens.issubset(product_tokens):
            inclusion_features.append(ScopeFeature(
                feature_type="inclusion",
                matched_term=kw,
                source="scope_keywords",
                weight=0.6,
                note=f"Product description matches scope keyword '{kw}'",
            ))

    # ------------------------------------------------------------------
    # 2. Scope text pattern matching (inclusion indicators)
    # ------------------------------------------------------------------
    for pattern, note, weight in _SCOPE_INCLUSION_INDICATORS:
        if re.search(pattern, product_desc, re.IGNORECASE):
            inclusion_features.append(ScopeFeature(
                feature_type="inclusion",
                matched_term=pattern,
                source="scope_text",
                weight=weight,
                note=note,
            ))

    # ------------------------------------------------------------------
    # 3. Scope exclusion pattern matching
    # ------------------------------------------------------------------
    for pattern, note, weight in _SCOPE_EXCLUSION_INDICATORS:
        if re.search(pattern, product_desc, re.IGNORECASE):
            exclusion_features.append(ScopeFeature(
                feature_type="exclusion",
                matched_term=pattern,
                source="scope_text",
                weight=weight,
                note=note,
            ))

    # ------------------------------------------------------------------
    # 4. Product characteristic signals
    # ------------------------------------------------------------------
    for fact_key, value in characteristics.items():
        if fact_key in _CHARACTERISTIC_SCOPE_SIGNALS:
            sig_note, sig_weight = _CHARACTERISTIC_SCOPE_SIGNALS[fact_key]
            if value:
                # Determine if this characteristic is an inclusion or exclusion signal
                is_exclusion = ("excluded" in sig_note.lower() or "outside" in sig_note.lower()
                                or "may be excluded" in sig_note.lower())
                if is_exclusion:
                    exclusion_features.append(ScopeFeature(
                        feature_type="exclusion",
                        matched_term=fact_key,
                        source="characteristic",
                        weight=sig_weight,
                        note=sig_note,
                    ))
                else:
                    inclusion_features.append(ScopeFeature(
                        feature_type="inclusion",
                        matched_term=fact_key,
                        source="characteristic",
                        weight=sig_weight,
                        note=sig_note,
                    ))

    # ------------------------------------------------------------------
    # 5. Compute scores
    # ------------------------------------------------------------------
    inclusion_score = sum(f.weight for f in inclusion_features)
    exclusion_score = sum(f.weight for f in exclusion_features)
    net_score = inclusion_score - exclusion_score

    # ------------------------------------------------------------------
    # 6. Determine confidence level
    # ------------------------------------------------------------------
    if exclusion_score > 1.2 and exclusion_score > inclusion_score:
        confidence = ScopeConfidence.SCOPE_EXCLUDED
    elif exclusion_score > 0.5 and exclusion_score >= inclusion_score * 0.75:
        confidence = ScopeConfidence.SCOPE_EXCLUSION_POSSIBLE
    elif inclusion_score >= 1.5:
        confidence = ScopeConfidence.SCOPE_CONFIRMED
    elif inclusion_score >= 0.6:
        # Multiple scope signals beyond just the HTS prefix — likely in scope
        confidence = ScopeConfidence.SCOPE_LIKELY
    elif hts_prefix_matched:
        # Only the HTS code matched — scope is uncertain, need product-level analysis
        confidence = ScopeConfidence.HTS_PREFIX_ONLY
    else:
        confidence = ScopeConfidence.HTS_PREFIX_ONLY

    # ------------------------------------------------------------------
    # 7. Build determination text and actions
    # ------------------------------------------------------------------
    determination, actions, legal_notes, requires_ruling = _build_scope_guidance(
        confidence=confidence,
        order_id=order_id,
        order_type=order_type,
        order_scope_text=order_scope_text,
        inclusion_features=inclusion_features,
        exclusion_features=exclusion_features,
        hts_prefix_matched=hts_prefix_matched,
    )

    return ADCVDScopeResult(
        order_id=order_id,
        order_type=order_type,
        product_description=order_scope_text[:200] if order_scope_text else "",
        scope_confidence=confidence,
        inclusion_features=inclusion_features,
        exclusion_features=exclusion_features,
        inclusion_score=round(inclusion_score, 3),
        exclusion_score=round(exclusion_score, 3),
        scope_determination=determination,
        recommended_actions=actions,
        legal_notes=legal_notes,
        requires_scope_ruling=requires_ruling,
    )


def _build_scope_guidance(
    *,
    confidence: ScopeConfidence,
    order_id: str,
    order_type: str,
    order_scope_text: str,
    inclusion_features: List[ScopeFeature],
    exclusion_features: List[ScopeFeature],
    hts_prefix_matched: bool,
) -> Tuple[str, List[str], List[str], bool]:
    """Build plain-English scope guidance and recommended actions."""

    order_label = f"{order_type} order {order_id}"

    if confidence == ScopeConfidence.SCOPE_CONFIRMED:
        determination = (
            f"SCOPE CONFIRMED: Product characteristics strongly indicate this product is "
            f"within scope of {order_label}. "
            f"AD/CVD liability is likely. Verify scope and pay applicable duties."
        )
        actions = [
            f"Confirm scope with your customs broker — {order_label} scope text matches product characteristics",
            f"Ensure {order_type} duties are being paid on all entries",
            "Maintain documentation of scope determination for CBP audit purposes",
        ]
        legal_notes = [
            f"Tariff Act of 1930 §731 ({order_type})",
            f"Active order: {order_id}",
            "19 CFR Part 351 (AD/CVD regulations)",
        ]
        return determination, actions, legal_notes, False

    elif confidence == ScopeConfidence.SCOPE_LIKELY:
        determination = (
            f"SCOPE LIKELY: HTS code and/or product description suggest this product "
            f"is likely within scope of {order_label}. "
            "Professional scope review required before confirming."
        )
        actions = [
            f"Engage licensed customs broker to review scope of {order_label}",
            "Request a scope ruling from Commerce Department if product scope is unclear",
            "Review Federal Register notice for the order scope language",
            "Document scope determination and retain for CBP audit",
        ]
        legal_notes = [
            f"Active order: {order_id}",
            "19 CFR §351.225 — Scope rulings",
        ]
        return determination, actions, legal_notes, True

    elif confidence == ScopeConfidence.HTS_PREFIX_ONLY:
        determination = (
            f"HTS PREFIX MATCH ONLY: The product's HTS code matches {order_label}'s "
            "coverage list, but product characteristics are ambiguous. "
            "Scope cannot be confirmed or excluded from HTS code alone."
        )
        actions = [
            f"Review scope language of {order_label} against your product specifications",
            "Engage customs broker to confirm whether your product is within scope",
            "Consider a scope ruling if your product sits at the boundary of scope",
        ]
        legal_notes = [
            f"Active order: {order_id}",
            "19 CFR §351.225 — Scope rulings",
            "Note: HTS code is not determinative for scope — product characteristics control",
        ]
        return determination, actions, legal_notes, True

    elif confidence == ScopeConfidence.SCOPE_EXCLUSION_POSSIBLE:
        top_excl = exclusion_features[0].note if exclusion_features else ""
        determination = (
            f"POSSIBLE SCOPE EXCLUSION: Product may be outside scope of {order_label}. "
            f"Basis: {top_excl}. "
            "Confirm exclusion before ceasing to pay duties."
        )
        actions = [
            "Document product characteristics that support scope exclusion",
            "Request a scope ruling from Commerce Department to formally confirm exclusion",
            "Do not stop paying duties until scope exclusion is formally confirmed",
        ]
        legal_notes = [
            f"Active order: {order_id}",
            "19 CFR §351.225(k) — Scope exclusion methodology",
        ]
        return determination, actions, legal_notes, True

    else:  # SCOPE_EXCLUDED
        top_excl = exclusion_features[0].note if exclusion_features else ""
        determination = (
            f"LIKELY OUTSIDE SCOPE: Strong indicators that product is NOT within scope "
            f"of {order_label}. "
            f"Basis: {top_excl}. "
            "Obtain a scope ruling to formally document this position."
        )
        actions = [
            "Request a scope ruling from Commerce Department to formally confirm exclusion",
            "Document product specifications (material, dimensions, end use) to support ruling",
            "Consult with trade attorney before relying on scope exclusion",
        ]
        legal_notes = [
            f"Active order: {order_id}",
            "19 CFR §351.225 — Scope rulings",
            "Scope exclusion must be formally documented to be defensible at CBP audit",
        ]
        return determination, actions, legal_notes, True


# ---------------------------------------------------------------------------
# Batch scope analysis: analyze a product against multiple orders
# ---------------------------------------------------------------------------

def analyze_scope_for_orders(
    *,
    product_description: str,
    orders: List[Dict],
    product_characteristics: Optional[Dict[str, bool]] = None,
    hts_matched_order_ids: Optional[Set[str]] = None,
) -> List[ADCVDScopeResult]:
    """Analyze product scope against multiple AD/CVD orders.

    Args:
        product_description: Free-text product description.
        orders: List of order dicts from adcvd_orders_full.json.
        product_characteristics: Optional fact dict for characteristic matching.
        hts_matched_order_ids: Set of order IDs that matched by HTS prefix
            (from adcvd_registry.py). Used to set hts_prefix_matched.

    Returns:
        List of ADCVDScopeResult, one per order analyzed.
        Only returns results for orders where hts_prefix_matched=True or
        where scope keywords produce a match.
    """
    hts_matched = hts_matched_order_ids or set()
    results: List[ADCVDScopeResult] = []

    for order in orders:
        order_id = order.get("order_id", "")
        order_type = order.get("type", order.get("order_type", "AD"))
        scope_keywords = list(order.get("scope_keywords", []))
        scope_text = order.get("scope_text", "")

        # Only analyze orders that matched by HTS prefix or have scope keywords that match
        is_hts_matched = order_id in hts_matched
        if not is_hts_matched and not scope_keywords:
            continue

        # Check for any keyword hits before building full result (performance)
        product_tokens = _tokenize(product_description)
        has_keyword_hit = any(
            _tokenize(kw).issubset(product_tokens)
            for kw in scope_keywords
            if kw
        )

        if not is_hts_matched and not has_keyword_hit:
            continue

        result = analyze_scope(
            product_description=product_description,
            order_id=order_id,
            order_type=order_type,
            order_scope_text=scope_text or order.get("product_description", ""),
            order_scope_keywords=scope_keywords,
            product_characteristics=product_characteristics,
            hts_prefix_matched=is_hts_matched,
        )
        results.append(result)

    return results


def scope_confidence_to_adcvd_confidence(scope_confidence: ScopeConfidence) -> str:
    """Map ScopeConfidence to the legacy adcvd_confidence string for compatibility."""
    mapping = {
        ScopeConfidence.SCOPE_CONFIRMED: "high",
        ScopeConfidence.SCOPE_LIKELY: "medium",
        ScopeConfidence.HTS_PREFIX_ONLY: "low",
        ScopeConfidence.SCOPE_EXCLUSION_POSSIBLE: "low",
        ScopeConfidence.SCOPE_EXCLUDED: "none",
    }
    return mapping.get(scope_confidence, "low")
