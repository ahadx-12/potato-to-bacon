"""General Rules of Interpretation (GRI) Classification Engine.

The GRI are the six legally binding rules that govern HTS classification in
every WTO member country.  They are applied in strict cascade order: GRI 1
first, and each subsequent rule only when the prior rule is insufficient to
determine classification.

A tariff CALCULATOR ignores GRI.  It accepts an HTS code and returns a rate.

A tariff ENGINEER applies GRI explicitly to:
  1. Determine the legally correct classification for a product
  2. Identify alternative classifications that are also defensible
  3. Surface reclassification opportunities when the current code is wrong
  4. Provide legal grounding for every recommendation (citing the specific rule)

This engine takes a list of candidate HTS headings (from hts_search) and
product facts (description, BOM materials), applies GRI 1-6 in cascade,
and returns the winning heading with the specific rule that determined it
and the full legal reasoning chain.

GRI Reference:
  GRI 1:  Classification by heading text + section/chapter notes
  GRI 2a: Incomplete/unfinished articles = classified as the finished article
  GRI 2b: Mixtures/composites → apply GRI 3
  GRI 3a: Most specific heading (among equally applicable headings)
  GRI 3b: Essential character of the good (primary material)
  GRI 3c: Last in order (tie-breaker: highest HTS number wins)
  GRI 4:  Most akin (last resort: no prior rule determined it)
  GRI 6:  Subheading determination uses same rules
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from potatobacon.tariff.hts_search import HTSSearchResult


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class GRIReasoning:
    """The legal reasoning chain for a GRI determination."""

    gri_rule: str               # "GRI_1", "GRI_2a", "GRI_3a", "GRI_3b", "GRI_3c", "GRI_4"
    rule_text: str              # Canonical rule text
    applied_test: str           # What test was applied (e.g. "heading text match")
    result: str                 # "unique match", "essential character: steel 60%", etc.
    legal_citation: str         # Citation string suitable for opportunity.legal_basis


@dataclass
class GRIClassification:
    """The output of a GRI cascade: the winning HTS heading and why."""

    winning_code: str                          # Best HTS code (4+ digits)
    winning_heading: str                       # 4-digit heading
    winning_description: str                   # Heading legal text
    winning_chapter: int                       # Chapter number
    base_duty_rate: str                        # Raw rate string
    gri_chain: List[GRIReasoning]              # All rules applied in order
    determining_rule: str                      # The specific rule that decided it
    alternatives: List[HTSSearchResult]        # Other candidates considered (ranked)
    confidence: str                            # "high", "medium", "low"
    reclassification_opportunity: bool         # True if winning ≠ first candidate
    notes: List[str]                           # Substantive notes for the engineer


_GRI_TEXTS = {
    "GRI_1": (
        "Classification shall be determined according to the terms of the headings "
        "and any relative section or chapter notes."
    ),
    "GRI_2a": (
        "Any reference to an article shall include a reference to that article "
        "incomplete or unfinished, provided it has the essential character of the "
        "complete or finished article."
    ),
    "GRI_2b": (
        "Any reference to a material or substance shall include mixtures or "
        "combinations of that material or substance with other materials. "
        "Classification of mixed goods → GRI 3."
    ),
    "GRI_3a": (
        "When goods are prima facie classifiable under two or more headings, the "
        "heading which provides the most specific description shall be preferred."
    ),
    "GRI_3b": (
        "When goods cannot be classified under GRI 3a, they shall be classified "
        "under the heading for the material or component that gives them their "
        "essential character."
    ),
    "GRI_3c": (
        "When goods cannot be classified under GRI 3a or 3b, they shall be "
        "classified under the heading which occurs last in numerical order."
    ),
    "GRI_4": (
        "Goods which cannot be classified in accordance with GRI 1-3 shall be "
        "classified under the heading appropriate to the goods to which they "
        "are most akin."
    ),
    "GRI_6": (
        "Classification at the subheading level is determined by the terms of "
        "subheadings, any related notes, and GRI 1-5 mutatis mutandis."
    ),
}

# Words that indicate an unfinished/incomplete article (triggers GRI 2a)
_UNFINISHED_MARKERS = frozenset([
    "unfinished", "incomplete", "blank", "semi-finished", "semifinished",
    "casting", "forging", "preform", "green", "unfired", "rough",
    "raw", "crude", "billet", "ingot", "slab", "bloom",
])

# Words indicating a composite/mixture (may trigger GRI 2b → GRI 3)
_COMPOSITE_MARKERS = frozenset([
    "composite", "combination", "mixture", "mixed", "combined",
    "reinforced", "coated", "laminated", "plated", "clad",
])


# ---------------------------------------------------------------------------
# Specificity scoring for GRI 3a
# ---------------------------------------------------------------------------

def _specificity_score(result: HTSSearchResult, query_tokens: List[str]) -> float:
    """Score how specifically a heading covers the product.

    Higher specificity → GRI 3a preference.

    Factors:
    - Length of HTS code (more digits = more specific subheading)
    - Number of matched query terms
    - Whether the code has an explicit non-zero duty rate
    - Score from the search engine (text similarity)
    """
    code_digits = re.sub(r"\D", "", result.hts_code)
    # Longer code = more specific
    digit_bonus = len(code_digits) * 0.1

    # Matched terms as proportion of query
    match_ratio = len(result.matched_terms) / max(len(query_tokens), 1)

    # Penalise "NESOI" / "not elsewhere specified" headings — they are catchalls
    nesoi_penalty = 0.3 if "nesoi" in result.description.lower() else 0.0
    catchall_penalty = 0.2 if "other" in result.description.lower()[:20] else 0.0

    return (result.score * 0.6 + digit_bonus + match_ratio * 0.3
            - nesoi_penalty - catchall_penalty)


# ---------------------------------------------------------------------------
# Essential character determination for GRI 3b
# ---------------------------------------------------------------------------

def _essential_character(
    materials: List[Dict[str, str]],
    candidates: List[HTSSearchResult],
) -> Optional[HTSSearchResult]:
    """Determine essential character from BOM materials.

    The essential character is the component that gives the good its
    defining nature.  CBP generally uses weight or value as the proxy.

    This implementation:
    1. Extracts material names from the BOM
    2. Scores each candidate by how well its heading description matches
       the dominant materials
    3. Returns the candidate best matching the dominant material
    """
    if not materials or not candidates:
        return None

    # Build a simple material frequency map
    material_counts: Dict[str, int] = {}
    for m in materials:
        mat = (m.get("material") or "").lower().strip()
        if mat:
            for token in re.split(r"[^a-z]+", mat):
                if token and len(token) > 2:
                    material_counts[token] = material_counts.get(token, 0) + 1

    if not material_counts:
        return None

    best_candidate: Optional[HTSSearchResult] = None
    best_score = -1.0

    for candidate in candidates:
        desc_lower = candidate.description.lower()
        score = 0.0
        for mat_token, count in material_counts.items():
            if mat_token in desc_lower:
                score += count
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate if best_score > 0 else None


# ---------------------------------------------------------------------------
# Core GRI cascade
# ---------------------------------------------------------------------------

def apply_gri(
    description: str,
    materials: Optional[List[Dict[str, str]]],
    candidates: List[HTSSearchResult],
) -> GRIClassification:
    """Apply GRI rules 1-6 to select the winning HTS heading.

    Args:
        description: Product description (free text).
        materials: BOM material list, e.g. [{"component": "body", "material": "steel"}]
        candidates: HTS search results, sorted by relevance score descending.

    Returns:
        GRIClassification with the winning heading, determining rule, and full
        legal reasoning chain.
    """
    if not candidates:
        return _no_classification(description)

    gri_chain: List[GRIReasoning] = []
    notes: List[str] = []
    desc_lower = description.lower()
    query_tokens = re.split(r"[^a-z0-9]+", desc_lower)
    query_tokens = [t for t in query_tokens if t and len(t) > 1]
    materials = materials or []

    # ------------------------------------------------------------------
    # GRI 2a: Unfinished/incomplete article — check FIRST (before GRI 1)
    # so the note is always in the reasoning chain when applicable.
    # GRI 2a is an annotation, not a redirection — it permits classification
    # of the incomplete article as if it were complete, then falls through.
    # ------------------------------------------------------------------
    desc_tokens = set(re.split(r"[^a-z]+", desc_lower))
    unfinished_hit = desc_tokens & _UNFINISHED_MARKERS
    if unfinished_hit:
        notes.append(
            f"GRI 2(a) triggered: product description contains unfinished-article "
            f"marker(s): {', '.join(sorted(unfinished_hit))}. "
            "Classifying as if the article were complete."
        )
        gri_chain.append(GRIReasoning(
            gri_rule="GRI_2a",
            rule_text=_GRI_TEXTS["GRI_2a"],
            applied_test=f"Unfinished marker detected: {sorted(unfinished_hit)}",
            result="Proceed as if article were complete/finished",
            legal_citation="GRI Rule 2(a) — incomplete articles",
        ))
        # Fall through — GRI 2a does not change the heading selection, it
        # just authorises classifying the incomplete article as the finished one.

    # ------------------------------------------------------------------
    # GRI 1: Classify by heading terms + chapter notes
    # Single candidate with a clearly dominant score → GRI 1 determination
    # ------------------------------------------------------------------
    if len(candidates) == 1:
        winner = candidates[0]
        gri_chain.append(GRIReasoning(
            gri_rule="GRI_1",
            rule_text=_GRI_TEXTS["GRI_1"],
            applied_test="Single candidate heading — no conflict",
            result=f"Heading {winner.heading}: {winner.description[:80]}",
            legal_citation=f"GRI Rule 1 — HTS heading {winner.heading}",
        ))
        return _build_result(
            winner=winner,
            candidates=candidates,
            gri_chain=gri_chain,
            determining_rule="GRI_1",
            notes=notes,
            first_candidate=candidates[0],
        )

    # Check if the top candidate has a dominant score (≥ 2x the next)
    if (candidates[0].score > 0 and len(candidates) >= 2 and
            candidates[0].score >= 2.0 * candidates[1].score):
        winner = candidates[0]
        gri_chain.append(GRIReasoning(
            gri_rule="GRI_1",
            rule_text=_GRI_TEXTS["GRI_1"],
            applied_test="Dominant heading text match (score ≥ 2× next candidate)",
            result=(
                f"Heading {winner.heading} score={winner.score:.3f} vs "
                f"next={candidates[1].score:.3f}"
            ),
            legal_citation=f"GRI Rule 1 — HTS heading {winner.heading}",
        ))
        return _build_result(
            winner=winner,
            candidates=candidates,
            gri_chain=gri_chain,
            determining_rule="GRI_1",
            notes=notes,
            first_candidate=candidates[0],
        )

    # GRI 2b: Composite/mixture check
    composite_hit = desc_tokens & _COMPOSITE_MARKERS
    has_multiple_materials = len(materials) >= 2
    if composite_hit or has_multiple_materials:
        notes.append(
            "GRI 2(b) / GRI 3 pathway: product has multiple materials or "
            "composite markers. Essential character analysis required."
        )
        gri_chain.append(GRIReasoning(
            gri_rule="GRI_2b",
            rule_text=_GRI_TEXTS["GRI_2b"],
            applied_test=(
                f"Composite markers: {sorted(composite_hit)}, "
                f"BOM materials: {len(materials)}"
            ),
            result="Multiple candidate headings → proceed to GRI 3",
            legal_citation="GRI Rule 2(b) → GRI Rule 3 cascade",
        ))

    # ------------------------------------------------------------------
    # GRI 3a: Most specific heading
    # ------------------------------------------------------------------
    specificity = [(c, _specificity_score(c, query_tokens)) for c in candidates]
    specificity.sort(key=lambda x: -x[1])
    top_spec = specificity[0]
    second_spec = specificity[1] if len(specificity) >= 2 else None

    if second_spec is None or top_spec[1] > second_spec[1] * 1.25:
        # Clear specificity winner
        winner = top_spec[0]
        gri_chain.append(GRIReasoning(
            gri_rule="GRI_3a",
            rule_text=_GRI_TEXTS["GRI_3a"],
            applied_test="Specificity scoring: matched terms + code length + NESOI penalty",
            result=(
                f"Heading {winner.heading} specificity={top_spec[1]:.3f} "
                f"(next={second_spec[1]:.3f if second_spec else 'n/a'})"
            ),
            legal_citation=f"GRI Rule 3(a) — most specific heading {winner.heading}",
        ))
        return _build_result(
            winner=winner,
            candidates=candidates,
            gri_chain=gri_chain,
            determining_rule="GRI_3a",
            notes=notes,
            first_candidate=candidates[0],
        )

    # ------------------------------------------------------------------
    # GRI 3b: Essential character
    # ------------------------------------------------------------------
    ec_candidate = _essential_character(materials, candidates[:5])
    if ec_candidate:
        gri_chain.append(GRIReasoning(
            gri_rule="GRI_3b",
            rule_text=_GRI_TEXTS["GRI_3b"],
            applied_test="Material composition match against heading descriptions",
            result=(
                f"Dominant material matched heading {ec_candidate.heading}: "
                f"{ec_candidate.description[:60]}"
            ),
            legal_citation=f"GRI Rule 3(b) — essential character, heading {ec_candidate.heading}",
        ))
        return _build_result(
            winner=ec_candidate,
            candidates=candidates,
            gri_chain=gri_chain,
            determining_rule="GRI_3b",
            notes=notes,
            first_candidate=candidates[0],
        )

    # ------------------------------------------------------------------
    # GRI 3c: Last heading in numerical order (tie-breaker)
    # ------------------------------------------------------------------
    def _heading_num(r: HTSSearchResult) -> int:
        try:
            return int(re.sub(r"\D", "", r.hts_code)[:8] or "0")
        except (ValueError, TypeError):
            return 0

    by_code = sorted(candidates[:5], key=_heading_num)
    winner = by_code[-1]  # Last in numerical order
    gri_chain.append(GRIReasoning(
        gri_rule="GRI_3c",
        rule_text=_GRI_TEXTS["GRI_3c"],
        applied_test="Tie-breaker: last heading in numerical order",
        result=f"Heading {winner.heading} is last among equally specific headings",
        legal_citation=f"GRI Rule 3(c) — last heading {winner.heading}",
    ))
    notes.append(
        "GRI 3(c) tie-breaker applied. Consider requesting a CBP binding ruling "
        "to confirm classification — ambiguity between headings creates audit risk."
    )

    return _build_result(
        winner=winner,
        candidates=candidates,
        gri_chain=gri_chain,
        determining_rule="GRI_3c",
        notes=notes,
        first_candidate=candidates[0],
    )


def _build_result(
    *,
    winner: HTSSearchResult,
    candidates: List[HTSSearchResult],
    gri_chain: List[GRIReasoning],
    determining_rule: str,
    notes: List[str],
    first_candidate: HTSSearchResult,
) -> GRIClassification:
    """Assemble the final GRIClassification object."""
    reclassification_opportunity = (
        winner.heading != first_candidate.heading
    )

    # Confidence: GRI_1 high, GRI_3a medium, GRI_3b/3c low
    confidence_map = {
        "GRI_1": "high",
        "GRI_2a": "high",
        "GRI_3a": "medium",
        "GRI_3b": "medium",
        "GRI_3c": "low",
        "GRI_4": "low",
    }
    confidence = confidence_map.get(determining_rule, "medium")

    if reclassification_opportunity:
        notes.append(
            f"Reclassification opportunity: GRI analysis places product in heading "
            f"{winner.heading} ({winner.description[:60]}), but search results led "
            f"with heading {first_candidate.heading} "
            f"({first_candidate.description[:60]}). "
            "This warrants a CBP binding ruling request."
        )

    alternatives = [c for c in candidates if c.heading != winner.heading]

    return GRIClassification(
        winning_code=winner.hts_code,
        winning_heading=winner.heading,
        winning_description=winner.description,
        winning_chapter=winner.chapter,
        base_duty_rate=winner.base_duty_rate,
        gri_chain=gri_chain,
        determining_rule=determining_rule,
        alternatives=alternatives,
        confidence=confidence,
        reclassification_opportunity=reclassification_opportunity,
        notes=notes,
    )


def _no_classification(description: str) -> GRIClassification:
    """Return an empty classification when no candidates are available."""
    return GRIClassification(
        winning_code="",
        winning_heading="",
        winning_description="No candidate headings found",
        winning_chapter=0,
        base_duty_rate="",
        gri_chain=[],
        determining_rule="NONE",
        alternatives=[],
        confidence="low",
        reclassification_opportunity=False,
        notes=[
            "No HTS heading candidates found for this product description. "
            "Manual classification by a licensed customs broker is required. "
            "Provide a more detailed product description or BOM."
        ],
    )


# ---------------------------------------------------------------------------
# Convenience: classify and return legal_basis strings for opportunity model
# ---------------------------------------------------------------------------

def gri_legal_basis(classification: GRIClassification) -> List[str]:
    """Extract legal_basis strings from a GRIClassification for use in opportunities."""
    basis = []
    for reasoning in classification.gri_chain:
        basis.append(reasoning.legal_citation)
    if classification.winning_heading:
        basis.append(f"HTS heading {classification.winning_heading}: {classification.winning_description[:80]}")
    return basis
