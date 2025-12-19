from __future__ import annotations

from typing import Any, Dict, List, Tuple


def normalize_compiled_facts(facts: Dict[str, bool]) -> Tuple[Dict[str, bool], List[str]]:
    """Normalize mutually exclusive fact groups and record decisions."""

    normalized = dict(facts)
    notes: List[str] = []

    textile_key = "surface_contact_textile_gt_50"
    rubber_key = "surface_contact_rubber_gt_50"
    if normalized.get(textile_key) and normalized.get(rubber_key):
        normalized[rubber_key] = False
        notes.append("surface_contact_textile_gt_50 dominates over surface_contact_rubber_gt_50")

    steel_key = "material_steel"
    aluminum_key = "material_aluminum"
    if normalized.get(steel_key) and normalized.get(aluminum_key):
        normalized["requires_measurement"] = True
        notes.append("material_steel and material_aluminum both True; mark requires_measurement")

    knit_key = "textile_knit"
    woven_key = "textile_woven"
    if normalized.get(knit_key) and normalized.get(woven_key):
        normalized[woven_key] = False
        notes.append("textile_knit and textile_woven both True; favor knit")

    return normalized, notes


def validate_minimum_inputs(product_spec: Dict[str, Any], facts: Dict[str, bool]) -> List[str]:
    """Return missing critical inputs by category."""

    category = product_spec.get("product_category") or "OTHER"
    missing: List[str] = []

    if category == "FOOTWEAR":
        if not (facts.get("surface_contact_textile_gt_50") or facts.get("surface_contact_rubber_gt_50")):
            missing.append("surface contact dominance")
        if not (facts.get("material_rubber") or facts.get("material_textile")):
            missing.append("upper/sole materials")
    elif category == "FASTENER":
        if not facts.get("product_type_chassis_bolt"):
            missing.append("product_type_chassis_bolt")
        if not (facts.get("material_steel") or facts.get("material_aluminum")):
            missing.append("fastener material")
    elif category == "ELECTRONICS":
        if not (
            facts.get("product_type_electronics")
            or facts.get("electronics_cable_or_connector")
            or facts.get("electronics_enclosure")
        ):
            missing.append("electronics form factor")
        if not (
            facts.get("material_plastic")
            or facts.get("material_steel")
            or facts.get("material_aluminum")
        ):
            missing.append("dominant material")
    elif category == "APPAREL_TEXTILE":
        if not (facts.get("textile_knit") or facts.get("textile_woven")):
            missing.append("textile knit/woven flag")
        if not (
            facts.get("fiber_cotton_dominant")
            or facts.get("fiber_polyester_dominant")
            or facts.get("material_textile")
        ):
            missing.append("dominant fiber")

    return missing
