"""Auto-generate guard tokens from HTS descriptions and hierarchy.

Guard tokens are the boolean predicates that Z3 uses to match products
to tariff classifications.  Instead of hand-crafting them per line, this
module derives them from:

1. **Chapter-level tokens**: ``chapter_XX`` (e.g., ``chapter_85``)
2. **Material keywords**: ``material_steel``, ``material_plastic``, etc.
3. **Product type keywords**: ``product_type_vehicle``, ``product_type_footwear``
4. **Construction keywords**: ``textile_knit``, ``textile_woven``
5. **Origin/trade tokens**: ``assembled_in_usmca``, ``green_energy_certified``
6. **Hierarchy inheritance**: parent description tokens propagate to children

The generated tokens map to facts produced by the fact compiler, enabling
the Z3 solver to automatically match BOMs to HTS classifications.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Keyword → guard token mapping tables
# ---------------------------------------------------------------------------

# Material keywords found in HTS descriptions
_MATERIAL_KEYWORDS: Dict[str, str] = {
    "steel": "material_steel",
    "iron": "material_steel",
    "aluminum": "material_aluminum",
    "aluminium": "material_aluminum",
    "copper": "material_copper",
    "textile": "material_textile",
    "rubber": "material_rubber",
    "plastic": "material_plastic",
    "plastics": "material_plastic",
    "leather": "material_leather",
    "wood": "material_wood",
    "wooden": "material_wood",
    "glass": "material_glass",
    "ceramic": "material_ceramic",
    "cotton": "fiber_cotton",
    "polyester": "fiber_polyester",
    "nylon": "fiber_nylon",
    "synthetic": "material_synthetic",
    "silk": "fiber_silk",
    "wool": "fiber_wool",
    "linen": "fiber_linen",
    "polyethylene": "material_plastic",
    "polypropylene": "material_plastic",
    "pvc": "material_plastic",
    "stainless": "material_stainless_steel",
}

# Product type keywords
_PRODUCT_TYPE_KEYWORDS: Dict[str, str] = {
    "vehicle": "product_type_vehicle",
    "automobile": "product_type_vehicle",
    "motor car": "product_type_vehicle",
    "passenger": "product_type_vehicle",
    "truck": "product_type_vehicle",
    "tractor": "product_type_tractor",
    "motorcycle": "product_type_motorcycle",
    "bicycle": "product_type_bicycle",
    "footwear": "product_type_footwear",
    "shoe": "product_type_footwear",
    "boot": "product_type_footwear",
    "sandal": "product_type_footwear",
    "bolt": "product_type_fastener",
    "screw": "product_type_fastener",
    "nut": "product_type_fastener",
    "washer": "product_type_fastener",
    "rivet": "product_type_fastener",
    "battery": "product_type_battery",
    "accumulator": "product_type_battery",
    "cable": "product_type_cable",
    "wire": "product_type_cable",
    "conductor": "product_type_cable",
    "harness": "product_type_cable",
    "furniture": "product_type_furniture",
    "seat": "product_type_furniture",
    "chair": "product_type_furniture",
    "desk": "product_type_furniture",
    "table": "product_type_furniture",
    "toy": "product_type_toy",
    "doll": "product_type_toy",
    "game": "product_type_toy",
    "exercise": "product_type_sporting",
    "sporting": "product_type_sporting",
    "pharmaceutical": "product_type_pharma",
    "medicament": "product_type_pharma",
    "drug": "product_type_pharma",
    "fertilizer": "product_type_chemical",
    "chemical": "product_type_chemical",
    "reagent": "product_type_chemical",
    "pump": "product_type_machinery",
    "engine": "product_type_machinery",
    "motor": "product_type_machinery",
    "compressor": "product_type_machinery",
    "turbine": "product_type_machinery",
    "transformer": "product_type_electronics",
    "circuit": "product_type_electronics",
    "semiconductor": "product_type_electronics",
    "diode": "product_type_electronics",
    "transistor": "product_type_electronics",
    "integrated circuit": "product_type_electronics",
    "router": "product_type_electronics",
    "switch": "product_type_electronics",
    "garment": "product_type_apparel",
    "shirt": "product_type_apparel",
    "trouser": "product_type_apparel",
    "jacket": "product_type_apparel",
    "coat": "product_type_apparel",
    "dress": "product_type_apparel",
    "skirt": "product_type_apparel",
    "suit": "product_type_apparel",
}

# Construction / process keywords
_CONSTRUCTION_KEYWORDS: Dict[str, str] = {
    "knit": "textile_knit",
    "knitted": "textile_knit",
    "woven": "textile_woven",
    "nonwoven": "textile_nonwoven",
    "non-woven": "textile_nonwoven",
    "felted": "textile_felted",
    "braided": "textile_braided",
    "cast": "manufacturing_cast",
    "forged": "manufacturing_forged",
    "molded": "manufacturing_molded",
    "moulded": "manufacturing_molded",
    "welded": "manufacturing_welded",
    "assembled": "manufacturing_assembled",
    "laminated": "has_lamination",
    "coated": "has_coating",
    "plated": "has_plating",
    "insulated": "is_insulated",
    "dyed": "textile_dyed",
    "printed": "textile_printed",
    "bleached": "textile_bleached",
    "unbleached": "textile_unbleached",
}

# Attribute keywords
_ATTRIBUTE_KEYWORDS: Dict[str, str] = {
    "electric": "is_electric",
    "electronic": "product_type_electronics",
    "spark-ignition": "engine_type_spark_ignition",
    "spark ignition": "engine_type_spark_ignition",
    "compression-ignition": "engine_type_compression_ignition",
    "diesel": "engine_type_compression_ignition",
    "lithium": "battery_type_lithium",
    "lithium-ion": "battery_type_lithium_ion",
    "lead-acid": "battery_type_lead_acid",
    "not exceeding": "has_threshold",
    "exceeding": "has_threshold",
    "containing": "has_content_spec",
    "parts": "is_part",
    "accessories": "is_accessory",
    "thereof": "is_part",
}

# Chapter → broad category mapping
_CHAPTER_CATEGORIES: Dict[str, str] = {
    "01": "animal", "02": "meat", "03": "fish", "04": "dairy",
    "06": "plants", "07": "vegetables", "08": "fruit", "09": "spice",
    "10": "cereal", "11": "milling", "15": "fats",
    "17": "sugar", "18": "cocoa", "19": "prepared_cereals",
    "20": "prepared_veg", "21": "misc_food", "22": "beverages",
    "25": "mineral", "26": "ore", "27": "fuel",
    "28": "inorganic_chemical", "29": "organic_chemical",
    "30": "pharma", "31": "fertilizer", "32": "pigment",
    "33": "essential_oil", "34": "soap", "38": "misc_chemical",
    "39": "plastic", "40": "rubber",
    "41": "leather_raw", "42": "leather_goods", "43": "fur",
    "44": "wood", "47": "pulp", "48": "paper",
    "50": "silk", "51": "wool", "52": "cotton",
    "53": "vegetable_fiber", "54": "synthetic_filament",
    "55": "synthetic_staple", "56": "wadding", "57": "carpet",
    "58": "special_fabric", "59": "coated_fabric", "60": "knitted_fabric",
    "61": "knitted_apparel", "62": "woven_apparel", "63": "textile_article",
    "64": "footwear", "65": "headgear", "66": "umbrella",
    "68": "stone", "69": "ceramic", "70": "glass",
    "71": "jewelry", "72": "iron_steel", "73": "steel_article",
    "74": "copper", "75": "nickel", "76": "aluminum",
    "78": "lead", "79": "zinc", "80": "tin", "81": "other_metal",
    "82": "tool", "83": "misc_metal",
    "84": "machinery", "85": "electronics",
    "86": "railway", "87": "vehicle", "88": "aircraft", "89": "ship",
    "90": "instrument", "91": "clock", "92": "musical_instrument",
    "93": "arms", "94": "furniture", "95": "toy", "96": "misc_article",
    "97": "art",
}


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for keyword matching."""
    return re.sub(r"[^a-z0-9\s-]", " ", text.lower())


def _extract_keywords(
    text: str, keyword_map: Dict[str, str]
) -> Set[str]:
    """Find all matching keywords in text and return their token values."""
    normalized = _normalize_text(text)
    tokens: Set[str] = set()
    # Sort by length descending to match longer phrases first
    for keyword in sorted(keyword_map.keys(), key=len, reverse=True):
        if keyword in normalized:
            tokens.add(keyword_map[keyword])
    return tokens


def generate_guard_tokens(
    *,
    htsno: str,
    description: str,
    parent_descriptions: List[str],
    indent: int,
    chapter: str,
) -> List[str]:
    """Generate guard tokens for an HTS line from its description and hierarchy.

    Tokens are derived from:
    - Chapter number (``chapter_XX``)
    - Chapter category (``category_vehicle``, ``category_electronics``)
    - Material keywords in the description
    - Product type keywords
    - Construction/process keywords
    - Attribute keywords
    - Inherited tokens from parent descriptions in the hierarchy
    """
    tokens: Set[str] = set()

    # 1. Chapter-level token
    if chapter:
        tokens.add(f"chapter_{chapter}")
        cat = _CHAPTER_CATEGORIES.get(chapter)
        if cat:
            tokens.add(f"category_{cat}")

    # 2. Extract tokens from current description
    tokens.update(_extract_keywords(description, _MATERIAL_KEYWORDS))
    tokens.update(_extract_keywords(description, _PRODUCT_TYPE_KEYWORDS))
    tokens.update(_extract_keywords(description, _CONSTRUCTION_KEYWORDS))
    tokens.update(_extract_keywords(description, _ATTRIBUTE_KEYWORDS))

    # 3. Inherit tokens from parent descriptions (chapter/heading/subheading)
    for parent_desc in parent_descriptions:
        tokens.update(_extract_keywords(parent_desc, _MATERIAL_KEYWORDS))
        tokens.update(_extract_keywords(parent_desc, _PRODUCT_TYPE_KEYWORDS))
        tokens.update(_extract_keywords(parent_desc, _CONSTRUCTION_KEYWORDS))

    return sorted(tokens)
