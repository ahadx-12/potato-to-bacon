"""Universal fact vocabulary for bridging BOM facts and HTS guard tokens.

The fact compiler (ProductSpecModel -> facts) and the guard token generator
(HTS descriptions -> guard_tokens) historically used different vocabularies.
This module defines the canonical mapping between them so that:

1. Facts compiled from a BOM can match guard tokens on USITC-sourced atoms.
2. Guard tokens from HTS descriptions can be understood by the mutation engine.

The vocabulary defines three relationships:
- **Synonyms**: tokens that mean the same thing (e.g. ``is_fastener`` == ``product_type_fastener``)
- **Entailments**: token A implies token B (e.g. ``product_type_chassis_bolt`` -> ``product_type_fastener``)
- **Category mappings**: ProductCategory -> chapter/category tokens
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

from .product_schema import ProductCategory


# ---------------------------------------------------------------------------
# Synonym groups: tokens within a group are interchangeable
# ---------------------------------------------------------------------------
# Each tuple is a group of equivalent tokens.  When any token in the group
# is present in a fact set, all tokens in the group should be considered true.
SYNONYM_GROUPS: List[FrozenSet[str]] = [
    frozenset({"is_fastener", "product_type_fastener"}),
    frozenset({"product_type_apparel_textile", "product_type_apparel"}),
    frozenset({"has_coating_or_lamination", "has_coating", "has_lamination"}),
    frozenset({"material_steel", "material_stainless_steel"}),
]

# Build lookup: token -> set of synonyms (including itself)
_SYNONYM_LOOKUP: Dict[str, FrozenSet[str]] = {}
for _group in SYNONYM_GROUPS:
    for _token in _group:
        _SYNONYM_LOOKUP[_token] = _group


def get_synonyms(token: str) -> FrozenSet[str]:
    """Return all synonyms for a token, including itself."""
    return _SYNONYM_LOOKUP.get(token, frozenset({token}))


# ---------------------------------------------------------------------------
# Entailment rules: token A implies token B
# ---------------------------------------------------------------------------
# Directional: the presence of the key token implies all tokens in the value set.
ENTAILMENTS: Dict[str, Set[str]] = {
    # Specific product types entail broader categories
    "product_type_chassis_bolt": {"product_type_fastener", "is_fastener"},
    "product_type_vehicle": {"category_vehicle"},
    "product_type_battery": {"product_type_electronics"},
    "product_type_cable": {"product_type_electronics"},
    "product_type_machinery": {"category_machinery"},

    # Material entailments
    "material_stainless_steel": {"material_steel", "material_metal"},
    "material_steel": {"material_metal"},
    "material_aluminum": {"material_metal"},
    "fastener_stainless": {"material_stainless_steel", "material_steel", "material_metal"},

    # Fiber tokens: presence implies material
    "fiber_cotton": {"material_textile"},
    "fiber_polyester": {"material_textile", "material_synthetic"},
    "fiber_nylon": {"material_textile", "material_synthetic"},
    "fiber_silk": {"material_textile"},
    "fiber_wool": {"material_textile"},
    "fiber_cotton_dominant": {"fiber_cotton", "material_textile"},
    "fiber_polyester_dominant": {"fiber_polyester", "material_textile", "material_synthetic"},

    # Electronics sub-types
    "electronics_is_cable_assembly": {"product_type_cable", "product_type_electronics"},
    "electronics_enclosure": {"product_type_electronics"},
    "contains_battery": {"product_type_electronics"},

    # Construction entailments
    "textile_knit": {"material_textile"},
    "textile_woven": {"material_textile"},
    "textile_nonwoven": {"material_textile"},
    "textile_felted": {"material_textile"},
}


def get_entailed_tokens(token: str) -> Set[str]:
    """Return all tokens entailed by the given token (transitive)."""
    result: Set[str] = set()
    stack = [token]
    visited: Set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        implied = ENTAILMENTS.get(current, set())
        for imp in implied:
            result.add(imp)
            stack.append(imp)
    return result


# ---------------------------------------------------------------------------
# ProductCategory -> HTS chapter/category token mappings
# ---------------------------------------------------------------------------
# Maps each ProductCategory to the chapter_XX and category_XX tokens that
# should be added to compiled facts, enabling USITC atoms to match.
CATEGORY_TO_CHAPTERS: Dict[ProductCategory, List[str]] = {
    ProductCategory.FOOTWEAR: [
        "chapter_64", "category_footwear",
    ],
    ProductCategory.FASTENER: [
        "chapter_73", "category_steel_article", "category_iron_steel",
    ],
    ProductCategory.ELECTRONICS: [
        "chapter_85", "category_electronics",
    ],
    ProductCategory.TEXTILE: [
        "chapter_52", "chapter_54", "chapter_55",
        "category_cotton", "category_synthetic_filament", "category_synthetic_staple",
    ],
    ProductCategory.APPAREL_TEXTILE: [
        "chapter_61", "chapter_62", "category_knitted_apparel", "category_woven_apparel",
    ],
    ProductCategory.FURNITURE: [
        "chapter_94", "category_furniture",
    ],
    ProductCategory.OTHER: [],
}

# Reverse lookup: chapter/category token -> set of ProductCategory values
_CHAPTER_TO_CATEGORIES: Dict[str, Set[ProductCategory]] = {}
for _cat, _tokens in CATEGORY_TO_CHAPTERS.items():
    for _tok in _tokens:
        _CHAPTER_TO_CATEGORIES.setdefault(_tok, set()).add(_cat)


def chapters_for_category(category: ProductCategory) -> List[str]:
    """Return the chapter/category tokens for a ProductCategory."""
    return list(CATEGORY_TO_CHAPTERS.get(category, []))


def categories_for_chapter(chapter_token: str) -> Set[ProductCategory]:
    """Return ProductCategory values that map to a chapter/category token."""
    return _CHAPTER_TO_CATEGORIES.get(chapter_token, set())


# ---------------------------------------------------------------------------
# Material -> chapter mapping (for BOM materials)
# ---------------------------------------------------------------------------
MATERIAL_CHAPTER_TOKENS: Dict[str, List[str]] = {
    "material_steel": ["chapter_72", "chapter_73", "category_iron_steel", "category_steel_article"],
    "material_aluminum": ["chapter_76", "category_aluminum"],
    "material_copper": ["chapter_74", "category_copper"],
    "material_plastic": ["chapter_39", "category_plastic"],
    "material_rubber": ["chapter_40", "category_rubber"],
    "material_leather": ["chapter_41", "chapter_42", "category_leather_raw", "category_leather_goods"],
    "material_wood": ["chapter_44", "category_wood"],
    "material_glass": ["chapter_70", "category_glass"],
    "material_ceramic": ["chapter_69", "category_ceramic"],
    "material_textile": ["chapter_52", "chapter_54", "chapter_55"],
}


# ---------------------------------------------------------------------------
# Convenience: expand a full fact dict
# ---------------------------------------------------------------------------
def expand_facts(facts: Dict[str, object]) -> Dict[str, object]:
    """Expand a fact dictionary by adding all synonym and entailment tokens.

    This is the primary bridge function: call it after ``compile_facts()``
    to ensure the fact set includes all tokens that USITC-sourced guard
    tokens might require.

    Only boolean-true facts are expanded.  Does not remove existing facts.
    """
    expanded = dict(facts)

    # Collect all true boolean fact keys
    true_keys: Set[str] = set()
    for key, value in facts.items():
        if value is True:
            true_keys.add(key)

    # Add synonym expansions
    for key in list(true_keys):
        for syn in get_synonyms(key):
            if syn not in expanded or expanded[syn] is not True:
                expanded[syn] = True
                true_keys.add(syn)

    # Add entailment expansions (iterate until stable)
    changed = True
    iterations = 0
    while changed and iterations < 10:
        changed = False
        iterations += 1
        for key in list(true_keys):
            for implied in get_entailed_tokens(key):
                if implied not in true_keys:
                    expanded[implied] = True
                    true_keys.add(implied)
                    changed = True

    # Add category/chapter tokens based on product_category
    product_cat_value = facts.get("product_category")
    if product_cat_value:
        try:
            cat = ProductCategory(product_cat_value)
            for ch_token in chapters_for_category(cat):
                expanded[ch_token] = True
        except ValueError:
            pass

    # Add material-derived chapter tokens
    for mat_key, ch_tokens in MATERIAL_CHAPTER_TOKENS.items():
        if expanded.get(mat_key) is True:
            for ch_tok in ch_tokens:
                expanded[ch_tok] = True

    return expanded
