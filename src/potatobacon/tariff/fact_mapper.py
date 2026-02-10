"""Bidirectional mapper between BOM-compiled facts and HTS guard tokens.

Provides two core operations:

1. **facts_to_guard_tokens**: Given compiled facts, produce the full set of
   guard tokens that could satisfy an atom's guard clause.
2. **guard_tokens_to_required_facts**: Given an atom's guard tokens, determine
   what BOM properties would need to be true to satisfy them.

This module is the runtime bridge used by the TEaaS endpoint and mutation
engine to match BOM-compiled facts against USITC-sourced PolicyAtoms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from .fact_vocabulary import (
    ENTAILMENTS,
    SYNONYM_GROUPS,
    expand_facts,
    get_entailed_tokens,
    get_synonyms,
)


def facts_to_guard_tokens(facts: Dict[str, Any]) -> Set[str]:
    """Convert compiled facts into the full set of satisfiable guard tokens.

    Takes the output of ``compile_facts()`` and returns every token that
    a PolicyAtom guard clause might reference, including:
    - Direct fact keys that are True
    - Synonym expansions
    - Entailment expansions
    - Chapter/category tokens from product category and materials

    Usage::

        facts, evidence = compile_facts(product_spec)
        tokens = facts_to_guard_tokens(facts)
        # tokens now includes chapter_73, category_steel_article, etc.
    """
    expanded = expand_facts(facts)
    return {key for key, value in expanded.items() if value is True}


def guard_tokens_to_required_facts(guard_tokens: List[str]) -> Dict[str, bool]:
    """Determine what facts must be true/false to satisfy a guard clause.

    For each token in the guard:
    - Positive tokens (e.g. ``material_steel``) require the fact to be True
    - Negated tokens (e.g. ``\\u00acmaterial_steel``) require the fact to be False

    Returns a dict of {fact_key: required_value}.
    """
    required: Dict[str, bool] = {}
    for token in guard_tokens:
        if token.startswith("\u00ac"):
            fact_key = token[1:]
            required[fact_key] = False
        else:
            required[token] = True
    return required


def compute_fact_gap(
    current_facts: Dict[str, Any],
    target_guard: List[str],
) -> Dict[str, Any]:
    """Compute the minimal fact patch to satisfy a target guard clause.

    Expands current facts through the vocabulary bridge before computing
    the diff, so synonym and entailment equivalences are considered.

    Returns a dict of fact changes needed.  Empty dict means the guard
    is already satisfied.
    """
    # Expand current facts to include all equivalent tokens
    expanded = expand_facts(current_facts)
    expanded_true = {k for k, v in expanded.items() if v is True}

    patch: Dict[str, Any] = {}
    for token in target_guard:
        if token.startswith("\u00ac"):
            fact_key = token[1:]
            if fact_key in expanded_true:
                patch[fact_key] = False
        else:
            if token not in expanded_true:
                # Check if any synonym is satisfied
                synonyms = get_synonyms(token)
                if not (synonyms & expanded_true):
                    # Check if any token that entails this one is present
                    # (reverse entailment check)
                    found_via_entailment = False
                    for candidate in expanded_true:
                        if token in get_entailed_tokens(candidate):
                            found_via_entailment = True
                            break
                    if not found_via_entailment:
                        patch[token] = True
    return patch


def can_satisfy_guard(
    facts: Dict[str, Any],
    guard_tokens: List[str],
) -> bool:
    """Check if the expanded facts can satisfy all guard tokens."""
    return len(compute_fact_gap(facts, guard_tokens)) == 0
