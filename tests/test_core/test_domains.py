"""Tests for domain metadata utilities."""

from potatobacon.core.domains import (
    CLASSICAL,
    DOMAIN_REGISTRY,
    QUANTUM,
    RELATIVISTIC,
    STATISTICAL,
    get_domain,
    list_domains,
)


def test_classical_metadata():
    assert CLASSICAL.name == "classical"
    assert "Newtonian" in CLASSICAL.description
    assert "dimensional_consistency" in CLASSICAL.validation_rules


def test_relativistic_metadata():
    assert RELATIVISTIC.name == "relativistic"
    assert "Lorentz" in " ".join(RELATIVISTIC.assumptions)


def test_quantum_metadata():
    assert QUANTUM.name == "quantum"
    assert any("uncertainty" in assumption.lower() for assumption in QUANTUM.assumptions)


def test_statistical_metadata():
    assert STATISTICAL.name == "statistical"
    assert "thermodynamics" in STATISTICAL.description.lower()


def test_domain_registry():
    assert set(list_domains()) == set(DOMAIN_REGISTRY.keys())


def test_get_domain_case_insensitive():
    assert get_domain("CLASSICAL") is CLASSICAL
    assert get_domain("relativistic") is RELATIVISTIC
    assert get_domain("missing") is None
