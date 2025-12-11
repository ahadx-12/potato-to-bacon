import pytest

from potatobacon.tariff.atoms_hts import tariff_policy_atoms
from potatobacon.tariff.context_loader import (
    get_default_tariff_context,
    get_tariff_atoms_for_context,
)


def test_default_tariff_context_loads_atoms():
    default_context = get_default_tariff_context()
    atoms_for_context = get_tariff_atoms_for_context(default_context)
    assert atoms_for_context
    assert len(atoms_for_context) == len(tariff_policy_atoms())


def test_unknown_context_raises_error():
    with pytest.raises(ValueError):
        get_tariff_atoms_for_context("UNKNOWN_CONTEXT")
