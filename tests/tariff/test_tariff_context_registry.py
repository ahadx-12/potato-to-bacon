import pytest

from potatobacon.tariff.context_registry import (
    DEFAULT_CONTEXT_ID,
    get_context_manifest,
    list_context_manifests,
    load_atoms_for_context,
)


def test_list_contexts_contains_default():
    manifests = list_context_manifests()
    assert any(manifest["context_id"] == DEFAULT_CONTEXT_ID for manifest in manifests)


def test_load_atoms_for_default_context():
    atoms, metadata = load_atoms_for_context(DEFAULT_CONTEXT_ID)
    assert atoms
    assert metadata["atoms_count"] == len(atoms)
    assert metadata["atoms_count"] > 0


def test_manifest_hash_is_deterministic():
    _, meta_first = load_atoms_for_context(DEFAULT_CONTEXT_ID)
    _, meta_second = load_atoms_for_context(DEFAULT_CONTEXT_ID)
    assert meta_first["manifest_hash"] == meta_second["manifest_hash"]


def test_unknown_context_raises():
    with pytest.raises(KeyError):
        get_context_manifest("NOPE")
