import pytest

from potatobacon.tariff.context_registry import compute_context_hash, get_context_manifest
from potatobacon.tariff.hts_ingest.ingest import load_hts_policy_atoms


EXPECTED_HASH = "6194f17510a71a93160b8bedf7a728cb2934e4ef1fcc18795cc861e75964d0c5"


def test_hts_manifest_hash_deterministic():
    manifest = get_context_manifest("HTS_US_2025_SLICE")

    first = load_hts_policy_atoms()
    second = load_hts_policy_atoms()

    hash_one = compute_context_hash(manifest, first.atoms)
    hash_two = compute_context_hash(manifest, second.atoms)

    assert hash_one == EXPECTED_HASH
    assert hash_two == EXPECTED_HASH
    assert hash_one == hash_two


def test_atoms_include_citations_and_duty_rates():
    result = load_hts_policy_atoms()
    atoms = result.atoms

    low_voltage = next(atom for atom in atoms if atom.source_id == "HTS_ELECTRONICS_SIGNAL_LOW_VOLT")
    metadata = low_voltage.metadata or {}
    citation = metadata.get("citation") or {}

    assert metadata.get("duty_rate") == pytest.approx(1.0)
    assert metadata.get("effective_date") == "2025-01-01"
    assert citation.get("heading") == "8544"
    assert citation.get("note_id") == "HD8544_NOTE_VOLTAGE"

    active_cable = next(atom for atom in atoms if atom.source_id == "HTS_ELECTRONICS_ACTIVE_CABLE")
    assert active_cable.metadata is not None
    assert active_cable.metadata.get("description")
    assert active_cable.metadata.get("duty_rate") == pytest.approx(1.25)
