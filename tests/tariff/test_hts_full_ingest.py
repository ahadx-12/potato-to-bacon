from potatobacon.tariff.hts_ingest.full_ingest import (
    get_atom_by_hts,
    ingest_chapter,
    parse_duty_rate,
)


def test_hts_ingestion_chapter_84():
    atoms = ingest_chapter(84)
    assert len(atoms) > 500
    assert any(atom.hts_code == "8471.30.01" for atom in atoms)
    assert all(atom.base_duty_rate is not None for atom in atoms)


def test_duty_rate_parsing():
    assert parse_duty_rate("5.7%").type == "ad_valorem"
    assert parse_duty_rate("5.7%").ad_valorem == 0.057
    assert parse_duty_rate("Free").type == "free"
    dr = parse_duty_rate("$0.37/kg + 5%")
    assert dr.type == "compound"
    assert dr.specific == 0.37 and dr.specific_unit == "kg"
    assert dr.ad_valorem == 0.05


def test_special_rates_included():
    atom = get_atom_by_hts("8471.30.01")
    assert "CA" in atom.special_rates
    assert "MX" in atom.special_rates
