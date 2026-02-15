"""Tests for full USITC HTS data ingest pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


def test_build_seed_from_records():
    """build_seed_from_records should create seed with correct metadata and rate entries."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from fetch_full_hts import build_seed_from_records

    records = [
        {
            "htsno": "0101.21.00",
            "indent": 2,
            "description": "Live purebred breeding horses",
            "general": "Free",
            "special": "",
        },
        {
            "htsno": "0201.10.05",
            "indent": 2,
            "description": "Fresh/chilled beef carcasses",
            "general": "4.4 cents/kg",
            "special": "Free (A,AU,BH,CL,CO,D,E,IL,JO,KR,MA,OM,P,PA,PE,SG)",
        },
        {
            "htsno": "8471.30.01",
            "indent": 2,
            "description": "Portable automatic data processing machines",
            "general": "Free",
            "special": "",
        },
        {
            "htsno": "6402.99.31",
            "indent": 2,
            "description": "Tennis shoes, basketball shoes",
            "general": "20%",
            "special": "Free (AU,BH,CL,CO,D,E,IL,JO,KR,MA,OM,P,PA,PE,SG)",
        },
        {
            "htsno": "3901.10.10",
            "indent": 2,
            "description": "Polyethylene having a specific gravity less than 0.94",
            "general": "6.5%",
            "special": "Free (A+,AU,BH,CL,CO,D,E,IL,JO,KR,MA,OM,P,PA,PE,SG)",
        },
        {
            # Header line — should be skipped (indent=0, <8 digits)
            "htsno": "01",
            "indent": 0,
            "description": "Live animals",
            "general": "",
            "special": "",
        },
        {
            "htsno": "2204.21.50",
            "indent": 2,
            "description": "Red wine in containers of 2 liters or less",
            "general": "6.3 cents/liter",
            "special": "Free (A+,AU,BH,CL,CO,D,E,IL,JO,KR,MA,OM,P,PA,PE,SG)",
        },
        {
            "htsno": "8471.50.01",
            "indent": 2,
            "description": "Processing units, digital, with storage",
            "general": "2.5% + 4.6 cents/kg",
            "special": "",
            "units": ["kg", "No."],
        },
    ]

    seed = build_seed_from_records(records)

    # Verify metadata
    assert "metadata" in seed
    assert "rates" in seed
    meta = seed["metadata"]
    assert "7 entries" in meta["coverage"]  # 7 valid entries (header skipped)
    assert meta["rate_type_breakdown"]["free"] >= 2
    assert meta["rate_type_breakdown"]["ad_valorem"] >= 1
    assert meta["rate_type_breakdown"]["specific"] >= 1
    assert meta["rate_type_breakdown"]["compound"] >= 1

    # Verify rate entries
    rates = seed["rates"]
    assert len(rates) == 7

    # Check free rate
    horse = next(r for r in rates if r["hts_code"] == "0101.21.00")
    assert horse["general"] == "Free"

    # Check specific rate → should have general_structured
    beef = next(r for r in rates if r["hts_code"] == "0201.10.05")
    assert "general_structured" in beef
    assert beef["general_structured"]["type"] == "specific"

    # Check ad valorem rate
    shoe = next(r for r in rates if r["hts_code"] == "6402.99.31")
    assert shoe["general"] == "20%"

    # Check compound rate
    cpu = next(r for r in rates if r["hts_code"] == "8471.50.01")
    assert "general_structured" in cpu
    assert cpu["general_structured"]["type"] == "compound"


def test_rate_store_load_full_seed(tmp_path: Path):
    """MFNRateStore.load_full_seed should load entries from seed JSON."""
    from potatobacon.tariff.rate_store import MFNRateStore

    seed = {
        "metadata": {"source": "test"},
        "rates": [
            {"hts_code": "0101.21.00", "general": "Free", "description": "Horses"},
            {"hts_code": "6402.99.31", "general": "20%", "description": "Shoes"},
            {"hts_code": "0201.10.05", "general": "4.4 cents/kg", "description": "Beef"},
        ],
    }
    seed_path = tmp_path / "test_seed.json"
    seed_path.write_text(json.dumps(seed), encoding="utf-8")

    store = MFNRateStore()
    count = store.load_full_seed(seed_path)

    assert count == 3
    assert store.entry_count >= 3

    # Verify lookup works
    result = store.lookup("0101.21.00")
    assert result.found

    result2 = store.lookup("6402.99.31")
    assert result2.found
    assert result2.ad_valorem_rate is not None
    assert abs(result2.ad_valorem_rate - 0.20) < 0.01


def test_rate_store_load_usitc_edition(tmp_path: Path):
    """MFNRateStore.load_usitc_edition should parse raw USITC JSONL."""
    from potatobacon.tariff.rate_store import MFNRateStore

    records = [
        {"htsno": "0101.21.00", "indent": 2, "general": "Free", "description": "Horses", "special": ""},
        {"htsno": "6402.99.31", "indent": 2, "general": "20%", "description": "Shoes", "special": ""},
        {"htsno": "01", "indent": 0, "general": "", "description": "Header", "special": ""},
    ]
    jsonl_path = tmp_path / "USITC_test.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    store = MFNRateStore()
    count = store.load_usitc_edition(jsonl_path)

    assert count == 2  # Header line skipped
    assert store.entry_count == 2

    result = store.lookup("0101.21.00")
    assert result.found


def test_rate_store_loading_priority(tmp_path: Path):
    """Verify loading priority: full seed overrides chapter data."""
    from potatobacon.tariff.rate_store import MFNRateStore

    # Chapter file has 5% rate
    chapter_data = [
        {"hts_code": "0101.21.00", "base_duty_rate": "5%", "description": "Chapter rate"},
    ]
    chapter_path = tmp_path / "ch01.jsonl"
    with chapter_path.open("w", encoding="utf-8") as f:
        for rec in chapter_data:
            f.write(json.dumps(rec) + "\n")

    # Full seed has Free rate
    seed = {
        "metadata": {"source": "full"},
        "rates": [
            {"hts_code": "0101.21.00", "general": "Free", "description": "Full seed rate"},
        ],
    }
    seed_path = tmp_path / "full_seed.json"
    seed_path.write_text(json.dumps(seed), encoding="utf-8")

    store = MFNRateStore()
    store.load_chapter_jsonl(chapter_path)  # Load chapter first
    store.load_full_seed(seed_path)  # Full seed overrides

    result = store.lookup("0101.21.00")
    assert result.found
    # Full seed should have overridden the chapter data
    assert result.description == "Full seed rate"
