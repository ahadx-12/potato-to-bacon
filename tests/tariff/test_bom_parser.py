"""Tests for Sprint C: BOM Parser & Upload pipeline.

Covers:
- BOM parser with CSV, JSON, XLSX formats
- Fuzzy column matching
- Material extraction from descriptions
- HTS hint resolution
- Edge cases: empty rows, section headers, partial data
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from potatobacon.tariff.bom_parser import (
    BOMParseResult,
    ParsedBOMItem,
    _normalize_header,
    compute_material_percentages,
    extract_facts_from_description,
    match_columns,
    parse_bom_file,
    parse_csv,
    parse_json,
    parse_xlsx,
)
from potatobacon.tariff.hts_hint_resolver import (
    adjacent_headings,
    extract_chapter,
    extract_heading,
    filter_atoms_by_headings,
    normalize_hts_code,
    resolve_hts_hint,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =====================================================================
# BOM Parser: CSV
# =====================================================================
class TestCSVParser:
    def test_clean_csv_all_columns(self):
        """Parse a clean CSV with all standard columns."""
        csv_path = FIXTURES_DIR / "clean_bom.csv"
        result = parse_csv(csv_path.read_bytes())

        assert len(result.items) == 7
        assert len(result.skipped) == 0
        assert "part_id" in result.detected_columns
        assert "description" in result.detected_columns
        assert "material" in result.detected_columns
        assert "weight_kg" in result.detected_columns
        assert "value_usd" in result.detected_columns
        assert "origin_country" in result.detected_columns
        assert "hts_code" in result.detected_columns

        # Check first item
        first = result.items[0]
        assert first.part_id == "P001"
        assert "steel" in first.description.lower()
        assert first.material == "Steel"
        assert first.weight_kg == 0.045
        assert first.value_usd == 0.85
        assert first.origin_country == "CN"
        assert first.hts_code == "7318.15.20"

    def test_messy_csv_variant_columns(self):
        """Parse a messy CSV with variant column names, empty rows, section headers."""
        csv_path = FIXTURES_DIR / "messy_bom.csv"
        result = parse_csv(csv_path.read_bytes())

        # Should map variant names correctly
        assert "part_id" in result.detected_columns  # from "Part No."
        assert "description" in result.detected_columns  # from "Desc"
        assert "value_usd" in result.detected_columns  # from "Unit Cost (USD)"
        assert "origin_country" in result.detected_columns  # from "Country"

        # Should have some parseable items and some skipped
        assert len(result.items) > 0
        assert len(result.skipped) > 0

        # Section headers and empty rows should be skipped
        skip_reasons = [s.reason for s in result.skipped]
        assert any("Empty row" in r or "Section header" in r for r in skip_reasons)

        # Actual parts should be parsed
        part_ids = [item.part_id for item in result.items if item.part_id]
        assert "M-001" in part_ids or any("M-001" in (p or "") for p in part_ids)

    def test_csv_string_input(self):
        """CSV parser accepts string input."""
        csv_text = "part_id,description,material\nP1,Steel bolt,Steel\n"
        result = parse_csv(csv_text)
        assert len(result.items) == 1
        assert result.items[0].description == "Steel bolt"

    def test_csv_empty_file(self):
        """Empty CSV returns no items with warning."""
        result = parse_csv(b"")
        assert len(result.items) == 0
        assert len(result.warnings) > 0

    def test_csv_currency_in_value(self):
        """Values with currency symbols ($) are parsed correctly."""
        csv_text = "description,value_usd\nSteel bolt,$1.25\nAluminum bracket,\"$3,500.00\"\n"
        result = parse_csv(csv_text)
        assert len(result.items) == 2
        assert result.items[0].value_usd == 1.25
        assert result.items[1].value_usd == 3500.00


# =====================================================================
# BOM Parser: JSON
# =====================================================================
class TestJSONParser:
    def test_json_array_format(self):
        """Parse a JSON array of BOM items."""
        json_path = FIXTURES_DIR / "bom_items.json"
        result = parse_json(json_path.read_bytes())

        assert len(result.items) == 5
        assert len(result.skipped) == 0

        # Check column mapping from variant names
        assert "part_id" in result.detected_columns  # from "part_number"
        assert "description" in result.detected_columns  # from "name"
        assert "material" in result.detected_columns  # from "materials"
        assert "origin_country" in result.detected_columns  # from "coo"

        first = result.items[0]
        assert first.part_id == "J-001"
        assert "bolt" in first.description.lower()

    def test_json_object_with_items_key(self):
        """Parse a JSON object with an 'items' array."""
        data = {
            "items": [
                {"description": "Steel bolt", "material": "Steel", "country": "CN"},
                {"description": "Rubber gasket", "material": "Rubber", "country": "MX"},
            ]
        }
        result = parse_json(json.dumps(data))
        assert len(result.items) == 2

    def test_json_invalid(self):
        """Invalid JSON returns error."""
        result = parse_json(b"not valid json {{{")
        assert len(result.items) == 0
        assert any("Invalid JSON" in w for w in result.warnings)

    def test_json_empty_array(self):
        """Empty array returns no items."""
        result = parse_json(b"[]")
        assert len(result.items) == 0


# =====================================================================
# BOM Parser: XLSX
# =====================================================================
class TestXLSXParser:
    def test_multi_sheet_xlsx_uses_bom_sheet(self):
        """Parser picks the 'BOM' sheet over the default sheet."""
        xlsx_path = FIXTURES_DIR / "multi_sheet_bom.xlsx"
        result = parse_xlsx(xlsx_path.read_bytes())

        # Should have items from the BOM sheet, not Summary
        assert len(result.items) >= 4  # 5 items minus empty row handling
        assert len(result.skipped) >= 1  # At least the empty row

        # Verify correct columns detected
        assert "part_id" in result.detected_columns
        assert "description" in result.detected_columns
        assert "material" in result.detected_columns

        # Check an item
        part_ids = [item.part_id for item in result.items]
        assert "X-001" in part_ids

    def test_xlsx_invalid_content(self):
        """Invalid XLSX content returns error."""
        result = parse_xlsx(b"not an xlsx file")
        assert len(result.items) == 0
        assert len(result.warnings) > 0


# =====================================================================
# Fuzzy column matching
# =====================================================================
class TestFuzzyColumnMatching:
    def test_exact_match(self):
        """Exact canonical names match correctly."""
        mapping, matched, unmatched = match_columns(
            ["part_id", "description", "material", "value_usd"]
        )
        assert "part_id" in matched
        assert "description" in matched
        assert "material" in matched
        assert "value_usd" in matched
        assert len(unmatched) == 0

    def test_variant_names(self):
        """Common variant names are matched to canonical names."""
        mapping, matched, unmatched = match_columns(
            ["Part No.", "Desc", "Unit Cost (USD)", "Country of Origin"]
        )
        assert mapping.get("Part No.") == "part_id"
        assert mapping.get("Desc") == "description"
        assert mapping.get("Unit Cost (USD)") == "value_usd"
        assert mapping.get("Country of Origin") == "origin_country"

    def test_mixed_case_and_spaces(self):
        """Headers with mixed case and extra spaces match correctly."""
        mapping, matched, _ = match_columns(
            ["  Part_ID  ", "DESCRIPTION", "Material Type", "Weight_Kg"]
        )
        assert "part_id" in matched
        assert "description" in matched
        assert "material" in matched
        assert "weight_kg" in matched

    def test_sku_as_part_id(self):
        """SKU header maps to part_id."""
        mapping, matched, _ = match_columns(["SKU", "Description", "Price"])
        assert mapping.get("SKU") == "part_id"
        assert mapping.get("Price") == "value_usd"

    def test_normalize_header(self):
        """Header normalization strips special chars properly."""
        assert _normalize_header("Unit_Cost_(USD)") == "unit cost (usd)"
        assert _normalize_header("  PART-NUMBER  ") == "part number"
        assert _normalize_header("Wéight_KG") == "weight kg"


# =====================================================================
# Material extraction from descriptions
# =====================================================================
class TestMaterialExtraction:
    def test_steel_fastener(self):
        """'galvanized steel hex bolt' extracts steel and fastener facts."""
        facts = extract_facts_from_description("galvanized steel hex bolt")
        assert facts.get("material_steel") is True
        assert facts.get("is_fastener") is True
        assert facts.get("surface_treatment_galvanized") is True

    def test_copper_wire(self):
        """'copper wire harness assembly' extracts copper and cable facts."""
        facts = extract_facts_from_description("copper wire harness assembly")
        assert facts.get("material_copper") is True
        assert facts.get("product_type_cable") is True

    def test_plastic_housing(self):
        """'ABS plastic enclosure housing' extracts plastic."""
        facts = extract_facts_from_description("ABS plastic enclosure housing")
        assert facts.get("material_plastic") is True

    def test_textile_material(self):
        """'cotton woven fabric panel' extracts textile facts."""
        facts = extract_facts_from_description("cotton woven fabric panel")
        assert facts.get("material_textile") is True

    def test_entailment_propagation(self):
        """Material entailments propagate (e.g., material_steel -> material_metal)."""
        facts = extract_facts_from_description("stainless steel bolt")
        assert facts.get("material_steel") is True
        # Steel entails metal
        assert facts.get("material_metal") is True

    def test_empty_description(self):
        """Empty description returns no facts."""
        facts = extract_facts_from_description("")
        assert len(facts) == 0

    def test_multiple_materials(self):
        """Description with multiple materials extracts all."""
        facts = extract_facts_from_description("steel bolt with rubber gasket and plastic cap")
        assert facts.get("material_steel") is True
        assert facts.get("material_rubber") is True
        assert facts.get("material_plastic") is True


# =====================================================================
# ParsedBOMItem to ProductSpecModel conversion
# =====================================================================
class TestBOMItemConversion:
    def test_item_to_product_spec(self):
        """ParsedBOMItem converts to ProductSpecModel correctly."""
        item = ParsedBOMItem(
            row_number=1,
            part_id="P001",
            description="Steel hex bolt M10x50",
            material="Steel",
            value_usd=0.85,
            origin_country="CN",
            extracted_facts={"material_steel": True, "is_fastener": True},
            inferred_category="fastener",
        )
        spec = item.to_product_spec()
        assert spec.product_category.value == "fastener"
        assert len(spec.materials) >= 1
        assert spec.origin_country == "CN"
        assert spec.declared_value_per_unit == 0.85

    def test_item_without_material_uses_facts(self):
        """When material field is empty, facts-based materials are used."""
        item = ParsedBOMItem(
            row_number=1,
            description="Galvanized steel hex bolt",
            extracted_facts={"material_steel": True, "material_metal": True},
            inferred_category="fastener",
        )
        spec = item.to_product_spec()
        material_names = [m.material for m in spec.materials]
        # Should have inferred materials from facts
        assert len(material_names) > 0


# =====================================================================
# Material percentage computation
# =====================================================================
class TestMaterialPercentages:
    def test_weight_percentages(self):
        """Material percentages by weight are computed correctly."""
        items = [
            ParsedBOMItem(
                row_number=1, description="Steel bolt", material="Steel",
                weight_kg=0.5, value_usd=1.0, quantity=1,
                extracted_facts={}, inferred_category="fastener",
            ),
            ParsedBOMItem(
                row_number=2, description="Aluminum bracket", material="Aluminum",
                weight_kg=0.3, value_usd=2.0, quantity=1,
                extracted_facts={}, inferred_category="other",
            ),
            ParsedBOMItem(
                row_number=3, description="Rubber gasket", material="Rubber",
                weight_kg=0.2, value_usd=0.5, quantity=1,
                extracted_facts={}, inferred_category="other",
            ),
        ]
        pcts = compute_material_percentages(items)

        # Total weight = 1.0, steel = 0.5 -> 50%
        assert pcts["by_weight"]["steel"] == 50.0
        assert pcts["by_weight"]["aluminum"] == 30.0
        assert pcts["by_weight"]["rubber"] == 20.0

        # Total value = 3.5, steel = 1.0 -> ~28.57%
        assert abs(pcts["by_value"]["steel"] - 28.57) < 0.01

    def test_no_weight_data(self):
        """Missing weight data returns empty by_weight dict."""
        items = [
            ParsedBOMItem(
                row_number=1, description="Steel bolt", material="Steel",
                value_usd=1.0,
                extracted_facts={}, inferred_category="fastener",
            ),
        ]
        pcts = compute_material_percentages(items)
        assert pcts["by_weight"] == {}
        assert pcts["by_value"]["steel"] == 100.0


# =====================================================================
# HTS hint resolution
# =====================================================================
class TestHTSHintResolution:
    def test_normalize_hts_code_dotted(self):
        """Dotted HTS codes normalize correctly."""
        assert normalize_hts_code("7318.15.20") == "7318.15.20"
        assert normalize_hts_code("7318.15") == "7318.15"

    def test_normalize_hts_code_numeric(self):
        """Numeric-only HTS codes get formatted."""
        result = normalize_hts_code("73181520")
        assert result == "7318.15.20"

    def test_normalize_hts_code_invalid(self):
        """Invalid codes return None."""
        assert normalize_hts_code("") is None
        assert normalize_hts_code("12") is None

    def test_extract_chapter(self):
        """Chapter extraction from HTS code."""
        assert extract_chapter("7318.15.20") == "73"
        assert extract_chapter("8534.00") == "85"

    def test_extract_heading(self):
        """Heading extraction from HTS code."""
        assert extract_heading("7318.15.20") == "7318"
        assert extract_heading("8534.00") == "8534"

    def test_adjacent_headings(self):
        """Adjacent headings includes same chapter neighbors."""
        adj = adjacent_headings("7318", range_size=2)
        assert "7316" in adj
        assert "7317" in adj
        assert "7318" in adj
        assert "7319" in adj
        assert "7320" in adj
        # Should not cross chapter boundary
        assert all(h[:2] == "73" for h in adj)

    def test_resolve_hts_hint_not_found(self):
        """Declared HTS code not in atoms produces warning and falls back."""
        from potatobacon.law.solver_z3 import PolicyAtom

        atoms = [
            PolicyAtom(
                guard=["chapter_73", "material_steel"],
                outcome={"modality": "OBLIGE", "action": "duty_rate_25_0"},
                source_id="7318.15.20_line",
                section="7318.15.20",
                text="Bolts of iron or steel",
            ),
        ]
        duty_rates = {"7318.15.20_line": 25.0}

        matched, headings, warnings = resolve_hts_hint(
            "9999.99.99", atoms, duty_rates
        )
        assert matched is None
        assert len(warnings) > 0
        assert any("not found" in w for w in warnings)

    def test_resolve_hts_hint_found(self):
        """Declared HTS code matching an atom is resolved correctly."""
        from potatobacon.law.solver_z3 import PolicyAtom

        atoms = [
            PolicyAtom(
                guard=["chapter_73", "material_steel"],
                outcome={"modality": "OBLIGE", "action": "duty_rate_25_0"},
                source_id="7318.15.20_line",
                section="7318.15.20",
                text="Bolts of iron or steel",
            ),
        ]
        duty_rates = {"7318.15.20_line": 25.0}

        matched, headings, warnings = resolve_hts_hint(
            "7318.15.20", atoms, duty_rates
        )
        assert matched is not None
        assert matched.source_id == "7318.15.20_line"
        assert "7318" in headings
        assert len(warnings) == 0


# =====================================================================
# parse_bom_file (format detection)
# =====================================================================
class TestParseBOMFile:
    def test_csv_detection(self):
        """File with .csv extension is parsed as CSV."""
        csv_path = FIXTURES_DIR / "clean_bom.csv"
        result = parse_bom_file(csv_path.read_bytes(), "test.csv")
        assert len(result.items) > 0

    def test_json_detection(self):
        """File with .json extension is parsed as JSON."""
        json_path = FIXTURES_DIR / "bom_items.json"
        result = parse_bom_file(json_path.read_bytes(), "test.json")
        assert len(result.items) > 0

    def test_xlsx_detection(self):
        """File with .xlsx extension is parsed as XLSX."""
        xlsx_path = FIXTURES_DIR / "multi_sheet_bom.xlsx"
        result = parse_bom_file(xlsx_path.read_bytes(), "test.xlsx")
        assert len(result.items) > 0

    def test_unsupported_format(self):
        """Unsupported extension returns error."""
        result = parse_bom_file(b"some data", "test.pdf")
        assert len(result.items) == 0
        assert any("Unsupported" in w for w in result.warnings)


# =====================================================================
# Edge cases: partial rows, unparseable data
# =====================================================================
class TestEdgeCases:
    def test_3_of_10_unparseable(self):
        """Upload where 3 of 10 rows are unparseable — 7 good rows analyzed, 3 reported."""
        csv_text = """part_id,description,material,value_usd,origin_country
P1,Steel bolt M10,Steel,1.25,CN
P2,Aluminum bracket,Aluminum,3.50,TW
,,,,
P3,Rubber gasket,Rubber,0.45,MX
SECTION HEADER,,,,
P4,Copper terminal,Copper,1.80,US
P5,Nylon spacer,Nylon,0.25,CN
P6,Plastic housing,Plastic,2.10,CN
,,,,
P7,Steel washer,Steel,0.35,CN
"""
        result = parse_csv(csv_text)

        # Should have 7 parseable items
        assert len(result.items) == 7

        # Should have 3 skipped (2 empty rows + 1 section header)
        assert len(result.skipped) == 3

        # Verify skipped rows have reasons
        for skip in result.skipped:
            assert skip.reason
            assert skip.row_number > 0

    def test_unicode_descriptions(self):
        """Non-ASCII characters in descriptions are handled."""
        csv_text = 'description,material\n"Tornillo de acero inoxidable",Acero\n"Rondelle en laiton",Laiton\n'
        result = parse_csv(csv_text)
        assert len(result.items) == 2
        assert "inoxidable" in result.items[0].description

    def test_merged_description_field(self):
        """Description field with merged content (commas inside quotes) is handled."""
        csv_text = 'part_id,description,material\nP1,"Steel bolt, galvanized, M10x50",Steel\n'
        result = parse_csv(csv_text)
        assert len(result.items) == 1
        assert "galvanized" in result.items[0].description
