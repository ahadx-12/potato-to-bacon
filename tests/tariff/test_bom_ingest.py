from potatobacon.tariff.bom_ingest import bom_aggregate_material_signals, bom_to_text, parse_bom_csv


def test_parse_and_render_bom_csv():
    csv_text = """part_id,description,material,quantity,unit_cost,country_of_origin
P1,ABS enclosure housing,ABS,2,1.5,CN
P2,controller board,PCB,1,7.0,CN
"""
    bom = parse_bom_csv(csv_text)
    assert len(bom.items) == 2
    assert bom.items[0].part_id == "P1"
    rendered = bom_to_text(bom)
    assert "item0; description=ABS enclosure housing; part=P1; material=ABS; origin=CN; qty=2.0; cost=1.5" in rendered

    aggregates = bom_aggregate_material_signals(bom)
    assert aggregates["dominant_material"] == "abs"
    assert aggregates["primary_origin"] == "CN"
