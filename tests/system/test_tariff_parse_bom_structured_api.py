from potatobacon.tariff.bom_ingest import bom_to_text
from potatobacon.tariff.models import StructuredBOMModel


def test_tariff_parse_accepts_structured_bom(system_client):
    payload = {
        "sku_id": "ELEC-1",
        "description": "ABS enclosure with PCB and battery",
        "bom_json": {
            "currency": "USD",
            "items": [
                {
                    "part_id": "HSG",
                    "description": "ABS housing",
                    "material": "ABS",
                    "country_of_origin": "CN",
                },
                {
                    "part_id": "PCB",
                    "description": "controller board",
                    "material": "PCB",
                    "country_of_origin": "CN",
                },
            ],
        },
    }
    response = system_client.post("/api/tariff/parse", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["compiled_facts"].get("product_type_electronics") is True
    assert data["compiled_facts"].get("material_plastic") is True
    assert any(ev["source"] == "bom_json" for ev in data["fact_evidence"][0]["evidence"]) or any(
        ev.get("source") == "bom_json" for item in data["fact_evidence"] for ev in item.get("evidence", [])
    )

    rendered = bom_to_text(StructuredBOMModel(**payload["bom_json"]))  # ensures deterministic render path
    assert "ABS housing" in rendered
