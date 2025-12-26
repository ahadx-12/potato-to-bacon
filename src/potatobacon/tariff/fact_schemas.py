from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FactDef:
    type: Literal["bool", "categorical", "numeric", "dict", "list"]
    question: str | None = None
    values: list[str] | None = None
    unit: str | None = None


ELECTRONICS_FACTS = {
    "copper_conductor": FactDef(type="bool"),
    "insulation_material": FactDef(type="categorical", values=["PVC", "rubber", "TPE", "silicone"]),
    "insulation_thickness_mm": FactDef(type="numeric", unit="mm"),
    "voltage_rating_v": FactDef(type="numeric", unit="V"),
    "current_rating_a": FactDef(type="numeric", unit="A"),
    "data_transmission": FactDef(type="bool"),
    "shielded": FactDef(type="bool"),
    "assembled": FactDef(type="bool"),
    "assembly_operations": FactDef(type="list"),
    "pcb_present": FactDef(type="bool"),
    "connector_count": FactDef(type="numeric"),
    "ul_listed": FactDef(type="bool"),
    "ce_marked": FactDef(type="bool"),
    "rohs_compliant": FactDef(type="bool"),
    "connector_type": FactDef(type="categorical", values=["usb", "hdmi", "ethernet", "other"]),
    "cable_length_m": FactDef(type="numeric", unit="m"),
    "wire_gauge_awg": FactDef(type="numeric", unit="AWG"),
    "connector_material": FactDef(type="categorical", values=["metal", "plastic", "mixed"]),
}

APPAREL_FACTS = {
    "textile_content_pct": FactDef(type="dict"),
    "outer_surface_material": FactDef(type="categorical"),
    "lining_material": FactDef(type="categorical"),
    "knit_or_woven": FactDef(type="categorical", values=["knit", "woven", "other"]),
    "garment_type": FactDef(type="categorical", values=["top", "bottom", "outerwear", "underwear"]),
    "gender": FactDef(type="categorical", values=["men", "women", "unisex", "children"]),
    "pockets": FactDef(type="bool"),
    "buttons": FactDef(type="numeric"),
    "zipper": FactDef(type="bool"),
    "sleeve_length": FactDef(type="categorical", values=["short", "long", "sleeveless"]),
    "collar_present": FactDef(type="bool"),
    "elastic_waist": FactDef(type="bool"),
}

MACHINERY_FACTS = {
    "self_propelled": FactDef(type="bool"),
    "electric_motor": FactDef(type="bool"),
    "hydraulic": FactDef(type="bool"),
    "power_output_kw": FactDef(type="numeric", unit="kW"),
    "has_computer_control": FactDef(type="bool"),
    "material_processing": FactDef(type="bool"),
    "weight_kg": FactDef(type="numeric", unit="kg"),
    "capacity": FactDef(type="numeric"),
    "engine_type": FactDef(type="categorical", values=["diesel", "gasoline", "electric", "hybrid"]),
    "operating_voltage_v": FactDef(type="numeric", unit="V"),
}

FURNITURE_FACTS = {
    "primary_material": FactDef(type="categorical", values=["wood", "metal", "plastic", "upholstered"]),
    "wood_type": FactDef(type="categorical", values=["hardwood", "softwood", "particleboard", "MDF"]),
    "seating": FactDef(type="bool"),
    "storage": FactDef(type="bool"),
    "lighting": FactDef(type="bool"),
    "upholstered_pct": FactDef(type="numeric"),
    "adjustable": FactDef(type="bool"),
    "foldable": FactDef(type="bool"),
    "assembly_required": FactDef(type="bool"),
    "surface_finish": FactDef(type="categorical", values=["painted", "varnished", "laminate", "unfinished"]),
}

PLASTICS_FACTS = {
    "polymer_type": FactDef(type="categorical"),
    "recycled_content_pct": FactDef(type="numeric"),
    "is_rigid": FactDef(type="bool"),
    "is_film": FactDef(type="bool"),
}
