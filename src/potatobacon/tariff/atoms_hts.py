from __future__ import annotations

from typing import List

from potatobacon.law.solver_z3 import PolicyAtom

DUTY_RATES = {
    "HTS_6404_11_90": 37.5,
    "HTS_6404_19_35": 3.0,
    "HTS_STEEL_BOLT": 6.5,
    "HTS_ALUMINUM_PART": 2.5,
    "HTS_ELECTRONICS_GENERAL": 5.0,
    "HTS_ELECTRONICS_CONNECTOR": 2.0,
    "HTS_ELECTRONICS_PLASTIC_ENCLOSURE": 4.0,
    "HTS_ELECTRONICS_METAL_ENCLOSURE": 7.0,
    "HTS_APPAREL_WOVEN_COTTON": 12.0,
    "HTS_APPAREL_WOVEN_POLY": 8.5,
    "HTS_APPAREL_KNIT_POLY": 7.0,
    "HTS_APPAREL_COATED": 14.0,
}


def tariff_policy_atoms() -> List[PolicyAtom]:
    """Return the canonical HTS policy atoms for footwear tariffs."""

    atoms: list[PolicyAtom] = [
        PolicyAtom(
            guard=[
                "upper_material_textile",
                "outer_sole_material_rubber_or_plastics",
                "surface_contact_rubber_gt_50",
            ],
            outcome={
                "modality": "OBLIGE",
                "action": "apply_HTS_6404_11_90",
                "subject": "footwear",
                "jurisdiction": "US",
            },
            source_id="HTS_6404_11_90",
            statute="US Harmonized Tariff Schedule",
            section="6404.11.90",
            text=(
                "Footwear with textile uppers and rubber or plastic outer soles with more than "
                "50% ground contact from rubber or plastics."
            ),
            rule_type="TARIFF",
            atom_id="HTS_6404_11_90_atom",
        ),
        PolicyAtom(
            guard=[
                "upper_material_textile",
                "outer_sole_material_rubber_or_plastics",
                "surface_contact_textile_gt_50",
            ],
            outcome={
                "modality": "OBLIGE",
                "action": "apply_HTS_6404_19_35",
                "subject": "footwear",
                "jurisdiction": "US",
            },
            source_id="HTS_6404_19_35",
            statute="US Harmonized Tariff Schedule",
            section="6404.19.35",
            text=(
                "Footwear with textile uppers and outer soles where textile materials dominate "
                "ground contact (>50%)."
            ),
            rule_type="TARIFF",
            atom_id="HTS_6404_19_35_atom",
        ),
        PolicyAtom(
            guard=["felt_covering_gt_50"],
            outcome={
                "modality": "PERMIT",
                "action": "treat_felt_as_textile",
                "subject": "outer_sole",
                "jurisdiction": "US",
            },
            source_id="GRI_NOTE_4",
            statute="General Rules of Interpretation",
            section="Note 4(b)",
            text=(
                "Felt applied over more than half of the outer sole counts as textile for "
                "surface-contact determinations."
            ),
            rule_type="TARIFF",
            atom_id="GRI_NOTE_4_atom",
        ),
        PolicyAtom(
            guard=["surface_contact_rubber_gt_50"],
            outcome={
                "modality": "FORBID",
                "action": "surface_contact_textile_gt_50",
                "subject": "outer_sole",
                "jurisdiction": "US",
            },
            source_id="HTS_CONTACT_EXCLUSION",
            statute="US Harmonized Tariff Schedule",
            section="General Exclusions",
            text="Rubber-dominant soles cannot simultaneously be textile-dominant.",
            rule_type="TARIFF",
            atom_id="HTS_CONTACT_EXCLUSION_atom",
        ),
        PolicyAtom(
            guard=["product_type_chassis_bolt", "material_steel"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_6_5",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_STEEL_BOLT",
            statute="HTS",
            section="STEEL_BOLT",
            text="Chassis bolt made of steel ⇒ 6.5% duty",
            rule_type="TARIFF",
            atom_id="HTS_STEEL_BOLT_atom",
        ),
        PolicyAtom(
            guard=["product_type_chassis_bolt", "material_aluminum"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_2_5",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_ALUMINUM_PART",
            statute="HTS",
            section="ALUMINUM_PART",
            text="Chassis bolt made of aluminum ⇒ 2.5% duty",
            rule_type="TARIFF",
            atom_id="HTS_ALUMINUM_PART_atom",
        ),
        PolicyAtom(
            guard=["product_type_electronics"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_5_0",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_ELECTRONICS_GENERAL",
            statute="HTS",
            section="ELECTRONICS_GENERAL",
            text="General electronics classification baseline.",
            rule_type="TARIFF",
            atom_id="HTS_ELECTRONICS_GENERAL_atom",
        ),
        PolicyAtom(
            guard=["electronics_cable_or_connector"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_2_0",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_ELECTRONICS_CONNECTOR",
            statute="HTS",
            section="ELECTRONICS_CONNECTOR",
            text="Electronics classified primarily as cables/connectors.",
            rule_type="TARIFF",
            atom_id="HTS_ELECTRONICS_CONNECTOR_atom",
        ),
        PolicyAtom(
            guard=["electronics_enclosure", "material_plastic"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_4_0",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_ELECTRONICS_PLASTIC_ENCLOSURE",
            statute="HTS",
            section="ELECTRONICS_PLASTIC_ENCLOSURE",
            text="Electronics enclosures that are predominantly plastic.",
            rule_type="TARIFF",
            atom_id="HTS_ELECTRONICS_PLASTIC_ENCLOSURE_atom",
        ),
        PolicyAtom(
            guard=["electronics_enclosure", "material_steel"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_7_0",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_ELECTRONICS_METAL_ENCLOSURE",
            statute="HTS",
            section="ELECTRONICS_METAL_ENCLOSURE",
            text="Electronics enclosures made with metal housings.",
            rule_type="TARIFF",
            atom_id="HTS_ELECTRONICS_METAL_ENCLOSURE_atom",
        ),
        PolicyAtom(
            guard=["product_type_apparel_textile", "textile_woven", "fiber_cotton_dominant"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_12_0",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_APPAREL_WOVEN_COTTON",
            statute="HTS",
            section="APPAREL_WOVEN_COTTON",
            text="Woven apparel with dominant cotton fibers.",
            rule_type="TARIFF",
            atom_id="HTS_APPAREL_WOVEN_COTTON_atom",
        ),
        PolicyAtom(
            guard=["product_type_apparel_textile", "textile_woven", "fiber_polyester_dominant"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_8_5",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_APPAREL_WOVEN_POLY",
            statute="HTS",
            section="APPAREL_WOVEN_POLY",
            text="Woven apparel dominated by polyester fibers.",
            rule_type="TARIFF",
            atom_id="HTS_APPAREL_WOVEN_POLY_atom",
        ),
        PolicyAtom(
            guard=["product_type_apparel_textile", "textile_knit", "fiber_polyester_dominant"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_7_0",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_APPAREL_KNIT_POLY",
            statute="HTS",
            section="APPAREL_KNIT_POLY",
            text="Knit apparel with polyester dominance.",
            rule_type="TARIFF",
            atom_id="HTS_APPAREL_KNIT_POLY_atom",
        ),
        PolicyAtom(
            guard=["product_type_apparel_textile", "has_coating_or_lamination"],
            outcome={
                "modality": "OBLIGE",
                "action": "duty_rate_14_0",
                "subject": "import_duty",
                "jurisdiction": "US",
            },
            source_id="HTS_APPAREL_COATED",
            statute="HTS",
            section="APPAREL_COATED",
            text="Apparel with coating or lamination layers.",
            rule_type="TARIFF",
            atom_id="HTS_APPAREL_COATED_atom",
        ),
    ]
    return atoms
