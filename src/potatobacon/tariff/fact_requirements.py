from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class FactRequirement:
    """Deterministic prompt/evidence guidance for a fact key."""

    fact_key: str
    question_template: str
    evidence_types: Sequence[str]
    measurement_hint: str | None = None

    def render_question(self) -> str:
        return self.question_template.format(fact_key=self.fact_key)


class FactRequirementRegistry:
    """Local registry of fact -> question/evidence templates."""

    def __init__(self) -> None:
        self._registry: Dict[str, FactRequirement] = {
            "origin_country": FactRequirement(
                fact_key="origin_country",
                question_template="What is the country of origin for the SKU and its key components?",
                evidence_types=[
                    "certificate_pdf",
                    "commercial_invoice",
                    "bom_csv_origin_column",
                    "supplier_declaration",
                ],
                measurement_hint="Use ISO-3166-1 alpha-2 codes per component if multi-origin.",
            ),
            "origin_country_export": FactRequirement(
                fact_key="origin_country_export",
                question_template="Confirm the export country's origin treatment for this SKU.",
                evidence_types=[
                    "certificate_pdf",
                    "customs_ruling_pdf",
                    "commercial_invoice",
                    "bom_csv_origin_column",
                ],
            ),
            "electronics_voltage_rating_known": FactRequirement(
                fact_key="electronics_voltage_rating_known",
                question_template="What voltage/current rating applies to the cable or connector assembly?",
                evidence_types=["connector_spec_sheet", "safety_datasheet_pdf", "bom_csv"],
                measurement_hint="Provide rated volts/amps from spec or certification.",
            ),
        "electronics_insulated_conductors": FactRequirement(
            fact_key="electronics_insulated_conductors",
            question_template="Are the conductors insulated/jacketed as part of the assembly?",
            evidence_types=[
                "harness_cross_section_photo",
                "material_declaration",
                "bom_csv",
                "spec_sheet",
                "manufacturer_datasheet_pdf",
                "lab_test_report",
                "product_photo_label",
            ],
        ),
            "electronics_has_connectors": FactRequirement(
                fact_key="electronics_has_connectors",
                question_template="Does the assembly terminate with defined connectors?",
                evidence_types=["connector_drawing_pdf", "harness_pinout", "bom_csv"],
            ),
            "electronics_is_cable_assembly": FactRequirement(
                fact_key="electronics_is_cable_assembly",
                question_template="Is this SKU sold as a complete cable or harness assembly?",
                evidence_types=["harness_assembly_drawing", "bom_csv_section_highlight", "photo"],
            ),
            "surface_contact_textile_gt_50": FactRequirement(
                fact_key="surface_contact_textile_gt_50",
                question_template="Does textile material cover more than 50% of the outsole contact surface?",
                evidence_types=["coverage_photo", "measurement_diagram", "lab_certificate_pdf"],
                measurement_hint="Include coverage measurement method or diagram.",
            ),
            "surface_contact_rubber_gt_50": FactRequirement(
                fact_key="surface_contact_rubber_gt_50",
                question_template="Is rubber or plastic covering more than 50% of the outsole contact surface?",
                evidence_types=["coverage_photo", "measurement_diagram", "material_certificate"],
                measurement_hint="Document % coverage with measurement notes.",
            ),
        }
        self._prefix_registry: List[tuple[str, FactRequirement]] = [
            (
                "origin_country_",
                FactRequirement(
                    fact_key="origin_country_*",
                    question_template="What is the country of origin for {fact_key}?",
                    evidence_types=[
                        "certificate_pdf",
                        "commercial_invoice",
                        "bom_csv_origin_column",
                        "supplier_statement",
                    ],
                    measurement_hint="Use ISO-3166-1 alpha-2 codes.",
                ),
            ),
            (
                "material_",
                FactRequirement(
                    fact_key="material_*",
                    question_template="Confirm the material evidence for '{fact_key}'.",
                    evidence_types=["material_certificate_pdf", "bom_csv", "spec_sheet_pdf", "photo"],
                    measurement_hint="Include % composition or weight share where applicable.",
                ),
            ),
            (
                "electronics_",
                FactRequirement(
                    fact_key="electronics_*",
                    question_template="Provide design evidence to resolve electronics attribute '{fact_key}'.",
                    evidence_types=["connector_spec_sheet", "assembly_drawing", "bom_csv", "photo"],
                ),
            ),
            (
                "fiber_",
                FactRequirement(
                    fact_key="fiber_*",
                    question_template="Confirm fiber composition for '{fact_key}'.",
                    evidence_types=["lab_certificate_pdf", "composition_test", "bom_csv"],
                    measurement_hint="Share fiber % split and lab/QA source.",
                ),
            ),
        ]

    def describe(self, fact_key: str) -> FactRequirement:
        if fact_key in self._registry:
            return self._registry[fact_key]
        for prefix, template in self._prefix_registry:
            if fact_key.startswith(prefix):
                return FactRequirement(
                    fact_key=fact_key,
                    question_template=template.question_template,
                    evidence_types=list(template.evidence_types),
                    measurement_hint=template.measurement_hint,
                )
        return FactRequirement(
            fact_key=fact_key,
            question_template="Provide a definitive value for '{fact_key}'.",
            evidence_types=["bom_csv", "spec_sheet_pdf", "photo", "lab_certificate_pdf"],
        )
