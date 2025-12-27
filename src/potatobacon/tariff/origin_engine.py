from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, List, Optional

from potatobacon.cale.engine import CALEEngine
from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.models import ProductGraph, ProductGraphComponent, ProductOperation


DEFAULT_RVC_THRESHOLD = Decimal("60.0")
DEFAULT_RVC_MARGIN = Decimal("2.0")
_RVC_PRECISION = Decimal("0.0001")


def _to_decimal(value: float | int | str | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _quantize(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return value.quantize(_RVC_PRECISION, rounding=ROUND_HALF_UP)


def _normalize_hts_code(raw: str | None) -> str:
    if not raw:
        return ""
    return "".join(ch for ch in str(raw) if ch.isdigit())


def _heading(hts_code: str) -> str:
    return hts_code[:4] if len(hts_code) >= 4 else ""


def _subheading(hts_code: str) -> str:
    return hts_code[:6] if len(hts_code) >= 6 else ""


def _component_originating(
    component: ProductGraphComponent,
    declared_origin_country: str | None,
) -> bool:
    if component.is_originating_material is not None:
        return bool(component.is_originating_material)
    if declared_origin_country and component.origin_country:
        return component.origin_country.upper() == declared_origin_country.upper()
    return False


def _component_value(
    component: ProductGraphComponent,
    adjusted_value: Decimal | None,
) -> Decimal | None:
    component_value = _to_decimal(component.component_value)
    if component_value is not None:
        return component_value
    if adjusted_value is None:
        return None
    share = _to_decimal(component.value_share)
    if share is None:
        return None
    return adjusted_value * share


@dataclass(slots=True)
class TariffShiftComponent:
    name: str
    component_hts: str | None
    heading_change: bool
    subheading_change: bool
    component_heading: str | None = None
    component_subheading: str | None = None


@dataclass(slots=True)
class TariffShiftResult:
    final_hts: str
    final_heading: str
    final_subheading: str
    components: list[TariffShiftComponent] = field(default_factory=list)

    @property
    def has_tariff_shift(self) -> bool:
        return any(comp.heading_change or comp.subheading_change for comp in self.components)


@dataclass(slots=True)
class RVCResult:
    adjusted_value: Decimal | None
    originating_value: Decimal
    non_originating_value: Decimal
    build_down: Decimal | None
    build_up: Decimal | None
    non_originating_share: Decimal | None


@dataclass(slots=True)
class OriginAnalysisResult:
    tariff_shift: TariffShiftResult | None
    rvc: RVCResult | None
    substantial_transformation: bool
    manufacturing_steps: list[str]
    missing_facts: list[str]
    conflict_intensity: float | None = None
    requires_review: bool = False
    rvc_threshold: Decimal = DEFAULT_RVC_THRESHOLD
    rvc_margin: Decimal = DEFAULT_RVC_MARGIN

    def facts_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.tariff_shift:
            payload["origin_tariff_shift"] = self.tariff_shift.has_tariff_shift
        if self.rvc:
            payload["origin_rvc_build_down"] = float(self.rvc.build_down) if self.rvc.build_down is not None else None
            payload["origin_rvc_build_up"] = float(self.rvc.build_up) if self.rvc.build_up is not None else None
            payload["origin_non_originating_share"] = (
                float(self.rvc.non_originating_share) if self.rvc.non_originating_share is not None else None
            )
        payload["origin_substantial_transformation"] = self.substantial_transformation
        if self.conflict_intensity is not None:
            payload["origin_conflict_intensity"] = float(self.conflict_intensity)
        payload["origin_requires_review"] = self.requires_review
        return payload


def tariff_shift_test(product_graph: ProductGraph, final_hts: str | None) -> TariffShiftResult | None:
    normalized_final = _normalize_hts_code(final_hts)
    if not normalized_final:
        return None

    final_heading = _heading(normalized_final)
    final_subheading = _subheading(normalized_final)
    components: list[TariffShiftComponent] = []
    for component in product_graph.components:
        component_hts = _normalize_hts_code(component.hts_code)
        if not component_hts:
            continue
        component_heading = _heading(component_hts)
        component_subheading = _subheading(component_hts)
        heading_change = bool(component_heading and component_heading != final_heading)
        subheading_change = bool(component_subheading and component_subheading != final_subheading)
        components.append(
            TariffShiftComponent(
                name=component.name,
                component_hts=component.hts_code,
                heading_change=heading_change,
                subheading_change=subheading_change,
                component_heading=component_heading or None,
                component_subheading=component_subheading or None,
            )
        )
    return TariffShiftResult(
        final_hts=normalized_final,
        final_heading=final_heading,
        final_subheading=final_subheading,
        components=components,
    )


def compute_rvc(
    product_graph: ProductGraph,
    adjusted_value: float | int | Decimal | None,
    *,
    declared_origin_country: str | None = None,
) -> RVCResult | None:
    adjusted = _to_decimal(adjusted_value)
    if adjusted is None or adjusted <= 0:
        return None

    originating_value = Decimal("0.0")
    non_originating_value = Decimal("0.0")
    for component in product_graph.components:
        value = _component_value(component, adjusted)
        if value is None:
            continue
        if _component_originating(component, declared_origin_country):
            originating_value += value
        else:
            non_originating_value += value

    if originating_value == 0 and non_originating_value == 0:
        return None

    build_down = ((adjusted - non_originating_value) / adjusted) * Decimal("100")
    build_up = (originating_value / adjusted) * Decimal("100")
    non_originating_share = non_originating_value / adjusted
    return RVCResult(
        adjusted_value=_quantize(adjusted),
        originating_value=_quantize(originating_value) or Decimal("0.0"),
        non_originating_value=_quantize(non_originating_value) or Decimal("0.0"),
        build_down=_quantize(build_down),
        build_up=_quantize(build_up),
        non_originating_share=_quantize(non_originating_share),
    )


def manufacturing_transformations(ops: Iterable[ProductOperation]) -> list[str]:
    industrial_keywords = {
        "assembly",
        "extrusion",
        "drawing",
        "welding",
        "casting",
        "forging",
        "molding",
        "braiding",
        "crimping",
        "soldering",
        "machining",
        "fabrication",
        "lamination",
        "spinning",
    }
    simple_keywords = {"mixing", "dilution", "repacking", "labeling", "sorting"}

    recognized: list[str] = []
    for op in ops:
        step_lower = op.step.lower()
        if any(keyword in step_lower for keyword in simple_keywords):
            continue
        if any(keyword in step_lower for keyword in industrial_keywords):
            recognized.append(op.step)
    return sorted(set(recognized), key=str.lower)


def _origin_conflict_intensity(
    declared_origin_country: str,
    non_originating_share: Decimal,
    *,
    has_tariff_shift: bool,
) -> float:
    engine = CALEEngine()
    declared_text = (
        f"Importer MUST accept origin claim IF declared origin is {declared_origin_country}."
    )
    non_origin_pct = (non_originating_share * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    shift_clause = "no tariff shift" if not has_tariff_shift else "tariff shift documented"
    contradiction_text = (
        "Importer MUST NOT accept origin claim IF non-originating materials exceed "
        f"{non_origin_pct} percent AND {shift_clause}."
    )
    left = engine._ensure_rule(
        {
            "text": declared_text,
            "jurisdiction": "Origin",
            "statute": "Origin Claim",
            "section": "Declared Origin",
            "enactment_year": 2024,
        },
        fallback_id="ORIGIN_DECLARED",
    )
    right = engine._ensure_rule(
        {
            "text": contradiction_text,
            "jurisdiction": "Origin",
            "statute": "Origin Claim",
            "section": "Conflict Check",
            "enactment_year": 2024,
        },
        fallback_id="ORIGIN_CONFLICT",
    )
    analysis, _, _ = engine._prepare_analysis(left, right)
    return float(analysis.CI)


def evaluate_origin_analysis(
    *,
    product_graph: ProductGraph,
    final_hts: str | None,
    adjusted_value: float | int | Decimal | None,
    declared_origin_country: str | None,
    rvc_threshold: Decimal = DEFAULT_RVC_THRESHOLD,
    rvc_margin: Decimal = DEFAULT_RVC_MARGIN,
) -> OriginAnalysisResult:
    tariff_shift = tariff_shift_test(product_graph, final_hts)
    rvc_result = compute_rvc(product_graph, adjusted_value, declared_origin_country=declared_origin_country)
    transformations = manufacturing_transformations(product_graph.ops)
    substantial_transformation = bool(tariff_shift and tariff_shift.has_tariff_shift and transformations)

    missing_facts: list[str] = []
    if not product_graph.components:
        missing_facts.append("sub_supplier_cert_origin")
    if not product_graph.ops:
        missing_facts.append("manufacturing_step_logs")
    if rvc_result is None:
        missing_facts.append("labor_cost_summary")
    else:
        if any(
            comp.origin_country is None and comp.is_originating_material is None
            for comp in product_graph.components
        ):
            missing_facts.append("sub_supplier_cert_origin")

    conflict_intensity = None
    requires_review = False
    if (
        declared_origin_country
        and rvc_result
        and rvc_result.non_originating_share is not None
        and rvc_result.non_originating_share >= Decimal("0.9")
        and not (tariff_shift and tariff_shift.has_tariff_shift)
    ):
        conflict_intensity = _origin_conflict_intensity(
            declared_origin_country,
            rvc_result.non_originating_share,
            has_tariff_shift=tariff_shift.has_tariff_shift if tariff_shift else False,
        )
        requires_review = conflict_intensity >= 0.45

    return OriginAnalysisResult(
        tariff_shift=tariff_shift,
        rvc=rvc_result,
        substantial_transformation=substantial_transformation,
        manufacturing_steps=transformations,
        missing_facts=sorted(set(missing_facts)),
        conflict_intensity=conflict_intensity,
        requires_review=requires_review,
        rvc_threshold=rvc_threshold,
        rvc_margin=rvc_margin,
    )


def build_origin_policy_atoms(manifest_hash: str | None = None) -> list[PolicyAtom]:
    metadata = {"manifest_hash": manifest_hash, "chapter": "USMCA_CH4"} if manifest_hash else {"chapter": "USMCA_CH4"}
    return [
        PolicyAtom(
            guard=["origin_substantial_transformation"],
            outcome={"modality": "PERMIT", "action": "origin_fta_eligible"},
            source_id="ORIGIN_SUBSTANTIAL_TRANSFORMATION",
            statute="USMCA",
            section="Chapter 4",
            text="Origin rules apply when substantial transformation is achieved.",
            modality="PERMIT",
            action="origin_fta_eligible",
            rule_type="ORIGIN",
            metadata=metadata,
        )
    ]
