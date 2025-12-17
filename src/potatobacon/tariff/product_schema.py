"""Canonical product schema for tariff optimization workflows."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProductCategory(str, Enum):
    """High-level product categories used to drive rule selection."""

    FOOTWEAR = "footwear"
    FASTENER = "fastener"
    ELECTRONICS = "electronics"
    TEXTILE = "textile"
    APPAREL_TEXTILE = "apparel_textile"
    FURNITURE = "furniture"
    OTHER = "other"


class ManufacturingProcess(str, Enum):
    """Enumerates typical manufacturing processes relevant to tariff rules."""

    CAST = "cast"
    FORGED = "forged"
    MACHINED = "machined"
    ASSEMBLED = "assembled"
    WELDED = "welded"
    MOLDED = "molded"


class MaterialBreakdown(BaseModel):
    """Describes a material component within a product."""

    component: str = Field(..., description="Component name (e.g., upper, sole, housing)")
    material: str = Field(..., description="Primary material (e.g., steel, textile, plastic)")
    percent_by_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Percent of total weight attributable to this material.",
    )
    percent_by_value: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Percent of declared value attributable to this material.",
    )

    model_config = ConfigDict(extra="forbid")


class Dimensions(BaseModel):
    """Dimension data in millimeters to support size-based thresholds."""

    length_mm: Optional[float] = Field(default=None, ge=0.0)
    width_mm: Optional[float] = Field(default=None, ge=0.0)
    height_mm: Optional[float] = Field(default=None, ge=0.0)
    diameter_mm: Optional[float] = Field(default=None, ge=0.0)
    thickness_mm: Optional[float] = Field(default=None, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class SurfaceCoverage(BaseModel):
    """Represents surface coating/coverage by material and percentage."""

    material: str = Field(..., description="Surface material (e.g., textile, rubber, leather)")
    percent_coverage: float = Field(..., ge=0.0, le=100.0)
    coating_type: Optional[str] = Field(
        default=None, description="Coating or overlay type (e.g., felt, paint, plating)"
    )

    model_config = ConfigDict(extra="forbid")


class OriginInput(BaseModel):
    """Tracks origin and transformation details for each component."""

    component: str
    country_of_origin: str
    transformation: Optional[str] = Field(
        default=None, description="Manufacturing steps applied to this component"
    )

    model_config = ConfigDict(extra="forbid")


class ProductSpecModel(BaseModel):
    """Universal product specification accepted by the tariff optimizer."""

    product_category: ProductCategory = Field(
        ..., description="High-level category steering rules and mutations"
    )
    materials: List[MaterialBreakdown] = Field(
        default_factory=list,
        description="List of materials by component with percentages",
    )
    manufacturing_process: Optional[ManufacturingProcess] = Field(
        default=None, description="Primary manufacturing process"
    )
    use_function: Optional[str] = Field(
        default=None, description="Primary use or function (e.g., fastener, housing, conductive)"
    )
    has_pcb: Optional[bool] = Field(
        default=None,
        description="Indicates whether the product contains a printed circuit board",
    )
    is_cable_or_connector: Optional[bool] = Field(
        default=None,
        description="True when the primary function is a cable or connector",
    )
    is_enclosure_or_housing: Optional[bool] = Field(
        default=None,
        description="True when the product is an enclosure or housing for electronics",
    )
    contains_battery: Optional[bool] = Field(
        default=None, description="Indicates presence of a battery or cell"
    )
    is_knit: Optional[bool] = Field(default=None, description="Garment uses knit construction")
    is_woven: Optional[bool] = Field(default=None, description="Garment uses woven construction")
    fiber_cotton_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Cotton content percentage"
    )
    fiber_polyester_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Polyester content percentage"
    )
    fiber_nylon_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Nylon content percentage"
    )
    has_coating_or_lamination: Optional[bool] = Field(
        default=None, description="Whether the textile has coating/lamination"
    )
    gender_or_age: Optional[str] = Field(
        default=None, description="Intended wearer (e.g., men, women, children)"
    )
    origin_country: Optional[str] = Field(
        default=None, description="Country of origin; stored as provided"
    )
    export_country: Optional[str] = Field(
        default=None, description="Exporting country code"
    )
    import_country: Optional[str] = Field(
        default=None, description="Importing country code (defaults to US when origin/export known)"
    )
    dimensions: Optional[Dimensions] = None
    surface_coverage: List[SurfaceCoverage] = Field(
        default_factory=list,
        description="Surface material coverage percentages",
    )
    country_of_origin_inputs: List[OriginInput] = Field(
        default_factory=list, description="Origin details for each component"
    )
    declared_value_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Declared customs value per unit"
    )
    annual_volume: Optional[int] = Field(
        default=None, ge=0, description="Expected annual shipment volume"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("materials", mode="after")
    def _validate_materials(cls, values: List[MaterialBreakdown]) -> List[MaterialBreakdown]:
        # Ensure deterministic ordering for downstream fact compilation.
        return sorted(values, key=lambda item: (item.component.lower(), item.material.lower()))

    @field_validator("surface_coverage", mode="after")
    def _validate_surface_coverage(
        cls, values: List[SurfaceCoverage]
    ) -> List[SurfaceCoverage]:
        return sorted(values, key=lambda item: (item.material.lower(), item.coating_type or ""))

    @field_validator("country_of_origin_inputs", mode="after")
    def _validate_origin_inputs(cls, values: List[OriginInput]) -> List[OriginInput]:
        return sorted(values, key=lambda item: (item.component.lower(), item.country_of_origin.lower()))

    @field_validator("import_country", mode="before")
    def _default_import_country(cls, value: str | None, values: dict) -> str | None:
        raw_values = values
        if hasattr(values, "data"):
            raw_values = values.data
        provided_origin = raw_values.get("origin_country") or raw_values.get("export_country") if isinstance(raw_values, dict) else None
        if value is None and provided_origin:
            return "US"
        return value
