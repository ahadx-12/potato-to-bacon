"""FastAPI router exposing unit parsing and inference helpers."""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from potatobacon.core.units import analyze_units_map, suggest_units
from potatobacon.parser.units_text import parse_units_text
from potatobacon.units.algebra import UnitParseError
from potatobacon.units.infer import UnitInferenceError, infer_from_equation


router = APIRouter(prefix="/v1/units", tags=["units"])


class UnitsTextReq(BaseModel):
    text: str = Field(default="", description="Multiline name: unit mapping")


class UnitsTextResp(BaseModel):
    units: Dict[str, str]
    warnings: List[str]


@router.post("/parse", response_model=UnitsTextResp)
def parse_units(req: UnitsTextReq) -> UnitsTextResp:
    result = parse_units_text(req.text)
    return UnitsTextResp(units=result.units, warnings=result.warnings)


class UnitsValidateReq(BaseModel):
    units: Dict[str, str]


class UnitDiagnosticModel(BaseModel):
    symbol: str
    code: str
    message: str
    hint: str | None = None


class UnitsValidateResp(BaseModel):
    ok: bool
    canonical: Dict[str, str]
    diagnostics: List[UnitDiagnosticModel]


@router.post("/validate", response_model=UnitsValidateResp)
def validate_units(req: UnitsValidateReq) -> UnitsValidateResp:
    _, canonical, diagnostics = analyze_units_map(req.units)
    diag_models = [
        UnitDiagnosticModel(symbol=d.symbol, code=d.code, message=d.message, hint=d.hint)
        for d in diagnostics
    ]
    return UnitsValidateResp(ok=not diagnostics, canonical=canonical, diagnostics=diag_models)


class UnitsInferReq(BaseModel):
    dsl: str
    known: Dict[str, str] = Field(default_factory=dict)


class UnitsInferResp(BaseModel):
    ok: bool
    units: Dict[str, str]
    trace: List[Dict[str, object]]


@router.post("/infer", response_model=UnitsInferResp)
def infer_units(req: UnitsInferReq) -> UnitsInferResp:
    # Validate known units first so errors surface cleanly.
    _, canonical, diagnostics = analyze_units_map(req.known)
    if diagnostics:
        diag = diagnostics[0]
        raise HTTPException(status_code=422, detail={"symbol": diag.symbol, "message": diag.message})

    try:
        units_map, trace = infer_from_equation(req.dsl, canonical)
    except UnitParseError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)})
    except UnitInferenceError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail={"message": str(exc)})

    return UnitsInferResp(ok=True, units=units_map, trace=trace)


class UnitsSuggestReq(BaseModel):
    variables: List[str]
    system: str = "SI"
    existing: Dict[str, str] = Field(default_factory=dict)


class UnitsSuggestResp(BaseModel):
    suggestions: Dict[str, str]


@router.post("/suggest", response_model=UnitsSuggestResp)
def suggest(req: UnitsSuggestReq) -> UnitsSuggestResp:
    suggestions = suggest_units(req.variables, system=req.system, existing=req.existing)
    return UnitsSuggestResp(suggestions=suggestions)


__all__ = ["router"]

