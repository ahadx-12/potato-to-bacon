from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from potatobacon.api.routes_units import router as units_router
from potatobacon.cale.bootstrap import CALEServices, build_services
from potatobacon.cale.engine import CALEEngine
from potatobacon.cale.types import LegalRule
from potatobacon.codegen.reference import generate_numpy
from potatobacon.core.units import analyze_units_map
from potatobacon.manifest.store import ComputationManifest
from potatobacon.parser.dsl_parser import parse_dsl
from potatobacon.semantics.canonicalizer import canonicalize
from potatobacon.semantics.schema_builder import build_theory_schema
from potatobacon.storage import load_manifest as load_persisted_manifest
from potatobacon.storage import save_code, save_manifest, save_schema
from potatobacon.validation.pipeline import validate_all
from potatobacon.units.infer import evaluate_equation_dimensions, UnitInferenceError

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        services = build_services()
        app.state.cale = services
        app.state.cale_engine = CALEEngine(services)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("CALE bootstrap failed")
        app.state.cale = None
        app.state.cale_engine = None
    yield
    app.state.cale = None
    app.state.cale_engine = None


app = FastAPI(title="potato-to-bacon API", version="v1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(units_router)

# Conditionally mount docs/examples if folders exist (avoids startup errors)
if Path("docs").exists():
    app.mount("/static/docs", StaticFiles(directory="docs", html=True), name="docs")

if Path("examples").exists():
    app.mount("/static/examples", StaticFiles(directory="examples", html=False), name="examples")

cale_static_dir = Path("static/cale")
if cale_static_dir.exists():
    app.mount("/demo/cale", StaticFiles(directory=cale_static_dir, html=True), name="cale-demo")

# -----------------------------------------------------------------------------
# UI mount (safe if /web missing)
# -----------------------------------------------------------------------------
# Force absolute path to /app/web (works in Docker/Railway)
web_dir = Path("/app/web")


if web_dir.exists():
    app.mount("/ui", StaticFiles(directory=web_dir, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        """Redirect root to the UI page."""
        return RedirectResponse(url="/ui/")
else:
    print(f"⚠️ Warning: UI directory not found at {web_dir}, skipping UI mount.")


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------
class TranslateReq(BaseModel):
    dsl: str
    domain: str = "classical"


class TranslateResp(BaseModel):
    success: bool
    expression: str
    canonical: str


class ValidateReq(BaseModel):
    dsl: str
    domain: str = "classical"
    units: Dict[str, str] = Field(default_factory=dict)
    result_unit: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    pde_space_vars: List[str] = Field(default_factory=list)
    pde_time_var: Optional[str] = None
    checks: List[str] = Field(default_factory=list)


class ValidateResp(BaseModel):
    ok: bool
    report: Dict[str, Any]


class SchemaReq(BaseModel):
    name: str
    domain: str
    dsl: str
    units: Dict[str, str] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)


class SchemaResp(BaseModel):
    """Avoids naming conflict with Pydantic's schema() method."""
    model_config = ConfigDict(populate_by_name=True)
    schema_json_payload: Dict[str, Any] = Field(
        serialization_alias="schema_json",
        validation_alias="schema_json",
    )
    canonical: str

    @property
    def schema_json(self) -> Dict[str, Any]:
        return self.schema_json_payload


class CodegenReq(BaseModel):
    dsl: str
    name: str = "compute"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodegenResp(BaseModel):
    code: str


class ManifestReq(BaseModel):
    dsl: str
    domain: str
    units: Dict[str, str] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    result_unit: Optional[str] = None
    checks: List[str] = Field(default_factory=list)
    pde_space_vars: List[str] = Field(default_factory=list)
    pde_time_var: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ManifestResp(BaseModel):
    manifest_hash: str
    code_digest: str


class RuleInput(BaseModel):
    text: str = Field(..., min_length=3)
    jurisdiction: str
    statute: str
    section: str
    enactment_year: int


class ConflictRequest(BaseModel):
    rule1: RuleInput
    rule2: RuleInput


class SuggestionItem(BaseModel):
    condition: str
    justification: Dict[str, float]
    estimated_ccs: float
    suggested_text: str


class CCScores(BaseModel):
    textualist: float
    living: float
    pragmatic: float


class SuggestedAmendmentSummary(BaseModel):
    condition: str
    impact: float
    justification: str


class SuggestionResponse(BaseModel):
    precedent_count: int
    candidates_considered: int
    suggestions: List[SuggestionItem]
    best: SuggestionItem


def _ensure_bootstrap() -> CALEServices:
    services: CALEServices | None = getattr(app.state, "cale", None)
    engine: CALEEngine | None = getattr(app.state, "cale_engine", None)
    if services is None or engine is None:
        try:
            services = build_services()
            engine = CALEEngine(services)
        except Exception as exc:  # pragma: no cover - defensive bootstrap
            logger.exception("CALE bootstrap failed during request")
            raise HTTPException(status_code=503, detail="CALE not initialised") from exc
        app.state.cale = services
        app.state.cale_engine = engine
    if not services.corpus:
        raise HTTPException(status_code=503, detail="CALE corpus empty / suggester not fitted")
    return services


def _cale_services() -> CALEServices:
    return _ensure_bootstrap()


def _cale_engine() -> CALEEngine:
    _ensure_bootstrap()
    engine: CALEEngine | None = getattr(app.state, "cale_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="CALE engine not initialised")
    return engine


def _parse_and_populate_rule(inp: RuleInput, services: CALEServices) -> LegalRule:
    metadata = inp.model_dump(exclude={"text"})
    metadata.setdefault("id", None)
    rule = services.parser.parse(inp.text, metadata)
    return services.feature_engine.populate(rule)


# -----------------------------------------------------------------------------
# Health + Info Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    services = _ensure_bootstrap()
    return {"ok": True, "status": "ok", "corpus": len(services.corpus)}


@app.get("/v1/health")
def legacy_health() -> Dict[str, Any]:
    return health()


@app.get("/v1/info")
def info() -> Dict[str, Any]:
    return {
        "version": "0.1.0",
        "git_sha": os.getenv("GIT_COMMIT"),
        "dsl_features": [
            "return-based DSL",
            "sympy-backed expression parsing",
            "operators: +, -, *, /, **, pow(int)",
            "derivatives: d(expr, *vars), d2(expr, var)",
            "equations with ==",
        ],
        "validators": ["dimensional_fast", "constraints", "relativistic", "pde_class"],
    }


# -----------------------------------------------------------------------------
# Core API Endpoints
# -----------------------------------------------------------------------------
@app.post("/v1/translate", response_model=TranslateResp)
def translate(req: TranslateReq) -> TranslateResp:
    try:
        expr = parse_dsl(req.dsl)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(e),
                "hint": "Try examples like 'E = m*c^2', 'E == m*c**2', or 'return E - m*c**2'.",
            },
        )
    canon = canonicalize(expr)
    return TranslateResp(success=True, expression=str(expr), canonical=canon.canonical_str)


@app.post("/v1/validate", response_model=ValidateResp)
def validate(req: ValidateReq) -> ValidateResp:
    expr = parse_dsl(req.dsl)

    _, canonical_units, unit_diagnostics = analyze_units_map(req.units)

    dimensions_ok = True
    dimension_error: str | None = None
    dimension_trace: List[Dict[str, object]] = []
    dimension_summary: List[Dict[str, str]] = []

    try:
        _, dimension_trace, dimension_summary = evaluate_equation_dimensions(expr, canonical_units)
    except UnitInferenceError as exc:
        dimensions_ok = False
        dimension_error = str(exc)

    pipeline_report: Dict[str, Any]
    if not unit_diagnostics and dimensions_ok:
        pipeline_report = validate_all(
            expr,
            canonical_units,
            req.result_unit,
            req.constraints,
            req.domain,
            req.pde_space_vars,
            req.pde_time_var,
            req.checks or None,
        )
    else:
        pipeline_report = {
            "ok": False,
            "errors": [
                {
                    "stage": "units",
                    "message": dimension_error or "Unit diagnostics must be resolved before full validation.",
                }
            ],
        }

    overall_ok = (
        pipeline_report.get("ok", False)
        and dimensions_ok
        and not unit_diagnostics
    )

    report = {
        "ok": overall_ok,
        "units": canonical_units,
        "unit_diagnostics": [
            {
                "symbol": diag.symbol,
                "code": diag.code,
                "message": diag.message,
                "hint": diag.hint,
            }
            for diag in unit_diagnostics
        ],
        "dimensions": {
            "ok": dimensions_ok,
            "summary": dimension_summary,
            "trace": dimension_trace,
            "error": dimension_error,
        },
        "pipeline": pipeline_report,
    }

    return ValidateResp(ok=overall_ok, report=report)


@app.post("/v1/schema", response_model=SchemaResp)
def schema(req: SchemaReq) -> SchemaResp:
    expr = parse_dsl(req.dsl)
    canon = canonicalize(expr)
    schema_dict = build_theory_schema(expr, req.domain, req.units, req.constraints)
    return SchemaResp(schema_json=schema_dict, canonical=canon.canonical_str)


@app.post("/v1/codegen", response_model=CodegenResp)
def codegen(req: CodegenReq) -> CodegenResp:
    expr = parse_dsl(req.dsl)
    code = generate_numpy(expr, name=req.name, metadata=req.metadata)
    return CodegenResp(code=code)


@app.post("/v1/manifest", response_model=ManifestResp)
def manifest(req: ManifestReq) -> ManifestResp:
    expr = parse_dsl(req.dsl)
    canon = canonicalize(expr)
    report = validate_all(
        expr,
        req.units,
        req.result_unit,
        req.constraints,
        req.domain,
        req.pde_space_vars,
        req.pde_time_var,
        req.checks or None,
    )

    if not report["ok"]:
        raise HTTPException(400, detail={"report": report})

    schema_dict = build_theory_schema(expr, req.domain, req.units, req.constraints)
    schema_digest = save_schema(schema_dict)

    code = generate_numpy(expr, name=req.metadata.get("name", "compute"), metadata=req.metadata)
    code_digest = save_code(code)

    manifest = ComputationManifest(
        version="1.0",
        canonical=canon.canonical_str,
        domain=req.domain,
        units=req.units,
        constraints=req.constraints,
        checks_report=report,
        schema_digest=schema_digest,
        code_digest=code_digest,
    )

    manifest_hash = save_manifest(asdict(manifest))
    return ManifestResp(manifest_hash=manifest_hash, code_digest=code_digest)


@app.post("/v1/law/analyze")
def analyze_conflict(req: ConflictRequest) -> Dict[str, Any]:
    _cale_services()  # ensure bootstrap succeeded
    engine = _cale_engine()

    try:
        result = engine.suggest(req.rule1.model_dump(), req.rule2.model_dump())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Conflict analysis failed")
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc

    return result


@app.post("/v1/law/suggest_amendment", response_model=SuggestionResponse)
def suggest_amendment(req: ConflictRequest) -> SuggestionResponse:
    _cale_services()  # ensure bootstrap succeeded
    engine = _cale_engine()

    try:
        result = engine.suggest(req.rule1.model_dump(), req.rule2.model_dump())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Amendment suggestion failed")
        raise HTTPException(status_code=500, detail=f"suggestion failed: {exc}") from exc

    return SuggestionResponse(**result)


@app.post("/v1/law/train/dry_run")
def train_dry_run(req: ConflictRequest) -> Dict[str, Any]:
    _cale_services()  # ensure bootstrap succeeded
    engine = _cale_engine()

    allow_training = os.getenv("ALLOW_TRAIN_API") == "1"
    training_result: Dict[str, Any] | None = None

    if allow_training:
        services = _cale_services()
        if not services.corpus:
            raise HTTPException(status_code=400, detail="CALE corpus unavailable")

        try:
            from potatobacon.cale.train import CALETrainer, LegalConflictDataset
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise HTTPException(status_code=503, detail="Training dependencies unavailable") from exc
        except RuntimeError as exc:  # pragma: no cover - fallback when torch missing
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        dataset = LegalConflictDataset(
            csv_path=Path("data/cale/expert_labels.csv"),
            corpus=services.corpus,
            symbolic=services.symbolic,
        )
        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="No training examples found")

        feature_dim = len(services.corpus[0].feature_vector)
        trainer = CALETrainer(feature_dim)
        history = trainer.train(
            dataset,
            symbolic=services.symbolic,
            corpus=services.corpus,
            num_epochs=1,
            use_ssl=False,
            use_graph=False,
        )
        weights = trainer.export_weights()
        services.ccs.weights.update(
            {
                "w_CI": float(weights[0]),
                "w_K": float(weights[1]),
                "w_H": float(weights[2]),
                "w_TD": float(weights[3]),
            }
        )
        training_result = {"history": history}

    try:
        result = engine.suggest(req.rule1.model_dump(), req.rule2.model_dump())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Dry-run training evaluation failed")
        raise HTTPException(status_code=500, detail=f"dry-run failed: {exc}") from exc

    result["training"] = training_result or {"status": "skipped"}
    return result


@app.get("/v1/manifest/{manifest_hash}")
def get_manifest(manifest_hash: str) -> Dict[str, Any]:
    try:
        return load_persisted_manifest(manifest_hash)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Not found") from exc

