from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Query,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from potatobacon.api.routes_units import router as units_router
from potatobacon.api.routes_tariff import router as tariff_router
from potatobacon.api.routes_tariff_explain import router as tariff_explain_router
from potatobacon.api.routes_tariff_optimize import router as tariff_optimize_router
from potatobacon.api.routes_tariff_sku_optimize import router as tariff_sku_optimize_router
from potatobacon.api.routes_tariff_batch_scan import router as tariff_batch_scan_router
from potatobacon.api.routes_tariff_suggest import router as tariff_suggest_router
from potatobacon.api.routes_tariff_parse import router as tariff_parse_router
from potatobacon.api.routes_tariff_sku_dossier import router as tariff_sku_dossier_router
from potatobacon.api.routes_tariff_sku_dossier_v2 import router as tariff_sku_dossier_v2_router
from potatobacon.api.routes_tariff_skus import router as tariff_skus_router
from potatobacon.api.routes_tariff_sessions import router as tariff_sessions_router
from potatobacon.api.routes_tariff_evidence import router as tariff_evidence_router
from potatobacon.api.routes_proofs import router as proofs_router
from potatobacon.api.routes_law_contexts import router as law_contexts_router
from potatobacon.api.security import ENGINE_VERSION, require_api_key
from potatobacon.cale.bootstrap import CALEServices, build_services
from potatobacon.cale.engine import CALEEngine
from potatobacon.cale.types import LegalRule
from potatobacon.codegen.reference import generate_numpy
from potatobacon.core.units import analyze_units_map
from potatobacon.dashboard import register_tax_dashboard
from potatobacon.law.jobs import job_manager
from potatobacon.manifest.store import ComputationManifest
from potatobacon.parser.dsl_parser import parse_dsl
from potatobacon.semantics.canonicalizer import canonicalize
from potatobacon.semantics.schema_builder import build_theory_schema
from potatobacon.storage import load_manifest as load_persisted_manifest
from potatobacon.storage import save_code, save_manifest, save_schema
from potatobacon.storage import latest_manifest_hash
from potatobacon.persistence import get_store
from potatobacon.observability import (
    bind_run_id,
    current_run_id,
    log_event,
    new_run_id,
    redact_api_key,
    reset_run_id,
)
from uuid import uuid4
from potatobacon.validation.pipeline import validate_all
from potatobacon.units.infer import evaluate_equation_dimensions, UnitInferenceError
from potatobacon.law.arbitrage_hunter import run_arbitrage_hunt
from potatobacon.law.arbitrage_models import ArbitrageDossierModel
from potatobacon.law.cale_metrics import batch_metrics, compute_scenario_metrics, sample_scenarios
from potatobacon.law.manifest import LawSource, ingest_sources, load_latest_law_manifest
from potatobacon.law.pdf_ingest import build_sources_from_pdf, extract_text_from_pdf
from potatobacon.law.solver_z3 import build_policy_atoms_from_rules

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
app.include_router(tariff_router)
app.include_router(tariff_explain_router)
app.include_router(tariff_optimize_router)
app.include_router(tariff_sku_optimize_router)
app.include_router(tariff_batch_scan_router)
app.include_router(tariff_suggest_router)
app.include_router(tariff_parse_router)
app.include_router(tariff_sku_dossier_router)
app.include_router(tariff_skus_router)
app.include_router(tariff_sku_dossier_v2_router)
app.include_router(tariff_sessions_router)
app.include_router(tariff_evidence_router)
app.include_router(proofs_router)
app.include_router(law_contexts_router)

persistence_store = get_store()


@app.middleware("http")
async def attach_run_id(request: Request, call_next):
    run_id = current_run_id() or new_run_id()
    token = bind_run_id(run_id)
    redacted_key = redact_api_key(request.headers.get("X-API-Key"))
    log_event("request.start", path=str(request.url.path), api_key=redacted_key)
    try:
        response = await call_next(request)
        response.headers["X-Run-ID"] = run_id
        return response
    finally:
        log_event("request.end", path=str(request.url.path), api_key=redacted_key)
        reset_run_id(token)

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
# CALE-LAW dashboard registration
# -----------------------------------------------------------------------------
register_tax_dashboard(app)


def _redact_message(msg: str) -> str:
    if msg.lower().startswith("value error, "):
        msg = msg.split(", ", 1)[1]
    lowered = msg.lower()
    if "key" in lowered or "token" in lowered or "secret" in lowered:
        return "Invalid request payload"
    return msg


def _normalize_validation_errors(raw_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields: List[Dict[str, str]] = []
    for err in raw_errors:
        loc = err.get("loc", [])
        loc_parts = [str(part) for part in loc if part != "body"]
        path = ".".join(["request", *loc_parts]) if loc_parts else "request"
        message = _redact_message(err.get("msg", "Invalid request"))
        fields.append({"path": path, "message": message})
    return {"error": "VALIDATION_ERROR", "fields": fields}


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
):  # pragma: no cover - exercised via system tests
    content = _normalize_validation_errors(exc.errors())
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(ValidationError)
async def handle_pydantic_validation_error(
    request: Request, exc: ValidationError
):  # pragma: no cover - defensive normalization
    content = _normalize_validation_errors(exc.errors())
    return JSONResponse(status_code=422, content=content)


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
    engine_version: str = ENGINE_VERSION


class AssetSummary(BaseModel):
    id: str
    jurisdiction: str
    created_at: str
    metrics: Dict[str, Any]
    provenance_chain: List[Dict[str, Any]] | List[Any]
    dependency_graph: Dict[str, Any] | None = None
    engine_version: Optional[str] = None
    manifest_hash: Optional[str] = None
    run_id: Optional[str] = None


class AssetListResponse(BaseModel):
    items: List[AssetSummary]
    next_cursor: Optional[str] = None


class AssetDetailResponse(BaseModel):
    id: str
    jurisdiction: str
    created_at: str
    dossier: Dict[str, Any]
    metrics: Dict[str, Any]
    provenance_chain: List[Any]
    dependency_graph: Dict[str, Any]
    engine_version: Optional[str] = None
    manifest_hash: Optional[str] = None
    run_id: Optional[str] = None


class VersionResp(BaseModel):
    engine_version: str
    build: str | None = None
    manifest_hash: str | None = None


class BulkSource(BaseModel):
    id: str
    text: str
    jurisdiction: str | None = None
    statute: str | None = None
    section: str | None = None
    enactment_year: int | None = None


class BulkOptions(BaseModel):
    replace_existing: bool = False


class BulkManifestRequest(BaseModel):
    domain: str
    sources: List[BulkSource]
    options: BulkOptions = Field(default_factory=BulkOptions)


class BulkManifestResponse(BaseModel):
    manifest_hash: str
    rules_count: int
    domain: str
    sources_ingested: List[str]
    engine_version: str = ENGINE_VERSION
    extracted_sections: int | None = None


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


class ArbitrageObjective(str, Enum):
    MAXIMIZE_NET_AFTER_TAX = "MAXIMIZE(net_after_tax_income)"
    MINIMIZE_RISK = "MINIMIZE(risk)"


class ArbitrageRequestModel(BaseModel):
    jurisdictions: List[str] = Field(default_factory=list)
    domain: str = "tax"
    objective: ArbitrageObjective = ArbitrageObjective.MAXIMIZE_NET_AFTER_TAX
    constraints: Dict[str, Any] = Field(default_factory=dict)
    risk_tolerance: str = Field(default="medium", pattern="^(low|medium|high)$")
    seed: int | None = None
    manifest_hash: str | None = None

    @model_validator(mode="before")
    @classmethod
    def unwrap_request(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(value, Mapping) and "request" in value:
            inner = dict(value.get("request") or {})
            manifest_hash = value.get("manifest_hash")
            if manifest_hash is not None:
                inner.setdefault("manifest_hash", manifest_hash)
            value = inner

        if isinstance(value, Mapping):
            normalized = dict(value)
            objective = normalized.get("objective")
            alias_map = {
                "MAXIMIZE_NET_AFTER_TAX": ArbitrageObjective.MAXIMIZE_NET_AFTER_TAX,
                "MINIMIZE_RISK": ArbitrageObjective.MINIMIZE_RISK,
            }
            if isinstance(objective, str) and objective in alias_map:
                normalized["objective"] = alias_map[objective]
            return normalized

        return value

    @field_validator("jurisdictions")
    @classmethod
    def _validate_jurisdictions(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one jurisdiction is required")
        return value


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


def _active_manifest_hash(domain: str | None = None) -> str | None:
    manifest_map: Dict[str, str] = getattr(app.state, "active_manifest_hash", {}) or {}
    if domain and domain in manifest_map:
        return manifest_map[domain]
    domain_hash = latest_manifest_hash(domain) if domain else None
    return domain_hash or latest_manifest_hash(None)


def _update_active_manifest(domain: str, manifest_hash: str) -> None:
    manifest_map: Dict[str, str] = getattr(app.state, "active_manifest_hash", {}) or {}
    manifest_map[domain] = manifest_hash
    app.state.active_manifest_hash = manifest_map


def _parse_from_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    # Ignore future filters to keep tests deterministic
    if parsed > datetime.now(timezone.utc) + timedelta(days=1):
        return None
    return parsed


def _persist_asset_from_dossier(
    dossier: Dict[str, Any],
    req: ArbitrageRequestModel,
    api_key: str | None,
) -> Dict[str, Any]:
    manifest_hash = _active_manifest_hash(req.domain)
    run_id = current_run_id()
    asset_id = dossier.get("id") or str(uuid4())
    jurisdiction = req.jurisdictions[0] if req.jurisdictions else "Unknown"
    provenance_chain = dossier.get("provenance_chain") or [
        {
            "step": 0,
            "jurisdiction": jurisdiction,
            "rule_id": "seed",
            "type": "seed",
            "role": "generated",
        }
    ]
    dependency_graph = dossier.get("dependency_graph") or {"nodes": [], "edges": []}
    persistence_store.save_asset(
        asset_id=asset_id,
        jurisdiction=jurisdiction,
        payload=dossier,
        metrics_summary=dossier.get("metrics", {}),
        provenance_chain=provenance_chain,
        dependency_graph=dependency_graph,
        engine_version=ENGINE_VERSION,
        manifest_hash=manifest_hash,
        run_id=run_id,
    )
    dossier["id"] = asset_id
    dossier["engine_version"] = ENGINE_VERSION
    dossier["manifest_hash"] = manifest_hash
    dossier["provenance_chain"] = provenance_chain
    dossier["dependency_graph"] = dependency_graph
    dossier["run_id"] = run_id
    log_event(
        "asset.persisted",
        asset_id=asset_id,
        jurisdiction=jurisdiction,
        api_key=redact_api_key(api_key),
    )
    return dossier


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


@app.get("/v1/version", response_model=VersionResp)
def version() -> VersionResp:
    return VersionResp(
        engine_version=ENGINE_VERSION,
        build=os.getenv("GIT_COMMIT"),
        manifest_hash=_active_manifest_hash(None),
    )


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
def manifest(req: ManifestReq, api_key: str = Depends(require_api_key)) -> ManifestResp:
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
    _update_active_manifest(req.domain, manifest_hash)
    return ManifestResp(manifest_hash=manifest_hash, code_digest=code_digest)


@app.post("/v1/law/analyze")
def analyze_conflict(req: ConflictRequest, api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
    services = _cale_services()  # ensure bootstrap succeeded
    engine = _cale_engine()

    # Parse the incoming rules so we can compute scenario-aware metrics
    rule1 = _parse_and_populate_rule(req.rule1, services)
    rule2 = _parse_and_populate_rule(req.rule2, services)
    atoms = build_policy_atoms_from_rules([rule1, rule2], services.mapper)
    scenario_summary = batch_metrics(atoms, sample_size=8)
    sampled = sample_scenarios(atoms, sample_size=3)
    scenario_metrics = [
        asdict(compute_scenario_metrics(scenario, atoms)) for scenario in sampled
    ]

    try:
        result = engine.suggest(req.rule1.model_dump(), req.rule2.model_dump())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Conflict analysis failed")
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc

    result["scenario_metrics"] = {
        "summary": scenario_summary,
        "samples": scenario_metrics,
    }
    result["engine_version"] = ENGINE_VERSION
    result["manifest_hash"] = _active_manifest_hash(None)
    return result


@app.post("/v1/law/suggest_amendment", response_model=SuggestionResponse)
def suggest_amendment(req: ConflictRequest, api_key: str = Depends(require_api_key)) -> SuggestionResponse:
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
def train_dry_run(req: ConflictRequest, api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
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


@app.post("/api/law/arbitrage/hunt", response_model=ArbitrageDossierModel)
def arbitrage_hunt(req: ArbitrageRequestModel, api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
    services = _cale_services()
    dossier = run_arbitrage_hunt(services, req.model_dump())
    dossier = _persist_asset_from_dossier(dossier, req, api_key)
    return dossier


@app.post("/api/law/arbitrage/hunt/job")
def arbitrage_hunt_job(
    req: ArbitrageRequestModel,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(require_api_key),
) -> Dict[str, str]:
    services = _cale_services()
    job = job_manager.create_job("arbitrage_hunt")

    def runner() -> Dict[str, Any]:
        dossier = run_arbitrage_hunt(services, req.model_dump())
        return _persist_asset_from_dossier(dossier, req, api_key)

    background_tasks.add_task(job_manager.run_job, job, runner)
    return {"job_id": job.id, "engine_version": ENGINE_VERSION}


@app.get("/api/law/jobs/{job_id}")
def get_job(job_id: str, api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "run_id": job.run_id,
        "engine_version": ENGINE_VERSION,
    }


@app.get("/api/law/arbitrage/assets", response_model=AssetListResponse)
def list_assets(
    jurisdiction: str | None = None,
    from_: str | None = Query(None, alias="from"),
    limit: int = Query(10, ge=1, le=50),
    cursor: str | None = None,
    api_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    from_date = _parse_from_date(from_)
    cursor_int = int(cursor) if cursor else None
    items, next_cursor = persistence_store.list_assets(jurisdiction, from_date, limit, cursor_int)
    summaries = [AssetSummary(**{k: v for k, v in item.items() if k != "_cursor"}) for item in items]
    log_event("assets.list", count=len(summaries), api_key=redact_api_key(api_key))
    return {"items": summaries, "next_cursor": str(next_cursor) if next_cursor else None}


@app.get("/api/law/arbitrage/assets/{asset_id}", response_model=AssetDetailResponse)
def get_asset(asset_id: str, api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
    stored = persistence_store.get_asset(asset_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Asset not found")
    log_event("assets.get", asset_id=asset_id, api_key=redact_api_key(api_key))
    payload = stored.get("payload") or {}
    return {
        "id": stored["id"],
        "jurisdiction": stored.get("jurisdiction", "Unknown"),
        "created_at": stored.get("created_at"),
        "dossier": payload,
        "metrics": stored.get("metrics", {}),
        "provenance_chain": stored.get("provenance_chain", []),
        "dependency_graph": stored.get("dependency_graph") or {},
        "engine_version": stored.get("engine_version"),
        "manifest_hash": stored.get("manifest_hash"),
        "run_id": stored.get("run_id"),
    }


@app.get("/v1/manifest/{manifest_hash}")
def get_manifest(manifest_hash: str) -> Dict[str, Any]:
    try:
        return load_persisted_manifest(manifest_hash)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Not found") from exc


@app.post("/v1/manifest/bulk_ingest", response_model=BulkManifestResponse)
def bulk_ingest_manifest(req: BulkManifestRequest, api_key: str = Depends(require_api_key)) -> BulkManifestResponse:
    services = _cale_services()
    sources = [
        LawSource(
            id=src.id,
            text=src.text,
            jurisdiction=src.jurisdiction or "Unknown",
            statute=src.statute or src.id,
            section=src.section or "",
            enactment_year=src.enactment_year or 2000,
        )
        for src in req.sources
    ]
    result = ingest_sources(services, req.domain, sources, replace_existing=req.options.replace_existing)
    _update_active_manifest(req.domain, result["manifest_hash"])
    return BulkManifestResponse(engine_version=ENGINE_VERSION, **result)


@app.post("/v1/manifest/ingest_pdf", response_model=BulkManifestResponse)
async def ingest_pdf_manifest(
    domain: str = Form(...),
    base_id: str | None = Form(None),
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
) -> BulkManifestResponse:
    payload = await file.read()
    text = extract_text_from_pdf(payload)
    sources = build_sources_from_pdf(text, base_id)
    services = _cale_services()
    result = ingest_sources(services, domain, sources)
    _update_active_manifest(domain, result["manifest_hash"])
    return BulkManifestResponse(
        engine_version=ENGINE_VERSION,
        extracted_sections=len(sources),
        **result,
    )
