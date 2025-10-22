from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from potatobacon.api.routes_units import router as units_router
from potatobacon.cale.ccs import CCSCalculator
from potatobacon.cale.embed import FeatureEngine, LegalEmbedder
from potatobacon.cale.graph import compute_authority_scores, load_citation_graph
from potatobacon.cale.parser import PredicateMapper, RuleParser
from potatobacon.cale.suggest import AmendmentSuggester
from potatobacon.cale.symbolic import SymbolicConflictChecker
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
app = FastAPI(title="potato-to-bacon API", version="v1")
app.include_router(units_router)

# Conditionally mount docs/examples if folders exist (avoids startup errors)
if Path("docs").exists():
    app.mount("/static/docs", StaticFiles(directory="docs", html=True), name="docs")

if Path("examples").exists():
    app.mount("/static/examples", StaticFiles(directory="examples", html=False), name="examples")

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


class LawRuleInput(BaseModel):
    text: str = Field(..., min_length=3)
    jurisdiction: str
    statute: str
    section: str
    enactment_year: int


class ConflictRequest(BaseModel):
    rule1: LawRuleInput
    rule2: LawRuleInput


class SuggestionItem(BaseModel):
    condition: str
    justification: Dict[str, float]
    estimated_ccs: float
    suggested_text: str


class SuggestionResponse(BaseModel):
    precedent_count: int
    candidates_considered: int
    suggestions: List[SuggestionItem]
    best: Optional[SuggestionItem]


_cale_predicates: Optional[PredicateMapper] = None
_cale_parser: Optional[RuleParser] = None
_cale_embedder: Optional[LegalEmbedder] = None
_cale_features: Optional[FeatureEngine] = None
_cale_ccs: Optional[CCSCalculator] = None
_cale_symbolic: Optional[SymbolicConflictChecker] = None
_cale_authorities: Dict[str, float] = {}
_cale_corpus_rules: List[LegalRule] = []
_cale_suggester: Optional[AmendmentSuggester] = None


_DEMO_RULE_TEXTS: Dict[str, str] = {
    "R1": "Organizations MUST collect personal data IF consent.",
    "R2": "Security agencies MUST NOT collect personal data IF emergency.",
    "R3": "Organizations MAY collect personal data IF emergency.",
    "R4": "Financial institutions MUST report personal data breaches IF data breach.",
    "R5": "Organizations MAY collect personal data IF national security threat.",
    "R6": "Provincial agencies MAY collect personal data IF court order.",
    "R7": "International agreements MAY require data disclosure IF treaty obligation.",
}


def _load_demo_rules() -> List[LegalRule]:
    if (
        _cale_parser is None
        or _cale_features is None
    ):
        raise RuntimeError("CALE components not initialised")

    demo_path = DATA_ROOT / "data/cale/demo_corpus.json"
    if not demo_path.exists():
        return []

    payload = json.loads(demo_path.read_text())
    rules: List[LegalRule] = []
    for item in payload:
        rid = str(item["id"])
        text = _DEMO_RULE_TEXTS.get(rid)
        if not text:
            continue
        metadata = {
            "id": rid,
            "jurisdiction": item.get("jurisdiction", "CA.Federal"),
            "statute": item.get("statute", ""),
            "section": item.get("section", ""),
            "enactment_year": item.get("year", 2000),
        }
        rule = _cale_parser.parse(text, metadata)
        populated = _cale_features.populate_features(rule, authorities=_cale_authorities)
        rules.append(populated)
    return rules


@app.on_event("startup")
def _initialise_cale() -> None:
    global _cale_predicates, _cale_parser, _cale_embedder, _cale_features, _cale_ccs
    global _cale_symbolic, _cale_authorities, _cale_corpus_rules, _cale_suggester

    _cale_predicates = PredicateMapper()
    _cale_parser = RuleParser(_cale_predicates)
    _cale_embedder = LegalEmbedder(backend=os.getenv("CALE_EMBED_BACKEND", "hash"))
    _cale_features = FeatureEngine(_cale_embedder)
    _cale_ccs = CCSCalculator()
    _cale_symbolic = SymbolicConflictChecker(_cale_predicates)

    demo_corpus = DATA_ROOT / "data/cale/demo_corpus.json"
    if demo_corpus.exists():
        graph = load_citation_graph(demo_corpus)
        _cale_authorities = compute_authority_scores(graph)
    else:
        _cale_authorities = {}

    _cale_corpus_rules = []
    if _cale_parser is not None and _cale_features is not None:
        _cale_corpus_rules = _load_demo_rules()

    if (
        _cale_corpus_rules
        and _cale_embedder is not None
        and _cale_ccs is not None
        and _cale_predicates is not None
    ):
        _cale_suggester = AmendmentSuggester(
            _cale_corpus_rules, _cale_embedder, _cale_ccs, _cale_predicates
        )
    else:
        _cale_suggester = None


def _parse_and_populate_rule(inp: LawRuleInput, rid: str) -> LegalRule:
    if _cale_parser is None or _cale_features is None:
        raise RuntimeError("CALE components not initialised")
    metadata = {
        "id": rid,
        "jurisdiction": inp.jurisdiction,
        "statute": inp.statute,
        "section": inp.section,
        "enactment_year": inp.enactment_year,
    }
    rule = _cale_parser.parse(inp.text, metadata)
    return _cale_features.populate_features(rule, authorities=_cale_authorities)


def _ensure_cale_ready() -> None:
    if _cale_symbolic is None or _cale_ccs is None or _cale_suggester is None:
        _initialise_cale()


# -----------------------------------------------------------------------------
# Health + Info Endpoints
# -----------------------------------------------------------------------------
@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "status": "ok"}


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
    _ensure_cale_ready()
    if _cale_symbolic is None or _cale_ccs is None:
        raise HTTPException(status_code=500, detail="CALE components not initialised")

    try:
        rule1 = _parse_and_populate_rule(req.rule1, "R_api_1")
        rule2 = _parse_and_populate_rule(req.rule2, "R_api_2")
        ci = _cale_symbolic.check_conflict(rule1, rule2)
        analysis = _cale_ccs.compute_multiperspective(rule1, rule2, ci)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=422, detail=f"analysis failed: {exc!r}") from exc

    return {
        "conflict_scores": {
            "textualist": analysis.CCS_textualist,
            "living": analysis.CCS_living,
            "pragmatic": analysis.CCS_pragmatic,
        },
        "components": {
            "symbolic_conflict": analysis.CI,
            "contextual_similarity": analysis.K,
            "authority": analysis.H,
            "temporal_drift": analysis.TD,
        },
        "interpretation": analysis.interpretation,
        "variance": analysis.variance,
    }


@app.post("/v1/law/suggest_amendment", response_model=SuggestionResponse)
def suggest_amendment(req: ConflictRequest) -> SuggestionResponse:
    _ensure_cale_ready()
    if _cale_symbolic is None or _cale_ccs is None or _cale_suggester is None:
        raise HTTPException(status_code=500, detail="CALE components not initialised")

    try:
        rule1 = _parse_and_populate_rule(req.rule1, "R_api_1")
        rule2 = _parse_and_populate_rule(req.rule2, "R_api_2")
        ci = _cale_symbolic.check_conflict(rule1, rule2)
        analysis = _cale_ccs.compute_multiperspective(rule1, rule2, ci)
        result = _cale_suggester.suggest_amendment(rule1, rule2, analysis)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=422, detail=f"suggestion failed: {exc!r}") from exc

    return SuggestionResponse(**result)


@app.get("/v1/manifest/{manifest_hash}")
def get_manifest(manifest_hash: str) -> Dict[str, Any]:
    try:
        return load_persisted_manifest(manifest_hash)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Not found") from exc
DATA_ROOT = Path(__file__).resolve().parents[3]

