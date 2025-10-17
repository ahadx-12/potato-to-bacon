from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from potatobacon.codegen.reference import generate_numpy
from potatobacon.manifest.store import ComputationManifest
from potatobacon.parser.dsl_parser import parse_dsl
from potatobacon.semantics.canonicalizer import canonicalize
from potatobacon.semantics.schema_builder import build_theory_schema
from potatobacon.storage import load_manifest as load_persisted_manifest
from potatobacon.storage import save_code, save_manifest, save_schema
from potatobacon.validation.pipeline import validate_all

app = FastAPI(title="potato-to-bacon API", version="v1")
app.mount("/static/docs", StaticFiles(directory="docs", html=True), name="docs")
app.mount("/static/examples", StaticFiles(directory="examples", html=False), name="examples")


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
    schema: Dict[str, Any]
    canonical: str


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


@app.get("/v1/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/info")
def info() -> Dict[str, Any]:
    return {
        "version": "0.1.0",
        "git_sha": os.getenv("GIT_COMMIT"),
        "dsl_features": [
            "return-based DSL",
            "sympy-backed expr parsing",
            "functions: +, -, *, /, pow(int)",
            "derivatives: d(expr, *vars), d2(expr, var)",
            "equations with ==",
        ],
        "validators": ["dimensional_fast", "constraints", "relativistic", "pde_class"],
    }


@app.post("/v1/translate", response_model=TranslateResp)
def translate(req: TranslateReq) -> TranslateResp:
    expr = parse_dsl(req.dsl)
    canon = canonicalize(expr)
    return TranslateResp(success=True, expression=str(expr), canonical=canon.canonical_str)


@app.post("/v1/validate", response_model=ValidateResp)
def validate(req: ValidateReq) -> ValidateResp:
    expr = parse_dsl(req.dsl)
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
    return ValidateResp(ok=report["ok"], report=report)


@app.post("/v1/schema", response_model=SchemaResp)
def schema(req: SchemaReq) -> SchemaResp:
    expr = parse_dsl(req.dsl)
    canon = canonicalize(expr)
    schema_dict = build_theory_schema(expr, req.domain, req.units, req.constraints)
    return SchemaResp(schema=schema_dict, canonical=canon.canonical_str)


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


@app.get("/v1/manifest/{manifest_hash}")
def get_manifest(manifest_hash: str) -> Dict[str, Any]:
    try:
        return load_persisted_manifest(manifest_hash)
    except FileNotFoundError as exc:  # pragma: no cover - simple 404 path
        raise HTTPException(404, "Not found") from exc
