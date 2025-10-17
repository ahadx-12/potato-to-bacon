from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import sympy as sp

from potatobacon.parser.dsl_parser import parse_dsl  # existing
from potatobacon.semantics.canonicalizer import canonicalize  # existing
from potatobacon.semantics.ir import TheoryIR  # existing
from potatobacon.validation.pipeline import validate_all
from potatobacon.semantics.schema_builder import build_theory_schema  # existing
from potatobacon.codegen.reference import generate_numpy
from potatobacon.manifest.store import ComputationManifest, persist_manifest, persist_code, load_manifest

app = FastAPI(title="potato-to-bacon API", version="v1")

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
def health():
    return {"ok": True}

@app.post("/v1/translate", response_model=TranslateResp)
def translate(req: TranslateReq):
    expr = parse_dsl(req.dsl)   # SymPy expr or Eq
    canon = canonicalize(expr)
    return TranslateResp(success=True, expression=str(expr), canonical=canon.canonical_str)

@app.post("/v1/validate", response_model=ValidateResp)
def validate(req: ValidateReq):
    expr = parse_dsl(req.dsl)
    rep = validate_all(expr, req.units, req.result_unit, req.constraints, req.domain,
                       req.pde_space_vars, req.pde_time_var, req.checks or None)
    return ValidateResp(ok=rep["ok"], report=rep)

@app.post("/v1/schema", response_model=SchemaResp)
def schema(req: SchemaReq):
    expr = parse_dsl(req.dsl)
    canon = canonicalize(expr)
    schema = build_theory_schema(expr, req.domain, req.units, req.constraints)
    return SchemaResp(schema=schema, canonical=canon.canonical_str)

@app.post("/v1/codegen", response_model=CodegenResp)
def codegen(req: CodegenReq):
    expr = parse_dsl(req.dsl)
    code = generate_numpy(expr, name=req.name, metadata=req.metadata)
    return CodegenResp(code=code)

@app.post("/v1/manifest", response_model=ManifestResp)
def manifest(req: ManifestReq):
    expr = parse_dsl(req.dsl)
    canon = canonicalize(expr)
    report = validate_all(expr, req.units, req.result_unit, req.constraints, req.domain,
                          req.pde_space_vars, req.pde_time_var, req.checks or None)
    if not report["ok"]:
        raise HTTPException(400, detail={"report": report})

    schema = build_theory_schema(expr, req.domain, req.units, req.constraints)
    code = generate_numpy(expr, name=req.metadata.get("name", "compute"), metadata=req.metadata)

    code_digest = persist_code(code)
    man = ComputationManifest(
        version="1.0",
        canonical=canon.canonical_str,
        domain=req.domain,
        units=req.units,
        constraints=req.constraints,
        checks_report=report,
        schema_digest=hashlib_sha(schema),
        code_digest=code_digest,
    )
    h = persist_manifest(man)
    return ManifestResp(manifest_hash=h, code_digest=code_digest)

def hashlib_sha(obj: dict) -> str:
    import json, hashlib
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()

@app.get("/v1/manifest/{h}")
def get_manifest(h: str):
    try:
        return load_manifest(h)
    except FileNotFoundError:
        raise HTTPException(404, "Not found")
