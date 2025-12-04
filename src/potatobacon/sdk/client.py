"""Typed Python SDK for interacting with the potato-to-bacon API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests  # type: ignore[import-untyped]
from pydantic import BaseModel


class TranslateResp(BaseModel):
    success: bool
    expression: str
    canonical: str


class ValidateResp(BaseModel):
    ok: bool
    report: Dict[str, Any]


class CodegenResp(BaseModel):
    code: str


class ManifestResp(BaseModel):
    manifest_hash: str
    code_digest: str


class VersionInfo(BaseModel):
    engine_version: str
    build: Optional[str] = None
    manifest_hash: Optional[str] = None


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


class AssetDetail(BaseModel):
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


class InfoResp(BaseModel):
    version: str
    git_sha: Optional[str]
    dsl_features: List[str]
    validators: List[str]


@dataclass
class PBConfig:
    """Configuration for :class:`PBClient`."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        env_value = os.getenv("PTB_BASE_URL")
        if env_value:
            return env_value
        return "http://localhost:8000"

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_value = os.getenv("PTB_API_KEY") or os.getenv("CALE_API_KEY")
        if env_value:
            return env_value
        return "dev-key"


class PBClient:
    """High-level synchronous client for the potato-to-bacon REST API."""

    def __init__(self, cfg: Optional[PBConfig] = None, session: Optional[requests.Session] = None):
        self.cfg = cfg or PBConfig()
        self._session = session or requests.Session()
        self._session.headers.setdefault("X-API-Key", self.cfg.resolved_api_key)
        self.assets = AssetsClient(self)

    @property
    def base_url(self) -> str:
        return self.cfg.resolved_base_url

    def translate(self, dsl: str, domain: str = "classical") -> TranslateResp:
        """Translate DSL text into canonical SymPy form.

        Args:
            dsl: DSL text to translate.
            domain: Optional domain override.
        Returns:
            Parsed :class:`TranslateResp` payload.
        """

        response = self._session.post(
            f"{self.base_url}/v1/translate", json={"dsl": dsl, "domain": domain}
        )
        response.raise_for_status()
        return TranslateResp.model_validate(response.json())

    def validate(
        self,
        dsl: str,
        domain: str = "classical",
        units: Optional[Dict[str, str]] = None,
        result_unit: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        checks: Optional[List[str]] = None,
        pde_space_vars: Optional[List[str]] = None,
        pde_time_var: Optional[str] = None,
    ) -> ValidateResp:
        """Run the validation pipeline on the DSL payload.

        Args:
            dsl: DSL text to validate.
            domain: Equation domain hint.
            units: Mapping of symbol to unit string.
            result_unit: Optional expected output unit.
            constraints: Additional variable constraints.
            checks: Optional subset of validators to run.
            pde_space_vars: Spatial variables for PDE classification.
            pde_time_var: Time variable for PDE classification.
        Returns:
            Parsed :class:`ValidateResp` payload with the validation report.
        """

        payload = {
            "dsl": dsl,
            "domain": domain,
            "units": units or {},
            "result_unit": result_unit,
            "constraints": constraints or {},
            "checks": checks or [],
            "pde_space_vars": pde_space_vars or [],
            "pde_time_var": pde_time_var,
        }
        response = self._session.post(f"{self.base_url}/v1/validate", json=payload)
        response.raise_for_status()
        return ValidateResp.model_validate(response.json())

    def codegen(
        self, dsl: str, name: str = "compute", metadata: Optional[Dict[str, Any]] = None
    ) -> CodegenResp:
        """Generate reference NumPy code for the DSL expression."""

        payload = {"dsl": dsl, "name": name, "metadata": metadata or {}}
        response = self._session.post(f"{self.base_url}/v1/codegen", json=payload)
        response.raise_for_status()
        return CodegenResp.model_validate(response.json())

    def manifest(
        self,
        dsl: str,
        domain: str = "classical",
        units: Optional[Dict[str, str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        result_unit: Optional[str] = None,
        checks: Optional[List[str]] = None,
        pde_space_vars: Optional[List[str]] = None,
        pde_time_var: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManifestResp:
        """Create and persist a computation manifest via the API."""

        payload = {
            "dsl": dsl,
            "domain": domain,
            "units": units or {},
            "constraints": constraints or {},
            "result_unit": result_unit,
            "checks": checks or [],
            "pde_space_vars": pde_space_vars or [],
            "pde_time_var": pde_time_var,
            "metadata": metadata or {},
        }
        response = self._session.post(f"{self.base_url}/v1/manifest", json=payload)
        response.raise_for_status()
        return ManifestResp.model_validate(response.json())

    def get_manifest(self, manifest_hash: str) -> Dict[str, Any]:
        """Retrieve a stored computation manifest by hash."""

        response = self._session.get(f"{self.base_url}/v1/manifest/{manifest_hash}")
        response.raise_for_status()
        return response.json()

    def bulk_ingest(self, domain: str, sources: List[Dict[str, Any]], replace_existing: bool = False) -> Dict[str, Any]:
        """Ingest multiple law sources via the bulk manifest API."""

        payload = {"domain": domain, "sources": sources, "options": {"replace_existing": replace_existing}}
        response = self._session.post(f"{self.base_url}/v1/manifest/bulk_ingest", json=payload)
        response.raise_for_status()
        return response.json()

    def analyze(self, rule1: Dict[str, Any], rule2: Dict[str, Any]) -> Dict[str, Any]:
        """Run conflict analysis between two rules."""

        payload = {"rule1": rule1, "rule2": rule2}
        response = self._session.post(f"{self.base_url}/v1/law/analyze", json=payload)
        response.raise_for_status()
        return response.json()

    def hunt(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a synchronous arbitrage hunt."""

        response = self._session.post(f"{self.base_url}/api/law/arbitrage/hunt", json=request)
        response.raise_for_status()
        return response.json()

    def version(self) -> VersionInfo:
        """Fetch the API version metadata."""

        response = self._session.get(f"{self.base_url}/v1/version")
        response.raise_for_status()
        return VersionInfo.model_validate(response.json())

    def info(self) -> InfoResp:
        """Return API metadata including available validators and DSL features."""

        response = self._session.get(f"{self.base_url}/v1/info")
        response.raise_for_status()
        return InfoResp.model_validate(response.json())


class AssetsClient:
    """Namespace for asset-related API helpers."""

    def __init__(self, client: PBClient):
        self._client = client

    def list(
        self,
        jurisdiction: Optional[str] = None,
        from_date: Optional[str] = None,
        limit: int = 10,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "jurisdiction": jurisdiction,
            "from": from_date,
            "limit": limit,
            "cursor": cursor,
        }
        response = self._client._session.get(
            f"{self._client.base_url}/api/law/arbitrage/assets", params=params
        )
        response.raise_for_status()
        data = response.json()
        data["items"] = [AssetSummary.model_validate(item) for item in data.get("items", [])]
        return data

    def get(self, asset_id: str) -> AssetDetail:
        response = self._client._session.get(
            f"{self._client.base_url}/api/law/arbitrage/assets/{asset_id}"
        )
        response.raise_for_status()
        return AssetDetail.model_validate(response.json())
