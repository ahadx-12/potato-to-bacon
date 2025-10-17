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


class InfoResp(BaseModel):
    version: str
    git_sha: Optional[str]
    dsl_features: List[str]
    validators: List[str]


@dataclass
class PBConfig:
    """Configuration for :class:`PBClient`."""

    base_url: Optional[str] = None

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        env_value = os.getenv("PTB_BASE_URL")
        if env_value:
            return env_value
        return "http://localhost:8000"


class PBClient:
    """High-level synchronous client for the potato-to-bacon REST API."""

    def __init__(self, cfg: Optional[PBConfig] = None, session: Optional[requests.Session] = None):
        self.cfg = cfg or PBConfig()
        self._session = session or requests.Session()

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

    def info(self) -> InfoResp:
        """Return API metadata including available validators and DSL features."""

        response = self._session.get(f"{self.base_url}/v1/info")
        response.raise_for_status()
        return InfoResp.model_validate(response.json())
