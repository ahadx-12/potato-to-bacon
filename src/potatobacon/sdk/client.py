from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import requests

@dataclass
class PBConfig:
    base_url: str = "http://localhost:8000"

class PBClient:
    def __init__(self, cfg: Optional[PBConfig] = None):
        self.cfg = cfg or PBConfig()
        self.s = requests.Session()

    def translate(self, dsl: str) -> Dict[str, Any]:
        r = self.s.post(f"{self.cfg.base_url}/v1/translate", json={"dsl": dsl})
        r.raise_for_status()
        return r.json()

    def validate(self, dsl: str, domain="classical", units=None, result_unit=None,
                 constraints=None, checks=None, pde_space_vars=None, pde_time_var=None) -> Dict[str, Any]:
        payload = {
            "dsl": dsl, "domain": domain,
            "units": units or {}, "result_unit": result_unit,
            "constraints": constraints or {}, "checks": checks or [],
            "pde_space_vars": pde_space_vars or [], "pde_time_var": pde_time_var
        }
        r = self.s.post(f"{self.cfg.base_url}/v1/validate", json=payload)
        r.raise_for_status()
        return r.json()

    def codegen(self, dsl: str, name="compute", metadata=None) -> str:
        r = self.s.post(f"{self.cfg.base_url}/v1/codegen", json={"dsl": dsl, "name": name, "metadata": metadata or {}})
        r.raise_for_status()
        return r.json()["code"]

    def manifest(self, dsl: str, domain="classical", units=None, constraints=None,
                 result_unit=None, checks=None, pde_space_vars=None, pde_time_var=None, metadata=None) -> Dict[str, Any]:
        payload = {
            "dsl": dsl, "domain": domain, "units": units or {},
            "constraints": constraints or {}, "result_unit": result_unit,
            "checks": checks or [], "pde_space_vars": pde_space_vars or [],
            "pde_time_var": pde_time_var, "metadata": metadata or {}
        }
        r = self.s.post(f"{self.cfg.base_url}/v1/manifest", json=payload)
        r.raise_for_status()
        return r.json()

    def get_manifest(self, h: str) -> Dict[str, Any]:
        r = self.s.get(f"{self.cfg.base_url}/v1/manifest/{h}")
        r.raise_for_status()
        return r.json()
