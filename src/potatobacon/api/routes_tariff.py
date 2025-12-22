from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.context_helpers import unknown_law_context_error
from potatobacon.api.security import require_api_key
from potatobacon.tariff.engine import run_tariff_hack
from potatobacon.tariff.models import TariffDossierModel, TariffHuntRequestModel

router = APIRouter(
    prefix="/api/tariff",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/analyze", response_model=TariffDossierModel)
def analyze_tariff(request: TariffHuntRequestModel) -> TariffDossierModel:
    """Analyze tariff exposure and potential optimization for a product scenario."""

    scenario = dict(request.scenario)
    if request.origin_country and "origin_country" not in scenario:
        scenario["origin_country"] = request.origin_country
    if request.import_country and "import_country" not in scenario:
        scenario["import_country"] = request.import_country
    if request.export_country and "export_country" not in scenario:
        scenario["export_country"] = request.export_country

    try:
        dossier = run_tariff_hack(
            base_facts=scenario,
            mutations=request.mutations,
            law_context=request.law_context,
            seed=request.seed or 2025,
        )
    except KeyError as exc:
        attempted = exc.args[0] if exc.args else request.law_context
        raise unknown_law_context_error(attempted) from exc
    return dossier
