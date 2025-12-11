from __future__ import annotations

from fastapi import APIRouter, Depends

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

    dossier = run_tariff_hack(
        base_facts=request.scenario,
        mutations=request.mutations,
        seed=request.seed or 2025,
    )
    return dossier
