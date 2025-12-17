from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.security import require_api_key
from potatobacon.tariff.models import (
    TariffParseRequestModel,
    TariffParseResponseModel,
)
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec

router = APIRouter(
    prefix="/api/tariff",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/parse", response_model=TariffParseResponseModel)
def parse_tariff_request(request: TariffParseRequestModel) -> TariffParseResponseModel:
    """Parse free-text into product specs and compiled tariff facts."""

    spec, extraction_evidence = extract_product_spec(
        description=request.description, bom_text=request.bom_text
    )
    compiled_facts, fact_evidence = compile_facts_with_evidence(
        spec, request.description, request.bom_text
    )

    return TariffParseResponseModel(
        sku_id=request.sku_id,
        product_spec=spec,
        compiled_facts=compiled_facts,
        fact_evidence=fact_evidence,
        extraction_evidence=extraction_evidence,
    )
