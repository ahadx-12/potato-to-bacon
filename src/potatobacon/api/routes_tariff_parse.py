from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.security import require_api_key
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
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

    bom_structured = request.bom_json
    if bom_structured is None and request.bom_csv:
        bom_structured = parse_bom_csv(request.bom_csv)

    normalized_bom_text = request.bom_text
    if bom_structured is not None:
        normalized_bom_text = bom_to_text(bom_structured)

    spec, extraction_evidence = extract_product_spec(
        description=request.description,
        bom_text=normalized_bom_text,
        bom_structured=bom_structured,
        origin_country=request.origin_country,
        export_country=request.export_country,
        import_country=request.import_country,
    )
    compiled_facts, fact_evidence = compile_facts_with_evidence(
        spec,
        request.description,
        normalized_bom_text,
        bom_structured=bom_structured,
    )

    return TariffParseResponseModel(
        sku_id=request.sku_id,
        product_spec=spec,
        compiled_facts=compiled_facts,
        fact_evidence=fact_evidence,
        extraction_evidence=extraction_evidence,
    )
