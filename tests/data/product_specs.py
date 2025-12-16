from potatobacon.tariff.product_schema import (
    ManufacturingProcess,
    MaterialBreakdown,
    OriginInput,
    ProductCategory,
    ProductSpecModel,
    SurfaceCoverage,
)


footwear_spec = ProductSpecModel(
    product_category=ProductCategory.FOOTWEAR,
    materials=[
        MaterialBreakdown(component="upper", material="leather", percent_by_weight=40),
        MaterialBreakdown(component="upper", material="textile", percent_by_weight=20),
    ],
    surface_coverage=[
        SurfaceCoverage(material="textile", percent_coverage=55.0, coating_type="felt"),
    ],
    declared_value_per_unit=60.0,
    annual_volume=1000,
)

bolt_spec = ProductSpecModel(
    product_category=ProductCategory.FASTENER,
    materials=[MaterialBreakdown(component="body", material="steel", percent_by_weight=90)],
    manufacturing_process=ManufacturingProcess.FORGED,
    use_function="fastener",
    declared_value_per_unit=2.0,
    annual_volume=500000,
    country_of_origin_inputs=[
        OriginInput(component="body", country_of_origin="US", transformation="forged"),
    ],
)

electronics_spec = ProductSpecModel(
    product_category=ProductCategory.ELECTRONICS,
    materials=[
        MaterialBreakdown(component="housing", material="aluminum", percent_by_weight=60),
        MaterialBreakdown(component="pcb", material="plastic", percent_by_weight=15),
    ],
    manufacturing_process=ManufacturingProcess.ASSEMBLED,
    use_function="housing",
    declared_value_per_unit=120.0,
    annual_volume=15000,
)
