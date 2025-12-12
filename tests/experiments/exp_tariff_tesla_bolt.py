from potatobacon.tariff.optimizer import optimize_tariff

TESLA_BOLT_BASE = {
    "product_type_chassis_bolt": True,
    "material_steel": True,
    "material_aluminum": False,
}

CANDIDATES = {
    "material_steel": [True, False],
    "material_aluminum": [False, True],
}

def run_tesla_bolt_demo():
    """Run the Tesla bolt demo optimization."""
    result = optimize_tariff(
        base_facts=TESLA_BOLT_BASE,
        candidate_mutations=CANDIDATES,
        law_context=None,
    )
    return result
