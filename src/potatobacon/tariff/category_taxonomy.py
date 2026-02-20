from __future__ import annotations

# ---------------------------------------------------------------------------
# Category taxonomy: maps product categories to HTS chapters and keywords.
#
# This is the primary routing table for the general tariff engine.
# Every product in the HTS falls into one or more of these categories.
# When a product arrives without an HTS hint, the engine uses this table
# to route it to the right set of tariff rules (chapters).
#
# Coverage target: every HTS chapter should be reachable from this table.
# ---------------------------------------------------------------------------

CATEGORIES = {
    # -----------------------------------------------------------------------
    # Chapters 84-85: Machinery and Electronics
    # -----------------------------------------------------------------------
    "electronics": {
        "chapters": [84, 85, 90],
        "keywords": [
            "electronic", "circuit", "cable", "computer", "phone", "usb", "charger",
            "connector", "pcb", "transistor", "semiconductor", "sensor", "display",
            "monitor", "antenna", "battery", "capacitor", "resistor", "transformer",
            "switch", "relay", "module", "microcontroller", "inverter", "led",
        ],
        "subcategories": ["cables", "components", "devices", "accessories"],
    },
    "machinery": {
        "chapters": [84, 85],
        "keywords": [
            "machine", "engine", "pump", "motor", "equipment", "compressor", "gear",
            "turbine", "generator", "boiler", "valve", "bearing", "coupling",
            "cylinder", "piston", "hydraulic", "pneumatic", "conveyor", "press",
            "drill", "milling", "lathe", "extruder", "mixer", "centrifuge",
            "filter", "heat exchanger", "reactor", "agitator",
        ],
        "subcategories": ["industrial", "agricultural", "hvac", "processing"],
    },
    # -----------------------------------------------------------------------
    # Chapter 87: Vehicles and Automotive Parts
    # -----------------------------------------------------------------------
    "automotive": {
        "chapters": [87],
        "keywords": [
            "vehicle", "automobile", "car", "truck", "bus", "motorcycle",
            "auto part", "automotive", "brake", "transmission", "axle", "suspension",
            "chassis", "bumper", "seat belt", "airbag", "spark plug", "exhaust",
            "catalytic converter", "alternator", "starter", "radiator",
            "windshield", "mirror", "door panel", "dashboard", "steering",
            "clutch", "differential", "crankshaft", "camshaft", "piston ring",
        ],
        "subcategories": ["oem", "aftermarket", "ev_components", "safety_systems"],
    },
    # -----------------------------------------------------------------------
    # Chapters 72-83: Base Metals
    # -----------------------------------------------------------------------
    "metals_steel": {
        "chapters": [72, 73],
        "keywords": [
            "steel", "iron", "stainless", "alloy steel", "flat rolled",
            "pipe", "tube", "wire rod", "bar", "sheet", "coil", "structural",
            "angle", "channel", "beam", "rail", "casting", "forging",
            "fitting", "flange", "fastener", "bolt", "nut", "screw", "washer",
        ],
        "subcategories": ["flat_products", "long_products", "tubular", "fabricated"],
    },
    "metals_aluminum": {
        "chapters": [76],
        "keywords": [
            "aluminum", "aluminium", "alumina", "bauxite", "al alloy",
            "aluminum sheet", "aluminum plate", "aluminum extrusion",
            "aluminum foil", "aluminum casting",
        ],
        "subcategories": ["wrought", "cast", "fabricated"],
    },
    "metals_other": {
        "chapters": [74, 75, 78, 79, 80, 81, 82, 83],
        "keywords": [
            "copper", "brass", "bronze", "nickel", "lead", "zinc", "tin",
            "cobalt", "tungsten", "molybdenum", "tantalum", "titanium",
            "tool", "blade", "saw", "knife", "cutlery", "lock", "hinge",
        ],
        "subcategories": ["copper_products", "tools", "hardware"],
    },
    # -----------------------------------------------------------------------
    # Chapters 61-64: Apparel and Footwear
    # -----------------------------------------------------------------------
    "apparel": {
        "chapters": [61, 62],
        "keywords": [
            "shirt", "pants", "jacket", "clothing", "t-shirt", "sweater", "sock",
            "garment", "dress", "blouse", "skirt", "coat", "suit", "underwear",
            "sportswear", "activewear", "outerwear", "knitwear", "jersey",
            "hoodie", "shorts", "leggings", "uniform",
        ],
        "subcategories": ["tops", "bottoms", "outerwear", "underwear", "sports"],
    },
    "footwear": {
        "chapters": [64],
        "keywords": [
            "shoe", "boot", "sandal", "footwear", "sneaker", "slipper",
            "loafer", "pump", "heel", "outsole", "upper", "insole",
        ],
        "subcategories": ["athletic", "casual", "work", "dress"],
    },
    # -----------------------------------------------------------------------
    # Chapters 50-63: Textiles
    # -----------------------------------------------------------------------
    "textiles": {
        "chapters": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63],
        "keywords": [
            "textile", "fabric", "cotton", "polyester", "nylon", "silk", "wool",
            "linen", "rayon", "fiber", "yarn", "thread", "woven", "knit",
            "nonwoven", "felt", "carpet", "rug", "lace", "embroidery",
            "canvas", "denim", "twill", "fleece", "velvet",
        ],
        "subcategories": ["natural_fibers", "synthetic_fibers", "made_up_articles"],
    },
    # -----------------------------------------------------------------------
    # Chapters 39-40: Plastics and Rubber
    # -----------------------------------------------------------------------
    "plastics": {
        "chapters": [39],
        "keywords": [
            "plastic", "polymer", "resin", "vinyl", "polyethylene", "polypropylene",
            "pvc", "abs", "polystyrene", "nylon", "acrylic", "polycarbonate",
            "injection molded", "blow molded", "extruded", "film", "sheet",
            "container", "bottle", "bag", "packaging",
        ],
        "subcategories": ["resins", "films", "containers", "components"],
    },
    "rubber": {
        "chapters": [40],
        "keywords": [
            "rubber", "silicone", "elastomer", "gasket", "seal", "o-ring",
            "hose", "belt", "tire", "latex",
        ],
        "subcategories": ["natural", "synthetic", "fabricated"],
    },
    # -----------------------------------------------------------------------
    # Chapter 94: Furniture and Lighting
    # -----------------------------------------------------------------------
    "furniture": {
        "chapters": [94],
        "keywords": [
            "chair", "table", "furniture", "lamp", "lighting", "sofa", "cabinet",
            "desk", "shelf", "bed", "mattress", "pillow", "cushion", "couch",
            "fixture", "luminaire", "chandelier", "ceiling light",
        ],
        "subcategories": ["seating", "tables", "storage", "lighting", "bedding"],
    },
    # -----------------------------------------------------------------------
    # Chapter 90: Optical, Medical, Scientific Instruments
    # -----------------------------------------------------------------------
    "medical_optical": {
        "chapters": [90],
        "keywords": [
            "medical", "surgical", "diagnostic", "optical", "lens", "microscope",
            "telescope", "camera", "x-ray", "ultrasound", "mri", "dental",
            "orthopedic", "implant", "catheter", "syringe", "instrument",
            "spectrometer", "photovoltaic", "solar panel", "thermometer",
        ],
        "subcategories": ["medical_devices", "optical_instruments", "measuring"],
    },
    # -----------------------------------------------------------------------
    # Chapters 28-38: Chemicals and Pharmaceuticals
    # -----------------------------------------------------------------------
    "chemicals": {
        "chapters": [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        "keywords": [
            "chemical", "acid", "base", "solvent", "reagent", "compound",
            "pharmaceutical", "drug", "medicine", "fertilizer", "pesticide",
            "paint", "coating", "adhesive", "lubricant", "detergent",
            "perfume", "cosmetic", "cleaning", "polymer precursor",
        ],
        "subcategories": ["inorganic", "organic", "pharmaceutical", "agrochemical"],
    },
    # -----------------------------------------------------------------------
    # Chapters 1-24: Agricultural Products and Food
    # -----------------------------------------------------------------------
    "agricultural_food": {
        "chapters": [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ],
        "keywords": [
            "food", "agricultural", "grain", "meat", "fish", "dairy", "fruit",
            "vegetable", "sugar", "oil", "beverage", "wine", "beer", "tobacco",
            "coffee", "tea", "flour", "starch", "animal feed",
        ],
        "subcategories": ["grains", "meats", "produce", "beverages", "processed_food"],
    },
    # -----------------------------------------------------------------------
    # Chapters 44-49: Wood, Paper, Publishing
    # -----------------------------------------------------------------------
    "wood_paper": {
        "chapters": [44, 45, 46, 47, 48, 49],
        "keywords": [
            "wood", "timber", "lumber", "plywood", "mdf", "particle board",
            "paper", "cardboard", "packaging", "book", "print", "publication",
            "furniture component", "wooden",
        ],
        "subcategories": ["lumber", "engineered_wood", "paper_products", "publishing"],
    },
    # -----------------------------------------------------------------------
    # Chapters 93, 95-96: Consumer Goods, Toys, Misc.
    # -----------------------------------------------------------------------
    "consumer_goods": {
        "chapters": [93, 95, 96],
        "keywords": [
            "toy", "game", "sport", "fitness", "weapon", "firearm", "ammunition",
            "pen", "pencil", "brush", "button", "zipper", "umbrella",
            "stationery", "novelty", "collectible",
        ],
        "subcategories": ["toys", "sporting_goods", "stationery", "misc"],
    },
}


def chapters_for_description(description: str) -> list[int]:
    """Return candidate HTS chapters for a product description.

    Searches the keyword lists in CATEGORIES and returns the union of all
    matching chapters, sorted.  Falls back to an empty list (caller should
    use the full atom set).
    """
    lower = description.lower()
    matched_chapters: set[int] = set()
    for _cat_name, cat_data in CATEGORIES.items():
        if any(kw in lower for kw in cat_data["keywords"]):
            matched_chapters.update(cat_data["chapters"])
    return sorted(matched_chapters)
