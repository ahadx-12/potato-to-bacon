from __future__ import annotations

from typing import List


class BOMFormatDetector:
    """Detect BOM format for better extraction"""

    FORMATS = {
        "standard": ["part_number", "description", "quantity"],
        "detailed": ["part_number", "description", "material", "origin", "unit_price"],
        "simple": ["description", "quantity"],
    }

    def detect_format(self, columns: List[str]) -> str:
        normalized = {column.strip().lower() for column in columns}
        for name, required in self.FORMATS.items():
            if all(column in normalized for column in required):
                return name
        return "unknown"
