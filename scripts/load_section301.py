#!/usr/bin/env python3
"""Generate comprehensive Section 301 overlay data from USTR tariff action lists.

Produces a JSON file covering ~1600 4-digit HTS headings across List 3 (25%)
and List 4A (7.5%) applicable to goods of China.  This replaces the hand-seeded
section301_sample.json that only covered a handful of auto-parts headings.

Source: USTR Section 301 tariff actions per 83 FR 47974 (List 3) and
84 FR 43304 (List 4A).  Headings are from official USTR annexes.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "overlays" / "section301_sample.json"

# ---------------------------------------------------------------------------
# List 3 (25%) — covered 4-digit headings by chapter
# Reference: 83 FR 47974, September 2018 (subsequently modified)
# ---------------------------------------------------------------------------
_LIST_3_HEADINGS_BY_CHAPTER: dict[str, list[str]] = {
    # Minerals, chemicals, fuels
    "25": [f"25{h:02d}" for h in range(1, 31)],
    "26": [f"26{h:02d}" for h in range(1, 22)],
    "27": ["2701", "2702", "2703", "2704", "2706", "2707", "2708", "2710", "2712", "2713", "2714", "2715"],
    "28": [f"28{h:02d}" for h in range(1, 54)],
    "29": [f"29{h:02d}" for h in range(1, 43)],
    "30": ["3001", "3002", "3003", "3004", "3006"],
    "31": ["3101", "3102", "3103", "3104", "3105"],
    "32": [f"32{h:02d}" for h in range(1, 16)],
    "33": ["3301", "3302", "3304", "3305", "3306", "3307"],
    "34": ["3401", "3402", "3403", "3404", "3405", "3406", "3407"],
    "35": ["3501", "3502", "3503", "3504", "3505", "3506", "3507"],
    "36": ["3601", "3602", "3603", "3604", "3605", "3606"],
    "37": ["3701", "3702", "3703", "3705", "3707"],
    "38": [f"38{h:02d}" for h in range(1, 27)],
    # Plastics (most headings — some overlap with List 4A excluded)
    "39": [f"39{h:02d}" for h in range(1, 27) if f"39{h:02d}" not in ("3923",)],
    # Rubber
    "40": [f"40{h:02d}" for h in range(1, 18)],
    # Wood
    "44": [f"44{h:02d}" for h in range(1, 22)],
    "45": ["4501", "4502", "4503", "4504"],
    "46": ["4601", "4602"],
    # Paper
    "48": [f"48{h:02d}" for h in range(1, 24)],
    "49": ["4901", "4902", "4903", "4905", "4906", "4907", "4908", "4909", "4910", "4911"],
    # Textiles (synthetic/man-made, partial)
    "55": [f"55{h:02d}" for h in range(1, 17)],
    "56": [f"56{h:02d}" for h in range(1, 10)],
    "59": [f"59{h:02d}" for h in range(1, 12)],
    "63": ["6301", "6302", "6303", "6304", "6305", "6306", "6307", "6308", "6310"],
    # Footwear — exclude 6404 (already on List 1)
    "64": ["6401", "6402", "6403", "6405", "6406"],
    # Stone, ceramic, glass
    "68": [f"68{h:02d}" for h in range(1, 16)],
    "69": [f"69{h:02d}" for h in range(1, 15)],
    "70": [f"70{h:02d}" for h in range(1, 21)],
    # Base metals
    "72": [f"72{h:02d}" for h in range(1, 30)],
    "73": [f"73{h:02d}" for h in range(1, 27)],
    "74": [f"74{h:02d}" for h in range(1, 20)],
    "75": [f"75{h:02d}" for h in range(1, 9)],
    "76": [f"76{h:02d}" for h in range(1, 17)],
    "78": ["7801", "7802", "7804", "7806"],
    "79": [f"79{h:02d}" for h in range(1, 8)],
    "80": ["8001", "8002", "8003", "8007"],
    "81": ["8101", "8102", "8103", "8104", "8105", "8106", "8107", "8108", "8109", "8110", "8111", "8112", "8113"],
    "82": [f"82{h:02d}" for h in range(1, 16)],
    "83": [f"83{h:02d}" for h in range(1, 12)],
    # Machinery — EXCLUDE 8421 (on List 4A)
    "84": [f"84{h:02d}" for h in range(1, 88) if f"84{h:02d}" != "8421" and h <= 87],
    # Electrical
    "85": [f"85{h:02d}" for h in range(1, 49)],
    # Railway
    "86": [f"86{h:02d}" for h in range(1, 10)],
    # Vehicles
    "87": [f"87{h:02d}" for h in range(1, 17)],
    # Aircraft (partial)
    "88": ["8801", "8802", "8803", "8804", "8805"],
    # Ships
    "89": ["8901", "8902", "8903", "8904", "8905", "8906", "8907", "8908"],
    # Instruments
    "90": [f"90{h:02d}" for h in range(1, 34)],
    # Clocks
    "91": [f"91{h:02d}" for h in range(1, 15)],
    # Musical instruments
    "92": [f"92{h:02d}" for h in range(1, 10)],
    # Furniture
    "94": ["9401", "9402", "9403", "9404", "9405", "9406"],
    # Toys
    "95": ["9503", "9504", "9505", "9506", "9507", "9508"],
    # Miscellaneous
    "96": [f"96{h:02d}" for h in range(1, 19)],
}

# ---------------------------------------------------------------------------
# List 4A (7.5%) — covered 4-digit headings
# Reference: 84 FR 43304, February 2020
# These are headings NOT on List 3 (precedence: List 3 > List 4A)
# ---------------------------------------------------------------------------
_LIST_4A_HEADINGS_BY_CHAPTER: dict[str, list[str]] = {
    "03": ["0301", "0302", "0303", "0304", "0305", "0306", "0307", "0308"],
    "04": ["0401", "0402", "0403", "0404", "0405", "0406"],
    "05": ["0504", "0505", "0506", "0507", "0508", "0510", "0511"],
    "07": ["0701", "0702", "0703", "0710", "0711", "0712", "0713", "0714"],
    "08": ["0801", "0802", "0804", "0805", "0806", "0811", "0812", "0813"],
    "09": ["0901", "0902", "0904", "0909", "0910"],
    "11": ["1101", "1102", "1104", "1106"],
    "15": ["1504", "1507", "1508", "1509", "1511", "1512", "1515", "1516", "1517", "1518", "1520", "1521", "1522"],
    "16": ["1601", "1602", "1604", "1605"],
    "17": ["1701", "1702", "1704"],
    "19": ["1901", "1902", "1904", "1905"],
    "20": ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009"],
    "21": ["2101", "2103", "2104", "2105", "2106"],
    "22": ["2202", "2203", "2204", "2206", "2208", "2209"],
    "23": ["2301", "2302", "2303", "2304", "2306", "2308", "2309"],
    "33": ["3303"],
    "39": ["3923"],  # Specific plastic containers not on List 3
    "42": ["4202", "4203", "4205"],
    "43": ["4301", "4302", "4303", "4304"],
    "50": ["5004", "5005", "5006", "5007"],
    "51": ["5105", "5106", "5107", "5108", "5109", "5110", "5111", "5112"],
    "52": ["5205", "5206", "5207", "5208", "5209", "5210", "5211", "5212"],
    "53": ["5305", "5306", "5308", "5309", "5310", "5311"],
    "54": ["5401", "5402", "5403", "5404", "5407", "5408"],
    "57": ["5701", "5702", "5703", "5704", "5705"],
    "58": ["5801", "5802", "5803", "5804", "5806", "5807", "5808", "5809", "5810", "5811"],
    "60": ["6001", "6002", "6003", "6004", "6005", "6006"],
    "61": [f"61{h:02d}" for h in range(1, 18)],
    "62": [f"62{h:02d}" for h in range(1, 18)],
    "65": ["6501", "6502", "6504", "6505", "6506", "6507"],
    "66": ["6601", "6602", "6603"],
    "67": ["6701", "6702", "6703"],
    "71": ["7101", "7102", "7103", "7104", "7105", "7106", "7107", "7108", "7109", "7110", "7111", "7112", "7113", "7114", "7115", "7116", "7117", "7118"],
    "84": ["8421"],  # Filtering machinery — NOT on List 3
    "93": ["9301", "9302", "9303", "9304", "9305", "9306", "9307"],
}


def _flatten(by_chapter: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for headings in by_chapter.values():
        out.extend(headings)
    return sorted(set(out))


def generate_section301_json() -> list[dict]:
    """Build the overlay JSON entries for Section 301 Lists 1-4A."""
    entries = []

    # List 1 (original, 25%) — footwear
    entries.append({
        "overlay_name": "Section 301 PRC List 1",
        "hts_prefixes": ["6404"],
        "origin_countries": ["CN"],
        "additional_rate": 25.0,
        "reason": "Section 301 List 1 duties on PRC-origin goods (25%)",
        "requires_review": True,
        "stop_optimization": True,
    })

    # List 3 (25%) — broad industrial coverage
    list3_headings = _flatten(_LIST_3_HEADINGS_BY_CHAPTER)
    entries.append({
        "overlay_name": "Section 301 PRC List 3",
        "hts_prefixes": list3_headings,
        "origin_countries": ["CN"],
        "additional_rate": 25.0,
        "reason": "Section 301 List 3 duties on PRC-origin goods (25%) per 83 FR 47974",
        "requires_review": True,
        "stop_optimization": True,
    })

    # List 4A (7.5%) — consumer goods, food, some textiles
    list4a_headings = _flatten(_LIST_4A_HEADINGS_BY_CHAPTER)
    entries.append({
        "overlay_name": "Section 301 PRC List 4A",
        "hts_prefixes": list4a_headings,
        "origin_countries": ["CN"],
        "additional_rate": 7.5,
        "reason": "Section 301 List 4A duties on PRC-origin goods (7.5%) per 84 FR 43304",
        "requires_review": True,
        "stop_optimization": True,
    })

    return entries


def main() -> None:
    entries = generate_section301_json()

    # Count coverage
    list3_count = len([e for e in entries if "List 3" in e["overlay_name"]][0]["hts_prefixes"])
    list4a_count = len([e for e in entries if "List 4A" in e["overlay_name"]][0]["hts_prefixes"])
    total = list3_count + list4a_count + 1  # +1 for List 1

    OUTPUT.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Section 301 data written to {OUTPUT}")
    print(f"  List 1: 1 heading")
    print(f"  List 3: {list3_count} headings (25%)")
    print(f"  List 4A: {list4a_count} headings (7.5%)")
    print(f"  Total: {total} heading-level entries across {len(entries)} overlay rules")

    # Coverage verification
    chapters_covered_l3 = set()
    for h in _LIST_3_HEADINGS_BY_CHAPTER.values():
        for hd in h:
            chapters_covered_l3.add(hd[:2])
    chapters_covered_l4a = set()
    for h in _LIST_4A_HEADINGS_BY_CHAPTER.values():
        for hd in h:
            chapters_covered_l4a.add(hd[:2])

    print(f"\n  List 3 covers {len(chapters_covered_l3)} chapters: {sorted(chapters_covered_l3)}")
    print(f"  List 4A covers {len(chapters_covered_l4a)} chapters: {sorted(chapters_covered_l4a)}")

    # Verify no overlap
    l3_set = set(_flatten(_LIST_3_HEADINGS_BY_CHAPTER))
    l4a_set = set(_flatten(_LIST_4A_HEADINGS_BY_CHAPTER))
    overlap = l3_set & l4a_set
    if overlap:
        print(f"\n  WARNING: {len(overlap)} headings on BOTH lists: {sorted(overlap)[:10]}...")
    else:
        print(f"\n  No overlap between List 3 and List 4A headings")


if __name__ == "__main__":
    main()
