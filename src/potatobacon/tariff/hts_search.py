"""HTS text classification engine.

A real tariff engineer does not match products to HTS codes by keyword lookup.
They read the HTS schedule — the legal text of every heading — and apply the
General Rules of Interpretation to determine the most defensible classification.

This module builds a searchable index over the HTS heading descriptions
(the legal text in the schedule) and scores product descriptions against it
using TF-IDF so that any product can be routed to candidate headings even
without a prior HTS hint.

The output feeds:
  - The GRI engine (to determine which of several candidates actually wins)
  - The chapter_filter (to scope the atom set before Z3 solving)
  - POST /v1/engineering/classify (direct classification endpoint)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Stopwords (not informative for HTS classification)
# ---------------------------------------------------------------------------

_STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "for", "and", "or", "in", "to", "with",
    "at", "by", "from", "as", "is", "are", "be", "that", "this",
    "other", "not", "parts", "part", "accessories", "nesoi",
    "not", "elsewhere", "specified", "included", "thereof",
    "whether", "not", "also", "only", "having", "used",
    "such", "any", "all", "its", "their",
}

# Characters to strip from tokens

# ---------------------------------------------------------------------------
# Chapter-level vocabulary supplement
#
# The synthetic JSONL fixtures have minimal descriptive content ("Chapter 84
# synthetic item N").  This vocabulary map bridges the gap between product
# description keywords and HTS chapters, compensating for sparse training data.
#
# In production, with full USITC HTSA data, this vocabulary would be learned
# automatically from the heading descriptions.  Here we provide it explicitly
# from domain knowledge.
# ---------------------------------------------------------------------------

_CHAPTER_VOCABULARY: Dict[int, Set[str]] = {
    # Chapter 28-38: Chemicals and Pharmaceuticals
    28: {"chemical", "inorganic", "oxide", "acid", "compound", "mineral"},
    29: {"organic", "chemical", "compound", "resin", "polymer", "solvent"},
    30: {"pharmaceutical", "medicine", "drug", "tablet", "capsule", "vaccine", "medicament"},
    38: {"chemical", "adhesive", "lubricant", "preparation", "surface"},
    # Chapter 39: Plastics
    39: {"plastic", "polymer", "resin", "polyethylene", "polypropylene", "polyester",
         "pvc", "abs", "polycarbonate", "nylon", "injection", "molded", "extrusion",
         "housing", "casing", "container", "film", "sheet", "pipe", "tube"},
    # Chapter 40: Rubber
    40: {"rubber", "silicone", "gasket", "seal", "tire", "tyre", "elastomer",
         "vulcanized", "latex", "neoprene", "belt", "hose"},
    # Chapter 44: Wood
    44: {"wood", "lumber", "plywood", "veneer", "timber", "wooden", "hardwood"},
    # Chapter 48: Paper
    48: {"paper", "paperboard", "cardboard", "carton", "tissue", "packaging"},
    # Chapter 50-60: Textiles / Fabrics
    50: {"silk", "silkworm", "woven"},
    51: {"wool", "fleece", "cashmere", "worsted", "knit"},
    52: {"cotton", "denim", "woven", "fabric", "yarn", "thread"},
    54: {"polyester", "nylon", "synthetic", "fiber", "filament", "yarn"},
    55: {"synthetic", "staple", "polyester", "acrylic", "nylon", "rayon"},
    # Chapter 61-62: Apparel
    61: {"knit", "knitwear", "sweater", "pullover", "jersey", "shirt", "tshirt",
         "garment", "apparel", "clothing", "hosiery", "underwear", "socks"},
    62: {"woven", "shirt", "trouser", "jacket", "suit", "blazer", "coat",
         "garment", "apparel", "clothing", "dress", "skirt", "blouse"},
    # Chapter 64: Footwear
    64: {"shoe", "footwear", "boot", "sandal", "sneaker", "outsole", "upper",
         "leather", "rubber", "sole", "athletic", "running"},
    # Chapter 72-73: Iron and Steel
    72: {"steel", "iron", "alloy", "flat", "rolled", "bar", "rod", "wire",
         "stainless", "coil", "sheet", "plate", "ingot", "billet", "bloom"},
    73: {"steel", "iron", "tube", "pipe", "fastener", "bolt", "screw", "nut",
         "washer", "rivet", "chain", "nail", "fitting", "flange", "bracket",
         "casting", "forging", "structure", "fabricated", "wire"},
    # Chapter 74-83: Non-ferrous metals
    74: {"copper", "brass", "bronze", "wire", "tube", "pipe", "fitting"},
    76: {"aluminum", "aluminium", "extrusion", "sheet", "plate", "bar", "rod",
         "tube", "pipe", "frame", "window", "door", "structure", "foil"},
    # Chapter 84: Machinery
    84: {"machine", "machinery", "engine", "pump", "motor", "compressor", "generator",
         "turbine", "boiler", "valve", "bearing", "gear", "conveyor", "press",
         "drill", "milling", "lathe", "centrifugal", "hydraulic", "pneumatic",
         "industrial", "equipment", "mechanical", "injection", "diesel", "gasoline",
         "fuel", "filter", "heat", "exchanger", "reactor", "processing", "plant",
         "hvac", "refrigeration", "freezer", "dishwasher", "washing"},
    # Chapter 85: Electronics
    85: {"electronic", "electrical", "circuit", "cable", "wire", "connector",
         "battery", "charger", "monitor", "display", "computer", "laptop", "phone",
         "television", "camera", "microphone", "speaker", "transformer", "switch",
         "relay", "sensor", "led", "module", "semiconductor", "pcb", "usb",
         "inverter", "antenna", "radio", "telephone", "luminaire", "lighting",
         "motor", "generator", "panel", "lithium"},
    # Chapter 87: Vehicles
    87: {"vehicle", "automobile", "car", "truck", "bus", "motorcycle", "trailer",
         "automotive", "brake", "caliper", "transmission", "axle", "suspension",
         "chassis", "bumper", "airbag", "exhaust", "alternator", "starter",
         "radiator", "steering", "clutch", "differential", "piston", "crankshaft",
         "stroller", "carriage", "baby", "bicycle", "moped", "tractor"},
    # Chapter 90: Optical / Medical
    90: {"optical", "medical", "instrument", "surgical", "lens", "camera", "microscope",
         "telescope", "thermometer", "meter", "gauge", "sensor", "analyzer",
         "diagnostic", "laboratory", "scientific", "measurement", "spectroscope",
         "ophthalmologic", "dental", "forceps", "scissors", "scalpel", "fiber",
         "photographic"},
    # Chapter 94: Furniture
    94: {"furniture", "chair", "seat", "table", "desk", "sofa", "couch", "bed",
         "mattress", "shelf", "cabinet", "lamp", "lighting", "fixture", "upholstered",
         "wooden", "metal", "dining", "office", "bedroom"},
    # Chapter 95: Toys, games, sports
    95: {"toy", "game", "sport", "bicycle", "doll", "puzzle", "video", "console",
         "playground", "fishing", "golf", "tennis"},
}

_TOKEN_RE = re.compile(r"[^a-z0-9]")


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and blanks."""
    lower = text.lower()
    words = re.split(r"[^a-z0-9]+", lower)
    return [w for w in words if w and len(w) > 1 and w not in _STOPWORDS]


def _chapter_from_hts(hts_code: str) -> int:
    """Extract 2-digit chapter as int from HTS code string."""
    digits = re.sub(r"\D", "", hts_code or "")
    if len(digits) >= 2:
        try:
            return int(digits[:2])
        except ValueError:
            pass
    return 0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HTSSearchResult:
    """A single candidate HTS heading returned by the search engine."""

    hts_code: str            # Full code, e.g. "8471.30.01"
    heading: str             # 4-digit heading, e.g. "8471"
    chapter: int             # 2-digit chapter int
    description: str         # Full HTS legal description
    base_duty_rate: str      # Raw duty rate string from schedule
    score: float             # TF-IDF relevance score (higher = better match)
    matched_terms: List[str] # Query terms that matched this entry
    rationale: str           # Human-readable explanation of why this matched


@dataclass
class _IndexEntry:
    """Internal: a single indexed HTS line."""

    hts_code: str
    heading: str
    chapter: int
    description: str
    base_duty_rate: str
    tokens: List[str]       # All tokens from description (with repeats for TF)
    token_set: Set[str]     # Unique tokens


@dataclass
class HTSSearchIndex:
    """TF-IDF index over all loaded HTS heading descriptions.

    Built once at startup and cached.  Supports free-text product description
    search returning ranked candidate headings with match rationale.
    """

    _entries: List[_IndexEntry] = field(default_factory=list, repr=False)
    _idf: Dict[str, float] = field(default_factory=dict, repr=False)
    _total_docs: int = 0

    def _build_idf(self) -> None:
        """Compute IDF weights from the loaded corpus."""
        doc_freq: Dict[str, int] = {}
        for entry in self._entries:
            for token in entry.token_set:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        n = max(len(self._entries), 1)
        self._idf = {
            token: math.log((n + 1) / (df + 1)) + 1.0
            for token, df in doc_freq.items()
        }
        self._total_docs = n

    def search(
        self,
        description: str,
        top_n: int = 5,
        chapter_filter: Optional[List[int]] = None,
    ) -> List[HTSSearchResult]:
        """Return top-N HTS heading candidates for the given product description.

        Args:
            description: Free-text product description.
            top_n: Maximum number of results to return.
            chapter_filter: If provided, restrict results to these chapter numbers.

        Returns:
            List of HTSSearchResult sorted by score descending.
        """
        if not self._entries:
            return []

        query_tokens = _tokenize(description)
        if not query_tokens:
            return []

        # Build per-chapter vocabulary bonus from chapter_vocabulary
        # This compensates for synthetic fixture data with generic descriptions.
        chapter_bonus: Dict[int, float] = {}
        chapter_matched: Dict[int, List[str]] = {}
        for qt in set(query_tokens):
            for chapter, vocab in _CHAPTER_VOCABULARY.items():
                if qt in vocab:
                    chapter_bonus[chapter] = chapter_bonus.get(chapter, 0.0) + 0.15
                    chapter_matched.setdefault(chapter, []).append(qt)

        # Score each indexed entry using TF-IDF cosine-style scoring
        scored: List[Tuple[float, _IndexEntry, List[str]]] = []
        for entry in self._entries:
            if chapter_filter and entry.chapter not in chapter_filter:
                continue

            matched: List[str] = []
            score = 0.0
            entry_len = max(len(entry.tokens), 1)

            for qt in set(query_tokens):
                if qt not in entry.token_set:
                    continue
                # TF: count of qt in entry description / entry length
                tf = entry.tokens.count(qt) / entry_len
                idf = self._idf.get(qt, 1.0)
                score += tf * idf
                matched.append(qt)

            # Apply chapter-level vocabulary bonus
            bonus = chapter_bonus.get(entry.chapter, 0.0)
            if bonus > 0:
                score += bonus
                matched = list(set(matched) | set(chapter_matched.get(entry.chapter, [])))

            if score > 0:
                scored.append((score, entry, matched))

        scored.sort(key=lambda x: -x[0])

        results: List[HTSSearchResult] = []
        seen_headings: Set[str] = set()

        for score, entry, matched in scored:
            # Deduplicate by heading — one result per heading is sufficient
            if entry.heading in seen_headings:
                continue
            seen_headings.add(entry.heading)

            rationale = (
                f"Matched {len(matched)} term(s): {', '.join(sorted(matched)[:8])}. "
                f"HTS {entry.hts_code}: {entry.description[:100]}"
            )

            results.append(
                HTSSearchResult(
                    hts_code=entry.hts_code,
                    heading=entry.heading,
                    chapter=entry.chapter,
                    description=entry.description,
                    base_duty_rate=entry.base_duty_rate,
                    score=round(score, 6),
                    matched_terms=sorted(matched),
                    rationale=rationale,
                )
            )

            if len(results) >= top_n:
                break

        return results

    def search_by_chapter(self, chapter: int) -> List[HTSSearchResult]:
        """Return all indexed entries for a specific chapter."""
        results = []
        for entry in self._entries:
            if entry.chapter == chapter:
                results.append(
                    HTSSearchResult(
                        hts_code=entry.hts_code,
                        heading=entry.heading,
                        chapter=entry.chapter,
                        description=entry.description,
                        base_duty_rate=entry.base_duty_rate,
                        score=1.0,
                        matched_terms=[],
                        rationale=f"Chapter {chapter} entry",
                    )
                )
        return results

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def chapters_indexed(self) -> List[int]:
        return sorted({e.chapter for e in self._entries})


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def _load_chapter_jsonl(path: Path) -> List[_IndexEntry]:
    """Load a single chapter JSONL file into index entries."""
    entries: List[_IndexEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            hts_code = str(record.get("hts_code") or "")
            description = str(record.get("description") or "")
            base_duty_rate = str(record.get("base_duty_rate") or "Free")
            heading = str(record.get("heading") or "")

            if not hts_code or not description:
                continue

            chapter = _chapter_from_hts(hts_code)
            if not heading and len(hts_code.replace(".", "").replace(" ", "")) >= 4:
                digits = re.sub(r"\D", "", hts_code)
                heading = digits[:4]

            tokens = _tokenize(description)
            entries.append(
                _IndexEntry(
                    hts_code=hts_code,
                    heading=heading,
                    chapter=chapter,
                    description=description,
                    base_duty_rate=base_duty_rate,
                    tokens=tokens,
                    token_set=set(tokens),
                )
            )
    return entries


def build_search_index(chapter_paths: Dict[int, Path]) -> HTSSearchIndex:
    """Build a TF-IDF search index from a set of HTS chapter JSONL files.

    Args:
        chapter_paths: Mapping of chapter number → path to .jsonl file.

    Returns:
        Populated HTSSearchIndex ready to query.
    """
    index = HTSSearchIndex()
    for chapter_num in sorted(chapter_paths.keys()):
        path = chapter_paths[chapter_num]
        if not path.exists():
            continue
        entries = _load_chapter_jsonl(path)
        index._entries.extend(entries)

    index._build_idf()
    return index


# ---------------------------------------------------------------------------
# Module-level singleton index (loaded once, shared by all callers)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FULL_CHAPTERS_DIR = _REPO_ROOT / "data" / "hts_extract" / "full_chapters"
_SAMPLE_LINES_PATH = _REPO_ROOT / "data" / "hts_extract" / "hts_lines_sample.jsonl"

# All available chapter JSONL files
_DEFAULT_CHAPTER_PATHS: Dict[int, Path] = {}
for _ch in [39, 64, 72, 73, 76, 84, 85, 87, 90, 94]:
    _p = _FULL_CHAPTERS_DIR / f"ch{_ch}.jsonl"
    if _p.exists():
        _DEFAULT_CHAPTER_PATHS[_ch] = _p

# Also index the sample lines file (covers ch64, ch73, ch85 from HTS_US_2025_SLICE)
_SLICE_EXTRAS: Optional[Path] = _SAMPLE_LINES_PATH if _SAMPLE_LINES_PATH.exists() else None

_INDEX: Optional[HTSSearchIndex] = None


def _load_sample_as_entries(path: Path) -> List[_IndexEntry]:
    """Load the hts_lines_sample.jsonl (which uses a slightly different schema)."""
    # The sample file uses TariffLine schema — load as generic JSONL
    return _load_chapter_jsonl(path)


def get_search_index() -> HTSSearchIndex:
    """Return the module-level singleton HTS search index, building if needed."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    index = build_search_index(_DEFAULT_CHAPTER_PATHS)

    # Supplement with the sample lines (covers ch64/73/85)
    if _SLICE_EXTRAS:
        extra_entries = _load_sample_as_entries(_SLICE_EXTRAS)
        # Avoid duplicating entries already in the full chapters
        existing_codes = {e.hts_code for e in index._entries}
        for entry in extra_entries:
            if entry.hts_code not in existing_codes:
                index._entries.append(entry)
        index._build_idf()

    _INDEX = index
    return _INDEX


def search_hts_by_description(
    description: str,
    top_n: int = 5,
    chapter_filter: Optional[List[int]] = None,
) -> List[HTSSearchResult]:
    """Search the HTS schedule by free-text product description.

    This is the primary entry point for classification when no HTS hint is
    provided.  Returns ranked candidate headings with match rationale.

    Args:
        description: Free-text product description (e.g. "USB-C charging cable
            for smartphones").
        top_n: Maximum number of candidate headings to return.
        chapter_filter: Optionally restrict search to specific chapters.

    Returns:
        List of HTSSearchResult, best match first.
    """
    return get_search_index().search(
        description=description,
        top_n=top_n,
        chapter_filter=chapter_filter,
    )


def top_chapters_for_description(description: str, top_n: int = 3) -> List[int]:
    """Return the top candidate chapter numbers for a product description.

    Useful for pre-filtering atoms before the Z3 solve step.
    """
    results = search_hts_by_description(description, top_n=top_n * 3)
    seen: Set[int] = set()
    chapters: List[int] = []
    for r in results:
        if r.chapter not in seen:
            seen.add(r.chapter)
            chapters.append(r.chapter)
            if len(chapters) >= top_n:
                break
    return chapters
