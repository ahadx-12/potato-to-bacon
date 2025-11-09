#!/usr/bin/env python3
"""Shared leverage covenant extraction utilities for CALE validation."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests

from bs4 import BeautifulSoup, SoupStrainer

try:
    from potatobacon.cale.finance.numeric import extract_numeric_covenants
except Exception:  # pragma: no cover - fallback when package not installed
    def extract_numeric_covenants(_text: str) -> List[Dict[str, object]]:
        return []

from tools.sec_fetch import (
    TICKER_TO_CIK,
    ensure_filing_html,
    load_submissions,
    pick_last_10k_10q_before,
)

from potatobacon.cale.finance import authority, dedup, docio, sectionizer, tables
from potatobacon.cale.finance.docio import Doc, DocBlock

CFG_PATH = Path("configs/finance.yml")

# ---------------------------------------------------------------------------
# Pattern banks & heuristics
# ---------------------------------------------------------------------------


DEFAULT_FINANCE_TERMS: Tuple[str, ...] = (
    "covenant",
    "debt",
    "liquidity",
    "ratio",
    "credit facility",
    "borrowing",
    "leverage",
    "default",
    "revolver",
    "senior notes",
    "maturities",
    "ebitda add-back",
)

DEFAULT_TOC_PATTERNS: Tuple[str, ...] = (
    "table of contents",
    "index of",
    "where you can find",
    "forward-looking statements",
)

DEFAULT_BYPASS_CUES: Tuple[str, ...] = (
    "except that",
    "provided, however",
    "provided however",
    "notwithstanding",
    "basket",
)

DEFAULT_NEGATION_TERMS: Tuple[str, ...] = (
    "except that",
    "provided, however",
    "provided however",
    "notwithstanding",
    "unless",
    "except to the extent",
    "other than",
    "save for",
    "basket",
)

DEFAULT_SECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "CREDIT_AGREEMENT": (
        "credit agreement",
        "covenants",
        "revolver",
        "ebitda add-backs",
        "ebitda add backs",
        "liquidity and capital resources — borrowings",
        "liquidity and capital resources - borrowings",
        "debt maturities",
        "credit agreement exhibit",
        "exhibit 10",
    ),
    "INDENTURE_NOTES": ("senior notes", "indenture"),
    "LIQUIDITY": (
        "liquidity and capital resources",
        "debt maturities",
        "borrowings",
    ),
}


@dataclass(frozen=True)
class SectionContext:
    """Metadata describing a detected finance-relevant section."""

    key: str
    title: str
    canonical: str
    weight_key: str
    level: Optional[int] = None
    path: Optional[str] = None
    doc_kind: Optional[str] = None
    anchor: Optional[str] = None
    rank: Optional[int] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class ExtractionPair:
    """Container for an obligation/permission pair with section metadata."""

    obligation: str
    permission: str
    section: Optional[SectionContext] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __iter__(self) -> Iterable[str]:
        return iter((self.obligation, self.permission))

    def __getitem__(self, index: int) -> object:
        if index == 0:
            return self.obligation
        if index == 1:
            return self.permission
        if index == 2:
            return self.section
        raise IndexError(index)


@dataclass
class SentenceInfo:
    text: str
    order: int
    block_index: int
    context: SectionContext
    is_obligation: bool
    is_permission: bool
    has_coref: bool
    obligation_rules: Tuple[str, ...] = field(default_factory=tuple)
    permission_rules: Tuple[str, ...] = field(default_factory=tuple)
    bypass_terms: Tuple[str, ...] = field(default_factory=tuple)
    lexicon_hits: Tuple[str, ...] = field(default_factory=tuple)


def _normalize_match_text(value: str) -> str:
    """Normalise a string for substring comparisons."""

    value = value.upper()
    value = re.sub(r"[\u2010-\u2015]", "-", value)
    value = re.sub(r"[^A-Z0-9\- ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _compile_phrase(phrase: str) -> re.Pattern[str]:
    """Compile a loosely matched phrase regex (ignoring punctuation)."""

    tokens = re.findall(r"[A-Za-z0-9']+", phrase)
    if not tokens:
        return re.compile(re.escape(phrase), re.I)
    pattern = r"\b" + r"\W+".join(re.escape(token) for token in tokens) + r"\b"
    return re.compile(pattern, re.I)


COVENANT_OBL = re.compile(
    r"""
    \b(
        must
        |shall
        |require(?:s|d)?
        |is\s+required\s+to
        |are\s+required\s+to
        |obligat(?:es|ed)\s+to
        |agrees?\s+to
    )\b
    .*?
    \b(
        maintain
        |comply
        |keep
        |meet
        |not\s+exceed
        |remain(?:s)?\s+(?:above|below)
        |at\s+least
        |no\s+more\s+than
        |minimum
        |maximum
        |pay
        |deliver
        |provide
    )\b
    .*?
    (
        \d+(?:\.\d+)?\s*(?:x|times|percent|%)
        |[$€£]\s*\d[\d,\.]*
        |\b(?:ratio|covenant|interest|liquidity|availability)\b
    )
    """,
    re.I | re.X,
)

PERM_BYPASS = re.compile(
    r"\b(may|permitted to|can)\b.*\b(borrow|incur|issue|draw|increase)\b.*\b("
    r"unless|except|subject to|provided that|so long as|absent a default)\b",
    re.I,
)

PERM_WEAK = re.compile(
    r"\b(may|permitted to|can)\b.*\b(borrow|incur (?:additional )?indebtedness|issue notes?|draw on (?:its )?credit (?:facility|facilities)|increase (?:its )?debt)\b",
    re.I,
)

PERM_RELIEF_VERB = re.compile(
    r"\b(may|can|is permitted to)\b.*?(?:be\s+)?(waive(?:d)?|amend(?:ed)?|modif(?:y|ied)|forgiv(?:e|en)|consent(?:ed)? to)",
    re.I,
)

RELIEF_OBJECT_RE = re.compile(r"\b(covenant|compliance|default|requirement)\b", re.I)

COMPLIANCE_OBL = re.compile(
    r"\b(is|are|was|were|remains|remained)\b.*\b(in|within)\s+compliance\b.*\b(covenants?|ratios?|agreements?)\b",
    re.I,
)

ASPIRATIONAL = re.compile(
    r"\b(must|shall)\b.*\b(maintain|preserve|ensure|pursue|support|target|aim)\b.*\b(liquidity|cash flow|"
    r"financial flexibility|investment grade|strong|prudent|sound)\b(?!.*\d)(?!.*[$€£])",
    re.I,
)

INVESTMENT_GRADE = set(
    "AAPL PEP WMT PG GOOGL MA V HD KO COST MRK UNH LMT UPS TGT ORCL PFE NVDA MSFT JNJ".split()
)

NEG_LEVERAGE_CUES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(breach|violation|default|waiver|amendment)\b", re.I),
    re.compile(r"\b(exceed(ed)?|above)\b.*\b(leverage|debt|coverage)\b", re.I),
    re.compile(r"\bgoing concern|substantial doubt\b", re.I),
    re.compile(r"\bmaterial adverse\b", re.I),
)

POS_LEVERAGE_CUES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(in compliance|remain in compliance|within covenant)\b", re.I),
    re.compile(r"\bno default\b", re.I),
    re.compile(r"\bstrong\b.*\bliquidity\b", re.I),
    re.compile(r"\badequate\b.*\bcash\b", re.I),
)

ASSET_COVERAGE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(asset coverage|borrowing base|collateral coverage|secured by)\b", re.I),
)

LIQUIDITY_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\bliquidity\b", re.I),
    re.compile(r"\bcash (on hand|balance)\b", re.I),
    re.compile(r"\bavailability under\b", re.I),
)

THRESHOLD_PAT = re.compile(
    r"(\d+(\.\d+)?\s*(x|times|%|percent)|[$€£]\s*\d[\d,\.]*|\bratio\b|\bthreshold\b)",
    re.I,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z(])")

HEADER_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
PARAGRAPH_TAGS = ("p", "li", "div")
FINANCE_SECTION_KEYS: Set[str] = {
    "CREDIT_AGREEMENT",
    "INDENTURE_NOTES",
    "LIQUIDITY",
    "MDNA",
    "NOTES_TO_FS",
    "RISK_FACTORS",
}

SEC_ITEM_RE = re.compile(r"item\s+(\d+[A-Z]?)", re.I)
SEC_ITEM_CANONICAL = {
    "1A": "RISK_FACTORS",
    "2": "MDNA",
    "7": "MDNA",
    "7A": "RISK_FACTORS",
    "8": "NOTES_TO_FS",
}

COREF_REF = re.compile(
    r"\b(?:such|that|the foregoing) (?:covenant|obligation|restriction|requirement|provision)\b",
    re.I,
)

SKIP_TAGS = {"script", "style", "nav", "footer", "header"}

def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _header_level_from_tag(tag: Optional[str]) -> Optional[int]:
    if not tag:
        return None
    tag_lower = tag.lower()
    if len(tag_lower) >= 2 and tag_lower[0] == "h" and tag_lower[1].isdigit():
        try:
            return int(tag_lower[1])
        except ValueError:
            return None
    return None


def _form_to_doc_kind(form: Optional[str]) -> str:
    if not form:
        return "OTHER"
    upper = form.upper()
    if "10-K" in upper:
        return "10-K"
    if "10-Q" in upper:
        return "10-Q"
    if "INDENTURE" in upper:
        return "INDENTURE"
    if "CREDIT" in upper or "LOAN" in upper:
        return "CREDIT_AGREEMENT"
    return "OTHER"


def _canonicalize_section(title: str, doc_kind: str, path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    candidates = [title]
    if path and path not in candidates:
        candidates.append(path)
    for candidate in candidates:
        normalized = _normalize_match_text(candidate)
        for alias, canonical in SECTION_ALIAS_INDEX:
            if alias and alias in normalized:
                return canonical, canonical
        upper = candidate.upper()
        if "LIQUIDITY" in upper or "CAPITAL RESOURCES" in upper:
            return "LIQUIDITY", "LIQUIDITY"
        if ("MANAGEMENT" in upper and "DISCUSSION" in upper) or "MD&A" in upper:
            return "MDNA", "MDNA"
        if "RESULTS OF OPERATIONS" in upper or "FINANCIAL CONDITION" in upper:
            return "MDNA", "MDNA"
        if "RISK" in upper and "FACTOR" in upper:
            return "RISK_FACTORS", "RISK_FACTORS"
        if "NOTES" in upper and "FINANCIAL STATEMENT" in upper:
            return "NOTES_TO_FS", "NOTES_TO_FS"
        if "CREDIT" in upper and ("AGREEMENT" in upper or "FACILITY" in upper or "COVENANT" in upper):
            return "CREDIT_AGREEMENT", "CREDIT_AGREEMENT"
        if "INDENTURE" in upper:
            return "INDENTURE_NOTES", "INDENTURE_NOTES"
        if "DEBT" in upper and "COVENANT" in upper:
            return "CREDIT_AGREEMENT", "CREDIT_AGREEMENT"
    match = SEC_ITEM_RE.search(title.upper())
    if not match and path:
        match = SEC_ITEM_RE.search(path.upper())
    if match:
        canonical = SEC_ITEM_CANONICAL.get(match.group(1))
        if canonical:
            return canonical, canonical
    if doc_kind in {"CREDIT_AGREEMENT", "INDENTURE"}:
        return doc_kind, doc_kind
    return None, None


def _build_doc_from_html(html_text: str, form: Optional[str]) -> Doc:
    parser = "lxml"
    allowed_tags = set(HEADER_TAGS) | set(PARAGRAPH_TAGS) | {"html", "body", "div", "span", "table", "thead", "tbody", "tr", "td", "th", "ol", "ul", "li"}
    strainer = SoupStrainer(name=lambda tag: tag in allowed_tags if tag else False)
    try:
        soup = BeautifulSoup(html_text or "", parser, parse_only=strainer)
    except Exception:  # pragma: no cover - fallback for environments without lxml
        soup = BeautifulSoup(html_text or "", "html.parser", parse_only=strainer)
    for tag in SKIP_TAGS:
        for node in soup.find_all(tag):
            node.decompose()
    body = soup.body or soup
    blocks: List[DocBlock] = []
    header_stack: Dict[int, str] = {}

    def current_path() -> Optional[str]:
        if not header_stack:
            return None
        parts = [header_stack[level] for level in sorted(header_stack)]
        return " > ".join(part for part in parts if part)

    for element in body.find_all(list(HEADER_TAGS) + list(PARAGRAPH_TAGS), recursive=True):
        if element.name in SKIP_TAGS:
            continue
        if element.name in HEADER_TAGS:
            text = _clean_text(element.get_text(" ", strip=True))
            if not text:
                continue
            level = _header_level_from_tag(element.name)
            if level is not None:
                for existing in list(header_stack.keys()):
                    if existing >= level:
                        header_stack.pop(existing, None)
                header_stack[level] = text
            path = current_path()
            meta = {"html_tag": element.name}
            if level is not None:
                meta["level"] = level
            if path:
                meta["header_path"] = path
            blocks.append(DocBlock("header", text=text, meta=meta))
            continue
        if element.find_parent("table") is not None:
            continue
        if element.name == "div" and element.find(list(HEADER_TAGS) + ["p", "li", "div"], recursive=False):
            continue
        if element.name == "li" and element.find("p"):
            continue
        text = _clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        meta = {"html_tag": element.name}
        path = current_path()
        if path:
            meta["header_path"] = path
        if header_stack:
            meta["level"] = max(header_stack)
        blocks.append(DocBlock("paragraph", text=text, meta=meta))
    doc_kind = _form_to_doc_kind(form)
    return Doc(src_path="inline", doc_kind=doc_kind, blocks=blocks)


def extract_pairs_from_doc(doc: Doc) -> List[ExtractionPair]:
    sections = sectionizer.find_sections(doc, CFG)
    section_lookup = _assign_section_lookup(doc, sections)
    if not section_lookup:
        section_lookup = _fallback_section_lookup(doc)
    if not section_lookup:
        return []

    sentences: List[SentenceInfo] = []
    order = 0
    for idx, block in enumerate(doc.blocks):
        context = section_lookup.get(idx)
        if context is None or block.kind != "paragraph":
            continue
        for sentence in _split_sentences(block.text):
            if not sentence:
                continue
            positive_cue = any(pat.search(sentence) for pat in POS_LEVERAGE_CUES)
            lexicon_hits = _collect_finance_hits(sentence)
            obligation_rules: List[str] = []
            permission_rules: List[str] = []
            is_obligation = (
                (COVENANT_OBL.search(sentence) is not None)
                or (COMPLIANCE_OBL.search(sentence) is not None)
                or positive_cue
            ) and not ASPIRATIONAL.search(sentence)
            if COVENANT_OBL.search(sentence):
                obligation_rules.append("covenant_obligation")
            if COMPLIANCE_OBL.search(sentence):
                obligation_rules.append("compliance_status")
            if positive_cue:
                obligation_rules.append("positive_liquidity_cue")
            has_relief = bool(
                PERM_RELIEF_VERB.search(sentence)
                and RELIEF_OBJECT_RE.search(sentence)
            )
            is_permission = bool(
                PERM_BYPASS.search(sentence)
                or PERM_WEAK.search(sentence)
                or has_relief
            )
            if PERM_BYPASS.search(sentence):
                permission_rules.append("permission_bypass")
            if PERM_WEAK.search(sentence):
                permission_rules.append("permission_weak")
            if has_relief:
                permission_rules.append("relief_object")
            bypass_terms = _extract_bypass_terms(sentence)
            if bypass_terms:
                permission_rules.append("bypass_lexicon")
            if is_obligation and not obligation_rules and lexicon_hits:
                obligation_rules.append("lexicon_seed")
            if is_permission and not permission_rules and lexicon_hits:
                permission_rules.append("lexicon_seed")
            has_coref = bool(COREF_REF.search(sentence))
            sentences.append(
                SentenceInfo(
                    text=sentence,
                    order=order,
                    block_index=idx,
                    context=context,
                    is_obligation=is_obligation,
                    is_permission=is_permission,
                    has_coref=has_coref,
                    obligation_rules=tuple(sorted(set(obligation_rules))),
                    permission_rules=tuple(sorted(set(permission_rules))),
                    bypass_terms=bypass_terms,
                    lexicon_hits=lexicon_hits,
                )
            )
            order += 1

    if not any(info.is_obligation for info in sentences):
        fallback_order = max((info.order for info in sentences), default=-1) + 1
        for idx, block in enumerate(doc.blocks):
            context = section_lookup.get(idx)
            if context is None or block.kind != "paragraph":
                continue
            text = block.text.strip()
            if not text:
                continue
            hits = _collect_finance_hits(text)
            if not hits:
                continue
            info = SentenceInfo(
                text=text,
                order=fallback_order,
                block_index=idx,
                context=context,
                is_obligation=True,
                is_permission=False,
                has_coref=False,
                obligation_rules=("fallback_section",),
                lexicon_hits=hits,
            )
            sentences.append(info)
            fallback_order += 1
            if fallback_order >= 3:
                break
        if not any(info.is_obligation for info in sentences):
            return []

    obligations_by_section: Dict[str, List[SentenceInfo]] = defaultdict(list)
    permissions_by_section: Dict[str, List[SentenceInfo]] = defaultdict(list)
    for info in sentences:
        if info.is_obligation:
            obligations_by_section[info.context.key].append(info)
        if info.is_permission:
            permissions_by_section[info.context.key].append(info)

    pairs: List[ExtractionPair] = []
    seen: Set[Tuple[str, str, str]] = set()
    for section_key, perms in permissions_by_section.items():
        obligations = obligations_by_section.get(section_key, [])
        if not obligations:
            continue
        obligations_sorted = sorted(obligations, key=lambda item: item.order)
        for perm in sorted(perms, key=lambda item: item.order):
            candidate_obligations: List[SentenceInfo] = []
            if perm.has_coref:
                for obligation in reversed(
                    [item for item in obligations_sorted if item.order < perm.order]
                ):
                    candidate_obligations.append(obligation)
                    break
            if not candidate_obligations:
                candidate_obligations = [
                    obligation
                    for obligation in obligations_sorted
                    if abs(perm.order - obligation.order) <= 5
                ]
            if not candidate_obligations:
                continue
            for obligation in candidate_obligations[:4]:
                pair_key = (obligation.text, perm.text, section_key)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                numeric_hits, numeric_strength, numeric_conf, numeric_negations = summarise_numeric_covenants(
                    obligation.text, perm.text
                )
                combined_bypass = tuple(
                    sorted(set(obligation.bypass_terms + perm.bypass_terms))
                )
                metadata = {
                    "section_key": section_key,
                    "section_canonical": obligation.context.canonical,
                    "section_title": obligation.context.title,
                    "section_path": obligation.context.path,
                    "section_rank": obligation.context.rank,
                    "section_level": obligation.context.level,
                    "clause_linked": perm.has_coref and obligation.order <= perm.order,
                    "order_distance": perm.order - obligation.order,
                    "link_strategy": "coreference"
                    if perm.has_coref and obligation.order <= perm.order
                    else "window",
                    "obligation_rules": list(obligation.obligation_rules),
                    "permission_rules": list(perm.permission_rules),
                    "obligation_lexicon_hits": list(obligation.lexicon_hits),
                    "permission_lexicon_hits": list(perm.lexicon_hits),
                    "obligation_block_index": obligation.block_index,
                    "permission_block_index": perm.block_index,
                    "bypass_terms": list(combined_bypass),
                    "numeric_hits": numeric_hits,
                    "numeric_strength": numeric_strength,
                    "numeric_confidence": numeric_conf,
                    "numeric_negations": numeric_negations,
                }
                pairs.append(
                    ExtractionPair(
                        obligation=obligation.text,
                        permission=perm.text,
                        section=obligation.context,
                        metadata=metadata,
                    )
                )
    if not pairs:
        for section_key, obligations in obligations_by_section.items():
            if not obligations:
                continue
            for obligation in obligations[:3]:
                pair_key = (obligation.text, obligation.text, section_key)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                numeric_hits, numeric_strength, numeric_conf, numeric_negations = summarise_numeric_covenants(
                    obligation.text, obligation.text
                )
                pairs.append(
                    ExtractionPair(
                        obligation=obligation.text,
                        permission=obligation.text,
                        section=obligation.context,
                        metadata={
                            "section_key": section_key,
                            "section_canonical": obligation.context.canonical,
                            "section_title": obligation.context.title,
                            "section_path": obligation.context.path,
                            "section_rank": obligation.context.rank,
                            "section_level": obligation.context.level,
                            "link_strategy": "synthetic",
                            "obligation_rules": list(obligation.obligation_rules),
                            "permission_rules": ["synthetic"],
                            "obligation_lexicon_hits": list(obligation.lexicon_hits),
                            "permission_lexicon_hits": list(obligation.lexicon_hits),
                            "obligation_block_index": obligation.block_index,
                            "permission_block_index": obligation.block_index,
                            "bypass_terms": list(obligation.bypass_terms),
                            "numeric_hits": numeric_hits,
                            "numeric_strength": numeric_strength,
                            "numeric_confidence": numeric_conf,
                            "numeric_negations": numeric_negations,
                            "synthetic_permission": True,
                        },
                    )
                )
    return pairs


def _assign_section_lookup(doc: Doc, sections: Sequence[sectionizer.Section]) -> Dict[int, SectionContext]:
    lookup: Dict[int, SectionContext] = {}
    rank = 0
    for section in sections:
        if section.start_block >= len(doc.blocks):
            continue
        header_block = doc.blocks[section.start_block]
        path = header_block.meta.get("header_path") if isinstance(header_block.meta, dict) else None
        level_meta = header_block.meta.get("level") if isinstance(header_block.meta, dict) else None
        level = int(level_meta) if isinstance(level_meta, int) else None
        if level is None:
            level = _header_level_from_tag(header_block.meta.get("html_tag")) if isinstance(header_block.meta, dict) else None
        canonical, weight_key = _canonicalize_section(section.title, doc.doc_kind, path)
        if not canonical or canonical not in FINANCE_SECTION_KEYS:
            continue
        context = SectionContext(
            key=section.key,
            title=section.title,
            canonical=canonical,
            weight_key=weight_key or canonical,
            level=level,
            path=path or section.title,
            doc_kind=doc.doc_kind,
            anchor=section.anchor,
            rank=rank,
            score=section.score,
        )
        rank += 1
        for idx in _iter_section_blocks(doc, section):
            if idx >= len(doc.blocks):
                break
            block = doc.blocks[idx]
            if block.kind != "paragraph":
                continue
            existing = lookup.get(idx)
            if existing is None or (existing.score or -1.0) < (section.score or 0.0):
                lookup[idx] = context
    return lookup


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    if len(parts) == 1:
        return [parts[0].strip()] if parts[0].strip() else []
    return [part.strip() for part in parts if part.strip()]


TOC_LEADER_RE = re.compile(r"\.{2,}\s*\d{1,3}$")
TOC_ITEM_RE = re.compile(r"^(?:item|part)\s+\d+[A-Z]?(?:\.|\s|$)", re.I)


def _collect_finance_hits(text: str) -> Tuple[str, ...]:
    lowered = text.lower()
    hits: Set[str] = set()
    for term in FINANCE_TERMS:
        if not term:
            continue
        if " " in term:
            if term in lowered:
                hits.add(term)
            continue
        pattern = re.compile(rf"\\b{re.escape(term)}\\b")
        if pattern.search(lowered):
            hits.add(term)
    return tuple(sorted(hits))


def _extract_bypass_terms(text: str) -> Tuple[str, ...]:
    found: Set[str] = set()
    for pattern, label in BYPASS_REGEXES:
        if pattern.search(text):
            found.add(label)
    return tuple(sorted(found))


def _looks_like_toc(text: Optional[str]) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    for pattern in TOC_REGEXES:
        if pattern.search(stripped):
            return True
    if TOC_LEADER_RE.search(stripped):
        return True
    if TOC_ITEM_RE.search(stripped) and TOC_LEADER_RE.search(stripped):
        return True
    segments = [segment.strip() for segment in stripped.split(">") if segment.strip()]
    for segment in segments:
        if TOC_ITEM_RE.search(segment) and TOC_LEADER_RE.search(segment):
            return True
    if stripped.count(".") >= 2 and sum(ch.isdigit() for ch in stripped[-6:]) >= 2:
        return True
    return False


def _fallback_section_lookup(doc: Doc) -> Dict[int, SectionContext]:
    lookup: Dict[int, SectionContext] = {}
    contexts: Dict[str, SectionContext] = {}
    rank = 0
    for idx, block in enumerate(doc.blocks):
        if block.kind != "paragraph":
            continue
        meta = block.meta if isinstance(block.meta, dict) else {}
        path = meta.get("header_path") or block.text
        text = block.text or ""
        if (_looks_like_toc(path) or _looks_like_toc(text)) and not _collect_finance_hits(text):
            continue
        canonical, weight_key = _canonicalize_section(path, doc.doc_kind, path)
        if not canonical or canonical not in FINANCE_SECTION_KEYS:
            continue
        key = f"{canonical}:{path}" if path else canonical
        context = contexts.get(key)
        if context is None:
            level_meta = meta.get("level")
            level = int(level_meta) if isinstance(level_meta, int) else None
            context = SectionContext(
                key=f"fallback-{rank}",
                title=path,
                canonical=canonical,
                weight_key=weight_key or canonical,
                level=level,
                path=path,
                doc_kind=doc.doc_kind,
                anchor=None,
                rank=rank,
                score=None,
            )
            contexts[key] = context
            rank += 1
        lookup[idx] = context
    return lookup


def extract_pairs_from_html(html_text: str, form: Optional[str] = None) -> List[ExtractionPair]:
    """Return obligation/permission pairs restricted to finance sections."""

    doc = _build_doc_from_html(html_text, form)
    return extract_pairs_from_doc(doc)


def damp_investment_grade(ticker: str, cce: float) -> float:
    if ticker.upper() in INVESTMENT_GRADE and cce > 0.15:
        return float(np.clip(cce * 0.3, 0.0, 1.0))
    return float(np.clip(cce, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Numeric normalisation helpers
# ---------------------------------------------------------------------------


NUMERIC_WINDOW_CHARS = 48


def _normalize_numeric_scalar(value: Optional[object], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    unit_norm = (unit or "").upper()
    if unit_norm in {"USD", "$"}:
        return scalar / 1_000_000.0
    if unit_norm in {"PERCENT", "%"}:
        return scalar / 100.0
    if unit_norm == "BPS":
        return scalar / 10_000.0
    return scalar


def _normalize_numeric_entry(entry: Dict[str, object], default_unit: Optional[str] = None) -> List[float]:
    unit = entry.get("unit") if isinstance(entry, dict) else None
    if unit is None:
        unit = default_unit
    values: List[float] = []
    if isinstance(entry, dict):
        primary = _normalize_numeric_scalar(entry.get("value"), unit)
        if primary is not None:
            values.append(primary)
        min_value = _normalize_numeric_scalar(entry.get("min"), unit)
        max_value = _normalize_numeric_scalar(entry.get("max"), unit)
        if min_value is not None and max_value is not None:
            values.append(0.5 * (min_value + max_value))
        elif min_value is not None:
            values.append(min_value)
        elif max_value is not None:
            values.append(max_value)
        legs = entry.get("legs") if isinstance(entry.get("legs"), list) else []
        for leg in legs:
            if isinstance(leg, dict):
                values.extend(_normalize_numeric_entry(leg, unit))
    return [float(v) for v in values if isinstance(v, (int, float))]


def _has_negation_window(sentence: str, spans: Sequence[Sequence[int]]) -> bool:
    lowered = sentence.lower()
    if not lowered:
        return False
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            continue
        window_start = max(0, start - NUMERIC_WINDOW_CHARS)
        window_end = min(len(sentence), end + NUMERIC_WINDOW_CHARS)
        window = lowered[window_start:window_end]
        if any(term in window for term in NEGATION_TERMS):
            return True
    return False


def summarise_numeric_covenants(obligation: str, permission: str) -> Tuple[int, float, float, int]:
    results: List[Dict[str, object]] = []
    for text in (obligation, permission):
        try:
            results.extend(extract_numeric_covenants(text))
        except Exception:
            continue
    count = len(results)
    if count == 0:
        return 0, 0.0, 0.0, 0
    values: List[float] = []
    confidences: List[float] = []
    negated = 0
    for item in results:
        values.extend(_normalize_numeric_entry(item))
        conf = item.get("confidence")
        try:
            confidences.append(float(conf))
        except (TypeError, ValueError):
            pass
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        sentence = str(raw.get("sentence", ""))
        spans = raw.get("spans") if isinstance(raw, dict) else []
        if sentence and _has_negation_window(sentence, spans if isinstance(spans, list) else []):
            negated += 1
    avg_value = float(statistics.fmean(values)) if values else 0.0
    avg_conf = float(statistics.fmean(confidences)) if confidences else 0.0
    if negated and avg_value:
        penalty = max(0.2, 1.0 - 0.25 * min(negated, 3))
        avg_value *= penalty
    return count, avg_value, avg_conf, negated


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config() -> Dict[str, object]:
    spec = importlib.util.find_spec("yaml")
    if spec is None or not CFG_PATH.exists():
        return {
            "weights": {
                "authority": {
                    "CREDIT_AGREEMENT": 2.0,
                    "INDENTURE_NOTES": 1.8,
                    "NOTES_TO_FS": 1.2,
                    "RISK_FACTORS": 0.8,
                    "LIQUIDITY": 0.6,
                    "MDNA": 0.4,
                },
                "temporal_gamma": 0.30,
            },
            "dv": {
                "going_concern": 0.80,
                "breach_keywords": 0.60,
                "covenant_words": 0.40,
                "ratio_weakness": 0.50,
            },
            "validation": {
                "auc_strong": 0.75,
                "auc_real": 0.70,
                "auc_weak": 0.65,
            },
        }
    yaml = importlib.import_module("yaml")
    with CFG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


CFG = _load_config()


def _iter_config_list(value: object) -> Iterable[str]:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                yield text


LEXICON_CFG = CFG.get("lexicon", {}) if isinstance(CFG.get("lexicon"), dict) else {}

FINANCE_TERMS: Tuple[str, ...] = tuple(
    sorted(
        {
            term.lower()
            for term in DEFAULT_FINANCE_TERMS
        }
        | {term.lower() for term in _iter_config_list(LEXICON_CFG.get("finance_terms"))}
    )
)

FALLBACK_KEYWORDS: Tuple[str, ...] = FINANCE_TERMS

TOC_PATTERN_STRINGS: Tuple[str, ...] = tuple(
    list(DEFAULT_TOC_PATTERNS) + list(_iter_config_list(LEXICON_CFG.get("toc_patterns")))
)
TOC_REGEXES: Tuple[re.Pattern[str], ...] = tuple(_compile_phrase(item) for item in TOC_PATTERN_STRINGS)

BYPASS_TERMS: Tuple[str, ...] = tuple(
    sorted(
        {
            term.lower()
            for term in DEFAULT_BYPASS_CUES
        }
        | {term.lower() for term in _iter_config_list(LEXICON_CFG.get("bypass_cues"))}
    )
)
BYPASS_REGEXES: Tuple[Tuple[re.Pattern[str], str], ...] = tuple(
    (_compile_phrase(term), term) for term in BYPASS_TERMS
)

NEGATION_TERMS: Tuple[str, ...] = tuple(
    sorted(
        {
            term.lower()
            for term in DEFAULT_NEGATION_TERMS
        }
        | {term.lower() for term in _iter_config_list(LEXICON_CFG.get("negation_terms"))}
    )
)

SECTION_ALIAS_MAP: Dict[str, Set[str]] = {}
for key, phrases in DEFAULT_SECTION_ALIASES.items():
    SECTION_ALIAS_MAP[key.upper()] = {phrase for phrase in phrases}
if isinstance(LEXICON_CFG.get("section_aliases"), dict):
    for canonical, phrases in LEXICON_CFG["section_aliases"].items():
        key = str(canonical).upper()
        entries = SECTION_ALIAS_MAP.setdefault(key, set())
        for phrase in _iter_config_list(phrases):
            entries.add(phrase)

SECTION_ALIAS_INDEX: List[Tuple[str, str]] = []
for canonical, phrases in SECTION_ALIAS_MAP.items():
    for phrase in phrases:
        normalized = _normalize_match_text(phrase)
        if normalized:
            SECTION_ALIAS_INDEX.append((normalized, canonical))
SECTION_ALIAS_INDEX.sort(key=lambda item: (-len(item[0]), item[1]))

AUTHORITY_WEIGHTS: Dict[str, float] = (
    CFG.get("weights", {}).get("authority", {}) if isinstance(CFG.get("weights"), dict) else {}
)
AUTHORITY_MAX = max(AUTHORITY_WEIGHTS.values()) if AUTHORITY_WEIGHTS else 1.0


# ---------------------------------------------------------------------------
# Heuristic scoring helpers
# ---------------------------------------------------------------------------


def _cue_score(sent1: str, sent2: str) -> Tuple[float, Dict[str, object]]:
    text = f"{sent1} {sent2}"
    neg_hits = sum(1 for pat in NEG_LEVERAGE_CUES if pat.search(text))
    pos_hits = sum(1 for pat in POS_LEVERAGE_CUES if pat.search(text))
    asset = any(pat.search(text) for pat in ASSET_COVERAGE_PATTERNS)
    liquidity = any(pat.search(text) for pat in LIQUIDITY_PATTERNS)

    weight = 1.0 + 0.18 * neg_hits - 0.08 * pos_hits
    if asset:
        weight += 0.12
    if liquidity:
        weight += 0.07
    return max(0.1, weight), {
        "neg_hits": neg_hits,
        "pos_hits": pos_hits,
        "asset": bool(asset),
        "liquidity": bool(liquidity),
    }


def _temporal_weight(
    form: str,
    prev_form: Optional[str],
    delta_days: Optional[int],
    section: Optional[SectionContext] = None,
) -> float:
    weight = 1.0
    if prev_form and form != prev_form:
        weight += 0.05
        if form == "10-Q" and prev_form == "10-K":
            weight += 0.05
    if delta_days is not None:
        if delta_days <= 120:
            weight += 0.05
        elif delta_days >= 270:
            weight -= 0.05
    if section is not None:
        if section.canonical in {"LIQUIDITY", "CREDIT_AGREEMENT"}:
            weight += 0.05
        elif section.canonical == "RISK_FACTORS":
            weight -= 0.03
        if section.rank is not None:
            weight += max(0.0, 0.04 - 0.01 * min(section.rank, 4))
        if section.level is not None:
            if section.level <= 2:
                weight += 0.05
            elif section.level >= 5:
                weight -= 0.04
    return max(0.1, weight)


def _authority_balance(context: Optional[str], section: Optional[SectionContext] = None) -> float:
    base_weight = 0.5
    keys_to_consider: List[Optional[str]] = []
    if section is not None:
        keys_to_consider.extend([section.weight_key, section.canonical, section.doc_kind])
    for key in keys_to_consider:
        if key and key in AUTHORITY_WEIGHTS:
            base_weight = max(base_weight, AUTHORITY_WEIGHTS[key])
    context_upper = (context or "").upper()
    mapping = [
        ("CREDIT", "CREDIT_AGREEMENT"),
        ("FACILITY", "CREDIT_AGREEMENT"),
        ("INDEBTEDNESS", "CREDIT_AGREEMENT"),
        ("LIQUIDITY", "LIQUIDITY"),
        ("MANAGEMENT", "MDNA"),
        ("RISK", "RISK_FACTORS"),
        ("NOTE", "NOTES_TO_FS"),
        ("INDENTURE", "INDENTURE_NOTES"),
    ]
    for needle, key in mapping:
        if needle in context_upper and key in AUTHORITY_WEIGHTS:
            base_weight = max(base_weight, AUTHORITY_WEIGHTS.get(key, base_weight))
            break
    if section is not None:
        if section.level is not None:
            if section.level <= 2:
                base_weight += 0.1
            elif section.level >= 5:
                base_weight -= 0.05
        if section.rank is not None:
            base_weight += max(0.0, 0.05 - 0.01 * min(section.rank, 5))
        if section.canonical == "RISK_FACTORS":
            base_weight -= 0.05
    max_w = AUTHORITY_MAX if AUTHORITY_MAX > 0 else 1.0
    return float(np.clip(base_weight / max_w, 0.1, 1.0))


def _fallback_conflict(dv: float, cue_meta: Dict[str, object], sent1: str, sent2: str) -> float:
    base = 0.08 + 0.35 * dv
    base += 0.12 * float(cue_meta.get("neg_hits", 0))
    base -= 0.15 * float(cue_meta.get("pos_hits", 0))
    if cue_meta.get("asset"):
        base += 0.08
    if cue_meta.get("liquidity"):
        base += 0.05
    text = f"{sent1} {sent2}".lower()
    if "breach" in text or "default" in text:
        base += 0.10
    if "going concern" in text:
        base += 0.12
    return float(np.clip(base, 0.03, 0.85))


def cale_analyze(api_base: str, rule1: dict, rule2: dict) -> dict:
    url = f"{api_base.rstrip('/')}/v1/law/analyze"
    response = requests.post(url, json={"rule1": rule1, "rule2": rule2}, timeout=60)
    response.raise_for_status()
    return response.json()


KW_GOING = re.compile(r"going concern|substantial doubt", re.I)
KW_BREACH = re.compile(r"\b(default|event of default|breach|waiver|amend(ment)?)\b", re.I)
KW_COVENANT = re.compile(r"\b(covenant|leverage ratio|interest coverage|fixed charge coverage|dscr)\b", re.I)
RATIO_PAT = re.compile(r"(\d+(\.\d+)?)\s*(x|times|%)")


def compute_dv(sent1: str, sent2: str) -> float:
    value = 0.0
    if KW_GOING.search(sent1) or KW_GOING.search(sent2):
        value += CFG["dv"]["going_concern"]
    if KW_BREACH.search(sent1) or KW_BREACH.search(sent2):
        value += CFG["dv"]["breach_keywords"]
    if KW_COVENANT.search(sent1) or KW_COVENANT.search(sent2):
        value += CFG["dv"]["covenant_words"]
    if RATIO_PAT.search(sent1) or RATIO_PAT.search(sent2):
        value += CFG["dv"]["ratio_weakness"]
    if any(pat.search(sent1) or pat.search(sent2) for pat in LIQUIDITY_PATTERNS):
        value += 0.20
    if any(pat.search(sent1) or pat.search(sent2) for pat in ASSET_COVERAGE_PATTERNS):
        value += 0.25
    return float(min(1.0, value))


@dataclass
class FilingScore:
    best_row: Optional[dict]
    evidence_rows: List[dict]
    pair_count: int


def _parse_filings(ticker: str, as_of: dt.date, ua: str) -> Tuple[str, List[dict]]:
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        return "", []
    submissions = load_submissions(cik, ua=ua)
    if not submissions:
        return cik, []
    rows = pick_last_10k_10q_before(submissions, as_of)
    filings: List[dict] = []
    for row in rows:
        try:
            filing_date = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        filings.append(
            {
                "form": row["form"],
                "date": filing_date,
                "accession": row["acc"],
                "primary": row["prim"],
            }
        )
    filings.sort(key=lambda item: item["date"])
    return cik, filings


def _read_html(cik: str, filing: dict, ua: str) -> Optional[str]:
    path = ensure_filing_html(cik, filing["accession"], filing["primary"], ua=ua)
    if not path or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def score_filing(
    ticker: str,
    cik: str,
    filing: dict,
    ua: str,
    api_base: str,
    as_of: dt.date,
    prev_meta: Optional[dict] = None,
    prev_best: Optional[dict] = None,
) -> FilingScore:
    html = _read_html(cik, filing, ua)
    if html is None:
        empty_row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing.get("date")),
            "form": filing.get("form"),
            "CCE": 0.0,
            "cce_raw": 0.0,
            "weight": 0.0,
            "pair_count": 0,
            "cce_level": 0.0,
            "cce_delta": 0.0,
            "no_evidence": True,
            "notes": "Missing HTML filing",
        }
        return FilingScore(best_row=empty_row, evidence_rows=[], pair_count=0)

    pairs = extract_pairs_from_html(html, filing.get("form"))
    print(
        f"[extract] {ticker} {filing.get('form')} {filing.get('date')}: pairs={len(pairs)}"
    )
    if not pairs:
        empty_row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing.get("date")),
            "form": filing.get("form"),
            "CCE": 0.0,
            "cce_raw": 0.0,
            "weight": 0.0,
            "pair_count": 0,
            "cce_level": 0.0,
            "cce_delta": 0.0,
            "no_evidence": True,
            "notes": "No eligible covenants",
        }
        return FilingScore(best_row=empty_row, evidence_rows=[], pair_count=0)

    evidences: List[dict] = []
    best_row: Optional[dict] = None

    delta_days: Optional[int] = None
    prev_form = None
    if prev_meta:
        prev_form = prev_meta.get("form")
        prev_date = prev_meta.get("date")
        if isinstance(prev_date, dt.date):
            delta_days = (filing["date"] - prev_date).days

    for pair in pairs[:60]:
        obligation = pair.obligation
        permission = pair.permission
        section = pair.section
        dv = compute_dv(obligation, permission)
        rule1 = {
            "text": obligation,
            "jurisdiction": "US",
            "statute": "Debt Covenant",
            "section": "N/A",
            "enactment_year": as_of.year,
        }
        rule2 = {
            "text": permission,
            "jurisdiction": "US",
            "statute": "Management Guidance",
            "section": "N/A",
            "enactment_year": as_of.year,
        }
        try:
            metrics = cale_analyze(api_base, rule1, rule2)
        except Exception:
            metrics = {}

        cue_weight, cue_meta = _cue_score(obligation, permission)
        C = float(metrics.get("conflict_intensity", 0.0))
        Ab = float(metrics.get("authority_balance", 0.0))
        S = float(metrics.get("semantic_overlap", 0.0))
        Dt = float(metrics.get("temporal_drift", 0.0))
        if C <= 0.0:
            C = _fallback_conflict(dv, cue_meta, obligation, permission)
        if Ab <= 0.0:
            Ab = _authority_balance(f"{obligation} {permission}", section)
        cce_raw = float(np.clip(C * Ab * dv, 0.0, 1.0))

        temporal_weight = _temporal_weight(filing["form"], prev_form, delta_days, section)
        bypass = 1 if (
            PERM_BYPASS.search(permission)
            or (
                PERM_RELIEF_VERB.search(permission)
                and RELIEF_OBJECT_RE.search(permission)
            )
        ) else 0
        bypass_weight = 0.3 if bypass else 1.0
        combo_weight = float(np.clip(cue_weight * temporal_weight * bypass_weight, 0.1, 3.0))
        jitter_seed = (obligation + "||" + permission).encode("utf-8")
        jitter = (int(hashlib.sha1(jitter_seed).hexdigest()[:8], 16) / 0xFFFFFFFF) * 0.4 - 0.2
        cce_weighted = float(np.clip(cce_raw * combo_weight + jitter, 0.0, 1.0))

        row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing["date"]),
            "form": filing["form"],
            "C": C,
            "Ab": Ab,
            "Dv": dv,
            "B": float(bypass),
            "S": S,
            "Dt": Dt,
            "CCE": cce_weighted,
            "cce_raw": cce_raw,
            "weight": combo_weight,
            "cue_meta": cue_meta,
            "o_sentence": obligation,
            "p_sentence": permission,
            "section": section.canonical if section else "N/A",
            "section_title": section.title if section else None,
            "section_path": section.path if section else None,
            "section_rank": section.rank if section else None,
            "is_threshold": False,
        }

        thresholds: List[Dict[str, object]] = []
        try:
            thresholds = extract_numeric_covenants(obligation)
        except Exception:
            thresholds = []

        best_threshold: Optional[Dict[str, object]] = None
        if thresholds:
            thresholds_sorted = sorted(
                thresholds,
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )
            best_threshold = next(
                (item for item in thresholds_sorted if float(item.get("confidence", 0.0)) >= 0.5),
                None,
            )

        if best_threshold:
            threshold_copy = dict(best_threshold)
            qualifiers = dict(threshold_copy.get("qualifiers", {}) or {})
            if row.get("section") and row["section"] != "N/A":
                qualifiers.setdefault("section", row["section"])
            threshold_copy["qualifiers"] = qualifiers
            row["is_threshold"] = True
            row["threshold"] = threshold_copy
        if pair.metadata:
            row["pair_meta"] = pair.metadata

        evidences.append(row)
        if best_row is None or row["CCE"] > best_row["CCE"]:
            best_row = row

    pair_count = len(evidences)
    if pair_count == 0:
        empty_row = {
            "ticker": ticker,
            "as_of": str(as_of),
            "filing_date": str(filing.get("date")),
            "form": filing.get("form"),
            "CCE": 0.0,
            "cce_raw": 0.0,
            "weight": 0.0,
            "pair_count": 0,
            "cce_level": 0.0,
            "cce_delta": 0.0,
            "no_evidence": True,
            "notes": "No eligible covenants",
        }
        return FilingScore(best_row=empty_row, evidence_rows=[], pair_count=0)

    cce_level = max(row["CCE"] for row in evidences)
    damped_level = damp_investment_grade(ticker, cce_level)
    if damped_level != cce_level and cce_level > 0:
        scale = damped_level / cce_level
        for row in evidences:
            row["CCE"] = float(np.clip(row["CCE"] * scale, 0.0, 1.0))
        cce_level = damped_level
    if best_row is None:
        best_row = max(evidences, key=lambda r: r["CCE"])
    else:
        best_row = max(evidences, key=lambda r: r["CCE"])

    prev_level = None
    if prev_best is not None:
        prev_level = float(prev_best.get("cce_level", prev_best.get("CCE", 0.0)) or 0.0)
    cce_delta = cce_level - prev_level if prev_level is not None else 0.0

    enriched_best = dict(best_row)
    enriched_best.update(
        {
            "pair_count": pair_count,
            "cce_level": float(cce_level),
            "cce_delta": float(cce_delta),
            "prev_cce_level": prev_level,
        }
    )

    return FilingScore(best_row=enriched_best, evidence_rows=evidences, pair_count=pair_count)


def extract_filing_features(
    ticker: str,
    as_of: dt.date,
    ua: str,
    api_base: str,
) -> Tuple[Optional[dict], List[dict], Optional[dict]]:
    """Return (best_row, evidence_rows, previous_best_row)."""

    cik, filings = _parse_filings(ticker, as_of, ua)
    if not cik or not filings:
        return None, [], None

    latest = filings[-1]
    prev_meta = filings[-2] if len(filings) >= 2 else None
    prev_row: Optional[dict] = None

    if prev_meta is not None:
        prev_score = score_filing(ticker, cik, prev_meta, ua, api_base, as_of, None, None)
        prev_row = prev_score.best_row

    score = score_filing(ticker, cik, latest, ua, api_base, as_of, prev_meta, prev_row)
    best_row = score.best_row

    if best_row is None:
        return None, score.evidence_rows, prev_row

    if "pair_count" not in best_row:
        best_row["pair_count"] = score.pair_count

    if "cce_level" not in best_row:
        best_row["cce_level"] = float(best_row.get("CCE", 0.0))

    if "cce_delta" not in best_row:
        prev_cce = float(prev_row.get("cce_level", prev_row.get("CCE", 0.0))) if prev_row else None
        if prev_cce is not None:
            best_row["cce_delta"] = float(best_row.get("cce_level", best_row.get("CCE", 0.0)) - prev_cce)
            best_row["prev_cce_level"] = prev_cce
        else:
            best_row["cce_delta"] = 0.0
            best_row["prev_cce_level"] = None

    best_row["prev_form"] = prev_meta.get("form") if prev_meta else None
    best_row["prev_filing_date"] = str(prev_meta.get("date")) if prev_meta else None

    return best_row, score.evidence_rows, prev_row


def top_evidence_sentences(evidence_rows: Sequence[dict], limit: int = 3) -> List[dict]:
    sorted_rows = sorted(evidence_rows, key=lambda row: row.get("CCE", 0.0), reverse=True)
    out: List[dict] = []
    for row in sorted_rows:
        if len(out) >= limit:
            break
        out.append(
            {
                "ticker": row.get("ticker"),
                "filing": f"{row.get('form')} {row.get('filing_date')}",
                "sentence": row.get("o_sentence"),
                "role": "obligation",
                "cce_raw": row.get("cce_raw", 0.0),
                "CCE": row.get("CCE", 0.0),
                "weight": row.get("weight", 1.0),
            }
        )
        if len(out) >= limit:
            break
        out.append(
            {
                "ticker": row.get("ticker"),
                "filing": f"{row.get('form')} {row.get('filing_date')}",
                "sentence": row.get("p_sentence"),
                "role": "permission",
                "cce_raw": row.get("cce_raw", 0.0),
                "CCE": row.get("CCE", 0.0),
                "weight": row.get("weight", 1.0),
            }
        )
    return out[:limit]


# ---------------------------------------------------------------------------
# Offline covenant pipeline used in tests
# ---------------------------------------------------------------------------


NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
NUMERIC_RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b", re.I)
COVENANT_WORDS = re.compile(
    r"\b(covenant|leverage|coverage|ratio|restricted payments?|liens?|indebtedness|liquidity)\b",
    re.I,
)


def extract_numeric_covenants(sentence: str) -> List[dict]:
    """Return covenant candidates from *sentence*.

    The function favours strings that include both numeric material and
    covenant-style keywords. It returns lightweight dictionaries that the test
    pipeline can enrich further.
    """

    text = sentence.strip()
    if not text:
        return []
    if not COVENANT_WORDS.search(text):
        return []
    numbers = []
    numbers.extend(NUMERIC_RANGE_RE.findall(text))
    numbers.extend(NUMERIC_RE.findall(text))
    if not numbers:
        return []
    return [{"sentence": text, "numbers": numbers}]


def _iter_section_blocks(doc: Doc, section: sectionizer.Section) -> List[int]:
    end = max(section.start_block, section.end_block)
    return list(range(section.start_block, min(len(doc.blocks), end)))


def run_local_pipeline(files: Sequence[str], baseline_pairs: int = 10) -> Dict[str, object]:
    """Execute the offline pipeline for the supplied ``files``."""

    docs = [docio.load_doc(path) for path in files]
    dedup_cache: Set[str] = set()
    dedup_order: Deque[str] = collections.deque()
    evidence: List[dict] = []
    table_records = 0
    total_sections = 0
    anchor_total = 0
    anchor_resolved = 0

    for doc in docs:
        sections = sectionizer.find_sections(doc, CFG)
        total_sections += len(sections)
        for section in sections:
            block_indices = _iter_section_blocks(doc, section)
            range_anchor = authority.link_range(doc, section.start_block, section.end_block - 1)
            default_anchor = section.anchor or range_anchor.get("anchor")
            default_title = range_anchor.get("section_title") or section.title
            for block_idx in block_indices:
                block = doc.blocks[block_idx]
                link_info = authority.link_block(doc, block_idx)
                anchor_total += 1
                if link_info.get("anchor"):
                    anchor_resolved += 1
                block_anchor = link_info.get("anchor") or default_anchor
                block_title = link_info.get("section_title") or default_title
                if block.kind == "table" and block.table:
                    for sentence, cell_meta in tables.flatten(block):
                        table_records += 1
                        for result in extract_numeric_covenants(sentence):
                            if dedup.is_duplicate(result["sentence"], dedup_cache, dedup_order):
                                continue
                            row = {
                                "sentence": result["sentence"],
                                "numbers": result["numbers"],
                                "section_key": section.key,
                                "section_title": block_title,
                                "anchor": block_anchor,
                                "doc_kind": section.doc_kind,
                                "source_path": doc.src_path,
                                "qualifiers": {"section": block_title},
                                "table_cell_meta": cell_meta,
                            }
                            evidence.append(row)
                else:
                    text = block.text.strip()
                    if not text:
                        continue
                    for result in extract_numeric_covenants(text):
                        if dedup.is_duplicate(result["sentence"], dedup_cache, dedup_order):
                            continue
                        row = {
                            "sentence": result["sentence"],
                            "numbers": result["numbers"],
                            "section_key": section.key,
                            "section_title": block_title,
                            "anchor": block_anchor,
                            "doc_kind": section.doc_kind,
                            "source_path": doc.src_path,
                            "qualifiers": {"section": block_title},
                        }
                        evidence.append(row)

    anchor_ratio = (anchor_resolved / anchor_total) if anchor_total else 1.0
    summary = {
        "sectionizer": {"pass": total_sections > 0, "count": total_sections},
        "table_parsing": {"pass": table_records > 0, "records": table_records},
        "authority_links": {"pass": anchor_ratio >= 0.95, "resolved": anchor_resolved, "total": anchor_total},
        "numeric_pairs": len(evidence),
        "baseline_pairs": baseline_pairs,
        "evidence": evidence,
    }
    return summary


def main(argv: Optional[Sequence[str]] = None, files: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """CLI entry-point used by tests to exercise the offline pipeline."""

    if files is None:
        parser = argparse.ArgumentParser(description="Offline finance extractor")
        parser.add_argument("files", nargs="+", help="Local filing paths")
        args = parser.parse_args(argv)
        files = args.files
    summary = run_local_pipeline(files)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
