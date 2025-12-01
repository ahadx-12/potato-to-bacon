from __future__ import annotations

import io
import re
from typing import Dict, List

try:  # pragma: no cover - optional dependency
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    pdfplumber = None

from potatobacon.law.manifest import LawSource


def extract_text_from_pdf(file: bytes) -> str:
    """Extract plain text from a PDF payload."""
    if pdfplumber is not None:
        with pdfplumber.open(io.BytesIO(file)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()

    # Fallback extraction that pulls text tokens inside parentheses
    decoded = file.decode("latin-1", errors="ignore")
    matches = re.findall(r"\(([^\)]+)\)", decoded)
    return "\n".join(matches)


def split_into_sections(text: str) -> List[Dict[str, str]]:
    """Split the text into coarse sections based on headings/newlines."""

    sections: List[Dict[str, str]] = []
    current: List[str] = []
    section_id = 1
    heading_pattern = re.compile(r"^(Section|SEC\.)?\s*\d+[\.:)]?\s", re.IGNORECASE)

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if heading_pattern.match(stripped) and current:
            sections.append({"id": f"section_{section_id}", "text": " ".join(current).strip()})
            current = []
            section_id += 1
        current.append(stripped)

    if current:
        sections.append({"id": f"section_{section_id}", "text": " ".join(current).strip()})
    return sections


def build_sources_from_pdf(text: str, base_id: str | None = None) -> List[LawSource]:
    sources: List[LawSource] = []
    for idx, section in enumerate(split_into_sections(text), start=1):
        section_id = section["id"]
        full_id = f"{base_id or 'pdf'}_{section_id}_{idx}"
        sources.append(LawSource(id=full_id, text=section["text"]))
    if not sources and text:
        sources.append(LawSource(id=f"{base_id or 'pdf'}_1", text=text))
    return sources

