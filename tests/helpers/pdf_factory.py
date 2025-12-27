from __future__ import annotations

from io import BytesIO
from typing import List

from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle
from reportlab.pdfgen import canvas


_FIXED_DATE = "D:20200101000000Z"

rl_config.invariant = True


def _apply_deterministic_metadata(pdf_canvas: canvas.Canvas) -> None:
    pdf_canvas.setAuthor("")
    pdf_canvas.setCreator("")
    pdf_canvas.setProducer("")
    pdf_canvas.setSubject("")
    pdf_canvas.setTitle("")
    if hasattr(pdf_canvas, "_doc"):
        info = pdf_canvas._doc.info
        info.creationDate = _FIXED_DATE
        info.modDate = _FIXED_DATE


def create_test_pdf_with_text(text: str) -> bytes:
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=A4)
    _apply_deterministic_metadata(pdf_canvas)
    pdf_canvas.setFont("Helvetica", 12)
    text_object = pdf_canvas.beginText(50, 750)
    for line in text.splitlines() or [""]:
        text_object.textLine(line)
    pdf_canvas.drawText(text_object)
    pdf_canvas.showPage()
    pdf_canvas.save()
    return buffer.getvalue()


def create_test_pdf_with_table(rows: List[List[str]]) -> bytes:
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=A4)
    _apply_deterministic_metadata(pdf_canvas)
    table = Table(rows)
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 12),
                ("GRID", (0, 0), (-1, -1), 0.5, (0, 0, 0)),
            ]
        )
    )
    width, height = table.wrapOn(pdf_canvas, 500, 700)
    table.drawOn(pdf_canvas, 50, 750 - height)
    pdf_canvas.showPage()
    pdf_canvas.save()
    return buffer.getvalue()
