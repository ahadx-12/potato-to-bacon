"""Mount the tax law dashboard and related static assets."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from potatobacon.law.api import router as api_router

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@router.get("/law/tax", include_in_schema=False, response_class=HTMLResponse)
async def tax_dashboard(request: Request) -> HTMLResponse:
    """Serve the dashboard HTML shell."""

    context = {"request": request}
    return templates.TemplateResponse(request, "tax_dashboard.html", context)


def register_tax_dashboard(app: FastAPI) -> None:
    """Register UI and API routes on the provided FastAPI app."""

    if STATIC_DIR.exists():
        app.mount("/static/law/tax", StaticFiles(directory=str(STATIC_DIR)), name="law-tax-static")
    app.include_router(api_router)
    app.include_router(router)
